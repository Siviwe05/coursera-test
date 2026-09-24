"""
situations.py -- builds the solver's memory: every candidate moment, described as a situation,
together with what every trade plan would actually have returned from that moment.

A situation (all causal: only data up to the moment itself) contains
  * the price story at 4 scales         moves over 5 / 15 / 60 / 240 min, in units of the pair's
                                         own recent hourly volatility (so pairs and regimes compare)
  * where price is                       position in today's range; distance to previous-day
                                         high/low and Asian-range high/low
  * the tick engine                      net effort over 15 and 60 min (5-min candle frame),
                                         scaled by its own recent typical size
  * who is moving it (cross-pair)        strength of the pair's base and quote currency over
                                         15 and 60 min, from all 5 pairs
  * time of day

Plans: buy or sell, with stop / target sized in units of the pair's hourly volatility U:
  (stop, target) in {(0.75U, 1U), (0.75U, 1.5U), (1.25U, 1.5U), (1.25U, 2.5U)}, 4-hour limit.
Outcomes use real bid/ask fills and 1 pip commission, in pips and in R (multiples of the stop).

The holdout (everything from CUTOFF on) is never loaded.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from serotiny2.engine import SymbolSpec, CandleEngine          # noqa: E402
from serotiny2.ticks import load_ticks                           # noqa: E402

PAIRS = ["GBPUSD", "EURUSD", "AUDUSD", "USDJPY", "EURGBP"]
CURRENCIES = ["USD", "EUR", "GBP", "AUD", "JPY"]
SAMPLE_EVERY_MIN = 5
ACTIVE_HOURS = range(10, 20)          # server time (UTC+3): London + New York
MAX_HOLD_S = 4 * 3600
COMMISSION_PIPS = 1.0
MAX_SPREAD_PIPS = 1.5
MIN_STOP_PIPS = 4.0
PLANS = [(0.75, 1.0), (0.75, 1.5), (1.25, 1.5), (1.25, 2.5)]   # (stop, target) in units of U
FEATURES = ["r5", "r15", "r60", "r240", "range_pos", "d_pdh", "d_pdl", "d_ash", "d_asl",
            "eff15", "eff60", "base15", "base60", "quote15", "quote60", "hour"]
FEATURE_GROUPS = {
    "price": ["r5", "r15", "r60", "r240", "range_pos", "d_pdh", "d_pdl", "d_ash", "d_asl", "hour"],
    "engine": ["eff15", "eff60"],
    "cross": ["base15", "base60", "quote15", "quote60"],
}


def cutoff_msc(tick_root):
    """Start of the holdout: the last 25% of GBPUSD's span (same cut as all earlier research)."""
    t, _, _ = load_ticks(os.path.join(tick_root, "GBPUSD"))
    return int(t[0] + (t[-1] - t[0]) * 0.75)


class PairTicks:
    def __init__(self, tick_root, pair, cutoff):
        self.pair = pair
        self.spec = SymbolSpec.from_name(pair)
        t, b, a = load_ticks(os.path.join(tick_root, pair))
        k = t < cutoff
        self.t, self.bid, self.ask = t[k], b[k], a[k]
        self.mid = np.round((self.bid + self.ask) / 2, 5)
        cache = os.path.join(tick_root, "_cache", f"{pair}_eff300_{len(self.t)}_{self.t[-1]}.npy")
        if os.path.exists(cache):
            net = np.load(cache)
        else:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            eng = CandleEngine(self.spec, 300)
            net = np.zeros(len(self.t))
            for i in range(len(self.t)):
                _, dx, dy = eng.step(int(self.t[i] // 1000), self.mid[i])
                net[i] = dx - dy
            np.save(cache, net)
        self.cum_net = np.concatenate([[0], np.cumsum(net)])

    def idx(self, times):
        return np.searchsorted(self.t, times, side="right") - 1

    def price(self, times):
        return self.mid[np.maximum(self.idx(times), 0)]

    def effort(self, times, w_s):
        i, j = self.idx(times), np.maximum(self.idx(times - w_s * 1000), 0)
        return self.cum_net[i + 1] - self.cum_net[j + 1]


def plan_outcomes(pt: PairTicks, i, U):
    """Net pips and R for every (direction, plan) from tick i."""
    pip = pt.spec.pip
    end = np.searchsorted(pt.t, pt.t[i] + MAX_HOLD_S * 1000, side="right")
    out = {}
    for direction in (1, -1):
        if direction > 0:
            entry, path = pt.ask[i], pt.bid[i + 1:end]
            gain = (path - entry) / pip
        else:
            entry, path = pt.bid[i], pt.ask[i + 1:end]
            gain = (entry - path) / pip
        for p, (sm, tm) in enumerate(PLANS):
            sl = max(sm * U, MIN_STOP_PIPS)
            tp = sl * tm / sm
            tp_hit = np.flatnonzero(gain >= tp - 1e-6)
            sl_hit = np.flatnonzero(gain <= -sl + 1e-6)
            if len(tp_hit) and (not len(sl_hit) or tp_hit[0] < sl_hit[0]):
                g, j = tp, tp_hit[0]
            elif len(sl_hit):
                g, j = -sl, sl_hit[0]
            else:
                g, j = gain[-1], len(gain) - 1
            net = g - COMMISSION_PIPS
            key = f"{'buy' if direction > 0 else 'sell'}{p}"
            out[f"{key}_pips"] = net
            out[f"{key}_R"] = net / sl
            out[f"{key}_exit"] = int(pt.t[i + 1 + j])
    return out


def build(tick_root, out_path=None):
    cut = cutoff_msc(tick_root)
    ticks = {p: PairTicks(tick_root, p, cut) for p in PAIRS}
    start = max(pt.t[0] for pt in ticks.values()) + 2 * 86_400_000   # need a day of history first
    minute = np.arange(start - start % 60_000, cut, 60_000)

    # 1-minute mid prices and hourly volatility U (pips), causal
    px = {p: ticks[p].price(minute) for p in PAIRS}
    U = {}
    for p in PAIRS:
        r5 = pd.Series(np.r_[np.full(5, np.nan), px[p][5:] - px[p][:-5]] / ticks[p].spec.pip)
        U[p] = (r5.iloc[::1].rolling(24 * 60, min_periods=6 * 60).std() * np.sqrt(12)).to_numpy()

    def move(p, w_min):
        m = np.full(len(minute), np.nan)
        m[w_min:] = (px[p][w_min:] - px[p][:-w_min]) / ticks[p].spec.pip
        return m

    norm_moves = {(p, w): move(p, w) / U[p] for p in PAIRS for w in (5, 15, 60, 240)}
    strength = {}
    for w in (15, 60):
        for c in CURRENCIES:
            parts = [norm_moves[(p, w)] * (1 if p.startswith(c) else -1) for p in PAIRS if c in p]
            strength[(c, w)] = np.nanmean(parts, axis=0)

    ts = pd.to_datetime(minute, unit="ms")
    sample_mask = (ts.minute % SAMPLE_EVERY_MIN == 0) & np.isin(ts.hour, ACTIVE_HOURS) & (ts.dayofweek < 5) \
        & (minute + MAX_HOLD_S * 1000 < cut)
    rows = []
    for p in PAIRS:
        pt = ticks[p]
        pip = pt.spec.pip
        base, quote = p[:3], p[3:]
        # daily levels (server day)
        day = ts.normalize()
        dfp = pd.DataFrame({"day": day, "px": px[p], "hour": ts.hour})
        day_hi = dfp.groupby("day").px.cummax().to_numpy()
        day_lo = dfp.groupby("day").px.cummin().to_numpy()
        full_hi = dfp.groupby("day").px.max()
        full_lo = dfp.groupby("day").px.min()
        prev_day = {d: full_hi.index[k - 1] if k > 0 else None for k, d in enumerate(full_hi.index)}
        asia = dfp[(dfp.hour >= 3) & (dfp.hour < 10)].groupby("day").px.agg(["max", "min"])
        eff15 = pt.effort(minute, 900)
        eff60 = pt.effort(minute, 3600)
        # scale effort by its own trailing typical size (5 trading days of minutes)
        e15s = pd.Series(np.abs(eff15)).rolling(5 * 1440, min_periods=1440).median().to_numpy()
        e60s = pd.Series(np.abs(eff60)).rolling(5 * 1440, min_periods=1440).median().to_numpy()
        for m_i in np.flatnonzero(sample_mask):
            u = U[p][m_i]
            d = day[m_i]
            pd_ = prev_day.get(d)
            if not np.isfinite(u) or u <= 0 or pd_ is None or d not in asia.index or not np.isfinite(e60s[m_i]):
                continue
            i = pt.idx(minute[m_i])
            if i < 0 or (pt.ask[i] - pt.bid[i]) / pip > MAX_SPREAD_PIPS or minute[m_i] - pt.t[i] > 120_000:
                continue   # stale quote (market quiet/closed) or spread too wide
            price = pt.mid[i]
            hi, lo = day_hi[m_i], day_lo[m_i]
            row = {"pair": p, "time": int(minute[m_i]), "tick_i": int(i), "U": u,
                   "r5": norm_moves[(p, 5)][m_i], "r15": norm_moves[(p, 15)][m_i],
                   "r60": norm_moves[(p, 60)][m_i], "r240": norm_moves[(p, 240)][m_i],
                   "range_pos": (price - lo) / (hi - lo) if hi > lo else 0.5,
                   "d_pdh": np.clip((full_hi[pd_] - price) / pip / u, -5, 5),
                   "d_pdl": np.clip((full_lo[pd_] - price) / pip / u, -5, 5),
                   "d_ash": np.clip((asia.loc[d, "max"] - price) / pip / u, -5, 5),
                   "d_asl": np.clip((asia.loc[d, "min"] - price) / pip / u, -5, 5),
                   "eff15": eff15[m_i] / max(e15s[m_i], 1), "eff60": eff60[m_i] / max(e60s[m_i], 1),
                   "base15": strength[(base, 15)][m_i], "base60": strength[(base, 60)][m_i],
                   "quote15": strength[(quote, 15)][m_i], "quote60": strength[(quote, 60)][m_i],
                   "hour": ts.hour[m_i] + ts.minute[m_i] / 60}
            row.update(plan_outcomes(pt, i, u))
            rows.append(row)
        print(f"  {p}: {sum(r['pair'] == p for r in rows)} situations", flush=True)
    df = pd.DataFrame(rows).dropna(subset=FEATURES).reset_index(drop=True)
    df["day"] = pd.to_datetime(df.time, unit="ms").dt.normalize()
    if out_path:
        df.to_pickle(out_path)
    return df, cut


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "ticks"
    df, cut = build(root, os.path.join(root, "_cache", "situations.pkl"))
    print(f"{len(df):,} situations, {df.day.nunique()} days, "
          f"{df.day.min().date()} -> {df.day.max().date()} (holdout from {pd.Timestamp(cut, unit='ms')})")
