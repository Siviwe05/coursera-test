"""Effort vs result at critical zones -- implements PREREG_effort_vs_result.md exactly.

Effort = quote activity (quote updates per minute / causal same-minute-of-day baseline),
measured separately from price. Research window only; holdout never loaded.
Run from the repo root:  python -m serotiny2.research.effort_vs_result ticks
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from serotiny2.engine import SymbolSpec                    # noqa: E402
from serotiny2.ticks import load_ticks                     # noqa: E402
from serotiny2.outcomes import first_touch                 # noqa: E402
from serotiny2.solver.situations import cutoff_msc         # noqa: E402

PAIRS = ["GBPUSD", "EURUSD", "AUDUSD", "USDJPY", "EURGBP"]
BASELINE_DAYS, SHUFFLE_SHIFT = 20, 7
W_MIN, LAST_MIN = 15, 5
C1_PIPS, C2_RATIO, C3_PIPS = 3.0, 1.5, 1.0
MAX_SPREAD, HOLD_S, COMM = 1.5, 4 * 3600, 1.0


def minute_series(t, mid, pip):
    m0 = t[0] // 60_000
    m = t // 60_000 - m0
    n = m[-1] + 1
    counts = np.bincount(m, minlength=n).astype(float)
    step = np.r_[0.0, np.abs(np.diff(mid))] / pip
    path = np.bincount(m, weights=step, minlength=n)
    last_i = np.full(n, -1)
    last_i[m] = np.arange(len(t))                # last tick index in each minute
    last_i = np.maximum.accumulate(last_i)
    end_mid = mid[np.maximum(last_i, 0)]
    move = np.r_[0.0, np.diff(end_mid)] / pip    # minute's mid change
    return m0, counts, path, move


def activity_ratios(m0, counts):
    """counts / median of the same minute-of-day over the previous 20 weekdays; also the
    'shuffled' version taken from 7 weekdays earlier. NaN where no baseline yet."""
    n = len(counts)
    minutes = m0 + np.arange(n)
    day = minutes // 1440
    days = np.unique(day)
    wd = [d for d in days if pd.Timestamp(d * 86_400_000, unit="ms").dayofweek < 5]
    mat = np.zeros((len(wd), 1440))
    for k, d in enumerate(wd):
        sel = day == d
        mat[k, minutes[sel] % 1440] = counts[sel]
    act = np.full(n, np.nan)
    shuf = np.full(n, np.nan)
    for k, d in enumerate(wd):
        if k < BASELINE_DAYS:
            continue
        base = np.maximum(np.median(mat[k - BASELINE_DAYS:k], axis=0), 1.0)
        sel = day == d
        mod = minutes[sel] % 1440
        act[sel] = mat[k, mod] / base[mod]
        shuf[sel] = mat[k - SHUFFLE_SHIFT, mod] / base[mod]
    return act, shuf


def side_stats(move, effort, direction):
    r = move * direction
    mine = r > 0
    res = r[mine].sum()
    eff = effort[mine].sum()
    return res, eff, (res / eff if eff > 0 else 0.0)


def checklist(move_w, effort_w, winner_dir):
    _, _, e_win = side_stats(move_w, effort_w, winner_dir)
    _, _, e_los = side_stats(move_w, effort_w, -winner_dir)
    c2 = e_win > 0 and (e_los == 0 or e_win / e_los >= C2_RATIO)
    last_res, _, _ = side_stats(move_w[-LAST_MIN:], effort_w[-LAST_MIN:], -winner_dir)
    c3 = last_res <= C3_PIPS
    return c2, c3


def events_for_pair(root, pair, cut):
    spec = SymbolSpec.from_name(pair)
    pip = spec.pip
    t, b, a = load_ticks(os.path.join(root, pair))
    k = t < cut
    t, b, a = t[k], b[k], a[k]
    mid = np.round((b + a) / 2, 5)
    m0, counts, path, move = minute_series(t, mid, pip)
    act, shuf = activity_ratios(m0, counts)

    ts = pd.to_datetime(t, unit="ms")
    dates, hours = ts.normalize(), ts.hour
    day_list = dates.unique()
    rows = []
    for j, d in enumerate(day_list):
        if d.dayofweek >= 5 or j == 0:
            continue
        in_day = np.flatnonzero(dates == d)
        prev = np.flatnonzero(dates == day_list[j - 1])
        asia = in_day[(hours[in_day] >= 3) & (hours[in_day] < 10)]
        active = in_day[(hours[in_day] >= 10) & (hours[in_day] < 20)]
        if len(active) == 0 or len(asia) == 0 or len(prev) == 0:
            continue
        levels = {"PDH": mid[prev].max(), "PDL": mid[prev].min(), "ASH": mid[asia].max(), "ASL": mid[asia].min()}
        start = mid[active[0]]
        for name, lvl in levels.items():
            direction = 1 if name.endswith("H") else -1          # attacker's direction
            if (start - lvl) * direction >= 0:
                continue
            hit = np.flatnonzero((mid[active] - lvl) * direction > 0)
            if len(hit) == 0:
                continue
            i_touch = int(active[hit[0]])
            mt = t[i_touch] // 60_000 - m0
            w = np.arange(mt + 1, mt + 1 + W_MIN)
            T = (m0 + w[-1] + 1) * 60_000
            if T + HOLD_S * 1000 >= cut or w[-1] >= len(move) or np.isnan(act[w]).any():
                continue
            i_T = int(np.searchsorted(t, T, side="right") - 1)
            if (a[i_T] - b[i_T]) / pip > MAX_SPREAD:
                continue
            net = (mid[i_T] - mid[i_touch]) / pip * direction
            if abs(net) < C1_PIPS:
                winner = 0
            else:
                winner = direction if net > 0 else -direction
            row = {"pair": pair, "zone": name, "time": int(T), "winner": winner,
                   "kind": "breakout" if winner == direction else ("rejection" if winner else "none")}
            if winner:
                mw = move[w]
                for tag, eff in (("act", act[w]), ("path", path[w]), ("shuf", shuf[w])):
                    c2, c3 = checklist(mw, eff, winner)
                    row[f"{tag}_ok"] = bool(c2 and c3)
                # exits
                seg = mid[i_touch:i_T + 1]
                if winner > 0:
                    stop = (a[i_T] - (seg.min() - pip)) / pip
                else:
                    stop = ((seg.max() + pip) - b[i_T]) / pip
                stop = float(np.clip(stop, 5, 25))
                s = first_touch(t, b, a, i_T, winner, 1.5 * stop, stop, HOLD_S, pip, COMM)
                f = first_touch(t, b, a, i_T, winner, 10, 10, HOLD_S, pip, COMM)
                row.update(S_R=s["pnl_pips"] / stop, S_pips=s["pnl_pips"], F_R=f["pnl_pips"] / 10,
                           F_pips=f["pnl_pips"], stop=stop)
            rows.append(row)
    print(f"  {pair}: {len(rows)} zone events", flush=True)
    return rows


def summarize(ev, mask_name, mask, half_mid):
    out = []
    for plan in ("S", "F"):
        for part, sel in (("A", ev.time < half_mid), ("B", ev.time >= half_mid), ("all", ev.time > 0)):
            x = ev.loc[mask & sel, f"{plan}_R"]
            if len(x) == 0:
                out.append({"strategy": mask_name, "plan": plan, "half": part, "trades": 0})
                continue
            out.append({"strategy": mask_name, "plan": plan, "half": part, "trades": len(x),
                        "win%": round(100 * (x > 0).mean(), 1), "avg_R": round(x.mean(), 3),
                        "total_R": round(x.sum(), 1),
                        "t": round(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))), 2) if len(x) > 2 else np.nan})
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "ticks"
    cut = cutoff_msc(root)
    rows = []
    for p in PAIRS:
        rows += events_for_pair(root, p, cut)
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(root, "_cache", "effort_vs_result_events.csv"), index=False)
    half_mid = ev.time.min() + (ev.time.max() - ev.time.min()) / 2
    print(f"\n{len(ev)} zone events; {int((ev.winner != 0).sum())} with a clear winner by result (C1); "
          f"half B starts {pd.Timestamp(half_mid, unit='ms').date()}")
    w = ev.winner != 0
    res = []
    res += summarize(ev, "CHECKLIST (quote activity)", w & ev.act_ok.fillna(False).astype(bool), half_mid)
    res += summarize(ev, "price only (C1)", w, half_mid)
    res += summarize(ev, "checklist w/ price-derived effort", w & ev.path_ok.fillna(False).astype(bool), half_mid)
    res += summarize(ev, "checklist w/ SHUFFLED activity", w & ev.shuf_ok.fillna(False).astype(bool), half_mid)
    pd.set_option("display.width", 200)
    df = pd.DataFrame(res)
    print("\nPlan S (pre-chosen):")
    print(df[df.plan == "S"].drop(columns="plan").to_string(index=False))
    print("\nPlan F:")
    print(df[df.plan == "F"].drop(columns="plan").to_string(index=False))

    chk = ev[w & ev.act_ok.fillna(False).astype(bool)]
    if len(chk):
        print("\nCHECKLIST trades by pair and kind (Plan S):")
        print(chk.groupby(["pair"]).S_R.agg(["size", "mean"]).round(3).to_string())
        print(chk.groupby(["kind"]).S_R.agg(["size", "mean"]).round(3).to_string())

    # verdict against the pre-registered criteria
    def avg(mask, sel):
        x = ev.loc[mask & sel, "S_R"]
        return x.mean() if len(x) else np.nan
    A, B = ev.time < half_mid, ev.time >= half_mid
    chkm = w & ev.act_ok.fillna(False).astype(bool)
    crit = {
        "avg R > 0 in half A (Plan S)": avg(chkm, A) > 0,
        "avg R > 0 in half B (Plan S)": avg(chkm, B) > 0,
        "beats price-only in A": avg(chkm, A) > avg(w, A),
        "beats price-only in B": avg(chkm, B) > avg(w, B),
        "beats shuffled activity (all)": avg(chkm, ev.time > 0) > avg(w & ev.shuf_ok.fillna(False).astype(bool), ev.time > 0),
        ">= 40 trades": chkm.sum() >= 40,
    }
    print("\nPRE-REGISTERED PASS CRITERIA:")
    for k, v in crit.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("  => " + ("PASSES -- holdout may be opened once" if all(crit.values()) else "DOES NOT PASS -- holdout stays sealed"))


if __name__ == "__main__":
    main()
