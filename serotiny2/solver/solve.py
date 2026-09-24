"""
solve.py -- the situation solver, tested walk-forward.

For every candidate moment of day D, using ONLY situations from days before D:
  1. find the K most similar past situations (all pairs pooled; features are volatility-scaled)
  2. look at what each of the 8 trade plans (buy/sell x 4 stop/target shapes) returned in them
  3. take the plan with the best cautious score  mean_R - Z * standard_error,
     but only if that score is above zero and the mean clears MIN_EDGE_R; otherwise pass.
Then trades are executed one position at a time per pair, with real fills and commission
already inside the plan outcomes.

Controls (to know what luck looks like):
  * SHUFFLED MEMORY: the same solver, but the past outcomes are shuffled between situations,
    so the situation carries no information. Anything the real solver does is only meaningful
    if it clearly beats this.
  * feature ablations: without the engine, without cross-pair, price only.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from serotiny2.solver.situations import FEATURES, FEATURE_GROUPS, PLANS   # noqa: E402

PLAN_KEYS = [f"{d}{p}" for d in ("buy", "sell") for p in range(len(PLANS))]
WARMUP_DAYS = 15


def solve(df, features, K=100, Z=1.0, MIN_EDGE_R=0.10, shuffle=False, seed=0):
    days = np.sort(df.day.unique())
    R = df[[f"{k}_R" for k in PLAN_KEYS]].to_numpy()
    if shuffle:
        R = R[np.random.default_rng(seed).permutation(len(R))]
    X = df[features].to_numpy(float)
    decisions = []
    for D in days[WARMUP_DAYS:]:
        lib = np.flatnonzero(df.day.to_numpy() < D)
        q = np.flatnonzero(df.day.to_numpy() == D)
        mu, sd = X[lib].mean(0), X[lib].std(0) + 1e-9
        L, Q = (X[lib] - mu) / sd, (X[q] - mu) / sd
        dist = (Q ** 2).sum(1)[:, None] + (L ** 2).sum(1)[None, :] - 2 * Q @ L.T
        nn = np.argpartition(dist, K, axis=1)[:, :K]
        nR = R[lib][nn]                                   # (queries, K, plans)
        mean = nR.mean(1)
        se = nR.std(1) / np.sqrt(K)
        score = mean - Z * se
        best = score.argmax(1)
        for n, qi in enumerate(q):
            b = best[n]
            if score[n, b] > 0 and mean[n, b] >= MIN_EDGE_R:
                decisions.append((qi, b, mean[n, b], score[n, b]))
    return decisions


def execute(df, decisions):
    """One position at a time per pair, in time order."""
    dec = pd.DataFrame(decisions, columns=["row", "plan", "exp_R", "score"])
    if dec.empty:
        return dec
    dec = dec.join(df[["pair", "time", "day"]], on="row").sort_values("time")
    busy, keep = {}, []
    for r in dec.itertuples():
        if r.time < busy.get(r.pair, -1):
            continue
        k = PLAN_KEYS[r.plan]
        keep.append({"pair": r.pair, "time": r.time, "day": r.day, "plan": k, "exp_R": r.exp_R,
                     "R": df.at[r.row, f"{k}_R"], "pips": df.at[r.row, f"{k}_pips"]})
        busy[r.pair] = df.at[r.row, f"{k}_exit"]
    return pd.DataFrame(keep)


def summarize(name, trades, days_mid):
    out = []
    for part, sel in (("A", trades.day < days_mid), ("B", trades.day >= days_mid)) if len(trades) else ():
        t = trades[sel]
        if len(t) == 0:
            continue
        out.append({"variant": name, "half": part, "trades": len(t),
                    "per_day": round(len(t) / t.day.nunique(), 1),
                    "win%": round(100 * (t.R > 0).mean(), 1), "avg_R": round(t.R.mean(), 3),
                    "total_R": round(t.R.sum(), 1), "avg_pips": round(t.pips.mean(), 2),
                    "t_stat": round(t.R.mean() / (t.R.std() / np.sqrt(len(t))), 2)})
    if not out:
        out.append({"variant": name, "half": "-", "trades": 0})
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "ticks"
    df = pd.read_pickle(os.path.join(root, "_cache", "situations.pkl"))
    days = np.sort(df.day.unique())
    test_days = days[WARMUP_DAYS:]
    mid = test_days[len(test_days) // 2]
    print(f"{len(df):,} situations over {len(days)} days; walk-forward on {len(test_days)} days "
          f"({pd.Timestamp(test_days[0]).date()} -> {pd.Timestamp(test_days[-1]).date()}), "
          f"half B starts {pd.Timestamp(mid).date()}\n")

    no = lambda *g: [f for f in FEATURES if not any(f in FEATURE_GROUPS[x] for x in g)]
    variants = [
        ("ALL info, K=100", dict(features=FEATURES, K=100)),
        ("ALL info, K=300", dict(features=FEATURES, K=300)),
        ("no engine, K=100", dict(features=no("engine"), K=100)),
        ("no cross-pair, K=100", dict(features=no("cross"), K=100)),
        ("price only, K=100", dict(features=FEATURE_GROUPS["price"], K=100)),
        ("CONTROL shuffled memory, K=100", dict(features=FEATURES, K=100, shuffle=True)),
        ("CONTROL shuffled memory #2, K=100", dict(features=FEATURES, K=100, shuffle=True, seed=1)),
    ]
    rows, all_trades = [], {}
    for name, kw in variants:
        tr = execute(df, solve(df, **kw))
        all_trades[name] = tr
        rows += summarize(name, tr, mid)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))

    main_tr = all_trades["ALL info, K=100"]
    if len(main_tr):
        print("\nALL info, K=100 -- by pair:")
        print(main_tr.groupby("pair").agg(trades=("R", "size"), win=("R", lambda r: round(100 * (r > 0).mean(), 1)),
                                          avg_R=("R", "mean"), total_R=("R", "sum"),
                                          avg_pips=("pips", "mean")).round(3).to_string())
        print("\nexpected vs realised R (is the solver's confidence meaningful?):")
        main_tr["exp_bucket"] = pd.qcut(main_tr.exp_R, 4, duplicates="drop")
        print(main_tr.groupby("exp_bucket", observed=True).R.agg(["size", "mean"]).round(3).to_string())
        main_tr.to_csv(os.path.join(root, "_cache", "solver_trades.csv"), index=False)


if __name__ == "__main__":
    main()
