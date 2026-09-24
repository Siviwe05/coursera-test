"""Does the story carry ANY directional information? Symmetric first-touch on mid, no costs.
Train on the first 55% of time, test on 55-75%. The last 25% (holdout) is never touched."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from serotiny2.engine import SymbolSpec
from serotiny2.pipeline import PipelineConfig
from serotiny2.ticks import load_ticks
from serotiny2.tools.research_story import run_reads

spec = SymbolSpec.from_name("GBPUSD")
t, b, a = load_ticks((sys.argv[1] if len(sys.argv) > 1 else 'ticks/GBPUSD'))
mid = (a + b) / 2
t0, t1 = t[0], t[-1]
cut_train = t0 + (t1 - t0) * 0.55
cut_res = t0 + (t1 - t0) * 0.75


def sym_label(i, n_pips, max_s):
    end = np.searchsorted(t, t[i] + max_s * 1000, side='right')
    p = mid[i + 1:end] - mid[i]
    up = np.flatnonzero(p >= n_pips * spec.pip - 1e-9)
    dn = np.flatnonzero(p <= -n_pips * spec.pip + 1e-9)
    if len(up) == 0 and len(dn) == 0:
        return np.nan
    if len(dn) == 0 or (len(up) and up[0] < dn[0]):
        return 1.0
    return 0.0


for rev in (2.0, 3.0, 5.0):
    df = run_reads(t, b, a, spec, PipelineConfig(reversal_pips=rev))
    df = df[(df.time_msc < cut_res) & (df.f_spread_pips <= 1.5)].copy()
    df['hour'] = pd.to_datetime(df.time_msc, unit='ms').dt.hour
    for n_pips in (5.0,):
        df['y'] = [sym_label(int(i), n_pips, 1800) for i in df.i]
        d = df.dropna(subset=['y'])
        feats = [c for c in d.columns if c.startswith('f_') and c not in ('f_bid', 'f_ask')] + \
                ['edge', 'strength', 'new_leg_direction', 'invalidation_pips', 'hour']
        X = d[feats].apply(pd.to_numeric, errors='coerce').astype(float)
        tr, te = d.time_msc < cut_train, d.time_msc >= cut_train
        base = ['new_leg_direction', 'hour', 'f_last_leg_size_pips']
        out = []
        for name, cols in (('trivial (leg dir, hour, size)', base), ('full story', feats)):
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                               min_samples_leaf=50, random_state=0)
            m.fit(X.loc[tr, cols], d.y[tr])
            p = m.predict_proba(X.loc[te, cols])[:, 1]
            yt = d.y[te].to_numpy()
            auc = roc_auc_score(yt, p)
            conf = np.abs(p - 0.5)
            top = conf >= np.quantile(conf, 0.9)
            acc_top = ((p[top] > 0.5) == (yt[top] == 1)).mean()
            out.append(f"{name}: AUC {auc:.3f}, top-10% confident acc {acc_top:.1%} (n={top.sum()})")
        # the hand rules, and simple momentum
        te_d = d[te]
        v = te_d[te_d.verdict != 'UNCLEAR']
        rule_acc = ((v.verdict == 'BUY') == (v.y == 1)).mean()
        mom = ((te_d.new_leg_direction == 1) == (te_d.y == 1)).mean()
        print(f"\nreversal {rev} pips | ±{n_pips} pips / 30min | reads/day ~{len(df) / 22:.0f} | "
              f"train {tr.sum()} test {te.sum()} | up-rate {d.y.mean():.1%}")
        print(f"  hand rules verdict accuracy {rule_acc:.1%} (n={len(v)}) | follow-new-leg accuracy {mom:.1%}")
        for o in out:
            print("  " + o)
