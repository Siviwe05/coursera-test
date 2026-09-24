"""
research_story.py -- do the story reads actually predict the next move?

    python -m serotiny2.tools.research_story --ticks ticks/EURUSD --symbol EURUSD
    python -m serotiny2.tools.research_story --synthetic      # sanity check on a random walk

Replays ticks through the SAME pipeline the live bot will use, labels
every read with what really happened next (target before stop, within
the time limit, real bid/ask fills, commission included), and prints how
each kind of read performed against the break-even win rate.

HOLDOUT: by default the most recent 25% of the data is NOT reported. We
research on the older data only; the holdout is opened once, at the end,
to confirm (use --include-holdout for that). Looking at it early and
tuning to it would make it worthless.

Output: a summary table, plus a CSV of every read (all features + outcome)
for deeper analysis.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from serotiny2.engine import SymbolSpec                      # noqa: E402
from serotiny2.pipeline import StoryPipeline, PipelineConfig  # noqa: E402
from serotiny2.ticks import load_ticks, synthetic_ticks       # noqa: E402
from serotiny2.outcomes import first_touch, breakeven_winrate  # noqa: E402


def run_reads(times, bid, ask, spec, cfg):
    pipe = StoryPipeline(spec, cfg)
    rows = []
    for i in range(len(times)):
        r = pipe.on_tick(int(times[i]), float(bid[i]), float(ask[i]))
        if r is None:
            continue
        row = {"i": i, "time_msc": r.time_msc, "price": r.price, "verdict": r.verdict,
               "edge": r.edge, "strength": r.strength, "new_leg_direction": r.new_leg_direction,
               "invalidation_pips": r.invalidation_pips,
               "observations": " | ".join(o.key for o in r.observations),
               "narrative": r.narrative}
        row.update({f"f_{k}": v for k, v in r.features.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def label(df, times, bid, ask, spec, direction_col, tp, sl, max_s, comm, prefix):
    out = []
    for i, d in zip(df["i"].to_numpy(), df[direction_col].to_numpy()):
        if d == 0:
            out.append({"outcome": None, "pnl_pips": np.nan, "exit_msc": None, "hold_s": np.nan})
        else:
            out.append(first_touch(times, bid, ask, int(i), int(d), tp, sl, max_s, spec.pip, comm))
    lab = pd.DataFrame(out).add_prefix(prefix)
    return pd.concat([df.reset_index(drop=True), lab], axis=1)


def one_at_a_time(df, prefix):
    """Keep only trades a single-position bot could actually take."""
    keep, busy_until = [], -1
    for idx, row in df.iterrows():
        if row["time_msc"] < busy_until:
            continue
        keep.append(idx)
        busy_until = row[f"{prefix}exit_msc"]
    return df.loc[keep]


def stats(label_, sub, prefix, be):
    sub = sub[sub[f"{prefix}outcome"].isin(["TP", "SL", "TIME"])]
    n = len(sub)
    if n == 0:
        return {"group": label_, "trades": 0}
    wins = (sub[f"{prefix}pnl_pips"] > 0).mean()
    return {"group": label_, "trades": n, "win_rate": f"{wins:.1%}", "breakeven": f"{be:.1%}",
            "avg_pips": round(sub[f"{prefix}pnl_pips"].mean(), 2),
            "total_pips": round(sub[f"{prefix}pnl_pips"].sum(), 1),
            "tp/sl/time": "/".join(str((sub[f"{prefix}outcome"] == k).sum()) for k in ("TP", "SL", "TIME"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", help="tick file or directory (from export_ticks.py)")
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--synthetic", action="store_true", help="use a random-walk tick stream instead")
    ap.add_argument("--reversal-pips", type=float, default=2.0)
    ap.add_argument("--tp", type=float, default=6.0)
    ap.add_argument("--sl", type=float, default=5.0)
    ap.add_argument("--max-minutes", type=float, default=15.0)
    ap.add_argument("--commission-pips", type=float, default=1.0)
    ap.add_argument("--max-spread-pips", type=float, default=1.5)
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--include-holdout", action="store_true")
    ap.add_argument("--out", default="story_reads.csv")
    args = ap.parse_args()

    spec = SymbolSpec.from_name(args.symbol)
    if args.synthetic:
        times, bid, ask = synthetic_ticks(n=400_000, seed=7)
    else:
        if not args.ticks:
            raise SystemExit("--ticks is required (or use --synthetic)")
        times, bid, ask = load_ticks(args.ticks)
    print(f"{len(times):,} ticks loaded")

    cutoff = times[0] + (times[-1] - times[0]) * (1 - args.holdout_frac)
    cfg = PipelineConfig(reversal_pips=args.reversal_pips)
    df = run_reads(times, bid, ask, spec, cfg)
    if df.empty:
        raise SystemExit("no reads produced -- not enough data?")
    print(f"{len(df):,} story reads ({(df.verdict != 'UNCLEAR').sum():,} with a verdict)")

    df["story_dir"] = df["verdict"].map({"BUY": 1, "SELL": -1}).fillna(0).astype(int)
    df = label(df, times, bid, ask, spec, "story_dir", args.tp, args.sl, args.max_minutes * 60,
               args.commission_pips, "s_")
    df = label(df, times, bid, ask, spec, "new_leg_direction", args.tp, args.sl, args.max_minutes * 60,
               args.commission_pips, "b_")
    df.to_csv(args.out, index=False)

    research = df[df["time_msc"] < cutoff] if not args.include_holdout else df
    research = research[research["f_spread_pips"] <= args.max_spread_pips]
    be = breakeven_winrate(args.tp, args.sl, args.commission_pips)
    mid = research["time_msc"].min() + (research["time_msc"].max() - research["time_msc"].min()) / 2

    story = research[research["story_dir"] != 0]
    rows = [
        stats("BASELINE: follow every new leg", research, "b_", be),
        stats("STORY: every verdict", story, "s_", be),
        stats("STORY: one position at a time", one_at_a_time(story, "s_"), "s_", be),
        stats("STORY: verdict WITH the new leg", story[story.story_dir == story.new_leg_direction], "s_", be),
        stats("STORY: verdict AGAINST new leg", story[story.story_dir != story.new_leg_direction], "s_", be),
        stats("STORY: strength < 0.4", story[story.strength < 0.4], "s_", be),
        stats("STORY: strength >= 0.4", story[story.strength >= 0.4], "s_", be),
        stats("STORY: first half", story[story.time_msc < mid], "s_", be),
        stats("STORY: second half", story[story.time_msc >= mid], "s_", be),
    ]
    for key in ("structure", "wasted_effort", "cost_per_pip", "retrace", "fading"):
        rows.append(stats(f"STORY: has '{key}'", story[story.observations.str.contains(key)], "s_", be))

    scope = "ALL DATA (holdout included)" if args.include_holdout else \
        f"research window only (last {args.holdout_frac:.0%} held out)"
    print(f"\nTP {args.tp} / SL {args.sl} pips, max {args.max_minutes:.0f} min, "
          f"commission {args.commission_pips} pip, spread <= {args.max_spread_pips} -- {scope}")
    print(pd.DataFrame(rows).fillna("").to_string(index=False))
    print(f"\nAll reads with features and outcomes -> {args.out}")

    sample = story.tail(3)
    if len(sample):
        print("\nLatest story reads:")
        for _, r in sample.iterrows():
            print(f"  [{pd.Timestamp(r.time_msc, unit='ms')}] {r.narrative}")


if __name__ == "__main__":
    main()
