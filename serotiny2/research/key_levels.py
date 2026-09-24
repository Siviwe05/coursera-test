"""Does the story at KEY LEVELS tell rejection from breakout?

Levels (server time, broker day = server date):
  PDH / PDL  previous day's high / low
  ASH / ASL  Asian range high / low (server 03:00-10:00, i.e. 00:00-07:00 UTC)
Event: the FIRST time each day, during server 10:00-20:00, that price trades through the level.
Outcome: from the touch, which comes first on mid price: N pips back (REJECT) or N pips through (BREAK).
Question: do engine readings on the approach (last 30 min) separate rejects from breaks, beyond what
plain price (approach size and speed) already says? Then a cost-inclusive trade test.
Research window only; A/B halves reported separately."""
import sys

import numpy as np
import pandas as pd

from common import Data
from serotiny2.outcomes import first_touch, breakeven_winrate

d = Data(sys.argv[1] if len(sys.argv) > 1 else "ticks/GBPUSD")
ts = pd.to_datetime(d.t, unit="ms")
dates = ts.normalize()
hours = ts.hour
half = d.t[0] + (d.t[-1] - d.t[0]) / 2
pip = d.spec.pip

day_hi = pd.Series(d.mid).groupby(dates).max()
day_lo = pd.Series(d.mid).groupby(dates).min()
events = []
udays = dates.unique()
for k, day in enumerate(udays):
    in_day = np.flatnonzero(dates == day)
    if len(in_day) < 1000 or day.dayofweek >= 5:
        continue
    levels = {}
    if k > 0:
        prev = udays[k - 1]
        levels["PDH"], levels["PDL"] = day_hi[prev], day_lo[prev]
    asia = in_day[(hours[in_day] >= 3) & (hours[in_day] < 10)]
    if len(asia):
        levels["ASH"], levels["ASL"] = d.mid[asia].max(), d.mid[asia].min()
    active = in_day[(hours[in_day] >= 10) & (hours[in_day] < 20)]
    if len(active) == 0:
        continue
    start_price = d.mid[active[0]]
    for name, lvl in levels.items():
        is_high = name.endswith("H")
        if (is_high and start_price >= lvl) or (not is_high and start_price <= lvl):
            continue   # already beyond the level when the session opened
        hit = np.flatnonzero(d.mid[active] > lvl) if is_high else np.flatnonzero(d.mid[active] < lvl)
        if len(hit) == 0:
            continue
        i = int(active[hit[0]])
        events.append({"i": i, "time": d.t[i], "level": name, "is_high": is_high, "price": lvl})
ev = pd.DataFrame(events)
T = ev.time.to_numpy()
toward = np.where(ev.is_high, 1, -1)                     # direction price arrived from
appr_move = d.move(T, 1800) * toward                        # pips travelled toward the level, last 30m
appr_eff = d.effort(T, 1800, 300) * toward                  # engine effort toward the level
beta = np.polyfit(appr_move, appr_eff, 1)[0]
ev["approach_pips"] = appr_move
ev["approach_effort"] = appr_eff
ev["effort_beyond_price"] = appr_eff - beta * appr_move     # >0: pushing hard for little progress
ev["last5_pips"] = d.move(T, 300) * toward                  # speed into the level
ev["half"] = np.where(T < half, "A", "B")

for N in (10, 15, 20):
    out = []
    for i, tw in zip(ev.i, toward):
        end = np.searchsorted(d.t, d.t[i] + 4 * 3600_000, side="right")
        p = (d.mid[i + 1:end] - d.mid[i]) * tw / pip
        thru, back = np.flatnonzero(p >= N), np.flatnonzero(p <= -N)
        if len(thru) == 0 and len(back) == 0:
            out.append(np.nan)
        else:
            out.append(1.0 if len(thru) == 0 or (len(back) and back[0] < thru[0]) else 0.0)
    ev[f"reject_{N}"] = out

print(f"{len(ev)} level events over {ev.time.map(lambda x: pd.Timestamp(x, unit='ms').date()).nunique()} days "
      f"({ev.level.value_counts().to_dict()})\n")
print("Base rate of REJECTION (N pips back before N pips through), by level and half:")
print(ev.groupby(["level", "half"])[["reject_10", "reject_15", "reject_20"]].mean().unstack().round(2).to_string())
print("\nall levels:", ev.groupby("half")[["reject_10", "reject_15", "reject_20"]].agg(["mean", "count"]).round(2).to_string())

print("\nRejection rate when the reading is ABOVE vs BELOW its median (by half). "
      "A useful reading shows the same gap in A and B:")
rows = []
for feat in ("effort_beyond_price", "approach_effort", "approach_pips", "last5_pips"):
    med = ev.loc[ev.half == "A", feat].median()
    for N in (10, 15):
        r = {"reading": feat, "N": N}
        for h in ("A", "B"):
            s = ev[ev.half == h]
            r[f"{h} high"] = round(s.loc[s[feat] > med, f"reject_{N}"].mean(), 2)
            r[f"{h} low"] = round(s.loc[s[feat] <= med, f"reject_{N}"].mean(), 2)
        rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\nTrade test at the touch (bid/ask fills, 1 pip commission):")
rows = []
for side_name, sgn in (("fade (bet on rejection)", -1), ("follow (bet on breakout)", 1)):
    for tp, sl in ((10, 10), (15, 10), (20, 15)):
        be = breakeven_winrate(tp, sl, 1.0)
        for h in ("A", "B"):
            s = ev[ev.half == h]
            res = [first_touch(d.t, d.bid, d.ask, int(i), int(sgn * tw), tp, sl, 4 * 3600, pip, 1.0)["pnl_pips"]
                   for i, tw in zip(s.i, np.where(s.is_high, 1, -1))]
            res = np.array(res)
            rows.append({"trade": side_name, "tp/sl": f"{tp}/{sl}", "half": h, "n": len(res),
                         "win%": round(100 * (res > 0).mean(), 1), "be%": round(100 * be, 1),
                         "avg": round(res.mean(), 2), "total": round(res.sum(), 1)})
print(pd.DataFrame(rows).to_string(index=False))
ev.to_csv("key_level_events.csv", index=False)
