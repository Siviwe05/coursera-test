"""Trade test: fade large 30/60-minute moves (GBPUSD gives them back over 1-2 hours).
Thresholds are fixed on half A of the research window and evaluated on half B.
Variants test whether a 'the push has stopped' confirmation helps, and whether the ENGINE
version of that confirmation beats a plain-price version.
Real bid/ask fills, 1 pip commission, one position at a time, server 10-20h."""
import sys

import numpy as np
import pandas as pd

from common import Data
from serotiny2.outcomes import first_touch, breakeven_winrate

d = Data(sys.argv[1] if len(sys.argv) > 1 else "ticks/GBPUSD")
grid = np.arange(d.t[0] + 3600_000, d.t[-1] - 4 * 3600_000, 60_000)
grid = grid[np.isin(d.server_hour(grid), range(10, 20))]
idx = d.at(grid)
half = d.t[0] + (d.t[-1] - d.t[0]) / 2
A, B = grid < half, grid >= half
days = {p: len(set(d.day(grid[m]))) for p, m in (("A", A), ("B", B))}

rows = []
for W in (1800, 3600):
    move = d.move(grid, W)
    last15_move = d.move(grid, 900)
    last15_eff = d.effort(grid, 900, 300)
    for q in (0.90, 0.95):
        thr = np.quantile(np.abs(move[A]), q)
        base = np.where(move >= thr, -1, np.where(move <= -thr, 1, 0))       # fade direction
        variants = {
            "fade only": base,
            "+ price stalled (last 15m against move)": np.where(np.sign(last15_move) == base, base, 0),
            "+ ENGINE stalled (last 15m effort against move)": np.where(np.sign(last15_eff) == base, base, 0),
        }
        for vname, sig in variants.items():
            for tp, sl, mins in ((10, 10, 60), (15, 15, 120), (20, 15, 120), (25, 20, 240)):
                be = breakeven_winrate(tp, sl, 1.0)
                for part, mask in (("A", A), ("B", B)):
                    busy, res = -1, []
                    for g, i, s in zip(grid[mask], idx[mask], sig[mask]):
                        if s == 0 or g < busy or d.ask[i] - d.bid[i] > 1.5 * d.spec.pip:
                            continue
                        r = first_touch(d.t, d.bid, d.ask, int(i), int(s), tp, sl, mins * 60, d.spec.pip, 1.0)
                        if r["outcome"] == "NODATA":
                            continue
                        busy = r["exit_msc"]
                        res.append(r["pnl_pips"])
                    res = np.array(res)
                    rows.append({"lookback": f"{W // 60}m", "q": q, "thr_pips": round(thr, 1), "variant": vname,
                                 "tp/sl/min": f"{tp}/{sl}/{mins}", "half": part, "trades": len(res),
                                 "per_day": round(len(res) / days[part], 2),
                                 "win": round(100 * (res > 0).mean(), 1) if len(res) else np.nan,
                                 "be": round(100 * be, 1),
                                 "avg": round(res.mean(), 2) if len(res) else np.nan,
                                 "total": round(res.sum(), 1)})
df = pd.DataFrame(rows)
wide = df.pivot_table(index=["lookback", "q", "thr_pips", "variant", "tp/sl/min"], columns="half",
                      values=["trades", "win", "avg", "total"], aggfunc="first")
wide.columns = [f"{a}_{b}" for a, b in wide.columns]
wide = wide[["trades_A", "win_A", "avg_A", "total_A", "trades_B", "win_B", "avg_B", "total_B"]]
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print(wide.to_string())
df.to_csv("fade_trade_results.csv", index=False)
