"""Does unexplained effort predict the move, and at what time scale?

For each (look-back window, candle frame) we split net effort into the part explained by the
price move and the leftover ("effort beyond price"). Then we measure the rank correlation (IC)
of each part with the forward move, computed per day and summarised as mean and t-stat across
days. Research window only, active hours only (server 10-20h = London + NY)."""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import Data

d = Data(sys.argv[1] if len(sys.argv) > 1 else "ticks/GBPUSD")
step = 60_000
grid = np.arange(d.t[0] + 4 * 3600_000, d.t[-1] - 4 * 3600_000, step)
h = d.server_hour(grid)
grid = grid[(h >= 10) & (h < 20)]
day = d.day(grid)
half = d.t[0] + (d.t[-1] - d.t[0]) / 2
print(f"{len(grid):,} samples, {len(set(day))} days, research window "
      f"{pd.Timestamp(d.t[0], unit='ms').date()} -> {pd.Timestamp(d.t[-1], unit='ms').date()}\n")


def daily_ic(x, y, mask):
    df = pd.DataFrame({"x": x[mask], "y": y[mask], "d": np.array(day)[mask]})
    s = df.groupby("d").apply(lambda g: spearmanr(g.x, g.y)[0] if g.x.std() > 0 else np.nan,
                              include_groups=False).dropna()
    return s.mean(), s.mean() / (s.std() / np.sqrt(len(s)))


rows = []
for W, frame in ((600, 300), (1800, 300), (1800, 900), (3600, 900)):
    e = d.effort(grid, W, frame)
    m = d.move(grid, W)
    beta = np.polyfit(m, e, 1)[0]
    resid = e - beta * m
    for H in (900, 1800, 3600, 7200):
        y = d.fwd(grid, H)
        r = {"lookback_min": W // 60, "frame_min": frame // 60, "horizon_min": H // 60}
        for name, x in (("momentum", m), ("effort", e), ("beyond_price", resid)):
            for part, mask in (("A", grid < half), ("B", grid >= half)):
                ic, tstat = daily_ic(x, y, mask)
                r[f"{name}_{part}"] = f"{ic:+.3f} ({tstat:+.1f})"
        rows.append(r)
pd.set_option("display.width", 250)
print("IC (t-stat) in each half of the research window, A = first half, B = second half")
print(pd.DataFrame(rows).to_string(index=False))
