"""Engine-level test: does effort (x - y) over the recent past predict the forward mid-price move?
Sampled every 10s during active hours. Research window only (first 75%). Daily rank-IC, mean and t-stat."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from serotiny2.engine import SymbolSpec, CandleEngine, mid_price
from serotiny2.ticks import load_ticks

spec = SymbolSpec.from_name("GBPUSD")
t, b, a = load_ticks((sys.argv[1] if len(sys.argv) > 1 else 'ticks/GBPUSD'))
cut = t[0] + (t[-1] - t[0]) * 0.75
keep = t < cut
t, b, a = t[keep], b[keep], a[keep]
mid = np.array([mid_price(x, y) for x, y in zip(b, a)])

eff = {}
for bs in (60, 300):
    eng = CandleEngine(spec, bs)
    dx = np.zeros(len(t)); dy = np.zeros(len(t))
    for k in range(len(t)):
        _, dx[k], dy[k] = eng.step(int(t[k] // 1000), mid[k])
    eff[bs] = np.concatenate([[0], np.cumsum(dx - dy)])      # cumulative net effort
    eff[f"{bs}_tot"] = np.concatenate([[0], np.cumsum(dx + dy)])

grid = np.arange(t[0] + 900_000, t[-1] - 900_000, 10_000)
hours = pd.to_datetime(grid, unit='ms').hour
grid = grid[(hours >= 10) & (hours < 20)]          # server time 10-20 = London + NY overlap
idx = np.searchsorted(t, grid, side='right') - 1    # last tick at/before the sample time
day = pd.to_datetime(grid, unit='ms').date


def past(arr_cum, W):
    j = np.searchsorted(t, grid - W * 1000, side='right') - 1
    return arr_cum[idx + 1] - arr_cum[np.maximum(j, 0) + 1]


def fwd_ret(H):
    j = np.searchsorted(t, grid + H * 1000, side='right') - 1
    return (mid[j] - mid[idx]) / spec.pip


def ic(x, y):
    df = pd.DataFrame({'x': x, 'y': y, 'd': day}).dropna()
    daily = df.groupby('d').apply(lambda g: spearmanr(g.x, g.y)[0] if g.x.std() > 0 else np.nan).dropna()
    return daily.mean(), daily.mean() / (daily.std() / np.sqrt(len(daily)))


print(f"{len(grid):,} samples over {len(set(day))} days (server 10-20h)\n")
rows = []
for W in (30, 120, 600):
    ret_w = (mid[idx] - mid[np.maximum(np.searchsorted(t, grid - W * 1000, side='right') - 1, 0)]) / spec.pip
    for bs in (60, 300):
        e = past(eff[bs], W)
        etot = past(eff[f"{bs}_tot"], W)
        # "effort not making a difference": net effort left over after what price movement explains
        beta = np.polyfit(ret_w, e, 1)[0]
        resid = e - beta * ret_w
        for H in (60, 300, 900):
            y = fwd_ret(H)
            r = {'window_s': W, 'candle_s': bs, 'horizon_s': H}
            r['IC price momentum'], r['t_mom'] = ic(ret_w, y)
            r['IC net effort'], r['t_eff'] = ic(e, y)
            r['IC effort beyond price'], r['t_resid'] = ic(resid, y)
            r['corr(effort, price)'] = np.corrcoef(e, ret_w)[0, 1]
            rows.append(r)
out = pd.DataFrame(rows).round(3)
pd.set_option('display.width', 250)
print(out.to_string(index=False))
