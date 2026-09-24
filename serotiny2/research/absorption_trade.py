"""Trade test of the absorption signal: effort beyond what price shows -> fade it.
Thresholds and beta fixed on the FIRST half of the research window, evaluated on the SECOND half.
Real bid/ask fills, 1 pip commission, one position at a time. Holdout (last 25%) untouched."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from serotiny2.engine import SymbolSpec, CandleEngine, mid_price
from serotiny2.ticks import load_ticks
from serotiny2.outcomes import first_touch, breakeven_winrate

spec = SymbolSpec.from_name("GBPUSD")
t, b, a = load_ticks((sys.argv[1] if len(sys.argv) > 1 else 'ticks/GBPUSD'))
cut = t[0] + (t[-1] - t[0]) * 0.75
k = t < cut
t, b, a = t[k], b[k], a[k]
mid = np.array([mid_price(x, y) for x, y in zip(b, a)])
half = t[0] + (t[-1] - t[0]) * 0.5

eng = CandleEngine(spec, 300)
net = np.zeros(len(t))
for i in range(len(t)):
    _, dx, dy = eng.step(int(t[i] // 1000), mid[i])
    net[i] = dx - dy
cum = np.concatenate([[0], np.cumsum(net)])

grid = np.arange(t[0] + 900_000, t[-1] - 900_000, 10_000)
h = pd.to_datetime(grid, unit='ms').hour
grid = grid[(h >= 10) & (h < 20)]
idx = np.searchsorted(t, grid, side='right') - 1

rows = []
for W in (120, 600):
    j = np.maximum(np.searchsorted(t, grid - W * 1000, side='right') - 1, 0)
    e = cum[idx + 1] - cum[j + 1]
    ret = (mid[idx] - mid[j]) / spec.pip
    first = grid < half
    beta = np.polyfit(ret[first], e[first], 1)[0]
    resid = e - beta * ret
    for q in (0.95, 0.98, 0.99):
        hi, lo = np.quantile(resid[first], q), np.quantile(resid[first], 1 - q)
        sig = np.where(resid >= hi, -1, np.where(resid <= lo, 1, 0))   # fade the unrewarded effort
        for tp, sl, mins in ((6, 5, 15), (5, 5, 15), (8, 6, 30), (10, 8, 45)):
            be = breakeven_winrate(tp, sl, 1.0)
            for part, mask in (('fit-half', grid < half), ('TEST-half', grid >= half)):
                busy, res = -1, []
                for g, i, s in zip(grid[mask], idx[mask], sig[mask]):
                    if s == 0 or g < busy or a[i] - b[i] > 1.5 * spec.pip:
                        continue
                    r = first_touch(t, b, a, int(i), int(s), tp, sl, mins * 60, spec.pip, 1.0)
                    if r['outcome'] == 'NODATA':
                        continue
                    busy = r['exit_msc']
                    res.append(r['pnl_pips'])
                res = np.array(res)
                days = len(set(pd.to_datetime(grid[mask], unit='ms').date))
                rows.append({'W': W, 'q': q, 'tp/sl/min': f"{tp}/{sl}/{mins}", 'part': part,
                             'trades': len(res), 'per_day': round(len(res) / days, 1),
                             'win': f"{(res > 0).mean():.1%}" if len(res) else '', 'be': f"{be:.1%}",
                             'avg_pips': round(res.mean(), 2) if len(res) else np.nan,
                             'total': round(res.sum(), 1)})
pd.set_option('display.width', 200)
df = pd.DataFrame(rows)
print(df.to_string(index=False))
