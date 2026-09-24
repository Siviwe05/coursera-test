"""Shared helpers for the research scripts: load ticks, build and cache per-tick engine effort,
split the data into research and holdout. The holdout is the last HOLDOUT_FRAC of time and is
never returned unless explicitly asked for."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from serotiny2.engine import SymbolSpec, CandleEngine, mid_price   # noqa: E402
from serotiny2.ticks import load_ticks                               # noqa: E402

HOLDOUT_FRAC = 0.25
SERVER_UTC_OFFSET = 3     # from meta.json: server clock = UTC+3


class Data:
    def __init__(self, tick_dir, symbol="GBPUSD", frames=(60, 300, 900), include_holdout=False,
                 cache_dir=None):
        self.spec = SymbolSpec.from_name(symbol)
        t, b, a = load_ticks(tick_dir)
        self.holdout_start = t[0] + (t[-1] - t[0]) * (1 - HOLDOUT_FRAC)
        if not include_holdout:
            k = t < self.holdout_start
            t, b, a = t[k], b[k], a[k]
        self.t, self.bid, self.ask = t, b, a
        self.mid = np.round((b + a) / 2, 5)
        self.cum_net, self.cum_tot = {}, {}
        cache_dir = cache_dir or os.path.join(tick_dir, "_cache")
        os.makedirs(cache_dir, exist_ok=True)
        for bs in frames:
            f = os.path.join(cache_dir, f"effort_{bs}_{len(t)}_{t[0]}_{t[-1]}.npz")
            if os.path.exists(f):
                z = np.load(f)
                net, tot = z["net"], z["tot"]
            else:
                eng = CandleEngine(self.spec, bs)
                net = np.zeros(len(t))
                tot = np.zeros(len(t))
                for i in range(len(t)):
                    _, dx, dy = eng.step(int(t[i] // 1000), self.mid[i])
                    net[i] = dx - dy
                    tot[i] = dx + dy
                np.savez(f, net=net, tot=tot)
            self.cum_net[bs] = np.concatenate([[0], np.cumsum(net)])
            self.cum_tot[bs] = np.concatenate([[0], np.cumsum(tot)])

    # index of the last tick at or before each time
    def at(self, times):
        return np.searchsorted(self.t, times, side="right") - 1

    def effort(self, times, window_s, frame):
        i, j = self.at(times), np.maximum(self.at(times - window_s * 1000), 0)
        return self.cum_net[frame][i + 1] - self.cum_net[frame][j + 1]

    def move(self, times, window_s):
        """Price change in pips over the past window."""
        i, j = self.at(times), np.maximum(self.at(times - window_s * 1000), 0)
        return (self.mid[i] - self.mid[j]) / self.spec.pip

    def fwd(self, times, horizon_s):
        i, j = self.at(times), self.at(times + horizon_s * 1000)
        return (self.mid[j] - self.mid[i]) / self.spec.pip

    def server_hour(self, times):
        return pd.to_datetime(times, unit="ms").hour

    def day(self, times):
        return pd.to_datetime(times, unit="ms").date
