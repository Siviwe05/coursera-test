"""
ticks.py -- loading tick files, plus a synthetic generator for tests.

Tick files are what tools/export_ticks.py writes on the MT5 machine:
CSV (optionally .gz) with columns  time_msc,bid,ask  -- one file per day.
"""
from __future__ import annotations

import glob
import os

import numpy as np


def load_ticks(path: str):
    """Load one tick file or a directory of them. Returns (time_msc, bid, ask)
    numpy arrays, sorted by time, with invalid quotes dropped."""
    import pandas as pd
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")) + glob.glob(os.path.join(path, "*.csv.gz")))
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"no tick files found at {path}")
    frames = [pd.read_csv(f, usecols=["time_msc", "bid", "ask"]) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])]
    df = df.sort_values("time_msc", kind="stable").reset_index(drop=True)
    return (df["time_msc"].to_numpy(np.int64), df["bid"].to_numpy(np.float64),
            df["ask"].to_numpy(np.float64))


def synthetic_ticks(n: int = 20000, seed: int = 0, start: float = 1.10000, pip: float = 0.0001,
                    spread_pips: float = 0.3, step_pips: float = 0.15, drift_pips: float = 0.0,
                    start_msc: int = 1_700_000_000_000, mean_gap_ms: float = 400.0):
    """A random-walk tick stream. With drift_pips=0 there is, by design,
    NOTHING to predict -- which is exactly what makes it a useful test:
    any strategy that 'wins' on this after costs has a bug."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift_pips, step_pips, n) * pip
    mid = np.round(start + np.cumsum(steps), 5)
    gaps = rng.exponential(mean_gap_ms, n).astype(np.int64) + 1
    t = start_msc + np.cumsum(gaps)
    half = spread_pips * pip / 2
    return t, np.round(mid - half, 5), np.round(mid + half, 5)
