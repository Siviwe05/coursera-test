"""
outcomes.py -- what actually happened after a read.

For a read at tick i we ask: if we had entered right then, would price
have hit the target before the stop, within the time limit?

Fills are honest about the spread:
  BUY  enters at the ASK, exits at the BID
  SELL enters at the BID, exits at the ASK
Commission is subtracted in pips (FundingPips: $10/lot round trip ~= 1.0 pip).
Stops fill exactly at the stop level -- no slippage modelled yet, so live
results will be slightly worse than this on losing trades.
"""
from __future__ import annotations

import numpy as np


def first_touch(times, bid, ask, i, direction, tp_pips, sl_pips, max_seconds, pip,
                commission_pips=1.0):
    """Returns dict(outcome='TP'|'SL'|'TIME'|'NODATA', pnl_pips, exit_msc, hold_s)."""
    t0 = times[i]
    end = np.searchsorted(times, t0 + int(max_seconds * 1000), side="right")
    if end <= i + 1:
        return {"outcome": "NODATA", "pnl_pips": 0.0, "exit_msc": t0, "hold_s": 0.0}

    eps = pip * 1e-6   # a price landing exactly on the level counts as a touch
    if direction > 0:
        entry = ask[i]
        path = bid[i + 1:end]
        tp_hit = path >= entry + tp_pips * pip - eps
        sl_hit = path <= entry - sl_pips * pip + eps
        sign = 1.0
    else:
        entry = bid[i]
        path = ask[i + 1:end]
        tp_hit = path <= entry - tp_pips * pip + eps
        sl_hit = path >= entry + sl_pips * pip - eps
        sign = -1.0

    tp_i = int(np.argmax(tp_hit)) if tp_hit.any() else None
    sl_i = int(np.argmax(sl_hit)) if sl_hit.any() else None
    if tp_i is not None and (sl_i is None or tp_i < sl_i):
        j, outcome, gross = tp_i, "TP", tp_pips
    elif sl_i is not None:
        j, outcome, gross = sl_i, "SL", -sl_pips
    else:
        j, outcome = len(path) - 1, "TIME"
        gross = sign * (path[j] - entry) / pip
    exit_msc = int(times[i + 1 + j])
    return {"outcome": outcome, "pnl_pips": gross - commission_pips,
            "exit_msc": exit_msc, "hold_s": (exit_msc - t0) / 1000.0}


def breakeven_winrate(tp_pips, sl_pips, commission_pips):
    """Win rate needed to break even when every trade ends at TP or SL.
    (Spread is already inside the fills, so it isn't in this formula.)"""
    win = tp_pips - commission_pips
    loss = sl_pips + commission_pips
    return loss / (win + loss)
