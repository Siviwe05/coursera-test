"""The engine must reproduce the ORIGINAL serotiny_core math exactly."""
import itertools
import random

import numpy as np

from serotiny2.engine import (TRANSITION_RULES, ALLOWED_TRANSITIONS, SymbolSpec, RunScorer,
                              score_sequence, CandleEngine, mid_price)
from serotiny2.ticks import synthetic_ticks


# ---- verbatim copies of the original serotiny_core logic, as the reference ----
def legacy_calculate_scores_v2(sequence):
    if not sequence:
        return 0, 0
    runs = [(int(k), len(list(g))) for k, g in itertools.groupby(sequence)]
    x, y = 0, 0
    first_state, first_count = runs[0]
    if first_state == 4:
        y += 2 * first_count
    if first_state == 9:
        x += 2 * first_count
    for i in range(len(runs) - 1):
        prev_state = runs[i][0]
        curr_state, curr_count = runs[i + 1]
        rule = TRANSITION_RULES.get((prev_state, curr_state))
        if rule:
            a_val, side = rule
            if side == 0:
                x += a_val * curr_count
            else:
                y += a_val * curr_count
    return x, y


class LegacyMarketEngine:
    def __init__(self, symbol):
        self.symbol = symbol
        self.reset()

    def reset(self, open_price=None):
        self.candle_open = self.candle_high = self.candle_low = self.previous_close = open_price
        self.sequence = ""
        self.last_state = None

    def update_state(self, price):
        if self.candle_open is None:
            self.reset(price)
            return None
        delta = price - self.previous_close
        if delta == 0:
            return None
        point = 0.001 if "JPY" in self.symbol else 0.00001
        units = max(1, int(round(abs(delta) / point)))
        if delta > 0:
            if price > self.candle_high:
                raw = 9
            elif self.previous_close <= self.candle_open and price > self.candle_open:
                raw = 8
            elif price > self.candle_open:
                raw = 7
            else:
                raw = 6
        else:
            if price < self.candle_low:
                raw = 4
            elif self.previous_close >= self.candle_open and price < self.candle_open:
                raw = 3
            elif price < self.candle_open:
                raw = 2
            else:
                raw = 1
        state = self.validate(raw, delta)
        self.sequence += str(state) * units
        self.last_state = state
        self.candle_high = max(self.candle_high, price)
        self.candle_low = min(self.candle_low, price)
        self.previous_close = price
        return state

    def validate(self, raw, delta):
        if self.last_state is None:
            return raw if raw in {4, 9} else (9 if delta > 0 else 4)
        allowed = ALLOWED_TRANSITIONS.get(self.last_state, set())
        if raw in allowed:
            return raw
        candidates = allowed & ({6, 7, 8, 9} if delta > 0 else {1, 2, 3, 4})
        if not candidates:
            candidates = allowed
        for p in ([9, 8, 7, 6] if delta > 0 else [4, 3, 2, 1]):
            if p in candidates:
                return p
        return next(iter(candidates))


def legacy_replay(t_secs, prices, bar_seconds, symbol):
    engine = LegacyMarketEngine(symbol)
    bars, cur, o = [], None, None
    for t, p in zip(t_secs, prices):
        ts = int(t - (t % bar_seconds))
        if cur is not None and ts != cur:
            x, y = legacy_calculate_scores_v2(engine.sequence)
            bars.append((cur, x, y, engine.previous_close))
            engine.reset(p)
        if cur is None or ts != cur:
            cur = ts
            if o is None:
                o = p
                engine.reset(p)
        engine.update_state(p)
    return bars


# ---- tests ----
def test_score_sequence_matches_legacy():
    rng = random.Random(1)
    for _ in range(500):
        seq = "".join(str(rng.choice([1, 2, 3, 4, 6, 7, 8, 9])) * rng.randint(1, 5)
                      for _ in range(rng.randint(0, 30)))
        assert score_sequence(seq) == legacy_calculate_scores_v2(seq)


def test_run_scorer_is_incremental_score_sequence():
    rng = random.Random(2)
    for _ in range(300):
        s = RunScorer()
        seq = ""
        tot_dx = tot_dy = 0
        for _ in range(rng.randint(1, 40)):
            st, u = rng.choice([1, 2, 3, 4, 6, 7, 8, 9]), rng.randint(1, 6)
            dx, dy = s.add(st, u)
            tot_dx += dx
            tot_dy += dy
            seq += str(st) * u
            assert (s.x, s.y) == legacy_calculate_scores_v2(seq)
        assert (tot_dx, tot_dy) == (s.x, s.y)


def test_candle_engine_matches_original_replay():
    for symbol, start, pip in (("EURUSD", 1.1, 0.0001), ("USDJPY", 150.0, 0.01)):
        t, bid, ask = synthetic_ticks(n=90000, seed=3, start=start, pip=pip)
        prices = [mid_price(b, a) for b, a in zip(bid, ask)]
        t_secs = [int(x // 1000) for x in t]
        expected = legacy_replay(t_secs, prices, 300, symbol)

        eng = CandleEngine(SymbolSpec.from_name(symbol), 300)
        got = []
        for ts, p in zip(t_secs, prices):
            bar, _, _ = eng.step(ts, p)
            if bar is not None:
                got.append((bar.time_sec, bar.x, bar.y, bar.close))
        assert len(got) == len(expected) > 50
        assert got == expected
