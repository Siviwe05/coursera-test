"""
engine.py -- the Serotiny tick engine.

The math is carried over UNCHANGED from the original serotiny_core.py:
the 9 tick states, the allowed-transition graph, and the transition
scoring table. What changed is packaging:

  * TickClassifier  -- the original MarketEngine's classification logic,
                       measured against an explicit FRAME (open/high/low).
                       The frame used to always be a clock candle; now the
                       caller decides (a candle, or the current leg -- see
                       legs.py).
  * RunScorer       -- an incremental, O(1)-per-tick version of the
                       original calculate_scores_v2(). Same totals, but it
                       can tell you how much effort EACH tick contributed,
                       which is what lets us attribute effort to legs.
  * CandleEngine    -- the original per-candle engine (reset at every bar
                       open), kept so the classic candle-framed c_val is
                       still available and provably identical to before.

State reference (delta = price - previous price):
  up-ticks:    9 = new frame high
               8 = crossed UP through the frame open
               7 = rising, above the open
               6 = rising, still below the open
  down-ticks:  4 = new frame low
               3 = crossed DOWN through the frame open
               2 = falling, below the open
               1 = falling, still above the open
Each state is repeated once per point moved, so bigger moves weigh more.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

# (previous_state, state) -> (points, side) ; side 0 = bullish effort (x), 1 = bearish effort (y)
TRANSITION_RULES = {
    (1, 3): (3, 1), (1, 7): (2, 0),
    (2, 6): (1, 0), (2, 4): (3, 1), (2, 8): (4, 0),
    (3, 8): (5, 0), (3, 4): (3, 1), (3, 2): (2, 1), (3, 6): (2, 0),
    (4, 6): (2, 0), (4, 8): (5, 0),
    (6, 8): (3, 0), (6, 2): (2, 1),
    (7, 9): (3, 0), (7, 3): (4, 1), (7, 1): (1, 1),
    (8, 3): (5, 1), (8, 9): (3, 0), (8, 7): (2, 0), (8, 1): (2, 1),
    (9, 1): (2, 1), (9, 3): (5, 1),
}

ALLOWED_TRANSITIONS = {
    1: {1, 3, 7}, 2: {2, 4, 6, 8}, 3: {2, 4, 6, 8}, 4: {4, 6, 8},
    6: {2, 6, 8}, 7: {1, 3, 7, 9}, 8: {1, 3, 7, 9}, 9: {1, 3, 9},
}

BULL, BEAR = 0, 1


@dataclass(frozen=True)
class SymbolSpec:
    """Price geometry for one instrument."""
    name: str
    pip: float     # 0.0001 for most majors, 0.01 for JPY pairs
    point: float   # smallest quoted increment the engine counts units in

    @classmethod
    def from_name(cls, name: str) -> "SymbolSpec":
        if "JPY" in name.upper():
            return cls(name, pip=0.01, point=0.001)
        return cls(name, pip=0.0001, point=0.00001)


def mid_price(bid: float, ask: float) -> float:
    """Same mid-price convention as the original engine."""
    return round((bid + ask) / 2, 5)


# ------------------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------------------
def score_sequence(sequence) -> tuple[int, int]:
    """Batch scorer. Identical to the original calculate_scores_v2():
    run-length encode the state sequence, give the first run a bonus if it
    is a 4 or a 9, then score every transition between runs."""
    if not sequence:
        return 0, 0
    runs = [(int(k), len(list(g))) for k, g in itertools.groupby(sequence)]
    x = y = 0
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
            points, side = rule
            if side == BULL:
                x += points * curr_count
            else:
                y += points * curr_count
    return x, y


class RunScorer:
    """Incremental score_sequence(). add(state, units) returns the (dx, dy)
    this tick contributed; .x/.y always equal score_sequence() of everything
    added so far."""
    __slots__ = ("x", "y", "_state", "_rate", "_side")

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = 0
        self.y = 0
        self._state = None
        self._rate = 0
        self._side = BULL

    def add(self, state: int, units: int) -> tuple[int, int]:
        if state != self._state:
            if self._state is None:
                if state == 4:
                    self._rate, self._side = 2, BEAR
                elif state == 9:
                    self._rate, self._side = 2, BULL
                else:
                    self._rate, self._side = 0, BULL
            else:
                self._rate, self._side = TRANSITION_RULES.get((self._state, state), (0, BULL))
            self._state = state
        d = self._rate * units
        if self._side == BULL:
            self.x += d
            return d, 0
        self.y += d
        return 0, d


# ------------------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------------------
class TickClassifier:
    """The original MarketEngine classification, against an explicit frame.

    anchor(price) starts a new frame at `price` (the original reset()).
    step(price) classifies one tick and returns (state, units), or None if
    the price did not move (or the frame was not anchored yet -- in which
    case this tick anchors it, exactly like the original)."""
    __slots__ = ("point", "open", "high", "low", "prev", "last_state")

    def __init__(self, point: float):
        self.point = point
        self.anchor(None)

    def anchor(self, open_price):
        self.open = open_price
        self.high = open_price
        self.low = open_price
        self.prev = open_price
        self.last_state = None

    def step(self, price: float):
        if self.open is None:
            self.anchor(price)
            return None
        delta = price - self.prev
        if delta == 0:
            return None
        units = max(1, int(round(abs(delta) / self.point)))

        if delta > 0:
            if price > self.high:
                raw = 9
            elif self.prev <= self.open and price > self.open:
                raw = 8
            elif price > self.open:
                raw = 7
            else:
                raw = 6
        else:
            if price < self.low:
                raw = 4
            elif self.prev >= self.open and price < self.open:
                raw = 3
            elif price < self.open:
                raw = 2
            else:
                raw = 1

        state = self._validate(raw, delta)
        self.last_state = state
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.prev = price
        return state, units

    def _validate(self, raw: int, delta: float) -> int:
        if self.last_state is None:
            return raw if raw in (4, 9) else (9 if delta > 0 else 4)
        allowed = ALLOWED_TRANSITIONS.get(self.last_state, set())
        if raw in allowed:
            return raw
        candidates = allowed & ({6, 7, 8, 9} if delta > 0 else {1, 2, 3, 4})
        if not candidates:
            candidates = allowed
        for p in ((9, 8, 7, 6) if delta > 0 else (4, 3, 2, 1)):
            if p in candidates:
                return p
        return next(iter(candidates))


def score_path(prices, point: float) -> tuple[int, int]:
    """Score a price path in its own frame: anchored at prices[0], every
    later price classified and scored. Used for leg-framed effort."""
    clf = TickClassifier(point)
    scorer = RunScorer()
    clf.anchor(prices[0])
    for p in prices[1:]:
        r = clf.step(p)
        if r is not None:
            scorer.add(*r)
    return scorer.x, scorer.y


# ------------------------------------------------------------------------------
# Candle-framed engine (the original behaviour)
# ------------------------------------------------------------------------------
@dataclass
class Bar:
    time_sec: int      # bar open, epoch seconds
    open: float
    high: float
    low: float
    close: float
    x: int
    y: int
    n_ticks: int

    @property
    def c_val(self) -> int:
        return self.x - self.y


class CandleEngine:
    """Tick-by-tick candle engine, identical in behaviour to the original
    replay_ticks_to_bars(): the frame resets at every bar open.

    step() returns (closed_bar_or_None, dx, dy) where dx/dy is the effort
    the CURRENT tick added inside its candle."""

    def __init__(self, spec: SymbolSpec, bar_seconds: int):
        self.bar_seconds = bar_seconds
        self.clf = TickClassifier(spec.point)
        self.scorer = RunScorer()
        self.bar_ts = None
        self.o = self.h = self.l = None
        self.n = 0

    def step(self, t_sec: int, price: float):
        bar_ts = t_sec - (t_sec % self.bar_seconds)
        closed = None
        if self.bar_ts is not None and bar_ts != self.bar_ts:
            closed = Bar(self.bar_ts, self.o, self.h, self.l, self.clf.prev,
                         self.scorer.x, self.scorer.y, self.n)
            self.clf.anchor(price)
            self.scorer.reset()
            self.o = self.h = self.l = price
            self.n = 0
        if self.bar_ts is None or bar_ts != self.bar_ts:
            self.bar_ts = bar_ts
            if self.o is None:
                self.o = self.h = self.l = price
                self.clf.anchor(price)

        dx = dy = 0
        r = self.clf.step(price)
        if r is not None:
            dx, dy = self.scorer.add(*r)
        if price > self.h:
            self.h = price
        if price < self.l:
            self.l = price
        self.n += 1
        return closed, dx, dy
