"""
legs.py -- cuts the tick stream into chapters ("legs").

A leg is one side's push: price moving one way until the other side
takes back at least `reversal_pips`. Up legs are buyer pushes, down legs
are seller pushes. The story is the sequence of legs.

NO LOOK-AHEAD, by construction: a leg only becomes known when the
reversal is CONFIRMED (price has come back `reversal_pips` off the
extreme). The leg's end is the extreme, but it is released at
`confirmed_msc`, and nothing downstream may act on it earlier. The ticks
between the extreme and the confirmation are handed to the NEW leg, which
is what they really were.

Each leg carries two effort readings from the tick engine:
  * leg frame (x, y)      -- the leg's own ticks scored against the leg
                              itself (open = the leg's starting pivot).
                              Measures the fight INSIDE the push: new
                              highs vs. dips.
  * candle frame (bar_x,  -- effort the classic candle engine attributed
    bar_y)                    to this leg's ticks. Includes the big
                              open-crossing transitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .engine import SymbolSpec, score_path

UP, DOWN = 1, -1


@dataclass
class Leg:
    index: int
    direction: int          # +1 buyers' push, -1 sellers' push
    start_price: float
    end_price: float        # the extreme reached
    start_msc: int
    end_msc: int            # time of the extreme
    confirmed_msc: Optional[int]   # when the leg became knowable (None = still running)
    n_ticks: int
    path_pips: float        # total distance travelled tick to tick
    max_pullback_pips: float  # deepest retrace inside the leg before its extreme
    x: int                  # leg-frame bullish effort
    y: int                  # leg-frame bearish effort
    bar_x: int              # candle-frame bullish effort on these ticks
    bar_y: int
    pip: float

    @property
    def side(self) -> str:
        return "BUYERS" if self.direction == UP else "SELLERS"

    @property
    def size_pips(self) -> float:
        return abs(self.end_price - self.start_price) / self.pip

    @property
    def duration_s(self) -> float:
        return (self.end_msc - self.start_msc) / 1000.0

    @property
    def effort_with(self) -> int:
        """Effort spent pushing in the leg's own direction."""
        return self.x if self.direction == UP else self.y

    @property
    def effort_against(self) -> int:
        """Effort the other side put in during this leg (the dips/fight-back)."""
        return self.y if self.direction == UP else self.x

    @property
    def effort_per_pip(self) -> float:
        """How much effort each pip of progress cost. High = expensive progress."""
        return self.effort_with / max(self.size_pips, 1e-9)

    @property
    def directness(self) -> float:
        """Net progress / distance travelled (1.0 = a straight line)."""
        return self.size_pips / self.path_pips if self.path_pips > 0 else 0.0

    @property
    def conviction(self) -> float:
        """(with - against) / total, in [-1, 1]. How one-sided the fight was."""
        tot = self.x + self.y
        return (self.effort_with - self.effort_against) / tot if tot else 0.0

    @property
    def bar_effort_with(self) -> int:
        return self.bar_x if self.direction == UP else self.bar_y

    @property
    def bar_effort_against(self) -> int:
        return self.bar_y if self.direction == UP else self.bar_x


class LegTracker:
    """Streaming zig-zag. Feed update() every tick; it returns a completed
    Leg exactly on the tick that confirms it, else None."""

    def __init__(self, spec: SymbolSpec, reversal_pips: float = 2.0):
        self.spec = spec
        self.threshold = reversal_pips * spec.pip
        self.direction = None
        self._buf = []          # (msc, price, bar_dx, bar_dy); element 0 is the leg's starting pivot
        self._ext_idx = 0
        self._count = 0

    # -- public ------------------------------------------------------------
    def update(self, msc: int, price: float, bar_dx: int = 0, bar_dy: int = 0) -> Optional[Leg]:
        self._buf.append((msc, price, bar_dx, bar_dy))
        if self.direction is None:
            return self._seek_first_pivot(price)

        ext_price = self._buf[self._ext_idx][1]
        if self.direction == UP:
            if price > ext_price:
                self._ext_idx = len(self._buf) - 1
            elif ext_price - price >= self.threshold:
                return self._confirm(msc)
        else:
            if price < ext_price:
                self._ext_idx = len(self._buf) - 1
            elif price - ext_price >= self.threshold:
                return self._confirm(msc)
        return None

    def current_leg(self) -> Optional[Leg]:
        """The leg in progress (unconfirmed), built from ticks seen so far."""
        if self.direction is None or len(self._buf) < 2:
            return None
        return self._build(self._buf[: self._ext_idx + 1], self.direction, None)

    @property
    def last_price(self) -> Optional[float]:
        return self._buf[-1][1] if self._buf else None

    # -- internals ---------------------------------------------------------
    def _seek_first_pivot(self, price):
        prices = [b[1] for b in self._buf]
        lo_i = min(range(len(prices)), key=prices.__getitem__)
        hi_i = max(range(len(prices)), key=prices.__getitem__)
        if price - prices[lo_i] >= self.threshold:
            self.direction = UP
            self._buf = self._buf[lo_i:]
        elif prices[hi_i] - price >= self.threshold:
            self.direction = DOWN
            self._buf = self._buf[hi_i:]
        else:
            return None
        ps = [b[1] for b in self._buf]
        pick = max if self.direction == UP else min
        self._ext_idx = pick(range(len(ps)), key=ps.__getitem__)
        return None

    def _confirm(self, msc):
        leg = self._build(self._buf[: self._ext_idx + 1], self.direction, msc)
        self._buf = self._buf[self._ext_idx:]
        self.direction = -self.direction
        ps = [b[1] for b in self._buf]
        pick = max if self.direction == UP else min
        self._ext_idx = pick(range(len(ps)), key=ps.__getitem__)
        return leg

    def _build(self, ticks, direction, confirmed_msc) -> Leg:
        pip = self.spec.pip
        prices = [t[1] for t in ticks]
        path = sum(abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))) / pip
        pull = 0.0
        run_ext = prices[0]
        for p in prices:
            if direction == UP:
                run_ext = max(run_ext, p)
                pull = max(pull, run_ext - p)
            else:
                run_ext = min(run_ext, p)
                pull = max(pull, p - run_ext)
        x, y = score_path(prices, self.spec.point)
        bar_x = sum(t[2] for t in ticks[1:])
        bar_y = sum(t[3] for t in ticks[1:])
        idx = self._count
        if confirmed_msc is not None:
            self._count += 1
        return Leg(index=idx, direction=direction,
                   start_price=prices[0], end_price=prices[-1],
                   start_msc=ticks[0][0], end_msc=ticks[-1][0], confirmed_msc=confirmed_msc,
                   n_ticks=len(ticks) - 1, path_pips=path, max_pullback_pips=pull / pip,
                   x=x, y=y, bar_x=bar_x, bar_y=bar_y, pip=pip)
