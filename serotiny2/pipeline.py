"""
pipeline.py -- ticks in, story reads out.

This is the ONE code path shared by research/backtesting and the live
bot: both feed ticks one at a time into StoryPipeline.on_tick(). A read
can only ever be built from ticks that already arrived, so a backtest
cannot see anything the live bot couldn't.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .engine import SymbolSpec, CandleEngine, mid_price
from .legs import LegTracker, Leg
from .story import StoryReader, StoryConfig, StoryRead


@dataclass
class PipelineConfig:
    reversal_pips: float = 2.0     # how far the other side must take back to end a leg
    candle_seconds: int = 300      # frame for the classic candle-framed effort
    story: StoryConfig = None


class StoryPipeline:
    def __init__(self, spec: SymbolSpec, config: Optional[PipelineConfig] = None):
        self.spec = spec
        self.cfg = config or PipelineConfig()
        self.candles = CandleEngine(spec, self.cfg.candle_seconds)
        self.legs = LegTracker(spec, self.cfg.reversal_pips)
        self.reader = StoryReader(spec.pip, self.cfg.story)
        self.last_leg: Optional[Leg] = None

    def on_tick(self, time_msc: int, bid: float, ask: float) -> Optional[StoryRead]:
        price = mid_price(bid, ask)
        _, dx, dy = self.candles.step(int(time_msc // 1000), price)
        leg = self.legs.update(time_msc, price, dx, dy)
        if leg is None:
            return None
        self.last_leg = leg
        read = self.reader.on_leg(leg, time_msc, price)
        if read is not None:
            read.features["bid"] = bid
            read.features["ask"] = ask
            read.features["spread_pips"] = (ask - bid) / self.spec.pip
        return read
