"""
story.py -- reads the story told by the recent legs.

Every time a leg is confirmed, the reader looks at the last few chapters
and asks the questions a discretionary reader asks:

  * Who is winning the STRUCTURE?      (higher highs/lows vs lower highs/lows)
  * Is anyone's effort being WASTED?   (a push with as much effort as the
                                         last one, that still fails to get
                                         further -- "buyers are putting in a
                                         lot of effort but it's not making a
                                         difference")
  * Who is paying more per pip?        (effort spent / progress made)
  * How deep are the fight-backs?      (does one side erase the other's
                                         push, or barely dent it?)
  * Is anyone fading?                  (each push smaller than the last)

Each answer is an Observation that credits one side, with a weight and a
plain-English sentence. The read's verdict is whichever side the
observations favour, by how much, plus the level the other side would
need to reclaim to change the story ("had they done this they could've
stood a chance").

The weights below are a STARTING POINT from reasoning, not from data.
Every underlying number is also exported in `features`, so the research
script can measure what actually predicts the next 5-10 pips and we can
replace hand-set weights with measured ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .legs import Leg, UP, DOWN


@dataclass
class StoryConfig:
    window_legs: int = 8            # how many recent chapters to read
    min_legs: int = 4               # don't read a story shorter than this
    w_structure: float = 2.0
    w_wasted_effort: float = 1.5
    w_cost_per_pip: float = 1.0
    w_retrace: float = 1.0
    w_fading: float = 0.5
    cost_ratio: float = 1.5         # "paying X times more per pip" threshold
    deep_retrace: float = 1.0       # counter-leg >= 100% of the push it answered
    shallow_retrace: float = 0.5    # counter-leg <= 50% of the push it answered
    verdict_edge: float = 2.5       # net score needed to call a side


@dataclass
class Observation:
    favours: str      # 'BUYERS' or 'SELLERS'
    weight: float
    key: str
    text: str


@dataclass
class StoryRead:
    time_msc: int
    price: float
    verdict: str                  # 'BUY', 'SELL' or 'UNCLEAR'
    edge: float                   # sellers_score - buyers_score  (>0 favours SELL)
    strength: float               # |edge| / max possible, 0..1
    buyers_score: float
    sellers_score: float
    invalidation_price: Optional[float]   # level the losing side must reclaim
    invalidation_pips: Optional[float]    # distance from price to that level
    new_leg_direction: int        # the leg that just STARTED (+1 up, -1 down)
    observations: list = field(default_factory=list)
    features: dict = field(default_factory=dict)
    narrative: str = ""


def _other(side):
    return "SELLERS" if side == "BUYERS" else "BUYERS"


def _name(side):
    return "Buyers" if side == "BUYERS" else "Sellers"


class StoryReader:
    def __init__(self, pip: float, config: Optional[StoryConfig] = None):
        self.pip = pip
        self.cfg = config or StoryConfig()
        self.legs: list[Leg] = []

    def on_leg(self, leg: Leg, now_msc: int, price: float) -> Optional[StoryRead]:
        """Call with every confirmed leg (and the tick that confirmed it).
        Returns a read once enough story exists."""
        self.legs.append(leg)
        if len(self.legs) > 4 * self.cfg.window_legs:
            self.legs = self.legs[-2 * self.cfg.window_legs:]
        window = self.legs[-self.cfg.window_legs:]
        if len(window) < self.cfg.min_legs:
            return None
        return self._read(window, now_msc, price)

    # ------------------------------------------------------------------
    def _read(self, legs, now_msc, price) -> StoryRead:
        c = self.cfg
        pushes = {"BUYERS": [l for l in legs if l.direction == UP],
                  "SELLERS": [l for l in legs if l.direction == DOWN]}
        obs: list[Observation] = []
        f = {}

        # ---- 1. structure ------------------------------------------------
        ups, downs = pushes["BUYERS"], pushes["SELLERS"]
        hh = hl = ll = lh = False
        if len(ups) >= 2 and len(downs) >= 2:
            hh = ups[-1].end_price > ups[-2].end_price
            lh = ups[-1].end_price < ups[-2].end_price
            hl = downs[-1].end_price > downs[-2].end_price
            ll = downs[-1].end_price < downs[-2].end_price
            if hh and hl:
                obs.append(Observation("BUYERS", c.w_structure, "structure",
                                       "buyers are building higher highs and higher lows"))
            elif ll and lh:
                obs.append(Observation("SELLERS", c.w_structure, "structure",
                                       "sellers are building lower highs and lower lows"))
        f.update(hh=hh, hl=hl, ll=ll, lh=lh)

        # ---- 2. wasted effort (the core read) ------------------------------
        for side in ("BUYERS", "SELLERS"):
            p = pushes[side]
            key = side.lower()
            f[f"{key}_effort_change"] = None
            f[f"{key}_failed_extreme"] = None
            if len(p) < 2:
                continue
            prev, last = p[-2], p[-1]
            effort_change = last.effort_with / max(prev.effort_with, 1)
            if side == "BUYERS":
                failed = last.end_price <= prev.end_price
                level = prev.end_price
                where = "above"
            else:
                failed = last.end_price >= prev.end_price
                level = prev.end_price
                where = "below"
            progress_change = last.size_pips / max(prev.size_pips, 1e-9)
            f[f"{key}_effort_change"] = effort_change
            f[f"{key}_progress_change"] = progress_change
            f[f"{key}_failed_extreme"] = failed
            if effort_change >= 1.0 and (failed or progress_change < 0.7):
                if failed:
                    why = f"couldn't get {where} {level:.5f}"
                else:
                    why = f"only got {last.size_pips:.1f} pips vs {prev.size_pips:.1f} before"
                obs.append(Observation(_other(side), c.w_wasted_effort, f"wasted_effort_{key}",
                                       f"{_name(side).lower()} pushed with as much or more effort than before "
                                       f"({prev.effort_with} -> {last.effort_with}) but {why} -- "
                                       f"their effort isn't making a difference"))

        # ---- 3. cost per pip -----------------------------------------------
        def side_cost(p):
            prog = sum(l.size_pips for l in p)
            eff = sum(l.effort_with for l in p)
            return (eff / prog) if prog > 0 else None
        bc, sc_ = side_cost(ups), side_cost(downs)
        f["buyers_effort_per_pip"] = bc
        f["sellers_effort_per_pip"] = sc_
        f["cost_ratio_buy_over_sell"] = (bc / sc_) if (bc and sc_) else None
        if bc and sc_:
            if bc / sc_ >= c.cost_ratio:
                obs.append(Observation("SELLERS", c.w_cost_per_pip, "cost_per_pip",
                                       f"buyers are paying {bc / sc_:.1f}x more effort per pip than sellers"))
            elif sc_ / bc >= c.cost_ratio:
                obs.append(Observation("BUYERS", c.w_cost_per_pip, "cost_per_pip",
                                       f"sellers are paying {sc_ / bc:.1f}x more effort per pip than buyers"))

        # ---- 4. retrace of the last push -------------------------------------
        last, prev = legs[-1], legs[-2]
        r = last.size_pips / max(prev.size_pips, 1e-9)
        f["last_retrace_ratio"] = r
        if r >= c.deep_retrace:
            obs.append(Observation(last.side, c.w_retrace, "retrace",
                                   f"{_name(last.side).lower()} wiped out the whole last "
                                   f"{_name(prev.side).lower()[:-1]} push ({r:.0%})"))
        elif r <= c.shallow_retrace:
            obs.append(Observation(prev.side, c.w_retrace, "retrace",
                                   f"{_name(last.side).lower()} could only take back {r:.0%} "
                                   f"of the {_name(prev.side).lower()[:-1]} push"))

        # ---- 5. fading pushes --------------------------------------------------
        for side in ("BUYERS", "SELLERS"):
            p = pushes[side]
            fading = len(p) >= 3 and p[-3].size_pips > p[-2].size_pips > p[-1].size_pips
            f[f"{side.lower()}_fading"] = fading
            if fading:
                sizes = " -> ".join(f"{l.size_pips:.1f}" for l in p[-3:])
                obs.append(Observation(_other(side), c.w_fading, f"fading_{side.lower()}",
                                       f"{_name(side).lower()}' pushes keep shrinking ({sizes} pips)"))

        # ---- raw numbers for research ------------------------------------------
        for side, p in pushes.items():
            k = side.lower()
            f[f"{k}_legs"] = len(p)
            f[f"{k}_progress_pips"] = sum(l.size_pips for l in p)
            f[f"{k}_mean_conviction"] = (sum(l.conviction for l in p) / len(p)) if p else None
            f[f"{k}_mean_directness"] = (sum(l.directness for l in p) / len(p)) if p else None
        f["last_leg_size_pips"] = last.size_pips
        f["last_leg_effort_per_pip"] = last.effort_per_pip
        f["last_leg_conviction"] = last.conviction
        f["last_leg_duration_s"] = last.duration_s
        f["last_leg_bar_effort_with"] = last.bar_effort_with
        f["last_leg_bar_effort_against"] = last.bar_effort_against

        # ---- verdict ------------------------------------------------------------
        buyers = sum(o.weight for o in obs if o.favours == "BUYERS")
        sellers = sum(o.weight for o in obs if o.favours == "SELLERS")
        edge = sellers - buyers
        max_possible = c.w_structure + 2 * c.w_wasted_effort + c.w_cost_per_pip + c.w_retrace + 2 * c.w_fading
        if edge >= c.verdict_edge:
            verdict = "SELL"
        elif -edge >= c.verdict_edge:
            verdict = "BUY"
        else:
            verdict = "UNCLEAR"

        # the level the losing side would need to reclaim
        inv = None
        if verdict == "SELL" and ups:
            inv = max(ups[-1].end_price, price)
        elif verdict == "BUY" and downs:
            inv = min(downs[-1].end_price, price)
        inv_pips = abs(inv - price) / self.pip if inv is not None else None

        read = StoryRead(time_msc=now_msc, price=price, verdict=verdict, edge=edge,
                         strength=abs(edge) / max_possible, buyers_score=buyers, sellers_score=sellers,
                         invalidation_price=inv, invalidation_pips=inv_pips,
                         new_leg_direction=-last.direction, observations=obs, features=f)
        read.narrative = self._narrate(read)
        return read

    @staticmethod
    def _narrate(r: StoryRead) -> str:
        if r.verdict == "UNCLEAR":
            if not r.observations:
                return "Nothing is being said right now -- no side is showing its hand."
            parts = "; ".join(o.text for o in r.observations)
            return f"Contested story: {parts}. No clear winner, keep watching."
        winner = "SELLERS" if r.verdict == "SELL" else "BUYERS"
        loser = _other(winner)
        for_ = [o.text for o in r.observations if o.favours == winner]
        against = [o.text for o in r.observations if o.favours == loser]
        s = f"{_name(winner)} have the advantage: " + "; ".join(for_) + "."
        if against:
            s += " In " + _name(loser).lower() + "' favour: " + "; ".join(against) + "."
        if r.invalidation_price is not None:
            side_word = "above" if r.verdict == "SELL" else "below"
            s += (f" {_name(loser)} would need to get back {side_word} {r.invalidation_price:.5f} "
                  f"({r.invalidation_pips:.1f} pips away) to change the story.")
        s += f" Read: {r.verdict}."
        return s
