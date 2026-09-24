import numpy as np

from serotiny2.engine import SymbolSpec
from serotiny2.legs import LegTracker, UP, DOWN
from serotiny2.pipeline import StoryPipeline
from serotiny2.ticks import synthetic_ticks
from serotiny2.outcomes import first_touch, breakeven_winrate

EU = SymbolSpec.from_name("EURUSD")
PIP = EU.pip


def path(*waypoints_pips, step=0.2, base=1.10000):
    """Build a tick path through waypoints (in pips from base), step pips per tick.
    Waypoints can be (pips, wiggle) tuples to add back-and-forth effort."""
    prices = [base]
    for wp in waypoints_pips:
        target, wiggle = (wp if isinstance(wp, tuple) else (wp, 0))
        cur = (prices[-1] - base) / PIP
        n = max(1, int(round(abs(target - cur) / step)))
        for k in range(1, n + 1):
            p = cur + (target - cur) * k / n
            prices.append(round(base + p * PIP, 5))
            if wiggle and k % 2 == 0 and k < n:
                prices.append(round(base + (p - np.sign(target - cur) * wiggle) * PIP, 5))
                prices.append(round(base + p * PIP, 5))
    return prices


def feed(tracker, prices, t0=0, gap=500):
    legs = []
    for k, p in enumerate(prices):
        leg = tracker.update(t0 + k * gap, p)
        if leg:
            legs.append((k, leg))
    return legs


def test_legs_confirm_only_after_reversal():
    tr = LegTracker(EU, reversal_pips=2.0)
    prices = path(5, 2, 8, 4)   # up 5, down 3, up 6, down 4
    legs = feed(tr, prices)
    dirs = [l.direction for _, l in legs]
    sizes = [round(l.size_pips, 1) for _, l in legs]
    assert dirs == [UP, DOWN, UP]
    assert sizes == [5.0, 3.0, 6.0]
    for k, leg in legs:
        # the leg is released strictly AFTER its extreme, once price came back 2 pips
        assert leg.confirmed_msc > leg.end_msc
        back = abs(prices[k] - leg.end_price) / PIP
        assert back >= 2.0 - 1e-9


def test_no_lookahead_prefix_consistency():
    """Reads produced on a prefix of the data must be identical to the reads
    produced at the same moments when the full data is replayed."""
    t, bid, ask = synthetic_ticks(n=60000, seed=11)

    def reads(n):
        pipe = StoryPipeline(EU)
        out = []
        for i in range(n):
            r = pipe.on_tick(int(t[i]), float(bid[i]), float(ask[i]))
            if r:
                out.append((r.time_msc, r.verdict, round(r.edge, 6), r.narrative))
        return out

    full, part = reads(60000), reads(35000)
    assert len(part) > 20
    assert full[:len(part)] == part


def test_story_reads_wasted_buyer_effort_as_sell():
    """Your example: sellers have the advantage, buyers keep putting in effort
    that isn't making a difference -> SELL."""
    prices = path(
        12,               # early buyer push
        6,                # sellers answer
        (9, 0.6),         # buyers grind back up with lots of back-and-forth (effort, little result)
        2,                # sellers take it all back and more: lower low
        (5, 0.8),         # buyers grind again, even more effort, lower high
        -3,               # sellers push to another lower low
        (0.5, 1.0),       # buyers try again: expensive and going nowhere
        -2.0,             # sellers start again -> confirms the buyer leg
    )
    pipe = StoryPipeline(EU)
    last = None
    for k, p in enumerate(prices):
        r = pipe.on_tick(k * 500, p - 0.00001, p + 0.00001)
        if r:
            last = r
    assert last is not None
    assert last.verdict == "SELL", last.narrative
    keys = {o.key for o in last.observations}
    assert "structure" in keys
    assert "cost_per_pip" in keys or "wasted_effort_buyers" in keys
    assert last.invalidation_price is not None and last.invalidation_price > last.price
    assert "Sellers have the advantage" in last.narrative


def test_first_touch_fills_at_bid_ask():
    t = np.arange(10, dtype=np.int64) * 1000
    mid = 1.1 + np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) * PIP
    bid, ask = mid - 0.5 * PIP, mid + 0.5 * PIP
    r = first_touch(t, bid, ask, 0, +1, tp_pips=3, sl_pips=3, max_seconds=60, pip=PIP, commission_pips=1.0)
    # entry at ask(0)=+0.5, needs bid >= +3.5 -> bid at mid 4 = 3.5
    assert r["outcome"] == "TP" and abs(r["pnl_pips"] - 2.0) < 1e-9 and r["exit_msc"] == 4000
    r = first_touch(t, bid, ask, 0, -1, tp_pips=3, sl_pips=3, max_seconds=60, pip=PIP, commission_pips=1.0)
    assert r["outcome"] == "SL" and abs(r["pnl_pips"] + 4.0) < 1e-9
    assert abs(breakeven_winrate(6, 5, 1) - 6 / 11) < 1e-12
