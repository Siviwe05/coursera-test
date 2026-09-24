# Serotiny v2 — reading the market's story

Rebuilt from the tick engine up. The belief: the market tells a story
tick by tick. The job is to read it, and trade only when it's clear:
take 5–10 pips and get out, a handful of times a day.

```
ticks ──> ENGINE ──> LEGS ──> STORY READER ──> (decision) ──> (execution)
          9 states    chapters  who has the       next          next
          + scoring   (pushes)  advantage, why
```

Live trading and research use **one code path**: `StoryPipeline.on_tick()`.
A read can only be built from ticks that already arrived, so a backtest
can't see anything the live bot couldn't. A test checks this.

## Files

| file | what it does |
|---|---|
| `engine.py` | The original tick engine: 9 states, allowed transitions, transition scoring. The math is unchanged and tests check it tick for tick against the original code. It also has an incremental scorer, so each tick's effort can be credited to the leg it belongs to. |
| `legs.py` | Cuts ticks into **legs**: one side's push, which ends when the other side takes back `reversal_pips`. A leg is released only once that reversal is confirmed. Each leg records size, duration, path length, deepest pullback, and effort measured two ways: against the leg itself, and against the classic candle. |
| `story.py` | Reads the last few legs the way a discretionary trader would (see below) and gives a verdict, a strength, the level the losing side would need to reclaim, and a plain-English narrative. |
| `pipeline.py` | Ticks in, reads out. |
| `outcomes.py` | What really happened after a read: target before stop within the time limit. Buys fill at the ask, sells at the bid, commission included. |
| `ticks.py` | Loads exported tick files, plus a random-walk generator for tests. |
| `tools/export_ticks.py` | **Run on your MT5 machine.** Exports real bid/ask ticks, one gzip CSV per day. |
| `tools/research_story.py` | Replays ticks, labels every read with its outcome, and reports results against the break-even win rate. |

## How the story is read

Each time a leg is confirmed, the reader asks:

1. **Structure.** Higher highs and higher lows, or lower highs and lower lows?
2. **Wasted effort.** Did a side push with as much effort as last time and still fail to get further? ("buyers are putting a lot of effort but it's not making a difference")
3. **Cost per pip.** Which side pays more effort for each pip of progress?
4. **Retrace.** Did the last move wipe out the other side's push, or barely dent it?
5. **Fading.** Is one side's push shrinking each time?

Each answer credits one side. The verdict is BUY, SELL or UNCLEAR. Example output:

> Sellers have the advantage: sellers are building lower highs and lower lows; buyers pushed with as much or more effort than before (140 -> 210) but couldn't get above 1.10090 -- their effort isn't making a difference; buyers are paying 1.8x more effort per pip than sellers. Buyers would need to get back above 1.10090 (3.1 pips away) to change the story. Read: SELL.

The weights are a **starting point from reasoning, not from data**. Every
underlying number is exported so research can replace them with measured ones.

## Workflow

```bash
# 1. on the MT5 machine (pip install MetaTrader5 pandas)
python -m serotiny2.tools.export_ticks --symbols EURUSD GBPUSD --days 120

# 2. research
python -m serotiny2.tools.research_story --ticks ticks/EURUSD --symbol EURUSD

# sanity check: a random walk must NOT show an edge
python -m serotiny2.tools.research_story --synthetic

# tests
python -m pytest serotiny2/tests -q
```

**Holdout discipline:** the research script hides the most recent 25% of
the data by default. We tune on the older data only, and open the holdout
once, at the end (`--include-holdout`).

**Break-even:** with a 6-pip target, 5-pip stop and 1 pip commission (spread
is already in the fills), the break-even win rate is 54.5%. Every kind of
read has to clear that on unseen data before it trades.

## Findings (GBPUSD, 86 trading days, 28 May – 24 Sep 2026)

Research window: 28 May – 25 Aug. **Holdout: 25 Aug – 24 Sep, still unopened** (nothing has earned
a test on it). Every result is reported separately for the two halves of the research window (A, B);
a real effect has to show up in both. Scripts are in `research/`; run them from that folder with the
tick folder as the argument.

1. **Story rules (5–10 pips):** no edge. The verdicts won 38.5% of the time against a 54.5%
   break-even, no better than following every leg (`tools/research_story.py`, first 30 days).
2. **Story numbers when a leg completes:** no directional information. A model trained on older
   data scored AUC 0.45–0.49 on later days at leg sizes of 2, 3 and 5 pips (`info_test.py`).
3. **The engine's effort re-measures price movement.** It is 73–90% correlated with the price move,
   and the leftover part ("effort beyond price") predicts nothing: its sign flips between A and B at
   every scale from 10 to 60 minutes (`scale_ic.py`). The weak signal reported on the first 30 days
   did not replicate.
4. **Fading large moves:** there is a rank correlation, but it isn't tradeable. Measured in pips,
   the largest moves continue slightly rather than revert. Every fade variant with costs loses in at
   least one half (`fade_trade.py`).
5. **Key levels** (previous day high/low, Asian range): out of 127 first touches, levels broke
   slightly more often than they held (about 58%). Betting on the rejection loses about 3 pips per
   trade. Following the breakout comes out roughly even after costs (−0.03 / +0.48 pips per trade,
   10/10 target/stop). One engine reading pointed the same way in both halves: breakouts with strong
   effort into the level did better (A +1.9 vs −2.0; B +0.7 vs +0.3 pips). That comes from about 30
   events per group, so it is a lead worth testing on more pairs, not an edge (`key_levels.py`).

6. **The situation solver** (`solver/`, all 5 pairs; 34,461 situations; walk-forward over 43 days).
   For each 5-minute moment during London and New York hours it found the 100 most similar past
   situations, using price story, key levels, engine effort and cross-pair currency strength. It then
   took the best of 8 buy/sell stop-and-target plans, or passed.
   Result: about 1,000 trades, **−0.21R per trade in half A and −0.23R in half B** (t ≈ −4), with a
   36% win rate. Before costs it is about zero; the loss is the spread plus commission. The solver's
   confidence meant nothing: trades it expected to earn +0.1R and trades it expected to earn +0.8R
   realised the same −0.2R. Removing the engine or the cross-pair data changed little. Shuffled-memory
   controls, where situations carry no information, did no better and no worse in any meaningful way.
   No pair was positive (USDJPY was the least negative at −0.02R).

**Bottom line so far:** no approach tried, fixed rules or the situation solver, with or without the engine or
cross-pair data, has shown a tradeable edge after costs on 5 pairs over 3 months. The engine is a faithful, deterministic reading of the price path. It has not yet shown
information that the price path itself doesn't already carry.

## Status

- [x] Engine v2 (checked against the original), leg tracker, story reader, honest outcome labelling
- [x] Tick exporter, research script, random-walk sanity check (no false edge)
- [x] Real-tick research on 86 days of GBPUSD: no tradeable edge yet (see Findings)
- [ ] Visual story viewer (a session chart with legs and narrative, to compare with your own reading)
- [ ] Decision rules from evidence, then a tick-level backtest
- [ ] Live MT5 shell around the same pipeline, then a demo forward test
