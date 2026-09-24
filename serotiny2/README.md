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

## Status

- [x] Engine v2 (checked against the original), leg tracker, story reader, honest outcome labelling
- [x] Tick exporter, research script, random-walk sanity check (no false edge)
- [ ] Run on real ticks: which reads actually predict the move
- [ ] Visual story viewer (a session chart with legs and narrative, to compare with your own reading)
- [ ] Decision rules from evidence, then a tick-level backtest
- [ ] Live MT5 shell around the same pipeline, then a demo forward test
