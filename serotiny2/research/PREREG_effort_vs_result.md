# Pre-registration: effort vs result at critical zones

Written and committed **before** any code for this test was run. Nothing below may be changed after
seeing results. Any later variation gets reported as exploratory, not as the test.

## Idea (from the diary)
At a critical zone set by the previous session, watch how each side interacts with it. Trade only when
there is a clear winner, a clear loser, and the loser has no credible way back. Effort is measured
**separately from price**, using quote activity.

## Data
GBPUSD, EURUSD, AUDUSD, USDJPY, EURGBP ticks. Research window only (before the GBPUSD holdout cutoff,
about 25 Aug 2026). Halves A/B of the research window are reported separately. The holdout stays sealed.

## Definitions (server time, UTC+3)
- **Activity (effort):** number of quote updates in a minute, divided by the median count for the same
  minute of day over the previous 20 weekdays (causal). The first 20 weekdays only build the baseline.
- **Move (result):** change in mid price over the minute, in pips, signed toward the side being measured.
- **Zones:** previous day's high/low (PDH/PDL) and the Asian range high/low (ASH/ASL, 03:00–10:00).
- **Event:** the first time each zone is traded through during 10:00–20:00, provided price opened the
  session on the inside of it.
- **Battle window W:** the 15 minutes after the touch. **Decision time T** = touch + 15 min.
- **Sides:** the attacker pushes into the zone (buyers at highs, sellers at lows); the defender is the
  other side. For each side, over a window: effort = summed activity of the minutes that moved in that
  side's direction; result = summed pips of those minutes; efficiency = result / effort.

## The checklist (all must be true at T)
- **C1, clear winner by result:** net move over W is at least 3 pips in one side's direction. That side
  is the winner. If the attacker wins it's a breakout; if the defender wins it's a rejection.
- **C2, winner earns more per effort:** the winner's efficiency over W is at least 1.5 times the loser's.
- **C3, no credible pathway for the loser:** the loser's result over the last 5 minutes of W is at most
  1 pip.
- Spread at T is at most 1.5 pips, and the trade has 4 hours of data left before the cutoff.

Trade direction is the winner's direction. Entry is the quote at T (buys at the ask, sells at the bid).

## Exits (both reported)
- **Plan S (structure):** stop 1 pip beyond the loser's extreme during W, clamped to 5–25 pips;
  target 1.5 × stop; 4 h limit.
- **Plan F (fixed):** 10-pip stop and 10-pip target; 4 h limit.
Commission is 1 pip round trip, and fills use real bid/ask.

## Comparisons
1. **Price only:** C1 alone (winner by result, with no effort conditions). Does effort add anything?
2. **Price-derived effort:** C2 and C3 computed with effort = path length in pips (the old engine's
   view) instead of quote activity. Does quote activity add anything beyond price?
3. **Shuffled activity:** activity taken from a different day at the same minute of day (a 7-weekday
   shift), with everything else the same. This shows what luck looks like.

## Pass criteria (decided now)
The checklist strategy passes only if **all** of the following hold:
- average R > 0 **in both halves A and B**, under at least one exit plan chosen in advance: Plan S;
- it beats the price-only comparison in both halves;
- it beats the shuffled-activity control;
- it has at least 40 trades across pairs in total.

Only if it passes do we open the sealed holdout, once.
