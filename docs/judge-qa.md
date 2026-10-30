# Judge Q&A practice — EventEcho

Short answers to say out loud: the first sentence answers the question, and the rest is backup if asked. All numbers come from `results/reproduced/` (frozen snapshot `eventecho-historical-40848349737`).

## A. The 30-second pitch

> EventEcho asks whether Indian large-cap stocks behave differently around important events, and how sure we can be. We built a reproducible event study: frozen prices with checksums, a market model, three windows, strict multiple-testing correction, cluster-bootstrap uncertainty, a held-out later sample, and 78 automatic integrity checks in CI. The strongest finding is that on a company's own results day the stock moves about 2.7 times as much as on an ordinary day; on placebo days the same measure is 0.9. Macro events show no reliable average move, and the trading rule shows no edge. We report what holds up and what does not.

## B. Method questions

**1. Why a market model and not raw returns?**
Raw returns mix the event with the whole market's move that day. The model removes the part explained by NIFTY 50 (α + β × market), so what's left (the abnormal return) is closer to the stock-specific reaction.

**2. Why a 100-day estimation window with a 20-day gap?**
100 sessions is about five months: enough to estimate β, recent enough to be relevant. The gap keeps pre-event news (leaks, run-ups) out of "normal". We also re-ran with 60 and 150 days; the counts change a little (conventional 12/8/11), and we show that rather than hide it.

**3. How did you choose Day 0?**
Day 0 is the first session in which prices could react. After-close or weekend news maps to the next session. For 19 of 21 registry events we found a timed source (for example, demonetisation was reported at 20:57 IST, and RBI decisions were reported around 10:20–10:40 IST). All 19 agree with our Day 0. We never set Day 0 from a price move.

**4. Why three windows?**
A narrow window is cleaner but can miss slow reactions. A wide one catches more but adds noise. We show all three, with how often verdicts flip: about a third of event-stock pairs change sign between ±1 and ±5. A result that holds in one window only is fragile.

**5. 441 tests: aren't some significant by chance?**
Yes. That's why we use Holm correction across all 441 at once. 8 survive (10 with HAC). Uncorrected, about 22 would pass at 5% by luck alone.

**6. Your t-test assumes the event doesn't change volatility. Isn't that wrong?**
Right, and events do raise volatility, which makes a plain t-test too eager. So for groups we added the BMP test, which uses the observed spread of standardized CARs. We also added the Kolari–Pynnönen adjustment for stocks sharing the same day, and a rank test that ignores extreme sizes. For the 21 macro events all three agree: no reliable average move (p = 0.92, 0.93, 0.61).

**7. Why bootstrap clusters, not single events?**
Events that overlap in time, or belong to one episode (like the 2022 rate hikes), are not independent. Resampling whole clusters keeps them together, so the interval isn't falsely narrow. With fewer than 3 clusters we show no interval at all.

## C. The earnings study

**8. Where did 266 dates come from, and how do you know they're right?**
- Company board-meeting histories for six stocks, and dated news reports for ITC and for the gaps.
- Every row has a link.
- A seeded random sample of 12 matched independent news 12 of 12 times, and 22 targeted checks also matched.
- We found and fixed 3 problems (a missing quarter, a Sunday date, a double entry) and noted each one.
- A test flags any two results less than 25 or more than 150 days apart.

**9. Why look at volatility instead of whether the stock went up?**
Without analyst forecasts we can't tell a beat from a miss, so ups and downs cancel. Indeed the mean CAR is −0.28% with p = 0.13. What we can measure honestly is *how much* the stock moves, and that is 2.7 times normal.

**10. Isn't 2.7× just fat tails? Stocks jump sometimes anyway.**
That's why we ran a placebo: the same stocks, 30 sessions after each results day. There the ratio is 0.88 (0.70–1.08), against 2.68 (2.17–3.23) on results days. The intervals don't come close.

**11. Why is the biggest move on Day +1, not Day 0?**
Many Indian companies report after the close or on Saturday. The first session that can react is the next one. The data match: Day +1 is 4.8× normal and Day 0 is 1.8×.

**12. Does it hold on later data?**
Pooled, yes: 2.15 for 2016–2022 and 3.82 for 2023–2025, both well above 1. Per stock, only TCS and Infosys clear the bar in both periods. With 12 holdout announcements per stock the intervals are wide, so we don't over-claim.

**13. Do IT stocks react more than banks?**
We can't say that. IT is 3.5× and banking 3.1×, but the intervals overlap heavily. Adani Enterprises is the one clear outlier: it moves a lot on ordinary days too, so results add little.

**14. Does the stock keep drifting after results, like the US "post-earnings drift"?**
We tested it, with the design fixed before the run. Up-move stocks minus down-move stocks over Day +2 to +20: +0.44 pp, interval −1.19 to +2.13. No reliable drift or reversal. On fake prices with no drift, the test wrongly fires about 1 time in 20, so it is not biased.

**15. Do TCS results move Infosys?**
They move together (correlation 0.52 on TCS results days), but they move together about as much on ordinary days (0.65). So no extra spillover is shown. Clean tests are hard because the two often report within days of each other.

**16. Your "volatility is rising" result — isn't the whole market just more volatile now?**
That is exactly why we compare with placebo days, which use the same stocks and the same model. Placebo days show no trend (−0.01 per year); results days rise +0.21 per year. The difference, +0.22 (+0.03 to +0.41), excludes zero, but only just, and without macro-event overlap it touches zero. So we call it moderate. Dropping any single year keeps it positive.

**17. You asked three new questions and one came out positive. Isn't that fishing?**
All three were written down and pushed before the run, with the main statistic for each named in advance, and we report the two null results with the same weight. One of three passing at 5% is still more than chance would usually give, but we call it moderate, not proven.

## D. Results and honesty

**18. What's your strongest result?**
The results-day volatility effect: 266 events, a placebo, held-out years, and cluster-robust intervals. On the macro side, the largest moves are Adani stocks around the Hindenburg report (−75.6% CAR over ±5 for Adani Enterprises). But that window overlaps an RBI decision, so we call it an association, not a measured cause.

**19. Can I trade on this?**
No. The hypothetical rule's holdout excess return interval is −0.26 to +0.38 pp, which includes zero. The earnings finding is about the size of moves, not their direction. Knowing a stock will move a lot is not knowing which way.

**20. What does significant mean here?**
That the move was unusual for the market model. It doesn't tell you why: overlapping news, shared episodes and timing uncertainty all remain.

**21. Why only 7 stocks and 21 macro events?**
The stocks are large, liquid names across five sectors; the events are major, sourced Indian events. That's a selected sample, and we say so. The earnings study is the counterweight: 266 events chosen by rule (every quarter), not by hand.

**22. What's the NLP part for? Isn't it decoration?**
It cross-checks our hand-assigned event categories. It scores 81% on the 21 registry headlines (never used for training), against 29% for always guessing the most common class. It disagrees on 4, which we show. It does not affect any return result.

## E. Data and engineering

**23. How can we reproduce it?**
- Prices are frozen in a snapshot with SHA-256 checksums.
- CI re-runs the analysis offline from that snapshot.
- `verify_release.py` runs 87 checks, for example: CAR = sum of abnormal returns, Holm recomputes, dashboard tables equal the CSVs, and the variance ratio recomputes from the SCARs.
- `pytest` runs 149 tests without network access.

**24. Yahoo Finance isn't official data.**
True. It's the practical free source. We record the exact field (Close for both stocks and NIFTY, so dividends are excluded on both sides) and freeze the bytes. A checksum proves sameness, not correctness. A second-vendor check is on our limits list.

**25. Why price returns and not total returns?**
NIFTY 50 is a price index, so we use stock Close too. That way both sides exclude dividends. Mixing adjusted stock prices with a price index would bias abnormal returns upward around dividend dates.

## F. Hard follow-ups (stay calm, answer short)

| Question | Answer |
|---|---|
| "Your placebo shift of 30 sessions is arbitrary." | We fixed it before running and didn't try others, so we couldn't pick a friendly one. Mid-quarter is the furthest point from both results days. |
| "Kolari-Pynnönen needs same-date events; your earnings are spread out." | Correct, and that is why it barely changes the earnings test (average correlation ≈ 0). It matters for the macro events, where 7 stocks share a day. |
| "Board-meeting dates aren't release times." | Agreed. That's why (−1,+1) is the main window, and why Day +1 carries the effect. |
| "Small holdout." | 84 announcements pooled and 12 per stock; we report the intervals and claim per-stock replication only where both intervals clear 1. |
| "What would you do next?" | Add analyst-consensus surprise data to study direction, add exchange-filing timestamps for all 266, add more stocks, and use an intraday check for the RBI and budget days. |

## G. Things never to say

- "The event caused the drop."
- "This strategy makes money."
- "IT reacts more than banks."
- "All timestamps are verified." Release *sessions* are sourced for 19 of 21 events; exact minutes are mostly bounds.
