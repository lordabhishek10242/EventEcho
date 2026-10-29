# EventEcho

EventEcho is a reproducible **event study** of Indian stocks. It asks: around important events — company earnings, policy changes, economic data releases and major industry news — did stock returns differ from what the market alone would predict, how big and how certain were those differences, and did a simple fixed trading rule hold up on later, unseen events?

It is an evidence explorer, not a prediction engine. Results show **association**, not cause.

> **Current verified result set:** [`results/reproduced/`](results/reproduced/) — one CI run (Historical Rebuild run 40915721578) on frozen snapshot `eventecho-historical-40848349737`, with its [manifest](results/reproduced/manifest.json), [integrity report](results/reproduced/release_validation.json) and [dashboard](results/reproduced/eventecho_demo.html).
> Read the [final evidence audit](docs/final-evidence-audit-2026-10-29.md) before quoting any number.

## Questions the project answers

1. What were actual stock returns around each event, and how did they compare with NIFTY 50?
2. What were the abnormal returns (actual minus market-model expected) and cumulative abnormal returns (CAR) over three event windows?
3. Which CARs are statistically different from zero after correcting for many tests, and do they survive different model settings?
4. Did stock volatility change from before to after the event?
5. What is the average reaction across events (CAAR), and how do event classes, sectors and stocks compare, given how few independent episodes exist?
6. How much does the choice of event window change the conclusions?
7. Do patterns seen in earlier events repeat in later, unseen events?
8. Can a simple text classifier label event headlines into event classes?
9. Did a fixed, hypothetical post-event trading rule beat NIFTY 50 on a later holdout, after costs and with honest uncertainty?

Every line of the competition problem statement (PS 03) is mapped to code, output files and dashboard panels in [docs/problem-statement-coverage.md](docs/problem-statement-coverage.md).

## Universe and data

- **Stocks:** RELIANCE, TCS, INFOSYS (INFY), HDFC_BANK, ITC, ADANI_ENTERPRISES, ADANI_PORTS (NSE tickers on Yahoo Finance).
- **Benchmark:** NIFTY 50 (`^NSEI`), a price index, not a total-return index.
- **Period:** 2016-01-01 to 2025-12-31 (observed 2016-01-04 to 2025-12-31).
- **Events:** 21 sourced events in [`results/event_master.csv`](results/event_master.csv): demonetisation, COVID lockdown, GST rollout, two Union Budgets, six RBI policy decisions, two Lok Sabha results, two GDP releases, five FY2025 Q4 earnings releases and the Hindenburg report on Adani. This is a small, hand-selected sample, not a representative population.
- **Price convention:** Yahoo Finance `Close` for stocks **and** `Close` for NIFTY 50. These are **price returns**: dividends are excluded on both sides, so this is **not total shareholder return**. The run manifest records the fields used, and the dashboard reads its statement from there.
- **Calendar:** observed NIFTY 50 sessions (non-missing close). Stock prices are reindexed to these sessions before returns are computed; missing prices are never filled.
- **Provenance:** prices are frozen in a snapshot with raw provider responses and SHA-256 checksums. A checksum proves the bytes are the same; it does not prove the provider's prices are correct, and a later download may differ.

## Method in brief

| Step | What is done |
|---|---|
| Day 0 | The first trading session on which the event could affect closing prices. After-close or non-trading-day news maps to the next session. Exact clock times are recorded only when a source supports them; they are never inferred from price moves. |
| Market model | OLS: stock return = α + β × NIFTY 50 return + error, fitted on 100 complete sessions that end 20 sessions before the event window starts (Days −125 to −26 when there are no gaps). |
| Abnormal return | actual return − (α + β × NIFTY 50 return). |
| CAR windows | CAR(−1,+1) = 3 sessions, CAR(−3,+3) = 7, CAR(−5,+5) = 11. A CAR is left missing if any session is missing. |
| Significance | Conventional test with parameter uncertainty, plus a Newey–West (HAC) robustness version. Both are Holm-corrected across all 441 event × stock × window tests. All tests are exploratory. |
| Robustness | Market model re-fitted with 60, 100 and 150 sessions; event-timing alternatives for uncertain events; extreme-day check for COVID. |
| Volatility | Realized volatility in sessions −20:−1 vs +1:+20 (Day 0 excluded), annualized with √252. |
| Overlap | CAR-window and volatility-window overlaps are computed on trading sessions. Overlapping events are excluded from category summaries and grouped into dependency clusters with shared-episode events. |
| Average reaction (CAAR) | For each event, abnormal returns are averaged over the chosen stocks, then over events, and added up from Day −5. 95% bands come from resampling whole dependency clusters (5,000 draws). Groups with fewer than 3 clusters get no band. Groups: all events, events without overlap, 6 event classes, 5 sectors, 7 stocks, development and holdout. |
| Sectors | A broad project grouping (`STOCK_SECTORS` in `src/config.py`): Energy & Conglomerate, IT Services (2 stocks), Banking, FMCG, Infrastructure (Adani Group, 2 stocks). Not an official classification. |
| Window choice | For each window: test counts, significance counts, average size and standard error, plus how often the CAR sign or the significance verdict changes between windows. |
| Replication | Patterns from development events (to Feb 2023) are checked on holdout events (from Oct 2023): overall direction, class-level direction, a sign-prediction rule against a coin flip, and stock-level reaction size. |
| Headline classifier (NLP) | TF-IDF (words + characters) and logistic regression, trained on 374 real, sourced headlines in `results/headline_corpus.csv`. Tested by holding out whole search groups and on the 21 registry headlines, which are never used for training. Compared with keyword rules and a majority-class baseline. See [docs/headline-classifier.md](docs/headline-classifier.md). |
| Strategy (hypothetical) | Buy at the close of the first session after Day 0, sell 3 sessions later, equal weight, 10 bps per side (also 0, 5, 20, 30). Development events 2016 to Feb 2023, holdout Oct 2023 to 2025. Uncertainty from a cluster bootstrap (5,000 draws). |

Details: [timing, overlap and return convention](docs/event-timing-overlap-return-convention.md), [event source audit](docs/event-source-and-validation-audit.md), [problem-statement coverage](docs/problem-statement-coverage.md).

## Main findings (run 40915721578)

- 8 of 441 CAR tests are significant after Holm correction (conventional); 10 with HAC. Across 60/100/150-session models the counts change (conventional 12/8/11, HAC 14/10/10).
- 7 results are significant in all three model settings. All 7 have an attribution caveat: 6 involve Adani stocks around the Hindenburg report (or the RBI decision whose window overlaps it), and 1 is ITC around the COVID lockdown.
- Only Monetary Policy has enough independent episodes (3) for a category number, and its mean CAR is close to zero. Categories cannot be ranked.
- Average reaction across all 21 events is small and uncertain: mean CAR(−5,+5) is −0.75% (95% band −2.62% to +1.33%). Without the 8 overlapping events it is +0.80% (−0.21% to +2.40%). The Adani stocks drive most of the negative average.
- Event classes: only government policy (5 events) and monetary policy (6 events, 3 clusters) have enough independent clusters for an interval. Monetary policy CAR(−1,+1) is +0.62% (+0.03% to +1.77%), but with 3 clusters that interval is very rough. Other classes are descriptive only.
- Window choice matters: 4, 1 and 5 HAC-Holm significant results for the ±1, ±3 and ±5 windows. Between ±1 and ±5, a third of event-stock pairs flip the CAR sign.
- Replication is weak. The overall direction repeats in only one window of three, and using development class signs to guess holdout signs is right 4/5, 3/5 and 2/5 times, no better than a coin flip given so few events. Stocks with big moves in one sample also move more in the other, but that mostly reflects how volatile they are.
- Headline classifier: 81% accuracy on the 21 registry headlines (95% CI 62–95%), against 29% for always guessing the most common class, and 75% under grouped cross-validation. Hand-written keyword rules score higher, but they were written after reading the data, so that comparison is not blind.
- Strategy: across all events the mean excess return vs NIFTY 50 is −0.22 pp at 10 bps per side and is negative at every cost level. Holdout mean excess is +0.11 pp, but its 95% interval is −0.26 to +0.38 pp, which **includes zero**. No trading edge is established.
- 19 of 21 events still lack a verified exact release time, and some CARs change a lot under the alternative Day 0 (for example the Hindenburg report).

## Quick start

Python 3.10+ is recommended (CI uses 3.11).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q                   # network-free tests
```

**Open the dashboard:** open `results/reproduced/eventecho_demo.html` in any modern browser. It is one self-contained file with the data embedded; it needs no server or internet.

**Check the committed outputs:**

```bash
python scripts/verify_release.py --analysis-dir results/reproduced
```

**Rebuild from a frozen snapshot (offline).** Put the snapshot folder at `data/snapshots/<snapshot-id>/` (it is in the artifact of the GitHub run that created it), then:

```bash
python scripts/run_analysis.py --snapshot-id <snapshot-id> --offline --output-dir results/reproduced
python scripts/build_demo.py --analysis-dir results/reproduced
python scripts/verify_release.py --analysis-dir results/reproduced
```

**Full rebuild with fresh prices (needs internet):**

```bash
python scripts/rebuild_historical.py
```

This downloads prices, freezes a new uniquely named snapshot, analyses it offline and builds the dashboard. A new download may differ from the frozen snapshot.

## CI workflows

| Workflow | When | What |
|---|---|---|
| Tests | pushes to `main`/`feat/**`, pull requests | network-free test suite |
| Historical Rebuild | changes on `main`, or manual | tests → snapshot (new download, or `source_run_id` to reuse a frozen one) → analysis → dashboard → `verify_release.py`; uploads a 7-day artifact. With `publish_results` on a non-default branch it commits the verified outputs. |
| Recover frozen market snapshot | registry/method changes on `main`, or manual | acquisition with a hard failure if incomplete, offline analysis, dashboard, integrity checks |

A failed acquisition or a failed integrity check fails the workflow; artifacts are still uploaded for diagnosis.

## Repository layout

```text
src/                       analysis library (data, events, event study, statistics,
                           volatility, overlap audit, trading, bootstrap, pooled CAAR,
                           headline classifier, reporting)
scripts/run_analysis.py    run the full analysis on a snapshot or price file
scripts/build_demo.py      build the HTML dashboard from one analysis folder
scripts/verify_release.py  integrity checks on one analysis folder
scripts/rebuild_historical.py, scripts/acquire_market_snapshot.py  snapshot acquisition
tests/                     network-free unit and regression tests
results/event_master.csv   the event registry (input)
results/headline_corpus.csv labelled, sourced headlines for the classifier (input)
results/reproduced/        current verified outputs, manifest, dashboard
results/legacy_2026-10-27/ quarantined original 9-event exports (do not cite)
docs/                      audits and methodology notes
notebook/, notebooks/      original notebook (unchanged) and an exploration notebook
```

## How to read the results

- **% vs pp:** returns and CARs are in %. Differences between returns (excess return, CAR changes) are in percentage points (pp).
- **Significant is not causal.** Events overlap, share episodes, and some timestamps are uncertain. A significant CAR says the move was unusual for the market model, not why it happened.
- **Holm correction** controls false positives across the whole set of 441 tests, but the tests are not independent, so the results stay exploratory.
- **HAC** is a robustness check for serial correlation, not a fix for overlap or event-day volatility.
- **Peer stocks in earnings events** show possible spillover, not the issuer's own effect.
- **The strategy is hypothetical:** close-price fills, fixed costs, no market impact, taxes or position limits. It is not live trading performance.

## Limitations

- 21 hand-picked events and 7 large stocks chosen after the fact; few independent episodes.
- Daily closes cannot separate intraday reactions; 19 events lack a verified exact release time.
- One-factor market model; event-day volatility increases are not modelled.
- Overlap audits cover registered events only; other news may confound windows.
- Price returns only; no total-return comparison.
- Most sectors contain one stock, and several event classes have one or two events, so those comparisons are descriptive.
- The headline corpus uses web search titles labelled by one person; it is not a random news sample.
- Yahoo Finance is a third-party source and may revise history; prices were not checked against a second vendor. Raw prices are not redistributed here.

## Data and licence

Market data come from Yahoo Finance through the unofficial `yfinance` package; check the provider's terms before redistributing data. Event sources are linked in the registry. No project licence has been chosen yet, so no licence is granted for reuse.

## Disclaimer

Educational research project. Not investment advice.
