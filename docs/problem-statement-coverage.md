# Problem-statement coverage (PS 03)

Each line of the problem statement is mapped to the code that does it, the output file that shows it, and where to see it on the dashboard. "Done" means it is built, tested and present in the verified run in `results/reproduced/`. It does not mean the result supports a causal claim.

## Basic functionalities

| PS requirement | Code | Output file(s) | Dashboard panel | Status |
|---|---|---|---|---|
| Import historical stock prices, benchmark returns and event dates | `src/snapshots.py`, `src/data.py`, `src/events.py`, `scripts/build_earnings_calendar.py` | `manifest.json`, `snapshot_metadata/`, `event_master.csv`, `earnings_coverage.csv` | Data provenance | Done: frozen snapshot with checksums; 21 sourced events plus 266 sourced results dates |
| Define estimation windows and event windows | `src/config.py`, `src/event_study.py` | `manifest.json` (100-session estimation, 20-session gap, ±1/±3/±5 windows) | "How to read" note, Data provenance | Done |
| Calculate actual, expected and abnormal returns | `src/event_study.py` | `multi_stock_event_study.csv` | Selected event chart | Done |
| Aggregate abnormal returns across events using CAR | `src/event_study.py`, `src/pooled_analysis.py` | `multi_stock_car.csv`, `caar_event_time.csv`, `caar_window_summary.csv` | CAR heatmap, Average reaction across events (CAAR) | Done: per-event CAR plus CAAR across events with cluster-bootstrap bands |
| Visualize event-time returns and volatility changes | `src/reporting.py`, `src/volatility.py` | `event_volatility.csv`, `caar_event_time.csv` | Return chart, CAAR chart, volatility bar chart | Done |
| Compare reactions across stocks, sectors and event categories | `src/pooled_analysis.py`, `src/category_analysis.py`, `src/config.py` (`STOCK_SECTORS`) | `caar_window_summary.csv`, `category_car_summary.csv`, `cross_stock_consistency.csv` | Compare event classes, sectors and stocks | Done. Sector grouping is the project's own, and most sectors hold one stock |

## Advanced functionalities

| PS requirement | Code | Output file(s) | Dashboard panel | Status |
|---|---|---|---|---|
| Estimate expected returns with a market model / linear regression | `src/event_study.py` (OLS, Newey–West coefficient covariance) | `multi_stock_event_study.csv` | Selected event | Done |
| Basic statistical significance measures | `src/statistics.py`, `src/robust_tests.py` | `multi_stock_significance.csv`, `robust_tests_registry.csv`, `earnings_summary.csv` | Inference & uncertainty; Robust group tests | Done: t-test with parameter uncertainty, HAC version, Holm over 441 tests; plus BMP, Kolari–Pynnönen and Corrado rank tests for groups |
| Compare alternative event-window lengths | `src/pooled_analysis.py` (`window_choice_tables`) | `window_choice_summary.csv`, `window_choice_agreement.csv` | How the event-window choice changes conclusions | Done |
| Categorize event headlines using NLP or a basic text classifier | `src/headline_classifier.py`, `results/headline_corpus.csv` | `headline_classifier_*.csv`, `event_headline_classification.csv` | Event headline classifier (NLP) | Done: see [headline-classifier.md](headline-classifier.md) |
| Test whether event patterns generalize to a separate sample (earnings) | `src/earnings_study.py` | `earnings_replication.csv`, `earnings_placebo.csv` | Company results days | Done: 182 development vs 84 holdout announcements; placebo days |
| Test whether event patterns generalize to a separate sample | `src/config.py` (fixed split), `src/pooled_analysis.py` (`pattern_generalization`), `src/clustered_inference.py` | `pattern_generalization.csv`, `clustered_strategy_inference.csv`, `sample_performance.csv` | Do patterns repeat on unseen events?, Holdout uncertainty | Done: development events up to Feb 2023, holdout from Oct 2023 |

## Brownie points

| PS item | Where | Status |
|---|---|---|
| Compare multiple event windows and explain how the choice affects conclusions | Window-choice panel and files; estimation-window robustness (60/100/150 sessions) | Done |
| Account for broader market movements when estimating abnormal returns | Market model against NIFTY 50 on a shared session calendar | Done |
| Identify possible confounding events or overlapping announcements | `src/event_audit.py`: CAR-window and volatility-window overlap, shared episodes, timing cautions; overlapping events excluded from category summaries and clustered for bootstrap | Done |
| Demonstrate statistical uncertainty rather than only averages | CIs on every CAR (conventional and HAC), CAAR bands, cluster-bootstrap strategy intervals, classifier accuracy intervals | Done |
| Evaluate whether a pattern could support a trading hypothesis after basic costs | `src/trading.py`, `src/sensitivity.py`: 3-session hypothetical rule, 0–30 bps per side, development vs holdout | Done: no edge is established |

## Separating evidence from cause

Every panel and doc labels results as associations. A significant CAR shows that a move was unusual for the market model, not why it happened. The main reasons attribution stays uncertain are listed in the README: overlapping events, shared episodes, uncertain release times, and a small, hand-picked sample.

## Known gaps (documented, not hidden)

1. Release sessions are sourced for 19 of 21 events (docs/release-time-verification.md); exact minutes are mostly bounds, and two earnings releases remain unresolved with timing alternatives exported.
2. Several event classes have one or two events, so they get no interval. Sector rows mostly show single stocks.
3. The headline corpus is drawn from search-result titles with one annotator.
4. Yahoo Finance data are third-party, and the frozen snapshot artifact expires on 2026-11-05.

## Submission acceptance checklist

- [x] Price levels are aligned to the benchmark session calendar before calculating returns; missing prices are not filled.
- [x] Snapshot event-eligibility diagnostics use observed benchmark sessions.
- [x] Unit tests pass (149 tests).
- [x] Latest historical rebuild (run 41011220268) passes, and its manifest and integrity report (78 checks) are inspected; outputs are committed in `results/reproduced/`.
- [x] Unresolved timestamps remain flagged, and timing alternatives are exported.
- [x] Dashboard tables equal the shipped CSVs.
- [x] Significance, size, robustness and causal interpretation are kept apart.
- [x] Holdout metrics are reported with uncertainty and no claim of an edge.
- [x] Every PS-03 basic, advanced and brownie item is mapped above.
