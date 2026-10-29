# Problem-statement coverage matrix

This matrix maps the EventEcho problem statement to the current repository implementation. “Implemented” means a corresponding pipeline or output exists; it does not mean every result is fully verified or supports causal interpretation.

| Problem-statement requirement | Repository evidence | Status | Remaining acceptance check |
|---|---|---|---|
| Corporate earnings, policy changes, economic releases, major industry events | `results/event_master.csv`; `src/config.py`; event registry validation | Implemented with selected 21-event registry | Verify source/date/time fields; this is not a representative event universe. |
| Actual stock returns vs benchmark | `src/data.py`; `src/event_study.py`; `multi_stock_event_study.csv` output | Implemented; shared-session alignment required | Align price levels to benchmark sessions before calculating returns and define strategy holding periods on that same calendar; never pair returns spanning different intervals. Explain adjusted-stock-close vs Nifty price-close dividend mismatch. |
| Expected returns and abnormal returns | `src/event_study.py`; market-model outputs | Implemented | Keep OLS one-factor assumptions explicit. |
| CAR across multiple event windows | `src/config.py`; CAR outputs for (−1,+1), (−3,+3), (−5,+5) | Implemented | Check completeness and timing sensitivity for each important event. |
| Statistical significance and uncertainty | `src/statistics.py`; HAC/Newey–West inference; Holm correction | Implemented, approximate | Do not imply exact finite-sample correction; mention multiplicity and dependence. |
| Pre/post volatility around events | `src/volatility.py`; event-volatility output | Implemented | Interpret volatility changes as descriptive and audit overlapping volatility windows. |
| Compare stocks and event categories | `src/reporting.py`; cross-stock and category summaries | Implemented | Category summaries have few independent episodes; earnings peer responses are not direct issuer effects. |
| Account for overlapping events/confounds | `src/event_audit.py`; overlap and episode audits; clustered inference | Partially addressed | Audit manually high-impact overlaps; dependence adjustment does not identify which event caused a move. |
| Robustness to estimation-window choices | `src/estimation_sensitivity.py`; 60/100/150-session outputs | Implemented | Explain significance changes and CAR differences; do not select the most favorable specification. |
| Unseen historical validation | Explicit development/holdout event IDs in `src/config.py`; `src/events.py`; strategy inference outputs | Implemented with limitations | Audit leakage/peeking in analysis choices; small dependent holdout means no reliable edge claim. |
| Optional ML-based event categorization | No validated event-text classifier required by the current core pipeline | Optional / not a blocker | Add only with labelled data, chronological evaluation, and evidence it improves decisions. |
| Reproducible run and reviewable results | `scripts/run_analysis.py`; `src/snapshots.py`; manifest and CI workflows | Implemented with data-source limitation | Use the latest successful workflow artifact; distinguish frozen snapshots from live Yahoo Finance data. |
| Explain evidence vs causality | Dashboard, robustness ledger, README limitations | Implemented as guardrails | Final presentation must avoid causal wording and clearly show timing/source confidence. |

## High-priority gaps

1. **Event metadata reliability:** exact release timestamps remain unresolved for several RBI decisions, Hindenburg, and some earnings. See [event source and validation audit](event-source-and-validation-audit.md).
2. **Timing robustness:** daily-close prices cannot isolate intraday announcement effects; same-session/next-session sensitivity must be reviewed for uncertain events.
3. **Holdout interpretation:** a positive point estimate is not sufficient when the dependence-aware confidence interval crosses zero.
4. **Data provenance:** Yahoo Finance can revise historical prices; a repeatable workflow is not independent data verification.
5. **Final narrative:** show the small sample, dependent episodes, selected stock universe, overlapping events, dividend convention mismatch, and trading-simulation simplifications.

## Submission acceptance checklist

- [ ] Price levels are aligned to the benchmark session calendar before calculating returns and strategy holding windows; missing stock prices are not filled or bridged.
- [ ] Latest unit tests pass.
- [ ] Latest historical rebuild passes and the latest artifact/manifest are inspected.
- [ ] Event-source audit is reflected in the event master without inventing timestamps.
- [ ] Timing sensitivities are reviewed for unresolved and intraday events.
- [ ] Dashboard tables and figures correspond to the latest run and current 21-event/seven-stock universe.
- [ ] Statistical significance, economic magnitude, robustness, and causal interpretation are clearly distinguished.
- [ ] Holdout metrics are reported with uncertainty and no claim of a proven profitable edge.
- [ ] Demo can be explained end-to-end: event registry → trading-session mapping → market model → AR/CAR → inference → robustness/overlap → chronological strategy check.
