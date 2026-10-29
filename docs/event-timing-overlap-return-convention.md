# Event timing, overlap, and return-convention policy

## Timestamp evidence: do not manufacture precision

The generated event_master.csv adds:

- event_timestamp_ist: normalized ISO-8601 timestamp with +05:30 only when the registry has a clock time explicitly marked IST and its source-status field claims verification; otherwise null.
- timestamp_semantics and timestamp_evidence_status: distinguish a missing clock time, qualitative/date-only timing, a time that is only contextual (for example, a management call or board approval), and a recorded clock time whose source status claims verification.
- timestamp_review_required: true unless the registry records a clock time with an explicit verified/confirmed status and the time is not marked as context-only.
- event_timing_class: broad session-timing category for review, not a substitute for an exact timestamp.
- inference_scope: records that the current selected 21-event set is exploratory and was not pre-registered as a confirmatory sample.

These are audit labels, not independent verification. Do not infer missing times from the price chart or observed CAR. For events with unknown publication time, preserve same-session/next-session sensitivity and avoid causal language. If the source reports a management call or board approval time, do not silently relabel it as the exact public-results release timestamp.

## Overlapping windows

The configured CAR window is ±5 sessions; volatility windows are also audited. event_window_audit.csv now includes overlap_treatment and event_window_overlap_detected so overlap handling is visible in the artifact:

- Overlapping events stay in the full event × stock × window Holm family; the adjusted p-values are exploratory and do not remove dependence.
- Overlapping events are excluded from category-level summaries.
- The strategy cluster bootstrap resamples connected components formed by overlapping CAR windows or shared episode_id.
- A non-overlapping window means only that no other registered event window intersects it; unregistered confounders may still exist.

Do not claim that clustering identifies which event caused a price move. The 21-event registry is small and selected, and the effective independent sample can be much smaller than the event count.

## Dividend convention

The default Yahoo Finance acquisition now uses Close for both stocks and the Nifty 50 benchmark, rather than stock Adj Close against index Close. This is a consistent price-return convention: dividends are excluded on both sides. It is not a total-return comparison and must not be described as total shareholder return. The benchmark remains a price index. A future total-return analysis should use a reliable Nifty 50 TRI series and stock total-return data from a compatible, documented source.

For local CSV inputs, the pipeline cannot infer whether columns are raw close, split-adjusted close, dividend-adjusted close, or TRI. The user must label the source and convention in the provenance sidecar (`adjustment_policy.stock_field` / `benchmark_field`); the pipeline preserves the supplied values and does not silently transform them. The run manifest records the fields actually used in `price_fields`. A sidecar or snapshot that records any field other than Close/Close is refused, and an input that records no fields is reported as "not recorded" rather than labelled Close. The dashboard's convention statement is generated from this manifest record.

## Session calendar

The wide snapshot `prices.csv` is the union of every provider's dates, so the NIFTY 50 column can be empty on a date that only stocks returned. The analysis calendar is the set of observed NIFTY 50 sessions (non-missing close); empty benchmark rows are not sessions. Stock prices are reindexed to this calendar before returns are computed, so a stock return and a benchmark return on the same row always cover the same interval. This matches the snapshot validation calendar used for event eligibility. The number of sessions used is recorded as `benchmark_sessions_used` in the run manifest.

## Acceptance checks

- Verify all exact timestamps against the linked source before promoting an event from exploratory status.
- Report same-session/next-session sensitivity where exact timing is unresolved or intraday.
- Keep overlap flags in result tables; do not treat the 441 event-stock-window tests as independent observations.
- Keep the chronological holdout untouched when choosing events, windows, or model settings.
- Do not add RNN/LSTM to compensate for unresolved timing, confounding, or return-convention problems.
