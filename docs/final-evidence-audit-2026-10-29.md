# EventEcho final evidence audit — 2026-10-29

## Purpose and status

This note audits the successful historical rebuild artifact from GitHub Actions run **40848349737**, commit `e89bf416445435a3cd0c651a61d0f44a79a31793`. It is an exploratory research/demo snapshot, not evidence of a deployable trading edge.

**Use the run artifact as the current release source of truth.** The older committed files under `results/reproduced/` have a 2026-10-27 manifest with 9 events and a different stock/return setup; do not cite those old CSVs as the 2026-10-29 results unless they are regenerated from the frozen snapshot.

## Frozen data and provenance

- Snapshot: `eventecho-historical-40848349737`
- Provider: Yahoo Finance via yfinance 1.7.0
- Requested interval: 2016-01-01 inclusive through 2026-01-01 exclusive; observed through 2025-12-31
- Universe: 7 stocks plus NIFTY 50 price index
- Benchmark sessions: 2,464; each configured stock has 2,464 benchmark-aligned observations
- Snapshot status: complete; no provider failures or missing required ticker labels; six benchmark observations are missing in the provider response/calendar diagnostic.
- Price convention: `Close` for stocks and NIFTY 50. Dividends are excluded; this is not a total-shareholder-return comparison.
- Prices SHA-256 from run manifest: `4d23f6b004e94bcaf0b21f6485f226d44845c8a5c28995de67999126d308c5bf`
- Source independence: source manifest says market data were not independently verified by a second vendor.

## Coverage and statistical audit

- Events: 21; stocks: 7; CAR rows: 441 (21 × 7 × 3 event windows).
- Event windows: 11 sessions for CAR windows -5:+5, -3:+3 and -1:+1.
- Estimation-window sensitivity: 60, 100 and 150 sessions; 20-session gap before the event window.
- The configured ±5-session CAR windows contain 8 overlapping event pairs involving 8 event IDs: EVT007, EVT010, EVT011, EVT012, EVT013, EVT017, EVT018 and EVT019.
- The broader `event_master.csv` `overlap_flag` has 9 True, 9 False and 3 unset values. It is not the same field as the configured CAR-window overlap audit: EVT020 and EVT021 also carry master-level flags because broader window/dependence review includes volatility-window considerations. Preserve the distinction; do not report the master flag count as the number of CAR-window-overlap events.
- Timestamp registry: 2 exact clock timestamps recorded (the GDP releases EVT018 and EVT021 at 17:30 IST); 19 events still require timestamp review. Do not infer exact times from stock-price movements.
- At the 100-session specification, Holm-adjusted conventional tests mark 8/441 CAR rows significant; HAC-based Holm-adjusted tests mark 10/441. Seven rows remain HAC-significant in all three estimation-window specifications and have stable CAR direction. These are selected event-stock results and do not establish causation.
- Sensitivity changes the count of conventional Holm-significant rows to 12 (60 sessions), 8 (100) and 11 (150); HAC counts are 14, 10 and 10. CAR correlation with the 100-session baseline is high (0.982 for 60; 0.994 for 150), but individual effect sizes and significance can change.
- Category CAR summaries: 24 of 27 category/window rows are labelled insufficient independent episodes; the remaining 3 are descriptive monetary-policy summaries. Do not rank categories as if they were well-powered population estimates.

## Strategy results (exploratory)

The hypothetical rule uses a 3-session holding period and 10 basis points transaction cost per side.

| Metric | Result |
|---|---:|
| All-event mean net return | -0.0459% |
| All-event mean excess return vs NIFTY 50 | -0.2178 percentage points |
| Mean excess return at 30 bps per side | -0.6178 percentage points |
| Development sample mean excess (9 events) | -0.6568 percentage points |
| Holdout mean excess (12 events) | +0.1115 percentage points |
| Holdout 95% cluster-bootstrap CI for mean excess | -0.2629 to +0.3781 percentage points |

The holdout interval includes zero; a robust trading edge is **not established**. Cluster counts are small (8 dependency clusters in holdout), so the bootstrap interval itself is exploratory.

## Release acceptance checklist

- [x] Latest successful historical-rebuild artifact downloaded and inspected.
- [x] Snapshot provenance, ticker coverage, return convention and checksum checked.
- [x] 21-event / 7-stock universe and 441 CAR rows confirmed.
- [x] Overlap flags, timestamp status, estimation-window sensitivity and category sample adequacy reviewed.
- [x] Strategy result and holdout uncertainty checked.
- [ ] Update or clearly quarantine stale committed `results/reproduced/` CSVs so they cannot be mistaken for the fresh artifact.
- [ ] For every event, retain primary-source evidence and explicitly separate date verification from exact publication-time verification.
- [ ] Re-run the final dashboard and report from the same frozen snapshot; include snapshot ID/checksum in the UI/report.
- [ ] Before presenting event-specific “findings”, review timestamp-sensitive and overlapping cases, and present them as associations rather than causal effects.

## Competition-safe claims

**Supported:** EventEcho builds an event-study pipeline, estimates OLS market-model abnormal returns, reports CAR and uncertainty, checks overlap/timing limitations, compares estimation-window specifications, and provides an exploratory chronological holdout for a fixed hypothetical strategy.

**Not supported:** “The strategy beats the market”; “this event caused the observed move”; broad category rankings; or treating all 441 tests as independent discoveries.

## Reproduction

The workflow artifact from run 40848349737 contains the frozen snapshot, manifest, generated result tables, and `eventecho_demo.html`. Prefer rerunning the analysis offline from that snapshot rather than fetching fresh Yahoo data when reproducing this exact result set.
