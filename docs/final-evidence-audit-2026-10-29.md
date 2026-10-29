# EventEcho final evidence audit — 2026-10-29

## Status

This audit covers the committed outputs in [`results/reproduced/`](../results/reproduced/). They were produced by GitHub Actions **Historical Rebuild run 40865554350** on branch `release/final-competition-2026-10-29`, code commit `7ba2166`, by re-analysing offline the frozen price snapshot `eventecho-historical-40848349737` (downloaded by run 40848349737 on 2026-10-29 09:17 UTC). The run's own integrity report is [`results/reproduced/release_validation.json`](../results/reproduced/release_validation.json).

The result set is an **exploratory** event study and a **hypothetical** strategy check. It is not evidence of a trading edge and not evidence that the events caused the price moves.

Every number below was read from the committed CSVs of that run. They agree with the figures recorded for run 40848349737 (the earlier rebuild of the same snapshot).

## What was fixed for this release

| Problem found | Fix | How it is checked |
|---|---|---|
| The snapshot `prices.csv` is the union of all provider dates. The NIFTY 50 column is empty on 6 stock-only dates (2,470 union dates vs 2,464 benchmark sessions). The analysis treated those empty rows as sessions, so its calendar did not match the validation calendar used for event eligibility. | Analysis calendar = observed NIFTY 50 closes only (`src/data.py`). Manifest records `benchmark_sessions_used` = 2,464. | `tests/test_release_integrity.py`; manifest field. For this snapshot the headline figures are unchanged by the fix. |
| Run manifest did not record the price fields it used; text was hard-coded. | Manifest records `price_fields`; a snapshot with any policy other than Close/Close is refused. | Tests; `verify_release.py`. |
| Dashboard said stocks used Adjusted Close while NIFTY used Close (stale). | Statement is built from the manifest. Current text: both sides use Close; price returns; not total shareholder return. | Tests; headless render check. |
| Dashboard JavaScript turned empty cells into 0 (`Number(null) === 0`), so "insufficient episode" category rows showed 0.00%. | Missing values stay missing and show as "—". | Test + render check. |
| Status badges printed as raw `<span>` text; stray literal `\n` text between sections. | Fixed. | Test + render check. |
| PR #36 test failed (it looked for text that only exists after JavaScript runs). | Convention note is rendered on the server side, so it is in the HTML. | Test suite. |
| Committed `results/` and `results/reproduced/` held 2026-10-27 outputs (9 events, 5 stocks). | Legacy files moved to `results/legacy_2026-10-27/`; `results/reproduced/` now holds this run only. | Manifest row counts. |
| Overlap terms were mixed: the dashboard showed the manual registry flag as "overlap". | Computed CAR-window overlap, computed volatility-window overlap, and the manual registry flag are shown separately. | Test. |

## Data and provenance

- Snapshot `eventecho-historical-40848349737`; Yahoo Finance via yfinance 1.7.0; requested 2016-01-01 to 2026-01-01 (end exclusive); observed 2016-01-04 to 2025-12-31.
- Price SHA-256: `4d23f6b004e94bcaf0b21f6485f226d44845c8a5c28995de67999126d308c5bf`. All snapshot file checksums were re-verified before analysis.
- Stocks: RELIANCE, TCS, INFOSYS, HDFC_BANK, ITC, ADANI_ENTERPRISES, ADANI_PORTS. Benchmark: NIFTY 50 price index (^NSEI).
- **Price convention:** `Close` for stocks and for NIFTY 50. These are **price returns**. Dividends are excluded on both sides, so this is **not total shareholder return**.
- Calendar: 2,464 observed NIFTY 50 sessions; no stock price is missing on any of them; no price is filled.
- The checksum proves these are the same downloaded bytes. It does not prove that the provider's prices are correct, and a later download may differ. Market data were not checked against a second vendor.
- Raw prices are not committed (provider terms). The snapshot lives in the run artifact, which expires on 2026-11-05. After that, a fresh rebuild downloads a new snapshot with a new checksum.

## Coverage

| Output | Rows | Note |
|---|---:|---|
| Events | 21 | 9 development, 12 holdout |
| Daily event panel | 1,617 | 21 × 7 × 11 sessions; no missing session |
| CAR panel | 441 | 21 × 7 × 3 windows; all 441 complete |
| Significance tests | 441 | one per complete CAR |
| Estimation-window sensitivity | 1,323 | 3 lengths × 441, same keys |
| Trades | 147 | 21 events × 7 stocks |

## Event timing (from `event_master.csv` of this run)

- Exact release clock time recorded and marked verified: **2 of 21** (EVT018 and EVT021, GDP releases at 17:30 IST, mapped to the next session).
- Time context only, not a release timestamp: 2 (EVT011 management call time; EVT012 board approval time on a Saturday).
- Date only or qualitative time: 8. No time recorded: 9.
- So **19 of 21 events still need timestamp review.** No timestamp was inferred from price moves.
- Timing sensitivity matters. For EVT019 (Hindenburg report), moving Day 0 from 2023-01-25 to 2023-01-24 changes the CAR(−1,+1) of Adani Enterprises from −18.8% to −2.6%, and of Adani Ports from −20.3% to −7.3%. For EVT013 (Reliance results), same-day vs next-session CAR(−1,+1) is +5.2% vs +6.9%. Event-specific claims for these events depend on the timing assumption.

## Overlap and dependence (computed from this run)

- **CAR-window overlap (±5 sessions):** 8 event pairs, 8 events: EVT007–EVT019; EVT010, EVT011, EVT012, EVT013 (all pairs among the April 2025 earnings); EVT017–EVT018. EVT011 and EVT012 share the same Day 0 (2025-04-21).
- **Volatility-window overlap (−20:−1 / +1:+20):** 14 events.
- **Manual registry `overlap_flag`:** 9 True, 9 False, 3 unset. This is a broader review flag. It is **not** the same as computed overlap (for example EVT007 is unset in the registry but its CAR window overlaps EVT019; EVT020 and EVT021 are True because of volatility-window concerns only).
- Overlapping events are excluded from category summaries and joined into dependency clusters for the strategy bootstrap: 6 clusters in development, 8 in holdout. EVT007 (development) and EVT008 (holdout) share episode `RBI_POLICY_2023`, so the two samples are not fully independent.

## Statistical results (100-session baseline)

- Holm-adjusted across all 441 tests: **8** conventional and **10** HAC tests are significant at 5%.
- Across 60/100/150-session estimation windows: conventional 12 / 8 / 11; HAC 14 / 10 / 10. CAR correlation with the 100-session baseline: 0.983 (60) and 0.994 (150).
- **7** event-stock-window results stay HAC-Holm significant in all three specifications with the same CAR sign. All 7 carry an attribution caveat:
  - 6 are Adani stocks around **EVT019** (Hindenburg report) or **EVT007** (RBI hike on 2023-02-08, whose window overlaps the Hindenburg aftermath). The EVT007 result should not be read as an RBI effect.
  - 1 is ITC CAR(−5,+5) around **EVT002** (COVID lockdown), a period of extreme market moves.
- Category CAR summaries: 24 of 27 rows have too few independent episodes for a number. The 3 numeric rows are Monetary Policy (3 episodes, 5 events), with mean event CAR between −0.17% and −0.02%, which is close to zero. Categories cannot be ranked.
- Realized volatility (−20:−1 vs +1:+20, √252): median post/pre ratio by event ranges from 0.58 to 1.48. This is descriptive and not proof that the event caused the change.

## Hypothetical strategy (3-session hold, 10 bps per side)

Rule: buy each stock at the close of the first session after Day 0, sell 3 sessions later at the close, equal-weight within each event, charge 10 bps on entry and 10 bps on exit. Benchmark: NIFTY 50 over the same sessions.

| Metric | Result |
|---|---:|
| All 21 events: mean net return | −0.046% |
| All 21 events: mean excess vs NIFTY 50 | −0.218 pp |
| Mean net return at 0 / 10 / 30 bps per side | +0.154% / −0.046% / −0.446% |
| Mean excess at 0 / 10 / 30 bps per side | −0.018 / −0.218 / −0.618 pp |
| Development (9 events, 6 clusters): mean net / mean excess | −1.834% / −0.657 pp |
| Development 95% cluster-bootstrap CI for mean excess | −1.97 to +0.83 pp |
| Holdout (12 events, 8 clusters): mean net / mean excess | +1.295% / +0.112 pp |
| **Holdout 95% cluster-bootstrap CI for mean excess** | **−0.26 to +0.38 pp (includes zero)** |
| Holdout share of events beating NIFTY 50 | 50% (95% CI 12.5% to 71.4%) |

Mean excess return is negative at every cost level on the full sample, and the holdout interval includes zero. **A trading edge is not established.** Bootstrap uses 5,000 replicates (seed 2026) over few clusters, so the intervals themselves are rough.

## Validation performed

| Check | Result |
|---|---|
| Unit and regression tests (`python -m pytest -q`) | 117 passed locally (Python 3.13); also run inside the CI rebuild (Python 3.11) |
| `scripts/verify_release.py` on the committed outputs | 47 of 47 checks passed (in CI and again locally) |
| Dashboard JavaScript | syntax check passed; headless run over all 126 filter combinations with no script errors, no "NaN"/"undefined", no raw HTML text |
| Dashboard layout | screenshots at 1440 px and 390 px wide; no horizontal page overflow |
| Dashboard ↔ CSV | embedded manifest and every embedded table equal the shipped files |
| Run-to-run repeatability | two CI runs on the same snapshot (40864534684 and 40865554350) gave identical text fields and counts; numeric values differ by at most about 1e-14 (floating-point rounding) |

## Release checklist

- [x] Latest outputs come from one verified run and are committed with their manifest and integrity report.
- [x] Price convention is Close/Close in code, snapshot, manifest, docs and dashboard.
- [x] Analysis calendar matches the benchmark-session validation calendar.
- [x] Stale 2026-10-27 outputs quarantined.
- [x] Overlap kinds and timestamp status shown separately and honestly.
- [x] Holdout uncertainty shown, with a warning because the interval includes zero.
- [ ] 19 events still lack a verified exact release time. This is a documented limitation, not something this release can fix without new primary-source evidence.
- [ ] The frozen snapshot artifact expires on 2026-11-05. Download and archive it privately if exact re-analysis is needed later.

## Claims that are safe to make

**Supported:** EventEcho runs a reproducible OLS market-model event study on a frozen snapshot, reports CARs with conventional and HAC uncertainty and Holm correction, checks estimation-window robustness, audits overlap and timing, and evaluates a fixed hypothetical rule on a chronological holdout with dependence-aware intervals.

**Not supported:** "the strategy beats the market"; "this event caused this move"; "RBI hikes move Adani stocks"; ranking event categories; treating the 441 tests as independent findings.
