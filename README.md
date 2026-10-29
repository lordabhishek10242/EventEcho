# EventEcho

EventEcho is an exploratory quantitative event-study system for examining stock returns around selected Indian policy, macroeconomic, corporate-earnings, election, and company-specific industry events. The question is whether observed stock returns around these dates differ from a simple market-model expectation, and whether a fixed, hypothetical post-event holding rule showed consistent results in a small chronological holdout.

This repository includes a modular Python implementation, the original research notebook, a new notebook interface, archived CSV outputs supplied with the project, and a fresh run in `results/reproduced/`. The legacy outputs have **not** been regenerated from raw prices. No historical price snapshot was included in the archive; the fresh run used current Yahoo Finance history and is not a frozen-vendor reproduction.

## Competition readiness audits

- [Problem-statement coverage matrix](docs/problem-statement-coverage.md) maps each requirement to code/output evidence and defines final acceptance checks.
- [Event-source and validation audit](docs/event-source-and-validation-audit.md) distinguishes verified event dates from unresolved timestamps, documents daily-close timing limitations, and records the remaining high-priority checks.
- [Event timing, overlap, and return-convention policy](docs/event-timing-overlap-return-convention.md) defines timestamp evidence labels, overlap treatment, and the consistent price-return convention.

These audits deliberately preserve unresolved timestamps as unresolved rather than inferring them from observed stock-price moves.

## Research Universe

- Stocks: Reliance Industries (`RELIANCE.NS`), TCS (`TCS.NS`), Infosys (`INFY.NS`), HDFC Bank (`HDFCBANK.NS`), ITC (`ITC.NS`), Adani Enterprises (`ADANIENT.NS`), and Adani Ports (`ADANIPORTS.NS`).
- Benchmark: Nifty 50 (`^NSEI`) close, a **price index**, not a total-return index.
- Intended price period: 2016-01-01 through 2025-12-31 (provider end date is exclusive); actual dates depend on data availability.
- Events: 21 selected records in the expanded event master spanning policy/tax reform, monetary policy, corporate earnings, election results, GDP release, and a company-specific industry event. This remains a small, selected sample rather than a representative event population. Records are selected by event relevance and source availability, not by observed return direction.
- The current acquisition uses Yahoo Finance `Close` for both stocks and the Nifty 50 `Close` price index. This consistent price-return convention excludes dividends on both sides; it is not a total-return comparison. Historical runs generated before this change used stock `Adj Close` and are superseded for comparisons involving the return convention.

Event dates, effective dates, source links, episode IDs, timing confidence, and notes are recorded in `results/event_master.csv`. For corporate-earnings events, `affected_stocks` identifies the focal issuer; the analysis still reports the configured seven-stock cross-section to inspect possible peer/sector spillovers. Those peer responses must not be described as direct issuer effects. Five FY2025 corporate earnings records have been added from company-primary sources, alongside GST rollout, the 2019 and 2024 Lok Sabha result dates, the Q4 FY2023-24 GDP release, and the Hindenburg Research report on Adani. TCS explicitly schedules its earnings release after trading hours; Infosys provides an after-close management-event schedule on 2025-04-17; Good Friday (2025-04-18) means its next regular session is 2025-04-21; HDFC Bank's board approval occurred on a Saturday. For Reliance and ITC the cited source confirms the release date but not the exact publication time, so next-session alignment is provisional and must be tested against same-day alignment before making event-specific claims. A source URL and recorded verification status are provenance fields, not independent verification by this refactor. In particular, daily prices cannot identify the intraday reaction to announcements during market hours.

## Audit Summary

The original notebook has 143 cells and contains data acquisition, returns, event definitions/alignment, event-specific OLS models, abnormal returns, CAR inference, a COVID-period sensitivity check, cross-stock analysis, trading simulation, chronological evaluation, and Kaggle-path artifact export. Its cells are not executed in the supplied notebook metadata. The attached artifact archive contains 11 result CSVs and a manifest, but no raw historical prices. The modular pipeline was successfully run on 2026-10-27 into `results/reproduced/`; it fetched the same 2016-01-04 to 2025-12-31 sample dates from Yahoo Finance.

The downloadable analysis is a workflow artifact tied to a specific run and snapshot manifest. Older artifacts may reflect earlier event/stock universes; always use the latest successful workflow artifact and its manifest when citing counts or metrics. Registry or methodology changes on main trigger a fresh snapshot acquisition and rebuild. The single-stock file is a Reliance-focused export; the current multi-stock analysis is generated from the 21-event registry and seven-stock universe after the latest successful rebuild.

Implementation risks addressed in the reusable code include per-stock missing-value handling, explicit estimation-window completeness, complete-window CAR reporting, trading-session-based event windows, matching benchmark holding periods, and parameter uncertainty in the approximate CAR standard error. The original notebook used a 100-session training window ending at Day -20. Since the event window begins on Day -5, that leaves only 14 omitted sessions (Days -19 through -6), not a 20-session gap. The modular implementation uses the specified 20 omitted sessions (Days -25 through -6); legacy CAR estimates are therefore not method-identical. Further limitations are listed below.

## Architecture

```text
src/config.py          Parameters, ticker universe, explicit event split
src/data.py            Local/Yahoo prices, validation, simple returns
src/events.py          Event validation, session alignment, sample labels
src/event_study.py     Estimation samples, OLS market model, AR and CAR
src/statistics.py      Approximate CAR inference and Holm correction
src/trading.py         Long-only trades, costs, benchmark, event aggregation
src/sensitivity.py     Transaction-cost, extreme-market, and event-timing sensitivities
src/volatility.py      Pre/post realized and residual volatility diagnostics
src/category_analysis.py Episode-level CAR and volatility summaries with overlap exclusions
src/reporting.py       CSV exports, dashboard datasets, and run manifest
scripts/run_analysis.py  Repository-root command-line workflow
tests/                 Network-free synthetic tests
notebooks/EventEcho.ipynb  Modular results exploration
notebook/notebook5b4a79ed78.ipynb  Preserved original notebook
results/               Supplied legacy CSV exports and provenance note
results/reproduced/    Fresh Yahoo Finance run from the modular CLI
```

## Full historical rebuild

Run the end-to-end acquisition, analysis, and demo workflow locally with:

```bash
python scripts/rebuild_historical.py
```

This downloads the configured daily price history, freezes a uniquely named snapshot with raw responses, provenance and checksums, runs the analysis offline from that snapshot, and builds `results/reproduced/eventecho_demo.html`. The analysis will refuse to treat an incomplete acquisition as a successful historical snapshot. The command needs network access for acquisition; the resulting snapshot can then be reused offline with `scripts/run_analysis.py --snapshot-id <snapshot-id> --offline`.

The **Historical Rebuild** GitHub Actions workflow runs this process on relevant changes to `main` or manually via Actions → Historical Rebuild → Run workflow. It uploads the frozen snapshot, result tables, manifest and dashboard as a seven-day artifact. Each run has a unique snapshot ID so a prior snapshot is not silently overwritten.

## Installation and Execution

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

Default run: load the preserved event master and fetch market data through `yfinance` (network access required). Outputs go to `results/reproduced/`; the supplied legacy files in `results/` are not overwritten:

```bash
python scripts/run_analysis.py --verbose
```

For reproducible/offline input, provide a wide CSV with a `date` column, one column per stock (`RELIANCE`, `TCS`, `INFOSYS`, `HDFC_BANK`, `ITC`, `ADANI_ENTERPRISES`, `ADANI_PORTS`), and `NIFTY50` benchmark close. Missing cells may be blank. The input file is ignored by Git because it may be large or subject to provider terms.

```bash
mkdir -p data
python scripts/run_analysis.py --prices data/prices.csv --offline --output-dir results/fresh
```

The CLI refuses offline runs without local prices, fails clearly if an explicitly supplied input path is missing or no usable stock/benchmark data is available, and writes a run-specific `manifest.json`. Repository inputs are recorded as repository-relative paths; external input paths are recorded by filename to avoid machine-specific paths. Network calls are not made by tests.

The new `notebooks/EventEcho.ipynb` displays the supplied archived tables for review. The original notebook is retained unchanged at `notebook/notebook5b4a79ed78.ipynb`; its hardcoded `/kaggle/working` export location applies only to its original execution environment.

## Robustness evidence ledger

`robustness_evidence_ledger.csv` consolidates the 60/100/150-session sensitivity results at the event-stock-window level. It distinguishes HAC-Holm significance that survives all three estimation lengths from findings that appear only under some specifications. It also carries configured event-window overlap, shared-episode, and timing/source review flags so statistical stability is not mistaken for clean attribution. These flags are not a complete confounder census, and stable significance is not causal proof.

## Estimation-window robustness

The run also exports `estimation_window_sensitivity.csv` and `estimation_window_sensitivity_summary.csv`. The market model is re-estimated with 60, 100, and 150 pre-event sessions, holding the event gap and event windows fixed. The dashboard reports CAR correlation and absolute CAR changes versus the 100-session baseline, plus conventional and HAC Holm-significance counts for each specification. Holm correction is applied within each specification; the comparison is a pre-specified robustness check, not a search for the most favorable p-value.

## Dependence-aware strategy uncertainty

`clustered_strategy_inference.csv` reports percentile intervals from a cluster bootstrap for development and chronological holdout strategy metrics. Events are placed in the same dependence cluster when their event-study windows overlap or the registry assigns the same `episode_id`; whole connected clusters are resampled rather than individual events. This avoids treating known overlapping windows or explicitly shared episodes as independent bootstrap units.

The intervals are still exploratory: the event universe is small and selected, clusters can be few or uneven, and this procedure does not establish causality or guarantee out-of-sample profitability. Read it alongside `sample_performance.csv`, overlap audits, transaction-cost sensitivity, and the frozen snapshot manifest. Do not treat a positive point estimate as evidence of a reliable edge when its uncertainty is wide or the holdout contains few independent clusters.

## Methodology

For close price $P_t$, simple return is $R_t = P_t/P_{t-1} - 1$. The pipeline first reindexes stock price levels to the benchmark's observed trading-session calendar, then computes stock and benchmark returns over the same consecutive sessions. Event windows and hypothetical trade entry/exit sessions also use that common calendar. No missing prices are filled: if a stock price is missing on a benchmark session, the stock return for that session and the next session remains unavailable rather than bridging the gap. A missing value for one stock does not globally discard another stock's observation.

For each event and stock, the market model is fit on 100 complete trading-session observations ending at Day -26. The 20 omitted sessions are Days -25 through -6, immediately before the event window begins on Day -5:

$$R_{i,t} = \alpha_i + \beta_i R_{m,t} + \epsilon_{i,t}.$$

Expected return is $\widehat{R}_{i,t} = \hat{\alpha}_i + \hat{\beta}_i R_{m,t}$ and abnormal return is $AR_{i,t} = R_{i,t} - \widehat{R}_{i,t}$. CAR is the sum of daily abnormal returns within the selected session window. Windows are (−1,+1), (−3,+3), and (−5,+5). A CAR is missing rather than silently partial if any required event-window observation is unavailable.

The effective date is the designated Day 0 from the event master; an effective date that is not an available trading session is an error in the analysis path rather than silently shifted. Announcement times need separate verification. The supplied event table includes episode identifiers and overlap flags, but users should inspect shared episodes and overlapping announcements before treating tests as independent.

Realized volatility is measured separately from market-model residual volatility. Stock and benchmark realized volatility use trading-session windows −20:−1 and +1:+20, excluding Day 0, with annualization by √252. Category summaries average stock responses within an event and related events within an episode; overlapping windows are excluded. Numeric category summaries require at least three independent episodes.

## Inference and Sensitivity

The reusable implementation reports two approximate two-sided CAR inference variants: (1) the conventional variance estimate using estimation residual variance plus OLS parameter-estimation variance, and (2) a Newey–West/Bartlett HAC sensitivity estimate using long-run estimation-residual variance and HAC sandwich covariance for market-model coefficients. Each p-value family is Holm-adjusted across all generated event × stock × window tests. The HAC variant is a robustness sensitivity, not an exact finite-sample event-study test: it does not fully account for cross-event dependence, event-induced variance, or overlapping windows. Statistical significance is not economic significance or causality.

The original notebook logs COVID-period confounders and compares CAR after excluding event sessions where the absolute Nifty return exceeds 5%. Such exclusions are diagnostic only and can remove genuine event-period response. The modular pipeline exports this extreme-day sensitivity when EVT002 is present. The original notebook also compares market-model and market-adjusted expected returns; the latter assumes alpha zero and beta one and is not inherently superior.

## Trading Simulation

The hypothetical strategy is long-only and enters at the close of the first trading session strictly after the effective event date. It exits at the close three sessions after entry. Per-event portfolios equally weight eligible stock trades. Costs are a fixed number of basis points per side, deducted on entry and exit as a simple return haircut. The benchmark return compounds Nifty daily returns over the same dates after entry through exit.

The sensitivity levels are 0, 5, 10, 20, and 30 bps per side. The original split is retained explicitly from the supplied manifest: development `EVT001`–`EVT005` plus `EVT007`; holdout `EVT008`, `EVT009`, and `EVT006`. Each event is the observation for sample-level comparisons, not each stock trade.

This simulation assumes close-price execution and omits market impact, liquidity constraints, slippage beyond the fixed cost haircut, taxes, corporate-action cashflows, and position overlap. It is not a strategy backtest suitable for investment decisions.

## Supplied Legacy Findings

The legacy figures below are reported by the supplied `results/` CSVs and embedded archive manifest. They are retained as originally supplied and were not regenerated from raw price data:

- Multi-stock sample: 9 events, 5 stocks, 45 event-stock combinations, 135 CAR-window estimates and 135 approximate tests.
- The legacy significance CSV marks 4 of 135 tests as significant after Holm correction at 5%. The simplified test and selected events limit interpretation.
- At 10 bps per side, legacy mean event-level net return is −0.655%, mean excess return versus the Nifty 50 is +0.305 percentage points, and event-level positive-return rate is 55.56% (9 events).
- At 30 bps per side, legacy mean excess return is −0.095 percentage points.
- Development (6 events): mean net return −1.237%, mean excess return +0.655 percentage points.
- Holdout (3 events): mean net return +0.508%, mean excess return −0.397 percentage points.

The legacy outputs do not show consistent out-of-sample benchmark outperformance. These figures are included for artifact traceability; the original notebook's outputs and archive have not been independently validated against their original raw data.

## Historical baseline run (2026-10-27; superseded)

> **Historical traceability note:** The figures in this subsection are from an earlier 9-event/5-stock baseline run. They are not the current expanded result set. For the current 21-event/seven-stock run, use the [latest successful historical rebuild workflow](https://github.com/lordabhishek10242/EventEcho/actions/runs/40837512955) and its artifact/manifest. Do not combine the counts or metrics across runs.


The modular CLI successfully downloaded market history through Yahoo Finance and generated 12 non-empty artifacts under `results/reproduced/` for 9 events, 5 stocks, and dates 2016-01-04 through 2025-12-31. This is a fresh calculation using provider data available at run time, not an exact reproduction from a frozen price snapshot.

- 495 daily stock-event observations, 135 CAR estimates, and 135 approximate tests were generated; no CAR window was partial.
- 1 of 135 tests remained significant at 5% after Holm correction under the corrected 20-session gap and parameter-aware standard error: EVT002 / ITC / CAR(-5,+5), 22.63%, adjusted p = 0.0176. This is exploratory and subject to COVID-period confounding and the small selected sample.
- Trading results were consistent with the legacy exports to rounding: 10 bps per side produced -0.655% mean net return and +0.305 percentage points mean excess return across 9 events.
- Development (6 events): -1.237% mean net return and +0.655 percentage points mean excess return.
- Holdout (3 events): +0.508% mean net return and -0.397 percentage points mean excess return.
- At 30 bps per side, mean excess return was -0.095 percentage points.

The fresh significance count differs from the legacy 4/135: the modular calculation uses the literal 20-session gap before the event window and adds OLS parameter-estimation uncertainty to the standard error; the legacy code used a shorter effective gap and a residual-only standard error. Yahoo Finance data can also be revised. This is a methodology/data-version difference, not evidence that the underlying archived output was corrupted.

## Data, Licensing, and Provenance

Yahoo Finance data is accessed through the third-party `yfinance` Python package. This is an unofficial interface; users must review the data provider's current terms and restrictions and not redistribute data without permission. Event source links are stored in the event table; attribution and source terms should be checked for any public redistribution.

No project license was assigned because the intended license was not provided. The source code and supplied result artifacts therefore have no new license grant; decide and add one before reuse or contribution under open-source terms. No raw price data or credentials are included.

`results/README.md` documents the legacy artifact source and row counts. `results/manifest.json` is the supplied historical manifest. Fresh run manifests identify their inputs, dates, methods, split, and outputs.

## Limitations and Future Work

- Nine hand-selected events are too few and heterogeneous for generalization; episode-level dependence reduces effective sample size.
- Survivorship and selection bias are possible: the seven-stock universe is fixed retrospectively and not a point-in-time index constituent universe.
- Event dates/timing and source descriptions require independent primary-source review. Market-hours events need intraday prices for clean reaction timing.
- Stock Close and benchmark price-index Close both exclude dividends; this is a price-return comparison, not a total-return comparison.
- The market model is a one-factor OLS model; robust inference and event-induced volatility methods would be useful improvements.
- Overlapping event windows should be pre-specified, flagged, and addressed in inference rather than considered independent tests.
- Execution, corporate actions, taxes, and realistic portfolio sizing are absent from the trading simulation.
- Future work should use frozen, licensed price snapshots, a larger pre-specified event universe, a point-in-time stock universe, and robust event-study inference.

## Disclaimer

EventEcho is an educational research project, not investment advice. Observational event-study associations do not prove causal effects or future profitability.