# EventEcho

**Do Indian large-cap stocks move differently around big events — and how sure can we be?**

A reproducible, bias-audited **event study** on 7 NSE large caps and NIFTY 50, 2016–2025, from **one frozen price snapshot**: OLS market model, abnormal returns, three event windows, Holm-corrected significance, BMP / Kolari–Pynnönen / rank robustness, overlap and timing audits, a 266-event quarterly-results panel with a placebo, an NLP headline classifier, and a cost-inclusive hypothetical trading rule on a chronological holdout.

> **Submitted to:** QUANTACTIVE — CODE CONFLUX · Problem Statement **PS-03** · Business Strategy and Analytics Club (BSAC), MNNIT Allahabad

**The headline is deliberately uncomfortable:** the project's strongest, best-replicated finding is that *company results days move a stock ~2.7× as much as normal* — a statement about **size**, not direction. The trading rule shows **no edge**, and the macro-event average is **not distinguishable from zero**. Everything below separates what is demonstrated from what is merely plausible.

---

## Also in this repository: Invoice Action Agent (A2A 1.0)

A separate, dependency-free service that reads invoice claim batches, proposes
**one typed business action per package** with the three decisive evidence
references, gates every execution on the reviewer's `ACCEPTED` result, and stores
task state so retries can never repeat a payment.

- Agent Card: `https://8000-i08mehv473n5yap8hpkdj.e2b.app/.well-known/agent-card.json`
- A2A base URL: `https://8000-i08mehv473n5yap8hpkdj.e2b.app/a2a/`
- Code: `invoice_agent/` · Docs: [`docs/INVOICE_AGENT.md`](docs/INVOICE_AGENT.md)
- Verify: `python3 scripts/verify_invoice_agent.py` · Tests:
  `python3 -m unittest tests.test_invoice_agent`

The agent study below is unchanged and independent of the service.

---

## Headline numbers (all from one verified run)

| Metric | Value | Meaning |
|---|---|---|
| **2.68×** | results-day variance ratio, 95% CI **2.17–3.23** | 266 quarterly results days move the stock far more than an ordinary day |
| **0.88×** | the *same* measure on placebo days, CI **0.70–1.08** | the effect is not just "stocks are jumpy" |
| **8 / 441** | macro CAR tests significant after Holm correction | of 441 tests; ~22 would pass at 5% by chance |
| **−0.12%** | average macro CAR(−1,+1), CI −0.53% to +0.49% | BMP / KP / rank tests all agree: **p = 0.92 / 0.93 / 0.61** |
| **−0.26 to +0.38 pp** | holdout 95% interval for the trading rule's excess return | **includes zero → no trading edge claimed** |
| **81%** | headline classifier accuracy on 21 unseen registry headlines | vs **29%** majority-class baseline |
| **155 / 87** | network-free unit tests / release integrity checks | run in CI on every push, re-verified locally |

**Verification status:** results come from Historical Rebuild run **41019176924** (code commit `7f3ea88`) on frozen snapshot `eventecho-historical-40848349737`, re-analysis of which is bit-identical apart from floating-point rounding. `pytest` → **155 passed**. `verify_release.py` → **87 passed, 0 failed**.

---

## 1. Problem understanding

### The core problem

Financial news arrives as discrete, dated events. The naive question — "did the stock go up on the event?" — is unanswerable as evidence, because on any given day the whole market moves too. The real question is:

> **After removing the move explained by the market, was the stock's reaction around the event large enough and unusual enough to be distinguishable from noise — and does that survive multiple-testing correction, alternative model settings, overlapping events, uncertain release timing, and genuinely unseen data?**

Three things make this hard in the Indian context specifically:

1. **Few independent events.** Big macro events are rare; several of ours fall in the same week or are driven by one episode (the 2022 rate-hike cycle, the Adani/Hindenburg episode). 21 events are **not** 21 independent observations.
2. **Uncertain release timing.** A daily close cannot tell you whether news landed at 10:20 IST (intraday) or 20:57 IST (after close). Getting Day 0 wrong by one session can flip a conclusion — for the Hindenburg report, Adani Enterprises' CAR(−1,+1) is **−18.8%** with one Day 0 and **−2.6%** with the alternative.
3. **No direction without surprise.** Without analyst-consensus beat/miss data, good and bad results cancel out. So for the 266 results days we deliberately study the **magnitude** of the reaction, not its sign.

**Real-world relevance:** this is the exact machinery a quant desk, risk team, or event-driven fund uses to ask whether a signal exists before risking capital — and the discipline to conclude "not yet" is the deliverable.

### Assumptions (stated, not hidden)

| # | Assumption | Consequence if violated |
|---|---|---|
| A1 | A one-factor market model (α + β·NIFTY) captures "normal" returns | Sector/factor news is attributed to the event; AR is overstated |
| A2 | Event windows are short enough that β is stable across them | β drift biases AR in the window direction |
| A3 | Day 0 = first session that *could* react | Timing error displaces the reaction (quantified in §3) |
| A4 | Daily closes are sufficient to characterise the reaction | Intraday reversal/overreaction is invisible |
| A5 | 100-session estimation window is representative of pre-event normal | Pre-event regime dominates β (tested at 60/100/150) |
| A6 | Fills at the closing price with fixed bps costs | Real impact, taxes, and liquidity make results optimistic |

### Constraints prioritised

Correctness of inference **over** feature count; reproducibility **over** data volume; honest uncertainty **over** a clean-looking story. The architecture follows from this: freeze the data → analyse offline → verify the shipped artefacts → then talk.

### Why this methodology (and what we rejected)

| Choice | Why | Rejected alternative and why |
|---|---|---|
| Event study with a market model | Standard, auditable, isolates stock-specific reaction from market-wide moves | Raw returns (confounded by market); CAPM/Fama-French multi-factor (NIFTY has no clean free factor set here — noted as future work) |
| **Both** conventional and HAC t-tests | Events raise volatility → plain t is too eager; HAC handles residual serial correlation | HAC alone is **not** a fix for overlap; we keep both visible rather than picking the friendlier one |
| Holm across all 441 tests | Family-wise error control over the whole grid; honest about exploratory status | No correction (≈22 false positives); BH/FDR would be defensible but more permissive — not chosen |
| BMP / Kolari–Pynnönen / rank tests | Directly target event-induced variance, same-day cross-correlation, and outliers | Relying on one parametric t-test (defeated by exactly these effects) |
| Cluster bootstrap (whole clusters) | Preserves dependence from overlap and shared episodes | Event-level bootstrap (intervals falsely narrow); no interval at all |
| Chronological dev/holdout split | Only credible defence against overfitting | Random split (leaks the future into the past) |
| **Variance ratio** for results days | Direction is unmeasurable without surprise data | Signed mean CAR (reported, p = 0.13 — correctly null) |

---

## 2. Quantitative correctness

### Return and abnormal-return definitions

For stock *i* on session *t*, with benchmark return *R*<sub>m,t</sub>:

```
Estimation (OLS, 100 sessions, ending 20 sessions before the window):
    R_it = α_i + β_i · R_mt + ε_it

Abnormal return:        AR_it = R_it − (α̂_i + β̂_i · R_mt)
Cumulative abnormal:    CAR_i(τ1,τ2) = Σ_{t=τ1..τ2} AR_it
Average reaction:       CAAR(τ) = (1/N) Σ_i CAR_i(τ), accumulated from Day −5
Results-day measure:    Variance ratio = mean(SCAR_i²),  SCAR_i = CAR_i / SE(CAR_i)
```

`AR` and `CAR` are in **%**; differences between returns (excess return, CAAR changes) are in **percentage points (pp)** — kept distinct throughout the code and the dashboard.

### Windows, and why three of them

Estimation **−125…−26**, gap **−25…−6**, event windows **±1 (3 sessions), ±3 (7), ±5 (11)**. A CAR is left *missing* if any session in it is missing — never silently filled. The gap exists so that pre-event leakage/run-up does not contaminate the estimate of "normal".

The window choice is treated as a **result**, not a setting: HAC-Holm significant counts are **4 / 1 / 5** for ±1/±3/±5, and about **a third of event-stock pairs flip the CAR sign** between ±1 and ±5. Any claim resting on one window is labelled fragile.

### Inference stack (and what each layer defends against)

| Layer | Formula / setting | Defends against | Where it bites |
|---|---|---|---|
| t-test with parameter uncertainty | `SE² = L·σ² + xᵀVx`, `t = CAR/SE` | Treating α̂, β̂ as known | Baseline p-values per CAR |
| Newey–West (HAC) | Bartlett weights, bandwidth `4(n/100)^(2/9)` | Serial correlation in residuals | 10 vs 8 significant |
| Holm | `p̃(k) = max_{j≤k} ((m−j+1)·p(j))`, m = 441 | ~22 lucky hits across 441 tests | 8 survive |
| BMP | `t = √N · mean(SCAR)/sd(SCAR)` | Event-day volatility inflation | p = 0.92 |
| Kolari–Pynnönen | `t_KP = t_BMP · √((1−r̄)/(1+(N−1)r̄))` | Stocks moving together on the same day | p = 0.93 |
| Corrado/Cowan rank | `U = rank/(T+1) − ½`, `z = Ū/√(L·S_U)` | Outliers / non-normal returns | p = 0.61 |
| Cluster bootstrap | 5,000 draws over whole clusters, seed 2026, 2.5/97.5 percentiles | Overlapping events treated as independent | 13 clusters from 21 events |
| Variance ratio | `mean(SCAR²) ≈ 1.02` under the null | Missing a *size* effect when direction is unknown | 2.68 vs 0.88 placebo |

**Guards that are easy to get wrong, and are handled explicitly:**

- **Insignificant ≠ no effect.** Absence of a signed average is reported as such, not as evidence of nothing.
- **Significant ≠ causal.** Every output is labelled an association.
- **Tests are not independent.** Holm is applied (conservative under dependence) and the bootstrap resamples clusters. Both are described as exploratory.
- **Intervals are withheld** for any group with **< 3 dependency clusters** rather than printing a meaningless band.

### Execution, costs and constraints (trading module)

Buy at the **close of the first session after Day 0**, hold **3 sessions**, sell at the close, equal weight within an event, **10 bps per side** base with a **0 / 5 / 10 / 20 / 30 bps** grid. Mean excess vs NIFTY 50 at 0/10/30 bps is **−0.02 / −0.22 / −0.62 pp** — negative at every cost level on the full sample.

The result is **dominated by costs, not by the signal**, which is exactly the point of including them. Excluded by design and stated: market impact, bid–ask spread beyond the bps charge, STT/brokerage/taxes, position limits, borrow, and liquidity constraints on size.

### Why the results-day measure is the variance ratio

Without surprise data, direction is unknowable: mean CAR(−1,+1) is **−0.28%**, median −0.17%, 46% positive, BMP p = 0.13. Reporting that as "results are bad news" would be wrong. The variance ratio asks the answerable question — *did the event change the amount of movement?* — and lands at **2.68 [2.17, 3.23]** against **0.88 [0.70, 1.08]** on placebo days.

---

## 3. Data & validation

### Sources and periods

| Data | Detail | Source |
|---|---|---|
| Prices | 7 stocks + `^NSEI`, **2,464 NIFTY 50 sessions**, observed 2016-01-04 → 2025-12-31 | Yahoo Finance via `yfinance`, frozen |
| Macro events | **21** dated events with source URL, episode ID, timing confidence | Hand-curated, each linked (`results/event_master.csv`) |
| Earnings dates | **266** = 7 stocks × 38 quarters (Q1 FY17 – Q2 FY26) | 222 board-meeting histories (Trendlyne), 43 dated news reports, 1 NSE filing |
| Headlines | **374** labelled, sourced headlines in 49 search groups, 6 classes | Web search titles, labelled by the team |
| Benchmark | NIFTY 50 **price** index | Same snapshot |

**Universe:** RELIANCE, TCS, INFOSYS, HDFC_BANK, ITC, ADANI_ENTERPRISES, ADANI_PORTS — large and liquid, but **selected after the fact**, across 5 project-defined sector groupings (`STOCK_SECTORS`), not an official classification.

### Preprocessing rules (all enforced in code, not by convention)

1. **Calendar** = sessions with a non-missing NIFTY 50 close. Stock prices are **re-indexed** to it; missing prices are **never** forward-filled or interpolated.
2. **Price convention** = `Close` for stocks **and** `Close` for NIFTY 50 → **price returns**, dividends excluded on both sides. This is **not total shareholder return**, and mixing an adjusted stock series with a price index would bias abnormal returns around dividend dates. The manifest records the fields; `run_analysis.py` **refuses** a snapshot on any other basis.
3. **Day 0** = first session on/after the effective date. Only source-supported clock times are recorded; **no timestamp is ever inferred from a price move.**
4. **CARs are all-or-nothing**: 441 of 441 complete in the current run.
5. **Earnings calendar validation**: the builder refuses a missing or duplicated quarter; a test requires consecutive announcements to be **25–150 days** apart. Three real defects were found and fixed (a missing HDFC Bank quarter, a Sunday date, a duplicated Reliance date).

### Bias controls — explicit answers

| Bias | Control | Evidence |
|---|---|---|
| **Look-ahead** | Day 0 uses only the announcement date; estimation ends 20 sessions before the window; no post-event information enters the model | Timing audit; `tests/test_event_windows.py` |
| **Survivorship** | Universe is fixed *a priori* for the panel; no delisted-name backfill is possible in a 7-name set — and this is a **limitation**, not a solved problem | Stated in §9 |
| **Overfitting / data snooping** | Chronological dev (≤ Feb 2023, 9 events) vs holdout (≥ Oct 2023, 12 events); three earnings follow-ups **pre-registered in the module docstring before the run**; results-day design fixed before seeing outcomes | `src/earnings_extensions.py`; PR #39 body |
| **Confounding / overlap** | CAR-window and volatility-window overlap computed on sessions; 8 events share windows; shared episodes merged into **13 dependency clusters**; overlapping events excluded from category summaries; attribution caveats in the ledger | `event_pair_overlap_audit.csv`, `event_level_audit.csv` |
| **Multiple testing** | Holm over the entire 441-test grid; published counts under 3 estimation lengths and 3 windows | `multi_stock_significance.csv`, `estimation_window_sensitivity*.csv` |
| **Wrong-day risk** | 19/21 release *sessions* verified against timed sources; all 19 agree with Day 0; 2 unresolved events get exported alternative Day 0 | [release-time-verification.md](docs/release-time-verification.md) |
| **"Just fat tails"** | Placebo pseudo-events **+30 sessions** (mid-quarter), fixed before running, never tuned | 0.88 vs 2.68, non-overlapping intervals |
| **Fishing on follow-ups** | Null-drift simulation on 20 synthetic price sets: test fires **1 of 20**, i.e. calibrated near 5% | `scripts/simulate_drift_null.py` |

### Data provenance & reproducibility

- Raw provider responses + **SHA-256** per file. Prices SHA-256: `4d23f6b0…c5bf`. Snapshot ID recorded in the manifest along with the code commit.
- Three independent CI runs on the same snapshot produced identical text fields and counts, with numeric differences ≤ ~1e-14.
- **Synthetic data disclosure:** the only synthetic data are the 20 no-drift price paths used *solely* to calibrate the drift test. Every reported financial number comes from real historical market data. There is no simulated or projected result in the results tables.

---

## 4. Engineering & implementation

### Pipeline

```
Acquire            Freeze              Analyse              Dashboard          Verify              Publish
yfinance      →  snapshot +      →  run_analysis.py   →  build_demo.py   →  verify_release   →  commit +
(raw + SHA)      SHA-256, CI        offline, no          one self-         87 checks, CI        PR, artifact
                 artifact           network              contained HTML     fails on drift
```

The design rule: **the demo never recomputes a number.** `build_demo.py` embeds the verified CSVs and the manifest into a single 3.1 MB HTML file, and `verify_release.py` asserts table-by-table equality between the dashboard payload and the shipped files.

### Architecture

| Layer | Modules | Responsibility |
|---|---|---|
| Data | `data.py`, `snapshots.py` | snapshot loading, checksums, calendar alignment, returns, refusal of non-Close/Close data |
| Event registry | `events.py`, `event_audit.py` | schema validation, Day 0 alignment, timestamp-evidence classification, overlap & confounder audit, sample split integrity |
| Core study | `event_study.py`, `statistics.py`, `robust_tests.py` | OLS market model, AR/CAR, t/HAC, Holm, BMP, KP, rank, variance ratio |
| Aggregation | `pooled_analysis.py`, `category_analysis.py`, `clustered_inference.py`, `estimation_sensitivity.py`, `sensitivity.py` | CAAR by class/sector/stock, window choice, cluster bootstrap, 60/150-session re-fits |
| Extensions | `earnings_study.py`, `earnings_extensions.py`, `headline_classifier.py`, `volatility.py`, `trading.py` | 266 results days + placebo, drift/peer/trend follow-ups, NLP classifier, realised volatility, hypothetical execution |
| Reporting | `reporting.py`, `robustness_ledger.py` | offline dashboard, key findings computed from the run's own tables, evidence ledger |
| Entry points | `scripts/run_analysis.py`, `build_demo.py`, `verify_release.py`, `rebuild_historical.py`, `acquire_market_snapshot.py`, `build_earnings_calendar.py`, `simulate_drift_null.py` | CLI surface |

**Size:** `src/` **5,009 lines** across 21 modules · `scripts/` 1,281 lines · `tests/` 2,858 lines in 26 files · 45 result CSVs.

### Edge-case handling (not decorative — each has a test)

| Case | Behaviour |
|---|---|
| Missing price on a session | CAR set to missing; never filled |
| Snapshot not on Close/Close basis | Analysis **refuses to run** |
| Stock-only dates with empty NIFTY column | Excluded from the calendar (this was a real bug, fixed and regression-tested) |
| Duplicate / missing event ID, empty source fields | Registry validation raises |
| Dev/holdout IDs overlapping or not chronologically separated | Split builder raises |
| Group with < 3 clusters | No interval printed (shows "—") |
| Dashboard null cell | Rendered as "—", never coerced to `0` (a real JS bug, fixed) |
| Earnings quarter missing/duplicated | Calendar builder refuses to write |
| Announcement spacing outside 25–150 days | Test fails |
| Snapshot fails a checksum | Pipeline fails loudly; artefacts still uploaded for diagnosis |

### Testing & CI

- **155 network-free tests** (`python -m pytest -q`) covering returns, calendar alignment, event windows, overlap treatment, Holm recomputation, clustering, trading arithmetic, classifier leakage, earnings gaps, reporting, and release integrity.
- **87 release checks** (`python scripts/verify_release.py`) including: CAR = Σ AR, Holm p-values recomputed over all 441 tests, dashboard tables == shipped CSVs, variance ratio recomputed from SCARs, split ordering, cost arithmetic, overlap symmetry, headline-corpus checksum.
- **3 GitHub Actions workflows**: Tests (push/PR), Historical Rebuild (fresh or reused snapshot → analysis → dashboard → verify, 7-day artifact), Recover frozen market snapshot (hard-fails on incomplete acquisition).
- Dashboard is headless-tested across **126 filter combinations** for script errors, `NaN`/`undefined` leakage, raw HTML text, and layout overflow at 1440 px and 390 px.

---

## 5. Innovation & results

### Demonstrated (measured, with uncertainty, in the shipped run)

| # | Finding | Evidence | Baseline it beats |
|---|---|---|---|
| 1 | **Results days move a stock 2.68× normal**, CI 2.17–3.23 | 266 announcements | Placebo days **0.88 [0.70–1.08]** — intervals do not overlap |
| 2 | The effect **peaks on Day +1** (4.77 mean squared SCAR vs 1.77 on Day 0) | Event-time panel | Naive "Day 0 is the reaction" assumption; consistent with after-close/weekend reporting |
| 3 | It **replicates out of sample**: 2.15 (≤2022) and 3.82 (2023–25), both intervals above 1 | 182 dev / 84 holdout | Majority of single-sample event studies |
| 4 | **No directed macro effect**: BMP 0.92, KP 0.93, rank 0.61 | 21 events, 13 clusters | A single t-test, which would over-reject here |
| 5 | **No trading edge**: holdout excess −0.26 to +0.38 pp | 12 holdout events, 8 clusters | Cost-free backtests; the rule is negative at every cost level |
| 6 | **Classifier generalises** to 21 never-trained headlines at 81% (CI 62–95%) | Grouped CV 75% | Majority class 29%; TF-IDF+LR beats keyword rules *in the blind comparison* |
| 7 | **Results-day reactions have grown** ~+0.22 variance-ratio units/year beyond placebo (CI +0.03 to +0.41) | 2016–2025 | Placebo trend −0.01 (flat) — **labelled moderate**, 2 post-hoc checks disclosed |

### Also demonstrated — negative results reported as results

- **No post-earnings drift** in this sample: up-minus-down CAR(+2,+20) = **+0.44 pp [−1.19, +2.13]**, test calibrated at 1/20 false fires on no-drift data.
- **No peer spillover beyond normal co-movement**: every results-day correlation interval contains its ordinary-day level.
- **Macro replication is weak**: class-sign prediction beats a coin flip in 4/5, 3/5, 2/5 cases (binomial p ≥ 0.38).

### What is *not* claimed (theoretical or projected only)

- Any causal statement ("the RBI hike moved Adani"). The 7 results that survive all three model settings **all** carry an attribution caveat: 6 are Adani stocks around the Hindenburg report or the overlapping RBI window; 1 is ITC around the COVID lockdown.
- Any sector ranking (IT 3.54 vs Banking 3.08 — intervals overlap), any event-class ranking (most classes have 1–2 episodes), any directional earnings signal.
- Any live-trading performance. The strategy is a **hypothetical simulation**.
- A projection that results-day volatility will keep rising. That is a trend observed in-sample, only just significant.

### Why the complexity is justified

Every added layer answers a specific objection a quant reviewer would raise: BMP/KP/rank answer "your t-test assumes constant volatility and independent stocks"; the cluster bootstrap answers "your intervals assume independent events"; the placebo answers "stocks are just jumpy"; the variance ratio answers "you can't measure direction without surprises"; the pre-registration answers "you fished for a result"; the holdout answers "you fit the sample". Complexity that did *not* buy an answer was not added — and the dashboard exposes the nulls next to the hits.

---

## 6. Presentation & technical defence

- **Deck:** [`docs/EventEcho_Presentation_v3.1.pdf`](docs/EventEcho_Presentation_v3.1.pdf) — problem → data → method → statistics → macro results → results days → holdout/strategy/NLP → code & audit → dashboard. 10 slides, every slide traceable to a CSV.
- **Judge Q&A rehearsal:** [`docs/judge-qa.md`](docs/judge-qa.md) — 25 anticipated questions with short answers, plus a "things never to say" list.
- **Coverage matrix:** [`docs/problem-statement-coverage.md`](docs/problem-statement-coverage.md) — every PS-03 basic, advanced and brownie item mapped to *code file → output file → dashboard panel*.
- **Method notes:** [event timing / overlap / return convention](docs/event-timing-overlap-return-convention.md) · [event source audit](docs/event-source-and-validation-audit.md) · [release-time verification](docs/release-time-verification.md) · [earnings & robust tests](docs/earnings-and-robust-tests.md) · [headline classifier](docs/headline-classifier.md) · [final evidence audit](docs/final-evidence-audit-2026-10-29.md).
- **Defence stance:** the team defends *decisions*, and the strongest defence in the room is the list of things we refuse to claim.

---

## 7. EventEcho problem-specific judging checks

The rubric's EventEcho line, item by item:

| Check | Our answer | Evidence |
|---|---|---|
| **Abnormal returns and benchmark** | OLS market model vs NIFTY 50; AR = actual − (α̂ + β̂·NIFTY); Close/Close on both sides so the benchmark is not a dividend-inclusive series | `multi_stock_event_study.csv`, `event_study.py` |
| **Suitable event and estimation windows** | Estimation 100 sessions (−125…−26), 20-session gap, windows ±1/±3/±5; sensitivity re-fit at 60 and 150 (CAR correlation 0.983 / 0.994); window choice reported as a result (4/1/5 significant) | `estimation_window_sensitivity*.csv`, `window_choice_*.csv` |
| **Statistical significance** | Conventional t + HAC, Holm over all 441; BMP, Kolari–Pynnönen, rank tests; variance ratio for the earnings panel; bootstrap intervals | `multi_stock_significance.csv`, `robust_tests_registry.csv`, `earnings_summary.csv` |
| **Confounding / overlapping events** | 8 events share CAR windows; shared episodes merged → 13 dependency clusters; overlapping events excluded from category summaries; per-event confounder ledger with documented and unverified flags | `event_pair_overlap_audit.csv`, `event_level_audit.csv`, `robustness_evidence_ledger.csv` |
| **Limits of causal claims** | Every panel and doc labels results as association; the 7 robust survivors are individually caveated; timing sensitivity shows CARs changing materially under alternative Day 0 | §9, [final evidence audit](docs/final-evidence-audit-2026-10-29.md) |

Also delivered beyond the minimum: a **second, cleaner sample** (266 results days chosen by rule, not by hand), a **placebo** to define the null empirically, a **pre-registered** follow-up set with 2 of 3 nulls reported, and an **NLP classifier** used purely as an independent check on hand-assigned categories — it never touches a return.

---

## 8. Submission, status, originality

### Feature status — implemented / partial / planned

| Feature | Status | Note |
|---|---|---|
| Frozen snapshot, SHA-256 provenance, offline rebuild | ✅ Implemented | 87 checks; reproducible to ~1e-14 |
| Market model, AR, CAR (±1/±3/±5), CAAR | ✅ Implemented | 441/441 complete |
| t, HAC, Holm; BMP, KP, rank; variance ratio | ✅ Implemented | Every CAAR group tested |
| Estimation-window & event-window sensitivity | ✅ Implemented | 60/100/150; 3 windows |
| Overlap, shared-episode clustering, confounder ledger | ✅ Implemented | 13 clusters |
| Release-session verification | ✅ Implemented (19/21) | 2 unresolved, alternatives exported |
| 266-event results-day study + placebo + dev/holdout | ✅ Implemented | 2.68 vs 0.88 |
| Drift / peer spillover / volatility-trend follow-ups | ✅ Implemented (pre-registered) | 1 of 3 positive, labelled moderate |
| NLP headline classifier + baselines | ✅ Implemented | 81% vs 29% |
| Hypothetical trading rule with cost grid + cluster CI | ✅ Implemented | No edge claimed |
| Offline single-file dashboard | ✅ Implemented | 3.1 MB, 126 filter combinations tested |
| CI: tests, rebuild, snapshot recovery | ✅ Implemented | 3 workflows |
| Sector / event-class **rankings** | ⚠️ Partial | Single-stock sectors, 1–2 event classes → descriptive only |
| Exact intraday release timestamps | ⚠️ Partial | Session-level for 19/21; minutes mostly bounds |
| Total-return comparison | ⚠️ Partial | Price returns only (benchmark is a price index) |
| Second-vendor price validation | ⚠️ Partial | Not done; disclosed |
| Directional earnings signal (beat/miss) | ❌ Planned | Needs analyst-consensus data |
| Exchange-filing timestamps for all 266 dates | ❌ Planned | Would tighten Day 0 |
| Intraday event check for RBI/Budget days | ❌ Planned | Would separate same-day vs next-day |
| Multi-factor / GARCH expected-return model | ❌ Planned | One-factor model currently |
| Live/paper trading | ❌ Not planned | Out of scope; no edge to trade |

**Submitted code matches the demo.** The dashboard embeds this run's manifest and tables; `verify_release.py` fails CI if any table diverges from the CSVs, so a stale demo cannot ship.

### Credits

- **Libraries:** `numpy`, `pandas`, `scipy`, `scikit-learn` (TF-IDF, logistic regression), `yfinance`, `pytest`. No other code was copied; the event-study, inference and bootstrap layers are project code.
- **Data:** Yahoo Finance (prices, third-party, terms apply); Trendlyne board-meeting histories and dated news reports (earnings dates); NSE filing; public news sources for macro-event release times — every row carries its `source_url` in `results/event_master.csv`, `results/earnings_events.csv` and `results/headline_corpus.csv`.
- **Method references:** MacKinlay (1997) event-study framework; Newey–West (1987); Holm (1979); Boehmer–Musumeci–Poulsen (1991); Kolari–Pynnönen (2010); Corrado (1989) / Cowan (1992).
- **Raw prices are not redistributed** here (provider terms); only derived results and the checksummed manifest are committed.

### AI-assisted development disclosure

Parts of this repository were built with an AI coding assistant (Claude, via Claude Code) used for implementation, refactoring, test-writing and documentation drafting, under human direction. All methodological choices, data curation judgements, and reported numbers were reviewed and re-verified by the team through the 155-test suite and the 87-check release verifier, and every number in this README was read back from the committed CSVs of run 41019176924. Where a claim could not be supported by that run, it is labelled partial or planned above rather than asserted.

---

## 9. Limitations, failure cases, and what we do NOT claim

### Limitations

1. **Small, selected samples.** 21 hand-picked macro events (9 dev / 12 holdout) and 7 large caps chosen after the fact; most event classes have 1–2 episodes and most sectors contain one stock.
2. **Dependence.** 13 clusters from 21 events; the dev and holdout samples are **not fully independent** (they share the `RBI_POLICY_2023` episode). Intervals are built on few clusters and are themselves rough.
3. **Daily closes.** No intraday separation of announcement vs reaction; 2 of 21 events still lack a verified release session, and exact minutes are mostly bounds.
4. **Timing sensitivity is real.** Hindenburg: Adani Enterprises CAR(−1,+1) is −18.8% or −2.6% depending on Day 0; EVT020 Budget flips Reliance from −3.2% to ≈0.0%.
5. **One-factor model.** Event-day volatility change is not modelled inside the expected-return equation (partly addressed by BMP/variance-ratio tests, not by the model itself).
6. **Confounding is unbounded.** The overlap audit covers *registered* events only; unregistered news can contaminate any window.
7. **Price returns only.** Not total shareholder return; dividends excluded on both sides.
8. **No surprise data** → no directional earnings claim; the post-earnings-drift null is a statement of power, not of truth.
9. **Single price vendor**, not cross-checked; Yahoo history can be revised.
10. **Earnings dates** are third-party compiled (12/12 seeded cross-check agreement, but no full exchange-filing audit).
11. **Headline corpus** is search-result titles with a single annotator — not a random news sample.
12. **Strategy is hypothetical**: close-price fills, fixed bps, no impact, taxes, spread, position limits or liquidation constraints.
13. **The frozen snapshot artifact expires 2026-11-05.** Exact re-analysis after that requires the archived snapshot; otherwise a fresh download yields a new checksum.

### Named failure cases

- A conclusion that survives only one window (e.g. an isolated significant CAR at ±5 but not ±1) is treated as **fragile**.
- An event whose Day 0 is unresolved can produce a **sign-flipped** CAR — we export the alternative rather than choose.
- A category with < 3 clusters yields **no interval**, and we print "—" instead of a number.
- A group whose bootstrap interval includes zero (all macro groups, the strategy holdout) is reported as **no established effect**, not as a weak positive.
- The drift test is known to fire ~1 in 20 times on pure noise; one of three follow-ups passing is disclosed as such.

### Safe claims vs unsupported claims

**Supported.** EventEcho runs a reproducible OLS market-model event study on a checksummed frozen snapshot, reports CARs with conventional and HAC uncertainty and Holm correction, adds BMP/KP/rank robustness, quantifies estimation and window sensitivity, audits overlap and release timing, studies 266 rule-selected results days against a placebo with replication, and evaluates a fixed cost-inclusive rule on a chronological holdout with dependence-aware intervals — and it ships an offline dashboard whose tables are byte-verified against those CSVs.

**Not supported.** "The strategy beats the market." "This event caused this move." "RBI hikes move Adani stocks." "IT reacts more than banks." Ranking event categories. Treating the 441 tests as 441 independent findings. Any forward projection.

---

## Quick start

Python **3.10+** recommended (CI uses 3.11).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q                  # 155 tests, no network needed
```

**Open the demo (no server, no internet):** open `results/reproduced/eventecho_demo.html` in any browser — a single self-contained file with the data embedded.

**Verify the committed results:**

```bash
python scripts/verify_release.py --analysis-dir results/reproduced   # 87 checks
```

**Rebuild offline from the frozen snapshot** (place the snapshot folder at `data/snapshots/<snapshot-id>/`, available from the CI run artifact):

```bash
python scripts/run_analysis.py --snapshot-id <snapshot-id> --offline --output-dir results/reproduced
python scripts/build_demo.py --analysis-dir results/reproduced
python scripts/verify_release.py --analysis-dir results/reproduced
```

**Rebuild with fresh prices (needs internet):**

```bash
python scripts/rebuild_historical.py     # downloads → freezes a new snapshot → analyses offline → builds dashboard
```

A fresh download may differ from the frozen snapshot; the manifest and checksums will differ accordingly.

**Rebuild the earnings calendar / calibrate the drift test:**

```bash
python scripts/build_earnings_calendar.py
python scripts/simulate_drift_null.py
```

---

## Repository layout

A full tree with a one-line explanation of **every file** is in [the appendix](#appendix--file-structure-and-what-each-file-means).

---

## How to read the results

- **% vs pp:** returns and CARs are in %, differences between returns in percentage points.
- **Significant ≠ causal.** Overlap, shared episodes and timing uncertainty all survive the statistics.
- **Holm** controls false positives across all 441 tests, but the tests are not independent → results remain **exploratory**.
- **HAC** is a serial-correlation robustness check, not a fix for overlap or event-day volatility (BMP/variance ratio address the latter).
- **Peer stocks in earnings events** show possible spillover, not the issuer's own effect.
- **Uncertainty is shown, not summarised away**: intervals, "—" where intervals are unjustified, and explicit nulls.

---

## Data licence & disclaimer

Market data are from Yahoo Finance via the unofficial `yfinance` package; check the provider's terms before redistributing. Event, earnings and headline sources are linked row-by-row in the input CSVs. No project licence has been chosen yet, so no licence is granted for reuse.

**Educational research project. Not investment advice.** No result here establishes a trading edge, and nothing here should be read as a causal explanation of any price move.

---

## Appendix — file structure and what each file means

Every path in the repository, with its purpose. Line counts are from the submitted commit; test counts are the number of cases `pytest` collects from each file (155 in total).

### Top level

```text
EventEcho/
├── README.md                     this document
├── requirements.txt              pinned runtime deps: numpy, pandas, scipy, yfinance, scikit-learn, pytest
├── .gitignore                    excludes venvs, caches and data/snapshots (raw prices are not redistributed)
├── src/                          reusable analysis library — 5,009 lines, 21 modules
├── scripts/                      command-line entry points — 1,281 lines, 7 scripts
├── tests/                        network-free test suite — 2,858 lines, 26 files, 155 tests
├── results/                      inputs (event registry, earnings dates, headlines) + verified outputs
├── docs/                         audits, method notes, judge Q&A, PS-03 coverage, deck PDF
├── .github/workflows/            3 CI pipelines
├── notebook/                     original submission notebook, kept unchanged for provenance
└── notebooks/                    thin exploration notebook
```

| Path | What it is | Why it exists |
|---|---|---|
| `README.md` | Project documentation | Setup, method, results, limits, file map |
| `requirements.txt` | Dependency list | One command installs everything; no version drift |
| `.gitignore` | Ignore rules | Keeps raw/restricted market data and venvs out of git |
| `notebook/notebook5b4a79ed78.ipynb` | The original notebook (1.4 MB) | Preserved untouched so judges can see the starting point |
| `notebook/EventEcho_artifacts.zip` | Artifacts exported from that notebook | Historical record of the first pass |
| `notebooks/EventEcho.ipynb` | Small exploration notebook | Scratch exploration, no reported numbers |

### `src/` — analysis library (21 modules, 5,009 lines)

```text
src/
├── __init__.py                     package marker
├── config.py                       all settings, tickers, sectors, split IDs
├── data.py                         load prices, align calendar, build returns
├── snapshots.py                    freeze raw data, SHA-256, integrity diagnostics
├── events.py                       event registry validation, Day 0, sample split
├── event_audit.py                  overlap, shared-episode and confounder audit
├── event_study.py                  market model, abnormal returns, CAR windows
├── statistics.py                   t-test, HAC, Holm correction
├── robust_tests.py                 BMP, Kolari–Pynnönen, rank test, variance ratio
├── pooled_analysis.py              CAAR, window choice, replication tests
├── category_analysis.py            category summaries with episodes as the unit
├── clustered_inference.py          cluster bootstrap for strategy inference
├── estimation_sensitivity.py       60 / 100 / 150-session re-fits
├── sensitivity.py                  cost grid and extreme-day sensitivity
├── volatility.py                   realised volatility before vs after each event
├── trading.py                      hypothetical post-event trade simulation
├── earnings_study.py               266 results days, placebo, replication
├── earnings_extensions.py          three pre-registered follow-up questions
├── headline_classifier.py          TF-IDF + logistic regression, baselines
├── reporting.py                    CSV schemas, manifest, HTML dashboard
└── robustness_ledger.py            reviewer-facing caveat ledger
```

| Module | Lines | What it does | Key outputs |
|---|---:|---|---|
| `__init__.py` | 3 | Marks `src` as an importable package | — |
| `config.py` | 79 | Single source of truth: ticker map, sector grouping, category→class map, estimation/gap/window settings, cost levels, holding period, and the fixed development/holdout event IDs. `AnalysisConfig` validates its own arguments (e.g. gap ≥ window radius, non-negative costs) | `manifest.json` blocks |
| `data.py` | 199 | Loads a frozen snapshot or price CSV, **refuses any basis other than Close/Close**, takes the observed-NIFTY session calendar, re-indexes stocks to it and builds close-to-close returns. Never fills a missing price | in-memory panels → `manifest.json` |
| `snapshots.py` | 433 | Downloads provider data, writes raw responses plus per-file SHA-256, and runs integrity diagnostics (session counts, coverage, gaps) before anything is analysed | `snapshot_metadata/`, checksums |
| `events.py` | 120 | Validates the event registry (required columns, unique IDs, non-empty sources), classifies timestamp evidence without inferring times from prices, aligns each event to its first tradeable session, and enforces that development/holdout IDs are disjoint and chronologically ordered | `event_master.csv` (validated) |
| `event_audit.py` | 481 | Computes **CAR-window** and **volatility-window** overlap on trading sessions, finds shared episodes, records documented and unverified confounders, and exposes the timing sensitivity of uncertain events | `event_pair_overlap_audit.csv`, `event_level_audit.csv`, `event_window_audit.csv`, `event_timing_sensitivity.csv` |
| `event_study.py` | 169 | Fits the OLS market model (`α + β·NIFTY`) on 100 complete sessions ending 20 before the window, then computes abnormal returns and CARs for ±1/±3/±5. A CAR is left missing if any session is missing | `multi_stock_event_study.csv`, `multi_stock_car.csv` |
| `statistics.py` | 160 | Parameter-aware CAR standard errors (`SE² = Lσ² + xᵀVx`), the conventional t-test, the Newey–West/HAC version, and Holm correction across all 441 tests | `multi_stock_significance.csv` |
| `robust_tests.py` | 238 | Cross-sectional tests that do not assume constant event-day volatility: BMP standardised test, Kolari–Pynnönen same-day correlation adjustment, Corrado/Cowan rank test, and the event-induced variance ratio with a cluster bootstrap | `robust_tests_registry.csv`, `earnings_summary.csv` |
| `pooled_analysis.py` | 296 | Builds CAAR paths and window summaries from the per-event panels, compares windows (±1/±3/±5, sign and verdict flips), and runs development→holdout replication checks | `caar_event_time.csv`, `caar_window_summary.csv`, `window_choice_*.csv`, `pattern_generalization.csv` |
| `category_analysis.py` | 211 | Descriptive summaries by event class, with **episodes / dependency clusters** as the analysis unit; withholds a number when the unit count is too small | `category_car_summary.csv`, `category_volatility_summary.csv` |
| `clustered_inference.py` | 155 | Cluster bootstrap (5,000 draws, seed 2026) over connected components of overlapping windows and shared episodes, used for every strategy interval | `clustered_strategy_inference.csv`, `sample_performance.csv` |
| `estimation_sensitivity.py` | 77 | Re-fits the market model at 60 and 150 sessions and compares CARs and significance counts with the 100-session baseline | `estimation_window_sensitivity*.csv` |
| `sensitivity.py` | 229 | Transaction-cost grid (0/5/10/20/30 bps per side) and the COVID extreme-day drop test | `transaction_cost_sensitivity.csv`, `covid_extreme_day_sensitivity.csv` |
| `volatility.py` | 254 | Realised volatility in sessions −20:−1 vs +1:+20 (Day 0 excluded), annualised by √252, plus market-model residual volatility | `event_volatility.csv` |
| `trading.py` | 109 | Simulates the rule: buy at the close of the first session after Day 0, hold 3 sessions, sell at the close, equal weight, costs charged on both sides, benchmark measured over the same sessions | `event_trading_results.csv`, `trade_candidates.csv` |
| `earnings_study.py` | 418 | The 266-announcement results-day panel: each company studied only on its own results day; variance ratio as the main measure, placebo at +30 sessions, development/holdout replication, per-stock and per-sector breakouts | `earnings_summary.csv`, `earnings_placebo.csv`, `earnings_replication.csv`, `earnings_event_car.csv`, `earnings_coverage.csv` |
| `earnings_extensions.py` | 300 | Three follow-ups written down **before** the run: post-results drift/reversal, peer spillover, and the yearly volatility trend — reported as they came out (2 nulls, 1 moderate) | `earnings_drift_*.csv`, `earnings_peer_*.csv`, `earnings_volatility_trend.csv` |
| `headline_classifier.py` | 237 | TF-IDF (words 1–2, chars 3–5) + logistic regression for 6 event classes, with a keyword baseline, a majority-class baseline, and group-aware cross-validation so no search group leaks across the split | `headline_classifier_*.csv`, `event_headline_classification.csv` |
| `reporting.py` | 737 | Single writer for every result table (stable column order), the run manifest with provenance, and the **offline single-file HTML dashboard** whose tables are embedded verbatim | all CSVs, `manifest.json`, `eventecho_demo.html` |
| `robustness_ledger.py` | 114 | Turns the audit tables into a per-result caveat ledger, so a reader sees the attribution warning next to the number | `robustness_evidence_ledger.csv` |

### `scripts/` — entry points (7 scripts, 1,281 lines)

| Script | Lines | What it does | When to run it |
|---|---:|---|---|
| `run_analysis.py` | 401 | The orchestrator: loads the snapshot (offline or via download), runs every module above in order, writes an output directory | Main analysis step |
| `build_demo.py` | 31 | Turns one verified analysis directory into the self-contained HTML dashboard | After `run_analysis.py` |
| `verify_release.py` | 392 | 87 invariant checks on an output directory (CAR = Σ AR, Holm p-values recomputed, dashboard tables == shipped CSVs, split order, cost arithmetic, overlap symmetry, checksums). **Not** expected-value checks, so it catches drift without freezing results | Before publishing; runs in CI |
| `rebuild_historical.py` | 72 | One-command full rebuild: download → freeze a new uniquely named snapshot → analyse offline → build dashboard | Fresh end-to-end run |
| `acquire_market_snapshot.py` | 88 | Opt-in acquisition of a frozen price snapshot only (no analysis) | Creating a shareable snapshot |
| `build_earnings_calendar.py` | 263 | Rebuilds `results/earnings_events.csv` from the hand-collected, sourced dates; refuses to write if a quarter is missing or duplicated | Regenerating the results calendar |
| `simulate_drift_null.py` | 35 | Runs the drift test on simulated no-drift price paths — used to show the test is calibrated (~1 in 20 false fires) | Method validation |

### `tests/` — network-free suite (26 files, 2,858 lines, 155 cases)

| Test file | Cases | What it protects |
|---|---:|---|
| `test_release_integrity.py` | 15 | Regression tests for the release fixes: benchmark-session calendar, manifest price policy, dashboard/CSV equality |
| `test_snapshots.py` | 13 | Snapshot writing, checksums, acquisition failures, CLI behaviour |
| `test_returns.py` | 11 | Close-to-close return construction, calendar alignment, no imputation |
| `test_phase1_reports.py` | 11 | Report schemas, row counts and required columns |
| `test_robust_tests.py` | 11 | BMP / KP / rank / variance-ratio numerics |
| `test_statistics.py` | 8 | CAR variance, HAC, Holm correction behaviour |
| `test_trading.py` | 7 | Trade arithmetic, cost application, benchmark alignment |
| `test_earnings_study.py` | 7 | Results-day windowing, placebo exclusion, variance ratio |
| `test_volatility.py` | 6 | Pre/post volatility windows, annualisation, missing-day handling |
| `test_sensitivity.py` | 6 | Cost grid and extreme-day sensitivity |
| `test_reporting.py` | 6 | Dashboard content and embedding |
| `test_event_windows.py` | 6 | Window bounds, gap rules, missing-session behaviour |
| `test_event_relationship_audit.py` | 6 | Overlap symmetry and shared-episode logic |
| `test_event_registry.py` | 6 | Registry validation and required provenance fields |
| `test_earnings_extensions.py` | 6 | Drift, peer and trend follow-up calculations |
| `test_pooled_analysis.py` | 5 | CAAR construction and window-choice agreement |
| `test_data_provenance.py` | 5 | Manifest provenance fields and basis refusal |
| `test_headline_classifier.py` | 4 | No leakage across groups; registry headlines never in training |
| `test_event_timing_metadata.py` | 4 | Timestamp-evidence classification; no price-inferred timing |
| `test_clustered_inference.py` | 4 | Cluster construction and bootstrap determinism |
| `test_estimation_sensitivity.py` | 3 | 60/100/150-session re-fit consistency |
| `test_price_return_convention.py` | 1 | Close/Close price-return basis is enforced |
| `test_overlap_treatment.py` | 1 | Overlapping events excluded from category summaries |
| `test_robustness_ledger.py` | 1 | Caveat ledger covers every result |
| `test_rebuild_script.py` | 1 | Rebuild entry point smoke test |
| `test_snapshot_cli.py` | 1 | Snapshot CLI wiring |

### `results/` — inputs and outputs

**Inputs (hand-curated, committed, each row carries a source URL):**

| File | Rows | What it is |
|---|---:|---|
| `event_master.csv` | 21 | The macro-event registry: date, effective date, headline, category, source, episode ID, timing confidence, release-session check |
| `earnings_events.csv` | 266 | Every quarterly results date, 7 stocks × 38 quarters (Q1 FY17 – Q2 FY26) |
| `headline_corpus.csv` | 374 | Labelled headlines in 49 search groups — training data for the NLP check only |

**Verified outputs (`results/reproduced/`, 45 CSVs + manifest + dashboard):**

| Group | Files | What they contain |
|---|---|---|
| Event study core | `multi_stock_event_study.csv` (1,617), `multi_stock_car.csv` (441), `single_stock_event_study.csv` (231) | Daily ARs per event × stock × session, and CARs for three windows |
| Significance & robustness | `multi_stock_significance.csv` (441), `robust_tests_registry.csv` (66), `robustness_evidence_ledger.csv` (441) | t/HAC/Holm results per test, BMP/KP/rank/variance ratio per group, caveats per result |
| Aggregation | `caar_event_time.csv` (242), `caar_window_summary.csv` (66), `category_car_summary.csv` (27), `category_volatility_summary.csv` (27), `cross_stock_consistency.csv` (63) | CAAR paths and bands, per class/sector/stock/sample summaries |
| Window & model sensitivity | `window_choice_summary.csv` (3), `window_choice_agreement.csv` (3), `estimation_window_sensitivity.csv` (1,323), `estimation_window_sensitivity_summary.csv` (3), `covid_extreme_day_sensitivity.csv` (21) | How verdicts change with the window and with 60/100/150-session models |
| Audit | `event_level_audit.csv` (21), `event_pair_overlap_audit.csv` (210), `event_window_audit.csv` (21), `event_relationship_highlights.csv` (10), `event_timing_sensitivity.csv` (714), `event_volatility.csv` (147) | Overlap, confounders, timing alternatives, volatility before/after |
| Earnings | `earnings_summary.csv` (90), `earnings_placebo.csv` (24), `earnings_replication.csv` (7), `earnings_event_car.csv` (266), `earnings_event_time.csv` (88), `earnings_coverage.csv` (266), `earnings_drift_events.csv` (266), `earnings_drift_summary.csv` (20), `earnings_peer_events.csv` (152), `earnings_peer_spillover.csv` (8), `earnings_volatility_trend.csv` (14) | The 266-event panel, its placebo, replication test and the three pre-registered follow-ups |
| NLP | `headline_classifier_metrics.csv` (6), `headline_classifier_confusion.csv` (24), `headline_classifier_per_class.csv` (6), `event_headline_classification.csv` (21) | Accuracy vs baselines, confusion matrix, per-class scores, headline-level predictions |
| Strategy | `event_trading_results.csv` (21), `trade_candidates.csv` (147), `clustered_strategy_inference.csv` (2), `sample_performance.csv` (2), `final_strategy_assessment.csv` (7), `transaction_cost_sensitivity.csv` (5), `pattern_generalization.csv` (18) | Per-event trades, cluster-bootstrap intervals, cost sensitivity, replication verdicts |
| Provenance | `manifest.json`, `release_validation.json` (87 checks), `snapshot_metadata/` | Code commit, snapshot ID, price SHA-256, settings, integrity results |
| Demo | `eventecho_demo.html` (3.1 MB) | Self-contained offline dashboard with all tables embedded |
| Quarantined | `results/legacy_2026-10-27/` | The original 9-event exports — kept for history, **never cite** |

### `docs/`

| File | Purpose |
|---|---|
| `EventEcho_Presentation_v3.1.pdf` | The submitted 10-slide deck |
| `final-evidence-audit-2026-10-29.md` | Full release audit: what was fixed, coverage, timing, overlap, statistics, validation |
| `problem-statement-coverage.md` | Every PS-03 basic/advanced/brownie item → code → output → dashboard panel |
| `judge-qa.md` | 25 anticipated judge questions with short answers, plus "things never to say" |
| `earnings-and-robust-tests.md` | The 266-day results study, placebo, replication and the three follow-ups |
| `release-time-verification.md` | Per-event release-session evidence and the 2 unresolved cases |
| `event-timing-overlap-return-convention.md` | Day 0, overlap and Close/Close convention in detail |
| `event-source-and-validation-audit.md` | Provenance and validation of every registry event |
| `headline-classifier.md` | Classifier design, leakage controls and scores |

### `.github/workflows/`

| Workflow | Trigger | What it does |
|---|---|---|
| `tests.yml` | Push to `main`/`feat/**`, pull requests | Runs the 155-test suite, no network |
| `historical-rebuild.yml` | Changes on `main`, or manual | Tests → snapshot (fresh or reused via `source_run_id`) → analysis → dashboard → `verify_release.py`; uploads a 7-day artifact, publishes outputs when asked |
| `recover-market-snapshot.yml` | Registry/method changes on `main`, or manual | Re-acquires a frozen snapshot and re-runs the analysis; hard-fails on an incomplete snapshot |

---

**Reading tip for judges:** start at `src/config.py` (every assumption lives there), then `src/event_study.py` → `src/statistics.py` → `src/robust_tests.py` for the numbers, and `scripts/verify_release.py` for how those numbers are defended.
