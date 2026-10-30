# Company results days and robust group tests

Results below are from Historical Rebuild run **41011220268** (code `34a6cfa`, frozen snapshot `eventecho-historical-40848349737`). The integrity check passed 78 of 78 checks.

## 1. Why a separate earnings study

The 21-event registry mostly holds market-wide events, which hit all seven stocks on the same day. Those events are few, they overlap, and their effects point in different directions. Quarterly results are the opposite case: there are many of them, each one is about a single company, and the dates are public. So the project adds a second, cleaner sample: **every quarterly results day of the seven companies from Q1 FY17 (Jul 2016) to Q2 FY26 (Nov 2025), 266 announcements**, each studied only for the company that reported.

## 2. The dates (`results/earnings_events.csv`)

- 266 rows, exactly 38 per stock. `scripts/build_earnings_calendar.py` rebuilds the file and refuses to write it if a quarter is missing or duplicated.
- Sources:
  - 222 dates come from company board-meeting histories (Trendlyne).
  - 43 come from dated news reports of the results themselves (all ITC dates, plus the gaps and conflicts listed below).
  - 1 comes from an NSE filing.
- Every row has a `source_url`.
- Cross-checks:
  - **Random sample.** A seeded random sample of 12 board-history dates (`random_state=2026`) was checked against independent news reports, and **12 of 12 agreed**.
  - **Targeted checks.** 22 more dates were checked on purpose: unclear Adani Ports meetings, early years and others. All of them agreed.
- Problems found and fixed using news dates, each noted in the file:
  - HDFC Bank Q1 FY18 was missing from the board history; the date is 24 Jul 2017.
  - HDFC Bank Q2 FY24 is listed as Sunday 15 Oct 2023; the results came out on 16 Oct.
  - Reliance Q1 FY21 had two meeting dates (24 Jul and a revised 30 Jul 2020); the results came out on 30 Jul.
  - Three 2025 Reliance dates came after the history page ended.
- A test checks that consecutive announcements of a stock are 25–150 days apart, which catches a wrong month or year. The shortest gap is ITC's COVID-delayed Q4 FY20 (26 Jun 2020) followed by Q1 FY21 (24 Jul 2020).
- **No date was inferred from a price move.**

## 3. Design (fixed before results were seen)

| Choice | Setting |
|---|---|
| Day 0 | First NIFTY 50 session on or after the announcement date (37 announcements fell on a weekend or holiday) |
| Model | Same market model as the main study: 100-session estimation, 20-session gap |
| Main window | (−1,+1). Many companies report after the close, so this window always contains the first full session after the release |
| Main measure | **Event-induced variance ratio**: the average of SCAR², where SCAR = CAR ÷ its own standard error. About 1.02 if results days were ordinary days |
| Why not the signed average? | Without analyst forecasts we cannot tell a beat from a miss, so good and bad news cancel. The signed CAR is reported only as a secondary check |
| Split | Development: to 31 Dec 2022 (182 announcements). Holdout: from 1 Jan 2023 (84) |
| Dependence | Announcements whose ±5 windows overlap in time (for example TCS and Infosys a day apart) form one cluster: 49 clusters in total |
| Overlap with macro events | 49 announcements share a window with a registry event; a summary without them is reported |
| Placebo | The same stocks, 30 sessions after each results day (mid-quarter). Pseudo-events touching a real results window are dropped (2 dropped) |

## 4. Results

### Results days move stocks much more than ordinary days

| Sample, Day −1 to +1 | Announcements | Variance ratio | 95% interval |
|---|---:|---:|---:|
| All results days | 266 | **2.68** | 2.17 – 3.23 |
| Without macro-event overlap | 217 | 2.55 | 2.00 – 3.15 |
| Development (to 2022) | 182 | 2.15 | 1.69 – 2.67 |
| Holdout (2023 onward) | 84 | 3.82 | 2.72 – 4.95 |
| **Placebo: ordinary days** | 264 | **0.88** | 0.70 – 1.08 |

- 19% of results days have |SCAR| above 1.96, against 5% expected for ordinary days.
- The average absolute CAR is 2.90% on results days and 1.76% on placebo days.
- The effect is concentrated on **Day +1**: the mean squared standardized abnormal return is 4.77 there, against 1.77 on Day 0 and about 1.0–1.7 on other days. This fits a release after the close or on a weekend.
- The effect weakens in wider windows (1.90 for ±3, 1.43 for ±5), because the extra ordinary days dilute it.

### No average direction

The mean CAR(−1,+1) is −0.28% (median −0.17%, 46% positive). BMP p = 0.13, Kolari–Pynnönen p = 0.12, rank test p = 0.26. This is expected without surprise data, and it is **not** evidence that results are bad news.

### By stock and sector (Day −1 to +1)

| Stock | Variance ratio [95% CI] | Mean \|CAR\| |
|---|---|---:|
| INFOSYS | 4.56 [2.43, 7.21] | 3.84% |
| HDFC_BANK | 3.08 [1.65, 4.70] | 2.04% |
| ADANI_PORTS | 2.55 [0.97, 5.11] | 3.34% |
| TCS | 2.53 [1.59, 3.57] | 2.95% |
| ITC | 2.46 [1.25, 3.87] | 2.31% |
| RELIANCE | 2.44 [1.49, 3.55] | 2.58% |
| ADANI_ENTERPRISES | 1.15 [0.56, 1.93] | 3.27% |

- Sector intervals overlap: IT Services 3.54, Banking 3.08, FMCG 2.46, Energy 2.44, Adani Group 1.85. So **no sector ranking is claimed**. "IT reacts more than banks" is not supported.
- Adani Enterprises is the clear exception. Its stock moves a lot on ordinary days (mean |CAR| on placebo days is 2.1%), and results add little to that.

### Does it replicate?

| Level | Outcome |
|---|---|
| Pooled | Yes. The interval excludes 1 in both the development and the holdout sample |
| Per stock (lower bound above 1 in **both** samples) | Only **TCS and Infosys**. HDFC Bank shows it in development only, Reliance in holdout only |
| Per stock, with 12 holdout announcements each | The intervals are wide |
| Ranking stocks by variance ratio | Does not persist (rank correlation −0.11) |

## 5. Robust tests for the 21 macro events (`robust_tests_registry.csv`)

The same three tests are run for every CAAR group:

- **BMP:** standardized CARs with their observed spread across events.
- **Kolari–Pynnönen:** shrinks BMP for residual correlation. A pair counts only when its event windows coincide in calendar time.
- **Corrado/Cowan rank test.**

All events, Day −1 to +1:

| Measure | Value |
|---|---|
| Mean CAR | −0.12% (bootstrap interval −0.53% to +0.49%) |
| BMP | p = 0.92 |
| KP | p = 0.93 |
| Rank test | p = 0.61 |

**This confirms the earlier conclusion: there is no reliable average move across the macro events.**

- No group has a rank p below 0.05 in any window.
- One group, election results at (−1,+1), has a KP p of 0.04. It rests on only 2 events and is uncorrected across 66 rows, so it is treated as descriptive.

## 6. Limits

- No earnings-surprise data, so direction cannot be studied. A natural next step would be consensus estimates, which are not freely available.
- Board-meeting dates are third-party compiled. The sample check found 12/12 agreement, but a full exchange-filing audit was not done.
- Release times within the day are not verified for these 266 dates. The (−1,+1) window is the safeguard.
- The placebo uses one fixed shift (30 sessions). Other shifts were not tried, to avoid choosing the friendliest one.
- KP correlations are averaged over all pairs. Pairs whose event days never coincide count as zero. This is a project choice, documented in `src/robust_tests.py`.
