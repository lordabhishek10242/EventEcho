# Event-source, timing, and validation audit

Audit prepared: 2026-10-29  
Scope: the 21-event master in `results/event_master.csv`, daily-close event-study timing, and the existing chronological strategy split.

## Status legend

- **Date verified** means the cited primary/publisher source supports the event date. It does not mean the exact publication timestamp is known.
- **Time verified** means the source explicitly provides a release time or schedule that supports the session mapping.
- **Provisional alignment** means daily-close data cannot establish the exact information arrival time; alternate session mapping must remain visible.
- **Unresolved** means do not describe the event as cleanly timed or causally identified.

## Findings from primary-source review

| Event | Source evidence | Session-alignment assessment | Required interpretation |
|---|---|---|---|
| EVT001 — demonetisation (2016-11-08) | PM India address page: https://www.pmindia.gov.in/en/news_updates/prime-ministers-address-to-the-nation/?comment=disable | Event master maps to 2016-11-09. Evening timing is recorded, but exact timestamp is not stored in the master. | Keep next-session mapping; retain timing caveat until timestamp is captured. |
| EVT002 — nationwide COVID lockdown (2020-03-24) | PM India primary page: https://www.pmindia.gov.in/en/news_updates/pm-calls-for-complete-lockdown-of-entire-nation-for-21-days/ | The official address says nationwide restrictions begin at midnight and embeds PMO posts dated 24 March. The repository records an evening announcement and maps Day 0 to 2020-03-25. | Next-session mapping is directionally supported. Do not claim the official page alone proves an exact 20:57 publication timestamp. COVID period is heavily confounded. |
| EVT003 — RBI 40 bp hike (2022-05-04) | RBI MPC resolution: https://www.rbi.org.in/commonman/Upload/English/PressRelease/PDFs/PR154MPC04052022.pdf | Primary release confirms date and rate decision; the accessible document text does not establish an intraday publication time. | Exact release time unresolved; run same-session vs next-session sensitivity. |
| EVT004 — RBI 50 bp hike (2022-06-08) | Current event-master source is PIB: https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=1832000&lang=2&reg=48 | Date is recorded, but exact release time needs source-level verification. | Keep timing unresolved; verify original RBI MPC release if available. |
| EVT005 — RBI 50 bp hike (2022-08-05) | Event-master source: https://systemhealth.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx_prid=54148%281%29.html | Date recorded; exact publication time not established in this audit. | Keep timing unresolved; verify official RBI MPC release and test alternate alignment. |
| EVT006 — Union Budget 2025-26 (2025-02-01) | PIB source in event master: https://www.pib.gov.in/Pressreleaseshare.aspx?PRID=2098352&lang=2&reg=48 | Saturday special trading session; exact event-time alignment requires separate verification. | Do not automatically treat this as a conventional next-session event. |
| EVT007 — RBI 25 bp hike (2023-02-08) | RBI source recorded in event master: https://systemhealth.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx_prid%3D55178.html | Date/session recorded; exact announcement time pending. Its event window overlaps EVT019's window in the current audit. | Do not attribute Adani CAR solely to the Hindenburg report or RBI decision. |
| EVT008 — RBI holds repo rate (2023-10-06) | RBI source recorded in event master: https://systemhealth.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx_prid%3D56502.html | Date/session recorded; exact time pending. | Keep timing caveat. |
| EVT009 — RBI holds rate, stance neutral (2024-10-09) | RBI source recorded in event master: https://systemhealth.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx_prid%3D58850.html | Date/session recorded; exact time pending. | Keep timing caveat. |
| EVT010 — TCS Q4 FY25 (2025-04-10) | TCS official schedule: https://www.tcs.com/content/dam/tcs/pdf/discover-tcs/investor-relations/corporate-actions/2024-25/tcs-schedule-of-events-q4-25.pdf | Schedule explicitly says release after trading hours; effective session 2025-04-11 is consistent. | Stronger timing confidence, but cross-stock moves are peer responses, not direct TCS effects. |
| EVT011 — Infosys Q4 FY25 (2025-04-17) | Infosys IR page: https://www.infosys.com/investors/reports-filings/quarterly-results/2024-2025/q4.html | Management comments were scheduled 16:15 IST, but that is not necessarily the exact results-publication timestamp. Good Friday 18 April means next regular session 21 April. | Treat 2025-04-21 as plausible, but distinguish management-call time from results filing time. |
| EVT012 — HDFC Bank Q4 FY25 (2025-04-19) | Official HDFC Bank filing URL in event master (dated 19 April 2025) | Saturday board approval; next regular session 2025-04-21 is plausible. | Preserve as next-session event; confirm filing timestamp if event-specific claims are made. |
| EVT013 — Reliance Q4 FY25 (2025-04-25) | Reliance official release: https://www.ril.com/news-media/press-releases/ril-q4-fy2025-financial-and-operational-performance | Date confirmed; exact publication time not in cited release metadata. | Same-day vs next-session timing sensitivity required. |
| EVT014 — ITC Q4 FY25 (2025-05-22) | ITC official release: https://itcportal.com/media-centre/press-releases/Media-Statement-Financial-Results-for-the-Quarter-ended-31st-March-2025.html | Date confirmed; exact publication time unresolved. | Same-day vs next-session timing sensitivity required. |
| EVT015 — GST rollout (2017-07-01) | PIB source in event master: https://www.pib.gov.in/PressReleasePage.aspx?PRID=1513001&lang=2&reg=3 | Midnight rollout; 2017-07-03 is the next regular session after the weekend. | This is a broad policy regime change, not a single precisely timed surprise. |
| EVT016 — 2019 Lok Sabha results (2019-05-23) | ECI source recorded in event master: https://www.eci.gov.in/general-elections/ | Counting/results evolve intraday; no single announcement time. | Treat as an event-exposed session, not a clean post-announcement session. |
| EVT017 — 2024 Lok Sabha counting (2024-06-04) | ECI official note: https://elections24.eci.gov.in/docs/NuSOkXHRFY.pdf | Counting evolves through the trading session; overlaps the GDP-release event window. | Daily-close CAR combines many information arrivals and overlapping events. |
| EVT018 — Q4 FY24 GDP release (2024-05-31) | PIB release: https://www.pib.gov.in/PressReleseDetailm.aspx?PRID=2022323&lang=2&reg=48 | 17:30 IST release recorded; next session 2024-06-03 is consistent. | Time alignment is well supported; still check overlap with nearby event windows. |
| EVT019 — Hindenburg report on Adani (2023-01-24) | Publisher's report page: https://hindenburgresearch.com/adani/ | The page explicitly says “Published on January 24, 2023”; exact clock time was not established in this review. Master maps to 2023-01-25. | Keep next-session mapping provisional until a reliable timestamp is captured. Attribute allegations to Hindenburg; publication date alone does not independently verify claims. The event window overlaps EVT007. |
| EVT020 — Union Budget 2024-25 (2024-07-23) | PIB source: https://www.pib.gov.in/PressReleseDetailm.aspx?PRID=2036078&lang=2&reg=48 | Budget presented during the market session; the cited PIB summary was posted at 20:22 IST, which is not the budget's intraday presentation time. | Daily data cannot isolate the speech/announcement sequence. Compare same-session vs next-session alignment; label the day event-exposed. |
| EVT021 — Q1 FY25 GDP release (2024-08-30) | PIB release: https://www.pib.gov.in/PressReleasePage.aspx?PRID=2050139&lang=2&reg=48 | 17:30 IST release recorded; next regular session 2024-09-02 is consistent. | Timing is well supported; preserve volatility-window overlap caveat. |

## Important methodological checks

1. **Do not equate a source URL with verified timing.** Store separate date-verification and time-verification fields; exact timestamp must be supported by the source.
2. **After-close releases:** when event information arrives after the regular close, Day 0 must be the next trading session. Never use the pre-release same-day close as the reaction.
3. **Intraday events:** daily OHLC/close data cannot identify the sequence or intraday reaction. Mark the day as event-exposed and report a next-session sensitivity; do not imply precise causal attribution.
4. **Overlapping events:** EVT007 and EVT019 overlap; EVT017 and EVT018 overlap; EVT020's window overlaps other macro/political windows. Clustered inference handles some dependence but cannot disentangle event attribution.
5. **Earnings cross-section:** for EVT010–EVT014, the focal issuer is the direct event target. Other stocks are peer/spillover observations, not direct issuer effects.
6. **Hindenburg claims:** describe these as claims made by the report publisher. Do not present allegations as adjudicated facts.
7. **Small, selected event set:** 21 selected events and seven retrospectively chosen stocks do not support broad population-level conclusions.

## Validation audit: current guardrails and remaining risks

- The code uses an explicit development/holdout event-ID list and rejects omissions, overlap, and non-chronological separation. This is a useful guardrail, not proof that all analysis decisions were made without holdout peeking.
- The strategy holdout contains 12 events but only about eight dependence clusters in the latest reported analysis; the holdout excess-return interval crosses zero. Do not claim a reliable trading edge.
- Estimation-window sensitivity is already computed for 60, 100, and 150 sessions. Significance changes across specifications; stable significance is a robustness description, not causality.
- HAC/Newey–West CAR inference is explicitly approximate and does not fully resolve cross-event dependence or event-induced variance.
- The cluster bootstrap resamples connected event-window/episode clusters, but the number of clusters is small and clusters may be uneven.
- Yahoo Finance prices are third-party and may be revised. A reproducible run is not the same as an independently verified or frozen data source. Stock adjusted close and Nifty price close have different dividend conventions.
- The market model is a one-factor benchmark model. Market-wide and sector-wide confounds, point-in-time universe selection, intraday response, execution costs, and market impact remain limitations.

## Required actions before final submission

- [ ] Capture source title, canonical URL, publication date/time (when actually available), access date, and whether timestamp was independently verified for every event.
- [ ] Verify exact RBI release times for EVT003–EVT005 and EVT007–EVT009; otherwise leave time unresolved.
- [ ] Verify whether the exact Hindenburg publication timestamp can be independently supported; do not infer it from first market reaction.
- [ ] Verify exchange/company filing timestamps for EVT013–EVT014 and distinguish Infosys' management-call time from its results-release time.
- [ ] Keep both same-session and next-session timing scenarios visible for EVT006, EVT013, EVT014, EVT016, EVT017, EVT019, and EVT020 where daily data cannot identify the arrival time.
- [ ] Review event windows and realized-volatility windows jointly; flag overlapping events before narrating individual effects.
- [ ] Run tests and the historical rebuild after any metadata or alignment change, and inspect the latest artifact rather than stale archived CSVs.
- [ ] In the final presentation, separate statistical evidence, robustness, plausible explanations, and causal claims.

This audit records what is currently supported and what remains unresolved; it does not silently rewrite event dates or assert that every source timestamp has been verified.
