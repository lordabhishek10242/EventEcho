# Release-time verification (checked 2026-10-29)

## Why this matters

Day 0 must be the first trading session in which prices *could* react. Evening news (after the 15:30 IST close) belongs to the next session. News released during trading belongs to the same day. If Day 0 is wrong, the event window can miss the reaction or include the day before the news.

Rule kept throughout: **a release time is never inferred from a price move.** Only dated, timestamped sources are used.

## What was checked

For every registry event, `results/event_master.csv` now has:

| Column | Meaning |
|---|---|
| `release_session_check` | `after_close`, `during_session`, `non_trading_day` or `unresolved` |
| `release_time_evidence` | What the source says, with its timestamp |
| `release_time_source_url` | Where it says it |
| `day0_consistent_with_release` | Whether the existing Day 0 agrees with the evidence |

No event date or Day 0 changed. All 19 events with evidence were already mapped correctly. Two remain unresolved.

| Event | Evidence | Session | Day 0 OK? |
|---|---|---|---|
| EVT001 Demonetisation (8 Nov 2016) | Evening address; PTI report published 20:57 IST, notes "not legal tender from midnight tonight" | after close | Yes (9 Nov) |
| EVT002 COVID lockdown (24 Mar 2020) | Evening address; PTI report published 20:24 IST | after close | Yes (25 Mar) |
| EVT003 RBI surprise hike (4 May 2022) | Governor's unscheduled statement announced for 14:00 IST | during session | Yes |
| EVT004 RBI (8 Jun 2022) | RBI notice: statement at 10:00 am | during session | Yes |
| EVT005 RBI (5 Aug 2022) | Decision reported 10:40 IST | during session | Yes |
| EVT007 RBI (8 Feb 2023) | Decision reported 10:18 IST | during session | Yes |
| EVT008 RBI (6 Oct 2023) | Announcement set for 10 AM (Business Standard, 09:36 IST) | during session | Yes |
| EVT009 RBI (9 Oct 2024) | Decision reported 10:31 IST | during session | Yes |
| EVT006 Budget (Sat 1 Feb 2025) | Full trading session 09:15–15:30; speech expected 11:00 IST | during session | Yes |
| EVT019 Hindenburg (24 Jan 2023) | AP: "Hindenburg issued a report late Tuesday"; big falls reported Wednesday 25 Jan | after close | Yes (25 Jan) |
| EVT016 / EVT017 Election results | Counting from 08:00 IST; results arrive all day | during session | Yes |
| EVT020 Budget (23 Jul 2024) | Presented during the session (market live blog) | during session | Yes |
| EVT010, EVT011, EVT018, EVT021 | Already verified in the registry (after close) | after close | Yes |
| EVT012, EVT015 | Already verified (Saturday / midnight rollout) | non-trading day | Yes |
| EVT013 Reliance Q4 FY25, EVT014 ITC Q4 FY25 | No publication time found | unresolved | timing alternatives exported |

## Honest limits

* "Reported at 10:40 IST" bounds the release (it was out by then), it is not the exact minute. Daily closing prices cannot use minutes anyway; what matters is the session, and that is now sourced.
* Demonetisation and lockdown times come from the publication time of a wire report, not from a government transcript. Both clearly fall after 15:30.
* Hindenburg's exact clock time is still not established. The AP wording ("late Tuesday", New Delhi dateline) plus the Wednesday sell-off support Day 0 = 25 Jan.
* The older `event_time` and `timing_confidence` fields were left unchanged, so the existing timing-sensitivity outputs still run on the same rules. A unit test checks that each `release_session_check` agrees with the Day 0 mapping.

## Earnings calendar timing

For the 266 company results days (`results/earnings_events.csv`), only the date is sourced. Many Indian companies release after the close or on Saturdays, so the earnings study uses the first session on or after the date as Day 0 and treats the (−1,+1) window, which always contains the first full session after the release, as the main window. One case with a documented after-hours release (ITC, 6 Nov 2020) is noted in the file.
