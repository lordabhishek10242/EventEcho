"""Build results/earnings_events.csv from hand-collected, sourced announcement dates.

Every date below was read from a public source on 2026-10-29. No date is
inferred from price movements. ``announcement_date`` is the calendar date the
company released (or its board approved) quarterly results. The intraday
release time is not verified, so event day 0 is the first NIFTY 50 session on
or after this date; windows such as (-1,+1) cover a release after the close.

Sources
-------
* Trendlyne board-meeting history pages (one page per company). These list
  board-meeting dates whose purpose is "Quarterly Results" / "Audited Results".
* For ITC, whose Trendlyne page was not reachable from this session, each date
  comes from a dated news report of the results (mostly Business Standard /
  PTI / Business Today), whose headline reports the results themselves.
* Where Trendlyne was missing a quarter, listed two dates, or disagreed with
  news reports, the news date is used and the row has a note.

Run: python scripts/build_earnings_calendar.py
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "earnings_events.csv"
COLLECTED_ON = "2026-10-29"

TRENDLYNE = {
    "TCS": "https://trendlyne.com/equity/board-meeting/TCS/1372/tata-consultancy-services-ltd-board_meeting",
    "INFOSYS": "https://trendlyne.com/equity/board-meeting/INFY/630/infosys-ltd-board_meeting",
    "HDFC_BANK": "https://trendlyne.com/equity/board-meeting/HDFCBANK/533/hdfc-bank-ltd-board_meeting",
    "RELIANCE": "https://trendlyne.com/equity/board-meeting/RELIANCE/1127/reliance-industries-ltd-board_meeting",
    "ADANI_ENTERPRISES": "https://trendlyne.com/equity/board-meeting/ADANIENT/26/adani-enterprises-ltd-board_meeting",
    "ADANI_PORTS": "https://trendlyne.com/equity/board-meeting/ADANIPORTS/27/adani-ports-special-economic-zone-ltd-board_meeting",
}

# Board-meeting dates read from the Trendlyne pages (results meetings only).
TRENDLYNE_DATES = {
    "TCS": """2016-07-14 2016-10-13 2017-01-12 2017-04-18 2017-07-13 2017-10-12
        2018-01-11 2018-04-19 2018-07-10 2018-10-11 2019-01-10 2019-04-12 2019-07-09 2019-10-10
        2020-01-17 2020-04-16 2020-07-09 2020-10-07 2021-01-08 2021-04-12 2021-07-08 2021-10-08
        2022-01-12 2022-04-11 2022-07-08 2022-10-10 2023-01-09 2023-04-12 2023-07-12 2023-10-11
        2024-01-11 2024-04-12 2024-07-11 2024-10-10 2025-01-09 2025-04-10 2025-07-10 2025-10-09""",
    "INFOSYS": """2016-07-15 2016-10-14 2017-01-13 2017-04-13 2017-07-14 2017-10-24
        2018-01-12 2018-04-13 2018-07-13 2018-10-16 2019-01-11 2019-04-12 2019-07-12 2019-10-11
        2020-01-10 2020-04-20 2020-07-15 2020-10-14 2021-01-13 2021-04-14 2021-07-14 2021-10-13
        2022-01-12 2022-04-13 2022-07-24 2022-10-13 2023-01-12 2023-04-13 2023-07-20 2023-10-12
        2024-01-11 2024-04-18 2024-07-18 2024-10-17 2025-01-16 2025-04-17 2025-07-23 2025-10-16""",
    "HDFC_BANK": """2016-07-21 2016-10-25 2017-01-24 2017-04-21 2017-10-24
        2018-01-19 2018-04-21 2018-07-21 2018-10-20 2019-01-19 2019-04-20 2019-07-20 2019-10-19
        2020-01-18 2020-04-18 2020-07-18 2020-10-17 2021-01-16 2021-04-17 2021-07-17 2021-10-16
        2022-01-15 2022-04-16 2022-07-16 2022-10-15 2023-01-14 2023-04-15 2023-07-17
        2024-01-16 2024-04-20 2024-07-20 2024-10-19 2025-01-22 2025-04-19 2025-07-19 2025-10-18""",
    "RELIANCE": """2016-07-15 2016-10-20 2017-01-16 2017-04-24 2017-07-20 2017-10-13
        2018-01-19 2018-04-27 2018-07-27 2018-10-17 2019-01-17 2019-04-18 2019-07-19 2019-10-18
        2020-01-17 2020-04-30 2020-10-30 2021-01-22 2021-04-30 2021-07-23 2021-10-22
        2022-01-21 2022-05-06 2022-07-22 2022-10-21 2023-01-20 2023-04-21 2023-07-21 2023-10-27
        2024-01-19 2024-04-22 2024-07-19 2024-10-14 2025-01-16""",
    "ADANI_ENTERPRISES": """2016-08-10 2016-10-24 2017-02-14 2017-05-24 2017-08-12 2017-11-13
        2018-01-18 2018-05-10 2018-08-07 2018-10-31 2019-02-07 2019-05-29 2019-08-08 2019-11-12
        2020-02-05 2020-05-06 2020-08-06 2020-11-04 2021-02-03 2021-05-05 2021-08-03 2021-10-27
        2022-02-14 2022-05-03 2022-08-04 2022-11-03 2023-02-14 2023-05-04 2023-08-03 2023-11-02
        2024-02-01 2024-05-02 2024-08-01 2024-10-29 2025-01-30 2025-05-01 2025-07-31 2025-11-04""",
    # The Adani Ports page shows meeting dates, mostly without a purpose. Only
    # dates in the usual results months are kept; the ambiguous ones (Q2FY17,
    # Q3FY18, Q4FY18, Q4FY19, Q4FY20, Q1FY21, Q2FY24) were checked against news.
    "ADANI_PORTS": """2016-08-09 2016-10-25 2017-02-14 2017-05-24 2017-08-12 2017-11-13
        2018-01-18 2018-05-03 2018-08-06 2018-10-23 2019-02-06 2019-05-27 2019-08-07 2019-11-11
        2020-02-04 2020-05-05 2020-08-11 2020-11-03 2021-02-09 2021-05-04 2021-08-03 2021-10-27
        2022-02-01 2022-05-24 2022-08-08 2022-11-01 2023-02-07 2023-05-30 2023-08-08 2023-11-09
        2024-02-01 2024-05-02 2024-08-01 2024-10-29 2025-01-30 2025-05-01 2025-08-05 2025-11-04""",
}

BS = "https://www.business-standard.com/amp/article/"
# Rows that come from news reports: (stock, date, url, note)
NEWS_ROWS = [
    ("HDFC_BANK", "2017-07-24", "https://www.businesstoday.in/industry/banks/story/hdfc-bank-net-profit-rises-to-rs-3893-crore-85475-2017-07-24",
     "Missing from the Trendlyne page; Q1 FY18 results reported 24 Jul 2017 (also Business Standard 117072400668)."),
    ("HDFC_BANK", "2023-10-16", "https://www.business-standard.com/amp/companies/results/hdfc-bank-q2-result-net-profit-rises-50-to-rs-15-796-cr-nii-up-30-3-123101600664_1.html",
     "Trendlyne lists 15 Oct 2023 (a Sunday); results were reported 16 Oct 2023. Day 0 is 16 Oct either way."),
    ("RELIANCE", "2020-07-30", "https://www.business-standard.com/amp/article/news-cm/ril-q1-pat-jumps-31-to-rs-13-248-cr-120073001907_1.html",
     "Trendlyne lists 24 Jul 2020 and a revised 30 Jul 2020 meeting for Q1 FY21; results were reported 30 Jul 2020."),
    ("RELIANCE", "2025-04-25", "https://www.businesstoday.in/markets/stocks/story/ril-q4-results-net-profit-rises-24-to-rs-19407-crore-beats-estimates-dividend-announced-473696-2025-04-25",
     "After the Trendlyne page's last row; Q4 FY25 results reported 25 Apr 2025."),
    ("RELIANCE", "2025-07-18", "https://www.republicworld.com/business/ril-q1-fy26-results-mukesh-ambani-led-reliance-industries-net-profit-jumps-78-to-rs-26994-crore-check-full-earnings",
     "After the Trendlyne page's last row; Q1 FY26 results reported 18 Jul 2025."),
    ("RELIANCE", "2025-10-17", "https://www.businesstoday.in/markets/stocks/story/ril-q2-fy26-results-profit-jumps-10-to-rs-18165-crore-revenue-up-10-498829-2025-10-17",
     "After the Trendlyne page's last row; Q2 FY26 results reported 17 Oct 2025."),
    # ITC: each date from a dated news report of the results.
    ("ITC", "2016-07-21", BS + "companies/itc-q1-net-profit-up-10-to-rs-2-385-crore-116072100992_1.html", ""),
    ("ITC", "2016-10-26", BS + "pti-stories/itc-q2-net-up-10-at-rs-2-500-cr-116102600590_1.html", ""),
    ("ITC", "2017-01-27", "https://business-standard.com/article/companies/itc-q3-net-rises-5-7-to-rs-2-647-cr-cigarette-business-under-strain-117012700457_1.html", ""),
    ("ITC", "2017-05-26", BS + "pti-stories/itc-q4-net-profit-up-12-to-rs-2-669-47-crore-117052600997_1.html", ""),
    ("ITC", "2017-07-27", BS + "pti-stories/itc-q1-net-profit-up-7-37-to-rs-2-560-50-crore-117072701427_1.html", ""),
    ("ITC", "2017-10-27", BS + "companies/itc-q2-net-increases-nearly-6-at-rs-2-640-cr-117102700520_1.html", ""),
    ("ITC", "2018-01-19", BS + "pti-stories/itc-ltd-q3-net-profit-up-17-pc-at-rs-3-090-crore-118011900788_1.html", ""),
    ("ITC", "2018-05-16", "https://www.businesstoday.in/latest/corporate/story/itc-reports-over-9-per-cent-jump-in-net-profit-in-q4-fy18-beating-analyst-predictions-106936-2018-05-16", ""),
    ("ITC", "2018-07-26", BS + "reuters/itc-first-quarter-profit-rises-10-percent-beats-estimates-118072600877_1.html", ""),
    ("ITC", "2018-10-26", BS + "companies/itc-q2fy19-profit-up-12-at-rs-29-55-billion-vs-26-40-billion-118102600651_1.html", ""),
    ("ITC", "2019-01-23", BS + "markets/itc-q3-net-profit-rises-4-yoy-to-rs-3-209-07-crore-119012300604_1.html", ""),
    ("ITC", "2019-05-13", BS + "markets/itc-q4-net-profit-up-19-announces-dividend-of-rs-5-75-per-share-119051300536_1.html", ""),
    ("ITC", "2019-08-02", "https://www.businesstoday.in/latest/corporate/story/itc-q1-profit-climbs-13-to-rs-3174-crore-cigarettes-revenue-up-6-220444-2019-08-02", ""),
    ("ITC", "2019-10-24", BS + "companies/itc-q2-net-profit-up-37-to-rs-4-173-72-crore-net-sales-6-16-119102401505_1.html", ""),
    ("ITC", "2020-01-31", BS + "pti-stories/itc-q3-net-profit-up-29-pc-at-rs-4-047-87-crore-120013101824_1.html", ""),
    ("ITC", "2020-06-26", "https://business-standard.com/amp/article/companies/itc-q4-consolidated-net-profit-rises-9-28-to-rs-3-926-cr-sales-down-5-120062601537_1.html",
     "Q4 FY20 results were delayed to June by the COVID-19 lockdown."),
    ("ITC", "2020-07-24", "https://www.businesstoday.in/amp/latest/corporate/story/itc-q1-results-profit-falls-25-to-rs-2567-crore-amid-coronavirus-pandemic-268074-2020-07-24", ""),
    ("ITC", "2020-11-06", BS + "news-cm/itc-q2-pat-declines-20-to-rs-3-232-cr-120110700319_1.html",
     "The article says the result was announced after market hours on 6 Nov 2020, so the first reaction session is 9 Nov 2020."),
    ("ITC", "2021-02-11", "https://www.business-standard.com/article/companies/itc-q3-results-net-profit-declines-11-yoy-to-rs-3-587-crore-121021101791_1.html", ""),
    ("ITC", "2021-06-01", BS + "companies/itc-q4-consolidated-net-profit-falls-3-declares-final-dividend-of-rs-5-75-121060101191_1.html",
     "Q4 FY21 results were reported in June after the second COVID-19 wave."),
    ("ITC", "2021-07-24", "https://businesstoday.in/latest/corporate/story/itc-q1-results-net-profit-rises-28-to-rs-3013-cr-revenue-up-36-302301-2021-07-24",
     "Saturday release; day 0 is the next session."),
    ("ITC", "2021-10-27", BS + "companies/itc-q2-consolidated-net-profit-rises-10-09-at-rs-3-763-73-crore-121102701749_1.html", ""),
    ("ITC", "2022-02-03", "https://www.businesstoday.in/latest/corporate/story/itc-q3-net-profit-rises-15-to-rs-4056-cr-interim-dividend-of-rs-525-declared-321374-2022-02-03", ""),
    ("ITC", "2022-05-18", "https://businesstoday.in/latest/corporate/story/itc-q4-results-net-profit-rises-12-to-rs-4196-cr-final-dividend-declared-334135-2022-05-18", ""),
    ("ITC", "2022-08-01", BS + "companies/itc-q1-results-net-profit-rises-38-to-rs-4-169-cr-122080101067_1.html", ""),
    ("ITC", "2022-10-20", "https://businesstoday.in/latest/corporate/story/itc-q2-results-profit-rises-24-to-rs-467032-crore-350468-2022-10-20", ""),
    ("ITC", "2023-02-03", BS + "companies/itc-q3-net-profit-rises-23-4-to-rs-5-006-cr-on-growth-across-segments-123020301649_1.html", ""),
    ("ITC", "2023-05-18", "https://www.business-standard.com/amp/companies/results/itc-q4fy23-net-profit-rises-21-1-declares-dividend-of-rs-9-5-share-123051800622_1.html", ""),
    ("ITC", "2023-08-14", "https://archives.nseindia.com/corporate/JuneResults2023_14082023163650.pdf",
     "NSE filing of the June 2023 quarter results, stamped 14 Aug 2023 16:36."),
    ("ITC", "2023-10-19", "https://business-standard.com/amp/companies/results/itc-q2fy24-results-profit-after-tax-up-6-02-at-rs-4-898-07-crore-123101901328_1.html", ""),
    ("ITC", "2024-01-29", "https://www.business-standard.com/amp/companies/results/itc-q3-results-net-profit-rises-10-8-to-rs-5-572-crore-beats-estimates-124012900753_1.html", ""),
    ("ITC", "2024-05-23", "https://www.businesstoday.in/markets/company-stock/story/itc-q4-results-net-profit-falls-10-to-rs-5020-cr-dividend-announced-430654-2024-05-23", ""),
    ("ITC", "2024-08-01", "https://businesstoday.in/markets/company-stock/story/itc-q1-fy25-results-profit-up-marginally-revenue-climbs-7-yoy-439754-2024-08-01", ""),
    ("ITC", "2024-10-24", "https://www.businesstoday.in/latest/corporate/story/itc-q2-results-net-profit-up-31-yoy-at-rs-50783-crore-revenue-jumps-168-451462-2024-10-24", ""),
    ("ITC", "2025-02-06", "https://www.business-standard.com/amp/companies/results/itc-q3fy25-results-profit-down-7-27-to-rs-5-013-16-cr-revenue-up-9-125020601289_1.html", ""),
    ("ITC", "2025-05-22", "https://www.businesstoday.in/amp/markets/stocks/story/itc-q4-results-profit-zooms-to-rs-19727-crore-on-exceptional-gain-rs-785-per-share-dividend-announced-477397-2025-05-22", ""),
    ("ITC", "2025-08-01", "https://www.business-standard.com/amp/companies/quarterly-results/itc-q1-fy26-results-net-profit-revenue-sales-earnings-shares-dividend-125080101369_1.html", ""),
    ("ITC", "2025-10-30", "https://www.business-standard.com/amp/companies/quarterly-results/itc-q2-results-net-profit-rises-2-7-to-5-126-11-cr-revenue-down-1-3-125103001412_1.html", ""),
]

# Independent news reports that agree with a Trendlyne date (cross-checks).
CROSS_CHECKS = {
    ("HDFC_BANK", "2018-07-21"): BS + "pti-stories/hdfc-bank-q1-net-profit-up-18-pc-at-rs-4-601-cr-118072100620_1.html",
    ("TCS", "2024-01-11"): "TCS investor-relations results calendar (checked earlier in this project)",
    ("TCS", "2018-04-19"): "TCS investor-relations results calendar (checked earlier in this project)",
    ("ADANI_PORTS", "2016-10-25"): "https://www.business-standard.com/article/markets/adani-ports-soars-as-q2-profit-beats-forecast-116102500584_1.html",
    ("ADANI_PORTS", "2017-05-24"): BS + "pti-stories/adani-ports-q4-net-profit-up-27-to-rs-1-179-cr-117052400702_1.html",
    ("ADANI_PORTS", "2018-01-18"): BS + "pti-stories/adani-ports-q3-consolidated-net-up-19-51-at-rs-1-001-cr-118011800989_1.html",
    ("ADANI_PORTS", "2018-05-03"): BS + "news-ians/adani-ports-q4-net-at-rs-927-cr-dips-21-on-more-tax-outgo-118050301120_1.html",
    ("ADANI_PORTS", "2019-05-27"): BS + "pti-stories/adani-ports-q4-profit-up-41-at-rs-1-314-cr-119052700991_1.html",
    ("ADANI_PORTS", "2020-05-05"): BS + "pti-stories/adani-ports-q4-net-profit-falls-74-pc-to-rs-340-cr-120050501326_1.html",
    ("ADANI_PORTS", "2020-08-11"): BS + "news-cm/adani-ports-sez-q1-pat-tumbles-26-to-rs-758-cr-120081101245_1.html",
    ("ADANI_PORTS", "2021-05-04"): "https://www.business-standard.com/article/news-cm/adani-ports-q4-pat-soars-285-yoy-to-rs-1288-cr-121050400925_1.html",
    ("ADANI_PORTS", "2023-02-07"): "https://trendlyne.com/posts/3941672/adani-ports-and-special-economic-zone-ltd-532921-board-meeting-outcome-for-outcome-of-board-meeting-held-on-february-7-2023-and-submission-of-unaudited-financial-results-for-the-quarter-and-nine-months-ended-december-31-2022",
    ("ADANI_PORTS", "2023-11-09"): "https://www.businesstoday.in/amp/markets/company-stock/story/adani-ports-q2-results-profit-up-42-at-rs-1748-crore-sales-jumps-276-shares-fall-405207-2023-11-09",
    ("ADANI_PORTS", "2025-05-01"): "https://trendlyne.com/posts/5091163/adani-ports-and-special-economic-zone-ltd-532921-outcome-of-board-meeting-held-on-may-1-2025-and-submission-of-audited-financial-results-standalone-and-consolidated-for-the-quarter-and-year-ended-march-31-2025",
    ("ADANI_PORTS", "2025-11-04"): "https://www.arihantcapital.com/company-information/board-meetings/21281",
    ("ADANI_ENTERPRISES", "2019-05-29"): "https://www.business-standard.com/article/pti-stories/adani-enterprises-q4-profit-rises-47-pc-to-rs-266-cr-119052900812_1.html",
    ("ADANI_ENTERPRISES", "2018-10-31"): BS + "pti-stories/adani-enterprises-q2-pat-jumps-three-fold-to-rs-172-cr-118103100956_1.html",
    ("ADANI_ENTERPRISES", "2020-11-04"): "https://www.dtnext.in/business/2020/11/04/adani-enterprises-q2-consolidated-net-profit-at-rs-436-cr",
    ("ADANI_ENTERPRISES", "2021-08-03"): BS + "companies/adani-enterprises-q1-fy22-consol-pat-at-rs-271-crore-up-9-times-yoy-121080301796_1.html",
    ("ADANI_ENTERPRISES", "2023-11-02"): "https://businesstoday.in/markets/company-stock/story/adani-enterprises-q2-results-profit-tanks-51-to-rs-228-crore-sales-down-41-404292-2023-11-02",
    ("ADANI_ENTERPRISES", "2025-11-04"): "https://www.businesstoday.in/amp/markets/stocks/story/adani-enterprises-q2-earnings-net-profit-rises-total-income-at-rs-21844-crore-500803-2025-11-04",
    ("ADANI_PORTS", "2017-11-13"): "https://www.business-standard.com/article/pti-stories/adani-ports-q2-consolidated-net-dips-8-to-rs-992-cr-117111300870_1.html",
}

# A seeded random sample (pandas .sample(12, random_state=2026)) of Trendlyne
# rows that had no cross-check yet, each checked against an independent news
# report. All 12 agreed.
RANDOM_SAMPLE_CHECKS = {
    ("HDFC_BANK", "2024-01-16"): "https://business-standard.com/amp/companies/results/hdfc-bank-q3-results-profit-up-34-to-rs-16-373-crore-gnpa-rises-to-1-26-124011600564_1.html",
    ("ADANI_ENTERPRISES", "2022-11-03"): "https://www.businesstoday.in/latest/corporate/story/adani-enterprises-q2-profit-surges-117-to-rs-461-crore-revenue-up-183-351775-2022-11-03",
    ("ADANI_ENTERPRISES", "2017-11-13"): BS + "pti-stories/adani-enterprises-q2-pat-falls-6-to-rs-59-cr-117111301167_1.html",
    ("ADANI_ENTERPRISES", "2020-08-06"): BS + "companies/adani-enterprises-reports-consolidated-loss-of-rs-65-67-cr-in-q1-120080601500_1.html",
    ("INFOSYS", "2019-07-12"): "https://www.business-standard.com/article/news-ani/infosys-profit-up-5-3-pc-to-rs-3-802-crore-in-q1-fy20-119071200997_1.html",
    ("ADANI_ENTERPRISES", "2019-11-12"): "https://business-standard.com/article/companies/adani-enterprises-posts-rs-75-cr-q2-loss-as-group-firm-writes-off-rs-130-cr-119111201067_1.html",
    ("RELIANCE", "2021-07-23"): BS + "companies/ril-q1-results-consolidated-net-profit-falls-7-to-rs-12-273-crore-121072301309_1.html",
    ("HDFC_BANK", "2017-01-24"): BS + "pti-stories/hdfc-bank-q3-net-rises-15-14-to-rs-3-865-33-crore-117012400425_1.html",
    ("INFOSYS", "2022-10-13"): "https://business-standard.com/article/news-cm/infosys-q2-pat-rises-over-12-yoy-board-oks-rs-9300-cr-share-buyback-122101300935_1.html",
    ("ADANI_PORTS", "2016-08-09"): BS + "pti-stories/apsez-q1-net-up-31-at-rs-836-cr-116080900777_1.html",
    ("HDFC_BANK", "2022-04-16"): "https://businesstoday.in/industry/banks/story/hdfc-bank-q4-results-net-profit-up-228-yoy-to-over-rs-10000-crore-330031-2022-04-16",
    ("RELIANCE", "2018-07-27"): "https://www.businesstoday.in/latest/corporate/story/reliance-industries-q1-net-profit-rises-to-rs-9459-crore-108073-2018-07-27",
}


def fiscal_quarter(d: date) -> str:
    """Indian fiscal-year quarter label of the results period being reported."""
    if d.month in (1, 2, 3):
        return f"Q3FY{(d.year) % 100:02d}"
    if d.month in (4, 5, 6):
        return f"Q4FY{(d.year) % 100:02d}"
    if d.month in (7, 8, 9):
        return f"Q1FY{(d.year + 1) % 100:02d}"
    return f"Q2FY{(d.year + 1) % 100:02d}"


def build_rows() -> list[dict]:
    rows: dict[tuple[str, str], dict] = {}
    for stock, text in TRENDLYNE_DATES.items():
        for value in text.split():
            rows[(stock, value)] = {
                "stock": stock,
                "announcement_date": value,
                "source_url": TRENDLYNE[stock],
                "source_type": "board_meeting_history",
                "note": "",
            }
    for stock, value, url, note in NEWS_ROWS:
        source_type = "exchange_filing" if "nseindia" in url else (
            "board_meeting_history" if "trendlyne.com/equity" in url else "news_report")
        rows[(stock, value)] = {
            "stock": stock,
            "announcement_date": value,
            "source_url": url,
            "source_type": source_type,
            "note": note,
        }
    out = []
    for key in sorted(rows, key=lambda k: (k[0], k[1])):
        row = rows[key]
        d = date.fromisoformat(row["announcement_date"])
        row["fiscal_quarter"] = fiscal_quarter(d)
        row["weekday"] = d.strftime("%a")
        row["cross_check_source"] = CROSS_CHECKS.get(key, "") or RANDOM_SAMPLE_CHECKS.get(key, "")
        row["cross_check_kind"] = ("random_sample" if key in RANDOM_SAMPLE_CHECKS
                                   else "targeted" if key in CROSS_CHECKS else "")
        out.append(row)
    for i, row in enumerate(out, 1):
        row["earnings_id"] = f"ERN{i:03d}"
        row["collected_on"] = COLLECTED_ON
    return out


def validate(rows: list[dict]) -> None:
    by_stock: dict[str, list[dict]] = {}
    for row in rows:
        by_stock.setdefault(row["stock"], []).append(row)
    for stock, items in by_stock.items():
        quarters = [r["fiscal_quarter"] for r in items]
        if len(quarters) != len(set(quarters)):
            dupes = sorted({q for q in quarters if quarters.count(q) > 1})
            raise ValueError(f"{stock}: more than one date for {dupes}")
        if len(items) != 38:
            raise ValueError(f"{stock}: expected 38 quarters (Q1FY17-Q2FY26), found {len(items)}")
    if set(CROSS_CHECKS) & set(RANDOM_SAMPLE_CHECKS):
        raise ValueError("a date is in both cross-check lists")
    for key in list(CROSS_CHECKS) + list(RANDOM_SAMPLE_CHECKS):
        if not any((r["stock"], r["announcement_date"]) == key for r in rows):
            raise ValueError(f"cross-check for a date not in the calendar: {key}")


def main() -> None:
    rows = build_rows()
    validate(rows)
    fields = ["earnings_id", "stock", "fiscal_quarter", "announcement_date", "weekday", "source_type",
              "source_url", "cross_check_source", "cross_check_kind", "note", "collected_on"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
