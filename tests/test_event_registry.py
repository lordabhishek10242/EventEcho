from pathlib import Path

import pandas as pd

from src.config import DEVELOPMENT_EVENT_IDS, HOLDOUT_EVENT_IDS


ROOT = Path(__file__).resolve().parents[1]


def test_expanded_earnings_events_have_primary_sources_and_reviewable_timing():
    events = pd.read_csv(ROOT / "results" / "event_master.csv").set_index("event_id")
    expected_domains = {
        "EVT010": "www.tcs.com",
        "EVT011": "www.infosys.com",
        "EVT012": "www.hdfcbank.com",
        "EVT013": "www.ril.com",
        "EVT014": "itcportal.com",
    }
    for event_id, domain in expected_domains.items():
        row = events.loc[event_id]
        assert row["category"] == "Corporate Earnings"
        assert row["source_url"].startswith("https://")
        assert domain in row["source_url"]
        assert pd.notna(row["timing_confidence"])
        assert pd.notna(row["verification_status"])
        assert pd.notna(row["notes"])
        assert event_id in HOLDOUT_EVENT_IDS
        assert event_id not in DEVELOPMENT_EVENT_IDS


def test_infosys_earnings_day_zero_respects_good_friday_market_holiday():
    events = pd.read_csv(ROOT / "results" / "event_master.csv").set_index("event_id")
    row = events.loc["EVT011"]
    assert row["event_date"] == "2025-04-17"
    assert row["effective_event_date"] == "2025-04-21"
    assert "Good Friday" in row["notes"]
