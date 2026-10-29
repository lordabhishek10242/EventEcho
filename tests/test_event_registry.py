from pathlib import Path

import pandas as pd

from src.config import DEVELOPMENT_EVENT_IDS, HOLDOUT_EVENT_IDS, STOCK_TICKERS
from src.events import chronological_split


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


def test_macro_economic_and_industry_events_expand_coverage_without_leaking_into_wrong_split():
    events = pd.read_csv(ROOT / "results" / "event_master.csv")
    by_id = events.set_index("event_id")
    expected = {
        "EVT015": ("pib.gov.in", "Government Policy / Tax Reform", DEVELOPMENT_EVENT_IDS),
        "EVT016": ("eci.gov.in", "Election / Political Event", DEVELOPMENT_EVENT_IDS),
        "EVT017": ("eci.gov.in", "Election / Political Event", HOLDOUT_EVENT_IDS),
        "EVT018": ("pib.gov.in", "Economic Release", HOLDOUT_EVENT_IDS),
        "EVT019": ("hindenburgresearch.com", "Major Industry Event", DEVELOPMENT_EVENT_IDS),
    }
    for event_id, (domain, category, expected_split) in expected.items():
        row = by_id.loc[event_id]
        assert domain in row["source_url"]
        assert row["category"] == category
        assert row["source_url"].startswith("https://")
        assert event_id in expected_split
        assert pd.notna(row["timing_confidence"])
        assert pd.notna(row["verification_status"])
        assert pd.notna(row["notes"])

    assert len(events) == 19
    assert "ADANI_ENTERPRISES" in STOCK_TICKERS
    assert "ADANI_PORTS" in STOCK_TICKERS
    adani = by_id.loc["EVT019"]
    assert "ADANIENT.NS" in adani["affected_stocks"]
    assert "ADANIPORTS.NS" in adani["affected_stocks"]
    assert "short position disclosed" in adani["source_name"].lower()
    assert "allegations are attributed" in adani["notes"].lower()


def test_expanded_registry_has_non_overlapping_chronological_development_and_holdout():
    events = pd.read_csv(ROOT / "results" / "event_master.csv")
    split = chronological_split(events, DEVELOPMENT_EVENT_IDS, HOLDOUT_EVENT_IDS)
    labels = split.set_index("event_id")["sample"]
    assert labels["EVT015"] == "Development"
    assert labels["EVT016"] == "Development"
    assert labels["EVT019"] == "Development"
    assert labels["EVT017"] == "Holdout"
    assert labels["EVT018"] == "Holdout"
