import pandas as pd

from src.events import validate_events


def test_missing_clock_times_are_explicitly_unresolved_not_inferred():
    events = pd.DataFrame([{
        "event_id": "E1",
        "event_date": "2024-01-02",
        "effective_event_date": "2024-01-03",
        "event_time": "",
        "headline": "Example policy announcement",
        "category": "Policy",
        "source_name": "Official source",
        "source_url": "https://example.com/announcement",
        "episode_id": "EXAMPLE",
        "timing_confidence": "date_verified_time_unverified",
    }])

    result = validate_events(events)

    assert result.loc[0, "timestamp_evidence_status"] == "unresolved_missing_clock_time"
    assert bool(result.loc[0, "timestamp_review_required"])
    assert result.loc[0, "inference_scope"] == "exploratory_selected_event_not_preregistered"


def test_context_time_is_not_mislabelled_as_release_timestamp():
    events = pd.DataFrame([{
        "event_id": "E2",
        "event_date": "2024-01-02",
        "effective_event_date": "2024-01-03",
        "event_time": "Management comments scheduled 16:15 IST",
        "headline": "Earnings release",
        "category": "Corporate Earnings",
        "source_name": "Company IR",
        "source_url": "https://example.com/earnings",
        "episode_id": "EARNINGS",
        "timing_confidence": "medium_after_market_context_time_not_release_timestamp",
        "notes": "Management comments were scheduled at 16:15 IST.",
    }])

    result = validate_events(events)

    assert result.loc[0, "timestamp_evidence_status"] == "time_context_not_release_timestamp"
    assert bool(result.loc[0, "timestamp_review_required"])
