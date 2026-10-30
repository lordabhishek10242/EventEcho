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



def test_verified_ist_publication_time_is_normalized_with_timezone():
    events = pd.DataFrame([{
        "event_id": "E3",
        "event_date": "2024-05-31",
        "effective_event_date": "2024-06-03",
        "event_time": "17:30 IST official PIB publication",
        "headline": "GDP release",
        "category": "Economic Release",
        "source_name": "Press Information Bureau",
        "source_url": "https://example.com/gdp",
        "episode_id": "GDP",
        "timing_confidence": "high_after_market_release_time_confirmed",
    }])

    result = validate_events(events)

    assert result.loc[0, "event_timestamp_ist"] == "2024-05-31T17:30:00+05:30"
    assert result.loc[0, "timestamp_semantics"] == "source_status_verified_publication_time_IST"


def test_registry_release_session_checks_are_sourced_and_consistent_with_day0():
    import pandas as pd

    registry = pd.read_csv("results/event_master.csv", dtype=str, keep_default_na=False)
    allowed = {"after_close", "during_session", "non_trading_day", "unresolved"}
    assert set(registry["release_session_check"]) <= allowed
    assert registry["release_time_evidence"].str.len().gt(20).all()
    assert registry["release_time_source_url"].str.startswith("http").all()
    resolved = registry[registry["release_session_check"] != "unresolved"]
    same_day = resolved["release_session_check"] == "during_session"
    # A release inside the session keeps Day 0 on that date; otherwise Day 0 moves to a later session.
    assert (resolved.loc[same_day, "effective_event_date"] == resolved.loc[same_day, "event_date"]).all()
    assert (resolved.loc[~same_day, "effective_event_date"] > resolved.loc[~same_day, "event_date"]).all()
    assert set(registry["day0_consistent_with_release"]) <= {"True", "unresolved"}
