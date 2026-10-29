import pandas as pd

from src.event_audit import audit_event_windows


def test_overlapping_event_windows_have_explicit_treatment():
    dates = pd.bdate_range("2024-01-02", periods=40)
    events = pd.DataFrame([
        {"event_id": "E1", "effective_event_date": dates[20], "episode_id": "A"},
        {"event_id": "E2", "effective_event_date": dates[23], "episode_id": "B"},
        {"event_id": "E3", "effective_event_date": dates[35], "episode_id": "C"},
    ])

    audit = audit_event_windows(events, dates, event_window_radius=2)
    rows = audit.set_index("event_id")

    assert rows.loc["E1", "event_window_overlap_count"] == 1
    assert rows.loc["E2", "event_window_overlap_count"] == 1
    assert bool(rows.loc["E1", "event_window_overlap_detected"])
    assert rows.loc["E1", "overlap_treatment"] == "exclude_from_category_summary_cluster_for_strategy_uncertainty"
    assert rows.loc["E3", "event_window_overlap_count"] == 0
    assert not bool(rows.loc["E3", "event_window_overlap_detected"])
