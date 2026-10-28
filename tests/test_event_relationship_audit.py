import pandas as pd

from src.event_audit import audit_event_windows, build_event_relationship_audits
from src.events import align_event_date


def run_audit(events, dates, confounder_audit=None, radius=5):
    windows = audit_event_windows(events, dates, event_window_radius=radius)
    return build_event_relationship_audits(
        events,
        windows,
        pd.DataFrame() if confounder_audit is None else confounder_audit,
        dates,
        event_window_radius=radius,
    )


def event(event_id, day, *, announcement_date=None, category="Policy", episode="EP1", **extra):
    return {
        "event_id": event_id,
        "event_date": announcement_date or day,
        "effective_event_date": day,
        "headline": f"Description for {event_id}",
        "category": category,
        "episode_id": episode,
        "source_name": f"Source {event_id}",
        "source_url": f"https://example.test/{event_id}",
        "timing_confidence": "date verified",
        "verification_status": "recorded",
        "notes": "",
        **extra,
    }


def test_temporal_overlap_counts_actual_sessions_and_unique_pair_once():
    dates = pd.bdate_range("2024-01-01", periods=50)
    events = pd.DataFrame([
        event("E1", dates[10], category="A", episode="P1"),
        event("E2", dates[15], category="B", episode="P2"),
        event("E3", dates[35], category="C", episode="P3"),
    ])

    event_rows, pairs, report = run_audit(events, dates)

    assert len(pairs) == 3
    assert not pairs.duplicated(["event_id_a", "event_id_b"]).any()
    overlap = pairs[(pairs.event_id_a == "E1") & (pairs.event_id_b == "E2")].iloc[0]
    assert overlap["temporal_overlap"]
    assert overlap["overlapping_session_count"] == 6
    assert overlap["overlapping_session_start"] == dates[10]
    assert overlap["overlapping_session_end"] == dates[15]
    assert not pairs[(pairs.event_id_a == "E1") & (pairs.event_id_b == "E3")].iloc[0]["temporal_overlap"]
    assert event_rows.set_index("event_id").loc["E1", "temporal_overlap_partner_count"] == 1
    assert event_rows.set_index("event_id").loc["E2", "temporal_overlap_partner_count"] == 1
    assert len(report) == 3


def test_non_temporal_shared_episode_is_policy_clustering_not_temporal_overlap():
    dates = pd.bdate_range("2024-01-01", periods=70)
    events = pd.DataFrame([
        event("E1", dates[12], category="Monetary Policy", episode="RATE_CYCLE"),
        event("E2", dates[50], category="Monetary Policy", episode="RATE_CYCLE"),
    ])

    _, pairs, _ = run_audit(events, dates, radius=3)

    pair = pairs.iloc[0]
    assert not pair["temporal_overlap"]
    assert pair["overlapping_session_count"] == 0
    assert pair["shared_episode_id"]
    assert pair["same_category"]
    assert "shared_episode_policy_clustering" in pair["overlap_types"]
    assert "same_category_clustering" in pair["overlap_types"]
    assert "no causal relationship" in pair["explanation"]


def test_missing_document_provenance_is_not_labeled_documented_and_caution_stays_unverified():
    dates = pd.bdate_range("2024-01-01", periods=40)
    events = pd.DataFrame([
        event(
            "E1", dates[15], announcement_date=pd.Timestamp("2024-01-13"),
            timing_confidence="exact time pending", verification_status="unverified",
            notes="Verify the effective trading session.",
        ),
        event("E2", dates[30], category="Other", episode="P2", notes="Review source timing."),
    ])
    confounders = pd.DataFrame([{
        "event_id": "E1",
        "confounder_id": "C1",
        "confounder_date": dates[15],
        "inside_event_window": True,
        "headline": "Unverified record",
        "source_name": "",
        "source_url": "",
        "provenance_note": "",
    }])

    event_rows, pairs, _ = run_audit(events, dates, confounders, radius=2)

    audited = event_rows.set_index("event_id").loc["E1"]
    assert audited["event_date"] == pd.Timestamp("2024-01-13")
    assert audited["effective_event_date"] == dates[15]
    assert audited["unverified_possibility_flag"]
    assert not audited["documented_potential_confounder_flag"]
    assert audited["documented_potential_development_count"] == 0
    assert "unverified_review_possibility" in audited["potential_confounder_flags"]
    assert "unverified possibility" in audited["potential_confounder_explanation"]
    assert pairs.iloc[0]["documented_potential_development_count"] == 0
    assert event_rows.set_index("event_id").loc["E2", "unverified_possibility_flag"]


def test_sourced_potential_development_is_preserved_in_event_and_pair_audits():
    dates = pd.bdate_range("2024-01-01", periods=45)
    events = pd.DataFrame([
        event("E1", dates[15], category="Policy A", episode="P1"),
        event("E2", dates[17], category="Policy B", episode="P2"),
    ])
    source_record = pd.DataFrame([{
        "event_id": "E1",
        "confounder_id": "C1",
        "confounder_date": dates[16],
        "inside_event_window": True,
        "headline": "Potential concurrent development",
        "source_name": "Recorded source",
        "source_url": "https://example.test/development",
        "provenance_note": "Recorded provenance; not independently verified.",
    }])

    event_rows, pairs, _ = run_audit(events, dates, source_record, radius=3)

    audited = event_rows.set_index("event_id").loc["E1"]
    pair = pairs.iloc[0]
    assert audited["documented_potential_confounder_flag"]
    assert audited["documented_potential_development_in_window_count"] == 1
    assert "https://example.test/development" in audited["documented_potential_sources"]
    assert "not independently verified" in audited["documented_potential_provenance"]
    assert pair["documented_potential_development_in_shared_sessions"]
    assert pair["documented_potential_development_count"] == 1
    assert pair["documented_source_url"] == "https://example.test/development"


def test_saturday_announcement_uses_existing_effective_session_without_shifting_announcement():
    dates = pd.bdate_range("2025-01-27", periods=20)
    saturday = pd.Timestamp("2025-02-01")
    monday = pd.Timestamp("2025-02-03")
    events = pd.DataFrame([
        event("BUDGET", monday, announcement_date=saturday, category="Budget", episode="BUDGET_2025")
    ])
    assert align_event_date(saturday, dates) == monday

    event_rows, pairs, report = run_audit(events, dates, radius=2)

    audited = event_rows.iloc[0]
    assert audited["event_date"] == saturday
    assert audited["effective_event_date"] == monday
    assert audited["event_window_start"] == dates[dates.get_loc(monday) - 2]
    assert audited["event_window_end"] == dates[dates.get_loc(monday) + 2]
    assert pairs.empty
    assert len(report) == 1


def test_empty_and_single_event_inputs_are_supported():
    dates = pd.bdate_range("2024-01-01", periods=20)
    empty = pd.DataFrame(columns=["event_id", "effective_event_date"])
    event_rows, pairs, report = run_audit(empty, dates)
    assert event_rows.empty and pairs.empty and report.empty

    single = pd.DataFrame([event("E1", dates[10])])
    event_rows, pairs, report = run_audit(single, dates)
    assert len(event_rows) == 1
    assert pairs.empty
    assert len(report) == 1
