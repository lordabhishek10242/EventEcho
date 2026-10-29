import pandas as pd

from src.robustness_ledger import build_robustness_evidence_ledger


def test_ledger_identifies_stable_significance_and_adds_overlap_timing_caveats():
    details = pd.DataFrame([
        {"estimation_sessions": length, "event_id": event, "stock": stock, "window": "CAR_-1_+1",
         "CAR_pct": car, "hac_significant_after_holm_5pct": sig, "significant_after_holm_5pct": sig,
         "absolute_CAR_delta_vs_100_pct": delta}
        for event, stock, values in [
            ("E1", "A", [(60, -4.0, True, 0.5), (100, -4.2, True, 0.0), (150, -4.1, True, 0.1)]),
            ("E2", "B", [(60, 1.0, True, 0.2), (100, 0.8, False, 0.0), (150, 0.9, False, 0.1)]),
            ("E3", "C", [(60, 0.1, False, 0.2), (100, -0.1, False, 0.0), (150, 0.05, False, 0.05)]),
        ]
        for length, car, sig, delta in values
    ])
    events = pd.DataFrame([
        {"event_id": "E1", "headline": "Overlap event", "category": "Policy", "episode_id": "EP1",
         "timing_confidence": "time_unverified", "verification_status": "Review source"},
        {"event_id": "E2", "headline": "Sensitive event", "category": "Policy", "episode_id": "EP2",
         "timing_confidence": "verified", "verification_status": "verified"},
        {"event_id": "E3", "headline": "Null event", "category": "Policy", "episode_id": "EP3",
         "timing_confidence": "verified", "verification_status": "verified"},
    ])
    windows = pd.DataFrame([
        {"event_id": "E1", "event_window_overlap_count": 1, "event_window_overlap_event_ids": "E2"},
        {"event_id": "E2", "event_window_overlap_count": 1, "event_window_overlap_event_ids": "E1"},
        {"event_id": "E3", "event_window_overlap_count": 0, "event_window_overlap_event_ids": ""},
    ])
    episodes = pd.DataFrame([
        {"event_id": "E1", "shared_episode_partner_count": 1, "shared_episode_partner_ids": "E2"},
        {"event_id": "E2", "shared_episode_partner_count": 0, "shared_episode_partner_ids": ""},
        {"event_id": "E3", "shared_episode_partner_count": 0, "shared_episode_partner_ids": ""},
    ])

    result = build_robustness_evidence_ledger(details, events, windows, episodes).set_index("event_id")
    assert result.loc["E1", "hac_significant_at_all_estimation_windows"]
    assert result.loc["E1", "review_status"] == "Stable statistical signal; attribution/timing caveat"
    assert result.loc["E2", "review_status"] == "No stable HAC significance across all tested windows"
    assert result.loc["E2", "hac_holm_significant_window_count"] == 1
    assert result.loc["E3", "car_direction_stable_across_estimation_windows"] is False or not result.loc["E3", "car_direction_stable_across_estimation_windows"]
