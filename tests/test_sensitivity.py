import numpy as np
import pandas as pd

from src.sensitivity import extreme_market_sensitivity


def test_excluding_extreme_days_does_not_label_partial_sum_as_car():
    frame = pd.DataFrame({
        "event_id": ["EVT002"] * 3,
        "stock": ["ITC"] * 3,
        "event_day": [-1, 0, 1],
        "market_return": [0.01, 0.06, -0.02],
        "abnormal_return": [0.01, 0.02, 0.03],
    })

    result = extreme_market_sensitivity(frame, "EVT002", threshold=0.05)
    row = result[result["window"] == "CAR_-1_+1"].iloc[0]

    assert np.isclose(row["original_CAR"], 0.06)
    assert np.isnan(row["sensitivity_CAR"])
    assert np.isclose(row["retained_abnormal_return_sum"], 0.04)
    assert np.isclose(row["retained_mean_abnormal_return"], 0.02)
    assert row["extreme_days_excluded"] == 1
    assert row["sessions_retained"] == 2
    assert row["sensitivity_status"] == "partial_window_not_comparable"


def test_no_extreme_days_preserves_full_window_car():
    frame = pd.DataFrame({
        "event_id": ["EVT001"] * 3,
        "stock": ["RELIANCE"] * 3,
        "event_day": [-1, 0, 1],
        "market_return": [0.01, -0.02, 0.03],
        "abnormal_return": [0.01, 0.02, -0.01],
    })

    result = extreme_market_sensitivity(frame, "EVT001", threshold=0.05)
    row = result[result["window"] == "CAR_-1_+1"].iloc[0]

    assert np.isclose(row["original_CAR"], row["sensitivity_CAR"])
    assert row["extreme_days_excluded"] == 0
    assert row["sessions_retained"] == 3
    assert row["sensitivity_status"] == "complete_window_no_extreme_days_excluded"


def test_incomplete_event_window_is_not_reported_as_car():
    frame = pd.DataFrame({
        "event_id": ["EVT001"] * 2,
        "stock": ["TCS"] * 2,
        "event_day": [-1, 1],
        "market_return": [0.01, 0.02],
        "abnormal_return": [0.01, 0.03],
    })

    result = extreme_market_sensitivity(frame, "EVT001")
    row = result[result["window"] == "CAR_-1_+1"].iloc[0]

    assert np.isnan(row["original_CAR"])
    assert np.isnan(row["sensitivity_CAR"])
    assert row["sensitivity_status"] == "incomplete_original_window"
