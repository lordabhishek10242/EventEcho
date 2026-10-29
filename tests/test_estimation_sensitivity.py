import numpy as np
import pandas as pd
import pytest

from src.estimation_sensitivity import estimation_window_sensitivity


def _data():
    dates = pd.bdate_range("2022-01-03", periods=500)
    market = 0.008 * np.sin(np.arange(500) * 0.13) + 0.002 * np.cos(np.arange(500) * 0.03)
    stock_a = 0.0002 + 1.1 * market + 0.004 * np.sin(np.arange(500) * 0.47)
    stock_b = -0.0001 + 0.75 * market + 0.003 * np.cos(np.arange(500) * 0.29)
    returns = pd.DataFrame({"A": stock_a, "B": stock_b, "NIFTY50_RETURN": market}, index=dates)
    events = pd.DataFrame({
        "event_id": ["E1", "E2"],
        "event_date": [dates[300], dates[400]],
        "effective_event_date": [dates[300], dates[400]],
    })
    return returns, events


def test_estimation_window_sensitivity_includes_baseline_and_alternatives():
    returns, events = _data()
    details, summary = estimation_window_sensitivity(
        returns, events, ["A", "B"], window_lengths=(60, 100, 150)
    )
    assert set(details["estimation_sessions"]) == {60, 100, 150}
    assert len(details) == 2 * 2 * 3 * 3
    assert set(summary["estimation_sessions"]) == {60, 100, 150}
    assert (summary["tests_available"] == 12).all()
    base = details[details["estimation_sessions"] == 100]
    assert np.allclose(base["CAR_delta_vs_100_pct"], 0.0)
    assert base["conventional_holm_significant_tests"].sum() == base["significant_after_holm_5pct"].sum()


def test_estimation_window_sensitivity_requires_100_session_reference():
    returns, events = _data()
    with pytest.raises(ValueError):
        estimation_window_sensitivity(returns, events, ["A"], window_lengths=(60, 150))


def test_estimation_window_sensitivity_rejects_invalid_lengths():
    returns, events = _data()
    with pytest.raises(ValueError):
        estimation_window_sensitivity(returns, events, window_lengths=(60, 100, 2))
