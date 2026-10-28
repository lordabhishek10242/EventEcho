import numpy as np
import pandas as pd
import pytest

from src.event_study import calculate_event_study, estimation_sample, fit_market_model
from src.events import align_event_date


def make_returns(length=180):
    dates = pd.bdate_range("2023-01-02", periods=length)
    market = np.sin(np.arange(length) * 0.41) * 0.01 + np.cos(np.arange(length) * 0.13) * 0.004
    stock = 0.0007 + 1.25 * market
    return pd.DataFrame({"A": stock, "NIFTY50_RETURN": market}, index=dates)


def test_event_window_uses_sessions_and_separates_estimation_period():
    returns = make_returns()
    event_date = returns.index[140]
    sample = estimation_sample(returns, event_date, estimation_sessions=100, gap_sessions=20, stock_column="A")
    assert len(sample) == 100
    assert sample.index[-1] == returns.index[114]
    assert sample.index[-1] < returns.index[140 - 5 - 20]
    assert sample.index[0] == returns.index[15]


def test_non_trading_effective_date_is_explicitly_aligned_forward():
    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-03", "2024-01-04"])
    assert align_event_date("2024-01-02", dates) == pd.Timestamp("2024-01-03")
    assert align_event_date("2024-01-05", dates) is None


def test_abnormal_returns_and_car_match_known_synthetic_values():
    returns = make_returns()
    event_date = returns.index[140]
    abnormal = {day: day / 1000 for day in range(-5, 6)}
    abnormal[0] += 0.02
    for day, value in abnormal.items():
        date = returns.index[140 + day]
        returns.loc[date, "A"] += value
    events = pd.DataFrame({"event_id": ["E1"], "effective_event_date": [event_date]})
    event_study, cars, models = calculate_event_study(
        returns, events, ["A"], estimation_sessions=100, gap_sessions=20
    )
    assert len(event_study) == 11
    assert models[("E1", "A")]["observations"] == 100
    assert cars.set_index("window").loc["CAR_-1_+1", "CAR"] == pytest.approx(0.02, abs=1e-10)
    assert cars.set_index("window").loc["CAR_-3_+3", "CAR"] == pytest.approx(0.02, abs=1e-10)
    assert cars.set_index("window").loc["CAR_-5_+5", "CAR"] == pytest.approx(0.02, abs=1e-10)


def test_incomplete_event_window_does_not_report_partial_car():
    returns = make_returns()
    event_date = returns.index[140]
    returns.loc[returns.index[140], "A"] = np.nan
    events = pd.DataFrame({"event_id": ["E1"], "effective_event_date": [event_date]})
    _, cars, _ = calculate_event_study(returns, events, ["A"])
    assert cars["CAR"].isna().all()


def test_market_model_coefficients_recover_known_linear_relationship():
    sample = make_returns().iloc[:100]
    model = fit_market_model(sample, "A", "NIFTY50_RETURN")
    assert model["alpha"] == pytest.approx(0.0007, abs=1e-10)
    assert model["beta"] == pytest.approx(1.25, abs=1e-10)