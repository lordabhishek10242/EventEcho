import numpy as np
import pandas as pd
import pytest

from src.volatility import calculate_realized_volatility, summarize_event_volatility


def test_constant_price_series_has_zero_realized_volatility():
    prices = pd.Series([100.0] * 5, index=pd.bdate_range("2024-01-01", periods=5))

    result = calculate_realized_volatility(prices)

    assert result["observations"] == 4
    assert result["daily_volatility"] == pytest.approx(0.0)
    assert result["annualized_volatility"] == pytest.approx(0.0)
    assert result["status"] == "minimum_met_descriptive"


def test_known_simple_returns_have_explicit_sample_standard_deviation():
    prices = pd.Series([100.0, 110.0, 99.0], index=pd.bdate_range("2024-01-01", periods=3))

    result = calculate_realized_volatility(prices)

    assert result["daily_volatility"] == pytest.approx(np.sqrt(0.02))
    assert result["annualized_volatility"] == pytest.approx(np.sqrt(0.02 * 252))
    assert result["observations"] == 2


def test_missing_prices_are_not_imputed_and_insufficient_sample_is_nan():
    prices = pd.Series(
        [100.0, 110.0, np.nan, 121.0, 133.1],
        index=pd.bdate_range("2024-01-01", periods=5),
    )

    result = calculate_realized_volatility(prices, minimum_observations=3)

    assert result["observations"] == 2
    assert np.isnan(result["daily_volatility"])
    assert np.isnan(result["annualized_volatility"])
    assert result["status"] == "insufficient_observations"


def test_event_windows_use_effective_session_exclude_day_zero_and_report_benchmark():
    dates = pd.bdate_range("2024-01-01", periods=35)
    effective_dates = [dates[10], dates[22]]
    returns = pd.DataFrame(
        {
            "A": np.linspace(-0.02, 0.03, len(dates)),
            "B": np.linspace(0.04, -0.01, len(dates)),
            "NIFTY50_RETURN": np.sin(np.arange(len(dates))) / 100,
        },
        index=dates,
    )
    events = pd.DataFrame(
        {
            "event_id": ["E1", "E2"],
            "event_date": [effective_dates[0] - pd.Timedelta(days=2), effective_dates[1]],
            "effective_event_date": effective_dates,
        }
    )

    result = summarize_event_volatility(
        returns,
        events,
        ["A", "B"],
        minimum_realized_observations=2,
        pre_event_window=(-3, -1),
        post_event_window=(1, 3),
    )

    assert len(result) == 4
    first = result[(result.event_id == "E1") & (result.stock == "A")].iloc[0]
    assert first["effective_event_date"] == effective_dates[0]
    assert first["pre_event_window"] == "-3:-1"
    assert first["post_event_window"] == "1:3"
    assert first["pre_event_window_start_date"] == dates[7]
    assert first["pre_event_window_end_date"] == dates[9]
    assert first["post_event_window_start_date"] == dates[11]
    assert first["post_event_window_end_date"] == dates[13]
    assert first["pre_event_observations"] == 3
    assert first["post_event_observations"] == 3
    assert first["benchmark_pre_event_observations"] == 3
    assert first["benchmark_post_event_observations"] == 3
    assert first["benchmark_pre_event_realized_volatility_daily"] == pytest.approx(
        returns["NIFTY50_RETURN"].iloc[7:10].std(ddof=1)
    )
    assert first["pre_event_window_end_date"] < effective_dates[0]
    assert first["post_event_window_start_date"] > effective_dates[0]


def test_zero_pre_event_volatility_keeps_absolute_change_but_suppresses_percent_change():
    dates = pd.bdate_range("2024-01-01", periods=15)
    center = 7
    stock_returns = np.zeros(len(dates))
    stock_returns[center + 1 : center + 4] = [0.01, -0.01, 0.02]
    returns = pd.DataFrame(
        {"A": stock_returns, "NIFTY50_RETURN": np.zeros(len(dates))}, index=dates
    )
    events = pd.DataFrame(
        {"event_id": ["E1"], "effective_event_date": [dates[center]]}
    )

    result = summarize_event_volatility(
        returns,
        events,
        ["A"],
        minimum_realized_observations=2,
        pre_event_window=(-3, -1),
        post_event_window=(1, 3),
    ).iloc[0]

    assert result["pre_event_realized_volatility_daily"] == pytest.approx(0.0)
    assert result["realized_volatility_absolute_change_daily"] > 0
    assert np.isnan(result["realized_volatility_percentage_change"])
    assert np.isnan(result["post_to_pre_realized_volatility_ratio"])
    assert result["volatility_comparison_status"] == "zero_pre_event_volatility"


def test_missing_event_window_returns_are_counted_without_zero_imputation():
    dates = pd.bdate_range("2024-01-01", periods=15)
    center = 7
    returns = pd.DataFrame(
        {"A": np.linspace(0.0, 0.014, len(dates)), "NIFTY50_RETURN": 0.001},
        index=dates,
    )
    returns.loc[dates[center - 3], "A"] = np.nan
    events = pd.DataFrame(
        {"event_id": ["E1"], "effective_event_date": [dates[center]]}
    )

    row = summarize_event_volatility(
        returns,
        events,
        ["A"],
        minimum_realized_observations=3,
        pre_event_window=(-3, -1),
        post_event_window=(1, 3),
    ).iloc[0]

    assert row["pre_event_observations"] == 2
    assert row["pre_event_expected_sessions"] == 3
    assert row["pre_event_volatility_status"] == "insufficient_observations"
    assert np.isnan(row["pre_event_realized_volatility_daily"])
    assert row["post_event_volatility_status"] == "minimum_met_descriptive"
