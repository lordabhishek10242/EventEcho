import numpy as np
import pandas as pd
import pytest

from src.data import align_stock_benchmark_returns, calculate_returns


def test_simple_returns_and_missing_prices_are_not_filled():
    prices = pd.DataFrame({"A": [100.0, 110.0, np.nan, 99.0]}, index=pd.date_range("2024-01-01", periods=4))
    result = calculate_returns(prices)
    assert result.iloc[1, 0] == pytest.approx(0.10)
    assert np.isnan(result.iloc[2, 0])
    assert np.isnan(result.iloc[3, 0])


def test_cumulative_return_compounds_simple_returns():
    returns = pd.Series([0.10, -0.10])
    compounded = (1 + returns).prod() - 1
    assert compounded == pytest.approx(-0.01)


def test_benchmark_alignment_does_not_drop_other_stock_for_one_missing_value():
    dates = pd.date_range("2024-01-01", periods=3)
    stocks = pd.DataFrame({"A": [0.01, np.nan, 0.02], "B": [0.02, 0.03, 0.04]}, index=dates)
    benchmark = pd.Series([0.0, 0.01, 0.0], index=dates)
    aligned = align_stock_benchmark_returns(stocks, benchmark)
    assert len(aligned) == 3
    assert np.isnan(aligned.loc[dates[1], "A"])
    assert aligned.loc[dates[1], "B"] == pytest.approx(0.03)


def test_duplicate_price_dates_are_rejected():
    dates = pd.to_datetime(["2024-01-01", "2024-01-01"])
    with pytest.raises(ValueError, match="duplicate"):
        calculate_returns(pd.Series([10, 11], index=dates))


@pytest.mark.parametrize("bad_price", [0.0, -1.0, np.inf, -np.inf])
def test_nonpositive_or_infinite_prices_are_rejected(bad_price):
    prices = pd.Series([100.0, bad_price], index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="finite and strictly positive"):
        calculate_returns(prices)