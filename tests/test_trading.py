import pandas as pd
import pytest

from src.config import DEVELOPMENT_EVENT_IDS, HOLDOUT_EVENT_IDS
from src.events import chronological_split
from src.sensitivity import transaction_cost_sensitivity
from src.trading import simulate_trades


def test_trading_entry_exit_cost_and_benchmark_alignment():
    dates = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame({"A": [100, 110, 121, 133.1, 146.41, 161.051]}, index=dates)
    benchmark = pd.Series([0.01] * 6, index=dates)
    events = pd.DataFrame({"event_id": ["E1"], "effective_event_date": [dates[1]]})
    trades = simulate_trades(prices, benchmark, events, ["A"], holding_sessions=2, cost_bps_per_side=10)
    row = trades.iloc[0]
    assert row["entry_date"] == dates[2]
    assert row["exit_date"] == dates[4]
    assert row["gross_return"] == pytest.approx(0.21)
    assert row["net_return"] == pytest.approx(0.208)
    assert row["benchmark_return"] == pytest.approx((1.01**2) - 1)


def test_missing_benchmark_session_produces_missing_comparison():
    dates = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame({"A": [100, 101, 102, 103, 104, 105]}, index=dates)
    benchmark = pd.Series([0.01, 0.01, 0.01, float("nan"), 0.01, 0.01], index=dates)
    events = pd.DataFrame({"event_id": ["E1"], "effective_event_date": [dates[1]]})
    trades = simulate_trades(prices, benchmark, events, ["A"], holding_sessions=3, cost_bps_per_side=0)
    assert pd.isna(trades.iloc[0]["benchmark_return"])


def test_transaction_cost_sensitivity_charges_both_sides():
    trades = pd.DataFrame({
        "event_id": ["E1"], "stock": ["A"], "gross_return": [0.01],
        "net_return": [0.008], "benchmark_return": [0.0],
    })
    summary = transaction_cost_sensitivity(trades, (0, 10, 30)).set_index("cost_bps_per_side")
    assert summary.loc[0, "avg_net_return_pct"] == pytest.approx(1.0)
    assert summary.loc[10, "avg_net_return_pct"] == pytest.approx(0.8)
    assert summary.loc[30, "avg_net_return_pct"] == pytest.approx(0.4)


def test_missing_benchmark_observations_are_excluded_from_outperformance_rate():
    trades = pd.DataFrame({
        "event_id": ["E1", "E2"], "stock": ["A", "A"],
        "gross_return": [0.02, 0.02], "net_return": [0.02, 0.02],
        "benchmark_return": [0.01, float("nan")],
    })
    sensitivity = transaction_cost_sensitivity(trades, (0,))
    assert sensitivity.iloc[0]["benchmark_outperformance_rate_pct"] == pytest.approx(100.0)


def test_chronological_split_preserves_original_event_membership():
    events = pd.DataFrame({
        "event_id": list(DEVELOPMENT_EVENT_IDS + HOLDOUT_EVENT_IDS),
        "event_date": pd.date_range("2016-01-01", periods=9),
    })
    result = chronological_split(events, DEVELOPMENT_EVENT_IDS, HOLDOUT_EVENT_IDS)
    labels = result.set_index("event_id")["sample"]
    assert labels["EVT007"] == "Development"
    assert labels["EVT008"] == "Holdout"
    assert labels["EVT006"] == "Holdout"


def test_split_rejects_missing_or_overlapping_ids():
    events = pd.DataFrame({"event_id": ["E1", "E2"]})
    with pytest.raises(ValueError, match="overlap"):
        chronological_split(events, ["E1"], ["E1", "E2"])
    with pytest.raises(ValueError, match="exactly one"):
        chronological_split(events, ["E1"], [])


def test_split_rejects_non_chronological_membership():
    events = pd.DataFrame({
        "event_id": ["E1", "E2"],
        "effective_event_date": pd.to_datetime(["2024-02-01", "2024-01-01"]),
    })
    with pytest.raises(ValueError, match="chronologically"):
        chronological_split(events, ["E1"], ["E2"])