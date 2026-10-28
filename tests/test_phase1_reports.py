import json

import numpy as np
import pandas as pd
import pytest

from src.category_analysis import summarize_category_car, summarize_category_volatility
from src.event_audit import audit_documented_confounders, audit_event_windows
from src.volatility import summarize_event_volatility


def make_fixture():
    dates = pd.bdate_range("2024-01-02", periods=180)
    steps = np.arange(len(dates), dtype=float)
    market = 0.0002 * np.sin(steps * 0.3)
    returns = pd.DataFrame({
        "A": 0.001 + 1.2 * market + 0.0001 * np.cos(steps * 0.7),
        "B": 0.0005 + 0.8 * market + 0.0002 * np.sin(steps * 0.4),
        "NIFTY50_RETURN": market,
    }, index=dates)
    positions = {"E1": 60, "E2": 62, "E3": 110, "E4": 145}
    events = pd.DataFrame([
        {
            "event_id": event_id,
            "event_date": dates[position],
            "effective_event_date": dates[position],
            "category": "Monetary Policy" if event_id != "E4" else "Budget",
            "episode_id": {"E1": "EP1", "E2": "EP1", "E3": "EP2", "E4": "EP3"}[event_id],
        }
        for event_id, position in positions.items()
    ])
    models = {
        (event_id, stock): {
            "alpha": 0.0001,
            "beta": 1.1,
            "residual_std": 0.001,
            "observations": 100,
        }
        for event_id in positions
        for stock in ("A", "B")
    }
    return dates, returns, events, models


def test_realized_volatility_uses_explicit_windows_and_is_separate_from_residual_volatility():
    dates, returns, events, models = make_fixture()
    summary = summarize_event_volatility(returns, events.iloc[[0]], ["A"], models)
    row = summary.iloc[0]
    center = dates.get_loc(events.iloc[0].effective_event_date)
    expected_pre = returns["A"].iloc[center - 20 : center].std(ddof=1)
    expected_post = returns["A"].iloc[center + 1 : center + 21].std(ddof=1)
    assert row["pre_event_window"] == "-20:-1"
    assert row["post_event_window"] == "1:20"
    assert row["pre_event_observations"] == 20
    assert row["post_event_observations"] == 20
    assert row["pre_event_window_start_date"] == dates[center - 20]
    assert row["pre_event_window_end_date"] == dates[center - 1]
    assert row["post_event_window_start_date"] == dates[center + 1]
    assert row["post_event_window_end_date"] == dates[center + 20]
    assert row["pre_event_volatility_status"] == "minimum_met_descriptive"
    assert row["post_event_volatility_status"] == "minimum_met_descriptive"
    assert row["pre_event_realized_volatility_daily"] == pytest.approx(expected_pre)
    assert row["post_event_realized_volatility_daily"] == pytest.approx(expected_post)
    assert row["event_window_market_model_residual_volatility_daily"] != pytest.approx(
        row["post_event_realized_volatility_daily"]
    )
    assert "not realized" in row["residual_volatility_definition"]


def test_volatility_with_missing_observations_reports_counts_and_status():
    dates, returns, events, models = make_fixture()
    event_date = events.iloc[0].effective_event_date
    center = dates.get_loc(event_date)
    returns.loc[dates[center - 20 : center - 5], "A"] = np.nan
    returns.loc[dates[center + 2 : center + 17], "A"] = np.nan
    result = summarize_event_volatility(
        returns, events.iloc[[0]], ["A"], models, minimum_realized_observations=10
    ).iloc[0]
    assert result["pre_event_observations"] == 5
    assert result["post_event_observations"] == 5
    assert result["pre_event_volatility_status"] == "insufficient_observations"
    assert np.isnan(result["pre_event_realized_volatility_daily"])
    assert np.isnan(result["post_to_pre_realized_volatility_ratio"])


def test_truncated_volatility_windows_are_not_reported_as_reliable():
    dates, returns, events, models = make_fixture()
    edge_event = events.iloc[[0]].copy()
    edge_event.loc[:, "effective_event_date"] = dates[5]
    result = summarize_event_volatility(returns, edge_event, ["A"], models).iloc[0]
    assert result["pre_event_volatility_status"] == "truncated_at_data_boundary"
    assert np.isnan(result["pre_event_realized_volatility_daily"])


def test_volatility_window_definitions_must_be_pre_and_post_event():
    dates, returns, events, models = make_fixture()
    with pytest.raises(ValueError, match="end before Day 0"):
        summarize_event_volatility(
            returns, events.iloc[[0]], ["A"], models, pre_event_window=(-5, 0)
        )
    with pytest.raises(ValueError, match="start after Day 0"):
        summarize_event_volatility(
            returns, events.iloc[[0]], ["A"], models, post_event_window=(-1, 5)
        )


def test_event_window_audit_detects_car_and_wider_volatility_overlaps():
    dates, _, events, _ = make_fixture()
    audit = audit_event_windows(events, dates, event_window_radius=5).set_index("event_id")
    assert audit.loc["E1", "event_window_overlap_event_ids"] == "E2"
    assert audit.loc["E2", "event_window_overlap_event_ids"] == "E1"
    assert audit.loc["E1", "event_window_overlap_count"] == 1
    assert audit.loc["E1", "volatility_window_overlap_count"] == 1
    assert audit.loc["E1", "volatility_window_sessions"] == 40
    assert audit.loc["E3", "event_window_overlap_count"] == 0


def test_documented_confounder_flag_is_not_the_same_as_event_overlap():
    dates, _, events, _ = make_fixture()
    audit = audit_event_windows(events.iloc[[0]], dates)
    confounders = pd.DataFrame([{
        "confounder_id": "C1",
        "event_id": "E1",
        "date": dates[57],
        "headline": "Documented release",
        "source_name": "Test source",
        "source_url": "https://example.test/source",
        "potential_relevance": "Potential concurrent information",
        "provenance_note": "Synthetic test fixture",
    }])
    result = audit_documented_confounders(confounders, audit, dates).iloc[0]
    assert result["audit_type"] == "documented_potential_confounder"
    assert result["inside_event_window"]
    assert not result["causal_effect_estimated"]
    assert result["source_url"] == "https://example.test/source"
    assert audit.iloc[0]["event_window_overlap_count"] == 0


def test_confounded_record_preserves_all_source_fields_from_reference_json():
    with open("data/confounders.json", encoding="utf-8") as file:
        records = json.load(file)
    assert len(records) == 4
    assert all(record["source_name"] and record["source_url"] for record in records)
    assert all("not independently re-verified" in record["provenance_note"] for record in records)


def test_category_car_averages_stocks_then_collapses_related_events_by_episode():
    _, _, events, _ = make_fixture()
    audit = pd.DataFrame({
        "event_id": events.event_id,
        "event_window_overlap_count": [0, 0, 0, 1],
    })
    rows = []
    values = {"E1": [0.01, 0.03], "E2": [0.05, 0.07], "E3": [0.10, 0.12], "E4": [0.2, 0.22]}
    for event_id, stock_values in values.items():
        for stock, car in zip(("A", "B"), stock_values):
            rows.append({"event_id": event_id, "stock": stock, "window": "CAR_-1_+1", "CAR": car})
    result = summarize_category_car(events, pd.DataFrame(rows), audit).set_index("category")
    monetary = result.loc["Monetary Policy"]
    assert monetary["events_contributing"] == 3
    assert monetary["independent_episodes"] == 2
    assert monetary["excluded_overlapping_events"] == 0
    assert monetary["status"] == "insufficient_independent_episodes"
    assert np.isnan(monetary["mean_event_CAR_pct"])
    adequate_summary = summarize_category_car(
        events, pd.DataFrame(rows), audit, minimum_independent_episodes=2
    ).set_index(["category", "window"])
    adequate = adequate_summary.loc[("Monetary Policy", "CAR_-1_+1")]
    assert adequate["events_contributing"] == 3
    assert adequate["independent_episodes"] == 2
    assert adequate["status"] == "descriptive_summary"
    assert adequate["mean_event_CAR_pct"] == pytest.approx(7.5)
    assert adequate["category_event_count_in_master"] == 3
    assert adequate["stock_event_pairs_expected"] == 6
    assert adequate["stock_event_pairs_observed"] == 6
    assert adequate["stock_event_pairs_contributing"] == 6
    assert adequate["stock_event_pairs_missing"] == 0
    assert adequate["stock_event_pairs_incomplete"] == 0
    budget = result.loc["Budget"]
    assert budget["excluded_overlapping_events"] == 1
    assert budget["events_contributing"] == 0
    assert budget["stock_event_pairs_expected"] == 2
    assert budget["stock_event_pairs_excluded_overlap"] == 2
    assert budget["stock_event_pairs_contributing"] == 0


def test_category_volatility_counts_events_and_episodes_and_suppresses_sparse_means():
    _, _, events, _ = make_fixture()
    audit = pd.DataFrame({
        "event_id": events.event_id,
        "volatility_window_overlap_count": [0, 0, 0, 0],
    })
    rows = []
    for _, event in events.iterrows():
        for stock, pre, post in (("A", 0.20, 0.30), ("B", 0.10, 0.15)):
            rows.append({
                "event_id": event.event_id,
                "stock": stock,
                "pre_event_volatility_status": "minimum_met_descriptive",
                "post_event_volatility_status": "minimum_met_descriptive",
                "pre_event_realized_volatility_annualized": pre,
                "post_event_realized_volatility_annualized": post,
                "post_to_pre_realized_volatility_ratio": post / pre,
            })
    result = summarize_category_volatility(events, pd.DataFrame(rows), audit)
    monetary = result[
        (result.category == "Monetary Policy")
        & (result.metric == "post_event_realized_volatility_annualized_pct")
    ].iloc[0]
    assert monetary["events_contributing"] == 3
    assert monetary["independent_episodes"] == 2
    assert monetary["status"] == "insufficient_independent_episodes"
    assert np.isnan(monetary["mean_value"])
    adequate_summary = summarize_category_volatility(
        events, pd.DataFrame(rows), audit, minimum_independent_episodes=2
    )
    adequate = adequate_summary[
        (adequate_summary.category == "Monetary Policy")
        & (adequate_summary.metric == "post_event_realized_volatility_annualized_pct")
    ].iloc[0]
    assert adequate["events_contributing"] == 3
    assert adequate["independent_episodes"] == 2
    assert adequate["mean_value"] == pytest.approx(22.5)
    assert "percentage points" in adequate["metric_definition"]
    assert adequate["category_event_count_in_master"] == 3
    assert adequate["stock_event_pairs_expected"] == 6
    assert adequate["stock_event_pairs_observed"] == 6
    assert adequate["stock_event_pairs_contributing"] == 6
    assert adequate["stock_event_pairs_missing"] == 0


def test_category_car_reports_missing_and_incomplete_stock_event_pairs():
    _, _, events, _ = make_fixture()
    audit = pd.DataFrame({
        "event_id": events.event_id,
        "event_window_overlap_count": [0, 0, 0, 0],
    })
    rows = []
    values = {"E1": [0.01, 0.03], "E2": [0.05, 0.07], "E3": [0.10, 0.12]}
    for event_id, stock_values in values.items():
        for stock, car in zip(("A", "B"), stock_values):
            rows.append({"event_id": event_id, "stock": stock, "window": "CAR_-1_+1", "CAR": car})
    cars = pd.DataFrame(rows).drop(index=0).reset_index(drop=True)
    cars.loc[(cars.event_id == "E2") & (cars.stock == "A"), "CAR"] = np.nan

    result = summarize_category_car(
        events, cars, audit, minimum_independent_episodes=2
    ).set_index(["category", "window"])
    monetary = result.loc[("Monetary Policy", "CAR_-1_+1")]

    assert monetary["category_event_count_in_master"] == 3
    assert monetary["stock_event_pairs_expected"] == 6
    assert monetary["stock_event_pairs_observed"] == 5
    assert monetary["stock_event_pairs_contributing"] == 4
    assert monetary["stock_event_pairs_missing"] == 1
    assert monetary["stock_event_pairs_incomplete"] == 1


def test_category_volatility_reports_missing_and_insufficient_stock_event_pairs():
    _, _, events, _ = make_fixture()
    audit = pd.DataFrame({
        "event_id": events.event_id,
        "volatility_window_overlap_count": [0, 0, 0, 0],
    })
    rows = []
    for event_id in ("E1", "E2", "E3"):
        for stock in ("A", "B"):
            rows.append({
                "event_id": event_id,
                "stock": stock,
                "pre_event_volatility_status": "minimum_met_descriptive",
                "post_event_volatility_status": "minimum_met_descriptive",
                "pre_event_realized_volatility_annualized": 0.1,
                "post_event_realized_volatility_annualized": 0.2,
                "post_to_pre_realized_volatility_ratio": 2.0,
            })
    volatility = pd.DataFrame(rows).drop(index=0).reset_index(drop=True)
    volatility.loc[
        (volatility.event_id == "E2") & (volatility.stock == "A"),
        "pre_event_volatility_status",
    ] = "insufficient_observations"
    volatility.loc[
        (volatility.event_id == "E2") & (volatility.stock == "A"),
        "pre_event_realized_volatility_annualized",
    ] = np.nan

    result = summarize_category_volatility(
        events, volatility, audit, minimum_independent_episodes=2
    )
    monetary = result[
        (result.category == "Monetary Policy")
        & (result.metric == "pre_event_realized_volatility_annualized_pct")
    ].iloc[0]

    assert monetary["category_event_count_in_master"] == 3
    assert monetary["stock_event_pairs_expected"] == 6
    assert monetary["stock_event_pairs_observed"] == 5
    assert monetary["stock_event_pairs_contributing"] == 4
    assert monetary["stock_event_pairs_missing"] == 1
    assert monetary["stock_event_pairs_insufficient_sample"] == 1
