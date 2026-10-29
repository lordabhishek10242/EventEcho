import numpy as np
import pandas as pd
import pytest

from src.clustered_inference import clustered_strategy_inference, _dependency_clusters


def _fixtures():
    events = pd.DataFrame({
        "event_id": ["E1", "E2", "E3", "E4", "E5", "E6"],
        "episode_id": ["POLICY_A", "POLICY_A", "E3", "E4", "E5", "E6"],
    })
    overlap = pd.DataFrame({
        "event_id": ["E1", "E2", "E3", "E4", "E5", "E6"],
        "event_window_overlap_event_ids": ["", "", "E4", "E3", "", ""],
    })
    returns = pd.DataFrame({
        "event_id": ["E1", "E2", "E3", "E4", "E5", "E6"],
        "portfolio_net_return": [0.01, 0.03, -0.02, 0.04, 0.02, -0.01],
        "benchmark_return": [0.00, 0.01, -0.01, 0.01, 0.00, 0.00],
    })
    return events, overlap, returns


def test_dependency_clusters_join_shared_episode_and_overlapping_windows():
    events, overlap, _ = _fixtures()
    clusters = _dependency_clusters(events, overlap)
    assert clusters["E1"] == clusters["E2"]
    assert clusters["E3"] == clusters["E4"]
    assert clusters["E5"] != clusters["E6"]
    assert len(set(clusters.values())) == 4


def test_cluster_bootstrap_outputs_reproducible_uncertainty_for_both_splits():
    events, overlap, returns = _fixtures()
    first = clustered_strategy_inference(
        returns, events, overlap, ("E1", "E2", "E5"), ("E3", "E4", "E6"),
        n_bootstrap=500, seed=42,
    )
    second = clustered_strategy_inference(
        returns, events, overlap, ("E1", "E2", "E5"), ("E3", "E4", "E6"),
        n_bootstrap=500, seed=42,
    )
    assert first["independent_dependency_clusters"].tolist() == [2, 2]
    assert first["events_available"].tolist() == [3, 3]
    assert first["inference_status"].eq("cluster_bootstrap_exploratory").all()
    np.testing.assert_allclose(first["mean_excess_return_pct_point"], [1.6666666667, 0.3333333333])
    np.testing.assert_allclose(
        first[["mean_excess_return_pct_ci95_lower", "mean_excess_return_pct_ci95_upper"]].to_numpy(),
        second[["mean_excess_return_pct_ci95_lower", "mean_excess_return_pct_ci95_upper"]].to_numpy(),
    )
    assert (first["mean_excess_return_pct_ci95_lower"] <= first["mean_excess_return_pct_point"]).all()
    assert (first["mean_excess_return_pct_ci95_upper"] >= first["mean_excess_return_pct_point"]).all()


def test_cluster_bootstrap_marks_single_cluster_as_insufficient():
    events, overlap, returns = _fixtures()
    result = clustered_strategy_inference(
        returns, events, overlap, ("E1", "E2"), ("E3",),
        n_bootstrap=100, seed=1,
    )
    holdout = result.set_index("sample").loc["Holdout"]
    assert holdout["independent_dependency_clusters"] == 1
    assert holdout["inference_status"] == "insufficient_independent_clusters"
    assert np.isnan(holdout["mean_excess_return_pct_ci95_lower"])


def test_cluster_bootstrap_rejects_too_few_resamples():
    events, overlap, returns = _fixtures()
    with pytest.raises(ValueError):
        clustered_strategy_inference(
            returns, events, overlap, ("E1",), ("E3",), n_bootstrap=99
        )
