"""Descriptive category summaries with events/episodes as the analysis unit."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _summary_row(
    category: str,
    events_contributing: int,
    episode_values: pd.Series,
    minimum_episodes: int,
    excluded_overlapping_events: int,
) -> dict:
    independent_episodes = int(episode_values.notna().sum())
    enough = independent_episodes >= minimum_episodes
    return {
        "category": category,
        "events_contributing": events_contributing,
        "independent_episodes": independent_episodes,
        "excluded_overlapping_events": excluded_overlapping_events,
        "minimum_independent_episodes": minimum_episodes,
        "status": "descriptive_summary" if enough else "insufficient_independent_episodes",
        "mean": float(episode_values.mean()) if enough else np.nan,
        "median": float(episode_values.median()) if enough else np.nan,
        "standard_deviation": float(episode_values.std(ddof=1)) if enough else np.nan,
    }


def summarize_category_car(
    events: pd.DataFrame,
    car_results: pd.DataFrame,
    event_window_audit: pd.DataFrame,
    minimum_independent_episodes: int = 3,
) -> pd.DataFrame:
    """Summarize CAR by category/window, averaging stocks within event first.

    Temporally overlapping CAR windows are excluded. Remaining event CARs are
    averaged within episode so related events do not count as independent
    observations. Each window is summarized separately; nested windows are never
    pooled as if independent.
    """
    if minimum_independent_episodes < 2:
        raise ValueError("minimum_independent_episodes must be at least two")
    metadata = events[["event_id", "category"] + (["episode_id"] if "episode_id" in events else [])].copy()
    if "episode_id" not in metadata:
        metadata["episode_id"] = metadata["event_id"]
    metadata["episode_id"] = metadata["episode_id"].fillna(metadata["event_id"]).astype(str)
    overlap = event_window_audit[["event_id", "event_window_overlap_count"]]
    expected_stocks = int(car_results["stock"].nunique())
    values = car_results.merge(metadata, on="event_id", validate="many_to_one").merge(
        overlap, on="event_id", validate="many_to_one"
    )
    output = []
    for category in sorted(metadata["category"].dropna().unique()):
        category_events = metadata[metadata["category"] == category]
        for window in sorted(car_results["window"].dropna().unique()):
            candidates = values[(values["category"] == category) & (values["window"] == window)]
            excluded_ids = set(candidates.loc[candidates["event_window_overlap_count"] > 0, "event_id"])
            usable = candidates[
                (candidates["event_window_overlap_count"] == 0) & candidates["CAR"].notna()
            ]
            event_values = usable.groupby(["event_id", "episode_id"], as_index=False).agg(
                event_mean_CAR=("CAR", "mean"),
                stocks_averaged=("stock", "nunique"),
            )
            episode_values = event_values.groupby("episode_id")["event_mean_CAR"].mean()
            row = _summary_row(
                category,
                int(event_values["event_id"].nunique()),
                episode_values,
                minimum_independent_episodes,
                len(excluded_ids),
            )
            mean_car = row.pop("mean")
            median_car = row.pop("median")
            car_std = row.pop("standard_deviation")
            row.update({
                "window": window,
                "mean_event_CAR_pct": mean_car * 100 if pd.notna(mean_car) else np.nan,
                "median_event_CAR_pct": median_car * 100 if pd.notna(median_car) else np.nan,
                "event_CAR_standard_deviation_pct": car_std * 100 if pd.notna(car_std) else np.nan,
                "stock_event_pairs_expected": int(category_events["event_id"].nunique()) * expected_stocks,
                "stock_event_pairs_observed": len(candidates),
                "stock_event_pairs_contributing": len(usable),
                "stock_event_pairs_missing": max(
                    int(category_events["event_id"].nunique()) * expected_stocks - len(candidates), 0
                ),
                "stock_event_pairs_incomplete": int(candidates["CAR"].isna().sum()),
                "stock_event_pairs_excluded_overlap": int(
                    (candidates["event_window_overlap_count"] > 0).sum()
                ),
                "aggregation_unit": "episode; stock CARs averaged within event, then event CARs averaged within episode",
                "category_event_count_in_master": int(category_events["event_id"].nunique()),
            })
            output.append(row)
    return pd.DataFrame(output)


def summarize_category_volatility(
    events: pd.DataFrame,
    volatility_results: pd.DataFrame,
    event_window_audit: pd.DataFrame,
    minimum_independent_episodes: int = 3,
) -> pd.DataFrame:
    """Summarize realized volatility by category using non-overlapping episodes.

    Stock values are averaged within event, then related event values are
    averaged within episode. Rows with inadequate pre/post samples or overlapping
    configured volatility windows are excluded from estimates and counted.
    """
    if minimum_independent_episodes < 2:
        raise ValueError("minimum_independent_episodes must be at least two")
    metadata = events[["event_id", "category"] + (["episode_id"] if "episode_id" in events else [])].copy()
    if "episode_id" not in metadata:
        metadata["episode_id"] = metadata["event_id"]
    metadata["episode_id"] = metadata["episode_id"].fillna(metadata["event_id"]).astype(str)
    overlap = event_window_audit[["event_id", "volatility_window_overlap_count"]]
    expected_stocks = int(volatility_results["stock"].nunique())
    values = volatility_results.drop(columns=["category", "episode_id"], errors="ignore").merge(
        metadata, on="event_id", validate="many_to_one"
    ).merge(
        overlap, on="event_id", validate="many_to_one"
    )
    reportable_status = {
        "minimum_met_descriptive",
        "minimum_met_descriptive_with_missing_sessions",
    }
    adequate = values[
        values["pre_event_volatility_status"].isin(reportable_status)
        & values["post_event_volatility_status"].isin(reportable_status)
        & values["pre_event_realized_volatility_annualized"].notna()
        & values["post_event_realized_volatility_annualized"].notna()
    ].copy()
    output = []
    for category in sorted(metadata["category"].dropna().unique()):
        category_metadata = metadata[metadata["category"] == category]
        category_values = values[values["category"] == category]
        for name, column in (
            ("pre_event_realized_volatility_annualized_pct", "pre_event_realized_volatility_annualized"),
            ("post_event_realized_volatility_annualized_pct", "post_event_realized_volatility_annualized"),
            ("post_to_pre_realized_volatility_ratio", "post_to_pre_realized_volatility_ratio"),
        ):
            scale = 100.0 if name.endswith("_pct") else 1.0
            overlapping_ids = set(
                category_values.loc[category_values["volatility_window_overlap_count"] > 0, "event_id"]
            )
            usable = adequate[
                (adequate["category"] == category)
                & (adequate["volatility_window_overlap_count"] == 0)
                & adequate[column].notna()
            ].assign(_summary_value=lambda frame: frame[column] * scale)
            event_values = usable.groupby(["event_id", "episode_id"], as_index=False).agg(
                event_value=("_summary_value", "mean"), stocks_averaged=("stock", "nunique")
            )
            episode_values = event_values.groupby("episode_id")["event_value"].mean()
            row = _summary_row(
                category,
                int(event_values["event_id"].nunique()),
                episode_values,
                minimum_independent_episodes,
                len(overlapping_ids),
            )
            mean_value = row.pop("mean")
            median_value = row.pop("median")
            standard_deviation = row.pop("standard_deviation")
            row.update({
                "metric": name,
                "mean_value": mean_value,
                "median_value": median_value,
                "episode_level_standard_deviation": standard_deviation,
                "stock_event_pairs_expected": int(category_metadata["event_id"].nunique()) * expected_stocks,
                "stock_event_pairs_observed": len(category_values),
                "stock_event_pairs_contributing": int(
                    (
                        (category_values["volatility_window_overlap_count"] == 0)
                        & category_values["pre_event_volatility_status"].isin(reportable_status)
                        & category_values["post_event_volatility_status"].isin(reportable_status)
                        & category_values[column].notna()
                    ).sum()
                ),
                "stock_event_pairs_missing": max(
                    int(category_metadata["event_id"].nunique()) * expected_stocks - len(category_values), 0
                ),
                "stock_event_pairs_insufficient_sample": int(
                    (
                        ~category_values["pre_event_volatility_status"].isin(reportable_status)
                        | ~category_values["post_event_volatility_status"].isin(reportable_status)
                    ).sum()
                ),
                "stock_event_pairs_excluded_overlap": int(
                    (category_values["volatility_window_overlap_count"] > 0).sum()
                ),
                "inadequate_sample_events": int(
                    category_values.loc[
                        ~category_values["pre_event_volatility_status"].isin(reportable_status)
                        | ~category_values["post_event_volatility_status"].isin(reportable_status),
                        "event_id",
                    ].nunique()
                ),
                "category_event_count_in_master": int(category_metadata["event_id"].nunique()),
                "aggregation_unit": "episode; stock volatility averaged within event, then events averaged within episode",
                "metric_definition": (
                    "annualized sample standard deviation of daily stock returns; sqrt(252); percentage points"
                    if "volatility_annualized" in name
                    else "ratio of post-event to pre-event realized daily standard deviations"
                ),
            })
            output.append(row)
    return pd.DataFrame(output)