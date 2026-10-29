"""Estimation-window sensitivity for market-model event studies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .event_study import calculate_event_study
from .statistics import event_study_significance


def estimation_window_sensitivity(
    returns: pd.DataFrame,
    events: pd.DataFrame,
    stock_columns: list[str],
    window_lengths: tuple[int, ...] = (60, 100, 150),
    gap_sessions: int = 20,
    event_window_radius: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-fit all event/stock models at fixed alternative estimation lengths.

    Holm adjustment is computed independently within each estimation-window
    specification. This is a robustness comparison, not an additional search
    for a specification that maximizes statistical significance.
    """
    lengths = tuple(dict.fromkeys(int(length) for length in window_lengths))
    if not lengths or any(length < 3 for length in lengths):
        raise ValueError("window_lengths must contain integers of at least 3")
    if 100 not in lengths:
        raise ValueError("window_lengths must include the 100-session baseline")

    details: list[pd.DataFrame] = []
    baseline: pd.DataFrame | None = None
    for length in lengths:
        event_study, cars, models = calculate_event_study(
            returns, events, stock_columns,
            estimation_sessions=length,
            gap_sessions=gap_sessions,
            event_window_radius=event_window_radius,
        )
        inference = event_study_significance(event_study, cars, models)
        inference.insert(0, "estimation_sessions", length)
        details.append(inference)
        if length == 100:
            baseline = cars[["event_id", "stock", "window", "CAR"]].copy()

    result = pd.concat(details, ignore_index=True)
    if baseline is None:
        raise RuntimeError("Baseline CAR results were not generated")
    baseline = baseline.rename(columns={"CAR": "baseline_100_session_CAR"})
    result = result.merge(baseline, on=["event_id", "stock", "window"], how="left", validate="many_to_one")
    result["CAR_delta_vs_100_pct"] = (result["CAR_pct"] / 100.0 - result["baseline_100_session_CAR"]) * 100
    result["absolute_CAR_delta_vs_100_pct"] = result["CAR_delta_vs_100_pct"].abs()

    summaries: list[dict] = []
    for length, group in result.groupby("estimation_sessions", sort=True):
        valid = group.dropna(subset=["CAR_pct", "baseline_100_session_CAR"])
        baseline_pct = valid["baseline_100_session_CAR"].to_numpy(dtype=float) * 100
        current_pct = valid["CAR_pct"].to_numpy(dtype=float)
        correlation = (
            float(np.corrcoef(current_pct, baseline_pct)[0, 1])
            if len(valid) >= 2 and np.std(current_pct) > 0 and np.std(baseline_pct) > 0
            else float("nan")
        )
        summaries.append({
            "estimation_sessions": int(length),
            "tests_available": int(group["raw_p_value"].notna().sum()),
            "unique_events": int(group["event_id"].nunique()),
            "unique_stocks": int(group["stock"].nunique()),
            "conventional_holm_significant_tests": int(group["significant_after_holm_5pct"].sum()),
            "hac_holm_significant_tests": int(group["hac_significant_after_holm_5pct"].sum()),
            "median_absolute_CAR_delta_vs_100_pct": float(valid["absolute_CAR_delta_vs_100_pct"].median()) if not valid.empty else float("nan"),
            "mean_absolute_CAR_delta_vs_100_pct": float(valid["absolute_CAR_delta_vs_100_pct"].mean()) if not valid.empty else float("nan"),
            "CAR_correlation_vs_100_session": correlation,
            "comparison_basis": "same event-stock-window rows; Holm correction applied within each estimation-window specification",
        })
    return result, pd.DataFrame(summaries)
