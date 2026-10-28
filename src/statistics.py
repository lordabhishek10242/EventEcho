"""CAR inference with approximate parameter-aware variance and Holm correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def holm_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, preserving original input order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.isnan(values).any() or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be a one-dimensional array in [0, 1]")
    count = len(values)
    if not count:
        return values.copy()
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    adjusted_sorted = np.maximum.accumulate((count - np.arange(count)) * sorted_values)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def car_significance(
    car: float,
    event_days: list[int] | np.ndarray,
    market_returns: list[float] | np.ndarray,
    model: dict,
) -> dict[str, float]:
    """Approximate CAR test including OLS parameter uncertainty.

    Residuals are still assumed independent and homoskedastic over the event
    window; this is a diagnostic rather than definitive event-study inference.
    """
    days = np.asarray(event_days, dtype=int)
    market = np.asarray(market_returns, dtype=float)
    if len(days) != len(market) or len(days) == 0 or not np.isfinite(car):
        raise ValueError("CAR and non-empty event-window observations are required")
    design = np.column_stack((np.ones(len(market)), market))
    covariance = np.asarray(model["covariance"], dtype=float)
    parameter_variance = float(design.sum(axis=0) @ covariance @ design.sum(axis=0))
    residual_variance = len(days) * float(model["residual_variance"])
    standard_error = float(np.sqrt(max(parameter_variance + residual_variance, 0.0)))
    if standard_error == 0:
        raise ValueError("CAR standard error is zero")
    t_statistic = float(car / standard_error)
    df = int(model["degrees_freedom"])
    p_value = float(2 * stats.t.sf(abs(t_statistic), df=df))
    critical = float(stats.t.ppf(0.975, df=df))
    return {
        "standard_error": standard_error,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "ci_lower_95": float(car - critical * standard_error),
        "ci_upper_95": float(car + critical * standard_error),
    }


def add_holm_correction(results: pd.DataFrame, p_column: str = "p_value") -> pd.DataFrame:
    output = results.copy()
    output["holm_p_value"] = holm_adjust(output[p_column].to_numpy())
    output["significant_after_holm_5pct"] = output["holm_p_value"] < 0.05
    return output


def event_study_significance(
    event_study: pd.DataFrame,
    car_results: pd.DataFrame,
    models: dict[tuple[str, str], dict],
) -> pd.DataFrame:
    """Test complete per-stock CAR windows and correct across all generated tests."""
    rows = []
    for _, car_row in car_results.dropna(subset=["CAR"]).iterrows():
        event_id, stock, window_name = car_row["event_id"], car_row["stock"], car_row["window"]
        lower, upper = {"CAR_-1_+1": (-1, 1), "CAR_-3_+3": (-3, 3), "CAR_-5_+5": (-5, 5)}[window_name]
        group = event_study[
            (event_study["event_id"] == event_id)
            & (event_study["stock"] == stock)
            & event_study["event_day"].between(lower, upper)
        ].sort_values("event_day")
        result = car_significance(
            float(car_row["CAR"]),
            group["event_day"].to_numpy(),
            group["market_return"].to_numpy(),
            models[(event_id, stock)],
        )
        rows.append({
            "event_id": event_id, "stock": stock, "window": window_name,
            "CAR_pct": float(car_row["CAR"]) * 100,
            "standard_error_pct": result["standard_error"] * 100,
            "t_statistic": result["t_statistic"],
            "raw_p_value": result["p_value"],
            "ci_lower_pct": result["ci_lower_95"] * 100,
            "ci_upper_pct": result["ci_upper_95"] * 100,
        })
    return add_holm_correction(pd.DataFrame(rows), "raw_p_value")