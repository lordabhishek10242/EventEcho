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



def car_significance_hac(
    car: float,
    market_returns: list[float] | np.ndarray,
    model: dict,
) -> dict[str, float]:
    """Approximate CAR inference using Newey-West long-run residual variance.

    Estimation-window residuals estimate long-run daily abnormal-return
    variance. This is a serial-correlation-robust sensitivity estimate, not an
    exact finite-sample event-study test.
    """
    market = np.asarray(market_returns, dtype=float)
    residuals = np.asarray(model.get("estimation_residuals", []), dtype=float)
    if len(market) == 0 or not np.isfinite(car) or not np.isfinite(market).all():
        raise ValueError("CAR and finite event-window market returns are required")
    if len(residuals) < 3 or not np.isfinite(residuals).all():
        raise ValueError("At least three finite estimation residuals are required")
    n = len(residuals)
    bandwidth = min(n - 1, max(1, int(np.floor(4 * (n / 100.0) ** (2 / 9)))))
    centered = residuals - residuals.mean()
    gamma0 = float(centered @ centered / n)
    long_run_variance = gamma0
    for lag in range(1, bandwidth + 1):
        gamma = float(centered[lag:] @ centered[:-lag] / n)
        weight = 1.0 - lag / (bandwidth + 1.0)
        long_run_variance += 2.0 * weight * gamma
    long_run_variance = max(long_run_variance, 0.0)
    design = np.column_stack((np.ones(len(market)), market))
    covariance = np.asarray(model.get("hac_covariance", model["covariance"]), dtype=float)
    parameter_variance = float(design.sum(axis=0) @ covariance @ design.sum(axis=0))
    variance = max(parameter_variance + len(market) * long_run_variance, 0.0)
    standard_error = float(np.sqrt(variance))
    if standard_error == 0:
        raise ValueError("HAC CAR standard error is zero")
    t_statistic = float(car / standard_error)
    df = int(model["degrees_freedom"])
    critical = float(stats.t.ppf(0.975, df=df))
    return {
        "standard_error": standard_error,
        "t_statistic": t_statistic,
        "p_value": float(2 * stats.t.sf(abs(t_statistic), df=df)),
        "ci_lower_95": float(car - critical * standard_error),
        "ci_upper_95": float(car + critical * standard_error),
        "long_run_residual_variance": long_run_variance,
        "hac_bandwidth": float(bandwidth),
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
        model = models[(event_id, stock)]
        result = car_significance(
            float(car_row["CAR"]), group["event_day"].to_numpy(),
            group["market_return"].to_numpy(), model,
        )
        robust = car_significance_hac(
            float(car_row["CAR"]), group["market_return"].to_numpy(), model,
        )
        rows.append({
            "event_id": event_id, "stock": stock, "window": window_name,
            "CAR_pct": float(car_row["CAR"]) * 100,
            "standard_error_pct": result["standard_error"] * 100,
            "t_statistic": result["t_statistic"],
            "raw_p_value": result["p_value"],
            "ci_lower_pct": result["ci_lower_95"] * 100,
            "ci_upper_pct": result["ci_upper_95"] * 100,
            "hac_standard_error_pct": robust["standard_error"] * 100,
            "hac_t_statistic": robust["t_statistic"],
            "hac_raw_p_value": robust["p_value"],
            "hac_ci_lower_pct": robust["ci_lower_95"] * 100,
            "hac_ci_upper_pct": robust["ci_upper_95"] * 100,
            "hac_long_run_residual_variance": robust["long_run_residual_variance"],
            "hac_bandwidth": int(robust["hac_bandwidth"]),
        })
    output = add_holm_correction(pd.DataFrame(rows), "raw_p_value")
    output["hac_holm_p_value"] = holm_adjust(output["hac_raw_p_value"].to_numpy())
    output["hac_significant_after_holm_5pct"] = output["hac_holm_p_value"] < 0.05
    return output