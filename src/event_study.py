"""Trading-session market models and abnormal-return event windows."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .config import CAR_WINDOWS


def _session_position(dates: pd.DatetimeIndex, target: object) -> int:
    normalized = pd.Timestamp(target).tz_localize(None).normalize()
    position = dates.get_indexer([normalized])[0]
    if position < 0:
        raise ValueError(f"Effective event date {normalized.date()} is not a trading session")
    return int(position)


def estimation_sample(
    returns: pd.DataFrame,
    event_date: object,
    estimation_sessions: int = 100,
    gap_sessions: int = 20,
    stock_column: str | None = None,
    market_column: str = "NIFTY50_RETURN",
    event_window_radius: int = 5,
) -> pd.DataFrame:
    """Select N complete sessions, leaving gap_sessions before the event window.

    The omitted sessions are the final gap_sessions trading dates before the
    event window's first session (Day -5 by default).
    """
    dates = pd.DatetimeIndex(returns.index).tz_localize(None).normalize().unique().sort_values()
    event_position = _session_position(dates, event_date)
    event_window_start = event_position - event_window_radius
    end_position = event_window_start - gap_sessions - 1
    if end_position < 0:
        raise ValueError("Insufficient history for requested estimation window")
    columns = [market_column] if stock_column is None else [stock_column, market_column]

    # Missing prices can make some return rows incomplete. Preserve the event
    # gap, then look farther back as needed to obtain the requested number of
    # complete observations instead of failing on a fixed-width calendar slice.
    eligible = returns.loc[dates[: end_position + 1], columns].dropna()
    if len(eligible) < estimation_sessions:
        raise ValueError(
            f"Expected {estimation_sessions} complete estimation observations; got {len(eligible)}"
        )
    return eligible.tail(estimation_sessions)


def fit_market_model(sample: pd.DataFrame, stock_column: str, market_column: str) -> dict:
    """Fit OLS with intercept and return coefficients plus residual diagnostics."""
    clean = sample[[stock_column, market_column]].dropna()
    if len(clean) < 3:
        raise ValueError("At least three complete observations are required")
    x = np.column_stack((np.ones(len(clean)), clean[market_column].to_numpy(dtype=float)))
    y = clean[stock_column].to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != 2:
        raise ValueError("Market model is rank deficient")
    residuals = y - x @ coefficients
    degrees_freedom = len(y) - x.shape[1]
    residual_variance = float(residuals @ residuals / degrees_freedom)
    bread = np.linalg.inv(x.T @ x)
    covariance = residual_variance * bread

    # Newey-West sandwich covariance for the estimated market-model coefficients.
    n = len(y)
    bandwidth = min(n - 1, max(1, int(np.floor(4 * (n / 100.0) ** (2 / 9)))))
    scores = x * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)
    hac_covariance = bread @ meat @ bread
    return {
        "alpha": float(coefficients[0]),
        "beta": float(coefficients[1]),
        "residual_variance": residual_variance,
        "residual_std": float(np.sqrt(residual_variance)),
        "degrees_freedom": degrees_freedom,
        "covariance": covariance,
        "hac_covariance": hac_covariance,
        "hac_bandwidth": bandwidth,
        "estimation_residuals": residuals.tolist(),
        "observations": len(clean),
    }


def calculate_event_study(
    returns: pd.DataFrame,
    events: pd.DataFrame,
    stock_columns: list[str],
    estimation_sessions: int = 100,
    gap_sessions: int = 20,
    event_window_radius: int = 5,
    market_column: str = "NIFTY50_RETURN",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], dict]]:
    """Return AR rows, CAR rows and per-(event, stock) model diagnostics.

    Each stock is processed independently; an unavailable observation for one
    stock does not remove another stock's event-window observation.
    """
    if not returns.index.is_monotonic_increasing:
        returns = returns.sort_index()
    dates = pd.DatetimeIndex(returns.index).tz_localize(None).normalize().unique().sort_values()
    event_rows: list[dict] = []
    car_rows: list[dict] = []
    models: dict[tuple[str, str], dict] = {}
    for _, event in events.iterrows():
        event_id = str(event["event_id"])
        effective_date = pd.Timestamp(event["effective_event_date"]).tz_localize(None).normalize()
        event_position = _session_position(dates, effective_date)
        for stock in stock_columns:
            if stock not in returns or market_column not in returns:
                continue
            sample = estimation_sample(
                returns, effective_date, estimation_sessions, gap_sessions,
                stock, market_column, event_window_radius,
            )
            model = fit_market_model(sample, stock, market_column)
            models[(event_id, stock)] = model
            start = event_position - event_window_radius
            end = event_position + event_window_radius
            if start < 0 or end >= len(dates):
                continue
            for event_day in range(-event_window_radius, event_window_radius + 1):
                trading_date = dates[event_position + event_day]
                actual = returns.at[trading_date, stock]
                market = returns.at[trading_date, market_column]
                if pd.isna(actual) or pd.isna(market):
                    continue
                expected = model["alpha"] + model["beta"] * float(market)
                event_rows.append({
                    "event_id": event_id, "stock": stock, "trading_date": trading_date,
                    "effective_event_date": effective_date, "event_day": event_day,
                    "actual_return": float(actual), "market_return": float(market),
                    "expected_return": expected, "abnormal_return": float(actual) - expected,
                })

    event_study = pd.DataFrame(event_rows)
    if event_study.empty:
        raise ValueError("No complete event-window observations were generated")
    for (event_id, stock), group in event_study.groupby(["event_id", "stock"]):
        indexed = group.set_index("event_day")
        for window_name, (lower, upper) in CAR_WINDOWS.items():
            selected = indexed.loc[indexed.index.to_series().between(lower, upper).to_numpy()]
            expected_sessions = upper - lower + 1
            complete = len(selected) == expected_sessions
            car_rows.append({
                "event_id": event_id, "stock": stock, "window": window_name,
                "sessions": len(selected),
                "CAR": float(selected["abnormal_return"].sum()) if complete else np.nan,
            })
    return event_study, pd.DataFrame(car_rows), models


def calculate_market_adjusted_car(
    group: pd.DataFrame, window: tuple[int, int]
) -> float:
    lower, upper = window
    selected = group[group["event_day"].between(lower, upper)]
    if len(selected) != upper - lower + 1:
        return float("nan")
    return float((selected["actual_return"] - selected["market_return"]).sum())