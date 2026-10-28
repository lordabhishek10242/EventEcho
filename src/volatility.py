"""Descriptive realized-return and market-model residual volatility reports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import calculate_returns

PRE_EVENT_WINDOW = (-20, -1)
POST_EVENT_WINDOW = (1, 20)
EVENT_RESIDUAL_WINDOW = (-5, 5)
TRADING_DAYS_PER_YEAR = 252


def calculate_realized_volatility(
    prices: pd.Series,
    minimum_observations: int = 2,
) -> dict[str, float | int | str]:
    """Calculate sample volatility from simple close-to-close price returns.

    Annualization assumes 252 trading sessions per year and scales the daily
    sample standard deviation by ``sqrt(252)``. Missing prices are not filled;
    windows below the minimum count return NaN estimates. A constant-return
    series is valid and has zero volatility.
    """
    if minimum_observations < 2:
        raise ValueError("minimum_observations must be at least two")
    daily_returns = calculate_returns(prices).dropna()
    count = len(daily_returns)
    adequate = count >= minimum_observations
    daily_volatility = float(daily_returns.std(ddof=1)) if adequate else np.nan
    return {
        "daily_volatility": daily_volatility,
        "annualized_volatility": daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR),
        "observations": count,
        "minimum_observations": minimum_observations,
        "status": "minimum_met_descriptive" if adequate else "insufficient_observations",
    }


def _window_values(
    dates: pd.DatetimeIndex,
    values: pd.Series,
    center: int,
    relative_window: tuple[int, int],
    minimum_observations: int,
) -> dict:
    lower, upper = relative_window
    requested_start, requested_end = center + lower, center + upper
    start, end = max(0, requested_start), min(len(dates) - 1, requested_end)
    truncated = requested_start < 0 or requested_end >= len(dates)
    if start > end:
        selected = pd.Series(dtype=float)
        expected_sessions = upper - lower + 1
        start_date = pd.NaT
        end_date = pd.NaT
    else:
        selected_dates = dates[start : end + 1]
        selected = pd.to_numeric(values.reindex(selected_dates), errors="coerce").dropna()
        expected_sessions = upper - lower + 1
        start_date = selected_dates[0]
        end_date = selected_dates[-1]
    count = len(selected)
    if truncated:
        status = "truncated_at_data_boundary"
    elif count < minimum_observations:
        status = "insufficient_observations"
    elif count < expected_sessions:
        status = "minimum_met_descriptive_with_missing_sessions"
    else:
        status = "minimum_met_descriptive"
    estimate = float(selected.std(ddof=1)) if count >= minimum_observations else np.nan
    return {
        "volatility": estimate,
        "observations": count,
        "expected_sessions": expected_sessions,
        "status": status,
        "start_date": start_date,
        "end_date": end_date,
    }


def summarize_event_volatility(
    returns: pd.DataFrame,
    events: pd.DataFrame,
    stock_columns: list[str],
    models: dict[tuple[str, str], dict] | None = None,
    minimum_realized_observations: int = 15,
    minimum_residual_observations: int = 8,
    pre_event_window: tuple[int, int] = PRE_EVENT_WINDOW,
    post_event_window: tuple[int, int] = POST_EVENT_WINDOW,
    residual_window: tuple[int, int] = EVENT_RESIDUAL_WINDOW,
    market_column: str = "NIFTY50_RETURN",
) -> pd.DataFrame:
    """Summarize realized close-to-close volatility before/after each event.

    Realized volatility is the sample standard deviation of daily simple returns,
    annualized using sqrt(252). The default event-relative windows are trading
    sessions -20:-1 and +1:+20, excluding Day 0 and sharing no observations.
    Benchmark volatility uses the aligned benchmark returns. Event-window
    market-model residual volatility is separate from realized volatility.
    Low or truncated samples retain counts/status but do not produce estimates.
    """
    if minimum_realized_observations < 2 or minimum_residual_observations < 2:
        raise ValueError("Minimum volatility sample sizes must be at least two")
    if pre_event_window[0] > pre_event_window[1] or pre_event_window[1] >= 0:
        raise ValueError("pre_event_window must be ordered and end before Day 0")
    if post_event_window[0] <= 0 or post_event_window[0] > post_event_window[1]:
        raise ValueError("post_event_window must be ordered and start after Day 0")
    if pre_event_window[1] >= post_event_window[0]:
        raise ValueError("pre-event and post-event windows must not overlap")
    if residual_window[0] > residual_window[1]:
        raise ValueError("residual_window must be ordered")
    dates = pd.DatetimeIndex(returns.index).tz_localize(None).normalize().unique().sort_values()
    records = []
    for _, event in events.iterrows():
        event_id = str(event["event_id"])
        event_date = pd.Timestamp(event["effective_event_date"]).tz_localize(None).normalize()
        center = dates.get_indexer([event_date])[0]
        if center < 0:
            raise ValueError(f"Effective event date {event_date.date()} is not a trading session")
        for stock in stock_columns:
            if stock not in returns.columns or market_column not in returns.columns:
                continue
            pre = _window_values(
                dates, returns[stock], int(center), pre_event_window, minimum_realized_observations
            )
            post = _window_values(
                dates, returns[stock], int(center), post_event_window, minimum_realized_observations
            )
            reportable_statuses = {
                "minimum_met_descriptive",
                "minimum_met_descriptive_with_missing_sessions",
            }
            pre_acceptable = pre["status"] in reportable_statuses
            post_acceptable = post["status"] in reportable_statuses
            pre_volatility = pre["volatility"] if pre_acceptable else np.nan
            post_volatility = post["volatility"] if post_acceptable else np.nan
            absolute_change = (
                post_volatility - pre_volatility
                if pre_acceptable and post_acceptable else np.nan
            )
            positive_baseline = pre_acceptable and pre_volatility > 0
            ratio = post_volatility / pre_volatility if positive_baseline and post_acceptable else np.nan
            percentage_change = (
                (post_volatility - pre_volatility) / pre_volatility * 100
                if positive_baseline and post_acceptable else np.nan
            )
            if not pre_acceptable or not post_acceptable:
                comparison_status = "insufficient_observations"
            elif pre_volatility == 0:
                comparison_status = "zero_pre_event_volatility"
            else:
                comparison_status = "descriptive_comparison"

            benchmark_pre = _window_values(
                dates, returns[market_column], int(center), pre_event_window,
                minimum_realized_observations,
            )
            benchmark_post = _window_values(
                dates, returns[market_column], int(center), post_event_window,
                minimum_realized_observations,
            )

            residual = np.nan
            residual_count = 0
            residual_status = "model_unavailable"
            model = models.get((event_id, stock)) if models is not None else None
            if model is not None:
                residual_data = []
                lower, upper = residual_window
                start, end = max(0, center + lower), min(len(dates) - 1, center + upper)
                residual_truncated = center + lower < 0 or center + upper >= len(dates)
                if start <= end:
                    for date in dates[start : end + 1]:
                        actual = returns.at[date, stock]
                        market = returns.at[date, market_column]
                        if pd.notna(actual) and pd.notna(market):
                            residual_data.append(
                                float(actual) - model["alpha"] - model["beta"] * float(market)
                            )
                residual_count = len(residual_data)
                if residual_truncated:
                    residual_status = "truncated_at_data_boundary"
                elif residual_count < minimum_residual_observations:
                    residual_status = "insufficient_observations"
                elif residual_count < upper - lower + 1:
                    residual_status = "minimum_met_descriptive_with_missing_sessions"
                    residual = float(np.std(residual_data, ddof=1))
                else:
                    residual_status = "minimum_met_descriptive"
                    residual = float(np.std(residual_data, ddof=1))

            records.append({
                "event_id": event_id,
                "stock": stock,
                "category": event.get("category"),
                "episode_id": event.get("episode_id"),
                "effective_event_date": event_date,
                "pre_event_window": f"{pre_event_window[0]}:{pre_event_window[1]}",
                "pre_event_window_start_date": pre["start_date"],
                "pre_event_window_end_date": pre["end_date"],
                "pre_event_realized_volatility_daily": pre_volatility,
                "pre_event_realized_volatility_annualized": (
                    pre_volatility * np.sqrt(TRADING_DAYS_PER_YEAR) if pre_acceptable else np.nan
                ),
                "pre_event_observations": pre["observations"],
                "pre_event_expected_sessions": pre["expected_sessions"],
                "pre_event_volatility_status": pre["status"],
                "post_event_window": f"{post_event_window[0]}:{post_event_window[1]}",
                "post_event_window_start_date": post["start_date"],
                "post_event_window_end_date": post["end_date"],
                "post_event_realized_volatility_daily": post_volatility,
                "post_event_realized_volatility_annualized": (
                    post_volatility * np.sqrt(TRADING_DAYS_PER_YEAR) if post_acceptable else np.nan
                ),
                "post_event_observations": post["observations"],
                "post_event_expected_sessions": post["expected_sessions"],
                "post_event_volatility_status": post["status"],
                "realized_volatility_absolute_change_daily": absolute_change,
                "realized_volatility_percentage_change": percentage_change,
                "volatility_comparison_status": comparison_status,
                "post_to_pre_realized_volatility_ratio": ratio,
                "benchmark_pre_event_realized_volatility_daily": (
                    benchmark_pre["volatility"] if benchmark_pre["status"] in reportable_statuses else np.nan
                ),
                "benchmark_pre_event_realized_volatility_annualized": (
                    benchmark_pre["volatility"] * np.sqrt(TRADING_DAYS_PER_YEAR)
                    if benchmark_pre["status"] in reportable_statuses else np.nan
                ),
                "benchmark_pre_event_observations": benchmark_pre["observations"],
                "benchmark_pre_event_volatility_status": benchmark_pre["status"],
                "benchmark_post_event_realized_volatility_daily": (
                    benchmark_post["volatility"] if benchmark_post["status"] in reportable_statuses else np.nan
                ),
                "benchmark_post_event_realized_volatility_annualized": (
                    benchmark_post["volatility"] * np.sqrt(TRADING_DAYS_PER_YEAR)
                    if benchmark_post["status"] in reportable_statuses else np.nan
                ),
                "benchmark_post_event_observations": benchmark_post["observations"],
                "benchmark_post_event_volatility_status": benchmark_post["status"],
                "realized_volatility_definition": "sample standard deviation (ddof=1) of daily simple returns; annualized by sqrt(252)",
                "event_window_residual_window": f"{residual_window[0]}:{residual_window[1]}",
                "event_window_market_model_residual_volatility_daily": residual,
                "event_window_residual_observations": residual_count,
                "event_window_residual_status": residual_status,
                "estimation_market_model_residual_volatility_daily": (
                    model["residual_std"] if model is not None else np.nan
                ),
                "estimation_residual_observations": model["observations"] if model is not None else 0,
                "residual_volatility_definition": "sample standard deviation of actual minus market-model expected returns; not realized stock-return volatility",
            })
    return pd.DataFrame(records)