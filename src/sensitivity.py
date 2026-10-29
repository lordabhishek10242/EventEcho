"""Cost and extreme-market-movement sensitivity summaries."""

from __future__ import annotations

import pandas as pd

from .trading import event_portfolio_returns


def transaction_cost_sensitivity(
    trades: pd.DataFrame, cost_levels_bps: tuple[int, ...] = (0, 5, 10, 20, 30)
) -> pd.DataFrame:
    rows = []
    for cost in cost_levels_bps:
        adjusted = trades.copy()
        adjusted["net_return"] = adjusted["gross_return"] - 2 * cost / 10000
        events = event_portfolio_returns(adjusted)
        excess = events["portfolio_net_return"] - events["benchmark_return"]
        valid_excess = excess.dropna()
        rows.append({
            "cost_bps_per_side": cost,
            "avg_net_return_pct": events["portfolio_net_return"].mean() * 100,
            "median_net_return_pct": events["portfolio_net_return"].median() * 100,
            "event_win_rate_pct": (events["portfolio_net_return"] > 0).mean() * 100,
            "avg_excess_return_pct": excess.mean() * 100,
            "benchmark_outperformance_rate_pct": (
                (valid_excess > 0).mean() * 100 if not valid_excess.empty else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def extreme_market_sensitivity(
    event_study: pd.DataFrame,
    event_id: str,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Report extreme-day diagnostics without mislabelling partial sums as CAR.

    A CAR is defined over its full configured event window. If one or more
    extreme-market sessions are excluded, the remaining abnormal-return sum is
    not a comparable CAR. In that case sensitivity_CAR is NaN and the
    retained-session sum/mean are exposed as descriptive diagnostics.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    required = {"event_id", "stock", "event_day", "market_return", "abnormal_return"}
    missing = required - set(event_study.columns)
    if missing:
        raise ValueError(f"event_study is missing required columns: {sorted(missing)}")
    subset = event_study[event_study["event_id"] == event_id].copy()
    if subset.empty:
        raise ValueError(f"No observations for event {event_id}")
    subset["extreme_market_day"] = subset["market_return"].abs() > threshold
    rows = []
    for stock, group in subset.groupby("stock"):
        for name, lower, upper in (("CAR_-1_+1", -1, 1), ("CAR_-3_+3", -3, 3), ("CAR_-5_+5", -5, 5)):
            window = group[group["event_day"].between(lower, upper)]
            expected_sessions = upper - lower + 1
            complete = (
                len(window) == expected_sessions
                and set(window["event_day"].astype(int)) == set(range(lower, upper + 1))
                and window["abnormal_return"].notna().all()
                and window["market_return"].notna().all()
            )
            retained = window[
                (~window["extreme_market_day"]) & window["abnormal_return"].notna()
            ]
            excluded = int(window["extreme_market_day"].sum())
            original_car = float(window["abnormal_return"].sum()) if complete else float("nan")
            comparable_sensitivity_car = (
                float(retained["abnormal_return"].sum())
                if complete and excluded == 0 and len(retained) == expected_sessions
                else float("nan")
            )
            rows.append({
                "event_id": event_id,
                "stock": stock,
                "window": name,
                "original_CAR": original_car,
                "sensitivity_CAR": comparable_sensitivity_car,
                "retained_abnormal_return_sum": float(retained["abnormal_return"].sum()),
                "retained_mean_abnormal_return": (
                    float(retained["abnormal_return"].mean()) if not retained.empty else float("nan")
                ),
                "extreme_days_excluded": excluded,
                "sessions_in_window": len(window),
                "sessions_retained": len(retained),
                "sensitivity_status": (
                    "incomplete_original_window" if not complete
                    else "partial_window_not_comparable" if excluded > 0
                    else "complete_window_no_extreme_days_excluded"
                ),
            })
    return pd.DataFrame(rows)
