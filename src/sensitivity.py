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
    """Compare CARs with and without sessions above an absolute market-return threshold."""
    subset = event_study[event_study["event_id"] == event_id].copy()
    if subset.empty:
        raise ValueError(f"No observations for event {event_id}")
    subset["extreme_market_day"] = subset["market_return"].abs() > threshold
    rows = []
    for stock, group in subset.groupby("stock"):
        for name, lower, upper in (("CAR_-1_+1", -1, 1), ("CAR_-3_+3", -3, 3), ("CAR_-5_+5", -5, 5)):
            window = group[group["event_day"].between(lower, upper)]
            complete = len(window) == upper - lower + 1
            retained = window[~window["extreme_market_day"]]
            rows.append({
                "event_id": event_id, "stock": stock, "window": name,
                "original_CAR": window["abnormal_return"].sum() if complete else float("nan"),
                "sensitivity_CAR": retained["abnormal_return"].sum(),
                "extreme_days_excluded": int(window["extreme_market_day"].sum()),
                "sessions_retained": len(retained),
            })
    return pd.DataFrame(rows)