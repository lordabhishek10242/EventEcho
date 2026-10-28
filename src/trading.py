"""Long-only event trade simulation with aligned benchmark periods."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_trades(
    prices: pd.DataFrame,
    benchmark_returns: pd.Series,
    events: pd.DataFrame,
    stock_columns: list[str],
    holding_sessions: int = 3,
    cost_bps_per_side: float = 10,
) -> pd.DataFrame:
    """Enter at next session close, exit N sessions later, charge both sides."""
    if holding_sessions < 1 or cost_bps_per_side < 0:
        raise ValueError("Holding sessions must be positive and costs non-negative")
    price_data = prices.sort_index().copy()
    price_data.index = pd.DatetimeIndex(price_data.index).tz_localize(None).normalize()
    benchmark = benchmark_returns.sort_index().copy()
    benchmark.index = pd.DatetimeIndex(benchmark.index).tz_localize(None).normalize()
    if price_data.index.has_duplicates:
        raise ValueError("Price index contains duplicate sessions")
    dates = pd.DatetimeIndex(price_data.index)
    cost = 2 * cost_bps_per_side / 10000
    rows: list[dict] = []
    for _, event in events.iterrows():
        event_date = pd.Timestamp(event["effective_event_date"]).tz_localize(None).normalize()
        first_position = dates.searchsorted(event_date, side="right")
        exit_position = first_position + holding_sessions
        if first_position >= len(dates) or exit_position >= len(dates):
            continue
        entry_date, exit_date = dates[first_position], dates[exit_position]
        market_period = benchmark.loc[(benchmark.index > entry_date) & (benchmark.index <= exit_date)]
        benchmark_return = float((1 + market_period.dropna()).prod() - 1)
        benchmark_observations = market_period.dropna()
        if (
            len(benchmark_observations) != holding_sessions
            or benchmark_observations.index[-1] != exit_date
        ):
            benchmark_return = np.nan
        for stock in stock_columns:
            if stock not in price_data:
                continue
            entry_price, exit_price = price_data.at[entry_date, stock], price_data.at[exit_date, stock]
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue
            gross = float(exit_price / entry_price - 1)
            net = gross - cost
            rows.append({
                "event_id": event["event_id"],
                "effective_event_date": event_date,
                "stock": stock,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "gross_return": gross,
                "net_return": net,
                "benchmark_return": benchmark_return,
                "net_excess_return": net - benchmark_return if pd.notna(benchmark_return) else np.nan,
            })
    return pd.DataFrame(rows)


def event_portfolio_returns(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        raise ValueError("No eligible trades were generated")
    output = (
        trades.groupby("event_id", as_index=False)
        .agg(
            portfolio_net_return=("net_return", "mean"),
            benchmark_return=("benchmark_return", "first"),
            stocks_traded=("stock", "nunique"),
        )
    )
    output["excess_return"] = output["portfolio_net_return"] - output["benchmark_return"]
    return output


def summarize_samples(
    event_returns: pd.DataFrame, development_ids: tuple[str, ...], holdout_ids: tuple[str, ...]
) -> pd.DataFrame:
    labels = {event_id: "Development" for event_id in development_ids}
    labels.update({event_id: "Holdout" for event_id in holdout_ids})
    frame = event_returns.copy()
    frame["sample"] = frame["event_id"].map(labels)
    rows = []
    for sample in ("Development", "Holdout"):
        group = frame[frame["sample"] == sample]
        if group.empty:
            continue
        excess = group["portfolio_net_return"] - group["benchmark_return"]
        valid_excess = excess.dropna()
        rows.append({
            "sample": sample,
            "number_of_events": group["event_id"].nunique(),
            "mean_net_return_pct": group["portfolio_net_return"].mean() * 100,
            "median_net_return_pct": group["portfolio_net_return"].median() * 100,
            "event_win_rate_pct": (group["portfolio_net_return"] > 0).mean() * 100,
            "mean_benchmark_return_pct": group["benchmark_return"].mean() * 100,
            "mean_excess_return_pct": excess.mean() * 100,
            "benchmark_outperformance_rate_pct": (
                (valid_excess > 0).mean() * 100 if not valid_excess.empty else np.nan
            ),
        })
    return pd.DataFrame(rows)