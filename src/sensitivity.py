"""Cost and extreme-market-movement sensitivity summaries."""

from __future__ import annotations

import pandas as pd

from .event_study import calculate_event_study
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



def event_timing_sensitivity(
    returns: pd.DataFrame,
    events: pd.DataFrame,
    stock_columns: list[str],
    recorded_car_results: pd.DataFrame,
    estimation_sessions: int = 100,
    gap_sessions: int = 20,
    event_window_radius: int = 5,
) -> pd.DataFrame:
    """Compare plausible session alignments for events with unresolved timing.

    If the source/context indicates an after-market or evening announcement,
    the first scenario is strictly the next available trading session. It must
    never use the same-day close, which occurred before the information became
    public. Otherwise compare the first session on/after the announcement date
    with the following session. This is a timing robustness diagnostic, not an
    automatic correction or causal estimate.
    """
    required = {"event_id", "event_date", "effective_event_date"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events is missing required columns: {sorted(missing)}")
    if recorded_car_results.empty:
        raise ValueError("recorded_car_results must not be empty")

    dates = pd.DatetimeIndex(returns.index).tz_localize(None).normalize().unique().sort_values()
    if len(dates) < 2:
        raise ValueError("At least two trading sessions are required")
    unresolved_keywords = (
        "pending", "unverified", "not_in_master", "not in master",
        "check_required", "required", "assumption", "review",
        "not_release_timestamp", "not separately confirmed", "not reported",
    )
    after_market_tokens = (
        "after trading hours", "after-market", "after market",
        "after-close", "after close", "evening announcement",
        "announced in the evening", "made in the evening", "16:15 ist",
        "scheduled 16:15",
    )
    rows: list[dict] = []
    for _, event in events.iterrows():
        metadata = " ".join(
            str(event.get(column, "")) for column in (
                "event_time", "timing_confidence", "verification_status", "notes"
            )
        ).casefold()
        if not any(keyword in metadata for keyword in unresolved_keywords):
            continue

        event_id = str(event["event_id"])
        announcement_date = pd.Timestamp(event["event_date"]).tz_localize(None).normalize()
        announcement_date_is_session = announcement_date in dates
        after_market_announcement = any(token in metadata for token in after_market_tokens)

        # A known after-market release cannot be assigned to that day's close.
        # If the calendar date itself is a session, move strictly past it; if
        # it is a weekend/holiday, searchsorted(left) already gives next session.
        side = "right" if after_market_announcement and announcement_date_is_session else "left"
        first_position = int(dates.searchsorted(announcement_date, side=side))
        if first_position >= len(dates) - 1:
            continue
        first_session = dates[first_position]
        following_session = dates[first_position + 1]
        recorded_effective = pd.Timestamp(event["effective_event_date"]).tz_localize(None).normalize()
        baseline = recorded_car_results[recorded_car_results["event_id"] == event_id]

        if after_market_announcement:
            timing_basis = "after_market_or_evening_release"
            scenarios = (
                ("first_session_after_announcement", first_session),
                ("following_session_after_first", following_session),
            )
        else:
            timing_basis = "intraday_or_exact_release_time_unresolved"
            scenarios = (
                ("first_session_on_or_after_announcement", first_session),
                ("following_trading_session", following_session),
            )

        for scenario, scenario_date in scenarios:
            scenario_events = event.to_frame().T.copy()
            scenario_events.loc[:, "effective_event_date"] = scenario_date
            _, scenario_cars, _ = calculate_event_study(
                returns,
                scenario_events,
                stock_columns,
                estimation_sessions,
                gap_sessions,
                event_window_radius,
            )
            for _, car in scenario_cars.iterrows():
                base = baseline[
                    (baseline["stock"] == car["stock"])
                    & (baseline["window"] == car["window"])
                ]
                recorded_car = (
                    float(base.iloc[0]["CAR"])
                    if not base.empty and pd.notna(base.iloc[0]["CAR"])
                    else float("nan")
                )
                scenario_car = float(car["CAR"]) if pd.notna(car["CAR"]) else float("nan")
                rows.append({
                    "event_id": event_id,
                    "headline": event.get("headline", ""),
                    "category": event.get("category", ""),
                    "announcement_date": announcement_date,
                    "announcement_date_is_trading_session": bool(announcement_date_is_session),
                    "recorded_effective_event_date": recorded_effective,
                    "timing_confidence": event.get("timing_confidence", ""),
                    "timing_basis": timing_basis,
                    "scenario": scenario,
                    "scenario_effective_date": scenario_date,
                    "matches_recorded_alignment": bool(scenario_date == recorded_effective),
                    "stock": car["stock"],
                    "window": car["window"],
                    "scenario_CAR": scenario_car,
                    "scenario_CAR_pct": scenario_car * 100,
                    "recorded_CAR": recorded_car,
                    "recorded_CAR_pct": recorded_car * 100,
                    "delta_vs_recorded_CAR_pct": (
                        (scenario_car - recorded_car) * 100
                        if pd.notna(scenario_car) and pd.notna(recorded_car)
                        else float("nan")
                    ),
                    "interpretation": "Timing sensitivity only; not a corrected estimate or causal effect.",
                })
    return pd.DataFrame(rows)
