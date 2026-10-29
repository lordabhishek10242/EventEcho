"""Issuer-level earnings event study (one event = one company's results day).

The 21-event registry mostly holds market-wide events that hit all seven
stocks on the same day. This module adds a much larger and cleaner sample:
every quarterly results announcement of the seven companies from Q1 FY17 to
Q2 FY26 (266 announcements), each studied **only for the company that
reported**.

Design, fixed before any result was seen:

* Day 0 is the first NIFTY 50 session on or after the announcement date.
  Release times are not verified; many Indian companies report after the
  close or on a Saturday, so the (-1,+1) window is the main window because it
  always contains the first full trading session after the release.
* Same market model, 100-session estimation window and 20-session gap as the
  main study.
* We do not know whether each result beat or missed expectations (no analyst
  forecast data is used), so good and bad surprises are expected to cancel in
  the average CAR. The **main question is therefore volatility**: do results
  days move these stocks more than ordinary days? This is measured by the
  event-induced variance ratio (average SCAR squared; about 1 with no effect).
  The signed average CAR is reported as a secondary check.
* Development sample: announcements up to 31 Dec 2022. Holdout: from 1 Jan
  2023. The split is by calendar date and was set before running the study.
* Dependence: announcements whose (-5,+5) windows overlap in calendar time
  (for example TCS and Infosys a day apart) form one cluster for bootstrap
  intervals, and the Kolari-Pynnönen adjustment uses residual correlations.
* Announcements whose window overlaps a registry (macro) event window are
  flagged and a summary without them is reported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import CAR_WINDOWS, STOCK_SECTORS
from .event_study import estimation_sample, fit_market_model
from .robust_tests import EVENT_DAYS, EventUnit, group_tests
from .statistics import car_significance

HOLDOUT_START = pd.Timestamp("2023-01-01")
# Placebo: the same stocks, 30 sessions after each results day (mid-quarter).
# Fixed before the study was run. Shows the variance ratio on ordinary days,
# which is above 1 because daily returns have fat tails and volatility drifts.
PLACEBO_SHIFT_SESSIONS = 30
MAIN_WINDOW = "CAR_-1_+1"
SEED = 2026
REQUIRED_COLUMNS = {"earnings_id", "stock", "fiscal_quarter", "announcement_date", "source_url"}


def load_earnings_calendar(path_or_frame, stocks: Sequence[str]) -> pd.DataFrame:
    frame = path_or_frame.copy() if isinstance(path_or_frame, pd.DataFrame) else pd.read_csv(path_or_frame)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"earnings calendar is missing columns: {sorted(missing)}")
    if frame["earnings_id"].duplicated().any():
        raise ValueError("earnings_id values must be unique")
    if frame[["stock", "fiscal_quarter"]].duplicated().any():
        raise ValueError("each stock may have only one announcement per fiscal quarter")
    if frame["source_url"].isna().any() or (frame["source_url"].astype(str).str.strip() == "").any():
        raise ValueError("every announcement date needs a source")
    unknown = sorted(set(frame["stock"]) - set(stocks))
    if unknown:
        raise ValueError(f"unknown stocks in earnings calendar: {unknown}")
    frame["announcement_date"] = pd.to_datetime(frame["announcement_date"], errors="raise").dt.normalize()
    return frame.sort_values(["announcement_date", "stock"]).reset_index(drop=True)


def first_session_on_or_after(sessions: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    position = sessions.searchsorted(pd.Timestamp(day).normalize(), side="left")
    return None if position >= len(sessions) else sessions[position]


def build_event_units(
    returns: pd.DataFrame,
    table: pd.DataFrame,
    estimation_sessions: int,
    gap_sessions: int,
    radius: int,
    market_column: str = "NIFTY50_RETURN",
) -> tuple[list[EventUnit], pd.DataFrame, pd.DataFrame]:
    """Fit one market model per row of ``table`` (unit_id, stock, event_date).

    Returns (units, abnormal-return rows, skipped rows with reasons). Rows
    without enough history or a complete (-5,+5) window are skipped and listed.
    """
    sessions = pd.DatetimeIndex(returns.index).tz_localize(None).normalize()
    units: list[EventUnit] = []
    ar_rows: list[dict] = []
    skipped: list[dict] = []
    for _, row in table.iterrows():
        unit_id, stock, day0 = str(row["unit_id"]), str(row["stock"]), pd.Timestamp(row["event_date"])
        try:
            sample = estimation_sample(returns, day0, estimation_sessions, gap_sessions, stock, market_column, radius)
            model = fit_market_model(sample, stock, market_column)
        except ValueError as exc:
            skipped.append({"unit_id": unit_id, "stock": stock, "event_date": day0, "reason": str(exc)})
            continue
        position = sessions.get_loc(day0)
        if position - radius < 0 or position + radius >= len(sessions):
            skipped.append({"unit_id": unit_id, "stock": stock, "event_date": day0,
                            "reason": "event window extends beyond the price history"})
            continue
        window = returns.iloc[position - radius: position + radius + 1][[stock, market_column]]
        if window.isna().any().any():
            skipped.append({"unit_id": unit_id, "stock": stock, "event_date": day0,
                            "reason": "missing price in the (-5,+5) window"})
            continue
        expected = model["alpha"] + model["beta"] * window[market_column]
        abnormal = (window[stock] - expected).to_numpy(dtype=float)
        ars = pd.Series(abnormal, index=list(EVENT_DAYS))
        cars, ses = {}, {}
        for name, (lower, upper) in CAR_WINDOWS.items():
            days = list(range(lower, upper + 1))
            car = float(ars[days].sum())
            market = window[market_column].to_numpy(dtype=float)[[d + radius for d in days]]
            cars[name] = car
            ses[name] = car_significance(car, days, market, model)["standard_error"]
        residuals = pd.Series(model["estimation_residuals"], index=pd.DatetimeIndex(sample.index))
        units.append(EventUnit(
            unit_id=unit_id, stock=stock, estimation_residuals=residuals, abnormal_returns=ars,
            car_standard_errors=ses, cars=cars, degrees_freedom=int(model["degrees_freedom"]),
            event_position=int(position),
        ))
        for offset, day in enumerate(EVENT_DAYS):
            ar_rows.append({
                "unit_id": unit_id, "stock": stock, "event_day": day,
                "trading_date": sessions[position + day],
                "abnormal_return": abnormal[offset],
                "standardized_ar": abnormal[offset] / model["residual_std"],
            })
    return units, pd.DataFrame(ar_rows), pd.DataFrame(skipped)


def window_clusters(day0_positions: Mapping[str, int], radius: int) -> dict[str, str]:
    """Join units whose (-radius,+radius) session windows overlap in time."""
    ordered = sorted(day0_positions.items(), key=lambda item: (item[1], item[0]))
    clusters: dict[str, str] = {}
    label, last_end = 0, None
    for unit_id, position in ordered:
        start, end = position - radius, position + radius
        if last_end is None or start > last_end:
            label += 1
            last_end = end
        else:
            last_end = max(last_end, end)
        clusters[unit_id] = f"ECL_{label:03d}"
    return clusters


def registry_overlaps(
    day0: Mapping[str, pd.Timestamp], sessions: pd.DatetimeIndex, registry: pd.DataFrame, radius: int
) -> dict[str, str]:
    """Registry event IDs whose (-radius,+radius) window overlaps each unit's window."""
    registry_positions = {}
    for _, event in registry.iterrows():
        date = first_session_on_or_after(sessions, pd.Timestamp(event["effective_event_date"]))
        if date is not None:
            registry_positions[str(event["event_id"])] = sessions.get_loc(date)
    out = {}
    for unit_id, date in day0.items():
        position = sessions.get_loc(date)
        hits = [event_id for event_id, other in registry_positions.items() if abs(other - position) <= 2 * radius]
        out[unit_id] = ";".join(sorted(hits))
    return out


def _groups(meta: pd.DataFrame) -> list[tuple[str, str, list[str]]]:
    ids = meta["earnings_id"].tolist()
    groups = [
        ("all", "All announcements", ids),
        ("all", "Without registry-event overlap", meta.loc[meta["overlapping_registry_events"] == "", "earnings_id"].tolist()),
        ("sample", "Development (to 2022)", meta.loc[meta["sample"] == "development", "earnings_id"].tolist()),
        ("sample", "Holdout (2023 onward)", meta.loc[meta["sample"] == "holdout", "earnings_id"].tolist()),
    ]
    for sector in sorted(meta["sector"].unique()):
        groups.append(("sector", sector, meta.loc[meta["sector"] == sector, "earnings_id"].tolist()))
    for stock in sorted(meta["stock"].unique()):
        groups.append(("stock", stock, meta.loc[meta["stock"] == stock, "earnings_id"].tolist()))
    for stock in sorted(meta["stock"].unique()):
        for sample in ("development", "holdout"):
            members = meta.loc[(meta["stock"] == stock) & (meta["sample"] == sample), "earnings_id"].tolist()
            groups.append(("stock_by_sample", f"{stock} | {sample}", members))
    return groups


def run_earnings_study(
    returns: pd.DataFrame,
    calendar: pd.DataFrame,
    registry_events: pd.DataFrame,
    estimation_sessions: int,
    gap_sessions: int,
    radius: int,
) -> dict[str, pd.DataFrame]:
    sessions = pd.DatetimeIndex(returns.index).tz_localize(None).normalize()
    calendar = calendar.copy()
    calendar["event_date"] = [first_session_on_or_after(sessions, day) for day in calendar["announcement_date"]]
    calendar["day0_rule"] = np.where(
        calendar["event_date"] == calendar["announcement_date"], "announcement day is a session",
        "next session after a non-trading announcement day",
    )
    calendar.loc[calendar["event_date"].isna(), "day0_rule"] = "after the price history"
    usable = calendar.dropna(subset=["event_date"]).rename(columns={"earnings_id": "unit_id"})
    units, ar_rows, skipped = build_event_units(returns, usable[["unit_id", "stock", "event_date"]],
                                                estimation_sessions, gap_sessions, radius)
    unit_map = {unit.unit_id: unit for unit in units}
    calendar["included"] = calendar["earnings_id"].isin(unit_map)
    if not unit_map:
        # Nothing usable (for example a short test price file): report coverage only.
        return {"earnings_coverage.csv": _coverage(calendar, skipped)}
    meta = calendar[calendar["included"]].copy()
    meta["sector"] = meta["stock"].map(STOCK_SECTORS)
    meta["sample"] = np.where(meta["announcement_date"] < HOLDOUT_START, "development", "holdout")
    day0 = dict(zip(meta["earnings_id"], meta["event_date"]))
    clusters = window_clusters({k: sessions.get_loc(v) for k, v in day0.items()}, radius)
    meta["dependency_cluster"] = meta["earnings_id"].map(clusters)
    meta["overlapping_registry_events"] = meta["earnings_id"].map(
        registry_overlaps(day0, sessions, registry_events, radius))

    event_rows = []
    for _, row in meta.iterrows():
        unit = unit_map[row["earnings_id"]]
        record = {
            "earnings_id": row["earnings_id"], "stock": row["stock"], "sector": row["sector"],
            "fiscal_quarter": row["fiscal_quarter"],
            "announcement_date": row["announcement_date"].date().isoformat(),
            "day0_session": row["event_date"].date().isoformat(), "day0_rule": row["day0_rule"],
            "sample": row["sample"], "dependency_cluster": row["dependency_cluster"],
            "overlapping_registry_events": row["overlapping_registry_events"],
        }
        for window in CAR_WINDOWS:
            car, se = unit.cars[window], unit.car_standard_errors[window]
            record[f"{window}_pct"] = car * 100
            record[f"{window}_se_pct"] = se * 100
            record[f"{window}_scar"] = car / se
        event_rows.append(record)
    event_table = pd.DataFrame(event_rows)

    summary_rows = []
    for index, (group_type, group, members) in enumerate(_groups(meta)):
        group_units = [unit_map[m] for m in members if m in unit_map]
        n_clusters = len({clusters[m] for m in members if m in clusters})
        for row in group_tests(group_units, CAR_WINDOWS, clusters, SEED + 10 * index):
            summary_rows.append({"group_type": group_type, "group": group, "clusters": n_clusters, **row})
    summary = pd.DataFrame(summary_rows)

    ar_rows = ar_rows.merge(meta[["earnings_id", "sample"]].rename(columns={"earnings_id": "unit_id"}), on="unit_id")
    path_rows = []
    for scope, frame in [("All announcements", ar_rows)] + [(s, g) for s, g in ar_rows.groupby("stock")]:
        for day, part in frame.groupby("event_day"):
            path_rows.append({
                "group": scope, "event_day": int(day), "events": int(part["unit_id"].nunique()),
                "mean_AR_pct": part["abnormal_return"].mean() * 100,
                "mean_abs_AR_pct": part["abnormal_return"].abs().mean() * 100,
                "mean_squared_standardized_AR": float((part["standardized_ar"] ** 2).mean()),
            })
    paths = pd.DataFrame(path_rows)

    outputs = {
        "earnings_event_car.csv": event_table,
        "earnings_summary.csv": summary,
        "earnings_event_time.csv": paths,
        "earnings_coverage.csv": _coverage(calendar, skipped),
    }
    replication = _replication(summary)
    if not replication.empty:
        outputs["earnings_replication.csv"] = replication
    placebo = _placebo(returns, meta, estimation_sessions, gap_sessions, radius)
    if not placebo.empty:
        outputs["earnings_placebo.csv"] = placebo
    return outputs


def _placebo(
    returns: pd.DataFrame, meta: pd.DataFrame, estimation_sessions: int, gap_sessions: int, radius: int
) -> pd.DataFrame:
    """Variance ratio on pseudo-event days that are not results days.

    Each real Day 0 is moved forward by PLACEBO_SHIFT_SESSIONS sessions. A
    pseudo-event is dropped when its (-5,+5) window would touch any real
    results window of the same stock, so the placebo never contains results.
    """
    sessions = pd.DatetimeIndex(returns.index).tz_localize(None).normalize()
    real_positions: dict[str, list[int]] = {}
    for stock, day in zip(meta["stock"], meta["event_date"]):
        real_positions.setdefault(stock, []).append(sessions.get_loc(day))
    rows, dropped = [], 0
    for earnings_id, stock, day in zip(meta["earnings_id"], meta["stock"], meta["event_date"]):
        position = sessions.get_loc(day) + PLACEBO_SHIFT_SESSIONS
        if position + radius >= len(sessions):
            dropped += 1
            continue
        if any(abs(position - other) <= 2 * radius for other in real_positions[stock]):
            dropped += 1
            continue
        rows.append({"unit_id": f"P_{earnings_id}", "stock": stock, "event_date": sessions[position]})
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame()
    units, _, _ = build_event_units(returns, table, estimation_sessions, gap_sessions, radius)
    clusters = window_clusters({u.unit_id: u.event_position for u in units}, radius)
    groups = [("all", "All placebo days", units)] + [
        ("stock", stock, [u for u in units if u.stock == stock]) for stock in sorted({u.stock for u in units})
    ]
    out = []
    for index, (group_type, group, members) in enumerate(groups):
        for row in group_tests(members, CAR_WINDOWS, clusters, SEED + 500 + 10 * index):
            out.append({"group_type": group_type, "group": group, "shift_sessions": PLACEBO_SHIFT_SESSIONS,
                        "dropped_for_overlap_or_range": dropped, **row})
    return pd.DataFrame(out)


def _coverage(calendar: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    coverage = calendar[["earnings_id", "stock", "fiscal_quarter", "announcement_date", "event_date", "day0_rule", "included"]].copy()
    reasons = (skipped.rename(columns={"unit_id": "earnings_id"})[["earnings_id", "reason"]] if not skipped.empty
               else pd.DataFrame(columns=["earnings_id", "reason"]))
    coverage = coverage.merge(reasons, on="earnings_id", how="left")
    coverage["announcement_date"] = coverage["announcement_date"].dt.date.astype(str)
    coverage["event_date"] = pd.to_datetime(coverage["event_date"]).dt.date.astype(str)
    return coverage


def _replication(summary: pd.DataFrame) -> pd.DataFrame:
    """Does the main-window volatility effect show up in both samples, per stock?"""
    main = summary[(summary["window"] == MAIN_WINDOW) & (summary["group_type"] == "stock_by_sample")
                   & (summary["units"] > 0)].copy()
    if main.empty:
        return pd.DataFrame()
    main[["stock", "sample"]] = main["group"].str.split(" | ", regex=False, expand=True)
    rows = []
    for stock, part in main.groupby("stock"):
        by = part.set_index("sample")
        if not {"development", "holdout"} <= set(by.index):
            continue
        dev, hold = by.loc["development"], by.loc["holdout"]

        def above_one(r):
            return bool(np.isfinite(r["variance_ratio_ci_lower"]) and r["variance_ratio_ci_lower"] > 1)

        rows.append({
            "stock": stock,
            "dev_events": int(dev["units"]), "holdout_events": int(hold["units"]),
            "dev_variance_ratio": dev["variance_ratio"],
            "dev_ci_lower": dev["variance_ratio_ci_lower"], "dev_ci_upper": dev["variance_ratio_ci_upper"],
            "holdout_variance_ratio": hold["variance_ratio"],
            "holdout_ci_lower": hold["variance_ratio_ci_lower"], "holdout_ci_upper": hold["variance_ratio_ci_upper"],
            "dev_mean_abs_CAR_pct": dev["mean_abs_CAR_pct"], "holdout_mean_abs_CAR_pct": hold["mean_abs_CAR_pct"],
            "dev_mean_CAR_pct": dev["mean_CAR_pct"], "holdout_mean_CAR_pct": hold["mean_CAR_pct"],
            "volatility_effect_in_dev": above_one(dev), "volatility_effect_in_holdout": above_one(hold),
            "replicates": above_one(dev) and above_one(hold),
        })
    table = pd.DataFrame(rows)
    if len(table) >= 3:
        rho = table["dev_variance_ratio"].rank().corr(table["holdout_variance_ratio"].rank())
        table["rank_correlation_dev_vs_holdout_across_stocks"] = rho
    return table


def registry_robust_tests(
    returns: pd.DataFrame,
    events: pd.DataFrame,
    stocks: Sequence[str],
    event_clusters: Mapping[str, str],
    groups: Sequence[tuple[str, str, list[str], list[str]]],
    estimation_sessions: int,
    gap_sessions: int,
    radius: int,
) -> pd.DataFrame:
    """BMP, Kolari-Pynnönen and rank tests for the registry (macro) groups.

    One unit is one (event, stock) pair. Pairs from the same event share a
    calendar date, so their residual correlation is large and the KP
    adjustment matters most here. Bootstrap clusters are the event-level
    dependency clusters used elsewhere in the project.
    """
    sessions = pd.DatetimeIndex(returns.index).tz_localize(None).normalize()
    table = pd.DataFrame([
        {"unit_id": f"{event_id}|{stock}", "stock": stock,
         "event_date": first_session_on_or_after(sessions, pd.Timestamp(date))}
        for event_id, date in zip(events["event_id"], events["effective_event_date"]) for stock in stocks
    ])
    units, _, _ = build_event_units(returns, table, estimation_sessions, gap_sessions, radius)
    by_id = {unit.unit_id: unit for unit in units}
    clusters = {unit_id: event_clusters.get(unit_id.split("|")[0], unit_id) for unit_id in by_id}
    rows = []
    for index, (group_type, group, event_ids, members) in enumerate(groups):
        chosen = [by_id[f"{e}|{s}"] for e in event_ids for s in members if f"{e}|{s}" in by_id]
        n_clusters = len({clusters[u.unit_id] for u in chosen})
        for row in group_tests(chosen, CAR_WINDOWS, clusters, SEED + 10 * index):
            rows.append({"group_type": group_type, "group": group, "events": len(set(event_ids)),
                         "clusters": n_clusters, **row})
    return pd.DataFrame(rows)
