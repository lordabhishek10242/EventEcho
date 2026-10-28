"""Event master validation, trading-session alignment, and sample splitting."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

REQUIRED_EVENT_COLUMNS = (
    "event_id", "event_date", "effective_event_date", "headline", "category",
    "source_name", "source_url", "episode_id", "timing_confidence",
)


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_EVENT_COLUMNS) - set(events.columns))
    if missing:
        raise ValueError(f"Missing required event columns: {missing}")
    result = events.copy()
    if result["event_id"].isna().any() or result["event_id"].duplicated().any():
        raise ValueError("Event IDs must be present and unique")
    for column in ("event_date", "effective_event_date"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    if result["event_date"].isna().any() or result["effective_event_date"].isna().any():
        raise ValueError("Event and effective dates must be valid")
    for column in ("headline", "category", "source_name", "source_url"):
        if result[column].isna().any() or result[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must be present for every event")
    return result


def align_event_date(event_date: object, trading_dates: pd.Index) -> pd.Timestamp | None:
    """Return the first trading session on or after an effective event date."""
    dates = pd.DatetimeIndex(trading_dates).tz_localize(None).normalize().unique().sort_values()
    target = pd.Timestamp(event_date).tz_localize(None).normalize()
    position = dates.searchsorted(target, side="left")
    return dates[position] if position < len(dates) else None


def chronological_split(
    events: pd.DataFrame,
    development_ids: Sequence[str],
    holdout_ids: Sequence[str],
) -> pd.DataFrame:
    """Label events using explicit original IDs and reject omissions/overlap."""
    development, holdout = set(development_ids), set(holdout_ids)
    if development & holdout:
        raise ValueError("Development and holdout event IDs overlap")
    result = events.copy()
    all_ids = set(result["event_id"])
    if development | holdout != all_ids:
        raise ValueError("Every event must belong to exactly one configured sample")
    result["sample"] = result["event_id"].map(
        lambda value: "Development" if value in development else "Holdout"
    )
    if "effective_event_date" in result.columns:
        dates = pd.to_datetime(result["effective_event_date"], errors="coerce")
        development_end = dates[result["sample"] == "Development"].max()
        holdout_start = dates[result["sample"] == "Holdout"].min()
        if pd.isna(development_end) or pd.isna(holdout_start) or development_end >= holdout_start:
            raise ValueError("Configured development/holdout IDs are not chronologically separated")
    return result