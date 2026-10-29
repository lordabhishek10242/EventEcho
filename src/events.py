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

    # Provenance URLs and event dates do not, by themselves, prove a clock time.
    # Preserve unknowns explicitly; never infer timestamps from market moves.
    if "event_time" not in result.columns:
        result["event_time"] = ""
    event_time = result["event_time"].fillna("").astype(str).str.strip()
    confidence = result["timing_confidence"].fillna("").astype(str).str.lower()
    has_clock = event_time.str.contains(r"(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?!\d)", regex=True)
    context_only = event_time.str.contains(
        r"management comments|not the exact|context|intraday timing varies|exact intraday|approval time|board approved|board approval|approved results",
        case=False, regex=True,
    )
    result["timestamp_evidence_status"] = "date_only_or_qualitative_time"
    result.loc[event_time.eq(""), "timestamp_evidence_status"] = "unresolved_missing_clock_time"
    result.loc[has_clock & context_only, "timestamp_evidence_status"] = "time_context_not_release_timestamp"
    result.loc[
        has_clock & ~context_only & confidence.str.contains(r"confirmed|verified", regex=True),
        "timestamp_evidence_status",
    ] = "clock_time_recorded_source_status_claims_verified"
    result["timestamp_review_required"] = ~result["timestamp_evidence_status"].eq(
        "clock_time_recorded_source_status_claims_verified"
    )

    # Normalize only explicit IST clock times whose source-status metadata says
    # verified/confirmed and which are not merely call/approval context.
    clock_token = event_time.str.extract(
        r"((?:[01]?\d|2[0-3]):[0-5]\d)", expand=False
    )
    explicit_ist = event_time.str.contains(r"\bIST\b|India Standard Time", case=False, regex=True)
    timestamp_ok = (
        result["timestamp_evidence_status"].eq("clock_time_recorded_source_status_claims_verified")
        & explicit_ist & clock_token.notna()
    )
    result["event_timestamp_ist"] = pd.NA
    result.loc[timestamp_ok, "event_timestamp_ist"] = (
        result.loc[timestamp_ok, "event_date"].dt.strftime("%Y-%m-%d")
        + "T" + clock_token.loc[timestamp_ok] + ":00+05:30"
    )
    result["timestamp_semantics"] = "exact_publication_timestamp_not_established"
    result.loc[
        result["timestamp_evidence_status"].eq("time_context_not_release_timestamp"),
        "timestamp_semantics",
    ] = "contextual_schedule_or_board_approval_not_release_timestamp"
    result.loc[timestamp_ok, "timestamp_semantics"] = "source_status_verified_publication_time_IST"

    timing_text = (event_time + " " + result.get("notes", pd.Series("", index=result.index)).fillna("").astype(str)).str.lower()
    result["event_timing_class"] = "session_timing_unresolved"
    result.loc[
        timing_text.str.contains(r"after trading hours|after.market|after-close|evening|midnight|17:30", regex=True),
        "event_timing_class",
    ] = "after_close_or_nontrading_day"
    result.loc[
        timing_text.str.contains(r"during market session|intraday|counting day|results evolve|exact intraday", regex=True),
        "event_timing_class",
    ] = "intraday_event_exposed"
    # The current 21-event registry is selected/exploratory, not a pre-registered
    # confirmatory sample. This is a claim guard, not an event exclusion.
    result["inference_scope"] = "exploratory_selected_event_not_preregistered"
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