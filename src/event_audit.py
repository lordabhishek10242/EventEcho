"""Trading-session overlap and documented-confounder audits."""

from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import combinations

import pandas as pd


def _session_window(
    dates: pd.DatetimeIndex, center_position: int, relative_window: tuple[int, int]
) -> tuple[pd.DatetimeIndex, bool]:
    lower, upper = relative_window
    requested_start = center_position + lower
    requested_end = center_position + upper
    start = max(0, requested_start)
    end = min(len(dates) - 1, requested_end)
    if start > end:
        return pd.DatetimeIndex([]), True
    return dates[start : end + 1], requested_start < 0 or requested_end >= len(dates)


def _session_windows(
    dates: pd.DatetimeIndex,
    center_position: int,
    relative_windows: Sequence[tuple[int, int]],
) -> tuple[pd.DatetimeIndex, bool]:
    windows = [
        _session_window(dates, center_position, window)
        for window in relative_windows
    ]
    selected = pd.DatetimeIndex([])
    for window_dates, _ in windows:
        selected = selected.union(window_dates)
    truncated = any(is_truncated for _, is_truncated in windows)
    return selected.sort_values(), truncated


def audit_event_windows(
    events: pd.DataFrame,
    trading_dates: Sequence[object],
    event_window_radius: int = 5,
    volatility_windows: Sequence[tuple[int, int]] = ((-20, -1), (1, 20)),
) -> pd.DataFrame:
    """Audit overlaps in configured CAR and volatility session ranges.

    Effective dates must exist in the supplied trading calendar. Overlap means
    two distinct events share at least one actual trading session in their
    respective configured window; it is not evidence that the events are
    causally related.
    """
    if event_window_radius < 0 or not volatility_windows or any(
        lower > upper for lower, upper in volatility_windows
    ):
        raise ValueError("Invalid event or volatility window")
    dates = pd.DatetimeIndex(trading_dates).tz_localize(None).normalize().unique().sort_values()
    if dates.empty or dates.has_duplicates:
        raise ValueError("Trading calendar must contain unique sessions")
    if events["event_id"].isna().any() or events["event_id"].duplicated().any():
        raise ValueError("Event IDs must be present and unique")

    records: list[dict] = []
    for _, event in events.iterrows():
        event_date = pd.Timestamp(event["effective_event_date"]).tz_localize(None).normalize()
        position = dates.get_indexer([event_date])[0]
        if position < 0:
            raise ValueError(f"Effective event date {event_date.date()} is not a trading session")
        car_dates, car_truncated = _session_window(
            dates, int(position), (-event_window_radius, event_window_radius)
        )
        volatility_dates, volatility_truncated = _session_windows(
            dates, int(position), volatility_windows
        )
        records.append({
            "event_id": str(event["event_id"]),
            "event_date": event_date,
            "event_window_start": car_dates[0] if len(car_dates) else pd.NaT,
            "event_window_end": car_dates[-1] if len(car_dates) else pd.NaT,
            "event_window_sessions": len(car_dates),
            "event_window_status": "truncated_at_data_boundary" if car_truncated else "complete",
            "volatility_window_start": volatility_dates[0] if len(volatility_dates) else pd.NaT,
            "volatility_window_end": volatility_dates[-1] if len(volatility_dates) else pd.NaT,
            "volatility_window_sessions": len(volatility_dates),
            "volatility_windows": ";".join(
                f"{lower}:{upper}" for lower, upper in volatility_windows
            ),
            "volatility_window_status": (
                "truncated_at_data_boundary" if volatility_truncated else "complete"
            ),
            "_car_dates": car_dates,
            "_volatility_dates": volatility_dates,
        })

    for prefix, date_key in (("event_window", "_car_dates"), ("volatility_window", "_volatility_dates")):
        for record in records:
            partners: list[str] = []
            shared_sessions: set[pd.Timestamp] = set()
            for other in records:
                if record["event_id"] == other["event_id"]:
                    continue
                overlap = record[date_key].intersection(other[date_key])
                if len(overlap):
                    partners.append(other["event_id"])
                    shared_sessions.update(overlap)
            record[f"{prefix}_overlap_event_ids"] = ";".join(sorted(partners))
            record[f"{prefix}_overlap_count"] = len(partners)
            record[f"{prefix}_shared_sessions"] = len(shared_sessions)

    # Make the policy machine-readable: overlapping events remain in the
    # all-tests Holm family, are excluded from category summaries, and are
    # connected into dependence clusters for strategy bootstrap inference.
    for record in records:
        overlaps = record.get("event_window_overlap_count", 0)
        record["overlap_treatment"] = (
            "exclude_from_category_summary_cluster_for_strategy_uncertainty"
            if overlaps > 0 else
            "no_configured_event_window_overlap_detected"
        )
        record["CAR_tests_are_independent"] = False if overlaps > 0 else pd.NA

    columns = [key for key in records[0] if not key.startswith("_")] if records else []
    return pd.DataFrame([{key: row[key] for key in columns} for row in records], columns=columns)


def audit_documented_confounders(
    confounders: pd.DataFrame, event_window_audit: pd.DataFrame, trading_dates: Sequence[object]
) -> pd.DataFrame:
    """Attach manually documented potential confounders to event windows.

    The output distinguishes a documented potential confounder from temporal
    overlap and does not infer a causal relationship.
    """
    required = {
        "confounder_id", "event_id", "date", "headline", "source_name",
        "source_url", "potential_relevance", "provenance_note",
    }
    missing = required - set(confounders.columns)
    if missing:
        raise ValueError(f"Confounder records are missing fields: {sorted(missing)}")
    dates = pd.DatetimeIndex(trading_dates).tz_localize(None).normalize().unique().sort_values()
    windows = event_window_audit.set_index("event_id")
    rows = []
    for _, record in confounders.iterrows():
        event_id = str(record["event_id"])
        if event_id not in windows.index:
            raise ValueError(f"Confounder references unknown event: {event_id}")
        date = pd.Timestamp(record["date"]).tz_localize(None).normalize()
        window = windows.loc[event_id]
        in_window = bool(window["event_window_start"] <= date <= window["event_window_end"])
        position = dates.get_indexer([date])[0]
        event_position = dates.get_indexer([pd.Timestamp(window["event_date"])])[0]
        rows.append({
            "audit_type": "documented_potential_confounder",
            "confounder_id": record["confounder_id"],
            "event_id": event_id,
            "event_date": window["event_date"],
            "confounder_date": date,
            "session_offset_from_day0": int(position - event_position) if position >= 0 else pd.NA,
            "confounder_date_is_trading_session": position >= 0,
            "inside_event_window": in_window,
            "temporal_match_definition": "confounder calendar date falls inclusively between the configured event-window session dates",
            "headline": record["headline"],
            "source_name": record["source_name"],
            "source_url": record["source_url"],
            "potential_relevance": record["potential_relevance"],
            "provenance_note": record["provenance_note"],
            "causal_effect_estimated": False,
        })
    return pd.DataFrame(rows)


def build_event_relationship_audits(
    events: pd.DataFrame,
    event_window_audit: pd.DataFrame,
    confounder_audit: pd.DataFrame,
    trading_dates: Sequence[object],
    event_window_radius: int = 5,
    maximum_report_rows: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build event-level, unique-pair, and concise ranked audit tables.

    Episode IDs and categories are treated as clustering metadata, not causal
    evidence. A sourced potential development is reported only when its
    provenance is present and its recorded date is inside that event's event
    window. Pending or unverified event-master text remains an unverified
    review flag. Pair rows enumerate each unordered pair exactly once.
    """
    if maximum_report_rows < 1:
        raise ValueError("maximum_report_rows must be positive")
    if events.empty:
        empty_event = pd.DataFrame(columns=[
            "event_id", "event_date", "effective_event_date", "category", "description",
            "event_window_start", "event_window_end", "event_window_overlap_event_ids",
            "temporal_overlap_partner_count", "shared_episode_partner_count",
            "same_category_partner_count", "documented_potential_confounder_flag",
            "documented_potential_development_count", "documented_potential_development_in_window_count",
            "unverified_possibility_flag", "potential_confounder_explanation",
            "event_source_name", "event_source_url", "relationship_provenance",
            "documented_potential_sources", "documented_potential_provenance",
        ])
        empty_pairs = pd.DataFrame(columns=[
            "event_id_a", "event_id_b", "temporal_overlap", "overlapping_session_count",
            "overlapping_session_start", "overlapping_session_end", "shared_episode_id",
            "same_category", "documented_potential_development_in_shared_sessions",
            "unverified_possibility_flag", "overlap_types", "explanation",
            "relationship_provenance", "event_a_source_url", "event_b_source_url",
            "documented_source_name", "documented_source_url", "documented_provenance_note",
        ])
        empty_report = pd.DataFrame(columns=[
            "rank", "event_id", "temporal_overlap_partner_count",
            "documented_potential_development_in_window_count",
            "shared_episode_partner_count", "same_category_partner_count", "unverified_possibility_flag",
            "audit_highlight",
        ])
        return empty_event, empty_pairs, empty_report

    required = {"event_id", "effective_event_date"}
    if required - set(events.columns):
        raise ValueError(f"Events are missing fields: {sorted(required - set(events.columns))}")
    if events["event_id"].isna().any() or events["event_id"].duplicated().any():
        raise ValueError("Event IDs must be present and unique")
    window_by_id = event_window_audit.set_index("event_id", drop=False)
    if not set(events["event_id"].astype(str)).issubset(set(window_by_id.index.astype(str))):
        raise ValueError("Event-window audit must include every event")
    dates = pd.DatetimeIndex(trading_dates).tz_localize(None).normalize().unique().sort_values()
    if dates.empty or event_window_radius < 0:
        raise ValueError("A non-empty trading calendar and non-negative event window are required")

    docs = confounder_audit.copy()
    if docs.empty:
        docs = pd.DataFrame(columns=[
            "event_id", "confounder_id", "confounder_date", "inside_event_window",
            "headline", "source_name", "source_url", "provenance_note",
        ])
    docs_by_id = {
        str(event_id): group.copy()
        for event_id, group in docs.groupby("event_id", dropna=True)
    } if "event_id" in docs.columns else {}
    event_records: list[dict] = []
    event_metadata: dict[str, dict] = {}

    def present(value: object) -> bool:
        return pd.notna(value) and bool(str(value).strip())

    for _, event in events.iterrows():
        event_id = str(event["event_id"])
        window = window_by_id.loc[event_id]
        linked_docs = docs_by_id.get(event_id, pd.DataFrame())
        sourced_docs = linked_docs[
            linked_docs.apply(
                lambda row: present(row.get("source_url")) or present(row.get("provenance_note")),
                axis=1,
            )
        ] if not linked_docs.empty else linked_docs
        if not sourced_docs.empty and "inside_event_window" in sourced_docs:
            in_window_docs = sourced_docs[
                sourced_docs["inside_event_window"].fillna(False).astype(bool)
            ]
        else:
            in_window_docs = sourced_docs.iloc[0:0]
        status_text = " ".join(str(event.get(column, "")) for column in (
            "timing_confidence", "verification_status", "notes"
        )).lower()
        unverified = any(term in status_text for term in (
            "verify", "unverified", "pending", "required", "review", "not independently"
        ))
        event_metadata[event_id] = {
            "event": event,
            "window": window,
            "sourced_docs": sourced_docs,
            "in_window_docs": in_window_docs,
            "unverified": unverified,
        }
        event_records.append({"_event_id": event_id})

    for event_record in event_records:
        event_id = event_record.pop("_event_id")
        metadata = event_metadata[event_id]
        event = metadata["event"]
        window = metadata["window"]
        in_window_docs = metadata["in_window_docs"]
        sourced_docs = metadata["sourced_docs"]
        unverified = metadata["unverified"]
        source_fields = [
            f"{row.get('source_name', '')}: {row.get('source_url', '')}"
            for _, row in in_window_docs.iterrows()
        ]
        provenance_fields = [
            str(row["provenance_note"])
            for _, row in in_window_docs.iterrows()
            if present(row.get("provenance_note"))
        ]
        episode = event.get("episode_id")
        category = event.get("category")
        overlap_ids = str(window.get("event_window_overlap_event_ids", "") or "").split(";")
        overlap_ids = [value for value in overlap_ids if value]
        episode_partners = []
        category_partners = []
        for other_id, other_meta in event_metadata.items():
            if other_id == event_id:
                continue
            other = other_meta["event"]
            if present(episode) and present(other.get("episode_id")) and str(episode) == str(other["episode_id"]):
                episode_partners.append(other_id)
            if present(category) and present(other.get("category")) and str(category).strip().casefold() == str(other["category"]).strip().casefold():
                category_partners.append(other_id)
        flags = []
        explanations = []
        if overlap_ids:
            flags.append("temporal_overlap")
            explanations.append(
                f"Event-study window intersects the windows for {', '.join(overlap_ids)} in trading sessions."
            )
        if episode_partners:
            flags.append("shared_episode_policy_clustering")
            explanations.append(
                f"Shares event-master episode_id with {', '.join(sorted(episode_partners))}; this is clustering metadata, not causal evidence."
            )
        if category_partners:
            flags.append("same_category_clustering")
            explanations.append(
                f"Shares category with {', '.join(sorted(category_partners))}; category similarity alone does not establish a relationship."
            )
        if len(in_window_docs):
            flags.append("documented_potential_development")
            explanations.append(
                f"{len(in_window_docs)} sourced potential development record(s) fall inside this event window; sources/provenance are retained and not independently verified."
            )
        if unverified:
            flags.append("unverified_review_possibility")
            explanations.append(
                "Event-master timing, verification, or notes contain a pending/verification caution; this is an unverified possibility, not a confirmed confounder."
            )
        event_record.update({
            "event_id": event_id,
            "event_date": event.get("event_date", pd.NaT),
            "effective_event_date": event["effective_event_date"],
            "category": category,
            "description": event.get("headline", ""),
            "episode_id": episode,
            "event_window_start": window.get("event_window_start", pd.NaT),
            "event_window_end": window.get("event_window_end", pd.NaT),
            "event_window_sessions": window.get("event_window_sessions", 0),
            "event_window_status": window.get("event_window_status", "unknown"),
            "event_window_overlap_event_ids": ";".join(sorted(overlap_ids)),
            "temporal_overlap_partner_count": len(overlap_ids),
            "shared_episode_partner_count": len(episode_partners),
            "shared_episode_partner_ids": ";".join(sorted(episode_partners)),
            "same_category_partner_count": len(category_partners),
            "same_category_partner_ids": ";".join(sorted(category_partners)),
            "documented_potential_confounder_flag": bool(len(in_window_docs)),
            "documented_potential_development_count": len(sourced_docs),
            "documented_potential_development_in_window_count": len(in_window_docs),
            "unverified_possibility_flag": unverified,
            "potential_confounder_flags": ";".join(flags) if flags else "none_recorded",
            "potential_confounder_explanation": " ".join(explanations) if explanations else "No configured overlap or sourced potential development is flagged by this audit.",
            "potential_confounder_flag_explanations": json.dumps(
                dict(zip(flags, explanations)), ensure_ascii=True
            ),
            "event_source_name": event.get("source_name", ""),
            "event_source_url": event.get("source_url", ""),
            "event_verification_status": event.get("verification_status", ""),
            "timing_confidence": event.get("timing_confidence", ""),
            "event_notes": event.get("notes", ""),
            "relationship_provenance": "event_master.episode_id and event_master.category; no independent relationship source asserted",
            "documented_potential_sources": " | ".join(source_fields),
            "documented_potential_provenance": " | ".join(provenance_fields),
        })

    event_level = pd.DataFrame(event_records)
    pair_records: list[dict] = []
    session_windows: dict[str, pd.DatetimeIndex] = {}
    for event_id, metadata in event_metadata.items():
        event_date = pd.Timestamp(metadata["event"]["effective_event_date"]).tz_localize(None).normalize()
        center = dates.get_indexer([event_date])[0]
        if center < 0:
            raise ValueError(f"Effective event date {event_date.date()} is not a trading session")
        start = max(0, int(center) - event_window_radius)
        end = min(len(dates) - 1, int(center) + event_window_radius)
        session_windows[event_id] = dates[start : end + 1]
    for event_id_a, event_id_b in combinations(sorted(event_metadata), 2):
        metadata_a = event_metadata[event_id_a]
        metadata_b = event_metadata[event_id_b]
        event_a = metadata_a["event"]
        event_b = metadata_b["event"]
        row_a = window_by_id.loc[event_id_a]
        row_b = window_by_id.loc[event_id_b]
        shared_sessions = session_windows[event_id_a].intersection(session_windows[event_id_b])
        shared_episode = present(event_a.get("episode_id")) and present(event_b.get("episode_id")) and str(event_a["episode_id"]) == str(event_b["episode_id"])
        same_category = present(event_a.get("category")) and present(event_b.get("category")) and str(event_a["category"]).strip().casefold() == str(event_b["category"]).strip().casefold()
        shared_docs = []
        for metadata in (metadata_a, metadata_b):
            for _, doc in metadata["sourced_docs"].iterrows():
                doc_date = pd.to_datetime(doc.get("confounder_date"), errors="coerce")
                if pd.notna(doc_date) and pd.Timestamp(doc_date).normalize() in shared_sessions:
                    shared_docs.append(doc)
        types = []
        explanations = []
        if len(shared_sessions):
            types.append("temporal_overlap")
            explanations.append(f"The event-study windows share {len(shared_sessions)} trading session(s).")
        if shared_episode:
            types.append("shared_episode_policy_clustering")
            explanations.append(f"Both events carry episode_id {event_a['episode_id']} in the event master; no causal relationship is inferred.")
        if same_category:
            types.append("same_category_clustering")
            explanations.append(f"Both events are categorized as {event_a['category']}; same-category membership alone is not a documented policy relationship.")
        if shared_docs:
            types.append("documented_potential_development")
            explanations.append("A source-attributed potential-development record falls on a shared event-window session; provenance is supplied but not independently verified.")
        unverified = bool(metadata_a["unverified"] or metadata_b["unverified"])
        if unverified:
            types.append("unverified_review_possibility")
            explanations.append("At least one event has a verification/timing caution in the event master; it remains unverified.")
        if not types:
            types.append("none_detected")
            explanations.append("No temporal intersection, shared episode/category metadata, or sourced development in shared sessions was detected.")
        pair_records.append({
            "event_id_a": event_id_a,
            "event_id_b": event_id_b,
            "temporal_overlap": bool(len(shared_sessions)),
            "overlapping_session_count": len(shared_sessions),
            "overlapping_session_start": shared_sessions[0] if len(shared_sessions) else pd.NaT,
            "overlapping_session_end": shared_sessions[-1] if len(shared_sessions) else pd.NaT,
            "shared_episode_id": bool(shared_episode),
            "episode_id": event_a.get("episode_id") if shared_episode else "",
            "same_category": bool(same_category),
            "category": event_a.get("category") if same_category else "",
            "documented_potential_development_in_shared_sessions": bool(shared_docs),
            "documented_potential_development_count": len(shared_docs),
            "unverified_possibility_flag": unverified,
            "overlap_types": ";".join(types),
            "explanation": " ".join(explanations),
            "relationship_provenance": "event_master.episode_id/category only; no external relationship source asserted",
            "event_a_source_url": event_a.get("source_url", ""),
            "event_b_source_url": event_b.get("source_url", ""),
            "documented_source_name": " | ".join(str(doc.get("source_name", "")) for doc in shared_docs),
            "documented_source_url": " | ".join(str(doc.get("source_url", "")) for doc in shared_docs),
            "documented_provenance_note": " | ".join(str(doc.get("provenance_note", "")) for doc in shared_docs),
        })
    pair_level = pd.DataFrame(pair_records, columns=None if pair_records else [
        "event_id_a", "event_id_b", "temporal_overlap", "overlapping_session_count",
        "overlapping_session_start", "overlapping_session_end", "shared_episode_id", "episode_id",
        "same_category", "category", "documented_potential_development_in_shared_sessions",
        "documented_potential_development_count", "unverified_possibility_flag", "overlap_types",
        "explanation", "relationship_provenance", "event_a_source_url", "event_b_source_url",
        "documented_source_name", "documented_source_url", "documented_provenance_note",
    ])

    report = event_level.sort_values(
        [
            "temporal_overlap_partner_count",
            "documented_potential_development_in_window_count",
            "shared_episode_partner_count",
            "same_category_partner_count",
        ],
        ascending=[False, False, False, False],
        kind="stable",
    ).head(maximum_report_rows).copy()
    report["rank"] = range(1, len(report) + 1)
    report["audit_highlight"] = report.apply(
        lambda row: "; ".join(
            label for label, active in (
                ("temporal overlap", row["temporal_overlap_partner_count"] > 0),
                ("sourced potential development", row["documented_potential_development_in_window_count"] > 0),
                ("shared episode", row["shared_episode_partner_count"] > 0),
                ("same category", row["same_category_partner_count"] > 0),
                ("unverified review caution", row["unverified_possibility_flag"]),
            ) if active
        ) or "no overlap or sourced potential development flagged",
        axis=1,
    )
    report = report[[
        "rank", "event_id", "event_date", "effective_event_date", "category", "description",
        "temporal_overlap_partner_count", "documented_potential_development_in_window_count",
        "shared_episode_partner_count", "same_category_partner_count",
        "unverified_possibility_flag", "audit_highlight",
    ]]
    return event_level, pair_level, report