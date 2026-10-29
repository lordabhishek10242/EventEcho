"""Build a reviewer-facing ledger of robustness and attribution caveats."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_robustness_evidence_ledger(
    estimation_window_details: pd.DataFrame,
    events: pd.DataFrame,
    event_window_audit: pd.DataFrame,
    event_level_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize which CAR signals survive all estimation lengths and their caveats.

    Statistical stability across specifications is not proof of causality.
    Overlap and metadata flags are warnings only; absence of a configured flag
    does not establish that an event had no confounders.
    """
    required_detail = {
        "estimation_sessions", "event_id", "stock", "window", "CAR_pct",
        "hac_significant_after_holm_5pct", "significant_after_holm_5pct",
        "absolute_CAR_delta_vs_100_pct",
    }
    if not required_detail.issubset(estimation_window_details.columns):
        raise ValueError(f"estimation_window_details must include {sorted(required_detail)}")
    if events["event_id"].astype(str).duplicated().any():
        raise ValueError("events must contain unique event_id values")

    detail = estimation_window_details.copy()
    detail["event_id"] = detail["event_id"].astype(str)
    group_keys = ["event_id", "stock", "window"]
    rows: list[dict] = []
    expected_lengths = sorted(detail["estimation_sessions"].dropna().astype(int).unique())
    for key, group in detail.groupby(group_keys, sort=True):
        by_length = group.set_index("estimation_sessions")
        hac_sig = by_length["hac_significant_after_holm_5pct"].fillna(False).astype(bool)
        conventional_sig = by_length["significant_after_holm_5pct"].fillna(False).astype(bool)
        cars = by_length["CAR_pct"].dropna().astype(float)
        signs = np.sign(cars.to_numpy())
        all_same_sign = bool(len(signs) == len(expected_lengths) and (signs == signs[0]).all())
        stable = bool(len(hac_sig) == len(expected_lengths) and hac_sig.all())
        rows.append({
            "event_id": key[0],
            "stock": key[1],
            "window": key[2],
            "estimation_windows_tested": len(group),
            "hac_holm_significant_window_count": int(hac_sig.sum()),
            "conventional_holm_significant_window_count": int(conventional_sig.sum()),
            "hac_significant_at_all_estimation_windows": stable,
            "car_direction_stable_across_estimation_windows": all_same_sign,
            "mean_CAR_pct_across_estimation_windows": float(cars.mean()) if not cars.empty else np.nan,
            "median_absolute_CAR_change_vs_100_pct": float(group["absolute_CAR_delta_vs_100_pct"].median()),
            "max_absolute_CAR_change_vs_100_pct": float(group["absolute_CAR_delta_vs_100_pct"].max()),
            "evidence_label": (
                "HAC significance stable across all estimation windows"
                if stable else
                "HAC significance varies by estimation window"
                if hac_sig.any() else
                "No HAC Holm-significant result across tested windows"
            ),
        })
    ledger = pd.DataFrame(rows)
    metadata_cols = [
        column for column in (
            "event_id", "headline", "category", "episode_id",
            "timing_confidence", "verification_status", "event_time",
        ) if column in events.columns
    ]
    ledger = ledger.merge(events[metadata_cols].copy(), on="event_id", how="left", validate="many_to_one")
    audit_cols = [
        column for column in (
            "event_id", "event_window_overlap_count", "event_window_overlap_event_ids",
        ) if column in event_window_audit.columns
    ]
    if len(audit_cols) > 1:
        ledger = ledger.merge(event_window_audit[audit_cols], on="event_id", how="left", validate="many_to_one")
    episode_cols = [
        column for column in (
            "event_id", "shared_episode_partner_count", "shared_episode_partner_ids",
        ) if column in event_level_audit.columns
    ]
    if len(episode_cols) > 1:
        ledger = ledger.merge(event_level_audit[episode_cols], on="event_id", how="left", validate="many_to_one")

    for column in ("event_window_overlap_count", "shared_episode_partner_count"):
        if column not in ledger:
            ledger[column] = 0
        ledger[column] = pd.to_numeric(ledger[column], errors="coerce").fillna(0).astype(int)
    for column in ("event_window_overlap_event_ids", "shared_episode_partner_ids", "timing_confidence", "verification_status"):
        if column not in ledger:
            ledger[column] = ""
        ledger[column] = ledger[column].fillna("").astype(str)

    def review_status(row: pd.Series) -> str:
        if not row["hac_significant_at_all_estimation_windows"]:
            return "No stable HAC significance across all tested windows"
        timing_text = f'{row["timing_confidence"]} {row["verification_status"]}'.casefold()
        timing_caution = any(word in timing_text for word in (
            "unverified", "pending", "assumption", "review", "required", "unresolved",
        ))
        if row["event_window_overlap_count"] > 0 or row["shared_episode_partner_count"] > 0 or timing_caution:
            return "Stable statistical signal; attribution/timing caveat"
        return "Stable statistical signal; check external confounders before interpretation"

    ledger["review_status"] = ledger.apply(review_status, axis=1)
    ledger["interpretation_limit"] = (
        "Specification stability is not causality; selected event sample and unobserved confounders remain."
    )
    return ledger.sort_values(
        ["hac_significant_at_all_estimation_windows", "hac_holm_significant_window_count", "event_id", "stock", "window"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
