"""Regression tests for the final release fixes.

Covers the observed-benchmark session calendar, the manifest price-field
guard, the dashboard price-convention statement, holdout uncertainty display,
overlap terminology, and dashboard/CSV agreement.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest

from scripts.run_analysis import resolve_price_fields
from src.data import calculate_aligned_returns, observed_benchmark_prices
from src.reporting import build_demo_dashboard, dashboard_evidence, price_convention_summary


# ---------------------------------------------------------------- calendar

def test_dates_with_missing_benchmark_close_are_not_analysis_sessions():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    # 2024-01-03 exists only because a stock was returned on that date.
    benchmark = pd.Series([100.0, 110.0, np.nan, 121.0], index=dates, name="NIFTY50")
    stocks = pd.DataFrame({"A": [50.0, 55.0, 57.0, 60.5]}, index=dates)

    aligned = calculate_aligned_returns(stocks, benchmark)

    assert list(aligned.index) == list(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]))
    # The stock return on 2024-01-04 spans 01-02 -> 01-04, the same interval
    # as the benchmark return on that row; nothing is forward-filled.
    assert aligned.loc["2024-01-04", "A"] == pytest.approx(60.5 / 55.0 - 1)
    assert aligned.loc["2024-01-04", "NIFTY50_RETURN"] == pytest.approx(121.0 / 110.0 - 1)
    assert observed_benchmark_prices(benchmark).index.equals(aligned.index)


def test_missing_stock_price_on_benchmark_session_stays_missing():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    benchmark = pd.Series([100.0, 101.0, 102.0], index=dates)
    stocks = pd.DataFrame({"A": [10.0, np.nan, 11.0]}, index=dates)

    aligned = calculate_aligned_returns(stocks, benchmark)

    assert np.isnan(aligned.loc["2024-01-02", "A"])
    assert np.isnan(aligned.loc["2024-01-03", "A"])


# ------------------------------------------------------------ price fields

def test_run_rejects_snapshot_recorded_with_a_different_price_field():
    with pytest.raises(ValueError, match="does not match the configured Close/Close"):
        resolve_price_fields({"adjustment_policy": {"stock_field": "Adj Close", "benchmark_field": "Close"}}, downloaded=False)


def test_run_records_price_fields_without_guessing():
    recorded = resolve_price_fields({"adjustment_policy": {"stock_field": "Close", "benchmark_field": "Close"}}, downloaded=False)
    assert (recorded["stock_field"], recorded["benchmark_field"]) == ("Close", "Close")
    unknown = resolve_price_fields({"classification": "unverified_local_input"}, downloaded=False)
    assert unknown["stock_field"] is None and unknown["benchmark_field"] is None
    live = resolve_price_fields(None, downloaded=True)
    assert (live["stock_field"], live["benchmark_field"]) == ("Close", "Close")


@pytest.mark.parametrize(
    ("manifest", "status", "phrase"),
    [
        ({"price_fields": {"stock_field": "Close", "benchmark_field": "Close"}}, "consistent", "not a total shareholder return"),
        ({"input_data_provenance": {"adjustment_policy": {"stock_field": "Adj Close", "benchmark_field": "Close"}}}, "mismatch", "stocks use Adj Close while the benchmark uses Close"),
        ({}, "not_recorded", "cannot be confirmed"),
    ],
)
def test_price_convention_summary_reads_the_manifest(manifest, status, phrase):
    summary = price_convention_summary(manifest)
    assert summary["status"] == status
    assert phrase in summary["note"]


# --------------------------------------------------------------- dashboard

def _write_analysis(tmp_path, holdout_ci=(-0.26, 0.38), price_fields=("Close", "Close")):
    events = [
        {"event_id": "E1", "event_date": "2024-01-02", "effective_event_date": "2024-01-02", "headline": "Event one",
         "category": "Policy", "episode_id": "EP1", "overlap_flag": True, "timestamp_evidence_status": "unresolved_missing_clock_time",
         "timestamp_review_required": True},
        {"event_id": "E2", "event_date": "2024-01-04", "effective_event_date": "2024-01-04", "headline": "Event two",
         "category": "Policy", "episode_id": "EP2", "overlap_flag": None,
         "timestamp_evidence_status": "clock_time_recorded_source_status_claims_verified", "timestamp_review_required": False},
        {"event_id": "E3", "event_date": "2024-03-01", "effective_event_date": "2024-03-01", "headline": "Event three",
         "category": "Earnings", "episode_id": "EP3", "overlap_flag": True,
         "timestamp_evidence_status": "unresolved_missing_clock_time", "timestamp_review_required": True},
    ]
    tables = {
        "event_master.csv": events,
        "multi_stock_event_study.csv": [{"event_id": "E1", "stock": "A", "event_day": 0, "abnormal_return": 0.01}],
        "multi_stock_car.csv": [
            {"event_id": "E1", "stock": "A", "window": "CAR_-1_+1", "CAR": 0.01},
            {"event_id": "E2", "stock": "A", "window": "CAR_-1_+1", "CAR": None},
        ],
        "multi_stock_significance.csv": [{"event_id": "E1", "stock": "A", "window": "CAR_-1_+1", "CAR_pct": 1.0,
                                          "raw_p_value": 0.2, "holm_p_value": 0.4, "significant_after_holm_5pct": False,
                                          "hac_raw_p_value": 0.3, "hac_holm_p_value": 0.6, "hac_significant_after_holm_5pct": False}],
        "event_window_audit.csv": [
            {"event_id": "E1", "event_window_overlap_count": 1, "event_window_overlap_event_ids": "E2", "volatility_window_overlap_count": 1},
            {"event_id": "E2", "event_window_overlap_count": 1, "event_window_overlap_event_ids": "E1", "volatility_window_overlap_count": 1},
            # E3 carries a manual registry flag but has no computed overlap.
            {"event_id": "E3", "event_window_overlap_count": 0, "event_window_overlap_event_ids": None, "volatility_window_overlap_count": 0},
        ],
        "event_pair_overlap_audit.csv": [
            {"event_id_a": "E1", "event_id_b": "E2", "overlapping_session_count": 3, "overlap_types": "temporal_overlap"},
            {"event_id_a": "E1", "event_id_b": "E3", "overlapping_session_count": 0, "overlap_types": "same_category_clustering"},
        ],
        "clustered_strategy_inference.csv": [
            {"sample": "Development", "events_available": 1, "independent_dependency_clusters": 1,
             "mean_excess_return_pct_point": -0.6, "mean_excess_return_pct_ci95_lower": None, "mean_excess_return_pct_ci95_upper": None},
            {"sample": "Holdout", "events_available": 2, "independent_dependency_clusters": 2, "bootstrap_replicates": 5000,
             "mean_excess_return_pct_point": 0.11, "mean_excess_return_pct_ci95_lower": holdout_ci[0],
             "mean_excess_return_pct_ci95_upper": holdout_ci[1]},
        ],
    }
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(tmp_path / name, index=False)
    stock_field, benchmark_field = price_fields
    (tmp_path / "manifest.json").write_text(json.dumps({
        "input_data_classification": "historical_market_data", "real_market_evidence": True,
        "snapshot_id": "snap-1", "generated_utc": "2026-10-29T00:00:00+00:00", "code_commit": "abc123def4567890",
        "input_checksums_sha256": {"prices": "f" * 64},
        "price_fields": {"stock_field": stock_field, "benchmark_field": benchmark_field},
        "event_window_sessions": 5, "holding_sessions": 3, "transaction_cost_bps_per_side": 10,
    }), encoding="utf-8")
    return tables


def _embedded_payload(document: str) -> dict:
    match = re.search(r'<script id="eventecho-data" type="application/json">(.*?)</script>', document, re.S)
    assert match, "embedded payload missing"
    return json.loads(match.group(1).replace("<\\/", "</"))


def test_dashboard_states_close_close_price_returns_and_no_stale_adjusted_claim(tmp_path):
    _write_analysis(tmp_path)
    document = build_demo_dashboard(tmp_path).read_text(encoding="utf-8")

    assert 'data-status="consistent"' in document
    assert "Both stocks and the benchmark use the Close field" in document
    assert "not a total shareholder return comparison" in document
    assert "adjusted-close fields" not in document
    assert "dividend conventions differ" not in document


def test_dashboard_has_no_literal_backslash_n_between_sections(tmp_path):
    _write_analysis(tmp_path)
    document = build_demo_dashboard(tmp_path).read_text(encoding="utf-8")
    assert "</section>\\n<section" not in document


def test_dashboard_warns_when_holdout_interval_includes_zero(tmp_path):
    _write_analysis(tmp_path, holdout_ci=(-0.26, 0.38))
    document = build_demo_dashboard(tmp_path).read_text(encoding="utf-8")
    assert "Warning: the holdout 95% interval includes zero" in document
    assert "-0.26 to +0.38 pp" in document
    assert "no reliable trading edge is established" in document


def test_dashboard_does_not_claim_zero_crossing_when_interval_excludes_zero(tmp_path):
    _write_analysis(tmp_path, holdout_ci=(0.05, 0.40))
    document = build_demo_dashboard(tmp_path).read_text(encoding="utf-8")
    assert "Warning: the holdout 95% interval includes zero" not in document
    assert "excludes zero" in document
    assert "remains exploratory evidence" in document


def test_dashboard_separates_computed_overlap_from_registry_flag(tmp_path):
    _write_analysis(tmp_path)
    document = build_demo_dashboard(tmp_path).read_text(encoding="utf-8")
    overlap = _embedded_payload(document)["evidence"]["overlap"]

    assert overlap["car_window_overlap_event_ids"] == ["E1", "E2"]
    assert overlap["car_window_overlap_pairs"] == 1
    # Manual registry flag: E1 True, E2 unset, E3 True (with no computed overlap).
    assert (overlap["registry_flag_true"], overlap["registry_flag_false"], overlap["registry_flag_unset"]) == (2, 0, 1)
    assert "CAR-window overlap (computed)" in document
    assert "Registry overlap flag (manual)" in document


def test_dashboard_payload_matches_csv_outputs_and_run_identity(tmp_path):
    _write_analysis(tmp_path)
    document = build_demo_dashboard(tmp_path).read_text(encoding="utf-8")
    payload = _embedded_payload(document)

    for name, rows in payload["datasets"].items():
        frame = pd.read_csv(tmp_path / name)
        expected = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
        assert rows == json.loads(json.dumps(expected, default=str)), name
    # Missing CAR stays missing; it is never shown as zero.
    assert payload["datasets"]["multi_stock_car.csv"][1]["CAR"] is None
    assert payload["evidence"]["run"]["snapshot_id"] == "snap-1"
    assert "snap-1" in document and "abc123def456" in document
    assert payload["evidence"]["timing"]["review_required"] == 2


def test_dashboard_script_never_turns_missing_numbers_into_zero(tmp_path):
    # Number(null) is 0 in JavaScript; the parser must return null instead,
    # and percentages must not be computed as num(x)*100 on missing values.
    _write_analysis(tmp_path)
    document = build_demo_dashboard(tmp_path).read_text(encoding="utf-8")
    script = re.search(r"<script>(.*?)</script>", document, re.S).group(1)
    assert 'if(v===null||v===undefined||(typeof v==="string"&&v.trim()===""))return null;' in script
    assert not re.search(r"num\([^()]*\)\*100", script)


def test_dashboard_table_cells_escape_text_and_only_render_explicit_html(tmp_path):
    _write_analysis(tmp_path)
    document = build_demo_dashboard(tmp_path).read_text(encoding="utf-8")
    script = re.search(r"<script>(.*?)</script>", document, re.S).group(1)
    assert 'v.__html!==undefined?v.__html' in script
    # Status pills are passed as explicit HTML, so they render as badges.
    assert "raw(status===\"includes_zero\"?'<span class=\"pill warn\">Includes zero</span>'" in script
    assert "Generated 2026-10-29 00:00 UTC" in document
