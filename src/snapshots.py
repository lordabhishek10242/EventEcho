"""Frozen historical market-data acquisition and integrity diagnostics."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Callable

import numpy as np
import pandas as pd

from .config import (
    BENCHMARK_TICKER,
    DEFAULT_DATA_DIR,
    END_DATE,
    ESTIMATION_SESSIONS,
    EVENT_WINDOW_RADIUS,
    GAP_SESSIONS,
    START_DATE,
    STOCK_TICKERS,
)
from .data import _extract_price_column, calculate_returns, load_price_provenance, sha256_file

SNAPSHOT_SCHEMA_VERSION = 1
STOCK_FIELD = "Adj Close"
BENCHMARK_FIELD = "Close"
FREQUENCY = "1d"
LARGE_JUMP_THRESHOLD = 0.40


class SnapshotAcquisitionError(RuntimeError):
    """An acquisition ended with fatal integrity errors; snapshot diagnostics remain."""

    def __init__(self, snapshot_path: Path, report: dict):
        super().__init__(f"Snapshot is incomplete; see {snapshot_path / 'manifest.json'}")
        self.snapshot_path = snapshot_path
        self.report = report


def resolve_market_snapshot(snapshot_root: str | Path, snapshot_id: str) -> tuple[Path, dict, dict]:
    """Return canonical CSV, manifest, and provenance after hash verification."""
    snapshot_id = _safe_snapshot_id(snapshot_id)
    snapshot_path = Path(snapshot_root) / snapshot_id
    manifest_path = snapshot_path / "manifest.json"
    prices_path = snapshot_path / "prices.csv"
    if not manifest_path.is_file() or not prices_path.is_file():
        raise FileNotFoundError(f"Snapshot {snapshot_id!r} is missing its manifest or prices.csv")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("snapshot_id") != snapshot_id:
        raise ValueError("Snapshot ID does not match the manifest")
    if manifest.get("status") != "complete" or manifest.get("classification") != "historical_market_data":
        raise ValueError(f"Snapshot {snapshot_id!r} is not a complete historical-data snapshot")
    for relative_path, expected_hash in manifest.get("files_sha256", {}).items():
        file_path = snapshot_path / relative_path
        if not file_path.is_file() or sha256_file(file_path) != expected_hash:
            raise ValueError(f"Snapshot file checksum mismatch: {relative_path}")
    provenance = load_price_provenance(prices_path)
    if provenance.get("snapshot_id") != snapshot_id or provenance["classification"] != "historical_market_data":
        raise ValueError(f"Snapshot {snapshot_id!r} provenance sidecar is missing or inconsistent")
    return prices_path, manifest, provenance


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_snapshot_id(snapshot_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", snapshot_id):
        raise ValueError("snapshot_id must be a simple filename-safe identifier")
    return snapshot_id


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def validate_market_prices(
    prices: pd.DataFrame,
    start_date: str,
    end_date_exclusive: str,
    events: pd.DataFrame | None = None,
    large_jump_threshold: float = LARGE_JUMP_THRESHOLD,
) -> dict:
    """Report fatal structural faults and non-fatal market-data warnings.

    Dates represented in the canonical provider response union are treated as
    observed candidate sessions, matching the analysis date index; weekdays are
    never inferred. Benchmark non-missing dates and per-stock alignment are
    separately reported. A jump is a review warning, not a claim that the source
    value is wrong.
    """
    required = [*STOCK_TICKERS, "NIFTY50"]
    fatal: list[dict] = []
    warnings: list[dict] = []
    if not isinstance(prices.index, pd.DatetimeIndex):
        fatal.append({"code": "invalid_date_index", "message": "Price index must be datetime."})
        return {"status": "incomplete", "fatal_errors": fatal, "warnings": warnings}

    normalized = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    if normalized.has_duplicates:
        fatal.append({"code": "duplicate_normalized_dates", "message": "Dates duplicate after timezone removal and normalization."})
        return {
            "status": "incomplete",
            "fatal_errors": fatal,
            "warnings": warnings,
            "observed_date_start": normalized.min() if len(normalized) else None,
            "observed_date_end": normalized.max() if len(normalized) else None,
        }
    if not normalized.is_monotonic_increasing:
        warnings.append({"code": "dates_sorted", "message": "Input observations were not sorted by date."})
    absent = [column for column in required if column not in prices.columns]
    if absent:
        fatal.append({"code": "missing_required_tickers", "tickers": absent})

    frame = prices.copy()
    frame.index = normalized
    frame = frame.sort_index()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date_exclusive).normalize()
    outside = frame.index[(frame.index < start) | (frame.index >= end)]
    if len(outside):
        fatal.append({"code": "dates_outside_requested_range", "count": len(outside)})

    counts: dict[str, int] = {}
    missing_counts: dict[str, int] = {}
    invalid_counts: dict[str, int] = {}
    coverage_by_ticker: dict[str, dict] = {}
    for column in required:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        counts[column] = int(values.notna().sum())
        missing_counts[column] = int(values.isna().sum())
        observed_dates = values.index[values.notna()]
        coverage_by_ticker[column] = {
            "first_observed_date": observed_dates.min() if len(observed_dates) else None,
            "last_observed_date": observed_dates.max() if len(observed_dates) else None,
            "observed_date_count": int(len(observed_dates)),
        }
        if counts[column] == 0:
            fatal.append({"code": "no_usable_prices", "ticker": column})
        invalid = values.notna() & (~np.isfinite(values) | (values <= 0))
        invalid_counts[column] = int(invalid.sum())
        if invalid.any():
            fatal.append({"code": "invalid_price_values", "ticker": column, "count": int(invalid.sum())})
        if values.isna().any():
            warnings.append({"code": "missing_price_observations", "ticker": column, "count": int(values.isna().sum())})
        if not invalid.any():
            returns = calculate_returns(values)
            jumps = returns[returns.abs() > large_jump_threshold]
            if len(jumps):
                warnings.append({
                    "code": "large_price_jumps_review",
                    "ticker": column,
                    "threshold_return": large_jump_threshold,
                    "dates": [date.date().isoformat() for date in jumps.index],
                    "returns": [float(value) for value in jumps],
                    "message": "Large return flagged for review; may reflect market movement, corporate action, or provider data and is not automatically rejected.",
                })

    sessions = pd.DatetimeIndex(frame.index).unique().sort_values()
    benchmark_sessions = (
        pd.DatetimeIndex(frame.index[frame["NIFTY50"].notna()]).unique().sort_values()
        if "NIFTY50" in frame else pd.DatetimeIndex([])
    )
    alignment: dict[str, int] = {}
    stock_dates_outside_benchmark: dict[str, int] = {}
    if len(sessions):
        for column in STOCK_TICKERS:
            if column in frame:
                alignment[column] = int(frame.loc[benchmark_sessions, column].notna().sum())
                stock_dates_outside_benchmark[column] = int(
                    (frame[column].notna() & ~frame.index.isin(benchmark_sessions)).sum()
                )

    event_diagnostics = []
    if events is not None:
        returns_frame = (
            frame[required].reindex(benchmark_sessions).pct_change(fill_method=None)
            if not absent and not any(invalid_counts.values()) else pd.DataFrame(index=benchmark_sessions)
        )
        for _, event in events.iterrows():
            event_id = str(event.get("event_id", ""))
            effective = pd.to_datetime(event.get("effective_event_date"), errors="coerce")
            item = {"event_id": event_id, "effective_event_date": effective, "available": False, "stock_eligibility": {}}
            if pd.isna(effective) or effective.normalize() not in benchmark_sessions:
                item["warning"] = "effective_event_date_not_an_observed_benchmark_session"
                warnings.append({"code": "event_date_unavailable", "event_id": event_id, "effective_event_date": effective})
                event_diagnostics.append(item)
                continue
            item["available"] = True
            # Event eligibility is defined on observed benchmark sessions, not
            # the union of dates returned by all instruments.
            position = benchmark_sessions.get_loc(effective.normalize())
            first_event = position - EVENT_WINDOW_RADIUS
            end_estimation = first_event - GAP_SESSIONS - 1
            start_estimation = end_estimation - ESTIMATION_SESSIONS + 1
            for stock in STOCK_TICKERS:
                if stock not in returns_frame.columns:
                    item["stock_eligibility"][stock] = {"eligible": False, "reason": "stock_column_missing"}
                    continue
                event_dates = benchmark_sessions[max(0, first_event):min(len(benchmark_sessions), position + EVENT_WINDOW_RADIUS + 1)]
                available_estimation = (
                    returns_frame.loc[benchmark_sessions[:end_estimation + 1], [stock, "NIFTY50"]].dropna()
                    if end_estimation >= 0 else pd.DataFrame()
                )
                estimation_count = int(len(available_estimation))
                estimation_sample = available_estimation.tail(ESTIMATION_SESSIONS)
                event_count = int(returns_frame.loc[event_dates, [stock, "NIFTY50"]].dropna().shape[0]) if len(event_dates) else 0
                eligible = len(estimation_sample) == ESTIMATION_SESSIONS and event_count == 2 * EVENT_WINDOW_RADIUS + 1
                item["stock_eligibility"][stock] = {
                    "eligible": bool(eligible),
                    "estimation_observations_available_before_cutoff": estimation_count,
                    "estimation_observations_used": int(len(estimation_sample)),
                    "estimation_sample_start": estimation_sample.index[0] if len(estimation_sample) else None,
                    "estimation_sample_end": estimation_sample.index[-1] if len(estimation_sample) else None,
                    "estimation_cutoff": benchmark_sessions[end_estimation] if end_estimation >= 0 else None,
                    "event_window_observations": event_count,
                    "required_estimation_observations": ESTIMATION_SESSIONS,
                    "required_event_window_observations": 2 * EVENT_WINDOW_RADIUS + 1,
                }
                if not eligible:
                    warnings.append({"code": "event_analysis_window_ineligible", "event_id": event_id, "stock": stock, "estimation_observations_available_before_cutoff": estimation_count, "estimation_observations_used": int(len(estimation_sample)), "event_window_observations": event_count})
            event_diagnostics.append(item)

    return {
        "status": "complete" if not fatal else "incomplete",
        "requested_start_inclusive": start,
        "requested_end_exclusive": end,
        "observed_date_start": frame.index.min() if len(frame) else None,
        "observed_date_end": frame.index.max() if len(frame) else None,
        "observed_session_count": len(sessions),
        "observed_benchmark_session_count": len(benchmark_sessions),
        "session_calendar_source": "canonical union of observed provider response dates; no weekday calendar inferred",
        "event_analysis_calendar_source": "observed dates with non-missing benchmark close; stock prices are reindexed to this calendar before returns are computed",
        "required_tickers": required,
        "price_observations_by_ticker": counts,
        "date_coverage_by_ticker": coverage_by_ticker,
        "missing_observations_by_ticker": missing_counts,
        "invalid_observations_by_ticker": invalid_counts,
        "benchmark_aligned_observations_by_stock": alignment,
        "stock_observations_outside_benchmark_calendar": stock_dates_outside_benchmark,
        "fatal_errors": fatal,
        "warnings": warnings,
        "event_eligibility": event_diagnostics,
    }


def _yfinance_download(ticker: str, start: str, end: str, frequency: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance before running market-data acquisition") from exc
    return yf.download(
        ticker,
        start=start,
        end=end,
        interval=frequency,
        auto_adjust=False,
        actions=False,
        repair=False,
        keepna=True,
        progress=False,
        threads=False,
        group_by="column",
    )


def create_market_snapshot(
    snapshot_id: str,
    start_date: str,
    end_date_exclusive: str,
    events: pd.DataFrame,
    download_fn: Callable[[str, str, str, str], pd.DataFrame] = _yfinance_download,
    root: str | Path = DEFAULT_DATA_DIR / "snapshots",
    frequency: str = FREQUENCY,
    source: str = "Yahoo Finance via yfinance",
    stock_adjustment: str = "adjusted-close",
    benchmark_adjustment: str = "price-index-close",
) -> dict:
    """Acquire, validate, and freeze one snapshot without replacing prior data.

    ``download_fn`` is injectable for deterministic tests; production defaults
    to an explicit yfinance request with auto_adjust disabled. Stocks use Adj
    Close and the benchmark uses Close to preserve the established analysis
    convention. Provider end dates are exclusive.
    """
    snapshot_id = _safe_snapshot_id(snapshot_id)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date_exclusive).normalize()
    if start >= end:
        raise ValueError("start_date must be earlier than exclusive end_date")
    if frequency != "1d":
        raise ValueError("Only daily ('1d') snapshots are currently supported")
    if stock_adjustment != "adjusted-close" or benchmark_adjustment != "price-index-close":
        raise ValueError("Supported policy is stock adjusted-close and benchmark price-index-close")
    stock_field = STOCK_FIELD if stock_adjustment == "adjusted-close" else None
    benchmark_field = BENCHMARK_FIELD if benchmark_adjustment == "price-index-close" else None

    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    snapshot_path = base / snapshot_id
    snapshot_path.mkdir(exist_ok=False)
    acquisition_started = _timestamp()
    raw_dir = snapshot_path / "raw"
    raw_dir.mkdir()
    provider_failures: list[dict] = []
    unavailable: list[str] = []
    response_rows: dict[str, int] = {}
    series: dict[str, pd.Series] = {}
    instruments = [(label, ticker, stock_field) for label, ticker in STOCK_TICKERS.items()]
    instruments.append(("NIFTY50", BENCHMARK_TICKER, benchmark_field))

    for label, ticker, field in instruments:
        try:
            response = download_fn(ticker, start.date().isoformat(), end.date().isoformat(), frequency)
            if response is None or response.empty:
                unavailable.append(label)
                provider_failures.append({"label": label, "ticker": ticker, "error": "provider returned no rows"})
                continue
            raw_path = raw_dir / f"{label}.csv"
            response.to_csv(raw_path)
            response_rows[label] = int(len(response))
            selected = _extract_price_column(response, field, ticker).rename(label)
            if selected.empty or selected.notna().sum() == 0:
                unavailable.append(label)
                provider_failures.append({"label": label, "ticker": ticker, "error": f"no usable {field} values"})
                continue
            series[label] = selected
        except Exception as exc:
            unavailable.append(label)
            provider_failures.append({"label": label, "ticker": ticker, "error": f"{type(exc).__name__}: {exc}"})

    retrieved = _timestamp()

    canonical_path = snapshot_path / "prices.csv"
    if series:
        combined = pd.concat(series.values(), axis=1).sort_index()
        combined.index.name = "date"
        combined.to_csv(canonical_path)
    else:
        combined = pd.DataFrame(
            columns=[*STOCK_TICKERS, "NIFTY50"],
            index=pd.DatetimeIndex([], name="date"),
        )

    report = validate_market_prices(combined, start.date().isoformat(), end.date().isoformat(), events)
    if unavailable:
        report["fatal_errors"].append({"code": "unavailable_required_tickers", "tickers": sorted(set(unavailable))})
        report["status"] = "incomplete"
    if "NIFTY50" not in series:
        report["fatal_errors"].append({"code": "missing_benchmark_data", "ticker": BENCHMARK_TICKER})
        report["status"] = "incomplete"
    validation_path = snapshot_path / "validation.json"
    _write_json(validation_path, report)
    (snapshot_path / "README.md").write_text(
        "# Frozen market-data snapshot\n\n"
        "This directory contains provider responses, a canonical `prices.csv` input, diagnostics, and manifests. "
        "See `manifest.json` for source, timestamps, adjustment policy, coverage, and checksums. "
        "Provider data may be revised; archive checksums identify these exact downloaded bytes.\n",
        encoding="utf-8",
    )

    try:
        import yfinance
        provider_version = yfinance.__version__
    except ImportError:
        provider_version = None
    missing_stock_labels = [label for label in STOCK_TICKERS if label not in series]
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "status": report["status"],
        "classification": "historical_market_data" if report["status"] == "complete" else "incomplete_historical_acquisition",
        "source": source,
        "provider": "Yahoo Finance",
        "package": "yfinance",
        "package_version": provider_version,
        "retrieved_at_utc": retrieved,
        "acquisition_started_at_utc": acquisition_started,
        "requested_start_inclusive": start.date().isoformat(),
        "requested_end_exclusive": end.date().isoformat(),
        "frequency": frequency,
        "ticker_symbols": {**STOCK_TICKERS, "NIFTY50": BENCHMARK_TICKER},
        "adjustment_policy": {
            "yfinance_auto_adjust": False,
            "stock_adjustment": stock_adjustment,
            "stock_field": stock_field,
            "stock_interpretation": "Yahoo Finance adjusted close, including provider adjustments for distributions/splits where supplied",
            "benchmark_adjustment": benchmark_adjustment,
            "benchmark_field": benchmark_field,
            "benchmark_interpretation": "NIFTY 50 price-index close; not a total-return index",
            "actions_downloaded": False,
        },
        "exclusive_end_semantics": "The requested end date is exclusive at the provider; observations on end_date_exclusive are not requested.",
        "unavailable_tickers": sorted(set(unavailable)),
        "provider_failures": provider_failures,
        "provider_response_rows": response_rows,
        "missing_required_stock_labels": missing_stock_labels,
        "diagnostics": report,
        "files_sha256": {},
        "limitations": [
            "Yahoo Finance data can be revised after retrieval.",
            "The snapshot checksum guarantees byte identity, not provider correctness or independent source verification.",
            "Stock adjusted close and benchmark price-index close have different dividend conventions.",
            "Daily data cannot resolve intraday announcement timing.",
        ],
    }
    for path in sorted(snapshot_path.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["files_sha256"][path.relative_to(snapshot_path).as_posix()] = sha256_file(path)
    sidecar_path = snapshot_path / "prices.provenance.json"
    sidecar = {
        "classification": manifest["classification"],
        "snapshot_id": snapshot_id,
        "source": source,
        "provider": manifest["provider"],
        "retrieved_at_utc": retrieved,
        "adjustment_policy": manifest["adjustment_policy"],
        "sha256": manifest["files_sha256"].get("prices.csv"),
        "manifest": "manifest.json",
        "provenance_verified": bool(manifest["files_sha256"].get("prices.csv")),
        "checksum_verified": bool(manifest["files_sha256"].get("prices.csv")),
        "source_independently_verified": False,
    }
    _write_json(sidecar_path, sidecar)
    manifest["files_sha256"]["prices.provenance.json"] = sha256_file(sidecar_path)
    _write_json(snapshot_path / "manifest.json", manifest)
    if report["fatal_errors"]:
        raise SnapshotAcquisitionError(snapshot_path, report)
    return manifest
