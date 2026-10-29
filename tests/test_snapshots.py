import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import BENCHMARK_TICKER, STOCK_TICKERS
from src.data import sha256_file
from src.snapshots import (
    SnapshotAcquisitionError,
    create_market_snapshot,
    resolve_market_snapshot,
    validate_market_prices,
)


def provider_frame(dates, adjusted=100.0):
    return pd.DataFrame(
        {"Adj Close": [adjusted + i for i in range(len(dates))], "Close": [adjusted + 10 + i for i in range(len(dates))]},
        index=pd.DatetimeIndex(dates),
    )


def fake_provider(start, end, fail_ticker=None, responses=None):
    calls = []
    responses = responses or {}

    def download(ticker, requested_start, requested_end, frequency):
        calls.append((ticker, requested_start, requested_end, frequency))
        assert requested_start == start
        assert requested_end == end
        assert frequency == "1d"
        if ticker == fail_ticker:
            raise ConnectionError("offline test provider")
        return responses.get(ticker, provider_frame(pd.date_range(start, end, inclusive="left", freq="D")))

    return download, calls


def test_snapshot_writes_raw_canonical_manifest_and_reproducible_checksums(tmp_path):
    start, end = "2024-01-01", "2024-01-05"
    download, calls = fake_provider(start, end)

    manifest = create_market_snapshot(
        "fixture-1", start, end, pd.DataFrame(), download_fn=download, root=tmp_path
    )

    snapshot = tmp_path / "fixture-1"
    assert manifest["status"] == "complete"
    assert manifest["classification"] == "historical_market_data"
    assert manifest["source"] == "Yahoo Finance via yfinance"
    assert manifest["retrieved_at_utc"].endswith("+00:00")
    assert manifest["frequency"] == "1d"
    assert manifest["adjustment_policy"]["stock_field"] == "Adj Close"
    assert manifest["adjustment_policy"]["benchmark_field"] == "Close"
    assert manifest["requested_end_exclusive"] == end
    assert len(calls) == len(STOCK_TICKERS) + 1
    assert {call[0] for call in calls} == {*STOCK_TICKERS.values(), BENCHMARK_TICKER}
    assert all(call[1:] == (start, end, "1d") for call in calls)
    assert set(manifest["files_sha256"]) >= {
        "prices.csv", "validation.json", "README.md", "prices.provenance.json",
        *(f"raw/{label}.csv" for label in [*STOCK_TICKERS, "NIFTY50"]),
    }
    for relative_path, expected_hash in manifest["files_sha256"].items():
        assert sha256_file(snapshot / relative_path) == expected_hash

    canonical = pd.read_csv(snapshot / "prices.csv", parse_dates=["date"])
    assert canonical.date.min() == pd.Timestamp(start)
    assert canonical.date.max() == pd.Timestamp("2024-01-04")
    assert "2024-01-05" not in canonical.date.astype(str).tolist()
    assert canonical["RELIANCE"].iloc[0] == pytest.approx(100.0)
    assert canonical["NIFTY50"].iloc[0] == pytest.approx(110.0)
    sidecar = json.loads((snapshot / "prices.provenance.json").read_text())
    assert sidecar["classification"] == "historical_market_data"
    assert sidecar["sha256"] == manifest["files_sha256"]["prices.csv"]


def test_existing_snapshot_id_is_never_overwritten(tmp_path):
    start, end = "2024-01-01", "2024-01-05"
    download, _ = fake_provider(start, end)
    create_market_snapshot("fixed-id", start, end, pd.DataFrame(), download_fn=download, root=tmp_path)
    original_manifest = (tmp_path / "fixed-id" / "manifest.json").read_bytes()

    with pytest.raises(FileExistsError):
        create_market_snapshot("fixed-id", start, end, pd.DataFrame(), download_fn=download, root=tmp_path)

    assert (tmp_path / "fixed-id" / "manifest.json").read_bytes() == original_manifest


def test_missing_stock_ticker_is_recorded_and_snapshot_is_incomplete(tmp_path):
    start, end = "2024-01-01", "2024-01-05"
    failed_label = next(iter(STOCK_TICKERS))
    download, _ = fake_provider(start, end, fail_ticker=STOCK_TICKERS[failed_label])

    with pytest.raises(SnapshotAcquisitionError) as error:
        create_market_snapshot("missing-stock", start, end, pd.DataFrame(), download_fn=download, root=tmp_path)

    manifest = json.loads((error.value.snapshot_path / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert failed_label in manifest["unavailable_tickers"]
    assert manifest["provider_failures"][0]["ticker"] == STOCK_TICKERS[failed_label]
    assert failed_label in manifest["missing_required_stock_labels"]


def test_missing_benchmark_is_fatal_and_recorded(tmp_path):
    start, end = "2024-01-01", "2024-01-05"
    download, _ = fake_provider(start, end, fail_ticker=BENCHMARK_TICKER)

    with pytest.raises(SnapshotAcquisitionError) as error:
        create_market_snapshot("missing-benchmark", start, end, pd.DataFrame(), download_fn=download, root=tmp_path)

    manifest = json.loads((error.value.snapshot_path / "manifest.json").read_text())
    assert "NIFTY50" in manifest["unavailable_tickers"]
    assert any(item["code"] == "missing_benchmark_data" for item in manifest["diagnostics"]["fatal_errors"])
    assert (error.value.snapshot_path / "prices.csv").exists()


def test_duplicate_normalized_dates_are_fatal(tmp_path):
    dates = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 12:00"])
    prices = pd.DataFrame({"RELIANCE": [100, 101], "NIFTY50": [1000, 1001]}, index=dates)

    report = validate_market_prices(prices, "2024-01-01", "2024-01-03")

    assert report["status"] == "incomplete"
    assert report["fatal_errors"][0]["code"] == "duplicate_normalized_dates"


def test_nonpositive_and_infinite_price_values_are_fatal():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    columns = {label: [100.0, 101.0, 102.0] for label in STOCK_TICKERS}
    columns["NIFTY50"] = [1000.0, np.inf, 1020.0]
    columns["ITC"] = [100.0, 0.0, 102.0]

    report = validate_market_prices(pd.DataFrame(columns, index=dates), "2024-01-01", "2024-01-05")

    assert report["status"] == "incomplete"
    invalid = {(row["ticker"], row["count"]) for row in report["fatal_errors"] if row["code"] == "invalid_price_values"}
    assert ("NIFTY50", 1) in invalid
    assert ("ITC", 1) in invalid


def test_missing_values_and_large_jumps_are_warnings_not_silent_fills():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    columns = {label: [100.0, 101.0, 102.0, 103.0] for label in STOCK_TICKERS}
    columns["NIFTY50"] = [1000.0, 1010.0, 1600.0, np.nan]
    prices = pd.DataFrame(columns, index=dates)

    report = validate_market_prices(prices, "2024-01-01", "2024-01-05")

    assert report["status"] == "complete"
    codes = {warning["code"] for warning in report["warnings"]}
    assert "missing_price_observations" in codes
    assert "large_price_jumps_review" in codes
    assert report["missing_observations_by_ticker"]["NIFTY50"] == 1
    assert report["price_observations_by_ticker"]["NIFTY50"] == 3


def test_effective_event_session_and_window_eligibility_are_reported():
    dates = pd.bdate_range("2023-01-02", periods=180)
    columns = {label: np.linspace(100.0, 120.0, len(dates)) for label in STOCK_TICKERS}
    columns["NIFTY50"] = np.linspace(10000.0, 12000.0, len(dates))
    prices = pd.DataFrame(columns, index=dates)
    events = pd.DataFrame([{
        "event_id": "E1",
        "effective_event_date": dates[140],
    }, {
        "event_id": "WEEKEND",
        "effective_event_date": pd.Timestamp("2023-01-07"),
    }])

    report = validate_market_prices(prices, "2023-01-02", "2023-12-31", events)

    assert report["event_eligibility"][0]["available"]
    assert report["event_eligibility"][0]["stock_eligibility"]["RELIANCE"]["eligible"]
    assert not report["event_eligibility"][1]["available"]
    assert any(warning["code"] == "event_date_unavailable" for warning in report["warnings"])


def test_eligibility_extends_estimation_sample_backwards_over_missing_returns():
    dates = pd.bdate_range("2023-01-02", periods=180)
    columns = {label: np.linspace(100.0, 120.0, len(dates)) for label in STOCK_TICKERS}
    columns["NIFTY50"] = np.linspace(10000.0, 12000.0, len(dates))
    prices = pd.DataFrame(columns, index=dates)
    position = 140
    cutoff_position = position - 5 - 20 - 1
    prices.loc[dates[cutoff_position - 3], "NIFTY50"] = np.nan
    prices.loc[dates[cutoff_position - 8], "NIFTY50"] = np.nan
    events = pd.DataFrame([{"event_id": "E1", "effective_event_date": dates[position]}])

    report = validate_market_prices(prices, "2023-01-02", "2023-12-31", events)

    eligibility = report["event_eligibility"][0]["stock_eligibility"]["RELIANCE"]
    assert eligibility["eligible"]
    assert eligibility["estimation_observations_used"] == 100
    assert eligibility["estimation_observations_available_before_cutoff"] >= 100
    assert eligibility["estimation_sample_end"] <= eligibility["estimation_cutoff"]


def test_start_end_validation_and_network_failure_are_recorded_without_live_network(tmp_path):
    with pytest.raises(ValueError, match="earlier than exclusive"):
        create_market_snapshot("bad-range", "2024-01-05", "2024-01-05", pd.DataFrame(), root=tmp_path)

    def unavailable(ticker, start, end, frequency):
        raise OSError("network unavailable in deterministic test")

    with pytest.raises(SnapshotAcquisitionError) as error:
        create_market_snapshot(
            "offline-failure", "2024-01-01", "2024-01-05", pd.DataFrame(),
            download_fn=unavailable, root=tmp_path,
        )

    manifest = json.loads((error.value.snapshot_path / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert len(manifest["provider_failures"]) == len(STOCK_TICKERS) + 1
    assert any(item["ticker"] == BENCHMARK_TICKER for item in manifest["provider_failures"])


def test_snapshot_resolver_verifies_all_archived_files_and_rejects_tampering(tmp_path):
    start, end = "2024-01-01", "2024-01-05"
    download, _ = fake_provider(start, end)
    create_market_snapshot(
        "verified-snapshot", start, end, pd.DataFrame(), download_fn=download, root=tmp_path
    )

    price_path, manifest, provenance = resolve_market_snapshot(tmp_path, "verified-snapshot")

    assert price_path.name == "prices.csv"
    assert manifest["snapshot_id"] == "verified-snapshot"
    assert provenance["classification"] == "historical_market_data"
    assert provenance["sha256"] == sha256_file(price_path)
    with (tmp_path / "verified-snapshot" / "raw" / "RELIANCE.csv").open("a", encoding="utf-8") as file:
        file.write("tampered\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_market_snapshot(tmp_path, "verified-snapshot")


def test_acquisition_cli_reports_provider_unavailable_without_live_network(monkeypatch, tmp_path, capsys):
    import scripts.acquire_market_snapshot as acquisition_cli

    def offline_acquisition(**kwargs):
        path = tmp_path / "attempt"
        report = {"fatal_errors": [{"code": "missing_benchmark_data"}]}
        raise SnapshotAcquisitionError(path, report)

    monkeypatch.setattr(acquisition_cli, "create_market_snapshot", offline_acquisition)
    monkeypatch.setattr("sys.argv", [
        "acquire_market_snapshot.py",
        "--snapshot-id", "offline-attempt",
        "--events", str(Path("results/event_master.csv").resolve()),
        "--output-root", str(tmp_path),
    ])

    assert acquisition_cli.main() == 2
    captured = capsys.readouterr()
    assert "Snapshot incomplete" in captured.err
    assert "missing_benchmark_data" in captured.err



def test_event_eligibility_uses_benchmark_sessions_not_union_calendar():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    prices = pd.DataFrame(
        {label: [100.0, 101.0, 102.0] for label in STOCK_TICKERS},
        index=dates,
    )
    prices["NIFTY50"] = [100.0, np.nan, 102.0]
    events = pd.DataFrame([{
        "event_id": "EVT_TEST",
        "effective_event_date": pd.Timestamp("2024-01-02"),
    }])

    report = validate_market_prices(
        prices, "2024-01-01", "2024-01-04", events=events
    )

    event = report["event_eligibility"][0]
    assert report["observed_session_count"] == 3
    assert report["observed_benchmark_session_count"] == 2
    assert event["available"] is False
    assert event["warning"] == "effective_event_date_not_an_observed_benchmark_session"
