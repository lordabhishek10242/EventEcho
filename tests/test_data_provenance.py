import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import load_price_provenance, sha256_file
import scripts.run_analysis as analysis_cli
from scripts.run_analysis import run_analysis


def test_price_provenance_binds_synthetic_classification_to_checksum(tmp_path):
    prices = tmp_path / "prices.csv"
    prices.write_text("date,A,NIFTY50\n2024-01-01,10,100\n", encoding="utf-8")
    sidecar = tmp_path / "prices.provenance.json"
    sidecar.write_text(json.dumps({
        "classification": "synthetic",
        "sha256": sha256_file(prices),
        "origin": "test fixture",
    }), encoding="utf-8")

    provenance = load_price_provenance(prices)

    assert provenance["classification"] == "synthetic"
    assert provenance["provenance_verified"]
    assert provenance["sha256"] == sha256_file(prices)


def test_price_provenance_rejects_mismatched_checksum(tmp_path):
    prices = tmp_path / "prices.csv"
    prices.write_text("date,A,NIFTY50\n2024-01-01,10,100\n", encoding="utf-8")
    (tmp_path / "prices.provenance.json").write_text(json.dumps({
        "classification": "synthetic",
        "sha256": "0" * 64,
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum does not match"):
        load_price_provenance(prices)


def test_known_synthetic_input_requires_explicit_cli_opt_in(tmp_path):
    prices = tmp_path / "prices.csv"
    prices.write_text("date,A,NIFTY50\n2024-01-01,10,100\n", encoding="utf-8")
    (tmp_path / "prices.provenance.json").write_text(json.dumps({
        "classification": "synthetic",
        "sha256": sha256_file(prices),
    }), encoding="utf-8")
    events = pd.DataFrame([{
        "event_id": "E1",
        "event_date": "2024-01-01",
        "effective_event_date": "2024-01-01",
        "headline": "Synthetic test event",
        "category": "Test",
        "source_name": "Test",
        "source_url": "https://example.test/event",
        "episode_id": "EP1",
        "timing_confidence": "test fixture",
    }])
    events_path = tmp_path / "events.csv"
    events.to_csv(events_path, index=False)

    with pytest.raises(ValueError, match="validation-only.*--allow-synthetic"):
        run_analysis(
            events_path,
            tmp_path / "out",
            prices_path=prices,
            offline=True,
        )


def test_unlabeled_local_input_is_not_claimed_as_historical(tmp_path):
    prices = tmp_path / "prices.csv"
    prices.write_text("date,A,NIFTY50\n2024-01-01,10,100\n", encoding="utf-8")

    provenance = load_price_provenance(prices)

    assert provenance["classification"] == "unverified_local_input"
    assert not provenance["provenance_verified"]


def test_analysis_snapshot_selection_records_exact_input_and_provenance(monkeypatch, tmp_path):
    prices = tmp_path / "prices.csv"
    dates = pd.bdate_range("2016-01-01", periods=260)
    frame = pd.DataFrame(
        {label: 100.0 + pd.Series(range(len(dates))).to_numpy() * (1 + i / 10)
         for i, label in enumerate(["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "ITC", "NIFTY50"])},
        index=dates,
    )
    frame.index.name = "date"
    frame.to_csv(prices)
    provenance = {
        "classification": "historical_market_data",
        "snapshot_id": "fixture-snapshot",
        "source": "mock provider fixture",
        "provider": "test provider",
        "retrieved_at_utc": "2026-01-01T00:00:00+00:00",
        "adjustment_policy": {"stock_field": "Adj Close", "benchmark_field": "Close"},
        "sha256": sha256_file(prices),
        "provenance_verified": True,
        "source_independently_verified": False,
    }
    monkeypatch.setattr(
        analysis_cli,
        "resolve_market_snapshot",
        lambda snapshot_root, snapshot_id: (prices, {"snapshot_id": snapshot_id}, provenance),
    )
    output = tmp_path / "snapshot-analysis"

    run_analysis(
        Path("results/event_master.csv"),
        output,
        offline=True,
        snapshot_id="fixture-snapshot",
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["snapshot_id"] == "fixture-snapshot"
    assert manifest["input_data_classification"] == "historical_market_data"
    assert manifest["input_prices_exact_path"] == prices.resolve().as_posix()
    assert manifest["input_checksums_sha256"]["prices"] == sha256_file(prices)
    assert manifest["real_market_evidence"] is True
    assert manifest["source_independently_verified"] is False
