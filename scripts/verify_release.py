#!/usr/bin/env python3
"""Check that one analysis output directory is internally consistent.

These are invariant checks on the generated tables, not expected-value checks:
they recompute what can be recomputed (CAR sums, Holm adjustments, cost
arithmetic, sample summaries) and confirm that the dashboard embeds the same
manifest and CSV rows that are shipped. A JSON report is written next to the
outputs, and the exit code is non-zero when any check fails.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    BASE_COST_BPS_PER_SIDE, CAR_WINDOWS, DEVELOPMENT_EVENT_IDS, HOLDOUT_EVENT_IDS, STOCK_TICKERS,
)
from src.data import sha256_file  # noqa: E402
from src.statistics import holm_adjust  # noqa: E402

TOLERANCE = 1e-9


class Checks:
    def __init__(self) -> None:
        self.results: list[dict] = []

    def check(self, name: str, passed: bool, detail: object = None) -> None:
        self.results.append({"check": name, "passed": bool(passed), "detail": detail})

    @property
    def failed(self) -> list[dict]:
        return [row for row in self.results if not row["passed"]]


def _read(directory: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(directory / name)


def _close(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray, tolerance: float = TOLERANCE) -> bool:
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    same_missing = np.array_equal(np.isnan(left), np.isnan(right))
    finite = ~np.isnan(left)
    return bool(same_missing and np.allclose(left[finite], right[finite], atol=tolerance, rtol=0))


def verify(directory: Path, registry_path: Path, dashboard_path: Path | None) -> Checks:
    checks = Checks()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    registry = pd.read_csv(registry_path)
    events = _read(directory, "event_master.csv")
    study = _read(directory, "multi_stock_event_study.csv")
    cars = _read(directory, "multi_stock_car.csv")
    sig = _read(directory, "multi_stock_significance.csv")
    est = _read(directory, "estimation_window_sensitivity.csv")
    audit = _read(directory, "event_window_audit.csv")
    pairs = _read(directory, "event_pair_overlap_audit.csv")
    trades = _read(directory, "trade_candidates.csv")
    event_trades = _read(directory, "event_trading_results.csv")
    samples = _read(directory, "sample_performance.csv")
    costs = _read(directory, "transaction_cost_sensitivity.csv")
    clustered = _read(directory, "clustered_strategy_inference.csv")

    # ---- manifest and provenance
    fields = manifest.get("price_fields") or {}
    checks.check("manifest records Close/Close price fields",
                 fields.get("stock_field") == "Close" and fields.get("benchmark_field") == "Close", fields)
    policy = (manifest.get("input_data_provenance") or {}).get("adjustment_policy") or {}
    checks.check("snapshot adjustment policy agrees with run price fields",
                 policy.get("stock_field") == fields.get("stock_field") and policy.get("benchmark_field") == fields.get("benchmark_field"), policy)
    checks.check("manifest identifies frozen historical snapshot",
                 manifest.get("input_data_classification") == "historical_market_data"
                 and manifest.get("real_market_evidence") is True and bool(manifest.get("snapshot_id")),
                 manifest.get("snapshot_id"))
    sha = (manifest.get("input_checksums_sha256") or {}).get("prices")
    checks.check("price checksum recorded", bool(sha and re.fullmatch(r"[0-9a-f]{64}", sha)), sha)
    checks.check("code commit recorded", bool(manifest.get("code_commit")), manifest.get("code_commit"))
    declared = {row["file"]: row["rows"] for row in manifest.get("artifacts", [])}
    row_mismatch = {name: (rows, len(_read(directory, name))) for name, rows in declared.items()
                    if len(_read(directory, name)) != rows}
    checks.check("manifest artifact row counts match files", not row_mismatch, row_mismatch or len(declared))

    # ---- registry coverage
    checks.check("event IDs unique in output registry", not events["event_id"].duplicated().any())
    checks.check("output registry covers the source registry",
                 set(events["event_id"]) == set(registry["event_id"]),
                 sorted(set(registry["event_id"]) ^ set(events["event_id"])))
    stocks = list(manifest.get("stock_universe") or [])
    checks.check("stock universe is the configured universe", stocks == list(STOCK_TICKERS), stocks)

    # ---- daily panel and CAR arithmetic
    keys = ["event_id", "stock", "event_day"]
    checks.check("daily panel keys unique", not study.duplicated(keys).any())
    checks.check("abnormal = actual - expected",
                 _close(study["abnormal_return"], study["actual_return"] - study["expected_return"], 1e-12))
    car_keys = ["event_id", "stock", "window"]
    checks.check("CAR keys unique", not cars.duplicated(car_keys).any())
    expected_car_rows = events["event_id"].nunique() * len(stocks) * len(CAR_WINDOWS)
    checks.check("CAR panel has one row per event x stock x window", len(cars) == expected_car_rows,
                 {"rows": len(cars), "expected": expected_car_rows})
    recomputed = []
    for _, row in cars.iterrows():
        lower, upper = CAR_WINDOWS[row["window"]]
        group = study[(study["event_id"] == row["event_id"]) & (study["stock"] == row["stock"])
                      & study["event_day"].between(lower, upper)]
        recomputed.append(group["abnormal_return"].sum() if len(group) == upper - lower + 1 else np.nan)
    checks.check("CAR equals sum of abnormal returns over the full window (missing when incomplete)",
                 _close(cars["CAR"], np.asarray(recomputed)), {"complete_CAR_rows": int(cars["CAR"].notna().sum())})

    # ---- inference
    complete = cars.dropna(subset=["CAR"])
    merged = complete.merge(sig, on=car_keys, how="outer", indicator=True)
    checks.check("one significance row per complete CAR", (merged["_merge"] == "both").all(),
                 merged["_merge"].value_counts().to_dict())
    both = merged[merged["_merge"] == "both"]
    checks.check("significance CAR_pct equals CAR x 100", _close(both["CAR_pct"], both["CAR"] * 100, 1e-9))
    checks.check("conventional Holm p-values recompute over the full test family",
                 _close(sig["holm_p_value"], holm_adjust(sig["raw_p_value"].to_numpy()), 1e-12))
    checks.check("HAC Holm p-values recompute over the full test family",
                 _close(sig["hac_holm_p_value"], holm_adjust(sig["hac_raw_p_value"].to_numpy()), 1e-12))
    checks.check("raw p <= Holm p", bool((sig["raw_p_value"] <= sig["holm_p_value"] + 1e-15).all()))

    # ---- estimation-window sensitivity
    lengths = sorted(est["estimation_sessions"].unique().tolist())
    checks.check("estimation lengths are 60/100/150", lengths == [60, 100, 150], lengths)
    base = est[est["estimation_sessions"] == 100].set_index(car_keys).sort_index()
    main = sig.set_index(car_keys).sort_index()
    checks.check("100-session sensitivity rows equal the baseline inference",
                 base.index.equals(main.index) and _close(base["CAR_pct"], main["CAR_pct"], 1e-9)
                 and _close(base["raw_p_value"], main["raw_p_value"], 1e-12))
    key_sets = {length: set(map(tuple, group[car_keys].to_numpy())) for length, group in est.groupby("estimation_sessions")}
    checks.check("every estimation length covers the same event-stock-window keys",
                 len({frozenset(value) for value in key_sets.values()}) == 1,
                 {int(k): len(v) for k, v in key_sets.items()})
    for length, group in est.groupby("estimation_sessions"):
        checks.check(f"Holm recomputes within the {length}-session family",
                     _close(group["hac_holm_p_value"], holm_adjust(group["hac_raw_p_value"].to_numpy()), 1e-12))

    # ---- overlap audit
    partners = {row.event_id: set(str(row.event_window_overlap_event_ids).split(";")) - {"nan", ""}
                for row in audit.itertuples()}
    symmetric = all(event in partners.get(partner, set()) for event, linked in partners.items() for partner in linked)
    checks.check("CAR-window overlap is symmetric", symmetric)
    checks.check("overlap counts match partner lists",
                 all(len(partners[row.event_id]) == row.event_window_overlap_count for row in audit.itertuples()))
    pair_set = {tuple(sorted((r.event_id_a, r.event_id_b))) for r in pairs.itertuples() if r.overlapping_session_count > 0}
    audit_pairs = {tuple(sorted((a, b))) for a, linked in partners.items() for b in linked}
    checks.check("pair audit and event audit agree on overlapping CAR-window pairs", pair_set == audit_pairs,
                 {"pairs": len(pair_set)})

    # ---- strategy
    cost = float(manifest.get("transaction_cost_bps_per_side", BASE_COST_BPS_PER_SIDE)) / 10000
    checks.check("cost charged once on entry and once on exit",
                 _close(trades["gross_return"] - trades["net_return"], np.full(len(trades), 2 * cost), 1e-12))
    checks.check("entry strictly after Day 0 and exit after entry",
                 bool((pd.to_datetime(trades["entry_date"]) > pd.to_datetime(trades["effective_event_date"])).all()
                      and (pd.to_datetime(trades["exit_date"]) > pd.to_datetime(trades["entry_date"])).all()))
    checks.check("one trade per event and stock", not trades.duplicated(["event_id", "stock"]).any())
    portfolio = trades.groupby("event_id")["net_return"].mean()
    joined = event_trades.set_index("event_id")
    checks.check("event portfolio return is the equal-weight mean of stock trades",
                 _close(joined.loc[portfolio.index, "portfolio_net_return"], portfolio, 1e-12))
    checks.check("event excess = portfolio net - benchmark",
                 _close(joined["excess_return"], joined["portfolio_net_return"] - joined["benchmark_return"], 1e-12))
    dev, hold = set(DEVELOPMENT_EVENT_IDS), set(HOLDOUT_EVENT_IDS)
    checks.check("manifest split equals configured split",
                 set(manifest.get("development_event_ids", [])) == dev and set(manifest.get("holdout_event_ids", [])) == hold)
    dates = events.set_index("event_id")["effective_event_date"].pipe(pd.to_datetime)
    checks.check("development events all precede holdout events",
                 dates[list(dev)].max() < dates[list(hold)].min(),
                 {"last_development": str(dates[list(dev)].max().date()), "first_holdout": str(dates[list(hold)].min().date())})
    for sample, ids in (("Development", dev), ("Holdout", hold)):
        group = joined[joined.index.isin(ids)]
        row = samples[samples["sample"] == sample].iloc[0]
        checks.check(f"{sample} summary recomputes from event results",
                     int(row["number_of_events"]) == len(group)
                     and abs(row["mean_net_return_pct"] - group["portfolio_net_return"].mean() * 100) < 1e-9
                     and abs(row["mean_excess_return_pct"] - group["excess_return"].mean() * 100) < 1e-9,
                     {"events": len(group)})
        crow = clustered[clustered["sample"] == sample].iloc[0]
        checks.check(f"{sample} bootstrap point estimate matches sample summary",
                     int(crow["events_available"]) == len(group)
                     and abs(crow["mean_excess_return_pct_point"] - group["excess_return"].dropna().mean() * 100) < 1e-9)
        lower, upper = crow["mean_excess_return_pct_ci95_lower"], crow["mean_excess_return_pct_ci95_upper"]
        checks.check(f"{sample} bootstrap interval is ordered", bool(pd.isna(lower) or lower <= upper),
                     {"lower_pp": lower, "upper_pp": upper, "clusters": int(crow["independent_dependency_clusters"])})
    base_cost = costs[costs["cost_bps_per_side"] == manifest.get("transaction_cost_bps_per_side")].iloc[0]
    checks.check("baseline cost row equals all-event mean net return",
                 abs(base_cost["avg_net_return_pct"] - joined["portfolio_net_return"].mean() * 100) < 1e-9)
    gross_mean = trades.groupby("event_id")["gross_return"].mean().mean() * 100
    checks.check("cost scenarios differ from gross by exactly 2 x cost",
                 _close(costs["avg_net_return_pct"], gross_mean - 2 * costs["cost_bps_per_side"] / 100, 1e-9))

    # ---- pooled reactions, window choice, replication, classifier
    path = _read(directory, "caar_event_time.csv")
    pooled = _read(directory, "caar_window_summary.csv")
    everything = path[path["group"] == "All events"].set_index("event_day")["CAAR_pct"]
    complete_cars = cars.dropna(subset=["CAR"])
    for window, (lower, upper) in CAR_WINDOWS.items():
        per_event = complete_cars[complete_cars["window"] == window].groupby("event_id")["CAR"].mean() * 100
        from_path = everything[upper] - (everything[lower - 1] if lower - 1 in everything.index else 0.0)
        row = pooled[(pooled["group"] == "All events") & (pooled["window"] == window)].iloc[0]
        checks.check(f"pooled mean CAR {window} equals CAAR path and per-event CARs",
                     abs(row["mean_CAR_pct"] - per_event.mean()) < 1e-9 and abs(from_path - per_event.mean()) < 1e-9,
                     {"mean_CAR_pct": round(float(row["mean_CAR_pct"]), 6)})
    with_interval = pooled.dropna(subset=["mean_CAR_ci95_lower_pct", "mean_CAR_ci95_upper_pct"])
    checks.check("pooled intervals are ordered and only reported with 3+ clusters",
                 bool((with_interval["mean_CAR_ci95_lower_pct"] <= with_interval["mean_CAR_ci95_upper_pct"]).all()
                      and (with_interval["dependency_clusters"] >= 3).all()))
    choice = _read(directory, "window_choice_summary.csv").set_index("window")
    checks.check("window-choice counts equal significance table counts",
                 all(int(choice.loc[w, "hac_holm_significant"]) == int(sig.loc[sig["window"] == w, "hac_significant_after_holm_5pct"].sum())
                     and int(choice.loc[w, "conventional_holm_significant"]) == int(sig.loc[sig["window"] == w, "significant_after_holm_5pct"].sum())
                     for w in CAR_WINDOWS))
    generalization = _read(directory, "pattern_generalization.csv")
    checks.check("replication tests cover every window", set(generalization["window"]) == set(CAR_WINDOWS))
    classified = _read(directory, "event_headline_classification.csv")
    checks.check("every registry headline is classified once",
                 sorted(classified["event_id"]) == sorted(events["event_id"]) and not classified["event_id"].duplicated().any())
    corpus_path = ROOT / str(manifest.get("headline_corpus") or "results/headline_corpus.csv")
    checks.check("headline corpus checksum matches manifest",
                 corpus_path.is_file() and sha256_file(corpus_path) == manifest.get("headline_corpus_sha256"),
                 manifest.get("headline_corpus_sha256"))

    # ---- dashboard agreement
    if dashboard_path is not None:
        document = dashboard_path.read_text(encoding="utf-8")
        match = re.search(r'<script id="eventecho-data" type="application/json">(.*?)</script>', document, re.S)
        checks.check("dashboard embeds a data payload", match is not None)
        if match:
            payload = json.loads(match.group(1).replace("<\\/", "</"))
            checks.check("dashboard manifest is this run's manifest", payload["manifest"] == manifest,
                         payload["manifest"].get("generated_utc"))
            mismatched = []
            for name, rows in payload["datasets"].items():
                frame = _read(directory, name)
                expected = json.loads(json.dumps(frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records"), default=str))
                if rows != expected:
                    mismatched.append(name)
            checks.check("dashboard tables equal shipped CSVs", not mismatched, mismatched or len(payload["datasets"]))
        checks.check("dashboard has no stale adjusted-close disclaimer",
                     "adjusted-close fields" not in document and "dividend conventions differ" not in document)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "results" / "reproduced")
    parser.add_argument("--events", type=Path, default=ROOT / "results" / "event_master.csv")
    parser.add_argument("--dashboard", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    dashboard = args.dashboard or (args.analysis_dir / "eventecho_demo.html")
    checks = verify(args.analysis_dir, args.events, dashboard if dashboard.is_file() else None)
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_dir": args.analysis_dir.name,
        "dashboard_checked": dashboard.is_file(),
        "passed": len(checks.results) - len(checks.failed),
        "failed": len(checks.failed),
        "checks": checks.results,
    }
    destination = args.report or (args.analysis_dir / "release_validation.json")
    destination.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    for row in checks.results:
        print(("PASS " if row["passed"] else "FAIL ") + row["check"])
    print(f"{report['passed']} passed, {report['failed']} failed -> {destination}")
    return 1 if checks.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
