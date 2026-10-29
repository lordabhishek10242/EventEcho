#!/usr/bin/env python3
"""Run EventEcho using a local price CSV or optional Yahoo Finance download."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    BASE_COST_BPS_PER_SIDE,
    BENCHMARK_TICKER,
    COST_LEVELS_BPS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    DEVELOPMENT_EVENT_IDS,
    END_DATE,
    ESTIMATION_SESSIONS,
    EVENT_WINDOW_RADIUS,
    GAP_SESSIONS,
    HOLDING_SESSIONS,
    HOLDOUT_EVENT_IDS,
    START_DATE,
    STOCK_TICKERS,
)
from src.data import align_stock_prices_to_benchmark_calendar, calculate_aligned_returns, download_prices, load_price_csv, load_price_provenance, sha256_file  # noqa: E402
from src.snapshots import resolve_market_snapshot  # noqa: E402
from src.event_study import calculate_event_study  # noqa: E402
from src.events import chronological_split, validate_events  # noqa: E402
from src.event_audit import audit_event_windows, build_event_relationship_audits  # noqa: E402
from src.volatility import summarize_event_volatility  # noqa: E402
from src.category_analysis import summarize_category_car, summarize_category_volatility  # noqa: E402
from src.reporting import write_artifacts  # noqa: E402
from src.sensitivity import (  # noqa: E402
    event_timing_sensitivity, extreme_market_sensitivity, transaction_cost_sensitivity,
)
from src.statistics import event_study_significance  # noqa: E402
from src.clustered_inference import clustered_strategy_inference  # noqa: E402
from src.estimation_sensitivity import estimation_window_sensitivity  # noqa: E402
from src.robustness_ledger import build_robustness_evidence_ledger  # noqa: E402
from src.trading import event_portfolio_returns, simulate_trades, summarize_samples  # noqa: E402

LOGGER = logging.getLogger("eventecho")


def _portable_input_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return f"external file: {resolved.name}"


def _cross_stock_consistency(car_results: pd.DataFrame) -> pd.DataFrame:
    return (
        car_results.dropna(subset=["CAR"])
        .groupby(["event_id", "window"])["CAR"]
        .agg(
            mean_CAR_pct=lambda values: values.mean() * 100,
            median_CAR_pct=lambda values: values.median() * 100,
            cross_stock_std_pct=lambda values: values.std(ddof=1) * 100,
            positive_stock_fraction=lambda values: (values > 0).mean(),
            negative_stock_fraction=lambda values: (values < 0).mean(),
            stocks_observed="count",
        )
        .reset_index()
    )


def _single_stock_export(event_study: pd.DataFrame, stock: str) -> pd.DataFrame:
    result = event_study[event_study["stock"] == stock].copy()
    return result.rename(columns={
        "actual_return": f"{stock}_RETURN",
        "market_return": "NIFTY50_RETURN",
        "expected_return": "EXPECTED_RETURN",
        "abnormal_return": "ABNORMAL_RETURN",
    })


def run_analysis(
    events_path: Path,
    output_dir: Path,
    prices_path: Path | None = None,
    offline: bool = False,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    cost_bps_per_side: float = BASE_COST_BPS_PER_SIDE,
    snapshot_id: str | None = None,
    allow_synthetic: bool = False,
) -> dict[str, pd.DataFrame]:
    events = validate_events(pd.read_csv(events_path))
    snapshot_manifest = None
    provenance = None
    if snapshot_id is not None:
        prices_path, snapshot_manifest, provenance = resolve_market_snapshot(
            ROOT / "data" / "snapshots", snapshot_id
        )
    elif prices_path is not None:
        provenance = load_price_provenance(prices_path)
        if provenance.get("classification") == "synthetic" and not allow_synthetic:
            raise ValueError("Synthetic prices are validation-only; pass --allow-synthetic to opt in")
    if prices_path is not None:
        if not prices_path.exists():
            raise FileNotFoundError(f"Specified price file does not exist: {prices_path}")
        prices = load_price_csv(prices_path)
        missing = [column for column in ("NIFTY50",) if column not in prices.columns]
        if missing:
            raise ValueError(f"Local price CSV is missing required columns: {missing}")
        benchmark_prices = prices.pop("NIFTY50")
        stock_prices = prices
    else:
        if offline:
            raise FileNotFoundError("Offline mode requires --prices with a local price file")
        stock_prices, benchmark_prices = download_prices(
            STOCK_TICKERS, start_date, end_date, BENCHMARK_TICKER
        )
    # Align price levels first; only then compute close-to-close returns.
    # This prevents mismatched stock/benchmark calendars from pairing returns
    # that span different intervals. Missing prices are never forward-filled.
    aligned_stock_prices = align_stock_prices_to_benchmark_calendar(stock_prices, benchmark_prices)
    returns = calculate_aligned_returns(aligned_stock_prices, benchmark_prices)
    benchmark_returns = returns["NIFTY50_RETURN"]
    stocks = [stock for stock in STOCK_TICKERS if stock in returns.columns]
    if not stocks:
        raise ValueError("No configured stock return series are available")

    event_study, car_results, models = calculate_event_study(
        returns, events, stocks, ESTIMATION_SESSIONS, GAP_SESSIONS, EVENT_WINDOW_RADIUS
    )
    significance = event_study_significance(event_study, car_results, models)
    estimation_window_details, estimation_window_summary = estimation_window_sensitivity(
        returns, events, stocks, window_lengths=(60, 100, 150),
        gap_sessions=GAP_SESSIONS, event_window_radius=EVENT_WINDOW_RADIUS,
    )
    consistency = _cross_stock_consistency(car_results)
    # Explicitly audit overlapping event windows and shared episode/category metadata.
    # These are dependence warnings, not claims that events caused one another.
    event_window_audit = audit_event_windows(
        events, returns.index, event_window_radius=EVENT_WINDOW_RADIUS
    )
    event_level_audit, event_pair_overlap_audit, event_audit_highlights = (
        build_event_relationship_audits(
            events, event_window_audit, pd.DataFrame(), returns.index,
            event_window_radius=EVENT_WINDOW_RADIUS,
        )
    )
    event_volatility = summarize_event_volatility(returns, events, stocks, models)
    category_car_summary = summarize_category_car(events, car_results, event_window_audit)
    category_volatility_summary = summarize_category_volatility(
        events, event_volatility, event_window_audit
    )
    trades = simulate_trades(
        aligned_stock_prices, benchmark_returns, events, stocks, HOLDING_SESSIONS, cost_bps_per_side
    )
    event_returns = event_portfolio_returns(trades)
    split_events = chronological_split(events, DEVELOPMENT_EVENT_IDS, HOLDOUT_EVENT_IDS)
    labels = split_events.set_index("event_id")["sample"]
    event_trading = event_returns.copy()
    event_trading["sample"] = event_trading["event_id"].map(labels)
    sample_performance = summarize_samples(event_returns, DEVELOPMENT_EVENT_IDS, HOLDOUT_EVENT_IDS)
    clustered_performance_inference = clustered_strategy_inference(
        event_returns, events, event_window_audit,
        DEVELOPMENT_EVENT_IDS, HOLDOUT_EVENT_IDS,
    )
    sensitivity_levels = tuple(sorted(set(COST_LEVELS_BPS + (cost_bps_per_side,))))
    cost_sensitivity = transaction_cost_sensitivity(trades, sensitivity_levels)
    base_cost = cost_sensitivity.loc[
        cost_sensitivity["cost_bps_per_side"] == cost_bps_per_side
    ]
    if base_cost.empty:
        raise ValueError("Could not calculate the configured base transaction cost")
    baseline = base_cost.iloc[0]
    assessment = pd.DataFrame({
        "Finding": [
            "Number of events", "Baseline cost per side (bps)",
            "Baseline average net return (%)", "Baseline average excess return (%)",
            "Baseline event win rate (%)", "Average excess return at 30 bps per side (%)",
            "Benchmark outperformance rate at baseline (%)",
        ],
        "Value": [
            events["event_id"].nunique(), cost_bps_per_side,
            baseline["avg_net_return_pct"], baseline["avg_excess_return_pct"],
            baseline["event_win_rate_pct"],
            cost_sensitivity.loc[cost_sensitivity["cost_bps_per_side"] == 30, "avg_excess_return_pct"].iloc[0],
            baseline["benchmark_outperformance_rate_pct"],
        ],
    })
    covid_sensitivity = extreme_market_sensitivity(event_study, "EVT002") if "EVT002" in set(events.event_id) else pd.DataFrame()
    timing_sensitivity = event_timing_sensitivity(
        returns, events, stocks, car_results, ESTIMATION_SESSIONS, GAP_SESSIONS, EVENT_WINDOW_RADIUS
    )

    artifacts = {
        "event_master.csv": events,
        "multi_stock_event_study.csv": event_study,
        "multi_stock_car.csv": car_results,
        "cross_stock_consistency.csv": consistency,
        "event_window_audit.csv": event_window_audit,
        "event_level_audit.csv": event_level_audit,
        "event_pair_overlap_audit.csv": event_pair_overlap_audit,
        "event_relationship_highlights.csv": event_audit_highlights,
        "event_volatility.csv": event_volatility,
        "category_car_summary.csv": category_car_summary,
        "category_volatility_summary.csv": category_volatility_summary,
        "trade_candidates.csv": trades,
        "event_trading_results.csv": event_trading,
        "sample_performance.csv": sample_performance,
        "clustered_strategy_inference.csv": clustered_performance_inference,
        "transaction_cost_sensitivity.csv": cost_sensitivity,
        "final_strategy_assessment.csv": assessment,
        "multi_stock_significance.csv": significance,
        "estimation_window_sensitivity.csv": estimation_window_details,
        "estimation_window_sensitivity_summary.csv": estimation_window_summary,
        "robustness_evidence_ledger.csv": build_robustness_evidence_ledger(
            estimation_window_details, events, event_window_audit, event_level_audit
        ),
        "single_stock_event_study.csv": _single_stock_export(event_study, "RELIANCE"),
    }
    if not covid_sensitivity.empty:
        artifacts["covid_extreme_day_sensitivity.csv"] = covid_sensitivity
    if not timing_sensitivity.empty:
        artifacts["event_timing_sensitivity.csv"] = timing_sensitivity
    manifest = {
        "purpose": "Freshly generated exploratory event study and hypothetical trading simulation",
        "input_events": _portable_input_path(events_path),
        "input_prices": _portable_input_path(prices_path) if prices_path is not None else "Yahoo Finance download",
        "period_start": str(returns.index.min().date()),
        "period_end": str(returns.index.max().date()),
        "event_count": int(events.event_id.nunique()),
        "stock_universe": stocks,
        "benchmark": f"{BENCHMARK_TICKER} price index; not a total-return index",
        "estimation_sessions": ESTIMATION_SESSIONS,
        "gap_sessions": GAP_SESSIONS,
        "event_window_sessions": EVENT_WINDOW_RADIUS,
        "holding_sessions": HOLDING_SESSIONS,
        "transaction_cost_bps_per_side": cost_bps_per_side,
        "development_event_ids": list(DEVELOPMENT_EVENT_IDS),
        "holdout_event_ids": list(HOLDOUT_EVENT_IDS),
        "inference_note": "CAR t-tests and HAC results are approximate; cluster-bootstrap strategy intervals resample connected components of overlapping event windows and shared episode IDs. Small cluster counts limit inferential strength.",
        "data_note": "Stock returns use adjusted close; benchmark returns use price close. Local missing prices are not filled.",
        "provider_note": "This run uses a frozen local copy of Yahoo Finance history captured at retrieval time. The snapshot makes this run reproducible, but Yahoo Finance remains a third-party provider and the data are not independently verified; later provider downloads may differ.",
        "input_data_classification": (
            provenance.get("classification", "unverified_local_input") if provenance else "live_provider_download"
        ),
        "real_market_evidence": bool(
            provenance and provenance.get("classification") == "historical_market_data"
            and provenance.get("provenance_verified")
        ),
        "source_independently_verified": bool(provenance and provenance.get("source_independently_verified")),
        "input_checksums_sha256": {"prices": provenance.get("sha256")} if provenance else {},
        "input_data_provenance": provenance or {},
        "input_prices_exact_path": prices_path.resolve().as_posix() if prices_path else None,
        "snapshot_id": snapshot_id,
    }
    if snapshot_manifest is not None:
        manifest["snapshot_manifest"] = snapshot_manifest
        manifest["input_prices"] = f"data/snapshots/{snapshot_id}/prices.csv"
    write_artifacts(artifacts, output_dir, manifest)
    LOGGER.info("Wrote %d non-empty artifacts to %s", len(artifacts), output_dir)
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_RESULTS_DIR / "event_master.csv")
    parser.add_argument("--prices", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--offline", action="store_true", help="Require local prices and skip Yahoo Finance")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--cost-bps-per-side", type=float, default=BASE_COST_BPS_PER_SIDE)
    parser.add_argument("--snapshot-id", default=None, help="Use a verified frozen historical market snapshot")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow explicitly labelled synthetic prices for validation only")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        run_analysis(
            args.events, args.output_dir, args.prices, args.offline,
            args.start, args.end, args.cost_bps_per_side, args.snapshot_id, args.allow_synthetic,
        )
    except Exception as exc:
        LOGGER.error("Analysis failed: %s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()