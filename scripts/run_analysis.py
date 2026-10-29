#!/usr/bin/env python3
"""Run EventEcho using a local price CSV or optional Yahoo Finance download."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
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
from src.snapshots import BENCHMARK_FIELD, STOCK_FIELD, resolve_market_snapshot  # noqa: E402
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
from src.clustered_inference import _dependency_clusters, clustered_strategy_inference  # noqa: E402
from src.headline_classifier import run_headline_classification  # noqa: E402
from src.pooled_analysis import _group_definitions, caar_tables, pattern_generalization, window_choice_tables  # noqa: E402
from src.earnings_study import HOLDOUT_START as EARNINGS_HOLDOUT_START, load_earnings_calendar, registry_robust_tests, run_earnings_study  # noqa: E402
from src.config import STOCK_SECTORS  # noqa: E402
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


def _code_commit() -> str | None:
    """Identify the code revision that produced the run, when it can be known."""
    commit = os.environ.get("GITHUB_SHA")
    if commit:
        return commit
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def resolve_price_fields(provenance: dict | None, downloaded: bool) -> dict:
    """Return the price fields actually used, and refuse a conflicting input.

    The manifest's return-convention statement must describe the data that was
    analysed. A snapshot or sidecar that records a different field (for example
    an older Adj Close snapshot) is rejected instead of being labelled Close.
    Inputs that record no policy are reported as not recorded, never guessed.
    """
    if downloaded:
        return {"stock_field": STOCK_FIELD, "benchmark_field": BENCHMARK_FIELD, "source": "live provider download"}
    policy = (provenance or {}).get("adjustment_policy") or {}
    stock_field, benchmark_field = policy.get("stock_field"), policy.get("benchmark_field")
    if stock_field is None and benchmark_field is None:
        return {"stock_field": None, "benchmark_field": None, "source": "not recorded by the input provenance"}
    if stock_field != STOCK_FIELD or benchmark_field != BENCHMARK_FIELD:
        raise ValueError(
            "Input price-field policy does not match the configured Close/Close price-return "
            f"convention: stock={stock_field!r}, benchmark={benchmark_field!r}"
        )
    return {"stock_field": stock_field, "benchmark_field": benchmark_field, "source": "snapshot adjustment_policy"}


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
    corpus_path: Path | None = DEFAULT_RESULTS_DIR / "headline_corpus.csv",
    earnings_path: Path | None = DEFAULT_RESULTS_DIR / "earnings_events.csv",
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
    price_fields = resolve_price_fields(provenance, downloaded=prices_path is None)
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
    # Pooled reactions (CAAR) by event class, sector, stock and sample, using
    # the same dependency clusters as the strategy bootstrap.
    clusters = _dependency_clusters(events, event_window_audit)
    caar_path, caar_windows = caar_tables(
        event_study, car_results, events, stocks, event_window_audit, clusters,
        DEVELOPMENT_EVENT_IDS, HOLDOUT_EVENT_IDS,
    )
    window_summary, window_agreement = window_choice_tables(significance)
    generalization = pattern_generalization(car_results, events, stocks, DEVELOPMENT_EVENT_IDS, HOLDOUT_EVENT_IDS)
    # Robust group tests (BMP, Kolari-Pynnonen, Corrado rank) for the same
    # groups as the CAAR tables.
    robust_registry = registry_robust_tests(
        returns, events, stocks, clusters,
        _group_definitions(events, stocks, event_window_audit, DEVELOPMENT_EVENT_IDS, HOLDOUT_EVENT_IDS),
        ESTIMATION_SESSIONS, GAP_SESSIONS, EVENT_WINDOW_RADIUS,
    )
    earnings_outputs = {}
    if earnings_path is not None and Path(earnings_path).is_file():
        earnings_calendar = load_earnings_calendar(earnings_path, list(STOCK_TICKERS))
        # Only stocks with a return series in this run can be studied.
        earnings_calendar = earnings_calendar[earnings_calendar["stock"].isin(stocks)].reset_index(drop=True)
        earnings_outputs = run_earnings_study(
            returns, earnings_calendar, events, ESTIMATION_SESSIONS, GAP_SESSIONS, EVENT_WINDOW_RADIUS
        )
    headline_outputs = {}
    if corpus_path is not None and Path(corpus_path).is_file():
        headline_outputs = run_headline_classification(pd.read_csv(corpus_path), events)

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
        "caar_event_time.csv": caar_path,
        "caar_window_summary.csv": caar_windows,
        "window_choice_summary.csv": window_summary,
        "window_choice_agreement.csv": window_agreement,
        "pattern_generalization.csv": generalization,
        "robust_tests_registry.csv": robust_registry,
        **earnings_outputs,
        **headline_outputs,
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
        "return_convention": (
            "Close-to-close price returns for stocks and Nifty price index; dividends are excluded from both sides, so this is not a total-return comparison."
            if price_fields["stock_field"] is not None else
            "Price fields were not recorded by the input provenance; the return convention cannot be confirmed for this run."
        ),
        "price_fields": price_fields,
        "analysis_calendar": "Observed NIFTY 50 sessions with a non-missing close; stock prices are reindexed to these sessions before returns are computed, and missing prices are never filled.",
        "benchmark_sessions_used": int(len(returns.index)),
        "code_commit": _code_commit(),
        "stock_sectors": {stock: STOCK_SECTORS.get(stock) for stock in stocks},
        "headline_corpus": _portable_input_path(Path(corpus_path)) if headline_outputs else None,
        "headline_corpus_sha256": sha256_file(corpus_path) if headline_outputs else None,
        "earnings_calendar": _portable_input_path(Path(earnings_path)) if earnings_outputs else None,
        "earnings_calendar_sha256": sha256_file(earnings_path) if earnings_outputs else None,
        "earnings_design": {
            "unit": "one company on its own results day",
            "day0": "first NIFTY 50 session on or after the announcement date",
            "main_window": "CAR_-1_+1",
            "main_measure": "event-induced variance ratio (mean SCAR squared)",
            "holdout_start": str(EARNINGS_HOLDOUT_START.date()),
        } if earnings_outputs else None,
        "estimation_sessions": ESTIMATION_SESSIONS,
        "gap_sessions": GAP_SESSIONS,
        "event_window_sessions": EVENT_WINDOW_RADIUS,
        "holding_sessions": HOLDING_SESSIONS,
        "transaction_cost_bps_per_side": cost_bps_per_side,
        "development_event_ids": list(DEVELOPMENT_EVENT_IDS),
        "holdout_event_ids": list(HOLDOUT_EVENT_IDS),
        "inference_note": "CAR t-tests and HAC results are approximate and remain exploratory across the full event-stock-window family; overlapping CAR windows are dependent, excluded from category summaries, and joined into dependence clusters for strategy bootstrap intervals. Small cluster counts limit inferential strength.",
        "data_note": "Stock and benchmark returns both use Close price series; dividends are excluded on both sides. This resolves the adjusted-stock-close vs price-index mismatch but does not measure total shareholder return. Local missing prices are not filled. Event clock times remain explicitly unresolved where source evidence does not support a precise timestamp; daily closes cannot identify intraday reaction timing.",
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