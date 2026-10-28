#!/usr/bin/env python3
"""Opt-in acquisition of a frozen Yahoo Finance historical price snapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, END_DATE, START_DATE  # noqa: E402
from src.snapshots import (  # noqa: E402
    BENCHMARK_FIELD,
    FREQUENCY,
    STOCK_FIELD,
    SnapshotAcquisitionError,
    create_market_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=START_DATE, help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end-exclusive", default=END_DATE, help="Exclusive provider end date (YYYY-MM-DD)")
    parser.add_argument(
        "--snapshot-id",
        default=None,
        help="New snapshot directory name; defaults to a UTC retrieval timestamp",
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_RESULTS_DIR / "event_master.csv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATA_DIR / "snapshots")
    parser.add_argument("--source", choices=["Yahoo Finance"], default="Yahoo Finance")
    parser.add_argument("--frequency", choices=[FREQUENCY], default=FREQUENCY)
    parser.add_argument(
        "--stock-adjustment",
        choices=["adjusted-close"],
        default="adjusted-close",
        help=f"Explicit stock price field policy; currently fixed to {STOCK_FIELD}",
    )
    parser.add_argument(
        "--benchmark-adjustment",
        choices=["price-index-close"],
        default="price-index-close",
        help=f"Explicit NIFTY 50 field policy; currently fixed to {BENCHMARK_FIELD}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot_id = args.snapshot_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    events = pd.read_csv(args.events, parse_dates=["event_date", "effective_event_date"])
    try:
        manifest = create_market_snapshot(
            snapshot_id=snapshot_id,
            start_date=args.start,
            end_date_exclusive=args.end_exclusive,
            events=events,
            root=args.output_root,
            frequency=args.frequency,
            source=args.source,
            stock_adjustment=args.stock_adjustment,
            benchmark_adjustment=args.benchmark_adjustment,
        )
    except SnapshotAcquisitionError as exc:
        print(f"Snapshot incomplete: {exc.snapshot_path}", file=sys.stderr)
        for error in exc.report.get("fatal_errors", []):
            print(f"FATAL: {error}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Snapshot acquisition failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Snapshot created: {args.output_root / snapshot_id}")
    print(f"Status: {manifest['status']}; source: {manifest['source']}; interval: {manifest['frequency']}")
    print(f"Date range: {manifest['requested_start_inclusive']} to {manifest['requested_end_exclusive']} (exclusive)")
    warning_count = len(manifest["diagnostics"].get("warnings", []))
    print(f"Warnings: {warning_count}; ticker failures: {len(manifest['provider_failures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
