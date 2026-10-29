#!/usr/bin/env python3
"""Acquire a frozen market snapshot, run the complete analysis, and build the demo."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import END_DATE, START_DATE, DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR
from src.snapshots import create_market_snapshot
from src.reporting import build_demo_dashboard
from scripts.run_analysis import run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-id",
        default=None,
        help="Unique snapshot identifier. Defaults to a UTC timestamped identifier.",
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_RESULTS_DIR / "event_master.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR / "reproduced")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE, help="Exclusive end date.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_id = args.snapshot_id or (
        "eventecho-historical-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    events = pd.read_csv(args.events)

    snapshot = create_market_snapshot(
        snapshot_id=snapshot_id,
        start_date=args.start,
        end_date_exclusive=args.end,
        events=events,
        root=DEFAULT_DATA_DIR / "snapshots",
    )
    if snapshot.get("status") != "complete":
        raise RuntimeError(f"Snapshot {snapshot_id} is not complete; refusing analysis.")

    run_analysis(
        events_path=args.events,
        output_dir=args.output_dir,
        offline=True,
        start_date=args.start,
        end_date=args.end,
        snapshot_id=snapshot_id,
    )
    dashboard = build_demo_dashboard(
        args.output_dir,
        output_path=args.output_dir / "eventecho_demo.html",
    )
    print(f"Snapshot: {DEFAULT_DATA_DIR / 'snapshots' / snapshot_id}")
    print(f"Analysis: {args.output_dir}")
    print(f"Dashboard: {dashboard}")


if __name__ == "__main__":
    main()
