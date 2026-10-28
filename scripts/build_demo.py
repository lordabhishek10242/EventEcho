#!/usr/bin/env python3
"""Build the offline interactive demo for one verified historical analysis run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting import build_demo_dashboard  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = build_demo_dashboard(args.analysis_dir, args.output)
    except Exception as exc:
        parser.error(str(exc))
    print(f"Demo dashboard: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
