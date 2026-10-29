import sys

from scripts.acquire_market_snapshot import parse_args


def test_snapshot_cli_defaults_to_consistent_close_price_policy(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["acquire_market_snapshot.py"])
    args = parse_args()

    assert args.stock_adjustment == "close-price"
    assert args.benchmark_adjustment == "price-index-close"
