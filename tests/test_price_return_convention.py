import sys
import types

import pandas as pd

from src.data import download_prices


def test_download_prices_uses_close_for_stocks_and_benchmark(monkeypatch):
    calls = []

    def fake_download(ticker, **kwargs):
        calls.append((ticker, kwargs))
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        # Adj Close deliberately differs so the test catches a convention regression.
        return pd.DataFrame(
            {"Close": [100.0, 101.0], "Adj Close": [90.0, 91.0]},
            index=dates,
        )

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    stocks, benchmark = download_prices({"A": "A.NS"}, "2024-01-01", "2024-01-04", "^NSEI")

    assert stocks["A"].tolist() == [100.0, 101.0]
    assert benchmark.tolist() == [100.0, 101.0]
    assert [call[0] for call in calls] == ["A.NS", "^NSEI"]
