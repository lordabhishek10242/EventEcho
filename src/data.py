"""Market-data acquisition, validation, and return construction."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_price_provenance(path: str | Path) -> dict:
    """Load an adjacent provenance sidecar and verify its checksum.

    A missing sidecar is explicitly labelled unverified; it is never silently
    promoted to historical-market evidence.
    """
    price_path = Path(path)
    sidecar = price_path.with_name(f"{price_path.stem}.provenance.json")
    if not sidecar.is_file():
        return {
            "classification": "unverified_local_input",
            "provenance_verified": False,
            "source_independently_verified": False,
            "sha256": sha256_file(price_path) if price_path.is_file() else None,
        }
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    expected = payload.get("sha256")
    actual = sha256_file(price_path)
    if not expected or expected != actual:
        raise ValueError("Price provenance checksum does not match the price file")
    return {
        **payload,
        "sha256": actual,
        "provenance_verified": True,
        "source_independently_verified": bool(payload.get("source_independently_verified", False)),
    }


def _extract_price_column(data: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    if data.empty:
        raise ValueError(f"No observations returned for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        if field not in data.columns.get_level_values(0):
            raise ValueError(f"{field!r} is unavailable for {ticker}")
        values = data[field]
        series = values.iloc[:, 0] if isinstance(values, pd.DataFrame) else values
    else:
        if field not in data.columns:
            raise ValueError(f"{field!r} is unavailable for {ticker}")
        series = data[field]
    series = pd.to_numeric(series, errors="coerce").rename(ticker)
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    if series.index.has_duplicates:
        raise ValueError(f"Duplicate normalized dates returned for {ticker}")
    return series.sort_index()


def download_prices(
    tickers: Mapping[str, str],
    start: str,
    end: str,
    benchmark_ticker: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Download stock close prices and the benchmark price close on a price-return basis.

    yfinance is imported lazily so synthetic/unit-test workflows have no network
    or data-provider dependency.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance or provide local price CSVs") from exc

    stocks: dict[str, pd.Series] = {}
    failures: list[str] = []
    for label, ticker in tickers.items():
        try:
            raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
            stocks[label] = _extract_price_column(raw, "Close", ticker).rename(label)
        except Exception as exc:  # Provider failures vary by ticker and network state.
            LOGGER.warning("Could not download %s (%s): %s", label, ticker, exc)
            failures.append(label)
    if not stocks:
        raise RuntimeError("No stock price series could be downloaded")
    if failures:
        LOGGER.warning("Omitting unavailable stock series: %s", ", ".join(failures))

    try:
        benchmark_raw = yf.download(
            benchmark_ticker, start=start, end=end, auto_adjust=False, progress=False
        )
        benchmark = _extract_price_column(benchmark_raw, "Close", benchmark_ticker)
    except Exception as exc:
        raise RuntimeError(f"Could not download benchmark {benchmark_ticker}: {exc}") from exc
    return pd.concat(stocks.values(), axis=1).sort_index(), benchmark.rename("NIFTY50")


def load_price_csv(path: str, date_column: str = "date") -> pd.DataFrame:
    """Load a local wide price CSV indexed by date; columns are instrument labels."""
    frame = pd.read_csv(path, parse_dates=[date_column]).set_index(date_column)
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    frame = frame.apply(pd.to_numeric, errors="coerce").sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Price CSV contains duplicate dates")
    if frame.empty or frame.dropna(how="all").empty:
        raise ValueError("Price CSV contains no usable prices")
    return frame


def calculate_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Compute simple close-to-close returns without filling missing prices."""
    ordered = prices.sort_index().copy()
    ordered.index = pd.DatetimeIndex(ordered.index).tz_localize(None).normalize()
    if ordered.index.has_duplicates:
        raise ValueError("Price observations contain duplicate dates")
    values = ordered.to_numpy(dtype=float, na_value=np.nan)
    observed = ~np.isnan(values)
    if np.isinf(values[observed]).any() or (values[observed] <= 0).any():
        raise ValueError("Observed prices must be finite and strictly positive")
    return ordered.pct_change(fill_method=None)


def align_stock_benchmark_returns(
    stock_returns: pd.DataFrame, benchmark_returns: pd.Series
) -> pd.DataFrame:
    """Align already-computed returns by date.

    This function cannot repair returns computed across different calendars.
    Prefer calculate_aligned_returns when starting from price levels.
    """
    benchmark = benchmark_returns.rename("NIFTY50_RETURN")
    aligned = stock_returns.join(benchmark, how="inner").sort_index()
    return aligned


def align_stock_prices_to_benchmark_calendar(
    stock_prices: pd.DataFrame, benchmark_prices: pd.Series
) -> pd.DataFrame:
    """Reindex stock price levels to observed benchmark sessions without filling gaps."""
    stocks = stock_prices.copy().sort_index()
    benchmark = benchmark_prices.copy().sort_index()
    stocks.index = pd.DatetimeIndex(stocks.index).tz_localize(None).normalize()
    benchmark.index = pd.DatetimeIndex(benchmark.index).tz_localize(None).normalize()
    if stocks.index.has_duplicates or benchmark.index.has_duplicates:
        raise ValueError("Price observations contain duplicate normalized dates")
    if stocks.empty or benchmark.empty:
        raise ValueError("Stock and benchmark prices must both contain observations")
    return stocks.reindex(benchmark.index)


def calculate_aligned_returns(
    stock_prices: pd.DataFrame, benchmark_prices: pd.Series
) -> pd.DataFrame:
    """Align price levels to benchmark sessions before computing returns.

    Returns are close-to-close across the same benchmark-session calendar.
    Missing stock prices remain missing: no forward fill is performed, and a
    stock return is unavailable until two consecutive aligned sessions have
    observed prices. This prevents a stock return spanning a holiday/missing
    benchmark session from being paired with a one-session benchmark return.
    """
    stocks = align_stock_prices_to_benchmark_calendar(stock_prices, benchmark_prices)
    benchmark = benchmark_prices.copy().sort_index()
    benchmark.index = pd.DatetimeIndex(benchmark.index).tz_localize(None).normalize()
    stock_returns = calculate_returns(stocks)
    benchmark_returns = calculate_returns(benchmark).rename("NIFTY50_RETURN")
    return align_stock_benchmark_returns(stock_returns, benchmark_returns)
