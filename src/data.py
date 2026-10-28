"""Market-data acquisition, validation, and return construction."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


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
    """Download adjusted stock closes and the benchmark price close.

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
            stocks[label] = _extract_price_column(raw, "Adj Close", ticker).rename(label)
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
    """Align by date without dropping a date globally for one stock's missing data."""
    benchmark = benchmark_returns.rename("NIFTY50_RETURN")
    aligned = stock_returns.join(benchmark, how="inner").sort_index()
    return aligned