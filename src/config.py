"""Deterministic project configuration and paths."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "reproduced"

STOCK_TICKERS = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "HDFC_BANK": "HDFCBANK.NS",
    "ITC": "ITC.NS",
}
BENCHMARK_TICKER = "^NSEI"
START_DATE = "2016-01-01"
END_DATE = "2026-01-01"  # yfinance end dates are exclusive.
ESTIMATION_SESSIONS = 100
GAP_SESSIONS = 20
EVENT_WINDOW_RADIUS = 5
CAR_WINDOWS = {"CAR_-1_+1": (-1, 1), "CAR_-3_+3": (-3, 3), "CAR_-5_+5": (-5, 5)}
COST_LEVELS_BPS = (0, 5, 10, 20, 30)
BASE_COST_BPS_PER_SIDE = 10
HOLDING_SESSIONS = 3

# Retained exactly from the supplied artifact manifest; do not infer split membership by row order.
DEVELOPMENT_EVENT_IDS = ("EVT001", "EVT002", "EVT003", "EVT004", "EVT005", "EVT007")
HOLDOUT_EVENT_IDS = ("EVT008", "EVT009", "EVT006")


@dataclass(frozen=True)
class AnalysisConfig:
    start_date: str = START_DATE
    end_date: str = END_DATE
    estimation_sessions: int = ESTIMATION_SESSIONS
    gap_sessions: int = GAP_SESSIONS
    event_window_radius: int = EVENT_WINDOW_RADIUS
    holding_sessions: int = HOLDING_SESSIONS
    cost_bps_per_side: float = BASE_COST_BPS_PER_SIDE

    def __post_init__(self) -> None:
        if self.estimation_sessions < 3:
            raise ValueError("estimation_sessions must be at least 3")
        if self.gap_sessions < self.event_window_radius:
            raise ValueError("gap_sessions must be at least event_window_radius")
        if self.event_window_radius < 1 or self.holding_sessions < 1:
            raise ValueError("event window radius and holding period must be positive")
        if self.cost_bps_per_side < 0:
            raise ValueError("cost_bps_per_side cannot be negative")