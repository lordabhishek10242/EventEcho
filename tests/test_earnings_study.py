import numpy as np
import pandas as pd
import pytest

from src.config import STOCK_TICKERS
from src.earnings_study import (
    first_session_on_or_after, load_earnings_calendar, registry_overlaps, run_earnings_study, window_clusters,
)

SESSIONS = pd.bdate_range("2016-01-04", "2025-12-31")


def _returns(shocks: dict[tuple[str, pd.Timestamp], float], seed=11):
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, len(SESSIONS))
    frame = pd.DataFrame({"NIFTY50_RETURN": market}, index=SESSIONS)
    for stock in ("TCS", "ITC"):
        frame[stock] = 0.0002 + 0.9 * market + rng.normal(0, 0.012, len(SESSIONS))
    for (stock, day), value in shocks.items():
        frame.loc[day, stock] += value
    return frame


def _calendar():
    rows = []
    days = pd.date_range("2016-11-15", "2025-08-15", freq="91D")
    for stock in ("TCS", "ITC"):
        for i, day in enumerate(days):
            rows.append({"earnings_id": f"{stock}_{i:02d}", "stock": stock, "fiscal_quarter": f"Q{i}",
                         "announcement_date": day.date().isoformat(), "source_url": "https://example.org/x"})
    return pd.DataFrame(rows)


def test_day0_is_first_session_on_or_after_announcement():
    saturday = pd.Timestamp("2021-07-24")
    assert first_session_on_or_after(SESSIONS, saturday) == pd.Timestamp("2021-07-26")
    assert first_session_on_or_after(SESSIONS, pd.Timestamp("2021-07-23")) == pd.Timestamp("2021-07-23")
    assert first_session_on_or_after(SESSIONS, pd.Timestamp("2030-01-01")) is None


def test_window_clusters_join_overlapping_windows_only():
    clusters = window_clusters({"a": 100, "b": 108, "c": 125, "d": 300}, radius=5)
    assert clusters["a"] == clusters["b"]
    assert clusters["c"] != clusters["b"] and clusters["d"] != clusters["c"]


def test_registry_overlap_flags_nearby_macro_events():
    registry = pd.DataFrame({"event_id": ["EVT_X"], "effective_event_date": [SESSIONS[200]]})
    flags = registry_overlaps({"near": SESSIONS[208], "far": SESSIONS[230]}, SESSIONS, registry, radius=5)
    assert flags == {"near": "EVT_X", "far": ""}


def test_calendar_validation_rejects_duplicates_and_missing_sources():
    calendar = _calendar()
    with pytest.raises(ValueError, match="one announcement per fiscal quarter"):
        load_earnings_calendar(pd.concat([calendar, calendar.iloc[[0]].assign(earnings_id="dup")]), STOCK_TICKERS)
    broken = calendar.copy()
    broken.loc[0, "source_url"] = " "
    with pytest.raises(ValueError, match="needs a source"):
        load_earnings_calendar(broken, STOCK_TICKERS)
    with pytest.raises(ValueError, match="unknown stocks"):
        renamed = calendar.copy()
        renamed.loc[0, "stock"] = "NOPE"
        load_earnings_calendar(renamed, STOCK_TICKERS)


def test_earnings_study_finds_injected_volatility_but_no_signed_effect():
    calendar = load_earnings_calendar(_calendar(), STOCK_TICKERS)
    rng = np.random.default_rng(5)
    shocks = {}
    for _, row in calendar[calendar["stock"] == "TCS"].iterrows():
        day0 = first_session_on_or_after(SESSIONS, row["announcement_date"])
        shocks[("TCS", day0)] = rng.choice([-0.06, 0.06])
    returns = _returns(shocks)
    registry = pd.DataFrame({"event_id": ["EVT_A"], "effective_event_date": [pd.Timestamp("2020-03-24")]})
    out = run_earnings_study(returns, calendar, registry, 100, 20, 5)
    summary = out["earnings_summary.csv"].set_index(["group", "window"])
    tcs = summary.loc[("TCS", "CAR_-1_+1")]
    itc = summary.loc[("ITC", "CAR_-1_+1")]
    assert tcs["variance_ratio_ci_lower"] > 2
    assert itc["variance_ratio_ci_lower"] < 1.5 < tcs["variance_ratio_ci_lower"]
    assert abs(tcs["bmp_t"]) < 3  # random signs: no average direction
    events = out["earnings_event_car.csv"]
    assert set(events["sample"]) == {"development", "holdout"}
    assert (events["announcement_date"] < "2023-01-01").eq(events["sample"] == "development").all()
    path = out["earnings_event_time.csv"]
    tcs_path = path[path["group"] == "TCS"].set_index("event_day")
    assert tcs_path.loc[0, "mean_abs_AR_pct"] > 3 * tcs_path.loc[-3, "mean_abs_AR_pct"]
    replication = out["earnings_replication.csv"].set_index("stock")
    assert bool(replication.loc["TCS", "replicates"]) and not bool(replication.loc["ITC", "replicates"])
    placebo = out["earnings_placebo.csv"].set_index(["group", "window"])
    # Placebo days carry no injected shocks, so TCS looks ordinary there.
    assert placebo.loc[("TCS", "CAR_-1_+1"), "variance_ratio_ci_upper"] < tcs["variance_ratio_ci_lower"]


def test_announcements_without_enough_history_are_listed_not_dropped_silently():
    calendar = _calendar()
    early = pd.DataFrame([{"earnings_id": "EARLY", "stock": "TCS", "fiscal_quarter": "QX",
                           "announcement_date": "2016-02-01", "source_url": "https://example.org/y"}])
    calendar = load_earnings_calendar(pd.concat([calendar, early]), STOCK_TICKERS)
    out = run_earnings_study(_returns({}), calendar, pd.DataFrame(columns=["event_id", "effective_event_date"]), 100, 20, 5)
    coverage = out["earnings_coverage.csv"].set_index("earnings_id")
    assert not coverage.loc["EARLY", "included"]
    assert "history" in coverage.loc["EARLY", "reason"]
    assert "EARLY" not in set(out["earnings_event_car.csv"]["earnings_id"])


def test_shipped_earnings_calendar_is_complete_and_sourced():
    calendar = load_earnings_calendar("results/earnings_events.csv", STOCK_TICKERS)
    assert len(calendar) == 266
    assert calendar.groupby("stock").size().eq(38).all()
    gaps = calendar.sort_values("announcement_date").groupby("stock")["announcement_date"].diff().dt.days.dropna()
    assert gaps.between(25, 150).all()  # catches typos such as a wrong month or year
    assert calendar["announcement_date"].between("2016-07-01", "2025-11-30").all()
    raw = pd.read_csv("results/earnings_events.csv")
    assert raw["source_url"].str.startswith("https://").all()
    assert (raw["cross_check_kind"] == "random_sample").sum() == 12
