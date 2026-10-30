import numpy as np
import pandas as pd

from src.config import STOCK_TICKERS
from src.earnings_extensions import _corr_moments, _diff, _mean, _slope_moments, cluster_bootstrap
from src.earnings_study import first_session_on_or_after, load_earnings_calendar, run_earnings_study

SESSIONS = pd.bdate_range("2016-01-04", "2025-12-31")
NO_REGISTRY = pd.DataFrame(columns=["event_id", "effective_event_date"])


def _calendar(offsets=None):
    offsets = offsets or {"TCS": 0, "INFOSYS": 30}
    rows = []
    for stock, offset in offsets.items():
        days = pd.date_range(pd.Timestamp("2016-11-15") + pd.Timedelta(days=offset), "2025-08-15", freq="91D")
        for i, day in enumerate(days):
            rows.append({"earnings_id": f"{stock}_{i:02d}", "stock": stock, "fiscal_quarter": f"Q{i}",
                         "announcement_date": day.date().isoformat(), "source_url": "https://example.org/x"})
    return load_earnings_calendar(pd.DataFrame(rows), STOCK_TICKERS)


def _returns(calendar, shock_fn, follow=0.0, peer_share=0.0, seed=3):
    """Market-model returns plus results-day shocks.

    ``follow``: extra daily return on Days +2..+20 in the direction of the shock.
    ``peer_share``: fraction of a TCS shock that also hits INFOSYS that day.
    """
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, len(SESSIONS))
    frame = pd.DataFrame({"NIFTY50_RETURN": market}, index=SESSIONS)
    frame.index.name = "date"
    for stock in ("TCS", "INFOSYS"):
        frame[stock] = 0.0002 + 0.9 * market + rng.normal(0, 0.012, len(SESSIONS))
    for _, row in calendar.iterrows():
        day0 = first_session_on_or_after(SESSIONS, row["announcement_date"])
        pos = SESSIONS.get_loc(day0)
        shock = shock_fn(row, day0, rng)
        frame.iloc[pos, frame.columns.get_loc(row["stock"])] += shock
        if follow:
            frame.iloc[pos + 2: pos + 21, frame.columns.get_loc(row["stock"])] += np.sign(shock) * follow
        if peer_share and row["stock"] == "TCS":
            frame.iloc[pos, frame.columns.get_loc("INFOSYS")] += peer_share * shock
    return frame


def _random_sign(row, day0, rng):
    return rng.choice([-0.06, 0.06])


# --------------------------------------------------------------------------- bootstrap
def test_bootstrap_point_values_match_direct_formulas():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=60), rng.normal(size=60)
    clusters = np.arange(60) // 3
    up = x > 0
    assert np.isclose(cluster_bootstrap(clusters, _mean(y), 1)[0], y.mean())
    assert np.isclose(cluster_bootstrap(clusters, _diff(y, up, ~up), 1)[0], y[up].mean() - y[~up].mean())
    assert np.isclose(cluster_bootstrap(clusters, _slope_moments(x, y), 1)[0], np.polyfit(x, y, 1)[0])
    assert np.isclose(cluster_bootstrap(clusters, _corr_moments(x, y), 1)[0], np.corrcoef(x, y)[0, 1])
    value, lo, hi, k = cluster_bootstrap(clusters, _mean(y), 1)
    assert k == 20 and lo < value < hi


def test_bootstrap_resamples_whole_clusters():
    # Two clusters of identical values: any resample mean is one of three values.
    y = np.array([0.0, 0.0, 1.0, 1.0])
    _, lo, hi, k = cluster_bootstrap(np.array([0, 0, 1, 1]), _mean(y), 1)
    assert k == 2 and np.isnan(lo)  # fewer than 3 clusters: no interval
    y = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    _, lo, hi, _ = cluster_bootstrap(np.array([0, 0, 1, 1, 2, 2]), _mean(y), 1, draws=500)
    assert lo >= 0 and hi <= 2


# --------------------------------------------------------------------------- 1. drift
def test_drift_detects_injected_continuation_and_not_without_it():
    calendar = _calendar()
    with_drift = run_earnings_study(_returns(calendar, _random_sign, follow=0.004), calendar, NO_REGISTRY, 100, 20, 5)
    no_drift = run_earnings_study(_returns(calendar, _random_sign), calendar, NO_REGISTRY, 100, 20, 5)

    def primary(out):
        s = out["earnings_drift_summary.csv"]
        return s[(s["sample"] == "All announcements") & s["primary"]].iloc[0]

    hit = primary(with_drift)
    assert hit["value"] > 10 and hit["ci95_lower"] > 0  # 19 days x 0.4% each way ≈ 15 pp
    slope = with_drift["earnings_drift_summary.csv"].set_index(["sample", "statistic"]).loc[("All announcements", "slope_after_on_results_move")]
    assert slope["value"] > 0
    null = primary(no_drift)
    assert null["ci95_lower"] < 0 < null["ci95_upper"]


def test_drift_excludes_windows_with_next_results_or_past_the_data():
    calendar = _calendar()
    extra = calendar[calendar["stock"] == "TCS"].iloc[[3]].copy()
    extra["earnings_id"] = "TCS_CLOSE"
    extra["fiscal_quarter"] = "QX"
    extra["announcement_date"] = extra["announcement_date"] + pd.Timedelta(days=14)
    calendar = load_earnings_calendar(pd.concat([calendar, extra]), STOCK_TICKERS)
    short = _returns(calendar, _random_sign).loc[:"2025-08-25"]
    out = run_earnings_study(short, calendar, NO_REGISTRY, 100, 20, 5)
    events = out["earnings_drift_events.csv"].set_index("earnings_id")
    assert "same stock" in events.loc["TCS_03", "excluded"]
    assert "past the price history" in events.loc["TCS_35", "excluded"]
    kept = events[events["excluded"] == ""]
    assert kept["CAR_+2_+20_pct"].notna().all()


# --------------------------------------------------------------------------- 2. peers
def test_peer_spillover_finds_injected_co_movement():
    calendar = _calendar()
    out = run_earnings_study(_returns(calendar, _random_sign, peer_share=0.5), calendar, NO_REGISTRY, 100, 20, 5)
    summary = out["earnings_peer_spillover.csv"].set_index(["pair", "sample"])
    tcs_to_inf = summary.loc[("TCS → INFOSYS", "all days")]
    assert tcs_to_inf["correlation_ci95_lower"] > 0.5
    assert tcs_to_inf["variance_ratio_ci95_lower"] > tcs_to_inf["peer_placebo_variance_ratio"]
    reverse = summary.loc[("INFOSYS → TCS", "all days")]
    assert reverse["correlation_ci95_lower"] < 0.3  # INFOSYS shocks were not shared
    events = out["earnings_peer_events.csv"]
    assert events.loc[events["pair"] == "TCS → INFOSYS", "clean"].all()  # own results 30 days apart


# --------------------------------------------------------------------------- 3. trend
def test_volatility_trend_separates_growing_results_shocks_from_placebo():
    calendar = _calendar()

    def growing(row, day0, rng):
        return rng.choice([-1, 1]) * 0.012 * (day0.year - 2015)

    out = run_earnings_study(_returns(calendar, growing), calendar, NO_REGISTRY, 100, 20, 5)
    trend = out["earnings_volatility_trend.csv"].set_index(["row_type", "label"])
    assert trend.loc[("slope", "slope per year: results days"), "ci95_lower"] > 0
    placebo = trend.loc[("slope", "slope per year: placebo days")]
    # Placebo days get no shocks, so their slope stays small. It can be slightly
    # negative: growing results shocks sit inside later placebo estimation
    # windows and inflate the standard error there (documented limit).
    assert abs(placebo["variance_ratio"]) < 0.25 * trend.loc[("slope", "slope per year: results days"), "variance_ratio"]
    assert trend.loc[("slope", "difference: results minus placebo slope"), "ci95_lower"] > 0
    years = trend.loc["year"]
    assert years["results_days"].sum() == len(out["earnings_event_car.csv"]["earnings_id"].unique())
