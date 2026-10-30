"""Null simulation for the drift test: no drift is injected, so about 5% of runs
should show an interval that excludes zero.

Usage: python3 scripts/simulate_drift_null.py FIRST_SEED LAST_SEED (e.g. 0 20).
Simulated prices only; the shipped results calendar supplies the dates.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from src.config import STOCK_TICKERS
from src.earnings_study import load_earnings_calendar, build_event_units, first_session_on_or_after, HOLDOUT_START
from src.earnings_extensions import drift_events, drift_summary
import src.earnings_extensions as ext
ext.DRAWS = 1000
cal = load_earnings_calendar(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "results" / "earnings_events.csv"), list(STOCK_TICKERS))
S = pd.bdate_range("2016-01-04", "2025-12-31")
stocks = sorted(cal.stock.unique())
hits = []; vals = []
for seed in range(int(sys.argv[1]), int(sys.argv[2])):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, .01, len(S))
    R = pd.DataFrame({"NIFTY50_RETURN": m}, index=S); R.index.name = "date"
    for s in stocks: R[s] = 0.9*m + rng.normal(0, .015, len(S))
    for _, row in cal.iterrows():
        d = first_session_on_or_after(S, row.announcement_date); R.loc[d, row.stock] += rng.choice([-.04, .04])
    meta = cal.copy(); meta["event_date"] = [first_session_on_or_after(S, d) for d in meta.announcement_date]
    meta["unit_id"] = meta.earnings_id
    meta["sample"] = np.where(meta.announcement_date >= HOLDOUT_START, "holdout", "development")
    meta["overlapping_registry_events"] = ""
    units, _, skipped = build_event_units(R, meta[["unit_id", "stock", "event_date"]], 100, 20, 5)
    um = {u.unit_id: u for u in units}; meta = meta[meta.earnings_id.isin(um)]
    ev = drift_events(R, meta, um, 100, 20, 5)
    sm = drift_summary(ev); row = sm[(sm["sample"] == "All announcements") & sm.primary].iloc[0]
    hits.append(bool(row.interval_excludes_zero)); vals.append(row.value)
    print(seed, round(row.value, 2), round(row.ci95_lower, 2), round(row.ci95_upper, 2), row.clusters, flush=True)
print("rejection rate", np.mean(hits), "mean value", np.mean(vals))
