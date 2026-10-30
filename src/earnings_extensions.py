"""Three follow-up questions on company results days.

All three designs were written down before any result was computed
(2026-10-30) and are reported whatever they show.

1. **Drift or reversal after the results move.** Each results day is
   labelled "up" or "down" by the sign of its CAR(-1,+1). We then measure the
   CAR over Days +2 to +20 with the same market model. Primary statistic: the
   difference in mean CAR(+2,+20) between "up" and "down" announcements.
   Secondary: the slope of CAR(+2,+20) on CAR(-1,+1). A positive difference or
   slope means the move continues (drift); a negative one means it reverses.
   Announcements whose next results day of the same stock falls inside the
   Day +2..+20 window, or whose window runs past the price history, are dropped
   and listed.
2. **Peer spillover.** For the two same-industry pairs (TCS/Infosys and
   Adani Enterprises/Adani Ports), on company A's results day we measure the
   peer's abnormal return. Statistics: the peer's variance ratio on A's day
   (compared with the peer's own placebo level) and the correlation between
   A's and the peer's CAR(-1,+1). Because peers often report within days of
   each other, a "clean" sample drops days where the peer's own results window
   overlaps A's Day -1..+1.
3. **Trend over time.** The variance ratio by calendar year, and the slope of
   SCAR(-1,+1)^2 on the year, for results days and for placebo days. If
   markets simply became more volatile, placebo days would trend too, so the
   difference in slopes is the quantity of interest.

Uncertainty: cluster bootstrap (5,000 draws, seed 2026), resampling whole
clusters of announcements whose windows overlap in calendar time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .event_study import estimation_sample, fit_market_model
from .robust_tests import EventUnit
from .statistics import car_significance

DRIFT_START, DRIFT_END = 2, 20
DRAWS = 5000
SEED = 2026
PEER_PAIRS = (("TCS", "INFOSYS"), ("INFOSYS", "TCS"), ("ADANI_ENTERPRISES", "ADANI_PORTS"), ("ADANI_PORTS", "ADANI_ENTERPRISES"))
MAIN = "CAR_-1_+1"


def _weights(k: int, seed: int, draws: int = DRAWS) -> np.ndarray:
    """(draws x k) matrix: how often each cluster is picked in each bootstrap draw."""
    picks = np.random.default_rng(seed).integers(0, k, size=(draws, k))
    weights = np.zeros((draws, k))
    np.add.at(weights, (np.repeat(np.arange(draws), k), picks.ravel()), 1.0)
    return weights


def _mean(y, mask=None):
    m = np.ones_like(y) if mask is None else mask.astype(float)
    return {"s": y * m, "n": m}, lambda t: t["s"] / t["n"]


def _diff(y, a, b):
    return ({"sa": y * a, "na": a.astype(float), "sb": y * b, "nb": b.astype(float)},
            lambda t: t["sa"] / t["na"] - t["sb"] / t["nb"])


def _slope_moments(x, y):
    one = np.ones_like(x)
    return ({"n": one, "x": x, "y": y, "xy": x * y, "xx": x * x},
            lambda t: (t["n"] * t["xy"] - t["x"] * t["y"]) / (t["n"] * t["xx"] - t["x"] ** 2))


def _corr_moments(x, y):
    one = np.ones_like(x)
    def fn(t):
        cov = t["n"] * t["xy"] - t["x"] * t["y"]
        return cov / np.sqrt((t["n"] * t["xx"] - t["x"] ** 2) * (t["n"] * t["yy"] - t["y"] ** 2))
    return {"n": one, "x": x, "y": y, "xy": x * y, "xx": x * x, "yy": y * y}, fn


def cluster_bootstrap(clusters, spec, seed: int, draws: int = DRAWS) -> tuple[float, float, float, int]:
    """Point value and 95% percentile interval from resampling whole clusters.

    ``spec`` is (row-level sums, function of totals). Each draw re-weights the
    per-cluster sums by how often the cluster was picked, so the statistic is
    recomputed exactly without copying data.
    """
    sums, fn = spec
    codes, labels = pd.factorize(pd.Series(clusters).astype(str))
    k = len(labels)
    per_cluster = {name: np.bincount(codes, weights=np.asarray(v, float), minlength=k) for name, v in sums.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        point = float(fn({name: v.sum() for name, v in per_cluster.items()})) if k else np.nan
        if k < 3:
            return point, np.nan, np.nan, k
        weights = _weights(k, seed, draws)
        values = fn({name: weights @ v for name, v in per_cluster.items()})
    values = values[np.isfinite(values)]
    if len(values) < draws * 0.5:
        return point, np.nan, np.nan, k
    return point, float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)), k


def _interval_clusters(starts: Mapping[str, int], ends: Mapping[str, int]) -> dict[str, str]:
    """Join items whose [start, end] session intervals overlap."""
    ordered = sorted(starts, key=lambda k: (starts[k], k))
    out, label, last_end = {}, 0, None
    for key in ordered:
        if last_end is None or starts[key] > last_end:
            label += 1
            last_end = ends[key]
        else:
            last_end = max(last_end, ends[key])
        out[key] = f"XC_{label:03d}"
    return out


# --------------------------------------------------------------------------- 1. drift
def drift_events(returns: pd.DataFrame, meta: pd.DataFrame, unit_map: Mapping[str, EventUnit],
                 estimation_sessions: int, gap_sessions: int, radius: int,
                 market_column: str = "NIFTY50_RETURN") -> pd.DataFrame:
    sessions = pd.DatetimeIndex(returns.index).tz_localize(None).normalize()
    positions: dict[str, list[int]] = {}
    for stock, day in zip(meta["stock"], meta["event_date"]):
        positions.setdefault(stock, []).append(sessions.get_loc(day))
    rows = []
    for _, row in meta.iterrows():
        eid, stock, day = row["earnings_id"], row["stock"], row["event_date"]
        pos = sessions.get_loc(day)
        record = {"earnings_id": eid, "stock": stock, "sample": row["sample"],
                  "announcement_date": pd.Timestamp(row["announcement_date"]).date().isoformat(),
                  "overlapping_registry_events": row["overlapping_registry_events"],
                  "CAR_-1_+1_pct": unit_map[eid].cars[MAIN] * 100, "position": pos}
        if pos + DRIFT_END >= len(sessions):
            record["excluded"] = "drift window runs past the price history"
        elif any(pos + DRIFT_START - 1 <= other <= pos + DRIFT_END + 1 for other in positions[stock] if other != pos):
            record["excluded"] = "next results day of the same stock inside Day +2..+20"
        else:
            sample = estimation_sample(returns, day, estimation_sessions, gap_sessions, stock, market_column, radius)
            model = fit_market_model(sample, stock, market_column)
            window = returns.iloc[pos + DRIFT_START: pos + DRIFT_END + 1][[stock, market_column]]
            if window.isna().any().any():
                record["excluded"] = "missing price in Day +2..+20"
            else:
                ar = window[stock] - (model["alpha"] + model["beta"] * window[market_column])
                car = float(ar.sum())
                se = car_significance(car, list(range(DRIFT_START, DRIFT_END + 1)), window[market_column].to_numpy(float), model)["standard_error"]
                record.update({"CAR_+2_+20_pct": car * 100, "CAR_+2_+20_se_pct": se * 100, "excluded": ""})
        record["direction"] = "up" if record["CAR_-1_+1_pct"] > 0 else "down"
        rows.append(record)
    out = pd.DataFrame(rows)
    kept = out[out["excluded"] == ""]
    clusters = _interval_clusters(dict(zip(kept["earnings_id"], kept["position"] - 1)),
                                  dict(zip(kept["earnings_id"], kept["position"] + DRIFT_END)))
    out["drift_cluster"] = out["earnings_id"].map(clusters).fillna("")
    return out.drop(columns=["position"])


def drift_summary(events: pd.DataFrame) -> pd.DataFrame:
    kept = events[events["excluded"] == ""].copy()
    samples = [("All announcements", kept), ("Development (to 2022)", kept[kept["sample"] == "development"]),
               ("Holdout (2023 onward)", kept[kept["sample"] == "holdout"]),
               ("Without registry-event overlap", kept[kept["overlapping_registry_events"].fillna("") == ""])]
    rows = []
    for si, (name, f) in enumerate(samples):
        y = f["CAR_+2_+20_pct"].to_numpy(float)
        x = f["CAR_-1_+1_pct"].to_numpy(float)
        up = (f["direction"] == "up").to_numpy()
        specs = {
            "difference_up_minus_down_pp": _diff(y, up, ~up),
            "mean_after_up_pct": _mean(y, up),
            "mean_after_down_pct": _mean(y, ~up),
            "slope_after_on_results_move": _slope_moments(x, y),
            "mean_after_all_pct": _mean(y),
        }
        for ki, (stat_name, spec) in enumerate(specs.items()):
            value, lo, hi, k = cluster_bootstrap(f["drift_cluster"].to_numpy(), spec, SEED + 100 * si + ki)
            rows.append({"sample": name, "statistic": stat_name, "value": value, "ci95_lower": lo, "ci95_upper": hi,
                         "announcements": len(f), "up": int(up.sum()), "down": int((~up).sum()), "clusters": k,
                         "interval_excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                         "primary": stat_name == "difference_up_minus_down_pp"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2. peers
def peer_spillover(returns: pd.DataFrame, meta: pd.DataFrame, unit_map: Mapping[str, EventUnit],
                   placebo_units: Sequence[EventUnit], placebo_issuer: Mapping[str, str],
                   build_units: Callable, estimation_sessions: int, gap_sessions: int, radius: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = pd.DatetimeIndex(returns.index).tz_localize(None).normalize()
    own_positions = {s: [sessions.get_loc(d) for d in meta.loc[meta["stock"] == s, "event_date"]] for s in meta["stock"].unique()}
    event_rows, summary_rows = [], []
    placebo_by_stock: dict[str, list[EventUnit]] = {}
    for u in placebo_units:
        placebo_by_stock.setdefault(u.stock, []).append(u)
    for pi, (issuer, peer) in enumerate(PEER_PAIRS):
        if issuer not in own_positions or peer not in own_positions:
            continue
        rows = meta[meta["stock"] == issuer]
        table = pd.DataFrame({"unit_id": [f"{e}>{peer}" for e in rows["earnings_id"]], "stock": peer, "event_date": rows["event_date"].to_numpy()})
        peer_units, _, _ = build_units(returns, table, estimation_sessions, gap_sessions, radius)
        by_id = {u.unit_id: u for u in peer_units}
        for _, r in rows.iterrows():
            uid = f"{r['earnings_id']}>{peer}"
            if uid not in by_id:
                continue
            pos = sessions.get_loc(r["event_date"])
            gap = min(abs(pos - other) for other in own_positions[peer])
            pu = by_id[uid]
            event_rows.append({"pair": f"{issuer} → {peer}", "issuer": issuer, "peer": peer, "earnings_id": r["earnings_id"],
                               "day0_session": pd.Timestamp(r["event_date"]).date().isoformat(), "position": pos,
                               "issuer_CAR_pct": unit_map[r["earnings_id"]].cars[MAIN] * 100,
                               "peer_CAR_pct": pu.cars[MAIN] * 100, "peer_SCAR": pu.cars[MAIN] / pu.car_standard_errors[MAIN],
                               "sessions_to_peer_own_results": int(gap), "clean": bool(gap > 2)})
        # Placebo baseline: the peer's own placebo days (no results of the peer nearby).
        base = [u.cars[MAIN] / u.car_standard_errors[MAIN] for u in placebo_by_stock.get(peer, [])]
        events = pd.DataFrame([e for e in event_rows if e["pair"] == f"{issuer} → {peer}"])
        if events.empty:
            continue
        events["cluster"] = events["position"].astype(str)
        for si, (sname, frame) in enumerate([("all days", events), ("clean days (peer did not report within 2 sessions)", events[events["clean"]])]):
            cl = frame["cluster"].to_numpy()
            v, vlo, vhi, k = cluster_bootstrap(cl, _mean(frame["peer_SCAR"].to_numpy(float) ** 2), SEED + 1000 + 10 * pi + si)
            c, clo, chi, _ = cluster_bootstrap(cl, _corr_moments(frame["issuer_CAR_pct"].to_numpy(float), frame["peer_CAR_pct"].to_numpy(float)), SEED + 2000 + 10 * pi + si)
            summary_rows.append({"pair": f"{issuer} → {peer}", "sample": sname, "days": len(frame),
                                 "peer_variance_ratio": v, "variance_ratio_ci95_lower": vlo, "variance_ratio_ci95_upper": vhi,
                                 "peer_placebo_variance_ratio": float(np.mean(np.square(base))) if base else np.nan,
                                 "correlation_issuer_vs_peer_CAR": c, "correlation_ci95_lower": clo, "correlation_ci95_upper": chi})
    events = pd.DataFrame(event_rows).drop(columns=["position"]) if event_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    # Baseline co-movement on ordinary days: same pair, both stocks' CARs on the issuer's placebo days.
    if not summary.empty and placebo_units:
        base_rows = []
        for issuer, peer in PEER_PAIRS:
            iss = {placebo_issuer[u.unit_id]: u for u in placebo_units if u.stock == issuer}
            table = pd.DataFrame({"unit_id": [f"{k}>{peer}" for k in iss], "stock": peer,
                                  "event_date": [sessions[u.event_position] for u in iss.values()]})
            if table.empty:
                continue
            pu, _, _ = build_units(returns, table, estimation_sessions, gap_sessions, radius)
            pmap = {u.unit_id.split(">")[0]: u for u in pu}
            pairs = [(iss[k].cars[MAIN], pmap[k].cars[MAIN]) for k in iss if k in pmap]
            if len(pairs) > 2:
                a, b = np.array(pairs).T
                base_rows.append({"pair": f"{issuer} → {peer}", "placebo_correlation": float(np.corrcoef(a, b)[0, 1])})
        if base_rows:
            summary = summary.merge(pd.DataFrame(base_rows), on="pair", how="left")
    return events, summary


# --------------------------------------------------------------------------- 3. trend
def volatility_trend(meta: pd.DataFrame, unit_map: Mapping[str, EventUnit], placebo_units: Sequence[EventUnit],
                     sessions: pd.DatetimeIndex, clusters: Mapping[str, str]) -> pd.DataFrame:
    real = pd.DataFrame({
        "id": meta["earnings_id"].to_numpy(),
        "year": pd.to_datetime(meta["event_date"]).dt.year.to_numpy(),
        "scar2": [(unit_map[e].cars[MAIN] / unit_map[e].car_standard_errors[MAIN]) ** 2 for e in meta["earnings_id"]],
        "overlap": meta["overlapping_registry_events"].fillna("").to_numpy(),
    })
    real["cluster"] = real["id"].map(clusters)
    plac = pd.DataFrame({
        "id": [u.unit_id for u in placebo_units],
        "year": [sessions[u.event_position].year for u in placebo_units],
        "scar2": [(u.cars[MAIN] / u.car_standard_errors[MAIN]) ** 2 for u in placebo_units],
    })
    # Same clustering rule as real announcements: pseudo-events whose Day -5..+5 windows overlap.
    placebo_clusters = _interval_clusters({u.unit_id: u.event_position - 5 for u in placebo_units},
                                          {u.unit_id: u.event_position + 5 for u in placebo_units})
    plac["cluster"] = plac["id"].map(placebo_clusters)
    rows = []
    for year in sorted(real["year"].unique()):
        r, p = real[real["year"] == year], plac[plac["year"] == year]
        v, lo, hi, _ = cluster_bootstrap(r["cluster"].to_numpy(), _mean(r["scar2"].to_numpy(float)), SEED + 3000 + int(year))
        rows.append({"row_type": "year", "label": str(year), "results_days": len(r), "variance_ratio": v,
                     "ci95_lower": lo, "ci95_upper": hi, "placebo_days": len(p),
                     "placebo_variance_ratio": float(p["scar2"].mean()) if len(p) else np.nan})
    def slope_spec(f):
        return _slope_moments(f["year"].to_numpy(float), f["scar2"].to_numpy(float))
    for name, frame, seed in [("slope per year: results days", real, 1), ("slope per year: results days without registry overlap", real[real["overlap"] == ""], 2),
                              ("slope per year: placebo days", plac, 3)]:
        v, lo, hi, k = cluster_bootstrap(frame["cluster"].to_numpy(), slope_spec(frame), SEED + 4000 + seed)
        rows.append({"row_type": "slope", "label": name, "results_days": len(frame), "variance_ratio": v,
                     "ci95_lower": lo, "ci95_upper": hi, "placebo_days": np.nan, "placebo_variance_ratio": np.nan})
    # Difference in slopes: resample both samples independently with the same draw count.
    def draws_for(frame, seed):
        sums, fn = slope_spec(frame)
        codes, labels = pd.factorize(frame["cluster"].astype(str))
        per = {n: np.bincount(codes, weights=v, minlength=len(labels)) for n, v in sums.items()}
        w = _weights(len(labels), seed)
        with np.errstate(divide="ignore", invalid="ignore"):
            return fn({n: w @ v for n, v in per.items()}), float(fn({n: v.sum() for n, v in per.items()}))
    dr, pr = draws_for(real, SEED + 4100)
    dp, pp_ = draws_for(plac, SEED + 4200)
    diff = dr - dp
    diff = diff[np.isfinite(diff)]
    rows.append({"row_type": "slope", "label": "difference: results minus placebo slope", "results_days": len(real),
                 "variance_ratio": pr - pp_, "ci95_lower": float(np.percentile(diff, 2.5)),
                 "ci95_upper": float(np.percentile(diff, 97.5)), "placebo_days": len(plac), "placebo_variance_ratio": np.nan})
    out = pd.DataFrame(rows)
    out["interval_excludes_zero"] = (out["row_type"] == "slope") & ((out["ci95_lower"] > 0) | (out["ci95_upper"] < 0))
    return out
