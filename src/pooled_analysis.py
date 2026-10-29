"""Pooled reactions across events, window-choice comparison and replication tests.

Everything here is built from the per-event, per-stock abnormal returns and
CARs that the event study already produces.

* CAAR (cumulative average abnormal return): for each event, abnormal returns
  are first averaged over the chosen stocks, then averaged over events, then
  added up from Day -5. Uncertainty comes from a cluster bootstrap that
  resamples whole dependency clusters (overlapping windows and shared
  episodes), the same unit used for the strategy intervals.
* Groups: all events, events without CAR-window overlap, each event class,
  each sector, each stock, and the development/holdout samples.
* Window choice: how counts, sizes and verdicts change between the
  (-1,+1), (-3,+3) and (-5,+5) windows.
* Replication: whether patterns seen in the development events show up again
  in the later holdout events.

All results are exploratory. Small groups get no interval rather than a
misleading one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .config import CAR_WINDOWS, EVENT_CLASS_BY_CATEGORY, STOCK_SECTORS

BOOTSTRAP_DRAWS = 5000
RANDOM_SEED = 2026
EVENT_DAYS = tuple(range(-5, 6))
# With fewer than three independent clusters a bootstrap "interval" only
# reshuffles one or two numbers, so no interval is reported.
NO_INTERVAL_STATUSES = ("no_complete_events", "single_event_descriptive_only", "fewer_than_3_clusters_no_interval")


def _group_definitions(
    events: pd.DataFrame,
    stocks: Sequence[str],
    event_window_audit: pd.DataFrame,
    development_ids: Iterable[str],
    holdout_ids: Iterable[str],
) -> list[tuple[str, str, list[str], list[str]]]:
    ids = events["event_id"].astype(str).tolist()
    classes = events.set_index("event_id")["category"].map(EVENT_CLASS_BY_CATEGORY)
    if classes.isna().any():
        raise ValueError(f"Unmapped categories: {sorted(events.loc[classes.isna().to_numpy(), 'category'])}")
    overlap = event_window_audit.set_index("event_id")["event_window_overlap_count"]
    clean = [event_id for event_id in ids if overlap.get(event_id, 0) == 0]
    groups = [
        ("all", "All events", ids, list(stocks)),
        ("all", "All events without CAR-window overlap", clean, list(stocks)),
        ("sample", "Development", [i for i in ids if i in set(development_ids)], list(stocks)),
        ("sample", "Holdout", [i for i in ids if i in set(holdout_ids)], list(stocks)),
    ]
    for event_class in sorted(classes.unique()):
        groups.append(("event_class", event_class, classes[classes == event_class].index.tolist(), list(stocks)))
    sectors: dict[str, list[str]] = {}
    for stock in stocks:
        sectors.setdefault(STOCK_SECTORS.get(stock, "Unassigned"), []).append(stock)
    for sector, members in sorted(sectors.items()):
        groups.append(("sector", sector, ids, members))
    for stock in stocks:
        groups.append(("stock", stock, ids, [stock]))
    return groups


def _event_paths(event_study: pd.DataFrame, event_ids: Sequence[str], stocks: Sequence[str]) -> pd.DataFrame:
    """Rows = events, columns = event days; values = mean AR over the stocks.

    An event is kept only if every chosen stock has all 11 sessions, so a
    missing observation never becomes a zero.
    """
    subset = event_study[event_study["event_id"].isin(event_ids) & event_study["stock"].isin(stocks)]
    complete = subset.groupby(["event_id", "stock"])["event_day"].nunique()
    complete = complete[complete == len(EVENT_DAYS)].reset_index()
    full_events = complete.groupby("event_id")["stock"].nunique()
    full_events = full_events[full_events == len(stocks)].index
    subset = subset[subset["event_id"].isin(full_events)]
    return subset.pivot_table(index="event_id", columns="event_day", values="abnormal_return", aggfunc="mean").reindex(columns=list(EVENT_DAYS))


def _cluster_bootstrap(values: pd.DataFrame, clusters: dict[str, str], cumulative: bool, seed: int):
    """Percentile-ready draws from resampling whole clusters of events.

    Each draw picks clusters with replacement, keeps every event in a picked
    cluster, and averages the event rows (then adds them up over event days
    when ``cumulative`` is true). Returns (draws or None, number of clusters).
    """
    cluster_of = np.array([clusters[event_id] for event_id in values.index])
    labels = sorted(set(cluster_of))
    if len(labels) < 2:
        return None, len(labels)
    matrix = values.to_numpy(dtype=float)
    members = [np.flatnonzero(cluster_of == label) for label in labels]
    sums = np.array([matrix[rows].sum(axis=0) for rows in members])
    sizes = np.array([len(rows) for rows in members], dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(labels), size=(BOOTSTRAP_DRAWS, len(labels)))
    draws = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)[:, None]
    if cumulative:
        draws = draws.cumsum(axis=1)
    return draws, len(labels)


def _status(n_events: int, n_clusters: int) -> str:
    if n_events == 0:
        return "no_complete_events"
    if n_events == 1:
        return "single_event_descriptive_only"
    if n_clusters < 3:
        return "fewer_than_3_clusters_no_interval"
    if n_clusters < 5:
        return "few_clusters_interval_very_rough"
    return "cluster_bootstrap_exploratory"


def caar_tables(
    event_study: pd.DataFrame,
    car_results: pd.DataFrame,
    events: pd.DataFrame,
    stocks: Sequence[str],
    event_window_audit: pd.DataFrame,
    clusters: dict[str, str],
    development_ids: Iterable[str],
    holdout_ids: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (CAAR path by event day, pooled CAR by window) for every group."""
    path_rows: list[dict] = []
    window_rows: list[dict] = []
    groups = _group_definitions(events, stocks, event_window_audit, development_ids, holdout_ids)
    for group_index, (group_type, group, event_ids, members) in enumerate(groups):
        paths = _event_paths(event_study, event_ids, members)
        n_events = len(paths)
        seed = RANDOM_SEED + group_index
        caar = paths.mean(axis=0).cumsum() if n_events else pd.Series(np.nan, index=list(EVENT_DAYS))
        aar = paths.mean(axis=0) if n_events else pd.Series(np.nan, index=list(EVENT_DAYS))
        draws, n_clusters = _cluster_bootstrap(paths, clusters, True, seed) if n_events else (None, 0)
        status = _status(n_events, n_clusters)
        for position, day in enumerate(EVENT_DAYS):
            lower = upper = np.nan
            if draws is not None and status not in NO_INTERVAL_STATUSES:
                lower, upper = np.quantile(draws[:, position], [0.025, 0.975])
            path_rows.append({
                "group_type": group_type, "group": group, "event_day": day,
                "events": n_events, "dependency_clusters": n_clusters, "stocks": len(members),
                "AAR_pct": aar[day] * 100, "CAAR_pct": caar[day] * 100,
                "CAAR_ci95_lower_pct": lower * 100, "CAAR_ci95_upper_pct": upper * 100,
                "status": status,
            })

        for window, (start, end) in CAR_WINDOWS.items():
            subset = car_results[car_results["event_id"].isin(paths.index) & car_results["stock"].isin(members) & (car_results["window"] == window)]
            event_car = subset.groupby("event_id")["CAR"].mean() if not subset.empty else pd.Series(dtype=float)
            event_car = event_car.dropna()
            values = event_car.to_frame("CAR")
            lower = upper = np.nan
            car_draws, clusters_used = _cluster_bootstrap(values, clusters, False, seed + 100) if len(values) else (None, 0)
            window_status = _status(len(values), clusters_used)
            if car_draws is not None and window_status not in NO_INTERVAL_STATUSES:
                lower, upper = np.quantile(car_draws[:, 0], [0.025, 0.975])
            t_stat = p_value = np.nan
            if len(values) >= 3 and event_car.std(ddof=1) > 0:
                t_stat = float(event_car.mean() / (event_car.std(ddof=1) / np.sqrt(len(event_car))))
                p_value = float(2 * stats.t.sf(abs(t_stat), df=len(event_car) - 1))
            window_rows.append({
                "group_type": group_type, "group": group, "window": window,
                "events": int(len(values)), "dependency_clusters": clusters_used, "stocks": len(members),
                "mean_CAR_pct": event_car.mean() * 100 if len(values) else np.nan,
                "median_CAR_pct": event_car.median() * 100 if len(values) else np.nan,
                "share_events_positive_pct": (event_car > 0).mean() * 100 if len(values) else np.nan,
                "mean_CAR_ci95_lower_pct": lower * 100, "mean_CAR_ci95_upper_pct": upper * 100,
                "interval_includes_zero": (bool(lower <= 0 <= upper) if np.isfinite(lower) else None),
                "cross_sectional_t": t_stat, "cross_sectional_p_value_assumes_independence": p_value,
                "status": window_status,
            })
    return pd.DataFrame(path_rows), pd.DataFrame(window_rows)


def window_choice_tables(significance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe how the CAR window changes sizes and verdicts."""
    summary: list[dict] = []
    for window, (start, end) in CAR_WINDOWS.items():
        rows = significance[significance["window"] == window]
        summary.append({
            "window": window, "sessions": end - start + 1, "tests": int(len(rows)),
            "conventional_holm_significant": int(rows["significant_after_holm_5pct"].sum()),
            "hac_holm_significant": int(rows["hac_significant_after_holm_5pct"].sum()),
            "mean_CAR_pct": rows["CAR_pct"].mean(),
            "mean_absolute_CAR_pct": rows["CAR_pct"].abs().mean(),
            "median_standard_error_pct": rows["standard_error_pct"].median(),
            "median_hac_standard_error_pct": rows["hac_standard_error_pct"].median(),
            "share_positive_pct": (rows["CAR_pct"] > 0).mean() * 100,
        })
    keyed = significance.set_index(["event_id", "stock", "window"])
    names = list(CAR_WINDOWS)
    agreement: list[dict] = []
    for i, narrow in enumerate(names):
        for wide in names[i + 1:]:
            a = keyed.xs(narrow, level="window")
            b = keyed.xs(wide, level="window")
            joined = a.merge(b, left_index=True, right_index=True, how="inner", suffixes=("_narrow", "_wide"))
            sig_a = joined["hac_significant_after_holm_5pct_narrow"].astype(bool)
            sig_b = joined["hac_significant_after_holm_5pct_wide"].astype(bool)
            same_sign = (np.sign(joined["CAR_pct_narrow"]) == np.sign(joined["CAR_pct_wide"])).mean() * 100
            row = {
                "narrow_window": narrow, "wide_window": wide, "event_stock_pairs": int(len(joined)),
                "same_sign_share_pct": same_sign,
                "CAR_correlation": (
                    float(np.corrcoef(joined["CAR_pct_narrow"], joined["CAR_pct_wide"])[0, 1])
                    if len(joined) >= 3 and joined["CAR_pct_narrow"].std() > 0 and joined["CAR_pct_wide"].std() > 0 else np.nan
                ),
                "hac_significant_in_both": int((sig_a & sig_b).sum()),
                "hac_significant_only_narrow": int((sig_a & ~sig_b).sum()),
                "hac_significant_only_wide": int((~sig_a & sig_b).sum()),
                "verdict_changes": int((sig_a != sig_b).sum()),
            }
            row["interpretation"] = (
                f"{row['same_sign_share_pct']:.0f}% of event-stock pairs keep the same CAR sign from {narrow} to {wide}; "
                f"{row['verdict_changes']} of {row['event_stock_pairs']} change their HAC-Holm significance verdict "
                f"({row['hac_significant_only_narrow']} significant only in the narrow window, {row['hac_significant_only_wide']} only in the wide one). "
                "Wider windows add more days of noise and more chance of other news, so a result that holds only in one window should be treated as fragile."
            )
            agreement.append(row)
    return pd.DataFrame(summary), pd.DataFrame(agreement)


def pattern_generalization(
    car_results: pd.DataFrame,
    events: pd.DataFrame,
    stocks: Sequence[str],
    development_ids: Iterable[str],
    holdout_ids: Iterable[str],
) -> pd.DataFrame:
    """Check whether development-sample patterns repeat in the holdout sample."""
    development, holdout = set(map(str, development_ids)), set(map(str, holdout_ids))
    classes = events.set_index("event_id")["category"].map(EVENT_CLASS_BY_CATEGORY)
    rows: list[dict] = []
    for window in CAR_WINDOWS:
        frame = car_results[(car_results["window"] == window) & car_results["stock"].isin(stocks)].dropna(subset=["CAR"])
        event_car = frame.groupby("event_id")["CAR"].mean() * 100
        dev = event_car[event_car.index.isin(development)]
        hold = event_car[event_car.index.isin(holdout)]

        same = bool(np.sign(dev.mean()) == np.sign(hold.mean()))
        rows.append({
            "test": "overall_mean_CAR_direction", "window": window, "group": "All events",
            "development_value": dev.mean(), "holdout_value": hold.mean(),
            "development_n": len(dev), "holdout_n": len(hold),
            "statistic": np.nan, "p_value": np.nan, "replicated": same,
            "interpretation": f"Mean event CAR is {dev.mean():+.2f}% in development and {hold.mean():+.2f}% in holdout; direction {'matches' if same else 'does not match'}.",
        })

        hits, tried = 0, 0
        for event_class in sorted(classes.unique()):
            in_class = classes[classes == event_class].index
            d, h = dev[dev.index.isin(in_class)], hold[hold.index.isin(in_class)]
            if d.empty or h.empty:
                continue
            match = bool(np.sign(d.mean()) == np.sign(h.mean()))
            rows.append({
                "test": "event_class_mean_CAR_direction", "window": window, "group": event_class,
                "development_value": d.mean(), "holdout_value": h.mean(),
                "development_n": len(d), "holdout_n": len(h),
                "statistic": np.nan, "p_value": np.nan, "replicated": match,
                "interpretation": f"{event_class}: development mean {d.mean():+.2f}% (n={len(d)}), holdout mean {h.mean():+.2f}% (n={len(h)}).",
            })
            predicted_sign = np.sign(d.mean())
            hits += int((np.sign(h) == predicted_sign).sum())
            tried += len(h)
        if tried:
            test = stats.binomtest(hits, tried, 0.5)
            rows.append({
                "test": "predict_holdout_sign_from_development_class_mean", "window": window, "group": "Classes present in both samples",
                "development_value": np.nan, "holdout_value": hits / tried * 100,
                "development_n": np.nan, "holdout_n": tried,
                "statistic": hits, "p_value": float(test.pvalue), "replicated": bool(test.pvalue < 0.05 and hits / tried > 0.5),
                "interpretation": f"Using each class's development mean sign to guess holdout event signs: {hits} of {tried} correct ({hits / tried:.0%}); a coin flip gets about 50% (binomial p = {test.pvalue:.2f}).",
            })

        per_stock = frame.assign(sample=frame["event_id"].map(lambda e: "dev" if e in development else "hold" if e in holdout else None))
        per_stock = per_stock.dropna(subset=["sample"]).assign(absolute=lambda f: f["CAR"].abs() * 100)
        size = per_stock.pivot_table(index="stock", columns="sample", values="absolute", aggfunc="mean").dropna()
        if len(size) >= 3:
            rho, p = stats.spearmanr(size["dev"], size["hold"])
            rows.append({
                "test": "stock_reaction_size_rank_persistence", "window": window, "group": "Stocks",
                "development_value": np.nan, "holdout_value": np.nan,
                "development_n": len(size), "holdout_n": len(size),
                "statistic": float(rho), "p_value": float(p), "replicated": bool(p < 0.05 and rho > 0),
                "interpretation": f"Rank correlation of each stock's average |CAR| between samples is {rho:.2f} (n={len(size)} stocks, p = {p:.2f}). This mainly shows that volatile stocks move more in any period; it is not evidence of a repeatable event effect.",
            })
    return pd.DataFrame(rows)
