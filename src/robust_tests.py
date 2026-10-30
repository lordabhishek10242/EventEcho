"""Cross-sectional event-study tests that are robust to event-day volatility.

The per-event t-test in ``statistics.py`` assumes that an event does not
change the stock's volatility. Announcements often do, which makes that test
reject too often. This module adds three standard checks for a *group* of
event-stock pairs:

* **BMP test** (Boehmer, Musumeci and Poulsen, 1991). Each CAR is divided by
  its own standard error (giving a standardized CAR, "SCAR"). The test then
  uses the *observed* spread of the SCARs across events, so extra event-day
  volatility widens the test instead of creating false signals.
* **Kolari-Pynnönen adjustment** (2010). When events share calendar days (for
  example seven stocks on one budget day), their abnormal returns move
  together and the BMP test is too confident. The adjustment shrinks the BMP
  statistic using the average correlation of estimation-window residuals.
* **Corrado rank test** (Corrado 1989; multi-day form from Cowan 1992). Each
  event's abnormal returns are replaced by their ranks within that event's own
  estimation-plus-event sample, so a few extreme days cannot drive the result
  and no normal-distribution assumption is needed.

The module also reports the **event-induced variance ratio**: the average of
SCAR squared. Without any event effect it is close to 1. A value well above 1
means the stock moved more around the event than on ordinary days, whatever
the direction. This is the right measure when the direction of the news is not
known (for example earnings without analyst forecasts).

All results are exploratory group-level evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

EVENT_DAYS = tuple(range(-5, 6))
MIN_OVERLAP_FOR_CORRELATION = 30


@dataclass(frozen=True)
class EventUnit:
    """One event-stock pair with everything the group tests need."""

    unit_id: str
    stock: str
    estimation_residuals: pd.Series  # indexed by trading date
    abnormal_returns: pd.Series  # indexed by event day (-5..+5)
    car_standard_errors: Mapping[str, float]  # window name -> SE of CAR
    cars: Mapping[str, float]  # window name -> CAR
    degrees_freedom: int
    cluster: str = ""
    labels: Mapping[str, str] = field(default_factory=dict)
    event_position: int | None = None  # session number of day 0, if known


def standardized_cars(units: Sequence[EventUnit], window: str) -> pd.Series:
    values = {}
    for unit in units:
        car = unit.cars.get(window, np.nan)
        se = unit.car_standard_errors.get(window, np.nan)
        if np.isfinite(car) and np.isfinite(se) and se > 0:
            values[unit.unit_id] = car / se
    return pd.Series(values, dtype=float)


def bmp_test(scars: pd.Series) -> dict[str, float]:
    """BMP t-statistic: mean SCAR divided by its cross-sectional standard error."""
    values = scars.dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 3:
        return {"n": n, "mean_scar": float(values.mean()) if n else np.nan, "sd_scar": np.nan,
                "bmp_t": np.nan, "bmp_p_value": np.nan}
    sd = float(values.std(ddof=1))
    t_stat = float(np.sqrt(n) * values.mean() / sd) if sd > 0 else np.nan
    return {
        "n": n,
        "mean_scar": float(values.mean()),
        "sd_scar": sd,
        "bmp_t": t_stat,
        "bmp_p_value": float(2 * stats.t.sf(abs(t_stat), df=n - 1)) if np.isfinite(t_stat) else np.nan,
    }


def mean_residual_correlation(
    units: Sequence[EventUnit],
    min_overlap: int = MIN_OVERLAP_FOR_CORRELATION,
    event_window_span: int = 2 * EVENT_DAYS[-1],
) -> dict[str, float]:
    """Average pairwise correlation of estimation-window residuals.

    The adjustment is about abnormal returns that happen on the *same* days.
    So a pair only contributes its residual correlation when the two event
    windows overlap in calendar time (day-0 sessions at most
    ``event_window_span`` apart, when known) and their estimation samples
    share at least ``min_overlap`` days. Every other pair counts as zero.
    Without this rule, two results days of the same company a quarter apart
    would look almost perfectly "correlated" only because their estimation
    windows reuse the same dates, although their event days never coincide.
    """
    n = len(units)
    if n < 2:
        return {"mean_residual_correlation": 0.0, "pairs": 0, "pairs_overlapping": 0}
    wide = pd.concat(
        {unit.unit_id: unit.estimation_residuals.astype(float) for unit in units}, axis=1, sort=True
    )
    corr = wide.corr(min_periods=min_overlap).to_numpy(dtype=float)
    positions = np.array([np.nan if u.event_position is None else u.event_position for u in units], dtype=float)
    distance = np.abs(positions[:, None] - positions[None, :])
    same_days = ~(distance > event_window_span)  # unknown positions (NaN) stay eligible
    corr = np.where(same_days, corr, np.nan)
    upper = np.triu_indices(n, k=1)
    pair_values = corr[upper]
    overlapping = np.isfinite(pair_values)
    rbar = float(np.where(overlapping, pair_values, 0.0).mean())
    return {
        "mean_residual_correlation": rbar,
        "pairs": int(len(pair_values)),
        "pairs_overlapping": int(overlapping.sum()),
    }


def kolari_pynnonen(t_stat: float, n: int, rbar: float) -> tuple[float, float]:
    """Shrink a BMP statistic for cross-correlation; returns (t, two-sided p)."""
    if not np.isfinite(t_stat) or n < 3:
        return np.nan, np.nan
    denominator = 1.0 + (n - 1) * rbar
    if denominator <= 0 or rbar >= 1:
        return np.nan, np.nan
    adjusted = float(t_stat * np.sqrt((1.0 - rbar) / denominator))
    return adjusted, float(2 * stats.t.sf(abs(adjusted), df=n - 1))


def corrado_rank_test(units: Sequence[EventUnit], lower: int, upper: int) -> dict[str, float]:
    """Corrado rank test for the window [lower, upper] (Cowan's multi-day form).

    For each unit, the estimation residuals and the 11 event-window abnormal
    returns are ranked together. Ranks become U = rank / (count + 1) - 0.5,
    which is uniform around zero under no effect. The statistic sums the
    cross-event average U over the window and scales it by the time-series
    spread of that average, which also absorbs any correlation between events
    that share calendar days.
    """
    rows = []
    for unit in units:
        event = unit.abnormal_returns.reindex(list(EVENT_DAYS))
        if event.isna().any():
            continue
        estimation = unit.estimation_residuals.dropna().to_numpy(dtype=float)
        combined = np.concatenate([estimation, event.to_numpy(dtype=float)])
        ranks = stats.rankdata(combined)
        u = ranks / (len(combined) + 1.0) - 0.5
        estimation_u = pd.Series(u[: len(estimation)][::-1], index=[f"e{k}" for k in range(1, len(estimation) + 1)])
        event_u = pd.Series(u[len(estimation):], index=list(EVENT_DAYS))
        rows.append(pd.concat([estimation_u, event_u]))
    n = len(rows)
    if n < 3:
        return {"rank_n": n, "rank_z": np.nan, "rank_p_value": np.nan}
    matrix = pd.DataFrame(rows)
    mean_u = matrix.mean(axis=0, skipna=True)
    s_u = float(np.sqrt((mean_u ** 2).mean()))
    window_days = list(range(lower, upper + 1))
    length = len(window_days)
    z = float(mean_u[window_days].sum() / (np.sqrt(length) * s_u)) if s_u > 0 else np.nan
    return {
        "rank_n": n,
        "rank_z": z,
        "rank_p_value": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
    }


def variance_ratio(scars: pd.Series, clusters: Mapping[str, str], seed: int, draws: int = 5000) -> dict[str, float]:
    """Mean of SCAR squared, with a cluster-bootstrap 95% interval.

    The expected value with no event effect is df/(df-2), about 1.02 for a
    100-day estimation window; we report it so readers can compare.
    """
    values = scars.dropna()
    n = len(values)
    out = {"variance_ratio": float((values ** 2).mean()) if n else np.nan,
           "variance_ratio_ci_lower": np.nan, "variance_ratio_ci_upper": np.nan,
           "variance_ratio_clusters": 0,
           "share_abs_scar_above_1_96": float((values.abs() > 1.96).mean()) if n else np.nan}
    if n < 3:
        return out
    labels = np.array([clusters.get(unit_id, unit_id) for unit_id in values.index])
    unique = sorted(set(labels))
    out["variance_ratio_clusters"] = len(unique)
    if len(unique) < 3:
        return out
    squares = (values.to_numpy(dtype=float)) ** 2
    sums = np.array([squares[labels == label].sum() for label in unique])
    sizes = np.array([(labels == label).sum() for label in unique], dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(unique), size=(draws, len(unique)))
    boot = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)
    out["variance_ratio_ci_lower"] = float(np.percentile(boot, 2.5))
    out["variance_ratio_ci_upper"] = float(np.percentile(boot, 97.5))
    return out


def group_tests(
    units: Sequence[EventUnit],
    windows: Mapping[str, tuple[int, int]],
    clusters: Mapping[str, str],
    seed: int,
) -> list[dict]:
    """All robust tests for one group of units, one row per window."""
    correlation = mean_residual_correlation(units)
    rows = []
    for offset, (window, (lower, upper)) in enumerate(windows.items()):
        scars = standardized_cars(units, window)
        cars = pd.Series({u.unit_id: u.cars.get(window, np.nan) for u in units}, dtype=float).dropna()
        bmp = bmp_test(scars)
        kp_t, kp_p = kolari_pynnonen(bmp["bmp_t"], bmp["n"], correlation["mean_residual_correlation"])
        rank = corrado_rank_test(units, lower, upper)
        ratio = variance_ratio(scars, clusters, seed + offset)
        dfs = [u.degrees_freedom for u in units if u.unit_id in scars.index]
        null_ratio = float(np.mean([d / (d - 2) for d in dfs])) if dfs else np.nan
        rows.append({
            "window": window,
            "units": len(cars),
            "mean_CAR_pct": float(cars.mean() * 100) if len(cars) else np.nan,
            "median_CAR_pct": float(cars.median() * 100) if len(cars) else np.nan,
            "mean_abs_CAR_pct": float(cars.abs().mean() * 100) if len(cars) else np.nan,
            "share_positive": float((cars > 0).mean()) if len(cars) else np.nan,
            **bmp,
            "mean_residual_correlation": correlation["mean_residual_correlation"],
            "overlapping_pairs": correlation["pairs_overlapping"],
            "kp_adjusted_t": kp_t,
            "kp_adjusted_p_value": kp_p,
            **rank,
            **ratio,
            "variance_ratio_null_expectation": null_ratio,
        })
    return rows
