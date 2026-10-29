import numpy as np
import pandas as pd
import pytest

from src.robust_tests import (
    EVENT_DAYS, EventUnit, bmp_test, corrado_rank_test, group_tests, kolari_pynnonen,
    mean_residual_correlation, standardized_cars, variance_ratio,
)

DATES = pd.bdate_range("2020-01-01", periods=400)


def _unit(unit_id, residuals, event_ars, car=None, se=None, df=98, start=0):
    residuals = pd.Series(residuals, index=DATES[start:start + len(residuals)])
    ars = pd.Series(event_ars, index=list(EVENT_DAYS))
    car_value = float(ars[[-1, 0, 1]].sum()) if car is None else car
    se_value = float(np.sqrt(3) * residuals.std(ddof=2)) if se is None else se
    return EventUnit(unit_id, "S", residuals, ars, {"CAR_-1_+1": se_value}, {"CAR_-1_+1": car_value}, df)


def test_bmp_is_quiet_under_no_effect_and_detects_a_shift():
    rng = np.random.default_rng(1)
    null = bmp_test(pd.Series(rng.standard_normal(300)))
    shifted = bmp_test(pd.Series(rng.standard_normal(300) + 0.5))
    assert abs(null["bmp_t"]) < 3
    assert shifted["bmp_t"] > 5 and shifted["bmp_p_value"] < 1e-4


def test_bmp_uses_observed_spread_so_extra_volatility_does_not_create_signals():
    rng = np.random.default_rng(2)
    wide = pd.Series(rng.standard_normal(300) * 4)  # event-day volatility four times normal
    result = bmp_test(wide)
    naive_z = wide.mean() * np.sqrt(len(wide))  # assumes SCAR variance of 1
    assert abs(result["bmp_t"]) < 3
    assert result["sd_scar"] == pytest.approx(wide.std(ddof=1))
    assert abs(naive_z) > abs(result["bmp_t"])


def test_bmp_needs_three_units():
    assert np.isnan(bmp_test(pd.Series([1.0, 2.0]))["bmp_t"])


def test_kolari_pynnonen_formula_and_no_change_without_correlation():
    t, _ = kolari_pynnonen(4.0, 50, 0.0)
    assert t == pytest.approx(4.0)
    t, p = kolari_pynnonen(4.0, 50, 0.2)
    assert t == pytest.approx(4.0 * np.sqrt(0.8 / (1 + 49 * 0.2)))
    assert 0 < p < 1


def test_mean_residual_correlation_counts_non_overlapping_pairs_as_zero():
    rng = np.random.default_rng(3)
    base = rng.standard_normal(100)
    units = [
        _unit("a", base, np.zeros(11)),
        _unit("b", base, np.zeros(11)),          # same dates, same residuals -> correlation 1
        _unit("c", base, np.zeros(11), start=200),  # no overlap with a or b
    ]
    result = mean_residual_correlation(units)
    assert result["pairs"] == 3 and result["pairs_overlapping"] == 1
    assert result["mean_residual_correlation"] == pytest.approx(1 / 3)


def test_corrado_rank_detects_event_day_shift_and_resists_one_outlier():
    rng = np.random.default_rng(4)
    shifted = [_unit(f"u{i}", rng.standard_normal(100), np.r_[np.zeros(5), [3.0], np.zeros(5)] + rng.standard_normal(11) * 0.1)
               for i in range(40)]
    assert corrado_rank_test(shifted, -1, 1)["rank_z"] > 4
    quiet = [_unit(f"q{i}", rng.standard_normal(100), rng.standard_normal(11)) for i in range(40)]
    outlier = quiet[:-1] + [_unit("big", rng.standard_normal(100), np.r_[np.zeros(5), [1000.0], np.zeros(5)])]
    assert abs(corrado_rank_test(outlier, -1, 1)["rank_z"]) < 3


def test_corrado_skips_incomplete_event_windows():
    rng = np.random.default_rng(5)
    units = [_unit(f"u{i}", rng.standard_normal(100), rng.standard_normal(11)) for i in range(5)]
    broken = units[0].abnormal_returns.copy()
    broken[0] = np.nan
    units[0] = EventUnit("x", "S", units[0].estimation_residuals, broken, {}, {}, 98)
    assert corrado_rank_test(units, -1, 1)["rank_n"] == 4


def test_variance_ratio_is_about_one_without_effect_and_large_with_extra_volatility():
    rng = np.random.default_rng(6)
    ids = [f"u{i}" for i in range(300)]
    clusters = {i: i for i in ids}
    null = variance_ratio(pd.Series(rng.standard_normal(300), index=ids), clusters, seed=1)
    loud = variance_ratio(pd.Series(rng.standard_normal(300) * 3, index=ids), clusters, seed=1)
    assert 0.8 < null["variance_ratio"] < 1.2
    assert null["variance_ratio_ci_lower"] < 1 < null["variance_ratio_ci_upper"]
    assert loud["variance_ratio_ci_lower"] > 5


def test_variance_ratio_gives_no_interval_with_fewer_than_three_clusters():
    ids = ["a", "b", "c", "d"]
    result = variance_ratio(pd.Series([1.0, 2.0, 1.5, 0.5], index=ids), {"a": "C1", "b": "C1", "c": "C2", "d": "C2"}, seed=1)
    assert np.isnan(result["variance_ratio_ci_lower"]) and result["variance_ratio_clusters"] == 2


def test_group_tests_rows_and_standardized_cars():
    rng = np.random.default_rng(7)
    units = [_unit(f"u{i}", rng.standard_normal(100) * 0.01, rng.standard_normal(11) * 0.01) for i in range(12)]
    scars = standardized_cars(units, "CAR_-1_+1")
    assert len(scars) == 12
    rows = group_tests(units, {"CAR_-1_+1": (-1, 1)}, {u.unit_id: u.unit_id for u in units}, seed=3)
    assert len(rows) == 1
    row = rows[0]
    for key in ("bmp_t", "kp_adjusted_t", "rank_z", "variance_ratio", "variance_ratio_null_expectation"):
        assert key in row and np.isfinite(row[key])
    assert row["variance_ratio_null_expectation"] == pytest.approx(98 / 96)


def test_residual_correlation_ignores_pairs_whose_event_days_do_not_coincide():
    rng = np.random.default_rng(8)
    base = rng.standard_normal(100)
    near_a = _unit("a", base, np.zeros(11))
    near_b = _unit("b", base, np.zeros(11))
    far = _unit("c", base, np.zeros(11))
    units = [
        EventUnit(u.unit_id, u.stock, u.estimation_residuals, u.abnormal_returns, {}, {}, 98, event_position=pos)
        for u, pos in ((near_a, 150), (near_b, 152), (far, 215))
    ]
    result = mean_residual_correlation(units)
    # a-b share event days -> counted; c's event days are 60+ sessions away -> zero
    assert result["pairs_overlapping"] == 1
    assert result["mean_residual_correlation"] == pytest.approx(1 / 3)
