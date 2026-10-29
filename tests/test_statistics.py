import numpy as np
import pytest
from scipy import stats

from src.statistics import car_significance, car_significance_hac, holm_adjust


def test_holm_adjustment_known_values_and_original_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.002])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.008])


def test_holm_adjustment_handles_ties_and_empty_input():
    np.testing.assert_allclose(holm_adjust([0.05, 0.05]), [0.1, 0.1])
    assert holm_adjust([]).size == 0


@pytest.mark.parametrize("values", [[-0.1], [1.1], [float("nan")], [[0.1]]])
def test_holm_rejects_invalid_p_values(values):
    with pytest.raises(ValueError):
        holm_adjust(values)


def test_car_standard_error_and_student_t_p_value_match_manual_formula():
    covariance = np.array([[0.000004, 0.0], [0.0, 0.0001]])
    model = {"covariance": covariance, "residual_variance": 0.0009, "degrees_freedom": 98}
    market = np.array([0.01, -0.02, 0.03])
    design = np.column_stack((np.ones(3), market))
    expected_variance = design.sum(axis=0) @ covariance @ design.sum(axis=0) + 3 * 0.0009
    result = car_significance(0.04, [-1, 0, 1], market, model)
    expected_t = 0.04 / np.sqrt(expected_variance)
    assert result["standard_error"] == pytest.approx(np.sqrt(expected_variance))
    assert result["t_statistic"] == pytest.approx(expected_t)
    assert result["p_value"] == pytest.approx(2 * stats.t.sf(abs(expected_t), df=98))


def test_hac_car_inference_reports_long_run_variance_for_serially_correlated_residuals():
    rng = np.random.default_rng(2026)
    residuals = np.zeros(120)
    innovations = rng.normal(0.0, 0.01, size=120)
    for index in range(1, len(residuals)):
        residuals[index] = 0.75 * residuals[index - 1] + innovations[index]
    model = {
        "covariance": np.array([[1e-7, 0.0], [0.0, 1e-5]]),
        "residual_variance": float(np.var(residuals, ddof=2)),
        "degrees_freedom": 118,
        "estimation_residuals": residuals.tolist(),
    }
    result = car_significance_hac(0.02, [0.01, -0.02, 0.03], model)
    assert result["standard_error"] > 0
    assert result["long_run_residual_variance"] > 0
    assert result["hac_bandwidth"] >= 1
    assert 0 <= result["p_value"] <= 1
    assert result["ci_lower_95"] < 0.02 < result["ci_upper_95"] or result["t_statistic"] > 0
