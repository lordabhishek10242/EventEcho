import numpy as np
import pytest
from scipy import stats

from src.statistics import car_significance, holm_adjust


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