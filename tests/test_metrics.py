import pytest

from retail_forecasting.metrics import calculate_metrics


def test_metrics_have_expected_values_and_zero_safe_smape():
    metrics = calculate_metrics([0, 10], [0, 12])
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx((2.0**2 / 2) ** 0.5)
    assert metrics["wape_pct"] == pytest.approx(20.0)
    assert metrics["smape_pct"] == pytest.approx((0 + 2 * 2 / 22) / 2 * 100)


def test_metrics_reject_different_shapes():
    with pytest.raises(ValueError, match="must match"):
        calculate_metrics([1, 2], [1])
