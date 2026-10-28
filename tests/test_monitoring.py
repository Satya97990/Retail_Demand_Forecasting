import numpy as np
import pytest

from retail_forecasting.monitoring import population_stability_index


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(7)
    values = rng.normal(size=1000)
    assert population_stability_index(values, values) == pytest.approx(0.0)


def test_psi_detects_a_shift():
    rng = np.random.default_rng(7)
    reference = rng.normal(0, 1, size=2000)
    current = rng.normal(2, 1, size=2000)
    assert population_stability_index(reference, current) > 0.25
