"""Seam 1: the estimator interface, checked against independent known answers."""

import numpy as np
import pytest
from scipy import stats

from candy.estimators import fit


def test_single_predictor_matches_scipy_linregress():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, size=40)
    y = 3.0 + 1.5 * x + rng.normal(0, 2.0, size=40)

    result = fit(x[:, None], y, names=["x"], cov="classical")
    ref = stats.linregress(x, y)

    assert result.coef["intercept"] == pytest.approx(ref.intercept)
    assert result.coef["x"] == pytest.approx(ref.slope)
    assert result.se["x"] == pytest.approx(ref.stderr)
    assert result.se["intercept"] == pytest.approx(ref.intercept_stderr)
    assert result.p["x"] == pytest.approx(ref.pvalue)
