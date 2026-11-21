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


# Small fixed dataset with HC3 results computed once by an external reference
# implementation (t-based inference, 5 residual degrees of freedom).
HC3_X = np.array([[1, 0.0], [2, 1], [3, 0], [4, 1], [5, 0], [6, 1], [7, 0], [8, 1]])
HC3_Y = np.array([2.1, 3.9, 6.2, 8.1, 9.8, 12.5, 13.9, 16.4])


def test_hc3_inference_matches_reference_values():
    result = fit(HC3_X, HC3_Y, names=["a", "b"], cov="hc3")

    np.testing.assert_allclose(result.coef, [-0.09, 2.0225, 0.2025], rtol=1e-9)
    np.testing.assert_allclose(
        result.se, [0.267470746222, 0.054223778513, 0.245165484998], rtol=1e-9
    )
    np.testing.assert_allclose(
        result.p, [7.501675750013e-01, 2.608983056973e-07, 4.464270832698e-01], rtol=1e-8
    )
    np.testing.assert_allclose(
        result.ci_low, [-0.777555441803, 1.883113339895, -0.42771794246], rtol=1e-9
    )
    np.testing.assert_allclose(
        result.ci_high, [0.597555441803, 2.161886660105, 0.83271794246], rtol=1e-9
    )
    assert result.df_resid == 5


def test_hc3_equals_sum_of_squared_leave_one_out_coefficient_shifts():
    # Exact identity: b - b_(i) = (X'X)^-1 x_i e_i / (1 - h_i), so the HC3
    # covariance is the sum over i of (b - b_(i))(b - b_(i))'. Brute-force refits
    # give an oracle independent of the closed-form implementation.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    y = X @ [1.0, -2.0, 0.5] + rng.normal(scale=1 + np.abs(X[:, 0]), size=30)
    names = ["a", "b", "c"]
    full = fit(X, y, names=names, cov="hc3")

    shifts = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        shifts.append(full.coef.to_numpy() - fit(X[keep], y[keep], names).coef.to_numpy())
    shifts = np.array(shifts)
    brute_force_se = np.sqrt(np.sum(shifts**2, axis=0))

    np.testing.assert_allclose(full.se, brute_force_se, rtol=1e-10)


def test_recovers_known_coefficients_from_synthetic_data():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(5000, 2))
    y = 10.0 + X @ [4.0, -3.0] + rng.normal(scale=0.5, size=5000)

    result = fit(X, y, names=["a", "b"], cov="hc3")

    np.testing.assert_allclose(result.coef, [10.0, 4.0, -3.0], atol=0.05)
    for term, truth in zip(result.coef.index, [10.0, 4.0, -3.0]):
        assert result.ci_low[term] < truth < result.ci_high[term]


def test_exact_linear_data_gives_perfect_fit():
    X = np.arange(10, dtype=float)[:, None]
    y = 2.0 + 0.5 * X[:, 0]

    result = fit(X, y, names=["x"])

    np.testing.assert_allclose(result.coef, [2.0, 0.5], atol=1e-12)
    assert result.r_squared == pytest.approx(1.0)
