"""Seam 1: the estimator interface, checked against independent known answers."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from candy.estimators import fit, holm


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


# Small fixed dataset. Expected HC3 values were computed once, outside this
# project, with statsmodels 0.15.0:
#   sm.OLS(HC3_Y, sm.add_constant(HC3_X)).fit(cov_type="HC3", use_t=True)
# statsmodels is not a project dependency; only its outputs are pinned here.
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
    np.testing.assert_allclose(result.residuals, 0.0, atol=1e-12)
    assert result.r_squared == pytest.approx(1.0)


def test_vifs_equal_one_for_orthogonal_predictors():
    # Full 2^3 factorial in +/-1 coding, replicated: columns are mutually
    # orthogonal and centred, so no predictor is explained by the others.
    levels = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    X = np.vstack([levels, levels]).astype(float)
    y = np.random.default_rng(3).normal(size=len(X))

    result = fit(X, y, names=["a", "b", "c"])

    assert list(result.vif.index) == ["a", "b", "c"]
    np.testing.assert_allclose(result.vif, 1.0, rtol=1e-12)


def test_vifs_match_auxiliary_regressions_for_correlated_predictors():
    # Definition: VIF_j = 1 / (1 - R²_j), with R²_j from regressing predictor j
    # on all the others.
    rng = np.random.default_rng(4)
    a = rng.normal(size=60)
    b = 0.8 * a + rng.normal(scale=0.5, size=60)
    c = rng.normal(size=60) - 0.4 * b
    X = np.column_stack([a, b, c])

    result = fit(X, rng.normal(size=60), names=["a", "b", "c"])

    for j, name in enumerate(["a", "b", "c"]):
        others = np.delete(X, j, axis=1)
        aux = fit(others, X[:, j], names=["u", "v"])
        assert result.vif[name] == pytest.approx(1 / (1 - aux.r_squared), rel=1e-10)


def test_loo_r_squared_matches_brute_force_leave_one_out_refits():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(25, 3))
    y = 1.0 + X @ [2.0, 0.0, -1.0] + rng.normal(size=25)
    names = ["a", "b", "c"]

    result = fit(X, y, names=names)

    held_out_errors = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        refit = fit(X[keep], y[keep], names)
        prediction = refit.coef["intercept"] + X[i] @ refit.coef[names].to_numpy()
        held_out_errors.append(y[i] - prediction)
    press = np.sum(np.square(held_out_errors))
    expected = 1 - press / np.sum((y - y.mean()) ** 2)

    assert result.loo_r_squared == pytest.approx(expected, rel=1e-10)
    assert result.loo_r_squared < result.r_squared


def test_holm_adjustment_matches_hand_computed_values():
    # Sorted p: 0.005, 0.01, 0.03, 0.04 with m = 4.
    # Step-down multipliers 4, 3, 2, 1 give 0.02, 0.03, 0.06, 0.04; enforcing a
    # running maximum lifts the last to 0.06. Labels keep the input order.
    p = pd.Series({"w": 0.01, "x": 0.04, "y": 0.03, "z": 0.005})

    adjusted = holm(p)

    assert list(adjusted.index) == ["w", "x", "y", "z"]
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02], rtol=1e-12)


def test_holm_adjusted_p_values_are_capped_at_one():
    adjusted = holm(pd.Series({"a": 0.4, "b": 0.6, "c": 0.9}))

    # 0.4 * 3 = 1.2 -> 1; 0.6 * 2 = 1.2 -> 1; 0.9 * 1 = 0.9 -> raised to 1.
    np.testing.assert_allclose(adjusted, [1.0, 1.0, 1.0])


def test_logit_target_recovers_an_exact_logistic_curve_on_the_win_rate_scale():
    # Win rates generated exactly as 100 / (1 + exp(-(-1 + 0.5 x))): on the
    # logit scale the fit is exact, so the true coefficients come back and the
    # back-transformed fitted values reproduce the win rates.
    x = np.linspace(-6, 8, 15)
    win_rate = 100 / (1 + np.exp(-(-1.0 + 0.5 * x)))

    result = fit(x[:, None], win_rate, names=["x"], target="logit")

    np.testing.assert_allclose(result.coef, [-1.0, 0.5], atol=1e-10)
    np.testing.assert_allclose(result.fitted, win_rate, rtol=1e-10)
    assert result.target == "logit"


def test_logit_fitted_values_stay_within_0_to_100():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(200, 2))
    win_rate = np.clip(50 + 30 * X[:, 0] + rng.normal(scale=10, size=200), 1, 99)

    result = fit(X, win_rate, names=["a", "b"], target="logit")

    assert np.all((result.fitted > 0) & (result.fitted < 100))
    # An unbounded linear fit on the same data overshoots the scale.
    linear = fit(X, win_rate, names=["a", "b"])
    assert np.any((linear.fitted < 0) | (linear.fitted > 100))


def test_logit_target_rejects_win_rates_outside_the_open_interval():
    with pytest.raises(ValueError, match="strictly between 0 and 100"):
        fit(np.arange(4.0)[:, None], np.array([0.0, 20, 50, 80]), names=["x"], target="logit")


def test_logit_effects_in_points_match_numerical_average_marginal_effects():
    # Oracle: nudge each predictor by a tiny step for every observation and
    # average the change in back-transformed predicted win rate.
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 2))
    win_rate = 100 / (1 + np.exp(-(0.3 + 0.8 * X[:, 0] - 0.4 * X[:, 1]))) + rng.normal(size=80)
    names = ["a", "b"]
    result = fit(X, win_rate, names=names, target="logit")

    def predict(design: np.ndarray) -> np.ndarray:
        z = result.coef["intercept"] + design @ result.coef[names].to_numpy()
        return 100 / (1 + np.exp(-z))

    step = 1e-6
    effects = result.effects_in_points()
    for j, name in enumerate(names):
        nudged = X.copy()
        nudged[:, j] += step
        numerical = np.mean((predict(nudged) - predict(X)) / step)
        assert effects.loc[name, "coef"] == pytest.approx(numerical, rel=1e-5)
        assert effects.loc[name, "ci_low"] < effects.loc[name, "coef"] < effects.loc[name, "ci_high"]
    assert list(effects.index) == names


def test_effects_in_points_are_the_coefficients_for_an_identity_target():
    result = fit(HC3_X, HC3_Y, names=["a", "b"], cov="hc3")

    effects = result.effects_in_points()

    np.testing.assert_allclose(effects["coef"], [2.0225, 0.2025], rtol=1e-9)
    np.testing.assert_allclose(effects["ci_low"], [1.883113339895, -0.42771794246], rtol=1e-9)


def _noisy_data(seed: int, n: int = 60):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = 1.0 + X @ [2.0, -1.0] + rng.normal(scale=1 + np.abs(X[:, 0]), size=n)
    return X, y


def test_bootstrap_with_a_fixed_seed_is_deterministic():
    X, y = _noisy_data(8)

    first = fit(X, y, names=["a", "b"], bootstrap_draws=200, seed=42).bootstrap
    second = fit(X, y, names=["a", "b"], bootstrap_draws=200, seed=42).bootstrap
    other = fit(X, y, names=["a", "b"], bootstrap_draws=200, seed=43).bootstrap

    pd.testing.assert_series_equal(first.ci_low, second.ci_low)
    pd.testing.assert_series_equal(first.ci_high, second.ci_high)
    assert not np.allclose(first.ci_low, other.ci_low)
    assert (first.seed, first.draws) == (42, 200)


def test_bootstrap_is_off_unless_draws_are_requested():
    X, y = _noisy_data(9)

    assert fit(X, y, names=["a", "b"]).bootstrap is None
    with pytest.raises(ValueError, match="seed"):
        fit(X, y, names=["a", "b"], bootstrap_draws=100)


def test_bootstrap_intervals_agree_with_hc3_intervals_on_a_large_sample():
    # Case resampling and HC3 both target the heteroskedasticity-robust
    # sampling variance, so with n large their 95% intervals nearly coincide.
    X, y = _noisy_data(10, n=2000)

    result = fit(X, y, names=["a", "b"], cov="hc3", bootstrap_draws=1000, seed=1)

    hc3_width = result.ci_high - result.ci_low
    boot_width = result.bootstrap.ci_high - result.bootstrap.ci_low
    np.testing.assert_allclose(boot_width, hc3_width, rtol=0.1)
    assert np.all(np.abs(result.bootstrap.ci_low - result.ci_low) < 0.1 * hc3_width)


def test_bootstrap_redraws_resamples_that_lose_a_rare_attribute():
    # Two of 20 rows carry the attribute: about 12% of resamples miss both,
    # leaving a column of zeros and an unidentifiable coefficient.
    rng = np.random.default_rng(11)
    rare = np.zeros(20)
    rare[[3, 15]] = 1
    X = np.column_stack([rng.normal(size=20), rare])
    y = X @ [1.0, 5.0] + rng.normal(size=20)

    boot = fit(X, y, names=["x", "rare"], bootstrap_draws=300, seed=3).bootstrap

    assert boot.discarded > 0
    assert np.all(np.isfinite(boot.ci_low)) and np.all(np.isfinite(boot.ci_high))
