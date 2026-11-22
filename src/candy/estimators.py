"""Linear-model estimators built on NumPy (algebra) and SciPy (distributions).

The project brief permits only NumPy and SciPy for modelling, so ordinary least
squares and its inference are implemented here rather than imported from a
statistics library.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import special, stats

INTERCEPT = "intercept"
# Leverage this close to 1 means one observation alone pins down a term.
LEVERAGE_LIMIT = 1 - 1e-8

Covariance = Literal["classical", "hc3"]
Target = Literal["identity", "logit"]


class EstimationError(ValueError):
    """The model cannot be estimated on this design (e.g. aliased columns)."""


@dataclass(frozen=True)
class Bootstrap:
    """Case-resampling percentile intervals for the coefficients, indexed by term.

    ``discarded`` counts resamples redrawn because their design matrix was rank
    deficient (e.g. a rare attribute missing from the resample).
    """

    ci_low: pd.Series
    ci_high: pd.Series
    draws: int
    seed: int
    discarded: int

    def table(self) -> pd.DataFrame:
        """Percentile interval table with one row per term."""
        return pd.DataFrame({"ci_low": self.ci_low, "ci_high": self.ci_high})


@dataclass(frozen=True)
class Prediction:
    """Predicted value for one new row, on the reporting scale (win rate for a
    logit target), with a CI for the mean and a prediction interval."""

    estimate: float
    ci_low: float
    ci_high: float
    pi_low: float
    pi_high: float


@dataclass(frozen=True)
class Difference:
    """Difference between the predicted means of two rows, with its CI."""

    estimate: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class FitResult:
    """Coefficients and inference of a fitted linear model, indexed by term name."""

    coef: pd.Series
    se: pd.Series
    t: pd.Series
    p: pd.Series
    ci_low: pd.Series
    ci_high: pd.Series
    residuals: np.ndarray
    df_resid: int
    n_obs: int
    r_squared: float
    loo_r_squared: float
    cov: Covariance
    vif: pd.Series
    target: Target
    fitted: np.ndarray
    bootstrap: Bootstrap | None
    predictors: np.ndarray
    vcov: pd.DataFrame
    residual_variance: float
    alpha: float

    def predict(self, row: Mapping[str, float]) -> Prediction:
        """Predict one new row given as {predictor name: value}.

        The mean CI uses the fit's coefficient covariance (HC3 or classical);
        the prediction interval adds the residual variance for a single new
        observation. Both use the t distribution on the residual degrees of
        freedom. For a logit target all values are back-transformed to 0-100.
        """
        x = self._design_row(row)
        estimate, mean_var = self._linear_combination(x)
        crit = self._t_critical()
        ci_half = crit * np.sqrt(mean_var)
        pi_half = crit * np.sqrt(mean_var + self.residual_variance)
        values = np.array(
            [estimate, estimate - ci_half, estimate + ci_half, estimate - pi_half, estimate + pi_half]
        )
        if self.target == "logit":
            values = 100 * special.expit(values)
        return Prediction(*(float(v) for v in values))

    def predict_difference(self, row_a: Mapping[str, float], row_b: Mapping[str, float]) -> Difference:
        """Predicted mean of ``row_a`` minus that of ``row_b``, with a CI from
        the coefficient covariance. Identity target only: on a logit target the
        gap in win-rate points is not linear in the coefficients."""
        if self.target != "identity":
            raise ValueError("prediction differences need an identity target")
        estimate, variance = self._linear_combination(
            self._design_row(row_a) - self._design_row(row_b)
        )
        half = self._t_critical() * np.sqrt(variance)
        return Difference(estimate, estimate - half, estimate + half)

    def _linear_combination(self, x: np.ndarray) -> tuple[float, float]:
        """x'b and its variance x'Vx under the fit's coefficient covariance."""
        return float(x @ self.coef.to_numpy()), float(x @ self.vcov.to_numpy() @ x)

    def _t_critical(self) -> float:
        return float(stats.t.ppf(1 - self.alpha / 2, self.df_resid))

    def _design_row(self, row: Mapping[str, float]) -> np.ndarray:
        names = list(self.coef.index.drop(INTERCEPT))
        if set(row) != set(names):
            raise ValueError(f"row must give exactly the predictors {names}")
        return np.array([1.0, *(float(row[name]) for name in names)])

    def effects_in_points(self) -> pd.DataFrame:
        """Each predictor's effect, CI included, in points of win rate.

        For an identity target the coefficients are already in points. For a
        logit target this is the average marginal effect over the sample: the
        mean change in predicted win rate when a 0/1 predictor switches from 0
        to 1 for every observation, or the mean derivative for a continuous
        predictor. Each interval is the logit-scale interval rescaled by the
        same factor, which is treated as fixed (its own uncertainty is ignored).
        """
        table = self.table().drop(index=INTERCEPT)[["coef", "ci_low", "ci_high"]]
        if self.target == "identity":
            return table
        scale = pd.Series(
            [self._points_per_logit_unit(j) for j in range(self.predictors.shape[1])],
            index=table.index,
        )
        return table.mul(scale, axis=0)

    def _points_per_logit_unit(self, j: int) -> float:
        """How many win-rate points one logit unit of predictor ``j`` is worth,
        averaged over the candies."""
        column = self.predictors[:, j]
        beta = self.coef.iloc[j + 1]
        is_yes_no = set(np.unique(column)) <= {0, 1}
        if is_yes_no and beta != 0:
            # Predict every candy twice, once with the attribute off and once
            # with it on, and average the change in win rate.
            logit_without = special.logit(self.fitted / 100) - beta * column
            win_rate_off = 100 * special.expit(logit_without)
            win_rate_on = 100 * special.expit(logit_without + beta)
            average_change = float(np.mean(win_rate_on - win_rate_off))
            return average_change / beta
        # Continuous predictor: the slope of the logistic curve, p(1 - p),
        # averaged over the candies.
        share = self.fitted / 100
        return 100 * float(np.mean(share * (1 - share)))

    def table(self) -> pd.DataFrame:
        """Coefficient table with one row per term."""
        return pd.DataFrame(
            {
                "coef": self.coef,
                "se": self.se,
                "t": self.t,
                "p": self.p,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
            }
        )


def fit(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    *,
    cov: Covariance = "classical",
    alpha: float = 0.05,
    target: Target = "identity",
    bootstrap_draws: int | None = None,
    seed: int | None = None,
) -> FitResult:
    """Fit y on X by ordinary least squares, with an intercept added first.

    ``cov`` selects the coefficient covariance estimator: ``"classical"`` or ``"hc3"``.
    Tests and intervals use the t distribution with n - k degrees of freedom.

    ``target="logit"`` treats y as a win rate on 0-100, rescales it to 0-1 and
    regresses its logit; coefficients and inference are then on the logit
    scale, while ``fitted`` is back-transformed to 0-100.

    ``bootstrap_draws`` adds case-resampling percentile CIs on the coefficients,
    reproducible from ``seed`` (required whenever draws are requested).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if target == "logit":
        if not np.all((y > 0) & (y < 100)):
            raise ValueError("logit target needs win rates strictly between 0 and 100")
        y = special.logit(y / 100)
    elif target != "identity":
        raise ValueError(f"unknown target: {target!r}")
    n = X.shape[0]
    design = np.column_stack([np.ones(n), X])
    terms = [INTERCEPT, *names]
    k = design.shape[1]
    df_resid = n - k

    if np.linalg.matrix_rank(design) < k:
        raise EstimationError(
            "design matrix is rank deficient: some terms are exact linear "
            "combinations of others and cannot be estimated separately"
        )

    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    linear_fit = design @ beta
    resid = y - linear_fit
    xtx_inv = np.linalg.inv(design.T @ design)
    # Leverage h_i: the i-th diagonal entry of the hat matrix X (X'X)^-1 X'.
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    if np.any(leverage > LEVERAGE_LIMIT):
        raise EstimationError(
            "an observation has leverage 1 (it alone identifies a term), so "
            "its leave-one-out residual, and with it leave-one-out R² and HC3 "
            "standard errors, is undefined"
        )

    if cov == "classical":
        sigma2 = resid @ resid / df_resid
        vcov = sigma2 * xtx_inv
    elif cov == "hc3":
        # MacKinnon & White (1985) HC3: residuals inflated by leverage, which
        # suits small samples with heteroskedastic errors.
        weights = (resid / (1 - leverage)) ** 2
        meat = (design * weights[:, None]).T @ design
        vcov = xtx_inv @ meat @ xtx_inv
    else:
        raise ValueError(f"unknown covariance estimator: {cov!r}")

    se = np.sqrt(np.diag(vcov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df_resid)
    crit = stats.t.ppf(1 - alpha / 2, df_resid)
    tss = np.sum((y - y.mean()) ** 2)
    # Leave-one-out residuals have the closed form e_i / (1 - h_i), so PRESS
    # needs no refitting.
    press = np.sum((resid / (1 - leverage)) ** 2)

    def series(values: np.ndarray) -> pd.Series:
        return pd.Series(values, index=terms)

    return FitResult(
        coef=series(beta),
        se=series(se),
        t=series(t),
        p=series(p),
        ci_low=series(beta - crit * se),
        ci_high=series(beta + crit * se),
        residuals=resid,
        df_resid=df_resid,
        n_obs=n,
        r_squared=float(1 - resid @ resid / tss),
        loo_r_squared=float(1 - press / tss),
        cov=cov,
        vif=pd.Series(_vifs(X), index=names),
        target=target,
        fitted=100 * special.expit(linear_fit) if target == "logit" else linear_fit,
        predictors=X,
        vcov=pd.DataFrame(vcov, index=terms, columns=terms),
        residual_variance=float(resid @ resid / df_resid),
        alpha=alpha,
        bootstrap=(
            None
            if bootstrap_draws is None
            else _bootstrap(design, y, terms, bootstrap_draws, seed, alpha)
        ),
    )


def _bootstrap(
    design: np.ndarray,
    y: np.ndarray,
    terms: list[str],
    draws: int,
    seed: int | None,
    alpha: float,
) -> Bootstrap:
    if seed is None:
        raise ValueError("bootstrap needs a seed so results are reproducible")
    rng = np.random.default_rng(seed)
    n, k = design.shape
    coefs = []
    discarded = 0
    while len(coefs) < draws:
        rows = rng.integers(0, n, size=n)
        sample = design[rows]
        if np.linalg.matrix_rank(sample) < k:
            discarded += 1
            if discarded > draws:
                raise ValueError("too many rank-deficient bootstrap resamples")
            continue
        beta, *_ = np.linalg.lstsq(sample, y[rows], rcond=None)
        coefs.append(beta)
    low, high = np.percentile(np.array(coefs), [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return Bootstrap(
        ci_low=pd.Series(low, index=terms),
        ci_high=pd.Series(high, index=terms),
        draws=draws,
        seed=seed,
        discarded=discarded,
    )


def _vifs(X: np.ndarray) -> np.ndarray:
    """Variance inflation factor of each predictor: regress it on the other
    predictors and take 1 / (1 - R²). A VIF near 1 means the predictor is not
    explained by the others."""
    n, k = X.shape
    vifs = np.empty(k)
    for j in range(k):
        target = X[:, j]
        others = np.column_stack([np.ones(n), np.delete(X, j, axis=1)])
        beta, *_ = np.linalg.lstsq(others, target, rcond=None)
        resid = target - others @ beta
        r_squared = 1 - resid @ resid / np.sum((target - target.mean()) ** 2)
        vifs[j] = 1 / (1 - r_squared)
    return vifs


def holm(p: pd.Series) -> pd.Series:
    """Holm (1979) step-down adjusted p-values for one family of tests.

    Sort the m p-values from smallest to largest. Multiply the smallest by m,
    the next by m - 1, and so on. An adjusted value may never be smaller than
    the one before it, and none exceeds 1. Rejecting where the adjusted p-value
    is below alpha controls the family-wise error rate at alpha. The result
    keeps the input's labels and order.
    """
    m = len(p)
    adjusted = {}
    previous = 0.0
    for position, (term, value) in enumerate(p.sort_values(kind="stable").items()):
        current = min(1.0, max(previous, value * (m - position)))
        adjusted[term] = current
        previous = current
    return pd.Series(adjusted)[p.index]
