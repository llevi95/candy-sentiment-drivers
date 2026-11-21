"""Hand-written linear-model estimators (NumPy for algebra, SciPy for distributions).

The project brief permits only NumPy and SciPy for modelling, so ordinary least
squares and its inference are implemented here rather than imported from a
statistics library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import special, stats

INTERCEPT = "intercept"

Covariance = Literal["classical", "hc3"]
Target = Literal["identity", "logit"]


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

    def effects_in_points(self) -> pd.DataFrame:
        """Each predictor's effect, CI included, in points of win rate.

        For a logit target this is the average marginal effect: the logit-scale
        coefficient times the mean of 100 * p(1 - p) over the fitted win rates.
        The scale factor is treated as fixed, so the intervals are the
        logit-scale intervals rescaled. For an identity target the coefficients
        are already in points.
        """
        if self.target == "logit":
            share = self.fitted / 100
            scale = 100 * float(np.mean(share * (1 - share)))
        else:
            scale = 1.0
        table = self.table().drop(index=INTERCEPT)
        return table[["coef", "ci_low", "ci_high"]] * scale

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

    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    linear_fit = design @ beta
    resid = y - linear_fit
    xtx_inv = np.linalg.inv(design.T @ design)
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)

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
    """Variance inflation factor of each predictor: 1 / (1 - R²) from regressing
    it on the others, read off the diagonal of the inverse correlation matrix."""
    corr = np.atleast_2d(np.corrcoef(X, rowvar=False))
    return np.diag(np.linalg.inv(corr))


def holm(p: pd.Series) -> pd.Series:
    """Holm (1979) step-down adjusted p-values for one family of tests.

    The smallest of m p-values is multiplied by m, the next by m - 1, and so on;
    a running maximum keeps the adjusted values monotone, and each is capped at 1.
    Rejecting where the adjusted p-value is below alpha controls the family-wise
    error rate at alpha. The result keeps the input's labels and order.
    """
    order = np.argsort(p.to_numpy(), kind="stable")
    m = len(p)
    stepped = p.to_numpy()[order] * (m - np.arange(m))
    adjusted = np.minimum(np.maximum.accumulate(stepped), 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return pd.Series(out, index=p.index)
