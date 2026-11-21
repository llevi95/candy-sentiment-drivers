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
from scipy import stats

INTERCEPT = "intercept"

Covariance = Literal["classical", "hc3"]


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
) -> FitResult:
    """Fit y on X by ordinary least squares, with an intercept added first.

    ``cov`` selects the coefficient covariance estimator: ``"classical"`` or ``"hc3"``.
    Tests and intervals use the t distribution with n - k degrees of freedom.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    design = np.column_stack([np.ones(n), X])
    terms = [INTERCEPT, *names]
    k = design.shape[1]
    df_resid = n - k

    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
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
