"""End-to-end analysis: raw data in, results bundle and figures out."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from candy.data import ATTRIBUTES, PREDICTORS, load_candies
from candy.estimators import Bootstrap, FitResult, Target, fit, holm
from candy.figures import attribute_comparison_chart, coefficient_chart, robustness_chart

BUNDLE_NAME = "results.json"
ALPHA = 0.05
SEED = 20261121
BOOTSTRAP_DRAWS = 2000
FIGURES_DIR = "figures"


def run(raw_csv: Path, out_dir: Path, seed: int = SEED) -> dict:
    """Run the full analysis and write the results bundle into ``out_dir``.

    ``seed`` drives every random step (the bootstrap), so equal seeds give
    identical bundles.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_candies(raw_csv)
    candies = data.candies
    main_model = _fit_main(candies, bootstrap_draws=BOOTSTRAP_DRAWS, seed=seed)
    coins_included = _fit_main(data.with_coins)
    logit_model = _fit_main(candies, target="logit")
    coin_shift = coins_included.coef - main_model.coef

    holm_results = _holm_table(main_model, family=PREDICTORS)
    comparison = _attribute_comparison_table(candies)

    bundle = {
        "seed": seed,
        "sample": {
            "n_raw": data.n_raw,
            "n_candies": len(data.candies),
            "excluded": data.excluded,
        },
        "main_model": _model_section(main_model, target="win_rate"),
        "holm": {"alpha": ALPHA, "results": _records(holm_results, "term")},
        "attribute_comparison": {
            "test": "welch",
            "alpha": ALPHA,
            "results": _records(comparison, "attribute"),
        },
        "robustness": {
            "bootstrap": _bootstrap_section(main_model.bootstrap),
            "coins_included": {
                **_model_section(coins_included, target="win_rate"),
                "shift_vs_main": coin_shift.to_dict(),
                "max_abs_shift_predictors": float(coin_shift[PREDICTORS].abs().max()),
            },
            "logit": {
                **_model_section(logit_model, target="logit_win_rate"),
                "effects_in_points": _records(logit_model.effects_in_points(), "term"),
                "effects_note": (
                    "Average marginal effects in win-rate points: discrete 0-to-1 "
                    "change for attributes, mean derivative for sugar and price "
                    "level. Intervals rescale the logit-scale intervals by a fixed "
                    "factor, ignoring that factor's own uncertainty."
                ),
            },
        },
    }

    figures = out_dir / FIGURES_DIR
    survivors = set(holm_results.index[holm_results["survives"]])
    coefficient_chart(main_model.table(), survivors, ALPHA, figures / "coefficients.png")
    attribute_comparison_chart(comparison, ALPHA, figures / "attribute_comparison.png")
    hc3 = main_model.effects_in_points()
    robustness_chart(
        {
            "OLS, HC3 robust CI": hc3,
            "OLS, bootstrap CI": hc3.assign(**main_model.bootstrap.table().loc[hc3.index]),
            "Logit win rate": logit_model.effects_in_points(),
        },
        ALPHA,
        "Logit effects are average marginal effects in win-rate points; "
        "their intervals ignore uncertainty in the conversion.",
        figures / "robustness.png",
    )

    (out_dir / BUNDLE_NAME).write_text(json.dumps(bundle, indent=2) + "\n")
    return bundle


def _fit_main(
    entries: pd.DataFrame,
    *,
    target: Target = "identity",
    bootstrap_draws: int | None = None,
    seed: int | None = None,
) -> FitResult:
    """Main model: win rate on the attributes plus sugar and price level, HC3."""
    return fit(
        entries[PREDICTORS].to_numpy(),
        entries["win_rate"].to_numpy(),
        PREDICTORS,
        cov="hc3",
        target=target,
        bootstrap_draws=bootstrap_draws,
        seed=seed,
    )


def _model_section(result: FitResult, target: str) -> dict:
    table = result.table().reset_index(names="term")
    return {
        "target": target,
        "cov": result.cov,
        "n_obs": result.n_obs,
        "df_resid": result.df_resid,
        "r_squared": result.r_squared,
        "loo_r_squared": result.loo_r_squared,
        "coefficients": table.to_dict(orient="records"),
        "vif": result.vif.to_dict(),
    }


def _bootstrap_section(boot: Bootstrap) -> dict:
    return {
        "method": "case resampling, percentile intervals",
        "seed": boot.seed,
        "draws": boot.draws,
        "discarded": boot.discarded,
        "alpha": ALPHA,
        "results": _records(boot.table(), "term"),
    }


def _holm_table(result: FitResult, family: list[str]) -> pd.DataFrame:
    """Holm-adjusted p-values for one family of model terms, indexed by term."""
    p = result.p[family]
    adjusted = holm(p)
    return pd.DataFrame({"p": p, "p_holm": adjusted, "survives": adjusted < ALPHA})


def _attribute_comparison_table(candies: pd.DataFrame) -> pd.DataFrame:
    """Welch t-test of win rate, candies with vs without each attribute."""
    rows = {}
    for attribute in ATTRIBUTES:
        has = candies[attribute] == 1
        with_attr = candies.loc[has, "win_rate"]
        without_attr = candies.loc[~has, "win_rate"]
        test = stats.ttest_ind(with_attr, without_attr, equal_var=False)
        ci = test.confidence_interval(confidence_level=1 - ALPHA)
        rows[attribute] = {
            "n_with": int(has.sum()),
            "n_without": int((~has).sum()),
            "mean_with": with_attr.mean(),
            "mean_without": without_attr.mean(),
            "diff": with_attr.mean() - without_attr.mean(),
            "ci_low": ci.low,
            "ci_high": ci.high,
            "t": test.statistic,
            "df": test.df,
            "p": test.pvalue,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _records(table: pd.DataFrame, key: str) -> list[dict]:
    """JSON-ready rows, with the table's index as the ``key`` field."""
    return table.reset_index(names=key).to_dict(orient="records")
