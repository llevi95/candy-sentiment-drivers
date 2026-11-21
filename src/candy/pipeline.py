"""End-to-end analysis: raw data in, results bundle and figures out."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy import stats

from candy.data import ATTRIBUTES, PREDICTORS, load_candies
from candy.estimators import (
    INTERCEPT,
    Bootstrap,
    EstimationError,
    FitResult,
    Target,
    fit,
    holm,
)
from candy.profiles import (
    REALISM_RULES,
    SUPPORT_DISTANCE,
    all_profiles,
    benchmarks,
    is_realistic,
    is_supported,
)
from candy.figures import (
    attribute_comparison_chart,
    coefficient_chart,
    ranked_profiles_chart,
    recommendation_gap_chart,
    robustness_chart,
)

BUNDLE_NAME = "results.json"
ALPHA = 0.05
SEED = 20261121
BOOTSTRAP_DRAWS = 2000

FIGURES_DIR = "figures"
RANKED_FIGURE_TOP = 15


@dataclass(frozen=True)
class Interaction:
    """Product of two attributes, added to the main model as one extra term."""

    attribute: str
    partner: str

    @property
    def term(self) -> str:
        return f"{self.attribute}_x_{self.partner}"


# Declared before any results were seen; tested one at a time on the main model.
INTERACTIONS = (
    Interaction("chocolate", "peanuty_almondy"),
    Interaction("chocolate", "crisped_rice_wafer"),
    Interaction("chocolate", "bar"),
)
ESTIMATED = "estimated"
NOT_ESTIMABLE = "not_estimable"
RETENTION_RULE = (
    "Retain an interaction only if its 95% CI excludes 0 and leave-one-out "
    "R² does not worsen relative to the main model."
)


def _is_retained(ci_low: float, ci_high: float, loo_change: float) -> bool:
    """RETENTION_RULE, as code."""
    return bool((ci_low > 0 or ci_high < 0) and loo_change >= 0)


def run(raw_csv: Path, gummy_csv: Path, out_dir: Path, seed: int = SEED) -> dict:
    """Run the full analysis and write the results bundle into ``out_dir``.

    ``seed`` drives every random step (the bootstrap), so equal seeds give
    identical bundles.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_candies(raw_csv, gummy_csv)
    candies = data.candies
    main_model = _fit_model(candies, bootstrap_draws=BOOTSTRAP_DRAWS, seed=seed)
    coins_included = _fit_model(data.with_coins)
    logit_model = _fit_model(candies, target="logit")
    coin_shift = coins_included.coef - main_model.coef

    interactions = [_test_interaction(candies, i, main_model) for i in INTERACTIONS]
    retained = tuple(
        i for i, row in zip(INTERACTIONS, interactions, strict=True) if row["retained"]
    )
    final_model = _fit_model(candies, interactions=retained) if retained else main_model
    gummy_model = _fit_model(candies, extra_attributes=("gummy",))
    levels, sugar_rule = _profile_levels(candies, final_model)
    profiles = all_profiles()
    realistic = is_realistic(profiles)
    supported = realistic & is_supported(profiles, candies)
    ranked = _rank_profiles(profiles[supported], levels, retained, final_model)
    recommendation = _recommend(ranked, levels, retained, final_model, candies)
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
        "interactions": {
            "declared": [i.term for i in INTERACTIONS],
            "declared_note": "Fixed before any model results were seen.",
            "rule": RETENTION_RULE,
            "results": interactions,
        },
        "final_model": {
            "predictors": list(final_model.coef.index.drop(INTERCEPT)),
            **_model_section(final_model, target="win_rate"),
        },
        "profiles": {
            "realism_rules": list(REALISM_RULES),
            "support_rule": f"within {SUPPORT_DISTANCE} attribute of at least one real candy",
            "levels": levels,
            "sugar_rule": sugar_rule,
            "n_enumerated": len(profiles),
            "n_realistic": int(realistic.sum()),
            "n_supported": int(supported.sum()),
            "ranked": ranked,
        },
        "recommendation": recommendation,
        "robustness": {
            "gummy": {
                "n_gummy": int(candies["gummy"].sum()),
                "n_unclear": int(candies["gummy_unclear"].sum()),
                "unclear_candies": candies.loc[candies["gummy_unclear"] == 1, "candy"].tolist(),
                **_model_section(gummy_model, target="win_rate"),
                "loo_r_squared_change": gummy_model.loo_r_squared - main_model.loo_r_squared,
            },
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

    ranked_profiles_chart(ranked, RANKED_FIGURE_TOP, ALPHA, figures / "ranked_profiles.png")
    recommendation_gap_chart(
        recommendation["primary"],
        recommendation["fruit_flavoured"],
        recommendation["gap"],
        ALPHA,
        figures / "recommendation_gap.png",
    )

    (out_dir / BUNDLE_NAME).write_text(json.dumps(bundle, indent=2) + "\n")
    return bundle


def _profile_levels(candies: pd.DataFrame, model: FitResult) -> tuple[dict, str]:
    """Sugar and price level at which every profile is predicted.

    Price is a control, fixed at the median. Sugar sits at the median unless
    its CI excludes zero, in which case it moves to the favourable quartile.
    """
    sugar = candies["sugar_level"]
    low, high = model.ci_low["sugar_level"], model.ci_high["sugar_level"]
    if low > 0:
        sugar_value, rule = sugar.quantile(0.75), "upper quartile (sugar CI above zero)"
    elif high < 0:
        sugar_value, rule = sugar.quantile(0.25), "lower quartile (sugar CI below zero)"
    else:
        sugar_value, rule = sugar.median(), "median (sugar CI includes zero)"
    levels = {
        "sugar_level": float(sugar_value),
        "price_level": float(candies["price_level"].median()),
    }
    return levels, rule


def _profile_row(
    profile: pd.Series, levels: dict, interactions: tuple[Interaction, ...]
) -> dict[str, float]:
    row = {attribute: int(profile[attribute]) for attribute in ATTRIBUTES} | levels
    for interaction in interactions:
        row[interaction.term] = row[interaction.attribute] * row[interaction.partner]
    return row


def _rank_profiles(
    profiles: pd.DataFrame,
    levels: dict,
    interactions: tuple[Interaction, ...],
    model: FitResult,
) -> list[dict]:
    """Predict every profile with the final model, best first."""
    rows = []
    for _, profile in profiles.iterrows():
        prediction = model.predict(_profile_row(profile, levels, interactions))
        rows.append(
            {
                "attributes": {a: int(profile[a]) for a in ATTRIBUTES},
                "estimate": prediction.estimate,
                "ci_low": prediction.ci_low,
                "ci_high": prediction.ci_high,
                "pi_low": prediction.pi_low,
                "pi_high": prediction.pi_high,
            }
        )
    rows.sort(key=lambda row: row["estimate"], reverse=True)
    return [{"rank": i + 1, **row} for i, row in enumerate(rows)]


def _recommend(
    ranked: list[dict],
    levels: dict,
    interactions: tuple[Interaction, ...],
    model: FitResult,
    candies: pd.DataFrame,
) -> dict:
    """Primary and fruit-flavoured profiles, their gap, and benchmark candies."""
    primary = ranked[0]
    fruit = next(row for row in ranked if row["attributes"]["fruity"] == 1)

    def with_benchmarks(row: dict) -> dict:
        profile = pd.Series(row["attributes"])
        return {**row, "benchmarks": benchmarks(profile, levels, candies)}

    def design(row: dict) -> dict[str, float]:
        return _profile_row(pd.Series(row["attributes"]), levels, interactions)

    gap = model.predict_difference(design(primary), design(fruit))
    return {
        "primary": with_benchmarks(primary),
        "fruit_flavoured": with_benchmarks(fruit),
        "gap": {"estimate": gap.estimate, "ci_low": gap.ci_low, "ci_high": gap.ci_high},
    }


def _fit_model(
    entries: pd.DataFrame,
    *,
    interactions: tuple[Interaction, ...] = (),
    extra_attributes: tuple[str, ...] = (),
    target: Target = "identity",
    bootstrap_draws: int | None = None,
    seed: int | None = None,
) -> FitResult:
    """Win rate on the attributes plus sugar and price level, HC3: the main
    model, optionally extended with extra attributes and interaction terms."""
    design = entries[[*PREDICTORS, *extra_attributes]].copy()
    for interaction in interactions:
        design[interaction.term] = entries[interaction.attribute] * entries[interaction.partner]
    return fit(
        design.to_numpy(),
        entries["win_rate"].to_numpy(),
        list(design.columns),
        cov="hc3",
        target=target,
        bootstrap_draws=bootstrap_draws,
        seed=seed,
    )


def _test_interaction(
    candies: pd.DataFrame, interaction: Interaction, main_model: FitResult
) -> dict:
    """Add one interaction to the main model and apply the retention rule.

    Support counts say how many candies identify the term: those with both
    attributes, and those with the partner but not the main attribute.
    """
    term = interaction.term
    has_attribute = candies[interaction.attribute] == 1
    has_partner = candies[interaction.partner] == 1
    support = {
        "n_both": int((has_attribute & has_partner).sum()),
        f"n_partner_without_{interaction.attribute}": int((~has_attribute & has_partner).sum()),
    }
    try:
        model = _fit_model(candies, interactions=(interaction,))
    except EstimationError as error:
        return {
            "term": term,
            "status": NOT_ESTIMABLE,
            **support,
            "reason": str(error),
            "retained": False,
        }
    ci_low, ci_high = model.ci_low[term], model.ci_high[term]
    loo_change = model.loo_r_squared - main_model.loo_r_squared
    return {
        "term": term,
        "status": ESTIMATED,
        **support,
        "coef": float(model.coef[term]),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p": float(model.p[term]),
        "loo_r_squared": model.loo_r_squared,
        "loo_r_squared_change": loo_change,
        "retained": _is_retained(ci_low, ci_high, loo_change),
    }


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
