"""End-to-end analysis: raw data in, results bundle and figures out."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from candy.data import ATTRIBUTES, PREDICTORS, Interaction, load_candies
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
    ProfileScorer,
    all_profiles,
    choose_levels,
    is_realistic,
    is_supported,
    rank,
    recommend,
)
from candy.figures import (
    LABELS,
    attribute_comparison_chart,
    coefficient_chart,
    ranked_profiles_chart,
    recommendation_gap_chart,
    robustness_chart,
    win_rate_distribution_chart,
)

BUNDLE_NAME = "results.json"
ALPHA = 0.05
SEED = 20261121
BOOTSTRAP_DRAWS = 2000

FIGURES_DIR = "figures"
APPENDIX_DIR = "appendix"
RANKED_FIGURE_TOP = 15


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
    levels, sugar_rule = choose_levels(candies, final_model)
    scorer = ProfileScorer(final_model, levels, retained)
    profiles = all_profiles()
    realistic = is_realistic(profiles)
    supported = realistic & is_supported(profiles, candies)
    ranked = rank(profiles[supported], scorer)
    recommendation = recommend(ranked, scorer, candies)
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
            "realism_rules": [rule.label for rule in REALISM_RULES],
            "support_rule": f"within {SUPPORT_DISTANCE} attribute of at least one real candy",
            "levels": levels,
            "sugar_rule": sugar_rule,
            "range_note": "CI and prediction interval bounds are capped to 0-100; "
            "'capped' marks profiles where the linear model's range left that scale.",
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

    bundle["data_notes"] = _data_notes(bundle)

    figures = out_dir / FIGURES_DIR
    win_rate_distribution_chart(candies, figures / "win_rate_distribution.png")
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

    _write_appendix(out_dir / APPENDIX_DIR, bundle, gummy_csv)
    (out_dir / BUNDLE_NAME).write_text(json.dumps(bundle, indent=2) + "\n")
    return bundle


def _data_notes(bundle: dict) -> list[str]:
    """Plain-language notes on the data for the deck appendix."""
    sample = bundle["sample"]
    counts = {
        row["attribute"]: row["n_with"] for row in bundle["attribute_comparison"]["results"]
    }
    gummy = bundle["robustness"]["gummy"]
    coins = bundle["robustness"]["coins_included"]
    return [
        f"{sample['n_raw']} market entries in the source data; the two coins "
        f"({' and '.join(sample['excluded'])}) are survey decoys, not candies, and are "
        f"excluded, leaving {sample['n_candies']} candies.",
        "Including the coins moves no attribute or level coefficient by more than "
        f"{coins['max_abs_shift_predictors']:.1f} points of win rate.",
        "Win rate is the share of roughly 269,000 head-to-head matchups a candy won. "
        "It measures relative preference, not sales or satisfaction.",
        "Sugar level and price level are percentile ranks within the dataset (0 to 1), "
        "not grams of sugar or prices.",
        "The source column 'peanutyalmondy' (spelled 'peanutalmondy' in the source "
        "documentation) is analysed as the peanut/almond attribute.",
        f"Only {counts['nougat']} candies have nougat and {counts['crisped_rice_wafer']} "
        "have crisped rice/wafer/cookie, so those effects carry wide intervals.",
        f"Gummy is a hand-coded attribute ({gummy['n_gummy']} of {sample['n_candies']} "
        f"candies; {gummy['n_unclear']} judgement calls flagged) used only as a "
        "supplementary check.",
    ]


def _write_appendix(folder: Path, bundle: dict, gummy_csv: Path) -> None:
    """Appendix tables for the deck: VIFs, the gummy coding and the data notes."""
    folder.mkdir(parents=True, exist_ok=True)
    vif = bundle["main_model"]["vif"]
    pd.DataFrame(
        {
            "predictor": list(vif),
            "label": [LABELS.get(term, term) for term in vif],
            "vif": list(vif.values()),
        }
    ).to_csv(folder / "vif.csv", index=False)
    pd.read_csv(gummy_csv).to_csv(folder / "gummy_coding.csv", index=False)
    (folder / "data_notes.md").write_text(
        "\n".join(f"- {note}" for note in bundle["data_notes"]) + "\n"
    )


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
        design[interaction.term] = interaction.values(entries)
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
