"""End-to-end analysis: raw data in, results bundle and figures out."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from candy.data import ATTRIBUTES, MATCHUPS, PREDICTORS, Interaction, load_candies
from candy.labels import LABELS, WIN_RATE_MEANING
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
    ABOVE,
    BELOW,
    REALISM_RULES,
    SUPPORT_DISTANCE,
    ProfileScorer,
    all_profiles,
    choose_levels,
    compare_cookie_and_gummy,
    evidence_tiers,
    is_realistic,
    is_supported,
    mark_ties,
    rank,
    recommend,
)
from candy.figures import (
    attribute_comparison_chart,
    coefficient_chart,
    cookie_vs_gummy_chart,
    ranked_profiles_chart,
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

# The largest brand family among the top candies. Its candies share brand
# equity the attributes cannot see, so the model is refitted without them.
BRAND_CHECK = "Reese's"


# Chosen on product grounds (chocolate with its most common partners) and
# tested one at a time on the main model.
INTERACTIONS = (
    Interaction("chocolate", "peanuty_almondy"),
    Interaction("chocolate", "crisped_rice_wafer"),
    Interaction("chocolate", "bar"),
)
ESTIMATED = "estimated"
NOT_ESTIMABLE = "not_estimable"
RETENTION_RULE = (
    "Retain an interaction only if its {level} CI excludes 0 and leave-one-out "
    "R² does not worsen relative to the main model."
)


def _level() -> str:
    """The confidence level behind every interval, e.g. '95%'."""
    return f"{1 - ALPHA:.0%}"


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
    brand = candies["candy"].str.startswith(BRAND_CHECK)
    without_brand = _fit_model(candies.loc[~brand])
    # Gummies flagged as judgement calls are counted as not gummy.
    is_clear_gummy = (candies["gummy"] == 1) & (candies["gummy_unclear"] == 0)
    clear_gummies = candies.assign(gummy=is_clear_gummy.astype(int))
    clear_gummy_model = _fit_model(clear_gummies, extra_attributes=("gummy",))
    levels = choose_levels(candies)
    scorer = ProfileScorer(final_model, levels, retained)
    profiles = all_profiles()
    realistic = is_realistic(profiles)
    supported = realistic & is_supported(profiles, candies)
    holm_results = _holm_table(main_model, family=PREDICTORS)
    survivors = set(holm_results.index[holm_results["survives"]])
    ranked = rank(profiles[supported], scorer)
    # Tiers come from the model that scores the profiles, so a retained
    # interaction cannot make the two disagree on what an attribute means.
    confirmed = holm_results if final_model is main_model else _holm_table(final_model, PREDICTORS)
    recommendation = recommend(
        ranked,
        candies,
        evidence_tiers(final_model, confirmed=set(confirmed.index[confirmed["survives"]])),
    )
    ranked = mark_ties(ranked, scorer, recommendation["primary"]["attributes"])
    chocolate_vs_fruity = _chocolate_vs_fruity(candies, scorer)
    gummy_scorer = ProfileScorer(gummy_model, levels, (), extra_attributes=("gummy",))
    cookie_vs_gummy = compare_cookie_and_gummy(gummy_scorer)
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
        "chocolate_vs_fruity": chocolate_vs_fruity,
        "attribute_comparison": {
            "test": "welch",
            "alpha": ALPHA,
            "results": _records(comparison, "attribute"),
        },
        "interactions": {
            "declared": [i.term for i in INTERACTIONS],
            "declared_note": "Chosen on product grounds: chocolate with its most common partners.",
            "rule": RETENTION_RULE.format(level=_level()),
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
            "levels_rule": "Sugar and price level fixed at their medians.",
            "range_note": "CI and prediction interval bounds are capped to 0-100; "
            "'capped' marks profiles where the linear model's range left that scale.",
            "n_enumerated": len(profiles),
            "n_realistic": int(realistic.sum()),
            "n_supported": int(supported.sum()),
            "n_tied_with_primary": sum(row["tied_with_primary"] for row in ranked),
            "n_above_primary": sum(row["vs_primary"] == ABOVE for row in ranked),
            "n_below_primary": sum(row["vs_primary"] == BELOW for row in ranked),
            "tie_note": f"A profile is tied with the primary profile when the {_level()} CI "
            "of the predicted gap between the two includes zero; otherwise it is above "
            "or below it. The intervals are not corrected for multiple testing, and a "
            "gap made of more attributes has a wider interval, so a profile can be "
            "above the primary one while higher-ranked profiles are tied with it. "
            "Above is therefore weaker evidence than a driver.",
            "ranked": ranked,
        },
        "recommendation": recommendation,
        "cookie_vs_gummy": {
            "note": "Predicted with the model that includes the gummy attribute, at "
            "the same sugar and price level as the profiles. Every cookie candy in "
            "the data is chocolate, so a non-chocolate cookie cannot be assessed. "
            "These scores come from the gummy model, so the chocolate score differs "
            "slightly from the primary profile's, which uses the main model; compare "
            "products within this section only.",
            **cookie_vs_gummy,
        },
        "robustness": {
            "gummy": {
                "n_gummy": int(candies["gummy"].sum()),
                "n_unclear": int(candies["gummy_unclear"].sum()),
                "n_unclear_gummy": int((candies["gummy"] * candies["gummy_unclear"]).sum()),
                "unclear_candies": candies.loc[candies["gummy_unclear"] == 1, "candy"].tolist(),
                **_model_section(gummy_model, target="win_rate"),
                "loo_r_squared_change": gummy_model.loo_r_squared - main_model.loo_r_squared,
                "clear_cases_only": {
                    "note": "Gummy judgement calls recoded as not gummy; "
                    "every other input unchanged.",
                    "n_gummy": int(clear_gummies["gummy"].sum()),
                    **_term_row(clear_gummy_model, "gummy"),
                },
            },
            "without_brand": {
                "brand": BRAND_CHECK,
                "excluded": candies.loc[brand, "candy"].tolist(),
                **_model_section(without_brand, target="win_rate"),
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
    coefficient_chart(
        main_model.table(),
        survivors,
        ALPHA,
        _reference_group_note(chocolate_vs_fruity),
        figures / "coefficients.png",
    )
    attribute_comparison_chart(comparison, figures / "attribute_comparison.png")
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

    ranked_profiles_chart(
        ranked,
        recommendation["primary"],
        RANKED_FIGURE_TOP,
        ALPHA,
        figures / "ranked_profiles.png",
    )
    cookie_vs_gummy_chart(cookie_vs_gummy, ALPHA, figures / "cookie_vs_gummy.png")

    _write_appendix(out_dir / APPENDIX_DIR, bundle, gummy_csv)
    (out_dir / BUNDLE_NAME).write_text(json.dumps(bundle, indent=2) + "\n")
    return bundle


def _chocolate_vs_fruity(candies: pd.DataFrame, scorer: ProfileScorer) -> dict:
    """Chocolate minus fruity, all else equal, with its CI.

    Nearly every candy is chocolate or fruity, so each coefficient alone
    compares with the few candies that are neither; this is the direct
    comparison of the two product types.
    """
    neither = (candies["chocolate"] == 0) & (candies["fruity"] == 0)
    blank = dict.fromkeys(ATTRIBUTES, 0)
    gap = scorer.gap(blank | {"chocolate": 1}, blank | {"fruity": 1})
    return {
        "note": "The chocolate and fruity coefficients each compare with the "
        f"{int(neither.sum())} candies that are neither, so a positive fruity "
        "coefficient does not mean fruit flavour raises preference. This is the "
        "direct comparison of the two, all else equal.",
        "n_neither": int(neither.sum()),
        "candies_neither": candies.loc[neither, "candy"].tolist(),
        "estimate": gap.estimate,
        "ci_low": gap.ci_low,
        "ci_high": gap.ci_high,
    }


def _reference_group_note(contrast: dict) -> str:
    return (
        f"Chocolate and fruity are each measured against the {contrast['n_neither']} "
        f"candies that are neither. Chocolate beats fruity by "
        f"{contrast['estimate']:.0f} points (CI {contrast['ci_low']:.0f} to "
        f"{contrast['ci_high']:.0f})."
    )


def _data_notes(bundle: dict) -> list[str]:
    """Plain-language notes on the data for the deck appendix."""
    sample = bundle["sample"]
    counts = {row["attribute"]: row["n_with"] for row in bundle["attribute_comparison"]["results"]}
    gummy = bundle["robustness"]["gummy"]
    coins = bundle["robustness"]["coins_included"]
    return [
        f"{sample['n_raw']} market entries in the source data; the "
        f"{len(sample['excluded'])} coins "
        f"({' and '.join(sample['excluded'])}) are survey decoys, not candies, and are "
        f"excluded, leaving {sample['n_candies']} candies.",
        "Including the coins moves no attribute or level coefficient by more than "
        f"{coins['max_abs_shift_predictors']:.1f} points of win rate.",
        f"Win rate is the share of {MATCHUPS} head-to-head matchups a candy won. "
        f"{WIN_RATE_MEANING}",
        "Sugar level and price level are percentile ranks within the dataset (0 to 1), "
        "not grams of sugar or prices.",
        "The source column 'peanutyalmondy' (spelled 'peanutalmondy' in the source "
        "documentation) is analysed as the peanut/almond attribute.",
        f"Only {counts['nougat']} candies have nougat and {counts['crisped_rice_wafer']} "
        "have crisped rice/wafer/cookie, so those effects carry wide intervals.",
        f"Gummy is a hand-coded attribute ({gummy['n_gummy']} of {sample['n_candies']} "
        "candies), used to test the gummy effect and to compare a gummy with "
        f"chocolate products. Judgement calls are flagged: {gummy['n_unclear_gummy']} "
        f"of the {gummy['n_gummy']} gummies and "
        f"{gummy['n_unclear'] - gummy['n_unclear_gummy']} of the other candies.",
        _gummy_sensitivity_note(gummy),
        _brand_note(bundle["main_model"], bundle["robustness"]["without_brand"]),
    ]


def _brand_note(main: dict, check: dict) -> str:
    """Which effects with a CI excluding zero keep it without the brand family."""
    before = {row["term"]: row for row in main["coefficients"]}
    after = {row["term"]: row for row in check["coefficients"]}

    def clear(row: dict) -> bool:
        return row["ci_low"] > 0 or row["ci_high"] < 0

    def effect(term: str) -> str:
        row = after[term]
        return (
            f"{LABELS[term].split(' (')[0].lower()} {row['coef']:+.1f} "
            f"(CI {row['ci_low']:.1f} to {row['ci_high']:.1f})"
        )

    tested = [term for term in PREDICTORS if clear(before[term])]
    kept = [effect(t) for t in tested if clear(after[t])]
    lost = [effect(t) for t in tested if not clear(after[t])]
    note = (
        f"Without the {len(check['excluded'])} {check['brand']} candies, these effects "
        f"stay clear: {', '.join(kept) or 'none'}."
    )
    if lost:
        note += f" These no longer are: {', '.join(lost)}."
    return note


def _gummy_sensitivity_note(gummy: dict) -> str:
    """One sentence on how the gummy effect moves without the judgement calls."""
    full = next(row for row in gummy["coefficients"] if row["term"] == "gummy")
    clear = gummy["clear_cases_only"]

    def effect(row: dict) -> str:
        return (
            f"{row['coef']:+.1f} points ({_level()} CI {row['ci_low']:.1f} to {row['ci_high']:.1f})"
        )

    verdict = (
        "its interval still excludes zero"
        if clear["ci_low"] > 0 or clear["ci_high"] < 0
        else "its interval then includes zero, so the gummy effect depends on the judgement calls"
    )
    return (
        f"Gummy effect with all {gummy['n_gummy']} gummies: {effect(full)}; with "
        f"only the {clear['n_gummy']} clear cases: {effect(clear)}, and {verdict}."
    )


def _write_appendix(folder: Path, bundle: dict, gummy_csv: Path) -> None:
    """Appendix tables for the deck: VIFs, the candies behind each interaction,
    the gummy coding and the data notes."""
    folder.mkdir(parents=True, exist_ok=True)
    vif = bundle["main_model"]["vif"]
    pd.DataFrame(
        {
            "predictor": list(vif),
            "label": [LABELS.get(term, term) for term in vif],
            "vif": list(vif.values()),
        }
    ).to_csv(folder / "vif.csv", index=False)
    pd.DataFrame(
        [
            {
                "interaction": f"{LABELS[i.attribute]} x {LABELS[i.partner]}",
                "n_both": row["n_both"],
                "n_partner_only": row[f"n_partner_without_{i.attribute}"],
                "n_attribute_only": row[f"n_{i.attribute}_without_partner"],
                "status": row["status"],
            }
            for i, row in zip(INTERACTIONS, bundle["interactions"]["results"], strict=True)
        ]
    ).to_csv(folder / "interaction_support.csv", index=False)
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
        alpha=ALPHA,
        target=target,
        bootstrap_draws=bootstrap_draws,
        seed=seed,
    )


def _test_interaction(
    candies: pd.DataFrame, interaction: Interaction, main_model: FitResult
) -> dict:
    """Add one interaction to the main model and apply the retention rule.

    Support counts say how many candies identify the term: those with both
    attributes, and those with only one of the two. The term can be told
    apart from the partner's own effect only by candies that have the partner
    without the main attribute.
    """
    term = interaction.term
    has_attribute = candies[interaction.attribute] == 1
    has_partner = candies[interaction.partner] == 1
    support = {
        "n_both": int((has_attribute & has_partner).sum()),
        f"n_partner_without_{interaction.attribute}": int((~has_attribute & has_partner).sum()),
        f"n_{interaction.attribute}_without_partner": int((has_attribute & ~has_partner).sum()),
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


def _term_row(result: FitResult, term: str) -> dict:
    """One term's estimate, standard error, p-value and CI."""
    row = result.table().loc[term]
    return {key: float(row[key]) for key in ("coef", "se", "p", "ci_low", "ci_high")}


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
