"""End-to-end analysis: raw data in, results bundle and figures out."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from candy.data import ATTRIBUTES, PREDICTORS, load_candies
from candy.estimators import FitResult, fit, holm
from candy.figures import attribute_comparison_chart, coefficient_chart

BUNDLE_NAME = "results.json"
ALPHA = 0.05
FIGURES_DIR = "figures"


def run(raw_csv: Path, out_dir: Path) -> dict:
    """Run the full analysis and write the results bundle into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_candies(raw_csv)
    candies = data.candies
    main_model = fit(
        candies[PREDICTORS].to_numpy(),
        candies["win_rate"].to_numpy(),
        PREDICTORS,
        cov="hc3",
    )

    bundle = {
        "sample": {
            "n_raw": data.n_raw,
            "n_candies": len(data.candies),
            "excluded": data.excluded,
        },
        "main_model": _model_section(main_model, target="win_rate"),
        "holm": _holm_section(main_model, family=PREDICTORS),
        "attribute_comparison": _attribute_comparison_section(candies),
    }
    figures = out_dir / FIGURES_DIR
    survivors = {row["term"] for row in bundle["holm"]["results"] if row["survives"]}
    coefficient_chart(main_model.table(), survivors, figures / "coefficients.png")
    comparison = pd.DataFrame(bundle["attribute_comparison"]["results"]).set_index("attribute")
    attribute_comparison_chart(comparison, figures / "attribute_comparison.png")

    (out_dir / BUNDLE_NAME).write_text(json.dumps(bundle, indent=2) + "\n")
    return bundle


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


def _holm_section(result: FitResult, family: list[str]) -> dict:
    p = result.p[family]
    adjusted = holm(p)
    return {
        "alpha": ALPHA,
        "results": [
            {
                "term": term,
                "p": float(p[term]),
                "p_holm": float(adjusted[term]),
                "survives": bool(adjusted[term] < ALPHA),
            }
            for term in family
        ],
    }


def _attribute_comparison_section(candies: pd.DataFrame) -> dict:
    """Welch t-test of win rate, candies with vs without each attribute."""
    results = []
    for attribute in ATTRIBUTES:
        has = candies[attribute] == 1
        with_, without = candies.loc[has, "win_rate"], candies.loc[~has, "win_rate"]
        test = stats.ttest_ind(with_, without, equal_var=False)
        ci = test.confidence_interval(confidence_level=1 - ALPHA)
        results.append(
            {
                "attribute": attribute,
                "n_with": int(has.sum()),
                "n_without": int((~has).sum()),
                "mean_with": float(with_.mean()),
                "mean_without": float(without.mean()),
                "diff": float(with_.mean() - without.mean()),
                "ci_low": float(ci.low),
                "ci_high": float(ci.high),
                "t": float(test.statistic),
                "df": float(test.df),
                "p": float(test.pvalue),
            }
        )
    return {"test": "welch", "alpha": ALPHA, "results": results}
