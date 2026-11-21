"""End-to-end analysis: raw data in, results bundle and figures out."""

from __future__ import annotations

import json
from pathlib import Path

from candy.data import PREDICTORS, load_candies
from candy.estimators import FitResult, fit
from candy.figures import coefficient_chart

BUNDLE_NAME = "results.json"
FIGURES_DIR = "figures"


def run(raw_csv: Path, out_dir: Path) -> dict:
    """Run the full analysis and write the results bundle into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_candies(raw_csv)
    candies = data.candies
    main = fit(
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
        "main_model": _model_section(main, target="win_rate"),
    }
    coefficient_chart(main.table(), out_dir / FIGURES_DIR / "coefficients.png")

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
        "coefficients": table.to_dict(orient="records"),
    }
