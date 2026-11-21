"""Seam 2: the whole pipeline, from raw market research data to the results bundle."""

import json
from pathlib import Path

import pytest

from candy.pipeline import run

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "candy-data.csv"


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("outputs")
    run(raw_csv=RAW, out_dir=out)
    return out


@pytest.fixture(scope="module")
def bundle(out_dir) -> dict:
    return json.loads((out_dir / "results.json").read_text())


def test_coins_are_excluded_and_83_candies_analysed(bundle):
    assert bundle["sample"]["n_raw"] == 85
    assert bundle["sample"]["n_candies"] == 83
    assert sorted(bundle["sample"]["excluded"]) == ["One dime", "One quarter"]


def test_malformed_attribute_values_are_rejected(tmp_path):
    corrupted = tmp_path / "bad.csv"
    corrupted.write_text(RAW.read_text().replace("100 Grand,1,", "100 Grand,2,", 1))

    with pytest.raises(ValueError, match="0/1"):
        run(raw_csv=corrupted, out_dir=tmp_path / "out")


MAIN_TERMS = [
    "intercept",
    "chocolate",
    "fruity",
    "caramel",
    "peanuty_almondy",
    "nougat",
    "crisped_rice_wafer",
    "hard",
    "bar",
    "pluribus",
    "sugar_level",
    "price_level",
]


def test_main_model_regresses_win_rate_on_attributes_and_levels_with_hc3(bundle):
    main = bundle["main_model"]

    assert main["target"] == "win_rate"
    assert main["cov"] == "hc3"
    assert main["n_obs"] == 83
    assert main["df_resid"] == 71
    assert [row["term"] for row in main["coefficients"]] == MAIN_TERMS


def test_main_model_matches_reference_estimates(bundle):
    # Literals from an external reference OLS implementation on the same data.
    rows = {row["term"]: row for row in bundle["main_model"]["coefficients"]}

    assert rows["chocolate"]["coef"] == pytest.approx(21.214083194, rel=1e-8)
    assert rows["chocolate"]["se"] == pytest.approx(4.6334798647, rel=1e-8)
    assert rows["price_level"]["coef"] == pytest.approx(-7.2808321692, rel=1e-8)
    assert rows["price_level"]["se"] == pytest.approx(6.4230212457, rel=1e-8)
    assert bundle["main_model"]["r_squared"] == pytest.approx(0.5455116138, rel=1e-8)


def test_every_coefficient_lies_inside_its_confidence_interval(bundle):
    for row in bundle["main_model"]["coefficients"]:
        assert row["ci_low"] < row["coef"] < row["ci_high"]
