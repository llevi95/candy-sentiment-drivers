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
    # Computed once, outside this project, with statsmodels 0.15.0 on the same
    # 83 candies: sm.OLS(win, sm.add_constant(X)).fit(cov_type="HC3", use_t=True)
    rows = {row["term"]: row for row in bundle["main_model"]["coefficients"]}

    assert rows["chocolate"]["coef"] == pytest.approx(21.214083194, rel=1e-8)
    assert rows["chocolate"]["se"] == pytest.approx(4.6334798647, rel=1e-8)
    assert rows["price_level"]["coef"] == pytest.approx(-7.2808321692, rel=1e-8)
    assert rows["price_level"]["se"] == pytest.approx(6.4230212457, rel=1e-8)
    assert bundle["main_model"]["r_squared"] == pytest.approx(0.5455116138, rel=1e-8)


def test_every_coefficient_lies_inside_its_confidence_interval(bundle):
    for row in bundle["main_model"]["coefficients"]:
        assert row["ci_low"] < row["coef"] < row["ci_high"]


@pytest.mark.parametrize("name", ["coefficients.png", "attribute_comparison.png"])
def test_figures_are_written_as_slide_ready_pngs(out_dir, name):
    header = (out_dir / "figures" / name).read_bytes()[:24]

    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    assert width >= 1600
    assert width / height == pytest.approx(16 / 9, rel=0.05)


def test_module_entry_point_builds_outputs(tmp_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "candy", "--raw", str(RAW), "--out", str(tmp_path)],
        check=True,
    )

    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "figures" / "coefficients.png").exists()


def test_main_model_reports_a_vif_for_every_predictor(bundle):
    vif = bundle["main_model"]["vif"]

    assert list(vif) == MAIN_TERMS[1:]
    assert all(value >= 1 for value in vif.values())


def test_main_model_reports_an_out_of_sample_r_squared_below_in_sample(bundle):
    main = bundle["main_model"]

    assert 0 < main["loo_r_squared"] < main["r_squared"]


def test_holm_correction_covers_the_eleven_main_effects(bundle):
    holm = bundle["holm"]
    main_p = {row["term"]: row["p"] for row in bundle["main_model"]["coefficients"]}

    assert holm["alpha"] == 0.05
    assert [row["term"] for row in holm["results"]] == MAIN_TERMS[1:]
    for row in holm["results"]:
        assert row["p"] == main_p[row["term"]]
        assert row["p"] <= row["p_holm"] <= 1
        assert row["survives"] == (row["p_holm"] < holm["alpha"])


def test_chocolate_survives_holm_correction(bundle):
    # Chocolate's HC3 p-value is about 2e-5, far below 0.05 / 11.
    survivors = [row["term"] for row in bundle["holm"]["results"] if row["survives"]]

    assert "chocolate" in survivors


ATTRIBUTES = MAIN_TERMS[1:10]


def test_attribute_comparison_splits_all_candies_with_and_without_each_attribute(bundle):
    rows = bundle["attribute_comparison"]["results"]

    assert bundle["attribute_comparison"]["test"] == "welch"
    assert [row["attribute"] for row in rows] == ATTRIBUTES
    for row in rows:
        assert row["n_with"] + row["n_without"] == 83
        assert row["diff"] == pytest.approx(row["mean_with"] - row["mean_without"])
        assert row["ci_low"] < row["diff"] < row["ci_high"]
        assert 0 <= row["p"] <= 1


def test_attribute_comparison_counts_match_the_data(bundle):
    # Counts in the source data (coins carry no attributes, so none are lost).
    counts = {row["attribute"]: row["n_with"] for row in bundle["attribute_comparison"]["results"]}

    assert counts["chocolate"] == 37
    assert counts["fruity"] == 38
    assert counts["nougat"] == 7
    assert counts["crisped_rice_wafer"] == 7
