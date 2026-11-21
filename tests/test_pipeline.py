"""Seam 2: the whole pipeline, from raw market research data to the results bundle."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from candy.pipeline import run

DATA = Path(__file__).resolve().parents[1] / "data"
RAW = DATA / "raw" / "candy-data.csv"
GUMMY = DATA / "derived" / "gummy.csv"


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("outputs")
    run(raw_csv=RAW, gummy_csv=GUMMY, out_dir=out)
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
        run(raw_csv=corrupted, gummy_csv=GUMMY, out_dir=tmp_path / "out")


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
ATTRIBUTES = [
    "chocolate",
    "fruity",
    "caramel",
    "peanuty_almondy",
    "nougat",
    "crisped_rice_wafer",
    "hard",
    "bar",
    "pluribus",
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


@pytest.mark.parametrize(
    "name",
    [
        "coefficients.png",
        "attribute_comparison.png",
        "robustness.png",
        "ranked_profiles.png",
        "recommendation_gap.png",
    ],
)
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
    assert (tmp_path / "figures" / "attribute_comparison.png").exists()


def test_main_model_reports_a_vif_for_every_predictor(bundle):
    vif = bundle["main_model"]["vif"]

    assert list(vif) == MAIN_TERMS[1:]
    assert all(value >= 1 for value in vif.values())


def test_main_model_reports_an_out_of_sample_r_squared_below_in_sample(bundle):
    main = bundle["main_model"]

    assert 0 < main["loo_r_squared"] < main["r_squared"]


def test_holm_correction_covers_the_eleven_main_effects(bundle):
    section = bundle["holm"]
    main_p = {row["term"]: row["p"] for row in bundle["main_model"]["coefficients"]}

    assert section["alpha"] == 0.05
    assert [row["term"] for row in section["results"]] == MAIN_TERMS[1:]
    for row in section["results"]:
        assert row["p"] == main_p[row["term"]]
        assert row["p"] <= row["p_holm"] <= 1
        assert row["survives"] == (row["p_holm"] < section["alpha"])


def test_chocolate_survives_holm_correction(bundle):
    # Chocolate's HC3 p-value is about 2e-5, far below 0.05 / 11.
    survivors = [row["term"] for row in bundle["holm"]["results"] if row["survives"]]

    assert "chocolate" in survivors


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


def test_coins_included_model_refits_the_main_model_on_all_85_entries(bundle):
    coins = bundle["robustness"]["coins_included"]

    assert coins["n_obs"] == 85
    assert coins["cov"] == "hc3"
    assert [row["term"] for row in coins["coefficients"]] == MAIN_TERMS


def test_logit_check_reports_effects_back_in_win_rate_points(bundle):
    logit = bundle["robustness"]["logit"]
    effects = logit["effects_in_points"]

    assert logit["target"] == "logit_win_rate"
    assert logit["n_obs"] == 83
    assert [row["term"] for row in effects] == MAIN_TERMS[1:]
    for row in effects:
        assert row["ci_low"] < row["coef"] < row["ci_high"]


def test_bootstrap_check_records_its_seed_and_draws(bundle):
    boot = bundle["robustness"]["bootstrap"]

    assert boot["seed"] == bundle["seed"]
    assert boot["draws"] == 2000
    assert boot["discarded"] >= 0
    assert [row["term"] for row in boot["results"]] == MAIN_TERMS
    for row in boot["results"]:
        assert row["ci_low"] < row["ci_high"]


def test_two_runs_with_the_same_seed_produce_identical_bundles(tmp_path):
    run(raw_csv=RAW, gummy_csv=GUMMY, out_dir=tmp_path / "first", seed=7)
    run(raw_csv=RAW, gummy_csv=GUMMY, out_dir=tmp_path / "second", seed=7)
    run(raw_csv=RAW, gummy_csv=GUMMY, out_dir=tmp_path / "other", seed=8)

    first = (tmp_path / "first" / "results.json").read_bytes()
    assert (tmp_path / "second" / "results.json").read_bytes() == first
    assert (tmp_path / "other" / "results.json").read_bytes() != first


def test_coins_included_check_states_how_far_each_coefficient_moved(bundle):
    coins = bundle["robustness"]["coins_included"]
    main = {row["term"]: row["coef"] for row in bundle["main_model"]["coefficients"]}
    with_coins = {row["term"]: row["coef"] for row in coins["coefficients"]}

    shifts = coins["shift_vs_main"]
    assert list(shifts) == MAIN_TERMS
    for term in MAIN_TERMS:
        assert shifts[term] == pytest.approx(with_coins[term] - main[term])
    assert coins["max_abs_shift_predictors"] == pytest.approx(
        max(abs(shifts[term]) for term in MAIN_TERMS[1:])
    )


EXPECTED_INTERACTION_TERMS = [
    "chocolate_x_peanuty_almondy",
    "chocolate_x_crisped_rice_wafer",
    "chocolate_x_bar",
]


def test_each_predeclared_interaction_is_reported_in_declared_order(bundle):
    rows = bundle["interactions"]["results"]

    assert [row["term"] for row in rows] == EXPECTED_INTERACTION_TERMS
    assert bundle["interactions"]["declared"] == EXPECTED_INTERACTION_TERMS
    assert all(row["status"] in {"estimated", "not_estimable"} for row in rows)


def test_retention_rule_is_applied_consistently_with_the_reported_numbers(bundle):
    # Spec: retain only if the CI excludes 0 and leave-one-out R² does not worsen.
    main_loo = bundle["main_model"]["loo_r_squared"]

    for row in bundle["interactions"]["results"]:
        if row["status"] == "not_estimable":
            assert row["retained"] is False
            assert row["reason"]
            continue
        assert row["ci_low"] < row["coef"] < row["ci_high"]
        assert row["loo_r_squared_change"] == pytest.approx(row["loo_r_squared"] - main_loo)
        excludes_zero = row["ci_low"] > 0 or row["ci_high"] < 0
        assert row["retained"] == (excludes_zero and row["loo_r_squared_change"] >= 0)


def test_interactions_the_data_cannot_identify_are_reported_as_not_estimable(bundle):
    # All 7 crisped/wafer candies are chocolate, so chocolate x crisped equals
    # crisped; only one bar is not chocolate, so that candy alone identifies
    # chocolate x bar. Chocolate x peanuty has two non-chocolate candies.
    rows = {row["term"]: row for row in bundle["interactions"]["results"]}

    assert {term: row["status"] for term, row in rows.items()} == {
        "chocolate_x_peanuty_almondy": "estimated",
        "chocolate_x_crisped_rice_wafer": "not_estimable",
        "chocolate_x_bar": "not_estimable",
    }
    support = {
        term: (row["n_both"], row["n_partner_without_chocolate"]) for term, row in rows.items()
    }
    assert support == {
        "chocolate_x_peanuty_almondy": (12, 2),
        "chocolate_x_crisped_rice_wafer": (7, 0),
        "chocolate_x_bar": (20, 1),
    }


def test_no_interaction_is_retained_on_this_data(bundle):
    # Chocolate x peanuty: CI about [-19.5, 48.3] spans zero and leave-one-out
    # R² falls (0.351 -> 0.324); the other two cannot be estimated.
    retained = {row["term"]: row["retained"] for row in bundle["interactions"]["results"]}

    assert retained == dict.fromkeys(EXPECTED_INTERACTION_TERMS, False)


def test_final_model_is_the_main_model_plus_every_retained_interaction(bundle):
    retained = [row["term"] for row in bundle["interactions"]["results"] if row["retained"]]
    final = bundle["final_model"]

    assert final["predictors"] == MAIN_TERMS[1:] + retained
    assert [row["term"] for row in final["coefficients"]] == ["intercept", *final["predictors"]]
    assert final["n_obs"] == 83


def test_gummy_check_adds_the_gummy_attribute_to_the_main_model(bundle):
    check = bundle["robustness"]["gummy"]

    assert check["n_gummy"] == 12  # approved coding: 12 of 83 candies
    assert check["n_obs"] == 83
    assert [row["term"] for row in check["coefficients"]] == [*MAIN_TERMS, "gummy"]


def test_gummy_coefficient_matches_an_independent_least_squares_fit(bundle):
    # Oracle: join the two CSVs by hand and solve the normal equations with
    # NumPy, bypassing the project's loader and estimator.
    raw = pd.read_csv(RAW)
    raw = raw[~raw["competitorname"].isin(["One dime", "One quarter"])]
    gummy = pd.read_csv(GUMMY).set_index("candy")["gummy"]
    columns = ["chocolate", "fruity", "caramel", "peanutyalmondy", "nougat",
               "crispedricewafer", "hard", "bar", "pluribus", "sugarpercent", "pricepercent"]
    design = np.column_stack([
        np.ones(len(raw)),
        raw[columns].to_numpy(),
        raw["competitorname"].map(gummy).to_numpy(),
    ])
    beta, *_ = np.linalg.lstsq(design, raw["winpercent"].to_numpy(), rcond=None)

    rows = {row["term"]: row for row in bundle["robustness"]["gummy"]["coefficients"]}
    assert rows["gummy"]["coef"] == pytest.approx(beta[-1], rel=1e-10)
    assert rows["gummy"]["ci_low"] < rows["gummy"]["coef"] < rows["gummy"]["ci_high"]


def test_gummy_check_lists_the_judgement_calls_for_the_appendix(bundle):
    check = bundle["robustness"]["gummy"]

    assert check["n_unclear"] == 10
    assert "Swedish Fish" in check["unclear_candies"]
    assert "Starburst" in check["unclear_candies"]


def _corrupt_gummy(tmp_path: Path, edit) -> Path:
    lines = GUMMY.read_text().splitlines()
    path = tmp_path / "gummy.csv"
    path.write_text("\n".join(edit(lines)) + "\n")
    return path


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda lines: [l for l in lines if not l.startswith("Twix,")], "missing.*Twix"),
        (lambda lines: [*lines, "Twixx,0,0,typo"], "unknown.*Twixx"),
        (lambda lines: [l.replace("Twix,0,", "Twix,2,") for l in lines], "0/1.*Twix"),
        (lambda lines: [l.replace("Twix,0,", "Twix,,") for l in lines], "missing value.*Twix"),
        (lambda lines: [*lines, "Twix,0,0,again"], "more than once.*Twix"),
    ],
    ids=["missing candy", "unknown candy", "non-binary value", "blank value", "duplicate row"],
)
def test_gummy_table_problems_fail_loudly_naming_the_candy(tmp_path, edit, message):
    gummy = _corrupt_gummy(tmp_path, edit)

    with pytest.raises(ValueError, match=message):
        run(raw_csv=RAW, gummy_csv=gummy, out_dir=tmp_path / "out")


def _real_attribute_vectors() -> set[tuple[int, ...]]:
    raw = pd.read_csv(RAW)
    raw = raw[~raw["competitorname"].isin(["One dime", "One quarter"])]
    raw_columns = ["chocolate", "fruity", "caramel", "peanutyalmondy", "nougat",
                   "crispedricewafer", "hard", "bar", "pluribus"]
    return {tuple(int(v) for v in row) for row in raw[raw_columns].to_numpy()}


def _is_realistic(p: dict) -> bool:
    # Spec realism rules.
    return (
        not (p["chocolate"] and p["fruity"])
        and not (p["hard"] and p["bar"])
        and not (p["bar"] and p["pluribus"])
    )


def test_ranked_profiles_are_exactly_the_realistic_supported_profiles(bundle):
    import itertools

    real = _real_attribute_vectors()
    expected = set()
    for bits in itertools.product([0, 1], repeat=len(ATTRIBUTES)):
        profile = dict(zip(ATTRIBUTES, bits))
        supported = any(sum(a != b for a, b in zip(bits, r)) <= 1 for r in real)
        if _is_realistic(profile) and supported:
            expected.add(bits)

    ranked = bundle["profiles"]["ranked"]
    got = {tuple(row["attributes"][a] for a in ATTRIBUTES) for row in ranked}
    assert got == expected
    assert len(ranked) == len(expected) == bundle["profiles"]["n_supported"]
    assert bundle["profiles"]["n_enumerated"] == 2 ** len(ATTRIBUTES)


def test_primary_profile_is_the_top_ranked_and_has_the_highest_prediction(bundle):
    ranked = bundle["profiles"]["ranked"]
    primary = bundle["recommendation"]["primary"]

    assert primary["attributes"] == ranked[0]["attributes"]
    assert primary["estimate"] == max(row["estimate"] for row in ranked)
    assert [row["estimate"] for row in ranked] == sorted(
        (row["estimate"] for row in ranked), reverse=True
    )


def test_fruit_flavoured_profile_is_the_best_fruity_profile(bundle):
    ranked = bundle["profiles"]["ranked"]
    fruit = bundle["recommendation"]["fruit_flavoured"]

    assert fruit["attributes"]["fruity"] == 1
    assert fruit["estimate"] == max(r["estimate"] for r in ranked if r["attributes"]["fruity"])


def test_gap_is_the_difference_of_the_two_predictions(bundle):
    rec = bundle["recommendation"]
    gap = rec["gap"]

    assert gap["estimate"] == pytest.approx(
        rec["primary"]["estimate"] - rec["fruit_flavoured"]["estimate"]
    )
    assert gap["ci_low"] < gap["estimate"] < gap["ci_high"]


def test_recommendations_name_two_or_three_real_benchmark_candies(bundle):
    real = set(pd.read_csv(RAW)["competitorname"]) - {"One dime", "One quarter"}

    for key in ("primary", "fruit_flavoured"):
        marks = bundle["recommendation"][key]["benchmarks"]
        assert 2 <= len(marks) <= 3
        assert {m["candy"] for m in marks} <= real
        distances = [m["attributes_differing"] for m in marks]
        assert distances == sorted(distances)


def _profile_rows(bundle) -> list[dict]:
    rec = bundle["recommendation"]
    return [*bundle["profiles"]["ranked"], rec["primary"], rec["fruit_flavoured"]]


def test_prediction_intervals_contain_and_are_wider_than_mean_cis(bundle):
    # Ranges are capped at 100, so a capped upper bound may coincide.
    for row in _profile_rows(bundle):
        assert row["pi_low"] < row["ci_low"] < row["estimate"] < row["ci_high"] <= row["pi_high"]
        if not row["capped"]:
            assert row["ci_high"] < row["pi_high"]


def test_reported_ranges_stay_on_the_win_rate_scale_and_flag_capping(bundle):
    for row in _profile_rows(bundle):
        for key in ("ci_low", "ci_high", "pi_low", "pi_high"):
            assert 0 <= row[key] <= 100
        assert row["capped"] == (row["pi_high"] == 100 or row["pi_low"] == 0)
    # The primary profile's raw CI runs to about 103.
    primary = bundle["recommendation"]["primary"]
    assert primary["capped"] and primary["ci_high"] == 100


def test_recommendations_report_how_far_they_extrapolate(bundle):
    raw = pd.read_csv(RAW)
    raw = raw[~raw["competitorname"].isin(["One dime", "One quarter"])]
    raw_columns = ["chocolate", "fruity", "caramel", "peanutyalmondy", "nougat",
                   "crispedricewafer", "hard", "bar", "pluribus"]
    most_on_a_real_candy = int(raw[raw_columns].sum(axis=1).max())

    for key in ("primary", "fruit_flavoured"):
        rec = bundle["recommendation"][key]
        assert rec["n_attributes"] == sum(rec["attributes"].values())
        assert rec["max_attributes_on_a_real_candy"] == most_on_a_real_candy
        assert rec["nearest_candy_distance"] == rec["benchmarks"][0]["attributes_differing"]
        assert rec["nearest_candy_distance"] <= 1


def test_sugar_is_fixed_by_the_sign_of_its_confidence_interval(bundle):
    # Spec: median unless the sugar CI excludes 0, then the favourable quartile.
    raw = pd.read_csv(RAW)
    sugar = raw.loc[~raw["competitorname"].isin(["One dime", "One quarter"]), "sugarpercent"]
    price = raw.loc[~raw["competitorname"].isin(["One dime", "One quarter"]), "pricepercent"]
    row = next(r for r in bundle["final_model"]["coefficients"] if r["term"] == "sugar_level")
    if row["ci_low"] > 0:
        expected = sugar.quantile(0.75)
    elif row["ci_high"] < 0:
        expected = sugar.quantile(0.25)
    else:
        expected = sugar.median()

    levels = bundle["profiles"]["levels"]
    assert levels["sugar_level"] == pytest.approx(expected)
    assert levels["price_level"] == pytest.approx(price.median())


def test_benchmarks_are_the_nearest_candies_with_sugar_and_price_breaking_ties(bundle):
    raw = pd.read_csv(RAW)
    raw = raw[~raw["competitorname"].isin(["One dime", "One quarter"])]
    raw_columns = ["chocolate", "fruity", "caramel", "peanutyalmondy", "nougat",
                   "crispedricewafer", "hard", "bar", "pluribus"]
    levels = bundle["profiles"]["levels"]

    for key in ("primary", "fruit_flavoured"):
        rec = bundle["recommendation"][key]
        profile = np.array([rec["attributes"][a] for a in ATTRIBUTES])
        attr = (raw[raw_columns].to_numpy() != profile).sum(axis=1)
        level = np.hypot(raw["sugarpercent"] - levels["sugar_level"],
                         raw["pricepercent"] - levels["price_level"])
        ranking = sorted(zip(attr, level, raw["competitorname"]))
        assert [m["candy"] for m in rec["benchmarks"]] == [name for *_, name in ranking[:3]]
