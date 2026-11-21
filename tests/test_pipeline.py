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
        "cookie_vs_gummy.png",
        "win_rate_distribution.png",
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


def test_one_alpha_drives_model_intervals_and_labels(tmp_path, monkeypatch):
    # Changing the pipeline's ALPHA must change the model CIs too, or charts
    # labelled "90% CI" would show 95% intervals.
    import candy.pipeline as pipeline
    from scipy import stats

    monkeypatch.setattr(pipeline, "ALPHA", 0.10)
    monkeypatch.setattr(pipeline, "BOOTSTRAP_DRAWS", 50)
    bundle = pipeline.run(raw_csv=RAW, gummy_csv=GUMMY, out_dir=tmp_path)
    model = bundle["main_model"]
    crit = stats.t.ppf(0.95, model["df_resid"])

    for row in model["coefficients"]:
        assert row["ci_high"] - row["coef"] == pytest.approx(crit * row["se"])
    # No text in the bundle may still call these 95% intervals.
    assert "95%" not in json.dumps(bundle)
    assert "90% CI" in bundle["interactions"]["rule"]


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


def test_each_chosen_interaction_is_reported_in_declared_order(bundle):
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


def test_profiles_are_scored_with_a_retained_interaction():
    # No interaction is retained on the real data, so this path needs its own
    # check: synthetic candies where chocolate and peanut reinforce each other.
    from candy.data import ATTRIBUTES, LEVELS, Interaction
    from candy.estimators import fit
    from candy.profiles import ProfileScorer

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.integers(0, 2, size=(300, len(ATTRIBUTES))), columns=ATTRIBUTES)
    for level in LEVELS:
        frame[level] = rng.uniform(0, 1, size=len(frame))
    interaction = Interaction("chocolate", "peanuty_almondy")
    frame[interaction.term] = interaction.values(frame)
    win_rate = (
        40 + 10 * frame["chocolate"] + 5 * frame["peanuty_almondy"]
        + 8 * frame[interaction.term] + rng.normal(0, 1, size=len(frame))
    )
    model = fit(frame.to_numpy(), win_rate.to_numpy(), list(frame.columns), cov="hc3")
    levels = {level: 0.5 for level in LEVELS}
    scorer = ProfileScorer(model, levels, (interaction,))

    blank = dict.fromkeys(ATTRIBUTES, 0)
    chocolate = blank | {"chocolate": 1}
    both = chocolate | {"peanuty_almondy": 1}
    coef = model.coef

    assert scorer.predict(both)["estimate"] == pytest.approx(
        coef["intercept"] + coef["chocolate"] + coef["peanuty_almondy"]
        + coef[interaction.term] + 0.5 * (coef["sugar_level"] + coef["price_level"])
    )
    # Adding peanut to chocolate earns the peanut effect plus the interaction.
    assert scorer.gap(both, chocolate).estimate == pytest.approx(
        coef["peanuty_almondy"] + coef[interaction.term]
    )
    assert scorer.gap(both, chocolate).estimate == pytest.approx(13, abs=1)


def test_gummy_check_adds_the_gummy_attribute_to_the_main_model(bundle):
    check = bundle["robustness"]["gummy"]

    assert check["n_gummy"] == 11  # approved coding: 11 of 83 candies
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

    assert check["n_unclear"] == 11
    # 11 gummies and 11 judgement calls is a coincidence: only 4 overlap.
    assert check["n_unclear_gummy"] == 4
    assert "4 of the 11 gummies and 7 of the other candies" in "\n".join(bundle["data_notes"])
    assert "Swedish Fish" in check["unclear_candies"]
    assert "Starburst" in check["unclear_candies"]


def test_gummy_check_reports_the_effect_on_clear_cases_only(bundle):
    # Oracle: recode judgement calls as not gummy and solve OLS directly.
    raw = pd.read_csv(RAW)
    raw = raw[~raw["competitorname"].isin(["One dime", "One quarter"])]
    coding = pd.read_csv(GUMMY).set_index("candy")
    clear = (coding["gummy"] * (1 - coding["unclear"]))
    columns = ["chocolate", "fruity", "caramel", "peanutyalmondy", "nougat",
               "crispedricewafer", "hard", "bar", "pluribus", "sugarpercent", "pricepercent"]
    design = np.column_stack([
        np.ones(len(raw)),
        raw[columns].to_numpy(),
        raw["competitorname"].map(clear).to_numpy(),
    ])
    beta, *_ = np.linalg.lstsq(design, raw["winpercent"].to_numpy(), rcond=None)

    check = bundle["robustness"]["gummy"]["clear_cases_only"]
    assert check["n_gummy"] == 7
    assert check["coef"] == pytest.approx(beta[-1], rel=1e-10)
    assert check["ci_low"] < check["coef"] < check["ci_high"]
    # On this data the effect is no longer clear without the judgement calls.
    assert check["ci_low"] < 0 < check["ci_high"]
    assert "depends on the judgement calls" in "\n".join(bundle["data_notes"])


def test_brand_check_refits_the_main_model_without_reeses(bundle):
    # Oracle: drop the Reese's candies by hand and solve OLS directly.
    raw = pd.read_csv(RAW)
    raw = raw[~raw["competitorname"].isin(["One dime", "One quarter"])]
    reeses = raw["competitorname"].str.startswith("Reese's")
    kept = raw.loc[~reeses]
    columns = ["chocolate", "fruity", "caramel", "peanutyalmondy", "nougat",
               "crispedricewafer", "hard", "bar", "pluribus", "sugarpercent", "pricepercent"]
    design = np.column_stack([np.ones(len(kept)), kept[columns].to_numpy()])
    beta, *_ = np.linalg.lstsq(design, kept["winpercent"].to_numpy(), rcond=None)

    check = bundle["robustness"]["without_brand"]
    assert sorted(check["excluded"]) == sorted(raw.loc[reeses, "competitorname"])
    assert check["n_obs"] == 79
    assert [row["coef"] for row in check["coefficients"]] == pytest.approx(beta, rel=1e-10)


def test_brand_check_shows_peanut_rests_on_reeses(bundle):
    rows = {r["term"]: r for r in bundle["robustness"]["without_brand"]["coefficients"]}
    note = next(n for n in bundle["data_notes"] if "Reese's" in n)

    assert rows["chocolate"]["ci_low"] > 0
    assert rows["peanuty_almondy"]["ci_low"] < 0 < rows["peanuty_almondy"]["ci_high"]
    assert "no longer" in note and "peanut" in note


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
        (lambda lines: [l.replace("Twix,0,0,", "Twix,0,2,") for l in lines], "0/1.*Twix"),
        (lambda lines: [l.replace("Twix,0,0,", "Twix,0,,") for l in lines], "missing value.*Twix"),
    ],
    ids=["missing candy", "unknown candy", "non-binary value", "blank value", "duplicate row",
         "non-binary unclear flag", "blank unclear flag"],
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


def test_profiles_are_ranked_by_predicted_win_rate(bundle):
    ranked = bundle["profiles"]["ranked"]

    assert [row["estimate"] for row in ranked] == sorted(
        (row["estimate"] for row in ranked), reverse=True
    )


def test_attribute_evidence_follows_holm_and_the_confidence_intervals(bundle):
    # Oracle: recompute the tiers from the published tables.
    holm = {row["term"]: row["survives"] for row in bundle["holm"]["results"]}
    coefs = {row["term"]: row for row in bundle["main_model"]["coefficients"]}

    expected = {}
    for attribute in ATTRIBUTES:
        row = coefs[attribute]
        excludes_zero = row["ci_low"] > 0 or row["ci_high"] < 0
        expected[attribute] = {
            "tier": "driver" if holm[attribute] else "tentative" if excludes_zero else "open",
            "direction": "positive" if row["coef"] > 0 else "negative",
        }

    assert bundle["recommendation"]["attribute_evidence"] == expected
    assert expected["chocolate"]["tier"] == "driver"
    assert expected["nougat"]["tier"] == "open"
    assert expected["hard"]["direction"] == "negative"


def test_chocolate_vs_fruity_is_the_difference_of_their_coefficients(bundle):
    coef = {r["term"]: r["coef"] for r in bundle["main_model"]["coefficients"]}
    contrast = bundle["chocolate_vs_fruity"]

    assert contrast["estimate"] == pytest.approx(coef["chocolate"] - coef["fruity"])
    assert contrast["ci_low"] < contrast["estimate"] < contrast["ci_high"]
    # 9 candies are neither; Tootsie Pop is the only one that is both.
    assert contrast["n_neither"] == len(contrast["candies_neither"]) == 9
    # Fruity's coefficient is positive, yet chocolate clearly beats fruity.
    assert coef["fruity"] > 0 and contrast["ci_low"] > 0


def test_primary_profile_is_the_best_profile_made_only_of_drivers(bundle):
    ranked = bundle["profiles"]["ranked"]
    rec = bundle["recommendation"]
    drivers = {a for a, e in rec["attribute_evidence"].items() if e["tier"] == "driver"}

    def only_drivers(row):
        return all(a in drivers for a, on in row["attributes"].items() if on)

    assert only_drivers(rec["primary"])
    assert rec["primary"]["estimate"] == max(r["estimate"] for r in ranked if only_drivers(r))
    # On this data chocolate is the only driver, so the brief is "chocolate".
    assert [a for a, on in rec["primary"]["attributes"].items() if on] == ["chocolate"]


def test_top_ranked_profile_is_not_recommended_and_is_tied_with_the_primary(bundle):
    # The top profile stacks six attributes, most of them without evidence;
    # its predicted lead over the primary profile has a CI that includes zero.
    ranked = bundle["profiles"]["ranked"]
    primary = bundle["recommendation"]["primary"]

    assert ranked[0]["attributes"] != primary["attributes"]
    assert ranked[0]["tied_with_primary"]


def test_ties_with_the_primary_profile_are_flagged_and_counted(bundle):
    ranked = bundle["profiles"]["ranked"]
    primary = bundle["recommendation"]["primary"]
    by_attributes = {tuple(r["attributes"].values()): r for r in ranked}

    # The primary profile's gap to itself is zero; it must not count as a tie.
    assert not by_attributes[tuple(primary["attributes"].values())]["tied_with_primary"]
    assert bundle["profiles"]["n_tied_with_primary"] == sum(
        r["tied_with_primary"] for r in ranked
    )
    assert 1 < bundle["profiles"]["n_tied_with_primary"] < len(ranked) - 1


def test_cookie_vs_gummy_compares_real_product_types(bundle):
    products = bundle["cookie_vs_gummy"]["products"]

    assert {name: p["attributes"] for name, p in products.items()} == {
        "chocolate": {"chocolate": 1},
        "chocolate_cookie": {"chocolate": 1, "crisped_rice_wafer": 1},
        "gummy": {"fruity": 1, "gummy": 1},
    }


def test_cookie_vs_gummy_differences_come_from_the_gummy_model(bundle):
    # Oracle: with levels held equal, each difference is a sum of gummy-model
    # coefficients, and equals the difference of the two predictions.
    coef = {r["term"]: r["coef"] for r in bundle["robustness"]["gummy"]["coefficients"]}
    debate = bundle["cookie_vs_gummy"]
    products, differences = debate["products"], debate["differences"]
    gummy = coef["fruity"] + coef["gummy"]

    assert set(differences) == {"chocolate_minus_gummy", "chocolate_cookie_minus_gummy"}
    assert differences["chocolate_minus_gummy"]["estimate"] == pytest.approx(
        coef["chocolate"] - gummy
    )
    assert differences["chocolate_cookie_minus_gummy"]["estimate"] == pytest.approx(
        coef["chocolate"] + coef["crisped_rice_wafer"] - gummy
    )
    for name, gap in differences.items():
        product = name.removesuffix("_minus_gummy")
        assert gap["estimate"] == pytest.approx(
            products[product]["estimate"] - products["gummy"]["estimate"]
        )
        assert gap["ci_low"] < gap["estimate"] < gap["ci_high"]
        assert gap["decided"] == (gap["ci_low"] > 0 or gap["ci_high"] < 0)


def test_cookie_vs_gummy_note_explains_why_chocolate_scores_differ(bundle):
    # Same product, two models: the note must say so, or the deck shows two
    # different scores for chocolate without explanation.
    debate = bundle["cookie_vs_gummy"]
    primary = bundle["recommendation"]["primary"]

    assert debate["products"]["chocolate"]["estimate"] != primary["estimate"]
    assert "gummy model" in debate["note"] and "main model" in debate["note"]


def test_neither_chocolate_product_clearly_beats_a_gummy_on_this_data(bundle):
    # Chocolate leads a gummy by about 4 points, CI about [-5, 13].
    differences = bundle["cookie_vs_gummy"]["differences"]

    assert all(gap["estimate"] > 0 for gap in differences.values())
    assert not any(gap["decided"] for gap in differences.values())


def test_recommendations_name_two_or_three_real_benchmark_candies(bundle):
    real = set(pd.read_csv(RAW)["competitorname"]) - {"One dime", "One quarter"}

    marks = bundle["recommendation"]["primary"]["benchmarks"]
    assert 2 <= len(marks) <= 3
    assert {m["candy"] for m in marks} <= real
    distances = [m["attributes_differing"] for m in marks]
    assert distances == sorted(distances)


def test_primary_profile_summarises_every_candy_within_one_attribute(bundle):
    # Oracle: recompute from the raw CSV. On this data 14 candies sit within
    # one attribute of "chocolate only", averaging about 56.6.
    raw = pd.read_csv(RAW)
    raw = raw[~raw["competitorname"].isin(["One dime", "One quarter"])]
    raw_columns = ["chocolate", "fruity", "caramel", "peanutyalmondy", "nougat",
                   "crispedricewafer", "hard", "bar", "pluribus"]
    rec = bundle["recommendation"]["primary"]
    profile = np.array([rec["attributes"][a] for a in ATTRIBUTES])
    near = raw.loc[(raw[raw_columns].to_numpy() != profile).sum(axis=1) <= 1, "winpercent"]

    nearby = rec["nearby_candies"]
    assert nearby["max_attributes_differing"] == 1
    assert nearby["n"] == len(near) == 14
    assert nearby["mean_win_rate"] == pytest.approx(near.mean())
    assert nearby["min_win_rate"] == pytest.approx(near.min())
    assert nearby["max_win_rate"] == pytest.approx(near.max())
    assert {m["candy"] for m in rec["benchmarks"]} <= set(
        raw.loc[near.index, "competitorname"]
    )


def _profile_rows(bundle) -> list[dict]:
    return [
        *bundle["profiles"]["ranked"],
        bundle["recommendation"]["primary"],
        *bundle["cookie_vs_gummy"]["products"].values(),
    ]


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
    # The top-ranked profile's raw CI runs to about 103.
    top = bundle["profiles"]["ranked"][0]
    assert top["capped"] and top["ci_high"] == 100
    assert not bundle["recommendation"]["primary"]["capped"]


def test_recommendations_report_how_far_they_extrapolate(bundle):
    raw = pd.read_csv(RAW)
    raw = raw[~raw["competitorname"].isin(["One dime", "One quarter"])]
    raw_columns = ["chocolate", "fruity", "caramel", "peanutyalmondy", "nougat",
                   "crispedricewafer", "hard", "bar", "pluribus"]
    most_on_a_real_candy = int(raw[raw_columns].sum(axis=1).max())

    rec = bundle["recommendation"]["primary"]
    assert rec["n_attributes"] == sum(rec["attributes"].values())
    assert rec["max_attributes_on_a_real_candy"] == most_on_a_real_candy
    assert rec["nearest_candy_distance"] == rec["benchmarks"][0]["attributes_differing"]
    assert rec["nearest_candy_distance"] <= 1


def test_sugar_and_price_are_fixed_at_their_medians(bundle):
    # Sugar fails the Holm correction, so it is not used as a lever even
    # though its own CI excludes zero.
    raw = pd.read_csv(RAW)
    candies = raw.loc[~raw["competitorname"].isin(["One dime", "One quarter"])]
    levels = bundle["profiles"]["levels"]

    assert levels["sugar_level"] == pytest.approx(candies["sugarpercent"].median())
    assert levels["price_level"] == pytest.approx(candies["pricepercent"].median())


def test_benchmarks_are_the_nearest_candies_with_sugar_and_price_breaking_ties(bundle):
    raw = pd.read_csv(RAW)
    raw = raw[~raw["competitorname"].isin(["One dime", "One quarter"])]
    raw_columns = ["chocolate", "fruity", "caramel", "peanutyalmondy", "nougat",
                   "crispedricewafer", "hard", "bar", "pluribus"]
    levels = bundle["profiles"]["levels"]

    rec = bundle["recommendation"]["primary"]
    profile = np.array([rec["attributes"][a] for a in ATTRIBUTES])
    attr = (raw[raw_columns].to_numpy() != profile).sum(axis=1)
    level = np.hypot(raw["sugarpercent"] - levels["sugar_level"],
                     raw["pricepercent"] - levels["price_level"])
    ranking = sorted(zip(attr, level, raw["competitorname"]))
    assert [m["candy"] for m in rec["benchmarks"]] == [name for *_, name in ranking[:3]]


def test_appendix_vif_table_lists_every_predictor(out_dir, bundle):
    vif = pd.read_csv(out_dir / "appendix" / "vif.csv")

    assert list(vif.columns) == ["predictor", "label", "vif"]
    assert list(vif["predictor"]) == MAIN_TERMS[1:]
    assert vif.set_index("predictor")["vif"].to_dict() == pytest.approx(
        bundle["main_model"]["vif"]
    )


def test_appendix_interaction_support_counts_the_candies_behind_each_term(out_dir):
    # Counted by hand from the raw data: 37 chocolate candies, of which 12
    # have peanut/almond, 7 crisped/wafer and 20 are bars.
    support = pd.read_csv(out_dir / "appendix" / "interaction_support.csv")

    assert support.to_dict(orient="records") == [
        {"interaction": "Chocolate x Peanut / almond", "n_both": 12,
         "n_partner_only": 2, "n_attribute_only": 25, "status": "estimated"},
        {"interaction": "Chocolate x Crisped rice / wafer / cookie", "n_both": 7,
         "n_partner_only": 0, "n_attribute_only": 30, "status": "not_estimable"},
        {"interaction": "Chocolate x Bar format", "n_both": 20,
         "n_partner_only": 1, "n_attribute_only": 17, "status": "not_estimable"},
    ]


def test_appendix_gummy_table_is_the_approved_coding(out_dir):
    appendix = pd.read_csv(out_dir / "appendix" / "gummy_coding.csv")
    approved = pd.read_csv(GUMMY)

    assert list(appendix.columns) == ["candy", "gummy", "unclear", "note"]
    pd.testing.assert_frame_equal(
        appendix.sort_values("candy").reset_index(drop=True),
        approved.sort_values("candy").reset_index(drop=True),
    )


def test_appendix_data_notes_state_exclusions_and_sample_size(out_dir, bundle):
    notes = (out_dir / "appendix" / "data_notes.md").read_text()

    assert notes == "\n".join(f"- {note}" for note in bundle["data_notes"]) + "\n"
    assert "One dime" in notes and "One quarter" in notes
    assert "83 candies" in notes
    assert "peanutyalmondy" in notes
