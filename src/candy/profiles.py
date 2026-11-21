"""Attribute profiles: candidate new products built from the attributes.

A profile is a 0/1 value for every attribute. Only realistic profiles that sit
within one attribute of an existing candy (supported profiles) are considered,
so the recommendation never strays far from products consumers have rated.

The recommended (primary) profile switches on only confirmed drivers. The
profile with the highest prediction is not recommended: it collects every
attribute with a positive coefficient, however weak the evidence, so it is
largely picked by noise.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from candy.data import ATTRIBUTES, Interaction
from candy.estimators import Difference, FitResult


@dataclass(frozen=True)
class RealismRule:
    """A combination of attributes no real product would have."""

    label: str
    violated_by: Callable[[pd.DataFrame], pd.Series]


REALISM_RULES = (
    RealismRule("not chocolate and fruity together", lambda p: p["chocolate"] & p["fruity"]),
    RealismRule("hard excludes bar", lambda p: p["hard"] & p["bar"]),
    RealismRule("bar excludes pluribus", lambda p: p["bar"] & p["pluribus"]),
)
SUPPORT_DISTANCE = 1  # max attributes a profile may differ from some real candy
BENCHMARK_COUNT = 3
WIN_RATE_SCALE = (0.0, 100.0)

# The purchasing team's debate as model rows. Every cookie candy in the data
# is chocolate, so the cookie side can only be a chocolate product.
DEBATE_PRODUCTS = {
    "chocolate": {"chocolate": 1},
    "chocolate_cookie": {"chocolate": 1, "crisped_rice_wafer": 1},
    "gummy": {"fruity": 1, "gummy": 1},
}
DEBATE_RIVAL = "gummy"

DRIVER, TENTATIVE, OPEN = "driver", "tentative", "open"
ABOVE, TIED, BELOW = "above", "tied", "below"
PRIMARY_RULE = (
    "The primary profile is the best supported profile made only of drivers: "
    "attributes whose positive effect survives the multiple-testing correction. "
    "Every other attribute is a design choice the data does not decide."
)


@dataclass(frozen=True)
class ProfileScorer:
    """Predicts profiles with one model at fixed sugar and price levels.

    ``extra_attributes`` names 0/1 predictors the model has beyond ATTRIBUTES
    (the gummy attribute); a profile that does not mention one has it off.
    """

    model: FitResult
    levels: dict[str, float]
    interactions: tuple[Interaction, ...]
    extra_attributes: tuple[str, ...] = ()

    def predict(self, attributes: dict[str, int]) -> dict:
        """Point estimate with mean CI and prediction interval, bounds capped
        to the win-rate scale; ``capped`` says whether any bound was cut."""
        raw = asdict(self.model.predict(self._row(attributes)))
        low, high = WIN_RATE_SCALE
        bounds = ("ci_low", "ci_high", "pi_low", "pi_high")
        capped = {key: float(np.clip(raw[key], low, high)) for key in bounds}
        return {
            "estimate": raw["estimate"],
            **capped,
            "capped": any(capped[key] != raw[key] for key in bounds),
        }

    def gap(self, a: dict[str, int], b: dict[str, int]) -> Difference:
        return self.model.predict_difference(self._row(a), self._row(b))

    def _row(self, attributes: dict[str, int]) -> dict[str, float]:
        row = {attribute: attributes[attribute] for attribute in ATTRIBUTES} | self.levels
        for extra in self.extra_attributes:
            row[extra] = attributes.get(extra, 0)
        for interaction in self.interactions:
            row[interaction.term] = row[interaction.attribute] * row[interaction.partner]
        return row


def all_profiles() -> pd.DataFrame:
    """Every 0/1 combination of the attributes, one row each."""
    return pd.DataFrame(
        list(itertools.product([0, 1], repeat=len(ATTRIBUTES))), columns=ATTRIBUTES
    )


def is_realistic(profiles: pd.DataFrame) -> pd.Series:
    """Mask of profiles that break none of REALISM_RULES."""
    broken = pd.Series(False, index=profiles.index)
    for rule in REALISM_RULES:
        broken |= rule.violated_by(profiles).astype(bool)
    return ~broken


def is_supported(profiles: pd.DataFrame, candies: pd.DataFrame) -> pd.Series:
    """Mask of profiles within SUPPORT_DISTANCE attributes of at least one candy."""
    distance = _attribute_distances(profiles, candies)
    return pd.Series(distance.min(axis=1) <= SUPPORT_DISTANCE, index=profiles.index)


def choose_levels(candies: pd.DataFrame) -> dict[str, float]:
    """Sugar and price level at which every profile is predicted: the median
    of each. Neither is a confirmed driver, so neither is used as a lever."""
    return {level: float(candies[level].median()) for level in ("sugar_level", "price_level")}


def rank(profiles: pd.DataFrame, scorer: ProfileScorer) -> list[dict]:
    """Predict every profile, best first."""
    rows = []
    for _, profile in profiles.iterrows():
        attributes = {a: int(profile[a]) for a in ATTRIBUTES}
        rows.append({"attributes": attributes, **scorer.predict(attributes)})
    rows.sort(key=lambda row: row["estimate"], reverse=True)
    return [{"rank": i + 1, **row} for i, row in enumerate(rows)]


def evidence_tiers(model: FitResult, confirmed: set[str]) -> dict[str, dict[str, str]]:
    """How well the data supports each attribute's effect, and its direction.

    ``tier`` is ``driver`` when the attribute is in ``confirmed`` (significant
    after multiple-testing correction), ``tentative`` when its own CI excludes
    zero but it is not confirmed, and ``open`` when the data does not decide,
    so it is a free design choice. ``direction`` is the sign of the estimate.
    """
    evidence = {}
    for attribute in ATTRIBUTES:
        excludes_zero = model.ci_low[attribute] > 0 or model.ci_high[attribute] < 0
        evidence[attribute] = {
            "tier": DRIVER if attribute in confirmed else TENTATIVE if excludes_zero else OPEN,
            "direction": "positive" if model.coef[attribute] > 0 else "negative",
        }
    return evidence


def mark_ties(ranked: list[dict], scorer: ProfileScorer, primary: dict[str, int]) -> list[dict]:
    """Place each other ranked profile against the primary one by the CI of
    their predicted gap: ``tied`` when it includes zero (the data cannot tell
    the two apart), else ``above`` or ``below``. A lower-ranked profile can be
    above while higher-ranked ones are tied, because a gap made of more
    attributes has a wider CI. The primary profile itself gets ``None`` and
    is not tied with itself."""

    def versus(attributes: dict[str, int]) -> str | None:
        if attributes == primary:
            return None
        gap = scorer.gap(attributes, primary)
        return ABOVE if gap.ci_low > 0 else BELOW if gap.ci_high < 0 else TIED

    placed = [{**row, "vs_primary": versus(row["attributes"])} for row in ranked]
    return [{**row, "tied_with_primary": row["vs_primary"] == TIED} for row in placed]


def recommend(
    ranked: list[dict],
    scorer: ProfileScorer,
    candies: pd.DataFrame,
    evidence: dict[str, dict[str, str]],
) -> dict:
    """The primary profile with its benchmark candies and how far it
    extrapolates beyond real candies.

    The primary profile is the best supported profile that switches on nothing
    but drivers with a positive effect.
    """
    favourable = {
        a for a, e in evidence.items() if e["tier"] == DRIVER and e["direction"] == "positive"
    }
    primary = next(
        row
        for row in ranked
        if all(a in favourable for a, on in row["attributes"].items() if on)
    )
    most_on_a_real_candy = int(candies[ATTRIBUTES].sum(axis=1).max())

    def describe(row: dict) -> dict:
        marks = benchmarks(row["attributes"], scorer.levels, candies)
        return {
            **row,
            "benchmarks": marks,
            "nearby_candies": nearby_candies(row["attributes"], candies),
            "n_attributes": sum(row["attributes"].values()),
            "max_attributes_on_a_real_candy": most_on_a_real_candy,
            "nearest_candy_distance": marks[0]["attributes_differing"],
        }

    return {
        "rule": PRIMARY_RULE,
        "attribute_evidence": evidence,
        "primary": describe(primary),
    }


def compare_cookie_and_gummy(scorer: ProfileScorer) -> dict:
    """Answer the cookie-versus-gummy debate with direct differences.

    ``scorer`` must hold a model with the gummy attribute. Each of
    DEBATE_PRODUCTS is predicted, and each chocolate product is compared with
    the gummy; ``decided`` says whether the difference's CI excludes zero.
    """
    blank = dict.fromkeys(ATTRIBUTES, 0)
    products = {name: blank | switched_on for name, switched_on in DEBATE_PRODUCTS.items()}
    differences = {}
    for name in products.keys() - {DEBATE_RIVAL}:
        gap = scorer.gap(products[name], products[DEBATE_RIVAL])
        differences[f"{name}_minus_{DEBATE_RIVAL}"] = {
            **asdict(gap),
            "decided": bool(gap.ci_low > 0 or gap.ci_high < 0),
        }
    return {
        "products": {
            name: {"attributes": DEBATE_PRODUCTS[name], **scorer.predict(row)}
            for name, row in products.items()
        },
        "differences": dict(sorted(differences.items())),
    }


def nearby_candies(attributes: dict[str, int], candies: pd.DataFrame) -> dict:
    """Win rates of every real candy within SUPPORT_DISTANCE attributes of the
    profile: the market evidence a prediction can be checked against."""
    distance = _attribute_distances(pd.DataFrame([attributes]), candies)[0]
    win_rates = candies.loc[distance <= SUPPORT_DISTANCE, "win_rate"]
    return {
        "max_attributes_differing": SUPPORT_DISTANCE,
        "n": int(len(win_rates)),
        "mean_win_rate": float(win_rates.mean()),
        "min_win_rate": float(win_rates.min()),
        "max_win_rate": float(win_rates.max()),
    }


def benchmarks(
    attributes: dict[str, int], levels: dict[str, float], candies: pd.DataFrame
) -> list[dict]:
    """The BENCHMARK_COUNT candies nearest the profile: fewest differing
    attributes, ties broken by Euclidean distance in sugar and price level."""
    profile = pd.DataFrame([attributes])
    attribute_distance = _attribute_distances(profile, candies)[0]
    level_distance = np.hypot(
        candies["sugar_level"] - levels["sugar_level"],
        candies["price_level"] - levels["price_level"],
    )
    order = np.lexsort((level_distance.to_numpy(), attribute_distance))[:BENCHMARK_COUNT]
    return [
        {
            "candy": candies["candy"].iloc[i],
            "attributes_differing": int(attribute_distance[i]),
            "win_rate": float(candies["win_rate"].iloc[i]),
        }
        for i in order
    ]


def _attribute_distances(profiles: pd.DataFrame, candies: pd.DataFrame) -> np.ndarray:
    """Hamming distance from each profile (rows) to each candy (columns)."""
    a = profiles[ATTRIBUTES].to_numpy()
    b = candies[ATTRIBUTES].to_numpy()
    return (a[:, None, :] != b[None, :, :]).sum(axis=2)
