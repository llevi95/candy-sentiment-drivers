"""Attribute profiles: candidate new products built from the attributes.

A profile is a 0/1 value for every attribute. Only realistic profiles that sit
within one attribute of an existing candy (supported profiles) are considered,
so the recommendation never strays far from products consumers have rated.
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


@dataclass(frozen=True)
class ProfileScorer:
    """Predicts profiles with one model at fixed sugar and price levels."""

    model: FitResult
    levels: dict[str, float]
    interactions: tuple[Interaction, ...]

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


def choose_levels(candies: pd.DataFrame, model: FitResult) -> tuple[dict[str, float], str]:
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


def rank(profiles: pd.DataFrame, scorer: ProfileScorer) -> list[dict]:
    """Predict every profile, best first."""
    rows = []
    for _, profile in profiles.iterrows():
        attributes = {a: int(profile[a]) for a in ATTRIBUTES}
        rows.append({"attributes": attributes, **scorer.predict(attributes)})
    rows.sort(key=lambda row: row["estimate"], reverse=True)
    return [{"rank": i + 1, **row} for i, row in enumerate(rows)]


def recommend(ranked: list[dict], scorer: ProfileScorer, candies: pd.DataFrame) -> dict:
    """Primary and fruit-flavoured profiles, the gap between them, benchmark
    candies, and how far each recommendation extrapolates beyond real candies."""
    primary = ranked[0]
    fruit = next(row for row in ranked if row["attributes"]["fruity"] == 1)
    most_on_a_real_candy = int(candies[ATTRIBUTES].sum(axis=1).max())

    def describe(row: dict) -> dict:
        marks = benchmarks(row["attributes"], scorer.levels, candies)
        return {
            **row,
            "benchmarks": marks,
            "n_attributes": sum(row["attributes"].values()),
            "max_attributes_on_a_real_candy": most_on_a_real_candy,
            "nearest_candy_distance": marks[0]["attributes_differing"],
        }

    gap = scorer.gap(primary["attributes"], fruit["attributes"])
    return {
        "primary": describe(primary),
        "fruit_flavoured": describe(fruit),
        "gap": asdict(gap),
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
