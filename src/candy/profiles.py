"""Attribute profiles: candidate new products built from the attributes.

A profile is a 0/1 value for every attribute. Only realistic profiles that sit
within one attribute of an existing candy (supported profiles) are considered,
so the recommendation never strays far from products consumers have rated.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from candy.data import ATTRIBUTES

REALISM_RULES = (
    "not chocolate and fruity together",
    "hard excludes bar",
    "bar excludes pluribus",
)
SUPPORT_DISTANCE = 1  # max attributes a profile may differ from some real candy


def all_profiles() -> pd.DataFrame:
    """Every 0/1 combination of the attributes, one row each."""
    return pd.DataFrame(
        list(itertools.product([0, 1], repeat=len(ATTRIBUTES))), columns=ATTRIBUTES
    )


def is_realistic(profiles: pd.DataFrame) -> pd.Series:
    """Mask of profiles that obey every rule in REALISM_RULES."""
    return ~(
        (profiles["chocolate"] & profiles["fruity"])
        | (profiles["hard"] & profiles["bar"])
        | (profiles["bar"] & profiles["pluribus"])
    ).astype(bool)


def is_supported(profiles: pd.DataFrame, candies: pd.DataFrame) -> pd.Series:
    """Mask of profiles within SUPPORT_DISTANCE attributes of at least one candy."""
    distance = _attribute_distances(profiles, candies)
    return pd.Series(distance.min(axis=1) <= SUPPORT_DISTANCE, index=profiles.index)


def benchmarks(
    profile: pd.Series, levels: dict[str, float], candies: pd.DataFrame, k: int = 3
) -> list[dict]:
    """The ``k`` candies nearest the profile: fewest differing attributes, ties
    broken by Euclidean distance in sugar and price level."""
    attribute_distance = _attribute_distances(profile.to_frame().T, candies)[0]
    level_distance = np.hypot(
        candies["sugar_level"] - levels["sugar_level"],
        candies["price_level"] - levels["price_level"],
    )
    order = np.lexsort((level_distance.to_numpy(), attribute_distance))[:k]
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
