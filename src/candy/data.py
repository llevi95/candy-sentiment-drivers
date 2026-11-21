"""Load and clean the market research candy data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Survey decoys included as jokes in the original matchups; not products.
COINS = ("One dime", "One quarter")

# Raw column -> analysis name. The raw file's "peanutyalmondy" (its README
# spells it "peanutalmondy") becomes "peanuty_almondy".
RENAME = {
    "competitorname": "candy",
    "chocolate": "chocolate",
    "fruity": "fruity",
    "caramel": "caramel",
    "peanutyalmondy": "peanuty_almondy",
    "nougat": "nougat",
    "crispedricewafer": "crisped_rice_wafer",
    "hard": "hard",
    "bar": "bar",
    "pluribus": "pluribus",
    "sugarpercent": "sugar_level",
    "pricepercent": "price_level",
    "winpercent": "win_rate",
}

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
LEVELS = ["sugar_level", "price_level"]
PREDICTORS = ATTRIBUTES + LEVELS


@dataclass(frozen=True)
class CandyData:
    candies: pd.DataFrame
    excluded: list[str]
    n_raw: int
    with_coins: pd.DataFrame  # every raw entry, coins included, for robustness only


def load_candies(raw_csv: Path) -> CandyData:
    """Read the raw CSV, normalise column names, validate, and drop the coins."""
    raw = pd.read_csv(raw_csv).rename(columns=RENAME)
    missing = {"candy", "win_rate", *PREDICTORS} - set(raw.columns)
    if missing:
        raise ValueError(f"raw data is missing columns: {sorted(missing)}")
    _validate(raw)

    is_coin = raw["candy"].isin(COINS)
    candies = raw.loc[~is_coin].reset_index(drop=True)
    return CandyData(
        candies=candies,
        excluded=raw.loc[is_coin, "candy"].tolist(),
        n_raw=len(raw),
        with_coins=raw,
    )


def _validate(df: pd.DataFrame) -> None:
    bad_binary = [c for c in ATTRIBUTES if not df[c].isin([0, 1]).all()]
    if bad_binary:
        raise ValueError(f"attributes must be 0/1: {bad_binary}")
    bad_level = [c for c in LEVELS if not df[c].between(0, 1).all()]
    if bad_level:
        raise ValueError(f"levels must be percentiles in [0, 1]: {bad_level}")
    if not df["win_rate"].between(0, 100).all():
        raise ValueError("win rate must lie in [0, 100]")
