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
class Interaction:
    """Product of two attributes, added to a model as one extra term."""

    attribute: str
    partner: str

    @property
    def term(self) -> str:
        return f"{self.attribute}_x_{self.partner}"

    def values(self, frame: pd.DataFrame) -> pd.Series:
        return frame[self.attribute] * frame[self.partner]


@dataclass(frozen=True)
class CandyData:
    candies: pd.DataFrame
    excluded: list[str]
    n_raw: int
    with_coins: pd.DataFrame  # every raw entry, coins included, for robustness only


def load_candies(raw_csv: Path, gummy_csv: Path) -> CandyData:
    """Read the raw CSV, normalise column names, validate, drop the coins, and
    join the hand-coded gummy attribute onto the candies."""
    raw = pd.read_csv(raw_csv).rename(columns=RENAME)
    missing = {"candy", "win_rate", *PREDICTORS} - set(raw.columns)
    if missing:
        raise ValueError(f"raw data is missing columns: {sorted(missing)}")
    _validate(raw)

    is_coin = raw["candy"].isin(COINS)
    candies = raw.loc[~is_coin].reset_index(drop=True)
    candies = candies.merge(_load_gummy(gummy_csv, candies["candy"]), on="candy")
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


def _load_gummy(gummy_csv: Path, candies: pd.Series) -> pd.DataFrame:
    """Read the approved gummy coding and check it covers exactly the candies."""
    gummy = pd.read_csv(gummy_csv, usecols=["candy", "gummy", "unclear"]).rename(
        columns={"unclear": "gummy_unclear"}
    )
    missing = sorted(set(candies) - set(gummy["candy"]))
    if missing:
        raise ValueError(f"gummy coding is missing candies: {missing}")
    unknown = sorted(set(gummy["candy"]) - set(candies))
    if unknown:
        raise ValueError(f"gummy coding names unknown candies: {unknown}")
    blank = gummy.loc[gummy["gummy"].isna(), "candy"].tolist()
    if blank:
        raise ValueError(f"gummy coding has a missing value for: {blank}")
    not_binary = gummy.loc[~gummy["gummy"].isin([0, 1]), "candy"].tolist()
    if not_binary:
        raise ValueError(f"gummy values must be 0/1: {not_binary}")
    duplicated = gummy.loc[gummy["candy"].duplicated(), "candy"].tolist()
    if duplicated:
        raise ValueError(f"gummy coding lists candies more than once: {duplicated}")
    return gummy.astype({"gummy": int, "gummy_unclear": int})
