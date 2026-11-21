"""Slide-ready figures (16:9, readable at presentation size)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from candy.estimators import INTERCEPT  # noqa: E402

FIGSIZE = (12.8, 7.2)  # inches; 16:9
DPI = 150  # 1920 x 1080 px

LABELS = {
    "chocolate": "Chocolate",
    "fruity": "Fruit-flavoured",
    "caramel": "Caramel",
    "peanuty_almondy": "Peanut / almond",
    "nougat": "Nougat",
    "crisped_rice_wafer": "Crisped rice / wafer / cookie",
    "hard": "Hard candy",
    "bar": "Bar format",
    "pluribus": "Many per pack",
    "sugar_level": "Sugar level (lowest → highest)",
    "price_level": "Price level (cheapest → priciest)",
}

DRIVER = "#1f6f8b"
TENTATIVE = "#8cc3d6"
NOT_CLEAR = "#a3a3a3"

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def coefficient_chart(table: pd.DataFrame, survivors: set[str], path: Path) -> None:
    """Horizontal CI chart of each term's effect on win rate, intercept excluded.

    ``table`` has one row per term with ``coef``, ``ci_low`` and ``ci_high``.
    Terms in ``survivors`` (Holm-significant) are highlighted; terms whose raw
    95% CI excludes zero but that do not survive Holm are shown as tentative.
    """
    rows = table.drop(index=INTERCEPT).sort_values("coef")
    excludes_zero = (rows["ci_low"] > 0) | (rows["ci_high"] < 0)
    colors = [
        DRIVER if term in survivors else TENTATIVE if clear else NOT_CLEAR
        for term, clear in excludes_zero.items()
    ]
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.hlines(y, rows["ci_low"], rows["ci_high"], colors=colors, linewidth=3)
    ax.scatter(rows["coef"], y, color=colors, s=90, zorder=3)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(list(y), [LABELS.get(t, t) for t in rows.index])
    ax.set_xlabel("Change in win rate (percentage points), 95% CI")
    ax.set_title("What moves consumer preference")
    _footnote(
        fig,
        "Dark blue: significant after Holm correction across all 11 effects. "
        "Light blue: 95% CI excludes zero, but not after correction. Grey: neither.",
    )
    _save(fig, path)


def attribute_comparison_chart(rows: pd.DataFrame, path: Path) -> None:
    """Average win rate of candies with vs without each attribute.

    ``rows`` has one row per attribute (index) with ``mean_with``,
    ``mean_without``, ``n_with``, ``n_without`` and Welch ``p``.
    """
    rows = rows.sort_values("mean_with")
    clear = rows["p"] < 0.05
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.hlines(y, rows["mean_without"], rows["mean_with"],
              colors=[DRIVER if c else NOT_CLEAR for c in clear], linewidth=3)
    ax.scatter(rows["mean_without"], y, color="white", edgecolor="#333333",
               s=90, zorder=3, label="Without attribute")
    ax.scatter(rows["mean_with"], y, color=[DRIVER if c else NOT_CLEAR for c in clear],
               edgecolor="#333333", s=90, zorder=3, label="With attribute")
    ax.set_yticks(
        y,
        [
            f"{LABELS.get(a, a)}  (n = {n_with} vs {n_without})"
            for a, n_with, n_without in zip(rows.index, rows["n_with"], rows["n_without"])
        ],
    )
    ax.set_xlabel("Average win rate (%)")
    ax.set_title("Candies with vs without each attribute")
    ax.legend(loc="lower right", frameon=False)
    _footnote(
        fig,
        "Blue: difference significant (Welch t-test, p < 0.05). "
        "Simple comparison; does not control for other attributes.",
    )
    _save(fig, path)


def _footnote(fig: plt.Figure, text: str) -> None:
    fig.text(0.99, 0.015, text, ha="right", fontsize=12, color="#555555")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
