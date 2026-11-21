"""Slide-ready figures (16:9, readable at presentation size)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
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


def coefficient_chart(
    table: pd.DataFrame, survivors: set[str], alpha: float, path: Path
) -> None:
    """Horizontal CI chart of each term's effect on win rate, intercept excluded.

    ``table`` has one row per term with ``coef``, ``ci_low`` and ``ci_high`` at
    level ``1 - alpha``. Terms in ``survivors`` (Holm-significant across every
    plotted term) are highlighted; terms whose raw CI excludes zero but that do
    not survive Holm are shown as tentative.
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
    ax.set_xlabel(f"Change in win rate (percentage points), {1 - alpha:.0%} CI")
    ax.set_title("What moves consumer preference")
    _footnote(
        fig,
        f"Dark blue: significant after Holm correction across all {len(rows)} effects. "
        f"Light blue: {1 - alpha:.0%} CI excludes zero, but not after correction. "
        "Grey: neither.",
    )
    _save(fig, path)


def attribute_comparison_chart(rows: pd.DataFrame, alpha: float, path: Path) -> None:
    """Average win rate of candies with vs without each attribute.

    ``rows`` has one row per attribute (index) with ``mean_with``,
    ``mean_without``, ``n_with``, ``n_without`` and Welch ``p``.
    """
    rows = rows.sort_values("mean_with")
    colors = [DRIVER if p < alpha else NOT_CLEAR for p in rows["p"]]
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.hlines(y, rows["mean_without"], rows["mean_with"], colors=colors, linewidth=3)
    ax.scatter(
        rows["mean_without"], y, color="white", edgecolor="#333333",
        s=90, zorder=3, label="Without attribute",
    )
    ax.scatter(
        rows["mean_with"], y, color=colors, edgecolor="#333333",
        s=90, zorder=3, label="With attribute",
    )
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
        f"Blue: difference significant (Welch t-test, p < {alpha:g}). "
        "Simple comparison; does not control for other attributes.",
    )
    _save(fig, path)


def robustness_chart(intervals: dict[str, pd.DataFrame], alpha: float, path: Path) -> None:
    """Side-by-side intervals for each predictor under several analyses.

    ``intervals`` maps an analysis label to a frame indexed by predictor with
    ``coef``, ``ci_low`` and ``ci_high`` in points of win rate. Predictors are
    ordered by the first analysis's estimate.
    """
    labels = list(intervals)
    order = intervals[labels[0]].sort_values("coef").index
    palette = [DRIVER, "#e07b39", "#6a4c93"]
    offsets = np.linspace(0.25, -0.25, len(labels))  # first label on top, as in legend

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    for label, offset, color in zip(labels, offsets, palette):
        rows = intervals[label].loc[order]
        y = np.arange(len(order)) + offset
        ax.hlines(y, rows["ci_low"], rows["ci_high"], colors=color, linewidth=3)
        ax.scatter(rows["coef"], y, color=color, s=50, zorder=3, label=label)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(range(len(order)), [LABELS.get(t, t) for t in order])
    ax.set_xlabel(f"Change in win rate (percentage points), {1 - alpha:.0%} interval")
    ax.set_title("Drivers hold up under alternative analyses")
    ax.legend(loc="lower right", frameon=False)
    _footnote(fig, "Logit effects are average marginal effects, converted back to win-rate points.")
    _save(fig, path)


def _footnote(fig: plt.Figure, text: str) -> None:
    fig.text(0.99, 0.015, text, ha="right", fontsize=12, color="#555555")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
