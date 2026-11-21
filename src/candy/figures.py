"""Slide-ready figures (16:9, readable at presentation size)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

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


def coefficient_chart(table: pd.DataFrame, path: Path) -> None:
    """Horizontal CI chart of each term's effect on win rate, intercept excluded.

    ``table`` has one row per term with ``coef``, ``ci_low`` and ``ci_high``.
    """
    rows = table.drop(index="intercept").sort_values("coef")
    clear = (rows["ci_low"] > 0) | (rows["ci_high"] < 0)
    colors = [DRIVER if c else NOT_CLEAR for c in clear]
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.hlines(y, rows["ci_low"], rows["ci_high"], colors=colors, linewidth=3)
    ax.scatter(rows["coef"], y, color=colors, s=90, zorder=3)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(list(y), [LABELS.get(t, t) for t in rows.index])
    ax.set_xlabel("Change in win rate (percentage points), 95% CI")
    ax.set_title("What moves consumer preference")
    fig.text(
        0.99,
        0.015,
        "Blue: interval excludes zero. Grey: effect not distinguishable from zero.",
        ha="right",
        fontsize=12,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
