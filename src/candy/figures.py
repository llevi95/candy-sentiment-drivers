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

# Compact attribute names used to spell out a profile in one line.
PROFILE_LABELS = {
    "chocolate": "Chocolate",
    "fruity": "Fruity",
    "caramel": "Caramel",
    "peanuty_almondy": "Peanut",
    "nougat": "Nougat",
    "crisped_rice_wafer": "Crisped/wafer",
    "hard": "Hard",
    "bar": "Bar",
    "pluribus": "Multi-pack",
}

DRIVER = "#1f6f8b"
TENTATIVE = "#8cc3d6"
ACCENT_ORANGE = "#e07b39"
ACCENT_PURPLE = "#6a4c93"
ANALYSIS_COLORS = (DRIVER, ACCENT_ORANGE, ACCENT_PURPLE)
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


def robustness_chart(
    intervals: dict[str, pd.DataFrame], alpha: float, footnote: str, path: Path
) -> None:
    """Side-by-side intervals for each predictor under up to three analyses.

    ``intervals`` maps an analysis label to a frame indexed by predictor with
    ``coef``, ``ci_low`` and ``ci_high`` in points of win rate. Predictors are
    ordered by the first analysis's estimate.
    """
    labels = list(intervals)
    if len(labels) > len(ANALYSIS_COLORS):
        raise ValueError(f"at most {len(ANALYSIS_COLORS)} analyses fit on one chart")
    order = intervals[labels[0]].sort_values("coef").index
    offsets = np.linspace(0.25, -0.25, len(labels))  # first label on top, as in legend

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    for label, offset, color in zip(labels, offsets, ANALYSIS_COLORS):
        rows = intervals[label].loc[order]
        y = np.arange(len(order)) + offset
        ax.hlines(y, rows["ci_low"], rows["ci_high"], colors=color, linewidth=3)
        ax.scatter(rows["coef"], y, color=color, s=50, zorder=3, label=label)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(range(len(order)), [LABELS.get(t, t) for t in order])
    ax.set_xlabel(f"Change in win rate (percentage points), {1 - alpha:.0%} interval")
    ax.set_title("Effect estimates under alternative analyses")
    ax.legend(loc="lower right", frameon=False)
    _footnote(fig, footnote)
    _save(fig, path)


def win_rate_distribution_chart(candies: pd.DataFrame, path: Path) -> None:
    """Histogram of candy win rates, stacked by chocolate-based, fruit-flavoured
    and neither, with the 50% line where a candy wins as often as it loses."""
    groups = [
        ("Chocolate-based", candies["chocolate"] == 1, DRIVER),
        ("Fruit-flavoured", (candies["fruity"] == 1) & (candies["chocolate"] == 0), ACCENT_ORANGE),
        ("Neither", (candies["chocolate"] == 0) & (candies["fruity"] == 0), NOT_CLEAR),
    ]
    width = 5
    low = np.floor(candies["win_rate"].min() / width) * width
    high = np.ceil(candies["win_rate"].max() / width) * width
    bins = np.arange(low, high + width, width)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.hist(
        [candies.loc[mask, "win_rate"] for _, mask, _ in groups],
        bins=bins,
        stacked=True,
        color=[color for *_, color in groups],
        label=[f"{name} (n = {int(mask.sum())})" for name, mask, _ in groups],
        edgecolor="white",
    )
    ax.axvline(50, color="black", linewidth=1, linestyle="--")
    ax.annotate("wins half its matchups", (50, ax.get_ylim()[1] * 0.95), xytext=(6, 0),
                textcoords="offset points", fontsize=12, color="#333333")
    ax.set_xlabel("Win rate (% of head-to-head matchups won)")
    ax.set_ylabel("Number of candies")
    ax.set_title(f"How the {len(candies)} candies score")
    ax.legend(loc="upper left", frameon=False)
    _footnote(fig, "Win rate measures relative preference in matchups, not sales or satisfaction.")
    _save(fig, path)


def profile_label(attributes: dict[str, int]) -> str:
    """Short name for a profile, e.g. 'Chocolate + Peanut + Bar'."""
    present = [PROFILE_LABELS[a] for a, on in attributes.items() if on]
    return " + ".join(present) if present else "No listed attribute"


def ranked_profiles_chart(ranked: list[dict], top: int, alpha: float, path: Path) -> None:
    """Top ``top`` supported profiles with mean CI (thick) and prediction
    interval (thin); fruit-flavoured profiles highlighted."""
    rows = ranked[:top][::-1]  # best at the top of the chart
    y = np.arange(len(rows))
    colors = [ACCENT_ORANGE if r["attributes"]["fruity"] else DRIVER for r in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    _prediction_intervals(ax, rows, y, colors, ci_width=5, pi_width=1.5, marker_size=50)
    ax.set_yticks(y, [f"{r['rank']}. {profile_label(r['attributes'])}" for r in rows], fontsize=12)
    ax.set_xlabel("Predicted win rate (%)")
    ax.set_title(f"Top {len(rows)} supported attribute profiles")
    any_fruit = any(r["attributes"]["fruity"] for r in rows)
    _footnote(fig, _interval_legend(alpha, "Orange: fruit-flavoured." if any_fruit else ""))
    _save(fig, path)


def recommendation_gap_chart(
    primary: dict, fruit: dict, gap: dict, alpha: float, path: Path
) -> None:
    """Primary vs fruit-flavoured profile, with intervals and the gap between them."""
    rows = [("Fruit-flavoured", fruit, ACCENT_ORANGE), ("Primary", primary, DRIVER)]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    profiles = [row for _, row, _ in rows]
    _prediction_intervals(
        ax, profiles, [0, 1], [color for *_, color in rows],
        ci_width=8, pi_width=2, marker_size=120,
    )
    for y, row in enumerate(profiles):
        ax.annotate(f"{row['estimate']:.0f}", (row["estimate"], y), xytext=(0, 14),
                    textcoords="offset points", ha="center", fontsize=14, fontweight="bold")
    ax.set_yticks(
        [0, 1],
        [f"{name}\n{profile_label(row['attributes'])}" for name, row, _ in rows],
    )
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Predicted win rate (%)")
    fig.suptitle(
        f"Primary vs fruit-flavoured: gap of {gap['estimate']:.0f} points "
        f"({1 - alpha:.0%} CI {gap['ci_low']:.0f} to {gap['ci_high']:.0f})",
        fontsize=20,
        fontweight="bold",
    )
    _footnote(fig, _interval_legend(alpha))
    _save(fig, path)


def _prediction_intervals(
    ax: plt.Axes,
    rows: list[dict],
    y,
    colors: list[str],
    *,
    ci_width: float,
    pi_width: float,
    marker_size: float,
) -> None:
    """Thin prediction interval, thick mean CI and a hollow point per profile,
    with a dashed line at the 100% ceiling the ranges are capped to."""
    ax.hlines(y, [r["pi_low"] for r in rows], [r["pi_high"] for r in rows],
              colors=colors, linewidth=pi_width, alpha=0.5)
    ax.hlines(y, [r["ci_low"] for r in rows], [r["ci_high"] for r in rows],
              colors=colors, linewidth=ci_width)
    ax.scatter([r["estimate"] for r in rows], y, color="white", edgecolor="#333333",
               s=marker_size, zorder=3)
    ax.axvline(100, color="#555555", linestyle="--", linewidth=1)


def _interval_legend(alpha: float, extra: str = "") -> str:
    return " ".join(
        part
        for part in (
            f"Thick: {1 - alpha:.0%} CI for the average. Thin: {1 - alpha:.0%} "
            "prediction interval for one launch.",
            extra,
            "Ranges capped at the 100% ceiling (dashed).",
        )
        if part
    )


def _footnote(fig: plt.Figure, text: str) -> None:
    fig.text(0.99, 0.015, text, ha="right", fontsize=12, color="#555555")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
