"""Figures for the two-hidden-state CT-HMM comparison.

Palette and conventions follow :mod:`ppmi_subtype.plots`: the same surface,
ink and grid colours, the fixed state colours for label identity, and direct
labels on every mark rather than a colour-only encoding.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .plots import GRID, INK, INK_MUTED, STATE_COLORS, SURFACE, _finish, _style

#: Short names for the figures, so a 60-character model name does not become
#: the axis. The mapping is explicit rather than derived from the string.
SHORT = {
    "CT-HMM (2 hidden x 2 observed, Indeterminate excluded)": "M1: 2 hidden x 2 obs",
    "CT-HMM (2 hidden x 2 observed, no emission floor)": "M1 (no floor)",
    "CT-HMM (2 hidden x 3 observed)": "M2: 2 hidden x 3 obs",
    "CT-HMM (3 hidden x 3 observed, existing model)": "Existing: 3 hidden x 3 obs",
    "Persistence (last observed label)": "Persistence",
    "Marginal (training label prior)": "Marginal (null)",
}


def short(name: str) -> str:
    return SHORT.get(name, name)


def accuracy_intervals(table: pd.DataFrame, path: Path, title: str,
                       subtitle: str = "") -> Path:
    """Accuracy with its patient-bootstrap interval, one row per model.

    Models are ordered by accuracy. The interval is the bar, not an error
    whisker on a bar chart starting at zero: the differences here are a couple
    of percentage points on a baseline near 0.84, and a zero-based bar makes
    them invisible while implying a precision the interval denies.
    """
    t = table.sort_values("accuracy").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.4, 0.52 * len(t) + 1.9))
    _style(ax)

    y = np.arange(len(t))
    for k, row in t.iterrows():
        lo, hi = float(row["accuracy_ci_low"]), float(row["accuracy_ci_high"])
        rule = bool(row.get("rule_based_probabilities", False))
        colour = INK_MUTED if rule else STATE_COLORS["TD"]
        ax.plot([lo, hi], [k, k], color=colour, linewidth=2.4, solid_capstyle="round",
                alpha=0.55, zorder=2)
        ax.plot([row["accuracy"]], [k], marker="o", markersize=7.5, color=colour,
                zorder=3)
        ax.annotate(f"{row['accuracy']:.4f}  [{lo:.4f}, {hi:.4f}]",
                    (hi, k), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels([short(m) for m in t["model"]], fontsize=9)
    ax.set_xlabel("next-visit accuracy (95% patient bootstrap)", fontsize=9,
                  color=INK_MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)

    span = float(t["accuracy_ci_high"].max() - t["accuracy_ci_low"].min())
    ax.set_xlim(t["accuracy_ci_low"].min() - 0.04 * span,
                t["accuracy_ci_high"].max() + 0.30 * span)
    ax.set_title(title, fontsize=11, pad=14 if subtitle else 8, loc="left")
    if subtitle:
        ax.annotate(subtitle, (0, 1), xycoords="axes fraction",
                    xytext=(0, 9), textcoords="offset points",
                    fontsize=8.5, color=INK_MUTED)
    return _finish(fig, Path(path))


def emission_heatmaps(fits: dict, path: Path, order: list) -> Path:
    """Each model's emission matrix, rows hidden states, columns observed labels.

    This is the figure the Indeterminate question turns on: it shows directly
    how much probability each hidden state puts on an Indeterminate reading,
    and whether the two hidden states of Model 2 separated at all.
    """
    order = [m for m in order if m in fits]
    widths = [fits[m].emission_frame().shape[1] for m in order]
    fig, axes = plt.subplots(
        1, len(order), figsize=(2.1 * sum(widths) + 1.6, 2.9),
        gridspec_kw={"width_ratios": widths})
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, order):
        E = fits[model].emission_frame()
        _style(ax)
        ax.imshow(E.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=1,
                  aspect="auto")
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                v = float(E.iloc[i, j])
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                        color=SURFACE if v > 0.55 else INK)
        ax.set_xticks(range(E.shape[1]))
        ax.set_xticklabels(E.columns, fontsize=8.5, rotation=20, ha="right")
        ax.set_yticks(range(E.shape[0]))
        ax.set_yticklabels(E.index, fontsize=8.5)
        ax.set_title(short(model), fontsize=9.5, loc="left", pad=6)
        ax.grid(False)

    fig.suptitle("Learned emission probabilities  P(observed label | hidden state)",
                 fontsize=11, x=0.01, ha="left", y=1.06, color=INK)
    return _finish(fig, Path(path))


def calibration_curves(curves: dict, path: Path, title: str) -> Path:
    """Observed PIGD rate against predicted P_PIGD, on the common subset."""
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    _style(ax)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.4, zorder=1)

    palette = [STATE_COLORS["TD"], STATE_COLORS["PIGD"], STATE_COLORS["Indeterminate"],
               INK_MUTED]
    for (model, frame), colour in zip(curves.items(), palette):
        if not len(frame):
            continue
        ax.plot(frame["mean_P_PIGD"], frame["observed_rate_PIGD"], marker="o",
                markersize=5.5, linewidth=1.8, color=colour, label=short(model),
                zorder=3)

    ax.set_xlabel("mean predicted P(PIGD)", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("observed PIGD rate", fontsize=9, color=INK_MUTED)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for text in leg.get_texts():
        text.set_color(INK)
    ax.set_title(title, fontsize=11, loc="left", pad=8)
    return _finish(fig, Path(path))
