"""Diagnostic figures for the patient-level leave-one-out validation.

Every figure is built from the same prediction table, so no figure can disagree
with another, and none of them uses information the model did not have at the
moment it predicted.

Styling follows :mod:`ppmi_subtype.plots`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from .plots import (  # noqa: E402
    GRID,
    INK,
    INK_MUTED,
    SEQUENTIAL_BLUE,
    STATE_COLORS,
    STATE_ORDER,
    SURFACE,
    _finish,
    _style,
)

FAINT = "#c9c8c4"
GOOD = "#2a78d6"
BAD = "#eb6834"
STATES = list(STATE_ORDER)


def _bar_labels(ax, xs, heights, notes, fmt="{:.1%}", dy=0.018):
    for x, h, note in zip(xs, heights, notes):
        ax.text(x, h + dy, fmt.format(h), ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=INK)
        ax.text(x, dy * 0.6, note, ha="center", va="bottom",
                fontsize=8.5, color=INK_MUTED)


# ---------------------------------------------------------------------------
#  LOO-1  Confusion matrix
# ---------------------------------------------------------------------------
def confusion_matrix(cm: pd.DataFrame, path: Path, normalize: bool = False) -> Path:
    """Rows are predicted, columns actual. Normalised by row when asked."""
    counts = cm.reindex(index=STATES, columns=STATES, fill_value=0).to_numpy(float)
    totals = counts.sum(axis=1, keepdims=True)
    shown = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0) \
        if normalize else counts
    scale = shown / shown.max() if shown.max() else shown

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    _style(ax)
    ax.imshow(scale, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1, aspect="auto")

    for i in range(3):
        for j in range(3):
            dark = scale[i, j] > 0.55
            colour = "#ffffff" if dark else INK
            main = f"{shown[i, j]:.1%}" if normalize else f"{int(counts[i, j]):,}"
            ax.text(j, i - 0.10, main, ha="center", va="center",
                    color=colour, fontsize=14, fontweight="bold")
            sub = (f"n = {int(counts[i, j]):,}" if normalize
                   else (f"{counts[i, j] / totals[i, 0]:.1%} of row" if totals[i, 0] else ""))
            ax.text(j, 0.20 + i, sub, ha="center", va="center",
                    color=colour, fontsize=9, alpha=0.85)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=SURFACE, linewidth=2))

    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(STATES, fontsize=10); ax.set_yticklabels(STATES, fontsize=10)
    ax.set_xlabel("actual next recorded subtype", color=INK_MUTED, fontsize=10, labelpad=8)
    ax.set_ylabel("predicted next subtype", color=INK_MUTED, fontsize=10, labelpad=8)
    ax.set_title(
        "Patient-level LOO-CV: next-visit subtype prediction"
        + ("\nrow-normalised" if normalize else "\ncounts"),
        fontsize=12, fontweight="bold", pad=14, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.008, -0.05,
             "Each cell is one prediction opportunity. Rows sum to the number of times the model made that "
             "prediction.\nThe target is the recorded label, which itself carries measurement variability.",
             fontsize=8.5, color=INK_MUTED)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  LOO-2  Accuracy by current observed subtype
# ---------------------------------------------------------------------------
def accuracy_by_current_state(frame: pd.DataFrame, path: Path) -> Path:
    g = frame.groupby("current")["correct"].agg(["mean", "size"]).reindex(STATES)
    overall = float(frame["correct"].mean())

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    _style(ax)
    x = np.arange(3)
    ax.bar(x, g["mean"].to_numpy(), width=0.6,
           color=[STATE_COLORS[s] for s in STATES], edgecolor=SURFACE, linewidth=1.6)
    _bar_labels(ax, x, g["mean"].to_numpy(), [f"n = {int(v):,}" for v in g["size"]])

    ax.axhline(overall, color=INK, linewidth=1.4, linestyle="--")
    ax.text(2.44, overall + 0.015, f"overall {overall:.1%}", ha="right",
            fontsize=9.5, color=INK, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels([f"currently {s}" for s in STATES], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("next-visit prediction accuracy", color=INK_MUTED, fontsize=10)
    ax.set_title("Accuracy by the subtype the patient is recorded in now",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6); ax.set_axisbelow(True)
    fig.text(0.008, -0.07,
             "Grouped by the CURRENT recorded subtype at the moment of prediction, not by the outcome.",
             fontsize=8.5, color=INK_MUTED)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  LOO-3  Accuracy by amount of history
# ---------------------------------------------------------------------------
def accuracy_by_history(frame: pd.DataFrame, path: Path, cap: int = 5) -> Path:
    h = frame["n_history"].clip(upper=cap)
    g = frame.assign(bucket=h).groupby("bucket")["correct"].agg(["mean", "size"])
    labels = [f"{int(k)}" if k < cap else f"{cap}+" for k in g.index]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    _style(ax)
    x = np.arange(len(g))
    ax.bar(x, g["mean"].to_numpy(), width=0.6, color=GOOD,
           edgecolor=SURFACE, linewidth=1.6)
    _bar_labels(ax, x, g["mean"].to_numpy(), [f"n = {int(v):,}" for v in g["size"]])

    overall = float(frame["correct"].mean())
    ax.axhline(overall, color=INK, linewidth=1.4, linestyle="--")
    ax.text(len(g) - 0.56, overall + 0.015, f"overall {overall:.1%}", ha="right",
            fontsize=9.5, color=INK, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("number of the patient's own visits available as history",
                  color=INK_MUTED, fontsize=10)
    ax.set_ylabel("prediction accuracy", color=INK_MUTED, fontsize=10)
    ax.set_title("Does more of the patient's own history help?",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6); ax.set_axisbelow(True)
    fig.text(0.008, -0.09,
             f"One bar per history length; lengths above {cap} are pooled because each becomes sparse. "
             "Predicting a patient's\nsecond visit uses one prior observation, the third uses two, and so on. "
             "Longer history also means longer follow-up,\nso this comparison is not a controlled one.",
             fontsize=8.5, color=INK_MUTED)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  LOO-4  Calibration
# ---------------------------------------------------------------------------
def calibration_plot(cal: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    _style(ax)
    ax.plot([0.3, 1.0], [0.3, 1.0], color=FAINT, linewidth=1.6, linestyle="--",
            label="perfect calibration")
    ax.plot(cal["mean_confidence"], cal["observed_accuracy"], "-o",
            color=GOOD, linewidth=2.2, markersize=8, label="observed")

    for _, r in cal.iterrows():
        ax.annotate(f"n = {int(r['n']):,}",
                    xy=(r["mean_confidence"], r["observed_accuracy"]),
                    xytext=(7, -11), textcoords="offset points",
                    fontsize=8.5, color=INK_MUTED)

    ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("mean predicted confidence in the bin", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("observed accuracy in the bin", color=INK_MUTED, fontsize=10)
    ax.set_title("Are the model's probabilities meaningful?",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6); ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.008, -0.075,
             "Confidence is the largest of the three predicted probabilities. Points above the dashed line mean the "
             "model is\nunder-confident, below it means over-confident. Sparse bins were merged upwards.",
             fontsize=8.5, color=INK_MUTED)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  LOO-5  Confidence for correct vs incorrect
# ---------------------------------------------------------------------------
def confidence_correct_vs_incorrect(frame: pd.DataFrame, path: Path) -> Path:
    ok = frame.loc[frame["correct"], "confidence"].to_numpy()
    bad = frame.loc[~frame["correct"], "confidence"].to_numpy()
    bins = np.linspace(1 / 3, 1.0, 31)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    _style(ax)
    ax.hist(ok, bins=bins, density=True, color=GOOD, alpha=0.66,
            edgecolor=SURFACE, linewidth=0.7, label=f"correct  (n = {len(ok):,})")
    ax.hist(bad, bins=bins, density=True, color=BAD, alpha=0.66,
            edgecolor=SURFACE, linewidth=0.7, label=f"incorrect  (n = {len(bad):,})")

    for arr, colour, dy in ((ok, GOOD, 1.06), (bad, BAD, 0.97)):
        ax.axvline(arr.mean(), color=colour, linewidth=1.8)
        ax.text(arr.mean(), ax.get_ylim()[1] * dy, f" mean {arr.mean():.3f}",
                color=colour, fontsize=9.5, fontweight="bold", va="top")

    ax.set_xlabel("confidence (largest predicted probability)", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("density", color=INK_MUTED, fontsize=10)
    ax.set_title("Does the model know when it is uncertain?",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6); ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.008, -0.075,
             "If the two distributions separate, confidence carries information about whether the prediction can be "
             "trusted.\nIf they overlap, it does not.",
             fontsize=8.5, color=INK_MUTED)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  LOO-6  Probability assigned to the class that actually occurred
# ---------------------------------------------------------------------------
def probability_by_actual(frame: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    _style(ax)

    data, labels, colours = [], [], []
    for state in STATES:
        g = frame[frame["actual"] == state]
        data.append(g[f"p_{state}"].to_numpy())
        labels.append(f"actual {state}\nn = {len(g):,}")
        colours.append(STATE_COLORS[state])

    parts = ax.violinplot(data, positions=np.arange(3), widths=0.75,
                          showmedians=True, showextrema=False)
    for body, colour in zip(parts["bodies"], colours):
        body.set_facecolor(colour); body.set_alpha(0.55); body.set_edgecolor(colour)
    parts["cmedians"].set_color(INK); parts["cmedians"].set_linewidth(2.0)

    for k, arr in enumerate(data):
        ax.text(k, 1.03, f"median {np.median(arr):.2f}", ha="center",
                fontsize=9.5, fontweight="bold", color=colours[k])

    ax.axhline(1 / 3, color=FAINT, linewidth=1.4, linestyle="--")
    ax.text(2.42, 1 / 3 + 0.012, "chance (0.33)", ha="right", fontsize=9, color=INK_MUTED)

    ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("probability the model gave to the class\nthat actually occurred",
                  color=INK_MUTED, fontsize=10)
    ax.set_title("Does the model favour the subtype that actually turns up?",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6); ax.set_axisbelow(True)
    fig.text(0.008, -0.10,
             "Each violin shows, for the visits whose recorded subtype turned out to be that class, the probability "
             "the model had\nassigned to it beforehand. Mass well above 0.33 means the model was pointing the right "
             "way.",
             fontsize=8.5, color=INK_MUTED)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  LOO-7  Accuracy by transition type
# ---------------------------------------------------------------------------
def transition_accuracy_plot(table: pd.DataFrame, path: Path) -> Path:
    df = table.copy().sort_values("n", ascending=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    _style(ax)

    y = np.arange(len(df))
    colours = [STATE_COLORS[c] for c in df["current"]]
    acc = df["accuracy"].fillna(0.0).to_numpy()
    ax.barh(y, acc, height=0.68, color=colours, edgecolor=SURFACE, linewidth=1.5)

    for k, (_, r) in enumerate(df.iterrows()):
        value = "no cases" if r["n"] == 0 else f"{r['accuracy']:.0%}"
        ax.text(max(acc[k], 0) + 0.015, k, f"{value}   (n = {int(r['n']):,})",
                va="center", fontsize=9.5, color=INK,
                fontweight="bold" if r["n"] >= 100 else "normal")

    ax.set_yticks(y); ax.set_yticklabels(df["transition"], fontsize=10)
    ax.set_xlim(0, 1.30)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("share of these transitions the model predicted correctly",
                  color=INK_MUTED, fontsize=10)
    ax.set_title("Which transitions does the model get right?",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.6); ax.set_axisbelow(True)
    # Outside the axes: with the full dataset every row carries a bar, so any
    # in-axes corner would sit on top of one.
    ax.legend(handles=[Patch(facecolor=STATE_COLORS[s], label=f"from {s}") for s in STATES],
              frameon=False, fontsize=9.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=3, labelcolor=INK)
    fig.text(0.008, -0.16,
             "Every from-to combination is shown, including the rare ones, with its sample size. Bar colour is the "
             "subtype the\npatient was in. A combination the model never predicts scores zero here however often "
             "it occurs.",
             fontsize=8.5, color=INK_MUTED)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  LOO-8  Observed against predicted next-subtype counts
# ---------------------------------------------------------------------------
def observed_vs_predicted(frame: pd.DataFrame, path: Path) -> Path:
    observed = (pd.crosstab(frame["current"], frame["actual"])
                .reindex(index=STATES, columns=STATES, fill_value=0))
    predicted = (pd.crosstab(frame["current"], frame["predicted"])
                 .reindex(index=STATES, columns=STATES, fill_value=0))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
    vmax = max(observed.to_numpy().max(), predicted.to_numpy().max())

    for ax, mat, title in ((axes[0], observed, "Actually recorded"),
                           (axes[1], predicted, "Predicted by the model")):
        _style(ax)
        arr = mat.to_numpy(float)
        ax.imshow(arr / vmax, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1, aspect="auto")
        rows = arr.sum(axis=1, keepdims=True)
        for i in range(3):
            for j in range(3):
                dark = arr[i, j] / vmax > 0.55
                colour = "#ffffff" if dark else INK
                ax.text(j, i - 0.10, f"{int(arr[i, j]):,}", ha="center", va="center",
                        color=colour, fontsize=13, fontweight="bold")
                if rows[i, 0]:
                    ax.text(j, i + 0.20, f"{arr[i, j] / rows[i, 0]:.0%}", ha="center",
                            va="center", color=colour, fontsize=9, alpha=0.85)
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=SURFACE, linewidth=2))
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(STATES, fontsize=9.5); ax.set_yticklabels(STATES, fontsize=9.5)
        ax.set_xlabel("next subtype", color=INK_MUTED, fontsize=10)
        ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left", pad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("current subtype", color=INK_MUTED, fontsize=10)

    fig.suptitle("Does the model reproduce the overall transition behaviour?",
                 fontsize=12.5, fontweight="bold", x=0.008, ha="left", y=1.04)
    fig.text(0.008, -0.075,
             "Same rows, same columns, same totals. A model that never names a destination will show an empty "
             "column on the right\neven though that column is populated on the left.",
             fontsize=8.5, color=INK_MUTED)
    fig.tight_layout()
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  LOO-9  Per-patient accuracy
# ---------------------------------------------------------------------------
def patient_accuracy_distribution(summary: pd.DataFrame, path: Path) -> Path:
    single = summary[summary["n_predictions"] == 1]
    multi = summary[summary["n_predictions"] > 1]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    _style(ax)
    bins = np.linspace(0, 1, 21)
    ax.hist([multi["accuracy"], single["accuracy"]], bins=bins, stacked=True,
            color=[GOOD, FAINT], edgecolor=SURFACE, linewidth=0.8,
            label=[f"2 or more predictions  (n = {len(multi):,})",
                   f"exactly 1 prediction, so only 0% or 100% possible  (n = {len(single):,})"])

    med = float(summary["accuracy"].median())
    ax.axvline(med, color=INK, linewidth=1.8)
    # Keep the label on whichever side of the line has room; the tall bar sits
    # at 100% whenever the median is high.
    side, offset = ("right", -0.012) if med > 0.5 else ("left", 0.012)
    ax.text(med + offset, ax.get_ylim()[1] * 0.94, f"median {med:.0%}", ha=side,
            fontsize=9.5, fontweight="bold", color=INK)

    ax.set_xlabel("per-patient accuracy", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("number of patients", color=INK_MUTED, fontsize=10)
    ax.set_title("How much does performance vary from patient to patient?",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6); ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.008, -0.20,
             "The spikes at 0% and 100% are largely an artefact of short follow-up: a patient with two labelled "
             "visits offers a single\nprediction, which is either right or wrong.",
             fontsize=8.5, color=INK_MUTED)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  LOO-10  Accuracy against number of visits
# ---------------------------------------------------------------------------
def accuracy_vs_visits(summary: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    _style(ax)

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.16, 0.16, len(summary))
    ax.scatter(summary["n_labelled_visits"] + jitter, summary["accuracy"],
               s=13, color=GOOD, alpha=0.28, linewidth=0)

    g = summary.groupby("n_labelled_visits")["accuracy"].agg(["mean", "size"])
    g = g[g["size"] >= 10]
    ax.plot(g.index, g["mean"], "-o", color=INK, linewidth=2.2, markersize=6,
            label="mean per-patient accuracy")
    for v, r in g.iterrows():
        ax.text(v, -0.075, f"{int(r['size']):,}", ha="center", fontsize=8, color=INK_MUTED)
    ax.text(g.index.min() - 0.9, -0.075, "n =", ha="right", fontsize=8, color=INK_MUTED)

    ax.set_xlim(1.4, summary["n_labelled_visits"].max() + 0.6)
    ax.set_ylim(-0.10, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("number of labelled visits the patient has", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("per-patient accuracy", color=INK_MUTED, fontsize=10)
    ax.set_title("Accuracy against length of follow-up",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.5); ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.008, -0.085,
             "One faint point per patient, jittered horizontally. The line averages patients with the same number "
             "of visits, shown only\nwhere at least ten patients contribute. This is an association, not a cause: "
             "patients who attend longer differ in other ways.",
             fontsize=8.5, color=INK_MUTED)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  LOO-11  Accuracy over follow-up
# ---------------------------------------------------------------------------
def accuracy_over_followup(frame: pd.DataFrame, path: Path, min_n: int = 40) -> Path:
    year = frame["current_years"].astype(float).round().astype(int)
    g = frame.assign(year=year).groupby("year")["correct"].agg(["mean", "size"])
    g = g[g["size"] >= min_n]

    fig, (ax, ax_n) = plt.subplots(2, 1, figsize=(8.6, 5.4),
                                   height_ratios=[4, 1], sharex=True)
    _style(ax); _style(ax_n)

    ax.plot(g.index, g["mean"], "-o", color=GOOD, linewidth=2.4, markersize=7)
    overall = float(frame["correct"].mean())
    ax.axhline(overall, color=INK, linewidth=1.4, linestyle="--")
    ax.text(g.index.max(), overall + 0.015, f"overall {overall:.1%}", ha="right",
            fontsize=9.5, color=INK, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("prediction accuracy", color=INK_MUTED, fontsize=10)
    ax.set_title("Does accuracy hold up later in follow-up?",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6); ax.set_axisbelow(True)

    ax_n.bar(g.index, g["size"], color=FAINT, edgecolor=SURFACE, linewidth=1.4)
    ax_n.set_ylabel("predictions", color=INK_MUTED, fontsize=9)
    ax_n.set_xlabel("years since the patient's first visit, at the moment of prediction",
                    color=INK_MUTED, fontsize=10)
    ax_n.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6); ax_n.set_axisbelow(True)

    fig.text(0.008, -0.055,
             f"Grouped by the time of the CURRENT visit, rounded to the nearest year. Years contributing fewer than "
             f"{min_n} predictions are\nnot drawn; the count panel is part of the reading.",
             fontsize=8.5, color=INK_MUTED)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  LOO-12  Example held-out patients
# ---------------------------------------------------------------------------
def example_trajectories(frame: pd.DataFrame, chosen: list, path: Path,
                         notes: dict | None = None) -> Path:
    notes = notes or {}
    n = len(chosen)
    fig, axes = plt.subplots(n, 1, figsize=(10.2, 2.35 * n), sharex=False)
    axes = np.atleast_1d(axes)

    for ax, patno in zip(axes, chosen):
        _style(ax)
        g = frame[frame["PATNO"] == patno].sort_values("visit_index")
        t = g["target_years"].to_numpy(float)

        base = np.zeros(len(g))
        for state in STATES:
            share = g[f"p_{state}"].to_numpy(float)
            ax.bar(t, share, bottom=base, width=0.34, color=STATE_COLORS[state],
                   edgecolor=SURFACE, linewidth=0.8, alpha=0.92)
            base += share

        for k, (_, r) in enumerate(g.iterrows()):
            ax.scatter([t[k]], [1.10], s=64, color=STATE_COLORS[r["actual"]],
                       edgecolor=SURFACE, linewidth=1.3, zorder=5, clip_on=False)
            ax.text(t[k], 1.26, "OK" if r["correct"] else "X", ha="center", fontsize=8.5,
                    color=INK_MUTED if r["correct"] else BAD,
                    fontweight="normal" if r["correct"] else "bold")

        start = frame[frame["PATNO"] == patno].sort_values("visit_index").iloc[0]
        ax.scatter([start["current_years"]], [1.10], s=64,
                   color=STATE_COLORS[start["current"]], edgecolor=SURFACE,
                   linewidth=1.3, zorder=5, clip_on=False)

        ax.set_ylim(0, 1.0); ax.set_xlim(-0.4, max(t.max(), 1.0) + 0.6)
        ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", ".5", "1"], fontsize=8.5)
        ax.set_ylabel("P(next)", color=INK_MUTED, fontsize=9)
        acc = float(g["correct"].mean())
        label = f"{int(patno)}" if float(patno).is_integer() else f"{patno}"
        ax.set_title(f"patient {label} — {notes.get(patno, '')}   ({acc:.0%} of "
                     f"{len(g)} predictions correct)",
                     fontsize=10, fontweight="bold", loc="left", pad=40)
        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.5); ax.set_axisbelow(True)

    axes[-1].set_xlabel("years since the patient's first visit", color=INK_MUTED, fontsize=10)
    fig.suptitle("What the model does visit by visit, for held-out patients",
                 fontsize=12.5, fontweight="bold", x=0.008, ha="left", y=1.005)
    handles = [Patch(facecolor=STATE_COLORS[s], label=s) for s in STATES]
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=FAINT,
                          markersize=9, label="dot above = subtype actually recorded"))
    leg = fig.legend(handles=handles, frameon=False, fontsize=9, ncol=4,
                     loc="upper center", bbox_to_anchor=(0.5, -0.005))
    for t_ in leg.get_texts():
        t_.set_color(INK)
    fig.text(0.008, -0.045,
             "Each stacked bar is the probability the model gave to the three subtypes for that visit, using only "
             "earlier visits.\nThe dot above the bar is what was actually recorded; OK or X marks whether the "
             "most likely class was right.",
             fontsize=8.5, color=INK_MUTED)
    fig.tight_layout()
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  Model comparison
# ---------------------------------------------------------------------------
def model_comparison(table: pd.DataFrame, path: Path) -> Path:
    df = table.sort_values("accuracy", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    _style(ax)

    y = np.arange(len(df))
    colours = [GOOD if m == "HMM" else FAINT for m in df["model"]]
    ax.barh(y, df["accuracy"], height=0.6, color=colours,
            edgecolor=SURFACE, linewidth=1.5)

    err_low = df["accuracy"] - df["accuracy_ci_patient_bootstrap_low"]
    err_high = df["accuracy_ci_patient_bootstrap_high"] - df["accuracy"]
    ax.errorbar(df["accuracy"], y, xerr=[err_low, err_high], fmt="none",
                ecolor=INK, elinewidth=1.4, capsize=4)

    for k, r in df.iterrows():
        ax.text(r["accuracy_ci_patient_bootstrap_high"] + 0.008, k,
                f"{r['accuracy']:.1%}  "
                f"[{r['accuracy_ci_patient_bootstrap_low']:.1%}, "
                f"{r['accuracy_ci_patient_bootstrap_high']:.1%}]",
                va="center", fontsize=9.5, color=INK,
                fontweight="bold" if r["model"] == "HMM" else "normal")

    ax.set_yticks(y); ax.set_yticklabels(df["model"], fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("accuracy predicting the next recorded subtype",
                  color=INK_MUTED, fontsize=10)
    ax.set_title("Patient-level LOO-CV: every model on the same predictions",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.6); ax.set_axisbelow(True)
    fig.text(0.008, -0.10,
             "Identical held-out patients and identical prediction opportunities for every model. Bars carry 95% "
             "intervals from\nresampling patients, which is wider than resampling predictions because a patient's "
             "visits are correlated.",
             fontsize=8.5, color=INK_MUTED)
    return _finish(fig, path)
