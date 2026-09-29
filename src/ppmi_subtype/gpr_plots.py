"""Figures for the Step 3 trajectory analysis.

Every figure answers one stated question, and each is built from the same
prediction table the metrics come from, so a figure cannot disagree with the
numbers beside it. Nothing here uses information a model did not have when it
predicted.

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
from matplotlib.patches import Patch  # noqa: E402

from .plots import (  # noqa: E402
    GRID,
    INK,
    INK_MUTED,
    STATE_COLORS,
    STATE_ORDER,
    SURFACE,
    _finish,
    _style,
)

FAINT = "#c9c8c4"
GP_COLOR = "#2a78d6"
OBS_COLOR = "#0b0b0b"
FUTURE_COLOR = "#eb6834"
BAND = "#9dc3ef"

#: Fixed colour per model so a model keeps its identity across figures.
MODEL_COLORS = {
    "K0": "#7b9fd4", "K1": "#2a78d6", "K2": "#14509e", "K3": "#6f42c1",
    "hbayes": "#1baf7a", "persistence": "#eb6834", "patient_mean": "#c2410c",
    "population": "#8a8987", "patient_ols": "#b8912f",
}

TD_THRESHOLD = 1.15
PIGD_THRESHOLD = 0.90


def _band(ax, lo=PIGD_THRESHOLD, hi=TD_THRESHOLD):
    """Shade the Indeterminate band, the target every model is aiming into."""
    ax.axhspan(lo, hi, color=STATE_COLORS["Indeterminate"], alpha=0.10, zorder=0)
    ax.axhline(hi, color=STATE_COLORS["TD"], lw=0.9, ls="--", alpha=0.55, zorder=1)
    ax.axhline(lo, color=STATE_COLORS["PIGD"], lw=0.9, ls="--", alpha=0.55, zorder=1)


# ---------------------------------------------------------------------------
#  1-2  The ratio, and how it is distributed across patients
# ---------------------------------------------------------------------------
def ratio_distribution(traj: pd.DataFrame, path: Path) -> Path:
    """Where do the ratio values sit, and how much of the scale is unusable?

    The two spikes matter more than the shape between them: the mass at zero is
    patients with no tremor at all, and the bar off to the right counts visits
    with no finite ratio, which no continuous model can use.
    """
    finite = traj.loc[traj["ratio_defined"], "ratio"].to_numpy(float)
    n_undef = int((~traj["ratio_defined"]).sum())

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.6, 4.8), gridspec_kw={"width_ratios": [3.4, 1]}
    )
    fig.patch.set_facecolor(SURFACE)

    ax.hist(finite, bins=np.linspace(0, 12, 97), color=GP_COLOR, alpha=0.85,
            edgecolor="white", linewidth=0.4)
    _band(ax, PIGD_THRESHOLD, TD_THRESHOLD)
    ax.set_xlabel("TD/PIGD ratio at a visit")
    ax.set_ylabel("visits")
    ax.set_title("The continuous ratio, finite values only", loc="left",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-0.2, 12)
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)

    n_zero = int((finite == 0).sum())
    ax.annotate(
        f"{n_zero} visits at exactly 0\n(no tremor recorded)",
        xy=(0.05, n_zero * 0.92), xytext=(1.9, n_zero * 0.82),
        fontsize=9, color=INK_MUTED,
        arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.9),
    )
    ax.text(
        (PIGD_THRESHOLD + TD_THRESHOLD) / 2, ax.get_ylim()[1] * 0.97,
        "Indeterminate\n0.90-1.15", ha="center", va="top", fontsize=8.5,
        color=INK_MUTED,
    )

    ax2.bar([0, 1], [len(finite), n_undef],
            color=[GP_COLOR, FAINT], edgecolor="white", width=0.62)
    for x, v in zip([0, 1], [len(finite), n_undef]):
        ax2.text(x, v + len(traj) * 0.012, f"{v:,}\n{v/len(traj):.1%}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["finite\nratio", "undefined\n(PIGD = 0)"], fontsize=9)
    ax2.set_ylabel("visits")
    ax2.set_ylim(0, len(traj) * 1.12)
    ax2.set_title("What a ratio model can see", loc="left", fontsize=12,
                  fontweight="bold")
    _style(ax2)
    ax2.grid(axis="y", color=GRID, lw=0.7)
    ax2.set_axisbelow(True)

    fig.suptitle(
        "The ratio is undefined whenever the PIGD subscore is zero, and those "
        "visits are overwhelmingly TD",
        x=0.5, y=0.015, fontsize=9.5, color=INK_MUTED,
    )
    return _finish(fig, path)


def patient_ratio_statistics(summary: pd.DataFrame, path: Path) -> Path:
    """Is a patient's ratio a stable personal level, or does it roam?

    If the per-patient range were small next to the spread of medians, the
    ratio would be close to a fixed trait and the forecasting problem would be
    mostly about identifying each patient's level.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    fig.patch.set_facecolor(SURFACE)

    panels = [
        ("ratio_median", "Patient median ratio", GP_COLOR,
         "where each patient sits"),
        ("ratio_range", "Within-patient range (max - min)", FUTURE_COLOR,
         "how far each patient moves"),
        ("n_finite", "Finite-ratio visits per patient", "#1baf7a",
         "how much history there is"),
    ]
    for ax, (col, title, colour, note) in zip(axes, panels):
        values = summary[col].dropna().to_numpy(float)
        if col == "n_finite":
            bins = np.arange(-0.5, values.max() + 1.5)
        else:
            bins = np.linspace(0, np.nanpercentile(values, 99.5), 40)
        ax.hist(values, bins=bins, color=colour, alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.axvline(np.median(values), color=INK, lw=1.4, ls="--")
        ax.text(np.median(values), ax.get_ylim()[1] * 0.95,
                f"  median {np.median(values):.2f}", fontsize=9, color=INK,
                va="top")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel(note, fontsize=9, color=INK_MUTED)
        ax.set_ylabel("patients")
        _style(ax)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)

    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  3-5  Trajectories, extremes, filtering
# ---------------------------------------------------------------------------
def example_trajectories(traj: pd.DataFrame, path: Path, n: int = 12,
                         seed: int = 42) -> Path:
    """What does an individual patient's ratio actually look like over time?"""
    finite = traj[traj["ratio_defined"]]
    counts = finite.groupby("PATNO").size()
    eligible = counts[counts >= 4].index.to_numpy()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=min(n, len(eligible)), replace=False)

    rows = int(np.ceil(len(chosen) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(13.4, 2.65 * rows), sharex=False)
    fig.patch.set_facecolor(SURFACE)

    for ax, patno in zip(np.atleast_1d(axes).ravel(), chosen):
        g = finite[finite["PATNO"] == patno].sort_values("years_from_baseline")
        _band(ax)
        ax.plot(g["years_from_baseline"], g["ratio"], "-o", color=OBS_COLOR,
                lw=1.4, ms=4.2, zorder=3)
        ax.set_title(f"patient {patno}", loc="left", fontsize=9.5, color=INK_MUTED)
        _style(ax)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)

    for ax in np.atleast_1d(axes).ravel()[len(chosen):]:
        ax.set_visible(False)

    fig.supxlabel("years from the patient's first visit", fontsize=9.5,
                  color=INK_MUTED)
    fig.supylabel("TD/PIGD ratio", fontsize=9.5, color=INK_MUTED)
    fig.suptitle(
        "Individual ratio trajectories, patients with at least four finite visits",
        x=0.5, y=0.995, fontsize=12, fontweight="bold", color=INK, ha="center",
    )
    return _finish(fig, path)


def extreme_trajectories(traj: pd.DataFrame, extreme: pd.DataFrame,
                         criterion: str, path: Path, n: int = 8) -> Path:
    """Are the flagged patients persistently extreme, or just occasionally?

    The point of showing them is falsifiable: if these look like ordinary
    trajectories with one stray visit, the criterion is doing the wrong thing.
    """
    flagged = extreme.loc[extreme[criterion], "PATNO"]
    finite = traj[traj["ratio_defined"]]
    counts = traj[traj["PATNO"].isin(flagged)].groupby("PATNO").size()
    chosen = counts.sort_values(ascending=False).head(n).index

    fig, axes = plt.subplots(2, 4, figsize=(13.4, 5.6))
    fig.patch.set_facecolor(SURFACE)

    for ax, patno in zip(axes.ravel(), chosen):
        g = traj[traj["PATNO"] == patno].sort_values("years_from_baseline")
        _band(ax)
        usable = g[g["ratio_defined"]]
        ax.plot(usable["years_from_baseline"], usable["ratio"], "-o",
                color=OBS_COLOR, lw=1.3, ms=4.2, zorder=3)
        missing = g[~g["ratio_defined"]]
        if len(missing):
            ax.plot(missing["years_from_baseline"],
                    np.full(len(missing), ax.get_ylim()[1] * 0.95), "x",
                    color=FUTURE_COLOR, ms=6, zorder=4)
        ax.set_title(f"patient {patno}", loc="left", fontsize=9.5, color=INK_MUTED)
        _style(ax)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)

    for ax in axes.ravel()[len(chosen):]:
        ax.set_visible(False)

    fig.legend(
        handles=[
            Line2D([], [], color=OBS_COLOR, marker="o", lw=1.3, label="finite ratio"),
            Line2D([], [], color=FUTURE_COLOR, marker="x", lw=0,
                   label="ratio undefined (PIGD subscore 0)"),
        ],
        loc="lower center", ncol=2, frameon=False, fontsize=9.5,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        f"Trajectories flagged by criterion {criterion}, longest first",
        x=0.5, y=0.99, fontsize=12, fontweight="bold", color=INK, ha="center",
    )
    return _finish(fig, path)


def filtering_effect(comparison: pd.DataFrame, path: Path) -> Path:
    """What does each criterion cost, and does it change the task?

    The second panel is the one that matters. A criterion that removes mostly
    patients who never changed subtype leaves an easier-looking cohort behind,
    so its accuracy cannot be read against the unfiltered one.
    """
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6))
    fig.patch.set_facecolor(SURFACE)

    df = comparison.copy()
    labels = df["cohort"].tolist()
    x = np.arange(len(df))

    ax.bar(x - 0.19, df["patients"], width=0.36, color=GP_COLOR, label="patients kept")
    ax.bar(x + 0.19, df["patients_removed"], width=0.36, color=FAINT,
           label="patients removed")
    for i, (kept, gone) in enumerate(zip(df["patients"], df["patients_removed"])):
        ax.text(i - 0.19, kept + 12, f"{kept:,}", ha="center", fontsize=9.5,
                fontweight="bold", color=INK)
        ax.text(i + 0.19, gone + 12, f"{gone:,}", ha="center", fontsize=9.5,
                color=INK_MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("patients")
    ax.set_title("Cohort size after each criterion", loc="left", fontsize=11.5,
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)

    share = df["never_changed"] / (df["never_changed"] + df["ever_changed"])
    ax2.bar(x, share, width=0.5, color=FUTURE_COLOR, alpha=0.9)
    for i, v in enumerate(share):
        ax2.text(i, v + 0.008, f"{v:.1%}", ha="center", fontsize=10,
                 fontweight="bold", color=INK)
    ax2.axhline(share.iloc[0], color=INK, lw=1.1, ls="--")
    ax2.text(len(df) - 0.5, share.iloc[0] + 0.006, "unfiltered level",
             ha="right", fontsize=8.5, color=INK_MUTED)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylabel("share of remaining patients")
    ax2.set_title("Patients who never changed subtype", loc="left",
                  fontsize=11.5, fontweight="bold")
    _style(ax2)
    ax2.grid(axis="y", color=GRID, lw=0.7)
    ax2.set_axisbelow(True)

    fig.suptitle(
        "Filtering changes who is left, and with them how hard the prediction is",
        x=0.5, y=0.015, fontsize=9.5, color=INK_MUTED,
    )
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  6-7  Predicted trajectories
# ---------------------------------------------------------------------------
def example_forecast(steps: pd.DataFrame, traj: pd.DataFrame, model: str,
                     path: Path, n: int = 6, seed: int = 42) -> Path:
    """Does a forecast look right for one patient, point by point?

    History, the predicted mean with its band, and what actually happened. A
    summary statistic cannot show whether the band is the right width.
    """
    g_all = steps[steps["model"] == model]
    if g_all.empty:
        raise ValueError(f"no multi-step forecasts for model {model!r}")

    longest = (
        g_all.groupby("held_out_patient")["step"].max().sort_values(ascending=False)
    )
    rng = np.random.default_rng(seed)
    pool = longest[longest >= 2].index.to_numpy()
    chosen = (rng.choice(pool, size=min(n, len(pool)), replace=False)
              if len(pool) else longest.index[:n])

    finite = traj[traj["ratio_defined"]]
    rows = int(np.ceil(len(chosen) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(13.4, 3.5 * rows))
    fig.patch.set_facecolor(SURFACE)

    for ax, patno in zip(np.atleast_1d(axes).ravel(), chosen):
        g = g_all[g_all["held_out_patient"] == patno].sort_values("t_target")
        hist = finite[finite["PATNO"] == patno].sort_values("years_from_baseline")
        m = int(g["history_length"].iloc[0])

        _band(ax)
        ax.plot(hist["years_from_baseline"].iloc[:m], hist["ratio"].iloc[:m],
                "-o", color=OBS_COLOR, lw=1.5, ms=5, label="history (given)", zorder=4)
        ax.fill_between(g["t_target"], g["pred_mean"] - 1.96 * g["pred_sd"],
                        g["pred_mean"] + 1.96 * g["pred_sd"],
                        color=BAND, alpha=0.45, zorder=2, label="95% predictive")
        ax.plot(g["t_target"], g["pred_mean"], "-", color=GP_COLOR, lw=2.0,
                zorder=3, label="predicted mean")
        ax.plot(g["t_target"], g["observed_ratio"], "o", color=FUTURE_COLOR,
                ms=6, zorder=5, label="what happened")
        ax.set_title(f"patient {patno}", loc="left", fontsize=10, color=INK_MUTED)
        _style(ax)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)

    for ax in np.atleast_1d(axes).ravel()[len(chosen):]:
        ax.set_visible(False)

    handles, labels_ = np.atleast_1d(axes).ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.008))
    fig.supxlabel("years from the patient's first visit", fontsize=9.5,
                  color=INK_MUTED)
    fig.supylabel("TD/PIGD ratio", fontsize=9.5, color=INK_MUTED)
    fig.suptitle(
        f"{model}: forecasting the back half of a trajectory from the front half",
        x=0.5, y=0.995, fontsize=12, fontweight="bold", color=INK, ha="center",
    )
    return _finish(fig, path)


def subtype_probability_trajectories(steps: pd.DataFrame, model: str, path: Path,
                                     n: int = 4, seed: int = 7) -> Path:
    """What does the forecast say about subtype, as a probability over time?

    The clinical translation of the continuous forecast. Probabilities come
    from the whole predictive distribution, so a prediction sitting in the
    middle of the range supports all three classes at once.
    """
    g_all = steps[steps["model"] == model]
    longest = g_all.groupby("held_out_patient")["step"].max().sort_values(ascending=False)
    rng = np.random.default_rng(seed)
    pool = longest[longest >= 3].index.to_numpy()
    chosen = (rng.choice(pool, size=min(n, len(pool)), replace=False)
              if len(pool) else longest.index[:n])

    fig, axes = plt.subplots(1, len(chosen), figsize=(3.4 * len(chosen), 4.2),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for ax, patno in zip(axes[0], chosen):
        g = g_all[g_all["held_out_patient"] == patno].sort_values("t_target")
        ax.stackplot(
            g["t_target"],
            [g[f"p_{s}"] for s in STATE_ORDER],
            colors=[STATE_COLORS[s] for s in STATE_ORDER],
            alpha=0.85,
        )
        for _, row in g.iterrows():
            ax.plot(row["t_target"], 1.04, "v",
                    color=STATE_COLORS[row["observed_subtype"]], ms=7,
                    clip_on=False)
        ax.set_ylim(0, 1)
        ax.set_title(f"patient {patno}", loc="left", fontsize=10, color=INK_MUTED)
        _style(ax)

    fig.legend(
        handles=[Patch(facecolor=STATE_COLORS[s], label=s) for s in STATE_ORDER]
        + [Line2D([], [], marker="v", lw=0, color=INK_MUTED,
                  label="what was actually recorded")],
        loc="lower center", ncol=4, frameon=False, fontsize=9.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.supxlabel("years from the patient's first visit", fontsize=9.5,
                  color=INK_MUTED)
    fig.suptitle(
        f"{model}: subtype probabilities derived from the predicted ratio",
        x=0.5, y=0.99, fontsize=12, fontweight="bold", color=INK, ha="center",
    )
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  8-9  Where the error comes from
# ---------------------------------------------------------------------------
def _line_panel(ax, df, xcol, models, title, xlabel):
    for model in models:
        g = df[df["model"] == model].copy()
        if g.empty:
            continue
        ax.plot(range(len(g)), g["MAE"], "-o", lw=1.8, ms=4.5,
                color=MODEL_COLORS.get(model, INK_MUTED), label=model)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g[xcol], fontsize=8.5, rotation=0)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel("mean absolute error (ratio units)")
    _style(ax)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def error_by_stratum(stratified: pd.DataFrame, cohort: str, models: list,
                     path: Path) -> Path:
    """Does more history help, and does a longer horizon hurt?

    A trajectory model earns its name by improving as the patient's history
    grows. If the lines are flat in the left panel, the model is not learning
    anything patient-specific from the extra visits.
    """
    df = stratified[stratified["cohort"] == cohort]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8))
    fig.patch.set_facecolor(SURFACE)

    _line_panel(ax, df[df["by"] == "history_length"].sort_values("stratum"),
                "stratum", models, "Error against amount of history",
                "observations available before the target")
    _line_panel(ax2, df[df["by"] == "horizon_years"], "stratum", models,
                "Error against prediction horizon", "years ahead")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    return _finish(fig, path)


def predicted_vs_observed(predictions: pd.DataFrame, model: str, path: Path) -> Path:
    """Is the model tracking individuals, or predicting the cohort average?

    Points hugging a horizontal line mean the model is saying roughly the same
    thing about everyone; points on the diagonal mean it is tracking them.
    """
    g = predictions[predictions["model"] == model]
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    fig.patch.set_facecolor(SURFACE)

    ax.scatter(g["observed_ratio"], g["pred_mean"], s=13, alpha=0.30,
               color=GP_COLOR, edgecolors="none")
    lim = [0, float(max(g["observed_ratio"].max(), g["pred_mean"].max())) * 1.03]
    ax.plot(lim, lim, "--", color=INK, lw=1.2, label="perfect prediction")
    ax.axhline(g["pred_mean"].mean(), color=FUTURE_COLOR, lw=1.2, ls=":",
               label="predicting the mean every time")

    r = float(np.corrcoef(g["observed_ratio"], g["pred_mean"])[0, 1])
    mae = float(np.mean(np.abs(g["pred_mean"] - g["observed_ratio"])))
    ax.text(0.03, 0.97, f"r = {r:.3f}\nMAE = {mae:.3f}\nn = {len(g):,}",
            transform=ax.transAxes, va="top", fontsize=10.5, color=INK,
            fontweight="bold")

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("observed ratio at the next visit")
    ax.set_ylabel("predicted mean")
    ax.set_title(f"{model}: predicted against observed", loc="left",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    return _finish(fig, path)


def calibration(predictions: pd.DataFrame, models: list, path: Path) -> Path:
    """Is the stated uncertainty honest?

    Left: does a claimed 95% interval contain the truth 95% of the time. Right:
    the PIT histogram, which is flat when the predictive distribution has the
    right shape, humped when it is too wide and U-shaped when it is too narrow.
    """
    from scipy.stats import norm as _norm

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    fig.patch.set_facecolor(SURFACE)

    levels = np.linspace(0.05, 0.99, 25)
    for model in models:
        g = predictions[predictions["model"] == model]
        if g.empty:
            continue
        z = (g["observed_ratio"] - g["pred_mean"]) / g["pred_sd"].clip(lower=1e-9)
        empirical = [float(np.mean(np.abs(z) <= _norm.ppf(0.5 + level / 2)))
                     for level in levels]
        ax.plot(levels, empirical, "-", lw=1.8,
                color=MODEL_COLORS.get(model, INK_MUTED), label=model)

    ax.plot([0, 1], [0, 1], "--", color=INK, lw=1.2)
    ax.set_xlabel("stated interval level")
    ax.set_ylabel("proportion actually inside")
    ax.set_title("Interval calibration", loc="left", fontsize=11.5,
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    _style(ax)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)

    primary = models[0]
    g = predictions[predictions["model"] == primary]
    pit = _norm.cdf(g["observed_ratio"], loc=g["pred_mean"],
                    scale=g["pred_sd"].clip(lower=1e-9))
    ax2.hist(pit, bins=20, range=(0, 1), color=GP_COLOR, alpha=0.85,
             edgecolor="white", linewidth=0.5)
    ax2.axhline(len(pit) / 20, color=INK, lw=1.3, ls="--",
                label="flat = correctly shaped")
    ax2.set_xlabel("probability integral transform")
    ax2.set_ylabel("predictions")
    ax2.set_title(f"{primary}: PIT histogram", loc="left", fontsize=11.5,
                  fontweight="bold")
    ax2.legend(frameon=False, fontsize=9)
    _style(ax2)
    ax2.grid(axis="y", color=GRID, lw=0.7)
    ax2.set_axisbelow(True)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
#  12  Model comparison
# ---------------------------------------------------------------------------
def model_comparison(metrics: pd.DataFrame, cohort: str, path: Path) -> Path:
    """Does the flexible GP beat the linear model, and do either beat the trivial ones?

    The comparison the step exists to make. The reference line is the best
    baseline: a model that cannot clear it has not shown that a trajectory
    model is doing anything a rule of thumb could not.
    """
    df = metrics[metrics["cohort"] == cohort].sort_values("MAE")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0))
    fig.patch.set_facecolor(SURFACE)

    from .gpr_loo import BASELINES

    colours = [MODEL_COLORS.get(m, INK_MUTED) for m in df["model"]]
    y = np.arange(len(df))
    ax.barh(y, df["MAE"], color=colours, height=0.62)
    for i, (v, m) in enumerate(zip(df["MAE"], df["model"])):
        ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=9.5,
                fontweight="bold" if m not in BASELINES else "normal", color=INK)
    best_baseline = df[df["model"].isin(BASELINES)]["MAE"].min()
    ax.axvline(best_baseline, color=INK, lw=1.3, ls="--")
    ax.text(best_baseline, len(df) - 0.35, " best baseline", fontsize=8.5,
            color=INK_MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("mean absolute error (ratio units), lower is better")
    # Deliberately not "accuracy": in this project that word denotes the
    # categorical subtype metric, and the axis here is error on the ratio.
    ax.set_title(f"One-step-ahead error on the ratio, {cohort} cohort", loc="left",
                 fontsize=11.5, fontweight="bold")
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)

    ax2.scatter(df["CRPS"], df["coverage95"], s=90, c=colours, zorder=3)
    # K1, K2 and K3 sit almost on top of one another, so fan their labels out
    # vertically rather than letting them overprint.
    order = df.sort_values("CRPS").reset_index(drop=True)
    for i, row in order.iterrows():
        near = order[(order["CRPS"] - row["CRPS"]).abs() < 0.004].index
        rank = list(near).index(i) if len(near) > 1 else 0
        ax2.annotate(row["model"], (row["CRPS"], row["coverage95"]),
                     textcoords="offset points",
                     xytext=(8, 4 + 11 * rank - 5 * (len(near) - 1)),
                     fontsize=9, color=INK)
    ax2.axhline(0.95, color=INK, lw=1.3, ls="--")
    ax2.text(ax2.get_xlim()[1], 0.952, "honest 95% interval ", ha="right",
             fontsize=8.5, color=INK_MUTED)
    ax2.set_xlabel("CRPS, lower is better")
    ax2.set_ylabel("actual coverage of the stated 95% interval")
    ax2.set_title("Sharpness against honesty", loc="left", fontsize=11.5,
                  fontweight="bold")
    _style(ax2)
    ax2.grid(color=GRID, lw=0.7)
    ax2.set_axisbelow(True)

    return _finish(fig, path)


def cohort_comparison(metrics: pd.DataFrame, models: list, path: Path) -> Path:
    """Does filtering change the answer, or just the population?

    Bars are not comparable across cohorts as absolute numbers --- different
    patients, different points --- so what to read is whether the *ordering* of
    the models survives filtering.
    """
    order = [c for c in ["unfiltered", "A4", "D1", "E2"]
             if c in metrics["cohort"].unique()]
    fig, ax = plt.subplots(figsize=(11.6, 5.0))
    fig.patch.set_facecolor(SURFACE)

    width = 0.8 / max(len(models), 1)
    x = np.arange(len(order))
    for i, model in enumerate(models):
        vals = [
            float(metrics[(metrics.cohort == c) & (metrics.model == model)]["MAE"].iloc[0])
            if not metrics[(metrics.cohort == c) & (metrics.model == model)].empty
            else np.nan
            for c in order
        ]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width * 0.92,
               color=MODEL_COLORS.get(model, INK_MUTED), label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=10)
    ax.set_ylabel("mean absolute error (ratio units)")
    ax.set_title("Each cohort scored on its own points, with its own baselines",
                 loc="left", fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, ncol=len(models))
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    return _finish(fig, path)


def confusion_panel(confusions: dict, path: Path) -> Path:
    """Where do the derived subtype predictions land?

    Shown mainly to make the Indeterminate column visible: a band 0.25 wide
    against a predictive spread several times that can hold almost no
    probability mass, whatever the model.
    """
    models = list(confusions)
    fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 4.2),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for ax, model in zip(axes[0], models):
        cm = confusions[model].reindex(index=STATE_ORDER, columns=STATE_ORDER,
                                       fill_value=0).to_numpy(float)
        shown = cm / max(cm.sum(), 1)
        ax.imshow(shown, cmap="Blues", vmin=0, vmax=shown.max() or 1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center",
                        fontsize=10,
                        color="white" if shown[i, j] > shown.max() * 0.55 else INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(STATE_ORDER, fontsize=8.5, rotation=30, ha="right")
        ax.set_yticks(range(3))
        ax.set_yticklabels(STATE_ORDER, fontsize=8.5)
        ax.set_title(model, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("actual", fontsize=9, color=INK_MUTED)
        ax.set_ylabel("predicted", fontsize=9, color=INK_MUTED)

    return _finish(fig, path)
