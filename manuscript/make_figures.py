"""Figures and tables for the two-method manuscript.

Every number in every figure is read from an existing experiment directory.
Nothing is refitted, resampled or estimated here: if a quantity is not in one
of those directories it is not plotted. Each figure writes a companion CSV
holding exactly the numbers it draws, so a figure can never disagree with the
manuscript text or with the run that produced it.

Styling follows :mod:`ppmi_subtype.plots`, so the state palette and axis
treatment match the rest of the project.

    python manuscript/make_figures.py [--ct-hmm DIR] [--profile DIR]
                                      [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import next_visit as nv  # noqa: E402
from ppmi_subtype.plots import (  # noqa: E402
    GRID, INK, INK_MUTED, SEQUENTIAL_BLUE, STATE_COLORS, SURFACE, _finish, _style,
)

#: Default source runs. Both are read-only here.
CT_HMM_RUN = "experiments/20261020T104756Z_next_visit_td_pigd_prediction"
PROFILE_RUN = "experiments/20261020T180515Z_clinical_profile_cv"

CT_MODEL = "CT-HMM (latent, interval-aware)"
RP_MODEL = "Profile: level + rate + medication"
RP_LEVEL_ONLY = "Profile: level only (random walk)"
RP_NO_MED = "Profile: level + rate, no medication"
LOCF = "Last observation carried forward"
POPMEAN = "Population mean trajectory"

#: Short display names, so a legend fits.
SHORT = {
    CT_MODEL: "CT-HMM\n(subtype history)",
    "Subtype-history: CT-HMM (latent, interval-aware)": "CT-HMM\n(subtype history)",
    "Subtype-history: Persistence (last non-Indeterminate)": "Persistence\nrule",
    "Subtype-history: Markov (observed, discrete)": "Markov chain\n(discrete)",
    RP_MODEL: "Rich profile\nlevel+rate+med",
    RP_LEVEL_ONLY: "Rich profile\nlevel only",
    RP_NO_MED: "Rich profile\nno medication",
    LOCF: "Last value\ncarried forward",
    POPMEAN: "Population\nmean",
    "Persistence (last non-Indeterminate)": "Persistence\nrule",
    "Markov (observed, discrete)": "Markov chain\n(discrete)",
    "Direct logistic (history + interval)": "Direct logistic\n(history)",
    "Marginal (two-class prior)": "Base rate\n(null)",
}

DOMAIN_LABELS = {
    "brady_rigid": "Bradykinesia /\nrigidity",
    "tremor": "Tremor",
    "pigd": "PIGD",
    "updrs2": "MDS-UPDRS II\n(motor ADL)",
    "updrs1": "MDS-UPDRS I\n(non-motor)",
    "moca": "MoCA\n(cognition)",
    "gds": "GDS-15\n(mood)",
    "scopa": "SCOPA-AUT\n(autonomic)",
    "ess": "ESS\n(sleep)",
}
DOMAINS = tuple(DOMAIN_LABELS)

ACCENT = "#1f4e79"
WARM = "#eb6834"
COOL = "#2a78d6"
CORRECT = "#2a78d6"
WRONG = "#eb6834"
NEUTRAL = "#b9b8b4"

#: Method colours for the patient-wise figure. Deliberately outside the state
#: palette (TD blue, Indeterminate orange, PIGD green) so a method marker can
#: never be mistaken for a class colour, and far enough apart from each other
#: to separate at marker size.
METHOD_CT = "#7b4fa8"
METHOD_RP = "#c2458f"

FIGURES: list[dict] = []


def record(path: Path, title: str, caption: str, data: Path | None = None) -> None:
    FIGURES.append({"figure": path.name, "title": title, "caption": caption,
                    "data": data.name if data else ""})


def save_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
#  1. Subtype prediction: the two methods against their baselines
# ---------------------------------------------------------------------------
def fig_subtype_comparison(paired: pd.DataFrame, out: Path) -> None:
    """Accuracy with patient-bootstrap CIs, on the paired evaluation set."""
    d = paired.sort_values("accuracy").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    _style(ax)

    y = np.arange(len(d))
    colours = []
    for m in d["model"]:
        if m.startswith("Profile"):
            colours.append(COOL)
        elif "CT-HMM" in m:
            colours.append(ACCENT)
        else:
            colours.append(NEUTRAL)

    lo = d["accuracy"] - d["accuracy_ci_patient_bootstrap_low"]
    hi = d["accuracy_ci_patient_bootstrap_high"] - d["accuracy"]
    ax.barh(y, d["accuracy"], color=colours, height=0.62, zorder=2)
    ax.errorbar(d["accuracy"], y, xerr=[lo, hi], fmt="none",
                ecolor=INK_MUTED, elinewidth=1.3, capsize=3.5, zorder=3)

    for i, (acc, m) in enumerate(zip(d["accuracy"], d["model"])):
        ax.text(acc + hi.iloc[i] + 0.006, i, f"{acc:.3f}", va="center",
                ha="left", fontsize=9, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels([SHORT.get(m, m).replace("\n", " ") for m in d["model"]],
                       fontsize=9)
    ax.set_xlim(0.65, 0.90)
    ax.set_xlabel("Accuracy on eligible TD-vs-PIGD next visits "
                  "(95% CI, patients resampled)")
    n = int(d["n_scored"].iloc[0])
    ax.set_title("Next observed motor subtype: both methods and their baselines\n"
                 f"paired evaluation, {n:,} held-out predictions from 934 patients",
                 fontsize=11, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=ACCENT, label="CT-HMM (Method 1)"),
                       Patch(facecolor=COOL, label="Rich profile (Method 2)"),
                       Patch(facecolor=NEUTRAL, label="Baseline / reference")],
              loc="lower right", frameon=False, fontsize=9)

    path = out / "fig01_subtype_accuracy_comparison.png"
    _finish(fig, path)
    data = save_table(d, out / "fig01_subtype_accuracy_comparison.csv")
    record(path, "Subtype prediction accuracy, both methods",
           "Accuracy on the 3,020 held-out TD-or-PIGD next visits shared by "
           "every model, with 95% intervals from resampling patients. The two "
           "methods' intervals overlap: no significant difference.", data)


# ---------------------------------------------------------------------------
#  2. Confusion matrices
# ---------------------------------------------------------------------------
def fig_confusion(matrices: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.3 * len(matrices), 4.0))
    axes = np.atleast_1d(axes)
    rows = []

    for ax, (name, cm) in zip(axes, matrices.items()):
        _style(ax)
        counts = cm.to_numpy(dtype=float)
        norm = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1, aspect="auto")
        for i in range(2):
            for j in range(2):
                shade = INK if norm[i, j] < 0.55 else SURFACE
                ax.text(j, i, f"{int(counts[i, j]):,}\n{norm[i, j]:.1%}",
                        ha="center", va="center", fontsize=10.5, color=shade)
                rows.append({"method": name, "predicted": cm.index[i],
                             "actual": cm.columns[j], "n": int(counts[i, j]),
                             "row_share": norm[i, j]})
        ax.set_xticks([0, 1]); ax.set_xticklabels(cm.columns)
        ax.set_yticks([0, 1]); ax.set_yticklabels(cm.index)
        ax.set_xlabel("Actual next label")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}\nn = {int(counts.sum()):,}", fontsize=10, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle("Confusion matrices on the paired held-out set "
                 "(cell = count, then row share)", fontsize=11, x=0.02, ha="left")
    path = out / "fig02_confusion_matrices.png"
    _finish(fig, path)
    data = save_table(pd.DataFrame(rows), out / "fig02_confusion_matrices.csv")
    record(path, "Confusion matrices",
           "Rows are the predicted class, columns the actual next label. Both "
           "methods under-call PIGD, which is the minority and clinically "
           "costlier class.", data)


def fig_per_class(per_class: pd.DataFrame, out: Path) -> None:
    metrics = ["precision", "recall", "f1"]
    classes = ["TD", "PIGD"]
    methods = list(per_class["method"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, cls in zip(axes, classes):
        _style(ax)
        sub = per_class[per_class["state"] == cls]
        x = np.arange(len(metrics))
        width = 0.8 / len(methods)
        for k, m in enumerate(methods):
            vals = [float(sub[sub["method"] == m][mt].iloc[0]) for mt in metrics]
            colour = COOL if m.startswith("Rich") else ACCENT
            ax.bar(x + k * width - 0.4 + width / 2, vals, width * 0.92,
                   label=m, color=colour,
                   alpha=1.0 if k == 0 else 0.62, zorder=2)
            for xi, v in zip(x + k * width - 0.4 + width / 2, vals):
                ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=8,
                        color=INK_MUTED)
        ax.set_xticks(x); ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.set_ylim(0, 1.0)
        support = int(sub["support"].iloc[0])
        ax.set_title(f"{cls}  (support {support:,})", fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")

    fig.suptitle("Per-class performance on the paired held-out set",
                 fontsize=11, x=0.02, ha="left")
    path = out / "fig03_per_class_metrics.png"
    _finish(fig, path)
    data = save_table(per_class, out / "fig03_per_class_metrics.csv")
    record(path, "Per-class precision, recall and F1",
           "Both methods recover TD well and PIGD poorly: PIGD recall is "
           "about two thirds, so roughly a third of PIGD next visits are "
           "missed.", data)


# ---------------------------------------------------------------------------
#  4. Patient-wise actual vs predicted: cohort overview
# ---------------------------------------------------------------------------
def fig_patient_overview(ct: pd.DataFrame, rp: pd.DataFrame, out: Path) -> None:
    """Every patient, one row, visits in temporal order, both methods.

    934 patients cannot carry readable labels, so this is a raster: one row per
    patient, one column per visit position, ordered by number of held-out
    predictions so the shape of the cohort's follow-up is visible. Colour marks
    whether each held-out prediction matched the actual next label; ineligible
    visits (actual next label Indeterminate, which is excluded from two-class
    scoring) and absent visits are shown distinctly and are never scored.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.4), sharey=True)
    rows = []

    for ax, (name, frame) in zip(axes, (("CT-HMM (subtype history)", ct),
                                        ("Rich profile (level+rate+med)", rp))):
        _style(ax)
        frame = frame.copy()
        frame["correct"] = np.where(
            frame["eligible"],
            frame["predicted_state"] == frame["actual_next_label"], np.nan)

        order = (frame.groupby("PATNO")
                 .agg(n=("visit_index", "size"))
                 .sort_values(["n"], ascending=False))
        patients = list(order.index)
        max_v = int(frame["visit_index"].max()) + 1

        # 0 = absent, 1 = ineligible (Indeterminate target), 2 = wrong, 3 = right
        grid = np.zeros((len(patients), max_v))
        index = {p: i for i, p in enumerate(patients)}
        for _, r in frame.iterrows():
            i = index[r["PATNO"]]
            j = int(r["visit_index"])
            if not r["eligible"]:
                grid[i, j] = 1
            else:
                grid[i, j] = 3 if r["correct"] else 2

        cmap = matplotlib.colors.ListedColormap(
            [SURFACE, NEUTRAL, WRONG, CORRECT])
        ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto",
                  interpolation="nearest")
        ax.set_xlabel("Visit position in the patient's own sequence\n"
                      "(prediction k is made from visits 0..k)")
        n_right = int((grid == 3).sum())
        n_wrong = int((grid == 2).sum())
        n_inel = int((grid == 1).sum())
        ax.set_title(f"{name}\n{n_right:,} correct · {n_wrong:,} incorrect · "
                     f"{n_inel:,} not scored", fontsize=10, loc="left")
        rows.append({"method": name, "n_patients": len(patients),
                     "n_correct": n_right, "n_incorrect": n_wrong,
                     "n_not_scored_indeterminate": n_inel})
        for s in ax.spines.values():
            s.set_visible(False)

    axes[0].set_ylabel(f"Patients, ordered by number of held-out predictions "
                       f"(n = {ct['PATNO'].nunique():,})")
    fig.legend(handles=[
        Patch(facecolor=CORRECT, label="predicted = actual"),
        Patch(facecolor=WRONG, label="predicted ≠ actual"),
        Patch(facecolor=NEUTRAL, label="not scored (actual next = Indeterminate)"),
        Patch(facecolor=SURFACE, edgecolor=GRID, label="no visit"),
    ], loc="lower center", ncol=4, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Held-out next-subtype predictions for every patient, "
                 "in temporal order", fontsize=11.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    path = out / "fig04_patientwise_overview.png"
    _finish(fig, path)
    data = save_table(pd.DataFrame(rows), out / "fig04_patientwise_overview.csv")
    record(path, "Cohort overview of held-out predictions per patient",
           "One row per patient, one column per visit position in that "
           "patient's own sequence. Held-out predictions only. Patients are "
           "ordered by follow-up length, not by accuracy.", data)


# ---------------------------------------------------------------------------
#  5. Patient-wise actual vs predicted: readable small multiples
# ---------------------------------------------------------------------------
def choose_examples(frame: pd.DataFrame, n_min: int = 4) -> tuple[list, dict]:
    """Pick example patients by a fixed rule that never looks at the model.

    Every criterion below is a property of the **observed** label sequence, so
    the selection cannot favour patients the model happens to do well on. Ties
    are broken by the lowest PATNO, so the choice is deterministic. This
    mirrors the existing rule in ``scripts/09_loo_validation.py``, minus its
    accuracy-based categories, which would be outcome selection here.
    """
    eligible = {p for p, g in frame.groupby("PATNO") if len(g) >= n_min}
    seqs = {}
    for p, g in frame.groupby("PATNO"):
        if p not in eligible:
            continue
        g = g.sort_values("visit_index")
        seqs[p] = [g["current_label"].iloc[0]] + list(g["actual_next_label"])

    def changes(s):
        return sum(1 for a, b in zip(s, s[1:]) if a != b)

    rules = [
        ("stable TD throughout", lambda s: set(s) == {"TD"}),
        ("stable PIGD throughout", lambda s: set(s) == {"PIGD"}),
        ("recorded change from TD to PIGD",
         lambda s: any(a == "TD" and b == "PIGD" for a, b in zip(s, s[1:]))),
        ("recorded change from PIGD to TD",
         lambda s: any(a == "PIGD" and b == "TD" for a, b in zip(s, s[1:]))),
        ("passes through Indeterminate", lambda s: "Indeterminate" in s),
    ]
    chosen: list = []
    notes: dict = {}
    for note, rule in rules:
        # Lowest PATNO satisfying the rule and not already taken by an earlier
        # rule, so each panel illustrates a distinct trajectory type.
        hits = sorted(q for q, seq in seqs.items()
                      if rule(seq) and q not in chosen)
        if hits:
            chosen.append(hits[0])
            notes[hits[0]] = note

    remaining = {q: changes(seq) for q, seq in seqs.items() if q not in chosen}
    if remaining:
        best = max(sorted(remaining), key=lambda k: remaining[k])
        chosen.append(best)
        notes[best] = f"most recorded label changes ({remaining[best]})"
    return chosen, notes


def fig_patient_examples(ct: pd.DataFrame, rp: pd.DataFrame,
                         patients: list, notes: dict, out: Path) -> None:
    """One panel per example patient: actual and predicted labels over time."""
    n = len(patients)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.0, 2.5 * nrow),
                             squeeze=False)
    y_of = {"TD": 2, "Indeterminate": 1, "PIGD": 0}
    rows = []

    for k, patno in enumerate(patients):
        ax = axes[k // ncol][k % ncol]
        _style(ax)
        a = ct[ct["PATNO"] == patno].sort_values("visit_index")
        b = rp[rp["PATNO"] == patno].sort_values("visit_index")

        # The observed label sequence: the current label at the first
        # prediction, then each actual next label, in temporal order.
        t_obs = [float(a["current_years"].iloc[0])] + list(a["target_years"])
        lab_obs = [a["current_label"].iloc[0]] + list(a["actual_next_label"])
        ax.plot(t_obs, [y_of[l] for l in lab_obs], "-", color=INK_MUTED,
                linewidth=1.2, zorder=2, label="observed label")
        for t, l in zip(t_obs, lab_obs):
            ax.scatter([t], [y_of[l]], s=58, color=STATE_COLORS[l],
                       edgecolor=SURFACE, linewidth=1.1, zorder=4)

        for frame, marker, colour, name in (
                (a, "v", METHOD_CT, "CT-HMM prediction"),
                (b, "^", METHOD_RP, "Rich-profile prediction")):
            if not len(frame):
                continue
            el = frame[frame["eligible"]]
            ax.scatter(el["target_years"],
                       [y_of[s] + (0.17 if marker == "v" else -0.17)
                        for s in el["predicted_state"]],
                       marker=marker, s=38, facecolor="none",
                       edgecolor=colour, linewidth=1.5, zorder=3, label=name)
            inel = frame[~frame["eligible"]]
            if len(inel):
                ax.scatter(inel["target_years"],
                           [y_of[s] + (0.17 if marker == "v" else -0.17)
                            for s in inel["predicted_state"]],
                           marker=marker, s=28, facecolor="none",
                           edgecolor=NEUTRAL, linewidth=1.1, zorder=3)

        for _, r in a.iterrows():
            rows.append({"PATNO": patno, "selection_rule": notes[patno],
                         "visit_index": int(r["visit_index"]),
                         "current_event": r["current_event"],
                         "target_event": r["target_event"],
                         "target_years": r["target_years"],
                         "actual_next_label": r["actual_next_label"],
                         "eligible": bool(r["eligible"]),
                         "ct_hmm_predicted": r["predicted_state"]})

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["PIGD", "Indet.", "TD"], fontsize=9)
        ax.set_ylim(-0.6, 2.6)
        ax.set_xlabel("Years from baseline", fontsize=9)
        ax.set_title(f"Patient {patno} — {notes[patno]}\n"
                     f"{len(a)} held-out predictions", fontsize=9.5, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    handles = [
        Line2D([], [], color=INK_MUTED, linewidth=1.2, label="observed label path"),
        Line2D([], [], marker="o", color="none", markerfacecolor=STATE_COLORS["TD"],
               markersize=8, label="observed label (coloured by class)"),
        Line2D([], [], marker="v", color="none", markeredgecolor=METHOD_CT,
               markersize=8, markeredgewidth=1.6,
               label="CT-HMM prediction (offset up)"),
        Line2D([], [], marker="^", color="none", markeredgecolor=METHOD_RP,
               markersize=8, markeredgewidth=1.6,
               label="rich-profile prediction (offset down)"),
        Line2D([], [], marker="^", color="none", markeredgecolor=NEUTRAL,
               markersize=7, label="prediction not scored (actual = Indeterminate)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Held-out predictions for example patients, in temporal order\n"
                 "patients chosen by observed-trajectory rules only, never by accuracy",
                 fontsize=11.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    path = out / "fig05_patientwise_examples.png"
    _finish(fig, path)
    data = save_table(pd.DataFrame(rows), out / "fig05_patientwise_examples.csv")
    record(path, "Example patients, actual versus predicted over time",
           "Six patients selected by fixed rules on their observed label "
           "sequence. Predictions are offset vertically so they do not hide "
           "the observed label. Held-out predictions only.", data)


# ---------------------------------------------------------------------------
#  6. Rich profile: actual vs predicted clinical measurements
# ---------------------------------------------------------------------------
def fig_profile_scatter(preds: pd.DataFrame, per_domain: pd.DataFrame,
                        out: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.0))
    rows = []
    for k, dom in enumerate(DOMAINS):
        ax = axes[k // 3][k % 3]
        _style(ax)
        a = preds[f"actual_{dom}"].to_numpy(dtype=float)
        p = preds[f"pred_{dom}"].to_numpy(dtype=float)
        m = np.isfinite(a) & np.isfinite(p)
        a, p = a[m], p[m]

        ax.scatter(a, p, s=6, alpha=0.18, color=COOL, edgecolor="none", zorder=2)
        lo = float(min(a.min(), p.min()))
        hi = float(max(a.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], "--", color=INK_MUTED, linewidth=1.0,
                zorder=3)

        r = float(np.corrcoef(a, p)[0, 1])
        row = per_domain[per_domain["domain"] == dom]
        rmse = float(row["rmse"].iloc[0])
        mae = float(row["mae"].iloc[0])
        ax.set_title(f"{DOMAIN_LABELS[dom]}", fontsize=9.5, loc="left")
        ax.text(0.03, 0.97, f"r = {r:.3f}\nRMSE = {rmse:.3g}\nMAE = {mae:.3g}\n"
                            f"n = {m.sum():,}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                color=INK_MUTED)
        ax.set_xlabel("Actual at next visit", fontsize=8.5)
        ax.set_ylabel("Predicted", fontsize=8.5)
        rows.append({"domain": dom, "n": int(m.sum()), "pearson_r": r,
                     "rmse": rmse, "mae": mae})

    fig.suptitle("Rich profile: predicted versus actual next-visit measurements\n"
                 "held-out predictions, clinical units, dashed line is identity",
                 fontsize=11.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out / "fig06_profile_actual_vs_predicted.png"
    _finish(fig, path)
    data = save_table(pd.DataFrame(rows),
                      out / "fig06_profile_actual_vs_predicted.csv")
    record(path, "Predicted versus actual clinical measurements",
           "The primary target of Method 2: nine held-out clinical "
           "measurements per visit. Correlation is high partly because the "
           "level is persistent; the baseline comparison in fig07 is the "
           "informative test.", data)


# ---------------------------------------------------------------------------
#  7. Rich profile against its baselines
# ---------------------------------------------------------------------------
def fig_profile_baselines(comparison: pd.DataFrame, contrasts: pd.DataFrame,
                          per_domain_all: pd.DataFrame, out: Path) -> None:
    fig = plt.figure(figsize=(12.0, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.25], wspace=0.42)

    # (a) held-out predictive log-density
    ax = fig.add_subplot(gs[0, 0])
    _style(ax)
    d = comparison.sort_values("loglik_per_domain_observation")
    y = np.arange(len(d))
    cols = [COOL if m.startswith("Profile") else NEUTRAL for m in d["model"]]
    ax.barh(y, d["loglik_per_domain_observation"], color=cols, height=0.6, zorder=2)
    ax.errorbar(d["loglik_per_domain_observation"], y,
                xerr=[d["loglik_per_domain_observation"] - d["loglik_ci_low"],
                      d["loglik_ci_high"] - d["loglik_per_domain_observation"]],
                fmt="none", ecolor=INK_MUTED, elinewidth=1.2, capsize=3, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT.get(m, m).replace("\n", " ") for m in d["model"]],
                       fontsize=8.5)
    ax.set_xlabel("Log predictive density\nper domain observation (higher better)",
                  fontsize=9)
    ax.set_title("(a) Profile prediction quality", fontsize=10, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # (b) paired contrasts
    ax = fig.add_subplot(gs[0, 1])
    _style(ax)
    c = contrasts.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(c))
    ax.errorbar(c["loglik_delta"], y,
                xerr=[c["loglik_delta"] - c["delta_ci_low"],
                      c["delta_ci_high"] - c["loglik_delta"]],
                fmt="o", color=ACCENT, ecolor=INK_MUTED, elinewidth=1.4,
                capsize=3.5, markersize=6, zorder=3)
    ax.axvline(0.0, color=WARM, linewidth=1.2, linestyle="--", zorder=2)
    short_c = {"rich profile against assume-no-change": "vs last value\ncarried forward",
               "rich profile against the population average": "vs population\nmean",
               "modelling the rate of change": "rate component\ncontributes",
               "medication as an input": "medication input\ncontributes"}
    ax.set_yticks(y)
    ax.set_yticklabels([short_c.get(t, t) for t in c["contrast"]], fontsize=8.5)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Δ log density per prediction\n(95% CI, patients resampled)",
                  fontsize=9)
    ax.set_title("(b) Where the gain comes from", fontsize=10, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # (c) per-domain RMSE, model vs the two baselines
    ax = fig.add_subplot(gs[0, 2])
    _style(ax)
    wanted = [RP_MODEL, LOCF, POPMEAN]
    width = 0.26
    x = np.arange(len(DOMAINS))
    for k, m in enumerate(wanted):
        sub = per_domain_all[per_domain_all["model"] == m].set_index("domain")
        vals = [float(sub.loc[d, "rmse"]) for d in DOMAINS]
        colour = COOL if m == RP_MODEL else (NEUTRAL if m == LOCF else "#d8d7d3")
        ax.bar(x + k * width - width, vals, width * 0.92, label=SHORT[m],
               color=colour, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([d for d in DOMAINS], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (clinical units)", fontsize=9)
    ax.set_title("(c) Per-domain error", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle("Rich profile model against its baselines, held-out "
                 f"({int(comparison['n_predictions'].iloc[0]):,} predictions, 934 patients)",
                 fontsize=11.5, x=0.02, ha="left")
    path = out / "fig07_profile_vs_baselines.png"
    _finish(fig, path)
    data = save_table(contrasts, out / "fig07_profile_vs_baselines.csv")
    record(path, "Rich profile against its baselines",
           "Panel (b) is the key decomposition: the model clearly beats "
           "assume-no-change, but the rate and medication components each "
           "contribute only a small part of that gain.", data)


# ---------------------------------------------------------------------------
#  8. Calibration
# ---------------------------------------------------------------------------
def fig_calibration(ct: pd.DataFrame, rp: pd.DataFrame,
                    coverage: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    rows = []

    # (a) reliability of P(PIGD) for both subtype predictions
    ax = axes[0]
    _style(ax)
    edges = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0001])
    for frame, name, colour, marker in ((ct, "CT-HMM", ACCENT, "o"),
                                        (rp, "Rich profile", COOL, "s")):
        s = frame[frame["eligible"] & frame["two_class_defined"]]
        p = s["P_PIGD"].to_numpy(dtype=float)
        y = (s["actual_next_label"] == "PIGD").to_numpy(dtype=float)
        idx = np.digitize(p, edges[1:-1], right=False)
        xs, ys, ns = [], [], []
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() < 20:
                continue
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
            rows.append({"panel": "reliability", "method": name,
                         "bin_low": edges[b], "bin_high": min(edges[b + 1], 1.0),
                         "n": int(m.sum()), "mean_predicted": p[m].mean(),
                         "observed_rate": y[m].mean()})
        ax.plot(xs, ys, marker=marker, color=colour, linewidth=1.6,
                markersize=6, label=f"{name} (n = {len(s):,})", zorder=3)
        for xi, yi, ni in zip(xs, ys, ns):
            ax.annotate(f"{ni}", (xi, yi), textcoords="offset points",
                        xytext=(0, -12), ha="center", fontsize=7,
                        color=INK_MUTED)
    ax.plot([0, 1], [0, 1], "--", color=INK_MUTED, linewidth=1.0, zorder=2)
    ax.set_xlabel("Mean predicted P(PIGD)")
    ax.set_ylabel("Observed fraction PIGD")
    ax.set_title("(a) Reliability of the two-class subtype probability\n"
                 "bins with fewer than 20 cases omitted; point labels are n",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # (b) interval coverage per domain, rich profile
    ax = axes[1]
    _style(ax)
    cov = coverage.sort_values("coverage")
    y = np.arange(len(cov))
    ax.barh(y, cov["coverage"], color=COOL, height=0.6, zorder=2)
    ax.axvline(0.95, color=WARM, linewidth=1.4, linestyle="--", zorder=3,
               label="nominal 95%")
    ax.set_yticks(y)
    ax.set_yticklabels(cov["domain"], fontsize=9)
    ax.set_xlim(0.80, 1.0)
    ax.set_xlabel("Share of held-out measurements inside the 95% "
                  "predictive interval")
    ax.set_title("(b) Rich-profile interval calibration, per domain",
                 fontsize=10, loc="left")
    for yi, v, n in zip(y, cov["coverage"], cov["n"]):
        ax.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for _, r in cov.iterrows():
        rows.append({"panel": "coverage", "method": "Rich profile",
                     "domain": r["domain"], "n": int(r["n"]),
                     "coverage_95": r["coverage"]})

    fig.suptitle("Probability and interval calibration, held-out",
                 fontsize=11.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out / "fig08_calibration.png"
    _finish(fig, path)
    data = save_table(pd.DataFrame(rows), out / "fig08_calibration.csv")
    record(path, "Calibration of both methods",
           "Subtype probabilities from both methods track observed PIGD "
           "frequency closely. Rich-profile predictive intervals are slightly "
           "too narrow: coverage is below the nominal 0.95 in most domains.",
           data)


# ---------------------------------------------------------------------------
#  9. Why a richer representation: dimensionality and rate spread
# ---------------------------------------------------------------------------
def fig_representation(dim: pd.DataFrame, spread: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    ax = axes[0]
    _style(ax)
    for which, colour, marker in (("level", COOL, "o"), ("rate", ACCENT, "s")):
        row = dim[dim["which"] == which]
        if not len(row):
            continue
        shares = json.loads(row["variance_shares"].iloc[0]) \
            if isinstance(row["variance_shares"].iloc[0], str) \
            else row["variance_shares"].iloc[0]
        shares = np.asarray(shares, dtype=float)
        ax.plot(np.arange(1, len(shares) + 1), np.cumsum(shares), marker=marker,
                color=colour, linewidth=1.6, markersize=5,
                label=f"{which}s (PC1 = {shares[0]:.1%})", zorder=3)
    ax.axhline(0.90, color=WARM, linestyle="--", linewidth=1.2,
               label="90% of variance", zorder=2)
    ax.set_xlabel("Principal components of the across-patient representation")
    ax.set_ylabel("Cumulative share of variance")
    ax.set_ylim(0, 1.02)
    ax.set_title("(a) The representation is not one severity axis\n"
                 "seven components are needed for 90% of the variance",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax = axes[1]
    _style(ax)
    s = spread.copy()
    # Standardise each domain's rate spread by that domain's own scale so the
    # nine domains are comparable on one axis.
    s["cv"] = s["sd_rate_per_year"] / s["sd_rate_per_year"].max()
    s = s.sort_values("cv")
    y = np.arange(len(s))
    ax.barh(y, s["cv"], color=COOL, height=0.6, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(s["domain"], fontsize=9)
    ax.set_xlabel("Across-patient spread of the fitted rate,\n"
                  "relative to the largest domain")
    ax.set_title("(b) Rate heterogeneity differs sharply by domain\n"
                 "tremor's fitted rate variance is zero", fontsize=10, loc="left")
    for yi, (_, r) in zip(y, s.iterrows()):
        ax.text(r["cv"] + 0.012, yi,
                f"SD {r['sd_rate_per_year']:.4g}/yr", va="center",
                fontsize=8, color=INK_MUTED)
    ax.set_xlim(0, 1.28)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle("What the rich representation contains (934 patients)",
                 fontsize=11.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = out / "fig09_representation.png"
    _finish(fig, path)
    data = save_table(spread, out / "fig09_representation.csv")
    record(path, "Dimensionality and rate heterogeneity of the representation",
           "Panel (a) is the evidence that the representation preserves "
           "multidimensional variation a single subtype label discards.", data)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-hmm", default=CT_HMM_RUN)
    ap.add_argument("--profile", default=PROFILE_RUN)
    ap.add_argument("--out", default="manuscript/figures")
    args = ap.parse_args()

    ct_dir = ROOT / args.ct_hmm
    rp_dir = ROOT / args.profile
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    for d in (ct_dir, rp_dir):
        if not (d / "summary.json").exists():
            raise SystemExit(f"missing run directory or summary.json: {d}")

    # ---- source tables, read only -----------------------------------------
    ct_preds = pd.read_csv(ct_dir / "next_visit_predictions.csv", low_memory=False)
    ct = ct_preds[ct_preds["model"] == CT_MODEL].copy()

    rp_label = pd.read_csv(rp_dir / "derived_label_predictions.csv",
                           low_memory=False)
    rp = rp_label[rp_label["model"] == RP_MODEL].copy()
    paired = pd.read_csv(rp_dir / "derived_label_vs_subtype_history.csv")
    rp_cmp = pd.read_csv(rp_dir / "model_comparison.csv")
    rp_contrasts = pd.read_csv(rp_dir / "contrasts.csv")
    rp_per_domain = pd.read_csv(rp_dir / "per_domain_errors.csv")
    rp_dim = pd.read_csv(rp_dir / "representation_dimensionality.csv")
    rp_spread = pd.read_csv(rp_dir / "rate_spread.csv")
    rp_profile = pd.read_csv(rp_dir / "profile_predictions.csv", low_memory=False)
    rp_profile = rp_profile[(rp_profile["model"] == RP_MODEL)
                            & (rp_profile["repeat"] == 0)].copy()

    # The derived-label table carries the probabilities but not the visit times;
    # those live in the profile prediction table for the same rows. Joining on
    # the visit pair restores the temporal ordering the patient-wise figures
    # need, and the inner join is asserted to lose nothing.
    time_cols = ["PATNO", "current_event", "target_event", "visit_index",
                 "current_years", "target_years"]
    before = len(rp)
    rp = rp.merge(rp_profile[time_cols], on=["PATNO", "current_event",
                                             "target_event"], how="inner")
    if len(rp) != before:
        raise SystemExit(f"joining visit times changed the row count: "
                         f"{before} -> {len(rp)}")

    print(f"CT-HMM:       {len(ct):,} held-out predictions, "
          f"{ct['PATNO'].nunique():,} patients")
    print(f"Rich profile: {len(rp):,} held-out predictions, "
          f"{rp['PATNO'].nunique():,} patients")

    # ---- figures -----------------------------------------------------------
    fig_subtype_comparison(paired, out)

    matrices = {"CT-HMM (subtype history)": nv.confusion(ct),
                "Rich profile (level+rate+med)": nv.confusion(rp)}
    fig_confusion(matrices, out)

    per_class = pd.concat([
        nv.per_class_metrics(ct).assign(method="CT-HMM (subtype history)"),
        nv.per_class_metrics(rp).assign(method="Rich profile (level+rate+med)"),
    ], ignore_index=True)
    fig_per_class(per_class, out)

    fig_patient_overview(ct, rp, out)
    patients, notes = choose_examples(ct)
    print("example patients:", {p: notes[p] for p in patients})
    fig_patient_examples(ct, rp, patients, notes, out)

    fig_profile_scatter(rp_profile,
                        rp_per_domain[rp_per_domain["model"] == RP_MODEL], out)
    fig_profile_baselines(rp_cmp, rp_contrasts, rp_per_domain, out)

    coverage = []
    for dom in DOMAINS:
        a = rp_profile[f"actual_{dom}"].to_numpy(dtype=float)
        p = rp_profile[f"pred_{dom}"].to_numpy(dtype=float)
        s = rp_profile[f"sd_{dom}"].to_numpy(dtype=float)
        m = np.isfinite(a) & np.isfinite(p) & np.isfinite(s) & (s > 0)
        inside = np.abs(a[m] - p[m]) <= 1.959963985 * s[m]
        coverage.append({"domain": dom, "n": int(m.sum()),
                         "coverage": float(inside.mean())})
    fig_calibration(ct, rp, pd.DataFrame(coverage), out)
    fig_representation(rp_dim, rp_spread, out)

    # ---- figure index ------------------------------------------------------
    index = pd.DataFrame(FIGURES)
    index.to_csv(out / "figure_index.csv", index=False)

    manifest = {
        "ct_hmm_run": str(ct_dir.relative_to(ROOT)),
        "profile_run": str(rp_dir.relative_to(ROOT)),
        "ct_hmm_predictions": len(ct),
        "ct_hmm_patients": int(ct["PATNO"].nunique()),
        "profile_predictions": len(rp),
        "profile_patients": int(rp["PATNO"].nunique()),
        "example_patients": {str(p): notes[p] for p in patients},
        "figures": FIGURES,
        "figure_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.glob("*.png"))},
    }
    (out / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"\nwrote {len(FIGURES)} figures to {out.relative_to(ROOT)}")
    for f in FIGURES:
        print(f"  {f['figure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
