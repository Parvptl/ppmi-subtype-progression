"""Individual patient-level accuracy audit of Experiments 6A and 6B.

The rich-profile experiments reported **visit-level** metrics: every eligible
target visit counted once, so a patient with eleven predictions carried eleven
times the weight of a patient with one. This script asks the complementary
question -- is the aggregate number representative of a typical *patient*, or
is it carried by a subset of patients with many easy visits?

It is an analysis-only script. No model is fitted and no prediction is
recomputed: it reads the two saved prediction files, re-derives correctness
from the stored true and predicted labels, and regroups by patient.

WHAT THE INSPECTION ESTABLISHED
-------------------------------
*   **Sources.** `window_predictions.csv` (6A, `scripts/73`) and
    `condition_predictions.csv` (6B, `scripts/74`).
*   **Eligibility.** Both scripts score
    `actual_next_label in {TD, PIGD} and two_class_defined`. The saved files
    keep the Indeterminate target rows, so the filter has to be re-applied.
    `y` is `(actual_next_label == "PIGD")` and is therefore **0 for
    Indeterminate rows as well as TD rows** -- `y` alone is not an eligibility
    flag and must never be used as one.
*   **Row key.** `scripts/73` keys a target by
    `["PATNO", "current_event", "target_event"]`. `visit_index` is NOT a safe
    key: the persistence rows carry no `visit_index` (the rule does not run
    through `predict_sequence`), so keying on it reports 3,177 phantom
    duplicates. This script keys on the project's own triple.
*   **Cohorts.** The eligible target sets nest exactly -- last 5 subset of
    last 3 subset of last 2 subset of full history -- and the 6A and 6B
    eligible sets are identical for the conditions they share. So every paired
    comparison can be run on matched patients *and* matched target visits.

DEFINITIONS USED HERE
---------------------
*   **Visit-level accuracy** -- correct eligible targets / eligible targets.
    This is what 6A and 6B reported.
*   **Patient accuracy** -- for one patient, correct eligible targets /
    that patient's eligible targets.
*   **Mean patient-level accuracy** -- the unweighted mean of patient
    accuracies. Each patient counts once regardless of visit count. It is a
    different estimand from visit-level accuracy and the two are not expected
    to agree.
*   **Paired difference** -- per patient, accuracy under condition X minus
    accuracy under condition Y, both computed on the *same* matched target
    visits.

All uncertainty is a patient-level (cluster) bootstrap: patients are resampled
with replacement, carrying all of their visits, never individual visits.

    python scripts/75_patient_level_accuracy_audit.py --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype.plots import GRID, INK, INK_MUTED, _finish, _style  # noqa: E402

RUN_6A = "experiments/20261103T052237Z_profile_prediction_history"
RUN_6B = "experiments/20261103T072507Z_profile_two_visit_retraining"
PRED_6A = "window_predictions.csv"
PRED_6B = "condition_predictions.csv"

KEY = ["PATNO", "current_event", "target_event"]
FULL = "full history"
PERSIST = "persistence"

ORDER_6A = ["last 1", "last 2", "last 3", "last 5", FULL, PERSIST]
ORDER_6B = ["A: full-trained, full history",
            "B: full-trained, last 2",
            "C: window-trained, full history",
            "D: window-trained, last 2",
            "C': disjoint-window-trained, full history",
            "D': disjoint-window-trained, last 2",
            PERSIST]

#  (label, condition X, condition Y) -- the difference reported is X - Y.
PAIRS_6A = [
    ("last 1 vs full history", "last 1", FULL),
    ("last 2 vs full history", "last 2", FULL),
    ("last 3 vs full history", "last 3", FULL),
    ("last 5 vs full history", "last 5", FULL),
    ("full history vs persistence", FULL, PERSIST),
    ("last 2 vs persistence", "last 2", PERSIST),
]
PAIRS_6B = [
    ("1. D vs B: retraining, last-2 prediction",
     "D: window-trained, last 2", "B: full-trained, last 2"),
    ("2. C vs A: retraining, full prediction history",
     "C: window-trained, full history", "A: full-trained, full history"),
    ("3. D' vs B: disjoint retraining, last-2 prediction",
     "D': disjoint-window-trained, last 2", "B: full-trained, last 2"),
    ("4. C' vs A: disjoint retraining, full prediction history",
     "C': disjoint-window-trained, full history", "A: full-trained, full history"),
    ("5. B vs A: prediction history, full-sequence training",
     "B: full-trained, last 2", "A: full-trained, full history"),
    ("6. D vs C: prediction history, overlapping-window training",
     "D: window-trained, last 2", "C: window-trained, full history"),
    ("7. D' vs C': prediction history, disjoint-window training",
     "D': disjoint-window-trained, last 2", "C': disjoint-window-trained, full history"),
    ("8. D vs persistence (reference)",
     "D: window-trained, last 2", PERSIST),
    ("9. B vs persistence (reference)",
     "B: full-trained, last 2", PERSIST),
]

BANDS = ["exactly 0%", "above 0% and below 50%", "50% to below 80%",
         "80% to below 100%", "exactly 100%"]
VISIT_BINS = ["1", "2-3", "4-5", "more than 5"]
COMPOSITION = ["all TD targets", "majority TD", "evenly split",
               "majority PIGD", "all PIGD targets"]

ACCENT = "#1f4e79"
COOL = "#2a78d6"
WARM = "#c2552c"
MUTED = "#8c8c88"

lines: list[str] = []


def say(t: str = "") -> None:
    print(t, flush=True)
    lines.append(t)


def section(t: str) -> None:
    say("\n" + "=" * 78)
    say(t)
    say("=" * 78)


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
#  Loading and eligibility
# ---------------------------------------------------------------------------
def verify_manifest(run: Path) -> tuple[int, int]:
    """Re-hash every file the source run recorded. Returns (ok, total)."""
    man = run / "MANIFEST.sha256"
    ok = total = 0
    for line in man.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        total += 1
        p = run / rel
        if p.is_file() and sha(p) == digest:
            ok += 1
    return ok, total


def load_eligible(run: Path, pred_name: str,
                  cond_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Eligible scored rows (correctness re-derived), and the raw frame."""
    raw = pd.read_csv(run / pred_name, low_memory=False)
    need = set(KEY) | {cond_col, "actual_next_label", "predicted_state",
                       "two_class_defined", "y", "pred_bin", "P_PIGD"}
    missing = need - set(raw.columns)
    if missing:
        raise SystemExit(f"{run/pred_name}: missing columns {sorted(missing)}")

    #  The project's own eligibility rule, re-applied. `y` is NOT a flag:
    #  Indeterminate targets are stored with y = 0.
    elig = raw[raw["actual_next_label"].isin(["TD", "PIGD"])
               & raw["two_class_defined"].astype(bool)].copy()
    elig["condition"] = elig[cond_col]
    #  Re-derive rather than trust, then check agreement downstream.
    elig["y_recomputed"] = (elig["actual_next_label"] == "PIGD").astype(int)
    elig["pred_recomputed"] = (elig["predicted_state"] == "PIGD").astype(int)
    elig["correct"] = (elig["y_recomputed"] == elig["pred_recomputed"]).astype(int)
    keep = KEY + ["condition", "actual_next_label", "predicted_state",
                  "y", "pred_bin", "y_recomputed", "pred_recomputed",
                  "correct", "P_PIGD"]
    return elig[keep].reset_index(drop=True), raw


def patient_table(elig: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Per (condition, patient) counts, accuracy and class-specific rates."""
    g = elig.groupby(["condition", "PATNO"], sort=True)
    t = g.agg(n_eligible=("correct", "size"),
              n_correct=("correct", "sum"),
              n_TD_targets=("y_recomputed", lambda s: int((s == 0).sum())),
              n_PIGD_targets=("y_recomputed", "sum")).reset_index()
    #  Correct calls split by the TRUE class of the target.
    corr = (elig.assign(cls=np.where(elig["y_recomputed"] == 1, "PIGD", "TD"))
            .groupby(["condition", "PATNO", "cls"])["correct"].sum()
            .unstack("cls").reindex(columns=["TD", "PIGD"]).fillna(0)
            .rename(columns={"TD": "n_TD_correct", "PIGD": "n_PIGD_correct"})
            .reset_index())
    t = t.merge(corr, on=["condition", "PATNO"], how="left")
    t["patient_accuracy"] = t["n_correct"] / t["n_eligible"]
    #  Not applicable, not zero, when the patient has no target of that class.
    t["patient_PIGD_sensitivity"] = np.where(
        t["n_PIGD_targets"] > 0, t["n_PIGD_correct"] / t["n_PIGD_targets"], np.nan)
    t["patient_TD_specificity"] = np.where(
        t["n_TD_targets"] > 0, t["n_TD_correct"] / t["n_TD_targets"], np.nan)
    t["accuracy_band"] = [band(a) for a in t["patient_accuracy"]]
    t["visit_count_bin"] = [visit_bin(n) for n in t["n_eligible"]]
    t["subtype_composition"] = [
        composition(p, n) for p, n in zip(t["n_PIGD_targets"], t["n_eligible"])]
    t["experiment"] = experiment
    return t


def band(a: float) -> str:
    if a == 0.0:
        return BANDS[0]
    if a < 0.5:
        return BANDS[1]
    if a < 0.8:
        return BANDS[2]
    if a < 1.0:
        return BANDS[3]
    return BANDS[4]


def visit_bin(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 5:
        return "4-5"
    return "more than 5"


def composition(n_pigd: int, n_total: int) -> str:
    share = n_pigd / n_total
    if share == 0.0:
        return COMPOSITION[0]
    if share < 0.5:
        return COMPOSITION[1]
    if share == 0.5:
        return COMPOSITION[2]
    if share < 1.0:
        return COMPOSITION[3]
    return COMPOSITION[4]


# ---------------------------------------------------------------------------
#  Patient-level bootstrap
# ---------------------------------------------------------------------------
def boot_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """One matrix of patient indices, drawn with replacement, shape (B, n)."""
    return np.random.default_rng(seed).integers(0, n, (n_boot, n))


def boot_ci(values: np.ndarray, idx: np.ndarray, stat) -> tuple[float, float]:
    draws = stat(values[idx], axis=1)
    ok = draws[np.isfinite(draws)]
    if not len(ok):
        return float("nan"), float("nan")
    return float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))


def distribution_summary(pt: pd.DataFrame, order: list[str],
                         n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for c in order:
        s = pt[pt["condition"] == c]
        if s.empty:
            continue
        a = s["patient_accuracy"].to_numpy(float)
        idx = boot_indices(len(a), n_boot, seed)
        mlo, mhi = boot_ci(a, idx, np.mean)
        dlo, dhi = boot_ci(a, idx, np.median)
        counts = s["accuracy_band"].value_counts()
        row = {"condition": c, "n_patients": len(a),
               "n_eligible_targets": int(s["n_eligible"].sum()),
               "visit_level_accuracy": s["n_correct"].sum() / s["n_eligible"].sum(),
               "mean_patient_accuracy": float(a.mean()),
               "mean_ci_lo": mlo, "mean_ci_hi": mhi,
               "median_patient_accuracy": float(np.median(a)),
               "median_ci_lo": dlo, "median_ci_hi": dhi,
               "sd_patient_accuracy": float(a.std(ddof=1)),
               "q25": float(np.quantile(a, 0.25)),
               "q75": float(np.quantile(a, 0.75)),
               "iqr": float(np.quantile(a, 0.75) - np.quantile(a, 0.25)),
               "min_patient_accuracy": float(a.min()),
               "max_patient_accuracy": float(a.max()),
               "median_eligible_per_patient": float(s["n_eligible"].median()),
               "max_eligible_per_patient": int(s["n_eligible"].max())}
        for b in BANDS:
            row[f"n_{b}"] = int(counts.get(b, 0))
            row[f"pct_{b}"] = 100.0 * counts.get(b, 0) / len(a)
        #  The brief also names two overlapping (cumulative) categories.
        row["n_below_50pct_cumulative"] = int((a < 0.5).sum())
        row["pct_below_50pct_cumulative"] = 100.0 * float((a < 0.5).mean())
        row["n_at_least_80pct_cumulative"] = int((a >= 0.8).sum())
        row["pct_at_least_80pct_cumulative"] = 100.0 * float((a >= 0.8).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparison(elig: pd.DataFrame, label: str, cx: str, cy: str,
                      n_boot: int, seed: int) -> dict:
    """Patient accuracy difference on MATCHED patients and MATCHED targets."""
    x = elig[elig["condition"] == cx].set_index(KEY)
    y = elig[elig["condition"] == cy].set_index(KEY)
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]
    if not x["y_recomputed"].equals(y["y_recomputed"]):
        raise SystemExit(f"{label}: matched targets disagree on the true label")

    pat = common.get_level_values("PATNO").to_numpy()
    d = pd.DataFrame({"PATNO": pat,
                      "cx": x["correct"].to_numpy(int),
                      "cy": y["correct"].to_numpy(int)})
    per = d.groupby("PATNO").agg(n=("cx", "size"), sx=("cx", "sum"),
                                 sy=("cy", "sum")).reset_index()
    ax = (per["sx"] / per["n"]).to_numpy(float)
    ay = (per["sy"] / per["n"]).to_numpy(float)
    diff = ax - ay
    idx = boot_indices(len(diff), n_boot, seed)
    mlo, mhi = boot_ci(diff, idx, np.mean)
    dlo, dhi = boot_ci(diff, idx, np.median)
    #  Both condition means come from the SAME patient draws.
    xlo, xhi = boot_ci(ax, idx, np.mean)
    ylo, yhi = boot_ci(ay, idx, np.mean)
    return {
        "comparison": label, "condition_x": cx, "condition_y": cy,
        "n_matched_patients": len(diff),
        "n_matched_targets": int(per["n"].sum()),
        "mean_patient_accuracy_x": float(ax.mean()),
        "mean_x_ci_lo": xlo, "mean_x_ci_hi": xhi,
        "mean_patient_accuracy_y": float(ay.mean()),
        "mean_y_ci_lo": ylo, "mean_y_ci_hi": yhi,
        "visit_level_accuracy_x": float(per["sx"].sum() / per["n"].sum()),
        "visit_level_accuracy_y": float(per["sy"].sum() / per["n"].sum()),
        "mean_paired_difference": float(diff.mean()),
        "diff_ci_lo": mlo, "diff_ci_hi": mhi,
        "diff_excludes_zero": bool(mlo > 0 or mhi < 0),
        "median_paired_difference": float(np.median(diff)),
        "median_diff_ci_lo": dlo, "median_diff_ci_hi": dhi,
        "n_improved": int((diff > 0).sum()),
        "n_worsened": int((diff < 0).sum()),
        "n_unchanged": int((diff == 0).sum()),
        "pct_improved": 100.0 * float((diff > 0).mean()),
        "pct_worsened": 100.0 * float((diff < 0).mean()),
        "pct_unchanged": 100.0 * float((diff == 0).mean()),
    }


def by_visit_count(pt: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    rows = []
    for c in conditions:
        s = pt[pt["condition"] == c]
        for b in VISIT_BINS:
            g = s[s["visit_count_bin"] == b]
            if g.empty:
                continue
            a = g["patient_accuracy"].to_numpy(float)
            rows.append({"condition": c, "visit_count_bin": b,
                         "n_patients": len(g),
                         "n_eligible_targets": int(g["n_eligible"].sum()),
                         "mean_patient_accuracy": float(a.mean()),
                         "median_patient_accuracy": float(np.median(a)),
                         "sd_patient_accuracy": (float(a.std(ddof=1))
                                                 if len(a) > 1 else float("nan")),
                         "visit_level_accuracy":
                             float(g["n_correct"].sum() / g["n_eligible"].sum()),
                         "pct_exactly_100": 100.0 * float((a == 1.0).mean()),
                         "pct_exactly_0": 100.0 * float((a == 0.0).mean())})
    return pd.DataFrame(rows)


def by_composition(pt: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    rows = []
    for c in conditions:
        s = pt[pt["condition"] == c]
        for b in COMPOSITION:
            g = s[s["subtype_composition"] == b]
            if g.empty:
                continue
            a = g["patient_accuracy"].to_numpy(float)
            rows.append({"condition": c, "subtype_composition": b,
                         "n_patients": len(g),
                         "n_eligible_targets": int(g["n_eligible"].sum()),
                         "mean_patient_accuracy": float(a.mean()),
                         "median_patient_accuracy": float(np.median(a)),
                         "visit_level_accuracy":
                             float(g["n_correct"].sum() / g["n_eligible"].sum()),
                         "median_eligible_per_patient":
                             float(g["n_eligible"].median())})
    return pd.DataFrame(rows)


def subtype_metrics(pt: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    """Per-condition summary of the per-patient class-specific rates."""
    rows = []
    for c in conditions:
        s = pt[pt["condition"] == c]
        sens = s["patient_PIGD_sensitivity"]
        spec = s["patient_TD_specificity"]
        has_p, has_t = sens.notna(), spec.notna()
        multi_p = has_p & (s["n_PIGD_targets"] >= 2)
        multi_t = has_t & (s["n_TD_targets"] >= 2)
        rows.append({
            "condition": c, "n_patients": len(s),
            "n_patients_with_PIGD_targets": int(has_p.sum()),
            "n_patients_with_TD_targets": int(has_t.sum()),
            "n_patients_no_PIGD_targets_not_applicable": int((~has_p).sum()),
            "n_patients_no_TD_targets_not_applicable": int((~has_t).sum()),
            "mean_patient_PIGD_sensitivity": float(sens[has_p].mean()),
            "median_patient_PIGD_sensitivity": float(sens[has_p].median()),
            "mean_patient_TD_specificity": float(spec[has_t].mean()),
            "median_patient_TD_specificity": float(spec[has_t].median()),
            "n_PIGD_patients_zero_correct": int((sens[has_p] == 0).sum()),
            "pct_PIGD_patients_zero_correct": 100.0 * float((sens[has_p] == 0).mean()),
            "n_TD_patients_zero_correct": int((spec[has_t] == 0).sum()),
            "pct_TD_patients_zero_correct": 100.0 * float((spec[has_t] == 0).mean()),
            "n_PIGD_patients_all_correct": int((sens[has_p] == 1).sum()),
            "n_TD_patients_all_correct": int((spec[has_t] == 1).sum()),
            #  Restricted to patients with at least two targets of the class,
            #  because a single target forces a rate of exactly 0 or 1.
            "n_patients_ge2_PIGD_targets": int(multi_p.sum()),
            "mean_PIGD_sensitivity_ge2_targets": float(sens[multi_p].mean()),
            "pct_zero_correct_ge2_PIGD_targets":
                100.0 * float((sens[multi_p] == 0).mean()) if multi_p.any() else float("nan"),
            "n_patients_ge2_TD_targets": int(multi_t.sum()),
            "mean_TD_specificity_ge2_targets": float(spec[multi_t].mean()),
            "pct_zero_correct_ge2_TD_targets":
                100.0 * float((spec[multi_t] == 0).mean()) if multi_t.any() else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  The whole analysis as one pure function, so it can be re-run for repro
# ---------------------------------------------------------------------------
def analyse(e6a: pd.DataFrame, e6b: pd.DataFrame,
            n_boot: int, seed: int) -> dict[str, pd.DataFrame]:
    pt6a = patient_table(e6a, "6A")
    pt6b = patient_table(e6b, "6B")
    out = {
        "patients": pd.concat([pt6a, pt6b], ignore_index=True),
        "dist_6a": distribution_summary(pt6a, ORDER_6A, n_boot, seed),
        "dist_6b": distribution_summary(pt6b, ORDER_6B, n_boot, seed),
        "paired_6a": pd.DataFrame(
            [paired_comparison(e6a, *p, n_boot, seed) for p in PAIRS_6A]),
        "paired_6b": pd.DataFrame(
            [paired_comparison(e6b, *p, n_boot, seed) for p in PAIRS_6B]),
        "visit_counts": pd.concat([
            by_visit_count(pt6a, ORDER_6A).assign(experiment="6A"),
            by_visit_count(pt6b, ORDER_6B).assign(experiment="6B")],
            ignore_index=True),
        "composition": pd.concat([
            by_composition(pt6a, ORDER_6A).assign(experiment="6A"),
            by_composition(pt6b, ORDER_6B).assign(experiment="6B")],
            ignore_index=True),
        "subtype": pd.concat([
            subtype_metrics(pt6a, ORDER_6A).assign(experiment="6A"),
            subtype_metrics(pt6b, ORDER_6B).assign(experiment="6B")],
            ignore_index=True),
    }
    out["paired_6a"].insert(0, "experiment", "6A")
    out["paired_6b"].insert(0, "experiment", "6B")
    return out


# ---------------------------------------------------------------------------
#  Figures.  No patient identifier appears in any of them.
# ---------------------------------------------------------------------------
def fig_accuracy_histogram(pt: pd.DataFrame, dist: pd.DataFrame, out: Path) -> None:
    s = pt[pt["condition"] == FULL]
    a = s["patient_accuracy"].to_numpy(float)
    d = dist[dist["condition"] == FULL].iloc[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.hist(a, bins=np.linspace(0, 1, 21), color=COOL, edgecolor="white",
            linewidth=0.8, zorder=3)
    for v, c, lab in ((d["visit_level_accuracy"], WARM, "visit-level accuracy"),
                      (d["mean_patient_accuracy"], ACCENT, "mean patient accuracy"),
                      (d["median_patient_accuracy"], INK, "median patient accuracy")):
        ax.axvline(v, color=c, linewidth=1.6, linestyle="--", zorder=4,
                   label=f"{lab} = {v:.3f}")
    ax.set_xlabel("patient accuracy (correct eligible targets / eligible targets)")
    ax.set_ylabel("patients")
    ax.set_title(f"Experiment 6A, full history: patient accuracy\n"
                 f"{int(d['n_patients'])} patients, "
                 f"{int(d['n_eligible_targets'])} eligible target visits",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED)
    say(f"  wrote {_finish(fig, out / 'fig1_patient_accuracy_full_history.png').name}")


def fig_accuracy_boxplot(pt: pd.DataFrame, out: Path) -> None:
    conds = [c for c in ORDER_6A if c in set(pt["condition"])]
    data = [pt[pt["condition"] == c]["patient_accuracy"].to_numpy(float)
            for c in conds]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, zorder=3,
                    medianprops=dict(color=INK, linewidth=1.6),
                    whiskerprops=dict(color=INK_MUTED),
                    capprops=dict(color=INK_MUTED),
                    flierprops=dict(marker="o", markersize=2.5,
                                    markerfacecolor=MUTED,
                                    markeredgecolor="none", alpha=0.5))
    for b in bp["boxes"]:
        b.set(facecolor="#d6e4f2", edgecolor=INK_MUTED, linewidth=0.9)
    ax.set_xticks(range(1, len(conds) + 1))
    ax.set_xticklabels([f"{c}\nn={len(d)}" for c, d in zip(conds, data)],
                       fontsize=8.5)
    ax.set_ylabel("patient accuracy")
    ax.set_title("Experiment 6A: patient accuracy by prediction-history condition\n"
                 "cohorts DIFFER across conditions -- descriptive only, "
                 "not a paired comparison", fontsize=11)
    say(f"  wrote {_finish(fig, out / 'fig2_patient_accuracy_by_history.png').name}")


def fig_paired_differences(elig: pd.DataFrame, pairs: list, paired: pd.DataFrame,
                           out: Path, name: str, title: str) -> None:
    pairs = pairs[:min(len(pairs), 9)]
    ncol = 2 if len(pairs) <= 4 else 3
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow),
                             squeeze=False)
    for ax, (label, cx, cy) in zip(axes.ravel(), pairs):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        x = elig[elig["condition"] == cx].set_index(KEY)
        y = elig[elig["condition"] == cy].set_index(KEY)
        common = x.index.intersection(y.index)
        d = pd.DataFrame({"PATNO": common.get_level_values("PATNO").to_numpy(),
                          "cx": x.loc[common, "correct"].to_numpy(int),
                          "cy": y.loc[common, "correct"].to_numpy(int)})
        per = d.groupby("PATNO").agg(n=("cx", "size"), sx=("cx", "sum"),
                                     sy=("cy", "sum"))
        diff = (per["sx"] / per["n"] - per["sy"] / per["n"]).to_numpy(float)
        r = paired[paired["comparison"] == label].iloc[0]
        #  The zero bar holds the large majority of patients and would flatten
        #  everything else to invisibility, so the patients who did NOT move
        #  are reported as a number and only the movers are drawn.
        moved = diff[diff != 0.0]
        n_same = int((diff == 0.0).sum())
        ax.hist(moved, bins=np.linspace(-1, 1, 33), color=COOL,
                edgecolor="white", linewidth=0.6, zorder=3)
        ax.axvline(0.0, color=INK_MUTED, linewidth=1.0, zorder=4)
        ax.axvline(float(r["mean_paired_difference"]), color=WARM,
                   linewidth=1.6, linestyle="--", zorder=5)
        head = "\n".join(textwrap.wrap(label, 44))
        ax.set_title(f"{head}\nmean {r['mean_paired_difference']:+.4f} "
                     f"[{r['diff_ci_lo']:+.4f}, {r['diff_ci_hi']:+.4f}]\n"
                     f"{len(moved)} of {int(r['n_matched_patients'])} patients "
                     f"moved, {n_same} unchanged", fontsize=8)
        ax.set_xlabel("patient accuracy difference (movers only)", fontsize=8.5)
        ax.set_ylabel("patients who moved", fontsize=8.5)
    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    fig.suptitle(title + "\ndashed line: the mean over ALL matched patients, "
                 "including the unchanged ones", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    say(f"  wrote {_finish(fig, out / name).name}")


def fig_accuracy_vs_count(pt: pd.DataFrame, vc: pd.DataFrame, out: Path) -> None:
    s = pt[pt["condition"] == FULL]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    _style(ax)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    #  Horizontal jitter only, so the accuracy value is never distorted.
    jit = rng.uniform(-0.18, 0.18, len(s))
    ax.scatter(s["n_eligible"] + jit, s["patient_accuracy"], s=9,
               color=COOL, alpha=0.35, edgecolors="none", zorder=3)
    grp = s.groupby("n_eligible")["patient_accuracy"].agg(["mean", "size"])
    grp = grp[grp["size"] >= 5]
    ax.plot(grp.index, grp["mean"], color=ACCENT, linewidth=1.8,
            marker="o", markersize=4, zorder=5,
            label="mean patient accuracy (counts with n>=5 patients)")
    ax.set_xlabel("eligible target visits for the patient")
    ax.set_ylabel("patient accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Experiment 6A, full history: patient accuracy against the "
                 "number of eligible predictions", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))
    say(f"  wrote {_finish(fig, out / 'fig5_accuracy_vs_visit_count.png').name}")


def fig_direction_bars(p6a: pd.DataFrame, p6b: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for ax, tab, title in ((axes[0], p6a, "Experiment 6A"),
                           (axes[1], p6b, "Experiment 6B")):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        labs = tab["comparison"].tolist()
        ypos = np.arange(len(labs))[::-1]
        left = np.zeros(len(labs))
        for col, colour, lab in (("pct_worsened", WARM, "worsened"),
                                 ("pct_unchanged", "#cfcec9", "unchanged"),
                                 ("pct_improved", COOL, "improved")):
            v = tab[col].to_numpy(float)
            ax.barh(ypos, v, left=left, color=colour, height=0.62,
                    zorder=3, label=lab, edgecolor="white", linewidth=0.6)
            left = left + v
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{l}  (n={int(n)})" for l, n in
                            zip(labs, tab["n_matched_patients"])], fontsize=8)
        ax.set_xlabel("per cent of matched patients")
        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, labelcolor=INK_MUTED,
               ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Direction of the per-patient accuracy change in each "
                 "matched comparison", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    say(f"  wrote {_finish(fig, out / 'fig6_improved_worsened_unchanged.png').name}")


def fig_subtype_rates(pt: pd.DataFrame, out: Path) -> None:
    conds = [FULL, "last 2", "last 1"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    for ax, (col, lab, flt) in zip(axes, (
            ("patient_PIGD_sensitivity", "per-patient PIGD sensitivity",
             "n_PIGD_targets"),
            ("patient_TD_specificity", "per-patient TD specificity",
             "n_TD_targets"))):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        bins = np.linspace(0, 1, 6)
        xpos = np.arange(len(bins) - 1)
        width = 0.27
        for i, (c, colour) in enumerate(zip(conds, (ACCENT, COOL, MUTED))):
            s = pt[(pt["condition"] == c) & pt[col].notna()]
            h, _ = np.histogram(s[col].to_numpy(float), bins=bins)
            ax.bar(xpos + (i - 1) * width, 100.0 * h / h.sum(), width=width,
                   color=colour, zorder=3, edgecolor="white", linewidth=0.6,
                   label=f"{c} (n={len(s)} patients)")
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{bins[j]:.1f}-{bins[j + 1]:.1f}"
                            for j in xpos], fontsize=8.5)
        ax.set_xlabel(lab + "  (the final bin includes 1.0)")
        ax.set_ylabel("per cent of patients with targets of that class")
        ax.set_title(f"{lab}\npatients without a target of that class are "
                     f"excluded, not scored zero", fontsize=10)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED)
    fig.tight_layout()
    say(f"  wrote {_finish(fig, out / 'fig7_patient_subtype_rates.png').name}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-repro", action="store_true",
                    help="skip the second identical analysis pass")
    args = ap.parse_args()
    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    seed = cfg.seed

    section("1. SOURCES")
    r6a, r6b = ROOT / RUN_6A, ROOT / RUN_6B
    files_exist = (r6a / PRED_6A).is_file() and (r6b / PRED_6B).is_file()
    say(f"  6A run   {RUN_6A}")
    say(f"  6A preds {PRED_6A}  exists={bool((r6a / PRED_6A).is_file())}")
    say(f"  6B run   {RUN_6B}")
    say(f"  6B preds {PRED_6B}  exists={bool((r6b / PRED_6B).is_file())}")
    man6a, man6b = verify_manifest(r6a), verify_manifest(r6b)
    say(f"  6A MANIFEST.sha256 verified {man6a[0]}/{man6a[1]} files")
    say(f"  6B MANIFEST.sha256 verified {man6b[0]}/{man6b[1]} files")
    say("  This script reads those files and writes nothing into either run.")

    section("2. ELIGIBILITY, RE-APPLIED FROM THE PROJECT DEFINITION")
    e6a, raw6a = load_eligible(r6a, PRED_6A, "window")
    e6b, raw6b = load_eligible(r6b, PRED_6B, "condition")
    n_ind_6a = int((raw6a["actual_next_label"] == "Indeterminate").sum())
    n_ind_6b = int((raw6b["actual_next_label"] == "Indeterminate").sum())
    say(f"  6A rows {len(raw6a)} -> eligible {len(e6a)}  "
        f"(Indeterminate target rows dropped: {n_ind_6a})")
    say(f"  6B rows {len(raw6b)} -> eligible {len(e6b)}  "
        f"(Indeterminate target rows dropped: {n_ind_6b})")
    say("  Rule: actual_next_label in {TD, PIGD} AND two_class_defined.")
    say("  `y` is 0 for Indeterminate as well as TD, so it is not an")
    say("  eligibility flag and is not used as one here.")

    src6a = pd.read_csv(r6a / "cohort_counts.csv")
    mine6a = (e6a.groupby("condition")
              .agg(n_targets=("correct", "size"), n_patients=("PATNO", "nunique"))
              .reset_index().rename(columns={"condition": "window"}))
    chk_cohort = src6a.merge(mine6a, on="window", suffixes=("_src", "_mine"))
    cohort_agree = bool(
        (chk_cohort["n_targets_src"] == chk_cohort["n_targets_mine"]).all()
        and (chk_cohort["n_patients_src"] == chk_cohort["n_patients_mine"]).all())
    say(f"\n  eligible counts reproduce 6A cohort_counts.csv: {cohort_agree}")
    say(chk_cohort[["window", "n_targets_src", "n_targets_mine",
                    "n_patients_src", "n_patients_mine"]].to_string(index=False))

    section("3. ANALYSIS")
    res = analyse(e6a, e6b, args.n_boot, seed)
    pats = res["patients"]
    pats.to_csv(out / "patient_metrics_by_condition.csv", index=False)
    pd.concat([res["dist_6a"].assign(experiment="6A"),
               res["dist_6b"].assign(experiment="6B")], ignore_index=True
              ).to_csv(out / "patient_accuracy_distribution.csv", index=False)
    paired = pd.concat([res["paired_6a"], res["paired_6b"]], ignore_index=True)
    paired.to_csv(out / "paired_patient_comparisons.csv", index=False)
    res["subtype"].to_csv(out / "patient_subtype_metrics.csv", index=False)
    res["visit_counts"].to_csv(out / "patient_accuracy_by_visit_count.csv",
                               index=False)
    res["composition"].to_csv(out / "patient_accuracy_by_subtype_composition.csv",
                              index=False)

    for tag, dist in (("6A", res["dist_6a"]), ("6B", res["dist_6b"])):
        say(f"\n  --- {tag}: patient accuracy distribution by condition ---")
        say(dist[["condition", "n_patients", "n_eligible_targets",
                  "visit_level_accuracy", "mean_patient_accuracy",
                  "mean_ci_lo", "mean_ci_hi", "median_patient_accuracy",
                  "sd_patient_accuracy", "q25", "q75", "iqr",
                  "min_patient_accuracy", "max_patient_accuracy"]]
            .round(4).to_string(index=False))
        say(f"\n  --- {tag}: mutually exclusive accuracy bands (per cent of patients) ---")
        cols = ["condition"] + [f"pct_{b}" for b in BANDS]
        say(dist[cols].round(2).to_string(index=False))
        say(f"  cumulative, as the brief also names them:")
        say(dist[["condition", "pct_below_50pct_cumulative",
                  "pct_at_least_80pct_cumulative"]].round(2).to_string(index=False))

    for tag, tab in (("6A", res["paired_6a"]), ("6B", res["paired_6b"])):
        say(f"\n  --- {tag}: matched paired comparisons (patients AND targets matched) ---")
        say(tab[["comparison", "n_matched_patients", "n_matched_targets",
                 "mean_patient_accuracy_x", "mean_patient_accuracy_y",
                 "mean_paired_difference", "diff_ci_lo", "diff_ci_hi",
                 "diff_excludes_zero", "median_paired_difference",
                 "n_improved", "n_worsened", "n_unchanged"]]
            .round(4).to_string(index=False))

    say("\n  --- patient accuracy by number of eligible predictions ---")
    say(res["visit_counts"][res["visit_counts"]["condition"].isin(
        [FULL, "last 2", "A: full-trained, full history",
         "D: window-trained, last 2"])][
        ["experiment", "condition", "visit_count_bin", "n_patients",
         "n_eligible_targets", "mean_patient_accuracy",
         "median_patient_accuracy", "visit_level_accuracy",
         "pct_exactly_100", "pct_exactly_0"]].round(4).to_string(index=False))

    say("\n  --- patient accuracy by subtype composition of the targets ---")
    say(res["composition"][res["composition"]["condition"].isin(
        [FULL, "last 2"])][
        ["condition", "subtype_composition", "n_patients",
         "n_eligible_targets", "mean_patient_accuracy",
         "visit_level_accuracy", "median_eligible_per_patient"]]
        .round(4).to_string(index=False))

    say("\n  --- per-patient class-specific rates ---")
    say(res["subtype"][["experiment", "condition",
                        "n_patients_with_PIGD_targets",
                        "mean_patient_PIGD_sensitivity",
                        "n_PIGD_patients_zero_correct",
                        "pct_PIGD_patients_zero_correct",
                        "n_patients_with_TD_targets",
                        "mean_patient_TD_specificity",
                        "n_TD_patients_zero_correct",
                        "pct_TD_patients_zero_correct"]]
        .round(4).to_string(index=False))

    section("4. FIGURES")
    pt6a = pats[pats["experiment"] == "6A"]
    fig_accuracy_histogram(pt6a, res["dist_6a"], out)
    fig_accuracy_boxplot(pt6a, out)
    fig_paired_differences(e6a, PAIRS_6A, res["paired_6a"], out,
                           "fig3_paired_differences_6a.png",
                           "Experiment 6A: per-patient accuracy differences, "
                           "matched patients and matched target visits")
    fig_paired_differences(e6b, PAIRS_6B, res["paired_6b"], out,
                           "fig4_paired_differences_6b.png",
                           "Experiment 6B: per-patient accuracy differences, "
                           "matched patients and matched target visits")
    fig_accuracy_vs_count(pt6a, res["visit_counts"], out)
    fig_direction_bars(res["paired_6a"], res["paired_6b"], out)
    fig_subtype_rates(pt6a, out)

    section("5. VALIDATION")
    #  (a) metrics recompute from the saved CSVs, not from memory
    reread = pd.read_csv(out / "patient_metrics_by_condition.csv")
    redist = distribution_summary(reread[reread["experiment"] == "6A"],
                                  ORDER_6A, args.n_boot, seed)
    saved_dist = pd.read_csv(out / "patient_accuracy_distribution.csv")
    saved_6a = saved_dist[saved_dist["experiment"] == "6A"].reset_index(drop=True)
    recompute_ok = bool(np.allclose(
        redist[["mean_patient_accuracy", "median_patient_accuracy",
                "visit_level_accuracy", "sd_patient_accuracy", "iqr",
                "mean_ci_lo", "mean_ci_hi"]].to_numpy(float),
        saved_6a[["mean_patient_accuracy", "median_patient_accuracy",
                  "visit_level_accuracy", "sd_patient_accuracy", "iqr",
                  "mean_ci_lo", "mean_ci_hi"]].to_numpy(float),
        rtol=0, atol=1e-12))

    #  (b) the bootstrap resamples PATIENTS, carrying their visits
    s_full = pt6a[pt6a["condition"] == FULL]
    nv = s_full["n_eligible"].to_numpy(int)
    idx = boot_indices(len(nv), 200, seed)
    rep_visits = nv[idx].sum(1)
    boot_is_patient_level = bool(
        idx.shape == (200, len(nv))
        and idx.max() < len(nv)
        and len(np.unique(rep_visits)) > 1          # visit totals vary ...
        and (idx.shape[1] == len(nv)))              # ... patient count fixed
    boot_visit_lo, boot_visit_hi = int(rep_visits.min()), int(rep_visits.max())
    say(f"  bootstrap replicate patient count fixed at {len(nv)}, "
        f"replicate visit totals range {boot_visit_lo}-{boot_visit_hi} "
        f"(a visit-level bootstrap would hold the visit total fixed instead)")

    #  (c) liveness: the metric must react to a change in the predictions
    tampered = e6a.copy()
    victim = s_full.sort_values("n_eligible", ascending=False)["PATNO"].iloc[0]
    sel = (tampered["condition"] == FULL) & (tampered["PATNO"] == victim)
    tampered.loc[sel, "correct"] = 1 - tampered.loc[sel, "correct"]
    t_dist = distribution_summary(patient_table(tampered, "6A"), [FULL],
                                  50, seed)
    base_mean = float(res["dist_6a"].set_index("condition")
                      .loc[FULL, "mean_patient_accuracy"])
    tamper_mean = float(t_dist.set_index("condition")
                        .loc[FULL, "mean_patient_accuracy"])
    n_vic = int(sel.sum())
    tamper_live = bool(abs(tamper_mean - base_mean) > 1e-9)
    say(f"  liveness: flipping all {n_vic} predictions of one patient moved the "
        f"mean patient accuracy {base_mean:.6f} -> {tamper_mean:.6f}")

    #  (d) reproducibility: a second identical pass
    if args.skip_repro:
        repro = "skipped"
    else:
        again = analyse(e6a, e6b, args.n_boot, seed)
        repro = all(res[k].equals(again[k]) for k in res)
        say(f"  second identical analysis pass reproduces every table: {repro}")

    #  (e) matched comparisons really do share patients and targets
    matched_ok = True
    for elig, pairs in ((e6a, PAIRS_6A), (e6b, PAIRS_6B)):
        for label, cx, cy in pairs:
            x = set(map(tuple, elig[elig["condition"] == cx][KEY].values))
            y = set(map(tuple, elig[elig["condition"] == cy][KEY].values))
            common = x & y
            r = paired[paired["comparison"] == label].iloc[0]
            px = {t[0] for t in common}
            if (int(r["n_matched_targets"]) != len(common)
                    or int(r["n_matched_patients"]) != len(px)):
                matched_ok = False

    #  Recorded so the report can cite it: keying on visit_index instead of the
    #  project key would report this many duplicate rows, all of them persistence.
    n_phantom_dupes = int(raw6a.duplicated(["PATNO", "visit_index", "window"],
                                           keep=False).sum())
    say(f"  keying on visit_index instead of the project key would report "
        f"{n_phantom_dupes} duplicate rows in 6A, all persistence rows")

    denom_ok = bool((pats["n_eligible"]
                     == pats["n_TD_targets"] + pats["n_PIGD_targets"]).all())
    per_cond_tot = (pats.groupby(["experiment", "condition"])["n_eligible"].sum())
    elig_tot = pd.concat([
        e6a.assign(experiment="6A"), e6b.assign(experiment="6B")]
    ).groupby(["experiment", "condition"]).size()
    denom_total_ok = bool(per_cond_tot.sort_index().equals(elig_tot.sort_index()))

    na_ok = bool(
        (pats.loc[pats["n_PIGD_targets"] == 0, "patient_PIGD_sensitivity"]
         .isna().all())
        and (pats.loc[pats["n_TD_targets"] == 0, "patient_TD_specificity"]
             .isna().all()))

    bands_ok = bool(
        (pats.groupby(["experiment", "condition"])["accuracy_band"].size()
         == pats.groupby(["experiment", "condition"])["accuracy_band"].count()).all()
        and set(pats["accuracy_band"]) <= set(BANDS)
        and all(
            int(d[[f"n_{b}" for b in BANDS]].sum()) == int(d["n_patients"])
            for _, d in pd.concat([res["dist_6a"], res["dist_6b"]]).iterrows()))

    checks = {
        "both source prediction files exist": bool(files_exist),
        "6A source MANIFEST.sha256 verifies for every recorded file":
            man6a[0] == man6a[1] and man6a[1] > 0,
        "6B source MANIFEST.sha256 verifies for every recorded file":
            man6b[0] == man6b[1] and man6b[1] > 0,
        "patient identifiers are present and non-missing":
            bool(pats["PATNO"].notna().all()
                 and e6a["PATNO"].notna().all() and e6b["PATNO"].notna().all()),
        "saved y agrees with y recomputed from actual_next_label":
            bool((e6a["y"] == e6a["y_recomputed"]).all()
                 and (e6b["y"] == e6b["y_recomputed"]).all()),
        "saved pred_bin agrees with the label recomputed from predicted_state":
            bool((e6a["pred_bin"] == e6a["pred_recomputed"]).all()
                 and (e6b["pred_bin"] == e6b["pred_recomputed"]).all()),
        "correctness recomputes as (true label == predicted label)":
            bool((e6a["correct"] == (e6a["y"] == e6a["pred_bin"]).astype(int)).all()
                 and (e6b["correct"] == (e6b["y"] == e6b["pred_bin"]).astype(int)).all()),
        "condition labels are exactly the expected sets":
            bool(set(e6a["condition"]) == set(ORDER_6A)
                 and set(e6b["condition"]) == set(ORDER_6B)),
        "no duplicate (PATNO, current_event, target_event, condition) rows":
            bool(e6a.duplicated(KEY + ["condition"]).sum() == 0
                 and e6b.duplicated(KEY + ["condition"]).sum() == 0),
        "every apparent duplicate on visit_index is a persistence row only":
            bool(set(raw6a[raw6a.duplicated(["PATNO", "visit_index", "window"],
                                            keep=False)]["window"]) <= {PERSIST}
                 and n_phantom_dupes > 0),
        "Indeterminate targets are excluded from the binary cohort":
            bool(set(e6a["actual_next_label"]) == {"TD", "PIGD"}
                 and set(e6b["actual_next_label"]) == {"TD", "PIGD"}),
        "eligible counts reproduce the 6A cohort_counts.csv of the source run":
            cohort_agree,
        "6A and 6B agree on the eligible target set they share":
            bool(set(map(tuple, e6a[e6a.condition == FULL][KEY].values))
                 == set(map(tuple, e6b[e6b.condition == ORDER_6B[0]][KEY].values))),
        "patient accuracy denominators equal n_TD + n_PIGD for every patient":
            denom_ok,
        "patient denominators sum to the eligible target count per condition":
            denom_total_ok,
        "accuracy bands are mutually exclusive and sum to the patient count":
            bands_ok,
        "patients without a target of a class are NaN, not zero":
            na_ok,
        "matched comparisons use identical patient and target sets":
            matched_ok,
        "patient bootstrap draws whole patients, not individual visits":
            boot_is_patient_level,
        "bootstrap intervals are non-degenerate":
            bool((res["dist_6a"]["mean_ci_hi"]
                  > res["dist_6a"]["mean_ci_lo"]).all()),
        "the accuracy metric reacts to a change in the saved predictions":
            tamper_live,
        "all reported distribution metrics recompute from the written CSV":
            recompute_ok,
        "a second identical analysis pass reproduces every table":
            repro if repro == "skipped" else bool(repro),
    }
    hard = {k: v for k, v in checks.items() if v != "skipped"}
    for n, v in checks.items():
        tag = "SKIP" if v == "skipped" else ("PASS" if v else "FAIL")
        say(f"    [{tag}]  {n}")
    pd.DataFrame([{"check": k, "result": str(v)} for k, v in checks.items()]
                 ).to_csv(out / "validation_checks.csv", index=False)
    n_pass = sum(1 for v in hard.values() if v)
    n_fail = sum(1 for v in hard.values() if not v)
    n_skip = len(checks) - len(hard)
    say(f"\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    (out / "validation_results.json").write_text(json.dumps(
        {"checks": {k: str(v) for k, v in checks.items()},
         "n_passed": n_pass, "n_failed": n_fail, "n_skipped": n_skip,
         "source_manifests": {"6A": {"verified": man6a[0], "recorded": man6a[1]},
                              "6B": {"verified": man6b[0], "recorded": man6b[1]}},
         "liveness": {"patient_predictions_flipped": n_vic,
                      "mean_patient_accuracy_before": base_mean,
                      "mean_patient_accuracy_after": tamper_mean},
         "bootstrap": {"n_replicates": args.n_boot, "seed": seed,
                       "patients_per_replicate": int(len(nv)),
                       "replicate_visit_total_min": boot_visit_lo,
                       "replicate_visit_total_max": boot_visit_hi},
         "n_phantom_duplicates_if_keyed_on_visit_index": n_phantom_dupes},
        indent=2, default=float), encoding="utf-8")
    if n_fail:
        raise SystemExit(f"CRITICAL CHECK FAILED: "
                         f"{[k for k, v in hard.items() if not v]}")

    section("6. PROVENANCE")
    env = {"python": sys.version.split()[0], "numpy": np.__version__,
           "pandas": pd.__version__, "node": platform.node(),
           "n_bootstrap": args.n_boot, "seed": seed,
           "source_6A": RUN_6A, "source_6B": RUN_6B,
           "analysis_only": True, "models_fitted": 0,
           "elapsed_seconds": round(time.time() - started, 1)}
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:                                   # noqa: BLE001
        env["git_commit"] = "unknown"
    for k, v in env.items():
        say(f"  {k:<22} {v}")

    (out / "summary.json").write_text(json.dumps(
        {"environment": env,
         "checks": {k: str(v) for k, v in checks.items()},
         "code_sha256": {str(Path(__file__).relative_to(ROOT)):
                         sha(Path(__file__))},
         "source_prediction_sha256": {
             f"{RUN_6A}/{PRED_6A}": sha(r6a / PRED_6A),
             f"{RUN_6B}/{PRED_6B}": sha(r6b / PRED_6B)},
         "n_indeterminate_target_rows_excluded": {"6A": n_ind_6a, "6B": n_ind_6b},
         "conditions": {"6A": ORDER_6A, "6B": ORDER_6B},
         "distribution_6A": res["dist_6a"].to_dict(orient="records"),
         "distribution_6B": res["dist_6b"].to_dict(orient="records"),
         "paired": paired.to_dict(orient="records"),
         "subtype": res["subtype"].to_dict(orient="records"),
         "by_visit_count": res["visit_counts"].to_dict(orient="records"),
         "by_composition": res["composition"].to_dict(orient="records")},
        indent=2, default=float), encoding="utf-8")
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    skip = {"MANIFEST.sha256", "MANIFEST.json", "driver.log"}
    files = sorted(p for p in out.rglob("*")
                   if p.is_file() and p.name not in skip)
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files),
        encoding="utf-8")
    (out / "MANIFEST.json").write_text(json.dumps(
        {"generated_files": [
            {"path": str(p.relative_to(out)), "bytes": p.stat().st_size,
             "sha256": sha(p)} for p in files],
         "n_files": len(files)}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
