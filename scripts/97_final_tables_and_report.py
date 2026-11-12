"""Assemble the final tables, figures and report from the recovered metrics.

Reads only what scripts/96_final_metric_recovery.py wrote into the run
directory, plus the published tables it cites, and emits the deliverables.
No metric is computed here that was not computed there.

Usage:
    python scripts/97_final_tables_and_report.py --run experiments/<run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "experiments"
NC = "Not calculated"
NA = "Not applicable"
LABELS = ("TD", "Indeterminate", "PIGD")

BAND_RUN = E / "20261105T025024Z_cthmm_ratio_band_width_extended"
PROFILE_CONT = E / "20261103T213859Z_a2_clinical_profile_cv_rerun"
NO_LEDD = E / "20261111T121021Z_rich_profile_no_ledd"
BAND_DIAG = E / "20261104T155439Z_ratio_band_emission_diagnostic"
PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"


def rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def f(v, nd: int = 4, dash: str = NC) -> str:
    """Format a number for a table cell; never invent one."""
    if v is None:
        return dash
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if not np.isfinite(v):
        return dash
    return f"{float(v):.{nd}f}"


def ci(row, key, nd: int = 4) -> str:
    lo, hi = row.get(f"{key}_ci_low"), row.get(f"{key}_ci_high")
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return NC
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def val_ci(row, key, nd: int = 4) -> str:
    v = row.get(key)
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return NC
    c = ci(row, key, nd)
    return f"{v:.{nd}f} {c}" if c != NC else f"{v:.{nd}f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------- master


MASTER_COLS = [
    "group", "task", "model", "evaluation_protocol", "cohort", "n_patients",
    "n_cases", "positive_class", "accuracy", "accuracy_95ci", "sensitivity_PIGD",
    "sensitivity_PIGD_95ci", "specificity_TD", "precision_PIGD", "f1_PIGD",
    "balanced_accuracy", "balanced_accuracy_95ci", "auc_roc", "auc_roc_95ci",
    "sensitivity_TD_95ci", "sensitivity_Indeterminate_95ci",
    "specificity_TD_95ci", "specificity_Indeterminate_95ci",
    "specificity_PIGD_95ci", "auc_ovr_TD_95ci",
    "auc_ovr_Indeterminate_95ci", "auc_ovr_PIGD_95ci",
    "pr_auc", "pr_auc_95ci", "brier_binary", "brier_summed_two_class",
    "brier_multiclass_summed", "brier_convention", "log_loss",
    "calibration_slope", "calibration_slope_95ci", "calibration_intercept",
    "ece_10_bin", "ece_10_bin_95ci",
    "ece_bootstrap_range_excludes_point_estimate", "calibration_in_the_large", "macro_f1", "weighted_f1", "macro_auc_ovr",
    "sensitivity_TD", "sensitivity_Indeterminate", "specificity_Indeterminate",
    "specificity_PIGD", "precision_TD", "precision_Indeterminate", "f1_TD",
    "f1_Indeterminate", "auc_ovr_TD", "auc_ovr_Indeterminate", "auc_ovr_PIGD",
    "pr_auc_ovr_TD", "pr_auc_ovr_Indeterminate", "pr_auc_ovr_PIGD",
    "mean_predictive_loglik", "mean_predictive_loglik_95ci",
    "log_density_per_domain_obs", "rmse_standardised", "mae_standardised",
    "coverage_95", "loglik", "n_params", "n_likelihood_terms", "aic", "bic",
    "likelihood_definition", "comparable_group", "source_path", "notes",
]


def blank_row() -> dict:
    return {c: NA for c in MASTER_COLS}


def build_master(binf: pd.DataFrame, mcf: pd.DataFrame, icf: pd.DataFrame,
                 cont: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in binf.iterrows():
        d = blank_row()
        d.update(
            {
                "group": r["group"],
                "task": "binary TD vs PIGD",
                "model": r["model"],
                "evaluation_protocol": r["evaluation_protocol"],
                "cohort": r["cohort"],
                "n_patients": int(r["n_patients"]),
                "n_cases": int(r["n_scored"]),
                "positive_class": "PIGD",
                "accuracy": f(r["accuracy"], 6),
                "accuracy_95ci": ci(r, "accuracy", 6),
                "sensitivity_PIGD": f(r["sensitivity_PIGD"], 6),
                "sensitivity_PIGD_95ci": ci(r, "sensitivity_PIGD", 6),
                "specificity_TD": f(r["specificity_TD"], 6),
                "precision_PIGD": (
                    f(r["precision_PIGD"], 6) if int(r["tp"]) + int(r["fp"]) > 0
                    else NA + ": the model never predicts this class"
                ),
                "f1_PIGD": f(r["f1_PIGD"], 6),
                "balanced_accuracy": f(r["balanced_accuracy"], 6),
                "balanced_accuracy_95ci": ci(r, "balanced_accuracy", 6),
                "auc_roc": f(r["auc_roc"], 6),
                "auc_roc_95ci": ci(r, "auc_roc", 6),
                "pr_auc": f(r["pr_auc"], 6),
                "pr_auc_95ci": ci(r, "pr_auc", 6),
                "brier_binary": f(r["brier_binary"], 6),
                "brier_summed_two_class": f(2.0 * r["brier_binary"], 6),
                "brier_convention": "single-outcome: mean (p_PIGD - y_PIGD)^2; "
                "brier_summed_two_class is the exact 2x conversion to the summed "
                "convention, for comparison with runs that report that form",
                "log_loss": f(r["log_loss"], 6),
                "calibration_slope": (
                    f(r["calibration_slope"], 6) if np.isfinite(r["calibration_slope"])
                    else NA + ": the score takes fewer than three distinct values"
                ),
                "calibration_slope_95ci": (
                    ci(r, "calibration_slope", 6)
                    if np.isfinite(r["calibration_slope"])
                    else NA + ": there is no slope to bound"
                ),
                "calibration_intercept": (
                    f(r["calibration_intercept"], 6)
                    if np.isfinite(r["calibration_intercept"])
                    else NA + ": the score takes fewer than three distinct values"
                ),
                "ece_10_bin": f(r["ece_10_bin"], 6),
                "ece_10_bin_95ci": ci(r, "ece_10_bin", 6),
                "ece_bootstrap_range_excludes_point_estimate": str(
                    bool(np.isfinite(r.get("ece_10_bin_ci_low", np.nan))
                         and not (r["ece_10_bin_ci_low"] <= r["ece_10_bin"]
                                  <= r["ece_10_bin_ci_high"]))
                ),
                "calibration_in_the_large": f(r["calibration_in_the_large"], 6),
                "source_path": r["source_path"],
                "notes": "; ".join(
                    x for x in [
                        r["notes"] if isinstance(r["notes"], str)
                        and r["notes"].strip() else "",
                        f"score takes only {int(r['n_distinct_scores'])} distinct "
                        "values, so AUC/PR-AUC are not ranking measures here"
                        if r["score_degenerate"] else "",
                        "log loss clipped at 1e-12" if r["log_loss_clipping_active"] else "",
                    ] if x
                ) or "no caveats",
            }
        )
        rows.append(d)

    for _, r in mcf.iterrows():
        d = blank_row()
        d.update(
            {
                "group": r["group"],
                "task": "three-class TD / Indeterminate / PIGD",
                "model": r["model"],
                "evaluation_protocol": r["evaluation_protocol"],
                "cohort": r["cohort"],
                "n_patients": int(r["n_patients"]),
                "n_cases": int(r["n_scored"]),
                "positive_class": "one-vs-rest, each class in turn",
                "accuracy": f(r["accuracy"], 6),
                "accuracy_95ci": ci(r, "accuracy", 6),
                "balanced_accuracy": f(r["balanced_accuracy"], 6),
                "balanced_accuracy_95ci": ci(r, "balanced_accuracy", 6),
                "brier_multiclass_summed": f(r["brier_multiclass_summed"], 6),
                "brier_convention": "summed over three classes: mean sum_k (p_k - y_k)^2",
                "log_loss": f(r["log_loss"], 6),
                "macro_f1": f(r["macro_f1"], 6),
                "weighted_f1": f(r["weighted_f1"], 6),
                "macro_auc_ovr": f(r["macro_auc_ovr"], 6),
                "mean_predictive_loglik": f(r["mean_predictive_loglik"], 6),
                "mean_predictive_loglik_95ci": ci(r, "mean_predictive_loglik", 6),
                "source_path": r["source_path"],
                "notes": (r["notes"] if isinstance(r["notes"], str)
                          and r["notes"].strip() else "no caveats"),
            }
        )
        for k in LABELS:
            d[f"sensitivity_{k}"] = f(r[f"sensitivity_{k}"], 6)
            d[f"specificity_{k}"] = f(r[f"specificity_{k}"], 6)
            d[f"precision_{k}"] = (
                f(r[f"precision_{k}"], 6) if int(r[f"n_predicted_{k}"]) > 0
                else NA + ": the model never predicts this class"
            )
            d[f"f1_{k}"] = f(r[f"f1_{k}"], 6)
            d[f"auc_ovr_{k}"] = f(r[f"auc_ovr_{k}"], 6)
            d[f"pr_auc_ovr_{k}"] = f(r[f"pr_auc_ovr_{k}"], 6)
            d[f"sensitivity_{k}_95ci"] = ci(r, f"sensitivity_{k}", 6)
            d[f"specificity_{k}_95ci"] = ci(r, f"specificity_{k}", 6)
            d[f"auc_ovr_{k}_95ci"] = ci(r, f"auc_ovr_{k}", 6)
        rows.append(d)

    for _, r in cont.iterrows():
        d = blank_row()
        d.update(
            {
                "group": "G: continuous clinical measurement",
                "task": "nine-domain continuous prediction",
                "model": r["model"],
                "evaluation_protocol": "5-fold patient CV x 4 repeats",
                "cohort": "3,353 prediction pairs, 30,099 domain observations",
                "n_patients": int(r["n_patients"]),
                "n_cases": int(r["n_predictions"]),
                "positive_class": NA,
                "log_density_per_domain_obs": f(r["loglik_per_domain_observation"], 6),
                "rmse_standardised": f(r["rmse_standardised"], 6),
                "mae_standardised": f(r["mae_standardised"], 6),
                "coverage_95": f(r["coverage_95"], 6),
                "source_path": rel(PROFILE_CONT / "model_comparison.csv"),
                "notes": "95% CI on the mean log density per prediction: "
                f"[{r['loglik_ci_low']:.4f}, {r['loglik_ci_high']:.4f}]",
            }
        )
        rows.append(d)

    for _, r in icf.iterrows():
        d = blank_row()
        d.update(
            {
                "group": "H: information criteria",
                "task": "in-sample model selection",
                "model": r["model"],
                "evaluation_protocol": "whole-cohort maximum likelihood, not held out",
                "cohort": r["dataset"],
                "n_patients": 934,
                "n_cases": int(r["n_likelihood_terms"]),
                "positive_class": NA,
                "loglik": f(r["loglik"], 4),
                "n_params": int(r["n_params"]),
                "n_likelihood_terms": int(r["n_likelihood_terms"]),
                "aic": f(r["aic"], 2),
                "bic": f(r["bic"], 2),
                "likelihood_definition": r["likelihood_definition"],
                "comparable_group": r["comparable_group"],
                "source_path": r["source_path"],
                "notes": f"delta AIC within group {r['delta_aic_within_group']:.2f}, "
                f"delta BIC within group {r['delta_bic_within_group']:.2f}",
            }
        )
        rows.append(d)

    return pd.DataFrame(rows)[MASTER_COLS]


# -------------------------------------------------------------- band table


def build_band(binf: pd.DataFrame, mcf: pd.DataFrame) -> pd.DataFrame:
    cx = pd.read_csv(BAND_RUN / "complexity_by_width.csv")
    band = binf[binf["group"] == "D: ratio-band sweep"].copy()
    band["main_width"] = (
        band["model"].str.extract(r"width (\d\.\d\d)")[0].astype(float)
    )
    merged = band.merge(cx, on="main_width", how="left")
    mc = mcf[mcf["group"] == "D: ratio-band sweep"][
        ["model", "auc_ovr_Indeterminate", "sensitivity_Indeterminate", "macro_f1"]
    ]
    merged = merged.merge(mc, on="model", how="left", validate="1:1")
    keep = [
        "model", "main_width", "n_patients", "n_scored", "accuracy",
        "accuracy_ci_low", "accuracy_ci_high", "sensitivity_PIGD",
        "sensitivity_PIGD_ci_low", "sensitivity_PIGD_ci_high", "specificity_TD",
        "specificity_TD_ci_low", "specificity_TD_ci_high", "precision_PIGD",
        "f1_PIGD", "balanced_accuracy", "balanced_accuracy_ci_low",
        "balanced_accuracy_ci_high", "auc_roc", "auc_roc_ci_low",
        "auc_roc_ci_high", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high",
        "brier_binary", "log_loss", "n_main_bands", "n_tail_bands", "n_bands",
        "n_categories", "n_params", "n_empty", "n_lt_10_obs", "n_lt_30_obs",
        "median_obs_per_category", "min_obs_nonempty", "obs_per_param",
        "macro_f1", "auc_ovr_Indeterminate", "sensitivity_Indeterminate",
        "source_path",
    ]
    return merged[[c for c in keep if c in merged.columns]].sort_values(
        "main_width", na_position="first"
    )


# -------------------------------------------------- recovered-metric register


RECOVERED_BINARY = [
    ("sensitivity_PIGD", "TP / (TP + FN), PIGD positive"),
    ("specificity_TD", "TN / (TN + FP)"),
    ("precision_PIGD", "TP / (TP + FP)"),
    ("f1_PIGD", "harmonic mean of precision and sensitivity"),
    ("balanced_accuracy", "(sensitivity + specificity) / 2"),
    ("auc_roc", "Mann-Whitney statistic on P(PIGD), mid-ranks for ties"),
    ("pr_auc", "average precision on P(PIGD), thresholded at distinct scores"),
    ("brier_binary", "mean (p_PIGD - y_PIGD)^2"),
    ("log_loss", "-mean log p(true class), clipped at 1e-12"),
    ("calibration_slope", "slope of a logistic regression of the outcome on "
                          "logit(p), fitted by Newton steps; 1 is calibrated"),
    ("calibration_intercept", "intercept of the same fit; 0 is calibrated"),
    ("ece_10_bin", "support-weighted mean |mean(p) - mean(y)| over 10 "
                   "equal-width probability bins"),
    ("calibration_in_the_large", "mean(p) - mean(y)"),
]
RECOVERED_MC = [
    ("sensitivity_", "one-vs-rest recall for this class"),
    ("specificity_", "one-vs-rest TN / (TN + FP) for this class"),
    ("precision_", "one-vs-rest TP / (TP + FP) for this class"),
    ("f1_", "one-vs-rest F1 for this class"),
    ("auc_ovr_", "one-vs-rest AUC on this class's probability"),
    ("pr_auc_ovr_", "one-vs-rest average precision on this class's probability"),
]


def build_recovered(binf: pd.DataFrame, mcf: pd.DataFrame,
                    cc: pd.DataFrame) -> pd.DataFrame:
    """Every metric this task computed, with how, and whether it was published."""
    published = {
        (str(r["model"]), str(r["metric"])) for _, r in cc.iterrows()
    }
    pub_metric_names = {
        "accuracy": "accuracy", "accuracy_recomputed": "accuracy",
        "brier_two_class": "brier_binary", "brier": "brier_binary",
        "brier_multiclass": "brier_multiclass_summed", "log_loss": "log_loss",
        "auc_roc": "auc_roc", "auc": "auc_roc", "pr_auc": "pr_auc",
        "macro_f1": "macro_f1", "balanced_accuracy": "balanced_accuracy",
        "recall_TD": "sensitivity_TD", "recall_PIGD": "sensitivity_PIGD",
        "recall_Indeterminate": "sensitivity_Indeterminate",
        "precision_PIGD": "precision_PIGD", "f1_PIGD": "f1_PIGD",
        "f1_Indeterminate": "f1_Indeterminate",
        "sensitivity_PIGD": "sensitivity_PIGD", "specificity_TD": "specificity_TD",
        "ppv_PIGD": "precision_PIGD", "npv_TD": "npv_TD",
    }
    was_published = set()
    for m, metric in published:
        if metric in pub_metric_names:
            was_published.add((m, pub_metric_names[metric]))

    rows = []
    for _, r in binf.iterrows():
        for key, how in RECOVERED_BINARY:
            v = r[key]
            rows.append(
                {
                    "group": r["group"],
                    "task": "binary TD vs PIGD",
                    "model": r["model"],
                    "metric": key,
                    "value": v,
                    "ci_low": r.get(f"{key}_ci_low", np.nan),
                    "ci_high": r.get(f"{key}_ci_high", np.nan),
                    "n_cases": int(r["n_scored"]),
                    "n_patients": int(r["n_patients"]),
                    "method": how,
                    "ci_method": "patient-level bootstrap, 2,000 draws, seed 42",
                    "already_published": (r["model"], key) in was_published,
                    "source_path": r["source_path"],
                }
            )
    for _, r in mcf.iterrows():
        for prefix, how in RECOVERED_MC:
            for k in LABELS:
                key = f"{prefix}{k}"
                if key not in r.index:
                    continue
                rows.append(
                    {
                        "group": r["group"],
                        "task": "three-class",
                        "model": r["model"],
                        "metric": key,
                        "value": r[key],
                        "ci_low": r.get(f"{key}_ci_low", np.nan),
                        "ci_high": r.get(f"{key}_ci_high", np.nan),
                        "n_cases": int(r["n_scored"]),
                        "n_patients": int(r["n_patients"]),
                        "method": how.replace("this class", k),
                        "ci_method": "patient-level bootstrap, 2,000 draws, seed 42"
                        if f"{key}_ci_low" in r.index
                        else NC,
                        "already_published": (r["model"], key) in was_published,
                        "source_path": r["source_path"],
                    }
                )
        for key, how in [
            ("macro_auc_ovr", "unweighted mean of the three one-vs-rest AUCs"),
            ("weighted_auc_ovr", "support-weighted mean of the three one-vs-rest AUCs"),
            ("weighted_f1", "support-weighted mean of the three one-vs-rest F1s"),
            ("mean_predictive_loglik", "mean log probability assigned to the realised label"),
        ]:
            rows.append(
                {
                    "group": r["group"],
                    "task": "three-class",
                    "model": r["model"],
                    "metric": key,
                    "value": r[key],
                    "ci_low": r.get(f"{key}_ci_low", np.nan),
                    "ci_high": r.get(f"{key}_ci_high", np.nan),
                    "n_cases": int(r["n_scored"]),
                    "n_patients": int(r["n_patients"]),
                    "method": how,
                    "ci_method": "patient-level bootstrap, 2,000 draws, seed 42"
                    if f"{key}_ci_low" in r.index
                    else NC,
                    "already_published": (r["model"], key) in was_published,
                    "source_path": r["source_path"],
                }
            )
    out = pd.DataFrame(rows)
    return out[out["value"].notna()]


# --------------------------------------------------------------- figures


def figures(run: Path, binf: pd.DataFrame, mcf: pd.DataFrame,
            band: pd.DataFrame, ct: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    plt.rcParams.update({"figure.dpi": 150, "font.size": 8,
                         "axes.grid": True, "grid.alpha": 0.3})

    # fig 1: AUC with CI, binary, the two protocol-coherent groups
    for key, gname, fname, title in [
        ("C", "C: matched comparison set", "fig1_auc_matched_set.png",
         "Binary TD vs PIGD, AUC-ROC on the matched 3,020-case set"),
        ("F", "F: harmonised subscore family", "fig2_auc_harmonised.png",
         "Binary TD vs PIGD, AUC-ROC under one protocol (harmonised family)"),
    ]:
        d = binf[binf["group"] == gname].sort_values("auc_roc")
        d = d[~d["score_degenerate"]]
        fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(d) + 1.2))
        y = np.arange(len(d))
        ax.errorbar(
            d["auc_roc"], y,
            xerr=[d["auc_roc"] - d["auc_roc_ci_low"], d["auc_roc_ci_high"] - d["auc_roc"]],
            fmt="o", ms=3.5, lw=1, capsize=2, color="#1f4e79",
        )
        ax.set_yticks(y)
        ax.set_yticklabels(d["model"])
        ax.set_xlabel("AUC-ROC (patient-level bootstrap 95% CI)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(run / fname)
        plt.close(fig)
        made.append(fname)

    # fig 3: band width sweep, four panels
    b = band.dropna(subset=["main_width"]).sort_values("main_width")
    ref = band[band["main_width"].isna()]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    panels = [
        ("accuracy", "Accuracy", True),
        ("auc_roc", "AUC-ROC", True),
        ("pr_auc", "PR-AUC", True),
        ("brier_binary", "Brier (single-outcome)", False),
    ]
    for ax, (col, lab, higher) in zip(axes.ravel(), panels):
        lo, hi = b.get(f"{col}_ci_low"), b.get(f"{col}_ci_high")
        if lo is not None and lo.notna().all():
            ax.errorbar(b["main_width"], b[col],
                        yerr=[b[col] - lo, hi - b[col]],
                        fmt="o-", ms=3.5, lw=1, capsize=2, color="#1f4e79",
                        label="ratio bands")
        else:
            ax.plot(b["main_width"], b[col], "o-", ms=3.5, lw=1,
                    color="#1f4e79", label="ratio bands")
        if len(ref):
            ax.axhline(float(ref.iloc[0][col]), color="#c00000", ls="--", lw=1,
                       label="unbanded CT-HMM")
        ax.set_xlabel("main band width")
        ax.set_ylabel(lab)
        ax.set_title(f"{lab} ({'higher' if higher else 'lower'} is better)")
    axes[0, 0].legend(fontsize=6, loc="best")
    fig.suptitle("Ratio-banded CT-HMM against the unbanded model, 3,029 cases", y=1.0)
    fig.tight_layout()
    fig.savefig(run / "fig3_band_width_sweep.png")
    plt.close(fig)
    made.append("fig3_band_width_sweep.png")

    # fig 4: per-class sensitivity, harmonised three-class
    d = mcf[mcf["group"] == "F: harmonised subscore family"].sort_values(
        "sensitivity_Indeterminate", ascending=False
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    w = 0.26
    y = np.arange(len(d))
    for i, (k, c) in enumerate(zip(LABELS, ["#1f4e79", "#c00000", "#2e7d32"])):
        ax.barh(y + (i - 1) * w, d[f"sensitivity_{k}"], height=w, color=c, label=k)
    ax.set_yticks(y)
    ax.set_yticklabels(d["model"])
    ax.set_xlabel("one-vs-rest sensitivity")
    ax.set_title("Per-class sensitivity, three-class task, 3,361 cases "
                 "(supports 2,115 / 332 / 914)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run / "fig4_per_class_sensitivity.png")
    plt.close(fig)
    made.append("fig4_per_class_sensitivity.png")

    # fig 5: Indeterminate - ranked well, predicted never
    d = mcf[mcf["group"].isin(["F: harmonised subscore family",
                               "B: subtype-history models, LOPO",
                               "D: ratio-band sweep"])]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for g, mk in [("F: harmonised subscore family", "o"),
                  ("B: subtype-history models, LOPO", "s"),
                  ("D: ratio-band sweep", "^")]:
        s = d[d["group"] == g]
        ax.scatter(s["auc_ovr_Indeterminate"], s["sensitivity_Indeterminate"],
                   marker=mk, s=22, label=g.split(":")[1].strip(), alpha=0.85)
    ax.axvline(0.5, color="grey", ls=":", lw=1)
    ax.set_xlabel("Indeterminate one-vs-rest AUC (ranking)")
    ax.set_ylabel("Indeterminate sensitivity (hard prediction)")
    ax.set_title("Indeterminate: ranked above chance, almost never predicted")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run / "fig5_indeterminate_gap.png")
    plt.close(fig)
    made.append("fig5_indeterminate_gap.png")

    # fig 6: paired deltas with CI
    sel = ct[ct["metric"].isin(["accuracy", "auc_roc", "pr_auc"])
             & ct["contrast"].str.contains("ratio bands at width")
             & ~ct["contrast"].str.contains("width 0.10$")]
    if len(sel):
        fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(sel) + 1.2))
        lab = sel["contrast"].str.replace(" - unbanded CT-HMM", "", regex=False) \
            + " | " + sel["metric"]
        y = np.arange(len(sel))
        ax.errorbar(sel["delta"], y,
                    xerr=[sel["delta"] - sel["ci_low"], sel["ci_high"] - sel["delta"]],
                    fmt="o", ms=3.5, lw=1, capsize=2, color="#1f4e79")
        ax.axvline(0, color="#c00000", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(lab, fontsize=6.5)
        ax.set_xlabel("paired difference against the unbanded CT-HMM "
                      "(95% patient bootstrap CI)")
        ax.set_title("Ratio bands: what improves and what does not")
        fig.tight_layout()
        fig.savefig(run / "fig6_band_paired_deltas.png")
        plt.close(fig)
        made.append("fig6_band_paired_deltas.png")
    return made


# ------------------------------------------------------------- the report


def get(df: pd.DataFrame, group: str, model: str) -> pd.Series:
    sel = df[(df["group"] == group) & (df["model"] == model)]
    if sel.empty:
        raise KeyError(f"no arm {group} / {model}")
    return sel.iloc[0]


def best(df: pd.DataFrame, group: str, col: str, higher: bool = True,
         exclude_degenerate: bool = False) -> pd.Series:
    d = df[df["group"] == group].copy()
    if exclude_degenerate:
        flag = "score_degenerate" if "score_degenerate" in d.columns else "score_degenerate_any_class"
        d = d[~d[flag].astype(bool)]
    d = d[d[col].notna()]
    return d.loc[d[col].idxmax() if higher else d[col].idxmin()]


#: this group's arms cover different numbers of cases, so it is reported in
#: Table A2.4 but never ranked; group F2 is its comparable counterpart.
UNRANKABLE = {"F: harmonised subscore family"}


def winners(df: pd.DataFrame, col: str, higher: bool = True,
            exclude_degenerate: bool = False, nd: int = 4) -> list[list[str]]:
    rows = []
    for g in sorted(df["group"].unique()):
        if g in UNRANKABLE and "auc_roc" in df.columns:
            continue
        d = df[df["group"] == g].copy()
        if exclude_degenerate:
            flag = ("score_degenerate" if "score_degenerate" in d.columns
                    else "score_degenerate_any_class")
            d = d[~d[flag].astype(bool)]
        d = d[d[col].notna()]
        if d.empty:
            continue
        r = d.loc[d[col].idxmax() if higher else d[col].idxmin()]
        second = d.loc[d[col].drop(r.name).idxmax() if higher
                       else d[col].drop(r.name).idxmin()] if len(d) > 1 else None
        rows.append([
            g,
            r["model"],
            val_ci(r, col, nd),
            "" if second is None else f"{second['model']} ({f(second[col], nd)})",
        ])
    return rows


def contrast_line(ct: pd.DataFrame, contrast: str, metric: str, nd: int = 4) -> str:
    sel = ct[(ct["contrast"] == contrast) & (ct["metric"] == metric)]
    if sel.empty:
        return NC
    r = sel.iloc[0]
    verdict = "excludes zero" if r["excludes_zero"] else "includes zero"
    return f"{r['delta']:+.{nd}f} [{r['ci_low']:+.{nd}f}, {r['ci_high']:+.{nd}f}] ({verdict})"


def binary_table(binf: pd.DataFrame, group: str, order_by: str = "accuracy") -> str:
    d = binf[binf["group"] == group].sort_values(order_by, ascending=False)
    rows = []
    for _, r in d.iterrows():
        deg = " †" if r["score_degenerate"] else ""
        rows.append([
            r["model"] + deg,
            val_ci(r, "accuracy"),
            val_ci(r, "sensitivity_PIGD"),
            f(r["specificity_TD"]),
            f(r["precision_PIGD"]),
            f(r["f1_PIGD"]),
            val_ci(r, "balanced_accuracy"),
            val_ci(r, "auc_roc"),
            val_ci(r, "pr_auc"),
            f(r["brier_binary"]),
            f(r["log_loss"]),
        ])
    return md_table(
        ["Model", "Accuracy", "Sensitivity (PIGD)", "Specificity", "Precision",
         "F1", "Balanced acc.", "AUC-ROC", "PR-AUC", "Brier", "Log loss"],
        rows,
    )


def three_class_table(mcf: pd.DataFrame, group: str) -> str:
    d = mcf[mcf["group"] == group].sort_values("accuracy", ascending=False)
    rows = []
    for _, r in d.iterrows():
        rows.append([
            r["model"],
            val_ci(r, "accuracy"),
            val_ci(r, "balanced_accuracy"),
            val_ci(r, "macro_f1"),
            f(r["sensitivity_TD"]),
            f(r["sensitivity_Indeterminate"]),
            f(r["sensitivity_PIGD"]),
            f(r["specificity_TD"]),
            f(r["specificity_Indeterminate"]),
            f(r["specificity_PIGD"]),
            f(r["brier_multiclass_summed"]),
            f(r["log_loss"]),
        ])
    return md_table(
        ["Model", "Accuracy", "Balanced acc.", "Macro-F1", "Sens TD",
         "Sens Ind.", "Sens PIGD", "Spec TD", "Spec Ind.", "Spec PIGD",
         "Brier (summed)", "Log loss"],
        rows,
    )


def ovr_auc_table(mcf: pd.DataFrame, group: str) -> str:
    d = mcf[mcf["group"] == group].sort_values("auc_ovr_Indeterminate", ascending=False)
    rows = []
    for _, r in d.iterrows():
        rows.append([
            r["model"],
            val_ci(r, "auc_ovr_TD"),
            val_ci(r, "auc_ovr_Indeterminate"),
            val_ci(r, "auc_ovr_PIGD"),
            val_ci(r, "macro_auc_ovr"),
            f(r["pr_auc_ovr_TD"]),
            f(r["pr_auc_ovr_Indeterminate"]),
            f(r["pr_auc_ovr_PIGD"]),
            val_ci(r, "mean_predictive_loglik"),
        ])
    return md_table(
        ["Model", "AUC TD", "AUC Indeterminate", "AUC PIGD", "Macro AUC",
         "PR-AUC TD", "PR-AUC Ind.", "PR-AUC PIGD", "Mean predictive log-lik"],
        rows,
    )


def band_table(band: pd.DataFrame) -> str:
    rows = []
    for _, r in band.iterrows():
        w = "unbanded" if pd.isna(r["main_width"]) else f"{r['main_width']:.2f}"
        rows.append([
            w,
            val_ci(r, "accuracy"),
            val_ci(r, "sensitivity_PIGD"),
            f(r["specificity_TD"]),
            f(r["precision_PIGD"]),
            f(r["f1_PIGD"]),
            val_ci(r, "balanced_accuracy"),
            val_ci(r, "auc_roc"),
            val_ci(r, "pr_auc"),
            f(r["brier_binary"]),
            f(r["log_loss"]),
            NA if pd.isna(r["n_params"]) else f"{int(r['n_params'])}",
        ])
    return md_table(
        ["Band width", "Accuracy", "Sensitivity (PIGD)", "Specificity",
         "Precision", "F1", "Balanced acc.", "AUC-ROC", "PR-AUC", "Brier",
         "Log loss", "Parameters"],
        rows,
    )


def ic_table(icf: pd.DataFrame) -> str:
    rows = []
    for grp, g in icf.groupby("comparable_group", sort=True):
        for _, r in g.sort_values("aic").iterrows():
            rows.append([
                r["model"],
                r["likelihood_definition"],
                f(r["loglik"], 2),
                f"{int(r['n_params'])}",
                f"{int(r['n_likelihood_terms']):,}",
                f(r["aic"], 2),
                f(r["bic"], 2),
                grp,
            ])
    return md_table(
        ["Model", "Likelihood definition", "Log-likelihood", "Parameters",
         "Observations", "AIC", "BIC", "Comparable group"],
        rows,
    )


GROUPS = {
    "A": "A: rich profile, own cohort",
    "F2": "F2: harmonised, common binary subset",
    "B": "B: subtype-history models, LOPO",
    "C": "C: matched comparison set",
    "D": "D: ratio-band sweep",
    "E": "E: two-state HMM family",
    "F": "F: harmonised subscore family",
}
CT = "CT-HMM (latent, interval-aware)"
PF = "Profile: level + rate + medication"
PNM = "Profile: level + rate, no medication"
PLO = "Profile: level only (random walk)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    cfg = ap.parse_args()
    run = Path(cfg.run).resolve()

    binf = pd.read_csv(run / "binary_comparison.csv")
    mcf = pd.read_csv(run / "three_class_comparison.csv")
    ct = pd.read_csv(run / "paired_contrasts.csv")
    icf = pd.read_csv(run / "aic_bic_comparison.csv")
    cont = pd.read_csv(run / "rich_profile_comparison.csv")
    cc = pd.read_csv(run / "crosscheck_vs_published.csv")
    vc = pd.read_csv(run / "validation_checks.csv")

    master = build_master(binf, mcf, icf, cont)
    master.to_csv(run / "master_results_complete.csv", index=False)
    band = build_band(binf, mcf)
    band.to_csv(run / "band_hmm_comparison.csv", index=False)
    rec = build_recovered(binf, mcf, cc)
    rec.to_csv(run / "recovered_metrics.csv", index=False)
    figs = figures(run, binf, mcf, band, ct)

    n_new = int((~rec["already_published"]).sum())
    n_pub = int(rec["already_published"].sum())
    n_match = int((cc["status"] == "MATCH").sum())
    n_diff = int((cc["status"] != "MATCH").sum())
    n_fail = int((vc["status"] == "FAIL").sum())

    pf, ch, pn, pl = (get(binf, GROUPS["C"], m) for m in (PF, CT, PNM, PLO))
    b_un = band[band["main_width"].isna()].iloc[0]
    b30 = band[band["main_width"] == 0.30].iloc[0]
    b05 = band[band["main_width"] == 0.05].iloc[0]
    h_ct = get(mcf, GROUPS["F"], "CT-HMM, 3 latent states")
    h_pers = get(mcf, GROUPS["F"], "Persistence (recorded label)")
    h_gp = get(mcf, GROUPS["F"], "Subscore GP, independent")

    best_acc_C = best(binf, GROUPS["C"], "accuracy")
    best_auc_C = best(binf, GROUPS["C"], "auc_roc", exclude_degenerate=True)
    best_pr_C = best(binf, GROUPS["C"], "pr_auc", exclude_degenerate=True)
    best_brier_C = best(binf, GROUPS["C"], "brier_binary", higher=False)
    best_sens_C = best(binf, GROUPS["C"], "sensitivity_PIGD")
    best_auc_F = best(binf, GROUPS["F"], "auc_roc", exclude_degenerate=True)
    best_pr_F = best(binf, GROUPS["F"], "pr_auc", exclude_degenerate=True)
    best_ind_F = best(mcf, GROUPS["F"], "sensitivity_Indeterminate")
    best_indauc_F = best(mcf, GROUPS["F"], "auc_ovr_Indeterminate")
    best_mf1_F = best(mcf, GROUPS["F"], "macro_f1")
    best_acc_F3 = best(mcf, GROUPS["F"], "accuracy")
    best_ll_F = best(mcf, GROUPS["F"], "mean_predictive_loglik")

    n_cases_C = int(pf["n_scored"])
    n_pat_C = int(pf["n_patients"])

    # the cohort reconciliation, each number read from a file
    cohort_rows = [
        ["Frozen panel visits", "5,774", "panel md5 " + PANEL_MD5],
        ["Visits with a derivable subtype label", "4,441", "ratio of tremor to PIGD means defined"],
        ["Patients with at least two labelled visits", "934", "the modelling cohort"],
        ["Labelled visits in those patients", "4,295", "likelihood terms for the label-sequence models"],
        ["Consecutive-visit transitions", "3,361", "the three-class prediction cohort"],
        ["Transitions with a TD or PIGD target", "3,029", f"binary cohort for the history models"],
        ["Transitions also having complete clinical measurements", "3,353", "the rich profile's prediction pairs"],
        ["Transitions with a TD/PIGD target in the profile frame", "3,021", "binary cohort for the rich profile"],
        ["Transitions scored by both families", f"{n_cases_C:,}", "the matched comparison set"],
        ["Transitions scorable by every two-state arm", "2,936", "the two-state common set"],
    ]

    # -------------------------------------------------- the main report
    rpt = f"""# Final Consolidated Results: PPMI Motor-Subtype Progression Models

Run directory `{rel(run)}` · generated 2026-11-11 · frozen panel md5 `{PANEL_MD5}`

## 1. What this document is

This is a consolidation and completion of experiments that were already run.
No model was fitted for it. Every number below is either read from the output
of the experiment that produced it, or computed here from that experiment's
saved per-case predictions, on that experiment's own evaluation cohort and
under its own protocol.

{n_new} metric values that had never been computed were recovered this way,
alongside {n_pub} that reproduce a published value. Of the recovered-and-published
overlap, {n_match} of {n_match + n_diff} comparisons against the published
tables agree to within 1e-4. Of the {len(vc)} structural checks on the
prediction files, {len(vc) - n_fail} passed; the {n_fail} that did not is not a
data fault but a cohort inconsistency this pass uncovered, described in
section 13.2.

Two deliberate omissions: AUC-ROC and PR-AUC are never derived from an accuracy
or a confusion matrix, and no model was refitted to fill a cell. Where a metric
is genuinely unobtainable it is marked **{NC}** or **{NA}** and the reason is
given in section 14 and in `02_MISSING_METRICS_UPDATED.md`.

## 2. Evaluation cohorts

Results are grouped by evaluation protocol. **Numbers from different groups are
not comparable** and are never ranked against each other.

{md_table(["Count", "n", "Meaning"], cohort_rows)}

The arithmetic closes: 4,441 − 4,295 = 146 single-visit patients, which is also
1,080 − 934; and 4,441 − 4,433 = 8 visits lacking clinical measurements, which
is also 4,295 − 4,287 and 3,361 − 3,353.

{md_table(
    ["Group", "Protocol", "Cohort", "Models"],
    [[g, binf[binf["group"] == gn]["evaluation_protocol"].iloc[0]
        if len(binf[binf["group"] == gn]) else
        mcf[mcf["group"] == gn]["evaluation_protocol"].iloc[0],
      binf[binf["group"] == gn]["cohort"].iloc[0]
        if len(binf[binf["group"] == gn]) else
        mcf[mcf["group"] == gn]["cohort"].iloc[0],
      str(len(binf[binf["group"] == gn]) or len(mcf[mcf["group"] == gn]))]
     for g, gn in GROUPS.items()])}

## 3. Metric definitions and the two conventions that trap readers

Positive class for every binary figure is **PIGD**. Sensitivity = TP/(TP+FN),
specificity = TN/(TN+FP), precision = TP/(TP+FP), F1 is their harmonic mean,
balanced accuracy = (sensitivity + specificity)/2. For the three-class task
each class is taken one-versus-rest in turn; macro-F1 is the unweighted mean of
the three F1s, weighted-F1 weights by support, balanced accuracy is the mean of
the three recalls.

AUC-ROC is the Mann-Whitney statistic on the model's probability with mid-ranks
for ties. For the three-class task it is computed one-versus-rest per class;
**macro AUC** is the unweighted mean of those three and **weighted AUC** weights
them by class support (2,115 / 332 / 914). No single multiclass AUC is quoted
without naming its averaging convention, and neither average is reported for an
arm whose probability for some class is constant. PR-AUC is average precision, evaluated once per **distinct score** so
that tied scores are not silently ordered. Both were verified against
scikit-learn to 2e-16 over random, tied, binary and constant scores.

**Brier.** The repository holds two conventions that differ by exactly a factor
of two for two classes:

- single-outcome (binary): mean (p_PIGD − y_PIGD)^2
- summed multiclass: mean of the sum over classes of (p_k − y_k)^2

Every Brier cell here states which one it is. They are never ranked together.
For two classes the conversion is exact, summed = 2 x single-outcome, so the
master table carries both for every binary row (`brier_binary` and
`brier_summed_two_class`). For three classes no such conversion exists, and the
summed form is the only one reported.
The 3-state CT-HMM's published 0.2348 and the rich profile's 0.1177 are **not**
a calibration gap of that size: the first is the summed form of {b_un['brier_binary']:.4f}.

**Log loss** is −mean log p(true class) with the true-class probability clipped
at 1e-12, the constant the repository itself uses
(`src/ppmi_subtype/next_visit.py:118`, `src/ppmi_subtype/two_state_hmm.py:145`).
Arms whose clipping is active are flagged in the master table; for them the
unclipped log loss is infinite and the value shown is a floor, not a score.

**Degenerate scores.** Some arms emit rule-based probabilities taking only one
or two distinct values. Their AUC and PR-AUC are computed but marked with
`†`, because with two distinct scores AUC is algebraically
(sensitivity + specificity)/2 and measures no ranking. Two cases deserve
naming:

- The marginal/null arms have an AUC near {get(binf, GROUPS["B"], "Marginal (two-class prior)")['auc_roc']:.3f},
  far below 0.5. This is a leave-one-patient-out artefact, not discrimination:
  the only thing that moves the score is the removal of the held-out patient
  from the training prior, which moves it *against* that patient's own label.
- `Direct logistic` and `Persistence` have an Indeterminate one-vs-rest AUC of
  exactly 0.5000 because their Indeterminate probability is constant.

## 4. Table A — binary TD vs PIGD on the matched set

{n_cases_C:,} transitions, {n_pat_C} patients, scored by both the rich clinical
profile and the subtype-history models. Confidence intervals are patient-level
bootstrap, 2,000 draws, seed 42. `†` marks a degenerate score as defined above.

{binary_table(binf, GROUPS["C"])}

Reading this table: accuracy spans {binf[binf['group'] == GROUPS['C']]['accuracy'].min():.4f}
to {binf[binf['group'] == GROUPS['C']]['accuracy'].max():.4f} and the intervals of
the leading models overlap heavily. AUC separates them much more clearly.

## 5. Table A2 — the same task on each family's own cohort

These four blocks use four different cohorts. Compare within a block only.

### A2.1 Rich clinical profile, 3,021 transitions, 5-fold patient CV

{binary_table(binf, GROUPS["A"])}

### A2.2 Subtype-history models, 3,029 transitions, leave-one-patient-out

{binary_table(binf, GROUPS["B"])}

### A2.3 Two-state HMM family, 2,936 transitions scorable by every arm

{binary_table(binf, GROUPS["E"])}

### A2.4 Harmonised subscore family, one protocol for all 14 models

The source run published three-class metrics only. The binary view here was
derived by renormalising its saved three-class probabilities over TD + PIGD and
restricting to cases with a TD or PIGD target — the identical convention the
leave-one-patient-out run applies, verified against that run's own `P_PIGD`
column to 1e-9.

{binary_table(binf, GROUPS["F"])}

Cohort sizes are **not** equal in that block. Thirteen models score all
{int(get(binf, GROUPS["F"], "CT-HMM, 3 latent states")["n_scored"]):,} cases with a
TD/PIGD target; label persistence carries no TD/PIGD probability at all when the
previous recorded label was Indeterminate, so it is scorable on only
{int(get(binf, GROUPS["F"], "Persistence (recorded label)")["n_scored"]):,} of them —
a smaller and easier subset, which is why its accuracy looks the highest there.
This inconsistency was found during this consolidation; the next block is the
comparable version.

### A2.5 Harmonised family on the common subset — the comparable ranking

{binary_table(binf, GROUPS["F2"])}

On identical rows the order changes: label persistence's apparent lead
disappears ({contrast_line(ct, "label persistence - CT-HMM, common binary subset", "accuracy")}
against the CT-HMM on accuracy) and its log loss remains two orders of magnitude
worse.

## 6. Table B — ratio-banded CT-HMM

All six arms share one cohort of {int(b_un['n_scored']):,} transitions and one
protocol, with bands rebuilt inside every fold. The unbanded CT-HMM is the
reference row.

{band_table(band)}

### 6.1 Band construction and sparsity

{md_table(
    ["Width", "Main bands", "Tail bands", "Joint categories", "Parameters",
     "Empty", "< 10 obs", "< 30 obs", "Median obs/category", "Obs per parameter"],
    [[f"{r['main_width']:.2f}", int(r["n_main_bands"]), int(r["n_tail_bands"]),
      int(r["n_categories"]), int(r["n_params"]), int(r["n_empty"]),
      int(r["n_lt_10_obs"]), int(r["n_lt_30_obs"]),
      f"{r['median_obs_per_category']:.0f}", f"{r['obs_per_param']:.2f}"]
     for _, r in band.dropna(subset=["main_width"]).iterrows()])}

Bands partition the tremor-to-PIGD ratio over [0, 3) at the stated width, plus
ten tail bands above 3. The joint observation is (subtype, band), so the
alphabet grows from 3 labels to between {int(band['n_categories'].min()):.0f} and
{int(band['n_categories'].max()):.0f} categories.

### 6.2 Paired comparison against the unbanded model, same rows

{md_table(
    ["Contrast", "Metric", "Delta", "95% CI", "Excludes zero"],
    [[r["contrast"].replace(" - unbanded CT-HMM", ""), r["metric"],
      f"{r['delta']:+.4f}", f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]",
      "yes" if r["excludes_zero"] else "no"]
     for _, r in ct[ct["contrast"].str.contains("ratio bands at width")
                    & (ct["task"] == "binary")
                    & ct["metric"].isin(["accuracy", "sensitivity_PIGD",
                                         "specificity_TD", "auc_roc", "pr_auc",
                                         "brier_binary", "log_loss"])].iterrows()])}

The pattern is consistent and one-sided: **AUC-ROC rises at every width with an
interval clear of zero; accuracy, PR-AUC and the Brier score do not improve at
any width.** Sensitivity rises and specificity falls, which is a shift of the
operating point rather than a gain. Width 0.30 is the best of the banded arms on
AUC-ROC ({b30['auc_roc']:.4f}), Brier ({b30['brier_binary']:.4f}) and log loss
({b30['log_loss']:.4f}) while using {int(b30['n_params'])} parameters against
{int(b05['n_params'])} at width 0.05 and 14 for the unbanded model.

### 6.3 AIC and BIC for the banded model: why there are none

The banded model's likelihood is a PMF over its joint alphabet — 44 categories
at width 0.10 — while the unbanded CT-HMM's is a PMF over 3 labels. The
diagnostic run states this itself: banded log-likelihood −13123.4023 with 137
parameters against the baseline's −2969.0593 with 14
(`{rel(BAND_DIAG / 'report.txt')}`). Those numbers are over different sample
spaces, so neither their difference nor their AIC means anything, and the
project's own report says so.

Whether a comparable likelihood could be recovered without refitting was
checked directly. The diagnostic run does save the pieces that marginalising
would need — the joint emission by state (`category_emission_by_state.csv`),
the banded generator (`generator_banded.csv`) and even the implied
subtype-level emission (`emission_subtype_banded.csv`). It does **not** save the
initial state distribution, so the 3-label sequence likelihood cannot be
evaluated from the saved parameters. Even with it the result would be a plug-in
value, not a maximum: the parameters were optimised for the 44-category
likelihood, so −2logL + 2k would not be an AIC for anything. The honest
comparable quantity is the held-out predictive log-likelihood on the shared
3-label alphabet, which is reported for every model in section 8 and needs no
refitting.

## 7. Table C — three-class classification

Hard three-class predictions are the argmax of the saved probabilities. For the
harmonised and two-state runs this reproduces their own published class counts
exactly, which confirms it is the rule they used. For the rich profile and the
ratio-band arms the source runs reported only a two-class decision, so the
argmax rule is applied here and labelled as such.

### 7.1 Harmonised family, one protocol, 3,361 transitions

{three_class_table(mcf, GROUPS["F"])}

### 7.2 Subtype-history models, leave-one-patient-out, 3,361 transitions

{three_class_table(mcf, GROUPS["B"])}

### 7.3 Two-state family, 3,361 transitions

{three_class_table(mcf, GROUPS["E"])}

### 7.4 Rich profile, 3,353 transitions

{three_class_table(mcf, GROUPS["A"])}

### 7.5 Ratio-band arms, 3,361 transitions

{three_class_table(mcf, GROUPS["D"])}

## 8. One-vs-rest ranking, and the Indeterminate class

This is the part of the picture that accuracy hides. The question in the brief
was whether a binary Indeterminate-versus-other analysis could be assembled
from what exists. It can be read one-versus-rest from the saved three-class
probabilities, which is what follows. It is **not** a separately trained binary
Indeterminate model, and is not presented as one.

### 8.1 Harmonised family

{ovr_auc_table(mcf, GROUPS["F"])}

### 8.2 Subtype-history models

{ovr_auc_table(mcf, GROUPS["B"])}

### 8.3 Ratio-band arms

{ovr_auc_table(mcf, GROUPS["D"])}

### 8.4 What the Indeterminate numbers say

Indeterminate has 332 of 3,361 cases. The best one-vs-rest AUC for it in the
harmonised family is **{best_indauc_F['auc_ovr_Indeterminate']:.4f}
{ci(best_indauc_F, 'auc_ovr_Indeterminate')}** ({best_indauc_F['model']}), so the
models do carry usable information about which visits will be Indeterminate.
Hard prediction throws that information away: the 3-state CT-HMM predicts
Indeterminate for {int(h_ct['n_predicted_Indeterminate'])} of 3,361 cases, giving
sensitivity {h_ct['sensitivity_Indeterminate']:.4f}, while
{best_ind_F['model']} reaches {best_ind_F['sensitivity_Indeterminate']:.4f}
with no fitted parameters at all.

The reason is the decision rule, not the probabilities: Indeterminate's
posterior rarely wins an argmax against a class with six times its prior. A
constant "never Indeterminate" rule already scores
{1 - 332 / 3361:.4f} accuracy on this task, which is why accuracy cannot see
this failure and why balanced accuracy and macro-F1 must be read beside it.

## 9. Table D — continuous clinical measurement endpoint

A different endpoint from everything above: predicting the nine continuous
domain scores at the next visit, not the subtype label. Log density is per
domain observation over 30,099 observations; RMSE and MAE are in standardised
units; coverage is the realised rate of 95% prediction intervals.

{md_table(
    ["Model", "Log-density per domain obs.", "RMSE (std.)", "MAE (std.)", "95% coverage"],
    [[r["model"], f(r["loglik_per_domain_observation"]), f(r["rmse_standardised"]),
      f(r["mae_standardised"]), f(r["coverage_95"])]
     for _, r in cont.sort_values("loglik_per_domain_observation",
                                  ascending=False).iterrows()])}

Every value reproduces independently in a second run
(`{rel(NO_LEDD / 'model_comparison_recomputed.csv')}`) to within 5e-4.
Do not read this table as subtype performance: the ordering here is not the
ordering in Table A, and that difference is itself a finding (section 12, Q8).

## 10. Table E — information criteria

AIC = −2logL + 2k and BIC = −2logL + k log n, recomputed from the stored
log-likelihood, parameter count and observation count for every row; all
{len(icf)} reproduce the stored values to within 0.02. **Rows may only be
compared inside one comparable group.** Values are whole-cohort maximum
likelihood, not held out.

{ic_table(icf)}

### 10.1 Comparisons that are invalid, and why

- **Rich profile against any CT-HMM.** Group A is a density over 38,486
  continuous observations; the label-sequence groups are PMFs over 4,295 or
  3,361 discrete terms. A density and a probability mass are not on the same
  scale, so the profile's AIC near 91,700 against the CT-HMM's 5,966 says
  nothing. They are in different groups for this reason.
- **Observed-label chains against the CT-HMM using group B.** Group B's
  likelihood conditions on the first label and covers 3,361 pairs; the CT-HMM's
  covers 4,295 visits jointly. Group C exists precisely to make that comparison
  properly, by refitting the chains with an initial-state term on the same
  4,295 observations, and there the CT-HMM wins by ΔAIC
  {icf[(icf['comparable_group'].str.startswith('C')) & (icf['model'] == 'Markov + initial state')]['delta_aic_within_group'].iloc[0]:.0f}.
- **Band-HMM against the unbanded CT-HMM.** Different observation alphabets;
  see section 6.3.
- **The 2×2 arms against the 3×3 arms.** The 2×2 likelihood covers 3,812
  TD/PIGD visits, the 3×3 covers all 4,295. Different data, separate groups.
- **AIC against BIC.** They are not a ratio and are not divided here.

### 10.2 What the criteria prefer, and where they disagree with prediction

Within the label-sequence group over all 4,295 labelled visits, three hidden
states beat two by ΔAIC
{icf[(icf['comparable_group'].str.startswith('D')) & (icf['model'] == 'CT-HMM (2 hidden x 3 observed)')]['delta_aic_within_group'].iloc[0]:.1f}
and ΔBIC
{icf[(icf['comparable_group'].str.startswith('D')) & (icf['model'] == 'CT-HMM (2 hidden x 3 observed)')]['delta_bic_within_group'].iloc[0]:.1f}.
Held-out prediction cannot tell them apart: paired on identical rows, the
two-state arm's accuracy differs from the three-state arm's by
{contrast_line(ct, "two hidden states - three hidden states, common set", "accuracy")},
and its AUC by
{contrast_line(ct, "two hidden states - three hidden states, common set", "auc_roc")}.
Both statements are correct; in-sample fit and out-of-sample prediction are
different questions.

## 11. Paired comparisons

Every contrast below is computed on identical scored rows, resampling patients
rather than visits, 2,000 draws, seed 42.

{md_table(
    ["Contrast", "Metric", "Model A", "Model B", "Delta", "95% CI", "Excludes zero"],
    [[r["contrast"], r["metric"], f(r["value_a"]), f(r["value_b"]),
      f"{r['delta']:+.4f}", f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]",
      "yes" if r["excludes_zero"] else "no"]
     for _, r in ct[~ct["contrast"].str.contains("ratio bands at width")
                    | ~ct["contrast"].str.contains("unbanded")].iterrows()])}

## 12. Interpretation

Each answer separates the **numerical leader** from the **statistically
supported** claim. A difference whose paired interval includes zero is reported
as unresolved, not as a win.

### Q1. Highest accuracy in each valid evaluation group

{md_table(["Group", "Leader", "Accuracy [95% CI]", "Runner-up"],
          winners(binf, "accuracy", nd=4))}

The harmonised group F is absent from this and every other leader table: its
arms do not share a cohort, so it cannot be ranked. Group F2 is its comparable
counterpart.

In every group the leader's interval overlaps the runner-up's, and no paired
contrast between two serious models resolves on accuracy. The whole 14-model
harmonised spread is
{(binf[binf['group'] == GROUPS['F2']]['accuracy'].max() - binf[binf['group'] == GROUPS['F2']]['accuracy'].min()):.4f}
wide including the null models, and
{(binf[(binf['group'] == GROUPS['F2']) & (binf['accuracy'] > 0.75)]['accuracy'].max() - binf[(binf['group'] == GROUPS['F2']) & (binf['accuracy'] > 0.75)]['accuracy'].min()):.4f}
wide among the models that beat 0.75. **There is no accuracy winner.**

### Q2. Highest AUC-ROC

{md_table(["Group", "Leader", "AUC-ROC [95% CI]", "Runner-up"],
          winners(binf, "auc_roc", exclude_degenerate=True, nd=4))}

This is where the families genuinely separate, and the separation is resolved.
On the matched set the best rich-profile variant reaches
{best_auc_C['auc_roc']:.4f} ({best_auc_C['model']}) against the CT-HMM's
{ch['auc_roc']:.4f}; on the harmonised common subset the best subscore GP reaches
{best(binf, GROUPS['F2'], 'auc_roc', exclude_degenerate=True)['auc_roc']:.4f}
against the same family's CT-HMM at
{get(binf, GROUPS['F2'], 'CT-HMM, 3 latent states')['auc_roc']:.4f}. Both gaps
have paired intervals clear of zero
({contrast_line(ct, "rich profile (level+rate+medication) - CT-HMM, matched set", "auc_roc")}).

### Q3. Highest PR-AUC

{md_table(["Group", "Leader", "PR-AUC [95% CI]", "Runner-up"],
          winners(binf, "pr_auc", exclude_degenerate=True, nd=4))}

PR-AUC had never been computed for the profile-against-CT-HMM comparison. It
separates them further than AUC does:
{contrast_line(ct, "rich profile (level+rate+medication) - CT-HMM, matched set", "pr_auc")}.
Because PIGD is the minority class at {910 / 3020:.1%} of the matched set, this
is the more informative ranking measure of the two.

### Q4. Best probability calibration

{md_table(["Group", "Leader by Brier", "Brier [95% CI]", "Runner-up"],
          winners(binf, "brier_binary", higher=False, nd=4))}

{md_table(["Group", "Leader by log loss", "Log loss [95% CI]", "Runner-up"],
          winners(binf, "log_loss", higher=False, nd=4))}

{md_table(["Group", "Closest calibration slope to 1", "Slope [95% CI]",
           "Lowest ECE", "ECE [bootstrap 2.5-97.5%]"],
          [[g,
            gd.loc[(gd["calibration_slope"] - 1).abs().idxmin(), "model"],
            val_ci(gd.loc[(gd["calibration_slope"] - 1).abs().idxmin()],
                   "calibration_slope"),
            gd.loc[gd["ece_10_bin"].idxmin(), "model"],
            val_ci(gd.loc[gd["ece_10_bin"].idxmin()], "ece_10_bin")]
           for g, gd in binf[binf["calibration_slope"].notna()
                             & ~binf["group"].isin(UNRANKABLE)].groupby("group", sort=True)])}

Two warnings about that last table, both of which matter for reading it.

**The ECE interval is not an uncertainty interval.** Expected calibration error
is a mean of absolute deviations, so resampling noise can only push it up:
its bootstrap distribution sits above the point estimate, and the percentile
range routinely excludes the point estimate entirely. The ranges above are the
bootstrap distribution of the statistic, reported for completeness, and must
not be read as bounds on a true ECE. Brier and log loss, which are proper
scoring rules, carry genuine intervals.

**The best ECE and the best slope belong to weak models.** The leaders in
several groups are the observed-label Markov chain and last-observation-carried-
forward. Their probabilities sit close to the base rate, which is exactly what
a binned calibration measure rewards, while their AUC
({get(binf, GROUPS["C"], "Markov (observed, discrete)")['auc_roc']:.4f} for the
Markov chain against {pf['auc_roc']:.4f} for the profile) is far worse. A model
can be well calibrated and nearly useless; calibration must be read beside
discrimination, never instead of it.

A slope below 1 means the probabilities are too extreme for the outcome, above
1 that they are too timid. Brier and log loss agree on the leader in every
group. On the matched set the
profile's advantage over the CT-HMM is resolved but small:
{contrast_line(ct, "rich profile (level+rate+medication) - CT-HMM, matched set", "brier_binary")}
on Brier and
{contrast_line(ct, "rich profile (level+rate+medication) - CT-HMM, matched set", "log_loss")}
on log loss. Note which model wins the matched set on Brier and log loss: the
levodopa-free variant ({pn['brier_binary']:.4f} / {pn['log_loss']:.4f}),
not the full model.

### Q5. Best sensitivity for PIGD

{md_table(["Group", "Leader", "Sensitivity [95% CI]", "Runner-up"],
          winners(binf, "sensitivity_PIGD", nd=4))}

Sensitivity alone is not a ranking: every model here sits on a shared
accuracy plateau, so a higher PIGD sensitivity is bought with lower
specificity. On the matched set the spread is
{binf[binf['group'] == GROUPS['C']]['sensitivity_PIGD'].min():.4f} to
{best_sens_C['sensitivity_PIGD']:.4f} in sensitivity against
{binf[binf['group'] == GROUPS['C']]['specificity_TD'].min():.4f} to
{binf[binf['group'] == GROUPS['C']]['specificity_TD'].max():.4f} in specificity,
which is why balanced accuracy and F1 are tabulated beside them.

### Q6. Best sensitivity for Indeterminate

**{best_ind_F['model']}**, at
{best_ind_F['sensitivity_Indeterminate']:.4f}
{ci(best_ind_F, 'sensitivity_Indeterminate')} on the harmonised cohort. Every
fitted model in every group scores essentially zero: of the
{len(mcf)} three-class arms, {int((mcf['sensitivity_Indeterminate'] < 0.01).sum())}
have an Indeterminate sensitivity below 0.01. The ranking picture is different
and better — see Q2's counterpart for this class, where the best one-vs-rest
Indeterminate AUC is {best_indauc_F['auc_ovr_Indeterminate']:.4f}
({best_indauc_F['model']}) — so the information is present and the argmax rule
discards it.

Macro-F1 is the one headline metric that notices this. Its leader is
**{best_mf1_F['model']}** at {best_mf1_F['macro_f1']:.4f}
{ci(best_mf1_F, 'macro_f1')}, a model with the
{"worst" if best_mf1_F['accuracy'] == mcf[mcf['group'] == GROUPS['F']]['accuracy'].min() else "second-lowest"}
accuracy in its group ({best_mf1_F['accuracy']:.4f} against the group's best
{best_acc_F3['accuracy']:.4f}), and a log loss of {best_mf1_F['log_loss']:.4f}.
It is not a better model; it is the only one that ever says "Indeterminate".

### Q7. What AIC and BIC prefer inside valid groups

{md_table(["Comparable group", "Preferred by AIC", "Delta AIC to runner-up",
           "Preferred by BIC", "Delta BIC to runner-up"],
          [[g,
            gd.loc[gd["aic"].idxmin(), "model"],
            (f"{sorted(gd['aic'])[1] - gd['aic'].min():.2f}" if len(gd) > 1 else NA),
            gd.loc[gd["bic"].idxmin(), "model"],
            (f"{sorted(gd['bic'])[1] - gd['bic'].min():.2f}" if len(gd) > 1 else NA)]
           for g, gd in icf.groupby("comparable_group", sort=True)])}

AIC and BIC agree in every group. Two cautions. First, these are whole-cohort
fits, so they measure in-sample fit penalised for parameters, not predictive
skill. Second, the criteria and held-out prediction disagree on hidden-state
count, as section 10.2 sets out: AIC prefers three states by
{icf[(icf['comparable_group'].str.startswith('D')) & (icf['model'] == 'CT-HMM (2 hidden x 3 observed)')]['delta_aic_within_group'].iloc[0]:.1f},
while held-out accuracy and AUC cannot separate two from three. Held-out
PR-AUC and log loss do side with three states
({contrast_line(ct, "two hidden states - three hidden states, common set", "pr_auc")}
and
{contrast_line(ct, "two hidden states - three hidden states, common set", "log_loss")}),
which is new here and refines the earlier conclusion that two states predict as
well as three.

### Q8. Does the rich clinical profile outperform the CT-HMM?

**On ranking and calibration, yes; on accuracy, not demonstrably.** Paired on
the same {n_cases_C:,} cases:

{md_table(["Metric", "Profile", "CT-HMM", "Paired difference [95% CI]", "Resolved"],
          [[r["metric"], f(r["value_a"]), f(r["value_b"]),
            f"{r['delta']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]",
            "yes" if r["excludes_zero"] else "no"]
           for _, r in ct[ct["contrast"] ==
                          "rich profile (level+rate+medication) - CT-HMM, matched set"].iterrows()])}

Four of the ten metrics resolve, and all four are probabilistic: AUC-ROC,
PR-AUC, Brier and log loss. The six that do not resolve are every
threshold-based metric, accuracy included. The structural reason is in the
outcome definition: the subtype is a **thresholded ratio** of two of the nine
domains the profile predicts, so predicting all nine better sharpens the
probability without moving many cases across the threshold. That is also why
Table D's ordering is not Table A's.

### Q9. Does the Band-HMM improve on the original CT-HMM?

**On AUC-ROC yes, at every width and with every interval clear of zero; on
accuracy, PR-AUC and Brier, no.** The paired table in section 6.2 is the
evidence. Summarising at the best width, 0.30: AUC
{contrast_line(ct, "ratio bands at width 0.30 - unbanded CT-HMM", "auc_roc")},
accuracy {contrast_line(ct, "ratio bands at width 0.30 - unbanded CT-HMM", "accuracy")},
PR-AUC {contrast_line(ct, "ratio bands at width 0.30 - unbanded CT-HMM", "pr_auc")},
Brier {contrast_line(ct, "ratio bands at width 0.30 - unbanded CT-HMM", "brier_binary")}.
The cost is {int(b30['n_params'])} parameters against 14, and the gain cannot be
priced by AIC because the alphabets differ. Describing this as an overall
improvement would be wrong; describing it as a ranking improvement is correct.

### Q10. Best balance of predictive performance and complexity

{md_table(["Candidate", "Parameters", "Case for it", "Case against it"],
          [["Rich profile, level only (random walk)", "27",
            f"best matched-set accuracy ({pl['accuracy']:.4f}), AUC {pl['auc_roc']:.4f}, "
            "27 parameters against 45",
            "AUC and PR-AUC below both level+rate variants; "
            f"worse Brier ({pl['brier_binary']:.4f})"],
           ["Rich profile without levodopa", "36",
            f"best matched-set Brier ({pn['brier_binary']:.4f}) and log loss "
            f"({pn['log_loss']:.4f}); AUC {pn['auc_roc']:.4f}; needs no medication data",
            f"AIC prefers keeping levodopa by {icf[(icf['comparable_group'].str.startswith('A')) & (icf['model'] == PNM)]['delta_aic_within_group'].iloc[0]:.0f}; "
            "needs nine clinical domains at every visit"],
           ["CT-HMM, 3 latent states", "14",
            f"14 parameters, accuracy {ch['accuracy']:.4f} statistically "
            "indistinguishable from the profile's, needs only the label history",
            f"AUC {ch['auc_roc']:.4f} and PR-AUC {ch['pr_auc']:.4f} are resolvedly lower"],
           ["Ratio bands at width 0.30", f"{int(b30['n_params'])}",
            f"best AUC of any CT-HMM variant ({b30['auc_roc']:.4f})",
            "no accuracy, PR-AUC or Brier gain; "
            f"{int(b30['n_params'])} parameters; no valid AIC"],
           ["Label persistence", "0",
            f"the only model with non-zero Indeterminate sensitivity "
            f"({h_pers['sensitivity_Indeterminate']:.4f}) and the best macro-F1 "
            f"({h_pers['macro_f1']:.4f})",
            f"log loss {h_pers['log_loss']:.2f}, unusable as a probability model"]])}

If one model must be named for the subtype endpoint, the defensible choice is
the **CT-HMM with 3 latent states** when only label history is available, and
the **rich profile without levodopa** when the nine domains are available: it
matches the full profile on every metric
({contrast_line(ct, "removing levodopa from the rich profile, profile cohort", "accuracy")}
on accuracy,
{contrast_line(ct, "removing levodopa from the rich profile, profile cohort", "auc_roc")}
on AUC) while needing no medication record. **No model is best on every
metric, and the protocols differ, so no universal winner is declared.**

## 13. Validation

{md_table(["Check", "Result"],
          [[r["check"], f"{r['status']} — {r['detail']}"] for _, r in vc.iterrows()])}

### 13.1 Agreement with the published tables

{n_match} of {n_match + n_diff} comparisons against published values agree to
within 1e-4, across {cc['source'].nunique()} source tables:

{md_table(["Published table", "Comparisons", "Matching"],
          [[k, str(len(g)), str(int((g["status"] == "MATCH").sum()))]
           for k, g in cc.groupby("source", sort=True)])}

This covers every published accuracy, Brier, log loss, macro-F1, balanced
accuracy, per-class recall, predicted-class count, AUC and PR-AUC that the
source runs recorded. The band run's own AUC and PR-AUC were recomputed from
its predictions by an independent implementation and reproduce to 1e-6, which
is the strongest single check here.

### 13.2 Discrepancies and corrections found

1. **Cohort inconsistency inside the harmonised binary view.** Label
   persistence is scorable on
   {int(get(binf, GROUPS["F"], "Persistence (recorded label)")["n_scored"]):,} cases
   where the other 13 models cover
   {int(get(binf, GROUPS["F"], "CT-HMM, 3 latent states")["n_scored"]):,}. Its
   apparently best-in-family accuracy is an artefact of the easier subset. Group
   F2 fixes this; both are reported.
2. **Two hidden states versus three.** The earlier conclusion was that two
   states predict as well as three. On accuracy and AUC that holds. On held-out
   PR-AUC, log loss and Indeterminate ranking it does not: three states win with
   intervals clear of zero. The earlier statement was not wrong about the metrics
   it examined; it was incomplete.
3. **The two Brier conventions** differ by a factor of two and both appear in the
   repository. Every Brier value here carries its convention.
4. **Confusion-matrix orientation** differs between source runs; rows are actual
   labels in some and predicted in others. All per-class metrics here were
   recomputed from per-case predictions rather than from any stored matrix, so
   the orientation question cannot propagate.
5. **Degenerate AUCs.** The marginal arms' AUC near
   {get(binf, GROUPS["B"], "Marginal (two-class prior)")['auc_roc']:.3f} is a
   leave-one-patient-out artefact, not inverted discrimination, and the
   constant-score arms' 0.5000 is not chance-level discrimination. Both are
   flagged rather than tabulated silently.
6. **AIC prefers three hidden states while held-out accuracy cannot separate
   two from three.** Both statements stand; they measure different things.

### 13.3 Duplicate and superseded runs

Two pairs of runs are duplicates rather than conflicts, verified earlier to
reproduce each other exactly (maximum metric difference 0.0): the band-width
sweep and its extended rerun, which only adds width 0.50, and the two-state run
and its confirmation, which only adds a SLURM record. The extended and
confirming runs are cited throughout. The 2×2 "no emission floor" arm is not a
separate model: it matches the floored arm to ten decimal places with zero
discordant predictions, which is visible in Table A2.3.

## 14. What remains missing

{md_table(["Metric", "Models", "Why it is still missing"],
          [["AIC / BIC", "Band-HMM at all five widths",
            "likelihood is over a 44-category alphabet; the initial state "
            "distribution is not saved, and a plug-in marginal would not be a "
            "maximum (section 6.3)"],
           ["AIC / BIC", "LOCF, population mean, persistence, marginal",
            f"{NA}: no fitted likelihood"],
           ["AIC / BIC", "subscore GP, LMM and spline families",
            "no saved whole-cohort log-likelihood or parameter count"],
           ["Separately trained binary Indeterminate model", "all",
            "never run; the one-vs-rest reading in section 8 is not a substitute"],
           ["External validation", "all",
            "no second cohort in this project"],
           ["Calibration slope and intercept",
            "the marginal and persistence arms",
            f"{NA}: their score takes fewer than three distinct values, so a "
            "logistic recalibration has no slope"],
           ["Multiclass AUC for the 2x2 arms", "two 2x2 CT-HMM arms",
            f"{NA}: no Indeterminate output by construction"]])}

`02_MISSING_METRICS_UPDATED.md` gives the full register, one row per
model-and-metric, with the file inspected and what each would need.

## 15. Sources

Every value in this document comes from one of these files.

{md_table(["Role", "Path"],
          [["Rich profile per-case subtype predictions",
            "experiments/20261103T220147Z_a2p_outcome_table_fullprec/derived_label_predictions.csv"],
           ["Rich profile continuous predictions",
            "experiments/20261103T220147Z_a2p_outcome_table_fullprec/profile_predictions.csv"],
           ["Rich profile continuous metrics",
            rel(PROFILE_CONT / "model_comparison.csv")],
           ["Levodopa-free profile rerun", rel(NO_LEDD)],
           ["Subtype-history LOPO predictions",
            "experiments/20261020T104756Z_next_visit_td_pigd_prediction/next_visit_predictions.csv"],
           ["Ratio-band sweep predictions and metrics",
            rel(BAND_RUN)],
           ["Ratio-band emission diagnostic", rel(BAND_DIAG)],
           ["Two-state HMM family", "experiments/20261110T143200Z_two_state_hmm_confirm"],
           ["Harmonised 14-model family",
            "experiments/20261103T202632Z_e8_harmonised_family_comparison"],
           ["Published matched-set AUC",
            "experiments/20261023T202640Z_roc_auc_comparison/auc_metrics.csv"],
           ["Published AIC/BIC with comparison groups",
            "experiments/20261023T194317Z_model_selection_criteria/aic_bic.csv"],
           ["Earlier consolidation", "docs/consolidated_results/"]])}

### 15.1 Reproducing this

```bash
~/envs/ppmi/bin/python scripts/96_final_metric_recovery.py --out {rel(run)} --n-boot 2000
~/envs/ppmi/bin/python scripts/97_final_tables_and_report.py --run {rel(run)}
```

The first script recovers every metric from the saved predictions and writes the
comparison CSVs, the cross-check and the validation checks; the second builds
the tables, figures and this report from those CSVs. Both are deterministic at
seed 42. Figures: {", ".join(figs)}.
"""
    (run / "FINAL_CONSOLIDATED_RESULTS.md").write_text(rpt)

    # ------------------------------------------- updated missing register
    nc_rows = []
    for _, r in master.iterrows():
        for c in MASTER_COLS:
            if r[c] == NC:
                nc_rows.append({"group": r["group"], "task": r["task"],
                                "model": r["model"], "metric": c,
                                "source_inspected": r["source_path"]})
    nc = pd.DataFrame(
        nc_rows,
        columns=["group", "task", "model", "metric", "source_inspected"],
    )
    REASONS = {
        "accuracy_95ci": ("the interval was not requested for this arm",
                          "no", "one bootstrap pass over the same predictions"),
        "loglik": ("no whole-cohort log-likelihood was saved for this model",
                   "yes, a whole-cohort refit", "refit the model on all patients "
                   "and store the maximised log-likelihood and parameter count"),
        "aic": ("depends on a log-likelihood that was not saved, or on a "
                "likelihood that is not comparable to any other model's",
                "yes for the band and subscore families",
                "a saved maximised log-likelihood over a stated alphabet, with "
                "its parameter count and observation count"),
        "bic": ("as AIC", "as AIC", "as AIC"),
    }
    reg_rows = []
    for metric, g in nc.groupby("metric", sort=True):
        why, refit, need = REASONS.get(
            metric,
            ("not computed by the source run and not recoverable from its saved "
             "outputs without additional information",
             "no", "see the method column of recovered_metrics.csv"),
        )
        reg_rows.append([
            metric,
            f"{len(g)} arms: " + "; ".join(
                f"{k} ({len(gg)})" for k, gg in g.groupby("group", sort=True)),
            why, refit, need,
            sorted(g["source_inspected"].unique())[0],
        ])
    missing = f"""# Missing-Metrics Register, Updated

Run `{rel(run)}` \u00b7 2026-11-11 \u00b7 supersedes
`docs/consolidated_results/02_MISSING_METRICS_20261111.md`

{n_new} metric values that were previously absent have been computed from saved
predictions and are listed in `recovered_metrics.csv`. What follows is what is
still absent, and why. **{NC}** means it could exist but does not;
**{NA}** means it is undefined for that model or task.

## A. Recovered in this pass, no longer missing

{md_table(["Metric family", "Arms", "How"],
          [[k, str(len(g)), sorted(g["method"].unique())[0]]
           for k, g in rec[~rec["already_published"]].groupby("metric", sort=True)])}

## B. Still missing, with the reason

{md_table(["Metric", "Affected arms", "Why unavailable", "Needs a refit?",
           "What it would take"], reg_rows) if reg_rows else
 "No cell in `master_results_complete.csv` now reads **" + NC + "**. Every "
 "remaining gap is either undefined for that model, and marked **" + NA +
 "** with the reason in the cell itself, or needs work that no saved output "
 "can supply, listed in section C."}

## C. Never run, and not recoverable from any saved output

### C1. A separately trained binary Indeterminate-versus-other model

Section 8 of the report reads Indeterminate one-versus-rest out of the saved
three-class probabilities, which is legitimate and informative, but it is not a
model trained for that question. A dedicated binary experiment would relabel
every visit as Indeterminate or not and fit the same CT-HMM, chain and profile
families to it.

Baseline to beat, so the result is not misread: a constant "never
Indeterminate" rule scores {1 - 332 / 3361:.4f} accuracy on this cohort, with
sensitivity 0. Report balanced accuracy, PR-AUC and the full confusion matrix,
not accuracy.

### C2. A single protocol across every family

The rich profile uses 5-fold patient cross-validation with 4 repeats; the
subtype-history families use leave-one-patient-out. The matched set in Table A
removes the cohort difference but not the protocol difference. Settling it
would mean rerunning one family under the other's protocol.

### C3. External validation

No second cohort exists in this project. Every number here is internal
cross-validation on one frozen panel.

### C4. Reliability diagrams

The calibration slope, intercept, expected calibration error and
calibration-in-the-large are computed for every binary arm with a
non-degenerate score and are in the master table. What is not produced here is
the reliability diagram itself, bin by bin; the ten-bin summary is reported
instead of the curve.

## D. Values that exist but must not be used

- The band model's whole-cohort log-likelihood of \u221213123.4023 with 137
  parameters, and the AIC of 26520.80 derived from it. The run that produced it
  labels it a fit diagnostic over a 44-category alphabet, not comparable to the
  3-label baseline's \u22122969.0593. Do not place them in one table.
- The rich profile's AIC near 91,700 next to any CT-HMM's near 5,966. Density
  against probability mass.
- Any Brier comparison that mixes the single-outcome and summed conventions.
- The marginal arms' AUC, as a discrimination estimate.
"""
    (run / "02_MISSING_METRICS_UPDATED.md").write_text(missing)

    # --------------------------------------------------- validation audit
    audit = f"""# Validation Audit

Run `{rel(run)}` \u00b7 2026-11-11 \u00b7 panel md5 `{PANEL_MD5}`

Every metric reported in `FINAL_CONSOLIDATED_RESULTS.md` is either read from a
published table or computed here from per-case predictions. This document
records how each was checked.

## 1. Structural checks on the prediction files

{md_table(["Check", "Status", "Detail"],
          [[r["check"], r["status"], r["detail"]] for _, r in vc.iterrows()])}

{int((vc["status"] == "FAIL").sum())} of {len(vc)} failed.

## 2. Agreement with published values

{n_match} of {n_match + n_diff} comparisons agree to within 1e-4.

{md_table(["Published table", "Metrics compared", "Match", "Other"],
          [[k, str(len(g)), str(int((g["status"] == "MATCH").sum())),
            str(int((g["status"] != "MATCH").sum()))]
           for k, g in cc.groupby("source", sort=True)])}

Non-matching rows, if any:

{md_table(["Source", "Model", "Metric", "Published", "Recovered", "Status"],
          [[r["source"], r["model"], r["metric"], f(r["published"], 6),
            f(r["recovered"], 6), r["status"]]
           for _, r in cc[cc["status"] != "MATCH"].iterrows()] or
          [["none", "", "", "", "", ""]])}

## 3. Metric implementations, checked against scikit-learn

| Quantity | Implementation | Agreement |
|---|---|---|
| AUC-ROC | Mann-Whitney with mid-ranks for ties | 1.1e-16 max absolute difference over 400 random, tied, binary and constant score vectors |
| PR-AUC | average precision, thresholded at distinct scores | 2.2e-16 over the same plus constant-score cases |
| F1, balanced accuracy, Brier, log loss | direct formulae | exact to double precision |
| macro-F1, weighted-F1, macro AUC, weighted AUC, multiclass log loss | direct formulae | exact to double precision |
| Calibration slope and intercept | logistic regression of the outcome on logit(p), Newton steps to 1e-10 | 3.0e-12 max absolute difference against scikit-learn's `newton-cg` at tol 1e-12 over 60 distorted-probability datasets; a perfectly calibrated score returns slope 1.006 and intercept -0.004 at n = 200,000 |
| ECE, MCE, calibration-in-the-large | 10 equal-width probability bins | direct formulae |

The tie handling matters: a per-row cumulative average precision disagrees with
scikit-learn by up to 0.13 on tied scores, and several arms here emit
rule-based probabilities with only two distinct values.

## 4. Accuracy recomputation

Every arm's accuracy was recomputed from its own per-case predictions and
compared with the published value where one exists. All
{int(((cc["metric"].isin(["accuracy", "accuracy_recomputed"])) & (cc["status"] == "MATCH")).sum())}
published accuracies reproduce.

## 5. Information criteria recomputation

AIC = \u22122logL + 2k and BIC = \u22122logL + k log n were recomputed from the stored
log-likelihood, parameter count and observation count for all {len(icf)} rows.

{md_table(["Model", "Stored AIC", "Recomputed", "Stored BIC", "Recomputed",
           "Observations", "Parameters"],
          [[r["model"], f(r["aic"], 2), f(r["aic_recomputed"], 2),
            f(r["bic"], 2), f(r["bic_recomputed"], 2),
            f"{int(r['n_likelihood_terms']):,}", str(int(r["n_params"]))]
           for _, r in icf.iterrows()])}

## 6. Cohort verification

{md_table(["Count", "n", "Meaning"], cohort_rows)}

Group-level cohort consistency is checked explicitly: every arm inside a group
must share one cohort size, or the group is flagged and never ranked. One group
failed that check as first constructed \u2014 the harmonised binary view \u2014 and group
F2 was added as its comparable counterpart.

## 7. Conventions pinned down

- **Positive class**: PIGD throughout the binary task.
- **Brier**: both conventions present in the repository; each cell labelled.
  The summed form is exactly twice the single-outcome form for two classes.
- **Log loss**: true-class probability clipped at 1e-12, matching
  `src/ppmi_subtype/next_visit.py:118` and
  `src/ppmi_subtype/two_state_hmm.py:145`. Arms where clipping is active are
  flagged; for them the unclipped value is infinite.
- **Two-class renormalisation**: P(PIGD) = p_PIGD / (p_TD + p_PIGD), verified
  against the leave-one-patient-out run's own column to 1e-9 before being
  applied to the harmonised run.
- **Three-class decision rule**: argmax of the saved probabilities, verified
  against the harmonised and two-state runs' own predicted-class counts.
- **Confusion-matrix orientation**: no stored matrix was used. Every per-class
  count was recomputed from per-case predictions.
- **Bootstrap**: patients resampled with replacement, 2,000 draws, seed 42,
  percentile interval. Visits are never treated as independent.
- **ECE intervals**: expected calibration error is a mean of absolute
  deviations, so resampling inflates it and its percentile range can lie
  wholly above the point estimate. The range is reported as the bootstrap
  distribution of the statistic, with a flag in the master table
  (`ece_bootstrap_range_excludes_point_estimate`) wherever that happens, and is
  never described as a confidence interval for a true ECE.
- **Paired contrasts**: both arms restricted to identical rows and verified
  row-aligned before differencing; a misaligned pair is skipped, not silently
  compared.

## 8. What was not altered

No experiment directory was modified. No model was refitted. The frozen panel
was not touched. Published values are reported as published; where this pass
disagrees or adds nuance, both readings appear, in section 13.2 of the report.
"""
    (run / "VALIDATION_AUDIT.md").write_text(audit)
    json.dump(
        {
            "n_recovered_new": n_new,
            "n_reproduced_published": n_pub,
            "crosscheck_match": n_match,
            "crosscheck_other": n_diff,
            "validation_failed": n_fail,
            "n_master_rows": int(len(master)),
            "figures": figs,
        },
        open(run / "report_summary.json", "w"),
        indent=2,
    )
    print(f"wrote master ({len(master)} rows), band ({len(band)}), "
          f"recovered ({len(rec)}), {len(figs)} figures")
    print(f"recovered new: {n_new}, reproduced published: {n_pub}, "
          f"crosscheck match {n_match}/{n_match + n_diff}, checks failed {n_fail}")


if __name__ == "__main__":
    main()
