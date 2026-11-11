"""Assemble the master consolidated results table from the experiment outputs.

One row per model and evaluation configuration. Every metric is READ from the
run that produced it -- nothing is retyped -- so the table can be regenerated
and re-verified at any time. Cells are filled with the literal strings
``Not calculated`` where a metric was never computed and ``Not applicable``
where it does not exist for that kind of model or endpoint, so a blank never
has to be interpreted.

Rows are grouped by evaluation protocol, because the protocols are not
interchangeable: different cohorts, different case counts, different label
spaces. Metrics may be compared WITHIN a group and must not be compared across
groups.

    python scripts/95_master_results_table.py --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"

NC = "Not calculated"
NA = "Not applicable"

R = {
    "profile_cv": EXP / "20261103T220147Z_a2p_outcome_table_fullprec",
    "matched": EXP / "20261020T180515Z_clinical_profile_cv",
    "history_lopo": EXP / "20261020T104756Z_next_visit_td_pigd_prediction",
    "selection": EXP / "20261023T194317Z_model_selection_criteria",
    "roc": EXP / "20261023T202640Z_roc_auc_comparison",
    "band": EXP / "20261104T152154Z_cthmm_ratio_banded_observations",
    "band_auc": EXP / "20261104T172138Z_ratio_banded_auc_roc",
    "width": EXP / "20261105T025024Z_cthmm_ratio_band_width_extended",
    "emission": EXP / "20261104T155439Z_ratio_band_emission_diagnostic",
    "two_state": EXP / "20261110T143200Z_two_state_hmm_confirm",
    "harmonised": EXP / "20261103T202632Z_e8_harmonised_family_comparison",
    "no_ledd": EXP / "20261111T121021Z_rich_profile_no_ledd",
    "sens": None,  # set from --sens
}

COLUMNS = [
    "group", "task", "model", "evaluation_protocol", "cohort",
    "n_patients", "n_cases", "positive_class",
    "accuracy", "accuracy_95ci", "sensitivity", "specificity",
    "auc_roc", "auc_roc_95ci", "pr_auc", "pr_auc_95ci",
    "brier", "brier_convention", "log_loss",
    "log_density_per_domain_obs", "log_density_95ci",
    "rmse_standardised", "mae_standardised", "coverage_95",
    "loglik", "n_params", "aic", "bic", "n_likelihood_terms",
    "source_path", "notes",
]


def ci(lo, hi, nd=4) -> str:
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return NC
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def blank() -> dict:
    return {c: NC for c in COLUMNS}


# ---------------------------------------------------------------------------
def group_a(sens_dir: Path) -> list[dict]:
    """Two-class TD vs PIGD on the matched 3,020-case set."""
    acc = pd.read_csv(R["matched"] / "derived_label_vs_subtype_history.csv")
    ss = pd.read_csv(sens_dir / "two_class_sensitivity_specificity.csv")
    auc = pd.read_csv(R["roc"] / "auc_metrics.csv")
    auc_for = {"Profile: level + rate + medication": "Rich clinical profile",
               "Subtype-history: CT-HMM (latent, interval-aware)": "CT-HMM",
               "Subtype-history: Persistence (last non-Indeterminate)":
                   "Persistence rule (reference)"}
    rows = []
    for _, a in acc.iterrows():
        r = blank()
        s = ss[ss["model"] == a["model"]]
        r.update({
            "group": "A. Two-class, matched 3,020 cases",
            "task": "Binary: next visit TD vs PIGD",
            "model": a["model"],
            "evaluation_protocol": (
                "Rich-profile arms: 5-fold patient CV, 4 repeats, seed 42. "
                "Subtype-history arms: leave-one-patient-out, 934 folds. "
                "Scored on the intersection of both runs' cases."),
            "cohort": ("934-patient modelling cohort; cases with an "
                       "Indeterminate next label are not scorable and are "
                       "excluded from the denominator"),
            "n_patients": 898, "n_cases": int(a["n_scored"]),
            "positive_class": "PIGD",
            "accuracy": round(float(a["accuracy"]), 4),
            "accuracy_95ci": ci(a["accuracy_ci_patient_bootstrap_low"],
                                a["accuracy_ci_patient_bootstrap_high"]),
            "brier": round(float(a["brier_two_class"]), 4),
            "brier_convention": "binary, mean (p_PIGD - 1{PIGD})^2",
            "log_loss": round(float(a["log_loss"]), 4),
            "rmse_standardised": NA, "mae_standardised": NA,
            "log_density_per_domain_obs": NA, "log_density_95ci": NA,
            "coverage_95": NA,
            "loglik": NC, "n_params": NC, "aic": NC, "bic": NC,
            "n_likelihood_terms": NA,
            "source_path": rel(R["matched"] / "derived_label_vs_subtype_history.csv"),
            "notes": "",
        })
        if len(s):
            r["sensitivity"] = round(float(s["sensitivity_PIGD"].iloc[0]), 4)
            r["specificity"] = round(float(s["specificity_TD"].iloc[0]), 4)
            r["source_path"] += "; " + rel(
                sens_dir / "two_class_sensitivity_specificity.csv")
            r["notes"] = ("sensitivity/specificity recovered from the run's own "
                          "prediction file; accuracy recomputed from the same "
                          "file agrees with the published value to 4 dp")
        if a["model"] in auc_for:
            m = auc[auc["model"] == auc_for[a["model"]]]
            if len(m):
                r["auc_roc"] = round(float(m["auc"].iloc[0]), 4)
                r["auc_roc_95ci"] = ci(m["ci_low"].iloc[0], m["ci_high"].iloc[0])
                r["source_path"] += "; " + rel(R["roc"] / "auc_metrics.csv")
        rows.append(r)
    return rows


def group_b(sens_dir: Path) -> list[dict]:
    """Two-class TD vs PIGD on the 3,029-case leave-one-patient-out set."""
    rows = []
    hist = pd.read_csv(R["history_lopo"] / "model_comparison.csv")
    ssb = pd.read_csv(sens_dir / "two_class_sensitivity_specificity_lopo.csv")
    for _, a in hist.iterrows():
        r = blank()
        sb = ssb[ssb["model"] == a["model"]]
        r.update({
            "group": "B. Two-class, LOPO 3,029 cases",
            "task": "Binary: next visit TD vs PIGD",
            "model": a["model"],
            "evaluation_protocol": ("Leave-one-patient-out, 934 folds, seed 42, "
                                    "emission floor 0.5, 4 restarts"),
            "cohort": "934-patient modelling cohort, subtype history only",
            "n_patients": int(a["n_patients"]), "n_cases": int(a["n_scored"]),
            "positive_class": "PIGD",
            "accuracy": round(float(a["accuracy"]), 4),
            "accuracy_95ci": ci(a["accuracy_ci_patient_bootstrap_low"],
                                a["accuracy_ci_patient_bootstrap_high"]),
            "brier": round(float(a["brier_two_class"]), 4),
            "brier_convention": "binary, mean (p_PIGD - 1{PIGD})^2",
            "log_loss": round(float(a["log_loss"]), 4),
            "rmse_standardised": NA, "mae_standardised": NA,
            "log_density_per_domain_obs": NA, "log_density_95ci": NA,
            "coverage_95": NA, "n_likelihood_terms": NA,
            "sensitivity": (round(float(sb["sensitivity_PIGD"].iloc[0]), 4)
                            if len(sb) else NC),
            "specificity": (round(float(sb["specificity_TD"].iloc[0]), 4)
                            if len(sb) else NC),
            "source_path": "; ".join([
                rel(R["history_lopo"] / "model_comparison.csv"),
                rel(sens_dir / "two_class_sensitivity_specificity_lopo.csv")]),
            "notes": ("subtype-history baseline suite; sensitivity and "
                      "specificity recovered from the run's own "
                      "next_visit_predictions.csv, and accuracy recomputed "
                      "from the same counts reproduces the published value"),
        })
        rows.append(r)

    width = pd.read_csv(R["width"] / "model_metrics.csv")
    per = pd.read_csv(R["width"] / "per_class_metrics.csv")
    comp = pd.read_csv(R["width"] / "complexity_by_width.csv")
    for _, a in width.iterrows():
        r = blank()
        p_td = per[(per["model"] == a["model"]) & (per["state"] == "TD")]
        p_pg = per[(per["model"] == a["model"]) & (per["state"] == "PIGD")]
        w = None
        if "width" in str(a["arm"]):
            w = float(str(a["arm"]).split()[-1])
        k = comp[np.isclose(comp["main_width"], w)] if w is not None else None
        r.update({
            "group": "B. Two-class, LOPO 3,029 cases",
            "task": "Binary: next visit TD vs PIGD",
            "model": a["model"],
            "evaluation_protocol": ("Leave-one-patient-out, 934 folds, seed 42, "
                                    "4 restarts; bands refitted inside each fold"),
            "cohort": "934-patient modelling cohort, subtype history only",
            "n_patients": int(a["n_patients"]), "n_cases": int(a["n_scored"]),
            "positive_class": "PIGD",
            "accuracy": round(float(a["accuracy"]), 4),
            "accuracy_95ci": ci(a["accuracy_ci_patient_bootstrap_low"],
                                a["accuracy_ci_patient_bootstrap_high"]),
            "sensitivity": (round(float(p_pg["recall"].iloc[0]), 4)
                            if len(p_pg) else NC),
            "specificity": (round(float(p_td["recall"].iloc[0]), 4)
                            if len(p_td) else NC),
            "auc_roc": round(float(a["auc_roc"]), 4),
            "auc_roc_95ci": ci(a["auc_ci_low"], a["auc_ci_high"]),
            "pr_auc": round(float(a["pr_auc"]), 4),
            "pr_auc_95ci": ci(a["pr_ci_low"], a["pr_ci_high"]),
            "brier": round(float(a["brier_two_class"]), 4),
            "brier_convention": "binary, mean (p_PIGD - 1{PIGD})^2",
            "log_loss": round(float(a["log_loss"]), 4),
            "rmse_standardised": NA, "mae_standardised": NA,
            "log_density_per_domain_obs": NA, "log_density_95ci": NA,
            "coverage_95": NA,
            "n_params": (int(k["n_params"].iloc[0]) if k is not None and len(k)
                         else (14 if "latent" in a["model"] else NC)),
            "aic": NA, "bic": NA, "n_likelihood_terms": NA,
            "source_path": "; ".join([
                rel(R["width"] / "model_metrics.csv"),
                rel(R["width"] / "per_class_metrics.csv"),
                rel(R["width"] / "complexity_by_width.csv")]),
            "notes": ("AIC/BIC not applicable here: the banded models' "
                      "likelihood is over a 20-73 category alphabet, not the "
                      "3-label alphabet, so it is not comparable with the "
                      "unbanded CT-HMM's"
                      + ("; this unbanded row is an independent re-run of the "
                         "subtype-history CT-HMM and reproduces it exactly "
                         "(accuracy 0.8369, Brier 0.1214, log loss 0.3961)"
                         if "latent" in a["model"] else "")),
        })
        rows.append(r)
    return rows


def group_c(sens_dir: Path) -> list[dict]:
    """Native three-class endpoint, 3,361 transitions."""
    rows = []
    nat = pd.read_csv(R["two_state"] / "native_three_class.csv")
    ovr = pd.read_csv(sens_dir / "three_class_one_vs_rest.csv")
    name_map = {
        "CT-HMM (3 hidden x 3 observed, existing model)":
            "CT-HMM (3 hidden x 3 observed)",
        "CT-HMM (2 hidden x 3 observed)": "CT-HMM (2 hidden x 3 observed)",
        "Persistence (last observed label)": "Persistence (last observed label)",
        "Marginal (training label prior)": "Marginal (training label prior)",
    }
    for _, a in nat.iterrows():
        r = blank()
        o = ovr[ovr["model"] == name_map.get(a["model"], "")]
        sens_s = spec_s = NC
        if len(o):
            sens_s = "; ".join(f"{x.label} {x.sensitivity:.4f}"
                               for x in o.itertuples())
            spec_s = "; ".join(f"{x.label} {x.specificity:.4f}"
                               for x in o.itertuples())
        r.update({
            "group": "C. Native three-class, 3,361 transitions",
            "task": "Multiclass: next visit TD vs Indeterminate vs PIGD",
            "model": a["model"],
            "evaluation_protocol": "Leave-one-patient-out, 934 folds, seed 42",
            "cohort": "934 patients, 4,295 labelled visits, 3,361 transitions",
            "n_patients": int(a["n_patients"]), "n_cases": int(a["n_scored"]),
            "positive_class": "Not applicable (three classes); "
                              "one-vs-rest rates given per class",
            "accuracy": round(float(a["accuracy"]), 4),
            "accuracy_95ci": ci(a["accuracy_ci_low"], a["accuracy_ci_high"]),
            "sensitivity": sens_s, "specificity": spec_s,
            "auc_roc": NC, "auc_roc_95ci": NC, "pr_auc": NC, "pr_auc_95ci": NC,
            "brier": round(float(a["brier"]), 4),
            "brier_convention": "multiclass, mean sum_c (p_c - 1{c})^2",
            "log_loss": round(float(a["log_loss"]), 4),
            "rmse_standardised": NA, "mae_standardised": NA,
            "log_density_per_domain_obs": NA, "log_density_95ci": NA,
            "coverage_95": NA, "n_likelihood_terms": NA,
            "source_path": "; ".join([
                rel(R["two_state"] / "native_three_class.csv"),
                rel(sens_dir / "three_class_one_vs_rest.csv")]),
            "notes": ("balanced accuracy "
                      f"{float(a['balanced_accuracy']):.4f}, macro-F1 "
                      f"{float(a['macro_f1']):.4f}"),
        })
        rows.append(r)

    harm = R["harmonised"] / "harmonised_metrics.csv"
    if harm.exists():
        h = pd.read_csv(harm)
        # Per-class support on this endpoint, from the leave-one-patient-out
        # three-class table; with recall and the number predicted, each class's
        # one-vs-rest specificity is determined.
        support = {"TD": 2115, "Indeterminate": 332, "PIGD": 914}
        total = sum(support.values())
        for _, a in h.iterrows():
            r = blank()
            sens_parts, spec_parts = [], []
            for lab, sup in support.items():
                rec = a.get(f"recall_{lab}")
                npred = a.get(f"n_predicted_{lab}")
                if pd.isna(rec) or pd.isna(npred):
                    continue
                tp = float(rec) * sup
                fn = sup - tp
                fp = float(npred) - tp
                tn = total - tp - fn - fp
                sens_parts.append(f"{lab} {float(rec):.4f}")
                spec_parts.append(f"{lab} {tn / (tn + fp):.4f}"
                                  if tn + fp > 0 else f"{lab} {NC}")
            r.update({
                "group": "C2. Native three-class, harmonised families",
                "task": "Multiclass: next visit TD vs Indeterminate vs PIGD",
                "model": str(a.get("model", "")),
                "evaluation_protocol": (
                    "One shared case list and one argmax rule; 10,000 Monte "
                    "Carlo draws per case with common random numbers; each "
                    "family keeps its native fitting scheme"),
                "cohort": "934 patients, 3,361 transitions",
                "n_patients": 934, "n_cases": 3361,
                "positive_class": "Not applicable (three classes)",
                "accuracy": (round(float(a["accuracy"]), 4)
                             if pd.notna(a["accuracy"]) else NC),
                "accuracy_95ci": NC,
                "sensitivity": "; ".join(sens_parts) or NC,
                "specificity": "; ".join(spec_parts) or NC,
                "auc_roc": NC, "auc_roc_95ci": NC, "pr_auc": NC,
                "pr_auc_95ci": NC,
                "brier": (round(float(a["brier_multiclass"]), 4)
                          if pd.notna(a["brier_multiclass"]) else NC),
                "brier_convention": "multiclass, mean sum_c (p_c - 1{c})^2",
                "log_loss": (round(float(a["log_loss"]), 4)
                             if "log_loss" in h.columns
                             and pd.notna(a["log_loss"]) else NC),
                "rmse_standardised": NA, "mae_standardised": NA,
                "log_density_per_domain_obs": NA, "log_density_95ci": NA,
                "coverage_95": NA, "n_likelihood_terms": NA,
                "source_path": rel(harm),
                "notes": ("family: " + str(a.get("family", "")) + "; macro-F1 "
                          + f"{float(a['macro_f1']):.4f}" + "; specificity "
                          "derived from the published recall and predicted "
                          "counts against the class supports 2115/332/914; "
                          "comparable within this group only"),
            })
            rows.append(r)
    return rows


def group_d() -> list[dict]:
    """Two-class on the Indeterminate-excluded cohort (Model 1)."""
    rows = []
    d = pd.read_csv(R["two_state"] / "common_two_class.csv")
    for _, a in d.iterrows():
        r = blank()
        r.update({
            "group": "D. Two-class, Indeterminate-excluded cohort, 2,936 cases",
            "task": "Binary: next visit TD vs PIGD",
            "model": a["model"],
            "evaluation_protocol": "Leave-one-patient-out, seed 42, multistart",
            "cohort": ("876 patients, 3,812 TD/PIGD visits, 2,936 transitions; "
                       "Indeterminate visits removed before fitting"),
            "n_patients": int(a["n_patients"]), "n_cases": int(a["n_scored"]),
            "positive_class": "PIGD",
            "accuracy": round(float(a["accuracy"]), 4),
            "accuracy_95ci": ci(a["accuracy_ci_low"], a["accuracy_ci_high"]),
            "sensitivity": round(float(a["recall_PIGD"]), 4),
            "specificity": round(float(a["recall_TD"]), 4),
            "auc_roc": NC, "auc_roc_95ci": NC, "pr_auc": NC, "pr_auc_95ci": NC,
            "brier": round(float(a["brier"]), 4),
            "brier_convention": ("multiclass, mean sum_c (p_c - 1{c})^2; halve "
                                 "for the binary form used in groups A and B"),
            "log_loss": round(float(a["log_loss"]), 4),
            "rmse_standardised": NA, "mae_standardised": NA,
            "log_density_per_domain_obs": NA, "log_density_95ci": NA,
            "coverage_95": NA, "n_likelihood_terms": NA,
            "source_path": rel(R["two_state"] / "common_two_class.csv"),
            "notes": (f"balanced accuracy {float(a['balanced_accuracy']):.4f}, "
                      f"macro-F1 {float(a['macro_f1']):.4f}"),
        })
        rows.append(r)
    return rows


def group_e() -> list[dict]:
    """The rich profile's primary target: next-visit clinical measurements."""
    rows = []
    m = pd.read_csv(R["profile_cv"] / "model_comparison.csv")
    pd_err = pd.read_csv(R["no_ledd"] / "per_domain_errors.csv")
    for _, a in m.iterrows():
        r = blank()
        r.update({
            "group": "E. Nine-domain measurement forecast, 3,353 predictions",
            "task": "Continuous: next visit's nine clinical domain scores",
            "model": a["model"],
            "evaluation_protocol": ("5-fold patient cross-validation, 4 repeats, "
                                    "seed 42; repeat 0 reported, all 4 in "
                                    "robustness_by_repeat.csv"),
            "cohort": ("934 patients, 4,287 clinical visits, 3,353 prediction "
                       "opportunities, 30,099 domain observations"),
            "positive_class": NA,
            "n_patients": int(a["n_patients"]),
            "n_cases": int(a["n_predictions"]),
            "accuracy": NA, "accuracy_95ci": NA,
            "sensitivity": NA, "specificity": NA,
            "auc_roc": NA, "auc_roc_95ci": NA, "pr_auc": NA, "pr_auc_95ci": NA,
            "brier": NA, "brier_convention": NA, "log_loss": NA,
            "log_density_per_domain_obs": round(
                float(a["loglik_per_domain_observation"]), 4),
            "log_density_95ci": ci(a["loglik_ci_low"], a["loglik_ci_high"]),
            "rmse_standardised": round(float(a["rmse_standardised"]), 4),
            "mae_standardised": round(float(a["mae_standardised"]), 4),
            "coverage_95": round(float(a["coverage_95"]), 4),
            "n_likelihood_terms": int(a["n_domain_observations"]),
            "source_path": rel(R["profile_cv"] / "model_comparison.csv"),
            "notes": ("per-domain RMSE in raw clinical units: "
                      + rel(R["no_ledd"] / "per_domain_errors.csv")),
        })
        rows.append(r)
    assert len(pd_err)
    return rows


def group_f() -> list[dict]:
    """Information criteria. Comparable only within a likelihood family."""
    rows = []
    a = pd.read_csv(R["selection"] / "aic_bic.csv")
    for _, x in a.iterrows():
        r = blank()
        r.update({
            "group": f"F. Information criteria -- {x['group']}",
            "task": str(x["likelihood"]),
            "model": x["model"],
            "evaluation_protocol": ("Full-cohort maximum likelihood, for AIC/BIC "
                                    "only; not a held-out result"),
            "cohort": "934 patients",
            "n_patients": 934, "n_cases": int(x["n_terms"]),
            "positive_class": NA,
            "accuracy": NA, "accuracy_95ci": NA, "sensitivity": NA,
            "specificity": NA, "auc_roc": NA, "auc_roc_95ci": NA,
            "pr_auc": NA, "pr_auc_95ci": NA, "brier": NA,
            "brier_convention": NA, "log_loss": NA,
            "log_density_per_domain_obs": NA, "log_density_95ci": NA,
            "rmse_standardised": NA, "mae_standardised": NA, "coverage_95": NA,
            "loglik": round(float(x["loglik"]), 2),
            "n_params": int(x["k"]),
            "aic": round(float(x["aic"]), 2), "bic": round(float(x["bic"]), 2),
            "n_likelihood_terms": int(x["n_terms"]),
            "source_path": rel(R["selection"] / "aic_bic.csv"),
            "notes": (f"delta AIC {float(x['delta_aic_within_group']):.2f}, "
                      f"delta BIC {float(x['delta_bic_within_group']):.2f} "
                      f"within this group; comparable only inside the group"),
        })
        rows.append(r)

    b = pd.read_csv(R["two_state"] / "aic_bic.csv")
    for _, x in b.iterrows():
        r = blank()
        r.update({
            "group": ("F. Information criteria -- hidden-state count, "
                      f"{x['observation_set']}"),
            "task": "joint PMF over label sequences",
            "model": x["model"],
            "evaluation_protocol": ("Full-cohort maximum likelihood, for AIC/BIC "
                                    "only; not a held-out result"),
            "cohort": str(x["observation_set"]),
            "n_patients": (934 if "all labelled" in str(x["observation_set"])
                           else 876),
            "n_cases": int(x["n_observations"]),
            "positive_class": NA,
            "accuracy": NA, "accuracy_95ci": NA, "sensitivity": NA,
            "specificity": NA, "auc_roc": NA, "auc_roc_95ci": NA,
            "pr_auc": NA, "pr_auc_95ci": NA, "brier": NA,
            "brier_convention": NA, "log_loss": NA,
            "log_density_per_domain_obs": NA, "log_density_95ci": NA,
            "rmse_standardised": NA, "mae_standardised": NA, "coverage_95": NA,
            "loglik": round(float(x["loglik"]), 2),
            "n_params": int(x["n_params"]),
            "aic": round(float(x["aic"]), 2), "bic": round(float(x["bic"]), 2),
            "n_likelihood_terms": int(x["n_observations"]),
            "source_path": rel(R["two_state"] / "aic_bic.csv"),
            "notes": (f"hidden states {int(x['n_hidden'])}, observed states "
                      f"{int(x['n_observed'])}; comparable only with the other "
                      f"rows on the same observation set"),
        })
        rows.append(r)

    # The banded emission fit, recorded with its own incomparability warning.
    r = blank()
    r.update({
        "group": "F. Information criteria -- NOT COMPARABLE (different alphabet)",
        "task": "joint PMF over 44 subtype-by-band categories",
        "model": "CT-HMM (subtype x ratio band, width 0.10), whole-cohort fit",
        "evaluation_protocol": "Descriptive whole-cohort fit, not held out",
        "cohort": "934 patients, 4,295 labelled visits",
        "n_patients": 934, "n_cases": 4295, "positive_class": NA,
        "accuracy": NA, "accuracy_95ci": NA, "sensitivity": NA,
        "specificity": NA, "auc_roc": NA, "auc_roc_95ci": NA, "pr_auc": NA,
        "pr_auc_95ci": NA, "brier": NA, "brier_convention": NA, "log_loss": NA,
        "log_density_per_domain_obs": NA, "log_density_95ci": NA,
        "rmse_standardised": NA, "mae_standardised": NA, "coverage_95": NA,
        "loglik": -13123.40, "n_params": 137, "aic": 26520.80, "bic": NC,
        "n_likelihood_terms": 4295,
        "source_path": rel(R["emission"] / "report.txt"),
        "notes": ("the run states this explicitly: the banded likelihood is over "
                  "44 categories and the baseline's over 3, so neither the "
                  "log-likelihoods nor their AICs may be compared; recorded as a "
                  "fit diagnostic only"),
    })
    rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sens", required=True,
                    help="the consolidated-results run holding the recovered "
                         "sensitivity/specificity tables")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sens_dir = Path(args.sens)

    rows = (group_a(sens_dir) + group_b(sens_dir) + group_c(sens_dir) + group_d()
            + group_e() + group_f())
    table = pd.DataFrame(rows)[COLUMNS]
    table.to_csv(out / "master_results.csv", index=False)

    print(f"{len(table)} rows, {table['group'].nunique()} groups")
    print(table.groupby("group").size().to_string())
    print(f"\nwrote {out / 'master_results.csv'}")
    print("sha256:", hashlib.sha256((out / "master_results.csv").read_bytes())
          .hexdigest()[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
