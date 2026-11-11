"""Consolidate every completed model result into one verified table.

This script does no new modelling. It reads the experiment directories that
already hold held-out predictions and metrics, recovers the two metrics that
were never written down (sensitivity and specificity, which the project's own
prediction files determine exactly), re-derives accuracy from the predictions
as a cross-check against each run's published value, and writes one row per
model-and-configuration with its evaluation protocol, cohort size, positive
class and source path.

Sensitivity and specificity need a positive class. Throughout this project the
two-class endpoint is scored with PIGD as the positive outcome -- that is the
class whose probability ``P_PIGD`` is reported, and the class counted as
positive in the AUC runs -- so the same choice is used here:

    sensitivity = recall of PIGD = P(predict PIGD | actual PIGD)
    specificity = recall of TD   = P(predict TD   | actual TD)

For the three-class endpoint a single sensitivity is not defined, so one-vs-rest
sensitivity and specificity are given per class instead.

    python scripts/94_consolidated_results.py --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
sys.path.insert(0, str(ROOT / "src"))

#: Runs read. Each is named by what it is the authoritative source of.
RUNS = {
    "profile_cv": EXP / "20261103T220147Z_a2p_outcome_table_fullprec",
    "profile_cv_vs_history": EXP / "20261020T180515Z_clinical_profile_cv",
    "subtype_history_lopo": EXP / "20261020T104756Z_next_visit_td_pigd_prediction",
    "model_selection": EXP / "20261023T194317Z_model_selection_criteria",
    "roc_auc": EXP / "20261023T202640Z_roc_auc_comparison",
    "ratio_band": EXP / "20261104T152154Z_cthmm_ratio_banded_observations",
    "ratio_band_auc": EXP / "20261104T172138Z_ratio_banded_auc_roc",
    "band_width": EXP / "20261105T025024Z_cthmm_ratio_band_width_extended",
    "band_width_prior": EXP / "20261105T003425Z_cthmm_ratio_band_width_sensitivity",
    "band_emission": EXP / "20261104T155439Z_ratio_band_emission_diagnostic",
    "two_state": EXP / "20261110T143200Z_two_state_hmm_confirm",
    "two_state_prior": EXP / "20261110T141727Z_two_state_hmm",
    "no_ledd": EXP / "20261111T121021Z_rich_profile_no_ledd",
}

KEY = ["PATNO", "current_event", "target_event"]
CTHMM = "CT-HMM (latent, interval-aware)"

lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    lines.append(text)


def section(title: str) -> None:
    say("\n" + "=" * 78)
    say(title)
    say("=" * 78)


def block(frame, **kw) -> str:
    return "    " + frame.to_string(**kw).replace("\n", "\n    ")


def two_class_counts(frame: pd.DataFrame) -> dict:
    """Confusion counts and derived rates for one model, PIGD positive.

    Scored cases only: eligible (the actual next label is TD or PIGD) and
    two-class defined, which is the same restriction the project's own
    accuracy numbers use.
    """
    s = frame[frame["eligible"].astype(bool)
              & frame["two_class_defined"].astype(bool)]
    actual_pigd = (s["actual_next_label"] == "PIGD").to_numpy()
    pred_pigd = (s["predicted_state"] == "PIGD").to_numpy()
    tp = int((actual_pigd & pred_pigd).sum())
    fn = int((actual_pigd & ~pred_pigd).sum())
    fp = int((~actual_pigd & pred_pigd).sum())
    tn = int((~actual_pigd & ~pred_pigd).sum())
    n = tp + fn + fp + tn
    return {
        "n_scored": n,
        "n_patients": int(s["PATNO"].nunique()),
        "n_positive_PIGD": tp + fn,
        "n_negative_TD": fp + tn,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy_recomputed": (tp + tn) / n if n else np.nan,
        "sensitivity_PIGD": tp / (tp + fn) if tp + fn else np.nan,
        "specificity_TD": tn / (tn + fp) if tn + fp else np.nan,
        "ppv_PIGD": tp / (tp + fp) if tp + fp else np.nan,
        "npv_TD": tn / (tn + fn) if tn + fn else np.nan,
    }


def one_vs_rest(confusion: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Per-class sensitivity and specificity from a 3-class confusion matrix.

    ``confusion`` is indexed by the ACTUAL label with one column per predicted
    label, so rows sum to each class's support.
    """
    M = confusion.loc[labels, labels].to_numpy(dtype=float)
    total = M.sum()
    rows = []
    for i, lab in enumerate(labels):
        tp = M[i, i]
        fn = M[i, :].sum() - tp
        fp = M[:, i].sum() - tp
        tn = total - tp - fn - fp
        rows.append({"label": lab, "support": int(M[i, :].sum()),
                     "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
                     "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                     "specificity": tn / (tn + fp) if tn + fp else np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    section("0. RUNS READ")
    for name, path in RUNS.items():
        if not path.exists():
            raise SystemExit(f"missing run: {path}")
        say(f"  {name:22s} {path.name}")

    # ------------------------------------------------------------------
    section("1. TWO-CLASS SENSITIVITY AND SPECIFICITY, MATCHED 3,020 CASES")
    say("  Positive class: PIGD. Cohort: the intersection of the rich")
    say("  profile's cross-validation rows and the CT-HMM's leave-one-")
    say("  patient-out rows, which is the set every published matched")
    say("  comparison uses.")

    prof = pd.read_csv(RUNS["profile_cv"] / "derived_label_predictions.csv",
                       low_memory=False)
    hist = pd.read_csv(RUNS["subtype_history_lopo"] / "next_visit_predictions.csv",
                       low_memory=False)
    hist = hist.merge(prof[KEY].drop_duplicates(), on=KEY, how="inner")
    hist["model"] = "Subtype-history: " + hist["model"]

    cols = sorted(set(prof.columns) & set(hist.columns))
    joint = pd.concat([prof[cols], hist[cols]], ignore_index=True)

    common = None
    for _, g in joint.groupby("model", sort=False):
        ss = set(map(tuple, g[KEY].to_numpy()))
        common = ss if common is None else (common & ss)
    mask = np.array([tuple(r) in common for r in joint[KEY].to_numpy()])
    dropped = int((~mask).sum())
    joint = joint[mask].reset_index(drop=True)
    say(f"  {len(common):,} cases shared by all {joint['model'].nunique()} models"
        + (f"; {dropped:,} rows dropped to pair them." if dropped else "."))

    rows = [{"model": name, **two_class_counts(g)}
            for name, g in joint.groupby("model", sort=False)]
    sens = pd.DataFrame(rows).sort_values("accuracy_recomputed", ascending=False)
    sens.round(6).to_csv(out / "two_class_sensitivity_specificity.csv",
                         index=False)
    say("")
    say(block(sens[["model", "n_scored", "tp", "fn", "fp", "tn",
                    "accuracy_recomputed", "sensitivity_PIGD",
                    "specificity_TD", "ppv_PIGD", "npv_TD"]].round(4),
              index=False))

    # Cross-check: the recomputed accuracy against each run's own number.
    section("2. CROSS-CHECK: RECOMPUTED ACCURACY VS PUBLISHED ACCURACY")
    pub = pd.read_csv(RUNS["profile_cv_vs_history"]
                      / "derived_label_vs_subtype_history.csv")
    chk = sens[["model", "accuracy_recomputed"]].merge(
        pub[["model", "accuracy"]].rename(columns={"accuracy": "accuracy_published"}),
        on="model", how="left")
    chk["abs_diff"] = (chk["accuracy_recomputed"] - chk["accuracy_published"]).abs()
    chk["agrees_to_4dp"] = chk["abs_diff"] < 5e-5
    chk.round(6).to_csv(out / "accuracy_crosscheck.csv", index=False)
    say(block(chk.round(5), index=False))

    # ------------------------------------------------------------------
    section("2b. TWO-CLASS SENSITIVITY AND SPECIFICITY, LOPO 3,029 CASES")
    say("  The subtype-history baseline suite on its own full cohort, which is")
    say("  nine cases larger than the matched set above. Same positive class.")
    full = pd.read_csv(RUNS["subtype_history_lopo"] / "next_visit_predictions.csv",
                       low_memory=False)
    rows_b = [{"model": name, **two_class_counts(g)}
              for name, g in full.groupby("model", sort=False)]
    sens_b = pd.DataFrame(rows_b).sort_values("accuracy_recomputed",
                                              ascending=False)
    pub_b = pd.read_csv(RUNS["subtype_history_lopo"] / "model_comparison.csv")
    sens_b = sens_b.merge(
        pub_b[["model", "accuracy"]].rename(
            columns={"accuracy": "accuracy_published"}), on="model", how="left")
    sens_b["abs_diff"] = (sens_b["accuracy_recomputed"]
                          - sens_b["accuracy_published"]).abs()
    sens_b["agrees_to_4dp"] = sens_b["abs_diff"] < 5e-5
    sens_b.round(6).to_csv(out / "two_class_sensitivity_specificity_lopo.csv",
                           index=False)
    say("")
    say(block(sens_b[["model", "n_scored", "tp", "fn", "fp", "tn",
                      "accuracy_recomputed", "accuracy_published",
                      "agrees_to_4dp", "sensitivity_PIGD",
                      "specificity_TD"]].round(4), index=False))

    # ------------------------------------------------------------------
    section("3. THREE-CLASS ONE-VS-REST SENSITIVITY AND SPECIFICITY")
    say("  Native three-class endpoint, 3,361 transitions, 934 patients,")
    say("  leave-one-patient-out. A single sensitivity is undefined for three")
    say("  classes, so each class is scored one-against-the-rest.")
    labels = ["TD", "Indeterminate", "PIGD"]
    frames = []
    for model, fname in (
            ("CT-HMM (3 hidden x 3 observed)",
             "confusion_three_class_ct_hmm_3_hidden_x_3_observed_existing_model.csv"),
            ("CT-HMM (2 hidden x 3 observed)",
             "confusion_three_class_ct_hmm_2_hidden_x_3_observed.csv"),
            ("Persistence (last observed label)",
             "confusion_three_class_persistence_last_observed_label.csv"),
            ("Marginal (training label prior)",
             "confusion_three_class_marginal_training_label_prior.csv")):
        cm = pd.read_csv(RUNS["two_state"] / fname).set_index("actual_label")
        f = one_vs_rest(cm, labels)
        f.insert(0, "model", model)
        frames.append(f)
    three = pd.concat(frames, ignore_index=True)
    three.round(6).to_csv(out / "three_class_one_vs_rest.csv", index=False)
    say("")
    say(block(three.round(4), index=False))

    # ------------------------------------------------------------------
    section("4. DUPLICATE AND SUPERSEDED RUNS")
    dup = []
    for label, a, b, file_, keys in (
        ("band-width sweep", "band_width", "band_width_prior", "model_metrics.csv",
         ["accuracy", "auc_roc", "pr_auc", "brier_two_class", "log_loss"]),
        ("two-hidden-state HMM", "two_state", "two_state_prior", "common_two_class.csv",
         ["accuracy", "macro_f1", "balanced_accuracy", "brier", "log_loss"]),
    ):
        fa = pd.read_csv(RUNS[a] / file_)
        fb = pd.read_csv(RUNS[b] / file_)
        m = fa[["model"] + keys].merge(fb[["model"] + keys], on="model",
                                      suffixes=("_new", "_old"), how="outer")
        worst = 0.0
        for k in keys:
            d = (m[k + "_new"] - m[k + "_old"]).abs()
            worst = max(worst, float(d.max(skipna=True)) if d.notna().any() else 0.0)
        only_new = sorted(set(fa["model"]) - set(fb["model"]))
        dup.append({"comparison": label,
                    "kept_run": RUNS[a].name, "superseded_run": RUNS[b].name,
                    "n_models_shared": int(m[[k + "_old" for k in keys]]
                                           .notna().any(axis=1).sum()),
                    "max_abs_metric_difference": worst,
                    "models_only_in_kept_run": "; ".join(only_new) or "none"})
    dupf = pd.DataFrame(dup)
    dupf.to_csv(out / "duplicate_runs.csv", index=False)
    say(block(dupf, index=False))

    # ------------------------------------------------------------------
    section("5. BRIER CONVENTION AUDIT")
    say("  The repository uses two Brier definitions. Mixing them would")
    say("  make a model look twice as badly calibrated as it is.")
    conv = pd.DataFrame([
        {"column": "brier_two_class",
         "definition": "mean (p_PIGD - 1{PIGD})^2, the single-outcome binary form",
         "source": "src/ppmi_subtype/next_visit.py:628",
         "used_by": "profile CV, subtype-history LOPO, ratio-band runs, band-width sweep"},
        {"column": "brier",
         "definition": "mean sum over classes (p_c - 1{c})^2, which is 2x the "
                       "binary form when there are two classes",
         "source": "src/ppmi_subtype/two_state_hmm.py:1138 and eight sibling modules",
         "used_by": "two-hidden-state run, harmonised family comparison, LOO, CTMC"},
    ])
    say(block(conv, index=False))
    two = pd.read_csv(RUNS["two_state"] / "common_two_class.csv")
    r = two[two["model"] == "CT-HMM (3 hidden x 3 observed, existing model)"]
    say("")
    say(f"  Worked example: the 3-hidden-state model on the 2,936-case")
    say(f"  two-class view reports brier = {float(r['brier'].iloc[0]):.4f};")
    say(f"  halved, {float(r['brier'].iloc[0]) / 2:.4f}, is the binary form "
        f"comparable with brier_two_class.")

    # ------------------------------------------------------------------
    summary = {
        "runs": {k: str(v.relative_to(ROOT)) for k, v in RUNS.items()},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "matched_two_class_cases": len(common),
        "two_class_sensitivity_specificity": sens.to_dict(orient="records"),
        "two_class_sensitivity_specificity_lopo": sens_b.to_dict(orient="records"),
        "accuracy_crosscheck": chk.to_dict(orient="records"),
        "three_class_one_vs_rest": three.to_dict(orient="records"),
        "duplicate_runs": dupf.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2,
                                                 default=float) + "\n")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
