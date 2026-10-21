"""Comparison tables for the manuscript, and a check that its numbers are real.

Two jobs:

1. Build the head-to-head comparison table from the two experiment runs.
2. Re-read every headline number quoted in ``MANUSCRIPT.md`` out of the run
   directories and fail loudly if any of them disagrees. The manuscript is
   written by hand, so this is what stops a transcription slip from surviving.

    python manuscript/make_tables.py [--ct-hmm DIR] [--profile DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CT_HMM_RUN = "experiments/20261020T104756Z_next_visit_td_pigd_prediction"
PROFILE_RUN = "experiments/20261020T180515Z_clinical_profile_cv"


def build_comparison(ct_dir: Path, rp_dir: Path) -> pd.DataFrame:
    """The fair head-to-head: both methods on the paired evaluation set.

    Protocol differences that cannot be removed by pairing are carried in the
    table as their own rows, so the reader sees them next to the numbers.
    """
    paired = pd.read_csv(rp_dir / "derived_label_vs_subtype_history.csv")
    ct_sum = json.loads((ct_dir / "summary.json").read_text())
    rp_sum = json.loads((rp_dir / "summary.json").read_text())

    def row(model: str) -> dict:
        r = paired[paired["model"] == model]
        if not len(r):
            raise SystemExit(f"model not in paired table: {model}")
        r = r.iloc[0]
        return {"accuracy": r["accuracy"],
                "ci_low": r["accuracy_ci_patient_bootstrap_low"],
                "ci_high": r["accuracy_ci_patient_bootstrap_high"],
                "brier": r["brier_two_class"], "log_loss": r["log_loss"],
                "n_scored": int(r["n_scored"])}

    ct = row("Subtype-history: CT-HMM (latent, interval-aware)")
    rp = row("Profile: level + rate + medication")
    contrasts = pd.read_csv(rp_dir / "derived_label_contrasts.csv")
    head = contrasts[contrasts["contrast"].str.startswith(
        "Profile: level + rate + medication")].iloc[0]

    rows = [
        {"quantity": "Primary prediction target",
         "method_1_ct_hmm": "next observed subtype label",
         "method_2_rich_profile": "next visit's 9 clinical measurements"},
        {"quantity": "Representation of the patient",
         "method_1_ct_hmm": "3-state latent phenotype + misclassification",
         "method_2_rich_profile": "9 domains x (level, rate), continuous"},
        {"quantity": "Medication",
         "method_1_ct_hmm": "not used",
         "method_2_rich_profile": "LEDD as an observation-equation input"},
        {"quantity": "Elapsed time between visits",
         "method_1_ct_hmm": "exp(Q dt), exact",
         "method_2_rich_profile": "local linear trend over dt, exact"},
        {"quantity": "Validation scheme",
         "method_1_ct_hmm": "leave-one-patient-out, 934 folds",
         "method_2_rich_profile": "5-fold patient CV x 4 repeats, 20 folds"},
        {"quantity": "Patients",
         "method_1_ct_hmm": f"{ct_sum['counts']['n_patients']:,}",
         "method_2_rich_profile": f"{rp_sum['counts']['n_patients']:,}"},
        {"quantity": "Prediction opportunities",
         "method_1_ct_hmm": f"{ct_sum['counts']['n_cases']:,}",
         "method_2_rich_profile": f"{rp_sum['counts']['n_cases']:,}"},
        {"quantity": "Failed folds",
         "method_1_ct_hmm": str(ct_sum["counts"]["n_folds_failed"]),
         "method_2_rich_profile": str(rp_sum["counts"]["n_failed"])},
        {"quantity": "Automated checks passed",
         "method_1_ct_hmm": f"{sum(ct_sum['checks'].values())}/{len(ct_sum['checks'])}",
         "method_2_rich_profile": f"{sum(rp_sum['checks'].values())}/{len(rp_sum['checks'])}"},
        {"quantity": "--- Subtype endpoint, PAIRED on "
                     f"{ct['n_scored']:,} shared held-out cases ---",
         "method_1_ct_hmm": "", "method_2_rich_profile": ""},
        {"quantity": "Accuracy",
         "method_1_ct_hmm": f"{ct['accuracy']:.4f} "
                            f"[{ct['ci_low']:.4f}, {ct['ci_high']:.4f}]",
         "method_2_rich_profile": f"{rp['accuracy']:.4f} "
                                  f"[{rp['ci_low']:.4f}, {rp['ci_high']:.4f}]"},
        {"quantity": "Brier score (two-class)",
         "method_1_ct_hmm": f"{ct['brier']:.4f}",
         "method_2_rich_profile": f"{rp['brier']:.4f}"},
        {"quantity": "Log loss",
         "method_1_ct_hmm": f"{ct['log_loss']:.4f}",
         "method_2_rich_profile": f"{rp['log_loss']:.4f}"},
        {"quantity": "Paired difference (profile - CT-HMM)",
         "method_1_ct_hmm": "reference",
         "method_2_rich_profile":
             f"{head['accuracy_delta']:+.4f} "
             f"[{head['delta_ci_patient_bootstrap_low']:+.4f}, "
             f"{head['delta_ci_patient_bootstrap_high']:+.4f}]  "
             f"-> interval includes zero, no significant difference"},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Verification: every headline number must exist in a run directory
# ---------------------------------------------------------------------------
def collect_truth(ct_dir: Path, rp_dir: Path) -> dict[str, float]:
    paired = pd.read_csv(rp_dir / "derived_label_vs_subtype_history.csv"
                         ).set_index("model")
    ct_cmp = pd.read_csv(ct_dir / "model_comparison.csv").set_index("model")
    rp_cmp = pd.read_csv(rp_dir / "model_comparison.csv").set_index("model")
    rp_con = pd.read_csv(rp_dir / "contrasts.csv").set_index("contrast")
    lab_con = pd.read_csv(rp_dir / "derived_label_contrasts.csv")
    dim = pd.read_csv(rp_dir / "representation_dimensionality.csv").set_index("which")
    h3 = pd.read_csv(rp_dir / "h3_component_vs_subtype.csv")

    CT = "Subtype-history: CT-HMM (latent, interval-aware)"
    RP = "Profile: level + rate + medication"
    truth = {
        "paired CT-HMM accuracy": paired.loc[CT, "accuracy"],
        "paired CT-HMM ci_low": paired.loc[CT, "accuracy_ci_patient_bootstrap_low"],
        "paired CT-HMM ci_high": paired.loc[CT, "accuracy_ci_patient_bootstrap_high"],
        "paired profile accuracy": paired.loc[RP, "accuracy"],
        "paired profile ci_low": paired.loc[RP, "accuracy_ci_patient_bootstrap_low"],
        "paired profile ci_high": paired.loc[RP, "accuracy_ci_patient_bootstrap_high"],
        "paired persistence accuracy":
            paired.loc["Subtype-history: Persistence (last non-Indeterminate)",
                       "accuracy"],
        "CT-HMM own LOPO accuracy":
            ct_cmp.loc["CT-HMM (latent, interval-aware)", "accuracy"],
        "profile loglik": rp_cmp.loc[RP, "loglik_per_domain_observation"],
        "locf loglik": rp_cmp.loc["Last observation carried forward",
                                  "loglik_per_domain_observation"],
        "popmean loglik": rp_cmp.loc["Population mean trajectory",
                                     "loglik_per_domain_observation"],
        "delta vs locf": rp_con.loc["rich profile against assume-no-change",
                                    "loglik_delta"],
        "delta vs popmean":
            rp_con.loc["rich profile against the population average", "loglik_delta"],
        "delta rate": rp_con.loc["modelling the rate of change", "loglik_delta"],
        "delta medication": rp_con.loc["medication as an input", "loglik_delta"],
        "paired label delta": float(lab_con[lab_con["contrast"].str.startswith(RP)]
                                    ["accuracy_delta"].iloc[0]),
        "level pc1": dim.loc["level", "pc1_share"],
        "rate pc1": dim.loc["rate", "pc1_share"],
        "hy level delta": float(h3[(h3["outcome"] == "hy_ge_3")
                                   & (h3["kind"] == "level")]["delta"].iloc[0]),
        "mci level delta": float(h3[(h3["outcome"] == "mci")
                                    & (h3["kind"] == "level")]["delta"].iloc[0]),
    }
    return truth


#: (label, value as written in the manuscript). Checked against `collect_truth`.
QUOTED = {
    "paired CT-HMM accuracy": 0.8371,
    "paired CT-HMM ci_low": 0.8202,
    "paired CT-HMM ci_high": 0.8540,
    "paired profile accuracy": 0.8404,
    "paired profile ci_low": 0.8248,
    "paired profile ci_high": 0.8566,
    "paired persistence accuracy": 0.8338,
    "CT-HMM own LOPO accuracy": 0.8369,
    "profile loglik": -1.0453,
    "locf loglik": -1.1187,
    "popmean loglik": -1.4342,
    "delta vs locf": 0.0733,
    "delta vs popmean": 0.3889,
    "delta rate": 0.0040,
    "delta medication": 0.0058,
    "paired label delta": 0.0033,
    "level pc1": 0.422,
    "rate pc1": 0.372,
    "hy level delta": 0.105,
    "mci level delta": 0.161,
}


def verify(ct_dir: Path, rp_dir: Path) -> pd.DataFrame:
    truth = collect_truth(ct_dir, rp_dir)
    rows = []
    for key, quoted in QUOTED.items():
        actual = float(truth[key])
        # Quoted to the precision printed in the manuscript, so compare at that
        # precision rather than demanding exact equality.
        dp = len(str(quoted).split(".")[1]) if "." in str(quoted) else 0
        ok = round(actual, dp) == round(quoted, dp)
        rows.append({"quantity": key, "manuscript": quoted,
                     "source_value": actual, "matches": ok})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-hmm", default=CT_HMM_RUN)
    ap.add_argument("--profile", default=PROFILE_RUN)
    ap.add_argument("--out", default="manuscript/tables")
    args = ap.parse_args()

    ct_dir = ROOT / args.ct_hmm
    rp_dir = ROOT / args.profile
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison(ct_dir, rp_dir)
    comparison.to_csv(out / "table1_method_comparison.csv", index=False)
    print("Table 1 — method comparison\n")
    print(comparison.to_string(index=False))

    # Supporting tables, copied verbatim from the runs so the manuscript and
    # the experiment cannot drift apart.
    pd.read_csv(rp_dir / "derived_label_vs_subtype_history.csv").to_csv(
        out / "table2_subtype_paired_all_models.csv", index=False)
    pd.read_csv(rp_dir / "model_comparison.csv").to_csv(
        out / "table3_profile_prediction.csv", index=False)
    pd.read_csv(rp_dir / "contrasts.csv").to_csv(
        out / "table4_profile_contrasts.csv", index=False)
    pd.read_csv(rp_dir / "h3_component_vs_subtype.csv").to_csv(
        out / "table5_clinical_outcome_associations.csv", index=False)

    check = verify(ct_dir, rp_dir)
    check.to_csv(out / "manuscript_number_check.csv", index=False)
    print("\n\nVerification of quoted numbers against the run directories\n")
    print(check.to_string(index=False))

    bad = check[~check["matches"]]
    if len(bad):
        print("\nMISMATCHES:")
        print(bad.to_string(index=False))
        return 1
    print(f"\nAll {len(check)} quoted numbers match their source files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
