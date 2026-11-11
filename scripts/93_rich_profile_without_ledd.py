"""The rich profile with levodopa removed: what the LEDD-free model delivers.

`clinical_profile.py` already keeps LEDD out of the nine predicted domains and
enters it as an observation-equation input instead. Removing "the LEDD part"
therefore means fitting the model with ``use_medication=False``, which drops
the ``beta[d] * ledd`` term and the nine coefficients that carry it; LEDD then
plays no role anywhere in the model. That variant (``cp.MODEL_NO_MED``) was
fitted in the cross-validation alongside the full model, so its held-out
predictions already exist and are re-read here rather than refitted.

What this script adds is the comparisons the original driver only ran for the
*full* model, so the LEDD-free model can be judged on its own:

* its paired predictive-density contrasts against assume-no-change, the
  population average and the level-only variant;
* its paired subtype-accuracy contrast against the subtype-history CT-HMM;
* the size of the medication coefficients that removal discards.

Every contrast uses the same helpers, pairing and seed as the original driver,
so the numbers are commensurable with the published tables.

    python scripts/93_rich_profile_without_ledd.py --out <dir>
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
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import clinical_profile as cp  # noqa: E402
from ppmi_subtype import next_visit as nv  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

#: The cross-validation run that holds the held-out predictions of every
#: variant at full precision. Its model_comparison.csv is the source of the
#: published rich-profile table.
SOURCE = ROOT / "experiments" / "20261103T220147Z_a2p_outcome_table_fullprec"

#: The subtype-history experiment the profile route is benchmarked against.
BASELINE = ROOT / "experiments" / "20261020T104756Z_next_visit_td_pigd_prediction"

CTHMM = "CT-HMM (latent, interval-aware)"

lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    lines.append(text)


def section(title: str) -> None:
    say("\n" + "=" * 78)
    say(title)
    say("=" * 78)


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def block(frame, **kw) -> str:
    return "    " + frame.to_string(**kw).replace("\n", "\n    ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    section("0. PROVENANCE")
    inputs = {
        "profile_predictions.csv": SOURCE / "profile_predictions.csv",
        "derived_label_predictions.csv": SOURCE / "derived_label_predictions.csv",
        "fold_coefficients.csv": SOURCE / "fold_coefficients.csv",
        "model_comparison.csv": SOURCE / "model_comparison.csv",
        "next_visit_predictions.csv": BASELINE / "next_visit_predictions.csv",
    }
    digests = {}
    for name, path in inputs.items():
        digests[name] = sha(path)
        say(f"  {name:32s} {digests[name][:16]}  {path.relative_to(ROOT)}")
    say("")
    say(f"  variant removed from: {cp.MODEL_FULL}")
    say(f"  variant reported:     {cp.MODEL_NO_MED}")
    say("  removal = use_medication=False: the beta[d]*ledd term and its nine")
    say("  coefficients are dropped, so LEDD enters nowhere in the model.")

    preds = pd.read_csv(inputs["profile_predictions.csv"], low_memory=False)
    primary = preds[preds["repeat"] == 0].reset_index(drop=True)

    # -----------------------------------------------------------------
    section("1. HELD-OUT PREDICTIVE DENSITY, LEDD-FREE MODEL VS THE REST")
    rows = [cp.profile_metrics(g.reset_index(drop=True), seed=cfg.seed)
            for _, g in primary.groupby("model", sort=False)]
    cmp_frame = pd.DataFrame([{k: v for k, v in r.items() if k != "per_domain"}
                              for r in rows])
    cmp_frame = cmp_frame.sort_values("loglik_per_domain_observation",
                                      ascending=False)
    cmp_frame.round(6).to_csv(out / "model_comparison_recomputed.csv",
                              index=False)
    say(block(cmp_frame[["model", "n_predictions",
                         "loglik_per_domain_observation", "loglik_ci_low",
                         "loglik_ci_high", "rmse_standardised",
                         "mae_standardised", "coverage_95"]].round(4),
              index=False))

    per_domain = [{"model": r["model"], **d} for r in rows
                  for d in r.get("per_domain", [])]
    pd.DataFrame(per_domain).round(6).to_csv(out / "per_domain_errors.csv",
                                             index=False)

    # -----------------------------------------------------------------
    section("2. PAIRED CONTRASTS FOR THE LEDD-FREE MODEL")
    pairs = [("LEDD-free profile against assume-no-change",
              cp.MODEL_NO_MED, cp.MODEL_LOCF),
             ("LEDD-free profile against the population average",
              cp.MODEL_NO_MED, cp.MODEL_POPMEAN),
             ("LEDD-free profile against level only",
              cp.MODEL_NO_MED, cp.MODEL_NO_RATE),
             ("cost of removing LEDD (full minus LEDD-free)",
              cp.MODEL_FULL, cp.MODEL_NO_MED)]
    crows = []
    for what, a, b in pairs:
        fa = primary[primary["model"] == a]
        fb = primary[primary["model"] == b]
        crows.append({"contrast": what, "model_a": a, "model_b": b,
                      **cp.paired_loglik_difference(fa, fb, seed=cfg.seed)})
    contrasts = pd.DataFrame(crows)
    contrasts.round(6).to_csv(out / "contrasts.csv", index=False)
    say(block(contrasts[["contrast", "n_paired", "loglik_delta",
                         "delta_ci_low", "delta_ci_high",
                         "share_a_better"]].round(4), index=False))
    say("")
    say("  loglik_delta is the per-prediction mean log-density difference,")
    say("  paired on identical held-out visits, with a patient-clustered CI.")

    # -----------------------------------------------------------------
    section("3. SECONDARY ENDPOINT: THE DERIVED TD/PIGD LABEL")
    label_preds = pd.read_csv(inputs["derived_label_predictions.csv"],
                              low_memory=False)
    bl = pd.read_csv(inputs["next_visit_predictions.csv"], low_memory=False)
    key = ["PATNO", "current_event", "target_event"]
    bl = bl.merge(label_preds[key].drop_duplicates(), on=key, how="inner")
    bl = bl[bl["model"] == CTHMM].copy()
    bl["model"] = "Subtype-history: " + bl["model"]

    # The saved prediction file carries neither the tie flag nor visit_index,
    # which the pairing helpers need. The flag is rebuilt from its definition;
    # visit_index is taken back from the prediction file of the same run, where
    # the visit triple identifies it uniquely.
    label_preds["is_tie"] = (label_preds["two_class_defined"]
                             & (label_preds["P_TD"] == label_preds["P_PIGD"]))
    vidx = (primary[key + ["visit_index"]].drop_duplicates(subset=key))
    label_preds = label_preds.merge(vidx, on=key, how="left", validate="m:1")
    if label_preds["visit_index"].isna().any():
        raise RuntimeError("a derived-label case has no matching visit_index")

    cols = list(set(label_preds.columns) & set(bl.columns))
    joint = pd.concat([label_preds[cols], bl[cols]], ignore_index=True)
    common = None
    for _, g in joint.groupby("model", sort=False):
        ss = set(map(tuple, g[key].to_numpy()))
        common = ss if common is None else (common & ss)
    mask = np.array([tuple(r) in common for r in joint[key].to_numpy()])
    n_dropped = int((~mask).sum())
    joint = joint[mask].reset_index(drop=True)
    say(f"  {len(common):,} cases shared by all models"
        + (f"; {n_dropped:,} rows dropped to pair them." if n_dropped else "."))

    lrows = [nv.two_class_metrics(g.reset_index(drop=True), seed=cfg.seed)
             for _, g in joint.groupby("model", sort=False)]
    lcmp = pd.DataFrame(lrows).sort_values("accuracy", ascending=False)
    lcmp.round(6).to_csv(out / "derived_label_comparison.csv", index=False)
    say("")
    say(block(lcmp[["model", "n_scored", "accuracy",
                    "accuracy_ci_patient_bootstrap_low",
                    "accuracy_ci_patient_bootstrap_high",
                    "brier_two_class", "log_loss"]].round(4), index=False))

    ref = "Subtype-history: " + CTHMM
    jc = []
    for what, a, b in (("LEDD-free profile vs subtype-history CT-HMM",
                        cp.MODEL_NO_MED, ref),
                       ("LEDD-free profile vs the full profile",
                        cp.MODEL_NO_MED, cp.MODEL_FULL)):
        jc.append({"contrast": what, **nv.paired_accuracy_difference(
            joint[joint["model"] == a], joint[joint["model"] == b],
            seed=cfg.seed)})
    lcontrast = pd.DataFrame(jc)
    lcontrast.round(6).to_csv(out / "derived_label_contrasts.csv", index=False)
    say("")
    say(block(lcontrast[["contrast", "accuracy_delta",
                         "delta_ci_patient_bootstrap_low",
                         "delta_ci_patient_bootstrap_high",
                         "a_right_b_wrong", "b_right_a_wrong",
                         "mcnemar_exact_p_ignores_clustering"]].round(4),
              index=False))

    # -----------------------------------------------------------------
    section("4. WHAT REMOVAL DISCARDS: THE MEDICATION COEFFICIENTS")
    coef = pd.read_csv(inputs["fold_coefficients.csv"])
    beta = (coef.groupby("domain")["medication_beta"]
            .agg(n_folds="size", mean="mean", sd="std", min="min", max="max")
            .reindex(list(cp.DOMAINS)).reset_index())
    beta["share_folds_positive"] = [
        float((coef.loc[coef["domain"] == d, "medication_beta"] > 0).mean())
        for d in beta["domain"]]
    beta.round(6).to_csv(out / "medication_coefficients.csv", index=False)
    say(block(beta.round(4), index=False))
    say("")
    say("  Units: standard deviations of the domain per standardised LEDD unit,")
    say("  from the full model's folds. These are the nine numbers the")
    say("  LEDD-free model does not have.")

    # -----------------------------------------------------------------
    summary = {
        "source_run": str(SOURCE.relative_to(ROOT)),
        "baseline_run": str(BASELINE.relative_to(ROOT)),
        "input_sha256": digests,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True).stdout.strip(),
        "seed": cfg.seed,
        "model_comparison": cmp_frame.to_dict(orient="records"),
        "contrasts": contrasts.to_dict(orient="records"),
        "derived_label_comparison": lcmp.to_dict(orient="records"),
        "derived_label_contrasts": lcontrast.to_dict(orient="records"),
        "medication_coefficients": beta.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2,
                                                 default=float) + "\n")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
