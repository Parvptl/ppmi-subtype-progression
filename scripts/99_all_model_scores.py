"""Every model's scores in one table: classification metrics, AIC and BIC.

Reads the final consolidated run for the rows it already holds, computes the
binary TD-vs-PIGD metrics the 10-fold HMM configuration study never published
(from its saved per-case predictions), and gathers every saved AIC/BIC,
including the ratio-band fits' per-fold training AIC with BIC computed from the
fold's own training visit count. No model is refitted.

Usage:
    python scripts/99_all_model_scores.py --out experiments/<run>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "experiments"
FINAL = E / "20261111T170322Z_final_consolidated_results"
S4_CV = E / "20261014T100057Z_s4_hmm_cv"
S4_POP = E / "20261014T094649Z_s4_hmm_stability" / "analysis" / "population_summary.csv"
S4_POP_MAXFUN = E / "20261014T094839Z_s4_hmm_population_maxfun" / "analysis" / "population_summary.csv"
IC = E / "20261023T194317Z_model_selection_criteria" / "aic_bic.csv"
TWO_IC = E / "20261110T143200Z_two_state_hmm_confirm" / "aic_bic.csv"
BAND = E / "20261105T025024Z_cthmm_ratio_band_width_extended"
BAND_DIAG = E / "20261104T155439Z_ratio_band_emission_diagnostic" / "summary.json"
PROFILE_CONT = E / "20261103T213859Z_a2_clinical_profile_cv_rerun" / "model_comparison.csv"
N_VISITS = 4295

CONFIGS = {
    "C3_f50": "CT-HMM, 3 states, floor 0.5",
    "C3_f60": "CT-HMM, 3 states, floor 0.6",
    "C3_f70": "CT-HMM, 3 states, floor 0.7",
    "C3_free": "CT-HMM, 3 states, no floor",
    "C4_free": "CT-HMM, 4 states, no floor",
    "C5_free": "CT-HMM, 5 states, no floor",
}
L = "L: label sequences, 4,295 labelled visits"
T = "T: label sequences, 3,812 TD/PIGD visits"
D = "D: density over 9 clinical domains, 38,486 observations"
BG = "Bw: ratio-band joint alphabet, its own per width"


def rel(p) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def engine():
    spec = importlib.util.spec_from_file_location(
        "recov", ROOT / "scripts" / "96_final_metric_recovery.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recov"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    cfg = ap.parse_args()
    out = Path(cfg.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rc = engine()
    rows: list[dict] = []

    def put(block, model, r, aic=np.nan, bic=np.nan, ic_group="", aic_note="",
            note="", extra=None):
        d = {
            "block": block, "model": model,
            "evaluation_protocol": r["evaluation_protocol"],
            "n_cases": int(r["n_scored"]), "n_patients": int(r["n_patients"]),
        }
        for k in ("accuracy", "sensitivity_PIGD", "specificity_TD", "precision_PIGD",
                  "f1_PIGD", "balanced_accuracy", "auc_roc", "pr_auc",
                  "brier_binary", "log_loss"):
            d[k] = r[k]
            if f"{k}_ci_low" in r.index:
                d[f"{k}_ci_low"] = r[f"{k}_ci_low"]
                d[f"{k}_ci_high"] = r[f"{k}_ci_high"]
        d.update({"score_degenerate": bool(r.get("score_degenerate", False)),
                  "aic": aic, "bic": bic, "aic_bic_group": ic_group,
                  "aic_bic_note": aic_note, "source_path": r["source_path"],
                  "notes": note})
        if extra:
            d.update(extra)
        rows.append(d)

    # ------------------------------ CT-HMM configuration study (10-fold CV)
    pub = pd.read_csv(S4_CV / "analysis" / "cv_config_metrics.csv")
    pub = pub[pub["rule"] == "argmax"].set_index("config")
    pop = pd.read_csv(S4_POP).set_index("config")
    popm = pd.read_csv(S4_POP_MAXFUN).set_index("config")
    for code, name in CONFIGS.items():
        g = pd.concat([pd.read_csv(f) for f in
                       sorted((S4_CV / "cv" / code).glob("outer*.csv"))],
                      ignore_index=True).sort_values(["PATNO", "visit_index"],
                                                     kind="mergesort")
        y = g["to_state"].to_numpy() - 1      # run codes 1, 2, 3 = TD, Ind, PIGD
        P = g[["p_TD", "p_Indeterminate", "p_PIGD"]].to_numpy(dtype=float)
        if abs(np.mean(P.argmax(1) == y) - pub.loc[code, "accuracy"]) > 1e-9:
            raise RuntimeError(f"{code}: published three-class accuracy does not reproduce")
        keep = y != 1
        pp = P[keep, 2] / (P[keep, 0] + P[keep, 2])
        arm = rc.BinaryArm(
            group="CT-HMM configuration study", model=name,
            protocol="10-fold patient cross-validation",
            cohort="3,029 transitions", source=rel(S4_CV / "cv" / code),
            pos=y[keep] == 2, score=pp, called=pp >= 0.5,
            patients=g["PATNO"].to_numpy()[keep])
        r = pd.Series(arm.row(cfg.n_boot))
        src, p = S4_POP, pop.loc[code]
        if code in popm.index and popm.loc[code, "best_loglik"] > p["best_loglik"]:
            src, p = S4_POP_MAXFUN, popm.loc[code]
        ll, k = float(p["best_loglik"]), int(p["n_params"])
        aic, bic = -2 * ll + 2 * k, -2 * ll + k * np.log(N_VISITS)
        if abs(aic - p["AIC"]) > 1e-6 or abs(bic - p["BIC_n_obs"]) > 1e-6:
            raise RuntimeError(f"{code}: stored AIC/BIC do not reproduce")
        share = float(p["share_within_0.01"])
        put("CT-HMM configurations", name, r, aic, bic, L,
            f"whole-cohort fit, {int(round(share * 40))} of 40 starts reached the best optimum",
            extra={"loglik": ll, "n_params": k,
                   "three_class_accuracy": float(pub.loc[code, "accuracy"])})
        print(f"{name:<30} acc {r['accuracy']:.4f} auc {r['auc_roc']:.4f} AIC {aic:,.2f}")

    b = pd.read_csv(FINAL / "binary_comparison.csv")

    def pick(group, model):
        s = b[(b["group"] == group) & (b["model"] == model)]
        if len(s) != 1:
            raise KeyError(f"{group} / {model}")
        return s.iloc[0]

    # --------------------------------------------- two-state family
    tic = pd.read_csv(TWO_IC).set_index("model")
    for label, model, grp in [
        ("Model 1 (2x2)", "CT-HMM (2 hidden x 2 observed, Indeterminate excluded)", T),
        ("Model 1 (2x2), no emission floor",
         "CT-HMM (2 hidden x 2 observed, no emission floor)", T),
        ("Model 2 (2x3)", "CT-HMM (2 hidden x 3 observed)", L),
        ("Existing (3x3)", "CT-HMM (3 hidden x 3 observed, existing model)", L),
    ]:
        i = tic.loc[model]
        put("Two-state CT-HMM", label, pick("E: two-state HMM family", model),
            float(i["aic"]), float(i["bic"]), grp, "whole-cohort fit",
            extra={"loglik": float(i["loglik"]), "n_params": int(i["n_params"])})

    # ------------------------------- ratio bands, fold-level training AIC/BIC
    fb = pd.read_csv(BAND / "fold_band_structures.csv")
    if int(fb[fb["main_width"].isna()]["n_cases"].sum()) != 3361:
        raise RuntimeError("held-out transition counts do not sum to 3,361")
    fb["n_train_visits"] = N_VISITS - (fb["n_cases"] + 1)
    fb["bic"] = -2 * fb["train_loglik"] + fb["n_params"] * np.log(fb["n_train_visits"])
    if not np.allclose(fb["aic"], -2 * fb["train_loglik"] + 2 * fb["n_params"]):
        raise RuntimeError("stored fold AIC does not reproduce")
    band_ic = []
    for w, g in fb.groupby(fb["main_width"].fillna(0.0)):
        band_ic.append({
            "width": "no bands" if w == 0 else f"{w:.2f}",
            "n_categories": int(g["n_categories"].iloc[0]),
            "n_params": int(g["n_params"].iloc[0]), "n_folds": len(g),
            "train_loglik_mean": g["train_loglik"].mean(),
            "aic_fold_mean": g["aic"].mean(), "aic_fold_min": g["aic"].min(),
            "aic_fold_max": g["aic"].max(), "bic_fold_mean": g["bic"].mean(),
            "bic_fold_min": g["bic"].min(), "bic_fold_max": g["bic"].max(),
            "all_converged": bool(g["converged"].all()),
            "source_path": rel(BAND / "fold_band_structures.csv")})
    s = json.load(open(BAND_DIAG))
    band_ic.append({
        "width": "0.10, whole cohort", "n_categories": s["n_categories"],
        "n_params": s["banded_params"], "n_folds": 1,
        "train_loglik_mean": s["banded_loglik"],
        "aic_fold_mean": -2 * s["banded_loglik"] + 2 * s["banded_params"],
        "bic_fold_mean": -2 * s["banded_loglik"] + s["banded_params"] * np.log(N_VISITS),
        "all_converged": True, "source_path": rel(BAND_DIAG)})
    bic_df = pd.DataFrame(band_ic)
    bic_df.to_csv(out / "band_aic_bic.csv", index=False)
    bidx = bic_df.set_index("width")
    D_ = "D: ratio-band sweep"
    for label, model in [("no bands", "CT-HMM (latent, interval-aware)")] + [
            (w, f"CT-HMM (subtype x ratio band, width {w})")
            for w in ("0.05", "0.10", "0.20", "0.30", "0.50")]:
        i = bidx.loc[label]
        put("CT-HMM with ratio bands", label, pick(D_, model),
            float(i["aic_fold_mean"]), float(i["bic_fold_mean"]),
            L if label == "no bands" else BG,
            "mean of 934 training fits inside leave-one-patient-out",
            extra={"loglik": float(i["train_loglik_mean"]),
                   "n_params": int(i["n_params"]),
                   "n_categories": int(i["n_categories"])})

    # ---------------------------------------------------- rich profile
    ica = pd.read_csv(IC)
    prof_ic = ica[ica["group"].str.startswith("A")].set_index("model")
    cont = pd.read_csv(PROFILE_CONT).set_index("model")
    C = "C: matched comparison set"
    for label, model in [
        ("Profile: level only", "Profile: level only (random walk)"),
        ("Profile: level + rate + medication", "Profile: level + rate + medication"),
        ("Profile: level + rate, no medication", "Profile: level + rate, no medication"),
        ("Last observation carried forward", "Last observation carried forward"),
        ("Population mean trajectory", "Population mean trajectory"),
    ]:
        c = cont.loc[model]
        fitted = model in prof_ic.index
        put("Rich clinical profile", label, pick(C, model),
            float(prof_ic.loc[model, "aic"]) if fitted else np.nan,
            float(prof_ic.loc[model, "bic"]) if fitted else np.nan,
            D if fitted else "",
            "whole-cohort fit" if fitted else "no fitted likelihood",
            extra={"log_density_per_obs": c["loglik_per_domain_observation"],
                   "rmse_std": c["rmse_standardised"], "mae_std": c["mae_standardised"],
                   "coverage_95": c["coverage_95"],
                   "loglik": float(prof_ic.loc[model, "loglik"]) if fitted else np.nan,
                   "n_params": int(prof_ic.loc[model, "k"]) if fitted else np.nan})

    # ------------------------------------- subtype-history baselines (LOPO)
    joint = ica[ica["group"].str.startswith("C")].set_index("model")
    B_ = "B: subtype-history models, LOPO"
    for label, model, ic_name in [
        ("Markov chain, discrete", "Markov (observed, discrete)", "Markov + initial state"),
        ("CTMC, interval-aware", "CTMC (observed, interval-aware)", "CTMC + initial state"),
        ("Direct logistic (history + interval)", "Direct logistic (history + interval)", None),
        ("Direct logistic (history only)", "Direct logistic (history only)", None),
        ("Persistence (last non-Indeterminate)", "Persistence (last non-Indeterminate)", None),
        ("Marginal (two-class prior)", "Marginal (two-class prior)", None),
    ]:
        put("Baselines and comparison models", label, pick(B_, model),
            float(joint.loc[ic_name, "aic"]) if ic_name else np.nan,
            float(joint.loc[ic_name, "bic"]) if ic_name else np.nan,
            L if ic_name else "",
            "whole-cohort fit with an initial-state term" if ic_name
            else "no comparable likelihood saved")

    # ------------------------------- harmonised family, common 2,759 subset
    F2 = "F2: harmonised, common binary subset"
    for model in ["Subscore GP, independent", "Subscore GP, joint",
                  "Joint subscore LMM", "Time-varying LMM, spline",
                  "Time-varying LMM, constant", "Patient mean subscore",
                  "Persistence (subscore LOCF)", "Population mean subscore"]:
        put("Subscore models (harmonised)", model, pick(F2, model),
            aic_note="no whole-cohort likelihood saved")

    t = pd.DataFrame(rows)
    t.to_csv(out / "all_model_scores.csv", index=False)
    print(f"wrote {len(t)} rows to {rel(out)}")


if __name__ == "__main__":
    main()
