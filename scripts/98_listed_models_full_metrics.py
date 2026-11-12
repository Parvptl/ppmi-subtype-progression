"""Full metric table for the models shared with the supervisor.

Adds what scripts/96 did not cover: the three CT-HMM floor variants from the
10-fold HMM study, whose source run published three-class accuracy only. Their
binary TD-vs-PIGD metrics are computed here from that run's saved per-case
predictions, with the same conventions as scripts/96 (PIGD positive,
probabilities renormalised over TD + PIGD, patient-level bootstrap). Every
other row is read from the final consolidated run, and AIC/BIC are read from
the whole-cohort fits that produced them; nothing is refitted.

Usage:
    python scripts/98_listed_models_full_metrics.py --out experiments/<run>
"""

from __future__ import annotations

import argparse
import importlib.util
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
BAND_DIAG = E / "20261104T155439Z_ratio_band_emission_diagnostic" / "summary.json"

FLOOR = {
    "C3_f50": "CT-HMM, 3 states, floor 0.5",
    "C5_free": "CT-HMM, 5 states, no floor",
    "C3_free": "CT-HMM, 3 states, no floor",
}
N_LABELLED_VISITS = 4295


def load_engine():
    spec = importlib.util.spec_from_file_location(
        "recov", ROOT / "scripts" / "96_final_metric_recovery.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recov"] = mod
    spec.loader.exec_module(mod)
    return mod


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    cfg = ap.parse_args()
    out = Path(cfg.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rc = load_engine()

    # ---------------------------------------------- floor variants, binary
    pub = pd.read_csv(S4_CV / "analysis" / "cv_config_metrics.csv")
    pub = pub[pub["rule"] == "argmax"].set_index("config")
    rows = []
    for code, name in FLOOR.items():
        g = pd.concat(
            [pd.read_csv(f) for f in sorted((S4_CV / "cv" / code).glob("outer*.csv"))],
            ignore_index=True,
        ).sort_values(["PATNO", "visit_index"], kind="mergesort")
        if g.duplicated(["PATNO", "visit_index"]).any():
            raise RuntimeError(f"{code}: duplicate predictions")
        # the run codes states 1, 2, 3 = TD, Indeterminate, PIGD; confirmed by
        # the class supports 2,115 / 332 / 914 and by the published accuracy
        y = g["to_state"].to_numpy() - 1
        P = g[["p_TD", "p_Indeterminate", "p_PIGD"]].to_numpy(dtype=float)
        acc3 = float(np.mean(P.argmax(axis=1) == y))
        if abs(acc3 - float(pub.loc[code, "accuracy"])) > 1e-9:
            raise RuntimeError(f"{code}: three-class accuracy does not reproduce")
        keep = y != 1
        p_pigd = P[keep, 2] / (P[keep, 0] + P[keep, 2])
        arm = rc.BinaryArm(
            group="H: 10-fold HMM configuration study",
            model=name,
            protocol="10-fold patient cross-validation (outer folds of the nested study)",
            cohort="3,029 transitions with a TD/PIGD target",
            source=rel(S4_CV / "cv" / code),
            pos=(y[keep] == 2),
            score=p_pigd,
            called=p_pigd >= 0.5,
            patients=g["PATNO"].to_numpy()[keep],
            notes="binary view derived from saved three-class probabilities; "
            f"three-class accuracy {acc3:.6f} reproduces the published value",
        )
        r = arm.row(cfg.n_boot)
        r["config"] = code
        r["three_class_accuracy_published"] = float(pub.loc[code, "accuracy"])
        rows.append(r)
        print(f"{name:<30} acc {r['accuracy']:.4f} sens {r['sensitivity_PIGD']:.4f} "
              f"spec {r['specificity_TD']:.4f} auc {r['auc_roc']:.4f}")
    floor = pd.DataFrame(rows)
    floor.to_csv(out / "floor_variants_binary.csv", index=False)

    # ------------------------------------------ whole-cohort information criteria
    pop = pd.read_csv(S4_POP).set_index("config")
    popm = pd.read_csv(S4_POP_MAXFUN).set_index("config")
    ic_rows = []
    for code, name in FLOOR.items():
        src, r = S4_POP, pop.loc[code]
        # a longer optimiser budget found a marginally higher optimum for the
        # free fits; the higher log-likelihood is the better estimate of the max
        if code in popm.index and popm.loc[code, "best_loglik"] > r["best_loglik"]:
            src, r = S4_POP_MAXFUN, popm.loc[code]
        ll, k = float(r["best_loglik"]), int(r["n_params"])
        ic_rows.append({
            "model": name, "loglik": ll, "n_params": k,
            "n_observations": N_LABELLED_VISITS,
            "aic": -2 * ll + 2 * k, "bic": -2 * ll + k * np.log(N_LABELLED_VISITS),
            "aic_stored": float(r["AIC"]), "bic_stored": float(r["BIC_n_obs"]),
            "share_starts_within_0.01": float(r["share_within_0.01"]),
            "comparable_group": "label sequences, all 4,295 labelled visits",
            "source_path": rel(src),
        })
    ic = pd.DataFrame(ic_rows)
    if not ((ic["aic"] - ic["aic_stored"]).abs() < 1e-6).all():
        raise RuntimeError("stored AIC does not reproduce -2logL + 2k")
    if not ((ic["bic"] - ic["bic_stored"]).abs() < 1e-6).all():
        raise RuntimeError("stored BIC does not reproduce with n = 4,295")
    ic.to_csv(out / "floor_variants_aic_bic.csv", index=False)
    print("floor-variant AIC/BIC reproduce -2logL + 2k and k log(4295) exactly")

    # ------------------------------------------------- the combined table
    b = pd.read_csv(FINAL / "binary_comparison.csv")
    ica = pd.read_csv(IC).set_index("model")
    ict = pd.read_csv(TWO_IC).set_index("model")

    def pick(group, model):
        s = b[(b["group"] == group) & (b["model"] == model)]
        if len(s) != 1:
            raise KeyError(f"{group} / {model}")
        return s.iloc[0]

    listed = []

    def add(block, label, r, aic, bic, ic_group, note=""):
        listed.append({
            "block": block, "model": label,
            "n_cases": int(r["n_scored"]), "n_patients": int(r["n_patients"]),
            "evaluation_protocol": r["evaluation_protocol"],
            "accuracy": r["accuracy"],
            "accuracy_ci_low": r["accuracy_ci_low"], "accuracy_ci_high": r["accuracy_ci_high"],
            "sensitivity_PIGD": r["sensitivity_PIGD"],
            "sensitivity_ci_low": r["sensitivity_PIGD_ci_low"],
            "sensitivity_ci_high": r["sensitivity_PIGD_ci_high"],
            "specificity_TD": r["specificity_TD"],
            "specificity_ci_low": r["specificity_TD_ci_low"],
            "specificity_ci_high": r["specificity_TD_ci_high"],
            "auc_roc": r["auc_roc"],
            "auc_ci_low": r["auc_roc_ci_low"], "auc_ci_high": r["auc_roc_ci_high"],
            "aic": aic, "bic": bic, "aic_bic_comparable_group": ic_group,
            "source_path": r["source_path"], "notes": note,
        })

    L = "label sequences, all 4,295 labelled visits"
    for code, name in FLOOR.items():
        r = floor[floor["config"] == code].iloc[0]
        i = ic[ic["model"] == name].iloc[0]
        add("CT-HMM", name, r, i["aic"], i["bic"], L,
            "5-state fit is multimodal: only 1 of 40 starts reached its best optimum"
            if code == "C5_free" else "")

    C = "C: matched comparison set"
    D9 = "density over 9 continuous domains, 38,486 observations"
    for label, model in [
        ("Profile: level only", "Profile: level only (random walk)"),
        ("Profile: level + rate + medication", "Profile: level + rate + medication"),
        ("Profile: level + rate, no medication", "Profile: level + rate, no medication"),
    ]:
        add("Rich profile", label, pick(C, model),
            float(ica.loc[model, "aic"]), float(ica.loc[model, "bic"]), D9)

    Eg = "E: two-state HMM family"
    for label, model, grp in [
        ("Model 1 (2x2)", "CT-HMM (2 hidden x 2 observed, Indeterminate excluded)",
         "label sequences, 3,812 TD/PIGD visits only"),
        ("Model 2 (2x3)", "CT-HMM (2 hidden x 3 observed)", L),
        ("Existing (3x3)", "CT-HMM (3 hidden x 3 observed, existing model)", L),
    ]:
        add("Two-state CT-HMM", label, pick(Eg, model),
            float(ict.loc[model, "aic"]), float(ict.loc[model, "bic"]), grp)

    D = "D: ratio-band sweep"
    unb = pick(D, "CT-HMM (latent, interval-aware)")
    # the 3-state CT-HMM is listed in two comparison groups with identical values
    ct3 = ica.loc[["CT-HMM (3 latent states)"]]
    if ct3[["aic", "bic"]].nunique().max() != 1:
        raise RuntimeError("the 3-state CT-HMM's two AIC/BIC rows disagree")
    add("CT-HMM with ratio bands", "no bands", unb,
        float(ct3["aic"].iloc[0]), float(ct3["bic"].iloc[0]), L,
        "the unbanded model is the 3-state floor-0.5 CT-HMM; whole-cohort fit")
    for w in ("0.05", "0.10", "0.20", "0.30", "0.50"):
        add("CT-HMM with ratio bands", w,
            pick(D, f"CT-HMM (subtype x ratio band, width {w})"),
            np.nan, np.nan, "none",
            "no comparable AIC/BIC: each width's likelihood is over its own "
            "joint (subtype x band) alphabet, so it cannot be compared with the "
            "unbanded model or with another width")
    table = pd.DataFrame(listed)
    table.to_csv(out / "listed_models_full_metrics.csv", index=False)
    print(f"wrote {len(table)} rows to {rel(out)}")


if __name__ == "__main__":
    main()
