"""EXPLORATORY DIAGNOSTIC D7 (read-only): re-express saved estimates on the truth's scale convention.

Plan: docs/cluster_phase/41_STAGE2_NEXT_STEP_D7_SCALE_CONVENTION_PLAN_PROPOSED_20261019.md (approved as a
read-only diagnostic). Not a repair, an estimator or a recovery test; Step 6 remains FAILED (doc 38) and
D1 remains NOT REPRODUCED.

For an estimate ψ, the pooled visit-level latent moments over the real design (the `truth_on_stage1_scale`
formula, computed from ψ itself — never from the truth):
    μ_s(ψ) = mean_v M^s_v(ψ),   v_s(ψ) = mean_v K^s_vv(ψ) + var_v M^s_v(ψ)
and  S(ψ) = relabel_psi(ψ, c_T = √v_T, d_T = μ_T, c_P = √v_P, d_P = μ_P)  (script 53).
S is invariant under any joint latent relabelling, idempotent, and leaves ψ_true unchanged.
Uncertainty: V_S = G V Gᵀ with G = ∂S/∂ψ (central differences, h = 1e-6), V = H⁻¹ (saved fixed-item Hessian)
or, for ψ̂(2), the D4 full-chain sandwich. Inputs: saved Step 6, D4, D5, D6 outputs (hash-verified).
No fit, no simulation, no random numbers; the real design (times, treatment) only.

    python scripts/55_m2_scale_convention_readonly.py --diag <diag dir>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppmi_subtype import m2_grm as M  # noqa: E402

S6 = ROOT / "experiments/20261018T222857Z_stage2_step6_full_pipeline_recovery/step6"
FAILED = ("beta0_P", "Le00_log", "Le11_log")
LABEL = ("EXPLORATORY DIAGNOSTIC D7 — read-only; not a recovery test; Step 6 remains FAILED (doc 38); "
         "D1 remains NOT REPRODUCED")
H_JAC = 1e-6


def sha(p, kind="sha256"):
    h = hashlib.new(kind)
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R53 = load_module("r53_d7", ROOT / "scripts/53_m2_step6_diag_readonly.py")


def pooled_moments(groups, psi):
    """(μ_T, μ_P), (v_T, v_P): the truth_on_stage1_scale formula evaluated at ψ."""
    beta, D, _, S, _ = M.unpack(psi)
    means, vars_ = {0: [], 1: []}, {0: [], 1: []}
    for n, g in groups.items():
        Mn = np.einsum("k,kmi->mi", beta, M.mean_design(g.T, g.X))
        K, _ = M.covariance(g.T, D, S)
        dg = np.einsum("mii->mi", K)
        for s_, sl in ((0, slice(0, n)), (1, slice(n, 2 * n))):
            means[s_].append(Mn[:, sl].ravel())
            vars_[s_].append(dg[:, sl].ravel())
    mu = np.array([np.concatenate(means[s_]).mean() for s_ in (0, 1)])
    tot = np.array([np.concatenate(vars_[s_]).mean() + np.concatenate(means[s_]).var() for s_ in (0, 1)])
    return mu, tot


def standardise(groups, psi):
    mu, v = pooled_moments(groups, psi)
    c = np.sqrt(v)
    return R53.relabel_psi(psi, c[0], mu[0], c[1], mu[1]), {"mu_T": mu[0], "mu_P": mu[1], "v_T": v[0], "v_P": v[1],
                                                           "c_T": c[0], "c_P": c[1], "d_T": mu[0], "d_P": mu[1]}


def jacobian(groups, psi, h=H_JAC):
    psi = np.asarray(psi, float)
    G = np.zeros((psi.size, psi.size))
    for k in range(psi.size):
        e = np.zeros(psi.size)
        e[k] = h
        G[:, k] = (standardise(groups, psi + e)[0] - standardise(groups, psi - e)[0]) / (2 * h)
    return G


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    a = ap.parse_args()
    diag = Path(a.diag)
    out = diag / "d7_readonly"
    out.mkdir(exist_ok=False)
    base = json.loads((diag / "00_baseline.json").read_text())

    # ---- integrity: frozen locks, inputs vs baseline and manifests --------------------------------
    I = {}
    locks = sorted((ROOT / "docs/cluster_phase/_frozen").glob("*.sha256"))
    I["frozen_locks"] = all(sha(ROOT / ln.split()[1]) == ln.split()[0] for f in locks
                            for ln in f.read_text().splitlines() if ln.strip())
    I["step6_outputs_vs_baseline"] = all(sha(ROOT / p) == h for p, h in base["prior_outputs_sha256"].items()
                                         if p.startswith("experiments/20261018T222857Z"))
    man = {}
    for d in ("readonly", "d4", "d5", "d6"):
        lines = (diag / d / "MANIFEST.sha256").read_text().splitlines()
        man[d] = all(sha(diag / d / ln.split()[1]) == ln.split()[0] for ln in lines)
    I["diag_manifests"] = man
    I["code_vs_baseline"] = all(sha(ROOT / f) == base["code_sha256"][f] for f in
                                ("src/ppmi_subtype/m2_grm.py", "scripts/45_m2_pilot.py"))
    I["script53_vs_readonly_run"] = sha(ROOT / "scripts/53_m2_step6_diag_readonly.py") == json.loads(
        (diag / "readonly/summary.json").read_text())["provenance"]["script_sha256"]
    if not (I["frozen_locks"] and I["step6_outputs_vs_baseline"] and all(man.values()) and I["code_vs_baseline"]
            and I["script53_vs_readonly_run"]):
        (out / "STOPPED.json").write_text(json.dumps({"integrity": I}, indent=1))
        print("integrity failed", I)
        return 2

    # ---- design (no responses) and truth (for comparison and the fixed-point check only) ----------
    P = load_module("m2_pilot_d7", ROOT / "scripts/45_m2_pilot.py")
    from ppmi_subtype.config import load_config
    _, times, treat, YTd, YPd, *_ = P.load_cohort(load_config())
    groups = M.make_groups(times, treat, YTd, YPd)
    psi_true, _, scale = P.truth_on_stage1_scale(groups)
    s6 = json.loads((S6 / "summary.json").read_text())
    names = s6["names"]
    I["truth_equals_step6_record"] = bool(np.array_equal(psi_true, np.array(s6["psi_true"])))
    st_true, mom_true = standardise(groups, psi_true)
    st_raw, _ = standardise(groups, np.array(scale["psi_raw"]))
    checks = {"S_truth_minus_truth_max": float(np.max(np.abs(st_true - psi_true))),
              "S_rawprior_minus_truth_max": float(np.max(np.abs(st_raw - psi_true))),
              "truth_moments": mom_true}

    # ---- collect saved estimates ------------------------------------------------------------------
    est = {r: [] for r in range(5)}                     # (label, k, psi, V or None)
    for r in range(5):
        d6r = json.loads((S6 / f"S1_dataset{r}.json").read_text())
        H1, H2 = np.array(d6r["H1_nonbinding"]), np.array(d6r["H2"])
        est[r].append(("psi1_step6", 1, np.array(d6r["stage2_fit_1"]["psi"]), np.linalg.inv(H1)))
        est[r].append(("psi2_step6", 2, np.array(d6r["stage2_fit_2"]["psi"]), np.linalg.inv(H2)))
        d5r = json.loads((diag / f"d5/D5_dataset{r}.json").read_text())
        for h in d5r["history"][2:]:
            V = np.linalg.inv(np.array(d5r["H_final"])) if h["k"] == d5r["history"][-1]["k"] else None
            est[r].append((f"d5_k{h['k']}", h["k"], np.array(h["psi"]), V))
        d6d = json.loads((diag / f"d6/D6_dataset{r}.json").read_text())
        est[r].append(("d6_oracle_items", None, np.array(d6d["fit"]["psi"]), np.linalg.inv(np.array(d6d["H"]))))
        d4r = json.loads((diag / f"d4/D4_dataset{r}.json").read_text())
        Vf = np.array(d4r["V_full"])
        est[r].append(("psi2_step6_fullchainV", 2, np.array(d6r["stage2_fit_2"]["psi"]), Vf[-19:, -19:]))

    # ---- transform, Jacobians, z --------------------------------------------------------------------
    rows, trans, jchecks = [], [], []
    se_ref = {}
    for r in range(5):
        for label, k, psi, V in est[r]:
            sps, mom = standardise(groups, psi)
            idem = float(np.max(np.abs(standardise(groups, sps)[0] - sps)))
            trans.append({"dataset": r, "estimate": label, "k": k, **mom, "idempotence_max": idem})
            se_raw = np.sqrt(np.diag(V)) if V is not None else np.full(len(names), np.nan)
            if V is not None:
                G = jacobian(groups, psi)
                se_s = np.sqrt(np.diag(G @ V @ G.T))
                # analytic special case: S is invariant under relabelling ⇒ G·v_relabel = 0
                for sub, par in (("T", "c"), ("T", "d"), ("P", "c"), ("P", "d")):
                    e = 1e-6
                    args = lambda x: ((1 + x, 0.0, 1.0, 0.0) if (sub, par) == ("T", "c") else  # noqa: E731
                                      (1.0, x, 1.0, 0.0) if (sub, par) == ("T", "d") else
                                      (1.0, 0.0, 1 + x, 0.0) if (sub, par) == ("P", "c") else (1.0, 0.0, 1.0, x))
                    v = (R53.relabel_psi(psi, *args(e)) - R53.relabel_psi(psi, *args(-e))) / (2 * e)
                    jchecks.append({"dataset": r, "estimate": label, "direction": f"{sub}_{par}",
                                    "G_v_rel": float(np.linalg.norm(G @ v) / max(np.linalg.norm(v), 1e-300))})
            else:
                se_s = np.full(len(names), np.nan)
            if label == "psi2_step6":
                se_ref[r] = se_s
            for j, nm in enumerate(names):
                rows.append({"dataset": r, "estimate": label, "k": k, "param": nm, "truth": psi_true[j],
                             "raw": psi[j], "raw_se": se_raw[j], "raw_z": (psi[j] - psi_true[j]) / se_raw[j],
                             "std": sps[j], "std_se": se_s[j], "std_z": (sps[j] - psi_true[j]) / se_s[j]})
    L = pd.DataFrame(rows)
    # errors of every estimate in the SE_S(ψ̂(2)) units of its dataset (for the D5 iterates without Hessians)
    L["std_err_in_seS2"] = [(row.std - row.truth) / se_ref[row.dataset][names.index(row.param)] for row in L.itertuples()]
    L["raw_err_in_seS2"] = [(row.raw - row.truth) / se_ref[row.dataset][names.index(row.param)] for row in L.itertuples()]
    T = pd.DataFrame(trans)
    J = pd.DataFrame(jchecks)
    L.to_csv(out / "D7_estimates_long.csv", index=False)
    T.to_csv(out / "D7_transformations.csv", index=False)
    J.to_csv(out / "D7_jacobian_invariance_checks.csv", index=False)

    # ---- prediction (a): D5 collapse --------------------------------------------------------------
    raw_se2 = {r: np.sqrt(np.diag(np.linalg.inv(np.array(json.loads((S6 / f"S1_dataset{r}.json").read_text())["H2"]))))
               for r in range(5)}
    col = []
    for r in range(5):
        traj = L[(L.dataset == r) & (L.estimate.isin(["psi2_step6"] + [f"d5_k{k}" for k in range(3, 7)]))]
        for nm in names:
            x = traj[traj.param == nm]
            col.append({"dataset": r, "param": nm,
                        "raw_range_over_rawSE2": (x.raw.max() - x.raw.min()) / raw_se2[r][names.index(nm)],
                        "std_range_over_seS2": (x["std"].max() - x["std"].min()) / se_ref[r][names.index(nm)],
                        "std_range_over_rawSE2": (x["std"].max() - x["std"].min()) / raw_se2[r][names.index(nm)]})
    C = pd.DataFrame(col)
    C.to_csv(out / "D7_d5_collapse.csv", index=False)
    collapse = C.groupby("dataset").agg(raw_max=("raw_range_over_rawSE2", "max"), std_max=("std_range_over_seS2", "max"),
                                        std_max_rawSE=("std_range_over_rawSE2", "max")).reset_index()

    # ---- prediction (b): bias of β0_P and Le11_log (and Le00_log) ---------------------------------
    summ = []
    for label in ("psi1_step6", "psi2_step6", "psi2_step6_fullchainV", "d5_k6", "d6_oracle_items"):
        for nm in names:
            x = L[(L.estimate == label) & (L.param == nm)].sort_values("dataset")
            summ.append({"estimate": label, "param": nm, "raw_mean_z": x.raw_z.mean(), "std_mean_z": x.std_z.mean(),
                         "raw_covered": int(np.sum(np.abs(x.raw_z) <= 1.96)),
                         "std_covered": int(np.sum(np.abs(x.std_z) <= 1.96)),
                         "raw_mean_err": (x.raw - x.truth).mean(), "std_mean_err": (x["std"] - x.truth).mean()})
    B = pd.DataFrame(summ)
    B.to_csv(out / "D7_bias_summary.csv", index=False)

    summary = {"label": LABEL, "integrity": I, "checks": checks,
               "max_idempotence": float(T.idempotence_max.max()),
               "max_G_v_relabel": float(J.G_v_rel.max()),
               "collapse_by_dataset": collapse.to_dict(orient="records"),
               "failed_params": B[B.param.isin(FAILED)].round(4).to_dict(orient="records"),
               "notes": {"d5_k3_to_k5": "no Hessian was saved for D5 iterates 3-5; their errors are reported in the "
                                        "SE_S units of ψ̂(2) of the same dataset (column std_err_in_seS2) and enter "
                                        "only the collapse statistic",
                         "collapse_primary": "max over components of range_k S(ψ̂_k), k = 2..6, / SE_S(ψ̂(2)) (V = H2⁻¹)",
                         "prediction_a_caveat": "S is exactly invariant under joint relabelling; D5 showed 99.5–99.9 % of "
                                                "the drift is a relabelling, so collapse is largely implied and D7 measures "
                                                "the remainder",
                         "truth_use": "ψ_true is used only to compute errors/z and for the fixed-point check S(ψ_true) = ψ_true; "
                                      "it never enters the transformation"},
               "provenance": {"script_sha256": sha(Path(__file__)), "tests_sha256": sha(ROOT / "tests/test_d7_scale_convention.py"),
                              "inputs": {str(p.relative_to(ROOT)): sha(p) for p in
                                         [S6 / "summary.json"] + [S6 / f"S1_dataset{r}.json" for r in range(5)]
                                         + [diag / d / f"{d.upper()}_dataset{r}.json" for d in ("d4", "d5", "d6") for r in range(5)]},
                              "h_jacobian": H_JAC}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: float(o) if isinstance(o, np.floating)
                                                 else int(o) if isinstance(o, np.integer) else o.tolist()))
    files = sorted(p for p in out.rglob("*") if p.is_file())
    (out / "MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files))
    print("D7 written to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
