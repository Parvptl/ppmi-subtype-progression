"""PLANNING VERIFICATION for the proposed Rev 3 recovery test (docs 44-47). Not a recovery test.

Read-only on saved outputs (Step 6, D4 full-chain covariances, D7 transformation code) plus the real design.
No model is fitted and no synthetic PPMI data are generated. The Monte Carlo draws only standard-normal
z-vectors (seed [42, 50, 0], unused elsewhere) to evaluate the proposed DECISION RULE under stated scenarios.

  1. Convention-free functionals (15 ψ-only + 2 item-anchored): exact invariance under the joint relabelling
     (numerically), evaluability, Jacobians, z on the five development datasets (descriptive, NOT a result).
  2. Gate correlation structure under the primary (full-chain) covariance, propagated through S and the
     functionals; rank.
  3. Operating characteristics of C1 (coverage >= 88/100) and C2 (|mean z| <= 0.30) for the 19 S-parameters
     and C3/C4 for the 17 functionals: exact per-gate values and Monte Carlo joint values with dependence,
     against the independence approximation, under calibrated, under-coverage and bias scenarios.

    python scripts/56_m2_rev3_planning_checks.py --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppmi_subtype import grm as G  # noqa: E402
from ppmi_subtype import m2_grm as M  # noqa: E402

S6 = ROOT / "experiments/20261018T222857Z_stage2_step6_full_pipeline_recovery/step6"
DIAG = ROOT / "experiments/20261019T084559Z_stage2_step6_diagnostics"
R, COV_MIN, BIAS_MAX, NTRIAL = 100, 88, 0.30, 20000
KT, KP = 11, 5
NT, NP = 5 * KT, 5 * KP


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D7 = load_module("d7_plan", ROOT / "scripts/55_m2_scale_convention_readonly.py")

# b = (intercept_T, slope_T, intercept_P, slope_P); Σε = (T, P)
FNAMES = ([f"{q}_{s}" for s in ("T", "P") for q in ("slope_over_sdint", "treat_over_sdint", "half_log_var_slope_over_int",
                                                     "half_log_var_eps_over_int")]
          + [f"corrD_{i}{j}" for i in range(4) for j in range(i)] + ["corrEps_TP"]
          + ["item_anchored_location_T", "item_anchored_location_P"])


def functionals(psi, phiT, phiP):
    """The 17 convention-free functionals (exact definitions in doc 47 §6)."""
    beta, D, _, S, _ = M.unpack(psi)
    out = []
    for s, (i, j, e) in (("T", (0, 1, 0)), ("P", (2, 3, 1))):
        b1, g = (beta[1], beta[2]) if s == "T" else (beta[4], beta[5])
        out += [b1 / np.sqrt(D[i, i]), g / np.sqrt(D[i, i]), 0.5 * np.log(D[j, j] / D[i, i]), 0.5 * np.log(S[e, e] / D[i, i])]
    sd = np.sqrt(np.diag(D))
    out += [D[i, j] / (sd[i] * sd[j]) for i in range(4) for j in range(i)]
    out += [S[1, 0] / np.sqrt(S[0, 0] * S[1, 1])]
    for b0, phi, K in ((beta[0], phiT, KT), (beta[3], phiP, KP)):
        a, tau = G.unpack_items(phi, K)
        out += [float(np.mean(a * (b0 - tau.mean(axis=1))))]
    return np.array(out)


def jac(f, x, h=1e-6):
    x = np.asarray(x, float)
    f0 = f(x)
    Jm = np.zeros((f0.size, x.size))
    for k in range(x.size):
        e = np.zeros(x.size)
        e[k] = h
        Jm[:, k] = (f(x + e) - f(x - e)) / (2 * h)
    return Jm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    P = load_module("pilot_plan", ROOT / "scripts/45_m2_pilot.py")
    from ppmi_subtype.config import load_config
    _, t, x, yt, yp, *_ = P.load_cohort(load_config())
    groups = M.make_groups(t, x, yt, yp)
    psi_true, it_true, scale = P.truth_on_stage1_scale(groups)
    phiT0, phiP0 = G.pack_items(it_true.aT, it_true.tauT), G.pack_items(it_true.aP, it_true.tauP)
    f_true = functionals(psi_true, phiT0, phiP0)
    # truth functionals on the raw a-priori scale must equal those on the stage-1 scale (exact invariance)
    tl = P.TRUTH_ITEMS
    raw_tauT = tl["tauT1"][:, None] + tl["step"] * np.arange(4)
    raw_tauP = tl["tauP1"][:, None] + tl["step"] * np.arange(4)
    f_raw = functionals(np.array(scale["psi_raw"]), G.pack_items(tl["aT"], raw_tauT), G.pack_items(tl["aP"], raw_tauP))
    res = {"label": "PLANNING VERIFICATION — not a recovery test; Step 6 remains FAILED; D1 remains NOT REPRODUCED",
           "functional_names": FNAMES, "truth_functionals": f_true,
           "check_truth_raw_vs_stage1_scale_max": float(np.max(np.abs(f_raw - f_true))), "datasets": []}

    names = list(M.NAMES)
    Rgate = []
    for r in range(5):
        d6 = json.loads((S6 / f"S1_dataset{r}.json").read_text())
        d4 = json.loads((DIAG / f"d4/D4_dataset{r}.json").read_text())
        psi2 = np.array(d6["stage2_fit_2"]["psi"])
        it2 = {s: (np.array(d6["items_2"][s]["a"]), np.array(d6["items_2"][s]["tau"])) for s in ("T", "P")}
        phiT, phiP = G.pack_items(*it2["T"]), G.pack_items(*it2["P"])
        V = np.array(d4["V_full"])
        idx = np.r_[np.arange(99, 179), np.arange(179, 198)]           # (φ₂, ψ₂) block of the full chain
        Vpp = V[np.ix_(idx, idx)]
        # transformation and its Jacobian (w.r.t. ψ₂ only)
        sps, mom = D7.standardise(groups, psi2)
        Gs = D7.jacobian(groups, psi2)
        Jg = np.zeros((19, 99))
        Jg[:, 80:] = Gs
        # functionals and invariance under the joint relabelling with the same (c, d)
        f_raw_est = functionals(psi2, phiT, phiP)
        cT, cP, dT, dP = mom["c_T"], mom["c_P"], mom["d_T"], mom["d_P"]
        f_rel = functionals(sps, G.pack_items(it2["T"][0] * cT, (it2["T"][1] - dT) / cT),
                            G.pack_items(it2["P"][0] * cP, (it2["P"][1] - dP) / cP))
        # random relabelling as well (any (c, d) must leave the functionals unchanged)
        cc = (1.13, -0.07, 0.91, 0.05)
        f_rel2 = functionals(D7.R53.relabel_psi(psi2, *cc), G.pack_items(it2["T"][0] * cc[0], (it2["T"][1] - cc[1]) / cc[0]),
                             G.pack_items(it2["P"][0] * cc[2], (it2["P"][1] - cc[3]) / cc[2]))
        Jf = jac(lambda v: functionals(v[80:], v[:NT], v[NT:80]), np.r_[phiT, phiP, psi2])
        Jall = np.vstack([Jg, Jf])                                      # 36 × 99
        Vg = Jall @ Vpp @ Jall.T
        se_g = np.sqrt(np.diag(Vg))
        est = np.r_[sps, f_raw_est]
        tru = np.r_[psi_true, f_true]
        z = (est - tru) / se_g
        ev = np.linalg.eigvalsh(Vg / np.outer(se_g, se_g))
        Rgate.append(Vg / np.outer(se_g, se_g))
        res["datasets"].append({"dataset": r, "invariance_max_S_cd": float(np.max(np.abs(f_rel - f_raw_est))),
                                "invariance_max_arbitrary_cd": float(np.max(np.abs(f_rel2 - f_raw_est))),
                                "z_S": z[:19], "z_F": z[19:], "se_F": se_g[19:], "rank_tol1e-8": int(np.sum(ev > 1e-8)),
                                "rank_tol1e-6": int(np.sum(ev > 1e-6)), "min_se_ratio": float(np.min(se_g / np.abs(tru + 1e-12)))})
    Rm = np.mean(Rgate, axis=0)
    gnames = names + FNAMES
    np.save(out / "gate_correlation_mean.npy", Rm)
    ev, evec = np.linalg.eigh(Rm)
    ev = np.clip(ev, 0, None)
    res["gate_corr_eigs_top"] = sorted(ev.tolist(), reverse=True)[:20]
    res["gate_corr_effective_rank_gt_1e-6"] = int(np.sum(ev > 1e-6))
    Lf = evec * np.sqrt(ev)                                             # Rm = Lf Lfᵀ
    zS = np.array([d["z_S"] for d in res["datasets"]])
    zF = np.array([d["z_F"] for d in res["datasets"]])
    res["dev_mean_z"] = dict(zip(gnames, np.r_[zS.mean(0), zF.mean(0)].round(3).tolist()))
    # strongest correlations among gates (|r| > 0.9) — redundancy
    red = [(gnames[i], gnames[j], round(float(Rm[i, j]), 3)) for i in range(36) for j in range(i) if abs(Rm[i, j]) > 0.9]
    res["gate_pairs_abs_corr_gt_0.9"] = red

    # ---------------- operating characteristics ---------------------------------------------------------
    exact = {"cov_pass_per_gate_calibrated": float(binom.sf(COV_MIN - 1, R, 0.95)),
             "bias_pass_per_gate_calibrated": float(1 - 2 * norm.sf(BIAS_MAX * np.sqrt(R))),
             "coverage_under_bias_b": {b: float(norm.cdf(1.96 - b) - norm.cdf(-1.96 - b)) for b in (0.1, 0.2, 0.3, 0.5, 0.6)},
             "coverage_under_sd_k": {k: float(2 * norm.cdf(1.96 / k) - 1) for k in (1.1, 1.2, 1.3)},
             "bias_pass_given_b": {b: float(norm.cdf((BIAS_MAX - b) * 10) - norm.cdf((-BIAS_MAX - b) * 10)) for b in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)}}
    res["exact"] = exact
    rng = np.random.default_rng([42, 50, 0])
    iLe00 = names.index("Le00_log")
    dev_mean = np.r_[zS.mean(0), zF.mean(0)]
    scen = {"S0_calibrated": (np.zeros(36), 1.0),
            "S1_se_understated_10pct": (np.zeros(36), 1.1),
            "S2_se_understated_20pct": (np.zeros(36), 1.2),
            "S3a_Le00_bias_0.3": (np.eye(36)[iLe00] * 0.3, 1.0),
            "S3b_Le00_bias_0.5": (np.eye(36)[iLe00] * 0.5, 1.0),
            "S3c_Le00_bias_0.61": (np.eye(36)[iLe00] * 0.61, 1.0),
            "S4_all_gates_bias_0.15": (np.full(36, 0.15), 1.0),
            "S5_dev_means_half": (0.5 * dev_mean, 1.0),
            "S6_dev_means_full": (dev_mean, 1.0)}
    oc = {}
    for nm, (mu, k) in scen.items():
        passS = passF = passAll = 0
        per = np.zeros(36)
        c1 = c2 = 0
        for chunk in range(NTRIAL // 500):
            u = rng.standard_normal((500, R, 36))
            zz = k * (u @ Lf.T) + mu                                     # (trials, R, gates)
            cov = (np.abs(zz) <= 1.96).sum(axis=1) >= COV_MIN
            bias = np.abs(zz.mean(axis=1)) <= BIAS_MAX
            g = cov & bias
            per += g.sum(0)
            passS += np.sum(g[:, :19].all(1))
            passF += np.sum(g[:, 19:].all(1))
            passAll += np.sum(g.all(1))
            c1 += np.sum(cov[:, :19].all(1))
            c2 += np.sum(bias[:, :19].all(1))
        per /= NTRIAL
        indep = float(np.prod(per))
        oc[nm] = {"P_pass_all_36": passAll / NTRIAL, "P_pass_C1C2_19": passS / NTRIAL, "P_pass_C3C4_17": passF / NTRIAL,
                  "P_C1_all19": c1 / NTRIAL, "P_C2_all19": c2 / NTRIAL, "indep_approx_all_36": indep,
                  "mc_se": float(np.sqrt(max(passAll / NTRIAL * (1 - passAll / NTRIAL), 1e-12) / NTRIAL)),
                  "lowest_gate_pass": sorted(zip(per.round(4).tolist(), gnames))[:4]}
    res["oc_mc"] = oc
    res["mc"] = {"seed": [42, 50, 0], "trials": NTRIAL, "R": R, "gate_correlation": "mean over 5 development datasets of "
                 "corr([G; J_F] V_(φ2,ψ2) [G; J_F]ᵀ) from the D4 full-chain V"}
    res["inputs_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in [S6 / f"S1_dataset{r}.json" for r in range(5)]
                            + [DIAG / f"d4/D4_dataset{r}.json" for r in range(5)]}
    res["script_sha256"] = sha(Path(__file__))

    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=conv))
    files = sorted(p for p in out.rglob("*") if p.is_file())
    (out / "MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files))
    print("written", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
