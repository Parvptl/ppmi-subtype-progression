"""Stage 2 M2 pilot: synthetic recovery, fit, convergence checks and label-free fidelity gates.

Frozen specification: docs/cluster_phase/16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.md
(B4-B6, B11, D3). The run refuses to start if the specification's SHA-256
differs from the frozen record.

LABEL-FREE: this script never calls the label rule, never reads label values
(the cohort uses only whether a visit is labelled, i.e. has all 16 valid
items), and computes no prevalence or transition statistic. No calibration,
positive control or verdict.

Order (doc 16 B11: a failure in steps 1-5 stops the run before the gates):
  0 provenance           1 synthetic recovery (a-priori truth, real design)
  2 item fits (3 starts) 3 longitudinal fit (3 starts) + Hessian
  4 Laplace accuracy     5 patient sampler (split-R̂, bulk ESS)
  6 fidelity gates G1-G7 (40 datasets); G4 decides whether target 2 is evaluable

    scripts/cluster/submit.sh stage2_m2_pilot --partition=normal --time=08:00:00 -- \
        scripts/45_m2_pilot.py --out @RUN_DIR@/results
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import grm as G  # noqa: E402
from ppmi_subtype import loaders  # noqa: E402
from ppmi_subtype import m2_gates as MG  # noqa: E402
from ppmi_subtype import m2_grm as M  # noqa: E402
from ppmi_subtype import raw_scores as RS  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

SPEC = "docs/cluster_phase/16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.md"
SPEC_LOCK = "docs/cluster_phase/_frozen/16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.sha256"
PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
CODE = ["scripts/45_m2_pilot.py", "src/ppmi_subtype/m2_grm.py", "src/ppmi_subtype/m2_gates.py",
        "src/ppmi_subtype/grm.py", "src/ppmi_subtype/raw_scores.py", "src/ppmi_subtype/subtype.py"]
N_RECOVERY, N_GATE_SIMS, N_CHECK = 5, 40, 30
SAMPLER = {"chains": 4, "warmup": 200, "keep": 1000}

# ---- a-priori recovery truth (fixed before any real-data fit; doc 16 D3) ------------------
TRUTH_ITEMS = {
    "aT": np.linspace(1.2, 2.4, 11), "tauT1": np.linspace(-0.5, 2.0, 11),
    "aP": np.linspace(1.5, 2.5, 5), "tauP1": np.linspace(0.0, 2.0, 5), "step": 0.8}
TRUTH_LONG = {"beta": [0.0, 0.05, 0.10, -0.20, 0.15, 0.20],
              "sd_b": [0.80, 0.08, 0.80, 0.12], "corr_b": {(0, 1): 0.1, (0, 2): 0.1, (2, 3): 0.1},
              "sd_e": [0.5, 0.5], "corr_e": 0.1}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(path, kind="sha256"):
    h = hashlib.new(kind)
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _json(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=_json))


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------
def load_cohort(cfg):
    """Analysed visits: labelled (all 16 items valid), patients with >= 2 such visits."""
    s = cfg.subtype
    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    table = RS.item_visit_table(loaders.load_updrs_part2(cfg), loaders.load_updrs_part3(cfg), s)
    frame = RS.build_visit_frame(panel, table, s)
    if not RS.validate_reconstruction(frame)["exact"]:
        raise RuntimeError("item reconstruction mismatch")
    itT, itP = list(s.tremor_items), list(s.pigd_items)
    items = table[RS.KEY].join(pd.DataFrame({c: pd.to_numeric(table[c], errors="coerce")
                                             for c in itT + itP}))
    v = frame.loc[frame.subtype_name.notna(), RS.KEY + ["years_from_baseline", s.treatment_column,
                                                        "tremor_sum", "pigd_sum"]]
    v = v.merge(items, on=RS.KEY, how="left", validate="one_to_one")
    v = v.sort_values(["PATNO", "years_from_baseline"]).reset_index(drop=True)
    v = v[v.PATNO.map(v.PATNO.value_counts()).ge(2)].reset_index(drop=True)
    Y = v[itT + itP].to_numpy(float)
    assert not np.isnan(Y).any() and ((Y >= 0) & (Y <= 4)).all()
    assert np.array_equal(v[itT].sum(1), v.tremor_sum) and np.array_equal(v[itP].sum(1), v.pigd_sum)
    x = (pd.to_numeric(v[s.treatment_column], errors="coerce") == 1).astype(float)
    times, treat, YT, YP = {}, {}, {}, {}
    for p, g in v.groupby("PATNO", sort=True):
        times[p] = g.years_from_baseline.to_numpy(float)
        treat[p] = x[g.index].to_numpy(float)
        YT[p] = g[itT].to_numpy(float)
        YP[p] = g[itP].to_numpy(float)
    return v, times, treat, YT, YP, itT, itP


def flatten(groups, YT=None, YP=None):
    """Visit-level arrays in group order; YT/YP default to the groups' own items."""
    outT, outP, pid, base = [], [], [], 0
    for n, g in groups.items():
        yt = g.YT if YT is None else YT[n]
        yp = g.YP if YP is None else YP[n]
        m = g.T.shape[0]
        outT.append(yt.reshape(m * n, -1))
        outP.append(yp.reshape(m * n, -1))
        pid.append(np.repeat(np.arange(base, base + m), n))
        base += m
    return np.concatenate(outT).astype(int), np.concatenate(outP).astype(int), np.concatenate(pid)


# ---------------------------------------------------------------------------
#  Fitting helpers
# ---------------------------------------------------------------------------
def fit_items_multistart(Y, seed_tag, starts=3):
    nodes = G.gh_nodes()
    x0 = G.initial_items(Y)
    res = []
    for k in range(starts):
        init = x0 if k == 0 else x0 + np.random.default_rng([42, 45, 5, seed_tag, k]).normal(0, 0.3, x0.size)
        r = minimize(G.stage1_negloglik, init, args=(Y, nodes), jac=True, method="L-BFGS-B",
                     options={"maxiter": 3000})
        a, tau = G.unpack_items(r.x, Y.shape[1])
        res.append({"nll": float(r.fun), "converged": bool(r.success), "message": str(r.message),
                    "a": a, "tau": tau, "a_out_of_range": bool(np.any((a < 0.05) | (a > 20)))})
    best = min(res, key=lambda d: d["nll"])
    nlls = [d["nll"] for d in res]
    check = {"nll": nlls, "spread": max(nlls) - min(nlls), "all_converged": all(d["converged"] for d in res),
             "bound_hit": any(d["a_out_of_range"] for d in res)}
    check["passed"] = check["spread"] <= 0.1 and check["all_converged"] and not check["bound_hit"]
    return best, check, res


def fit_long_multistart(groups, it, n_patients, starts=3):
    fits = []
    for k in range(starts):
        psi0 = M.default_psi() if k == 0 else \
            M.default_psi() + np.random.default_rng([42, 45, 6, k]).normal(0, 0.2, M.N_PSI)
        t0 = time.time()
        f = M.fit(groups, it, psi0)
        fits.append((f, time.time() - t0))
        log(f"  longitudinal start {k}: nll {f.nll:.3f}, conv {f.converged}, |g| {f.grad_norm:.3g}, "
            f"{fits[-1][1]:.0f}s, bounds {f.at_bound}")
    best = min(fits, key=lambda z: z[0].nll)[0]
    H = M.hessian(best.psi, groups, it)
    eig = np.linalg.eigvalsh(H)
    nlls = [f.nll for f, _ in fits]
    check = {"nll": nlls, "spread": max(nlls) - min(nlls), "grad_norm": best.grad_norm,
             "grad_tol": 1e-3 * n_patients, "at_bound": best.at_bound, "converged": best.converged,
             "message": best.message, "hessian_min_eig": float(eig.min()),
             "fit_seconds": [s for _, s in fits]}
    check["passed"] = (check["spread"] <= 0.1 and best.grad_norm < check["grad_tol"]
                       and not best.at_bound and eig.min() > 0)
    return best, H, check


# ---------------------------------------------------------------------------
#  Recovery truth on the stage-1 scale
# ---------------------------------------------------------------------------
def truth_on_stage1_scale(groups):
    """A-priori truth, re-expressed so the pooled visit-level θ^s at the real design has
    mean 0 and variance 1 (the stage-1 identification), with items transformed to match."""
    sd_b, sd_e = np.array(TRUTH_LONG["sd_b"]), np.array(TRUTH_LONG["sd_e"])
    C = np.eye(4)
    for (i, j), r in TRUTH_LONG["corr_b"].items():
        C[i, j] = C[j, i] = r
    D = np.outer(sd_b, sd_b) * C
    S = np.outer(sd_e, sd_e) * np.array([[1, TRUTH_LONG["corr_e"]], [TRUTH_LONG["corr_e"], 1]])
    beta = np.array(TRUTH_LONG["beta"], float)
    psi = M.to_psi(beta, D, S)
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
    sd = np.sqrt(tot)
    # θ' = (θ - μ_s)/sd_s:  β0' = (β0 - μ)/sd, other mean terms / sd; D, Σε scaled
    beta2 = beta.copy()
    beta2[[0, 1, 2]] = [(beta[0] - mu[0]) / sd[0], beta[1] / sd[0], beta[2] / sd[0]]
    beta2[[3, 4, 5]] = [(beta[3] - mu[1]) / sd[1], beta[4] / sd[1], beta[5] / sd[1]]
    sc4 = np.array([sd[0], sd[0], sd[1], sd[1]])
    D2 = D / np.outer(sc4, sc4)
    S2 = S / np.outer(sd, sd)
    aT, aP = TRUTH_ITEMS["aT"], TRUTH_ITEMS["aP"]
    tauT = TRUTH_ITEMS["tauT1"][:, None] + TRUTH_ITEMS["step"] * np.arange(4)
    tauP = TRUTH_ITEMS["tauP1"][:, None] + TRUTH_ITEMS["step"] * np.arange(4)
    it2 = M.Items(aT * sd[0], (tauT - mu[0]) / sd[0], aP * sd[1], (tauP - mu[1]) / sd[1])
    return M.to_psi(beta2, D2, S2), it2, {"mu": mu, "sd": sd, "psi_raw": psi}


def simulate_groups(rng, groups, psi, it):
    YT, YP = {}, {}
    for n, g in groups.items():
        th = M.simulate_latent(rng, g.T, g.X, psi)
        yt, yp = M.simulate_items(rng, th, n, it)
        YT[n], YP[n] = yt, yp
    return YT, YP


def regroup(groups, YT, YP):
    return {n: M.Group(g.T, g.X, YT[n].astype(float), YP[n].astype(float), g.patients)
            for n, g in groups.items()}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="code-path check: 120 patients, reduced counts; outputs are not results")
    a = ap.parse_args()
    global N_RECOVERY, N_GATE_SIMS, N_CHECK
    if a.smoke:
        N_RECOVERY, N_GATE_SIMS, N_CHECK = 1, 5, 3
        SAMPLER.update(warmup=50, keep=200)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t_start = time.time()
    status = {"steps": {}, "label_free": True}

    def finish(code, msg):
        status.update({"exit": code, "message": msg, "runtime_s": time.time() - t_start})
        write(out / "status.json", status)
        log(f"{msg} (exit {code}); runtime {status['runtime_s']:.0f}s")
        return code

    # ---- 0. provenance -------------------------------------------------------------------
    cfg = load_config()
    frozen_spec = (ROOT / SPEC_LOCK).read_text().split()[0]
    prov = {"spec": SPEC, "spec_sha256": sha(ROOT / SPEC), "spec_frozen_sha256": frozen_spec,
            "panel_md5": sha(cfg.path("outputs") / "visit_panel.csv", "md5"),
            "code_sha256": {f: sha(ROOT / f) for f in CODE}, "host": platform.node(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "truth_items": TRUTH_ITEMS, "truth_long": {k: (v if k != "corr_b" else {str(kk): vv for kk, vv in v.items()})
                                                       for k, v in TRUTH_LONG.items()},
            "seeds": {"recovery": "[42,45,1,r]", "gate_sims": "[42,45,2,r]", "sampler": "[42,45,3,i,c]",
                      "check_patients": "[42,45,4]", "item_starts": "[42,45,5,tag,k]",
                      "long_starts": "[42,45,6,k]"},
            "smoke": a.smoke, "sampler": SAMPLER, "n_recovery": N_RECOVERY, "n_gate_sims": N_GATE_SIMS, "n_check": N_CHECK}
    write(out / "provenance.json", prov)
    if prov["spec_sha256"] != frozen_spec or prov["panel_md5"] != PANEL_MD5:
        return finish(2, "specification or panel hash differs from the frozen record")
    log("provenance ok")

    v, times, treat, YTd, YPd, itT, itP = load_cohort(cfg)
    if a.smoke:                                   # code-path check only; never a result
        keep = sorted(times)[:120]
        times, treat, YTd, YPd = ({p: d[p] for p in keep} for d in (times, treat, YTd, YPd))
        v = v[v.PATNO.isin(keep)]
    groups = M.make_groups(times, treat, YTd, YPd)
    n_pat = len(times)
    status["cohort"] = {"patients": n_pat, "visits": len(v), "treatment_missing_coded_0":
                        int(pd.to_numeric(v[cfg.subtype.treatment_column], errors="coerce").isna().sum()),
                        "visit_counts": {int(n): len(g.patients) for n, g in groups.items()}}
    log(f"cohort {n_pat} patients, {len(v)} visits")

    # ---- 1. synthetic recovery ----------------------------------------------------------
    t0 = time.time()
    psi_true, it_true, scale = truth_on_stage1_scale(groups)
    rec_rows, item_rec = [], []
    for r in range(N_RECOVERY):
        rng = np.random.default_rng([42, 45, 1, r])
        sYT, sYP = simulate_groups(rng, groups, psi_true, it_true)
        sg = regroup(groups, sYT, sYP)
        fT, _, _ = fit_items_multistart(flatten(sg)[0].astype(float), 100 + r, starts=1)
        fP, _, _ = fit_items_multistart(flatten(sg)[1].astype(float), 200 + r, starts=1)
        it_hat = M.Items(fT["a"], fT["tau"], fP["a"], fP["tau"])
        fit = M.fit(sg, it_hat)
        H = M.hessian(fit.psi, sg, it_hat)
        try:
            se = np.sqrt(np.diag(np.linalg.inv(H)))
        except np.linalg.LinAlgError:
            se = np.full(M.N_PSI, np.nan)
        cover = np.abs(fit.psi - psi_true) <= 1.96 * se
        rec_rows.append({"dataset": r, "nll": fit.nll, "converged": fit.converged, "at_bound": fit.at_bound,
                         "psi_hat": fit.psi, "se": se, "covered": cover})
        item_rec.append({"dataset": r,
                         "a_corr_T": float(np.corrcoef(fT["a"], it_true.aT)[0, 1]),
                         "a_maxerr_T": float(np.max(np.abs(fT["a"] - it_true.aT))),
                         "tau_maxerr_T": float(np.max(np.abs(fT["tau"] - it_true.tauT))),
                         "a_corr_P": float(np.corrcoef(fP["a"], it_true.aP)[0, 1]),
                         "a_maxerr_P": float(np.max(np.abs(fP["a"] - it_true.aP))),
                         "tau_maxerr_P": float(np.max(np.abs(fP["tau"] - it_true.tauP)))})
        log(f"recovery {r}: covered {int(cover.sum())}/{M.N_PSI}, conv {fit.converged}, "
            f"{time.time() - t0:.0f}s")
    cov_counts = np.sum([rr["covered"] for rr in rec_rows], axis=0)
    rec = {"psi_true": psi_true, "names": M.NAMES, "coverage_counts": cov_counts,
           "passed": bool(np.all(cov_counts >= 4)), "datasets": rec_rows, "items": item_rec,
           "stage1_scale": scale, "runtime_s": time.time() - t0}
    status["steps"]["1_recovery"] = {"passed": rec["passed"], "coverage_counts": dict(zip(M.NAMES, cov_counts)),
                                     "runtime_s": rec["runtime_s"]}
    write(out / "step1_recovery.json", rec)

    # ---- 2. item fits on the real data --------------------------------------------------
    t0 = time.time()
    YTv, YPv, pidv = flatten(groups)
    bestT, chkT, _ = fit_items_multistart(YTv.astype(float), 1)
    bestP, chkP, _ = fit_items_multistart(YPv.astype(float), 2)
    it = M.Items(bestT["a"], bestT["tau"], bestP["a"], bestP["tau"])
    status["steps"]["2_item_fit"] = {"tremor": chkT, "pigd": chkP,
                                     "passed": chkT["passed"] and chkP["passed"], "runtime_s": time.time() - t0}
    write(out / "step2_items.json", {"items_T": itT, "items_P": itP, "aT": it.aT, "tauT": it.tauT,
                                     "aP": it.aP, "tauP": it.tauP, "check_T": chkT, "check_P": chkP})
    log(f"item fits: tremor {chkT['passed']}, pigd {chkP['passed']}")

    # ---- 3. longitudinal fit -----------------------------------------------------------
    t0 = time.time()
    best, H, chkL = fit_long_multistart(groups, it, n_pat)
    se = np.sqrt(np.diag(np.linalg.inv(H))) if chkL["hessian_min_eig"] > 0 else np.full(M.N_PSI, np.nan)
    beta, D, _, S, _ = M.unpack(best.psi)
    chkL["runtime_s"] = time.time() - t0
    status["steps"]["3_longitudinal_fit"] = chkL
    write(out / "step3_longitudinal.json", {"psi": best.psi, "se": se, "names": M.NAMES, "beta": beta,
                                            "D": D, "Sigma_e": S, "check": chkL, "hessian": H})
    log(f"longitudinal fit passed={chkL['passed']}")

    # ---- 4. Laplace accuracy --------------------------------------------------------------
    t0 = time.time()
    pats = sorted(times)
    pick = np.random.default_rng([42, 45, 4]).choice(len(pats), N_CHECK, replace=False)
    lap_rows = []
    for i in pick:
        p = pats[i]
        n = len(times[p])
        lap = M.laplace_one(times[p], treat[p], YTd[p], YPd[p], best.psi, it)
        if 2 * n <= 6:
            ref, se_ref, method = M.agq_one(times[p], treat[p], YTd[p], YPd[p], best.psi, it, nodes=7), 0.0, "AGQ7"
        else:
            ref, se_ref = M.is_one(times[p], treat[p], YTd[p], YPd[p], best.psi, it,
                                   np.random.default_rng([42, 45, 4, int(i)]), draws=20000)
            method = "IS20000_t5"
        lap_rows.append({"patient_index": int(i), "n_visits": n, "laplace": lap, "reference": ref,
                         "ref_mc_se": se_ref, "method": method, "error": lap - ref})
    max_err = max(abs(r["error"]) for r in lap_rows)
    lapc = {"max_abs_error": max_err, "tolerance": 0.05, "passed": max_err <= 0.05,
            "max_mc_se": max(r["ref_mc_se"] for r in lap_rows), "runtime_s": time.time() - t0}
    status["steps"]["4_laplace"] = lapc
    write(out / "step4_laplace.json", {"summary": lapc, "patients": lap_rows})
    log(f"Laplace: max |error| {max_err:.4f}, passed={lapc['passed']}")

    # ---- 5. patient sampler ------------------------------------------------------------------
    t0 = time.time()
    samp_rows = []
    for i in pick:
        p = pats[i]
        n = len(times[p])
        g, Mn, K = M.one_patient(times[p], treat[p], YTd[p], YPd[p], best.psi)
        f_mode, C_post, *_ = M.mode_and_cov(times[p], treat[p], YTd[p], YPd[p], best.psi, it)
        Kc = np.linalg.cholesky(K)
        Cp = np.linalg.cholesky(4 * C_post)

        def logl(fv, g=g):
            return float(M.loglik(fv[None], g.YT, g.YP, it)[0])
        draws = np.zeros((SAMPLER["chains"], SAMPLER["keep"], 6))
        for c in range(SAMPLER["chains"]):
            rng = np.random.default_rng([42, 45, 3, int(i), c])
            f = f_mode + Cp @ rng.standard_normal(len(f_mode))
            for s_ in range(SAMPLER["warmup"] + SAMPLER["keep"]):
                f = M.elliptical_slice(f, Mn, Kc, logl, rng)
                if s_ >= SAMPLER["warmup"]:
                    mb, _ = M.conditional_b(f, times[p], treat[p], best.psi)
                    eT, eP = M.expected_sums(f, n, it)
                    draws[c, s_ - SAMPLER["warmup"]] = np.r_[mb, eT, eP]
        rh = [MG.split_rhat(draws[:, :, q]) for q in range(6)]
        es = [MG.bulk_ess(draws[:, :, q]) for q in range(6)]
        samp_rows.append({"patient_index": int(i), "n_visits": n, "rhat": rh, "ess": es,
                          "passed": bool(max(rh) < 1.01 and min(es) >= 100)})
    sampc = {"quantities": ["b0T", "b1T", "b0P", "b1P", "E_tremor_sum", "E_pigd_sum"],
             "max_rhat": max(max(r["rhat"]) for r in samp_rows), "min_ess": min(min(r["ess"]) for r in samp_rows),
             "patients_failing": [r["patient_index"] for r in samp_rows if not r["passed"]],
             "runtime_s": time.time() - t0}
    sampc["passed"] = not sampc["patients_failing"]
    status["steps"]["5_sampler"] = sampc
    write(out / "step5_sampler.json", {"summary": sampc, "patients": samp_rows})
    log(f"sampler: max R̂ {sampc['max_rhat']:.4f}, min ESS {sampc['min_ess']:.0f}, passed={sampc['passed']}")

    b11 = {"recovery": rec["passed"], "item_fit": status["steps"]["2_item_fit"]["passed"],
           "longitudinal_fit": chkL["passed"], "laplace": lapc["passed"], "sampler": sampc["passed"]}
    status["B11"] = b11
    if not all(b11.values()):
        failed = [k for k, ok in b11.items() if not ok]
        status["gates_run"] = False
        return finish(0, f"B11 check(s) failed: {failed}. Per doc 16 B11 the fidelity gates were NOT run")

    # ---- 6. fidelity gates ------------------------------------------------------------------
    t0 = time.time()
    obs = MG.features(YTv, YPv, pidv)
    sims = []
    for r in range(N_GATE_SIMS):
        rng = np.random.default_rng([42, 45, 2, r])
        sYT, sYP = simulate_groups(rng, groups, best.psi, it)
        yt, yp, pid_s = flatten(groups, sYT, sYP)
        sims.append(MG.features(yt, yp, pid_s))
    ev = MG.evaluate(obs, sims)
    ev["runtime_s"] = time.time() - t0
    status["gates_run"] = True
    status["steps"]["6_gates"] = {"summary": ev["summary"], "decision": ev["decision"]}
    write(out / "step6_gates.json", ev)
    log(f"gates: {ev['summary']}; decision {ev['decision']}")
    if not ev["decision"]["G4_pass"]:
        return finish(0, "G4 (PIGD-zero persistence) FAILED: target 2 is UNINFORMATIVE under the "
                         "prespecified rule (doc 16 B6). Stopped; no calibration or evaluation")
    if not ev["decision"]["core_gates_G1_G3_G5_G7_pass"]:
        return finish(0, "a core fidelity gate failed: Stage 2 stops before calibration (doc 16 B6)")
    return finish(0, "pilot complete: all B11 checks and fidelity gates passed; calibration requires review")


if __name__ == "__main__":
    sys.exit(main())
