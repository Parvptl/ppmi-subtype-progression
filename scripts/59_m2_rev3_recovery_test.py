"""Stage 2 M2 Rev 3: independent recovery test (docs 46/47 R1.3, frozen 2026-10-19).

Implements doc 47 exactly; reuses frozen code paths unchanged:
  scripts/52  pipeline_dataset / fit_record / coverage       (stages (1)-(6), Step 6 Hessian path)
  scripts/54  raw_chunks / eval_s2 / _item_grad_task / stacked_sandwich / patient_slices / pack_set /
              items_from_phi / affine_tangent               (full-chain sandwich, run_d4 order)
  scripts/55  standardise / jacobian (S and G);  scripts/56 functionals / jac (17 functionals)
  scripts/51, scripts/45, src/ppmi_subtype                   (item fits, data generation, model)

Modes:
  smoke   labelled code path (NOT A RESULT): 60 patients of old dataset [42,45,1,0], reduced iterations
  phase1  node002: preflight, regeneration, regression R-a..R-d on [42,45,1,0], fresh datasets 0-2, runtime gate
  phase2  --main A B --control C D: main datasets first (seed order), then the control arm
  evaluate  gates, verdict (doc 47 §8), comparators, labels, tables, figures (login node; no fitting)

Historical verdicts are not touched: Step 6 = FAILED; D1 = NOT REPRODUCED.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import chi2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppmi_subtype import grm as G  # noqa: E402
from ppmi_subtype import m2_grm as M  # noqa: E402
from ppmi_subtype import m2_quad as MQ  # noqa: E402

FROZEN = ROOT / "docs/cluster_phase/_frozen"
S6DIR = ROOT / "experiments/20261018T222857Z_stage2_step6_full_pipeline_recovery/step6"
DIAG = ROOT / "experiments/20261019T084559Z_stage2_step6_diagnostics"
PLAN_LOCKS = {"doc46": "46_STAGE2_REV3_AMENDMENT.sha256", "doc47": "47_STAGE2_REV3_RECOVERY_TEST_PLAN.sha256"}
Q, G_REP, JOBS = 15, 5, 32
H_FD, H_FD2 = 1e-4, 2e-4
KT, KP = 11, 5
NT, NP = 5 * KT, 5 * KP
NPHI = NT + NP
R_TOTAL = 100
Z95, COV_FRAC, DEF_FRAC, BIAS_T, FLAG_MAX = 1.96, 0.88, 0.05, 0.40, 10
TOL_ID, TOL_G, TOL_N34, TOL_N1, TOL_N5, TOL_N6, TOL_N7, TOL_N8 = 1e-8, 1e-6, 1e-5, 1e-8, 1e-4, 1e12, 1e-10, 0.05
RANK_TOL = 1e-8
TEST_RELABEL = (1.13, -0.07, 0.91, 0.05)                 # (c_T, d_T, c_P, d_P), doc 47 §7 (S6)
FIT_LIMIT_S, GATE_PROJ_MAX, CAP_NH, L2_MAX_H = 3600.0, 28.0, 30.0, 13.5
KEY_OLD0, KEY_SMOKE = -1, -2          # integer pool keys for scripts/51 (fresh datasets use r = 0..99)
LABEL = "Stage 2 M2 Rev 3 recovery test (docs 46/47 R1.3 frozen). Step 6 remains FAILED; D1 remains NOT REPRODUCED."

PSI_NAMES = list(M.NAMES)
F_NAMES = ([f"{q}_{s}" for s in ("T", "P") for q in ("slope_over_sdint", "treat_over_sdint",
                                                     "half_log_var_slope_over_int", "half_log_var_eps_over_int")]
           + [f"corrD_{i}{j}" for i in range(4) for j in range(i)] + ["corrEps_TP"]
           + ["item_anchored_location_T", "item_anchored_location_P"])
GUARD_NAMES = F_NAMES[:15]
RECOVERY_NAMES = PSI_NAMES + F_NAMES[15:]                  # 19 re-expressed + ν_T, ν_P
GATE_NAMES = RECOVERY_NAMES + GUARD_NAMES                   # 21 recovery, then 15 guards


def sha(p, kind="sha256"):
    h = hashlib.new(kind)
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _conv(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (tuple, set)):
        return list(o)
    raise TypeError(type(o))


def write_atomic(path, obj, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_conv))
    os.replace(tmp, path)


# =====================================================================================================
#  Pure functions (unit-tested): masks, gates, verdict, labels, runtime gate
# =====================================================================================================
def mask_psi(sub):
    """ψ coordinates whose Jacobian under θ' = (θ − d)/c depends on c (doc 46 §3.5)."""
    keep = {"T": ["beta0_T", "beta1_T", "gamma_T", "L10"],
            "P": ["beta0_P", "beta1_P", "gamma_P", "L20", "L21", "L30", "L31", "L32", "Le10"]}[sub]
    return np.array([n in keep for n in PSI_NAMES])


def mask_phi(sub):
    """packed item coordinates γ1 (first threshold) of subscore `sub`."""
    m = np.zeros(NPHI, bool)
    if sub == "T":
        m[[5 * k + 1 for k in range(KT)]] = True
    else:
        m[[NT + 5 * k + 1 for k in range(KP)]] = True
    return m


def gate_values(est, se, truth):
    """z, defined, covered for arrays of gates (doc 47 §5, S3). Undefined: non-finite est/SE or SE ≤ 0."""
    est, se, truth = (np.asarray(x, float) for x in (est, se, truth))
    defined = np.isfinite(est) & np.isfinite(se) & (se > 0)
    z = np.full(est.shape, np.nan)
    z[defined] = (est[defined] - truth[defined]) / se[defined]
    covered = defined & (np.abs(np.where(defined, z, np.inf)) <= Z95)
    return z, defined, covered


def gate_status(z, defined, covered):
    """Per-gate status on a dataset set: arrays (n_datasets, n_gates) → list of 'pass'/'fail'/'ne'."""
    z, defined, covered = np.asarray(z, float), np.asarray(defined, bool), np.asarray(covered, bool)
    n = z.shape[0]
    out = []
    for g in range(z.shape[1]):
        undef = int(np.sum(~defined[:, g]))
        if undef > math.floor(DEF_FRAC * n):
            out.append("ne")
            continue
        c1 = int(np.sum(covered[:, g])) >= math.ceil(COV_FRAC * n)
        zd = z[defined[:, g], g]
        c2 = bool(zd.size > 0 and abs(float(np.mean(zd))) <= BIAS_T)
        out.append("pass" if (c1 and c2) else "fail")
    return out


def verdict(integrity_fail, execution_fail, z, defined, covered, flagged, n_recovery=21):
    """doc 47 §8 (R3, R8): returns dict(verdict, reasons, status_A, status_AF)."""
    if integrity_fail:
        return {"verdict": "INTEGRITY STOP", "reasons": [integrity_fail]}
    if execution_fail:
        return {"verdict": "INCOMPLETE", "reasons": [execution_fail]}
    flagged = np.asarray(flagged, bool)
    if int(flagged.sum()) > FLAG_MAX:
        return {"verdict": "INCONCLUSIVE — FLAG EXCESS", "reasons": [f"|F| = {int(flagged.sum())} > {FLAG_MAX}"]}
    sA = gate_status(z, defined, covered)
    keep = ~flagged
    sAF = gate_status(np.asarray(z)[keep], np.asarray(defined)[keep], np.asarray(covered)[keep])
    rf = [GATE_NAMES[g] for g in range(n_recovery) if sA[g] == "fail" and sAF[g] == "fail"]
    res = {"status_A": dict(zip(GATE_NAMES, sA)), "status_AF": dict(zip(GATE_NAMES, sAF)), "robust_failures": rf}
    if rf:
        return {"verdict": "FAIL", "reasons": [f"robust recovery failure: {rf}"], **res}
    if all(a == "pass" and b == "pass" for a, b in zip(sA, sAF)):
        return {"verdict": "PASS", "reasons": [], **res}
    reasons = []
    if any(s == "ne" for s in sA):
        reasons.append("NE")
    if any((sA[g] == "fail") != (sAF[g] == "fail") for g in range(n_recovery)):
        reasons.append("FLAG-SENSITIVE")
    if any(sA[g] == "fail" or sAF[g] == "fail" for g in range(n_recovery, len(GATE_NAMES))):
        reasons.append("GUARD")
    return {"verdict": "INCONCLUSIVE — OTHER", "reasons": reasons, **res}


def failure_label(z_col, defined_col, covered_col):
    """S1 (explanatory only): bias-type / dispersion-type / mixed / unclassified for a failing gate on A."""
    zd = np.asarray(z_col, float)[np.asarray(defined_col, bool)]
    n = zd.size
    c1 = int(np.sum(covered_col)) >= math.ceil(COV_FRAC * len(z_col))
    c2 = n > 0 and abs(float(np.mean(zd))) <= BIAS_T
    lower = float(np.std(zd, ddof=1) * np.sqrt((n - 1) / chi2.ppf(0.975, n - 1))) if n > 1 else float("nan")
    disp = (not c1) and lower > 1
    if not c2 and disp:
        return "mixed", lower
    if not c2:
        return "bias-type", lower
    if disp:
        return "dispersion-type", lower
    return "unclassified", lower


def runtime_gate(U1_h, t_r, t_c, fit_walls, U_smoke_h, L2_h):
    """doc 47 §9 (B1). Missing/invalid timing → gate fails."""
    vals = list(t_r) + [t_c, U1_h] + list(fit_walls)
    if len(t_r) != 3 or not fit_walls or any((v is None) or (not np.isfinite(v)) or v < 0 for v in vals):
        return {"passed": False, "reason": "missing or invalid timing"}
    mt = float(np.mean(t_r))
    P = U1_h + 97 * mt / 3600 + 10 * t_c / 3600
    per_job = (49 * mt + 5 * t_c) / 3600
    chk = {"fits_le_3600s": bool(max(fit_walls) <= FIT_LIMIT_S), "P_le_28": bool(P <= GATE_PROJ_MAX),
           "smoke_plus_P_le_30": bool(U_smoke_h + P <= CAP_NH), "per_job_le_L2": bool(per_job <= L2_h)}
    return {"passed": all(chk.values()), "P_node_hours": P, "per_job_projection_h": per_job, "L2_h": L2_h,
            "mean_t_r_s": mt, "t_c_s": t_c, "U1_h": U1_h, "U_smoke_h": U_smoke_h, "max_fit_s": max(fit_walls), **chk}


# =====================================================================================================
#  Context: modules, design, truth
# =====================================================================================================
class Ctx:
    def __init__(self):
        self.P = load_module("m2_pilot_r3", ROOT / "scripts/45_m2_pilot.py")
        self.S1A = load_module("s1a_r3", ROOT / "scripts/51_m2a1_s1a_true_latent_items.py")
        self.S6 = load_module("s6_r3", ROOT / "scripts/52_m2a1_full_pipeline_recovery.py")
        self.D54 = load_module("d54_r3", ROOT / "scripts/54_m2_step6_diag_compute.py")
        self.D55 = load_module("d55_r3", ROOT / "scripts/55_m2_scale_convention_readonly.py")
        self.F56 = load_module("f56_r3", ROOT / "scripts/56_m2_rev3_planning_checks.py")
        from ppmi_subtype.config import load_config
        self.cfg = load_config()
        _, t, x, yt, yp, *_ = self.P.load_cohort(self.cfg)
        self.g0 = M.make_groups(t, x, yt, yp)                     # analysis design (no responses used)
        self.psi_true, self.it_true, _ = self.P.truth_on_stage1_scale(self.g0)
        self.phi_true = np.concatenate([G.pack_items(self.it_true.aT, self.it_true.tauT),
                                        G.pack_items(self.it_true.aP, self.it_true.tauP)])
        self.f_true = self.F56.functionals(self.psi_true, self.phi_true[:NT], self.phi_true[NT:])
        self.relabel = self.D55.R53.relabel_psi
        self.psi_default = M.default_psi()

    def dataset(self, seed):
        return self.P.regroup(self.g0, *self.P.simulate_groups(np.random.default_rng(seed), self.g0,
                                                               self.psi_true, self.it_true))

    @staticmethod
    def subset(groups, k_tot):
        keep, acc = {}, 0
        for n, g in groups.items():
            k = min(len(g.patients), max(0, k_tot - acc))
            if k:
                keep[n] = M.Group(g.T[:k], g.X[:k], g.YT[:k], g.YP[:k], g.patients[:k])
                acc += k
        return keep

    def responses_hash(self, sg):
        YT, YP, _ = self.P.flatten(sg)
        return hashlib.sha256(np.ascontiguousarray(np.concatenate([YT.ravel(), YP.ravel()]).astype(np.int64)).tobytes()).hexdigest()

    # -------- S / gates --------
    def reexpress(self, psi):
        """S(ψ) from the estimate and the design only (no truth argument; truth-swap test)."""
        return self.D55.standardise(self.g0, psi)

    def evaluate_gates(self, psi2, phi2, V, truth_psi, truth_F):
        """36 gate values from (ψ̂2, φ̂2, full-chain V); truth enters only z/coverage."""
        sps, mom = self.reexpress(psi2)
        Gm = self.D55.jacobian(self.g0, psi2)
        Vpp = V[np.ix_(np.r_[99:179, 179:198], np.r_[99:179, 179:198])]
        Vss = V[179:198, 179:198]
        with np.errstate(invalid="ignore"):
            se_S = np.sqrt(np.diag(Gm @ Vss @ Gm.T)) if np.all(np.isfinite(Vss)) else np.full(19, np.nan)
        F_hat = self.F56.functionals(psi2, phi2[:NT], phi2[NT:])
        J_F = self.F56.jac(lambda v: self.F56.functionals(v[80:], v[:NT], v[NT:80]), np.r_[phi2, psi2])
        with np.errstate(invalid="ignore"):
            se_F = np.sqrt(np.diag(J_F @ Vpp @ J_F.T)) if np.all(np.isfinite(Vpp)) else np.full(17, np.nan)
        est = np.r_[sps, F_hat[15:], F_hat[:15]]
        se = np.r_[se_S, se_F[15:], se_F[:15]]
        tru = np.r_[truth_psi, truth_F[15:], truth_F[:15]]
        z, defined, covered = gate_values(est, se, tru)
        return {"S_psi2": sps, "moments": mom, "G": Gm, "se_S": se_S, "F_hat": F_hat, "J_F": J_F, "se_F": se_F,
                "est": est, "se": se, "z": z, "defined": defined, "covered": covered}


# =====================================================================================================
#  Per-dataset computation (doc 47 §16)
# =====================================================================================================
_TIMING: dict = {}


class Ops:
    """Makes exactly the scripts/52 calls (pipeline_dataset contract)."""

    def __init__(self, ctx, key, jobs, maxiter_s2, maxiter_it):
        self.ctx, self.key, self.jobs, self.m2, self.mi = ctx, key, jobs, maxiter_s2, maxiter_it
        self.t = {"fits": [], "hessians": []}
        self.objs = []

    def fit_stage2(self, sg, it, psi0):
        t = time.perf_counter()
        f = MQ.fit(sg, it, Q=Q, psi0=psi0, jobs=self.jobs, maxiter=self.m2)
        self.t["fits"].append(time.perf_counter() - t)
        return f

    def hessian(self, sg, it, f):
        o = MQ.Objective(sg, it, Q, jobs=self.jobs)
        o.state = dict(f.state)
        t = time.perf_counter()
        try:
            H = MQ.fd_hessian(o, f.psi)
        finally:
            o.close()
        w = time.perf_counter() - t
        self.t["hessians"].append(w)
        return {"H": H, "wall_s": w, "penalised_fd_evals": sum(1 for e in o.log if e["penalty"]),
                "penalty_reasons": [e["reasons"] for e in o.log if e["penalty"]], "state": dict(f.state)}

    def reported(self, sg, it, f):
        t = time.perf_counter()
        rep = MQ.evaluate_reported(f.psi, sg, it, Q, G_REP, init_state=f.state)
        rep["wall_s"] = time.perf_counter() - t
        return rep

    def refit_items(self, sg, psi1):
        S1A = self.ctx.S1A
        vl = S1A.visit_latent(sg, psi1)
        for s in ("T", "P"):
            S1A._FIT["lat"][(self.key, s)] = vl[s]
        t = time.perf_counter()
        fits = {s: f for _, _, s, f in S1A.run_pool([("oracle", self.key, s, self.mi) for s in ("T", "P")], self.jobs)}
        w = time.perf_counter() - t
        self.t["s1b"] = w
        it2 = M.Items(fits["T"]["a"], fits["T"]["tau"], fits["P"]["a"], fits["P"]["tau"])
        return {"items": it2, "fits": fits, "wall_s": w}


def stage1_batch(ctx, keyed_groups, jobs):
    """Stage 1 N(0,1) item fits for several datasets in one scripts/51 pool (doc 47 §16.1)."""
    S1A, P = ctx.S1A, ctx.P
    S1A._FIT.setdefault("Y", {})
    S1A._FIT.setdefault("lat", {})
    S1A._FIT["P"] = P
    for key, sg in keyed_groups.items():
        YT, YP, _ = P.flatten(sg)
        S1A._FIT["Y"][(key, "T")] = YT.astype(float)
        S1A._FIT["Y"][(key, "P")] = YP.astype(float)
    t = time.perf_counter()
    res = {(k, s): f for _, k, s, f in S1A.run_pool([("N01", k, s, 3000) for k in keyed_groups for s in ("T", "P")], jobs)}
    return res, time.perf_counter() - t


def full_chain(ctx, sg, it1, it2, psi1, psi2, H1, H2, jobs, n8=False, ops=None, f2=None):
    """doc 46 §3.2 path in run_d4 order. Returns V, blocks, scores, checks inputs, timings."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    D54, S1A, P = ctx.D54, ctx.S1A, ctx.P
    phi1, phi2 = np.concatenate(D54.pack_set(it1)), np.concatenate(D54.pack_set(it2))
    YT, YP, pid = P.flatten(sg)
    D54._W.clear()
    D54._W.update(chunks=D54.raw_chunks(sg), Y={"T": YT.astype(float), "P": YP.astype(float)}, groups=sg, S1A=S1A)
    tup = lambda it: (it.aT, it.tauT, it.aP, it.tauP)  # noqa: E731
    out = {"timing": {}}
    pool = ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) if jobs > 1 else None
    try:
        t = time.perf_counter()
        base = D54.eval_s2(pool, [(psi1, tup(it1)), (psi2, tup(it2))], None, per_patient=True)
        b1, b2 = base[0]["b"], base[1]["b"]

        def cross(psi, phi, bmap, h):
            cfgs = []
            for k in range(NPHI):
                for sgn in (+1, -1):
                    pp = phi.copy()
                    pp[k] += sgn * h
                    cfgs.append((psi, tup(D54.items_from_phi(pp[:NT], pp[NT:], KT, KP))))
            res = D54.eval_s2(pool, cfgs, bmap)
            fails = sum(1 for x in res if x["failures"]) + sum(1 for x in res if not x["finite"])
            C = np.column_stack([(res[2 * k]["grad"] - res[2 * k + 1]["grad"]) / (2 * h) for k in range(NPHI)])
            return C, fails, sum(x["n_fallback"] for x in res)
        C1, f1, fb1 = cross(psi1, phi1, b1, H_FD)
        C2, f2c, fb2 = cross(psi2, phi2, b2, H_FD)
        out["timing"]["cross_s"] = time.perf_counter() - t
        t = time.perf_counter()
        tasks = []
        for s, (lo, n) in (("T", (0, NT)), ("P", (NT, NP))):
            for k in range(n):
                for sgn in (+1, -1):
                    p1 = phi1[lo:lo + n].copy()
                    p1[k] += sgn * H_FD
                    tasks.append((("A1", s, k, sgn), "n01", s, p1, None))
                    p2 = phi2[lo:lo + n].copy()
                    p2[k] += sgn * H_FD
                    tasks.append((("A3", s, k, sgn), "lat", s, p2, psi1))
            for j in range(M.N_PSI):
                for sgn in (+1, -1):
                    q = psi1.copy()
                    q[j] += sgn * H_FD
                    tasks.append((("E3", s, j, sgn), "lat", s, phi2[lo:lo + n], q))
        res = {tag: g for tag, f, g in (pool.map(D54._item_grad_task, tasks, chunksize=4) if pool else
                                        map(D54._item_grad_task, tasks))}
        A1, A3, E3 = np.zeros((NPHI, NPHI)), np.zeros((NPHI, NPHI)), np.zeros((NPHI, M.N_PSI))
        for s, (lo, n) in (("T", (0, NT)), ("P", (NT, NP))):
            for k in range(n):
                A1[lo:lo + n, lo + k] = (res[("A1", s, k, 1)] - res[("A1", s, k, -1)]) / (2 * H_FD)
                A3[lo:lo + n, lo + k] = (res[("A3", s, k, 1)] - res[("A3", s, k, -1)]) / (2 * H_FD)
            for j in range(M.N_PSI):
                E3[lo:lo + n, j] = (res[("E3", s, j, 1)] - res[("E3", s, j, -1)]) / (2 * H_FD)
        out["timing"]["item_fd_s"] = time.perf_counter() - t
        n8res = None
        if n8:
            t = time.perf_counter()
            C2h, f2h, _ = cross(psi2, phi2, b2, H_FD2)
            out["timing"]["n8_cross_s"] = time.perf_counter() - t
            n8res = {"C2_2h": C2h, "fails": f2h}
    finally:
        if pool is not None:
            pool.shutdown()
    t = time.perf_counter()
    vl1 = S1A.visit_latent(sg, psi1)
    nodes = G.gh_nodes()
    sl = D54.patient_slices(pid)
    U1, U3 = np.zeros((len(sl), NPHI)), np.zeros((len(sl), NPHI))
    for i, q in enumerate(sl):
        for s, Y, lo, n in (("T", YT, 0, NT), ("P", YP, NT, NP)):
            Yq = Y[q].astype(float)
            U1[i, lo:lo + n] = G.stage1_negloglik(phi1[lo:lo + n], Yq, nodes)[1]
            U3[i, lo:lo + n] = S1A.latent_negloglik(phi2[lo:lo + n], Yq, vl1[s][0][q], vl1[s][1][q], nodes)[1]
    tot1 = np.r_[G.stage1_negloglik(phi1[:NT], YT.astype(float), nodes)[1], G.stage1_negloglik(phi1[NT:], YP.astype(float), nodes)[1]]
    tot3 = np.r_[S1A.latent_negloglik(phi2[:NT], YT.astype(float), *vl1["T"], nodes)[1],
                 S1A.latent_negloglik(phi2[NT:], YP.astype(float), *vl1["P"], nodes)[1]]
    out["timing"]["scores_s"] = time.perf_counter() - t
    U2a, U4 = base[0]["pp"], base[1]["pp"]
    p = M.N_PSI
    Jm = np.zeros((2 * NPHI + 2 * p, 2 * NPHI + 2 * p))
    i1, i2, i3, i4 = slice(0, NPHI), slice(NPHI, NPHI + p), slice(NPHI + p, 2 * NPHI + p), slice(2 * NPHI + p, None)
    Jm[i1, i1] = A1
    Jm[i2, i1], Jm[i2, i2] = C1, H1
    Jm[i3, i2], Jm[i3, i3] = E3, A3
    Jm[i4, i3], Jm[i4, i4] = C2, H2
    U = np.hstack([U1, U2a, U3, U4])
    try:
        V, B = D54.stacked_sandwich(Jm, U)
    except np.linalg.LinAlgError:
        V, B = np.full(Jm.shape, np.nan), U.T @ U
    out.update(V=V, B=B, Jm=Jm, A1=A1, A3=A3, E3=E3, C1=C1, C2=C2, H1=H1, H2=H2, U=U, phi1=phi1, phi2=phi2,
               g2=base[0]["grad"], g4=base[1]["grad"], g1=tot1, g3=tot3,
               base_fail=[base[0]["failures"], base[1]["failures"]], base_fallback=[base[0]["n_fallback"], base[1]["n_fallback"]],
               cross_fail=f1 + f2c, cross_fallback=fb1 + fb2, n8=n8res)
    if n8 and ops is not None and f2 is not None:                 # H2 at 2h (Step 6 path, fit state)
        o = MQ.Objective(sg, it2, Q, jobs=jobs)
        o.state = dict(f2.state)
        t = time.perf_counter()
        try:
            out["n8"]["H2_2h"] = MQ.fd_hessian(o, psi2, h=H_FD2)
        finally:
            o.close()
        out["timing"]["n8_hess_s"] = time.perf_counter() - t
    return out


def checks_and_flags(ctx, rec, fc, ev, it2, psi1, psi2, maxiter_s2, n8):
    """N1–N10, transformation identities, flags F1–F8 (doc 46 §3.5, doc 47 §7)."""
    D54, relabel = ctx.D54, ctx.relabel
    chk = {}
    # N1 additivity per block
    U = fc["U"]
    blocks = {"phi1": (slice(0, 80), fc["g1"]), "psi1": (slice(80, 99), fc["g2"]),
              "phi2": (slice(99, 179), fc["g3"]), "psi2": (slice(179, 198), fc["g4"])}
    chk["N1"] = {k: float(np.max(np.abs(U[:, s].sum(0) - tot)) / max(1.0, float(np.linalg.norm(tot)))) for k, (s, tot) in blocks.items()}
    # N3 / N4 exact identities with the gradient term
    n3, n4 = {}, {}
    for sub in ("T", "P"):
        for par in ("c", "d"):
            vphi, vpsi = D54.affine_tangent(it2, psi2, sub, par, relabel)
            r3 = fc["C2"] @ vphi + fc["H2"] @ vpsi - (mask_psi(sub) * fc["g4"] if par == "c" else 0)
            n3[f"{sub}_{par}"] = float(np.linalg.norm(r3) / np.linalg.norm(fc["H2"] @ vpsi))
            vphi4, vpsi4 = D54.affine_tangent(it2, psi1, sub, par, relabel)
            r4 = fc["A3"] @ vphi4 + fc["E3"] @ vpsi4 - (mask_phi(sub) * fc["g3"] if par == "c" else 0)
            n4[f"{sub}_{par}"] = float(np.linalg.norm(r4) / np.linalg.norm(fc["A3"] @ vphi4))
    chk["N3"], chk["N4"] = n3, n4
    chk["N5"] = {k: float(np.max(np.abs(X - X.T)) / max(np.max(np.abs(X)), 1e-300)) for k, X in
                 (("A1", fc["A1"]), ("A3", fc["A3"]), ("H1", fc["H1"]), ("H2", fc["H2"]))}
    try:
        sv = np.linalg.svd(fc["Jm"], compute_uv=False)
        chk["N6"] = float(sv[0] / sv[-1])
    except np.linalg.LinAlgError:
        chk["N6"] = float("inf")
    V = fc["V"]
    if np.all(np.isfinite(V)):
        ev_ = np.linalg.eigvalsh(0.5 * (V + V.T))
        chk["N7_min_over_max"] = float(ev_.min() / ev_.max())
        n7_ok = bool(ev_.min() >= -TOL_N7 * ev_.max())
    else:
        chk["N7_min_over_max"], n7_ok = float("nan"), False
    chk["N7_ok"] = n7_ok
    chk["N9"] = (np.diag(U[:, 179:198].T @ U[:, 179:198]) / np.diag(fc["H2"])).tolist()
    # transformation identities (F5) and N10
    sps = ev["S_psi2"]
    chk["idempotence"] = float(np.max(np.abs(ctx.reexpress(sps)[0] - sps)))
    chk["relabel_invariance"] = float(np.max(np.abs(ctx.reexpress(relabel(psi2, *TEST_RELABEL))[0] - sps)))
    Gm = ev["G"]
    n10 = []
    for sub in ("T", "P"):
        for par in ("c", "d"):
            _, vpsi = D54.affine_tangent(it2, psi2, sub, par, relabel)
            n10.append(float(np.linalg.norm(Gm @ vpsi) / np.linalg.norm(vpsi)))
    sg_ = np.linalg.svd(Gm, compute_uv=False)
    chk["N10_annihilation_max"] = max(n10)
    chk["N10_rank"] = int(np.sum(sg_ > RANK_TOL * sg_[0]))
    # N8 (fresh datasets 0–2)
    if n8 and fc.get("n8") and "H2_2h" in fc["n8"]:
        Jm2 = fc["Jm"].copy()
        Jm2[179:198, 99:179] = fc["n8"]["C2_2h"]
        Jm2[179:198, 179:198] = fc["n8"]["H2_2h"]
        try:
            V2, _ = D54.stacked_sandwich(Jm2, U)
            se2 = np.sqrt(np.diag(ev["G"] @ V2[179:198, 179:198] @ ev["G"].T))
            chk["N8"] = float(np.max(np.abs(se2 / ev["se_S"] - 1)))
        except np.linalg.LinAlgError:
            chk["N8"] = float("inf")
    # -------- flags --------
    f1, f2_ = rec["fit_1"], rec["fit_2"]
    flags = {}
    flags["F1"] = [k for k, f in (("fit_2", f1), ("fit_4", f2_)) if (not f.converged) or f.n_iter >= maxiter_s2]
    itf = []
    for lab, fits in (("N01", rec["items_1_fits"]), ("S1b", rec["items_2"]["fits"])):
        for s in ("T", "P"):
            a = np.asarray(fits[s]["a"])
            if (not fits[s]["converged"]) or np.any((a < 0.05) | (a > 20)):
                itf.append(f"{lab}_{s}")
    flags["F2"] = itf
    flags["F3"] = [k for k, f in (("psi1", f1), ("psi2", f2_)) if f.at_bound]
    flags["F4"] = [k for k, r in (("psi1", rec["reported_1"]), ("psi2", rec["reported_2"]))
                   if r["n_fallback_patients"] > 0 or r["budget_triggers"] > 0]
    f5 = []
    if chk["idempotence"] > TOL_ID:
        f5.append("idempotence")
    if chk["relabel_invariance"] > TOL_ID:
        f5.append("relabel_invariance")
    if chk["N10_annihilation_max"] > TOL_G or chk["N10_rank"] != 15:
        f5.append("N10")
    flags["F5"] = f5
    f6 = []
    if max(chk["N1"].values()) > TOL_N1:
        f6.append("N1")
    if max(n3.values()) > TOL_N34:
        f6.append("N3")
    if max(n4.values()) > TOL_N34:
        f6.append("N4")
    if max(chk["N5"].values()) > TOL_N5:
        f6.append("N5")
    if not chk["N6"] <= TOL_N6:
        f6.append("N6")
    if rec["hess_1"]["penalised_fd_evals"] or rec["hess_2"]["penalised_fd_evals"]:
        f6.append("penalised_H_fd")
    if fc["cross_fail"] or any(fc["base_fail"]):
        f6.append("C_eval_failure")
    flags["F6"] = f6
    flags["F7"] = ["N8"] if (n8 and not chk.get("N8", 0.0) <= TOL_N8) else []
    flags["F8"] = ([] if (n7_ok and bool(np.all(ev["defined"]))) else
                   ["undefined:" + ",".join(np.array(GATE_NAMES)[~ev["defined"]].tolist()) + ("" if n7_ok else ";N7")])
    return chk, {k: v for k, v in flags.items()}


def process_dataset(ctx, key, sg, res1, jobs, maxiter_s2=1000, maxiter_it=3000, n8=False, truth_swap=False, out_npz=None):
    """Stages (1)–(6) + sandwich + S + gates + checks + flags for one dataset."""
    t0 = time.perf_counter()
    S1A = ctx.S1A
    S1A._FIT.setdefault("lat", {})
    it1 = M.Items(res1[(key, "T")]["a"], res1[(key, "T")]["tau"], res1[(key, "P")]["a"], res1[(key, "P")]["tau"])
    ops = Ops(ctx, key, jobs, maxiter_s2, maxiter_it)
    rec = ctx.S6.pipeline_dataset(sg, it1, ops, ctx.psi_default)
    rec["items_1_fits"] = {s: res1[(key, s)] for s in ("T", "P")}
    psi1, psi2 = rec["fit_1"].psi, rec["fit_2"].psi
    it2 = rec["items_2"]["items"]
    for f in (rec["fit_1"], rec["fit_2"]):
        if not (np.isfinite(f.nll) and np.all(np.isfinite(f.psi))):
            raise FloatingPointError("non-finite objective or estimate at an accepted optimum")
    t_pipe = time.perf_counter() - t0
    fc = full_chain(ctx, sg, it1, it2, psi1, psi2, rec["hess_1"]["H"], rec["hess_2"]["H"], jobs, n8=n8, ops=ops, f2=rec["fit_2"])
    t_ev = time.perf_counter()
    ev = ctx.evaluate_gates(psi2, fc["phi2"], fc["V"], ctx.psi_true, ctx.f_true)
    chk, flags = checks_and_flags(ctx, rec, fc, ev, it2, psi1, psi2, maxiter_s2, n8)
    # secondary (fixed-item) intervals, raw results, N(0,1)-only comparator, Le00 attribution, item recovery
    H1, H2 = rec["hess_1"]["H"], rec["hess_2"]["H"]
    sec = {}
    try:
        H2i = np.linalg.inv(H2) if np.linalg.eigvalsh(H2).min() > 0 else None
    except np.linalg.LinAlgError:
        H2i = None
    if H2i is not None:
        se_S2 = np.sqrt(np.diag(ev["G"] @ H2i @ ev["G"].T))
        JFp = ev["J_F"][:, 80:]
        se_F2 = np.sqrt(np.diag(JFp @ H2i @ JFp.T))
        se_sec = np.r_[se_S2, se_F2[15:], se_F2[:15]]
        sec["z"], sec["defined"], sec["covered"] = gate_values(ev["est"], se_sec, np.r_[ctx.psi_true, ctx.f_true[15:], ctx.f_true[:15]])
        sec["raw_se_H2"] = np.sqrt(np.diag(H2i))
    else:
        sec["z"] = sec["defined"] = sec["covered"] = None
        sec["raw_se_H2"] = np.full(19, np.nan)
    raw = {"psi2": psi2, "se_fullchain_raw": np.sqrt(np.clip(np.diag(fc["V"][179:198, 179:198]), 0, None)),
           "z_fixed_item": (psi2 - ctx.psi_true) / sec["raw_se_H2"]}
    raw["covered_D3"] = (ctx.S6.coverage(psi2, ctx.psi_true, H2)["covered"]).tolist()
    try:
        s1, _ = ctx.reexpress(psi1)
        G1 = ctx.D55.jacobian(ctx.g0, psi1)
        H1i = np.linalg.inv(H1)
        se1 = np.sqrt(np.diag(G1 @ H1i @ G1.T))
        n01 = {"S_psi1": s1, "z": (s1 - ctx.psi_true) / se1}
    except Exception as e:  # noqa: BLE001  (reported comparator only)
        n01 = {"error": str(e)}
    try:
        J2 = -np.linalg.inv(H2) @ fc["C2"]
        dphi = fc["phi2"] - ctx.phi_true
        al = {s: ctx.D55.R53.affine_fit(getattr(it2, f"a{s}"), getattr(it2, f"tau{s}"), getattr(ctx.it_true, f"a{s}"),
                                         getattr(ctx.it_true, f"tau{s}")) for s in ("T", "P")}
        it_aff = M.Items(ctx.it_true.aT * al["T"]["c_joint"], (ctx.it_true.tauT - al["T"]["d_joint"]) / al["T"]["c_joint"],
                         ctx.it_true.aP * al["P"]["c_joint"], (ctx.it_true.tauP - al["P"]["d_joint"]) / al["P"]["c_joint"])
        daff = np.concatenate(ctx.D54.pack_set(it_aff)) - ctx.phi_true
        mT = np.r_[np.ones(NT), np.zeros(NP)].astype(bool)
        j = PSI_NAMES.index("Le00_log")
        le00 = {"pred_total": float(J2[j] @ dphi), "pred_T_items": float(J2[j, mT] @ dphi[mT]),
                "pred_P_items": float(J2[j, ~mT] @ dphi[~mT]), "pred_affine": float(J2[j] @ daff),
                "pred_nonaffine": float(J2[j] @ (dphi - daff)), "units": "raw ψ (not SE)"}
    except Exception as e:  # noqa: BLE001
        le00 = {"error": str(e)}
    mom = ev["moments"]
    items_re = {"aT": it2.aT * mom["c_T"], "tauT": (it2.tauT - mom["d_T"]) / mom["c_T"],
                "aP": it2.aP * mom["c_P"], "tauP": (it2.tauP - mom["d_P"]) / mom["c_P"]}
    item_rec = {s: {"a_corr": float(np.corrcoef(items_re[f"a{s}"], getattr(ctx.it_true, f"a{s}"))[0, 1]),
                    "a_maxerr": float(np.max(np.abs(items_re[f"a{s}"] - getattr(ctx.it_true, f"a{s}")))),
                    "tau_maxerr": float(np.max(np.abs(items_re[f"tau{s}"] - getattr(ctx.it_true, f"tau{s}"))))}
                for s in ("T", "P")}
    swap, t_swap = None, 0.0
    if truth_swap:
        ts = time.perf_counter()
        ev2 = ctx.evaluate_gates(psi2, fc["phi2"], fc["V"], ctx.psi_true + 1, ctx.f_true)
        swap = {"S_identical": bool(np.array_equal(ev2["S_psi2"], ev["S_psi2"], equal_nan=True)),
                "seS_identical": bool(np.array_equal(ev2["se_S"], ev["se_S"], equal_nan=True)),
                "z_changed": bool(np.all(ev2["z"][:19] != ev["z"][:19]))}
        swap["passed"] = swap["S_identical"] and swap["seS_identical"] and swap["z_changed"]
        t_swap = time.perf_counter() - ts
    t_ev = time.perf_counter() - t_ev
    fr = {k: ctx.S6.fit_record(rec[k], w, maxiter_s2) for k, w in (("fit_1", ops.t["fits"][0]), ("fit_2", ops.t["fits"][1]))}
    timing = {"pipeline_s": t_pipe, **fc["timing"], "evaluation_s": t_ev, "swap_s": t_swap,
              "fit_walls_s": ops.t["fits"], "hessian_walls_s": ops.t["hessians"], "s1b_s": ops.t.get("s1b")}
    timing["T_r_excl_n8_swap_s"] = (t_pipe + fc["timing"]["cross_s"] + fc["timing"]["item_fd_s"] + fc["timing"]["scores_s"]
                                    + t_ev - (timing.get("swap_s") or 0.0))
    if out_npz is not None:
        np.savez_compressed(out_npz, V=fc["V"], Jm=fc["Jm"], B=fc["B"], H1=H1, H2=H2, G=ev["G"], J_F=ev["J_F"])
    return {"key": key, "psi1": psi1, "psi2": psi2, "phi1": fc["phi1"], "phi2": fc["phi2"],
            "S_psi2": ev["S_psi2"], "moments": mom, "gate_names": GATE_NAMES, "est": ev["est"], "se": ev["se"],
            "z": ev["z"], "defined": ev["defined"], "covered": ev["covered"], "secondary": sec, "raw": raw,
            "n01_comparator": n01, "le00_attribution": le00, "item_recovery": item_rec, "checks": chk,
            "flags": flags, "flagged": any(len(v) > 0 for v in flags.values()), "fits": fr,
            "items_1": {s: {"a": res1[(key, s)]["a"], "tau": res1[(key, s)]["tau"], "nll": res1[(key, s)]["nll"],
                            "converged": res1[(key, s)]["converged"]} for s in ("T", "P")},
            "items_2": {s: {k: rec["items_2"]["fits"][s][k] for k in ("a", "tau", "nll", "converged", "n_iter", "grad_norm")}
                        for s in ("T", "P")},
            "reported": {"psi1": rec["reported_1"], "psi2": rec["reported_2"]},
            "hessian": {"penalised_fd_evals": [rec["hess_1"]["penalised_fd_evals"], rec["hess_2"]["penalised_fd_evals"]]},
            "truth_swap": swap, "timing": timing, "host": platform.node(), "complete": True}


def control_dataset(ctx, r, sg, jobs, maxiter_s2=1000):
    """Report-only control arm: stage 2 with true items, FD Hessian, S (doc 47 §10.3)."""
    t = time.perf_counter()
    f = MQ.fit(sg, ctx.it_true, Q=Q, psi0=ctx.psi_default, jobs=jobs, maxiter=maxiter_s2)
    tf = time.perf_counter() - t
    o = MQ.Objective(sg, ctx.it_true, Q, jobs=jobs)
    o.state = dict(f.state)
    t = time.perf_counter()
    try:
        H = MQ.fd_hessian(o, f.psi)
    finally:
        o.close()
    th = time.perf_counter() - t
    sps, mom = ctx.reexpress(f.psi)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        se = np.full(19, np.nan)
    return {"r": r, "psi": f.psi, "S_psi": sps, "moments": mom, "se_raw": se, "shift_over_se": (sps - f.psi) / se,
            "converged": f.converged, "fit_s": tf, "hessian_s": th, "host": platform.node(), "complete": True}


# =====================================================================================================
#  Preflight
# =====================================================================================================
def preflight(ctx, freeze_file=None):
    I = {}
    locks = sorted(FROZEN.glob("*.sha256"))
    I["frozen_locks_all_ok"] = all(sha(ROOT / ln.split()[1]) == ln.split()[0] for f in locks for ln in f.read_text().splitlines() if ln.strip())
    I["plan_locks_present"] = all((FROZEN / v).exists() for v in PLAN_LOCKS.values())
    I["panel_md5"] = sha(ctx.cfg.path("outputs") / "visit_panel.csv", "md5") == ctx.P.PANEL_MD5
    s6 = json.loads((S6DIR / "summary.json").read_text())
    I["truth_equals_record"] = bool(np.array_equal(ctx.psi_true, np.array(s6["psi_true"])))
    if freeze_file is not None:
        rec = dict(ln.split()[::-1] for ln in Path(freeze_file).read_text().splitlines() if ln.strip())
        I["code_freeze_ok"] = all(sha(ROOT / p) == h for p, h in rec.items())
        I["code_freeze_files"] = len(rec)
    S_t, _ = ctx.reexpress(ctx.psi_true)
    I["S_truth_fixed_point"] = float(np.max(np.abs(S_t - ctx.psi_true)))
    ok = (I["frozen_locks_all_ok"] and I["plan_locks_present"] and I["panel_md5"] and I["truth_equals_record"]
          and I.get("code_freeze_ok", True))
    return ok, I


def provenance(args):
    git = lambda *x: subprocess.run(["git", *x], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa: E731
    return {"label": LABEL, "mode": args.mode, "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "cpu": next((ln.split(":", 1)[1].strip() for ln in subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines()
                         if ln.startswith("Model name")), "?"), "ncpu": os.cpu_count(), "jobs": args.jobs,
            "python": sys.version.split()[0], "numpy": np.__version__, "git_head": git("rev-parse", "HEAD"),
            "driver_sha256": sha(Path(__file__)), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}


def record_for_json(r):
    out = dict(r)
    for k in ("z", "defined", "covered", "est", "se"):
        out[k] = np.asarray(r[k]).tolist()
    return out


# =====================================================================================================
#  Modes
# =====================================================================================================
def run_smoke(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    ctx = Ctx()
    ok, I = preflight(ctx)
    S = {"NOT_A_RESULT": True, "provenance": provenance(a), "preflight": I}
    try:
        sg = ctx.subset(ctx.dataset([42, 45, 1, 0]), 60)
        res1, t1 = stage1_batch(ctx, {KEY_SMOKE: sg}, a.jobs)
        r = process_dataset(ctx, KEY_SMOKE, sg, res1, a.jobs, maxiter_s2=5, maxiter_it=20, n8=True, truth_swap=True,
                            out_npz=out / "smoke_matrices.npz")
        write_atomic(out / "smoke_dataset.json", record_for_json(r))
        z, d, c = (np.asarray(r[k])[None, :] for k in ("z", "defined", "covered"))
        S["verdict_codepath"] = verdict(None, None, z, d, c, [r["flagged"]])
        S["runtime_gate_codepath"] = runtime_gate(0.1, [100.0, 100.0, 100.0], 50.0, r["timing"]["fit_walls_s"], 0.1, 13.5)
        S["flags"] = r["flags"]
        S["truth_swap"] = r["truth_swap"]
        S["checks"] = {k: v for k, v in r["checks"].items() if k != "N9"}
        S["stage1_s"] = t1
    except Exception as e:  # noqa: BLE001
        S["error"] = f"{type(e).__name__}: {e}"
        S["trace"] = traceback.format_exc()[-4000:]
    S["runtime_s"] = time.time() - t0
    write_atomic(out / "summary.json", S)
    log(f"smoke done in {S['runtime_s']:.0f}s; error {S.get('error')}")
    return 0


def run_phase1(a):
    out = Path(a.out)
    (out / "main").mkdir(parents=True, exist_ok=False)
    (out / "regression").mkdir()
    t_job = time.time()
    ctx = Ctx()
    S = {"provenance": provenance(a), "label": LABEL}
    ok, I = preflight(ctx, a.freeze)
    S["preflight"] = I
    stop = None
    if not ok:
        stop = "INTEGRITY STOP: preflight"
    if not stop and I["S_truth_fixed_point"] > TOL_ID:
        stop = "INTEGRITY STOP: S(psi_true) != psi_true"
    try:
        if not stop:
            regen = {}
            for r in (0, R_TOTAL - 1):
                h1, h2 = ctx.responses_hash(ctx.dataset([42, 49, 1, r])), ctx.responses_hash(ctx.dataset([42, 49, 1, r]))
                regen[r] = {"sha256": h1, "identical": h1 == h2}
            S["regeneration"] = regen
            if not all(v["identical"] for v in regen.values()):
                stop = "INTEGRITY STOP: regeneration"
        if not stop:
            data = {KEY_OLD0: ctx.dataset([42, 45, 1, 0])}
            for r in range(3):
                data[r] = ctx.dataset([42, 49, 1, r])
            res1, T_stage1 = stage1_batch(ctx, data, a.jobs)
            S["T_stage1_s"], S["n_batch"] = T_stage1, len(data)
            log(f"stage-1 batch {T_stage1:.0f}s")
            # ---- regression R-a..R-d on old dataset 0 ----
            rg = process_dataset(ctx, KEY_OLD0, data[KEY_OLD0], res1, a.jobs, out_npz=out / "regression/old_dataset0_matrices.npz")
            d6 = json.loads((S6DIR / "S1_dataset0.json").read_text())
            import pandas as pd
            L7 = pd.read_csv(DIAG / "d7_readonly/D7_estimates_long.csv", float_precision="round_trip")
            x7 = L7[(L7.dataset == 0) & (L7.estimate == "psi2_step6")].set_index("param").loc[PSI_NAMES, "std"].to_numpy()
            d4 = json.loads((DIAG / "d4/D4_dataset0.json").read_text())
            seD4 = np.array(d4["se_psi2"]["full_chain"])
            reg = {"R_a_psi1_bitwise": bool(np.array_equal(rg["psi1"], np.array(d6["stage2_fit_1"]["psi"]))),
                   "R_a_psi2_bitwise": bool(np.array_equal(rg["psi2"], np.array(d6["stage2_fit_2"]["psi"]))),
                   "R_b_max_abs": float(np.max(np.abs(rg["S_psi2"] - x7))),
                   "R_c_max_rel": float(np.max(np.abs(rg["raw"]["se_fullchain_raw"] / seD4 - 1))),
                   "R_d_N3_max": max(rg["checks"]["N3"].values()), "R_d_N4_max": max(rg["checks"]["N4"].values()),
                   "H1_equals_step6": bool(np.array_equal(np.load(out / "regression/old_dataset0_matrices.npz")["H1"],
                                                          np.array(d6["H1_nonbinding"]))),
                   "H2_equals_step6": bool(np.array_equal(np.load(out / "regression/old_dataset0_matrices.npz")["H2"],
                                                          np.array(d6["H2"])))}
            reg["passed"] = (reg["R_a_psi1_bitwise"] and reg["R_a_psi2_bitwise"] and reg["R_b_max_abs"] <= 1e-12
                             and reg["R_c_max_rel"] <= 1e-6 and reg["R_d_N3_max"] <= TOL_N34 and reg["R_d_N4_max"] <= TOL_N34)
            S["regression"] = reg
            write_atomic(out / "regression/old_dataset0.json", record_for_json(rg))
            log(f"regression: {reg}")
            fit_walls = list(rg["timing"]["fit_walls_s"])
            if not reg["passed"]:
                stop = "INTEGRITY STOP: regression"
        if not stop:
            t_r, fh = [], []
            for r in range(3):
                key = r
                rr = process_dataset(ctx, key, data[key], res1, a.jobs, n8=True, truth_swap=(r == 0),
                                     out_npz=out / f"main/dataset_r{r:02d}_matrices.npz")
                write_atomic(out / f"main/dataset_r{r:02d}.json", record_for_json(rr))
                t_r.append(T_stage1 / len(data) + rr["timing"]["T_r_excl_n8_swap_s"])
                fit_walls += rr["timing"]["fit_walls_s"]
                fh += [w + h for w, h in zip(rr["timing"]["fit_walls_s"], rr["timing"]["hessian_walls_s"])]
                log(f"fresh r{r:02d}: flags {rr['flags']}; T_r {t_r[-1]:.0f}s")
                if r == 0:
                    S["truth_swap"] = rr["truth_swap"]
                    if not rr["truth_swap"]["passed"]:
                        stop = "INTEGRITY STOP: truth-swap"
                        break
            if not stop:
                U1 = (time.time() - t_job) / 3600
                L2 = min(L2_MAX_H, (CAP_NH - (a.smoke_nh + U1)) / 2)
                gate = runtime_gate(U1, t_r, float(np.mean(fh)), fit_walls, a.smoke_nh, L2)
                S["runtime_gate"] = gate
                if not gate["passed"]:
                    stop = "INCOMPLETE: runtime gate"
    except Exception as e:  # noqa: BLE001
        stop = f"INCOMPLETE: execution error {type(e).__name__}: {e}"
        S["trace"] = traceback.format_exc()[-4000:]
    S["stop"] = stop
    S["phase1_passed"] = stop is None
    S["runtime_s"] = time.time() - t_job
    write_atomic(out / "summary.json", S)
    log(f"phase 1 done: stop={stop}; {S['runtime_s']:.0f}s")
    return 0


def run_phase2(a):
    out = Path(a.out)
    (out / "main").mkdir(parents=True, exist_ok=True)
    (out / "control").mkdir(parents=True, exist_ok=True)
    t_job = time.time()
    ctx = Ctx()
    S = {"provenance": provenance(a), "label": LABEL, "main_range": [a.main_from, a.main_to],
         "control_range": [a.control_from, a.control_to]}
    ok, I = preflight(ctx, a.freeze)
    S["preflight"] = I
    mains = list(range(a.main_from, a.main_to + 1))
    controls = list(range(a.control_from, a.control_to + 1))
    existing = [str(p) for p in [out / f"main/dataset_r{r:02d}.json" for r in mains] + [out / f"control/control_r{r:02d}.json" for r in controls] if p.exists()]
    stop = None
    if not ok or I["S_truth_fixed_point"] > TOL_ID:
        stop = "INTEGRITY STOP: preflight"
    elif existing:
        stop = f"refusing to start: existing outputs {existing[:3]}"
    S["stop"] = stop
    write_atomic(out / "job_start.json", S, overwrite=True)
    if stop:
        return 0
    data = {r: ctx.dataset([42, 49, 1, r]) for r in mains}
    try:
        res1, T1 = stage1_batch(ctx, data, a.jobs)
        S["T_stage1_s"] = T1
        log(f"stage-1 batch for {len(data)} datasets: {T1:.0f}s")
    except Exception as e:  # noqa: BLE001
        S["stop"] = f"INCOMPLETE: stage-1 batch error {e}"
        write_atomic(out / "job_end.json", S, overwrite=True)
        return 0
    for r in mains:
        key = r
        try:
            rr = process_dataset(ctx, key, data[key], res1, a.jobs, out_npz=out / f"main/dataset_r{r:02d}_matrices.npz")
            write_atomic(out / f"main/dataset_r{r:02d}.json", record_for_json(rr))
            log(f"main r{r:02d}: flags {[k for k, v in rr['flags'].items() if v]}; {rr['timing']['pipeline_s']:.0f}s pipeline")
        except Exception as e:  # noqa: BLE001
            write_atomic(out / f"main/dataset_r{r:02d}.error.json", {"key": key, "error": f"{type(e).__name__}: {e}",
                                                                    "trace": traceback.format_exc()[-4000:], "complete": False})
            S["stop"] = f"INCOMPLETE: execution error in main r{r:02d}"
            log(S["stop"])
            write_atomic(out / "job_end.json", S, overwrite=True)
            return 0
    S["main_done_s"] = time.time() - t_job
    for r in controls:
        try:
            cr = control_dataset(ctx, r, ctx.dataset([42, 49, 1, r]), a.jobs)
            write_atomic(out / f"control/control_r{r:02d}.json", cr)
            log(f"control r{r:02d}: shift/SE max {np.nanmax(np.abs(cr['shift_over_se'])):.2f}")
        except Exception as e:  # noqa: BLE001  (report-only; never affects the main verdict)
            write_atomic(out / f"control/control_r{r:02d}.error.json", {"r": r, "error": f"{type(e).__name__}: {e}", "complete": False})
    S["runtime_s"] = time.time() - t_job
    write_atomic(out / "job_end.json", S, overwrite=True)
    log(f"phase 2 job done in {S['runtime_s']:.0f}s")
    return 0


def run_evaluate(a):
    """Post-processing only (no fitting): gates, verdict, labels, comparators, control arm, tables, figures."""
    import pandas as pd
    run = Path(a.run)
    out = run / "evaluation"
    out.mkdir(exist_ok=False)
    p1 = json.loads((run / "phase1/summary.json").read_text())
    recs, missing = {}, []
    for r in range(R_TOTAL):
        cands = [run / ph / f"main/dataset_r{r:02d}.json" for ph in ("phase1", "phase2a", "phase2b")]
        found = [c for c in cands if c.exists()]
        if found:
            recs[r] = json.loads(found[0].read_text())
        else:
            missing.append(r)
    integrity = p1["stop"] if (p1.get("stop") or "").startswith("INTEGRITY") else None
    exec_fail = None
    if p1.get("stop") and not integrity:
        exec_fail = p1["stop"]
    for ph in ("phase2a", "phase2b"):
        je = run / ph / "job_end.json"
        if je.exists():
            st = json.loads(je.read_text()).get("stop")
            if st and st.startswith("INTEGRITY"):
                integrity = integrity or st
            elif st:
                exec_fail = exec_fail or st
    if missing:
        exec_fail = exec_fail or f"{len(missing)} main datasets incomplete: {missing[:10]}"
    S = {"label": LABEL, "n_complete": len(recs), "missing": missing}
    order = sorted(recs)
    if order:
        z = np.array([recs[r]["z"] for r in order], float)
        d = np.array([recs[r]["defined"] for r in order], bool)
        c = np.array([recs[r]["covered"] for r in order], bool)
        fl = np.array([recs[r]["flagged"] for r in order], bool)
    else:
        z = d = c = fl = None
    V = verdict(integrity, exec_fail, z, d, c, fl) if order else verdict(integrity, exec_fail or "no datasets", None, None, None, [])
    S["verdict"] = V
    if order:
        S["flagged_datasets"] = {int(r): {k: v for k, v in recs[r]["flags"].items() if v} for r in order if recs[r]["flagged"]}
        rows = []
        for g, name in enumerate(GATE_NAMES):
            zd = z[d[:, g], g]
            n = zd.size
            sdz = float(np.std(zd, ddof=1)) if n > 1 else float("nan")
            ci = (sdz * np.sqrt((n - 1) / chi2.ppf(0.975, n - 1)), sdz * np.sqrt((n - 1) / chi2.ppf(0.025, n - 1))) if n > 1 else (np.nan, np.nan)
            sec = [recs[r]["secondary"]["covered"][g] for r in order if recs[r]["secondary"]["covered"] is not None]
            row = {"gate": name, "role": "recovery" if g < 21 else "guard", "covered_A": int(c[:, g].sum()), "n_A": len(order),
                   "mean_z_A": float(np.mean(zd)) if n else float("nan"), "sd_z": sdz, "sd_z_ci_lo": ci[0], "sd_z_ci_hi": ci[1],
                   "undefined_A": int((~d[:, g]).sum()), "status_A": V.get("status_A", {}).get(name),
                   "status_AF": V.get("status_AF", {}).get(name), "secondary_covered": int(np.sum(sec)), "secondary_n": len(sec),
                   "mean_est": float(np.nanmean([recs[r]["est"][g] if recs[r]["est"][g] is not None else np.nan for r in order]))}
            if V.get("status_A", {}).get(name) == "fail":
                lab, lo = failure_label(z[:, g], d[:, g], c[:, g])
                row["failure_label"], row["sd_z_lower_for_label"] = lab, lo
            rows.append(row)
        G_df = pd.DataFrame(rows)
        G_df.to_csv(out / "gates.csv", index=False)
        # historical D3 comparator (datasets 0-4) and Rev 3 gates on datasets 0-4
        d3 = [r for r in order if r < 5]
        if len(d3) == 5:
            cov = np.array([recs[r]["raw"]["covered_D3"] for r in d3], bool).sum(0)
            S["historical_D3_comparator"] = {"counts": dict(zip(PSI_NAMES, cov.tolist())), "met": bool(np.all(cov >= 4))}
            cov3 = c[[order.index(r) for r in d3]].sum(0)
            S["rev3_gates_on_datasets_0_4"] = dict(zip(GATE_NAMES, cov3.tolist()))
        # N(0,1)-only comparator, raw, checks, timing
        n01 = np.array([recs[r]["n01_comparator"]["z"] for r in order if "z" in recs[r]["n01_comparator"]], float)
        if n01.size:
            S["n01_comparator"] = {"mean_z": dict(zip(PSI_NAMES, n01.mean(0).round(3).tolist())),
                                   "covered": dict(zip(PSI_NAMES, (np.abs(n01) <= Z95).sum(0).tolist()))}
        rawz = np.array([recs[r]["raw"]["z_fixed_item"] for r in order], float)
        S["raw_fixed_item"] = {"mean_z": dict(zip(PSI_NAMES, np.nanmean(rawz, 0).round(3).tolist())),
                               "covered": dict(zip(PSI_NAMES, (np.abs(rawz) <= Z95).sum(0).tolist()))}
        S["checks_max"] = {k: float(np.nanmax([max(recs[r]["checks"][k].values()) for r in order])) for k in ("N3", "N4")}
        S["checks_max"]["N6"] = float(np.nanmax([recs[r]["checks"]["N6"] for r in order]))
        S["checks_max"]["N1"] = float(np.nanmax([max(recs[r]["checks"]["N1"].values()) for r in order]))
        S["checks_max"]["idempotence"] = float(np.nanmax([recs[r]["checks"]["idempotence"] for r in order]))
        S["checks_max"]["relabel_invariance"] = float(np.nanmax([recs[r]["checks"]["relabel_invariance"] for r in order]))
        S["checks_max"]["N10"] = float(np.nanmax([recs[r]["checks"]["N10_annihilation_max"] for r in order]))
        S["N8"] = {int(r): recs[r]["checks"].get("N8") for r in order if "N8" in recs[r]["checks"]}
        le = [recs[r]["le00_attribution"] for r in order if "pred_total" in recs[r]["le00_attribution"]]
        if le:
            S["le00_attribution_mean"] = {k: float(np.mean([x[k] for x in le])) for k in ("pred_total", "pred_T_items", "pred_P_items",
                                                                                         "pred_affine", "pred_nonaffine")}
        S["item_recovery_mean"] = {s: {k: float(np.mean([recs[r]["item_recovery"][s][k] for r in order])) for k in ("a_corr", "a_maxerr", "tau_maxerr")}
                                   for s in ("T", "P")}
        S["moments"] = {k: [float(np.min([recs[r]["moments"][k] for r in order])), float(np.max([recs[r]["moments"][k] for r in order]))]
                        for k in ("c_T", "d_T", "c_P", "d_P")}
        S["hosts"] = sorted({recs[r]["host"] for r in order})
        S["convergence"] = {"fit_2_not_converged": sum(not recs[r]["fits"]["fit_1"]["converged"] for r in order),
                            "fit_4_not_converged": sum(not recs[r]["fits"]["fit_2"]["converged"] for r in order)}
    # control arm (report only)
    ctrl = []
    for ph in ("phase2a", "phase2b"):
        for f in sorted((run / ph / "control").glob("control_r*.json")):
            ctrl.append(json.loads(f.read_text()))
    if ctrl:
        sh = np.array([x["shift_over_se"] for x in ctrl], float)
        S["control_arm"] = {"n": len(ctrl), "mean_shift_over_se": dict(zip(PSI_NAMES, np.nanmean(sh, 0).round(3).tolist())),
                            "max_abs_shift_over_se": float(np.nanmax(np.abs(sh))),
                            "moments_range": {k: [float(min(x["moments"][k] for x in ctrl)), float(max(x["moments"][k] for x in ctrl))]
                                              for k in ("c_T", "d_T", "c_P", "d_P")}}
    write_atomic(out / "summary.json", S)
    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import beta as betad
        if order:
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.boxplot([z[d[:, g], g] for g in range(len(GATE_NAMES))], showfliers=True)
            ax.axhline(1.96, ls="--", c="k", lw=0.7); ax.axhline(-1.96, ls="--", c="k", lw=0.7); ax.axhline(0, c="grey", lw=0.5)
            ax.axvline(21.5, c="r", lw=0.8)
            ax.set_xticks(range(1, len(GATE_NAMES) + 1)); ax.set_xticklabels(GATE_NAMES, rotation=80, fontsize=7)
            ax.set_ylabel("z"); ax.set_title(f"Rev 3 recovery test: z by gate (left of red: 21 recovery gates; right: 15 guards) — verdict {V['verdict']}")
            fig.tight_layout(); fig.savefig(out / "z_by_gate.png", dpi=130); plt.close(fig)
            fig, ax = plt.subplots(figsize=(13, 4.5))
            k = c.sum(0); n = c.shape[0]
            lo = [betad.ppf(0.025, x, n - x + 1) if x > 0 else 0 for x in k]; hi = [betad.ppf(0.975, x + 1, n - x) if x < n else 1 for x in k]
            ax.errorbar(range(len(GATE_NAMES)), k / n, yerr=[k / n - np.array(lo), np.array(hi) - k / n], fmt="o", ms=4)
            ax.axhline(0.95, c="grey", lw=0.6); ax.axhline(COV_FRAC, c="r", ls="--", lw=0.8)
            ax.set_xticks(range(len(GATE_NAMES))); ax.set_xticklabels(GATE_NAMES, rotation=80, fontsize=7)
            ax.set_ylabel("coverage (CP 95 %)"); ax.set_title("Coverage per gate on A (red: 0.88 criterion)")
            fig.tight_layout(); fig.savefig(out / "coverage_by_gate.png", dpi=130); plt.close(fig)
            fig, axs = plt.subplots(1, 2, figsize=(9, 4))
            for ax, s in zip(axs, ("T", "P")):
                ax.scatter([recs[r]["moments"][f"c_{s}"] for r in order], [recs[r]["moments"][f"d_{s}"] for r in order], s=10)
                ax.set_xlabel(f"c_{s} = sqrt(v_{s})"); ax.set_ylabel(f"d_{s} = mu_{s}"); ax.set_title(f"re-expression of psi-hat(2), {s}")
            fig.tight_layout(); fig.savefig(out / "cd_distribution.png", dpi=130); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        S["figure_error"] = str(e)
        write_atomic(out / "summary.json", S, overwrite=True)
    files = sorted(p for p in out.rglob("*") if p.is_file())
    (out / "MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files))
    log(f"evaluation: {V['verdict']} {V.get('reasons')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("smoke", "phase1", "phase2", "evaluate"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--run", default=None, help="run directory (evaluate)")
    ap.add_argument("--jobs", type=int, default=JOBS)
    ap.add_argument("--freeze", default=None, help="code-freeze record (sha256sum format)")
    ap.add_argument("--smoke-nh", dest="smoke_nh", type=float, default=0.0, help="smoke-run node-hours used (sacct)")
    ap.add_argument("--main-from", dest="main_from", type=int)
    ap.add_argument("--main-to", dest="main_to", type=int)
    ap.add_argument("--control-from", dest="control_from", type=int)
    ap.add_argument("--control-to", dest="control_to", type=int)
    a = ap.parse_args()
    return {"smoke": run_smoke, "phase1": run_phase1, "phase2": run_phase2, "evaluate": run_evaluate}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
