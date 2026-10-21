"""Exact-gradient AGQ fallback: in-job checks and synthetic benchmarks (doc 27 §5 C2/C3, E, D; §6).

Specification (frozen): docs/cluster_phase/27_STAGE2_EXACT_AGQ_GRADIENT_AMENDMENT_PROPOSED_20261018.md.
Synthetic dataset 0 only (items at truth). Not Step 3: no real item data fitted, no V2,
order selection, recovery, gates, calibration or evaluation statistics.

  S1  C2/C3 on the full cohort: Laplace bitwise vs reference v3 (serial, pool) at ψ_true and
      2 perturbations; agq_exact value bitwise == agq value
  S2  primary Laplace fit (pool 32) -> ψ̂_L and modes (must reproduce the doc 23 optimum)
  S3  gate E: exact AGQ gradient vs central FD (h = 1e-4) of the moving-node objective,
      60-patient subset, G = 3, Q = 10, at ψ̂_L; <= 1e-5 max(1, |fd|). Failure stops S5.
  S4  timings: agq_exact likelihood and likelihood + gradient, G = 3, 5 (pool 32)
  S5  T6 / T7 with agq_exact: G = 3 warm x3 + cold, G = 5 warm x3; FD Hessian per optimum
  S6  projections and runtime gates

    scripts/cluster/submit.sh ... -w node002 --time=1-00:00:00 -- scripts/48d_m2_exact_agq_benchmark.py \
        --out <execution dir>/bench --jobs 32
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import m2_grm as M  # noqa: E402
from ppmi_subtype import m2_quad as MQ  # noqa: E402
from ppmi_subtype import m2_quad_reference_v3 as REF3  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

FROZEN = ROOT / "docs/cluster_phase/_frozen"
LOCKS = {"doc16": "16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.sha256", "rev2": "21_STAGE2_DOC16_AMENDMENT1_REV2.sha256",
         "doc25": "25_STAGE2_ADDENDUM_A_FINAL.sha256", "doc27": "27_STAGE2_EXACT_AGQ_GRADIENT.sha256"}
REF3_SHA = "33c02d6a4cd4fd688ff60fb8ea7f6ae47c9e142c1f6045ea50754b605fec65a7"
DOC23_NLL = 57022.74225348176
CODE = ["src/ppmi_subtype/m2_quad.py", "src/ppmi_subtype/m2_quad_reference_v3.py", "src/ppmi_subtype/m2_grm.py",
        "src/ppmi_subtype/grm.py", "scripts/45_m2_pilot.py", "scripts/48d_m2_exact_agq_benchmark.py",
        "tests/test_m2_quad_exact_agq_gradient.py"]
LO = np.array([b[0] for b in M.BOUNDS])
HI = np.array([b[1] for b in M.BOUNDS])
OTHER_S = 716.0 + 0.62 * 3600          # stage-1 item fits (observed) + S1-b refits and checks (assumed)
TOL_E = 1e-5


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jdump(path, obj):
    def conv(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=1, default=conv))


def bit_or_rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.array_equal(a, b):
        return "bitwise", True
    d = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
    return f"rel {d:.2e}", False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--smoke", action="store_true", help="code path only: 40 patients, short fits")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t_all = time.time()
    locks = {k: (FROZEN / v).read_text().split() for k, v in LOCKS.items()}
    prov = {"spec": "docs/cluster_phase/27_STAGE2_EXACT_AGQ_GRADIENT_AMENDMENT_PROPOSED_20261018.md",
            "docs": {k: {"file": v[1], "frozen": v[0], "current": sha(ROOT / v[1])} for k, v in locks.items()},
            "ref3_sha256": sha(ROOT / "src/ppmi_subtype/m2_quad_reference_v3.py"),
            "code_sha256": {f: sha(ROOT / f) for f in CODE}, "host": platform.node(), "jobs": a.jobs,
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "smoke": a.smoke,
            "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "numpy": np.__version__, "python": sys.version.split()[0],
            "cpu": next((ln.split(":", 1)[1].strip() for ln in subprocess.run(
                ["lscpu"], capture_output=True, text=True).stdout.splitlines() if ln.startswith("Model name")), "?")}
    jdump(out / "S0_provenance.json", prov)
    if any(d["frozen"] != d["current"] for d in prov["docs"].values()) or prov["ref3_sha256"] != REF3_SHA:
        log("hash mismatch: stop")
        return 2

    spec = importlib.util.spec_from_file_location("m2_pilot", ROOT / "scripts/45_m2_pilot.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    v, times, treat, YTd, YPd, *_ = P.load_cohort(load_config())       # design only
    g0 = M.make_groups(times, treat, YTd, YPd)
    psi_true, it, _ = P.truth_on_stage1_scale(g0)
    full = P.regroup(g0, *P.simulate_groups(np.random.default_rng([42, 45, 1, 0]), g0, psi_true, it))

    def subset(groups, k_tot):
        keep, acc = {}, 0
        for n, g in groups.items():
            k = min(len(g.patients), max(0, k_tot - acc))
            if k:
                keep[n] = M.Group(g.T[:k], g.X[:k], g.YT[:k], g.YP[:k], g.patients[:k])
                acc += k
        return keep
    sub60 = subset(full, 60)
    maxiter = 1000
    if a.smoke:
        full, maxiter = subset(full, 40), 4
    S: dict = {"stages": {}, "gates": {}}

    def stage(name, fn):
        t0 = time.time()
        try:
            res = fn()
            res["stage_wall_s"] = time.time() - t0
            jdump(out / f"{name}.json", res)
            S["stages"][name] = {"ok": True, "wall_s": res["stage_wall_s"], "passed": res.get("passed")}
            log(f"{name}: passed={res.get('passed')} ({res['stage_wall_s']:.0f}s)")
            return res
        except Exception as e:
            S["stages"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
            jdump(out / f"{name}.json", S["stages"][name])
            log(f"{name}: ERROR {e}")
            return None

    # ---- S1: C2 / C3 -----------------------------------------------------------------------------
    def s1():
        pts = {"psi_true": psi_true}
        for k in (1, 2):
            pts[f"perturb_{k}"] = np.clip(psi_true + np.random.default_rng([42, 47, 1, k]).normal(0, 0.05, M.N_PSI), LO, HI)
        rows = []
        for name, p in pts.items():
            f0, g0_ = REF3.negloglik(p, full, it, 10)
            f1, g1_ = MQ.negloglik(p, full, it, 10)
            r0, r1 = REF3.Objective(full, it, 10, jobs=a.jobs), MQ.Objective(full, it, 10, jobs=a.jobs)
            try:
                p0, pg0 = r0(p)
                p1, pg1 = r1(p)
            finally:
                r0.close()
                r1.close()
            res = [bit_or_rel(f1, f0), bit_or_rel(g1_, g0_), bit_or_rel(p1, p0), bit_or_rel(pg1, pg0)]
            rows.append({"point": name, "laplace_serial_value": res[0][0], "laplace_serial_grad": res[1][0],
                         "laplace_pool_value": res[2][0], "laplace_pool_grad": res[3][0],
                         "passed": all(x[1] for x in res)})
        oa, oe = MQ.Objective(full, it, 10, "agq", 3, jobs=a.jobs), MQ.Objective(full, it, 10, "agq_exact", 3, jobs=a.jobs)
        try:
            va, ve = oa(psi_true, want_grad=False), oe(psi_true, want_grad=False)
        finally:
            oa.close()
            oe.close()
        c3 = bit_or_rel(ve, va)
        return {"test": "C2/C3", "points": rows, "agq_exact_value_vs_agq": c3[0],
                "passed": all(r["passed"] for r in rows) and c3[1]}
    r1 = stage("S1_C2_C3", s1)

    # ---- S2: primary Laplace fit ------------------------------------------------------------------
    def s2():
        f = MQ.fit(full, it, Q=10, jobs=a.jobs, maxiter=maxiter)
        S["_L"] = f
        pen = sum(1 for e in f.log if e["penalty"])
        return {"nll": f.nll, "n_eval": f.n_eval, "converged": f.converged, "grad_norm": f.grad_norm,
                "wall_s": sum(e["wall_s"] for e in f.log), "penalty_evals": pen, "psi": f.psi,
                "reproduces_doc23": abs(f.nll - DOC23_NLL) <= 1e-4, "passed": None}
    r2 = stage("S2_laplace_fit", s2)

    # ---- S3: gate E --------------------------------------------------------------------------------
    def s3():
        psiL = S["_L"].psi if not a.smoke else psi_true
        st0: dict = {}
        MQ.negloglik(psiL, sub60, it, 10, "agq", 3, state=st0, want_grad=False)
        f, g = MQ.negloglik(psiL, sub60, it, 10, "agq_exact", 3, state=copy.deepcopy(st0))
        _, gfix = MQ.negloglik(psiL, sub60, it, 10, "agq", 3, state=copy.deepcopy(st0))
        h = 1e-4
        fd = np.array([(MQ.negloglik(psiL + h * e, sub60, it, 10, "agq", 3, state=copy.deepcopy(st0), want_grad=False)
                        - MQ.negloglik(psiL - h * e, sub60, it, 10, "agq", 3, state=copy.deepcopy(st0), want_grad=False))
                       / (2 * h) for e in np.eye(M.N_PSI)])
        scale = np.maximum(1.0, np.abs(fd))
        err, errf = np.abs(g - fd) / scale, np.abs(gfix - fd) / scale
        return {"test": "E", "h": h, "max_rel_err_exact": float(err.max()), "max_rel_err_fixed": float(errf.max()),
                "per_param_exact": dict(zip(M.NAMES, err.tolist())), "per_param_fixed": dict(zip(M.NAMES, errf.tolist())),
                "grad_norm_exact": float(np.linalg.norm(g)), "grad_norm_fixed": float(np.linalg.norm(gfix)),
                "tolerance": TOL_E, "passed": bool(err.max() <= TOL_E)}
    r3 = stage("S3_gate_E", s3) if "_L" in S else None
    S["gates"]["E_passed"] = bool(r3 and r3["passed"])

    # ---- S4: timings ---------------------------------------------------------------------------------
    def s4():
        rows = {}
        for G in (3, 5):
            o = MQ.Objective(full, it, 10, "agq_exact", G, jobs=a.jobs)
            if "_L" in S:
                o.state = {k: np.array(v) for k, v in S["_L"].state.items() if k in o.chunks}
            try:
                for kind, fn in (("lik", lambda: o(psi_true, want_grad=False)), ("lik_grad", lambda: o(psi_true))):
                    fn()
                    ts = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        fn()
                        ts.append(time.perf_counter() - t0)
                    rows[f"agq_exact_G{G}_{kind}"] = {"median_s": float(np.median(ts)), "min_s": min(ts), "max_s": max(ts)}
            finally:
                o.close()
        return {"timings": rows, "passed": None}
    r4 = stage("S4_timings", s4)

    # ---- S5: fallback T6 / T7 --------------------------------------------------------------------------
    def fb_fit(G, psi0, init_state, label):
        t0 = time.time()
        f = MQ.fit(full, it, Q=10, method="agq_exact", Gn=G, psi0=psi0, jobs=a.jobs, init_state=init_state,
                   max_seconds=3 * 3600, maxiter=maxiter)
        fit_s = sum(e["wall_s"] for e in f.log)
        o = MQ.Objective(full, it, 10, "agq_exact", G, jobs=a.jobs)
        o.state = dict(f.state)
        try:
            rep: dict = {}
            v_opt, g_opt = o(f.psi, report=rep)
            inexact = o.log[-1]["gradient_inexact_patients"]
            t = time.perf_counter()
            H = MQ.fd_hessian(o, f.psi)
            h_s = time.perf_counter() - t
        finally:
            o.close()
        eig = float(np.linalg.eigvalsh(H).min())
        crit = {"grad_lt_0.934": f.grad_norm < 0.934, "no_bound_hits": not f.at_bound, "hessian_pd": eig > 0,
                "no_inexact_patients": inexact == 0, "not_capped": not f.stopped_by_time}
        pen = {}
        for e in f.log:
            for k, c in e["reasons"].items():
                pen[k] = pen.get(k, 0) + c
        r = {"label": label, "G": G, "nll": f.nll, "psi": f.psi, "n_eval": f.n_eval, "n_iter": f.n_iter,
             "grad_norm": f.grad_norm, "grad_norm_reeval": float(np.linalg.norm(g_opt)), "at_bound": f.at_bound,
             "message": f.message, "optimiser_success": f.converged, "fit_s": fit_s, "hessian_s": h_s,
             "hessian_min_eig": eig, "gradient_inexact_patients_at_opt": inexact,
             "non_nd_at_opt": rep.get("not_negative_definite"), "penalty_reasons": pen,
             "max_gradient_inexact_over_fit": max((e.get("gradient_inexact_patients", 0) for e in f.log), default=0),
             "criteria": crit, "meets_F1_4": all(crit.values()), "wall_s": time.time() - t0, "log": f.log}
        jdump(out / f"S5_{label}.json", r)
        log(f"{label}: nll {r['nll']:.4f}, evals {r['n_eval']}, |g| {r['grad_norm']:.4f}, F1(4) {r['meets_F1_4']}, "
            f"fit {fit_s:.0f}s, hess {h_s:.0f}s")
        return {k: v for k, v in r.items() if k not in ("log", "psi")}

    def s5():
        psiL, stL = S["_L"].psi, S["_L"].state
        res = {}
        for G in (3, 5):
            starts = [psiL] + [np.clip(psiL + np.random.default_rng([42, 46, 8, k]).normal(0, 0.2, M.N_PSI), LO, HI)
                               for k in (1, 2)]
            fits = [fb_fit(G, s0, stL, f"warm_G{G}_start{k}") for k, s0 in enumerate(starts, 1)]
            nll = [f["nll"] for f in fits]
            res[f"G{G}"] = {"per_start": fits, "spread": max(nll) - min(nll),
                            "T7_passed": (max(nll) - min(nll) <= 0.1) and all(f["meets_F1_4"] for f in fits)}
            if G == 3:
                cold = fb_fit(3, M.default_psi(), None, "cold_G3")
                res["G3_cold"] = cold
                res["T6_dnll"] = abs(cold["nll"] - fits[0]["nll"])
                res["T6_passed"] = res["T6_dnll"] <= 0.1 and cold["meets_F1_4"] and fits[0]["meets_F1_4"]
        res["passed"] = bool(res["T6_passed"] and res["G3"]["T7_passed"] and res["G5"]["T7_passed"])
        return res
    r5 = None
    if S["gates"]["E_passed"]:
        r5 = stage("S5_fallback_T6_T7", s5)
    else:
        S["stopped"] = "gate E failed or not run: fallback benchmark sequence stopped (doc 27 §5 E)"
        log(S["stopped"])

    # ---- S6: projections ---------------------------------------------------------------------------
    def s6():
        lap_s = r2["wall_s"]
        proj = {}
        for G in (3, 5):
            ps = r5[f"G{G}"]["per_start"]
            fit_mean = float(np.mean([p["fit_s"] for p in ps]))
            fit_max = float(max(p["fit_s"] for p in ps))
            hess = float(np.mean([p["hessian_s"] for p in ps]))
            tot = 18 * (lap_s + fit_mean) + 11 * hess + OTHER_S
            proj[f"G{G}"] = {"laplace_fit_s": lap_s, "warm_fit_mean_s": fit_mean, "warm_fit_max_s": fit_max,
                             "hessian_mean_s": hess, "projected_node_hours": tot / 3600,
                             "gate_fit_le_60min": fit_max <= 3600, "gate_pilot_le_24": tot / 3600 <= 24}
        return {"accounting": "1 exclusive node x wall hours at 32 processes; other components 716 s observed "
                              "+ 0.62 h assumed", "fallback": proj, "passed": None}
    if r5 is not None and r2 is not None:
        r6 = stage("S6_projections", s6)
        if r6:
            S["gates"]["fallback_runtime"] = r6["fallback"]
    S["gates"]["C2_C3_passed"] = bool(r1 and r1["passed"])
    if r5:
        S["gates"].update(T6=r5["T6_passed"], T7_G3=r5["G3"]["T7_passed"], T7_G5=r5["G5"]["T7_passed"])
    S["runtime_s"] = time.time() - t_all
    S["provenance"] = {k: prov[k] for k in ("host", "cpu", "jobs", "slurm_job", "numpy", "python")}
    jdump(out / "summary.json", {k: v for k, v in S.items() if not k.startswith("_")})
    log(f"done in {S['runtime_s']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
