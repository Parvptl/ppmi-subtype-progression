"""Addendum A execution: tests T2-T4, T6-T7 and benchmarks (synthetic data only).

Specification (frozen): docs/cluster_phase/25_STAGE2_AMENDMENT1_REV2_ADDENDUM_A_FINAL_20261017.md
§3-§4. Not Step 3: no real item data are fitted; no V2 on pilot patients, order
checks, recovery, gates, calibration or evaluation statistics. Q is not selected.

Stages (each writes <out>/<stage>.json; failures are recorded, never converted):
  S1  T4(i)   60-patient subset, serial: reference v2 path vs updated path
  S2  T3      the reference path's penalty points: penalty, reason, time ratio <= 0.10
  S3  T4(ii)  full cohort, pool 32: reference v2 path (must reproduce doc 23) vs updated
  S4  T2      reference v2 vs updated at ψ_true, the doc 23 optimum, 3 perturbations
  S5  microbenchmarks (updated): Laplace Q 7/10/15/20, serial and pool
  S6  primary serial complete fit (updated) + reported evaluations at the optima
  S7  fallback (AGQ): warm G=3 x3, cold G=3, warm G=5 x3, FD Hessian per optimum
  S8  projections and runtime gates
Stopping rules (§3): T2 failure skips S5-S8; T4 failure labels S6 non-equivalent
and runs S7 only if the updated Laplace optimum equals reference v2's within 1e-6.

    scripts/cluster/submit.sh stage2_m2a1_addendumA_bench --partition=normal -w node002 --time=1-00:00:00 -- \
        scripts/48c_m2a1_addendum_a_benchmark.py --out <execution dir>/bench --jobs 32
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
from scipy.optimize import minimize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import m2_grm as M  # noqa: E402
from ppmi_subtype import m2_quad as MQ  # noqa: E402
from ppmi_subtype import m2_quad_reference_v2 as REF2  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

FROZEN = ROOT / "docs/cluster_phase/_frozen"
LOCKS = {"doc16": "16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.sha256",
         "rev2": "21_STAGE2_DOC16_AMENDMENT1_REV2.sha256",
         "addendumA_final": "25_STAGE2_ADDENDUM_A_FINAL.sha256"}
REF2_SHA = "23c2fde7afbccc4ace5f72ed0ee069866a9e45ca86d51962ca0e4c1aeb06fae8"
DOC23_NLL, DOC23_EVALS = 57022.74225348176, 104
CODE = ["src/ppmi_subtype/m2_quad.py", "src/ppmi_subtype/m2_quad_reference_v2.py", "src/ppmi_subtype/m2_grm.py",
        "src/ppmi_subtype/grm.py", "scripts/45_m2_pilot.py", "scripts/48c_m2a1_addendum_a_benchmark.py",
        "tests/test_m2_quad_addendum_a.py"]
OPTS = {"maxiter": 1000, "ftol": 1e-15, "gtol": 1e-7, "maxfun": 4000}
LO = np.array([b[0] for b in M.BOUNDS])
HI = np.array([b[1] for b in M.BOUNDS])
G_REPORTED = 5                     # declared fallback G for reported evaluations (§2 P2; not a selection)
OBS = {"stage1_item_fits_s": 2 * 358.0, "assumed_s1b_and_checks_h": 0.62}


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
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=1, default=conv))


def drive(obj, psi0, max_seconds=None, snapshot=False):
    """L-BFGS-B exactly as m2_quad.fit / reference fit, logging every evaluation."""
    psi0 = np.clip(np.asarray(psi0, float), LO, HI)
    path, t0 = [], time.time()

    class Stop(Exception):
        pass

    def f(p):
        if max_seconds is not None and time.time() - t0 > max_seconds:
            raise Stop
        st = copy.deepcopy(obj.state) if snapshot else None
        t = time.perf_counter()
        v, g = obj(p)
        path.append({"psi": p.copy(), "value": float(v), "grad_norm": float(np.linalg.norm(g)),
                     "wall_s": time.perf_counter() - t, "state_before": st})
        return v, g
    try:
        res = minimize(f, psi0, jac=True, method="L-BFGS-B", bounds=M.BOUNDS, options=OPTS)
        out = {"psi": res.x, "nll": float(res.fun), "converged": bool(res.success), "message": str(res.message),
               "n_iter": int(res.nit), "grad_norm": float(np.linalg.norm(res.jac)), "capped": False}
    except Stop:
        out = {"psi": path[-1]["psi"], "nll": path[-1]["value"], "converged": False, "message": "capped",
               "n_iter": -1, "grad_norm": path[-1]["grad_norm"], "capped": True}
    out.update(n_eval=len(path), wall_s=time.time() - t0, path=path,
               at_bound=[nm for nm, v, (a, b) in zip(M.NAMES, out["psi"], M.BOUNDS)
                         if abs(v - a) < 1e-6 or abs(v - b) < 1e-6])
    return out


def compare_paths(ref, new):
    n = min(len(ref["path"]), len(new["path"]))
    dpsi = [float(np.max(np.abs(ref["path"][i]["psi"] - new["path"][i]["psi"]))) for i in range(n)]
    first_div = next((i for i, d in enumerate(dpsi) if d > 1e-12), None)
    bad_triggers = []
    for i in range(n):
        rv, nv = ref["path"][i]["value"], new["path"][i]["value"]
        if nv >= 1e12 and rv < 1e12:
            bad_triggers.append({"eval": i + 1, "psi": new["path"][i]["psi"].tolist(), "ref_value": rv,
                                 "new_value": nv})
    res = {"n_eval_ref": len(ref["path"]), "n_eval_new": len(new["path"]), "max_dpsi_over_path": max(dpsi) if dpsi else None,
           "first_divergence_eval": None if first_div is None else first_div + 1,
           "final_dpsi": float(np.max(np.abs(ref["psi"] - new["psi"]))), "final_dnll": abs(ref["nll"] - new["nll"]),
           "converged_ref": ref["converged"], "converged_new": new["converged"],
           "penalty_evals_ref": [i + 1 for i, e in enumerate(ref["path"]) if e["value"] >= 1e12],
           "penalty_evals_new": [i + 1 for i, e in enumerate(new["path"]) if e["value"] >= 1e12],
           "triggers_where_reference_finite": bad_triggers}
    res["passed"] = bool(res["n_eval_ref"] == res["n_eval_new"] and first_div is None
                         and res["final_dpsi"] <= 1e-12 and res["final_dnll"] <= 1e-12
                         and res["converged_ref"] == res["converged_new"] and not bad_triggers)
    return res


def strip(fit):
    out = {k: v for k, v in fit.items() if k != "path"}
    out["path"] = [{k: v for k, v in e.items() if k != "state_before"} for e in fit["path"]]
    out["eval_wall_s"] = {"median": float(np.median([e["wall_s"] for e in fit["path"]])),
                          "max": float(np.max([e["wall_s"] for e in fit["path"]])),
                          "sum": float(np.sum([e["wall_s"] for e in fit["path"]]))}
    return out


def timed(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"median_s": float(np.median(ts)), "min_s": float(min(ts)), "max_s": float(max(ts)), "reps": reps}


def rel_ok(a, b, tol=1e-12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.array_equal(a, b):
        return "bitwise", True
    d = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
    return f"rel {d:.2e}", d <= tol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--smoke", action="store_true", help="code path only: 40-patient cohort, tiny fits")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t_all = time.time()
    locks = {k: (FROZEN / v).read_text().split() for k, v in LOCKS.items()}
    prov = {"spec": "docs/cluster_phase/25_STAGE2_AMENDMENT1_REV2_ADDENDUM_A_FINAL_20261017.md",
            "docs": {k: {"file": v[1], "frozen": v[0], "current": sha(ROOT / v[1])} for k, v in locks.items()},
            "ref2_sha256": sha(ROOT / "src/ppmi_subtype/m2_quad_reference_v2.py"), "ref2_expected": REF2_SHA,
            "code_sha256": {f: sha(ROOT / f) for f in CODE}, "host": platform.node(), "jobs": a.jobs,
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "smoke": a.smoke,
            "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "cpu": next((ln.split(":", 1)[1].strip() for ln in subprocess.run(
                ["lscpu"], capture_output=True, text=True).stdout.splitlines() if ln.startswith("Model name")), "?"),
            "budget": {"MAX_NONND": MQ.MAX_NONND, "MAX_HALVINGS": MQ.MAX_HALVINGS}, "G_reported": G_REPORTED}
    jdump(out / "S0_provenance.json", prov)
    if any(d["frozen"] != d["current"] for d in prov["docs"].values()) or prov["ref2_sha256"] != REF2_SHA:
        log("hash mismatch: stop")
        return 2

    spec = importlib.util.spec_from_file_location("m2_pilot", ROOT / "scripts/45_m2_pilot.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    v, times, treat, YTd, YPd, *_ = P.load_cohort(load_config())      # design only
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
    if a.smoke:
        full = subset(full, 40)
        global OPTS
        OPTS = {**OPTS, "maxiter": 5, "maxfun": 12}
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
        except Exception as e:                                      # recorded, never hidden
            S["stages"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                 "trace": traceback.format_exc()[-3000:]}
            jdump(out / f"{name}.json", S["stages"][name])
            log(f"{name}: ERROR {e}")
            return None

    # ---- S1: T4(i) subset serial -------------------------------------------------------
    def s1():
        ref = drive(REF2.Objective(sub60, it, 10), M.default_psi(), snapshot=True)
        new_obj = MQ.Objective(sub60, it, 10)
        new = drive(new_obj, M.default_psi())
        cmp_ = compare_paths(ref, new)
        S["_sub_ref"] = ref
        return {"test": "T4(i)", **cmp_, "ref": strip(ref), "new": strip(new), "new_monitor_log": new_obj.log}
    r1 = stage("S1_T4_subset", s1)

    # ---- S2: T3 extreme points ------------------------------------------------------------
    def s2():
        ref = S["_sub_ref"]
        pts = [i for i, e in enumerate(ref["path"]) if e["value"] >= 1e12]
        rows = []
        for i in pts:
            e = ref["path"][i]
            ro = REF2.Objective(sub60, it, 10)
            ro.state = copy.deepcopy(e["state_before"])
            t = time.perf_counter()
            rv = ro(e["psi"])[0]
            rt = time.perf_counter() - t
            no = MQ.Objective(sub60, it, 10)
            no.state = copy.deepcopy(e["state_before"])
            rep: dict = {}
            t = time.perf_counter()
            nv = no(e["psi"], report=rep)[0]
            nt = time.perf_counter() - t
            rows.append({"eval": i + 1, "ref_value": rv, "new_value": nv, "ref_s": rt, "new_s": nt,
                         "ratio": nt / rt, "reasons": rep.get("reasons"), "monitor": no.log[-1],
                         "passed": bool(nv >= 1e12 and rv >= 1e12 and nt <= 0.10 * rt)})
        return {"test": "T3", "points": rows, "n_points": len(rows),
                "passed": bool(rows) and all(r["passed"] for r in rows)}
    stage("S2_T3_extreme", s2) if "_sub_ref" in S else None

    # ---- S3: T4(ii) full cohort pooled ----------------------------------------------------------
    def s3():
        ro = REF2.Objective(full, it, 10, jobs=a.jobs)
        try:
            ref = drive(ro, M.default_psi())
        finally:
            ro.close()
        no = MQ.Objective(full, it, 10, jobs=a.jobs)
        try:
            new = drive(no, M.default_psi())
        finally:
            no.close()
        cmp_ = compare_paths(ref, new)
        doc23 = {"nll_matches": abs(ref["nll"] - DOC23_NLL) <= 1e-4, "evals_match": ref["n_eval"] == DOC23_EVALS}
        S["_full_ref"], S["_full_new"], S["_full_new_state"] = ref, new, dict(no.state)
        rep = MQ.evaluate_reported(new["psi"], full, it, 10, G_REPORTED, init_state=no.state)
        return {"test": "T4(ii)", **cmp_, "reference_reproduces_doc23": doc23, "ref": strip(ref), "new": strip(new),
                "new_monitor_log": no.log, "reported_at_new_optimum": rep}
    r3 = stage("S3_T4_full_pool", s3)

    # ---- S4: T2 -----------------------------------------------------------------------------
    def s4():
        opt = S["_full_ref"]["psi"]
        pts = {"psi_true": psi_true, "doc23_optimum": opt}
        for k in (1, 2, 3):
            pts[f"perturb_{k}"] = np.clip(opt + np.random.default_rng([42, 46, 7, k]).normal(0, 0.05, M.N_PSI), LO, HI)
        rows = []
        for name, p in pts.items():
            fs0, gs0 = REF2.negloglik(p, full, it, 10)
            fs1, gs1 = MQ.negloglik(p, full, it, 10)
            ro, no = REF2.Objective(full, it, 10, jobs=a.jobs), MQ.Objective(full, it, 10, jobs=a.jobs)
            try:
                fp0, gp0 = ro(p)
                fp1, gp1 = no(p)
            finally:
                ro.close()
                no.close()
            cs_v, ok1 = rel_ok(fs1, fs0)
            cs_g, ok2 = rel_ok(gs1, gs0)
            cp_v, ok3 = rel_ok(fp1, fp0)
            cp_g, ok4 = rel_ok(gp1, gp0)
            rows.append({"point": name, "serial_value": cs_v, "serial_grad": cs_g, "pool_value": cp_v,
                         "pool_grad": cp_g, "penalty_ref": fs0 >= 1e12, "penalty_new": fs1 >= 1e12,
                         "passed": ok1 and ok2 and ok3 and ok4})
        return {"test": "T2", "points": rows, "passed": all(r["passed"] for r in rows)}
    r4 = stage("S4_T2", s4) if "_full_ref" in S else None
    t2_ok = bool(r4 and r4["passed"])
    t4_ok = bool(r1 and r1["passed"] and r3 and r3["passed"])
    S["gates"]["T2_passed"], S["gates"]["T4_passed"] = t2_ok, t4_ok
    if not t2_ok:
        S["stopped"] = "T2 failed or not run: S5-S8 skipped (Addendum A §3)"
        log(S["stopped"])
        return finish(out, S, t_all, prov)

    # ---- S5: microbenchmarks ---------------------------------------------------------------------
    def s5():
        rows = {}
        for jobs, reps in ((1, 3), (a.jobs, 5)):
            for Q in (7, 10, 15, 20):
                o = MQ.Objective(full, it, Q, jobs=jobs)
                try:
                    rows[f"{'serial' if jobs == 1 else f'pool{jobs}'}_Q{Q}_lik"] = timed(lambda: o(psi_true, want_grad=False), reps)
                    rows[f"{'serial' if jobs == 1 else f'pool{jobs}'}_Q{Q}_lik_grad"] = timed(lambda: o(psi_true), reps)
                finally:
                    o.close()
        return {"timings": rows, "passed": None}
    r5 = stage("S5_micro", s5)

    # ---- S6: primary serial complete fit + reported evaluation --------------------------------
    def s6():
        o = MQ.Objective(full, it, 10)
        f = drive(o, M.default_psi())
        rep = MQ.evaluate_reported(f["psi"], full, it, 10, G_REPORTED, init_state=o.state)
        pen = {}
        for e in o.log:
            for k2, c in e["reasons"].items():
                pen[k2] = pen.get(k2, 0) + c
        return {"fit": strip(f), "monitor_log": o.log, "penalty_reasons_total": pen,
                "reported_at_optimum": rep, "equivalence_label": "equivalent" if t4_ok else "NON-EQUIVALENT (T4 failed)",
                "passed": None}
    r6 = stage("S6_primary_serial_fit", s6)

    # ---- S7: fallback ------------------------------------------------------------------------------
    fb_allowed = t4_ok or (r3 is not None and r3.get("final_dpsi", 1) <= 1e-6)
    S["gates"]["fallback_sequence_allowed"] = fb_allowed

    def fb_fit(G, psi0, init_state, label):
        f = MQ.fit(full, it, Q=10, method="agq", Gn=G, psi0=psi0, jobs=a.jobs, init_state=init_state,
                   max_seconds=3 * 3600, maxiter=OPTS["maxiter"])
        o = MQ.Objective(full, it, 10, "agq", G, jobs=a.jobs)
        o.state = dict(f.state)
        t = time.perf_counter()
        try:
            H = MQ.fd_hessian(o, f.psi)
        finally:
            o.close()
        h_s = time.perf_counter() - t
        eig = np.linalg.eigvalsh(H)
        wall = sum(e["wall_s"] for e in f.log)
        crit = {"grad_lt_0.934": f.grad_norm < 1e-3 * 934, "no_bound_hits": not f.at_bound,
                "hessian_pd": bool(eig.min() > 0), "converged_flag": f.converged, "not_capped": not f.stopped_by_time}
        return {"label": label, "G": G, "nll": f.nll, "psi": f.psi, "n_eval": f.n_eval, "n_iter": f.n_iter,
                "grad_norm": f.grad_norm, "at_bound": f.at_bound, "message": f.message,
                "fit_wall_s_sum_evals": wall, "hessian_s": h_s, "hessian_min_eig": float(eig.min()),
                "criteria": crit, "meets_F1_4": all(crit.values()),
                "penalties": sum(1 for e in f.log if e["penalty"]), "log": f.log}

    def s7():
        psiL = S["_full_new"]["psi"]
        stL = S["_full_new_state"]
        res = {"laplace_start": psiL}
        for G in (3, 5):
            starts = [psiL] + [np.clip(psiL + np.random.default_rng([42, 46, 8, k]).normal(0, 0.2, M.N_PSI), LO, HI)
                               for k in (1, 2)]
            fits = []
            for k, s0 in enumerate(starts, 1):
                t = time.time()
                r = fb_fit(G, s0, stL, f"warm_G{G}_start{k}")
                r["wall_s"] = time.time() - t
                fits.append(r)
                log(f"fallback G{G} start {k}: nll {r['nll']:.4f}, evals {r['n_eval']}, F1(4) {r['meets_F1_4']}, "
                    f"{r['wall_s']:.0f}s")
                jdump(out / f"S7_fallback_G{G}_start{k}.json", r)
            nlls = [r["nll"] for r in fits]
            res[f"G{G}"] = {"nll": nlls, "spread": max(nlls) - min(nlls),
                            "T7_passed": (max(nlls) - min(nlls) <= 0.1) and all(r["meets_F1_4"] for r in fits),
                            "per_start": [{k: r[k] for k in ("label", "nll", "n_eval", "grad_norm", "at_bound",
                                                             "hessian_min_eig", "meets_F1_4", "criteria", "wall_s",
                                                             "fit_wall_s_sum_evals", "hessian_s", "penalties")}
                                          for r in fits]}
            if G == 3:
                t = time.time()
                cold = fb_fit(3, M.default_psi(), None, "cold_G3")
                cold["wall_s"] = time.time() - t
                jdump(out / "S7_fallback_G3_cold.json", cold)
                res["G3_cold"] = {k: cold[k] for k in ("nll", "n_eval", "grad_norm", "at_bound", "hessian_min_eig",
                                                       "meets_F1_4", "criteria", "wall_s", "fit_wall_s_sum_evals",
                                                       "hessian_s", "penalties")}
                res["T6_passed"] = (abs(cold["nll"] - fits[0]["nll"]) <= 0.1 and cold["meets_F1_4"]
                                    and fits[0]["meets_F1_4"])
                res["T6_dnll"] = abs(cold["nll"] - fits[0]["nll"])
                log(f"T6: |dnll| {res['T6_dnll']:.4f}, passed {res['T6_passed']}")
        res["passed"] = bool(res.get("T6_passed") and res["G3"]["T7_passed"] and res["G5"]["T7_passed"])
        return res
    r7 = stage("S7_fallback", s7) if fb_allowed and "_full_new" in S else None
    if not fb_allowed:
        S["stages"]["S7_fallback"] = {"ok": False, "skipped": "T4 failed and the Laplace optimum differs (§3)"}

    # ---- S8: projections and gates -------------------------------------------------------------
    def s8():
        tm = r5["timings"]
        pool_fit_s = S["_full_new"]["wall_s"]
        lg10 = tm[f"pool{a.jobs}_Q10_lik_grad"]["median_s"]
        other = OBS["stage1_item_fits_s"] + OBS["assumed_s1b_and_checks_h"] * 3600
        prim = {}
        for Q in (7, 10, 15, 20):
            lg = tm[f"pool{a.jobs}_Q{Q}_lik_grad"]["median_s"]
            fit_s = pool_fit_s * lg / lg10
            tot = 18 * fit_s + 11 * 38 * lg + other
            prim[f"Q{Q}"] = {"fit_s": fit_s, "fit_basis": "measured" if Q == 10 else "scaled estimate",
                             "lik_grad_s": lg, "projected_node_hours": tot / 3600,
                             "gate_fit_le_60min": fit_s <= 3600, "gate_pilot_le_24": tot / 3600 <= 24}
        fbp = {}
        if r7:
            for G in (3, 5):
                ps = r7[f"G{G}"]["per_start"]
                fit_mean = float(np.mean([p["fit_wall_s_sum_evals"] for p in ps]))
                hess = float(np.mean([p["hessian_s"] for p in ps]))
                tot = 18 * (pool_fit_s + fit_mean) + 11 * hess + other
                fbp[f"G{G}"] = {"warm_fit_mean_s": fit_mean, "warm_fit_max_s": max(p["fit_wall_s_sum_evals"] for p in ps),
                                "hessian_mean_s": hess, "projected_node_hours": tot / 3600,
                                "gate_fit_le_60min": max(p["fit_wall_s_sum_evals"] for p in ps) <= 3600,
                                "gate_pilot_le_24": tot / 3600 <= 24}
        return {"accounting": "node-hours = 1 exclusive node x wall hours at 32 processes; other components: "
                              "stage-1 item fits observed 716 s, S1-b refits and checks assumed 0.62 h",
                "primary": prim, "fallback": fbp, "passed": None}
    if r5 is not None:
        r8 = stage("S8_projections", s8)
        if r8:
            S["gates"]["primary"] = r8["primary"]
            S["gates"]["fallback"] = r8["fallback"]
    return finish(out, S, t_all, prov)


def finish(out, S, t_all, prov):
    S = {k: v for k, v in S.items() if not k.startswith("_")}
    S["runtime_s"] = time.time() - t_all
    S["provenance"] = {k: prov[k] for k in ("host", "cpu", "jobs", "slurm_job", "budget", "G_reported")}
    jdump(out / "summary.json", S)
    log(f"done in {S['runtime_s']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
