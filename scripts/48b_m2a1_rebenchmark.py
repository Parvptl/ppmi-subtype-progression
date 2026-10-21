"""Re-benchmark of the optimised Amendment 1 prototype, and fallback feasibility.

Authorised 2026-10-17 (after the B0 review): code-only optimisation, re-benchmark,
separate investigation of the adaptive-quadrature (AGQ) fallback. Not Step 3: no
quadrature-order selection, no V2 on the pilot's patients, no recovery, no real
item data fitted, no gate, label or evaluation statistic. Final Q is NOT chosen here.

Part A (comparable to B0, scripts/48_m2a1_benchmark.py): synthetic dataset 0 of the
pilot design, items at truth, ψ = truth. Timings, warm start, median of 3 unless a
call exceeds --slow:
  * reference implementation (m2_quad_reference, = the B0 code) at Q = 10, same node;
  * optimised, serial (jobs = 1) and process pool (jobs = --jobs), Laplace over b at
    Q = 7, 10, 15, 20; AGQ fallback at G = 3, 5 (Q = 10);
  * one complete Laplace fit at Q = 10, serial and pooled, each capped at 60 min.
Part B (fallback feasibility, synthetic only): on 30 synthetic patients (indices drawn
with the V2 seed [42, 45, 4] from the synthetic cohort; not the pilot's real
patients), per-patient log p(Y_i) of the AGQ fallback at G = 3, 5, 7 (Q = 10) against
(i) G_ref = 7 (order tolerance 0.005) and (ii) the unchanged D3 reference
(m2_grm.agq_one when 2n <= 6, else m2_grm.is_one, 20,000 draws, t5; tolerance 0.05).
Reported as feasibility evidence only; it does not replace V2 or the §4.1 checks.

    scripts/cluster/submit.sh stage2_m2a1_rebenchmark --partition=normal -w node002 --time=06:00:00 -- \
        scripts/48b_m2a1_rebenchmark.py --out @RUN_DIR@/results --jobs 32
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import m2_grm as M  # noqa: E402
from ppmi_subtype import m2_quad as MQ  # noqa: E402
from ppmi_subtype import m2_quad_reference as REF  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

FROZEN = ROOT / "docs/cluster_phase/_frozen"
LOCKS = {"doc16": FROZEN / "16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.sha256",
         "amendment1_rev2": FROZEN / "21_STAGE2_DOC16_AMENDMENT1_REV2.sha256"}
B0 = ROOT / "experiments/20261016T223545Z_stage2_m2a1_benchmark/results"
CODE = ["src/ppmi_subtype/m2_quad.py", "src/ppmi_subtype/m2_quad_reference.py", "src/ppmi_subtype/m2_grm.py",
        "src/ppmi_subtype/grm.py", "scripts/45_m2_pilot.py", "scripts/48b_m2a1_rebenchmark.py",
        "tests/test_m2_quad.py"]
FIT_CAP_S = 3600
OBS = {"stage1_item_fits_real_3starts_x2_s": 358.0, "pilot_laplace_check_30pat_s": 17.0,
       "pilot_sampler_30pat_s": 208.0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timed(fn, reps, slow, warm=True):
    if warm:
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
        if ts[-1] > slow:
            break
    return {"median_s": float(np.median(ts)), "all_s": ts, "reps": len(ts)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def projection(fit_s, lg_s, n_eval_fit, label):
    """Rev 2 §6 steps 3-7, as in B0: 18 fits, 11 Hessians x 38 gradient evaluations, other items."""
    comp = {"fits_18": 18 * fit_s, "hessians_11x38": 11 * 2 * M.N_PSI * lg_s,
            "stage1_item_fits_observed": OBS["stage1_item_fits_real_3starts_x2_s"] * 2,
            "s1b_item_refits_assumed": 5 * 5 * OBS["stage1_item_fits_real_3starts_x2_s"] / 6,
            "checks_assumed": 3 * (OBS["pilot_laplace_check_30pat_s"] + OBS["pilot_sampler_30pat_s"])}
    return {"basis": label, "fit_s": fit_s, "lik_grad_s": lg_s, "evals_per_fit": n_eval_fit,
            "components_s": comp, "total_node_hours": sum(comp.values()) / 3600}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--slow", type=float, default=600.0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t_start = time.time()
    locks = {k: v.read_text().split() for k, v in LOCKS.items()}
    prov = {"purpose": "re-benchmark after code-only optimisation; fallback feasibility (synthetic); not Step 3",
            "smoke": a.smoke, "jobs": a.jobs,
            "doc_sha256": {k: {"file": v[1], "frozen": v[0], "current": sha(ROOT / v[1])} for k, v in locks.items()},
            "reference_equals_B0_code": sha(ROOT / "src/ppmi_subtype/m2_quad_reference.py")
            == json.loads((B0 / "provenance.json").read_text())["code_sha256"]["src/ppmi_subtype/m2_quad.py"],
            "code_sha256": {f: sha(ROOT / f) for f in CODE}, "host": platform.node(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "cpu": next((ln.split(":", 1)[1].strip() for ln in subprocess.run(
                ["lscpu"], capture_output=True, text=True).stdout.splitlines() if ln.startswith("Model name")), "?")}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    if any(d["frozen"] != d["current"] for d in prov["doc_sha256"].values()) or not prov["reference_equals_B0_code"]:
        print("hash mismatch", file=sys.stderr)
        return 2

    spec = importlib.util.spec_from_file_location("m2_pilot", ROOT / "scripts/45_m2_pilot.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    v, times, treat, YTd, YPd, *_ = P.load_cohort(load_config())       # design only
    g0 = M.make_groups(times, treat, YTd, YPd)
    psi, it, _ = P.truth_on_stage1_scale(g0)
    sg = P.regroup(g0, *P.simulate_groups(np.random.default_rng([42, 45, 1, 0]), g0, psi, it))
    if a.smoke:
        keep, acc = {}, 0
        for n, g in sg.items():
            k = min(len(g.patients), max(0, 60 - acc))
            if k:
                keep[n] = M.Group(g.T[:k], g.X[:k], g.YT[:k], g.YP[:k], g.patients[:k])
                acc += k
        sg = keep
    reps = 1 if a.smoke else 3
    res = {"n_patients": sum(len(g.patients) for g in sg.values()),
           "n_visits": sum(g.T.size for g in sg.values()), "timings": {}}
    Tm = res["timings"]

    # ---- Part A: timings -------------------------------------------------------------
    st: dict = {}
    Tm["reference_Q10_lik"] = timed(lambda: REF.negloglik(psi, sg, it, 10, state=st, want_grad=False), reps, a.slow)
    Tm["reference_Q10_lik_grad"] = timed(lambda: REF.negloglik(psi, sg, it, 10, state=st), reps, a.slow)
    log(f"reference Q10: {Tm['reference_Q10_lik']['median_s']:.2f} / {Tm['reference_Q10_lik_grad']['median_s']:.2f}s")
    for jobs in (1, a.jobs):
        tag = "serial" if jobs == 1 else f"pool{jobs}"
        for Q in (7, 10, 15, 20):
            obj = MQ.Objective(sg, it, Q, jobs=jobs)
            try:
                rep: dict = {}
                Tm[f"opt_{tag}_Q{Q}_lik"] = timed(lambda: obj(psi, want_grad=False), reps, a.slow)
                Tm[f"opt_{tag}_Q{Q}_lik_grad"] = timed(lambda: obj(psi, report=rep), reps, a.slow)
                Tm[f"opt_{tag}_Q{Q}_lik_grad"]["not_negative_definite"] = rep.get("not_negative_definite")
            finally:
                obj.close()
            log(f"opt {tag} Q{Q}: {Tm[f'opt_{tag}_Q{Q}_lik']['median_s']:.2f} / "
                f"{Tm[f'opt_{tag}_Q{Q}_lik_grad']['median_s']:.2f}s")
        for Gn in (3, 5):
            obj = MQ.Objective(sg, it, 10, "agq", Gn, jobs=jobs)
            try:
                lap = MQ.Objective(sg, it, 10, jobs=1)
                lap(psi, want_grad=False)                                   # warm modes from Laplace
                obj.state = {k: lap.state[k] for k in obj.chunks}
                Tm[f"opt_{tag}_agq_G{Gn}_lik"] = timed(lambda: obj(psi, want_grad=False), reps, a.slow, warm=False)
                Tm[f"opt_{tag}_agq_G{Gn}_lik_grad"] = timed(lambda: obj(psi), reps, a.slow, warm=False)
            finally:
                obj.close()
            log(f"opt {tag} AGQ G{Gn}: {Tm[f'opt_{tag}_agq_G{Gn}_lik']['median_s']:.2f} / "
                f"{Tm[f'opt_{tag}_agq_G{Gn}_lik_grad']['median_s']:.2f}s")
    res["fits"] = {}
    for jobs in (1, a.jobs):
        tag = "serial" if jobs == 1 else f"pool{jobs}"
        t0 = time.time()
        f = MQ.fit(sg, it, Q=10, max_seconds=120 if a.smoke else FIT_CAP_S, jobs=jobs)
        res["fits"][tag] = {"wall_s": time.time() - t0, "stopped_by_time": f.stopped_by_time,
                            "converged": f.converged, "message": f.message, "n_iter": f.n_iter,
                            "n_eval": f.n_eval, "grad_norm": f.grad_norm, "nll": f.nll, "at_bound": f.at_bound}
        log(f"fit {tag}: {res['fits'][tag]['wall_s']:.0f}s, conv {f.converged}, evals {f.n_eval}")

    # ---- projections (measured fit at Q10; other Q and AGQ scaled by measured lik+grad) ----
    proj = {}
    for tag in ("serial", f"pool{a.jobs}"):
        fq = res["fits"][tag]
        ne = fq["n_eval"]
        lg10 = Tm[f"opt_{tag}_Q10_lik_grad"]["median_s"]
        for Q in (7, 10, 15, 20):
            lg = Tm[f"opt_{tag}_Q{Q}_lik_grad"]["median_s"]
            measured = Q == 10 and not fq["stopped_by_time"]
            fit_s = fq["wall_s"] * lg / lg10
            proj[f"{tag}_laplace_Q{Q}"] = projection(fit_s, lg, ne, "measured fit" if measured
                                                     else "fit scaled by measured lik+grad ratio (estimate)")
        for Gn in (3, 5):
            lg = Tm[f"opt_{tag}_agq_G{Gn}_lik_grad"]["median_s"]
            proj[f"{tag}_agq_G{Gn}_Q10"] = projection(ne * lg, lg, ne,
                                                      "estimate: Laplace fit's evaluation count x measured AGQ lik+grad")
    res["projection"] = proj
    log("projections: " + json.dumps({k: round(v["total_node_hours"], 2) for k, v in proj.items()}))

    # ---- Part B: fallback accuracy feasibility (synthetic patients) ---------------------
    pats = [(n, r) for n, g in sg.items() for r in range(len(g.patients))]
    order = sorted(range(len(pats)), key=lambda i: (pats[i][0], pats[i][1]))
    pick = np.random.default_rng([42, 45, 4]).choice(len(order), min(30, len(order)), replace=False)
    rows = []
    t0 = time.time()
    for i in pick:
        n, r = pats[order[i]]
        g = sg[n]
        one = M.Group(g.T[r:r + 1], g.X[r:r + 1], g.YT[r:r + 1], g.YP[r:r + 1], [0])
        vals = {Gn: float(MQ.group_agq(psi, one, it, 10, Gn, want_grad=False)[0][0]) for Gn in (3, 5, 7)}
        t, x, yt, yp = g.T[r], g.X[r], g.YT[r], g.YP[r]
        if 2 * n <= 6:
            ref, se, meth = M.agq_one(t, x, yt, yp, psi, it, nodes=7), 0.0, "AGQ7"
        else:
            ref, se = M.is_one(t, x, yt, yp, psi, it, np.random.default_rng([42, 48, 2, int(i)]), draws=20000)
            meth = "IS20000_t5"
        rows.append({"index": int(i), "n_visits": int(n), **{f"agq_G{k}": vv for k, vv in vals.items()},
                     "d3_reference": ref, "d3_mc_se": se, "d3_method": meth})
    fb = {"patients": rows, "runtime_s": time.time() - t0}
    for Gn in (3, 5):
        fb[f"max_abs_G{Gn}_minus_G7"] = max(abs(r_[f"agq_G{Gn}"] - r_["agq_G7"]) for r_ in rows)
    for Gn in (3, 5, 7):
        fb[f"max_abs_G{Gn}_minus_D3"] = max(abs(r_[f"agq_G{Gn}"] - r_["d3_reference"]) for r_ in rows)
    fb["order_tol"], fb["accuracy_tol"] = 0.005, 0.05
    fb["max_d3_mc_se"] = max(r_["d3_mc_se"] for r_ in rows)
    res["fallback_feasibility"] = fb
    log("fallback: " + json.dumps({k: round(v, 4) for k, v in fb.items() if k.startswith("max")}))
    res["runtime_s"] = time.time() - t_start
    (out / "results.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
