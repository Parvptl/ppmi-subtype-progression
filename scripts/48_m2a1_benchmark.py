"""B0 benchmark for the Amendment 1 (Rev 2) prototype (§4.2). Timing only.

Approved for Step 2 only (prototype, unit tests, B0). Uses synthetic dataset 0 of
the pilot's recovery design (seed [42, 45, 1, 0], a-priori truth on the stage-1
scale, real visit design of 934 patients, items at their generating values). No
real item responses are fitted; no quadrature-order check, recovery, gate, label
or evaluation statistic is computed.

Measured (single-threaded BLAS, warm start, median of 3 unless one call exceeds
`--slow`): likelihood and likelihood + gradient for the Laplace-over-b primary at
Q = 7, 10, 15, 20; the AGQ-over-b fallback at G = 3, 5 (Q = 10); the current
`m2_grm` Laplace for comparison; one complete fit (primary, Q = 10, the default
pending the not-yet-authorised §4.1 order check), capped at 60 min.

    scripts/cluster/submit.sh stage2_m2a1_benchmark --partition=normal --time=06:00:00 -- \
        scripts/48_m2a1_benchmark.py --out @RUN_DIR@/results
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
from ppmi_subtype.config import load_config  # noqa: E402

FROZEN = ROOT / "docs/cluster_phase/_frozen"
LOCKS = {"doc16": FROZEN / "16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.sha256",
         "amendment1_rev2": FROZEN / "21_STAGE2_DOC16_AMENDMENT1_REV2.sha256"}
CODE = ["src/ppmi_subtype/m2_quad.py", "src/ppmi_subtype/m2_grm.py", "src/ppmi_subtype/grm.py",
        "scripts/45_m2_pilot.py", "scripts/48_m2a1_benchmark.py", "tests/test_m2_quad.py"]
FIT_CAP_S = 3600
# Projection inputs observed in earlier runs (doc 17 pilot, doc 18) [O]
OBS = {"stage1_item_fits_real_3starts_x2_s": 358.0, "pilot_laplace_check_30pat_s": 17.0,
       "pilot_sampler_30pat_s": 208.0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timed(fn, reps, slow):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
        if ts[-1] > slow:
            break
    return {"median_s": float(np.median(ts)), "all_s": ts, "reps": len(ts)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="code path only: 60 patients, 1 rep, 120 s fit cap")
    ap.add_argument("--slow", type=float, default=600.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t_start = time.time()
    locks = {k: v.read_text().split() for k, v in LOCKS.items()}
    prov = {"purpose": "B0 timing benchmark (Rev 2 §4.2); not a validation", "smoke": a.smoke,
            "doc_sha256": {k: {"file": v[1], "frozen": v[0], "current": sha(ROOT / v[1])} for k, v in locks.items()},
            "code_sha256": {f: sha(ROOT / f) for f in CODE}, "host": platform.node(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "cpu": next((ln.split(":", 1)[1].strip() for ln in subprocess.run(
                ["lscpu"], capture_output=True, text=True).stdout.splitlines() if ln.startswith("Model name")), "?")}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    if any(d["frozen"] != d["current"] for d in prov["doc_sha256"].values()):
        print("document hash mismatch", file=sys.stderr)
        return 2

    spec = importlib.util.spec_from_file_location("m2_pilot", ROOT / "scripts/45_m2_pilot.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    v, times, treat, YTd, YPd, *_ = P.load_cohort(load_config())     # design only
    groups0 = M.make_groups(times, treat, YTd, YPd)
    psi_true, it_true, _ = P.truth_on_stage1_scale(groups0)
    sg = P.regroup(groups0, *P.simulate_groups(np.random.default_rng([42, 45, 1, 0]), groups0, psi_true, it_true))
    if a.smoke:
        keep, acc = {}, 0
        for n, g in sg.items():
            k = min(len(g.patients), max(0, 60 - acc))
            if k:
                keep[n] = M.Group(g.T[:k], g.X[:k], g.YT[:k], g.YP[:k], g.patients[:k])
                acc += k
        sg = keep
    n_pat = sum(len(g.patients) for g in sg.values())
    n_vis = sum(g.T.size for g in sg.values())
    reps = 1 if a.smoke else 3
    res = {"n_patients": n_pat, "n_visits": n_vis, "timings": {}}

    def rec(name, fn, warm=True):
        if warm:
            fn()                                                     # warm-up (also warms modes)
        res["timings"][name] = timed(fn, reps, a.slow)
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {res['timings'][name]['median_s']:.2f}s", flush=True)

    # current m2_grm (2n-dimensional Laplace) on the same data, for comparison
    st0: dict = {}
    rec("current_m2grm_lik", lambda: M.negloglik(psi_true, sg, it_true, st0, want_grad=False))
    rec("current_m2grm_lik_grad", lambda: M.negloglik(psi_true, sg, it_true, st0))
    states = {}
    for Q in (7, 10, 15, 20):
        st: dict = {}
        states[Q] = st
        rep: dict = {}
        rec(f"laplace_Q{Q}_lik", lambda: MQ.negloglik(psi_true, sg, it_true, Q, state=st, want_grad=False, report=rep))
        rec(f"laplace_Q{Q}_lik_grad", lambda: MQ.negloglik(psi_true, sg, it_true, Q, state=st, report=rep))
        res["timings"][f"laplace_Q{Q}_lik_grad"]["not_negative_definite"] = rep.get("not_negative_definite")
    for Gn in (3, 5):                       # warm-started from the Laplace Q10 modes; no extra warm-up call
        st = dict(states[10])
        rec(f"agq_G{Gn}_Q10_lik", lambda: MQ.negloglik(psi_true, sg, it_true, 10, "agq", Gn, state=st,
                                                       want_grad=False), warm=False)
        rec(f"agq_G{Gn}_Q10_lik_grad", lambda: MQ.negloglik(psi_true, sg, it_true, 10, "agq", Gn, state=st),
            warm=False)

    # one complete fit: primary, Q = 10 (default; §4.1 not authorised), capped at 60 min
    cap = 120 if a.smoke else FIT_CAP_S
    t0 = time.time()
    f = MQ.fit(sg, it_true, Q=10, max_seconds=cap)
    res["fit_Q10"] = {"wall_s": time.time() - t0, "cap_s": cap, "stopped_by_time": f.stopped_by_time,
                      "converged": f.converged, "message": f.message, "n_iter": f.n_iter, "n_eval": f.n_eval,
                      "grad_norm": f.grad_norm, "nll": f.nll, "at_bound": f.at_bound}
    print(f"[{time.strftime('%H:%M:%S')}] fit Q10: {res['fit_Q10']['wall_s']:.0f}s, conv {f.converged}, "
          f"capped {f.stopped_by_time}", flush=True)

    # projected pilot cost (Rev 2 §4.2; §6 steps 3-7), serial on one node
    T = res["timings"]
    fit_s = res["fit_Q10"]["wall_s"]
    capped = res["fit_Q10"]["stopped_by_time"]
    proj = {"fit_capped": capped,
            "note": ("fit hit the cap: every fit-based figure is a LOWER BOUND" if capped else "fit completed")}
    for Q in (7, 10, 15, 20):
        scale = T[f"laplace_Q{Q}_lik_grad"]["median_s"] / T["laplace_Q10_lik_grad"]["median_s"]
        fit_q = fit_s * scale
        hess = 11 * 2 * M.N_PSI * T[f"laplace_Q{Q}_lik_grad"]["median_s"]
        comp = {"fits_18": 18 * fit_q, "hessians_11x38": hess,
                "stage1_item_fits_real_and_recovery": OBS["stage1_item_fits_real_3starts_x2_s"] * 2,
                "s1b_item_refits_5_assumed_5x_stage1": 5 * 5 * OBS["stage1_item_fits_real_3starts_x2_s"] / 6,
                "checks_v2_sampler_v5_assumed": 3 * (OBS["pilot_laplace_check_30pat_s"] + OBS["pilot_sampler_30pat_s"])
                + 30 * (7 / 5) ** 4 * T["agq_G5_Q10_lik"]["median_s"] / n_pat}
        tot = sum(comp.values())
        proj[f"Q{Q}"] = {"components_s": comp, "total_node_hours_serial": tot / 3600,
                         "fit_projected_s": fit_q, "lower_bound": capped}
    res["projection"] = proj
    res["fit_complete"] = not res["fit_Q10"]["stopped_by_time"]
    res["criterion"] = {"fit_le_60min": bool(res["fit_complete"] and fit_s <= 3600),
                        "pilot_le_24_node_hours_Q10": bool((not capped) and proj["Q10"]["total_node_hours_serial"] <= 24)}
    res["proceed_allowed_by_B0_rule"] = bool(all(res["criterion"].values())) and not a.smoke
    res["runtime_s"] = time.time() - t_start
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"criterion": res["criterion"], "fit_capped": capped, "projection_node_hours":
                      {k: round(v["total_node_hours_serial"], 2) for k, v in proj.items()
                       if isinstance(v, dict)}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
