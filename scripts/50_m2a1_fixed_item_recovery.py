"""Stage 2 M2, Step 4: fixed-item recovery (Rev 2 §6 step 4).

Frozen execution plan: docs/cluster_phase/31_STAGE2_STEP4_FIXED_ITEM_RECOVERY_PLAN_FINAL_20261019.md
(reviewer decisions in its §0). Synthetic data only: 5 datasets, seeds [42, 45, 1, r], a-priori
truth on the stage-1 scale (scripts/45_m2_pilot.py), real visit design; item parameters fixed at
truth. Primary model: Laplace over b, Q = 15 (Step 3), default start, one start per dataset.
95 % Wald intervals from the central finite-difference Hessian (h = 1e-4) of the exact gradient.
Criterion: every ψ component covered in >= 4 of 5 datasets; a non-PD Hessian -> nothing covered
for that dataset. Runtime gate after dataset 0 (§0.2). No real item responses are fitted.

    scripts/cluster/submit.sh ... -w node002 -- scripts/50_m2a1_fixed_item_recovery.py --out <exec dir>/step4 --jobs 32
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
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
from ppmi_subtype.config import load_config  # noqa: E402

FROZEN = ROOT / "docs/cluster_phase/_frozen"
LOCKS = {"doc16": "16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.sha256", "rev2": "21_STAGE2_DOC16_AMENDMENT1_REV2.sha256",
         "doc25": "25_STAGE2_ADDENDUM_A_FINAL.sha256", "doc27": "27_STAGE2_EXACT_AGQ_GRADIENT.sha256",
         "plan29": "29_STAGE2_STEP3_PLAN.sha256", "plan31": "31_STAGE2_STEP4_PLAN.sha256"}
M2QUAD_SHA = "50f164c6fc3e04b1d72e7726d87dfa7b5306f93bc590e494b6aca349797b093f"
IDENT = ROOT / "experiments/20261017T233307Z_stage2_m2_exact_agq/bench/S2_laplace_fit.json"
Q, G_REPORTED, N_DATA = 15, 5, 5
OTHER_S = 716.0 + 0.62 * 3600


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--smoke", action="store_true", help="code path only: 40 patients, 2 datasets, maxiter 3")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t_all = time.time()
    locks = {k: (FROZEN / v).read_text().split() for k, v in LOCKS.items()}
    psi_start = M.default_psi()
    prov = {"plan": "docs/cluster_phase/31_STAGE2_STEP4_FIXED_ITEM_RECOVERY_PLAN_FINAL_20261019.md",
            "docs": {k: {"file": v[1], "frozen": v[0], "current": sha(ROOT / v[1])} for k, v in locks.items()},
            "m2_quad_sha256": sha(ROOT / "src/ppmi_subtype/m2_quad.py"),
            "code_sha256": {f: sha(ROOT / f) for f in ("src/ppmi_subtype/m2_quad.py", "src/ppmi_subtype/m2_grm.py",
                                                       "src/ppmi_subtype/grm.py", "scripts/45_m2_pilot.py",
                                                       "scripts/50_m2a1_fixed_item_recovery.py")},
            "Q": Q, "G_reported": G_REPORTED, "start_psi": psi_start, "start_names": M.NAMES,
            "seeds": {r: [42, 45, 1, r] for r in range(N_DATA)}, "hessian": "central FD h=1e-4 of exact gradient",
            "host": platform.node(), "jobs": a.jobs, "slurm_job": os.environ.get("SLURM_JOB_ID"), "smoke": a.smoke,
            "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "numpy": np.__version__, "python": sys.version.split()[0],
            "cpu": next((ln.split(":", 1)[1].strip() for ln in subprocess.run(
                ["lscpu"], capture_output=True, text=True).stdout.splitlines() if ln.startswith("Model name")), "?")}
    jdump(out / "S0_provenance.json", prov)
    if any(d["frozen"] != d["current"] for d in prov["docs"].values()) or prov["m2_quad_sha256"] != M2QUAD_SHA:
        log("hash mismatch: nothing computed")
        return 2

    spec = importlib.util.spec_from_file_location("m2_pilot", ROOT / "scripts/45_m2_pilot.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    v, times, treat, YTd, YPd, *_ = P.load_cohort(load_config())    # design only
    g0 = M.make_groups(times, treat, YTd, YPd)
    psi_true, it_true, scale = P.truth_on_stage1_scale(g0)
    pilot_truth = np.array(json.loads((ROOT / "experiments/20261016T200350Z_stage2_m2_pilot/results/step1_recovery.json")
                                      .read_text())["psi_true"])
    assert np.array_equal(psi_true, pilot_truth), "truth differs from the pilot record"
    S: dict = {"psi_true": psi_true, "names": M.NAMES, "datasets": [], "gates": {}}
    sigma = lambda p: {"sigma_T": float(np.exp(p[16])), "sigma_P": float(np.sqrt(p[17] ** 2 + np.exp(2 * p[18])))}  # noqa: E731
    S["truth_sigma"] = sigma(psi_true)

    def dataset(r):
        return P.regroup(g0, *P.simulate_groups(np.random.default_rng([42, 45, 1, r]), g0, psi_true, it_true))

    def subset(groups, k_tot):
        keep, acc = {}, 0
        for n, g in groups.items():
            k = min(len(g.patients), max(0, k_tot - acc))
            if k:
                keep[n] = M.Group(g.T[:k], g.X[:k], g.YT[:k], g.YP[:k], g.patients[:k])
                acc += k
        return keep

    # ---- identity check of the synthetic data (plan §0) ---------------------------------------
    if not a.smoke:
        d0 = dataset(0)
        ref = json.loads(IDENT.read_text())
        # serial objective (cohort summation order of the recorded value's code path); the pooled
        # chunked sum differs in the last bit (run 1 defect, RUN1_DEFECT_NOTE.txt)
        nll = MQ.negloglik(np.array(ref["psi"]), d0, it_true, 10, want_grad=False)
        S["identity_check"] = {"nll": nll, "recorded": ref["nll"], "equal": nll == ref["nll"]}
        log(f"identity check: {nll} vs {ref['nll']} -> {nll == ref['nll']}")
        if nll != ref["nll"]:
            S["stopped"] = "identity check failed: synthetic data differ from the recorded ones"
            return finish(out, S, t_all, prov)

    rows = []
    n_data = 2 if a.smoke else N_DATA
    for r in range(n_data):
        t0 = time.time()
        try:
            sg = dataset(r)
            if a.smoke:
                sg = subset(sg, 40)
            f = MQ.fit(sg, it_true, Q=Q, psi0=psi_start, jobs=a.jobs, maxiter=3 if a.smoke else 1000)
            fit_s = sum(e["wall_s"] for e in f.log)
            o = MQ.Objective(sg, it_true, Q, jobs=a.jobs)
            o.state = dict(f.state)
            try:
                th = time.perf_counter()
                H = MQ.fd_hessian(o, f.psi)
                h_s = time.perf_counter() - th
            finally:
                o.close()
            eig = np.linalg.eigvalsh(H)
            pd = bool(eig.min() > 0)
            try:
                se = np.sqrt(np.diag(np.linalg.inv(H))) if pd else np.full(M.N_PSI, np.nan)
            except np.linalg.LinAlgError:
                se, pd = np.full(M.N_PSI, np.nan), False
            z = (f.psi - psi_true) / se
            covered = (np.abs(z) <= 1.96) & pd
            rep = MQ.evaluate_reported(f.psi, sg, it_true, Q, G_REPORTED, init_state=f.state)
            pen = {}
            for e in f.log:
                for k, c in e["reasons"].items():
                    pen[k] = pen.get(k, 0) + c
            d = {"dataset": r, "seed": [42, 45, 1, r], "nll": f.nll, "converged": f.converged, "message": f.message,
                 "n_iter": f.n_iter, "n_eval": f.n_eval, "grad_norm": f.grad_norm, "grad_norm_lt_0.934_reference_only":
                 f.grad_norm < 0.934, "at_bound": f.at_bound, "stopped_by_time": f.stopped_by_time,
                 "hessian_min_eig": float(eig.min()), "hessian_pd": pd, "psi_hat": f.psi, "se": se,
                 "lower": f.psi - 1.96 * se, "upper": f.psi + 1.96 * se, "z": z, "covered": covered,
                 "n_covered": int(covered.sum()), "sigma_hat": sigma(f.psi),
                 "penalty_evals": sum(1 for e in f.log if e["penalty"]), "penalty_reasons_chunks": pen,
                 "non_nd_at_accepted_evals": sum(e["n_non_nd"] for e in f.log if not e["penalty"]),
                 "reported_evaluation": rep, "fallback_invoked_at_optimum": rep["n_fallback_patients"] > 0,
                 "fit_s": fit_s, "hessian_s": h_s, "lik_grad_s_from_hessian": h_s / (2 * M.N_PSI),
                 "wall_s": time.time() - t0}
            jdump(out / f"S1_dataset{r}.json", {**d, "log": f.log})
            S["datasets"].append(d)
            for j, nm in enumerate(M.NAMES):
                rows.append({"dataset": r, "param": nm, "truth": psi_true[j], "estimate": f.psi[j], "se": se[j],
                             "lower": f.psi[j] - 1.96 * se[j], "upper": f.psi[j] + 1.96 * se[j], "z": z[j],
                             "covered": bool(covered[j]), "hessian_pd": pd})
            log(f"dataset {r}: covered {int(covered.sum())}/19, conv {f.converged}, |g| {f.grad_norm:.4g}, "
                f"PD {pd}, fit {fit_s:.0f}s, hess {h_s:.0f}s, fallback {rep['n_fallback_patients']}")
        except Exception as e:
            S["datasets"].append({"dataset": r, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]})
            log(f"dataset {r}: ERROR {e}")
            S["stopped"] = f"dataset {r} raised an execution error"
            break
        if r == 0 and not a.smoke:                       # runtime gate (plan §0.2)
            fit_s, lg = S["datasets"][0]["fit_s"], S["datasets"][0]["lik_grad_s_from_hessian"]
            proj = (18 * fit_s + 11 * 2 * M.N_PSI * lg + OTHER_S) / 3600
            S["gates"]["runtime"] = {"measured_fit_s_Q15": fit_s, "fit_le_60min": fit_s <= 3600,
                                     "lik_grad_s_measured": lg, "projected_pilot_node_hours": proj,
                                     "pilot_le_24": proj <= 24, "basis": "measured dataset-0 fit and Hessian; "
                                     "716 s observed item fits; 0.62 h assumed"}
            log(f"runtime gate: fit {fit_s:.0f}s, projected {proj:.2f} node-hours")
            if not (fit_s <= 3600 and proj <= 24):
                S["stopped"] = "runtime gate failed after dataset 0 (plan §0.2): datasets 1-4 not run"
                break
    with open(out / "S2_coverage_table.csv", "w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    done = [d for d in S["datasets"] if "covered" in d]
    if len(done) == n_data:
        cov = np.sum([d["covered"] for d in done], axis=0)
        S["coverage_counts"] = dict(zip(M.NAMES, cov.tolist()))
        S["criterion_passed"] = bool(np.all(cov >= 4)) if not a.smoke else None
    return finish(out, S, t_all, prov)


def finish(out, S, t_all, prov):
    S["runtime_s"] = time.time() - t_all
    S["provenance"] = {k: prov[k] for k in ("host", "cpu", "jobs", "slurm_job", "numpy", "python", "smoke")}
    jdump(out / "summary.json", S)
    log(f"done in {S['runtime_s']:.0f}s; criterion {S.get('criterion_passed')}; stopped {S.get('stopped')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
