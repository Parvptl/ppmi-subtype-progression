"""Stage 2 M2, Step 3 (Rev 2 §6 step 3): V2, §4.1 order checks, §2.5 mode checks; V3/V4 reported.

Execution plan (frozen): docs/cluster_phase/29_STAGE2_STEP3_EXECUTION_PLAN_20261019.md
(reviewer resolutions R-a..R-d in its §1). Likelihood EVALUATION only, at fixed frozen
parameters (pilot items and ψ̂): no parameter is estimated. 30 designated patients
(real item responses), plus synthetic dataset 0 for V3. No recovery, B11, sampler, gates,
calibration, positive controls or evaluation statistics.

    scripts/cluster/submit.sh ... -w node002 -- scripts/49_m2a1_checks.py --out <exec dir>/step3 --jobs 32
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
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
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
         "plan29": "29_STAGE2_STEP3_PLAN.sha256"}
PILOT = ROOT / "experiments/20261016T200350Z_stage2_m2_pilot/results"
FIXED = ROOT / "experiments/20261016T203341Z_stage2_m2_recovery_fixed_items_ds0to4/results.json"
M2QUAD_SHA = "50f164c6fc3e04b1d72e7726d87dfa7b5306f93bc590e494b6aca349797b093f"
DESIGNATED = [4115, 4102, 177395, 311402, 3254, 116529, 266248, 4123, 4109, 143116, 238181, 149940, 3815, 137264,
              101675, 3415, 100952, 4026, 130013, 3467, 3253, 3023, 3108, 3052, 3659, 168299, 148699, 182340, 101038, 215045]
G_REPORTED = 5
V3_GRID = [0.3550, 0.3675, 0.3795, 0.3925, 0.4053, 0.4200, 0.4350]
V3_REF_ARGMAX, V3_TOL = 0.3973, 0.0134
_W: dict = {}


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


def one(i):
    t, x, yt, yp = _W["pat"][i]
    return M.Group(t[None], x[None], yt[None], yp[None], [0]), t, x, yt, yp


def task(args):
    """One patient-level evaluation; never raises (failures are returned)."""
    kind, i, k, Q, G = args
    psi, it = _W["psis"][k], _W["it"]
    g, t, x, yt, yp = one(i)
    try:
        if kind == "A":                                  # Laplace over b, reported context
            info: dict = {}
            val, b, nd, sH, _ = MQ.group_laplace(psi, g, it, Q, want_grad=False, info=info)
            out = {"value": float(val[0]), "nd": bool(nd[0]), "budget": bool(info.get("failed")),
                   "iterations": info.get("iterations"), "halvings": info.get("halvings"), "fallback_used": False}
            if info.get("failed"):
                out["value"] = float("nan")
            elif not nd[0]:
                fb = MQ.group_agq(psi, g, it, Q, G_REPORTED, want_grad=False)[0]
                out.update(value=float(fb[0]), fallback_used=True)
            return (kind, i, k, Q, G), out
        if kind == "F":
            info = {}
            val, b, nd, _ = MQ.group_agq(psi, g, it, Q, G, want_grad=False, info=info, chunk=256)   # memory only
            return (kind, i, k, Q, G), {"value": float(val[0]) if not info.get("failed") else float("nan"),
                                        "nd": bool(nd[0]), "budget": bool(info.get("failed"))}
        if kind == "D3":
            idx = _W["cohort_index"][i]
            if 2 * len(t) <= 6:
                return (kind, i, k, Q, G), {"value": M.agq_one(t, x, yt, yp, psi, it, nodes=7), "se": 0.0, "method": "AGQ7"}
            est, se = M.is_one(t, x, yt, yp, psi, it, np.random.default_rng([42, 45, 4, int(idx)]), draws=20000)
            return (kind, i, k, Q, G), {"value": est, "se": se, "method": "IS20000_t5"}
        if kind == "MODE":
            d = MQ.prepare(g, it)
            beta, D, *_ = M.unpack(psi)
            Dinv = np.linalg.inv(D)
            E, logw, *_ = MQ.eps_nodes(psi, Q)
            mu = MQ.visit_means(psi, d.T, d.X)
            b0, nd0, loc0 = MQ.modes(d, mu, E, logw, Dinv, np.zeros((1, 4)))
            sd = np.sqrt(np.diag(np.linalg.inv(-loc0.hess[0]))) if nd0[0] else np.full(4, np.nan)
            res = [b0[0]]
            nds = [bool(nd0[0])]
            for sgn in (-1.0, 1.0):
                bs, nds_, _ = MQ.modes(d, mu, E, logw, Dinv, (b0[0] + sgn * 2 * sd)[None])
                res.append(bs[0])
                nds.append(bool(nds_[0]))
            res = np.array(res)
            return (kind, i, k, Q, G), {"modes": res, "max_abs_diff": float(np.max(np.abs(res - res[0]))),
                                        "nd_each_start": nds, "sd": sd}
    except Exception as e:                                # recorded, never hidden
        return (kind, i, k, Q, G), {"error": f"{type(e).__name__}: {e}", "value": float("nan")}
    raise ValueError(kind)


def run(tasks, jobs):
    with ProcessPoolExecutor(jobs, initializer=_W.update, initargs=(dict(_W),)) as ex:
        return dict(ex.map(task, tasks, chunksize=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--smoke", action="store_true", help="code path only: 4 patients, Q <= 10, G <= 5, no V3")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t_all = time.time()
    locks = {k: (FROZEN / v).read_text().split() for k, v in LOCKS.items()}
    prov = {"plan": "docs/cluster_phase/29_STAGE2_STEP3_EXECUTION_PLAN_20261019.md",
            "docs": {k: {"file": v[1], "frozen": v[0], "current": sha(ROOT / v[1])} for k, v in locks.items()},
            "m2_quad_sha256": sha(ROOT / "src/ppmi_subtype/m2_quad.py"),
            "code_sha256": {f: sha(ROOT / f) for f in ("src/ppmi_subtype/m2_quad.py", "src/ppmi_subtype/m2_grm.py",
                                                       "src/ppmi_subtype/grm.py", "scripts/45_m2_pilot.py",
                                                       "scripts/49_m2a1_checks.py")},
            "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (PILOT / "step2_items.json",
                                                                          PILOT / "step3_longitudinal.json",
                                                                          PILOT / "step4_laplace.json", FIXED)},
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
    cfg = load_config()
    v, times, treat, YTd, YPd, *_ = P.load_cohort(cfg)
    pats = sorted(times)
    pick = np.random.default_rng([42, 45, 4]).choice(len(pats), 30, replace=False)
    assert [int(pats[i]) for i in pick] == DESIGNATED, "designated patients differ"
    if a.smoke:
        pick = pick[:4]
    itj = json.loads((PILOT / "step2_items.json").read_text())
    it = M.Items(np.array(itj["aT"]), np.array(itj["tauT"]), np.array(itj["aP"]), np.array(itj["tauP"]))
    psi0 = np.array(json.loads((PILOT / "step3_longitudinal.json").read_text())["psi"])
    LO, HI = np.array([b[0] for b in M.BOUNDS]), np.array([b[1] for b in M.BOUNDS])
    psis = [psi0] + [np.clip(psi0 + np.random.default_rng([42, 46, 7, k]).normal(0, 0.05, M.N_PSI), LO, HI)
                     for k in (1, 2, 3)]
    pat = [(times[pats[i]], treat[pats[i]], YTd[pats[i]], YPd[pats[i]]) for i in pick]
    _W.update(pat=pat, psis=psis, it=it, cohort_index=[int(i) for i in pick])
    npat = len(pat)
    nvis = [len(p[0]) for p in pat]
    top5 = sorted(range(npat), key=lambda j: (-nvis[j], int(pick[j])))[:5]
    S: dict = {"designated_patients": [int(pats[i]) for i in pick], "n_visits": nvis, "top5_most_visits": top5,
               "psis": psis, "sigma": [{"sigma_T": float(np.sqrt(M.unpack(p)[3][0, 0])),
                                        "sigma_P": float(np.sqrt(M.unpack(p)[3][1, 1]))} for p in psis],
               "stages": {}, "criteria": {}}
    Qs = (7, 10) if a.smoke else (7, 10, 15, 20, 25)

    def stage(name, fn):
        t0 = time.time()
        try:
            r = fn()
            r["stage_wall_s"] = time.time() - t0
            jdump(out / f"{name}.json", r)
            S["stages"][name] = {"ok": True, "wall_s": r["stage_wall_s"]}
            log(f"{name} done ({r['stage_wall_s']:.0f}s)")
            return r
        except Exception as e:
            S["stages"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
            log(f"{name}: ERROR {e}")
            return None

    def table(path, rows):
        if rows:
            fields = list(dict.fromkeys(k for r in rows for k in r))          # union of columns
            with open(out / path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields, restval="")
                w.writeheader()
                w.writerows(rows)

    # ---- S1: D3 reference at the 4 ψ; reproducibility at ψ̂ --------------------------------------
    def s1():
        R = run([("D3", i, k, 0, 0) for k in range(4) for i in range(npat)], a.jobs)
        S["_D3"] = R
        pilot = {p["patient_index"]: p["reference"] for p in json.loads((PILOT / "step4_laplace.json").read_text())["patients"]}
        rep = [{"patient": j, "PATNO": int(pats[pick[j]]), "ours": R[("D3", j, 0, 0, 0)]["value"],
                "pilot": pilot[int(pick[j])], "equal": R[("D3", j, 0, 0, 0)]["value"] == pilot[int(pick[j])]}
               for j in range(npat)]
        rows = [{"patient": j, "psi": k, "method": R[("D3", j, k, 0, 0)].get("method"),
                 "value": R[("D3", j, k, 0, 0)]["value"], "mc_se": R[("D3", j, k, 0, 0)].get("se"),
                 "error": R[("D3", j, k, 0, 0)].get("error")} for k in range(4) for j in range(npat)]
        table("S1_reference_d3.csv", rows)
        bad = [r for r in rows if r["error"] or not np.isfinite(r["value"])]
        return {"reproduces_pilot_at_psi_hat": all(r["equal"] for r in rep), "per_patient_repro": rep,
                "n_failures": len(bad), "failures": bad, "max_mc_se": max(r["mc_se"] or 0 for r in rows)}
    r1 = stage("S1_reference_d3", s1)

    # ---- S2: order (a) and (a′) ---------------------------------------------------------------------
    def s2():
        tasks = [("A", i, k, Q, 0) for Q in Qs if Q != 25 for k in range(4) for i in range(npat)]
        tasks += [("A", i, k, 25, 0) for k in range(4) for i in top5] if 25 in Qs else []
        R = run(tasks, a.jobs)
        S["_A"] = R
        val = lambda i, k, Q: R[("A", i, k, Q, 0)]["value"]  # noqa: E731
        res = {}
        if 25 in Qs:
            d_ap = [abs(val(i, k, 20) - val(i, k, 25)) for k in range(4) for i in top5]
            res["a_prime"] = {"max_abs": float(max(d_ap)), "tol": 0.001, "passed": max(d_ap) <= 0.001}
            qref = 20 if res["a_prime"]["passed"] else 25
            if qref == 25:                               # (a) repeated against 25 on all patients
                R.update(run([("A", i, k, 25, 0) for k in range(4) for i in range(npat) if i not in top5], a.jobs))
        else:
            qref = max(Qs)
        rows, sel = [], None
        for Q in [q for q in (7, 10, 15) if q in Qs and q != qref]:
            d = [abs(val(i, k, Q) - val(i, k, qref)) for k in range(4) for i in range(npat)]
            ok = bool(np.all(np.isfinite(d)) and max(d) <= 0.005)
            res[f"Q{Q}"] = {"max_abs_vs_ref": float(np.nanmax(d)), "passed": ok}
            if ok and sel is None:
                sel = Q
        res.update(Q_ref=qref, Q_star=sel, passed=sel is not None)
        for k in range(4):
            for i in range(npat):
                rows.append({"patient": i, "psi": k, **{f"A_Q{Q}": val(i, k, Q) for Q in Qs
                                                         if ("A", i, k, Q, 0) in R},
                             "nd_Qref": R[("A", i, k, qref, 0)]["nd"], "budget_any": any(
                                 R[t].get("budget") for t in R if t[1] == i and t[2] == k),
                             "fallback_used_any": any(R[t].get("fallback_used") for t in R if t[1] == i and t[2] == k),
                             "errors": ";".join(R[t].get("error", "") for t in R if t[1] == i and t[2] == k and "error" in R[t])})
        table("S2_order_a.csv", rows)
        res["counts"] = {"non_nd_evals": sum(1 for r in R.values() if r.get("nd") is False),
                         "fallback_used": sum(1 for r in R.values() if r.get("fallback_used")),
                         "budget": sum(1 for r in R.values() if r.get("budget")),
                         "errors": sum(1 for r in R.values() if "error" in r),
                         "nonfinite": sum(1 for r in R.values() if not np.isfinite(r["value"]))}
        return res
    r2 = stage("S2_order_a", s2) if r1 else None
    Qs_star = r2["Q_star"] if r2 else None
    S["criteria"]["order_a"] = bool(r2 and r2["passed"])
    if r2 and "a_prime" in r2:
        S["criteria"]["order_a_prime"] = r2["a_prime"]["passed"]
    if not Qs_star and a.smoke:                           # code-path check only: never in a real run
        Qs_star = 10
        S["smoke_override"] = "no Q selected in smoke; Q = 10 forced to exercise S3-S7 (not a result)"
    if r2 is None:
        S["stopped"] = "S2 (order a) raised an execution error: Step 3 not evaluated (see stages)"
        log(S["stopped"])
        return finish(out, S, t_all, prov)
    if not Qs_star:
        S["stopped"] = "order (a) selected no Q: Step 3 stops (plan §4)"
        log(S["stopped"])
        return finish(out, S, t_all, prov)
    Q = Qs_star

    # ---- S3: V2 (and V2-F) --------------------------------------------------------------------------
    def s3():
        A, D3 = S["_A"], S["_D3"]
        F = run([("F", i, 0, Q, G) for G in (3, 5) for i in range(npat)], a.jobs)
        S["_F0"] = F
        rows = []
        for i in range(npat):
            ref, se = D3[("D3", i, 0, 0, 0)]["value"], D3[("D3", i, 0, 0, 0)]["se"]
            e = A[("A", i, 0, Q, 0)]["value"] - ref
            rows.append({"patient": i, "PATNO": int(pats[pick[i]]), "n_visits": nvis[i],
                         "approx_laplace_Q": A[("A", i, 0, Q, 0)]["value"], "reference_d3": ref,
                         "d3_method": D3[("D3", i, 0, 0, 0)]["method"], "mc_se": se, "error": e,
                         "abs_error": abs(e), "nd": A[("A", i, 0, Q, 0)]["nd"],
                         "fallback_used": A[("A", i, 0, Q, 0)]["fallback_used"],
                         "reference_limited": bool(abs(e) - 2 * se <= 0.05 < abs(e) + 2 * se),
                         "error_F_G3": F[("F", i, 0, Q, 3)]["value"] - ref, "error_F_G5": F[("F", i, 0, Q, 5)]["value"] - ref,
                         **{f"error_Q{q}": A[("A", i, 0, q, 0)]["value"] - ref for q in Qs if ("A", i, 0, q, 0) in A}})
        table("S3_v2_patients.csv", rows)
        ae = np.array([r["abs_error"] for r in rows])
        res = {"Q_star": Q, "max_abs_error": float(np.nanmax(ae)), "mean_abs_error": float(np.nanmean(ae)),
               "median_abs_error": float(np.nanmedian(ae)), "n_nonfinite": int((~np.isfinite(ae)).sum()),
               "n_reference_limited": sum(r["reference_limited"] for r in rows),
               "passed": bool(np.all(np.isfinite(ae)) and ae.max() <= 0.05), "patients": rows,
               "descriptive_max_abs_error_by_Q": {q: float(np.nanmax([abs(r[f"error_Q{q}"]) for r in rows]))
                                                  for q in Qs if f"error_Q{q}" in rows[0]}}
        for G in (3, 5):
            aeg = np.array([abs(r[f"error_F_G{G}"]) for r in rows])
            res[f"V2F_G{G}"] = {"max_abs_error": float(np.nanmax(aeg)), "passed": bool(np.all(np.isfinite(aeg)) and aeg.max() <= 0.05),
                                "binding": not (np.all(np.isfinite(ae)) and ae.max() <= 0.05)}
        return res
    r3 = stage("S3_v2", s3)
    S["criteria"]["V2"] = bool(r3 and r3["passed"])

    # ---- S4: order (b) and (b′) ---------------------------------------------------------------------
    def s4():
        Gs = (3, 5) if a.smoke else (3, 5, 7)
        R = run([("F", i, k, Q, G) for G in Gs for k in range(4) for i in range(npat)]
                + ([] if a.smoke else [("F", i, k, Q, 9) for k in range(4) for i in top5]), a.jobs)
        gref = max(Gs)
        val = lambda i, k, G: R[("F", i, k, Q, G)]["value"]  # noqa: E731
        res = {"G_ref": gref}
        if not a.smoke:
            d9 = [abs(val(i, k, 9) - val(i, k, 7)) for k in range(4) for i in top5]
            res["b_prime"] = {"max_abs": float(max(d9)), "tol": 0.001, "passed": max(d9) <= 0.001}
        sel = None
        for G in [g for g in (3, 5) if g != gref]:
            d = [abs(val(i, k, G) - val(i, k, gref)) for k in range(4) for i in range(npat)]
            ok = bool(np.all(np.isfinite(d)) and max(d) <= 0.005)
            res[f"G{G}"] = {"max_abs_vs_ref": float(np.nanmax(d)), "passed": ok,
                            "per_patient_max_over_psi": [float(max(abs(val(i, k, G) - val(i, k, gref)) for k in range(4)))
                                                         for i in range(npat)]}
            if ok and sel is None:
                sel = G
        res["inconclusive"] = bool(res.get("b_prime") and not res["b_prime"]["passed"])
        res["G_star"] = None if res["inconclusive"] else sel
        res["counts"] = {"budget": sum(1 for r in R.values() if r.get("budget")),
                         "non_nd": sum(1 for r in R.values() if r.get("nd") is False),
                         "errors": sum(1 for r in R.values() if "error" in r)}
        rows = [{"patient": i, "psi": k, **{f"F_G{G}": val(i, k, G) for G in Gs},
                 **({"F_G9": val(i, k, 9)} if ("F", i, k, Q, 9) in R else {})} for k in range(4) for i in range(npat)]
        table("S4_order_b.csv", rows)
        return res
    r4 = stage("S4_order_b", s4)

    # ---- S5: §2.5 mode checks ------------------------------------------------------------------------
    def s5():
        R = run([("MODE", i, 0, Q, 0) for i in range(npat)], a.jobs)
        rows = [{"patient": i, "PATNO": int(pats[pick[i]]), "max_abs_diff": R[("MODE", i, 0, Q, 0)].get("max_abs_diff"),
                 "nd_each_start": R[("MODE", i, 0, Q, 0)].get("nd_each_start"), "error": R[("MODE", i, 0, Q, 0)].get("error")}
                for i in range(npat)]
        table("S5_modes.csv", rows)
        diffs = [r["max_abs_diff"] for r in rows]
        unique = all(d is not None and d <= 1e-6 for d in diffs)
        nd_all = all(r["nd_each_start"] and all(r["nd_each_start"]) for r in rows)
        return {"Q": Q, "uniqueness_passed": unique, "max_diff": float(np.nanmax([d if d is not None else np.nan for d in diffs])),
                "patients_disagreeing": [r["patient"] for r in rows if r["max_abs_diff"] is None or r["max_abs_diff"] > 1e-6],
                "all_modes_nd": nd_all, "n_non_nd_returned": sum(1 for r in rows if r["nd_each_start"] and not r["nd_each_start"][0]),
                "patients": rows}
    r5 = stage("S5_modes", s5)
    S["criteria"]["mode_uniqueness"] = bool(r5 and r5["uniqueness_passed"])

    # ---- S6: V3 (synthetic dataset 0) -------------------------------------------------------------------
    def s6():
        sp = importlib.util.spec_from_file_location("p47", ROOT / "scripts/47_m2_profile_diagnostic.py")
        p47 = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(p47)
        g0 = M.make_groups(times, treat, YTd, YPd)
        psi_t, it_t, _ = P.truth_on_stage1_scale(g0)
        sg = P.regroup(g0, *P.simulate_groups(np.random.default_rng([42, 45, 1, 0]), g0, psi_t, it_t))
        psi_hat = np.array(json.loads(FIXED.read_text())["datasets"][0]["psi_hat"])
        res = {}
        for q in sorted({Q, S.get("_Qref", r2["Q_ref"])}):
            o = MQ.Objective(sg, it_t, q, jobs=a.jobs)
            try:
                lp = [-o(p47.psi_at_sigma_p(psi_hat, s), want_grad=False) for s in V3_GRID]
            finally:
                o.close()
            c = np.polyfit(V3_GRID, lp, 2)
            vtx = float(-c[1] / (2 * c[0])) if c[0] < 0 else float("nan")
            res[f"Q{q}"] = {"loglik": lp, "argmax": vtx, "within_tol_of_0.3973": abs(vtx - V3_REF_ARGMAX) <= V3_TOL}
        res["argmax_change_Qstar_vs_Qref"] = res[f"Q{Q}"]["argmax"] - res[f"Q{r2['Q_ref']}"]["argmax"]
        return res
    r6 = stage("S6_v3", s6) if not a.smoke else None

    # ---- S7: V4 totals and parameter-level effect ------------------------------------------------------
    def s7():
        A, D3 = S["_A"], S["_D3"]
        qref = r2["Q_ref"]
        rows = []
        for k in range(4):
            e = [A[("A", i, k, Q, 0)]["value"] - D3[("D3", i, k, 0, 0)]["value"] for i in range(npat)]
            se = np.sqrt(sum(D3[("D3", i, k, 0, 0)]["se"] ** 2 for i in range(npat)))
            rows.append({"psi": k, **S["sigma"][k], "sum_error_Qstar_minus_D3": float(np.sum(e)), "mc_se_sum": float(se),
                         "total_Qstar": float(sum(A[("A", i, k, Q, 0)]["value"] for i in range(npat))),
                         "total_Qref": float(sum(A[("A", i, k, qref, 0)]["value"] for i in range(npat)))})
        dd = [{"psi_pair": f"{k}-0", "diff_Qstar": rows[k]["total_Qstar"] - rows[0]["total_Qstar"],
               "diff_Qref": rows[k]["total_Qref"] - rows[0]["total_Qref"]} for k in (1, 2, 3)]
        for x in dd:
            x["change"] = x["diff_Qstar"] - x["diff_Qref"]
        return {"V4": rows, "param_effect_total_differences": dd}
    stage("S7_v4_param_effect", s7)
    S["criteria"]["step3_passed"] = bool(S["criteria"].get("order_a") and S["criteria"].get("V2")
                                         and S["criteria"].get("mode_uniqueness"))
    S["selected"] = {"Q_star": Q, "Q_ref": r2["Q_ref"], "G_star_fallback": r4["G_star"] if r4 else None,
                     "G_order_inconclusive": r4["inconclusive"] if r4 else None}
    return finish(out, S, t_all, prov)


def finish(out, S, t_all, prov):
    S = {k: v for k, v in S.items() if not k.startswith("_")}
    S["runtime_s"] = time.time() - t_all
    S["provenance"] = {k: prov[k] for k in ("host", "cpu", "jobs", "slurm_job", "numpy", "python", "smoke")}
    jdump(out / "summary.json", S)
    log(f"done in {S['runtime_s']:.0f}s; criteria {S['criteria']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
