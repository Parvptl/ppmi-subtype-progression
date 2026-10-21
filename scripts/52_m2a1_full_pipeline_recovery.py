"""Stage 2 M2, Step 6: full-pipeline recovery with S1-b (Rev 2 §5, §6 step 6). BINDING criterion.

Frozen execution plan: docs/cluster_phase/37_STAGE2_STEP6_FULL_PIPELINE_RECOVERY_PLAN_FINAL_20261019.md
(decisions E-1..E-12 in its §4). Synthetic data only: 5 datasets, seeds [42, 45, 1, r], a-priori
truth on the stage-1 scale (scripts/45_m2_pilot.py), real visit design. Per dataset, exactly one S1-b
iteration: (1) N(0, 1) item fits (the pilot path, reproduced exactly, E-11) -> (2) stage 2, Laplace
Q = 15, default ψ, cold modes -> ψ̂(1) [Hessian + reported evaluation: non-binding, E-8] -> (3) items
refitted under the visit-specific fitted latent N(M_ij, K_ij,jj) at ψ̂(1) (E-1; script 51's
latent_negloglik, hash-verified, E-12), one start at initial_items (E-3) -> (4) stage 2 refit from the
default ψ, cold modes (E-4) -> ψ̂(2) -> (5) FD Hessian (h = 1e-4) -> (6) reported evaluation (G = 5).
Criterion: every ψ component covered by its 95 % Wald interval in >= 4 of 5 datasets (doc 16 D3).
Outcome classes and flagged events: plan §7. No real responses, no labels.

    scripts/cluster/submit.sh ... -w node002 -- scripts/52_m2a1_full_pipeline_recovery.py --out <exec dir>/step6 --jobs 32
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

from ppmi_subtype import grm as G  # noqa: E402
from ppmi_subtype import m2_grm as M  # noqa: E402
from ppmi_subtype import m2_quad as MQ  # noqa: E402

FROZEN = ROOT / "docs/cluster_phase/_frozen"
LOCKS = {"doc16": "16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.sha256", "rev2": "21_STAGE2_DOC16_AMENDMENT1_REV2.sha256",
         "doc25": "25_STAGE2_ADDENDUM_A_FINAL.sha256", "doc27": "27_STAGE2_EXACT_AGQ_GRADIENT.sha256",
         "plan29": "29_STAGE2_STEP3_PLAN.sha256", "plan31": "31_STAGE2_STEP4_PLAN.sha256",
         "plan34": "34_STAGE2_STEP5_PLAN.sha256", "plan37": "37_STAGE2_STEP6_PLAN.sha256"}
PLAN37_SHA = "c02be4ce70e1b791384dafb7a6d491b483d85dbca22c4d07e1a01bed94b50bba"
CODE = {"src/ppmi_subtype/m2_quad.py": "50f164c6fc3e04b1d72e7726d87dfa7b5306f93bc590e494b6aca349797b093f",
        "src/ppmi_subtype/m2_grm.py": "585c878b897b99cb15acf38246b0ee4063948219e6ef99866773cf0fbf6a4a46",
        "src/ppmi_subtype/grm.py": "1584bc62643bea544c27515fe85e02ba6f9383703b3170ce69b7473609c8b7a8",
        "scripts/45_m2_pilot.py": "a25e150313dc1096b272ae15c38852c3224244510179984a9e3a03016983c684",
        "scripts/51_m2a1_s1a_true_latent_items.py": "517c4cc166750c3bbb48d1643b76c4aa3ea25387468af69d6b5f82b8fcbc8ff9",
        "tests/test_s1a_latent_items.py": "ddda689f919756911605c9b2c2ea5cd58adaa64cd8d85f8bb32c1b598d3623cc"}
PILOT = ROOT / "experiments/20261016T200350Z_stage2_m2_pilot/results"
STEP4 = ROOT / "experiments/20261018T184713Z_stage2_step4_fixed_item_recovery"
STEP5 = ROOT / "experiments/20261018T215102Z_stage2_step5_s1a"
IDENT = ROOT / "experiments/20261017T233307Z_stage2_m2_exact_agq/bench/S2_laplace_fit.json"
INPUTS = {"pilot_step1_recovery": (PILOT / "step1_recovery.json",
                                   "d16d5435b85eae99ff84f22b69594b0853ff23833c983a2b583e082b79188603"),
          "pilot_provenance": (PILOT / "provenance.json",
                               "68bc14b59d4c40cf35e6041a9bdd59bc2902b62743d34618a72ebd5ff78e7d1f"),
          "step4_summary": (STEP4 / "step4_run2/summary.json",
                            "00433150c23c90ecc039bff39fe2cf87d1fa13614b597cad0a611964af7afa39"),
          "identity_ref": (IDENT, "dbeb36b7f9edc055cb258ec9bdcfe12bb80411b9df541d10391e51bb1bc096dc"),
          "step5_summary": (STEP5 / "step5/summary.json",
                            "5ccc690cd6eb5b5d93ceb06e7d9d7678ee9c3471b6fc247b1f1e46d64b429dae"),
          "step5_items_dataset0": (STEP5 / "step5/S1_items_dataset0.json",
                                   "c16ecfbc8b984214c7e5028ab4b503fb509ae00083f602efce0b8831436790a1")}
IDENT_NLL = 57022.74225348176
Q, G_REPORTED, N_DATA, Z95 = 15, 5, 5, 1.96
FIT_LIMIT_S, PILOT_LIMIT_NH = 3600.0, 24.0
ITEM_FITS_S, S1B_ASSUMED_S = 716.0, 0.62 * 3600
METRICS = ("a_corr_T", "a_maxerr_T", "tau_maxerr_T", "a_corr_P", "a_maxerr_P", "tau_maxerr_P")
SUBSCORES = ("T", "P")


def sha(p, kind="sha256"):
    h = hashlib.new(kind)
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _conv(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=_conv))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                     # process-pool tasks defined in it must be picklable by name
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
#  Coverage, flags, outcome classes (plan §7) and the runtime projection (plan §10)
# ---------------------------------------------------------------------------
def coverage(psi_hat, psi_true, H):
    """E-6: non-PD / singular / non-finite H -> nothing covered; non-finite SE -> not covered."""
    psi_hat, psi_true = np.asarray(psi_hat, float), np.asarray(psi_true, float)
    k = psi_true.size
    out = {"hessian_finite": bool(np.all(np.isfinite(H))), "hessian_pd": False, "hessian_singular": False,
           "min_eig": float("nan"), "max_eig": float("nan"), "se": np.full(k, np.nan), "z": np.full(k, np.nan),
           "lower": np.full(k, np.nan), "upper": np.full(k, np.nan), "covered": np.zeros(k, bool)}
    if not out["hessian_finite"]:
        out["reason"] = "hessian not finite"
        return out
    eig = np.linalg.eigvalsh(0.5 * (H + H.T))
    out["min_eig"], out["max_eig"] = float(eig.min()), float(eig.max())
    if not eig.min() > 0:
        out["reason"] = "hessian not positive definite"
        return out
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        out["hessian_singular"] = True
        out["reason"] = "hessian singular"
        return out
    out["hessian_pd"] = True
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.diag(Hinv))
    z = (psi_hat - psi_true) / se
    fin = np.isfinite(se) & np.isfinite(z)
    out.update(se=se, z=z, lower=psi_hat - Z95 * se, upper=psi_hat + Z95 * se,
               covered=fin & (np.abs(np.where(fin, z, np.inf)) <= Z95),
               missing_interval=[int(i) for i in np.flatnonzero(~fin)])
    out["reason"] = None
    return out


def flags_for(d):
    """F1-F4 (plan §7) from a dataset record; returns the list of flag strings."""
    fl = []
    for stage in ("stage2_fit_1", "stage2_fit_2"):
        f = d[stage]
        if not f["converged"] or f["stopped_by_time"] or f["maxiter_reached"]:
            fl.append(f"F1:{stage}")
    for stage in ("items_1", "items_2"):
        for s in SUBSCORES:
            f = d[stage][s]
            if not f["converged"] or f["a_out_of_range"]:
                fl.append(f"F2:{stage}_{s}")
    if d["stage2_fit_2"]["at_bound"]:
        fl.append("F3:psi2_at_bound")
    rep = d["reported_2"]
    if rep["n_fallback_patients"] > 0 or rep["budget_triggers"] > 0:
        fl.append("F4:psi2_fallback_or_budget")
    return fl


def classify(stop, n_complete, n_data, covered_by_ds, flagged_ds):
    """Outcome class and formal verdict (plan §7). covered_by_ds: list of (19,) bool arrays."""
    if stop and stop.startswith("INTEGRITY"):
        return {"class": "INTEGRITY STOP", "formal_verdict": None, "stop": stop}
    if stop or n_complete < n_data:
        return {"class": "INCOMPLETE", "formal_verdict": None, "stop": stop}
    cov = np.array(covered_by_ds, bool)
    counts = cov.sum(0)
    met = bool(np.all(counts >= 4))
    res = {"coverage_counts": counts, "criterion_met_as_computed": met}
    if not flagged_ds:
        res.update({"class": "COMPLETE, no flagged event", "formal_verdict": "PASS" if met else "FAIL"})
        return res
    hi, lo = cov.copy(), cov.copy()
    for r in flagged_ds:
        hi[r], lo[r] = True, False
    res.update({"class": "COMPLETE, with flagged events", "formal_verdict": "RESERVED FOR REVIEW",
                "flagged_datasets": list(flagged_ds),
                "dependence": {"met_if_flagged_all_covered": bool(np.all(hi.sum(0) >= 4)),
                               "met_if_flagged_none_covered": bool(np.all(lo.sum(0) >= 4))}})
    res["dependence"]["outcome_depends_on_flagged"] = (res["dependence"]["met_if_flagged_all_covered"]
                                                       != res["dependence"]["met_if_flagged_none_covered"])
    return res


def projection(t2, t4, lg, t_s1b=None, t_h1=None):
    """Frozen doc 25 formula (plan §10); the measured-S1-b variant is reported only."""
    p = (18 * max(t2, t4) + 11 * 38 * lg + ITEM_FITS_S + S1B_ASSUMED_S) / 3600
    out = {"projected_pilot_node_hours": p, "fit_2_le_60min": t2 <= FIT_LIMIT_S, "fit_4_le_60min": t4 <= FIT_LIMIT_S,
           "pilot_le_24": p <= PILOT_LIMIT_NH}
    out["gate_passed"] = out["fit_2_le_60min"] and out["fit_4_le_60min"] and out["pilot_le_24"]
    if t_s1b is not None and t_h1 is not None:
        out["variant_measured_s1b_node_hours"] = (18 * max(t2, t4) + 11 * 38 * lg + ITEM_FITS_S
                                                  + N_DATA * (t_s1b + t_h1)) / 3600
    return out


# ---------------------------------------------------------------------------
#  One dataset: exactly one S1-b iteration (plan §6). `ops` supplies the numerical stages.
# ---------------------------------------------------------------------------
def pipeline_dataset(sg, items_1, ops, psi_default):
    rec = {}
    f1 = ops.fit_stage2(sg, items_1, psi0=psi_default)                      # (2), cold modes (E-4)
    rec["fit_1"] = f1
    rec["hess_1"] = ops.hessian(sg, items_1, f1)                             # (2h) non-binding (E-8)
    rec["reported_1"] = ops.reported(sg, items_1, f1)
    items_2 = ops.refit_items(sg, f1.psi)                                    # (3) S1-b, once (E-1, E-3)
    rec["items_2"] = items_2
    f2 = ops.fit_stage2(sg, items_2["items"], psi0=psi_default)             # (4), default ψ, cold (E-4)
    rec["fit_2"] = f2
    rec["hess_2"] = ops.hessian(sg, items_2["items"], f2)                    # (5)
    rec["reported_2"] = ops.reported(sg, items_2["items"], f2)               # (6)
    return rec


def fit_record(f, wall, maxiter):
    pen = {}
    for e in (f.log or []):
        for k, c in e["reasons"].items():
            pen[k] = pen.get(k, 0) + c
    return {"nll": f.nll, "converged": f.converged, "message": f.message, "n_iter": f.n_iter, "n_eval": f.n_eval,
            "grad_norm": f.grad_norm, "grad_norm_lt_0.934_reference_only": bool(f.grad_norm < 0.934),
            "at_bound": f.at_bound, "stopped_by_time": f.stopped_by_time,
            "maxiter_reached": bool(f.n_iter >= maxiter or "ITERATIONS REACHED LIMIT" in f.message.upper()),
            "penalty_evals": sum(1 for e in (f.log or []) if e["penalty"]), "penalty_reasons_chunks": pen,
            "non_nd_at_accepted_evals": sum(e["n_non_nd"] for e in (f.log or []) if not e["penalty"]),
            "fit_eval_wall_s": sum(e["wall_s"] for e in (f.log or [])), "wall_s": wall, "psi": f.psi}


def item_record(fT, fP):
    out = {}
    for s, f in (("T", fT), ("P", fP)):
        a = np.asarray(f["a"])
        out[s] = {"nll": f["nll"], "converged": f["converged"], "message": f["message"],
                  "n_iter": f.get("n_iter"), "grad_norm": f.get("grad_norm"), "a": a, "tau": np.asarray(f["tau"]),
                  "a_out_of_range": bool(np.any((a < 0.05) | (a > 20))), "wall_s": f.get("wall_s")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--smoke", action="store_true",
                    help="code path only (plan §11): 1 dataset, 60 patients, reduced maxiter; NOT A RESULT")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t_all = time.time()
    S: dict = {"plan": "docs/cluster_phase/37_STAGE2_STEP6_FULL_PIPELINE_RECOVERY_PLAN_FINAL_20261019.md",
               "smoke": a.smoke, "NOT_A_RESULT": bool(a.smoke), "integrity": {}, "datasets": [], "flags": {}}
    stage2_maxiter = 5 if a.smoke else 1000
    item_maxiter = 20 if a.smoke else 3000
    git = lambda *x: subprocess.run(["git", *x], capture_output=True, text=True, cwd=ROOT).stdout  # noqa: E731
    st = git("status", "--porcelain").splitlines()
    prov = {"plan_sha256_expected": PLAN37_SHA,
            "docs": {}, "code_sha256": {f: sha(ROOT / f) for f in CODE},
            "inputs_sha256": {k: sha(p) for k, (p, _) in INPUTS.items()},
            "driver_sha256": sha(Path(__file__)), "tests_sha256": sha(ROOT / "tests/test_s1b_pipeline.py"),
            "git": {"head": git("rev-parse", "HEAD").strip(), "tracked_changes": sum(1 for x in st if not x.startswith("??")),
                    "staged": len(git("diff", "--cached", "--name-only").split())},
            "Q": Q, "G_reported": G_REPORTED, "seeds": {r: [42, 45, 1, r] for r in range(N_DATA)},
            "default_psi": M.default_psi(), "names": M.NAMES, "stage2_maxiter": stage2_maxiter,
            "item_maxiter": item_maxiter, "host": platform.node(), "jobs": a.jobs,
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "smoke": a.smoke,
            "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "numpy": np.__version__, "python": sys.version.split()[0],
            "cpu": next((ln.split(":", 1)[1].strip() for ln in subprocess.run(
                ["lscpu"], capture_output=True, text=True).stdout.splitlines() if ln.startswith("Model name")), "?")}
    for k, v in LOCKS.items():
        h, f = (FROZEN / v).read_text().split()
        prov["docs"][k] = {"file": f, "frozen": h, "current": sha(ROOT / f)}

    def stop_integrity(msg):
        S["stopped"] = f"INTEGRITY STOP: {msg}"
        log(S["stopped"])
        return finish(out, S, t_all, prov)

    # ---- I-1 .. I-3, I-7, I-10 --------------------------------------------------------------
    I = S["integrity"]
    I["I-1_locks"] = all(d["frozen"] == d["current"] for d in prov["docs"].values()) and \
        prov["docs"]["plan37"]["frozen"] == PLAN37_SHA
    I["I-2_code"] = all(prov["code_sha256"][f] == h for f, h in CODE.items())
    pilot_prov = json.loads(INPUTS["pilot_provenance"][0].read_text())
    I["I-2_pilot_code_set"] = all(sha(ROOT / f) == h for f, h in pilot_prov["code_sha256"].items())
    from ppmi_subtype.config import load_config
    cfg = load_config()
    P = load_module("m2_pilot", ROOT / "scripts/45_m2_pilot.py")
    I["I-3_panel_md5"] = sha(cfg.path("outputs") / "visit_panel.csv", "md5") == P.PANEL_MD5
    I["I-3_inputs"] = all(prov["inputs_sha256"][k] == h for k, (_, h) in INPUTS.items())
    s5prov = json.loads((STEP5 / "step5/S0_provenance.json").read_text())
    s5sub = dict(ln.split()[::-1] for ln in (STEP5 / "02_driver_sha256.txt").read_text().splitlines() if ln.strip())
    I["I-7_script51"] = (prov["code_sha256"]["scripts/51_m2a1_s1a_true_latent_items.py"] == CODE[
        "scripts/51_m2a1_s1a_true_latent_items.py"] == s5prov["driver_sha256"]
        == s5sub["scripts/51_m2a1_s1a_true_latent_items.py"])
    I["I-10_seeds_git"] = {"seeds": prov["seeds"], "git": prov["git"], "staged_zero": prov["git"]["staged"] == 0}
    jdump(out / "S0_provenance.json", prov)
    bad = [k for k in ("I-1_locks", "I-2_code", "I-2_pilot_code_set", "I-3_panel_md5", "I-3_inputs", "I-7_script51")
           if not I[k]]
    if not I["I-10_seeds_git"]["staged_zero"]:
        bad.append("I-10_staged")
    if bad:
        return stop_integrity(f"failed {bad}")
    log("I-1, I-2, I-3, I-7, I-10 ok")

    # ---- I-4, I-8: unit tests inside the job -------------------------------------------------
    for key, tf in (("I-4_step6_tests", "tests/test_s1b_pipeline.py"), ("I-8_step5_tests", "tests/test_s1a_latent_items.py")):
        pt = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / tf)],
                            capture_output=True, text=True, cwd=ROOT)
        I[key] = {"returncode": pt.returncode, "tail": pt.stdout.strip().splitlines()[-2:]}
        if pt.returncode != 0:
            return stop_integrity(f"{key} failed")
    log(f"I-4 {I['I-4_step6_tests']['tail'][-1]}; I-8 {I['I-8_step5_tests']['tail'][-1]}")

    S1A = load_module("s1a", ROOT / "scripts/51_m2a1_s1a_true_latent_items.py")      # hash verified (I-7)

    # ---- I-5, I-6: truth and dataset identity ------------------------------------------------
    v, times, treat, YTd, YPd, *_ = P.load_cohort(cfg)                    # design only
    g0 = M.make_groups(times, treat, YTd, YPd)
    psi_true, it_true, scale = P.truth_on_stage1_scale(g0)
    pilot_rec = json.loads(INPUTS["pilot_step1_recovery"][0].read_text())
    I["I-5_truth"] = bool(np.array_equal(psi_true, np.array(pilot_rec["psi_true"])))
    if not I["I-5_truth"]:
        return stop_integrity("I-5 truth differs from the pilot record")
    S["psi_true"], S["names"] = psi_true, M.NAMES
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

    n_data = 1 if a.smoke else N_DATA
    data = {r: dataset(r) for r in range(n_data)}
    if a.smoke:
        data = {r: subset(g, 60) for r, g in data.items()}
        I["I-6_identity"] = I["I-9_latent_reproduction"] = I["I-11_pilot_reproduction"] = "skipped (smoke)"
    else:
        ref = json.loads(IDENT.read_text())
        nll = MQ.negloglik(np.array(ref["psi"]), data[0], it_true, 10, want_grad=False)
        I["I-6_identity"] = {"nll": nll, "recorded": ref["nll"], "equal": nll == ref["nll"] == IDENT_NLL}
        log(f"I-6 identity: {nll} -> {I['I-6_identity']['equal']}")
        if not I["I-6_identity"]["equal"]:
            return stop_integrity("I-6 dataset identity failed")

        # ---- I-9: reused latent functions reproduce Step 5's recorded values exactly ------------
        s5sum = json.loads(INPUTS["step5_summary"][0].read_text())
        s5it = json.loads(INPUTS["step5_items_dataset0"][0].read_text())
        vl = S1A.visit_latent(data[0], psi_true)
        Y0 = {"T": P.flatten(data[0])[0].astype(float), "P": P.flatten(data[0])[1].astype(float)}
        chk = {}
        nodes = G.gh_nodes()
        for s in SUBSCORES:
            m, sd = vl[s]
            got = {"pooled_mean": float(m.mean()), "pooled_var": float((sd ** 2).mean() + m.var()),
                   "s_min": float(sd.min()), "s_max": float(sd.max()), "m_min": float(m.min()), "m_max": float(m.max())}
            rec5 = s5sum["construction_check"][f"0{s}"]
            fo = s5it["fits"][f"oracle_{s}"]
            phi = G.pack_items(np.array(fo["a"]), np.array(fo["tau"]))
            nll_o, _ = S1A.latent_negloglik(phi, Y0[s], m, sd, nodes)
            chk[s] = {"construction_equal": all(got[k] == rec5[k] for k in got),
                      "oracle_nll": nll_o, "recorded_oracle_nll": s5it["item_objectives"][f"oracle_{s}"]["oracle_nll"]}
            chk[s]["oracle_nll_equal"] = chk[s]["oracle_nll"] == chk[s]["recorded_oracle_nll"]
        I["I-9_latent_reproduction"] = chk
        ok9 = all(c["construction_equal"] and c["oracle_nll_equal"] for c in chk.values())
        log(f"I-9 latent reproduction exact: {ok9}")
        if not ok9:
            return stop_integrity("I-9 reused latent functions do not reproduce Step 5's recorded values")

    # ---- (1) N(0, 1) item fits for all datasets; I-11 / E-11 ----------------------------------
    Ys = {(r, s): P.flatten(g)[0 if s == "T" else 1].astype(float) for r, g in data.items() for s in SUBSCORES}
    S1A._FIT.update(Y=Ys, lat={}, P=P)
    t0 = time.time()
    res1 = {(r, s): f for _, r, s, f in S1A.run_pool([("N01", r, s, item_maxiter) for r in data for s in SUBSCORES],
                                                     a.jobs)}
    S["stage1_wall_s"] = time.time() - t0
    for (r, s), f in res1.items():                                        # gradient norm at the optimum (report)
        _, gr = G.stage1_negloglik(G.pack_items(f["a"], f["tau"]), Ys[(r, s)], G.gh_nodes())
        f["grad_norm"] = float(np.linalg.norm(gr))
    if any(not np.isfinite(f["nll"]) for f in res1.values()):
        S["stopped"] = "non-finite stage-1 objective"
        return finish(out, S, t_all, prov)
    if not a.smoke:
        rep = {}
        for r in data:
            got = S1A.item_metrics(res1[(r, "T")], res1[(r, "P")], it_true)
            rep[r] = {m: {"recomputed": got[m], "pilot": pilot_rec["items"][r][m],
                          "equal": got[m] == pilot_rec["items"][r][m]} for m in METRICS}
        I["I-11_pilot_reproduction"] = rep
        I["I-11_exact"] = all(x["equal"] for d in rep.values() for x in d.values())
        log(f"(1) stage-1 fits {S['stage1_wall_s']:.0f}s; I-11 exact {I['I-11_exact']}")
        if not I["I-11_exact"]:
            return stop_integrity("I-11 pilot N(0,1) item-fit records not reproduced exactly (E-11)")

    # ---- stages (2)-(6) per dataset -------------------------------------------------------------
    psi_default = M.default_psi()
    timing: dict = {}

    class Ops:
        def __init__(self, r):
            self.r = r

        def fit_stage2(self, sg, it, psi0):
            t = time.perf_counter()
            f = MQ.fit(sg, it, Q=Q, psi0=psi0, jobs=a.jobs, maxiter=stage2_maxiter)
            timing.setdefault(self.r, {}).setdefault("fits", []).append(time.perf_counter() - t)
            return f

        def hessian(self, sg, it, f):
            o = MQ.Objective(sg, it, Q, jobs=a.jobs)
            o.state = dict(f.state)
            t = time.perf_counter()
            try:
                H = MQ.fd_hessian(o, f.psi)
            finally:
                o.close()
            w = time.perf_counter() - t
            timing.setdefault(self.r, {}).setdefault("hessians", []).append(w)
            pen = sum(1 for e in o.log if e["penalty"])
            return {"H": H, "wall_s": w, "penalised_fd_evals": pen,
                    "penalty_reasons": [e["reasons"] for e in o.log if e["penalty"]]}

        def reported(self, sg, it, f):
            t = time.perf_counter()
            rep = MQ.evaluate_reported(f.psi, sg, it, Q, G_REPORTED, init_state=f.state)
            rep["wall_s"] = time.perf_counter() - t
            return rep

        def refit_items(self, sg, psi1):
            vl = S1A.visit_latent(sg, psi1)
            for s in SUBSCORES:
                S1A._FIT["lat"][(self.r, s)] = vl[s]
            t = time.perf_counter()
            fits = {s: f for _, _, s, f in S1A.run_pool([("oracle", self.r, s, item_maxiter) for s in SUBSCORES],
                                                         a.jobs)}
            w = time.perf_counter() - t
            timing.setdefault(self.r, {})["s1b"] = w
            mom = {s: {"pooled_mean": float(vl[s][0].mean()), "pooled_var": float((vl[s][1] ** 2).mean() + vl[s][0].var())}
                   for s in SUBSCORES}
            it2 = M.Items(fits["T"]["a"], fits["T"]["tau"], fits["P"]["a"], fits["P"]["tau"])
            return {"items": it2, "fits": fits, "wall_s": w, "fitted_mixture_moments": mom}

    cov_rows, cov1_rows, item_rows, conv_rows = [], [], [], []
    covered_by_ds, flagged_ds = [], []
    for r, sg in data.items():
        t_ds = time.time()
        try:
            it1 = M.Items(res1[(r, "T")]["a"], res1[(r, "T")]["tau"], res1[(r, "P")]["a"], res1[(r, "P")]["tau"])
            rec = pipeline_dataset(sg, it1, Ops(r), psi_default)
            fits_w = timing[r]["fits"]
            d = {"dataset": r, "seed": [42, 45, 1, r],
                 "items_1": item_record(res1[(r, "T")], res1[(r, "P")]),
                 "items_2": item_record(rec["items_2"]["fits"]["T"], rec["items_2"]["fits"]["P"]),
                 "fitted_mixture_moments_at_psi1": rec["items_2"]["fitted_mixture_moments"],
                 "stage2_fit_1": fit_record(rec["fit_1"], fits_w[0], stage2_maxiter),
                 "stage2_fit_2": fit_record(rec["fit_2"], fits_w[1], stage2_maxiter),
                 "reported_1_nonbinding": rec["reported_1"], "reported_2": rec["reported_2"]}
            if not (np.isfinite(rec["fit_2"].nll) and np.all(np.isfinite(rec["fit_2"].psi))
                    and np.isfinite(rec["fit_1"].nll) and np.all(np.isfinite(rec["fit_1"].psi))):
                S["datasets"].append(d)
                S["stopped"] = f"dataset {r}: non-finite objective or ψ at a final optimum (INCOMPLETE)"
                break
            c2 = coverage(rec["fit_2"].psi, psi_true, rec["hess_2"]["H"])
            c1 = coverage(rec["fit_1"].psi, psi_true, rec["hess_1"]["H"])
            d["coverage_2"] = {**c2, "n_covered": int(c2["covered"].sum()), "hess_wall_s": rec["hess_2"]["wall_s"],
                               "hess_penalised_fd_evals": rec["hess_2"]["penalised_fd_evals"]}
            d["coverage_1_nonbinding"] = {**c1, "n_covered": int(c1["covered"].sum()),
                                          "hess_wall_s": rec["hess_1"]["wall_s"],
                                          "hess_penalised_fd_evals": rec["hess_1"]["penalised_fd_evals"]}
            d["sigma_1_nonbinding"], d["sigma_2"] = sigma(rec["fit_1"].psi), sigma(rec["fit_2"].psi)
            for stage, items in (("stage1", d["items_1"]), ("s1b", d["items_2"])):
                for s, a0, t0_ in (("T", it_true.aT, it_true.tauT), ("P", it_true.aP, it_true.tauP)):
                    e = S1A.err_summary(items[s]["a"], items[s]["tau"], a0, t0_)
                    d.setdefault("item_recovery", {})[f"{stage}_{s}"] = e
                    for j in range(len(a0)):
                        item_rows.append({"dataset": r, "stage": stage, "subscore": s, "item": j, "a_true": a0[j],
                                          "a_hat": items[s]["a"][j], "a_err": items[s]["a"][j] - a0[j],
                                          **{f"tau{c+1}_err": items[s]["tau"][j, c] - t0_[j, c] for c in range(4)}})
            d["flags"] = flags_for(d)
            d["wall_s"] = time.time() - t_ds
            d["H2"], d["H1_nonbinding"] = rec["hess_2"]["H"], rec["hess_1"]["H"]
            jdump(out / f"S1_dataset{r}.json", {**d, "fit_1_log": rec["fit_1"].log, "fit_2_log": rec["fit_2"].log})
            for key in ("H2", "H1_nonbinding"):
                d.pop(key)
            S["datasets"].append(d)
            covered_by_ds.append(c2["covered"])
            if d["flags"]:
                flagged_ds.append(r)
                S["flags"][r] = d["flags"]
            for rows, c, est in ((cov_rows, c2, rec["fit_2"].psi), (cov1_rows, c1, rec["fit_1"].psi)):
                for j, nm in enumerate(M.NAMES):
                    rows.append({"dataset": r, "param": nm, "truth": psi_true[j], "estimate": est[j], "se": c["se"][j],
                                 "lower": c["lower"][j], "upper": c["upper"][j], "z": c["z"][j],
                                 "covered": bool(c["covered"][j]), "hessian_pd": c["hessian_pd"],
                                 "at_bound": nm in (d["stage2_fit_2"] if c is c2 else d["stage2_fit_1"])["at_bound"]})
            for stage in ("stage2_fit_1", "stage2_fit_2"):
                f = d[stage]
                conv_rows.append({"dataset": r, "stage": stage, **{k: f[k] for k in (
                    "converged", "message", "n_iter", "n_eval", "grad_norm", "stopped_by_time", "maxiter_reached",
                    "penalty_evals", "non_nd_at_accepted_evals", "wall_s")}, "at_bound": ";".join(f["at_bound"])})
            for stage in ("items_1", "items_2"):
                for s in SUBSCORES:
                    f = d[stage][s]
                    conv_rows.append({"dataset": r, "stage": f"{stage}_{s}", "converged": f["converged"],
                                      "message": f["message"], "n_iter": f["n_iter"], "grad_norm": f["grad_norm"],
                                      "wall_s": f["wall_s"], "at_bound": "a_out_of_range" if f["a_out_of_range"] else ""})
            log(f"dataset {r}: ψ̂(2) covered {int(c2['covered'].sum())}/19 (PD {c2['hessian_pd']}); "
                f"ψ̂(1) {int(c1['covered'].sum())}/19 non-binding; fits {fits_w[0]:.0f}s/{fits_w[1]:.0f}s, "
                f"S1-b {timing[r]['s1b']:.0f}s; fallback {rec['reported_2']['n_fallback_patients']}, "
                f"budget {rec['reported_2']['budget_triggers']}; flags {d['flags']}; {d['wall_s']:.0f}s")
        except Exception as e:
            S["datasets"].append({"dataset": r, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]})
            S["stopped"] = f"dataset {r} raised an execution error (INCOMPLETE, E-7)"
            log(S["stopped"])
            break
        if r == 0:                                                          # runtime gate (E-9, plan §10)
            tm = timing[0]
            g = projection(tm["fits"][0], tm["fits"][1], tm["hessians"][1] / 38, tm["s1b"], tm["hessians"][0])
            g["measured"] = {"t2_s": tm["fits"][0], "t4_s": tm["fits"][1], "lg_s": tm["hessians"][1] / 38,
                             "t_s1b_s": tm["s1b"], "t_h1_s": tm["hessians"][0]}
            S["runtime_gate"] = g
            log(f"runtime gate: t2 {tm['fits'][0]:.0f}s t4 {tm['fits'][1]:.0f}s, projected "
                f"{g['projected_pilot_node_hours']:.2f} node-hours -> {g['gate_passed']}")
            if not g["gate_passed"] and not a.smoke:
                S["stopped"] = "runtime gate breached after dataset 0 (E-9): datasets 1-4 not run (INCOMPLETE)"
                break

    for name, rows in (("S2_coverage_table.csv", cov_rows), ("S2b_coverage_table_psi1_nonbinding.csv", cov1_rows),
                       ("S3_item_recovery.csv", item_rows), ("S4_convergence.csv", conv_rows)):
        with open(out / name, "w", newline="") as fh:
            if rows:
                keys = list(dict.fromkeys(k for row in rows for k in row))
                w = csv.DictWriter(fh, fieldnames=keys, restval="")
                w.writeheader()
                w.writerows(rows)
    n_complete = sum(1 for d in S["datasets"] if "coverage_2" in d)
    S["outcome"] = classify(S.get("stopped"), n_complete, n_data, covered_by_ds, flagged_ds)
    if a.smoke:
        S["outcome"]["NOTE"] = "smoke code path: NOT A RESULT"
    if covered_by_ds:
        S["coverage_counts_as_computed"] = dict(zip(M.NAMES, np.sum(covered_by_ds, axis=0).tolist()))
        S["coverage_counts_psi1_nonbinding"] = dict(zip(M.NAMES, np.sum(
            [d["coverage_1_nonbinding"]["covered"] for d in S["datasets"] if "coverage_2" in d], axis=0).tolist()))
    try:
        figures(out, S, psi_true, it_true)
    except Exception as e:                                                  # figures never affect the outcome
        S["figures_error"] = f"{type(e).__name__}: {e}"
    log(f"outcome: {S['outcome']['class']}; formal verdict {S['outcome'].get('formal_verdict')}")
    return finish(out, S, t_all, prov)


def figures(out, S, psi_true, it_true):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = [d for d in S["datasets"] if "coverage_2" in d]
    if not ds:
        return
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    k = len(M.NAMES)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, d in enumerate(ds):
        ax.scatter(np.arange(k) + 0.1 * (i - 2), d["coverage_2"]["z"], s=18, label=f"ψ̂(2) ds {d['dataset']}")
        ax.scatter(np.arange(k) + 0.1 * (i - 2), d["coverage_1_nonbinding"]["z"], s=10, marker="x", c="grey",
                   label="ψ̂(1) (non-binding)" if i == 0 else None)
    ax.axhline(1.96, ls="--", c="k", lw=0.8)
    ax.axhline(-1.96, ls="--", c="k", lw=0.8)
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels(M.NAMES, rotation=60, ha="right")
    ax.set_ylabel("z = (estimate − truth) / SE")
    ax.set_title("Step 6: z by ψ component (±1.96); binding: ψ̂(2)" + (" — SMOKE, NOT A RESULT" if S["smoke"] else ""))
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(fd / "z_by_component.png", dpi=130)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, s, a0 in ((axs[0], "T", it_true.aT), (axs[1], "P", it_true.aP)):
        for d in ds:
            ax.scatter(a0, d["items_1"][s]["a"], s=12, c="tab:red", alpha=0.6)
            ax.scatter(a0, d["items_2"][s]["a"], s=12, c="tab:blue", alpha=0.6)
        lim = [min(a0) * 0.9, max(a0) * 1.25]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_title(f"{s}: discrimination (red: stage 1 N(0,1); blue: S1-b)")
        ax.set_xlabel("true a")
        ax.set_ylabel("estimated a")
    fig.tight_layout()
    fig.savefig(fd / "item_discrimination.png", dpi=130)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, s, t0 in ((axs[0], "T", it_true.tauT), (axs[1], "P", it_true.tauP)):
        for d in ds:
            ax.scatter(t0.ravel(), np.asarray(d["items_1"][s]["tau"]).ravel(), s=8, c="tab:red", alpha=0.5)
            ax.scatter(t0.ravel(), np.asarray(d["items_2"][s]["tau"]).ravel(), s=8, c="tab:blue", alpha=0.5)
        lim = [t0.min() - 0.3, t0.max() + 0.3]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_title(f"{s}: thresholds (red: stage 1; blue: S1-b)")
        ax.set_xlabel("true τ")
        ax.set_ylabel("estimated τ")
    fig.tight_layout()
    fig.savefig(fd / "item_thresholds.png", dpi=130)
    plt.close(fig)


def finish(out, S, t_all, prov):
    S["runtime_s"] = time.time() - t_all
    S["provenance"] = {k: prov[k] for k in ("host", "cpu", "jobs", "slurm_job", "numpy", "python", "smoke")}
    if "outcome" not in S:
        S["outcome"] = classify(S.get("stopped"), 0, N_DATA, [], [])
    jdump(out / "summary.json", S)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    (out / "MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files))
    log(f"done in {S['runtime_s']:.0f}s; outcome {S['outcome']['class']}; stopped {S.get('stopped')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
