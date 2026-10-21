"""Stage 2 M2, Step 5: S1-a diagnostic (Rev 2 §5, §6 step 5). Reported only; no criterion.

Frozen execution plan: docs/cluster_phase/34_STAGE2_STEP5_S1A_PLAN_FINAL_20261019.md (reviewer
decisions D-1..D-5 in its §6). Synthetic data only: the 5 recovery datasets, seeds [42, 45, 1, r],
a-priori truth on the stage-1 scale (scripts/45_m2_pilot.py), real visit design. For each dataset
and subscore the items are fitted twice with the same per-visit composite likelihood, 61-node GH,
L-BFGS-B and start: (1) θ ~ N(0, 1) (the pilot path, reproduced exactly) and (2) the oracle
θ_v ~ N(M_ij, K_ij,jj) at the true ψ (D-1). At ψ_true the stage-2 log-likelihood and gradient
(Laplace, Q = 15; non-ND patients by AGQ G = 5, exact gradient, D-4) are evaluated for the true,
N(0, 1) and oracle item sets, with the first-order shift Δψ = −H⁻¹g (H: FD Hessian at ψ_true,
true items) in Step 4 SE units (D-2). No stage-2 optimisation; no real responses; no labels.

    scripts/cluster/submit.sh ... -w node002 -- scripts/51_m2a1_s1a_true_latent_items.py --out <exec dir>/step5 --jobs 32
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
from scipy.optimize import minimize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import grm as G  # noqa: E402
from ppmi_subtype import m2_grm as M  # noqa: E402
from ppmi_subtype import m2_quad as MQ  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

FROZEN = ROOT / "docs/cluster_phase/_frozen"
LOCKS = {"doc16": "16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.sha256", "rev2": "21_STAGE2_DOC16_AMENDMENT1_REV2.sha256",
         "doc25": "25_STAGE2_ADDENDUM_A_FINAL.sha256", "doc27": "27_STAGE2_EXACT_AGQ_GRADIENT.sha256",
         "plan29": "29_STAGE2_STEP3_PLAN.sha256", "plan31": "31_STAGE2_STEP4_PLAN.sha256",
         "plan34": "34_STAGE2_STEP5_PLAN.sha256"}
PILOT = ROOT / "experiments/20261016T200350Z_stage2_m2_pilot/results"
STEP4 = ROOT / "experiments/20261018T184713Z_stage2_step4_fixed_item_recovery"
STEP4_SUMMARY_SHA = "00433150c23c90ec"
IDENT = ROOT / "experiments/20261017T233307Z_stage2_m2_exact_agq/bench/S2_laplace_fit.json"
PANEL_MD5 = "fb038ddb"
Q, G_REPORTED, N_DATA = 15, 5, 5
SUBSCORES = ("T", "P")
ITEM_SETS = ("true", "N01", "oracle")
METRICS = ("a_corr_T", "a_maxerr_T", "tau_maxerr_T", "a_corr_P", "a_maxerr_P", "tau_maxerr_P")


def sha(p, kind="sha256"):
    h = hashlib.new(kind)
    h.update(Path(p).read_bytes())
    return h.hexdigest()


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


# ---------------------------------------------------------------------------
#  Oracle-latent item likelihood: stage1_negloglik with a per-visit node grid
# ---------------------------------------------------------------------------
def latent_negloglik(phi, Y, m, s, nodes=None):
    """Negative per-visit marginal log-likelihood and analytic gradient, θ_v ~ N(m_v, s_v²).

    Y (V, K) responses (NaN = missing); m, s (V,). θ_qv = m_v + s_v x_q on the probabilists'
    GH nodes x_q with the N(0, 1) weights. m = 0, s = 1 gives grm.stage1_negloglik.
    """
    V, K = Y.shape
    x, w = nodes if nodes is not None else G.gh_nodes()
    a, tau = G.unpack_items(phi, K)
    th = np.asarray(m, float)[None, :] + np.asarray(s, float)[None, :] * x[:, None]   # (Q, V)
    lp = G.category_logprob(th, a, tau, Y[None])                          # (Q, V, K)
    ll = lp.sum(-1) + np.log(w)[:, None]                                  # (Q, V)
    mx = ll.max(0)
    post = np.exp(ll - mx)
    norm = post.sum(0)
    total = float((mx + np.log(norm)).sum())
    post /= norm

    F, _ = G._cum(th, a, tau)                                             # (Q, V, K, 6)
    s_ = F * (1 - F)
    yv = np.nan_to_num(Y, nan=0).astype(int)[None]
    yv = np.broadcast_to(yv, (len(x), V, K))
    P = np.maximum(np.exp(lp), 1e-300)
    taup = np.concatenate([np.zeros((K, 1)), tau, np.zeros((K, 1))], 1)
    thv = th[:, :, None]
    sc = np.take_along_axis(s_, yv[..., None], -1)[..., 0]
    sn = np.take_along_axis(s_, (yv + 1)[..., None], -1)[..., 0]
    tc = taup[np.arange(K)[None, None, :], yv]
    tn = taup[np.arange(K)[None, None, :], yv + 1]
    has_c, has_n = yv >= 1, yv <= 3
    dlogP_da = (np.where(has_c, sc * (thv - tc), 0) - np.where(has_n, sn * (thv - tn), 0)) / P
    miss = np.broadcast_to(np.isnan(Y)[None], dlogP_da.shape)
    dlogP_da = np.where(miss, 0, dlogP_da)
    dtau = np.zeros((len(x), V, K, 4))
    for c in range(1, 5):
        dtau[..., c - 1] = (np.where(yv == c, -a * sc, 0) + np.where(yv + 1 == c, a * sn, 0)) / P
    dtau = np.where(miss[..., None], 0, dtau)
    g_a = np.einsum("qv,qvk->k", post, dlogP_da)
    g_tau = np.einsum("qv,qvkc->kc", post, dtau)
    p = np.asarray(phi, float).reshape(K, G.N_CAT)
    grad = np.zeros((K, G.N_CAT))
    grad[:, 0] = g_a * a
    grad[:, 1] = g_tau.sum(1)
    for k in range(2, 5):
        grad[:, k] = g_tau[:, k - 1:].sum(1) * np.exp(p[:, k])
    return -total, -grad.ravel()


def visit_latent(groups, psi):
    """Per-visit oracle mean and SD (flatten order) per subscore at ψ (D-1)."""
    beta, D, _, S, _ = M.unpack(psi)
    out = {s: ([], []) for s in SUBSCORES}
    for n, g in groups.items():
        Mn = np.einsum("k,kmi->mi", beta, M.mean_design(g.T, g.X))
        K, _ = M.covariance(g.T, D, S)
        dg = np.einsum("mii->mi", K)
        for s, sl in (("T", slice(0, n)), ("P", slice(n, 2 * n))):
            out[s][0].append(Mn[:, sl].ravel())
            out[s][1].append(np.sqrt(dg[:, sl]).ravel())
    return {s: (np.concatenate(v[0]), np.concatenate(v[1])) for s, v in out.items()}


def fit_oracle(Y, m, s, maxiter=3000):
    """One start at initial_items, as the pilot N(0, 1) fit (D-3)."""
    nodes = G.gh_nodes()
    t0 = time.perf_counter()
    r = minimize(latent_negloglik, G.initial_items(Y), args=(Y, m, s, nodes), jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter})
    a, tau = G.unpack_items(r.x, Y.shape[1])
    return {"nll": float(r.fun), "converged": bool(r.success), "message": str(r.message), "n_iter": int(r.nit),
            "n_eval": int(r.nfev), "grad_norm": float(np.linalg.norm(r.jac)), "a": a, "tau": tau,
            "wall_s": time.perf_counter() - t0}


# ---------------------------------------------------------------------------
#  Item-fit tasks (process pool; each task is a serial deterministic fit)
# ---------------------------------------------------------------------------
_FIT: dict = {}


def _fit_task(args):
    kind, r, s, maxiter = args
    Y = _FIT["Y"][(r, s)]
    if kind == "N01":
        t0 = time.perf_counter()
        best, check, res = _FIT["P"].fit_items_multistart(Y, (100 if s == "T" else 200) + r, starts=1)
        return kind, r, s, {"nll": best["nll"], "converged": best["converged"], "message": best["message"],
                            "a": best["a"], "tau": best["tau"], "a_out_of_range": best["a_out_of_range"],
                            "wall_s": time.perf_counter() - t0}
    m, sd = _FIT["lat"][(r, s)]
    return kind, r, s, fit_oracle(Y, m, sd, maxiter)


def run_pool(tasks, jobs):
    if jobs <= 1:
        return [_fit_task(t) for t in tasks]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(min(jobs, len(tasks)), mp_context=mp.get_context("fork")) as ex:
        return list(ex.map(_fit_task, tasks))


# ---------------------------------------------------------------------------
#  Stage-2 reported-evaluation context with gradient (plan §5.4, D-4)
# ---------------------------------------------------------------------------
_REP: dict = {}


def _rep_init(chunks, items):
    _REP.update(chunks=chunks, items=items)


def _rep_task(args):
    I, key, psi, want_grad = args
    d, it = _REP["chunks"][(I, key)], _REP["items"][I]
    t0 = time.perf_counter()
    info: dict = {}
    try:
        val, b, nd, sH, gr = MQ.group_laplace(psi, d, it, Q, None, want_grad, info=info)
    except np.linalg.LinAlgError as e:
        return key, {"failure": f"linalg: {e}", "n": len(d.T), "wall_s": time.perf_counter() - t0}
    if info.get("failed"):
        return key, {"failure": "mode_budget", "n": len(d.T), "budget": info.get("budget"),
                     "wall_s": time.perf_counter() - t0}
    fb = None
    if np.any(~nd):
        r = np.flatnonzero(~nd)
        finfo: dict = {}
        try:
            fv, _, fnd, fgr = MQ.group_agq(psi, d.rows(r), it, Q, G_REPORTED, b[r], want_grad, info=finfo,
                                           gradient="exact")
        except np.linalg.LinAlgError as e:
            return key, {"failure": f"fallback linalg: {e}", "n": len(d.T), "wall_s": time.perf_counter() - t0}
        if finfo.get("failed"):
            return key, {"failure": "fallback mode_budget", "n": len(d.T), "wall_s": time.perf_counter() - t0}
        val = val.copy()
        val[r] = fv
        if want_grad:
            gr = gr.copy()
            gr[r] = fgr
        fb = {"rows": r.tolist(), "n": int(len(r)), "agq_non_nd": int((~fnd).sum()),
              "gradient_inexact_patients": int(finfo.get("gradient_inexact_patients", 0))}
    return key, {"val": float(val.sum()), "grad": None if gr is None else gr.sum(0), "n": len(d.T),
                 "finite": bool(np.all(np.isfinite(val)) and (gr is None or np.all(np.isfinite(gr)))),
                 "fallback": fb, "iterations": info.get("iterations", 0), "halvings": info.get("halvings", 0),
                 "wall_s": time.perf_counter() - t0}


class Reported:
    """−ℓ and gradient over the cohort for item set I; records every fallback invocation."""

    def __init__(self, pool, keys, I, label_log):
        self.pool, self.keys, self.I, self.records = pool, keys, I, label_log
        self.label = None

    def __call__(self, psi, want_grad=True):
        t0 = time.perf_counter()
        psi = np.asarray(psi, float)
        args = [(self.I, k, psi, want_grad) for k in self.keys]
        out = list(self.pool.map(_rep_task, args)) if self.pool is not None else [_rep_task(a) for a in args]
        fails = {str(k): r["failure"] for k, r in out if "failure" in r}
        fbs = [{"chunk": list(k), **r["fallback"]} for k, r in out if r.get("fallback")]
        nonfinite = [str(k) for k, r in out if "failure" not in r and not r["finite"]]
        rec = {"item_set": self.I, "label": self.label, "failures": fails, "fallbacks": fbs,
               "n_fallback_patients": sum(f["n"] for f in fbs), "nonfinite_chunks": nonfinite,
               "halvings": sum(r.get("halvings", 0) for _, r in out),
               "n_patients": sum(r["n"] for _, r in out)}
        if fails or nonfinite:
            val, grad = np.nan, np.full(M.N_PSI, np.nan)
        else:
            val = -sum(r["val"] for _, r in out)
            grad = -np.sum([r["grad"] for _, r in out], axis=0) if want_grad else None
        rec.update(value=val, wall_s=time.perf_counter() - t0)
        self.records.append(rec)
        return (val, grad) if want_grad else val


def item_metrics(fT, fP, it_true):
    """The pilot's six item-recovery metrics, computed exactly as scripts/45 step 1."""
    return {"a_corr_T": float(np.corrcoef(fT["a"], it_true.aT)[0, 1]),
            "a_maxerr_T": float(np.max(np.abs(fT["a"] - it_true.aT))),
            "tau_maxerr_T": float(np.max(np.abs(fT["tau"] - it_true.tauT))),
            "a_corr_P": float(np.corrcoef(fP["a"], it_true.aP)[0, 1]),
            "a_maxerr_P": float(np.max(np.abs(fP["a"] - it_true.aP))),
            "tau_maxerr_P": float(np.max(np.abs(fP["tau"] - it_true.tauP)))}


def err_summary(a, tau, a0, tau0):
    ea, et = a - a0, tau - tau0
    return {"a_maxabs": float(np.max(np.abs(ea))), "a_mean": float(ea.mean()), "a_rmse": float(np.sqrt((ea ** 2).mean())),
            "a_corr": float(np.corrcoef(a, a0)[0, 1]), "tau_maxabs": float(np.max(np.abs(et))),
            "tau_mean": float(et.mean()), "tau_rmse": float(np.sqrt((et ** 2).mean()))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--smoke", action="store_true",
                    help="code path only (plan §11): 1 dataset, 60 patients, maxiter 20; not a result")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t_all = time.time()
    S: dict = {"status": "reported only; no acceptance criterion (Rev 2 §5, R4)", "smoke": a.smoke,
               "datasets": [], "integrity": {}}

    # ---- 1. hashes (plan §9.1) ----------------------------------------------------------------
    locks = {k: (FROZEN / v).read_text().split() for k, v in LOCKS.items()}
    base4 = json.loads((STEP4 / "00_baseline.json").read_text())
    pilot_prov = json.loads((PILOT / "provenance.json").read_text())
    code_now = {f: sha(ROOT / f) for f in sorted(set(base4["code_sha256"]) | set(pilot_prov["code_sha256"]))}
    prov = {"plan": "docs/cluster_phase/34_STAGE2_STEP5_S1A_PLAN_FINAL_20261019.md",
            "docs": {k: {"file": v[1], "frozen": v[0], "current": sha(ROOT / v[1])} for k, v in locks.items()},
            "code_sha256": code_now, "driver_sha256": sha(Path(__file__)),
            "code_vs_step4_baseline": {f: code_now[f] == h for f, h in base4["code_sha256"].items()},
            "code_vs_pilot": {f: code_now[f] == h for f, h in pilot_prov["code_sha256"].items()},
            "panel_md5": sha(load_config().path("outputs") / "visit_panel.csv", "md5"),
            "inputs_sha256": {"pilot_step1_recovery": sha(PILOT / "step1_recovery.json"),
                              "step4_summary": sha(STEP4 / "step4_run2/summary.json"),
                              "identity_ref": sha(IDENT)},
            "Q": Q, "G_reported": G_REPORTED, "seeds": {r: [42, 45, 1, r] for r in range(N_DATA)},
            "host": platform.node(), "jobs": a.jobs, "slurm_job": os.environ.get("SLURM_JOB_ID"), "smoke": a.smoke,
            "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "numpy": np.__version__, "python": sys.version.split()[0],
            "cpu": next((ln.split(":", 1)[1].strip() for ln in subprocess.run(
                ["lscpu"], capture_output=True, text=True).stdout.splitlines() if ln.startswith("Model name")), "?")}
    jdump(out / "S0_provenance.json", prov)
    bad = [k for k, d in prov["docs"].items() if d["frozen"] != d["current"]]
    bad += [f for f, ok in prov["code_vs_step4_baseline"].items() if not ok]
    bad += [f for f, ok in prov["code_vs_pilot"].items() if not ok]
    if not prov["panel_md5"].startswith(PANEL_MD5):
        bad.append("panel_md5")
    if not prov["inputs_sha256"]["step4_summary"].startswith(STEP4_SUMMARY_SHA):
        bad.append("step4_summary")
    S["integrity"]["hashes_ok"] = not bad
    if bad:
        S["stopped"] = f"hash mismatch, nothing computed: {bad}"
        return finish(out, S, t_all, prov)
    log("hashes ok")

    # ---- 2. unit tests of the new function (plan §9.2) ----------------------------------------
    pt = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                         str(ROOT / "tests/test_s1a_latent_items.py")], capture_output=True, text=True, cwd=ROOT)
    S["integrity"]["unit_tests"] = {"returncode": pt.returncode, "tail": pt.stdout.strip().splitlines()[-3:]}
    if pt.returncode != 0:
        S["stopped"] = "unit tests failed: nothing fitted"
        return finish(out, S, t_all, prov)
    log(f"unit tests: {S['integrity']['unit_tests']['tail'][-1]}")

    # ---- 3. truth, datasets, identity check (plan §4, §9.2) -----------------------------------
    spec = importlib.util.spec_from_file_location("m2_pilot", ROOT / "scripts/45_m2_pilot.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    v, times, treat, YTd, YPd, *_ = P.load_cohort(load_config())    # design only
    g0 = M.make_groups(times, treat, YTd, YPd)
    psi_true, it_true, scale = P.truth_on_stage1_scale(g0)
    pilot_rec = json.loads((PILOT / "step1_recovery.json").read_text())
    S["integrity"]["truth_equals_pilot"] = bool(np.array_equal(psi_true, np.array(pilot_rec["psi_true"])))
    if not S["integrity"]["truth_equals_pilot"]:
        S["stopped"] = "truth differs from the pilot record: nothing fitted"
        return finish(out, S, t_all, prov)
    S["psi_true"], S["names"] = psi_true, M.NAMES

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
    else:
        ref = json.loads(IDENT.read_text())
        nll = MQ.negloglik(np.array(ref["psi"]), data[0], it_true, 10, want_grad=False)   # serial, as Step 4
        S["integrity"]["identity_check"] = {"nll": nll, "recorded": ref["nll"], "equal": nll == ref["nll"]}
        log(f"identity check: {nll} vs {ref['nll']} -> {nll == ref['nll']}")
        if nll != ref["nll"]:
            S["stopped"] = "identity check failed: synthetic data differ from the recorded ones; nothing fitted"
            return finish(out, S, t_all, prov)
    S["integrity"]["datasets_hash"] = {r: hashlib.sha256(np.concatenate([x.ravel() for x in P.flatten(g)[:2]])
                                                         .tobytes()).hexdigest() for r, g in data.items()}

    # ---- 4. N(0, 1) item fits and pilot reproduction (plan §5.1, D-5) --------------------------
    Ys = {(r, s): P.flatten(g)[0 if s == "T" else 1].astype(float) for r, g in data.items() for s in SUBSCORES}
    lat = {}
    for r, g in data.items():
        vl = visit_latent(g, psi_true)
        for s in SUBSCORES:
            lat[(r, s)] = vl[s]
    _FIT.update(Y=Ys, lat=lat, P=P)
    maxiter = 20 if a.smoke else 3000
    t0 = time.time()
    fits = {(k, r, s): res for k, r, s, res in run_pool([("N01", r, s, maxiter) for r in data for s in SUBSCORES], a.jobs)}
    S["n01_fits_wall_s"] = time.time() - t0
    nonfinite = [k for k, f in fits.items() if not np.isfinite(f["nll"])]
    repro = {}
    for r in data:
        got = item_metrics(fits[("N01", r, "T")], fits[("N01", r, "P")], it_true)
        rec = pilot_rec["items"][r]
        repro[r] = {m: {"recomputed": got[m], "pilot": rec[m], "equal": got[m] == rec[m]} for m in METRICS}
    S["integrity"]["pilot_reproduction"] = repro
    S["integrity"]["pilot_reproduction_exact"] = all(x["equal"] for d in repro.values() for x in d.values())
    log(f"N(0,1) fits {S['n01_fits_wall_s']:.0f}s; pilot reproduction exact: {S['integrity']['pilot_reproduction_exact']}")
    if nonfinite:
        S["stopped"] = f"non-finite N(0,1) item objective {nonfinite}"
        return finish(out, S, t_all, prov, fits)
    if not a.smoke and not S["integrity"]["pilot_reproduction_exact"]:
        S["stopped"] = "pilot N(0,1) item-fit records not reproduced exactly (D-5): oracle fits not run"
        return finish(out, S, t_all, prov, fits)

    # ---- 5. oracle-latent item fits (plan §5.2) -----------------------------------------------
    S["construction_check"] = {}
    for (r, s), (m, sd) in lat.items():
        S["construction_check"][f"{r}{s}"] = {"pooled_mean": float(m.mean()),
                                              "pooled_var": float((sd ** 2).mean() + m.var()),
                                              "s_min": float(sd.min()), "s_max": float(sd.max()),
                                              "m_min": float(m.min()), "m_max": float(m.max())}
    t0 = time.time()
    for k, r, s, res in run_pool([("oracle", r, s, maxiter) for r in data for s in SUBSCORES], a.jobs):
        fits[(k, r, s)] = res
    S["oracle_fits_wall_s"] = time.time() - t0
    nonfinite = [k for k, f in fits.items() if not np.isfinite(f["nll"])]
    if nonfinite:
        S["stopped"] = f"non-finite oracle item objective {nonfinite}"
        return finish(out, S, t_all, prov, fits)
    log(f"oracle fits {S['oracle_fits_wall_s']:.0f}s")

    # ---- 6. per dataset: item comparison, item likelihoods, stage-2 diagnostics ----------------
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    err_rows, shift_rows = [], []
    true_items = {"T": (it_true.aT, it_true.tauT), "P": (it_true.aP, it_true.tauP)}
    nodes = G.gh_nodes()
    for r, sg in data.items():
        t_ds = time.time()
        D_: dict = {"dataset": r, "seed": [42, 45, 1, r]}
        try:
            sets = {"true": true_items}
            for k in ("N01", "oracle"):
                sets[k] = {s: (fits[(k, r, s)]["a"], fits[(k, r, s)]["tau"]) for s in SUBSCORES}
            items_rec = {"fits": {f"{k}_{s}": {kk: vv for kk, vv in fits[(k, r, s)].items()}
                                  for k in ("N01", "oracle") for s in SUBSCORES},
                         "true": {s: {"a": true_items[s][0], "tau": true_items[s][1]} for s in SUBSCORES},
                         "errors": {}, "item_objectives": {}}
            for s in SUBSCORES:
                a0, t0_ = true_items[s]
                for k in ("N01", "oracle"):
                    a1, t1 = sets[k][s]
                    items_rec["errors"][f"{k}_{s}"] = err_summary(a1, t1, a0, t0_)
                aN, tN = sets["N01"][s]
                aO, tO = sets["oracle"][s]
                for j in range(len(a0)):
                    row = {"dataset": r, "subscore": s, "item": j, "a_true": a0[j],
                           "a_N01": aN[j], "a_oracle": aO[j], "aerr_N01": aN[j] - a0[j], "aerr_oracle": aO[j] - a0[j],
                           "a_N01_minus_oracle": aN[j] - aO[j]}
                    for c in range(4):
                        row.update({f"tau{c+1}_true": t0_[j, c], f"tau{c+1}err_N01": tN[j, c] - t0_[j, c],
                                    f"tau{c+1}err_oracle": tO[j, c] - t0_[j, c],
                                    f"tau{c+1}_N01_minus_oracle": tN[j, c] - tO[j, c]})
                    err_rows.append(row)
                # §5.3a item objectives at each item set
                Y = Ys[(r, s)]
                m, sd = lat[(r, s)]
                for I in ITEM_SETS:
                    phi = G.pack_items(*sets[I][s])
                    f1, g1 = G.stage1_negloglik(phi, Y, nodes)
                    f2, g2 = latent_negloglik(phi, Y, m, sd, nodes)
                    items_rec["item_objectives"][f"{I}_{s}"] = {
                        "N01_nll": f1, "N01_grad_norm": float(np.linalg.norm(g1)),
                        "oracle_nll": f2, "oracle_grad_norm": float(np.linalg.norm(g2))}
            jdump(out / f"S1_items_dataset{r}.json", items_rec)
            D_["item_errors"] = items_rec["errors"]
            D_["item_objectives"] = items_rec["item_objectives"]
            D_["item_fit_status"] = {f"{k}_{s}": {x: fits[(k, r, s)].get(x) for x in
                                                  ("nll", "converged", "message", "n_iter", "grad_norm", "wall_s")}
                                     for k in ("N01", "oracle") for s in SUBSCORES}

            # stage-2 at ψ_true (§5.4)
            its = {I: M.Items(sets[I]["T"][0], sets[I]["T"][1], sets[I]["P"][0], sets[I]["P"][1]) for I in ITEM_SETS}
            chunks, keys = {}, None
            for I in ITEM_SETS:
                ob = MQ.Objective(sg, its[I], Q)                     # chunking only (jobs = 1: no pool)
                keys = list(ob.chunks)
                for k, d in ob.chunks.items():
                    chunks[(I, k)] = d
            records: list = []
            pool = ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("fork"), initializer=_rep_init,
                                       initargs=(chunks, its)) if a.jobs > 1 else None
            if pool is None:
                _rep_init(chunks, its)
            try:
                ev = {}
                for I in ITEM_SETS:
                    R = Reported(pool, keys, I, records)
                    R.label = "psi_true"
                    tt = time.perf_counter()
                    f, gI = R(psi_true)
                    ev[I] = {"nll": f, "loglik": -f, "grad": gI, "grad_norm": float(np.linalg.norm(gI)),
                             "wall_s": time.perf_counter() - tt, "record": records[-1]}
                # consistency of the evaluator with evaluate_reported and Objective (true items; reported)
                rep = MQ.evaluate_reported(psi_true, sg, it_true, Q, G_REPORTED)
                ob = MQ.Objective(sg, it_true, Q, jobs=a.jobs)
                try:
                    ofv = ob(psi_true, want_grad=False)
                finally:
                    ob.close()
                D_["evaluator_consistency"] = {"reported_with_grad_loglik": ev["true"]["loglik"],
                                               "evaluate_reported_loglik": rep["loglik"],
                                               "evaluate_reported_n_fallback": rep["n_fallback_patients"],
                                               "evaluate_reported_budget": rep["budget_triggers"],
                                               "objective_value": -ofv,
                                               "diff_vs_evaluate_reported": ev["true"]["loglik"] - rep["loglik"],
                                               "diff_vs_objective": ev["true"]["loglik"] - (-ofv)}
                Rt = Reported(pool, keys, "true", records)
                Rt.label = "fd_hessian"
                th = time.perf_counter()
                H = MQ.fd_hessian(Rt, psi_true)
                h_s = time.perf_counter() - th
            finally:
                if pool is not None:
                    pool.shutdown()
            hess_recs = [x for x in records if x["label"] == "fd_hessian"]
            finite_H = bool(np.all(np.isfinite(H)))
            eig = np.linalg.eigvalsh(H) if finite_H else np.full(M.N_PSI, np.nan)
            pd = bool(finite_H and eig.min() > 0)
            s4 = json.loads((STEP4 / f"step4_run2/S1_dataset{r}.json").read_text())
            se4, psi4 = np.array(s4["se"]), np.array(s4["psi_hat"])
            if a.smoke:
                se4 = np.full(M.N_PSI, np.nan)
            Hinv = np.linalg.inv(H) if pd else np.full((M.N_PSI, M.N_PSI), np.nan)
            se0 = np.sqrt(np.diag(Hinv))
            dpsi = {I: -Hinv @ ev[I]["grad"] for I in ITEM_SETS}
            D_["stage2"] = {
                "evaluations": {I: {k: ev[I][k] for k in ("loglik", "grad", "grad_norm", "wall_s")} |
                                {"n_fallback_patients": ev[I]["record"]["n_fallback_patients"],
                                 "fallbacks": ev[I]["record"]["fallbacks"], "failures": ev[I]["record"]["failures"],
                                 "nonfinite_chunks": ev[I]["record"]["nonfinite_chunks"]} for I in ITEM_SETS},
                "loglik_diff": {"N01_minus_true": ev["N01"]["loglik"] - ev["true"]["loglik"],
                                "oracle_minus_true": ev["oracle"]["loglik"] - ev["true"]["loglik"],
                                "N01_minus_oracle": ev["N01"]["loglik"] - ev["oracle"]["loglik"]},
                "hessian": {"finite": finite_H, "pd": pd, "min_eig": float(np.nanmin(eig)) if finite_H else np.nan,
                            "max_eig": float(np.nanmax(eig)) if finite_H else np.nan, "wall_s": h_s,
                            "n_evals": len(hess_recs),
                            "n_fallback_patients_total": sum(x["n_fallback_patients"] for x in hess_recs),
                            "n_evals_with_fallback": sum(1 for x in hess_recs if x["n_fallback_patients"]),
                            "n_evals_failed": sum(1 for x in hess_recs if x["failures"] or x["nonfinite_chunks"])},
                "shift": {I: dpsi[I] for I in ITEM_SETS},
                "shift_z_step4": {I: dpsi[I] / se4 for I in ITEM_SETS},
                "shift_z_se0": {I: dpsi[I] / se0 for I in ITEM_SETS},
                "contrast_z_step4": {"N01_minus_true": (dpsi["N01"] - dpsi["true"]) / se4,
                                     "oracle_minus_true": (dpsi["oracle"] - dpsi["true"]) / se4,
                                     "N01_minus_oracle": (dpsi["N01"] - dpsi["oracle"]) / se4},
                "se_step4": se4, "se0_at_psi_true": se0,
                "first_order_check": {"step4_psi_hat_minus_truth": psi4 - psi_true,
                                      "shift_true": dpsi["true"],
                                      "max_abs_diff_z": float(np.nanmax(np.abs((psi4 - psi_true - dpsi["true"]) / se4)))
                                      if pd and not a.smoke else np.nan}}
            jdump(out / f"S3_downstream_dataset{r}.json", {**D_["stage2"], "H": H, "eig": eig, "records": records,
                                                            "evaluator_consistency": D_["evaluator_consistency"]})
            for j, nm in enumerate(M.NAMES):
                row = {"dataset": r, "param": nm, "truth": psi_true[j], "se_step4": se4[j], "se0": se0[j],
                       "step4_dev_z": (psi4[j] - psi_true[j]) / se4[j]}
                for I in ITEM_SETS:
                    row[f"shift_{I}"] = dpsi[I][j]
                    row[f"shift_z_{I}"] = dpsi[I][j] / se4[j]
                    row[f"grad_{I}"] = ev[I]["grad"][j]
                row["contrast_z_N01_minus_true"] = (dpsi["N01"][j] - dpsi["true"][j]) / se4[j]
                row["contrast_z_oracle_minus_true"] = (dpsi["oracle"][j] - dpsi["true"][j]) / se4[j]
                row["contrast_z_N01_minus_oracle"] = (dpsi["N01"][j] - dpsi["oracle"][j]) / se4[j]
                row["hessian_pd"] = pd
                shift_rows.append(row)
            D_["n_fallback_patients_all_evals"] = sum(x["n_fallback_patients"] for x in records)
            D_["fallback_invocations"] = sum(len(x["fallbacks"]) for x in records)
            D_["failed_evals"] = sum(1 for x in records if x["failures"] or x["nonfinite_chunks"])
            D_["wall_s"] = time.time() - t_ds
            S["datasets"].append(D_)
            log(f"dataset {r}: ll true {ev['true']['loglik']:.3f}, N01 {ev['N01']['loglik']:.3f}, oracle "
                f"{ev['oracle']['loglik']:.3f}; H PD {pd}; fallback patients {D_['n_fallback_patients_all_evals']}; "
                f"{D_['wall_s']:.0f}s")
            if any(x["nonfinite_chunks"] for x in records):
                S["stopped"] = f"dataset {r}: non-finite stage-2 value or gradient (plan §9.4)"
                break
        except Exception as e:
            S["datasets"].append({"dataset": r, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]})
            log(f"dataset {r}: ERROR {e}")
            S["stopped"] = f"dataset {r} raised an execution error"
            break
    for name, rows in (("S2_item_errors.csv", err_rows), ("S4_downstream_shift.csv", shift_rows)):
        with open(out / name, "w", newline="") as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    return finish(out, S, t_all, prov)


def finish(out, S, t_all, prov, fits=None):
    if fits is not None:
        jdump(out / "S1_item_fits_partial.json", {f"{k}_{r}_{s}": v for (k, r, s), v in fits.items()})
    S["runtime_s"] = time.time() - t_all
    S["provenance"] = {k: prov[k] for k in ("host", "cpu", "jobs", "slurm_job", "numpy", "python", "smoke")}
    jdump(out / "summary.json", S)
    log(f"done in {S['runtime_s']:.0f}s; stopped {S.get('stopped')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
