"""EXPLORATORY DIAGNOSTIC (Step 6 failure investigation, Phase B): D6, D4, D5. Not a recovery test.

Step 6 (doc 38) formally FAILED its frozen criterion; nothing here re-scores it, replaces its intervals,
or defines a new estimator. Uses only the five frozen synthetic datasets (seeds [42, 45, 1, r], the
a-priori truth, the real design), regenerated exactly, plus saved Step 5/6 item and ψ estimates.
No new random numbers are drawn.

  d6  Stage 2 refit with Step 5's saved true-latent ("oracle") items: Laplace Q = 15, default ψ, cold
      modes, maxiter 1000, FD Hessian (h = 1e-4), reported evaluation (G = 5) — the Step 4/6 settings.
  d4  Item uncertainty and item-stage-2 coupling at the saved Step 6 estimates. Stacked estimating
      equations for the whole S1-b chain (φ1: N(0,1) items; ψ1; φ2: S1-b items; ψ2), patient-clustered
      sandwich V = J⁻¹ B J⁻ᵀ with J from central finite differences of analytic gradients and B from
      per-patient scores. Checks: score additivity, affine-invariance identity C·v_φ + H·v_ψ ≈ 0,
      Hessian symmetry, saved-NLL reproduction.
  d5  Further S1-b steps from the saved Step 6 state (ψ̂(2), S1-b items): exactly the Step 6 step (3)-(4)
      map (items refitted under N(M_ij, K_ij,jj) at the current ψ̂ from initial_items; stage 2 from the
      default ψ with cold modes), k = 3..6, per-dataset settle rule max_j |Δψ_j|/SE_j < 0.05 (pre-declared).

    scripts/cluster/submit.sh ... -w node002 -- scripts/54_m2_step6_diag_compute.py d6 --diag <diag dir> --jobs 32
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
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import grm as G  # noqa: E402
from ppmi_subtype import m2_grm as M  # noqa: E402
from ppmi_subtype import m2_quad as MQ  # noqa: E402

S5 = ROOT / "experiments/20261018T215102Z_stage2_step5_s1a/step5"
S6 = ROOT / "experiments/20261018T222857Z_stage2_step6_full_pipeline_recovery/step6"
IDENT = ROOT / "experiments/20261017T233307Z_stage2_m2_exact_agq/bench/S2_laplace_fit.json"
IDENT_NLL = 57022.74225348176
Q, G_REPORTED, N_DATA, CHUNK = 15, 5, 5, 64
H_ITEM, H_PSI = 1e-4, 1e-4
SETTLE_Z, K_MAX = 0.05, 6
SUB = ("T", "P")
FAILED = ("beta0_P", "Le00_log", "Le11_log")
LABEL = "EXPLORATORY DIAGNOSTIC — not a recovery test; Step 6 remains FAILED (doc 38); D1 remains NOT REPRODUCED"


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
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=_conv))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
#  Pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def pack_set(items):
    """M.Items -> (φ_T, φ_P) in grm's packed parameterisation."""
    return G.pack_items(items.aT, items.tauT), G.pack_items(items.aP, items.tauP)


def items_from_phi(phiT, phiP, KT, KP):
    aT, tT = G.unpack_items(phiT, KT)
    aP, tP = G.unpack_items(phiP, KP)
    return M.Items(aT, tT, aP, tP)


def stacked_sandwich(Jm, U):
    """M-estimation sandwich for stacked estimating equations: V = J⁻¹ (Σ_i u_i u_iᵀ) J⁻ᵀ."""
    B = U.T @ U
    Ji = np.linalg.inv(Jm)
    return Ji @ B @ Ji.T, B


def patient_slices(pid):
    """Contiguous visit slices per patient (flatten order)."""
    pid = np.asarray(pid)
    cut = np.flatnonzero(np.diff(pid)) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(pid)]])
    return [slice(a, b) for a, b in zip(starts, ends)]


def affine_tangent(it, psi, s, par, relabel, e=1e-6):
    """Tangent (v_φ, v_ψ) at c = 1, d = 0 of the relabelling θ'_s = (θ_s − d)/c of subscore s:
    items (c a, (τ − d)/c) in packed form, ψ' = relabel(ψ, cT, dT, cP, dP); central differences."""
    def tr(x):
        cT, dT, cP, dP = 1.0, 0.0, 1.0, 0.0
        if s == "T":
            cT, dT = (1 + x, 0.0) if par == "c" else (1.0, x)
        else:
            cP, dP = (1 + x, 0.0) if par == "c" else (1.0, x)
        itx = M.Items(it.aT * cT, (it.tauT - dT) / cT, it.aP * cP, (it.tauP - dP) / cP)
        return np.concatenate(pack_set(itx)), relabel(psi, cT, dT, cP, dP)
    (pf, qf), (pb, qb) = tr(e), tr(-e)
    return (pf - pb) / (2 * e), (qf - qb) / (2 * e)


def settled(psi_new, psi_old, se):
    return bool(np.max(np.abs(np.asarray(psi_new) - np.asarray(psi_old)) / np.asarray(se)) < SETTLE_Z)


def aitken(x0, x1, x2):
    """Componentwise Aitken Δ² extrapolation of a sequence (descriptive only)."""
    x0, x1, x2 = (np.asarray(v, float) for v in (x0, x1, x2))
    den = x2 - 2 * x1 + x0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = x2 - (x2 - x1) ** 2 / den
    return np.where(np.abs(den) > 1e-12, out, x2)


# ---------------------------------------------------------------------------
#  Worker state and tasks (fork-inherited)
# ---------------------------------------------------------------------------
_W: dict = {}


def raw_chunks(groups):
    out = {}
    for n, g in groups.items():
        for s0 in range(0, len(g.patients), CHUNK):
            r = slice(s0, min(s0 + CHUNK, len(g.patients)))
            out[(n, s0)] = M.Group(g.T[r], g.X[r], g.YT[r], g.YP[r], g.patients[r])
    return out


def _s2_task(args):
    """Stage-2 NLL, gradient (and per-patient NLL gradients) for one chunk, reported context:
    non-ND patients by AGQ G = 5 with the exact gradient (doc 25 P2(b), doc 27)."""
    cfg, key, psi, items, b0, per_patient = args
    g = _W["chunks"][key]
    it = M.Items(*items)
    d = MQ.prepare(g, it)
    info: dict = {}
    try:
        val, b, nd, sH, gr = MQ.group_laplace(psi, d, it, Q, b0, True, info=info)
    except np.linalg.LinAlgError as e:
        return cfg, key, {"failure": f"linalg {e}"}
    if info.get("failed"):
        return cfg, key, {"failure": "mode_budget"}
    n_fb = 0
    if np.any(~nd):
        r = np.flatnonzero(~nd)
        finfo: dict = {}
        fv, _, _, fg = MQ.group_agq(psi, d.rows(r), it, Q, G_REPORTED, b[r], True, info=finfo, gradient="exact")
        if finfo.get("failed"):
            return cfg, key, {"failure": "fallback mode_budget"}
        val, gr = val.copy(), gr.copy()
        val[r], gr[r] = fv, fg
        n_fb = len(r)
    return cfg, key, {"nll": -float(val.sum()), "grad": -gr.sum(0), "pp": (-gr if per_patient else None), "b": b,
                      "n_fallback": n_fb, "finite": bool(np.all(np.isfinite(val)) and np.all(np.isfinite(gr)))}


def eval_s2(pool, configs, b0map=None, per_patient=False):
    """configs: list of (psi, items-tuple). Returns per-config dict (nll, grad, pp, b-map, fallback)."""
    keys = list(_W["chunks"])
    args = [(c, k, np.asarray(psi, float), items, None if b0map is None else b0map[k], per_patient)
            for c, (psi, items) in enumerate(configs) for k in keys]
    res = pool.map(_s2_task, args, chunksize=1) if pool is not None else map(_s2_task, args)
    out = [{"nll": 0.0, "grad": np.zeros(M.N_PSI), "pp": [], "b": {}, "n_fallback": 0, "failures": {}, "finite": True}
           for _ in configs]
    tmp = {}
    for c, k, r in res:
        tmp[(c, k)] = r
    for c in range(len(configs)):
        o = out[c]
        for k in keys:
            r = tmp[(c, k)]
            if "failure" in r:
                o["failures"][str(k)] = r["failure"]
                continue
            o["nll"] += r["nll"]
            o["grad"] = o["grad"] + r["grad"]
            o["b"][k] = r["b"]
            o["n_fallback"] += r["n_fallback"]
            o["finite"] &= r["finite"]
            if per_patient:
                o["pp"].append(r["pp"])
        if per_patient and not o["failures"]:
            o["pp"] = np.concatenate(o["pp"])
    return out


def _item_grad_task(args):
    """Gradient of an item objective: kind 'n01' (stage 1, θ ~ N(0,1)) or 'lat' (θ_v ~ N(M, K) at ψ)."""
    tag, kind, s, phi, psi = args
    Y = _W["Y"][s]
    nodes = G.gh_nodes()
    if kind == "n01":
        f, g = G.stage1_negloglik(phi, Y, nodes)
    else:
        m, sd = _W["S1A"].visit_latent(_W["groups"], psi)[s]
        f, g = _W["S1A"].latent_negloglik(phi, Y, m, sd, nodes)
    return tag, f, g


# ---------------------------------------------------------------------------
#  Setup shared by all modes
# ---------------------------------------------------------------------------
def setup(a, mode):
    diag = Path(a.diag)
    out = diag / (a.outname or (f"{mode}_smoke" if a.smoke else mode))
    out.mkdir(parents=True, exist_ok=False)
    base = json.loads((diag / "00_baseline.json").read_text())
    ro = json.loads((diag / "readonly/summary.json").read_text())
    I: dict = {}
    locks = sorted((ROOT / "docs/cluster_phase/_frozen").glob("*.sha256"))
    I["frozen_locks"] = all(sha(ROOT / ln.split()[1]) == ln.split()[0] for f in locks
                            for ln in f.read_text().splitlines() if ln.strip())
    code = ["src/ppmi_subtype/m2_quad.py", "src/ppmi_subtype/m2_grm.py", "src/ppmi_subtype/grm.py",
            "scripts/45_m2_pilot.py", "scripts/51_m2a1_s1a_true_latent_items.py", "scripts/52_m2a1_full_pipeline_recovery.py"]
    I["code_unchanged_vs_baseline"] = all(sha(ROOT / f) == base["code_sha256"][f] for f in code)
    I["script53_equals_readonly_run"] = sha(ROOT / "scripts/53_m2_step6_diag_readonly.py") == ro["provenance"]["script_sha256"]
    inputs = [p for p in base["prior_outputs_sha256"] if p.startswith(("experiments/20261018T215102Z", "experiments/20261018T222857Z"))]
    I["step5_step6_outputs_unchanged"] = all(sha(ROOT / p) == base["prior_outputs_sha256"][p] for p in inputs)
    I["panel_md5"] = sha(ROOT / "outputs/visit_panel.csv", "md5") == base["panel_md5"]
    prov = {"label": LABEL, "mode": mode, "smoke": a.smoke, "driver_sha256": sha(Path(__file__)),
            "tests_sha256": sha(ROOT / "tests/test_step6_diagnostics.py"), "integrity": I,
            "seeds": {r: [42, 45, 1, r] for r in range(N_DATA)}, "Q": Q, "G_reported": G_REPORTED,
            "h_item": H_ITEM, "h_psi": H_PSI, "settle_z": SETTLE_Z, "k_max": K_MAX,
            "host": platform.node(), "jobs": a.jobs, "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "numpy": np.__version__, "python": sys.version.split()[0],
            "cpu": next((ln.split(":", 1)[1].strip() for ln in subprocess.run(
                ["lscpu"], capture_output=True, text=True).stdout.splitlines() if ln.startswith("Model name")), "?")}
    if not all(v for v in I.values()):
        jdump(out / "STOPPED.json", {"reason": "integrity check failed", **prov})
        raise SystemExit(f"integrity failed: {I}")
    from ppmi_subtype.config import load_config
    P = load_module("m2_pilot", ROOT / "scripts/45_m2_pilot.py")
    S1A = load_module("s1a", ROOT / "scripts/51_m2a1_s1a_true_latent_items.py")
    S6M = load_module("s6", ROOT / "scripts/52_m2a1_full_pipeline_recovery.py")
    R53 = load_module("r53", ROOT / "scripts/53_m2_step6_diag_readonly.py")
    v, times, treat, YTd, YPd, *_ = P.load_cohort(load_config())
    g0 = M.make_groups(times, treat, YTd, YPd)
    psi_true, it_true, _ = P.truth_on_stage1_scale(g0)
    s6 = json.loads((S6 / "summary.json").read_text())
    I["truth_equals_step6_record"] = bool(np.array_equal(psi_true, np.array(s6["psi_true"])))

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

    rs = [0] if a.smoke else list(range(N_DATA))
    data = {r: dataset(r) for r in rs}
    if a.smoke:
        data = {r: subset(g, 60) for r, g in data.items()}
    else:
        ref = json.loads(IDENT.read_text())
        nll = MQ.negloglik(np.array(ref["psi"]), data[0], it_true, 10, want_grad=False)
        I["dataset0_identity"] = {"nll": nll, "equal": nll == ref["nll"] == IDENT_NLL}
        # exact per-dataset alignment against Step 5's recorded item objectives (N(0,1) and oracle)
        al = {}
        nodes = G.gh_nodes()
        for r in rs:
            rec = json.loads((S5 / f"S1_items_dataset{r}.json").read_text())
            Yr = P.flatten(data[r])
            vl = S1A.visit_latent(data[r], psi_true)
            for i, s in enumerate(SUB):
                Y = Yr[i].astype(float)
                fn = rec["fits"][f"N01_{s}"]
                fo = rec["fits"][f"oracle_{s}"]
                v1, _ = G.stage1_negloglik(G.pack_items(np.array(fn["a"]), np.array(fn["tau"])), Y, nodes)
                v2, _ = S1A.latent_negloglik(G.pack_items(np.array(fo["a"]), np.array(fo["tau"])), Y, *vl[s], nodes)
                al[f"{r}{s}"] = {"N01_equal": v1 == rec["item_objectives"][f"N01_{s}"]["N01_nll"],
                                 "oracle_equal": v2 == rec["item_objectives"][f"oracle_{s}"]["oracle_nll"]}
        I["dataset_alignment_vs_step5_records"] = al
        I["dataset_alignment_exact"] = all(x["N01_equal"] and x["oracle_equal"] for x in al.values())
        if not (I["dataset0_identity"]["equal"] and I["dataset_alignment_exact"] and I["truth_equals_step6_record"]):
            jdump(out / "STOPPED.json", {"reason": "dataset identity/alignment failed", **prov})
            raise SystemExit("dataset identity/alignment failed")
    log(f"{mode}: integrity ok {I if a.smoke else {k: v for k, v in I.items() if k != 'dataset_alignment_vs_step5_records'}}")
    ctx = dict(out=out, prov=prov, P=P, S1A=S1A, S6M=S6M, R53=R53, data=data, psi_true=psi_true, it_true=it_true,
               names=M.NAMES, s6=s6, t0=time.time())
    return ctx


def saved_step6(r):
    d = json.loads((S6 / f"S1_dataset{r}.json").read_text())
    it1 = M.Items(*(np.array(d["items_1"][s][k]) for s in SUB for k in ("a", "tau")))
    it2 = M.Items(*(np.array(d["items_2"][s][k]) for s in SUB for k in ("a", "tau")))
    return d, it1, it2


def finish(ctx, summary):
    out = ctx["out"]
    summary.update(label=LABEL, provenance=ctx["prov"], runtime_s=time.time() - ctx["t0"])
    jdump(out / "summary.json", summary)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    (out / "MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files))
    log(f"done in {summary['runtime_s']:.0f}s")
    return 0


def fit_and_hessian(ctx, sg, it, jobs, maxiter):
    t = time.perf_counter()
    f = MQ.fit(sg, it, Q=Q, psi0=M.default_psi(), jobs=jobs, maxiter=maxiter)
    t_fit = time.perf_counter() - t
    o = MQ.Objective(sg, it, Q, jobs=jobs)
    o.state = dict(f.state)
    t = time.perf_counter()
    try:
        H = MQ.fd_hessian(o, f.psi)
    finally:
        o.close()
    t_h = time.perf_counter() - t
    rep = MQ.evaluate_reported(f.psi, sg, it, Q, G_REPORTED, init_state=f.state)
    rec = ctx["S6M"].fit_record(f, t_fit, maxiter)
    rec.update(hessian_wall_s=t_h, hessian_penalised_fd_evals=sum(1 for e in o.log if e["penalty"]), reported=rep)
    return f, H, rec


# ---------------------------------------------------------------------------
#  D6
# ---------------------------------------------------------------------------
def run_d6(a):
    ctx = setup(a, "d6")
    names, truth = ctx["names"], ctx["psi_true"]
    maxiter = 3 if a.smoke else 1000
    S: dict = {"diagnostic": "D6: stage 2 with Step 5's saved true-latent (oracle) items", "datasets": []}
    covered = []
    for r, sg in ctx["data"].items():
        t_ds = time.time()
        try:
            rec5 = json.loads((S5 / f"S1_items_dataset{r}.json").read_text())
            it_or = M.Items(*(np.array(rec5["fits"][f"oracle_{s}"][k]) for s in SUB for k in ("a", "tau")))
            f, H, rec = fit_and_hessian(ctx, sg, it_or, a.jobs, maxiter)
            cov = ctx["S6M"].coverage(f.psi, truth, H)
            d = {"dataset": r, "fit": rec, "coverage": cov, "n_covered": int(cov["covered"].sum()), "H": H,
                 "z": cov["z"], "wall_s": time.time() - t_ds}
            jdump(ctx["out"] / f"D6_dataset{r}.json", d)
            S["datasets"].append({k: v for k, v in d.items() if k != "H"})
            covered.append(cov["covered"])
            log(f"D6 dataset {r}: covered {d['n_covered']}/19 (PD {cov['hessian_pd']}), conv {f.converged}, "
                f"fit {rec['wall_s']:.0f}s, fallback {rec['reported']['n_fallback_patients']}; "
                + " ".join(f"{p} z {cov['z'][names.index(p)]:+.2f}" for p in FAILED))
        except Exception as e:
            S["datasets"].append({"dataset": r, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]})
            S["stopped"] = f"dataset {r} execution error"
            log(S["stopped"])
            break
    if covered:
        S["diagnostic_coverage_counts"] = dict(zip(names, np.sum(covered, axis=0).tolist()))
        S["note"] = ("diagnostic counts under the Step 4/6 interval construction with counterfactual oracle items; "
                     "NOT a recovery test and NOT a re-scoring of Step 6")
    return finish(ctx, S)


# ---------------------------------------------------------------------------
#  D4
# ---------------------------------------------------------------------------
def run_d4(a):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    ctx = setup(a, "d4")
    names, truth, S1A, P, R53 = ctx["names"], ctx["psi_true"], ctx["S1A"], ctx["P"], ctx["R53"]
    it_true = ctx["it_true"]
    S: dict = {"diagnostic": "D4: item uncertainty and item–stage-2 coupling (stacked sandwich over the S1-b chain)",
               "datasets": []}
    KT, KP = len(it_true.aT), len(it_true.aP)
    nT, nP = 5 * KT, 5 * KP
    nphi = nT + nP
    phi_true = np.concatenate(pack_set(it_true))
    for r, sg in ctx["data"].items():
        t_ds = time.time()
        try:
            d6, it1, it2 = saved_step6(r)
            psi1, psi2 = np.array(d6["stage2_fit_1"]["psi"]), np.array(d6["stage2_fit_2"]["psi"])
            H1, H2 = np.array(d6["H1_nonbinding"]), np.array(d6["H2"])
            if a.smoke:                                           # code path only: Hessians are not those of the subset
                H1 = H2 = np.eye(M.N_PSI) * 100.0
            phi1, phi2 = np.concatenate(pack_set(it1)), np.concatenate(pack_set(it2))
            YT, YP, pid = ctx["P"].flatten(sg)
            _W.clear()
            _W.update(chunks=raw_chunks(sg), Y={"T": YT.astype(float), "P": YP.astype(float)}, groups=sg, S1A=S1A)
            pool = ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("fork")) if a.jobs > 1 else None
            try:
                tup = lambda it: (it.aT, it.tauT, it.aP, it.tauP)  # noqa: E731
                # base evaluations (cold modes), per-patient scores
                base = eval_s2(pool, [(psi1, tup(it1)), (psi2, tup(it2))], None, per_patient=True)
                b1, b2 = base[0]["b"], base[1]["b"]
                chk = {"nll_psi1_vs_saved": base[0]["nll"] - d6["stage2_fit_1"]["nll"],
                       "nll_psi2_vs_saved": base[1]["nll"] - d6["stage2_fit_2"]["nll"],
                       "grad_norm_psi1": float(np.linalg.norm(base[0]["grad"])),
                       "grad_norm_psi2": float(np.linalg.norm(base[1]["grad"])),
                       "fallback_base": [base[0]["n_fallback"], base[1]["n_fallback"]],
                       "failures_base": [base[0]["failures"], base[1]["failures"]],
                       "pp_sum_minus_total_psi2": float(np.max(np.abs(base[1]["pp"].sum(0) - base[1]["grad"])))}
                # C1, C2: central FD of the stage-2 gradient over the 80 item parameters (warm modes)
                def cross(psi, phi, bmap):
                    cfgs = []
                    for k in range(nphi):
                        for sgn in (+1, -1):
                            p = phi.copy()
                            p[k] += sgn * H_ITEM
                            cfgs.append((psi, tup(items_from_phi(p[:nT], p[nT:], KT, KP))))
                    res = eval_s2(pool, cfgs, bmap)
                    fails = sum(1 for x in res if x["failures"]) + sum(1 for x in res if not x["finite"])
                    fb = sum(x["n_fallback"] for x in res)
                    C = np.column_stack([(res[2 * k]["grad"] - res[2 * k + 1]["grad"]) / (2 * H_ITEM) for k in range(nphi)])
                    return C, fails, fb
                t = time.perf_counter()
                C1, f1, fb1 = cross(psi1, phi1, b1)
                C2, f2, fb2 = cross(psi2, phi2, b2)
                t_cross = time.perf_counter() - t
                # item-objective Jacobians: A1 (N(0,1) at φ1), A3 (latent at ψ1, φ2), E3 = ∂g3/∂ψ1
                t = time.perf_counter()
                tasks = []
                for s, (lo, n) in (("T", (0, nT)), ("P", (nT, nP))):
                    for k in range(n):
                        for sgn in (+1, -1):
                            p1 = phi1[lo:lo + n].copy()
                            p1[k] += sgn * H_ITEM
                            tasks.append((("A1", s, k, sgn), "n01", s, p1, None))
                            p2 = phi2[lo:lo + n].copy()
                            p2[k] += sgn * H_ITEM
                            tasks.append((("A3", s, k, sgn), "lat", s, p2, psi1))
                    for j in range(M.N_PSI):
                        for sgn in (+1, -1):
                            q = psi1.copy()
                            q[j] += sgn * H_PSI
                            tasks.append((("E3", s, j, sgn), "lat", s, phi2[lo:lo + n], q))
                res = {tag: g for tag, f, g in (pool.map(_item_grad_task, tasks, chunksize=4) if pool else
                                                map(_item_grad_task, tasks))}
                A1, A3, E3 = np.zeros((nphi, nphi)), np.zeros((nphi, nphi)), np.zeros((nphi, M.N_PSI))
                for s, (lo, n) in (("T", (0, nT)), ("P", (nT, nP))):
                    for k in range(n):
                        A1[lo:lo + n, lo + k] = (res[("A1", s, k, 1)] - res[("A1", s, k, -1)]) / (2 * H_ITEM)
                        A3[lo:lo + n, lo + k] = (res[("A3", s, k, 1)] - res[("A3", s, k, -1)]) / (2 * H_ITEM)
                    for j in range(M.N_PSI):
                        E3[lo:lo + n, j] = (res[("E3", s, j, 1)] - res[("E3", s, j, -1)]) / (2 * H_PSI)
                t_items = time.perf_counter() - t
            finally:
                if pool is not None:
                    pool.shutdown()
            # per-patient item scores
            vl1 = S1A.visit_latent(sg, psi1)
            nodes = G.gh_nodes()
            sl = patient_slices(pid)
            U1, U3 = np.zeros((len(sl), nphi)), np.zeros((len(sl), nphi))
            for i, q in enumerate(sl):
                for s, Y, lo, n in (("T", YT, 0, nT), ("P", YP, nT, nP)):
                    Yq = Y[q].astype(float)
                    U1[i, lo:lo + n] = G.stage1_negloglik(phi1[lo:lo + n], Yq, nodes)[1]
                    U3[i, lo:lo + n] = S1A.latent_negloglik(phi2[lo:lo + n], Yq, vl1[s][0][q], vl1[s][1][q], nodes)[1]
            U2a, U4 = base[0]["pp"], base[1]["pp"]
            # stacked system θ = (φ1, ψ1, φ2, ψ2)
            p = M.N_PSI
            Jm = np.zeros((2 * nphi + 2 * p, 2 * nphi + 2 * p))
            i1, i2, i3, i4 = slice(0, nphi), slice(nphi, nphi + p), slice(nphi + p, 2 * nphi + p), slice(2 * nphi + p, None)
            Jm[i1, i1] = A1
            Jm[i2, i1], Jm[i2, i2] = C1, H1
            Jm[i3, i2], Jm[i3, i3] = E3, A3
            Jm[i4, i3], Jm[i4, i4] = C2, H2
            U = np.hstack([U1, U2a, U3, U4])
            V, B = stacked_sandwich(Jm, U)
            # sub-systems
            V34, _ = stacked_sandwich(np.block([[A3, np.zeros((nphi, p))], [C2, H2]]), np.hstack([U3, U4]))
            V12, _ = stacked_sandwich(np.block([[A1, np.zeros((nphi, p))], [C1, H1]]), np.hstack([U1, U2a]))
            H2i, H1i = np.linalg.inv(H2), np.linalg.inv(H1)
            se = {"frozen_model_H2": np.sqrt(np.diag(H2i)),
                  "stage2_sandwich_only": np.sqrt(np.diag(H2i @ (U4.T @ U4) @ H2i)),
                  "two_step_S1b_items_at_fixed_psi1": np.sqrt(np.diag(V34[nphi:, nphi:])),
                  "full_chain": np.sqrt(np.diag(V[i4, i4]))}
            se1 = {"frozen_model_H1": np.sqrt(np.diag(H1i)), "two_step_N01_items": np.sqrt(np.diag(V12[nphi:, nphi:]))}
            # numerical checks: affine-invariance identity C v_φ + H v_ψ ≈ 0 at (ψ̂2, φ2), score additivity, symmetry
            inv = {}
            for s in SUB:
                for par in ("c", "d"):
                    vphi, vpsi = affine_tangent(it2, psi2, s, par, R53.relabel_psi)
                    lhs = C2 @ vphi + H2 @ vpsi
                    inv[f"{s}_{par}"] = {"rel_residual": float(np.linalg.norm(lhs) / max(np.linalg.norm(H2 @ vpsi), 1e-300)),
                                         "abs_residual": float(np.linalg.norm(lhs))}
            jac_chk = {"A1_asym": float(np.max(np.abs(A1 - A1.T))), "A3_asym": float(np.max(np.abs(A3 - A3.T))),
                       "H2_asym": float(np.max(np.abs(H2 - H2.T))), "item_score_sum_N01_max": float(np.max(np.abs(U1.sum(0)))),
                       "item_score_sum_S1b_max": float(np.max(np.abs(U3.sum(0)))),
                       "B44_diag_over_H2_diag": np.diag(U4.T @ U4) / np.diag(H2), "affine_invariance": inv,
                       "C_fd_failures": f1 + f2, "C_fd_fallback_patients": fb1 + fb2}
            # first-order attribution at (ψ̂2, φ2): J2 = −H2⁻¹ C2 applied to item errors (T vs P; affine vs residual)
            J2 = -H2i @ C2
            dphi = phi2 - phi_true
            al = {s: R53.affine_fit(*(getattr(it2, f"a{s}"), getattr(it2, f"tau{s}")),
                                    getattr(it_true, f"a{s}"), getattr(it_true, f"tau{s}")) for s in SUB}
            it_aff = M.Items(it_true.aT * al["T"]["c_joint"], (it_true.tauT - al["T"]["d_joint"]) / al["T"]["c_joint"],
                             it_true.aP * al["P"]["c_joint"], (it_true.tauP - al["P"]["d_joint"]) / al["P"]["c_joint"])
            dphi_aff = np.concatenate(pack_set(it_aff)) - phi_true
            mT = np.r_[np.ones(nT), np.zeros(nP)].astype(bool)
            s4 = json.loads((ROOT / f"experiments/20261018T184713Z_stage2_step4_fixed_item_recovery/step4_run2/S1_dataset{r}.json").read_text())
            se4 = np.array(s4["se"])
            attr = {"obs_psi2_minus_step4_z4": (psi2 - np.array(s4["psi_hat"])) / se4,
                    "pred_total_z4": (J2 @ dphi) / se4, "pred_T_items_z4": (J2[:, mT] @ dphi[mT]) / se4,
                    "pred_P_items_z4": (J2[:, ~mT] @ dphi[~mT]) / se4, "pred_affine_part_z4": (J2 @ dphi_aff) / se4,
                    "pred_nonaffine_part_z4": (J2 @ (dphi - dphi_aff)) / se4}
            z = {k: (psi2 - truth) / v for k, v in se.items()}
            rec = {"dataset": r, "checks": chk, "jacobian_checks": jac_chk, "se_psi2": se, "se_psi1": se1,
                   "se_ratio_to_frozen": {k: v / se["frozen_model_H2"] for k, v in se.items()},
                   "z_psi2_under_each_se": z,
                   "diagnostic_cover_counts_under_each_se_NOT_A_RESCORING": {k: int(np.sum(np.abs(v) <= 1.96)) for k, v in z.items()},
                   "attribution_first_order": attr, "corr_psi2_full_chain": V[i4, i4] / np.outer(se["full_chain"], se["full_chain"]),
                   "sensitivity_dpsi2_dphi2_J2": J2, "timings_s": {"cross_fd": t_cross, "item_fd": t_items},
                   "wall_s": time.time() - t_ds}
            jdump(ctx["out"] / f"D4_dataset{r}.json", {**rec, "C1": C1, "C2": C2, "A1": A1, "A3": A3, "E3": E3,
                                                        "V_full": V, "B_full": B})
            S["datasets"].append({k: v for k, v in rec.items() if k not in ("sensitivity_dpsi2_dphi2_J2", "corr_psi2_full_chain")})
            log(f"D4 dataset {r}: SE ratio full-chain/frozen " + " ".join(
                f"{p} {rec['se_ratio_to_frozen']['full_chain'][names.index(p)]:.2f}" for p in FAILED)
                + f"; affine-invariance rel residual max {max(x['rel_residual'] for x in inv.values()):.1e}; "
                f"{rec['wall_s']:.0f}s")
        except Exception as e:
            S["datasets"].append({"dataset": r, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]})
            S["stopped"] = f"dataset {r} execution error"
            log(S["stopped"])
            break
    return finish(ctx, S)


# ---------------------------------------------------------------------------
#  D5
# ---------------------------------------------------------------------------
def run_d5(a):
    ctx = setup(a, "d5")
    names, truth, S1A, P, R53 = ctx["names"], ctx["psi_true"], ctx["S1A"], ctx["P"], ctx["R53"]
    it_true = ctx["it_true"]
    maxiter_s2, maxiter_it = (3, 20) if a.smoke else (1000, 3000)
    k_max = 3 if a.smoke else K_MAX
    S: dict = {"diagnostic": "D5: further S1-b steps (the Step 6 step (3)-(4) map) from the saved Step 6 state",
               "settle_rule": f"max_j |ψ̂_k − ψ̂_(k−1)|_j / SE(2)_j < {SETTLE_Z}", "k_max": k_max, "datasets": {}}
    state = {}
    for r, sg in ctx["data"].items():
        d6, it1, it2 = saved_step6(r)
        se2 = np.array(d6["coverage_2"]["se"])
        state[r] = {"psi": np.array(d6["stage2_fit_2"]["psi"]), "items": it2, "se2": se2, "active": True,
                    "hist": [{"k": 1, "psi": np.array(d6["stage2_fit_1"]["psi"]), "nll": d6["stage2_fit_1"]["nll"]},
                             {"k": 2, "psi": np.array(d6["stage2_fit_2"]["psi"]), "nll": d6["stage2_fit_2"]["nll"]}],
                    "f_last": None}
        Y = P.flatten(sg)
        S1A._FIT.setdefault("Y", {}).update({(r, s): Y[i].astype(float) for i, s in enumerate(SUB)})
        S1A._FIT["P"] = P
        S1A._FIT.setdefault("lat", {})

    def zmeta(psi, se):
        z = (psi - truth) / se
        return {"z": z, "rms_z_all": float(np.sqrt(np.mean(z ** 2))),
                **{f"z_{p}": float(z[names.index(p)]) for p in FAILED}}

    stopped = None
    for k in range(3, k_max + 1):
        act = [r for r in state if state[r]["active"]]
        if not act:
            break
        try:
            t = time.time()
            for r in act:
                vl = S1A.visit_latent(ctx["data"][r], state[r]["psi"])
                for s in SUB:
                    S1A._FIT["lat"][(r, s)] = vl[s]
                state[r]["anchor_moments"] = {s: {"mean": float(vl[s][0].mean()),
                                                  "var": float((vl[s][1] ** 2).mean() + vl[s][0].var())} for s in SUB}
            fits = {(r, s): f for _, r, s, f in S1A.run_pool([("oracle", r, s, maxiter_it) for r in act for s in SUB], a.jobs)}
            t_items = time.time() - t
            for r in act:
                sg = ctx["data"][r]
                itk = M.Items(fits[(r, "T")]["a"], fits[(r, "T")]["tau"], fits[(r, "P")]["a"], fits[(r, "P")]["tau"])
                t = time.perf_counter()
                f = MQ.fit(sg, itk, Q=Q, psi0=M.default_psi(), jobs=a.jobs, maxiter=maxiter_s2)
                wall = time.perf_counter() - t
                fr = ctx["S6M"].fit_record(f, wall, maxiter_s2)
                aff = {s: {kk: vv for kk, vv in R53.affine_fit(getattr(itk, f"a{s}"), getattr(itk, f"tau{s}"),
                                                              getattr(it_true, f"a{s}"), getattr(it_true, f"tau{s}")).items()
                           if not kk.startswith("resid_")} for s in SUB}
                prev = state[r]["psi"]
                h = {"k": k, "psi": f.psi, "nll": f.nll, "fit": fr, "items": {s: {"a": getattr(itk, f"a{s}"),
                                                                                    "tau": getattr(itk, f"tau{s}"),
                                                                                    "nll": fits[(r, s)]["nll"],
                                                                                    "converged": fits[(r, s)]["converged"],
                                                                                    "n_iter": fits[(r, s)]["n_iter"]}
                                                                                for s in SUB},
                     "affine": aff, "anchor_moments_at_previous_psi": state[r]["anchor_moments"],
                     "step_max_abs_z": float(np.max(np.abs(f.psi - prev) / state[r]["se2"])),
                     **zmeta(f.psi, state[r]["se2"]), "item_refit_wall_s_all_active": t_items}
                state[r]["hist"].append(h)
                state[r]["psi"], state[r]["items"], state[r]["f_last"] = f.psi, itk, f
                if settled(f.psi, prev, state[r]["se2"]):
                    state[r]["active"] = False
                    state[r]["settled_at"] = k
                log(f"D5 k={k} dataset {r}: step max|Δ|/SE {h['step_max_abs_z']:.3f}; " + " ".join(
                    f"{p} z {h['z_' + p]:+.2f}" for p in FAILED) + f"; c_P {aff['P']['c_joint']:.3f}; "
                    f"conv {f.converged}; fit {wall:.0f}s" + ("; SETTLED" if not state[r]["active"] else ""))
        except Exception as e:
            stopped = f"iteration {k} execution error: {type(e).__name__}: {e}"
            S["trace"] = traceback.format_exc()[-3000:]
            log(stopped)
            break
    S["stopped"] = stopped
    for r, st in state.items():
        out_r = {"dataset": r, "settled_at": st.get("settled_at"), "history": st["hist"]}
        if st["f_last"] is not None and stopped is None:
            f = st["f_last"]
            o = MQ.Objective(ctx["data"][r], st["items"], Q, jobs=a.jobs)
            o.state = dict(f.state)
            try:
                H = MQ.fd_hessian(o, f.psi)
            finally:
                o.close()
            cov = ctx["S6M"].coverage(f.psi, truth, H)
            out_r.update(final_coverage_diagnostic=cov, final_n_covered_diagnostic=int(cov["covered"].sum()),
                         final_reported=MQ.evaluate_reported(f.psi, ctx["data"][r], st["items"], Q, G_REPORTED,
                                                             init_state=f.state), H_final=H)
        ps = [h["psi"] for h in st["hist"]]
        if len(ps) >= 3:
            out_r["aitken_extrapolated_psi_descriptive"] = aitken(*ps[-3:])
            out_r["aitken_z_descriptive"] = (out_r["aitken_extrapolated_psi_descriptive"] - truth) / st["se2"]
        jdump(ctx["out"] / f"D5_dataset{r}.json", out_r)
        S["datasets"][r] = {k: v for k, v in out_r.items() if k not in ("history", "H_final")}
        S["datasets"][r]["trajectory"] = [{kk: h.get(kk) for kk in ("k", "nll", "step_max_abs_z", "rms_z_all",
                                                                     *(f"z_{p}" for p in FAILED))} for h in st["hist"]]
    if stopped is None and all("final_coverage_diagnostic" in v for v in S["datasets"].values()):
        S["diagnostic_cover_counts_final_NOT_A_RECOVERY_RESULT"] = dict(zip(names, np.sum(
            [S["datasets"][r]["final_coverage_diagnostic"]["covered"] for r in state], axis=0).tolist()))
    return finish(ctx, S)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("d6", "d4", "d5"))
    ap.add_argument("--diag", required=True)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--smoke", action="store_true", help="code path only (1 dataset, 60 patients); NOT A RESULT")
    ap.add_argument("--outname", default=None, help="output sub-directory name (default: mode or mode_smoke)")
    a = ap.parse_args()
    return {"d6": run_d6, "d4": run_d4, "d5": run_d5}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
