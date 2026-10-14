"""Stage 6b: skewed / heavy-tailed no-groups null for the clusterability audit.

Design, targets and decision rule are fixed in
docs/cluster_phase/06_SKEWED_NULL_PREREG.md (its SHA-256 is recorded in the
run config). Reuses script 26's evaluation unchanged (imported, not copied):
same BLUP-slope features, GMM BIC, silhouette and perturbation stability.

Stages (each skipped if its outputs exist; pass --resume to continue a run):

calibrate  indirect inference for (delta_intT, delta_intP, delta_slopeT,
           delta_slopeP, nu) on BLUP skewness/kurtosis only, then the
           pre-registered fidelity gate. Writes calibration.json,
           calibration_trace.csv; a failed gate writes FIDELITY_FAILED and stops.
simulate   R replicates of null_skewt and null_tails, plus the real data
           (which must reproduce script 26's real.csv). Writes simulations.csv,
           real.csv, fidelity.csv.
decide     pre-registered rule with Monte Carlo uncertainty -> decision.csv,
           summary.csv.

    python scripts/29_clusterability_skewed_null.py --out <run_dir>
        [--stages calibrate,simulate,decide] [--reps 200] [--cal-reps 24]
        [--cal-maxfev 200] [--jobs 60] [--smoke]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import beta as beta_dist, kurtosis, skew  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ppmi_subtype import nongaussian as NG  # noqa: E402
from ppmi_subtype import timevarying as TV  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

S26 = importlib.import_module("26_clusterability_extended")

SEED = 42
PREREG = ROOT / "docs" / "cluster_phase" / "06_SKEWED_NULL_PREREG.md"
COMPONENTS = ("int_tremor", "int_pigd", "slope_tremor", "slope_pigd")
SKEW_TOL, KURT_TOL = 0.30, 1.5                  # fidelity gate (pre-registered)
KURT_WEIGHT = 0.25
SIMPLEX_STEP = 1.0                              # initial Nelder-Mead step, transformed scale
STREAM = {"cal": 7, "null_skewt": 8, "null_tails": 9}
_S: dict = {}


# ---------------------------------------------------------------------------
#  Simulation and shape summaries
# ---------------------------------------------------------------------------
def simulate(data, fit, rng, deltas, nu):
    """null_measured with non-Gaussian random effects; everything else identical."""
    B = NG.random_effects(rng, fit.sigma_b, deltas, nu, len(data.patients))
    times, resp = {}, {}
    for i, p in enumerate(data.patients):
        t = data.times[p]
        e = rng.multivariate_normal(np.zeros(2), fit.sigma_e, size=len(t)).reshape(-1)
        y = (TV.design(t, ()) @ fit.beta + TV.random_design(t, True) @ B[i] + e).reshape(-1, 2)
        y = np.clip(y, 0.0, 4.0)
        y = np.column_stack([np.round(y[:, 0] / S26.GRID[0]) * S26.GRID[0],
                             np.round(y[:, 1] / S26.GRID[1]) * S26.GRID[1]])
        times[p], resp[p] = t, y
    return S26.repack(times, resp)


def shape(data, fit=None) -> dict:
    """Skewness / excess kurtosis of the four BLUP components, and floor shares."""
    fit = fit or TV.fit(data, knots=(), random_slopes=True)
    blup = np.array([TV.blup(fit, data.times[p], data.responses[p])[0] for p in data.patients])
    Y = np.concatenate([data.responses[p] for p in data.patients])
    out = {}
    for j, c in enumerate(COMPONENTS):
        out[f"skew_{c}"] = float(skew(blup[:, j]))
        out[f"kurt_{c}"] = float(kurtosis(blup[:, j]))
    out["share_pigd_zero_visits"] = float(np.mean(Y[:, 1] == 0))
    out["share_tremor_zero_visits"] = float(np.mean(Y[:, 0] == 0))
    out["share_patients_all_pigd_zero"] = float(np.mean(
        [np.all(data.responses[p][:, 1] == 0) for p in data.patients]))
    out["converged"] = bool(fit.converged)
    return out


def _cal_task(args):
    rep, deltas, nu = args
    rng = np.random.default_rng([SEED, STREAM["cal"], rep])     # common random numbers
    return shape(simulate(_S["data"], _S["fit"], rng, deltas, nu))


def unpack(x):
    """(delta_j, nu_j) for the four components; amendment 1 of the pre-registration."""
    return np.tanh(x[:4]), NG.NU_MIN + np.exp(x[4:8])


def objective_parts(mean_shape: dict, target: dict) -> float:
    v = 0.0
    for c in COMPONENTS:
        v += (mean_shape[f"skew_{c}"] - target[f"skew_{c}"]) ** 2
        v += KURT_WEIGHT * (mean_shape[f"kurt_{c}"] - target[f"kurt_{c}"]) ** 2
    return v


def stage_calibrate(run, data, fit, real_shape, args, pool):
    trace = []

    def f(x):
        deltas, nu = unpack(x)
        shapes = list(pool.map(_cal_task, [(r, deltas, nu) for r in range(args.cal_reps)]))
        m = pd.DataFrame(shapes).mean(numeric_only=True).to_dict()
        val = objective_parts(m, real_shape)
        trace.append({"eval": len(trace), **{f"delta_{c}": d for c, d in zip(COMPONENTS, deltas)},
                      **{f"nu_{c}": v for c, v in zip(COMPONENTS, nu)},
                      "objective": val, **{f"sim_{k}": v for k, v in m.items()}})
        pd.DataFrame(trace).to_csv(run / "calibration_trace.csv", index=False)
        print(f"  cal {len(trace):3d}  obj {val:8.4f}  deltas {np.round(deltas, 3)}  "
              f"nu {np.round(nu, 2)}", flush=True)
        return val

    # Neutral start (delta = 0.5, nu = 8 for every component). scipy's default
    # initial simplex uses 5% steps (~0.03 in atanh delta), which on the first
    # real run crawled near the start for 29 evaluations; an explicit simplex
    # with unit steps in every transformed coordinate lets the search reach the
    # whole parameter range. The start is not taken from the feasibility probe.
    x0 = np.r_[np.arctanh(np.full(4, 0.5)), np.full(4, np.log(8.0 - NG.NU_MIN))]
    simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * SIMPLEX_STEP
                                for i in range(len(x0))])
    res = minimize(f, x0, method="Nelder-Mead",
                   options={"maxfev": args.cal_maxfev, "xatol": 1e-3, "fatol": 1e-4,
                            "initial_simplex": simplex})
    deltas, nu = unpack(res.x)
    shapes = pd.DataFrame(list(pool.map(_cal_task, [(r, deltas, nu)
                                                   for r in range(args.cal_reps)])))
    rows = []
    for c in COMPONENTS:
        for kind, tol in (("skew", SKEW_TOL), ("kurt", KURT_TOL)):
            k = f"{kind}_{c}"
            sim = float(shapes[k].mean())
            mcse = float(shapes[k].std(ddof=1) / np.sqrt(len(shapes)))
            gated = kind == "skew" or c.startswith("slope")
            rows.append({"target": k, "real": real_shape[k], "simulated_mean": sim,
                         "mc_se": mcse, "abs_diff": abs(sim - real_shape[k]),
                         "tolerance": tol if gated else None,
                         "within": (abs(sim - real_shape[k]) <= tol) if gated else None})
    for k in ("share_pigd_zero_visits", "share_tremor_zero_visits",
              "share_patients_all_pigd_zero"):
        rows.append({"target": k, "real": real_shape[k],
                     "simulated_mean": float(shapes[k].mean()),
                     "mc_se": float(shapes[k].std(ddof=1) / np.sqrt(len(shapes))),
                     "abs_diff": abs(float(shapes[k].mean()) - real_shape[k]),
                     "tolerance": None, "within": None})
    gate = pd.DataFrame(rows)
    gate.to_csv(run / "calibration_fidelity.csv", index=False)
    ok = bool(gate.within.dropna().astype(bool).all())
    cal = {"deltas": dict(zip(COMPONENTS, map(float, deltas))),
           "nu": dict(zip(COMPONENTS, map(float, nu))),
           "objective": float(res.fun), "n_evals": int(res.nfev),
           "optimizer_message": str(res.message), "fidelity_gate_passed": ok}
    (run / "calibration.json").write_text(json.dumps(cal, indent=2))
    print(gate.round(3).to_string(index=False), flush=True)
    if not ok:
        (run / "FIDELITY_FAILED").write_text("calibration did not meet the pre-registered "
                                             "tolerances; see calibration_fidelity.csv\n")
    return cal


def _sim_task(args):
    kind, rep, deltas, nu = args
    rng = np.random.default_rng([SEED, STREAM[kind], rep])
    sim = simulate(_S["data"], _S["fit"], rng, deltas, nu)
    row = S26.evaluate(sim, SEED + rep)
    row.update({k: v for k, v in shape(sim).items() if k != "converged"})
    row.update({"dataset": kind, "rep": rep})
    return row


def stage_simulate(run, data, fit, cal, args, pool):
    real = S26.evaluate(data, SEED)
    real.update({k: v for k, v in shape(data, fit).items() if k != "converged"})
    real.update({"dataset": "real", "rep": 0})
    pd.DataFrame([real]).to_csv(run / "real.csv", index=False)
    deltas = np.array([cal["deltas"][c] for c in COMPONENTS])
    nus = np.array([cal["nu"][c] for c in COMPONENTS])
    tasks = ([("null_skewt", r, deltas, nus) for r in range(args.reps)]
             + [("null_tails", r, np.zeros(4), nus) for r in range(args.reps)])
    rows, t0 = [], time.time()
    for i, row in enumerate(pool.map(_sim_task, tasks, chunksize=1), 1):
        rows.append(row)
        if i % 50 == 0 or i == len(tasks):
            pd.DataFrame(rows).to_csv(run / "simulations.csv", index=False)
            print(f"  {i}/{len(tasks)} simulations, {(time.time() - t0) / 60:.1f} min", flush=True)


def mc_summary(x: float, y: np.ndarray, rng) -> dict:
    y = np.asarray(y, float)
    R = len(y)
    k = int((y >= x).sum())
    q95 = float(np.quantile(y, 0.95))
    boot = np.quantile(y[rng.integers(0, R, (4000, R))], 0.95, axis=1)
    lo_b, hi_b = np.percentile(boot, [2.5, 97.5])
    cp_lo = float(beta_dist.ppf(0.025, k, R - k + 1)) if k > 0 else 0.0
    cp_hi = float(beta_dist.ppf(0.975, k + 1, R - k)) if k < R else 1.0
    return {"real": x, "null_median": float(np.median(y)), "null_q95": q95,
            "null_q95_mc_lo": float(lo_b), "null_q95_mc_hi": float(hi_b),
            "n_null": R, "n_null_ge_real": k, "p_mc": (1 + k) / (R + 1),
            "exceedance_share_cp_lo": cp_lo, "exceedance_share_cp_hi": cp_hi,
            "borderline": bool(lo_b <= x <= hi_b)}


def stage_decide(run):
    real = pd.read_csv(run / "real.csv").iloc[0]
    sims = pd.read_csv(run / "simulations.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for null in ("null_skewt", "null_tails"):
        g = sims[sims.dataset == null]
        a = mc_summary(real.dBIC, g.dBIC.to_numpy(), rng)
        b = mc_summary(real.drop1_ARI_K2, g.drop1_ARI_K2.to_numpy(), rng)
        A = a["real"] > a["null_q95"]
        B = b["real"] > b["null_q95"] and b["real"] >= 0.6
        verdict = {(False, False): "consistent with continuous heterogeneity; no evidence "
                                   "for discrete groups",
                   (True, False): "inconclusive: separation exceeds this continuum but "
                                  "assignments are not reproducible",
                   (True, True): "structure beyond this continuum, reproducible; groups "
                                 "possible but not established phenotypes",
                   (False, True): "inconclusive: stability without excess separation"}[(A, B)]
        for crit, s, met in (("A_dBIC_above_q95", a, A), ("B_drop1_ARI_K2_above_q95_and_ge_0.6",
                                                           b, B)):
            rows.append({"null": null, "criterion": crit, **s, "met": bool(met)})
        rows.append({"null": null, "criterion": "CONCLUSION", "verdict": verdict,
                     "primary": null == "null_skewt",
                     "any_borderline": bool(a["borderline"] or b["borderline"])})
    pd.DataFrame(rows).to_csv(run / "decision.csv", index=False)
    cols = ["dBIC", "best_K", "silhouette_K2", "drop1_ARI_K2", "drop2_ARI_K2", "trunc3_ARI_K2",
            "drop1_spearman_pigd", "skew_slope_pigd", "kurt_slope_pigd", "skew_slope_tremor",
            "kurt_slope_tremor", "share_pigd_zero_visits", "share_patients_all_pigd_zero"]
    summ = []
    for ds, g in pd.concat([sims, pd.DataFrame([real])]).groupby("dataset"):
        r = {"dataset": ds, "n": len(g)}
        for c in cols:
            if c in g:
                r[f"{c}_median"] = float(g[c].median())
                r[f"{c}_q05"] = float(g[c].quantile(0.05))
                r[f"{c}_q95"] = float(g[c].quantile(0.95))
        summ.append(r)
    pd.DataFrame(summ).to_csv(run / "summary.csv", index=False)
    print(pd.read_csv(run / "decision.csv").to_string(index=False))


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="calibrate,simulate,decide")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--cal-reps", type=int, default=24)
    ap.add_argument("--cal-maxfev", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force-past-gate", action="store_true",
                    help="run the null even if the fidelity gate failed (not for reporting)")
    args = ap.parse_args()

    cfg = load_config()
    run = Path(args.out)
    run.mkdir(parents=True, exist_ok=True)
    panel_md5 = S26.md5(cfg.path("outputs") / "visit_panel.csv")
    if panel_md5 != S26.EXPECTED_PANEL_MD5 and not args.smoke:
        raise SystemExit(f"panel MD5 {panel_md5} != frozen {S26.EXPECTED_PANEL_MD5}")
    data = S26.load_data(cfg)
    fit = TV.fit(data, knots=(), random_slopes=True)
    real_shape = shape(data, fit)
    import scipy
    import sklearn
    meta = {"argv": sys.argv, "seed": SEED, "panel_md5": panel_md5, "smoke_test": args.smoke,
            "n_patients": data.n_patients, "reps": args.reps, "cal_reps": args.cal_reps,
            "cal_maxfev": args.cal_maxfev, "skew_tol": SKEW_TOL, "kurt_tol": KURT_TOL,
            "kurt_weight": KURT_WEIGHT, "streams": STREAM, "simplex_step": SIMPLEX_STEP,
            "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
            "real_shape": real_shape,
            "sigma_b": fit.sigma_b.tolist(), "sigma_e": fit.sigma_e.tolist(),
            "beta": fit.beta.tolist(), "fit_converged": fit.converged,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip(),
            "git_dirty": subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    cfg_path = run / "run_config.json"
    if cfg_path.exists():
        with open(run / "resume_log.jsonl", "a") as fh:
            fh.write(json.dumps(meta, default=str) + "\n")
    else:
        cfg_path.write_text(json.dumps(meta, indent=2, default=str))

    stages = args.stages.split(",")
    with ProcessPoolExecutor(args.jobs, initializer=_S.update,
                             initargs=({"data": data, "fit": fit},)) as pool:
        if "calibrate" in stages:
            if (run / "calibration.json").exists():
                print("calibration exists; reusing it", flush=True)
            else:
                stage_calibrate(run, data, fit, real_shape, args, pool)
        cal = json.loads((run / "calibration.json").read_text())
        if not cal["fidelity_gate_passed"] and not args.force_past_gate:
            print("FIDELITY GATE FAILED: stopping before the null simulations.", flush=True)
            return 2
        if "simulate" in stages:
            if (run / "simulations.csv").exists() and len(
                    pd.read_csv(run / "simulations.csv")) == 2 * args.reps:
                print("simulations exist; reusing them", flush=True)
            else:
                stage_simulate(run, data, fit, cal, args, pool)
    if "decide" in stages:
        stage_decide(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
