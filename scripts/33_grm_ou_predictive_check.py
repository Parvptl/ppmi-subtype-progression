"""Stage 6b, Amendment 3: GRM + OU latent persistence — predictive check only.

Implements docs/cluster_phase/06_AMENDMENT3_PERSISTENCE_SPEC.md (SHA-256
recorded). Reuses script 32's item loader, feature function and gated-target
list unchanged, so the two checks differ only in the latent model. No
clustering metric is computed.

    python scripts/33_grm_ou_predictive_check.py --out <dir> --amendment2-run <dir>
        [--reps 40] [--jobs 40] [--agq-patients 30]
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ppmi_subtype import grm as G  # noqa: E402
from ppmi_subtype import grm_ou as O  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

S26 = importlib.import_module("26_clusterability_extended")
C32 = importlib.import_module("32_grm_predictive_check")

SEED = 42
SPEC = ROOT / "docs" / "cluster_phase" / "06_AMENDMENT3_PERSISTENCE_SPEC.md"
_S: dict = {}


def _task(rep):
    data, it, lf = _S["data"], _S["items_fit"], _S["ou_fit"]
    rng = np.random.default_rng([SEED, 33, rep])
    times = {p: data.times[p] for p in data.patients}
    sim_items = O.simulate(rng, times, lf["beta"], lf["D"], lf["sigma_u"], lf["ell"],
                           it["a"], it["tau"])
    resp = {p: np.column_stack([data.responses[p][:, 0], sim_items[p].mean(axis=1)])
            for p in data.patients}
    row = C32.features(S26.repack(times, resp), sim_items)
    row["rep"] = rep
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--amendment2-run", required=True)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=40)
    ap.add_argument("--agq-patients", type=int, default=30)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a2 = Path(args.amendment2_run)
    cfg = load_config()
    md5 = S26.md5(cfg.path("outputs") / "visit_panel.csv")
    if md5 != S26.EXPECTED_PANEL_MD5:
        raise SystemExit("panel MD5 differs from the frozen panel")
    names = list(cfg.subtype.pigd_items)
    C32._S["item_names"] = names
    data = S26.load_data(cfg)
    items, _ = C32.load_items(cfg, data, names)
    meta = {"argv": sys.argv, "seed": SEED, "reps": args.reps, "panel_md5": md5,
            "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
            "amendment2_run": str(a2),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip(),
            "git_dirty": subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(), "numpy": np.__version__,
            "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    (out / "run_config.json").write_text(json.dumps(meta, indent=2))

    # stage 1: identical refit; must equal the Amendment 2 item parameters
    Yall = np.concatenate([items[p] for p in data.patients])
    it = G.fit_items(Yall, names)
    prev = json.loads((a2 / "item_fit.json").read_text())
    same = bool(np.allclose(it.a, prev["a"], atol=1e-8) and np.allclose(it.tau, prev["tau"], atol=1e-8))
    (out / "item_fit.json").write_text(json.dumps(
        {"a": it.a.tolist(), "tau": it.tau.tolist(), "nll": it.nll, "converged": it.converged,
         "identical_to_amendment2": same}, indent=2))
    if not same:
        raise SystemExit("stage-1 item parameters differ from the Amendment 2 run")

    # stage 2: GRM + OU, start at the Amendment 2 estimates with sigma_u = 0.3, ell = 2
    l2 = json.loads((a2 / "longitudinal_fit.json").read_text())
    psi0 = O.to_psi(np.array(l2["beta"]), np.array(l2["D"]), 0.3, 2.0)
    grouped = O.group({p: data.times[p] for p in data.patients}, items)
    t0 = time.time()
    ou = O.fit(grouped, it.a, it.tau, psi0)
    ou_json = {"beta": ou.beta.tolist(), "D": ou.D.tolist(), "sigma_u": ou.sigma_u,
               "ell": ou.ell, "nll": ou.nll, "converged": ou.converged, "message": ou.message,
               "n_iter": ou.n_iter, "grad_norm": ou.grad_norm, "at_bound": ou.at_bound,
               "seconds": time.time() - t0,
               "sd_intercept": float(np.sqrt(ou.D[0, 0])), "sd_slope": float(np.sqrt(ou.D[1, 1])),
               "amendment2_laplace_nll": l2["nll"],
               "laplace_loglik_gain_vs_amendment2": l2["nll"] - ou.nll}
    (out / "ou_fit.json").write_text(json.dumps(ou_json, indent=2))
    print("OU fit:", json.dumps(ou_json, indent=1), flush=True)

    psi = O.to_psi(ou.beta, ou.D, ou.sigma_u, ou.ell)
    rng = np.random.default_rng(SEED)
    small = [p for p in data.patients if 2 <= len(data.times[p]) <= 3]
    agq = []
    for p in rng.choice(small, min(args.agq_patients, len(small)), replace=False):
        lap = O.laplace_one(data.times[p], items[p], psi, it.a, it.tau)
        q = O.agq_one(data.times[p], items[p], psi, it.a, it.tau, nodes=9)
        agq.append({"PATNO": int(p), "n_visits": len(data.times[p]),
                    "all_items_zero": bool(np.all(items[p] == 0)), "laplace": lap, "agq": q,
                    "abs_diff": abs(lap - q)})
    agq = pd.DataFrame(agq)
    agq.to_csv(out / "laplace_vs_agq.csv", index=False)

    real = C32.features(data, items)
    (out / "real_features.json").write_text(json.dumps(real, indent=2))
    state = {"data": data, "item_names": names, "items_fit": {"a": it.a, "tau": it.tau},
             "ou_fit": {"beta": ou.beta, "D": ou.D, "sigma_u": ou.sigma_u, "ell": ou.ell}}

    def init(st):
        _S.update(st)
        C32._S["item_names"] = st["item_names"]
    with ProcessPoolExecutor(args.jobs, initializer=init, initargs=(state,)) as pool:
        sims = pd.DataFrame(list(pool.map(_task, range(args.reps), chunksize=1)))
    sims.to_csv(out / "simulated_features.csv", index=False)

    gated = list(C32.GATED) + [(f"item_pos_{n}", 0.03) for n in names]
    rows = []
    for name, tol in gated:
        y = sims[name].to_numpy(float)
        mean, sd = float(np.nanmean(y)), float(np.nanstd(y, ddof=1))
        se = sd / np.sqrt(np.isfinite(y).sum())
        rv = real[name]
        if not np.isfinite(rv) or not np.isfinite(mean):
            status = "fail (undefined)"
        elif se > tol / 2:
            status = "undecided (MC SE > tol/2)"
        else:
            status = "pass" if abs(mean - rv) <= tol else "fail"
        rows.append({"target": name, "real": rv, "sim_mean": mean, "sim_sd": sd, "mc_se": se,
                     "sim_q05": float(np.nanquantile(y, 0.05)),
                     "sim_q95": float(np.nanquantile(y, 0.95)),
                     "error": mean - rv, "tolerance": tol, "status": status})
    table = pd.DataFrame(rows)
    table.to_csv(out / "calibration_table.csv", index=False)
    rep_cols = [c for c in real if c not in dict(gated)]
    pd.DataFrame([{"feature": c, "real": real[c], "sim_mean": sims[c].mean(),
                   "sim_sd": sims[c].std(ddof=1)} for c in rep_cols]).to_csv(
        out / "reported_features.csv", index=False)
    n_pass = int((table.status == "pass").sum())
    decision = {"n_gated": len(table), "n_pass": n_pass, "GO": n_pass == len(table),
                "failing": table.loc[table.status != "pass", "target"].tolist(),
                "ou_converged": ou.converged, "ou_at_bound": ou.at_bound,
                "laplace_agq_max_abs_diff": float(agq.abs_diff.max()),
                "laplace_agq_max_abs_diff_non_floor": float(
                    agq.loc[~agq.all_items_zero, "abs_diff"].max())}
    (out / "decision.json").write_text(json.dumps(decision, indent=2))
    print(table.round(4).to_string(index=False))
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
