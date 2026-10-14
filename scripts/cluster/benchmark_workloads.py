"""Benchmark the project's representative workloads before choosing how to run them.

Runs on a SYNTHETIC cohort with the real cohort's shape (934 patients, the
audited visit-count distribution, roughly annual visits, subscore magnitudes
from the Step 1 fit). No PPMI data is read, so this can be run before the data
is on the cluster and its output can be shared freely.

What it measures
----------------
1. Single-thread wall time of one fit for each workload that dominates an
   experiment: single-output GP (K1, K3), multi-output GP (K0/K1/K2/K3, full and
   diag), joint LMM, the 3-state HMM with emission floor, the 5-state HMM.
2. Throughput when P fits run concurrently on one node (P = 1, 16, 32, 60), to
   find where physical cores / memory bandwidth saturate.
3. The multi-output GP objective on GPU (PyTorch, float64) against the audited
   NumPy objective: agreement of value and gradient first, then time per
   evaluation and time per full L-BFGS-B fit.

    python scripts/cluster/benchmark_workloads.py --out <dir> [--skip-gpu]
           [--skip-scaling] [--quick]

Writes <dir>/single_fit_times.csv, scaling.csv, gpu_vs_cpu.csv, env.json.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

from ppmi_subtype import gpr, hmm, jointlmm, mogp as M  # noqa: E402

SEED = 42


# ---------------------------------------------------------------------------
#  Synthetic cohort with the real cohort's shape
# ---------------------------------------------------------------------------
#: Visit-count distribution from outputs/clustering_audit/data_audit.csv as
#: quoted in the GPR-onwards report (n=2 18.4%, >=3 81.6%, >=4 53.7%, >=5 35.7%,
#: >=8 15.5%, >=10 7.8%), spread uniformly inside each band.
_COUNT_BANDS = [((2, 2), 0.184), ((3, 3), 0.279), ((4, 4), 0.180),
                ((5, 7), 0.202), ((8, 9), 0.077), ((10, 14), 0.078)]


def synthetic_cohort(n_patients: int = 934, seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    probs = np.array([p for _, p in _COUNT_BANDS])
    probs = probs / probs.sum()
    comps = [("const", np.array([[0.10, 0.0], [0.0, 0.08]]), None),
             ("rbf", np.array([[0.03, 0.0], [0.0, 0.03]]), 3.7)]
    sigma_e = np.array([[0.06, 0.0], [0.0, 0.04]])
    beta = np.array([0.55, 0.023, 0.25, 0.066])
    seqs = []
    for i in range(n_patients):
        (lo, hi), = [_COUNT_BANDS[rng.choice(len(_COUNT_BANDS), p=probs)][0]]
        n = int(rng.integers(lo, hi + 1))
        gaps = rng.gamma(shape=8.0, scale=1.0 / 8.0, size=n - 1)
        t = np.concatenate([[0.0], np.cumsum(gaps)])
        K = M.build_K(t, comps, sigma_e)
        L = np.linalg.cholesky(K + 1e-10 * np.eye(2 * n))
        y = (M._mean_design(t) @ beta + L @ rng.standard_normal(2 * n)).reshape(n, 2)
        y = np.clip(y, 0.0, 4.0)
        labels = np.where(y[:, 1] <= 0, 0,
                          np.digitize(y[:, 0] / np.maximum(y[:, 1], 1e-9),
                                      [0.90, 1.15]))
        labels = 2 - labels                           # 0 TD, 1 Indet, 2 PIGD
        seqs.append({"PATNO": i, "t": t, "y": y, "labels": labels.astype(int),
                     "gaps": np.diff(t)})
    return seqs


# ---------------------------------------------------------------------------
#  One-fit workloads
# ---------------------------------------------------------------------------
def _fit(name: str, seqs: list[dict]) -> dict:
    t0 = time.perf_counter()
    if name.startswith("gp1_"):
        kernel = name.split("_")[1]
        f = gpr.fit_gpr([{"t": s["t"], "y": s["y"][:, 1]} for s in seqs],
                        kernel=kernel, n_restarts=1, seed=SEED)
        conv = f.converged
    elif name.startswith("mogp_"):
        _, kernel, coregion = name.split("_")
        f = M.fit_mogp(M.pack([{"t": s["t"], "y": s["y"]} for s in seqs]),
                       kernel=kernel, coregion=coregion, n_restarts=1, seed=SEED)
        conv = f.converged
    elif name == "joint_lmm":
        rows = [{"PATNO": s["PATNO"], "years_from_baseline": tt,
                 "tremor_score": yy[0], "pigd_score": yy[1]}
                for s in seqs for tt, yy in zip(s["t"], s["y"])]
        f = jointlmm.fit_joint_lmm(jointlmm.pack(pd.DataFrame(rows)))
        conv = f.converged
    elif name.startswith("hmm"):
        n_hidden = 5 if name == "hmm5" else 3
        f = hmm.fit_hmm([{"PATNO": s["PATNO"], "labels": s["labels"],
                          "gaps": s["gaps"]} for s in seqs],
                        states=[0, 1, 2], seed=SEED, n_restarts=1,
                        emission_floor=None if n_hidden == 5 else 0.5,
                        n_hidden=n_hidden)
        conv = f.converged
    else:
        raise ValueError(name)
    return {"workload": name, "seconds": time.perf_counter() - t0,
            "converged": bool(conv)}


def _fit_worker(args):
    name, n_patients = args
    return _fit(name, synthetic_cohort(n_patients))


WORKLOADS = ["gp1_K1", "gp1_K3", "mogp_K0_full", "mogp_K1_full", "mogp_K1_diag",
             "mogp_K2_full", "mogp_K3_full", "mogp_K3_diag", "joint_lmm",
             "hmm3", "hmm5"]


# ---------------------------------------------------------------------------
#  GPU comparison
# ---------------------------------------------------------------------------
def gpu_comparison(seqs, kernels, device_list) -> list[dict]:
    import torch
    from torch_mogp import TorchMOGP

    packed = M.pack([{"t": s["t"], "y": s["y"]} for s in seqs])
    rows = []
    rng = np.random.default_rng(SEED)
    for kernel in kernels:
        for coregion in ("full", "diag"):
            theta0 = M.initial_theta(kernel, coregion, packed)
            probe = theta0 + rng.normal(scale=0.2, size=theta0.shape)
            v_np, g_np = M.objective(probe, kernel, coregion, packed, 1e-8)
            # NumPy reference timing
            t0 = time.perf_counter()
            for _ in range(5):
                M.objective(probe, kernel, coregion, packed, 1e-8)
            t_np = (time.perf_counter() - t0) / 5
            rows.append({"kernel": kernel, "coregion": coregion,
                         "backend": "numpy_1thread", "sec_per_eval": t_np,
                         "abs_value_diff": 0.0, "max_rel_grad_diff": 0.0})
            for dev in device_list:
                if dev == "cuda" and not torch.cuda.is_available():
                    continue
                tm = TorchMOGP(packed, kernel, coregion, device=dev)
                v_t, g_t = tm.objective(probe)        # warm-up + agreement
                rel = np.max(np.abs(g_t - g_np) / np.maximum(np.abs(g_np), 1e-6))
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    tm.objective(probe)
                if dev == "cuda":
                    torch.cuda.synchronize()
                rows.append({"kernel": kernel, "coregion": coregion,
                             "backend": f"torch_{dev}",
                             "sec_per_eval": (time.perf_counter() - t0) / 20,
                             "abs_value_diff": abs(v_t - v_np),
                             "max_rel_grad_diff": float(rel)})
            print(pd.DataFrame(rows[-3:]).to_string(index=False), flush=True)
    return rows


def full_fit_gpu_vs_cpu(seqs, kernels) -> list[dict]:
    """End-to-end single-restart fit, NumPy objective vs torch-on-GPU objective."""
    import torch
    from torch_mogp import TorchMOGP

    if not torch.cuda.is_available():
        return []
    packed = M.pack([{"t": s["t"], "y": s["y"]} for s in seqs])
    rows = []
    for kernel in kernels:
        coregion = "full"
        layout = M.parameter_layout(kernel, coregion)
        bounds = [M._LOG_BOUNDS if r == "log" else (-20.0, 20.0) for _, r in layout]
        theta0 = np.clip(M.initial_theta(kernel, coregion, packed),
                         [b[0] for b in bounds], [b[1] for b in bounds])
        t0 = time.perf_counter()
        r_np = minimize(M.objective, theta0, args=(kernel, coregion, packed, 1e-8),
                        jac=True, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 300})
        t_np = time.perf_counter() - t0
        tm = TorchMOGP(packed, kernel, coregion, device="cuda")
        t0 = time.perf_counter()
        r_gpu = minimize(tm.objective, theta0, jac=True, method="L-BFGS-B",
                         bounds=bounds, options={"maxiter": 300})
        t_gpu = time.perf_counter() - t0
        rows.append({"kernel": kernel, "numpy_sec": t_np, "numpy_nfev": r_np.nfev,
                     "numpy_nll": r_np.fun, "gpu_sec": t_gpu, "gpu_nfev": r_gpu.nfev,
                     "gpu_nll": r_gpu.fun, "nll_diff": r_gpu.fun - r_np.fun,
                     "max_abs_theta_diff": float(np.max(np.abs(r_gpu.x - r_np.x)))})
        print(rows[-1], flush=True)
    return rows


# ---------------------------------------------------------------------------
def environment() -> dict:
    env = {"host": socket.gethostname(), "python": sys.version.split()[0],
           "platform": platform.platform(), "numpy": np.__version__,
           "cpu_count": os.cpu_count(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "slurm_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE")}
    try:
        import scipy
        env["scipy"] = scipy.__version__
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpus"] = [torch.cuda.get_device_name(i)
                           for i in range(torch.cuda.device_count())]
    except ImportError:
        env["torch"] = None
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--skip-scaling", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="200 synthetic patients instead of 934 (smoke test)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)          # never overwrite a run
    n_pat = 200 if args.quick else 934

    env = environment()
    (out / "env.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=2), flush=True)

    seqs = synthetic_cohort(n_pat)
    print(f"synthetic cohort: {len(seqs)} patients, "
          f"{sum(len(s['t']) for s in seqs)} visits, "
          f"max {max(len(s['t']) for s in seqs)} visits", flush=True)

    # 1. single-thread single-fit times
    rows = []
    for w in WORKLOADS:
        r = _fit(w, seqs)
        print(f"  {w:14s} {r['seconds']:8.1f}s  converged={r['converged']}", flush=True)
        rows.append(r)
        pd.DataFrame(rows).to_csv(out / "single_fit_times.csv", index=False)

    # 2. concurrency scaling on one representative workload
    if not args.skip_scaling:
        srows = []
        ctx = mp.get_context("spawn")
        for wl in ("mogp_K1_full", "hmm3"):
            for P in (1, 16, 32, 60):
                t0 = time.perf_counter()
                with ctx.Pool(P) as pool:
                    res = pool.map(_fit_worker, [(wl, n_pat)] * P)
                wall = time.perf_counter() - t0
                per = float(np.mean([r["seconds"] for r in res]))
                srows.append({"workload": wl, "processes": P, "wall_sec": wall,
                              "mean_fit_sec": per, "fits_per_hour": 3600 * P / wall})
                print(f"  scaling {wl} P={P:3d}: wall {wall:7.1f}s  "
                      f"mean fit {per:7.1f}s  {3600 * P / wall:8.1f} fits/h", flush=True)
                pd.DataFrame(srows).to_csv(out / "scaling.csv", index=False)

    # 3. GPU
    if not args.skip_gpu:
        g = gpu_comparison(seqs, ["K1", "K3"], ["cpu", "cuda"])
        pd.DataFrame(g).to_csv(out / "gpu_vs_cpu.csv", index=False)
        f = full_fit_gpu_vs_cpu(seqs, ["K1", "K3"])
        pd.DataFrame(f).to_csv(out / "gpu_full_fit.csv", index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
