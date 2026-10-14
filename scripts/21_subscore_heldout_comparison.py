"""Stage 3: persistence vs LMM vs independent GP vs joint GP, on identical folds.

Completes the comparison Step 2 (script 18) left partial. Every arm below is
refitted inside the **same** leave-one-patient-out folds over the same 934
patients and scored on the **same** 3,361 one-step-ahead points, so every
paired difference is exactly paired:

    persistence, patient_mean         fold-estimated Gaussian error model
    lmm_ri                            joint random-intercept LMM (Step 1 model)
    gp_indep_K0..K3                   two single-output GPs (`gpr.fit_gpr`)
    gp_joint_K0..K3_diag              joint GP, cross-output block pinned to zero
    gp_joint_K0..K3_full              joint GP, cross-output block free

`full` vs `diag` differ only in the cross-output parameters (the nested test of
coregionalization). `diag` vs `indep` differ in whether the two outputs share a
lengthscale. K2/K3 were population-only in Step 2 because of runtime; here they
are evaluated held out.

What every arm stores per point: predictive mean (2,), full predictive
covariance (2x2), and the fold's fit diagnostics. Scoring is a separate stage
(`22_analyse_subscore_heldout.py`) that reads only these saved files.

Baseline uncertainty
--------------------
Script 18 gave persistence an ad hoc SD (history SD, or 0.3 with one visit).
Here both baselines get a *fold-estimated* Gaussian error model: the 2x2
covariance of that baseline's own one-step errors over the training patients
of the fold. Point predictions are unchanged from script 18; only the
probabilistic scores differ, and they now use no held-out information and no
arbitrary constant.

Execution
---------
The unit of work is one (arm, held-out patient) task, so the slow K2/K3 fits
load-balance across workers. Each task writes `folds/<arm>/<PATNO>.csv` (and a
`_fit.json`) atomically. A rerun with `--resume <run_dir>` skips finished tasks,
so a killed job loses at most the tasks in flight.

    python scripts/21_subscore_heldout_comparison.py --out <run_dir>
        [--arms all|name,name,...] [--jobs 60] [--fold-restarts 1]
        [--limit N] [--resume]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import gpr, jointlmm as J, mogp as M, trajectory as tj  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

SEED = 42
JITTER = 1e-8
EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"

ALL_ARMS = (
    ["persistence", "patient_mean", "lmm_ri"]
    + [f"gp_indep_{k}" for k in ("K0", "K1", "K2", "K3")]
    + [f"gp_joint_{k}_{c}" for k in ("K0", "K1", "K2", "K3") for c in ("diag", "full")]
)

_STATE: dict = {}


# ---------------------------------------------------------------------------
#  Cohort (identical to scripts 17 and 18)
# ---------------------------------------------------------------------------
def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_sequences(cfg):
    """934 patients, every labelled visit, both subscores; as script 18."""
    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    traj = tj.build_trajectories(panel, cfg)
    tj.check_ordering(traj)
    frame = traj.merge(
        panel[["PATNO", "EVENT_ID", "tremor_score", "pigd_score"]]
        .drop_duplicates(["PATNO", "EVENT_ID"]),
        on=["PATNO", "EVENT_ID"], how="left", validate="one_to_one")
    if frame[["tremor_score", "pigd_score"]].isna().any().any():
        raise SystemExit("a visit is missing a subscore after the merge")
    seqs = {}
    for patno, g in frame.groupby("PATNO", sort=True):
        g = g.sort_values("years_from_baseline", kind="mergesort")
        seqs[int(patno)] = {
            "t": g["years_from_baseline"].to_numpy(float),
            "y": g[["tremor_score", "pigd_score"]].to_numpy(float),
            "subtype": g["subtype_name"].to_numpy(),
            "event": g["EVENT_ID"].to_numpy(),
        }
    return frame, seqs


# ---------------------------------------------------------------------------
#  Arms
# ---------------------------------------------------------------------------
def _baseline_point(name, Y_hist):
    return Y_hist[-1].astype(float) if name == "persistence" else Y_hist.mean(axis=0)


def _baseline_error_cov(name, train):
    """Covariance of the baseline's own one-step errors over training patients."""
    errs = []
    for s in train:
        for k in range(1, len(s["t"])):
            errs.append(s["y"][k] - _baseline_point(name, s["y"][:k]))
    return np.cov(np.asarray(errs).T, ddof=1)


def fit_arm(arm, train, restarts, max_iter):
    """Fit one arm on the training patients. Returns (predict_fn, diagnostics)."""
    t0 = time.perf_counter()
    if arm in ("persistence", "patient_mean"):
        C = _baseline_error_cov(arm, train)

        def pred(t_hist, Y_hist, t_star, C=C):
            return _baseline_point(arm, Y_hist), C
        diag = {"converged": True, "cov_tremor": C[0, 0], "cov_pigd": C[1, 1],
                "cov_cross": C[0, 1]}

    elif arm == "lmm_ri":
        rows = [{"PATNO": i, "years_from_baseline": tt, "tremor_score": yy[0],
                 "pigd_score": yy[1]}
                for i, s in enumerate(train) for tt, yy in zip(s["t"], s["y"])]
        fit = J.fit_joint_lmm(J.pack(pd.DataFrame(rows)), reml=True)

        def pred(t_hist, Y_hist, t_star, fit=fit):
            b_hat, post = J.blup(fit, t_hist, Y_hist)
            mean = np.array([fit.beta[0] + fit.beta[1] * t_star,
                             fit.beta[2] + fit.beta[3] * t_star]) + b_hat
            return mean, post + fit.sigma_e
        diag = {"converged": bool(fit.converged), "objective": float(fit.neg2ll),
                **{f"beta_{i}": float(b) for i, b in enumerate(fit.beta)},
                "var_b_tremor": fit.sigma_b[0, 0], "var_b_pigd": fit.sigma_b[1, 1],
                "cov_b": fit.sigma_b[0, 1], "var_e_tremor": fit.sigma_e[0, 0],
                "var_e_pigd": fit.sigma_e[1, 1], "cov_e": fit.sigma_e[0, 1]}

    elif arm.startswith("gp_indep_"):
        kernel = arm.split("_")[-1]
        fits = [gpr.fit_gpr([{"t": s["t"], "y": s["y"][:, d]} for s in train],
                            kernel=kernel, n_restarts=restarts, max_iter=max_iter,
                            jitter=JITTER, seed=SEED) for d in range(2)]

        def pred(t_hist, Y_hist, t_star, fits=fits):
            mean, cov = np.empty(2), np.zeros((2, 2))
            for d in range(2):
                m, sd = gpr.predict_one(fits[d], t_hist, Y_hist[:, d], t_star,
                                        jitter=JITTER)
                mean[d], cov[d, d] = m, sd ** 2
            return mean, cov
        diag = {"converged": bool(fits[0].converged and fits[1].converged),
                "objective": -(fits[0].log_marginal_likelihood
                               + fits[1].log_marginal_likelihood)}
        for d, name in enumerate(M.OUTPUTS):
            f = fits[d]
            diag.update({f"{name}_converged": f.converged, f"{name}_message": f.message,
                         f"{name}_n_iter": f.n_iter, f"{name}_n_fev": f.n_fev,
                         f"{name}_at_bound": "|".join(f.at_boundary),
                         f"{name}_beta0": float(f.beta[0]), f"{name}_beta1": float(f.beta[1]),
                         **{f"{name}_{k}": float(v) for k, v in
                            zip(gpr.parameter_names(f.kernel), f.theta)}})

    elif arm.startswith("gp_joint_"):
        _, _, kernel, coregion = arm.split("_")
        packed = M.pack([{"t": s["t"], "y": s["y"]} for s in train])
        fit = M.fit_mogp(packed, kernel=kernel, coregion=coregion,
                         n_restarts=restarts, max_iter=max_iter, jitter=JITTER,
                         seed=SEED)

        def pred(t_hist, Y_hist, t_star, fit=fit):
            return M.predict_one(fit, t_hist, Y_hist, t_star, jitter=JITTER)
        diag = {"converged": bool(fit.converged), "objective": float(fit.nll),
                "message": fit.message, "n_iter": fit.n_iter, "n_fev": fit.n_fev,
                "grad_norm": fit.grad_norm, "at_bound": "|".join(fit.at_bound),
                "restart_nlls": json.dumps(fit.restart_nlls),
                **fit.parameter_dict(),
                **{f"corr_{k}": v for k, v in fit.coregion_correlation().items()}}
    else:
        raise ValueError(f"unknown arm {arm!r}")
    diag["fit_seconds"] = time.perf_counter() - t0
    return pred, diag


# ---------------------------------------------------------------------------
#  One task = one (arm, held-out patient)
# ---------------------------------------------------------------------------
def _init(seqs, run_dir, restarts, max_iter):
    _STATE.update(seqs=seqs, run_dir=Path(run_dir), restarts=restarts,
                  max_iter=max_iter)


def _atomic_write_csv(df, path: Path):
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run_task(arm, patno):
    seqs, run_dir = _STATE["seqs"], _STATE["run_dir"]
    out_csv = run_dir / "folds" / arm / f"{patno}.csv"
    if out_csv.exists():
        return arm, patno, "skipped", 0.0
    t0 = time.perf_counter()
    train = [s for p, s in seqs.items() if p != patno]
    # Leakage guard: the held-out patient's arrays must not be in `train`.
    held = seqs[patno]
    if any(s is held for s in train) or len(train) != len(seqs) - 1:
        raise RuntimeError(f"held-out patient {patno} leaked into training")

    pred, diag = fit_arm(arm, train, _STATE["restarts"], _STATE["max_iter"])
    t_all, Y_all = held["t"], held["y"]
    rows = []
    for k in range(1, len(t_all)):
        mean, cov = pred(t_all[:k], Y_all[:k], float(t_all[k]))
        rows.append({
            "held_out_patient": patno, "model": arm, "target_index": k,
            "history_length": k, "event_target": held["event"][k],
            "t_last_history": float(t_all[k - 1]), "t_target": float(t_all[k]),
            "horizon_years": float(t_all[k] - t_all[k - 1]),
            "obs_tremor": float(Y_all[k, 0]), "obs_pigd": float(Y_all[k, 1]),
            "obs_subtype": held["subtype"][k],
            "pred_tremor": float(mean[0]), "pred_pigd": float(mean[1]),
            "var_tremor": float(cov[0, 0]), "var_pigd": float(cov[1, 1]),
            "cov_tremor_pigd": float(cov[0, 1]),
            "n_train_patients": len(train),
        })
    diag.update({"held_out_patient": patno, "model": arm,
                 "n_train_patients": len(train),
                 "task_seconds": time.perf_counter() - t0})
    (run_dir / "folds" / arm).mkdir(parents=True, exist_ok=True)
    fit_json = out_csv.with_name(f"{patno}_fit.json")
    tmp = fit_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(diag, default=float))
    os.replace(tmp, fit_json)
    _atomic_write_csv(pd.DataFrame(rows), out_csv)       # written last = done marker
    return arm, patno, "done", time.perf_counter() - t0


# ---------------------------------------------------------------------------
def write_metadata(run_dir: Path, args, cfg, panel_md5, seqs, arms, scored):
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
        except Exception:                                      # noqa: BLE001
            return None
    meta = {
        "stage": "3 - subscore held-out comparison",
        "script": "scripts/21_subscore_heldout_comparison.py",
        "argv": sys.argv, "arms": arms, "seed": SEED, "jitter": JITTER,
        "fold_restarts": args.fold_restarts, "max_iter": args.max_iter,
        "panel_md5": panel_md5, "expected_panel_md5": EXPECTED_PANEL_MD5,
        "smoke_test": bool(args.smoke),
        "n_patients_cohort": len(seqs), "n_patients_scored": len(scored),
        "n_points_per_arm": int(sum(len(seqs[p]["t"]) - 1 for p in scored)),
        "config_file": os.environ.get("PPMI_CONFIG", "config/config.yaml"),
        "subtype_thresholds": [cfg.subtype.pigd_threshold, cfg.subtype.td_threshold],
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_files": git("status", "--porcelain"),
        "python": platform.python_version(), "numpy": np.__version__,
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = run_dir / "run_config.json"
    if path.exists():                       # resume: keep the original, log the new
        with open(run_dir / "resume_log.jsonl", "a") as f:
            f.write(json.dumps(meta) + "\n")
        prior = json.loads(path.read_text())
        for key in ("arms", "seed", "jitter", "fold_restarts", "max_iter", "panel_md5"):
            if prior[key] != meta[key]:
                raise SystemExit(f"resume refused: {key} changed "
                                 f"({prior[key]!r} -> {meta[key]!r})")
    else:
        path.write_text(json.dumps(meta, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run directory (unique per run)")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--fold-restarts", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N patients")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="allow a non-frozen panel (synthetic smoke tests only)")
    args = ap.parse_args()

    cfg = load_config()
    run_dir = Path(args.out)
    if run_dir.exists() and any(run_dir.glob("folds/*/*.csv")) and not args.resume:
        raise SystemExit(f"{run_dir} already holds results; pass --resume to continue it")
    run_dir.mkdir(parents=True, exist_ok=True)

    panel_path = cfg.path("outputs") / "visit_panel.csv"
    panel_md5 = md5(panel_path)
    if panel_md5 != EXPECTED_PANEL_MD5 and not args.smoke:
        raise SystemExit(f"panel MD5 {panel_md5} != frozen {EXPECTED_PANEL_MD5}; refusing")

    arms = ALL_ARMS if args.arms == "all" else args.arms.split(",")
    unknown = set(arms) - set(ALL_ARMS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}")

    frame, seqs = build_sequences(cfg)
    patients = sorted(seqs)
    scored = patients[:args.limit] if args.limit else patients
    write_metadata(run_dir, args, cfg, panel_md5, seqs, arms, scored)
    print(f"cohort {len(seqs)} patients, scoring {len(scored)}; arms {arms}; "
          f"{args.jobs} workers", flush=True)

    # Longest tasks first so the tail is short.
    cost = {"gp_joint_K2_full": 9, "gp_joint_K3_full": 9, "gp_joint_K2_diag": 5,
            "gp_joint_K3_diag": 5, "gp_joint_K1_full": 4, "gp_joint_K0_full": 3}
    tasks = sorted(((a, p) for a in arms for p in scored),
                   key=lambda ap_: -cost.get(ap_[0], 1))
    tasks = [(a, p) for a, p in tasks
             if not (run_dir / "folds" / a / f"{p}.csv").exists()]
    print(f"{len(tasks)} tasks pending", flush=True)

    t0, done, failed = time.time(), 0, []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                             initargs=(seqs, str(run_dir), args.fold_restarts,
                                       args.max_iter)) as pool:
        futs = {pool.submit(run_task, a, p): (a, p) for a, p in tasks}
        for fut in as_completed(futs):
            a, p = futs[fut]
            try:
                fut.result()
            except Exception:                                  # noqa: BLE001
                failed.append({"arm": a, "patient": p, "error": traceback.format_exc()})
                with open(run_dir / "failed_tasks.jsonl", "a") as f:
                    f.write(json.dumps(failed[-1]) + "\n")
            done += 1
            if done % 200 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  {done}/{len(tasks)}  {el / 60:.1f} min  "
                      f"eta {(len(tasks) - done) * el / done / 60:.1f} min  "
                      f"failed {len(failed)}", flush=True)
    print(f"finished: {done} tasks, {len(failed)} failed, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
