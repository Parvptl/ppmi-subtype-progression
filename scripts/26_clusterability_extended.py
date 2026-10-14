"""Stage 6: is there anything reliable to cluster? Extends script 20.

Script 20 compared the real data with ONE simulated no-groups dataset, drawn
from a Gaussian model with continuous, unbounded scores. Two gaps follow:

1. One replicate gives no null *distribution*; "real exceeds null" is a single
   draw.
2. Real subscores are bounded at 0 and discrete (tremor on a 1/11 grid, PIGD on
   a 1/5 grid; 692 PIGD == 0 visits). A floor plus discreteness can make
   shrunken slopes non-Gaussian, which a Gaussian mixture splits into
   "clusters". A Gaussian null cannot reproduce that, so "real > null" could be
   a measurement artefact.

This script therefore uses, with the LMM REFITTED on every replicate:

    null_gauss       one population, Gaussian, real visit times (script 20's null)
    null_measured    the same, then clipped to [0, 4] and rounded to the item grid
    planted_<d>      two latent groups whose PIGD slope differs by d random-slope
                     SDs (positive control: what can this visit density detect?)

Features are the BLUP random slopes (tremor, PIGD) from the random-slope LMM,
standardised, as in script 20. For every dataset (real and simulated):

    separation   GMM BIC for K = 1..5, dBIC = BIC(1) - min_K BIC(K); silhouette K=2,3
    stability    refit after dropping 1 or 2 random visits per patient, or after
                 truncating follow-up at 2, 3, 5 years; Spearman correlation of
                 BLUP slopes with the full-data ones, and ARI of the K=2/K=3
                 assignments

Pre-specified decision rule (written before the real run):
    clustering is supported only if, for the real data,
      (a) dBIC exceeds the 95th percentile of null_measured, AND
      (b) drop-one-visit ARI (K=2) exceeds the 95th percentile of null_measured
          AND is at least 0.6.
Otherwise no clustering model is fitted. The planted controls say how large a
real group difference would have to be for (a) and (b) to be met.

    python scripts/26_clusterability_extended.py --out <run_dir>
        [--null-reps 200] [--planted 0.5,1,2] [--planted-reps 50] [--jobs 60]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import jointlmm as J  # noqa: E402
from ppmi_subtype import timevarying as TV  # noqa: E402
from ppmi_subtype import trajectory as tj  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

SEED = 42
EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
GRID = (1 / 11, 1 / 5)                 # tremor = mean of 11 items, PIGD of 5
PERTURB = ["drop1", "drop2", "trunc2", "trunc3", "trunc5"]
KS = (1, 2, 3, 4, 5)
_S: dict = {}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_data(cfg):
    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    traj = tj.build_trajectories(panel, cfg)
    tj.check_ordering(traj)
    frame = traj.merge(panel[["PATNO", "EVENT_ID", *TV.OUTCOMES]].drop_duplicates(
        ["PATNO", "EVENT_ID"]), on=["PATNO", "EVENT_ID"], how="left", validate="one_to_one")
    return J.pack(frame)


def repack(times: dict, resp: dict) -> J.PackedData:
    rows = [{"PATNO": p, "years_from_baseline": t, "tremor_score": y[0], "pigd_score": y[1]}
            for p in times for t, y in zip(times[p], resp[p])]
    return J.pack(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
#  Features, separation, stability
# ---------------------------------------------------------------------------
def blup_slopes(data, fit):
    return np.array([TV.blup(fit, data.times[p], data.responses[p])[0][2:4]
                     for p in data.patients])


def gmm_labels(Z, K, seed=0):
    return GaussianMixture(K, covariance_type="full", random_state=seed,
                           n_init=5).fit_predict(Z)


def separation(S):
    Z = (S - S.mean(0)) / S.std(0)
    bic = {K: GaussianMixture(K, covariance_type="full", random_state=0,
                              n_init=5).fit(Z).bic(Z) for K in KS}
    out = {f"BIC_K{K}": v for K, v in bic.items()}
    out["dBIC"] = bic[1] - min(bic.values())
    out["best_K"] = int(min(bic, key=bic.get))
    for K in (2, 3):
        out[f"silhouette_K{K}"] = float(silhouette_score(Z, gmm_labels(Z, K)))
    return out, Z


def perturb(data, kind, rng):
    times, resp = {}, {}
    for p in data.patients:
        t, Y = data.times[p], data.responses[p]
        keep = np.ones(len(t), bool)
        if kind.startswith("drop"):
            n_drop = int(kind[-1])
            if len(t) - n_drop >= 2:
                keep[rng.choice(len(t), n_drop, replace=False)] = False
        else:
            keep = t <= float(kind[5:])
            if keep.sum() < 2:
                keep = np.zeros(len(t), bool)
                keep[:2] = True                 # everyone keeps their first two visits
        times[p], resp[p] = t[keep], Y[keep]
    return repack(times, resp)


def evaluate(data, seed):
    """Separation on the full data, and stability under every perturbation."""
    fit = TV.fit(data, knots=(), random_slopes=True)
    S = blup_slopes(data, fit)
    sep, Z = separation(S)
    ref = {K: gmm_labels(Z, K) for K in (2, 3)}
    rng = np.random.default_rng(seed)
    row = {**sep, "converged": fit.converged,
           "slope_sd_tremor": float(np.sqrt(fit.sigma_b[2, 2])),
           "slope_sd_pigd": float(np.sqrt(fit.sigma_b[3, 3]))}
    for kind in PERTURB:
        d2 = perturb(data, kind, rng)
        f2 = TV.fit(d2, knots=(), random_slopes=True)
        S2 = blup_slopes(d2, f2)
        Z2 = (S2 - S.mean(0)) / S.std(0)             # same scaling as the reference
        for j, name in enumerate(("tremor", "pigd")):
            row[f"{kind}_spearman_{name}"] = float(spearmanr(S[:, j], S2[:, j]).statistic)
        for K in (2, 3):
            row[f"{kind}_ARI_K{K}"] = float(adjusted_rand_score(ref[K], gmm_labels(Z2, K)))
    return row


# ---------------------------------------------------------------------------
#  Simulation
# ---------------------------------------------------------------------------
def simulate(data, fit, rng, measured: bool, planted_delta: float = 0.0):
    times, resp = {}, {}
    sd_slope_pigd = np.sqrt(fit.sigma_b[3, 3])
    for p in data.patients:
        t = data.times[p]
        b = rng.multivariate_normal(np.zeros(4), fit.sigma_b)
        if planted_delta:
            # Two equal-sized latent groups separated in PIGD slope by
            # planted_delta SDs, centred so the population mean is unchanged.
            b[3] += (0.5 if rng.random() < 0.5 else -0.5) * planted_delta * sd_slope_pigd
        e = rng.multivariate_normal(np.zeros(2), fit.sigma_e, size=len(t)).reshape(-1)
        y = (TV.design(t, ()) @ fit.beta + TV.random_design(t, True) @ b + e).reshape(-1, 2)
        if measured:
            y = np.clip(y, 0.0, 4.0)
            y = np.column_stack([np.round(y[:, 0] / GRID[0]) * GRID[0],
                                 np.round(y[:, 1] / GRID[1]) * GRID[1]])
        times[p], resp[p] = t, y
    return repack(times, resp)


def _task(args):
    kind, rep, delta = args
    data, fit = _S["data"], _S["fit"]
    stream = {"null_gauss": 1, "null_measured": 2}.get(kind, 3)
    rng = np.random.default_rng([SEED, stream, rep, int(round(1000 * delta))])
    sim = simulate(data, fit, rng, measured=kind != "null_gauss", planted_delta=delta)
    row = evaluate(sim, SEED + rep)
    row.update({"dataset": kind, "rep": rep, "planted_delta_sd": delta,
                "planted_delta_pigd_per_year": delta * float(np.sqrt(fit.sigma_b[3, 3]))})
    return row


# ---------------------------------------------------------------------------
def decide(real: pd.Series, null: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-specified rule against the measurement-realistic null."""
    nm = null[null.dataset == "null_measured"]
    rows = []
    for name, col, floor in [("a_dBIC_above_null95", "dBIC", None),
                             ("b_drop1_ARI_K2_above_null95", "drop1_ARI_K2", 0.6)]:
        val, thr = float(real[col]), float(nm[col].quantile(0.95))
        rows.append({"criterion": name, "real": val, "null_measured_q95": thr,
                     "absolute_floor": floor,
                     "real_percentile_in_null": float((nm[col] < val).mean()),
                     "met": bool(val > thr and (floor is None or val >= floor))})
    rows.append({"criterion": "CLUSTERING SUPPORTED", "met": all(r["met"] for r in rows)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--null-reps", type=int, default=200)
    ap.add_argument("--planted", default="0.5,1,2")
    ap.add_argument("--planted-reps", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--min-visits", type=int, default=None,
                    help="restrict to patients with >= this many visits (exploratory)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    run = Path(args.out)
    run.mkdir(parents=True, exist_ok=True)
    panel_md5 = md5(cfg.path("outputs") / "visit_panel.csv")
    if panel_md5 != EXPECTED_PANEL_MD5 and not args.smoke:
        raise SystemExit(f"panel MD5 {panel_md5} != frozen {EXPECTED_PANEL_MD5}")
    data = load_data(cfg)
    if args.min_visits:
        data = J.subset(data, [p for p in data.patients
                               if len(data.times[p]) >= args.min_visits])
    deltas = [float(x) for x in args.planted.split(",") if x]
    meta = {"argv": sys.argv, "seed": SEED, "panel_md5": panel_md5, "smoke_test": args.smoke,
            "n_patients": data.n_patients, "null_reps": args.null_reps,
            "planted_deltas_sd": deltas, "planted_reps": args.planted_reps,
            "min_visits": args.min_visits, "perturbations": PERTURB,
            "decision_rule": "dBIC > null_measured q95 AND drop1 ARI(K=2) > null_measured "
                             "q95 AND >= 0.6",
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(), "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    (run / "run_config.json").write_text(json.dumps(meta, indent=2))

    t0 = time.time()
    fit = TV.fit(data, knots=(), random_slopes=True)
    real = evaluate(data, SEED)
    real.update({"dataset": "real", "rep": 0, "planted_delta_sd": 0.0})
    pd.DataFrame([real]).to_csv(run / "real.csv", index=False)
    print(f"real data evaluated in {time.time() - t0:.0f}s: dBIC {real['dBIC']:.1f}, "
          f"best K {real['best_K']}, drop1 ARI K2 {real['drop1_ARI_K2']:.3f}", flush=True)

    tasks = ([("null_gauss", r, 0.0) for r in range(args.null_reps)]
             + [("null_measured", r, 0.0) for r in range(args.null_reps)]
             + [(f"planted_{d:g}", r, d) for d in deltas for r in range(args.planted_reps)])
    rows = []
    with ProcessPoolExecutor(args.jobs, initializer=_S.update,
                             initargs=({"data": data, "fit": fit},)) as pool:
        for i, row in enumerate(pool.map(_task, tasks, chunksize=1), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(tasks):
                pd.DataFrame(rows).to_csv(run / "simulations.csv", index=False)
                print(f"  {i}/{len(tasks)} simulations, {(time.time() - t0) / 60:.1f} min",
                      flush=True)
    sims = pd.DataFrame(rows)
    sims.to_csv(run / "simulations.csv", index=False)

    summ = []
    cols = ["dBIC", "best_K", "silhouette_K2", "silhouette_K3",
            *[f"{k}_ARI_K2" for k in PERTURB], *[f"{k}_spearman_pigd" for k in PERTURB]]
    for ds, g in pd.concat([sims, pd.DataFrame([real])]).groupby("dataset"):
        r = {"dataset": ds, "n": len(g)}
        for c in cols:
            r[f"{c}_median"] = float(g[c].median())
            r[f"{c}_q05"] = float(g[c].quantile(0.05))
            r[f"{c}_q95"] = float(g[c].quantile(0.95))
        summ.append(r)
    pd.DataFrame(summ).to_csv(run / "summary.csv", index=False)
    decide(pd.Series(real), sims).to_csv(run / "decision.csv", index=False)
    print(pd.read_csv(run / "decision.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
