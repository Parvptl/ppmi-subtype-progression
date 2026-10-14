"""Compare candidate PIGD measurement models by simulation-based predictive checks.

Diagnostic for the Stage 6b redesign. No clustering metric is computed.

    M1  current: Gaussian residual with constant variance (RS-LMM Sigma_e),
        then clip to [0, 4] and round to the 1/5 item grid
    M2  heteroscedastic Gaussian: residual SD linear in the fitted level
        (slope/intercept from the conditional-residual audit, rescaled so the
        mean variance equals the fitted Sigma_e), then clip and round
    M5  beta-binomial mixed model on the item sum S = 5 * PIGD (n = 20),
        Gaussian random intercept and slope on the logit scale

Each model is fitted to the real PIGD subscore, then `--reps` datasets are
simulated on every patient's real visit times. Tremor is kept at its real
values, so the bivariate Gaussian RS-LMM and its BLUPs (the Stage 6 feature
pipeline) are applied exactly as to the real data. Features compared with the
real data:

    zero share overall and by follow-up year band; P(next = 0 | current S);
    share of patients with PIGD = 0 at every visit; skewness and 95th centile
    of S | S > 0; within-patient SD ratio (top vs bottom octile of patient
    means); BLUP skewness / excess kurtosis of PIGD slope and intercept

The two-part (hurdle) model is assessed through M5: if M5 reproduces the zero
frequency and the transitions into zero at every level, the data give no
evidence for a separate zero-generating process.

    python scripts/31_pigd_measurement_models.py --out <dir> [--reps 20] [--jobs 60]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
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
from scipy.stats import kurtosis, skew  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ppmi_subtype import bbglmm as BB  # noqa: E402
from ppmi_subtype import timevarying as TV  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

S26 = importlib.import_module("26_clusterability_extended")
SEED = 42
GRID_P = 1 / 5
_S: dict = {}


def features(data, fit=None) -> dict:
    """Measurement-level and BLUP-shape features. Not clustering metrics."""
    fit = fit or TV.fit(data, knots=(), random_slopes=True)
    blup = np.array([TV.blup(fit, data.times[p], data.responses[p])[0] for p in data.patients])
    out = {"blup_skew_slope_pigd": skew(blup[:, 3]), "blup_kurt_slope_pigd": kurtosis(blup[:, 3]),
           "blup_skew_int_pigd": skew(blup[:, 1]), "blup_skew_slope_tremor": skew(blup[:, 2])}
    S_all, yrs, nxt = [], [], []
    for p in data.patients:
        S = np.round(data.responses[p][:, 1] * 5)
        S_all.append(S)
        yrs.append(data.times[p])
        nxt += list(zip(S[:-1], S[1:]))
    S_all, yrs = np.concatenate(S_all), np.concatenate(yrs)
    out["zero_share"] = float(np.mean(S_all == 0))
    for lo, hi, lab in ((0, 1, "y0"), (1, 3, "y1_2"), (3, 5, "y3_4"), (5, 99, "y5plus")):
        m = (yrs >= lo) & (yrs < hi)
        out[f"zero_share_{lab}"] = float(np.mean(S_all[m] == 0))
    nxt = np.array(nxt)
    for k in (0, 1, 2):
        m = nxt[:, 0] == k
        out[f"P_next_zero_given_S{k}"] = float(np.mean(nxt[m, 1] == 0)) if m.any() else np.nan
    out["share_patients_all_zero"] = float(np.mean(
        [np.all(data.responses[p][:, 1] == 0) for p in data.patients]))
    pos = S_all[S_all > 0]
    out["pos_skew"], out["pos_q95"] = float(skew(pos)), float(np.quantile(pos, 0.95))
    st = pd.DataFrame([(data.responses[p][:, 1].mean(), data.responses[p][:, 1].std(ddof=1))
                       for p in data.patients if len(data.times[p]) >= 4], columns=["m", "s"])
    oct_ = pd.qcut(st.m.rank(method="first"), 8, labels=False)
    # Octile means, not medians: in the bottom octile most within-patient SDs
    # are exactly 0 (all-zero patients), which makes a median ratio undefined.
    out["within_sd_ratio_top_bottom"] = float(st.s[oct_ == 7].mean() / st.s[oct_ == 0].mean())
    return out


def with_pigd(data, pigd: dict):
    times = {p: data.times[p] for p in data.patients}
    resp = {p: np.column_stack([data.responses[p][:, 0], pigd[p]]) for p in data.patients}
    return S26.repack(times, resp)


def sim_gaussian(rng, data, fit, sd_fn):
    out = {}
    B = rng.multivariate_normal(np.zeros(4), fit.sigma_b, size=len(data.patients))
    for i, p in enumerate(data.patients):
        t = data.times[p]
        mu = (TV.design(t, ()) @ fit.beta + TV.random_design(t, True) @ B[i]).reshape(-1, 2)[:, 1]
        y = mu + sd_fn(mu) * rng.standard_normal(len(t))
        out[p] = np.round(np.clip(y, 0, 4) / GRID_P) * GRID_P
    return out


def _task(args):
    model, rep = args
    data, fit, bb, hs = _S["data"], _S["fit"], _S["bb"], _S["hs"]
    rng = np.random.default_rng([SEED, {"M1": 1, "M2": 2, "M5": 5}[model], rep])
    if model == "M1":
        sd = float(np.sqrt(fit.sigma_e[1, 1]))
        pigd = sim_gaussian(rng, data, fit, lambda mu: np.full_like(mu, sd))
    elif model == "M2":
        a, b, scale = hs
        pigd = sim_gaussian(rng, data, fit,
                            lambda mu: scale * np.maximum(a + b * np.clip(mu, 0, None), 0.02))
    else:
        sums = BB.simulate(rng, bb, {p: data.times[p] for p in data.patients})
        pigd = {p: sums[p] / 5.0 for p in data.patients}
    row = features(with_pigd(data, pigd))
    row.update({"model": model, "rep": rep})
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--audit-dir", default=None,
                    help="script 30 run dir; M2's SD function comes from its residual_by_fitted.csv")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    md5 = S26.md5(cfg.path("outputs") / "visit_panel.csv")
    if md5 != S26.EXPECTED_PANEL_MD5 and not args.smoke:
        raise SystemExit("panel MD5 differs from the frozen panel")
    data = S26.load_data(cfg)
    fit = TV.fit(data, knots=(), random_slopes=True)
    real = features(data, fit)

    t0 = time.time()
    packed = BB.pack({p: data.times[p] for p in data.patients},
                     {p: np.round(data.responses[p][:, 1] * 5) for p in data.patients}, 20)
    bb = BB.fit(packed)
    bb_secs = time.time() - t0

    if args.audit_dir:
        rb = pd.read_csv(Path(args.audit_dir) / "residual_by_fitted.csv")
        rb = rb[rb.outcome == "pigd"]
    else:                                   # smoke: estimate from this fit's residuals
        rb = None
    if rb is not None and len(rb) >= 3:
        b, a = np.polyfit(rb.fitted_mean.clip(lower=0), rb.resid_sd, 1, w=rb.n)
    else:
        a, b = float(np.sqrt(fit.sigma_e[1, 1])), 0.0
    mus = np.concatenate([(TV.design(data.times[p], ()) @ fit.beta).reshape(-1, 2)[:, 1]
                          for p in data.patients])
    raw = np.maximum(a + b * np.clip(mus, 0, None), 0.02)
    scale = float(np.sqrt(fit.sigma_e[1, 1] / np.mean(raw ** 2)))

    meta = {"argv": sys.argv, "panel_md5": md5, "smoke": args.smoke, "reps": args.reps,
            "seed": SEED, "gaussian_sigma_e_pigd": float(fit.sigma_e[1, 1]),
            "M2_sd_fn": {"a": float(a), "b": float(b), "scale": scale},
            "M5_fit": {"beta": bb.beta.tolist(), "D": bb.D.tolist(), "phi": bb.phi,
                       "rho": bb.rho, "nll": bb.nll, "converged": bb.converged,
                       "message": bb.message, "at_bound": bb.at_bound, "seconds": bb_secs},
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(), "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    (out / "run_config.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["M5_fit"], indent=1), flush=True)

    tasks = [(m, r) for m in ("M1", "M2", "M5") for r in range(args.reps)]
    with ProcessPoolExecutor(args.jobs, initializer=_S.update,
                             initargs=({"data": data, "fit": fit, "bb": bb,
                                        "hs": (a, b, scale)},)) as pool:
        rows = list(pool.map(_task, tasks, chunksize=1))
    sims = pd.DataFrame(rows)
    sims.to_csv(out / "simulated_features.csv", index=False)
    pd.DataFrame([{**real, "model": "real"}]).to_csv(out / "real_features.csv", index=False)
    feats = [c for c in real]
    summ = []
    for m, g in sims.groupby("model"):
        for c in feats:
            summ.append({"model": m, "feature": c, "real": real[c], "sim_mean": g[c].mean(),
                         "sim_q05": g[c].quantile(0.05), "sim_q95": g[c].quantile(0.95),
                         "real_inside_5_95": bool(g[c].quantile(0.05) <= real[c]
                                                  <= g[c].quantile(0.95))})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "predictive_checks.csv", index=False)
    wide = summ.pivot(index="feature", columns="model", values="sim_mean")
    wide.insert(0, "real", pd.Series(real))
    print(wide.round(3).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
