"""Stage 6b: item-level (graded-response) PIGD measurement model — predictive check.

Implements docs/cluster_phase/06_AMENDMENT2_PREDICTIVE_CHECK_SPEC.md exactly
(its SHA-256 is recorded). Measurement-model check only: no clustering metric,
no GMM, no ARI is computed; script 26 is used only for its data loader and
`repack` helper.

    python scripts/32_grm_predictive_check.py --out <dir> [--reps 40] [--jobs 60]

Writes item_fit.json, item_fit_dif.csv, longitudinal_fit.json,
laplace_vs_agq.csv, real_features.json, simulated_features.csv,
calibration_table.csv, reported_features.csv, decision.json.
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
from scipy.stats import kurtosis, skew  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ppmi_subtype import grm as G  # noqa: E402
from ppmi_subtype import timevarying as TV  # noqa: E402
from ppmi_subtype import trajectory as tj  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype.subtype import clean_items  # noqa: E402

S26 = importlib.import_module("26_clusterability_extended")
A30 = importlib.import_module("30_pigd_measurement_audit")

SEED = 42
SPEC = ROOT / "docs" / "cluster_phase" / "06_AMENDMENT2_PREDICTIVE_CHECK_SPEC.md"
COMPONENTS = ("int_tremor", "int_pigd", "slope_tremor", "slope_pigd")

#: (target, tolerance) — fixed in the specification, section 5.
GATED = ([(f"blup_skew_{c}", 0.30) for c in COMPONENTS]
         + [("blup_kurt_slope_tremor", 1.5), ("blup_kurt_slope_pigd", 1.5),
            ("zero_share", 0.025), ("zero_share_y0", 0.04), ("zero_share_y5plus", 0.04),
            ("P_next_zero_given_S0", 0.05), ("P_next_zero_given_S1", 0.035),
            ("share_patients_all_zero", 0.02), ("within_sd_ratio_top_bottom", 1.5)])
_S: dict = {}


def load_items(cfg, data, item_names):
    """Source items per patient, in the BLUP pipeline's visit order; verified."""
    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    traj = tj.build_trajectories(panel, cfg)
    raw = A30.raw_items(cfg)
    cl = pd.concat([raw[["PATNO", "EVENT_ID"]], clean_items(raw, tuple(item_names), cfg.subtype)],
                   axis=1)
    f = traj[["PATNO", "EVENT_ID", "years_from_baseline"]].merge(
        cl, on=["PATNO", "EVENT_ID"], how="left", validate="one_to_one")
    f = f.sort_values(["PATNO", "years_from_baseline"], kind="mergesort")
    items = {}
    for p, g in f.groupby("PATNO", sort=True):
        Y = g[item_names].to_numpy(float)
        if not np.allclose(g.years_from_baseline.to_numpy(), data.times[p]):
            raise SystemExit(f"visit order mismatch for patient {p}")
        if not np.allclose(np.nanmean(Y, axis=1), data.responses[p][:, 1], atol=1e-12):
            raise SystemExit(f"item mean != pigd_score for patient {p}")
        items[p] = Y
    return items, f


def features(data, items) -> dict:
    """Every gated and reported feature of the specification (no clustering)."""
    fit = TV.fit(data, knots=(), random_slopes=True)
    blup = np.array([TV.blup(fit, data.times[p], data.responses[p])[0] for p in data.patients])
    out = {}
    for j, c in enumerate(COMPONENTS):
        out[f"blup_skew_{c}"] = float(skew(blup[:, j]))
        out[f"blup_kurt_{c}"] = float(kurtosis(blup[:, j]))
    S_all, yrs, pairs = [], [], []
    for p in data.patients:
        S = np.round(data.responses[p][:, 1] * 5)
        S_all.append(S)
        yrs.append(data.times[p])
        pairs += list(zip(S[:-1], S[1:]))
    S_all, yrs, pairs = np.concatenate(S_all), np.concatenate(yrs), np.array(pairs)
    out["zero_share"] = float(np.mean(S_all == 0))
    for lo, hi, lab in ((0, 1, "y0"), (1, 3, "y1_2"), (3, 5, "y3_4"), (5, 99, "y5plus")):
        m = (yrs >= lo) & (yrs < hi)
        out[f"zero_share_{lab}"] = float(np.mean(S_all[m] == 0))
    for lab, sel in (("S0", pairs[:, 0] == 0), ("S1", pairs[:, 0] == 1),
                     ("S2", pairs[:, 0] == 2), ("S3plus", pairs[:, 0] >= 3)):
        out[f"P_next_zero_given_{lab}"] = float(np.mean(pairs[sel, 1] == 0)) if sel.any() else np.nan
    out["share_patients_all_zero"] = float(np.mean(
        [np.all(data.responses[p][:, 1] == 0) for p in data.patients]))
    st = pd.DataFrame([(data.responses[p][:, 1].mean(), data.responses[p][:, 1].std(ddof=1))
                       for p in data.patients if len(data.times[p]) >= 4], columns=["m", "s"])
    oct_ = pd.qcut(st.m.rank(method="first"), 8, labels=False)
    out["within_sd_ratio_top_bottom"] = float(st.s[oct_ == 7].mean() / st.s[oct_ == 0].mean())
    Y = np.concatenate([items[p] for p in data.patients])
    for k, name in enumerate(_S["item_names"]):
        out[f"item_pos_{name}"] = float(np.mean(Y[:, k] >= 1))
        for c in range(5):
            out[f"item_cat{c}_{name}"] = float(np.mean(Y[:, k] == c))
    res, fitted = [], []
    for p in data.patients:
        t, R = data.times[p], data.responses[p]
        b, _ = TV.blup(fit, t, R)
        mu = (TV.design(t, ()) @ fit.beta + TV.random_design(t, True) @ b).reshape(-1, 2)[:, 1]
        res.append(R[:, 1] - mu)
        fitted.append(mu)
    res, fitted = np.concatenate(res), np.concatenate(fitted)
    dec = pd.qcut(pd.Series(fitted).rank(method="first"), 10, labels=False)
    for d in range(10):
        out[f"resid_sd_decile{d}"] = float(res[dec.to_numpy() == d].std(ddof=1))
    return out


def _task(rep):
    data, it, lf = _S["data"], _S["items_fit"], _S["long_fit"]
    rng = np.random.default_rng([SEED, 32, rep])
    times = {p: data.times[p] for p in data.patients}
    sim_items = G.simulate(rng, times, lf["beta"], lf["D"], it["a"], it["tau"])
    resp = {p: np.column_stack([data.responses[p][:, 0], sim_items[p].mean(axis=1)])
            for p in data.patients}
    sim = S26.repack(times, resp)
    row = features(sim, sim_items)
    row["rep"] = rep
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--agq-patients", type=int, default=30)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    md5 = S26.md5(cfg.path("outputs") / "visit_panel.csv")
    if md5 != S26.EXPECTED_PANEL_MD5:
        raise SystemExit("panel MD5 differs from the frozen panel")
    names = list(cfg.subtype.pigd_items)
    _S["item_names"] = names
    data = S26.load_data(cfg)
    items, f = load_items(cfg, data, names)
    meta = {"argv": sys.argv, "seed": SEED, "reps": args.reps, "panel_md5": md5,
            "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
            "n_patients": len(items), "n_visits": int(sum(len(v) for v in items.values())),
            "n_missing_items": int(sum(np.isnan(v).sum() for v in items.values())),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip(),
            "git_dirty": subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(), "numpy": np.__version__,
            "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    (out / "run_config.json").write_text(json.dumps(meta, indent=2))

    # ---------------------------------------------------------------- stage 1
    Yall = np.concatenate([items[p] for p in data.patients])
    t0 = time.time()
    it = G.fit_items(Yall, names)
    tall = np.concatenate([data.times[p] for p in data.patients])
    dif = []
    for lab, sel in (("all", np.ones(len(tall), bool)), ("year0", tall < 1), ("year3plus", tall >= 3)):
        fi = it if lab == "all" else G.fit_items(Yall[sel], names)
        for k, n in enumerate(names):
            dif.append({"subset": lab, "n_visits": int(sel.sum()), "item": n, "a": fi.a[k],
                        **{f"tau{c + 1}": fi.tau[k, c] for c in range(4)},
                        "converged": fi.converged})
    pd.DataFrame(dif).to_csv(out / "item_fit_dif.csv", index=False)
    item_json = {"a": it.a.tolist(), "tau": it.tau.tolist(), "nll": it.nll,
                 "converged": it.converged, "message": it.message, "items": names,
                 "seconds": time.time() - t0}
    (out / "item_fit.json").write_text(json.dumps(item_json, indent=2))
    print("stage 1:", json.dumps(item_json, indent=1), flush=True)

    # ---------------------------------------------------------------- stage 2
    t0 = time.time()
    packed = G.pack({p: data.times[p] for p in data.patients}, items)
    lf = G.fit_longitudinal(packed, it.a, it.tau)
    long_json = {"beta": lf.beta.tolist(), "D": lf.D.tolist(), "nll": lf.nll,
                 "converged": lf.converged, "message": lf.message, "n_iter": lf.n_iter,
                 "grad_norm": lf.grad_norm, "seconds": time.time() - t0,
                 "sd_intercept": float(np.sqrt(lf.D[0, 0])),
                 "sd_slope": float(np.sqrt(lf.D[1, 1])),
                 "corr": float(lf.D[0, 1] / np.sqrt(lf.D[0, 0] * lf.D[1, 1]))}
    (out / "longitudinal_fit.json").write_text(json.dumps(long_json, indent=2))
    print("stage 2:", json.dumps(long_json, indent=1), flush=True)

    rng = np.random.default_rng(SEED)
    agq = []
    for p in rng.choice(data.patients, min(args.agq_patients, len(data.patients)), replace=False):
        lap = G.laplace_loglik_one(data.times[p], items[p], lf.beta, lf.D, it.a, it.tau)
        q = G.agq_loglik(data.times[p], items[p], lf.beta, lf.D, it.a, it.tau)
        agq.append({"PATNO": int(p), "n_visits": len(data.times[p]), "laplace": lap, "agq": q,
                    "abs_diff": abs(lap - q)})
    agq = pd.DataFrame(agq)
    agq.to_csv(out / "laplace_vs_agq.csv", index=False)
    print(f"Laplace vs AGQ: max |diff| {agq.abs_diff.max():.4f}, "
          f"median {agq.abs_diff.median():.4f}", flush=True)

    # ---------------------------------------------------------------- features
    real = features(data, items)
    (out / "real_features.json").write_text(json.dumps(real, indent=2))
    state = {"data": data, "item_names": names,
             "items_fit": {"a": it.a, "tau": it.tau}, "long_fit": {"beta": lf.beta, "D": lf.D}}
    with ProcessPoolExecutor(args.jobs, initializer=_S.update, initargs=(state,)) as pool:
        sims = pd.DataFrame(list(pool.map(_task, range(args.reps), chunksize=1)))
    sims.to_csv(out / "simulated_features.csv", index=False)

    gated = [(n, tol) for n, tol in GATED] + [(f"item_pos_{n}", 0.03) for n in names]
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
                "stage1_converged": it.converged, "stage2_converged": lf.converged,
                "laplace_agq_max_abs_diff": float(agq.abs_diff.max())}
    (out / "decision.json").write_text(json.dumps(decision, indent=2))
    print(table.round(4).to_string(index=False))
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
