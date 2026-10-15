"""Student-t vs Gaussian patient slopes in the TD/PIGD-ratio growth model.

Design: docs/cluster_phase/07_STUDENT_T_SLOPE_SPEC.md. The Gaussian baseline is
`hbayes.fit_growth_model`, unchanged; the alternative is
`hbayes_t.fit_t_growth_model` (Student-t b1i, everything else identical).

    python scripts/35_student_t_slope_model.py --out <dir> --stage smoke [--jobs 8]
    python scripts/35_student_t_slope_model.py --out <dir> --stage full  [--jobs 8]

smoke: parameter recovery on the real visit design, multi-chain sampler
       behaviour on the primary training split, and an evaluation-pipeline
       check on a few test patients (Gaussian scorer vs hbayes.evaluate only;
       no Student-t vs Gaussian predictive comparison is computed).
full:  the held-out comparison on script 04's primary split and its six
       robustness splits, plus full-cohort nu posterior and prior sensitivity.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import hbayes  # noqa: E402
from ppmi_subtype import hbayes_t as HT  # noqa: E402
from ppmi_subtype import slope_clustering as SC  # noqa: E402
from ppmi_subtype import splitting  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

SEED = 42
PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
ROBUST_SPLIT_SEEDS = (0, 1, 2, 3, 4, 5)        # script 04, section 5 (n_splits = 6)
_S: dict = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def split_panels(panel, cfg, seed):
    man = splitting.split_patients(panel, cfg.splits["train"], seed)
    tr = set(man.loc[man["split"] == "train", "PATNO"])
    te = set(man.loc[man["split"] == "test", "PATNO"])
    return panel[panel["PATNO"].isin(tr)], panel[panel["PATNO"].isin(te)]


def chain_diag(chains, names):
    rows = []
    for name in names:
        c = np.stack([getattr(p, name) if name != "nu" else p.nu for p in chains])
        if name == "beta":
            for k, lab in enumerate(("beta0", "beta1")):
                rows.append(_diag_row(lab, c[:, :, k]))
            continue
        v = np.sqrt(c) if name in ("sigma2", "tau0_2", "tau1_2") else c
        lab = {"sigma2": "sigma", "tau0_2": "tau0", "tau1_2": "tau1"}.get(name, name)
        rows.append(_diag_row(lab, v))
    return rows


def _diag_row(lab, c):
    return {"parameter": lab, "mean": float(c.mean()), "q2.5": float(np.quantile(c, .025)),
            "q50": float(np.median(c)), "q97.5": float(np.quantile(c, .975)),
            "rhat": SC.split_rhat(c), "ess_total": SC.ess(c), "ess_chain1": SC.ess(c[:1])}


# ---------------------------------------------------------------------------
#  Smoke: parameter recovery on the real visit design
# ---------------------------------------------------------------------------
RESID_CODE = {"gaussian": 0, "t4": 1, "empirical": 2}


def _recovery_task(args):
    """Simulate on the real design; nu_true = inf means Gaussian slopes.

    resid: 'gaussian' (model-consistent), 't4' (t_4 noise scaled to sigma) or
    'empirical' (standardised residuals of the real Gaussian fit, resampled) --
    the last two test whether heavy-tailed noise masquerades as small nu.
    """
    nu_true, resid, rep = args
    d, th, kw = _S["data"], _S["theta"], _S["fit_kw"]
    rng = np.random.default_rng([SEED, 35, int(nu_true if np.isfinite(nu_true) else 0),
                                 RESID_CODE[resid], rep])
    M, i = d.n_patients, d.patient_index
    z = rng.standard_normal(M) if np.isinf(nu_true) else rng.standard_t(nu_true, M)
    b1 = np.sqrt(th["tau1_2"]) * z
    b0 = rng.normal(0, np.sqrt(th["tau0_2"]), M)
    if resid == "gaussian":
        e = rng.standard_normal(d.n_obs)
    elif resid == "t4":
        e = rng.standard_t(4, d.n_obs) / np.sqrt(2.0)          # unit variance
    else:
        e = rng.choice(_S["std_resid"], d.n_obs, replace=True)
    y = th["beta0"] + b0[i] + (th["beta1"] + b1[i]) * d.t + np.sqrt(th["sigma2"]) * e
    sim = hbayes.GrowthData(y=y, t=d.t, patient_index=i, patients=d.patients,
                            n_patients=M, n_obs=d.n_obs)
    t0 = time.time()
    post = HT.fit_t_growth_model(sim, seed=SEED + 100 + rep, **kw)
    nu = post.nu
    tau1 = np.sqrt(post.tau1_2)
    return {"nu_true": nu_true, "resid": resid, "rep": rep, "fit_s": time.time() - t0,
            "nu_q2.5": np.quantile(nu, .025), "nu_q50": np.median(nu),
            "nu_q97.5": np.quantile(nu, .975), "nu_ess": SC.ess(nu[None]),
            "nu_covered": bool(np.quantile(nu, .025) <= min(nu_true, 300)
                               <= np.quantile(nu, .975)),
            "P_nu_lt_10": float((nu < 10).mean()),
            "beta1_true": th["beta1"], "beta1_mean": post.beta[:, 1].mean(),
            "beta1_covered": bool(np.quantile(post.beta[:, 1], .025) <= th["beta1"]
                                  <= np.quantile(post.beta[:, 1], .975)),
            "tau1_true": np.sqrt(th["tau1_2"]), "tau1_q50": np.median(tau1),
            "tau1_covered": bool(np.quantile(tau1, .025) <= np.sqrt(th["tau1_2"])
                                 <= np.quantile(tau1, .975)),
            "sigma_true": np.sqrt(th["sigma2"]), "sigma_mean": np.sqrt(post.sigma2).mean(),
            "sample_sd_b1": float(b1.std())}


def _chain_task(args):
    kind, seed, prior = args
    d, kw = _S["data"], _S["fit_kw"]
    t0 = time.time()
    if kind == "gaussian":
        p = hbayes.fit_growth_model(d, seed=seed, **kw)
    else:
        lp = HT.log_prior_nu_gamma if prior == "gamma" else HT.log_prior_nu_loguniform
        p = HT.fit_t_growth_model(d, seed=seed, log_prior_nu=lp, **kw)
    return kind, seed, prior, p, time.time() - t0


def _init(state):
    _S.update(state)


def pool_map(fn, items, jobs, state):
    if jobs <= 1:
        _init(state)
        return [fn(a) for a in items]
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=(state,)) as ex:
        return list(ex.map(fn, items))


def smoke(a, cfg, panel, out):
    fit_kw = {}                                                   # function defaults
    meta = {"stage": "smoke", "fit": "defaults (4000 draws, 1000 burn-in, thin 2)"}

    # ---- B. parameter recovery on the full eligible cohort's visit design ---
    data_all = hbayes.build_growth_data(panel)
    g = hbayes.fit_growth_model(data_all, seed=SEED, keep_random_effects=True)
    theta = {"beta0": float(g.beta[:, 0].mean()), "beta1": float(g.beta[:, 1].mean()),
             "tau0_2": float(g.tau0_2.mean()), "tau1_2": float(g.tau1_2.mean()),
             "sigma2": float(g.sigma2.mean())}
    X = data_all.design()
    bm = g.random_effects.mean(axis=0)
    r = (data_all.y - X @ g.beta.mean(axis=0) - np.einsum("ij,ij->i", X, bm[data_all.patient_index]))
    std_resid = (r - r.mean()) / r.std()
    from scipy import stats
    meta["real_residuals"] = {"excess_kurtosis": float(stats.kurtosis(std_resid)),
                              "skew": float(stats.skew(std_resid)),
                              "frac_abs_gt_3": float(np.mean(np.abs(std_resid) > 3)),
                              "gaussian_frac_abs_gt_3": float(2 * stats.norm.sf(3))}
    reps = 2 if a.quick else 6
    tasks = [(nu, "gaussian", r) for nu in (2.0, 3.0, 6.0, np.inf) for r in range(reps)] \
        + [(np.inf, res, r) for res in ("t4", "empirical") for r in range(reps)]
    rec = pd.DataFrame(pool_map(_recovery_task, tasks, a.jobs,
                                {"data": data_all, "theta": theta, "fit_kw": fit_kw,
                                 "std_resid": std_resid}))
    rec.to_csv(out / "recovery.csv", index=False)
    log(f"recovery done: {len(rec)} fits")
    meta["recovery_theta"] = theta

    # ---- C. sampler behaviour on the primary training split ----------------
    train, test = split_panels(panel, cfg, cfg.seed)
    d_tr = hbayes.build_growth_data(train)
    rows = []
    long_kw = {"n_draws": 4000, "burn_in": 2000, "thin": 5}
    for cfg_name, kw in (("default", fit_kw), ("long_thin5", long_kw)):
        chains = pool_map(_chain_task,
                          [("student_t", SEED + k, "gamma") for k in range(4)]
                          + [("student_t", SEED + k, "loguniform") for k in range(4)]
                          + [("gaussian", SEED + k, None) for k in range(4)],
                          a.jobs, {"data": d_tr, "fit_kw": kw})
        for kind, prior in (("student_t", "gamma"), ("student_t", "loguniform"),
                            ("gaussian", None)):
            cs = [c for c in chains if c[0] == kind and c[2] == prior]
            names = ["beta", "sigma2", "tau0_2", "tau1_2"] + (["nu"] if kind != "gaussian"
                                                              else [])
            for r in chain_diag([c[3] for c in cs], names):
                rows.append({"config": cfg_name, "model": kind, "nu_prior": prior, **r,
                             "fit_s_mean": float(np.mean([c[4] for c in cs]))})
        if cfg_name == "default":
            default_chains = chains
    chains = default_chains
    sd = pd.DataFrame(rows)
    sd.to_csv(out / "sampler_diagnostics.csv", index=False)
    p0 = [c for c in chains if c[0] == "student_t" and c[2] == "gamma"][0][3]
    nu = p0.nu
    acf = [float(np.corrcoef(nu[:-k], nu[k:])[0, 1]) for k in (1, 5, 20, 50)]
    meta["train_split"] = {"patients": d_tr.n_patients, "obs": d_tr.n_obs,
                           "nu_lag_autocorr_1_5_20_50": acf,
                           "nu_distinct_grid_values_visited": int(len(np.unique(nu)))}
    np.savez_compressed(out / "nu_traces.npz",
                        **{f"{c[0]}_{c[2]}_seed{c[1]}": c[3].nu for c in chains
                           if c[0] == "student_t"})
    log("sampler diagnostics done")

    # ---- D. evaluation-pipeline check on a few test patients ---------------
    pg = [c for c in chains if c[0] == "gaussian" and c[1] == SEED][0][3]
    ids = np.sort(test["PATNO"].unique())[:25]
    sub = test[test["PATNO"].isin(ids)]
    t0 = time.time()
    preds = HT.evaluate_both(pg, p0, sub, max_draws=100, seed=SEED)
    t_both = time.time() - t0
    t0 = time.time()
    ref, ref_preds = hbayes.evaluate(pg, sub, max_draws=400, seed=SEED)
    t_ref = time.time() - t0
    mine = HT.metrics(preds[preds.model == "gaussian"])
    meta["eval_check"] = {
        "test_patients": len(ids), "transitions": int((preds.model == "gaussian").sum()),
        "hbayes_evaluate_transitions": ref["n_pairs"],
        "gaussian_rb_loglik_per_pair": mine["loglik_per_pair"],
        "hbayes_evaluate_loglik_per_pair": ref["loglik_per_pair"],
        "gaussian_rb_brier": mine["brier"], "hbayes_evaluate_brier": ref["brier"],
        "max_abs_prob_diff": float(np.abs(
            preds[preds.model == "gaussian"][["TD", "Indeterminate", "PIGD"]].to_numpy()
            - ref_preds[["p_TD", "p_Indeterminate", "p_PIGD"]].to_numpy()).max()),
        "student_t_rows_finite": bool(np.isfinite(
            preds[preds.model == "student_t"][["TD", "log_density"]].to_numpy()).all()),
        "seconds_both_models_100_draws": round(t_both, 1),
        "seconds_hbayes_evaluate_400_draws": round(t_ref, 1),
        "note": "Student-t vs Gaussian predictive metrics deliberately not computed in smoke",
    }
    meta["test_split"] = {"patients": int(test.PATNO.nunique())}
    return meta


# ---------------------------------------------------------------------------
#  Full (not run until authorised)
# ---------------------------------------------------------------------------
def _split_task(split_seed):
    panel, cfg, kw, md, ni = _S["panel"], _S["cfg"], _S["fit_kw"], _S["max_draws"], _S["n_inner"]
    train, test = split_panels(panel, cfg, split_seed)
    d = hbayes.build_growth_data(train)
    pg = hbayes.fit_growth_model(d, seed=split_seed, **kw)
    pt = HT.fit_t_growth_model(d, seed=split_seed, **kw)
    preds = HT.evaluate_both(pg, pt, test, max_draws=md, seed=split_seed, n_inner=ni)
    preds["split_seed"] = split_seed
    return split_seed, preds, {"nu_q50": float(np.median(pt.nu)),
                               "nu_q2.5": float(np.quantile(pt.nu, .025)),
                               "nu_q97.5": float(np.quantile(pt.nu, .975)),
                               "nu_ess": SC.ess(pt.nu[None]),
                               "train_patients": d.n_patients}


def full(a, cfg, panel, out):
    # Longer chains than the function defaults, for both models: smoke showed nu
    # ESS ~85 per default chain vs ~200 with burn-in 2000 / thin 5.
    fit_kw = {"n_draws": 4000, "burn_in": 2000, "thin": 5}
    if a.quick:                                   # code-path check only
        keep = np.sort(panel["PATNO"].unique())[:150]
        panel = panel[panel["PATNO"].isin(keep)]
        fit_kw = {"n_draws": 200, "burn_in": 100, "thin": 1}
    seeds = [cfg.seed, *ROBUST_SPLIT_SEEDS]
    res = pool_map(_split_task, seeds, a.jobs,
                   {"panel": panel, "cfg": cfg, "fit_kw": fit_kw, "max_draws": 400,
                    "n_inner": 20})
    log("split fits and held-out predictions done")
    # reference: script 04's own Monte-Carlo scorer for the Gaussian baseline, primary split
    tr, te = split_panels(panel, cfg, cfg.seed)
    ref, _ = hbayes.evaluate(hbayes.fit_growth_model(hbayes.build_growth_data(tr),
                                                     seed=cfg.seed, **fit_kw),
                             te, max_draws=400, seed=cfg.seed)
    preds = pd.concat([r[1] for r in res])
    preds.to_csv(out / "heldout_predictions.csv", index=False)
    rows = []
    for s, p, nu in res:
        for m in ("gaussian", "student_t"):
            rows.append({"split_seed": s, "model": m, **HT.metrics(p[p.model == m]),
                         **(nu if m == "student_t" else {})})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "heldout_by_split.csv", index=False)

    # paired patient bootstrap on the primary split
    p = preds[preds.split_seed == cfg.seed]
    w = p.pivot_table(index=["PATNO", "step"], columns="model",
                      values="log_density").reset_index()
    for m in ("gaussian", "student_t"):
        q = p[p.model == m].set_index(["PATNO", "step"])
        probs = np.clip(q[["TD", "Indeterminate", "PIGD"]].to_numpy(), 1e-6, None)
        probs /= probs.sum(1, keepdims=True)
        j = q.observed.map({"TD": 0, "Indeterminate": 1, "PIGD": 2}).to_numpy()
        w[f"cls_{m}"] = np.log(probs[np.arange(len(j)), j])
    w["d_lpd"] = w.student_t - w.gaussian
    w["d_cls"] = w.cls_student_t - w.cls_gaussian
    pat = w.groupby("PATNO")[["d_lpd", "d_cls"]].agg(["sum", "size"])
    rng = np.random.default_rng([SEED, 35, 9])
    ids = pat.index.to_numpy()
    boot = []
    for _ in range(2000):
        s = pat.loc[rng.choice(ids, len(ids))]
        n = s[("d_lpd", "size")].sum()
        boot.append((s[("d_lpd", "sum")].sum() / n, s[("d_cls", "sum")].sum() / n))
    boot = np.array(boot)
    by_hist = w.assign(n_hist=w.step + 1).groupby(
        pd.cut(w.step + 1, [0, 1, 2, 4, 100], labels=["1", "2", "3-4", ">=5"]),
        observed=True)[["d_lpd", "d_cls"]].agg(["mean", "size"])
    by_hist.to_csv(out / "primary_split_difference_by_history.csv")

    # full-cohort nu posterior: 4 chains x 2 priors
    d_all = hbayes.build_growth_data(panel)
    chains = pool_map(_chain_task, [("student_t", SEED + k, pr) for pr in ("gamma", "loguniform")
                                    for k in range(4)], a.jobs, {"data": d_all, "fit_kw": fit_kw})
    diag = []
    for pr in ("gamma", "loguniform"):
        cs = [c[3] for c in chains if c[2] == pr]
        diag += [{"nu_prior": pr, **r} for r in chain_diag(cs, ["beta", "sigma2", "tau0_2",
                                                              "tau1_2", "nu"])]
    pd.DataFrame(diag).to_csv(out / "full_cohort_diagnostics.csv", index=False)
    meta = {"stage": "full", "fit": fit_kw, "split_seeds": seeds,
            "hbayes_evaluate_reference_primary": ref,
            "primary": {"mean_d_lpd": float(w.d_lpd.mean()),
                        "d_lpd_ci95": np.quantile(boot[:, 0], [.025, .975]).tolist(),
                        "mean_d_class_loglik": float(w.d_cls.mean()),
                        "d_class_ci95": np.quantile(boot[:, 1], [.025, .975]).tolist()}}
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("smoke", "full"), required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    cfg = load_config()
    path = cfg.path("outputs") / "visit_panel.csv"
    if md5(path) != PANEL_MD5:
        print("panel MD5 mismatch", file=sys.stderr)
        return 2
    panel = pd.read_csv(path, low_memory=False)
    meta = (smoke if a.stage == "smoke" else full)(a, cfg, panel, out)
    meta.update({
        "script": "scripts/35_student_t_slope_model.py", "seed": SEED, "quick": a.quick,
        "panel_md5": PANEL_MD5, "nu_prior": "Gamma(2, rate 0.1) on grid [1, 300], 400 pts",
        "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "host": platform.node(), "runtime_s": round(time.time() - t0, 1)})
    (out / "results.json").write_text(json.dumps(meta, indent=2, default=float))
    log(f"done in {meta['runtime_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
