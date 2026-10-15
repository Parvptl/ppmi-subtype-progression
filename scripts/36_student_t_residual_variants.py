"""Slope tails vs residual tails: four-configuration growth models (smoke stage).

Design: docs/cluster_phase/08_STUDENT_T_RESIDUAL_SPEC.md. Configurations GG, TG,
GT, TT from `hbayes_tt` (GG = hbayes, TG = hbayes_t, exactly).

    python scripts/36_student_t_residual_variants.py --out <dir> --stage smoke [--jobs 12] [--quick]

smoke:
  C. real data, script 04's primary training split: 4 chains per configuration
     (+ log-uniform nu prior for TT); R-hat, ESS, nu posteriors, posterior
     dependence between slope and residual df. No held-out comparison.
  B. recovery on the primary split's real visit design: six truths (GG, slopes
     t2 / t3, residuals t4, both, patient-level noise heterogeneity), each
     fitted with all four configurations on the training patients and scored
     on the test patients' one-step-ahead transitions -- the operating
     characteristics of the proposed decision rule.
The full stage (real held-out comparison over seven splits) is not implemented
in this script until the design is approved.
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
from ppmi_subtype import hbayes_tt as TT  # noqa: E402
from ppmi_subtype import slope_clustering as SC  # noqa: E402
from ppmi_subtype import splitting  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

SEED = 42
PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
LONG = {"n_draws": 4000, "burn_in": 2000, "thin": 5}
CONTRASTS = [("TG", "GG"), ("GT", "GG"), ("TT", "GT"), ("TT", "TG")]
# truth: (slope nu, residual nu, patient-level noise heterogeneity)
SCENARIOS = {
    "GG": (np.inf, np.inf, False),
    "T2G": (2.0, np.inf, False),
    "T3G": (3.0, np.inf, False),
    "GT4": (np.inf, 4.0, False),
    "T3T4": (3.0, 4.0, False),
    "G_patnoise": (np.inf, np.inf, True),
    # residual tail matched to the real GT/TT fits (nu_resid ~ 2.2)
    "GT2.2": (np.inf, 2.2, False),
    "T2T2.2": (2.0, 2.2, False),
}
_S: dict = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def label(r):
    return np.where(r >= hbayes.TD_THRESHOLD, "TD",
                    np.where(r <= hbayes.PIGD_THRESHOLD, "PIGD", "Indeterminate"))


def class_logp(df):
    probs = np.clip(df[["TD", "Indeterminate", "PIGD"]].to_numpy(), 1e-6, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    j = df["observed"].map({"TD": 0, "Indeterminate": 1, "PIGD": 2}).to_numpy(int)
    return np.log(probs[np.arange(len(j)), j])


def contrasts(preds, n_boot, rng, pairs=None):
    """Paired per-transition differences A - B with a patient-cluster bootstrap."""
    p = preds.assign(cls=class_logp(preds))
    wide = p.pivot_table(index=["PATNO", "step"], columns="model",
                         values=["cls", "log_density"])
    out = {}
    pats = wide.index.get_level_values(0).to_numpy()
    uniq, inv = np.unique(pats, return_inverse=True)
    for a, b in (pairs or CONTRASTS):
        for metric in ("cls", "log_density"):
            d = (wide[(metric, a)] - wide[(metric, b)]).to_numpy()
            s = np.bincount(inv, weights=d)
            n = np.bincount(inv)
            bt = []
            for _ in range(n_boot):
                k = rng.integers(0, len(uniq), len(uniq))
                bt.append(s[k].sum() / n[k].sum())
            lo, hi = np.quantile(bt, [.025, .975])
            out[f"{a}-{b}:{metric}"] = (float(d.mean()), float(lo), float(hi))
    return out


def chain_rows(posts, label_):
    rows = []
    get = {"beta1": lambda p: p.beta[:, 1], "sigma": lambda p: np.sqrt(p.sigma2),
           "tau0": lambda p: np.sqrt(p.tau0_2), "tau1": lambda p: np.sqrt(p.tau1_2),
           "nu_slope": lambda p: p.nu, "nu_resid": lambda p: p.nu_e}
    for name, f in get.items():
        c = np.stack([f(p) for p in posts])
        if not np.isfinite(c).all():
            continue
        rows.append({**label_, "parameter": name, "mean": float(c.mean()),
                     "q2.5": float(np.quantile(c, .025)), "q50": float(np.median(c)),
                     "q97.5": float(np.quantile(c, .975)), "rhat": SC.split_rhat(c),
                     "ess_total": SC.ess(c), "ess_chain1": SC.ess(c[:1])})
    if posts[0].config == "TT":
        ls, le = np.log(np.concatenate([p.nu for p in posts])), \
            np.log(np.concatenate([p.nu_e for p in posts]))
        rows.append({**label_, "parameter": "corr(log nu_slope, log nu_resid)",
                     "mean": float(np.corrcoef(ls, le)[0, 1])})
    return rows


def _chain_task(args):
    config, seed, prior = args
    lp = HT.log_prior_nu_gamma if prior == "gamma" else HT.log_prior_nu_loguniform
    t0 = time.time()
    p = TT.fit(_S["data"], config, seed=seed, log_prior_nu=lp, **_S["fit_kw"])
    return config, seed, prior, p, time.time() - t0


def _recovery_task(args):
    scen, rep = args
    nu_s, nu_e, patnoise = SCENARIOS[scen]
    d, te, th, kw = _S["train"], _S["test"], _S["theta"], _S["fit_kw"]
    rng = np.random.default_rng([SEED, 36, list(SCENARIOS).index(scen), rep])
    # one cohort over train + test patients on their real visit times
    allp = pd.concat([d, te])
    pats = np.sort(allp.PATNO.unique())
    M = len(pats)
    pi = pd.Series(np.arange(M), index=pats)
    z = rng.standard_normal(M) if np.isinf(nu_s) else rng.standard_t(nu_s, M)
    tau1 = th["tau1_t"] if np.isfinite(nu_s) else th["tau1"]
    b1 = tau1 * z
    b0 = rng.normal(0, np.sqrt(th["tau0_2"]), M)
    n = len(allp)
    if patnoise:                                   # per-patient noise scale, t4 marginally
        kappa = rng.gamma(2.0, 0.5, M)
        e = rng.standard_normal(n) / np.sqrt(kappa[pi.loc[allp.PATNO].to_numpy()]) / np.sqrt(2.0)
        sig = th["sigma"]
    elif np.isfinite(nu_e):
        e, sig = rng.standard_t(nu_e, n), th["sigma_t"]
    else:
        e, sig = rng.standard_normal(n), th["sigma"]
    i = pi.loc[allp.PATNO].to_numpy()
    t = allp.years_from_baseline.to_numpy()
    y = th["beta0"] + b0[i] + (th["beta1"] + b1[i]) * t + sig * e
    sim = allp[["PATNO", "years_from_baseline"]].assign(ratio=y, subtype_name=label(y))
    train = sim[sim.PATNO.isin(set(d.PATNO))]
    test = sim[sim.PATNO.isin(set(te.PATNO))]
    gd = hbayes.build_growth_data(train)
    posts, out = {}, {"scenario": scen, "rep": rep}
    for c in TT.CONFIGS:
        posts[c] = TT.fit(gd, c, seed=SEED + 10 * rep, **kw)
    tt = posts["TT"]
    for c in ("TG", "TT"):
        out[f"{c}_nu_slope_q50"] = float(np.median(posts[c].nu))
        out[f"{c}_nu_slope_q2.5"] = float(np.quantile(posts[c].nu, .025))
        out[f"{c}_nu_slope_q97.5"] = float(np.quantile(posts[c].nu, .975))
    for c in ("GT", "TT"):
        out[f"{c}_nu_resid_q50"] = float(np.median(posts[c].nu_e))
        out[f"{c}_nu_resid_q2.5"] = float(np.quantile(posts[c].nu_e, .025))
        out[f"{c}_nu_resid_q97.5"] = float(np.quantile(posts[c].nu_e, .975))
    out["TT_corr_log_nu"] = float(np.corrcoef(np.log(tt.nu), np.log(tt.nu_e))[0, 1])
    out["TT_P_nu_slope_lt_10"] = float((tt.nu < 10).mean())
    out["TT_P_nu_resid_lt_10"] = float((tt.nu_e < 10).mean())
    out["TT_nu_slope_ess"] = SC.ess(tt.nu[None])
    out["TT_nu_resid_ess"] = SC.ess(tt.nu_e[None])
    out["TT_beta1_mean"] = float(tt.beta[:, 1].mean())
    out["TT_beta1_covered"] = bool(np.quantile(tt.beta[:, 1], .025) <= th["beta1"]
                                   <= np.quantile(tt.beta[:, 1], .975))
    preds = TT.evaluate_models(posts, test, max_draws=_S["max_draws"], seed=rep)
    out["n_transitions"] = int((preds.model == "GG").sum())
    for k, (m, lo, hi) in contrasts(preds, _S["n_boot"], rng).items():
        out[f"{k}:mean"], out[f"{k}:lo"], out[f"{k}:hi"] = m, lo, hi
    return out


def _init(state):
    _S.update(state)


def pool_map(fn, items, jobs, state):
    if jobs <= 1:
        _init(state)
        return [fn(a) for a in items]
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=(state,)) as ex:
        return list(ex.map(fn, items))


def smoke(a, cfg, panel, out):
    meta = {"stage": "smoke", "fit": LONG}
    man = splitting.split_patients(panel, cfg.splits["train"], cfg.seed)
    tr_ids = set(man.loc[man["split"] == "train", "PATNO"])
    te_ids = set(man.loc[man["split"] == "test", "PATNO"])
    train, test = panel[panel.PATNO.isin(tr_ids)], panel[panel.PATNO.isin(te_ids)]
    d_tr = hbayes.build_growth_data(train)
    fit_kw = {"n_draws": 400, "burn_in": 200, "thin": 1} if a.quick else LONG

    # ---- C. real-data sampler behaviour --------------------------------------
    n_ch = 2 if a.quick else 4
    tasks = [(c, SEED + k, "gamma") for c in TT.CONFIGS for k in range(n_ch)] \
        + [("TT", SEED + k, "loguniform") for k in range(n_ch)]
    chains = pool_map(_chain_task, tasks, a.jobs, {"data": d_tr, "fit_kw": fit_kw})
    rows = []
    for c, prior in [(c, "gamma") for c in TT.CONFIGS] + [("TT", "loguniform")]:
        cs = [x for x in chains if x[0] == c and x[2] == prior]
        rows += chain_rows([x[3] for x in cs], {"config": c, "nu_prior": prior,
                                                "fit_s": float(np.mean([x[4] for x in cs]))})
    diag = pd.DataFrame(rows)
    diag.to_csv(out / "real_sampler_diagnostics.csv", index=False)
    np.savez_compressed(out / "real_nu_traces.npz",
                        **{f"{x[0]}_{x[2]}_seed{x[1]}_{k}": getattr(x[3], k)
                           for x in chains for k in ("nu", "nu_e")
                           if np.isfinite(getattr(x[3], k)).all()})
    tt = [x[3] for x in chains if x[0] == "TT" and x[2] == "gamma"][0]
    meta["train_split"] = {"patients": d_tr.n_patients, "obs": d_tr.n_obs,
                           "test_patients": len(te_ids)}
    meta["TT_nu_autocorr_lag1_5_20"] = {
        k: [float(np.corrcoef(v[:-L], v[L:])[0, 1]) for L in (1, 5, 20)]
        for k, v in (("nu_slope", tt.nu), ("nu_resid", tt.nu_e))}
    log("real-data chains done")

    # ---- B. recovery: truths from the real fits ------------------------------
    def pm(c, f):
        return float(np.mean([f(x[3]) for x in chains if x[0] == c and x[2] == "gamma"]))
    theta = {"beta0": pm("GG", lambda p: p.beta[:, 0].mean()),
             "beta1": pm("GG", lambda p: p.beta[:, 1].mean()),
             "tau0_2": pm("GG", lambda p: p.tau0_2.mean()),
             "tau1": pm("GG", lambda p: np.sqrt(p.tau1_2).mean()),
             "tau1_t": pm("TT", lambda p: np.sqrt(p.tau1_2).mean()),
             "sigma": pm("GG", lambda p: np.sqrt(p.sigma2).mean()),
             "sigma_t": pm("TT", lambda p: np.sqrt(p.sigma2).mean())}
    meta["recovery_theta"] = theta
    reps = 1 if a.quick else 6
    tasks = [(s, r) for s in SCENARIOS for r in range(reps)]
    rec = pd.DataFrame(pool_map(_recovery_task, tasks, a.jobs,
                                {"train": train[np.isfinite(train.ratio)],
                                 "test": test[np.isfinite(test.ratio)],
                                 "theta": theta, "fit_kw": fit_kw,
                                 "max_draws": 100 if a.quick else 200,
                                 "n_boot": 100 if a.quick else 500}))
    rec.to_csv(out / "recovery.csv", index=False)
    log(f"recovery done: {len(rec)} replicates x 4 configurations")
    return meta



# ===========================================================================
#  FULL STAGE (frozen: docs/cluster_phase/08_STUDENT_T_RESIDUAL_SPEC.md section 7)
# ===========================================================================
SPLIT_SEEDS = (42, 0, 1, 2, 3, 4, 5)
FULL_FIT = {"n_draws": 4000, "burn_in": 2000, "thin": 10}
CHAIN_SEEDS = (42, 43, 44, 45)
PRIORS = ("gamma", "loguniform")
N_CAL = 20
CAL_SCENARIOS = ("C1", "C2", "C3", "P")
NU_THRESH = 10.0
RHAT_MAX, ESS_MIN = 1.01, 400


def _lp(prior):
    return HT.log_prior_nu_gamma if prior == "gamma" else HT.log_prior_nu_loguniform


def _split_task(seed):
    panel, cfg = _S["panel"], _S["cfg"]
    man = splitting.split_patients(panel, cfg.splits["train"], seed)
    tr = panel[panel.PATNO.isin(set(man.loc[man["split"] == "train", "PATNO"]))]
    te = panel[panel.PATNO.isin(set(man.loc[man["split"] == "test", "PATNO"]))]
    d = hbayes.build_growth_data(tr)
    t0 = time.time()
    posts = {c: TT.fit(d, c, seed=seed, **_S["split_fit"]) for c in TT.CONFIGS}
    t_fit = time.time() - t0
    preds = TT.evaluate_models(posts, te, max_draws=400, seed=seed, n_inner=20, n_omega=8)
    preds["split_seed"] = seed
    tt = posts["TT"]
    info = {"split_seed": seed, "train_patients": d.n_patients, "train_obs": d.n_obs,
            "test_patients": int(te.PATNO.nunique()),
            "n_transitions": int((preds.model == "GG").sum()), "fit_s": t_fit,
            "TT_nu_slope_q50": float(np.median(tt.nu)),
            "TT_nu_slope_q97.5": float(np.quantile(tt.nu, .975)),
            "TT_nu_slope_ess": SC.ess(tt.nu[None]),
            "TT_nu_resid_q50": float(np.median(tt.nu_e)),
            "GT_nu_resid_q50": float(np.median(posts["GT"].nu_e)),
            "GT_nu_resid_q97.5": float(np.quantile(posts["GT"].nu_e, .975)),
            "TG_nu_slope_q50": float(np.median(posts["TG"].nu))}
    ref = None
    if seed == 42:
        ref, _ = hbayes.evaluate(posts["GG"], te, max_draws=400, seed=seed)
    return info, preds, ref


def _cohort_chain_task(args):
    config, prior, seed = args
    keep = config == "TT" and prior == "gamma" and seed == 42
    t0 = time.time()
    p = TT.fit(_S["data"], config, seed=seed, log_prior_nu=_lp(prior),
               keep_random_effects=keep, **_S["full_fit"])
    bmean = p.random_effects.mean(axis=0) if keep else None
    p.random_effects = None
    return config, prior, seed, p, bmean, time.time() - t0


def _simulate_cal(scen, rep):
    d, th = _S["data"], _S["theta"]
    rng = np.random.default_rng([SEED, 36, CAL_SCENARIOS.index(scen), rep])
    M, i, t = d.n_patients, d.patient_index, d.t
    b0 = rng.normal(0, th["tau0"], M)
    if scen == "P":
        b1 = th["tau1_TT"] * rng.standard_t(2.0, M)
    else:
        b1 = rng.normal(0, th["tau1"], M)
    y = th["beta0"] + b0[i] + (th["beta1"] + b1[i]) * t
    if scen == "C3":
        delta = rng.normal(0, th["tau1"], M)
        c = rng.uniform(_S["t_first"], _S["t_last"])
        y = y + delta[i] * np.maximum(0.0, t - c[i])
    if scen == "C2":
        e = np.empty(d.n_obs)
        donors = _S["donors"]                      # n_visits -> list of patient indices
        for k in range(M):
            rows = _S["rows"][k]
            pool = donors[len(rows)]
            j = pool[rng.integers(len(pool))] if len(pool) > 1 else k
            e[rows] = _S["real_resid"][_S["rows"][j]]
    else:
        e = th["sigma"] * rng.standard_t(th["nu_e"], d.n_obs)
    return hbayes.GrowthData(y=y + e, t=t, patient_index=i, patients=d.patients,
                             n_patients=M, n_obs=d.n_obs)


def _cal_task(args):
    scen, rep = args
    sim = _simulate_cal(scen, rep)
    out = {"scenario": scen, "rep": rep}
    t0 = time.time()
    for prior in PRIORS:
        posts = [TT.fit(sim, "TT", seed=1000 + 10 * rep + k, log_prior_nu=_lp(prior),
                        **_S["full_fit"]) for k in range(4)]
        nu = np.stack([p.nu for p in posts])
        out[f"{prior}_nu_slope_q50"] = float(np.median(nu))
        out[f"{prior}_nu_slope_q97.5"] = float(np.quantile(nu, .975))
        out[f"{prior}_nu_slope_rhat"] = SC.split_rhat(nu)
        out[f"{prior}_nu_slope_ess"] = SC.ess(nu)
        out[f"{prior}_nu_resid_q50"] = float(np.median([p.nu_e for p in posts]))
    out["flag"] = bool(all(out[f"{p}_nu_slope_q97.5"] < NU_THRESH for p in PRIORS))
    out["fit_s"] = time.time() - t0
    return out


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def full(a, cfg, panel, out):
    frozen = {
        "spec": "docs/cluster_phase/08_STUDENT_T_RESIDUAL_SPEC.md (section 7)",
        "sha256": {f: sha256(ROOT / f) for f in (
            "docs/cluster_phase/08_STUDENT_T_RESIDUAL_SPEC.md", "src/ppmi_subtype/hbayes_tt.py",
            "src/ppmi_subtype/hbayes_t.py", "src/ppmi_subtype/hbayes.py",
            "scripts/36_student_t_residual_variants.py")},
        "split_seeds": SPLIT_SEEDS, "split_fit": LONG, "full_fit": FULL_FIT,
        "chain_seeds": CHAIN_SEEDS, "priors": PRIORS,
        "nu_grid": "400 log-spaced points on [1, 300]",
        "eval": {"max_draws": 400, "n_inner": 20, "n_omega": 8, "rng_seed": "split seed"},
        "bootstrap": {"n": 2000, "seed": [SEED, 36, 7]},
        "calibration": {"scenarios": CAL_SCENARIOS, "reps": N_CAL,
                        "rng": "[42, 36, scenario index, rep]",
                        "fit_seeds": "1000 + 10*rep + chain"},
        "thresholds": {"nu_slope_q97.5_lt": NU_THRESH, "rhat_lt": RHAT_MAX,
                       "ess_ge": ESS_MIN, "splits_needed": "5/7", "cal_fp_max": "1/20"},
        "quick": a.quick,
    }
    (out / "frozen_config.json").write_text(json.dumps(frozen, indent=2))
    log("frozen config written")
    n_cal = 2 if a.quick else N_CAL
    split_seeds = SPLIT_SEEDS[:2] if a.quick else SPLIT_SEEDS
    split_fit, full_fit = (({"n_draws": 200, "burn_in": 100, "thin": 1},) * 2 if a.quick
                           else (LONG, FULL_FIT))

    # ---- full-cohort chains + split predictions (one pool) ------------------
    d_all = hbayes.build_growth_data(panel)
    tasks_c = [(c, pr, s) for c in ("GT", "TT") for pr in PRIORS for s in CHAIN_SEEDS]
    state = {"panel": panel, "cfg": cfg, "data": d_all, "split_fit": split_fit,
             "full_fit": full_fit}
    if a.quick:
        tasks_c = [x for x in tasks_c if x[2] in (42, 43)]
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(state,)) as ex:
        fut_c = [ex.submit(_cohort_chain_task, x) for x in tasks_c]
        fut_s = [ex.submit(_split_task, s) for s in split_seeds]
        chains = [f.result() for f in fut_c]
        log("full-cohort chains done")
        splits = [f.result() for f in fut_s]
    log("split predictions done")

    rows = []
    for c in ("GT", "TT"):
        for pr in PRIORS:
            cs = [x for x in chains if x[0] == c and x[1] == pr]
            rows += chain_rows([x[3] for x in cs], {"config": c, "nu_prior": pr,
                                                    "fit_s": float(np.mean([x[5] for x in cs]))})
    cohort = pd.DataFrame(rows)
    cohort.to_csv(out / "full_cohort_diagnostics.csv", index=False)
    np.savez_compressed(out / "full_cohort_nu_traces.npz",
                        **{f"{x[0]}_{x[1]}_seed{x[2]}_{k}": getattr(x[3], k)
                           for x in chains for k in ("nu", "nu_e")
                           if np.isfinite(getattr(x[3], k)).all()})

    # ---- held-out results --------------------------------------------------------
    info = pd.DataFrame([s[0] for s in splits])
    info.to_csv(out / "split_info.csv", index=False)
    preds = pd.concat([s[1] for s in splits])
    preds.to_csv(out / "heldout_predictions.csv.gz", index=False)
    ref = [s[2] for s in splits if s[2] is not None][0]
    met = []
    for (seed, m), g in preds.groupby(["split_seed", "model"]):
        met.append({"split_seed": seed, "model": m, **HT.metrics(g)})
    met = pd.DataFrame(met)
    met.to_csv(out / "heldout_metrics_by_split.csv", index=False)
    # paired per-split mean differences
    pdiff = []
    for seed, g in preds.groupby("split_seed"):
        g = g.assign(cls=class_logp(g))
        w = g.pivot_table(index=["PATNO", "step"], columns="model", values=["cls", "log_density"])
        assert w.notna().all().all(), "unmatched transitions"
        for a_, b_ in CONTRASTS + [("GT", "GG"), ("TT", "GG")]:
            for metric in ("cls", "log_density"):
                pdiff.append({"split_seed": seed, "contrast": f"{a_}-{b_}", "metric": metric,
                              "mean_diff": float((w[(metric, a_)] - w[(metric, b_)]).mean()),
                              "n_transitions": len(w)})
    pdiff = pd.DataFrame(pdiff).drop_duplicates(["split_seed", "contrast", "metric"])
    pdiff.to_csv(out / "heldout_paired_differences_by_split.csv", index=False)
    pairs = list(dict.fromkeys(CONTRASTS + [("GT", "GG"), ("TT", "GG")]))
    boot = contrasts(preds[preds.split_seed == 42], 200 if a.quick else 2000,
                     np.random.default_rng([SEED, 36, 7]), pairs)
    boot_df = pd.DataFrame([{"contrast": k.split(":")[0], "metric": k.split(":")[1],
                             "mean": v[0], "ci_lo": v[1], "ci_hi": v[2]}
                            for k, v in boot.items()])
    boot_df.to_csv(out / "primary_split_bootstrap.csv", index=False)

    # ---- calibration -----------------------------------------------------------------
    gt = [x[3] for x in chains if x[0] == "GT" and x[1] == "gamma"]
    tt = [x[3] for x in chains if x[0] == "TT" and x[1] == "gamma"]
    bmean = [x[4] for x in chains if x[4] is not None][0]
    tt42 = [x[3] for x in chains if x[0] == "TT" and x[1] == "gamma" and x[2] == 42][0]
    theta = {"beta0": float(np.mean([p.beta[:, 0].mean() for p in gt])),
             "beta1": float(np.mean([p.beta[:, 1].mean() for p in gt])),
             "tau0": float(np.mean([np.sqrt(p.tau0_2).mean() for p in gt])),
             "tau1": float(np.mean([np.sqrt(p.tau1_2).mean() for p in gt])),
             "sigma": float(np.mean([np.sqrt(p.sigma2).mean() for p in gt])),
             "nu_e": float(np.median(np.concatenate([p.nu_e for p in gt]))),
             "tau1_TT": float(np.mean([np.sqrt(p.tau1_2).mean() for p in tt]))}
    X = d_all.design()
    real_resid = d_all.y - X @ tt42.beta.mean(axis=0) \
        - np.einsum("ij,ij->i", X, bmean[d_all.patient_index])
    real_resid = real_resid - real_resid.mean()
    rows_by_pat = [np.flatnonzero(d_all.patient_index == k) for k in range(d_all.n_patients)]
    donors = {}
    for k, r in enumerate(rows_by_pat):
        donors.setdefault(len(r), []).append(k)
    t_first = np.array([d_all.t[r].min() for r in rows_by_pat])
    t_last = np.array([d_all.t[r].max() for r in rows_by_pat])
    cal_state = {"data": d_all, "theta": theta, "real_resid": real_resid, "rows": rows_by_pat,
                 "donors": donors, "t_first": t_first, "t_last": t_last,
                 "full_fit": full_fit}
    cal = pd.DataFrame(pool_map(_cal_task, [(s, r) for s in CAL_SCENARIOS for r in range(n_cal)],
                                a.jobs, cal_state))
    cal.to_csv(out / "calibration.csv", index=False)
    log("calibration done")
    singleton_counts = int(sum(len(v) == 1 for v in donors.values()))

    # ---- decisions (frozen rule) ------------------------------------------------------
    def bs(con, metric):
        r = boot_df[(boot_df.contrast == con) & (boot_df.metric == metric)].iloc[0]
        return float(r["mean"]), float(r.ci_lo), float(r.ci_hi)

    def n_pos(con, metric):
        g = pdiff[(pdiff.contrast == con) & (pdiff.metric == metric)]
        return int((g.mean_diff > 0).sum()), len(g)

    need = 5 if not a.quick else 1
    q1_parts = {}
    for con in ("GT-GG", "TT-TG"):
        m, lo, hi = bs(con, "log_density")
        k, n = n_pos(con, "log_density")
        cm, clo, chi = bs(con, "cls")
        q1_parts[con] = {"lpd_mean": m, "lpd_ci": [lo, hi], "lpd_splits_positive": f"{k}/{n}",
                         "cls_mean": cm, "cls_ci": [clo, chi],
                         "pass": bool(lo > 0 and k >= need and not chi < 0)}
    q1 = "SUPPORTED" if all(v["pass"] for v in q1_parts.values()) else "NOT SUPPORTED"

    conv = {}
    a_ok = {}
    for pr in PRIORS:
        r = cohort[(cohort.config == "TT") & (cohort.nu_prior == pr) &
                   (cohort.parameter == "nu_slope")].iloc[0]
        conv[pr] = {"rhat": float(r.rhat), "ess": float(r.ess_total),
                    "pass": bool(r.rhat < RHAT_MAX and r.ess_total >= ESS_MIN)}
        a_ok[pr] = {"q50": float(r.q50), "q97.5": float(r["q97.5"]),
                    "pass": bool(r["q97.5"] < NU_THRESH)}
    b_k = int((info["TT_nu_slope_q97.5"] < NU_THRESH).sum())
    c_res = {}
    for scen in CAL_SCENARIOS:
        g = cal[cal.scenario == scen]
        k = int(g.flag.sum())
        c_res[scen] = {"flags": k, "n": len(g), "rate": k / len(g),
                       "cp95": clopper_pearson(k, len(g)),
                       "pass": bool(k <= 1) if scen != "P" else None}
    conv_ok = all(v["pass"] for v in conv.values())
    a_pass = all(v["pass"] for v in a_ok.values())
    b_pass = b_k >= need
    c_pass = all(c_res[s]["pass"] for s in ("C1", "C2", "C3"))
    if not conv_ok:
        q2 = "INCONCLUSIVE (convergence)"
    elif not (a_pass and b_pass):
        q2 = "NOT SUPPORTED: no evidence of extreme slope tails"
    elif not c_pass:
        q2 = "INCONCLUSIVE (calibration)"
    else:
        q2 = "SUPPORTED"

    q3 = {}
    for mdl in ("TG", "GT", "TT"):
        m, lo, hi = bs(f"{mdl}-GG", "cls")
        k, n = n_pos(f"{mdl}-GG", "cls")
        q3[mdl] = {"cls_mean": m, "cls_ci": [lo, hi], "splits_positive": f"{k}/{n}",
                   "better_than_GG": bool(lo > 0 and k >= need)}

    return {"stage": "full", "frozen": frozen, "calibration_theta": theta,
            "c2_visit_counts_with_single_patient": singleton_counts,
            "hbayes_evaluate_reference_primary_GG": ref,
            "Q1_residual_tails": {"verdict": q1, **q1_parts},
            "Q2_slope_tails": {"verdict": q2, "convergence": conv, "a_full_cohort": a_ok,
                               "b_splits_q97.5_lt_10": f"{b_k}/{len(info)}",
                               "c_calibration": c_res},
            "Q3_predictive_value": q3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("smoke", "full"), required=True)
    ap.add_argument("--jobs", type=int, default=12)
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
        "script": "scripts/36_student_t_residual_variants.py", "seed": SEED, "quick": a.quick,
        "panel_md5": PANEL_MD5, "nu_prior": "Gamma(2, rate 0.1) on log grid [1, 300], 400 pts",
        "scenarios": {k: [str(x) for x in v] for k, v in SCENARIOS.items()},
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
