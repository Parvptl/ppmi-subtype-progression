"""Q2 reassessment: patient-slope tails with the collapsed slice sampler.

Frozen design: docs/cluster_phase/09_Q2_REASSESSMENT_SPEC.md. Reassesses Q2
only; the original full run (experiments/20261015T190228Z_student_t_four_model_full)
and its verdicts are read, never modified.

    python scripts/38_q2_reassessment.py --out <dir> [--jobs 16] [--smoke]

--smoke: short chains and 2 replicates per scenario, to check the code path
(no verdict is meaningful).
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
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import hbayes  # noqa: E402
from ppmi_subtype import hbayes_t as HT  # noqa: E402
from ppmi_subtype import hbayes_tt_nu as NU  # noqa: E402
from ppmi_subtype import slope_clustering as SC  # noqa: E402
from ppmi_subtype import splitting  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

SEED = 42
PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
ORIGINAL_RUN = ROOT / "experiments" / "20261015T190228Z_student_t_four_model_full"
SPEC = ROOT / "docs" / "cluster_phase" / "09_Q2_REASSESSMENT_SPEC.md"
PRIORS = {"gamma2_true": NU.log_prior_nu_gamma2_true,          # required, primary
          "loguniform": HT.log_prior_nu_loguniform,            # required
          "exp_like_continuity": HT.log_prior_nu_gamma}        # sensitivity only
REQUIRED = ("gamma2_true", "loguniform")
PRIMARY = "gamma2_true"
SPLIT_SEEDS = (42, 0, 1, 2, 3, 4, 5)
CHAIN_SEEDS = (42, 43, 44, 45)
COHORT_FIT = {"n_draws": 4000, "burn_in": 2000, "thin": 10}
SPLIT_FIT = {"n_draws": 4000, "burn_in": 2000, "thin": 5}
CAL_SCENARIOS = ("C1", "C2", "C3", "P")
N_CAL = 60
NU_THRESH, RHAT_MAX, ESS_MIN, FP_MAX = 10.0, 1.01, 400, 3
SAMPLER = {"nu_update": "collapsed_slice", "nu_init": 10.0}
_S: dict = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def digest(path, kind="sha256"):
    h = hashlib.new(kind)
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def q975_mcse_interval(chains, p=0.975):
    """MC interval for the pooled p-quantile via the ESS of 1{nu <= q}."""
    c = np.asarray(chains, float)
    q = np.quantile(c, p)
    ind = (c <= q).astype(float)
    e = SC.ess(ind) if ind.std() > 0 else float(c.size)
    se = np.sqrt(p * (1 - p) / max(e, 1.0))
    lo, hi = np.quantile(c, [max(p - 2 * se, 0.0), min(p + 2 * se, 1.0)])
    return float(q), float(lo), float(hi), float(e)


def summarise(chains_by_param):
    rows = []
    for name, c in chains_by_param.items():
        c = np.stack(c)
        if not np.isfinite(c).all():
            continue
        rows.append({"parameter": name, "mean": float(c.mean()),
                     "q2.5": float(np.quantile(c, .025)), "q50": float(np.median(c)),
                     "q97.5": float(np.quantile(c, .975)), "rhat": SC.split_rhat(c),
                     "ess": SC.ess(c)})
    return rows


def params(p):
    return {"nu_slope": p.nu, "nu_resid": p.nu_e, "beta1": p.beta[:, 1],
            "tau1": np.sqrt(p.tau1_2), "sigma": np.sqrt(p.sigma2), "tau0": np.sqrt(p.tau0_2)}


# ---------------------------------------------------------------------------
def _cohort_task(args):
    prior, seed = args
    keep = prior == PRIMARY and seed == 42
    t0 = time.time()
    p = NU.fit(_S["data"], "TT", seed=seed, log_prior_nu=PRIORS[prior],
               keep_random_effects=keep, **SAMPLER, **_S["cohort_fit"])
    bmean = p.random_effects.mean(axis=0) if keep else None
    p.random_effects = None
    return prior, seed, p, bmean, time.time() - t0


def _split_task(args):
    seed, prior = args
    panel, cfg = _S["panel"], _S["cfg"]
    man = splitting.split_patients(panel, cfg.splits["train"], seed)
    tr = panel[panel.PATNO.isin(set(man.loc[man["split"] == "train", "PATNO"]))]
    d = hbayes.build_growth_data(tr)
    t0 = time.time()
    p = NU.fit(d, "TT", seed=seed, log_prior_nu=PRIORS[prior], **SAMPLER, **_S["split_fit"])
    q, lo, hi, e_ind = q975_mcse_interval(p.nu[None])
    return {"split_seed": seed, "prior": prior, "train_patients": d.n_patients,
            "nu_slope_q2.5": float(np.quantile(p.nu, .025)), "nu_slope_q50": float(np.median(p.nu)),
            "nu_slope_q97.5": q, "q97.5_mc_lo": lo, "q97.5_mc_hi": hi,
            "nu_slope_ess": SC.ess(p.nu[None]), "nu_resid_q50": float(np.median(p.nu_e)),
            "fit_s": time.time() - t0}


def _simulate(scen, rep):
    d, th = _S["data"], _S["theta"]
    rng = np.random.default_rng([SEED, 36, CAL_SCENARIOS.index(scen), rep])
    M, i, t = d.n_patients, d.patient_index, d.t
    b0 = rng.normal(0, th["tau0"], M)
    b1 = th["tau1_TT"] * rng.standard_t(2.0, M) if scen == "P" else rng.normal(0, th["tau1"], M)
    y = th["beta0"] + b0[i] + (th["beta1"] + b1[i]) * t
    if scen == "C3":
        delta = rng.normal(0, th["tau1"], M)
        c = rng.uniform(_S["t_first"], _S["t_last"])
        y = y + delta[i] * np.maximum(0.0, t - c[i])
    if scen == "C2":
        e = np.empty(d.n_obs)
        for k in range(M):
            rows = _S["rows"][k]
            pool = _S["donors"][len(rows)]
            j = pool[rng.integers(len(pool))] if len(pool) > 1 else k
            e[rows] = _S["real_resid"][_S["rows"][j]]
    else:
        e = th["sigma"] * rng.standard_t(th["nu_e"], d.n_obs)
    return hbayes.GrowthData(y=y + e, t=t, patient_index=i, patients=d.patients,
                             n_patients=M, n_obs=d.n_obs)


def _cal_task(args):
    scen, rep = args
    sim = _simulate(scen, rep)
    out = {"scenario": scen, "rep": rep}
    t0 = time.time()
    for prior in REQUIRED:
        posts = [NU.fit(sim, "TT", seed=1000 + 10 * rep + k, log_prior_nu=PRIORS[prior],
                        **SAMPLER, **_S["cohort_fit"]) for k in range(4)]
        nu = np.stack([p.nu for p in posts])
        q, lo, hi, e_ind = q975_mcse_interval(nu)
        out.update({f"{prior}_nu_slope_q50": float(np.median(nu)),
                    f"{prior}_nu_slope_q97.5": q, f"{prior}_q97.5_mc_lo": lo,
                    f"{prior}_q97.5_mc_hi": hi, f"{prior}_nu_slope_rhat": SC.split_rhat(nu),
                    f"{prior}_nu_slope_ess": SC.ess(nu),
                    f"{prior}_nu_resid_q50": float(np.median([p.nu_e for p in posts]))})
    out["flag"] = bool(all(out[f"{p}_nu_slope_q97.5"] < NU_THRESH for p in REQUIRED))
    out["borderline"] = bool(any(out[f"{p}_q97.5_mc_lo"] < NU_THRESH <= out[f"{p}_q97.5_mc_hi"]
                                 for p in REQUIRED))
    out["fit_s"] = time.time() - t0
    return out


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi), float(stats.beta.ppf(1 - alpha, k + 1, n - k)) if k < n else 1.0


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    cfg = load_config()
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    if digest(panel_path, "md5") != PANEL_MD5:
        print("panel MD5 mismatch", file=sys.stderr)
        return 2
    orig = json.loads((ORIGINAL_RUN / "results.json").read_text())
    theta = orig["calibration_theta"]

    short = {"n_draws": 200, "burn_in": 100, "thin": 1}
    cohort_fit = short if a.smoke else COHORT_FIT
    split_fit = short if a.smoke else SPLIT_FIT
    n_cal = 2 if a.smoke else N_CAL
    config = {
        "spec": str(SPEC.relative_to(ROOT)),
        "sha256": {str(Path(f)): digest(ROOT / f) for f in (
            "docs/cluster_phase/09_Q2_REASSESSMENT_SPEC.md",
            "docs/cluster_phase/_frozen/08_STUDENT_T_RESIDUAL_SPEC.as_frozen_20261016.md",
            "src/ppmi_subtype/hbayes_tt_nu.py", "src/ppmi_subtype/hbayes_tt.py",
            "src/ppmi_subtype/hbayes_t.py", "src/ppmi_subtype/hbayes.py",
            "src/ppmi_subtype/slope_clustering.py", "scripts/38_q2_reassessment.py")},
        "original_run_results_sha256": digest(ORIGINAL_RUN / "results.json"),
        "sampler": SAMPLER, "priors": list(PRIORS), "required_priors": REQUIRED,
        "primary_prior": PRIMARY, "cohort_fit": cohort_fit, "split_fit": split_fit,
        "chain_seeds": CHAIN_SEEDS, "split_seeds": SPLIT_SEEDS,
        "calibration": {"scenarios": CAL_SCENARIOS, "reps": n_cal,
                        "data_rng": "[42, 36, scenario index, rep]",
                        "fit_seeds": "1000 + 10*rep + chain", "theta": theta},
        "thresholds": {"nu_q97.5_lt": NU_THRESH, "rhat_lt": RHAT_MAX, "ess_ge": ESS_MIN,
                       "splits_needed": "5/7", "cal_fp_max": f"{FP_MAX}/{N_CAL}"},
        "smoke": a.smoke,
    }
    (out / "reassessment_config.json").write_text(json.dumps(config, indent=2))
    log("frozen reassessment config written")

    data = hbayes.build_growth_data(panel := pd.read_csv(panel_path, low_memory=False))
    state = {"data": data, "panel": panel, "cfg": cfg, "cohort_fit": cohort_fit,
             "split_fit": split_fit}
    seeds = CHAIN_SEEDS[:2] if a.smoke else CHAIN_SEEDS
    with ProcessPoolExecutor(a.jobs, initializer=_S.update, initargs=(state,)) as ex:
        fc = [ex.submit(_cohort_task, (p, s)) for p in PRIORS for s in seeds]
        fs = [ex.submit(_split_task, (s, p)) for s in SPLIT_SEEDS for p in PRIORS]
        cohort = [f.result() for f in fc]
        log("A. full-cohort chains done")
        splits = pd.DataFrame([f.result() for f in fs])
    log("B. per-split fits done")

    rows = []
    for prior in PRIORS:
        cs = [c for c in cohort if c[0] == prior]
        by = {k: [params(c[2])[k] for c in cs] for k in params(cs[0][2])}
        for r in summarise(by):
            rows.append({"prior": prior, **r, "fit_s_mean": float(np.mean([c[4] for c in cs]))})
        q, lo, hi, e_ind = q975_mcse_interval(np.stack(by["nu_slope"]))
        rows.append({"prior": prior, "parameter": "nu_slope q97.5 MC interval",
                     "q97.5": q, "q2.5": lo, "mean": hi, "ess": e_ind})
        for c in cs:
            rows.append({"prior": prior, "parameter": f"nu_slope chain seed {c[1]}",
                         "q50": float(np.median(c[2].nu)),
                         "q97.5": float(np.quantile(c[2].nu, .975)),
                         "ess": SC.ess(c[2].nu[None])})
    cohort_tab = pd.DataFrame(rows)
    cohort_tab.to_csv(out / "A_full_cohort.csv", index=False)
    np.savez_compressed(out / "A_full_cohort_nu_traces.npz",
                        **{f"{c[0]}_seed{c[1]}_{k}": getattr(c[2], k) for c in cohort
                           for k in ("nu", "nu_e")})
    splits.to_csv(out / "B_per_split.csv", index=False)

    # ---- C. calibration -------------------------------------------------------------
    bmean = [c[3] for c in cohort if c[3] is not None][0]
    p42 = [c[2] for c in cohort if c[0] == PRIMARY and c[1] == 42][0]
    X = data.design()
    resid = data.y - X @ p42.beta.mean(axis=0) - np.einsum("ij,ij->i", X, bmean[data.patient_index])
    resid -= resid.mean()
    rows_by = [np.flatnonzero(data.patient_index == k) for k in range(data.n_patients)]
    donors = {}
    for k, r in enumerate(rows_by):
        donors.setdefault(len(r), []).append(k)
    cal_state = {"data": data, "theta": theta, "real_resid": resid, "rows": rows_by,
                 "donors": donors, "cohort_fit": cohort_fit,
                 "t_first": np.array([data.t[r].min() for r in rows_by]),
                 "t_last": np.array([data.t[r].max() for r in rows_by])}
    with ProcessPoolExecutor(a.jobs, initializer=_S.update, initargs=(cal_state,)) as ex:
        cal = pd.DataFrame(list(ex.map(_cal_task, [(s, r) for s in CAL_SCENARIOS
                                                   for r in range(n_cal)])))
    cal.to_csv(out / "C_calibration.csv", index=False)
    log("C. calibration done")

    # ---- decision (frozen rule) -------------------------------------------------------
    def a_row(prior, par="nu_slope"):
        return cohort_tab[(cohort_tab.prior == prior) & (cohort_tab.parameter == par)].iloc[0]

    conv = {p: {"rhat": float(a_row(p).rhat), "ess": float(a_row(p).ess),
                "pass": bool(a_row(p).rhat < RHAT_MAX and a_row(p).ess >= ESS_MIN)}
            for p in REQUIRED}
    a_res = {p: {"q50": float(a_row(p).q50), "q2.5": float(a_row(p)["q2.5"]),
                 "q97.5": float(a_row(p)["q97.5"]), "pass": bool(a_row(p)["q97.5"] < NU_THRESH)}
             for p in REQUIRED}
    prim = splits[splits.prior == PRIMARY]
    b_k = int((prim["nu_slope_q97.5"] < NU_THRESH).sum())
    c_res = {}
    for s in CAL_SCENARIOS:
        g = cal[cal.scenario == s]
        k = int(g.flag.sum())
        lo, hi, ub1 = clopper_pearson(k, len(g))
        c_res[s] = {"flags": k, "n": len(g), "rate": k / len(g), "cp95_two_sided": [lo, hi],
                    "cp95_one_sided_upper": ub1, "borderline": int(g.borderline.sum()),
                    "max_rhat_required_priors": float(max(g[f"{p}_nu_slope_rhat"].max()
                                                          for p in REQUIRED)),
                    "min_ess_required_priors": float(min(g[f"{p}_nu_slope_ess"].min()
                                                         for p in REQUIRED)),
                    "pass": (k <= FP_MAX) if s != "P" else None}
    conv_ok = all(v["pass"] for v in conv.values())
    a_ok = all(v["pass"] for v in a_res.values())
    b_ok = b_k >= 5
    c_ok = all(c_res[s]["pass"] for s in ("C1", "C2", "C3"))
    if not conv_ok:
        verdict = "INCONCLUSIVE (convergence)"
    elif not (a_ok and b_ok):
        verdict = "NOT SUPPORTED: no evidence of extreme slope tails"
    elif not c_ok:
        verdict = "INCONCLUSIVE (calibration)"
    else:
        verdict = "SUPPORTED, within the stated calibration precision"
    continuity = {p: {"q50": float(a_row(p).q50), "q97.5": float(a_row(p)["q97.5"]),
                      "rhat": float(a_row(p).rhat), "ess": float(a_row(p).ess)}
                  for p in PRIORS}
    meta = {"Q2_reassessment_verdict": verdict, "smoke": a.smoke,
            "conv": conv, "a": a_res, "b": f"{b_k}/{len(prim)}", "c": c_res,
            "full_cohort_nu_slope_by_prior": continuity,
            "per_split_q97.5_by_prior": {p: splits[splits.prior == p]
                                         .set_index("split_seed")["nu_slope_q97.5"].round(3)
                                         .to_dict() for p in PRIORS},
            "original_Q2_verdict_unchanged": orig["Q2_slope_tails"]["verdict"],
            "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__},
            "host": platform.node(), "runtime_s": round(time.time() - t0, 1)}
    (out / "results.json").write_text(json.dumps(meta, indent=2, default=float))
    log(f"done in {meta['runtime_s']} s: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
