"""Convergence pilot for the slope degrees of freedom nu_s (TT model, full cohort).

Compares three samplers for the same posterior (see src/ppmi_subtype/hbayes_tt_nu.py):
  lambda_grid      original: nu ~ p(nu | lambda) on the grid (as in the frozen run)
  collapsed_grid   block (nu, lambda) | b1, tau1; nu exactly on the same grid
  collapsed_slice  same block; nu by slice sampling on log nu

under the frozen priors (grid "gamma" weights = truncated Exponential(0.1);
log-uniform) and, for the collapsed samplers, a Jacobian-corrected true
Gamma(2, 0.1). 4 chains each, over-dispersed starts nu_s in {1.2, 4, 30, 250},
seeds 42-45, settings of the frozen full-cohort fits (4000 draws, burn-in 2000,
thin 10). Thresholds unchanged: split-R-hat < 1.01, ESS >= 400.

    python scripts/37_nu_slope_sampler_pilot.py --out <dir> [--jobs 16] [--quick]
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
from ppmi_subtype import hbayes_tt_nu as NU  # noqa: E402
from ppmi_subtype import slope_clustering as SC  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
FIT = {"n_draws": 4000, "burn_in": 2000, "thin": 10}
SEEDS = (42, 43, 44, 45)
INITS = (1.2, 4.0, 30.0, 250.0)
PRIORS = {"gamma_as_frozen": HT.log_prior_nu_gamma,
          "loguniform": HT.log_prior_nu_loguniform,
          "gamma2_true": NU.log_prior_nu_gamma2_true}
GROUPS = [("lambda_grid", "gamma_as_frozen"), ("lambda_grid", "loguniform")] + \
    [(m, p) for m in ("collapsed_grid", "collapsed_slice") for p in PRIORS]
RHAT_MAX, ESS_MIN = 1.01, 400
FULL_RUN = ROOT / "experiments" / "20261015T190228Z_student_t_four_model_full"
_S: dict = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(path, kind="md5"):
    h = hashlib.new(kind)
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _task(args):
    method, prior, k = args
    t0 = time.time()
    p = NU.fit(_S["data"], "TT", nu_update=method, seed=SEEDS[k], nu_init=INITS[k],
               log_prior_nu=PRIORS[prior], keep_trace=True, **_S["fit"])
    return method, prior, k, p, time.time() - t0


def diag(posts):
    rows = {}
    get = {"nu_slope": lambda p: p.nu, "nu_resid": lambda p: p.nu_e,
           "beta1": lambda p: p.beta[:, 1], "tau1": lambda p: np.sqrt(p.tau1_2),
           "sigma": lambda p: np.sqrt(p.sigma2)}
    for name, f in get.items():
        c = np.stack([f(p) for p in posts])
        rows[name] = {"rhat": SC.split_rhat(c), "ess": SC.ess(c),
                      "q2.5": float(np.quantile(c, .025)), "q50": float(np.median(c)),
                      "q97.5": float(np.quantile(c, .975)), "mean": float(c.mean())}
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    cfg = load_config()
    path = cfg.path("outputs") / "visit_panel.csv"
    if sha(path) != PANEL_MD5:
        print("panel MD5 mismatch", file=sys.stderr)
        return 2
    fit = {"n_draws": 300, "burn_in": 100, "thin": 1} if a.quick else FIT
    frozen = {"sha256": {f: sha(ROOT / f, "sha256") for f in (
        "src/ppmi_subtype/hbayes_tt_nu.py", "src/ppmi_subtype/hbayes_tt.py",
        "src/ppmi_subtype/hbayes_t.py", "src/ppmi_subtype/hbayes.py",
        "scripts/37_nu_slope_sampler_pilot.py")},
        "fit": fit, "seeds": SEEDS, "nu_inits": INITS, "groups": GROUPS,
        "thresholds": {"rhat_lt": RHAT_MAX, "ess_ge": ESS_MIN},
        "pilot_pass_rule": ("a collapsed sampler passes if, under BOTH frozen priors "
                            "(gamma_as_frozen, loguniform), nu_slope has R-hat < 1.01 and "
                            "ESS >= 400, and beta1, tau1, sigma, nu_resid have R-hat < 1.01"),
        "quick": a.quick}
    (out / "pilot_config.json").write_text(json.dumps(frozen, indent=2))

    panel = pd.read_csv(path, low_memory=False)
    data = hbayes.build_growth_data(panel)
    tasks = [(m, p, k) for m, p in GROUPS for k in range(4)]
    with ProcessPoolExecutor(a.jobs, initializer=_S.update, initargs=({"data": data, "fit": fit},)) as ex:
        res = list(ex.map(_task, tasks))
    log(f"{len(res)} chains done")

    rows, traces = [], {}
    for m, p in GROUPS:
        g = sorted([r for r in res if r[0] == m and r[1] == p], key=lambda r: r[2])
        posts = [r[3] for r in g]
        d = diag(posts)
        for par, v in d.items():
            rows.append({"method": m, "prior": p, "parameter": par, **v,
                         "fit_s_mean": float(np.mean([r[4] for r in g]))})
        for r in g:
            nu = r[3].nu
            rows.append({"method": m, "prior": p, "parameter": f"nu_slope chain{r[2]} "
                         f"(init {INITS[r[2]]}, seed {SEEDS[r[2]]})",
                         "q2.5": float(np.quantile(nu, .025)), "q50": float(np.median(nu)),
                         "q97.5": float(np.quantile(nu, .975)), "mean": float(nu.mean()),
                         "ess": SC.ess(nu[None]), "max": float(nu.max()),
                         "frac_gt_10": float((nu > 10).mean())})
            traces[f"{m}__{p}__chain{r[2]}"] = r[3].trace["nu"]
            traces[f"{m}__{p}__chain{r[2]}__tau1"] = r[3].trace["tau1"]
            if m == "lambda_grid":
                traces[f"{m}__{p}__chain{r[2]}__sum_log_lam"] = r[3].trace["sum_log_lam"]
                traces[f"{m}__{p}__chain{r[2]}__sum_lam"] = r[3].trace["sum_lam"]
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "pilot_diagnostics.csv", index=False)
    np.savez_compressed(out / "pilot_traces.npz", **traces)

    # ---- why the original step mixes slowly: width of p(nu | lambda) -----------
    M = data.n_patients
    g = NU.NU_GRID
    widths = []
    for k in range(4):
        sl = traces[f"lambda_grid__gamma_as_frozen__chain{k}__sum_log_lam"][fit["burn_in"]::50]
        sm = traces[f"lambda_grid__gamma_as_frozen__chain{k}__sum_lam"][fit["burn_in"]::50]
        for a_, b_ in zip(sl, sm):
            lp = HT.nu_conditional_logpdf(a_, b_, M, g, HT.log_prior_nu_gamma)
            p_ = np.exp(lp - lp.max())
            p_ /= p_.sum()
            mu = (p_ * g).sum()
            widths.append(np.sqrt((p_ * (g - mu) ** 2).sum()))
    post_sd = {f"{m}/{p}": float(np.std(np.concatenate([r[3].nu for r in res
                                                         if r[0] == m and r[1] == p])))
               for m, p in GROUPS}

    # ---- burn-in: sweeps until each chain first enters the pooled central 95% ---
    ref = np.concatenate([r[3].nu for r in res if r[0] == "collapsed_grid"
                          and r[1] == "gamma_as_frozen"])
    lo, hi = np.quantile(ref, [.025, .975])
    entry = {}
    for key, tr in traces.items():
        if key.count("__") == 2:
            inside = np.flatnonzero((tr >= lo) & (tr <= hi))
            entry[key] = int(inside[0]) if len(inside) else None

    # ---- verdict (pilot rule) ---------------------------------------------------------
    def ok(m, p):
        t = tab[(tab.method == m) & (tab.prior == p)].set_index("parameter")
        nu_ok = t.loc["nu_slope", "rhat"] < RHAT_MAX and t.loc["nu_slope", "ess"] >= ESS_MIN
        oth = all(t.loc[x, "rhat"] < RHAT_MAX for x in ("beta1", "tau1", "sigma", "nu_resid"))
        return bool(nu_ok), bool(oth)
    verdict = {f"{m}/{p}": dict(zip(("nu_slope_pass", "others_rhat_pass"), ok(m, p)))
               for m, p in GROUPS}
    pilot_pass = {m: all(all(verdict[f"{m}/{p}"].values()) for p in ("gamma_as_frozen",
                                                                       "loguniform"))
                  for m in ("lambda_grid", "collapsed_grid", "collapsed_slice")}

    # ---- agreement with the frozen full run (same model, priors, original sampler) -
    frozen_cmp = {}
    fz = FULL_RUN / "full_cohort_nu_traces.npz"
    if fz.exists():
        z = np.load(fz)
        for pr, key in (("gamma_as_frozen", "gamma"), ("loguniform", "loguniform")):
            v = np.concatenate([z[f"TT_{key}_seed{s}_nu"] for s in SEEDS])
            frozen_cmp[pr] = {"q2.5": float(np.quantile(v, .025)), "q50": float(np.median(v)),
                              "q97.5": float(np.quantile(v, .975))}

    figures(out, traces, tab, fit)
    meta = {"pilot_config": frozen, "verdict_by_group": verdict, "pilot_pass": pilot_pass,
            "lambda_grid_conditional_sd_of_nu_given_lambda": {
                "median": float(np.median(widths)), "q90": float(np.quantile(widths, .9))},
            "posterior_sd_nu_by_group": post_sd,
            "burnin_entry_sweep_into_central95": entry, "central95_reference": [lo, hi],
            "frozen_full_run_nu_slope": frozen_cmp,
            "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__},
            "host": platform.node(), "runtime_s": round(time.time() - t0, 1)}
    (out / "results.json").write_text(json.dumps(meta, indent=2, default=float))
    log(f"done in {meta['runtime_s']} s; pilot_pass={pilot_pass}")
    return 0


def figures(out, traces, tab, fit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    cols = ["C0", "C1", "C2", "C3"]

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fd / f"{name}.{ext}", dpi=140)
        plt.close(fig)

    fig, axes = plt.subplots(len(GROUPS), 1, figsize=(11, 2.0 * len(GROUPS)), sharex=True)
    for ax, (m, p) in zip(axes, GROUPS):
        for k in range(4):
            ax.plot(traces[f"{m}__{p}__chain{k}"], lw=0.3, color=cols[k],
                    label=f"init {INITS[k]}")
        ax.axvline(fit["burn_in"], color="k", lw=0.8, ls="--")
        ax.set_yscale("log")
        t = tab[(tab.method == m) & (tab.prior == p) & (tab.parameter == "nu_slope")].iloc[0]
        ax.set_title(f"{m} / {p}:  R-hat {t.rhat:.4f}, ESS {t.ess:.0f}", fontsize=9)
        ax.set_ylabel("nu_s")
    axes[0].legend(fontsize=7, ncol=4, loc="upper right")
    axes[-1].set_xlabel("sweep (dashed line = end of burn-in)")
    save(fig, "trace_nu_slope_all_sweeps")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
    for ax, m in zip(axes, ("lambda_grid", "collapsed_grid", "collapsed_slice")):
        for k in range(4):
            ax.plot(traces[f"{m}__gamma_as_frozen__chain{k}"][:600], color=cols[k], lw=0.8,
                    label=f"init {INITS[k]}")
        ax.set_yscale("log")
        ax.set_title(f"{m}: first 600 sweeps (gamma_as_frozen)", fontsize=9)
        ax.set_xlabel("sweep")
    axes[0].set_ylabel("nu_s")
    axes[0].legend(fontsize=7)
    save(fig, "trace_nu_slope_initialisation")

    fig, ax = plt.subplots(figsize=(6, 4))
    for m, c in (("lambda_grid", "k"), ("collapsed_grid", "C0"), ("collapsed_slice", "C1")):
        tr = traces[f"{m}__gamma_as_frozen__chain0"][fit["burn_in"]:][::fit["thin"]]
        z = tr - tr.mean()
        f = np.fft.rfft(z, 2 * len(z))
        acf = np.fft.irfft(f * np.conj(f))[:60]
        ax.plot(acf / acf[0], color=c, label=m)
    ax.set_xlabel("lag (retained draws, thin 10)")
    ax.set_ylabel("autocorrelation of nu_s")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.legend(fontsize=8)
    save(fig, "acf_nu_slope")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    bins = np.exp(np.linspace(np.log(1), np.log(30), 60))
    for ax, m in zip(axes, ("collapsed_grid", "collapsed_slice")):
        for p, c in zip(PRIORS, ("C0", "C1", "C2")):
            v = np.concatenate([traces[f"{m}__{p}__chain{k}"][fit["burn_in"]:][::fit["thin"]]
                                for k in range(4)])
            ax.hist(v, bins=bins, histtype="step", density=True, color=c, label=p)
        ax.set_xscale("log")
        ax.set_title(f"{m}: posterior of nu_s by prior", fontsize=9)
        ax.set_xlabel("nu_s")
        ax.legend(fontsize=7)
    save(fig, "posterior_nu_slope_by_prior")


if __name__ == "__main__":
    sys.exit(main())
