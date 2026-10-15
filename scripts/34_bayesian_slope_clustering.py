"""Part A: do Bayesian patient-specific TD/PIGD-ratio slopes form groups?

Exploratory. Fits the existing hierarchical growth model (hbayes, unchanged
specification, function-default sampler settings, project seed) to every
patient that `hbayes.build_growth_data` admits, takes each patient's total
slope s_i = beta1 + b1_i from the retained joint draws, and asks whether a
one-dimensional Gaussian mixture with K > 1 is warranted beyond a single
continuous population plus estimation uncertainty.

Pre-specified (fixed before any clustering was run):
  * feature: posterior-mean total slope, ratio units / year, unscaled;
  * mixtures: K = 1..5, 20 initialisations, seed 42;
  * selection: lowest BIC among converged fits whose every component holds
    >= max(10, 2% of n) patients by maximum-probability assignment;
  * evidence: dBIC = BIC(K=1) - best eligible BIC(K>=2); > 0 favours mixture;
  * posterior-draw sensitivity: 500 seeded joint draws, same rule;
  * visit sensitivity: >= 3 and >= 5 finite-ratio visits;
  * nulls: Gaussian (500), Johnson SU (500), hierarchical re-estimation (200);
  * Silverman critical-bandwidth unimodality test (500 bootstraps).

The model is fitted to all eligible patients rather than script 04's 70%
training split, because nothing here is evaluated on held-out data.

    python scripts/34_bayesian_slope_clustering.py --out <dir> [--jobs 8] [--quick]
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
from sklearn.metrics import adjusted_rand_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import hbayes  # noqa: E402
from ppmi_subtype import slope_clustering as SC  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

SEED = 42
PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
N_INIT = 20
_S: dict = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def visit_bucket(n):
    return np.select([n == 2, n == 3, (n >= 4) & (n <= 7), n >= 8], ["2", "3", "4-7", ">=8"],
                     default="<2")


# ---------------------------------------------------------------------------
#  Cohort accounting
# ---------------------------------------------------------------------------
def cohort_tables(panel, data):
    lab = panel[panel["subtype_name"].notna()].sort_values(["PATNO", "years_from_baseline"])
    fin = panel[np.isfinite(panel["ratio"])]
    per = lab.groupby("PATNO").agg(n_labelled=("subtype_name", "size"),
                                   baseline_subtype=("subtype_name", "first"),
                                   frac_td=("subtype_name", lambda s: (s == "TD").mean()))
    per["n_finite"] = fin.groupby("PATNO").size().reindex(per.index).fillna(0).astype(int)
    per["status"] = np.where(per.n_finite >= 2, "included",
                             np.where(per.n_finite == 1, "excluded: 1 finite ratio",
                                      "excluded: 0 finite ratios"))
    assert (per.status == "included").sum() == data.n_patients
    counts = {
        "panel_rows": len(panel),
        "labelled_visits": len(lab), "labelled_patients": lab.PATNO.nunique(),
        "finite_ratio_visits": len(fin), "finite_ratio_patients": fin.PATNO.nunique(),
        "labelled_visits_undefined_ratio": int((~np.isfinite(lab["ratio"])).sum()),
        "undefined_ratio_by_subtype": lab[~np.isfinite(lab["ratio"])].subtype_name
        .value_counts().to_dict(),
        "eligible_patients": int(data.n_patients), "eligible_observations": int(data.n_obs),
        "patients_excluded_one_finite": int((per.status == "excluded: 1 finite ratio").sum()),
        "patients_excluded_zero_finite": int((per.status == "excluded: 0 finite ratios").sum()),
    }
    ex = per.reset_index().groupby("status").agg(
        patients=("PATNO", "size"), median_labelled_visits=("n_labelled", "median"),
        mean_frac_td_visits=("frac_td", "mean"),
        baseline_TD=("baseline_subtype", lambda s: (s == "TD").mean()),
        baseline_PIGD=("baseline_subtype", lambda s: (s == "PIGD").mean()),
        baseline_Indeterminate=("baseline_subtype", lambda s: (s == "Indeterminate").mean()),
    ).reset_index()
    return counts, ex


# ---------------------------------------------------------------------------
#  Fits and diagnostics
# ---------------------------------------------------------------------------
def fit(data, seed, **kw):
    return hbayes.fit_growth_model(data, seed=seed, keep_random_effects=True, **kw)


def diagnostics(chains):
    """Split-R-hat and ESS for global parameters and every patient's total slope."""
    rows = []
    glob = {"beta0": lambda p: p.beta[:, 0], "beta1": lambda p: p.beta[:, 1],
            "sigma": lambda p: np.sqrt(p.sigma2), "tau0": lambda p: np.sqrt(p.tau0_2),
            "tau1": lambda p: np.sqrt(p.tau1_2)}
    for name, f in glob.items():
        c = np.stack([f(p) for p in chains])
        rows.append({"parameter": name, "rhat": SC.split_rhat(c), "ess": SC.ess(c),
                     "ess_chain1": SC.ess(c[:1])})
    S = np.stack([SC.total_slope_draws(p) for p in chains])          # (C, S, M)
    pr = np.array([SC.split_rhat(S[:, :, i]) for i in range(S.shape[2])])
    pe = np.array([SC.ess(S[:, :, i]) for i in range(S.shape[2])])
    pe1 = np.array([SC.ess(S[:1, :, i]) for i in range(S.shape[2])])
    for q, lab in [(None, "max"), (0.5, "median"), (0.99, "q99")]:
        rows.append({"parameter": f"patient slopes rhat {lab}",
                     "rhat": float(pr.max() if q is None else np.quantile(pr, q))})
    for q, lab in [(0.0, "min"), (0.01, "q01"), (0.5, "median")]:
        rows.append({"parameter": f"patient slopes ess {lab}",
                     "ess": float(np.quantile(pe, q)), "ess_chain1": float(np.quantile(pe1, q))})
    rows.append({"parameter": "patient slopes with rhat > 1.01", "rhat": int((pr > 1.01).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Parallel tasks
# ---------------------------------------------------------------------------
def sel_row(sel, n):
    f = sel.fits
    return {"selected_K": sel.selected_K, "unrestricted_K": sel.unrestricted_K,
            "delta_bic": sel.delta_bic, "delta_bic_unrestricted": sel.delta_bic_unrestricted,
            "all_converged": all(x.converged for x in f.values()),
            "n_nonconverged": sum(not x.converged for x in f.values()),
            "eligible_K": "|".join(str(K) for K, e in sel.eligible.items() if e),
            "selected_sizes": "|".join(map(str, f[sel.selected_K].sizes)),
            "K2_sizes": "|".join(map(str, f[2].sizes)),
            "K2_means": "|".join(f"{m:.4f}" for m in f[2].means),
            "K2_converged": f[2].converged,
            "min_floor": SC.min_component_size(n)}


def _draw_task(m):
    x = _S["S"][m]
    sel = SC.select_K(x, seed=SEED, n_init=N_INIT)
    row = {"draw_index": int(m), **sel_row(sel, len(x))}
    return row, sel.fits[sel.selected_K].labels.astype(np.int8), \
        sel.fits[2].labels.astype(np.int8)


def _null_marginal(args):
    kind, rep = args
    rng = np.random.default_rng([SEED, 34, {"gauss": 1, "jsu": 2}[kind], rep])
    n = _S["n"]
    if kind == "gauss":
        x = rng.normal(_S["mu"], _S["sd"], n)
    else:
        x = stats.johnsonsu.rvs(*_S["jsu"], size=n, random_state=rng)
    sel = SC.select_K(x, seed=SEED, n_init=N_INIT)
    return {"null": kind, "rep": rep, **sel_row(sel, n),
            "h_crit": SC.critical_bandwidth(x, 1) / x.std()}


def _null_hier(rep):
    """Simulate ratios from the fitted model with Gaussian slopes; refit; cluster."""
    d, th = _S["data"], _S["theta"]
    rng = np.random.default_rng([SEED, 34, 3, rep])
    M = d.n_patients
    b0 = rng.normal(0, np.sqrt(th["tau0_2"]), M)
    b1 = rng.normal(0, np.sqrt(th["tau1_2"]), M)
    i = d.patient_index
    y = (th["beta0"] + b0[i]) + (th["beta1"] + b1[i]) * d.t \
        + rng.normal(0, np.sqrt(th["sigma2"]), d.n_obs)
    sim = hbayes.GrowthData(y=y, t=d.t, patient_index=i, patients=d.patients,
                            n_patients=M, n_obs=d.n_obs)
    post = fit(sim, seed=SEED + 1000 + rep, **_S["fit_kw"])
    x = SC.total_slope_draws(post).mean(axis=0)
    sel = SC.select_K(x, seed=SEED, n_init=N_INIT)
    return {"null": "hierarchical", "rep": rep, **sel_row(sel, M),
            "h_crit": SC.critical_bandwidth(x, 1) / x.std(),
            "sd_post_means": float(x.std()), "tau1_hat": float(np.sqrt(post.tau1_2).mean()),
            "true_b1_sd": float(b1.std())}


def _init(state):
    _S.update(state)


def pool_map(fn, items, jobs, state):
    if jobs <= 1:
        _init(state)
        return [fn(a) for a in items]
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=(state,)) as ex:
        return list(ex.map(fn, items, chunksize=max(1, len(items) // (4 * jobs))))


# ---------------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------------
def figures(out, summ, beta1, sel, comp, coc_order, coc, nulls, obs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    x = summ.slope_mean.to_numpy()
    grid = np.linspace(x.min() - 0.05, x.max() + 0.05, 600)

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fd / f"{name}.{ext}", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(x, bins=60, density=True, color="0.75", edgecolor="white")
    ax.plot(grid, stats.gaussian_kde(x)(grid), "k-", lw=1.2, label="KDE (Scott)")
    ax.axvline(beta1, color="C3", ls="--", label=f"population slope β1 = {beta1:.3f}")
    ax.axvline(0, color="0.4", lw=0.6)
    ax.set_xlabel("posterior-mean total slope β1 + b1i (ratio units / yr)")
    ax.set_ylabel("density")
    ax.set_title(f"Fig 1. Patient slope distribution (n = {len(x)})")
    ax.legend(fontsize=8)
    save(fig, "fig1_slope_distribution")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, K in zip(axes, (1, 2, 3)):
        f = sel.fits[K]
        ax.hist(x, bins=60, density=True, color="0.85", edgecolor="white")
        tot = np.zeros_like(grid)
        for w, m, s in zip(f.weights, f.means, f.sds):
            comp_d = w * stats.norm.pdf(grid, m, s)
            tot += comp_d
            ax.plot(grid, comp_d, lw=1)
        ax.plot(grid, tot, "k-", lw=1.4)
        ax.set_title(f"K = {K}  BIC = {f.bic:.1f}  sizes {'/'.join(map(str, f.sizes))}",
                     fontsize=9)
        ax.set_xlabel("slope (ratio / yr)")
    axes[0].set_ylabel("density")
    fig.suptitle("Fig 2. Gaussian mixture fits to posterior-mean slopes", fontsize=10)
    save(fig, "fig2_mixture_fits")

    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.plot(comp.K, comp.bic, "ko-")
    for _, r in comp.iterrows():
        if not r.eligible:
            ax.plot(r.K, r.bic, "rx", ms=10)
    ax.set_xlabel("K")
    ax.set_ylabel("BIC (lower is better)")
    ax.set_title("Fig 3. BIC vs K (red x = ineligible)", fontsize=10)
    save(fig, "fig3_bic_vs_k")

    o = summ.sort_values("slope_mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.vlines(np.arange(len(o)), o["slope_q2.5"], o["slope_q97.5"], color="0.7", lw=0.5)
    ax.plot(np.arange(len(o)), o.slope_mean, "k.", ms=1.5)
    ax.axhline(beta1, color="C3", ls="--", lw=1, label="β1")
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_xlabel("patients ordered by posterior mean")
    ax.set_ylabel("total slope, 95% CrI")
    ax.set_title("Fig 4. Patient slopes with 95% credible intervals", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, "fig4_patient_credible_intervals")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(coc[np.ix_(coc_order, coc_order)], cmap="viridis", vmin=0, vmax=1,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="co-assignment frequency")
    ax.set_xlabel("patients ordered by posterior-mean slope")
    ax.set_ylabel("patients ordered by posterior-mean slope")
    ax.set_title("Fig 5. Co-clustering across posterior draws (fixed K = 2)", fontsize=9)
    save(fig, "fig5_coclustering")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for kind, c in [("gauss", "C0"), ("jsu", "C1"), ("hierarchical", "C2")]:
        v = nulls.loc[nulls.null == kind, "delta_bic_unrestricted"]
        axes[0].hist(v, bins=30, alpha=0.5, color=c, label=f"{kind} null (n={len(v)})")
        h = nulls.loc[nulls.null == kind, "h_crit"]
        axes[1].hist(h, bins=30, alpha=0.5, color=c, label=kind)
    axes[0].axvline(obs["delta_bic_unrestricted"], color="k", lw=2, label="observed")
    axes[0].set_xlabel("BIC(K=1) − min BIC(K≥2)   (> 0 favours mixture)")
    axes[1].axvline(obs["h_crit"], color="k", lw=2, label="observed")
    axes[1].set_xlabel("Silverman critical bandwidth / SD (larger = more multimodal)")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle("Fig 6. Observed statistics vs continuous no-subgroup nulls", fontsize=10)
    save(fig, "fig6_null_comparison")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="smoke test: tiny replicate counts")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    q = a.quick
    n_draw_sub, n_gauss, n_jsu, n_hier, n_silv = (20, 10, 10, 4, 30) if q else (500, 500, 500, 200, 500)
    fit_kw = {"n_draws": 400, "burn_in": 100} if q else {}       # else function defaults
    t0 = time.time()

    cfg = load_config()
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    if not panel_path.exists():
        print(f"missing: {panel_path}", file=sys.stderr)
        return 2
    pmd5 = md5(panel_path)
    if pmd5 != PANEL_MD5:
        print(f"panel MD5 {pmd5} != frozen {PANEL_MD5}", file=sys.stderr)
        return 2
    assert cfg.seed == SEED
    panel = pd.read_csv(panel_path, low_memory=False)
    data = hbayes.build_growth_data(panel)                # existing eligibility, unchanged
    counts, excl = cohort_tables(panel, data)
    log(f"eligible {data.n_patients} patients, {data.n_obs} observations")
    excl.to_csv(out / "exclusions.csv", index=False)

    # ---- primary fit + extra diagnostic chains --------------------------------
    post = fit(data, SEED, **fit_kw)
    log(f"primary fit done: {post.n_draws} draws, RE {post.random_effects.shape}")
    extra = [fit(data, SEED + k, **fit_kw) for k in (1, 2, 3)]
    diag = diagnostics([post, *extra])
    diag.to_csv(out / "sampling_diagnostics.csv", index=False)
    del extra
    S = SC.total_slope_draws(post)                         # (S, M) = beta1 + b1i
    assert np.allclose(S, post.beta[:, 1][:, None] + post.random_effects[:, :, 1])
    np.savez_compressed(out / "slope_draws.npz", slope_draws=S.astype(np.float32),
                        patients=np.asarray(post.patients), beta1=post.beta[:, 1],
                        beta0=post.beta[:, 0], tau1_2=post.tau1_2, tau0_2=post.tau0_2,
                        sigma2=post.sigma2)
    pd.DataFrame({"column": np.arange(len(post.patients)), "PATNO": post.patients}) \
        .to_csv(out / "slope_draws_patient_order.csv", index=False)
    post.summary().to_csv(out / "posterior_global.csv", index=False)

    summ = SC.patient_slope_summary(post, data)
    summ["visit_bucket"] = visit_bucket(summ.n_ratio_obs.to_numpy())
    vd = summ.visit_bucket.value_counts().reindex(["2", "3", "4-7", ">=8"]).fillna(0)
    counts["visit_distribution"] = {k: int(v) for k, v in vd.items()}
    x = summ.slope_mean.to_numpy()
    n = len(x)

    # ---- primary mixture comparison --------------------------------------------
    sel = SC.select_K(x, seed=SEED, n_init=N_INIT)
    comp = SC.comparison_table(sel, n)
    comp.to_csv(out / "gmm_model_comparison.csv", index=False)
    log(f"primary: selected K={sel.selected_K}, unrestricted K={sel.unrestricted_K}, "
        f"dBIC={sel.delta_bic:.2f}")
    f2 = sel.fits[2]
    fs = sel.fits[sel.selected_K]
    assign = summ[["PATNO", "slope_mean", "slope_sd", "n_ratio_obs", "followup_years"]].copy()
    assign["selected_K"] = sel.selected_K
    assign["label_selected"] = fs.labels
    assign["maxprob_selected"] = fs.probs.max(axis=1)
    assign["label_K2"] = f2.labels
    assign["prob_K2_comp1"] = f2.probs[:, 1]

    # ---- posterior-draw sensitivity + co-clustering --------------------------
    rng = np.random.default_rng([SEED, 34, 0])
    sub = np.sort(rng.choice(post.n_draws, n_draw_sub, replace=False))
    res = pool_map(_draw_task, list(sub), a.jobs, {"S": S})
    dtab = pd.DataFrame([r[0] for r in res])
    dtab.to_csv(out / "posterior_draw_sensitivity.csv", index=False)
    C_sel = np.zeros((n, n))
    C_2 = np.zeros((n, n))
    for _, lab_sel, lab2 in res:
        SC.coclustering_update(C_sel, lab_sel)
        SC.coclustering_update(C_2, lab2)
    C_sel /= len(res)
    C_2 /= len(res)
    np.savez_compressed(out / "coclustering.npz", fixed_K2=C_2.astype(np.float32),
                        per_draw_selected_K=C_sel.astype(np.float32),
                        patients=np.asarray(post.patients), draw_indices=sub)
    # stability of the primary K=2 partition: mean co-assignment within / between
    same = f2.labels[:, None] == f2.labels[None, :]
    off = ~np.eye(n, dtype=bool)
    ari_draw = [adjusted_rand_score(f2.labels, r[2]) for r in res]
    # per-patient: how often a patient lands with the primary-partition majority
    pat_stab = np.array([C_2[i, same[i] & off[i]].mean() for i in range(n)])
    assign["K2_coassign_with_own_component"] = pat_stab
    assign.to_csv(out / "cluster_assignments.csv", index=False)
    stability = {
        "n_draws_used": int(len(res)),
        "selected_K_frequency": dtab.selected_K.value_counts().sort_index().to_dict(),
        "unrestricted_K_frequency": dtab.unrestricted_K.value_counts().sort_index().to_dict(),
        "draws_with_nonconverged_fit": int((~dtab.all_converged).sum()),
        "K2_within_component_coassign_mean": float(C_2[same & off].mean()),
        "K2_between_component_coassign_mean": float(C_2[~same].mean()),
        "K2_ARI_draw_vs_primary_median": float(np.median(ari_draw)),
        "K2_ARI_draw_vs_primary_q05_q95": [float(np.quantile(ari_draw, .05)),
                                           float(np.quantile(ari_draw, .95))],
        "patients_K2_coassign_below_0.7": int((pat_stab < 0.7).sum()),
        "draw_delta_bic_median": float(dtab.delta_bic_unrestricted.median()),
        "draw_delta_bic_q05_q95": [float(dtab.delta_bic_unrestricted.quantile(.05)),
                                   float(dtab.delta_bic_unrestricted.quantile(.95))],
    }
    log(f"posterior draws: {stability['selected_K_frequency']}")

    # ---- visit-count sensitivity -------------------------------------------------
    srows = []
    for mv in (3, 5):
        keep = summ.n_ratio_obs.to_numpy() >= mv
        for variant in ("primary posterior, restricted", "refit on restricted cohort"):
            if variant.startswith("primary"):
                xs, ids = x[keep], summ.PATNO.to_numpy()[keep]
            else:
                dsub = hbayes.build_growth_data(panel, min_visits=mv)
                ps = fit(dsub, SEED, **fit_kw)
                xs, ids = SC.total_slope_draws(ps).mean(axis=0), np.asarray(ps.patients)
            ss = SC.select_K(xs, seed=SEED, n_init=N_INIT)
            prim = pd.Series(f2.labels, index=summ.PATNO).loc[ids].to_numpy()
            sub_lab = pd.Series(ss.fits[2].labels, index=ids).to_numpy()
            srows.append({"min_visits": mv, "variant": variant, "n_patients": len(xs),
                          **sel_row(ss, len(xs)),
                          "ARI_K2_vs_primary_common": adjusted_rand_score(prim, sub_lab),
                          "ARI_selected_vs_primary_selected": adjusted_rand_score(
                              pd.Series(fs.labels, index=summ.PATNO).loc[ids].to_numpy(),
                              ss.fits[ss.selected_K].labels),
                          "slope_sd": float(xs.std())})
    sens = pd.DataFrame(srows)
    sens.to_csv(out / "sensitivity_analysis.csv", index=False)
    log("visit sensitivity done")

    # ---- association with data quantity -------------------------------------
    assoc = []
    for col in ("n_ratio_obs", "followup_years"):
        for target in ("slope_mean", "slope_sd"):
            r, p = stats.spearmanr(summ[col], summ[target])
            assoc.append({"x": col, "y": target, "test": "spearman", "stat": r, "p": p})
        r, p = stats.spearmanr(summ[col], np.abs(x - post.beta[:, 1].mean()))
        assoc.append({"x": col, "y": "|slope_mean - beta1|", "test": "spearman",
                      "stat": r, "p": p})
        groups = [summ.loc[f2.labels == k, col] for k in range(2)]
        h, p = stats.mannwhitneyu(*groups)
        assoc.append({"x": col, "y": "K2 label", "test": "mann-whitney", "stat": h, "p": p,
                      "median_comp0": groups[0].median(), "median_comp1": groups[1].median()})
    ct = pd.crosstab(summ.visit_bucket, f2.labels)
    chi = stats.chi2_contingency(ct)
    assoc.append({"x": "visit_bucket", "y": "K2 label", "test": "chi2", "stat": chi[0],
                  "p": chi[1], "table": ct.to_dict()})
    pd.DataFrame(assoc).to_csv(out / "association_with_data_quantity.csv", index=False)

    # ---- Silverman unimodality test on the observed slopes -------------------
    silv = SC.silverman_test(x, n_boot=n_silv, seed=SEED)
    silv["h_crit_over_sd"] = silv["h_crit"] / x.std()
    log(f"Silverman: {silv}")

    # ---- nulls ------------------------------------------------------------------
    jsu = stats.johnsonsu.fit(x)
    st_marg = {"n": n, "mu": float(x.mean()), "sd": float(x.std(ddof=1)), "jsu": jsu}
    nm = pool_map(_null_marginal, [("gauss", r) for r in range(n_gauss)]
                  + [("jsu", r) for r in range(n_jsu)], a.jobs, st_marg)
    log("marginal nulls done")
    theta = {"beta0": float(post.beta[:, 0].mean()), "beta1": float(post.beta[:, 1].mean()),
             "tau0_2": float(post.tau0_2.mean()), "tau1_2": float(post.tau1_2.mean()),
             "sigma2": float(post.sigma2.mean())}
    nh = pool_map(_null_hier, list(range(n_hier)), a.jobs,
                  {"data": data, "theta": theta, "fit_kw": fit_kw})
    nulls = pd.DataFrame(nm + nh)
    nulls.to_csv(out / "null_simulations.csv", index=False)
    obs = {"delta_bic": sel.delta_bic, "delta_bic_unrestricted": sel.delta_bic_unrestricted,
           "h_crit": silv["h_crit_over_sd"], "selected_K": sel.selected_K}
    nsum = []
    for kind, g in nulls.groupby("null"):
        nsum.append({
            "null": kind, "n_reps": len(g),
            "frac_selected_K_ge2": float((g.selected_K >= 2).mean()),
            "frac_unrestricted_K_ge2": float((g.unrestricted_K >= 2).mean()),
            "null_dBIC_unrestricted_median": float(g.delta_bic_unrestricted.median()),
            "null_dBIC_unrestricted_q95": float(g.delta_bic_unrestricted.quantile(.95)),
            "p_dBIC_unrestricted": (1 + (g.delta_bic_unrestricted
                                         >= obs["delta_bic_unrestricted"]).sum()) / (len(g) + 1),
            "null_hcrit_median": float(g.h_crit.median()),
            "p_hcrit": (1 + (g.h_crit >= obs["h_crit"]).sum()) / (len(g) + 1),
            "nonconverged_reps": int((~g.all_converged).sum()),
        })
    nsum = pd.DataFrame(nsum)
    nsum.to_csv(out / "null_summary.csv", index=False)
    log("nulls done")

    # ---- figures, metadata -------------------------------------------------------
    summ.to_csv(out / "patient_slopes.csv", index=False)
    figures(out, summ, float(post.beta[:, 1].mean()), sel, comp,
            np.argsort(x), C_2, nulls, obs)
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    meta = {
        "script": "scripts/34_bayesian_slope_clustering.py", "quick": q, "git_head": git,
        "panel_md5": pmd5, "seed": SEED, "fit": {"n_draws": post.n_draws,
                                                  **({"burn_in": 1000, "thin": 2} if not q
                                                     else fit_kw)},
        "n_init": N_INIT, "size_floor": SC.min_component_size(n),
        "replicates": {"posterior_draws": n_draw_sub, "gauss": n_gauss, "jsu": n_jsu,
                       "hierarchical": n_hier, "silverman_boot": n_silv},
        "counts": counts,
        "posterior": {"beta0": theta["beta0"], "beta1": theta["beta1"],
                      "beta1_ci": np.quantile(post.beta[:, 1], [.025, .975]).tolist(),
                      "tau1": float(np.sqrt(post.tau1_2).mean()),
                      "tau1_ci": np.quantile(np.sqrt(post.tau1_2), [.025, .975]).tolist(),
                      "sigma": float(np.sqrt(post.sigma2).mean()),
                      "tau0": float(np.sqrt(post.tau0_2).mean())},
        "slopes": {"mean_of_means": float(x.mean()), "sd_of_means": float(x.std(ddof=1)),
                   "median_posterior_sd": float(summ.slope_sd.median()),
                   "frac_CrI_excluding_zero": float(((summ["slope_q97.5"] < 0)
                                                     | (summ["slope_q2.5"] > 0)).mean()),
                   "skew": float(stats.skew(x)), "excess_kurtosis": float(stats.kurtosis(x))},
        "primary": {"selected_K": sel.selected_K, "unrestricted_K": sel.unrestricted_K,
                    "delta_bic": sel.delta_bic,
                    "delta_bic_unrestricted": sel.delta_bic_unrestricted},
        "johnson_su_params": list(map(float, jsu)), "silverman": silv,
        "stability": stability, "hier_null_theta": theta,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__,
                     "sklearn": __import__("sklearn").__version__,
                     "scipy": __import__("scipy").__version__},
        "host": platform.node(), "runtime_s": round(time.time() - t0, 1),
    }
    (out / "results.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"done in {meta['runtime_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
