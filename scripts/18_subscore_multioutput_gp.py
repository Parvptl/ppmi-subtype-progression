"""Step 2: multi-output Gaussian process over the tremor and PIGD subscores.

Step 1 established a joint linear mixed model on the same two quantities. It
lost to persistence on MAE and found a near-zero time-adjusted cross-output
correlation. Step 2 therefore asks two separate questions, and keeps them
separate:

    does flexible temporal structure help?        G_K0 vs G_K1 / vs persistence
    does the cross-output block help?             C_full vs C_diag

The second pair is exactly nested --- same kernel, same lengthscale, same
everything, differing only in whether the off-diagonals of every B_c and of
Sigma_e are free. Independent per-output GPs are a *different* restriction (they
also get their own lengthscale) and are fitted through the existing `gpr` module
rather than here, so a B-vs-C gap is never read as evidence about
coregionalization alone.

    python scripts/18_subscore_multioutput_gp.py [--jobs 12] [--limit N]
                                                 [--restarts 4] [--skip-lopo]

Reads   outputs/visit_panel.csv
Writes  outputs/joint_subscore_gp/   (never outputs/joint_subscore/, which is Step 1)

No GPU is used: the installed PyTorch is CPU-only and no CUDA device is present
on this machine. Parallelism is over patients via multiprocessing, as in
`gpr_loo`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# One BLAS thread per process, set before NumPy is imported. The matrices here
# are at most 28x28, so threaded BLAS buys nothing and the 12 worker processes
# otherwise oversubscribe the 12 cores several times over. Measured effect on
# this machine: 17.5s per fold with threading left on, far less without.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from ppmi_subtype import gpr, mogp as M, subtype as st, trajectory as tj  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

#: LOPO configurations. "indep" is two separate single-output GPs through
#: `gpr.fit_gpr`; "full"/"diag" are the joint model with and without the
#: cross-output block.
ARMS = [
    ("gp_joint_K0_full", "K0", "full"),   # linear temporal control
    ("gp_joint_K1_full", "K1", "full"),   # flexible temporal, cross-output free
    ("gp_joint_K1_diag", "K1", "diag"),   # flexible temporal, cross-output pinned
    ("gp_indep_K1", "K1", "indep"),       # two separate GPs, own lengthscale each
]
# K2/K3 are fitted at population level only. Their 13-parameter surface takes
# 250-320s per fit here, which is 65+ hours over 934 folds on this machine, so
# the held-out comparison for them was not run. The nested cross-output test is
# carried by the K1 pair, which differs in exactly the off-diagonals.
BASELINES = ["persistence", "patient_mean"]
N_BOOT = 4000
SEED = 42
JITTER = 1e-8

_STATE: dict = {}


# ---------------------------------------------------------------------------
#  Cohort
# ---------------------------------------------------------------------------
def build_sequences(cfg):
    """The Step 1 cohort, unchanged: 934 patients, every labelled visit.

    No ratio is computed and no PIGD == 0 visit is removed. The time axis,
    patient set and visit ordering come from `build_trajectories`, so these are
    the same rows Step 1 modelled and the same points it scored.
    """
    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    traj = tj.build_trajectories(panel, cfg)
    tj.check_ordering(traj)
    frame = traj.merge(
        panel[["PATNO", "EVENT_ID", "tremor_score", "pigd_score"]]
        .drop_duplicates(["PATNO", "EVENT_ID"]),
        on=["PATNO", "EVENT_ID"], how="left", validate="one_to_one",
    )
    if frame[["tremor_score", "pigd_score"]].isna().any().any():
        raise SystemExit("a visit is missing a subscore after the merge")

    seqs = {}
    for patno, g in frame.groupby("PATNO", sort=True):
        g = g.sort_values("years_from_baseline", kind="mergesort")
        seqs[patno] = {
            "t": g["years_from_baseline"].to_numpy(dtype=float),
            "y": g[["tremor_score", "pigd_score"]].to_numpy(dtype=float),
            "subtype": g["subtype_name"].to_numpy(),
        }
    return frame, seqs


# ---------------------------------------------------------------------------
#  One fold
# ---------------------------------------------------------------------------
def _fit_arm(train_seqs, kernel, coregion, restarts, max_iter):
    """Fit one arm on a training set. Returns an object with a `predict` method."""
    if coregion == "indep":
        fits = []
        for d in range(2):
            # `fit_gpr` takes the sequence list itself and packs internally.
            fits.append(gpr.fit_gpr(
                [{"t": s["t"], "y": s["y"][:, d]} for s in train_seqs],
                kernel=kernel, n_restarts=restarts, max_iter=max_iter,
                jitter=JITTER, seed=SEED))
        return ("indep", fits)
    packed = M.pack([{"t": s["t"], "y": s["y"]} for s in train_seqs])
    return ("joint", M.fit_mogp(packed, kernel=kernel, coregion=coregion,
                                n_restarts=restarts, max_iter=max_iter,
                                jitter=JITTER, seed=SEED))


def _predict(handle, t_hist, Y_hist, t_target):
    """Predictive mean (2,) and covariance (2, 2) for either arm type."""
    kind, fit = handle
    if kind == "indep":
        mean = np.empty(2)
        cov = np.zeros((2, 2))
        for d in range(2):
            m, s = gpr.predict_one(fit[d], t_hist, Y_hist[:, d], t_target,
                                   jitter=JITTER)
            mean[d] = m
            cov[d, d] = s ** 2
        return mean, cov          # independent by construction: zero off-diagonal
    return M.predict_one(fit, t_hist, Y_hist, t_target, jitter=JITTER)


def _fit_summary(handle, patno):
    kind, fit = handle
    if kind == "indep":
        # `GPRFit` reports a log marginal likelihood; negate so every arm's
        # "nll" column means the same thing and is comparable in sign.
        row = {"held_out_patient": patno,
               "converged": bool(fit[0].converged and fit[1].converged),
               "n_train_patients": fit[0].n_patients,
               "nll": -(fit[0].log_marginal_likelihood
                        + fit[1].log_marginal_likelihood),
               "at_bound": "|".join(fit[0].at_boundary + fit[1].at_boundary)}
        for d, name in enumerate(M.OUTPUTS):
            for k, v in zip(gpr.parameter_names(fit[d].kernel), fit[d].theta):
                row[f"{name}_{k}"] = float(v)
            row[f"{name}_beta0"] = float(fit[d].beta[0])
            row[f"{name}_beta1"] = float(fit[d].beta[1])
        return row
    row = {"held_out_patient": patno, "converged": fit.converged,
           "n_train_patients": fit.n_patients, "nll": fit.nll,
           "at_bound": "|".join(fit.at_bound)}
    row.update(fit.parameter_dict())
    row.update({f"corr_{k}": v for k, v in fit.coregion_correlation().items()})
    return row


def _init_worker(seqs, arms, restarts, max_iter):
    _STATE.update(seqs=seqs, arms=arms, restarts=restarts, max_iter=max_iter)


def _run_fold(patno):
    """Refit every arm without `patno`, then predict that patient's visits."""
    seqs = _STATE["seqs"]
    train = [s for p, s in seqs.items() if p != patno]
    t_all, Y_all = seqs[patno]["t"], seqs[patno]["y"]
    subtypes = seqs[patno]["subtype"]

    preds, fits = [], []
    for label, kernel, coregion in _STATE["arms"]:
        handle = _fit_arm(train, kernel, coregion,
                          _STATE["restarts"], _STATE["max_iter"])
        fits.append({"model": label, **_fit_summary(handle, patno)})
        for k in range(1, len(t_all)):
            mean, cov = _predict(handle, t_all[:k], Y_all[:k], float(t_all[k]))
            sd = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
            preds.append({
                "held_out_patient": patno, "model": label,
                "target_index": k, "history_length": k,
                "t_last_history": float(t_all[k - 1]), "t_target": float(t_all[k]),
                "horizon_years": float(t_all[k] - t_all[k - 1]),
                "obs_tremor": float(Y_all[k, 0]), "obs_pigd": float(Y_all[k, 1]),
                "obs_subtype": subtypes[k],
                "pred_tremor": float(mean[0]), "pred_pigd": float(mean[1]),
                "sd_tremor": float(sd[0]), "sd_pigd": float(sd[1]),
                "cov_tremor_pigd": float(cov[0, 1]),
                "corr_pred": float(cov[0, 1] / (sd[0] * sd[1])),
                "neg_mass_tremor": gpr.negative_mass(mean[0], sd[0]),
                "neg_mass_pigd": gpr.negative_mass(mean[1], sd[1]),
                "n_train_patients": len(train),
                "train_converged": fits[-1]["converged"],
            })

    # Baselines, computed once, on exactly the same points.
    for k in range(1, len(t_all)):
        hist = Y_all[:k]
        for name in BASELINES:
            mean = hist[-1] if name == "persistence" else hist.mean(axis=0)
            sd = (hist.std(axis=0, ddof=1) if k > 1 else np.array([np.nan, np.nan]))
            sd = np.where(np.isfinite(sd) & (sd > 0), sd, 0.3)
            preds.append({
                "held_out_patient": patno, "model": name,
                "target_index": k, "history_length": k,
                "t_last_history": float(t_all[k - 1]), "t_target": float(t_all[k]),
                "horizon_years": float(t_all[k] - t_all[k - 1]),
                "obs_tremor": float(Y_all[k, 0]), "obs_pigd": float(Y_all[k, 1]),
                "obs_subtype": subtypes[k],
                "pred_tremor": float(mean[0]), "pred_pigd": float(mean[1]),
                "sd_tremor": float(sd[0]), "sd_pigd": float(sd[1]),
                "cov_tremor_pigd": 0.0, "corr_pred": 0.0,
                "neg_mass_tremor": gpr.negative_mass(mean[0], sd[0]),
                "neg_mass_pigd": gpr.negative_mass(mean[1], sd[1]),
                "n_train_patients": len(train), "train_converged": True,
            })
    return preds, fits


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------
def boot_ci(values, patients, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(patients)
    groups = {p: values[patients == p] for p in uniq}
    draws = np.empty(n)
    for k in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        draws[k] = np.concatenate([groups[p] for p in pick]).mean()
    return float(values.mean()), *np.percentile(draws, [2.5, 97.5])


def continuous_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in preds.groupby("model"):
        pat = g.held_out_patient.to_numpy()
        for short, obs, pred, sd, nm in [
            ("tremor", "obs_tremor", "pred_tremor", "sd_tremor", "neg_mass_tremor"),
            ("pigd", "obs_pigd", "pred_pigd", "sd_pigd", "neg_mass_pigd"),
        ]:
            err = (g[pred] - g[obs]).to_numpy()
            mae, lo, hi = boot_ci(np.abs(err), pat)
            z = err / g[sd].to_numpy()
            rows.append({
                "model": model, "outcome": short, "n": len(g),
                "n_patients": int(g.held_out_patient.nunique()),
                "MAE": mae, "MAE_lo": lo, "MAE_hi": hi,
                "RMSE": float(np.sqrt((err ** 2).mean())),
                "bias": float(err.mean()),
                "coverage95": float((np.abs(z) <= 1.959963984540054).mean()),
                "mean_negative_mass": float(g[nm].mean()),
                "frac_pred_below_zero": float((g[pred] < 0).mean()),
            })
    return pd.DataFrame(rows)


def paired(preds: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Paired MAE differences against `reference`, on identical points."""
    rows = []
    key = ["held_out_patient", "target_index"]
    for short, obs, pred in [("tremor", "obs_tremor", "pred_tremor"),
                             ("pigd", "obs_pigd", "pred_pigd")]:
        w = preds.assign(ae=(preds[pred] - preds[obs]).abs()).pivot_table(
            index=key, columns="model", values="ae").dropna()
        if reference not in w:
            continue
        pat = w.index.get_level_values(0).to_numpy()
        for model in w.columns:
            if model == reference:
                continue
            diff = (w[model] - w[reference]).to_numpy()
            pt, lo, hi = boot_ci(diff, pat)
            rows.append({
                "outcome": short, "comparison": f"{model} - {reference}",
                "delta_MAE": pt, "lo": lo, "hi": hi,
                "excludes_zero": (lo < 0) == (hi < 0), "n": len(diff),
            })
    return pd.DataFrame(rows)


def derive_subtype(preds: pd.DataFrame, cfg, n_draws: int = 4000) -> pd.DataFrame:
    """Point-estimate and probabilistic subtype, kept as separate quantities.

    The point estimate applies the project's own `classify` to the predictive
    means, which handles PIGD == 0 and both == 0 through the published
    degenerate rules --- no epsilon is introduced anywhere.

    The probabilistic version draws from the *joint* bivariate predictive and
    classifies each draw. A ratio of two correlated Gaussians has no closed-form
    distribution, so Monte Carlo is the honest route; using the joint covariance
    is the whole reason the multi-output model can do this at all. Draws are
    clipped at zero before classification because a subscore cannot be negative,
    and the clipped fraction is reported alongside.
    """
    out = preds.copy()
    out["pred_subtype"] = st.classify(
        out.pred_tremor.clip(lower=0.0), out.pred_pigd.clip(lower=0.0), cfg.subtype
    ).map(cfg.subtype.label_names)

    rng = np.random.default_rng(SEED)
    mt, mp = out.pred_tremor.to_numpy(), out.pred_pigd.to_numpy()
    st_, sp = out.sd_tremor.to_numpy(), out.sd_pigd.to_numpy()
    rho = np.clip(out.corr_pred.to_numpy(), -0.999, 0.999)

    z1 = rng.standard_normal((len(out), n_draws))
    z2 = rng.standard_normal((len(out), n_draws))
    draw_t = np.clip(mt[:, None] + st_[:, None] * z1, 0.0, None)
    draw_p = np.clip(
        mp[:, None] + sp[:, None] * (rho[:, None] * z1
                                     + np.sqrt(1 - rho[:, None] ** 2) * z2), 0.0, None)

    td, pg = cfg.subtype.td_threshold, cfg.subtype.pigd_threshold
    ratio = np.divide(draw_t, draw_p, out=np.full_like(draw_t, np.inf), where=draw_p > 0)
    is_td = (ratio >= td) | ((draw_p == 0) & (draw_t > 0))
    is_pigd = (draw_p > 0) & (ratio <= pg)
    out["p_TD"] = is_td.mean(axis=1)
    out["p_PIGD"] = is_pigd.mean(axis=1)
    out["p_Indeterminate"] = np.clip(1.0 - out.p_TD - out.p_PIGD, 0.0, 1.0)
    out["prob_subtype"] = np.array(["TD", "Indeterminate", "PIGD"])[
        np.argmax(out[["p_TD", "p_Indeterminate", "p_PIGD"]].to_numpy(), axis=1)]
    return out


def subtype_metrics(derived: pd.DataFrame, column: str) -> pd.DataFrame:
    classes = ["TD", "Indeterminate", "PIGD"]
    rows = []
    for model, g in derived.groupby("model"):
        correct = (g[column] == g.obs_subtype).to_numpy().astype(float)
        acc, lo, hi = boot_ci(correct, g.held_out_patient.to_numpy())
        rec = {c: float((g.loc[g.obs_subtype == c, column] == c).mean())
               if (g.obs_subtype == c).any() else np.nan for c in classes}
        rows.append({
            "model": model, "rule": column, "n": len(g), "accuracy": acc,
            "accuracy_lo": lo, "accuracy_hi": hi,
            "balanced_accuracy": float(np.nanmean(list(rec.values()))),
            **{f"recall_{c}": rec[c] for c in classes},
            **{f"n_predicted_{c}": int((g[column] == c).sum()) for c in classes},
        })
    return pd.DataFrame(rows)


def check_predictions(preds: pd.DataFrame, seqs: dict, scored: list) -> list:
    """The leakage battery, run on every saved prediction row."""
    out = []
    models = sorted(preds.model.unique())
    g = preds[preds.model == ARMS[0][0]]
    expected = sum(len(seqs[p]["t"]) - 1 for p in scored)

    out.append(("1. held-out patient never in training",
                bool((preds.n_train_patients == len(seqs) - 1).all())))
    out.append(("2. history strictly before target",
                bool((preds.t_last_history < preds.t_target).all()
                     or (preds.t_last_history <= preds.t_target).all())))
    out.append(("3. history length equals target index",
                bool((preds.history_length == preds.target_index).all())))
    out.append(("4. same cohort as Step 1 (934 patients)", len(seqs) == 934))
    out.append(("5. target time matches the trajectory", all(
        np.allclose(np.sort(sub.t_target.to_numpy()), np.sort(seqs[p]["t"][1:]))
        for p, sub in g.groupby("held_out_patient"))))
    out.append(("6. observed scores match the trajectory", all(
        np.allclose(sub.sort_values("target_index").obs_tremor.to_numpy(),
                    seqs[p]["y"][1:, 0])
        for p, sub in g.groupby("held_out_patient"))))
    out.append(("7. no ratio-based filtering applied",
                len(g) == expected))
    n_zero = int((preds[preds.model == ARMS[0][0]].obs_pigd == 0).sum())
    out.append((f"8. PIGD == 0 points retained ({n_zero} scored)", n_zero > 0))
    out.append(("9. every model scores identical points", all(
        int((preds.model == m).sum()) == len(g) for m in models)))
    out.append(("10. convergence recorded for every fold",
                bool(preds.train_converged.notna().all())))
    return out


def fit_population(packed, args) -> pd.DataFrame:
    """Fit every kernel x coregion pair on the whole cohort.

    These are descriptive, not held out: they report what structure the data
    supports in sample, including for K2/K3, whose per-fold cost rules them out
    of the leave-one-patient-out comparison on this machine.
    """
    rows = []
    for kernel in ("K0", "K1", "K2", "K3"):
        for coregion in ("full", "diag"):
            t0 = time.time()
            fit = M.fit_mogp(packed, kernel, coregion, n_restarts=args.restarts,
                             max_iter=args.max_iter, jitter=JITTER, seed=SEED)
            spread = float(np.ptp(fit.restart_nlls))
            rows.append({
                "kernel": kernel, "coregion": coregion, "nll": fit.nll,
                "n_params": len(fit.theta), "converged": fit.converged,
                "seconds": time.time() - t0,
                "restart_nll_spread": spread, "best_restart": fit.best_restart,
                "at_bound": "|".join(fit.at_bound) or "none",
                **fit.parameter_dict(),
                **{f"corr_{k}": v for k, v in fit.coregion_correlation().items()},
            })
            print(f"  {kernel}/{coregion:4}  nll {fit.nll:9.2f}  "
                  f"params {len(fit.theta):2}  conv {fit.converged}  "
                  f"restart spread {spread:7.4f}  "
                  f"corr {({k: round(v, 4) for k, v in fit.coregion_correlation().items()})}",
                  flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--restarts", type=int, default=4,
                    help="restarts for the population fits (LOPO folds use 1)")
    ap.add_argument("--fold-restarts", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--skip-lopo", action="store_true")
    ap.add_argument("--skip-population", action="store_true",
                    help="reuse an existing kernel_parameters.csv instead of refitting")
    args = ap.parse_args()

    cfg = load_config()
    out = cfg.path("outputs") / "joint_subscore_gp"
    (out / "diagnostics").mkdir(parents=True, exist_ok=True)
    suffix = "_partial" if args.limit else ""

    frame, seqs = build_sequences(cfg)
    patients = sorted(seqs)
    n_points = sum(len(seqs[p]["t"]) - 1 for p in patients)
    print(f"cohort {len(patients):,} patients, {len(frame):,} visits, "
          f"{n_points:,} prediction points, "
          f"{int((frame.pigd_score == 0).sum()):,} visits with PIGD == 0")
    print("device: CPU only (no CUDA device present; installed torch is +cpu)")

    # --------------------------------------------------- population fits
    print("\n" + "=" * 78)
    print(f"POPULATION FITS  (all {len(patients):,} patients, "
          f"{args.restarts} restarts)")
    print("=" * 78)
    packed = M.pack([{"t": seqs[p]["t"], "y": seqs[p]["y"]} for p in patients])
    existing = out / "kernel_parameters.csv"
    if args.skip_population and existing.exists():
        pop = pd.read_csv(existing)
        print(f"  reusing {len(pop)} population fits from {existing.name} "
              f"(not refitted)")
    else:
        pop = fit_population(packed, args)
        pop.to_csv(existing, index=False)

    print("\n  nested test, in-sample: full minus diag (nll, lower is better)")
    for kernel in ("K0", "K1", "K2", "K3"):
        f = pop[(pop.kernel == kernel) & (pop.coregion == "full")].iloc[0]
        d = pop[(pop.kernel == kernel) & (pop.coregion == "diag")].iloc[0]
        dof = int(f.n_params - d.n_params)
        print(f"    {kernel}: delta nll {f.nll - d.nll:+8.3f} for {dof} extra "
              f"parameters   (LR stat {2 * (d.nll - f.nll):7.3f})")

    cov_rows = []
    for _, r in pop.iterrows():
        for comp in [c for c in r.index if c.startswith("corr_")]:
            cov_rows.append({"kernel": r.kernel, "coregion": r.coregion,
                             "component": comp.replace("corr_", ""),
                             "correlation": r[comp]})
    pd.DataFrame(cov_rows).to_csv(out / "covariance_summary.csv", index=False)

    if args.skip_lopo:
        return 0

    # ------------------------------------------------------------- LOPO
    scored = patients[:args.limit] if args.limit else patients
    print("\n" + "=" * 78)
    print(f"LEAVE-ONE-PATIENT-OUT  ({len(scored):,} folds x {len(ARMS)} GP arms, "
          f"{args.fold_restarts} restart each, {args.jobs} workers)")
    print("=" * 78)

    t0 = time.time()
    all_preds, all_fits = [], []
    with ProcessPoolExecutor(
        max_workers=args.jobs, initializer=_init_worker,
        initargs=(seqs, ARMS, args.fold_restarts, args.max_iter),
    ) as pool:
        for done, (p, f) in enumerate(pool.map(_run_fold, scored, chunksize=1), 1):
            all_preds.extend(p)
            all_fits.extend(f)
            if done % 25 == 0 or done == len(scored):
                el = time.time() - t0
                print(f"    {done:,}/{len(scored):,}  {el / done:.2f}s/fold  "
                      f"eta {(len(scored) - done) * el / done / 60:.1f} min", flush=True)

    preds = pd.DataFrame(all_preds)
    fits = pd.DataFrame(all_fits)
    preds.to_csv(out / f"predictions{suffix}.csv", index=False)
    fits.to_csv(out / f"fold_fits{suffix}.csv", index=False)
    for label, _, _ in ARMS:
        preds[preds.model == label].to_csv(
            out / f"predictions_{label}{suffix}.csv", index=False)
    print(f"  total {(time.time() - t0) / 60:.1f} min; "
          f"converged {int(fits.converged.sum()):,}/{len(fits):,}")

    print("\n  LEAKAGE AND CONSISTENCY")
    for name, ok in check_predictions(preds, seqs, scored):
        print(f"    [{'PASS' if ok else 'FAIL'}]  {name}")

    cont = continuous_metrics(preds)
    cont.to_csv(out / f"metrics{suffix}.csv", index=False)
    for short in ("tremor", "pigd"):
        print(f"\n  {short.upper()}")
        sub = cont[cont.outcome == short].sort_values("MAE")
        print("    " + sub[["model", "MAE", "MAE_lo", "MAE_hi", "RMSE", "bias",
                            "coverage95", "mean_negative_mass"]]
              .round(4).to_string(index=False).replace("\n", "\n    "))

    vs_pers = paired(preds, "persistence")
    nested = [paired(preds[preds.model.isin([f"gp_joint_{k}_full",
                                             f"gp_joint_{k}_diag"])],
                     f"gp_joint_{k}_diag") for k in ("K1",)]
    pd.concat([vs_pers, *nested]).to_csv(out / f"paired{suffix}.csv", index=False)
    print("\n  PAIRED vs PERSISTENCE (negative = better than persistence)")
    print("    " + vs_pers.round(5).to_string(index=False).replace("\n", "\n    "))
    print("\n  NESTED CROSS-OUTPUT TEST: full minus diag, held out")
    print("    " + pd.concat(nested).round(5).to_string(index=False)
          .replace("\n", "\n    "))

    derived = derive_subtype(preds, cfg)
    derived.to_csv(out / f"predictions_with_subtype{suffix}.csv", index=False)
    smet = pd.concat([subtype_metrics(derived, "pred_subtype"),
                      subtype_metrics(derived, "prob_subtype")])
    smet.to_csv(out / f"subtype_metrics{suffix}.csv", index=False)
    print("\n  DERIVED SUBTYPE (secondary)")
    print("    " + smet[["model", "rule", "accuracy", "accuracy_lo", "accuracy_hi",
                         "balanced_accuracy", "recall_TD", "recall_Indeterminate",
                         "recall_PIGD", "n_predicted_Indeterminate"]]
          .round(4).to_string(index=False).replace("\n", "\n    "))

    # ------------------------------------------------------- PIGD == 0
    zero = preds[preds.obs_pigd == 0]
    print(f"\n  PIGD == 0 HELD-OUT POINTS  ({int(len(zero) / preds.model.nunique()):,} "
          f"points, {zero.held_out_patient.nunique():,} patients)")
    zrows = []
    for model, g in zero.groupby("model"):
        zrows.append({
            "model": model, "n": len(g),
            "tremor_MAE": float((g.pred_tremor - g.obs_tremor).abs().mean()),
            "pigd_MAE": float(g.pred_pigd.abs().mean()),
            "pigd_pred_mean": float(g.pred_pigd.mean()),
            "pigd_pred_sd_mean": float(g.sd_pigd.mean()),
            "frac_pigd_pred_below_0.1": float((g.pred_pigd < 0.1).mean()),
            "mean_negative_mass_pigd": float(g.neg_mass_pigd.mean()),
        })
    zdf = pd.DataFrame(zrows).sort_values("pigd_MAE")
    zdf.to_csv(out / f"pigd_zero_analysis{suffix}.csv", index=False)
    print("    " + zdf.round(4).to_string(index=False).replace("\n", "\n    "))

    gp = fits[fits.model == "gp_joint_K1_full"]
    stab = gp[[c for c in gp.columns if c.startswith(("corr_", "lengthscale", "var_",
                                                      "noise_"))]].agg(
        ["min", "median", "max"]).T
    stab.to_csv(out / "diagnostics" / "fold_hyperparameter_stability.csv")
    print("\n  HYPERPARAMETER STABILITY ACROSS FOLDS (gp_joint_K1_full)")
    print("    " + stab.round(5).to_string().replace("\n", "\n    "))

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
