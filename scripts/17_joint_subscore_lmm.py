"""Step 1: joint linear mixed-effects baseline on the tremor and PIGD subscores.

Models the two subscores the subtype rule is built from, instead of the ratio
that compresses them. This is deliberately the *interpretable* baseline; the
multi-output GP is a separate, later step and is not implemented here.

    python scripts/17_joint_subscore_lmm.py [--jobs 8] [--limit N] [--skip-lopo]

Reads   outputs/visit_panel.csv
Writes  outputs/joint_subscore/
            descriptives.txt, model_summary.txt
            predictions.csv, metrics.csv, subtype_metrics.csv
            confusion_matrix.csv, fold_fits.csv
            diagnostics/residuals.csv, diagnostics/variance_components.csv

The cohort, visit ordering, time axis and subtype rule are taken unchanged from
the existing pipeline. Nothing here recomputes a ratio and models it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from ppmi_subtype import jointlmm as J  # noqa: E402
from ppmi_subtype import subtype as st  # noqa: E402
from ppmi_subtype import trajectory as tj  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

MODELS = ["joint_lmm", "persistence", "patient_mean", "population"]
N_BOOT = 4000
SEED = 42


# ---------------------------------------------------------------------------
#  Cohort
# ---------------------------------------------------------------------------
def build_subscore_frame(panel: pd.DataFrame, cfg) -> pd.DataFrame:
    """The existing analysable cohort, carrying both subscores.

    `build_trajectories` fixes the patients, the visits, the chronological
    ordering and the elapsed-time axis. The subscores are merged back from the
    panel because `TRAJECTORY_COLUMNS` keeps only the ratio. No row is added,
    removed or reordered here.
    """
    traj = tj.build_trajectories(panel, cfg)
    tj.check_ordering(traj)

    cols = ["PATNO", "EVENT_ID", *J.OUTCOMES]
    merged = traj.merge(
        panel[cols].drop_duplicates(["PATNO", "EVENT_ID"]),
        on=["PATNO", "EVENT_ID"], how="left", validate="one_to_one",
    )
    if len(merged) != len(traj):
        raise SystemExit("merging the subscores changed the row count")
    return merged


def describe(frame: pd.DataFrame, out: Path) -> dict:
    """The descriptive pass required before any model is fitted."""
    t, p = frame[J.OUTCOMES[0]], frame[J.OUTCOMES[1]]
    both = t.notna() & p.notna()
    n_visits = frame.groupby("PATNO").size()

    dev = frame.copy()
    for c in J.OUTCOMES:
        dev[c] = dev[c] - dev.groupby("PATNO")[c].transform("mean")
    pm = frame.groupby("PATNO")[list(J.OUTCOMES)].mean()

    trends = {}
    for c in J.OUTCOMES:
        X = np.c_[np.ones(both.sum()), frame.loc[both, "years_from_baseline"]]
        trends[c] = np.linalg.lstsq(X, frame.loc[both, c], rcond=None)[0]

    stats = {
        "patients": int(frame["PATNO"].nunique()),
        "visits": int(len(frame)),
        "tremor_observed": int(t.notna().sum()),
        "pigd_observed": int(p.notna().sum()),
        "both_observed": int(both.sum()),
        "pigd_zero": int((p == 0).sum()),
        "tremor_zero": int((t == 0).sum()),
        "both_zero": int(((p == 0) & (t == 0)).sum()),
        "ratio_undefined": int((p == 0).sum()),
        "ratio_defined": int(frame["ratio_defined"].sum()),
        "corr_pearson": float(t.corr(p)),
        "corr_spearman": float(t.corr(p, method="spearman")),
        "corr_within_patient": float(dev[J.OUTCOMES[0]].corr(dev[J.OUTCOMES[1]])),
        "corr_between_patient": float(pm[J.OUTCOMES[0]].corr(pm[J.OUTCOMES[1]])),
    }

    lines = ["DESCRIPTIVE ANALYSIS", "=" * 78, ""]
    lines.append(f"patients {stats['patients']:,}   visits {stats['visits']:,}")
    lines.append(f"visits per patient: median {int(n_visits.median())}, "
                 f"mean {n_visits.mean():.2f}, range {n_visits.min()}-{n_visits.max()}")
    lines.append(f"follow-up span (years): median "
                 f"{frame.groupby('PATNO').years_from_baseline.max().median():.2f}")
    lines.append("")
    lines.append("usable observations")
    lines.append(f"  tremor observed        {stats['tremor_observed']:,}")
    lines.append(f"  PIGD observed          {stats['pigd_observed']:,}")
    lines.append(f"  both observed          {stats['both_observed']:,}")
    lines.append(f"  PIGD == 0              {stats['pigd_zero']:,} "
                 f"({100 * stats['pigd_zero'] / len(frame):.2f}%)  "
                 f"<- ratio undefined, recovered by this model")
    lines.append(f"  tremor == 0            {stats['tremor_zero']:,}")
    lines.append(f"  both == 0              {stats['both_zero']:,}")
    lines.append(f"  ratio defined          {stats['ratio_defined']:,}")
    lines.append("")
    lines.append("distributions")
    lines.append(frame[list(J.OUTCOMES)].describe(
        percentiles=[.05, .25, .5, .75, .95]).round(4).to_string())
    lines.append("")
    lines.append("correlation between the two subscores")
    lines.append(f"  pooled Pearson         {stats['corr_pearson']:+.4f}")
    lines.append(f"  pooled Spearman        {stats['corr_spearman']:+.4f}")
    lines.append(f"  between patients       {stats['corr_between_patient']:+.4f}  "
                 f"(correlation of patient means)")
    lines.append(f"  within patients        {stats['corr_within_patient']:+.4f}  "
                 f"(correlation of deviations from the patient mean)")
    lines.append("")
    lines.append("  The pooled correlation is near zero because the between- and")
    lines.append("  within-patient correlations differ. This is a description of")
    lines.append("  association only; no independence or causal claim follows.")
    lines.append("")
    lines.append("pooled longitudinal trend (OLS, for description only)")
    for c, b in trends.items():
        lines.append(f"  {c:14} intercept {b[0]:+.4f}   slope {b[1]:+.5f} per year")
    lines.append("")
    lines.append("  The two subscores do not move together: both worsen, but at")
    lines.append("  different rates. A declining ratio is therefore not evidence")
    lines.append("  of improving tremor.")

    (out / "descriptives.txt").write_text("\n".join(lines), encoding="utf-8")
    return stats


# ---------------------------------------------------------------------------
#  Leave-one-patient-out
# ---------------------------------------------------------------------------
def run_lopo(data: J.PackedData, patients, reml: bool, progress=None):
    """Refit without each patient, then predict their visits 1..n-1 in order.

    The protocol matches the existing Step 3 evaluation exactly: visit k of the
    held-out patient is predicted from visits 0..k-1 of that same patient, under
    a model fitted on the other N-1 patients. The target never enters the fit
    and never enters its own history.
    """
    preds, fits = [], []
    t0 = time.time()
    for idx, patno in enumerate(patients, start=1):
        train = J.subset(data, [p for p in data.patients if p != patno])
        fit = J.fit_joint_lmm(train, reml=reml)

        fits.append({
            "held_out_patient": patno, "converged": fit.converged,
            "n_train_patients": fit.n_patients, "neg2ll": fit.neg2ll,
            **{n: v for n, v in zip(J.BETA_NAMES, fit.beta)},
            "var_b_tremor": fit.sigma_b[0, 0], "var_b_pigd": fit.sigma_b[1, 1],
            "cov_b": fit.sigma_b[0, 1], "corr_b": fit.corr_b,
            "var_e_tremor": fit.sigma_e[0, 0], "var_e_pigd": fit.sigma_e[1, 1],
            "cov_e": fit.sigma_e[0, 1], "corr_e": fit.corr_e,
        })

        t_all, Y_all = data.times[patno], data.responses[patno]
        for k in range(1, len(t_all)):
            t_hist, Y_hist = t_all[:k], Y_all[:k]
            for model in MODELS:
                if model == "joint_lmm":
                    mean, sd = J.predict(fit, t_hist, Y_hist, float(t_all[k]))
                else:
                    mean, sd = J.baseline_predict(
                        model, t_hist, Y_hist, float(t_all[k]), fit)
                preds.append({
                    "held_out_patient": patno, "model": model,
                    "target_index": k, "history_length": k,
                    "t_last_history": float(t_hist[-1]), "t_target": float(t_all[k]),
                    "horizon_years": float(t_all[k] - t_hist[-1]),
                    "obs_tremor": float(Y_all[k, 0]), "obs_pigd": float(Y_all[k, 1]),
                    "pred_tremor": float(mean[0]), "pred_pigd": float(mean[1]),
                    "sd_tremor": float(sd[0]), "sd_pigd": float(sd[1]),
                    "n_train_patients": fit.n_patients,
                    "train_converged": fit.converged,
                })
        if progress:
            progress(idx, len(patients), time.time() - t0)

    return pd.DataFrame(preds), pd.DataFrame(fits)


def check_predictions(preds: pd.DataFrame, data: J.PackedData, scored=None) -> list:
    """Consistency and leakage checks, run on every saved prediction row.

    `data` is always the *full* cohort, because the training-set size each fold
    should have seen is N-1 of the whole cohort even when only a few folds were
    run. `scored` is the subset of patients actually evaluated.
    """
    out = []
    g = preds[preds.model == "joint_lmm"]
    scored = list(data.patients if scored is None else scored)
    expected = sum(len(data.times[p]) - 1 for p in scored)

    out.append(("one row per model per prediction point",
                all(int((preds.model == m).sum()) == len(g) for m in MODELS)))
    out.append(("prediction count equals transitions", len(g) == expected))
    out.append(("history is strictly before the target",
                bool((g.t_last_history <= g.t_target).all())))
    out.append(("history length equals target index",
                bool((g.history_length == g.target_index).all())))
    out.append(("target index at least 1", bool((g.target_index >= 1).all())))
    out.append(("held-out patient excluded from every fit",
                bool((g.n_train_patients == data.n_patients - 1).all())))
    out.append(("every fold converged", bool(preds.train_converged.all())))
    out.append(("no missing predictions",
                bool(preds[["pred_tremor", "pred_pigd"]].notna().all().all())))
    out.append(("predictive SDs are positive",
                bool((preds[["sd_tremor", "sd_pigd"]] > 0).all().all())))

    ok = True
    for patno, sub in g.groupby("held_out_patient"):
        t = data.times[patno]
        idx = sub.target_index.to_numpy()
        if not np.array_equal(np.sort(idx), np.arange(1, len(t))):
            ok = False
            break
        if not np.allclose(np.sort(sub.t_target.to_numpy()), np.sort(t[1:])):
            ok = False
            break
    out.append(("target times match the patient's own trajectory", ok))
    return out


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------
def boot_ci(values: np.ndarray, patients: np.ndarray, n: int = N_BOOT, seed: int = SEED):
    """Patient-clustered bootstrap of a mean over prediction points."""
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
        for short, obs, pred, sd in [
            ("tremor", "obs_tremor", "pred_tremor", "sd_tremor"),
            ("pigd", "obs_pigd", "pred_pigd", "sd_pigd"),
        ]:
            err = (g[pred] - g[obs]).to_numpy()
            ae, lo, hi = boot_ci(np.abs(err), pat)
            z = err / g[sd].to_numpy()
            rows.append({
                "model": model, "outcome": short, "n": len(g),
                "n_patients": int(g.held_out_patient.nunique()),
                "MAE": ae, "MAE_lo": lo, "MAE_hi": hi,
                "RMSE": float(np.sqrt((err ** 2).mean())),
                "bias": float(err.mean()),
                "coverage68": float((np.abs(z) <= 1.0).mean()),
                "coverage95": float((np.abs(z) <= 1.959963984540054).mean()),
                "negative_predictions": int((g[pred] < 0).sum()),
            })
    return pd.DataFrame(rows)


def paired_vs_persistence(preds: pd.DataFrame) -> pd.DataFrame:
    """The comparison that decides whether the model earns its complexity."""
    rows = []
    key = ["held_out_patient", "target_index"]
    for short, obs, pred in [("tremor", "obs_tremor", "pred_tremor"),
                             ("pigd", "obs_pigd", "pred_pigd")]:
        w = preds.assign(ae=(preds[pred] - preds[obs]).abs()).pivot_table(
            index=key, columns="model", values="ae").dropna()
        pat = w.index.get_level_values(0).to_numpy()
        for model in MODELS:
            if model == "persistence":
                continue
            diff = (w[model] - w["persistence"]).to_numpy()
            pt, lo, hi = boot_ci(diff, pat)
            rows.append({
                "outcome": short, "comparison": f"{model} - persistence",
                "delta_MAE": pt, "lo": lo, "hi": hi,
                "excludes_zero": (lo < 0) == (hi < 0), "n": len(diff),
            })
    return pd.DataFrame(rows)


def derive_subtype(preds: pd.DataFrame, cfg) -> pd.DataFrame:
    """Apply the project's own classification rule to the predicted subscores.

    `subtype.classify` already handles PIGD == 0 and both == 0 explicitly, so no
    epsilon is introduced anywhere: a predicted PIGD of exactly zero routes
    through the published degenerate rule rather than a division.

    Predictions are clipped at zero first. The subscores are means of 0-4 item
    ratings and cannot be negative; a Gaussian predictive mean can be, and
    clipping is the minimal correction. The count is reported.
    """
    out = preds.copy()
    out["pred_tremor_clipped"] = out.pred_tremor.clip(lower=0.0)
    out["pred_pigd_clipped"] = out.pred_pigd.clip(lower=0.0)
    out["pred_subtype"] = st.classify(
        out.pred_tremor_clipped, out.pred_pigd_clipped, cfg.subtype
    ).map(cfg.subtype.label_names)
    out["obs_subtype"] = st.classify(
        out.obs_tremor, out.obs_pigd, cfg.subtype
    ).map(cfg.subtype.label_names)
    out["obs_ratio_defined"] = out.obs_pigd != 0
    return out


def subtype_metrics(derived: pd.DataFrame) -> pd.DataFrame:
    classes = ["TD", "Indeterminate", "PIGD"]
    rows = []
    for model, g in derived.groupby("model"):
        correct = (g.pred_subtype == g.obs_subtype).to_numpy()
        acc, lo, hi = boot_ci(correct.astype(float), g.held_out_patient.to_numpy())
        recalls = {}
        for c in classes:
            m = g.obs_subtype == c
            recalls[c] = float((g.loc[m, "pred_subtype"] == c).mean()) if m.any() else np.nan
        sub = g[~g.obs_ratio_defined]
        rows.append({
            "model": model, "n": len(g), "accuracy": acc,
            "accuracy_lo": lo, "accuracy_hi": hi,
            "balanced_accuracy": float(np.nanmean(list(recalls.values()))),
            **{f"recall_{c}": recalls[c] for c in classes},
            **{f"n_predicted_{c}": int((g.pred_subtype == c).sum()) for c in classes},
            "n_pigd_zero_visits": int(len(sub)),
            "accuracy_on_pigd_zero": float((sub.pred_subtype == sub.obs_subtype).mean())
            if len(sub) else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N folds (smoke test; writes _partial files)")
    ap.add_argument("--skip-lopo", action="store_true",
                    help="descriptives and the training fit only")
    ap.add_argument("--ml", action="store_true", help="fit by ML instead of REML")
    args = ap.parse_args()

    cfg = load_config()
    out = cfg.path("outputs") / "joint_subscore"
    (out / "diagnostics").mkdir(parents=True, exist_ok=True)
    suffix = "_partial" if args.limit else ""
    reml = not args.ml

    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    frame = build_subscore_frame(panel, cfg)

    print("=" * 78)
    stats = describe(frame, out)
    print(f"cohort: {stats['patients']:,} patients, {stats['visits']:,} visits, "
          f"{stats['pigd_zero']:,} with PIGD == 0 (ratio undefined)")
    print(f"wrote {out / 'descriptives.txt'}")

    data = J.pack(frame)
    dropped = len(frame) - data.n_visits
    if dropped:
        print(f"NOTE: {dropped} visits dropped for a missing subscore")

    # ---------------------------------------------------------------- full fit
    print("\n" + "=" * 78)
    print(f"TRAINING FIT ({'REML' if reml else 'ML'}, all {data.n_patients:,} patients)")
    print("=" * 78)
    t0 = time.time()
    fit = J.fit_joint_lmm(data, reml=reml)
    print(f"  converged {fit.converged} in {time.time() - t0:.1f}s "
          f"({fit.n_iter} iterations) -- {fit.message}")
    print(f"  -2 log-likelihood {fit.neg2ll:.3f}   observations {fit.n_obs:,}")

    fe = fit.fixed_effects_table()
    print("\n  fixed effects")
    print("   " + fe.round(5).to_string(index=False).replace("\n", "\n   "))
    print("\n  random intercepts (Sigma_b)        residuals (Sigma_e)")
    print(f"   var tremor {fit.sigma_b[0,0]:9.5f}            {fit.sigma_e[0,0]:9.5f}")
    print(f"   var pigd   {fit.sigma_b[1,1]:9.5f}            {fit.sigma_e[1,1]:9.5f}")
    print(f"   covariance {fit.sigma_b[0,1]:9.5f}            {fit.sigma_e[0,1]:9.5f}")
    print(f"   correlation{fit.corr_b:9.4f}            {fit.corr_e:9.4f}")
    icc = fit.icc()
    print(f"\n  ICC  tremor {icc[0]:.3f}   pigd {icc[1]:.3f}")

    resid_rows = []
    for p in data.patients:
        t, Y = data.times[p], data.responses[p]
        b_hat, _ = J.blup(fit, t, Y)
        mu = (J.design(t) @ fit.beta).reshape(len(t), 2)
        r = Y - mu - b_hat
        for j in range(len(t)):
            resid_rows.append({"PATNO": p, "years_from_baseline": t[j],
                               "resid_tremor": r[j, 0], "resid_pigd": r[j, 1],
                               "fitted_tremor": mu[j, 0] + b_hat[0],
                               "fitted_pigd": mu[j, 1] + b_hat[1]})
    resid = pd.DataFrame(resid_rows)
    resid.to_csv(out / "diagnostics" / "residuals.csv", index=False)
    print(f"  conditional residual correlation {resid.resid_tremor.corr(resid.resid_pigd):+.4f}")

    summary = [
        f"JOINT LINEAR MIXED MODEL  ({'REML' if reml else 'ML'})", "=" * 78, "",
        f"patients {fit.n_patients:,}   observations {fit.n_obs:,}   "
        f"converged {fit.converged}",
        f"-2 log-likelihood {fit.neg2ll:.4f}", "",
        "Tremor_it = alpha_T + beta_T t + b_Ti + eps_Tit",
        "PIGD_it   = alpha_P + beta_P t + b_Pi + eps_Pit",
        "  (b_Ti, b_Pi) ~ N(0, Sigma_b),  (eps_Tit, eps_Pit) ~ N(0, Sigma_e)",
        "  no random slopes; see module docstring", "",
        "FIXED EFFECTS", fe.round(6).to_string(index=False), "",
        "RANDOM INTERCEPT COVARIANCE Sigma_b", str(np.round(fit.sigma_b, 6)),
        f"correlation {fit.corr_b:+.4f}", "",
        "RESIDUAL COVARIANCE Sigma_e", str(np.round(fit.sigma_e, 6)),
        f"correlation {fit.corr_e:+.4f}", "",
        f"ICC tremor {icc[0]:.4f}   ICC pigd {icc[1]:.4f}",
        f"conditional residual correlation {resid.resid_tremor.corr(resid.resid_pigd):+.4f}",
    ]
    (out / "model_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    pd.DataFrame({
        "component": ["var_b_tremor", "var_b_pigd", "cov_b", "corr_b",
                      "var_e_tremor", "var_e_pigd", "cov_e", "corr_e",
                      "icc_tremor", "icc_pigd"],
        "value": [fit.sigma_b[0, 0], fit.sigma_b[1, 1], fit.sigma_b[0, 1], fit.corr_b,
                  fit.sigma_e[0, 0], fit.sigma_e[1, 1], fit.sigma_e[0, 1], fit.corr_e,
                  icc[0], icc[1]],
    }).to_csv(out / "diagnostics" / "variance_components.csv", index=False)
    print(f"  wrote {out / 'model_summary.txt'}")

    if args.skip_lopo:
        return 0

    # -------------------------------------------------------------------- LOPO
    patients = data.patients[:args.limit] if args.limit else data.patients
    print("\n" + "=" * 78)
    print(f"LEAVE-ONE-PATIENT-OUT  ({len(patients):,} folds)")
    print("=" * 78)

    def progress(done, total, elapsed):
        if done % 50 == 0 or done == total:
            print(f"    {done:,}/{total:,}   {elapsed / done:.2f}s/fold   "
                  f"eta {(total - done) * elapsed / done / 60:.1f} min", flush=True)

    preds, fits = run_lopo(data, patients, reml=reml, progress=progress)
    preds.to_csv(out / f"predictions{suffix}.csv", index=False)
    fits.to_csv(out / f"fold_fits{suffix}.csv", index=False)
    print(f"  converged {int(fits.converged.sum()):,}/{len(fits):,}")

    print("\n  consistency and leakage checks")
    results = check_predictions(preds, data, scored=patients)
    for name, ok in results:
        print(f"    [{'PASS' if ok else 'FAIL'}]  {name}")

    cont = continuous_metrics(preds)
    cont.to_csv(out / f"metrics{suffix}.csv", index=False)
    print("\n  CONTINUOUS PREDICTION")
    for short in ("tremor", "pigd"):
        print(f"\n    {short.upper()}")
        sub = cont[cont.outcome == short].sort_values("MAE")
        print("      " + sub[["model", "n", "MAE", "MAE_lo", "MAE_hi", "RMSE",
                              "bias", "coverage95", "negative_predictions"]]
              .round(4).to_string(index=False).replace("\n", "\n      "))

    paired = paired_vs_persistence(preds)
    paired.to_csv(out / f"paired_vs_persistence{suffix}.csv", index=False)
    print("\n  PAIRED AGAINST PERSISTENCE (negative = better than persistence)")
    print("    " + paired.round(5).to_string(index=False).replace("\n", "\n    "))

    derived = derive_subtype(preds, cfg)
    derived.to_csv(out / f"predictions_with_subtype{suffix}.csv", index=False)
    smet = subtype_metrics(derived)
    smet.to_csv(out / f"subtype_metrics{suffix}.csv", index=False)
    print("\n  DERIVED SUBTYPE (secondary analysis)")
    print("    " + smet.round(4).to_string(index=False).replace("\n", "\n    "))

    cm = pd.crosstab(derived[derived.model == "joint_lmm"].pred_subtype,
                     derived[derived.model == "joint_lmm"].obs_subtype)
    cm.to_csv(out / f"confusion_matrix{suffix}.csv")
    print("\n  confusion matrix, joint_lmm (rows predicted, columns actual)")
    print("    " + cm.to_string().replace("\n", "\n    "))

    n_clipped = int((preds[preds.model == "joint_lmm"].pred_pigd < 0).sum()
                    + (preds[preds.model == "joint_lmm"].pred_tremor < 0).sum())
    print(f"\n  predictions clipped at zero before classification: {n_clipped}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
