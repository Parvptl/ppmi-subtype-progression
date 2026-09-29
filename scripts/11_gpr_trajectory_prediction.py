"""Step 3 - Gaussian process trajectory prediction on the TD/PIGD ratio.

The question is whether a patient's future ratio can be predicted from their
own previous observations, and whether a flexible Gaussian process adds
anything over the linear hierarchical growth model the project already has.

    python scripts/11_gpr_trajectory_prediction.py --stage cohorts
    python scripts/11_gpr_trajectory_prediction.py --stage lopo --jobs 12
    python scripts/11_gpr_trajectory_prediction.py --stage report

`--stage all` runs the three in order. They are separable because the middle
one takes hours and the other two take seconds, so a change to a figure does
not mean refitting 2,861 folds.

Writes to outputs/step3_gpr/. Nothing here touches outputs/loo, outputs/hbayes,
outputs/model or any experiments/ directory.
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

from ppmi_subtype import gpr, gpr_loo, hbayes, trajectory as tj  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

OUT_SUBDIR = "step3_gpr"
COHORTS = ["unfiltered", "A4", "D1", "E2"]

lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def section(title: str) -> None:
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


def frame(df: pd.DataFrame, index: bool = False) -> str:
    return "    " + df.to_string(index=index).replace("\n", "\n    ")


def settings_from(cfg) -> dict:
    s = cfg.subtype
    return {
        "n_restarts": int(cfg.gpr["n_restarts"]),
        "seed": int(cfg.seed),
        "max_iter": int(cfg.gpr["max_iter"]),
        "jitter": float(cfg.gpr["jitter"]),
        "td_threshold": float(s.td_threshold),
        "pigd_threshold": float(s.pigd_threshold),
        # The growth model keeps its own defaults; Step 3 compares against the
        # existing model, so its sampler settings are not retuned here.
        "hbayes_draws": 4000,
        "hbayes_burn_in": 1000,
        "hbayes_pred_draws": 400,
    }


def cohort_of(traj, cfg, name: str):
    return tj.build_cohort(traj, cfg, criterion=None if name == "unfiltered" else name)


# ---------------------------------------------------------------------------
#  Stage 1 - trajectories, extreme profiles, cohorts, equivalence check
# ---------------------------------------------------------------------------
def stage_cohorts(cfg, out: Path) -> None:
    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    traj = tj.build_trajectories(panel, cfg)
    tj.check_ordering(traj)

    section("1. TRAJECTORIES")
    say("  One row per labelled visit, ordered in time within each patient.")
    say("  Visits whose ratio is undefined are kept and flagged, not dropped.")
    say()
    say(f"  patients                            {traj['PATNO'].nunique():,}")
    say(f"  labelled visits                     {len(traj):,}")
    finite = int(traj["ratio_defined"].sum())
    say(f"  with a finite ratio                 {finite:,}  ({finite/len(traj):.1%})")
    undefined = traj[~traj["ratio_defined"]]
    say(f"  undefined (PIGD subscore is zero)   {len(undefined):,}  "
        f"({len(undefined)/len(traj):.1%})")
    say(f"    affecting                         {undefined['PATNO'].nunique():,} patients")
    say(f"    their recorded subtypes           "
        f"{undefined['subtype_name'].value_counts().to_dict()}")
    zero = int((traj["ratio"] == 0).sum())
    say(f"  ratio exactly zero (no tremor)      {zero:,}  ({zero/len(traj):.1%})")
    say()
    say("  The undefined visits are overwhelmingly TD, so a model defined on the")
    say("  ratio cannot see a non-random slice of the data. That is a property of")
    say("  the ratio, not a choice made here, and it is carried as a limitation.")

    summary = tj.trajectory_summary(traj)
    say()
    say("  Labelled visits per patient:")
    say(f"    {summary['n_labelled'].describe()[['mean','min','50%','max']].round(2).to_dict()}")
    say("  Finite-ratio visits per patient:")
    say(f"    {summary['n_finite'].value_counts().sort_index().to_dict()}")

    # ------------------------------------------------------- extreme profiles
    section("2. PERSISTENTLY EXTREME PROFILES")
    say("  Every criterion is a statement about a patient across visits, never")
    say("  about a single visit. Definitions are in config.yaml.")
    say()
    extreme = tj.extreme_patient_summary(traj, cfg)
    specs = cfg.trajectory["criteria"]
    for name in sorted(specs):
        spec = specs[name]
        detail = ", ".join(f"{k}={v}" for k, v in spec.items() if k != "type")
        say(f"    {name}  {spec['type']}  ({detail})")

    say()
    say("  Consequences of each. 'never changed' counts patients whose recorded")
    say("  subtype is the same at every labelled visit.")
    say()
    rows = []
    for name in ["unfiltered", *sorted(specs)]:
        cohort = cohort_of(traj, cfg, name)
        rows.append(cohort.describe())
    comparison = pd.DataFrame(rows)
    say(frame(comparison))

    say()
    say("  Read the removal counts against the never-changed column. A criterion")
    say("  that mostly removes patients who never changed subtype makes the")
    say("  remaining prediction task a different one, which is why the unfiltered")
    say("  cohort stays the reference and every baseline is recomputed per cohort.")

    removed = {}
    for name in sorted(specs):
        c = cohort_of(traj, cfg, name)
        removed[name] = c.removed_patients
        say()
        say(f"  {name}: removes {len(c.removed_patients)} patients "
            f"({len(c.removed_patients)/traj['PATNO'].nunique():.1%} of the cohort)")

    # ------------------------------------------------- K0 vs the growth model
    section("3. K0 REPRODUCES THE HIERARCHICAL GROWTH MODEL")
    say("  The growth model is a Gaussian process whose kernel is constant plus")
    say("  linear plus noise:")
    say()
    say("      Cov(R_it, R_it') = tau0^2 + tau1^2 t t' + sigma^2 [t = t']")
    say()
    say("  so K0 is that model in function-space form. If the two disagree, one")
    say("  of the implementations is wrong and nothing downstream is meaningful.")

    cohort = cohort_of(traj, cfg, "unfiltered")
    data = gpr_loo.prepare(cohort)

    labelled = panel[panel["subtype"].notna()]
    growth = hbayes.build_growth_data(labelled)
    post = hbayes.fit_growth_model(growth, seed=cfg.seed)
    tau0_2 = float(post.tau0_2.mean())
    tau1_2 = float(post.tau1_2.mean())
    sigma2 = float(post.sigma2.mean())
    beta = post.beta.mean(axis=0)

    k0_analytic = gpr.GPRFit(
        "K0", gpr.growth_model_params(tau0_2, tau1_2, sigma2), beta, 0.0,
        growth.n_patients, growth.n_obs, True, 1,
    )

    # Compared in relative terms. The two routes accumulate rounding
    # differently --- a Cholesky solve against a 2x2 explicit inverse --- so the
    # absolute gap grows with the size of the ratio, and ratios here reach 11.4.
    # A relative bound of 1e-7 is still six orders of magnitude below anything
    # that could change a conclusion, and it does not quietly tighten or loosen
    # with the scale of the data.
    worst_mean = worst_sd = 0.0
    n_checked = 0
    for patno in data.patients[:150]:
        t, y = data.times[patno], data.ratios[patno]
        for k in range(1, len(y)):
            got = gpr.predict_one(k0_analytic, t[:k], y[:k], float(t[k]))
            want = gpr.growth_model_predictive(
                beta, tau0_2, tau1_2, sigma2, t[:k], y[:k], float(t[k])
            )
            worst_mean = max(worst_mean, abs(got[0] - want[0]) / max(abs(want[0]), 1.0))
            worst_sd = max(worst_sd, abs(got[1] - want[1]) / max(abs(want[1]), 1.0))
            n_checked += 1

    say()
    say(f"  (a) Same predictive, two derivations, on {n_checked:,} real prediction points:")
    say(f"        largest relative difference in mean  {worst_mean:.3e}")
    say(f"        largest relative difference in SD    {worst_sd:.3e}")
    if max(worst_mean, worst_sd) > 1e-7:
        raise SystemExit(
            "K0 does not reproduce the growth model's predictive distribution. "
            "Stopping: no GPR result can be trusted until this is understood."
        )
    say("        PASS")

    sequences = data.sequences()
    fitted = gpr.fit_gpr(sequences, "K0", n_restarts=cfg.gpr["n_restarts"],
                         seed=cfg.seed, jitter=cfg.gpr["jitter"])
    say()
    say("  (b) Fitting K0 by marginal likelihood, against the Gibbs sampler's")
    say("      posterior means on the same patients. Two different estimators, so")
    say("      they should be close rather than identical:")
    say()
    compare = pd.DataFrame(
        [
            {"parameter": "beta0 (population baseline)", "K0 (marginal likelihood)": fitted.beta[0],
             "hbayes (posterior mean)": beta[0]},
            {"parameter": "beta1 (progression, /yr)", "K0 (marginal likelihood)": fitted.beta[1],
             "hbayes (posterior mean)": beta[1]},
            {"parameter": "tau0 (SD of patient baselines)",
             "K0 (marginal likelihood)": np.sqrt(fitted.params["var_const"]),
             "hbayes (posterior mean)": np.sqrt(tau0_2)},
            {"parameter": "tau1 (SD of patient slopes)",
             "K0 (marginal likelihood)": np.sqrt(fitted.params["var_linear"]),
             "hbayes (posterior mean)": np.sqrt(tau1_2)},
            {"parameter": "sigma (residual SD)",
             "K0 (marginal likelihood)": np.sqrt(fitted.params["noise"]),
             "hbayes (posterior mean)": np.sqrt(sigma2)},
        ]
    ).round(4)
    say(frame(compare))

    # ---------------------------------------------------------------- outputs
    out.mkdir(parents=True, exist_ok=True)
    traj.to_csv(out / "patient_trajectories.csv", index=False)
    summary.to_csv(out / "trajectory_summary.csv", index=False)
    extreme.to_csv(out / "extreme_patient_summary.csv", index=False)
    comparison.to_csv(out / "cohort_comparison.csv", index=False)

    primary = cfg.trajectory["primary"]
    cohort_of(traj, cfg, primary).traj.to_csv(
        out / "filtered_patient_trajectories.csv", index=False
    )

    section("4. FILES WRITTEN")
    for name in ("patient_trajectories.csv", "trajectory_summary.csv",
                 "extreme_patient_summary.csv", "cohort_comparison.csv",
                 "filtered_patient_trajectories.csv"):
        say(f"  {out / name}")
    say()
    say(f"  filtered_patient_trajectories.csv uses the primary criterion, {primary}.")


# ---------------------------------------------------------------------------
#  Stage 2 - leave-one-patient-out
# ---------------------------------------------------------------------------
def stage_lopo(cfg, out: Path, cohorts: list, kernels: list, jobs: int,
               limit: int | None) -> None:
    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    traj = tj.build_trajectories(panel, cfg)
    settings = settings_from(cfg)
    min_finite = int(cfg.trajectory["min_finite_visits_multistep"])

    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for name in cohorts:
        cohort = cohort_of(traj, cfg, name)
        data = gpr_loo.prepare(cohort)
        patients = data.patients[:limit] if limit else data.patients

        print(f"\n=== {name}: {len(patients):,} folds, "
              f"{cohort.n_prediction_points:,} prediction points, {jobs} worker(s)")

        t0 = time.time()

        def progress(done, total, _t0=t0, _n=name):
            if done % 25 == 0 or done == total:
                rate = (time.time() - _t0) / done
                left = rate * (total - done)
                print(f"    {_n}: {done:,}/{total:,} folds  "
                      f"{rate:.1f}s/fold  ~{left/60:.0f} min left", flush=True)

        # Checkpointed: a cohort takes hours, and an interruption used to lose
        # every completed fold. Resuming skips folds already on disk.
        predictions, fits, steps = gpr_loo.run_lopo(
            data, kernels, settings, patients=patients,
            with_hbayes=True, with_baselines=True, multistep=True,
            min_finite=min_finite, jobs=jobs, progress=progress,
            checkpoint=(out / "checkpoints" / f"{name}") if limit is None else None,
        )

        predictions.to_csv(out / f"gpr_predictions_{name}.csv", index=False)
        fits.to_csv(out / f"gpr_fold_fits_{name}.csv", index=False)
        steps.to_csv(out / f"trajectory_predictions_{name}.csv", index=False)

        print(f"    {name}: done in {(time.time()-t0)/60:.1f} min, "
              f"{len(predictions):,} prediction rows")

        if limit is None:
            failures = [
                label for label, ok in
                gpr_loo.check_predictions(predictions, data, name) if not ok
            ]
            if failures:
                raise SystemExit(
                    f"{name}: leakage or consistency checks failed: {failures}"
                )
            print(f"    {name}: all consistency checks passed")

    print(f"\nall cohorts finished in {(time.time()-started)/60:.1f} min")


# ---------------------------------------------------------------------------
#  Stage 3 - metrics and report
# ---------------------------------------------------------------------------
def stage_report(cfg, out: Path, cohorts: list) -> None:
    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    traj = tj.build_trajectories(panel, cfg)

    available = [c for c in cohorts if (out / f"gpr_predictions_{c}.csv").exists()]
    if not available:
        raise SystemExit(
            f"no prediction files in {out}; run --stage lopo first"
        )

    metrics, stratified, trajectories, confusions = [], [], [], {}
    for name in available:
        preds = pd.read_csv(out / f"gpr_predictions_{name}.csv")
        steps = pd.read_csv(out / f"trajectory_predictions_{name}.csv")
        fits = pd.read_csv(out / f"gpr_fold_fits_{name}.csv")

        metrics.append(gpr_loo.metrics_table(preds, name))
        stratified.append(gpr_loo.stratified_metrics(preds, name, "history_length"))
        stratified.append(
            gpr_loo.stratified_metrics(
                preds, name, "horizon_years", bins=[0, 1.05, 2.05, 3.05, 5.05, 30]
            )
        )
        if not steps.empty:
            trajectories.append(gpr_loo.trajectory_metrics(steps, name))
        confusions[name] = {m: gpr_loo.confusion(g) for m, g in preds.groupby("model")}

        if name == available[0]:
            _report_headline(cfg, name, preds, steps, fits, traj)

    all_metrics = pd.concat(metrics, ignore_index=True)
    all_metrics.to_csv(out / "model_comparison.csv", index=False)
    all_metrics[~all_metrics["model"].isin(gpr_loo.BASELINES)].to_csv(
        out / "gpr_metrics.csv", index=False
    )
    all_metrics[all_metrics["model"].isin(gpr_loo.BASELINES)].to_csv(
        out / "baseline_metrics.csv", index=False
    )
    pd.concat(stratified, ignore_index=True).to_csv(
        out / "stratified_metrics.csv", index=False
    )
    if trajectories:
        pd.concat(trajectories, ignore_index=True).to_csv(
            out / "trajectory_metrics.csv", index=False
        )

    section("FILTERED VERSUS UNFILTERED")
    say("  Each cohort is scored on its own prediction points with its own")
    say("  baselines. A model's MAE in one cohort is not comparable with another")
    say("  cohort's, because filtering changes which patients are being predicted.")
    say()
    pivot = all_metrics.pivot_table(index="model", columns="cohort", values="MAE")
    order = [c for c in ["unfiltered", "A4", "D1", "E2"] if c in pivot.columns]
    say(frame(pivot[order].round(4), index=True))
    say()
    say("  Prediction points per cohort:")
    counts = all_metrics.pivot_table(index="model", columns="cohort", values="n").iloc[0]
    say(f"    {counts[order].astype(int).to_dict()}")

    figures = _make_figures(cfg, out, traj, available, all_metrics,
                            pd.concat(stratified, ignore_index=True), confusions)
    section("FIGURES")
    say("  Each answers one question; none uses information a model did not have.")
    say()
    for question, path in figures:
        say(f"  {path.name}")
        say(f"      {question}")

    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out / 'report.txt'}")


def _make_figures(cfg, out: Path, traj, cohorts: list, metrics, stratified,
                  confusions) -> list:
    """Build every figure, returning (question answered, path) pairs."""
    from ppmi_subtype import gpr_plots as gp

    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    primary = cohorts[0]
    summary = pd.read_csv(out / "trajectory_summary.csv")
    extreme = pd.read_csv(out / "extreme_patient_summary.csv")
    comparison = pd.read_csv(out / "cohort_comparison.csv")
    preds = pd.read_csv(out / f"gpr_predictions_{primary}.csv")
    steps = pd.read_csv(out / f"trajectory_predictions_{primary}.csv")

    kernels = [k for k in cfg.gpr["kernels"] if k in set(preds["model"])]
    headline = [k for k in kernels] + ["hbayes", "persistence", "patient_mean"]
    headline = [m for m in headline if m in set(preds["model"])]
    best_gp = (
        metrics[(metrics.cohort == primary) & (metrics.model.isin(kernels))]
        .sort_values("MAE")["model"].iloc[0]
    )

    made = []

    def add(question, fn, name, *args, **kwargs):
        try:
            path = fn(*args, figs / name, **kwargs)
            made.append((question, path))
        except Exception as exc:                       # noqa: BLE001
            print(f"  figure {name} skipped: {exc}")

    add("Where do ratio values sit, and how much of the scale is unusable?",
        gp.ratio_distribution, "ratio_distribution.png", traj)
    add("Is the ratio a stable personal level, or does it roam within a patient?",
        gp.patient_ratio_statistics, "patient_ratio_statistics.png", summary)
    add("What does an individual patient's ratio trajectory look like?",
        gp.example_trajectories, "example_trajectories.png", traj)
    add("Are the flagged patients persistently extreme, or occasionally so?",
        gp.extreme_trajectories, "extreme_trajectories.png", traj, extreme,
        cfg.trajectory["primary"])
    add("What does each extreme criterion cost, and does it change the task?",
        gp.filtering_effect, "filtering_effect.png", comparison)
    add(f"Does a {best_gp} forecast look right for one patient, point by point?",
        gp.example_forecast, "example_forecast.png", steps, traj, best_gp)
    add("What does the forecast say about subtype, as a probability over time?",
        gp.subtype_probability_trajectories, "subtype_probability_trajectories.png",
        steps, best_gp)
    add("Does more history help, and does a longer horizon hurt?",
        gp.error_by_stratum, "error_by_stratum.png", stratified, primary, headline)
    add("Is the model tracking individuals, or predicting the cohort average?",
        gp.predicted_vs_observed, "predicted_vs_observed.png", preds, best_gp)
    add("Is the stated uncertainty honest?",
        gp.calibration, "calibration.png", preds, headline)
    add("Does the flexible GP beat the linear model, and either beat the trivial ones?",
        gp.model_comparison, "model_comparison.png", metrics, primary)
    add("Does filtering change the answer, or only the population?",
        gp.cohort_comparison, "cohort_comparison.png", metrics,
        [m for m in [best_gp, "hbayes", "persistence", "patient_mean"]
         if m in set(preds["model"])])
    add("Where do the derived subtype predictions land?",
        gp.confusion_panel, "confusion_panel.png",
        {m: confusions[primary][m] for m in headline[:4] if m in confusions[primary]})

    return made


def _report_headline(cfg, name, preds, steps, fits, traj) -> None:
    """The primary-cohort results, written out in full."""
    section(f"5. PRIMARY RESULT --- {name.upper()} COHORT")
    say("  Leave one patient out. Every model refitted on the other patients;")
    say("  visit k predicted from visits 0..k-1 of the held-out patient only.")
    say()
    say(f"  patients            {preds['held_out_patient'].nunique():,}")
    say(f"  prediction points   {int(len(preds) / preds['model'].nunique()):,}")
    say(f"  models scored       {preds['model'].nunique()}")
    say()
    say("  Continuous prediction of the ratio. This is the headline: the")
    say("  categorical numbers below are derived from these distributions.")
    say()
    table = gpr_loo.metrics_table(preds, name)
    cols = ["model", "n", "MAE", "RMSE", "MedAE", "CRPS", "coverage95",
            "interval_width", "corr"]
    say(frame(table[cols].round(4)))

    section("6. WHAT THE KERNELS DID")
    say("  Hyperparameters across folds. A parameter pinned to its bound in most")
    say("  folds is not an estimate; it is the optimiser reporting that the data")
    say("  cannot identify that component.")
    say()
    for kernel, g in fits.groupby("kernel"):
        params = [c for c in g.columns if c in gpr.parameter_names(kernel)]
        med = g[params].median().round(4).to_dict()
        boundary = g["at_boundary"].fillna("").replace("", np.nan).dropna()
        say(f"    {kernel}: {med}")
        say(f"        converged {int(g['converged'].sum()):,}/{len(g):,}   "
            f"folds with a parameter at a bound {len(boundary):,}")
        if len(boundary):
            hits = boundary.str.split(",").explode().value_counts().to_dict()
            say(f"        which parameter: {hits}")

    section("7. SUBTYPE PROBABILITIES DERIVED FROM THE RATIO")
    say("  Secondary. The Indeterminate band is 0.25 wide and the predictive")
    say("  spread is several times that, so little probability mass can ever fall")
    say("  inside it; a low Indeterminate recall describes the band, not the model.")
    say()
    cat = ["model", "accuracy", "balanced_accuracy", "brier", "loglik_subtype",
           "recall_TD", "recall_Indeterminate", "recall_PIGD"]
    say(frame(gpr_loo.metrics_table(preds, name)[cat].round(4)))
    say()
    say("  These are NOT comparable with the 75.4% from the hidden Markov model.")
    say("  That model predicts the next label directly and scores 3,361 points,")
    say(f"  including visits with an undefined ratio; this scores "
        f"{int(len(preds)/preds['model'].nunique()):,} points, where the ratio exists.")

    if not steps.empty:
        section("8. MULTI-STEP TRAJECTORY FORECASTS")
        say("  The first ceil(T/2) observations are history; everything after is")
        say("  forecast at the times those visits actually happened.")
        say()
        say(frame(gpr_loo.trajectory_metrics(steps, name).round(4)))

    section("9. PREDICTIVE MASS BELOW ZERO")
    say("  A Gaussian predictive on the raw ratio puts some mass where no ratio")
    say("  can lie. Reported rather than truncated; it is the main argument for")
    say("  examining a transformed scale in a later step.")
    say()
    neg = preds.groupby("model")["negative_mass"].mean().sort_values(ascending=False)
    say(frame(neg.round(4).reset_index()))


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "cohorts", "lopo", "report"])
    ap.add_argument("--cohorts", default=",".join(COHORTS),
                    help="comma-separated: unfiltered,A4,D1,E2")
    ap.add_argument("--kernels", default=None,
                    help="comma-separated; defaults to gpr.kernels in config")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes; folds are independent so this "
                         "changes runtime and nothing else")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N patients per cohort, for a smoke test")
    args = ap.parse_args()

    cfg = load_config()
    out = cfg.path("outputs") / OUT_SUBDIR
    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    kernels = ([k.strip() for k in args.kernels.split(",")]
               if args.kernels else list(cfg.gpr["kernels"]))

    unknown = [c for c in cohorts if c != "unfiltered" and c not in cfg.trajectory["criteria"]]
    if unknown:
        raise SystemExit(f"unknown cohort(s) {unknown}; config defines "
                         f"{sorted(cfg.trajectory['criteria'])} plus 'unfiltered'")

    if args.stage in ("all", "cohorts"):
        stage_cohorts(cfg, out)
    if args.stage in ("all", "lopo"):
        stage_lopo(cfg, out, cohorts, kernels, args.jobs, args.limit)
    if args.stage in ("all", "report"):
        stage_report(cfg, out, cohorts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
