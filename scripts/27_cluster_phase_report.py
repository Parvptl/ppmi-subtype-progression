"""Assemble the cluster-phase results document from saved analysis CSVs only.

    python scripts/27_cluster_phase_report.py --s3 <run> --s4-pop <run> \\
        [--s4-pop-maxfun <run>] --s4-cv <run> --s5 <run> --s6 <run> [--s6-min8 <run>] \\
        --out docs/cluster_phase/04_RESULTS_GENERATED.md

Every number in the output is read from a CSV written by scripts 22, 24, 25 or
26. Sentences are templates filled from those files, so a rerun of any
experiment changes the document without anyone retyping a value. Contains
aggregate statistics only (no participant-level rows), so it may be committed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def md(df: pd.DataFrame, fmt="{:.4f}") -> str:
    if df is None or df.empty:
        return "_(no rows)_\n"
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "\n".join("| " + " | ".join(
        fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
        else ("" if isinstance(v, float) and not np.isfinite(v) else str(v))
        for v in r) + " |" for r in df.itertuples(index=False))
    return head + body + "\n"


def ci(row, est="difference", lo="lo", hi="hi", fmt="{:+.4f}"):
    return f"{fmt.format(row[est])} [{fmt.format(row[lo])}, {fmt.format(row[hi])}]"


def provenance(run: Path, name="run_config.json") -> str:
    files = sorted(run.glob("run_config*.json"))
    m = json.loads(files[0].read_text()) if files else {}
    return (f"`{run.name}` · git `{str(m.get('git_commit'))[:10]}` · panel "
            f"`{m.get('panel_md5')}`" + (" · **SMOKE**" if m.get("smoke_test") else ""))


# ---------------------------------------------------------------------------
def stage3(run: Path) -> list[str]:
    an = run / "analysis"
    chk = pd.read_csv(an / "checks.csv")
    cont = pd.read_csv(an / "continuous_metrics.csv")
    par = pd.read_csv(an / "paired.csv")
    sub = pd.read_csv(an / "subtype_metrics.csv")
    diag = pd.read_csv(an / "fit_diagnostics.csv")
    L = ["## Stage 3 — persistence vs LMM vs independent GP vs joint GP (held out)\n",
         provenance(run) + "\n",
         f"Leakage/consistency battery: **{int(chk['pass'].sum())}/{len(chk)} passed**.\n"]
    if not chk["pass"].all():
        L.append("**Some checks failed; the results below must not be used.**\n")
        L.append(md(chk[~chk["pass"]]))

    L.append("### Primary: nested cross-output test (joint full − joint diag)\n")
    nest = par[par.family == "1_nested_cross_output"]
    rows = []
    for (m, o, metric), g in nest.groupby(["model", "outcome", "metric"]):
        r = g.iloc[0]
        rows.append({"kernel": m.split("_")[2], "outcome": o, "metric": metric,
                     "full − diag [95% CI]": ci(r, fmt="{:+.5f}"), "favours": r.better})
    L.append(md(pd.DataFrame(rows)))
    n_sig = int(nest.excludes_zero.sum())
    L.append(f"{n_sig} of {len(nest)} nested contrasts exclude zero"
             + (" — see the rows marked `model` or `reference` above.\n" if n_sig else ".\n"))

    for fam, title in [("3_joint_vs_independent", "Joint (full) − independent GPs"),
                       ("2_shared_lengthscale", "Shared lengthscale: joint diag − independent"),
                       ("4_temporal_flexibility", "Temporal flexibility: Kx − K0")]:
        g = par[(par.family == fam) & par.metric.isin(["crps", "log_score", "ae"])]
        L.append(f"### {title}\n")
        L.append(md(g.assign(ci=g.apply(lambda r: ci(r, fmt="{:+.5f}"), axis=1))
                    [["model", "reference", "outcome", "metric", "ci", "better"]]))

    L.append("### Against persistence and the LMM (CRPS and MAE)\n")
    g = par[par.family.isin(["5_vs_persistence", "5_vs_lmm_ri"])
            & par.metric.isin(["ae", "crps"]) & (par.outcome != "joint")]
    L.append(md(g.assign(ci=g.apply(lambda r: ci(r, fmt="{:+.5f}"), axis=1))
                [["model", "reference", "outcome", "metric", "ci", "better"]]))

    L.append("### Continuous metrics and calibration\n")
    c = cont[cont.outcome != "joint"].copy()
    L.append(md(c[["model", "outcome", "MAE", "RMSE", "CRPS", "log_score", "cov50", "cov80",
                   "cov95", "pit_ks_stat", "mean_negative_mass"]]
                .sort_values(["outcome", "CRPS"])))
    j = cont[cont.outcome == "joint"][["model", "log_score", "log_score_lo", "log_score_hi"]]
    L.append("Bivariate (joint) log predictive density:\n")
    L.append(md(j.sort_values("log_score", ascending=False)))

    L.append("### Derived subtype\n")
    L.append(md(sub[["model", "rule", "accuracy", "accuracy_lo", "accuracy_hi",
                     "balanced_accuracy", "balanced_accuracy_lo", "balanced_accuracy_hi",
                     "recall_TD", "recall_Indeterminate", "recall_PIGD",
                     "n_predicted_Indeterminate", "log_loss", "brier"]]
                .sort_values(["rule", "log_loss"])))
    if (an / "subtype_paired_vs_hmm.csv").exists():
        h = pd.read_csv(an / "subtype_paired_vs_hmm.csv")
        L.append("Paired against the HMM LOO predictions (script 09), identical 3,361 points:\n")
        L.append(md(h.assign(ci=h.apply(lambda r: ci(r, fmt="{:+.4f}"), axis=1))
                    [["model", "metric", "ci", "excludes_zero"]]))
    if (an / "subtype_paired_vs_label_persistence.csv").exists():
        lp = pd.read_csv(an / "subtype_paired_vs_label_persistence.csv")
        lp = lp[lp.model != "persistence"]
        L.append("Paired against LABEL persistence (repeat the last label; script 28). "
                 "Overall accuracy and balanced accuracy, model − persistence:\n")
        L.append(md(lp.assign(
            accuracy_diff_ci=lp.apply(lambda r: ci(r, "accuracy_diff", "accuracy_diff_lo",
                                                   "accuracy_diff_hi"), axis=1),
            balanced_diff_ci=lp.apply(lambda r: ci(r, "balanced_diff", "balanced_diff_lo",
                                                   "balanced_diff_hi"), axis=1))
            [["model", "rule", "accuracy", "accuracy_diff_ci", "balanced_accuracy",
              "balanced_diff_ci", "recall_Indeterminate"]]))
    L.append("### Convergence and fit diagnostics\n")
    L.append(md(diag))
    if (an / "reproduction.csv").exists():
        L.append("### Reproduction of the historical Windows predictions\n")
        L.append(md(pd.read_csv(an / "reproduction.csv"), fmt="{:.2e}"))
    return L


def stage4(pop: Path, pop_mf: Path | None, cv: Path) -> list[str]:
    L = ["## Stage 4 — HMM stability\n"]
    for run, label in [(pop, "default evaluation budget (scipy maxfun 15,000)"),
                       (pop_mf, "maxfun 400,000")]:
        if run is None:
            continue
        s = pd.read_csv(run / "analysis" / "population_summary.csv")
        L += [f"### A. Population fits, {label} — EXPLORATORY\n", provenance(run) + "\n",
              md(s[["config", "n_starts", "n_params", "best_loglik", "worst_loglik", "AIC",
                    "BIC_n_obs", "share_within_0.01", "n_loglik_levels_0.01",
                    "default_start_gap", "max_abs_dE_among_best", "share_success_flag",
                    "messages"]])]
    an = cv / "analysis"
    L += ["### B. Held-out per configuration (outer 10-fold) — sensitivity, not selection\n",
          provenance(cv) + "\n"]
    cm = pd.read_csv(an / "cv_config_metrics.csv")
    L.append(md(cm[["config", "rule", "accuracy", "balanced_accuracy", "recall_TD",
                    "recall_Indeterminate", "recall_PIGD", "n_predicted_Indeterminate",
                    "log_loss", "brier", "auc_Indeterminate", "mean_p_Indet_when_Indet",
                    "mean_p_Indet_otherwise", "max_p_Indet"]]))
    pv = pd.read_csv(an / "cv_paired_vs_reference.csv")
    L += ["Paired against `C3_f50_audited`:\n",
          md(pv.assign(ci=pv.apply(lambda r: ci(r, fmt="{:+.4f}"), axis=1))
             [["config", "metric", "ci", "excludes_zero"]])]
    sel = pd.read_csv(an / "selection.csv")
    sp = pd.read_csv(an / "selected_procedure_metrics.csv")
    L += ["### C. Selected procedure — the unbiased estimate\n",
          "Inner-CV choice per outer fold: "
          + ", ".join(f"{k}×{v}" for k, v in sel.selected_inner_cv.value_counts().items())
          + "; training-BIC choice: "
          + ", ".join(f"{k}×{v}" for k, v in sel.selected_bic.value_counts().items()) + ".\n",
          md(sp[["procedure", "rule", "accuracy", "accuracy_lo", "accuracy_hi",
                 "balanced_accuracy", "balanced_accuracy_lo", "balanced_accuracy_hi",
                 "recall_Indeterminate", "log_loss", "log_loss_lo", "log_loss_hi"]])]
    iv = pd.read_csv(an / "cv_init_variation.csv")
    agg = iv.groupby("config").agg(
        test_ll_sd_median=("test_ll_sd", "median"),
        test_ll_range_median=("test_ll_max", lambda s: float("nan")),
        share_starts_at_best_train=("share_starts_within_0.01_train", "median"),
        corr_train_test=("corr_train_test_ll", "median"),
        oracle_gap_median=("oracle_gap_DO_NOT_REPORT_AS_PERFORMANCE", "median"),
        success_flag=("share_success_flag", "mean")).drop(columns="test_ll_range_median")
    L += ["### D. Variation across initialisations (held-out log-lik per transition)\n",
          md(agg.reset_index())]
    return L


def stage5(run: Path) -> list[str]:
    L = ["## Stage 5 — temporal structure\n", provenance(run) + "\n"]
    d = run / "describe"
    if (d / "late_period_sparsity.csv").exists():
        L += ["### Late-follow-up sparsity\n", md(pd.read_csv(d / "late_period_sparsity.csv"))]
    b = run / "bootstrap"
    if (b / "period_slopes_boot_vs_wald.csv").exists():
        s = pd.read_csv(b / "period_slopes_boot_vs_wald.csv")
        L += ["### Period slopes: patient bootstrap vs Wald\n",
              md(s[["model", "outcome", "period", "slope", "wald_lo", "wald_hi", "boot_lo",
                    "boot_hi", "se_ratio_boot_over_wald"]]),
              md(pd.read_csv(b / "contrasts_boot.csv"))]
    k = run / "knots"
    if (k / "sup_lr_test.csv").exists():
        dist = pd.read_csv(k / "knot_argmax_bootstrap.csv")
        q = dist.knot.quantile([0.1, 0.5, 0.9])
        L += ["### Where is the knot, and is there one at all?\n",
              md(pd.read_csv(k / "sup_lr_test.csv")),
              f"Bootstrap argmax knot ({len(dist)} replicates): 10th/50th/90th percentile "
              f"{q.iloc[0]:g} / {q.iloc[1]:g} / {q.iloc[2]:g} years.\n"]
    st = run / "structures"
    if (st / "ml_fits.csv").exists():
        L += ["### Alternative temporal structures (ML)\n",
              md(pd.read_csv(st / "ml_fits.csv")[["model", "n_params", "AIC", "dAIC_vs_best",
                                                  "BIC_patients", "dBIC_patients_vs_best"]]),
              "Period-average slopes with bootstrap CIs:\n",
              md(pd.read_csv(st / "period_slopes_boot.csv")[["model", "outcome", "period",
                                                             "slope", "boot_lo", "boot_hi"]]),
              "10-fold CV one-step prediction, paired against `linear` (CRPS):\n"]
        cv = pd.read_csv(st / "cv_paired_vs_linear.csv")
        cv = cv[cv.metric == "crps"]
        L.append(md(cv.assign(ci=cv.apply(lambda r: ci(r, fmt="{:+.5f}"), axis=1))
                    [["model", "outcome", "ci", "excludes_zero"]]))
    t = run / "timeaxis"
    if (t / "followup_slope_changes.csv").exists():
        L += ["### Follow-up slope changes adjusted for disease duration and treatment\n",
              md(pd.read_csv(t / "treated_share_by_year.csv")),
              md(pd.read_csv(t / "ml_fits.csv")),
              md(pd.read_csv(t / "followup_slope_changes.csv"))]
    elif (t / "status.csv").exists():
        L += ["### Time axis\n", md(pd.read_csv(t / "status.csv"))]
    a = run / "attrition"
    if (a / "completer_slopes.csv").exists():
        L += ["### Attrition\n", "Loss of labelled follow-up vs current scores (logit):\n",
              md(pd.read_csv(a / "loss_of_followup_logit.csv")),
              "Completers only:\n", md(pd.read_csv(a / "completer_slopes.csv")),
              "Early (0–2 y) slopes by eventual follow-up length:\n",
              md(pd.read_csv(a / "pattern_mixture_early_slopes.csv"))]
    return L


def stage6(run: Path, label: str) -> list[str]:
    L = [f"## Stage 6 — clusterability ({label})\n", provenance(run) + "\n"]
    if not (run / "decision.csv").exists():
        return L + ["_(not complete)_\n"]
    L += ["### Pre-specified decision\n", md(pd.read_csv(run / "decision.csv"))]
    s = pd.read_csv(run / "summary.csv")
    cols = ["dataset", "n", "dBIC_median", "dBIC_q95", "best_K_median", "silhouette_K2_median",
            "drop1_ARI_K2_median", "drop1_ARI_K2_q95", "trunc3_ARI_K2_median",
            "drop1_spearman_pigd_median", "trunc3_spearman_pigd_median"]
    L += ["### Real data against nulls and planted controls\n",
          md(s[[c for c in cols if c in s.columns]])]
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("--s3", "--s4-pop", "--s4-pop-maxfun", "--s4-cv", "--s5", "--s6", "--s6-min8"):
        ap.add_argument(a, type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    L = ["# Cluster phase — generated results\n",
         "Generated by `scripts/27_cluster_phase_report.py` from the analysis CSVs of the "
         "runs named in each section. Do not edit by hand; rerun the script.\n"]
    if args.s3:
        L += stage3(args.s3)
    if args.s4_cv:
        L += stage4(args.s4_pop, args.s4_pop_maxfun, args.s4_cv)
    if args.s5:
        L += stage5(args.s5)
    if args.s6:
        L += stage6(args.s6, "all 934 patients")
    if args.s6_min8:
        L += stage6(args.s6_min8, "patients with >= 8 visits — EXPLORATORY")
    args.out.write_text("\n".join(L))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
