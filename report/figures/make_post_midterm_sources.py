"""Print every number quoted in BTP_Post_Midterm_Report.tex, read from saved files.

    python report/figures/make_post_midterm_sources.py > report/BTP_Post_Midterm_Report_sources.txt
"""
import json
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[2]
O, E = R / "outputs", R / "experiments"
S3 = E / "20261014T094751Z_s3_subscore_heldout/analysis"
S4 = E / "20261014T100057Z_s4_hmm_cv/analysis"
S4P = E / "20261014T094649Z_s4_hmm_stability/analysis"
S4M = E / "20261014T094839Z_s4_hmm_population_maxfun/analysis"
S5 = E / "20261014T100107Z_s5_temporal_audit"
S6 = E / "20261014T100107Z_s6_clusterability"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def show(title, df):
    print(f"\n### {title}")
    print(df.round(4).to_string(index=False))


g = O / "step3_gpr"
rm = pd.read_csv(g / "reconciled_metrics.csv")
show("GPR ratio, unfiltered cohort (held-out LOPO)", rm[rm.cohort == "unfiltered"][
    ["model", "n_patients", "n_points", "MAE_pooled", "CRPS", "coverage95", "negative_mass",
     "accuracy", "balanced_accuracy"]])
pc = pd.read_csv(g / "paired_comparisons.csv")
show("GPR paired, pooled MAE", pc[pc.model_a.isin(["K1", "K3"])][
    ["cohort", "model_a", "model_b", "diff_pooled", "pooled_lo", "pooled_hi"]])
gm = pd.read_csv(g / "gpr_metrics.csv")
show("GPR subtype recall", gm[gm.cohort == "unfiltered"][["model", "accuracy", "recall_TD",
                                                         "recall_Indeterminate", "recall_PIGD"]])
mh = pd.read_csv(g / "multistep_audit_by_horizon.csv")
show("GPR multistep MAE by horizon (unfiltered)", mh[(mh.cohort == "unfiltered") & (mh.metric == "MAE")
                                                 & mh.model.isin(["K1", "persistence"])])
show("GPR mean-function held-out", pd.read_csv(g / "mean_function_holdout_summary.csv")[
    ["model", "n_points", "MAE_pooled", "CRPS", "diff_pooled", "diff_pooled_lo", "diff_pooled_hi"]])
print("\n### Joint LMM (in-sample REML)")
print((O / "joint_subscore/model_summary.txt").read_text().split("RANDOM INTERCEPT")[0])

show("Stage 3 checks", pd.read_csv(S3 / "checks.csv")[["check", "pass"]])
cm = pd.read_csv(S3 / "continuous_metrics.csv")
show("Stage 3 continuous (held-out LOPO, 3,361 points, 934 patients)",
     cm[(cm.outcome != "joint") & cm.model.isin(["persistence", "lmm_ri", "gp_indep_K1",
                                                 "gp_indep_K3", "gp_joint_K3_full"])]
     [["model", "outcome", "n", "MAE", "CRPS", "cov95", "mean_negative_mass", "pit_ks_stat"]])
pa = pd.read_csv(S3 / "paired.csv")
sel = pa[pa.metric.isin(["ae", "crps", "log_score"])]
show("Stage 3 nested full - diag", sel[sel.family == "1_nested_cross_output"][
    ["model", "outcome", "metric", "difference", "lo", "hi", "better"]])
show("Stage 3 temporal Kx - K0 (indep)", sel[(sel.family == "4_temporal_flexibility")
                                            & sel.model.str.startswith("gp_indep")][
    ["model", "outcome", "metric", "difference", "lo", "hi", "better"]])
show("Stage 3 vs persistence / LMM", sel[sel.family.isin(["5_vs_persistence", "5_vs_lmm_ri"])
                                        & sel.model.isin(["gp_indep_K1", "gp_indep_K3", "lmm_ri"])
                                        & sel.metric.isin(["ae", "crps"]) & (sel.outcome != "joint")][
    ["model", "reference", "outcome", "metric", "difference", "lo", "hi", "better"]])
sm = pd.read_csv(S3 / "subtype_metrics.csv")
show("Stage 3 subtype", sm[sm.model.isin(["persistence", "gp_indep_K1", "gp_joint_K3_full",
                                          "hmm_loo_09", "lmm_ri"])][
    ["model", "rule", "accuracy", "balanced_accuracy", "recall_Indeterminate", "log_loss"]])
show("Stage 3 subtype vs HMM", pd.read_csv(S3 / "subtype_paired_vs_hmm.csv").query(
    "model in ['gp_indep_K1','gp_joint_K3_full','persistence']"))
show("Stage 3 subtype vs label persistence", pd.read_csv(
    S3 / "subtype_paired_vs_label_persistence.csv").query("model in ['gp_indep_K1','hmm_loo_09']")[
    ["model", "rule", "accuracy", "accuracy_diff", "accuracy_diff_lo", "accuracy_diff_hi",
     "balanced_accuracy", "balanced_diff", "balanced_diff_lo", "balanced_diff_hi",
     "recall_Indeterminate"]])

show("Stage 4 population (default budget)", pd.read_csv(S4P / "population_summary.csv")[
    ["config", "n_starts", "best_loglik", "share_within_0.01", "n_loglik_levels_0.01",
     "max_abs_dE_among_best", "share_success_flag", "default_start_gap"]])
show("Stage 4 population (maxfun 400k)", pd.read_csv(S4M / "population_summary.csv")[
    ["config", "n_starts", "share_within_0.01", "n_loglik_levels_0.01", "share_success_flag", "AIC",
     "BIC_n_obs"]])
c4 = pd.read_csv(S4 / "cv_config_metrics.csv")
show("Stage 4 CV per config", c4[c4.config.isin(["C3_f50_audited", "C3_free", "C5_free",
                                                 "Persistence (stay)", "Markov (discrete)"])][
    ["config", "rule", "accuracy", "balanced_accuracy", "recall_Indeterminate", "log_loss",
     "auc_Indeterminate", "max_p_Indet"]])
show("Stage 4 selected procedure", pd.read_csv(S4 / "selected_procedure_metrics.csv")[
    ["procedure", "rule", "accuracy", "accuracy_lo", "accuracy_hi", "balanced_accuracy",
     "log_loss", "selections"]])

show("Stage 5 period slopes, boot vs Wald", pd.read_csv(
    S5 / "bootstrap/period_slopes_boot_vs_wald.csv").query("model=='primary_RI'")[
    ["outcome", "period", "slope", "wald_lo", "wald_hi", "boot_lo", "boot_hi",
     "se_ratio_boot_over_wald"]])
show("Stage 5 contrasts", pd.read_csv(S5 / "bootstrap/contrasts_boot.csv").query("model=='primary_RI'"))
show("Stage 5 sup-LR", pd.read_csv(S5 / "knots/sup_lr_test.csv"))
kb = pd.read_csv(S5 / "knots/knot_argmax_bootstrap.csv")
print("knot argmax q10/q50/q90:", kb.knot.quantile([.1, .5, .9]).tolist(), "n", len(kb))
show("Stage 5 ML structures", pd.read_csv(S5 / "structures/ml_fits.csv")[
    ["model", "AIC", "dAIC_vs_best", "dBIC_patients_vs_best"]])
cv = pd.read_csv(S5 / "structures/cv_paired_vs_linear.csv")
show("Stage 5 CV CRPS vs linear", cv[(cv.metric == "crps") & cv.model.isin(["poly2", "pw_2_5"])])
show("Stage 5 linear slopes (boot)", pd.read_csv(S5 / "structures/period_slopes_boot.csv").query(
    "model=='linear' and period=='0-2y'")[["outcome", "slope", "boot_lo", "boot_hi"]])
show("Stage 5 time axis", pd.read_csv(S5 / "timeaxis/followup_slope_changes.csv"))
show("Stage 5 treated share", pd.read_csv(S5 / "timeaxis/treated_share_by_year.csv"))
show("Stage 5 loss of follow-up logit", pd.read_csv(S5 / "attrition/loss_of_followup_logit.csv"))
show("Stage 5 completers", pd.read_csv(S5 / "attrition/completer_slopes.csv").query(
    "period.str.contains('late')", engine="python"))

show("Stage 6 decision", pd.read_csv(S6 / "decision.csv"))
show("Stage 6 summary", pd.read_csv(S6 / "summary.csv")[
    ["dataset", "n", "dBIC_median", "dBIC_q95", "drop1_ARI_K2_median", "drop1_ARI_K2_q95"]])
ca = O / "clustering_audit"
lo = pd.read_csv(ca / "leave_one_visit_out.csv")
print("\n### Script 20 sign flips:", {o: (int(g_.sign_flipped.sum()), len(g_)) for o, g_ in lo.groupby("outcome")})
show("Script 20 stability", pd.read_csv(ca / "cluster_stability.csv"))
rr = pd.read_csv(ca / "random_slope_reliability.csv")
print("reliability median tremor/pigd:", round(rr.reliability_tremor.median(), 3),
      round(rr.reliability_pigd.median(), 3))
for name, run in [("6b cal2", "20261014T133917Z_s6b_skewed_null_cal2/calibration_fidelity.csv"),
                  ("6b M1-M5", "20261014T144511Z_pigd_measurement_models/predictive_checks.csv"),
                  ("6b GRM", "20261014T165801Z_s6b_grm_predictive_check/calibration_table.csv"),
                  ("6b GRM+OU", "20261014T180533Z_s6b_grm_ou_predictive_check/calibration_table.csv")]:
    d = pd.read_csv(E / run)
    if "status" in d:
        print(f"\n### {name}: {int((d.status == 'pass').sum())}/{len(d)} pass")
        print(d[d.status != "pass"][["target", "real", "sim_mean", "tolerance"]].round(3).to_string(index=False))
    elif "model" in d:
        print(f"\n### {name}")
        print(d.pivot(index="feature", columns="model", values="sim_mean").round(3).loc[
            ["zero_share", "P_next_zero_given_S1", "blup_skew_slope_pigd"]].to_string())
    else:
        show(name, d[d.target.isin(["skew_slope_pigd", "share_pigd_zero_visits"])])
ou = json.loads((E / "20261014T180533Z_s6b_grm_ou_predictive_check/ou_fit.json").read_text())
D = ou["D"]
print("OU ell", round(ou["ell"], 2), "corr", round(D[0][1] / (D[0][0] * D[1][1]) ** .5, 3))
au = E / "20261014T143819Z_pigd_measurement_audit"
show("PIGD audit transitions", pd.read_csv(au / "transitions.csv").head(5))
show("PIGD audit items", pd.read_csv(au / "items.csv")[["item", "share_0"]])
