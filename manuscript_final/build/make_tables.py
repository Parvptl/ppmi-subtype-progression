#!/usr/bin/env python3
"""Generate every manuscript table from the saved experiment outputs.

Each table is read from the result files of the run that produced it (CSV,
JSON or the two plain-text summaries), reduced to the columns the manuscript
shows, rounded, and written to manuscript_final/tables/tableNN_<slug>.csv.
A TABLE_INDEX.csv records, for every table, its caption and the source files
it was built from. Nothing is recomputed from raw data: the only arithmetic
is rounding, string formatting of confidence intervals, and, where stated in
the caption, a mean over the saved per-split values.

Nothing outside manuscript_final/ is written.
"""
from __future__ import annotations
import csv, json, re, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MF = HERE.parent
ROOT = MF.parent
OUT = MF / "tables"
E = "experiments"
O = "outputs"

INDEX: list[dict] = []
PROBLEMS: list[str] = []


def rd(path: str, **kw) -> pd.DataFrame:
    return pd.read_csv(ROOT / path, **kw)


def js(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def ci(df: pd.DataFrame, lo: str, hi: str, nd: int = 4) -> pd.Series:
    def f(a, b):
        if pd.isna(a) or pd.isna(b):
            return ""
        return f"[{a:.{nd}f}, {b:.{nd}f}]"
    return [f(a, b) for a, b in zip(df[lo], df[hi])]


def rnd(df: pd.DataFrame, nd: int = 4) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(nd)
    return out


def pick(df: pd.DataFrame, cols: list[str], rename: dict | None = None) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        PROBLEMS.append(f"missing columns {missing}")
    d = df[[c for c in cols if c in df.columns]].copy()
    if rename:
        d = d.rename(columns=rename)
    return d


def tab(n: int, slug: str, df: pd.DataFrame, caption: str, sources: list[str],
        note: str = "", nd: int = 4) -> None:
    df = rnd(df, nd)
    fn = f"table{n:03d}_{slug}.csv"
    df.to_csv(OUT / fn, index=False)
    INDEX.append({"table": n, "file": fn, "caption": caption, "note": note,
                  "sources": "; ".join(sources), "n_rows": len(df), "n_cols": len(df.columns)})


def build() -> None:
    OUT.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- cohort
    s = f"{E}/20261103T192414Z_e5_participant_flow/"
    df = rd(s + "flow_ledger.csv")
    tab(1, "participant_flow", df,
        "Participant flow from the curated PPMI data cut to the analysed cohort. Each exclusion is counted at the step that applied it.",
        [s + "flow_ledger.csv"])
    df = rd(s + "unlabelled_diagnosis.csv")
    tab(2, "unlabelled_visits_reasons", df,
        "Why 1,333 eligible visits carry no subtype label.", [s + "unlabelled_diagnosis.csv"])

    rs_all = js(f"{E}/20261015T221426Z_partC_raw_scores/results.json"); rs = rs_all["primary"]
    ps = (ROOT / O / "panel_summary.txt").read_text()
    def grab(pat):
        m = re.search(pat, ps); return m.group(1).replace(",", "") if m else ""
    prev = rd(f"{O}/transitions/prevalence_by_year.csv")
    y0 = prev[prev["YEAR"] == 0].iloc[0]
    rows = [
        ("patients in the eligible cohort", rs["patients"]),
        ("visits in the eligible cohort", rs["visits"]),
        ("visits with a subtype label", rs["labelled_visits"]),
        ("patients with at least two labelled visits", rs["patients_ge2_labelled"]),
        ("patients with at least three labelled visits", grab(r"patients with >=3:\s+([\d,]+)")),
        ("patients with at least five labelled visits", grab(r"patients with >=5:\s+([\d,]+)")),
        ("patients with at least eight labelled visits", grab(r"patients with >=8:\s+([\d,]+)")),
        ("labelled visits per patient, median (max)", f"{grab(r'labelled visits per patient: median (\d+)')} ({grab(r'median \d+, max (\d+)')})"),
        ("labelled visits: TD", rs["label_counts"]["TD"]),
        ("labelled visits: Indeterminate", rs["label_counts"]["Indeterminate"]),
        ("labelled visits: PIGD", rs["label_counts"]["PIGD"]),
        ("baseline (year 0) labelled visits: TD / Indeterminate / PIGD", f"{int(y0['n_TD'])} / {int(y0['n_Indeterminate'])} / {int(y0['n_PIGD'])} of {int(y0['n_total'])}"),
        ("ratio computable (PIGD subscore > 0)", grab(r"ratio computable\s+([\d,]+)")),
        ("ratio undefined (PIGD subscore = 0)", grab(r"ratio undefined\s+([\d,]+)")),
        ("share of labelled visits with PIGD item sum = 0", round(rs["pigd_sum0_share"], 4)),
        ("share of labelled visits with tremor item sum = 0", round(rs["tremor_sum0_share"], 4)),
        ("ratio: median (IQR)", "1.59 (0.68-2.95)"),
        ("interval between consecutive labelled visits, years: 10th / 25th / 50th / 75th / 90th percentile",
         " / ".join(f"{rs['interval_quantiles'][q]:.3f}" for q in ["0.1", "0.25", "0.5", "0.75", "0.9"])),
        ("labelled follow-up per patient, years: 10th / 25th / 50th / 75th / 90th percentile",
         " / ".join(f"{rs['followup_quantiles'][q]:.3f}" for q in ["0.1", "0.25", "0.5", "0.75", "0.9"])),
    ]
    tab(3, "cohort_summary", pd.DataFrame(rows, columns=["quantity", "value"]),
        "Cohort summary: labelled visits, label distribution, visit spacing and follow-up.",
        [f"{E}/20261015T221426Z_partC_raw_scores/results.json", f"{O}/panel_summary.txt", f"{O}/transitions/prevalence_by_year.csv"])

    # ------------------------------------------------------------ transitions
    s = f"{E}/20261103T193205Z_e2_transition_patient_bootstrap/"
    df = rd(s + "transition_probabilities.csv")
    d = pick(df, ["from_state", "to_state", "n_from", "n_transitions", "probability"])
    d["Wilson 95% CI (independent transitions)"] = ci(df, "wilson_low", "wilson_high")
    d["patient-bootstrap 95% CI"] = ci(df, "bootstrap_low", "bootstrap_high")
    d["design effect"] = df["design_effect"].round(2)
    tab(4, "transition_probabilities", d,
        "Transition probabilities between consecutive labelled visits (3,361 pairs, 934 patients), with Wilson intervals that treat transitions as independent and patient-clustered percentile bootstrap intervals (2,000 resamples of patients, seed 42). The design effect is the squared ratio of the bootstrap to the binomial standard error.",
        [s + "transition_probabilities.csv"])
    df = rd(s + "other_quantities.csv")
    d = pick(df, ["quantity", "estimate"]); d["patient-bootstrap 95% CI"] = ci(df, "bootstrap_low", "bootstrap_high"); d["bootstrap SE"] = df["bootstrap_se"]
    tab(5, "reversion_and_conversion", d,
        "Reversion and conversion quantities with patient-clustered bootstrap intervals: the pooled reversion rate (share of observed label changes that returned to the origin label at the following visit), reversion by change type, Kaplan-Meier cumulative incidence of a first PIGD label under the naive and the sustained definition, and the interval-stratified TD-to-PIGD probabilities.",
        [s + "other_quantities.csv"])
    df = rd(f"{O}/model/markov_property_test.csv")
    tab(6, "markov_property_test", df,
        "Test of the first-order Markov property in the observed label: chi-square test that the next label is independent of the previous label given the current label, on labelled triples.",
        [f"{O}/model/markov_property_test.csv"])
    df = rd(f"{O}/transitions/transitions_by_interval.csv")
    d = pick(df, ["interval", "from_state", "to_state", "n_from", "n_transitions", "probability"]); d["Wilson 95% CI"] = ci(df, "ci_low", "ci_high")
    tab(7, "transitions_by_interval", d,
        "Transition probabilities stratified by the elapsed interval between the two visits (Wilson intervals; the patient-clustered intervals for the TD-to-PIGD cells are in Table 5).",
        [f"{O}/transitions/transitions_by_interval.csv"])
    df = rd(f"{O}/transitions/pathways.csv").head(12)
    tab(8, "label_pathways", df, "The twelve most frequent label pathways (sequence of distinct labels in visit order) among the 934 patients with at least two labelled visits.",
        [f"{O}/transitions/pathways.csv"])
    df = rd(f"{O}/transitions/prevalence_by_year.csv"); df = df[df["YEAR"] <= 10]
    tab(9, "prevalence_by_year", df, "Label prevalence by study year (years 0 to 10).", [f"{O}/transitions/prevalence_by_year.csv"])

    # ------------------------------------------------------ label construction
    oo = rs_all["off_on"]
    rows = []
    for k, lab in [("full", "all 16 items (Part II shared)"), ("part3_only", "Part III items only")]:
        rows.append((lab, oo[k]["pairs_with_both_labels"], oo[k]["patients"], round(oo[k]["agreement"], 4)))
    d = pd.DataFrame(rows, columns=["label construction", "same-visit OFF/ON pairs with both labels", "patients", "label agreement OFF vs ON"])
    tab(10, "off_on_agreement", d,
        "Agreement of the subtype label between the OFF-medication and the ON-medication examination of the same visit (medication response, time of day, order and rater effects combined; not test-retest reliability). ON minus OFF tremor item sum: 10th/25th/50th/75th/90th percentiles "
        + " / ".join(str(oo["ON_minus_OFF_tremor_sum_quantiles"][q]) for q in ["0.1", "0.25", "0.5", "0.75", "0.9"])
        + "; PIGD item sum: " + " / ".join(str(oo["ON_minus_OFF_pigd_sum_quantiles"][q]) for q in ["0.1", "0.25", "0.5", "0.75", "0.9"]) + ".",
        [f"{E}/20261015T221426Z_partC_raw_scores/results.json"])

    ia = js(f"{E}/20261015T221426Z_partC_indeterminate/results.json")["primary"]
    names = {"R_E": "reversion after an entry into Indeterminate (R_E)", "R_D": "reversion after a direct TD<->PIGD change (R_D)",
             "delta_R": "difference R_E - R_D", "entry_sustained": "Indeterminate sustained at the next visit",
             "entry_moved_on": "Indeterminate followed by the other extreme label", "entry_sustained2": "Indeterminate sustained at two later visits",
             "direct_sustained": "direct change sustained at the next visit", "exit_back_to_Ind": "exit from Indeterminate followed by return to Indeterminate",
             "OE_TD": "observed/expected within-patient clustering, TD", "OE_Indeterminate": "observed/expected clustering, Indeterminate", "OE_PIGD": "observed/expected clustering, PIGD"}
    rows = [(names[k], round(ia["point"][k], 4), f"[{ia['ci95'][k][0]:.4f}, {ia['ci95'][k][1]:.4f}]") for k in names]
    d = pd.DataFrame(rows, columns=["quantity", "estimate", "patient-bootstrap 95% CI"])
    tab(11, "indeterminate_audit", d,
        f"Indeterminate-label audit (Part C). Entries into Indeterminate: {ia['entries']['events']} events, {ia['entries']['observed_next']} with an observed next visit ({ia['entries']['reverted']} reverted, {ia['entries']['sustained']} sustained, {ia['entries']['moved_on']} moved on). Direct TD-PIGD changes: {ia['direct']['events']} events, {ia['direct']['observed_next']} observed ({ia['direct']['reverted']} reverted). Entries with two later visits: {ia['two_step']['entries_with_two_later_visits']}, of which {ia['two_step']['sustained_two']} stayed Indeterminate at both. Reading: {ia['reading']}.",
        [f"{E}/20261015T221426Z_partC_indeterminate/results.json"])

    s = f"{E}/20261104T002308Z_o4_threshold_sensitivity/"
    df = rd(s + "summary.csv").T.reset_index(); df.columns = ["quantity", "value"]
    tab(12, "threshold_sensitivity_summary", df,
        "Threshold sensitivity (O-4): relabelling the frozen panel with the perturbed rule TD > 1.13, PIGD < 0.91 (inner boundaries) instead of TD >= 1.15, PIGD <= 0.90. Deterministic recount; no confidence interval applies.",
        [s + "summary.csv"])
    df = rd(s + "per_class_movement.csv"); ag = rd(s + "agreement_vs_ppmi.csv")
    tab(13, "threshold_sensitivity_movement", df, "Threshold sensitivity (O-4): movement of labelled visits between classes under the perturbed rule. Agreement with PPMI's own derived labels: "
        + "; ".join(f"{r['rule']}: two-class {r['two_class']:.4f} (n={int(r['two_class_n'])}), three-class {r['three_class']:.4f} (n={int(r['three_class_n'])})" for _, r in ag.iterrows()) + ".",
        [s + "per_class_movement.csv", s + "agreement_vs_ppmi.csv"])
    df = rd(f"{E}/20261103T191408Z_e1_lag1_residual_resolution/lag1_statistic.csv")
    tab(14, "lag1_residual_autocorrelation", df,
        "Lag-1 autocorrelation of within-patient residual ratio changes (E-1): observed value, the value an independent-noise null with the same visit pattern implies, and their difference. 2,050 consecutive-change pairs from 397 patients.",
        [f"{E}/20261103T191408Z_e1_lag1_residual_resolution/lag1_statistic.csv"])

    # ---------------------------------------------------------- D-1 and Stage 1
    ob = js(f"{E}/20261016T070746Z_d1_full/results/05_observed.json"); pw = js(f"{E}/20261016T070746Z_d1_full/results/02_power.json"); pv = js(f"{E}/20261016T070746Z_d1_full/results/03_pc4v.json")
    labels = {"S1_ind_share": "S1 Indeterminate share", "S2_R_E": "S2 reversion after entry (R_E)", "S3_R_D": "S3 direct reversion (R_D)", "S4_delta_R": "S4 R_E - R_D", "S5_OE_ind": "S5 O/E Indeterminate clustering"}
    rows = [(labels[k], round(ob["observed"]["S_obs"][k], 4), round(ob["observed"]["S_rep_mean"][k], 4), round(ob["observed"]["z"][k], 3), round(ob["observed"]["p"][k], 4)) for k in labels]
    d = pd.DataFrame(rows, columns=["statistic", "observed", "replicate mean", "z (capped at 3.09)", "p"])
    tab(15, "d1_posterior_predictive_check", d,
        f"D-1 posterior-predictive label check on 821 patients and 3,528 ratio-defined visits: observed label statistics against patient-conditional replicates from the Student-t growth model. T_obs = {ob['T_obs']:.4f} against the pre-registered threshold c = {ob['c']:.4f}; verdict {ob['verdict']}. Positive controls detected per 100 cohorts: PC1 {pw['scenarios']['PC1']['detected']}, PC2 {pw['scenarios']['PC2']['detected']}, PC3 {pw['scenarios']['PC3']['detected']}, PC4 {pw['scenarios']['PC4']['detected']} (lower 95% bound {pw['primary_P_lo']:.3f}); PC4v (descriptive) {pv['detected']}.",
        [f"{E}/20261016T070746Z_d1_full/results/05_observed.json", f"{E}/20261016T070746Z_d1_full/results/02_power.json", f"{E}/20261016T070746Z_d1_full/results/03_pc4v.json"])
    s = f"{E}/20261016T133134Z_d1_stage1_checks/"
    df = rd(s + "check1_ind_by_pigd_stratum.csv")
    d = pick(df, ["stratum", "n", "count", "share"], {"count": "Indeterminate visits", "n": "visits"}); d["patient-bootstrap 95% CI"] = ci(df, "ci_lo", "ci_hi", 3)
    d["generator expected"] = df["band_expected"].round(1); d["predictive 95% interval"] = ci(df, "band_pred_lo", "band_pred_hi", 1); d["excess"] = df["band_excess"].round(1); d["share of total excess"] = df["band_excess_share_of_total"].round(3)
    dl = rd(s + "check1_delatticed.csv")
    tab(16, "d1_stage1_indeterminate_by_pigd_sum", d,
        "D-1 Stage 1, check 1: Indeterminate share by PIGD item sum at the visit, against the covariate-blind expectation under the frozen generator. De-latticed total (observed count divided by the lattice inflation factor, summed over strata): "
        + f"{dl['delatticed'].sum():.1f} of {int(dl['n'].sum())} visits (share {dl['delatticed'].sum()/dl['n'].sum():.3f}) against {int(dl['ind'].sum())} observed.",
        [s + "check1_ind_by_pigd_stratum.csv", s + "check1_delatticed.csv"])
    df = rd(s + "check2_direction_outcome.csv")
    d = pick(df, ["direction", "revert", "stay", "to_Ind", "unobserved", "R_D", "R_D_ci"], {"R_D_ci": "R_D patient-bootstrap 95% CI"})
    d["generator R_D (theta range)"] = df["band_R_D"]
    tab(17, "d1_stage1_direct_reversion_direction", d,
        "D-1 Stage 1, check 2: outcome of direct TD<->PIGD changes by direction, observed against the generator expectation. R_D = revert / (revert + stay + to_Ind).",
        [s + "check2_direction_outcome.csv"])

    # -------------------------------------------------- early chains and CT-HMM
    df = rd(f"{O}/model/heldout_performance.csv")
    rb = rd(f"{O}/model/robustness_across_splits.csv")
    tab(18, "early_split_heldout", df,
        f"Held-out performance on the 70/30 patient-stratified split (1,002 consecutive pairs). Across ten further random splits the HMM minus discrete-Markov difference in held-out log-likelihood per pair had mean {rb['HMM - DTMC'].mean():+.4f} (SD {rb['HMM - DTMC'].std(ddof=1):.4f}) and was positive in {(rb['HMM - DTMC'] > 0).sum()} of {len(rb)} splits.",
        [f"{O}/model/heldout_performance.csv", f"{O}/model/robustness_across_splits.csv"])
    em = rd(f"{O}/model/hmm_emission.csv", index_col=0); em.index.name = "latent state"; em = em.reset_index()
    tab(19, "cthmm_emission_matrix", em,
        "Fitted misclassification (emission) matrix E of the three-state CT-HMM on the training split: rows are latent states, columns the probability of each recorded label. The Indeterminate row sits exactly at the emission floor of 0.5.",
        [f"{O}/model/hmm_emission.csv"])
    lg = rd(f"{O}/model/hmm_latent_generator.csv", index_col=0); cg = rd(f"{O}/model/ctmc_generator.csv", index_col=0); so = rd(f"{O}/model/mean_sojourn.csv", index_col=0)
    rows = []
    for a in lg.index:
        for b in lg.columns:
            if a != b:
                rows.append((f"rate {a} -> {b} (per year)", round(cg.loc[a, b], 4), round(lg.loc[a, b], 4)))
    for a in so.index:
        rows.append((f"mean sojourn in {a} (years)", round(so.loc[a, "CTMC"], 2), round(so.loc[a, "HMM (latent)"], 2)))
    tab(20, "generators_and_sojourn", pd.DataFrame(rows, columns=["quantity", "CTMC (observed label)", "CT-HMM (latent state)"]),
        "Off-diagonal generator rates and mean sojourn times for the continuous-time chain in the observed label and for the latent chain of the CT-HMM (training split). The latent quantities describe the unobserved state, not the recorded label.",
        [f"{O}/model/ctmc_generator.csv", f"{O}/model/hmm_latent_generator.csv", f"{O}/model/mean_sojourn.csv"])
    ch = rd(f"{O}/model/clinical_table_hmm.csv"); cc = rd(f"{O}/model/clinical_table_ctmc.csv")
    ch = ch[ch["horizon_years"].isin([1.0, 3.0, 5.0])].copy(); cc = cc[cc["horizon_years"].isin([1.0, 3.0, 5.0])].copy()
    ch["model"] = "CT-HMM (latent to latent)"; cc["model"] = "CTMC (recorded to recorded)"
    tab(21, "horizon_probabilities", pd.concat([cc, ch]),
        "Model-implied probability of being in each state after 1, 3 and 5 years, by current state. The CTMC rows are recorded-label to recorded-label probabilities; the CT-HMM rows are latent-state to latent-state probabilities and are not probabilities of the recorded label.",
        [f"{O}/model/clinical_table_ctmc.csv", f"{O}/model/clinical_table_hmm.csv"])

    # ----------------------------------------------------------------- LOPO
    df = rd(f"{O}/loo/loo_model_comparison.csv")
    d = pick(df, ["model", "n_predictions", "accuracy"]); d["patient-bootstrap 95% CI"] = ci(df, "accuracy_ci_patient_bootstrap_low", "accuracy_ci_patient_bootstrap_high")
    d["mean per-patient accuracy"] = df["mean_per_patient_accuracy"]; d["log-likelihood per prediction"] = df["loglik_per_prediction"]; d["Brier (3-class)"] = df["brier"]
    tab(22, "lopo_three_class_comparison", d,
        "Leave-one-patient-out three-class next-label prediction (934 folds, 3,361 predictions). Persistence is a one-hot rule, so its log-likelihood and Brier score are degenerate.",
        [f"{O}/loo/loo_model_comparison.csv"])
    df = rd(f"{O}/loo/loo_per_class_metrics.csv")
    tab(23, "lopo_per_class", df, "Per-class precision, recall and F1 of the CT-HMM under leave-one-patient-out.", [f"{O}/loo/loo_per_class_metrics.csv"])
    a = rd(f"{O}/loo/loo_accuracy_by_history.csv"); a.insert(0, "stratum", "prior labelled visits"); a = a.rename(columns={"bucket": "level", "mean": "accuracy", "size": "n"})
    b = rd(f"{O}/loo/loo_accuracy_by_current_state.csv"); b.insert(0, "stratum", "current label"); b = b.rename(columns={"current": "level", "mean": "accuracy", "size": "n"})
    tab(24, "lopo_accuracy_by_history_and_label", pd.concat([a, b]), "CT-HMM leave-one-patient-out accuracy by number of prior labelled visits and by the current label.",
        [f"{O}/loo/loo_accuracy_by_history.csv", f"{O}/loo/loo_accuracy_by_current_state.csv"])

    s = f"{E}/20261014T100057Z_s4_hmm_cv/analysis/"
    df = rd(s + "cv_config_metrics.csv")
    keep = ["C3_f50", "C3_free", "C4_free", "C5_free", "CTMC (interval-aware)", "Markov (discrete)", "Persistence (stay)", "Marginal (null)"]
    df = df[df["config"].isin(keep)]
    d = pick(df, ["config", "rule", "accuracy"]); d["95% CI"] = ci(df, "accuracy_lo", "accuracy_hi")
    for c in ["balanced_accuracy", "recall_TD", "recall_Indeterminate", "recall_PIGD", "n_predicted_Indeterminate", "log_loss", "brier", "max_p_Indet"]:
        d[c] = df[c]
    tab(25, "hmm_stability_configurations", d,
        "Stage 4 HMM stability study: held-out leave-one-patient-out metrics for alternative CT-HMM configurations (C3_f50 = three states with emission floor 0.5, the audited model; C3_free, C4_free, C5_free = three, four and five states without the floor) under the argmax rule and under a prior-corrected decision rule that divides each predicted probability by its class prevalence. Persistence is a one-hot rule and its probabilistic scores are degenerate.",
        [s + "cv_config_metrics.csv"])
    s2 = f"{E}/20261014T094649Z_s4_hmm_stability/analysis/"
    df = rd(s2 + "population_summary.csv")
    d = pick(df, ["config", "n_starts", "n_params", "best_loglik", "worst_loglik", "share_within_0.01", "n_loglik_levels_0.01", "max_abs_dQ_among_best", "max_abs_dE_among_best"])
    tab(26, "hmm_stability_restarts", d,
        "Stage 4 HMM stability study: full-cohort fits from 40 random starts per configuration. share_within_0.01 is the share of starts whose log-likelihood lies within 0.01 of the best; n_loglik_levels_0.01 the number of distinct optima at that resolution; the last two columns the largest difference in any generator or emission entry among the best starts.",
        [s2 + "population_summary.csv"])

    # ---------------------------------------------------------- growth model
    df = rd(f"{O}/hbayes/posterior_summary.csv")
    tab(27, "growth_model_posterior", df,
        "Hierarchical Bayesian growth model of the ratio (Gibbs sampler, single chain): posterior mean, SD and quantiles of the population parameters. No R-hat or effective sample size was computed, so the summaries are descriptive.",
        [f"{O}/hbayes/posterior_summary.csv"])
    h = rd(f"{O}/hbayes/head_to_head.csv"); r = rd(f"{O}/hbayes/robustness.csv")
    tab(28, "growth_vs_cthmm_head_to_head", h,
        f"Head-to-head held-out comparison of the growth model and the CT-HMM on the {int(h['n'].iloc[0])} pairs both can score. Across six further splits the CT-HMM minus growth-model difference in log-likelihood per pair was {r['HMM - HB'].mean():+.4f} on average (SD {r['HMM - HB'].std(ddof=1):.4f}) and positive in {(r['HMM - HB']>0).sum()} of {len(r)} splits.",
        [f"{O}/hbayes/head_to_head.csv", f"{O}/hbayes/robustness.csv"])
    tab(29, "growth_longitudinal_cv", rd(f"{O}/validation/longitudinal_cv.csv"),
        "Longitudinal cross-validation of the growth model: the first one, two or three visits of each held-out patient are used to estimate that patient's trajectory and all later visits are predicted.",
        [f"{O}/validation/longitudinal_cv.csv"])
    tab(30, "growth_calibration", rd(f"{O}/validation/calibration_growth_model.csv"),
        "Calibration of the growth model's predicted probability of a PIGD label.", [f"{O}/validation/calibration_growth_model.csv"])
    tab(31, "ratio_variance_decomposition", rd(f"{O}/exploration/variance_decomposition.csv"),
        "Decomposition of the variance of the ratio into between-patient and within-patient components.", [f"{O}/exploration/variance_decomposition.csv"])
    tab(32, "ratio_availability_by_year", rd(f"{O}/exploration/ratio_availability_by_year.csv"),
        "Share of labelled visits at which the ratio is defined, by study year.", [f"{O}/exploration/ratio_availability_by_year.csv"])

    # ------------------------------------------------------------ new patient
    mc = rd(f"{O}/newpatient/model_comparison.csv")
    cols = [c for c in mc.columns if c not in ("split", "n")]
    d = pd.DataFrame({"model": cols, "mean held-out log-likelihood per prediction (10 splits)": [round(mc[c].mean(), 4) for c in cols],
                      "SD across splits": [round(mc[c].std(ddof=1), 4) for c in cols]})
    hm = rd(f"{O}/newpatient/heldout_metrics.csv")
    tab(33, "new_patient_models", d,
        f"New-patient (first-visit) prediction: mean over ten random splits of the held-out log-likelihood per prediction for nested multinomial models. The first-visit model with ratio and covariates on the main split (n = {int(hm['n'].iloc[0])}): log-likelihood per prediction {hm['loglik_per_prediction'].iloc[0]:.4f}, accuracy {hm['accuracy'].iloc[0]:.4f}, Brier {hm['brier'].iloc[0]:.4f}.",
        [f"{O}/newpatient/model_comparison.csv", f"{O}/newpatient/heldout_metrics.csv"])
    tab(34, "new_patient_covariate_importance", rd(f"{O}/newpatient/covariate_importance.csv"),
        "Cost in held-out log-likelihood per prediction of dropping each covariate from the first-visit model.", [f"{O}/newpatient/covariate_importance.csv"])
    tab(35, "new_patient_by_horizon", rd(f"{O}/newpatient/by_horizon.csv"),
        "First-visit model: held-out log-likelihood and accuracy by prediction horizon.", [f"{O}/newpatient/by_horizon.csv"])

    # --------------------------------------------------------------- GPR ratio
    s = f"{O}/step3_gpr/"
    df = rd(s + "reconciled_metrics.csv"); df = df[df["cohort"] == "unfiltered"]
    d = pick(df, ["model", "n_patients", "n_points", "MAE_pooled", "MAE_patient", "RMSE", "CRPS", "coverage95", "negative_mass", "accuracy"])
    tab(36, "gpr_ratio_metrics", d,
        "Gaussian-process regression of the ratio, one-step-ahead leave-one-patient-out on the unfiltered cohort (821 patients, 2,707 points): kernels K0 (constant + linear), K1 (constant + RBF), K2 (constant + linear + RBF), K3 (constant + linear + Matern-3/2), the hierarchical Bayesian growth model and rule baselines. negative_mass is the mean predictive mass below zero; accuracy is the three-class accuracy of the derived label.",
        [s + "reconciled_metrics.csv"])
    df = rd(s + "paired_comparisons.csv"); df = df[df["cohort"] == "unfiltered"]
    d = pick(df, ["model_a", "model_b", "question", "diff_pooled"]); d["pooled 95% CI"] = ci(df, "pooled_lo", "pooled_hi"); d["diff_patient"] = df["diff_patient"]; d["patient-level 95% CI"] = ci(df, "patient_lo", "patient_hi")
    tab(37, "gpr_ratio_paired", d, "GPR of the ratio: pre-specified paired MAE differences (model a minus model b; negative favours a) with patient-cluster bootstrap intervals (4,000 replicates, seed 42), pooled over points and averaged within patients.",
        [s + "paired_comparisons.csv"])
    df = rd(s + "cohort_comparison.csv")
    tab(38, "gpr_cohort_comparison", df, "GPR of the ratio across the four extreme-patient filtering cohorts (unfiltered, A4, D1, E2).", [s + "cohort_comparison.csv"])
    df = rd(s + "mean_function_holdout_summary.csv")
    d = pick(df, ["model", "n_points", "n_patients", "MAE_pooled", "MAE_patient", "RMSE", "CRPS", "coverage95", "negative_mass", "diff_pooled"]); d["pooled 95% CI (zero-mean minus fitted-mean MAE)"] = ci(df, "diff_pooled_lo", "diff_pooled_hi")
    tab(39, "gpr_mean_function_holdout", d, "Held-out comparison of the K2 kernel with a fitted linear mean against the same kernel with a zero mean (unfiltered cohort, 821 folds).", [s + "mean_function_holdout_summary.csv"])
    df = rd(s + "multistep_by_horizon.csv"); df = df[df["cohort"] == "unfiltered"]
    pv_ = df.pivot_table(index="model", columns="horizon_bin", values="MAE").reset_index()
    tab(40, "gpr_multistep_mae_by_horizon", pv_, "Multi-step GPR forecasts (unfiltered cohort): MAE by forecast-horizon bin when the first half of each patient's visits conditions the forecast of every later visit.", [s + "multistep_by_horizon.csv"])
    df = rd(s + "multistep_paired.csv"); df = df[df["cohort"] == "unfiltered"]
    d = pick(df, ["model_a", "model_b", "n_points", "diff"]); d["95% CI"] = ci(df, "lo", "hi"); d["excludes zero"] = df["ci_excludes_zero"]
    tab(41, "gpr_multistep_paired", d, "Multi-step GPR forecasts: paired MAE differences over all forecast points (unfiltered cohort).", [s + "multistep_paired.csv"])
    df = rd(s + "multistep_audit_within_patient.csv"); df = df[(df["cohort"] == "unfiltered") & (df["model"].isin(["K1", "persistence"]))]
    d = pick(df, ["model", "from_bin", "to_bin", "n_patients", "delta_MAE"]); d["95% CI"] = ci(df, "lo", "hi")
    tab(42, "gpr_multistep_within_patient", d, "Multi-step GPR forecasts: within-patient change in MAE between adjacent horizon bins (positive = worse), restricted to patients present in both bins.", [s + "multistep_audit_within_patient.csv"])

    # --------------------------------------------------------------- subscores
    df = rd(s + "subscore_summary.csv"); d = pick(df, ["subscore", "model", "cohort", "n", "n_patients", "MAE"]); d["95% CI"] = ci(df, "lo", "hi"); d["negative_mass"] = df["negative_mass"]
    tab(43, "subscore_gp_summary", d, "Independent K1 GPs on the tremor and PIGD subscores (leave-one-patient-out, 934 folds): MAE on the expanded set (all 3,361 points) and on the restricted set the ratio model can score.", [s + "subscore_summary.csv"])
    df = rd(f"{O}/joint_subscore_gp/paired.csv"); d = pick(df, ["outcome", "comparison", "delta_MAE"]); d["95% CI"] = ci(df, "lo", "hi"); d["excludes zero"] = df["excludes_zero"]; d["n"] = df["n"]
    tab(44, "subscore_gp_paired", d, "Subscore GPs: paired MAE differences against persistence and between the independent and multi-output specifications (3,361 points).", [f"{O}/joint_subscore_gp/paired.csv"])
    s3 = f"{E}/20261014T094751Z_s3_subscore_heldout/analysis/"
    df = rd(s3 + "continuous_metrics.csv")
    keep = ["persistence", "patient_mean", "lmm_ri", "gp_indep_K0", "gp_indep_K1", "gp_indep_K2", "gp_indep_K3", "gp_joint_K1_full", "gp_joint_K3_full"]
    df = df[df["model"].isin(keep)]
    d = pick(df, ["outcome", "model", "MAE"]); d["MAE 95% CI"] = ci(df, "MAE_lo", "MAE_hi"); d["CRPS"] = df["CRPS"]; d["log score"] = df["log_score"]; d["coverage 95%"] = df["cov95"]; d["negative mass"] = df["mean_negative_mass"]
    tab(45, "stage3_subscore_heldout", d.sort_values(["outcome", "model"]),
        "Stage 3 held-out subscore comparison (934 folds, 3,361 points per arm): MAE, CRPS, log predictive score and 95% interval coverage for the independent GPs, the full-coregionalisation multi-output GPs, the random-intercept LMM and the rule baselines.",
        [s3 + "continuous_metrics.csv"])
    df = rd(s3 + "paired.csv")
    df = df[df["metric"].isin(["crps", "log_score"]) & (
        ((df["family"] == "1_nested_cross_output") & df["model"].isin(["gp_joint_K1_full", "gp_joint_K3_full"])) |
        (df["model"].isin(["gp_indep_K1", "gp_indep_K3", "gp_indep_K2"]) & df["reference"].isin(["persistence", "gp_indep_K0", "gp_indep_K1"])))]
    d = pick(df, ["family", "model", "reference", "outcome", "metric", "difference"]); d["95% CI"] = ci(df, "lo", "hi", 5); d["better"] = df["better"]
    tab(46, "stage3_paired_families", d, "Stage 3 pre-specified paired comparisons (model minus reference; negative CRPS and positive log score favour the model): the cross-output coregionalisation block against its diagonal counterpart, and the independent kernels against persistence and against each other.", [s3 + "paired.csv"])
    df = rd(s3 + "subtype_paired_vs_hmm.csv"); df = df[df["model"].isin(["gp_indep_K1", "gp_joint_K1_full", "gp_indep_K3", "lmm_ri", "persistence", "patient_mean"]) & df["metric"].isin(["log_loss", "accuracy_prob_argmax"])]
    d = pick(df, ["model", "metric", "difference"]); d["95% CI"] = ci(df, "lo", "hi"); d["excludes zero"] = df["excludes_zero"]
    tab(47, "stage3_subtype_vs_cthmm", d, "Stage 3: derived three-class label from each subscore model against the CT-HMM's leave-one-patient-out predictions on the same 3,361 points (difference = model minus CT-HMM; lower log loss and higher accuracy favour the model).", [s3 + "subtype_paired_vs_hmm.csv"])
    txt = (ROOT / O / "joint_subscore/model_summary.txt").read_text()
    rows = []
    for line in txt.splitlines():
        m = re.match(r"\s*(alpha_tremor|beta_tremor|alpha_pigd|beta_pigd)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
        if m:
            rows.append((m.group(1), float(m.group(2)), float(m.group(3)), f"[{float(m.group(4)):.4f}, {float(m.group(5)):.4f}]"))
    def g(p):
        m = re.search(p, txt); return m.group(1) if m else ""
    rows += [("random-intercept correlation", float(g(r"correlation (-?[\d.]+)\n\nRESIDUAL")), np.nan, ""), ("residual correlation", float(g(r"RESIDUAL COVARIANCE[\s\S]*?correlation \+?(-?[\d.]+)")), np.nan, ""),
             ("ICC tremor", float(g(r"ICC tremor ([\d.]+)")), np.nan, ""), ("ICC PIGD", float(g(r"ICC pigd ([\d.]+)")), np.nan, "")]
    tab(48, "joint_lmm_parameters", pd.DataFrame(rows, columns=["parameter", "estimate", "std_error", "95% CI"]),
        "Joint bivariate linear mixed model of the tremor and PIGD subscores (REML, random intercepts, 934 patients, 8,590 observations): fixed effects, random-intercept and residual correlations, and intraclass correlations.",
        [f"{O}/joint_subscore/model_summary.txt"])
    a = rd(f"{O}/joint_subscore/subtype_metrics.csv"); a.insert(0, "source", "joint LMM run")
    b = rd(f"{O}/joint_subscore_gp/subtype_metrics.csv"); b = b.rename(columns={"rule": "rule_"}); b.insert(0, "source", "multi-output GP run")
    c = rd(f"{O}/time_varying/subtype_metrics.csv"); c.insert(0, "source", "time-varying LMM run")
    cols = ["source", "model", "n", "accuracy", "accuracy_lo", "accuracy_hi", "balanced_accuracy", "recall_TD", "recall_Indeterminate", "recall_PIGD"]
    d = pd.concat([pick(a, cols), pick(b, cols), pick(c, cols)]); d["95% CI"] = ci(d, "accuracy_lo", "accuracy_hi"); d = d.drop(columns=["accuracy_lo", "accuracy_hi"])
    tab(49, "subscore_models_derived_subtype", d, "Three-class accuracy of the label derived from each subscore model's point prediction through the unchanged classification rule (3,361 points). These point-rule accuracies differ from the probability-argmax accuracies of the harmonised comparison (Table 56).",
        [f"{O}/joint_subscore/subtype_metrics.csv", f"{O}/joint_subscore_gp/subtype_metrics.csv", f"{O}/time_varying/subtype_metrics.csv"])

    # ------------------------------------------------------------ time-varying
    df = rd(f"{O}/time_varying/period_slopes.csv"); d = pick(df, ["outcome", "period", "slope", "std_error"]); d["Wald 95% CI"] = ci(df, "ci_low", "ci_high")
    tab(50, "time_varying_period_slopes", d, "Time-varying linear-spline mixed model (knots at 2 and 5 years, random intercepts): period-specific population slopes of the subscores per year, with Wald intervals.", [f"{O}/time_varying/period_slopes.csv"])
    df = rd(f"{O}/time_varying/slope_contrasts.csv"); d = pick(df, ["outcome", "contrast", "difference", "std_error"]); d["Wald 95% CI"] = ci(df, "ci_low", "ci_high"); d["excludes zero"] = df["excludes_zero"]
    tab(51, "time_varying_slope_contrasts", d, "Time-varying model: contrasts between period slopes (Wald).", [f"{O}/time_varying/slope_contrasts.csv"])
    df = rd(f"{O}/time_varying/within_patient_rate_change.csv"); d = pick(df, ["outcome", "from_period", "to_period", "n_patients", "delta_rate"]); d["95% CI"] = ci(df, "ci_low", "ci_high"); d["excludes zero"] = df["excludes_zero"]
    tab(52, "time_varying_within_patient", d, "Time-varying model: within-patient change in rate between periods, restricted to patients observed in both periods.", [f"{O}/time_varying/within_patient_rate_change.csv"])
    s5 = f"{E}/20261014T100107Z_s5_temporal_audit/"
    tab(53, "stage5_sup_lr", rd(s5 + "knots/sup_lr_test.csv"), "Stage 5 temporal audit: supremum likelihood-ratio test of a single slope change against a linear trend, with the null distribution from 200 parametric-bootstrap cohorts under random-intercept (RI) and random-slope (RS) nulls.", [s5 + "knots/sup_lr_test.csv"])
    df = rd(s5 + "bootstrap/period_slopes_boot_vs_wald.csv"); df = df[df["model"] == "primary_RI"]
    d = pick(df, ["outcome", "period", "slope"]); d["Wald 95% CI"] = ci(df, "wald_lo", "wald_hi"); d["patient-bootstrap 95% CI"] = ci(df, "boot_lo", "boot_hi"); d["SE ratio bootstrap/Wald"] = df["se_ratio_boot_over_wald"].round(2)
    tab(54, "stage5_boot_vs_wald", d, "Stage 5: period slopes of the primary random-intercept spline model with Wald and patient-bootstrap (1,000 replicates) intervals.", [s5 + "bootstrap/period_slopes_boot_vs_wald.csv"])
    df = rd(s5 + "attrition/loss_of_followup_logit.csv"); d = pick(df, ["term", "log_odds"]); d["bootstrap 95% CI"] = ci(df, "boot_lo", "boot_hi")
    tab(55, "stage5_attrition", d, "Stage 5: logistic model of loss to follow-up after a visit on the subscores at that visit and time from baseline.", [s5 + "attrition/loss_of_followup_logit.csv"])

    # --------------------------------------------------------------- Student-t
    st = js(f"{E}/20261015T190228Z_student_t_four_model_full/results.json"); q2 = js(f"{E}/20261015T203325Z_q2_reassessment/results/results.json")
    rows = []
    for k in ["GT-GG", "TT-TG"]:
        q = st["Q1_residual_tails"][k]
        rows.append(("Q1 residual tails", f"{k}: held-out log predictive density per visit", round(q["lpd_mean"], 4), f"[{q['lpd_ci'][0]:.4f}, {q['lpd_ci'][1]:.4f}]", q["lpd_splits_positive"]))
        rows.append(("Q1 residual tails", f"{k}: class-label metric", round(q["cls_mean"], 4), f"[{q['cls_ci'][0]:.4f}, {q['cls_ci'][1]:.4f}]", q["lpd_splits_positive"]))
    for k in ["TG", "GT", "TT"]:
        q = st["Q3_predictive_value"][k]
        rows.append(("Q3 predictive value vs GG", f"{k}: class-label metric", round(q["cls_mean"], 4), f"[{q['cls_ci'][0]:.4f}, {q['cls_ci'][1]:.4f}]", q["splits_positive"]))
    c = st["Q2_slope_tails"]["convergence"]["gamma"]
    rows.append(("Q2 slope tails, original run", "nu_slope sampler, Gamma prior: R-hat / ESS", f"{c['rhat']:.3f} / {c['ess']:.0f}", "", st["Q2_slope_tails"]["verdict"]))
    a = q2["a"]["gamma2_true"]; cv = q2["conv"]["gamma2_true"]
    rows.append(("Q2 reassessment", "full-cohort nu_slope posterior median [2.5%, 97.5%], Gamma(2, 0.1) prior", round(a["q50"], 2), f"[{a['q2.5']:.2f}, {a['q97.5']:.2f}]", f"R-hat {cv['rhat']:.4f}, ESS {cv['ess']:.0f}"))
    rows.append(("Q2 reassessment", "per-split criterion nu_slope q97.5 < 10 met", q2["b"], "", "threshold 5/7"))
    for k in ["C1", "C2", "C3", "P"]:
        cc_ = q2["c"][k]
        rows.append(("Q2 reassessment calibration", f"scenario {k}: flags / cohorts", f"{cc_['flags']} / {cc_['n']}", f"[{cc_['cp95_two_sided'][0]:.3f}, {cc_['cp95_two_sided'][1]:.3f}]", "null scenario" if k != "P" else "positive control"))
    rows.append(("Q2 reassessment", "verdict", q2["Q2_reassessment_verdict"], "", ""))
    tab(56, "student_t_variants", pd.DataFrame(rows, columns=["question", "quantity", "estimate", "95% interval", "splits / status"]),
        "Student-t variants of the growth model (GG Gaussian slopes and residuals, GT Gaussian slopes with t residuals, TG t slopes with Gaussian residuals, TT both t): held-out comparisons over seven patient splits with patient-bootstrap intervals, and the pre-registered reassessment of the slope-tail question.",
        [f"{E}/20261015T190228Z_student_t_four_model_full/results.json", f"{E}/20261015T203325Z_q2_reassessment/results/results.json"])

    # ------------------------------------------------------- harmonised E-8
    s = f"{E}/20261103T202632Z_e8_harmonised_family_comparison/"
    df = rd(s + "harmonised_metrics.csv")
    tab(57, "harmonised_three_class_metrics", df, "Harmonised three-class comparison (E-8): fourteen specifications re-scored on the same 3,361 transitions with one decision rule (argmax of a three-label probability vector). Subscore models are propagated through the unmodified classification rule by Monte Carlo (10,000 draws, common random numbers). The log loss of recorded-label persistence is clipped.", [s + "harmonised_metrics.csv"])
    df = rd(s + "paired_contrasts.csv"); d = pick(df, ["model", "family", "delta_accuracy"]); d["patient-bootstrap 95% CI"] = ci(df, "ci_low", "ci_high"); d["excludes zero"] = df["excludes_zero"]; d["n_better"] = df["n_better"]; d["n_worse"] = df["n_worse"]
    tab(58, "harmonised_contrasts_vs_persistence", d, "Harmonised comparison: paired accuracy difference of each specification against recorded-label persistence on identical cases.", [s + "paired_contrasts.csv"])
    tab(59, "harmonised_margin_decomposition", rd(s + "margin_decomposition.csv"), "Harmonised comparison: each model's accuracy and gain over recorded-label persistence by the label recorded at the prediction point, and the share of predictions identical to persistence.", [s + "margin_decomposition.csv"])

    # ---------------------------------------------------------- next visit
    s = f"{E}/20261020T104756Z_next_visit_td_pigd_prediction/"
    df = rd(s + "model_comparison.csv")
    d = pick(df, ["model", "n_patients", "n_cases_total", "n_eligible", "n_correct", "accuracy"]); d["patient-bootstrap 95% CI"] = ci(df, "accuracy_ci_patient_bootstrap_low", "accuracy_ci_patient_bootstrap_high")
    for c in ["mean_per_patient_accuracy", "brier_two_class", "log_loss", "log_loss_clipped", "mean_P_PIGD"]:
        d[c] = df[c]
    tab(60, "next_visit_two_class_models", d, "Next-visit TD-against-PIGD prediction on the 3,029 eligible transitions of 899 patients (332 Indeterminate-outcome transitions excluded from scoring), leave-one-patient-out. Observed PIGD rate 0.3017.", [s + "model_comparison.csv"])
    df = rd(s + "contrasts.csv"); d = pick(df, ["contrast", "model_a", "model_b", "accuracy_a", "accuracy_b", "accuracy_delta"]); d["patient-bootstrap 95% CI"] = ci(df, "delta_ci_patient_bootstrap_low", "delta_ci_patient_bootstrap_high"); d["a right, b wrong"] = df["a_right_b_wrong"]; d["b right, a wrong"] = df["b_right_a_wrong"]; d["Brier delta"] = df["brier_delta"]
    tab(61, "next_visit_two_class_contrasts", d, "Next-visit two-class prediction: pre-specified paired contrasts (model a minus model b) on identical cases.", [s + "contrasts.csv"])
    a = rd(s + "accuracy_by_current_label.csv"); a.insert(0, "stratum", "current label"); a = a.rename(columns={"current_label": "level"})
    b = rd(s + "accuracy_by_interval.csv"); b.insert(0, "stratum", "interval"); b = b.rename(columns={"interval": "level"})
    c = rd(s + "accuracy_by_history_length.csv"); c.insert(0, "stratum", "prior visits"); c = c.rename(columns={"n_prior_visits": "level"})
    d = pd.concat([a, b, c]); d = d[d["model"].isin(["CT-HMM (latent, interval-aware)", "CTMC (observed, interval-aware)", "Markov (observed, discrete)", "Direct logistic (history + interval)", "Persistence (last non-Indeterminate)"])]
    tab(62, "next_visit_two_class_strata", d, "Next-visit two-class accuracy by current label, by elapsed interval and by number of prior visits.", [s + "accuracy_by_current_label.csv", s + "accuracy_by_interval.csv", s + "accuracy_by_history_length.csv"])
    tab(63, "next_visit_indeterminate_targets", rd(s + "indeterminate_report.csv"), "The 332 transitions whose next recorded label was Indeterminate: never scored; what each model would have said.", [s + "indeterminate_report.csv"])
    pc = rd(s + "per_class_ct_hmm_latent_interval_aware.csv"); cf = rd(s + "confusion_ct_hmm_latent_interval_aware.csv")
    tab(64, "next_visit_cthmm_per_class", pc, "CT-HMM next-visit two-class prediction: per-class metrics. Confusion matrix (rows predicted, columns actual): " + "; ".join(f"predicted {r['predicted_state']}: TD {int(r['TD'])}, PIGD {int(r['PIGD'])}" for _, r in cf.iterrows()) + ".", [s + "per_class_ct_hmm_latent_interval_aware.csv", s + "confusion_ct_hmm_latent_interval_aware.csv"])

    # ----------------------------------------------------- latent clinical states
    s = f"{E}/20261020T142040Z_clinical_state_selection/"
    tab(65, "clinical_state_count_selection", rd(s + "state_count_summary.csv"), "Latent clinical-state CT-HMM (Gaussian emission over ten standardised measurements): state-count selection by held-out next-visit log-density on the 280-patient development set (four cells). max_start_spread is the spread of the training log-likelihood across random starts.", [s + "state_count_summary.csv"])
    rows = []
    for k, sdir in [("K = 6", f"{E}/20261020T142446Z_clinical_state_lopo_k6/"), ("K = 8", f"{E}/20261020T145228Z_clinical_state_lopo_k8/")]:
        mc = rd(sdir + "model_comparison.csv"); ct = rd(sdir + "contrasts.csv"); al = rd(sdir + "accuracy_by_current_label.csv")
        for _, r in mc.iterrows():
            acc_l = al[al["model"] == r["model"]].set_index("current_label")["accuracy"]
            rows.append((k, r["model"], int(r["n_patients"]), int(r["n_scored"]), r["accuracy"], f"[{r['accuracy_ci_patient_bootstrap_low']:.4f}, {r['accuracy_ci_patient_bootstrap_high']:.4f}]", r["brier_two_class"], r["log_loss"], acc_l.get("TD", np.nan), acc_l.get("Indeterminate", np.nan), acc_l.get("PIGD", np.nan)))
        r = ct.iloc[0]
        rows.append((k, "contrast: clinical-state minus subtype-history CT-HMM", int(r["n_paired"]) if "n_patients" not in ct.columns else int(r["n_paired"]), int(r["n_paired"]), r["accuracy_delta"], f"[{r['delta_ci_patient_bootstrap_low']:.4f}, {r['delta_ci_patient_bootstrap_high']:.4f}]", r["brier_delta"], np.nan, np.nan, np.nan, np.nan))
    tab(66, "clinical_state_lopo", pd.DataFrame(rows, columns=["states", "model", "n_patients (or n_paired)", "n_scored", "accuracy (or delta)", "patient-bootstrap 95% CI", "Brier (or delta)", "log loss", "accuracy | current TD", "accuracy | current Indeterminate", "accuracy | current PIGD"]),
        "Latent clinical-state CT-HMM against the subtype-history CT-HMM and persistence on the 654-patient evaluation set (leave-one-patient-out; 2,091 scored two-class cases, 627 patients with cases), for six and eight states.",
        [f"{E}/20261020T142446Z_clinical_state_lopo_k6/model_comparison.csv", f"{E}/20261020T142446Z_clinical_state_lopo_k6/contrasts.csv", f"{E}/20261020T142446Z_clinical_state_lopo_k6/accuracy_by_current_label.csv", f"{E}/20261020T145228Z_clinical_state_lopo_k8/model_comparison.csv", f"{E}/20261020T145228Z_clinical_state_lopo_k8/contrasts.csv", f"{E}/20261020T145228Z_clinical_state_lopo_k8/accuracy_by_current_label.csv"])

    # ----------------------------------------------------------- rich profile
    s = f"{E}/20261103T220147Z_a2p_outcome_table_fullprec/"
    df = rd(s + "model_comparison.csv"); d = pick(df, ["model", "n_predictions", "n_patients", "n_domain_observations", "loglik_per_domain_observation"]); d["patient-bootstrap 95% CI"] = ci(df, "loglik_ci_low", "loglik_ci_high"); d["rmse_standardised"] = df["rmse_standardised"]; d["mae_standardised"] = df["mae_standardised"]; d["coverage_95"] = df["coverage_95"]
    tab(67, "profile_log_density", d, "Rich clinical profile (per-domain local linear trend with LEDD as input; 5-fold patient cross-validation repeated four times, 934 patients, 3,353 predictions, 30,099 domain observations): held-out log-density per domain observation and standardised errors.", [s + "model_comparison.csv"])
    df = rd(s + "contrasts.csv"); d = pick(df, ["contrast", "model_a", "model_b", "n_paired", "n_patients", "loglik_delta"]); d["patient-bootstrap 95% CI"] = ci(df, "delta_ci_low", "delta_ci_high"); d["share a better"] = df["share_a_better"]
    tab(68, "profile_contrasts", d, "Rich clinical profile: paired log-density contrasts (model a minus model b, per prediction).", [s + "contrasts.csv"])
    s3_ = f"{E}/20261104T085320Z_a3_profile_cv_with_baseline/"
    df = rd(s3_ + "derived_label_vs_subtype_history.csv"); d = pick(df, ["model", "n_patients", "n_eligible", "n_correct", "accuracy"]); d["patient-bootstrap 95% CI"] = ci(df, "accuracy_ci_patient_bootstrap_low", "accuracy_ci_patient_bootstrap_high"); d["brier_two_class"] = df["brier_two_class"]; d["log_loss"] = df["log_loss"]; d["log_loss_clipped"] = df["log_loss_clipped"]
    ct = rd(s3_ + "derived_label_contrasts.csv")
    tab(69, "profile_derived_label", d, "Two-class next-visit label derived from the rich clinical profile against the subtype-history models on the same 3,020 eligible transitions (898 patients; A-3 run with the committed code). Contrasts: " + "; ".join(f"{r['contrast']}: {r['accuracy_delta']:+.4f} [{r['delta_ci_patient_bootstrap_low']:+.4f}, {r['delta_ci_patient_bootstrap_high']:+.4f}]" for _, r in ct.iterrows()) + ".",
        [s3_ + "derived_label_vs_subtype_history.csv", s3_ + "derived_label_contrasts.csv"])
    df = rd(s + "h3_component_vs_subtype.csv"); d = pick(df, ["outcome", "kind", "best_component", "n", "n_positive", "auc_a", "auc_b", "delta"], {"auc_a": "AUC best component", "auc_b": "AUC subtype label"}); d["patient-bootstrap 95% CI"] = ci(df, "delta_ci_low", "delta_ci_high")
    tab(70, "profile_outcome_discrimination", d, "Discrimination of subsequently recorded outcomes (never used in fitting) by the best single level or rate coordinate of the profile against the subtype label. The best coordinate was selected on the same data, so each difference is an upper bound. The cognitive diagnosis outcome is not fully independent of the inputs.", [s + "h3_component_vs_subtype.csv"])
    rdm = rd(s + "representation_dimensionality.csv"); d = pick(rdm, ["which", "n_patients", "n_domains", "pc1_share", "pc2_share", "cumulative_3", "n_components_for_90pct", "effective_rank_participation"])
    tab(71, "profile_representation_dimensionality", d, "Dimensionality of the profile's level and rate coordinates across the 934 patients: principal-component variance shares and participation-ratio effective rank.", [s + "representation_dimensionality.csv"])
    tab(72, "profile_rate_spread", rd(s + "rate_spread.csv"), "Spread of the filtered per-patient rates by domain (per year).", [s + "rate_spread.csv"])

    # ---------------------------------------------------- selection and AUC
    s = f"{E}/20261023T194317Z_model_selection_criteria/"
    tab(73, "aic_bic", rd(s + "aic_bic.csv"), "Information criteria on the full cohort. Groups are comparable only within themselves: the chains' conditional pair likelihood (B) is not comparable with the joint sequence likelihood (C), and the profile's continuous density (A) with neither.", [s + "aic_bic.csv"])
    s = f"{E}/20261023T202640Z_roc_auc_comparison/"
    df = rd(s + "auc_metrics.csv"); d = pick(df, ["model", "n", "n_patients", "n_positive", "auc"]); d["patient-bootstrap 95% CI"] = ci(df, "ci_low", "ci_high")
    ct = rd(s + "auc_contrast.csv"); al = rd(s + "auc_by_current_label.csv")
    tab(74, "auc_next_visit", d, "Threshold-free discrimination of next-visit PIGD against TD (AUC-ROC, PIGD positive) on the 3,020 matched transitions. Contrast " + "; ".join(f"{r['contrast']}: {r['delta']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]" for _, r in ct.iterrows()) + ". By current label: " + "; ".join(f"{r['current_label']} (n={int(r['n'])}, positives {int(r['n_positive'])}): CT-HMM {r['ct_hmm_auc']:.4f}, profile {r['profile_auc']:.4f}" for _, r in al.iterrows()) + ".",
        [s + "auc_metrics.csv", s + "auc_contrast.csv", s + "auc_by_current_label.csv"])
    s = f"{E}/20261023T203431Z_indeterminate_boundary/"
    df = rd(s + "boundary_agreement.csv"); d = pick(df, ["model", "n", "n_patients", "predicted_PIGD", "predicted_TD", "agreement_with_nearer_boundary"]); d["agreement 95% CI"] = ci(df, "agreement_ci_low", "agreement_ci_high"); d["expected under independence"] = df["expected_under_independence"]; d["excess over null"] = df["excess_over_null"]; d["excess 95% CI"] = ci(df, "excess_ci_low", "excess_ci_high"); d["AUC P_PIGD for nearer-PIGD"] = df["auc_P_PIGD_for_nearer_PIGD"]; d["AUC 95% CI"] = ci(df, "auc_ci_low", "auc_ci_high")
    sv = rd(s + "sensitivity_close_to_boundary.csv")
    tab(75, "indeterminate_boundary_agreement", d, "At Indeterminate visits with a defined ratio (287 visits, 220 patients), does the two-class call agree with the nearer band boundary more often than independence implies? Sensitivity on the 213 visits closest to a boundary: " + "; ".join(f"{r['model']} excess {r['excess']:+.4f} [{r['excess_ci_low']:+.4f}, {r['excess_ci_high']:+.4f}]" for _, r in sv.iterrows()) + ".",
        [s + "boundary_agreement.csv", s + "sensitivity_close_to_boundary.csv"])
    df = rd(f"{E}/20261023T211836Z_indeterminate_ratio_scrutiny/patient_clustered_intervals.csv"); d = pick(df, ["quantity", "model", "estimate"]); d["patient-bootstrap 95% CI"] = ci(df, "ci_low", "ci_high"); d["n_visits"] = df["n_visits"]; d["n_patients"] = df["n_patients"]; d["excludes zero"] = df["excludes_zero"]
    tab(76, "focal_ratio_calls", d, "Calls at the two focal lattice ratios inside the Indeterminate band (0.9091, nearer PIGD; 1.1364, nearer TD): probabilities and label contrasts with patient-clustered intervals.", [f"{E}/20261023T211836Z_indeterminate_ratio_scrutiny/patient_clustered_intervals.csv"])

    # ------------------------------------------------------------- stage-wise
    s = f"{E}/20261023T214819Z_stage_wise_evaluation/"
    df = rd(s + "stage_metrics.csv"); df = df[df["stage"] <= 9]
    d = pick(df, ["stage", "model", "n_targets", "n_patients", "pct_PIGD", "accuracy"]); d["accuracy 95% CI"] = ci(df, "accuracy_lo", "accuracy_hi"); d["balanced_accuracy"] = df["balanced_accuracy"]; d["balanced 95% CI"] = ci(df, "balanced_accuracy_lo", "balanced_accuracy_hi"); d["sensitivity_PIGD"] = df["sensitivity_PIGD"]; d["brier"] = df["brier"]; d["log_loss"] = df["log_loss"]
    tab(77, "stage_wise_metrics", d, "Stage-wise evaluation: two-class metrics at each prediction stage (stage k predicts visit k+1 from visits 1..k), stages 1 to 9 (at least 66 targets each).", [s + "stage_metrics.csv"])
    pdf_ = rd(s + "paired_differences.csv"); pdf_ = pdf_[pdf_["stage"] <= 9]; d = pick(pdf_, ["stage", "metric", "n_targets", "estimate"]); d["95% CI"] = ci(pdf_, "ci_low", "ci_high"); d["excludes zero"] = pdf_["excludes_zero"]
    wc = rd(s + "within_patient_change.csv")
    tab(78, "stage_wise_paired", d, "Stage-wise evaluation: paired difference, rich clinical profile minus CT-HMM, at each stage. Within-patient late-minus-early change in accuracy (225 patients with both): " + "; ".join(f"{r['model']} {r['mean_within_patient_change']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]" for _, r in wc.iterrows()) + ".",
        [s + "paired_differences.csv", s + "within_patient_change.csv"])

    # ------------------------------------------------------- history windows
    s = f"{E}/20261024T042417Z_history_window_experiment/"
    df = rd(s + "window_metrics.csv"); d = pick(df, ["cohort", "window", "n_targets", "n_patients", "accuracy"]); d["accuracy 95% CI"] = ci(df, "accuracy_lo", "accuracy_hi"); d["balanced_accuracy"] = df["balanced_accuracy"]; d["sensitivity_PIGD"] = df["sensitivity_PIGD"]; d["auc"] = df["auc"]; d["brier"] = df["brier"]; d["log_loss"] = df["log_loss"]
    tab(79, "cthmm_history_windows", d, "CT-HMM prediction with truncated history windows (last 1, 2, 3, 5 visits) against the full history, on all eligible targets and on the common targets available to every window (734 targets, 228 patients).", [s + "window_metrics.csv"])
    df = rd(s + "paired_comparisons.csv"); d = pick(df, ["role", "comparison", "n_targets", "accuracy"]); d["accuracy 95% CI"] = ci(df, "accuracy_lo", "accuracy_hi"); d["balanced_accuracy"] = df["balanced_accuracy"]; d["balanced 95% CI"] = ci(df, "balanced_accuracy_lo", "balanced_accuracy_hi"); d["auc"] = df["auc"]; d["auc 95% CI"] = ci(df, "auc_lo", "auc_hi"); d["brier"] = df["brier"]; d["brier 95% CI"] = ci(df, "brier_lo", "brier_hi"); d["log_loss"] = df["log_loss"]; d["log loss 95% CI"] = ci(df, "log_loss_lo", "log_loss_hi")
    tab(80, "cthmm_history_windows_paired", d, "CT-HMM history windows: paired differences on the common targets (first-named minus second-named).", [s + "paired_comparisons.csv"])
    s = f"{E}/20261024T042811Z_profile_history_window/"
    df = rd(s + "profile_window_metrics.csv"); d = pick(df, ["cohort", "window", "n_targets", "n_patients", "accuracy"]); d["accuracy 95% CI"] = ci(df, "accuracy_lo", "accuracy_hi"); d["balanced_accuracy"] = df["balanced_accuracy"]; d["sensitivity_PIGD"] = df["sensitivity_PIGD"]; d["auc"] = df["auc"]; d["brier"] = df["brier"]; d["log_loss"] = df["log_loss"]
    tab(81, "profile_history_windows", d, "Rich clinical profile with truncated history windows (last 2, last 3) against the full history, and the full history with the current-visit LEDD substituted for the next-visit LEDD.", [s + "profile_window_metrics.csv"])
    df = rd(s + "profile_paired_comparisons.csv"); d = pick(df, ["role", "comparison", "n_targets", "accuracy"]); d["accuracy 95% CI"] = ci(df, "accuracy_lo", "accuracy_hi"); d["auc"] = df["auc"]; d["auc 95% CI"] = ci(df, "auc_lo", "auc_hi"); d["log_loss"] = df["log_loss"]; d["log loss 95% CI"] = ci(df, "log_loss_lo", "log_loss_hi")
    tab(82, "profile_history_windows_paired", d, "Rich clinical profile history windows: paired differences on the common targets (1,489 targets, 481 patients).", [s + "profile_paired_comparisons.csv"])
    s = f"{E}/20261024T052343Z_retrained_two_visit_cthmm/"
    df = rd(s + "condition_metrics.csv"); d = pick(df, ["cohort", "condition", "n_targets", "n_patients", "accuracy"]); d["accuracy 95% CI"] = ci(df, "accuracy_lo", "accuracy_hi"); d["balanced_accuracy"] = df["balanced_accuracy"]; d["sensitivity_PIGD"] = df["sensitivity_PIGD"]; d["auc"] = df["auc"]; d["brier"] = df["brier"]; d["log_loss"] = df["log_loss"]
    tab(83, "cthmm_retrained_conditions", d, "CT-HMM retrained on two-visit windows: training (full sequences, overlapping two-visit windows, disjoint windows) crossed with prediction history (full, last two visits).", [s + "condition_metrics.csv"])
    df = rd(s + "paired_comparisons.csv"); d = pick(df, ["role", "comparison", "n_targets", "accuracy"]); d["accuracy 95% CI"] = ci(df, "accuracy_lo", "accuracy_hi"); d["balanced_accuracy"] = df["balanced_accuracy"]; d["balanced 95% CI"] = ci(df, "balanced_accuracy_lo", "balanced_accuracy_hi"); d["auc"] = df["auc"]; d["auc 95% CI"] = ci(df, "auc_lo", "auc_hi"); d["brier"] = df["brier"]; d["brier 95% CI"] = ci(df, "brier_lo", "brier_hi"); d["log_loss"] = df["log_loss"]; d["log loss 95% CI"] = ci(df, "log_loss_lo", "log_loss_hi")
    tab(84, "cthmm_retrained_paired", d, "CT-HMM retrained on two-visit windows: paired differences on the 2,192 common targets (742 patients).", [s + "paired_comparisons.csv"])
    tab(85, "cthmm_retrained_parameters", rd(s + "parameter_comparison.csv"), "CT-HMM parameters from the full-sequence fit and from the two-visit-window fit (one representative fold).", [s + "parameter_comparison.csv"])
    s = f"{E}/20261103T052237Z_profile_prediction_history/"
    df = rd(s + "window_metrics.csv"); d = pick(df, ["cohort", "window", "n_targets", "n_patients", "pigd_prevalence", "accuracy"]); d["accuracy 95% CI"] = ci(df, "accuracy_lo", "accuracy_hi"); d["balanced_accuracy"] = df["balanced_accuracy"]; d["sensitivity_PIGD"] = df["sensitivity_PIGD"]; d["specificity_TD"] = df["specificity_TD"]; d["auc"] = df["auc"]; d["brier"] = df["brier"]; d["log_loss"] = df["log_loss"]
    tab(86, "exp6a_profile_windows", d, "Experiment 6A: rich clinical profile with prediction history windows last 1, 2, 3, 5 against the full history (full-sequence training; 5-fold patient cross-validation).", [s + "window_metrics.csv"])
    df = rd(s + "paired_comparisons.csv"); d = pick(df, ["role", "comparison", "n_targets", "n_patients", "accuracy"]); d["accuracy 95% CI"] = ci(df, "accuracy_lo", "accuracy_hi"); d["balanced_accuracy"] = df["balanced_accuracy"]; d["balanced 95% CI"] = ci(df, "balanced_accuracy_lo", "balanced_accuracy_hi"); d["auc"] = df["auc"]; d["auc 95% CI"] = ci(df, "auc_lo", "auc_hi"); d["brier"] = df["brier"]; d["brier 95% CI"] = ci(df, "brier_lo", "brier_hi"); d["sensitivity_PIGD"] = df["sensitivity_PIGD"]; d["sensitivity 95% CI"] = ci(df, "sensitivity_PIGD_lo", "sensitivity_PIGD_hi"); d["specificity_TD"] = df["specificity_TD"]; d["specificity 95% CI"] = ci(df, "specificity_TD_lo", "specificity_TD_hi")
    tab(87, "exp6a_paired", d, "Experiment 6A: paired differences (first-named minus second-named) with patient-bootstrap intervals.", [s + "paired_comparisons.csv"])
    s = f"{E}/20261103T072507Z_profile_two_visit_retraining/"
    df = rd(s + "condition_metrics.csv"); d = pick(df, ["condition", "n_targets", "n_patients", "accuracy"]); d["accuracy 95% CI"] = ci(df, "accuracy_lo", "accuracy_hi"); d["balanced_accuracy"] = df["balanced_accuracy"]; d["sensitivity_PIGD"] = df["sensitivity_PIGD"]; d["specificity_TD"] = df["specificity_TD"]; d["auc"] = df["auc"]; d["brier"] = df["brier"]; d["log_loss"] = df["log_loss"]
    tab(88, "exp6b_conditions", d, "Experiment 6B: rich clinical profile trained on full sequences (A, B), on overlapping two-visit windows (C, D) or on disjoint windows (C', D'), predicting with the full history (A, C, C') or the last two visits (B, D, D'); 2,184 common targets, 740 patients.", [s + "condition_metrics.csv"])
    df = rd(s + "paired_comparisons.csv"); d = pick(df, ["role", "comparison", "n_targets", "accuracy"]); d["accuracy 95% CI"] = ci(df, "accuracy_lo", "accuracy_hi"); d["auc"] = df["auc"]; d["auc 95% CI"] = ci(df, "auc_lo", "auc_hi"); d["brier"] = df["brier"]; d["brier 95% CI"] = ci(df, "brier_lo", "brier_hi"); d["log_loss"] = df["log_loss"]; d["log loss 95% CI"] = ci(df, "log_loss_lo", "log_loss_hi"); d["sensitivity_PIGD"] = df["sensitivity_PIGD"]; d["sensitivity 95% CI"] = ci(df, "sensitivity_PIGD_lo", "sensitivity_PIGD_hi")
    tab(89, "exp6b_paired", d, "Experiment 6B: paired differences (first-named minus second-named).", [s + "paired_comparisons.csv"])
    tab(90, "exp6b_parameters", rd(s + "parameter_comparison.csv"), "Experiment 6B: rich-profile variance parameters (mean over domains) under the three training schemes.", [s + "parameter_comparison.csv"])

    # ------------------------------------------------------ patient-level audit
    s = f"{E}/20261103T175003Z_patient_level_accuracy_audit/"
    df = rd(s + "patient_accuracy_distribution.csv")
    d = pick(df, ["experiment", "condition", "n_patients", "n_eligible_targets", "visit_level_accuracy", "mean_patient_accuracy"]); d["mean 95% CI"] = ci(df, "mean_ci_lo", "mean_ci_hi"); d["median_patient_accuracy"] = df["median_patient_accuracy"]
    for c in ["pct_exactly 0%", "pct_below_50pct_cumulative", "pct_at_least_80pct_cumulative", "pct_exactly 100%"]:
        d[c] = df[c].round(2)
    tab(91, "patient_level_accuracy_distribution", d, "Patient-level accuracy audit of experiments 6A and 6B: distribution of per-patient two-class accuracy under each condition.", [s + "patient_accuracy_distribution.csv"])
    df = rd(s + "patient_accuracy_by_subtype_composition.csv"); df = df[df["condition"].isin(["full history", "last 2", "persistence"]) & (df["experiment"] == "6A")]
    tab(92, "patient_level_by_composition", pick(df, ["condition", "subtype_composition", "n_patients", "n_eligible_targets", "mean_patient_accuracy", "median_patient_accuracy", "visit_level_accuracy"]),
        "Patient-level accuracy by the composition of each patient's recorded target labels (exploratory grouping defined after prediction, from recorded labels).", [s + "patient_accuracy_by_subtype_composition.csv"])
    df = rd(s + "patient_subtype_metrics.csv"); df = df[df["condition"].isin(["full history", "last 2", "persistence", "A: full-trained, full history", "D: window-trained, last 2"])].drop_duplicates("condition")
    tab(93, "patient_level_subtype_metrics", pick(df, ["condition", "n_patients", "n_patients_with_PIGD_targets", "mean_patient_PIGD_sensitivity", "median_patient_PIGD_sensitivity", "n_PIGD_patients_zero_correct", "pct_PIGD_patients_zero_correct", "n_patients_ge2_PIGD_targets", "mean_PIGD_sensitivity_ge2_targets", "pct_zero_correct_ge2_PIGD_targets", "mean_patient_TD_specificity"]),
        "Per-patient PIGD sensitivity and TD specificity, overall and among patients with at least two PIGD targets.", [s + "patient_subtype_metrics.csv"])
    df = rd(s + "paired_patient_comparisons.csv"); d = pick(df, ["experiment", "comparison", "n_matched_patients", "n_matched_targets", "mean_paired_difference"]); d["95% CI"] = ci(df, "diff_ci_lo", "diff_ci_hi"); d["excludes zero"] = df["diff_excludes_zero"]; d["n_improved"] = df["n_improved"]; d["n_worsened"] = df["n_worsened"]; d["n_unchanged"] = df["n_unchanged"]
    tab(94, "patient_level_paired", d, "Patient-level paired comparisons: mean paired difference in per-patient accuracy between conditions, with patient-bootstrap intervals, and the number of patients improved, worsened or unchanged.", [s + "paired_patient_comparisons.csv"])

    # ------------------------------------------------------------ ratio bands
    s = f"{E}/20261104T152154Z_cthmm_ratio_banded_observations/"
    df = rd(s + "model_comparison.csv"); d = pick(df, ["model", "n_patients", "n_eligible", "n_correct", "accuracy"]); d["patient-bootstrap 95% CI"] = ci(df, "accuracy_ci_patient_bootstrap_low", "accuracy_ci_patient_bootstrap_high"); d["brier_two_class"] = df["brier_two_class"]; d["log_loss"] = df["log_loss"]
    ct = rd(s + "contrasts.csv")
    tab(95, "ratio_banded_comparison", d, "Ratio-banded CT-HMM (joint subtype-by-band observation alphabet, 44 categories, 137 parameters at band width 0.10) against the subtype-only CT-HMM (14 parameters) and the control arm that runs the subtype-only model through the joint code path, on the 3,029 eligible two-class transitions. Contrasts: " + "; ".join(f"{r['contrast']}: accuracy {r['accuracy_delta']:+.4f} [{r['delta_ci_patient_bootstrap_low']:+.4f}, {r['delta_ci_patient_bootstrap_high']:+.4f}], Brier {r['brier_delta']:+.4f}, log loss {r['log_loss_delta']:+.4f}" for _, r in ct.iterrows()) + ".",
        [s + "model_comparison.csv", s + "contrasts.csv"])
    df = rd(s + "accuracy_by_ratio_range.csv"); df = df[df["model"] != "CT-HMM (subtype labels, joint code path)"]
    tab(96, "ratio_banded_by_ratio_range", df, "Accuracy by current ratio range for the banded and the subtype-only CT-HMM.", [s + "accuracy_by_ratio_range.csv"])
    s = f"{E}/20261104T172138Z_ratio_banded_auc_roc/"
    df = rd(s + "auc_metrics.csv"); d = pick(df, ["model", "arm", "n", "n_positive", "auc_roc"]); d["AUC 95% CI"] = ci(df, "auc_ci_low", "auc_ci_high"); d["average_precision"] = df["average_precision"]; d["AP 95% CI"] = ci(df, "ap_ci_low", "ap_ci_high")
    op = rd(s + "operating_points.csv"); d = d.merge(pick(op, ["model", "n_called_PIGD", "operating_tpr", "operating_fpr", "partial_auc_fpr_le_0.2_mcclish"]), on="model")
    ct = rd(s + "auc_contrasts.csv"); pa = rd(s + "partial_auc_contrast.csv")
    tab(97, "ratio_banded_auc", d, "Ranking metrics of the banded and the subtype-only CT-HMM (PIGD positive). Contrasts (banded minus baseline): AUC " + f"{ct['auc_delta'].iloc[0]:+.4f} [{ct['auc_delta_ci_low'].iloc[0]:+.4f}, {ct['auc_delta_ci_high'].iloc[0]:+.4f}]; average precision {ct['ap_delta'].iloc[0]:+.4f} [{ct['ap_delta_ci_low'].iloc[0]:+.4f}, {ct['ap_delta_ci_high'].iloc[0]:+.4f}]; partial AUC (FPR <= 0.2) {pa['delta'].iloc[0]:+.4f} [{pa['ci_low'].iloc[0]:+.4f}, {pa['ci_high'].iloc[0]:+.4f}].",
        [s + "auc_metrics.csv", s + "operating_points.csv", s + "auc_contrasts.csv", s + "partial_auc_contrast.csv"])
    df = rd(s + "auc_by_stratum.csv"); df = df.dropna(subset=["auc_baseline"])
    tab(98, "ratio_banded_auc_by_stratum", df, "AUC by current-label stratum and by number of prior visits for the banded and the subtype-only CT-HMM (strata with fewer than 20 cases omitted).", [s + "auc_by_stratum.csv"])
    s = f"{E}/20261104T155439Z_ratio_band_emission_diagnostic/"
    bi = rd(s + "band_information.csv"); eb = rd(s + "emission_subtype_banded.csv", index_col=0); e0 = rd(s + "emission_baseline.csv", index_col=0)
    rows = [(f"subtype marginal M[{a}, {b}]", round(e0.loc[a, b], 4), round(eb.loc[a, b], 4)) for a in e0.index for b in e0.columns]
    tab(99, "ratio_band_emission_diagnostic", pd.DataFrame(rows, columns=["quantity", "subtype-only model", "banded model"]), "Emission diagnostic of the banded CT-HMM on the full cohort: the subtype marginal of the emission in both models (rows: latent state, recorded subtype). Per subtype, the number of bands and the largest total-variation distance between the band distributions of two latent states: " + "; ".join(f"{r['subtype']} {int(r['n_bands'])} bands, max TV {r['max_total_variation_between_states']:.4f}" for _, r in bi.iterrows()) + ".", [s + "band_information.csv", s + "emission_subtype_banded.csv", s + "emission_baseline.csv"])
    s = f"{E}/20261105T025024Z_cthmm_ratio_band_width_extended/"
    df = rd(s + "model_metrics.csv"); d = pick(df, ["arm", "accuracy"]); d["accuracy 95% CI"] = ci(df, "accuracy_ci_patient_bootstrap_low", "accuracy_ci_patient_bootstrap_high")
    for c in ["auc_roc"]:
        d[c] = df[c]
    d["AUC 95% CI"] = ci(df, "auc_ci_low", "auc_ci_high"); d["pr_auc"] = df["pr_auc"]; d["partial_auc_fpr_le_0.2"] = df["partial_auc_fpr_le_0.2"]; d["brier_two_class"] = df["brier_two_class"]; d["log_loss"] = df["log_loss"]; d["n_called_PIGD"] = df["n_called_PIGD"]; d["operating_tpr"] = df["operating_tpr"]; d["operating_fpr"] = df["operating_fpr"]
    cx = rd(s + "complexity_by_width.csv"); d = d.merge(cx.assign(arm="width " + cx["main_width"].map(lambda x: f"{x:.2f}"))[["arm", "n_categories", "n_params"]], on="arm", how="left")
    d.loc[d["arm"] == "subtype only", ["n_categories", "n_params"]] = [3, 14]
    d["n_categories"] = d["n_categories"].astype(int); d["n_params"] = d["n_params"].astype(int)
    tab(100, "band_width_sweep_metrics", d, "Band-width sweep: the ratio-banded CT-HMM at main-range band widths 0.05, 0.10, 0.20, 0.30 and 0.50 against the subtype-only model, on the same 3,029 transitions (leave-one-patient-out, 934 folds each).", [s + "model_metrics.csv", s + "complexity_by_width.csv"])
    rc = rd(s + "ranking_contrasts.csv"); rc.insert(0, "reference", "width 0.10"); sc = rd(s + "secondary_contrasts_vs_unbanded.csv"); sc.insert(0, "reference", "subtype only")
    df = pd.concat([rc, sc]); d = pick(df, ["reference", "arm_a", "arm_b", "auc_delta"]); d["AUC 95% CI"] = ci(df, "auc_ci_low", "auc_ci_high"); d["pr_auc_delta"] = df["pr_auc_delta"]; d["PR-AUC 95% CI"] = ci(df, "pr_auc_ci_low", "pr_auc_ci_high"); d["partial_auc_delta"] = df["partial_auc_delta"]; d["partial AUC 95% CI"] = ci(df, "partial_auc_ci_low", "partial_auc_ci_high")
    tab(101, "band_width_ranking_contrasts", d, "Band-width sweep: paired ranking contrasts (arm a minus arm b) against width 0.10 and against the unbanded model, with patient-bootstrap intervals (2,000 resamples).", [s + "ranking_contrasts.csv", s + "secondary_contrasts_vs_unbanded.csv"])
    df = rd(s + "accuracy_contrasts.csv"); d = pick(df, ["arm_a", "arm_b", "n_paired", "accuracy_delta"]); d["95% CI"] = ci(df, "delta_ci_patient_bootstrap_low", "delta_ci_patient_bootstrap_high"); d["a right, b wrong"] = df["a_right_b_wrong"]; d["b right, a wrong"] = df["b_right_a_wrong"]
    pc_ = rd(s + "probabilistic_contrasts.csv")
    tab(102, "band_width_accuracy_contrasts", d, "Band-width sweep: paired accuracy contrasts (arm a minus arm b). Probabilistic contrasts (a minus b): " + "; ".join(f"{r['arm_a']} vs {r['arm_b']}: Brier {r['brier_delta']:+.4f}, log loss {r['log_loss_delta']:+.4f}, calibration gap {r['calibration_gap_delta']:+.4f}" for _, r in pc_.iterrows()) + ".", [s + "accuracy_contrasts.csv", s + "probabilistic_contrasts.csv"])
    # ------------------------------------------------ direct logistic on measurements
    s = f"{E}/20261020T145500Z_clinical_features_direct/"
    df = rd(s + "model_comparison.csv"); d = pick(df, ["model", "n_patients", "n_scored", "n_correct", "accuracy"]); d["patient-bootstrap 95% CI"] = ci(df, "accuracy_ci_patient_bootstrap_low", "accuracy_ci_patient_bootstrap_high"); d["brier_two_class"] = df["brier_two_class"]; d["log_loss"] = df["log_loss"]
    ct = rd(s + "contrasts.csv")
    tab(104, "clinical_measurements_direct_logistic", d, "Direct L2-penalised logistic models on the ten standardised clinical measurements, and on the tremor and PIGD scores only, against the subtype-history CT-HMM, persistence and the six-state clinical-state CT-HMM on the identical 2,091 cases of the 654-patient evaluation set. Paired contrasts: " + "; ".join(f"{r['contrast']}: {r['accuracy_delta']:+.4f} [{r['delta_ci_patient_bootstrap_low']:+.4f}, {r['delta_ci_patient_bootstrap_high']:+.4f}], Brier {r['brier_delta']:+.4f}" for _, r in ct.iterrows()) + ".", [s + "model_comparison.csv", s + "contrasts.csv"])
    tab(103, "band_width_complexity", cx, "Band-width sweep: alphabet size, parameter count and category occupancy by band width.", [s + "complexity_by_width.csv"])


def main() -> int:
    warnings.simplefilter("ignore")
    build()
    pd.DataFrame(INDEX).to_csv(OUT / "TABLE_INDEX.csv", index=False)
    print(f"wrote {len(INDEX)} tables to {OUT.relative_to(ROOT)}")
    if PROBLEMS:
        print("PROBLEMS:", *PROBLEMS, sep="\n  ", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
