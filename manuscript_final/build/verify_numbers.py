#!/usr/bin/env python3
"""Check every number quoted in the manuscript text against its source file.

Each check names the section, the string as printed in full_research_paper.md,
the source file and a locator (a CSV row and column, a JSON path, a regular
expression over a text file, or a short pandas expression), an optional scale
(100 for percentages) and whether the absolute value is quoted. The check
passes when (a) the printed string occurs in the assembled manuscript and (b)
the source value, scaled, agrees with the printed value to half a unit in the
last printed decimal. The result is written to tables/numerical_traceability.csv
and summarised on stdout. The script also confirms that the manuscript contains
none of the strings that would indicate excluded material.

Read-only outside manuscript_final/; nothing is fitted or recomputed beyond
means over saved per-split values where the manuscript says so.
"""
from __future__ import annotations
import csv, glob, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MF = HERE.parent
ROOT = MF.parent
TEXT = (MF / "full_research_paper.md").read_text()
TEXT_ASCII = TEXT.replace("−", "-").replace("–", "-")

E, O = "experiments/", "outputs/"
FLOW = E + "20261103T192414Z_e5_participant_flow/"
RAWJ = E + "20261015T221426Z_partC_raw_scores/results.json"
INDJ = E + "20261015T221426Z_partC_indeterminate/results.json"
INDR = E + "20261015T221426Z_partC_indeterminate/REPORT.md"
E2 = E + "20261103T193205Z_e2_transition_patient_bootstrap/"
O4 = E + "20261104T002308Z_o4_threshold_sensitivity/"
LAG = E + "20261103T191408Z_e1_lag1_residual_resolution/lag1_statistic.csv"
LAGR = "research_audit/generated_reports/lag1_residual_resolution_report.md"
D1 = E + "20261016T070746Z_d1_full/results/"
S1 = E + "20261016T133134Z_d1_stage1_checks/"
S1DOC = "docs/cluster_phase/14_D1_STAGE1_CHECKS_20261016.md"
S4CV = E + "20261014T100057Z_s4_hmm_cv/analysis/"
S4ST = E + "20261014T094649Z_s4_hmm_stability/analysis/"
ST = E + "20261015T190228Z_student_t_four_model_full/results.json"
Q2 = E + "20261015T203325Z_q2_reassessment/results/results.json"
MEMO = "docs/gpr_audit/NEXT_RESEARCH_STEPS.md"
S3 = E + "20261014T094751Z_s3_subscore_heldout/analysis/"
GPRDOC = "docs/GPR_onwards_explanatory_report.md"
S5 = E + "20261014T100107Z_s5_temporal_audit/"
E8 = E + "20261103T202632Z_e8_harmonised_family_comparison/"
NV = E + "20261020T104756Z_next_visit_td_pigd_prediction/"
CS6 = E + "20261020T142446Z_clinical_state_lopo_k6/"
CS8 = E + "20261020T145228Z_clinical_state_lopo_k8/"
CSD = E + "20261020T145500Z_clinical_features_direct/"
A2 = E + "20261103T220147Z_a2p_outcome_table_fullprec/"
A3 = E + "20261104T085320Z_a3_profile_cv_with_baseline/"
MSC = E + "20261023T194317Z_model_selection_criteria/"
ROC = E + "20261023T202640Z_roc_auc_comparison/"
IB = E + "20261023T203431Z_indeterminate_boundary/"
IRS = E + "20261023T211836Z_indeterminate_ratio_scrutiny/"
SW = E + "20261023T214819Z_stage_wise_evaluation/"
HW = E + "20261024T042417Z_history_window_experiment/"
PHW = E + "20261024T042811Z_profile_history_window/"
RT = E + "20261024T052343Z_retrained_two_visit_cthmm/"
X6A = E + "20261103T052237Z_profile_prediction_history/"
X6B = E + "20261103T072507Z_profile_two_visit_retraining/"
PL = E + "20261103T175003Z_patient_level_accuracy_audit/"
RB = E + "20261104T152154Z_cthmm_ratio_banded_observations/"
RBA = E + "20261104T172138Z_ratio_banded_auc_roc/"
RBE = E + "20261104T155439Z_ratio_band_emission_diagnostic/"
BW = E + "20261105T025024Z_cthmm_ratio_band_width_extended/"
RPDOC = "docs/latent_clinical_states/02_RICH_PROFILE_DESIGN_20261020.md"
LCSDOC = "docs/latent_clinical_states/01_LATENT_CLINICAL_STATES_20261020.md"
S1AUDIT = "docs/manuscript/01_STAGE1_RESEARCH_AUDIT_20261104.md"

CTH = "CT-HMM (latent, interval-aware)"
DIR = "Direct logistic (history + interval)"
DIR0 = "Direct logistic (history only)"
PERS = "Persistence (last non-Indeterminate)"
MKV2 = "Markov (observed, discrete)"
CTMC2 = "CTMC (observed, interval-aware)"
NULL2 = "Marginal (two-class prior)"
RBM = "CT-HMM (subtype x ratio band)"
PROF = "Profile: level + rate + medication"
LO = "Profile: level only (random walk)"
NMED = "Profile: level + rate, no medication"
LOCF = "Last observation carried forward"
POP = "Population mean trajectory"
CSM = "Clinical-state CT-HMM (10 measurements)"
SHM = "Subtype-history CT-HMM (labels only)"
DL10 = "Direct logistic on the 10 clinical measurements"
DL2 = "Direct logistic on tremor and PIGD scores only"
RCP = "Rich clinical profile"

CHECKS: list[tuple] = []


def C(section, printed, source, locator, scale=1.0, absval=False, value=None):
    """Register a check: `printed` must occur in the text; `value` (default printed) is compared."""
    CHECKS.append((section, printed, value if value is not None else printed, source, locator, scale, absval))


def csvw(file, col, **where):
    return {"kind": "csv", "file": file, "col": col, "where": where}


def csvi(file, col, row):
    return {"kind": "csv", "file": file, "col": col, "row": row}


def jsn(file, path):
    return {"kind": "json", "file": file, "path": path}


def txt(file, rx):
    return {"kind": "txt", "file": file, "rx": rx}


def ex(py, file=None):
    return {"kind": "expr", "py": py, "file": file}


# ---------------------------------------------------------------- abstract / cohort
FL = FLOW + "flow_ledger.csv"
C("Abstract", "19,450", FL, csvw(FL, "visits", stage="curated data cut, as published"))
C("Abstract", "4,788", FL, csvw(FL, "patients", stage="curated data cut, as published"))
C("Abstract", "5,774", FL, csvw(FL, "visits", stage="~^exclude events"))
C("Abstract", "1,080", FL, csvw(FL, "patients", stage="~^exclude events"))
C("Abstract", "4,441", RAWJ, jsn(RAWJ, "primary/labelled_visits"))
C("Abstract", "76.9%", O + "panel_summary.txt", txt(O + "panel_summary.txt", r"visits with a label\s+4,441\s+\(([\d.]+)%\)"))
C("Abstract", "934", RAWJ, jsn(RAWJ, "primary/patients_ge2_labelled"))
C("Abstract", "3,361", O + "loo/loo_overall_metrics.json", jsn(O + "loo/loo_overall_metrics.json", "n_predictions"))
OQ = E2 + "other_quantities.csv"
C("Abstract", "35.2%", OQ, csvw(OQ, "estimate", quantity="reversion_pooled"), 100)
C("Abstract", "31.8 to 38.8", OQ, csvw(OQ, "bootstrap_low", quantity="reversion_pooled"), 100, value="31.8")
C("Abstract", "31.8 to 38.8", OQ, csvw(OQ, "bootstrap_high", quantity="reversion_pooled"), 100, value="38.8")
C("Abstract", "51.7%", OQ, csvw(OQ, "estimate", quantity="sustained_removes_share"), 100)
MK = O + "model/markov_property_test.csv"
C("Abstract", "156.6", MK, csvw(MK, "chi2", current_state="TD"))
C("Abstract", "6.7%", O4 + "summary.csv", csvi(O4 + "summary.csv", "share_relabelled", 0), 100)
C("Abstract", "−0.22", LAG, csvw(LAG, "estimate", quantity="observed lag-1 correlation"))
C("Abstract", "−0.33", LAG, csvw(LAG, "estimate", quantity="~^iid null"))
HM = E8 + "harmonised_metrics.csv"
C("Abstract", "0.745 and 0.758", HM, csvw(HM, "accuracy", model="CTMC, interval-aware"), value="0.745")
C("Abstract", "0.745 and 0.758", HM, csvw(HM, "accuracy", model="Subscore GP, independent"), value="0.758")
C("Abstract", "0.723", HM, csvw(HM, "accuracy", model="Persistence (recorded label)"))
C("Abstract", "0.629", HM, csvw(HM, "accuracy", model="Marginal null"))
LJ = O + "loo/loo_overall_metrics.json"
C("Abstract", "0.754 (95% CI 0.735 to 0.773)", LJ, jsn(LJ, "accuracy"), value="0.754")
C("Abstract", "0.754 (95% CI 0.735 to 0.773)", LJ, jsn(LJ, "accuracy_ci_patient_bootstrap_low"), value="0.735")
C("Abstract", "0.754 (95% CI 0.735 to 0.773)", LJ, jsn(LJ, "accuracy_ci_patient_bootstrap_high"), value="0.773")
MD = E8 + "margin_decomposition.csv"
C("Abstract", "94.3%", MD, csvw(MD, "share_of_total_gain", model="CT-HMM, 3 latent states", current_label="Indeterminate"), 100)
C("Abstract", "9.9%", NV + "outcome_counts.csv", csvw(NV + "outcome_counts.csv", "share", actual_next_label="Indeterminate"), 100)
NM = NV + "model_comparison.csv"
C("Abstract", "3,029", NM, csvw(NM, "n_eligible", model=CTH))
C("Abstract", "0.837 (0.820 to 0.854)", NM, csvw(NM, "accuracy", model=CTH), value="0.837")
C("Abstract", "0.837 (0.820 to 0.854)", NM, csvw(NM, "accuracy_ci_patient_bootstrap_low", model=CTH), value="0.820")
C("Abstract", "0.837 (0.820 to 0.854)", NM, csvw(NM, "accuracy_ci_patient_bootstrap_high", model=CTH), value="0.854")
C("Abstract", "persistence 0.834", NM, csvw(NM, "accuracy", model=PERS), value="0.834")
C("Abstract", "direct logistic model 0.836", NM, csvw(NM, "accuracy", model=DIR), value="0.836")
AD = A3 + "derived_label_vs_subtype_history.csv"
C("Abstract", "clinical profile 0.840", AD, csvw(AD, "accuracy", model=PROF), value="0.840")
C("Abstract", "lost 11.3 points", CS6 + "contrasts.csv", csvw(CS6 + "contrasts.csv", "accuracy_delta", model_a=CSM, model_b=SHM), 100, True, value="11.3")
RAM = ROC + "auc_metrics.csv"
C("Abstract", "AUC from 0.858 to 0.890", RAM, csvw(RAM, "auc", model="CT-HMM"), value="0.858")
C("Abstract", "AUC from 0.858 to 0.890", RAM, csvw(RAM, "auc", model="~[Pp]rofile"), value="0.890")
RAMx = RBA + "auc_metrics.csv"
C("Abstract", "from 0.859 to 0.876", RAMx, csvw(RAMx, "auc_roc", arm="baseline (subtype only)"), value="0.859")
C("Abstract", "from 0.859 to 0.876", RAMx, csvw(RAMx, "auc_roc", arm="ratio-banded (subtype x band)"), value="0.876")

# ---------------------------------------------------------------- sections 4 and 5
UD = FLOW + "unlabelled_diagnosis.csv"
C("4.3", "5,789", FL, csvw(FL, "visits", stage="~^subgroup"))
C("4.3", "1,195", UD, csvw(UD, "visits", reason="~ON-medication"))
C("4.3", "89.6%", UD, csvw(UD, "share", reason="~ON-medication"), 100)
C("4.3", "65 have no Part III", UD, csvw(UD, "visits", reason="~^no Part III"), value="65")
C("4.3", "57 have a partially scored", UD, csvw(UD, "visits", reason="~partially scored"), value="57")
C("4.3", "4,295", S4ST + "population_starts.csv", csvi(S4ST + "population_starts.csv", "n_observations", 0))
C("4.3", "332", NM, csvw(NM, "n_indeterminate_next", model=CTH))
C("4.3", "899", NM, csvw(NM, "n_patients", model=CTH))
AG = O4 + "agreement_vs_ppmi.csv"
C("4.2", "4,431 of 4,431", AG, csvw(AG, "two_class_n", rule="~^baseline"), value="4431")
C("4.2", "1,595 of 1,597", O + "validation.txt", txt(O + "validation.txt", r"n_agree\s+(1595)"), value="1595")
C("4.2", "(0.9987)", AG, csvw(AG, "three_class", rule="~^baseline"), value="0.9987")
C("4.2", "390 participants", O + "validation.txt", txt(O + "validation.txt", r"n_patients\s+(390)"), value="390")
PY = O + "transitions/prevalence_by_year.csv"
C("4.4", "2,891 (65.1%)", RAWJ, jsn(RAWJ, "primary/label_counts/TD"), value="2891")
C("4.4", "2,891 (65.1%)", O + "panel_summary.txt", txt(O + "panel_summary.txt", r"TD\s+2,891\s+\(\s*([\d.]+)%\)"), value="65.1")
C("4.4", "441 (9.9%)", RAWJ, jsn(RAWJ, "primary/label_counts/Indeterminate"), value="441")
C("4.4", "1,109 (25.0%)", RAWJ, jsn(RAWJ, "primary/label_counts/PIGD"), value="1109")
C("4.4", "772 (71.7%)", PY, csvw(PY, "n_TD", YEAR="0"), value="772")
C("4.4", "109 (10.1%)", PY, csvw(PY, "n_Indeterminate", YEAR="0"), value="109")
C("4.4", "195 (18.1%)", PY, csvw(PY, "n_PIGD", YEAR="0"), value="195")
C("4.4", "1,076 labelled baseline", PY, csvw(PY, "n_total", YEAR="0"), value="1076")
C("4.4", "725 labelled visits (16.3%)", O + "panel_summary.txt", txt(O + "panel_summary.txt", r"ratio undefined\s+([\d,]+)"), value="725")
C("4.4", "725 labelled visits (16.3%)", RAWJ, jsn(RAWJ, "primary/pigd_sum0_share"), 100, value="16.3")
C("4.4", "665 of those are TD", O + "hbayes/report.txt", txt(O + "hbayes/report.txt", r"'TD':\s*(\d+)"), value="665")
C("4.4", "60 are Indeterminate", O + "hbayes/report.txt", txt(O + "hbayes/report.txt", r"'Indeterminate':\s*(\d+)"), value="60")
C("4.4", "1.00 years (10th to 90th percentile 0.91 to 2.00)", RAWJ, jsn(RAWJ, "primary/interval_quantiles/0.5"), value="1.00")
C("4.4", "0.91 to 2.00", RAWJ, jsn(RAWJ, "primary/interval_quantiles/0.1"), value="0.91")
C("4.4", "0.91 to 2.00", RAWJ, jsn(RAWJ, "primary/interval_quantiles/0.9"), value="2.00")
C("4.4", "3.1 years (10th to 90th percentile 1.2 to 11.0)", RAWJ, jsn(RAWJ, "primary/followup_quantiles/0.5"), value="3.1")
C("4.4", "1.2 to 11.0", RAWJ, jsn(RAWJ, "primary/followup_quantiles/0.1"), value="1.2")
C("4.4", "1.2 to 11.0", RAWJ, jsn(RAWJ, "primary/followup_quantiles/0.9"), value="11.0")
RA = O + "exploration/ratio_availability_by_year.csv"
C("4.4", "0.74 at baseline", RA, csvw(RA, "ratio_share", YEAR="0"), value="0.74")
SD_ = FLOW + "sentinel_denominators.csv"
C("5.2", "4,302 raw records", SD_, txt(SD_, r"raw Part III file,\d+,(\d+)"), value="4302")
C("5.2", "3,206 after OFF-state selection", SD_, txt(SD_, r"after OFF/untreated selection,\d+,(\d+)"), value="3206")
C("5.2", "3,205 after de-duplication", FLOW + "sentinel_counts.csv", csvi(FLOW + "sentinel_counts.csv", "n_sentinel_101", 0), value="3205")
C("5.3", "97.6% of labelled visits", LCSDOC, txt(LCSDOC, r"\(([\d.]+)% carry all ten\)"), value="97.6")
C("5.4", "758 tests passing", S1AUDIT, txt(S1AUDIT, r"\b(758)\b"), value="758")

# ---------------------------------------------------------------- 9.1 label dynamics
TP = E2 + "transition_probabilities.csv"
C("9.1", "0.808 (TD to TD", TP, csvw(TP, "probability", from_state="TD", to_state="TD"), value="0.808")
C("9.1", "0.788 to 0.826", TP, csvw(TP, "bootstrap_low", from_state="TD", to_state="TD"), value="0.788")
C("9.1", "0.788 to 0.826", TP, csvw(TP, "bootstrap_high", from_state="TD", to_state="TD"), value="0.826")
C("9.1", "0.702 (PIGD to PIGD", TP, csvw(TP, "probability", from_state="PIGD", to_state="PIGD"), value="0.702")
C("9.1", "0.661 to 0.738", TP, csvw(TP, "bootstrap_low", from_state="PIGD", to_state="PIGD"), value="0.661")
C("9.1", "0.661 to 0.738", TP, csvw(TP, "bootstrap_high", from_state="PIGD", to_state="PIGD"), value="0.738")
C("9.1", "only 0.192", TP, csvw(TP, "probability", from_state="Indeterminate", to_state="Indeterminate"), value="0.192")
C("9.1", "0.148 to 0.234", TP, csvw(TP, "bootstrap_low", from_state="Indeterminate", to_state="Indeterminate"), value="0.148")
C("9.1", "0.148 to 0.234", TP, csvw(TP, "bootstrap_high", from_state="Indeterminate", to_state="Indeterminate"), value="0.234")
C("9.1", "TD at 0.440", TP, csvw(TP, "probability", from_state="Indeterminate", to_state="TD"), value="0.440")
C("9.1", "PIGD at 0.368", TP, csvw(TP, "probability", from_state="Indeterminate", to_state="PIGD"), value="0.368")
C("9.1", "1.38 (TD to TD)", TP, csvw(TP, "design_effect", from_state="TD", to_state="TD"), value="1.38")
C("9.1", "1.46 (PIGD to PIGD)", TP, csvw(TP, "design_effect", from_state="PIGD", to_state="PIGD"), value="1.46")
C("9.1", "673 observed label changes", OQ, csvw(OQ, "estimate", quantity="n_changes"), value="673")
C("9.1", "0.352, 95% CI 0.318 to 0.388", OQ, csvw(OQ, "estimate", quantity="reversion_pooled"), value="0.352")
C("9.1", "0.352, 95% CI 0.318 to 0.388", OQ, csvw(OQ, "bootstrap_low", quantity="reversion_pooled"), value="0.318")
C("9.1", "0.352, 95% CI 0.318 to 0.388", OQ, csvw(OQ, "bootstrap_high", quantity="reversion_pooled"), value="0.388")
C("9.1", "TD to Indeterminate 0.560", OQ, csvw(OQ, "estimate", quantity="revert[TD->Indeterminate]"), value="0.560")
C("9.1", "PIGD to Indeterminate 0.500", OQ, csvw(OQ, "estimate", quantity="revert[PIGD->Indeterminate]"), value="0.500")
C("9.1", "Indeterminate to PIGD 0.183", OQ, csvw(OQ, "estimate", quantity="revert[Indeterminate->PIGD]"), value="0.183")
C("9.1", "Indeterminate to TD 0.176", OQ, csvw(OQ, "estimate", quantity="revert[Indeterminate->TD]"), value="0.176")
C("9.1", "0.320 (TD to PIGD)", OQ, csvw(OQ, "estimate", quantity="revert[TD->PIGD]"), value="0.320")
C("9.1", "0.393 (PIGD to TD)", OQ, csvw(OQ, "estimate", quantity="revert[PIGD->TD]"), value="0.393")
TR = O + "transitions/report.txt"
C("9.1", "476 never change label", TR, txt(TR, r"no recorded change of state:\s+(\d+) of 934"), value="476")
C("9.1", "86 distinct label pathways", TR, txt(TR, r"\((\d+) distinct pathways"), value="86")
C("9.1", "0.125 by year two", OQ, csvw(OQ, "estimate", quantity="km_naive_yr2"), value="0.125")
C("9.1", "0.353 (0.308 to 0.399)", OQ, csvw(OQ, "estimate", quantity="km_naive_yr5"), value="0.353")
C("9.1", "0.353 (0.308 to 0.399)", OQ, csvw(OQ, "bootstrap_low", quantity="km_naive_yr5"), value="0.308")
C("9.1", "0.353 (0.308 to 0.399)", OQ, csvw(OQ, "bootstrap_high", quantity="km_naive_yr5"), value="0.399")
C("9.1", "it is 0.065", OQ, csvw(OQ, "estimate", quantity="km_sustained_yr2"), value="0.065")
C("9.1", "0.153 (0.120 to 0.190)", OQ, csvw(OQ, "estimate", quantity="km_sustained_yr5"), value="0.153")
C("9.1", "0.153 (0.120 to 0.190)", OQ, csvw(OQ, "bootstrap_low", quantity="km_sustained_yr5"), value="0.120")
C("9.1", "0.153 (0.120 to 0.190)", OQ, csvw(OQ, "bootstrap_high", quantity="km_sustained_yr5"), value="0.190")
C("9.1", "0.517 (0.457 to 0.577)", OQ, csvw(OQ, "estimate", quantity="sustained_removes_share"), value="0.517")
C("9.1", "0.517 (0.457 to 0.577)", OQ, csvw(OQ, "bootstrap_low", quantity="sustained_removes_share"), value="0.457")
C("9.1", "0.517 (0.457 to 0.577)", OQ, csvw(OQ, "bootstrap_high", quantity="sustained_removes_share"), value="0.577")
C("9.1", "263 first PIGD labels become 127", OQ, csvw(OQ, "estimate", quantity="naive_converted"), value="263")
C("9.1", "263 first PIGD labels become 127", OQ, csvw(OQ, "estimate", quantity="sustained_converted"), value="127")
C("9.1", "885 patients at risk", OQ, csvw(OQ, "estimate", quantity="naive_at_risk"), value="885")
C("9.1", "0.100 (0.087 to 0.116)", OQ, csvw(OQ, "estimate", quantity="p[TD->PIGD | <=1.5y]"), value="0.100")
C("9.1", "0.100 (0.087 to 0.116)", OQ, csvw(OQ, "bootstrap_low", quantity="p[TD->PIGD | <=1.5y]"), value="0.087")
C("9.1", "0.100 (0.087 to 0.116)", OQ, csvw(OQ, "bootstrap_high", quantity="p[TD->PIGD | <=1.5y]"), value="0.116")
C("9.1", "0.289 (0.167 to 0.432)", OQ, csvw(OQ, "estimate", quantity="p[TD->PIGD | >4y]"), value="0.289")
C("9.1", "0.289 (0.167 to 0.432)", OQ, csvw(OQ, "bootstrap_low", quantity="p[TD->PIGD | >4y]"), value="0.167")
C("9.1", "0.289 (0.167 to 0.432)", OQ, csvw(OQ, "bootstrap_high", quantity="p[TD->PIGD | >4y]"), value="0.432")
C("9.1", "from 0.820 to 0.578", OQ, csvw(OQ, "estimate", quantity="p[TD->TD | <=1.5y]"), value="0.820")
C("9.1", "from 0.820 to 0.578", OQ, csvw(OQ, "estimate", quantity="p[TD->TD | >4y]"), value="0.578")
C("9.1", "1,583 triples", MK, csvw(MK, "n_triples", current_state="TD"), value="1583")
MR = O + "model/report.txt"
C("9.1", "probability 0.086", MR, txt(MR, r"TD\s+0\.849\s+0\.065\s+(0\.086)"), value="0.086")
C("9.1", "probability 0.393", MR, txt(MR, r"PIGD\s+0\.482\s+0\.125\s+(0\.393)"), value="0.393")
C("9.1", "chi-square 61.2", MK, csvw(MK, "chi2", current_state="PIGD"), value="61.2")
C("9.1", "609 triples", MK, csvw(MK, "n_triples", current_state="PIGD"), value="609")
C("9.1", "chi-square 15.3", MK, csvw(MK, "chi2", current_state="Indeterminate"), value="15.3")
C("9.1", "235 triples", MK, csvw(MK, "n_triples", current_state="Indeterminate"), value="235")

# ---------------------------------------------------------------- 9.2 construction audits
C("9.2", "75.2% of pairs", RAWJ, jsn(RAWJ, "off_on/full/agreement"), 100, value="75.2")
C("9.2", "72.1% with the Part III items only", RAWJ, jsn(RAWJ, "off_on/part3_only/agreement"), 100, value="72.1")
C("9.2", "83.2% of visits", RAWJ, jsn(RAWJ, "sensitivity_constructions/part3_only/agreement"), 100, value="83.2")
C("9.2", "2,343 pairs", RAWJ, jsn(RAWJ, "off_on/full/pairs_with_both_labels"), value="2343")
C("9.2", "2,353 pairs", RAWJ, jsn(RAWJ, "off_on/part3_only/pairs_with_both_labels"), value="2353")
C("9.2", "714 patients", RAWJ, jsn(RAWJ, "off_on/full/patients"), value="714")
C("9.2", "16.3% of labelled visits", RAWJ, jsn(RAWJ, "primary/pigd_sum0_share"), 100, value="16.3")
C("9.2", "tremor sum at 7.2%", RAWJ, jsn(RAWJ, "primary/tremor_sum0_share"), 100, value="7.2")
C("9.2", "0.543 (0.471 to 0.615)", INDJ, jsn(INDJ, "primary/point/R_E"), value="0.543")
C("9.2", "0.543 (0.471 to 0.615)", INDJ, jsn(INDJ, "primary/ci95/R_E/0"), value="0.471")
C("9.2", "0.543 (0.471 to 0.615)", INDJ, jsn(INDJ, "primary/ci95/R_E/1"), value="0.615")
C("9.2", "0.349 (0.298 to 0.397)", INDJ, jsn(INDJ, "primary/point/R_D"), value="0.349")
C("9.2", "0.349 (0.298 to 0.397)", INDJ, jsn(INDJ, "primary/ci95/R_D/0"), value="0.298")
C("9.2", "0.349 (0.298 to 0.397)", INDJ, jsn(INDJ, "primary/ci95/R_D/1"), value="0.397")
C("9.2", "+0.194 (+0.108 to +0.281)", INDJ, jsn(INDJ, "primary/point/delta_R"), value="0.194")
C("9.2", "+0.194 (+0.108 to +0.281)", INDJ, jsn(INDJ, "primary/ci95/delta_R/0"), value="0.108")
C("9.2", "+0.194 (+0.108 to +0.281)", INDJ, jsn(INDJ, "primary/ci95/delta_R/1"), value="0.281")
C("9.2", "17.6% of entries", INDJ, jsn(INDJ, "primary/point/entry_sustained"), 100, value="17.6")
C("9.2", "3 of 135", INDJ, jsn(INDJ, "primary/two_step/sustained_two"), value="3")
C("9.2", "3 of 135", INDJ, jsn(INDJ, "primary/two_step/entries_with_two_later_visits"), value="135")
C("9.2", "1.18 (0.99 to 1.39)", INDJ, jsn(INDJ, "primary/point/OE_Indeterminate"), value="1.18")
C("9.2", "1.18 (0.99 to 1.39)", INDJ, jsn(INDJ, "primary/ci95/OE_Indeterminate/0"), value="0.99")
C("9.2", "1.18 (0.99 to 1.39)", INDJ, jsn(INDJ, "primary/ci95/OE_Indeterminate/1"), value="1.39")
C("9.2", "1.08 years", INDR, txt(INDR, r"\| Indeterminate \| 244 \| 69 \| [\d.]+ \| [\d.]+ \| ([\d.]+) y \|"), value="1.08")
C("9.2", "2.08 years for PIGD", INDR, txt(INDR, r"\| PIGD \| 344 \| 114 \| [\d.]+ \| [\d.]+ \| ([\d.]+) y \|"), value="2.08")
C("9.2", "5.41 years for TD", INDR, txt(INDR, r"\| TD \| 775 \| 126 \| [\d.]+ \| [\d.]+ \| ([\d.]+) y \|"), value="5.41")
SM4 = O4 + "summary.csv"; PM4 = O4 + "per_class_movement.csv"
C("9.2", "relabels 298", SM4, csvi(SM4, "visits_relabelled", 0), value="298")
C("9.2", "(6.71%)", SM4, csvi(SM4, "share_relabelled", 0), 100, value="6.71")
C("9.2", "100 to TD, 198 to PIGD", PM4, csvw(PM4, "delta", **{"class": "TD"}), value="100")
C("9.2", "100 to TD, 198 to PIGD", PM4, csvw(PM4, "delta", **{"class": "PIGD"}), value="198")
C("9.2", "from 441 to 143", SM4, csvi(SM4, "indeterminate_after", 0), value="143")
C("9.2", "−67.6%", PM4, csvw(PM4, "pct_change", **{"class": "Indeterminate"}), 100, value="-67.6")
C("9.2", "235 of 1,080 participants (21.76%)", SM4, csvi(SM4, "patients_with_any_relabelled", 0), value="235")
C("9.2", "235 of 1,080 participants (21.76%)", SM4, csvi(SM4, "share_patients_affected", 0), 100, value="21.76")
C("9.2", "1.000 to 0.977", AG, csvw(AG, "two_class", rule="~^narrow"), value="0.977")
C("9.2", "0.999 to 0.929", AG, csvw(AG, "three_class", rule="~^narrow"), value="0.929")
AR4 = O4 + "attainable_ratios.csv"
C("9.2", "0.9091 (198 visits)", AR4, ex("df.loc[(df.iloc[:,0]-0.9091).abs()<1e-3].filter(regex='^n').iloc[0,0]", AR4), value="198")
C("9.2", "1.1364 (100 visits)", AR4, ex("df.loc[(df.iloc[:,0]-1.1364).abs()<1e-3].filter(regex='^n').iloc[0,0]", AR4), value="100")
C("9.2", "(−0.219, patient-bootstrap 95% CI −0.273 to −0.162)", LAG, csvw(LAG, "estimate", quantity="observed lag-1 correlation"), value="-0.219")
C("9.2", "−0.273 to −0.162", LAG, csvw(LAG, "ci_low", quantity="observed lag-1 correlation"), value="-0.273")
C("9.2", "−0.273 to −0.162", LAG, csvw(LAG, "ci_high", quantity="observed lag-1 correlation"), value="-0.162")
C("9.2", "(−0.333, Monte Carlo 95% range −0.369 to −0.295)", LAG, csvw(LAG, "estimate", quantity="~^iid null"), value="-0.333")
C("9.2", "−0.369 to −0.295", LAG, csvw(LAG, "ci_low", quantity="~^iid null"), value="-0.369")
C("9.2", "−0.369 to −0.295", LAG, csvw(LAG, "ci_high", quantity="~^iid null"), value="-0.295")
C("9.2", "+0.113 (+0.046 to +0.179", LAG, csvw(LAG, "estimate", quantity="observed minus null"), value="0.113")
C("9.2", "+0.113 (+0.046 to +0.179", LAG, csvw(LAG, "ci_low", quantity="observed minus null"), value="0.046")
C("9.2", "+0.113 (+0.046 to +0.179", LAG, csvw(LAG, "ci_high", quantity="observed minus null"), value="0.179")
C("9.2", "397 patients", LAGR, txt(LAGR, r"\*\*Eligible patients\*\* \| (\d+)"), value="397")
C("9.2", "2,050 pairs", LAGR, txt(LAGR, r"\*\*Consecutive residual pairs\*\* \| ([\d,]+)"), value="2050")

# ---------------------------------------------------------------- 9.3 D-1
OB = D1 + "05_observed.json"; PW = D1 + "02_power.json"
C("9.3", "0.103 against a replicate mean of 0.065", OB, jsn(OB, "observed/S_obs/S1_ind_share"), value="0.103")
C("9.3", "0.103 against a replicate mean of 0.065", OB, jsn(OB, "observed/S_rep_mean/S1_ind_share"), value="0.065")
C("9.3", "0.311 against 0.436", OB, jsn(OB, "observed/S_obs/S3_R_D"), value="0.311")
C("9.3", "0.311 against 0.436", OB, jsn(OB, "observed/S_rep_mean/S3_R_D"), value="0.436")
C("9.3", "0.268 against 0.120", OB, jsn(OB, "observed/S_obs/S4_delta_R"), value="0.268")
C("9.3", "0.268 against 0.120", OB, jsn(OB, "observed/S_rep_mean/S4_delta_R"), value="0.120")
C("9.3", "(0.580 against 0.556)", OB, jsn(OB, "observed/S_obs/S2_R_E"), value="0.580")
C("9.3", "(0.580 against 0.556)", OB, jsn(OB, "observed/S_rep_mean/S2_R_E"), value="0.556")
C("9.3", "(1.033 against 1.077)", OB, jsn(OB, "observed/S_obs/S5_OE_ind"), value="1.033")
C("9.3", "(1.033 against 1.077)", OB, jsn(OB, "observed/S_rep_mean/S5_OE_ind"), value="1.077")
C("9.3", "= 3.091", OB, jsn(OB, "T_obs"), value="3.091")
C("9.3", "= 2.577", OB, jsn(OB, "c"), value="2.577")
C("9.3", "100 of 100 cohorts", PW, jsn(PW, "scenarios/PC4/detected"), value="100")
C("9.3", "0.964", PW, jsn(PW, "primary_P_lo"), value="0.964")
C("9.3", "6, 7 and 5 of 100", PW, jsn(PW, "scenarios/PC1/detected"), value="6")
C("9.3", "6, 7 and 5 of 100", PW, jsn(PW, "scenarios/PC2/detected"), value="7")
C("9.3", "6, 7 and 5 of 100", PW, jsn(PW, "scenarios/PC3/detected"), value="5")
C("9.3", "PC4v in 3 of 100", D1 + "03_pc4v.json", jsn(D1 + "03_pc4v.json", "detected"), value="3")
IBS = S1 + "check1_ind_by_pigd_stratum.csv"; DL = S1 + "check1_delatticed.csv"
C("9.3", "(183 visits)", IBS, csvw(IBS, "band_excess", stratum="all"), value="183")
C("9.3", "sums 2 (50% of the excess)", IBS, csvw(IBS, "band_excess_share_of_total", stratum="2"), 100, value="50")
C("9.3", "3 (24%)", IBS, csvw(IBS, "band_excess_share_of_total", stratum="3"), 100, value="24")
C("9.3", "4 to 5 (22%)", IBS, csvw(IBS, "band_excess_share_of_total", stratum="4-5"), 100, value="22")
C("9.3", "(70 against 71.8)", IBS, csvw(IBS, "count", stratum="1"), value="70")
C("9.3", "(70 against 71.8)", IBS, csvw(IBS, "band_expected", stratum="1"), value="71.8")
C("9.3", "1.82 times its width", DL, csvw(DL, "inflation", pigd_sum="1"), value="1.82")
C("9.3", "about 0.068", DL, ex("df['delatticed'].sum()/df['n'].sum()", DL), value="0.068")
RC1 = S1 + "check1_ratio_cdf.csv"
C("9.3", "0.403 against 0.340 at 1.15", RC1, csvw(RC1, "observed_share", cut="1.15"), value="0.403")
C("9.3", "0.403 against 0.340 at 1.15", RC1, csvw(RC1, "generator_share", cut="1.15"), value="0.340")
DO = S1 + "check2_direction_outcome.csv"
C("9.3", "0.253 (0.187 to 0.325)", DO, csvw(DO, "R_D", direction="TD->PIGD"), value="0.253")
C("9.3", "0.253 (0.187 to 0.325)", DO, ex("float(df.loc[df.direction=='TD->PIGD','R_D_ci'].iloc[0].strip('()').split(',')[0])", DO), value="0.187")
C("9.3", "0.253 (0.187 to 0.325)", DO, ex("float(df.loc[df.direction=='TD->PIGD','R_D_ci'].iloc[0].strip('()').split(',')[1])", DO), value="0.325")
C("9.3", "expectation of 0.469", DO, ex("float(df.loc[df.direction=='TD->PIGD','band_R_D'].iloc[0].strip('()').split(',')[0])", DO), value="0.469")
C("9.3", "(37 against 83 expected", DO, csvw(DO, "revert", direction="TD->PIGD"), value="37")
C("9.3", "(37 against 83 expected", DO, csvw(DO, "band_E_revert", direction="TD->PIGD"), value="83")
C("9.3", "0.398 against 0.396", DO, csvw(DO, "R_D", direction="PIGD->TD"), value="0.398")
C("9.3", "0.398 against 0.396", DO, ex("float(df.loc[df.direction=='PIGD->TD','band_R_D'].iloc[0].strip('()').split(',')[0])", DO), value="0.396")
C("9.3", "(244 against 315", DO, ex("int(df.loc[df.direction=='both',['revert','stay','to_Ind']].sum(axis=1).iloc[0])", DO), value="244")
C("9.3", "(244 against 315", DO, ex("float(df.loc[df.direction=='both',['band_E_revert','band_E_stay','band_E_to_Ind']].sum(axis=1).iloc[0])", DO), value="315")
RP = S1 + "check3_residual_proxy.csv"
C("9.3", "interquartile range 1.20", RP, csvw(RP, "iqr", stratum="1-2"), value="1.20")
C("9.3", "against 0.31 at 8 and above", RP, csvw(RP, "iqr", stratum="8+"), value="0.31")
C("9.3", "the 585 patients", S1 + "check4_affected.csv", csvw(S1 + "check4_affected.csv", "patients", group="unaffected"), value="585")

# ---------------------------------------------------------------- 9.4 early split
HP = O + "model/heldout_performance.csv"
C("9.4", "reaches −0.638", HP, csvw(HP, "loglik_per_pair", model="~^HMM"), value="-0.638")
C("9.4", "against −0.665", HP, csvw(HP, "loglik_per_pair", model="~^Discrete"), value="-0.665")
C("9.4", "−0.675 for the continuous", HP, csvw(HP, "loglik_per_pair", model="~^CTMC"), value="-0.675")
C("9.4", "−0.863 for the null", HP, csvw(HP, "loglik_per_pair", model="~^Marginal"), value="-0.863")
C("9.4", "accuracies are 0.769, 0.767, 0.765 and 0.625", HP, csvw(HP, "accuracy", model="~^HMM"), value="0.769")
C("9.4", "accuracies are 0.769, 0.767, 0.765 and 0.625", HP, csvw(HP, "accuracy", model="~^Discrete"), value="0.767")
C("9.4", "accuracies are 0.769, 0.767, 0.765 and 0.625", HP, csvw(HP, "accuracy", model="~^CTMC"), value="0.765")
C("9.4", "accuracies are 0.769, 0.767, 0.765 and 0.625", HP, csvw(HP, "accuracy", model="~^Marginal"), value="0.625")
C("9.4", "Brier scores 0.346, 0.363, 0.367 and 0.518", HP, csvw(HP, "brier", model="~^HMM"), value="0.346")
C("9.4", "Brier scores 0.346, 0.363, 0.367 and 0.518", HP, csvw(HP, "brier", model="~^Discrete"), value="0.363")
C("9.4", "Brier scores 0.346, 0.363, 0.367 and 0.518", HP, csvw(HP, "brier", model="~^CTMC"), value="0.367")
C("9.4", "Brier scores 0.346, 0.363, 0.367 and 0.518", HP, csvw(HP, "brier", model="~^Marginal"), value="0.518")
RBS = O + "model/robustness_across_splits.csv"
C("9.4", "by +0.028 log-likelihood per pair on average (SD 0.009)", RBS, ex("df['HMM - DTMC'].mean()", RBS), value="0.028")
C("9.4", "by +0.028 log-likelihood per pair on average (SD 0.009)", RBS, ex("df['HMM - DTMC'].std(ddof=1)", RBS), value="0.009")
HBI = O + "model/heldout_by_interval.csv"
C("9.4", "by 0.005 per pair at intervals up to 1.5 years", HBI, csvw(HBI, "CTMC - DTMC", interval_years="<= 1.5"), absval=True, value="0.005")
C("9.4", "0.024 at 1.5 to 2.5 years", HBI, csvw(HBI, "CTMC - DTMC", interval_years="1.5-2.5"), absval=True, value="0.024")
C("9.4", "0.099 beyond", HBI, csvw(HBI, "CTMC - DTMC", interval_years="> 2.5"), absval=True, value="0.099")
EM = O + "model/hmm_emission.csv"
C("9.4", "correctly at 0.941", EM, ex("df.set_index(df.columns[0]).loc['TD','TD']", EM), value="0.941")
C("9.4", "PIGD state at 0.842", EM, ex("df.set_index(df.columns[0]).loc['PIGD','PIGD']", EM), value="0.842")
LG = O + "model/hmm_latent_generator.csv"
C("9.4", "PIGD directly at 0.006 per year", LG, ex("df.set_index(df.columns[0]).loc['TD','PIGD']", LG), value="0.006")
C("9.4", "Indeterminate at 0.094 per year", LG, ex("df.set_index(df.columns[0]).loc['TD','Indeterminate']", LG), value="0.094")
C("9.4", "PIGD at 0.356 per year", LG, ex("df.set_index(df.columns[0]).loc['Indeterminate','PIGD']", LG), value="0.356")
SJ = O + "model/mean_sojourn.csv"
C("9.4", "10.1, 2.1 and 16.5 years", SJ, ex("df.set_index(df.columns[0]).loc['TD','HMM (latent)']", SJ), value="10.1")
C("9.4", "10.1, 2.1 and 16.5 years", SJ, ex("df.set_index(df.columns[0]).loc['Indeterminate','HMM (latent)']", SJ), value="2.1")
C("9.4", "10.1, 2.1 and 16.5 years", SJ, ex("df.set_index(df.columns[0]).loc['PIGD','HMM (latent)']", SJ), value="16.5")
C("9.4", "3.4, 0.4 and 2.3 years", SJ, ex("df.set_index(df.columns[0]).loc['TD','CTMC']", SJ), value="3.4")
C("9.4", "3.4, 0.4 and 2.3 years", SJ, ex("df.set_index(df.columns[0]).loc['Indeterminate','CTMC']", SJ), value="0.4")
C("9.4", "3.4, 0.4 and 2.3 years", SJ, ex("df.set_index(df.columns[0]).loc['PIGD','CTMC']", SJ), value="2.3")
C("9.4", "PIGD given TD is 0.191", O + "model/clinical_table_hmm.csv", csvw(O + "model/clinical_table_hmm.csv", "PIGD", current_state="TD", horizon_years="5.0"), value="0.191")
C("9.4", "chain gives 0.296", O + "model/clinical_table_ctmc.csv", csvw(O + "model/clinical_table_ctmc.csv", "PIGD", current_state="TD", horizon_years="5.0"), value="0.296")

# ---------------------------------------------------------------- 9.5 LOPO and stability
C("9.5", "predicts 2,534 of 3,361", LJ, jsn(LJ, "n_correct"), value="2534")
C("9.5", "Wilson interval 0.739 to 0.768", LJ, jsn(LJ, "accuracy_ci_wilson_low"), value="0.739")
C("9.5", "Wilson interval 0.739 to 0.768", LJ, jsn(LJ, "accuracy_ci_wilson_high"), value="0.768")
C("9.5", "log-likelihood per prediction −0.673", LJ, jsn(LJ, "loglik_per_prediction"), value="-0.673")
C("9.5", "Brier score 0.369", LJ, jsn(LJ, "brier"), value="0.369")
LM = O + "loo/loo_model_comparison.csv"
C("9.5", "discrete chain reaches 0.747", LM, csvw(LM, "accuracy", model="Markov (discrete)"), value="0.747")
C("9.5", "the continuous chain 0.745", LM, csvw(LM, "accuracy", model="~^CTMC"), value="0.745")
C("9.5", "recorded-label persistence 0.723", LM, csvw(LM, "accuracy", model="~^Persistence"), value="0.723")
C("9.5", "the marginal null 0.629", LM, csvw(LM, "accuracy", model="~^Marginal"), value="0.629")
PC = O + "loo/loo_per_class_metrics.csv"
C("9.5", "TD is recalled at 0.910 with precision 0.786", PC, csvw(PC, "recall", state="TD"), value="0.910")
C("9.5", "TD is recalled at 0.910 with precision 0.786", PC, csvw(PC, "precision", state="TD"), value="0.786")
C("9.5", "PIGD at 0.666 with precision 0.669", PC, csvw(PC, "recall", state="PIGD"), value="0.666")
C("9.5", "PIGD at 0.666 with precision 0.669", PC, csvw(PC, "precision", state="PIGD"), value="0.669")
CM = S4CV + "cv_config_metrics.csv"
C("9.5", "its maximum is 0.371", CM, csvw(CM, "max_p_Indet", config="C3_f50", rule="argmax"), value="0.371")
AC = O + "loo/loo_accuracy_by_current_state.csv"
C("9.5", "0.811 (TD), 0.488 (Indeterminate) and 0.702 (PIGD)", AC, csvw(AC, "mean", current="TD"), value="0.811")
C("9.5", "0.811 (TD), 0.488 (Indeterminate) and 0.702 (PIGD)", AC, csvw(AC, "mean", current="Indeterminate"), value="0.488")
C("9.5", "0.811 (TD), 0.488 (Indeterminate) and 0.702 (PIGD)", AC, csvw(AC, "mean", current="PIGD"), value="0.702")
AH = O + "loo/loo_accuracy_by_history.csv"
C("9.5", "0.747, 0.790, 0.751, 0.739 and 0.736", AH, csvi(AH, "mean", 0), value="0.747")
C("9.5", "0.747, 0.790, 0.751, 0.739 and 0.736", AH, csvi(AH, "mean", 1), value="0.790")
C("9.5", "0.747, 0.790, 0.751, 0.739 and 0.736", AH, csvi(AH, "mean", 2), value="0.751")
C("9.5", "0.747, 0.790, 0.751, 0.739 and 0.736", AH, csvi(AH, "mean", 3), value="0.739")
C("9.5", "0.747, 0.790, 0.751, 0.739 and 0.736", AH, csvi(AH, "mean", 4), value="0.736")
PS = S4ST + "population_summary.csv"
C("9.5", "agreeing to 0.0014", PS, csvw(PS, "max_abs_dQ_among_best", config="C3_f50"), value="0.0014")
C("9.5", "emission entries to 0.0003", PS, csvw(PS, "max_abs_dE_among_best", config="C3_f50"), value="0.0003")
C("9.5", "38 of 40 starts", PS, csvw(PS, "share_within_0.01", config="C3_free"), 40, value="38")
C("9.5", "differing by up to 0.187", PS, csvw(PS, "max_abs_dQ_among_best", config="C3_free"), value="0.187")
C("9.5", "emission entries by up to 0.93", PS, csvw(PS, "max_abs_dE_among_best", config="C3_free"), value="0.93")
C("9.5", "one of 40 starts", PS, csvw(PS, "share_within_0.01", config="C5_free"), 40, value="1")
C("9.5", "7.4 log-likelihood units", PS, csvw(PS, "default_start_gap", config="C5_free"), value="7.4")
SP = S4CV + "selected_procedure_metrics.csv"
C("9.5", "reaches 0.755 (0.735 to 0.774)", SP, csvw(SP, "accuracy", rule="argmax"), value="0.755")
C("9.5", "reaches 0.755 (0.735 to 0.774)", SP, csvw(SP, "accuracy_lo", rule="argmax"), value="0.735")
C("9.5", "reaches 0.755 (0.735 to 0.774)", SP, csvw(SP, "accuracy_hi", rule="argmax"), value="0.774")
C("9.5", "Indeterminate recall of 0.25 to 0.32", CM, csvw(CM, "recall_Indeterminate", config="C3_f50", rule="prior_corrected"), value="0.25")
C("9.5", "Indeterminate recall of 0.25 to 0.32", CM, csvw(CM, "recall_Indeterminate", config="C5_free", rule="prior_corrected"), value="0.32")
C("9.5", "falls to 0.68 to 0.71", CM, csvw(CM, "accuracy", config="C5_free", rule="prior_corrected"), value="0.68")
C("9.5", "falls to 0.68 to 0.71", CM, csvw(CM, "accuracy", config="C3_f50", rule="prior_corrected"), value="0.71")
C("9.5", "reaches 0.751", HM, csvw(HM, "accuracy", model="CT-HMM, 5 latent states"), value="0.751")
L5 = O + "loo_5hidden/loo_predictions_all_models.csv"
C("9.5", "404 of 934 folds", L5, ex("int(df[df.model=='HMM'].groupby('held_out_patient')['train_converged'].first().sum())", L5), value="404")
C("9.5", "59.2% of its predictions", L5, ex("100*(1-df[df.model=='HMM']['train_converged'].mean())", L5), value="59.2")

# ---------------------------------------------------------------- 9.6 growth / Student-t
PSM = O + "hbayes/posterior_summary.csv"
C("9.6", "starts at 2.29 (95% posterior interval 2.16 to 2.43)", PSM, csvw(PSM, "mean", parameter="~^beta0"), value="2.29")
C("9.6", "2.16 to 2.43", PSM, csvw(PSM, "q2.5", parameter="~^beta0"), value="2.16")
C("9.6", "2.16 to 2.43", PSM, csvw(PSM, "q97.5", parameter="~^beta0"), value="2.43")
C("9.6", "falls by 0.093 per year", PSM, csvw(PSM, "mean", parameter="~^beta1"), absval=True, value="0.093")
C("9.6", "baseline SD is 1.34", PSM, csvw(PSM, "mean", parameter="~^tau0"), value="1.34")
C("9.6", "slope SD 0.103", PSM, csvw(PSM, "mean", parameter="~^tau1"), value="0.103")
C("9.6", "residual SD 1.14", PSM, csvw(PSM, "mean", parameter="~^sigma"), value="1.14")
VD = O + "exploration/variance_decomposition.csv"
C("9.6", "54% between patients and 46% within", VD, csvw(VD, "share", component="between patients"), 100, value="54")
C("9.6", "54% between patients and 46% within", VD, csvw(VD, "share", component="within patients"), 100, value="46")
C("9.6", "within-patient SD of 1.22", VD, csvw(VD, "sd", component="within patients"), value="1.22")
HH = O + "hbayes/head_to_head.csv"
C("9.6", "reaches −0.656 log-likelihood per pair, 0.761 accuracy and 0.357 Brier", HH, csvw(HH, "loglik_per_pair", model="~^HMM"), value="-0.656")
C("9.6", "0.761 accuracy", HH, csvw(HH, "accuracy", model="~^HMM"), value="0.761")
C("9.6", "0.357 Brier", HH, csvw(HH, "brier", model="~^HMM"), value="0.357")
C("9.6", "−0.681, 0.737 and 0.379", HH, csvw(HH, "loglik_per_pair", model="~^Hierarchical"), value="-0.681")
C("9.6", "−0.681, 0.737 and 0.379", HH, csvw(HH, "accuracy", model="~^Hierarchical"), value="0.737")
C("9.6", "−0.681, 0.737 and 0.379", HH, csvw(HH, "brier", model="~^Hierarchical"), value="0.379")
C("9.6", "825 pairs", HH, csvw(HH, "n", model="~^HMM"), value="825")
RBH = O + "hbayes/robustness.csv"
C("9.6", "(mean +0.058 per pair, SD 0.019)", RBH, ex("df['HMM - HB'].mean()", RBH), value="0.058")
C("9.6", "(mean +0.058 per pair, SD 0.019)", RBH, ex("df['HMM - HB'].std(ddof=1)", RBH), value="0.019")
LCV = O + "validation/longitudinal_cv.csv"
C("9.6", "from 0.576 with one conditioning visit to 0.647 with two and 0.666 with three", LCV, csvw(LCV, "accuracy", train_visits="1"), value="0.576")
C("9.6", "from 0.576 with one conditioning visit to 0.647 with two and 0.666 with three", LCV, csvw(LCV, "accuracy", train_visits="2"), value="0.647")
C("9.6", "from 0.576 with one conditioning visit to 0.647 with two and 0.666 with three", LCV, csvw(LCV, "accuracy", train_visits="3"), value="0.666")
C("9.6", "falls from 0.974 to 0.904", LCV, csvw(LCV, "coverage_95", train_visits="1"), value="0.974")
C("9.6", "falls from 0.974 to 0.904", LCV, csvw(LCV, "coverage_95", train_visits="3"), value="0.904")
CG = O + "validation/calibration_growth_model.csv"
C("9.6", "mean gap of 0.105", CG, ex("(df['observed_rate']-df['mean_predicted']).abs().mean()", CG), value="0.105")
C("9.6", "gains +0.112 (95% CI +0.061 to +0.168)", ST, jsn(ST, "Q1_residual_tails/GT-GG/lpd_mean"), value="0.112")
C("9.6", "+0.061 to +0.168", ST, jsn(ST, "Q1_residual_tails/GT-GG/lpd_ci/0"), value="0.061")
C("9.6", "+0.061 to +0.168", ST, jsn(ST, "Q1_residual_tails/GT-GG/lpd_ci/1"), value="0.168")
C("9.6", "+0.038 (+0.020 to +0.055)", ST, jsn(ST, "Q1_residual_tails/GT-GG/cls_mean"), value="0.038")
C("9.6", "+0.038 (+0.020 to +0.055)", ST, jsn(ST, "Q1_residual_tails/GT-GG/cls_ci/0"), value="0.020")
C("9.6", "+0.038 (+0.020 to +0.055)", ST, jsn(ST, "Q1_residual_tails/GT-GG/cls_ci/1"), value="0.055")
C("9.6", "gains +0.106 (+0.058 to +0.158)", ST, jsn(ST, "Q1_residual_tails/TT-TG/lpd_mean"), value="0.106")
C("9.6", "gains +0.106 (+0.058 to +0.158)", ST, jsn(ST, "Q1_residual_tails/TT-TG/lpd_ci/0"), value="0.058")
C("9.6", "gains +0.106 (+0.058 to +0.158)", ST, jsn(ST, "Q1_residual_tails/TT-TG/lpd_ci/1"), value="0.158")
C("9.6", "about 2.1", ST, jsn(ST, "calibration_theta/nu_e"), value="2.1")
C("9.6", "posterior 1.88 to 2.39", S1DOC, txt(S1DOC, r"posterior ν_e is 2\.10 \(95 % ([\d.]+)"), value="1.88")
C("9.6", "posterior 1.88 to 2.39", S1DOC, txt(S1DOC, r"posterior ν_e is 2\.10 \(95 % [\d.]+–([\d.]+)\)"), value="2.39")
C("9.6", "(R-hat 1.018, effective sample size 248)", ST, jsn(ST, "Q2_slope_tails/convergence/gamma/rhat"), value="1.018")
C("9.6", "(R-hat 1.018, effective sample size 248)", ST, jsn(ST, "Q2_slope_tails/convergence/gamma/ess"), value="248")
C("9.6", "converged (R-hat 1.004, effective sample size 937)", Q2, jsn(Q2, "conv/gamma2_true/rhat"), value="1.004")
C("9.6", "converged (R-hat 1.004, effective sample size 937)", Q2, jsn(Q2, "conv/gamma2_true/ess"), value="937")
C("9.6", "median of 2.09 (1.48 to 4.11)", Q2, jsn(Q2, "a/gamma2_true/q50"), value="2.09")
C("9.6", "median of 2.09 (1.48 to 4.11)", Q2, jsn(Q2, "a/gamma2_true/q2.5"), value="1.48")
C("9.6", "median of 2.09 (1.48 to 4.11)", Q2, jsn(Q2, "a/gamma2_true/q97.5"), value="4.11")
C("9.6", "55 of 60 detections", Q2, jsn(Q2, "c/P/flags"), value="55")
C("9.6", "zero false flags in 180", Q2, ex("sum(d['c'][k]['flags'] for k in ['C1','C2','C3'])", Q2), value="0")

# ---------------------------------------------------------------- 9.7 first visit
NPM = O + "newpatient/model_comparison.csv"
C("9.7", "gives −0.781", NPM, ex("df['subtype + horizon'].mean()", NPM), value="-0.781")
C("9.7", "adding the baseline ratio −0.773", NPM, ex("df['+ baseline ratio'].mean()", NPM), value="-0.773")
C("9.7", "ten baseline measures −0.763", NPM, ex("df['+ covariates'].mean()", NPM), value="-0.763")
C("9.7", "and both −0.755", NPM, ex("df['+ ratio + covariates'].mean()", NPM), value="-0.755")
C("9.7", "against −0.787 for the CT-HMM", NPM, ex("df['HMM (no history)'].mean()", NPM), value="-0.787")
NPH = O + "newpatient/heldout_metrics.csv"
C("9.7", "reaches 0.719 accuracy and 0.413 Brier", NPH, csvi(NPH, "accuracy", 0), value="0.719")
C("9.7", "reaches 0.719 accuracy and 0.413 Brier", NPH, csvi(NPH, "brier", 0), value="0.413")
NPC = O + "newpatient/covariate_importance.csv"
C("9.7", "per prediction (0.015)", NPC, csvw(NPC, "cost_of_dropping", covariate="updrs2_score"), value="0.015")
C("9.7", "cost 0.005, 0.004 and 0.002", NPC, csvw(NPC, "cost_of_dropping", covariate="MIA_PUTAMEN_BILAT"), value="0.005")
C("9.7", "cost 0.005, 0.004 and 0.002", NPC, csvw(NPC, "cost_of_dropping", covariate="age"), value="0.004")
C("9.7", "cost 0.005, 0.004 and 0.002", NPC, csvw(NPC, "cost_of_dropping", covariate="hy"), value="0.002")
NPB = O + "newpatient/by_horizon.csv"
C("9.7", "from 0.776 within 1.5 years to 0.616 beyond 6.5 years", NPB, csvw(NPB, "accuracy", horizon="<1.5y"), value="0.776")
C("9.7", "from 0.776 within 1.5 years to 0.616 beyond 6.5 years", NPB, csvw(NPB, "accuracy", horizon=">6.5y"), value="0.616")

# ---------------------------------------------------------------- 9.8 GPR
RM = O + "step3_gpr/reconciled_metrics.csv"
C("9.8", "pooled MAE of 0.972 (K1, K2) and 0.971 (K3)", RM, csvw(RM, "MAE_pooled", cohort="unfiltered", model="K1"), value="0.972")
C("9.8", "pooled MAE of 0.972 (K1, K2) and 0.971 (K3)", RM, csvw(RM, "MAE_pooled", cohort="unfiltered", model="K2"), value="0.972")
C("9.8", "pooled MAE of 0.972 (K1, K2) and 0.971 (K3)", RM, csvw(RM, "MAE_pooled", cohort="unfiltered", model="K3"), value="0.971")
C("9.8", "against 1.005 for the linear K0", RM, csvw(RM, "MAE_pooled", cohort="unfiltered", model="K0"), value="1.005")
C("9.8", "1.006 for the growth model", RM, csvw(RM, "MAE_pooled", cohort="unfiltered", model="hbayes"), value="1.006")
C("9.8", "1.043 for persistence", RM, csvw(RM, "MAE_pooled", cohort="unfiltered", model="persistence"), value="1.043")
C("9.8", "1.063 for the patient mean", RM, csvw(RM, "MAE_pooled", cohort="unfiltered", model="patient_mean"), value="1.063")
C("9.8", "1.336 for the patient OLS line", RM, csvw(RM, "MAE_pooled", cohort="unfiltered", model="patient_ols"), value="1.336")
C("9.8", "1.395 for the population mean", RM, csvw(RM, "MAE_pooled", cohort="unfiltered", model="population"), value="1.395")
PCG = O + "step3_gpr/paired_comparisons.csv"
C("9.8", "K1 minus K0 −0.033 (−0.047 to −0.021)", PCG, csvw(PCG, "diff_pooled", cohort="unfiltered", model_a="K1", model_b="K0"), value="-0.033")
C("9.8", "K1 minus K0 −0.033 (−0.047 to −0.021)", PCG, csvw(PCG, "pooled_lo", cohort="unfiltered", model_a="K1", model_b="K0"), value="-0.047")
C("9.8", "K1 minus K0 −0.033 (−0.047 to −0.021)", PCG, csvw(PCG, "pooled_hi", cohort="unfiltered", model_a="K1", model_b="K0"), value="-0.021")
C("9.8", "−0.019 (−0.028 to −0.011) per patient", PCG, csvw(PCG, "diff_patient", cohort="unfiltered", model_a="K1", model_b="K0"), value="-0.019")
C("9.8", "−0.019 (−0.028 to −0.011) per patient", PCG, csvw(PCG, "patient_lo", cohort="unfiltered", model_a="K1", model_b="K0"), value="-0.028")
C("9.8", "−0.019 (−0.028 to −0.011) per patient", PCG, csvw(PCG, "patient_hi", cohort="unfiltered", model_a="K1", model_b="K0"), value="-0.011")
C("9.8", "K1 minus persistence −0.071 (−0.098 to −0.045)", PCG, csvw(PCG, "diff_pooled", cohort="unfiltered", model_a="K1", model_b="persistence"), value="-0.071")
C("9.8", "K1 minus persistence −0.071 (−0.098 to −0.045)", PCG, csvw(PCG, "pooled_lo", cohort="unfiltered", model_a="K1", model_b="persistence"), value="-0.098")
C("9.8", "K1 minus persistence −0.071 (−0.098 to −0.045)", PCG, csvw(PCG, "pooled_hi", cohort="unfiltered", model_a="K1", model_b="persistence"), value="-0.045")
C("9.8", "intervals is 0.941", RM, csvw(RM, "coverage95", cohort="unfiltered", model="K1"), value="0.941")
C("9.8", "12.9% to 13.9%", RM, csvw(RM, "negative_mass", cohort="unfiltered", model="K3"), 100, value="12.9")
C("9.8", "12.9% to 13.9%", RM, csvw(RM, "negative_mass", cohort="unfiltered", model="K0"), 100, value="13.9")
C("9.8", "K1 reaches 0.724 accuracy", RM, csvw(RM, "accuracy", cohort="unfiltered", model="K1"), value="0.724")
C("9.8", "below the 0.733 of ratio persistence", RM, csvw(RM, "accuracy", cohort="unfiltered", model="persistence"), value="0.733")
C("9.8", "All 11,444 kernel fits", O + "step3_gpr/", ex("sum(len(pd.read_csv(ROOT/('outputs/step3_gpr/gpr_fold_fits_%s.csv' % c))) for c in ['unfiltered','A4','D1','E2'])"), value="11444")
MFH = O + "step3_gpr/mean_function_holdout_summary.csv"
C("9.8", "(zero-mean minus fitted-mean −0.007, −0.019 to +0.006)", MFH, csvw(MFH, "diff_pooled", model="K2_zeromean"), value="-0.007")
C("9.8", "(zero-mean minus fitted-mean −0.007, −0.019 to +0.006)", MFH, csvw(MFH, "diff_pooled_lo", model="K2_zeromean"), value="-0.019")
C("9.8", "(zero-mean minus fitted-mean −0.007, −0.019 to +0.006)", MFH, csvw(MFH, "diff_pooled_hi", model="K2_zeromean"), value="0.006")
C("9.8", "CRPS (+0.019, +0.013 to +0.025", MEMO, txt(MEMO, r"\| CRPS \| \*\*\+([\d.]+)\*\* \|"), value="0.019")
C("9.8", "mean squared error (+0.097, +0.063 to +0.131)", MEMO, txt(MEMO, r"\| MSE \| \*\*\+([\d.]+)\*\* \|"), value="0.097")
C("9.8", "(0.130 against 0.170)", MFH, csvw(MFH, "negative_mass", model="K2"), value="0.130")
C("9.8", "(0.130 against 0.170)", MFH, csvw(MFH, "negative_mass", model="K2_zeromean"), value="0.170")
MSP = O + "step3_gpr/multistep_paired.csv"
C("9.8", "(K1 minus K0 −0.062, −0.095 to −0.031 over 1,389", MSP, csvw(MSP, "diff", cohort="unfiltered", model_a="K1", model_b="K0"), value="-0.062")
C("9.8", "(K1 minus K0 −0.062, −0.095 to −0.031 over 1,389", MSP, csvw(MSP, "lo", cohort="unfiltered", model_a="K1", model_b="K0"), value="-0.095")
C("9.8", "(K1 minus K0 −0.062, −0.095 to −0.031 over 1,389", MSP, csvw(MSP, "hi", cohort="unfiltered", model_a="K1", model_b="K0"), value="-0.031")
C("9.8", "over 1,389 forecast points", MSP, csvw(MSP, "n_points", cohort="unfiltered", model_a="K1", model_b="K0"), value="1389")
C("9.8", "grows to −0.158 (−0.233 to −0.086)", MSP, csvw(MSP, "diff", cohort="unfiltered", model_a="K1", model_b="~persist"), value="-0.158")
C("9.8", "grows to −0.158 (−0.233 to −0.086)", MSP, csvw(MSP, "lo", cohort="unfiltered", model_a="K1", model_b="~persist"), value="-0.233")
C("9.8", "grows to −0.158 (−0.233 to −0.086)", MSP, csvw(MSP, "hi", cohort="unfiltered", model_a="K1", model_b="~persist"), value="-0.086")
MWP = O + "step3_gpr/multistep_audit_within_patient.csv"
C("9.8", "by +0.230 (+0.051 to +0.402)", MWP, csvw(MWP, "delta_MAE", cohort="unfiltered", model="K1", from_bin="0-1y", to_bin="1-2y"), value="0.230")
C("9.8", "by +0.230 (+0.051 to +0.402)", MWP, csvw(MWP, "lo", cohort="unfiltered", model="K1", from_bin="0-1y", to_bin="1-2y"), value="0.051")
C("9.8", "by +0.230 (+0.051 to +0.402)", MWP, csvw(MWP, "hi", cohort="unfiltered", model="K1", from_bin="0-1y", to_bin="1-2y"), value="0.402")
C("9.8", "(−0.371, −0.655 to −0.104 in the unfiltered cohort)", MEMO, txt(MEMO, r"\| unfiltered \| [^|]+\| [^|]+\| \*\*(−0\.371)"), value="-0.371")

# ---------------------------------------------------------------- 9.9 subscores
SS = O + "step3_gpr/subscore_summary.csv"
C("9.9", "0.053 for tremor", SS, csvw(SS, "negative_mass", subscore="tremor", model="K1", cohort="expanded"), value="0.053")
C("9.9", "median within-patient SD 0.19 for tremor and 0.14 for PIGD against 0.76 for the ratio", MEMO, txt(MEMO, r"\| median within-patient SD \| ([\d.]+) \|"), value="0.19")
C("9.9", "0.14 for PIGD", MEMO, txt(MEMO, r"\| median within-patient SD \| [\d.]+ \| ([\d.]+) \|"), value="0.14")
C("9.9", "against 0.76 for the ratio", MEMO, txt(MEMO, r"\| median within-patient SD \| [\d.]+ \| [\d.]+ \| \*\*([\d.]+)\*\*"), value="0.76")
JP = O + "joint_subscore_gp/paired.csv"
C("9.9", "K1 minus persistence is −0.003 (−0.008 to +0.002) on the tremor score", JP, csvw(JP, "delta_MAE", outcome="tremor", comparison="gp_indep_K1 - persistence"), value="-0.003")
C("9.9", "−0.008 to +0.002", JP, csvw(JP, "lo", outcome="tremor", comparison="gp_indep_K1 - persistence"), value="-0.008")
C("9.9", "−0.008 to +0.002", JP, csvw(JP, "hi", outcome="tremor", comparison="gp_indep_K1 - persistence"), value="0.002")
C("9.9", "+0.013 (+0.008 to +0.019) on the PIGD score", JP, csvw(JP, "delta_MAE", outcome="pigd", comparison="gp_indep_K1 - persistence"), value="0.013")
C("9.9", "+0.013 (+0.008 to +0.019) on the PIGD score", JP, csvw(JP, "lo", outcome="pigd", comparison="gp_indep_K1 - persistence"), value="0.008")
C("9.9", "+0.013 (+0.008 to +0.019) on the PIGD score", JP, csvw(JP, "hi", outcome="pigd", comparison="gp_indep_K1 - persistence"), value="0.019")
PZ = O + "joint_subscore_gp/pigd_zero_analysis.csv"
C("9.9", "446 visits where the PIGD score is zero", PZ, csvw(PZ, "n", model="persistence"), value="446")
C("9.9", "below 0.1 at 62% of them", PZ, csvw(PZ, "frac_pigd_pred_below_0.1", model="persistence"), 100, value="62")
C("9.9", "mean prediction there is 0.242", PZ, csvw(PZ, "pigd_pred_mean", model="gp_indep_K1"), value="0.242")
JT = O + "joint_subscore/model_summary.txt"
C("9.9", "+0.023 per year (0.020 to 0.026) for tremor", JT, txt(JT, r"beta_tremor\s+([\d.]+)"), value="0.023")
C("9.9", "+0.066 per year (0.062 to 0.069) for PIGD", JT, txt(JT, r"beta_pigd\s+([\d.]+)"), value="0.066")
C("9.9", "random-intercept correlation of −0.015", JT, txt(JT, r"correlation (-[\d.]+)\n\nRESIDUAL"), value="-0.015")
C("9.9", "residual correlation of +0.019", JT, txt(JT, r"RESIDUAL COVARIANCE[\s\S]*?correlation \+([\d.]+)"), value="0.019")
C("9.9", "intraclass correlations of 0.61 and 0.45", JT, txt(JT, r"ICC tremor ([\d.]+)"), value="0.61")
C("9.9", "intraclass correlations of 0.61 and 0.45", JT, txt(JT, r"ICC pigd ([\d.]+)"), value="0.45")
JM = O + "joint_subscore/metrics.csv"
C("9.9", "(0.255 for tremor, 0.239 for PIGD)", JM, csvw(JM, "MAE", model="joint_lmm", outcome="tremor"), value="0.255")
C("9.9", "(0.255 for tremor, 0.239 for PIGD)", JM, csvw(JM, "MAE", model="joint_lmm", outcome="pigd"), value="0.239")
JPP = O + "joint_subscore/paired_vs_persistence.csv"
C("9.9", "(+0.012, +0.004 to +0.020; +0.035, +0.027 to +0.044)", JPP, csvw(JPP, "delta_MAE", outcome="tremor", comparison="joint_lmm - persistence"), value="0.012")
C("9.9", "(+0.012, +0.004 to +0.020; +0.035, +0.027 to +0.044)", JPP, csvw(JPP, "lo", outcome="tremor", comparison="joint_lmm - persistence"), value="0.004")
C("9.9", "(+0.012, +0.004 to +0.020; +0.035, +0.027 to +0.044)", JPP, csvw(JPP, "hi", outcome="tremor", comparison="joint_lmm - persistence"), value="0.020")
C("9.9", "(+0.012, +0.004 to +0.020; +0.035, +0.027 to +0.044)", JPP, csvw(JPP, "delta_MAE", outcome="pigd", comparison="joint_lmm - persistence"), value="0.035")
S3P = S3 + "paired.csv"
C("9.9", "tremor CRPS by −0.0002 at most", S3P, csvw(S3P, "difference", model="gp_joint_K0_full", reference="gp_joint_K0_diag", outcome="tremor", metric="crps"), value="-0.0002")
C("9.9", "(K2 minus K0 joint log score +0.073, +0.051 to +0.096)", S3P, csvw(S3P, "difference", model="gp_indep_K2", reference="gp_indep_K0", outcome="joint", metric="log_score"), value="0.073")
C("9.9", "(K2 minus K0 joint log score +0.073, +0.051 to +0.096)", S3P, csvw(S3P, "lo", model="gp_indep_K2", reference="gp_indep_K0", outcome="joint", metric="log_score"), value="0.051")
C("9.9", "(K2 minus K0 joint log score +0.073, +0.051 to +0.096)", S3P, csvw(S3P, "hi", model="gp_indep_K2", reference="gp_indep_K0", outcome="joint", metric="log_score"), value="0.096")
C("9.9", "(K1 minus persistence joint log score +0.144, +0.102 to +0.186)", S3P, csvw(S3P, "difference", model="gp_indep_K1", reference="persistence", outcome="joint", metric="log_score"), value="0.144")
C("9.9", "(K1 minus persistence joint log score +0.144, +0.102 to +0.186)", S3P, csvw(S3P, "lo", model="gp_indep_K1", reference="persistence", outcome="joint", metric="log_score"), value="0.102")
C("9.9", "(K1 minus persistence joint log score +0.144, +0.102 to +0.186)", S3P, csvw(S3P, "hi", model="gp_indep_K1", reference="persistence", outcome="joint", metric="log_score"), value="0.186")
C("9.9", "All 3,736 MOGP fits converged", GPRDOC, txt(GPRDOC, r"(3,736) of 3,736 fits converged"), value="3736")
SPH = S3 + "subtype_paired_vs_hmm.csv"
C("9.9", "+0.0024 (−0.0045 to +0.0092) for the independent K1", SPH, csvw(SPH, "difference", model="gp_indep_K1", metric="accuracy_prob_argmax"), value="0.0024")
C("9.9", "+0.0024 (−0.0045 to +0.0092) for the independent K1", SPH, csvw(SPH, "lo", model="gp_indep_K1", metric="accuracy_prob_argmax"), value="-0.0045")
C("9.9", "+0.0024 (−0.0045 to +0.0092) for the independent K1", SPH, csvw(SPH, "hi", model="gp_indep_K1", metric="accuracy_prob_argmax"), value="0.0092")
C("9.9", "+0.0045 (−0.0026 to +0.0117) for the joint K1", SPH, csvw(SPH, "difference", model="gp_joint_K1_full", metric="accuracy_prob_argmax"), value="0.0045")
C("9.9", "+0.0045 (−0.0026 to +0.0117) for the joint K1", SPH, csvw(SPH, "lo", model="gp_joint_K1_full", metric="accuracy_prob_argmax"), value="-0.0026")
C("9.9", "+0.0045 (−0.0026 to +0.0117) for the joint K1", SPH, csvw(SPH, "hi", model="gp_joint_K1_full", metric="accuracy_prob_argmax"), value="0.0117")
C("9.9", "better than the CT-HMM (−0.020, −0.032 to −0.007)", SPH, csvw(SPH, "difference", model="gp_indep_K1", metric="log_loss"), value="-0.020")
C("9.9", "better than the CT-HMM (−0.020, −0.032 to −0.007)", SPH, csvw(SPH, "lo", model="gp_indep_K1", metric="log_loss"), value="-0.032")
C("9.9", "better than the CT-HMM (−0.020, −0.032 to −0.007)", SPH, csvw(SPH, "hi", model="gp_indep_K1", metric="log_loss"), value="-0.007")
C("9.9", "LMM is worse (+0.018, +0.004 to +0.032)", SPH, csvw(SPH, "difference", model="lmm_ri", metric="log_loss"), value="0.018")
C("9.9", "LMM is worse (+0.018, +0.004 to +0.032)", SPH, csvw(SPH, "lo", model="lmm_ri", metric="log_loss"), value="0.004")
C("9.9", "LMM is worse (+0.018, +0.004 to +0.032)", SPH, csvw(SPH, "hi", model="lmm_ri", metric="log_loss"), value="0.032")
SPL = S3 + "subtype_paired_vs_label_persistence.csv"
C("9.9", "reach 0.756 (independent K1) and 0.758 (joint K1, full)", SPL, csvw(SPL, "accuracy", model="gp_indep_K1", rule="prob_argmax"), value="0.756")
C("9.9", "reach 0.756 (independent K1) and 0.758 (joint K1, full)", SPL, csvw(SPL, "accuracy", model="gp_joint_K1_full", rule="prob_argmax"), value="0.758")
C("9.9", "the GPs reach 0.730", SPL, csvw(SPL, "accuracy", model="gp_indep_K1", rule="point"), value="0.730")
C("9.9", "Indeterminate with recall 0.17", SPL, csvw(SPL, "recall_Indeterminate", model="gp_indep_K1", rule="point"), value="0.17")

# ---------------------------------------------------------------- 9.10 time-varying / Stage 5
TVS = O + "time_varying/period_slopes.csv"
C("9.10", "(0.045, 0.057 and 0.080 per year", TVS, csvw(TVS, "slope", outcome="pigd", period="0-2y"), value="0.045")
C("9.10", "(0.045, 0.057 and 0.080 per year", TVS, csvw(TVS, "slope", outcome="pigd", period="2-5y"), value="0.057")
C("9.10", "(0.045, 0.057 and 0.080 per year", TVS, csvw(TVS, "slope", outcome="pigd", period="5y+"), value="0.080")
C("9.10", "(0.033, 0.033 and 0.013 per year)", TVS, csvw(TVS, "slope", outcome="tremor", period="0-2y"), value="0.033")
C("9.10", "(0.033, 0.033 and 0.013 per year)", TVS, csvw(TVS, "slope", outcome="tremor", period="5y+"), value="0.013")
TVC = O + "time_varying/slope_contrasts.csv"
C("9.10", "+0.035 (+0.020 to +0.050) for PIGD", TVC, csvw(TVC, "difference", outcome="pigd", contrast="5y+ - 0-2y"), value="0.035")
C("9.10", "+0.035 (+0.020 to +0.050) for PIGD", TVC, csvw(TVC, "ci_low", outcome="pigd", contrast="5y+ - 0-2y"), value="0.020")
C("9.10", "+0.035 (+0.020 to +0.050) for PIGD", TVC, csvw(TVC, "ci_high", outcome="pigd", contrast="5y+ - 0-2y"), value="0.050")
C("9.10", "−0.020 (−0.034 to −0.006) for tremor", TVC, csvw(TVC, "difference", outcome="tremor", contrast="5y+ - 0-2y"), value="-0.020")
C("9.10", "−0.020 (−0.034 to −0.006) for tremor", TVC, csvw(TVC, "ci_low", outcome="tremor", contrast="5y+ - 0-2y"), value="-0.034")
C("9.10", "−0.020 (−0.034 to −0.006) for tremor", TVC, csvw(TVC, "ci_high", outcome="tremor", contrast="5y+ - 0-2y"), value="-0.006")
TVW = O + "time_varying/within_patient_rate_change.csv"
C("9.10", "within the 226 patients", TVW, csvw(TVW, "n_patients", outcome="pigd", from_period="0-2y", to_period="5y+"), value="226")
C("9.10", "rises by +0.073 (+0.043 to +0.101)", TVW, csvw(TVW, "delta_rate", outcome="pigd", from_period="0-2y", to_period="5y+"), value="0.073")
C("9.10", "rises by +0.073 (+0.043 to +0.101)", TVW, csvw(TVW, "ci_low", outcome="pigd", from_period="0-2y", to_period="5y+"), value="0.043")
C("9.10", "rises by +0.073 (+0.043 to +0.101)", TVW, csvw(TVW, "ci_high", outcome="pigd", from_period="0-2y", to_period="5y+"), value="0.101")
C("9.10", "falls by −0.040 (−0.073 to −0.008)", TVW, csvw(TVW, "delta_rate", outcome="tremor", from_period="0-2y", to_period="5y+"), value="-0.040")
C("9.10", "falls by −0.040 (−0.073 to −0.008)", TVW, csvw(TVW, "ci_low", outcome="tremor", from_period="0-2y", to_period="5y+"), value="-0.073")
C("9.10", "falls by −0.040 (−0.073 to −0.008)", TVW, csvw(TVW, "ci_high", outcome="tremor", from_period="0-2y", to_period="5y+"), value="-0.008")
TVP = O + "time_varying/paired.csv"
C("9.10", "by −0.0034 (−0.0044 to −0.0023)", TVP, csvw(TVP, "delta_MAE", outcome="pigd", comparison="lmm_spline_2_5 - lmm_constant"), value="-0.0034")
C("9.10", "by −0.0034 (−0.0044 to −0.0023)", TVP, csvw(TVP, "lo", outcome="pigd", comparison="lmm_spline_2_5 - lmm_constant"), value="-0.0044")
C("9.10", "by −0.0034 (−0.0044 to −0.0023)", TVP, csvw(TVP, "hi", outcome="pigd", comparison="lmm_spline_2_5 - lmm_constant"), value="-0.0023")
C("9.10", "derived label is 0.709", O + "time_varying/subtype_metrics.csv", csvw(O + "time_varying/subtype_metrics.csv", "accuracy", model="lmm_spline_2_5"), value="0.709")
SLR = S5 + "knots/sup_lr_test.csv"
C("9.10", "supremum likelihood ratio 39.1", SLR, csvw(SLR, "real_sup_lr", null="RI"), value="39.1")
C("9.10", "95th percentile of 7.7", SLR, csvw(SLR, "null_q95", null="RI"), value="7.7")
C("9.10", "20.4 under the random-slope null", SLR, csvw(SLR, "null_q95", null="RS"), value="20.4")
C("9.10", "= 0.005 under both", SLR, csvw(SLR, "p_value", null="RI"), value="0.005")
C("9.10", "best single knot is at 4.75 years", SLR, csvw(SLR, "real_argmax_knot", null="RI"), value="4.75")
KB = S5 + "knots/knot_argmax_bootstrap.csv"
C("9.10", "median of 5.0 years", KB, ex("df['knot'].median()", KB), value="5.0")
C("9.10", "interquartile range of 4.25 to 9.25 years", KB, ex("df['knot'].quantile(0.25)", KB), value="4.25")
C("9.10", "interquartile range of 4.25 to 9.25 years", KB, ex("df['knot'].quantile(0.75)", KB), value="9.25")
MLF = S5 + "structures/ml_fits.csv"
C("9.10", "(37.8 units better than linear)", MLF, csvw(MLF, "dAIC_vs_best", model="linear"), value="37.8")
CVL = S5 + "structures/cv_paired_vs_linear.csv"
C("9.10", "(absolute error −0.003, CRPS −0.001)", CVL, csvw(CVL, "difference", model="ns_df3", outcome="pigd", metric="ae"), value="-0.003")
C("9.10", "(absolute error −0.003, CRPS −0.001)", CVL, csvw(CVL, "difference", model="ns_df3", outcome="pigd", metric="crps"), value="-0.001")
BVW = S5 + "bootstrap/period_slopes_boot_vs_wald.csv"
C("9.10", "factor of 1.7 (tremor) and 2.2 (PIGD)", BVW, csvw(BVW, "se_ratio_boot_over_wald", model="primary_RI", outcome="tremor", period="5-10y"), value="1.7")
C("9.10", "factor of 1.7 (tremor) and 2.2 (PIGD)", BVW, csvw(BVW, "se_ratio_boot_over_wald", model="primary_RI", outcome="pigd", period="5-10y"), value="2.2")
ATT = S5 + "attrition/loss_of_followup_logit.csv"
C("9.10", "rise by +0.52 (+0.36 to +0.71) per point of PIGD score", ATT, csvw(ATT, "log_odds", term="pigd_score"), value="0.52")
C("9.10", "rise by +0.52 (+0.36 to +0.71) per point of PIGD score", ATT, csvw(ATT, "boot_lo", term="pigd_score"), value="0.36")
C("9.10", "rise by +0.52 (+0.36 to +0.71) per point of PIGD score", ATT, csvw(ATT, "boot_hi", term="pigd_score"), value="0.71")
C("9.10", "by +0.14 per year", ATT, csvw(ATT, "log_odds", term="years_from_baseline"), value="0.14")
CSL = S5 + "attrition/completer_slopes.csv"
C("9.10", "at least six years (247 patients)", CSL, csvw(CSL, "n_patients", completers_min_followup="6.0", outcome="pigd", period="0-2y"), value="247")
C("9.10", "(PIGD late minus early +0.044, +0.017 to +0.072", CSL, csvw(CSL, "slope", completers_min_followup="6.0", outcome="pigd", period="~^late"), value="0.044")
C("9.10", "(PIGD late minus early +0.044, +0.017 to +0.072", CSL, csvw(CSL, "boot_lo", completers_min_followup="6.0", outcome="pigd", period="~^late"), value="0.017")
C("9.10", "(PIGD late minus early +0.044, +0.017 to +0.072", CSL, csvw(CSL, "boot_hi", completers_min_followup="6.0", outcome="pigd", period="~^late"), value="0.072")
C("9.10", "tremor −0.046, −0.073 to −0.020)", CSL, csvw(CSL, "slope", completers_min_followup="6.0", outcome="tremor", period="~^late"), value="-0.046")
C("9.10", "tremor −0.046, −0.073 to −0.020)", CSL, csvw(CSL, "boot_lo", completers_min_followup="6.0", outcome="tremor", period="~^late"), value="-0.073")
C("9.10", "tremor −0.046, −0.073 to −0.020)", CSL, csvw(CSL, "boot_hi", completers_min_followup="6.0", outcome="tremor", period="~^late"), value="-0.020")
FSC = S5 + "timeaxis/followup_slope_changes.csv"
C("9.10", "(tremor −0.054, −0.115 to +0.018", FSC, csvw(FSC, "estimate", model="followup_pw_2_5+duration", outcome="tremor", term="change_at_5y"), value="-0.054")
C("9.10", "(tremor −0.054, −0.115 to +0.018", FSC, csvw(FSC, "boot_lo", model="followup_pw_2_5+duration", outcome="tremor", term="change_at_5y"), value="-0.115")
C("9.10", "(tremor −0.054, −0.115 to +0.018", FSC, csvw(FSC, "boot_hi", model="followup_pw_2_5+duration", outcome="tremor", term="change_at_5y"), value="0.018")
C("9.10", "PIGD −0.022, −0.098 to +0.064)", FSC, csvw(FSC, "estimate", model="followup_pw_2_5+duration", outcome="pigd", term="change_at_5y"), value="-0.022")
C("9.10", "PIGD −0.022, −0.098 to +0.064)", FSC, csvw(FSC, "boot_lo", model="followup_pw_2_5+duration", outcome="pigd", term="change_at_5y"), value="-0.098")
C("9.10", "PIGD −0.022, −0.098 to +0.064)", FSC, csvw(FSC, "boot_hi", model="followup_pw_2_5+duration", outcome="pigd", term="change_at_5y"), value="0.064")
TSY = S5 + "timeaxis/treated_share_by_year.csv"
C("9.10", "rises from 0.10 at baseline to 0.95 by year five", TSY, csvw(TSY, "mean", year="0.0"), value="0.10")
C("9.10", "rises from 0.10 at baseline to 0.95 by year five", TSY, csvw(TSY, "mean", year="5.0"), value="0.95")

# ---------------------------------------------------------------- 9.11 harmonised
for name, val in [("Subscore GP, independent", "0.758"), ("Subscore GP, joint", "0.756"), ("CT-HMM, 3 latent states", "0.754"), ("CT-HMM, 5 latent states", "0.751"), ("Time-varying LMM, constant", "0.749"), ("Time-varying LMM, spline", "0.746"), ("Joint subscore LMM", "0.749"), ("Markov chain, discrete", "0.747"), ("CTMC, interval-aware", "0.745"), ("Persistence (subscore LOCF)", "0.745"), ("Patient mean subscore", "0.742"), ("Persistence (recorded label)", "0.723"), ("Marginal null", "0.629"), ("Population mean subscore", "0.625")]:
    C("9.11", val, HM, csvw(HM, "accuracy", model=name))
HC = E8 + "paired_contrasts.csv"
C("9.11", "+0.035 (+0.025 to +0.045)", HC, csvw(HC, "delta_accuracy", model="Subscore GP, independent"), value="0.035")
C("9.11", "+0.035 (+0.025 to +0.045)", HC, csvw(HC, "ci_low", model="Subscore GP, independent"), value="0.025")
C("9.11", "+0.035 (+0.025 to +0.045)", HC, csvw(HC, "ci_high", model="Subscore GP, independent"), value="0.045")
C("9.11", "+0.034 for the joint GP", HC, csvw(HC, "delta_accuracy", model="Subscore GP, joint"), value="0.034")
C("9.11", "+0.031 (+0.022 to +0.041) for the CT-HMM", HC, csvw(HC, "delta_accuracy", model="CT-HMM, 3 latent states"), value="0.031")
C("9.11", "+0.031 (+0.022 to +0.041) for the CT-HMM", HC, csvw(HC, "ci_low", model="CT-HMM, 3 latent states"), value="0.022")
C("9.11", "+0.031 (+0.022 to +0.041) for the CT-HMM", HC, csvw(HC, "ci_high", model="CT-HMM, 3 latent states"), value="0.041")
C("9.11", "+0.029 for the five-state model", HC, csvw(HC, "delta_accuracy", model="CT-HMM, 5 latent states"), value="0.029")
C("9.11", "+0.027 (+0.014 to +0.040)", HC, csvw(HC, "delta_accuracy", model="Joint subscore LMM"), value="0.027")
C("9.11", "+0.027 (+0.014 to +0.040)", HC, csvw(HC, "ci_low", model="Joint subscore LMM"), value="0.014")
C("9.11", "+0.027 (+0.014 to +0.040)", HC, csvw(HC, "ci_high", model="Joint subscore LMM"), value="0.040")
C("9.11", "+0.025 for the discrete chain", HC, csvw(HC, "delta_accuracy", model="Markov chain, discrete"), value="0.025")
C("9.11", "+0.024 (+0.010 to +0.036)", HC, csvw(HC, "delta_accuracy", model="Time-varying LMM, spline"), value="0.024")
C("9.11", "+0.024 (+0.010 to +0.036)", HC, csvw(HC, "ci_low", model="Time-varying LMM, spline"), value="0.010")
C("9.11", "+0.024 (+0.010 to +0.036)", HC, csvw(HC, "ci_high", model="Time-varying LMM, spline"), value="0.036")
C("9.11", "+0.023 (+0.014 to +0.032)", HC, csvw(HC, "delta_accuracy", model="CTMC, interval-aware"), value="0.023")
C("9.11", "+0.023 (+0.014 to +0.032)", HC, csvw(HC, "ci_low", model="CTMC, interval-aware"), value="0.014")
C("9.11", "+0.023 (+0.014 to +0.032)", HC, csvw(HC, "ci_high", model="CTMC, interval-aware"), value="0.032")
C("9.11", "(Brier 0.360, log loss 0.654)", HM, csvw(HM, "brier_multiclass", model="Subscore GP, independent"), value="0.360")
C("9.11", "(Brier 0.360, log loss 0.654)", HM, csvw(HM, "log_loss", model="Subscore GP, independent"), value="0.654")
C("9.11", "follows (0.369, 0.673)", HM, csvw(HM, "brier_multiclass", model="CT-HMM, 3 latent states"), value="0.369")
C("9.11", "follows (0.369, 0.673)", HM, csvw(HM, "log_loss", model="CT-HMM, 3 latent states"), value="0.673")
C("9.11", "persistence issues 334", HM, csvw(HM, "n_predicted_Indeterminate", model="Persistence (recorded label)"), value="334")
C("9.11", "recalls 0.193", HM, csvw(HM, "recall_Indeterminate", model="Persistence (recorded label)"), value="0.193")
C("9.11", "macro-F1 (0.556)", HM, csvw(HM, "macro_f1", model="Persistence (recorded label)"), value="0.556")
C("9.11", "105 net cases", HC, ex("int(df.loc[df.model=='CT-HMM, 3 latent states','n_better'].iloc[0]-df.loc[df.model=='CT-HMM, 3 latent states','n_worse'].iloc[0])", HC), value="105")
C("9.11", "99 (94.3%)", MD, csvw(MD, "cases_gained", model="CT-HMM, 3 latent states", current_label="Indeterminate"), value="99")
C("9.11", "from the 334 transitions", MD, csvw(MD, "n", model="CT-HMM, 3 latent states", current_label="Indeterminate"), value="334")
C("9.11", "right 19.2% of the time and the CT-HMM 48.8%", MD, csvw(MD, "baseline_accuracy", model="CT-HMM, 3 latent states", current_label="Indeterminate"), 100, value="19.2")
C("9.11", "right 19.2% of the time and the CT-HMM 48.8%", MD, csvw(MD, "model_accuracy", model="CT-HMM, 3 latent states", current_label="Indeterminate"), 100, value="48.8")
C("9.11", "on the 2,261 transitions from TD", MD, csvw(MD, "n", model="CT-HMM, 3 latent states", current_label="TD"), value="2261")
C("9.11", "identical predictions 98.4% of the time", MD, csvw(MD, "identical_predictions", model="CT-HMM, 3 latent states", current_label="TD"), 100, value="98.4")
C("9.11", "the CT-HMM gains 6 cases", MD, csvw(MD, "cases_gained", model="CT-HMM, 3 latent states", current_label="TD"), value="6")
C("9.11", "on the 766 transitions from PIGD", MD, csvw(MD, "n", model="CT-HMM, 3 latent states", current_label="PIGD"), value="766")
C("9.11", "identical 96.9% of the time", MD, csvw(MD, "identical_predictions", model="CT-HMM, 3 latent states", current_label="PIGD"), 100, value="96.9")
C("9.11", "gains 9, 100 and 8 cases", MD, csvw(MD, "cases_gained", model="Subscore GP, independent", current_label="TD"), value="9")
C("9.11", "gains 9, 100 and 8 cases", MD, csvw(MD, "cases_gained", model="Subscore GP, independent", current_label="Indeterminate"), value="100")
C("9.11", "gains 9, 100 and 8 cases", MD, csvw(MD, "cases_gained", model="Subscore GP, independent", current_label="PIGD"), value="8")
C("9.11", "(85.5% from Indeterminate)", MD, csvw(MD, "share_of_total_gain", model="Subscore GP, independent", current_label="Indeterminate"), 100, value="85.5")
C("9.11", "the discrete chain 0, 83 and 0", MD, csvw(MD, "cases_gained", model="Markov chain, discrete", current_label="Indeterminate"), value="83")
C("9.11", "maximum accuracy difference 0.0015", E8 + "monte_carlo_error.csv", ex("df['abs_difference'].max()", E8 + "monte_carlo_error.csv"), value="0.0015")

# ---------------------------------------------------------------- 9.12 next visit
C("9.12", "2,535 of 3,029 correct", NM, csvw(NM, "n_correct", model=CTH), value="2535")
C("9.12", "two-class Brier 0.121 and log loss 0.396", NM, csvw(NM, "brier_two_class", model=CTH), value="0.121")
C("9.12", "two-class Brier 0.121 and log loss 0.396", NM, csvw(NM, "log_loss", model=CTH), value="0.396")
C("9.12", "(Brier 0.160; its log loss is undefined and is 4.19 when clipped)", NM, csvw(NM, "brier_two_class", model=PERS), value="0.160")
C("9.12", "(Brier 0.160; its log loss is undefined and is 4.19 when clipped)", NM, csvw(NM, "log_loss", model=PERS), value="4.19")
C("9.12", "the discrete chain 0.829", NM, csvw(NM, "accuracy", model=MKV2), value="0.829")
C("9.12", "the continuous chain 0.828", NM, csvw(NM, "accuracy", model=CTMC2), value="0.828")
C("9.12", "without the interval 0.819", NM, csvw(NM, "accuracy", model=DIR0), value="0.819")
C("9.12", "two-class prior 0.698", NM, csvw(NM, "accuracy", model=NULL2), value="0.698")
NC = NV + "contrasts.csv"
C("9.12", "+0.009 (+0.000 to +0.018)", NC, csvw(NC, "accuracy_delta", model_a=CTH, model_b=CTMC2), value="0.009")
C("9.12", "+0.009 (+0.000 to +0.018)", NC, csvw(NC, "delta_ci_patient_bootstrap_low", model_a=CTH, model_b=CTMC2), value="0.000")
C("9.12", "+0.009 (+0.000 to +0.018)", NC, csvw(NC, "delta_ci_patient_bootstrap_high", model_a=CTH, model_b=CTMC2), value="0.018")
C("9.12", "on 98 cases against 70", NC, csvw(NC, "a_right_b_wrong", model_a=CTH, model_b=CTMC2), value="98")
C("9.12", "on 98 cases against 70", NC, csvw(NC, "b_right_a_wrong", model_a=CTH, model_b=CTMC2), value="70")
C("9.12", "−0.002 (−0.004 to +0.000)", NC, csvw(NC, "accuracy_delta", model_a=CTMC2, model_b=MKV2), value="-0.002")
C("9.12", "−0.002 (−0.004 to +0.000)", NC, csvw(NC, "delta_ci_patient_bootstrap_low", model_a=CTMC2, model_b=MKV2), value="-0.004")
C("9.12", "+0.017 (+0.011 to +0.022)", NC, csvw(NC, "accuracy_delta", model_a=DIR, model_b=DIR0), value="0.017")
C("9.12", "+0.017 (+0.011 to +0.022)", NC, csvw(NC, "delta_ci_patient_bootstrap_low", model_a=DIR, model_b=DIR0), value="0.011")
C("9.12", "+0.017 (+0.011 to +0.022)", NC, csvw(NC, "delta_ci_patient_bootstrap_high", model_a=DIR, model_b=DIR0), value="0.022")
C("9.12", "+0.001 (−0.005 to +0.008)", NC, csvw(NC, "accuracy_delta", model_a=CTH, model_b=DIR), value="0.001")
C("9.12", "+0.001 (−0.005 to +0.008)", NC, csvw(NC, "delta_ci_patient_bootstrap_low", model_a=CTH, model_b=DIR), value="-0.005")
C("9.12", "+0.001 (−0.005 to +0.008)", NC, csvw(NC, "delta_ci_patient_bootstrap_high", model_a=CTH, model_b=DIR), value="0.008")
C("9.12", "+0.003 (−0.004 to +0.010)", NC, csvw(NC, "accuracy_delta", model_a=CTH, model_b=PERS), value="0.003")
C("9.12", "+0.003 (−0.004 to +0.010)", NC, csvw(NC, "delta_ci_patient_bootstrap_low", model_a=CTH, model_b=PERS), value="-0.004")
C("9.12", "+0.003 (−0.004 to +0.010)", NC, csvw(NC, "delta_ci_patient_bootstrap_high", model_a=CTH, model_b=PERS), value="0.010")
C("9.12", "+0.139 (+0.111 to +0.167)", NC, csvw(NC, "accuracy_delta", model_a=CTH, model_b=NULL2), value="0.139")
C("9.12", "+0.139 (+0.111 to +0.167)", NC, csvw(NC, "delta_ci_patient_bootstrap_low", model_a=CTH, model_b=NULL2), value="0.111")
C("9.12", "+0.139 (+0.111 to +0.167)", NC, csvw(NC, "delta_ci_patient_bootstrap_high", model_a=CTH, model_b=NULL2), value="0.167")
NL = NV + "accuracy_by_current_label.csv"
C("9.12", "is 0.881 when the current label is TD, 0.607", NL, csvw(NL, "accuracy", model=CTH, current_label="TD"), value="0.881")
C("9.12", "is 0.881 when the current label is TD, 0.607", NL, csvw(NL, "accuracy", model=CTH, current_label="Indeterminate"), value="0.607")
C("9.12", "(the chains reach 0.544 there)", NL, csvw(NL, "accuracy", model=MKV2, current_label="Indeterminate"), value="0.544")
C("9.12", "0.792 when it is PIGD", NL, csvw(NL, "accuracy", model=CTH, current_label="PIGD"), value="0.792")
NI = NV + "accuracy_by_interval.csv"
C("9.12", "0.842 at intervals up to 1.5 years and 0.808 beyond", NI, csvw(NI, "accuracy", model=CTH, interval="<= 1.5y"), value="0.842")
C("9.12", "0.842 at intervals up to 1.5 years and 0.808 beyond", NI, csvw(NI, "accuracy", model=CTH, interval="> 1.5y"), value="0.808")
C("9.12", "(the CTMC falls to 0.748 beyond 1.5 years)", NI, csvw(NI, "accuracy", model=CTMC2, interval="> 1.5y"), value="0.748")
NH = NV + "accuracy_by_history_length.csv"
C("9.12", "0.834, 0.864 and 0.827 with one, two and three prior visits", NH, csvw(NH, "accuracy", model=CTH, n_prior_visits="1"), value="0.834")
C("9.12", "0.834, 0.864 and 0.827 with one, two and three prior visits", NH, csvw(NH, "accuracy", model=CTH, n_prior_visits="2"), value="0.864")
C("9.12", "0.834, 0.864 and 0.827 with one, two and three prior visits", NH, csvw(NH, "accuracy", model=CTH, n_prior_visits="3"), value="0.827")
NPCL = NV + "per_class_ct_hmm_latent_interval_aware.csv"
C("9.12", "PIGD recall is 0.667 at precision 0.763", NPCL, csvw(NPCL, "recall", state="PIGD"), value="0.667")
C("9.12", "PIGD recall is 0.667 at precision 0.763", NPCL, csvw(NPCL, "precision", state="PIGD"), value="0.763")
NCF = NV + "confusion_ct_hmm_latent_interval_aware.csv"
C("9.12", "1,925 TD and 610 PIGD correct, 304 PIGD called TD, 190 TD called PIGD", NCF, csvw(NCF, "TD", predicted_state="TD"), value="1925")
C("9.12", "1,925 TD and 610 PIGD correct, 304 PIGD called TD, 190 TD called PIGD", NCF, csvw(NCF, "PIGD", predicted_state="PIGD"), value="610")
C("9.12", "1,925 TD and 610 PIGD correct, 304 PIGD called TD, 190 TD called PIGD", NCF, csvw(NCF, "PIGD", predicted_state="TD"), value="304")
C("9.12", "1,925 TD and 610 PIGD correct, 304 PIGD called TD, 190 TD called PIGD", NCF, csvw(NCF, "TD", predicted_state="PIGD"), value="190")
C("9.12", "calibration gap is 0.0065", NV + "report.txt", txt(NV + "report.txt", r"Mean absolute calibration gap: ([\d.]+)"), value="0.0065")
NIR = NV + "indeterminate_report.csv"
C("9.12", "probability of PIGD was 0.371", NIR, csvw(NIR, "mean_P_PIGD", model=CTH), value="0.371")
C("9.12", "raw Indeterminate mass 0.124", NIR, csvw(NIR, "mean_p_Indeterminate_raw", model=CTH), value="0.124")
C("9.12", "called 111 of them PIGD and 221 TD", NIR, csvw(NIR, "would_have_said_PIGD", model=CTH), value="111")
C("9.12", "called 111 of them PIGD and 221 TD", NIR, csvw(NIR, "would_have_said_TD", model=CTH), value="221")
C("9.12", "worst projected gradient 0.025", NV + "convergence.csv", csvw(NV + "convergence.csv", "max_proj_grad", model=CTH), value="0.025")

# ---------------------------------------------------------------- 9.13 latent clinical states
C6M = CS6 + "model_comparison.csv"; C8M = CS8 + "model_comparison.csv"
C("9.13", "0.726 accuracy (0.696 to 0.753)", C6M, csvw(C6M, "accuracy", model=CSM), value="0.726")
C("9.13", "0.726 accuracy (0.696 to 0.753)", C6M, csvw(C6M, "accuracy_ci_patient_bootstrap_low", model=CSM), value="0.696")
C("9.13", "0.726 accuracy (0.696 to 0.753)", C6M, csvw(C6M, "accuracy_ci_patient_bootstrap_high", model=CSM), value="0.753")
C("9.13", "eight-state model 0.734 (0.706 to 0.761)", C8M, csvw(C8M, "accuracy", model=CSM), value="0.734")
C("9.13", "eight-state model 0.734 (0.706 to 0.761)", C8M, csvw(C8M, "accuracy_ci_patient_bootstrap_low", model=CSM), value="0.706")
C("9.13", "eight-state model 0.734 (0.706 to 0.761)", C8M, csvw(C8M, "accuracy_ci_patient_bootstrap_high", model=CSM), value="0.761")
C("9.13", "against 0.838 (0.818 to 0.857)", C6M, csvw(C6M, "accuracy", model=SHM), value="0.838")
C("9.13", "against 0.838 (0.818 to 0.857)", C6M, csvw(C6M, "accuracy_ci_patient_bootstrap_low", model=SHM), value="0.818")
C("9.13", "against 0.838 (0.818 to 0.857)", C6M, csvw(C6M, "accuracy_ci_patient_bootstrap_high", model=SHM), value="0.857")
C("9.13", "and 0.835 for persistence", C6M, csvw(C6M, "accuracy", model=PERS), value="0.835")
C6C = CS6 + "contrasts.csv"; C8C = CS8 + "contrasts.csv"
C("9.13", "−0.113 (−0.144 to −0.081)", C6C, csvw(C6C, "accuracy_delta", model_a=CSM, model_b=SHM), value="-0.113")
C("9.13", "−0.113 (−0.144 to −0.081)", C6C, csvw(C6C, "delta_ci_patient_bootstrap_low", model_a=CSM, model_b=SHM), value="-0.144")
C("9.13", "−0.113 (−0.144 to −0.081)", C6C, csvw(C6C, "delta_ci_patient_bootstrap_high", model_a=CSM, model_b=SHM), value="-0.081")
C("9.13", "−0.105 (−0.136 to −0.076)", C8C, csvw(C8C, "accuracy_delta", model_a=CSM, model_b=SHM), value="-0.105")
C("9.13", "−0.105 (−0.136 to −0.076)", C8C, csvw(C8C, "delta_ci_patient_bootstrap_low", model_a=CSM, model_b=SHM), value="-0.136")
C("9.13", "−0.105 (−0.136 to −0.076)", C8C, csvw(C8C, "delta_ci_patient_bootstrap_high", model_a=CSM, model_b=SHM), value="-0.076")
C6L = CS6 + "accuracy_by_current_label.csv"; C8L = CS8 + "accuracy_by_current_label.csv"
C("9.13", "(accuracy 0.397 and 0.437 against 0.787)", C6L, csvw(C6L, "accuracy", model=CSM, current_label="PIGD"), value="0.397")
C("9.13", "(accuracy 0.397 and 0.437 against 0.787)", C8L, csvw(C8L, "accuracy", model=CSM, current_label="PIGD"), value="0.437")
C("9.13", "(accuracy 0.397 and 0.437 against 0.787)", C6L, csvw(C6L, "accuracy", model=SHM, current_label="PIGD"), value="0.787")
C("9.13", "(0.218 and 0.233 against 0.659)", CS6 + "per_class_clinical_state_ct_hmm_10_measurements.csv", csvw(CS6 + "per_class_clinical_state_ct_hmm_10_measurements.csv", "recall", state="PIGD"), value="0.218")
C("9.13", "(0.218 and 0.233 against 0.659)", CS8 + "per_class_clinical_state_ct_hmm_10_measurements.csv", csvw(CS8 + "per_class_clinical_state_ct_hmm_10_measurements.csv", "recall", state="PIGD"), value="0.233")
C("9.13", "(0.218 and 0.233 against 0.659)", CS6 + "per_class_subtype_history_ct_hmm_labels_only.csv", csvw(CS6 + "per_class_subtype_history_ct_hmm_labels_only.csv", "recall", state="PIGD"), value="0.659")
CDM = CSD + "model_comparison.csv"; CDC = CSD + "contrasts.csv"
C("9.13", "reaches 0.831 (0.809 to 0.852)", CDM, csvw(CDM, "accuracy", model=DL10), value="0.831")
C("9.13", "reaches 0.831 (0.809 to 0.852)", CDM, csvw(CDM, "accuracy_ci_patient_bootstrap_low", model=DL10), value="0.809")
C("9.13", "reaches 0.831 (0.809 to 0.852)", CDM, csvw(CDM, "accuracy_ci_patient_bootstrap_high", model=DL10), value="0.852")
C("9.13", "best Brier (0.116) and log loss (0.365)", CDM, csvw(CDM, "brier_two_class", model=DL10), value="0.116")
C("9.13", "best Brier (0.116) and log loss (0.365)", CDM, csvw(CDM, "log_loss", model=DL10), value="0.365")
C("9.13", "+0.105 (+0.076 to +0.134) over the latent model", CDC, csvw(CDC, "accuracy_delta", model_a=DL10, model_b=CSM), value="0.105")
C("9.13", "+0.105 (+0.076 to +0.134) over the latent model", CDC, csvw(CDC, "delta_ci_patient_bootstrap_low", model_a=DL10, model_b=CSM), value="0.076")
C("9.13", "+0.105 (+0.076 to +0.134) over the latent model", CDC, csvw(CDC, "delta_ci_patient_bootstrap_high", model_a=DL10, model_b=CSM), value="0.134")
C("9.13", "−0.008 (−0.020 to +0.005) against the subtype-history CT-HMM", CDC, csvw(CDC, "accuracy_delta", model_a=DL10, model_b=SHM), value="-0.008")
C("9.13", "−0.008 (−0.020 to +0.005) against the subtype-history CT-HMM", CDC, csvw(CDC, "delta_ci_patient_bootstrap_low", model_a=DL10, model_b=SHM), value="-0.020")
C("9.13", "−0.008 (−0.020 to +0.005) against the subtype-history CT-HMM", CDC, csvw(CDC, "delta_ci_patient_bootstrap_high", model_a=DL10, model_b=SHM), value="0.005")
C("9.13", "changes nothing (−0.001, −0.011 to +0.009)", CDC, csvw(CDC, "accuracy_delta", model_a=DL2, model_b=DL10), value="-0.001")
C("9.13", "changes nothing (−0.001, −0.011 to +0.009)", CDC, csvw(CDC, "delta_ci_patient_bootstrap_low", model_a=DL2, model_b=DL10), value="-0.011")
C("9.13", "changes nothing (−0.001, −0.011 to +0.009)", CDC, csvw(CDC, "delta_ci_patient_bootstrap_high", model_a=DL2, model_b=DL10), value="0.009")
C("9.13", "2,091 scored two-class cases", C6M, csvw(C6M, "n_scored", model=CSM), value="2091")
C("9.13", "(627 patients with cases)", C6M, csvw(C6M, "n_patients", model=CSM), value="627")
C("9.13", "(33% of the variance in these measurements, 52% for PIGD)", RPDOC, txt(RPDOC, r"\*\*(\d+)% of the variance in these measurements is within-patient\*\*"), value="33")
C("9.13", "(33% of the variance in these measurements, 52% for PIGD)", RPDOC, txt(RPDOC, r"within-patient\*\* \(PIGD: (\d+)%\)"), value="52")

# ---------------------------------------------------------------- 9.14 profile
AM = A2 + "model_comparison.csv"
C("9.14", "reaches −1.045 log-density per domain observation (−1.071 to −1.022)", AM, csvw(AM, "loglik_per_domain_observation", model=PROF), value="-1.045")
C("9.14", "reaches −1.045 log-density per domain observation (−1.071 to −1.022)", AM, csvw(AM, "loglik_ci_low", model=PROF), value="-1.071")
C("9.14", "reaches −1.045 log-density per domain observation (−1.071 to −1.022)", AM, csvw(AM, "loglik_ci_high", model=PROF), value="-1.022")
C("9.14", "against −1.049 for the level-only variant", AM, csvw(AM, "loglik_per_domain_observation", model=LO), value="-1.049")
C("9.14", "−1.051 without medication", AM, csvw(AM, "loglik_per_domain_observation", model=NMED), value="-1.051")
C("9.14", "−1.119 for LOCF", AM, csvw(AM, "loglik_per_domain_observation", model=LOCF), value="-1.119")
C("9.14", "−1.434 for the population trajectory", AM, csvw(AM, "loglik_per_domain_observation", model=POP), value="-1.434")
ACn = A2 + "contrasts.csv"
C("9.14", "+0.073 (+0.066 to +0.081) over LOCF", ACn, csvw(ACn, "loglik_delta", model_a=PROF, model_b=LOCF), value="0.073")
C("9.14", "+0.073 (+0.066 to +0.081) over LOCF", ACn, csvw(ACn, "delta_ci_low", model_a=PROF, model_b=LOCF), value="0.066")
C("9.14", "+0.073 (+0.066 to +0.081) over LOCF", ACn, csvw(ACn, "delta_ci_high", model_a=PROF, model_b=LOCF), value="0.081")
C("9.14", "better in 75.1% of predictions", ACn, csvw(ACn, "share_a_better", model_a=PROF, model_b=LOCF), 100, value="75.1")
C("9.14", "+0.389 (+0.363 to +0.417)", ACn, csvw(ACn, "loglik_delta", model_a=PROF, model_b=POP), value="0.389")
C("9.14", "+0.389 (+0.363 to +0.417)", ACn, csvw(ACn, "delta_ci_low", model_a=PROF, model_b=POP), value="0.363")
C("9.14", "+0.389 (+0.363 to +0.417)", ACn, csvw(ACn, "delta_ci_high", model_a=PROF, model_b=POP), value="0.417")
C("9.14", "+0.004 (+0.002 to +0.006) for modelling the rate", ACn, csvw(ACn, "loglik_delta", model_a=PROF, model_b=LO), value="0.004")
C("9.14", "+0.004 (+0.002 to +0.006) for modelling the rate", ACn, csvw(ACn, "delta_ci_low", model_a=PROF, model_b=LO), value="0.002")
C("9.14", "+0.004 (+0.002 to +0.006) for modelling the rate", ACn, csvw(ACn, "delta_ci_high", model_a=PROF, model_b=LO), value="0.006")
C("9.14", "+0.006 (+0.004 to +0.008) for medication", ACn, csvw(ACn, "loglik_delta", model_a=PROF, model_b=NMED), value="0.006")
C("9.14", "+0.006 (+0.004 to +0.008) for medication", ACn, csvw(ACn, "delta_ci_low", model_a=PROF, model_b=NMED), value="0.004")
C("9.14", "+0.006 (+0.004 to +0.008) for medication", ACn, csvw(ACn, "delta_ci_high", model_a=PROF, model_b=NMED), value="0.008")
C("9.14", "Standardised RMSE is 0.70 and 95% coverage 0.941", AM, csvw(AM, "rmse_standardised", model=PROF), value="0.70")
C("9.14", "Standardised RMSE is 0.70 and 95% coverage 0.941", AM, csvw(AM, "coverage_95", model=PROF), value="0.941")
RS = A2 + "rate_spread.csv"
C("9.14", "standard deviation of 0.090 per year", RS, csvw(RS, "sd_rate_per_year", domain="pigd"), value="0.090")
C("9.14", "Part II rate 0.375 per year", RS, csvw(RS, "sd_rate_per_year", domain="updrs2"), value="0.375")
ACV = A2 + "convergence.csv"
C("9.14", "sixteen terminated at the 600-iteration limit", ACV, ex("int(df.loc[df.model=='%s','n_folds'].iloc[0]-df.loc[df.model=='%s','n_scipy_success'].iloc[0])" % (PROF, PROF), ACV), value="16")
C("9.14", "projected gradient of up to 1.75", ACV, csvw(ACV, "max_proj_grad", model=PROF), value="1.75")
C("9.14", "reaches 0.840 (0.825 to 0.857; Brier 0.118, log loss 0.377)", AD, csvw(AD, "accuracy_ci_patient_bootstrap_low", model=PROF), value="0.825")
C("9.14", "reaches 0.840 (0.825 to 0.857; Brier 0.118, log loss 0.377)", AD, csvw(AD, "accuracy_ci_patient_bootstrap_high", model=PROF), value="0.857")
C("9.14", "reaches 0.840 (0.825 to 0.857; Brier 0.118, log loss 0.377)", AD, csvw(AD, "brier_two_class", model=PROF), value="0.118")
C("9.14", "reaches 0.840 (0.825 to 0.857; Brier 0.118, log loss 0.377)", AD, csvw(AD, "log_loss", model=PROF), value="0.377")
C("9.14", "level-only profile 0.844 (0.829 to 0.860)", AD, csvw(AD, "accuracy", model=LO), value="0.844")
C("9.14", "level-only profile 0.844 (0.829 to 0.860)", AD, csvw(AD, "accuracy_ci_patient_bootstrap_low", model=LO), value="0.829")
C("9.14", "level-only profile 0.844 (0.829 to 0.860)", AD, csvw(AD, "accuracy_ci_patient_bootstrap_high", model=LO), value="0.860")
C("9.14", "the CT-HMM 0.837 (0.820 to 0.854)", AD, csvw(AD, "accuracy", model="~^Subtype-history: CT-HMM"), value="0.837")
C("9.14", "the CT-HMM 0.837 (0.820 to 0.854)", AD, csvw(AD, "accuracy_ci_patient_bootstrap_low", model="~^Subtype-history: CT-HMM"), value="0.820")
C("9.14", "the CT-HMM 0.837 (0.820 to 0.854)", AD, csvw(AD, "accuracy_ci_patient_bootstrap_high", model="~^Subtype-history: CT-HMM"), value="0.854")
C("9.14", "the discrete chain 0.830", AD, csvw(AD, "accuracy", model="~^Subtype-history: Markov"), value="0.830")
C("9.14", "LOCF 0.825", AD, csvw(AD, "accuracy", model=LOCF), value="0.825")
C("9.14", "population trajectory 0.691", AD, csvw(AD, "accuracy", model=POP), value="0.691")
ADC = A3 + "derived_label_contrasts.csv"
FPvs = "~^Profile: level \\+ rate \\+ medication vs"
C("9.14", "+0.003 (−0.005 to +0.011) for the full profile", ADC, csvw(ADC, "accuracy_delta", contrast=FPvs), value="0.003")
C("9.14", "+0.003 (−0.005 to +0.011) for the full profile", ADC, csvw(ADC, "delta_ci_patient_bootstrap_low", contrast=FPvs), value="-0.005")
C("9.14", "+0.003 (−0.005 to +0.011) for the full profile", ADC, csvw(ADC, "delta_ci_patient_bootstrap_high", contrast=FPvs), value="0.011")
C("9.14", "+0.007 (−0.001 to +0.015) for the level-only profile", ADC, csvw(ADC, "accuracy_delta", contrast="~^Profile: level only"), value="0.007")
C("9.14", "+0.007 (−0.001 to +0.015) for the level-only profile", ADC, csvw(ADC, "delta_ci_patient_bootstrap_low", contrast="~^Profile: level only"), value="-0.001")
C("9.14", "+0.007 (−0.001 to +0.015) for the level-only profile", ADC, csvw(ADC, "delta_ci_patient_bootstrap_high", contrast="~^Profile: level only"), value="0.015")
C("9.14", "on 81 cases against 71", ADC, csvw(ADC, "a_right_b_wrong", contrast=FPvs), value="81")
C("9.14", "on 81 cases against 71", ADC, csvw(ADC, "b_right_a_wrong", contrast=FPvs), value="71")
H3 = A2 + "h3_component_vs_subtype.csv"
C("9.14", "AUC 0.685 (Part II level) against 0.579", H3, csvw(H3, "auc_a", outcome="hy_ge_3", kind="level"), value="0.685")
C("9.14", "AUC 0.685 (Part II level) against 0.579", H3, csvw(H3, "auc_b", outcome="hy_ge_3", kind="level"), value="0.579")
C("9.14", "(+0.105, +0.054 to +0.156, 761 patients, 127 events)", H3, csvw(H3, "delta", outcome="hy_ge_3", kind="level"), value="0.105")
C("9.14", "(+0.105, +0.054 to +0.156, 761 patients, 127 events)", H3, csvw(H3, "delta_ci_low", outcome="hy_ge_3", kind="level"), value="0.054")
C("9.14", "(+0.105, +0.054 to +0.156, 761 patients, 127 events)", H3, csvw(H3, "delta_ci_high", outcome="hy_ge_3", kind="level"), value="0.156")
C("9.14", "(+0.105, +0.054 to +0.156, 761 patients, 127 events)", H3, csvw(H3, "n", outcome="hy_ge_3", kind="level"), value="761")
C("9.14", "(+0.105, +0.054 to +0.156, 761 patients, 127 events)", H3, csvw(H3, "n_positive", outcome="hy_ge_3", kind="level"), value="127")
C("9.14", "at 0.710 (MoCA level) against 0.549", H3, csvw(H3, "auc_a", outcome="mci", kind="level"), value="0.710")
C("9.14", "at 0.710 (MoCA level) against 0.549", H3, csvw(H3, "auc_b", outcome="mci", kind="level"), value="0.549")
C("9.14", "(+0.160, +0.109 to +0.213; 818 patients, 298 events)", H3, csvw(H3, "delta", outcome="mci", kind="level"), value="0.160")
C("9.14", "(+0.160, +0.109 to +0.213; 818 patients, 298 events)", H3, csvw(H3, "delta_ci_low", outcome="mci", kind="level"), value="0.109")
C("9.14", "(+0.160, +0.109 to +0.213; 818 patients, 298 events)", H3, csvw(H3, "delta_ci_high", outcome="mci", kind="level"), value="0.213")
C("9.14", "(+0.160, +0.109 to +0.213; 818 patients, 298 events)", H3, csvw(H3, "n", outcome="mci", kind="level"), value="818")
C("9.14", "(+0.160, +0.109 to +0.213; 818 patients, 298 events)", H3, csvw(H3, "n_positive", outcome="mci", kind="level"), value="298")
C("9.14", "(0.531 against 0.504, +0.027, −0.039 to +0.094)", H3, csvw(H3, "auc_a", outcome="motor_complications", kind="level"), value="0.531")
C("9.14", "(0.531 against 0.504, +0.027, −0.039 to +0.094)", H3, csvw(H3, "auc_b", outcome="motor_complications", kind="level"), value="0.504")
C("9.14", "(0.531 against 0.504, +0.027, −0.039 to +0.094)", H3, csvw(H3, "delta", outcome="motor_complications", kind="level"), value="0.027")
C("9.14", "(0.531 against 0.504, +0.027, −0.039 to +0.094)", H3, csvw(H3, "delta_ci_low", outcome="motor_complications", kind="level"), value="-0.039")
C("9.14", "(0.531 against 0.504, +0.027, −0.039 to +0.094)", H3, csvw(H3, "delta_ci_high", outcome="motor_complications", kind="level"), value="0.094")
C("9.14", "discriminated at 0.678 against 0.558", H3, csvw(H3, "auc_a", outcome="cognitive_dx", kind="level"), value="0.678")
C("9.14", "discriminated at 0.678 against 0.558", H3, csvw(H3, "auc_b", outcome="cognitive_dx", kind="level"), value="0.558")
RD = A2 + "representation_dimensionality.csv"
C("9.14", "first component carrying 42% (levels) and 37% (rates)", RD, csvw(RD, "pc1_share", which="level"), 100, value="42")
C("9.14", "first component carrying 42% (levels) and 37% (rates)", RD, csvw(RD, "pc1_share", which="rate"), 100, value="37")
C("9.14", "effective ranks of 4.3 and 5.1", RD, csvw(RD, "effective_rank_participation", which="level"), value="4.3")
C("9.14", "effective ranks of 4.3 and 5.1", RD, csvw(RD, "effective_rank_participation", which="rate"), value="5.1")
C("9.14", "seven components are needed", RD, csvw(RD, "n_components_for_90pct", which="level"), value="7")

# ---------------------------------------------------------------- 9.15 selection / AUC
AB = MSC + "aic_bic.csv"
C("9.15", "(14 parameters, AIC 5,966)", AB, csvw(AB, "aic", group="CT-HMM (alone)"), value="5966")
C("9.15", "by 224 AIC units and 186 BIC units", AB, csvw(AB, "delta_aic_within_group", model="Markov + initial state"), value="224")
C("9.15", "by 224 AIC units and 186 BIC units", AB, csvw(AB, "delta_bic_within_group", model="Markov + initial state"), value="186")
C("9.15", "continuous chain by 250 AIC units", AB, csvw(AB, "delta_aic_within_group", model="CTMC + initial state"), value="250")
C("9.15", "continuous one by 25 units", AB, csvw(AB, "delta_aic_within_group", group="B: observed-label chains", model="CTMC (interval-aware)"), value="25")
C("9.15", "level-only variant by 221 AIC and 67 BIC units", AB, csvw(AB, "delta_aic_within_group", model=LO), value="221")
C("9.15", "level-only variant by 221 AIC and 67 BIC units", AB, csvw(AB, "delta_bic_within_group", model=LO), value="67")
C("9.15", "no-medication variant by 344 and 267", AB, csvw(AB, "delta_aic_within_group", model=NMED), value="344")
C("9.15", "no-medication variant by 344 and 267", AB, csvw(AB, "delta_bic_within_group", model=NMED), value="267")
C("9.15", "AUC is 0.890 (0.875 to 0.905)", RAM, csvw(RAM, "ci_low", model="~[Pp]rofile"), value="0.875")
C("9.15", "AUC is 0.890 (0.875 to 0.905)", RAM, csvw(RAM, "ci_high", model="~[Pp]rofile"), value="0.905")
C("9.15", "0.858 (0.841 to 0.876) for the CT-HMM", RAM, csvw(RAM, "ci_low", model="CT-HMM"), value="0.841")
C("9.15", "0.858 (0.841 to 0.876) for the CT-HMM", RAM, csvw(RAM, "ci_high", model="CT-HMM"), value="0.876")
RCo = ROC + "auc_contrast.csv"
C("9.15", "+0.032 (+0.023 to +0.042)", RCo, csvi(RCo, "delta", 0), value="0.032")
C("9.15", "+0.032 (+0.023 to +0.042)", RCo, csvi(RCo, "ci_low", 0), value="0.023")
C("9.15", "+0.032 (+0.023 to +0.042)", RCo, csvi(RCo, "ci_high", 0), value="0.042")
C("9.15", "rule, scores 0.793", RAM, csvw(RAM, "auc", model="~[Pp]ersist"), value="0.793")
RAL = ROC + "auc_by_current_label.csv"
C("9.15", "(TD 0.780 against 0.668", RAL, csvw(RAL, "profile_auc", current_label="TD"), value="0.780")
C("9.15", "(TD 0.780 against 0.668", RAL, csvw(RAL, "ct_hmm_auc", current_label="TD"), value="0.668")
C("9.15", "Indeterminate 0.679 against 0.631", RAL, csvw(RAL, "profile_auc", current_label="Indeterminate"), value="0.679")
C("9.15", "Indeterminate 0.679 against 0.631", RAL, csvw(RAL, "ct_hmm_auc", current_label="Indeterminate"), value="0.631")
C("9.15", "PIGD 0.744 against 0.665", RAL, csvw(RAL, "profile_auc", current_label="PIGD"), value="0.744")
C("9.15", "PIGD 0.744 against 0.665", RAL, csvw(RAL, "ct_hmm_auc", current_label="PIGD"), value="0.665")

# ---------------------------------------------------------------- 9.16 Indeterminate boundary
BA = IB + "boundary_agreement.csv"
C("9.16", "at 0.488 of Indeterminate visits against 0.462", BA, csvw(BA, "agreement_with_nearer_boundary", model="CT-HMM"), value="0.488")
C("9.16", "at 0.488 of Indeterminate visits against 0.462", BA, csvw(BA, "expected_under_independence", model="CT-HMM"), value="0.462")
C("9.16", "excess of +0.026 (−0.029 to +0.078)", BA, csvw(BA, "excess_over_null", model="CT-HMM"), value="0.026")
C("9.16", "excess of +0.026 (−0.029 to +0.078)", BA, csvw(BA, "excess_ci_low", model="CT-HMM"), value="-0.029")
C("9.16", "excess of +0.026 (−0.029 to +0.078)", BA, csvw(BA, "excess_ci_high", model="CT-HMM"), value="0.078")
C("9.16", "profile at 0.502 against 0.471, +0.031 (−0.022 to +0.088)", BA, csvw(BA, "agreement_with_nearer_boundary", model=RCP), value="0.502")
C("9.16", "profile at 0.502 against 0.471, +0.031 (−0.022 to +0.088)", BA, csvw(BA, "expected_under_independence", model=RCP), value="0.471")
C("9.16", "profile at 0.502 against 0.471, +0.031 (−0.022 to +0.088)", BA, csvw(BA, "excess_over_null", model=RCP), value="0.031")
C("9.16", "profile at 0.502 against 0.471, +0.031 (−0.022 to +0.088)", BA, csvw(BA, "excess_ci_low", model=RCP), value="-0.022")
C("9.16", "profile at 0.502 against 0.471, +0.031 (−0.022 to +0.088)", BA, csvw(BA, "excess_ci_high", model=RCP), value="0.088")
SCB = IB + "sensitivity_close_to_boundary.csv"
C("9.16", "excesses are +0.038 (−0.029 to +0.103) and +0.062 (−0.002 to +0.127)", SCB, csvw(SCB, "excess", model="CT-HMM"), value="0.038")
C("9.16", "excesses are +0.038 (−0.029 to +0.103) and +0.062 (−0.002 to +0.127)", SCB, csvw(SCB, "excess_ci_low", model="CT-HMM"), value="-0.029")
C("9.16", "excesses are +0.038 (−0.029 to +0.103) and +0.062 (−0.002 to +0.127)", SCB, csvw(SCB, "excess_ci_high", model="CT-HMM"), value="0.103")
C("9.16", "excesses are +0.038 (−0.029 to +0.103) and +0.062 (−0.002 to +0.127)", SCB, csvw(SCB, "excess", model=RCP), value="0.062")
C("9.16", "excesses are +0.038 (−0.029 to +0.103) and +0.062 (−0.002 to +0.127)", SCB, csvw(SCB, "excess_ci_low", model=RCP), value="-0.002")
C("9.16", "excesses are +0.038 (−0.029 to +0.103) and +0.062 (−0.002 to +0.127)", SCB, csvw(SCB, "excess_ci_high", model=RCP), value="0.127")
C("9.16", "is 0.584 (0.516 to 0.653) and 0.575 (0.505 to 0.640)", BA, csvw(BA, "auc_P_PIGD_for_nearer_PIGD", model="CT-HMM"), value="0.584")
C("9.16", "is 0.584 (0.516 to 0.653) and 0.575 (0.505 to 0.640)", BA, csvw(BA, "auc_ci_low", model="CT-HMM"), value="0.516")
C("9.16", "is 0.584 (0.516 to 0.653) and 0.575 (0.505 to 0.640)", BA, csvw(BA, "auc_ci_high", model="CT-HMM"), value="0.653")
C("9.16", "is 0.584 (0.516 to 0.653) and 0.575 (0.505 to 0.640)", BA, csvw(BA, "auc_P_PIGD_for_nearer_PIGD", model=RCP), value="0.575")
C("9.16", "is 0.584 (0.516 to 0.653) and 0.575 (0.505 to 0.640)", BA, csvw(BA, "auc_ci_low", model=RCP), value="0.505")
C("9.16", "is 0.584 (0.516 to 0.653) and 0.575 (0.505 to 0.640)", BA, csvw(BA, "auc_ci_high", model=RCP), value="0.640")
PCI = IRS + "patient_clustered_intervals.csv"
C("9.16", "calls PIGD at 0.347 of the 0.9091 visits", PCI, csvw(PCI, "estimate", quantity="P(predict PIGD | ratio 0.9091)", model="CT-HMM"), value="0.347")
C("9.16", "TD at 0.731 of the 1.1364 visits", PCI, csvw(PCI, "estimate", quantity="P(predict TD | ratio 1.1364)", model="CT-HMM"), value="0.731")
C("9.16", "the profile at 0.389 and 0.746", PCI, csvw(PCI, "estimate", quantity="P(predict PIGD | ratio 0.9091)", model=RCP), value="0.389")
C("9.16", "the profile at 0.389 and 0.746", PCI, csvw(PCI, "estimate", quantity="P(predict TD | ratio 1.1364)", model=RCP), value="0.746")
C("9.16", "by +0.093 (+0.013 to +0.171) for the CT-HMM", PCI, csvw(PCI, "estimate", quantity="~^mean P_PIGD", model="CT-HMM"), value="0.093")
C("9.16", "by +0.093 (+0.013 to +0.171) for the CT-HMM", PCI, csvw(PCI, "ci_low", quantity="~^mean P_PIGD", model="CT-HMM"), value="0.013")
C("9.16", "by +0.093 (+0.013 to +0.171) for the CT-HMM", PCI, csvw(PCI, "ci_high", quantity="~^mean P_PIGD", model="CT-HMM"), value="0.171")
C("9.16", "+0.080 (+0.015 to +0.146) for the profile", PCI, csvw(PCI, "estimate", quantity="~^mean P_PIGD", model=RCP), value="0.080")
C("9.16", "+0.080 (+0.015 to +0.146) for the profile", PCI, csvw(PCI, "ci_low", quantity="~^mean P_PIGD", model=RCP), value="0.015")
C("9.16", "+0.080 (+0.015 to +0.146) for the profile", PCI, csvw(PCI, "ci_high", quantity="~^mean P_PIGD", model=RCP), value="0.146")
C("9.16", "is +0.079 (−0.050 to +0.205) for the CT-HMM", PCI, csvw(PCI, "estimate", quantity="~^P\\(PIGD \\| 0\\.9091\\) - P", model="CT-HMM"), value="0.079")
C("9.16", "is +0.079 (−0.050 to +0.205) for the CT-HMM", PCI, csvw(PCI, "ci_low", quantity="~^P\\(PIGD \\| 0\\.9091\\) - P", model="CT-HMM"), value="-0.050")
C("9.16", "is +0.079 (−0.050 to +0.205) for the CT-HMM", PCI, csvw(PCI, "ci_high", quantity="~^P\\(PIGD \\| 0\\.9091\\) - P", model="CT-HMM"), value="0.205")
C("9.16", "+0.135 (+0.005 to +0.260) for the profile", PCI, csvw(PCI, "estimate", quantity="~^P\\(PIGD \\| 0\\.9091\\) - P", model=RCP), value="0.135")
C("9.16", "+0.135 (+0.005 to +0.260) for the profile", PCI, csvw(PCI, "ci_low", quantity="~^P\\(PIGD \\| 0\\.9091\\) - P", model=RCP), value="0.005")
C("9.16", "+0.135 (+0.005 to +0.260) for the profile", PCI, csvw(PCI, "ci_high", quantity="~^P\\(PIGD \\| 0\\.9091\\) - P", model=RCP), value="0.260")
C("9.16", "(287 visits, 220 patients", BA, csvw(BA, "n", model="CT-HMM"), value="287")
C("9.16", "(287 visits, 220 patients", BA, csvw(BA, "n_patients", model="CT-HMM"), value="220")
C("9.16", "0.9091 (144 visits", PCI, csvw(PCI, "n_visits", quantity="P(predict PIGD | ratio 0.9091)", model="CT-HMM"), value="144")
C("9.16", "1.1364 (67 visits", PCI, csvw(PCI, "n_visits", quantity="P(predict TD | ratio 1.1364)", model="CT-HMM"), value="67")

# ---------------------------------------------------------------- 9.17 stage-wise
SMx = SW + "stage_metrics.csv"
C("9.17", "reaches 0.835 (0.810 to 0.860), the profile 0.846 (0.821 to 0.870) and persistence 0.832", SMx, csvw(SMx, "accuracy", stage="1", model="CT-HMM"), value="0.835")
C("9.17", "reaches 0.835 (0.810 to 0.860), the profile 0.846 (0.821 to 0.870) and persistence 0.832", SMx, csvw(SMx, "accuracy_lo", stage="1", model="CT-HMM"), value="0.810")
C("9.17", "reaches 0.835 (0.810 to 0.860), the profile 0.846 (0.821 to 0.870) and persistence 0.832", SMx, csvw(SMx, "accuracy_hi", stage="1", model="CT-HMM"), value="0.860")
C("9.17", "reaches 0.835 (0.810 to 0.860), the profile 0.846 (0.821 to 0.870) and persistence 0.832", SMx, csvw(SMx, "accuracy", stage="1", model=RCP), value="0.846")
C("9.17", "reaches 0.835 (0.810 to 0.860), the profile 0.846 (0.821 to 0.870) and persistence 0.832", SMx, csvw(SMx, "accuracy_lo", stage="1", model=RCP), value="0.821")
C("9.17", "reaches 0.835 (0.810 to 0.860), the profile 0.846 (0.821 to 0.870) and persistence 0.832", SMx, csvw(SMx, "accuracy_hi", stage="1", model=RCP), value="0.870")
C("9.17", "reaches 0.835 (0.810 to 0.860), the profile 0.846 (0.821 to 0.870) and persistence 0.832", SMx, csvw(SMx, "accuracy", stage="1", model="Persistence (reference)"), value="0.832")
C("9.17", "balanced accuracies of 0.729, 0.775 and 0.717", SMx, csvw(SMx, "balanced_accuracy", stage="1", model="CT-HMM"), value="0.729")
C("9.17", "balanced accuracies of 0.729, 0.775 and 0.717", SMx, csvw(SMx, "balanced_accuracy", stage="1", model=RCP), value="0.775")
C("9.17", "balanced accuracies of 0.729, 0.775 and 0.717", SMx, csvw(SMx, "balanced_accuracy", stage="1", model="Persistence (reference)"), value="0.717")
PDx = SW + "paired_differences.csv"
C("9.17", "+0.046 (+0.023 to +0.070)", PDx, csvw(PDx, "estimate", stage="1", metric="balanced_accuracy"), value="0.046")
C("9.17", "+0.046 (+0.023 to +0.070)", PDx, csvw(PDx, "ci_low", stage="1", metric="balanced_accuracy"), value="0.023")
C("9.17", "+0.046 (+0.023 to +0.070)", PDx, csvw(PDx, "ci_high", stage="1", metric="balanced_accuracy"), value="0.070")
C("9.17", "(from 0.717 at stage 1 to 0.816 at stage 9)", SMx, csvw(SMx, "balanced_accuracy", stage="9", model="Persistence (reference)"), value="0.816")
C("9.17", "rises from 23% to 48%", SMx, csvw(SMx, "pct_PIGD", stage="1", model="CT-HMM"), 100, value="23")
C("9.17", "rises from 23% to 48%", SMx, csvw(SMx, "pct_PIGD", stage="9", model="CT-HMM"), 100, value="48")
WC = SW + "within_patient_change.csv"
C("9.17", "is −0.024 (−0.075 to +0.028) for the CT-HMM", WC, csvw(WC, "mean_within_patient_change", model="CT-HMM"), value="-0.024")
C("9.17", "is −0.024 (−0.075 to +0.028) for the CT-HMM", WC, csvw(WC, "ci_low", model="CT-HMM"), value="-0.075")
C("9.17", "is −0.024 (−0.075 to +0.028) for the CT-HMM", WC, csvw(WC, "ci_high", model="CT-HMM"), value="0.028")
C("9.17", "−0.035 (−0.089 to +0.020) for the profile", WC, csvw(WC, "mean_within_patient_change", model=RCP), value="-0.035")
C("9.17", "−0.035 (−0.089 to +0.020) for the profile", WC, csvw(WC, "ci_low", model=RCP), value="-0.089")
C("9.17", "−0.035 (−0.089 to +0.020) for the profile", WC, csvw(WC, "ci_high", model=RCP), value="0.020")
C("9.17", "(225 patients)", WC, csvw(WC, "n_patients", model="CT-HMM"), value="225")
C("9.17", "835 targets", SMx, csvw(SMx, "n_targets", stage="1", model="CT-HMM"), value="835")

# ---------------------------------------------------------------- 9.18 history windows
WM = HW + "window_metrics.csv"
for w, v in [("last 1", "0.811"), ("last 2", "0.828"), ("last 3", "0.835"), ("last 5", "0.834"), ("full history", "0.832"), ("persistence", "0.827")]:
    C("9.18", v, WM, csvw(WM, "accuracy", cohort="common targets", window=w))
for w, v in [("last 1", "0.827"), ("last 2", "0.884"), ("last 3", "0.890"), ("last 5", "0.889"), ("full history", "0.888")]:
    C("9.18", "AUC is 0.827, 0.884, 0.890, 0.889 and 0.888", WM, csvw(WM, "auc", cohort="common targets", window=w), value=v)
HPc = HW + "paired_comparisons.csv"
L2F = "last 2 - full history"; L1F = "last 1 - full history"
C("9.18", "is −0.004 (−0.019 to +0.010) in accuracy", HPc, csvw(HPc, "accuracy", comparison=L2F, cohort="common targets"), value="-0.004")
C("9.18", "is −0.004 (−0.019 to +0.010) in accuracy", HPc, csvw(HPc, "accuracy_lo", comparison=L2F, cohort="common targets"), value="-0.019")
C("9.18", "is −0.004 (−0.019 to +0.010) in accuracy", HPc, csvw(HPc, "accuracy_hi", comparison=L2F, cohort="common targets"), value="0.010")
C("9.18", "−0.006 in balanced accuracy", HPc, csvw(HPc, "balanced_accuracy", comparison=L2F, cohort="common targets"), value="-0.006")
C("9.18", "−0.005 (−0.015 to +0.004) in AUC", HPc, csvw(HPc, "auc", comparison=L2F, cohort="common targets"), value="-0.005")
C("9.18", "−0.005 (−0.015 to +0.004) in AUC", HPc, csvw(HPc, "auc_lo", comparison=L2F, cohort="common targets"), value="-0.015")
C("9.18", "−0.005 (−0.015 to +0.004) in AUC", HPc, csvw(HPc, "auc_hi", comparison=L2F, cohort="common targets"), value="0.004")
C("9.18", "+0.005 in Brier and +0.011 in log loss", HPc, csvw(HPc, "brier", comparison=L2F, cohort="common targets"), value="0.005")
C("9.18", "+0.005 in Brier and +0.011 in log loss", HPc, csvw(HPc, "log_loss", comparison=L2F, cohort="common targets"), value="0.011")
C("9.18", "−0.022 (−0.042 to −0.001) in accuracy", HPc, csvw(HPc, "accuracy", comparison=L1F, cohort="common targets"), value="-0.022")
C("9.18", "−0.022 (−0.042 to −0.001) in accuracy", HPc, csvw(HPc, "accuracy_lo", comparison=L1F, cohort="common targets"), value="-0.042")
C("9.18", "−0.022 (−0.042 to −0.001) in accuracy", HPc, csvw(HPc, "accuracy_hi", comparison=L1F, cohort="common targets"), value="-0.001")
C("9.18", "−0.062 (−0.086 to −0.039) in AUC", HPc, csvw(HPc, "auc", comparison=L1F, cohort="common targets"), value="-0.062")
C("9.18", "−0.062 (−0.086 to −0.039) in AUC", HPc, csvw(HPc, "auc_lo", comparison=L1F, cohort="common targets"), value="-0.086")
C("9.18", "−0.062 (−0.086 to −0.039) in AUC", HPc, csvw(HPc, "auc_hi", comparison=L1F, cohort="common targets"), value="-0.039")
C("9.18", "734 targets of 228 patients", WM, csvw(WM, "n_targets", cohort="common targets", window="full history"), value="734")
C("9.18", "734 targets of 228 patients", WM, csvw(WM, "n_patients", cohort="common targets", window="full history"), value="228")
C("9.18", "934 folds and 11,644 cases", HW + "tamper_check.csv", ex("int(df['n_cases'].sum())", HW + "tamper_check.csv"), value="11644")
PP = PHW + "profile_paired_comparisons.csv"
C("9.18", "last two minus full is −0.001 (−0.010 to +0.009) in accuracy", PP, csvw(PP, "accuracy", comparison=L2F), value="-0.001")
C("9.18", "last two minus full is −0.001 (−0.010 to +0.009) in accuracy", PP, csvw(PP, "accuracy_lo", comparison=L2F), value="-0.010")
C("9.18", "last two minus full is −0.001 (−0.010 to +0.009) in accuracy", PP, csvw(PP, "accuracy_hi", comparison=L2F), value="0.009")
C("9.18", "−0.003 (−0.007 to +0.002) in AUC", PP, csvw(PP, "auc", comparison=L2F), value="-0.003")
C("9.18", "−0.003 (−0.007 to +0.002) in AUC", PP, csvw(PP, "auc_lo", comparison=L2F), value="-0.007")
C("9.18", "−0.003 (−0.007 to +0.002) in AUC", PP, csvw(PP, "auc_hi", comparison=L2F), value="0.002")
C("9.18", "log-loss cost (+0.010, +0.002 to +0.019)", PP, csvw(PP, "log_loss", comparison=L2F), value="0.010")
C("9.18", "log-loss cost (+0.010, +0.002 to +0.019)", PP, csvw(PP, "log_loss_lo", comparison=L2F), value="0.002")
C("9.18", "log-loss cost (+0.010, +0.002 to +0.019)", PP, csvw(PP, "log_loss_hi", comparison=L2F), value="0.019")
C("9.18", "changes accuracy by −0.001 (−0.006 to +0.003)", PP, csvw(PP, "accuracy", role="LEDD sensitivity"), value="-0.001")
C("9.18", "changes accuracy by −0.001 (−0.006 to +0.003)", PP, csvw(PP, "accuracy_lo", role="LEDD sensitivity"), value="-0.006")
C("9.18", "changes accuracy by −0.001 (−0.006 to +0.003)", PP, csvw(PP, "accuracy_hi", role="LEDD sensitivity"), value="0.003")
C("9.18", "1,489 common targets of 481 patients", PP, csvw(PP, "n_targets", comparison=L2F), value="1489")
C("9.18", "1,489 common targets of 481 patients", PP, csvw(PP, "n_patients", comparison=L2F), value="481")

# ---------------------------------------------------------------- 9.19 retraining
RPc = RT + "paired_comparisons.csv"
PRI = "~^window-trained, last 2\\s+-\\s+full-trained, full history$"
TE = "~^window-trained, last 2\\s+-\\s+full-trained, last 2$"
HE = "~^full-trained, last 2\\s+-\\s+full-trained, full history$"
VP = "~^window-trained, last 2\\s+-\\s+persistence$"
DJ = "~^non-overlap-window-trained, last 2\\s+-\\s+window-trained, last 2$"
C("9.19", "is −0.008 (−0.019 to +0.003) in accuracy", RPc, csvw(RPc, "accuracy", comparison=PRI), value="-0.008")
C("9.19", "is −0.008 (−0.019 to +0.003) in accuracy", RPc, csvw(RPc, "accuracy_lo", comparison=PRI), value="-0.019")
C("9.19", "is −0.008 (−0.019 to +0.003) in accuracy", RPc, csvw(RPc, "accuracy_hi", comparison=PRI), value="0.003")
C("9.19", "but −0.031 (−0.044 to −0.018) in balanced accuracy", RPc, csvw(RPc, "balanced_accuracy", comparison=PRI), value="-0.031")
C("9.19", "but −0.031 (−0.044 to −0.018) in balanced accuracy", RPc, csvw(RPc, "balanced_accuracy_lo", comparison=PRI), value="-0.044")
C("9.19", "but −0.031 (−0.044 to −0.018) in balanced accuracy", RPc, csvw(RPc, "balanced_accuracy_hi", comparison=PRI), value="-0.018")
C("9.19", "−0.021 (−0.030 to −0.012) in AUC", RPc, csvw(RPc, "auc", comparison=PRI), value="-0.021")
C("9.19", "−0.021 (−0.030 to −0.012) in AUC", RPc, csvw(RPc, "auc_lo", comparison=PRI), value="-0.030")
C("9.19", "−0.021 (−0.030 to −0.012) in AUC", RPc, csvw(RPc, "auc_hi", comparison=PRI), value="-0.012")
C("9.19", "+0.010 (+0.005 to +0.015) in Brier", RPc, csvw(RPc, "brier", comparison=PRI), value="0.010")
C("9.19", "+0.010 (+0.005 to +0.015) in Brier", RPc, csvw(RPc, "brier_lo", comparison=PRI), value="0.005")
C("9.19", "+0.010 (+0.005 to +0.015) in Brier", RPc, csvw(RPc, "brier_hi", comparison=PRI), value="0.015")
C("9.19", "+0.040 (+0.022 to +0.058) in log loss", RPc, csvw(RPc, "log_loss", comparison=PRI), value="0.040")
C("9.19", "+0.040 (+0.022 to +0.058) in log loss", RPc, csvw(RPc, "log_loss_lo", comparison=PRI), value="0.022")
C("9.19", "+0.040 (+0.022 to +0.058) in log loss", RPc, csvw(RPc, "log_loss_hi", comparison=PRI), value="0.058")
C("9.19", "is −0.025 (−0.035 to −0.015) in balanced accuracy", RPc, csvw(RPc, "balanced_accuracy", comparison=TE), value="-0.025")
C("9.19", "is −0.025 (−0.035 to −0.015) in balanced accuracy", RPc, csvw(RPc, "balanced_accuracy_lo", comparison=TE), value="-0.035")
C("9.19", "is −0.025 (−0.035 to −0.015) in balanced accuracy", RPc, csvw(RPc, "balanced_accuracy_hi", comparison=TE), value="-0.015")
C("9.19", "is −0.006 (−0.015 to +0.002)", RPc, csvw(RPc, "balanced_accuracy", comparison=HE), value="-0.006")
C("9.19", "is −0.006 (−0.015 to +0.002)", RPc, csvw(RPc, "balanced_accuracy_lo", comparison=HE), value="-0.015")
C("9.19", "is −0.006 (−0.015 to +0.002)", RPc, csvw(RPc, "balanced_accuracy_hi", comparison=HE), value="0.002")
C("9.19", "(−0.016, −0.029 to −0.003)", RPc, csvw(RPc, "balanced_accuracy", comparison=VP), value="-0.016")
C("9.19", "(−0.016, −0.029 to −0.003)", RPc, csvw(RPc, "balanced_accuracy_lo", comparison=VP), value="-0.029")
C("9.19", "(−0.016, −0.029 to −0.003)", RPc, csvw(RPc, "balanced_accuracy_hi", comparison=VP), value="-0.003")
RCM = RT + "condition_metrics.csv"
C("9.19", "(0.608 against 0.706", RCM, csvw(RCM, "sensitivity_PIGD", cohort="common targets", condition="window-trained, last 2"), value="0.608")
C("9.19", "(0.608 against 0.706", RCM, csvw(RCM, "sensitivity_PIGD", cohort="common targets", condition="full-trained, full history"), value="0.706")
C("9.19", "(+0.016, +0.009 to +0.022 against overlapping windows)", RPc, csvw(RPc, "auc", comparison=DJ), value="0.016")
C("9.19", "(+0.016, +0.009 to +0.022 against overlapping windows)", RPc, csvw(RPc, "auc_lo", comparison=DJ), value="0.009")
C("9.19", "(+0.016, +0.009 to +0.022 against overlapping windows)", RPc, csvw(RPc, "auc_hi", comparison=DJ), value="0.022")
RPC = RT + "parameter_comparison.csv"
C("9.19", "falls from 0.155 to 0.044", RPC, csvw(RPC, "full-sequence fit", parameter="pi[Indeterminate]"), value="0.155")
C("9.19", "falls from 0.155 to 0.044", RPC, csvw(RPC, "window fit", parameter="pi[Indeterminate]"), value="0.044")
C("9.19", "rises off its floor to 0.665", RPC, csvw(RPC, "window fit", parameter="E diag [Indeterminate]"), value="0.665")
C("9.19", "falls from 0.940 to 0.875", RPC, csvw(RPC, "full-sequence fit", parameter="E diag [TD]"), value="0.940")
C("9.19", "falls from 0.940 to 0.875", RPC, csvw(RPC, "window fit", parameter="E diag [TD]"), value="0.875")
C("9.19", "2,192 common targets of 742 patients", RPc, csvw(RPc, "n_targets", comparison=PRI), value="2192")
C("9.19", "2,192 common targets of 742 patients", RPc, csvw(RPc, "n_patients", comparison=PRI), value="742")
RCP_ = RT + "condition_predictions.csv"
C("6.15", "3,354 windows from 933 patients", RCP_, ex("int(df['n_train_windows_overlapping'].iloc[0])", RCP_), value="3354")
C("6.15", "non-overlapping windows (1,912)", RCP_, ex("int(df['n_train_windows_disjoint'].iloc[0])", RCP_), value="1912")

# ---------------------------------------------------------------- 9.20 experiments 6A and 6B
AP6 = X6A + "paired_comparisons.csv"
L1 = "~^last 1\\s+-\\s+full history$"; L2 = "~^last 2\\s+-\\s+full history$"; L3 = "~^last 3\\s+-\\s+full history$"; L5 = "~^last 5\\s+-\\s+full history$"; FP = "~^full history\\s+-\\s+persistence$"
C("9.20", "costs the profile −0.014 (−0.022 to −0.006) in accuracy", AP6, csvw(AP6, "accuracy", comparison=L1), value="-0.014")
C("9.20", "costs the profile −0.014 (−0.022 to −0.006) in accuracy", AP6, csvw(AP6, "accuracy_lo", comparison=L1), value="-0.022")
C("9.20", "costs the profile −0.014 (−0.022 to −0.006) in accuracy", AP6, csvw(AP6, "accuracy_hi", comparison=L1), value="-0.006")
C("9.20", "−0.011 in AUC", AP6, csvw(AP6, "auc", comparison=L1), value="-0.011")
C("9.20", "−0.033 (−0.050 to −0.016) in PIGD sensitivity", AP6, csvw(AP6, "sensitivity_PIGD", comparison=L1), value="-0.033")
C("9.20", "−0.033 (−0.050 to −0.016) in PIGD sensitivity", AP6, csvw(AP6, "sensitivity_PIGD_lo", comparison=L1), value="-0.050")
C("9.20", "−0.033 (−0.050 to −0.016) in PIGD sensitivity", AP6, csvw(AP6, "sensitivity_PIGD_hi", comparison=L1), value="-0.016")
C("9.20", "changes accuracy by +0.005 (−0.003 to +0.012)", AP6, csvw(AP6, "accuracy", comparison=L2), value="0.005")
C("9.20", "changes accuracy by +0.005 (−0.003 to +0.012)", AP6, csvw(AP6, "accuracy_lo", comparison=L2), value="-0.003")
C("9.20", "changes accuracy by +0.005 (−0.003 to +0.012)", AP6, csvw(AP6, "accuracy_hi", comparison=L2), value="0.012")
C("9.20", "PIGD-sensitivity cost of −0.018 (−0.036 to 0.000)", AP6, csvw(AP6, "sensitivity_PIGD", comparison=L2), value="-0.018")
C("9.20", "PIGD-sensitivity cost of −0.018 (−0.036 to 0.000)", AP6, csvw(AP6, "sensitivity_PIGD_lo", comparison=L2), value="-0.036")
C("9.20", "PIGD-sensitivity cost of −0.018 (−0.036 to 0.000)", AP6, csvw(AP6, "sensitivity_PIGD_hi", comparison=L2), value="0.000")
C("9.20", "TD-specificity gain of +0.016", AP6, csvw(AP6, "specificity_TD", comparison=L2), value="0.016")
C("9.20", "by +0.006 and −0.003", AP6, csvw(AP6, "accuracy", comparison=L3), value="0.006")
C("9.20", "by +0.006 and −0.003", AP6, csvw(AP6, "accuracy", comparison=L5), value="-0.003")
C("9.20", "(+0.022, +0.012 to +0.033)", AP6, csvw(AP6, "balanced_accuracy", comparison=FP), value="0.022")
C("9.20", "(+0.022, +0.012 to +0.033)", AP6, csvw(AP6, "balanced_accuracy_lo", comparison=FP), value="0.012")
C("9.20", "(+0.022, +0.012 to +0.033)", AP6, csvw(AP6, "balanced_accuracy_hi", comparison=FP), value="0.033")
C("9.20", "Brier (−0.042)", AP6, csvw(AP6, "brier", comparison=FP), value="-0.042")
C("9.20", "PIGD sensitivity (+0.064, +0.045 to +0.084)", AP6, csvw(AP6, "sensitivity_PIGD", comparison=FP), value="0.064")
C("9.20", "PIGD sensitivity (+0.064, +0.045 to +0.084)", AP6, csvw(AP6, "sensitivity_PIGD_lo", comparison=FP), value="0.045")
C("9.20", "PIGD sensitivity (+0.064, +0.045 to +0.084)", AP6, csvw(AP6, "sensitivity_PIGD_hi", comparison=FP), value="0.084")
C("9.20", "not on accuracy (+0.005, −0.003 to +0.013)", AP6, csvw(AP6, "accuracy", comparison=FP), value="0.005")
C("9.20", "not on accuracy (+0.005, −0.003 to +0.013)", AP6, csvw(AP6, "accuracy_lo", comparison=FP), value="-0.003")
C("9.20", "not on accuracy (+0.005, −0.003 to +0.013)", AP6, csvw(AP6, "accuracy_hi", comparison=FP), value="0.013")
BC = X6B + "condition_metrics.csv"; BP = X6B + "paired_comparisons.csv"
C("9.20", "conditions span 0.838 to 0.841 (persistence 0.834)", BC, csvw(BC, "accuracy", condition="D: window-trained, last 2"), value="0.838")
C("9.20", "conditions span 0.838 to 0.841 (persistence 0.834)", BC, csvw(BC, "accuracy", condition="B: full-trained, last 2"), value="0.841")
C("9.20", "conditions span 0.838 to 0.841 (persistence 0.834)", BC, csvw(BC, "accuracy", condition="persistence"), value="0.834")
DB = "~^D: window-trained, last 2\\s+-\\s+B: full-trained, last 2$"
CA = "~^C: window-trained, full history\\s+-\\s+A: full-trained, full history$"
BAc = "~^B: full-trained, last 2\\s+-\\s+A: full-trained, full history$"
C("9.20", "is −0.003 (−0.012 to +0.005) in accuracy", BP, csvw(BP, "accuracy", comparison=DB), value="-0.003")
C("9.20", "is −0.003 (−0.012 to +0.005) in accuracy", BP, csvw(BP, "accuracy_lo", comparison=DB), value="-0.012")
C("9.20", "is −0.003 (−0.012 to +0.005) in accuracy", BP, csvw(BP, "accuracy_hi", comparison=DB), value="0.005")
C("9.20", "−0.004 (−0.008 to −0.001) in AUC", BP, csvw(BP, "auc", comparison=DB), value="-0.004")
C("9.20", "−0.004 (−0.008 to −0.001) in AUC", BP, csvw(BP, "auc_lo", comparison=DB), value="-0.008")
C("9.20", "−0.004 (−0.008 to −0.001) in AUC", BP, csvw(BP, "auc_hi", comparison=DB), value="-0.001")
C("9.20", "+0.004 (+0.002 to +0.006) in Brier", BP, csvw(BP, "brier", comparison=DB), value="0.004")
C("9.20", "+0.004 (+0.002 to +0.006) in Brier", BP, csvw(BP, "brier_lo", comparison=DB), value="0.002")
C("9.20", "+0.004 (+0.002 to +0.006) in Brier", BP, csvw(BP, "brier_hi", comparison=DB), value="0.006")
C("9.20", "+0.007 (+0.000 to +0.013) in log loss", BP, csvw(BP, "log_loss", comparison=DB), value="0.007")
C("9.20", "+0.007 (+0.000 to +0.013) in log loss", BP, csvw(BP, "log_loss_lo", comparison=DB), value="0.000")
C("9.20", "+0.007 (+0.000 to +0.013) in log loss", BP, csvw(BP, "log_loss_hi", comparison=DB), value="0.013")
C("9.20", "with the full history is −0.007 in AUC", BP, csvw(BP, "auc", comparison=CA), value="-0.007")
C("9.20", "is +0.001 (−0.006 to +0.008) in accuracy", BP, csvw(BP, "accuracy", comparison=BAc), value="0.001")
C("9.20", "is +0.001 (−0.006 to +0.008) in accuracy", BP, csvw(BP, "accuracy_lo", comparison=BAc), value="-0.006")
C("9.20", "is +0.001 (−0.006 to +0.008) in accuracy", BP, csvw(BP, "accuracy_hi", comparison=BAc), value="0.008")
C("9.20", "PIGD-sensitivity cost of −0.018 (−0.036 to −0.001)", BP, csvw(BP, "sensitivity_PIGD", comparison=BAc), value="-0.018")
C("9.20", "PIGD-sensitivity cost of −0.018 (−0.036 to −0.001)", BP, csvw(BP, "sensitivity_PIGD_lo", comparison=BAc), value="-0.036")
C("9.20", "PIGD-sensitivity cost of −0.018 (−0.036 to −0.001)", BP, csvw(BP, "sensitivity_PIGD_hi", comparison=BAc), value="-0.001")
BPC = X6B + "parameter_comparison.csv"
C("9.20", "observation variance (0.194 to 0.148)", BPC, csvw(BPC, "full sequences", parameter="obs_var"), value="0.194")
C("9.20", "observation variance (0.194 to 0.148)", BPC, csvw(BPC, "overlapping windows", parameter="obs_var"), value="0.148")
C("9.20", "level innovation variance (0.103 to 0.189)", BPC, csvw(BPC, "full sequences", parameter="level_var"), value="0.103")
C("9.20", "level innovation variance (0.103 to 0.189)", BPC, csvw(BPC, "overlapping windows", parameter="level_var"), value="0.189")
C("9.20", "2,184 common targets of 740 patients", BC, csvw(BC, "n_targets", condition="D: window-trained, last 2"), value="2184")
C("9.20", "2,184 common targets of 740 patients", BC, csvw(BC, "n_patients", condition="D: window-trained, last 2"), value="740")

# ---------------------------------------------------------------- 9.21 patient-level
PAD = PL + "patient_accuracy_distribution.csv"
C("9.21", "visit-level accuracy is 0.839 and its mean per-patient accuracy 0.834 (0.817 to 0.852)", PAD, csvw(PAD, "visit_level_accuracy", condition="full history", experiment="6A"), value="0.839")
C("9.21", "mean per-patient accuracy 0.834 (0.817 to 0.852)", PAD, csvw(PAD, "mean_patient_accuracy", condition="full history", experiment="6A"), value="0.834")
C("9.21", "mean per-patient accuracy 0.834 (0.817 to 0.852)", PAD, csvw(PAD, "mean_ci_lo", condition="full history", experiment="6A"), value="0.817")
C("9.21", "mean per-patient accuracy 0.834 (0.817 to 0.852)", PAD, csvw(PAD, "mean_ci_hi", condition="full history", experiment="6A"), value="0.852")
C("9.21", "63.7% of patients (572)", PAD, csvw(PAD, "pct_exactly 100%", condition="full history", experiment="6A"), value="63.7")
C("9.21", "63.7% of patients (572)", PAD, csvw(PAD, "n_exactly 100%", condition="full history", experiment="6A"), value="572")
C("9.21", "5.0% (45) never correctly", PAD, csvw(PAD, "pct_exactly 0%", condition="full history", experiment="6A"), value="5.0")
C("9.21", "5.0% (45) never correctly", PAD, csvw(PAD, "n_exactly 0%", condition="full history", experiment="6A"), value="45")
C("9.21", "8.2% below 50%", PAD, csvw(PAD, "pct_below_50pct_cumulative", condition="full history", experiment="6A"), value="8.2")
C("9.21", "70.9% at 80% or better", PAD, csvw(PAD, "pct_at_least_80pct_cumulative", condition="full history", experiment="6A"), value="70.9")
C("9.21", "(0.831, 63.1% perfect)", PAD, csvw(PAD, "mean_patient_accuracy", condition="persistence", experiment="6A"), value="0.831")
C("9.21", "(0.831, 63.1% perfect)", PAD, csvw(PAD, "pct_exactly 100%", condition="persistence", experiment="6A"), value="63.1")
PBC = PL + "patient_accuracy_by_subtype_composition.csv"
for comp, v, n, pv in [("all TD targets", "0.944", "510", "0.970"), ("all PIGD targets", "0.811", "162", "0.733"), ("majority TD", "0.684", "105", "0.670"), ("majority PIGD", "0.592", "61", "0.541"), ("evenly split", "0.468", "60", "0.480")]:
    C("9.21", v, PBC, csvw(PBC, "mean_patient_accuracy", condition="full history", subtype_composition=comp, experiment="6A"))
    C("9.21", n, PBC, csvw(PBC, "n_patients", condition="full history", subtype_composition=comp, experiment="6A"))
    C("9.21", pv, PBC, csvw(PBC, "mean_patient_accuracy", condition="persistence", subtype_composition=comp, experiment="6A"), value=pv)
PSMx = PL + "patient_subtype_metrics.csv"
C("9.21", "PIGD sensitivity is 0.550 on average (0.722", PSMx, csvw(PSMx, "mean_patient_PIGD_sensitivity", condition="full history", experiment="6A"), value="0.550")
C("9.21", "PIGD sensitivity is 0.550 on average (0.722", PSMx, csvw(PSMx, "mean_PIGD_sensitivity_ge2_targets", condition="full history", experiment="6A"), value="0.722")
C("9.21", "among the 225 patients", PSMx, csvw(PSMx, "n_patients_ge2_PIGD_targets", condition="full history", experiment="6A"), value="225")
C("9.21", "falls from 32.5% to 6.2%", PSMx, csvw(PSMx, "pct_PIGD_patients_zero_correct", condition="full history", experiment="6A"), value="32.5")
C("9.21", "falls from 32.5% to 6.2%", PSMx, csvw(PSMx, "pct_zero_correct_ge2_PIGD_targets", condition="full history", experiment="6A"), value="6.2")
C("9.21", "TD specificity 0.858 (0.917", PSMx, csvw(PSMx, "mean_patient_TD_specificity", condition="full history", experiment="6A"), value="0.858")
C("9.21", "TD specificity 0.858 (0.917", PSMx, csvw(PSMx, "mean_TD_specificity_ge2_targets", condition="full history", experiment="6A"), value="0.917")
PPC = PL + "paired_patient_comparisons.csv"
C("9.21", "(6B, D against B) −0.004 (−0.016 to +0.008) with 660 of 740", PPC, csvw(PPC, "mean_paired_difference", comparison="~^1\\. D vs B"), value="-0.004")
C("9.21", "(6B, D against B) −0.004 (−0.016 to +0.008) with 660 of 740", PPC, csvw(PPC, "diff_ci_lo", comparison="~^1\\. D vs B"), value="-0.016")
C("9.21", "(6B, D against B) −0.004 (−0.016 to +0.008) with 660 of 740", PPC, csvw(PPC, "diff_ci_hi", comparison="~^1\\. D vs B"), value="0.008")
C("9.21", "(6B, D against B) −0.004 (−0.016 to +0.008) with 660 of 740", PPC, csvw(PPC, "n_unchanged", comparison="~^1\\. D vs B"), value="660")
C("9.21", "(C against A) −0.004 (−0.012 to +0.005) with 821 of 898", PPC, csvw(PPC, "mean_paired_difference", comparison="~^2\\. C vs A"), value="-0.004")
C("9.21", "(C against A) −0.004 (−0.012 to +0.005) with 821 of 898", PPC, csvw(PPC, "diff_ci_lo", comparison="~^2\\. C vs A"), value="-0.012")
C("9.21", "(C against A) −0.004 (−0.012 to +0.005) with 821 of 898", PPC, csvw(PPC, "diff_ci_hi", comparison="~^2\\. C vs A"), value="0.005")
C("9.21", "(C against A) −0.004 (−0.012 to +0.005) with 821 of 898", PPC, csvw(PPC, "n_unchanged", comparison="~^2\\. C vs A"), value="821")
C("9.21", "the single-visit window −0.012 (−0.023 to −0.002) with 774 of 898", PPC, csvw(PPC, "mean_paired_difference", comparison="last 1 vs full history"), value="-0.012")
C("9.21", "the single-visit window −0.012 (−0.023 to −0.002) with 774 of 898", PPC, csvw(PPC, "diff_ci_lo", comparison="last 1 vs full history"), value="-0.023")
C("9.21", "the single-visit window −0.012 (−0.023 to −0.002) with 774 of 898", PPC, csvw(PPC, "diff_ci_hi", comparison="last 1 vs full history"), value="-0.002")
C("9.21", "the single-visit window −0.012 (−0.023 to −0.002) with 774 of 898", PPC, csvw(PPC, "n_unchanged", comparison="last 1 vs full history"), value="774")
C("9.21", "full history against persistence +0.003 (−0.008 to +0.014)", PPC, csvw(PPC, "mean_paired_difference", comparison="full history vs persistence"), value="0.003")
C("9.21", "full history against persistence +0.003 (−0.008 to +0.014)", PPC, csvw(PPC, "diff_ci_lo", comparison="full history vs persistence"), value="-0.008")
C("9.21", "full history against persistence +0.003 (−0.008 to +0.014)", PPC, csvw(PPC, "diff_ci_hi", comparison="full history vs persistence"), value="0.014")

# ---------------------------------------------------------------- 9.22 ratio bands
RBMc = RB + "model_comparison.csv"; RBC = RB + "contrasts.csv"
C("9.22", "reaches 0.832 (0.814 to 0.850) against 0.837", RBMc, csvw(RBMc, "accuracy", model=RBM), value="0.832")
C("9.22", "reaches 0.832 (0.814 to 0.850) against 0.837", RBMc, csvw(RBMc, "accuracy_ci_patient_bootstrap_low", model=RBM), value="0.814")
C("9.22", "reaches 0.832 (0.814 to 0.850) against 0.837", RBMc, csvw(RBMc, "accuracy_ci_patient_bootstrap_high", model=RBM), value="0.850")
C("9.22", "−0.005 (−0.013 to +0.003)", RBC, csvw(RBC, "accuracy_delta", contrast="~^ratio bands added"), value="-0.005")
C("9.22", "−0.005 (−0.013 to +0.003)", RBC, csvw(RBC, "delta_ci_patient_bootstrap_low", contrast="~^ratio bands added"), value="-0.013")
C("9.22", "−0.005 (−0.013 to +0.003)", RBC, csvw(RBC, "delta_ci_patient_bootstrap_high", contrast="~^ratio bands added"), value="0.003")
C("9.22", "on 67 cases against 83", RBC, csvw(RBC, "a_right_b_wrong", contrast="~^ratio bands added"), value="67")
C("9.22", "on 67 cases against 83", RBC, csvw(RBC, "b_right_a_wrong", contrast="~^ratio bands added"), value="83")
C("9.22", "Brier is 0.122 against 0.121, log loss 0.393 against 0.396", RBC, csvw(RBC, "brier_a", contrast="~^ratio bands added"), value="0.122")
C("9.22", "Brier is 0.122 against 0.121, log loss 0.393 against 0.396", RBC, csvw(RBC, "brier_b", contrast="~^ratio bands added"), value="0.121")
C("9.22", "Brier is 0.122 against 0.121, log loss 0.393 against 0.396", RBC, csvw(RBC, "log_loss_a", contrast="~^ratio bands added"), value="0.393")
C("9.22", "Brier is 0.122 against 0.121, log loss 0.393 against 0.396", RBC, csvw(RBC, "log_loss_b", contrast="~^ratio bands added"), value="0.396")
BWS = BW + "summary.json"
C("9.22", "calibration gap 0.028 against 0.0065", BWS, jsn(BWS, "calibration_gap/width 0.10"), value="0.028")
C("9.22", "calibration gap 0.028 against 0.0065", BWS, jsn(BWS, "calibration_gap/subtype only"), value="0.0065")
ARR = RB + "accuracy_by_ratio_range.csv"
C("9.22", "loses 3.2 points on visits with a ratio in [1.15, 2.00)", ARR, ex("100*(df[(df.model=='%s')&(df.ratio_range=='[1.15,2.00)')]['accuracy'].iloc[0]-df[(df.model=='%s')&(df.ratio_range=='[1.15,2.00)')]['accuracy'].iloc[0])" % (CTH, RBM), ARR), value="3.2")
C("9.22", "rises from 0.859 (0.840 to 0.876) to 0.876 (0.859 to 0.893)", RAMx, csvw(RAMx, "auc_ci_low", arm="baseline (subtype only)"), value="0.840")
C("9.22", "rises from 0.859 (0.840 to 0.876) to 0.876 (0.859 to 0.893)", RAMx, csvw(RAMx, "auc_ci_high", arm="baseline (subtype only)"), value="0.876")
C("9.22", "rises from 0.859 (0.840 to 0.876) to 0.876 (0.859 to 0.893)", RAMx, csvw(RAMx, "auc_ci_low", arm="ratio-banded (subtype x band)"), value="0.859")
C("9.22", "rises from 0.859 (0.840 to 0.876) to 0.876 (0.859 to 0.893)", RAMx, csvw(RAMx, "auc_ci_high", arm="ratio-banded (subtype x band)"), value="0.893")
RAC = RBA + "auc_contrasts.csv"
C("9.22", "paired gain of +0.017 (+0.009 to +0.026)", RAC, csvw(RAC, "auc_delta", contrast="~^ratio bands added"), value="0.017")
C("9.22", "paired gain of +0.017 (+0.009 to +0.026)", RAC, csvw(RAC, "auc_delta_ci_low", contrast="~^ratio bands added"), value="0.009")
C("9.22", "paired gain of +0.017 (+0.009 to +0.026)", RAC, csvw(RAC, "auc_delta_ci_high", contrast="~^ratio bands added"), value="0.026")
C("9.22", "average precision falls by −0.010 (−0.032 to +0.013)", RAC, csvw(RAC, "ap_delta", contrast="~^ratio bands added"), value="-0.010")
C("9.22", "average precision falls by −0.010 (−0.032 to +0.013)", RAC, csvw(RAC, "ap_delta_ci_low", contrast="~^ratio bands added"), value="-0.032")
C("9.22", "average precision falls by −0.010 (−0.032 to +0.013)", RAC, csvw(RAC, "ap_delta_ci_high", contrast="~^ratio bands added"), value="0.013")
PAC = RBA + "partial_auc_contrast.csv"
C("9.22", "by −0.006 (−0.016 to +0.004)", PAC, csvi(PAC, "delta", 0), value="-0.006")
C("9.22", "by −0.006 (−0.016 to +0.004)", PAC, csvi(PAC, "ci_low", 0), value="-0.016")
C("9.22", "by −0.006 (−0.016 to +0.004)", PAC, csvi(PAC, "ci_high", 0), value="0.004")
OP = RBA + "operating_points.csv"
C("9.22", "calls 890 cases PIGD against 800", OP, csvw(OP, "n_called_PIGD", arm="ratio-banded (subtype x band)"), value="890")
C("9.22", "calls 890 cases PIGD against 800", OP, csvw(OP, "n_called_PIGD", arm="baseline (subtype only)"), value="800")
C("9.22", "from (0.667, 0.090) to (0.708, 0.115)", OP, csvw(OP, "operating_tpr", arm="baseline (subtype only)"), value="0.667")
C("9.22", "from (0.667, 0.090) to (0.708, 0.115)", OP, csvw(OP, "operating_fpr", arm="baseline (subtype only)"), value="0.090")
C("9.22", "from (0.667, 0.090) to (0.708, 0.115)", OP, csvw(OP, "operating_tpr", arm="ratio-banded (subtype x band)"), value="0.708")
C("9.22", "from (0.667, 0.090) to (0.708, 0.115)", OP, csvw(OP, "operating_fpr", arm="ratio-banded (subtype x band)"), value="0.115")
ABS = RBA + "auc_by_stratum.csv"
C("9.22", "current-TD stratum (0.669 to 0.750)", ABS, csvw(ABS, "auc_baseline", stratum="current label", level="TD"), value="0.669")
C("9.22", "current-TD stratum (0.669 to 0.750)", ABS, csvw(ABS, "auc_banded", stratum="current label", level="TD"), value="0.750")
C("9.22", "one prior visit (0.767 to 0.836)", ABS, csvw(ABS, "auc_baseline", stratum="prior visits", level="1"), value="0.767")
C("9.22", "one prior visit (0.767 to 0.836)", ABS, csvw(ABS, "auc_banded", stratum="prior visits", level="1"), value="0.836")
BI = RBE + "band_information.csv"
C("9.22", "up to 0.66 (TD), 0.50 (Indeterminate) and 0.63 (PIGD)", BI, csvw(BI, "max_total_variation_between_states", subtype="TD"), value="0.66")
C("9.22", "up to 0.66 (TD), 0.50 (Indeterminate) and 0.63 (PIGD)", BI, csvw(BI, "max_total_variation_between_states", subtype="Indeterminate"), value="0.50")
C("9.22", "up to 0.66 (TD), 0.50 (Indeterminate) and 0.63 (PIGD)", BI, csvw(BI, "max_total_variation_between_states", subtype="PIGD"), value="0.63")
FBS = RB + "fold_band_structures.csv"
C("9.22", "(44 categories, 137 parameters", FBS, csvi(FBS, "n_categories", 0), value="44")
C("9.22", "(44 categories, 137 parameters", FBS, csvi(FBS, "n_params", 0), value="137")
C("9.22", "ten tail bands", FBS, csvi(FBS, "n_tail_bands", 0), value="10")

# ---------------------------------------------------------------- 9.23 band width
BWM = BW + "model_metrics.csv"
for arm, auc, ap, pauc in [("width 0.05", "0.874", "0.733", "0.780"), ("width 0.10", "0.876", "0.740", "0.782"), ("width 0.20", "0.877", "0.754", "0.787"), ("width 0.30", "0.878", "0.752", "0.789"), ("width 0.50", "0.876", "0.741", "0.783")]:
    C("9.23", "AUC-ROC is 0.874, 0.876, 0.877, 0.878 and 0.876", BWM, csvw(BWM, "auc_roc", arm=arm), value=auc)
    C("9.23", "average precision 0.733, 0.740, 0.754, 0.752 and 0.741", BWM, csvw(BWM, "pr_auc", arm=arm), value=ap)
    C("9.23", "partial AUC 0.780, 0.782, 0.787, 0.789 and 0.783", BWM, csvw(BWM, "partial_auc_fpr_le_0.2", arm=arm), value=pauc)
C("9.23", "Accuracy lies between 0.832 and 0.836", BWM, ex("df[df.arm!='subtype only']['accuracy'].min()", BWM), value="0.832")
C("9.23", "Accuracy lies between 0.832 and 0.836", BWM, ex("df[df.arm!='subtype only']['accuracy'].max()", BWM), value="0.836")
C("9.23", "against 0.750", BWM, csvw(BWM, "pr_auc", arm="subtype only"), value="0.750")
C("9.23", "against 0.788", BWM, csvw(BWM, "partial_auc_fpr_le_0.2", arm="subtype only"), value="0.788")
BRC = BW + "ranking_contrasts.csv"
C("9.23", "(AUC +0.003, +0.000 to +0.005; average precision +0.012, +0.003 to +0.022; partial AUC +0.007, +0.002 to +0.012)", BRC, csvw(BRC, "auc_delta", arm_a="width 0.30", arm_b="width 0.10"), value="0.003")
C("9.23", "(AUC +0.003, +0.000 to +0.005; average precision +0.012, +0.003 to +0.022; partial AUC +0.007, +0.002 to +0.012)", BRC, csvw(BRC, "auc_ci_low", arm_a="width 0.30", arm_b="width 0.10"), value="0.000")
C("9.23", "(AUC +0.003, +0.000 to +0.005; average precision +0.012, +0.003 to +0.022; partial AUC +0.007, +0.002 to +0.012)", BRC, csvw(BRC, "auc_ci_high", arm_a="width 0.30", arm_b="width 0.10"), value="0.005")
C("9.23", "average precision +0.012, +0.003 to +0.022", BRC, csvw(BRC, "pr_auc_delta", arm_a="width 0.30", arm_b="width 0.10"), value="0.012")
C("9.23", "average precision +0.012, +0.003 to +0.022", BRC, csvw(BRC, "pr_auc_ci_low", arm_a="width 0.30", arm_b="width 0.10"), value="0.003")
C("9.23", "average precision +0.012, +0.003 to +0.022", BRC, csvw(BRC, "pr_auc_ci_high", arm_a="width 0.30", arm_b="width 0.10"), value="0.022")
C("9.23", "partial AUC +0.007, +0.002 to +0.012", BRC, csvw(BRC, "partial_auc_delta", arm_a="width 0.30", arm_b="width 0.10"), value="0.007")
C("9.23", "partial AUC +0.007, +0.002 to +0.012", BRC, csvw(BRC, "partial_auc_ci_low", arm_a="width 0.30", arm_b="width 0.10"), value="0.002")
C("9.23", "partial AUC +0.007, +0.002 to +0.012", BRC, csvw(BRC, "partial_auc_ci_high", arm_a="width 0.30", arm_b="width 0.10"), value="0.012")
BSC = BW + "secondary_contrasts_vs_unbanded.csv"
C("9.23", "gains AUC-ROC (+0.015 to +0.020)", BSC, ex("df['auc_delta'].min()", BSC), value="0.015")
C("9.23", "gains AUC-ROC (+0.015 to +0.020)", BSC, ex("df['auc_delta'].max()", BSC), value="0.020")
C("9.23", "(width 0.30: +0.003, −0.017 to +0.021 and +0.001, −0.007 to +0.009)", BSC, csvw(BSC, "pr_auc_delta", arm_a="width 0.30"), value="0.003")
C("9.23", "(width 0.30: +0.003, −0.017 to +0.021 and +0.001, −0.007 to +0.009)", BSC, csvw(BSC, "pr_auc_ci_low", arm_a="width 0.30"), value="-0.017")
C("9.23", "(width 0.30: +0.003, −0.017 to +0.021 and +0.001, −0.007 to +0.009)", BSC, csvw(BSC, "pr_auc_ci_high", arm_a="width 0.30"), value="0.021")
C("9.23", "(width 0.30: +0.003, −0.017 to +0.021 and +0.001, −0.007 to +0.009)", BSC, csvw(BSC, "partial_auc_delta", arm_a="width 0.30"), value="0.001")
C("9.23", "(width 0.30: +0.003, −0.017 to +0.021 and +0.001, −0.007 to +0.009)", BSC, csvw(BSC, "partial_auc_ci_low", arm_a="width 0.30"), value="-0.007")
C("9.23", "(width 0.30: +0.003, −0.017 to +0.021 and +0.001, −0.007 to +0.009)", BSC, csvw(BSC, "partial_auc_ci_high", arm_a="width 0.30"), value="0.009")
C("9.23", "best at width 0.30 (0.120, 0.387)", BWM, csvw(BWM, "brier_two_class", arm="width 0.30"), value="0.120")
C("9.23", "best at width 0.30 (0.120, 0.387)", BWM, csvw(BWM, "log_loss", arm="width 0.30"), value="0.387")
C("9.23", "worst at 0.05 and 0.50 (0.025, 0.029)", BWS, jsn(BWS, "calibration_gap/width 0.05"), value="0.025")
C("9.23", "worst at 0.05 and 0.50 (0.025, 0.029)", BWS, jsn(BWS, "calibration_gap/width 0.50"), value="0.029")
C("9.23", "best at 0.30 (0.022)", BWS, jsn(BWS, "calibration_gap/width 0.30"), value="0.022")
CX = BW + "complexity_by_width.csv"
C("9.23", "(73, 44, 29, 24 and 20 categories; 224, 137, 92, 77 and 65 parameters)", CX, csvw(CX, "n_categories", main_width="0.05"), value="73")
C("9.23", "(73, 44, 29, 24 and 20 categories; 224, 137, 92, 77 and 65 parameters)", CX, csvw(CX, "n_categories", main_width="0.1"), value="44")
C("9.23", "(73, 44, 29, 24 and 20 categories; 224, 137, 92, 77 and 65 parameters)", CX, csvw(CX, "n_categories", main_width="0.2"), value="29")
C("9.23", "(73, 44, 29, 24 and 20 categories; 224, 137, 92, 77 and 65 parameters)", CX, csvw(CX, "n_categories", main_width="0.3"), value="24")
C("9.23", "(73, 44, 29, 24 and 20 categories; 224, 137, 92, 77 and 65 parameters)", CX, csvw(CX, "n_categories", main_width="0.5"), value="20")
C("9.23", "(73, 44, 29, 24 and 20 categories; 224, 137, 92, 77 and 65 parameters)", CX, csvw(CX, "n_params", main_width="0.05"), value="224")
C("9.23", "(73, 44, 29, 24 and 20 categories; 224, 137, 92, 77 and 65 parameters)", CX, csvw(CX, "n_params", main_width="0.1"), value="137")
C("9.23", "(73, 44, 29, 24 and 20 categories; 224, 137, 92, 77 and 65 parameters)", CX, csvw(CX, "n_params", main_width="0.2"), value="92")
C("9.23", "(73, 44, 29, 24 and 20 categories; 224, 137, 92, 77 and 65 parameters)", CX, csvw(CX, "n_params", main_width="0.3"), value="77")
C("9.23", "(73, 44, 29, 24 and 20 categories; 224, 137, 92, 77 and 65 parameters)", CX, csvw(CX, "n_params", main_width="0.5"), value="65")
C("9.23", "holds at least eight training observations", CX, csvw(CX, "min_obs_nonempty", main_width="0.3"), value="8")

FORBIDDEN = ["clustering method review", "clusterab", "clustering audit", "slope clustering", "clustering review", "cluster phase", "varsha", "neurocomp"]


# ===================================================================== engine
def resolve_file(f: str) -> Path:
    if any(ch in f for ch in "*?"):
        hits = sorted(glob.glob(str(ROOT / f)))
        if not hits:
            raise FileNotFoundError(f)
        return Path(hits[0])
    return ROOT / f


_CACHE: dict = {}


def load_csv(f: str) -> pd.DataFrame:
    if f not in _CACHE:
        _CACHE[f] = pd.read_csv(resolve_file(f))
    return _CACHE[f]


def match(series: pd.Series, want: str) -> pd.Series:
    s = series.astype(str).str.strip()
    if want.startswith("~"):
        return s.str.contains(want[1:], regex=True)
    return s == want


def source_value(loc: dict):
    k = loc["kind"]
    if k == "csv":
        df = load_csv(loc["file"])
        if "row" in loc:
            return float(df[loc["col"]].iloc[loc["row"]])
        mask = pd.Series(True, index=df.index)
        for col, want in loc["where"].items():
            mask &= match(df[col], want)
        sub = df[mask]
        if len(sub) == 0:
            raise KeyError(f"no row matches {loc['where']} in {loc['file']}")
        return float(sub[loc["col"]].iloc[0])
    if k == "json":
        obj = json.loads(resolve_file(loc["file"]).read_text())
        for part in loc["path"].split("/"):
            obj = obj[int(part)] if isinstance(obj, list) else obj[part]
        return float(obj)
    if k == "txt":
        text = resolve_file(loc["file"]).read_text()
        m = re.search(loc["rx"], text)
        if not m:
            raise KeyError(f"regex {loc['rx']!r} not found in {loc['file']}")
        return float(m.group(1).replace(",", "").replace("−", "-").replace("–", "-"))
    if k == "expr":
        env = {"pd": pd, "np": np, "ROOT": ROOT, "json": json}
        if loc.get("file"):
            f = loc["file"]
            if f.endswith(".json"):
                env["d"] = json.loads(resolve_file(f).read_text())
            else:
                env["df"] = load_csv(f)
        return float(eval(loc["py"], env))
    raise ValueError(k)


def parse_printed(v: str) -> tuple[float, int]:
    s = v.strip().replace("−", "-").replace("–", "-").replace(",", "").rstrip("%").replace("+", "")
    dec = len(s.split(".")[1]) if "." in s else 0
    return float(s), dec


def main() -> int:
    rows, n_pass = [], 0
    for section, printed, value, source, loc, scale, absval in CHECKS:
        present = printed in TEXT or printed.replace("−", "-") in TEXT_ASCII
        try:
            sv = source_value(loc) * scale
            if absval:
                sv = abs(sv)
            pv, dec = parse_printed(value)
            tol = 0.5 * 10 ** (-dec) + 1e-9
            ok_val = abs(sv - pv) <= tol
            status = "PASS" if (ok_val and present) else ("VALUE MISMATCH" if not ok_val else "TEXT NOT FOUND")
            svs = f"{sv:.6g}"
        except Exception as exc:  # noqa: BLE001
            status, svs = f"ERROR: {type(exc).__name__}: {exc}"[:160], ""
        n_pass += status == "PASS"
        rows.append({"section": section, "printed": printed, "value": value, "source": source,
                     "locator": json.dumps(loc, default=str)[:300], "source_value": svs, "status": status})
    out = MF / "tables" / "numerical_traceability.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    bad = [r for r in rows if r["status"] != "PASS"]
    print(f"{n_pass} of {len(rows)} numerical checks passed; written to {out.relative_to(ROOT)}")
    for r in bad:
        print(f"  [{r['status']}] {r['section']}: {r['printed']!r} -> {r['source_value']} ({r['source']})")
    low = TEXT.lower()
    forb = [f for f in FORBIDDEN if f in low]
    print("forbidden-content check:", "PASS (none found)" if not forb else f"FAIL {forb}")
    return 0 if not bad and not forb else 1


if __name__ == "__main__":
    sys.exit(main())
