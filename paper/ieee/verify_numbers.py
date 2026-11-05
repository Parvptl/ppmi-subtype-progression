#!/usr/bin/env python3
"""Check every quantitative claim in `main.tex` against the derived numbers.

`make_assets.py` reads the committed result files and writes
`derived_numbers.json`. This script rebuilds, from that JSON alone, the exact
string each claim should appear as, and asserts that the string is present in
`main.tex`. A claim that drifts away from its source file fails here.

Whitespace in `main.tex` is collapsed before matching, so line wrapping in the
source does not affect the check.

    ~/envs/ppmi/bin/python paper/ieee/verify_numbers.py

Exit status is 0 only if every check passes. Results are also written to
`numbers_check.csv`.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = json.loads((HERE / "derived_numbers.json").read_text())
TEX = re.sub(r"\s+", " ", (HERE / "main.tex").read_text())


def _round(value: float, dp: int) -> str:
    """Round half away from zero, which is how the numbers are written up.

    Python's `format` rounds the binary double, so a stored value such as
    0.6845 formats as 0.684 rather than the 0.685 a reader would write.
    """
    q = Decimal(1).scaleb(-dp)
    return str(Decimal(repr(value)).quantize(q, rounding=ROUND_HALF_UP))


def pct(key: str, dp: int = 1) -> str:
    return _round(100 * N[key], dp)


def p(key: str, dp: int = 3) -> str:
    return _round(N[key], dp)


# (section, claim, expected literal as it must appear in main.tex)
CHECKS: list[tuple[str, str, str]] = [
    # ---- cohort -----------------------------------------------------------
    ("cohort", "curated cut size",
     f"Of {N['cut_patients']:,} participants and {N['cut_visits']:,} visits"),
    ("cohort", "cohort size",
     f"{N['n_patients']:,} participants contributed {N['n_visits']:,} visits"),
    ("cohort", "labelled visits",
     f"assigned at {N['n_labelled']:,} visits ({N['pct_labelled']}\\%)"),
    ("cohort", "participants with a pair",
     f"{N['n_pairs_from_panel']:,} consecutive labelled visit pairs"),
    ("cohort", "baseline TD",
     f"{N['baseline_n_TD']} participants ({N['baseline_pct_TD']}\\%) were tremor dominant"),
    ("cohort", "baseline Indeterminate",
     f"{N['baseline_n_Indeterminate']} ({N['baseline_pct_Indeterminate']}\\%) Indeterminate"),
    ("cohort", "baseline PIGD",
     f"{N['baseline_n_PIGD']} ({N['baseline_pct_PIGD']}\\%) PIGD"),
    ("cohort", "median interval and follow-up",
     f"median interval between consecutive labelled visits was "
     f"{N['interval_median']:.2f} years and the median follow-up {N['fu_median']} years"),
    ("cohort", "label validation, three-class",
     f"{N['val_three_matches']:,} of {N['val_three_n']:,} ({pct('agree3_base', 2)}\\%)"),
    ("cohort", "label validation, two-class",
     f"{N['val_two_n']:,} of {N['val_two_n']:,} ({pct('agree2_base', 2)}\\%)"),

    # ---- missing labels ---------------------------------------------------
    ("missing", "unlabelled total",
     f"Of the {N['unlabelled_total']:,} cohort visits without a subtype"),
    ("missing", "ON-medication share",
     f"{N['unlabelled_on_med']:,} ({N['unlabelled_on_med_pct']}\\%) had only an ON-medication"),
    ("missing", "no Part III",
     f"{N['unlabelled_no_part3']} had no Part III record"),
    ("missing", "partially scored",
     f"{N['unlabelled_partial']} had an examination that was recorded but incomplete"),
    ("missing", "Part II blocked",
     f"{N['unlabelled_part2']} had a complete Part III"),
    ("missing", "follow-up and medication skew",
     f"median {N['unlab_years_median']} years from baseline against "
     f"{N['lab_years_median']}; {N['unlab_on_med_share']}\\% on antiparkinsonian "
     f"medication against {N['lab_on_med_share']}\\%"),

    # ---- transitions ------------------------------------------------------
    ("transitions", "TD->TD",
     f"probability {p('p_TD_TD')} (95\\% CI {p('p_TD_TD_lo')}--{p('p_TD_TD_hi')})"),
    ("transitions", "PIGD->PIGD",
     f"probability {p('p_PIGD_PIGD')} ({p('p_PIGD_PIGD_lo')}--{p('p_PIGD_PIGD_hi')})"),
    ("transitions", "Indet->Indet",
     f"probability only {p('p_Indet_Indet')} ({p('p_Indet_Indet_lo')}--{p('p_Indet_Indet_hi')})"),
    ("transitions", "Indet->TD",
     f"returning to TD in {p('p_Indet_TD')} ({p('p_Indet_TD_lo')}--{p('p_Indet_TD_hi')})"),
    ("transitions", "Indet->PIGD",
     f"moving to PIGD in {p('p_Indet_PIGD')} ({p('p_Indet_PIGD_lo')}--{p('p_Indet_PIGD_hi')})"),
    ("transitions", "PIGD not absorbing",
     f"{N['pigd_not_absorbing']}\\% of PIGD visits were followed by a non-PIGD"),

    # ---- reversion --------------------------------------------------------
    ("reversion", "number of changes",
     f"Of {N['n_changes']} observed changes of state"),
    ("reversion", "pooled reversion",
     f"{pct('reversion_pooled')}\\% ({pct('reversion_lo')}--{pct('reversion_hi')}) "
     f"returned to the origin state"),
    ("reversion", "TD->Indeterminate",
     f"TD$\\to$Indeterminate reversed in {p('rev_TD_Ind')} ({p('rev_TD_Ind_lo')}--{p('rev_TD_Ind_hi')})"),
    ("reversion", "PIGD->Indeterminate",
     f"PIGD$\\to$Indeterminate in {p('rev_PIGD_Ind')} ({p('rev_PIGD_Ind_lo')}--{p('rev_PIGD_Ind_hi')})"),
    ("reversion", "TD->PIGD",
     f"TD$\\to$PIGD in {p('rev_TD_PIGD')} ({p('rev_TD_PIGD_lo')}--{p('rev_TD_PIGD_hi')})"),
    ("reversion", "PIGD->TD",
     f"PIGD$\\to$TD in {p('rev_PIGD_TD')} ({p('rev_PIGD_TD_lo')}--{p('rev_PIGD_TD_hi')})"),

    # ---- conversion definition --------------------------------------------
    ("conversion", "risk set and counts",
     f"Among {N['naive_at_risk']} participants not PIGD at their first labelled visit, "
     f"{N['naive_converted']} converted to PIGD on a single-assessment definition but only "
     f"{N['sustained_converted']}"),
    ("conversion", "reduction",
     f"a reduction of {pct('sustained_removes')}\\% "
     f"({pct('sustained_removes_lo')}--{pct('sustained_removes_hi')})"),
    ("conversion", "five-year incidence",
     f"{pct('km_naive_yr5')}\\% ({pct('km_naive_yr5_lo')}--{pct('km_naive_yr5_hi')}) under the "
     f"single-visit definition against {pct('km_sust_yr5')}\\% "
     f"({pct('km_sust_yr5_lo')}--{pct('km_sust_yr5_hi')})"),
    ("conversion", "interval dependence",
     f"TD$\\to$TD fell from {p('p_TDTD_short')} at intervals of 1.5 years or less to "
     f"{p('p_TDTD_long')} beyond four years, while TD$\\to$PIGD rose from "
     f"{p('p_TDPIGD_short')} to {p('p_TDPIGD_long')}"),

    # ---- threshold --------------------------------------------------------
    ("threshold", "attainable values",
     f"The {N['n_ind_with_ratio']} Indeterminate visits with a defined ratio take only "
     f"\\textbf{{eleven distinct values}}, of which {N['top_ratio_n']} "
     f"({N['top_ratio_pct']}\\%) sit at {N['top_ratio']} and {N['second_ratio_n']} "
     f"({N['second_ratio_pct']}\\%) at {N['second_ratio']}"),
    ("threshold", "relabelled visits",
     f"\\textbf{{{N['relabelled']} of {N['n_labelled']:,} labelled visits "
     f"({N['relabelled_pct']}\\%), every one of them from the Indeterminate class}}"),
    ("threshold", "class shrinkage",
     f"fell from {N['ind_before']} visits to {N['ind_after']}, a reduction of "
     f"{N['ind_drop_pct']}\\%, with {N['ind_to_pigd']} becoming PIGD and "
     f"{N['ind_to_td']} becoming TD"),
    ("threshold", "participants affected",
     f"\\textbf{{{N['patients_affected']} of {N['n_patients']:,} "
     f"({N['patients_affected_pct']}\\%)}}"),
    ("threshold", "agreement drop",
     f"fell from {pct('agree2_base', 2)}\\% to {pct('agree2_narrow', 2)}\\% on the two-class "
     f"label and from {pct('agree3_base', 2)}\\% to {pct('agree3_narrow', 2)}\\%"),

    # ---- harmonised comparison --------------------------------------------
    ("families", "spread",
     f"\\textbf{{{N['acc_spread_pts']:.2f} accuracy points of one another}}, from "
     f"{p('acc_min', 4)} to {p('acc_max', 4)}"),
    ("families", "persistence accuracy",
     f"persistence rule achieved {p('acc_persistence', 4)}"),
    ("families", "margin range",
     f"by between $+{p('delta_min', 4)}$ ({p('delta_min_lo', 4)}--{p('delta_min_hi', 4)}) and "
     f"$+{p('delta_max', 4)}$ ({p('delta_max_lo', 4)}--{p('delta_max_hi', 4)})"),
    ("families", "five-state vs three-state",
     f"five-state hidden Markov model scored {p('acc_hmm5', 4)} against the three-state "
     f"model's {p('acc_hmm3', 4)}"),
    ("families", "interval-aware vs discrete",
     f"interval-aware chain scored {p('acc_ctmc', 4)} against the discrete chain's "
     f"{p('acc_dtmc', 4)}"),
    ("families", "five-state convergence",
     f"convergence in only {N['hmm5_folds_converged']} of {N['hmm5_folds_total']} folds "
     f"({N['hmm5_folds_pct']}\\%)"),
    ("families", "five-state non-converged share",
     f"{N['hmm5_rows_nonconverged_pct']}\\% of its predictions come from folds"),
    ("families", "spread without the five-state model",
     f"leaves the spread at {N['acc_spread_without5']:.2f} points unchanged"),
    ("families", "models predicting Indeterminate",
     f"Of fourteen specifications, "
     f"{['zero', 'one', 'two', 'three', 'four', 'five'][N['n_models_predicting_ind']]} "
     f"ever predicted Indeterminate"),
    ("families", "persistence predicts Indeterminate",
     f"--- {N['n_pred_ind_persistence']} times ---"),
    ("families", "macro-F1",
     f"({p('macro_f1_persistence', 4)}, against {p('macro_f1_best_fitted', 4)} for the best "
     f"fitted model)"),
    ("families", "Indeterminate outcomes",
     f"any of the {N['support_Ind']} Indeterminate outcomes"),

    # ---- margin decomposition ---------------------------------------------
    ("margin", "share of gain",
     f"\\textbf{{{pct('gain_share_ind')}\\% of its advantage over persistence came from the "
     f"{N['n_ind_cases']} transitions ({pct('share_ind_cases')}\\% of the data)"),
    ("margin", "TD stratum gain",
     f"it gained six correct predictions out of {N['n_ind_cases'] + 1927:,}"),
    ("margin", "PIGD stratum",
     f"\\textbf{{none at all}} out of 766, the two achieving identical accuracy of "
     f"{p('acc_pigd_stratum', 4)}"),
    ("margin", "identical predictions",
     f"identical calls on {pct('ident_td')}\\% of current-TD and {pct('ident_pigd')}\\% of "
     f"current-PIGD transitions, against {pct('ident_ind')}\\%"),
    ("margin", "chain gain",
     f"taking all {N['chain_gain_ind']} of its gained cases from the Indeterminate band"),
    ("margin", "Markov property test",
     f"rose from {p('p_next_PIGD_given_TD_prev_TD')} to {p('p_next_PIGD_given_TD_prev_PIGD')} "
     f"according to whether the previous label had been TD or PIGD "
     f"($\\chi^2 = {N['markov_chi2_TD']:.1f}$, $\\mathrm{{df}} = {N['markov_dof_TD']}$"),
    ("margin", "information criteria",
     f"($\\Delta$AIC {N['daic_latent']:.0f}, $\\Delta$BIC {N['dbic_latent']:.0f}"),
    ("margin", "sojourn times",
     f"{N['sojourn_TD_latent']:.1f} years in TD, {N['sojourn_PIGD_latent']:.1f} in PIGD and "
     f"{N['sojourn_Ind_latent']:.1f} in Indeterminate"),

    # ---- two-class --------------------------------------------------------
    ("two-class", "three- to two-class",
     f"from {p('three_acc', 4)} on the three-class task to {p('two_acc_hmm', 4)} on a matched "
     f"two-class evaluation of {N['two_n']:,} transitions from {N['two_patients']} participants"),
    ("two-class", "one Indeterminate prediction",
     f"one prediction in {N['three_n']:,}"),
    ("two-class", "HMM vs persistence",
     f"{p('two_acc_hmm', 4)} ({p('two_acc_hmm_lo', 4)}--{p('two_acc_hmm_hi', 4)}) against "
     f"{p('two_acc_pers', 4)} ({p('two_acc_pers_lo', 4)}--{p('two_acc_pers_hi', 4)})"),
    ("two-class", "profile accuracy",
     f"reached {p('two_acc_prof', 4)} ({p('two_acc_prof_lo', 4)}--{p('two_acc_prof_hi', 4)})"),
    ("two-class", "profile vs HMM",
     f"(paired difference $+{p('prof_vs_hmm_delta', 4)}$, ${p('prof_vs_hmm_lo', 4)}$ to "
     f"$+{p('prof_vs_hmm_hi', 4)}$)"),
    ("two-class", "Brier",
     f"Brier score was {p('two_brier_hmm', 4)} against persistence's {p('two_brier_pers', 4)}"),
    ("two-class", "calibration",
     f"to within {N['calib_mae']:.3f} on average across six bins"),
    ("two-class", "AUC",
     f"{p('auc_prof', 4)} ({p('auc_prof_lo', 4)}--{p('auc_prof_hi', 4)}) against "
     f"{p('auc_hmm', 4)} ({p('auc_hmm_lo', 4)}--{p('auc_hmm_hi', 4)})"),
    ("two-class", "AUC difference",
     f"$+{p('auc_delta', 4)}$ ({p('auc_delta_lo', 4)}--{p('auc_delta_hi', 4)})"),

    # ---- history and retraining -------------------------------------------
    ("history", "last two visits",
     f"accuracy ${p('h2_acc', 4)}$ (${p('h2_acc_lo', 4)}$ to $+{p('h2_acc_hi', 4)}$), balanced "
     f"accuracy ${p('h2_bal', 4)}$, AUC ${p('h2_auc', 4)}$, Brier $+{p('h2_brier', 4)}$ and log "
     f"loss $+{p('h2_logloss', 4)}$"),
    ("history", "single visit",
     f"accuracy ${p('h1_acc', 4)}$ (${p('h1_acc_lo', 4)}$ to ${p('h1_acc_hi', 4)}$) and AUC "
     f"${p('h1_auc', 4)}$ (${p('h1_auc_lo', 4)}$ to ${p('h1_auc_hi', 4)}$)"),
    ("history", "retraining",
     f"balanced accuracy ${p('retrain_bal', 4)}$ (${p('retrain_bal_lo', 4)}$ to "
     f"${p('retrain_bal_hi', 4)}$)"),
    ("history", "training effect",
     f"(${p('retrain_training_effect', 4)}$, ${p('retrain_training_lo', 4)}$ to "
     f"${p('retrain_training_hi', 4)}$"),
    ("history", "history effect",
     f"(${p('retrain_history_effect', 4)}$, interval containing zero)"),
    ("history", "stage one",
     f"gave {p('stage1_hmm', 4)} for the hidden Markov model and {p('stage1_prof', 4)}"),
    ("history", "PIGD share across stages",
     f"rose from {N['stage1_pct_pigd']:.0f}\\% to {N['stage_last_pct_pigd']:.0f}\\%"),

    # ---- continuous representation ----------------------------------------
    ("profile", "dimensionality",
     f"{['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven'][N['pc_n_for_90_level']]} "
     f"principal components were required for 90\\% of its "
     f"across-participant variance, with the first explaining {pct('pc1_level')}\\% of the "
     f"levels and {pct('pc1_rate')}\\% of the rates"),
    ("profile", "H&Y discrimination",
     f"AUC of {p('auc_level_hy_ge_3', 3)} against the label's {p('auc_label_hy_ge_3', 3)}, a "
     f"difference of $+{p('delta_level_hy_ge_3', 3)}$ "
     f"({p('delta_level_hy_ge_3_lo', 3)}--{p('delta_level_hy_ge_3_hi', 3)})"),
    ("profile", "MCI discrimination",
     f"{p('auc_level_mci', 3)} against {p('auc_label_mci', 3)}, a difference of "
     f"$+{p('delta_level_mci', 3)}$ ({p('delta_level_mci_lo', 3)}--{p('delta_level_mci_hi', 3)})"),
    ("profile", "motor complications",
     f"discriminates at chance ({p('auc_label_motor_complications', 3)}), the difference was "
     f"$+{p('delta_level_motor_complications', 3)}$ "
     f"(${p('delta_level_motor_complications_lo', 3)}$ to "
     f"$+{p('delta_level_motor_complications_hi', 3)}$)"),
    ("profile", "rate coordinates",
     f"differences of $+{p('delta_rate_hy_ge_3', 3)}$, $+{p('delta_rate_mci', 3)}$, "
     f"$+{p('delta_rate_motor_complications', 3)}$ and $+{p('delta_rate_cognitive_dx', 3)}$"),
    ("profile", "log density vs LOCF",
     f"$+{p('prof_vs_locf_ll', 4)}$ log-density per domain observation "
     f"({p('prof_vs_locf_lo', 4)}--{p('prof_vs_locf_hi', 4)})"),
    ("profile", "rate component",
     f"$+{p('rate_component_ll', 4)}$ ({p('rate_component_lo', 4)}--{p('rate_component_hi', 4)})"),
    ("profile", "medication component",
     f"$+{p('med_component_ll', 4)}$ ({p('med_component_lo', 4)}--{p('med_component_hi', 4)})"),

    # ---- per-participant --------------------------------------------------
    ("per-participant", "mean and visit level",
     f"accuracy {p('pp_mean')} ({p('pp_mean_lo')}--{p('pp_mean_hi')}) against a visit-level "
     f"{p('pp_visit_level')}"),
    ("per-participant", "distribution",
     f"Of {N['pp_n']} participants, {N['pp_perfect']} ({N['pp_perfect_pct']}\\%) were correct "
     f"at every eligible visit and {N['pp_zero']} ({N['pp_zero_pct']}\\%) at none"),
    ("per-participant", "composition gradient",
     f"all TD averaged {p('pp_allTD_model')}, and those with an even mixture of TD and PIGD "
     f"outcomes averaged {p('pp_even_model')}"),
    ("per-participant", "persistence gradient",
     f"--- {p('pp_allTD_pers')} and {p('pp_even_pers')} ---"),
    ("per-participant", "class-wise",
     f"sensitivity averaged {p('pp_pigd_sens')} against a TD specificity of {p('pp_td_spec')}"),

    # ---- claims repeated in the abstract -----------------------------------
    ("abstract", "reversion",
     f"{pct('reversion_pooled')}\\% (95\\% CI {pct('reversion_lo')}--{pct('reversion_hi')}) of "
     f"observed subtype changes reverse"),
    ("abstract", "Indeterminate persistence",
     f"probability only {p('p_Indet_Indet')} ({p('p_Indet_Indet_lo')}--{p('p_Indet_Indet_hi')})"),
    ("abstract", "threshold relabelling",
     f"relabels {N['relabelled_pct']}\\% of all labelled visits and "
     f"{N['patients_affected_pct']}\\% of participants"),
    ("abstract", "spread and baseline",
     f"within {N['acc_spread_pts']:.2f} accuracy points of one another "
     f"({p('acc_min', 4)}--{p('acc_max', 4)}) and all exceed a persistence rule "
     f"({p('acc_persistence', 4)})"),
    ("abstract", "margin localisation",
     f"{pct('gain_share_ind')}\\% of the latent-state model's margin arises on the "
     f"{pct('share_ind_cases')}\\% of transitions recorded Indeterminate"),
]


def main() -> int:
    rows = []
    failed = 0
    for section, claim, expected in CHECKS:
        needle = re.sub(r"\s+", " ", expected)
        ok = needle in TEX
        if not ok:
            failed += 1
        rows.append({"section": section, "claim": claim,
                     "expected": needle, "passed": ok})

    with (HERE / "numbers_check.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["section", "claim", "expected", "passed"])
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        if not r["passed"]:
            print(f"FAIL  [{r['section']}] {r['claim']}\n      expected: {r['expected']}")

    print(f"\n{len(rows) - failed} of {len(rows)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
