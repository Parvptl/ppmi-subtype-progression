# Validation Audit

Run `experiments/20261111T170322Z_final_consolidated_results` · 2026-11-11 · panel md5 `fb038ddbdc19ab3368e13940f398e7ba`

Every metric reported in `FINAL_CONSOLIDATED_RESULTS.md` is either read from a
published table or computed here from per-case predictions. This document
records how each was checked.

## 1. Structural checks on the prediction files

| Check | Status | Detail |
|---|---|---|
| profile: three-class probabilities sum to 1 | PASS | min 1.000000000000 max 1.000000000000 |
| profile: no missing probabilities | PASS | 0 missing cells |
| LOPO: three-class probabilities sum to 1 | PASS | min 1.000000000000 max 1.000000000000 |
| LOPO: no missing probabilities | PASS | 0 missing cells |
| band: three-class probabilities sum to 1 | PASS | min 1.000000000000 max 1.000000000000 |
| band: no missing probabilities | PASS | 0 missing cells |
| two-state: three-class probabilities sum to 1 on the three-class arms | PASS | min 1.000000000000 max 1.000000000000 |
| two-state: no missing probabilities on the three-class arms | PASS | 0 missing cells |
| two-state: missing Indeterminate probabilities are confined to the 2x2 arms | PASS | 5872 cells, all in the two 2x2 arms, which have no Indeterminate output by construction |
| two-state: the 2x2 arms' two probabilities sum to 1 | PASS | min 1.000000000000 max 1.000000000000 |
| harmonised: all 14 models' probabilities sum to 1 | PASS | min 1.000000000000 max 1.000000000000 |
| profile: no duplicate predictions | PASS | 0 duplicate key rows |
| LOPO: no duplicate predictions | PASS | 0 duplicate key rows |
| band: no duplicate predictions | PASS | 0 duplicate key rows |
| two-state: no duplicate predictions | PASS | 0 duplicate key rows |
| harmonised: no duplicate rows | PASS | 0 duplicate key rows |
| profile: label set | PASS | ['Indeterminate', 'PIGD', 'TD'] |
| LOPO: label set | PASS | ['Indeterminate', 'PIGD', 'TD'] |
| band: label set | PASS | ['Indeterminate', 'PIGD', 'TD'] |
| two-state: label set | PASS | ['Indeterminate', 'PIGD', 'TD'] |
| harmonised: label set | PASS | ['Indeterminate', 'PIGD', 'TD'] |
| LOPO: P_PIGD equals p_PIGD_raw renormalised over TD+PIGD | PASS | confirms the two-class convention reused below for the harmonised run |
| LOPO and harmonised share the same three-class support | PASS | TD: 2115 vs 2115, Indeterminate: 332 vs 332, PIGD: 914 vs 914 |
| binary group 'A: rich profile, own cohort' has one cohort size | PASS | sizes [3021] |
| binary group 'B: subtype-history models, LOPO' has one cohort size | PASS | sizes [3029] |
| binary group 'C: matched comparison set' has one cohort size | PASS | sizes [3020] |
| binary group 'D: ratio-band sweep' has one cohort size | PASS | sizes [3029] |
| binary group 'E: two-state HMM family' has one cohort size | PASS | sizes [2936] |
| binary group 'F: harmonised subscore family' has one cohort size | FAIL | sizes [2759, 3029] - rows in this group are NOT ranked together |
| binary group 'F2: harmonised, common binary subset' has one cohort size | PASS | sizes [2759] |
| three-class group 'A: rich profile, own cohort' has one cohort size | PASS | sizes [3353] |
| three-class group 'B: subtype-history models, LOPO' has one cohort size | PASS | sizes [3361] |
| three-class group 'D: ratio-band sweep' has one cohort size | PASS | sizes [3361] |
| three-class group 'E: two-state HMM family' has one cohort size | PASS | sizes [3361] |
| three-class group 'F: harmonised subscore family' has one cohort size | PASS | sizes [3361] |
| AIC reproduces -2logL + 2k for every tabulated model | PASS | 13/13 within 0.02 |
| BIC reproduces -2logL + k log n for every tabulated model | PASS | 13/13 within 0.02 |
| continuous-endpoint metrics reproduce across the two profile runs | PASS | 20/20 cells agree |

1 of 38 failed.

## 2. Agreement with published values

359 of 359 comparisons agree to within 1e-4.

| Published table | Metrics compared | Match | Other |
|---|---|---|---|
| experiments/20261020T104756Z_next_visit_td_pigd_prediction/model_comparison.csv | 28 | 28 | 0 |
| experiments/20261023T202640Z_roc_auc_comparison/auc_metrics.csv | 6 | 6 | 0 |
| experiments/20261103T202632Z_e8_harmonised_family_comparison/harmonised_metrics.csv | 140 | 140 | 0 |
| experiments/20261105T025024Z_cthmm_ratio_band_width_extended/model_metrics.csv | 36 | 36 | 0 |
| experiments/20261110T143200Z_two_state_hmm_confirm/common_two_class.csv | 47 | 47 | 0 |
| experiments/20261110T143200Z_two_state_hmm_confirm/native_three_class.csv | 43 | 43 | 0 |
| experiments/20261111T133201Z_consolidated_results/two_class_sensitivity_specificity.csv | 59 | 59 | 0 |

Non-matching rows, if any:

| Source | Model | Metric | Published | Recovered | Status |
|---|---|---|---|---|---|
| none |  |  |  |  |  |

## 3. Metric implementations, checked against scikit-learn

| Quantity | Implementation | Agreement |
|---|---|---|
| AUC-ROC | Mann-Whitney with mid-ranks for ties | 1.1e-16 max absolute difference over 400 random, tied, binary and constant score vectors |
| PR-AUC | average precision, thresholded at distinct scores | 2.2e-16 over the same plus constant-score cases |
| F1, balanced accuracy, Brier, log loss | direct formulae | exact to double precision |
| macro-F1, weighted-F1, macro AUC, weighted AUC, multiclass log loss | direct formulae | exact to double precision |
| Calibration slope and intercept | logistic regression of the outcome on logit(p), Newton steps to 1e-10 | 3.0e-12 max absolute difference against scikit-learn's `newton-cg` at tol 1e-12 over 60 distorted-probability datasets; a perfectly calibrated score returns slope 1.006 and intercept -0.004 at n = 200,000 |
| ECE, MCE, calibration-in-the-large | 10 equal-width probability bins | direct formulae |

The tie handling matters: a per-row cumulative average precision disagrees with
scikit-learn by up to 0.13 on tied scores, and several arms here emit
rule-based probabilities with only two distinct values.

## 4. Accuracy recomputation

Every arm's accuracy was recomputed from its own per-case predictions and
compared with the published value where one exists. All
49
published accuracies reproduce.

## 5. Information criteria recomputation

AIC = −2logL + 2k and BIC = −2logL + k log n were recomputed from the stored
log-likelihood, parameter count and observation count for all 13 rows.

| Model | Stored AIC | Recomputed | Stored BIC | Recomputed | Observations | Parameters |
|---|---|---|---|---|---|---|
| CT-HMM (3 latent states) | 5966.12 | 5966.12 | 6055.23 | 6055.23 | 4,295 | 14 |
| CTMC (interval-aware) | 4773.00 | 4773.00 | 4809.72 | 4809.72 | 3,361 | 6 |
| Markov (discrete) | 4747.55 | 4747.54 | 4784.27 | 4784.26 | 3,361 | 6 |
| CTMC + initial state | 6215.68 | 6215.68 | 6266.60 | 6266.60 | 4,295 | 8 |
| Markov + initial state | 6190.23 | 6190.24 | 6241.15 | 6241.16 | 4,295 | 8 |
| CT-HMM (3 latent states) | 5966.12 | 5966.12 | 6055.23 | 6055.23 | 4,295 | 14 |
| Profile: level + rate + medication | 91714.73 | 91714.72 | 92099.84 | 92099.83 | 38,486 | 45 |
| Profile: level only (random walk) | 91935.53 | 91935.54 | 92166.60 | 92166.61 | 38,486 | 27 |
| Profile: level + rate, no medication | 92058.68 | 92058.68 | 92366.77 | 92366.77 | 38,486 | 36 |
| CT-HMM (2 hidden x 2 observed, Indeterminate excluded) | 3237.49 | 3237.49 | 3268.72 | 3268.72 | 3,812 | 5 |
| CT-HMM (2 hidden x 2 observed, no emission floor) | 3237.49 | 3237.49 | 3268.72 | 3268.72 | 3,812 | 5 |
| CT-HMM (2 hidden x 3 observed) | 6035.39 | 6035.39 | 6079.95 | 6079.95 | 4,295 | 7 |
| CT-HMM (3 hidden x 3 observed, existing model) | 5966.12 | 5966.12 | 6055.23 | 6055.23 | 4,295 | 14 |

## 6. Cohort verification

| Count | n | Meaning |
|---|---|---|
| Frozen panel visits | 5,774 | panel md5 fb038ddbdc19ab3368e13940f398e7ba |
| Visits with a derivable subtype label | 4,441 | ratio of tremor to PIGD means defined |
| Patients with at least two labelled visits | 934 | the modelling cohort |
| Labelled visits in those patients | 4,295 | likelihood terms for the label-sequence models |
| Consecutive-visit transitions | 3,361 | the three-class prediction cohort |
| Transitions with a TD or PIGD target | 3,029 | binary cohort for the history models |
| Transitions also having complete clinical measurements | 3,353 | the rich profile's prediction pairs |
| Transitions with a TD/PIGD target in the profile frame | 3,021 | binary cohort for the rich profile |
| Transitions scored by both families | 3,020 | the matched comparison set |
| Transitions scorable by every two-state arm | 2,936 | the two-state common set |

Group-level cohort consistency is checked explicitly: every arm inside a group
must share one cohort size, or the group is flagged and never ranked. One group
failed that check as first constructed — the harmonised binary view — and group
F2 was added as its comparable counterpart.

## 7. Conventions pinned down

- **Positive class**: PIGD throughout the binary task.
- **Brier**: both conventions present in the repository; each cell labelled.
  The summed form is exactly twice the single-outcome form for two classes.
- **Log loss**: true-class probability clipped at 1e-12, matching
  `src/ppmi_subtype/next_visit.py:118` and
  `src/ppmi_subtype/two_state_hmm.py:145`. Arms where clipping is active are
  flagged; for them the unclipped value is infinite.
- **Two-class renormalisation**: P(PIGD) = p_PIGD / (p_TD + p_PIGD), verified
  against the leave-one-patient-out run's own column to 1e-9 before being
  applied to the harmonised run.
- **Three-class decision rule**: argmax of the saved probabilities, verified
  against the harmonised and two-state runs' own predicted-class counts.
- **Confusion-matrix orientation**: no stored matrix was used. Every per-class
  count was recomputed from per-case predictions.
- **Bootstrap**: patients resampled with replacement, 2,000 draws, seed 42,
  percentile interval. Visits are never treated as independent.
- **ECE intervals**: expected calibration error is a mean of absolute
  deviations, so resampling inflates it and its percentile range can lie
  wholly above the point estimate. The range is reported as the bootstrap
  distribution of the statistic, with a flag in the master table
  (`ece_bootstrap_range_excludes_point_estimate`) wherever that happens, and is
  never described as a confidence interval for a true ECE.
- **Paired contrasts**: both arms restricted to identical rows and verified
  row-aligned before differencing; a misaligned pair is skipped, not silently
  compared.

## 8. What was not altered

No experiment directory was modified. No model was refitted. The frozen panel
was not touched. Published values are reported as published; where this pass
disagrees or adds nuance, both readings appear, in section 13.2 of the report.
