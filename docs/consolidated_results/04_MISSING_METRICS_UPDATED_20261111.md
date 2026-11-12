# Missing-Metrics Register, Updated

Run `experiments/20261111T170322Z_final_consolidated_results` · 2026-11-11 · supersedes
`docs/consolidated_results/02_MISSING_METRICS_20261111.md`

1269 metric values that were previously absent have been computed from saved
predictions and are listed in `recovered_metrics.csv`. What follows is what is
still absent, and why. **Not calculated** means it could exist but does not;
**Not applicable** means it is undefined for that model or task.

## A. Recovered in this pass, no longer missing

| Metric family | Arms | How |
|---|---|---|
| auc_ovr_Indeterminate | 36 | one-vs-rest AUC on Indeterminate's probability |
| auc_ovr_PIGD | 36 | one-vs-rest AUC on PIGD's probability |
| auc_ovr_TD | 36 | one-vs-rest AUC on TD's probability |
| auc_roc | 56 | Mann-Whitney statistic on P(PIGD), mid-ranks for ties |
| balanced_accuracy | 58 | (sensitivity + specificity) / 2 |
| brier_binary | 38 | mean (p_PIGD - y_PIGD)^2 |
| calibration_in_the_large | 64 | mean(p) - mean(y) |
| calibration_intercept | 61 | intercept of the same fit; 0 is calibrated |
| calibration_slope | 61 | slope of a logistic regression of the outcome on logit(p), fitted by Newton steps; 1 is calibrated |
| ece_10_bin | 64 | support-weighted mean |mean(p) - mean(y)| over 10 equal-width probability bins |
| f1_Indeterminate | 32 | one-vs-rest F1 for Indeterminate |
| f1_PIGD | 90 | harmonic mean of precision and sensitivity |
| f1_TD | 36 | one-vs-rest F1 for TD |
| log_loss | 10 | -mean log p(true class), clipped at 1e-12 |
| macro_auc_ovr | 36 | unweighted mean of the three one-vs-rest AUCs |
| mean_predictive_loglik | 36 | mean log probability assigned to the realised label |
| pr_auc | 56 | average precision on P(PIGD), thresholded at distinct scores |
| pr_auc_ovr_Indeterminate | 36 | one-vs-rest average precision on Indeterminate's probability |
| pr_auc_ovr_PIGD | 36 | one-vs-rest average precision on PIGD's probability |
| pr_auc_ovr_TD | 36 | one-vs-rest average precision on TD's probability |
| precision_Indeterminate | 18 | one-vs-rest TP / (TP + FP) for Indeterminate |
| precision_PIGD | 49 | TP / (TP + FP) |
| precision_TD | 36 | one-vs-rest TP / (TP + FP) for TD |
| sensitivity_Indeterminate | 18 | one-vs-rest recall for Indeterminate |
| sensitivity_PIGD | 10 | TP / (TP + FN), PIGD positive |
| sensitivity_TD | 18 | one-vs-rest recall for TD |
| specificity_Indeterminate | 36 | one-vs-rest TN / (TN + FP) for Indeterminate |
| specificity_PIGD | 36 | one-vs-rest TN / (TN + FP) for PIGD |
| specificity_TD | 62 | TN / (TN + FP) |
| weighted_auc_ovr | 36 | support-weighted mean of the three one-vs-rest AUCs |
| weighted_f1 | 36 | support-weighted mean of the three one-vs-rest F1s |

## B. Still missing, with the reason

No cell in `master_results_complete.csv` now reads **Not calculated**. Every remaining gap is either undefined for that model, and marked **Not applicable** with the reason in the cell itself, or needs work that no saved output can supply, listed in section C.

## C. Never run, and not recoverable from any saved output

### C1. A separately trained binary Indeterminate-versus-other model

Section 8 of the report reads Indeterminate one-versus-rest out of the saved
three-class probabilities, which is legitimate and informative, but it is not a
model trained for that question. A dedicated binary experiment would relabel
every visit as Indeterminate or not and fit the same CT-HMM, chain and profile
families to it.

Baseline to beat, so the result is not misread: a constant "never
Indeterminate" rule scores 0.9012 accuracy on this cohort, with
sensitivity 0. Report balanced accuracy, PR-AUC and the full confusion matrix,
not accuracy.

### C2. A single protocol across every family

The rich profile uses 5-fold patient cross-validation with 4 repeats; the
subtype-history families use leave-one-patient-out. The matched set in Table A
removes the cohort difference but not the protocol difference. Settling it
would mean rerunning one family under the other's protocol.

### C3. External validation

No second cohort exists in this project. Every number here is internal
cross-validation on one frozen panel.

### C4. Reliability diagrams

The calibration slope, intercept, expected calibration error and
calibration-in-the-large are computed for every binary arm with a
non-degenerate score and are in the master table. What is not produced here is
the reliability diagram itself, bin by bin; the ten-bin summary is reported
instead of the curve.

## D. Values that exist but must not be used

- The band model's whole-cohort log-likelihood of −13123.4023 with 137
  parameters, and the AIC of 26520.80 derived from it. The run that produced it
  labels it a fit diagnostic over a 44-category alphabet, not comparable to the
  3-label baseline's −2969.0593. Do not place them in one table.
- The rich profile's AIC near 91,700 next to any CT-HMM's near 5,966. Density
  against probability mass.
- Any Brier comparison that mixes the single-outcome and summed conventions.
- The marginal arms' AUC, as a discrimination estimate.
