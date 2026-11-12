# Models exceeding 80% accuracy

PPMI motor-subtype progression · next-visit prediction · 11 November 2026
Source: `experiments/20261111T170322Z_final_consolidated_results` · frozen panel md5 `fb038ddbdc19ab3368e13940f398e7ba`

## Read this first

**55 of 64** model-and-cohort arms exceed 80% accuracy on the
binary TD-versus-PIGD task. That is not a discriminating result, for three
reasons, and the threshold should not be used to choose a model:

1. **The trivial baseline is already 69.9%.** Always predicting TD on the
   matched set scores that, because PIGD is the minority class. Clearing 80%
   therefore means beating a constant rule by 10.1% or more, not that a
   model is good.
2. **No two leading models are separated by accuracy.** Every paired contrast
   between two serious models has a 95% confidence interval that includes zero.
   The models below are statistically indistinguishable from one another on this
   metric.
3. **Accuracy is blind to the Indeterminate class.** On the three-class task
   **no model reaches 80%** — the highest is 0.7602
   (Profile: level only (random walk)). Every fitted model has an
   Indeterminate sensitivity of essentially zero.

Accuracy is reported here because it was asked for. AUC-ROC is the column that
actually separates these models, and it is included beside it.

**Cohorts are not interchangeable.** Each block below is one evaluation cohort
and one protocol. Ranking across blocks is not valid.

## Matched comparison set

Both families on the same transitions. The headline cross-family table.  
**10 models above 80%** · 3,020 transitions · 898 patients · profile arm: 5-fold patient CV, repeat 0

| Model | Accuracy [95% CI] | Sensitivity (PIGD) | Specificity (TD) | AUC-ROC |
|---|---|---|---|---|
| Profile: level only (random walk) | 0.8440 [0.8285, 0.8598] | 0.6769 | 0.9161 | 0.8865 |
| Profile: level + rate + medication | 0.8404 [0.8248, 0.8566] | 0.6813 | 0.9090 | 0.8902 |
| Profile: level + rate, no medication | 0.8404 [0.8249, 0.8563] | 0.6769 | 0.9109 | 0.8903 |
| CT-HMM (latent, interval-aware) | 0.8371 [0.8202, 0.8540] | 0.6681 | 0.9100 | 0.8583 |
| Direct logistic (history + interval) | 0.8361 [0.8185, 0.8530] | 0.6670 | 0.9090 | 0.8564 |
| Persistence (last non-Indeterminate) | 0.8338 [0.8167, 0.8509] | 0.6231 | 0.9246 | 0.7927 |
| Markov (observed, discrete) | 0.8298 [0.8130, 0.8471] | 0.5901 | 0.9332 | 0.6777 |
| CTMC (observed, interval-aware) | 0.8281 [0.8105, 0.8455] | 0.5813 | 0.9346 | 0.8043 |
| Last observation carried forward | 0.8252 [0.8083, 0.8425] | 0.6659 | 0.8938 | 0.8780 |
| Direct logistic (history only) | 0.8195 [0.8020, 0.8371] | 0.6462 | 0.8943 | 0.8377 |

## Harmonised, common binary subset

Fourteen models under one protocol on identical rows.  
**12 models above 80%** · 2,759 transitions · 852 patients · leave-one-patient-out, one protocol for all 14 models

| Model | Accuracy [95% CI] | Sensitivity (PIGD) | Specificity (TD) | AUC-ROC |
|---|---|---|---|---|
| Subscore GP, independent | 0.8637 [0.8482, 0.8794] | 0.7155 | 0.9233 | 0.9006 |
| Subscore GP, joint | 0.8637 [0.8479, 0.8792] | 0.7168 | 0.9228 | 0.9000 |
| CT-HMM, 3 latent states | 0.8594 [0.8425, 0.8758] | 0.6941 | 0.9258 | 0.8626 |
| CT-HMM, 5 latent states | 0.8590 [0.8426, 0.8755] | 0.6991 | 0.9233 | 0.8652 |
| Persistence (subscore LOCF) | 0.8572 [0.8407, 0.8738] | 0.6802 | 0.9284 | 0.8844 |
| Persistence (recorded label) * | 0.8572 [0.8407, 0.8738] | 0.6802 | 0.9284 | 0.8043 |
| Markov chain, discrete | 0.8572 [0.8407, 0.8738] | 0.6802 | 0.9284 | 0.6464 |
| CTMC, interval-aware | 0.8554 [0.8386, 0.8722] | 0.6700 | 0.9299 | 0.8023 |
| Joint subscore LMM | 0.8547 [0.8370, 0.8716] | 0.6764 | 0.9263 | 0.8878 |
| Time-varying LMM, constant | 0.8547 [0.8370, 0.8716] | 0.6764 | 0.9263 | 0.8878 |
| Time-varying LMM, spline | 0.8492 [0.8306, 0.8672] | 0.6485 | 0.9299 | 0.8876 |
| Patient mean subscore | 0.8456 [0.8277, 0.8639] | 0.6068 | 0.9416 | 0.8768 |

## Two-state hmm family

Hidden-state count comparison; its own common set.  
**5 models above 80%** · 2,936 transitions · 876 patients · leave-one-patient-out

| Model | Accuracy [95% CI] | Sensitivity (PIGD) | Specificity (TD) | AUC-ROC |
|---|---|---|---|---|
| CT-HMM (2 hidden x 2 observed, Indeterminate excluded) | 0.8478 [0.8318, 0.8637] | 0.6794 | 0.9183 | 0.8530 |
| CT-HMM (2 hidden x 2 observed, no emission floor) | 0.8478 [0.8318, 0.8637] | 0.6794 | 0.9183 | 0.8530 |
| CT-HMM (3 hidden x 3 observed, existing model) | 0.8467 [0.8305, 0.8625] | 0.6874 | 0.9135 | 0.8599 |
| CT-HMM (2 hidden x 3 observed) | 0.8464 [0.8304, 0.8625] | 0.6897 | 0.9120 | 0.8583 |
| Persistence (last observed label) * | 0.8443 [0.8276, 0.8609] | 0.6563 | 0.9232 | 0.7897 |

## Ratio-band sweep

Ratio bands against the unbanded CT-HMM, bands refitted per fold.  
**6 models above 80%** · 3,029 transitions · 899 patients · leave-one-patient-out, bands refitted per fold

| Model | Accuracy [95% CI] | Sensitivity (PIGD) | Specificity (TD) | AUC-ROC |
|---|---|---|---|---|
| CT-HMM (latent, interval-aware) | 0.8369 [0.8204, 0.8541] | 0.6674 | 0.9102 | 0.8587 |
| CT-HMM (subtype x ratio band, width 0.30) | 0.8359 [0.8192, 0.8535] | 0.6980 | 0.8955 | 0.8784 |
| CT-HMM (subtype x ratio band, width 0.05) | 0.8336 [0.8169, 0.8513] | 0.7090 | 0.8875 | 0.8739 |
| CT-HMM (subtype x ratio band, width 0.20) | 0.8336 [0.8167, 0.8516] | 0.6991 | 0.8917 | 0.8775 |
| CT-HMM (subtype x ratio band, width 0.50) | 0.8320 [0.8152, 0.8507] | 0.7188 | 0.8809 | 0.8765 |
| CT-HMM (subtype x ratio band, width 0.10) | 0.8316 [0.8142, 0.8498] | 0.7079 | 0.8851 | 0.8758 |

## Two further blocks, not repeated above

The rich-profile family on its own cohort (4 models above 80%, 3,021 transitions) and the subtype-history family on its own (6 models, 3,029 transitions) give the
same models at the same values: the largest accuracy difference against the
matched set above, across all 12 shared arms, is 0.0005. The matched
set is the comparable version of both, so they are not tabulated again.

## Notes

- Positive class is PIGD throughout. Sensitivity = TP/(TP+FN),
  specificity = TN/(TN+FP).
- Confidence intervals are patient-level bootstrap, 2,000 draws, seed 42.
  Patients are resampled, not visits.
- `*` marks a rule-based score taking only two distinct values; its AUC is
  algebraically (sensitivity + specificity)/2 and measures no ranking.
- Every value is recomputed from the per-case predictions saved by the
  experiment that produced them. All 359 comparisons against previously
  published values agree to within 1e-4.
- The two identical 2x2 rows are one model: the emission floor makes no
  difference to its predictions.

Full report, all metrics and the validation audit:
`docs/consolidated_results/03_FINAL_CONSOLIDATED_RESULTS_20261111.md`
