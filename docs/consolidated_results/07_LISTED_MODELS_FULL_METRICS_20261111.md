# Listed models: accuracy, sensitivity, specificity, AUC-ROC, AIC, BIC

PPMI motor-subtype progression · next-visit TD vs PIGD · 11 November 2026  
Source: `experiments/20261111T181740Z_listed_models_full_metrics` · frozen panel md5 `fb038ddbdc19ab3368e13940f398e7ba`

Positive class is PIGD. Sensitivity = TP/(TP+FN), specificity = TN/(TN+FP). Brackets are 95% patient-level bootstrap intervals (2,000 draws, seed 42). AIC and BIC come from whole-cohort fits and **may only be compared inside the same comparison group** (letters in the last column; the key is below).

## CT-HMM

10-fold patient CV · 3,029 cases · 899 patients

| Model | Accuracy | Sensitivity (PIGD) | Specificity (TD) | AUC-ROC | AIC | BIC | Group |
|---|---|---|---|---|---|---|---|
| CT-HMM, 3 states, floor 0.5 | 83.72% [82.1, 85.4] | 66.85% [63.1, 70.1] | 91.02% [89.5, 92.6] | 0.8613 [0.843, 0.879] | 5,966.12 | 6,055.23 | L |
| CT-HMM, 5 states, no floor | 83.69% [82.1, 85.4] | 66.63% [62.9, 69.8] | 91.06% [89.6, 92.6] | 0.8638 [0.846, 0.881] | 5,945.65 | 6,162.06 | L |
| CT-HMM, 3 states, no floor | 83.66% [82.0, 85.4] | 64.88% [60.9, 68.3] | 91.77% [90.4, 93.2] | 0.8636 [0.846, 0.881] | 5,938.64 | 6,027.75 | L |

## Rich profile

5-fold patient CV · matched 3,020 cases · 898 patients

| Model | Accuracy | Sensitivity (PIGD) | Specificity (TD) | AUC-ROC | AIC | BIC | Group |
|---|---|---|---|---|---|---|---|
| Profile: level only | 84.40% [82.8, 86.0] | 67.69% [64.1, 71.0] | 91.61% [90.3, 92.9] | 0.8865 [0.870, 0.901] | 91,935.53 | 92,166.60 | D |
| Profile: level + rate + medication | 84.04% [82.5, 85.7] | 68.13% [64.5, 71.4] | 90.90% [89.5, 92.3] | 0.8902 [0.875, 0.905] | 91,714.73 | 92,099.84 | D |
| Profile: level + rate, no medication | 84.04% [82.5, 85.6] | 67.69% [64.0, 70.9] | 91.09% [89.7, 92.5] | 0.8903 [0.875, 0.905] | 92,058.68 | 92,366.77 | D |

## Two-state CT-HMM

leave-one-patient-out · 2,936 cases · 876 patients

| Model | Accuracy | Sensitivity (PIGD) | Specificity (TD) | AUC-ROC | AIC | BIC | Group |
|---|---|---|---|---|---|---|---|
| Model 1 (2x2) | 84.78% [83.2, 86.4] | 67.94% [64.2, 71.4] | 91.83% [90.5, 93.2] | 0.8530 [0.834, 0.870] | 3,237.49 | 3,268.72 | T |
| Model 2 (2x3) | 84.64% [83.0, 86.2] | 68.97% [65.3, 72.3] | 91.20% [89.8, 92.7] | 0.8583 [0.840, 0.875] | 6,035.39 | 6,079.95 | L |
| Existing (3x3) | 84.67% [83.1, 86.2] | 68.74% [65.0, 72.1] | 91.35% [89.9, 92.8] | 0.8599 [0.841, 0.877] | 5,966.12 | 6,055.23 | L |

## CT-HMM with ratio bands

leave-one-patient-out · 3,029 cases · 899 patients

| Model | Accuracy | Sensitivity (PIGD) | Specificity (TD) | AUC-ROC | AIC | BIC | Group |
|---|---|---|---|---|---|---|---|
| no bands | 83.69% [82.0, 85.4] | 66.74% [62.9, 70.1] | 91.02% [89.5, 92.6] | 0.8587 [0.840, 0.876] | 5,966.12 | 6,055.23 | L |
| 0.05 | 83.36% [81.7, 85.1] | 70.90% [67.4, 74.0] | 88.75% [86.8, 90.7] | 0.8739 [0.856, 0.891] | — | — | — |
| 0.10 | 83.16% [81.4, 85.0] | 70.79% [67.1, 74.0] | 88.51% [86.5, 90.5] | 0.8758 [0.859, 0.893] | — | — | — |
| 0.20 | 83.36% [81.7, 85.2] | 69.91% [66.0, 73.1] | 89.17% [87.3, 91.1] | 0.8775 [0.860, 0.894] | — | — | — |
| 0.30 | 83.59% [81.9, 85.4] | 69.80% [66.0, 73.0] | 89.55% [87.7, 91.4] | 0.8784 [0.862, 0.895] | — | — | — |
| 0.50 | 83.20% [81.5, 85.1] | 71.88% [68.3, 74.9] | 88.09% [86.1, 90.1] | 0.8765 [0.860, 0.894] | — | — | — |

## AIC and BIC comparison groups

- **L**: joint likelihood over label sequences, all 4,295 labelled visits, 3-label alphabet. Directly comparable across CT-HMM rows, Model 2 (2×3), Existing (3×3) and "no bands".
- **T**: same likelihood restricted to 3,812 TD/PIGD visits. Model 1 (2×2) alone; different data, so not comparable with L.
- **D**: density over 9 continuous clinical domains, 38,486 observations. Rich-profile rows only. A density is not on the scale of a probability mass, so never compare D with L.
- **—**: ratio-band widths. Each width's likelihood is over its own joint (subtype × band) alphabet, from 20 to 73 categories, so its AIC is comparable with nothing, not even another width.

Inside group L, lower is better:

| Model | Log-likelihood | Parameters | AIC | ΔAIC | BIC | ΔBIC |
|---|---|---|---|---|---|---|
| CT-HMM, 3 states, no floor | -2,955.32 | 14 | 5,938.64 | +0.00 | 6,027.75 | +0.00 |
| CT-HMM, 5 states, no floor | -2,938.82 | 34 | 5,945.65 | +7.00 | 6,162.06 | +134.31 |
| CT-HMM, 3 states, floor 0.5 | -2,969.06 | 14 | 5,966.12 | +27.48 | 6,055.23 | +27.48 |
| Model 2 (2x3) | -3,010.70 | 7 | 6,035.39 | +96.75 | 6,079.95 | +52.20 |

"Existing (3×3)" and "no bands" are the same fitted model as "CT-HMM, 3 states, floor 0.5", so it appears here once.

## Notes

- The three CT-HMM floor variants come from the 10-fold HMM configuration study, which published three-class accuracy only. Their sensitivity, specificity and AUC-ROC are newly computed from that study's saved per-case predictions, by renormalising the TD and PIGD probabilities. All three reproduce their published three-class accuracy exactly.
- Their AIC and BIC are read from whole-cohort fits with 40 random starts each. The stored values reproduce −2logL + 2k and −2logL + k·ln(4,295) exactly.
- The 5-state fit is multimodal: only 1 of its 40 starts reached the best optimum. A longer optimiser run found a log-likelihood 0.0021 higher, which is the value used. Treat its AIC/BIC as approximate.
- Accuracy differences between models are within noise: every accuracy interval overlaps every other one. AUC-ROC is what separates them.
- AIC/BIC measure in-sample fit, not prediction. In group L they prefer the 3-state no-floor model, while held-out accuracy and AUC-ROC cannot separate it from the floored one.
