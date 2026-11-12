# Final Consolidated Results: PPMI Motor-Subtype Progression Models

Run directory `experiments/20261111T170322Z_final_consolidated_results` · generated 2026-11-11 · frozen panel md5 `fb038ddbdc19ab3368e13940f398e7ba`

## 1. What this document is

This is a consolidation and completion of experiments that were already run.
No model was fitted for it. Every number below is either read from the output
of the experiment that produced it, or computed here from that experiment's
saved per-case predictions, on that experiment's own evaluation cohort and
under its own protocol.

1269 metric values that had never been computed were recovered this way,
alongside 323 that reproduce a published value. Of the recovered-and-published
overlap, 359 of 359 comparisons against the published
tables agree to within 1e-4. Of the 38 structural checks on the
prediction files, 37 passed; the 1 that did not is not a
data fault but a cohort inconsistency this pass uncovered, described in
section 13.2.

Two deliberate omissions: AUC-ROC and PR-AUC are never derived from an accuracy
or a confusion matrix, and no model was refitted to fill a cell. Where a metric
is genuinely unobtainable it is marked **Not calculated** or **Not applicable** and the reason is
given in section 14 and in `02_MISSING_METRICS_UPDATED.md`.

## 2. Evaluation cohorts

Results are grouped by evaluation protocol. **Numbers from different groups are
not comparable** and are never ranked against each other.

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

The arithmetic closes: 4,441 − 4,295 = 146 single-visit patients, which is also
1,080 − 934; and 4,441 − 4,433 = 8 visits lacking clinical measurements, which
is also 4,295 − 4,287 and 3,361 − 3,353.

| Group | Protocol | Cohort | Models |
|---|---|---|---|
| A | 5-fold patient CV x 4 repeats, repeat 0 reported | 3,021 transitions with a defined TD/PIGD target | 5 |
| F2 | leave-one-patient-out, one protocol for all 14 models | 2,759 transitions scorable by all 14 models | 14 |
| B | leave-one-patient-out, 934 folds | 3,029 transitions with a defined TD/PIGD target | 7 |
| C | profile arm: 5-fold patient CV, repeat 0 | 3,020 transitions scored by both families | 12 |
| D | leave-one-patient-out, bands refitted per fold | 3,029 transitions with a defined TD/PIGD target | 6 |
| E | leave-one-patient-out | 2,936 transitions scorable by every arm | 6 |
| F | leave-one-patient-out, one protocol for all 14 models | each model on every case it can score (sizes differ, see notes) | 14 |

## 3. Metric definitions and the two conventions that trap readers

Positive class for every binary figure is **PIGD**. Sensitivity = TP/(TP+FN),
specificity = TN/(TN+FP), precision = TP/(TP+FP), F1 is their harmonic mean,
balanced accuracy = (sensitivity + specificity)/2. For the three-class task
each class is taken one-versus-rest in turn; macro-F1 is the unweighted mean of
the three F1s, weighted-F1 weights by support, balanced accuracy is the mean of
the three recalls.

AUC-ROC is the Mann-Whitney statistic on the model's probability with mid-ranks
for ties. For the three-class task it is computed one-versus-rest per class;
**macro AUC** is the unweighted mean of those three and **weighted AUC** weights
them by class support (2,115 / 332 / 914). No single multiclass AUC is quoted
without naming its averaging convention, and neither average is reported for an
arm whose probability for some class is constant. PR-AUC is average precision, evaluated once per **distinct score** so
that tied scores are not silently ordered. Both were verified against
scikit-learn to 2e-16 over random, tied, binary and constant scores.

**Brier.** The repository holds two conventions that differ by exactly a factor
of two for two classes:

- single-outcome (binary): mean (p_PIGD − y_PIGD)^2
- summed multiclass: mean of the sum over classes of (p_k − y_k)^2

Every Brier cell here states which one it is. They are never ranked together.
For two classes the conversion is exact, summed = 2 x single-outcome, so the
master table carries both for every binary row (`brier_binary` and
`brier_summed_two_class`). For three classes no such conversion exists, and the
summed form is the only one reported.
The 3-state CT-HMM's published 0.2348 and the rich profile's 0.1177 are **not**
a calibration gap of that size: the first is the summed form of 0.1214.

**Log loss** is −mean log p(true class) with the true-class probability clipped
at 1e-12, the constant the repository itself uses
(`src/ppmi_subtype/next_visit.py:118`, `src/ppmi_subtype/two_state_hmm.py:145`).
Arms whose clipping is active are flagged in the master table; for them the
unclipped log loss is infinite and the value shown is a floor, not a score.

**Degenerate scores.** Some arms emit rule-based probabilities taking only one
or two distinct values. Their AUC and PR-AUC are computed but marked with
`†`, because with two distinct scores AUC is algebraically
(sensitivity + specificity)/2 and measures no ranking. Two cases deserve
naming:

- The marginal/null arms have an AUC near 0.068,
  far below 0.5. This is a leave-one-patient-out artefact, not discrimination:
  the only thing that moves the score is the removal of the held-out patient
  from the training prior, which moves it *against* that patient's own label.
- `Direct logistic` and `Persistence` have an Indeterminate one-vs-rest AUC of
  exactly 0.5000 because their Indeterminate probability is constant.

## 4. Table A — binary TD vs PIGD on the matched set

3,020 transitions, 898 patients, scored by both the rich clinical
profile and the subtype-history models. Confidence intervals are patient-level
bootstrap, 2,000 draws, seed 42. `†` marks a degenerate score as defined above.

| Model | Accuracy | Sensitivity (PIGD) | Specificity | Precision | F1 | Balanced acc. | AUC-ROC | PR-AUC | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|---|---|
| Profile: level only (random walk) | 0.8440 [0.8285, 0.8598] | 0.6769 [0.6406, 0.7100] | 0.9161 | 0.7768 | 0.7234 | 0.7965 [0.7766, 0.8151] | 0.8865 [0.8704, 0.9014] | 0.7980 [0.7610, 0.8300] | 0.1195 | 0.3814 |
| Profile: level + rate + medication | 0.8404 [0.8248, 0.8566] | 0.6813 [0.6448, 0.7137] | 0.9090 | 0.7635 | 0.7201 | 0.7952 [0.7762, 0.8137] | 0.8902 [0.8748, 0.9053] | 0.8043 [0.7684, 0.8359] | 0.1177 | 0.3766 |
| Profile: level + rate, no medication | 0.8404 [0.8249, 0.8563] | 0.6769 [0.6404, 0.7088] | 0.9109 | 0.7662 | 0.7188 | 0.7939 [0.7747, 0.8124] | 0.8903 [0.8747, 0.9052] | 0.8042 [0.7682, 0.8360] | 0.1175 | 0.3759 |
| CT-HMM (latent, interval-aware) | 0.8371 [0.8202, 0.8540] | 0.6681 [0.6313, 0.7013] | 0.9100 | 0.7619 | 0.7119 | 0.7890 [0.7688, 0.8083] | 0.8583 [0.8411, 0.8755] | 0.7490 [0.7102, 0.7864] | 0.1215 | 0.3963 |
| Direct logistic (history + interval) | 0.8361 [0.8185, 0.8530] | 0.6670 [0.6306, 0.6988] | 0.9090 | 0.7597 | 0.7104 | 0.7880 [0.7682, 0.8064] | 0.8564 [0.8377, 0.8743] | 0.7461 [0.7040, 0.7869] | 0.1226 | 0.4005 |
| Persistence (last non-Indeterminate) | 0.8338 [0.8167, 0.8509] | 0.6231 [0.5857, 0.6582] | 0.9246 | 0.7810 | 0.6932 | 0.7739 [0.7536, 0.7938] | 0.7927 [0.7725, 0.8127] | 0.6228 [0.5773, 0.6649] | 0.1598 | 4.1959 |
| Markov (observed, discrete) | 0.8298 [0.8130, 0.8471] | 0.5901 [0.5527, 0.6235] | 0.9332 | 0.7920 | 0.6763 | 0.7616 [0.7414, 0.7806] | 0.6777 [0.6461, 0.7081] | 0.5386 [0.4939, 0.5830] | 0.1327 | 0.4311 |
| CTMC (observed, interval-aware) | 0.8281 [0.8105, 0.8455] | 0.5813 [0.5440, 0.6156] | 0.9346 | 0.7931 | 0.6709 | 0.7580 [0.7377, 0.7768] | 0.8043 [0.7820, 0.8269] | 0.6519 [0.6065, 0.6975] | 0.1343 | 0.4348 |
| Last observation carried forward | 0.8252 [0.8083, 0.8425] | 0.6659 [0.6296, 0.6990] | 0.8938 | 0.7301 | 0.6966 | 0.7799 [0.7587, 0.7993] | 0.8780 [0.8609, 0.8937] | 0.7869 [0.7490, 0.8192] | 0.1235 | 0.3941 |
| Direct logistic (history only) | 0.8195 [0.8020, 0.8371] | 0.6462 [0.6089, 0.6781] | 0.8943 | 0.7250 | 0.6833 | 0.7702 [0.7492, 0.7887] | 0.8377 [0.8169, 0.8573] | 0.7358 [0.6912, 0.7768] | 0.1244 | 0.4052 |
| Marginal (two-class prior) | 0.6987 [0.6714, 0.7277] | 0.0000 [0.0000, 0.0000] | 1.0000 | Not calculated | 0.0000 | 0.5000 [0.5000, 0.5000] | 0.0683 [0.0575, 0.0799] | 0.1799 [0.1614, 0.1989] | 0.2110 | 0.6130 |
| Population mean trajectory | 0.6914 [0.6677, 0.7153] | 0.1319 [0.1059, 0.1587] | 0.9327 | 0.4580 | 0.2048 | 0.5323 [0.5182, 0.5463] | 0.5876 [0.5583, 0.6156] | 0.3822 [0.3369, 0.4338] | 0.2083 | 0.6067 |

Reading this table: accuracy spans 0.6914
to 0.8440 and the intervals of
the leading models overlap heavily. AUC separates them much more clearly.

## 5. Table A2 — the same task on each family's own cohort

These four blocks use four different cohorts. Compare within a block only.

### A2.1 Rich clinical profile, 3,021 transitions, 5-fold patient CV

| Model | Accuracy | Sensitivity (PIGD) | Specificity | Precision | F1 | Balanced acc. | AUC-ROC | PR-AUC | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|---|---|
| Profile: level only (random walk) | 0.8441 [0.8285, 0.8599] | 0.6773 [0.6409, 0.7105] | 0.9161 | 0.7771 | 0.7238 | 0.7967 [0.7767, 0.8153] | 0.8865 [0.8704, 0.9014] | 0.7981 [0.7611, 0.8300] | 0.1195 | 0.3815 |
| Profile: level + rate + medication | 0.8405 [0.8249, 0.8566] | 0.6817 [0.6452, 0.7141] | 0.9090 | 0.7638 | 0.7204 | 0.7953 [0.7764, 0.8139] | 0.8902 [0.8748, 0.9053] | 0.8044 [0.7687, 0.8359] | 0.1177 | 0.3767 |
| Profile: level + rate, no medication | 0.8401 [0.8246, 0.8559] | 0.6762 [0.6397, 0.7082] | 0.9109 | 0.7662 | 0.7184 | 0.7935 [0.7742, 0.8123] | 0.8903 [0.8747, 0.9052] | 0.8042 [0.7683, 0.8360] | 0.1175 | 0.3760 |
| Last observation carried forward | 0.8249 [0.8081, 0.8423] | 0.6652 [0.6291, 0.6983] | 0.8938 | 0.7301 | 0.6962 | 0.7795 [0.7585, 0.7990] | 0.8780 [0.8610, 0.8937] | 0.7870 [0.7491, 0.8193] | 0.1235 | 0.3942 |
| Population mean trajectory | 0.6912 [0.6677, 0.7153] | 0.1317 [0.1058, 0.1585] | 0.9327 | 0.4580 | 0.2046 | 0.5322 [0.5181, 0.5463] | 0.5873 [0.5579, 0.6153] | 0.3822 [0.3370, 0.4338] | 0.2083 | 0.6069 |

### A2.2 Subtype-history models, 3,029 transitions, leave-one-patient-out

| Model | Accuracy | Sensitivity (PIGD) | Specificity | Precision | F1 | Balanced acc. | AUC-ROC | PR-AUC | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|---|---|
| CT-HMM (latent, interval-aware) | 0.8369 [0.8204, 0.8541] | 0.6674 [0.6289, 0.7009] | 0.9102 | 0.7625 | 0.7118 | 0.7888 [0.7685, 0.8079] | 0.8587 [0.8402, 0.8764] | 0.7496 [0.7106, 0.7873] | 0.1214 | 0.3961 |
| Direct logistic (history + interval) | 0.8359 [0.8195, 0.8529] | 0.6663 [0.6277, 0.6989] | 0.9092 | 0.7603 | 0.7102 | 0.7878 [0.7672, 0.8065] | 0.8568 [0.8387, 0.8744] | 0.7467 [0.7038, 0.7874] | 0.1225 | 0.4003 |
| Persistence (last non-Indeterminate) | 0.8336 [0.8173, 0.8510] | 0.6225 [0.5834, 0.6566] | 0.9248 | 0.7816 | 0.6931 | 0.7737 [0.7523, 0.7937] | 0.7929 [0.7721, 0.8130] | 0.6234 [0.5774, 0.6656] | 0.1598 | 4.1929 |
| Markov (observed, discrete) | 0.8293 [0.8129, 0.8468] | 0.5886 [0.5506, 0.6231] | 0.9333 | 0.7923 | 0.6755 | 0.7610 [0.7406, 0.7805] | 0.6778 [0.6463, 0.7085] | 0.5392 [0.4968, 0.5843] | 0.1328 | 0.4314 |
| CTMC (observed, interval-aware) | 0.8277 [0.8114, 0.8451] | 0.5799 [0.5414, 0.6147] | 0.9348 | 0.7934 | 0.6700 | 0.7573 [0.7365, 0.7769] | 0.8049 [0.7817, 0.8266] | 0.6527 [0.6082, 0.6979] | 0.1343 | 0.4346 |
| Direct logistic (history only) | 0.8194 [0.8019, 0.8372] | 0.6455 [0.6071, 0.6794] | 0.8946 | 0.7257 | 0.6833 | 0.7700 [0.7489, 0.7898] | 0.8378 [0.8176, 0.8567] | 0.7361 [0.6923, 0.7771] | 0.1244 | 0.4055 |
| Marginal (two-class prior) | 0.6983 [0.6703, 0.7278] | 0.0000 [0.0000, 0.0000] | 1.0000 | Not calculated | 0.0000 | 0.5000 [0.5000, 0.5000] | 0.0683 [0.0578, 0.0797] | 0.1802 [0.1616, 0.1998] | 0.2111 | 0.6134 |

### A2.3 Two-state HMM family, 2,936 transitions scorable by every arm

| Model | Accuracy | Sensitivity (PIGD) | Specificity | Precision | F1 | Balanced acc. | AUC-ROC | PR-AUC | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|---|---|
| CT-HMM (2 hidden x 2 observed, Indeterminate excluded) | 0.8478 [0.8318, 0.8637] | 0.6794 [0.6416, 0.7140] | 0.9183 | 0.7770 | 0.7249 | 0.7988 [0.7789, 0.8186] | 0.8530 [0.8338, 0.8699] | 0.7398 [0.6999, 0.7787] | 0.1198 | 0.3955 |
| CT-HMM (2 hidden x 2 observed, no emission floor) | 0.8478 [0.8318, 0.8637] | 0.6794 [0.6416, 0.7140] | 0.9183 | 0.7770 | 0.7249 | 0.7988 [0.7789, 0.8186] | 0.8530 [0.8338, 0.8699] | 0.7398 [0.6999, 0.7787] | 0.1198 | 0.3955 |
| CT-HMM (3 hidden x 3 observed, existing model) | 0.8467 [0.8305, 0.8625] | 0.6874 [0.6503, 0.7210] | 0.9135 | 0.7690 | 0.7259 | 0.8005 [0.7801, 0.8193] | 0.8599 [0.8414, 0.8767] | 0.7511 [0.7123, 0.7895] | 0.1174 | 0.3868 |
| CT-HMM (2 hidden x 3 observed) | 0.8464 [0.8304, 0.8625] | 0.6897 [0.6534, 0.7234] | 0.9120 | 0.7667 | 0.7262 | 0.8009 [0.7813, 0.8204] | 0.8583 [0.8396, 0.8749] | 0.7404 [0.6999, 0.7801] | 0.1187 | 0.3918 |
| Persistence (last observed label) † | 0.8443 [0.8276, 0.8609] | 0.6563 [0.6197, 0.6909] | 0.9232 | 0.7816 | 0.7135 | 0.7897 [0.7693, 0.8098] | 0.7897 [0.7693, 0.8098] | 0.6144 [0.5695, 0.6582] | 0.1557 | 4.3009 |
| Marginal (training label prior) | 0.7047 [0.6754, 0.7334] | 0.0000 [0.0000, 0.0000] | 1.0000 | Not calculated | 0.0000 | 0.5000 [0.5000, 0.5000] | 0.0677 [0.0566, 0.0803] | 0.1748 [0.1565, 0.1950] | 0.2086 | 0.6080 |

### A2.4 Harmonised subscore family, one protocol for all 14 models

The source run published three-class metrics only. The binary view here was
derived by renormalising its saved three-class probabilities over TD + PIGD and
restricting to cases with a TD or PIGD target — the identical convention the
leave-one-patient-out run applies, verified against that run's own `P_PIGD`
column to 1e-9.

| Model | Accuracy | Sensitivity (PIGD) | Specificity | Precision | F1 | Balanced acc. | AUC-ROC | PR-AUC | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|---|---|
| Persistence (recorded label) † | 0.8572 [0.8407, 0.8738] | 0.6802 [0.6407, 0.7151] | 0.9284 | 0.7923 | 0.7320 | 0.8043 [0.7829, 0.8249] | 0.8043 [0.7829, 0.8249] | 0.6306 [0.5837, 0.6758] | 0.1428 | 3.9459 |
| Subscore GP, independent | 0.8409 [0.8248, 0.8573] | 0.6882 [0.6512, 0.7196] | 0.9069 | 0.7615 | 0.7230 | 0.7975 [0.7776, 0.8160] | 0.8903 [0.8743, 0.9053] | 0.8051 [0.7685, 0.8364] | 0.1199 | 0.3830 |
| Subscore GP, joint | 0.8392 [0.8231, 0.8559] | 0.6937 [0.6569, 0.7253] | 0.9021 | 0.7539 | 0.7225 | 0.7979 [0.7776, 0.8166] | 0.8895 [0.8733, 0.9046] | 0.8035 [0.7667, 0.8350] | 0.1196 | 0.3817 |
| CT-HMM, 3 latent states | 0.8369 [0.8204, 0.8541] | 0.6674 [0.6289, 0.7009] | 0.9102 | 0.7625 | 0.7118 | 0.7888 [0.7685, 0.8079] | 0.8587 [0.8402, 0.8764] | 0.7497 [0.7107, 0.7874] | 0.1214 | 0.3961 |
| CT-HMM, 5 latent states | 0.8336 [0.8173, 0.8511] | 0.6554 [0.6173, 0.6875] | 0.9106 | 0.7602 | 0.7039 | 0.7830 [0.7626, 0.8022] | 0.8600 [0.8421, 0.8768] | 0.7589 [0.7206, 0.7959] | 0.1225 | 0.3977 |
| Joint subscore LMM | 0.8316 [0.8137, 0.8494] | 0.6389 [0.5987, 0.6734] | 0.9149 | 0.7644 | 0.6961 | 0.7769 [0.7569, 0.7953] | 0.8755 [0.8579, 0.8921] | 0.7823 [0.7423, 0.8172] | 0.1315 | 0.4197 |
| Time-varying LMM, constant | 0.8316 [0.8137, 0.8494] | 0.6389 [0.5987, 0.6734] | 0.9149 | 0.7644 | 0.6961 | 0.7769 [0.7569, 0.7953] | 0.8755 [0.8579, 0.8921] | 0.7823 [0.7423, 0.8172] | 0.1315 | 0.4197 |
| Markov chain, discrete | 0.8293 [0.8129, 0.8468] | 0.5886 [0.5506, 0.6231] | 0.9333 | 0.7923 | 0.6755 | 0.7610 [0.7406, 0.7805] | 0.6778 [0.6463, 0.7085] | 0.5358 [0.4935, 0.5806] | 0.1328 | 0.4314 |
| Time-varying LMM, spline | 0.8280 [0.8101, 0.8464] | 0.6138 [0.5714, 0.6499] | 0.9206 | 0.7695 | 0.6829 | 0.7672 [0.7456, 0.7865] | 0.8759 [0.8579, 0.8929] | 0.7799 [0.7387, 0.8161] | 0.1298 | 0.4137 |
| CTMC, interval-aware | 0.8277 [0.8114, 0.8451] | 0.5799 [0.5414, 0.6147] | 0.9348 | 0.7934 | 0.6700 | 0.7573 [0.7365, 0.7769] | 0.8049 [0.7817, 0.8266] | 0.6527 [0.6082, 0.6979] | 0.1343 | 0.4346 |
| Persistence (subscore LOCF) | 0.8270 [0.8101, 0.8453] | 0.6652 [0.6289, 0.6979] | 0.8969 | 0.7361 | 0.6989 | 0.7811 [0.7609, 0.8012] | 0.8739 [0.8552, 0.8911] | 0.7799 [0.7397, 0.8158] | 0.1247 | 0.4510 |
| Patient mean subscore | 0.8237 [0.8062, 0.8423] | 0.5853 [0.5343, 0.6322] | 0.9267 | 0.7754 | 0.6671 | 0.7560 [0.7312, 0.7796] | 0.8647 [0.8457, 0.8824] | 0.7497 [0.7033, 0.7906] | 0.1296 | 0.4619 |
| Marginal null | 0.6983 [0.6703, 0.7278] | 0.0000 [0.0000, 0.0000] | 1.0000 | Not calculated | 0.0000 | 0.5000 [0.5000, 0.5000] | 0.0683 [0.0578, 0.0797] | 0.1782 [0.1597, 0.1977] | 0.2111 | 0.6134 |
| Population mean subscore | 0.6940 [0.6718, 0.7179] | 0.2057 [0.1726, 0.2413] | 0.9050 | 0.4833 | 0.2886 | 0.5553 [0.5372, 0.5738] | 0.6002 [0.5708, 0.6279] | 0.3954 [0.3498, 0.4479] | 0.2091 | 0.6087 |

Cohort sizes are **not** equal in that block. Thirteen models score all
3,029 cases with a
TD/PIGD target; label persistence carries no TD/PIGD probability at all when the
previous recorded label was Indeterminate, so it is scorable on only
2,759 of them —
a smaller and easier subset, which is why its accuracy looks the highest there.
This inconsistency was found during this consolidation; the next block is the
comparable version.

### A2.5 Harmonised family on the common subset — the comparable ranking

| Model | Accuracy | Sensitivity (PIGD) | Specificity | Precision | F1 | Balanced acc. | AUC-ROC | PR-AUC | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|---|---|
| Subscore GP, independent | 0.8637 [0.8482, 0.8794] | 0.7155 [0.6787, 0.7497] | 0.9233 | 0.7894 | 0.7507 | 0.8194 [0.7996, 0.8380] | 0.9006 [0.8838, 0.9155] | 0.8211 [0.7830, 0.8527] | 0.1086 | 0.3555 |
| Subscore GP, joint | 0.8637 [0.8479, 0.8792] | 0.7168 [0.6808, 0.7503] | 0.9228 | 0.7886 | 0.7510 | 0.8198 [0.7993, 0.8383] | 0.9000 [0.8829, 0.9148] | 0.8197 [0.7811, 0.8520] | 0.1082 | 0.3538 |
| CT-HMM, 3 latent states | 0.8594 [0.8425, 0.8758] | 0.6941 [0.6537, 0.7301] | 0.9258 | 0.7899 | 0.7389 | 0.8099 [0.7887, 0.8301] | 0.8626 [0.8430, 0.8810] | 0.7554 [0.7137, 0.7943] | 0.1102 | 0.3697 |
| CT-HMM, 5 latent states | 0.8590 [0.8426, 0.8755] | 0.6991 [0.6596, 0.7338] | 0.9233 | 0.7855 | 0.7398 | 0.8112 [0.7902, 0.8311] | 0.8652 [0.8459, 0.8833] | 0.7679 [0.7273, 0.8055] | 0.1108 | 0.3705 |
| Markov chain, discrete | 0.8572 [0.8407, 0.8738] | 0.6802 [0.6407, 0.7151] | 0.9284 | 0.7923 | 0.7320 | 0.8043 [0.7829, 0.8249] | 0.6464 [0.6111, 0.6806] | 0.5063 [0.4609, 0.5561] | 0.1213 | 0.4057 |
| Persistence (recorded label) † | 0.8572 [0.8407, 0.8738] | 0.6802 [0.6407, 0.7151] | 0.9284 | 0.7923 | 0.7320 | 0.8043 [0.7829, 0.8249] | 0.8043 [0.7829, 0.8249] | 0.6306 [0.5837, 0.6758] | 0.1428 | 3.9459 |
| Persistence (subscore LOCF) | 0.8572 [0.8407, 0.8738] | 0.6802 [0.6407, 0.7151] | 0.9284 | 0.7923 | 0.7320 | 0.8043 [0.7829, 0.8249] | 0.8844 [0.8651, 0.9019] | 0.7962 [0.7532, 0.8350] | 0.1118 | 0.4261 |
| CTMC, interval-aware | 0.8554 [0.8386, 0.8722] | 0.6700 [0.6303, 0.7069] | 0.9299 | 0.7934 | 0.7265 | 0.8000 [0.7790, 0.8207] | 0.8023 [0.7769, 0.8264] | 0.6487 [0.5999, 0.6979] | 0.1226 | 0.4087 |
| Time-varying LMM, constant | 0.8547 [0.8370, 0.8716] | 0.6764 [0.6351, 0.7156] | 0.9263 | 0.7868 | 0.7274 | 0.8013 [0.7790, 0.8216] | 0.8878 [0.8704, 0.9045] | 0.7986 [0.7590, 0.8352] | 0.1217 | 0.3965 |
| Joint subscore LMM | 0.8547 [0.8370, 0.8716] | 0.6764 [0.6351, 0.7156] | 0.9263 | 0.7868 | 0.7274 | 0.8013 [0.7790, 0.8216] | 0.8878 [0.8704, 0.9045] | 0.7986 [0.7590, 0.8352] | 0.1217 | 0.3965 |
| Time-varying LMM, spline | 0.8492 [0.8306, 0.8672] | 0.6485 [0.6057, 0.6881] | 0.9299 | 0.7880 | 0.7115 | 0.7892 [0.7663, 0.8102] | 0.8876 [0.8701, 0.9043] | 0.7950 [0.7545, 0.8323] | 0.1201 | 0.3906 |
| Patient mean subscore | 0.8456 [0.8277, 0.8639] | 0.6068 [0.5543, 0.6560] | 0.9416 | 0.8067 | 0.6926 | 0.7742 [0.7475, 0.7989] | 0.8768 [0.8580, 0.8953] | 0.7683 [0.7213, 0.8124] | 0.1175 | 0.4360 |
| Marginal null | 0.7133 [0.6834, 0.7428] | 0.0000 [0.0000, 0.0000] | 1.0000 | Not calculated | 0.0000 | 0.5000 [0.5000, 0.5000] | 0.0630 [0.0520, 0.0751] | 0.1681 [0.1495, 0.1885] | 0.2052 | 0.6007 |
| Population mean subscore | 0.7053 [0.6816, 0.7287] | 0.2099 [0.1719, 0.2473] | 0.9045 | 0.4689 | 0.2900 | 0.5572 [0.5365, 0.5786] | 0.6044 [0.5745, 0.6357] | 0.3816 [0.3314, 0.4381] | 0.2054 | 0.6008 |

On identical rows the order changes: label persistence's apparent lead
disappears (-0.0022 [-0.0072, +0.0030] (includes zero)
against the CT-HMM on accuracy) and its log loss remains two orders of magnitude
worse.

## 6. Table B — ratio-banded CT-HMM

All six arms share one cohort of 3,029 transitions and one
protocol, with bands rebuilt inside every fold. The unbanded CT-HMM is the
reference row.

| Band width | Accuracy | Sensitivity (PIGD) | Specificity | Precision | F1 | Balanced acc. | AUC-ROC | PR-AUC | Brier | Log loss | Parameters |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unbanded | 0.8369 [0.8204, 0.8541] | 0.6674 [0.6289, 0.7009] | 0.9102 | 0.7625 | 0.7118 | 0.7888 [0.7685, 0.8079] | 0.8587 [0.8402, 0.8764] | 0.7496 [0.7106, 0.7873] | 0.1214 | 0.3961 | Not applicable |
| 0.05 | 0.8336 [0.8169, 0.8513] | 0.7090 [0.6736, 0.7401] | 0.8875 | 0.7314 | 0.7200 | 0.7982 [0.7800, 0.8166] | 0.8739 [0.8561, 0.8912] | 0.7330 [0.6886, 0.7749] | 0.1227 | 0.3952 | 224 |
| 0.10 | 0.8316 [0.8142, 0.8498] | 0.7079 [0.6713, 0.7399] | 0.8851 | 0.7270 | 0.7173 | 0.7965 [0.7775, 0.8158] | 0.8758 [0.8588, 0.8933] | 0.7398 [0.6971, 0.7803] | 0.1221 | 0.3930 | 137 |
| 0.20 | 0.8336 [0.8167, 0.8516] | 0.6991 [0.6597, 0.7313] | 0.8917 | 0.7362 | 0.7172 | 0.7954 [0.7753, 0.8147] | 0.8775 [0.8603, 0.8944] | 0.7540 [0.7120, 0.7938] | 0.1209 | 0.3898 | 92 |
| 0.30 | 0.8359 [0.8192, 0.8535] | 0.6980 [0.6599, 0.7303] | 0.8955 | 0.7427 | 0.7197 | 0.7968 [0.7770, 0.8156] | 0.8784 [0.8616, 0.8949] | 0.7523 [0.7118, 0.7912] | 0.1199 | 0.3874 | 77 |
| 0.50 | 0.8320 [0.8152, 0.8507] | 0.7188 [0.6828, 0.7489] | 0.8809 | 0.7228 | 0.7208 | 0.7998 [0.7807, 0.8191] | 0.8765 [0.8596, 0.8936] | 0.7409 [0.7010, 0.7808] | 0.1217 | 0.3916 | 65 |

### 6.1 Band construction and sparsity

| Width | Main bands | Tail bands | Joint categories | Parameters | Empty | < 10 obs | < 30 obs | Median obs/category | Obs per parameter |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 60 | 10 | 73 | 224 | 11 | 28 | 40 | 22 | 19.17 |
| 0.10 | 30 | 10 | 44 | 137 | 1 | 6 | 12 | 71 | 31.35 |
| 0.20 | 15 | 10 | 29 | 92 | 0 | 1 | 2 | 135 | 46.68 |
| 0.30 | 10 | 10 | 24 | 77 | 1 | 2 | 2 | 137 | 55.78 |
| 0.50 | 6 | 10 | 20 | 65 | 0 | 0 | 0 | 148 | 66.08 |

Bands partition the tremor-to-PIGD ratio over [0, 3) at the stated width, plus
ten tail bands above 3. The joint observation is (subtype, band), so the
alphabet grows from 3 labels to between 20 and
73 categories.

### 6.2 Paired comparison against the unbanded model, same rows

| Contrast | Metric | Delta | 95% CI | Excludes zero |
|---|---|---|---|---|
| ratio bands at width 0.05 | accuracy | -0.0033 | [-0.0116, +0.0057] | no |
| ratio bands at width 0.05 | sensitivity_PIGD | +0.0416 | [+0.0245, +0.0611] | yes |
| ratio bands at width 0.05 | specificity_TD | -0.0227 | [-0.0324, -0.0129] | yes |
| ratio bands at width 0.05 | auc_roc | +0.0151 | [+0.0070, +0.0243] | yes |
| ratio bands at width 0.05 | pr_auc | -0.0166 | [-0.0396, +0.0057] | no |
| ratio bands at width 0.05 | brier_binary | +0.0013 | [-0.0021, +0.0047] | no |
| ratio bands at width 0.05 | log_loss | -0.0009 | [-0.0099, +0.0080] | no |
| ratio bands at width 0.10 | accuracy | -0.0053 | [-0.0134, +0.0034] | no |
| ratio bands at width 0.10 | sensitivity_PIGD | +0.0405 | [+0.0224, +0.0597] | yes |
| ratio bands at width 0.10 | specificity_TD | -0.0251 | [-0.0344, -0.0160] | yes |
| ratio bands at width 0.10 | auc_roc | +0.0171 | [+0.0089, +0.0260] | yes |
| ratio bands at width 0.10 | pr_auc | -0.0098 | [-0.0318, +0.0126] | no |
| ratio bands at width 0.10 | brier_binary | +0.0007 | [-0.0026, +0.0041] | no |
| ratio bands at width 0.10 | log_loss | -0.0031 | [-0.0118, +0.0058] | no |
| ratio bands at width 0.20 | accuracy | -0.0033 | [-0.0108, +0.0043] | no |
| ratio bands at width 0.20 | sensitivity_PIGD | +0.0317 | [+0.0151, +0.0492] | yes |
| ratio bands at width 0.20 | specificity_TD | -0.0184 | [-0.0266, -0.0100] | yes |
| ratio bands at width 0.20 | auc_roc | +0.0188 | [+0.0110, +0.0276] | yes |
| ratio bands at width 0.20 | pr_auc | +0.0044 | [-0.0167, +0.0247] | no |
| ratio bands at width 0.20 | brier_binary | -0.0005 | [-0.0036, +0.0024] | no |
| ratio bands at width 0.20 | log_loss | -0.0063 | [-0.0146, +0.0018] | no |
| ratio bands at width 0.30 | accuracy | -0.0010 | [-0.0082, +0.0065] | no |
| ratio bands at width 0.30 | sensitivity_PIGD | +0.0306 | [+0.0152, +0.0471] | yes |
| ratio bands at width 0.30 | specificity_TD | -0.0147 | [-0.0225, -0.0069] | yes |
| ratio bands at width 0.30 | auc_roc | +0.0197 | [+0.0120, +0.0285] | yes |
| ratio bands at width 0.30 | pr_auc | +0.0027 | [-0.0170, +0.0214] | no |
| ratio bands at width 0.30 | brier_binary | -0.0016 | [-0.0044, +0.0013] | no |
| ratio bands at width 0.30 | log_loss | -0.0086 | [-0.0167, -0.0009] | yes |
| ratio bands at width 0.50 | accuracy | -0.0050 | [-0.0131, +0.0031] | no |
| ratio bands at width 0.50 | sensitivity_PIGD | +0.0514 | [+0.0365, +0.0687] | yes |
| ratio bands at width 0.50 | specificity_TD | -0.0293 | [-0.0378, -0.0205] | yes |
| ratio bands at width 0.50 | auc_roc | +0.0177 | [+0.0101, +0.0265] | yes |
| ratio bands at width 0.50 | pr_auc | -0.0087 | [-0.0252, +0.0081] | no |
| ratio bands at width 0.50 | brier_binary | +0.0002 | [-0.0027, +0.0031] | no |
| ratio bands at width 0.50 | log_loss | -0.0045 | [-0.0124, +0.0032] | no |
| ratio bands at width 0.30 - width 0.10 | accuracy | +0.0043 | [-0.0013, +0.0100] | no |
| ratio bands at width 0.30 - width 0.10 | sensitivity_PIGD | -0.0098 | [-0.0219, +0.0011] | no |
| ratio bands at width 0.30 - width 0.10 | specificity_TD | +0.0104 | [+0.0040, +0.0171] | yes |
| ratio bands at width 0.30 - width 0.10 | auc_roc | +0.0026 | [+0.0001, +0.0051] | yes |
| ratio bands at width 0.30 - width 0.10 | pr_auc | +0.0125 | [+0.0026, +0.0218] | yes |
| ratio bands at width 0.30 - width 0.10 | brier_binary | -0.0022 | [-0.0038, -0.0007] | yes |
| ratio bands at width 0.30 - width 0.10 | log_loss | -0.0055 | [-0.0096, -0.0017] | yes |

The pattern is consistent and one-sided: **AUC-ROC rises at every width with an
interval clear of zero; accuracy, PR-AUC and the Brier score do not improve at
any width.** Sensitivity rises and specificity falls, which is a shift of the
operating point rather than a gain. Width 0.30 is the best of the banded arms on
AUC-ROC (0.8784), Brier (0.1199) and log loss
(0.3874) while using 77 parameters against
224 at width 0.05 and 14 for the unbanded model.

### 6.3 AIC and BIC for the banded model: why there are none

The banded model's likelihood is a PMF over its joint alphabet — 44 categories
at width 0.10 — while the unbanded CT-HMM's is a PMF over 3 labels. The
diagnostic run states this itself: banded log-likelihood −13123.4023 with 137
parameters against the baseline's −2969.0593 with 14
(`experiments/20261104T155439Z_ratio_band_emission_diagnostic/report.txt`). Those numbers are over different sample
spaces, so neither their difference nor their AIC means anything, and the
project's own report says so.

Whether a comparable likelihood could be recovered without refitting was
checked directly. The diagnostic run does save the pieces that marginalising
would need — the joint emission by state (`category_emission_by_state.csv`),
the banded generator (`generator_banded.csv`) and even the implied
subtype-level emission (`emission_subtype_banded.csv`). It does **not** save the
initial state distribution, so the 3-label sequence likelihood cannot be
evaluated from the saved parameters. Even with it the result would be a plug-in
value, not a maximum: the parameters were optimised for the 44-category
likelihood, so −2logL + 2k would not be an AIC for anything. The honest
comparable quantity is the held-out predictive log-likelihood on the shared
3-label alphabet, which is reported for every model in section 8 and needs no
refitting.

## 7. Table C — three-class classification

Hard three-class predictions are the argmax of the saved probabilities. For the
harmonised and two-state runs this reproduces their own published class counts
exactly, which confirms it is the rule they used. For the rich profile and the
ratio-band arms the source runs reported only a two-class decision, so the
argmax rule is applied here and labelled as such.

### 7.1 Harmonised family, one protocol, 3,361 transitions

| Model | Accuracy | Balanced acc. | Macro-F1 | Sens TD | Sens Ind. | Sens PIGD | Spec TD | Spec Ind. | Spec PIGD | Brier (summed) | Log loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Subscore GP, independent | 0.7575 [0.7385, 0.7762] | 0.5313 [0.5182, 0.5439] | 0.5074 [0.4938, 0.5203] | 0.9069 | 0.0000 | 0.6871 | 0.6003 | 1.0000 | 0.8705 | 0.3602 | 0.6537 |
| Subscore GP, joint | 0.7563 [0.7373, 0.7757] | 0.5319 [0.5189, 0.5449] | 0.5070 [0.4934, 0.5201] | 0.9021 | 0.0000 | 0.6937 | 0.6067 | 1.0000 | 0.8655 | 0.3594 | 0.6518 |
| CT-HMM, 3 latent states | 0.7539 [0.7348, 0.7731] | 0.5255 [0.5116, 0.5391] | 0.5037 [0.4896, 0.5173] | 0.9102 | 0.0000 | 0.6663 | 0.5787 | 0.9997 | 0.8770 | 0.3688 | 0.6726 |
| CT-HMM, 5 latent states | 0.7513 [0.7315, 0.7706] | 0.5220 [0.5084, 0.5355] | 0.5011 [0.4870, 0.5147] | 0.9106 | 0.0000 | 0.6554 | 0.5658 | 1.0000 | 0.8794 | 0.3700 | 0.6746 |
| Joint subscore LMM | 0.7495 [0.7296, 0.7685] | 0.5179 [0.5046, 0.5305] | 0.4977 [0.4835, 0.5107] | 0.9149 | 0.0000 | 0.6389 | 0.5618 | 1.0000 | 0.8790 | 0.3826 | 0.6919 |
| Time-varying LMM, constant | 0.7495 [0.7296, 0.7685] | 0.5179 [0.5046, 0.5305] | 0.4977 [0.4835, 0.5107] | 0.9149 | 0.0000 | 0.6389 | 0.5618 | 1.0000 | 0.8790 | 0.3826 | 0.6919 |
| Markov chain, discrete | 0.7474 [0.7272, 0.7670] | 0.5073 [0.4936, 0.5210] | 0.4929 [0.4784, 0.5072] | 0.9333 | 0.0000 | 0.5886 | 0.5016 | 1.0000 | 0.9068 | 0.3896 | 0.7066 |
| Time-varying LMM, spline | 0.7462 [0.7255, 0.7652] | 0.5115 [0.4978, 0.5243] | 0.4937 [0.4791, 0.5075] | 0.9206 | 0.0000 | 0.6138 | 0.5329 | 1.0000 | 0.8893 | 0.3794 | 0.6860 |
| Persistence (subscore LOCF) | 0.7453 [0.7245, 0.7651] | 0.5207 [0.5070, 0.5348] | 0.4970 [0.4833, 0.5112] | 0.8969 | 0.0000 | 0.6652 | 0.5867 | 0.9997 | 0.8611 | 0.3723 | 0.7556 |
| CTMC, interval-aware | 0.7453 [0.7250, 0.7646] | 0.5046 [0.4906, 0.5182] | 0.4909 [0.4766, 0.5054] | 0.9338 | 0.0000 | 0.5799 | 0.4960 | 0.9987 | 0.9085 | 0.3925 | 0.7105 |
| Patient mean subscore | 0.7423 [0.7227, 0.7624] | 0.5047 [0.4881, 0.5212] | 0.4903 [0.4741, 0.5063] | 0.9267 | 0.0030 | 0.5842 | 0.5072 | 0.9993 | 0.8978 | 0.3802 | 0.7594 |
| Persistence (recorded label) | 0.7227 [0.7009, 0.7437] | 0.5484 [0.5270, 0.5692] | 0.5559 [0.5349, 0.5764] | 0.8638 | 0.1928 | 0.5886 | 0.6517 | 0.9109 | 0.9068 | 0.5546 | 7.6620 |
| Marginal null | 0.6293 [0.6011, 0.6565] | 0.3333 [0.3333, 0.3333] | 0.2575 [0.2503, 0.2642] | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.5211 | 0.8756 |
| Population mean subscore | 0.6254 [0.6036, 0.6482] | 0.3702 [0.3573, 0.3831] | 0.3460 [0.3306, 0.3621] | 0.9050 | 0.0000 | 0.2057 | 0.1878 | 1.0000 | 0.8991 | 0.5174 | 0.8716 |

### 7.2 Subtype-history models, leave-one-patient-out, 3,361 transitions

| Model | Accuracy | Balanced acc. | Macro-F1 | Sens TD | Sens Ind. | Sens PIGD | Spec TD | Spec Ind. | Spec PIGD | Brier (summed) | Log loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CT-HMM (latent, interval-aware) | 0.7539 [0.7348, 0.7731] | 0.5255 [0.5116, 0.5391] | 0.5037 [0.4896, 0.5173] | 0.9102 | 0.0000 | 0.6663 | 0.5787 | 0.9997 | 0.8770 | 0.3688 | 0.6726 |
| Direct logistic (history + interval) | 0.7533 [0.7336, 0.7729] | 0.5252 [0.5118, 0.5387] | 0.5032 [0.4896, 0.5168] | 0.9092 | 0.0000 | 0.6663 | 0.5778 | 1.0000 | 0.8762 | 0.3888 | 3.0901 |
| Persistence (last non-Indeterminate) | 0.7513 [0.7320, 0.7706] | 0.5158 [0.5016, 0.5295] | 0.4984 [0.4838, 0.5120] | 0.9248 | 0.0000 | 0.6225 | 0.5353 | 1.0000 | 0.8950 | 0.4823 | 6.5081 |
| Markov (observed, discrete) | 0.7474 [0.7272, 0.7670] | 0.5073 [0.4936, 0.5210] | 0.4929 [0.4784, 0.5072] | 0.9333 | 0.0000 | 0.5886 | 0.5016 | 1.0000 | 0.9068 | 0.3896 | 0.7066 |
| CTMC (observed, interval-aware) | 0.7453 [0.7250, 0.7646] | 0.5046 [0.4906, 0.5182] | 0.4909 [0.4766, 0.5054] | 0.9338 | 0.0000 | 0.5799 | 0.4960 | 0.9987 | 0.9085 | 0.3925 | 0.7105 |
| Direct logistic (history only) | 0.7385 [0.7178, 0.7585] | 0.5134 [0.4995, 0.5272] | 0.4905 [0.4761, 0.5046] | 0.8946 | 0.0000 | 0.6455 | 0.5706 | 1.0000 | 0.8594 | 0.3921 | 3.0948 |
| Marginal (two-class prior) | 0.6293 [0.6011, 0.6565] | 0.3333 [0.3333, 0.3333] | 0.2575 [0.2503, 0.2642] | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.5365 | 3.2822 |

### 7.3 Two-state family, 3,361 transitions

| Model | Accuracy | Balanced acc. | Macro-F1 | Sens TD | Sens Ind. | Sens PIGD | Spec TD | Spec Ind. | Spec PIGD | Brier (summed) | Log loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CT-HMM (3 hidden x 3 observed, existing model) | 0.7539 [0.7348, 0.7731] | 0.5255 [0.5116, 0.5391] | 0.5037 [0.4896, 0.5173] | 0.9102 | 0.0000 | 0.6663 | 0.5787 | 0.9997 | 0.8770 | 0.3688 | 0.6726 |
| CT-HMM (2 hidden x 3 observed) | 0.7536 [0.7341, 0.7731] | 0.5249 [0.5113, 0.5383] | 0.5030 [0.4892, 0.5165] | 0.9106 | 0.0000 | 0.6641 | 0.5795 | 1.0000 | 0.8758 | 0.3748 | 0.6835 |
| Persistence (last observed label) | 0.7227 [0.7009, 0.7437] | 0.5484 [0.5270, 0.5692] | 0.5559 [0.5349, 0.5764] | 0.8638 | 0.1928 | 0.5886 | 0.6517 | 0.9109 | 0.9068 | 0.5546 | 7.6620 |
| Marginal (training label prior) | 0.6293 [0.6011, 0.6565] | 0.3333 [0.3333, 0.3333] | 0.2575 [0.2503, 0.2642] | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.5211 | 0.8756 |

### 7.4 Rich profile, 3,353 transitions

| Model | Accuracy | Balanced acc. | Macro-F1 | Sens TD | Sens Ind. | Sens PIGD | Spec TD | Spec Ind. | Spec PIGD | Brier (summed) | Log loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Profile: level only (random walk) | 0.7602 [0.7408, 0.7788] | 0.5310 [0.5174, 0.5440] | 0.5095 [0.4961, 0.5226] | 0.9156 | 0.0000 | 0.6773 | 0.5841 | 0.9990 | 0.8837 | 0.3581 | 0.6559 |
| Profile: level + rate + medication | 0.7569 [0.7375, 0.7758] | 0.5299 [0.5167, 0.5425] | 0.5066 [0.4931, 0.5195] | 0.9090 | 0.0000 | 0.6806 | 0.5969 | 0.9993 | 0.8722 | 0.3551 | 0.6521 |
| Profile: level + rate, no medication | 0.7566 [0.7376, 0.7757] | 0.5289 [0.5158, 0.5419] | 0.5064 [0.4930, 0.5191] | 0.9104 | 0.0000 | 0.6762 | 0.5889 | 0.9993 | 0.8759 | 0.3546 | 0.6513 |
| Last observation carried forward | 0.7432 [0.7224, 0.7634] | 0.5197 [0.5058, 0.5336] | 0.4956 [0.4814, 0.5097] | 0.8938 | 0.0000 | 0.6652 | 0.5849 | 1.0000 | 0.8587 | 0.3656 | 0.6700 |
| Population mean trajectory | 0.6227 [0.5993, 0.6468] | 0.3546 [0.3448, 0.3650] | 0.3198 [0.3059, 0.3344] | 0.9332 | 0.0000 | 0.1306 | 0.1199 | 1.0000 | 0.9300 | 0.5163 | 0.8704 |

### 7.5 Ratio-band arms, 3,361 transitions

| Model | Accuracy | Balanced acc. | Macro-F1 | Sens TD | Sens Ind. | Sens PIGD | Spec TD | Spec Ind. | Spec PIGD | Brier (summed) | Log loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CT-HMM (latent, interval-aware) | 0.7539 [0.7348, 0.7731] | 0.5255 [0.5116, 0.5391] | 0.5037 [0.4896, 0.5173] | 0.9102 | 0.0000 | 0.6663 | 0.5787 | 0.9997 | 0.8770 | 0.3688 | 0.6726 |
| CT-HMM (subtype x ratio band, width 0.50) | 0.7400 [0.7193, 0.7602] | 0.5299 [0.5150, 0.5444] | 0.5184 [0.5016, 0.5354] | 0.8785 | 0.0361 | 0.6751 | 0.6413 | 0.9726 | 0.8594 | 0.3748 | 0.6786 |
| CT-HMM (subtype x ratio band, width 0.30) | 0.7331 [0.7127, 0.7533] | 0.5364 [0.5194, 0.5545] | 0.5379 [0.5194, 0.5576] | 0.8738 | 0.0964 | 0.6389 | 0.6565 | 0.9393 | 0.8835 | 0.3734 | 0.6771 |
| CT-HMM (subtype x ratio band, width 0.05) | 0.7307 [0.7099, 0.7507] | 0.5339 [0.5164, 0.5508] | 0.5374 [0.5178, 0.5558] | 0.8771 | 0.1054 | 0.6193 | 0.6413 | 0.9399 | 0.8872 | 0.3814 | 0.6887 |
| CT-HMM (subtype x ratio band, width 0.20) | 0.7284 [0.7076, 0.7491] | 0.5337 [0.5155, 0.5519] | 0.5357 [0.5163, 0.5550] | 0.8681 | 0.0994 | 0.6335 | 0.6573 | 0.9330 | 0.8843 | 0.3761 | 0.6808 |
| CT-HMM (subtype x ratio band, width 0.10) | 0.7260 [0.7055, 0.7471] | 0.5308 [0.5137, 0.5474] | 0.5335 [0.5148, 0.5510] | 0.8695 | 0.1024 | 0.6204 | 0.6453 | 0.9353 | 0.8843 | 0.3806 | 0.6871 |

## 8. One-vs-rest ranking, and the Indeterminate class

This is the part of the picture that accuracy hides. The question in the brief
was whether a binary Indeterminate-versus-other analysis could be assembled
from what exists. It can be read one-versus-rest from the saved three-class
probabilities, which is what follows. It is **not** a separately trained binary
Indeterminate model, and is not presented as one.

### 8.1 Harmonised family

| Model | AUC TD | AUC Indeterminate | AUC PIGD | Macro AUC | PR-AUC TD | PR-AUC Ind. | PR-AUC PIGD | Mean predictive log-lik |
|---|---|---|---|---|---|---|---|---|
| Subscore GP, joint | 0.8590 [0.8409, 0.8759] | 0.6666 [0.6381, 0.6957] | 0.8664 [0.8487, 0.8825] | 0.7973 [0.7799, 0.8142] | 0.9034 | 0.1507 | 0.7414 | -0.6518 [-0.6854, -0.6193] |
| Subscore GP, independent | 0.8596 [0.8414, 0.8765] | 0.6627 [0.6342, 0.6916] | 0.8670 [0.8494, 0.8833] | 0.7964 [0.7790, 0.8131] | 0.9038 | 0.1484 | 0.7428 | -0.6537 [-0.6865, -0.6216] |
| Patient mean subscore | 0.8325 [0.8126, 0.8507] | 0.6588 [0.6279, 0.6918] | 0.8430 [0.8228, 0.8619] | 0.7781 [0.7593, 0.7964] | 0.8768 | 0.1618 | 0.6871 | -0.7594 [-0.8212, -0.7035] |
| Persistence (subscore LOCF) | 0.8413 [0.8210, 0.8601] | 0.6552 [0.6242, 0.6848] | 0.8511 [0.8314, 0.8693] | 0.7825 [0.7634, 0.8011] | 0.8831 | 0.1687 | 0.7138 | -0.7556 [-0.8198, -0.6983] |
| Time-varying LMM, spline | 0.8442 [0.8246, 0.8621] | 0.6313 [0.5993, 0.6610] | 0.8519 [0.8331, 0.8696] | 0.7758 [0.7576, 0.7936] | 0.8921 | 0.1338 | 0.7147 | -0.6860 [-0.7173, -0.6548] |
| Joint subscore LMM | 0.8449 [0.8251, 0.8621] | 0.6256 [0.5943, 0.6553] | 0.8502 [0.8312, 0.8675] | 0.7735 [0.7552, 0.7912] | 0.8905 | 0.1332 | 0.7157 | -0.6919 [-0.7228, -0.6612] |
| Time-varying LMM, constant | 0.8449 [0.8251, 0.8621] | 0.6256 [0.5943, 0.6553] | 0.8502 [0.8312, 0.8675] | 0.7735 [0.7552, 0.7912] | 0.8905 | 0.1331 | 0.7157 | -0.6919 [-0.7228, -0.6612] |
| CT-HMM, 3 latent states | 0.8176 [0.7979, 0.8363] | 0.6123 [0.5777, 0.6472] | 0.8387 [0.8192, 0.8570] | 0.7562 [0.7376, 0.7741] | 0.8406 | 0.1479 | 0.6794 | -0.6726 [-0.7082, -0.6382] |
| CT-HMM, 5 latent states | 0.8221 [0.8031, 0.8402] | 0.6035 [0.5715, 0.6382] | 0.8386 [0.8202, 0.8568] | 0.7547 [0.7372, 0.7720] | 0.8440 | 0.1337 | 0.6815 | -0.6746 [-0.7098, -0.6393] |
| Persistence (recorded label) | 0.7578 [0.7373, 0.7778] | 0.5518 [0.5286, 0.5747] | 0.7477 [0.7271, 0.7680] | 0.6858 [0.6700, 0.7015] | 0.7837 | 0.1167 | 0.5253 | -7.6620 [-8.2650, -7.0824] |
| CTMC, interval-aware | 0.7607 [0.7379, 0.7840] | 0.5431 [0.5035, 0.5816] | 0.7879 [0.7652, 0.8104] | 0.6972 [0.6756, 0.7186] | 0.7851 | 0.1226 | 0.5785 | -0.7105 [-0.7465, -0.6752] |
| Population mean subscore | 0.5846 [0.5560, 0.6123] | 0.5161 [0.4810, 0.5517] | 0.5946 [0.5644, 0.6229] | 0.5651 [0.5409, 0.5884] | 0.6763 | 0.1039 | 0.3551 | -0.8716 [-0.8938, -0.8487] |
| Markov chain, discrete | 0.6252 [0.5945, 0.6560] | 0.3756 [0.3323, 0.4208] | 0.6602 [0.6287, 0.6915] | 0.5537 [0.5277, 0.5809] | 0.6528 | 0.0948 | 0.4586 | -0.7066 [-0.7429, -0.6713] |
| Marginal null | 0.0928 [0.0806, 0.1062] | 0.0969 [0.0826, 0.1123] | 0.0865 [0.0746, 0.0993] | 0.0921 [0.0811, 0.1031] | 0.4433 | 0.0561 | 0.1602 | -0.8756 [-0.9057, -0.8468] |

### 8.2 Subtype-history models

| Model | AUC TD | AUC Indeterminate | AUC PIGD | Macro AUC | PR-AUC TD | PR-AUC Ind. | PR-AUC PIGD | Mean predictive log-lik |
|---|---|---|---|---|---|---|---|---|
| CT-HMM (latent, interval-aware) | 0.8176 [0.7979, 0.8363] | 0.6123 [0.5777, 0.6472] | 0.8387 [0.8192, 0.8570] | 0.7562 [0.7376, 0.7741] | 0.8406 | 0.1479 | 0.6794 | -0.6726 [-0.7082, -0.6382] |
| CTMC (observed, interval-aware) | 0.7607 [0.7379, 0.7841] | 0.5431 [0.5035, 0.5816] | 0.7879 [0.7652, 0.8104] | 0.6972 [0.6756, 0.7186] | 0.7851 | 0.1226 | 0.5785 | -0.7105 [-0.7465, -0.6752] |
| Direct logistic (history + interval) | 0.8151 [0.7956, 0.8336] | 0.5000 [0.5000, 0.5000] | 0.8358 [0.8172, 0.8548] | 0.7170 [0.7051, 0.7286] | 0.8332 | 0.0988 | 0.6663 | -3.0901 [-3.4051, -2.7798] |
| Direct logistic (history only) | 0.7935 [0.7733, 0.8139] | 0.5000 [0.5000, 0.5000] | 0.8181 [0.7969, 0.8388] | 0.7038 [0.6910, 0.7169] | 0.7816 | 0.0988 | 0.6612 | -3.0948 [-3.4101, -2.7851] |
| Marginal (two-class prior) | 0.1159 [0.1005, 0.1329] | 0.5000 [0.5000, 0.5000] | 0.0874 [0.0756, 0.1006] | 0.2345 [0.2264, 0.2433] | 0.4530 | 0.0988 | 0.1623 | -3.2822 [-3.5903, -2.9739] |
| Persistence (last non-Indeterminate) | 0.7521 [0.7315, 0.7734] | 0.5000 [0.5000, 0.5000] | 0.7756 [0.7548, 0.7956] | 0.6759 [0.6627, 0.6889] | 0.7784 | 0.0988 | 0.5510 | -6.5081 [-7.0292, -5.9786] |
| Markov (observed, discrete) | 0.6252 [0.5945, 0.6560] | 0.3756 [0.3323, 0.4208] | 0.6602 [0.6287, 0.6915] | 0.5537 [0.5277, 0.5809] | 0.6528 | 0.0948 | 0.4586 | -0.7066 [-0.7429, -0.6713] |

### 8.3 Ratio-band arms

| Model | AUC TD | AUC Indeterminate | AUC PIGD | Macro AUC | PR-AUC TD | PR-AUC Ind. | PR-AUC PIGD | Mean predictive log-lik |
|---|---|---|---|---|---|---|---|---|
| CT-HMM (subtype x ratio band, width 0.30) | 0.8434 [0.8244, 0.8609] | 0.6270 [0.5961, 0.6582] | 0.8534 [0.8345, 0.8708] | 0.7746 [0.7572, 0.7910] | 0.8807 | 0.1383 | 0.6672 | -0.6771 [-0.7124, -0.6423] |
| CT-HMM (subtype x ratio band, width 0.20) | 0.8435 [0.8247, 0.8612] | 0.6264 [0.5946, 0.6582] | 0.8523 [0.8334, 0.8695] | 0.7741 [0.7565, 0.7908] | 0.8810 | 0.1370 | 0.6616 | -0.6808 [-0.7167, -0.6462] |
| CT-HMM (subtype x ratio band, width 0.50) | 0.8421 [0.8232, 0.8597] | 0.6219 [0.5907, 0.6536] | 0.8505 [0.8319, 0.8675] | 0.7715 [0.7537, 0.7882] | 0.8799 | 0.1340 | 0.6517 | -0.6786 [-0.7146, -0.6444] |
| CT-HMM (subtype x ratio band, width 0.10) | 0.8406 [0.8213, 0.8583] | 0.6190 [0.5872, 0.6511] | 0.8501 [0.8315, 0.8671] | 0.7699 [0.7527, 0.7865] | 0.8792 | 0.1338 | 0.6478 | -0.6871 [-0.7224, -0.6526] |
| CT-HMM (subtype x ratio band, width 0.05) | 0.8379 [0.8183, 0.8555] | 0.6160 [0.5839, 0.6474] | 0.8483 [0.8292, 0.8656] | 0.7674 [0.7501, 0.7841] | 0.8764 | 0.1344 | 0.6407 | -0.6887 [-0.7237, -0.6545] |
| CT-HMM (latent, interval-aware) | 0.8176 [0.7979, 0.8363] | 0.6123 [0.5777, 0.6472] | 0.8387 [0.8192, 0.8570] | 0.7562 [0.7376, 0.7741] | 0.8406 | 0.1479 | 0.6794 | -0.6726 [-0.7082, -0.6382] |

### 8.4 What the Indeterminate numbers say

Indeterminate has 332 of 3,361 cases. The best one-vs-rest AUC for it in the
harmonised family is **0.6666
[0.6381, 0.6957]** (Subscore GP, joint), so the
models do carry usable information about which visits will be Indeterminate.
Hard prediction throws that information away: the 3-state CT-HMM predicts
Indeterminate for 1 of 3,361 cases, giving
sensitivity 0.0000, while
Persistence (recorded label) reaches 0.1928
with no fitted parameters at all.

The reason is the decision rule, not the probabilities: Indeterminate's
posterior rarely wins an argmax against a class with six times its prior. A
constant "never Indeterminate" rule already scores
0.9012 accuracy on this task, which is why accuracy cannot see
this failure and why balanced accuracy and macro-F1 must be read beside it.

## 9. Table D — continuous clinical measurement endpoint

A different endpoint from everything above: predicting the nine continuous
domain scores at the next visit, not the subtype label. Log density is per
domain observation over 30,099 observations; RMSE and MAE are in standardised
units; coverage is the realised rate of 95% prediction intervals.

| Model | Log-density per domain obs. | RMSE (std.) | MAE (std.) | 95% coverage |
|---|---|---|---|---|
| Profile: level + rate + medication | -1.0453 | 0.7007 | 0.5011 | 0.9406 |
| Profile: level only (random walk) | -1.0492 | 0.7025 | 0.5012 | 0.9407 |
| Profile: level + rate, no medication | -1.0511 | 0.7055 | 0.5018 | 0.9400 |
| Last observation carried forward | -1.1187 | 0.7426 | 0.5212 | 0.9444 |
| Population mean trajectory | -1.4342 | 1.0129 | 0.7749 | 0.9451 |

Every value reproduces independently in a second run
(`experiments/20261111T121021Z_rich_profile_no_ledd/model_comparison_recomputed.csv`) to within 5e-4.
Do not read this table as subtype performance: the ordering here is not the
ordering in Table A, and that difference is itself a finding (section 12, Q8).

## 10. Table E — information criteria

AIC = −2logL + 2k and BIC = −2logL + k log n, recomputed from the stored
log-likelihood, parameter count and observation count for every row; all
13 reproduce the stored values to within 0.02. **Rows may only be
compared inside one comparable group.** Values are whole-cohort maximum
likelihood, not held out.

| Model | Likelihood definition | Log-likelihood | Parameters | Observations | AIC | BIC | Comparable group |
|---|---|---|---|---|---|---|---|
| Profile: level + rate + medication | density over 9 continuous domains | -45812.36 | 45 | 38,486 | 91714.73 | 92099.84 | A: rich-profile variants |
| Profile: level only (random walk) | density over 9 continuous domains | -45940.77 | 27 | 38,486 | 91935.53 | 92166.60 | A: rich-profile variants |
| Profile: level + rate, no medication | density over 9 continuous domains | -45993.34 | 36 | 38,486 | 92058.68 | 92366.77 | A: rich-profile variants |
| Markov (discrete) | conditional PMF over pairs | -2367.77 | 6 | 3,361 | 4747.55 | 4784.27 | B: observed-label chains |
| CTMC (interval-aware) | conditional PMF over pairs | -2380.50 | 6 | 3,361 | 4773.00 | 4809.72 | B: observed-label chains |
| CT-HMM (3 latent states) | joint PMF over label sequences | -2969.06 | 14 | 4,295 | 5966.12 | 6055.23 | C: joint likelihood over whole sequences |
| Markov + initial state | joint PMF over label sequences | -3087.12 | 8 | 4,295 | 6190.23 | 6241.15 | C: joint likelihood over whole sequences |
| CTMC + initial state | joint PMF over label sequences | -3099.84 | 8 | 4,295 | 6215.68 | 6266.60 | C: joint likelihood over whole sequences |
| CT-HMM (3 latent states) | joint PMF over label sequences | -2969.06 | 14 | 4,295 | 5966.12 | 6055.23 | CT-HMM (alone) |
| CT-HMM (3 hidden x 3 observed, existing model) | joint PMF over label sequences (all labelled visits) | -2969.06 | 14 | 4,295 | 5966.12 | 6055.23 | D: label-sequence likelihood, all labelled visits |
| CT-HMM (2 hidden x 3 observed) | joint PMF over label sequences (all labelled visits) | -3010.70 | 7 | 4,295 | 6035.39 | 6079.95 | D: label-sequence likelihood, all labelled visits |
| CT-HMM (2 hidden x 2 observed, Indeterminate excluded) | joint PMF over label sequences (TD/PIGD visits only) | -1613.74 | 5 | 3,812 | 3237.49 | 3268.72 | E: label-sequence likelihood, TD/PIGD visits only |
| CT-HMM (2 hidden x 2 observed, no emission floor) | joint PMF over label sequences (TD/PIGD visits only) | -1613.74 | 5 | 3,812 | 3237.49 | 3268.72 | E: label-sequence likelihood, TD/PIGD visits only |

### 10.1 Comparisons that are invalid, and why

- **Rich profile against any CT-HMM.** Group A is a density over 38,486
  continuous observations; the label-sequence groups are PMFs over 4,295 or
  3,361 discrete terms. A density and a probability mass are not on the same
  scale, so the profile's AIC near 91,700 against the CT-HMM's 5,966 says
  nothing. They are in different groups for this reason.
- **Observed-label chains against the CT-HMM using group B.** Group B's
  likelihood conditions on the first label and covers 3,361 pairs; the CT-HMM's
  covers 4,295 visits jointly. Group C exists precisely to make that comparison
  properly, by refitting the chains with an initial-state term on the same
  4,295 observations, and there the CT-HMM wins by ΔAIC
  224.
- **Band-HMM against the unbanded CT-HMM.** Different observation alphabets;
  see section 6.3.
- **The 2×2 arms against the 3×3 arms.** The 2×2 likelihood covers 3,812
  TD/PIGD visits, the 3×3 covers all 4,295. Different data, separate groups.
- **AIC against BIC.** They are not a ratio and are not divided here.

### 10.2 What the criteria prefer, and where they disagree with prediction

Within the label-sequence group over all 4,295 labelled visits, three hidden
states beat two by ΔAIC
69.3
and ΔBIC
24.7.
Held-out prediction cannot tell them apart: paired on identical rows, the
two-state arm's accuracy differs from the three-state arm's by
-0.0003 [-0.0049, +0.0042] (includes zero),
and its AUC by
-0.0017 [-0.0043, +0.0009] (includes zero).
Both statements are correct; in-sample fit and out-of-sample prediction are
different questions.

## 11. Paired comparisons

Every contrast below is computed on identical scored rows, resampling patients
rather than visits, 2,000 draws, seed 42.

| Contrast | Metric | Model A | Model B | Delta | 95% CI | Excludes zero |
|---|---|---|---|---|---|---|
| rich profile (level+rate+medication) - CT-HMM, matched set | accuracy | 0.8404 | 0.8371 | +0.0033 | [-0.0046, +0.0117] | no |
| rich profile (level+rate+medication) - CT-HMM, matched set | sensitivity_PIGD | 0.6813 | 0.6681 | +0.0132 | [-0.0066, +0.0338] | no |
| rich profile (level+rate+medication) - CT-HMM, matched set | specificity_TD | 0.9090 | 0.9100 | -0.0009 | [-0.0091, +0.0072] | no |
| rich profile (level+rate+medication) - CT-HMM, matched set | precision_PIGD | 0.7635 | 0.7619 | +0.0016 | [-0.0153, +0.0189] | no |
| rich profile (level+rate+medication) - CT-HMM, matched set | f1_PIGD | 0.7201 | 0.7119 | +0.0081 | [-0.0068, +0.0239] | no |
| rich profile (level+rate+medication) - CT-HMM, matched set | balanced_accuracy | 0.7952 | 0.7890 | +0.0061 | [-0.0045, +0.0171] | no |
| rich profile (level+rate+medication) - CT-HMM, matched set | auc_roc | 0.8902 | 0.8583 | +0.0319 | [+0.0229, +0.0419] | yes |
| rich profile (level+rate+medication) - CT-HMM, matched set | pr_auc | 0.8043 | 0.7490 | +0.0553 | [+0.0356, +0.0736] | yes |
| rich profile (level+rate+medication) - CT-HMM, matched set | brier_binary | 0.1177 | 0.1215 | -0.0038 | [-0.0076, -0.0003] | yes |
| rich profile (level+rate+medication) - CT-HMM, matched set | log_loss | 0.3766 | 0.3963 | -0.0197 | [-0.0320, -0.0078] | yes |
| rich profile without levodopa - CT-HMM, matched set | accuracy | 0.8404 | 0.8371 | +0.0033 | [-0.0048, +0.0116] | no |
| rich profile without levodopa - CT-HMM, matched set | sensitivity_PIGD | 0.6769 | 0.6681 | +0.0088 | [-0.0111, +0.0299] | no |
| rich profile without levodopa - CT-HMM, matched set | specificity_TD | 0.9109 | 0.9100 | +0.0009 | [-0.0068, +0.0092] | no |
| rich profile without levodopa - CT-HMM, matched set | precision_PIGD | 0.7662 | 0.7619 | +0.0043 | [-0.0127, +0.0216] | no |
| rich profile without levodopa - CT-HMM, matched set | f1_PIGD | 0.7188 | 0.7119 | +0.0068 | [-0.0087, +0.0225] | no |
| rich profile without levodopa - CT-HMM, matched set | balanced_accuracy | 0.7939 | 0.7890 | +0.0049 | [-0.0060, +0.0159] | no |
| rich profile without levodopa - CT-HMM, matched set | auc_roc | 0.8903 | 0.8583 | +0.0320 | [+0.0231, +0.0419] | yes |
| rich profile without levodopa - CT-HMM, matched set | pr_auc | 0.8042 | 0.7490 | +0.0552 | [+0.0356, +0.0729] | yes |
| rich profile without levodopa - CT-HMM, matched set | brier_binary | 0.1175 | 0.1215 | -0.0041 | [-0.0078, -0.0006] | yes |
| rich profile without levodopa - CT-HMM, matched set | log_loss | 0.3759 | 0.3963 | -0.0204 | [-0.0327, -0.0087] | yes |
| rich profile (level only) - CT-HMM, matched set | accuracy | 0.8440 | 0.8371 | +0.0070 | [-0.0010, +0.0148] | no |
| rich profile (level only) - CT-HMM, matched set | sensitivity_PIGD | 0.6769 | 0.6681 | +0.0088 | [-0.0105, +0.0280] | no |
| rich profile (level only) - CT-HMM, matched set | specificity_TD | 0.9161 | 0.9100 | +0.0062 | [-0.0014, +0.0141] | no |
| rich profile (level only) - CT-HMM, matched set | precision_PIGD | 0.7768 | 0.7619 | +0.0149 | [-0.0018, +0.0316] | no |
| rich profile (level only) - CT-HMM, matched set | f1_PIGD | 0.7234 | 0.7119 | +0.0115 | [-0.0041, +0.0265] | no |
| rich profile (level only) - CT-HMM, matched set | balanced_accuracy | 0.7965 | 0.7890 | +0.0075 | [-0.0030, +0.0181] | no |
| rich profile (level only) - CT-HMM, matched set | auc_roc | 0.8865 | 0.8583 | +0.0281 | [+0.0190, +0.0383] | yes |
| rich profile (level only) - CT-HMM, matched set | pr_auc | 0.7980 | 0.7490 | +0.0490 | [+0.0282, +0.0680] | yes |
| rich profile (level only) - CT-HMM, matched set | brier_binary | 0.1195 | 0.1215 | -0.0021 | [-0.0059, +0.0015] | no |
| rich profile (level only) - CT-HMM, matched set | log_loss | 0.3814 | 0.3963 | -0.0150 | [-0.0269, -0.0031] | yes |
| removing levodopa from the rich profile, profile cohort | accuracy | 0.8401 | 0.8405 | -0.0003 | [-0.0030, +0.0023] | no |
| removing levodopa from the rich profile, profile cohort | sensitivity_PIGD | 0.6762 | 0.6817 | -0.0055 | [-0.0130, +0.0011] | no |
| removing levodopa from the rich profile, profile cohort | specificity_TD | 0.9109 | 0.9090 | +0.0019 | [+0.0000, +0.0044] | no |
| removing levodopa from the rich profile, profile cohort | precision_PIGD | 0.7662 | 0.7638 | +0.0023 | [-0.0024, +0.0077] | no |
| removing levodopa from the rich profile, profile cohort | f1_PIGD | 0.7184 | 0.7204 | -0.0021 | [-0.0074, +0.0028] | no |
| removing levodopa from the rich profile, profile cohort | balanced_accuracy | 0.7935 | 0.7953 | -0.0018 | [-0.0056, +0.0017] | no |
| removing levodopa from the rich profile, profile cohort | auc_roc | 0.8903 | 0.8902 | +0.0001 | [-0.0005, +0.0007] | no |
| removing levodopa from the rich profile, profile cohort | pr_auc | 0.8042 | 0.8044 | -0.0001 | [-0.0013, +0.0010] | no |
| removing levodopa from the rich profile, profile cohort | brier_binary | 0.1175 | 0.1177 | -0.0002 | [-0.0005, +0.0001] | no |
| removing levodopa from the rich profile, profile cohort | log_loss | 0.3760 | 0.3767 | -0.0007 | [-0.0015, +0.0001] | no |
| ratio bands at width 0.30 - width 0.10 | accuracy | 0.8359 | 0.8316 | +0.0043 | [-0.0013, +0.0100] | no |
| ratio bands at width 0.30 - width 0.10 | sensitivity_PIGD | 0.6980 | 0.7079 | -0.0098 | [-0.0219, +0.0011] | no |
| ratio bands at width 0.30 - width 0.10 | specificity_TD | 0.8955 | 0.8851 | +0.0104 | [+0.0040, +0.0171] | yes |
| ratio bands at width 0.30 - width 0.10 | precision_PIGD | 0.7427 | 0.7270 | +0.0158 | [+0.0046, +0.0278] | yes |
| ratio bands at width 0.30 - width 0.10 | f1_PIGD | 0.7197 | 0.7173 | +0.0024 | [-0.0068, +0.0115] | no |
| ratio bands at width 0.30 - width 0.10 | balanced_accuracy | 0.7968 | 0.7965 | +0.0003 | [-0.0062, +0.0067] | no |
| ratio bands at width 0.30 - width 0.10 | auc_roc | 0.8784 | 0.8758 | +0.0026 | [+0.0001, +0.0051] | yes |
| ratio bands at width 0.30 - width 0.10 | pr_auc | 0.7523 | 0.7398 | +0.0125 | [+0.0026, +0.0218] | yes |
| ratio bands at width 0.30 - width 0.10 | brier_binary | 0.1199 | 0.1221 | -0.0022 | [-0.0038, -0.0007] | yes |
| ratio bands at width 0.30 - width 0.10 | log_loss | 0.3874 | 0.3930 | -0.0055 | [-0.0096, -0.0017] | yes |
| two hidden states - three hidden states, common set | accuracy | 0.8464 | 0.8467 | -0.0003 | [-0.0049, +0.0042] | no |
| two hidden states - three hidden states, common set | sensitivity_PIGD | 0.6897 | 0.6874 | +0.0023 | [-0.0085, +0.0133] | no |
| two hidden states - three hidden states, common set | specificity_TD | 0.9120 | 0.9135 | -0.0014 | [-0.0062, +0.0034] | no |
| two hidden states - three hidden states, common set | precision_PIGD | 0.7667 | 0.7690 | -0.0024 | [-0.0123, +0.0074] | no |
| two hidden states - three hidden states, common set | f1_PIGD | 0.7262 | 0.7259 | +0.0002 | [-0.0081, +0.0088] | no |
| two hidden states - three hidden states, common set | balanced_accuracy | 0.8009 | 0.8005 | +0.0004 | [-0.0053, +0.0064] | no |
| two hidden states - three hidden states, common set | auc_roc | 0.8583 | 0.8599 | -0.0017 | [-0.0043, +0.0009] | no |
| two hidden states - three hidden states, common set | pr_auc | 0.7404 | 0.7511 | -0.0107 | [-0.0210, -0.0012] | yes |
| two hidden states - three hidden states, common set | brier_binary | 0.1187 | 0.1174 | +0.0013 | [-0.0003, +0.0029] | no |
| two hidden states - three hidden states, common set | log_loss | 0.3918 | 0.3868 | +0.0050 | [+0.0007, +0.0093] | yes |
| subscore GP (independent) - CT-HMM, harmonised binary view | accuracy | 0.8409 | 0.8369 | +0.0040 | [-0.0034, +0.0113] | no |
| subscore GP (independent) - CT-HMM, harmonised binary view | sensitivity_PIGD | 0.6882 | 0.6674 | +0.0208 | [+0.0032, +0.0376] | yes |
| subscore GP (independent) - CT-HMM, harmonised binary view | specificity_TD | 0.9069 | 0.9102 | -0.0033 | [-0.0103, +0.0046] | no |
| subscore GP (independent) - CT-HMM, harmonised binary view | precision_PIGD | 0.7615 | 0.7625 | -0.0010 | [-0.0158, +0.0148] | no |
| subscore GP (independent) - CT-HMM, harmonised binary view | f1_PIGD | 0.7230 | 0.7118 | +0.0112 | [-0.0025, +0.0247] | no |
| subscore GP (independent) - CT-HMM, harmonised binary view | balanced_accuracy | 0.7975 | 0.7888 | +0.0087 | [-0.0009, +0.0179] | no |
| subscore GP (independent) - CT-HMM, harmonised binary view | auc_roc | 0.8903 | 0.8587 | +0.0316 | [+0.0226, +0.0412] | yes |
| subscore GP (independent) - CT-HMM, harmonised binary view | pr_auc | 0.8051 | 0.7497 | +0.0554 | [+0.0352, +0.0734] | yes |
| subscore GP (independent) - CT-HMM, harmonised binary view | brier_binary | 0.1199 | 0.1214 | -0.0015 | [-0.0054, +0.0025] | no |
| subscore GP (independent) - CT-HMM, harmonised binary view | log_loss | 0.3830 | 0.3961 | -0.0131 | [-0.0252, -0.0012] | yes |
| subscore GP (independent) - CT-HMM, common binary subset | accuracy | 0.8637 | 0.8594 | +0.0043 | [-0.0025, +0.0110] | no |
| subscore GP (independent) - CT-HMM, common binary subset | sensitivity_PIGD | 0.7155 | 0.6941 | +0.0215 | [+0.0054, +0.0378] | yes |
| subscore GP (independent) - CT-HMM, common binary subset | specificity_TD | 0.9233 | 0.9258 | -0.0025 | [-0.0090, +0.0042] | no |
| subscore GP (independent) - CT-HMM, common binary subset | precision_PIGD | 0.7894 | 0.7899 | -0.0005 | [-0.0158, +0.0147] | no |
| subscore GP (independent) - CT-HMM, common binary subset | f1_PIGD | 0.7507 | 0.7389 | +0.0118 | [-0.0005, +0.0241] | no |
| subscore GP (independent) - CT-HMM, common binary subset | balanced_accuracy | 0.8194 | 0.8099 | +0.0095 | [+0.0009, +0.0180] | yes |
| subscore GP (independent) - CT-HMM, common binary subset | auc_roc | 0.9006 | 0.8626 | +0.0381 | [+0.0278, +0.0494] | yes |
| subscore GP (independent) - CT-HMM, common binary subset | pr_auc | 0.8211 | 0.7554 | +0.0658 | [+0.0436, +0.0862] | yes |
| subscore GP (independent) - CT-HMM, common binary subset | brier_binary | 0.1086 | 0.1102 | -0.0016 | [-0.0058, +0.0025] | no |
| subscore GP (independent) - CT-HMM, common binary subset | log_loss | 0.3555 | 0.3697 | -0.0142 | [-0.0275, -0.0016] | yes |
| label persistence - CT-HMM, common binary subset | accuracy | 0.8572 | 0.8594 | -0.0022 | [-0.0072, +0.0030] | no |
| label persistence - CT-HMM, common binary subset | sensitivity_PIGD | 0.6802 | 0.6941 | -0.0139 | [-0.0273, -0.0012] | yes |
| label persistence - CT-HMM, common binary subset | specificity_TD | 0.9284 | 0.9258 | +0.0025 | [-0.0021, +0.0078] | no |
| label persistence - CT-HMM, common binary subset | precision_PIGD | 0.7923 | 0.7899 | +0.0024 | [-0.0092, +0.0143] | no |
| label persistence - CT-HMM, common binary subset | f1_PIGD | 0.7320 | 0.7389 | -0.0069 | [-0.0172, +0.0033] | no |
| label persistence - CT-HMM, common binary subset | balanced_accuracy | 0.8043 | 0.8099 | -0.0057 | [-0.0128, +0.0012] | no |
| label persistence - CT-HMM, common binary subset | auc_roc | 0.8043 | 0.8626 | -0.0583 | [-0.0711, -0.0463] | yes |
| label persistence - CT-HMM, common binary subset | pr_auc | 0.6306 | 0.7554 | -0.1248 | [-0.1506, -0.1020] | yes |
| label persistence - CT-HMM, common binary subset | brier_binary | 0.1428 | 0.1102 | +0.0326 | [+0.0256, +0.0400] | yes |
| label persistence - CT-HMM, common binary subset | log_loss | 3.9459 | 0.3697 | +3.5762 | [+3.1417, +4.0087] | yes |
| CT-HMM - label persistence, harmonised three-class | accuracy | 0.7539 | 0.7227 | +0.0312 | [+0.0223, +0.0411] | yes |
| CT-HMM - label persistence, harmonised three-class | balanced_accuracy | 0.5255 | 0.5484 | -0.0229 | [-0.0378, -0.0069] | yes |
| CT-HMM - label persistence, harmonised three-class | macro_f1 | 0.5037 | 0.5559 | -0.0522 | [-0.0677, -0.0358] | yes |
| CT-HMM - label persistence, harmonised three-class | auc_ovr_Indeterminate | 0.6123 | 0.5518 | +0.0605 | [+0.0316, +0.0891] | yes |
| CT-HMM - label persistence, harmonised three-class | sensitivity_Indeterminate | 0.0000 | 0.1928 | -0.1928 | [-0.2374, -0.1497] | yes |
| CT-HMM - label persistence, harmonised three-class | macro_auc_ovr | 0.7562 | 0.6858 | +0.0704 | [+0.0573, +0.0834] | yes |
| CT-HMM - label persistence, harmonised three-class | brier_multiclass_summed | 0.3688 | 0.5546 | -0.1857 | [-0.2073, -0.1644] | yes |
| CT-HMM - label persistence, harmonised three-class | log_loss | 0.6726 | 7.6620 | -6.9894 | [-7.5583, -6.4465] | yes |
| CT-HMM - label persistence, harmonised three-class | mean_predictive_loglik | -0.6726 | -7.6620 | +6.9894 | [+6.4465, +7.5583] | yes |
| subscore GP (independent) - CT-HMM, harmonised three-class | accuracy | 0.7575 | 0.7539 | +0.0036 | [-0.0029, +0.0104] | no |
| subscore GP (independent) - CT-HMM, harmonised three-class | balanced_accuracy | 0.5313 | 0.5255 | +0.0058 | [-0.0003, +0.0121] | no |
| subscore GP (independent) - CT-HMM, harmonised three-class | macro_f1 | 0.5074 | 0.5037 | +0.0037 | [-0.0019, +0.0095] | no |
| subscore GP (independent) - CT-HMM, harmonised three-class | auc_ovr_Indeterminate | 0.6627 | 0.6123 | +0.0503 | [+0.0185, +0.0836] | yes |
| subscore GP (independent) - CT-HMM, harmonised three-class | sensitivity_Indeterminate | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | no |
| subscore GP (independent) - CT-HMM, harmonised three-class | macro_auc_ovr | 0.7964 | 0.7562 | +0.0402 | [+0.0280, +0.0533] | yes |
| subscore GP (independent) - CT-HMM, harmonised three-class | brier_multiclass_summed | 0.3602 | 0.3688 | -0.0086 | [-0.0157, -0.0016] | yes |
| subscore GP (independent) - CT-HMM, harmonised three-class | log_loss | 0.6537 | 0.6726 | -0.0190 | [-0.0313, -0.0067] | yes |
| subscore GP (independent) - CT-HMM, harmonised three-class | mean_predictive_loglik | -0.6537 | -0.6726 | +0.0190 | [+0.0067, +0.0313] | yes |
| two hidden states - three hidden states, three-class | accuracy | 0.7536 | 0.7539 | -0.0003 | [-0.0046, +0.0039] | no |
| two hidden states - three hidden states, three-class | balanced_accuracy | 0.5249 | 0.5255 | -0.0006 | [-0.0046, +0.0035] | no |
| two hidden states - three hidden states, three-class | macro_f1 | 0.5030 | 0.5037 | -0.0007 | [-0.0045, +0.0031] | no |
| two hidden states - three hidden states, three-class | auc_ovr_Indeterminate | 0.5530 | 0.6123 | -0.0593 | [-0.0820, -0.0352] | yes |
| two hidden states - three hidden states, three-class | sensitivity_Indeterminate | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | no |
| two hidden states - three hidden states, three-class | macro_auc_ovr | 0.7341 | 0.7562 | -0.0221 | [-0.0302, -0.0139] | yes |
| two hidden states - three hidden states, three-class | brier_multiclass_summed | 0.3748 | 0.3688 | +0.0059 | [+0.0025, +0.0093] | yes |
| two hidden states - three hidden states, three-class | log_loss | 0.6835 | 0.6726 | +0.0109 | [+0.0053, +0.0166] | yes |
| two hidden states - three hidden states, three-class | mean_predictive_loglik | -0.6835 | -0.6726 | -0.0109 | [-0.0166, -0.0053] | yes |

## 12. Interpretation

Each answer separates the **numerical leader** from the **statistically
supported** claim. A difference whose paired interval includes zero is reported
as unresolved, not as a win.

### Q1. Highest accuracy in each valid evaluation group

| Group | Leader | Accuracy [95% CI] | Runner-up |
|---|---|---|---|
| A: rich profile, own cohort | Profile: level only (random walk) | 0.8441 [0.8285, 0.8599] | Profile: level + rate + medication (0.8405) |
| B: subtype-history models, LOPO | CT-HMM (latent, interval-aware) | 0.8369 [0.8204, 0.8541] | Direct logistic (history + interval) (0.8359) |
| C: matched comparison set | Profile: level only (random walk) | 0.8440 [0.8285, 0.8598] | Profile: level + rate + medication (0.8404) |
| D: ratio-band sweep | CT-HMM (latent, interval-aware) | 0.8369 [0.8204, 0.8541] | CT-HMM (subtype x ratio band, width 0.30) (0.8359) |
| E: two-state HMM family | CT-HMM (2 hidden x 2 observed, Indeterminate excluded) | 0.8478 [0.8318, 0.8637] | CT-HMM (2 hidden x 2 observed, no emission floor) (0.8478) |
| F2: harmonised, common binary subset | Subscore GP, independent | 0.8637 [0.8482, 0.8794] | Subscore GP, joint (0.8637) |

The harmonised group F is absent from this and every other leader table: its
arms do not share a cohort, so it cannot be ranked. Group F2 is its comparable
counterpart.

In every group the leader's interval overlaps the runner-up's, and no paired
contrast between two serious models resolves on accuracy. The whole 14-model
harmonised spread is
0.1584
wide including the null models, and
0.0181
wide among the models that beat 0.75. **There is no accuracy winner.**

### Q2. Highest AUC-ROC

| Group | Leader | AUC-ROC [95% CI] | Runner-up |
|---|---|---|---|
| A: rich profile, own cohort | Profile: level + rate, no medication | 0.8903 [0.8747, 0.9052] | Profile: level + rate + medication (0.8902) |
| B: subtype-history models, LOPO | CT-HMM (latent, interval-aware) | 0.8587 [0.8402, 0.8764] | Direct logistic (history + interval) (0.8568) |
| C: matched comparison set | Profile: level + rate, no medication | 0.8903 [0.8747, 0.9052] | Profile: level + rate + medication (0.8902) |
| D: ratio-band sweep | CT-HMM (subtype x ratio band, width 0.30) | 0.8784 [0.8616, 0.8949] | CT-HMM (subtype x ratio band, width 0.20) (0.8775) |
| E: two-state HMM family | CT-HMM (3 hidden x 3 observed, existing model) | 0.8599 [0.8414, 0.8767] | CT-HMM (2 hidden x 3 observed) (0.8583) |
| F2: harmonised, common binary subset | Subscore GP, independent | 0.9006 [0.8838, 0.9155] | Subscore GP, joint (0.9000) |

This is where the families genuinely separate, and the separation is resolved.
On the matched set the best rich-profile variant reaches
0.8903 (Profile: level + rate, no medication) against the CT-HMM's
0.8583; on the harmonised common subset the best subscore GP reaches
0.9006
against the same family's CT-HMM at
0.8626. Both gaps
have paired intervals clear of zero
(+0.0319 [+0.0229, +0.0419] (excludes zero)).

### Q3. Highest PR-AUC

| Group | Leader | PR-AUC [95% CI] | Runner-up |
|---|---|---|---|
| A: rich profile, own cohort | Profile: level + rate + medication | 0.8044 [0.7687, 0.8359] | Profile: level + rate, no medication (0.8042) |
| B: subtype-history models, LOPO | CT-HMM (latent, interval-aware) | 0.7496 [0.7106, 0.7873] | Direct logistic (history + interval) (0.7467) |
| C: matched comparison set | Profile: level + rate + medication | 0.8043 [0.7684, 0.8359] | Profile: level + rate, no medication (0.8042) |
| D: ratio-band sweep | CT-HMM (subtype x ratio band, width 0.20) | 0.7540 [0.7120, 0.7938] | CT-HMM (subtype x ratio band, width 0.30) (0.7523) |
| E: two-state HMM family | CT-HMM (3 hidden x 3 observed, existing model) | 0.7511 [0.7123, 0.7895] | CT-HMM (2 hidden x 3 observed) (0.7404) |
| F2: harmonised, common binary subset | Subscore GP, independent | 0.8211 [0.7830, 0.8527] | Subscore GP, joint (0.8197) |

PR-AUC had never been computed for the profile-against-CT-HMM comparison. It
separates them further than AUC does:
+0.0553 [+0.0356, +0.0736] (excludes zero).
Because PIGD is the minority class at 30.1% of the matched set, this
is the more informative ranking measure of the two.

### Q4. Best probability calibration

| Group | Leader by Brier | Brier [95% CI] | Runner-up |
|---|---|---|---|
| A: rich profile, own cohort | Profile: level + rate, no medication | 0.1175 [0.1087, 0.1262] | Profile: level + rate + medication (0.1177) |
| B: subtype-history models, LOPO | CT-HMM (latent, interval-aware) | 0.1214 [0.1109, 0.1316] | Direct logistic (history + interval) (0.1225) |
| C: matched comparison set | Profile: level + rate, no medication | 0.1175 [0.1087, 0.1262] | Profile: level + rate + medication (0.1177) |
| D: ratio-band sweep | CT-HMM (subtype x ratio band, width 0.30) | 0.1199 [0.1088, 0.1297] | CT-HMM (subtype x ratio band, width 0.20) (0.1209) |
| E: two-state HMM family | CT-HMM (3 hidden x 3 observed, existing model) | 0.1174 [0.1074, 0.1275] | CT-HMM (2 hidden x 3 observed) (0.1187) |
| F2: harmonised, common binary subset | Subscore GP, joint | 0.1082 [0.0992, 0.1169] | Subscore GP, independent (0.1086) |

| Group | Leader by log loss | Log loss [95% CI] | Runner-up |
|---|---|---|---|
| A: rich profile, own cohort | Profile: level + rate, no medication | 0.3760 [0.3509, 0.4010] | Profile: level + rate + medication (0.3767) |
| B: subtype-history models, LOPO | CT-HMM (latent, interval-aware) | 0.3961 [0.3685, 0.4224] | Direct logistic (history + interval) (0.4003) |
| C: matched comparison set | Profile: level + rate, no medication | 0.3759 [0.3508, 0.4009] | Profile: level + rate + medication (0.3766) |
| D: ratio-band sweep | CT-HMM (subtype x ratio band, width 0.30) | 0.3874 [0.3582, 0.4138] | CT-HMM (subtype x ratio band, width 0.20) (0.3898) |
| E: two-state HMM family | CT-HMM (3 hidden x 3 observed, existing model) | 0.3868 [0.3602, 0.4131] | CT-HMM (2 hidden x 3 observed) (0.3918) |
| F2: harmonised, common binary subset | Subscore GP, joint | 0.3538 [0.3301, 0.3773] | Subscore GP, independent (0.3555) |

| Group | Closest calibration slope to 1 | Slope [95% CI] | Lowest ECE | ECE [bootstrap 2.5-97.5%] |
|---|---|---|---|---|
| A: rich profile, own cohort | Last observation carried forward | 0.9894 [0.8923, 1.0957] | Last observation carried forward | 0.0276 [0.0237, 0.0445] |
| B: subtype-history models, LOPO | Markov (observed, discrete) | 0.9978 [0.9159, 1.0869] | Markov (observed, discrete) | 0.0004 [0.0025, 0.0204] |
| C: matched comparison set | Markov (observed, discrete) | 0.9970 [0.9158, 1.0861] | Markov (observed, discrete) | 0.0008 [0.0028, 0.0197] |
| D: ratio-band sweep | CT-HMM (subtype x ratio band, width 0.50) | 1.0038 [0.9344, 1.0867] | CT-HMM (latent, interval-aware) | 0.0159 [0.0129, 0.0317] |
| E: two-state HMM family | CT-HMM (3 hidden x 3 observed, existing model) | 0.9944 [0.9286, 1.0682] | CT-HMM (2 hidden x 2 observed, Indeterminate excluded) | 0.0150 [0.0118, 0.0309] |
| F2: harmonised, common binary subset | Markov chain, discrete | 0.9991 [0.9174, 1.0889] | Markov chain, discrete | 0.0004 [0.0014, 0.0188] |

Two warnings about that last table, both of which matter for reading it.

**The ECE interval is not an uncertainty interval.** Expected calibration error
is a mean of absolute deviations, so resampling noise can only push it up:
its bootstrap distribution sits above the point estimate, and the percentile
range routinely excludes the point estimate entirely. The ranges above are the
bootstrap distribution of the statistic, reported for completeness, and must
not be read as bounds on a true ECE. Brier and log loss, which are proper
scoring rules, carry genuine intervals.

**The best ECE and the best slope belong to weak models.** The leaders in
several groups are the observed-label Markov chain and last-observation-carried-
forward. Their probabilities sit close to the base rate, which is exactly what
a binned calibration measure rewards, while their AUC
(0.6777 for the
Markov chain against 0.8902 for the profile) is far worse. A model
can be well calibrated and nearly useless; calibration must be read beside
discrimination, never instead of it.

A slope below 1 means the probabilities are too extreme for the outcome, above
1 that they are too timid. Brier and log loss agree on the leader in every
group. On the matched set the
profile's advantage over the CT-HMM is resolved but small:
-0.0038 [-0.0076, -0.0003] (excludes zero)
on Brier and
-0.0197 [-0.0320, -0.0078] (excludes zero)
on log loss. Note which model wins the matched set on Brier and log loss: the
levodopa-free variant (0.1175 / 0.3759),
not the full model.

### Q5. Best sensitivity for PIGD

| Group | Leader | Sensitivity [95% CI] | Runner-up |
|---|---|---|---|
| A: rich profile, own cohort | Profile: level + rate + medication | 0.6817 [0.6452, 0.7141] | Profile: level only (random walk) (0.6773) |
| B: subtype-history models, LOPO | CT-HMM (latent, interval-aware) | 0.6674 [0.6289, 0.7009] | Direct logistic (history + interval) (0.6663) |
| C: matched comparison set | Profile: level + rate + medication | 0.6813 [0.6448, 0.7137] | Profile: level only (random walk) (0.6769) |
| D: ratio-band sweep | CT-HMM (subtype x ratio band, width 0.50) | 0.7188 [0.6828, 0.7489] | CT-HMM (subtype x ratio band, width 0.05) (0.7090) |
| E: two-state HMM family | CT-HMM (2 hidden x 3 observed) | 0.6897 [0.6534, 0.7234] | CT-HMM (3 hidden x 3 observed, existing model) (0.6874) |
| F2: harmonised, common binary subset | Subscore GP, joint | 0.7168 [0.6808, 0.7503] | Subscore GP, independent (0.7155) |

Sensitivity alone is not a ranking: every model here sits on a shared
accuracy plateau, so a higher PIGD sensitivity is bought with lower
specificity. On the matched set the spread is
0.0000 to
0.6813 in sensitivity against
0.8938 to
1.0000 in specificity,
which is why balanced accuracy and F1 are tabulated beside them.

### Q6. Best sensitivity for Indeterminate

**Persistence (recorded label)**, at
0.1928
[0.1497, 0.2374] on the harmonised cohort. Every
fitted model in every group scores essentially zero: of the
36 three-class arms, 29
have an Indeterminate sensitivity below 0.01. The ranking picture is different
and better — see Q2's counterpart for this class, where the best one-vs-rest
Indeterminate AUC is 0.6666
(Subscore GP, joint) — so the information is present and the argmax rule
discards it.

Macro-F1 is the one headline metric that notices this. Its leader is
**Persistence (recorded label)** at 0.5559
[0.5349, 0.5764], a model with the
second-lowest
accuracy in its group (0.7227 against the group's best
0.7575), and a log loss of 7.6620.
It is not a better model; it is the only one that ever says "Indeterminate".

### Q7. What AIC and BIC prefer inside valid groups

| Comparable group | Preferred by AIC | Delta AIC to runner-up | Preferred by BIC | Delta BIC to runner-up |
|---|---|---|---|---|
| A: rich-profile variants | Profile: level + rate + medication | 220.80 | Profile: level + rate + medication | 66.76 |
| B: observed-label chains | Markov (discrete) | 25.45 | Markov (discrete) | 25.45 |
| C: joint likelihood over whole sequences | CT-HMM (3 latent states) | 224.11 | CT-HMM (3 latent states) | 185.92 |
| CT-HMM (alone) | CT-HMM (3 latent states) | Not applicable | CT-HMM (3 latent states) | Not applicable |
| D: label-sequence likelihood, all labelled visits | CT-HMM (3 hidden x 3 observed, existing model) | 69.28 | CT-HMM (3 hidden x 3 observed, existing model) | 24.72 |
| E: label-sequence likelihood, TD/PIGD visits only | CT-HMM (2 hidden x 2 observed, Indeterminate excluded) | 0.00 | CT-HMM (2 hidden x 2 observed, Indeterminate excluded) | 0.00 |

AIC and BIC agree in every group. Two cautions. First, these are whole-cohort
fits, so they measure in-sample fit penalised for parameters, not predictive
skill. Second, the criteria and held-out prediction disagree on hidden-state
count, as section 10.2 sets out: AIC prefers three states by
69.3,
while held-out accuracy and AUC cannot separate two from three. Held-out
PR-AUC and log loss do side with three states
(-0.0107 [-0.0210, -0.0012] (excludes zero)
and
+0.0050 [+0.0007, +0.0093] (excludes zero)),
which is new here and refines the earlier conclusion that two states predict as
well as three.

### Q8. Does the rich clinical profile outperform the CT-HMM?

**On ranking and calibration, yes; on accuracy, not demonstrably.** Paired on
the same 3,020 cases:

| Metric | Profile | CT-HMM | Paired difference [95% CI] | Resolved |
|---|---|---|---|---|
| accuracy | 0.8404 | 0.8371 | +0.0033 [-0.0046, +0.0117] | no |
| sensitivity_PIGD | 0.6813 | 0.6681 | +0.0132 [-0.0066, +0.0338] | no |
| specificity_TD | 0.9090 | 0.9100 | -0.0009 [-0.0091, +0.0072] | no |
| precision_PIGD | 0.7635 | 0.7619 | +0.0016 [-0.0153, +0.0189] | no |
| f1_PIGD | 0.7201 | 0.7119 | +0.0081 [-0.0068, +0.0239] | no |
| balanced_accuracy | 0.7952 | 0.7890 | +0.0061 [-0.0045, +0.0171] | no |
| auc_roc | 0.8902 | 0.8583 | +0.0319 [+0.0229, +0.0419] | yes |
| pr_auc | 0.8043 | 0.7490 | +0.0553 [+0.0356, +0.0736] | yes |
| brier_binary | 0.1177 | 0.1215 | -0.0038 [-0.0076, -0.0003] | yes |
| log_loss | 0.3766 | 0.3963 | -0.0197 [-0.0320, -0.0078] | yes |

Four of the ten metrics resolve, and all four are probabilistic: AUC-ROC,
PR-AUC, Brier and log loss. The six that do not resolve are every
threshold-based metric, accuracy included. The structural reason is in the
outcome definition: the subtype is a **thresholded ratio** of two of the nine
domains the profile predicts, so predicting all nine better sharpens the
probability without moving many cases across the threshold. That is also why
Table D's ordering is not Table A's.

### Q9. Does the Band-HMM improve on the original CT-HMM?

**On AUC-ROC yes, at every width and with every interval clear of zero; on
accuracy, PR-AUC and Brier, no.** The paired table in section 6.2 is the
evidence. Summarising at the best width, 0.30: AUC
+0.0197 [+0.0120, +0.0285] (excludes zero),
accuracy -0.0010 [-0.0082, +0.0065] (includes zero),
PR-AUC +0.0027 [-0.0170, +0.0214] (includes zero),
Brier -0.0016 [-0.0044, +0.0013] (includes zero).
The cost is 77 parameters against 14, and the gain cannot be
priced by AIC because the alphabets differ. Describing this as an overall
improvement would be wrong; describing it as a ranking improvement is correct.

### Q10. Best balance of predictive performance and complexity

| Candidate | Parameters | Case for it | Case against it |
|---|---|---|---|
| Rich profile, level only (random walk) | 27 | best matched-set accuracy (0.8440), AUC 0.8865, 27 parameters against 45 | AUC and PR-AUC below both level+rate variants; worse Brier (0.1195) |
| Rich profile without levodopa | 36 | best matched-set Brier (0.1175) and log loss (0.3759); AUC 0.8903; needs no medication data | AIC prefers keeping levodopa by 344; needs nine clinical domains at every visit |
| CT-HMM, 3 latent states | 14 | 14 parameters, accuracy 0.8371 statistically indistinguishable from the profile's, needs only the label history | AUC 0.8583 and PR-AUC 0.7490 are resolvedly lower |
| Ratio bands at width 0.30 | 77 | best AUC of any CT-HMM variant (0.8784) | no accuracy, PR-AUC or Brier gain; 77 parameters; no valid AIC |
| Label persistence | 0 | the only model with non-zero Indeterminate sensitivity (0.1928) and the best macro-F1 (0.5559) | log loss 7.66, unusable as a probability model |

If one model must be named for the subtype endpoint, the defensible choice is
the **CT-HMM with 3 latent states** when only label history is available, and
the **rich profile without levodopa** when the nine domains are available: it
matches the full profile on every metric
(-0.0003 [-0.0030, +0.0023] (includes zero)
on accuracy,
+0.0001 [-0.0005, +0.0007] (includes zero)
on AUC) while needing no medication record. **No model is best on every
metric, and the protocols differ, so no universal winner is declared.**

## 13. Validation

| Check | Result |
|---|---|
| profile: three-class probabilities sum to 1 | PASS — min 1.000000000000 max 1.000000000000 |
| profile: no missing probabilities | PASS — 0 missing cells |
| LOPO: three-class probabilities sum to 1 | PASS — min 1.000000000000 max 1.000000000000 |
| LOPO: no missing probabilities | PASS — 0 missing cells |
| band: three-class probabilities sum to 1 | PASS — min 1.000000000000 max 1.000000000000 |
| band: no missing probabilities | PASS — 0 missing cells |
| two-state: three-class probabilities sum to 1 on the three-class arms | PASS — min 1.000000000000 max 1.000000000000 |
| two-state: no missing probabilities on the three-class arms | PASS — 0 missing cells |
| two-state: missing Indeterminate probabilities are confined to the 2x2 arms | PASS — 5872 cells, all in the two 2x2 arms, which have no Indeterminate output by construction |
| two-state: the 2x2 arms' two probabilities sum to 1 | PASS — min 1.000000000000 max 1.000000000000 |
| harmonised: all 14 models' probabilities sum to 1 | PASS — min 1.000000000000 max 1.000000000000 |
| profile: no duplicate predictions | PASS — 0 duplicate key rows |
| LOPO: no duplicate predictions | PASS — 0 duplicate key rows |
| band: no duplicate predictions | PASS — 0 duplicate key rows |
| two-state: no duplicate predictions | PASS — 0 duplicate key rows |
| harmonised: no duplicate rows | PASS — 0 duplicate key rows |
| profile: label set | PASS — ['Indeterminate', 'PIGD', 'TD'] |
| LOPO: label set | PASS — ['Indeterminate', 'PIGD', 'TD'] |
| band: label set | PASS — ['Indeterminate', 'PIGD', 'TD'] |
| two-state: label set | PASS — ['Indeterminate', 'PIGD', 'TD'] |
| harmonised: label set | PASS — ['Indeterminate', 'PIGD', 'TD'] |
| LOPO: P_PIGD equals p_PIGD_raw renormalised over TD+PIGD | PASS — confirms the two-class convention reused below for the harmonised run |
| LOPO and harmonised share the same three-class support | PASS — TD: 2115 vs 2115, Indeterminate: 332 vs 332, PIGD: 914 vs 914 |
| binary group 'A: rich profile, own cohort' has one cohort size | PASS — sizes [3021] |
| binary group 'B: subtype-history models, LOPO' has one cohort size | PASS — sizes [3029] |
| binary group 'C: matched comparison set' has one cohort size | PASS — sizes [3020] |
| binary group 'D: ratio-band sweep' has one cohort size | PASS — sizes [3029] |
| binary group 'E: two-state HMM family' has one cohort size | PASS — sizes [2936] |
| binary group 'F: harmonised subscore family' has one cohort size | FAIL — sizes [2759, 3029] - rows in this group are NOT ranked together |
| binary group 'F2: harmonised, common binary subset' has one cohort size | PASS — sizes [2759] |
| three-class group 'A: rich profile, own cohort' has one cohort size | PASS — sizes [3353] |
| three-class group 'B: subtype-history models, LOPO' has one cohort size | PASS — sizes [3361] |
| three-class group 'D: ratio-band sweep' has one cohort size | PASS — sizes [3361] |
| three-class group 'E: two-state HMM family' has one cohort size | PASS — sizes [3361] |
| three-class group 'F: harmonised subscore family' has one cohort size | PASS — sizes [3361] |
| AIC reproduces -2logL + 2k for every tabulated model | PASS — 13/13 within 0.02 |
| BIC reproduces -2logL + k log n for every tabulated model | PASS — 13/13 within 0.02 |
| continuous-endpoint metrics reproduce across the two profile runs | PASS — 20/20 cells agree |

### 13.1 Agreement with the published tables

359 of 359 comparisons against published values agree to
within 1e-4, across 7 source tables:

| Published table | Comparisons | Matching |
|---|---|---|
| experiments/20261020T104756Z_next_visit_td_pigd_prediction/model_comparison.csv | 28 | 28 |
| experiments/20261023T202640Z_roc_auc_comparison/auc_metrics.csv | 6 | 6 |
| experiments/20261103T202632Z_e8_harmonised_family_comparison/harmonised_metrics.csv | 140 | 140 |
| experiments/20261105T025024Z_cthmm_ratio_band_width_extended/model_metrics.csv | 36 | 36 |
| experiments/20261110T143200Z_two_state_hmm_confirm/common_two_class.csv | 47 | 47 |
| experiments/20261110T143200Z_two_state_hmm_confirm/native_three_class.csv | 43 | 43 |
| experiments/20261111T133201Z_consolidated_results/two_class_sensitivity_specificity.csv | 59 | 59 |

This covers every published accuracy, Brier, log loss, macro-F1, balanced
accuracy, per-class recall, predicted-class count, AUC and PR-AUC that the
source runs recorded. The band run's own AUC and PR-AUC were recomputed from
its predictions by an independent implementation and reproduce to 1e-6, which
is the strongest single check here.

### 13.2 Discrepancies and corrections found

1. **Cohort inconsistency inside the harmonised binary view.** Label
   persistence is scorable on
   2,759 cases
   where the other 13 models cover
   3,029. Its
   apparently best-in-family accuracy is an artefact of the easier subset. Group
   F2 fixes this; both are reported.
2. **Two hidden states versus three.** The earlier conclusion was that two
   states predict as well as three. On accuracy and AUC that holds. On held-out
   PR-AUC, log loss and Indeterminate ranking it does not: three states win with
   intervals clear of zero. The earlier statement was not wrong about the metrics
   it examined; it was incomplete.
3. **The two Brier conventions** differ by a factor of two and both appear in the
   repository. Every Brier value here carries its convention.
4. **Confusion-matrix orientation** differs between source runs; rows are actual
   labels in some and predicted in others. All per-class metrics here were
   recomputed from per-case predictions rather than from any stored matrix, so
   the orientation question cannot propagate.
5. **Degenerate AUCs.** The marginal arms' AUC near
   0.068 is a
   leave-one-patient-out artefact, not inverted discrimination, and the
   constant-score arms' 0.5000 is not chance-level discrimination. Both are
   flagged rather than tabulated silently.
6. **AIC prefers three hidden states while held-out accuracy cannot separate
   two from three.** Both statements stand; they measure different things.

### 13.3 Duplicate and superseded runs

Two pairs of runs are duplicates rather than conflicts, verified earlier to
reproduce each other exactly (maximum metric difference 0.0): the band-width
sweep and its extended rerun, which only adds width 0.50, and the two-state run
and its confirmation, which only adds a SLURM record. The extended and
confirming runs are cited throughout. The 2×2 "no emission floor" arm is not a
separate model: it matches the floored arm to ten decimal places with zero
discordant predictions, which is visible in Table A2.3.

## 14. What remains missing

| Metric | Models | Why it is still missing |
|---|---|---|
| AIC / BIC | Band-HMM at all five widths | likelihood is over a 44-category alphabet; the initial state distribution is not saved, and a plug-in marginal would not be a maximum (section 6.3) |
| AIC / BIC | LOCF, population mean, persistence, marginal | Not applicable: no fitted likelihood |
| AIC / BIC | subscore GP, LMM and spline families | no saved whole-cohort log-likelihood or parameter count |
| Separately trained binary Indeterminate model | all | never run; the one-vs-rest reading in section 8 is not a substitute |
| External validation | all | no second cohort in this project |
| Calibration slope and intercept | the marginal and persistence arms | Not applicable: their score takes fewer than three distinct values, so a logistic recalibration has no slope |
| Multiclass AUC for the 2x2 arms | two 2x2 CT-HMM arms | Not applicable: no Indeterminate output by construction |

`02_MISSING_METRICS_UPDATED.md` gives the full register, one row per
model-and-metric, with the file inspected and what each would need.

## 15. Sources

Every value in this document comes from one of these files.

| Role | Path |
|---|---|
| Rich profile per-case subtype predictions | experiments/20261103T220147Z_a2p_outcome_table_fullprec/derived_label_predictions.csv |
| Rich profile continuous predictions | experiments/20261103T220147Z_a2p_outcome_table_fullprec/profile_predictions.csv |
| Rich profile continuous metrics | experiments/20261103T213859Z_a2_clinical_profile_cv_rerun/model_comparison.csv |
| Levodopa-free profile rerun | experiments/20261111T121021Z_rich_profile_no_ledd |
| Subtype-history LOPO predictions | experiments/20261020T104756Z_next_visit_td_pigd_prediction/next_visit_predictions.csv |
| Ratio-band sweep predictions and metrics | experiments/20261105T025024Z_cthmm_ratio_band_width_extended |
| Ratio-band emission diagnostic | experiments/20261104T155439Z_ratio_band_emission_diagnostic |
| Two-state HMM family | experiments/20261110T143200Z_two_state_hmm_confirm |
| Harmonised 14-model family | experiments/20261103T202632Z_e8_harmonised_family_comparison |
| Published matched-set AUC | experiments/20261023T202640Z_roc_auc_comparison/auc_metrics.csv |
| Published AIC/BIC with comparison groups | experiments/20261023T194317Z_model_selection_criteria/aic_bic.csv |
| Earlier consolidation | docs/consolidated_results/ |

### 15.1 Reproducing this

```bash
~/envs/ppmi/bin/python scripts/96_final_metric_recovery.py --out experiments/20261111T170322Z_final_consolidated_results --n-boot 2000
~/envs/ppmi/bin/python scripts/97_final_tables_and_report.py --run experiments/20261111T170322Z_final_consolidated_results
```

The first script recovers every metric from the saved predictions and writes the
comparison CSVs, the cross-check and the validation checks; the second builds
the tables, figures and this report from those CSVs. Both are deterministic at
seed 42. Figures: fig1_auc_matched_set.png, fig2_auc_harmonised.png, fig3_band_width_sweep.png, fig4_per_class_sensitivity.png, fig5_indeterminate_gap.png, fig6_band_paired_deltas.png.
