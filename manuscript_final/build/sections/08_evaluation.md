## 8. Evaluation protocol

### 8.1 Targets

The target of every label model is the patient's next recorded subtype label, produced by the rule of Section 4.2 from the next examination. It is not an independently verified clinical state. Accuracy at this target is accuracy at reproducing a record that itself carries measurement variability, and the whole paper should be read with that in mind. The three-class task scores all 3,361 prediction opportunities; the two-class task scores the 3,029 whose next label is TD or PIGD, with the 332 Indeterminate outcomes counted and reported but never scored. The continuous models are scored on their own targets (the ratio, or the two subscores) and, separately, on the label derived from their predictions.

### 8.2 Metrics

- **Accuracy** of the argmax (three-class) or of the 0.5 threshold on *P*<sub>PIGD</sub> (two-class); **balanced accuracy** as the mean of class recalls; per-class **precision, recall, F1** and macro-F1; **PIGD sensitivity** and **TD specificity** on the two-class task.
- **Mean per-patient accuracy** and its distribution, because visit-level accuracy over-weights patients with long follow-up.
- **Log-likelihood per prediction** (three-class) and **log loss** (two-class) as the proper score of the probability vector; persistence rules are one-hot, so their log loss is undefined and is reported clipped at a probability floor and flagged.
- **Brier score**: the multiclass Brier (sum over classes of squared probability error, range 0 to 2) for the three-class task and the two-class Brier for the two-class task; the two are not comparable.
- **Calibration**: reliability bins of the maximum predicted probability or of *P*<sub>PIGD</sub>; the **calibration gap** is the mean absolute difference between mean predicted and observed PIGD rate over bins.
- **AUC-ROC** with PIGD positive and the conditional *P*<sub>PIGD</sub> as score (ties take half credit); **PR-AUC** as average precision; **partial AUC** over false-positive rate at most 0.2, McClish-standardised so that 0.5 is chance in the region; the **operating point** (true- and false-positive rate at the 0.5 threshold).
- For continuous targets: **MAE** pooled over points and averaged within patients, **RMSE**, **CRPS**, **log predictive score**, 50/80/95% **interval coverage**, the **predictive mass below zero** (a diagnostic of Gaussian misspecification on a non-negative score), and a PIT Kolmogorov-Smirnov statistic.

### 8.3 Uncertainty

Every interval in this paper resamples **patients**, not visits: a patient with fourteen visits contributes fourteen correlated rows, and a visit-level interval would understate the uncertainty (Section 9.1 measures the design effect at 1.38 to 1.46 on the diagonal transition probabilities). The intervals are 95% percentile intervals from 2,000 bootstrap resamples with seed 42 unless stated (4,000 in the GPR and Stage 3 runs, 1,000 in the Stage 5 refits), and the same resampled patients are used for every model in a comparison, so a **paired contrast** reflects the within-patient correlation of the two models' errors. A contrast is reported as the difference of the first-named model minus the second with its interval; "excludes zero" is a description of the interval, not a hypothesis test, and McNemar *p*-values that ignore clustering are saved beside the contrasts but not used for inference. Wald intervals are reported where the original run produced them and are compared with the bootstrap in Section 9.10. Monte Carlo error of the harmonised propagation was bounded by comparing half-draw estimates (maximum difference 0.0015 in accuracy).

### 8.4 Reporting rules

The following rules were applied when writing this paper and are the reason some comparisons that the project's working documents drew are absent here.

1. Metrics from different cohorts or protocols are never placed in one comparison: LOPO three-class accuracies (3,361 cases) are not compared with two-class accuracies (3,029 cases); the five-fold profile and the LOPO CT-HMM are compared only through their paired predictions on identical cases.
2. Latent-to-latent quantities of the CT-HMM (sojourn times, horizon probabilities) are labelled as such and never presented as probabilities of the recorded label.
3. A claim that has no saved result file behind it is not made. One such claim in a module docstring (that the CT-HMM's advantage over a lookup table grows from +0.005 to +0.047 with history length) has no saved source and is not used.
4. A negative or inconclusive result is reported with the same prominence as a positive one, and the pre-registered reading rules of the audits are applied as written.
