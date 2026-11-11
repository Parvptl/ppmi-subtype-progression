# Metrics that could not be recovered, and what each one needs

**Compiled** 2026-11-11 · companion to [`01_MASTER_RESULTS_20261111.md`](01_MASTER_RESULTS_20261111.md)

Each entry says exactly which experiment output, confusion matrix or
calculation is required. Entries are sorted by how cheap they are to obtain.
Nothing in this list was estimated or filled in anywhere in the consolidation.

---

## A. Recoverable from files that already exist (no refitting)

These need a short script over prediction files that are already on disk. No
model has to be refitted, so none of them changes any existing result.

### A1. AUC-ROC for the models that were never scored for ranking

**Missing for:** rich profile *level-only* and *no-medication* variants; LOCF;
population mean; Markov (observed); CTMC (observed); both direct-logistic
models; marginal prior. AUC-ROC exists only for three models on the 3,020-case
set (full profile, CT-HMM, persistence) and for the six band arms on the
3,029-case set.

**What is needed:** `scripts/66_roc_auc_comparison.py` hard-codes
`RP_MODEL = "Profile: level + rate + medication"` and scores two named models.
The required change is to loop over every `model` value present in
`experiments/20261103T220147Z_a2p_outcome_table_fullprec/derived_label_predictions.csv`
and `experiments/20261020T104756Z_next_visit_td_pigd_prediction/next_visit_predictions.csv`.
Both files already carry `P_PIGD`, `actual_next_label`, `eligible` and
`two_class_defined`, which is everything the AUC and its patient-clustered
bootstrap need.

**Why it matters:** the headline claim that the rich profile *ranks* better than
the CT-HMM (+0.0319) rests on two models. Without AUC for the level-only and
no-medication variants it is not known whether the ranking advantage comes from
the rate term, the medication term, or neither.

### A2. PR-AUC (average precision) for everything outside the band runs

**Missing for:** all of group A (8 models), all of group C and C2 (18 models),
all of group D (6 models), and the seven subtype-history models in group B.
PR-AUC exists only for the unbanded CT-HMM and the five band widths in
`experiments/20261105T025024Z_cthmm_ratio_band_width_extended/model_metrics.csv`.

**What is needed:** the same loop as A1, computing average precision instead of
AUC. The band-width run already contains a working implementation with a
patient-clustered bootstrap (`pr_auc`, `pr_ci_low`, `pr_ci_high`); it needs to
be pointed at the other prediction files.

**Why it matters:** PIGD is 30% of scored cases, so PR-AUC is the more honest
discrimination metric. It is also the one metric on which the ratio bands did
**not** improve, which makes its absence for the rich profile a gap in the
central comparison.

### A3. Multiclass AUC for the three-class endpoint

**Missing for:** every model in groups C and C2.

**What is needed:** one-vs-rest AUC per class, or a Hand–Till multiclass AUC,
from the full probability vectors. `experiments/20261110T143200Z_two_state_hmm_confirm/predictions.csv`
and `experiments/20261103T202632Z_e8_harmonised_family_comparison/harmonised_predictions.csv`
both carry `p_TD`, `p_Indeterminate`, `p_PIGD` per case.

**Why it matters:** the finding that both CT-HMMs have 0.0000 sensitivity for
Indeterminate is an *argmax* result. A one-vs-rest AUC would show whether the
models rank Indeterminate cases higher at all while never making them the most
probable class — a completely different conclusion about whether the
information is present but the decision rule is wrong.

### A4. Balanced accuracy and macro-F1 for the binary groups

**Missing for:** every model in groups A and B. Both are published for groups C
and D.

**What is needed:** derivable directly from the confusion counts already in
`experiments/20261111T133201Z_consolidated_results/two_class_sensitivity_specificity.csv`
and `..._lopo.csv`. Balanced accuracy is (sensitivity + specificity) / 2;
macro-F1 needs the per-class F1 from the same four counts.

**Why it matters:** PIGD is the minority class, so accuracy is dominated by TD
performance. Leaving the degenerate population-mean baseline aside, group A
models span 0.5901 to 0.6813 in sensitivity while their accuracies span only
0.8252 to 0.8440 — balanced accuracy would separate them better than accuracy
does.

### A5. Calibration summaries (slope, intercept, expected calibration error)

**Missing as summary statistics for:** every model in every group. Calibration
*tables* (binned observed-vs-predicted) exist in
`experiments/20261104T152154Z_cthmm_ratio_banded_observations/calibration_*.csv`,
`experiments/20261105T025024Z_cthmm_ratio_band_width_extended/calibration.csv`
and `experiments/20261110T143200Z_two_state_hmm_confirm/calibration_common_*.csv`,
and the band-width run reports a `calibration_gap_delta`, but no run reduces
them to a slope, an intercept or an ECE.

**What is needed:** a logistic recalibration fit (predicted log-odds → outcome)
per model over the existing prediction files, plus a binned ECE.

**Why it matters:** Brier and log loss mix discrimination and calibration. The
claim that the rich profile is "better calibrated" than the CT-HMM would be
much better supported by a calibration slope near 1 than by a Brier difference
of 0.004.

### A6. Per-domain 95% coverage and per-domain log-density for group E

**Missing for:** all five models on the nine-domain endpoint. Only the pooled
coverage (0.9400–0.9451) and per-domain RMSE/MAE exist.

**What is needed:** the per-domain breakdown is already computed inside
`cp.profile_metrics` but only RMSE and MAE are kept in the `per_domain` records;
coverage and log-density are aggregated away. Adding two fields to that
dictionary and re-running `scripts/95_master_results_table.py` would recover
them from the saved predictions without refitting, since
`profile_predictions.csv` stores `z_pred_*`, `z_sd_*` and `z_actual_*` per
domain.

**Why it matters:** the pooled coverage being near 0.95 does not guarantee that
every domain is calibrated; one badly-covered domain can be masked by eight
good ones.

### A7. Confidence intervals for the harmonised family comparison

**Missing for:** all fourteen models in group C2 — accuracy, Brier and log loss
are point estimates with no interval.

**What is needed:** the patient-clustered bootstrap already used elsewhere
(`loo.bootstrap_patient_ci`), applied to
`experiments/20261103T202632Z_e8_harmonised_family_comparison/harmonised_predictions.csv`.
The run reports Monte Carlo error (`monte_carlo_error.csv`) but that is the
error from the 10,000 draws, not sampling error over patients.

**Why it matters:** the whole fourteen-model spread is 1.3 accuracy points.
Without intervals it is impossible to say whether the subscore GP's lead over
the CT-HMM (+0.0036) means anything, and the report currently cannot claim it
does.

---

## B. Needs a new calculation, but no new model fit

### B1. Comparable information criteria for the Band-HMM family

**Missing for:** all five band widths.

**Why it is not simply missing:** the banded likelihood is over a 20-to-73
category alphabet while the baseline's is over three, so their AICs are not on
the same scale. The band-emission run states this explicitly and reports its
numbers (log *L* = −13,123.40, *k* = 137, AIC = 26,520.80) as a fit diagnostic
only.

**What is needed:** one of two redesigns —
(a) **marginalise the banded model's likelihood down to the 3-label alphabet**,
    summing the joint category probabilities within each subtype, which puts
    both models on the same observation space and makes AIC/BIC comparable; or
(b) **held-out predictive log-likelihood** on the 3-label endpoint, which is
    protocol-free and already the natural quantity given everything else here
    is cross-validated.

Option (b) reuses the existing fold structure and is the cheaper of the two.

**Why it matters:** the Band-HMM costs 5× to 16× the parameters of the
unbanded model for a ranking gain of about 0.02 AUC. That trade has never been
formally priced.

### B2. Held-out likelihood model selection across the two strategies

**Missing entirely.** No quantity in this project can compare the rich profile
with the CT-HMM on a likelihood basis, because one is a density over nine
continuous domains and the other a PMF over labels.

**What is needed:** both models' **held-out predictive log-likelihood of the
next recorded label** on the same case list. The profile's label
probabilities already exist (`derived_label_predictions.csv` carries
`p_TD_raw`, `p_Indeterminate_raw`, `p_PIGD_raw`), as do the CT-HMM's, so this
is a comparison of two log-loss values on a shared 3-class endpoint — which is
close to what group C already does for the CT-HMM, but the profile has never
been scored natively on the three-class endpoint.

**Why it matters:** it is the only principled way to answer "which strategy
models the label better", and it would subsume the accuracy/AUC/Brier
patchwork currently used.

---

## C. Requires an experiment that has never been run

### C1. Binary Indeterminate-versus-other classification

**Status: not run.** The consolidation brief asked for this as a distinct
evaluation and it does not exist in any form.

**The nearest completed work, and why it is not a substitute:**

| Existing work | Why it does not answer the question |
|---|---|
| CT-HMM 2 hidden × 2 observed | *excludes* Indeterminate before fitting; it never sees the class |
| Three-class one-vs-rest rates (§3.C of the main report) | an argmax by-product, not a model trained or thresholded for the task; both CT-HMMs score 0.0000 |
| P-09 indeterminate boundary analysis (`experiments/20261023T203431Z_indeterminate_boundary`) | descriptive agreement counts on 287 Indeterminate-target cases from 220 patients; reports no classification metric |
| P-10 indeterminate ratio scrutiny (`experiments/20261023T211836Z_indeterminate_ratio_scrutiny`) | a visit-level reproduction check of both models' probabilities; no classification metric |

**What is needed:** relabel the endpoint as Indeterminate vs {TD, PIGD}, which
makes all 3,361 transitions scorable instead of 3,029, and score the existing
models' `p_Indeterminate` against it — accuracy, sensitivity, specificity,
AUC-ROC, PR-AUC, Brier, log loss, with a patient-clustered bootstrap. The
probabilities already exist; only the endpoint definition and the scoring are
new. Note that the class is 332 of 3,361 cases (9.9%), so PR-AUC and
sensitivity at a chosen operating point matter far more than accuracy, which a
constant "never Indeterminate" rule would score at 0.901.

**Why it matters:** Indeterminate is the clinically ambiguous middle of the
scale and the class every model currently fails on. It is also the class whose
boundary the threshold-sensitivity work (experiment O-4) shows to be fragile.

### C2. A single uniform validation protocol across all families

**Status: not run.** The rich-profile arms use 5-fold patient cross-validation
with 4 repeats; the subtype-history arms use leave-one-patient-out. The matched
3,020-case comparison controls the **case list** but not the **resampling
scheme**.

**What is needed:** either the rich profile under leave-one-patient-out (priced
at roughly 153 s per fit × 934 folds × 4 variants ≈ 8 hours, which is why
K-fold was chosen and recorded), or the subtype-history models under the same
5-fold × 4-repeat scheme, which is much cheaper and has not been done.

**Why it matters:** every cross-family statement in the consolidation carries
this caveat. The cheaper direction (history models under K-fold) would remove
it.

### C3. External validation

**Status: not run, and not possible with the data in this repository.** Every
result comes from one cohort (PPMI), one frozen panel
(md5 `fb038ddbdc19ab3368e13940f398e7ba`), and in most cases one seed. No
independent cohort has been scored.

---

## D. Values that exist but should not be used

| Value | Where | Why not |
|---|---|---|
| Banded CT-HMM AIC = 26,520.80 | `experiments/20261104T155439Z_ratio_band_emission_diagnostic/report.txt` | over a 44-category alphabet; not comparable with any other AIC in the project, as the run itself states |
| Closing item 4 of `outputs/exploration/report.txt` | read-only `outputs/` | a stale conclusion the committed code cannot reproduce; recorded as finding §G-1 of the 2026-11-04 audit and uncorrectable because `outputs/` is frozen |
| `brier` values read as if binary | groups C, C2, D | summed multiclass convention, exactly 2× the binary form; halve before comparing with groups A, B and E |
| M-02 single-split accuracies (0.7655–0.7695) | `outputs/model/heldout_performance.csv` | a 1,002-pair single split; higher than the same family's LOPO numbers for protocol reasons, not model reasons |
| "CT-HMM (2 hidden × 2 observed, no emission floor)" as a separate model | `experiments/20261110T143200Z_two_state_hmm_confirm` | identical to the floored model to ten decimal places with 0 discordant predictions; it is one model reported twice |
