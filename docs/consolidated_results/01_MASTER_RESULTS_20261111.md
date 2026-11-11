# Consolidated model results — PPMI motor subtype progression

**Compiled** 2026-11-11 · **Panel** `outputs/visit_panel.csv`, md5 `fb038ddbdc19ab3368e13940f398e7ba` (verified at compile time)
**Machine-readable companion** [`master_results.csv`](master_results.csv) — 64 rows, one per model and evaluation configuration
**Missing metrics and pending work** [`02_MISSING_METRICS_20261111.md`](02_MISSING_METRICS_20261111.md)

This document consolidates results from experiments that have already run. No
new model was fitted for it. Two metrics — sensitivity and specificity — were
never written down by the original runs but are determined exactly by their
saved prediction files, so they were recovered by counting; every such value is
marked as recovered and its source is given. Accuracy was independently
recomputed from the same prediction files as a check and agrees with every
published value to four decimal places.

---

## 1. Project and evaluation overview

### 1.1 The question

Parkinson's disease is clinically divided into a tremor-dominant (TD) and a
postural-instability/gait-difficulty (PIGD) motor subtype, with an
Indeterminate category between them. The subtype is not measured directly: it
is a **thresholded ratio** of two UPDRS sub-scores (tremor mean ÷ PIGD mean),
with the ratio bands ≥ 1.15 → TD, ≤ 0.90 → PIGD, and in between →
Indeterminate. Every experiment here asks a version of one question: **given a
patient's history, what will their motor subtype be at the next visit?**

Two strategies were pursued:

1. **Subtype-history models** (CT-HMM and its relatives). Work on the sequence
   of *recorded labels* and model transitions between them in continuous time.
2. **The rich clinical profile.** Ignores the label while predicting; forecasts
   the nine underlying clinical measurements, then applies the project's own
   unmodified classification rule to the predicted tremor and PIGD scores to
   obtain a label.

### 1.2 Cohort, and why the case counts differ between tables

| Quantity | Count | Note |
|---|---|---|
| Published panel | 5,774 visits, 1,080 patients | before any modelling cut |
| Visits with a recorded subtype | 4,441 | TD 2,891 · Indeterminate 441 · PIGD 1,109 |
| **Modelling cohort** (≥ 2 labelled visits) | **934 patients, 4,295 labelled visits** | 146 patients have one labelled visit and cannot contribute a transition |
| Label transitions available | **3,361** | 4,295 − 934 |
| Visits with the nine clinical domains available | 4,433 | 8 labelled visits lack required measurements |
| Clinical-profile cohort | 934 patients, 4,287 visits | → **3,353** prediction opportunities, **30,099** domain observations |
| Two-class scored cases, subtype-history LOPO | **3,029** of 3,361 | 332 cases whose next label is Indeterminate are not scorable in a TD-vs-PIGD task |
| Two-class scored cases, matched set | **3,020**, 898 patients | intersection of the profile's CV rows and the CT-HMM's LOPO rows |
| Indeterminate-excluded cohort ("Model 1") | **876 patients, 3,812 visits, 2,936 transitions** | Indeterminate visits removed *before* fitting |

Every count in this table was re-derived from the panel at compile time, and
they reconcile exactly: 4,441 − 4,295 = 146 single-visit patients =
1,080 − 934; 4,441 − 4,433 = 8 visits short of clinical data = 4,295 − 4,287 =
3,361 − 3,353 prediction pairs.

### 1.3 Evaluation protocols, and the rule for reading the tables

Results are grouped by protocol. **Metrics may be compared within a group and
must not be compared across groups** — the groups differ in cohort, in the
number of cases scored, and in the label space.

| Group | Endpoint | Cases | Patients | Validation |
|---|---|---|---|---|
| **A** | Binary TD vs PIGD, matched set | 3,020 | 898 | profile arms: 5-fold patient CV × 4 repeats; history arms: LOPO (934 folds); scored on the intersection |
| **B** | Binary TD vs PIGD | 3,029 | 899 | leave-one-patient-out, 934 folds |
| **C** | Three-class TD / Indeterminate / PIGD | 3,361 | 934 | leave-one-patient-out, 934 folds |
| **C2** | Three-class, harmonised families | 3,361 | 934 | one shared case list and argmax rule; each family keeps its native fitting scheme |
| **D** | Binary TD vs PIGD, Indeterminate excluded before fitting | 2,936 | 876 | leave-one-patient-out |
| **E** | Nine continuous clinical domains | 3,353 predictions (30,099 domain observations) | 934 | 5-fold patient CV × 4 repeats |
| **F** | Information criteria | full-cohort maximum likelihood — **not a held-out result** | 934 / 876 | — |

### 1.4 Metric definitions used here

- **Accuracy** — proportion of scored cases predicted correctly. Confidence
  intervals are patient-clustered bootstraps (2,000 draws) unless stated,
  because visits from one patient are not independent observations.
- **Sensitivity / specificity** — require a positive class. On the binary
  endpoint the positive class is **PIGD** throughout this project, so
  sensitivity = recall of PIGD and specificity = recall of TD. For the
  three-class endpoint a single sensitivity is undefined, so each class is
  scored **one-against-the-rest**.
- **AUC-ROC** — probability that a randomly chosen PIGD case is ranked above a
  randomly chosen TD case. Threshold-free: it measures *ranking*, not the
  decisions a model actually makes.
- **PR-AUC** (average precision) — area under the precision–recall curve, more
  informative than AUC-ROC when the positive class is a minority (PIGD is 30%
  of scored cases).
- **Brier score** — mean squared error of the predicted probability; lower is
  better. **The repository uses two conventions, and they differ by a factor
  of two.** See §8.1. All Brier values in groups A, B and E use the binary
  form; groups C and D use the summed multiclass form.
- **Log loss** — mean negative log of the probability given to the outcome that
  happened; lower is better, and it punishes confident mistakes severely. It is
  the same binary cross-entropy in every group.
- **Log-density** — for the continuous endpoint, the Gaussian predictive
  log-density per domain observation; higher (less negative) is better.
- **95% coverage** — proportion of outcomes falling inside the model's own 95%
  prediction interval. Should be near 0.95; it checks whether the stated
  uncertainty is honest.

---

## 2. Model descriptions

| Model | What it is | Inputs |
|---|---|---|
| **CT-HMM (3 hidden × 3 observed)** | Continuous-time hidden Markov model. Three latent states evolve by a generator matrix *Q*; the recorded label is a noisy emission of the latent state, with an emission floor of 0.5 on the Indeterminate row. Interval-aware: irregular visit spacing enters through the matrix exponential. | sequence of recorded labels + visit times |
| **CT-HMM (2 hidden × 3 observed)** | The same, with two latent states instead of three. | as above |
| **CT-HMM (2 hidden × 2 observed)** | Two latent, two observed states, fitted on the Indeterminate-excluded cohort. | as above, Indeterminate removed |
| **Ratio-banded CT-HMM / Band-HMM** | The observation is no longer the 3-level label but the **joint category (subtype × ratio band)**: the continuous tremor/PIGD ratio is binned, so the model sees *where inside* a subtype a patient sits. Bands are refitted inside every fold. | labels + binned continuous ratio |
| **Rich clinical profile** | A continuous-time local linear trend (integrated Brownian motion) per clinical domain. Each patient gets a *level* and a *rate* in each of nine domains, filtered causally by a Kalman filter. Levodopa dose (LEDD) enters the observation equation as an input, not as a domain. The subtype label is derived afterwards by the project's own rule. | nine clinical domains + LEDD + visit times |
| — *level + rate + medication* | the full model | |
| — *level only (random walk)* | rate pinned to zero; medication kept | |
| — *level + rate, no medication* | LEDD removed entirely (coefficients dropped from the parameter vector) | |
| **LOCF** | Last observation carried forward: assume no change. | last recorded measurements |
| **Population mean trajectory** | The training-set average trajectory, ignoring the individual. | none patient-specific |
| **Persistence** | Predict the last recorded (non-Indeterminate) label. | last label |
| **Markov / CTMC (observed)** | Transition models on the recorded label with no latent layer; CTMC is interval-aware, Markov is per-visit. | labels (+ times for CTMC) |
| **Marginal** | Predict the training-set class prior for everyone. | none |

---

## 3. Master performance comparison table

Cells read **"Not calculated"** where the metric was never computed and
**"Not applicable"** where it does not exist for that endpoint. Nothing is
estimated or imputed.

### 3.A Binary TD vs PIGD — matched 3,020 cases, 898 patients, positive class **PIGD**

| Model | Accuracy | 95% CI | Sens. | Spec. | AUC-ROC | PR-AUC | Brier | Log loss | AIC | BIC |
|---|---|---|---|---|---|---|---|---|---|---|
| Profile: level only (random walk) | **0.8440** | [0.8285, 0.8598] | 0.6769 | 0.9161 | Not calculated | Not calculated | 0.1195 | 0.3814 | see §7 | see §7 |
| Profile: level + rate + medication | 0.8404 | [0.8248, 0.8566] | **0.6813** | 0.9090 | **0.8902** [0.8748, 0.9053] | Not calculated | 0.1177 | 0.3766 | see §7 | see §7 |
| Profile: level + rate, no medication | 0.8404 | [0.8249, 0.8563] | 0.6769 | 0.9109 | Not calculated | Not calculated | **0.1175** | **0.3759** | see §7 | see §7 |
| Subtype-history: CT-HMM (3 latent states) | 0.8371 | [0.8202, 0.8540] | 0.6681 | 0.9100 | 0.8583 [0.8411, 0.8755] | Not calculated | 0.1215 | 0.3963 | see §7 | see §7 |
| Subtype-history: Persistence | 0.8338 | [0.8167, 0.8509] | 0.6231 | 0.9246 | 0.7927 [0.7725, 0.8127] | Not calculated | 0.1598 | 4.1959 | Not applicable | Not applicable |
| Subtype-history: Markov (observed) | 0.8298 | [0.8130, 0.8471] | 0.5901 | **0.9332** | Not calculated | Not calculated | 0.1327 | 0.4311 | see §7 | see §7 |
| Last observation carried forward | 0.8252 | [0.8083, 0.8425] | 0.6659 | 0.8938 | Not calculated | Not calculated | 0.1235 | 0.3941 | Not applicable | Not applicable |
| Population mean trajectory | 0.6914 | [0.6677, 0.7153] | 0.1319 | 0.9327 | Not calculated | Not calculated | 0.2083 | 0.6067 | Not applicable | Not applicable |

Brier is the binary form. Sensitivity and specificity are **recovered** values
(see §8.2). All eight confidence intervals overlap except the population-mean
baseline's, so **no difference among the top seven models is resolved by
accuracy alone**; §4.3 gives the paired tests that are the correct comparison.

### 3.B Binary TD vs PIGD — leave-one-patient-out, 3,029 cases, 899 patients, positive class **PIGD**

| Model | Accuracy | 95% CI | Sens. | Spec. | AUC-ROC | PR-AUC | Brier | Log loss | *k* |
|---|---|---|---|---|---|---|---|---|---|
| CT-HMM (3 latent states), unbanded | **0.8369** | [0.8204, 0.8541] | 0.6674 | 0.9102 | 0.8587 [0.8402, 0.8764] | 0.7496 [0.7106, 0.7873] | 0.1214 | 0.3961 | 14 |
| Band-HMM, band width 0.30 | 0.8359 | [0.8192, 0.8535] | 0.6980 | 0.8955 | **0.8784** [0.8616, 0.8949] | 0.7523 [0.7118, 0.7912] | **0.1199** | **0.3874** | 77 |
| Band-HMM, band width 0.05 | 0.8336 | [0.8169, 0.8513] | 0.7090 | 0.8875 | 0.8739 [0.8561, 0.8912] | 0.7330 [0.6886, 0.7749] | 0.1227 | 0.3952 | 224 |
| Band-HMM, band width 0.20 | 0.8336 | [0.8167, 0.8516] | 0.6991 | 0.8917 | 0.8775 [0.8603, 0.8944] | **0.7540** [0.7120, 0.7938] | 0.1209 | 0.3898 | 92 |
| Band-HMM, band width 0.50 | 0.8320 | [0.8152, 0.8507] | **0.7188** | 0.8809 | 0.8765 [0.8596, 0.8936] | 0.7409 [0.7010, 0.7808] | 0.1217 | 0.3916 | 65 |
| Band-HMM, band width 0.10 | 0.8316 | [0.8142, 0.8498] | 0.7079 | 0.8851 | 0.8758 [0.8588, 0.8933] | 0.7398 [0.6971, 0.7803] | 0.1221 | 0.3930 | 137 |
| Direct logistic (history + interval) | 0.8359 | [0.8195, 0.8529] | 0.6663 | 0.9092 | Not calculated | Not calculated | 0.1225 | 0.4003 | Not calculated |
| Persistence (last non-Indeterminate) | 0.8336 | [0.8173, 0.8510] | 0.6225 | 0.9248 | Not calculated | Not calculated | 0.1598 | 4.1929 | Not applicable |
| Markov (observed, discrete) | 0.8293 | [0.8129, 0.8468] | 0.5886 | 0.9333 | Not calculated | Not calculated | 0.1328 | 0.4314 | 6 |
| CTMC (observed, interval-aware) | 0.8277 | [0.8114, 0.8451] | 0.5799 | **0.9348** | Not calculated | Not calculated | 0.1343 | 0.4346 | 6 |
| Direct logistic (history only) | 0.8194 | [0.8019, 0.8372] | 0.6455 | 0.8946 | Not calculated | Not calculated | 0.1244 | 0.4055 | Not calculated |
| Marginal (two-class prior) | 0.6983 | [0.6703, 0.7278] | 0.0000 | 1.0000 | Not calculated | Not calculated | 0.2111 | 0.6134 | Not applicable |

Sensitivity and specificity for the seven subtype-history rows are
**recovered** values; accuracy recomputed from the same confusion counts
reproduces all seven published accuracies exactly (§8.2). AIC/BIC are **not
applicable** to the banded models on this table: their
likelihood is over a 20-to-73-category alphabet rather than the three-label
alphabet, so it is not comparable with the unbanded CT-HMM's (§7.3). The
unbanded CT-HMM appears twice in the sources — once in the subtype-history
baseline suite and once as the control arm of the band-width sweep — and the two
independent runs reproduce each other exactly (0.8369 / 0.1214 / 0.3961).

### 3.C Three-class TD / Indeterminate / PIGD — leave-one-patient-out, 3,361 transitions, 934 patients

Sensitivity and specificity are one-against-the-rest, per class.

| Model | Accuracy | 95% CI | Balanced acc. | Macro-F1 | Sens. TD / Ind / PIGD | Spec. TD / Ind / PIGD | Brier† | Log loss |
|---|---|---|---|---|---|---|---|---|
| CT-HMM (3 hidden × 3 observed) | **0.7539** | [0.7348, 0.7731] | 0.5255 | 0.5037 | 0.9102 / **0.0000** / 0.6663 | 0.5787 / 0.9997 / 0.8770 | **0.3688** | **0.6726** |
| CT-HMM (2 hidden × 3 observed) | 0.7536 | [0.7341, 0.7731] | 0.5249 | 0.5030 | 0.9106 / **0.0000** / 0.6641 | 0.5795 / 1.0000 / 0.8758 | 0.3748 | 0.6835 |
| Persistence (last observed label) | 0.7227 | [0.7009, 0.7437] | **0.5484** | **0.5559** | 0.8638 / **0.1928** / 0.5886 | 0.6517 / 0.9109 / 0.9068 | 0.5546 | 7.6620 |
| Marginal (training label prior) | 0.6293 | [0.6011, 0.6565] | 0.3333 | 0.2575 | 1.0000 / 0.0000 / 0.0000 | 0.0000 / 1.0000 / 1.0000 | 0.5211 | 0.8756 |

† summed multiclass convention — **not** comparable with the Brier values in
3.A and 3.B.

**The single most important row-level fact in this document:** neither CT-HMM
ever predicts Indeterminate. The 3-state model assigns that label to 1 case out
of 3,361 and the 2-state model to 0, giving a sensitivity of 0.0000 for a class
with 332 held-out cases. Persistence, a rule with no parameters, recovers 19.3%
of them. The CT-HMM's higher overall accuracy is bought entirely on the two
majority classes, which is why its balanced accuracy (0.5255) and macro-F1
(0.5037) are *worse* than persistence's (0.5484 / 0.5559).

### 3.C2 Three-class, harmonised across model families — 3,361 transitions, 934 patients

One shared case list, one argmax rule, 10,000 Monte Carlo draws per case with
common random numbers. Each family keeps its native fitting scheme, so this
isolates the predictive comparison from scoring differences.

| Model | Family | Accuracy | Macro-F1 | Brier† | Log loss |
|---|---|---|---|---|---|
| Subscore GP, independent | subscore GP | **0.7575** | 0.5074 | 0.3602 | 0.6537 |
| Subscore GP, joint | subscore GP | 0.7563 | 0.5070 | **0.3594** | **0.6518** |
| CT-HMM, 3 latent states | latent-state HMM | 0.7539 | 0.5037 | 0.3688 | 0.6726 |
| CT-HMM, 5 latent states | latent-state HMM | 0.7513 | 0.5011 | 0.3700 | 0.6746 |
| Time-varying LMM, constant | time-varying LMM | 0.7495 | 0.4977 | 0.3826 | 0.6919 |
| Joint subscore LMM | subscore LMM | 0.7495 | 0.4977 | 0.3826 | 0.6919 |
| Markov chain, discrete | observed-label chain | 0.7474 | 0.4929 | 0.3896 | 0.7066 |
| Time-varying LMM, spline | time-varying LMM | 0.7462 | 0.4937 | 0.3794 | 0.6860 |
| CTMC, interval-aware | observed-label chain | 0.7453 | 0.4909 | 0.3925 | 0.7105 |
| Persistence (subscore LOCF) | baseline | 0.7453 | 0.4970 | 0.3723 | 0.7556 |
| Patient mean subscore | baseline | 0.7423 | 0.4903 | 0.3802 | 0.7594 |
| Persistence (recorded label) | baseline | 0.7227 | **0.5559** | 0.5546 | 7.6620 |
| Marginal null | baseline | 0.6293 | 0.2575 | 0.5211 | 0.8756 |
| Population mean subscore | baseline | 0.6254 | 0.3460 | 0.5174 | 0.8716 |

† summed multiclass convention. Per-class sensitivity and specificity for all
fourteen rows are in [`master_results.csv`](master_results.csv); specificity
there is derived from each row's published recall and predicted counts against
the class supports 2,115 / 332 / 914.

The whole fourteen-model spread is **1.3 accuracy points wide between the best
model and plain subscore persistence** (0.7575 vs 0.7453), and 3.2 points from
the best model to label persistence.

### 3.D Binary TD vs PIGD, Indeterminate excluded before fitting — 2,936 cases, 876 patients

| Model | Accuracy | 95% CI | Sens. | Spec. | Balanced acc. | Macro-F1 | Brier† | Log loss | *k* | AIC | BIC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CT-HMM (2 hidden × 2 observed) | **0.8478** | [0.8318, 0.8637] | 0.6794 | **0.9183** | 0.7988 | **0.8098** | 0.2397 | 0.3955 | 5 | **3237.49** | **3268.72** |
| CT-HMM (2 hidden × 2 observed, no emission floor) | 0.8478 | [0.8318, 0.8637] | 0.6794 | 0.9183 | 0.7988 | 0.8098 | 0.2397 | 0.3955 | 5 | 3237.49 | 3268.72 |
| CT-HMM (3 hidden × 3 observed) | 0.8467 | [0.8305, 0.8625] | 0.6874 | 0.9135 | **0.8005** | 0.8098 | **0.2348** | **0.3868** | 14 | — | — |
| CT-HMM (2 hidden × 3 observed) | 0.8464 | [0.8304, 0.8625] | **0.6897** | 0.9120 | 0.8009 | 0.8097 | 0.2375 | 0.3918 | 7 | — | — |
| Persistence (last observed label) | 0.8443 | [0.8276, 0.8609] | 0.6563 | 0.9232 | 0.7897 | 0.8033 | 0.3113 | 4.3009 | Not applicable | — | — |
| Marginal (training label prior) | 0.7047 | [0.6754, 0.7334] | 0.0000 | 1.0000 | 0.5000 | 0.4134 | 0.4171 | 0.6080 | Not applicable | — | — |

† summed multiclass convention; halve for the binary form used in 3.A/3.B. The
three- and two-hidden-state models' AIC/BIC are over the all-labelled-visit
observation set, not this one, so they are shown in §7.2 rather than here.

**The emission floor is inactive in the 2 × 2 model.** With and without it the
log-likelihood agrees to ten decimal places (−1613.7437425) and every
prediction is identical (0 discordant cases). These are not two models; they
are one model reported twice.

### 3.E The rich profile's own primary target — nine clinical domains, 3,353 predictions, 934 patients

Classification metrics are **not applicable** to this endpoint: the prediction
is nine continuous numbers, not a class.

| Model | Log-density / domain obs. | 95% CI | RMSE (std.) | MAE (std.) | 95% coverage |
|---|---|---|---|---|---|
| Profile: level + rate + medication | **−1.0453** | [−1.0706, −1.0215] | **0.7007** | **0.5011** | 0.9406 |
| Profile: level only (random walk) | −1.0492 | [−1.0744, −1.0260] | 0.7025 | 0.5012 | 0.9407 |
| Profile: level + rate, no medication | −1.0511 | [−1.0765, −1.0270] | 0.7055 | 0.5018 | 0.9400 |
| Last observation carried forward | −1.1187 | [−1.1434, −1.0946] | 0.7426 | 0.5212 | 0.9444 |
| Population mean trajectory | −1.4342 | [−1.4693, −1.4021] | 1.0129 | 0.7749 | 0.9451 |

30,099 domain observations. All five coverages sit within 0.94–0.95, so the
stated uncertainties are honest. Per-domain RMSE in raw clinical units is in
`experiments/20261111T121021Z_rich_profile_no_ledd/per_domain_errors.csv`.

---

## 4. Rich Clinical Profile — detail

### 4.1 What each variant contributes

| Variant | What it adds or removes | Parameters |
|---|---|---|
| Level + rate + medication | the full model: per-domain level and rate of change, plus a levodopa coefficient per domain | 45 |
| Level only (random walk) | the rate is pinned to zero; medication kept | 27 |
| Level + rate, no medication | LEDD removed entirely — the coefficients are dropped from the parameter vector, not merely set to zero | 36 |
| LOCF | assume no change from the last recorded value | — |
| Population mean | the training-set average trajectory | — |

LEDD was never one of the nine predicted domains in any variant: it entered
only as an input to the observation equation.

### 4.2 Paired contrasts on the measurement target (group E)

Per-prediction mean log-density difference, paired on identical held-out
visits, patient-clustered 95% CI.

| Contrast | Δ log-density | 95% CI | Share of cases A better | Resolved? |
|---|---|---|---|---|
| Full profile vs LOCF | **+0.0733** | [+0.0657, +0.0813] | 75.1% | yes |
| Full profile vs population mean | **+0.3889** | [+0.3628, +0.4167] | 86.6% | yes |
| Modelling the rate of change (full vs level-only) | +0.0040 | [+0.0019, +0.0062] | 65.2% | yes, but small |
| Medication as an input (full vs no-medication) | +0.0058 | [+0.0041, +0.0075] | 62.4% | yes, but small |
| LEDD-free profile vs LOCF | **+0.0675** | [+0.0599, +0.0750] | 76.3% | yes |
| LEDD-free profile vs population mean | **+0.3831** | [+0.3573, +0.4104] | 85.8% | yes |
| LEDD-free profile vs level-only | −0.0018 | [−0.0048, +0.0010] | 49.7% | no |

Sources: `experiments/20261103T220147Z_a2p_outcome_table_fullprec/contrasts.csv`
and `experiments/20261111T121021Z_rich_profile_no_ledd/contrasts.csv`.

Both the rate term and the medication term earn their place on the density
target, and both gains are small — together about 13% of the margin the model
holds over assume-no-change.

### 4.3 Paired contrasts on the subtype label (group A), and against CT-HMM

| Contrast | Δ accuracy | 95% CI | A right / B right | McNemar *p*‡ | Resolved? |
|---|---|---|---|---|---|
| Full profile vs CT-HMM | +0.0033 | [−0.0050, +0.0112] | 81 / 71 | 0.47 | no |
| LEDD-free profile vs CT-HMM | +0.0033 | [−0.0050, +0.0113] | 82 / 72 | 0.47 | no |
| Level-only profile vs CT-HMM | +0.0070 | [−0.0010, +0.0149] | 81 / 60 | 0.09 | no |
| LEDD-free profile vs full profile | 0.0000 | [−0.0026, +0.0026] | 9 / 9 | 1.00 | no |
| **AUC: rich profile − CT-HMM** | **+0.0319** | **[+0.0229, +0.0419]** | — | — | **yes** |

‡ exact McNemar, which ignores patient clustering and is therefore
anti-conservative; the bootstrap interval is the primary test.

Sources: `experiments/20261023T194317Z_model_selection_criteria/heldout_contrasts.csv`,
`experiments/20261111T121021Z_rich_profile_no_ledd/derived_label_contrasts.csv`,
`experiments/20261023T202640Z_roc_auc_comparison/auc_contrast.csv`.

**This is the central nuance of the project.** The rich profile does not
classify the next subtype more accurately than the CT-HMM — the difference is
a third of a percentage point and the interval spans zero. But it *ranks*
patients substantially better (AUC 0.8902 vs 0.8583, +0.032, interval clear of
zero) and is better calibrated (Brier 0.1177 vs 0.1215; log loss 0.3766 vs
0.3963). The reason is visible in the discordance counts: out of 3,020 cases
the two models disagree on only 152. They are near-equivalent as classifiers at
the default threshold, and the profile's advantage is in the quality of the
probability it attaches, not in the label it picks.

### 4.4 Does adding rate of change help? Does adding medication?

**Rate of change: yes on measurements, no on the label.** On the density target
the full model beats level-only by +0.0040 [+0.0019, +0.0062], and AIC and BIC
both prefer it (ΔAIC 221, ΔBIC 67). On the subtype label the level-only variant
is the *most accurate* of all three profile arms (0.8440 vs 0.8404), though the
0.36-point gap is not resolved. Predicting the ratio's numerator and
denominator better does not reliably move a thresholded ratio.

**Medication: yes on measurements, not at all on the label.** Removing LEDD
costs +0.0058 [+0.0041, +0.0075] nats per prediction, and AIC/BIC prefer
keeping it (ΔAIC 344, ΔBIC 267). On the subtype label it costs nothing
measurable: accuracy identical to four decimals, 9 cases gained against 9 lost,
and the LEDD-free variant has the *best* Brier (0.1175) and log loss (0.3759) of
any model in group A. The loss is concentrated in autonomic (SCOPA-AUT, ΔRMSE
+0.100), non-motor (UPDRS-I, +0.058) and sleepiness (ESS, +0.034) domains, and
is negligible in tremor (+0.0003) and PIGD (+0.0016) — the two domains the
subtype rule actually reads. The LEDD-free variant also converged more cleanly
(20/20 optimiser successes against 4/20; max projected gradient 0.22 against
1.75).

---

## 5. CT-HMM — detail

### 5.1 Hidden-state count

On the three-class endpoint, two hidden states predict as well as three:
Δaccuracy −0.0003 [−0.0046, +0.0039], 28 cases gained against 29 lost,
McNemar *p* = 1.00. On the Indeterminate-excluded cohort, the 2 × 2 model is
nominally the most accurate of all (0.8478) and all pairwise differences among
the CT-HMM variants have intervals spanning zero. Against persistence, the
2 × 2 model's margin is +0.0034 [−0.0003, +0.0072] — **not resolved**.

So the latent layer is doing very little work on the label sequence. The only
CT-HMM contrast on this cohort whose interval excludes zero is against the
marginal prior (+0.1431 [+0.1159, +0.1708]).

### 5.2 Earlier protocols, reported separately

These used different cohorts or splits and must not be pooled with the above.

| Experiment | Model | Accuracy | Protocol | Source |
|---|---|---|---|---|
| M-02 | HMM (latent + misclassification) | 0.7695 | single held-out patient split, 1,002 pairs, 3-class | `outputs/model/heldout_performance.csv` |
| M-02 | Discrete Markov (per-visit) | 0.7675 | same split | same |
| M-02 | CTMC (interval-aware) | 0.7655 | same split | same |
| M-02 | Marginal (null) | 0.6248 | same split | same |
| M-03 | CT-HMM, 3 latent states | 0.7539 | LOPO, 934 folds, 3,361 transitions | `outputs/loo/loo_overall_metrics.json` |
| B-01 | HMM (latent states) | 0.7612 | shared held-out split, 825 pairs | `outputs/hbayes/head_to_head.csv` |
| B-01 | Hierarchical Bayesian growth | 0.7370 | same split | same |

The M-02 single-split numbers (0.7695) look higher than the LOPO numbers
(0.7539) for the same family. That is a protocol difference on a 1,002-pair
subset, not evidence of a better model, and the two must not be compared.

### 5.3 Binary Indeterminate-versus-other classification

**This experiment was never run.** The nearest completed work is (a) the
2 × 2 model, which *excludes* Indeterminate rather than targeting it, and (b)
the Indeterminate boundary analyses (P-09, P-10) on 287 Indeterminate-target
cases from 220 patients, which are descriptive and report no classification
metrics. The one-vs-rest rates in §3.C are the only quantification of
Indeterminate performance that exists, and they are 0.0000 sensitivity for both
CT-HMMs.

---

## 6. Ratio-Banded CT-HMM / Band-HMM — detail

### 6.1 Band construction

The continuous tremor/PIGD ratio is binned into fixed-width **main bands** over
[0, 3) plus ten **tail bands** covering [3, ∞). Bands are intersected with the
subtype label to form the joint observation alphabet; bands are refitted inside
every cross-validation fold. Complexity grows sharply as bands narrow:

| Main width | Main bands | Tail bands | Joint categories | Parameters | Median obs. / category | Empty | < 30 obs. | Obs. / param. |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 60 | 10 | 73 | 224 | 22 | 11 | 40 | 19.2 |
| 0.10 | 30 | 10 | 44 | 137 | 71 | 1 | 12 | 31.4 |
| 0.20 | 15 | 10 | 29 | 92 | 135 | 0 | 2 | 46.7 |
| 0.30 | 10 | 10 | 24 | 77 | 137 | 1 | 2 | 55.8 |
| 0.50 | 6 | 10 | 20 | 65 | 148 | 0 | 0 | 66.1 |
| unbanded | — | — | 3 | **14** | — | 0 | 0 | — |

### 6.2 Paired contrasts against the unbanded CT-HMM — 3,029 cases, 899 patients

| Band width | Δ accuracy | Δ accuracy 95% CI | Δ AUC-ROC [95% CI] | Δ PR-AUC [95% CI] | Δ Brier | Δ log loss |
|---|---|---|---|---|---|---|
| 0.05 | −0.0033 | Not calculated* | **+0.0151 [+0.0070, +0.0243]** | −0.0166 [−0.0396, +0.0057] | +0.0013 | −0.0009 |
| 0.10 | −0.0053 | [−0.0134, +0.0034] | **+0.0171 [+0.0089, +0.0260]** | −0.0098 [−0.0318, +0.0126] | +0.0007 | −0.0031 |
| 0.20 | −0.0033 | Not calculated* | **+0.0188 [+0.0110, +0.0276]** | +0.0044 [−0.0167, +0.0247] | −0.0005 | −0.0063 |
| 0.30 | −0.0010 | Not calculated* | **+0.0197 [+0.0120, +0.0285]** | +0.0027 [−0.0170, +0.0214] | −0.0015 | −0.0087 |
| 0.50 | −0.0050 | Not calculated* | **+0.0177 [+0.0101, +0.0265]** | −0.0087 [−0.0252, +0.0081] | +0.0002 | −0.0045 |

\* The sweep computed its paired **accuracy** contrasts against **width 0.10**
as the reference arm, not against the unbanded model, so only the
width-0.10-vs-unbanded interval exists. The other four Δ accuracy values are
differences of the published accuracies and **their intervals were never
computed**; they are not estimated here. The intervals that do exist for those
widths are against width 0.10: 0.05 +0.0020 [−0.0020, +0.0063], 0.20 +0.0020
[−0.0026, +0.0070], 0.30 +0.0043 [−0.0013, +0.0100], 0.50 +0.0003
[−0.0056, +0.0060] — all spanning zero, so no band width is distinguishable
from any other on accuracy either. Δ AUC-ROC and Δ PR-AUC *are* computed
directly against the unbanded model. Δ Brier and Δ log loss are differences of
the published values and carry no interval. Sources: `accuracy_contrasts.csv`,
`secondary_contrasts_vs_unbanded.csv`, `probabilistic_contrasts.csv`,
`model_metrics.csv` in
`experiments/20261105T025024Z_cthmm_ratio_band_width_extended/`.

### 6.3 Did ratio bands improve the model?

**Discrimination: yes, consistently.** AUC-ROC rises at every band width, by
+0.015 to +0.020, and every interval excludes zero. The best is width 0.30
(0.8784 vs 0.8587). A control arm — the baseline model run through the banded
code path — reproduces the unbanded result to six decimals (Δ AUC
−0.000003), so the gain is the bands, not the code path.

**Classification accuracy: no.** Every band width is *nominally less accurate*
than the unbanded model, and no interval excludes zero. Bands trade specificity
for sensitivity: at width 0.10, sensitivity rises 0.667 → 0.708 while
specificity falls 0.910 → 0.885, and the model calls 890 cases PIGD instead of
800. At the default threshold that is roughly a wash.

**Probability calibration: mixed and width-dependent.** Widths 0.20 and 0.30
improve both Brier and log loss; widths 0.05, 0.10 and 0.50 improve log loss
while leaving Brier the same or slightly worse. Only the 0.30 improvement is
better than third-decimal noise (Brier 0.1199 vs 0.1214, log loss 0.3874 vs
0.3961).

**PR-AUC: no.** Not one band width improves average precision with an interval
clear of zero, and the narrow bands make it worse. Since PIGD is the 30%
minority class, this is the metric that matters most for finding PIGD cases,
and it does not move.

**The honest summary:** ratio bands buy a real, reproducible improvement in
*ranking* at a cost of 5× to 16× more parameters, and they do not improve the
decisions the model makes, the precision–recall trade-off, or — except at width
0.30 — calibration. Describing this as "the Band-HMM beats the CT-HMM" would
overstate it.

### 6.4 State-dependent vs state-independent band probabilities

The band distribution **does** depend on the latent state. The largest
total-variation distance between two latent states' distributions over a
subtype's bands is 0.665 for TD (30 bands), 0.503 for Indeterminate (4 bands)
and 0.627 for PIGD (10 bands) — zero would mean the bands carry no information
about the latent state. Source:
`experiments/20261104T155439Z_ratio_band_emission_diagnostic/band_information.csv`.

This is a whole-cohort descriptive fit, not a held-out result, and the run says
so itself. It explains *why* the bands help ranking: they genuinely carry latent
state information. It does not establish that they help prediction.

---

## 7. AIC and BIC comparison

**How to read this section.** AIC = −2 log *L* + 2*k* and BIC = −2 log *L* +
*k* log *n* both trade fit against complexity, and among **comparable
candidates** a lower value indicates the model preferred on that trade-off.
Comparability requires the same data, the same observation units and the same
likelihood definition. Where that fails, the numbers are not a comparison at
all. AIC and BIC are never divided by one another or treated as a ratio; only
differences within a group are meaningful.

### 7.1 Valid comparisons

| Group | Model | Log-likelihood | *k* | AIC | BIC | Observations |
|---|---|---|---|---|---|---|
| **A. Rich-profile variants**<br>density over 9 continuous domains | Level + rate + medication | −45,812.36 | 45 | **91,714.73** | **92,099.84** | 38,486 |
| | Level only (random walk) | −45,940.77 | 27 | 91,935.53 | 92,166.60 | 38,486 |
| | Level + rate, no medication | −45,993.34 | 36 | 92,058.68 | 92,366.77 | 38,486 |
| **B. Observed-label chains**<br>conditional PMF over pairs | Markov (discrete) | −2,367.77 | 6 | **4,747.55** | **4,784.27** | 3,361 |
| | CTMC (interval-aware) | −2,380.50 | 6 | 4,773.00 | 4,809.72 | 3,361 |
| **C. Joint likelihood over whole sequences**<br>joint PMF over label sequences | CT-HMM (3 latent states) | −2,969.06 | 14 | **5,966.12** | **6,055.23** | 4,295 |
| | Markov + initial state | −3,087.12 | 8 | 6,190.23 | 6,241.15 | 4,295 |
| | CTMC + initial state | −3,099.84 | 8 | 6,215.68 | 6,266.60 | 4,295 |
| **D. Hidden-state count, all labelled visits**<br>joint PMF over label sequences | CT-HMM (3 hidden × 3 observed) | −2,969.06 | 14 | **5,966.12** | **6,055.23** | 4,295 |
| | CT-HMM (2 hidden × 3 observed) | −3,010.70 | 7 | 6,035.39 | 6,079.95 | 4,295 |
| **E. Hidden-state count, TD/PIGD visits only**<br>joint PMF, Indeterminate removed | CT-HMM (2 hidden × 2 observed) | −1,613.74 | 5 | **3,237.49** | **3,268.72** | 3,812 |
| | CT-HMM (2 hidden × 2 observed, no floor) | −1,613.74 | 5 | 3,237.49 | 3,268.72 | 3,812 |

Within-group conclusions:

- **A** — the full profile is preferred: ΔAIC 221 / ΔBIC 67 over level-only,
  ΔAIC 344 / ΔBIC 267 over the no-medication variant. Both extra components
  earn their parameters on the density target.
- **B** — the discrete Markov chain is preferred over the interval-aware CTMC
  (ΔAIC 25.5), which is a mild surprise: modelling the interval does not pay.
- **C** — the CT-HMM is preferred over both observed-label chains
  (ΔAIC 224–250, ΔBIC 186–211). The latent layer *does* earn its parameters on
  the likelihood, even though §5.1 shows it barely changes predictions.
- **D** — three hidden states are preferred over two (ΔAIC 69.3,
  ΔBIC 24.7). Note the tension with §5.1: the likelihood prefers three states
  while held-out prediction cannot tell two from three. The information
  criteria are in-sample; the held-out test is the one that speaks to
  generalisation.
- **E** — the two rows are the same fitted model (§3.D), so this is not a
  comparison.

### 7.2 Comparisons that are NOT valid, and why

| Attempted comparison | Why it fails |
|---|---|
| Rich-profile AIC (≈ 91,700) vs CT-HMM AIC (≈ 5,966) | Different likelihoods over different objects: a density over 9 continuous domains at 38,486 observations against a PMF over label sequences at 4,295. The profile's AIC is larger because it is scored on 9× more data of a different kind, not because it is a worse model. |
| Group B vs group C | Conditional PMF over 3,361 pairs against joint PMF over 4,295 sequence terms. |
| Group D vs group E | 4,295 all-labelled visits against 3,812 TD/PIGD-only visits — different data. |
| Banded CT-HMM vs unbanded CT-HMM | 44-category alphabet against 3-category. The banded whole-cohort fit gives log *L* = −13,123.40, *k* = 137, AIC = 26,520.80, but the run states plainly that neither the log-likelihood nor the AIC may be compared with the baseline's. Recorded as a fit diagnostic only. |
| Any AIC/BIC against a held-out accuracy | AIC/BIC here are full-cohort in-sample quantities; every accuracy in §3 is held out. |

### 7.3 Band-HMM AIC/BIC status

**Not validly calculable in a comparable form.** AIC/BIC for the banded models
are reported in this document only as the explicitly-flagged incomparable
diagnostic above. No held-out-likelihood-based model selection was run for the
band family.

---

## 8. Data validation and consistency checks

### 8.1 Two Brier conventions in the repository — resolved

| Column name | Definition | Source | Used by |
|---|---|---|---|
| `brier_two_class` | mean (*p*<sub>PIGD</sub> − 1{PIGD})², the single-outcome binary form | `src/ppmi_subtype/next_visit.py:628` | profile CV, subtype-history LOPO, ratio-band runs, band-width sweep |
| `brier` | mean Σ<sub>c</sub> (*p*<sub>c</sub> − 1{c})², which is **2×** the binary form when there are two classes | `src/ppmi_subtype/two_state_hmm.py:1138` and eight sibling modules | two-hidden-state run, harmonised comparison, LOO, CTMC |

Worked example: the 3-hidden-state model on the 2,936-case two-class view
reports `brier` = 0.2348; halved, 0.1174, is the binary form comparable with
group A and B values. **Mixing the two would make a model look twice as badly
calibrated as it is.** Every Brier cell in this document is labelled with its
convention. Log loss is the same binary cross-entropy in both families and
needs no adjustment.

### 8.2 Recovered metrics, and the accuracy cross-check

Sensitivity and specificity were never written down for the group A models.
They were recovered by counting the confusion matrix from each run's own
prediction file, with PIGD as the positive class and the same scored-case
restriction (eligible and two-class-defined) that the published accuracy uses.
As a check, accuracy was recomputed from the same counts:

| Model | Accuracy recomputed | Accuracy published | Agrees to 4 dp |
|---|---|---|---|
| Profile: level only | 0.84404 | 0.8440 | yes |
| Profile: level + rate + medication | 0.84040 | 0.8404 | yes |
| Profile: level + rate, no medication | 0.84040 | 0.8404 | yes |
| Subtype-history: CT-HMM | 0.83709 | 0.8371 | yes |
| Subtype-history: Persistence | 0.83377 | 0.8338 | yes |
| Subtype-history: Markov | 0.82980 | 0.8298 | yes |
| LOCF | 0.82517 | 0.8252 | yes |
| Population mean | 0.69139 | 0.6914 | yes |

Three further models in the matched set (direct logistic × 2, CTMC, marginal)
have recomputed accuracies but no published value in the matched-set table,
because that table kept only three subtype-history arms. Their published values
exist on the 3,029-case cohort (§3.B) and differ slightly, as expected from the
nine extra cases.

The same recovery was run on the 3,029-case leave-one-patient-out cohort for
all seven subtype-history models (§3.B). There, all seven recomputed
accuracies match their published values to four decimal places.

Sources: `experiments/20261111T133201Z_consolidated_results/accuracy_crosscheck.csv`,
`two_class_sensitivity_specificity.csv`,
`two_class_sensitivity_specificity_lopo.csv` in the same run.

### 8.3 Confusion-matrix orientation differs between runs

Two orientations are in use, and reading one as the other silently swaps
sensitivity and specificity:

- `experiments/20261104T152154Z_cthmm_ratio_banded_observations/confusion_*.csv` —
  **rows are predicted**, columns actual (row sums = predicted counts).
- `experiments/20261110T143200Z_two_state_hmm_confirm/confusion_*.csv` —
  **rows are actual**, columns predicted (row sums = class support).

Both were checked against their runs' own published per-class recalls before
use, and every value in this document is consistent with them.

### 8.4 Duplicate and superseded runs

| Comparison | Authoritative run | Superseded run | Max metric difference | Models only in the kept run |
|---|---|---|---|---|
| Band-width sweep | `20261105T025024Z_cthmm_ratio_band_width_extended` | `20261105T003425Z_cthmm_ratio_band_width_sensitivity` | **0.0** | width 0.50 |
| Two-hidden-state HMM | `20261110T143200Z_two_state_hmm_confirm` | `20261110T141727Z_two_state_hmm` | **0.0** | none |
| Rich-profile CV | `20261103T220147Z_a2p_outcome_table_fullprec` | `20261103T213859Z_a2_clinical_profile_cv_rerun`, `20261020T180515Z_clinical_profile_cv` | agree to the published 4 dp | full-precision values |

In both of the first two cases the later run reproduces the earlier one
*exactly* on every shared model and metric, so there is no conflict to resolve —
only a preferred citation. The band-width extended run adds width 0.50; the
two-state confirm run adds a SLURM job record and a reproduction check.

### 8.5 Discrepancies found and reported, not silently resolved

1. **CT-HMM accuracy differs between cohorts: 0.8369 (3,029 cases) vs 0.8371
   (3,020 cases).** Not an error — two different case sets. Both are reported,
   each in its own group.
2. **M-02 single-split accuracy (0.7695) exceeds LOPO accuracy (0.7539) for the
   same model family.** A protocol and cohort difference on 1,002 pairs. Both
   are reported, in separate tables (§5.2); they must not be pooled.
3. **AIC prefers three hidden states while held-out prediction cannot
   distinguish two from three** (§7.1 D vs §5.1). Both stand; they measure
   different things.
4. **`outputs/exploration/report.txt` contains a stale conclusion** that the
   committed code cannot reproduce. `outputs/` is read-only, so it cannot be
   corrected. Its closing item 4 must not be quoted. (Carried forward from the
   2026-11-04 audit, finding §G-1.)
5. **Macro-F1 ranks the harmonised models differently from accuracy.** Label
   persistence has the *best* macro-F1 of all fourteen (0.5559) while having
   the second-worst accuracy (0.7227); the best-accuracy model, the independent
   subscore GP, has macro-F1 0.5074. Both are reported in §3.C2; neither alone
   is the ranking.

### 8.6 Completed vs incomplete

Everything in §3–§7 is from a completed run with its own report, manifest and
recorded git commit. Experiments that were planned or are incomplete are listed
in `research_audit/incomplete_reports.md` and `research_audit/missing_reports.md`
and are **not** represented in any table here. The one experiment explicitly
requested in this consolidation that does not exist is binary
Indeterminate-versus-other classification (§5.3).

---

## 9. Interpretation and key findings

### 9.1 Which model is most accurate?

There is no single answer, because accuracy is only defined within a protocol.

- **Group A (3,020 matched cases):** profile level-only, 0.8440.
- **Group B (3,029 LOPO cases):** unbanded CT-HMM, 0.8369.
- **Group C (3,361 three-class):** CT-HMM 3-state, 0.7539 — but see §9.8.
- **Group C2 (harmonised):** independent subscore GP, 0.7575.
- **Group D (2,936, Indeterminate excluded):** CT-HMM 2 × 2, 0.8478.

**More important than any of these: within every group, the top models'
confidence intervals overlap and no paired contrast between two serious models
reaches significance.** The honest statement is that on this cohort, for this
endpoint, a dozen quite different models predict the next subtype about
equally well, and all of them sit within a couple of points of persistence —
the rule "assume the patient keeps their current subtype".

### 9.2 Which model has the highest AUC-ROC?

**The rich clinical profile, 0.8902 [0.8748, 0.9053]** on the 3,020-case set,
against the CT-HMM's 0.8583 [0.8411, 0.8755]; the paired difference +0.0319
[+0.0229, +0.0419] is the clearest resolved result in the project. The best
CT-HMM-family AUC is the Band-HMM at width 0.30, 0.8784 [0.8616, 0.8949], on
the 3,029-case set — nine cases more than the profile's set, so the two are
near-comparable but not identical.

### 9.3 Which model is best calibrated?

**The LEDD-free rich profile**: Brier 0.1175 and log loss 0.3759, the best of
both in group A, marginally ahead of the full profile (0.1177 / 0.3766) and
clearly ahead of the CT-HMM (0.1215 / 0.3963). In the CT-HMM family the
Band-HMM at width 0.30 is best (0.1199 / 0.3874). Persistence is by far the
worst calibrated despite competitive accuracy (log loss 4.1959): it emits hard
0/1 probabilities and is severely punished when wrong, which is exactly what
log loss is for.

### 9.4 Which model do AIC and BIC prefer?

Within each valid group (§7.1): the **full rich profile** among profile
variants; the **CT-HMM with 3 latent states** over observed-label chains on the
joint likelihood; **3 hidden states** over 2 on the all-labelled-visit set; and
the **discrete Markov chain** over the interval-aware CTMC among pair-conditional
chains. No cross-group preference can be stated, and the rich profile's AIC can
never be compared with the CT-HMM's.

### 9.5 Did the rich profile improve on the CT-HMM?

**Partly, and in a specific way.** Not in accuracy (+0.0033, interval spans
zero, 152 discordant cases out of 3,020). Yes in ranking (AUC +0.0319,
interval clear of zero) and in calibration (Brier and log loss both better).
On its own primary target — forecasting the nine clinical measurements — it
beats assume-no-change by +0.0733 [+0.0657, +0.0813] log-density units, which
the CT-HMM cannot be scored on at all because it does not predict
measurements.

The deeper finding is structural: the subtype is a *thresholded ratio* of two of
the nine numbers, so predicting all nine better does not translate into
predicting the threshold crossing better. The profile's advantage shows up
wherever the continuous probability is used and disappears when it is hardened
into a label.

### 9.6 Did the Ratio-Banded CT-HMM improve on the original?

**In discrimination only.** AUC-ROC improves at all five band widths by
+0.015 to +0.020 with every interval excluding zero. Accuracy does not improve
at any width. PR-AUC does not improve at any width. Calibration improves
meaningfully only at width 0.30. The cost is 65 to 224 parameters against the
baseline's 14, and at width 0.05 that leaves 40 of 73 categories with fewer
than 30 observations. If a single width is to be carried forward it is **0.30**,
which is the best AUC, the best Brier, the best log loss and the second-fewest
parameters of the banded family. §6.3 is the full statement.

### 9.7 Did LEDD and the rate of change help?

**LEDD:** yes on the measurement target (+0.0058 [+0.0041, +0.0075]; AIC/BIC
prefer keeping it), nothing on the subtype label (Δaccuracy 0.0000
[−0.0026, +0.0026]; 9 cases gained, 9 lost), and calibration is marginally
*better* without it. The loss from removing it falls in autonomic, non-motor
and sleepiness domains, not in tremor or PIGD.

**Rate of change:** yes on the measurement target (+0.0040
[+0.0019, +0.0062]; AIC/BIC prefer it), and on the subtype label the level-only
variant is nominally the *best* of the three profile arms (0.8440). Modelling
the slope helps the numbers and does not help the label.

### 9.8 Trade-offs, and why no model should be declared the winner

| If the goal is… | Prefer | Because |
|---|---|---|
| ranking patients by PIGD risk | rich clinical profile | AUC 0.8902, the only large resolved margin in the project |
| a well-calibrated probability | rich clinical profile, LEDD-free | best Brier and log loss; also the simplest profile arm to defend against treatment confounding |
| forecasting clinical measurements | rich profile, full | the only family that predicts them; beats LOCF decisively |
| a parsimonious, interpretable model of label dynamics | CT-HMM, 2 or 3 hidden states | 5–14 parameters; held-out prediction cannot distinguish 2 from 3 |
| the best label accuracy at the default threshold | none of them clearly | every serious model is within ~2 points of persistence, and no paired contrast is resolved |
| detecting Indeterminate patients | **none of them** | both CT-HMMs have 0.0000 sensitivity on 332 cases; persistence achieves 0.1928 |

The clinical-interpretability axis cuts against the statistical one. The CT-HMM
is a 14-parameter model of a 3-state process that a clinician can read directly;
the rich profile is a 45-parameter nine-domain state-space model whose
advantage is in probability quality rather than in the decisions it makes.
The Band-HMM sits awkwardly between them: a real ranking gain, no decision
gain, and 5–16× the parameters.

### 9.9 Major limitations and unresolved questions

1. **The label is a thresholded ratio, and the threshold does the work.** TD vs
   PIGD is decided by whether a continuous ratio crosses 1.15 or 0.90. Threshold
   sensitivity was measured separately (experiment O-4) and should be read
   alongside any accuracy number here.
2. **The Indeterminate class is effectively unpredicted.** 332 held-out cases,
   0.0000 sensitivity from both CT-HMMs. Any claim that these models describe
   the full three-state process is not supported.
3. **Persistence is extremely strong.** Within ~2 accuracy points of every
   model on every binary cohort, and not significantly beaten by the CT-HMM on
   the Indeterminate-excluded cohort. The marginal value of all this modelling
   over "assume no change" is small and, for accuracy, unresolved.
4. **Protocols are not uniform.** The profile arms use 5-fold CV × 4 repeats;
   the subtype-history arms use leave-one-patient-out. The matched 3,020-case
   comparison controls the *case list* but not the *resampling scheme*. A single
   protocol across all families on all endpoints has not been run.
5. **Information criteria are in-sample and cannot bridge families.** The one
   place they disagree with held-out prediction (3 vs 2 hidden states) is
   unresolved.
6. **No AIC/BIC or held-out likelihood model selection exists for the band
   family**, so the parameter cost of banding has never been formally priced.
7. **AUC and PR-AUC are missing for most models**, including every profile
   variant except the full one and every subtype-history baseline except the
   CT-HMM and persistence. Several conclusions about ranking therefore rest on
   two or three models rather than the full set. See
   [`02_MISSING_METRICS_20261111.md`](02_MISSING_METRICS_20261111.md).
8. **Nothing here is externally validated.** Every number is from one cohort
   (PPMI), one frozen panel, and in most cases one seed.

---

## 10. Source references

### 10.1 Experiment directories cited

| Run | Supplies |
|---|---|
| `experiments/20261103T220147Z_a2p_outcome_table_fullprec` | rich-profile CV at full precision: group E, group A predictions, density contrasts |
| `experiments/20261020T180515Z_clinical_profile_cv` | group A accuracy/Brier/log-loss table including the subtype-history arms |
| `experiments/20261020T104756Z_next_visit_td_pigd_prediction` | subtype-history LOPO suite: group B baselines |
| `experiments/20261023T194317Z_model_selection_criteria` | §7.1 groups A–C, matched held-out comparison, profile-vs-CT-HMM contrast |
| `experiments/20261023T202640Z_roc_auc_comparison` | AUC-ROC for profile, CT-HMM, persistence on 3,020 cases |
| `experiments/20261023T214819Z_stage_wise_evaluation` | stage-wise accuracy by visit index (not tabulated above) |
| `experiments/20261103T202632Z_e8_harmonised_family_comparison` | group C2 |
| `experiments/20261104T152154Z_cthmm_ratio_banded_observations` | Band-HMM at width 0.10, confusion and per-class tables, control arm |
| `experiments/20261104T155439Z_ratio_band_emission_diagnostic` | §6.4 state-dependence, banded whole-cohort fit and its incomparability note |
| `experiments/20261104T172138Z_ratio_banded_auc_roc` | Band-HMM AUC/PR-AUC at width 0.10, operating points |
| `experiments/20261105T025024Z_cthmm_ratio_band_width_extended` | band-width sweep 0.05–0.50: §6.1, §6.2, group B banded rows |
| `experiments/20261110T143200Z_two_state_hmm_confirm` | groups C and D, hidden-state AIC/BIC, three-class confusion matrices |
| `experiments/20261111T121021Z_rich_profile_no_ledd` | LEDD-free contrasts, per-domain errors, §4.4 |
| `experiments/20261111T133201Z_consolidated_results` | recovered sensitivity/specificity, accuracy cross-check, duplicate-run audit, Brier-convention audit |
| `experiments/20261111T133215Z_master_results_table` | the machine-readable master table |
| `research_audit/` | the 2026-11-04 repository audit: experiment inventory, contradictions, report status |

### 10.2 Reproducing this document's derived values

```bash
~/envs/ppmi/bin/python scripts/94_consolidated_results.py --out experiments/<new>
~/envs/ppmi/bin/python scripts/95_master_results_table.py --out experiments/<new> --sens experiments/<the 94 run>
```

Script 94 recovers sensitivity/specificity, cross-checks accuracy, audits the
duplicate runs and the Brier conventions. Script 95 assembles
`master_results.csv` by reading each metric from the run that produced it;
nothing is retyped.
