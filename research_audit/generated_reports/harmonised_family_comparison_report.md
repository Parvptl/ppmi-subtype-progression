# E-8 — Harmonised multi-family comparison of next-visit subtype prediction

## 1. Experiment overview

| | |
|---|---|
| **Experiment name** | E-8, harmonised multi-family comparison |
| **Identifier in this audit** | P-18 |
| **Research question** | When five dissimilar model families are placed under one decision rule, one case list and one baseline, how far apart are they on next-visit motor-subtype prediction? |
| **Objective** | Remove the protocol differences that make published model comparisons incomparable, and measure what is left |
| **Run** | `experiments/20261103T202632Z_e8_harmonised_family_comparison/`, 2026-11-03 |
| **Script** | `scripts/79_harmonised_family_comparison.py` |
| **Status** | Completed and verified; report written by this audit |
| **Related** | P-01 (next-visit prediction), P-07 (AIC/BIC and matched held-out), M-03 (CT-HMM LOPO), G-02 to G-04 (the subscore and time-varying models being compared) |

## 2. Scientific motivation

By late October 2026 the project had fitted models in five different families
to the same cohort: chains in the observed label, a continuous-time hidden
Markov model, a hierarchical subscore mixed model, Gaussian processes on the
subscores, and a time-varying spline mixed model. Each had been evaluated on
its own terms — its own validation scheme, its own target (sometimes a
continuous subscore, sometimes a categorical label), its own treatment of the
Indeterminate class, and in most cases no shared baseline.

That makes the numbers unquotable side by side. A model that predicts a
continuous tremor score cannot be compared with one that predicts a label
unless the continuous prediction is first pushed through the same
classification rule the label came from. Without that step, any ranking
reflects the protocol as much as the model.

E-8 exists to do that step once, for every family, and to find out what the
differences between families actually are when the protocol is held fixed.

## 3. Dataset and cohort

| | |
|---|---|
| **Dataset** | PPMI sporadic-PD cohort, frozen visit panel, md5 `fb038ddbdc19ab3368e13940f398e7ba` |
| **Patients** | 934 with at least two labelled visits |
| **Visits** | 5,774 cohort visits, 4,441 carrying a subtype label |
| **Prediction cases** | **3,361** consecutive labelled pairs — every pair, with no exclusions |
| **Target** | The next visit's recorded label, **all three classes**: TD, Indeterminate, PIGD |
| **Class balance of the current label** | TD 2,261 (67.3%), Indeterminate 334 (9.9%), PIGD 766 (22.8%) |
| **Missingness** | Handled upstream by each source model; E-8 reads saved predictions and adds none |

E-8 is the only experiment in the programme that scores all three classes. Every
two-class experiment drops the 332 Indeterminate targets and works on 3,029,
3,021, 3,020 or 3,017 cases depending on the run. **Its accuracies are
therefore not comparable with theirs.**

## 4. Methodology

### What was harmonised

1. **One probability vector per model per case.** Every specification is made
   to emit a three-element probability over {TD, Indeterminate, PIGD}.
2. **One decision rule.** `argmax` of that vector. No tuned thresholds, no
   per-model calibration.
3. **One case list.** The same 3,361 transitions for every specification.
4. **One baseline.** Persistence on the recorded label — predict that the next
   visit carries the current label.

### How continuous models were converted

Models that predict subscores rather than a label are pushed through the
project's **own** classification function, `subtype.classify`, by Monte Carlo:
10,000 draws per case from the model's predictive distribution, each draw
classified, and the resulting class frequencies taken as the probability
vector. Draws use **common random numbers** across models, so differences
between specifications are not Monte Carlo noise.

Where a model supplies a joint covariance over the tremor and PIGD subscores
(the GP family and its LOCF/patient-mean baselines), draws are taken from the
joint. Where it does not (the LMM family), the draws are independent. This is
recorded per specification in `model_inventory.csv`.

### Verification of the conversion

Before any model was converted, the Monte Carlo pipeline's classifier was
checked against `subtype.classify` on 5,000 synthetic cases including 400
both-subscores-zero and 600 PIGD-zero cases: **5,000 of 5,000 agree**.

### What was *not* harmonised

The **fitting and validation schemes**. The label-space models come from
leave-one-patient-out runs (`outputs/loo/`, `outputs/loo_5hidden/`); the
subscore models come from their own LOPO runs under
`outputs/joint_subscore*/` and `outputs/time_varying/`. E-8 harmonises the
*evaluation*, not the *estimation*. This is stated here because it is the
limitation most likely to be forgotten when the table is quoted.

## 5. Experimental configuration

| Setting | Value |
|---|---|
| Monte Carlo draws per case | 10,000, common random numbers |
| Bootstrap draws | 2,000, patient-clustered |
| Seed | 42 |
| Canonical baseline | Persistence (recorded label) |
| Specifications compared | 14 (9 fitted models, 5 baselines) |
| Source predictions | read-only, from `outputs/`; nothing refitted |
| Monte Carlo error check | accuracy recomputed on half the draws; max absolute difference **0.0015** |

Source of each specification is recorded in `model_inventory.csv` — file,
model name within that file, and derivation.

## 6. Results

### 6.1 The harmonised table

Three-class accuracy on 3,361 cases, sorted as the run reports it.
`harmonised_metrics.csv`.

| Specification | Family | Accuracy | Brier (3-class) | Log loss | Macro-F1 | Recall TD | Recall Indet. | Recall PIGD |
|---|---|---|---|---|---|---|---|---|
| Subscore GP, independent | subscore GP | **0.7575** | 0.3602 | 0.6537 | 0.5074 | 0.907 | 0.000 | 0.687 |
| Subscore GP, joint | subscore GP | 0.7563 | 0.3594 | 0.6518 | 0.5070 | 0.902 | 0.000 | 0.694 |
| CT-HMM, 3 latent states | latent-state HMM | 0.7539 | 0.3688 | 0.6726 | 0.5037 | 0.910 | 0.000 | 0.666 |
| CT-HMM, 5 latent states | latent-state HMM | 0.7513 | 0.3700 | 0.6746 | 0.5011 | 0.911 | 0.000 | 0.655 |
| Time-varying LMM, constant | time-varying LMM | 0.7495 | 0.3826 | 0.6919 | 0.4977 | 0.915 | 0.000 | 0.639 |
| Joint subscore LMM | subscore LMM | 0.7495 | 0.3826 | 0.6919 | 0.4977 | 0.915 | 0.000 | 0.639 |
| Markov chain, discrete | observed-label chain | 0.7474 | 0.3896 | 0.7066 | 0.4929 | 0.933 | 0.000 | 0.589 |
| Time-varying LMM, spline | time-varying LMM | 0.7462 | 0.3794 | 0.6860 | 0.4937 | 0.921 | 0.000 | 0.614 |
| CTMC, interval-aware | observed-label chain | 0.7453 | 0.3925 | 0.7105 | 0.4909 | 0.934 | 0.000 | 0.580 |
| Persistence (subscore LOCF) | baseline | 0.7453 | 0.3723 | 0.7556 | 0.4970 | 0.897 | 0.000 | 0.665 |
| Patient mean subscore | baseline | 0.7423 | 0.3802 | 0.7594 | 0.4903 | 0.927 | 0.003 | 0.584 |
| **Persistence (recorded label)** | **baseline** | **0.7227** | 0.5546 | 7.6620 | **0.5559** | 0.864 | **0.193** | 0.589 |
| Marginal null | baseline | 0.6293 | 0.5211 | 0.8756 | 0.2575 | 1.000 | 0.000 | 0.000 |
| Population mean subscore | baseline | 0.6254 | 0.5174 | 0.8716 | 0.3460 | 0.905 | 0.000 | 0.206 |

**The nine fitted specifications span 1.22 accuracy points** — 0.7453 (CTMC)
to 0.7575 (independent subscore GP). The trivial persistence rule sits at
0.7227, 2.3 points below the best.

### 6.2 Paired contrasts against the trivial rule

Patient-clustered bootstrap, 2,000 draws, seed 42. `paired_contrasts.csv`.

| Specification | Δ accuracy | 95% CI | Excludes zero |
|---|---|---|---|
| Subscore GP, independent | +0.0348 | 0.0252 – 0.0450 | yes |
| Subscore GP, joint | +0.0336 | 0.0243 – 0.0436 | yes |
| CT-HMM, 3 latent states | +0.0312 | 0.0223 – 0.0411 | yes |
| CT-HMM, 5 latent states | +0.0286 | 0.0196 – 0.0381 | yes |
| Joint subscore LMM | +0.0268 | 0.0136 – 0.0398 | yes |
| Time-varying LMM, constant | +0.0268 | 0.0136 – 0.0398 | yes |
| Markov chain, discrete | +0.0247 | 0.0159 – 0.0334 | yes |
| Time-varying LMM, spline | +0.0235 | 0.0102 – 0.0364 | yes |
| CTMC, interval-aware | +0.0226 | 0.0135 – 0.0316 | yes |
| Persistence (subscore LOCF) | +0.0226 | 0.0146 – 0.0311 | yes |
| Patient mean subscore | +0.0196 | 0.0061 – 0.0332 | yes |
| Marginal null | −0.0934 | −0.1202 – −0.0691 | yes |
| Population mean subscore | −0.0973 | −0.1215 – −0.0723 | yes |

Every fitted specification beats the trivial rule, and every interval excludes
zero. The intervals **overlap heavily with each other**: the gap between the
best and the worst fitted model is smaller than the width of any one interval.

### 6.3 Where the margin comes from

`margin_decomposition.csv` splits each model's gain over persistence by the
patient's **current** label.

| Model | Stratum | n | Share of cases | Model acc. | Baseline acc. | Cases gained | Share of total gain |
|---|---|---|---|---|---|---|---|
| CT-HMM, 3 states | TD | 2,261 | 67.3% | 0.8107 | 0.8080 | 6 | 5.7% |
| CT-HMM, 3 states | **Indeterminate** | **334** | **9.9%** | 0.4880 | 0.1916 | **99** | **94.3%** |
| CT-HMM, 3 states | PIGD | 766 | 22.8% | 0.7023 | 0.7023 | **0** | 0.0% |
| Subscore GP, independent | TD | 2,261 | 67.3% | 0.8120 | 0.8080 | 9 | 7.7% |
| Subscore GP, independent | Indeterminate | 334 | 9.9% | 0.4910 | 0.1916 | 100 | 85.5% |
| Subscore GP, independent | PIGD | 766 | 22.8% | 0.7128 | 0.7023 | 8 | 6.8% |
| Markov chain, discrete | TD | 2,261 | 67.3% | 0.8080 | 0.8080 | 0 | 0.0% |
| Markov chain, discrete | Indeterminate | 334 | 9.9% | 0.4401 | 0.1916 | 83 | 100.0% |
| Markov chain, discrete | PIGD | 766 | 22.8% | 0.7023 | 0.7023 | 0 | 0.0% |

**94.3% of the CT-HMM's margin arises on the 9.9% of cases whose current
label is Indeterminate**, and it gains nothing at all on cases currently
recorded PIGD — where it agrees with persistence on 96.9% of predictions. For
the discrete Markov chain the concentration is total: 100% of its gain is in
the Indeterminate stratum and its predictions are identical to persistence
everywhere else.

### 6.4 No model recovers the Indeterminate class

The recall column tells its own story. Persistence predicts Indeterminate 334
times and recovers 19.3% of them. Every fitted specification predicts it
essentially never — zero, one or four times out of 3,361 — and recovers 0%.
Consequently persistence attains the **highest macro-F1 of any specification
tested** (0.5559 against the best model's 0.5074) while having the lowest
accuracy.

## 7. Interpretation

**What the results support.**

* Under one protocol, five dissimilar families land within 1.22 accuracy
  points of one another. The ceiling is a property of the task, not of any
  one model's capacity.
* Every fitted model beats a trivial rule by a margin whose interval excludes
  zero, so the task is not purely trivial.
* That margin is not spread across the cohort. It is concentrated almost
  entirely in one stratum — the cases whose current label sits in the
  Indeterminate band — and is essentially absent where the current label is
  PIGD.
* Accuracy and macro-F1 rank the specifications in opposite orders, because
  accuracy rewards abandoning the minority class and macro-F1 does not.

**What the results do not support.**

* They do not show that any family is better than another. The between-family
  differences are smaller than the within-model uncertainty.
* They do not show that the subscore GP is the best model for this task. It is
  the highest row in one table by 0.0036 accuracy over the CT-HMM, against a
  bootstrap interval roughly 0.02 wide.
* They say nothing about estimation efficiency, interpretability or clinical
  usefulness, none of which this protocol measures.

**Exploratory or confirmatory.** The harmonisation protocol, the baseline and
the margin decomposition were specified before the table was read; the
specific finding that the margin concentrates in the Indeterminate stratum was
not anticipated and should be read as a discovery within a pre-specified
analysis, not a pre-registered test.

**Dependence on the evaluation design.** Strongly. On a two-class target with
Indeterminate removed, the same CT-HMM reports accuracy 0.837 rather than
0.754, and the stratum that supplies 94.3% of its margin here does not exist
there.

## 8. Limitations

1. **Estimation was not harmonised.** Only evaluation was. Families were
   fitted under their own schemes; a difference in fitting efficiency would
   show up here as a difference in accuracy and could not be separated.
2. **Monte Carlo conversion is not free.** Models without a joint covariance
   are drawn independently, which understates the probability of
   jointly-extreme subscores. The affected specifications are flagged in
   `model_inventory.csv`.
3. **The 5-state CT-HMM's source has no stored metrics.** Its predictions come
   from `outputs/loo_5hidden/`, a directory holding prediction files only (see
   M-04 in the audit). Its accuracy exists only as computed here.
4. **One cohort, one dataset.** PPMI sporadic PD. No external replication.
5. **Argmax is a choice.** A different decision rule — for instance, one that
   penalised abandoning the minority class — would reorder the table. The
   macro-F1 column is the evidence for that.
6. **The Indeterminate stratum is small.** 334 cases. The decomposition's
   headline percentage rests on a 99-case gain.

## 9. Reproducibility

```bash
~/envs/ppmi/bin/python scripts/79_harmonised_family_comparison.py
```

| Kind | Path |
|---|---|
| Script | `scripts/79_harmonised_family_comparison.py` |
| Classification rule | `src/ppmi_subtype/subtype.py` (`classify`) |
| Input predictions | `outputs/loo/loo_predictions.csv`, `outputs/loo_5hidden/loo_predictions_all_models.csv`, `outputs/joint_subscore_gp/predictions_with_subtype.csv`, `outputs/joint_subscore/predictions_with_subtype.csv`, `outputs/time_varying/predictions_with_subtype.csv` |
| Run directory | `experiments/20261103T202632Z_e8_harmonised_family_comparison/` |
| Integrity | `MANIFEST.sha256` in the run directory; panel md5 checked at start |

Seed 42 throughout; 10,000 Monte Carlo draws with common random numbers;
2,000 patient-clustered bootstrap draws.

## 10. Manuscript relevance

This is the **main-results** experiment of the current manuscript. It supplies
the central claim — that model capacity is not the binding constraint — and
the margin decomposition that explains why. The harmonised table belongs in
the main text; the per-specification Monte Carlo error check and the model
inventory belong in supplementary material.

It supports the scientific question *"is the predictability of this endpoint
limited by the choice of model family, or by the reliability of the label?"*
and, read with O-4 (threshold sensitivity) and E-2 (observed instability), it
is one of the three legs of the label-reliability argument.

## 11. Source evidence

| File | Contents |
|---|---|
| `experiments/20261103T202632Z_e8_harmonised_family_comparison/report.txt` | The run's own narrative report |
| `…/harmonised_metrics.csv` | The 14-specification table (§6.1) |
| `…/paired_contrasts.csv` | Bootstrap contrasts against persistence (§6.2) |
| `…/margin_decomposition.csv` | Stratified margin (§6.3) |
| `…/model_inventory.csv` | Source file and derivation per specification |
| `…/monte_carlo_error.csv` | Half-draw accuracy check |
| `…/versus_committed.csv` | Comparison against previously committed numbers |
| `…/validation_checks.csv` | The run's own assertions |
| `…/fig1_harmonised_family_comparison.png/.csv` | Figure and its data |
| `scripts/79_harmonised_family_comparison.py` | Implementation |
| `docs/manuscript/05_HARMONISED_FAMILY_COMPARISON_20261104.md` | The earlier, shorter write-up |
