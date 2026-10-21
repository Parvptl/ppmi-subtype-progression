# PPMI motor-subtype progression — consolidated research report

**Date:** 2026-10-16.
**Status:** synthesis of completed work. This report sits **beside** the earlier reports; it does not replace, delete or rewrite any of them.

Every number is taken from the source given in brackets:

| tag | meaning |
|---|---|
| **[R: path]** | reported in an existing report or output file |
| **[V]** | independently checked while preparing this report (code reading or arithmetic on saved outputs) |
| **[C]** | sources that differ, reconciled explicitly |

No new analysis was run for this report.

**Repository state:** `main` at `3a860ca`, two commits ahead of `origin/main`. The Part C code and plan are uncommitted; see Appendix C.

---

## Executive summary

**1. Measurement resolution limits everything downstream.** The TD/Indeterminate/PIGD label comes from a ratio of two item-sum means.

- The PIGD item sum (0–20) is 0 at 16.3 % of labelled visits and only 1–2 points at most of the rest [R: Part C1].
- The Indeterminate band (0.90–1.15) contains only 11 attainable ratio values [R: script 08], and every Indeterminate visit is one item point from another label [R: Part C2].
- Moving into or out of Indeterminate typically takes 3 item points, the same median change as between visits whose label did not change [R: Part C1].
- No validated minimal clinically important difference (MCID) exists for these item sums [R: plan §3.1], so no item-point difference is called clinically meaningful anywhere in this report.

**2. Observed labels change often, and many changes reverse.**

- The observed label process is not first-order Markov [R: `outputs/model/report.txt`].
- About 54 % of entries into Indeterminate and 35 % of direct TD ↔ PIGD changes are reversed at the next observed visit [R: Part C2; V: identical counts in script 02].

**3. Best predictive model for the label.** A three-state hidden Markov model with misclassification predicts the next label with 75.4 % accuracy [73.5, 77.3] in leave-one-patient-out validation [R: `outputs/loo/report.txt`].

- It never predicts Indeterminate.
- Its "latent Indeterminate" state is held at an imposed emission floor, so the parameters of that state are not interpretable [R: 05_FINDINGS Stage 4].

**4. Individual slopes are weakly identified; population trends are clear.**

- Patient-specific ratio slopes have median reliability 0.05 [R: `outputs/validation/report.txt`].
- Population trends are clear: the PIGD subscore rises and tremor flattens over follow-up, gradually [R: 05_FINDINGS Stage 5].

**5. Discrete trajectory groups are not established**, by any of three independent approaches (Stage 6, Stage 6b, Part A). This is not evidence that all patients progress alike.

**6. Visit-level noise in the ratio is very heavy-tailed** (degrees of freedom about 2.1), and modelling it improves held-out prediction in 7/7 splits [R: four-model run]. Extreme heavy tails in *patient slopes* are **not supported** by the pre-specified reassessment [R: Q2 reassessment]. Moderate departures from Gaussian slopes are unresolved.

**7. The Indeterminate audit is inconclusive** under its pre-specified, amended rule.

- Entries into Indeterminate revert more often than direct changes: +0.194 [0.108, 0.281].
- Whether Indeterminate visits cluster within patients is not established: observed/expected 1.18 [0.99, 1.39] [R: Part C2].
- Indeterminate is therefore shown to be neither measurement noise nor a distinct state.

---

## 1. Data and phenotype construction

### Definition [V: `src/ppmi_subtype/subtype.py`, `config/config.yaml`]

- **Tremor score:** mean of 11 items (1 patient-reported Part II item, 10 examiner-rated Part III items).
- **PIGD score:** mean of 5 items (2 Part II, 3 Part III).
- **Ratio:** tremor / PIGD.
- **Labels:** TD if ratio ≥ 1.15; PIGD if ratio ≤ 0.90; Indeterminate in between.
- **Special cases:** PIGD = 0 with tremor > 0 → TD; both zero → Indeterminate. The ratio is undefined in both cases, but the label is defined.
- **Missing values:** a missing, "unable to rate" (code 101) or out-of-range item makes its subscore, and therefore the label, missing.
- **Exam selection:** the Part III exam is the OFF-medication or untreated one. The Part II items carry no medication state.

### Reconstruction and coverage

- The item-level rebuild reproduces the frozen panel exactly at all 5,774 visits [R: Part C1 §1]: 1,080 sporadic-PD patients, 4,441 labelled visits, 934 patients with ≥ 2 labelled visits.
- The curated dataset's `td_pigd` variable is binary and merges Indeterminate with PIGD, so a three-class analysis requires reconstruction [R: plan §2].

**Why 1,333 visits are unlabelled** [R: Part C1 §2]:

| reason | visits |
|---|---|
| no usable OFF/untreated Part III exam | 686 |
| Part III row present but items absent | 559 |
| an item coded 101 | 52 (mostly postural stability) |
| no item rows | 36 |

**Selection:** patients with only one labelled visit are older (65.2 vs 63.0 years) and start as PIGD more often (26 % vs 17 %) [R: Part C1 §2].

### How much the label depends on construction choices

| check | result | caveat |
|---|---|---|
| Part III-only score construction | agrees with the primary label at 83.2 % of visits [R: Part C1 §4] | thresholds were never validated for that construction |
| OFF vs ON exam on the same day | same label at 75.2 % of 2,343 paired visits (72.1 % for Part III-only) [R: Part C1 §5] | **not test–retest**: medication response, time of day, exam order and rater differences are all mixed in |

---

## 2. Observed transitions

**Transition matrix** (3,361 consecutive labelled pairs) [R: `outputs/transitions/report.txt`]:

- TD persists 80.8 % of the time, PIGD 70.2 %, Indeterminate 19.2 %.
- TD → PIGD rises from 10.0 % for intervals ≤ 1.5 years to 28.9 % beyond 4 years. A single pooled matrix therefore mixes interval lengths.

**Reversion.** Of all observed changes, 35.2 % return to the origin label at the next visit [R: script 02]. Part C2 re-estimated the key contrast with patient-level bootstrap intervals:

| | reversion rate | events with an observed next visit | events with no later visit (excluded) |
|---|---|---|---|
| entries into Indeterminate | 0.543 [0.471, 0.615] | 188 | 80 |
| direct TD ↔ PIGD changes | 0.349 [0.298, 0.397] | 284 | 110 |

The counts reconcile exactly with script 02: 75/134 + 27/54 = 102/188, and 55/172 + 44/112 = 99/284 [V].

**Persistence.** Only 3 of 135 entries into Indeterminate are followed by Indeterminate at both of the next two visits [R: Part C2 §2].

**Interpretation boundary.** These are changes in a *recorded label* produced by a fixed rule. They are not model predictions, inferred latent states or biological subtype conversions.

---

## 3. Markov, CTMC and HMM models

| question | result | source |
|---|---|---|
| Is the label process Markov? | rejected for each current state (χ² p ≤ 0.004); past labels carry information | [R: `outputs/model/report.txt` §1] |
| Which model predicts best? | the HMM with misclassification beats the discrete Markov chain in 10/10 patient splits (+0.028 log-likelihood per transition). The interval-aware CTMC does not beat the discrete chain at any interval | [R: same §3–5] |
| Leave-one-patient-out accuracy | 0.754 [0.735, 0.773] (patient bootstrap); log-likelihood −0.673 per prediction | [R: `outputs/loo/report.txt` §4] |

The HMM's accuracy comes almost entirely from persistence:

- TD → TD is predicted correctly 99 % of the time, TD → PIGD only 8 %.
- Indeterminate is never predicted (recall 0).
- Its predicted probability carries weak information (one-vs-rest AUC about 0.60–0.63) [R: 05_FINDINGS Stage 4].

**Identifiability.**
- The 3-state model with emission floor 0.5 has a unique optimum. Without the floor it lies on a ridge, and 4- and 5-state models add no held-out benefit [R: 05_FINDINGS Stage 4].
- **The latent Indeterminate state's emission row sits exactly on the floor (0.25 / 0.50 / 0.25) in every fold** [R]. The constraint is active, so the data would make that state's own label even less informative.

**New-patient model.** For a patient seen once, baseline ratio plus covariates beat the history-free HMM by +0.032 log-likelihood per prediction [R: `outputs/newpatient/report.txt`].

### 3.1 Three Indeterminate "durations" estimate different things [C]

| estimate | value | what it estimates | source |
|---|---|---|---|
| CTMC mean sojourn | **0.44 y** | expected time until the *recorded label* changes, assuming the observed label is itself a continuous-time Markov state with exponential holding times | [R: `outputs/model/report.txt` §2] |
| HMM latent mean sojourn | **2.12 y** | expected holding time of a *latent* state in a model whose observations are misclassified, with the latent Indeterminate emission fixed at the 0.5 floor | [R: same] |
| Kaplan–Meier median time to leave | **1.08 y** | median time from a patient's *first* Indeterminate visit to the first later visit with a different label, censored at the last labelled visit; non-parametric, observed-label scale | [R: Part C2 §2] |

These are **not three measurements of one duration**:

1. **CTMC (0.44 y)** treats every visit's label as exact. Each one-visit Indeterminate excursion counts as a genuine, very short stay. The Markov assumption it relies on is rejected (§3).
2. **Kaplan–Meier (1.08 y)** is on the same observed-label scale but makes no Markov or exponential assumption.
   - It can be no shorter than the visit spacing (median interval 1.0 year): a label that changes at the next visit registers about 1 year, however brief the change.
   - It uses only first entries, and censoring and informative dropout affect it.
3. **HMM (2.12 y)** belongs to a latent state. Its emission row is pinned at the floor, so the model assigns half of that state's observations to TD or PIGD labels. The latent sojourn therefore depends on the floor and should not be read as how long patients are "truly" Indeterminate [R: 05_FINDINGS Stage 4 §2, §6].

**What can be said:** on the recorded-label scale, Indeterminate labels are short-lived, typically gone by the next annual visit. Whether there is a longer-lasting latent intermediate condition cannot be determined from these models.

---

## 4. Bayesian ratio growth model and slope clustering

**Growth model** (hierarchical Gaussian, 821 patients with ≥ 2 ratio-defined visits; PIGD = 0 visits are excluded because the ratio is undefined):

- population slope −0.090 per year [−0.109, −0.071];
- between-patient slope SD 0.102 [R: Part A manifest; `outputs/hbayes/report.txt`].

**Reliability.** Per-patient slope reliability has median 0.05 and exceeds 0.5 for only 8.2 % of patients; baseline reliability is about 0.80 [R: `outputs/validation/report.txt` Stage 4b]. Most individual slopes are shrunk almost entirely to the population mean.

**Slope clustering (Part A).**

- BIC on posterior-mean slopes selects K = 3 (ΔBIC 316).
- But K = 1 is selected in 500/500 joint posterior draws.
- A hierarchical model with *no* groups produces K ≥ 2 in 200/200 simulated cohorts (median ΔBIC 227).
- Silverman's unimodality test does not reject a single peak (p = 0.24).
- The K = 3 components differ in spread, not centre, and track visit count [R: `run_manifests/20261015T171236Z_bayesian_slope_clustering/REPORT.md`].

**Subscore clustering (Stages 6/6b).**
- Clustering was not supported, and assignments were no more stable than under a no-groups null.
- Stage 6b could not build an adequately calibrated null and closed as inconclusive [R: 05_FINDINGS].

**Conclusion.** No reproducible discrete groups. This is a failure to establish groups, not evidence that every patient follows the same trajectory: heterogeneity exists, and its non-Gaussian shape is partly a measurement effect [R: 05_FINDINGS Stage 6/6b].

---

## 5. Heavy-tailed residuals vs heavy-tailed patient slopes

**Design.** Four configurations of the ratio model: Gaussian or Student-t slopes × Gaussian or Student-t residuals, fitted on 7 patient-level splits and scored on identical held-out transitions [R: `run_manifests/20261015T190228Z_student_t_four_model_full/REPORT.md`].

| question | result | verdict |
|---|---|---|
| Visit-level residual tails | t-residual models beat Gaussian-residual models on held-out predictive density: +0.112 [0.061, 0.168] (GT − GG) and +0.106 [0.058, 0.158] (TT − TG), positive in 7/7 splits. Residual degrees of freedom 2.10 [1.88, 2.39] | **supported** |
| Prediction (script 04 class metric) | GT +0.038 [0.020, 0.055], TT +0.041 [0.023, 0.059] vs Gaussian, 7/7 splits | **improved, almost entirely through the residuals** |
| Patient-slope tails, original run | sampler failed the mandatory convergence check (R̂ 1.018, ESS 248) | **inconclusive (convergence)**, kept on record |
| Patient-slope tails, reassessment | collapsed slice sampler, correct Gamma(2, 0.1) and log-uniform priors, 60 calibration replicates per scenario. Convergence passed; full-cohort slope degrees of freedom 2.09 [1.48, 4.11]; calibration 0/60 false positives per scenario (exact 95 % CI [0, 5.96 %]); per-split replication **1/7** | **not supported** [R: `run_manifests/20261015T203325Z_q2_reassessment/REPORT.md`] |

**What is unresolved.** Moderate departures from Gaussian slopes (degrees of freedom about 3–6) cannot be distinguished from Gaussian on this visit design. The upper tail of the slope degrees of freedom is prior-sensitive in 70 % subsets [R].

---

## 6. Part C: raw scores and the Indeterminate audit

**Scope.** Descriptive score audit (C1) and a pre-specified Indeterminate audit (C2) [R: `experiments/20261015T221426Z_partC_raw_scores/REPORT.md`, `experiments/20261015T221426Z_partC_indeterminate/REPORT.md`; plan `10_PART_C_RESEARCH_PLAN.md` with Amendment 1].

**Score descriptives:**
- PIGD sum rises from 1.34 [1.27, 1.41] at year 0 to 3.52 [2.95, 4.15] at year 8.
- PIGD zero share falls from 0.24 to 0.05 over the same period.
- Attrition is informative, so later years over-represent patients who remain in follow-up.
- PIGD = 0 is sticky but not absorbing: it is followed by 0 again with probability 0.485 within 1.5 years [R: Part C1 §3].

### C2 primary result (934 patients; 4,000-resample patient-level bootstrap)

**Reversion contrast:** +0.194 [0.108, 0.281]. Its direction is established.

**Temporal clustering of Indeterminate:**

| quantity | value |
|---|---|
| observed adjacent Indeterminate pairs | 64 |
| exact expectation under random reordering of each patient's own labels | 54.26 |
| contributing patients | 94 |
| observed / expected | 1.179 [0.988, 1.391] |

The interval spans 1, so no direction is established.

**Verdict under the amended rule: inconclusive.** The rule requires *both* contrasts to establish a direction, and an interval spanning 0 or 1 is never read as equivalence. The original rule, superseded before the full run, would have said "transient"; that is recorded but not applied.

### Sensitivities

- The reversion contrast stays positive under every analysis except Part III-only scores: +0.073 [−0.035, 0.179].
- Clustering spans 1 everywhere except Part III-only: 1.436 [1.203, 1.753]. Under that construction clustering is clear but the reversion contrast is not.
- In the genetic-PD cohort, clustering is not evaluable (expected count 18.2).

**Conclusion.** The Indeterminate result depends on how the score is constructed.

---

## 7. Evidence matrix

The full matrix, with limitations and every path, is in `docs/cluster_phase/11_CROSS_PHASE_REVIEW.md` §2. This is the condensed version.

| question | analysis | main result (uncertainty) | supports | does not establish |
|---|---|---|---|---|
| Is the reconstruction faithful? | item rebuild | 0 mismatches / 5,774 visits | Part C scores = panel | — |
| Does the label depend on construction? | Part III-only; OFF/ON | 83.2 %; 75.2 % agreement | strong construction and state dependence | test–retest reliability |
| Do labels have memory? | Markov test | p ≤ 0.004 | latent or misclassification framing | which latent structure |
| Best label predictor | HMM, leave-one-patient-out | 0.754 [0.735, 0.773] | predictive use | biological states; Indeterminate prediction |
| Transience of Indeterminate entries | C2 reversion contrast | +0.194 [0.108, 0.281] | entries revert more than direct changes | noise vs state |
| Within-patient clustering of Indeterminate | C2 observed/expected | 1.18 [0.99, 1.39] | — | presence or absence |
| Individual slopes | posterior reliability | median 0.05 | strong shrinkage | — |
| Discrete slope groups | Part A, Stage 6/6b | K = 1 in 500/500 draws; null K ≥ 2 in 200/200 | not established | homogeneity |
| Residual tails | four-model comparison | +0.112 [0.061, 0.168]; 7/7 splits | heavy-tailed noise | its cause |
| Extreme slope tails | Q2 reassessment | 1/7 splits | not supported | Gaussian slopes |

---

## 8. Limitations (all phases)

- **Target.** All results are PPMI-only, and the outcome is the project's own label rule, not an independent diagnosis.
- **Selection.** Labels require a usable OFF/untreated exam and no item coded 101. Ratio models also exclude PIGD = 0 visits (725 labelled visits; 382 patients affected).
- **Time axis.** Dropout is informative (higher PIGD → loss of follow-up). Treatment initiation coincides with early follow-up. Disease stage cannot be separated from time in study [R: 05_FINDINGS Stage 5].
- **Medication state.** Part II items carry no medication state.
- **Measurement.** Every Gaussian or beta-binomial score model mis-fits PIGD = 0 [R: script 31; Stage 6b].
- **Validation design.** Patient splits overlap across the seven script-04 seeds, so "k of 7" summaries are stability heuristics, not independent replication [V].

---

## 9. Current conclusions

1. The recorded TD/Indeterminate/PIGD label is a coarse, construction-dependent and medication-sensitive read-out of two small item sums.
2. Label changes, especially into and out of Indeterminate, are frequently transient on the observed scale.
3. The best label predictions come from a misclassification HMM, but its latent states, especially Indeterminate, are not interpretable as biological states.
4. Patients differ in progression, but the data identify individual slopes poorly and do not support discrete trajectory groups.
5. Visit-level noise is heavy-tailed; patient-slope heavy tails are not supported.
6. Whether Indeterminate reflects a sustained intermediate pattern is **unresolved**.

---

## Appendix A — Superseded, refined and inconclusive claims

**Earlier claims (from 05_FINDINGS):**

| claim (where) | status now | evidence |
|---|---|---|
| "Indeterminate is not a state so much as a crossing" (`outputs/transitions/report.txt` §1) | **Refined.** Its transience on the observed scale is supported (reversion contrast), but the pre-specified test of within-patient clustering is inconclusive. Do not read it as "not a state" | Part C2 |
| 35.2 % of changes reverse (script 02) | **Refined.** Correct as a pair-level figure; patient-bootstrap intervals now exist for the key contrast | Part C2 §1 |
| HMM latent Indeterminate sojourn (2.12 y) as a duration | **Not interpretable** as a duration (emission floor binds) | §3.1 |
| Flexible kernels do not beat simple ones (GPR report §E.9, §I.1 #9) | **Superseded:** true for K1 only | 05_FINDINGS Stage 3 |
| GP loses to persistence on PIGD (§I.1 #11) | **Refined:** on MAE only | 05_FINDINGS Stage 3 |
| 5-state HMM converged 404/934 (midterm) | **Superseded:** budget-limited, multimodal | 05_FINDINGS Stage 4 |
| Rate changes "after about five years" (§F, §I.1 #12) | **Superseded:** gradual curvature, no identifiable knot | 05_FINDINGS Stage 5 |
| Late-period Wald intervals (`period_slopes.csv`) | **Superseded:** 1.7–2.2× too narrow | 05_FINDINGS Stage 5 |
| Real cluster signal 513 vs null 88 (§G.9) | **Superseded:** null 95th percentile about 117; clustering inconclusive | 05_FINDINGS Stage 6/6b |

**Claims from the latest phases:**

| claim (where) | status now | evidence |
|---|---|---|
| Part A K = 3 slope groups (BIC) | **Not supported** as groups | Part A manifest |
| Original Q2: inconclusive (convergence) | **Kept on record**; the reassessment's verdict is "not supported" | specs 08, 09 |
| ν prior "Gamma(2, 0.1)" (specs 07/08) | **Erratum:** it was a truncated Exponential(0.1) | dated errata; `_frozen/` copies |
| Part C2 original rule (plan §5.5) | **Superseded before the full run** by Amendment 1 | plan Amendment 1 |
| Splitting Indeterminate into near-TD / near-PIGD (script 08) | **Unchanged:** not supported | `outputs/patient_level/report.txt` |
| Joint subscore model superiority | **Unchanged:** not supported | 05_FINDINGS Stage 3 |
| Stage 6b no-groups null | **Inconclusive** (closed) | 05_FINDINGS Stage 6b |

---

## Appendix B — Review of the proposed posterior-predictive diagnostic (D1)

D1 was proposed in `11_CROSS_PHASE_REVIEW.md` §6. **It has not been run.** The review below found four design problems, and the protocol is revised accordingly (B.9).

### B.1 Question and null hypothesis

**Scientific question.** Is a model with a continuous patient trajectory, heavy-tailed visit noise and **no latent intermediate state** sufficient to reproduce the observed behaviour of Indeterminate labels?

**Formal null H0.** On ratio-defined visits, the observed label-sequence statistics (B.5) are draws from the posterior predictive distribution of the fitted TT ratio model (Student-t slopes, Student-t residuals), with labels produced from simulated ratios by the unchanged thresholds.

- **Rejection** means the continuous model misses a feature of the observed label sequences.
- **Non-rejection** means the model is *sufficient* for these statistics. It does **not** show that no intermediate state exists; a weak or rare state could be missed (hence the positive control in B.6).

### B.2 Is this question still open? Yes, and it is not answered by existing checks [V]

| existing check | what it examined | why it does not answer D1 |
|---|---|---|
| Stage 6b and script 31 predictive checks | subscore measurement models: zero shares, BLUP shapes | not label sequences |
| Part A nulls | the slope-clustering statistic | not label sequences |
| Ratio-model evaluation (scripts 04, 36) | one-step-ahead class log-likelihood only | no sequence features: no reversion, no adjacency |
| HMM | label sequences | contains an Indeterminate state by construction, so it cannot test whether one is needed |

**Not comparable:** the script 04 head-to-head (HMM −0.656 vs Gaussian growth −0.681 per transition, 825 transitions) and the four-model run (TT +0.041 vs Gaussian on 854 transitions) used different transition sets. TT has not been compared with the HMM on identical transitions [V].

### B.3 How a held-out patient's trajectory is estimated (issue found)

**As proposed:**
- Population parameters θ come from training patients only.
- Thresholds are the fixed 1.15 / 0.90.
- Each test patient's random effects b_i are drawn from p(b_i | y_i, θ), i.e. **conditioned on that patient's own observations**.

No held-out observation enters θ or the thresholds. **But** y_i is used twice: once to place the patient's simulated trajectory, and again as the data the replicates are compared with. Posterior-predictive p-values computed this way are *not* uniform under H0; they lean towards "consistent" (Bayarri & Berger 2000; Hjort et al. 2006).

**Revision:** the decision thresholds are taken from a **simulation-based calibration** of the p-values (B.6), not from nominal 0.025 / 0.975. A secondary "mixed" replicate, drawing b_i fresh from p(b | θ) rather than from the patient's own data, is reported beside the primary. It is less conservative but changes each patient's label mix.

### B.4 How posterior uncertainty is propagated

Each replicate r uses a different draw from the training posterior:

1. θ^(r) = (β, τ0, τ1, νs, σ, νe) from the training posterior;
2. b_i^(r) and the mixing weights (λ_i, ω_it) from their conditional given y_i and θ^(r), by the existing inner-Gibbs routine in `hbayes_tt.predictive_moments`, extended to all of the patient's visits;
3. new visit noise ε_it^(r) ~ t_νe^(r)(0, σ^(r)²) at the patient's real ratio-defined visit times;
4. simulated ratio → label.

Parameter, random-effect and noise uncertainty are all propagated. 500 replicates.

### B.5 Statistics, all computed with `indeterminate_audit`

Statistics are computed identically on observed and replicated sequences, on ratio-defined visits only:

| statistic | definition | role |
|---|---|---|
| S1 | Indeterminate visit share | decision |
| S2 | R_E (entry reversion) | decision |
| S3 | R_D (direct-change reversion) | decision |
| S4 | Δ_R = R_E − R_D | decision |
| S5 | observed/expected adjacency of Indeterminate | decision |
| S6 | two-step Indeterminate persistence | descriptive: small counts |

For each: posterior-predictive p_S = P(S_rep ≥ S_obs), with ties counted as ½.

### B.6 Calibration and power (added)

**Calibration (null).** 50 synthetic cohorts are generated from the TT model fitted to the full cohort, on the real ratio-defined visit design. The **entire** D1 pipeline is run on each (fold fits, replicates, p-values), giving the null distribution of each p_S. Calibrated thresholds are the 2.5 % and 97.5 % quantiles of that null distribution.

**Positive control (power).** 50 synthetic cohorts with a planted sticky intermediate state:

- a random 30 % of patients enter an episode during follow-up, starting at a uniformly chosen visit, in which the latent ratio is held at 1.02 for an exponentially distributed time with mean 2 years, plus the same TT noise;
- the other parameters are as in the null.

The planted values are illustrative, not clinical. They define a detectable alternative and are fixed now.

**Power is reported.** If fewer than 50 % of positive-control cohorts are flagged on S5 or S2, a non-rejection on the real data will be reported as **"uninformative (low power)"**, not as "reproduced".

### B.7 Validation design and dependence (issue found)

Script 04's seven 70/30 splits **overlap**. Each test set is a random 30 %, so any two test sets share about 30 % of their patients, and "≥ 5/7" is a stability heuristic, not independent replication [V].

**Revision:** a **disjoint 5-fold patient partition**, stratified by first label, seed 42.

- Every patient is tested exactly once, with θ fitted on the other four folds.
- Replicates are pooled across folds into whole-cohort replicate datasets (replicate r of every fold forms cohort replicate r), so each statistic has a single observed value and a single reference distribution on the full ratio-defined cohort.
- Fold-level p-values are reported only descriptively.

### B.8 What the ratio-defined cohort excludes [V], and what therefore cannot be concluded

**Excluded:**
- 725 of 4,441 labelled visits (665 TD with PIGD = 0; 60 both-zero Indeterminate visits, 13.6 % of all Indeterminate visits).
- Eligible patients fall from 934 to 821.
- 382 patients have at least one PIGD = 0 visit removed, so their sequences gain gaps, and adjacency is redefined over the remaining visits.

**Cannot be concluded:**
- anything about both-zero Indeterminate visits or about TD-by-rule (PIGD = 0) visits;
- whether the PIGD = 0 process itself is state-like (the zero-persistence misfit, Stage 6b);
- direct numerical comparison with the Part C2 primary statistics, which include those visits. D1's observed statistics are recomputed on the subset and compared only with replicates on the same subset.

### B.9 Revised D1 protocol (for approval; not run)

| item | specification |
|---|---|
| Question | B.1 |
| Population | ratio-defined visits of the frozen sporadic-PD cohort; patients with ≥ 2 such visits (821) |
| Models | TT (primary) and GG (reference): existing `hbayes_tt_nu` collapsed slice sampler and `hbayes`, frozen settings (4,000 draws, burn-in 2,000, thin 5), correct Gamma(2, 0.1) ν prior; no new likelihood |
| Validation | disjoint 5-fold patient partition, stratified, seed 42 |
| Replicates | 500 per fold; propagation as in B.4; primary replicate conditional on the patient's own data; secondary "mixed" replicate |
| Statistics | S1–S5 (decision), S6 (descriptive), as in B.5 |
| Calibration | 50 null synthetic cohorts; thresholds = 2.5 % / 97.5 % quantiles of the null p-value distribution |
| Power | 50 positive-control cohorts (B.6); flag rate reported |
| **Decision (TT)** | **"not reproduced"** if any of S2, S4 or S5 lies outside its calibrated central 95 % range; **"reproduced"** if S1–S5 all lie inside **and** power on S2 or S5 is ≥ 50 %; **"uninformative (low power)"** if all lie inside but power < 50 %. GG is reported alongside, with no decision |
| Interpretation | "reproduced" = the continuous model is sufficient for these statistics on ratio-defined visits; it does not exclude an intermediate state. "Not reproduced" names the missed feature; it is not evidence of a biological state |
| Missing data | unlabelled and ratio-undefined visits removed before sequences are formed, identically in observed and simulated data |
| Sensitivities (reported, no decision) | log-uniform ν prior; intervals ≤ 1.5 years; mixed replicate |
| Tests | the simulation reproduces known parameters on synthetic data; observed statistics via the D1 path equal `indeterminate_audit` on the same sequences; fold disjointness; no test patient in any training fit; hashes and seeds recorded before computation |
| Compute (estimate, to be confirmed by a timed smoke run) | about 5 fold fits × 2 models × (1 real + 50 null + 50 positive-control cohorts) ≈ 1,000 fits of about 20–45 s, plus replicate simulation: roughly 10 CPU-hours, about 20–40 min on one exclusive 32-core node |
| No MCID | used anywhere; statistics are on the label scale |

**RobustLPA:** deferred. No implementation or reference exists in the repository. The exact method, reference or paper from the professor is needed before feasibility can be evaluated, and the evaluation would have to address the calibrated-null problem that closed Stage 6b.

---

## Appendix C — Repository state and source paths

**Committed:**

| commit | content |
|---|---|
| `7a59dba` | Part A slope clustering |
| `3a860ca` | Student-t study and Q2 reassessment |

**Uncommitted, left untouched by this report:**
- `docs/cluster_phase/10_PART_C_RESEARCH_PLAN.md` (with Amendment 1) and `11_CROSS_PHASE_REVIEW.md`
- `src/ppmi_subtype/raw_scores.py`, `src/ppmi_subtype/indeterminate_audit.py`
- `scripts/39_raw_score_audit.py`, `scripts/40_indeterminate_audit.py`
- `tests/test_raw_scores.py`, `tests/test_indeterminate_audit.py`

**Source reports:**

| phase | path |
|---|---|
| transitions | `outputs/transitions/report.txt` |
| patient-level views (script 08) | `outputs/patient_level/report.txt` |
| HMM/CTMC | `outputs/model/report.txt` |
| leave-one-patient-out validation | `outputs/loo/report.txt`, `outputs/loo/loo_overall_metrics.json` |
| new-patient model | `outputs/newpatient/report.txt` |
| longitudinal validation | `outputs/validation/report.txt` |
| ratio growth model | `outputs/hbayes/report.txt` |
| cluster phase | `docs/cluster_phase/05_FINDINGS.md`, `04_RESULTS_GENERATED.md` |
| GPR report and audits | `docs/GPR_onwards_explanatory_report.md`, `docs/gpr_audit/` |
| Part A, Student-t, sampler pilot, Q2 reassessment | `docs/cluster_phase/run_manifests/20261015T171236Z_bayesian_slope_clustering/`, `…T182129Z…`, `…T184027Z…`, `…T190228Z_student_t_four_model_full/`, `…T200100Z_nu_slope_sampler_pilot/`, `…T203325Z_q2_reassessment/` |
| Part C | `experiments/20261015T221426Z_partC_raw_scores/REPORT.md`, `experiments/20261015T221426Z_partC_indeterminate/REPORT.md` |

**Specifications:**
- `docs/cluster_phase/06_*`, `07_STUDENT_T_SLOPE_SPEC.md`, `08_STUDENT_T_RESIDUAL_SPEC.md`, `09_Q2_REASSESSMENT_SPEC.md`, `10_PART_C_RESEARCH_PLAN.md`
- frozen copies in `docs/cluster_phase/_frozen/`

**Code:** `src/ppmi_subtype/{subtype,panel,transitions,hmm,ctmc,hbayes,hbayes_t,hbayes_tt,hbayes_tt_nu,slope_clustering,raw_scores,indeterminate_audit}.py`

---

## Appendix B.10 — D1 protocol, Revision 2 (2026-10-16): proposed for approval, not run

**Record.**
- **Revision 1:** B.1–B.9 above, kept unchanged as the record.
- **Revision 2:** this section. It **supersedes B.6 (calibration), B.7 (validation) and B.9 (decision rule and budget)** wherever they differ. B.1 (question), B.2 (novelty), B.4 (propagation), B.5 (statistics) and B.8 (exclusions) stay in force, with the clarifications below.
- **Nothing has been implemented or run.**

### R2.1 Scope and meaning (unchanged in substance, stated precisely)

**H0.** On ratio-defined visits of the frozen sporadic-PD cohort, the five decision statistics S1–S5 (B.5) behave as draws from the patient-conditional posterior predictive distribution of the fitted TT ratio model, which has no latent intermediate state. Labels are produced from simulated ratios by the fixed thresholds (1.15 / 0.90).

**Outside the scope of D1.** It excludes the 725 PIGD = 0 visits (665 TD by rule, 60 both-zero Indeterminate), so 821 of the 934 eligible patients are analysed and 382 have sequence gaps (B.8). **D1 cannot determine whether the full three-state phenotype process requires an intermediate state.**

**A "reproduced" verdict means only** that the specified model reproduces the pre-specified statistics adequately within this scope. It is **not** evidence that no intermediate state exists. "Not reproduced" identifies a misfit; it is **not** evidence of a biological state.

### R2.2 Patient-level partition

- 5 disjoint folds of the 821 eligible patients, stratified by first ratio-defined label, seed 42.
- **Every visit of a patient belongs to that patient's fold.** The partition is computed once, stored with its hash, and reused for every observed, null and positive-control cohort. The synthetic cohorts reuse the real patients' visit designs, so fold membership is identical.
- **Primary result:** pooled over folds; replicate r of every fold forms cohort-level replicate r.
- **Fold-level results:** each fold's statistics and calibrated discrepancies are reported descriptively. They are **not** treated as independent replications: folds share training patients, and fold-level verdicts are not combined.

### R2.3 Information used at each stage (validity of the comparison)

| stage | uses | does not use |
|---|---|---|
| A. Fit θ = (β, τ0, τ1, νs, σ, νe) | ratio-defined visits of the 4 training folds | any test-fold observation |
| B. Patient state for test patient i: b_i, λ_i, ω_i \| y_i, θ^(r) | all of patient i's own ratio-defined visits y_i, and θ^(r) from A | other test patients; thresholds are fixed constants |
| C. Replicate | fresh residual mixing weights and fresh t noise at patient i's real visit times; the replicate ratio becomes a label by the fixed thresholds | observed residuals (never reused) |
| D. Statistics | the same `indeterminate_audit` code on observed and replicated sequences | — |

**Why the comparison is valid despite conservativeness.**
- Because y_i informs stage B, each replicate is centred on the patient's own data, so nominal posterior-predictive p-values are pulled towards 0.5 under H0 (conservative).
- D1 therefore does **not** use nominal p-values for decisions. The decision threshold is the empirical null distribution of the same discrepancy, produced by running **the identical pipeline (stages A–D, same folds, same R, same seeds scheme)** on cohorts simulated from H0 (R2.4).
- Whatever bias the double use of y_i introduces, and the Monte-Carlo noise in each p (R = 500 replicates), is present equally in the null cohorts and in the observed cohort. The size of the test is controlled at the calibrated level, up to calibration Monte-Carlo error (R2.5) and up to how well the null-generating parameters approximate the truth: the null is simulated at posterior draws from the full-cohort fit, a parametric bootstrap, which is valid for a composite null only approximately. This is stated as a limitation.
- Conservativeness costs **power**, not size. Power is measured directly (R2.6).

### R2.4 Discrepancy statistic and calibration (frozen)

**Per-statistic discrepancy.** For each S in {S1, …, S5}:

- p̃_S = (#{S_rep > S_obs} + ½ · #{S_rep = S_obs} + ½) / (R + 1), with R = 500;
- z_S = Φ⁻¹(p̃_S), where Φ⁻¹ is the inverse standard normal CDF;
- |z_S| is large when the observed value lies in either tail of its replicate distribution.

**Single decision statistic:** **T = max_{S ∈ {S1..S5}} |z_S|.** Taking the maximum controls the family-wise false-alarm rate across the five statistics without choosing per-statistic levels.

**Null cohorts.** N₀ synthetic cohorts. For each:

1. draw θ from the full-cohort TT posterior (ratio-defined visits, frozen settings) at a seeded random retained draw;
2. draw fresh b_i, λ_i and ω_it for all 821 patients at their real ratio-defined visit times;
3. form labels;
4. run stages A–D with the R2.2 folds;
5. record T.

Seeds: data `[42, 41, 0, c]`; fit seeds as in the Q2 reassessment scheme.

**Threshold.** **c = the k-th order statistic of the N₀ null T values, with k = ⌈0.95 (N₀ + 1)⌉.** Exceeding c then has nominal size 5 %. c is computed once, written to `d1_threshold.json` together with its SHA-256, and frozen **before** any positive-control or observed-cohort computation.

### R2.5 Calibration budget and Monte-Carlo uncertainty

Given c, the realised false-alarm rate of "T > c" is Beta(N₀ + 1 − k, k)-distributed (exact for a continuous T):

| N₀ | expected size | SD | central 95 % range of the realised size |
|---|---|---|---|
| 50 (Revision 1) | 0.039 | 0.027 | 0.005 – 0.107 |
| 200 | 0.050 | 0.015 | 0.024 – 0.084 |
| 300 | 0.050 | 0.013 | 0.028 – 0.077 |
| **500 (proposed)** | **0.050** | **0.010** | **0.033 – 0.071** |
| 1,000 | 0.050 | 0.007 | 0.037 – 0.064 |

**50 null cohorts are not adequate:** the realised size could be anywhere from about 0.5 % to 11 %. **Proposed: N₀ = 500**, which keeps the realised size within about 3.3–7.1 % with 95 % probability.

The calibrated p-value of the observed cohort, (1 + #{T_null ≥ T_obs}) / (N₀ + 1), is reported with its binomial Monte-Carlo SE (about 0.010 near 0.05 for N₀ = 500).

### R2.6 Positive controls (power checks for constructed scenarios only)

**These are synthetic assumptions, not clinically established values.** The resulting detection rates are power against the stated constructed mechanisms. They are **not** power to detect a biological subtype.

**Common mechanism.** A fraction π of patients, chosen at random, has one episode:

- start time uniform between the first and last visit;
- duration ~ Exponential(mean D);
- during the episode the latent mean ratio is replaced by a level L, with residual scale multiplied by κ;
- everything else is drawn as in the null.

| scenario | π | L | D | κ | role |
|---|---|---|---|---|---|
| **PC1** | 0.30 | 1.02 | 2 y | 1 | **primary power scenario** (Revision 1 values) |
| PC2 | 0.15 | 1.02 | 2 y | 1 | lower prevalence |
| PC3 | 0.30 | 1.02 | 1 y | 1 | shorter episodes |
| PC4 | 0.30 | 1.02 | 2 y | 0.25 | episode with reduced visit-level noise ("stable intermediate" alternative) |

**Expected limitation, computed before any run [V].** Under the fitted residual law (νe ≈ 2.1, scale ≈ 0.58), a latent ratio of 1.02 yields an Indeterminate label with probability about **0.15**, against about 0.11 at a latent ratio of 0.6. With κ = 1, a planted episode therefore changes label behaviour little, and **PC1–PC3 may have low power**. PC4 represents an alternative in which the intermediate condition is also less noisy. Which scenario defines "adequate power" must be fixed before any run (R2.8 decision 2).

**Budget:** N₁ = 100 cohorts per scenario. Each scenario's detection rate (T > c, using the frozen c) is reported with its Clopper–Pearson 95 % interval; for example 60/100 → [0.497, 0.697].

### R2.7 Verdict rules (frozen before any positive-control or observed computation)

**Inputs:** the frozen c (R2.4); the pre-specified power scenario's detection rate and its CP 95 % lower bound P_lo; and T_obs.

| verdict | condition |
|---|---|
| **NOT REPRODUCED** | T_obs > c. The report names the statistic(s) attaining the maximum and gives every per-statistic p̃ descriptively. |
| **REPRODUCED (within scope)** | T_obs ≤ c **and** P_lo ≥ 0.50 in the pre-specified power scenario |
| **UNINFORMATIVE (low power)** | T_obs ≤ c **and** P_lo < 0.50 |

**Order of computation, enforced by the script and recorded in its manifest:**

1. freeze the protocol (hash);
2. null calibration → c, frozen;
3. positive controls with the frozen c;
4. the observed cohort, once.

Steps 3 and 4 cannot alter c, the statistics or the rules. The GG model and the "mixed" replicate (B.3) are reported for every cohort type as descriptive comparisons, with no verdict.

**Not used anywhere:** MCIDs or clinical thresholds. All statistics are on the label scale.

### R2.8 Compute estimate (to be confirmed by a timed smoke run)

Per cohort: 5 TT fold fits (4,000 draws, burn-in 2,000, thin 5, about 15–25 s each on roughly 660 training patients) plus 500 conditional replicates for 821 patients (about 5–15 s). That is roughly **100–150 CPU-s per cohort**.

| item | cohorts |
|---|---|
| null calibration | 500 |
| positive controls | 4 × 100 = 400 |
| observed | 1 |
| **total** | **901** |

That comes to about **25–40 CPU-hours**, or about **1–1.5 hours on one exclusive 32-core node** (2–3 hours allowing a 2× margin). GG descriptive fits add about 30 %. The run would go through SLURM, not the login node.

**Tests required before the smoke run:**
- fold disjointness and whole-patient folds;
- stage-B conditioning uses only the patient's own visits and the training θ;
- on a synthetic cohort the replicate distribution recovers known parameters;
- the observed-statistics path equals `indeterminate_audit` on the same sequences;
- the threshold file is written before steps 3–4 and is read-only afterwards;
- seeds and hashes are recorded.

### R2.9 Decisions needed before freezing Revision 2

1. **Calibration budget:** N₀ = 500 null cohorts (proposed), or 1,000 (realised size range 3.7–6.4 %, about 2× compute).
2. **Primary power scenario:**
   - PC1 (the original approved values; likely low power given R2.6, so "uninformative" is a plausible outcome); or
   - PC4 (low-noise episode).

   Must be fixed before any run.
3. **Positive-control budget:** 100 cohorts per scenario, or fewer scenarios (for example PC1 and PC4 only) to save compute.
4. **The descriptive GG and mixed-replicate arms:** keep them (about +30 % compute), or drop them.

---

## Appendix B.11 — D1 Amendment 3 (2026-10-16): reviewer decisions and pre-launch protocol checks

**Additive amendment.** B.1–B.10 are unchanged and remain the record. Nothing has been implemented or run. The verification computations below were scratch calculations, not the diagnostic.

### B.11.1 Reviewer decisions (2026-10-16), frozen

| item | decision |
|---|---|
| Null calibration cohorts | **N₀ = 500** |
| **Primary power scenario** | **PC4** (fixed before any computation; not changeable after detection results) |
| Positive controls | PC1, PC2, PC3, PC4 at **100 cohorts each**. PC1 is kept as the original, harder-to-detect scenario. PC1–PC3 are secondary stress tests |
| Gaussian-model (GG) and "mixed" (fresh random effects) comparison arms | **dropped** for this phase (R2.7's descriptive arms and R2.9 item 4 are withdrawn) |

**PC4's reduced episode noise is an illustrative synthetic mechanism, not a clinically validated assumption.** This applies to κ = 0.25, as to π = 0.30, L = 1.02 and D = 2 years. Detection rates are power against constructed mechanisms, not power to detect a biological subtype.

### B.11.2 Check 1: calibration correctness [V, scratch simulation]

**(a) Quantile index.**
- With N₀ = 500, c = T₍₄₇₆₎, the 476th smallest null T, because k = ⌈0.95 × 501⌉ = 476.
- Rejection is **strict**: T_obs > c.
- If T_obs and the 500 null T values are exchangeable and continuous, the exact marginal size is (N₀ + 1 − k)/(N₀ + 1) = 25/501 = **0.0499**.
- Simulation with continuous T: **0.0497** (SE 0.0005).

**(b) Mid-p tail probability and ties.**
- For each statistic, p̃ = (#{S_rep > S_obs} + ½ · #{S_rep = S_obs} + ½)/(R + 1), with R = 500.
- This treats the observed value as one extra draw. It lies in [0.5/501, 500.5/501], so p̃ is never 0 or 1.
- z = Φ⁻¹(p̃) is therefore bounded: **|z| ≤ 3.091**.

T = max |z| over S1–S5 is then discrete. Ties between T_obs and c are resolved as non-rejection, so ties can only make the test **conservative**. In a simulation with integer-valued, mutually dependent statistics and R = 500:

| quantity | value |
|---|---|
| realised size | **0.046** (SE 0.0009) |
| P(T_obs = c) | 0.0046 |
| null T at the cap | 1.2 % |

Pre-specified guard: **if c equals the cap of 3.091, the calibration is declared degenerate** and "NOT REPRODUCED" cannot be issued. The smoke run reports the share of null T at the cap.

**(c) Maximum-|z| threshold.**
- The maximum is calibrated as a whole, so dependence among S1–S5 needs no correction: the null T values already contain it.
- The five statistics, their direction (two-sided via |z|) and the order statistic are fixed now.

**(d) What 500 null cohorts can establish, and what remains uncertain.**

| source | what it covers | magnitude / effect |
|---|---|---|
| **Monte-Carlo (calibration)** | given a finite set of null cohorts, the realised size conditional on c | **Beta(25, 476)**: mean 0.0499, SD 0.0097, 95 % range **[0.033, 0.071]** |
| **Monte-Carlo (replicates)** | noise in each p̃ from R = 500 replicates | absorbed: null and observed cohorts use the same R, so this noise is part of the calibrated null distribution |
| **Null-model fitting** | null cohorts are generated at posterior draws from the TT fit to the real cohort (a parametric bootstrap) | the size is guaranteed only if the data arise from TT with parameters near those draws; this is a composite null and is not quantified by the 500 cohorts |
| **Null-model representativeness** | the real data contain features the null does not generate | (i) visit attendance and dropout that depend on scores [R: Stage 5]; (ii) **outcome-dependent removal of PIGD = 0 visits**: simulated cohorts reuse the real ratio-defined visit times but cannot reproduce the selection *mechanism*; (iii) residual skew (real skew about 1.2 vs a symmetric t). Any of these can raise T_obs, so **"NOT REPRODUCED" may reflect them rather than an intermediate state.** The report must say so whenever that verdict occurs. |

### B.11.3 Check 2: primary power scenario frozen

**PC4 is the primary power scenario**, frozen now:

| parameter | value |
|---|---|
| π | 0.30 |
| L | 1.02 |
| D | 2 years |
| κ | 0.25 |

It is recorded in the run manifest and cannot change after detection results are seen.

### B.11.4 Check 3: positive-control implementation, exactly

**Per synthetic cohort**, starting from the null generation (θ drawn from the full-cohort TT posterior, fresh b_i, λ_i, ω_it, real ratio-defined visit times):

1. **Selection:** each of the 821 patients is planted independently with probability π, from a seeded Bernoulli draw.
2. **Episode window:** s_i ~ Uniform(first, last ratio-defined visit time); d_i ~ Exponential(mean D); the window is [s_i, s_i + d_i).
3. **Latent trajectory:** at visits inside the window, the latent mean (β0 + b0i) + (β1 + b1i)t is **replaced by L**. Outside it, the patient's own line applies; there is no permanent shift after the episode.
4. **Observation noise:** inside the window ε_it = κ · σ · e_it with e_it ~ standard t_νe (same mixing construction); outside the window κ = 1.
5. **Visit timing:** unchanged; the real design is reused and no visits are added or removed.
6. **Labels:** the fixed thresholds applied to the simulated ratio.

**Coverage on the real design (821 patients, 3,528 ratio-defined visits)** [V, scratch]:

| D | episodes covering 0 visits | 1 visit | ≥ 2 visits | mean visits per planted patient |
|---|---|---|---|---|
| 2 years | 0.28 | 0.50 | 0.22 | 1.03 |
| 1 year | 0.45 | 0.45 | 0.10 | 0.67 |

Persistence at the level of adjacent labels can only arise in the ≥ 2-visit subset.

**Does PC4 test persistent intermediate behaviour, not merely a variance change?** Indeterminate-label probability at a single visit under the fitted residual law (νe = 2.10, σ = 0.589) [V]:

| condition | P(Indeterminate) |
|---|---|
| PC1 episode (μ = 1.02, κ = 1) | 0.149 |
| **PC4 episode (μ = 1.02, κ = 0.25)** | **0.518** |
| variance change only, typical patient (μ = 2.3, κ = 0.25) | 0.002 (vs 0.025 at κ = 1) |
| variance change only, near-band patient (μ = 1.3, κ = 0.25) | 0.152 (vs 0.129 at κ = 1) |

PC4 therefore plants **repeated Indeterminate labels during an episode**, because both the level and the reduced noise are required. A variance reduction alone leaves Indeterminate rates nearly unchanged near the band and *lowers* them elsewhere.

**Caveat.** T is two-sided and omnibus, so a pure variance change could also be flagged, with a different signature (S1 decreasing rather than S1 and S5 increasing). For every detection the report will give the direction of each z_S.

**Recommended, needing approval:** a **descriptive specificity control, PC4v** (π, D, κ as PC4, but the latent mean left on the patient's own line), 100 cohorts, about 10 % extra compute. A high PC4v detection rate would show that D1 responds to variance as well as to intermediate persistence. It would not change the verdict rules.

### B.11.5 Check 4: verdict rules and the power bound

The three verdicts of R2.7 are retained:

| verdict | condition |
|---|---|
| **NOT REPRODUCED** | T_obs > c (c not degenerate) |
| **REPRODUCED (within scope)** | T_obs ≤ c **and** P_lo(PC4) ≥ 0.50 |
| **UNINFORMATIVE (low power)** | T_obs ≤ c **and** P_lo(PC4) < 0.50 |

**P_lo** is the lower limit of the **two-sided 95 % Clopper–Pearson interval** for the PC4 detection rate x/100: P_lo = Beta⁻¹(0.025; x, 100 − x + 1). Exact values: 59/100 → 0.487, 60 → 0.497, **61 → 0.507**, 62 → 0.518. **P_lo ≥ 0.50 ⇔ x ≥ 61 of 100.**

**Order:**

1. protocol hash;
2. 500 null cohorts → c, written to `d1_threshold.json` (made read-only);
3. PC1–PC4 (and PC4v if approved) with that c → x and P_lo, written to `d1_power.json` (made read-only);
4. the observed cohort, evaluated once.

The verdict function takes c, P_lo and T_obs from those files only.

### B.11.6 Check 5: scope (restated, binding)

- D1 analyses the 3,528 ratio-defined visits of 821 patients.
- It **excludes the 725 PIGD = 0 visits** (665 TD by rule; 60 both-zero Indeterminate), and 382 patients have sequence gaps.
- It **cannot establish whether an intermediate state is needed for the full three-state phenotype process.**
- A pass means only that the specified TT model reproduces S1–S5 within the ratio-defined scope.

### B.11.7 Final protocol summary (for approval)

| item | frozen value |
|---|---|
| Model | TT, `hbayes_tt_nu` collapsed slice sampler; correct Gamma(2, 0.1) ν prior; 4,000 draws, burn-in 2,000, thin 5 |
| Folds | 5 disjoint, whole-patient, stratified by first ratio-defined label, seed 42; one stored partition for all cohorts |
| Replicates | R = 500 per cohort, patient-conditional (B.4 / R2.3) |
| Statistics | S1–S5 (decision), S6 (descriptive) |
| Discrepancy | mid-p, z = Φ⁻¹(p̃), T = max \|z\| |
| Calibration | N₀ = 500; c = T₍₄₇₆₎; strict rejection; degenerate if c = 3.091 |
| Power | PC4 primary, 100 cohorts; P_lo from two-sided 95 % Clopper–Pearson; ≥ 61/100 needed |
| Secondary stress tests | PC1, PC2, PC3 (100 each), plus PC4v if approved |
| Seeds | null `[42, 41, 0, c]`; PC s ∈ {1, 2, 3, 4, 5 = PC4v} `[42, 41, s, c]`; observed `[42, 41, 9, 0]` |
| Hashes | recorded in the run manifest before computation: this report, the code, the panel MD5, the fold file, then `d1_threshold.json` and `d1_power.json` as they are frozen |
| Compute | 1 + 500 + 400 (+ 100) = 901 (1,001) cohorts × about 100–150 CPU-s ≈ **25–42 CPU-hours**; **about 1–1.5 hours on one exclusive 32-core SLURM node** (2–3 hours allowing a 2× margin) |

### B.11.8 Smoke-test plan (before any full run)

1. **Unit tests** for the new module:
   - fold disjointness and whole-patient folds;
   - stage-B conditioning uses only the patient's own visits and the training θ;
   - the replicate generator recovers known parameters on synthetic data;
   - observed statistics equal `indeterminate_audit` on the same sequences;
   - mid-p, z cap and tie handling against hand cases;
   - the order statistic k = 476 for N₀ = 500;
   - Clopper–Pearson P_lo equals the values above;
   - the episode generator: selection rate, window, level replacement, κ applied only inside the window;
   - the order enforcement: steps 3–4 refuse to run without the frozen threshold file.
2. **Timed smoke on a compute node:**
   - 3 null cohorts, 1 PC4 cohort and the observed cohort, with R = 50, short chains, under `experiments/_smoke/`;
   - reports seconds per cohort, the share of p̃ at the bounds, and the share of T at the cap;
   - *no thresholds or verdicts are computed from smoke output*.
3. **Mini-calibration check:**
   - 20 null cohorts at full settings, to confirm the per-cohort time and that the null T values are not degenerate (cap share);
   - these 20 are **not** reused in the 500.

The full run starts only after the smoke results are reported and approved.

---

## Appendix B.12 — D1 Amendment 4 (2026-10-16): PC4v specificity control, final pre-implementation check

**Additive amendment.** B.1–B.11 are unchanged.

**Reviewer decisions (2026-10-16):**
- PC4v is added as a **descriptive specificity control**: 100 cohorts. It is separate from the primary power calculation and from every verdict rule.
- B.11.7 and B.11.8 are approved as the frozen protocol and smoke plan.
- Only implementation, unit tests, the timed smoke test and the 20-cohort mini-calibration are authorised.

### B.12.1 PC4v mechanics, exactly

PC4v is **PC4 without the level replacement**. Every other element is identical, and the two are generated with **common random numbers**:

| element | PC4 | PC4v |
|---|---|---|
| data seed of cohort c | `[42, 41, 4, c]` | **the same** `[42, 41, 4, c]`; B.11.7's provisional `s = 5` stream is replaced by this pairing |
| θ draw, b_i, λ_i, ω_it, base t-noise e_it | from that seed | **identical** draws |
| planted patients (Bernoulli π = 0.30) and windows [s_i, s_i + d_i), d_i ~ Exponential(mean 2 y) | from that seed | **identical** |
| latent mean inside the window | **replaced by L = 1.02** | **unchanged**: (β0 + b0i) + (β1 + b1i)t, the patient's own line |
| observation noise inside the window | ε_it = κ · σ · e_it, κ = 0.25 | **identical** ε_it = κ · σ · e_it, κ = 0.25 |
| observation noise outside the window | σ · e_it | identical |
| visit times | real ratio-defined design | identical |
| labels | fixed thresholds on the simulated ratio | identical rule |
| diagnostic pipeline (folds, fits, R = 500 replicates, S1–S5, T, c) | B.11.7 | **identical**; fit and replicate seeds derived from the cohort seed in the same way |

**Noise scaling.** The base residual is drawn once, e_it = z_it / √ω_it with z ~ N(0, 1) and ω ~ Gamma(νe/2, rate νe/2), so e_it is a standard t_νe draw. The observed ratio is then the latent mean plus σ · e_it outside the window, or plus κ · σ · e_it inside it. κ multiplies the *entire* t draw, i.e. its scale; the shape (νe) is unchanged.

**What the comparison isolates.**
- **PC4v vs null:** the effect of reducing observation noise inside episodes alone.
- **PC4 vs PC4v**, paired cohort by cohort through common random numbers: the effect of the intermediate level, given the same reduced noise and the same windows.
- **PC4 vs null:** both together.

Statistics: S1–S5 and T are computed by the identical code path. For every positive-control cohort the report gives:
- detection (T > c) with its Clopper–Pearson interval;
- the **signed** z_S of each statistic, so the direction can be read: an intermediate level should raise S1 (Indeterminate share) and S5 (clustering), whereas reduced noise alone should lower S1 or leave it unchanged (B.11.4).

PC4v detections never enter P_lo or any verdict.

### B.12.2 Seeds for non-reusable checks

- Smoke cohorts use tag 91: `[42, 41, 91, c]`.
- Mini-calibration cohorts use tag 90: `[42, 41, 90, c]`.

Neither overlaps the frozen null (tag 0) or positive-control (tags 1–4) streams, so none is reused in the full run.

### B.12.3 What the smoke and mini runs may and may not produce

- **May:** timing; the share of p̃ at its bounds and of T at the cap; null-T summaries; and, for PC4 vs PC4v vs null, the label-level statistics of the *generated* cohorts and the signed z.
- **Must not:** compute or freeze c; compute P_lo; evaluate the observed cohort's T or z. The observed cohort runs in the smoke **for timing only**; its statistics are not written.

The run manifest records, before any computation:
- the SHA-256 of this report;
- the SHA-256 of all source files;
- the panel MD5;
- the SHA-256 of the fold file.

---

## Appendix B.13 — D1 smoke and mini-calibration log (2026-10-16)

**Additive.** B.1–B.12 are unchanged. No threshold c was computed or frozen, no P_lo was computed, and no verdict was produced.

**Runs** (all under `experiments/`, participant-level data, not committed):

| run | job | content |
|---|---|---|
| `20261015T225127Z_d1_smoke/` | 13450 | first smoke |
| `20261015T225228Z_d1_smoke_v2/` | 13451 | smoke on the corrected code (below) |
| `20261015T225334Z_d1_mini_calibration/` | 13452 | 20 null cohorts at full settings, tag 90 |

Each run's `run_manifest.json` records this report's hash, the source hashes, the panel MD5 and the fold-file hash, written before computation.

**Deviation (recorded, not hidden).** In the first smoke, the observed cohort's row recorded two partial discrepancy indicators: the number of p̃ at bounds (3 of 5) and whether T reached the cap (yes). This was at R = 50 with 400-draw chains, and was contrary to B.12.3.

- No observed T, z or statistic was written.
- The module was corrected so that `record_stats=False` returns timing only, and a test was added. The corrected smoke (v2) writes timing only for the observed cohort.
- The v1 output is preserved unchanged.
- The indicators were produced under smoke settings and cannot affect c, the positive-control results or the verdict rule, all of which are frozen in B.11/B.12.

**Sign convention (clarification).** Because p̃ = P(S_rep > S_obs), **z < 0 means the observed statistic lies above its replicate distribution**. T uses |z|, so the convention does not affect any decision.

**Findings (synthetic cohorts only):**

- **Timing at full settings** (mini, 20 cohorts):
  - 109 s per cohort (5 fold fits 100 s, replicates 8.5 s, statistics 0.1 s);
  - full-cohort generator fit 23 s.
  - Revised full-run estimate: 1,001 cohorts (500 null + 400 PC + 100 PC4v + 1 observed) × 109 s ≈ **30.3 CPU-hours**, or **about 1 hour** on one exclusive node with 32 workers (about 2 hours with margin).
- **Bounds and cap** (mini, R = 500):
  - 0 of 100 per-statistic p̃ at their bounds;
  - 0 of 20 T at the cap (3.091);
  - no undefined replicate statistics.
- **Null behaviour** (mini):
  - T ranged from 0.71 to 2.17 (median 1.18), well below the cap, so the calibration is not expected to be degenerate.
  - The per-statistic z had SD 0.73–1.06 (nominal 1). This is consistent with the expected conservativeness of the patient-conditional check, and confirms the need for calibration.
  - The maximising statistic varied across S1–S5.
  - These 20 values are **not** a threshold and will not be reused.
- **Direction check, from 50 generation-only paired cohorts** (label statistics of the generated data, no fitting):

  | quantity | S1 | S5 | S6 | R_E |
  |---|---|---|---|---|
  | PC4v − null (noise reduction alone) | +0.0005 | +0.029 | +0.002 | −0.006 |
  | PC4 − PC4v (the intermediate level) | +0.033 | +0.21 | +0.030 | −0.062 |

  The level, not the noise reduction, carries the planted signal. PC4v isolates the noise effect as intended.
- **Direction in the fitted check** (one smoke cohort each, R = 50; illustrative only):
  - PC4: T = 2.33 (at the R = 50 cap), z_S1 = −2.33, z_S5 = −1.89. Observed Indeterminate share and clustering lie above the replicates, as expected.
  - PC4v: T = 1.35, no extreme z.

**Limitations noted:**
- The generator, like the TT model, can produce negative ratios, which are labelled PIGD. Real ratios are ≥ 0.
- Only about 22 % of planted patients have ≥ 2 episode visits (B.11.4), which bounds the power on S5 and S6.
- Patient-conditional replicates partly absorb a planted episode into b_i. That loss of power is what the 100-cohort PC4 run will measure.

---

## Appendix B.14 — D1 Amendment 5 (2026-10-16): full-run execution safeguards (written before the full run)

**Additive.** B.1–B.13 are unchanged. The full run was approved by the reviewer on 2026-10-16.

### B.14.1 Execution order (verified against B.11.5, B.12 and B.13)

| step | content | what freezes it |
|---|---|---|
| 1 | 500 null cohorts (tag 0) | — |
| 2 | threshold c = T₍₄₇₆₎, with provenance | `01_threshold` |
| 3 | PC1–PC4, 100 cohorts each (tags 1–4), scored with the frozen c | `02_power` (PC4 primary; P_lo = two-sided 95 % Clopper–Pearson lower bound; ≥ 61/100 needed) |
| 4 | PC4v, 100 cohorts, paired with PC4, **descriptive** | `03_pc4v` |
| 5 | checksums of every cohort result and frozen file | `04_outputs` |
| 6 | the observed cohort (`[42, 41, 9, 0]`), **exactly once**, last | `05_observed` |

The order matches B.11.5, with PC4v as its own descriptive step and an explicit output-freeze step added. The threshold rule, primary scenario, power criterion and verdict rules are those of B.11.5 and B.12, unchanged.

### B.14.2 Handling of degenerate or extreme observed evaluations

This was fixed before the run; the protocol did not prescribe it explicitly.

| case | handling |
|---|---|
| **c equals the z cap (3.091)** | calibration degenerate (B.11.2): "NOT REPRODUCED" cannot be issued |
| **an observed decision statistic S1–S5 is undefined** | **verdict withheld** ("DEGENERATE (undefined observed statistic)") |
| **T_obs reaches the cap, or observed p̃ values sit at their bounds** | not degenerate: the observed statistic lies beyond every replicate. The frozen rule applies (T_obs > c → NOT REPRODUCED). **The flags are always reported with the verdict**, never silently |
| **a null or positive-control cohort with non-finite T** | the stage fails validation and nothing is frozen; the run stops |

### B.14.3 Safeguards (`src/ppmi_subtype/ppc_runner.py`; tests in `tests/test_ppc_runner.py`)

- **Frozen files.** Each is written once, made read-only, and paired with a lock that carries its SHA-256. Every read verifies the checksum. Rewriting is refused.
- **Downstream stages refuse to run** if an upstream lock is missing or its checksum fails.
- **Protocol drift.** The protocol lock holds this report's hash, every code hash, the panel MD5 and the fold-file hash. A restart with any changed hash is refused.
- **Interruption.** Per-cohort results are written atomically (temporary file, then rename). A restart computes only what is missing. Unreadable or inconsistent files are moved to `quarantine/`, preserved, and recomputed.
- **After step 5,** any change to a cohort file is detected and refused.
- **The observed stage** writes a `started` marker. If an earlier observed evaluation started and did not complete, it is **not** re-run automatically; that needs a reviewer decision.
- **PC4v** is never read by the power or verdict functions.
- **Failures** are written to `run_status.json`, and all outputs are kept.
- **Code-path check.** Before the full run, the complete wiring is exercised on real data in `--codepath-check` mode (19 nulls, 3 cohorts per scenario, R = 20, short chains, synthetic stand-in for the observed cohort), under `experiments/_smoke/`. It never evaluates the real observed cohort.

---

## Appendix B.15 — D1 full-run results (2026-10-16)

**Additive.** B.1–B.14 are unchanged. Everything below is read from the frozen files.

**Run:** `experiments/20261016T070746Z_d1_full/results/`, SLURM job 13467 on node004.

- Started 12:45:18 IST, finished 13:25:22 IST (2,399 s); median 70 s per cohort; exit 0.
- No quarantined files, no restarts.
- Frozen hashes: protocol lock `e0ea01e9…`; report `9d816fe7…` (B.1–B.14); fold file `3603fd87…`; panel MD5 `fb038ddb…`.
- All 7 locks and all 1,000 cohort files were re-verified after the run: no changes.

### 1. Null calibration

| quantity | value |
|---|---|
| null cohorts | 500 |
| median T | 1.380 |
| 90th percentile | 2.291 |
| 99th percentile | 3.091 (the cap) |
| null T at the cap | 14 / 500 (2.8 %) |
| **threshold c** | **T₍₄₇₆₎ = 2.5765** |
| degenerate | **no** |
| Monte-Carlo uncertainty of the realised size (B.11.2) | Beta(25, 476): 95 % range 3.3–7.1 % |

### 2. Positive controls (power checks for constructed scenarios; not power for a biological subtype)

| scenario | detected / 100 | rate, 95 % Clopper–Pearson | mean signed z: S1 / S5 | role |
|---|---|---|---|---|
| PC1 (π 0.30, L 1.02, D 2 y, κ 1) | 6 | 0.06 [0.022, 0.126] | −1.07 / 0.03 | secondary |
| PC2 (π 0.15) | 7 | 0.07 [0.029, 0.139] | −0.57 / −0.17 | secondary |
| PC3 (D 1 y) | 5 | 0.05 [0.016, 0.113] | −0.70 / −0.13 | secondary |
| **PC4 (κ 0.25)** | **100** | **1.00 [0.964, 1.000]** | −3.09 / −1.22 | **primary: P_lo = 0.964 ≥ 0.50** |

PC1–PC3 are detected at about the false-alarm rate: **no useful power** against intermediate episodes that carry the full fitted visit noise, as anticipated in B.11.4.

### 3. PC4v (descriptive specificity control; not used in any verdict)

| quantity | value |
|---|---|
| detected | 3 / 100: 0.03 [0.006, 0.085], at or below the nominal false-alarm rate |
| mean signed z, S1 / S5 | +0.27 / +0.07 |

Reducing visit-level noise alone is **not** detected. PC4's detection is therefore driven by the planted intermediate level, not by the variance change.

### 4. Observed cohort (evaluated once): primary verdict

Ratio-defined visits only: 821 patients, 3,528 visits.

| statistic | observed | replicate mean | p̃ | z |
|---|---|---|---|---|
| S1 Indeterminate share | **0.1029** | 0.0648 | 0.0010 (bound) | **−3.09** (observed above every replicate) |
| S2 entry reversion R_E | 0.580 | 0.556 | 0.310 | −0.49 |
| S3 direct reversion R_D | **0.311** | 0.436 | 0.9990 (bound) | **+3.09** (observed below every replicate) |
| S4 Δ_R | **0.268** | 0.120 | 0.0010 (bound) | **−3.09** |
| S5 O/E Indeterminate clustering | 1.033 | 1.077 | 0.582 | +0.21 |
| S6 two-step persistence (descriptive) | 0.000 | 0.014 | — | — |

**T_obs = 3.091 > c = 2.5765 → primary verdict: NOT REPRODUCED.**

**Flags (B.14.2):**
- T_obs is at the cap;
- 3 of 5 observed p̃ values are at their bounds;
- no undefined statistic;
- the calibration is not degenerate.

At the cap, the observed S1, S3 and S4 lie beyond all 500 replicates of the patient-conditional check.

**Fold level** (descriptive; folds are not independent replications): T is at the cap in all 5 folds. S1 is the most extreme statistic in every fold; S3 has z ≥ 2.2 in 4 of 5 folds.

### 5. Interpretation, within scope

**What the verdict means.** The continuous TT ratio model, with no intermediate state, does **not** reproduce the label behaviour of ratio-defined visits. It produces too few Indeterminate labels (6.5 % vs 10.3 % observed) and too much TD ↔ PIGD back-and-forth (direct reversion 0.44 vs 0.31). It *does* reproduce entry reversion (S2) and the within-patient clustering of Indeterminate (S5).

**The misfit pattern differs from the planted intermediate-state signature.** PC4 raises S1 *together with* S5 (mean z −1.22). The observed S5 is consistent with the model (z +0.21). The observed misfit is concentrated in how *often* Indeterminate occurs and how often direct changes reverse, not in Indeterminate visits persisting or clustering.

**The verdict does not establish an intermediate state.** B.11.2(d) names features of the real data that the null does not generate, and any of them could produce this pattern:

1. Ratios take only lattice values. For example 0.909 and 1.136, the band edges where 78 % of Indeterminate visits sit, arise from very common small item sums [R: script 08]. The continuous model cannot represent this discreteness.
2. Visits with PIGD = 0 are removed depending on the outcome.
3. Residuals are skewed (about 1.2) against a symmetric t.
4. Attendance and dropout are informative.

D1 cannot tell which of these, or an intermediate state, is responsible.

**Scope.** D1 excludes the 725 PIGD = 0 visits (including the 60 both-zero Indeterminate visits; 382 patients have sequence gaps). It **cannot establish whether the full three-state phenotype process needs an intermediate state**.

### 6. Deviations in this phase

- The code-path check ran on the login node at low priority: its queued SLURM job never started because the nodes were occupied, and was cancelled while pending (note in its directory).
- The B.13 smoke deviation (observed indicators at smoke settings) is recorded there.
- The full run had no deviations from B.14.
