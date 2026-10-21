# Cross-phase scientific review and next-step decision

**Date:** 2026-10-16.
**Scope:** read-only review. No analysis was run and no existing file was modified.

**Repository state at review:** branch `main` at `3a860ca`, two commits ahead of `origin/main` (Part A `7a59dba`; Student-t work `3a860ca`). The Part C work is uncommitted: `docs/cluster_phase/10_PART_C_RESEARCH_PLAN.md`, `src/ppmi_subtype/{raw_scores,indeterminate_audit}.py`, `scripts/{39,40}_*.py` and `tests/test_{raw_scores,indeterminate_audit}.py`. This document is the only file added by the review.

**Source tags:**

| tag | meaning |
|---|---|
| **[R]** | reported in an existing report or output file (path given) |
| **[V]** | independently checked during this review, by reading code or recomputing arithmetic from saved outputs |
| **[C]** | a conflict or tension between sources, documented rather than resolved |

---

## 1. Executive summary

1. **The labels are built from very coarse measurements.**
   - PIGD is a 5-item sum, 0 in 16 % and 1–2 in about 59 % of labelled visits [R, Part C1].
   - Every Indeterminate visit lies one item point from another label [R, Part C2]. The 0.90–1.15 band contains only 11 attainable ratio values [R, script 08].
   - Label changes into and out of Indeterminate involve a median of 3 item points, the same as visit pairs whose label did not change [R, Part C1/C2].
   - On the same day, OFF and ON exams give the same label for only 75 % of 2,343 visits. This is not test–retest [R, Part C1].
2. **Observed labels have memory, and transitions are dominated by short-lived changes.**
   - The first-order Markov property is rejected for observed labels (χ² p < 0.005 for each current state) [R, `outputs/model/report.txt`].
   - About 54 % of entries into Indeterminate and 35 % of direct TD ↔ PIGD changes revert at the next observed visit [R, Part C2; V: reconciles exactly with script 02].
3. **The best label-level model is a 3-state HMM with misclassification**, at 75.4 % accuracy [73.5, 77.3] in leave-one-patient-out validation.
   - It never predicts Indeterminate, and its Indeterminate emission sits on the imposed 0.5 floor [R].
   - Derived-subtype predictions from subscore models do not beat simple label persistence on balanced accuracy [R, 05_FINDINGS Stage 3].
4. **Continuous scores carry real but weakly identified individual signal.**
   - Per-patient ratio slopes have median reliability 0.05 [R, `outputs/validation/report.txt`].
   - Subscore random-slope reliability is 0.27 (tremor) and 0.53 (PIGD) [R, 05_FINDINGS].
   - Population trends are clear: PIGD rises and tremor flattens over follow-up, gradually, with informative attrition [R].
5. **Heterogeneity exists, but discrete groups are not established.**
   - Clustering in Stage 6/6b and Part A was not supported or was inconclusive. Part A's K = 3 arises in all 200 replicates of a hierarchical null with no groups [R].
   - This is a failure to establish groups, not evidence that all patients follow one trajectory.
6. **Noise model.** Visit-level ratio noise is very heavy-tailed (νe ≈ 2.1), and Student-t residuals improve held-out prediction in 7/7 splits [R].
   - Extreme heavy tails in patient slopes are *not supported* by the pre-specified reassessment (1/7 splits pass).
   - Moderate departures from Gaussian slopes remain unresolved [R].
7. **Indeterminate.** The primary Part C2 reading under the amended rule is **inconclusive**.
   - The reversion contrast establishes its direction: +0.194 [0.108, 0.281].
   - Temporal clustering does not: O/E 1.18 [0.99, 1.39] [R].
   - Indeterminate is therefore neither shown to be noise nor shown to be a distinct state.
8. **Recommendation.**
   - **First**, consolidate the results into one coherent research report (Option 6). Several earlier reports are known to be stale (05_FINDINGS "superseded claims" table), and the evidence now converges on a measurement-limited picture.
   - **Second**, run one pre-specified *diagnostic* (Option 1): a posterior-predictive label check of the existing best ratio model. It asks whether a continuous trajectory plus heavy-tailed visit noise, with no intermediate state, reproduces the observed Indeterminate behaviour.
   - No new model family, clustering method or hurdle model is recommended now.

---

## 2. Evidence matrix

### A. Phenotype construction and data quality

| question | analysis | main result | supports | does not establish | limitations | sources |
|---|---|---|---|---|---|---|
| How are the labels defined? | code review | Tremor = mean of 11 items (1 Part II, 10 Part III); PIGD = mean of 5 (2 Part II, 3 Part III). TD if ratio ≥ 1.15; PIGD if ≤ 0.90. PIGD = 0 with tremor > 0 → TD; both 0 → Indeterminate. A missing, 101 or out-of-range item makes the subscore missing. [V] | The label is defined at all 4,441 labelled visits, including the 725 with PIGD = 0 where the ratio is undefined [R] | — | Ratio models must drop the PIGD = 0 visits that the labels keep [V] | `src/ppmi_subtype/subtype.py`; `config/config.yaml` |
| Is the reconstruction faithful? | item-level rebuild | 0 mismatches in tremor score, PIGD score and label over 5,774 visits [R] | Part C scores = panel scores | — | — | `experiments/20261015T221426Z_partC_raw_scores/REPORT.md` §1; `results.json` |
| Why are visits unlabelled? | reason codes | 1,333 unlabelled: no OFF/untreated Part III 686; Part III row but items absent 559; a 101 code 52; no item rows 36 [R] | Exclusion is mostly exam availability, not ratio failure | Whether availability depends on severity | The OFF filter keeps untreated patients and rows with no PDSTATE [V] | Part C1 REPORT §2 |
| Can the curated label be used? | crosstab | The curated `td_pigd` is binary; all 441 Indeterminate visits are coded with PIGD [R] | Using the three-class label requires reconstruction | — | — | `10_PART_C_RESEARCH_PLAN.md` §2 |
| Does the Part II/III mixture matter? | Part III-only sensitivity | 83.2 % visit agreement; Indeterminate is the least stable class (85/231 retained) [R] | The label depends materially on the construction | That either construction is "correct" | Thresholds are not validated for Part III-only | Part C1 REPORT §4 |
| How coarse is the measurement? | lattice and distributions | PIGD sum = 0: 16.3 %; every Indeterminate visit has margin 1 [R]. OFF/ON same-visit agreement 75.2 % (Part III-only 72.1 %), 2,343 pairs, 714 patients [R] | Labels can flip with a 1-point change | That a 1-point change is clinically meaningful (no validated MCID) | OFF/ON is not test–retest | Part C1 REPORT §2, §5; Part C2 REPORT §2 |
| Who is excluded from longitudinal analyses? | included vs excluded | Patients with only one labelled visit are older (65.2 vs 63.0) and start PIGD more often (26 % vs 17 %) [R] | Mild selection towards less severe patients | — | — | Part C1 REPORT §2 |

### B. Observed transitions and HMM/CTMC

| question | analysis | main result | supports | does not establish | limitations | sources |
|---|---|---|---|---|---|---|
| How do observed labels change? | pooled and interval-stratified transition matrices (3,361 pairs, 934 patients) | TD → TD 0.808, PIGD → PIGD 0.702, Ind → Ind 0.192. TD → PIGD rises from 0.100 (≤ 1.5 y) to 0.289 (> 4 y) [R] | Indeterminate is rarely repeated; changes depend on the interval | Biological conversion | Pair-level Wilson intervals ignore clustering within patients | `outputs/transitions/report.txt` §1–2 |
| Do changes reverse? | reversion one visit later | 35.2 % of all changes reverse [R]. Part C2: entries into Ind 0.543 [0.471, 0.615] vs direct 0.349 [0.298, 0.397], patient bootstrap [R]; the counts reconcile exactly with script 02 (102/188; 99/284) [V] | A large share of observed change is transient | That transient = measurement error | 80 entries and 110 direct changes have no later visit (excluded) [R] | `outputs/transitions/report.txt` §4; Part C2 REPORT §1 |
| Is the label process Markov? | χ² on labelled triples | Rejected for every current state (p ≤ 0.004) [R] | A latent-state or misclassification framing | Which latent structure is right | — | `outputs/model/report.txt` §1 |
| Does a misclassification HMM predict better? | 10 patient splits; leave-one-patient-out | HMM beats the discrete chain in 10/10 splits (+0.028 log-likelihood per transition). LOPO accuracy 0.754 [0.735, 0.773], log-likelihood −0.673 [R] | The HMM is the best label predictor tested | That latent states are biological subtypes | Accuracy is dominated by persistence: TD → TD 99 % correct, TD → PIGD 8 %; Indeterminate never predicted (recall 0) [R] | `outputs/model/report.txt` §3, §5; `outputs/loo/report.txt` §4–6 |
| Are HMM parameters interpretable? | multi-start stability | The 3-state floor-0.5 model has a unique optimum; the unconstrained model lies on a ridge; the 5-state model is multimodal; extra states give no held-out gain [R] | Predictions, not parameters, are robust | Latent sojourn times | The Indeterminate emission is fixed at the floor (0.25/0.50/0.25) in every fold [R] | 05_FINDINGS Stage 4 |
| How long does Indeterminate last? | CTMC, HMM, Kaplan–Meier | Observed-label CTMC mean sojourn 0.44 y; HMM latent 2.12 y [R]; Kaplan–Meier median time to leave the observed label 1.08 y [R, Part C2] | — | — | **[C]** The three are different estimands. The HMM figure belongs to a latent state whose emission is constrained by the floor, so it is not an estimate of how long patients are "truly" Indeterminate | `outputs/model/report.txt` §2; Part C2 REPORT §2 |
| Does a first visit predict the future? | new-patient model (5 splits) | Baseline ratio + covariates beat the history-free HMM by +0.032 log-likelihood per prediction [R] | Baseline clinical measures carry information | Causal predictors | — | `outputs/newpatient/report.txt` |

### C. Bayesian ratio growth and slope clustering

| question | analysis | main result | supports | does not establish | limitations | sources |
|---|---|---|---|---|---|---|
| Population ratio trend | hierarchical Gaussian growth model (821 patients) | β1 = −0.090 /yr [−0.109, −0.071]; τ1 = 0.102 [R] | An average decline in the ratio | Individual trajectories | Excludes PIGD = 0 visits; the linear trend is an assumption | `outputs/hbayes/report.txt`; Part A manifest |
| Are individual slopes identified? | posterior reliability | Median slope reliability 0.05; > 0.5 for only 8.2 % of patients [R] | Patient slopes are mostly shrunk to the mean | — | Driven by visit count and span | `outputs/validation/report.txt` Stage 4b |
| Are there discrete slope groups? | Part A: GMM on posterior means, 500 posterior draws, 3 nulls, Silverman test | BIC selects K = 3 on posterior means (ΔBIC 316), but K = 1 in 500/500 posterior draws. The hierarchical no-groups null selects K ≥ 2 in 200/200 replicates (median ΔBIC 227). Silverman p = 0.24 [R] | **No reproducible discrete groups** | That all patients share one trajectory; that the slope distribution is Gaussian | The components differ in spread, not centre, and track visit count [R] | `docs/cluster_phase/run_manifests/20261015T171236Z_bayesian_slope_clustering/REPORT.md` |
| Subscore clustering (Stage 6/6b) | random-slope BLUP features, calibrated nulls | Not supported; Stage 6b inconclusive (no adequately calibrated null) [R] | As above | As above | The measurement model partly explains BLUP skew [R] | 05_FINDINGS Stage 6/6b |

### D. Heavy-tailed residuals vs patient slopes

| question | analysis | main result | supports | does not establish | limitations | sources |
|---|---|---|---|---|---|---|
| Is visit-level noise heavy-tailed? | GG/TG/GT/TT comparison over 7 patient splits | ΔLPD GT − GG +0.112 [0.061, 0.168] and TT − TG +0.106 [0.058, 0.158], 7/7 splits; νe 2.10 [1.88, 2.39], R̂ ≤ 1.001 [R] | **SUPPORTED** | The cause of the heavy tails | Symmetric t, while real residuals are skewed [R] | `run_manifests/20261015T190228Z_student_t_four_model_full/REPORT.md` |
| Do t residuals predict better? | script 04 class metric | Class log-likelihood vs GG: GT +0.038 [0.020, 0.055], TT +0.041 [0.023, 0.059], 7/7 splits; TG +0.003 [0.001, 0.006], 6/7 [R] | Yes, mainly through the residuals | — | One chain per split | same |
| Extreme slope tails? | Q2 reassessment (collapsed slice sampler) | Convergence passes (R̂ 1.004, ESS 937; log-uniform 1.0005 / 4,394). (a) passes under both priors (97.5 % quantile 4.11 / 2.99). **(b) 1/7 splits** (97.5 % quantiles 3.3–37.0). (c) 0/60 calibration false positives in each scenario (two-sided 95 % CP [0, 5.96 %]) [R] | **NOT SUPPORTED** under the pre-specified criteria | That slopes are Gaussian; moderate tails remain open | The upper tail of νs is prior-sensitive in 70 % subsets [R] | `run_manifests/20261015T203325Z_q2_reassessment/REPORT.md`; spec `09_Q2_REASSESSMENT_SPEC.md` |
| History of the verdict | original run | The original Q2 was INCONCLUSIVE (convergence; R̂ 1.018, ESS 248). The "Gamma" grid prior was really a truncated Exponential (erratum) [R] | — | — | Both verdicts are on record; neither is rewritten | specs 07/08 errata; `_frozen/` |

### E. Part C raw scores and the Indeterminate audit

| question | analysis | main result | supports | does not establish | limitations | sources |
|---|---|---|---|---|---|---|
| Reversion contrast | C2 primary (934 patients; 4,000-resample patient bootstrap) | Δ_R = +0.194 [0.108, 0.281]; entries 188 observed / 80 unobserved; direct 284 / 110 [R] | Entries into Indeterminate revert more often than direct changes | That entries are measurement error | Only events with an observed next visit | Part C2 REPORT §1 |
| Temporal clustering | exact within-patient reordering expectation | O = 64, E = 54.26 (evaluable), O/E = 1.179 [0.988, 1.391]; 94 contributing patients [R] | — | Presence *or* absence of clustering | Monotone drift alone can inflate O/E (TD 1.05, PIGD 1.14) [R] | same |
| Verdict | Amendment 1 rule | **Inconclusive.** The superseded rule would have said "transient" (record only) [R] | — | Noise; a distinct state; progression | The amendment was made before the full run, with approval | plan Amendment 1; Part C2 REPORT |
| Robustness | 7 sensitivities | Δ_R > 0 in all except Part III-only (+0.073 [−0.035, 0.179]). O/E spans 1 in all except Part III-only (1.436 [1.203, 1.753]); genetic PD not evaluable (E = 18.2) [R] | The reversion direction is robust to interval, treatment, missing-item and both-zero choices | — | Part III-only gives the opposite pattern: construction-dependent [R] | Part C2 REPORT §3 |
| Score changes at events | descriptive | Median total change 3 points for entries, exits and stays; 5 for direct changes [R] | Indeterminate transitions do not involve unusually large score changes | — | Biased by construction as a test (a label change requires a score change) | Part C1 REPORT §3; Part C2 REPORT §2 |
| Persistence | descriptive | Ind sustained at 2 later visits: 3/135 (0.022). Kaplan–Meier median time to leave: Ind 1.08 y, PIGD 2.08 y, TD 5.41 y [R] | Indeterminate labels are short-lived | Short-lived *latent* states | Censoring and informative dropout [R] | Part C2 REPORT §2 |

---

## 3. Established, unsupported and unresolved

**Established** (pre-specified or held-out, with uncertainty):

- The reconstruction is faithful.
- Labels depend on the score construction.
- Observed labels are not first-order Markov.
- The 3-state misclassification HMM predicts best among label models (but never predicts Indeterminate).
- Population PIGD rises and tremor flattens gradually.
- Visit-level ratio noise is heavy-tailed, and modelling it improves prediction.
- Entries into Indeterminate revert more often than direct TD ↔ PIGD changes.

**Not supported** (pre-specified criteria not met):

- Reproducible discrete slope or trajectory groups (Stage 6, Part A).
- Extreme patient-slope heavy tails (Q2 reassessment).
- Splitting Indeterminate into near-TD and near-PIGD (script 08).
- A joint subscore model being better than independent ones (Stage 3).

**Unresolved:**

- Whether Indeterminate shows sustained within-patient structure (O/E inconclusive).
- Moderate non-Gaussian slope heterogeneity.
- Why PIGD zeros persist differently from every fitted model (Stage 6b).
- Whether the follow-up curvature is disease stage or time-in-study (not separable [R]).
- How much medication timing contributes to label changes.

---

## 4. Remaining gaps (candidate questions 1–7)

| # | why it matters | already known | missing evidence | answerable with these data? | analysis that could answer it | new or duplicate | assumptions |
|---|---|---|---|---|---|---|---|
| 1. Indeterminate: transient or sustained? | Determines whether a 3-state framing is justified | Reversion contrast established; clustering inconclusive; HMM Indeterminate emission at the floor | Whether a model *without* an intermediate state reproduces the observed Indeterminate statistics | Partly. The O/E precision cannot improve without more data, but model adequacy can be checked | Posterior-predictive check of the existing heavy-tailed ratio model on the C2 statistics (§6) | **New** (diagnostic, no new model) | The ratio model covers ratio-defined visits only; conclusions are limited to that subset |
| 2. Reproducible subscore heterogeneity beyond labels | Core progression question | Population trends established; slope reliability low; clustering not supported | A reproducibility estimand for individual differences that does not depend on clustering | Weakly: few patients have ≥ 8 visits (145) | Split-half (early vs late visits) agreement of patient-level deviations | Partly duplicates Stage 6 (drop-one-visit ARI) | Within-patient stationarity of deviations |
| 3. Do zeros and missing items change conclusions? | Selection and measurement validity | One-missing-item sensitivity changes nothing for C2; PIGD = 0 is mis-modelled by every model (Stage 6b) | Effect of the zero misfit on the *ratio* conclusions | Yes | Re-run C2 or the four-model comparison with PIGD = 0 visits handled explicitly | Would duplicate Part C sensitivities unless a new likelihood is used | — |
| 4. Hurdle model: a fix or added complexity? | PIGD zero misfit is documented | Gaussian, heteroscedastic, beta-binomial and GRM all fail the zero features (script 31; Stage 6b) | Whether better zero modelling changes any substantive conclusion | Yes, technically | Separately specified hurdle model | New, but **no current conclusion is known to hinge on it** | Patient-level zero process; adequate identification |
| 5. Joint intercept + slope structure | Slope-only clustering may miss structure | Intercepts well identified (reliability 0.80), slopes not (0.05) [R]; Part A null | Evidence that adding intercepts creates *reproducible* groups rather than baseline-severity strata | Doubtful: intercepts dominate any joint feature space | Part B with the hierarchical null and per-draw checks | Partly duplicates Stage 6 (bivariate BLUPs were used) | Shrinkage-aware features |
| 6. Continuous heterogeneity vs discrete groups without clustering metrics | Clustering metrics proved uninformative | Hierarchical nulls make any K > 1 expected; Stage 6b could not calibrate a null | A likelihood-based comparison (e.g. Gaussian vs mixture random-effects distribution) with held-out evaluation | Limited by slope reliability | Held-out predictive comparison of random-effects distributions (Gaussian vs 2-component mixture) | New in form; outcome likely weak given reliability | Correct within-patient model |
| 7. What can be inferred about progression? | Interpretation boundary | Attrition informative; treatment initiation coincides with early follow-up; disease stage not separable from follow-up [R] | Medication-timing effects on labels | Partly (HRPOSTMED for 95 % of ON exams [R]) | Descriptive OFF/ON timing analysis | New, but descriptive | — |

---

## 5. Candidate next steps, ranked by scientific priority

| rank | option | scientific value | real gap? | overlap with past work | feasibility | key assumptions | validation needed | cost | risk of misleading results | what would justify going further |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **6. Consolidated research report** | High: several reports are stale (05_FINDINGS "superseded claims" table), and the cross-phase picture is coherent | Yes, for communication and integrity | None | Immediate | — | Every number traceable | Low | Low | — |
| **2** | **1. One pre-specified diagnostic: posterior-predictive label check (§6)** | Medium–high: tests whether "continuous trajectory + heavy-tailed noise" explains Indeterminate behaviour without a latent intermediate state | Yes (question 1) | Uses the existing TT/GG fits | High (fits take about 10–40 s each) | The ratio model is adequate for ratio-defined visits | Patient-level splits; pre-specified statistics | Low (< 1 h on one node) | Medium: a PPC "pass" is not proof of no state | A clear misfit would justify a model with an explicit intermediate component |
| 3 | 5. HMM/CTMC refinement | Low: predictions are already robust across 3–5 states; the Indeterminate emission binds at the floor | Partly (interpreting Indeterminate) | High | High | Misclassification structure | Already LOPO | Medium | High: latent-state parameters are not identified | — |
| 4 | 2. Hurdle model | Medium for measurement realism; low for current conclusions | The misfit is real, but no current conclusion is known to hinge on it | Extends script 31 / Stage 6b | Fits are slow (the beta-binomial fit took 68 min) | Patient-level zero process | 5-fold patient CV, pre-specified | High | Medium | A demonstrated conclusion that changes when zeros are modelled |
| 5 | 3. Joint intercept + slope (Part B) | Low | Questionable: intercepts dominate; slope reliability 0.05 | Stage 6 used bivariate features | Feasible | Shrinkage-aware features | Hierarchical null, per-draw checks | Medium | **High**: groups would mirror baseline severity | Only if question 6 first shows non-Gaussian *structure* that is reproducible |
| 6 | 4. Robust LPA / trajectory clustering | Low on current evidence | Not demonstrated | High (Stage 6, 6b, Part A) | Feasible | Mixture of profiles; adequate features | Calibrated no-groups null, which Stage 6b could not build | Medium | **High** (multiple-component solutions arise under continuous nulls [R]) | A calibrated null and reproducible assignments, neither of which is currently available |

**Note on RobustLPA.** No implementation or reference exists in the repository. Latent profile analysis models *cross-sectional* indicator profiles. Applied to trajectories, it needs derived per-patient features, which here have low reliability, and it faces the same no-groups-null problem that closed Stage 6b. Its exact definition and intended use need clarification (§7).

---

## 6. Recommended next analysis: pre-specified protocol (draft for approval)

**D1 — Posterior-predictive label check of the heavy-tailed ratio model.**

1. **Question.** Do the existing continuous ratio models (TT, with GG as reference), which have no latent intermediate state, reproduce the observed Indeterminate statistics of Part C2 on held-out patients?
2. **Population.** Script 04's patient-level splits (seeds 42, 0–5).
   - The analysis is restricted to **ratio-defined visits** (finite ratio), because the ratio model cannot generate PIGD = 0 visits.
   - Labels are recomputed from simulated ratios with the unchanged thresholds.
   - This restriction is a stated limitation. The ratio-defined subset contains 381 of the 441 Indeterminate visits [R, script 08].
3. **Models.** The existing `hbayes` (GG) and `hbayes_tt_nu` collapsed-slice TT, fitted on training patients with the frozen settings. No new likelihood.
4. **Replication.** For each test patient, draw θ from the training posterior and b_i from p(b_i | the patient's observed ratios, θ). Simulate new visit-level noise at the patient's real ratio-defined visit times. Use 500 replicates per split.
5. **Statistics** (pre-specified, computed identically on observed and simulated sequences with `indeterminate_audit`):
   - (S1) Indeterminate visit share;
   - (S2) R_E;
   - (S3) R_D;
   - (S4) Δ_R;
   - (S5) O/E_Ind;
   - (S6) two-step Indeterminate persistence.
6. **Primary estimand.** For each statistic, the posterior-predictive tail probability p = P(T_rep ≥ T_obs) on the primary split.
7. **Null / baseline.** GG serves as the reference model. TT is the model under test.
8. **Decision criteria**, for TT (GG reported alongside):
   - **Reproduced:** S1–S5 all within the central 95 % of the replicate distribution (0.025 ≤ p ≤ 0.975) on the primary split **and** in ≥ 5/7 splits.
   - **Not reproduced:** any of S2, S4 or S5 outside on the primary split **and** in ≥ 5/7 splits, in the same direction.
   - **Inconclusive:** otherwise.
   - S6 is descriptive, because its counts are small.
9. **Interpretation limits.**
   - "Reproduced" means the continuous model is *sufficient* for these statistics on ratio-defined visits. It is not proof that no intermediate state exists.
   - "Not reproduced" identifies which feature the continuous model misses. It is not proof of a biological state.
10. **Missing data.** Unlabelled and ratio-undefined visits are removed before sequences are formed, the same in observed and simulated data. Patients with fewer than 2 ratio-defined visits are excluded and counted.
11. **Uncertainty.** Replicate distributions per split, plus split-to-split variation. No p-value thresholds across the many descriptive outputs.
12. **Sensitivities.** Log-uniform ν prior; intervals ≤ 1.5 years; treatment-stable segments.
13. **Tests.**
    - The simulation reproduces the posterior predictive on synthetic data with known parameters.
    - Observed statistics through the simulation path equal those from the Part C2 path on the same sequences.
    - Patient separation: no test patient appears in the training data.
    - Seeds and SHA-256 hashes are recorded before computation.
14. **Compute.** 2 models × 7 splits, about 30 s to fit each, plus 500 replicates of cheap simulation: under 1 hour on one node.

No MCID is used anywhere. All differences are on the ratio or label scale.

---

## 7. Decisions needing your input

1. **Priority.** Approve the ranking: the consolidated report first, then D1. Or choose to run D1 before writing.
2. **D1 design.** Is conditioning b_i on the test patient's own observed ratios (a conditional posterior-predictive check) acceptable? Is the restriction to ratio-defined visits acceptable?
3. **RobustLPA.** What exactly is meant (package, model, features), and is it still wanted given §5? It is not recommended on current evidence.
4. **Part C commit.** Should the uncommitted Part C files be committed (as Mayank, with aggregate tables in `run_manifests/`)?
5. **Report scope.** Should the consolidated report supersede `GPR_onwards_explanatory_report.md` and the midterm-era reports, or sit beside them with a "superseded claims" appendix?

---

## 8. Source paths

**Code:**
- `src/ppmi_subtype/{subtype,panel,loaders,transitions,hmm,ctmc,hbayes,hbayes_t,hbayes_tt,hbayes_tt_nu,slope_clustering,raw_scores,indeterminate_audit,bbglmm,jointlmm}.py`
- `config/config.yaml`
- tests under `tests/`

**Reports and outputs:**

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
| Part A | `docs/cluster_phase/run_manifests/20261015T171236Z_bayesian_slope_clustering/` |
| Student-t work | `run_manifests/20261015T182129Z_student_t_slope_smoke/`, `20261015T184027Z_student_t_residual_smoke/`, `20261015T190228Z_student_t_four_model_full/`, `20261015T200100Z_nu_slope_sampler_pilot/`, `20261015T203325Z_q2_reassessment/` |
| Part C | `experiments/20261015T221426Z_partC_raw_scores/REPORT.md`, `experiments/20261015T221426Z_partC_indeterminate/REPORT.md` |
| PIGD measurement models | `run_manifests/20261014T144511Z_pigd_measurement_models/predictive_checks.csv` |

**Specifications:**
- `docs/cluster_phase/06_*`, `07_STUDENT_T_SLOPE_SPEC.md` (erratum), `08_STUDENT_T_RESIDUAL_SPEC.md` (erratum), `09_Q2_REASSESSMENT_SPEC.md`, `10_PART_C_RESEARCH_PLAN.md` (Amendment 1)
- frozen copies under `docs/cluster_phase/_frozen/`
