# Stage 5 — manuscript outline

**Date** 2026-11-04 · **Follows** `04_SCIENTIFIC_POSITIONING` and
`05_HARMONISED_FAMILY_COMPARISON` · **Target** npj Parkinson's Disease, Article

Every section carries the eight fields the brief asks for. **No manuscript prose
is drafted here.** Where a claim has no evidence yet, the section says so rather
than leaving a gap to be filled later.

**Working title.** *Label reliability, not model capacity, limits longitudinal
prediction of Parkinson's disease motor subtypes* — 14 words, no punctuation,
within the npj 15-word rule.

✅ **All four outstanding deliverables were built on 2026-11-04** by
`scripts/80_manuscript_assets.py` into `manuscript/npj/` — Figure 2, Figure 4,
Figure 5(b) and Table 1, with companion CSVs, 12/12 internal checks and 9/9
source checks against E-2, E-8 and the frozen panel. Every asset in §4 now
exists.

---

## 1. Abstract — content list

Unstructured, ≤150 words (npj Article; re-check at submission, §7 of Stage 4).
Drafted last, at Stage 10. It must carry, in this order:

1. The subtype is a thresholded ratio of two MDS-UPDRS item means and is widely
   used as a longitudinal endpoint.
2. PPMI, 1,080 patients, 4,441 labelled visits, 3,361 consecutive transitions,
   leave-one-patient-out.
3. 35.2% of observed subtype changes reverse at the very next visit.
4. Nine specifications across five model families fall within 1.2 accuracy
   points of each other.
5. All exceed a persistence rule by 2.3–3.5 points, but 94% of that margin
   comes from the 9.9% of visits recorded Indeterminate.
6. No model predicts Indeterminate; the trivial rule has the best macro-F1.
7. A continuous nine-domain profile discriminates prospective Hoehn & Yahr ≥ 3
   and MCI better than the label does.
8. Conclusion: the constraint is the reliability of the label, not model
   capacity.

**Not in the abstract:** any claim of clinical utility, any "first", the
two-class 0.84 figure without its 0.75 counterpart.

---

## 2. Introduction

| Field | Content |
|---|---|
| **Objective** | Establish that the TD/PIGD label is a routinely used longitudinal endpoint whose reliability is rarely measured before it is modelled, and state the question. |
| **Claim** | Subtype instability is established; what is not established is how much of it is immediate reversion, and whether model capacity or label reliability is the binding constraint on prediction. |
| **Evidence** | Literature only. |
| **Experiment** | None. |
| **Sources** | `03_LITERATURE_REVIEW_20261104.md` §2, §4. |
| **Citations** | Stebbins 2013 (definition); Simuni 2016, Lee 2019, Erro 2019, von Coelln 2021 (instability); von Coelln 2021, Ren 2021 (algorithms disagree); Kotagal 2016 (construct contested); Mestre 2021 (stability not assessed for most systems); Bouwmeester 2012 (calibration under-reported). |
| **Missing** | Nothing. |
| **Placement** | Main. |

**Paragraph plan.** (1) PD motor phenotype and the TD/PIGD rule. (2) It is used
as a longitudinal endpoint and a stratifier. (3) Four studies show it is
unstable and three algorithms disagree; Mestre records that stability was never
assessed for most systems. (4) Prior predictive work targets model-generated
clusters from baseline, rarely reports calibration, and — in what I retrieved —
never reports a persistence baseline. (5) The gap: instability has been
measured as net drift, not as immediate reversion, and no study compares model
families under one protocol. (6) **This paper proposes no model**; it reports a
protocol and a set of measurements. (7) Question and roadmap.

---

## 3. Results

### R1 — Cohort, label construction, and agreement with independent analyses

| Field | Content |
|---|---|
| **Objective** | Establish the analysis population and that the label reconstruction is sound. |
| **Claim** | The reconstructed label reproduces PPMI's own derivation exactly and matches an independent published analysis of the same cohort to within one percentage point. |
| **Evidence** | 1,080 patients, 5,774 visits, 4,441 labelled, 3,361 transitions from 934 patients. Internal agreement 4,431/4,431 (100.00%) 2-class and 1,595/1,597 (99.87%) 3-class. Baseline TD 71.7% / Ind 10.1% / PIGD 18.1% against Simuni's 71 / 11 / 18. |
| **Experiment** | E1 (panel build), E-5 (flow). |
| **Sources** | `outputs/panel_summary.txt`, `outputs/validation.txt`, E-5 run `flow_ledger.csv`. |
| **Citations** | Stebbins 2013; Simuni 2016; von Elm 2007 (STROBE). |
| **Missing** | **Table 1 does not exist** (§4). **C6:** `data/validation/` is empty, so the 99.87% check cannot currently be rerun — must be disclosed or the file restored. |
| **Placement** | Main. **Figure 1**, **Table 1**. |

**The framing that must not slip.** The internal agreement checks compare
against PPMI's own derivation of the *same formula from the same items*. They
verify the implementation, not the label. Call them a reproduction check. The
Simuni comparison is the external one and should carry the weight.

### R2 — The recorded label is unstable, and much of the change is immediate reversion

| Field | Content |
|---|---|
| **Objective** | Measure label reliability directly, before any model. |
| **Claim** | A third of observed subtype changes reverse at the very next visit; requiring two consecutive assessments removes half of apparent conversions; and the classification is fragile to threshold perturbations smaller than the measurement's own resolution. |
| **Evidence** | Transition matrix with patient-clustered intervals; reversion 35.2% (95% CI 31.8–38.8); sustained definition removes 51.7% (45.7–57.7) of conversions, 263 → 127; KM year-5 incidence 35.3% (30.8–39.9) naive against 15.3% (12.0–19.0) sustained; TD→PIGD 0.100 (0.087–0.116) at ≤1.5 y against 0.289 (0.167–0.432) beyond 4 y; a 0.02 threshold shift reclassifies 67.6% of the Indeterminate class; half of Indeterminate visits sit at ratio 0.9091 and a quarter at 1.1364. |
| **Experiment** | E2, **E-2** (intervals), E19 (focal ratios), threshold recomputation. |
| **Sources** | E-2 run `transition_probabilities.csv`, `other_quantities.csv`; `docs/model_comparison/04_INDETERMINATE_RATIO_SCRUTINY`; Stage-1 audit §13. |
| **Citations** | Simuni 2016, Lee 2019, von Coelln 2021 (for the comparison paragraph). |
| **Missing** | **Figure 2 does not exist** (§4). **O-4:** the threshold-sensitivity result is my recomputation from the frozen panel, not a saved run — run it so the claim rests on a manifest. |
| **Placement** | Main. **Figure 2**, **Table 2**. |

**Comparison paragraph.** Per-visit TD persistence of 0.808 compounds to
0.34 over five annual visits, inside von Coelln's 15–36% at 6.2 years; the
project's own >4-year stratum (0.578) matches Lee's 4-year figure (0.583)
despite Lee using the older 1.5/1.0 thresholds. State that cohort composition
explains von Coelln's lower figure (their PIGD prevalence is 59–75% against
PPMI's 25%).

### R3 — Nine specifications across five model families converge *(PRIMARY)*

| Field | Content |
|---|---|
| **Objective** | Test whether model family determines predictive performance. |
| **Claim** | Under one protocol, one decision rule and one evaluation set, nine specifications spanning five families fall within 1.2 accuracy points; all exceed a persistence rule by 2.3–3.5 points; and none recovers the Indeterminate class. |
| **Evidence** | Accuracy 0.7453–0.7575 across nine fitted specifications; persistence 0.7227; paired contrasts +0.0226 to +0.0348, every interval excluding zero; of fourteen specifications only five ever predict Indeterminate and four of those do so 1–4 times; **the trivial rule has the best macro-F1 of all fourteen (0.5559)**. |
| **Experiment** | **E-8**, drawing on E7, E8, E12, E13, E14, E15. |
| **Sources** | E-8 run `harmonised_metrics.csv`, `paired_contrasts.csv`, `fig1_harmonised_family_comparison.png`. |
| **Citations** | Jackson 2003 (the CT-HMM's framework); Severson 2021, Dadu 2022 (what prior predictive work targets instead). |
| **Missing** | Nothing. |
| **Placement** | Main. **Figure 3**, **Table 3**. |

**Must be stated here.** The harmonisation changed published numbers: the joint
subscore LMM and the time-varying LMM were previously recorded below
persistence and are not. Report the correction in the text, and report in
Methods that models already on a 3-class vector reproduced their committed
accuracy exactly (max difference 4 × 10⁻⁵), which is what validates the
re-scoring.

### R4 — The advantage lies entirely in the Indeterminate band

| Field | Content |
|---|---|
| **Objective** | Localise the margin found in R3. |
| **Claim** | The models do not predict progression better than the trivial rule; they decline to repeat a class that is almost never repeated, and that is the whole of their advantage. |
| **Evidence** | For the CT-HMM, 94.3% of its gain comes from the 334 cases (9.9%) where the current label is Indeterminate; on current-TD it gains 6 cases of 2,261 and on current-PIGD **exactly zero** of 766; it makes the same prediction as persistence on 98.4% and 96.9% of those two strata. The discrete Markov chain is identical to persistence on **100%** of current-TD and current-PIGD cases and takes 100% of its gain from the Indeterminate band. The latent state is worth +0.0092 (CI 0.0003–0.0179) over the identically interval-aware CTMC, concentrated as +6.3 pp on current-Indeterminate cases. The hidden layer is supported by ΔAIC 224 / ΔBIC 186, and a chain in the observed label is rejected (χ² = 156.6, df = 4, p = 7.9 × 10⁻³³). |
| **Experiment** | **E-8** decomposition; E17 contrasts; E19 (AIC/BIC); E3 (Markov test). |
| **Sources** | E-8 run `margin_decomposition.csv`; `experiments/…next_visit…/contrasts.csv`; `outputs/model/markov_property_test.csv`; `experiments/…model_selection_criteria/aic_bic.csv`. |
| **Citations** | Jackson 2003. |
| **Missing** | **Figure 4 does not exist** (§4). |
| **Placement** | Main. **Figure 4**. |

**Caveat to carry.** The decomposition is descriptive, computed on the same data
it describes, with no multiplicity correction. It localises an
already-established contrast; it is not a new hypothesis test.

### R5 — Restricting the target to two classes changes the task, not the model

| Field | Content |
|---|---|
| **Objective** | Pre-empt the most likely misreading of the project's own numbers. |
| **Claim** | The same model scores 0.754 on three classes and 0.837 on two; the entire difference is the removal of 332 targets it never predicts. On the two-class task it is indistinguishable from persistence on accuracy but ranks better. |
| **Evidence** | 2,534/3,361 = 0.7539 → 2,534/3,029 = 0.8366 → 0.8369 after renormalisation. CT-HMM 0.8369 vs persistence 0.8336, Δ +0.0033 (CI −0.0037 to +0.0100). Rich profile 0.8404, Δ vs CT-HMM +0.0033 (CI −0.0050 to +0.0112). AUC 0.8902 vs 0.8583, Δ +0.0319 (CI +0.0229 to +0.0419). CT-HMM calibration gap 0.0065 across five bins. |
| **Experiment** | E17, E18, E19. |
| **Sources** | `manuscript/NOTE_accuracy_clarification.md`; `experiments/…next_visit…/model_comparison.csv`; `…clinical_profile_cv/derived_label_vs_subtype_history.csv`; `…roc_auc_comparison/auc_*.csv`. |
| **Citations** | Bouwmeester 2012 (calibration is rarely reported). |
| **Missing** | Nothing. |
| **Placement** | Main. **Table 4**. |

**Fairness caveat, mandatory.** The CT-HMM is LOPO and the profile is 5-fold ×
4, so the CT-HMM is the more favourably validated. Matching cases removes the
evaluation-set difference, not the scheme difference. Also disclose that the
profile sees the target visit's prescribed LEDD, measured as worth −0.0013
(CI −0.0056 to +0.0026).

### R6 — History beyond two or three visits adds nothing, and retraining hurts

| Field | Content |
|---|---|
| **Objective** | Rule out "the models would win with more history". |
| **Claim** | Prediction-time history beyond about three visits is worth nothing; a single visit is measurably worse; retraining on two-visit windows degrades every probability metric; and stage-1 prediction is as accurate as stage-9. |
| **Evidence** | last-2 vs full: accuracy −0.0041 (CI −0.0189 to +0.0100), all five metrics' intervals containing zero. last-1 vs full: −0.0218 (CI −0.0423 to −0.0014), AUC −0.0617. Window retraining: balanced accuracy −0.0312 (CI −0.0441 to −0.0183), AUC −0.0210; the damage is in the training change (−0.0248) not the history restriction (−0.0064). Stage 1 accuracy 0.8347 against pooled 0.8373; no cross-stage accuracy contrast excludes zero; persistence's balanced accuracy rises identically across stages. |
| **Experiment** | E20, E21, E22. |
| **Sources** | `docs/history_window/*`, `docs/stage_wise_evaluation/*`. |
| **Citations** | None. |
| **Missing** | Nothing. |
| **Placement** | Main, **condensed to two paragraphs**; full tables to Supplementary S5. **Figure 6**. |

### R7 — A richer representation carries clinical information the label discards

| Field | Content |
|---|---|
| **Objective** | The constructive counterpart: the label is not only unreliable, it is lossy. |
| **Claim** | A continuous nine-domain level-and-rate representation is genuinely multidimensional and discriminates prospective outcomes better than the subtype label, while being no better at predicting the label itself. |
| **Evidence** | Seven principal components for 90% of across-patient variance; PC1 42.2% (levels) / 37.2% (rates); participation ratio 4.34 / 5.11. Best level coordinate vs the label: Hoehn & Yahr ≥ 3 0.6845 vs 0.5794, Δ +0.105 (CI +0.054 to +0.156); MCI from HVLT/Benton 0.7098 vs 0.5494, Δ +0.161 (CI +0.109 to +0.213); Part IV +0.035 (CI −0.017 to +0.083). Best *rate* coordinates: +0.053, +0.045, +0.041 — **all intervals include zero**. Primary target: +0.0733 (CI +0.0657 to +0.0813) over LOCF, of which the rate component supplies +0.0040 and medication +0.0058. |
| **Experiment** | E18. |
| **Sources** | `…clinical_profile_cv/{representation_dimensionality,outcome_associations,contrasts}.csv`; `manuscript/figures/fig09_*`, `fig07_*`. |
| **Citations** | Severson 2021 (the inspiration, explicitly not reproduced); Tomlinson 2010 (LEDD). |
| **Missing** | Nothing. |
| **Placement** | Main. **Figure 5**. |

**Three caveats, all mandatory.** The outcome AUCs are univariate with the
coordinate chosen on the same data it is compared against, so each Δ is an
upper bound. The investigator cognitive diagnosis is not fully independent of
the feature set and must stay flagged. The medication coefficients are
confounded by indication and support no causal reading.

### R8 — Exploratory: per-patient heterogeneity

| Field | Content |
|---|---|
| **Objective** | Show who the models get wrong, and pre-empt an over-reading. |
| **Claim** | Per-patient accuracy is bimodal rather than concentrated near the mean, and the dominant source of variation is whether the patient's label changes at all — a property of the label sequence, not of any model. |
| **Evidence** | Mean patient-level accuracy 0.8342 (CI 0.8166–0.8519) against visit-level 0.8391. 63.7% of patients correct on every eligible visit, 5.0% on none, median 1.000. By composition: all-TD 0.944, evenly split 0.469 — **but persistence shows 0.970 and 0.480 on the same groups**. Per-patient PIGD sensitivity 0.722 against TD specificity 0.917 among patients with ≥2 targets of the class. |
| **Experiment** | E22 patient-level audit. |
| **Sources** | `…patient_level_accuracy_audit/{patient_accuracy_distribution,patient_accuracy_by_subtype_composition,patient_subtype_metrics}.csv`. |
| **Citations** | None. |
| **Missing** | Nothing. |
| **Placement** | Main, **explicitly labelled exploratory**. **Table 5**. |

**The sentence that must appear.** The composition gradient is matched by the
persistence rule, so it describes the label sequence and not model behaviour.
Groups were defined post hoc from true labels.

---

## 4. Figures and tables

| # | Content | Status | Source |
|---|---|---|---|
| **F1** | Participant flow, every exclusion counted | ✅ **built** | E-5 `fig1_participant_flow.png` |
| **F2** | Label instability: transition matrix with patient-clustered intervals, reversion by change type, TD→PIGD by interval | ✅ **built** | `manuscript/npj/fig2_label_instability.png` |
| **F3** | Nine specifications under one decision rule, plus paired contrasts | ✅ **built** | E-8 `fig1_harmonised_family_comparison.png` |
| **F4** | Margin decomposition by current recorded label | ✅ **built** | `manuscript/npj/fig4_margin_decomposition.png` |
| **F5** | The representation: PCA cumulative variance and outcome AUCs | ✅ **built** — panel (a) `fig09_representation.png`, panel (b) `manuscript/npj/fig5b_outcome_discrimination.png` | as listed |
| **F6** | Robustness: history windows and stage-wise accuracy | ✅ **built** | `docs/history_window/figures/fig3_*`, `docs/stage_wise_evaluation/figures/fig1_*` |
| **T1** | Cohort characteristics at baseline | ✅ **built** | `manuscript/npj/table1_cohort_characteristics.csv` |
| **T2** | Transition counts and probabilities, patient-clustered intervals | ✅ **built** | E-2 `transition_probabilities.csv` |
| **T3** | Harmonised comparison, nine specifications + five baselines | ✅ **built** | E-8 `harmonised_metrics.csv` |
| **T4** | Two-class comparison: CT-HMM, rich profile, baselines | ✅ **built** | `…derived_label_vs_subtype_history.csv`, `auc_metrics.csv` |
| **T5** | Exploratory per-patient composition | ✅ **built** | `…patient_accuracy_by_subtype_composition.csv` |

**F2 specification.** Three panels. (a) 3×3 transition matrix, probabilities
with patient-bootstrap intervals, rows = current. (b) Reversion rate by change
type, six bars with intervals, ordered by rate. (c) TD→PIGD and TD→TD by
interval stratum, showing the gradient and the width of the >4 y interval.
Patient-clustered intervals throughout; n stated per cell.

**F5(b) specification.** Forest plot of ΔAUC (best level coordinate minus
subtype label) for the four prospective outcomes, with patient-bootstrap
intervals, and the corresponding best *rate* coordinates shown beneath in a
lighter weight to make visible that all four of those intervals include
zero. The investigator cognitive diagnosis must be marked as not fully
independent. Source `table5_clinical_outcome_associations.csv`.

**F4 specification.** Two panels. (a) Grouped bars: cases gained over
persistence by current recorded label (TD / Indeterminate / PIGD) for three
representative models, annotated with the share of total gain. (b) Share of
predictions identical to persistence, same strata — the panel that makes the
point. State n per stratum.

**T1 specification.** Columns: full cohort (n = 1,080) and analysed cohort
(n = 934, ≥2 labelled visits). Rows: age, sex, education years, disease
duration, age at onset, Hoehn & Yahr, MDS-UPDRS I/II/III, MoCA, RBDSQ,
SCOPA-AUT, UPSIT, DaTscan putamen SBR, LEDD, tremor and PIGD subscores, the
ratio (noting it is undefined for 26% of baseline visits), and baseline subtype
distribution. Median (IQR) or n (%); state missingness per row.

---

## 5. Discussion

| Field | Content |
|---|---|
| **Objective** | Interpret, compare with the literature, and state what follows. |
| **Claim** | The binding constraint on longitudinal subtype prediction is the reliability of the label, not the capacity of the model; effort belongs on the measurement and the endpoint. |
| **Evidence** | R2–R4 and R7. |
| **Sources** | All of the above. |
| **Citations** | Simuni 2016, Lee 2019, von Coelln 2021 (comparison); Mestre 2021 (field-level recommendation); Kotagal 2016 (construct); Severson 2021, Dadu 2022 (what prior prediction targets); Jackson 2003 (method). |
| **Missing** | Nothing. |
| **Placement** | Main. |

**Paragraph plan.** (1) Principal finding in one sentence. (2) Why convergence
across dissimilar families is stronger evidence about the data than any single
model's accuracy. (3) The Indeterminate band: not a clinical state but the
neutral zone of a noisy continuous quantity, half of whose visits sit one
hundredth from a boundary. (4) Comparison with Simuni, Lee and von Coelln,
including the cohort-composition explanation for von Coelln's lower figure.
(5) The constructive finding — the representation is better than the label as a
description, which points at a different endpoint. (6) What this implies for
trial design: a persistent, coarse label derived from two of nine measurements
is a weak longitudinal endpoint. (7) Limitations, cross-referenced. (8)
Conclusion integrated into the final paragraph, per npj style.

**Limitations to state explicitly** (from Stage 1 §I1–I18, condensed): no
external cohort; complete-case analysis with label loss concentrated in treated
patients later in follow-up; the ratio undefined at 16.3% of labelled visits;
Indeterminate excluded from every two-class result; conditional probabilities
are conditional; both models are told when the next visit occurs; the target is
the project's own rule, not adjudicated diagnosis; bounded scales modelled as
Gaussian; AIC/BIC assume independent observations; the two newer models use
different validation schemes; multiplicity largely uncorrected; the composition
analysis is post hoc.

---

## 6. Methods

Mapped to the brief's §22.1–22.16. Everything is documented in source; the
Methods section is assembly, not discovery.

| Subsection | Source of truth | Status |
|---|---|---|
| Study design | `config/config.yaml`, Stage-1 audit Audit C | ready |
| Dataset and participants | `docs/DATA_REQUIREMENTS.md`, E-5 ledger | ready |
| Clinical variables and subtype definition | `src/ppmi_subtype/subtype.py`, `config.yaml` | ready |
| Longitudinal construction | `src/ppmi_subtype/panel.py`, `transitions.py` | ready |
| Observed transition analysis | `transitions.py`, **E-2** for intervals | ready |
| CTMC and Markov models | `src/ppmi_subtype/ctmc.py` | ready |
| CT-HMM | `src/ppmi_subtype/hmm.py` | ready — **must state the emission floor is binding for Indeterminate (0.500 is the constraint, not an estimate)** |
| Bayesian growth model | `src/ppmi_subtype/hbayes.py` | **blocked by C3** — single chain, no R̂/ESS. Either refit with ≥4 chains or demote to descriptive supplementary with the gap disclosed |
| Subscore GP and LMM | `gpr.py`, `jointlmm.py`, `mogp.py`, `timevarying.py` | ready |
| Rich clinical profile | `src/ppmi_subtype/clinical_profile.py` | ready — **fix the stale innovation-covariance docstring first** |
| Harmonised comparison | **`scripts/79`**, E-8 | ready |
| History windows and retraining | `scripts/70–74` | ready |
| Validation design | `loo.py`, `splitting.py` | ready |
| Statistical analysis | metric definitions across runs | ready — state that every interval is a patient-clustered percentile bootstrap, 2,000 draws, no bias correction |
| Software and reproducibility | `pyproject.toml`, run `summary.json` files | ready — Python 3.13.16, numpy 2.2.4, scipy 1.15.2, pandas 2.2.3; seed 42 throughout; 758 tests |

---

## 7. Back matter

| Item | Status |
|---|---|
| **Data availability** | Draftable: PPMI via LONI under the DUA, not redistributable. Needs the current PPMI acknowledgement wording (`report/CARE_AI2026_Abstract.tex` carries a `% TODO` for exactly this). |
| **Code availability** | **Problem.** The GPR modules (`gpr.py`, `gpr_loo.py`, `gpr_plots.py`, `trajectory.py`, scripts 11–16 and three test files) are untracked and were deliberately never pushed. A code-availability statement cannot promise the GPR line unless they are committed, and R3 depends on the subscore GP. **Decide before submission.** |
| **Ethics** | **ABSENT (C5).** Needs the PPMI DUA reference and the institutional position on secondary analysis. |
| **Funding** | **ABSENT (C5).** |
| **Competing interests** | **ABSENT (C5).** |
| **Author contributions** | Parv Patel, Mayank Tewatia, Dr. Sahasranand K. R. per the CARE AI abstract. Needs a CRediT statement from the authors. |
| **AI disclosure** | npj policy not verified (Nature pages behind an auth redirect). Check at submission. |

---

## 8. Supplementary

| # | Content | Status |
|---|---|---|
| S1 | Hierarchical Bayesian growth model | built; **C3 unresolved** |
| S2 | GPR kernel comparison, four cohorts, with the two superseded claims corrected | built |
| S3 | Joint subscore LMM, multi-output GP, time-varying rate | built |
| S4 | First-visit model | built |
| S5 | Full history-window, retraining and stage-wise tables | built |
| S6 | Indeterminate boundary and focal-ratio analyses | built |
| S7 | Threshold sensitivity | **O-4 not run** |
| S8 | STROBE and TRIPOD+AI checklists | **E-7 not done** |
| S9 | Validation batteries and the 758-test suite | built |

---

## 9. What must happen before drafting

**Blocking the Results section**

1. ~~**Build Figure 2, Figure 4, Figure 5(b) and Table 1.**~~ ✅ **Done
   2026-11-04**, in `manuscript/npj/`.
2. **O-4** — run the threshold-sensitivity experiment so R2's second claim
   rests on a saved run rather than my recomputation.

**Blocking Methods**

3. **C3** — multi-chain refit of the growth model, or the decision to demote it.
   Yours.
4. **I18** — relabel or recompute the five-year clinical table, and save π.
5. Fix the stale `clinical_profile.py` innovation-covariance docstring.

**Blocking submission only**

6. **C5** — ethics, funding, competing interests, CRediT, PPMI acknowledgement.
   Yours and your supervisor's.
7. **C6** — restore `data/validation/` or disclose that the 99.87% check is not
   currently reproducible.
8. **Code availability** — commit the GPR modules or scope the statement.
9. **E-7** — STROBE and TRIPOD+AI checklists.

**Must never be used**

10. The "+0.005 / +0.021 / +0.047" HMM advantage by history length. No code path,
    no saved output; flagged inline in the README.

**Recommended drafting order** (Stage 6 onward): Methods §22.2–22.5 → Methods
§22.6–22.16 → R1–R2 → R3–R5 → R6–R8 → Introduction → Discussion → Abstract and
title → back matter → submission audit.
