# Experiment and report audit

**Project** Towards Longitudinal Prediction of Parkinson's Disease Motor Subtypes
**Repository** `ppmi-subtype-progression`, at commit `8268ea5`
**Date of audit** 2026-11-04
**Scope** Documentation and evidence audit. No experiment was re-run, no existing
report was modified, renamed or deleted, and nothing was written outside
`research_audit/`.

---

## A. Repository summary

| Location | What is there |
|---|---|
| `report/` | 28 files: **12 documents** (10 LaTeX, 1 Markdown script, 1 HTML deck), their compiled PDFs, one sources file and two figure scripts |
| `reports/` | 1 file: **1 document**, a 45-page PDF with no `.tex` beside it. Its source is `docs/GPR_onwards_explanatory_report.md` |
| `experiments/` | 81 directories, **76 timestamped runs**; 23 carry a machine-written `report.txt` |
| `outputs/` | 14 result folders, read-only, from the pre-October pipeline; frozen panel md5 `fb038ddbdc19ab3368e13940f398e7ba` |
| `docs/` | 119 Markdown documents, 77 of them the pre-registered `cluster_phase` stage-2 record |
| `scripts/` | 90 numbered analysis scripts (`00` … `86`) plus cluster helpers |
| `src/ppmi_subtype/` | 50 modules |
| `manuscript/`, `paper/` | Manuscript assembly; **not** treated as evidence in this audit |

**The two folders are not duplicates and neither is obsolete.** They document
different stages and were written for different purposes:

* `report/` is the **academic progress record to early October 2026** — weekly
  reports, two full project reports, a code-to-output walkthrough, a model
  comparison and a conference abstract. It ends at the hierarchical Bayesian
  comparison and the follow-up analyses.
* `reports/` holds **one document covering the GPR stage onwards** (Gaussian
  process regression, the move from the ratio to subscores, the joint LMM, the
  multi-output GP, time-varying rates and the clusterability audit), dated
  9 October 2026. It is the most carefully evidenced document in the
  repository; it states that every number was checked against a saved output
  file and labels those that could not be.

**The single most important finding of this audit is a coverage gap.** The GPR
document closes by naming "Direction 4" as the next research question.
Direction 4 is the entire probabilistic TD–PIGD prediction programme — 21
experiments run between 20 October and 4 November 2026, and the source of
every result the current manuscript rests on. **Neither report folder contains
a single document about it.** That work is documented only in `docs/` and in
the runs' own `report.txt` files.

---

## B. Existing report inventory

Full detail, field by field, is in `research_audit/report_inventory.csv`.

| # | Document | Folder | Documents | Completeness | Problems |
|---|---|---|---|---|---|
| 1 | `Week3-4_Report` | `report/` | Observed transition analysis | Complete | Wilson intervals ignore repeated visits per patient; superseded for uncertainty by E-2 |
| 2 | `Week5-7_Report` | `report/` | DTMC / CTMC / CT-HMM held-out comparison | Complete | Single held-out split (1,002 pairs); the headline CT-HMM figure is now the 934-fold LOPO run |
| 3 | `Week5-7_Walkthrough` | `report/` | Narrative companion to #2 | Complete for its purpose | No independent results; not a result source |
| 4 | `Model_Comparison_Report` | `report/` | Hierarchical Bayesian growth model vs CT-HMM | Complete | — |
| 5 | `Project_Report` | `report/` | Whole project to August 2026 | Complete for its date | Out of date; a reader cannot tell what the current best model is |
| 6 | `Subtype_Progression_Report` | `report/` | Whole project, longest version | Complete for its date | Same coverage gap; no AUC, Brier or calibration for the prediction task as later defined |
| 7 | `Pipeline_Walkthrough` | `report/` | Code→output trace, stage 0 to statistics | Complete, unusually well evidenced | Stops before `scripts/60`; silent on everything after |
| 8 | `Followup_Analysis_Report` | `report/` | Post-review patient-level analyses | Complete | Contains an explicit population correction — quote only the corrected section |
| 9 | `BTP_Post_Midterm_Report` | `report/` | GPR → clustering, short form | Complete | Low resolution version of #12 |
| 10 | `CARE_AI2026_Abstract` | `report/` | Conference abstract | Complete as an abstract | Predates E-2, E-8 and O-4; re-check every number before resubmission |
| 11 | `Midterm_Script` / `Midterm_Presentation` | `report/` | Midterm talk | Complete as a presentation | Not a result source |
| 12 | `GPR_Onwards_Explanatory_Report.pdf` | `reports/` | GPR, joint LMM, MOGP, time-varying, clusterability | Complete, best-evidenced document in the repository | Dated 9 Oct 2026; stops exactly where the current research programme starts |

### Cross-checks performed on these documents

Numbers were re-read from the underlying files rather than taken from the prose.

| Claim | Stated in | Source file | Found | Verdict |
|---|---|---|---|---|
| 19,450 → 5,774 → 4,441 → 3,361; 934 patients, 146 excluded | #7, #6 | `experiments/…e5_participant_flow/flow_ledger.csv`, `outputs/panel_summary.txt` | identical | ✅ |
| Label agreement 100.00% (2-class) and 99.87% (3-class) | #7 | `outputs/validation.txt` | 1.0000 (n=4,431); 0.9987 (n=1,597) | ✅ |
| TD→TD 0.808 | #1, #6 | `experiments/…e2_…/transition_probabilities.csv` | 0.8080 | ✅ |
| 35.2% of changes revert | #1, #6 | same, `other_quantities.csv` | 0.35215 | ✅ |
| 35.3% naive vs 15.3% sustained conversion at 5 y | #1, #6 | same | 0.35252 / 0.15327 | ✅ |
| Markov property rejected in all three states | #2 | `outputs/model/markov_property_test.csv` | p = 7.9e-33 / 4.1e-03 / 1.7e-12 | ✅ |
| HMM 0.7695 > DTMC 0.7675 > CTMC 0.7655 held-out | #2 | `outputs/model/heldout_performance.csv` | identical, n = 1,002 | ✅ |
| HMM beats the growth model on all three metrics | #4 | `outputs/hbayes/head_to_head.csv` | −0.6559/0.7612/0.3573 vs −0.6814/0.7370/0.3786, n = 825 | ✅ |
| GP beats persistence on the ratio, −0.0707 [−0.0979, −0.0451] | #12 | `outputs/step3_gpr/paired_comparisons.csv` | −0.07069 [−0.09787, −0.04511] | ✅ |
| No subscore GP beats persistence on PIGD | #12 | `outputs/joint_subscore_gp/paired.csv` | all five PIGD deltas positive, all exclude zero | ✅ |
| Slope reliability 0.270 (tremor) / 0.531 (PIGD) | #12 | `outputs/clustering_audit/random_slope_reliability.csv` | per-patient column present, medians consistent | ✅ |

**No numerical error was found in any existing report in either folder.** Every
spot-check reproduced. The problems listed above are problems of coverage,
staleness and uncertainty method — not of arithmetic.

---

## C. Experiment inventory

47 experiment identifiers, 222 result rows, in
`research_audit/master_experiment_results.csv`. Summarised by area:

### C.1 Dataset construction and validation

| ID | Experiment | Script | Results | Report | Status |
|---|---|---|---|---|---|
| D-01 | Panel construction and subtype classification | `00`, `01` | `outputs/panel_summary.txt`, `outputs/validation.txt` | `Pipeline_Walkthrough`, `Subtype_Progression_Report` | **A** |
| D-02 | E-5 participant flow | `78` | `experiments/…e5_participant_flow/` | `docs/manuscript/02` (shared with E-1, E-2) | **B** → report generated |
| D-03 | O-4 threshold sensitivity | `85` | `experiments/20261104T002308Z_o4_…/` | none | **C** → report generated |

### C.2 Observed subtype progression

| ID | Experiment | Script | Results | Report | Status |
|---|---|---|---|---|---|
| T-01 | Transition analysis | `02` | `outputs/transitions/` | `Week3-4_Report` | **A** |
| T-02 | E-2 patient-clustered bootstrap | `77` | `experiments/…e2_…/` | `docs/manuscript/02` (shared) | **B** → report generated |
| T-03 | Patient-level progression views | `08` | `outputs/patient_level/` | `Followup_Analysis_Report` | **A** |

### C.3 Markov, continuous-time Markov and hidden Markov models

| ID | Experiment | Script | Results | Report | Status |
|---|---|---|---|---|---|
| M-01 | Markov-property test | `03` | `outputs/model/markov_property_test.csv` | `Week5-7_Report` | **A** |
| M-02 | Held-out DTMC / CTMC / CT-HMM | `03` | `outputs/model/heldout_performance.csv` | `Week5-7_Report` | **A** |
| M-03 | 934-fold LOPO of the 3-state CT-HMM | `09` | `outputs/loo/` | `Subtype_Progression_Report` | **A** |
| M-04 | 5-hidden-state LOO shards | `10`, `merge_5hidden_loo` | `outputs/loo_5hidden/` (predictions only) | none | **C**, see §E |
| M-05 | HMM stability and emission floor | `23`, `24` | `experiments/…s4_hmm_stability/`, `…s4_hmm_cv/` | `BTP_Post_Midterm_Report` | **A** |
| M-06 | New-patient baseline model | `05` | `outputs/newpatient/` | `Project_Report` | **A** |

### C.4 Hierarchical Bayesian growth model

| ID | Experiment | Script | Results | Report | Status |
|---|---|---|---|---|---|
| B-01 | Growth model vs CT-HMM | `04` | `outputs/hbayes/` | `Model_Comparison_Report` | **A** |
| B-02 | Leave-future-visits-out validation | `07` | `outputs/validation/` | `Project_Report` | **A** |
| B-03 | Student-t slope / residual variants | `35`–`37` | `experiments/20261015T…student_t_*/` | `docs/cluster_phase/07`, `08` (+ frozen specs) | **A** |

### C.5 Gaussian processes and continuous-outcome modelling

| ID | Experiment | Script | Results | Report | Status |
|---|---|---|---|---|---|
| G-01 | GPR on the TD/PIGD ratio | `11`–`15` | `outputs/step3_gpr/` | `reports/GPR_…` Part B | **A** |
| G-02 | Joint subscore LMM | `17` | `outputs/joint_subscore/` | Part D | **A** |
| G-03 | Multi-output GP on the subscores | `18` | `outputs/joint_subscore_gp/` | Part E | **A** |
| G-04 | Time-varying progression rate | `19` | `outputs/time_varying/` | Part F | **A** |
| G-05 | Clusterability audit | `20`, `26`, `29` | `outputs/clustering_audit/` | Part G | **A** |

### C.6 Measurement-model stage (D-1, stage 2, GRM)

| ID | Experiment | Script | Results | Report | Status |
|---|---|---|---|---|---|
| S-01 | D-1 label posterior predictive check | `41`–`43` | `experiments/20261016T…d1_*/` | `docs/cluster_phase/13`–`15` | **A** |
| S-02 | Stage-2 integer-score model (M2 / M2-A1 / rev3) | `44`–`59` | `experiments/20261016–19T…stage2_*/` | `docs/cluster_phase/16`–`53`, plans sha256-frozen | **A** |
| S-03 | GRM and GRM-OU predictive checks | `32`, `33` | `experiments/20261014T…grm_*/` | `docs/cluster_phase/06` | **A** |
| S-04 | PIGD measurement audit and models | `30`, `31` | `experiments/20261014T14…pigd_measurement_*/` | **none located** | **D**, see §F |
| S-05 | Bayesian slope clustering | `34` | `experiments/20261015T171236Z_…/` | `reports/GPR_…` Part G (conclusion only) | **B** |

### C.7 Probabilistic TD–PIGD prediction — the §7 sub-inventory

This is the programme the two report folders do not cover.

| ID | Experiment | Script | Validation design | Cases | Report | Status |
|---|---|---|---|---|---|---|
| P-01 | Next-visit TD/PIGD from subtype history, 7 models | `60` | LOPO, 934 folds | 3,029 of 3,361 | `docs/next_visit_prediction/01` | **A** |
| P-02 | Clinical latent-state count selection | `61` | inner-fold log-density, 1-SE rule, 280 dev patients | — | `docs/latent_clinical_states/01` | **A** |
| P-03 | Clinical-state CT-HMM, k = 6 and k = 8 | `62` | LOPO on 654 evaluation patients | 2,091 | `docs/latent_clinical_states/01` | **A** (k8 variant **B**) |
| P-05 | Direct logistic on the 10 measurements | `63` | same 654-patient split | 2,091 | `docs/latent_clinical_states/01` | **A** |
| P-06 | Rich clinical profile, 9-domain local linear trend | `64` | 5-fold patient CV, 4 repeats | 3,021 / 3,020 | `docs/latent_clinical_states/02`, `03` | **B** — provenance defect, §G-3 |
| P-07 | AIC / BIC, information criteria | `65` | full-cohort maximum likelihood, criteria only | — | `docs/model_comparison/01` | **A** |
| P-07m | Matched held-out two-class comparison | `65` | intersection of the CT-HMM's LOPO rows and the profile's CV rows | 3,020 | `docs/model_comparison/01` | **A** |
| P-08 | ROC / AUC on identical cases | `66` | shared 3,020-case set, 2,000-draw patient bootstrap | 3,020 | `docs/model_comparison/02` | **A** |
| P-09 | Indeterminate boundary | `67` | descriptive, matched cases | 287 of 332 | `docs/model_comparison/03` | **A** |
| P-10 | Indeterminate ratio scrutiny | `68` | reproduction check + patient-clustered intervals | 287 | `docs/model_comparison/04` | **A** |
| P-11 | Stage-wise evaluation | `69` | reuse of held-out predictions, one stage per case | 3,017 | `docs/stage_wise_evaluation/` | **A** |
| P-12 | CT-HMM history window | `70` | LOPO, 934 folds, 5 windows | 3,029 / 734 common | `docs/history_window/HISTORY_WINDOW` | **A** |
| P-13 | Rich-profile history window | `71` | 5-fold CV, **one** repeat | 3,021 / 1,489 common | same doc, §6 | **B** |
| P-14 | Retrained two-visit CT-HMM | `72` | LOPO, 6 conditions | 3,029 / 2,192 common | `docs/history_window/RETRAINED_TWO_VISIT` | **A** |
| P-15 | Experiment 6A, profile prediction history | `73` | 5-fold CV, one repeat, parameters fixed across conditions | 3,021 / 734 common | `docs/history_window/EXPERIMENT_6A` | **A** |
| P-16 | Experiment 6B, profile two-visit retraining | `74` | 5-fold CV, 2×2 + sensitivity + persistence | 2,184 | `docs/history_window/EXPERIMENT_6B` | **A** |
| P-17 | Patient-level accuracy and composition audit | `75` | re-aggregation of 6A/6B predictions | per condition | `docs/history_window/PATIENT_LEVEL_ACCURACY_AUDIT` | **A**, exploratory |
| P-18 | E-8 harmonised multi-family comparison | `79` | one case list, one argmax rule, 10,000 MC draws | 3,361 | `docs/manuscript/05` | **B** → report generated |
| P-19 | A-2 clinical-profile CV rerun at HEAD | `64`, `83` | identical to P-06 | 3,353 | none | **C** → report generated |
| P-20 | A-2′ outcome-table round-trip | `84` | artifact verification | 8 contrasts | none | **C** → report generated |
| P-21 | A-1 outcome-contrast reference orientation | `82` | paired AUC bootstrap | 736–818 patients | none | **C** → report generated |
| P-22 | A-3 profile CV with the paired baseline | `64 --baseline` | 5-fold patient CV, 4 repeats | 3,020 | none | **C** → folded into the A-2 report |
| E-1 | Lag-1 residual autocorrelation resolution | `76` | 2,000 null draws + patient bootstrap | 2,050 residual pairs | `docs/manuscript/02` (shared) | **B** → report generated |

#### 7.1 Primary model comparison — what exists

Reports exist for the **CT-HMM** (P-01, `docs/next_visit_prediction/01`), the
**rich clinical profile** (P-06, `docs/latent_clinical_states/02` and `03`) and
the **persistence baseline**, which is carried as a reference arm in every one
of the runs above rather than documented separately. Other implemented
baselines — discrete Markov, interval-aware CTMC, two direct logistic models,
a marginal two-class prior, last-observation-carried-forward and a population
mean trajectory — appear in P-01, P-05 and P-06 but have no document of their
own; they do not need one, because they are comparison arms, not experiments.

**The validation designs are not interchangeable and must never be pooled:**

| Model | Fitting / validation | Eligible cases |
|---|---|---|
| CT-HMM and all subtype-history models | leave-one-patient-out, 934 folds | 3,029 |
| Rich clinical profile | 5-fold patient CV, 4 repeats | 3,021 |
| Matched comparison of the two | intersection of the two runs' rows | 3,020 |
| Stage-wise evaluation | same predictions, one stage per case | 3,017 |
| Harmonised E-8 benchmark | one case list, argmax, all three classes | 3,361 |

#### 7.2 Probability quality

Evaluated, with files: **AUC** (P-08, `auc_metrics.csv`; also P-12 to P-16),
**Brier** (every prediction run, `brier_two_class` or `brier_multiclass`),
**log loss** (same), **calibration** (P-01 writes a `calibration_*.csv` per
model; `outputs/loo/loo_calibration.csv`, `outputs/model/calibration_hmm.csv`),
**accuracy** and **balanced accuracy** (P-12 onwards).

Two cautions that the runs state themselves and that any write-up must repeat:

* The persistence rule emits a degenerate 0/1 probability. Its log loss
  (≈ 4.19, ≈ 4.20, up to 27.6 in thin stage-wise cells) is a **clipped** value
  and is not comparable with a model's log loss. The runs flag this with
  `rule_based_probabilities` / `degenerate_probabilities`.
* E-8's `brier_multiclass` is a three-class Brier score. It is a different
  quantity from the `brier_two_class` reported in P-01 and P-06 and the two
  must not be placed in one column.

#### 7.3 The Indeterminate category

Defined as the open band 0.90 < ratio < 1.15 on the tremor/PIGD ratio
(`config/config.yaml`). Its treatment differs by experiment, deliberately:

* **Excluded as a target** in every two-class evaluation (P-01, P-05 to P-16).
  332 Indeterminate targets are dropped from the 3,361 transitions; the
  remainder is the 3,029 / 3,021 / 3,020 / 3,017 set above.
* **Retained as history** everywhere — a patient's Indeterminate visits are
  still part of the input sequence.
* **Analysed separately** in P-09 and P-10, on the 287 Indeterminate-target
  cases that have a defined ratio (45 of 332 have a zero PIGD subscore, so no
  ratio and no boundary distance exist).
* **Scored as a class** only in E-8 (P-18), which is the single three-class
  evaluation in the programme.

The grouping in P-09, P-10 and P-17 uses the **true target labels**, not
predicted ones. That makes those analyses descriptive; P-17 is explicitly
exploratory and was not pre-specified.

#### 7.4 History windows

Four separate experiments, with different cohorts — this is the most common
place to draw a wrong conclusion.

| Experiment | Model | Windows | Cohort behaviour |
|---|---|---|---|
| P-12 | CT-HMM | persistence, last 1, 2, 3, 5, full | "all eligible" n falls 3,029 → 2,192 → 1,495 → 734 as the window lengthens |
| P-13 | profile | last 2, 3, full (+ LEDD sensitivity) | 3,021 → 1,489 |
| P-15 (6A) | profile | persistence, last 1, 2, 3, 5, full | 3,021 → 734 |
| P-14 (CT-HMM), P-16 (6B) | both | retraining conditions | fixed per experiment |

Every one of these experiments also reports a **common-targets** cohort — the
subset of cases every window can score (734 for P-12 and P-15, 1,489 for
P-13). **Only the common-targets rows are a like-for-like comparison.** The
"all eligible" rows answer a different question: how well can the model do on
everyone it can reach with that window.

#### 7.5 Short-window retraining

Two experiments, both with a pre-specified primary comparison:

* **P-14** (CT-HMM): six conditions crossing training window with prediction
  window, plus a non-overlapping-window control. Primary: *window-trained,
  last 2* vs *full-trained, full history*.
* **P-16** (profile): a 2 × 2 of training objective × prediction history, plus
  disjoint-window sensitivity arms and persistence. Primary: *D: window-trained,
  last 2* vs *B: full-trained, last 2* — prediction history identical, so only
  the training objective differs.

Both were measured under the experiment's own single validation design, held
fixed across conditions. **Neither supports an improvement from retraining.**
In P-16 all four cells of the 2 × 2 fall within 0.003 accuracy of one another.

#### 7.6 Patient-level composition audit

P-17 re-aggregates the 6A and 6B held-out predictions to the patient, and
groups patients by the composition of their own target labels (all-TD,
majority-TD, evenly split, majority-PIGD, all-PIGD). Group sizes under full
history: 510 / 105 / 60 / 61 / 162 patients. Mean per-patient accuracy ranges
from 0.944 (all-TD) to 0.468 (evenly split). Uncertainty is reported through
paired per-patient comparisons in `paired_patient_comparisons.csv`.

**The groups are defined by the outcome being predicted.** A patient whose
targets are all TD is, by construction, a patient a persistence rule cannot get
wrong. This analysis therefore describes where the errors fall; it does not
show that subtype composition *causes* the difference, and the audit records
it as exploratory.

---

## D. Missing reports (Status C)

Detail and justification in `research_audit/missing_reports.md`. Seven reports
were generated, in `research_audit/generated_reports/`:

| New report | Documents | Why it was needed |
|---|---|---|
| `harmonised_family_comparison_report.md` | P-18 (E-8) | The central benchmark of the current manuscript. Its only write-up is a 12 KB manuscript-support note |
| `participant_flow_report.md` | D-02 (E-5) | Shared a document with two unrelated analyses |
| `transition_patient_bootstrap_report.md` | T-02 (E-2) | Same; and it supersedes the Wilson intervals in `Week3-4_Report` |
| `lag1_residual_resolution_report.md` | E-1 | Same; and it is the only record of a live contradiction in `outputs/` |
| `threshold_sensitivity_report.md` | D-03 (O-4) | No `report.txt`, no document anywhere |
| `outcome_contrast_orientation_report.md` | P-21 (A-1) | No `report.txt`, no document; and its own script marks its output superseded |
| `clinical_profile_reproduction_report.md` | P-19, P-20 (A-2, A-2′) | No document; records a provenance defect in a run the manuscript depends on |

**Not generated, deliberately:** M-04 (5-hidden-state LOO) — the predictions
exist but no metrics were ever computed into `outputs/`, so a results report
would have to compute new results; and S-04 (PIGD measurement audit) — run
directories exist but their completion could not be verified (§F).

---

## E. Incomplete reports (Status B)

Detail in `research_audit/incomplete_reports.md`.

| Experiment | Document | What is missing |
|---|---|---|
| D-02, T-02, E-1 | `docs/manuscript/02_ESSENTIAL_ANALYSES_E1_E2_E5` | Three independent analyses in one manuscript-facing document; no per-experiment configuration, limitations or reproducibility section |
| P-18 | `docs/manuscript/05_HARMONISED_FAMILY_COMPARISON` | No cohort section, no per-family validation description, no limitations beyond one short list; the 14-specification table is given without the fitting scheme of each family |
| P-06 | `docs/latent_clinical_states/03_RICH_PROFILE_RESULTS` | Complete as science; silent on the provenance defect (§G-3), which was only discovered two weeks later |
| P-13 | `docs/history_window/HISTORY_WINDOW` §6 | A secondary section of another experiment's report; the one-repeat CV design is easy to miss and makes its numbers non-interchangeable with P-06's |
| P-03 (k8) | `docs/latent_clinical_states/01` | The k = 8 run is reported as a sensitivity arm; its own selection rationale (BIC and the arithmetic maximum both choose 8) deserves its own statement |
| S-05 | `reports/GPR_…` Part G | Only the conclusion is recorded; the run's own outputs are not described |
| M-04 | — | No report at all, and no metrics file to write one from |

---

## F. Unverified experiments (Status D) and planned work (Status E)

| Experiment | Evidence present | Evidence needed |
|---|---|---|
| **S-04** PIGD measurement audit (`scripts/30`, `31`) | Two run directories, 29 and 13 files, from 2026-10-14 | No `report.txt`, and no `docs/` file names either run. Needed: confirmation of which document, if any, consumed them, or a statement that the analysis was abandoned. **A rerun is not required** — the outputs exist; only their interpretation is missing |
| **Smoke and pilot runs** (`_smoke/`, `*_smoke`, `*_pilot`) | 705 files under `experiments/_smoke/` | These are deliberately throwaway. They need no report and should not appear in any results table |
| **`threshold_narrow_0.91_1.13/`** | A full alternative panel, no manifest of its own | It is the input to O-4, not an experiment. Its provenance is the `config_narrow_0.91_1.13.yaml` run of `scripts/01`; that invocation is not itself recorded in a run directory |
| **Status E (planned, not run)** | — | None found. Every experiment named in `docs/` as planned has a corresponding run directory. The `docs/cluster_phase` documents marked `PROPOSED` were all either frozen and executed or explicitly superseded by a later revision |

---

## G. Contradictions and discrepancies

Recorded, not resolved.

### G-1. A stale sentence in a read-only output contradicts its own script

`outputs/exploration/report.txt` contains the sentence *"Nothing here argues
for the state-space alternative."* Re-running the committed
`scripts/06_ratio_exploration.py` unmodified produces the **opposite** reading —
that the lag-1 residual structure is a signature of accumulating process noise
and favours the state-space form. E-1 searched all 45 commits, whitespace
insensitively, and found that the committed script **cannot produce the stale
sentence**. The arithmetic was never wrong: the observed lag-1 correlation
−0.2195 reproduces exactly; only the conclusion drawn from it was stale.

* Report's value: the prose conclusion in `outputs/exploration/report.txt`
* Source value: `experiments/…e1_lag1_residual_resolution/rerun_outputs/exploration/report.txt`, 13 differing lines, full diff in `report_diff.txt`
* Explainable: yes — a stale artefact from before a script revision
* Correction needed: `outputs/` is read-only by project rule, so the file
  cannot be rewritten. **Nothing may be quoted from that paragraph.** The
  authoritative statement is E-1's.

### G-2. The IEEE manuscript reads a file its own script marks superseded

`paper/ieee/make_assets.py:744` reads
`experiments/20261103T212122Z_a1_outcome_contrast_orientation/corrected_contrasts.csv`.
`scripts/82_outcome_contrast_orientation_fix.py` opens with
**"SUPERSEDED — do not use its numbers"**: it recomputed the contrasts from a
copy of `outcome_table.csv` that had been written with `.round(4)`, which is
destructive for rate coordinates of order 1e-3. The authoritative table is
`experiments/20261103T220147Z_a2p_outcome_table_fullprec/h3_component_vs_subtype.csv`
(identical to the A-2 rerun's).

| Quantity | A-1 file (used by the paper) | Authoritative | Difference |
|---|---|---|---|
| MCI, rate contrast | 0.0428 | **0.0448** | −0.0020 |
| MCI, best rate AUC | 0.5922 | **0.5942** | −0.0020 |
| cognitive_dx, rate contrast | 0.0356 | 0.0355 | +0.0001 |
| motor_complications, rate contrast | 0.0334 | 0.0333 | +0.0001 |
| H&Y ≥ 3, rate contrast CI high | 0.1185 | 0.1186 | −0.0001 |
| **All four *level* contrasts** | 0.1052 / 0.1605 / 0.0272 / 0.1194 | identical | **0.0000** |

* Consequence in the manuscript: `paper/ieee/main.tex:771` prints the rate
  differences as "+0.053, **+0.043**, +0.033 and +0.036"; the authoritative
  value for the second is **+0.045**. Figure 5(b)'s rate bar for MCI is drawn
  at 0.5922 instead of 0.5942.
* The paper's substantive claims rest on the **level** contrasts, which are
  unaffected.
* Correction needed: **yes.**
* **RESOLVED, 2026-11-04, after the audit was written.**
  `paper/ieee/make_assets.py` now reads
  `PROF / "h3_component_vs_subtype.csv"`; the superseded `A1` constant is gone.
  `paper/ieee/main.tex:771` now prints `+0.045`, and the regenerated Table VI
  shows MCI rate AUC 0.594 (was 0.592), Δ +0.045 (−0.005, +0.096) (was +0.043
  (−0.008, +0.094)) and motor-complications rate CI low −0.026 (was −0.025).
  The four level rows are unchanged, as predicted. The paper rebuilds with no
  overfull boxes and 91 of 91 number checks passing.

### G-3. A manuscript-critical run executed uncommitted code

The 2026-10-20 rich-profile CV run (P-06) records commit `3a860ca`, at which
`src/ppmi_subtype/clinical_profile.py` **did not exist** — it was first
committed five days later in `19bf8aa`. Nothing that run saved is reproducible
from any committed state.

**The rerun at HEAD (P-19) reproduces it exactly.** Every two-class figure is
identical to four decimal places (accuracy 0.8441 / 0.8405 / 0.8401 / 0.8249 /
0.6912; Brier and log loss likewise). The defect is one of provenance, not of
result.

**One file did not carry over.** The October run wrote
`derived_label_vs_subtype_history.csv` — the table holding the profile-versus-
CT-HMM two-class contrast, and the table `heldout_matched.csv` in P-07 is a
copy of it. The A-2 rerun directory does **not** contain that file.

* **RESOLVED, 2026-11-04, after the audit was written.** The cause was a
  missing command-line argument, not a code change:
  `scripts/64_clinical_profile_cv.py:495` writes that table only inside
  `if args.baseline:`, and `--baseline` defaults to `None`. The October run
  was given it; neither rerun was.
* Re-run as **P-22** (`experiments/20261104T085320Z_a3_profile_cv_with_baseline`),
  submitted through `scripts/cluster/submit.sh` at commit `8268ea5`, with
  neither `scripts/64` nor `src/ppmi_subtype/clinical_profile.py` appearing in
  the recorded working diff — so both ran exactly as committed.
* `derived_label_vs_subtype_history.csv` and `derived_label_contrasts.csv` are
  both restored and reproduce the October versions **exactly** (maximum
  absolute difference 0.0 on every numeric column of both files). Six other
  shared outputs also match the A-2 rerun to 0.0, which rules out any effect
  of the 60-worker setting used here.
* The two-class contrast now has an artifact produced by committed code, with
  `git_commit.txt`, `git_diff.patch`, `untracked_source.tar.gz` and
  `pip_freeze.txt` beside it.

### G-4. Four different "eligible case" counts for the same task

3,029 (CT-HMM LOPO), 3,021 (profile CV), 3,020 (matched intersection), 3,017
(stage-wise, after dropping 3 stage-disagreement cases), and 3,361 (E-8, which
scores all three classes). These are **not** errors — each is correct for its
run — but they are routinely conflated. In particular the same CT-HMM reports
accuracy **0.8369** on 3,029 cases and **0.8371** on 3,020. Any sentence of the
form "*n* of 3,361 becomes *n* of 3,029" is arithmetic that no run performed.

### G-5. The best model on accuracy is the worst on macro-F1, and vice versa

In E-8, persistence on the recorded label has the **lowest** three-class
accuracy of the fourteen specifications (0.7227) and the **highest** macro-F1
(0.5559 against the best model's 0.5074). The reason is visible in the same
table: persistence is the only specification that ever predicts Indeterminate
(334 predictions, recall 0.193); every fitted model predicts it essentially
never. This is a real tension in the results, not a defect, and it must be
reported rather than smoothed over by choosing the metric that favours the
models.

### G-6. O-4 was run three times within four minutes

`20261104T002141Z`, `20261104T002247Z` and `20261104T002308Z`. The first lacks
`attainable_ratios.csv` and its figure. The last is authoritative and is the
one the manuscript reads; the other two should be treated as superseded
attempts, and the three directories should not be read as three analyses.

### G-7. Degenerate probabilities make persistence's log loss incomparable

Reported above in §7.2. Flagged by the runs themselves; it is listed here
because a reader scanning `master_experiment_results.csv` will see 4.19 beside
0.396 and may draw the wrong conclusion.

---

## H. Recommended final set of standalone experiment reports

The repository does not need a report per run — 76 runs include smoke tests,
pilots and shards. It needs one report per *question answered*. The
recommended set, after this audit:

**Already adequate, keep as they are (Status A):**
`report/Week3-4_Report`, `report/Week5-7_Report`, `report/Model_Comparison_Report`,
`report/Pipeline_Walkthrough`, `report/Followup_Analysis_Report`,
`report/Subtype_Progression_Report`, `reports/GPR_Onwards_Explanatory_Report`,
and the nine `docs/` experiment write-ups for P-01, P-02/P-03/P-05, P-06,
P-07, P-08, P-09, P-10, P-11, P-12, P-14, P-15, P-16, P-17.

**Added by this audit (Status C → documented):** the seven reports in
`research_audit/generated_reports/`.

**Still to be written, by someone with knowledge this audit does not have:**

1. A report for **S-04** (PIGD measurement audit), or a note recording that the
   analysis was abandoned.
2. A metrics summary for **M-04** (5-hidden-state LOO), which requires
   computing metrics from the saved predictions — a new computation, outside
   the scope of a documentation audit.
3. A short **provenance note** for the `config_narrow_0.91_1.13` panel build,
   so that O-4's input has a run directory like every other artefact.

**Structural recommendation.** The two folders have drifted into different
roles. Rather than merging them, label them: `report/` as the academic progress
record, `reports/` as the explanatory long-form record, `docs/` as the working
experiment record, and `research_audit/` as the index that ties all three to
the runs. The one change worth making is to add, to each report folder, a
one-page pointer to `research_audit/experiment_report_audit.md`, so that a
reader who opens either folder learns immediately that the prediction
programme is documented elsewhere.
