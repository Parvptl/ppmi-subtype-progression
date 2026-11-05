# Missing reports (Status C)

Experiments whose completion is verifiable from saved outputs but which had no
adequate standalone report anywhere in the repository. Each is listed with the
evidence that it ran, what was missing, and the report generated for it.

Audit date 2026-11-04, commit `8268ea5`.

---

## 1. O-4 — threshold sensitivity of the subtype label  (D-03)

**Evidence it ran.** Three run directories on 2026-11-04
(`…002141Z`, `…002247Z`, `…002308Z_o4_threshold_sensitivity`), each with
`summary.csv`, `per_class_movement.csv`, `per_patient_movement.csv`,
`label_crosstab.csv`, `agreement_vs_ppmi.csv` and `manifest.json`; the last two
also carry `attainable_ratios.csv` and its figure. The comparison panel
`experiments/threshold_narrow_0.91_1.13/visit_panel.csv` exists, built from
`config/config_narrow_0.91_1.13.yaml`.

**What was missing.** `scripts/85_threshold_sensitivity.py` writes no
`report.txt`, and no document in `docs/`, `report/` or `reports/` describes the
analysis. It is referred to in passing in
`docs/manuscript/12_SUBMISSION_AUDIT` and `13_FIRST_DRAFT_REVIEW` but never
reported.

**Why a report is required.** This is the only experiment that quantifies how
fragile the project's dependent variable is. 298 of 4,441 labelled visits
(6.71%) change class under a threshold perturbation smaller than the spacing
between attainable ratio values, and 235 of 1,080 participants (21.76%) are
affected. The manuscript's central argument rests on it.

→ `generated_reports/threshold_sensitivity_report.md`

---

## 2. A-1 — outcome-contrast reference orientation  (P-21)

**Evidence it ran.** `experiments/20261103T212122Z_a1_outcome_contrast_orientation/`
with `before_after.csv`, `corrected_contrasts.csv`, `drift_vs_october_run.csv`
and `manifest.json`.

**What was missing.** No `report.txt`, no document. Only a mention in
`docs/manuscript/12_SUBMISSION_AUDIT`.

**Why a report is required.** The run fixed a real asymmetry — the comparator
component was given a one-time orientation flip while the subtype-label
reference was not, which inflates the reported difference on any outcome whose
label AUC falls below chance. It also, unintentionally, produced slightly wrong
rate-coordinate numbers from a lossy saved table, and those numbers are the
ones the current IEEE manuscript reads. Both facts needed recording in one
place.

→ `generated_reports/outcome_contrast_orientation_report.md`

---

## 3. A-2 and A-2′ — clinical-profile reproduction and precision  (P-19, P-20)

**Evidence it ran.** `experiments/20261103T213859Z_a2_clinical_profile_cv_rerun/`
(22 files, with `report.txt`) and
`experiments/20261103T220147Z_a2p_outcome_table_fullprec/` (22 files, with
`report.txt`); the comparison tools are `scripts/83_compare_clinical_profile_runs.py`
and `scripts/84_verify_outcome_table_roundtrip.py`.

**What was missing.** Both runs have a machine-written `report.txt` describing
the *model*, but neither has a document describing the *question* — whether the
October run reproduces, and whether its saved artifacts can be reused. That
question is recorded only inside `docs/manuscript/12_SUBMISSION_AUDIT`, a
document about the manuscript rather than about the experiment.

**Why a report is required.** The October run executed code that was not yet
committed. Anyone auditing this project will find that and need to know, in one
place, (a) that a rerun at HEAD reproduces every number exactly, and (b) that
one file from the October run was never regenerated.

→ `generated_reports/clinical_profile_reproduction_report.md`

---

## 4. E-8 — harmonised multi-family comparison  (P-18)

**Evidence it ran.** `experiments/20261103T202632Z_e8_harmonised_family_comparison/`
with `report.txt`, `harmonised_metrics.csv` (14 specifications),
`margin_decomposition.csv`, `paired_contrasts.csv`, `monte_carlo_error.csv`,
`model_inventory.csv`, `versus_committed.csv` and `validation_checks.csv`.

**What was missing.** `docs/manuscript/05_HARMONISED_FAMILY_COMPARISON` is a
12 KB manuscript-support note — it presents the table and the margin
decomposition but has no cohort section, no per-family description of how each
specification was fitted, no reproducibility section and no source-evidence
list.

**Why a report is required.** This is the central result of the current
manuscript: nine specifications across five model families within 1.22
accuracy points of one another, and 94.3% of the best model's margin arising on
9.9% of the cases. A result carrying that much weight needs a standalone record
that states exactly what was harmonised and what was not.

→ `generated_reports/harmonised_family_comparison_report.md`

---

## 5. E-5 — participant flow  (D-02)

**Evidence it ran.** `experiments/20261103T192414Z_e5_participant_flow/` with
`report.txt`, `flow_ledger.csv`, `unlabelled_diagnosis.csv`,
`who_is_unlabelled.csv`, `excluded_by_subgroup.csv`, `excluded_by_event.csv`,
`sentinel_counts.csv`, `off_on_availability.csv` and a figure.

**What was missing.** Shares `docs/manuscript/02_ESSENTIAL_ANALYSES_E1_E2_E5`
with two unrelated analyses.

**Why a report is required.** Participant flow is a reporting requirement for
any clinical-cohort paper, and this run is the only place the full ledger from
19,450 published visits down to 3,361 transitions is reconciled with sentinel
counts.

→ `generated_reports/participant_flow_report.md`

---

## 6. E-2 — patient-clustered transition bootstrap  (T-02)

**Evidence it ran.** `experiments/20261103T193205Z_e2_transition_patient_bootstrap/`
with `report.txt`, `transition_probabilities.csv`, `other_quantities.csv`,
`bootstrap_draws.npz`, `replicate_sizes.csv` and a figure.

**What was missing.** Shares the same combined document.

**Why a report is required.** It supersedes the uncertainty method in
`report/Week3-4_Report` and `report/Subtype_Progression_Report`, both of which
use Wilson intervals that assume independent visits. Readers of those reports
need a document telling them which intervals to quote.

→ `generated_reports/transition_patient_bootstrap_report.md`

---

## 7. E-1 — lag-1 residual autocorrelation resolution

**Evidence it ran.** `experiments/20261103T191408Z_e1_lag1_residual_resolution/`
with `report.txt`, `lag1_statistic.csv`, `bootstrap_draws.npy`,
`null_draws.npy`, `report_diff.txt`, `rerun_outputs/` and `rerun_stdout.log`.

**What was missing.** Shares the same combined document.

**Why a report is required.** It is the only record of a live contradiction
inside the read-only `outputs/` tree, and `outputs/` cannot be corrected. A
standalone report is the only way a future reader learns not to quote that
paragraph.

→ `generated_reports/lag1_residual_resolution_report.md`

---

## Deliberately not generated

| Experiment | Why not |
|---|---|
| **M-04** 5-hidden-state LOO (`scripts/10`, `merge_5hidden_loo.py`) | `outputs/loo_5hidden/` holds predictions only — no metrics file, no report. Writing a results report would mean computing results that have never been computed. That is new analysis, not documentation, and it was not run |
| **S-04** PIGD measurement audit (`scripts/30`, `31`) | Run directories exist but completion cannot be verified against any document and no `report.txt` was written. Classified Status D; see §F of the audit |
| Smoke tests, pilots, shards (`experiments/_smoke/`, `*_smoke`, `*_pilot`, `*_DIAGNOSTIC`) | Throwaway by design |
| `experiments/threshold_narrow_0.91_1.13/` | A panel build, not an experiment. It is O-4's input and is described in O-4's report |
