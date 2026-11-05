# Incomplete reports (Status B)

Experiments that have a report, but one that is missing information a reader
would need. Nothing here was modified; this file records what is absent.

Audit date 2026-11-04, commit `8268ea5`.

---

## 1. `docs/manuscript/02_ESSENTIAL_ANALYSES_E1_E2_E5_20261104.md`

**Covers** three independent analyses: E-1 (lag-1 residual contradiction), E-2
(patient-clustered transition bootstrap) and E-5 (participant flow).

| Missing | Detail |
|---|---|
| Separation | Three different research questions in one document, with one shared "Reproduce" section at the end |
| Per-experiment configuration | Seeds, draw counts and eligibility rules are given in prose for some analyses and not others. E-2's 2,000 draws and seed 42, and E-1's 200-vs-2,000 null draw change, are easy to miss |
| Limitations | One combined discussion; no per-analysis limitation list |
| Source evidence | Files are named inline, not listed |

**Resolved by** three standalone reports in `generated_reports/`. The original
document is untouched and remains the manuscript-facing summary.

---

## 2. `docs/manuscript/05_HARMONISED_FAMILY_COMPARISON_20261104.md`

**Covers** E-8, the 14-specification harmonised benchmark.

| Missing | Detail |
|---|---|
| Cohort section | No statement of the case list's construction, the patient count or the class balance |
| Per-family methodology | The table lists "subscore GP", "time-varying LMM" and so on without saying how each was fitted or under which native validation scheme |
| What was *not* harmonised | The decision rule and case list were harmonised; the fitting and validation schemes were not. The document does not say so |
| Reproducibility | No script invocation, no seed statement, no file list |
| Limitations | One short paragraph |

**Resolved by** `generated_reports/harmonised_family_comparison_report.md`.

---

## 3. `docs/latent_clinical_states/03_RICH_PROFILE_RESULTS_20261020.md`

**Covers** P-06, the nine-domain rich clinical profile.

Complete and well structured as science — H1 to H4, robustness, limitations,
files. **Missing one thing, through no fault of its own:** the run it reports
executed uncommitted code (commit `3a860ca` predates
`src/ppmi_subtype/clinical_profile.py`). That was discovered two weeks later.

**What to add:** a pointer to
`generated_reports/clinical_profile_reproduction_report.md`, which records that
a rerun at HEAD reproduces every number exactly, and that one file —
`derived_label_vs_subtype_history.csv` — exists only in the October run.

---

## 4. `docs/history_window/HISTORY_WINDOW_20261024.md`, section 6

**Covers** P-13, the rich-profile history-window experiment, as a secondary
section of the CT-HMM experiment's report.

| Missing | Detail |
|---|---|
| Prominence of the design difference | P-13 uses **one** CV repeat; P-06 uses four. Its numbers are therefore not interchangeable with P-06's, and the distinction appears only in the run's `report.txt` |
| Its own cohort table | The profile windows reduce the cohort differently from the CT-HMM windows (3,021 → 1,489, not 3,029 → 734) |
| Its own limitations | Inherits the parent experiment's |

**What to add:** a sentence at the head of section 6 stating the repeat count
and the cohort behaviour. No new computation is required — both are in
`experiments/20261024T042811Z_profile_history_window/report.txt`.

---

## 5. `docs/latent_clinical_states/01_LATENT_CLINICAL_STATES_20261020.md` — the k = 8 arm

**Covers** P-02 (state-count selection) and P-03 at k = 6 and k = 8.

The one-standard-error rule chose 6 states; both the arithmetic maximum and BIC
would have chosen 8. Both were run. The k = 8 arm is presented as a sensitivity
check, and its result (accuracy 0.7336 against k = 6's 0.7255 — both far below
the subtype-history baseline's 0.8384) is reported without its own discussion.

**What to add:** one paragraph stating that the conclusion — latent compression
of the clinical measurements loses information — holds at both state counts,
so it is not an artefact of the selection rule.

---

## 6. `reports/GPR_Onwards_Explanatory_Report.pdf`, Part G — S-05

Bayesian slope clustering (`scripts/34`,
`experiments/20261015T171236Z_bayesian_slope_clustering/`, 29 files) is
mentioned only through the conclusion that clustering was rejected. The run's
own outputs are not described.

**What to add:** a short subsection, or a pointer to the run directory, so that
the negative result is traceable to the posterior summaries that produced it.

---

## 7. M-04 — five-hidden-state LOO: no report, and no metrics to write one from

`outputs/loo_5hidden/` contains `loo_predictions.csv` and
`loo_predictions_all_models.csv` and nothing else — no metrics JSON, no
`report.txt`, no document. The 5-state accuracy that appears in E-8 (0.7513) is
computed there, from these predictions, not read from a stored metric.

**What is needed:** either compute and save metrics for this run, or state in
the documentation that the 5-state model is reported only through E-8. The
first is new computation and was not performed in this audit.

---

## Summary

| Status | Count |
|---|---|
| A — complete report | 30 experiment arms across `report/`, `reports/` and `docs/` |
| B — incomplete report | 7 (listed above) |
| C — report missing, now generated | 7 |
| C — report missing, not generated | 1 (M-04) |
| D — completion unverified | 1 (S-04) |
| E — planned but not run | 0 |
