# research_audit/

A documentation and evidence audit of this repository, carried out on
2026-11-04 at commit `8268ea5`.

Nothing outside this directory was created, modified, renamed or deleted. No
experiment was re-run.

## Start here

| File | What it is |
|---|---|
| [`experiment_report_audit.md`](experiment_report_audit.md) | **The audit.** Repository summary, report inventory, experiment inventory, the probabilistic TD–PIGD sub-inventory (§C.7), missing and incomplete reports, unverified experiments, contradictions (§G), and the recommended final report set |
| [`master_experiment_results.csv`](master_experiment_results.csv) | 222 rows, 47 experiment ids, one row per experiment or model configuration. Metric rows were written **programmatically from the experiments' own CSVs**, not retyped. Missing values are `NA`, never 0 |
| [`report_inventory.csv`](report_inventory.csv) | All 13 documents in `report/` and `reports/`, field by field |
| [`missing_reports.md`](missing_reports.md) | Status C — completed experiments that had no report, with the evidence they ran |
| [`incomplete_reports.md`](incomplete_reports.md) | Status B — reports that exist but are missing something, and exactly what |
| [`verify_audit.py`](verify_audit.py) | Re-checks the deliverables against the repository. Run it after any change |
| [`generated_reports/`](generated_reports/) | Seven new standalone experiment reports |

## Generated reports

| Report | Experiment |
|---|---|
| [`harmonised_family_comparison_report.md`](generated_reports/harmonised_family_comparison_report.md) | E-8 — five model families under one protocol |
| [`threshold_sensitivity_report.md`](generated_reports/threshold_sensitivity_report.md) | O-4 — how much of the label survives a threshold perturbation |
| [`transition_patient_bootstrap_report.md`](generated_reports/transition_patient_bootstrap_report.md) | E-2 — patient-clustered intervals for the transition analysis |
| [`participant_flow_report.md`](generated_reports/participant_flow_report.md) | E-5 — published cut to analysis cohort |
| [`lag1_residual_resolution_report.md`](generated_reports/lag1_residual_resolution_report.md) | E-1 — a stale conclusion in a read-only output |
| [`outcome_contrast_orientation_report.md`](generated_reports/outcome_contrast_orientation_report.md) | A-1 — reference orientation in the outcome contrasts |
| [`clinical_profile_reproduction_report.md`](generated_reports/clinical_profile_reproduction_report.md) | A-2 / A-2′ — reproducing the rich clinical profile |

Each follows the same eleven-section structure: overview, motivation, dataset
and cohort, methodology, configuration, results, interpretation, limitations,
reproducibility, manuscript relevance, source evidence.

## Verifying

```bash
~/envs/ppmi/bin/python research_audit/verify_audit.py
```

Checks that every document in both report folders is inventoried, that every
experiment id in the master table is named in the audit, that no metric column
was padded with a placeholder, that 27 numbers quoted in the generated reports
still match the files they came from, and that every generated report carries
its limitations, reproducibility and source-evidence sections.

## The three findings that need action

1. ~~**§G-2 — the IEEE manuscript reads a superseded file.**~~
   **Fixed 2026-11-04.** `paper/ieee/make_assets.py` now reads the authoritative
   `h3_component_vs_subtype.csv`, `main.tex` prints `+0.045`, and the paper
   rebuilds with 91 of 91 checks passing. Kept here as the record of what was
   found and what changed.
2. **§G-1 — `outputs/exploration/report.txt` contains a stale conclusion**
   that the committed code cannot produce. `outputs/` is read-only, so the file
   cannot be corrected; its closing item 4 must not be quoted.
3. ~~**§G-3 — one file exists only in a run whose code was not committed.**~~
   **Fixed 2026-11-04.** The cause was a missing `--baseline` argument, not a
   code change. Re-run as P-22
   (`experiments/20261104T085320Z_a3_profile_cv_with_baseline`) at commit
   `8268ea5`; the restored table reproduces the October one exactly.
