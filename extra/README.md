# extra/

Everything here is parked, not deleted. Nothing in `src/`, `scripts/`,
`tests/` or `config/config.yaml` reads from this folder, so it can be ignored
while working on the current proposal.

## `old_ood_project/`

`Month1_Report.tex` and its PDF, from the project's first direction: outlier-aware
ensemble selection for PD prediction, using conformal p-values and a
Benjamini–Yekutieli step-up procedure over a pool of 12 models.

That direction was abandoned because the assembled PPMI data did not contain
enough genuine outliers to study. The report is kept because it documents two
real label-leakage findings and roughly 2,600 lines of pipeline work, but it
describes a plan that is **no longer active**. `Proposal.pdf` at the repository
root is the current source of truth.

## `unused_data/`

PPMI downloads not consumed by the current pipeline.

| Folder | Why it is parked |
|---|---|
| `02_Genetic_Subgroup/` | Genetics. A named *future extension* in the proposal (predictors of progression rate), not part of the TD/PIGD model. |
| `03_Biospecimen_SiteBatch/` | CSF and plasma assays. Same: future extension only. |
| `05_Online_ModalityShift/` | Remote MDS-UPDRS Part I/II. **Not usable** for this project at all — it has no Part III, so 13 of the 16 items the TD/PIGD ratio needs are absent. |
| `unused_MDS-UPDRS/` | Parts I, I-Patient-Questionnaire and IV from the Motor Assessments download. The subtype definition uses only Parts II and III. |
| `PPMI Data User Guide.pdf` | Byte-identical duplicate of `data/dictionaries/PPMI_Data_User_Guide_20240918.pdf`. |

The `_Subgroup`, `_SiteBatch`, `_ModalityShift` and `_TemporalShift` suffixes are
leftovers from the abandoned OOD work, where these were assembled as five
distribution-shift datasets. The fifth of them survives as
`data/validation/`, the sole three-class validation reference, and was renamed
because its old name described a purpose the project no longer has.

## Restoring something

If a future extension needs the genetics or biospecimen data, move the folder
under `data/` and add a path entry to `config/config.yaml`. Nothing else has to
change.
