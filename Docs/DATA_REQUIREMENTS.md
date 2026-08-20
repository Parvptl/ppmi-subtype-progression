# Data requirements

## What is already in the repository

| Path | Contents | Role |
|---|---|---|
| `Data/01_Current_Clinical/PPMI_Curated_Data_Cut_Public_20260511.xlsx` | 19,450 visits × 212 cols, 4,788 patients, BL → Year 15 | Visit skeleton, covariates, baseline predictors |
| `Data/04_Archived_TemporalShift/PPMI_Original_Cohort_BL_to_Year_5_Dataset_Apr2020.csv` | 683 patients, BL → Y5 | The only in-repo source with a **3-class** subtype label |
| `Data/02_Genetic_Subgroup/`, `03_Biospecimen_SiteBatch/` | Genetics, biospecimen assays | Future extensions |
| `Data/05_Online_ModalityShift/` | Remote MDS-UPDRS Part I/II | Not usable here — no Part III |
| `Docs/` | Annotated data dictionary, code list, user guides | Reference |

## What must be downloaded

Two files, both from the PPMI portal:

> **LONI IDA → PPMI → Download → Study Data → Motor Assessments**

| Download | PPMI module | Save as |
|---|---|---|
| MDS-UPDRS Part II: Patient Questionnaire | `NUPDRS2P` | `Data/06_Motor_UPDRS/MDS-UPDRS_Part_II__Patient_Questionnaire_<date>.csv` |
| MDS-UPDRS Part III: Motor Examination | `NUPDRS3TRT` | `Data/06_Motor_UPDRS/MDS-UPDRS_Part_III_<date>.csv` |

The date suffix does not matter — `config/config.yaml` matches these by glob
(`paths.updrs_part2_glob`, `paths.updrs_part3_glob`). If PPMI changes the file
naming, edit those two patterns rather than any source file.

### Why these two, specifically

The proposal's model consumes the continuous TD/PIGD ratio and the three-class
label. Neither can be derived from anything currently on disk:

1. **The curated cut has no tremor score.** It ships `pigd` (the 5-item PIGD
   mean) but no corresponding tremor subscore. Ratio = tremor ÷ PIGD, so only
   the denominator is available.
2. **The curated cut's `td_pigd` is 2-class.** Its dictionary decodes it as
   `1 = TD`, `2 = non-TD (PIGD or Indeterminate)`. The Indeterminate state —
   the middle column of the proposal's output table — has been collapsed away
   and cannot be recovered.
3. The archived Apr-2020 file has 3-class `td_pigd_old`, but covers only 683
   patients to Year 5, with no ratio. It is a validation reference, not an
   analysis dataset.

### Columns the pipeline reads

**From Part II** (3 items):

```
NP2TRMR   NP2WALK   NP2FREZ
```

**From Part III** (13 items plus medication state):

```
NP3PTRMR  NP3PTRML  NP3KTRMR  NP3KTRML  NP3RTARU  NP3RTALU  NP3RTARL
NP3RTALL  NP3RTALJ  NP3RTCON  NP3GAIT   NP3FRZGT  NP3PSTBL
PDSTATE   PDTRTMNT  NHY
```

Both files must also carry `PATNO` and `EVENT_ID`, which are the merge keys.

`PDSTATE` and `PDTRTMNT` are not optional. The canonical classification uses
the **OFF-medication or untreated** exam; levodopa suppresses tremor, so
scoring an ON exam shifts patients toward PIGD and would manufacture
transitions that are pharmacological rather than pathological.

### Definition applied

Following Stebbins et al., *Movement Disorders* 2013 — and matching the
derivation notes attached to `td_pigd` in the curated cut verbatim:

```
tremor_score = mean(11 tremor items)
pigd_score   = mean(5 gait/postural items)
ratio        = tremor_score / pigd_score

ratio >= 1.15                          -> TD             (1)
pigd_score == 0 and tremor_score > 0   -> TD             (1)
0.90 < ratio < 1.15                    -> Indeterminate  (2)
tremor_score == 0 and pigd_score == 0  -> Indeterminate  (2)
ratio <= 0.90                          -> PIGD           (3)
```

Codes are 1/2/3 to match the archived `td_pigd_old`, so recomputed labels can
be compared against it without remapping.

## Verifying the download

```bash
python scripts/00_data_audit.py
```

Section 1 should show all four boxes ticked. Then:

```bash
python scripts/01_build_panel.py
```

This writes `outputs/visit_panel.csv` and reports agreement against both PPMI
references. Expect near-perfect agreement with the curated `td_pigd` after
collapsing to binary; any material disagreement means the medication-state
filter or the item list needs revisiting before the panel is trusted.

## Known issue to decide before Stage 3

**16.2% of visits have a PIGD subscore of exactly zero**, measured on the
curated cut's `pigd` column for the sporadic PD cohort. Those visits are
classifiable (the rule sends them to TD) but the *ratio* is undefined, so the
observation model `observed ratio = latent phenotype + noise` has nothing to
consume for roughly one visit in six — concentrated, predictably, in early and
tremor-dominant patients, which is exactly the population the model most needs
to characterise.

This is a modelling decision, not a data problem. Three defensible routes:

- model `tremor_score` and `pigd_score` jointly as a bivariate latent process
  and derive the ratio's posterior, which keeps every visit;
- model `log(tremor + c) − log(pigd + c)` for a small fixed `c`, which is
  simple but makes the thresholds depend on `c`;
- drop the ratio and model the ordinal label directly, which forfeits the
  proposal's continuous-latent framing.

The first is the most faithful to the proposal and should be costed early.
