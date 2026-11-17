# Data requirements

No PPMI data is stored in this repository. Access is granted per user by PPMI
under its Data Use Agreement, so every file below has to be downloaded from the
PPMI portal (<https://www.ppmi-info.org/access-data-specimens/download-data>)
into a local `data/` folder, which is excluded from version control.

## Files the pipeline reads

| Path | Contents | Role |
|---|---|---|
| `data/clinical/PPMI_Curated_Data_Cut_Public_20260511.xlsx` | 19,450 visits × 212 columns, 4,788 participants, baseline to year 15 | Visit skeleton, covariates, baseline predictors |
| `data/motor_updrs/MDS-UPDRS_Part_II__Patient_Questionnaire_<date>.csv` | MDS-UPDRS Part II items | 3 of the 16 items behind the subtype |
| `data/motor_updrs/MDS-UPDRS_Part_III_<date>.csv` | MDS-UPDRS Part III items and medication state | 13 of the 16 items, and the OFF/ON state of each exam |
| `data/validation/PPMI_Original_Cohort_BL_to_Year_5_Dataset_Apr2020.csv` | 683 participants, baseline to year 5 | The only PPMI source with a **three-class** subtype label; used to validate ours |
| `data/dictionaries/` | Data dictionary, code list, user guides | Reference |

The analyses in this repository use data downloaded on 20 August 2026.

## The two item-level files

Both come from the PPMI portal:

> **LONI IDA → PPMI → Download → Study Data → Motor Assessments**

| Download | PPMI module | Save as |
|---|---|---|
| MDS-UPDRS Part II: Patient Questionnaire | `NUPDRS2P` | `data/motor_updrs/MDS-UPDRS_Part_II__Patient_Questionnaire_<date>.csv` |
| MDS-UPDRS Part III: Motor Examination | `NUPDRS3TRT` | `data/motor_updrs/MDS-UPDRS_Part_III_<date>.csv` |

The date suffix does not matter: `config/config.yaml` matches these by glob
(`paths.updrs_part2_glob`, `paths.updrs_part3_glob`). If PPMI changes the file
naming, edit those two patterns rather than any source file.

### Why the curated cut is not enough

The models need the continuous tremor/PIGD ratio and the three-class label.
Neither can be derived from the curated cut alone:

1. **The curated cut has no tremor score.** It ships `pigd` (the 5-item PIGD
   mean) but no corresponding tremor subscore. Ratio = tremor ÷ PIGD, so only
   the denominator is available.
2. **The curated cut's `td_pigd` is two-class.** Its dictionary decodes it as
   `1 = TD`, `2 = non-TD (PIGD or Indeterminate)`. The Indeterminate state has
   been collapsed away and cannot be recovered.
3. The archived April 2020 file has the three-class `td_pigd_old`, but covers
   only 683 participants to year 5, with no ratio. It is a validation
   reference, not an analysis dataset.

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

`PDSTATE` and `PDTRTMNT` are not optional. The classification uses the
**OFF-medication or untreated** exam: levodopa suppresses tremor, so scoring an
ON exam shifts patients toward PIGD and would manufacture transitions that are
pharmacological rather than pathological.

## Subtype definition

Following Stebbins et al., *Movement Disorders* 2013, and matching the
derivation notes attached to `td_pigd` in the curated cut:

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
be compared against it without remapping. The value 101 ("Unable to Rate") in
an item field is treated as missing, never as a score.

## Verifying the download

```bash
python scripts/00_data_audit.py
```

Section 1 should show all four boxes ticked. Then:

```bash
python scripts/01_build_panel.py
```

This writes `outputs/visit_panel.csv` and reports agreement against both PPMI
references. Expected: 4,431 of 4,431 visits against the curated two-class
label, and 1,595 of 1,597 against the archived three-class label. Any material
disagreement means the medication-state filter or the item list needs
revisiting before the panel is trusted.

## Visits where the ratio is undefined

**16.3% of labelled visits have a PIGD score of exactly zero** (725 of 4,441).
Those visits are still classifiable, since the published rule sends them to
TD, but the ratio itself does not exist. They are kept in every model that
works on the label, and left out only of the models that work on the ratio.
