# Probabilistic Motor Subtype Progression Model for Parkinson's Disease

B.Tech project, Department of Data Science, IIT Palakkad.
Data source: Parkinson's Progression Markers Initiative (PPMI).

Extends the clinical TD/PIGD motor subtype classification from a deterministic
rule into a hierarchical Bayesian progression model that reports, for an
individual patient, the probability of remaining in their current subtype or
transitioning to another over future follow-up.

## Status

| Stage | Deliverable | Status |
|---|---|---|
| Weeks 1–2 | Clean longitudinal dataset with subtype labels for every visit | **Complete** |
| Weeks 3–4 | Transition matrices and progression visualisations | **Complete** |
| Weeks 5–7 | Baseline probabilistic (Markov) model | Not started |
| Later | Hierarchical Bayesian model; σᵢ-as-phenotype analysis | Not started |

The panel covers **1,080 sporadic PD patients, 5,774 visits, 4,441 with a
subtype label**, out to 14 years of follow-up.

### Label validation

The recomputed labels are checked against both of PPMI's own derivations:

| Reference | Classes | Agreement |
|---|---|---|
| Curated cut `td_pigd` | 2 (TD / non-TD) | **4,431 / 4,431 = 100.00%** |
| Archived `td_pigd_old` | 3 (TD / Indet / PIGD) | **1,595 / 1,597 = 99.87%** |

The 3-class check covers 390 patients and is the only available test of the
Indeterminate/PIGD boundary, since the curated cut collapses those two.

## Quick start

```bash
pip install -e ".[dev]"
```

Audit what is on disk and run the interim analysis:

```bash
python scripts/00_data_audit.py
```

Once the MDS-UPDRS files are in `Data/06_Motor_UPDRS/`, build the panel:

```bash
python scripts/01_build_panel.py
```

Run the tests:

```bash
python -m pytest tests/ -q
```

## Layout

```
config/config.yaml         every item list, threshold, path and cohort rule
src/ppmi_subtype/
  config.py                loading and eager validation
  loaders.py               raw readers, one per source file
  subtype.py               TD/PIGD subscores, ratio and 3-class rule
  panel.py                 longitudinal patient-visit panel assembly
  transitions.py           empirical transition analysis (Weeks 3-4)
  validate.py              agreement against PPMI's own derivations
scripts/                   numbered pipeline stages
tests/                     55 tests over classification, transitions and validation
docs/DATA_REQUIREMENTS.md  what to download and why
outputs/                   generated, not tracked
extra/                     parked material, read by nothing — see extra/README.md
```

`Data/` holds only what the pipeline actually reads: the curated cut
(`01_Current_Clinical`), the three-class validation reference
(`04_Archived_TemporalShift`), the item-level MDS-UPDRS files
(`06_Motor_UPDRS`) and the PPMI dictionaries (`00_Reference`). Genetics,
biospecimen, online and unused MDS-UPDRS parts are parked under
`extra/unused_data/`, along with the report from the project's abandoned first
direction.

Nothing is hardcoded in the source: item lists, the 1.15 / 0.90 thresholds,
cohort membership, medication state and split ratios all live in
`config/config.yaml` and are validated on load.

## Findings so far

**Subtype distribution.** At baseline: TD 71.7%, Indeterminate 10.1%,
PIGD 18.1%. Pooled over all labelled visits: TD 65.1%, Indeterminate 9.9%,
PIGD 25.0%. Indeterminate is a real, populated state — roughly one visit in
ten — which is exactly what the curated cut's collapsed label throws away.

**The premise holds.** TD prevalence declines monotonically with follow-up —
71.7% at baseline, 63.7% at Y4, 47.9% at Y8, 39.1% at Y11 — so the TD → PIGD
drift the project is built around is present in the data rather than assumed.

**Subtypes are not monotone.** On the 2-class label, 26.2% of non-TD visits
revert to TD at the next assessment, against 19.3% moving the other way. A
patient's observed subtype is a noisy reading, not a ratchet. This is the
strongest available argument for the proposal's latent-state framing over
simple label counting, and it should be stated early.

**Transitions scale with interval.** TD → TD falls from 0.819 at ≤1.5 years to
0.578 beyond 4 years. Pooling over "consecutive visits" conflates progression
rate with attendance, so matrices must be stratified by interval or the model
moved to continuous time.

**The ratio is undefined for 16.3% of labelled visits** (725 of 4,441), where
the PIGD subscore is exactly zero. The visits are still classifiable — the rule
sends them to TD — but the continuous observation the hierarchical model
consumes does not exist for them. Where it is defined, the ratio has median
1.59 (IQR 0.68–2.95, max 11.36). This needs a modelling decision before the
hierarchical stage; options are in `docs/DATA_REQUIREMENTS.md`.

**"Unable to Rate" is coded 101 in the item fields.** All 13 Part III items
carry it; `NP3PSTBL` alone has 3,206 occurrences among OFF/untreated exams,
postural stability being frequently untestable. Averaged as a score it lifts
the 5-item PIGD mean to roughly 20 and mislabels tremor-dominant visits as
PIGD. `config.yaml` excludes it via `missing_value_codes`, and anything outside
the valid 0–4 range is dropped and counted.

**The strict missing-item policy costs almost nothing.** Allowing one missing
item per subscore recovers only 44 additional visits and changes zero existing
labels, so the pipeline keeps PPMI's strict rule.

**Part II predictors are ready.** The malignant-phenotype markers for the
trajectory-stability analysis are 97–100% complete at baseline: RBDSQ (`rem`),
SCOPA-AUT (`scopa`), UPSIT (`upsit`), MoCA (`moca`), DaTscan putamen SBR
(`MIA_PUTAMEN_BILAT`) and age.

## Weeks 3–4: transition analysis

`scripts/02_transition_analysis.py` → `outputs/transitions/` (9 CSV tables,
6 figures, `report.txt`). 3,361 consecutive labelled visit pairs from 934
patients, median interval 1.0 years.

Pooled transition matrix (rows = current visit, columns = next):

| | → TD | → Indeterminate | → PIGD |
|---|---|---|---|
| **TD** (n=2,261) | 0.808 | 0.080 | 0.112 |
| **Indeterminate** (n=334) | 0.440 | 0.192 | 0.368 |
| **PIGD** (n=766) | 0.184 | 0.114 | 0.702 |

**Indeterminate is a crossing, not a state.** Only 19.2% of patients labelled
Indeterminate are still Indeterminate at the next visit; 44.0% are back at TD
and 36.8% have moved to PIGD. It behaves exactly like the neutral band of a
continuous latent variable rather than a clinical category — which is the
proposal's premise, now measured rather than assumed.

**PIGD is not absorbing.** 29.8% of PIGD visits are followed by a non-PIGD
assessment, 18.4% of them straight back to TD. Any model that treats TD → PIGD
as a one-way progression is contradicted by the data.

**35.2% of all observed subtype changes reverse at the very next visit.** This
is the headline number of the whole analysis: a transition matrix built from
raw labels measures assessment noise as much as disease progression. Broken
down, TD → Indeterminate reverses 56% of the time and PIGD → Indeterminate 50%,
while the changes that stick are the ones crossing the full width of the scale.

**Requiring a conversion to hold for two consecutive visits eliminates 51.7% of
apparent conversions** (263 → 127). Kaplan-Meier cumulative incidence of PIGD
conversion by year 5 is 35.3% under a single-visit definition against 15.3%
under the sustained one. The choice of definition matters more than most
modelling choices downstream of it.

**Transitions scale with the gap between visits.** TD → PIGD rises from 0.100
at ≤1.5 years to 0.289 beyond 4 years, so a single pooled matrix is not a
well-defined quantity — it averages one-year and multi-year transitions in
whatever proportion attendance happened to produce. Weeks 5–7 should work in
continuous time rather than per-visit.

**Most patients never change state.** 622 of 1,080 (57.6%) hold one subtype for
their entire record; median state changes per patient is 0, mean 0.86. Across
86 distinct pathways, the commonest are TD-only (46.0%), PIGD-only (10.1%) and
TD → PIGD (7.7%).

## Reference

Stebbins GT, Goetz CG, Burn DJ, Jankovic J, Khoo TK, Tilley BC. How to identify
tremor dominant and postural instability/gait difficulty groups with the
Movement Disorder Society Unified Parkinson's Disease Rating Scale.
*Movement Disorders* 2013;28(5):668–70.
