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
| Weeks 5–7 | Baseline probabilistic model, validated on held-out patients | **Complete** |
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

## Repository layout

```
BTP/
├── config/
│   └── config.yaml            ← ALL parameters live here. Nothing is hardcoded:
│                                the 16 MDS-UPDRS items, the 1.15/0.90 thresholds,
│                                cohort rules, medication state, interval bins.
│
├── src/ppmi_subtype/          ← THE CODE. All logic lives here (~1,550 lines).
│   ├── config.py                 loads config.yaml, validates it on startup
│   ├── loaders.py                one reader per source file
│   ├── subtype.py                TD/PIGD subscores, ratio, 3-class decision rule
│   ├── panel.py                  assembles the longitudinal patient-visit panel
│   ├── transitions.py            transition matrices, reversion, Kaplan-Meier
│   ├── ctmc.py                   continuous-time Markov chain + baselines
│   ├── hmm.py                    latent state + misclassification model
│   ├── hbayes.py                 hierarchical Bayesian growth model (Gibbs)
│   ├── newpatient.py             first-visit progression model (the clinical one)
│   ├── splitting.py              patient-level train/test split
│   ├── validate.py               agreement against PPMI's own derivations
│   └── plots.py, model_plots.py  figures
│
├── scripts/                   ← Thin drivers. They call src/ and save results;
│   ├── 00_data_audit.py          they contain no analysis logic themselves.
│   ├── 01_build_panel.py         Run them in order.
│   ├── 02_transition_analysis.py
│   ├── 03_progression_model.py
│   ├── 04_hierarchical_bayes.py
│   └── 05_new_patient_model.py
│
├── tests/                     ← 134 tests, all exercising src/
│   ├── test_subtype.py           classification rule and its edge cases
│   ├── test_transitions.py       pairing, matrices, reversion, survival
│   ├── test_ctmc.py              generator recovery from simulated data
│   ├── test_hmm.py               latent recovery; reduces to plain chain
│   ├── test_hbayes.py            Gibbs sampler parameter recovery
│   └── test_validate.py          label-coding remaps
│
├── data/                      ← PPMI source data (gitignored, not redistributable)
│   ├── clinical/                 Curated Data Cut — the visit skeleton
│   ├── motor_updrs/              MDS-UPDRS Part II & III — the 16 raw items
│   ├── validation/               archived Apr-2020 cut, the 3-class reference
│   └── dictionaries/             PPMI data dictionaries, code lists, guides
│
├── outputs/                   ← Generated. Delete freely; scripts rebuild it.
│   ├── visit_panel.csv           one row per patient-visit, with subtype labels
│   ├── baseline.csv              one row per patient, with baseline predictors
│   ├── transitions/              9 CSV tables, 6 figures, report.txt
│   ├── model/                    14 CSV tables, 5 figures, report.txt
│   ├── hbayes/                   growth model + head-to-head comparison
│   └── newpatient/               first-visit model: the clinical deliverable
│
├── report/                    ← LaTeX source + compiled PDF
│   ├── Week3-4_Report.*          the Weeks 3–4 progress report (14 pp)
│   ├── Week5-7_Report.*          the Weeks 5–7 progression model (9 pp)
│   ├── Model_Comparison_Report.* growth model vs HMM (5 pp)
│   └── Pipeline_Walkthrough.*    how every number was generated (23 pp),
│                                 traced to the exact function that produced it
├── references/                ← The proposal and the source papers
├── docs/                      ← DATA_REQUIREMENTS.md: what to download and why
└── extra/                     ← Parked material, read by nothing. See its README.
```

**Where to look first.** The scientific method is in
[`src/ppmi_subtype/subtype.py`](src/ppmi_subtype/subtype.py) (how a subtype is
computed) and
[`src/ppmi_subtype/transitions.py`](src/ppmi_subtype/transitions.py) (how
transitions are measured). The scripts show the order things run in;
`config/config.yaml` shows every parameter that was chosen.

## Quick start

```bash
pip install -e ".[dev]"
```

Check which source files are present, and see the data audit:

```bash
python scripts/00_data_audit.py
```

Build the labelled longitudinal panel (Weeks 1–2 deliverable):

```bash
python scripts/01_build_panel.py
```

Run the transition analysis (Weeks 3–4 deliverable):

```bash
python scripts/02_transition_analysis.py
```

Fit and validate the progression model (Weeks 5–7 deliverable, ~20 min):

```bash
python scripts/03_progression_model.py
```

Run the tests:

```bash
python -m pytest tests/ -q
```

## The pipeline in one view

```
data/clinical/          data/motor_updrs/
  curated cut             Part II + Part III items
  (visit skeleton)        (16 raw scores)
        │                         │
        └───────────┬─────────────┘
                    ▼
        scripts/01_build_panel.py
          · filter to sporadic PD           panel.filter_cohort
          · keep OFF-medication exam        subtype.select_medication_state
          · drop "101 = Unable to Rate"     subtype.clean_items
          · tremor/PIGD means → ratio       subtype.compute_subscores
          · apply 1.15 / 0.90 rule          subtype.classify
          · validate vs PPMI (100%/99.87%)  validate.*
                    ▼
        outputs/visit_panel.csv     1,080 patients · 5,774 visits · 4,441 labelled
                    ▼
        scripts/02_transition_analysis.py
          · pair adjacent labelled visits   transitions.consecutive_pairs
          · 3×3 matrix + Wilson CIs         transitions.transition_matrix*
          · stratify by elapsed interval    transitions.transition_matrix_by_interval
          · reversion analysis              transitions.reversion_after_change
          · time to conversion              transitions.first_entry, kaplan_meier
          · pathways                        transitions.state_sequences
                    ▼
        outputs/transitions/        3,361 pairs · 9 tables · 6 figures
                    ▼
        scripts/03_progression_model.py
          · patient-level train/test split  splitting.split_patients
          · test the Markov property        chi-squared on labelled triples
          · discrete + continuous chains    ctmc.fit_dtmc, ctmc.fit_ctmc
          · latent state + misclassification hmm.fit_hmm
          · validate on held-out patients   10 independent splits
                    ▼
        outputs/model/              14 tables · 5 figures · clinical output table
```

`data/` holds only what the pipeline actually reads. Genetics, biospecimen, the
online Part I/II download and the unused MDS-UPDRS parts are parked under
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

## Weeks 5–7: the progression model

`scripts/03_progression_model.py` → `outputs/model/`. Three models fitted on
755 training patients, validated on 325 held-out ones.

**A Markov chain in the observed label is misspecified — provably.** For visits
currently labelled TD, arriving from PIGD raises P(PIGD next visit) from 0.086
to 0.393 (χ² = 156.6, p = 7.9×10⁻³³). The current state is not a sufficient
summary of the past, which is the signature of a latent state read through
noise.

**The hidden Markov model wins 10/10 held-out splits**, by +0.028 ± 0.009
log-likelihood per transition over the discrete baseline (paired t-test
p = 5.4×10⁻⁶). Ranking: HMM > discrete chain > continuous-time chain > null.

**The continuous-time chain did not help** — a negative result. It loses at
every interval, most at long ones, because P(t) = exp(Qt) imposes consistency
across intervals that a noisily-observed process does not satisfy.

**Progression is sequential.** The latent direct TD → PIGD rate is 16× smaller
than TD → Indeterminate, and PIGD → TD directly is zero. The model recovered an
ordering it was never given.

**The phenotype is far more stable than its label:** TD persists ~10 years
latent against 3.4 observed; PIGD 16.5 against 2.3.

**Clinical output.** A patient currently Indeterminate has a **63% chance of
being PIGD within five years**, against 19% for a patient currently TD.

## The clinical deliverable: predicting a brand-new patient

`scripts/05_new_patient_model.py` → `outputs/newpatient/`.

**The longitudinal models earn their accuracy from history, and a new patient
has none.** Measured directly — predicting subtype at horizon *h* from a first
visit only, five splits:

| Model (baseline information only) | Held-out log-lik |
|---|---|
| **Baseline subtype + ratio + covariates** | **−0.7550** |
| + covariates | −0.7625 |
| + baseline ratio | −0.7735 |
| Baseline subtype + horizon | −0.7813 |
| HMM (nothing to filter) | −0.7869 |

For a first visit the HMM is the *worst* of these. A direct model of
P(subtype at h | first visit) beats it by **+0.032**, at 71.9% accuracy and
well calibrated across the full probability range.

**The HMM's value grows with follow-up** — its advantage over a lookup table
runs +0.005 at one visit, +0.021 at two, +0.047 at six or more. The two models
are complementary: this one answers the first visit, the HMM answers every
visit after.

**Which baseline measures matter** (held-out cost of dropping each): only
`updrs2_score` (+0.015), DaTscan putamen binding (+0.005) and `age` (+0.004)
earn their place. The four non-motor markers — RBDSQ, SCOPA-AUT, UPSIT, MoCA —
all sit at or below zero. Worth knowing before the proposal's second half is
built on them, though not a refutation: that hypothesis concerns trajectory
*stability*, which this model does not estimate.

## The hierarchical Bayesian growth model

`scripts/04_hierarchical_bayes.py` → `outputs/hbayes/`. The model specified in
the proposal (pp. 13–15), implemented as written and compared against the HMM.

**Population progression: the TD/PIGD ratio declines 0.093 per year**
(95% CI 0.068–0.116). From the mean baseline ratio of 2.29, the population
average crosses the TD threshold after 12.3 years. Between-patient SD of the
slope is 0.103/year — patients differ as much as the average effect itself.

**On predicting the next subtype, the HMM is ahead in 6/6 splits** by
+0.058 ± 0.019 log-lik per transition (paired t-test p = 6.6×10⁻⁴). The reason
is measurable: the growth model's residual SD is σ = 1.138, **4.6× the width of
the entire Indeterminate band**, so band membership is noise-dominated. It also
places 12.2% of its five-year predictive mass on negative ratios, which are
impossible.

**Two problems in the model as specified**, found by implementing it: σ_obs and
σ_proc are provably not separable (only their sum is identified), and a Normal
on the raw ratio cannot represent a non-negative, right-skewed quantity.

**Keep both.** The HMM classifies the next visit better; the growth model is
the only one that gives an interpretable progression rate, per-patient slopes,
and a route to the σᵢ-as-phenotype analysis in the proposal's second half.

**The results are a lower bound on the drift.** 23.1% of visits carry no label.
Roughly half of that loss is exams that were performed but could not be fully
scored — and in 1,243 of those 1,253 exams the missing item is postural
stability, the pull test, which is often unsafe in a patient with poor balance.
Those exams come from more advanced patients (median Hoehn & Yahr 3.0 vs 2.0,
p = 1.4×10⁻¹⁷), and postural stability is a *PIGD* item. So the visits lost are
disproportionately the ones most likely to be PIGD. See §6.1 of the Weeks 3–4
report.

## Reference

Stebbins GT, Goetz CG, Burn DJ, Jankovic J, Khoo TK, Tilley BC. How to identify
tremor dominant and postural instability/gait difficulty groups with the
Movement Disorder Society Unified Parkinson's Disease Rating Scale.
*Movement Disorders* 2013;28(5):668–70.
