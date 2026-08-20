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
| Weeks 1–2 | Clean longitudinal dataset with subtype labels for every visit | **Blocked** — see below |
| Weeks 3–4 | Transition matrices and progression visualisations | Code written, awaiting labels |
| Weeks 5–7 | Baseline probabilistic (Markov) model | Not started |
| Later | Hierarchical Bayesian model; σᵢ-as-phenotype analysis | Not started |

**The blocker.** The TD/PIGD ratio and the three-class label cannot be derived
from any file currently in the repository. The curated data cut carries no
tremor subscore, and its `td_pigd` is collapsed to `TD` / `non-TD`, discarding
the Indeterminate state. Two item-level MDS-UPDRS files must be downloaded from
LONI; see [docs/DATA_REQUIREMENTS.md](docs/DATA_REQUIREMENTS.md).

Everything downstream of those two files is written and tested.

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
tests/                     40 tests over the classification and transition rules
docs/DATA_REQUIREMENTS.md  what to download and why
outputs/                   generated, not tracked
```

Nothing is hardcoded in the source: item lists, the 1.15 / 0.90 thresholds,
cohort membership, medication state and split ratios all live in
`config/config.yaml` and are validated on load.

## Findings so far

From `scripts/00_data_audit.py`, sporadic PD cohort (n = 1,080; 4,461 labelled
visits; median 3 visits per patient, up to 14 years of follow-up):

**The premise holds.** TD prevalence declines monotonically with follow-up —
71.7% at baseline, 63.7% at Y4, 47.9% at Y8, 39.1% at Y11 — so the TD → PIGD
drift the project is built around is present in the data rather than assumed.

**Subtypes are not monotone.** Using the 2-class label, 26.2% of non-TD visits
revert to TD at the next assessment, against 19.3% moving the other way. A
patient's observed subtype is a noisy reading, not a ratchet. This is the
strongest available argument for the proposal's latent-state framing over
simple label counting, and it should be stated early.

**Transitions scale with interval.** TD → TD falls from 0.819 at ≤1.5 years to
0.578 beyond 4 years. Pooling over "consecutive visits" conflates progression
rate with attendance, so matrices must be stratified by interval or the model
moved to continuous time.

**The ratio is undefined for 16.2% of visits**, where the PIGD subscore is
exactly zero. This needs a modelling decision before the hierarchical stage;
options are set out in `docs/DATA_REQUIREMENTS.md`.

**Part II predictors are ready.** The malignant-phenotype markers for the
trajectory-stability analysis are 97–100% complete at baseline: RBDSQ (`rem`),
SCOPA-AUT (`scopa`), UPSIT (`upsit`), MoCA (`moca`), DaTscan putamen SBR
(`MIA_PUTAMEN_BILAT`) and age.

## Reference

Stebbins GT, Goetz CG, Burn DJ, Jankovic J, Khoo TK, Tilley BC. How to identify
tremor dominant and postural instability/gait difficulty groups with the
Movement Disorder Society Unified Parkinson's Disease Rating Scale.
*Movement Disorders* 2013;28(5):668–70.
