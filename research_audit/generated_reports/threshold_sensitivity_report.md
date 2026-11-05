# O-4 — Threshold sensitivity of the motor-subtype label

## 1. Experiment overview

| | |
|---|---|
| **Experiment name** | O-4, threshold sensitivity |
| **Identifier in this audit** | D-03 |
| **Research question** | The subtype is a thresholded ratio. How much of the label survives a perturbation of the thresholds smaller than the measurement's own resolution? |
| **Objective** | Quantify the label's fragility using a perturbation already committed to the repository, with no model involved |
| **Run** | `experiments/20261104T002308Z_o4_threshold_sensitivity/`, 2026-11-04 (authoritative; two earlier attempts the same minute are superseded) |
| **Script** | `scripts/85_threshold_sensitivity.py` |
| **Status** | Completed and verified; **the script writes no `report.txt`** — this is the experiment's first report |
| **Related** | D-01 (the baseline label), D-02 (the cohort), P-18 (E-8, whose argument this supports) |

## 2. Scientific motivation

Every result in this project is a statement about a label that does not exist
in the data. The label is computed: eleven MDS-UPDRS items are averaged into a
tremor score, five into a PIGD score, and their ratio is compared against two
fixed thresholds. A visit is TD at ratio ≥ 1.15, PIGD at ≤ 0.90, and
Indeterminate between.

Those thresholds are conventions. If moving them by less than the gap between
two values the ratio can actually take reclassifies a large share of the data,
then the endpoint carries an irreducible arbitrariness, and no model — however
good — can predict past it. That is a statement about the measurement, and it
can be established without fitting anything.

## 3. Dataset and cohort

| | |
|---|---|
| **Dataset** | Frozen visit panel (baseline rule) vs a panel rebuilt under `config/config_narrow_0.91_1.13.yaml` |
| **Labelled visits** | 4,441 — identical in both panels |
| **Patients** | 1,080 — identical in both panels |
| **Target** | the three-class subtype label |
| **What differs** | the label column, and nothing else |

Both panels share every visit, every subscore and every cohort decision. Only
the classification rule applied to identical subscores changes. That is what
makes this a clean sensitivity analysis.

**Baseline class counts:** TD 2,891, Indeterminate 441, PIGD 1,109.

## 4. Methodology

The two rules:

```
baseline   PIGD <= 0.90    0.90 <  I <  1.15    TD >= 1.15    (outer, band open)
narrow     PIGD <  0.91    0.91 <= I <= 1.13    TD >  1.13    (inner, band closed)
```

**The perturbation changes three things, not two:** both numeric thresholds
*and* the side each band boundary is closed on. It must not be described as a
numeric shift alone.

The script compares the two panels visit by visit and patient by patient. It
also enumerates the ratio values the data can actually take — the ratio is a
quotient of two means of small-integer item scores, so it is discrete — in
order to express the perturbation in units of that resolution.

No model is fitted; no sampling is involved; the analysis is deterministic.

**Prerequisite.** `scripts/01_build_panel.py` must have been run with
`PPMI_CONFIG=config/config_narrow_0.91_1.13.yaml` first. That panel is at
`experiments/threshold_narrow_0.91_1.13/`. The script only compares; it never
rebuilds.

## 5. Experimental configuration

| Setting | Value |
|---|---|
| Baseline config | `config/config.yaml` (1.15 / 0.90, outer) |
| Narrow config | `config/config_narrow_0.91_1.13.yaml` (1.13 / 0.91, inner) |
| Comparison panel | `experiments/threshold_narrow_0.91_1.13/visit_panel.csv` |
| Randomness | none |
| Model | none |
| Run manifest | `manifest.json` in the run directory |

## 6. Results

### 6.1 How much moves

`summary.csv`:

| Quantity | Value |
|---|---|
| Labelled visits | 4,441 |
| **Visits relabelled** | **298 (6.71%)** |
| Patients | 1,080 |
| **Patients with at least one relabelled visit** | **235 (21.76%)** |
| Indeterminate visits, before | 441 |
| Indeterminate visits, after | 143 |

### 6.2 Every relabelled visit leaves the Indeterminate class

`per_class_movement.csv` and `label_crosstab.csv`:

| Class | Before | After | Δ | Leaving the class |
|---|---|---|---|---|
| TD | 2,891 | 2,991 | +100 | **0** |
| Indeterminate | 441 | 143 | −298 | **298 (67.6% of the class)** |
| PIGD | 1,109 | 1,307 | +198 | **0** |

Cross-tabulation, baseline (rows) against narrow (columns):

| | TD | Indeterminate | PIGD |
|---|---|---|---|
| **TD** | 2,891 | 0 | 0 |
| **Indeterminate** | 100 | 143 | 198 |
| **PIGD** | 0 | 0 | 1,109 |

No visit ever moves between TD and PIGD. The entire effect is the Indeterminate
band losing two-thirds of its members, 100 to TD and 198 to PIGD.

### 6.3 The perturbation is smaller than the data's resolution

`attainable_ratios.csv` lists the ratio values the cohort actually realises.
The two values that the perturbation crosses are:

| Ratio | Visits at this value |
|---|---|
| 0.9091 | 198 |
| 1.1364 | 100 |

198 + 100 = 298 — the whole effect. Both values sit between the baseline and
narrow thresholds, and neighbouring attainable values (0.974, 0.9848, 1.0,
1.1039, …) are further from the thresholds than the perturbation itself. The
label's fragility is not distributed smoothly along the ratio; it is
concentrated on two points at which many visits pile up, because the ratio of
two means of small integers takes a limited set of values.

### 6.4 Agreement against PPMI's own labels

`agreement_vs_ppmi.csv`:

| Rule | 2-class agreement (n = 4,431) | 3-class agreement (n = 1,597) |
|---|---|---|
| Baseline (1.15 / 0.90, outer) | **1.0000** | **0.9987** |
| Narrow (1.13 / 0.91, inner) | 0.9774 | 0.9286 |

The baseline rule reproduces PPMI's curated 2-class label exactly and its
archived 3-class label almost exactly. The narrow rule does not. **This does
not make the baseline rule true** — it makes it the rule PPMI used. The
comparison establishes which convention is in force, not which is correct.

## 7. Interpretation

**What the results support.**

* A threshold perturbation smaller than the spacing between attainable ratio
  values moves 6.71% of labelled visits and touches 21.76% of participants.
* The fragility is entirely a property of the Indeterminate class, which
  loses 67.6% of its members. The TD and PIGD classes are stable to this
  perturbation: no visit crosses between them.
* The effect is concentrated at exactly two attainable ratio values, which is
  a consequence of the ratio's discreteness, not of where the thresholds were
  placed.

**What the results do not support.**

* They do not show that either rule is wrong. Both are rules. "Which is right?"
  is not a question this analysis — or any analysis — can answer, because
  there is no adjudicating ground truth.
* They do not measure the effect on any downstream prediction result. No model
  was refitted under the narrow rule. The reader is told how much of the
  endpoint is convention-dependent, not how much of a given accuracy figure is.
* The 21.76% patient figure counts a patient as affected if **any** of their
  visits moves. It is not a statement that 21.76% of patients change subtype.

**Exploratory or confirmatory.** Confirmatory in design — the perturbation was
not chosen to produce an effect; it is the alternative configuration already
committed to the repository. The concentration of the whole effect on two
ratio values was discovered, not predicted.

## 8. Limitations

1. **One perturbation, not a range.** A single alternative rule is tested.
   A sweep over thresholds would characterise the fragility more completely
   and was not run.
2. **No downstream refit.** The effect on prediction accuracy is unmeasured.
   This is the single largest gap, and closing it means rerunning the
   prediction programme on the narrow panel.
3. **The perturbation confounds two changes.** Moving the thresholds and
   changing which side each boundary is closed on happen together, so their
   contributions cannot be separated from this run alone. Given §6.3, the
   boundary-closure change is likely to account for the bulk of it — 0.9091
   and 1.1364 are both strictly inside the baseline band and inside or on the
   narrow band's edge — but that is inference, not measurement.
4. **The panel build that produced the comparison panel has no run directory.**
   `experiments/threshold_narrow_0.91_1.13/` holds the panel, not a manifest
   of the invocation that created it.
5. **Three run directories exist**, within four minutes of one another. Only
   `20261104T002308Z` is authoritative; the first lacks `attainable_ratios.csv`.

## 9. Reproducibility

```bash
PPMI_CONFIG=config/config_narrow_0.91_1.13.yaml ~/envs/ppmi/bin/python scripts/01_build_panel.py
~/envs/ppmi/bin/python scripts/85_threshold_sensitivity.py
```

| Kind | Path |
|---|---|
| Script | `scripts/85_threshold_sensitivity.py` |
| Baseline config | `config/config.yaml` |
| Narrow config | `config/config_narrow_0.91_1.13.yaml` |
| Comparison panel | `experiments/threshold_narrow_0.91_1.13/visit_panel.csv` |
| Run directory | `experiments/20261104T002308Z_o4_threshold_sensitivity/` |
| Manifest | `…/manifest.json` |

Deterministic: no seed, no sampling.

## 10. Manuscript relevance

Main text. This is the cleanest single piece of evidence for the manuscript's
central claim, because it involves no model at all: the endpoint moves under a
perturbation smaller than its own measurement resolution, and all of the
movement is in the class every fitted model fails to predict.

Pair it with E-8's margin decomposition — 94.3% of the best model's advantage
arises in the Indeterminate stratum, which is the same 9.9% of cases whose
membership O-4 shows to be convention-dependent. The two results are
independent and point to the same conclusion.

The attainable-ratio table belongs in supplementary material.

## 11. Source evidence

| File | Contents |
|---|---|
| `experiments/20261104T002308Z_o4_threshold_sensitivity/summary.csv` | §6.1 |
| `…/per_class_movement.csv` | §6.2 |
| `…/label_crosstab.csv` | §6.2 cross-tabulation |
| `…/attainable_ratios.csv`, `…/fig_attainable_ratios.png` | §6.3 |
| `…/agreement_vs_ppmi.csv` | §6.4 |
| `…/per_patient_movement.csv` | Per-patient detail behind the 235 figure |
| `…/manifest.json` | Run manifest |
| `scripts/85_threshold_sensitivity.py` | Implementation and the rule statement |
| `config/config.yaml`, `config/config_narrow_0.91_1.13.yaml` | The two rules |
| `experiments/threshold_narrow_0.91_1.13/` | The comparison panel |
| `outputs/validation.txt` | Baseline agreement against PPMI, independently |

**Superseded attempts (do not read as separate analyses):**
`experiments/20261104T002141Z_o4_threshold_sensitivity/`,
`experiments/20261104T002247Z_o4_threshold_sensitivity/`.
