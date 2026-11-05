# E-2 — Patient-clustered intervals for the observed-transition analysis

## 1. Experiment overview

| | |
|---|---|
| **Experiment name** | E-2, patient-clustered transition bootstrap |
| **Identifier in this audit** | T-02 |
| **Research question** | The project's transition intervals assume independent visits, but patients contribute several each. How much do the intervals change when the resampling unit is the patient? |
| **Objective** | Replace visit-level Wilson intervals with patient-clustered bootstrap intervals for every observed-progression quantity the project reports |
| **Run** | `experiments/20261103T193205Z_e2_transition_patient_bootstrap/`, 2026-11-03 |
| **Script** | `scripts/77_transition_patient_bootstrap.py` |
| **Status** | Completed and verified; report written by this audit |
| **Related** | T-01 (the original transition analysis, `report/Week3-4_Report`), D-02 (E-5, the cohort), P-18 (E-8, which uses the same patient-clustered scheme) |

## 2. Scientific motivation

`report/Week3-4_Report` and `report/Subtype_Progression_Report` report the
transition matrix, the reversion rate and the Kaplan–Meier conversion curves
with Wilson score intervals. A Wilson interval treats each observation as
independent. Here the observations are visit pairs, and the 3,361 pairs come
from only 934 patients — a median of three per patient and up to fourteen.
Visits from one patient are correlated, so a visit-level interval is too
narrow by an unknown amount.

The question is not whether the point estimates are right. They are: this run
calls the project's own library functions unmodified and reproduces every one.
The question is how much credibility the intervals around them deserve.

## 3. Dataset and cohort

| | |
|---|---|
| **Dataset** | Frozen visit panel, md5 `fb038ddbdc19ab3368e13940f398e7ba` |
| **Patients** | 1,080 in the cohort; **934** contribute at least one transition |
| **Visits** | 4,441 labelled |
| **Transitions** | **3,361** consecutive labelled pairs |
| **Resampling unit** | the patient — all of a patient's pairs move together |
| **Target** | the observed transition structure, not a prediction |

## 4. Methodology

1. Point estimates are computed by calling the project's existing functions in
   `src/ppmi_subtype/transitions.py` — unmodified, so that the bootstrap
   cannot silently change the quantity being estimated.
2. 2,000 bootstrap replicates are drawn by resampling **patients** with
   replacement; each drawn patient brings all of their pairs.
3. Every reported quantity is recomputed on each replicate. Intervals are the
   2.5th and 97.5th percentiles of the replicate distribution; a
   normal-approximation width is recorded alongside as a shape check.
4. Replicate sizes are saved (`replicate_sizes.csv`) so that the variation in
   the number of pairs per replicate is visible rather than assumed away.
5. The raw draws are saved (`bootstrap_draws.npz`) so that any further
   quantity can be re-derived without rerunning.

## 5. Experimental configuration

| Setting | Value |
|---|---|
| Bootstrap draws | 2,000 |
| Resampling unit | patient |
| Seed | 42 |
| Interval | percentile (2.5 / 97.5), with a normal-approximation width reported for comparison |
| Point estimates | from `src/ppmi_subtype/transitions.py`, unmodified |

## 6. Results

### 6.1 Transition probabilities

`transition_probabilities.csv`. "Width ratio" is bootstrap width ÷ Wilson width.

| From | To | n from | Probability | Wilson 95% | Patient bootstrap 95% | Width ratio |
|---|---|---|---|---|---|---|
| TD | TD | 2,261 | 0.8080 | 0.7913 – 0.8238 | 0.7881 – 0.8264 | 1.18 |
| TD | Indeterminate | 2,261 | 0.0801 | 0.0696 – 0.0920 | 0.0684 – 0.0918 | 1.05 |
| TD | PIGD | 2,261 | 0.1119 | 0.0996 – 0.1256 | 0.0984 – 0.1275 | 1.12 |
| Indeterminate | TD | 334 | 0.4401 | 0.3879 – 0.4937 | 0.3809 – 0.5000 | 1.12 |
| Indeterminate | Indeterminate | 334 | **0.1916** | 0.1530 – 0.2372 | **0.1484 – 0.2339** | 1.02 |
| Indeterminate | PIGD | 334 | 0.3683 | 0.3183 – 0.4212 | 0.3179 – 0.4169 | 0.96 |
| PIGD | TD | 766 | 0.1841 | 0.1582 – 0.2131 | 0.1535 – 0.2180 | 1.18 |
| PIGD | Indeterminate | 766 | 0.1136 | 0.0930 – 0.1380 | 0.0914 – 0.1381 | 1.04 |
| PIGD | PIGD | 766 | 0.7023 | 0.6690 – 0.7337 | 0.6605 – 0.7380 | 1.20 |

### 6.2 Derived progression quantities

`other_quantities.csv`.

| Quantity | Estimate | Patient bootstrap 95% |
|---|---|---|
| **Reversion after any change (pooled)** | **0.3522** | **0.3180 – 0.3880** |
| revert TD → Indeterminate | 0.5597 | 0.4782 – 0.6417 |
| revert PIGD → Indeterminate | 0.5000 | 0.3617 – 0.6481 |
| revert PIGD → TD | 0.3929 | 0.3053 – 0.4808 |
| revert TD → PIGD | 0.3198 | 0.2530 – 0.3883 |
| revert Indeterminate → PIGD | 0.1828 | 0.1026 – 0.2644 |
| revert Indeterminate → TD | 0.1759 | 0.1000 – 0.2547 |
| KM conversion to PIGD at 2 y, naive | 0.1252 | 0.1025 – 0.1504 |
| **KM conversion to PIGD at 5 y, naive** | **0.3525** | **0.3077 – 0.3988** |
| KM conversion at 2 y, sustained | 0.0654 | 0.0490 – 0.0846 |
| **KM conversion at 5 y, sustained** | **0.1533** | **0.1200 – 0.1902** |
| Share of naive conversions removed by requiring sustainment | 0.5171 | 0.4572 – 0.5774 |
| p(TD → PIGD \| interval ≤ 1.5 y) | 0.1004 | 0.0865 – 0.1161 |
| p(TD → PIGD \| interval > 4 y) | 0.2889 | 0.1667 – 0.4318 |
| p(TD → TD \| interval ≤ 1.5 y) | 0.8203 | 0.7999 – 0.8392 |
| p(TD → TD \| interval > 4 y) | 0.5778 | 0.4317 – 0.7188 |

Replicate-level cohort sizes: pairs 3,361 (bootstrap 3,187 – 3,537), patients
with a transition 934 (913 – 955), changes 673 (600 – 753).

## 7. Interpretation

**What the results support.**

* The point estimates in the earlier reports are correct — they are reproduced
  here by the same library functions.
* Patient clustering widens most intervals, by **2% to 20%**. The widening is
  largest for the self-transitions of the two large states (TD → TD 1.18,
  PIGD → PIGD 1.20), where one patient can contribute many repeats of the same
  outcome, and smallest — in one case slightly negative — for the transitions
  out of the small Indeterminate state, where few patients contribute more
  than one.
* None of the project's qualitative conclusions changes. Indeterminate
  persistence remains far below the other two states; more than a third of
  observed changes still revert; requiring a conversion to be sustained still
  removes about half of the naive conversions.

**What the results do not support.**

* The widening is not a correction to any published point estimate.
* Interval-stratified probabilities at long gaps remain very uncertain
  (p(TD → PIGD | > 4 y) spans 0.167 to 0.432) and no narrative should be built
  on that cell.

**Which intervals to quote.** The bootstrap ones, everywhere. They are
reported here precisely so that the Wilson intervals in `Week3-4_Report` and
`Subtype_Progression_Report` are not quoted in new work.

## 8. Limitations

1. **The bootstrap corrects for clustering, not for informative visit timing.**
   Patients who attend more often contribute more pairs, and attendance is not
   independent of disease course.
2. **A percentile interval on a probability near a boundary can be skewed.**
   The normal-approximation widths are reported alongside for exactly this
   reason; they agree closely here, but the check belongs with the result.
3. **2,000 draws** are enough for a 95% percentile interval and not for
   extreme quantiles.
4. **Kaplan–Meier assumes non-informative censoring**, which the cohort's
   attendance pattern does not guarantee. The bootstrap widens the interval; it
   does not repair the assumption.
5. **The cell p(TD → PIGD | > 4 y) rests on few transitions** and its interval
   should be read as a statement that the quantity is unknown.

## 9. Reproducibility

```bash
~/envs/ppmi/bin/python scripts/77_transition_patient_bootstrap.py
```

| Kind | Path |
|---|---|
| Script | `scripts/77_transition_patient_bootstrap.py` |
| Library | `src/ppmi_subtype/transitions.py` (unmodified) |
| Run directory | `experiments/20261103T193205Z_e2_transition_patient_bootstrap/` |
| Raw draws | `…/bootstrap_draws.npz` |
| Integrity | `MANIFEST.sha256`; panel md5 asserted at start |

Seed 42, 2,000 patient-clustered draws.

## 10. Manuscript relevance

Main text. Every observed-progression number in the manuscript — the
transition matrix, the reversion rate, the two conversion curves — should
carry these intervals rather than the Wilson ones. The width-ratio column
belongs in supplementary material as the justification.

It supports the first leg of the label-reliability argument: that the recorded
label moves a great deal, and that more than a third of its movement is undone
at the very next assessment.

## 11. Source evidence

| File | Contents |
|---|---|
| `experiments/20261103T193205Z_e2_transition_patient_bootstrap/report.txt` | Narrative report |
| `…/transition_probabilities.csv` | §6.1 |
| `…/other_quantities.csv` | §6.2 |
| `…/replicate_sizes.csv` | Per-replicate cohort sizes |
| `…/bootstrap_draws.npz` | All 2,000 replicates |
| `…/fig1_wilson_vs_patient_bootstrap.png/.csv` | Width comparison figure |
| `…/validation_checks.csv` | The run's own assertions |
| `outputs/transitions/*.csv` | The original visit-level analysis |
| `docs/manuscript/02_ESSENTIAL_ANALYSES_E1_E2_E5_20261104.md` | Earlier combined write-up |
