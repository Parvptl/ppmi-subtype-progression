# A-1 — Reference orientation in the outcome-discrimination contrasts

## 1. Experiment overview

| | |
|---|---|
| **Experiment name** | A-1, outcome-contrast reference orientation |
| **Identifier in this audit** | P-21 |
| **Research question** | The comparator component was given a one-time orientation flip so that a higher score means more outcome; the subtype-label reference was not. Does fixing that asymmetry change the reported contrasts? |
| **Objective** | Isolate the effect of orienting the reference, under one version of the library, with nothing refitted |
| **Run** | `experiments/20261103T212122Z_a1_outcome_contrast_orientation/`, 2026-11-03, commit `7632d5a` |
| **Script** | `scripts/82_outcome_contrast_orientation_fix.py` |
| **Status** | Completed. **The run's output values are superseded** — see §6.3 and §7 |
| **Related** | P-06 (the source CV run), P-19 / P-20 (A-2 and A-2′, which produce the authoritative table) |

## 2. Scientific motivation

The rich clinical profile (P-06) is tested against four independently recorded
clinical outcomes — Hoehn & Yahr stage ≥ 3, mild cognitive impairment, motor
complications and a cognitive diagnosis — by asking whether its best
coordinate discriminates each outcome better than the subtype label does.

`clinical_profile.best_component_vs_subtype` gave the comparator component a
one-time orientation flip, so that a component whose AUC fell below 0.5 was
reflected and scored in its favourable direction. The subtype-label reference
received no such flip. On any outcome whose label AUC falls below chance, the
comparator is therefore allowed its favourable direction while the reference is
not, and the reported difference is inflated.

Only one of the four outcomes is affected in this cohort: motor complications,
where the label's raw AUC is 0.4963, below chance, against 0.5037 oriented.

## 3. Dataset and cohort

| | |
|---|---|
| **Dataset** | Frozen panel; the rich-profile CV run's saved anchor table |
| **Source run** | `experiments/20261020T180515Z_clinical_profile_cv` |
| **Patients per outcome** | H&Y ≥ 3: 761 (127 events) · MCI: 818 (298) · motor complications: 736 (470) · cognitive dx: 816 (327) |
| **Target** | each outcome, recorded later than the profile coordinates used to discriminate it |
| **Refits** | **zero** — the run reads saved tables |

## 4. Methodology

Both arms are computed **inside this run, under one version of the library**,
differing only in whether the reference is oriented. That isolates the change.
Comparing the fixed code against the figures saved by the October run would
also pick up unrelated library drift since; the run records that drift
separately, as information, not as the check.

For each outcome and each coordinate kind (level, rate):

1. select the best-discriminating component (the selection is flagged
   `selection_is_optimistic = True`, because it is chosen on the same data);
2. compute its AUC and the subtype label's AUC;
3. form the paired difference and bootstrap it, 2,000 draws, seed 42;
4. repeat with the reference oriented, and record the difference between the
   two arms.

## 5. Experimental configuration

| Setting | Value |
|---|---|
| Bootstrap draws | 2,000 |
| Seed | 42 |
| Refits | 0 |
| Source run | `20261020T180515Z_clinical_profile_cv` |
| Commit | `7632d5a` |
| Checks recorded in `manifest.json` | 9, all passing |

## 6. Results

### 6.1 Only motor complications moves

`before_after.csv`. Reference AUC is `auc_b`.

| Outcome | Kind | Label AUC before | Label AUC after | Δ before | Δ after | Change |
|---|---|---|---|---|---|---|
| H&Y ≥ 3 | rate | 0.5794 | 0.5794 | 0.0532 | 0.0532 | 0.0000 |
| H&Y ≥ 3 | level | 0.5794 | 0.5794 | 0.1052 | 0.1052 | 0.0000 |
| MCI | rate | 0.5494 | 0.5494 | 0.0428 | 0.0428 | 0.0000 |
| MCI | level | 0.5494 | 0.5494 | 0.1605 | 0.1605 | 0.0000 |
| **Motor complications** | **rate** | **0.4963** | **0.5037** | 0.0408 | **0.0334** | **−0.0074** |
| **Motor complications** | **level** | **0.4963** | **0.5037** | 0.0347 | **0.0272** | **−0.0075** |
| Cognitive dx | rate | 0.5582 | 0.5582 | 0.0356 | 0.0356 | 0.0000 |
| Cognitive dx | level | 0.5582 | 0.5582 | 0.1194 | 0.1194 | 0.0000 |

The fix behaves exactly as predicted: the three outcomes whose label AUC
exceeds chance are untouched, because the two conventions coincide there; the
one that falls below chance shrinks by about 0.0075 on both coordinate kinds.
Its interval still includes zero before and after — the asymmetry inflated a
difference that was never distinguishable from zero.

### 6.2 The run's own checks

All nine assertions in `manifest.json` pass, including *"only
motor_complications moved"*, *"motor_complications interval still includes
zero"*, *"every rate interval still includes zero"* and *"three of four level
intervals still exclude zero"*.

### 6.3 The run's output table is superseded

`scripts/82` opens with **"SUPERSEDED — do not use its numbers."** It reads
the CV run's saved `outcome_table.csv`, which `scripts/64_clinical_profile_cv.py`
wrote with `.round(4)` while computing on the full-precision in-memory table.
Four decimal places is destructive for rate coordinates of order 10⁻³.

The authoritative table is
`experiments/20261103T220147Z_a2p_outcome_table_fullprec/h3_component_vs_subtype.csv`,
produced by a full rerun at HEAD with the fix in place and verified to
round-trip (P-20). Differences against the A-1 run's
`corrected_contrasts.csv`:

| Outcome | Kind | A-1 run | **Authoritative** | Difference |
|---|---|---|---|---|
| MCI | rate | 0.0428 (AUC 0.5922) | **0.0448 (AUC 0.5942)** | **−0.0020** |
| cognitive dx | rate | 0.0356 | 0.0355 | +0.0001 |
| motor complications | rate | 0.0334 | 0.0333 | +0.0001 |
| H&Y ≥ 3 | rate | CI high 0.1185 | CI high 0.1186 | −0.0001 |
| **All four level contrasts** | level | 0.1052 / 0.1605 / 0.0272 / 0.1194 | **identical** | **0.0000** |

### 6.4 The authoritative values

From `h3_component_vs_subtype.csv` (identical in the A-2 and A-2′ runs):

| Outcome | Kind | Best component | n | Events | Component AUC | Label AUC | Δ | 95% CI |
|---|---|---|---|---|---|---|---|---|
| H&Y ≥ 3 | level | `level_updrs2` | 761 | 127 | 0.6845 | 0.5794 | **+0.1052** | 0.0540 – 0.1557 |
| H&Y ≥ 3 | rate | `rate_updrs2` | 761 | 127 | 0.6325 | 0.5794 | +0.0532 | −0.0133 – 0.1186 |
| MCI | level | `level_moca` | 818 | 298 | 0.7098 | 0.5494 | **+0.1605** | 0.1093 – 0.2132 |
| MCI | rate | `rate_moca` | 818 | 298 | 0.5942 | 0.5494 | +0.0448 | −0.0054 – 0.0957 |
| Cognitive dx | level | `level_moca` | 816 | 327 | 0.6776 | 0.5582 | **+0.1194** | 0.0690 – 0.1687 |
| Cognitive dx | rate | `rate_updrs2` | 816 | 327 | 0.5937 | 0.5582 | +0.0355 | −0.0133 – 0.0849 |
| Motor complications | level | `level_brady_rigid` | 736 | 470 | 0.5309 | 0.5037 | +0.0272 | −0.0394 – 0.0941 |
| Motor complications | rate | `rate_updrs1` | 736 | 470 | 0.5370 | 0.5037 | +0.0333 | −0.0255 – 0.0902 |

## 7. Interpretation

**What the results support.**

* The orientation asymmetry was real, its effect is confined to the one
  outcome whose label AUC falls below chance, and correcting it *reduces* the
  reported advantage rather than increasing it.
* Three of the four **level** contrasts exclude zero after the fix: the
  profile's level coordinates discriminate later-recorded H&Y ≥ 3, MCI and a
  cognitive diagnosis better than the subtype label does.
* **No rate contrast excludes zero**, before or after the fix. The rate
  coordinates are not shown to discriminate these outcomes better than the
  label.

**What the results do not support.**

* The component is selected on the same data it is evaluated on. The run flags
  this as `selection_is_optimistic = True` on every row. The level contrasts
  are therefore upper bounds, not unbiased estimates.
* Nothing here says the profile predicts the outcomes well in absolute terms.
  An AUC of 0.53 for motor complications is close to useless; the finding is
  comparative.
* The outcomes are recorded later than the coordinates, but no survival model
  or time-to-event treatment is involved.

**On the superseded values.** The A-1 run is kept because it is what isolated
the fix. Its *conclusion* stands; its *numbers* must be taken from the A-2′
table. The one practical consequence is in §10.

**Exploratory or confirmatory.** The orientation fix was a correction, not a
hypothesis test. The outcome contrasts themselves are exploratory: four
outcomes, two coordinate kinds, best-of selection, no multiplicity adjustment.

## 8. Limitations

1. **Optimistic selection on every row.** No nested selection, no held-out
   component choice.
2. **No multiplicity adjustment** across eight contrasts.
3. **The run's own output is lossy** on rate coordinates; see §6.3.
4. **Reads a single CV run** whose provenance was later found defective (see
   the A-2 report). The values reproduce, but the chain of custody runs
   through that run.
5. **Outcome definitions are taken as given** from the source tables; their
   ascertainment is not audited here.
6. **Nothing is refit**, so the contrasts inherit every modelling choice of
   P-06.

## 9. Reproducibility

```bash
~/envs/ppmi/bin/python scripts/82_outcome_contrast_orientation_fix.py    # isolates the fix (superseded output)
~/envs/ppmi/bin/python scripts/64_clinical_profile_cv.py                 # authoritative table, at HEAD
~/envs/ppmi/bin/python scripts/84_verify_outcome_table_roundtrip.py RUN_DIR
```

| Kind | Path |
|---|---|
| Isolating script | `scripts/82_outcome_contrast_orientation_fix.py` |
| Library | `src/ppmi_subtype/clinical_profile.py` (`best_component_vs_subtype`) |
| A-1 run | `experiments/20261103T212122Z_a1_outcome_contrast_orientation/` |
| **Authoritative table** | `experiments/20261103T220147Z_a2p_outcome_table_fullprec/h3_component_vs_subtype.csv` |

Seed 42, 2,000 bootstrap draws, zero refits.

## 10. Manuscript relevance

Main text, as the outcome-discrimination result: the profile's level
coordinates beat the subtype label on three of four later-recorded outcomes,
and the rate coordinates beat it on none. It is the dissociation that
separates "the label is hard to predict" from "the underlying clinical state
is uninformative".

**Action taken on the IEEE manuscript — resolved 2026-11-04.**
`paper/ieee/make_assets.py` had been reading the **superseded** A-1 file. It
now reads `h3_component_vs_subtype.csv` from the A-2′ run, and the superseded
`A1` path constant has been removed. What changed on rebuild:

| | Before | After |
|---|---|---|
| `main.tex` rate differences | +0.053, **+0.043**, +0.033, +0.036 | +0.053, **+0.045**, +0.033, +0.036 |
| Table VI, MCI rate AUC | 0.592 | **0.594** |
| Table VI, MCI rate Δ (95% CI) | +0.043 (−0.008, +0.094) | **+0.045 (−0.005, +0.096)** |
| Table VI, motor complications rate CI low | −0.025 | **−0.026** |
| Figure 5(b), MCI rate bar | 0.5922 | **0.5942** |
| All four **level** rows | — | **unchanged** |

The paper rebuilds with no overfull boxes and 91 of 91 number checks passing.

## 11. Source evidence

| File | Contents |
|---|---|
| `experiments/20261103T212122Z_a1_outcome_contrast_orientation/before_after.csv` | §6.1 |
| `…/corrected_contrasts.csv` | The run's output — **superseded** |
| `…/drift_vs_october_run.csv` | Saved-vs-recomputed drift, including the −0.0020 on MCI rate |
| `…/manifest.json` | Nine checks, all passing |
| `experiments/20261103T220147Z_a2p_outcome_table_fullprec/h3_component_vs_subtype.csv` | **Authoritative values, §6.4** |
| `experiments/20261103T213859Z_a2_clinical_profile_cv_rerun/h3_component_vs_subtype.csv` | Identical to the above |
| `scripts/82_outcome_contrast_orientation_fix.py` | Implementation and the supersession notice |
| `src/ppmi_subtype/clinical_profile.py` | The orientation logic |
| `docs/manuscript/12_SUBMISSION_AUDIT_20261104.md` | Where finding A-1 was first recorded |
