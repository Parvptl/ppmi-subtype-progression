# Individual patient-level accuracy audit of Experiments 6A and 6B

**Script** `scripts/75_patient_level_accuracy_audit.py`
**Run** `experiments/20261103T175003Z_patient_level_accuracy_audit`
**Sources** `experiments/20261103T052237Z_profile_prediction_history` (6A),
`experiments/20261103T072507Z_profile_two_visit_retraining` (6B)
**Date** 3 November 2026 · **Validation** 23 checks, 23 passed, 0 failed, 0 skipped

---

## 1. Executive summary

Experiments 6A and 6B reported **visit-level** accuracy: every eligible target
visit counted once, so a patient contributing eleven predictions carried eleven
times the weight of a patient contributing one. This audit regroups exactly the
same saved predictions by patient. No model was fitted and no prediction was
recomputed.

Four findings.

**The headline figure is not inflated by heavy contributors.** At full
prediction history the visit-level accuracy is **0.8391** and the mean
patient-level accuracy is **0.8342** (95% patient-bootstrap CI 0.8166 to
0.8519). The two estimands differ by 0.005. Weighting every patient equally
instead of every visit does not change the conclusion.

**But the aggregate does not describe a typical patient.** The distribution of
patient accuracy is strongly bimodal, not centred near 0.84. **63.7%** of
patients (572 of 898) are correct on *every* eligible visit and **5.0%** (45)
are correct on *none*. The median patient accuracy is **1.000**. Almost no
patient actually scores near 0.84; the mean sits in a sparse part of the
distribution (Figure 1). The honest statement is that the model is usually
right about a patient throughout, or wrong about them throughout.

**Retraining on two-visit windows does not improve patient-level accuracy, and
moves only a small minority of patients.** The primary comparison, D vs B,
gives a mean paired difference of **−0.0040** (CI −0.0157 to +0.0083) on 740
matched patients, and **660 of those 740 patients are completely unchanged**.
Of the 80 who move, 36 improve and 44 worsen. The same holds with full
prediction history (C vs A: **−0.0036**, CI −0.0125 to +0.0049, 821 of 898
unchanged). This reproduces the visit-level conclusion of 6B and adds that the
change is confined to roughly one patient in ten, with no consistent direction.

**Heterogeneity is driven by subtype composition, not by visit count.** Patients
whose eligible targets are all TD average **0.944**; patients with an even
mixture of TD and PIGD targets average **0.469**. Per-patient PIGD sensitivity
averages **0.550** against a TD specificity of **0.858**, and 32.5% of patients
with any PIGD target get none of them right. That last figure needs the caveat
the brief asks for: restricted to patients with **at least two** PIGD targets,
mean PIGD sensitivity is **0.722** and the zero-correct rate falls to **6.2%**.
Most of the apparent "complete failure" cases are patients with a single PIGD
target, where the per-patient rate is forced to be exactly 0 or exactly 1.

---

## 2. Research question

Is the aggregate accuracy reported by Experiments 6A and 6B representative of
individual patients, or is it carried by a subset of patients who contribute
many, easily predicted visits? And does two-visit retraining change accuracy
broadly across patients, or only for a few?

---

## 3. Data sources and experiment conditions

| | Experiment 6A | Experiment 6B |
|---|---|---|
| Source script | `scripts/73_profile_prediction_history.py` | `scripts/74_profile_two_visit_retraining.py` |
| Prediction file | `window_predictions.csv` | `condition_predictions.csv` |
| Rows in file | 14,966 | 20,668 |
| Eligible rows | 13,469 | 18,635 |
| Indeterminate target rows excluded | 1,497 | 2,033 |
| Source manifest re-verified | 12 / 12 files | 15 / 15 files |

**6A conditions** — last 1, last 2, last 3, last 5, full history, persistence.
**6B conditions** — A (full training / full history), B (full training / last 2),
C (overlapping-window training / full history), D (overlapping-window training /
last 2), C′ and D′ (the disjoint-window equivalents), persistence.

Eligible target counts per condition reproduce the source run's own
`cohort_counts.csv` exactly: 3,021 targets / 898 patients at full history,
2,184 / 740 at last 2, 1,489 / 481 at last 3, 734 / 228 at last 5. The 6A and
6B eligible target sets are **identical** for the conditions they share, which
was verified rather than assumed.

### Two schema points that had to be got right

*   **`y` is not an eligibility flag.** It is `(actual_next_label == "PIGD")`
    and is therefore 0 for Indeterminate targets as well as for TD targets.
    The saved files retain the Indeterminate rows. Using `y` as a filter would
    have silently scored 1,497 Indeterminate targets in 6A as TD. The project's
    rule — `actual_next_label in {TD, PIGD} and two_class_defined` — was
    re-applied instead.
*   **`visit_index` is not a row key.** Persistence rows carry no
    `visit_index`, because the rule does not run through `predict_sequence`.
    Keying on it reports 3,177 phantom duplicate rows in 6A. This audit uses
    the project's own key, `(PATNO, current_event, target_event)`, on which
    there are zero duplicates in either file. A dedicated check confirms every
    apparent `visit_index` duplicate is a persistence row.

---

## 4. Evaluation methodology

**Definitions.** *Visit-level accuracy* is correct eligible targets divided by
eligible targets across the cohort — what 6A and 6B reported. *Patient
accuracy* is, for one patient, their correct eligible targets divided by their
own eligible targets. *Mean patient-level accuracy* is the unweighted mean of
those patient accuracies: each patient counts once regardless of how many
visits they contribute. These are different estimands and are reported
separately throughout.

**Correctness** was recomputed from the stored `actual_next_label` and
`predicted_state` rather than taken on trust, then checked against the saved
`y` and `pred_bin` indicators. They agree on every eligible row in both files.

**Uncertainty** is a patient-level (cluster) bootstrap with 2,000 replicates and
seed 42: patients are drawn with replacement carrying all of their visits.
Individual visits are never resampled. A check confirms this empirically — the
patient count per replicate is fixed at 898 while the total number of visits in
a replicate ranges from 2,856 to 3,208, which can only happen if whole patients
are being drawn.

**Paired comparisons** are matched on both patients **and** target visits. The
eligible target sets nest exactly (last 5 ⊂ last 3 ⊂ last 2 ⊂ full history), so
for every comparison both conditions are scored on precisely the same visits of
precisely the same patients, and the two condition means in a paired row come
from the same patient draws. This is why, for example, the full-history mean in
the "last 2 vs full history" row (0.8315) differs from the full-history mean in
the descriptive table (0.8342): the paired row is restricted to the 740
patients and 2,184 targets that the last-2 condition can also score.

**A note on the median.** The median patient accuracy is 1.000 in every
condition, and its bootstrap interval is [1.000, 1.000]. That interval is
degenerate by construction, not by precision: more than half of all patients sit
at exactly 1.0, so every resample has a median of 1.0. The median is reported
because the brief asks for it, but it carries no information here and no
comparison should rest on it.

---

## 5. Individual patient accuracy results

### Experiment 6A, by condition (descriptive — cohorts differ)

| condition | patients | targets | visit-level acc. | mean patient acc. | 95% CI | median | SD | IQR |
|---|---|---|---|---|---|---|---|---|
| last 1 | 898 | 3,021 | 0.8252 | 0.8219 | 0.8030 – 0.8406 | 1.000 | 0.288 | 0.333 |
| last 2 | 740 | 2,184 | 0.8425 | 0.8395 | 0.8177 – 0.8604 | 1.000 | 0.288 | 0.250 |
| last 3 | 481 | 1,489 | 0.8321 | 0.8106 | 0.7829 – 0.8371 | 1.000 | 0.307 | 0.333 |
| last 5 | 228 | 734 | 0.8311 | 0.8105 | 0.7688 – 0.8475 | 1.000 | 0.295 | 0.333 |
| **full history** | **898** | **3,021** | **0.8391** | **0.8342** | **0.8166 – 0.8519** | **1.000** | **0.273** | **0.286** |
| persistence | 898 | 3,020 | 0.8338 | 0.8307 | 0.8129 – 0.8484 | 1.000 | 0.273 | 0.333 |

Minimum patient accuracy is 0.0 and maximum 1.0 in every condition. The median
number of eligible predictions per patient at full history is 3 and the maximum
is 12.

**These rows are not comparable across conditions**: each uses all patients
eligible in that condition, and the cohorts differ by design (898 patients at
full history, 228 at last 5). Section 6 gives the matched comparisons.

### Distribution shape (Experiment 6A, mutually exclusive bands, % of patients)

| condition | exactly 0% | 0–50% | 50–80% | 80–100% | exactly 100% |
|---|---|---|---|---|---|
| last 1 | 5.90 | 4.12 | 20.04 | 6.68 | 63.25 |
| last 2 | 6.62 | 3.11 | 16.49 | 5.54 | 68.24 |
| last 3 | 8.11 | 3.12 | 18.30 | 7.48 | 62.99 |
| last 5 | 7.46 | 2.63 | 21.93 | 7.89 | 60.09 |
| **full history** | **5.01** | **3.23** | **20.82** | **7.24** | **63.70** |
| persistence | 4.68 | 3.67 | 20.94 | 7.57 | 63.14 |

Bands are mutually exclusive and exhaustive, and sum to the patient count in
every condition (checked). The brief also names two overlapping cumulative
categories; at full history **8.24%** of patients fall below 50% accuracy and
**70.94%** reach 80% or above.

Figure 1 shows the shape directly: a spike of 572 patients at exactly 1.0, a
block of 45 at exactly 0.0, a secondary cluster around 0.5, and very few
patients anywhere near the 0.84 aggregate.

### Experiment 6B, by condition (descriptive)

| condition | patients | targets | visit-level acc. | mean patient acc. | 95% CI |
|---|---|---|---|---|---|
| A: full-trained, full history | 898 | 3,021 | 0.8408 | 0.8325 | 0.8156 – 0.8508 |
| B: full-trained, last 2 | 740 | 2,184 | 0.8411 | 0.8369 | 0.8146 – 0.8579 |
| C: window-trained, full history | 898 | 3,021 | 0.8381 | 0.8289 | 0.8108 – 0.8470 |
| D: window-trained, last 2 | 740 | 2,184 | 0.8379 | 0.8329 | 0.8103 – 0.8540 |
| C′: disjoint-window-trained, full history | 898 | 3,021 | 0.8408 | 0.8334 | 0.8161 – 0.8513 |
| D′: disjoint-window-trained, last 2 | 740 | 2,184 | 0.8388 | 0.8349 | 0.8117 – 0.8558 |
| persistence | 898 | 3,020 | 0.8338 | 0.8307 | 0.8129 – 0.8484 |

All seven confidence intervals overlap substantially.

---

## 6. Experiment 6A paired history comparisons

Matched patients and matched target visits. The difference is X − Y.

| comparison | patients | targets | mean acc. X | mean acc. Y | mean difference | 95% CI | excludes 0 | improved | worsened | unchanged |
|---|---|---|---|---|---|---|---|---|---|---|
| last 1 vs full history | 898 | 3,021 | 0.8219 | 0.8342 | **−0.0123** | −0.0225 to −0.0017 | **yes** | 46 | 78 | 774 |
| last 2 vs full history | 740 | 2,184 | 0.8395 | 0.8315 | +0.0080 | −0.0000 to +0.0164 | no | 41 | 30 | 669 |
| last 3 vs full history | 481 | 1,489 | 0.8106 | 0.8044 | +0.0062 | −0.0005 to +0.0135 | no | 23 | 14 | 444 |
| last 5 vs full history | 228 | 734 | 0.8105 | 0.8136 | −0.0031 | −0.0204 to +0.0126 | no | 5 | 7 | 216 |
| full history vs persistence | 898 | 3,020 | 0.8336 | 0.8307 | +0.0029 | −0.0078 to +0.0141 | no | 72 | 58 | 768 |
| last 2 vs persistence | 740 | 2,184 | 0.8395 | 0.8304 | +0.0092 | −0.0010 to +0.0202 | no | 44 | 27 | 669 |

The median paired difference is 0.0000 in all six comparisons, which simply
restates that the majority of patients are unchanged.

**Reading.** The patient-level picture agrees with the visit-level picture in
6A. Only the one-visit restriction is clearly worse, and it is the only
comparison whose interval excludes zero. Last 2 and last 3 are indistinguishable
from full history — if anything nominally above it, consistent with 6A's
visit-level accuracy differences of +0.0046 and +0.0060.

The striking number is the last column. Even for last 1 vs full history, **774
of 898 patients are completely unchanged**; the entire effect lives in 124
patients. Figure 3 plots only those movers, because on a full histogram the
zero bar flattens everything else to invisibility.

**A caution carried forward from the 6A report.** These accuracy comparisons do
not contradict 6A's class-specific finding. Restricting history reduced PIGD
sensitivity and raised TD specificity, and because TD targets outnumber PIGD
targets roughly two to one, that trade can leave overall accuracy flat or
nominally higher. Section 8 shows the same trade at patient level.

---

## 7. Experiment 6B paired retraining comparisons

| # | comparison | patients | targets | mean acc. X | mean acc. Y | mean difference | 95% CI | improved | worsened | unchanged |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **D vs B** (retraining, last-2 prediction) | 740 | 2,184 | 0.8329 | 0.8369 | **−0.0040** | −0.0157 to +0.0083 | 36 | 44 | 660 |
| 2 | **C vs A** (retraining, full history) | 898 | 3,021 | 0.8289 | 0.8325 | **−0.0036** | −0.0125 to +0.0049 | 34 | 43 | 821 |
| 3 | D′ vs B (disjoint retraining, last-2) | 740 | 2,184 | 0.8349 | 0.8369 | −0.0020 | −0.0120 to +0.0076 | 28 | 31 | 681 |
| 4 | C′ vs A (disjoint retraining, full history) | 898 | 3,021 | 0.8334 | 0.8325 | +0.0009 | −0.0056 to +0.0076 | 35 | 35 | 828 |
| 5 | B vs A (prediction history, full-seq training) | 740 | 2,184 | 0.8369 | 0.8293 | +0.0076 | −0.0002 to +0.0160 | 34 | 30 | 676 |
| 6 | D vs C (prediction history, overlapping training) | 740 | 2,184 | 0.8329 | 0.8283 | +0.0046 | −0.0024 to +0.0121 | 20 | 22 | 698 |
| 7 | D′ vs C′ (prediction history, disjoint training) | 740 | 2,184 | 0.8349 | 0.8338 | +0.0011 | −0.0039 to +0.0065 | 15 | 18 | 707 |
| 8 | D vs persistence (reference) | 740 | 2,184 | 0.8329 | 0.8304 | +0.0025 | −0.0088 to +0.0138 | 45 | 38 | 657 |
| 9 | B vs persistence (reference) | 740 | 2,184 | 0.8369 | 0.8304 | +0.0065 | −0.0036 to +0.0180 | 43 | 29 | 668 |

**Not one interval excludes zero**, and every median paired difference is
0.0000.

**Does retraining improve accuracy consistently across patients?** No. Both
primary retraining comparisons are nominally *negative* and both intervals
contain zero. More to the point, retraining is a no-op for the overwhelming
majority: 89.2% of patients are unchanged in comparison 1 and 91.4% in
comparison 2. Among those who do move, the split is close to even and slightly
unfavourable (36 improved vs 44 worsened; 34 vs 43). Disjoint-window retraining
behaves the same way with a smaller footprint — only 59 of 740 patients move in
comparison 3.

Overlapping and disjoint retraining are reported separately above, as the 6B
brief required, and neither is selected as "the" result on the basis of which
performed better.

Comparisons 5–7 isolate the prediction-history effect within each training
scheme. All three are nominally positive and none excludes zero, which is
consistent with 6A: two visits of prediction history are not measurably worse
than the full record on overall accuracy, under any of the three training
schemes.

---

## 8. Patient-level heterogeneity

### Is accuracy concentrated in a subset of patients?

Computed from `patient_metrics_by_condition.csv`, full-history condition:

*   The 572 patients at 100% accuracy hold 1,620 of the 3,021 eligible targets
    (**53.6%**).
*   The 45 patients at 0% accuracy hold 66 targets (**2.2%**).
*   The top decile by visit count — 90 patients — holds 803 targets (**26.6%**
    of all eligible targets) and has a visit-level accuracy of **0.8842**,
    above the cohort's 0.8391.
*   183 patients contribute exactly one eligible prediction; the mean is 3.36
    and the median 3.

So heavy contributors are somewhat easier than average, but not enough to
matter: re-weighting to one-patient-one-vote moves the headline by 0.005.

### Accuracy by number of eligible predictions (6A, full history)

| eligible predictions | patients | targets | mean patient acc. | visit-level acc. | % at exactly 100% | % at exactly 0% |
|---|---|---|---|---|---|---|
| 1 | 183 | 183 | 0.8306 | 0.8306 | 83.06 | 16.94 |
| 2–3 | 414 | 982 | 0.8390 | 0.8381 | 69.08 | 3.14 |
| 4–5 | 140 | 604 | 0.8118 | 0.8113 | 51.43 | 0.71 |
| more than 5 | 161 | 1,252 | 0.8454 | 0.8546 | 38.51 | 0.00 |

There is no monotone relationship between how many predictions a patient
contributes and how accurate those predictions are. What does change
monotonically is the *shape*: 83% of single-prediction patients are at exactly
100% and 17% at exactly 0%, against 38.5% and 0% among patients with more than
five. That is arithmetic, not signal — a patient with one prediction can only
score 0 or 1. **A patient with a single eligible prediction is not as reliable
an estimate of that patient's accuracy as a patient with eleven**, and the
bimodality in Figure 1 is partly this effect. Figure 5 shows the per-patient
scatter with the mean overlaid.

### Accuracy by subtype composition of the targets (6A, full history)

| composition | patients | targets | mean patient acc. | visit-level acc. | median targets/patient |
|---|---|---|---|---|---|
| all TD targets | 510 | 1,461 | 0.9444 | 0.9644 | 2 |
| majority TD | 105 | 600 | 0.6838 | 0.7167 | 5 |
| evenly split | 60 | 188 | 0.4685 | 0.5372 | 2 |
| majority PIGD | 61 | 333 | 0.5921 | 0.6486 | 5 |
| all PIGD targets | 162 | 439 | 0.8112 | 0.8633 | 2 |

This is the dominant source of heterogeneity, and it is far larger than
anything history length or retraining does. Patients whose record never changes
subtype are predicted well in both directions (0.944 for all-TD, 0.811 for
all-PIGD). Patients who cross the TD/PIGD boundary during follow-up are
predicted poorly, worst of all when the split is even (0.469 — below chance for
a two-class problem). The composition groups are defined from the **true**
target labels only.

One caution the brief asks for explicitly: this is a statement about where the
model's errors fall, not a clinical one. Nothing here licenses an inference
about disease progression or severity from prediction accuracy — "evenly split"
is a property of the observed label sequence, which is itself a thresholded
ratio, and the subtype rule's own instability near the boundary is a known
contributor (see `docs/model_comparison/04_INDETERMINATE_RATIO_SCRUTINY_20261024.md`).

---

## 9. PIGD and TD patient-level analysis

Per-patient rates. A patient with no eligible target of a class is marked **not
applicable** and excluded from that class's statistics — never assigned zero.
At full history, 510 of 898 patients have no PIGD target and 162 have no TD
target.

| condition | patients w/ PIGD targets | mean PIGD sens. | zero correct | % | patients w/ TD targets | mean TD spec. | zero correct | % |
|---|---|---|---|---|---|---|---|---|
| last 1 | 388 | 0.5236 | 132 | 34.0 | 736 | 0.8491 | 72 | 9.8 |
| last 2 | 335 | 0.5476 | 117 | 34.9 | 576 | 0.8855 | 46 | 8.0 |
| full history | 388 | 0.5503 | 126 | 32.5 | 736 | 0.8583 | 69 | 9.4 |
| persistence | 387 | 0.4679 | 146 | 37.7 | 736 | 0.8834 | 55 | 7.5 |
| A: full-trained, full history | 388 | 0.5446 | 129 | 33.2 | 736 | 0.8581 | 70 | 9.5 |
| B: full-trained, last 2 | 335 | 0.5418 | 120 | 35.8 | 576 | 0.8833 | 47 | 8.2 |
| D: window-trained, last 2 | 335 | 0.5508 | 115 | 34.3 | 576 | 0.8739 | 50 | 8.7 |

**The asymmetry is large and consistent.** Per-patient PIGD sensitivity averages
about 0.55 against a TD specificity of about 0.86, in every condition. Roughly a
third of patients with any PIGD target get none of them right, against under a
tenth for TD.

**Single-target patients inflate that "zero correct" figure, and it must be said
plainly.** A patient with one PIGD target scores exactly 0 or exactly 1 by
construction. Restricting to patients with **at least two** targets of the class:

| condition | patients ≥2 PIGD targets | mean PIGD sens. | % zero correct | patients ≥2 TD targets | mean TD spec. | % zero correct |
|---|---|---|---|---|---|---|
| last 1 | 225 | 0.6896 | 7.6 | 521 | 0.9135 | 1.2 |
| last 2 | 154 | 0.7171 | 5.8 | 335 | 0.9196 | 2.1 |
| full history | 225 | 0.7222 | 6.2 | 521 | 0.9170 | 1.5 |
| persistence | 225 | 0.6537 | 8.0 | 521 | 0.9332 | 0.8 |
| B: full-trained, last 2 | 154 | 0.7240 | 5.8 | 335 | 0.9187 | 2.1 |
| D: window-trained, last 2 | 154 | 0.7307 | 3.9 | 335 | 0.9145 | 1.8 |

Mean PIGD sensitivity rises from 0.550 to 0.722 and the zero-correct rate falls
from 32.5% to 6.2%. The class asymmetry survives — 0.72 against 0.92 — but the
claim "a third of PIGD patients are never identified" does not. It is an
artefact of per-patient rates computed on a single observation, and no
patient-level sensitivity estimate from a single target should be treated as
reliable.

**Retraining and PIGD.** D vs B at patient level moves mean PIGD sensitivity
from 0.5418 to 0.5508 (0.7240 to 0.7307 among patients with ≥2 PIGD targets).
This is the patient-level counterpart of 6B's visit-level finding of +0.0014
with an interval spanning zero: retraining does not recover the PIGD
sensitivity that restricted history costs. Figure 7 shows the full
distributions — both are U-shaped for PIGD, with roughly a third of patients in
the lowest bin and over 40% in the highest.

---

## 10. Limitations

1.  **This is a regrouping, not new evidence.** It uses the same fitted models,
    folds and predictions as 6A and 6B. It cannot discover anything those runs
    did not already contain; it can only change the unit of analysis.
2.  **Per-patient rates on few visits are noisy by construction.** The median
    patient has three eligible predictions, 183 have one, and per-patient
    accuracy can only take values k/n. The bimodality in Figure 1 is partly a
    real property of the model and partly this granularity. The two cannot be
    separated with these data.
3.  **The median patient accuracy is uninformative.** It is 1.000 everywhere
    with a degenerate bootstrap interval, because more than half of all patients
    are at exactly 1.0.
4.  **Validation remains what each source experiment used** — 5-fold patient
    cross-validation for both 6A and 6B. This audit does not alter it and does
    not make the two experiments' designs comparable to the CT-HMM's LOPO.
5.  **Patient subgroups here are not pre-specified strata.** The subtype
    composition and visit-count groupings were defined after seeing the
    predictions, as descriptive cuts. They should be treated as exploratory.
6.  **No clinical validation is claimed.** Nothing here establishes clinical
    utility, and a patient-level accuracy distribution says nothing about
    fairness or about generalisation to populations outside PPMI. The cohort is
    the project's frozen panel and the subtype labels are a thresholded ratio of
    MDS-UPDRS items, not an independently adjudicated diagnosis.

---

## 11. Conclusions

**Is the aggregate accuracy representative across patients?** In the narrow
sense, yes: mean patient-level accuracy (0.8342) is within 0.005 of visit-level
accuracy (0.8391), so the reported number is not an artefact of unequal visit
contributions. In the sense that matters for interpreting it, no: almost no
individual patient scores near 0.84. The model is right about 63.7% of patients
on every visit and wrong about 5.0% on every visit, and the aggregate is the
average of those two regimes rather than a description of typical per-patient
performance.

**Does retraining change accuracy broadly, or benefit a small subset?**
Neither, strictly — it changes very little for anyone. Two-visit retraining
leaves 89–91% of patients bit-identical, and among those it does move, slightly
more worsen than improve. Both primary comparisons are nominally negative with
intervals containing zero. This confirms 6B's visit-level conclusion at the
patient level and adds that there is no hidden subgroup for whom retraining
helps.

**What actually drives per-patient performance** is whether the patient's
subtype label changes during follow-up. Stable patients are predicted well
(0.944 all-TD, 0.811 all-PIGD); patients who cross the boundary are predicted
poorly (0.469 when evenly split). No variation in history length or training
scheme examined in 6A or 6B comes close to that in magnitude.

**For the manuscript**, the defensible phrasing is that mean patient-level
accuracy matches the reported visit-level accuracy, that the per-patient
distribution is strongly bimodal rather than concentrated near the mean, and
that per-patient PIGD sensitivity (0.72 among patients with at least two PIGD
targets) remains well below TD specificity (0.92) under every condition tested.

---

## 12. Validation and reproducibility

**23 checks defined, 23 passed, 0 failed, 0 skipped.** Full list in
`validation_checks.csv` and `validation_results.json`.

Sources and schema — both prediction files present; both source manifests
re-verified file by file (12/12 and 15/15); patient identifiers non-missing;
condition labels exactly the expected sets; no duplicate
`(PATNO, current_event, target_event, condition)` rows; every apparent
`visit_index` duplicate confirmed to be a persistence row.

Evaluation definitions — saved `y` agrees with the label recomputed from
`actual_next_label`; saved `pred_bin` agrees with the label recomputed from
`predicted_state`; correctness recomputes as their equality; Indeterminate
targets excluded from the binary cohort; eligible counts reproduce the source
run's `cohort_counts.csv`; 6A and 6B agree on the eligible target set they
share.

Patient-level arithmetic — per-patient denominators equal n_TD + n_PIGD;
denominators sum to the eligible target count in every condition; accuracy
bands mutually exclusive and summing to the patient count; patients without a
target of a class stored as NaN rather than zero; matched comparisons verified
to use identical patient *and* target sets.

Bootstrap and liveness — the bootstrap draws whole patients, demonstrated by a
fixed replicate patient count (898) against a varying replicate visit total
(2,856–3,208); intervals non-degenerate; and a liveness test in which flipping
all 12 predictions of one patient moves the mean patient accuracy from 0.834171
to 0.833428, confirming the metric reacts to the saved predictions rather than
being computed from something else.

Reproducibility — every distribution metric recomputes from the written CSV
rather than from memory, and a second identical analysis pass with the same seed
reproduces every table exactly.

**Provenance.** Python 3.13.16, numpy 2.2.4, pandas 2.2.3; 2,000 bootstrap
replicates, seed 42; git commit `3775a39`; analysis-only, zero models fitted.
`MANIFEST.json` and `MANIFEST.sha256` record all 17 generated files.

### Figures

| file | content |
|---|---|
| `fig1_patient_accuracy_full_history.png` | Distribution of patient accuracy at full history, with visit-level, mean and median reference lines |
| `fig2_patient_accuracy_by_history.png` | Patient accuracy across the six 6A conditions (descriptive; cohorts differ) |
| `fig3_paired_differences_6a.png` | Per-patient accuracy differences for the six 6A matched comparisons |
| `fig4_paired_differences_6b.png` | Per-patient accuracy differences for the nine 6B matched comparisons |
| `fig5_accuracy_vs_visit_count.png` | Patient accuracy against number of eligible predictions |
| `fig6_improved_worsened_unchanged.png` | Direction of change in every matched comparison |
| `fig7_patient_subtype_rates.png` | Per-patient PIGD sensitivity and TD specificity distributions |

Figures 3 and 4 plot only the patients whose accuracy changed, with the
unchanged count stated in each panel title: on a full histogram the zero bar
holds 80–95% of patients and hides everything else. The dashed line is the mean
over all matched patients, including the unchanged ones. No patient identifier
appears in any figure.
