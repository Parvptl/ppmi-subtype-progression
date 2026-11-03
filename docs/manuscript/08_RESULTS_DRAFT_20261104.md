# Stage 7 — Results, drafted

**Date** 2026-11-04 · **Target** npj Parkinson's Disease, Article · **Status**
first full draft

Every figure quoted below was re-read from its saved run immediately before
drafting, not carried forward from earlier text. No analysis is reported that
was not performed.

Conventions as in the Methods draft: **manuscript text is the numbered
sections**; **editorial notes are indented blockquotes** and are not manuscript
text.

---

## R1. Cohort and label construction

Of 4,788 participants and 19,450 visits in the curated data cut, 1,080
participants contributed 5,774 visits meeting the cohort criteria (Fig. 1). A
motor subtype could be assigned at 4,441 visits (76.9%); 934 participants had
at least two labelled visits and so contributed at least one prediction
opportunity, giving 3,361 consecutive labelled visit pairs. Table 1 gives
cohort characteristics. The analysed participants were slightly more
tremor dominant at baseline than the full cohort (72.5% against 71.7%) and
slightly less PIGD (16.8% against 18.1%).

At baseline, 772 participants (71.7%) were tremor dominant, 109 (10.1%)
indeterminate and 195 (18.1%) PIGD. These proportions agree closely with an
independent analysis of the same cohort by Simuni and colleagues, who reported
71%, 11% and 18% in 320 participants with one-year follow-up data. The
recomputed labels also reproduced PPMI's own derivations exactly where those
exist: 4,431 of 4,431 against the curated two-class label, and 1,595 of 1,597
(99.87%) against a three-class label in an archived release covering 390
participants.

> **Editorial.** The order here is deliberate. The internal agreement figures
> verify the implementation against PPMI's derivation of the *same rule from
> the same items* and are not independent; the Simuni comparison is, and should
> carry the weight. Do not let a later draft promote 100.00% / 99.87% to
> "validation".

---

## R2. The recorded label is unstable, and much of the change reverses

Across 3,361 transitions, tremor dominant was the most persistent state and
indeterminate by far the least (Fig. 2a, Table 2). A visit recorded tremor
dominant was followed by another tremor-dominant visit with probability 0.808
(95% CI 0.788–0.826); a PIGD visit by another PIGD visit with probability 0.702
(0.661–0.738). A visit recorded indeterminate remained indeterminate with
probability only 0.192 (0.148–0.234), returning to tremor dominant in 0.440
(0.381–0.500) of cases and moving to PIGD in 0.368 (0.318–0.417). PIGD was not
absorbing: 29.8% of PIGD visits were followed by a non-PIGD assessment.

**A third of observed changes reversed immediately.** Of 673 observed changes
of state, 35.2% (31.8–38.8) returned to the origin state at the following
visit (Fig. 2b). Reversion was concentrated in changes involving the
indeterminate band: tremor dominant to indeterminate reversed in 0.560
(0.478–0.642) of cases and PIGD to indeterminate in 0.500 (0.362–0.648), while
changes crossing the full width of the scale reversed less often — tremor
dominant to PIGD in 0.320 (0.253–0.388) and PIGD to tremor dominant in 0.393
(0.305–0.481).

The definition of a conversion therefore dominates its estimated frequency.
Among 885 participants not PIGD at their first labelled visit, 263 converted to
PIGD on a single-assessment definition but only 127 when conversion was
required to hold across two consecutive labelled visits — a reduction of 51.7%
(45.7–57.7). Kaplan–Meier cumulative incidence at five years was 35.3%
(30.8–39.9) under the single-visit definition against 15.3% (12.0–19.0) under
the sustained one.

Transition probabilities also scaled with the interval between visits
(Fig. 2c). Tremor dominant to tremor dominant fell from 0.820 (0.800–0.839) at
intervals of 1.5 years or less to 0.578 (0.432–0.719) beyond four years, while
tremor dominant to PIGD rose from 0.100 (0.087–0.116) to 0.289 (0.167–0.432).
A single pooled matrix therefore averages one-year and multi-year transitions
in whatever proportion attendance produced.

Finally, the classification was fragile to perturbations smaller than the
measurement's own resolution. Because the tremor score is a mean of eleven
integer ratings and the PIGD score a mean of five, few ratio values are
reachable inside a band 0.25 wide: half of indeterminate visits sat at a ratio
of 0.9091 and a further quarter at 1.1364, each about one hundredth from a
threshold. Moving each threshold by approximately 0.02 reclassified 298 of
4,441 labelled visits (6.7%), **all of them from the indeterminate class**,
which shrank by 67.6%, and reduced agreement with PPMI's two-class label from
100.00% to 97.74%.

> **Editorial.** The threshold result is currently a recomputation from the
> frozen panel rather than a saved run (O-4). Either run it or mark it as a
> post-hoc calculation in the caption.

---

## R3. Nine model specifications across five families converge

Under a single decision rule, a single evaluation set and a single baseline
(Methods M12a), nine fitted specifications spanning observed-label Markov
chains, latent-state hidden Markov models, a hierarchical subscore mixed model,
Gaussian processes on the subscores and a time-varying spline model fell within
**1.2 accuracy points of one another**, from 0.7453 to 0.7575 (Fig. 3,
Table 3). A parameter-free persistence rule — naming the currently recorded
label — achieved 0.7227.

All nine exceeded persistence, by between 0.0226 (0.0135–0.0316) and 0.0348
(0.0252–0.0450) accuracy points, every interval excluding zero. Neither
raising the number of latent states nor modelling elapsed time changed this
picture materially: a five-state hidden Markov model scored 0.7513 against the
three-state model's 0.7539, and the interval-aware chain scored 0.7453 against
the discrete chain's 0.7474.

**No fitted model recovered the indeterminate class.** Of fourteen
specifications, five ever predicted indeterminate at all, and four of those did
so between one and four times in 3,361 opportunities. Only the persistence
rule predicted it with any frequency — 334 times — and in consequence the
persistence rule achieved the **highest macro-F1 of any specification tested**
(0.5559, against 0.5074 for the best fitted model), because it was the only one
to recover any of the 332 indeterminate outcomes.

> **Editorial.** Two sentences of Methods belong near here in the final text:
> that models already producing a three-class probability vector reproduced
> their previously reported accuracy to 4 × 10⁻⁵, and that the harmonisation
> moved two families that had previously been recorded below persistence to
> above it. The second is a correction to the project's own earlier figures and
> should be stated, not quietly absorbed.

---

## R4. The advantage over the trivial rule lies entirely in one stratum

Stratifying the comparison by the label recorded at the prediction point
localised the whole effect (Fig. 4). For the three-state hidden Markov model,
**94.3% of its advantage over persistence came from the 334 transitions (9.9%
of the data) at which the current label was indeterminate.** Where the current
label was tremor dominant it gained six correct predictions out of 2,261; where
the current label was PIGD it gained **none at all** out of 766, the two
achieving identical accuracy of 0.7023.

The two approaches also made the same prediction almost always outside that
stratum: identical calls on 98.4% of current-tremor-dominant and 96.9% of
current-PIGD transitions, against 0.3% where the current label was
indeterminate. The discrete Markov chain made the point more starkly still,
predicting exactly what persistence predicted on **100%** of both outer strata
and taking all 83 of its gained cases from the indeterminate band. The subscore
Gaussian process was the only family to gain anywhere else, and gained nine and
eight cases respectively.

This localisation is consistent with the rest of the evidence. A chain in the
observed label is rejected by the data — among transitions whose current label
was tremor dominant, the probability of a PIGD outcome rose from 0.086 to 0.393
according to whether the previous label had been tremor dominant or PIGD
(χ² = 156.6, df = 4, *P* = 7.9 × 10⁻³³) — and adding a latent layer is strongly
supported on information criteria (ΔAIC 224, ΔBIC 186 against the best
alternative). Under the harmonised rule the latent-state model outscored the
interval-aware chain by 0.0086 accuracy points and the discrete chain by
0.0065, and that gain too sat in the same stratum: on transitions whose current
label was indeterminate it reached 0.488 against the discrete chain's 0.440,
where the observed-label chains are close to degenerate.

Taken together, the fitted models did not predict progression better than the
trivial rule. They declined to repeat a class that is rarely repeated, and that
accounts for essentially all of their advantage.

---

## R5. Restricting the target to two classes changes the task, not the model

Excluding the 332 transitions whose outcome was indeterminate raises the same
model's accuracy from 0.7539 to 0.8369 without altering a single prediction:
2,534 correct of 3,361 becomes 2,534 of 3,029, and renormalising the two
retained classes moves it to 0.8369. The increase is the removal of an outcome
the model never predicts, not an improvement in the model.

On this two-class task (Table 4) the hidden Markov model and the persistence rule were
**statistically indistinguishable on accuracy**: 0.8371 (0.8202–0.8540) against
0.8338 (0.8167–0.8509), paired difference +0.0033 (−0.0037 to +0.0100). The
continuous multi-domain representation reached 0.8404 (0.8248–0.8566), again
indistinguishable from the hidden Markov model (+0.0033, −0.0050 to +0.0112).

Within this two-class framing the latent layer was worth +0.0092 accuracy
(0.0003–0.0179) over an identically interval-aware chain in the observed label,
again concentrated where the recorded label is uninformative: +6.3 percentage
points on transitions whose current label was indeterminate.

The models were nonetheless separable from the trivial rule, and from each
other, on measures that use the probabilities rather than the thresholded call.
The hidden Markov model's two-class Brier score was 0.1215 against persistence's
0.1598, and its predicted probabilities tracked observed frequency to within
0.0065 on average across five bins spanning the probability range — persistence
emits only zero and one and cannot be calibrated at all. The multi-domain
representation discriminated better than the hidden Markov model by area under
the curve, 0.8902 (0.8748–0.9053) against 0.8583 (0.8411–0.8755), a paired
difference of +0.0319 (0.0229–0.0419).

> **Editorial — this caveat is mandatory wherever the AUC contrast appears.**
> The two models use different validation schemes, leave-one-participant-out
> against five-fold cross-validation, so the hidden Markov model is the more
> favourably validated of the two; matching cases removes the difference in
> which visits are scored but not the difference in scheme. The representation
> also sees the target visit's prescribed levodopa dose, measured as worth
> −0.0013 (−0.0056 to +0.0026). Both belong in the text, not only the
> limitations.

---

## R6. Neither more history nor retraining improves prediction

Restricting the hidden Markov model's prediction-time history to the two most
recent visits left performance indistinguishable from the full record on every
metric: accuracy −0.0041 (−0.0189 to +0.0100), balanced accuracy −0.0063, area
under the curve −0.0045, Brier +0.0051 and log loss +0.0112, all five intervals
containing zero. A single visit was measurably worse — accuracy −0.0218
(−0.0423 to −0.0014) and area under the curve −0.0617 (−0.0858 to −0.0385) —
so the second visit carries real information while the fourth and beyond do
not.

Refitting the model on two-visit sliding windows, so that training matched the
restricted input, did not help and degraded the probabilities: balanced
accuracy −0.0312 (−0.0441 to −0.0183) and area under the curve −0.0210
(−0.0301 to −0.0120) against the full-sequence model. A 2 × 2 design separated
the two factors and attributed the loss to the change in training (−0.0248
balanced accuracy with prediction history held fixed) rather than to the
history restriction (−0.0064, interval containing zero).

Prediction was also no more accurate later in follow-up than at the first
opportunity. Predicting the second visit from the first gave 0.8347 for the
hidden Markov model and 0.8455 for the multi-domain representation, against
pooled values of 0.8373 and 0.8406 across all stages; no change from stage one
excluded zero for either model at any stage. Balanced accuracy did rise with
follow-up, but so did that of the persistence rule, and the PIGD share of
outcomes rose from 23% to 48% over the same stages, so the trend reflects
cohort composition rather than anything the models learn.

---

## R7. A continuous representation carries information the label does not

The multi-domain representation was genuinely multidimensional rather than a
severity score in disguise: seven principal components were required for 90% of
its across-participant variance, with the first explaining 42.2% of the levels
and 37.2% of the rates (Fig. 5a).

Against outcomes recorded later and never used in fitting, single coordinates
of that representation discriminated better than the subtype label (Fig. 5b).
For Hoehn & Yahr stage 3 or worse, the best level coordinate achieved an area
under the curve of 0.685 against the label's 0.579, a difference of +0.105
(0.054–0.156); for mild cognitive impairment derived from verbal-learning and
visuospatial testing rather than from the MoCA, 0.710 against 0.549, a
difference of +0.161 (0.109–0.213). For motor complications, an outcome the
label itself discriminates at chance (0.504), the difference was +0.027
(−0.039 to +0.094).

**The rates were not established as informative on any outcome.** The best rate
coordinates gave differences of +0.053, +0.045, +0.033 and +0.036, every
interval including zero. The same pattern appeared on the representation's own
primary target: predicting the next visit's nine measurements improved on
carrying the last observation forward by +0.0733 log-density per domain
observation (0.0657–0.0813), of which the rate-of-change component supplied
+0.0040 (0.0019–0.0062) and the medication input +0.0058 (0.0041–0.0075) —
together under a sixth. Most of the gain was the level being filtered across a
participant's history rather than read off their most recent visit.

> **Correction applied 2026-11-04 (Stage 12 finding A1).** The
> motor-complications figures were previously +0.035 (level) and +0.041 (rate),
> computed against the subtype label's *raw* AUC of 0.4963 while the comparator
> coordinate was given its favourable orientation. The reference is now
> oriented the same way (0.5037), which is the conservative and internally
> consistent choice given that the comparator's selection is already an upper
> bound. Only this outcome moved; the other three have label AUCs above chance,
> where the two conventions coincide. See
> `experiments/20261103T220147Z_a2p_outcome_table_fullprec/h3_component_vs_subtype.csv`, a full rerun at HEAD
> carrying the fix. Only the motor-complications figures moved; every other
> quantity in that run reproduced the October values exactly, and this run's
> saved tables round-trip (Stage 12 A2').
>
> **Editorial.** Three caveats are mandatory wherever this section appears: the
> coordinate was chosen on the same data it is compared against, so each
> difference is an upper bound; the investigator cognitive diagnosis is not
> fully independent of the feature set; and the medication coefficients are
> confounded by indication and support no causal reading whatsoever.

---

## R8. Exploratory: performance varies by participant, but so does the trivial rule's

Weighting each participant equally rather than each visit changed the headline
little — mean per-participant accuracy 0.8342 (0.8166–0.8519) against a
visit-level 0.8391 — but the distribution was strongly bimodal rather than
concentrated near that mean. Of 898 participants, 572 (63.7%) were correct at
every eligible visit and 45 (5.0%) at none; the median per-participant accuracy
was 1.000. Few participants scored near the aggregate.

The dominant source of variation was whether a participant's recorded label
changed at all (Table 5). Participants whose eligible outcomes were all tremor dominant
averaged 0.944, and those with an even mixture of tremor-dominant and PIGD
outcomes averaged 0.469. **The persistence rule showed almost the same gradient
on the same groups — 0.970 and 0.480 — so this describes the label sequence
rather than model behaviour.** Per-participant PIGD sensitivity averaged 0.722
against a tremor-dominant specificity of 0.917 among participants with at least
two outcomes of the relevant class.

These groupings were defined after the predictions were generated and from
recorded labels only; the analysis is descriptive and is reported as
exploratory.

---

# §A — Verification

Every quantity above was re-read from a saved run immediately before drafting.

| Section | Source run or file |
|---|---|
| R1 | E-5 participant flow; `outputs/panel_summary.txt`; `outputs/validation.txt`; Table 1 |
| R2 | E-2 `transition_probabilities.csv`, `other_quantities.csv`; threshold recomputation |
| R3 | E-8 `harmonised_metrics.csv`, `paired_contrasts.csv` |
| R4 | E-8 `margin_decomposition.csv`; next-visit `contrasts.csv`; `markov_property_test.csv`; `aic_bic.csv` |
| R5 | `NOTE_accuracy_clarification.md`; `derived_label_vs_subtype_history.csv`; `auc_metrics.csv`, `auc_contrast.csv`; next-visit calibration |
| R6 | History-window, retraining and stage-wise `paired_comparisons.csv` / `stage_metrics.csv` |
| R7 | `representation_dimensionality.csv`, `outcome_associations.csv`, `contrasts.csv` |
| R8 | Patient-level audit `patient_accuracy_distribution.csv`, `patient_accuracy_by_subtype_composition.csv` |

**No unperformed analysis is reported.** Three quantities that exist in the
project's prose but not in any saved output are deliberately absent: the hidden
Markov model's advantage by history length ("+0.005 / +0.021 / +0.047"), the
claim that 790 participants begin treatment during follow-up, and the five-year
clinical output table, which is a latent-state quantity and is not quoted as a
statement about recorded labels.

# §B — Open items affecting Results

1. **O-4** — the threshold-sensitivity figures in R2 are a recomputation, not a
   saved run. Run it, or mark them post hoc in the caption.
2. **I18** — if any five-year projection is wanted in the text, it must be
   passed through the emission matrix first, and the initial distribution must
   be saved so the calculation is reproducible.
3. **C3** — no growth-model result is quoted in these Results. If the model is
   to appear beyond the supplementary material, it needs multi-chain
   diagnostics first.
4. The **Discussion** must carry the comparison with Simuni, Lee and
   von Coelln, including the cohort-composition explanation for von Coelln's
   lower long-horizon stability. It is not duplicated here.
