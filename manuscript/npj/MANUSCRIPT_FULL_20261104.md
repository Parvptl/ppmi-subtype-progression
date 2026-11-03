# Label reliability rather than model capacity limits longitudinal prediction of Parkinson's disease motor subtypes

> **Assembled by `scripts/81_assemble_manuscript.py` on 2026-11-04 from the Stage 6–10 drafts.** Manuscript text only: every editorial note in those drafts is excluded here and remains in the draft it belongs to. Regenerate rather than edit this file.

**Authors.** Parv Patel, Mayank Tewatia, Sahasranand K. R. *(order and affiliations to be confirmed by the authors; see Author contributions below.)*

---

## Abstract

The tremor-dominant versus postural-instability-gait-difficulty subtype, a
thresholded ratio of two MDS-UPDRS item means, is widely used as a longitudinal
endpoint in Parkinson's disease. Among 1,080 Parkinson's Progression Markers
Initiative participants, 934 contributed 3,361 consecutive labelled
transitions. 35.2% of observed subtype changes reversed at the next visit. Nine
specifications spanning five model families, from Markov chains to Gaussian
processes, fell within 1.2 accuracy points of one another under one decision
rule, evaluation set and baseline. All exceeded a parameter-free persistence
rule by 2.3 to 3.5 points, but 94% of the latent-state model's margin arose on
the 9.9% of transitions recorded indeterminate; no fitted model recovered that
class, and persistence achieved the highest macro-F1. A continuous nine-domain
profile of the same examinations discriminated subsequently recorded Hoehn and
Yahr stage 3 or worse, and mild cognitive impairment, better than the label.
The binding constraint was label reliability, not model capacity.

---

## Introduction

Parkinson's disease is clinically heterogeneous, and the most widely used
summary of that heterogeneity is the motor subtype: patients are classified as
tremor dominant, postural-instability/gait-difficulty (PIGD), or indeterminate.
The classification is arithmetically simple. Eleven MDS-UPDRS items are
averaged into a tremor score, five into a PIGD score, and the ratio of the two
is compared against a pair of thresholds, with the interval between them
forming the indeterminate band¹. Its appeal is that it is cheap, interpretable
and computable from an examination that is performed anyway.

That label is used well beyond description. It stratifies cohorts, defines
subgroups for biomarker comparison, and is treated as a longitudinal quantity
whose change marks progression — a role that a recent international appraisal
of Parkinson's disease subtype classifications examined across many such
systems, and for which it found the evidence base thin². Whether the PIGD
construct identifies a distinct phenotype at all has been separately
questioned³.

Several groups have shown the label to be unstable over follow-up. In the de
novo PPMI cohort, 39% of patients classified PIGD and 18% classified tremor
dominant changed subtype within twelve months⁴. Over four years in the same
cohort, the tremor-dominant proportion fell from 68% to 45% while PIGD rose
from 23% to 44%⁵. Comparable instability has been reported for PIGD and
akinetic-rigid subtyping over time⁶, and in a clinic-based cohort followed for
a mean of 6.2 years only 15–36% of patients classified tremor dominant at
baseline were still so classified at follow-up, depending on which of three
published algorithms was applied⁷. Those algorithms do not agree with one
another: prevalence of the tremor-dominant class ranged from 5% to 32% in the
same patients⁷, and in de novo patients three subtyping systems showed only
fair agreement (κ ≈ 0.36–0.38)⁸.

What this literature establishes is that the label *changes*. What it does not
establish is **why**. Every one of those studies measures change as a
comparison between a baseline and a later assessment, or as a shift in
prevalence across a cohort. That design cannot distinguish a patient whose
phenotype genuinely progressed from one whose single noisy assessment crossed a
threshold and crossed back, because both appear as a changed subtype.
Separating them requires three consecutive assessments and a statistic that
asks how often a change is undone — a quantity we did not find reported.

The question matters because the label is also routinely used as a prediction
target. Recent work has learned latent disease states from many clinical
measurements in this cohort⁹, and has predicted data-driven progression
subtypes from baseline data with external replication¹⁰; multistate Markov
models that treat a recorded state as a noisy reading of a latent one, through
an explicit misclassification matrix, are long established¹¹. But predictive
studies of Parkinson's disease subtypes generally report discrimination without
calibration — a pattern not specific to this field, calibration having appeared
in 12% of prediction studies against 27% for discrimination in a systematic
review of high-impact general medical journals¹² — and we did not find a study
that reports what a trivial rule achieves on the same task. Without that
comparison, an accuracy figure on a label this persistent is difficult to
interpret.

We therefore asked a question about the measurement rather than about any
model: **is longitudinal prediction of the motor subtype limited by the choice
of model family, or by the reliability of the label being predicted?** Using
3,361 consecutive labelled visit pairs contributed by 934 of 1,080 eligible
PPMI participants, we quantified how much observed subtype change reverses at
the very next assessment, and how sensitive the classification is to
perturbations smaller than its own measurement resolution. We then compared
nine model specifications spanning five families — observed-label Markov
chains, latent-state hidden Markov models, a hierarchical subscore mixed model,
Gaussian processes on the subscores, and a time-varying spline model — under
one evaluation protocol, on one set of prediction opportunities, against one
parameter-free baseline.

This study proposes no new model. Its unit of contribution is a protocol and a
set of measurements, and its central result is a convergence: a finding about
what the data will support rather than about what any particular method can do.

---

## Results

### Cohort and label construction

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
independent analysis of the same cohort by Simuni and colleagues⁴, who reported
71%, 11% and 18% in 320 participants with one-year follow-up data. The
recomputed labels also reproduced PPMI's own derivations exactly where those
exist: 4,431 of 4,431 against the curated two-class label, and 1,595 of 1,597
(99.87%) against a three-class label in an archived release covering 390
participants.

### The recorded label is unstable, and much of the change reverses

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

### Nine model specifications across five families converge

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

### The advantage over the trivial rule lies entirely in one stratum

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

### Restricting the target to two classes changes the task, not the model

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

### Neither more history nor retraining improves prediction

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

### A continuous representation carries information the label does not

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

### Exploratory: performance varies by participant, but so does the trivial rule's

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

## Discussion

Nine model specifications drawn from five dissimilar families predicted the
next recorded motor subtype within 1.2 accuracy points of one another, and
essentially all of their shared advantage over a parameter-free rule arose on
the 9.9% of assessments at which the current label was indeterminate. Over the
same transitions, a third of observed subtype changes were reversed at the
following visit. Read together, these results locate the limit on longitudinal
subtype prediction in this cohort in the reliability of the label rather than
in the capacity of the models.

The convergence carries more weight than any one accuracy figure. Markov chains
in the observed label, hidden Markov models with a misclassification matrix¹¹,
a hierarchical subscore mixed model, Gaussian processes on the subscores and a
time-varying spline model embody different assumptions about what generates the
data, and agreement among them is therefore evidence about the data rather than
about a modelling choice. The dissociation sharpens it. The continuous
multi-domain representation has access to nine clinical domains and prescribed
levodopa dose, far more than the two subscores the label is computed from, and
on the two-class label task it was still statistically indistinguishable from
the hidden Markov model — yet the same representation discriminated later
clinical outcomes better than the label did. A model does not reach a ceiling
on one target while exceeding the label on another because it lacks
information. The more parsimonious reading is that the label itself is close to
the limit of what is predictable about it, which follows from its
construction: the subtype is a deterministic threshold on a ratio of two means,
so predicting it reduces to predicting where that ratio falls relative to two
cut points.

That is where the indeterminate band becomes the central object rather than a
nuisance category. It behaves not as a clinical state but as the neutral zone
of a noisy continuous quantity. It was the least persistent state by a wide
margin, it was the origin and destination of the changes most likely to
reverse, and it is where every fitted model found its advantage — not by
predicting it, since no fitted specification recovered it at any useful rate,
but by declining to repeat a class that is rarely repeated. Its fragility is
arithmetic as much as clinical. Because the tremor score averages eleven
integer ratings and the PIGD score five, few ratio values fall inside a band
0.25 wide; half of indeterminate visits sat at a single value one hundredth
from a threshold, and moving the thresholds by about 0.02 — less than the
spacing between attainable values — reclassified 6.7% of all labelled visits,
every one of them out of the indeterminate class. The constraint our
specification placed on the emission matrix was binding for this state alone,
which is the same fact expressed in the model's parameters. Whether the
construct identifies a distinct phenotype has been questioned on clinical
grounds³; these results add a measurement-level reason for the same caution,
confined to the indeterminate category.

Our estimates sit coherently within the published literature where the cohorts
are comparable. The baseline distribution reproduced an independent analysis of
the same cohort to within a percentage point in each class⁴. Long-horizon
stability agreed closely with the four-year PPMI estimate — 57.8% of
tremor-dominant visits followed by another beyond a four-year interval against
58.3% of baseline tremor-dominant patients still tremor dominant at four
years⁵ — and the two agree despite resting on different thresholds, since that
analysis used the older ≥1.5 / ≤1.0 cut points rather than the ones we
applied¹. The clinic-based estimate of 15–36% is substantially lower⁷, and
cohort composition is the likely explanation: PIGD prevalence in that cohort
was 59–75% against 25.0% of labelled visits here, describing a more advanced
population, and the range itself spans three algorithms whose disagreement is
documented⁷ and corroborated elsewhere⁸. What this literature had measured as
net drift between two assessments, we measured as immediate reversion, and the
two are not interchangeable: the same cohort yields a five-year cumulative
incidence of conversion to PIGD of 35.3% when a single assessment defines the
event and 15.3% when the change must hold across two consecutive assessments.

The constructive finding is that a continuous description of the same
examinations carried information the categorical label did not. Seven principal
components were needed to span 90% of the across-participant variance in the
multi-domain representation, so it is not a severity score in disguise, and
single coordinates of it discriminated subsequently recorded Hoehn & Yahr stage
3 or worse and derived mild cognitive impairment better than the subtype label.
Those differences are upper bounds, for reasons set out below, and the
rate-of-change coordinates were not established as informative on any outcome
tested. The direction is nonetheless consistent with the appraisal that found
the evidence supporting existing subtype systems thin², and with the
observation that prior predictive work in this cohort has largely targeted
latent or data-driven states rather than the threshold rule⁹˒¹⁰.

Three implications follow for studies that use this label. First, a conversion
endpoint defined on a single assessment will overstate incidence in this cohort
by roughly a factor of two relative to one requiring a sustained change, so the
definition belongs in the protocol rather than in the analysis. Second, a
pooled transition matrix is an attendance-weighted average of intervals that
behave differently — tremor-dominant persistence fell from 0.820 within
eighteen months to 0.578 beyond four years — and is not a property of the
disease. Third, a predictive model for this label should be reported against a
persistence baseline and with calibration, not discrimination alone¹²: a rule
that names the current label reached 0.723 accuracy here and attained the
highest macro-F1 of any specification we tested, because it was the only one
that ever predicted the indeterminate class. An accuracy figure quoted without
that comparison is close to uninterpretable. More generally, these results
support assessing the longitudinal stability of a subtype system before
adopting it as an endpoint, which the field-level appraisal found had not been
done for most such systems².

### Limitations

This is a single-cohort study and none of it has been replicated externally;
the comparisons above are with published figures, not with analyses we
performed on another cohort. The target is the project's own recomputation of
a published rule, not an adjudicated clinical diagnosis, so the reversion we
measure is reversion of a rule's output and includes both genuine phenotypic
fluctuation and rating variability, which this design cannot separate.
Analyses were complete-case, and label loss is not random: 89.6% of unlabelled
visits lacked an OFF-medication examination, which concentrates the loss in
treated participants later in follow-up. The ratio is undefined when the PIGD
score is zero, at 16.3% of labelled visits, and the specifications that model
the ratio drop those visits while those that model the subscores retain them.
Every two-class result excludes the indeterminate outcomes, and the resulting
rise in accuracy reflects the removal of an outcome the models never predict
rather than better prediction. All transition probabilities are conditional on
attending the next visit, and both the hidden Markov model and the
representation are given the interval to that visit, so neither addresses when
a patient will next be seen. Bounded rating scales were modelled as Gaussian;
the information criteria we report assume independent observations and so
favour the latent layer more than clustered data warrant; multiplicity is
largely uncorrected across the specifications compared. The comparison of the
representation against the hidden Markov model by area under the curve rests
on two different validation schemes, leave-one-participant-out against
five-fold cross-validation, which favours the hidden Markov model, and the
representation additionally sees the target visit's prescribed dose. The
outcome coordinates in the representation analysis were selected on the same
data against which they were compared, making each reported difference an upper
bound, and the cognitive outcome is not fully independent of the feature set.
Medication enters all of these models as an input measured in routine care and
confounded by indication; no coefficient on it supports a causal reading. The
per-participant analysis and its grouping by label composition were specified
after the predictions were generated and are reported as descriptive. Finally,
one internal agreement figure quoted in the cohort section cannot currently be
recomputed, because the archived release it was checked against is no longer
present in the repository.

Within those limits, the measurement rather than the model was the binding
constraint on predicting this label in this cohort. The practical consequence
is not that the tremor-dominant/PIGD distinction should be discarded, which
these results do not support, but that its use as a longitudinal endpoint
should rest on a sustained-change definition, an interval-aware treatment of
follow-up, and a reported comparison against the trivial rule — and that where
the question is progression rather than classification, a continuous
description of the same examination is the more promising object of study.

---

## Methods

### Study design

This was a retrospective secondary analysis of a prospective observational
cohort. The analysis population, the subtype definition, the modelling and the
validation protocol were all fixed before the comparison reported here was run,
and every model was scored on one common set of prediction opportunities.

The primary objective was to determine whether the predictability of the next
recorded motor subtype is limited by the choice of model family or by the
reliability of the label itself. Secondary objectives were to quantify how much
observed subtype change reverses at the next assessment, and to assess whether
a continuous multi-domain clinical representation carries prognostic
information the label does not.

### Dataset and participants

Data came from the Parkinson's Progression Markers Initiative (PPMI), obtained
from the LONI Image and Data Archive under the PPMI Data Use Agreement. Three
sources were used: the PPMI Curated Data Cut (public release 2026-05-11), which
supplies the visit skeleton and covariates; the item-level MDS-UPDRS Part II
(patient questionnaire) and Part III (motor examination) files, which supply
the sixteen ratings from which the subtype is derived; and the participant
status file.

Participants were included if the curated cut assigned them to the sporadic
Parkinson's disease subgroup. Visits were excluded if they were unscheduled or
premature-withdrawal encounters (`EVENT_ID` in ST, PW, U01, U02) or if they
fell beyond fifteen years of follow-up. Of 4,788 participants and 19,450 visits
in the curated cut, 1,080 participants and 5,789 visits met the subgroup
criterion; 15 visits were then removed as unscheduled or premature withdrawal
(14 and 1 respectively) and none exceeded the follow-up cap, leaving **1,080
patients and 5,774 patient-visits**.

Of these, 4,441 visits (76.9%) could be assigned a motor subtype. The 1,333
that could not are examined in M4. Patients with at least two labelled visits —
**934 of 1,080** — contributed at least one prediction opportunity, giving
**3,361 consecutive labelled visit pairs**. Figure 1 gives the full flow with
every exclusion counted.

### Clinical variables and subtype definition

Motor subtype was computed per visit following Stebbins et al., as the ratio of
two means of MDS-UPDRS items.

The **tremor score** is the mean of eleven items: 2.10 (tremor), 3.15a and
3.15b (postural tremor, right and left hand), 3.16a and 3.16b (kinetic tremor,
right and left hand), 3.17a–e (rest tremor amplitude: right and left upper
extremity, right and left lower extremity, lip/jaw) and 3.18 (constancy of rest
tremor). The **PIGD score** is the mean of five items: 2.12 (walking and
balance), 2.13 (freezing), 3.10 (gait), 3.11 (freezing of gait) and 3.12
(postural stability).

A visit was classified tremor dominant if the ratio was at least 1.15, PIGD if
it was at most 0.90, and indeterminate between those bounds; the band is open
at both ends, so a ratio falling exactly on a threshold is assigned to the
outer class. Two degenerate cases are handled explicitly rather than by
division: a visit with a PIGD score of zero and a positive tremor score is
tremor dominant, and a visit with both scores zero is indeterminate.

Three data-handling rules materially affect the result and are stated in full.
First, PPMI records "unable to rate" as the sentinel value **101 in the same
numeric field as a 0–4 rating**; every value outside the valid 0–4 range was
blanked before any mean was taken. This is not a rounding concern — one such
value among the five PIGD items lifts their mean to roughly 20 and reclassifies
a tremor-dominant visit as PIGD. In the Part III file, item 3.12 alone carries
4,302 such values, 3,206 of them in examinations eligible for this analysis.
Second, a subscore was set to missing if **any** constituent item was missing,
matching PPMI's own derivation; relaxing this to allow one missing item per
subscore recovers 44 additional visits and changes no existing label. Third,
the classification uses the **OFF-medication or untreated** examination:
examinations recorded in the ON state in a treated participant were excluded,
since levodopa suppresses tremor and scoring them would manufacture apparent
PIGD transitions that are pharmacological rather than pathological. Where a
visit carried more than one eligible Part III examination, the record with the
fewest missing items was retained, ties broken by order of appearance.

Recomputed labels were compared against PPMI's own two derivations. Agreement
was 4,431 of 4,431 (100.00%) against the curated cut's two-class `td_pigd`, and
1,595 of 1,597 (99.87%) against a three-class label in an archived 2020 release
covering 390 patients.

### Missing labels

Labels were not imputed; analyses used complete cases. Because the pattern of
loss is not random, its mechanism is reported rather than assumed.

Of the 1,333 visits without a subtype, **1,195 (89.6%)** had only an
ON-medication Part III examination and were therefore excluded by the
medication-state rule; 65 had no Part III record; 57 had an examination that
was recorded but incomplete; and 16 had a complete Part III examination but an
incomplete Part II. Among the 57 incomplete examinations, item 3.12 (postural
stability) was the missing item in 54, and those examinations came from
participants at a more advanced stage (median Hoehn & Yahr 3.0 against 2.0 for
complete examinations, Mann–Whitney *P* = 6.4 × 10⁻¹⁸).

The dominant mechanism is therefore treatment-related. Unlabelled visits were
overwhelmingly from treated participants later in follow-up: 96.8% were on
antiparkinsonian medication against 57.9% of labelled visits, median
levodopa-equivalent daily dose 500 mg against 300 mg, and median 5.0 years from
baseline against 2.0. The same pattern appears independently in the curated
cut's own columns, where the OFF-state Hoehn & Yahr and Part III total are
recorded at 6.5% and 3.5% of unlabelled visits against 99.8% and 99.4% of
labelled ones. The analysed sample is consequently weighted toward earlier,
untreated follow-up, which attenuates rather than exaggerates observed
progression.

### Longitudinal construction and observed transitions

Visits were ordered within participant by elapsed time from that participant's
first visit, computed from the released month-and-year visit date. Unlabelled
visits were removed before pairing, so a gap year does not break a
participant's sequence; a transition is therefore a pair of **consecutive
labelled** visits, carrying the actual elapsed interval between them.

Transition counts and probabilities were computed over all 3,361 pairs. Because
one participant contributes several correlated pairs, every interval reported
for a transition quantity is a **patient-level cluster bootstrap**: participants
were drawn with replacement carrying all of their visits, and every statistic
was recomputed from scratch in each replicate, so the number of participants
qualifying for a given statistic varies between replicates as it would in a new
sample.

Two further quantities were derived. **Reversion** was computed over labelled
triples in which the middle step was an observed change, as the proportion
returning to the origin state at the following visit. **Time to PIGD
conversion** was estimated by Kaplan–Meier among participants not already PIGD
at their first labelled visit, under two definitions: a single PIGD assessment,
and PIGD sustained across two consecutive labelled visits.

### Observed-label Markov chains

Two chains treating the recorded label as the state were fitted as references.
The **discrete-time chain** estimates a row-stochastic transition matrix from
observed pairs, ignoring the elapsed interval. The **continuous-time chain**
estimates a generator *Q*, with the transition matrix over an interval *t*
given by exp(*Qt*); off-diagonal rates were parameterised as exp(θ) and fitted
by maximum likelihood.

Whether the recorded label satisfies the Markov property was tested directly on
labelled triples, comparing the distribution of the next label across possible
previous labels within each current label (χ² test).

### Continuous-time hidden Markov model

A three-state latent phenotype *X*(*t*) was modelled as a continuous-time
Markov chain with generator *Q*, so that transition probabilities over an
elapsed interval *t* are exp(*Qt*), evaluated by eigendecomposition. What is
observed at a visit is not *X* but a noisy reading related to it by a
misclassification matrix *E*, where *E*[*a*, *b*] is the probability that a
participant whose true state is *a* is recorded as *b*. This is the multistate
Markov model with classification error of Jackson et al.¹¹, applied to the motor
subtype. The likelihood of a participant's whole label sequence marginalises
over every latent path by the forward algorithm.

Off-diagonal rates were parameterised as exp(θ) and emission rows by softmax,
with one entry per row pinned for identifiability; the initial distribution was
a free softmax parameter estimated jointly. The model has 14 free parameters
(six rates, six emission, two initial). Estimation was by maximum likelihood
using L-BFGS-B from four starts — one deterministic, three perturbed — with
seed 42.

**One constraint is active and must be read carefully.** Each state was
required to be recorded correctly at least half the time (diagonal of *E* ≥
0.5). Without it the maximum-likelihood solution is degenerate and reproducible
across restarts: the latent indeterminate state emits uniformly and PIGD
becomes absorbing at the parameter boundary. In the fitted model the diagonal
entries are 0.941 for tremor dominant, 0.842 for PIGD, and **exactly 0.500 for
indeterminate** — that is, the constraint is binding for the indeterminate
state, and 0.500 is the constraint rather than an estimate. The latent
indeterminate state carries no identified emission information.

Prediction is by **filtering**: the distribution over the latent state given
labels up to and including visit *k* is propagated over the actual elapsed time
to visit *k* + 1 and pushed through *E*, giving a distribution over the next
recorded label. No backward pass is run, so no prediction uses the visit it
predicts.

A five-state variant was fitted identically as a sensitivity analysis; with
more latent states than observed labels, the emission floor is undefined and
was not applied.

### Hierarchical Bayesian growth model *(supplementary)*

The tremor/PIGD ratio was modelled as a linear function of time with
participant-level random intercept and slope, the specification given in the
project's original proposal. Observation and process error enter additively on
the same observation, so only their sum is identified; the sampler estimates
that sum and the manuscript reports it as a single residual variance rather
than a split the data cannot support.

Inference was by Gibbs sampling with conjugate updates throughout: a normal
prior with standard deviation 100 on the population coefficients, and
inverse-gamma(0.01, 0.01) on each variance component. The reported posterior
uses 4,000 sweeps after 1,000 discarded as burn-in, thinned by two, giving
2,000 retained draws.

### Subscore models *(supplementary)*

Three families were fitted to the two subscores directly rather than to their
ratio, which avoids the undefined-ratio exclusion.

A **joint linear mixed-effects model** fitted tremor and PIGD scores together
with participant-level random effects and a shared residual covariance. A
**Gaussian process regression** modelled each participant's subscore
trajectory, with a constant-plus-RBF kernel selected from four candidates by
held-out mean absolute error; an independent-output variant and a multi-output
variant with a fitted cross-output covariance were both fitted. A **time-varying
linear mixed model** replaced the constant population slope with a linear
spline with knots at two and five years, chosen a priori rather than estimated.

For each, the predicted subtype was obtained by passing the predictive
distribution over the two subscores through the project's unmodified
classification rule (M12), so no new classifier and no new threshold is
introduced anywhere.

### Continuous multi-domain clinical representation *(supplementary)*

Nine clinical domains were tracked per visit: a bradykinesia/rigidity subscore
built from the twenty Part III items that are neither tremor nor PIGD
constituents; the tremor and PIGD subscores; MDS-UPDRS Parts I and II totals;
MoCA; GDS-15; SCOPA-AUT; and the Epworth Sleepiness Scale. A visit entered if
it carried at least eight of the nine; 97.6% carried all nine.

Each domain was given a continuous local linear trend: a level and a rate, with
the exact integrated-Brownian innovation covariance over an interval Δ,

    Cov([w_level, w_rate]) = [[σ²_L·Δ + σ²_R·Δ³/3,  σ²_R·Δ²/2],
                              [σ²_R·Δ²/2,           σ²_R·Δ  ]]

Both variance terms are required: with only the integrated term the model
cannot represent a level that wanders without a trend and so would not nest the
random walk, and with σ²_R = 0 it reduces exactly to a random walk.
Levodopa-equivalent daily dose entered the observation equation as an input
with a fitted per-domain coefficient, not as a tenth domain. Medication is
highly dynamic in this cohort by protocol: 1,065 of 1,080 participants (98.6%)
have a recorded dose of zero at their first visit, and 891 have a non-zero dose
at some later visit. Including it as a domain would therefore encode "has
started treatment" as part of the clinical state. The model has 45 free parameters: an observation
variance, a level innovation variance, a rate innovation variance, an initial
rate prior variance and a medication coefficient for each of the nine domains.

Inference is by Kalman filter and smoother, with parameters shared across
participants and fitted on training participants only. The smoother is used
only for fitting; every prediction comes from the filter, so temporal causality
is a structural property of the recursion rather than a convention. Missing
measurements are marginalised exactly — with a diagonal observation covariance
the emission log-density is a sum over domains, so an unrecorded domain simply
contributes no term. Nothing is imputed.

### History-window and retraining analyses *(supplementary)*

Two questions about history were addressed separately, because conflating them
confounds the estimator with the input.

For the **history-window** analysis, model parameters were fitted once per fold
on training participants' full sequences and held fixed across conditions;
only the held-out participant's prediction-time history was truncated, to the
most recent one, two, three or five visits. Truncation rebuilds the filtered
state from the retained visits alone — a full-sequence filter is never computed
and then cut — and retains the original visit dates and intervals. Because
cohorts differ between window widths, inference used a common-target cohort
eligible under every condition.

For the **retraining** analysis, the model was additionally refitted on
two-visit sliding windows of the training sequences, in a complete 2 × 2 of
training set by prediction history. Overlapping windows make the objective a
composite rather than a true likelihood — each interior visit appears twice —
so a disjoint-window variant was fitted as a sensitivity, and the objective
value is not compared across training regimes.

### Deriving a reported prediction

#### Harmonised three-class comparison

Model families had been scored under different rules for deriving a subtype
from a prediction, and that choice is worth more than the differences between
the families: the same model and the same saved predictions move 2.7 accuracy
points on the derivation rule alone, and a baseline moves 2.1. The primary
comparison therefore re-scored every family onto a single footing.

Each model was expressed as a predictive probability vector over the three
labels on the same 3,361 transitions. Chains and hidden Markov models already
produce one. For subscore models the predictive distribution over tremor and
PIGD was propagated by Monte Carlo through the project's unmodified
classification rule: 10,000 draws per prediction from the bivariate predictive,
using the fitted cross-output covariance where available and independence
otherwise, clipped at zero because a subscore cannot be negative, each draw
classified and the resulting label frequencies taken as the probability vector.
The predicted class is the argmax of that vector for every model without
exception, and the canonical reference is **label persistence** — naming the
currently recorded label — which requires no subscores and no fitting.

Monte Carlo draws are shared: the same standard normal variates are used for
the same prediction under every model, so Monte Carlo error is common and
largely cancels in paired contrasts. Halving the number of draws changes
accuracy by at most 0.0015. Models already producing a probability vector
reproduce their previously reported accuracy exactly (maximum absolute
difference 4 × 10⁻⁵), which verifies the re-scoring.

#### Two-class secondary analysis

A secondary analysis restricted the target to tremor dominant against PIGD.
Two **conditional** probabilities were reported,

    P_TD = p_TD / (p_TD + p_PIGD),   P_PIGD = p_PIGD / (p_TD + p_PIGD)

computed only where the denominator exceeded 1 × 10⁻¹². These are conditional
on the next recorded label being tremor dominant or PIGD and are **not**
unconditional next-visit probabilities; the indeterminate mass is divided out,
not discarded, and is retained in the output. The predicted class is tremor
dominant where P_TD exceeds P_PIGD and PIGD otherwise; an exact tie resolves to
PIGD, the clinically costlier error to miss. The tie rule is deterministic and
independent of the fit, and no tie occurred in any of the 3,029 scored
predictions under any model.

Indeterminate is never recoded. It remains a third observed label throughout
the input history — the latent models emit it, the chains transition through
it. Only the *target* is restricted. The 332 transitions whose next recorded
label is indeterminate (9.9%) are reported separately and **never scored**:
the conditioning event did not occur, so neither class is the right answer, and
any rule for scoring them would be a decision about the label rather than a
measurement of the model. Headline two-class figures therefore describe 90.1%
of prediction opportunities.

### Validation

All models in the primary comparison were validated by **leave-one-participant-out**:
each participant was held out in turn, every model refitted on the remaining
933, and the held-out participant's transitions predicted in sequence. A
participant's visits were never split across training and test. Predictions for
a visit use only that participant's earlier visits.

Two analyses reported alongside the primary comparison use different schemes
and are not pooled with it. The continuous multi-domain representation and the
two-class next-visit comparison used **five-fold participant-level
cross-validation**; every learned quantity — model parameters, the
standardiser, the population mean and the last-observation spread — was
refitted inside each fold on training participants only. Earlier work using a
single stratified 70/30 participant split, stratified on baseline subtype, is
reported only where stated.

No hyperparameter was tuned on held-out data. The emission floor, restart count
and seed were fixed in configuration in advance; the discriminative reference
model's penalty was fixed at 1.0 and never searched.

### Statistical analysis

Accuracy is the proportion of predictions matching the recorded next label.
The multiclass Brier score is the mean squared difference between the predicted
probability vector and the one-hot outcome. Log loss is the mean negative log
probability assigned to the realised class. Macro-F1 is the unweighted mean of
per-class F1. For two-class analyses, discrimination was summarised by the area
under the receiver operating characteristic curve and calibration by the mean
absolute difference between predicted probability and observed frequency within
bins.

**Every interval is a percentile cluster bootstrap that resamples
participants**, not predictions, because one participant contributes several
correlated rows; 2,000 replicates, seed 42, no bias correction. Head-to-head
comparisons are **paired on identical held-out cases**, with one participant
resample shared by both conditions within each replicate. A difference whose
interval includes zero is not described as an improvement.

For the exploratory per-participant analysis, participants were grouped by the
subtype composition of their own eligible targets — all tremor dominant,
majority tremor dominant, evenly split, majority PIGD, or all PIGD. Groups were
defined from the **recorded labels only**, never from model output, and were
defined after the predictions were generated; the analysis is descriptive and
is reported as exploratory throughout.

Where several comparisons were made within one analysis, the pre-specified
primary comparison is identified and the remainder are reported as secondary or
exploratory; no multiplicity correction was applied, and the corresponding
Bonferroni threshold is stated so that readers can apply one.

### Software and reproducibility

All analyses were implemented in Python 3.13.16 using numpy 2.2.4, scipy
1.15.2 and pandas 2.2.3, with matplotlib 3.10.1 for figures. No machine-learning
library was used; the hidden Markov model, the continuous-time chains, the
Gibbs sampler, the Gaussian processes and the Kalman filter are implemented
directly on numpy and scipy. Random seeds are fixed at 42 throughout and
recorded with every run.

Analysis parameters — the sixteen item identifiers, the 1.15 and 0.90
thresholds and the side each is closed on, cohort membership, medication state,
the missing-item policy, the sentinel codes, interval bins, the emission floor,
restart counts and split ratios — are held in a single configuration file and
validated on load; none is hard-coded. The implementation is covered by 758
automated tests exercising the classification rule and its edge cases,
transition pairing and survival estimation, generator recovery from simulated
data, latent-state recovery, Gibbs sampler parameter recovery and label
remapping.

Each analysis run writes a manifest of SHA-256 digests over its own outputs
together with the environment, the git commit, the random seed and the result
of every internal validation check; runs exit non-zero if any check fails.

---

## Data availability

Data were obtained from the Parkinson's Progression Markers Initiative (PPMI) database and are available to qualified researchers from the PPMI repository at the Laboratory of Neuro Imaging (LONI) under PPMI's Data Use Agreement. The authors are not permitted to redistribute participant-level data. The analysis operates on a frozen visit panel whose checksum is recorded in the repository so that a reader with PPMI access can reconstruct the identical panel.

> **TO BE COMPLETED BEFORE SUBMISSION.** The current PPMI acknowledgement wording and the DUA reference are not in the repository (Stage 1 C5); `report/CARE_AI2026_Abstract.tex` carries a `% TODO` for exactly this.

## Code availability

> **UNRESOLVED — DECISION REQUIRED.** The pipeline, the Markov and hidden Markov models, the subscore models and every analysis script in `scripts/` are version-controlled. The Gaussian-process modules (`gpr.py`, `gpr_loo.py`, `gpr_plots.py`, `trajectory.py`, scripts 11–16 and three test files) are **untracked and were never pushed**, and the primary harmonised comparison depends on the subscore Gaussian process. A code-availability statement cannot cover the full analysis until those files are committed.

## Ethics declarations

> **ABSENT (Stage 1 C5).** No ethics approval, consent statement or institutional position on secondary analysis of PPMI exists anywhere in the repository. Required before submission.

## Funding

> **ABSENT (Stage 1 C5).** Required before submission.

## Competing interests

> **ABSENT (Stage 1 C5).** Required before submission.

## Author contributions

> **ABSENT (Stage 1 C5).** A CRediT statement must be supplied by the authors. Author list above is taken from the CARE AI 2026 abstract and is unconfirmed.

## Acknowledgements

> **ABSENT (Stage 1 C5).** PPMI requires specific acknowledgement wording, including its sponsor and funding-partner list. Not in the repository.

## References

1. **Stebbins GT, Goetz CG, Burn DJ, Jankovic J, Khoo TK, Tilley BC.** How to identify tremor dominant and postural instability/gait difficulty groups with the Movement Disorder Society Unified Parkinson's Disease Rating Scale: comparison with the Unified Parkinson's Disease Rating Scale. *Mov Disord* 2013;28(5):668–670. doi:10.1002/mds.25383
2. **Mestre TA, Fereshtehnejad SM, Berg D, et al.** Parkinson's disease subtypes: critical appraisal and recommendations. *J Parkinsons Dis* 2021;11(2):395–404. doi:10.3233/JPD-202472
3. **Kotagal V.** Is PIGD a legitimate motor subtype in Parkinson disease? *Ann Clin Transl Neurol* 2016;3(6):473–477. doi:10.1002/acn3.312
4. **Simuni T, Caspell-Garcia C, Coffey C, Lasch S, Tanner C, Marek K.** How stable are Parkinson's disease subtypes in de novo patients: analysis of the PPMI cohort? *Parkinsonism Relat Disord* 2016;28:62–67. doi:10.1016/j.parkreldis.2016.04.027. PMID:27132498
5. **Lee JW, Song YS, Kim H, Ku BD, Lee WW.** Alteration of tremor dominant and postural instability gait difficulty subtypes during the progression of Parkinson's disease: analysis of the PPMI cohort. *Front Neurol* 2019;10:471. doi:10.3389/fneur.2019.00471. PMID:31133973
6. **Erro R, Picillo M, Amboni M, et al.** Comparing postural instability and gait disorder and akinetic-rigid subtyping of Parkinson disease and their stability over time. *Eur J Neurol* 2019;26:1212–1218. doi:10.1111/ene.13968
7. **von Coelln R, Gruber-Baldini AL, Reich SG, Armstrong MJ, Savitt JM, Shulman LM.** The inconsistency and instability of Parkinson's disease motor subtypes. *Parkinsonism Relat Disord* 2021;88:13–18. doi:10.1016/j.parkreldis.2021.05.016. PMID:34091412
8. **Ren J, Pan C, Li Y, Li L, Hua P, Xu L, Zhang L, Zhang W, Xu P, Liu W.** Consistency and stability of motor subtype classifications in patients with de novo Parkinson's disease. *Front Neurosci* 2021;15:637896. doi:10.3389/fnins.2021.637896. PMID:33732106
9. **Severson KA, Chahine LM, Smolensky LA, Dhuliawala M, Frasier M, Ng K, Ghosh S, Hu J.** Discovery of Parkinson's disease states and disease progression modelling: a longitudinal data study using machine learning. *Lancet Digit Health* 2021;3(9):e555–e564. doi:10.1016/S2589-7500(21)00101-1. PMID:34334334
10. **Dadu A, Satone V, Kaur R, et al.** Identification and prediction of Parkinson's disease subtypes and progression using machine learning in two cohorts. *npj Parkinsons Dis* 2022;8(1):172. doi:10.1038/s41531-022-00439-z. PMID:36526647
11. **Jackson CH, Sharples LD, Thompson SG, Duffy SW, Couto E.** Multistate Markov models for disease progression with classification error. *J R Stat Soc Ser D (The Statistician)* 2003;52(2):193–209. doi:10.1111/1467-9884.00351
12. **Bouwmeester W, Zuithoff NPA, Mallett S, Geerlings MI, Vergouwe Y, Steyerberg EW, Altman DG, Moons KGM.** Reporting and methods in clinical prediction research: a systematic review. *PLoS Med* 2012;9(5):e1001221. doi:10.1371/journal.pmed.1001221. PMID:22629234

> **Not cited, deliberately.** The Stage 3 verified list also contains Collins 2024 (TRIPOD+AI) and von Elm 2007 (STROBE). Neither is cited because the manuscript has **not** been checked against either checklist (Stage 5 E-7 was never run). Citing a reporting standard the manuscript has not been assessed against would assert a compliance that does not exist. Run E-7, then add both.

---

## Figure legends

**Fig. 1 | Participant flow.** Of 4,788 participants and 19,450 visits in the curated data cut, 1,080 participants contributing 5,774 visits met the cohort criteria. A motor subtype could be assigned at 4,441 of those visits (76.9%), and 934 participants had at least two labelled visits, contributing 3,361 consecutive labelled transitions. Each exclusion is counted at the step that applied it.

**Fig. 2 | Instability of the recorded label.** **a**, Transition probabilities between consecutive labelled visits, with 95% patient-clustered bootstrap intervals (2,000 resamples of participants). **b**, Proportion of observed changes of state that returned to the origin state at the following visit, by change type. **c**, Transition probabilities stratified by the elapsed interval between the two visits.

**Fig. 3 | Harmonised comparison across five model families.** Accuracy of nine fitted specifications and five reference rules on the same 3,361 transitions, re-scored onto one decision rule, one evaluation set and one baseline. Every predicted class is the argmax of a three-label probability vector; subscore models are propagated through the unmodified classification rule by Monte Carlo. Label persistence is the canonical reference.

**Fig. 4 | Where the margin over persistence comes from.** Each model's gain and loss against label persistence, decomposed by the label recorded at the prediction point. Counts sum to each model's total margin and cover all 3,361 transitions.

**Fig. 5 | The continuous multi-domain representation.** **a**, Share of across-participant variance explained by successive principal components of the level and rate coordinates; seven components are required for 90% of either. **b**, Discrimination of subsequently recorded outcomes, never used in fitting, by the best single coordinate against the subtype label. Coordinates were selected on the same data against which they are compared, so each difference is an upper bound.

## Table captions

**Table 1 | Cohort characteristics.** Full eligible cohort (1,080 participants) and the analysed subset that contributed at least one prediction opportunity (934 participants).

**Table 2 | Transition probabilities between consecutive labelled visits.** Nine cells with 95% patient-clustered bootstrap intervals, and the design effect against an interval that treats transitions as independent.

**Table 3 | Harmonised three-class comparison.** All fourteen specifications on the same 3,361 transitions: accuracy, multiclass Brier score, log loss, per-class recall, the number of predictions issued for each class, and macro-F1.

**Table 4 | Two-class secondary analysis.** Tremor dominant against PIGD on the 3,029 transitions whose next recorded label was one of the two, using conditional probabilities. The 332 transitions with an indeterminate outcome are reported separately and never scored.

**Table 5 | Per-participant accuracy by recorded-label composition.** Exploratory. Groupings were defined after the predictions were generated and from recorded labels only; the persistence rule is shown on the same groups.

---

## Asset status

Where each display item currently stands. "Run figure" means a figure exists from its analysis run but has not been redrawn to journal specification.

| Item | Source | Status |
|---|---|---|
| Fig. 1 | `experiments/20261103T192414Z_e5_participant_flow/fig1_participant_flow.png` | run figure |
| Fig. 2 | `manuscript/npj/fig2_label_instability.png` | submission-ready |
| Fig. 3 | `experiments/20261103T202632Z_e8_harmonised_family_comparison/fig1_harmonised_family_comparison.png` | run figure |
| Fig. 4 | `manuscript/npj/fig4_margin_decomposition.png` | submission-ready |
| Fig. 5a | `experiments/20261020T180515Z_clinical_profile_cv/representation_dimensionality.csv` | NOT DRAWN — source data only |
| Fig. 5b | `manuscript/npj/fig5b_outcome_discrimination.png` | submission-ready |
| Table 1 | `manuscript/npj/table1_cohort_characteristics.csv` | submission-ready |
| Table 2 | `experiments/20261103T193205Z_e2_transition_patient_bootstrap/` | source CSVs, not formatted |
| Table 3 | `experiments/20261103T202632Z_e8_harmonised_family_comparison/harmonised_metrics.csv` | source CSV, not formatted |
| Table 4 | `outputs/joint_subscore/subtype_metrics.csv + auc_metrics.csv` | source CSVs, not formatted |
| Table 5 | `experiments/20261103T175003Z_patient_level_accuracy_audit/patient_accuracy_by_subtype_composition.csv` | source CSV, not formatted |

