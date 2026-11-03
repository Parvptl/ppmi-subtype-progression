# Stage 9 — Discussion, drafted

**Date** 2026-11-04 · **Target** npj Parkinson's Disease, Article · **Status**
first full draft

Interprets R1–R8 without restating them, compares with the four cohort studies
verified at Stage 3, and states the limitations that Stage 1 recorded. **No new
reference is introduced**: every citation is one already numbered in the
Introduction.

Conventions as before: **manuscript text is the numbered section**; editorial
notes are indented blockquotes and are not manuscript text.

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

# §A — What this Discussion claims, and on what

| Claim | Evidence | Scope guard used |
|---|---|---|
| The limit is the label, not the model | R3 convergence + R4 localisation + R7 dissociation | "in this cohort", "these results locate" — not a general law |
| The indeterminate band is a neutral zone, not a state | R2 persistence 0.192, reversion, threshold fragility, binding emission floor | Caution "confined to the indeterminate category" |
| Agreement with PPMI-based literature, not clinic-based | R1 vs Simuni; >4-y stratum vs Lee; von Coelln explained by composition | Composition offered as "the likely explanation", not proven |
| The representation beats the label as description | R7 AUC differences | Stated as upper bounds; rates explicitly negative |
| Single-assessment endpoints overstate incidence | R2 35.3% vs 15.3% | "in this cohort by roughly a factor of two" |
| Report against persistence and with calibration | R3 macro-F1, R5 calibration | Framed as a reporting recommendation |

**Deliberately not claimed:** that the label should be abandoned; that any model
here is superior; that the representation is validated as a biomarker; that
rating variability rather than true fluctuation drives reversion (the design
cannot separate them, and the text says so); any causal statement about
medication.

---

# §B — Open items

1. **C6 disclosure** — the final sentence of the limitations discloses that the
   99.87% three-class agreement check is not currently reproducible
   (`data/validation/` is empty). If the archived release is restored before
   submission, that sentence should be deleted and the check rerun.
2. **C3** — no growth-model result is quoted in Results or Discussion. If the
   model appears in supplementary material, the limitations need a sentence on
   its single-chain MCMC, and it must be labelled descriptive.
3. **O-4** — the threshold-sensitivity figures carried into paragraph three
   inherit R2's open item: run the experiment or mark the figures post hoc.
4. **C5 is not a limitation and is deliberately absent here.** Ethics, funding,
   competing interests, CRediT and the PPMI acknowledgement are submission
   requirements, not scientific caveats; they belong in the front and back
   matter and remain entirely unwritten.
5. **Word count** — manuscript prose is 1,479 words, of which the limitations
   subsection is 469. That is long for an npj Discussion; if it must be cut,
   the reporting-recommendation paragraph compresses most readily, and the
   limitations subsection should be cut last.
