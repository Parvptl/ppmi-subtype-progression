# Stage 6 — Methods, drafted

**Date** 2026-11-04 · **Target** npj Parkinson's Disease, Article · **Status**
first full draft

Everything below was written against the implementation, not against the
project's own prose. Where the code and an earlier description disagreed, the
code won and the disagreement is recorded in §B.

Two conventions in this file: **manuscript text is the numbered sections**, set
as it would appear; **editorial notes are indented blockquotes** and are not
part of the manuscript. §A and §B are not manuscript text at all.

One subsection — M8, the hierarchical Bayesian growth model — is **drafted in
the form that is honest under either outcome of the C3 decision**, so it needs
no rewriting whichever you choose.

---

## M1. Study design

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

---

## M2. Dataset and participants

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

---

## M3. Clinical variables and subtype definition

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

> **Editorial.** These two checks verify the *implementation* — that the
> formula, the sentinel handling and the medication-state selection reproduce
> PPMI's own derivation of the same rule from the same items. They are not
> external validation of the label, and the manuscript must not call them that.
> The external comparison is the one with Simuni et al.'s independently
> reported baseline distribution, reported in Results.

---

## M4. Missing labels

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

---

## M5. Longitudinal construction and observed transitions

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

---

## M6. Observed-label Markov chains

Two chains treating the recorded label as the state were fitted as references.
The **discrete-time chain** estimates a row-stochastic transition matrix from
observed pairs, ignoring the elapsed interval. The **continuous-time chain**
estimates a generator *Q*, with the transition matrix over an interval *t*
given by exp(*Qt*); off-diagonal rates were parameterised as exp(θ) and fitted
by maximum likelihood.

Whether the recorded label satisfies the Markov property was tested directly on
labelled triples, comparing the distribution of the next label across possible
previous labels within each current label (χ² test).

---

## M7. Continuous-time hidden Markov model

A three-state latent phenotype *X*(*t*) was modelled as a continuous-time
Markov chain with generator *Q*, so that transition probabilities over an
elapsed interval *t* are exp(*Qt*), evaluated by eigendecomposition. What is
observed at a visit is not *X* but a noisy reading related to it by a
misclassification matrix *E*, where *E*[*a*, *b*] is the probability that a
participant whose true state is *a* is recorded as *b*. This is the multistate
Markov model with classification error of Jackson et al., applied to the motor
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

---

## M8. Hierarchical Bayesian growth model *(supplementary)*

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

> **Editorial — this is the C3 blocker.** The implementation runs **a single
> chain**, and the only convergence evidence produced is a comparison of
> first-half against second-half posterior means. There is **no R̂, no effective
> sample size and no multi-chain run**. As written above, the text states what
> was done and claims no more; it is therefore honest if the model stays a
> descriptive supplementary analysis. If the model is to support any
> inferential claim, it must be refitted with at least four chains and R̂ and
> ESS reported, and this subsection gains one sentence. **The decision is
> yours; the draft does not pre-empt it.**
>
> Two further limitations belong in the Discussion rather than here: the ratio
> is undefined at 16.3% of labelled visits, which this model silently drops and
> which are overwhelmingly tremor dominant; and a normal likelihood on a
> non-negative right-skewed quantity places 12.2% of its five-year predictive
> mass on impossible negative values.

---

## M9. Subscore models *(supplementary)*

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

---

## M10. Continuous multi-domain clinical representation *(supplementary)*

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

> **Editorial.** The module docstring gives this covariance without the
> `σ²_L·Δ` term; the implementation includes it and the text above follows the
> implementation. The docstring should be corrected.

---

## M11. History-window and retraining analyses *(supplementary)*

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

---

## M12. Deriving a reported prediction

### M12a. Harmonised three-class comparison

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

> **Editorial.** One deliberate departure. An earlier implementation computed
> the draw ratio with an infinite fill where the PIGD draw was zero, so a draw
> with both subscores zero was classified tremor dominant; the project's own
> rule assigns both-zero to indeterminate. This analysis follows the rule.
> 0.59% of draws are affected and 14 of 3,361 predicted labels change. Figures
> here therefore differ very slightly from previously circulated subscore-GP
> numbers (by 0.0009 and 0.0015 in accuracy).

### M12b. Two-class secondary analysis

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

---

## M13. Validation

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

---

## M14. Statistical analysis

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

---

## M15. Software and reproducibility

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

# §A — Reproducibility check

| Element | Status |
|---|---|
| Cohort definition reproducible from configuration | ✅ replayed from the raw cut; 5,774 / 1,080 reproduced exactly |
| Subtype rule reproducible | ✅ reproduces the recorded panel label on all 4,441 labelled visits |
| Published agreement checks | ⚠️ the 100.00% check reproduces; **the 99.87% check cannot currently be rerun — `data/validation/` is empty** (C6) |
| Transition statistics | ✅ all nine probabilities, reversion, survival reproduced, now with cluster intervals |
| CT-HMM | ✅ fold-level predictions reproduce to 1.1 × 10⁻¹⁶ |
| Harmonised comparison | ✅ natively-3-class models reproduce to 4 × 10⁻⁵; subscore models are a fresh Monte Carlo and are not bit-reproducible by design |
| Growth model | ⚠️ reproducible but **not diagnosable** (C3) |
| GPR line | ✅ modules tracked and pushed (`dada934`, 2026-09-30); the earlier "untracked" note was wrong — see §B |
| Test suite | ✅ 758 tests, all passing (617 s) |
| Environment | ✅ pinned and recorded per run |

---

# §B — Missing information, and where the code disagreed with the prose

**Blocking submission**

1. **Ethics, funding, competing interests, author contributions** — absent from
   the repository entirely. Needs the PPMI Data Use Agreement reference and the
   institutional position on secondary analysis.
2. **PPMI acknowledgement wording** — the required citation and funding
   statement must be taken from PPMI's current guidance.
3. **Code availability** ✅ **CORRECTED 2026-11-04.** The earlier entry here
   claimed the Gaussian-process modules were untracked and never pushed. They
   are tracked, were added on 2026-09-30 in `dada934`, and that commit is
   already on `origin/main`. The whole chain the subscore GP depends on —
   `scripts/18_subscore_multioutput_gp.py`, `mogp.py`, `gpr.py`,
   `trajectory.py`, `subtype.py` — is tracked, as are its three test files.
   The statement can promise the GPR line.
4. **C6** — the archived 2020 release behind the 99.87% agreement figure is no
   longer present, so that check is not currently reproducible.

**Blocking this section**

5. **C3** — the growth model's single-chain sampler, as set out in M8.

**Where the code disagreed with an earlier description, and the code won**

6. The label loss is driven by the **OFF-medication requirement** (89.6%), not
   by unscorable postural stability (4.3%); M4 follows the measurement. The
   README has been corrected.
7. The **emission floor is binding** for the indeterminate state; 0.500 is a
   constraint, not an estimate. M7 says so. Earlier prose reported it as a
   misclassification probability.
8. The profile's **innovation covariance** carries a level-diffusion term that
   the module docstring omits; M10 follows the implementation. The docstring
   should be fixed.
9. The five-year clinical output table is a **latent-state** quantity, not a
   statement about a recorded label. It is therefore not quoted in Methods and
   must not be quoted clinically in Results without passing it through the
   emission matrix (I18).
10. The project reports that **"790 of 1,080 begin treatment during follow-up"**.
    That figure does not reproduce from the frozen panel under any definition
    tried: 884 by treatment flag, 891 by a non-zero levodopa-equivalent dose,
    or 814 restricted to the 934 analysed participants. It is presumably
    computed over the clinical-profile cohort's own visit subset. M10 quotes
    the measured figures instead. The companion claim that **98.7%** are
    untreated at baseline does reproduce, at 98.6% on a dose basis, and is
    retained. **Neither affects any result** — both support the same design
    decision — but the unverifiable figure should not be printed.

**Must not be used anywhere**

11. The "+0.005 / +0.021 / +0.047" figures for the hidden Markov model's
    advantage by history length. No code path produces them and no saved output
    contains them.
