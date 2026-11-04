# Ratio-banded observations for the next-visit CT-HMM

**Question.** The CT-HMM in `scripts/60_next_visit_subtype_prediction.py` sees one
of three recorded labels per visit. Those labels are a threshold cut of the
continuous tremor/PIGD ratio, so a visit at ratio 1.20 and one at 4.50 are both
simply "TD". Does giving the same chain the ratio band alongside the subtype
predict the next visit any better?

**Answer in one line.** No. Accuracy fell by 0.53 percentage points, from 83.69% to 83.16%, with a 95% paired patient-bootstrap interval of [-1.34, +0.34] points that includes zero. The Brier score was marginally worse, the log loss marginally better, and calibration clearly worse. The additional ratio resolution did not improve next-visit prediction, and the ten-fold increase in parameters is not justified by the result.

## A. What was changed

The observation alphabet, and nothing else. The observed value at a visit became
the pair

    observed category = (subtype, ratio band)

and the emission matrix became rectangular, 3 hidden states by 44 joint
categories. Everything the baseline fixes was held fixed: three hidden states,
the continuous-time generator `Q` built by `ctmc.build_generator`, interval-aware
transition matrices `exp(Q dt)`, forward filtering with no backward pass, the
same 934-patient cohort and 3,361 prediction cases, the same two-class
conditional target, Indeterminate kept as input evidence and never scored as a
target, the same tie rule, the same seed (42), the same four restarts and the
same emission floor (0.5). The scoring code is the baseline's, called on both
arms.

The subtype was *not* replaced. Every joint category carries its subtype; the
categories of one subtype form a contiguous block; and the next-visit prediction
is summed back over each subtype's block and renormalised before it is scored,

    P(TD) = sum over the TD categories, and likewise for Indeterminate and PIGD

so both arms predict exactly the same quantity.

## B. Why these bands

**Main range, 0.0 to 3.0, fixed 0.1-width bands (30 bands).** Specified for the
experiment, to keep fine ratio resolution where most of the data sits.

**Tail, 3.0 upward, 0.25-width candidate bands, merged.** The tail is not thin
here: 865 of the 3,603 computable ratios are at or above 3.0, from
416 patients, and the largest is 11.36. Wider candidates reduce the
number of near-empty categories a 0.1 grid would create out there.

**The merge rule, fixed before any model was fitted.**

    A tail band must hold >= 30 observations from >= 20 distinct patients.
    Sweeping upward from 3.0, consecutive 0.25-width candidates accumulate into
    the current band until both minima are met; then the band closes and a new
    one starts. A final band that never reaches the minima is absorbed into the
    band below it. The topmost band is [lo, inf).

The two minima were chosen from the shape of the training-side distribution, not
from any held-out result:

* each tail band costs one free emission parameter per hidden state, so with
  three hidden states a band needs about 3 x 5 = 15 observations to reach the
  conventional five-per-cell minimum for a multinomial cell; 30 is the smallest
  round number that clears that with a factor-of-two margin;
* the patient minimum exists because this is a longitudinal panel. A band can
  hold many observations from very few people — repeat visits of one patient are
  not independent evidence — so a count alone can overstate the information in a
  band.

No alternative value of either minimum was ever scored against LOPO accuracy.
The rule is recorded in `experiment_config.json` alongside the results.

**Boundaries.** Left-closed, right-open throughout: a ratio `r` falls in
`[lo, hi)`, so a value sitting exactly on an edge belongs to the band that edge
opens and no observation can belong to two bands. The topmost tail band is
`[lo, inf)`, which is what makes a held-out ratio larger than anything in
training assignable without anyone having looked at it. Two bands straddle a
classification threshold and therefore split into two joint categories:
`[0.90,1.00)` (PIGD owns 0.90 exactly under the published rule) and
`[1.10,1.20)` (the TD threshold 1.15 falls inside it).

**Ratio not computable.** The ratio is undefined whenever `pigd_score == 0`, and
that is **692 of the cohort's 4,295 labelled visits (16.1%)** — 634 labelled TD
(no gait or postural impairment scored, some tremor) and 58 labelled
Indeterminate (neither subscore above zero). Dropping them would have changed
the evaluation cohort, which the brief forbids, so each subtype that can produce
such a visit gets its own "undefined" category. A PIGD label requires a positive
PIGD subscore, so there is no PIGD-undefined category; that is a property of the
classification rule, not of the data.

## C. The emission model, and why it is constrained this way

The square model's floor is `P(label a | state a) >= 0.5`: an examination records
the true state correctly more often than not. Applied to a single joint category
that constraint would mean something quite different, and with ~44 categories it
would be unsatisfiable. It is applied to the **subtype marginal** instead. The
emission factorises as

    E[a, (s, b)] = M[a, s] * C[a, s, b]

with `M` a 3x3 subtype misclassification matrix, parameterised and bounded
exactly as the square model's `E` is, and `C[a, s, .]` a free distribution over
subtype `s`'s bands for hidden state `a`. Then

    sum over b of E[a, (s, b)] = M[a, s]

so the floor `M[a,a] >= 0.5` *is* the square model's floor, on the same quantity,
with the same logit bound. Three consequences matter:

1. **The square model is the exact special case.** With one band per subtype the
   parameter vector, the bounds and the likelihood reduce to `hmm.fit_hmm`'s.
   `tests/test_hmm_banded.py` checks this numerically, and the experiment runs it
   as a third arm — see E below.
2. **The bands cannot reshape the output distribution.** Because the subtype
   marginal of `E` is `M`, the aggregated prediction is exactly `alpha P(dt) M`,
   the same functional form as the baseline's `alpha P(dt) E`. The bands can only
   change the prediction through `alpha`, the filtered belief about the latent
   state. That is precisely the experiment's question, and it keeps the
   comparison clean.
3. **The cost is parameters.** Each band of each subtype adds one free parameter
   per hidden state: 137 free parameters against the baseline's 14, on the same 4,295
   observations. The fit stays identified (the floor pins `M`, finite logit bounds
   pin `C`), but this, and not the latent process, is what the experiment puts at
   risk.

## D. Leakage control

Every learned quantity, the band structure included, is derived inside the fold
from the training patients only:

1. hold one patient out whole;
2. rebuild the merged tail-band edges from the remaining 933 patients' ratios;
3. recode every patient's visits into joint categories under those edges;
4. fit on the 933, predict the held-out patient's visits.

The construction is then **re-verified rather than trusted**: every fold rebuilds
its edges from its training patients and raises `LeakageError` if the edges it
used differ (`next_visit_banded.assert_bands_training_only`). `loo.assert_no_leakage`
checks the sequences and transition pairs as before. A held-out ratio can only
fall outside every training band in one direction — above the largest training
ratio — and the open-topped band removes the need for any decision there.

Because the edges are learned per fold they are **not identical across folds**,
and the results report the distinct structures that occurred rather than
presenting one global definition as if it had been used everywhere.

## E. The control arm

The joint model has ~137 parameters. With L-BFGS-B's finite-difference gradient
that costs 138 likelihood evaluations per iteration, and a probe on the full
cohort exhausted a 138,000-evaluation budget after 935 iterations at a projected
gradient of 2.3e-2 — twenty times the tolerance this project calls converged.
Reporting that against a converged baseline would have confounded the
experiment's question with the optimiser's failure, so the joint model's gradient
is computed **exactly**, by forward-backward (`hmm_banded.loglik_and_grad`,
validated against central differences to ~1e-7 relative in the tests). Objective,
bounds, tolerances, optimiser, restarts and seed are unchanged.

That leaves one loose end: the two arms would then differ in their gradient
method as well as their alphabet. So a third arm is fitted in every fold — the
**square model through the joint code path**, over a one-category-per-subtype
alphabet, where the joint model *is* the square model. Any gap between it and the
baseline arm measures the implementation and the gradient, not the bands. It is a
control, reported separately, and never part of the primary comparison.

## F. What the bands turned out to be

Every fold built its tail edges from its own 933 training patients. All 934 folds
arrived at the **same** ten tail bands, because no single patient's observations
were pivotal at a merge threshold — removing one patient never moved an edge.
That is a result of the construction, not a shortcut around it: the edges were
rebuilt and re-verified in each fold, and 39 folds had a held-out patient whose
ratio sat above every training band and fell into the open-topped band as
designed.

    retained tail bands   [3.00,3.25) [3.25,3.75) [3.75,4.25) [4.25,4.75)
                          [4.75,5.25) [5.25,5.50) [5.50,6.00) [6.00,6.50)
                          [6.50,7.50) [7.50,inf)
    34 candidate bands (24 of them occupied) -> 10 merged bands
    30 main + 10 tail = 40 ratio bands -> 44 joint categories
    free parameters       137 (baseline 14), identical in every fold

Of the 44 categories, **1 is empty** (`PIGD|[0.90,1.00)`, which needs a ratio of
exactly 0.90 and never occurs), **8 carry fewer than 15 observations** and 11
carry fewer than 30. Almost all of the sparse ones are *main-range* TD bands —
`TD|[1.40,1.50)`, `TD|[2.30,2.40)`, `TD|[2.60,2.70)` and so on — not tail bands.
The main-range width is fixed at 0.1 by the experiment's design and is not
merged, so this sparsity was carried into every fit deliberately. Its cause is
structural: the ratio is a quotient of a mean over 11 tremor items and a mean
over 5 PIGD items, so it takes lumpy discrete values (multiples of 5/11b) and a
uniform 0.1 grid lands several bands almost empty while their neighbours hold
200. The merge rule was only ever allowed to touch the tail.

## G. Results

All 934 folds completed, none failed, and every fit in every arm converged
(scipy reported success in 934/934 folds for all three arms; worst projected
gradient 0.067 banded against 0.025 baseline, both under this project's
criterion).

### Accuracy

| model | n scored | accuracy | 95% CI (patient bootstrap) | Brier | log loss |
|---|---|---|---|---|---|
| CT-HMM (latent, interval-aware) — baseline | 3,029 | **0.8369** | [0.8204, 0.8541] | 0.1214 | 0.3961 |
| CT-HMM (subtype x ratio band) — experimental | 3,029 | **0.8316** | [0.8142, 0.8498] | 0.1221 | 0.3930 |
| CT-HMM (subtype labels, joint code path) — control | 3,029 | 0.8369 | [0.8204, 0.8541] | 0.1214 | 0.3961 |

Paired on the 3,029 cases both arms scored, from 899 patients:

    accuracy difference (banded - baseline)   -0.0053   (-0.53 percentage points)
    95% CI, paired patient bootstrap          [-0.0134, +0.0034]
    relative change                           -0.63%
    banded right / baseline wrong             67
    baseline right / banded wrong             83
    exact McNemar p (ignores clustering)      0.2205

The interval includes zero, so the experiment does not establish a difference in
either direction; the point estimate is negative. 48 patients did better under
the banded model, 58 worse, and 793 of 899 were unaffected.

**The control arm settles the one methodological worry.** The baseline model
refitted through the joint code path with the exact gradient gave *identical*
predictions: zero discordant cases, an accuracy difference of 0.00000, and a
worst per-fold predicted-probability gap of 1.1e-4. The baseline arm also
reproduces the published October run **exactly** — max |delta| of 0.000e+00 on
every probability column over all 3,361 cases, the same case set, the same
eligibility, the same targets and the same accuracy to six decimals. So the
difference above is attributable to the bands and to nothing else.

### Probabilistic performance

    Brier     baseline 0.1214   banded 0.1221   delta +0.0007  (worse)
    log loss  baseline 0.3961   banded 0.3930   delta -0.0031  (better)

These point in opposite directions, and the reason is visible in the calibration
tables. The banded model's mean absolute calibration gap is **0.0283** against
the baseline's **0.0065** — about four times worse — with its largest errors in
the middle of the range (predicted 0.150 against an observed PIGD rate of 0.206,
and predicted 0.380 against 0.329). It is also more willing to be extreme: 1.5%
of its predictions fall below 0.05 or above 0.95, against 0.0% for the baseline.
So the banded model is *not* better calibrated. Its slightly lower log loss comes
from being more confident where it happens to be right, while its worse Brier
score and much worse calibration gap reflect being confidently wrong elsewhere.
Calibration bin edges are not identical between the arms, because the sparse-bin
merge is adaptive; the comparison of gaps is therefore indicative rather than
exact.

### Where it improved and where it got worse

| cut | group | baseline | banded | change |
|---|---|---|---|---|
| current label | TD (n=2,080) | 0.8812 | 0.8716 | −0.96 pp |
| | Indeterminate (n=270) | 0.6074 | 0.6000 | −0.74 pp |
| | PIGD (n=679) | 0.7923 | 0.8012 | **+0.89 pp** |
| ratio at prediction | undefined (n=547) | 0.9104 | 0.8976 | −1.28 pp |
| | [0.00,0.90) (n=679) | 0.7923 | 0.8012 | **+0.89 pp** |
| | [0.90,1.15) (n=237) | 0.6034 | 0.6160 | **+1.26 pp** |
| | [1.15,2.00) (n=470) | 0.7277 | 0.6957 | **−3.20 pp** |
| | [2.00,3.00) (n=435) | 0.8897 | 0.8805 | −0.92 pp |
| | [3.00,inf) (n=661) | 0.9486 | 0.9501 | +0.15 pp |
| history | 1 prior visit (n=837) | 0.8339 | 0.8327 | −0.12 pp |
| | 5+ prior visits (n=734) | 0.8324 | 0.8174 | −1.50 pp |
| interval | <= 1.5y (n=2,597) | 0.8417 | 0.8371 | −0.46 pp |
| | > 1.5y (n=432) | 0.8079 | 0.7986 | −0.93 pp |

A coherent pattern, not noise in every cell. The banded model moves its decision
boundary toward PIGD: it predicts PIGD on 890 cases against the baseline's 800,
lifting PIGD recall from 0.667 to 0.708 and dropping TD recall from 0.910 to
0.885. Because TD is the majority outcome (2,115 of 3,029), overall accuracy
falls while **balanced accuracy rises from 0.7888 to 0.7965**. The one large
loss is concentrated in `[1.15,2.00)` — low-ratio TD visits just above the
threshold — where accuracy drops 3.2 points. The gains are on the PIGD side and
just inside the Indeterminate band. It does *worse* with more history (−1.5
points at 5+ visits), which is the opposite of what a model that had learned to
use finer evidence about the latent state should do.

### Did the bands carry latent-state information at all?

Yes, in the fit — and that is what makes the result informative rather than
merely null. A descriptive whole-cohort fit
(`scripts/88_ratio_band_emission_diagnostic.py`, every patient included, **not**
a held-out result) shows the band distribution differing strongly between hidden
states: the largest total-variation distance between two latent states'
distributions over a subtype's bands is 0.66 for TD, 0.50 for Indeterminate and
0.63 for PIGD. Zero would have meant the bands said nothing.

But look at *how* it uses them. The categories where the states disagree most are
ones where the latent PIGD and Indeterminate states assign essentially zero
probability — `TD|[4.25,4.75)` gets 0.044 from the latent TD state and 0.000 from
the other two, with largest-to-smallest ratios around 10^6, i.e. the band logits
driven onto their bound. The fitted model has concluded that a high-ratio TD
visit is *impossible* for a patient whose latent state is PIGD. That is a hard
separation learned from a few hundred observations spread over 44 categories, and
it is exactly the mechanism that makes the filtered latent state over-certain and
the predictions worse calibrated.

The fitted subtype marginal also moved in a telling direction (baseline `E`
against banded `M`): the PIGD state's own-label probability fell from 0.872 to
0.778, while the latent Indeterminate row sits **exactly on the emission floor**
[0.25, 0.50, 0.25] in both models — the floor binds for that state either way,
which is the degeneracy `hmm.DEFAULT_EMISSION_FLOOR` exists to prevent.

## H. Scientific interpretation

**Does ratio-band information add anything beyond the subtype label?** Not for
this task, this cohort and this model. The ratio band is real information — the
emission does use it, and it shifts predictions on 150 of 3,029 cases, sometimes
by as much as 0.66 in P(PIGD) — but it does not convert into better next-visit
prediction. The most defensible reading is that the information gained is
cancelled by the variance cost of estimating 137 emission parameters instead of
14 from 4,295 observations, and that the fit then over-separates the latent states
by pushing sparse categories to their bounds.

Three observations support that reading over the alternatives:

1. **The degradation grows with history.** A model that genuinely learned a
   sharper latent state should do *better* when it has more visits to filter
   over. This one does worse (−1.5 points at 5+ prior visits against −0.1 at
   one), which is what over-confident filtering looks like: errors compound as
   the belief is propagated.
2. **Calibration degrades while the point estimate barely moves.** Mean P(PIGD)
   is 0.3049 against the baseline's 0.3047 — the model has not changed its
   average view, it has redistributed confidence, 4x less accurately.
3. **The sparsity is in the fixed main range, not the merged tail.** The tail
   merge did its job: the smallest retained tail band carries 35 observations
   from 33 patients, comfortably above the 30/20 minima, and the far tail
   `[3.00,inf)` is the one region
   where the banded model is (marginally) better on both accuracy and Brier
   score. The eight near-empty categories are all main-range TD bands, whose
   0.1 width the experiment fixed in advance.

That third point is the most actionable: if this line were pursued, the obvious
next step is a band structure applied to the *main range* as well — either the
same minimum-count merge, or bands on the log-ratio, or an ordinal emission that
shares information across neighbouring bands instead of estimating each
independently. Those are different experiments, and none of them should be
selected by rerunning this comparison until one wins.

**What must not be concluded.** This experiment says nothing about disease
mechanism. The ratio band is not a biological state and the emission's preference
for certain bands is a property of a 44-category multinomial fitted to 4,295
observations, not evidence about progression. Nor does it say the ratio is
uninformative in general: it says that discretising it into these bands, inside
this emission, does not improve this prediction. A continuous treatment of the
ratio — which is what the project's Gaussian-process and mixed-model families
already do — is not tested here.

### Limitations

* **Power.** The paired interval is ±1.3 points wide. A true improvement smaller
  than about one point could not be detected with 899 patients and 3,029
  scored transitions, so "no improvement" means "no improvement of a size this
  design can see", not "exactly zero".
* **The target is a noisy label, not a diagnosis.** Accuracy here is accuracy at
  reproducing this project's own threshold classification of the next visit,
  which carries measurement variability of its own. Improved accuracy would not
  by itself demonstrate clinical benefit, and this experiment claims none.
* **One alphabet, one merge rule.** The band widths and the 30/20 minima were
  fixed in advance, as they had to be to keep the comparison honest. Other
  defensible choices exist and were not tried; the result is about this one.
* **16% of visits have no ratio at all.** They were kept as their own category
  rather than dropped, which preserves the cohort but means a sixth of the
  observations contribute no band resolution by construction — and that group is
  where the banded model loses 1.3 points.
* **Interval-aware by assumption.** As in the baseline, the elapsed time to the
  next visit is an input to both arms. The models are told when the next
  examination happens, never what it records, and PPMI's visit schedule is not
  independent of disease course.
* **The whole-cohort emission fit is descriptive.** Section G's mechanism
  evidence comes from a fit that saw every patient. It explains the held-out
  result; it is not itself a held-out result.

## I. Conclusion

* **Did the ratio-banded CT-HMM outperform the original?** No. 83.16% against
  83.69%, a difference of −0.53 points with a 95% paired interval of
  [−1.34, +0.34] that includes zero.
* **Is any improvement meaningful?** There is no improvement to qualify. The one
  genuine shift — balanced accuracy up 0.8 points, PIGD recall up 4.1 points — is
  a move along the TD/PIGD trade-off, not a gain in discrimination, and it comes
  with four-times-worse calibration.
* **Is it consistent across patients?** No. 793 of 899 patients are unaffected,
  48 improve and 58 worsen.
* **Is the additional complexity justified?** No. Ten times the parameters, no
  predictive gain, worse calibration, and a fit that drives sparse categories to
  their bounds.
* **Should the model be retained for further research?** Not in this form. The
  implementation should be kept — it is tested, leakage-checked and reduces
  exactly to the audited baseline when the bands are removed, so it is a working
  platform for the follow-ups named in H. The *44-category fixed-width alphabet*
  should not be carried forward.

The result is a clear negative rather than an inconclusive one on the primary
metric, and the direction of the point estimate, the calibration evidence and the
history-length pattern agree on why.

## J. Reproduction

    # the experiment: 934 LOPO folds, three arms, ~31 min on 24 cores
    python scripts/87_cthmm_ratio_banded_observations.py \
        --out experiments/<UTC>_cthmm_ratio_banded_observations --workers 24
    # interrupted? completed folds are checkpointed inside --out
    python scripts/87_cthmm_ratio_banded_observations.py --out <same dir> --resume

    # the descriptive emission diagnostic (whole-cohort fit, ~30 s)
    python scripts/88_ratio_band_emission_diagnostic.py --out experiments/<UTC>_...

    # tests
    python -m pytest tests/test_ratio_bands.py tests/test_hmm_banded.py \
                     tests/test_next_visit_banded.py

Run directories (gitignored, they contain participant-level predictions):

    experiments/20261104T152154Z_cthmm_ratio_banded_observations/
    experiments/20261104T155439Z_ratio_band_emission_diagnostic/

New code, all additive — no existing module was modified:

    src/ppmi_subtype/ratio_bands.py          the joint alphabet and the merge rule
    src/ppmi_subtype/hmm_banded.py           the rectangular-emission CT-HMM
    src/ppmi_subtype/next_visit_banded.py    one LOPO fold, three arms
    scripts/87_cthmm_ratio_banded_observations.py
    scripts/88_ratio_band_emission_diagnostic.py
    tests/test_ratio_bands.py                25 tests
    tests/test_hmm_banded.py                 16 tests
    tests/test_next_visit_banded.py          13 tests

Provenance: panel MD5 `fb038ddbdc19ab3368e13940f398e7ba`, seed 42, 4 restarts,
emission floor 0.5, numpy 2.2.4 / pandas 2.2.3 / scipy on Python 3.13.16, run on
`master` with 24 workers. Code hashes and the full environment are in
`summary.json` under each run directory.
