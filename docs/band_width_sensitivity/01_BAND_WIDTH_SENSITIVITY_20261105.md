# Effect of ratio-band width on CT-HMM next-visit prediction

**Question.** The ratio-banded CT-HMM in
`docs/ratio_banded_cthmm/01_RATIO_BANDED_CTHMM_20261104.md` cut the main range
`[0.0, 3.0)` into bands 0.10 wide. That width was chosen by hand and never
tested. Does changing it improve next-visit prediction, and which width balances
predictive information against model complexity best?

**Answer in one line.** Narrowing the bands does not help and costs three times
the parameters; widening them helps slightly, reliably and for free, but the
help is confined to a region of the ROC curve no clinician would operate in, so
the previous conclusion is unchanged at every width tested: the ratio bands do
not improve next-visit prediction.

Four widths were run — 0.05, 0.10 (the published baseline), 0.20 and 0.30 —
plus the unbanded subtype-only reference, over the same 934 leave-one-patient-out
folds. 4,670 model fits, all converged, no fold failed.

---

## A. What was tested

Exactly four main-range widths, as specified:

| width | main bands | tail bands | joint categories | free parameters |
|---|---:|---:|---:|---:|
| 0.05 | 60 | 10 | 73 | 224 |
| 0.10 (baseline) | 30 | 10 | 44 | 137 |
| 0.20 | 15 | 10 | 29 | 92 |
| 0.30 | 10 | 10 | 24 | 77 |

No width was added beyond the four requested. §H sets out the one case where a
fifth would be scientifically useful and why it is not run here.

**Band construction.** The main range `[0.0, 3.0)` is cut into fixed-width,
mutually exclusive bands, **left-closed and right-open**: a ratio sitting exactly
on an edge opens the higher band, so no observation can belong to two. Edges are
computed as `k * span / n`, not `k * width` — at width 0.30 the naive form gives
`3 * 0.3 = 0.8999999999999999`, which would put a ratio of exactly 0.90 one band
too low. At the default width the new edges are bitwise identical to the
published ones.

**The tail is identical in all four configurations.** The published rule is
unchanged: 0.25-wide candidate bands from 3.0 upward, merged upward until each
holds at least 30 observations from at least 20 training patients, topmost band
open. Over the whole cohort that turns **34 candidate bands into 10 final bands**
with the same lows — `[3.00, 3.25, 3.75, 4.25, 4.75, 5.25, 5.50, 6.00, 6.50,
7.50]` — at every width, because the tail does not depend on the main-range
width. The tail-merging strategy was not redesigned. Only the main-range width
varies, so the comparison isolates it.

**The subtype is never replaced.** Every observation is the pair
`(subtype, ratio band)`; the categories of one subtype form a contiguous block;
the prediction is summed back over each subtype's block and renormalised before
scoring. The emission stays `E[a,(s,b)] = M[a,s] × C[a,s,b]` with the 0.5 floor,
identical in all four arms.

**Everything else is held fixed**: three hidden states, the continuous-time
generator `Q`, interval-aware `exp(Q·dt)`, forward filtering, the 934-patient
cohort, 3,361 cases, 3,029 eligible, Indeterminate kept as input evidence and
never scored as a target, seed 42, four restarts, emission floor 0.5, the
two-class conditional target and the baseline's own scoring code.

**The comparison is fair by construction**, and this was asserted rather than
assumed. 21 checks pass, including: every arm produced the same 3,361 rows;
identical case sets, targets and eligibility in every arm; **the same 3,029-case
denominator for all five arms** — the 0.05 arm does not get to drop sparse
categories or score fewer patients; no fold saw its own held-out patient, in
training, in band construction, in tail merging or in category selection; and
the width-0.10 and subtype-only arms **reproduce the published run to 1.1e-16**,
so the baseline here is the baseline already on record.

---

## B. Did finer bands help? No.

Width 0.05 is the worst arm in the study on the metrics that separate the arms
at all, and it is never the best on any.

Against the 0.10 baseline (paired patient bootstrap, 2,000 resamples):

| metric | 0.05 − 0.10 | 95% CI | reliable? |
|---|---:|---|---|
| accuracy | +0.20 pp | [−0.20, +0.63] | no |
| AUC-ROC | −0.0020 | [−0.0042, +0.0000] | no (boundary) |
| PR-AUC | **−0.0068** | [−0.0143, −0.0000] | yes, worse (barely) |
| partial AUC (FPR ≤ 0.2) | −0.0021 | [−0.0061, +0.0015] | no |
| Brier | +0.0006 | — | worse |
| log loss | +0.0022 | — | worse |

For that it pays 224 parameters against 137, and the alphabet degrades badly:
**11 of its 73 categories are empty**, 40 hold fewer than 30 observations, 34 are
seen in fewer than 20 patients, and the median category holds 22 observations
against 71 at width 0.10. The observations-per-parameter ratio falls to 19.2.

This is the ordinary sparsity failure, and it is worth being precise about what
it is *not*: it is not a fitting failure. Every one of the 934 folds converged at
width 0.05, with a worst projected gradient of 0.063. The model fits; there is
simply less and less evidence behind each emission probability, and the extra
resolution it buys is not information the chain can use.

---

## C. Did wider bands help? Slightly, reliably, and at lower cost.

Against the 0.10 baseline:

| metric | 0.20 − 0.10 | 95% CI | 0.30 − 0.10 | 95% CI |
|---|---:|---|---:|---|
| accuracy | +0.20 pp | [−0.26, +0.70] | +0.43 pp | [−0.13, +1.00] |
| AUC-ROC | +0.0017 | [−0.0002, +0.0036] | **+0.0026** | **[+0.0001, +0.0051]** |
| PR-AUC | **+0.0142** | **[+0.0059, +0.0227]** | **+0.0125** | **[+0.0026, +0.0218]** |
| partial AUC | **+0.0045** | **[+0.0010, +0.0082]** | **+0.0068** | **[+0.0022, +0.0118]** |
| Brier | −0.0012 | — | −0.0022 | — |
| log loss | −0.0032 | — | −0.0055 | — |

Bold rows exclude zero. Both wider arms beat the baseline on PR-AUC and on
partial AUC with intervals that exclude zero; width 0.30 also does so on AUC-ROC,
though only barely (lower bound +0.0001).

The direction is consistent across every metric and monotone in the width:
0.05 → 0.10 → 0.20 → 0.30 improves AUC-ROC (0.8739, 0.8758, 0.8775, 0.8784),
Brier (0.1227, 0.1221, 0.1209, 0.1199) and log loss (0.3952, 0.3930, 0.3898,
0.3874) in order. Wider bands are simultaneously **better and cheaper**: width
0.30 is the best banded arm on five of six metrics while using 77 parameters
against the baseline's 137 and the 0.05 arm's 224.

---

## D. Which width was best on each metric

| metric | best | value | best banded |
|---|---|---:|---|
| accuracy | **subtype only** | 0.8369 | 0.30 (0.8359) |
| balanced accuracy | 0.05 | 0.7982 | 0.05 |
| AUC-ROC | **0.30** | 0.8784 | 0.30 |
| PR-AUC | **0.20** | 0.7540 | 0.20 |
| partial AUC (FPR ≤ 0.2) | **0.30** | 0.7892 | 0.30 |
| Brier | **0.30** | 0.1199 | 0.30 |
| log loss | **0.30** | 0.3874 | 0.30 |
| calibration gap | **subtype only** | 0.0065 | 0.30 (0.0222) |
| fewest parameters | **0.30** | 77 | 0.30 |

Width 0.30 is the best banded configuration on the ranking and probabilistic
metrics and is also the simplest. But the unbanded reference still has the best
raw accuracy and by far the best calibration, which is the tension §F and §H
take up.

**A caveat that applies to this whole table.** Four configurations were compared
on held-out data and a winner named. That is selection on the evaluation set, so
the held-out metric of whichever width is chosen is optimistically biased. The
honest reading of "0.30 is best" is *best among the four tested, on this
cohort* — not an unbiased estimate of what 0.30 would score on new patients.
Band width was never tuned against held-out performance *inside* the fits; the
bias here is the ordinary one of picking a winner after the fact.

---

## E. Which differences are statistically reliable

**Reliable** (95% paired patient-bootstrap interval excludes zero):

- 0.30 beats 0.10 on AUC-ROC, PR-AUC and partial AUC.
- 0.20 beats 0.10 on PR-AUC and partial AUC.
- 0.05 is **worse** than 0.10 on PR-AUC.
- Every banded arm beats subtype-only on AUC-ROC (+0.0151 to +0.0197).

**Not reliable** (interval includes zero):

- **Every accuracy comparison in the study.** No width differs from 0.10 in
  accuracy by an amount the data can distinguish from zero, and neither does the
  unbanded reference (+0.53 pp, [−0.34, +1.34]). McNemar's exact test agrees
  (p = 0.15 to 0.48), though it ignores the patient clustering and is reported
  only as a cross-check.
- 0.05 vs 0.10 on AUC-ROC, sitting exactly on the boundary
  ([−0.0042, +0.0000]).
- 0.20 vs 0.10 on AUC-ROC ([−0.0002, +0.0036]).
- **Every comparison against the unbanded reference in the usable region of the
  ROC curve** — see §F.

No claim of significance is made anywhere an interval includes zero. And where
an interval does exclude zero, the magnitude deserves the same scepticism: the
AUC gain of 0.30 over 0.10 is **+0.0026**. Reliable is not the same as
meaningful, and a difference in the third decimal place of AUC is not a reason
to change a model on its own.

---

## F. Calibration, and the one result that matters clinically

Mean absolute gap between predicted and observed P(PIGD), averaged over the
calibration bins unweighted, which is the project's existing convention and the
quantity differenced in `probabilistic_contrasts.csv`:

| arm | calibration gap | Brier |
|---|---:|---:|
| subtype only | **0.0065** | 0.1214 |
| width 0.05 | 0.0246 | 0.1227 |
| width 0.10 | 0.0283 | 0.1221 |
| width 0.20 | 0.0238 | 0.1209 |
| width 0.30 | 0.0222 | 0.1199 |

Two things are going on at once. The wider arms are the best calibrated of the
banded models — 0.0222 at width 0.30 against 0.0283 at the 0.10 baseline — but
the relationship is not monotone in the width, because 0.05 (0.0246) also
calibrates better than 0.10. The baseline width is simply the worst-calibrated
arm in the study. What *is* consistent: **every banded arm is between three and
a half and four and a half times worse calibrated than the unbanded
reference** (3.4x at width 0.30, 4.4x at 0.10), and widening the bands does not
come close to closing that gap.

Width 0.30 nevertheless has a better Brier score than subtype-only
(0.1199 vs 0.1214). Brier decomposes into
calibration and refinement, so this says the bands buy sharpness and pay for it
in calibration, and at width 0.30 the sharpness wins the sum. A model that is
sharper but less honest about its own confidence is not obviously the better
model, and which half matters depends on the use.

**The clinically decisive result.** Every banded arm beats subtype-only on
AUC-ROC with an interval excluding zero. But AUC integrates over the whole ROC
curve, including false-positive rates nobody would accept. All five arms
actually operate at FPR 0.09–0.11. Restricting to the usable region, FPR ≤ 0.2
(McClish-standardised partial AUC), the picture collapses:

| arm vs subtype only | AUC-ROC Δ | 95% CI | partial AUC Δ | 95% CI |
|---|---:|---|---:|---|
| width 0.05 | +0.0151 | [+0.0070, +0.0243] | −0.0081 | [−0.0186, +0.0026] |
| width 0.10 | +0.0171 | [+0.0089, +0.0260] | −0.0060 | [−0.0156, +0.0040] |
| width 0.20 | +0.0188 | [+0.0110, +0.0276] | −0.0015 | [−0.0102, +0.0081] |
| width 0.30 | +0.0197 | [+0.0120, +0.0285] | **+0.0009** | [−0.0074, +0.0090] |

*(Secondary, post-hoc: the pre-specified comparison was every width against
0.10. These contrasts against the unbanded reference were added after that
analysis to settle a question it raised, and refit nothing. Read them as
descriptive.)*

The AUC advantage of banding is real and it is **entirely outside the region
anyone can use**. In the usable region, not one width is distinguishable from no
banding at all, and the best of them is ahead by 0.0009. This reproduces, at
every band width, the finding already recorded for width 0.10. A higher AUC here
does not mean better clinical performance, and it should not be reported as
though it did.

---

## G. Which patients the width changes

**By current label** (AUC within the stratum, where the task is harder):

| stratum | n | subtype only | 0.05 | 0.10 | 0.20 | 0.30 |
|---|---:|---:|---:|---:|---:|---:|
| TD | 2,080 | 0.6693 | 0.7442 | 0.7501 | 0.7503 | **0.7511** |
| PIGD | 679 | 0.6649 | 0.6645 | 0.6728 | **0.6898** | 0.6857 |
| Indeterminate | 270 | 0.6328 | 0.6369 | 0.6348 | **0.6379** | 0.6326 |

The bands do most of their work inside the TD group — +0.082 AUC at width 0.30 —
which is exactly where the subtype label is least informative: "TD" spans ratios
from 1.15 to 11.36, so the band tells the chain something the label genuinely
does not. Width barely matters here; any banding captures nearly all of it.
Among PIGD-labelled visits the wider bands help more, and in the small
Indeterminate stratum nothing separates the arms.

**By history length** (AUC):

| prior visits | n | subtype only | 0.05 | 0.10 | 0.20 | 0.30 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 837 | 0.7665 | 0.8353 | 0.8359 | **0.8370** | 0.8363 |
| 2 | 697 | 0.8753 | **0.8923** | 0.8915 | 0.8901 | 0.8905 |
| 3 | 456 | 0.8370 | 0.8373 | 0.8418 | 0.8461 | **0.8516** |
| 4 | 305 | 0.8509 | **0.8888** | 0.8876 | 0.8885 | 0.8882 |
| 5+ | 734 | 0.8884 | 0.8832 | 0.8892 | 0.8933 | **0.8938** |

The large effect here is banding versus no banding at a single prior visit
(+0.070 AUC), not the width: with one visit of history the chain has almost
nothing else to go on, so the extra resolution in the observation matters most.
Across widths the differences within a stratum are in the third decimal and the
strata are small — 270 Indeterminate cases, 305 at four prior visits — so the
stratum-level ordering is not something to over-read. The ordering is not
monotone in width in any stratum, which is itself a sign these differences are
noise.

---

## H. Scientific interpretation

**The ratio carries information the subtype label does not, but it is coarse
information.** Everything in this study points the same way: the useful content
of the ratio is roughly *how far the patient sits from the threshold*, and
resolving that to 0.05 adds nothing while costing 87 extra parameters and 11
empty categories. The finest banding is the worst performer; the coarsest is the
best; every metric moves monotonically between them.

**The trend has not turned over inside the tested range.** Width 0.30 is at the
edge of what was tested and is still improving. That leaves two readings, and
the data here cannot choose between them:

1. The optimum is at some width above 0.30, and a fifth configuration would find it.
2. There is no interior optimum, and the trend runs all the way to the
   degenerate limit — a single main band, which is the subtype-only model plus a
   tail indicator.

Reading 2 deserves more weight than the monotone trend alone suggests, because
the unbanded reference *already* has the best accuracy (0.8369) and by far the
best calibration (0.0065), and in the usable region of the ROC curve no banded
arm beats it. The sequence 0.05 → 0.30 may simply be walking back toward the
unbanded model, with the metrics improving because the bands are getting less
harmful rather than more informative. A fifth width at 0.50 or 0.60 would
distinguish these two readings cleanly. It is not run here because the brief
fixed the design at four widths, and because the answer would not change the
recommendation: both readings end at "the bands are not earning their place."

**A confound to state plainly.** The subtype thresholds (0.90 and 1.15) land on
the band grid at some widths and not others — at 0.05 both are edges, at 0.10
only 0.90, at 0.20 neither, at 0.30 only 0.90. So the number of bands a
threshold cuts through is 1, 2, 2, 1, which is **not monotone in the width**.
The four configurations therefore differ slightly in more than resolution, and a
small part of any difference between them may come from this rather than from
band width. It is inherent to fixed-width banding over a range whose thresholds
are not multiples of the width; it is recorded in `complexity_by_width.csv`
rather than engineered away, since removing it would mean abandoning fixed-width
bands and changing more than the one variable under test.

**What this does not show.** It does not show that the tremor/PIGD ratio is
uninformative about progression — only that feeding it to *this* chain as a
banded observation does not improve *next-visit* prediction. A different target
(time to a milestone), a different horizon, or treating the ratio as a
continuous emission rather than a discretised one are all untested and are not
ruled out by anything here.

---

## I. What to tell the professor

> You suggested trying band widths above and below 0.1. Both directions were
> tested — 0.05, 0.10, 0.20, 0.30 — over the same 934 leave-one-patient-out
> folds, 4,670 fits, all converged, nothing failed.
>
> **Narrower is worse.** At 0.05 the alphabet grows to 73 categories and 224
> parameters, 11 of the categories are empty, and PR-AUC drops against the 0.10
> baseline by more than noise (−0.0068, interval excluding zero, though only
> barely). There is no metric on which it is the best choice.
>
> **Wider is slightly better, and cheaper.** At 0.30 the model uses 77
> parameters instead of 137 and improves on AUC-ROC (+0.0026), PR-AUC (+0.0125),
> partial AUC (+0.0068), Brier and log loss, with intervals excluding zero on
> the first three. So if the banded model is kept, 0.30 is the width to keep —
> better and simpler at once.
>
> **But the headline does not change.** No width improves next-visit accuracy:
> every accuracy interval in the study includes zero, and the plain subtype-only
> model still scores highest at 83.69%. And the AUC advantage of banding turns
> out to live entirely above a 20% false-positive rate. In the region anyone
> would actually operate in, no band width is distinguishable from not banding
> at all — the best one is ahead by 0.0009 with an interval spanning zero.
> Banding also costs calibration: every banded arm is three and a half to four
> and a half times worse calibrated than the unbanded model.
>
> **So the band width was not the problem.** The previous result held because
> the ratio's extra resolution does not help this prediction, not because 0.1
> was the wrong number. The one open question the data cannot settle is whether
> the improvement keeps running past 0.30 or is just converging back toward the
> unbanded model; a single run at 0.50 would tell us, and I can do that if it is
> worth knowing.

---

## Limitations

- **Selection on the evaluation set.** Four configurations were compared on
  held-out data and a winner named. The winner's held-out metrics are
  optimistically biased. Nothing was tuned against held-out performance inside
  the fits, but the final choice is a post-hoc one.
- **The secondary contrasts in §F were added after the primary analysis.** They
  refit nothing and the primary numbers are unchanged, bitwise, but they were
  not pre-specified.
- **Threshold-on-grid confound** (§H): straddling-band counts are 1, 2, 2, 1,
  not monotone in width.
- **Small strata.** The Indeterminate stratum has 270 cases and the four-prior-visit
  stratum 305. Within-stratum differences between widths sit in the third decimal
  and should not be read as orderings.
- **One cohort, one panel, one target.** Everything here is next-visit prediction
  on the frozen PPMI panel (md5 `fb038dd…`). Nothing is externally validated.
- **The widths tested do not bracket the optimum** if one exists above 0.30.

## Reproduction

```
python scripts/90_cthmm_ratio_band_width_sensitivity.py \
    --out experiments/<UTC timestamp>_cthmm_ratio_band_width_sensitivity --workers 24
```

Roughly 1h45m on 24 cores for the fits; `--resume` reuses the fold checkpoint and
redoes the analysis and figures in under two minutes. Deterministic: seed 42,
2,000 bootstrap resamples, verified bitwise reproducible across reruns.

Run on record: `experiments/20261105T003425Z_cthmm_ratio_band_width_sensitivity`
(git `44c6cf6`, 32 files under `MANIFEST.sha256`, all verifying).

Code: `src/ppmi_subtype/ratio_bands.py` (width-parameterised `main_edges`),
`src/ppmi_subtype/next_visit_banded.py` (`run_fold_widths`),
`src/ppmi_subtype/ranking_metrics.py` (AUC, PR-AUC, partial AUC and the paired
patient bootstrap, following the conventions of
`scripts/66_roc_auc_comparison.py`).
Tests: `tests/test_band_width.py`, `tests/test_ranking_metrics.py`.
