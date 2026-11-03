# E-8 — the harmonised multi-family comparison

**Date** 2026-11-04 · **Script** `scripts/79_harmonised_family_comparison.py`
**Run** `experiments/20261103T202632Z_e8_harmonised_family_comparison/` · **Checks 11/11**
**Models fitted: 0.** Re-scoring of saved predictions only; nothing under
`outputs/` was written.

Unblocks the primary analysis identified in
`04_SCIENTIFIC_POSITIONING_20261104.md` §6.

---

## 1. What was wrong, and what this fixes

Nine model specifications across five families were all scored on the same
3,361 held-out transitions from the same 934 patients — but under two different
label-derivation rules, and the rule was worth more than the differences
between the families. This re-scores every family onto **one 3-class
probability vector, one decision rule, one canonical baseline and one metric
suite**.

| Harmonised | How |
|---|---|
| Representation | Every model expressed as a probability vector over (TD, Indeterminate, PIGD). The chains and HMMs already are; the subscore models are converted by Monte Carlo through the project's unmodified `subtype.classify`. |
| Decision rule | Argmax of that vector, for every model without exception. |
| Canonical baseline | **Label persistence** — name the currently recorded label. The trivial rule a clinician would use; needs no subscores. The subscore-LOCF baseline is reported separately because it is a different construction, and conflating the two is how the comparison went wrong. |
| Uncertainty | Patient-clustered bootstrap, 2,000 draws, paired on identical cases. |

**Two decisions change published numbers, both deliberate and both disclosed.**

1. **Both-zero draws go to Indeterminate, not TD.** `scripts/18` computes the
   draw ratio as `np.divide(..., out=inf, where=draw_pigd > 0)`, so a draw with
   both subscores zero gets `ratio = inf` and is classified TD. The project's
   own `subtype.classify` sends tremor = 0 and PIGD = 0 to **Indeterminate**.
   0.59% of draws are affected, touching 2,230 of 3,361 cases and changing the
   argmax label on 14. This script follows `classify`, and an equivalence test
   against the real function is a validation check rather than an assertion.
2. **A fresh Monte Carlo with common random numbers.** The saved draws cannot
   be reproduced bit-for-bit because the RNG stream differs. Every model
   therefore sees the identical standard-normal draws for the identical case —
   seeded per chunk, not per model — so Monte Carlo error is shared and
   largely cancels in paired contrasts. Halving the draws moves accuracy by at
   most 0.004.

The pipeline is validated by the fact that **every model already on a 3-class
vector reproduces its committed accuracy exactly** (max |difference| 4 × 10⁻⁵).
Only the subscore models move, which is the point.

---

## 2. The harmonised table

Identical 3,361 transitions, 934 patients, one decision rule.

| Model | Family | Accuracy | Brier | Log loss | Macro-F1 | Indeterminate recall | Times it predicted Indeterminate |
|---|---|---:|---:|---:|---:|---:|---:|
| Subscore GP, independent | subscore GP | **0.7575** | 0.3602 | 0.6537 | 0.5074 | 0.000 | 0 |
| Subscore GP, joint | subscore GP | 0.7563 | 0.3594 | 0.6518 | 0.5070 | 0.000 | 0 |
| CT-HMM, 3 latent states | latent-state HMM | 0.7539 | 0.3688 | 0.6726 | 0.5037 | 0.000 | 1 |
| CT-HMM, 5 latent states | latent-state HMM | 0.7513 | 0.3700 | 0.6746 | 0.5011 | 0.000 | 0 |
| Time-varying LMM, constant | time-varying LMM | 0.7495 | 0.3826 | 0.6919 | 0.4977 | 0.000 | 0 |
| Joint subscore LMM | subscore LMM | 0.7495 | 0.3826 | 0.6919 | 0.4977 | 0.000 | 0 |
| Markov chain, discrete | observed-label chain | 0.7474 | 0.3896 | 0.7066 | 0.4929 | 0.000 | 0 |
| Time-varying LMM, spline | time-varying LMM | 0.7462 | 0.3794 | 0.6860 | 0.4937 | 0.000 | 0 |
| CTMC, interval-aware | observed-label chain | 0.7453 | 0.3925 | 0.7105 | 0.4909 | 0.000 | 4 |
| *Persistence (subscore LOCF)* | *baseline* | *0.7453* | *0.3723* | *0.7556* | *0.4970* | *0.000* | *1* |
| *Patient mean subscore* | *baseline* | *0.7423* | *0.3802* | *0.7594* | *0.4903* | *0.003* | *3* |
| **Persistence (recorded label)** | **baseline** | **0.7227** | 0.5546 | 7.6620 † | **0.5559** | **0.193** | **334** |
| *Marginal null* | *baseline* | *0.6293* | *0.5211* | *0.8756* | *0.2575* | *0.000* | *0* |
| *Population mean subscore* | *baseline* | *0.6254* | *0.5174* | *0.8716* | *0.3460* | *0.000* | *0* |

† A rule, not an estimate: persistence emits 0/1 probabilities, so its log loss
is finite only through clipping and is not comparable with the fitted models.

**Three readings.**

**The spread across the nine fitted specifications is 1.2 accuracy points**
(0.7453 to 0.7575), not the roughly five points the committed per-family
figures implied. Most of the apparent difference between families was the
decision rule. Five families spanning observed-label chains, latent-state
HMMs, a hierarchical subscore LMM, Gaussian processes and a time-varying
spline land inside a one-point band.

**The trivial rule has the best macro-F1 of all fourteen specifications**
(0.5559 against a best fitted value of 0.5074), because it is the only one that
ever meaningfully predicts Indeterminate. Of fourteen specifications, five
predict Indeterminate at all, and four of those do so between one and four
times in 3,361 opportunities. Only persistence does it 334 times — and it is
right 19.3% of the time, which is still the best Indeterminate recall anyone
achieves.

**Two independent runs agree exactly.** The joint subscore LMM and the
time-varying LMM's constant-rate variant are the same model fitted in two
separate experiments, and they reproduce each other to four decimals on
accuracy, Brier and log loss. That is a free internal consistency check and it
passes.

---

## 3. Paired contrasts against the trivial rule

Identical cases, patients resampled, one resample shared by both conditions.

| Model | Δ accuracy | 95% CI | Excludes zero |
|---|---:|---|---|
| Subscore GP, independent | **+0.0348** | +0.0252 to +0.0450 | yes |
| Subscore GP, joint | +0.0336 | +0.0243 to +0.0436 | yes |
| CT-HMM, 3 latent states | +0.0312 | +0.0223 to +0.0411 | yes |
| CT-HMM, 5 latent states | +0.0286 | +0.0196 to +0.0381 | yes |
| Joint subscore LMM | +0.0268 | +0.0136 to +0.0398 | yes |
| Time-varying LMM, constant | +0.0268 | +0.0136 to +0.0398 | yes |
| Markov chain, discrete | +0.0247 | +0.0159 to +0.0334 | yes |
| Time-varying LMM, spline | +0.0235 | +0.0102 to +0.0364 | yes |
| CTMC, interval-aware | +0.0226 | +0.0135 to +0.0316 | yes |
| *Persistence (subscore LOCF)* | *+0.0226* | *+0.0146 to +0.0311* | *yes* |
| *Marginal null* | *−0.0934* | *−0.1202 to −0.0691* | *yes* |

### This corrects something the project believed

Stage 4 recorded — from the committed per-family figures — that two families
fell **below** the persistence rule: the joint subscore LMM at 0.7087 and the
time-varying LMM at 0.7087/0.7093, against persistence at 0.7227. **Under the
harmonised rule neither does.** They score 0.7495 and 0.7462, and both beat
persistence with an interval excluding zero. The entire apparent deficit was
the label-derivation rule, which cost those families 3.3 to 3.7 accuracy
points.

**All nine fitted specifications beat the trivial rule, every interval
excluding zero.** That is a real change from what the project previously
believed, and the manuscript must carry the corrected version.

---

## 4. Where the margin comes from — the result that matters

The contrasts above are pooled. Splitting them by the label recorded at the
prediction point localises the entire effect. For the CT-HMM against
persistence:

| Current recorded label | n | Share of cases | CT-HMM | Persistence | Cases gained | Identical predictions |
|---|---:|---:|---:|---:|---:|---:|
| TD | 2,261 | 67.3% | 0.8107 | 0.8080 | **+6** | 98.4% |
| **Indeterminate** | **334** | **9.9%** | **0.4880** | **0.1916** | **+99** | **0.3%** |
| PIGD | 766 | 22.8% | 0.7023 | 0.7023 | **+0** | 96.9% |

**94.3% of the CT-HMM's entire advantage over the trivial rule comes from 9.9%
of the data.** On the other 90% it gains six cases out of 3,027 — and on
current-PIGD cases the two are **exactly tied**, 0.7023 against 0.7023, while
making the same prediction 96.9% of the time.

The discrete Markov chain makes the point even more sharply. It predicts
**exactly what persistence predicts on every single current-TD and
current-PIGD case** — 100.0% identical on 3,027 of 3,361 cases — and gains 83
cases entirely within the Indeterminate band. **100% of its advantage comes
from 9.9% of the data.** The subscore GP is the only family that gains
anywhere else, and it gains nine cases on current-TD and eight on current-PIGD
out of 3,027.

| Model | Gain on current-TD | Gain on current-Indeterminate | Gain on current-PIGD | Share from Indeterminate |
|---|---:|---:|---:|---:|
| CT-HMM, 3 latent states | +6 | +99 | +0 | **94.3%** |
| Subscore GP, independent | +9 | +100 | +8 | 85.5% |
| Markov chain, discrete | +0 | +83 | +0 | **100.0%** |

The mechanism is not subtle. When the recorded label is Indeterminate,
persistence repeats it; the fitted models essentially never predict
Indeterminate, so they guess TD or PIGD instead, and they are right about half
the time because the Indeterminate label is almost never repeated at the next
visit. The models are not predicting progression better. **They are declining
to believe an unreliable class, and that is the whole of their advantage.**

That is the project's thesis stated as precisely as the data allows, and it is
stronger evidence for it than the pooled comparison was.

---

## 5. What this changes for the manuscript

| Claim | Before | After |
|---|---|---|
| Spread across model families | ~5 accuracy points | **1.2 points** across nine specifications |
| Two families fall below persistence | Asserted from committed figures | **Wrong.** All nine beat it; the deficit was the derivation rule |
| Models beat persistence | "Indistinguishable" (true of the 2-class task) | **+0.023 to +0.035 on the 3-class task, all intervals excluding zero** |
| Why they beat it | Not analysed | **94.3% from the 9.9% of cases where the current label is Indeterminate** |
| Indeterminate | "No model recovers it" | Confirmed, and sharpened: **the trivial rule has the best macro-F1 of all fourteen specifications** because it is the only one that tries |

The 2-class and 3-class statements are both true and are not in conflict. On
the two-class task, with Indeterminate targets excluded, the CT-HMM and
persistence are indistinguishable (+0.0033, CI −0.0037 to +0.0100). On the
three-class task they differ by +0.0312 — and the decomposition shows that the
entire difference is the Indeterminate band, which the two-class task removes.
**The two results are the same fact seen from two sides**, and the manuscript
should present them together rather than choosing one.

---

## 6. Limitations

1. **Re-scoring, not re-fitting.** Every model's parameters, folds and
   predictions are those of the original runs. This changes how they are
   scored, not what they predicted.
2. **The subscore models' probabilities are a fresh Monte Carlo**, not the
   saved ones, and cannot be bit-reproduced. Common random numbers keep the
   contrasts clean; halving the draws moves accuracy by at most 0.004.
3. **Independence is assumed for the LMM and time-varying families**, whose
   saved files carry no cross-outcome correlation column. Their own reports put
   that correlation near zero, but the assumption is mine and is recorded as
   such. Only the GP family uses the joint covariance.
4. **Validation schemes still differ in kind.** Every model here is
   leave-one-patient-out, so this table is internally consistent — but it
   excludes the rich clinical profile and the CT-HMM next-visit work, which use
   5-fold cross-validation and a two-class target. Those belong in a separate
   table and must not be appended to this one.
5. **The decomposition in §4 is descriptive**, computed on the same data it
   describes, with no multiplicity correction. It is a localisation of an
   already-established contrast, not a new hypothesis test.
6. **One quirk in `scripts/18` is corrected here but not there.** The committed
   GP figures retain the both-zero-to-TD behaviour. The difference is small
   (+0.0009 and +0.0015 on accuracy) but the two sets of numbers are not
   identical, and the manuscript should quote these.
