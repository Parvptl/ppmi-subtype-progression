# Note: how to read the two headline accuracy numbers

*Clarification to read alongside the manuscript. Nothing here is a new result;
it explains numbers already in the report. Both methods have a headline figure
that looks stronger than the underlying model skill, for a different structural
reason in each case.*

## Short version

**The 83.7% is the same model as the 75.4%.** It is not an improved model and
should not be presented as one. The entire difference is that the newer
evaluation **excludes the visits whose next label is Indeterminate**, and those
were the cases the model always got wrong.

**Both methods are evaluated the same way.** The rich-profile model does *not*
score the Indeterminate cases either — it applies exactly the same exclusion.

## The arithmetic

Re-scoring the new run's predictions by the old 3-class rule reproduces the old
figure exactly — same model, same 934 patients, same 3,361 predictions:

| | 3-class score | 2-class score |
|---|---|---|
| CT-HMM | **0.7539** (2,534 / 3,361) | **0.8369** (on 3,029) |

```
3-class, all 3,361 cases:            2,534 / 3,361 = 0.7539
drop the 332 Indeterminate targets:  2,534 / 3,029 = 0.8366
renormalise TD/PIGD (flips a few):                   0.8369
```

## Why Indeterminate is always wrong

| Class | Support | Recall | Times predicted |
|---|---|---|---|
| TD | 2,115 | 0.910 | 2,450 |
| **Indeterminate** | **332** | **0.000** | **1** |
| PIGD | 914 | 0.666 | 910 |

The model predicted Indeterminate **once in 3,361 opportunities**. Indeterminate
is the middle band of a ratio (0.90 ≤ ratio ≤ 1.15) squeezed between TD and
PIGD on either side, so it is essentially never the most probable of the three.
All 332 such cases therefore counted as errors in the 3-class score.

## The correction: the richer representation does not fix this

It would be natural to assume the rich-profile model handles Indeterminate
better, since it never thresholds the measurements into a label internally.
**It does not.** Measured on the 3,352 matched cases:

| | 3-class accuracy | Indeterminate recall | Times it predicted Indeterminate |
|---|---|---|---|
| CT-HMM | 0.7539 | 0.000 | 1 of 3,352 |
| Rich profile | 0.7569 | 0.000 | 2 of 3,352 |

Both are blind to Indeterminate in the same way, and the 3-class accuracies are
within 0.3 points of each other. The rich profile does assign slightly more
probability mass to Indeterminate at its peak (max 0.454 against 0.369), but
never enough to make it the prediction.

The likely reason is that a visit landing in the 0.90–1.15 band is largely
**measurement noise in the ratio**, not a stable clinical state that can be
anticipated from the previous visit.

## What the rich representation actually contributes

Not better Indeterminate handling. Its case rests on three other things:

1. **It predicts nine clinical measurements**, not one label — a target that is
   measured rather than model-generated.
2. **It is a genuinely richer description of the patient** — seven principal
   components are needed for 90% of its across-patient variance, so it is not a
   single severity axis.
3. **It carries clinical information the label does not** — single coordinates
   beat the subtype label by 0.105–0.161 AUC on prospective Hoehn & Yahr ≥ 3
   and MCI, outcomes never used in fitting.

On the subtype endpoint itself the two methods are **statistically
indistinguishable**: 0.8371 against 0.8404 on 3,020 paired held-out cases,
difference +0.0033 (95% CI −0.0050 to +0.0112). Neither clearly beats a
persistence rule at 0.8338, because the subtype label is highly persistent.

## The same caution applies to the rich profile's own headline

The rich profile's primary target is not the subtype label at all — it is the
next visit's nine measurements. That number has its **own** inflation to watch
for, and it is the same shape of problem.

Predicted against actual, the model looks strong: correlation around 0.75 in
every domain, mean R² = 0.542. But "assume the patient has not changed" already
reaches R² = 0.485:

| Domain | r (model) | R² model | R² assume-no-change | R² population mean | model − no-change |
|---|---|---|---|---|---|
| brady/rigidity | 0.765 | 0.548 | 0.502 | 0.100 | +0.047 |
| tremor | 0.762 | 0.567 | 0.515 | −0.005 | +0.052 |
| PIGD | 0.746 | 0.530 | 0.500 | 0.112 | +0.031 |
| MDS-UPDRS II | 0.779 | 0.572 | 0.544 | 0.099 | +0.028 |
| MDS-UPDRS I | 0.748 | 0.539 | 0.480 | 0.043 | +0.060 |
| MoCA | 0.724 | 0.521 | 0.465 | −0.007 | +0.056 |
| GDS | 0.692 | 0.449 | 0.364 | −0.001 | +0.085 |
| SCOPA-AUT | 0.799 | 0.623 | 0.562 | 0.029 | +0.061 |
| ESS | 0.731 | 0.524 | 0.432 | 0.031 | +0.092 |
| **Mean** | **0.750** | **0.542** | **0.485** | **0.044** | **+0.057** |

**89.5% of the model's R² is already achieved by assuming nothing changes.**
Meanwhile a population-average trajectory reaches only R² = 0.044, so almost all
the explanatory power is "this patient is roughly where they were last time",
not "we understand how the disease progresses".

This is why the manuscript reports the **paired contrast against
assume-no-change** (+0.0733 log-density, 95% CI +0.0657 to +0.0813) as the
headline for this method rather than a correlation or an R². The correlation
is real but it mostly measures persistence, exactly as the 83.7% mostly
measured the removal of an unpredictable class.

Within that +0.0733, the components break down as:

| Component | Contribution | 95% CI |
|---|---|---|
| Rate of change | +0.0040 | +0.0019 to +0.0062 |
| Medication input | +0.0058 | +0.0041 to +0.0075 |
| Remainder (filtering the level across history) | ≈ +0.064 | — |

So the rate of change — the part that would represent actual *progression* —
contributes about **5%** of an already modest gain.

## The one-sentence summary

> The jump from 75.4% to 83.7% is a change of evaluation task, not a better
> model; both methods exclude the ~10% of visits whose next label is
> Indeterminate, because neither can predict that class, and on the remaining
> two-class task the two methods are statistically indistinguishable from each
> other and barely ahead of a persistence rule.

And the companion sentence for the second method:

> The rich profile's apparent strength on clinical measurements (r ≈ 0.75) is
> likewise mostly persistence — assuming no change already achieves 89.5% of
> its R² — so the honest figure is the +0.0733 log-density it adds over that
> baseline, of which the rate-of-change component supplies about 5%.

Both headline numbers are inflated by a structural property of the data rather
than by model skill, and in both cases the manuscript reports the paired
baseline contrast instead.

*Sources: `experiments/20261020T104756Z_next_visit_td_pigd_prediction/`,
`experiments/20261020T180515Z_clinical_profile_cv/`, and the earlier 3-class run
in `outputs/loo/`.*
