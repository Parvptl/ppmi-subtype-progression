# PPMI Motor Subtype Progression: An Explanatory Report from the GPR Stage Onwards

**Project:** Longitudinal modelling of Parkinson's disease motor subtype progression using PPMI data

**Institution:** Indian Institute of Technology Palakkad

**Date of this report:** 9 October 2026

---

## How to read this report

This report explains everything we did from the Gaussian Process Regression (GPR)
stage onwards. It is written so that you can explain the work aloud, defend the
decisions, and answer questions about why we rejected certain approaches.

Every number in this report was checked against the saved output files in the
repository. Where a number could not be verified from a saved file, it is
labelled clearly.

Three rules are used throughout:

1. **Description is not prediction.** A model can tell us something true about
   how the measurements behave, and still not predict the next visit better.
2. **A metric must always come with its comparison.** We never say a model is
   "better" without saying better on which metric, against which model, on which
   patients.
3. **Negative results are reported in full.** Most of the work after GPR produced
   negative or null results. Those are real findings, not failures.

---

# Part A — Where we stood before GPR

## A.1 Why study motor subtypes over time

Parkinson's disease affects people differently. Some patients have more tremor
(shaking). Others have more problems with walking, balance and posture. Doctors
often divide patients into motor subtypes based on which group of symptoms
dominates.

If a patient's subtype changes over the years, that may tell us something about
how their disease is progressing. So we wanted to study how subtype changes over
time, using the PPMI dataset (Parkinson's Progression Markers Initiative), which
follows patients over many years.

## A.2 What the Tremor/PIGD ratio is

The subtype is not measured directly. It is computed from the MDS-UPDRS, a
standard clinical examination with many scored items.

We compute two numbers from the exam:

- **Tremor score** = average of 11 tremor-related items
- **PIGD score** = average of 5 items about posture, gait and balance
  (PIGD stands for Postural Instability and Gait Difficulty)

Then:

```
ratio = Tremor score / PIGD score
```

The ratio is a single number saying which side dominates. A high ratio means
tremor dominates. A low ratio means gait and balance problems dominate.

This definition follows Stebbins et al., *Movement Disorders* 2013, and is
recorded in `config/config.yaml` in our repository.

## A.3 What TD, Indeterminate and PIGD mean here

| Label | Rule | Plain meaning |
|---|---|---|
| **TD** | ratio >= 1.15 | Tremor-dominant |
| **Indeterminate** | 0.90 < ratio < 1.15 | In between; neither side clearly dominates |
| **PIGD** | ratio <= 0.90 | Gait and balance problems dominate |

Two special cases are handled explicitly in our code:

- If PIGD score is 0 and tremor score is above 0, the patient is labelled TD.
- If both scores are 0, the patient is labelled Indeterminate.

## A.4 Why predicting the next subtype was hard

Earlier phases of the project established three facts that shaped everything
afterwards.

**Fact 1: The labels are noisy.** About 25 percent of observed subtype changes
reverse at the very next visit. A patient recorded as changing from TD to PIGD
is often back at TD at the following visit. So a single recorded change is weak
evidence of real progression.

**Fact 2: The middle class is almost impossible to hit.** The Indeterminate band
is only 0.25 wide on the ratio scale (from 0.90 to 1.15). But the ratio moves a
lot between visits. Our hierarchical Bayesian model estimated a total residual
spread of 1.138. That is about 4.6 times the width of the band. So even a
perfect trajectory estimate would land almost anywhere.

**Fact 3: A trivial rule is already strong.** Simply predicting "the patient
stays in the same subtype" gets 72.3 percent accuracy. Our best model at the
time, the hidden Markov model (HMM), got 75.4 percent. That is only about 3
points better.

## A.5 What the hierarchical Bayesian model taught us

Before GPR, we fitted a hierarchical Bayesian growth model on the ratio. "Growth
model" here means a straight-line trend over time, with each patient allowed
their own starting point and their own slope.

It found (verified in `outputs/hbayes/report.txt`):

- A population trend of **-0.0927 per year** in the ratio, with a 95 percent
  credible interval of -0.1159 to -0.0682. So on average the ratio falls.
- A residual spread of **1.138**, much wider than the Indeterminate band.

This was genuinely useful. It gave us an interpretable number for how fast the
ratio changes. But it did not help us predict the next subtype label.

**This is the key lesson to carry forward:** a model can describe a real trend in
the measurements and still be poor at predicting the next clinical category. The
two tasks are different.

## A.6 Why we decided to try GPR

The hierarchical Bayesian model assumes each patient follows a straight line. The
real question was: is a straight line too rigid?

If patients' ratios wander up and down in smooth but non-linear ways, a more
flexible model might track them better. A Gaussian process is exactly a way of
fitting smooth, flexible curves without deciding the shape in advance.

So the question for the GPR stage was simple:

> Given a patient's own earlier ratio measurements, can a flexible curve predict
> their next ratio better than a straight line can?

---

# Part B — Gaussian Process Regression

## B.1 What regression means in this project

Regression means predicting a number. Here the number is a patient's ratio (or,
later, their tremor and PIGD scores) at their next visit.

This is not the same as predicting their subtype label. Predicting a number is
regression. Predicting a label from three choices is classification. We did both,
but they are evaluated separately and must never be mixed in one comparison.

## B.2 What a Gaussian process is, in simple terms

A normal regression model says: "I think the trajectory is a straight line.
Please estimate its slope."

A Gaussian process says something different: "I think the trajectory is some
smooth curve. I will not say what shape. Instead I will say how strongly I expect
the value at one time to be related to the value at a nearby time."

That second statement is the whole model. It is a probability distribution over
*functions*, not over parameters.

A small example. Suppose a patient has a ratio of 2.0 at year 1. What do we
expect at year 1.2? Probably something close to 2.0, because two months is a
short time. What about at year 8? We have much less idea. A Gaussian process
encodes exactly this intuition, and it does so through the kernel.

## B.3 What a kernel is

The **kernel** is a function that says how related two time points are.

If the kernel value between year 1 and year 1.2 is high, the model expects those
two measurements to be similar. If the kernel value between year 1 and year 8 is
near zero, the model treats them as almost unrelated.

The kernel controls what kinds of trends the model can learn. This is the single
most important design choice in a Gaussian process.

Each kernel is built from simple pieces:

| Piece | What it lets the model represent |
|---|---|
| **constant** | Each patient has their own fixed offset, higher or lower than average |
| **linear** | Each patient has their own straight-line slope |
| **RBF** | Smooth wiggles that fade away over a learned time scale |
| **Matern-3/2** | Similar to RBF, but allows rougher, less smooth wiggles |

## B.4 The kernels we actually tested

We tested four combinations, named K0 to K3:

| Name | Pieces | What it can represent |
|---|---|---|
| **K0** | constant + linear | Own offset and own straight-line slope |
| **K1** | constant + RBF | Own offset, plus smooth wiggles |
| **K2** | constant + linear + RBF | Both of the above |
| **K3** | constant + linear + Matern-3/2 | Both, with rougher wiggles |

**K0 was included deliberately as a control.** A model where each patient has a
random starting point and a random slope is mathematically the same as the
hierarchical Bayesian growth model we already had. So if our GP code was correct,
K0 had to reproduce the earlier model. It did. This was our check that the new
code was not silently wrong.

## B.5 How prediction worked

For a held-out patient, we predict visit *k* using only visits 0 to *k-1* of that
same patient. The model's shared settings (the kernel parameters and the average
trend) come from all the *other* patients.

So the patient's own history is used to personalise the prediction, but the
patient never influences the fitted model. This is the honest way to evaluate
"forecasting for a patient who already has some follow-up".

## B.6 How patient-level evaluation was done

We used **leave-one-patient-out (LOPO)** validation.

The procedure: take one patient out. Fit the model on all the remaining patients.
Predict the held-out patient's visits. Put them back. Repeat for every patient.

This is different from randomly splitting visits. If we split visits randomly,
the same patient would appear in both training and test data, and the results
would look better than they really are. Splitting by patient avoids this.

A battery of ten automated checks ran on every saved prediction row, confirming
things like "the held-out patient was never in training" and "the history is
strictly before the target visit".

## B.7 What MAE means

**MAE** stands for Mean Absolute Error. It is the average size of the mistake,
ignoring whether the mistake was too high or too low.

Example: if the true ratio is 2.0 and we predict 2.3, the absolute error is 0.3.
Average that over all predictions and you get MAE.

Lower MAE is better. MAE is in the same units as the thing being predicted, so an
MAE of 0.97 on the ratio means we are typically wrong by about 0.97 ratio units.

## B.8 What persistence means, and why it is strong

**Persistence** is the simplest possible rule:

> If a patient's latest observed value is 0.6, predict that the next value will
> also be 0.6.

It has no parameters. It does not learn anything. It just repeats the last
measurement.

Why is this surprisingly strong for clinical data? Three reasons:

1. **Diseases change slowly.** Over one year, most patients do not change much.
   Last year's value is genuinely a good guess for this year's.
2. **It uses the most recent information.** A fitted model pulls predictions
   towards the population average. Persistence does not. For a patient who is far
   from average, persistence is often closer.
3. **It has nothing to overfit.** Fitted models can be dragged off by noise in
   the training data. Persistence cannot.

This is why we compare everything against persistence. Beating a fitted model is
easy. Beating "nothing changed" is the real test.

## B.9 GPR results on the continuous ratio

Cohort: **821 patients, 2,707 prediction points.** This is smaller than the full
934-patient cohort because the ratio does not exist when PIGD is zero.

Verified from `outputs/step3_gpr/reconciled_metrics.csv`:

| Model | MAE (pooled) | MAE (per patient) | RMSE | CRPS |
|---|---:|---:|---:|---:|
| **K3** const+linear+Matern | **0.9715** | **1.0006** | 1.3565 | 0.7247 |
| **K1** const+RBF | 0.9720 | 1.0012 | 1.3566 | 0.7249 |
| **K2** const+linear+RBF | 0.9720 | 1.0012 | 1.3566 | 0.7249 |
| K0 const+linear | 1.0053 | 1.0204 | 1.3870 | 0.7472 |
| hbayes (growth model) | 1.0062 | 1.0176 | 1.3910 | 0.7480 |
| persistence | 1.0427 | 1.0597 | 1.5396 | 0.8182 |
| patient mean | 1.0627 | 1.0649 | 1.4714 | 0.8003 |
| population only | 1.3949 | 1.4130 | 1.8008 | 0.9836 |

Two MAE columns are shown because there are two fair ways to average. "Pooled"
treats every prediction equally, so patients with many visits count more. "Per
patient" gives every patient equal weight. We report both because the choice
changes one conclusion later.

The key paired comparisons, verified from
`outputs/step3_gpr/paired_comparisons.csv`:

| Question | Difference (pooled) | 95% interval | Conclusion |
|---|---|---|---|
| Does a flexible kernel beat the linear one? (K1 - K0) | **-0.0333** | [-0.0468, -0.0205] | Yes, clearly |
| Does the GP beat persistence? (K1 - persistence) | **-0.0707** | [-0.0979, -0.0451] | Yes, clearly |
| Does Matern beat RBF? (K3 - K1) | -0.0005 | [-0.0015, +0.0005] | No difference |
| Does the GP reproduce the growth model? (K0 - hbayes) | -0.0009 | [-0.0033, +0.0014] | Indistinguishable |

Negative numbers mean the first model has lower error, so it is better.

## B.10 GPR results on the derived subtype

Here is the result that shaped the rest of the project.

We converted the predicted ratio into a predicted subtype and measured accuracy
on the same 2,707 points:

| Model | Ratio MAE | Derived subtype accuracy |
|---|---:|---:|
| K3 | **0.9715** (best) | 0.7270 |
| K1 / K2 | 0.9720 | 0.7237 |
| K0 | 1.0053 | 0.7141 |
| **persistence** | 1.0427 (worst of these) | **0.7329** (best) |

Read this table carefully. **Persistence has the worst MAE and the best subtype
accuracy.** The GP predicts the number better, and the category worse.

Also, the Indeterminate recall for every GP kernel was **0.000**. The GP never
correctly identified a single Indeterminate visit.

## B.11 What GPR improved and what it did not

**Improved:**

- Predicting the continuous ratio. The GP beat persistence by 0.0707 MAE and beat
  the linear model by 0.0333, both with intervals excluding zero.
- Calibration and uncertainty. The GP's prediction intervals covered the truth
  about 94 percent of the time, close to the intended 95 percent.
- Confirmed our code was correct, because K0 reproduced the existing growth
  model.

**Did not improve:**

- Subtype prediction. Persistence was better.
- The Indeterminate class. Never predicted.
- The practical size of the gain. The kernel choice is worth about 0.03 ratio
  units. Knowing *which patient* it is, is worth about 0.4. The within-patient
  spread of the ratio is 1.22. So kernel refinement operates at roughly a tenth
  of the individualisation effect and a fortieth of the noise.

---

# Part C — Why we moved from the ratio to direct Tremor/PIGD modelling

## C.1 The ratio hides severity

A ratio is one number made from two. That means information is lost.

Consider two patients:

| Patient | Tremor | PIGD | Ratio |
|---|---:|---:|---:|
| A | 2 | 4 | 0.5 |
| B | 10 | 20 | 0.5 |

Both have a ratio of 0.5. Both are labelled PIGD. But patient B has five times
the absolute symptom burden of patient A. They are not the same patient, and a
model working only on the ratio cannot tell them apart.

## C.2 The ratio is sometimes undefined

If the PIGD score is exactly 0, the ratio requires dividing by zero. It does not
exist.

In our 934-patient cohort this happens at **692 of 4,295 visits**, which is 16.1
percent. And it is not random: **349 patients (37.4 percent)** have at least one
such visit, and **38 patients** have PIGD zero at every single visit.

Worse, these visits are overwhelmingly tremor-dominant. So a ratio-based model is
blind to a slice of the data that is systematically TD. The cohort shrinks from
934 patients to 821, and the missing 113 are not a random sample.

## C.3 What we wanted to test

The midterm presentation listed this as the first future direction:

> "Model the two subscores separately, not their ratio."

What we wanted to test: does modelling tremor and PIGD directly keep information
that the ratio throws away?

What we expected might improve: more data (the 692 recovered visits), less
distortion from the division, and possibly better predictions.

What the experiments actually showed is covered in Parts D and E. The short
version: the extra coverage was real, the predictive improvement was not.

---

# Part D — Step 1: Joint Linear Mixed-Effects Model

## D.1 Purpose

Before trying anything flexible, we wanted a simple, interpretable baseline that
modelled tremor and PIGD directly. "Interpretable" means we can read the fitted
numbers and say what they mean in ordinary language.

## D.2 Four ideas that must be kept apart

| Term | What it means | Example |
|---|---|---|
| **Population-level trend** | The average rate of change across all patients | "On average, PIGD rises 0.066 per year" |
| **Patient-specific baseline** | How high or low one patient starts, compared to average | "This patient starts 0.3 above average" |
| **Patient-specific slope** | How fast one patient changes, compared to average | "This patient worsens faster than average" |
| **Random effects** | The statistical device that lets each patient have their own baseline or slope, while still sharing information across patients | — |

Random effects are important because each patient has few visits. Fitting a
completely separate model per patient would be hopeless with four visits. Random
effects let each patient borrow strength from the group.

## D.3 The model we actually implemented

For patient *i* at time *t*:

```
Tremor_it = alpha_T + beta_T * t + b_Ti + error_Tit
PIGD_it   = alpha_P + beta_P * t + b_Pi + error_Pit
```

In words: each measurement equals a population starting value, plus a population
trend times elapsed time, plus a patient-specific offset, plus noise.

**Important: the Step 1 model used random intercepts only. It did not use random
slopes.** Each patient gets their own starting level, but every patient shares the
same slope.

Why? Because the median patient has only four visits. Estimating a reliable slope
from four noisy points is not possible, and Direction 3 later confirmed this
directly. Adding random slopes would have added instability without adding
information.

Both the patient offsets and the noise were allowed to be correlated across the
two outcomes, so the model could discover whether tremor and PIGD move together.

Time is **elapsed years from each patient's own first visit**, computed as
`(visit date - first visit date) / 365.25`. Not visit number, because visits are
irregularly spaced.

## D.4 Cohort

| Quantity | Value |
|---|---:|
| Patients | 934 |
| Visits | 4,295 |
| Valid tremor observations | 4,295 (100 percent) |
| Valid PIGD observations | 4,295 (100 percent) |
| Visits dropped | 0 |
| Visits with PIGD = 0 | 692 |
| Held-out prediction points | 3,361 |

All 692 PIGD-zero visits were kept. No small number was added to the denominator
anywhere, because we never computed a ratio before fitting.

## D.5 Estimated trends

Verified from `outputs/joint_subscore/model_summary.txt`:

| Term | Estimate | 95% interval |
|---|---:|---|
| Tremor starting level | 0.5841 | [0.5594, 0.6089] |
| **Tremor trend per year** | **+0.0233** | [+0.0202, +0.0265] |
| PIGD starting level | 0.2301 | [0.2085, 0.2517] |
| **PIGD trend per year** | **+0.0655** | [+0.0621, +0.0689] |

**What this means in plain English:** both tremor and gait problems get worse over
time. But PIGD gets worse about **2.8 times faster** than tremor.

This is something the ratio could never have told us. The ratio falls by about
0.093 per year, which could mean tremor improving or PIGD worsening. Now we know:
**nothing is improving. The denominator is just growing faster than the
numerator.**

## D.6 The cross-outcome correlation was near zero

| Where | Correlation |
|---|---:|
| Between patients (random offsets) | **-0.0153** |
| Within patients (residual noise) | **+0.0186** |

The raw correlation between tremor and PIGD looks like +0.148. But once the model
accounts for the fact that both rise with time, almost all of that disappears.

This mattered enormously for Step 2, so it is worth stating clearly: the apparent
link between tremor and PIGD is mostly "both go up as time passes", not "when one
goes up, the other goes up".

## D.7 Prediction results

Verified from `outputs/joint_subscore/metrics.csv`, 3,361 points, 934 patients.

**Tremor:**

| Model | MAE | 95% interval | RMSE | 95% coverage |
|---|---:|---|---:|---:|
| persistence | **0.2426** | [0.2329, 0.2520] | 0.3271 | 0.799 |
| patient mean | 0.2498 | [0.2391, 0.2608] | 0.3313 | 0.795 |
| joint LMM | 0.2546 | [0.2458, 0.2640] | **0.3246** | **0.935** |

**PIGD:**

| Model | MAE | 95% interval | RMSE | 95% coverage |
|---|---:|---|---:|---:|
| persistence | **0.2040** | [0.1921, 0.2162] | **0.3416** | 0.848 |
| joint LMM | 0.2391 | [0.2266, 0.2516] | 0.3566 | **0.940** |
| patient mean | 0.2400 | [0.2243, 0.2557] | 0.4054 | 0.818 |

The direct comparison against persistence:

| Outcome | LMM - persistence | 95% interval | Verdict |
|---|---|---|---|
| tremor | **+0.0120** | [+0.0039, +0.0205] | LMM worse |
| PIGD | **+0.0350** | [+0.0266, +0.0436] | LMM worse |

Positive means worse. Both intervals exclude zero, so this is not a coin flip.
**The model lost to persistence on the primary metric.**

## D.8 Subtype results

| Model | Accuracy | 95% interval | Balanced acc. | Indeterminate recall |
|---|---:|---|---:|---:|
| persistence | **0.7227** | [0.7010, 0.7440] | **0.548** | **0.193** |
| patient mean | 0.7185 | [0.6972, 0.7405] | 0.522 | 0.172 |
| joint LMM | 0.7087 | [0.6865, 0.7306] | 0.527 | **0.175** |

"Balanced accuracy" averages the accuracy on each of the three classes
separately. It is a fairer measure when classes are unequal in size, which they
are here (TD is much more common).

## D.9 The PIGD = 0 visits

Among the scored points, **446 had an observed PIGD of exactly 0**. These are
visits that no ratio-based model can touch at all.

The joint LMM got **82.1 percent subtype accuracy** on them. That is new
information from data that was previously invisible.

## D.10 Why the model was still useful

It lost on MAE. So why keep it?

1. **It separated the two trends.** Tremor +0.023, PIGD +0.066. This is a
   scientific result the ratio cannot produce.
2. **It recovered 692 visits and 113 patients**, and scored them at 82 percent.
3. **It is far better calibrated.** When it says it is 95 percent confident, it
   is right about 94 percent of the time. Persistence is right only about 80
   percent of the time when making the same claim. If you want honest uncertainty
   rather than just a point guess, the model is better.
4. **It is nearly unbiased.** Persistence systematically under-predicts (bias
   -0.033 on tremor, -0.090 on PIGD) because it ignores the upward trend.
5. **It predicted Indeterminate 396 times with recall 0.175.** For comparison,
   the HMM predicts Indeterminate once in 3,361 opportunities, with recall 0.000.
   (This comparison was verified by matching the two prediction sets on the same
   3,361 points during the session; the join was complete and the observed labels
   agreed exactly.)

So the right summary is: **the model is better at describing and at quantifying
uncertainty; it is worse at point prediction.**

---

# Part E — Step 2: Multi-output Gaussian Process

## E.1 The research question

> Would predicting tremor and PIGD **together** improve prediction, compared with
> predicting them separately?

The idea behind joint modelling is "borrowing strength". If tremor and PIGD are
related, then knowing something about one tells you something about the other,
and a joint model can use that.

## E.2 Three model types

| Model | What it assumes |
|---|---|
| **Independent GPs** | Two completely separate models. Tremor knows nothing about PIGD. Each gets its own time scale and its own settings. |
| **Joint GP with no cross-output covariance ("diagonal")** | One model fitted to both outcomes at once, sharing the time scale, but the link between the outcomes is forced to zero. |
| **Joint GP with full cross-output covariance ("full")** | The same, but the link between the outcomes is estimated freely from the data. |

## E.3 What cross-output correlation means

Cross-output correlation asks: when a patient's tremor is higher than expected at
a visit, is their PIGD also higher than expected?

If the answer is yes, the correlation is positive and a joint model can use it. If
the answer is "no relationship", the correlation is near zero and joint modelling
has nothing to work with.

## E.4 Why we did not assume joint modelling would help

Step 1 had already measured this correlation: **-0.015 between patients and
+0.019 within patients**. Both are essentially zero.

So we went into Step 2 expecting the cross-output part to find nothing. We said
so in advance, and designed the experiment to test it properly rather than to
confirm a hope.

The design choice that makes this a fair test: **the "full" and "diagonal" models
differ in exactly one thing** — whether the off-diagonal terms are free or forced
to zero. Same kernel, same time scale, same everything else. So any difference
between them is caused by the cross-output link and nothing else.

Independent GPs differ in *more* than one way (they also get their own time
scale), so a gap between independent and joint cannot be blamed on the
cross-output link alone. That is exactly why the diagonal version exists.

## E.5 The mathematics, explained in words

The covariance for a patient is:

```
K[(visit j, outcome d), (visit j', outcome d')]
    = sum over kernel pieces c of  B_c[d,d'] * k_c(t_j, t_j')
      + Sigma_e[d,d'] * (1 if j = j' else 0)
```

In plain English: how related two measurements are depends on two things
multiplied together — how close in **time** they are (that is `k_c`), and how
related the two **outcomes** are (that is `B_c`). Plus measurement noise, which
can also be correlated across the two outcomes.

`B_c` is a small 2-by-2 table of numbers called the coregionalization matrix. Its
off-diagonal entry is the thing being tested.

## E.6 Computation and honesty about the environment

**No GPU was used, because none was available on this machine.** The installed
PyTorch is the CPU-only build (`2.6.0+cpu`), `torch.cuda.is_available()` returns
False, and `nvidia-smi` is not present. All computation was NumPy and SciPy on 12
CPU worker processes.

Two practical problems were hit and fixed:

1. Two runs were accidentally left running at the same time, competing for CPU
   cores. Both were stopped and the job was restarted cleanly.
2. The worker processes were each using multiple BLAS threads, oversubscribing
   the 12 cores. Pinning each worker to one thread cut the time per fold from
   17.5 seconds to about 7.3 seconds.

The final run, verified in `outputs/joint_subscore_gp/run.log`: **934 folds, 4 GP
arms per fold, 3,736 of 3,736 fits converged, total 114.3 minutes.**

**A scope compromise was made for runtime.** The K2 and K3 kernels (13 parameters
each) take 250 to 320 seconds per fit here. Across 934 folds that is more than 65
hours. So K2 and K3 were fitted **only on the whole cohort at once, for
description**. They were **not** evaluated in the leave-one-patient-out
comparison. This distinction is maintained throughout this report.

## E.7 Population-level kernel fits (description only, not held out)

Verified from `outputs/joint_subscore_gp/kernel_parameters.csv`. **These numbers
are in-sample fits on all 934 patients. They are not held-out results.**

| Kernel | Coregion | Negative log-likelihood | Parameters | Restart spread |
|---|---|---:|---:|---:|
| K0 | full / diag | -5726.35 / -5717.38 | 9 / 6 | 0.0001 / 0.0 |
| K1 | full / diag | -5643.00 / -5634.77 | 10 / 7 | 0.0 / 0.0 |
| K2 | full / diag | **-5977.29** / -5965.73 | 13 / 9 | 0.0007 |
| K3 | full / diag | **-5979.31** / -5967.61 | 13 / 9 | 0.0001 |

Lower negative log-likelihood means better in-sample fit. "Restart spread" is the
difference between the best and worst of four optimiser restarts. It is
essentially zero everywhere, which means the fitting was stable and did not get
stuck in different places on different attempts.

Freeing the cross-output terms improved the in-sample fit by only 8.2 to 11.7
units for 3 to 4 extra parameters. That is a small gain.

**One observation we are flagging rather than claiming.** On K2 and K3, the
constant-component cross-output correlation reads +0.85 and +0.98, which looks
large. But it appears alongside a linear-component correlation of -0.21, and we
suspect the two components are trading off against each other rather than
revealing real shared structure. Because K2 and K3 were never run held out, **we
cannot test this**. It is an open question, not a result.

## E.8 Held-out results: the continuous outcomes

Verified from `outputs/joint_subscore_gp/metrics.csv`. 3,361 points, 934 patients.

**Tremor:**

| Model | MAE | 95% interval | RMSE | Coverage | Negative mass |
|---|---:|---|---:|---:|---:|
| gp_joint_K0_full | **0.2380** | [0.2298, 0.2463] | 0.3099 | 0.940 | 0.063 |
| gp_joint_K1_full | 0.2391 | [0.2314, 0.2471] | **0.3075** | 0.936 | 0.054 |
| gp_joint_K1_diag | 0.2394 | [0.2316, 0.2473] | 0.3077 | 0.937 | 0.054 |
| gp_indep_K1 | 0.2395 | [0.2317, 0.2475] | 0.3079 | 0.936 | 0.054 |
| persistence | 0.2426 | [0.2329, 0.2520] | 0.3271 | 0.812 | 0.083 |

**PIGD:**

| Model | MAE | 95% interval | RMSE | Coverage | Negative mass |
|---|---:|---|---:|---:|---:|
| persistence | **0.2040** | [0.1921, 0.2162] | 0.3416 | 0.848 | 0.157 |
| gp_joint_K0_full | 0.2155 | [0.2048, 0.2262] | 0.3266 | 0.943 | 0.127 |
| gp_joint_K1_full | 0.2173 | [0.2072, 0.2276] | **0.3213** | 0.933 | 0.107 |
| gp_indep_K1 / K1_diag | 0.2175 | [0.2073, 0.2277] | 0.3214 | 0.933 | 0.107 |

## E.9 The four comparisons that answer the research questions

**Question 1: Does the joint model beat the diagonal one? (The cross-output test.)**

| Outcome | full - diagonal | 95% interval | Verdict |
|---|---|---|---|
| tremor | -0.00026 | [-0.00055, +0.00003] | Includes zero |
| PIGD | -0.00013 | [-0.00051, +0.00025] | Includes zero |

**No. The cross-output link provides no held-out benefit.** The learned
correlations were -0.010 (constant), -0.063 (RBF) and +0.091 (noise), all tiny.
The predictive correlation between the two outcomes had a median of 0.054, and
**zero percent of points had an absolute correlation above 0.1**.

This is not a failure to detect. In simulation, the same code recovered a planted
correlation of 0.55 correctly, and correctly returned near zero when the truth was
independence. The model can find a real correlation. There was not one to find.

**Question 2: Does flexible temporal structure help? (K1 versus K0.)**

| Outcome | K1_full - K0_full | 95% interval | Verdict |
|---|---|---|---|
| tremor | +0.00112 | [-0.00185, +0.00410] | Includes zero |
| PIGD | +0.00189 | [-0.00214, +0.00572] | Includes zero |

**No.** The flexible RBF kernel did not beat the simple linear-trend kernel. In
fact K0 had the lowest tremor MAE of all.

**Question 3: Does the GP beat persistence?**

| Outcome | Best GP - persistence | 95% interval | Verdict |
|---|---|---|---|
| tremor | -0.0035 | [-0.0087, +0.0018] | Includes zero: no difference |
| PIGD | **+0.0133** | [+0.0082, +0.0187] | Excludes zero: GP is worse |

**No.**

**Question 4: Does the GP beat the Step 1 LMM?**

On exactly the same 3,361 points, with the match verified (same target times,
same observed values):

| Outcome | GP - LMM | 95% interval | Verdict |
|---|---|---|---|
| tremor | **-0.0155** | [-0.0197, -0.0115] | GP better |
| PIGD | **-0.0217** | [-0.0268, -0.0166] | GP better |

**Yes.** This is the one clear positive result of Step 2.

*Note: this comparison was computed during the working session from the two saved
prediction files. It is reproducible from those files but is not itself stored as
a saved CSV.*

## E.10 PIGD = 0 visits: where the GP fails badly

Verified from `outputs/joint_subscore_gp/pigd_zero_analysis.csv`, 446 points:

| Model | PIGD MAE | Mean PIGD prediction | Fraction predicting below 0.1 |
|---|---:|---:|---:|
| patient mean | **0.0944** | 0.094 | 0.572 |
| persistence | 0.0969 | 0.097 | 0.621 |
| gp_joint_K1_full | 0.2417 | 0.242 | **0.002** |

The true value is 0. Persistence predicts near zero 62 percent of the time. **The
GP essentially never does — only 0.2 percent of the time.**

The reason is shrinkage. The GP pulls every prediction towards the population
average, which is about 0.41. At the boundary, that is exactly the wrong thing to
do.

## E.11 Negative predictive mass

**Negative predictive mass** is the probability the model assigns to values below
zero.

Tremor and PIGD scores are averages of items scored 0 to 4. They cannot be
negative. But a Gaussian distribution always has some probability below zero. So
negative predictive mass measures how wrong the Gaussian assumption is.

| Model | Tremor | PIGD |
|---|---:|---:|
| gp_joint_K1_full | 0.054 | 0.107 |
| persistence | 0.083 | 0.157 |

So the GP places about 10.7 percent of its PIGD probability on impossible values.
This is better than persistence, but still a real misspecification.

We did **not** clip negative predictions before computing the continuous metrics.
Clipping would have hidden the problem. Clipping was applied only inside the
subtype derivation, where it is unavoidable, and the count was reported.

## E.12 Subtype results

| Model | Rule | Accuracy | 95% interval | Balanced | Indeterminate recall |
|---|---|---:|---|---:|---:|
| gp_joint_K1_full | point estimate | 0.7301 | [0.7088, 0.7504] | **0.554** | **0.175** |
| gp_joint_K1_diag | **probabilistic** | **0.7587** | [0.7396, 0.7782] | 0.532 | **0.000** |
| persistence | point estimate | 0.7227 | [0.7010, 0.7440] | 0.548 | **0.193** |

Two rules were used, and they are different quantities.

The **point rule** converts the predicted tremor and PIGD numbers into a subtype
using the standard thresholds.

The **probabilistic rule** draws 4,000 samples from the joint predictive
distribution, classifies each one, and picks the most common answer. It is about
3 points more accurate, but it **never predicts Indeterminate**. The middle band
is too narrow to ever win a majority vote.

## E.13 Why a small numerical gain is not necessarily an important one

The joint GP beat the diagonal GP on tremor by 0.00026 MAE. On a quantity whose
typical error is 0.24, that is about one tenth of one percent. Even if the
interval had excluded zero, no clinician or researcher would act differently.

This is the difference between **statistical significance** (can we tell the
difference from noise?) and **practical importance** (does the difference
matter?). With 3,361 points and a patient-level bootstrap, we have enough data to
detect very small differences. Detecting them does not make them important.

## E.14 Why the joint model was not kept

Three reasons:

1. The cross-output link it adds is measurably absent in this data.
2. It does not beat the simpler diagonal version in any way that matters.
3. It does not beat persistence, which is the real benchmark.

The honest conclusion: **joint modelling of tremor and PIGD adds complexity
without adding information, in this dataset.**

---

# Part F — Direction 2: Does the progression rate change over time?

## F.1 The research question

From the midterm presentation, word for word:

> "Let progression rate vary with time. Every model here assumes the rate is
> constant over follow-up. We want to compare early, middle and late follow-up
> directly. This is a hypothesis to test — we have not established it."

## F.2 Why this was the natural next step

Every model up to this point assumed one fixed rate of change. The hierarchical
Bayesian model said the ratio falls 0.093 per year, always. Step 1 said PIGD
rises 0.066 per year, always.

But "always" is an assumption, not a finding. Maybe the disease moves quickly at
first and then settles. Maybe it is slow at first and accelerates. Nobody had
checked.

## F.3 Random slope versus time-varying population slope

This distinction is the most important idea in Direction 2, and it is easy to
confuse.

**A random slope** means different patients progress at different speeds.
Example: Ram worsens at 0.05 per year for his whole follow-up; Shyam worsens at
0.10 per year for his whole follow-up. Each person has one constant rate, but the
rates differ between people.

**A time-varying population slope** means the rate changes as follow-up goes on,
for everyone on average. Example: in their first two years, both Ram and Shyam
worsen slowly; after five years, both worsen faster.

These are completely different claims. Adding random slopes answers the first
question and says **nothing** about the second. This is why we did not simply add
random slopes and call it done.

We used random slopes only as a **control**, to check that the population-level
finding was not just patient variation in disguise.

## F.4 Models compared

| Model | Mean structure | Purpose |
|---|---|---|
| **A: constant** | straight line (this is Step 1's model exactly) | baseline |
| **B2: one knot** | slope allowed to change once, at 3 years | simplest change |
| **B3: two knots** | slope allowed to change at 2 and 5 years | early / middle / late |
| **C: A + random slopes** | constant population slope, patients differ | control |
| **D: B3 + random slopes** | time-varying population slope, patients differ | control |

A "knot" is simply the time point where the slope is allowed to change. We used
**linear splines**, which means the line bends at the knot but does not jump. The
trajectory stays connected.

## F.5 Why knots at 2 and 5 years

The cut points were chosen from the data distribution **before fitting**, not
tuned afterwards.

Verified from `outputs/time_varying/period_composition.csv`:

| Period | Visits | Patients | Median time | PIGD = 0 visits |
|---|---:|---:|---:|---:|
| 0 to 2 years | 1,924 | **934** | 0.91 | 428 |
| 2 to 5 years | 1,415 | **725** | 3.00 | 209 |
| 5 years and beyond | 956 | **273** | 7.92 | 55 |

Knots at 2 and 5 give all three periods at least 956 visits and at least 273
patients. The median patient follow-up is 3.08 years, which sits inside the
middle period.

We also tested three pre-specified alternatives: (1.5, 4), (2.5, 6), and a single
knot at 3 years. These four were fixed in advance. We did not search for the best
cut points, because searching and then reporting the best result would be
cheating.

**Knots were treated as fixed constants, never estimated from the data.** If we
had estimated a change point on the full data and then evaluated predictions,
that would leak information from the test set.

## F.6 The estimated rates

Verified from `outputs/time_varying/period_slopes.csv`:

| Outcome | 0 to 2 years | 2 to 5 years | 5+ years |
|---|---:|---:|---:|
| **Tremor** | +0.0327 | +0.0325 | **+0.0126** |
| **PIGD** | +0.0450 | +0.0574 | **+0.0800** |

With intervals:

| Outcome | Period | Slope | 95% interval |
|---|---|---:|---|
| tremor | 0-2y | 0.0327 | [0.0197, 0.0458] |
| tremor | 2-5y | 0.0325 | [0.0212, 0.0439] |
| tremor | 5y+ | 0.0126 | [0.0056, 0.0197] |
| PIGD | 0-2y | 0.0450 | [0.0309, 0.0592] |
| PIGD | 2-5y | 0.0574 | [0.0453, 0.0696] |
| PIGD | 5y+ | 0.0800 | [0.0724, 0.0875] |

**What these numbers mean in plain English.** In the first five years, tremor
scores rise by about 0.033 per year. After five years, they rise by only about
0.013 per year — the tremor side roughly stops getting worse. Meanwhile PIGD
starts at 0.045 per year and climbs to 0.080 per year — the gait and balance side
nearly doubles its speed.

So the two sides of the disease move apart as time goes on. This is why the ratio
falls, and it falls faster later.

## F.7 The actual hypothesis test

Comparing period slopes by eye is not a test. We computed the **differences**
between periods directly, with intervals. Verified from
`outputs/time_varying/slope_contrasts.csv`:

| Outcome | Contrast | Difference | 95% interval | Excludes zero? |
|---|---|---:|---|---|
| tremor | 2-5y minus 0-2y | -0.0002 | [-0.0215, +0.0211] | No |
| tremor | **5y+ minus 0-2y** | **-0.0201** | [-0.0343, -0.0059] | **Yes** |
| tremor | **5y+ minus 2-5y** | **-0.0199** | [-0.0361, -0.0037] | **Yes** |
| PIGD | 2-5y minus 0-2y | +0.0124 | [-0.0107, +0.0354] | No |
| PIGD | **5y+ minus 0-2y** | **+0.0349** | [+0.0196, +0.0502] | **Yes** |
| PIGD | **5y+ minus 2-5y** | **+0.0226** | [+0.0052, +0.0400] | **Yes** |

Note that **early and middle do not differ at all**. The change appears only in
the late period. So this is really a two-phase picture, not the three-phase
picture the midterm wording suggested.

## F.8 The composition control — the most important check

There is a serious trap here. The three periods contain **different patients**:
934, 725 and 273. Only patients with long follow-up appear in the late period, and
they are not a random sample.

So a difference between periods could mean "the rate changed" **or** "the patients
who stayed are different". These must be separated.

The control: restrict to patients who appear in **both** periods, compute the rate
change inside each patient, and average. This holds composition fixed by
construction.

Verified from `outputs/time_varying/within_patient_rate_change.csv`:

| Outcome | From | To | Patients | Change in rate | 95% interval | Excludes zero? |
|---|---|---|---:|---:|---|---|
| tremor | 0-2y | 5y+ | 226 | **-0.0396** | [-0.0731, -0.0079] | **Yes** |
| tremor | 2-5y | 5y+ | 242 | **-0.0353** | [-0.0666, -0.0037] | **Yes** |
| PIGD | 0-2y | 5y+ | 226 | **+0.0727** | [+0.0425, +0.1015] | **Yes** |
| PIGD | 2-5y | 5y+ | 242 | **+0.0790** | [+0.0500, +0.1108] | **Yes** |
| tremor | 0-2y | 2-5y | 533 | -0.0059 | [-0.0309, +0.0202] | No |
| PIGD | 0-2y | 2-5y | 533 | +0.0251 | [+0.0036, +0.0467] | Yes |

**The effect is larger within patients than between them.** So it is not a
selection artefact. This is the check that makes the result believable.

## F.9 Robustness across cut points

All four pre-specified knot sets agreed, in sign and roughly in size:

| Knot set | Tremor (late minus early) | PIGD (late minus early) |
|---|---|---|
| primary (2, 5) | -0.0201 [-0.0343, -0.0059] | +0.0349 [+0.0196, +0.0502] |
| alt (1.5, 4) | -0.0224 [-0.0414, -0.0035] | +0.0301 [+0.0095, +0.0506] |
| alt (2.5, 6) | -0.0225 [-0.0356, -0.0094] | +0.0362 [+0.0221, +0.0503] |
| one knot at 3 | -0.0190 [-0.0304, -0.0076] | +0.0304 [+0.0181, +0.0427] |

Every interval excludes zero.

## F.10 The random-slope control

When patients were allowed their own slopes, the population-level effect did not
disappear. It got **larger**: tremor -0.0346 and PIGD +0.0821 for the late-versus-
early contrast.

So patient-to-patient variation does not explain the finding.

## F.11 Method controls

Before trusting the result, we checked the method itself on simulated data:

- When we **planted** a known slope change, the method recovered it to within
  0.015.
- When we generated data with a genuinely **constant** slope and fitted it with
  knots anyway, the method reported **0 false positives out of 6 contrasts**.

So the method finds real changes and does not invent fake ones.

## F.12 The crucial finding: description improved, prediction did not

This is the heart of Direction 2.

Verified from `outputs/time_varying/paired.csv`:

| Outcome | spline minus constant | 95% interval | Excludes zero? |
|---|---|---|---|
| tremor | -0.00008 | [-0.00074, +0.00060] | No |
| PIGD | **-0.00337** | [-0.00441, -0.00232] | **Yes** |

The PIGD improvement is statistically real but **scientifically tiny**: 0.0034 on
a quantity whose typical error is 0.236. That is about **1.4 percent**.

And both models still lose badly to persistence:

| Outcome | Model | MAE | versus persistence |
|---|---|---:|---|
| tremor | persistence | **0.2426** | — |
| tremor | spline | 0.2545 | +0.0119 [+0.0038, +0.0204], worse |
| PIGD | persistence | **0.2040** | — |
| PIGD | spline | 0.2357 | +0.0317 [+0.0232, +0.0402], worse |

Subtype results:

| Model | Accuracy | Balanced accuracy | Indeterminate recall |
|---|---:|---:|---:|
| persistence | **0.7227** | **0.548** | **0.193** |
| lmm_spline_2_5 | 0.7093 | 0.521 | 0.175 |
| lmm_constant | 0.7087 | 0.527 | 0.175 |

The time-varying model gains **+0.0006 accuracy** over the constant one, has
**identical Indeterminate recall**, and actually has **lower balanced accuracy**.

## F.13 How can the rate change while prediction barely improves?

This sounds contradictory, but it is not. Here is the explanation.

The time-varying model changes the **population average trend**. But for
predicting one specific patient's next visit, the population trend is only a small
part of the answer. Far more important are:

1. Where that patient currently is (their own level), and
2. How much random noise there is in a single measurement.

The population trend contributes roughly 0.03 to 0.08 per year. The within-patient
noise is much larger. So improving the trend by a small amount barely moves the
needle on a single prediction.

An analogy. Knowing that prices rise 6 percent a year on average is genuinely
useful for understanding the economy. It is almost useless for guessing the price
of one specific item next week, because that item's own price history and its
day-to-day fluctuation dominate.

## F.14 Limitations of Direction 2

1. **Late period sparsity.** Only 273 patients (29 percent) reach five years. The
   late-period estimates have narrow intervals because they contain many visits,
   but those visits come from few people.
2. **Attrition bias is not modelled.** Patients who stop attending may differ
   systematically. We checked and found that patients followed five years or more
   had **lower** baseline severity (tremor 0.488 versus 0.594, p = 0.00014; PIGD
   0.190 versus 0.276, p < 0.00001). If dropping out is related to progression,
   the late-period slopes could be biased, and we cannot determine the direction
   of that bias from this design.
3. **The knots were chosen, not estimated.** We did **not** test whether five
   years is specifically the right change point. We only showed that a change
   exists somewhere in the later part of follow-up, robustly across four
   reasonable choices. **We cannot claim that five years is a biological
   transition point.**
4. **The boundary problem is unchanged.** On the 446 PIGD-zero points, the spline
   predicts 0.274 on average where the truth is 0. Persistence predicts 0.097. A
   time-varying mean cannot fix a boundary problem.

---

# Part G — Direction 3: Can we cluster patients at all?

## G.1 Why clustering was considered

The midterm plan said:

> "Cluster longitudinal patterns. If per-patient trajectory info can be extracted
> reliably, ask whether patients fall into distinct progression
> patterns/phenotypes; not simply 'run clustering.'"

The idea: maybe patients fall into natural groups. Perhaps some progress fast and
some slow, and those groups mean something clinically.

## G.2 Why we did not immediately run clustering

Read the midterm wording again. It begins with a condition: **"If per-patient
trajectory info can be extracted reliably."**

Clustering will always produce clusters. Give any algorithm a number of groups
and it will return that many groups, whether or not the groups exist. So the real
question is not "can we cluster?" but "is there anything reliable to cluster?"

Also, a critical logical point: **patients differing from one another does not
mean there are discrete groups.** Human height varies continuously. There are no
natural "height clusters". Continuous variation and discrete subgroups are
different things.

So we ran an audit first.

## G.3 The data audit

Verified from `outputs/clustering_audit/data_audit.csv`:

| Quantity | Value |
|---|---:|
| Patients | 934 |
| Visits | 4,295 |
| Median visits per patient | 4 |
| Patients with **exactly 2** visits | **172 (18.4%)** |
| Patients with >= 3 visits | 762 (81.6%) |
| Patients with >= 4 visits | 502 (53.7%) |
| Patients with >= 5 visits | 333 (35.7%) |
| Patients with >= 8 visits | **145 (15.5%)** |
| Patients with >= 10 visits | 73 (7.8%) |
| Median follow-up span | 3.08 years |
| Patients reaching >= 3 years | 545 |
| Patients reaching >= 5 years | 273 |
| Missing tremor / PIGD values | **0 / 0** |
| Visits with PIGD = 0 | 692 |
| Patients with >= 1 undefined-ratio visit | **349 (37.4%)** |
| Patients with PIGD = 0 at every visit | 38 |

The headline problem: **almost half the cohort has three visits or fewer.**

## G.4 Short versus long follow-up

Verified from `outputs/clustering_audit/followup_bias.csv`:

| Variable | Followed >= 5 years | Followed < 5 years | p-value |
|---|---:|---:|---:|
| Baseline tremor | 0.4879 (n=273) | 0.5941 (n=661) | 0.00014 |
| Baseline PIGD | 0.1897 (n=273) | 0.2762 (n=661) | < 0.00001 |

Patients with long follow-up are **less severe at baseline**. So follow-up length
is not random, and any clustering based on trajectories would partly be clustering
on "who stayed in the study".

## G.5 What makes a patient's slope reliable or unreliable

To estimate a slope you need enough points spread over enough time, and the real
signal must be bigger than the measurement noise.

With **two visits**, you can draw exactly one line through two points. There is no
room left to estimate the error. The uncertainty is not merely large — it is
mathematically undefined. 172 patients are in this position.

With **three or four visits**, you can estimate a slope and some uncertainty, but
both are poor.

## G.6 Leave-one-visit-out: a simple and brutal test

Here is the test, explained simply.

> Take a patient. Estimate their progression slope using all their visits. Now
> remove just one visit. Estimate the slope again. Did the answer change?

If removing a single routine visit changes the conclusion, then the "progression
pattern" is not a property of the patient. It is a property of which visits
happened to be recorded.

Verified from `outputs/clustering_audit/leave_one_visit_out.csv`, among patients
with at least three visits:

| Outcome | Patients whose slope **sign flips** when one visit is removed |
|---|---:|
| Tremor | **48.4%** |
| PIGD | **41.6%** |

Sign flip means the patient changes from "getting worse" to "getting better", or
the reverse. For patients with exactly three visits, this happens 62 to 66 percent
of the time.

## G.7 The slope spread depends mostly on visit count

Verified from `outputs/clustering_audit/per_patient_ols_slopes.csv`:

| Outcome | n=2 | n=3 | n=4 | n=5 | n>=8 |
|---|---:|---:|---:|---:|---:|
| Tremor slope spread (SD) | 0.271 | 0.159 | 0.112 | 0.096 | **0.059** |
| PIGD slope spread (SD) | 0.237 | 0.138 | 0.120 | 0.127 | **0.064** |

The apparent variation between patients shrinks by a factor of about **4.6** as
we get more visits per patient. Real biological variation does not change when you
measure more often. **Noise does.** So most of the apparent spread at low visit
counts is measurement error.

If we clustered on raw slopes, we would largely be clustering on how often each
patient attended.

## G.8 Shrinkage does not create information

A natural fix is to use **shrinkage**. Instead of each patient's raw slope, use a
"shrunken" estimate that pulls uncertain patients towards the population average.
Patients with many visits keep their own estimate; patients with few visits get
pulled towards the middle.

Shrinkage produces more stable numbers. But it does not add information. It simply
replaces an unreliable estimate with the population average. If we then cluster
those numbers, the patients with little data all sit near the middle, which can
look like a group but is just missing information.

We measured this properly using **reliability**:

```
reliability = 1 - (posterior variance / prior variance)
```

In words: what fraction of the apparent variation between patients is real signal
rather than noise? A reliability of 1.0 means perfect. A reliability of 0.0 means
the estimate carries no patient-specific information at all.

Verified from `outputs/clustering_audit/random_slope_reliability.csv`:

| Outcome | Median | n=2 | n=3 | n=4 | n=5 | n>=8 |
|---|---:|---:|---:|---:|---:|---:|
| Tremor slope | **0.270** | 0.085 | 0.165 | 0.272 | 0.388 | **0.840** |
| PIGD slope | **0.531** | 0.173 | 0.354 | 0.535 | 0.671 | **0.944** |

**For the median patient, 73 percent of the apparent variation in the tremor slope
is noise.** The usual standard for using a measurement to classify individuals is
about 0.7 or higher. Only patients with eight or more visits reach that — and
there are just 145 of them.

## G.9 The single-population null test

This was the decisive experiment, and it is worth understanding fully.

**What we did.** We took our fitted model and simulated a brand new dataset from
it. In this simulated dataset there is, by construction, **exactly one population
and no groups at all**. We kept every patient's real visit times, so the
simulated data has the same irregular, unequal structure as the real data.

Then we ran the standard clustering diagnostics on both.

Verified from `outputs/clustering_audit/real_vs_null_model_selection.csv`:

| K (number of groups) | Real BIC | **Null BIC** | Real silhouette | **Null silhouette** |
|---|---:|---:|---:|---:|
| 1 | 5251.5 | 5219.0 | — | — |
| 2 | 4808.6 | **5131.8** | 0.487 | **0.298** |
| 3 | **4738.4** | 5175.3 | 0.369 | 0.299 |
| 4 | 4749.6 | 5215.3 | 0.329 | 0.227 |

BIC is a model-selection score where lower is better. Silhouette measures how
well separated the groups look, from -1 to 1.

**Look at the null columns.** In data with no groups whatsoever, BIC prefers two
groups over one by 87 units, and the silhouette is 0.298.

So if we had simply run clustering, seen "BIC prefers K=2, silhouette around
0.3", and declared two phenotypes, **we would have reached that conclusion on data
that contains no groups at all.**

The real data does show more structure than the null (BIC improvement of 513
versus 88). So this is not purely an artefact. But the standard diagnostics alone
cannot tell the two situations apart, which means they cannot support a phenotype
claim.

## G.10 What the apparent clusters actually separated

We checked what candidate clusters differ in. We deliberately did **not** use
visit count, follow-up length or baseline severity as clustering inputs — we used
them afterwards to audit the result.

Verified from `outputs/clustering_audit/cluster_nuisance_audit.csv` (Kruskal-Wallis
p-values):

| K | Visit count | Follow-up span | Baseline tremor | **Baseline PIGD** |
|---|---:|---:|---:|---:|
| 2 | 0.123 | 0.030 | 0.977 | **< 1e-16** |
| 3 | **< 1e-16** | **< 1e-18** | 0.126 | **< 1e-17** |

**At K=2, the split is essentially on baseline PIGD severity** (group means 0.44
versus 0.22), with no tremor difference at all.

**At K=3, the clustering starts separating by visit count and follow-up length.**
Group 3 had 5.97 visits and 6.30 years of follow-up, against 4.19 and 4.04 for
group 1.

So the "clusters" are severity and attendance, not progression shape. This is
exactly the trap the brief warned about.

## G.11 Bootstrap stability versus drop-one-visit stability

Verified from `outputs/clustering_audit/cluster_stability.csv`:

| K | Bootstrap ARI (mean) | **Drop-last-visit ARI** |
|---|---:|---:|
| 2 | **0.936** | **0.456** |
| 3 | 0.821 | **0.335** |

ARI (Adjusted Rand Index) measures how much two groupings agree. 1.0 is perfect
agreement; 0.0 is what you would get by chance.

These two numbers measure very different things, and the contrast is the finding.

**Bootstrap** resamples which *patients* are included, keeping each patient's data
intact. It asks: "if we had a different sample of patients, would we get the same
groups?" Answer: yes, 0.936.

**Drop-one-visit** removes a single visit from each patient and re-estimates. It
asks: "if each patient had attended one fewer time, would they land in the same
group?" Answer: often not — 0.456.

So the clusters are stable to **sampling** but unstable to **measurement**. That
is the signature of structure driven by noise in the features rather than by real
patient properties. A patient's "phenotype" should not change because they missed
one appointment.

## G.12 Why no clustering model was implemented

We did not implement a clustering module. The audit answered the midterm's
conditional, and the answer was no.

To be precise about what we are claiming: **this is not a report of a clustering
algorithm that failed.** We never ran one on the full cohort. What we established
is that the inputs any clustering algorithm would need — reliable per-patient
trajectory parameters — do not exist at this visit density.

Specifically:

1. 46 percent of patients have three visits or fewer; 18.4 percent have two.
2. One removed visit flips the slope sign for roughly 45 percent of patients.
3. Median slope reliability is 0.270 (tremor) and 0.531 (PIGD), below the usual
   0.7 threshold.
4. Apparent slope spread varies 4.6-fold with visit count.
5. A no-groups null produces the same diagnostic signals.
6. Candidate clusters separate on severity and attendance.
7. Cluster membership is unstable to removing one visit.

## G.13 The limited exploratory option

There is one defensible restricted analysis: the **145 patients with eight or more
visits**, where slope reliability reaches 0.840 (tremor) and 0.944 (PIGD).

We did **not** run it, for two reasons:

1. **Selection bias.** These are precisely the patients with lower baseline
   severity and longer retention. They are not representative.
2. **Sample size.** 145 patients out of 934 cannot establish phenotypes for the
   cohort.

If this is run later, it must be labelled hypothesis-generating, with the null
comparison and the drop-one-visit check built in from the start.

## G.14 Conclusion of Direction 3

**The full-cohort evidence does not justify claiming distinct progression
phenotypes.** The evidence points towards **continuous heterogeneity** — patients
vary from one another smoothly — rather than discrete groups.

This is a real scientific result, and it is also useful: it explains *why* several
earlier experiments found nothing.

---

# Part H — Putting it all together

## H.1 The chronological table

| # | Research question | Why we asked it | Method | What we found | What we learned | Why we continued or stopped |
|---|---|---|---|---|---|---|
| 1 | Can a flexible curve predict the ratio better than a straight line? | The hierarchical Bayesian model assumed straight lines; maybe too rigid | GP with kernels K0-K3, LOPO on 821 patients / 2,707 points | Flexible kernel beat linear by 0.0333 MAE [-0.0468, -0.0205]; GP beat persistence by 0.0707 on the ratio. But subtype accuracy 0.724 versus persistence 0.733, and Indeterminate recall 0.000 | Better number prediction does **not** become better category prediction | Continued, but shifted target: the ratio itself was the problem |
| 2 | Does the ratio throw away important information? | 2/4 and 10/20 both give 0.5; ratio undefined at 692 visits | Descriptive analysis of the cohort | 692 visits (16.1%) and 113 patients invisible to ratio models; the lost visits are overwhelmingly TD | The ratio causes a non-random data loss, not just a modelling inconvenience | Moved to modelling tremor and PIGD directly |
| 3 | Can we model both subscores directly in a simple, interpretable way? | Needed a clean baseline before anything flexible | Joint LMM, random intercepts only, 934 patients / 3,361 points | Tremor +0.0233/yr, PIGD +0.0655/yr. Lost to persistence on MAE (+0.0120, +0.0350). Recovered all 692 visits, 82.1% accuracy on them. Cross-outcome correlation -0.015 / +0.019 | PIGD worsens 2.8x faster than tremor. The ratio's fall is **not** tremor improving | Continued to a flexible version, but warned that cross-output modelling would likely find nothing |
| 4 | Does predicting tremor and PIGD **together** help? | Joint models can borrow strength between related outcomes | Multi-output GP (LMC), full vs diagonal vs independent, 934 folds, 3,736 fits | Full minus diagonal: -0.00026 [-0.00055, +0.00003] tremor, -0.00013 [-0.00051, +0.00025] PIGD. Both include zero. GP beat the LMM (-0.0155, -0.0217) but lost to persistence on PIGD (+0.0133) | The cross-output link is measurably absent. Flexible kernels did not beat simple ones either | Stopped pursuing joint modelling. The method was verified able to detect planted correlations, so this is a real null |
| 5 | Does the progression rate change over follow-up? | Every model assumed one constant rate; nobody had checked | Linear-spline LMM, knots (2,5) plus 3 alternatives, with composition and random-slope controls | Tremor slows (0.033 to 0.013), PIGD accelerates (0.045 to 0.080) after 5 years. Survives composition control (effect is **larger** within patients), all 4 knot sets, and random slopes. But held-out gain only -0.0034 MAE on PIGD and -0.00008 on tremor | The constant-rate assumption **is wrong**. But fixing it barely helps prediction | Direction established as a descriptive finding; not adopted as a predictive improvement |
| 6 | Do patients fall into distinct progression groups? | Midterm asked, conditional on reliable trajectories | Clusterability audit: reliability, leave-one-visit-out, single-population null, nuisance audit, stability | Median slope reliability 0.270 / 0.531; 45% sign flip on one removed visit; a **no-groups null also "prefers" K=2**; candidate clusters separate on baseline PIGD and visit count; drop-one-visit ARI 0.456 | The inputs clustering needs do not exist at this visit density. Evidence favours continuous heterogeneity | **No clustering implemented.** Recommended against for the full cohort |

## H.2 How the questions connect

These were not six unrelated models. Each one came from the previous one's
failure.

**GPR asked:** is the model too rigid? Answer: making it flexible helps the
number, not the category.

**That pointed at the target, not the model.** So the next question was: is the
ratio the right thing to predict? Answer: no, it hides severity and deletes 692
visits.

**Step 1 asked:** can we model the two parts directly? Answer: yes, and it reveals
that PIGD worsens 2.8 times faster. But it still loses to persistence. It also
measured a near-zero cross-outcome correlation.

**Step 2 asked:** does joint modelling help? We already suspected not, because of
Step 1's correlation. Answer: it does not, confirmed held out with a properly
nested test.

**At this point three different kinds of model complexity had failed to help.**
So the next question moved away from model structure: maybe the **assumption** of a
constant rate is wrong.

**Direction 2 asked:** does the rate change? Answer: yes, robustly. But prediction
barely moved.

**That was the fourth time** a real finding about the process did not become a
predictive gain. So Direction 3 asked whether the per-patient information needed
for the next idea (clustering) even exists.

**Direction 3 answered:** largely no. And that answer **explains the earlier
nulls**. Per-patient slopes are mostly noise at four visits. That is why random
slopes added nothing in Step 1, why the multi-output GP found no structure to
borrow, and why clustering cannot work. One explanation now covers three separate
negative results.

## H.3 The single thread running through everything

Every experiment after GPR points at the same conclusion from a different angle:

> **The limitation is the measurement, not the model.**

The supporting evidence:

- The within-patient spread of the ratio is 1.22, against an Indeterminate band
  0.25 wide — about five times wider.
- About 25 percent of observed subtype changes reverse at the next visit.
- PIGD is exactly zero at 16.1 percent of visits, where a Gaussian model cannot
  work.
- The per-patient slope is unreliable at the median visit count.

Model structure cannot fix any of these.

---

# Part I — What we know and what remains uncertain

## I.1 Established by our experiments

These are directly supported by held-out results or by controlled analyses in this
project.

1. **A flexible kernel predicts the continuous ratio better than a linear one.**
   K1 minus K0 = -0.0333 [-0.0468, -0.0205] pooled, replicated in all four cohorts
   under both aggregations.
2. **The GP predicts the ratio better than persistence**, by -0.0707 [-0.0979,
   -0.0451].
3. **Better continuous prediction does not produce better subtype prediction.**
   Persistence had the worst ratio MAE and the best derived subtype accuracy
   (0.7329 against 0.7237 for K1).
4. **Both subscores worsen, and PIGD worsens about 2.8 times faster than tremor**
   (+0.0655 versus +0.0233 per year, both intervals excluding zero).
5. **The ratio's decline is not tremor improving.** It is the denominator growing
   faster.
6. **Modelling the subscores directly recovers 692 visits and 113 patients**, and
   predicts subtype on them at 82.1 percent accuracy.
7. **Tremor and PIGD are essentially uncorrelated once time is accounted for**
   (-0.015 between patients, +0.019 within), confirmed independently by the
   multi-output GP (predictive correlation median 0.054, no point above 0.1).
8. **Joint modelling of the two subscores gives no held-out benefit** over a
   properly nested diagonal control.
9. **Flexible temporal kernels do not beat simple ones** on the subscores (K1
   minus K0 includes zero for both outcomes).
10. **The multi-output GP beats the Step 1 LMM** on matched points (-0.0155 and
    -0.0217, both excluding zero).
11. **Neither the LMM nor the GP beats persistence** on PIGD.
12. **The population progression rate changes over follow-up.** Tremor slows and
    PIGD accelerates after about five years, surviving the composition control,
    four knot choices, and the random-slope control.
13. **Allowing a time-varying rate barely improves prediction**: -0.0034 MAE on
    PIGD (about 1.4 percent) and no detectable change on tremor.
14. **Per-patient trajectory slopes are unreliable at this visit density.** Median
    reliability 0.270 (tremor) and 0.531 (PIGD); one removed visit flips the sign
    for about 45 percent of patients.
15. **Standard clustering diagnostics cannot establish groups here**, because a
    no-groups null produces similar signals (BIC prefers K=2, silhouette 0.298).
16. **Gaussian models fail at the PIGD = 0 boundary.** They predict about 0.24 to
    0.27 where the truth is 0, and essentially never predict near zero.

## I.2 Supported but requiring caution

1. **The change in progression rate happens "after about five years."** The exact
   location is not established. We tested four fixed knot sets and did not
   estimate a change point.
2. **The late-period estimates rest on 273 patients** (29 percent). Intervals look
   narrow because those patients contribute many visits, but the number of
   *people* is small.
3. **Attrition may bias the late-period result.** Patients followed five years or
   more had significantly lower baseline severity. We could not determine the
   direction of any resulting bias.
4. **The K2 and K3 constant-component correlations of +0.85 and +0.98** come from
   population fits only. They were never evaluated held out. We suspect component
   trade-off rather than real structure, but this is untested.
5. **The derived subtype comparison between the GP and the HMM** was verified on
   matched points during the session, but is reproducible from saved prediction
   files rather than stored as its own CSV.
6. **The real data does show more cluster structure than the null** (BIC
   improvement 513 versus 88). Something non-Gaussian is present. We have not
   characterised what it is.

## I.3 Not established

1. **We have not established any clinical phenotype.** No group found in this work
   can be called a phenotype.
2. **We have not established that five years is a biological transition point.**
3. **We have not established clinical validity or utility of any model here.** The
   prediction target is our own rule applied to the next visit, not an independent
   clinical diagnosis.
4. **We have not established that the GP and the HMM are equivalent.** The matched
   comparison was a failure to detect a difference, which is not the same as
   demonstrating equivalence.
5. **We have not established that discrete progression groups do not exist.** We
   established that this data cannot detect them. Absence of evidence is not
   evidence of absence.
6. **We have not established any causal or biological mechanism** for the
   tremor/PIGD divergence.
7. **No external dataset has been used.** Everything is PPMI.

## I.4 The next research question: Direction 4

The planned next direction is:

> **Why and when does the HMM fail?**

Why this is the natural next step: every direction so far has tried to improve
prediction by changing the model, and every one has run into the same wall. The
errors are not spread evenly. They appear to be concentrated in specific,
identifiable places:

- The Indeterminate band, which almost no model ever predicts.
- The PIGD = 0 boundary, where Gaussian models break down.
- The five-latent-state HMM, which converged in only 404 of 934 folds (43
  percent) and did not improve on the three-state version.

Direction 4 would characterise *where* and *when* the models fail, rather than
trying another model family. This is the one remaining direction that addresses
the measurement problem directly rather than working around it.

**This analysis has not been started. No Direction 4 results exist.**

## I.5 The longer-term goal: a clinician-facing framework

The longer-term aim discussed for this project is a framework that could help a
clinician interpret model output. It would aim to:

- Distinguish a **sustained** change in subtype from a **temporary** switch.
  (Earlier work found that requiring two consecutive assessments removes 51.7
  percent of apparent PIGD conversions.)
- Communicate **uncertainty** honestly rather than giving a single label. Our
  models are well calibrated, which is the right foundation for this.
- Indicate **when the model is likely to be unreliable**, which is what Direction
  4 would inform.

**This framework is a future goal. Nothing of it has been implemented, designed in
detail, or validated. No part of this project should be described as a
clinician-facing tool.**

---

# Part J — Questions I should be able to answer

### 1. Why did we use GPR?

Our earlier hierarchical Bayesian model assumed each patient's ratio follows a
straight line. That is a strong assumption. A Gaussian process can fit smooth
curves of many shapes without deciding the shape in advance. So GPR let us test
whether the straight-line assumption was holding us back. The answer was that it
was, but only for predicting the number, not the category.

### 2. What is a kernel?

A kernel is a function that says how related two time points are. If the kernel
value between year 1 and year 1.2 is high, the model expects those measurements
to be similar. If the value between year 1 and year 8 is near zero, the model
treats them as unrelated. The kernel is what controls which shapes of trajectory
the model can learn.

### 3. Why is persistence such a strong baseline?

Persistence just repeats the patient's last measurement. It is strong for three
reasons. First, Parkinson's progresses slowly, so last year's value really is a
good guess. Second, it uses the most recent information and does not pull the
prediction towards the population average, which helps for unusual patients.
Third, it has no parameters, so there is nothing for it to overfit. In our work it
beat every fitted model on PIGD MAE and on subtype accuracy.

### 4. Why did we move away from the ratio?

Two reasons. First, the ratio hides severity: a patient with tremor 2 and PIGD 4
gets the same ratio 0.5 as a patient with tremor 10 and PIGD 20, even though the
second is five times more severe. Second, when PIGD is exactly zero the ratio does
not exist, which removed 692 visits and 113 patients from our analysis — and those
visits were overwhelmingly tremor-dominant, so the loss was not random.

### 5. What is the difference between an LMM and a hierarchical Bayesian model?

They model very similar things. The difference is how they estimate the
parameters. The hierarchical Bayesian model puts prior distributions on everything
and uses sampling (we used a Gibbs sampler) to get a full posterior distribution.
The linear mixed model estimates the parameters by maximising a likelihood (we
used REML) and reports standard errors. In our project they gave consistent
answers — the GP's K0 kernel is mathematically the same as our growth model, and
the two were statistically indistinguishable on held-out prediction, which was our
check that the code was correct.

### 6. Why did the joint GP not help?

Because there was no relationship between tremor and PIGD for it to use. Once the
model accounted for both of them rising with time, the leftover correlation was
about -0.01 to +0.09. The joint model's whole advantage is supposed to come from
that correlation. With no correlation, it has nothing to borrow.

### 7. What does a near-zero cross-output correlation mean?

It means that knowing whether a patient's tremor is above or below expectation at
a visit tells you almost nothing about whether their PIGD is above or below
expectation. The raw correlation looks like +0.148, but that is mostly because both
scores rise with time. Remove the shared time trend and only about +0.02 remains.

### 8. Why is PIGD = 0 a difficult case?

PIGD is an average of five items each scored 0 to 4, so it cannot go below zero.
In 16.1 percent of visits it is exactly zero. A Gaussian model assumes the value
can be anything, including negative, and it pulls predictions towards the
population average of about 0.41. So at the boundary it predicts about 0.24 when
the truth is 0. Persistence just repeats the last zero and does much better there.

### 9. What does negative predictive mass mean?

It is the probability a model assigns to values below zero. Since the scores
cannot be negative, this probability measures how wrong the Gaussian assumption
is. Our best GP put about 10.7 percent of its PIGD probability below zero. We
report this number instead of hiding it by clipping, because clipping would make
the model look better than it is.

### 10. What is the difference between statistical significance and practical importance?

Statistical significance asks: can we tell this difference from random noise?
Practical importance asks: is the difference big enough to matter? With 3,361
prediction points we can detect very small differences. For example, the
time-varying model improved PIGD MAE by 0.0034 and the interval excluded zero — so
it was statistically detectable. But 0.0034 on a typical error of 0.236 is about
1.4 percent, which nobody would act on. Detecting a difference does not make it
important.

### 11. How can the progression rate change while prediction barely improves?

The time-varying model improves the **population average trend**. But predicting
one patient's next visit depends mostly on where that patient currently is and on
how noisy a single measurement is. The population trend contributes only about
0.03 to 0.08 per year, which is small compared to the measurement noise. So fixing
the trend barely changes an individual prediction. It is like knowing average
inflation: useful for understanding the economy, nearly useless for guessing one
item's price next week.

### 12. Why did we test different knot locations?

A knot is where we allow the slope to change. If we had tried many locations and
reported the one that gave the best result, that would be cherry-picking. So we
fixed four reasonable choices in advance — (2,5), (1.5,4), (2.5,6) and a single
knot at 3 — and reported all of them. All four agreed, which is what makes the
finding trustworthy.

### 13. What is a random slope, and why is it not the same as a time-varying rate?

A random slope means different patients progress at different speeds, but each
patient has one constant speed. A time-varying population rate means the speed
changes as follow-up goes on, for everybody on average. Example: with random
slopes, Ram worsens at 0.05/year throughout and Shyam at 0.10/year throughout.
With a time-varying rate, both of them worsen slowly in the first two years and
faster after five years. We used random slopes only as a control, to make sure the
population finding was not just patient variation in disguise. It was not — the
effect actually got larger.

### 14. Why is the late-period result uncertain?

Three reasons. Only 273 patients (29 percent) reach five years, so the estimate
comes from few people even though it uses many visits. Those patients had
significantly lower baseline severity, so they are not representative. And we did
not test whether five years specifically is the change point — we only showed a
change exists somewhere later, robustly across four fixed choices.

### 15. Why did we reject full-cohort clustering?

Because the input that clustering needs — a reliable per-patient progression slope
— does not exist for most patients. Almost half the cohort has three visits or
fewer. Removing one visit flips the slope sign for about 45 percent of patients.
Median reliability is 0.270 for tremor, far below the usual 0.7 standard. And when
we checked, the candidate clusters separated on baseline PIGD severity and on
visit count, not on progression shape.

### 16. What does the single-population null test demonstrate?

We simulated a dataset from our fitted model that contains, by construction,
exactly one population and no groups, keeping every patient's real visit times.
Then we ran the standard clustering diagnostics on it. BIC still preferred two
groups over one, and the silhouette was 0.298. So those diagnostics would have
told us "there are two groups" in data where there are none. That means the
diagnostics alone cannot support a claim of distinct phenotypes.

### 17. Why is high bootstrap stability not sufficient?

Bootstrap resampling changes **which patients** are in the sample but keeps each
patient's data intact. It asks whether we would find the same groups in a
different sample of people. Our clusters passed that test (ARI 0.936). But when we
removed **one visit per patient** and re-estimated, agreement dropped to 0.456. So
the groups are stable to sampling and unstable to measurement. A real phenotype
should not change because a patient missed one appointment.

### 18. Why can't we call the apparent clusters phenotypes?

Four reasons. The features they are built from are mostly noise at the median
visit count. A no-groups null produces the same diagnostic signals. They separate
on baseline severity and visit count rather than on progression shape. And they
are unstable when one visit is removed. Any one of these would be enough to
withhold the word "phenotype".

### 19. What have we established about measurement noise?

That it is the main limitation of this project, established from several
independent angles. The within-patient spread of the ratio is 1.22 against an
Indeterminate band only 0.25 wide. About 25 percent of recorded subtype changes
reverse at the very next visit. Requiring two consecutive assessments removes 51.7
percent of apparent PIGD conversions. And individual progression slopes are
unreliable at the median of four visits. Together these explain why four different
modelling directions all produced real descriptive findings but almost no
predictive gain.

### 20. What is the purpose of Direction 4?

To find out **where** and **when** the models fail, instead of trying yet another
model family. The errors are not spread evenly — they concentrate in the
Indeterminate band, at the PIGD = 0 boundary, and in the five-state HMM that
converged in only 43 percent of folds. Understanding the failure pattern is more
likely to change what the project does next than another model would be. This has
not been started.

### 21. What is LOPO and why did we use it instead of a random split?

LOPO means leave-one-patient-out. We remove one patient, fit the model on everyone
else, predict that patient, then put them back and repeat. We used it because a
random split of *visits* would put the same patient in both training and test
data. The model would effectively have seen that patient already, and the results
would look better than they really are. Splitting by patient prevents this.

### 22. What does "nested comparison" mean, and why did we design Step 2 that way?

Nested means one model is a special case of the other. Our "diagonal" GP is
exactly our "full" GP with the cross-output terms forced to zero — same kernel,
same time scale, everything else identical. So any difference between them must
come from the cross-output terms and nothing else. If we had compared the joint GP
against fully independent GPs, two things would have changed at once (the
cross-output link and the time scale), and we could not have attributed a
difference to either.

### 23. Why did we report MAE two different ways in the GPR work?

Because there are two fair ways to average errors. "Pooled" treats every
prediction equally, so a patient with 10 visits counts five times more than a
patient with 2. "Per patient" gives every patient equal weight. The 134 patients
with six or more visits carry 40.4 percent of the pooled weight but only 16.3
percent of the patient weight. For one comparison (K0 versus hbayes) the sign of
the difference actually flipped between the two, so we report both rather than
choosing.

### 24. What is shrinkage, and why can it be misleading?

Shrinkage pulls uncertain individual estimates towards the population average.
Patients with many visits keep their own estimate; patients with few get pulled
to the middle. It produces more stable numbers, which is good. But it does not
create information. If we cluster shrunken estimates, all the low-information
patients sit near the middle, which can look like a group but is really just
missing data. That is why we measured reliability directly instead of trusting the
shrunken values.

### 25. Why did the GP beat the Step 1 LMM but lose to persistence?

These are different comparisons. Against the LMM, the GP is better because it can
represent smooth deviations from the population trend that a straight line cannot
— the gain was -0.0155 on tremor and -0.0217 on PIGD, both intervals excluding
zero. Against persistence, the GP loses on PIGD because persistence uses the last
measurement directly without shrinking it towards the population average, and for
this data that turns out to be hard to beat. So the GP is the best *fitted* model,
and still not the best *rule*.

### 26. Why was no GPU used, if GPUs were available?

On the machine where this work ran, there was no usable GPU. The installed
PyTorch was the CPU-only build, `torch.cuda.is_available()` returned False, and
`nvidia-smi` was not present. All computation was done on 12 CPU worker processes.
If a GPU machine is available elsewhere, that is a different environment and the
code would need to be pointed at it.

### 27. What compromise was made for runtime, and does it affect the conclusions?

The K2 and K3 kernels take 250 to 320 seconds per fit, which over 934 folds would
be more than 65 hours. So they were fitted only on the whole cohort for
description, and were **not** included in the held-out comparison. This does not
affect the main Step 2 conclusions, because the cross-output test was carried by
the K1 pair, which is properly nested. But it does mean the unusual K2/K3
correlation values of +0.85 and +0.98 remain untested, and we have labelled them
as an open question rather than a result.

### 28. If almost everything after GPR was a negative result, what was the value of the work?

Three things. First, we now know the ratio was the wrong target and why —
modelling the subscores directly recovers 692 visits and reveals that PIGD worsens
2.8 times faster than tremor, which the ratio hid completely. Second, we
established that the constant-rate assumption is wrong, robustly. Third, and most
importantly, we found out *why* the predictive attempts kept failing: individual
trajectory information is mostly noise at four visits per patient. That single
explanation covers the failure of random slopes, of the multi-output GP, and of
clustering. Knowing which wall you are hitting is worth more than trying a fifth
model against it.

---

# Glossary

| Term | Meaning in this project |
|---|---|
| **ARI** | Adjusted Rand Index. Measures agreement between two groupings. 1.0 is perfect, 0.0 is chance. |
| **Balanced accuracy** | Average of the accuracy on each class separately. Fairer than plain accuracy when classes are unequal. |
| **BIC** | Bayesian Information Criterion. A model-selection score; lower is better. Penalises extra parameters. |
| **BLUP** | Best Linear Unbiased Predictor. The shrunken estimate of a patient's own random effect. |
| **Coregionalization matrix** | A small table of numbers saying how strongly two outcomes share structure in a multi-output GP. |
| **Coverage (95 percent)** | How often the model's 95 percent prediction interval actually contains the true value. Should be near 0.95. |
| **CRPS** | A scoring rule that judges the whole predictive distribution, not just the point estimate. Lower is better. |
| **Indeterminate** | The middle subtype, when the ratio is between 0.90 and 1.15. |
| **Kernel** | The function in a GP that says how related two time points are. |
| **Knot** | The time at which a spline is allowed to change slope. |
| **LMM** | Linear Mixed Model. A regression with both population-level and patient-level effects. |
| **LOPO** | Leave-one-patient-out validation. |
| **MAE** | Mean Absolute Error. Average size of the mistake. |
| **Negative predictive mass** | The probability a model places on impossible values below zero. |
| **Persistence** | The baseline rule that predicts the next value equals the last observed value. |
| **PIGD** | Postural Instability and Gait Difficulty. Also the name of the subtype where it dominates. |
| **Random effect** | A device letting each patient have their own offset or slope while sharing information across patients. |
| **Reliability** | 1 minus (posterior variance / prior variance). The fraction of apparent variation that is real signal. |
| **REML** | Restricted Maximum Likelihood. A standard way of estimating variance components in mixed models. |
| **RMSE** | Root Mean Squared Error. Like MAE but punishes large mistakes more. |
| **Shrinkage** | Pulling uncertain individual estimates towards the population average. |
| **Silhouette** | A score from -1 to 1 saying how well separated clusters look. |
| **Spline (linear)** | A line allowed to bend at fixed points but not jump. |
| **TD** | Tremor Dominant. |

---

# Appendix: Where each number comes from

| Result area | Source file |
|---|---|
| GPR ratio metrics | `outputs/step3_gpr/reconciled_metrics.csv` |
| GPR paired comparisons | `outputs/step3_gpr/paired_comparisons.csv` |
| GPR subtype accuracy | `outputs/step3_gpr/gpr_metrics.csv` |
| Hierarchical Bayesian slope | `outputs/hbayes/report.txt` |
| Step 1 model and trends | `outputs/joint_subscore/model_summary.txt` |
| Step 1 prediction metrics | `outputs/joint_subscore/metrics.csv` |
| Step 1 versus persistence | `outputs/joint_subscore/paired_vs_persistence.csv` |
| Step 1 subtype | `outputs/joint_subscore/subtype_metrics.csv` |
| Step 2 population kernel fits | `outputs/joint_subscore_gp/kernel_parameters.csv` |
| Step 2 held-out metrics | `outputs/joint_subscore_gp/metrics.csv` |
| Step 2 paired tests | `outputs/joint_subscore_gp/paired.csv` |
| Step 2 subtype | `outputs/joint_subscore_gp/subtype_metrics.csv` |
| Step 2 PIGD = 0 | `outputs/joint_subscore_gp/pigd_zero_analysis.csv` |
| Step 2 runtime and convergence | `outputs/joint_subscore_gp/run.log` |
| Direction 2 composition | `outputs/time_varying/period_composition.csv` |
| Direction 2 slopes | `outputs/time_varying/period_slopes.csv` |
| Direction 2 contrasts | `outputs/time_varying/slope_contrasts.csv` |
| Direction 2 composition control | `outputs/time_varying/within_patient_rate_change.csv` |
| Direction 2 robustness | `outputs/time_varying/robustness_knots.csv` |
| Direction 2 held-out metrics | `outputs/time_varying/metrics.csv`, `paired.csv` |
| Direction 2 subtype | `outputs/time_varying/subtype_metrics.csv` |
| Direction 3 data audit | `outputs/clustering_audit/data_audit.csv` |
| Direction 3 follow-up bias | `outputs/clustering_audit/followup_bias.csv` |
| Direction 3 slope reliability | `outputs/clustering_audit/random_slope_reliability.csv` |
| Direction 3 leave-one-visit-out | `outputs/clustering_audit/leave_one_visit_out.csv` |
| Direction 3 null comparison | `outputs/clustering_audit/real_vs_null_model_selection.csv` |
| Direction 3 nuisance audit | `outputs/clustering_audit/cluster_nuisance_audit.csv` |
| Direction 3 stability | `outputs/clustering_audit/cluster_stability.csv` |

**One result is reproducible but not stored as its own file:** the matched
comparison between the Step 2 GP and the Step 1 LMM (-0.0155 tremor, -0.0217
PIGD). It was computed by joining
`outputs/joint_subscore_gp/predictions.csv` and
`outputs/joint_subscore/predictions.csv` on patient and target index. The join was
verified complete (3,361 of 3,361) with identical target times and observed
values.
