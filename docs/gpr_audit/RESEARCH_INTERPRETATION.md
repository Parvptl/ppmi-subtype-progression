# Step 3 — Gaussian process regression on the motor-subtype ratio

A research report for the project supervisor. Assumes familiarity with data
analysis; assumes no prior work with Gaussian processes.

Date **2026-09-30**. All numbers are from the audited run documented in
`FINAL_RESULTS.md`.

---

## 1. The question, and why it is a new one

Every earlier stage of this project predicted a **category**: is this patient
TD, Indeterminate, or PIGD at the next visit? The hidden Markov model does that
at 75.4% accuracy.

Step 3 asks something different:

> Given a patient's own previous ratio measurements, how well can we predict
> the **actual value** of their next one?

The motivation is that the category discards most of the information. The rule
labels a patient TD whenever the ratio is at or above 1.15 — but 1.16 and 4.50
are both TD and are not the same patient. Earlier work established that the
ratio moves by a median of 0.68 between visits against an Indeterminate band
only 0.25 wide, so the category is a coarse and noisy summary of a quantity
that is itself continuous.

Predicting the continuous value sidesteps the band entirely.

---

## 2. What is being predicted

The **TD/PIGD ratio**: mean of 11 tremor items ÷ mean of 5 gait/postural items
from the MDS-UPDRS, unchanged from the definition used throughout this project.

One property matters a great deal. When a patient's gait/postural score is
exactly zero the ratio is **undefined** — you would be dividing by zero. That
happens at 692 of 4,295 labelled visits, and **634 of those are TD**. So a
model defined on the ratio cannot see a slice of the data that is
overwhelmingly tremor-dominant. This is not a modelling choice; it is a
property of the ratio, and it means the Step 3 cohort (821 patients) is smaller
than the HMM cohort (934) and not a random subset of it.

---

## 3. Why personalised forecasting, and what that means here

The evaluation predicts visit *k* of a held-out patient from visits *0…k−1*
**of that same patient**, using a model fitted on the other 820 patients.

This is deliberately the task of forecasting for a patient **who already has
follow-up** — the situation in a clinic where someone has been seen several
times. It is not the task of predicting a brand-new patient with no history;
this project addressed that separately, and found the HMM to be the *worst*
model in that setting precisely because it has no history to condition on.

Using the patient's own earlier visits is the point of the task. What would
invalidate it is those visits influencing the model **fit**, and they do not:
every fitted quantity — kernel hyperparameters, mean function, baseline
calibration, the comparison Bayesian model — comes from the other patients
only. This was verified on all 88,000+ prediction rows, not assumed.

---

## 4. What a Gaussian process is doing here, in words

A Gaussian process is a way of putting a probability distribution over
**functions** rather than over parameters. Instead of saying "I think the
trajectory is a straight line, now estimate its slope", you say "I think the
trajectory is some smooth function, and here is how strongly I expect its value
at two nearby times to be related". That relationship is the **kernel**.

Two ingredients:

**The mean function** `m(t) = β₀ + β₁t` — where the model expects a typical
patient to be, and how that drifts. Fitted from the training patients: β₀ ≈
2.28, β₁ ≈ −0.082 per year, consistent with the −0.093/year the hierarchical
Bayesian model found earlier.

**The kernel** — how much a patient's deviation from that average at one time
tells you about their deviation at another. Four were compared:

| | Kernel | What it can represent |
|---|---|---|
| K0 | constant + linear | each patient has their own fixed offset and their own straight-line slope |
| K1 | constant + RBF | own offset, plus smooth wiggles that fade over a learned timescale |
| K2 | constant + linear + RBF | both |
| K3 | constant + linear + Matérn-3/2 | both, with rougher wiggles |

**K0 is not a new model.** A random intercept plus a random slope is
algebraically identical to the hierarchical Bayesian growth model the project
already has. Including it is the control: if the GP code is correct, K0 must
reproduce `hbayes`. It does — the two are statistically indistinguishable on
held-out prediction.

The learned RBF timescale is ≈3.7 years: a patient's deviation from the
population trend persists for a few years and then washes out.

---

## 5. Validation

**Leave one patient out.** Each of the 821 patients is withheld in turn, all
nine models are refitted on the other 820, and that patient's visits are
predicted one at a time using only what preceded each one.

Safeguards, all verified rather than assumed:

- No fitted quantity touches the held-out patient — checked on every row via a
  recorded `n_train_patients` field.
- No global preprocessing exists that could leak: the ratio is computed per
  visit from raw items far upstream, and nothing is centred or scaled across
  the cohort.
- No model was selected on held-out performance; all four kernels are reported.
- All nine models are scored on the **identical** set of points. The
  reconciliation script aborts if any point lacks a prediction from any model,
  so a paired comparison cannot silently drop rows.

**11,444 kernel fits, 100% converged.** Worth stating because the Step 2
five-latent-state HMM converged in only 43% of folds; these fits are far better
behaved, and hyperparameters barely move when one patient is removed.

---

## 6. Baselines

A model is only interesting if it beats the obvious alternatives:

| Baseline | Rule | What it tests |
|---|---|---|
| persistence | next = last observed | is there anything beyond "nothing changes"? |
| patient mean | next = average of that patient's history | is the trend doing work? |
| population only | next = β₀ + β₁t, ignoring the patient | does individualisation help? |
| patient OLS | fit a line to this patient alone, extrapolate | does pooling across patients help? |

---

## 7. Results

### Point prediction, unfiltered cohort (821 patients, 2,707 predictions)

| Model | MAE |
|---|---:|
| K3 | 0.9715 |
| K1 / K2 | 0.9720 |
| K0 ≡ growth model | 1.0053 |
| `hbayes` | 1.0062 |
| persistence | 1.0427 |
| patient mean | 1.0627 |
| patient OLS | 1.3357 |
| population only | 1.3949 |

Three things to read from this.

**The flexible kernel beats the linear one.** K1 − K0 = −0.033
[−0.047, −0.021], and the interval excludes zero in all four cohorts under both
ways of averaging. So allowing the trajectory to curve, rather than forcing a
straight line, genuinely helps.

**But the effect is small.** 0.03 ratio units, against a within-patient
standard deviation of 1.22. It is real and it replicates; it is not large.

**Individualisation matters far more than the kernel.** The population-only
baseline is 1.39 and every patient-aware model is near 1.0. Knowing *who the
patient is* is worth about ten times more than the choice of kernel.

### Uncertainty

The GP does not only produce a point; it produces a distribution. The stated
95% intervals contain the truth 93.8–94.5% of the time — close to nominal,
slightly narrow.

**However**, a Gaussian on a quantity that cannot be negative is misspecified,
and this is measurable: **12–13% of each predictive distribution lies below
zero**. No point prediction is negative, so this is a tail problem rather than
a location problem — but it means the lower interval endpoint, and any
probability derived from the lower tail (P(PIGD) in particular), absorbs
impossible mass and should not be trusted quantitatively.

### Sensitivity to the cohort

The analysis was repeated on four cohorts: unfiltered, and three definitions of
"persistently extreme" patients. **The model ordering is identical in all
four.** Absolute error falls as more extreme patients are removed, because the
remaining patients are easier — which is exactly why each cohort carries its
own recomputed baselines and cross-cohort numbers must not be compared.

### Mean function

The reference article for this work recommends a zero mean, on the grounds that
the kernel can absorb it. That is sound for data centred near zero; the ratio
has a population mean of 2.28 and a downward trend. Fitting the mean is better
by **473 log-likelihood units** for two extra parameters, under a strictly
matched comparison. The zero-mean fit tries to compensate by inflating its
constant kernel variance from 0.88 to 4.67.

This also explains an earlier puzzle. The linear *kernel* variance pins to zero
in essentially every fold — not because linear structure is absent, but because
the fitted linear *mean* already carries it. It cannot be in both places.

---

## 8. What this does and does not establish

**Established**

- A patient's future ratio is predictable from their own history better than by
  persistence or by their own average, consistently across four cohorts.
- A flexible kernel beats a linear one, by a small but replicable margin.
- The GP implementation reproduces the project's existing growth model exactly
  where it should.

**Not established**

- That GPR is better than the HMM. **The two solve different problems** —
  continuous value vs categorical label, 2,707 vs 3,361 scoring points, 821 vs
  934 patients. No number in this report should be compared with 75.4%.
- That Matérn beats RBF. Three of four cohorts give an interval including zero.
- Anything clinical. The target is a ratio computed by this project's own rule
  from a noisy examination, not a verified measure of disease progression. A
  better MAE does not imply a better understanding of the disease, and a fitted
  trend does not describe any individual patient's course.

---

## 9. Limitations

1. **The undefined-ratio problem.** 692 visits (634 of them TD) are excluded
   because the ratio does not exist there. The exclusion is systematic, not
   random, so the evaluation under-represents the most tremor-dominant visits.
2. **The Gaussian likelihood is wrong for this quantity** — non-negative,
   right-skewed, exactly zero at ~7% of visits. Quantified above.
3. **The mean-function comparison is in-sample.** A held-out zero-mean run was
   not performed.
4. **Single cohort, single study.** Everything rests on PPMI.
5. **The target is a noisy label.** Earlier work showed 35% of recorded subtype
   changes reverse at the very next visit; the continuous ratio inherits that
   measurement variability.

---

## 10. Reproducing this

```bash
cd "D:/SEM 7/BTP"
python scripts/11_gpr_trajectory_prediction.py --stage cohorts
python scripts/11_gpr_trajectory_prediction.py --stage lopo --jobs 12   # ~2.5 h
python scripts/12_gpr_reconcile_metrics.py
python scripts/11_gpr_trajectory_prediction.py --stage report
python -m pytest -q                                                     # 27 s
```

Seed 42 throughout; panel MD5 `fb038ddbdc19ab3368e13940f398e7ba`. The LOPO
stage checkpoints every 25 folds and resumes automatically.

---

## 11. Next research steps

Ordered by what the evidence actually supports, not by assuming GPR must win.

1. **Model the two subscores instead of their ratio.** This is the strongest
   step available and it addresses the deepest limitation. The ratio is
   undefined for 692 visits and discards severity (2/4 and 10/20 are both 0.5).
   Modelling tremor and gait jointly recovers both. It also removes the
   non-negativity problem, since the subscores are bounded and well behaved.
2. **Fix the likelihood.** 12–13% impossible predictive mass is a real defect.
   A log or square-root transform, or a censored likelihood for the exact
   zeros, evaluated under the same leakage-safe protocol as a *separate* model.
3. **Test the mean function on held-out data.** The 473-unit gap is in-sample.
4. **Analyse the multi-step forecasts.** `trajectory_predictions_*.csv` already
   exist for all four cohorts and were not examined in depth. They answer "how
   far ahead is this useful?", which is the clinically relevant question and is
   not addressed by one-step-ahead MAE.
5. **Do not pursue kernel refinement further.** The kernel choice is worth ~0.03
   ratio units while knowing the patient is worth ~0.4. Effort is better spent
   on the representation than on the covariance function.
