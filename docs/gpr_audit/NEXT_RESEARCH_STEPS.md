# Step 3 — research decision memo

Date **2026-09-30**. Follows the completed audit (`FINAL_RESULTS.md`, commit
`acff7a2`). Panel MD5 `fb038ddbdc19ab3368e13940f398e7ba`, seed 42 throughout.

**Status key** — ✅ established · ⚠️ not established · 🔬 proposal, not run

---

## 1. What the completed experiments establish

### ✅ Established

| Finding | Evidence |
|---|---|
| A flexible kernel predicts the next ratio better than a linear-trend GP | K1 − K0 = −0.033 [−0.047, −0.021] pooled; replicates in **all four cohorts** under both aggregations (8/8 intervals exclude zero) |
| The GP beats persistence and the patient's own mean | 8/8 intervals exclude zero across four cohorts |
| K0 reproduces the hierarchical Bayesian growth model | indistinguishable on held-out prediction; the sign of the difference flips with aggregation in all four cohorts |
| The fits are numerically healthy | 11,444 kernel fits, **100% converged**; hyperparameters barely move across folds |
| A fitted linear mean beats a zero mean on calibration and large errors | held-out: CRPS +0.0188 [+0.0130, +0.0250], MSE +0.0969 [+0.0634, +0.1307], both excluding zero — **but MAE shows no difference** (§3) |
| The GP's advantage over baselines **widens** with forecast horizon | multi-step: K1 − persistence −0.16 to −0.24 across cohorts, vs −0.07 to −0.14 one-step (§2) |
| Conclusions are insensitive to the extreme-patient criterion | identical model ordering in unfiltered / A4 / D1 / E2 |
| The linear *kernel* term is redundant **given** a fitted linear mean | `var_linear` pinned at its bound in ~99.9% of K2/K3 fits; the mean carries the trend (β₁ ≈ −0.082/yr) |

### ⚠️ Not established

| Claim | Why not |
|---|---|
| Matérn beats RBF | K3 − K1 includes zero in 3 of 4 cohorts one-step, and in **all four** multi-step |
| The GP is better than the HMM | **different tasks.** Continuous ratio vs categorical label; 2,707 vs 3,361 points; 821 vs 934 patients. Not comparable. |
| Any clinical utility | the target is a ratio computed by this project's own rule from a noisy exam, never validated against clinical outcome |

### The size of the effect, stated plainly

The kernel choice is worth **~0.03** ratio units. Knowing *which patient* it is
is worth **~0.4** (population-only baseline 1.39 vs patient-aware ~1.0). The
within-patient SD of the ratio is **1.22**. Kernel refinement is operating at
roughly a tenth of the individualisation effect and a fortieth of the noise.

---

## 2. Multi-step forecasts — Task 2 ✅ complete

**Command:** `python scripts/14_gpr_multistep_analysis.py --cohorts unfiltered,A4,D1,E2`
**Outputs:** `outputs/step3_gpr/multistep_by_horizon.csv`, `multistep_paired.csv`

### Design and what was available

The saved forecasts condition on the first `m = ceil(T/2)` finite observations
and predict every later visit at its actual time. Available: **630 / 617 / 503 /
451** patients and **1,389 / 1,364 / 1,150 / 985** forecast points across the four
cohorts, at horizons from 0.59 to 9.83 years (median 2.00).

Two gaps in the saved data, both closed here:

- **No baselines.** `run_multistep_fold` fits kernels only, so K0–K3 had nothing
  to be judged against. Persistence and patient-mean were reconstructed from the
  same histories — leakage-safe by construction, since both use only the first
  `m` observations the GP saw.
- **No independent leakage check.** Rebuilding the histories allowed every
  forecast row to be verified against its trajectory: the recorded origin is the
  `m`-th visit, the target is later than the origin, and the recorded target
  value matches the trajectory. **All rows passed in all four cohorts.**

### The main result: the GP's advantage *widens* with horizon

MAE by forecast horizon, unfiltered cohort:

| Model | 0–1 y (n=385) | 1–2 y (388) | 2–3 y (238) | 3–5 y (220) | 5+ y (158) |
|---|---:|---:|---:|---:|---:|
| K1 / K2 | 0.921 | 0.992 | 1.067 | 1.000 | **0.976** |
| K3 | 0.920 | 0.993 | 1.066 | 1.000 | 0.979 |
| K0 | 0.950 | 1.008 | 1.102 | 1.089 | 1.233 |
| persistence | 0.941 | 1.059 | 1.197 | 1.249 | **1.605** |
| patient mean | 0.971 | 1.047 | 1.228 | 1.282 | 1.499 |

At one year the GP barely beats persistence (0.921 vs 0.941). At five-plus years
the gap is large (0.976 vs 1.605). **The baselines degrade with horizon; the GP
does not.** The same pattern holds in every cohort — in D1 the GP actually
*improves* with horizon (0.847 → 0.754) while persistence degrades (0.940 → 1.428).

The mechanism is unsurprising once stated: persistence anchors on a single noisy
observation and that anchor ages badly, whereas the GP reverts toward a fitted
population trend, which is a better long-range guess than any one measurement.

### Paired comparisons over all forecast points

| Comparison | unfiltered | A4 | Verdict |
|---|---|---|---|
| K1 − K0 | −0.0620 [−0.0953, −0.0312] | −0.0606 [−0.0957, −0.0295] | ✅ excludes 0 |
| K3 − K1 | +0.0002 [−0.0022, +0.0025] | +0.0001 [−0.0022, +0.0025] | ⚠️ includes 0 |
| K1 − persistence | −0.1576 [−0.2328, −0.0861] | −0.1610 [−0.2376, −0.0904] | ✅ excludes 0 |
| K1 − patient mean | −0.1611 [−0.2124, −0.1137] | −0.1634 [−0.2155, −0.1144] | ✅ excludes 0 |

**K1 − K0 is −0.062 multi-step against −0.033 one-step: the flexible kernel
matters about twice as much for longer forecasts.** That is the most useful new
result in this memo, and it was invisible in the one-step analysis.

### ⚠️ Revised 2026-09-30 — horizon versus patient composition

The table above compares *bins*, and the bins contain different patients: the
5y+ bin has 100 patients against 385 in the 0–1y bin, and only patients with
long follow-up reach it. A between-bin comparison therefore mixes "predictions
get harder further out" with "the patients who get that far are different".

`scripts/15_gpr_multistep_audit.py` repeats the analysis **within patients**,
restricted at each step to patients present in both bins. Change in MAE
(positive = worse), all four cohorts:

| Transition | K1 | persistence |
|---|---|---|
| 0–1y → 1–2y | **+0.21 to +0.23, excludes 0 in 4/4** | **+0.28 to +0.30, excludes 0 in 4/4** |
| 1–2y → 2–3y | −0.04 to −0.11, includes 0 in 4/4 | −0.05 to −0.11, includes 0 in 4/4 |
| 2–3y → 3–5y | −0.09 to −0.14, includes 0 in 4/4 | +0.03 to +0.09, includes 0 in 4/4 |
| 3–5y → 5y+ | −0.20 to −0.38, excludes 0 in 2/4 (D1, E2) | +0.06 to +0.38, excludes 0 in 2/4 (unfiltered, A4) |

Two corrections follow.

**The between-bin view understated K1's early degradation.** It showed K1 nearly
flat from 0–1y to 1–2y (0.921 → 0.992); within patients the rise is +0.23 and
excludes zero in every cohort. Longer-horizon bins contain easier patients,
which flattened the curve.

**Does the advantage actually widen?** Tested directly as a
difference-in-differences — the K1-minus-persistence gap in a late bin minus the
same gap in the 0–1y bin, for patients present in both:

| Cohort | 0–1y → 2–3y | 0–1y → 3–5y | 0–1y → 5y+ |
|---|---|---|---|
| unfiltered | −0.024 [−0.157, +0.122] | −0.071 [−0.240, +0.096] | **−0.371 [−0.655, −0.104]** |
| A4 | −0.017 [−0.150, +0.126] | −0.068 [−0.237, +0.102] | **−0.364 [−0.648, −0.096]** |
| D1 | −0.085 [−0.283, +0.120] | **−0.242 [−0.460, −0.040]** | **−0.452 [−0.850, −0.084]** |
| E2 | +0.103 [−0.087, +0.323] | −0.189 [−0.380, +0.008] | −0.296 [−0.640, +0.025] |

**The widening is real, but only at long range.** It is not detectable out to
three years in any cohort. At 5y+ it excludes zero in three of four cohorts
(E2 fails with n = 34 patients, most likely power). The honest claim is:

> Beyond roughly five years, the GP's advantage over persistence grows, holding
> patient composition fixed. Within three years the two degrade in step and no
> widening is detectable.

That is a narrower claim than "the advantage widens with horizon", which is how
this was first written and which the bin-level view alone would support.

### Does the model ranking change with horizon?

K1/K2/K3 lead in every bin and K0 is the worst GP in every bin. The ordering
*within* K1/K2/K3 shuffles between bins, but their MAEs differ by <0.002 — that
shuffling is noise, not a horizon effect.

### Caveats

- Long horizons come from patients with long follow-up, who are not a random
  subset. The non-monotonicity (K1 worse at 2–3 y than at 5+ y) is most likely
  this survivorship effect, not a real improvement with distance.
- Steps 6–7 have n = 19 and n = 1. Those columns are noise and are not
  interpreted.
- Only K0–K3 were fitted multi-step; `hbayes` and the fitted baselines were not,
  so the multi-step comparison is narrower than the one-step one.

---

## 3. Held-out mean-function comparison — Task 1 ✅ complete

**Command:** `python scripts/13_gpr_mean_function_holdout.py --cohorts unfiltered --jobs 12`
**Runtime:** 44.3 min, 821 folds. **Outputs:** `mean_function_holdout_unfiltered.csv`,
`mean_function_holdout_fits_unfiltered.csv`, `mean_function_holdout_summary.csv`

### The comparison, and why it is the fair one

The article this work follows uses `m(x) = 0` and puts trend structure in the
kernel. So the opposing arm is **not** "K1 without a mean" — that would delete
the linear trend rather than relocate it. Both arms carry constant **and**
linear kernel terms; only the mean differs:

```
fitted   m(t) = b0 + b1 t   const + linear + rbf     (K2)
zero     m(t) = 0           const + linear + rbf     (K2, fit_mean=False)
```

**Implementation change required.** `run_fold` could only fit one mean
specification per kernel. A kernel entry may now be a triple
`("K2_zeromean", "K2", False)`, so both arms are fitted **inside the same fold**
— identical training set, identical held-out targets, identical histories. That
is what makes the pairing exact. Five regression tests cover it, including one
asserting the two arms are scored on identical points and one asserting they do
not produce identical predictions (which would mean the flag did nothing).

### Design

Identical to the audited protocol: 821 folds, hyperparameters and mean from
training patients only, visit *k* predicted from visits *0…k−1*, 4,000
patient-cluster bootstrap replicates with the same resampled patients for both
arms, seed 42. **Convergence 821/821 (100%) in both arms.** All 10 consistency
checks passed.

### Result

| Model | MAE pooled | MAE patient | RMSE | CRPS | coverage95 | negative mass |
|---|---:|---:|---:|---:|---:|---:|
| K2 (fitted linear mean) | 0.9720 | 1.0012 | **1.3566** | **0.7249** | 0.9409 | **0.1295** |
| K2 (zero mean) | **0.9649** | **0.9884** | 1.3918 | 0.7437 | 0.9387 | 0.1702 |

Paired, zero-mean minus fitted-mean (negative = zero-mean better):

| Metric | Difference | 95% CI | Verdict |
|---|---:|---|---|
| MAE | −0.0071 | [−0.0193, +0.0056] | includes 0 — **no difference** |
| MSE | **+0.0969** | [+0.0634, +0.1307] | **excludes 0 — fitted mean better** |
| CRPS | **+0.0188** | [+0.0130, +0.0250] | **excludes 0 — fitted mean better** |
| coverage | −0.0022 | [−0.0073, +0.0028] | includes 0 — no difference |

### What this establishes — and it is not what the in-sample result implied

**The +473 log-likelihood gap does not translate into better held-out MAE.** On
typical-case point accuracy the two are indistinguishable, and the zero-mean arm
is nominally *ahead*.

**But the fitted mean is clearly better on squared error and CRPS.** Both
intervals exclude zero, comfortably. The two facts together identify the
mechanism: the arms agree on the typical case but the zero-mean arm makes
occasional **large** errors, which MAE barely notices and MSE and CRPS punish.
That is exactly what a zero-mean prior does when a patient's history runs out —
it shrinks the extrapolation toward zero rather than toward the population
trend of 2.28. The worse negative predictive mass (0.170 vs 0.130) is the same
effect seen from another angle.

**Conclusion.** Keep the fitted linear mean. The justification is *not* the
in-sample likelihood, which over-promised; it is that the fitted mean produces
better-calibrated predictions and fewer large errors on held-out data. Anyone
reporting only MAE would wrongly conclude the choice does not matter.

⚠️ This also **supersedes the audit's own framing**: `FINAL_RESULTS.md` §4.3
presented the 473 gap as strong evidence for the fitted mean and flagged the
held-out test as outstanding. The held-out test now shows the in-sample gap was
a poor guide to the size, and to the metric, of the real advantage.

### Limitation

Run on the **unfiltered cohort only** (821 folds, 44 min). The other three
cohorts were not run; given how closely A4/D1/E2 tracked unfiltered on every
other comparison, replication is likely but is not evidence until done.

---

## 4. Modelling the subscores instead of the ratio — proposal ✅ **now tested**

The design below was documented first and then run as the smallest defensible
baseline. **Results are in §4.7; they do not support the proposal in the form
originally argued.**

### 4.1 How the ratio is currently built

From `src/ppmi_subtype/subtype.py`:

```
tremor_score = mean of 11 MDS-UPDRS tremor items      (each 0-4)
pigd_score   = mean of  5 gait/postural items          (each 0-4)
ratio        = tremor_score / pigd_score
```

Both subscores require **all** constituent items present (`max_missing = 0`).
Sentinel code 101 ("Unable to Rate") is excluded before averaging.

Degenerate cases are handled by the *label* rule but leave the *ratio*
undefined:

| Case | Ratio | Label assigned |
|---|---|---|
| `pigd_score == 0`, `tremor > 0` | undefined (÷0) | TD |
| `pigd_score == 0`, `tremor == 0` | undefined (0/0) | Indeterminate |
| `tremor_score == 0`, `pigd > 0` | exactly 0 | PIGD |

### 4.2 What is currently lost — verified against the data

In the 934-patient analysable cohort (4,295 labelled visits):

| | Visits | Patients | One-step points |
|---|---:|---:|---:|
| ratio model can use | 3,603 | 821 | 2,707 |
| both subscores present | 4,295 | 934 | 3,361 |
| **gain** | **+692** | **+113** | **+654** |

*(The frequently quoted "725 visits" is the same quantity over the full
4,441-visit labelled set including single-visit patients. Within the analysable
cohort it is 692. Both are correct for their population.)*

**The gain is not merely more data — it removes a known selection bias.** The
113 recovered patients have a mean PIGD subscore of **0.073** against **0.414**
cohort-wide, and their visits are 308 TD / 23 Indeterminate / 13 PIGD. They are
precisely the near-zero-gait, tremor-dominant patients the ratio model is
structurally blind to.

**And it makes the GPR directly comparable with the HMM for the first time**:
3,361 one-step points on 934 patients is *exactly* the HMM's scoring set.

### 4.3 Why the ratio is a bad modelling target — measured

| Quantity | tremor | PIGD | ratio |
|---|---:|---:|---:|
| median within-patient SD | 0.193 | 0.141 | **0.762** |
| median \|change\| between visits | 0.222 | 0.143 | **0.853** |
| distinct observed values | 30 | 20 | many |

The subscores are stable; the ratio is not. The division manufactures the
volatility: **ratio volatility is 1.7× higher for patients in the bottom third
of mean PIGD score** (median |Δratio| 1.061) than the top third (0.628). A small
noisy denominator is the problem, and it is the same denominator that goes to
zero 16.1% of the time.

The two subscores are also **nearly uncorrelated** — overall Pearson r = +0.047,
median within-patient r = +0.157. The ratio collapses two largely independent
signals into one number, discarding severity: 2/4 and 10/20 are both 0.5.

### 4.4 Proposed target and likelihood

**Target.** Predict the pair `(tremor_score, pigd_score)` jointly, and derive the
ratio and subtype afterwards by the existing rule. The subtype is then obtained
exactly as it is today, so the clinical definition is untouched.

**Range constraints the likelihood must respect.** Each subscore is a mean of
`k` integer items on 0–4, so the support is `{0, 1/k, …, 4}` — bounded,
non-negative, discrete, and heavily concentrated near zero (tremor = 0 at 7.2%,
PIGD = 0 at 16.1%; >2 at 0.65% and 1.33% respectively).

Three candidate likelihoods, in increasing fidelity and cost:

| | Likelihood | Respects | Cost |
|---|---|---|---|
| **A** | Gaussian on `√score` or `log(score + c)` | non-negativity approximately; stabilises variance | lowest — reuses the existing GP directly |
| **B** | Beta on `score / 4`, with zero-inflation | bounds **and** the point mass at 0 | moderate; needs a non-Gaussian GP likelihood |
| **C** | Binomial-type on the item sum (`k·score` counts out of `4k`) | the discreteness exactly | highest; the natural generative model |

**Recommendation: start with A**, because it is a drop-in change to the existing
audited GP and isolates the effect of changing the *target* from the effect of
changing the *likelihood*. Then B as a separate model. C only if B's zero
handling proves inadequate.

**Longitudinal structure.** Keep the current design: shared kernel
hyperparameters across patients, patient-specific posterior by conditioning on
that patient's own history. Two correlated outputs can be handled either as two
independent GPs (justified by r ≈ 0.05 overall) or as a multi-output GP with a
shared input kernel. **Start with two independent GPs** — the measured
correlation is weak, and independence is a testable simplification rather than
an assumption imposed for convenience.

### 4.5 Validation design

Identical to the audited protocol, so results are comparable:

- Leave-one-patient-out, 934 folds, hyperparameters and mean from training
  patients only.
- Visit *k* predicted from visits *0…k−1* of the held-out patient.
- Both aggregations (pooled, patient-averaged), patient-cluster bootstrap with
  the same resampled patients for both models.
- The same 10-check leakage battery.

**Comparison with the ratio model — and the one trap to avoid.** The subscore
model scores 3,361 points; the ratio model scores 2,707. A headline comparison
between them would be confounded by the extra 654 points, which are exactly the
*hard* ones. Two comparisons are therefore required:

1. **Restricted** — subscore model evaluated only on the 2,707 points the ratio
   model can also score. This is the like-for-like test of whether the
   representation helps.
2. **Full** — subscore model on all 3,361, reported separately as the coverage
   gain, never as a direct MAE comparison.

**Metric.** Derived ratio MAE is *not* defined on the 654 recovered points
(the ratio is infinite there). So the primary metric must be on the subscores
themselves, plus **categorical subtype accuracy on all 3,361 points** — which is
the quantity directly comparable with the HMM's 75.4%.

### 4.6 Risks

| Risk | Assessment |
|---|---|
| Two models instead of one doubles the fitting cost | Low — ~2.5 h becomes ~5 h; already tractable |
| Predicting two things well is harder than one | Real. The ratio may be noisy but it is the quantity the clinical rule uses. Mitigated by deriving the subtype from the pair and scoring that. |
| A Gaussian on a bounded score is still misspecified | Real, but strictly less so than on the ratio: the subscores are bounded and far less volatile |
| The zero point mass (16.1% of PIGD scores) | The main technical obstacle. Option A does not solve it; B or C would. |
| Derived-ratio metrics become undefined for recovered visits | Handled by the restricted/full split above |

---

## 5. Limitations carried forward

1. **Gaussian likelihood on a non-negative quantity.** 12–13% of each predictive
   distribution lies below zero. No point prediction is negative, so this is a
   tail defect — but lower interval endpoints and P(PIGD) absorb impossible mass.
2. **Non-random exclusion.** 692 visits (634 TD) are invisible to the ratio
   model. §4 addresses this directly.
3. **Single study.** Everything rests on PPMI.
4. **The target is a noisy label.** 35% of recorded subtype changes reverse at
   the very next visit; the continuous ratio inherits that variability.
5. **Multi-step is narrower than one-step** — only K0–K3 were forecast, without
   `hbayes` or the fitted baselines.
6. **The mean-function holdout covers one cohort** — unfiltered only.
7. **MAE alone is a misleading summary here.** §3 shows two models that are
   indistinguishable on MAE and clearly separated on CRPS and MSE. Any future
   comparison in this project should report a proper scoring rule alongside MAE.

---

## 6. Prioritised plan

| # | Step | Why now | Cost | Risk |
|---|---|---|---|---|
| ~~1~~ | ~~Subscore model, option A~~ | **Done (§4.7).** Recovers the data and removes the bias as predicted, but the GP loses its edge over persistence on the subscores, and the derived subtype is indistinguishable from the HMM. Adopt the target for representation, not accuracy. | — | — |
| **1** | **Multi-output GP on the two subscores** | §4.7 used two independent GPs; within-patient correlation is +0.157. A shared-input multi-output GP is the obvious next test and is the cheapest remaining idea with a plausible mechanism | ~1 day | moderate |
| **2** | **Zero-inflated / bounded likelihood** (option B) | The 12–13% impossible mass and the 16.1% zero point mass are the same problem | ~2 days | moderate — non-Gaussian GP likelihood is new machinery |
| **3** | **Multi-step for `hbayes` and the baselines** | The multi-step finding (GP advantage widens with horizon) currently rests on a narrower comparison than the one-step one | ~2 h | very low |
| **3b** | **Mean-function holdout on A4 / D1 / E2** | §3 ran on the unfiltered cohort only; replication is likely but unproven | ~2.5 h | very low |
| **4** | **Replicate on an independent cohort** | Everything rests on one study | unknown — depends on data access | high, external |
| — | ~~Further kernel refinement~~ | **Do not pursue.** Worth ~0.03 ratio units against ~0.4 for individualisation; Matérn is not distinguishable from RBF in any cohort at either horizon | — | — |

**The single recommendation:** do step 1. It is the only available change that
simultaneously fixes a bias, adds data, reduces the noise being modelled, and
makes the GPR work comparable with the project's existing HMM results.


---

## 4.7 Subscore baseline experiment — RESULT 🆕

**Command:** `python scripts/16_gpr_subscore_baseline.py --jobs 12`
**Runtime:** 9.0 + 9.8 min. **Outputs:** `subscore_predictions_{tremor,pigd}.csv`,
`subscore_fits_{tremor,pigd}.csv`, `subscore_summary.csv`

Option A without a transform — the same K1 kernel, the same LOPO protocol, the
same leakage battery, on the **raw** subscore scale. Raw scale is deliberate: it
isolates the effect of changing the *target* from changing the *likelihood*.
934 folds per subscore, **converged 934/934 in both**, all 10 checks passed.

### The target really is better behaved

| | ratio model | tremor | PIGD |
|---|---:|---:|---:|
| points scored | 2,707 | **3,361** | **3,361** |
| patients | 821 | **934** | **934** |
| K1 predictive mass below zero | **0.130** | **0.054** | 0.107 |

The coverage gain is exactly as predicted (+654 points, +113 patients), and the
Gaussian misspecification more than halves on tremor.

### But the GP's advantage over persistence does not survive

| Target | K1 MAE | persistence MAE | K1 − persistence | Verdict |
|---|---:|---:|---|---|
| ratio (for reference) | 0.9720 | 1.0427 | −0.0707 [−0.0979, −0.0451] | GP better |
| tremor score | 0.2395 | 0.2426 | −0.0031 [−0.0084, +0.0022] | **no difference** |
| PIGD score | 0.2175 | 0.2040 | **+0.0134 [+0.0083, +0.0187]** | **GP worse** |

On the PIGD subscore the trivial "next = last" rule **beats** the GP, and the
interval excludes zero. The likely reason is visible in the earlier variability
table: the subscores move by 0.14–0.19 per visit against the ratio's 0.76, so
persistence is already close to optimal and the GP's shrinkage toward the
population mean costs more than its smoothing gains.

**This is a negative result for the proposal as argued.** Modelling the
subscores does not inherit the ratio model's advantage over baselines.

### The comparison the subscore target was supposed to unlock

Deriving the ratio from the two marginal predictive Gaussians by Monte Carlo
(4,000 draws, scores clipped at zero, the existing degenerate rules applied)
gives a subtype prediction on **all 3,361 points of the 934-patient cohort —
exactly the HMM's scoring set.** Joined to the HMM's own leave-one-out
predictions on `(patient, n_history)`; the join is complete (3,361/3,361) and
validated on target times and observed subtypes.

| Model | Subtype accuracy |
|---|---:|
| GPR on subscores → derived subtype | **0.7566** |
| Hidden Markov model | **0.7539** |
| persistence (subscore-derived) | 0.7465 |
| patient mean (subscore-derived) | 0.7382 |

**Paired difference, GPR − HMM, identical points: +0.0027 [−0.0042, +0.0098] —
includes zero.** They disagree on 141 of 3,361 points, almost symmetrically
(75 GPR-only correct, 66 HMM-only).

**This is the first like-for-like comparison between the GPR line of work and
the project's HMM, and the answer is that they are indistinguishable.** That is
a genuine result, and it is not an improvement: no held-out evidence supports
claiming the GPR is better.

### What this changes

The subscore target should be adopted **for its representational properties,
not for accuracy**: it recovers 654 points and 113 patients, removes the
selection bias against tremor-dominant visits, halves the Gaussian
misspecification on tremor, and makes the work comparable with the HMM. It does
not make predictions better.

### Caveats

- Raw scale, Gaussian likelihood, two **independent** GPs. The measured
  within-patient correlation is +0.157, so independence is an approximation;
  a multi-output GP is untested.
- The derived subtype uses Monte Carlo through a ratio of two Gaussians with
  clipping at zero — a reasonable but not exact treatment of the degenerate cases.
- The subscore models train on 934 patients and the ratio model on 821, so the
  subscore-vs-ratio MAE numbers are not directly comparable (different targets,
  different units, different training sets). Only the derived-subtype
  comparison is like-for-like.
