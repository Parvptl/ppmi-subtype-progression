# Step 3 (GPR) — methodology audit

Audit date **2026-09-29**. Every statement below was established by reading the
implementation and running it, not from earlier summaries.

---

## A. The GP formulation, as actually implemented

Read from `src/ppmi_subtype/gpr.py`.

**Model.** `R_obs(t) = f(t) + ε`, `ε ~ N(0, σ_n²)`, `f(t) ~ GP(m(t), k(t,t'))`,
with `m(t) = β₀ + β₁t`.

**Posterior**, computed through a Cholesky factor, never an explicit inverse:

```
μ* = m(t*) + K*ᵀ [K + σ_n²I]⁻¹ (R − m(t))
Σ* = K**   − K*ᵀ [K + σ_n²I]⁻¹ K*
```

**Kernel components** (`KERNEL_COMPONENTS`, `_component_matrix`):

| Component | Form |
|---|---|
| constant | `σ_c²` |
| linear | `σ_l²·t·t'` |
| rbf | `σ_f²·exp(−(t−t')²/(2ℓ²))` |
| matern32 | `σ_m²·(1 + √3r/ℓ)·exp(−√3r/ℓ)`, r = \|t−t'\| — **Matérn ν = 3/2** |
| noise | `σ_n²·δ(t,t')`, added only to the training block |

**The four models are therefore:**

| | Kernel | Free hyperparameters |
|---|---|---|
| K0 | const + linear + noise | 3 |
| K1 | const + rbf + noise | 4 |
| K2 | const + linear + rbf + noise | 5 |
| K3 | const + linear + matern32 + noise | 5 |

All four carry the fitted linear mean.

**Verified numerically:**

- Kernel matrices symmetric and positive definite (`test_gpr.py`).
- Analytic gradients match finite differences to ≤ 2e-08 relative for all four
  kernels, in both the fitted-mean and zero-mean objectives.
- The batched (padded) marginal likelihood matches a plain per-patient
  reference implementation to **2e-16 relative** in value and **2e-14** in
  gradient, across ragged sequence lengths 1–14. This guards the padding/mask
  logic, which would otherwise produce a plausible but wrong number silently.
- `posterior` clips a negative diagonal to zero before any square root, and
  adds `σ_n²` only when predicting a *recorded* value rather than the latent
  trajectory.
- Optimisation: L-BFGS-B on log-parameters, bounds `(log 1e-6, log 1e6)`,
  `max_iter` 300, 4 deterministic restarts (restart 0 unperturbed, later ones
  seeded perturbations). Parameters landing on a bound are recorded per fold in
  `at_boundary`, not silently accepted.

**Numerical health, all 821 unfiltered folds × 4 kernels:** 100% converged.
Hyperparameters are near-identical across folds — e.g. K1 lengthscale IQR
3.7101–3.7147 — which is expected when one patient of 821 is removed.

---

## B. Mean function: the implementation versus the reference article

The reference (Kaixin Wang, *Introduction to Gaussian process regression, Part
1*, Data Science at Microsoft, 2022) was supplied **after** the implementation
was written. Assessed against it on 2026-09-29.

### Where the implementation already agrees

| Article | Implementation |
|---|---|
| `f ~ GP(m, k)`, finite sets jointly Gaussian | same |
| noisy form `y = f + ε`, i.i.d. | same |
| "add it to the diagonal of the covariance matrix" | `include_noise=True` adds `σ_n²I` |
| partitioned K / K\* / K\*\* posterior | same |
| RBF parameterised by lengthscale `ℓ` and variance `σ²` | `σ_f²exp(−r²/2ℓ²)` |
| linear kernel with a variance and an offset | `σ_c² + σ_l²tt'` |
| **"the combination of linear and RBF kernel … has the best balance"** | this is exactly **K2** |

### The two genuine differences

1. **Mean function.** The article states *"people typically use m(x) = 0 as GPs
   are flexible enough to model the mean."* The implementation fits
   `m(t) = β₀ + β₁t`.
2. **Hyperparameters.** The article selects them by eye from a sweep figure
   (*"the lengthscale chosen is 2.5 … the optimized variance is chosen as
   0.1"*). The implementation maximises the marginal likelihood.

### These are not equivalent, and the difference is measurable

A fitted deterministic linear mean is **not** the same object as a zero-mean GP
with constant and linear covariance terms:

- A **fitted mean** estimates β as a point value from the training patients.
  It contributes nothing to the predictive variance.
- A **constant + linear kernel** places a zero-mean Gaussian *prior* over the
  intercept and slope with variances σ_c², σ_l², and marginalises them. It
  does inflate the predictive variance, and it shrinks the implied trend
  toward zero rather than toward the data's trend.

They coincide only in the limit σ_c², σ_l² → ∞ with the mean profiled out, which
is not what either configuration does.

**Measured, under a strictly matched protocol** — same 821 patients, same 3,528
observations, same kernel, same jitter, same normalisation, same optimiser
(L-BFGS-B with analytic gradients), same 4 restarts:

| Kernel | Fitted linear mean | Zero mean | Gap | Extra params |
|---|---:|---:|---:|---:|
| K1 | −6245.21 | −6718.50 | **+473.29** | 2 |
| K2 | −6245.21 | −6718.50 | **+473.29** | 2 |
| K3 | −6245.12 | −6718.94 | **+473.83** | 2 |

As a likelihood-ratio test (the zero-mean model is nested: β₀ = β₁ = 0),
2Δ = 946.6 on χ²₂.

The zero-mean fit compensates by inflating `var_const` from **0.881 to 4.668** —
straining the constant kernel to represent a population mean of 2.28
(σ_c² ≈ 2.28² ≈ 5.2 would be needed). The article's advice is sound for its toy
data, which is centred near zero; the TD/PIGD ratio is not.

**Limitation, stated plainly.** This is an *in-sample* marginal-likelihood
comparison. It is strong evidence that the mean function is carrying real
structure, but it is not a held-out predictive comparison, and a
marginal-likelihood gap is not automatically a generalisation gap. A held-out
zero-mean LOPO run is listed as outstanding work.

### Why `var_linear` pins to its bound — reconciled

`var_linear` sits at the lower bound (1e-6) in **821 of 821** folds for K2 and
for K3. The earlier reading — "the data rejects linear structure" — is wrong.

The article puts the linear trend in the *kernel*; this implementation puts it
in the *mean*. It cannot be in both. The pinning is the optimiser correctly
reporting that the linear kernel term is redundant **given** a fitted linear
mean, not that linear structure is absent. The `β₁ = −0.082/yr` in the mean
function is that structure.

**Consequence for K1 vs K2.** They are not identical, as an earlier summary
said. Their predictions differ by at most **1.4e-04** in mean and 4.5e-05 in
standard deviation — practically but not exactly the same, because the pinned
`var_linear = 1e-6` still contributes a little.

---

## C. Leave-one-patient-out: what is and is not protected

Traced through `gpr_loo.run_fold`.

**Protected.** For held-out patient *i*, `data.sequences(exclude=i)` is the sole
route by which a training set is assembled, and every fitted component is
derived from it alone:

| Component | Source |
|---|---|
| Kernel hyperparameters θ | `fit_gpr(train, …)` |
| Mean function β | GLS inside the same fit |
| Population-only baseline | `_pooled_ols(train)` |
| Baseline predictive SDs | `_baseline_residual_sd(train, …)` |
| `hbayes` posterior | `fit_growth_model` on a panel built from `train` |

Patient *i* contributes to none of them. Verified by
`test_training_set_excludes_the_held_out_patient` and by
`n_train_patients == N−1` on every one of the 24,363 unfiltered prediction rows.

**No global preprocessing exists to leak through.** The ratio is computed
per visit from raw items upstream; there is no cohort-level centring, scaling,
or imputation anywhere in the Step 3 path. Cohort membership depends on the
extreme criteria, which are functions of a patient's own visits only.

### The task this measures — stated precisely

Visit *k* of the held-out patient is predicted from visits *0…k−1* **of that
same patient**, under a model fitted without them.

This is **personalised forecasting for a patient with history**. It is *not*
prediction for a wholly new patient, and it is not the same task as the
project's HMM subtype classification. Two consequences:

- A patient's own earlier visits are legitimately used — that is the point of
  the task — but they never enter the fit.
- GPR MAE and the HMM's 75.4% accuracy are **not comparable**: different
  targets (continuous ratio vs categorical label), different scoring sets
  (2,707 vs 3,361 points), different metrics. The subtype accuracy column in
  the GPR tables is a derived secondary quantity and is not the HMM number.

---

## D. Baseline fairness

All nine models are scored on the **same** `(patient, target_index)` rows —
enforced, not assumed: `12_gpr_reconcile_metrics.py` pivots predictions wide by
model and **aborts** if any point is unscored by any model, so a paired
comparison can never silently drop rows. Zero such rows in either completed
cohort.

| Baseline | Prediction | Information used |
|---|---|---|
| persistence | last observed ratio | patient history only |
| patient mean | mean of history | patient history only |
| population | β₀ + β₁t from pooled OLS | **training patients only** |
| patient OLS | least-squares line through history, extrapolated | patient history only |

Two fairness points worth recording:

- Point baselines need a distribution to be scored on coverage, CRPS and
  subtype probabilities. Each gets a Gaussian whose SD comes from that
  baseline's **training-fold** one-step residuals, never the held-out patient's.
- `patient_ols` is undefined with one observation or a zero time span; it
  carries the last value forward. This is documented behaviour, applied
  identically across cohorts, and is why it performs worst early.

No failed predictions were excluded anywhere: every model produces a value at
every point in both completed cohorts, verified by the "every model scored
every prediction point" check.

---

## E. Tests

See `TEST_RESULTS.md` for counts and outcomes.

---

## F. Defects found and fixed during this audit

| # | Defect | Severity | Fix | Evidence |
|---|---|---|---|---|
| 1 | `run_lopo` had **no checkpointing**; a cohort's folds lived in memory until the end. The machine slept during D1 and all completed folds were lost. | High — destroyed hours of compute, and made D1/E2 unrecoverable | `checkpoint=` / `checkpoint_every=` added; atomic `.tmp` + `replace` writes; completed patients skipped on resume | A run interrupted at fold 4 of 8 and resumed returns a frame **identical** to an uninterrupted run |
| 2 | Zero-mean vs fitted-mean compared with **different optimisers** (Nelder-Mead without gradients vs L-BFGS-B with them), confounding model difference with convergence quality | Medium — the 473 figure was not safely attributable | `fit_mean` flag added to `_objective` and `fit_gpr`, so both arms use the identical optimiser, gradients, normalisation and restarts | Gradients verified against finite differences (≤ 8e-09 rel.); gap reproduces at +473.29 |
| 3 | Paired comparisons bootstrapped the **patient-averaged** statistic while the headline table reported the **pooled** one — an interval belonging to a different estimand than the point estimate | Medium — misattributed uncertainty | `12_gpr_reconcile_metrics.py` recomputes the reported statistic inside each bootstrap replicate, for both estimands, with the same resampled patients for both models | See `FINAL_RESULTS.md`; both estimands now reported side by side |
| 4 | "K1 and K2 are identical" | Low — overstated | Corrected to "agree to 1.4e-04" | Max abs difference computed over all 2,707 points |
| 5 | "the linear kernel earns nothing" read as *the data rejects linear structure* | Low — wrong mechanism | Corrected: the fitted mean already carries it; the kernel term is redundant **given** the mean | §B above |

No defect was found in the GP mathematics, the LOPO protocol, or the leakage
controls. All 20 leakage/consistency checks pass on both completed cohorts.
