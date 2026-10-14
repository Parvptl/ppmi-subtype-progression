# Stage 6b — Amendment 3: within-patient persistence (predictive check only)

Dated 2026-10-14. Written before implementation and before any result for this
model. Companion to `06_AMENDMENT2_PREDICTIVE_CHECK_SPEC.md` (SHA-256
`3ac7255d…`), whose check returned NO-GO (14/18; run
`experiments/20261014T165801Z_s6b_grm_predictive_check`). That file, the
Amendment 2 proposal (`32f90ac4…`) and the original pre-registration with
Amendment 1 (`1b01bb6a…`) are unchanged. The run records this file's SHA-256.

**Scope: a measurement-model predictive check only.** No clustering
calibration, no null simulation, no clustering metric.

## 1. Motivation

The graded-response model reproduced the zero frequency (0.175 vs 0.161) but
not its persistence: P(next 0 | S = 0) 0.396 vs 0.459 and P(next 0 | S = 1)
0.177 vs 0.109. Given a random intercept and a linear slope, visits are
conditionally independent, so the model cannot keep a patient near the floor
for consecutive visits beyond what the line implies. Stage 3 showed the same
on the continuous scale (trend + stationary deviation kernels K2/K3 beat the
linear K0 held out). Amendment 2 listed an Ornstein–Uhlenbeck (OU) deviation as
the planned sensitivity. Option (a) was chosen by the user.

## 2. Model (the only change)

    θ_ij = β0 + β1 t_ij + b_i0 + b_i1 t_ij + u_ij
    (b_i0, b_i1) ~ N(0, D)
    u_i ~ N(0, σ_u² R_i),   R_i[j, k] = exp(−|t_ij − t_ik| / ℓ)        (OU / exponential)

so θ_i ~ N(m_i, K_i), K_i = Z_i D Z_iᵀ + σ_u² R_i (+ 1e-8 I). Items given θ:
the same GRM, with the same stage-1 item parameters (`grm.fit_items`, refitted
identically and deterministically from the same data).

Meaning and assumptions: u is a stationary, mean-zero, Gaussian deviation of
the latent PIGD trait around the patient's line, with correlation decaying
over ℓ years; σ_u and ℓ are shared by all patients. ℓ → ∞ makes u another
random intercept; ℓ → 0 makes it extra visit-level noise (no persistence). It
adds no groups: the latent distribution remains Gaussian and single-mode.

Parameters (7): β0, β1, D (log-Cholesky, 3), log σ_u, log ℓ. Bounds:
σ_u ∈ [e⁻⁶, e²], ℓ ∈ [0.05, 50] years. A parameter on a bound is reported as
weakly identified.

## 3. Estimation

Laplace approximation to each patient's marginal likelihood over the n-vector
θ_i (n = visits), mode by damped Newton (the GRM log-likelihood is concave in
θ, the prior Gaussian, so the mode is unique). Exact gradient: explicit terms
plus the implicit dependence of the mode, using third θ-derivatives of the
GRM log-likelihood (as in Rasmussen & Williams 2006, §5.5.1, with a non-zero
mean). Patients are batched by visit count. Optimiser: L-BFGS-B, start at the
Amendment 2 stage-2 estimates with σ_u = 0.3, ℓ = 2.

Diagnostics reported, not gated: convergence, bound hits, gradient norm;
Laplace vs adaptive quadrature for patients with 2–3 visits; the in-sample
Laplace log-likelihood gain over the Amendment 2 model.

## 4. Simulation, targets, tolerances, gate

Identical to `06_AMENDMENT2_PREDICTIVE_CHECK_SPEC.md` §4–§5: R = 40 datasets
(seeds `[42, 33, r]`), θ_i ~ N(m_i, K_i) on each patient's real visit times,
five items from the GRM, PIGD = mean of the items, tremor real, then the
unchanged Gaussian RS-LMM / BLUP feature pipeline. The same 18 gated targets
with the same tolerances, the same "undecided if MC SE > tol/2" rule, the same
reported-only features. **GO only if all 18 pass.**

A NO-GO ends this line of model extension unless the user decides otherwise;
no further component is added automatically.
