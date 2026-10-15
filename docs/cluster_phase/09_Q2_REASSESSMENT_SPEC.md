# Q2 reassessment (patient-slope tails) with the collapsed slice sampler — frozen design

**Status:** FROZEN 2026-10-16, before any reassessment computation. The run writes the SHA-256 of this file and of all code it uses into `reassessment_config.json` before sampling.

**Scope:** Q2 only. Q1 (residual tails: SUPPORTED) and Q3 (predictive value) from the original full run `experiments/20261015T190228Z_student_t_four_model_full/` are neither re-run nor revised. The original Q2 verdict, INCONCLUSIVE (convergence), stays on record unchanged. This reassessment is reported as a separate, later analysis.

**Basis:**
- original frozen design: `docs/cluster_phase/_frozen/08_STUDENT_T_RESIDUAL_SPEC.as_frozen_20261016.md` §7;
- erratum: `08_…SPEC.md`, end of file;
- sampler pilot: `experiments/20261015T200100Z_nu_slope_sampler_pilot/REPORT_PILOT.md`.

## 1. What changes, and why (approved 2026-10-16)

| item | original (frozen §7) | reassessment |
|---|---|---|
| νs update | νs ~ p(νs \| λ) on the grid | **collapsed slice:** (νs, λ) drawn as a block given (b1, τ1); νs by slice sampling on ln ν from p(νs \| b1, τ1) (`hbayes_tt_nu.fit(nu_update="collapsed_slice")`). Same posterior, better exploration (pilot). |
| required priors (for both νs and νe) | "Gamma(2, 0.1)" as implemented, which is really a truncated Exponential(0.1) (erratum), and log-uniform | **correct truncated Gamma(2, rate 0.1)** (`log_prior_nu_gamma2_true`, primary) and **log-uniform** (`log_prior_nu_loguniform`) |
| continuity sensitivity (not part of the rule) | — | the original Exponential-like prior (`hbayes_t.log_prior_nu_gamma`) |
| calibration replicates | 20 per scenario | **60 per scenario**, including the positive control |
| calibration pass threshold | ≤ 1/20 | **≤ 3/60** (observed rate ≤ 5 %) |

Everything else is unchanged from frozen §7:

- model (TT): R_it = (β0 + b0i) + (β1 + b1i)t + ε_it, b0i ~ N(0, τ0²), b1i ~ t_νs(0, τ1²), ε_it ~ t_νe(0, σ²);
- priors N(0, 100²) on β and IG(0.01, 0.01) on σ², τ0², τ1²;
- the νe update (grid, given ω);
- thresholds, splits, calibration scenario definitions, seeds and evaluation procedure.

## 2. Prior verification (done before freezing; reproduced in the smoke check and in `tests/test_hbayes_tt_nu.py`)

**Grid.** 400 points, log-spaced on [1, 300], Δ ln ν = 0.014295. Cells are equal in ln ν.

**Correct Gamma(2, 0.1).** Shape 2, rate 0.1 (scale 10). The log grid weight is ln f(ν) + ln ν = 2 ln ν − 0.1 ν + const, which is the density of u = ln ν when ν ~ Gamma(2, 0.1).

- The grid weights match the exact truncated Gamma(2, 0.1) cell masses, integrated between log-midpoints, to within 0.62 % (interior relative difference).
- The two boundary points receive full-cell weight rather than half; their total mass is about 1.3e-4.
- The slice target uses the same function as a continuous density in u on [0, ln 300]. Its difference from the exact log-density of ln ν is constant (checked).
- Truncation is to [1, 300], exactly as before.
- Implied prior: mean 20.09, median 16.95, P(ν < 10) = 0.263, P(ν < 3) = 0.032 (continuous truncated Gamma(2, 0.1): mean 20.09, P(ν < 10) = 0.261).

**Log-uniform.** Equal grid weights, flat in u. Mean 52.7, median 17.2, P(ν < 10) = 0.405.

**Continuity prior.** The original weights. It matches truncated Exponential(0.1) cell masses to within 0.72 %. Mean 11.0, median 7.95, P(ν < 10) = 0.596.

**Prior choice is shared by νs and νe within a fit,** as in the original design.

## 3. Analyses

All analyses use the TT model with `nu_update="collapsed_slice"` and initial νs = 10 (as in the original run).

**A. Full cohort** (821 patients, 3,528 observations).
- 4 chains, seeds 42–45, 4,000 draws, burn-in 2,000, thin 10.
- Each of 3 priors: Gamma(2, 0.1) true, log-uniform, continuity.
- Chain 42 under the true Gamma(2, 0.1) prior keeps random effects; this supplies the residuals for C2.

**B. Per-split fits.**
- Script 04's splits, seeds 42, 0, 1, 2, 3, 4, 5. Training patients only.
- One chain per split, seed = split seed, 4,000 draws, burn-in 2,000, thin 5.
- Each of the 3 priors.

**C. Calibration.** Full-cohort design: real patients and real visit times.

- **Truth for C1, C3 and P:** the original run's `calibration_theta`, used verbatim:
  - β0 2.12441, β1 −0.07997, τ0 1.34883, τ1 0.10542, σ 0.58897, νe 2.10297;
  - τ1 for P (TT scale) 0.07212.
- **Scenarios:**
  - **C1:** Gaussian slopes, i.i.d. t_νe residuals with scale σ.
  - **C2:** Gaussian slopes. Residuals are whole-patient blocks resampled from the real residual vectors of the full-cohort TT fit (A, true Gamma prior, chain 42; e = y − (β̄ + b̄_i)ᵀx, centred). Each simulated patient takes the residual vector of a random real patient with the same number of visits. If no other patient has that number, the patient's own vector is used.
  - **C3:** as C1 plus δi·max(0, t − ci), with δi ~ N(0, τ1²) and ci ~ U(first visit, last visit).
  - **P (positive control):** t₂ slopes with scale 0.07212, plus C1 residuals.
- **Replicates and seeds:**
  - 60 replicates per scenario.
  - Data seed `[42, 36, scenario index (C1 = 0, C2 = 1, C3 = 2, P = 3), rep]`, as in the original run. Replicates 0–19 of C1, C3 and P therefore reproduce the original simulated datasets.
- **Fits per replicate:** TT, collapsed slice, 4 chains (seeds 1000 + 10·rep + k, k = 0–3) × the 2 required priors, with A's settings (thin 10).
- **Flag:** pooled νs 97.5 % quantile < 10 under **both** required priors. Convergence is not required for a flag to count, which is conservative.
- **Also recorded per replicate:** R̂, ESS and a Monte-Carlo interval for the 97.5 % quantile. That interval is computed from the ESS of the indicator 1{νs ≤ q̂} and widened by ±2 MCSE in probability. A replicate is "borderline" if this interval contains 10.

## 4. Decision rule for Q2 (thresholds unchanged)

- **(conv) — mandatory.** In A, split-R̂(νs) < 1.01 and ESS(νs) ≥ 400 (4 chains pooled) under **each** required prior.
- **(a)** In A, the νs 97.5 % quantile is < 10 under **both** required priors.
- **(b)** In B, the νs 97.5 % quantile is < 10 in ≥ 5/7 per-split fits under the primary prior (true Gamma(2, 0.1)). The other priors are reported only.
- **(c)** In C, flags ≤ 3/60 in **each** of C1, C2 and C3.

**Verdict:**

| conditions | verdict |
|---|---|
| (conv) fails | **INCONCLUSIVE (convergence)** |
| (conv) holds; (a) or (b) fails | **NOT SUPPORTED: no evidence of extreme slope tails** (this does not mean the slopes are Gaussian) |
| (conv), (a), (b) hold; (c) fails | **INCONCLUSIVE (calibration)** |
| all hold | **SUPPORTED, within the stated calibration precision** |

The positive control and the continuity prior are descriptive. They are not part of the rule.

## 5. Operating characteristics of the ≤ 3/60 rule (computed before freezing)

If the true false-positive rate of the flag in a scenario is p, the flag count is K ~ Binomial(60, p) and the scenario passes when K ≤ 3:

| true p | P(pass one scenario) | P(all three pass, if independent) |
|---|---|---|
| 0.01 | 0.997 | 0.991 |
| 0.02 | 0.968 | 0.907 |
| 0.03 | 0.894 | 0.715 |
| **0.05** | **0.647** | 0.271 |
| 0.067 | 0.426 | 0.078 |
| 0.08 | 0.283 | 0.023 |
| 0.10 | 0.137 | 0.003 |
| 0.15 | 0.015 | < 0.001 |

A Monte-Carlo check with 200,000 binomial draws agrees with these exact values (p = 0.05 gives 0.6469, MC SE 0.0011, against 0.6473).

**Exact binomial (Clopper–Pearson) 95 % intervals for the observed count:**

| observed | two-sided 95 % interval | one-sided 95 % upper bound |
|---|---|---|
| 0/60 | [0, 5.96 %] | 4.87 % |
| 1/60 | [0.04 %, 8.94 %] | 7.66 % |
| 2/60 | [0.41 %, 11.53 %] | 10.12 % |
| 3/60 | [1.04 %, 13.92 %] | 12.42 % |

**What the rule does and does not show:**

- It caps the observed rate at 5 %. It does not show that the true rate is below 5 %.
- At 3/60 the data are compatible with true rates up to about 14 %.
- Even 0/60 gives a two-sided upper bound of 5.96 %. Only 0/60, judged by a one-sided 95 % bound (4.87 %), would support "true rate < 5 %", and the report will say so only in that case.
- Passing all three scenarios is unlikely (0.27) if every true rate is exactly 5 %. It is likely (≥ 0.91) if every true rate is ≤ 2 %.
- Each flag is computed from finite chains, which adds Monte-Carlo uncertainty. The borderline count in §3C quantifies it.

## 6. Reporting

- A: R̂, ESS and posterior intervals for all parameters under all three priors.
- B: per-split νs quantiles and ESS.
- C: for each scenario, flag counts with exact intervals, borderline counts, and the distribution of R̂ and ESS.
- Any failures, reported as failures.
- Prior sensitivity across the three priors.
