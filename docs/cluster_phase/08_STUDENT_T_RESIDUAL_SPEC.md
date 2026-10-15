# Slope tails vs residual tails — four-configuration design

Status: **design + smoke test complete; full run NOT launched (awaiting approval).** This supersedes the decision rule in `07_STUDENT_T_SLOPE_SPEC.md` §3. Script 35, its module `hbayes_t.py` and its smoke results are kept unchanged.
Smoke results: `experiments/20261015T184027Z_student_t_residual_smoke/REPORT_SMOKE.md`.

## 1. Model parameterisations

All four configurations share:

R_it = (β0 + b0i) + (β1 + b1i)·t_it + ε_it, with b0i ~ N(0, τ0²).

They differ only in the slope law and the residual law:

| config | patient slope b1i | visit residual ε_it | implementation |
|---|---|---|---|
| **GG** | N(0, τ1²) | N(0, σ²) | `hbayes.fit_growth_model` (baseline; reproduced exactly by `hbayes_tt.fit(…, "GG")`) |
| **TG** | t_νs(0, τ1²) | N(0, σ²) | `hbayes_t.fit_t_growth_model` (reproduced exactly by `hbayes_tt.fit(…, "TG")`) |
| **GT** | N(0, τ1²) | t_νe(0, σ²) | `hbayes_tt.fit(…, "GT")` |
| **TT** | t_νs(0, τ1²) | t_νe(0, σ²) | `hbayes_tt.fit(…, "TT")` |

**Scale-mixture form (used by the sampler):**

- b1i | λi ~ N(0, τ1²/λi), with λi ~ Gamma(νs/2, rate νs/2);
- ε_it | ωit ~ N(0, σ²/ωit), with ωit ~ Gamma(νe/2, rate νe/2);
- τ1 and σ are *scales*. They are SDs only when the matching ν is ∞, so they are not comparable across configurations.

**Priors:** N(0, 100²) on (β0, β1); IG(0.01, 0.01) on σ², τ0², τ1². Each of νs and νe has a Gamma(2, rate 0.1) prior on a 400-point log grid over [1, 300], and is sampled exactly on the grid. Sensitivity prior: log-uniform on the same grid.

**Gibbs sweep, in this order (matching the hbayes / hbayes_t random-number order when residuals are Gaussian):**

1. b_i | ·: precision Σ_t ωit x_it x_itᵀ/σ² + diag(1/τ0², λi/τ1²).
2. β | ·: weighted least squares with weights ωit.
3. σ² ~ IG(a + N/2, b + ½ Σ ωit e_it²); τ0² as in hbayes; τ1² ~ IG(a + M/2, b + ½ Σ λi b1i²).
4. If slopes are t: λi ~ Gamma((νs+1)/2, rate (νs + b1i²/τ1²)/2), then νs | λ on the grid.
5. If residuals are t: ωit ~ Gamma((νe+1)/2, rate (νe + e_it²/σ²)/2), then νe | ω on the grid.

**Held-out prediction.** (b, λ, ω_hist) | history, θ_s is sampled by 20 inner Gibbs steps from the priors, and b is then integrated out analytically. For t residuals, the next visit's ω is drawn 8 times from its prior. The predictive is an equal-weight mixture of Gaussians, which is exact for class probabilities and log density, up to the Monte-Carlo over ω.

## 2. Validation (full run)

- **Splits:** script 04's primary split (seed 42) plus robustness splits 0–5.
- **Fitting:** all four configurations on each training set, with identical settings (§4) and fit seed = split seed.
- **Transitions:** one evaluator scores all four models on **the same** one-step-ahead transitions, the same set as `hbayes.evaluate`.
- **Metrics:** class log-likelihood (script 04's metric), Brier score, accuracy and the next ratio's log predictive density (LPD).
- **Uncertainty:** paired differences with patient-cluster bootstrap 95 % CIs (2,000 resamples).
- **Full-cohort fits:** TT and GT on all 821 patients, with 4 chains × 2 ν priors.

## 3. What the smoke test says the data can and cannot identify

This uses the actual visit design of script 04's primary split: 573 training patients and 854 test transitions. There are 8 truths with 6 replicates each, and every configuration is fitted to every replicate.

- **Residual tails are well identified.**
  - TT's νe posterior separates t-noise truths (median 2–4, 97.5 % ≤ 6 in 30/30) from Gaussian-noise truths (median ≈ 27, 97.5 % ≥ 46 in 18/18).
  - Held-out LPD detects the residual contrast (GT−GG and TT−TG) in 30/30 heavy-noise replicates.
- **Slope tails are identified only when very heavy (ν ≈ 2), and only by TT's νs posterior.**
  - The flag "TT νs 97.5 % < 10" fires in **11/12** replicates with νs = 2 and in **2/12** with νs = 3.
  - It fires in **0/24** replicates with Gaussian slopes: GG, t₄ noise, patient-level noise heterogeneity, and t-noise matched to the real data (νe = 2.2).
- **TG is not safe.** With Gaussian slopes and real-level noise (GT2.2), TG's νs median is below 5 in 5/6 replicates. This is the confound found in the first smoke test, now confirmed at the real noise level.
- **Held-out prediction cannot detect slope tails.**
  - TT−GT never reaches significance on the class metric (0/48).
  - On LPD it is significant in 2/6 GT2.2 replicates, where the slopes are Gaussian, and in 0/6 of the T2T2.2 replicates, where they are not.
  - So a predictive gain of TT over GT is not evidence of slope tails, and its absence is not evidence against them.
- **Little posterior dependence:** corr(log νs, log νe) within TT is −0.06 to 0.07 across all scenarios and −0.05 on the real data.

## 4. Proposed full-run decision rule (pre-specified)

**Q1 — Are visit-level residuals heavy-tailed?** Supported only if all three hold:

- on the primary split, the 95 % CI of ΔLPD excludes 0 in favour of t residuals for **both** GT−GG and TT−TG;
- the same sign holds in ≥ 5/7 splits;
- class log-likelihood does not significantly favour the Gaussian-residual model.

LPD is used for Q1 because the class metric detected the contrast less consistently in the smoke test (GT−GG significant in 2–6/6 replicates, against 6/6 for LPD).

**Q2 — Is there heavy-tailed patient-slope heterogeneity beyond heavy-tailed noise?** Supported only if **all** hold:

- (a) The full-cohort TT νs has 97.5 % < 10 under both ν priors, with R̂ < 1.01 and νs ESS ≥ 400.
- (b) The TT νs 97.5 % < 10 in ≥ 5/7 per-split training fits.
- (c) **Calibration on the full-cohort design:** the false-positive rate of (a) is ≤ 5 % in ≥ 20 Gaussian-slope replicates of each of the following scenarios:
  - (i) νe at the full-cohort GT estimate;
  - (ii) residuals resampled as whole-patient blocks from the TT fit, which preserves within-patient clustering and skew;
  - (iii) piecewise-linear trajectories with a Gaussian random change in slope, which tests the nonlinearity confound.

If (a) and (b) hold but (c) fails, the result is reported as inconclusive. TG's νs is not used. TT−GT predictive differences are reported only descriptively. A finite νs estimate on its own is never sufficient.

**Q3 — Does any heavy-tailed configuration predict better for clinical use?** Report script 04's class log-likelihood, Brier and accuracy for all four configurations, with the same split-consistency rule as Q1. This is reported separately from Q1/Q2.

**Multiplicity:** four contrasts × two metrics are reported. Only the pre-specified combinations above are used for conclusions.

## 5. Settings for the full run

- **Per-split prediction fits:** 4,000 draws, burn-in 2,000, thin 5 (as in the smoke test), one chain per split.
- **Full-cohort TT / GT and calibration fits:** thin 10 (42,000 sweeps, about 30 s each). On the real training split, TT's νs had ESS of only about 45 per chain at thin 5 (lag-1 autocorrelation 0.955, R̂ 1.017).

## 6. Limitations

- The linear trajectory is assumed. Nonlinearity is the main untested alternative explanation for heavy slope tails, which is why calibration (c)(iii) is required.
- The t residuals are symmetric, while the real residuals are skewed (1.2). Calibration (c)(ii) addresses this.
- The ν grid is discrete.
- Identifiability is weak for νs ≳ 3: the design cannot tell νs = 3 or 6 from Gaussian slopes, so a negative Q2 result means "no evidence of *extreme* tails", not "slopes are Gaussian".
- Per-split prediction uses a single chain, as in script 04.

---

## 7. FROZEN ADDENDUM (fixed before the full run; approved 2026-10-16)

Everything below is fixed. The full run records the SHA-256 of this file, `hbayes_tt.py`, `hbayes_t.py`, `hbayes.py` and script 36 in `frozen_config.json` before computing anything.

**Models and priors:** exactly §1. The ν grid is 400 log-spaced points over [1, 300]. The primary ν prior is Gamma(2, rate 0.1); the sensitivity prior is log-uniform on the same grid.

**Splits:** `splitting.split_patients(panel, cfg.splits["train"], seed)` with seeds **42 (primary), 0, 1, 2, 3, 4, 5**.

**Fits:**

| fit | settings | seed |
|---|---|---|
| Per-split prediction fits (GG, TG, GT, TT) | 4,000 draws, burn-in 2,000, thin 5, one chain | split seed |
| Full cohort (821 patients): GT and TT | 4,000 draws, burn-in 2,000, thin 10; 4 chains × 2 priors | 42–45 |
| Calibration fits: TT only | as the full-cohort fits: 4 chains × 2 priors | 1000 + 10·rep + k, for chain k = 0…3 |

**Held-out scoring:** all four models are scored on the identical transition set (every test patient, every visit k ≥ 1 with a finite ratio). Settings: 400 posterior draws (evenly spaced), 20 inner Gibbs steps, 8 ω draws for the next visit. Random-number seed = split seed. Additionally, `hbayes.evaluate` is run on the primary-split GG fit as a reference.

**Bootstrap:** patient-cluster, 2,000 resamples, primary split only.

**Q1 (residual tails): SUPPORTED** if all of the following hold. Otherwise NOT SUPPORTED.

- Primary-split 95 % CI of mean ΔLPD > 0 for both GT−GG and TT−TG.
- Mean ΔLPD > 0 in ≥ 5/7 splits for both contrasts.
- The primary-split class-log-likelihood CI upper bound is not < 0 for either contrast.

**Q2 (patient-slope tails beyond residual tails): SUPPORTED** only if (conv), (a), (b) and (c) all hold.

- **(conv) — mandatory.** For the full-cohort TT fits under each prior (4 chains pooled): split-R̂(νs) < 1.01 and ESS(νs) ≥ 400. If this fails under either prior, Q2 is **INCONCLUSIVE (convergence)**, whatever (a)–(c) show.
- **(a)** The full-cohort TT νs 97.5 % quantile (4 chains pooled) is < 10 under **both** priors.
- **(b)** The TT νs 97.5 % quantile is < 10 in ≥ 5/7 per-split fits.
- **(c)** For each calibration scenario C1–C3 below, the number of false positives is ≤ 1 out of 20 (an observed rate ≤ 5 %).
  - A false positive is a replicate whose TT fit meets (a): the 97.5 % quantile is < 10 under both priors. Convergence is not required for this count, which is conservative.
  - The report gives Clopper–Pearson 95 % intervals. With 20 replicates the precision is limited: 0/20 has an upper bound of 16.8 % and 1/20 of 24.9 %. An observed 0/20 does not show that the true rate is below 5 %.
- If (conv), (a) and (b) hold and (c) fails: **INCONCLUSIVE (calibration)**.
- If (conv) holds and (a) or (b) fails: **NOT SUPPORTED: no evidence of extreme slope tails**. As §6 notes, this does not mean the slopes are Gaussian.

**Calibration scenarios** use the full-cohort design: 821 patients, real visit times, 20 replicates each. All slopes are Gaussian. Truth comes from the full-cohort GT fit (Gamma-prior chains pooled, posterior means): β0, β1, τ0, τ1, σ, and ν̂e = the posterior median of νe.

| scenario | residuals | slopes / trajectory |
|---|---|---|
| **C1** | i.i.d. t_ν̂e, scale σ | linear, Gaussian |
| **C2** | real residual vectors from the full-cohort TT fit (Gamma prior, chain 42; e = y − (β̄ + b̄_i)·x, centred), resampled as **whole-patient blocks**: each simulated patient takes the residual vector of a random real patient with the same number of visits. If no other patient has that count, the patient's own vector is used. | linear, Gaussian |
| **C3** | t_ν̂e residuals as in C1 | piecewise-linear: y += δi·max(0, t − ci), with δi ~ N(0, τ1²) and ci ~ Uniform(first visit, last visit) of patient i |
| **P (positive control; descriptive only, not part of the rule)** | t_ν̂e residuals | slopes t₂ with the full-cohort TT τ1 scale |

Random-number seed for each replicate: `[42, 36, scenario index, rep]`.

**Q3 (predictive value; reported separately):**

- **Metric:** script 04's class log-likelihood per transition.
- **Comparisons:** each of TG, GT and TT against GG.
- **Rule:** "better" if the primary-split CI excludes 0 in the model's favour **and** the mean Δ > 0 in ≥ 5/7 splits. Brier score, accuracy and LPD are reported descriptively.

---

## ERRATUM (2026-10-16): the ν prior is not Gamma(2, 0.1) as implemented

This erratum was appended after the full run. §1–§7 above are the frozen text, byte-identical to the copy preserved at `docs/cluster_phase/_frozen/08_STUDENT_T_RESIDUAL_SPEC.as_frozen_20261016.md`.

- That copy's SHA-256, `ddbbb0a068e90b0e687d1646d23e0eb30800ef9dacd6cd7d718f2535fb577da7`, is the hash recorded in `experiments/20261015T190228Z_student_t_four_model_full/frozen_config.json`.
- This live file's hash differs only because this erratum was appended.

**What is wrong.** §1 and §7 describe each ν (νs and νe) as having a "Gamma(2, rate 0.1)" prior on the log grid. The implementation (`hbayes_t.log_prior_nu_gamma`, used by `hbayes_tt`) weights each grid point by the Gamma(2, 0.1) density. Cells are equal in ln ν, so the implied prior is p(ν) ∝ e^(−0.1ν): an **Exponential(rate 0.1) truncated to [1, 300]**.

| prior | mean | P(ν < 10) |
|---|---|---|
| as implemented: truncated Exponential(0.1) | 11.0 | 0.60 |
| as documented: truncated Gamma(2, 0.1) | 20.1 | 0.26 |

The log-uniform sensitivity prior is correctly implemented.

**Consequences.**

- The full run's numbers and verdicts are correct *for the prior actually used*, labelled "gamma" in its outputs. They remain the frozen, original result:
  - Q1 SUPPORTED;
  - Q2 INCONCLUSIVE (convergence);
  - Q3 as reported.
- None of them is rewritten.
- The νe posterior was data-dominated (2.10 [1.88, 2.39]) under both priors.
- The Q2 reassessment (`09_Q2_REASSESSMENT_SPEC.md`) uses a correctly implemented Gamma(2, 0.1), `hbayes_tt_nu.log_prior_nu_gamma2_true`, as a required prior, and keeps this Exponential-like prior as a continuity sensitivity analysis.
