# Student-t patient slopes vs the Gaussian growth model — design

Status: **design + smoke test complete; full run NOT launched (awaiting approval).**
Motivation: Part A (`run_manifests/20261015T171236Z_bayesian_slope_clustering/`) found no stable slope clusters. Its one robust signal was that patient slopes look heavier-tailed than the Gaussian random-slope model allows: the hierarchical null p = 0.010 for ΔBIC.

## 1. Models

**Baseline (unchanged).** `hbayes.fit_growth_model`:

R_it = (β0 + b0i) + (β1 + b1i) t_it + ε_it, with b0i ~ N(0, τ0²), b1i ~ N(0, τ1²) and ε_it ~ N(0, σ²).

**Alternative.** `hbayes_t.fit_t_growth_model` (new module; `hbayes.py` is not modified). Identical except for the slope law:

b1i ~ t_ν(0, τ1²), written as b1i | λi ~ N(0, τ1²/λi) with λi ~ Gamma(ν/2, rate ν/2).

- The intercept stays Gaussian and the residual stays Gaussian.
- τ1 is now a *scale*, not an SD. The SD is τ1·√(ν/(ν−2)) for ν > 2 and is infinite for ν ≤ 2.
- Priors are as in hbayes: N(0, 100²) on β, IG(0.01, 0.01) on σ², τ0², τ1².
- ν has a Gamma(2, rate 0.1) prior (Juárez & Steel 2010; mean 20) on a 400-point log grid over [1, 300].
- Prior sensitivity: a log-uniform prior on the same grid.

## 2. Inference: changes to the sampler

The Gibbs sweep is hbayes's sweep with three changes:

1. **b_i | ·:** the per-patient prior precision becomes diag(1/τ0², λi/τ1²) instead of a shared diag(1/τ0², 1/τ1²).
2. **τ1² | ·:** IG(a + M/2, b + ½ Σ λi b1i²).
3. **Two new steps:**
   - λi | b1i, τ1², ν ~ Gamma((ν+1)/2, rate (ν + b1i²/τ1²)/2);
   - ν | λ is drawn exactly on the grid (griddy Gibbs) from M[(ν/2)log(ν/2) − lnΓ(ν/2)] + (ν/2 − 1)Σ log λi − (ν/2)Σ λi + log p(ν).

With ν fixed at ∞ (λ ≡ 1), the sampler consumes random numbers in the same order as hbayes. A test asserts that it then reproduces hbayes to floating-point precision.

**Held-out prediction for a new patient.** Script 04 draws b from its conditional given the patient's history and θ_s. For the t model, (b, λ) | history, θ_s is not conjugate. It is sampled by 20 inner Gibbs steps started from λ ~ prior. The predictive law of the next ratio is then Gaussian given λ, with b integrated out analytically. A test checks this against importance sampling over λ.

## 3. Validation strategy (full run)

- **Splits:** script 04's primary patient split (`cfg.splits.train`, seed 42) plus its six robustness splits (seeds 0–5).
- **Fitting:** both models are fitted to each training set with identical settings (4,000 draws, 2,000 burn-in, thin 5; see §4). One chain per split, as in script 04. Fit seed = split seed.
- **Scoring:** every held-out patient and every visit k ≥ 1 with finite ratios, predicted from visits < k. These are the same transitions as `hbayes.evaluate`.
- **Scorer:** both models are scored by one Rao-Blackwellised scorer. Class probabilities come from Gaussian CDFs mixed over 400 posterior draws, instead of script 04's Monte-Carlo counts. This removes Monte-Carlo noise that would otherwise blur a small difference. The script-04 scorer (`hbayes.evaluate`) is also run on the Gaussian baseline for the primary split, as a reference.

**Pre-specified metrics:**

| role | metric |
|---|---|
| **primary** | script 04's class log-likelihood per transition (TD / Indeterminate / PIGD at 1.15 / 0.90) |
| secondary | Brier score and accuracy (script 04) |
| secondary | continuous log predictive density of the next ratio, which is more sensitive to tails |

**Pre-specified decision rule.** The Student-t model is called predictively better only if all three hold:

- on the primary split, the patient-cluster bootstrap 95 % CI (2,000 resamples) of Δ = t − Gaussian class log-likelihood per transition excludes 0 in the t model's favour;
- the t model is ahead on that metric in at least 5 of the 7 splits;
- the continuous log density does not favour the Gaussian model.

Otherwise the result is reported as "no predictive advantage demonstrated", whatever ν is. Differences by history length (1, 2, 3–4, ≥5 prior visits) are reported descriptively.

**Degrees of freedom.** The full cohort (821 patients) is fitted with 4 chains × 2 ν priors. Reported: the ν posterior (median, 95 % CrI), R̂, ESS and prior sensitivity. Per split: ν median, CrI and ESS.

**A finite ν estimate is not evidence for the t model by itself.** The smoke test shows that Gaussian slopes on this design typically yield a ν posterior median of about 12. See the smoke report.

## 4. Smoke-test findings that shape the full run

Details are in `experiments/20261015T182129Z_student_t_slope_smoke/REPORT_SMOKE.md`.

- **Chain length:** ν mixes slowly (lag-1 autocorrelation 0.91). Burn-in 2,000 with thin 5 roughly doubles ν's ESS (to about 200 per chain) at about 10 s per fit. The full run therefore uses that setting for **both** models.
- **Identifiability:** ν is identified when ν ≤ 3 and **not** distinguishable between ν = 6 and ν = ∞.
- **Confound:** heavy-tailed *residuals* with Gaussian slopes produced a spurious ν ≈ 2–3 in 3 of 6 replicates (t₄ noise). A low ν for the slopes therefore cannot be attributed to slope heterogeneity unless the residual tails are also modelled. See §5.

## 5. Open design decision (for the reviewer)

The real residuals of the Gaussian fit are skewed (1.20), heavy-tailed (excess kurtosis 3.3), and have 5.7 times the Gaussian rate of |r| > 3. A t-residual variant would make the ν result interpretable:

- ε_it | ωit ~ N(0, σ²/ωit), with the same scale-mixture machinery;
- fitted (a) with Gaussian slopes and (b) with Student-t slopes.

That would be a further new model and has **not** been implemented. Without it, the full run can answer whether the t-slope model predicts better, but not whether the slopes themselves are heavy-tailed.

---

## ERRATUM (2026-10-16): the ν prior is not Gamma(2, 0.1) as implemented

The text above says ν has a "Gamma(2, rate 0.1) prior … restricted to a log-spaced grid". The code (`hbayes_t.log_prior_nu_gamma`) gives each grid point a weight proportional to the Gamma(2, 0.1) **density** at that point. On a log-spaced grid the cell width is proportional to ν, so the implied prior is

p(ν) ∝ Gamma(2, 0.1)(ν) / ν ∝ e^(−0.1ν).

That is an **Exponential(rate 0.1) truncated to [1, 300]**: mean 11.0, P(ν < 10) = 0.60. A truncated Gamma(2, 0.1) has mean 20.1 and P(ν < 10) = 0.26.

- **Verification:** the grid weights match exact truncated-Exponential(0.1) cell masses to within 0.7 % in the interior.
- **The log-uniform sensitivity prior is correctly implemented.**
- **Scope:** every result produced with this prior (script 35's smoke run) is correct for the prior actually used. Only this description is wrong.
- **Original text preserved:** `docs/cluster_phase/_frozen/07_STUDENT_T_SLOPE_SPEC.as_of_20261016.md` (SHA-256 `ad6ab045…adba1`).
- **Correct Gamma(2, 0.1):** `hbayes_tt_nu.log_prior_nu_gamma2_true` adds the log-Jacobian ln ν.
