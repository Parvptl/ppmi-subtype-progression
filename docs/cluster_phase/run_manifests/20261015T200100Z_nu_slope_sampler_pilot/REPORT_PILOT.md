# Slope degrees-of-freedom sampler — convergence pilot

Command:

```
nice -n 15 ~/envs/ppmi/bin/python scripts/37_nu_slope_sampler_pilot.py --out experiments/20261015T200100Z_nu_slope_sampler_pilot --jobs 16
```

Run on the login node in 182 s; panel MD5 verified. Code: `src/ppmi_subtype/hbayes_tt_nu.py` (new) and `tests/test_hbayes_tt_nu.py`. SHA-256 hashes are in `pilot_config.json`, which was written before sampling.

**Untouched:** the frozen full run `20261015T190228Z_student_t_four_model_full/` (outputs, `frozen_config.json` and verdict) and `hbayes.py`, `hbayes_t.py`, `hbayes_tt.py` (their hashes still match the frozen config).

**The scientific conclusions are unchanged:** residual heavy tails are supported, and heavy-tailed patient slopes remain **inconclusive**.

## 1. Diagnosis

The frozen sampler draws νs from p(νs | λ1…λM) exactly on the grid. That step is a correct Gibbs update. The problem is how narrow it is once M = 821 mixing weights are fixed:

- the median conditional SD of νs given λ is **0.086** (90th percentile 0.119);
- the posterior SD of νs is 0.75–0.88 under the frozen Gamma-labelled prior with the collapsed samplers.

So νs can only move as fast as the λ's drift, which is a classic data-augmentation bottleneck. The posterior has a thin right tail that a correct sampler visits briefly and often. The frozen sampler instead enters it rarely and then stays for thousands of sweeps (`figures/trace_nu_slope_all_sweeps.png`, top panel: chain 0 around sweep 27,000, chain 3 around sweep 36,000). Those long sticky excursions produced the R̂ 1.018 / ESS 248 failure.

The pilot reproduces it with the original sampler and new initialisations: **R̂ 1.019, ESS 187** under the Gamma-labelled prior. Slice-sampling the same conditional p(ν | λ) would not help, because it is the conditioning on λ that is too tight.

## 2. Proposed update: collapse λ out of the ν step

Update (νs, λ) as one block given (b1, τ1²):

1. Draw νs ~ p(νs | b1, τ1) ∝ p(νs) · Π_i t_νs(b1i/τ1)/τ1. This is the λ-marginal; the derivation is in the module docstring:

   log Π_i t_ν(z_i) = M[lnΓ((ν+1)/2) − lnΓ(ν/2) − ½ ln(νπ)] − ((ν+1)/2) Σ_i ln(1 + z_i²/ν), with z_i = b1i/τ1.

2. Draw λi ~ Gamma((νs+1)/2, rate (νs + b1i²/τ1²)/2), the unchanged conditional.

Since p(ν, λ | b1, τ1) = p(ν | b1, τ1)·p(λ | ν, b1, τ1), this is an exact blocked Gibbs step. The posterior is unchanged; only its exploration changes. All other blocks, including the residual νe step, are unchanged.

Two ways to draw step 1:

- **collapsed_grid:** exactly on the frozen 400-point grid with the frozen prior weights. The model and prior are identical to the frozen run.
- **collapsed_slice:** slice sampling (stepping out and shrinkage) on u = ln ν over [0, ln 300], with target l(e^u) + w(e^u). Here w is the same prior-weight function. On a log-spaced grid, w is the prior density in u, so this is the continuous analogue of the grid prior.

**Tests** (`tests/test_hbayes_tt_nu.py`, 9 tests, all pass):

- the original option reproduces `hbayes_tt.fit` draw for draw;
- the collapsed log-likelihood equals Σ `scipy.stats.t.logpdf` and equals the Monte-Carlo λ-marginal;
- grid draws match the exact conditional, and slice draws match its quantiles to within 5 %;
- the slice step respects the bounds and is uniform on a flat target;
- both new samplers agree in distribution with the original on simulated data and give at least 1.5× its ESS;
- the implied priors are as stated in §4.

## 3. Pilot results

Full cohort: 821 patients, 3,528 observations. TT model, frozen full-cohort settings (4,000 draws, burn-in 2,000, thin 10). 4 chains with **over-dispersed starts νs = 1.2, 4, 30, 250** and seeds 42–45. The thresholds are unchanged: R̂ < 1.01 and ESS ≥ 400.

| sampler | prior | νs median [95 % CrI] | R̂ | ESS | s per chain |
|---|---|---|---|---|---|
| original (λ-grid) | Gamma-labelled (frozen) | 1.99 [1.45, 3.67] | **1.019** | **187** | 40 |
| original (λ-grid) | log-uniform (frozen) | 1.93 [1.43, 3.01] | 1.001 | 1,420 | 40 |
| **collapsed grid** | Gamma-labelled (frozen) | 1.99 [1.45, 3.32] | **1.002** | **2,140** | 139 |
| **collapsed grid** | log-uniform (frozen) | 1.93 [1.43, 3.05] | **1.001** | **2,981** | 139 |
| collapsed grid | true Gamma(2, 0.1) | 2.10 [1.49, 5.40] | 1.002 | 1,456 | 129 |
| **collapsed slice** | Gamma-labelled (frozen) | 2.00 [1.45, 3.26] | **1.002** | **2,101** | 47 |
| **collapsed slice** | log-uniform (frozen) | 1.94 [1.43, 3.02] | **1.001** | **4,206** | 47 |
| collapsed slice | true Gamma(2, 0.1) | 2.09 [1.49, 4.75] | 1.004 | 1,127 | 44 |

**Pilot rule (fixed in `pilot_config.json` before sampling).** A sampler passes if, under both frozen priors, νs has R̂ < 1.01 and ESS ≥ 400, and β1, τ1, σ and νe all have R̂ < 1.01. **Collapsed grid: PASS. Collapsed slice: PASS. Original: FAIL.** For the collapsed samplers the other parameters have R̂ ≤ 1.005, with τ1 ESS 1,583–2,140.

**Sensitivity to initialisation.** Every collapsed chain, started anywhere from 1.2 to 250, enters the pooled central 95 % interval [1.45, 3.32] within 19–244 sweeps, against a burn-in of 2,000. The per-chain medians for each collapsed group lie within 1.93–2.02 under the frozen priors and within 2.07–2.13 under the true Gamma prior. See `figures/trace_nu_slope_initialisation.png`.

**Autocorrelation** (`figures/acf_nu_slope.png`): the collapsed samplers decorrelate within about 10 retained draws; the original needs about 50.

**Agreement between priors.** Under the two frozen priors the posteriors almost coincide: medians 1.99 vs 1.93, 97.5 % quantiles 3.26–3.32 vs 3.02–3.05. Under a true Gamma(2, 0.1) prior the median barely moves (2.09–2.10), but the upper tail is more prior-sensitive (97.5 % quantile 4.75–5.40). Per-chain 97.5 % quantiles range from 3.9 to 9.0 (`figures/posterior_nu_slope_by_prior.png`). The 97.5 % quantile is the statistic used by Q2(a), so its prior sensitivity matters.

**Agreement with the frozen full run** (original sampler, seeds 42–45, initial ν = 10):

| prior | frozen run, median [95 % CrI] | collapsed samplers, 97.5 % quantile |
|---|---|---|
| Gamma-labelled | 1.99 [1.45, 3.47] | 3.26–3.32 |
| log-uniform | 1.93 [1.43, 3.01] | 3.02–3.05 |

The frozen run's centre was right; only its tail exploration was unreliable.

**Cost:** the slice version takes about 47 s per chain, against about 139 s for the collapsed grid (400 grid evaluations per sweep) and about 40 s for the original.

## 4. Prior labelling issue found during this investigation

`hbayes_t.log_prior_nu_gamma` gives each point of the log-spaced grid a weight proportional to the Gamma(2, 0.1) *density*. Cell width grows with ν, so the implied prior on ν is Gamma(2, 0.1)(ν)/ν ∝ e^(−0.1ν). That is an **Exponential(0.1) truncated to [1, 300]** (mean 11.0, P(ν > 10) = 0.41), not the Gamma(2, 0.1) (mean 20.1, P(ν > 10) = 0.74) that specs 07 and 08 state.

- The frozen run applied this prior consistently, so its numbers are correct for the prior actually used. The spec text describes it wrongly.
- A Jacobian-corrected true Gamma(2, 0.1), `log_prior_nu_gamma2_true`, is included above. It shifts the νs posterior only slightly in the median, and more in the upper tail.
- The log-uniform prior is correctly implemented. Equal weights per log-cell give a log-uniform prior.
- The residual νe prior uses the same function, so it is also effectively Exponential(0.1). νe was data-dominated (posterior 2.10 [1.88, 2.39]).

## 5. Limitations of the pilot

- One dataset (the full cohort) and one model (TT). Per-split and calibration fits were not re-run with the new sampler.
- The thin right tail is rare. Pooled draws above 10 make up 0.0–0.4 % under the frozen priors and up to 2 % under the true Gamma prior. The 97.5 % quantile is reasonably stable, but its estimate is still noisier than the median.
- The collapsed grid is about 3× slower than the slice version for the same mixing.
