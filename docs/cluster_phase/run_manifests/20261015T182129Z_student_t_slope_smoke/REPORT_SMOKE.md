# Student-t slope model — smoke test

Command:

```
nice -n 15 ~/envs/ppmi/bin/python scripts/35_student_t_slope_model.py --out experiments/20261015T182129Z_student_t_slope_smoke --stage smoke --jobs 12
```

Run on the login node in 38.6 s, with panel MD5 verified. Design: `docs/cluster_phase/07_STUDENT_T_SLOPE_SPEC.md`. **No Student-t vs Gaussian predictive comparison was computed.**

## A. Implementation correctness (`tests/test_hbayes_t.py`, 10 tests, all pass)

- With ν fixed at ∞, the t sampler reproduces `hbayes.fit_growth_model` draw for draw: β, σ², τ0², τ1² and all random effects, rtol 1e-9.
- The grid conditional for ν equals the summed Gamma log-likelihood up to a constant.
- The λ scale mixture produces a t_ν slope (KS test).
- Gaussian predictive moments equal hbayes's per-draw conditional algebra.
- The inner-Gibbs held-out prediction agrees with importance sampling over λ (to within 0.02). For a tail patient it shrinks less than the Gaussian model, as it should.
- With ν = 10⁷ the t predictive matches the Gaussian one.
- Class probabilities sum to 1, and the log density is exact.
- On an identified synthetic design, ν = 2.5 is recovered (median < 8) and Gaussian truth gives a median > 15.

## B. Parameter recovery on the real visit design (`recovery.csv`)

Setup:

- 821 patients with their real visit times.
- θ is set to the posterior means of the Gaussian full-cohort fit, with b1 = τ1·z.
- Function-default chains.
- 6 replicates per scenario.

| truth | residuals | ν median (median over reps) | ν 95 % CrI covers truth | P(ν < 10) | β1 CrI covers | τ1 CrI covers |
|---|---|---|---|---|---|---|
| ν = 2 | Gaussian | 2.06 | 5/6 | 1.00 | 6/6 | 6/6 |
| ν = 3 | Gaussian | 3.52 | 6/6 | 0.96 | 5/6 | 5/6 |
| ν = 6 | Gaussian | 13.1 | 5/6 | 0.37 | 5/6 | 5/6 |
| Gaussian slopes | Gaussian | 12.5 | — (upper limits 29–53) | 0.36 | 6/6 | 6/6 |
| Gaussian slopes | t₄ | 8.7 | — | 0.58 | 4/6 | 4/6 |
| Gaussian slopes | real residuals, resampled | 13.1 | — | 0.36 | 5/6 | 4/6 |

What this shows:

- **ν is identified only in the very heavy regime (ν ≤ 3).**
  - ν = 6 and Gaussian slopes give indistinguishable posteriors (median ≈ 12–13, upper limit ≈ 40).
  - The Gamma(2, 0.1) prior keeps ν finite when the data are uninformative.
  - So a moderate ν estimate carries no evidence.
- **Confound.** With Gaussian slopes but t₄ residuals, 3 of 6 replicates returned ν ≈ 2–3, with CrIs as tight as [1.51, 3.47]. Heavy-tailed observation noise can masquerade as heavy-tailed slopes.
  - Resampled real residuals did *not* do this (minimum lower limit 2.6, median 13).
  - Resampling breaks any within-patient clustering of outlying visits, though, so this does not rule the confound out for the real data.
- β1 is recovered in all scenarios (means −0.086 to −0.094 against −0.090).

## C. Sampler behaviour on script 04's primary training split (573 patients, 2,426 obs; `sampler_diagnostics.csv`)

4 chains for each model / prior / configuration. ESS is computed from the 4,000 retained draws per chain.

| config | model (ν prior) | ν median [95 % CrI] | ν R̂ | ν ESS (4 chains / 1 chain) | τ1 ESS (1 chain) | s per fit |
|---|---|---|---|---|---|---|
| default (burn-in 1000, thin 2) | t (Gamma) | 1.95 [1.35, 3.01] | 1.010 | 324 / 89 | 76 | 4.4 |
| default | t (log-uniform) | 1.88 [1.31, 2.96] | 1.013 | 313 / 81 | 96 | 4.4 |
| long (burn-in 2000, thin 5) | t (Gamma) | 1.97 [1.33, 3.09] | 1.005 | 780 / 200 | 211 | 10.5 |
| long | t (log-uniform) | 1.89 [1.33, 2.92] | 1.002 | 844 / 228 | 253 | 10.5 |
| long | Gaussian | — | — | — | 693 | 7.7 |

Under the t model:

- β1 = −0.102 [−0.125, −0.078], against −0.093 under the Gaussian model on the same split;
- the τ1 scale is 0.062, against a τ1 SD of 0.105 in the Gaussian model;
- all R̂ ≤ 1.013;
- ν's lag-1, 5, 20 and 50 autocorrelations are 0.91, 0.70, 0.34 and 0.14.

The training data put ν at about 2, the regime where ν is identifiable, and the result does not depend on the prior. Given §B, this is equally consistent with heavy-tailed slopes or with heavy-tailed or clustered residual outliers. The real residuals of the Gaussian full-cohort fit have skew 1.20, excess kurtosis 3.31, and 1.53 % of |r| > 3 against 0.27 % expected.

## D. Evaluation pipeline (25 test patients, 124 transitions)

- The transitions are identical to those of `hbayes.evaluate`.
- For the Gaussian model, the Rao-Blackwellised scorer gives class log-likelihood −0.5788 per transition; script 04's Monte-Carlo scorer gives −0.5759. Brier is 0.3118 vs 0.3073.
- The largest per-transition probability difference is 0.061, consistent with Monte-Carlo noise from 400 sampled ratios.
- All Student-t predictions are finite.
- Timing: 0.3 s for both models at 100 draws.
- The full-stage code path was also exercised once on a 150-patient subset with short chains (`experiments/_smoke/s35full_codepath/`). That checks the code path only; its numbers were not inspected.

## Limitations

- The ν grid is discrete (400 points over [1, 300]), so draws of ν are restricted to grid values (98 distinct values visited).
- A single chain per split in the held-out comparison follows script 04. Convergence is assessed from the 4-chain runs.
- The confound in §B is unresolved; see spec §5.
- With ν ≈ 2 the slope variance is barely finite or infinite, so τ1 cannot be compared directly with the Gaussian τ1.
