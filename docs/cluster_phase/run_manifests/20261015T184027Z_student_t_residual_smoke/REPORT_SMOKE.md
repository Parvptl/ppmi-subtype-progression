# Slope-tail vs residual-tail models — smoke test

Command:

```
nice -n 15 ~/envs/ppmi/bin/python scripts/36_student_t_residual_variants.py --out experiments/20261015T184027Z_student_t_residual_smoke --stage smoke --jobs 12
```

Run on the login node in 265.8 s; panel MD5 verified. Design: `docs/cluster_phase/08_STUDENT_T_RESIDUAL_SPEC.md`. **No real-data held-out comparison was computed.**

The previous 6-scenario run is kept at `experiments/_smoke/superseded_20261015T183613Z_s36_6scen/`. Its recovery rows are byte-identical to the first 36 rows here, and its real-data diagnostics are identical apart from timing. The script-35 smoke run (`20261015T182129Z_student_t_slope_smoke`) is untouched.

## A. Tests

- `tests/test_hbayes_tt.py`: 9 tests, all pass.
  - GG reproduces `hbayes` draw for draw, and TG reproduces `hbayes_t` draw for draw. All parameters and random effects match to rtol 1e-9.
  - The ω conditional and scale mixture are correct.
  - GG / TG predictive moments equal the existing code.
  - GT down-weights an outlying history visit.
  - GT inner Gibbs agrees with importance sampling over ω.
  - On an identified synthetic design, TT attributes tails to the correct source in all three cases: slope only, residual only, and both.
- Full suite: 486 passed.

## B. Real data: script 04 primary training split

573 patients, 2,426 observations. Burn-in 2,000, thin 5, 4,000 draws, 4 chains (`real_sampler_diagnostics.csv`).

| config | β1 | σ (scale) | τ1 (scale) | νs median [95 % CrI] | νe median [95 % CrI] | worst R̂ | νs ESS (4 ch / 1 ch) |
|---|---|---|---|---|---|---|---|
| GG | −0.093 | 1.137 | 0.105 | — | — | 1.003 | — |
| TG | −0.102 | 1.117 | 0.062 | 1.88 [1.33, 3.09] | — | 1.005 | 780 / 200 |
| GT | −0.082 | 0.605 | 0.112 | — | 2.23 [1.93, 2.61] | 1.003 | — |
| TT | −0.087 | 0.593 | 0.076 | 2.04 [1.39, 3.89] | 2.23 [1.90, 2.61] | 1.017 (νs) | 265 / 45 |
| TT, log-uniform ν prior | −0.087 | 0.593 | 0.075 | 1.99 [1.39, 3.32] | 2.20 [1.90, 2.61] | 1.008 | 442 / 77 |

- Posterior correlation of log νs and log νe in TT: −0.049 (Gamma prior), −0.013 (log-uniform prior).
- TT νs autocorrelation at lags 1, 5 and 20: 0.955, 0.832 and 0.545. For νe: 0.667, 0.236 and 0.00.
- TT νs mixes poorly at these settings, so the full run needs thin 10 (spec §5).

## C. Recovery on the real visit design (`recovery.csv`)

- **Design:** the primary split's real visit times. Each replicate is fitted on the 573 training patients and scored on the test patients' 854 transitions.
- **Truth values:** taken from the real GG fit (β, τ0, τ1, σ). Where a truth has t slopes or t residuals, τ1 and σ use the real TT scale estimates.
- **Replicates:** 6 per scenario; all four configurations fitted to each.

### νs and νe posteriors (median over replicates)

| truth | TG νs med | TT νs med | TT νs 97.5 % | **TT νs 97.5 % < 10** | TT νe med | TT νe 97.5 % < 10 | TT β1 covered |
|---|---|---|---|---|---|---|---|
| GG | 13.6 | 13.6 | 42.7 | 0/6 | 27.4 | 0/6 | 5/6 |
| νs = 2, Gaussian residuals | 2.4 | 2.4 | 3.9 | 5/6 | 29.9 | 0/6 | 6/6 |
| νs = 3, Gaussian residuals | 8.6 | 9.2 | 37.3 | 0/6 | 28.0 | 0/6 | 6/6 |
| Gaussian slopes, νe = 4 | 6.1 | 13.9 | 47.8 | 0/6 | 4.0 | 6/6 | 6/6 |
| νs = 3, νe = 4 | 2.5 | 3.6 | 10.5 | 2/6 | 4.1 | 6/6 | 6/6 |
| Gaussian slopes, per-patient noise scale | 6.7 | 11.0 | 43.2 | 0/6 | 3.4 | 6/6 | 6/6 |
| **Gaussian slopes, νe = 2.2 (real level)** | **1.8** | 14.7 | 48.5 | **0/6** | 2.1 | 6/6 | 6/6 |
| **νs = 2, νe = 2.2 (real level)** | 1.6 | 2.2 | 3.5 | **6/6** | 2.2 | 6/6 | 6/6 |

Within-TT corr(log νs, log νe) has a median between −0.06 and 0.07 in every scenario.

What this shows:

- **TT's νs is a usable signal only in the extreme regime.** It flags νs ≈ 2 in 11/12 replicates and νs = 3 in 2/12. It raises no false flag in 0/24 Gaussian-slope replicates, including those with real-level heavy noise and per-patient noise heterogeneity.
- **TG is confounded.** With Gaussian slopes and real-level noise, TG reports νs below 5 in 5/6 replicates, with a median of 1.8.
- **Residual tails are cleanly identified** in every scenario.

### Held-out contrasts: replicates where the patient-bootstrap 95 % CI excludes 0

| truth | GT−GG cls / LPD | TT−TG cls / LPD | TG−GG LPD | TT−GT cls / LPD |
|---|---|---|---|---|
| GG | 0 / 0 | 0 / 0 (TG favoured 0/1) | 0 | 0 / 0 (GT favoured 0/1) |
| νs = 2, Gaussian residuals | 1 / 1 (GG favoured 1/2) | 0 / 0 (TG favoured 1/2) | 1 | 0 / 2 |
| νs = 3, Gaussian residuals | 0 / 0 | 1 / 0 (TG favoured 1/1) | 0 | 0 / 0 |
| Gaussian slopes, νe = 4 | 2 / 6 | 2 / 6 | 1 | 0 / 1 |
| νs = 3, νe = 4 | 3 / 6 | 4 / 6 | 4 | 0 / 2 |
| Gaussian slopes, per-patient noise | 2 / 6 | 2 / 6 | 1 | 0 / 1 |
| Gaussian slopes, νe = 2.2 | 6 / 6 | 6 / 6 | 2 | 0 / **2** |
| νs = 2, νe = 2.2 | 6 / 6 | 6 / 6 | 5 | 0 / **0** |

Counts are replicates favouring the first-named model; "(X favoured a/b)" gives class / LPD counts favouring the second-named model.

- Prediction detects residual tails: LPD 30/30 in heavy-noise scenarios.
- Prediction cannot identify slope tails. TT−GT on the class metric is 0/48. TT−GT on LPD is positive in 2/6 replicates where the slopes are Gaussian (GT2.2), and in 0/6 where they are heavy-tailed (T2T2.2).
- One-step-ahead predictions depend mostly on the patient's recent level, so the slope distribution's tails barely affect them.

## D. Identifiability limitations

1. The data can separate "extremely heavy slope tails (νs ≈ 2)" from "Gaussian slopes". They cannot separate moderate tails (νs = 3–6) from Gaussian slopes.
2. Only TT's νs is protected against heavy-tailed noise; TG's νs is not. In these scenarios TT's νs was also protected against per-patient noise heterogeneity, but other misspecifications have not been tested: nonlinear trajectories, skewed noise, and residual outliers clustered within patients. Full-run calibration (spec §4 Q2(c)) is required before any slope-tail conclusion.
3. Held-out one-step-ahead prediction, the script-04 evaluation, has essentially no power for the slope-tail question. It is informative for residual tails and for practical predictive value.
4. On the real training split, TT gives νs ≈ 2.0 [1.39, 3.89] alongside νe ≈ 2.2. That is the region in which the simulations flagged only true heavy slope tails. It is not yet a finding: it rests on one split, νs mixing is poor (R̂ 1.017), and the calibration in (2) has not been run.
