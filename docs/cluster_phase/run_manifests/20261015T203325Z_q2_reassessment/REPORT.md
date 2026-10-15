# Q2 reassessment: patient-slope tails (collapsed slice sampler)

- **Run directory:** `experiments/20261015T203325Z_q2_reassessment/` (all outputs under `results/`)
- **Job:** SLURM job 13448, partition `normal`, node002 (exclusive; CPU only, the GPUs were unused); 32 worker processes
- **Started / finished:** 2026-10-16 02:03:26 / 02:54:18 IST (3,044.7 s); exit 0; empty stderr
- **Pre-registration:** frozen design `docs/cluster_phase/09_Q2_REASSESSMENT_SPEC.md`. `results/reassessment_config.json` (SHA-256 of the spec, the preserved frozen spec 08, and all code used) was written at 02:03:34, before any sampling. The hashes equal those computed when the design was frozen.
- **Provenance:** the job also archived `git_diff.patch`, the untracked source (`untracked_source.tar.gz`) and `pip_freeze.txt`. Git HEAD `7a59dba`; the Student-t work is uncommitted.

**Scope.** Q2 only. Q1 (residual heavy tails: SUPPORTED) and Q3 (predictive value), both from the original full run, are unchanged. The original frozen Q2 verdict, **INCONCLUSIVE (convergence)**, stays on record in `experiments/20261015T190228Z_student_t_four_model_full/`, which this run read but did not modify.

## Verdict against each pre-specified criterion

| criterion (frozen) | result | met |
|---|---|---|
| **(conv)** full cohort, νs R̂ < 1.01 and ESS ≥ 400 under each required prior | true Gamma(2, 0.1): R̂ 1.0044, ESS 937. Log-uniform: R̂ 1.0005, ESS 4,394. | **yes** |
| **(a)** full cohort, νs 97.5 % < 10 under both required priors | Gamma(2, 0.1): 2.09 [1.48, **4.11**]. Log-uniform: 1.94 [1.42, **2.99**]. | **yes** |
| **(b)** per-split, νs 97.5 % < 10 in ≥ 5/7 splits under the primary prior (true Gamma(2, 0.1)) | **1/7**. The 97.5 % quantiles are 16.3, 29.6, 11.2, 37.0, **3.27**, 28.7 and 20.2. | **no** |
| **(c)** calibration flags ≤ 3/60 in each of C1, C2 and C3 | 0/60, 0/60, 0/60 | **yes** |

**Q2 reassessment verdict (frozen rule): NOT SUPPORTED, no evidence of extreme slope tails.**

What this does and does not mean:

- The frozen definition of "extreme" (97.5 % quantile < 10) is not robustly established across training subsets.
- It does **not** mean the slopes are Gaussian. The design cannot distinguish moderate tails (νs ≈ 3–6) from Gaussian slopes (spec 08 §6).
- No conclusion about discrete patient subtypes follows from this result, or would have followed from the opposite one. A heavy-tailed but continuous slope distribution is not a set of groups.

## A. Full cohort

821 patients, 3,528 observations. TT model, collapsed slice sampler, 4 chains, 4,000 draws, burn-in 2,000, thin 10.

| prior | νs median [95 % CrI] | R̂ | ESS | 97.5 % MC interval | νe | β1 | τ1 (scale) | σ (scale) |
|---|---|---|---|---|---|---|---|---|
| **Gamma(2, 0.1), primary** | 2.09 [1.48, 4.11] | 1.0044 | 937 | [3.63, 5.61] | 2.10 [1.88, 2.39] | −0.082 | 0.074 | 0.578 |
| **log-uniform, required** | 1.94 [1.42, 2.99] | 1.0005 | 4,394 | [2.92, 3.07] | 2.10 | −0.082 | 0.070 | 0.575 |
| Exponential-like, continuity only | 2.00 [1.46, 3.24] | 1.0013 | 2,689 | [3.14, 3.36] | 2.10 | −0.082 | 0.072 | 0.576 |

- All other parameters have R̂ ≤ 1.0025 under all three priors.
- Under the primary prior, the per-chain medians are 2.08–2.11 and the per-chain 97.5 % quantiles are 3.5–7.2. Seed 42 has ESS 186, while the pooled ESS is 937.
- Under the continuity prior the posterior agrees with the original frozen run (median 1.99 [1.45, 3.47]). The original verdict's centre is confirmed; its convergence failure was a sampler issue.
- **Prior sensitivity:** the median is stable (1.94–2.09). The upper tail is prior-sensitive: the 97.5 % quantile is 2.99 under log-uniform and 4.11 under the true Gamma(2, 0.1).

## B. Per-split fits

Script 04's seven training splits, 570–579 patients each. One chain per split, as frozen.

| split | Gamma(2, 0.1): median, 97.5 % [MC interval], ESS | log-uniform 97.5 % | continuity 97.5 % |
|---|---|---|---|
| 42 | 2.29, **16.3** [10.9, 26.5], 220 | 3.60 | 6.10 |
| 0 | 2.60, **29.6** [26.2, 36.7], 105 | 15.85 | 8.37 |
| 1 | 2.26, **11.2** [7.3, 21.0], 370 | 3.73 | 4.25 |
| 2 | 3.33, **37.0** [33.1, 42.3], 205 | 12.56 | 13.18 |
| 3 | 1.88, **3.27** [2.90, 4.76], 244 | 2.76 | 2.84 |
| 4 | 2.43, **28.7** [23.8, 38.4], 117 | 7.17 | 4.34 |
| 5 | 2.26, **20.2** [15.6, 32.5], 136 | 5.03 | 5.40 |
| **splits with 97.5 % < 10** | **1/7** (frozen criterion) | 5/7 (descriptive) | 6/7 (descriptive) |

How to read the per-split results:

- Every split's median is between 1.9 and 3.3, so the bulk of each posterior sits in the very heavy-tailed region.
- With 70 % of the patients, the data no longer exclude larger νs, so the upper tail is driven by the prior. Under the true Gamma(2, 0.1) prior, which puts 74 % of its mass above 10, six of seven splits have 97.5 % quantiles between 11 and 37. Under the flatter priors most splits stay below 10.
- Only split 1's Monte-Carlo interval spans 10. Even counting it as a pass, the criterion would be 2/7, so the failure of (b) is **not** a Monte-Carlo artefact.
- These single-chain per-split fits have νs ESS of 105–370. The frozen rule does not require a convergence check for (b), but the per-split tail quantiles are noisier than the full-cohort ones.

## C. Calibration

Full-cohort design, Gaussian slopes, 60 replicates per scenario, TT fitted with 4 chains × 2 required priors.

| scenario | flags | observed rate | exact 95 % CI (two-sided) | one-sided 95 % upper bound | borderline | max R̂ / min ESS (νs) | Gamma-prior νs 97.5 %: smallest / median |
|---|---|---|---|---|---|---|---|
| C1: i.i.d. t_2.10 residuals | **0/60** | 0 % | [0 %, 5.96 %] | 4.87 % | 0 | 1.001 / 5,730 | 47.0 / 59.4 |
| C2: real residuals resampled as whole-patient blocks | **0/60** | 0 % | [0 %, 5.96 %] | 4.87 % | 0 | 1.001 / 2,826 | 49.8 / 58.8 |
| C3: piecewise-linear trajectories | **0/60** | 0 % | [0 %, 5.96 %] | 4.87 % | 0 | 1.001 / 4,323 | 44.0 / 59.8 |
| P: t₂ slopes (positive control, descriptive) | 55/60 flagged | 91.7 % | [81.6 %, 97.2 %] | — | 3 | 1.188 / 25 | 2.32 / 3.89 |

**False positives.** None in 180 Gaussian-slope replicates, and none borderline. All their chains converged (R̂ ≤ 1.0013, ESS ≥ 2,826).

- For each scenario, the two-sided exact 95 % interval [0 %, 5.96 %] still includes rates slightly above 5 %.
- The one-sided 95 % upper bound of 4.87 % is the only basis on which "true false-positive rate < 5 %" can be stated, and it holds per scenario at that one-sided level.
- With 60 replicates, precision remains limited.

**Positive control.** It was missed in 5/60 replicates: reps 2, 16, 34, 37 and 55, with Gamma-prior 97.5 % quantiles of 12.9–36.6. Three replicates were borderline: reps 7, 37 and 55.

- 4/60 positive-control replicates under the Gamma prior and 3/60 under log-uniform had R̂ ≥ 1.01 and ESS < 400 (max R̂ 1.188).
- With truly very heavy tails, convergence is harder.
- These replicates are descriptive and do not enter the rule.

## Deviations and failures

- **Execution host.** The run used a SLURM compute node with 32 workers, not the login node with 16 as in the smoke check. Results do not depend on the worker count, because every task seeds its own generator. The design, priors, seeds, thresholds and rule are exactly as frozen. Nothing was changed after seeing results.
- **Calibration truth.** As frozen, C1, C3 and P use the original run's `calibration_theta` verbatim. C2's residuals come from this run's full-cohort TT fit (Gamma(2, 0.1), chain 42), so C2 replicates differ from the original run's C2 even where the seeds coincide.
- **No computational failures.** No convergence failures in A or C1–C3. The positive-control convergence failures are listed above.

## Conclusions (kept separate)

1. **Residual (visit-level) heavy tails: SUPPORTED.** From the original run; unchanged.
2. **Predictive value:** Student-t residuals improve held-out prediction. From the original run; unchanged.
3. **Patient-slope heavy tails beyond residual tails: NOT SUPPORTED by the pre-specified criteria in this reassessment.**
   - The full-cohort posterior converges and sits at νs ≈ 2 with 97.5 % quantiles of 3–4. The calibration shows that this full-cohort signal is not produced by heavy-tailed noise, clustered residuals or one form of nonlinearity, at the stated precision.
   - However, the frozen per-split replication criterion fails under the primary prior (1/7). The upper tail of νs is not identified robustly from 70 % training subsets and is prior-sensitive.
   - Overall: suggestive, but not established by this design.
   - The original verdict, INCONCLUSIVE (convergence), stands as the record of the original analysis.
4. **No inference about discrete patient subtypes** is drawn from any of these results.

## Files (`results/`)

| file | contents |
|---|---|
| `reassessment_config.json` | frozen configuration and hashes, written before sampling |
| `results.json` | verdict and every criterion input |
| `A_full_cohort.csv`, `A_full_cohort_nu_traces.npz` | full-cohort fits |
| `B_per_split.csv` | per-split fits |
| `C_calibration.csv` | 240 calibration replicates |

Job files in the run root: `job_header.txt`, `slurm-13448.out` / `.err`, `git_diff.patch`, `untracked_source.tar.gz`, `pip_freeze.txt`.
