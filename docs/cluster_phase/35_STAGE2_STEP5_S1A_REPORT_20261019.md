# Stage 2 M2, Step 5 (S1-a): execution report

**Date:** 2026-10-19.

**Status: diagnostic only, reported only.** S1-a has no acceptance criterion (Rev 2 §5, R4). It authorises nothing, selects nothing, and no parameter, setting or S1-b option is chosen from it.

**Plan:** `34_STAGE2_STEP5_S1A_PLAN_FINAL_20261019.md`, frozen as `9b514b25…` before any computation. It records decisions D-1 to D-5. The proposed doc 33 (`325471e5…`) is unchanged.

**Execution directory:** `experiments/20261018T215102Z_stage2_step5_s1a/`. The directory name is UTC; the run was on 2026-10-19 local time (IST).

**Jobs:**
- 13907: smoke code path, not a result;
- 13908: the Step 5 run.

## 1. Summary

**Integrity.** Every pre-fit check passed:
- frozen-record, code, panel and input hashes;
- the 3 unit tests;
- truth equal to the pilot record;
- the dataset identity check, exact (57022.74225348176 = recorded);
- pilot reproduction of the N(0, 1) item-fit records, exactly (30/30 metrics bitwise equal, D-5).

**Item fits.** All 20 converged (L-BFGS-B relative-reduction criterion). There were no non-finite values.

**Stage-2 evaluations.** There were 205 at the Step 3 primary configuration (Q = 15): 5 datasets × (3 at ψ_true + 38 finite-difference points).
- No patient needed the G = 5 fallback: **0 invocations**.
- There were no P1 budget failures and no non-finite chunks.
- All 5 Hessians at ψ_true are PD.
- The gradient-capable reported evaluator agrees with `evaluate_reported` to ≤ 1.5e-11 and with `Objective` exactly.

**Runtime:** 805 s for the run plus 113 s for the smoke check, on node002 (1 exclusive node, 64 CPUs). That is **0.255 node-hours** (AllocNodes × ElapsedRaw / 3600).

**Descriptive findings** (5 datasets; magnitudes and directions only):

1. **The N(0, 1) working assumption mainly affects the PIGD items.** Relative to the oracle-latent fit, the N(0, 1) fit inflates the discriminations, and the inflation grows with a_true.
   - The mean (N01 − oracle) a difference is +0.04 to +0.47 for the five P items (pooled mean +0.21) and +0.00 to +0.08 for the eleven T items (pooled +0.03).
   - The upper thresholds are pulled down: τ4 differences reach −0.38 (P) and −0.13 (T).
   - These differences are very stable across datasets (per-item SD 0.006–0.071).
   - Under the oracle latent distribution, the mean P discrimination error is near zero (+0.007, against +0.221 for N(0, 1)). The oracle's remaining item error is of the same order as the dataset-to-dataset sampling spread.
2. **First-order effect on ψ at ψ_true, (Δψ_N01 − Δψ_oracle)/SE**, consistent in sign in all 5 datasets for the larger components:

   | ψ component | mean shift (SE) | range over the 5 datasets |
   |---|---|---|
   | β0_P | **+2.16** | +1.98 to +2.27 |
   | Le11_log (PIGD residual SD) | **−1.78** | −2.21 to −1.49 |
   | L33_log | −1.08 | |
   | β1_P | −1.02 | |
   | L22_log | −0.59 | |
   | Le00_log | −0.50 | |
   | L32 | −0.47 | |
   | β0_T | +0.27 | |
   | all other components | \|mean\| ≤ 0.24 | |

   By contrast, the item sampling error under the oracle, (Δψ_oracle − Δψ_true)/SE, has means within ±0.06 SE for 13 of the 19 components. The other six are β0_T −0.43, L00_log −0.52, Le00_log −0.76, L11_log −0.16, β1_T −0.09 and γ_T −0.07. It varies in sign across datasets.
3. **Adequacy of the first-order approximation in its own reference case.** With the true items, Δψ_true reproduces the actual Step 4 deviation ψ̂ − ψ_true to within 0.26 SE for every component and dataset (T7a vs T7b).

These are first-order indications at ψ_true under the a-priori truth (limitations in §4). They are **not** recovery results.

## 2. Verification before fitting

| check (plan §9) | result |
|---|---|
| frozen locks (docs 16, 21, 25, 27, 29, 31, 34) | all OK |
| code against the Step 4 baseline and the pilot provenance (`m2_quad.py` `50f164c6…`, `m2_grm.py` `585c878b…`, `grm.py` `1584bc62…`, `45_m2_pilot.py` `a25e1503…`, …) | all unchanged |
| panel MD5 | `fb038ddb…` |
| inputs (pilot `step1_recovery.json`, Step 4 `summary.json` `00433150…` and `S1_dataset{0..4}.json`, identity reference) | unchanged |
| unit tests, `tests/test_s1a_latent_items.py` | 3 passed (in the job, before any fit) |
| ψ_true from `truth_on_stage1_scale` against the pilot record | equal |
| dataset identity (serial `MQ.negloglik`, dataset 0, true items, Q = 10, recorded ψ) | 57022.74225348176 = 57022.74225348176 |
| pilot N(0, 1) item-fit records, `items[r]` (6 metrics × 5 datasets), D-5 | **30/30 exactly equal** |
| oracle construction: pooled visit mean / variance of N(m_v, s_v²) per subscore | T: −0.000 / 1.000; P: 0.000 / 1.000 (as `truth_on_stage1_scale` implies). s_v ranges 0.903–1.458 (T) and 0.765–1.618 (P); m_v ranges −0.207 to 0.559 (T) and −0.477 to 1.388 (P) |

**Datasets.** Seeds `[42, 45, 1, r]`, r = 0…4. No other random numbers were drawn. The real design (times, treatment) was used; real responses and labels were not.

## 3. Results per dataset

**Definitions:**
- N01 is the stage-1 item fit under θ ~ N(0, 1), the pilot path.
- oracle is the item fit under θ_v ~ N(M_ij, K_ij,jj) at ψ_true (D-1).
- true is the generating items, it_true on the stage-1 scale.
- Errors are signed estimate − truth.
- ℓ is the stage-2 log-likelihood.
- In T1, "n/r" means the iteration count is not returned by the pilot path `fit_items_multistart`. Its gradient norm is taken from the §5.3a item-objective evaluation.

### T1. Item fits per dataset (all from initial_items, one start)

| dataset | fit | converged | NLL at optimum | ‖∇‖ own objective | L-BFGS iterations | wall s |
|---|---|---|---|---|---|---|
| 0 | N01 T | True | 41311.405 | 0.258 | n/r | 52.2 |
| 0 | N01 P | True | 18204.100 | 0.528 | n/r | 14.8 |
| 0 | oracle T | True | 41256.271 | 0.879 | 55 | 73.3 |
| 0 | oracle P | True | 17749.656 | 0.876 | 23 | 17.5 |
| 1 | N01 T | True | 42306.163 | 1.163 | n/r | 51.8 |
| 1 | N01 P | True | 19024.512 | 0.423 | n/r | 15.1 |
| 1 | oracle T | True | 42206.909 | 0.561 | 49 | 62.3 |
| 1 | oracle P | True | 18511.142 | 0.214 | 27 | 20.5 |
| 2 | N01 T | True | 42522.332 | 0.278 | n/r | 49.7 |
| 2 | N01 P | True | 18164.254 | 0.228 | n/r | 18.3 |
| 2 | oracle T | True | 42399.105 | 0.571 | 42 | 55.5 |
| 2 | oracle P | True | 17770.419 | 0.219 | 31 | 19.5 |
| 3 | N01 T | True | 42109.399 | 0.295 | n/r | 53.2 |
| 3 | N01 P | True | 18315.594 | 0.106 | n/r | 19.5 |
| 3 | oracle T | True | 41981.136 | 1.087 | 43 | 59.6 |
| 3 | oracle P | True | 17914.048 | 0.558 | 32 | 24.3 |
| 4 | N01 T | True | 42803.955 | 0.376 | n/r | 52.4 |
| 4 | N01 P | True | 18494.268 | 0.403 | n/r | 19.5 |
| 4 | oracle T | True | 42666.766 | 0.372 | 49 | 69.2 |
| 4 | oracle P | True | 18007.233 | 0.237 | 31 | 23.0 |

### T2. Item-parameter errors against the truth, per dataset (stage-1 scale)

| dataset | subscore | fit | a max\|err\| | a mean err | a RMSE | corr(a, a_true) | τ max\|err\| | τ mean err | τ RMSE |
|---|---|---|---|---|---|---|---|---|---|
| 0 | T | N01 | 0.134 | +0.016 | 0.065 | 0.9874 | 0.339 | +0.037 | 0.085 |
| 0 | T | oracle | 0.144 | +0.008 | 0.067 | 0.9867 | 0.379 | +0.031 | 0.091 |
| 0 | P | N01 | 0.322 | +0.146 | 0.187 | 0.9946 | 0.266 | -0.016 | 0.101 |
| 0 | P | oracle | 0.228 | -0.070 | 0.108 | 0.9876 | 0.248 | +0.065 | 0.085 |
| 1 | T | N01 | 0.246 | +0.024 | 0.095 | 0.9931 | 0.149 | -0.028 | 0.054 |
| 1 | T | oracle | 0.157 | -0.007 | 0.068 | 0.9935 | 0.181 | -0.008 | 0.052 |
| 1 | P | N01 | 0.463 | +0.273 | 0.314 | 0.9920 | 0.364 | -0.141 | 0.168 |
| 1 | P | oracle | 0.099 | +0.032 | 0.082 | 0.9858 | 0.110 | -0.052 | 0.066 |
| 2 | T | N01 | 0.282 | +0.141 | 0.160 | 0.9915 | 0.583 | -0.160 | 0.198 |
| 2 | T | oracle | 0.199 | +0.103 | 0.118 | 0.9923 | 0.482 | -0.132 | 0.162 |
| 2 | P | N01 | 0.867 | +0.335 | 0.460 | 0.9885 | 0.490 | -0.089 | 0.176 |
| 2 | P | oracle | 0.321 | +0.120 | 0.172 | 0.9968 | 0.178 | -0.044 | 0.077 |
| 3 | T | N01 | 0.135 | +0.007 | 0.057 | 0.9905 | 0.301 | -0.010 | 0.070 |
| 3 | T | oracle | 0.237 | -0.033 | 0.082 | 0.9867 | 0.498 | +0.026 | 0.098 |
| 3 | P | N01 | 0.280 | +0.082 | 0.131 | 0.9845 | 0.185 | +0.007 | 0.068 |
| 3 | P | oracle | 0.271 | -0.086 | 0.136 | 0.9748 | 0.223 | +0.062 | 0.083 |
| 4 | T | N01 | 0.194 | +0.089 | 0.111 | 0.9890 | 0.254 | -0.119 | 0.140 |
| 4 | T | oracle | 0.119 | +0.047 | 0.080 | 0.9869 | 0.197 | -0.082 | 0.105 |
| 4 | P | N01 | 0.620 | +0.267 | 0.355 | 0.9827 | 0.534 | -0.102 | 0.181 |
| 4 | P | oracle | 0.277 | +0.038 | 0.138 | 0.9683 | 0.178 | -0.023 | 0.058 |

### T3. Per-item a errors, mean (SD) over the 5 datasets

| subscore | item | a_true | N01 − true | oracle − true | N01 − oracle |
|---|---|---|---|---|---|
| P | 0 | 1.850 | +0.039 (0.060) | +0.000 (0.060) | +0.038 (0.006) |
| P | 1 | 2.159 | +0.053 (0.023) | -0.026 (0.027) | +0.079 (0.024) |
| P | 2 | 2.467 | +0.168 (0.116) | +0.006 (0.104) | +0.162 (0.037) |
| P | 3 | 2.776 | +0.441 (0.140) | +0.117 (0.120) | +0.324 (0.046) |
| P | 4 | 3.084 | +0.402 (0.295) | -0.062 (0.234) | +0.465 (0.071) |
| T | 0 | 1.254 | +0.008 (0.061) | +0.005 (0.061) | +0.003 (0.006) |
| T | 1 | 1.380 | +0.055 (0.034) | +0.051 (0.030) | +0.004 (0.009) |
| T | 2 | 1.505 | +0.052 (0.068) | +0.044 (0.064) | +0.009 (0.008) |
| T | 3 | 1.631 | +0.033 (0.085) | +0.019 (0.079) | +0.014 (0.013) |
| T | 4 | 1.756 | +0.033 (0.053) | +0.016 (0.051) | +0.017 (0.012) |
| T | 5 | 1.882 | +0.033 (0.077) | +0.010 (0.071) | +0.024 (0.013) |
| T | 6 | 2.007 | +0.073 (0.079) | +0.037 (0.078) | +0.036 (0.012) |
| T | 7 | 2.132 | +0.018 (0.095) | -0.021 (0.081) | +0.039 (0.017) |
| T | 8 | 2.258 | +0.136 (0.081) | +0.081 (0.080) | +0.054 (0.021) |
| T | 9 | 2.383 | +0.113 (0.119) | +0.047 (0.104) | +0.066 (0.023) |
| T | 10 | 2.509 | +0.058 (0.163) | -0.024 (0.162) | +0.082 (0.024) |

### T4. Per-item τ differences N01 − oracle, mean over the 5 datasets

| subscore | item | τ1 | τ2 | τ3 | τ4 |
|---|---|---|---|---|---|
| P | 0 | +0.029 | +0.026 | +0.011 | -0.017 |
| P | 1 | +0.035 | +0.016 | -0.018 | -0.064 |
| P | 2 | +0.029 | -0.011 | -0.067 | -0.146 |
| P | 3 | +0.004 | -0.058 | -0.139 | -0.251 |
| P | 4 | -0.042 | -0.120 | -0.234 | -0.375 |
| T | 0 | +0.004 | +0.003 | +0.001 | -0.002 |
| T | 1 | +0.004 | +0.003 | +0.000 | -0.004 |
| T | 2 | +0.005 | +0.002 | -0.003 | -0.010 |
| T | 3 | +0.005 | +0.000 | -0.007 | -0.017 |
| T | 4 | +0.004 | -0.002 | -0.012 | -0.025 |
| T | 5 | +0.003 | -0.007 | -0.019 | -0.035 |
| T | 6 | -0.000 | -0.013 | -0.029 | -0.050 |
| T | 7 | -0.003 | -0.018 | -0.037 | -0.061 |
| T | 8 | -0.008 | -0.027 | -0.052 | -0.090 |
| T | 9 | -0.014 | -0.036 | -0.065 | -0.101 |
| T | 10 | -0.024 | -0.051 | -0.088 | -0.133 |

### T5. Item objectives (NLL) at each item set, per dataset

| dataset | subscore | item set | N(0,1) NLL | ‖∇‖ | oracle NLL | ‖∇‖ |
|---|---|---|---|---|---|---|
| 0 | T | true | 41350.827 | 306.305 | 41295.779 | 308.822 |
| 0 | T | N01 | 41311.405 | 0.258 | 41256.965 | 23.448 |
| 0 | T | oracle | 41312.075 | 22.383 | 41256.271 | 0.879 |
| 0 | P | true | 18233.009 | 225.715 | 17767.164 | 171.564 |
| 0 | P | N01 | 18204.100 | 0.528 | 17768.456 | 110.534 |
| 0 | P | oracle | 18222.987 | 98.961 | 17749.656 | 0.876 |
| 1 | T | true | 42329.539 | 247.891 | 42229.958 | 248.354 |
| 1 | T | N01 | 42306.163 | 1.163 | 42208.054 | 22.774 |
| 1 | T | oracle | 42307.302 | 21.969 | 42206.909 | 0.561 |
| 1 | P | true | 19051.488 | 182.885 | 18528.828 | 203.103 |
| 1 | P | N01 | 19024.512 | 0.423 | 18531.789 | 109.040 |
| 1 | P | oracle | 19045.889 | 105.067 | 18511.142 | 0.214 |
| 2 | T | true | 42565.036 | 228.444 | 42435.584 | 229.090 |
| 2 | T | N01 | 42522.332 | 0.278 | 42400.314 | 19.442 |
| 2 | T | oracle | 42523.594 | 19.793 | 42399.105 | 0.571 |
| 2 | P | true | 18216.034 | 253.224 | 17788.164 | 234.466 |
| 2 | P | N01 | 18164.254 | 0.228 | 17787.020 | 119.656 |
| 2 | P | oracle | 18180.381 | 103.749 | 17770.419 | 0.219 |
| 3 | T | true | 42135.315 | 278.372 | 42008.110 | 275.365 |
| 3 | T | N01 | 42109.399 | 0.295 | 41982.435 | 19.379 |
| 3 | T | oracle | 42110.756 | 19.577 | 41981.136 | 1.087 |
| 3 | P | true | 18330.699 | 140.841 | 17922.636 | 116.700 |
| 3 | P | N01 | 18315.594 | 0.106 | 17927.747 | 106.937 |
| 3 | P | oracle | 18328.726 | 92.738 | 17914.048 | 0.558 |
| 4 | T | true | 42846.155 | 327.858 | 42705.504 | 329.598 |
| 4 | T | N01 | 42803.955 | 0.376 | 42668.120 | 20.281 |
| 4 | T | oracle | 42805.338 | 20.609 | 42666.766 | 0.372 |
| 4 | P | true | 18546.165 | 242.736 | 18026.126 | 194.062 |
| 4 | P | N01 | 18494.268 | 0.403 | 18024.928 | 100.497 |
| 4 | P | oracle | 18512.618 | 94.629 | 18007.233 | 0.237 |

### T6. Stage-2 at ψ_true (Laplace Q = 15, reported context), per dataset

| dataset | ℓ true items | ℓ N01 items | ℓ oracle items | ‖g‖ true / N01 / oracle | fallback patients (3 evals + 38 FD) | failed evals | H PD | H min eig | H max eig | evaluator check (Δ vs evaluate_reported; vs Objective) | dataset wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -57028.857 | -56995.595 | -56979.016 | 376.3 / 403.5 / 319.6 | 0 | 0 | True | 123.6 | 32425 | 0.0e+00; 0.0e+00 | 131 |
| 1 | -58732.220 | -58724.056 | -58701.285 | 271.9 / 425.6 / 235.5 | 0 | 0 | True | 145.3 | 33952 | 0.0e+00; 0.0e+00 | 130 |
| 2 | -58044.198 | -58033.821 | -58007.691 | 251.2 / 414.9 / 266.5 | 0 | 0 | True | 105.2 | 38662 | 7.3e-12; 0.0e+00 | 140 |
| 3 | -58009.445 | -57986.790 | -57978.879 | 375.8 / 380.6 / 377.0 | 0 | 0 | True | 103.3 | 34645 | 1.5e-11; 0.0e+00 | 130 |
| 4 | -58630.203 | -58606.579 | -58583.771 | 567.7 / 551.3 / 458.5 | 0 | 0 | True | 106.8 | 35716 | 1.5e-11; 0.0e+00 | 129 |

### T7a. Step 4 actual deviation (ψ̂_Step4 − ψ_true)/SE_Step4 — reference for T7b

| ψ component | ds 0 | ds 1 | ds 2 | ds 3 | ds 4 | mean |
|---|---|---|---|---|---|---|
| beta0_T | +0.46 | +0.18 | +0.74 | -0.81 | -0.55 | +0.00 |
| beta1_T | -1.68 | +0.10 | +0.28 | +0.28 | +2.24 | +0.24 |
| gamma_T | +0.28 | +1.53 | -0.77 | +0.69 | +0.45 | +0.44 |
| beta0_P | -0.39 | +1.10 | +0.00 | -0.84 | -0.28 | -0.08 |
| beta1_P | -0.06 | +0.10 | +1.68 | -0.32 | +1.16 | +0.51 |
| gamma_P | -0.62 | -0.20 | -2.54 | +1.28 | -1.11 | -0.64 |
| L00_log | +0.63 | -0.40 | +0.80 | -1.25 | +2.05 | +0.37 |
| L10 | +0.31 | -0.84 | +0.47 | +2.51 | -0.93 | +0.30 |
| L11_log | +0.05 | +0.18 | +0.10 | -1.65 | +0.28 | -0.21 |
| L20 | +1.13 | +0.10 | -0.05 | -0.22 | +1.34 | +0.46 |
| L21 | -0.96 | -1.58 | +0.05 | -0.22 | +2.01 | -0.14 |
| L22_log | -0.46 | -0.37 | +1.23 | +0.10 | -0.46 | +0.01 |
| L30 | -0.08 | -0.04 | -0.31 | +0.87 | -1.28 | -0.17 |
| L31 | -0.21 | +0.57 | -0.86 | -1.26 | -0.29 | -0.41 |
| L32 | -0.63 | +0.20 | -0.29 | -1.37 | -0.24 | -0.47 |
| L33_log | +0.73 | -0.51 | +0.17 | +0.17 | -0.42 | +0.03 |
| Le00_log | -0.61 | -1.41 | +0.68 | +1.29 | -1.11 | -0.23 |
| Le10 | +1.83 | +0.06 | -0.50 | +0.30 | -0.14 | +0.31 |
| Le11_log | -0.69 | +0.10 | -0.36 | +0.12 | -0.63 | -0.29 |

### T7b. First-order shift, true items: Δψ_true / SE_Step4

| ψ component | ds 0 | ds 1 | ds 2 | ds 3 | ds 4 | mean |
|---|---|---|---|---|---|---|
| beta0_T | +0.51 | +0.21 | +0.71 | -0.90 | -0.48 | +0.01 |
| beta1_T | -1.72 | +0.18 | +0.32 | +0.41 | +2.37 | +0.31 |
| gamma_T | +0.31 | +1.60 | -0.79 | +0.63 | +0.47 | +0.44 |
| beta0_P | -0.44 | +1.21 | +0.06 | -0.83 | -0.42 | -0.08 |
| beta1_P | -0.06 | +0.06 | +1.68 | -0.26 | +1.29 | +0.54 |
| gamma_P | -0.63 | -0.21 | -2.55 | +1.26 | -1.20 | -0.67 |
| L00_log | +0.60 | -0.40 | +0.78 | -1.28 | +2.11 | +0.36 |
| L10 | +0.33 | -1.00 | +0.39 | +2.64 | -1.19 | +0.24 |
| L11_log | -0.09 | +0.18 | +0.09 | -1.38 | +0.14 | -0.21 |
| L20 | +1.09 | +0.10 | -0.02 | -0.30 | +1.26 | +0.43 |
| L21 | -0.91 | -1.84 | +0.04 | -0.03 | +2.14 | -0.12 |
| L22_log | -0.43 | -0.33 | +1.23 | +0.08 | -0.41 | +0.03 |
| L30 | -0.04 | -0.03 | -0.28 | +0.96 | -1.23 | -0.12 |
| L31 | -0.24 | +0.65 | -0.83 | -1.13 | -0.47 | -0.40 |
| L32 | -0.61 | +0.20 | -0.31 | -1.33 | -0.14 | -0.44 |
| L33_log | +0.76 | -0.50 | +0.13 | +0.24 | -0.53 | +0.02 |
| Le00_log | -0.63 | -1.46 | +0.66 | +1.16 | -1.21 | -0.30 |
| Le10 | +1.85 | +0.04 | -0.53 | +0.30 | -0.12 | +0.31 |
| Le11_log | -0.71 | +0.09 | -0.45 | +0.10 | -0.65 | -0.32 |

### T7c. First-order shift, N(0,1) items: Δψ_N01 / SE_Step4

| ψ component | ds 0 | ds 1 | ds 2 | ds 3 | ds 4 | mean |
|---|---|---|---|---|---|---|
| beta0_T | +2.05 | -0.54 | -0.00 | -0.81 | -1.46 | -0.15 |
| beta1_T | -1.84 | +0.16 | -0.13 | +0.29 | +2.19 | +0.14 |
| gamma_T | +0.23 | +1.62 | -1.01 | +0.60 | +0.34 | +0.36 |
| beta0_P | +2.45 | +1.94 | +3.43 | +1.42 | +1.43 | +2.13 |
| beta1_P | -0.90 | -1.19 | +0.22 | -0.73 | +0.20 | -0.48 |
| gamma_P | -0.58 | -0.45 | -2.56 | +1.36 | -1.06 | -0.66 |
| L00_log | -0.08 | +0.01 | -0.94 | -1.80 | +0.82 | -0.40 |
| L10 | +0.42 | -1.12 | +0.31 | +2.67 | -1.21 | +0.21 |
| L11_log | -0.32 | +0.28 | -0.60 | -1.46 | -0.45 | -0.51 |
| L20 | +1.19 | -0.05 | -0.25 | -0.33 | +1.18 | +0.35 |
| L21 | -0.97 | -1.92 | -0.11 | -0.05 | +2.11 | -0.19 |
| L22_log | -0.77 | -1.63 | +0.46 | -0.02 | -0.59 | -0.51 |
| L30 | -0.11 | -0.03 | -0.26 | +0.98 | -1.28 | -0.14 |
| L31 | -0.20 | +0.73 | -0.91 | -1.19 | -0.47 | -0.41 |
| L32 | -0.98 | -0.09 | -1.12 | -1.55 | -0.98 | -0.95 |
| L33_log | -0.13 | -1.62 | -1.48 | -0.28 | -1.78 | -1.06 |
| Le00_log | -1.50 | -1.21 | -2.21 | +0.36 | -3.24 | -1.56 |
| Le10 | +1.87 | -0.13 | -0.78 | +0.28 | -0.29 | +0.19 |
| Le11_log | -2.10 | -2.17 | -3.14 | -0.66 | -2.39 | -2.09 |

### T7d. First-order shift, oracle items: Δψ_oracle / SE_Step4

| ψ component | ds 0 | ds 1 | ds 2 | ds 3 | ds 4 | mean |
|---|---|---|---|---|---|---|
| beta0_T | +1.58 | -0.92 | -0.18 | -1.04 | -1.57 | -0.42 |
| beta1_T | -1.81 | +0.25 | -0.02 | +0.43 | +2.27 | +0.22 |
| gamma_T | +0.21 | +1.61 | -0.97 | +0.62 | +0.38 | +0.37 |
| beta0_P | +0.22 | -0.33 | +1.24 | -0.70 | -0.54 | -0.02 |
| beta1_P | +0.23 | +0.01 | +1.09 | +0.16 | +1.21 | +0.54 |
| gamma_P | -0.59 | -0.40 | -2.57 | +1.28 | -0.98 | -0.65 |
| L00_log | -0.14 | +0.21 | -0.64 | -1.42 | +1.20 | -0.16 |
| L10 | +0.46 | -1.03 | +0.37 | +2.69 | -1.15 | +0.27 |
| L11_log | -0.31 | +0.42 | -0.45 | -1.29 | -0.23 | -0.37 |
| L20 | +1.14 | +0.08 | -0.11 | -0.26 | +1.29 | +0.43 |
| L21 | -0.94 | -1.81 | +0.00 | -0.03 | +2.06 | -0.14 |
| L22_log | -0.19 | -0.75 | +0.78 | +0.47 | +0.12 | +0.08 |
| L30 | -0.06 | +0.00 | -0.27 | +0.96 | -1.25 | -0.12 |
| L31 | -0.18 | +0.62 | -0.88 | -1.11 | -0.45 | -0.40 |
| L32 | -0.48 | +0.27 | -0.55 | -1.16 | -0.45 | -0.47 |
| L33_log | +1.15 | -0.35 | -0.58 | +0.69 | -0.80 | +0.02 |
| Le00_log | -1.46 | -0.80 | -1.58 | +1.07 | -2.53 | -1.06 |
| Le10 | +1.92 | +0.10 | -0.65 | +0.36 | -0.16 | +0.31 |
| Le11_log | -0.26 | +0.05 | -1.62 | +0.83 | -0.55 | -0.31 |

### T7e. Contrast attributable to the N(0,1) working assumption: (Δψ_N01 − Δψ_oracle) / SE_Step4

| ψ component | ds 0 | ds 1 | ds 2 | ds 3 | ds 4 | mean |
|---|---|---|---|---|---|---|
| beta0_T | +0.46 | +0.38 | +0.18 | +0.23 | +0.11 | +0.27 |
| beta1_T | -0.03 | -0.09 | -0.11 | -0.14 | -0.08 | -0.09 |
| gamma_T | +0.01 | +0.01 | -0.04 | -0.02 | -0.04 | -0.02 |
| beta0_P | +2.23 | +2.27 | +2.18 | +2.11 | +1.98 | +2.16 |
| beta1_P | -1.13 | -1.20 | -0.87 | -0.90 | -1.01 | -1.02 |
| gamma_P | +0.01 | -0.05 | +0.02 | +0.08 | -0.08 | -0.01 |
| L00_log | +0.06 | -0.20 | -0.30 | -0.37 | -0.37 | -0.24 |
| L10 | -0.04 | -0.10 | -0.06 | -0.03 | -0.06 | -0.06 |
| L11_log | -0.01 | -0.14 | -0.15 | -0.16 | -0.22 | -0.14 |
| L20 | +0.05 | -0.13 | -0.13 | -0.07 | -0.11 | -0.08 |
| L21 | -0.03 | -0.11 | -0.11 | -0.01 | +0.04 | -0.04 |
| L22_log | -0.58 | -0.87 | -0.32 | -0.50 | -0.71 | -0.59 |
| L30 | -0.05 | -0.04 | +0.00 | +0.02 | -0.03 | -0.02 |
| L31 | -0.02 | +0.11 | -0.04 | -0.08 | -0.02 | -0.01 |
| L32 | -0.50 | -0.36 | -0.57 | -0.40 | -0.54 | -0.47 |
| L33_log | -1.29 | -1.27 | -0.91 | -0.96 | -0.98 | -1.08 |
| Le00_log | -0.04 | -0.41 | -0.64 | -0.71 | -0.71 | -0.50 |
| Le10 | -0.05 | -0.23 | -0.14 | -0.08 | -0.13 | -0.12 |
| Le11_log | -1.84 | -2.21 | -1.52 | -1.49 | -1.84 | -1.78 |

### T7f. Contrast attributable to item sampling error under the oracle: (Δψ_oracle − Δψ_true) / SE_Step4

| ψ component | ds 0 | ds 1 | ds 2 | ds 3 | ds 4 | mean |
|---|---|---|---|---|---|---|
| beta0_T | +1.07 | -1.13 | -0.90 | -0.13 | -1.09 | -0.43 |
| beta1_T | -0.09 | +0.08 | -0.34 | +0.01 | -0.10 | -0.09 |
| gamma_T | -0.09 | +0.02 | -0.18 | -0.01 | -0.09 | -0.07 |
| beta0_P | +0.67 | -1.54 | +1.18 | +0.13 | -0.12 | +0.06 |
| beta1_P | +0.29 | -0.05 | -0.59 | +0.43 | -0.08 | -0.00 |
| gamma_P | +0.04 | -0.19 | -0.02 | +0.01 | +0.22 | +0.01 |
| L00_log | -0.75 | +0.61 | -1.41 | -0.15 | -0.92 | -0.52 |
| L10 | +0.13 | -0.03 | -0.02 | +0.06 | +0.03 | +0.03 |
| L11_log | -0.21 | +0.24 | -0.54 | +0.09 | -0.38 | -0.16 |
| L20 | +0.05 | -0.02 | -0.09 | +0.04 | +0.03 | +0.00 |
| L21 | -0.03 | +0.04 | -0.04 | +0.00 | -0.08 | -0.02 |
| L22_log | +0.24 | -0.42 | -0.46 | +0.39 | +0.53 | +0.06 |
| L30 | -0.02 | +0.03 | +0.01 | +0.00 | -0.03 | -0.00 |
| L31 | +0.05 | -0.03 | -0.05 | +0.02 | +0.01 | +0.00 |
| L32 | +0.13 | +0.08 | -0.24 | +0.17 | -0.30 | -0.03 |
| L33_log | +0.39 | +0.14 | -0.71 | +0.45 | -0.27 | +0.00 |
| Le00_log | -0.83 | +0.66 | -2.24 | -0.09 | -1.32 | -0.76 |
| Le10 | +0.07 | +0.06 | -0.11 | +0.06 | -0.05 | +0.00 |
| Le11_log | +0.45 | -0.04 | -1.17 | +0.73 | +0.10 | +0.01 |

**Reading T5.** Each fit minimises its own objective, with ‖∇‖ ≤ 1.2 at its optimum.
- Under the oracle objective, the N(0, 1) items for P are worse than the oracle items by 13.7–20.6 NLL units in every dataset. For T the gap is 0.7–1.4 units.
- The true items are not the optimum of either objective, because of sampling error.

**Reading T6.** At ψ_true, both estimated item sets give a higher stage-2 ℓ than the true items, by 8–50 units. This is expected, because the items are fitted to the same responses.

## 4. The first-order shift: formula, sign convention, limitations (plan §5.4)

**Formula.** f_I(ψ) = −ℓ(ψ; I), g_I = ∇f_I(ψ_true), and H = ∇²f_true(ψ_true). H is the central finite-difference Hessian of the exact gradient (h = 1e-4) with the true items, from cold mode starts.

**Shift and units.** Δψ_I = −H⁻¹g_I is one Newton step from ψ_true on f_I, using the true-item curvature. The table entries are Δψ_I / SE, where SE is the Step 4 standard error of the same dataset.
- The supplementary SE₀ = √diag(H⁻¹) at ψ_true is in `S4_downstream_shift.csv`.
- SE₀ / SE_Step4 lies between 0.74 and 1.12.

**Sign.** A positive entry means the stage-2 MLE with items I would, to first order, lie **above** ψ_true. For example, +2.16 for β0_P means the N(0, 1) items would place β0_P̂ about 2.2 SE higher than the oracle items would.

**Decomposition.**
- Δψ_true ≈ the fixed-item sampling deviation;
- Δψ_oracle − Δψ_true ≈ the effect of item sampling error with the latent distribution correct;
- Δψ_N01 − Δψ_oracle ≈ the effect of the N(0, 1) working assumption alone.

**Limitations.**
- It is a first-order (one-step) approximation. It ignores the change of curvature between item sets and along the path to ψ̂_I, and it is less reliable where f is far from quadratic.
- It is expressed in the log-Cholesky parameterisation of ψ, so it is not invariant to reparameterisation.
- It describes the location shift only, not interval width or coverage. It is evaluated only at ψ_true under the a-priori truth.
- Its one internal check (T7a vs T7b, ≤ 0.26 SE) covers the true-item case only. The larger N(0, 1) shifts (about 2 SE) are further from ψ_true, where the quadratic approximation is less assured.
- **It is not a refit and not a recovery result.** No stage-2 optimisation was run (D-2).

## 5. What S1-a does and does not establish

**It establishes**, descriptively and for these 5 synthetic datasets under the a-priori truth and the real design:
- the N(0, 1) stage-1 working assumption produces systematic, dataset-stable item errors, concentrated in the PIGD subscore (inflated a, lowered upper τ);
- to first order at ψ_true, these errors would move β0_P, Le11_log, L33_log and β1_P by about 1–2 SE, with consistent signs.

**It does not establish:**
- whether full-pipeline estimation recovers ψ (coverage);
- whether S1-b is needed, sufficient or unnecessary;
- anything about real data, misspecification or labels;
- anything about the composite (per-visit independence) likelihood, which both item fits share.

These results are **not** used to change the frozen protocol, to select a setting, or to decide on S1-b. The S1-b procedure and the Step 6 criterion stand as frozen in Rev 2.

## 6. Deviations, errata and fallback status

1. **Plan §11, test 3: erratum in the frozen text.** The frozen plan states the affine identity the wrong way round: it reads "items (a·s, (τ − m)/s) … equals the N(0, 1) value with items (a, τ)".
   - The correct identity, which is the one implemented and tested, is: the oracle value at items (a, τ) with a common (m, s) equals the N(0, 1) value at items (a·s, (τ − m)/s), since a(m + s·x − τ) = (a·s)(x − (τ − m)/s).
   - This is a description error only. It affects no computation, criterion or result.
   - The frozen document is unchanged; this report records the correction.
2. **Item-fit iteration counts for the N(0, 1) fits** are not exposed by the reused pilot function, so they are not reported (T1 "n/r"). Using that function unchanged was required for exact reproduction (D-5).
3. **Smoke check** (job 13907, `smoke_codepath/`): 1 dataset, 60 patients, oracle maxiter 20. It was a code path only. Its pilot-reproduction "False" and non-PD H are artefacts of the subset, not results.
4. **Fallback (D-4).**
   - G = 5 AGQ with the exact gradient was available for every non-ND patient mode. It was never invoked: 0 of 205 evaluations, 0 patients. Every evaluation's record is in `S3_downstream_dataset{r}.json` (`records`).
   - The unresolved fallback-optimisation cost issue (doc 28; deferral in doc 31 §0.1) did not affect Step 5, and it remains open for Step 7 under the recorded rule.

## 7. Files

**New files** (untracked; the only files written outside the execution directory):
- `scripts/51_m2a1_s1a_true_latent_items.py`;
- `tests/test_s1a_latent_items.py`;
- docs 33, 34, 35;
- `_frozen/34_STAGE2_STEP5_PLAN.{as_frozen_20261019.md,sha256}`.

**Execution directory:**
- `00_baseline.json`, `01_plan.sha256`, `02_driver_sha256.txt`;
- `step5/`:
  - `S0_provenance.json`;
  - `S1_items_dataset{0..4}.json`;
  - `S2_item_errors.csv`;
  - `S3_downstream_dataset{0..4}.json` (H, eigenvalues, all evaluation records);
  - `S4_downstream_shift.csv`;
  - `summary.json`;
- `smoke_codepath/`;
- `jobs_sacct.txt`, `99_post_run_integrity.txt`, `slurm-1390{7,8}.{out,err}`, `job_header.txt`, `REPORT.md`.

**Post-run integrity** (`99_post_run_integrity.txt`):
- every frozen lock is OK;
- code, documents and inputs are unchanged against the baseline;
- no file in the pilot, Step 3 or Step 4 directories, or under `docs/`, `src/`, `scripts/`, `tests/` or `outputs/`, was modified during the run;
- git HEAD is `3a860ca`, with 0 tracked changes and 0 staged.

## 8. Not done

The following were not run:
- Step 6 (full-pipeline recovery, S1-b);
- Step 7 (real-data B11, the sampler, V5, DIF);
- Step 8 (G1–G7);
- any real-data fit;
- calibration;
- positive controls;
- evaluation statistics;
- clustering.

Frozen records, prior reports and prior outputs are unchanged. The D1 verdict remains **NOT REPRODUCED**. Nothing was staged, committed or pushed.

**Step 6, full-pipeline recovery, remains the binding recovery test.** S1-a has no acceptance criterion and cannot establish full-pipeline recovery.
