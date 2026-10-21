# Stage 2 M2 Rev 3: independent recovery test — final report and completion record

**Date:** 2026-10-20 (evaluation 10:46Z).

**Protocol:** doc 46 (`e2b40c11…`) and doc 47 (`36144159…`), revision R1.3, frozen together on 2026-10-19 (change record: doc 50). Both locks still verify. Doc 52 is the independent review of R1.3.

**Run:** `experiments/20261019T173512Z_stage2_rev3_recovery_test/`.

> **VERDICT: PASS** (doc 47 §8, precedence row 5).
> - All 21 recovery gates and all 15 guards pass on A and on A \ F. The evaluation recorded no reason codes and no robust failures.
> - All 100 main datasets completed. **No dataset was flagged (|F| = 0)**, so A \ F = A. No interval was undefined.
> - **Historical verdicts are unchanged and are not re-scored:** Step 6 full-pipeline recovery = **FAILED**; D1 model-behaviour experiment = **NOT REPRODUCED**.
> - **Step H review is still required.** Steps 7–8, calibration, positive controls, real-data analysis and S1-c remain blocked. Nothing in this report authorises them.
> - Nothing was staged, committed or pushed.

---

## 1. Verdict and the rule that produced it

The rows of the frozen precedence (doc 47 §8) are applied in order:

| row | condition | observed | result |
|---|---|---|---|
| 1 INTEGRITY STOP | preflight, regeneration, regression R-a–R-d, truth-swap, F5 truth identity | all passed (§2) | not triggered |
| 2 INCOMPLETE | main-dataset exception, fewer than 100 complete, runtime gate | 100/100 complete with `"complete": true`; no `.error.json`; runtime gate passed; both Phase 2 jobs `stop = null` | not triggered |
| 3 FLAG EXCESS | \|F\| > 10 | \|F\| = 0 | not triggered |
| 4 FAIL | any recovery gate failing on both A and A \ F | none | not triggered |
| **5 PASS** | **all 36 gates pass on A and on A \ F** | **36/36 pass on both sets** | **PASS** |

**Gate status** is pass iff all three criteria hold:
- C1: coverage ≥ ⌈0.88·100⌉ = 88;
- C2: |mean z| ≤ 0.40;
- C0: undefined ≤ ⌊0.05·100⌋ = 5.

The driver's `evaluate` mode produced this verdict. I recomputed it independently from the 100 per-dataset records (same z, coverage and definedness arrays, criteria coded afresh) and got the same result.

**Margins:**
- The closest gate to any criterion is **L22_log, mean z −0.342** against the C2 limit of 0.40.
- Minimum coverage is **91/100** (L20, ν_T, corrD_20), against the C1 limit of 88.
- No interval was undefined anywhere (C0 margin 5).

## 2. Execution and integrity

| step | job | node | elapsed | outcome |
|---|---|---|---|---|
| defective smoke submission | 14013 | — | 0 s | cancelled while PENDING (`DEV_DEFECT_NOTE_pre_smoke.txt`); not run, not counted as a repeat |
| smoke run (NOT A RESULT) | 14017 | node003 | 301 s | code path complete, no error. F1/F2 arose by design (maxiter 5/20). Truth-swap passed. 0 of 2 repeats used (`03_smoke_record.txt`) |
| code freeze | — | — | — | `03_code_freeze.sha256`: 19 files (driver, tests, scripts 45 and 51–56, 9 `src` modules, `submit.sh`). Focused tests 38/38 passed |
| Phase 1 | 14094 | node002 | 3680 s | passed (below) |
| Phase 2a: main 3–51, control 0–4 | 14095 | node002 | 41682 s | complete, `stop = null` |
| Phase 2b: main 52–99, control 5–9 | 14096 | node003 | 40280 s | complete, `stop = null` |

**Phase 1** (`phase1/summary.json`):
- **Preflight:** frozen locks OK, plan locks present, panel MD5 OK, ψ_true equal to the Step 6 record, code freeze OK (19 files), S(ψ_true) fixed point 1.7e-9 (≤ 1e-8).
- **Regeneration:** r = 0 and r = 99 regenerated bit-identically.
- **Regression on old dataset 0** (`[42, 45, 1, 0]`):
  - **R-a:** ψ̂⁽¹⁾ and ψ̂⁽²⁾ are bitwise equal to Step 6, and so are H₁ and H₂.
  - **R-b:** max |S(ψ̂⁽²⁾) − D7| = 0.0 (≤ 1e-12).
  - **R-c:** max relative SE difference against D4 = 0.0 (≤ 1e-6).
  - **R-d:** N3 = 2.75e-7, N4 = 9.37e-8 (≤ 1e-5).
- **Truth-swap** (fresh r = 0, ψ_true + 1): S(ψ̂) and SE_S bit-identical, z changed → **passed**.
- **Runtime gate:**
  - P = 23.89 ≤ 28; U_smoke + P = 23.97 ≤ 30;
  - per-job projection 11.55 h ≤ L₂ = 13.5 h;
  - maximum fit 243 s ≤ 3600 s.
- **Phase 2 budget record:** `04_phase2_budget.txt`.
  - U = 1.1058 node-hours (sacct), so L₂ = min(13.5, 14.45) = 13.5 h.
  - Worst case 28.11 ≤ 30.

**Phase 2:**
- Both jobs re-ran preflight with the code freeze OK. The main datasets ran first, then the control arm.
- Hosts: node002 (52 datasets including Phase 1), node003 (48 datasets).
- Every stage-2 fit converged, with at most 125 iterations.
- Pipeline time per dataset: 508–712 s (mean 600 s). The longest stage-2 fit took 293 s.

**Post-run integrity** (`05_post_run_integrity.json`):
- the 419 prior output files and the 77 baseline docs are unchanged;
- all frozen locks verify;
- the panel MD5 matches;
- the code freeze verifies at evaluation time;
- HEAD is `3a860ca`, with nothing staged.

The one code file changed since `00_baseline.json` is the driver, which was still under development when the baseline was taken. The governing record is the post-smoke code freeze.

## 3. The 21 recovery gates

Definitions:
- **est − truth:** z·SE.
- **SD(z) CI:** χ²-based 95 % interval.
- **Secondary:** the fixed-item H₂⁻¹ interval on the re-expressed estimate (reported only).
- **Status:** given on A; A \ F is identical because |F| = 0.

| gate | truth | coverage /100 | mean z | SD(z) [95 % CI] | bias | RMSE | mean SE | secondary cov. | status |
|---|---|---|---|---|---|---|---|---|---|
| β0_T | −0.2066 | 95 | −0.064 | 0.960 [0.843, 1.116] | −0.0012 | 0.0190 | 0.0199 | 95 | pass |
| β1_T | 0.0478 | 97 | −0.024 | 0.858 [0.754, 0.997] | −0.0002 | 0.0061 | 0.0071 | 97 | pass |
| γ_T | 0.0957 | 99 | +0.093 | 0.815 [0.715, 0.946] | +0.0031 | 0.0260 | 0.0320 | 98 | pass |
| β0_P | −0.4768 | 96 | −0.059 | 0.979 [0.860, 1.138] | −0.0010 | 0.0211 | 0.0216 | 96 | pass |
| β1_P | 0.1216 | 96 | +0.201 | 0.833 [0.731, 0.968] | +0.0014 | 0.0063 | 0.0075 | 96 | pass |
| γ_P | 0.1621 | 98 | −0.172 | 0.836 [0.734, 0.972] | −0.0057 | 0.0285 | 0.0337 | 98 | pass |
| L00_log | −0.2674 | 93 | +0.049 | 0.992 [0.871, 1.152] | +0.0004 | 0.0263 | 0.0267 | 93 | pass |
| L10 | 0.0077 | 96 | +0.052 | 1.005 [0.882, 1.167] | +0.0003 | 0.0071 | 0.0072 | 95 | pass |
| L11_log | −2.5750 | 96 | −0.176 | 0.934 [0.820, 1.085] | −0.0240 | 0.0881 | 0.0896 | 97 | pass |
| L20 | 0.0648 | 91 | −0.071 | 1.095 [0.961, 1.271] | −0.0025 | 0.0353 | 0.0328 | 91 | pass |
| L21 | −0.0065 | 95 | +0.057 | 1.150 [1.010, 1.336] | +0.0030 | 0.0718 | 0.0633 | 94 | pass |
| L22_log | −0.4382 | 94 | **−0.342** | 1.025 [0.900, 1.190] | −0.0156 | 0.0426 | 0.0384 | 93 | pass |
| L30 | 0.0000 | 97 | +0.218 | 1.020 [0.895, 1.184] | +0.0017 | 0.0081 | 0.0078 | 96 | pass |
| L31 | 0.0000 | 95 | −0.183 | 0.994 [0.873, 1.154] | −0.0020 | 0.0111 | 0.0109 | 95 | pass |
| L32 | 0.0098 | 96 | +0.142 | 0.945 [0.830, 1.098] | +0.0013 | 0.0085 | 0.0089 | 96 | pass |
| L33_log | −2.3353 | 94 | −0.281 | 0.981 [0.861, 1.139] | −0.0291 | 0.0829 | 0.0783 | 93 | pass |
| Le00_log | −0.7374 | 96 | +0.004 | 0.987 [0.867, 1.147] | −0.0002 | 0.0284 | 0.0292 | 96 | pass |
| Le10 | 0.0405 | 95 | −0.086 | 1.023 [0.898, 1.188] | −0.0014 | 0.0169 | 0.0166 | 95 | pass |
| Le11_log | −0.9081 | 96 | +0.233 | 0.946 [0.831, 1.099] | +0.0076 | 0.0349 | 0.0365 | 96 | pass |
| ν_T (item-anchored location T) | −3.8100 | 91 | −0.127 | 1.105 [0.970, 1.283] | −0.0101 | 0.0768 | 0.0695 | **87** | pass |
| ν_P (item-anchored location P) | −5.0500 | 94 | −0.082 | 1.024 [0.899, 1.190] | −0.0119 | 0.1074 | 0.1036 | **82** | pass |

## 4. The 15 guards (ψ-only invariants, no transformation)

| guard | truth | coverage | mean z | SD(z) [95 % CI] | bias | RMSE | status |
|---|---|---|---|---|---|---|---|
| I1 slope/SD(int) T | 0.0625 | 97 | −0.037 | 0.851 [0.747, 0.988] | −0.0003 | 0.0080 | pass |
| I2 treat/SD(int) T | 0.1250 | 97 | +0.089 | 0.808 [0.710, 0.939] | +0.0039 | 0.0337 | pass |
| I3 ½ log var ratio slope/int T | −2.3026 | 95 | −0.149 | 0.963 [0.845, 1.118] | −0.0186 | 0.0899 | pass |
| I4 ½ log var ratio ε/int T | −0.4700 | 96 | −0.025 | 0.989 [0.868, 1.149] | −0.0006 | 0.0427 | pass |
| I5 slope/SD(int) P | 0.1875 | 98 | +0.235 | 0.816 [0.716, 0.948] | +0.0038 | 0.0125 | pass |
| I6 treat/SD(int) P | 0.2500 | 97 | −0.141 | 0.871 [0.765, 1.012] | −0.0066 | 0.0457 | pass |
| I7 ½ log var ratio slope/int P | −1.8971 | 93 | −0.028 | 1.092 [0.959, 1.268] | −0.0043 | 0.0916 | pass |
| I8 ½ log var ratio ε/int P | −0.4700 | 92 | +0.270 | 1.062 [0.933, 1.234] | +0.0159 | 0.0605 | pass |
| I9 corr D 1,0 | 0.1000 | 93 | +0.042 | 1.008 [0.885, 1.171] | +0.0079 | 0.0976 | pass |
| I10 corr D 2,0 | 0.1000 | 91 | −0.049 | 1.104 [0.970, 1.283] | −0.0031 | 0.0548 | pass |
| I11 corr D 2,1 | 0.0000 | 94 | +0.067 | 1.130 [0.992, 1.312] | +0.0062 | 0.1067 | pass |
| I12 corr D 3,0 | 0.0000 | 97 | +0.218 | 1.024 [0.899, 1.189] | +0.0178 | 0.0853 | pass |
| I13 corr D 3,1 | 0.0000 | 95 | −0.193 | 1.019 [0.895, 1.184] | −0.0203 | 0.1106 | pass |
| I14 corr D 3,2 | 0.1000 | 97 | +0.092 | 0.937 [0.823, 1.088] | +0.0128 | 0.0911 | pass |
| I15 corr ε T,P | 0.1000 | 95 | −0.103 | 1.004 [0.881, 1.166] | −0.0046 | 0.0405 | pass |

Full table: `evaluation/gates.csv`. Figures: `evaluation/z_by_gate.png`, `coverage_by_gate.png`, `cd_distribution.png`.

## 5. Findings: recovery and uncertainty calibration

These are kept separate, as the protocol requires.

**Parameter recovery**
- Every point estimate is close to unbiased in natural units. All bias magnitudes are below 0.03 on the log-Cholesky scale and below 0.006 for β and γ.
- The largest standardised biases are L22_log (−0.342), L33_log (−0.281), I8 (+0.270) and Le11_log (+0.233). They are within the 0.40 criterion but not negligible.
- β0_P has mean z −0.059 (coverage 96). In Step 6 it failed at 3/5, and on the raw scale here it still has mean z +2.63 (§6). On the Rev 3 convention it is recovered.

**Uncertainty calibration**
- Mean coverage across the 36 gates is 95.2 % (range 91–99).
- The full-chain sandwich SEs match the empirical SD of the estimates closely for the variance-component and location parameters.

**Two systematic departures, neither of which affects the verdict:**
- **Conservative intervals for slopes and treatment effects.** Seven gates have SD(z) CIs wholly below 1, with SD(z) 0.81–0.86 and coverage 96–99:
  - β1_T, γ_T, β1_P, γ_P;
  - the ratio guards I1, I2, I5.

  The mean SE exceeds the empirical SD by about 16–24 % (for example γ_T: SE 0.0320 against SD 0.0260). With 36 CIs, about 1.8 would exclude 1 by chance. Seven, all in one family, is a systematic over-estimate of uncertainty for these parameters. This test does not establish the cause.
- **Mild under-coverage for L21.** Its SD(z) CI is wholly above 1 (SD(z) 1.150 [1.010, 1.336]), although coverage is still 95/100. L20, ν_T and corrD_20/21 point the same way (SD(z) 1.10–1.13), but their CIs include 1.

**Failure labels** (bias-type, dispersion-type, mixed, unclassified) apply only to failing gates, so none were assigned.

**Full chain against fixed items:**
- For the 19 ψ components, the secondary fixed-item intervals cover about as well as the primary full-chain intervals (difference ≤ 1 dataset).
- For the item-anchored locations they do not: ν_T covers 87/100 with fixed items against 91 with the full chain, and ν_P covers 82 against 94. The fixed-item ν intervals would have failed C1 (< 88). The full-chain propagation of item uncertainty therefore matters mainly for the item-anchored locations.

## 6. Comparators (reported only; none enters the verdict)

**Historical D3 comparator** (doc 16 D3 literally: raw ψ̂⁽²⁾ against raw truth, fixed-item H₂⁻¹ intervals, fresh datasets 0–4; S5):
- The D3 rule is **not met**: β0_P 0/5, Le00_log 2/5, Le11_log 2/5, L22_log 3/5, L33_log 3/5. Every other component is ≥ 4/5.
- On the same five datasets, every Rev 3 gate is ≥ 4/5.
- This is the Step 6 failure pattern appearing again on fresh data under the historical definition. It is reported as a comparator. **It does not re-score Step 6, and Step 6 remains FAILED.**

**Raw fixed-item intervals, 100 datasets:**

| component | coverage | mean z |
|---|---|---|
| β0_P | 26 | +2.63 |
| L22_log | 53 | −1.84 |
| Le11_log | 57 | −1.66 |
| Le00_log | 73 | −0.62 |
| L33_log | 82 | −1.10 |
| β1_P | 86 | −0.84 |

The remaining components cover 91–99. On the raw ψ scale the estimates do not recover the truth for these components. The Rev 3 pooled-moment convention is what makes the recovery definition meetable. This is consistent with the D7 diagnosis of a scale-anchoring cause.

**N(0, 1)-only comparator** (S(ψ̂⁽¹⁾) with H₁⁻¹ delta-method intervals, without the S1-b refit):
- Coverage is 91–99. L33_log has mean z −0.672 and L22_log +0.363; every other component is within ±0.22.
- The S1-b refit (D1 of doc 46) is associated with the lower L33_log bias in the primary analysis (−0.281).
- The interval constructions differ, so this is descriptive only.

**Le00_log attribution** (first-order, raw ψ units, mean over 100 datasets):
- predicted total shift −0.0140, of which tremor items −0.0140 and PIGD items +5e-7;
- affine −0.0138, non-affine −0.0002.

Le00_log, the Step 6 failing component with mean z −0.61 in development, has mean z +0.004 here.

**Item recovery** (S1-b items against the truth, means): discrimination correlation 0.987 (T) and 0.985 (P); maximum |a| error 0.18 and 0.19; maximum |τ| error 0.27 and 0.15.

**Re-expression moments:**

| moment | range |
|---|---|
| c_T | 0.943–1.059 |
| d_T | −0.060 to 0.049 |
| c_P | 0.859–1.056 |
| d_P | −0.016 to 0.105 |

## 7. Control arm (true items, datasets 0–9; R9 report-only)

- 10/10 complete, all fits converged.
- **Mean over the 10 control datasets** of (S(ψ̂) − ψ̂)/SE_raw, per component: |mean| ≤ 0.17. The largest are β0_T +0.172, Le00_log −0.166 and L00_log −0.123.
- **Individual values:** the largest |(S(ψ̂) − ψ̂)/SE_raw| is 2.18 (dataset 1, β0_P); 15 of the 190 component-dataset values exceed 1.
- **Moments:** c_T 0.962–1.034, d_T −0.040 to 0.059, c_P 0.958–1.025, d_P −0.071 to 0.039.

This is consistent with the doc 47 §10.3 expectation: with correctly scaled items, S still moves estimates by amounts of the order of the sampling SE of (c, d). No systematic mean shift is evident; the largest mean is 0.17 SE. The control arm has no effect on the verdict.

## 8. Numerical checks (maxima over 100 datasets; doc 46 §3.5)

| check | observed | tolerance |
|---|---|---|
| N1 estimating equations | 8.1e-12 | 1e-8 relative |
| N3 stage-2 relabelling identity | 4.2e-7 | 1e-5 |
| N4 S1-b item relabelling identity | 7.7e-7 | 1e-5 |
| N5 symmetry | 2.0e-9 | 1e-4 |
| N6 cond₂ of the stacked Jacobian | 4.2e3 | 1e12 |
| N7 V PSD | all PSD (min/max eigenvalue 1.0e-10 to 1.1e-9, positive) | ≥ −1e-10 |
| N8 FD-step sensitivity, datasets 0–2 | 1.39e-6, 1.41e-6, 1.64e-6 | 0.05 |
| N9 information equality (reported only) | median 1.004, range 0.73–1.54 | — |
| N10 annihilation and rank | 1.4e-8; rank 15 everywhere | 1e-6; 15 |
| idempotence S(S(ψ̂)) | 1.8e-9 | 1e-8 |
| relabelling invariance | 1.6e-9 | 1e-8 |
| penalised FD Hessian evaluations | 0 | F6 if > 0 |

No flag F1–F8 was raised on any main dataset.

## 9. Compute usage (sacct ElapsedRaw, exclusive single nodes)

| item | node-hours |
|---|---|
| smoke 14017 | 0.084 |
| Phase 1 14094 | 1.022 |
| Phase 2a 14095 | 11.578 |
| Phase 2b 14096 | 11.189 |
| **total** | **23.873** of the **30** cap (projected at the gate: 23.97) |

The cancelled job 14013 used 0 node-hours. Evaluation ran on the login node (no fitting).

## 10. What this establishes and what it does not (doc 47 §13)

**What it establishes:**
- For the a-priori truth and the real 934-patient design, the Rev 3 pipeline (two-stage estimation, one S1-b iteration, pooled-moment re-expression, full-chain sandwich) meets every pre-registered criterion: approximately calibrated 95 % intervals and small bias for all 21 recovery gates, with all 15 guards consistent.
- After the user's step-H review, this satisfies the binding recovery prerequisite of the amended sequence.

**What it does not establish:**
- calibration of the sandwich in other settings;
- performance under other truths, designs or misspecification;
- anything about real PPMI data, model fit, subtypes or D1.

**Caveats for reading the PASS:**
- **The A matrix mixes objective contexts** (doc 52 F-C). H₁ and H₂ come from the optimiser context and C₁ and C₂ from the reported context, as Step 6 did. The PASS is an empirical calibration result for this exact construction, not a proof of M-estimation validity.
- **Two calibration departures are documented in §5:** conservative slope and treatment intervals, and mild L21 under-dispersion.
- **L22_log has the smallest bias margin** (|mean z| 0.342 against 0.40).
- **Phase 2 cross-node last-bit equivalence was not checked** (doc 50 §4). Phase 1's bitwise regression was on node002, and 2a also ran on node002.

**Historical verdicts:**
- **Step 6 full-pipeline recovery remains FAILED** under its own frozen criteria (doc 37/38). This test used new datasets, a new convention and a new recovery definition, and was not applied to the Step 6 datasets.
- **D1 remains NOT REPRODUCED.**
- Neither failure is evidence that the underlying model is false, and this PASS is not evidence that it is true.

## 11. Completion record

**Files created in this stage:**
- Run directory `experiments/20261019T173512Z_stage2_rev3_recovery_test/`:
  - `03_code_freeze.sha256` (`656e056c…`);
  - `03_smoke_record.txt`;
  - `04_phase2_budget.txt`;
  - `05_post_run_integrity.json`;
  - `smoke_codepath/` (with `job_context/`);
  - `phase1/` (with `job_context/`);
  - `phase2a/` and `phase2b/` (main, control, `job_start.json`, `job_end.json`, SLURM logs);
  - `evaluation/` (`summary.json` `b03a0f50…`, `gates.csv` `cc5f4aaa…`, three figures, `MANIFEST.sha256`);
  - `MANIFEST.sha256` for the run directory, written after this report.
- This report, doc 53.
- Earlier in the directive: doc 50, `scripts/58_m2_rev3_prefreeze_checks.py`, `scripts/59_m2_rev3_recovery_test.py`, `tests/test_rev3_recovery.py`, and the `_frozen/46_*` and `_frozen/47_*` locks and copies.

**Protocol locks:**
- `_frozen/46_STAGE2_REV3_AMENDMENT.sha256` (`c9b7b51c…`);
- `_frozen/47_STAGE2_REV3_RECOVERY_TEST_PLAN.sha256` (`6587b1be…`).

**SLURM jobs:** 14013 (cancelled pending), 14017, 14094, 14095, 14096.

**Repository:** HEAD `3a860ca`; nothing staged, committed or pushed.

**Stop condition:** **A — Completed.**
