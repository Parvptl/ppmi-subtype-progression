# Stage 2 M2, Step 4 report: fixed-item recovery

**Date:** 2026-10-19.

**Execution plan:** `31_STAGE2_STEP4_FIXED_ITEM_RECOVERY_PLAN_FINAL_20261019.md`, SHA-256 `830ce768a607421486f856bc6a6339a1a32926e5c5e1bc8e49f61918cb5c9b4c`.
- Frozen before computation, with a read-only copy in `_frozen/`.
- It carries the reviewer decisions of 2026-10-19 (§0).

**Execution directory:** `experiments/20261018T184713Z_stage2_step4_fixed_item_recovery/`. The results are in **`step4_run2/`** (job 13905).

**Scope.** Synthetic data only:
- 5 datasets, seeds `[42, 45, 1, r]`;
- the a-priori truth on the stage-1 scale;
- the real visit design (times and treatment of the 934 patients). **No real item responses were used.**
- Item parameters fixed at truth.
- Primary model: Laplace over b with per-visit Q = 15 (Step 3). Default start (`m2_grm.default_psi()`, values in `S0_provenance.json`), one start per dataset.
- 95 % Wald intervals from the central finite-difference Hessian (h = 1e-4) of the exact gradient.

D1 remains **NOT REPRODUCED**.

## 1. Outcome

| criterion | result |
|---|---|
| Identity of the synthetic data (plan §0) | serial Q = 10 objective at the doc 28 optimum = 57,022.74225348176, equal to the record. **Pass** |
| **Runtime gate (§0.2), after dataset 0** | measured Q = 15 fit **179 s** (≤ 60 min); projected pilot **1.88 node-hours** (≤ 24). **Pass** |
| **Coverage: each of the 19 ψ components covered in ≥ 4 of 5 datasets** | minimum 4 of 5 (6 components at 4/5, 13 at 5/5); 90 of 95 intervals cover (94.7 %). **PASS** |
| Hessian positive definite, all datasets | yes (minimum eigenvalue 56.9–157.2) |
| Fallback invoked at any optimum | **no** (0 patients; 0 budget triggers) |

**Step 4 (fixed-item recovery, a diagnostic pre-requisite) passes.** Under Rev 2 §5 this authorises nothing beyond itself: the binding recovery criterion is the full-pipeline recovery (step 6).

## 2. Per-dataset fits and diagnostics

No convergence acceptance threshold is applied (plan §0.3). ‖∇‖ against 0.934 is shown for reference only.

| dataset | negative log-likelihood | optimiser | iterations / evaluations | ‖∇‖ | bound hits | min Hessian eigenvalue | PD | covered | σ̂_T / σ̂_P | penalty evaluations (all `mode_budget`) | non-ND at accepted evaluations | fallback at optimum | fit / Hessian time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 57,022.167 | converged | 80 / 90 | 0.0031 | none | 126.6 | yes | 19/19 | 0.4712 / 0.3999 | 2 | 0 | 0 | 179 s / 54 s |
| 1 | 58,726.443 | converged | 84 / 96 | 0.00043 | none | 157.2 | yes | 19/19 | 0.4621 / 0.4068 | 3 | 0 | 0 | 209 s / 54 s |
| 2 | 58,037.323 | converged | 100 / 110 | 0.0012 | none | 110.3 | yes | 18/19 | 0.4861 / 0.3995 | 2 | 0 | 0 | 203 s / 54 s |
| 3 | 58,001.970 | converged | 108 / 124 | 0.00049 | none | 56.9 | yes | 18/19 | 0.4930 / 0.4075 | 3 | 0 | 0 | 244 s / 54 s |
| 4 | 58,617.649 | converged | 84 / 92 | 0.00098 | none | 138.0 | yes | 16/19 | 0.4655 / 0.3964 | 2 | 0 | 0 | 181 s / 55 s |

- Truth: σ_T = 0.4783, σ_P = 0.4053.
- "Converged" means the L-BFGS-B relative-reduction criterion.
- There were no non-convergences and no Hessian failures.

## 3. All 19 parameters × 5 datasets

Estimate [95 % Wald interval]; ✗ = not covered. The truth is on the stage-1 scale. The machine-readable version is `step4_run2/S2_coverage_table.csv` (estimate, SE, bounds, z, covered, PD).

| parameter | truth | dataset 0 | dataset 1 | dataset 2 | dataset 3 | dataset 4 | covered |
|---|---|---|---|---|---|---|---|
| beta0_T | -0.2066 | -0.1921 [-0.2539, -0.1303] | -0.2011 [-0.2613, -0.1409] | -0.1829 [-0.2453, -0.1206] | -0.2313 [-0.2910, -0.1717] | -0.2246 [-0.2884, -0.1608] | 5/5 |
| beta1_T | 0.0478 | 0.0357 [0.0216, 0.0499] | 0.0485 [0.0347, 0.0623] | 0.0499 [0.0356, 0.0642] | 0.0497 [0.0363, 0.0632] | 0.0638 [0.0498, 0.0778] ✗ | 4/5 |
| gamma_T | 0.0957 | 0.1047 [0.0419, 0.1676] | 0.1434 [0.0823, 0.2045] | 0.0707 [0.0074, 0.1339] | 0.1178 [0.0548, 0.1808] | 0.1100 [0.0480, 0.1721] | 5/5 |
| beta0_P | -0.4768 | -0.4885 [-0.5478, -0.4292] | -0.4435 [-0.5030, -0.3841] | -0.4768 [-0.5379, -0.4156] | -0.5023 [-0.5622, -0.4425] | -0.4853 [-0.5449, -0.4257] | 5/5 |
| beta1_P | 0.1216 | 0.1211 [0.1050, 0.1371] | 0.1224 [0.1069, 0.1378] | 0.1351 [0.1193, 0.1508] | 0.1190 [0.1033, 0.1347] | 0.1307 [0.1153, 0.1461] | 5/5 |
| gamma_P | 0.1621 | 0.1415 [0.0766, 0.2065] | 0.1556 [0.0913, 0.2199] | 0.0773 [0.0119, 0.1426] ✗ | 0.2046 [0.1396, 0.2695] | 0.1256 [0.0612, 0.1899] | 4/5 |
| L00_log | -0.2674 | -0.2466 [-0.3115, -0.1816] | -0.2809 [-0.3474, -0.2144] | -0.2409 [-0.3062, -0.1755] | -0.3111 [-0.3797, -0.2425] | -0.2000 [-0.2645, -0.1355] ✗ | 4/5 |
| L10 | 0.0077 | 0.0098 [-0.0041, 0.0238] | 0.0016 [-0.0124, 0.0156] | 0.0109 [-0.0027, 0.0244] | 0.0255 [0.0116, 0.0394] ✗ | 0.0007 [-0.0142, 0.0155] | 4/5 |
| L11_log | -2.5750 | -2.5702 [-2.7438, -2.3967] | -2.5607 [-2.7165, -2.4050] | -2.5655 [-2.7515, -2.3794] | -2.7926 [-3.0512, -2.5339] | -2.5512 [-2.7163, -2.3861] | 5/5 |
| L20 | 0.0648 | 0.1013 [0.0382, 0.1644] | 0.0682 [0.0052, 0.1312] | 0.0633 [-0.0016, 0.1281] | 0.0577 [-0.0073, 0.1227] | 0.1075 [0.0450, 0.1700] | 5/5 |
| L21 | -0.0065 | -0.0658 [-0.1867, 0.0552] | -0.0969 [-0.2090, 0.0152] | -0.0034 [-0.1281, 0.1213] | -0.0241 [-0.1791, 0.1309] | 0.1087 [-0.0036, 0.2209] ✗ | 4/5 |
| L22_log | -0.4382 | -0.4578 [-0.5419, -0.3736] | -0.4542 [-0.5391, -0.3692] | -0.3890 [-0.4674, -0.3105] | -0.4342 [-0.5164, -0.3521] | -0.4575 [-0.5398, -0.3753] | 5/5 |
| L30 | 0.0000 | -0.0006 [-0.0160, 0.0148] | -0.0003 [-0.0151, 0.0145] | -0.0023 [-0.0166, 0.0121] | 0.0068 [-0.0086, 0.0223] | -0.0102 [-0.0258, 0.0054] | 5/5 |
| L31 | 0.0000 | -0.0023 [-0.0236, 0.0190] | 0.0056 [-0.0135, 0.0247] | -0.0095 [-0.0309, 0.0120] | -0.0167 [-0.0428, 0.0093] | -0.0030 [-0.0231, 0.0171] | 5/5 |
| L32 | 0.0098 | 0.0043 [-0.0129, 0.0215] | 0.0114 [-0.0051, 0.0280] | 0.0073 [-0.0095, 0.0241] | -0.0027 [-0.0205, 0.0152] | 0.0077 [-0.0085, 0.0240] | 5/5 |
| L33_log | -2.3353 | -2.2852 [-2.4193, -2.1511] | -2.3739 [-2.5212, -2.2266] | -2.3233 [-2.4613, -2.1853] | -2.3228 [-2.4666, -2.1790] | -2.3681 [-2.5194, -2.2168] | 5/5 |
| Le00_log | -0.7374 | -0.7524 [-0.8006, -0.7042] | -0.7721 [-0.8202, -0.7239] | -0.7214 [-0.7680, -0.6747] | -0.7073 [-0.7532, -0.6613] | -0.7647 [-0.8129, -0.7165] | 5/5 |
| Le10 | 0.0405 | 0.0712 [0.0384, 0.1039] | 0.0416 [0.0089, 0.0743] | 0.0322 [-0.0003, 0.0647] | 0.0454 [0.0133, 0.0776] | 0.0382 [0.0058, 0.0707] | 5/5 |
| Le11_log | -0.9081 | -0.9326 [-1.0021, -0.8631] | -0.9047 [-0.9698, -0.8397] | -0.9207 [-0.9895, -0.8520] | -0.9039 [-0.9708, -0.8370] | -0.9301 [-0.9982, -0.8620] | 5/5 |

- **Mean z** across datasets is within ±0.64 for every component (`summary.json`).
- **The visit-level SDs no longer show a downward bias:**
  - Le11_log (PIGD visit SD): z = −0.69, +0.10, −0.36, +0.12, −0.63; covered 5/5. Doc 18, with the 2n-dimensional Laplace, had 3/5 with all z negative.
  - Le00_log (tremor visit SD): 5/5.
- **The non-covered intervals** are spread over unrelated components (β1_T, γ_P, L00, L10, L21), at about the nominal 5 % rate.

## 4. Runtime and accounting

**Measured:** the Q = 15 fit took 179–244 s (32 processes); the Hessian took 54 s (38 likelihood-plus-gradient evaluations, about 1.42 s each, including the warm state).

**Runtime gate (dataset 0):** projected pilot = 18 × 179 s + 11 × 38 × 1.42 s + 716 s (observed) + 0.62 h (**assumed**) = **1.88 node-hours.** Pass.

**Actual allocation** (node002, Xeon Gold 6326, 1 exclusive node):
- run 1 (job 13904): 11 s;
- run 2 (job 13905): 1,329 s;
- **total 0.37 node-hours.**

## 5. Execution defect (run 1)

- **What happened:** run 1 (job 13904, `step4/`) evaluated the identity check through the 32-process chunked objective. Its summation order differs from the recorded value's in the last bit (…177 against …176). The run stopped before any fit, as the frozen plan requires on a mismatch.
- **Diagnostic** (`03_identity_diagnostic.json`): the serial objective reproduces the recorded value exactly, and the doc 18 stage-1 item-refit identity reproduces the pilot record bitwise. The synthetic data are identical.
- **Fix:** the identity check now uses the serial objective (the plan's check, evaluated as defined); nothing else changed. The defective driver and a note are kept (`RUN1_DEFECT_NOTE.txt`, `50_m2a1_fixed_item_recovery.run1_defective.py`).
- **Driver hashes:** run 2 `b24516ae…`; run 1 `ba04e504…`.

## 6. Unresolved and limitations

1. **The fallback risk is deferred, as decided (plan §0.1).**
   - The accuracy-admissible fallback (G = 5) is unaffordable under the current budget (doc 28).
   - It was not needed here: no fallback at any optimum.
   - **Step 7 must stop if the primary fails V2 and no affordable, admissible fallback exists.**
2. **The 0.62 h in every projection is assumed.**
3. **Fixed-item recovery isolates stage 2.** The stage-1 contribution (doc 18: β0_P bias and part of the SD bias under estimated items) is tested only by the full-pipeline recovery with S1-b (step 6).
4. **T3** (doc 26) remains failed under its frozen timing rule.

## 7. Integrity and Git

- After the run (`99_post_run_integrity.txt`): the frozen records of docs 16, 21, 25, 27, 29 and 31 verify; the code, docs, Step 3 results and identity reference are unchanged.
- D1 locks verify; NOT REPRODUCED; the `outputs/` checksums match.
- Branch `main`, HEAD `3a860ca`, 0 tracked changes, 0 staged. **Nothing staged, committed or pushed.**

## 8. Status

**Step 4 is complete and passed.** The following remain unauthorised:
- step 5 (S1-a, diagnostic);
- step 6 (full-pipeline recovery with S1-b, the binding recovery criterion);
- step 7 (real-data B11: V2 at the new ψ̂, sampler, V5);
- step 8 (G1–G7);
- calibration, positive controls and evaluation statistics.

**Stop for review.**
