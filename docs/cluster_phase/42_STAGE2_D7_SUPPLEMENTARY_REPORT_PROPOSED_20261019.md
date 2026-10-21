# Stage 2 M2: D7 (scale-convention diagnostic) — supplementary report: STOPPED BEFORE EXECUTION

**Date:** 2026-10-19.

> **PROPOSED — NOT FROZEN.** Supplement to doc 40, which is unchanged.
> - **Step 6 remains FAILED** (doc 38). **D1 remains NOT REPRODUCED.**
> - D7 was **not executed**. No D7 result exists, and neither prediction has been evaluated.

## 1. What was authorised

D7, strictly as a read-only diagnostic (doc 41), with the instruction:
> "If D7 requires any unapproved computation, changes to the plan, or use of the truth to define the transformation, stop and document the issue instead of proceeding."

## 2. Pre-execution checks

| check | result |
|---|---|
| equations | **pass.** S(ψ) = `relabel_psi(ψ, √v_T, μ_T, √v_P, μ_P)` with μ_s, v_s from the `truth_on_stage1_scale` formula, evaluated at ψ. It reproduces `truth_on_stage1_scale` exactly: S(ψ_raw prior) = ψ_true, μ and sd equal to 1e-12 (toy design) |
| transformation uses only the estimate | **pass.** μ_s, v_s come from ψ̂ and the real design (times, treatment) only. ψ_true enters only the error/z computation and the fixed-point check |
| no refit, no data generation, no change to outputs, thresholds or verdicts | **pass** (by construction: script 55 reads saved outputs and writes only to a new `d7_readonly/`) |
| saved inputs | **pass.** Step 6 ψ̂⁽¹⁾/ψ̂⁽²⁾ with H⁽¹⁾/H⁽²⁾; D5 iterates with H at k = 6; D6 ψ̂ with H; D4 V_full (198 × 198). All manifests verify |
| D5 iterates k = 3–5 | no Hessian saved. The plan's per-estimate SE (i) cannot be applied to them; their errors would be shown in the SE units of ψ̂⁽²⁾, labelled as such. This is an implementation note, not a new computation |
| **plan's built-in numerical checks** ("S(ψ_true) = ψ_true to ≤ 1e-10"; "S(S(ψ)) = S(ψ) to ≤ 1e-10") | **FAIL — cannot be met as specified** (§3) |

## 3. The issue

The frozen model code adds a fixed numerical regulariser to every latent covariance: `m2_grm.covariance` includes `JITTER * I`, with `JITTER = 1e-8`. The plan's moment formula is the `truth_on_stage1_scale` formula, which uses `m2_grm.covariance` and therefore includes the jitter.

Because the jitter does not rescale under a latent relabelling, the convention is satisfied only up to O(JITTER):

| quantity (real design) | value |
|---|---|
| v_T(ψ_true) − 1 | 8.5e-10 |
| v_P(ψ_true) − 1 | 3.4e-9 |
| max \|S(ψ_true) − ψ_true\| | **1.7e-9** (> 1e-10) |
| idempotence at ψ_true | 6.9e-18 |
| idempotence at a generic ψ (toy design, fitted variance far from 1) | 1.4e-9 (> 1e-10) |

The algebra: after scaling by c, the pooled variance becomes (v − J)/c² + J. It equals 1 only when J = 0.

Two of the three new D7 unit tests encode exact invariance and idempotence (tolerance 1e-12). They **fail** for this reason (`tests/test_d7_scale_convention.py`). The Jacobian test passes. They have been left unchanged.

**Size.** The discrepancy is about 1e-9 in ψ units, against standard errors of 0.008–0.096: below 1e-7 SE. It cannot affect either prediction.

**Why D7 was nevertheless stopped.** Meeting the plan would require **changing a pre-stated numerical tolerance** (or the moment formula). Per the authorisation, that is a plan change, so it is documented here instead of being made.

## 4. Options for the reviewer

| option | change | consequence |
|---|---|---|
| **A (recommended)** | Replace the two built-in check tolerances 1e-10 with **1e-8** (= JITTER), and the unit-test tolerances likewise. Justification: an O(JITTER) term in the frozen model code. Nothing else changes | D7 runs exactly as planned. The checks then test the formula to the precision the frozen code allows |
| B | Exclude JITTER from the moment computation | S becomes exactly relabelling-invariant and idempotent. But the formula then differs from `truth_on_stage1_scale` by O(1e-8), and S(ψ_true) − ψ_true ≈ 5e-9 remains, so the fixed-point check still needs a tolerance. Not recommended |
| C | Do not run D7 | the scale-convention question stays open (doc 40, [H1]) |

None of the options touches the model, the frozen code, Step 6 or any saved output.

## 5. State after stopping

- **Not run:** `scripts/55_m2_scale_convention_readonly.py` (`3a6b63d0…`) has not been executed, and no `d7_readonly/` directory exists.
- **Read-only checks only:**
  - unit tests on a toy design (D7 tests: 1 passed, 2 failed for the reason in §3; doc 40's diagnostic tests: 10 passed);
  - the S(ψ_true) fixed-point and idempotence check on the real design (§3).
- **Integrity:**
  - all frozen locks OK;
  - all 195 prior output files, pre-existing docs and code unchanged against `experiments/20261019T084559Z_stage2_step6_diagnostics/00_baseline.json`;
  - the Step 6 and diagnostic manifests (readonly, d4, d5, d6) verify;
  - doc 40 unchanged (`fcf29773…`), doc 41 unchanged (`b03e662e…`).
- **New files since doc 40:**
  - `scripts/55_m2_scale_convention_readonly.py` (`3a6b63d0c9c159b4ae5822944dc90920ed3525044b1f7004bedc3c2c7352d61e`);
  - `tests/test_d7_scale_convention.py` (`9260345c46c24febf8505770422b319e3bb31c5820d22c087e7bb54ef9cc4615`);
  - this document.
- **Git:** HEAD `3a860ca`; 0 tracked changes; nothing staged, committed or pushed.

## 6. What remains true

- The doc 40 findings are unaffected.
- The two D7 predictions remain **untested**:
  - (a) whether the D5 drift collapses. This is largely implied by the invariance of S together with D5's 99.5–99.9 % relabelling share, so it is a weak test;
  - (b) whether the β0_P / Le11_log bias substantially disappears. This is the substantive test.
- No scale constraint, repair, S1-c, recovery experiment or real-data analysis has been implemented or run. Any methodological change still needs its own amendment and a new, independently evaluated recovery test.

**Decision requested:** approve option A (tolerances 1e-10 → 1e-8, justified by `JITTER = 1e-8`) so that D7 can run as planned, or choose B or C.

---

# Addendum (2026-10-19): D7 executed under the proposed amended tolerance (doc 43)

> **PROPOSED — NOT FROZEN.** Everything above this line is the original stop note, unchanged (SHA-256 of the original document: `7491393d14aa4700f4228b9763e076299e7429c75b6daa595e9e3b91d3b88edd`).
> - **The original D7 plan (doc 41) did not pass as written.** Its two built-in checks (≤ 1e-10) fail because of the frozen `JITTER = 1e-8` (§3 above).
> - Execution was authorised by the user under **Option A**, recorded as the proposed amendment doc 43 (`6a33a4b7…`): the two check tolerances and the corresponding unit-test tolerances change from 1e-10 / 1e-12 to **1e-8**. Nothing else changed.
> - **Step 6 remains FAILED. D1 remains NOT REPRODUCED.** D7 is a read-only diagnostic. It is not a recovery test, and it does not re-score Step 6.

## A1. Pre-execution verification (doc 43 §2)

**The tolerance tests the intended identities.** In a throwaway verification process (`JITTER` monkeypatched to 0 there only; no code changed; no saved estimate used), all three identities hold to machine precision without the jitter:

| identity | JITTER = 0 | JITTER = 1e-8 (frozen) |
|---|---|---|
| S(ψ_true) = ψ_true | 2.2e-16 | 1.7e-9 |
| idempotence | 4.4e-16 | 2.6e-9 |
| relabelling invariance | 3.3e-16 | 6.9e-10 |

The whole residual is the jitter term.

**Nothing else changed.** No scientific threshold, model parameter, transformation, estimator or interpretation was changed. Script 55 ran **unchanged** (`3a6b63d0…`). Only the tolerances of the corresponding D7 unit tests changed (`tests/test_d7_scale_convention.py` → `9ae7050e…`); all 3 pass.

## A2. Execution record

**Run 1** (login node) stopped while writing `summary.json`: `Path.relative_to` failed because `--diag` was given as a **relative** path.
- This is an invocation defect. No numerical check, tolerance or setting was involved.
- The CSV outputs had been written. The partial output is kept, uninspected, as `d7_readonly_run1_defective/`, with `D7_RUN1_DEFECT_NOTE.txt`.

**Run 2** used the identical script with an **absolute** `--diag` path. The computation is deterministic, and all five CSV outputs are **byte-identical** to run 1's.

**Output:** `experiments/20261019T084559Z_stage2_step6_diagnostics/d7_readonly/` (`MANIFEST.sha256` verified).

**Built-in checks (amended tolerance 1e-8): all pass.**

| check | value |
|---|---|
| S(ψ_true) − ψ_true | 1.71e-9 |
| S(prior ψ_raw) − ψ_true | 4.4e-16 |
| max idempotence over all 35 saved estimates | 1.81e-9 |
| Jacobian annihilation of the relabelling directions (no tolerance in the plan) | ≤ 1.4e-8 relative |
| integrity: frozen locks, Step 6 outputs, the readonly/d4/d5/d6 manifests, code, script 53, truth record | all OK |

**What was used.** Only the saved Step 6, D4, D5 and D6 estimates and covariances, and the real visit design (times, treatment).
- **No refit, no data generation, no random numbers.**
- **The transformation never uses the truth:** μ_s and v_s come from each estimate. ψ_true enters only the errors and z, and the fixed-point check.

## A3. Prediction (a): does the D5 drift collapse? — **HOLDS**

Range over k = 2…6 (the Step 6 ψ̂⁽²⁾ and the four further S1-b updates), as a maximum over the 19 components:

| dataset | max over 19 components: raw range / raw SE⁽²⁾ | re-expressed range / SE_S(ψ̂⁽²⁾) | β0_P raw → re-expressed | Le00_log raw → re-expressed | Le11_log raw → re-expressed |
|---|---|---|---|---|---|
| 0 | 2.39 | 0.043 | 0.223 → 0.0041 | 1.304 → 0.0002 | 0.642 → 0.0024 |
| 1 | 1.85 | 0.061 | 1.851 → 0.0068 | 1.594 → 0.0005 | 1.804 → 0.0031 |
| 2 | 3.75 | 0.048 | 2.645 → 0.0081 | 3.749 → 0.0003 | 1.726 → 0.0012 |
| 3 | 1.86 | 0.025 | 0.168 → 0.0033 | 1.303 → 0.0002 | 1.090 → 0.0025 |
| 4 | 1.35 | 0.045 | 0.018 → 0.0062 | 0.264 → 0.0004 | 1.348 → 0.0014 |

**Result.** Before re-expression the trajectories span 1.35–3.75 SE. Afterwards they span 0.025–0.061 SE in every dataset, below the plan's indicative 0.1.

**Caveat, as stated in advance.** This is a weak test. S is exactly invariant under joint relabelling, and D5 had already shown 99.5–99.9 % of the drift to be a relabelling. The collapse therefore confirms that the remaining non-relabelling movement is small (≤ 0.06 SE). It is not independent evidence for H1.

## A4. Prediction (b): does the β0_P / Le11_log bias substantially decrease? — **HOLDS** (Le00_log: partly, as predicted)

**The transformation per dataset.** These are the moments of each estimate's fitted latent distribution. Under the truth's convention all four would be (0, 1):

| dataset | estimate | μ_T | v_T | μ_P | v_P | c_T = √v_T | c_P = √v_P | idempotence |
|---|---|---|---|---|---|---|---|---|
| 0 | psi1_step6 | +0.0266 | 0.9787 | +0.0445 | 0.8948 | 0.9893 | 0.9459 | 5.9e-10 |
| 0 | psi2_step6 | +0.0440 | 0.9634 | +0.0487 | 0.9060 | 0.9815 | 0.9518 | 5.2e-10 |
| 0 | d5_k3 | +0.0611 | 0.9480 | +0.0498 | 0.9164 | 0.9737 | 0.9573 | 4.6e-10 |
| 0 | d5_k4 | +0.0781 | 0.9329 | +0.0506 | 0.9268 | 0.9659 | 0.9627 | 3.9e-10 |
| 0 | d5_k5 | +0.0949 | 0.9180 | +0.0515 | 0.9374 | 0.9581 | 0.9682 | 4.5e-10 |
| 0 | d5_k6 | +0.1116 | 0.9032 | +0.0524 | 0.9481 | 0.9504 | 0.9737 | 5.4e-10 |
| 0 | d6_oracle_items | +0.0121 | 0.9795 | +0.0021 | 1.0131 | 0.9897 | 1.0065 | 1.0e-10 |
| 1 | psi1_step6 | +0.0135 | 0.9653 | +0.0142 | 0.8509 | 0.9825 | 0.9225 | 8.8e-10 |
| 1 | psi2_step6 | +0.0162 | 0.9469 | -0.0027 | 0.8260 | 0.9731 | 0.9089 | 1.1e-09 |
| 1 | d5_k3 | +0.0186 | 0.9286 | -0.0219 | 0.8021 | 0.9636 | 0.8956 | 1.2e-09 |
| 1 | d5_k4 | +0.0209 | 0.9105 | -0.0410 | 0.7788 | 0.9542 | 0.8825 | 1.4e-09 |
| 1 | d5_k5 | +0.0233 | 0.8928 | -0.0598 | 0.7563 | 0.9449 | 0.8696 | 1.6e-09 |
| 1 | d5_k6 | +0.0256 | 0.8754 | -0.0784 | 0.7343 | 0.9356 | 0.8569 | 1.8e-09 |
| 1 | d6_oracle_items | +0.0046 | 0.9836 | -0.0201 | 0.9721 | 0.9918 | 0.9860 | 1.4e-10 |
| 2 | psi1_step6 | -0.0235 | 0.9304 | +0.0603 | 0.8808 | 0.9646 | 0.9385 | 6.8e-10 |
| 2 | psi2_step6 | -0.0476 | 0.8903 | +0.0769 | 0.8549 | 0.9435 | 0.9246 | 8.5e-10 |
| 2 | d5_k3 | -0.0712 | 0.8515 | +0.0899 | 0.8292 | 0.9227 | 0.9106 | 1.0e-09 |
| 2 | d5_k4 | -0.0943 | 0.8142 | +0.1026 | 0.8042 | 0.9023 | 0.8968 | 1.2e-09 |
| 2 | d5_k5 | -0.1170 | 0.7785 | +0.1150 | 0.7800 | 0.8824 | 0.8832 | 1.4e-09 |
| 2 | d5_k6 | -0.1391 | 0.7446 | +0.1272 | 0.7565 | 0.8629 | 0.8698 | 1.7e-09 |
| 2 | d6_oracle_items | -0.0258 | 0.9556 | +0.0132 | 0.9751 | 0.9776 | 0.9875 | 2.3e-10 |
| 3 | psi1_step6 | -0.0059 | 0.9885 | +0.0512 | 0.9289 | 0.9942 | 0.9638 | 3.8e-10 |
| 3 | psi2_step6 | -0.0176 | 1.0047 | +0.0594 | 0.9471 | 1.0023 | 0.9732 | 2.8e-10 |
| 3 | d5_k3 | -0.0297 | 1.0203 | +0.0653 | 0.9649 | 1.0101 | 0.9823 | 1.8e-10 |
| 3 | d5_k4 | -0.0419 | 1.0360 | +0.0711 | 0.9830 | 1.0179 | 0.9914 | 1.7e-10 |
| 3 | d5_k5 | -0.0542 | 1.0520 | +0.0769 | 1.0013 | 1.0257 | 1.0007 | 2.5e-10 |
| 3 | d5_k6 | -0.0665 | 1.0682 | +0.0828 | 1.0201 | 1.0335 | 1.0100 | 3.2e-10 |
| 3 | d6_oracle_items | -0.0094 | 1.0203 | +0.0065 | 1.0240 | 1.0101 | 1.0119 | 1.2e-10 |
| 4 | psi1_step6 | -0.0017 | 0.9703 | +0.0299 | 0.8717 | 0.9850 | 0.9337 | 7.4e-10 |
| 4 | psi2_step6 | -0.0093 | 0.9674 | +0.0271 | 0.8509 | 0.9836 | 0.9225 | 8.8e-10 |
| 4 | d5_k3 | -0.0170 | 0.9643 | +0.0218 | 0.8314 | 0.9820 | 0.9118 | 1.0e-09 |
| 4 | d5_k4 | -0.0247 | 0.9612 | +0.0165 | 0.8122 | 0.9804 | 0.9012 | 1.2e-09 |
| 4 | d5_k5 | -0.0324 | 0.9580 | +0.0112 | 0.7936 | 0.9788 | 0.8908 | 1.3e-09 |
| 4 | d5_k6 | -0.0400 | 0.9549 | +0.0060 | 0.7753 | 0.9772 | 0.8805 | 1.4e-09 |
| 4 | d6_oracle_items | -0.0012 | 1.0019 | -0.0043 | 0.9851 | 1.0010 | 0.9925 | 7.6e-11 |

**Before and after, per dataset.**
- "raw" is the saved estimate with its frozen fixed-item SE.
- "re-expressed" is S(ψ̂) with SE_S from G H⁻¹ Gᵀ.
- \* marks |z| > 1.96.

| dataset | estimate | param | raw estimate (SE) | raw z | re-expressed estimate (SE_S) | re-expressed z |
|---|---|---|---|---|---|---|
| 0 | psi2_step6 | beta0_P | -0.3963 (0.0290) | +2.78* | -0.4676 (0.0223) | +0.41 |
| 1 | psi2_step6 | beta0_P | -0.4363 (0.0275) | +1.47 | -0.4772 (0.0221) | -0.02 |
| 2 | psi2_step6 | beta0_P | -0.3482 (0.0286) | +4.49* | -0.4597 (0.0210) | +0.81 |
| 3 | psi2_step6 | beta0_P | -0.4272 (0.0300) | +1.66 | -0.5000 (0.0214) | -1.09 |
| 4 | psi2_step6 | beta0_P | -0.4245 (0.0288) | +1.82 | -0.4895 (0.0210) | -0.61 |
| 0 | psi2_step6 | Le00_log | -0.7811 (0.0247) | -1.77 | -0.7624 (0.0298) | -0.84 |
| 1 | psi2_step6 | Le00_log | -0.7751 (0.0246) | -1.53 | -0.7478 (0.0293) | -0.36 |
| 2 | psi2_step6 | Le00_log | -0.8088 (0.0238) | -3.00* | -0.7507 (0.0292) | -0.45 |
| 3 | psi2_step6 | Le00_log | -0.7168 (0.0235) | +0.88 | -0.7191 (0.0286) | +0.64 |
| 4 | psi2_step6 | Le00_log | -0.8149 (0.0247) | -3.14* | -0.7983 (0.0295) | -2.06* |
| 0 | psi2_step6 | Le11_log | -0.9725 (0.0355) | -1.81 | -0.9232 (0.0379) | -0.40 |
| 1 | psi2_step6 | Le11_log | -0.9881 (0.0327) | -2.45* | -0.8925 (0.0352) | +0.44 |
| 2 | psi2_step6 | Le11_log | -1.0270 (0.0355) | -3.35* | -0.9486 (0.0376) | -1.07 |
| 3 | psi2_step6 | Le11_log | -0.9179 (0.0342) | -0.29 | -0.8907 (0.0364) | +0.48 |
| 4 | psi2_step6 | Le11_log | -0.9995 (0.0346) | -2.64* | -0.9188 (0.0368) | -0.29 |
| 0 | psi1_step6 | beta0_P | -0.3991 (0.0294) | +2.64* | -0.4689 (0.0222) | +0.35 |
| 1 | psi1_step6 | beta0_P | -0.4276 (0.0285) | +1.72 | -0.4790 (0.0219) | -0.10 |
| 2 | psi1_step6 | beta0_P | -0.3733 (0.0296) | +3.50* | -0.4620 (0.0209) | +0.71 |
| 3 | psi1_step6 | beta0_P | -0.4319 (0.0301) | +1.49 | -0.5013 (0.0213) | -1.15 |
| 4 | psi1_step6 | beta0_P | -0.4288 (0.0297) | +1.62 | -0.4913 (0.0209) | -0.70 |
| 0 | psi1_step6 | Le00_log | -0.7737 (0.0247) | -1.47 | -0.7630 (0.0298) | -0.86 |
| 1 | psi1_step6 | Le00_log | -0.7663 (0.0247) | -1.17 | -0.7487 (0.0293) | -0.38 |
| 2 | psi1_step6 | Le00_log | -0.7877 (0.0239) | -2.10* | -0.7516 (0.0292) | -0.48 |
| 3 | psi1_step6 | Le00_log | -0.7258 (0.0236) | +0.49 | -0.7201 (0.0286) | +0.61 |
| 4 | psi1_step6 | Le00_log | -0.8146 (0.0247) | -3.12* | -0.7995 (0.0296) | -2.10* |
| 0 | psi1_step6 | Le11_log | -0.9820 (0.0358) | -2.06* | -0.9264 (0.0381) | -0.48 |
| 1 | psi1_step6 | Le11_log | -0.9737 (0.0328) | -2.00* | -0.8930 (0.0353) | +0.43 |
| 2 | psi1_step6 | Le11_log | -1.0113 (0.0355) | -2.91* | -0.9478 (0.0377) | -1.05 |
| 3 | psi1_step6 | Le11_log | -0.9297 (0.0343) | -0.63 | -0.8928 (0.0365) | +0.42 |
| 4 | psi1_step6 | Le11_log | -0.9880 (0.0347) | -2.30* | -0.9194 (0.0369) | -0.30 |
| 0 | d5_k6 | beta0_P | -0.4028 (0.0296) | +2.50* | -0.4675 (0.0223) | +0.42 |
| 1 | d5_k6 | beta0_P | -0.4872 (0.0258) | -0.40 | -0.4770 (0.0221) | -0.01 |
| 2 | d5_k6 | beta0_P | -0.2725 (0.0269) | +7.60* | -0.4596 (0.0210) | +0.82 |
| 3 | d5_k6 | beta0_P | -0.4221 (0.0311) | +1.76 | -0.5000 (0.0214) | -1.08 |
| 4 | d5_k6 | beta0_P | -0.4249 (0.0275) | +1.89 | -0.4894 (0.0210) | -0.60 |
| 0 | d5_k6 | Le00_log | -0.8133 (0.0247) | -3.07* | -0.7624 (0.0298) | -0.84 |
| 1 | d5_k6 | Le00_log | -0.8144 (0.0246) | -3.12* | -0.7478 (0.0293) | -0.36 |
| 2 | d5_k6 | Le00_log | -0.8981 (0.0238) | -6.74* | -0.7507 (0.0292) | -0.45 |
| 3 | d5_k6 | Le00_log | -0.6862 (0.0235) | +2.18* | -0.7191 (0.0286) | +0.64 |
| 4 | d5_k6 | Le00_log | -0.8214 (0.0247) | -3.40* | -0.7983 (0.0295) | -2.06* |
| 0 | d5_k6 | Le11_log | -0.9497 (0.0355) | -1.17 | -0.9231 (0.0379) | -0.39 |
| 1 | d5_k6 | Le11_log | -1.0470 (0.0327) | -4.25* | -0.8926 (0.0352) | +0.44 |
| 2 | d5_k6 | Le11_log | -1.0882 (0.0355) | -5.08* | -0.9486 (0.0376) | -1.08 |
| 3 | d5_k6 | Le11_log | -0.8807 (0.0341) | +0.80 | -0.8907 (0.0364) | +0.48 |
| 4 | d5_k6 | Le11_log | -1.0461 (0.0346) | -3.99* | -0.9188 (0.0368) | -0.29 |
| 0 | d6_oracle_items | beta0_P | -0.4685 (0.0306) | +0.27 | -0.4675 (0.0223) | +0.41 |
| 1 | d6_oracle_items | beta0_P | -0.4904 (0.0297) | -0.46 | -0.4770 (0.0221) | -0.01 |
| 2 | d6_oracle_items | beta0_P | -0.4405 (0.0305) | +1.19 | -0.4595 (0.0210) | +0.82 |
| 3 | d6_oracle_items | beta0_P | -0.4993 (0.0311) | -0.72 | -0.4999 (0.0214) | -1.08 |
| 4 | d6_oracle_items | beta0_P | -0.4898 (0.0309) | -0.42 | -0.4892 (0.0211) | -0.59 |
| 0 | d6_oracle_items | Le00_log | -0.7728 (0.0247) | -1.43 | -0.7624 (0.0298) | -0.84 |
| 1 | d6_oracle_items | Le00_log | -0.7561 (0.0246) | -0.76 | -0.7478 (0.0293) | -0.35 |
| 2 | d6_oracle_items | Le00_log | -0.7734 (0.0238) | -1.51 | -0.7507 (0.0292) | -0.45 |
| 3 | d6_oracle_items | Le00_log | -0.7092 (0.0235) | +1.20 | -0.7192 (0.0286) | +0.64 |
| 4 | d6_oracle_items | Le00_log | -0.7972 (0.0247) | -2.43* | -0.7982 (0.0295) | -2.06* |
| 0 | d6_oracle_items | Le11_log | -0.9167 (0.0355) | -0.24 | -0.9232 (0.0379) | -0.40 |
| 1 | d6_oracle_items | Le11_log | -0.9067 (0.0327) | +0.04 | -0.8926 (0.0352) | +0.44 |
| 2 | d6_oracle_items | Le11_log | -0.9614 (0.0355) | -1.50 | -0.9488 (0.0376) | -1.08 |
| 3 | d6_oracle_items | Le11_log | -0.8785 (0.0341) | +0.87 | -0.8904 (0.0364) | +0.49 |
| 4 | d6_oracle_items | Le11_log | -0.9263 (0.0346) | -0.52 | -0.9188 (0.0368) | -0.29 |

**Summary over the five datasets.** Mean errors are in ψ units. Counts are descriptive, **NOT a recovery test**.

| estimate | param | raw mean error | re-expressed mean error | raw mean z | re-expressed mean z | raw count | re-expressed count |
|---|---|---|---|---|---|---|---|
| psi1_step6 | beta0_P | +0.0646 | -0.0037 | +2.19 | -0.18 | 3/5 | 5/5 |
| psi1_step6 | Le00_log | -0.0362 | -0.0191 | -1.47 | -0.64 | 3/5 | 4/5 |
| psi1_step6 | Le11_log | -0.0688 | -0.0077 | -1.98 | -0.20 | 1/5 | 5/5 |
| psi2_step6 | beta0_P | +0.0703 | -0.0020 | +2.44 | -0.10 | 3/5 | 5/5 |
| psi2_step6 | Le00_log | -0.0419 | -0.0182 | -1.71 | -0.61 | 3/5 | 4/5 |
| psi2_step6 | Le11_log | -0.0729 | -0.0066 | -2.11 | -0.17 | 2/5 | 5/5 |
| psi2_step6_fullchainV | beta0_P | +0.0703 | -0.0020 | +2.19 | -0.10 | 3/5 | 5/5 |
| psi2_step6_fullchainV | Le00_log | -0.0419 | -0.0182 | -1.02 | -0.61 | 5/5 | 4/5 |
| psi2_step6_fullchainV | Le11_log | -0.0729 | -0.0066 | -1.39 | -0.17 | 4/5 | 5/5 |
| d5_k6 | beta0_P | +0.0749 | -0.0019 | +2.67 | -0.09 | 3/5 | 5/5 |
| d5_k6 | Le00_log | -0.0692 | -0.0182 | -2.83 | -0.61 | 0/5 | 4/5 |
| d5_k6 | Le11_log | -0.0942 | -0.0066 | -2.74 | -0.17 | 2/5 | 5/5 |
| d6_oracle_items | beta0_P | -0.0009 | -0.0019 | -0.03 | -0.09 | 5/5 | 5/5 |
| d6_oracle_items | Le00_log | -0.0243 | -0.0182 | -0.98 | -0.61 | 4/5 | 4/5 |
| d6_oracle_items | Le11_log | -0.0098 | -0.0066 | -0.27 | -0.17 | 5/5 | 5/5 |

**Result (point estimates, not SE artefacts).** For ψ̂⁽²⁾, the Step 6 binding estimate:

| component | raw mean error | re-expressed mean error | raw mean z | re-expressed mean z | raw count | re-expressed count |
|---|---|---|---|---|---|---|
| β0_P | +0.070 | **−0.002** | +2.44 | **−0.10** | 3/5 | **5/5** |
| Le11_log | −0.073 | **−0.007** | −2.11 | **−0.17** | 2/5 | **5/5** |
| Le00_log | −0.042 | −0.018 | −1.71 | −0.61 | 3/5 | 4/5 |

- The movements, −2.54 and +1.94 SE, match the doc 41 prediction from the D4 affine parts (+2.47 and −1.97).
- **Le00_log is only partly corrected**, as predicted: its non-affine tremor-item component remains. Dataset 4 is still outside ±1.96 (z −2.06).
- The same holds for ψ̂⁽¹⁾ (β0_P −0.18, Le11_log −0.20), for the D5 k = 6 iterate, and for the D6 true-latent-item fits. After re-expression, **all four estimate sets agree**: mean z β0_P −0.09 to −0.18, Le11_log −0.17 to −0.20, Le00_log −0.61 to −0.64. **Their raw differences were relabellings.**

**All 19 components** (ψ̂⁽²⁾; frozen V, with the D4 full-chain V in brackets):

| component | raw mean z | re-expressed mean z | raw count | re-expressed count | SE_S/SE (mean) |
|---|---|---|---|---|---|
| beta0_T | -0.31 (-0.27) | -0.58 (-0.58) | 4/5 (4/5) | 4/5 (4/5) | 0.65 |
| beta1_T | +0.02 (+0.02) | +0.18 (+0.19) | 5/5 (5/5) | 4/5 (4/5) | 1.02 |
| gamma_T | +0.32 (+0.31) | +0.40 (+0.39) | 5/5 (5/5) | 5/5 (5/5) | 1.03 |
| beta0_P | +2.44 (+2.19) | -0.10 (-0.10) | 3/5 (3/5) | 5/5 (5/5) | 0.75 |
| beta1_P | -0.52 (-0.49) | +0.57 (+0.57) | 5/5 (5/5) | 5/5 (5/5) | 1.00 |
| gamma_P | -0.90 (-0.88) | -0.56 (-0.56) | 4/5 (4/5) | 4/5 (4/5) | 1.08 |
| L00_log | -0.72 (-0.64) | -0.00 (+0.02) | 5/5 (5/5) | 4/5 (4/5) | 0.77 |
| L10 | +0.29 (+0.28) | +0.33 (+0.31) | 4/5 (4/5) | 4/5 (4/5) | 1.02 |
| L11_log | -0.55 (-0.56) | -0.28 (-0.31) | 5/5 (5/5) | 5/5 (5/5) | 0.98 |
| L20 | +0.33 (+0.33) | +0.47 (+0.47) | 5/5 (5/5) | 5/5 (5/5) | 1.07 |
| L21 | -0.14 (-0.14) | -0.14 (-0.14) | 4/5 (4/5) | 4/5 (4/5) | 1.07 |
| L22_log | -1.41 (-1.19) | +0.20 (+0.20) | 4/5 (4/5) | 5/5 (5/5) | 0.87 |
| L30 | -0.17 (-0.17) | -0.17 (-0.17) | 5/5 (5/5) | 5/5 (5/5) | 1.07 |
| L31 | -0.43 (-0.43) | -0.43 (-0.43) | 5/5 (5/5) | 5/5 (5/5) | 1.07 |
| L32 | -0.61 (-0.60) | -0.53 (-0.52) | 5/5 (5/5) | 5/5 (5/5) | 1.07 |
| L33_log | -0.88 (-0.81) | +0.02 (+0.03) | 5/5 (5/5) | 5/5 (5/5) | 0.95 |
| Le00_log | -1.71 (-1.02) | -0.61 (-0.61) | 3/5 (5/5) | 4/5 (4/5) | 1.21 |
| Le10 | +0.17 (+0.16) | +0.33 (+0.33) | 5/5 (5/5) | 5/5 (5/5) | 1.07 |
| Le11_log | -2.11 (-1.39) | -0.17 (-0.17) | 2/5 (4/5) | 5/5 (5/5) | 1.07 |

- After re-expression every component has a descriptive count ≥ 4/5.
- L22_log and L33_log also move to near 0 (+0.20, +0.02).
- Some components with small raw errors change slightly: β0_T −0.31 → −0.58; β1_P −0.52 → +0.57; L00_log 5/5 → 4/5; β1_T 5/5 → 4/5.

**Item uncertainty is almost entirely scale uncertainty [O].** On the raw scale the D4 full-chain SE exceeded the frozen SE by ×1.11–1.65. After re-expression the two agree to within 0.92–1.12 for every component and dataset: β0_P 0.98, Le00_log 1.02, Le11_log 1.02. The extra uncertainty from estimating the items lies almost entirely along the relabelling directions that the convention fixes.

## A5. What D7 establishes, and what it cannot

**Establishes (descriptively, on these five datasets):**
1. The Step 6 failure of β0_P and Le11_log disappears when the estimates are expressed on the convention that defines the truth. Correspondingly, the errors are (to within sampling noise) an **unidentified relabelling of the latent scale**, not stage-2 or item-shape estimation error. This supports H-A / [H1] of doc 40.
2. Step 6, D5 and the D6 counterfactual differ from each other **only by relabelling**.
3. Most of the understated uncertainty found by D4 is scale uncertainty, not shape uncertainty.
4. Le00_log keeps a smaller, separate error (mean z −0.61; 4/5), consistent with H-F (tremor-item, non-affine).

**Cannot establish:**
- **It is not a recovery test.** It uses the same five datasets that motivated the hypothesis, so it is hypothesis-strengthening only. It is not a validation of any procedure.
- **It does not change the Step 6 verdict (FAILED).** Step 6 was evaluated without re-standardisation (approved decision E-2). Re-expression is a different convention, adopted here only as a diagnostic.
- **It does not show that a re-standardised estimator would have correct coverage in general,** under other truths or designs, or with intervals that also account for the remaining shape uncertainty. A pre-specified, independent recovery test would be needed.
- **It says nothing about real PPMI data, model fit to real data, subtypes or D1.**
- **The convention is a definition, not a fact about the data.** It matches the truth here because the truth was constructed with it (`truth_on_stage1_scale`).

## A6. Integrity after D7

- **Frozen records and prior outputs:** all frozen locks OK. The Step 6 and diagnostic manifests (readonly, d4, d5, d6, d7_readonly) verify. All 195 prior output files, pre-existing docs and code are unchanged against `00_baseline.json`.
- **Docs:** doc 40 (`fcf29773…`) and doc 41 (`b03e662e…`) are unchanged. This document's original text is preserved verbatim above.
- **Tests:** 27 pass (D7 3, diagnostics 10, Step 6 driver 11, Step 5 3).
- **Git:** HEAD `3a860ca`; 0 tracked changes; nothing staged, committed or pushed.

## A7. Recommended next research decision (for review; not taken)

**D7 supports scale-convention mismatch** as the main cause of the β0_P / Le11_log failure, with a smaller separate tremor-item component for Le00_log. **The smallest defensible next step is a decision, not a computation:**

1. **Whether to draft a dated amendment (Rev 3)** to the Stage 2 estimation track. It would specify:
   - **(a) the identification convention.** Report ψ̂ re-expressed to the pooled-moment convention (pooled visit-level latent mean 0, variance 1 over the analysis design), which is the convention the protocol already uses to define the truth on the "stage-1 scale";
   - **(b) the interval method.** The delta method through that re-expression. Whether to also propagate item uncertainty (D4 shows it is small after re-expression: ×0.92–1.12), and whether that changes doc 16 B5;
   - **(c) the role of S1-b.** Under re-expression S1-b is nearly irrelevant: all estimate sets agree after D7. Its updates are unanchored (D5);
   - **(d) the Le00_log residual,** which may remain marginal (4/5 here).
2. **The design of a new, frozen, independent recovery test** for that amended procedure, decided **before** it is run. **Fresh seeds are recommended:** these five datasets have now been used for diagnosis. Also to decide: the number of datasets and the criterion (the unchanged ≥ 4/5 rule, or a larger study).

Steps 7–8, calibration and real-data work remain blocked until such a test is passed and reviewed. No repair, scale constraint, S1-c or new recovery test has been implemented.
