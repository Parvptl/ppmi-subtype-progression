# Stage 2 M2, Step 3 report: likelihood accuracy, quadrature order and mode validation

**Date:** 2026-10-19.

**Execution plan:** `29_STAGE2_STEP3_EXECUTION_PLAN_20261019.md`, SHA-256 `feb125b0df0d32653d50a328fd736d59b64a0fd50888f0c143def012a60fc91d`.
- Frozen before any computation, with a read-only copy in `_frozen/`.
- It includes the reviewer's resolutions of four protocol gaps (R-a to R-d; plan §1).

**Execution directory:** `experiments/20261018T180500Z_stage2_step3_accuracy/`. The results are in **`step3_run2/`** (job 13903).

**Boundaries.** This is likelihood **evaluation** at frozen parameters on the 30 designated patients, plus synthetic dataset 0 for V3.
- No parameter was estimated.
- No recovery, real-data B11, sampler, G1–G7, calibration, positive controls or evaluation statistics were run.
- D1 remains **NOT REPRODUCED**.

## 1. Outcome

| criterion (frozen; plan §4) | result | status |
|---|---|---|
| (a′) Q_ref adequacy: \|ℓ_20 − ℓ_25\|, 5 most-visit patients × 4 ψ, ≤ 0.001 | max 6.6e-5 | **pass** (Q_ref stays 20) |
| (a) ε-order: smallest Q ∈ {7, 10, 15} with max over 30 patients × 4 ψ of \|ℓ_Q − ℓ_20\| ≤ 0.005 | Q = 7: 0.0286 ✗; Q = 10: 0.00537 ✗; **Q = 15: 0.00067 ✓** | **pass, Q* = 15** |
| **V2:** max_i \|ℓ_A(Q*) − ℓ_D3\| ≤ 0.05 at ψ̂, 30 patients | **max 0.0321** (mean 0.0077, median 0.0064 — descriptive); 0 non-finite; 0 reference-limited | **PASS** |
| §2.5 mode uniqueness: 3 starts within 1e-6, 30 patients, ψ̂, Q* | max difference 8.9e-16; 0 disagreements; all modes negative definite | **PASS** |
| D3 reproducibility at ψ̂ (plan §3) | identical to the pilot's recorded reference values, all 30 | pass |
| (b′) G_ref adequacy (R-c): \|ℓ_G9 − ℓ_G7\|, 5 patients × 4 ψ, ≤ 0.001 | max 1.6e-5 | pass (G = 7 is a valid reference) |
| (b) b-order: smallest G ∈ {3, 5} with max \|ℓ_G − ℓ_G7\| ≤ 0.005 | **G = 3: 0.00906 ✗**; **G = 5: 0.00070 ✓** | **G* = 5.** Binding only if the fallback is invoked |
| V2-F (fallback against D3; binding only if V2 fails) | G = 3: 0.0111; G = 5: 0.0122 | both ≤ 0.05; not binding (V2 passed) |
| V3 (reported) | argmax σ_P = 0.39984 (Q = 15), 0.39983 (Q = 20); within 0.0134 of 0.3973 | reported |
| V4 (reported) | Σ errors over the 30 patients ≈ −0.20 to −0.22 at the 4 ψ (MC SE about 0.023) | reported |

**Step 3 passes** (plan §4: (a) selects Q*, and V2 and mode uniqueness pass).

**Selected configuration:**
- **primary: Laplace over b with per-visit Q = 15** (Q_ref = 20);
- **fallback order required by the accuracy rule: G = 5.** G = 3 fails the 0.005 order tolerance (0.0091), which confirms the earlier synthetic indication.

## 2. Records, environment and resources

**Hashes** (pre- and post-run, `00_baseline.json`, `99_post_run_integrity.txt`):
- docs 16, 20, 21, 24–28 and the frozen records of 16, 21, 25, 27 and 29: unchanged;
- `m2_quad.py` `50f164c6…`, `m2_grm.py` `585c878b…`, `grm.py`, the three references and the pilot script: unchanged;
- inputs unchanged: pilot `step2_items.json` `5fb6b08d…`, `step3_longitudinal.json` `427bd95a…`, `step4_laplace.json` `6920a105…`, doc 18 fixed-item results;
- panel MD5 `fb038ddb…`.

**Driver** `scripts/49_m2a1_checks.py`: `0340b7d6…` (run 2).

**Designated patients:** reproduced from seed `[42, 45, 4]`, identical to the pilot's set. The PATNOs are in the plan.

**Compute:**
- node002 (Intel Xeon Gold 6326), 1 exclusive node, 32 processes, single-threaded BLAS; Python 3.13.16, NumPy 2.2.4.
- Run 1 (job 13902): 16 s. Run 2 (job 13903): 116 s. **Actual allocation 0.037 node-hours.**
- By stage (run 2): D3 6 s; order (a) 1 s; V2 4 s; order (b) 74 s; modes 0.3 s; V3 21 s.

**Execution defect (run 1, job 13902, `step3/`):**
- S2 raised an exception while writing its CSV (only the 5 top patients have a Q = 25 column), and the driver mislabelled the crash as "order (a) selected no Q".
- **No order, V2 or mode result was produced.**
- Fixed with a union-of-columns writer and explicit crash reporting; the scientific logic is unchanged. Recorded in `RUN1_DEFECT_NOTE.txt` with the defective driver copy. Run 2 is the result.

**Smoke runs** (login node, 4 patients, `smoke_codepath_1`/`_2`): code path only; not results.

## 3. V2: all 30 patients

ψ̂ = pilot estimate; items = pilot estimates; Q* = 15.
- e = A(15) − D3. D3 is AGQ7 over the full θ for 2n ≤ 6 (12 patients), otherwise importance sampling with 20,000 draws and seed `[42, 45, 4, i]` (18 patients).
- The full table, including the per-Q errors, is in `step3_run2/S3_v2_patients.csv`.

| # | PATNO | visits | D3 method | ℓ_A(15) | ℓ_D3 | e | MC SE |
|---|---|---|---|---|---|---|---|
| 0 | 4115 | 9 | IS | −99.4606 | −99.4670 | +0.0064 | 0.0060 |
| 1 | 4102 | 8 | IS | −69.6590 | −69.6433 | −0.0157 | 0.0059 |
| 2 | 177395 | 4 | IS | −42.1117 | −42.1041 | −0.0077 | 0.0040 |
| 3 | 311402 | 3 | AGQ7 | −47.9696 | −47.9692 | −0.0004 | 0 |
| 4 | 3254 | 3 | AGQ7 | −39.9292 | −39.9199 | −0.0093 | 0 |
| 5 | 116529 | 2 | AGQ7 | −28.1537 | −28.1435 | −0.0103 | 0 |
| 6 | 266248 | 3 | AGQ7 | −46.3749 | −46.3707 | −0.0042 | 0 |
| 7 | 4123 | 11 | IS | −128.7154 | −128.7095 | −0.0059 | 0.0068 |
| 8 | 4109 | 6 | IS | −74.4740 | −74.4645 | −0.0096 | 0.0058 |
| 9 | 143116 | 3 | AGQ7 | −36.6861 | −36.6763 | −0.0099 | 0 |
| 10 | 238181 | 3 | AGQ7 | −38.7783 | −38.7760 | −0.0023 | 0 |
| 11 | 149940 | 3 | AGQ7 | −26.9709 | −26.9547 | −0.0162 | 0 |
| 12 | 3815 | 10 | IS | −162.4412 | −162.4405 | −0.0006 | 0.0062 |
| 13 | 137264 | 5 | IS | −59.2392 | −59.2298 | −0.0093 | 0.0047 |
| 14 | 101675 | 4 | IS | −51.4250 | −51.4242 | −0.0007 | 0.0042 |
| 15 | 3415 | 12 | IS | −175.4623 | −175.4658 | +0.0034 | 0.0068 |
| 16 | 100952 | 5 | IS | −80.9238 | −80.9165 | −0.0072 | 0.0047 |
| **17** | **4026** | **2** | **AGQ7** | **−22.6506** | **−22.6185** | **−0.0321 (max)** | 0 |
| 18 | 130013 | 5 | IS | −18.0589 | −18.0526 | −0.0063 | 0.0049 |
| 19 | 3467 | 10 | IS | −119.5938 | −119.5934 | −0.0003 | 0.0063 |
| 20 | 3253 | 2 | AGQ7 | −22.3099 | −22.3044 | −0.0055 | 0 |
| 21 | 3023 | 4 | IS | −51.7588 | −51.7408 | −0.0180 | 0.0041 |
| 22 | 3108 | 10 | IS | −130.2993 | −130.3011 | +0.0018 | 0.0062 |
| 23 | 3052 | 8 | IS | −113.0982 | −113.1009 | +0.0026 | 0.0057 |
| 24 | 3659 | 3 | AGQ7 | −29.1781 | −29.1660 | −0.0121 | 0 |
| 25 | 168299 | 3 | AGQ7 | −48.3943 | −48.3894 | −0.0050 | 0 |
| 26 | 148699 | 4 | IS | −51.3784 | −51.3756 | −0.0028 | 0.0040 |
| 27 | 182340 | 4 | IS | −40.4352 | −40.4367 | +0.0015 | 0.0042 |
| 28 | 101038 | 5 | IS | −61.6425 | −61.6288 | −0.0137 | 0.0046 |
| 29 | 215045 | 3 | AGQ7 | −42.7750 | −42.7653 | −0.0098 | 0 |

**Summary:**
- max |e| = **0.0321**; mean 0.0077; median 0.0064.
- All returned modes were negative definite: 0 fallback-evaluated patients, 0 budget triggers, 0 non-finite values.
- **Reference-limited (R-c i):** 0. The maximum is attained at an AGQ7 (deterministic) reference.
- **Descriptive** max |e| by Q: 0.0291 (Q = 7), 0.0317 (10), 0.0321 (15), 0.0320 (20). The remaining error is the Laplace-over-b approximation, not the ε quadrature.
- **Fallback against D3, descriptive** (V2 passed, so not binding): max |e| 0.0111 (G = 3), 0.0122 (G = 5).

## 4. Order checks: patient level

- **(a)** at 4 ψ: ψ̂ and 3 perturbations, N(0, 0.05²), seeds `[42, 46, 7, k]`. σ_T / σ_P = 0.353 / 0.207 (ψ̂), 0.389 / 0.189, 0.377 / 0.235, 0.332 / 0.225. Every value is in `S2_order_a.csv`. Max over 120 patient-ψ pairs: Q = 7 0.0286; Q = 10 **0.00537** (narrowly above 0.005); **Q = 15 0.00067**.
- **(a′):** 5 most-visit patients (12, 11, 10, 10 and 10 visits): max |ℓ_20 − ℓ_25| = 6.6e-5.
- **(b)** at Q* = 15 and the same 4 ψ (`S4_order_b.csv`):
  - G = 5 against G = 7: max 0.00070;
  - **G = 3: max 0.00906.** It exceeds 0.005 for 3 of 30 patients (max over ψ: 0.0082, 0.0091, 0.0062), at 2–3 visits.
- **(b′):** |ℓ_G9 − ℓ_G7| max 1.6e-5.
- **Counts across all order evaluations:** 0 budget triggers, 0 non-ND, 0 errors.

## 5. Mode and numerical-stability checks

- **Uniqueness:** for all 30 patients, the starts 0, b̂ − 2 SD and b̂ + 2 SD reached the same mode (max component difference 8.9e-16 ≤ 1e-6).
- **Hessians:** all modes from every start were negative definite.
- **Other counts:** no penalised evaluations and no budget triggers. There is no optimiser in Step 3, so no penalised trials exist.
- **Gradients:** not used in Step 3, which evaluates values only. The `gradient_inexact` flag does not apply.

## 6. V3, V4 and parameter-level effect (reported, not gated)

- **V3** (synthetic dataset 0, the doc 19 slice): argmax σ_P 0.39984 at Q = 15, 0.39983 at Q = 20. That is within the 0.0134 band of the D3 slice argmax 0.3973. The Laplace-over-b likelihood is no longer biased low on σ_P the way the 2n-dimensional Laplace was (0.3796, doc 19). The Q = 15 vs Q = 20 change is 1.3e-5.
- **V4:** Σ_i (A(15) − D3) over the 30 patients at the 4 ψ is −0.199, −0.206, −0.212 and −0.217 (MC SE about 0.023). There is a small level offset and no material trend in σ_T or σ_P.
- **Parameter effect** (Q = 15 vs Q = 20): the change in total-log-likelihood differences between ψ values is ≤ 0.0013 in absolute value.

## 7. Runtime and budget

**Actual:** 0.037 node-hours (jobs 13902 and 13903).

**Primary at Q* = 15:**
- doc 26 (32 processes) gives an **estimated** fit time of 214 s (scaled from the measured Q = 10 fit) and a projected pilot of **1.99 node-hours**;
- per-fit ≤ 60 min: **pass (estimate)**; pilot ≤ 24 h: **pass (estimate)**.
- No fit was measured at Q = 15. The projections include 0.62 h of assumed components.

**Fallback at the required G* = 5:**
- doc 28 measured, at Q = 10, warm fits of 83–104 min and a pilot of 37.4 node-hours: **fail.**
- At Q = 15 the cost is higher still [estimate: about Q² scaling].
- **The accuracy-admissible fallback is unaffordable under the current budget.**
- Under Rev 2 the fallback is invoked only if V2 fails for the primary. V2 passed, so **the fallback is not invoked at Step 3.** It remains unavailable for later steps: real-data B11 (step 7) repeats V2 at the new ψ̂. If the primary failed there, no affordable admissible fallback exists.

## 8. Numerical accuracy vs scientific validity

- **Shown (numerical, at these frozen inputs):** the Laplace-over-b likelihood at Q = 15 is within 0.0321 of the D3 integral for the 30 designated patients at the pilot ψ̂; the ε-quadrature at Q = 15 is converged to 6.7e-4; the modes are unique and regular.
- **Not shown:** recovery performance; real-data fit quality (B11); the sampler; fidelity (G1–G7); calibration; anything scientific, including subtypes or an intermediate state.
- **ψ̂ is the pilot's estimate** from the earlier 2n-dimensional Laplace fit, as the frozen V2 prescribes. Step 7 repeats V2 at the new ψ̂.

## 9. Unresolved and limitations

1. **No affordable fallback at the admissible G (= 5).** This matters only if the primary later fails V2.
2. **The primary runtime at Q = 15 is an estimate** (scaled), not a measured fit.
3. **The largest V2 error (0.032)** arises for 2-visit patients, where the Laplace-over-b approximation is least accurate. It is within tolerance.
4. **T3** (doc 26) remains failed under its frozen 10 % timing rule.
5. **R-a to R-d are reviewer resolutions of gaps** in the frozen text. They are recorded in the plan.

## 10. Integrity and Git

- After the run, all frozen records verify (docs 16, 21, 25, 27, 29); docs 20 and 24–28, the code, the references and the inputs are unchanged.
- D1 locks verify, and the verdict is NOT REPRODUCED. The `outputs/` checksums match.
- Branch `main`, HEAD `3a860ca`, 0 tracked changes, 67 untracked files. **Nothing staged, committed or pushed.**

## 11. Status

**Step 3 is complete and passed.** The next steps in Rev 2 §6 need separate authorisation and remain unauthorised:
- step 4, fixed-item recovery;
- step 5, S1-a;
- step 6, full-pipeline recovery with S1-b;
- step 7, real-data B11 (including V2 at the new ψ̂, the sampler and V5);
- step 8, G1–G7.

So do calibration, positive controls and evaluation statistics. **Stop for review.**
