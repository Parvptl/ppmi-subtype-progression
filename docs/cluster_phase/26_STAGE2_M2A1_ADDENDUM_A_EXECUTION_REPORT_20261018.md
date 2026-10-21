# Stage 2 M2: Addendum A execution report (implementation, tests, benchmarks)

**Date:** 2026-10-18.

**Specification:** `25_STAGE2_AMENDMENT1_REV2_ADDENDUM_A_FINAL_20261017.md`, SHA-256 `439cf0847ae81d6c5d28516e5376c0710b6fc9a89fbd92347a0b55d13ae778ad`. It was frozen before any code change, and is the implementation specification authorised by the user's execution instruction of 2026-10-17. It does not claim independent human approval.

**Execution directory:** `experiments/20261017T130820Z_stage2_m2a1_addendumA/`. It contains:
- `00_baseline.json`;
- the T1 and T5 test logs;
- `bench/`: per-stage JSON and `summary.json`;
- the SLURM logs and `job13838_sacct.txt`;
- the archived source.

**Boundaries.** This phase covers computational readiness only, on synthetic data. **Step 3 has not been run and remains unauthorised.** The D1 verdict, NOT REPRODUCED, stands. Scope: full three-state label primary, D1 ratio-defined cohort secondary.

## 1. Outcome at a glance

| item | result |
|---|---|
| T1 existing tests | **PASS** (56/56) |
| T2 identity with reference v2 at regular ψ | **PASS**: bitwise at all 5 points, serial and pooled |
| **T3 extreme-ψ time ratio ≤ 0.10** | **FAIL at 2 of 3 points** (0.128, 0.163; the third is 0.079). The penalty and reason were correct at all 3 |
| T4 optimiser-path identity | **PASS** on the 60-patient subset and the full cohort |
| T5 non-negative-definite with det > 0 | **PASS** |
| **T6 warm/cold fallback equivalence (G = 3)** | **FAIL.** \|Δ negative log-likelihood\| = 0.0053 (≤ 0.1), but the cold optimum and warm start 1 miss the gradient criterion |
| **T7 fallback multi-start, G = 3** | **FAIL.** Spread 0.0061, but starts 1 and 3 miss ‖∇‖ < 0.934 |
| T7 fallback multi-start, G = 5 | **PASS** (spread 2.7e-5; all criteria met) |
| Primary runtime gate (every Q tested) | **PASS** |
| **Fallback runtime gate, G = 5** | **FAIL:** fit 67.6–94.3 min > 60; projected pilot 32.0 node-hours > 24 |
| Fallback runtime gate, G = 3 | PASS on runtime, **but G = 3 fails T6/T7 convergence**, and missed the order tolerance in the doc 23 synthetic feasibility study |

## 2. Records verified before any change

All matched; recorded in `00_baseline.json` and re-checked by the job (`bench/S0_provenance.json`):

| record | SHA-256 |
|---|---|
| doc 16 (and its frozen copy) | `c2eace18…bbc` |
| Rev 1 (doc 20) | `c0209941…bd` |
| Rev 2 (doc 21, and its frozen copy) | `c53c015f…267` |
| proposal doc 24 (left unaltered) | `dfe24bb2…c32b` |
| B0 `m2_quad_reference.py` | `68d3dcef…` = the B0 provenance |
| `m2_quad_reference_v2.py` (new, read-only, a verbatim copy of the doc 23 code) | `23c2fde7…fae8` = the doc 23 provenance |
| B0 and doc 23 output files | no earlier hashes existed; recorded now as the baseline |

Git: branch `main`, HEAD `3a860ca`, no tracked changes; untracked files only.

## 3. Files created and modified

**Modified:**
- `src/ppmi_subtype/m2_quad.py`: P1 budget, P2 eigenvalue semantics with reasons, the monitoring log, `evaluate_reported`, `fd_hessian`, and a `fit(init_state=…)` warm start.

**Created:**
- `src/ppmi_subtype/m2_quad_reference_v2.py`
- `tests/test_m2_quad_addendum_a.py` (4 tests)
- `scripts/48c_m2a1_addendum_a_benchmark.py`
- `docs/cluster_phase/25_…_FINAL_20261017.md` plus `_frozen/25_STAGE2_ADDENDUM_A_FINAL.{sha256,as_frozen_20261017.md}`
- this report
- the execution directory

**Unchanged:**
- docs 16, 20, 21 and 24, and their frozen copies;
- `m2_grm.py`, `m2_quad_reference.py`, the B0 script and the doc 23 script;
- all earlier experiment outputs;
- all D1 outputs.

## 4. Tests

**T1.** 56 passed (`test_m2_quad.py` 13, `test_m2_grm.py` 10, `test_m2_recovery_diag.py` 3, `test_grm.py`, `test_grm_ou.py`), with assertions unchanged. Log: `T1_pytest.txt`.

**T5 and the new unit tests.** 4 passed (`T5_and_unit_pytest.txt`).
- A patient Hessian with eigenvalues (+1, +1, −1, −1), so det > 0 but not negative definite. Reference v2 accepted it with a finite value; the updated code returns `non_nd`.
- The reported context uses the G-fallback for that patient, with count 1.
- The P1 trigger reason is checked.
- Bitwise identity with reference v2 when the budget is not reached.
- Monitoring fields are present.

**T2** (synthetic dataset 0, full cohort): **bitwise** equality of value and gradient, serial and pooled (32), at:
- ψ = truth;
- the doc 23 optimum. It was reproduced by reference v2: negative log-likelihood 57,022.74225348176 and 104 evaluations, both equal to doc 23;
- 3 perturbations (seeds `[42, 46, 7, k]`).

**T3** (the three penalty points of reference v2's 60-patient path, serial):

| evaluation | reference v2 | updated | ratio | reason | ≤ 0.10? |
|---|---|---|---|---|---|
| 2 | 67.35 s | 8.59 s | 0.128 | `mode_budget` | **no** |
| 4 | 59.56 s | 9.72 s | 0.163 | `mode_budget` | **no** |
| 7 | 110.56 s | 8.75 s | 0.079 | `mode_budget` | yes |

**T3 fails.** The budget reduces these evaluations 7–13×, but not to the frozen 10 %. Both implementations return the penalty at all three points. The test and the budget were left unchanged.

**T4** (identical L-BFGS-B driver):
- **(i) 60-patient subset, serial:**
  - 119 evaluations each;
  - maximum |Δψ| over the path 0.0;
  - the same final ψ and negative log-likelihood (6,351.4759), both converged;
  - penalties at evaluations 2, 4 and 7 in both;
  - **no trigger where the reference was finite.**
  - Wall time 309 s (reference) against 105 s.
- **(ii) full cohort, 32 processes:**
  - 104 evaluations each, maximum |Δψ| 0.0;
  - the same optimum (57,022.74225348176), both converged;
  - penalties at evaluations 2, 4 and 7 in both (updated: `mode_budget` in 22 chunks each);
  - no trigger where the reference was finite.
  - Wall time 454 s against 95.9 s.
- **PASS.**

**T6 and T7** (fallback, Q = 10, 32 processes; criteria from §2 F1(4): ‖∇‖ < 0.934, no bound hits, positive-definite finite-difference Hessian):

| fit | negative log-likelihood | evaluations | ‖∇‖ | min Hessian eigenvalue | meets F1(4) | fit time (evaluations) | Hessian time |
|---|---|---|---|---|---|---|---|
| G = 3 warm start 1 (ψ̂_L) | 57,017.0135 | 27 | **2.738** | 125.3 | **no** | 217 s | 312 s |
| G = 3 warm start 2 | 57,017.0188 | 72 | 0.259 | 125.7 | yes | 605 s | 311 s |
| G = 3 warm start 3 | 57,017.0196 | 72 | **3.536** | 125.7 | **no** | 606 s | 312 s |
| G = 3 cold start | 57,017.0188 | 93 | **1.003** | 125.5 | **no** | 797 s | 312 s |
| G = 5 warm start 1 | 57,015.3887 | 64 | 0.359 | 127.1 | yes | 4,055 s (67.6 min) | 2,327 s |
| G = 5 warm start 2 | 57,015.3887 | 72 | 0.347 | 127.2 | yes | 4,430 s (73.8 min) | 2,347 s |
| G = 5 warm start 3 | 57,015.3887 | 90 | 0.231 | 127.2 | yes | 5,659 s (94.3 min) | 2,318 s |

- **T6 FAIL.** |Δ| = 0.0053 is within 0.1, but neither the cold optimum nor warm start 1 meets ‖∇‖ < 0.934.
- **T7 G = 3 FAIL.** Spread 0.0061; starts 1 and 3 fail the gradient criterion.
- **T7 G = 5 PASS.**
- Every optimiser run reported "converged" (relative reduction of F). No bound hits; every Hessian positive definite.
- [H] The G = 3 gradient misses are consistent with the Rev 2 design choice of a **fixed-placement gradient**, which is not the exact gradient of the moving-node objective. L-BFGS-B stops on relative reduction while that gradient remains above 0.934. The effect is plausibly larger with the coarser grid. This was not tested further.

## 5. Monitoring and the P1/P2 triggers

| run | evaluations | penalty evaluations (reason) | non-ND patients at non-penalty evaluations | fallback at the optimum (`evaluate_reported`, G = 5) | floating-point flags |
|---|---|---|---|---|---|
| T4(i) subset, updated | 119 | 3 (`mode_budget`) | 0 | — | 0 |
| T4(ii) full, pooled, updated | 104 | 3 (`mode_budget`; 22 chunks each) | 0 | **0 patients, 0 budget triggers → B11-reported OK** | 0 (max \|Φ\| 241.9) |
| S6 full, serial, updated | 104 | 3 (`mode_budget`) | 0 | **0, 0 → OK** | 0 |
| Fallback fits | 27–93 | 2–4 per fit (`mode_budget`) | — (AGQ has no `non_nd` penalty) | — | — |

No `non_nd` penalty occurred at any optimiser evaluation in these synthetic runs. The P2 correction was therefore exercised only in the unit tests (T5).

## 6. Timings (node002, Xeon Gold 6326; BLAS single-threaded)

**Per evaluation, at ψ = truth** (median; serial: 3 repetitions, pooled: 5; ranges in `S5_micro.json`):

| | B0 (doc 22) | doc 23 serial | **updated serial** | doc 23 pooled | **updated pooled** |
|---|---|---|---|---|---|
| Q = 7 likelihood / likelihood + gradient | 6.74 / 8.43 | 0.96 / 1.87 | **0.99 / 1.90** | 0.13 / 0.19 | **0.12 / 0.19** |
| Q = 10 | 14.44 / 18.17 | 1.93 / 3.72 | **2.00 / 3.78** | 0.26 / 0.41 | **0.24 / 0.36** |
| Q = 15 | 34.41 / 42.83 | 4.54 / 8.58 | **4.70 / 8.68** | 0.58 / 0.82 | **0.54 / 0.82** |
| Q = 20 | 62.72 / 77.83 | 8.42 / 15.64 | **8.61 / 15.82** | 0.97 / 1.45 | **0.94 / 1.37** |

The per-evaluation cost is unchanged from doc 23, as expected: P1/P2 add only counters at regular ψ.

**Complete primary fit, Q = 10:**

| | evaluations | wall | penalty evaluations |
|---|---|---|---|
| B0, serial | 92 | 50.3 min | (unrecorded) |
| doc 23, serial / pooled | 104 / 104 | 55.9 / 7.1 min | (3; unrecorded) |
| **updated, serial** (S6) | 104 | **11.7 min** (700 s) | 3, 8.6–11.0 s each |
| **updated, pooled** (T4(ii)) | 104 | **1.6 min** (95.9 s) | 3, 10.8–12.9 s each (reference v2: 114–145 s) |

## 7. Runtime accounting and projections

**Convention:** node-hours = allocated exclusive nodes × elapsed wall hours (SLURM `AllocNodes` × `ElapsedRaw`), with the process count stated.

**Actual allocations** [measured]:

| job | node | nodes | processes | elapsed | node-hours |
|---|---|---|---|---|---|
| B0, 13825 | node002 | 1 | 1 | 10,174 s | 2.83 |
| doc 23, 13835 | node002 | 1 | 1 and 32 | 7,187 s | 2.00 |
| **this run, 13838** | node002 | 1 | 1 and 32 | **26,967 s** | **7.49** |

Smoke runs ran on the login node and are not allocated node time.

**Reconciliation with B0's convention.** Docs 22 and 23 projected 18.0 node-hours "serial", meaning one process on one node. Under the present convention that is the same quantity (node-hours = wall hours of one node) at 1 process. At the declared 32 processes, the doc 23 code projected 3.0.

**Projected pilot** (Rev 2 §6 steps 3–7), at 32 processes:
- Fit times: Q = 10 measured (95.9 s); other Q scaled by the measured likelihood-plus-gradient ratio [estimate].
- Hessians: 11 × 38 × the measured likelihood-plus-gradient time.
- Stage-1 item fits: 716 s, observed earlier.
- S1-b refits and checks: **0.62 h, assumed.**

| primary | fit (s) | basis | projected node-hours | fit ≤ 60 min | pilot ≤ 24 h |
|---|---|---|---|---|---|
| Q = 7 | 51 | estimate | 1.10 | pass | pass |
| **Q = 10** | **96** | **measured** | **1.34** | **pass** | **pass** |
| Q = 15 | 214 | estimate | 1.99 | pass | pass |
| Q = 20 | 361 | estimate | 2.78 | pass | pass |

The assumed 0.62 h and the observed 0.20 h item fits now make up the larger part of each primary projection.

**Fallback pilot:** 18 × (pooled Laplace fit + mean warm fallback fit at G) + 11 fallback Hessians at G + the same other components. Measured fits and Hessians:

| fallback | mean / max warm fit | Hessian | projected node-hours | fit ≤ 60 min | pilot ≤ 24 h |
|---|---|---|---|---|---|
| G = 3 | 476 / 606 s | 312 s | 4.6 | pass | pass |
| **G = 5** | **4,715 / 5,659 s** | **2,331 s** | **32.0** | **fail** | **fail** |

**Uncertainty:**
- Fallback fit times vary 1.4× across starts.
- All figures are for synthetic data with items at truth. Real-data fits may need different numbers of evaluations.
- The required fallback G is decided only in Step 3. The doc 23 synthetic study had G = 3 missing the 0.005 order tolerance (0.0057), which points to G = 5 [informative only].

## 8. Gates and conclusions

- **Primary method (Laplace over b):** runtime gates **pass at every Q tested** (7–20). Q is **not** selected; Step 3's accuracy rule governs. Path identity (T4) **passes**: at every evaluation of both fits, P1/P2 did not change the optimiser's path or optimum.
- **T3 fails.** The budget is not quick enough at 2 of 3 extreme points under the frozen 10 % criterion. The practical effect on fits is shown in §6: penalty evaluations of 9–13 s against 60–145 s.
- **Fallback:**
  - **G = 5:** converges cleanly (T7 pass) but is **computationally infeasible under the current budget** (per-fit 68–94 min; pilot 32 node-hours).
  - **G = 3:** fits the budget but **fails the convergence criteria** (T6, T7 G = 3).
  - **Neither fallback configuration is currently available.** Its accuracy requirement was not relaxed, and no other method was substituted.
  - Consequence: if Step 3's V2 fails for the primary method, there is no fallback that is both convergent and affordable.

## 9. Computational correctness, numerical accuracy and scientific validation, kept separate

| | status |
|---|---|
| **Computational correctness** | shown on synthetic data: T1, T2, T4 and T5 pass; P1/P2 change no regular-ψ value or path |
| **Numerical accuracy** | **not established.** Whether any Q or G meets the 0.005 order tolerance and the 0.05 D3 tolerance on the pilot's patients is Step 3, not run |
| **Scientific validation** | **not established.** Recovery, real-data B11, the sampler, G1–G7, calibration and evaluation statistics are all unrun |
| **What the synthetic results do not show** | that M2 explains the PPMI data, or anything about an intermediate state or subtypes |

## 10. Integrity and Git

- After the run: the hashes of docs 16, 20, 21, 24 and 25 match their records. D1 locks verify, the verdict is NOT REPRODUCED, and the `outputs/` checksums match (§ final check in the run log).
- HEAD `3a860ca`, branch `main`. No tracked file modified; all new files are untracked.
- **Nothing staged, committed or pushed.**

## 11. Remaining blockers and the next authorisation needed

1. **T3 failure:** accept the 7–13× reduction as adequate, or require a stricter budget. A stricter budget would need a new addendum and a re-run of T2–T4.
2. **The fallback is unavailable:** G = 5 is too slow, and G = 3 fails convergence. Options (each needs a separately proposed addendum):
   - exact moving-node gradients for AGQ;
   - a different per-fit budget;
   - accepting that Step 3 proceeds with no fallback, so that a V2 failure of the primary stops Stage 2.
3. **Step 3** (V2 on the pilot patients, §4.1 order checks, §2.5 mode checks; V3/V4 reported) needs **separate explicit approval.** Recovery, real-data fitting, G1–G7, calibration, positive controls and evaluation statistics remain unauthorised.
