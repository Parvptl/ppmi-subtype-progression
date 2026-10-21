# Stage 2 M2, Amendment 1 (Rev 2): Step 2 prototype and B0 benchmark

**Date:** 2026-10-17.
**Authorisation:** Rev 2 §6 step 2 only (prototype, V1 unit tests, B0).
**Status:** implemented and benchmarked. **This is not a scientific validation.** No quadrature-order check, likelihood-accuracy (V2) check, recovery, real-data fit, sampler check, fidelity gate, calibration, positive control or evaluation statistic was run.

**Scope (unchanged):** the full three-state label is the primary scope for target 2; D1's ratio-defined cohort is secondary. The D1 verdict, **NOT REPRODUCED**, stands.

## 1. Document hashes (verified)

| document | SHA-256 | status |
|---|---|---|
| doc 16 (frozen spec) | `c2eace18999f45e05324575e58d6933109c3836be67f832bfee55c8ad6bffbbc` | matches the frozen record; byte-identical to `_frozen/…as_frozen_20261017.md`; read-only |
| Amendment 1 Rev 2 (doc 21) | `c53c015fcc315212e2f2252bfde1e57df3619f1412abdcebf922f18fd9638267` | unchanged since review. Frozen record created: `_frozen/21_STAGE2_DOC16_AMENDMENT1_REV2.sha256`, plus the copy `_frozen/21_…REV2.as_approved_step2_20261017.md`. Both read-only |
| Amendment 1 Rev 1 (doc 20) | `c02099415a5bfcae752b968ee9b2faaeb37391e996ff27459999ac21826c22bd` | unchanged |

The benchmark job re-verified the doc 16 and Rev 2 hashes at start (`provenance.json`).

## 2. Implementation

| file | content |
|---|---|
| `src/ppmi_subtype/m2_quad.py` (new) | Per-visit Q × Q Gauss–Hermite over ε (§2.2). Damped-Newton mode in b, with a gradient-ascent fallback and a non-negative-definite flag (§2.5). Laplace over b with the **exact ψ-gradient** (envelope and implicit-mode terms; b- and ψ-derivatives of log L_ij as cumulants under the node posterior) (§2.3). The adaptive-GH-over-b fallback, with the gradient at fixed node placement. Cohort objective, and a fit with a wall-clock cap |
| `tests/test_m2_quad.py` (new) | 7 V1 tests (below) |
| `scripts/48_m2a1_benchmark.py` (new) | B0 driver |
| `scripts/49_m2a1_checks.py` | **not created.** It is not needed for step 2; it belongs to step 3 |
| `src/ppmi_subtype/m2_grm.py` | unchanged (hash as recorded by the pilot) |

## 3. V1 unit tests (implementation correctness, not accuracy on data)

`tests/test_m2_quad.py`: **7 passed.**

1. **Closed form.** The ε nodes reproduce the Gaussian moment-generating function exp(½ cᵀΣ_ε c) to 1e-10, and the weights sum to 1.
2. **No items observed:** log-likelihood exactly 0, for Laplace and for AGQ.
3. **Same model as `m2_grm`.** For a one-visit patient, high-order quadrature here (Q = 30, G = 9) equals `m2_grm.agq_one` over the full 2-D θ to within 2e-4. Laplace over b is within 0.05.
4. **Exact Laplace gradient against central finite differences,** all 19 ψ components (relative 1e-4).
5. **AGQ gradient at fixed node placement** against finite differences.
6. **§2.5 mode safeguards on random cases:** the mode is the same from different starts (1e-6), and the Hessian is negative definite.
7. **Laplace over b against AGQ over b** (G = 5) on random small patients: |difference| < 0.1.

Re-run together with `test_m2_grm.py` and `test_m2_recovery_diag.py`: **20 passed.**

## 4. B0 benchmark

**Run:** `experiments/20261016T223545Z_stage2_m2a1_benchmark/results/`.
- SLURM job 13825, node002 (Intel Xeon Gold 6326 @ 2.90 GHz), single-threaded BLAS, serial.
- Elapsed 2 h 49 min.

**Data:** synthetic dataset 0 of the pilot design (934 patients, 4,295 visits), items at their generating values, ψ = truth for the evaluation timings. No real item responses were used.

**Smoke run:** a 60-patient code-path check, in `experiments/_smoke/20261016T221302Z_stage2_m2a1_benchmark_smoke`. Its numbers are not used.

### 4.1 Evaluation timings

Warm start; median of 3 unless a single call exceeded 600 s.

| method | likelihood (s) | likelihood + gradient (s) | patients with non-ND Hessian |
|---|---|---|---|
| current `m2_grm` (2n-D Laplace), comparison | 0.13 | 0.20 | — |
| **Laplace over b, Q = 7** | 6.74 | **8.43** | 0 |
| **Laplace over b, Q = 10** | 14.44 | **18.17** | 0 |
| **Laplace over b, Q = 15** | 34.41 | **42.83** | 0 |
| **Laplace over b, Q = 20** | 62.72 | **77.83** | 0 |
| AGQ over b, G = 3, Q = 10 (fallback) | 109.8 | 454.3 | — |
| AGQ over b, G = 5, Q = 10 (fallback) | 750.4 (1 rep) | 3,403.5 (1 rep) | — |

### 4.2 One complete fit

Laplace over b, **Q = 10**. Q = 10 is the Rev 2 default; the §4.1 order check that would choose Q is not authorised yet. Start: default ψ.

| wall time | iterations / evaluations | converged | ‖∇‖ | bound hits |
|---|---|---|---|---|
| **3,020 s (50.3 min)** | 75 / 92 | yes (relative-reduction criterion) | 0.00098 | none |

This only shows the prototype's fit *runs and stops* within the cap on synthetic data. It says nothing about recovery or accuracy, and the fitted values were not examined.

### 4.3 Projected pilot cost

Rev 2 §6 steps 3–7, serial on one node, in node-hours. The components are:
- 18 fits: 3 real starts, 5 fixed-item recovery fits, 10 full-pipeline S1-b stage-2 fits;
- 11 finite-difference Hessians × 38 gradient evaluations;
- stage-1 item fits, from observed pilot timings;
- S1-b item refits, **assumed** at 5× a single stage-1 fit;
- V2 / sampler / V5 checks, **assumed** from pilot timings.

Fit times for Q ≠ 10 are scaled by the measured likelihood-plus-gradient ratio.

| Q | fits | Hessians | other (assumed) | **total** | ≤ 24? |
|---|---|---|---|---|---|
| 7 | 7.0 | 1.0 | 0.8 | **8.8** | yes |
| **10** | 15.1 | 2.1 | 0.8 | **18.0** | **yes** |
| 15 | 35.6 | 5.0 | 0.8 | **41.4** | **no** |
| 20 | 64.7 | 9.0 | 0.8 | **74.6** | **no** |

### 4.4 B0 decision rule (Rev 2 §4.2)

- One fit ≤ 60 min: **met** (50.3 min, at Q = 10).
- Projected pilot ≤ 24 node-hours: **met at Q = 10** (18.0).
- **The rule is met, at Q = 10 only.** As instructed, nothing beyond B0 was started.

## 5. Findings and caveats for review

1. **The margin depends on Q.** The rule holds at Q = 7 and Q = 10. It fails at Q = 15 (41 node-hours) and Q = 20 (75). The §4.1 order check (not yet run) chooses the smallest Q within 0.005 of Q_ref = 20. **If it selects Q = 15, B0's condition would no longer be met.** Also, the order check itself needs Q = 20 and Q = 25 evaluations, which are not costed above.
2. **The fallback is impractical as implemented.** AGQ-over-b likelihood plus gradient costs 454 s (G = 3) and 3,404 s (G = 5) per evaluation. At about 92 evaluations per fit that is about 12 h and 87 h per fit. **If V2 fails with the Laplace primary, the pre-declared fallback could not be piloted within the B0 limits.**
3. **Bottleneck** (profiled on the smoke subset). About 90 % of time goes into the per-node GRM item probabilities and θ-derivatives (`grm.category_logprob`, `grm.theta_derivs`). These are evaluated at Q² nodes × 16 items × 4,295 visits and recomputed at every Newton and line-search step of the per-patient mode. The cumulant algebra for the exact gradient takes about 20 %. Cost grows roughly as Q² (measured: Q = 10 → 20 gives 4.3×). Possible speed-ups are not implemented and would need review:
   - fusing the value and derivative passes;
   - caching across line-search steps;
   - fewer Newton steps via warm starts;
   - parallelising over patients.
4. **The 50-minute fit is on synthetic data with items at truth,** from the default start. Real-data fits (3 starts) and S1-b fits may need more iterations. The 18-node-hour projection is serial; parallelising fits across nodes would cut wall time, not node-hours.
5. **Projection inputs labelled "assumed"** (the S1-b item refits and the checks) are not measured; they total 0.6 of 18.0 node-hours.
6. **What V5 shows, and what it does not.** V5 (not yet run) compares the sampler with AGQ over b **on the same approximate per-visit likelihood L̃**. It checks that the sampler targets that approximate posterior correctly. It does **not** validate the quadrature approximation itself. Quadrature accuracy is assessed **independently** by:
   - the §4.1 order checks (against Q_ref = 20 / 25);
   - **V2**, against the unchanged D3 reference that integrates the full 2n-dimensional θ.

   Neither has been run.

## 6. Integrity and Git

- No existing file was modified. `m2_grm.py`, the doc 17–19 run outputs, and all D1 and earlier-stage outputs are unchanged.
- D1 locks re-verified; NOT REPRODUCED stands.
- The `outputs/` checksums match.
- New, untracked files:
  - `src/ppmi_subtype/m2_quad.py`, `tests/test_m2_quad.py`, `scripts/48_m2a1_benchmark.py`;
  - `docs/cluster_phase/21_…_REV2` (now read-only), the `_frozen/21_*` records, and this report;
  - the benchmark run and smoke directories.
- Nothing was staged, committed or pushed.

## 7. Decisions for review

1. Accept B0 as met **at Q = 10**, with the Q-dependence caveat. Or require a pre-specified cap (for example Q ≤ 10) before the §4.1 order check is run.
2. How to handle the fallback's cost (finding 2): accept the risk, or require a cheaper fallback design (a new revision) before step 3.
3. Whether implementation-level speed-ups (finding 3) may be made before step 3. These would be code changes only, re-tested by V1 and re-benchmarked; no design change.
4. Authorisation, or not, for Rev 2 §6 step 3: V2, §4.1 order checks, §2.5 mode checks; V3/V4 reported.
