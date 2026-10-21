# Stage 2 M2: exact-gradient AGQ fallback, execution report

**Date:** 2026-10-18.

**Specification:** `27_STAGE2_EXACT_AGQ_GRADIENT_AMENDMENT_PROPOSED_20261018.md`, SHA-256 `17ca7a616e9629a6f12b57c24cdbef7e75f89f16da6a48ae08034a9c31226b96`.
- Frozen before any code change, with a read-only copy in `_frozen/`.
- It is a proposed implementation specification authorised by the user's instructions of 2026-10-18. It is **not** an independently approved protocol change.

**Execution directory:** `experiments/20261017T233307Z_stage2_m2_exact_agq/`. It contains:
- the baseline and verification records;
- the unit-test logs and the gradient-check tables;
- a login-node smoke run (code path only; not results);
- `bench/`: per-stage JSON and `summary.json`;
- the SLURM logs and `job13857_sacct.txt`;
- the post-run integrity record.

**Boundaries.**
- Synthetic data only.
- Step 3 has **not** been run and remains unauthorised.
- The D1 verdict, NOT REPRODUCED, stands.
- Scope: full three-state label primary, D1 ratio-defined cohort secondary.

## 1. Answers to the five separate questions

| question | answer |
|---|---|
| **1. Gradient correctness:** does the exact gradient match independent numerical checks? | **Yes, on synthetic cases** (§4): every check passed, and the fixed-placement gradient is shown to be inexact |
| **2. Fallback convergence:** do T6 and T7 pass? | **Yes, at G = 3 and at G = 5** (§5), with every frozen criterion |
| **3. Runtime feasibility:** per-fit and projected-budget limits | **G = 3 passes; G = 5 fails** (§6) |
| **4. Likelihood-approximation accuracy** (V2, order selection) | **Not established.** Step 3, not run |
| **5. Scientific validity** (recovery, real-data B11, G1–G7, calibration) | **Not established.** Not run |

**Is the fallback ready for Step 3 review?**
- **Conditionally.** A fallback that converges *and* fits the budget exists **only at G = 3.**
- Whether G = 3 is admissible is decided by Step 3's order check (G = 3 within 0.005 of G = 7 on the V2 patients). On synthetic patients G = 3 narrowly missed that tolerance (0.0057; doc 23 §5), which is informative only.
- If Step 3 requires **G = 5**, no viable fallback exists under the current budget.

## 2. Hashes, environment and resources

**Records:**

| record | SHA-256 | status |
|---|---|---|
| doc 27 (amendment) and its frozen copy | `17ca7a61…6b96` | verified before submission and after the run |
| docs 16 / 20 / 21 / 24 / 25 / 26 | `c2eace18…` / `c0209941…` / `c53c015f…` / `dfe24bb2…` / `439cf084…` / `e2ffc1bf…` | unchanged (`00_baseline.json`, `99_post_run_integrity.txt`) |
| `m2_quad_reference.py` (B0) / `_v2` (doc 23) / `_v3` (new, read-only; the code of job 13838) | `68d3dcef…` / `23c2fde7…` / `33c02d6a…` | unchanged |
| `m2_quad.py` (exact-gradient implementation) | `50f164c6…093f` | recorded at submission; unchanged after the run |
| `tests/test_m2_quad_exact_agq_gradient.py` | `0616a869…cdcb` | recorded at submission |
| `scripts/48d_m2_exact_agq_benchmark.py` | `36b31af7…fa55` | recorded at submission |

**Environment:**
- SLURM job **13857**, partition `normal`, **node002** (Intel Xeon Gold 6326 @ 2.90 GHz), 1 exclusive node (64 logical CPUs), **32 processes**.
- BLAS single-threaded (`OMP` / `OPENBLAS_NUM_THREADS = 1`). Python 3.13.16, NumPy 2.2.4 (full freeze in `pip_freeze.txt`).
- Start 2026-10-18 14:43:44, end 23:01:32.
- **Allocated: 1 node × 29,868 s = 8.30 node-hours** [measured].

## 3. Implementation

**Modified:** `src/ppmi_subtype/m2_quad.py`.
- `group_agq(..., gradient="exact")` adds the placement terms T_B (node shift through db̂/dψ), T_C (node rescaling through dL/dψ) and T_D (d log|L|), via `_agq_placement_terms`.
- New objective method `agq_exact`.
- Monitoring adds `gradient_inexact_patients`.

**Unchanged:**
- the default `group_agq` (value and fixed-placement gradient);
- the Laplace objective and gradient;
- P1/P2;
- the absolute 1e-12 line-search tolerance;
- L-BFGS-B options, the bounds and the prior;
- the full tensor-product grid. No sparse grid.

**Created:**
- `src/ppmi_subtype/m2_quad_reference_v3.py`;
- `tests/test_m2_quad_exact_agq_gradient.py` (6 tests);
- `scripts/48d_m2_exact_agq_benchmark.py`;
- doc 27 and its `_frozen/27_*` records;
- this report;
- the execution directory.

**Derivation** (doc 27 §3).

  dℓ_i/dψ = Σ_g ω_g ∂Φ/∂ψ(b_g) + m_gᵀ db̂/dψ + ⟨dL/dψ, Σ_g ω_g ∇_bΦ(b_g) z_gᵀ⟩ + d log|L|/dψ − ½ ∂ log|D|/∂ψ,

with:
- db̂/dψ = H⁻¹ ∂²Φ/∂b∂ψ (implicit-function theorem);
- dH/dψ including ∂H/∂b · db̂/dψ;
- dL = L Ψ(L⁻¹ dS L⁻ᵀ), S = H⁻¹.

Every derivative of Φ is analytic, reusing the already-validated Laplace cumulants. There is no automatic differentiation; finite differences appear only in tests and in the B11 Hessian.

**Assumption:** each patient's mode is a nondegenerate maximum. Patients where it is not are counted as `gradient_inexact_patients`. **The count was 0 at every evaluation of every fallback fit.**

## 4. Gradient correctness

**A. Finite differences of the same moving-node objective** (nodes re-placed and the mode re-solved at ψ ± h).
- **Setting:** small synthetic patients (n = 1, 2, 4 visits; 4 + 3 items), Q = 7; 4 ψ per n, including a 0.5 mean shift that moves b̂; G = 3 and G = 5.
- **Criterion:** at h = 1e-4, max relative error ≤ 1e-5. **PASS, 24 of 24 cases.** Tables in `02_gradient_check_tables.json`.

| | h = 1e-2 | 1e-3 | **1e-4 (gated)** | 1e-5 |
|---|---|---|---|---|
| exact gradient, G = 3 (max over cases) | 7.7e-4 | 7.7e-6 | **7.7e-8** | 1.2e-9 |
| exact gradient, G = 5 | 4.9e-3 | 4.9e-5 | **4.9e-7** | 4.7e-9 |
| fixed-placement gradient, G = 3 (contrast) | 1.9e-2 | 1.9e-2 | 1.9e-2 | 1.9e-2 |
| fixed-placement gradient, G = 5 (contrast) | 6.0e-3 | 1.1e-3 | 1.1e-3 | 1.1e-3 |

The exact gradient's error falls 100× per decade of h (O(h²)) until rounding. The fixed-placement error does not fall with h: it is the gradient of a different function. **This confirms doc 26's hypothesis.**

**B. Independent references:**
- **B1 (objective; reported):** AGQ at G = 3 / 5 / 7 against `m2_grm.agq_one`, full-θ AGQ, independent code, at high order. Differences: up to 6.2e-4 / 1.3e-5 / 4.2e-7.
- **B2 (gated):** at G = 1 the AGQ objective equals the Laplace objective (max |Δ| 7.1e-15), and the exact AGQ gradient equals the Laplace gradient (max |Δ| 1.5e-14). **PASS.**
- **B3 (reported):** the distance between the exact gradient and finite differences of the near-exact `agq_one` integral falls with G: 1.7e-3 (G = 3), 3.3e-5 (G = 5), 7.9e-7 (G = 7). For the fixed-placement gradient: 7.2e-3, 1.1e-4, 2.1e-6.

**E. In-job gate** (60-patient synthetic subset, G = 3, Q = 10, at the Laplace optimum):
- exact gradient: max relative error against moving-node finite differences **9.9e-7** (tolerance 1e-5): **PASS**;
- fixed-placement gradient: 0.225.

**C. Regression:**
- **C1:** all 60 existing M2 tests pass, assertions unchanged (`C1_existing_pytest.txt`).
- **C2:** the primary Laplace value and gradient are **bitwise** identical to reference v3 at ψ = truth and 2 perturbations, serial and pooled.
- **C3:** the default AGQ value and fixed-placement gradient are bitwise equal to reference v3, and the `agq_exact` value is bitwise equal to the `agq` value.
- The primary Laplace fit reproduced doc 23's optimum (negative log-likelihood 57,022.74225348176, 104 evaluations).

## 5. Fallback convergence (T6, T7)

**Setting:** Q = 10, 32 processes, synthetic dataset 0, the doc 25 starts and seeds.

**Criteria (frozen):**
- ‖∇‖ < 0.934, using the exact gradient;
- no bound hits;
- the finite-difference Hessian of the exact gradient positive definite;
- `gradient_inexact_patients` = 0;
- not capped.

| fit | negative log-likelihood | evaluations | ‖∇‖ | bound hits | min Hessian eigenvalue | inexact patients | optimiser | fit time | Hessian time | all criteria |
|---|---|---|---|---|---|---|---|---|---|---|
| G = 3 warm 1 (ψ̂_L) | 57,016.99014 | 79 | 0.00197 | none | 127.27 | 0 | converged | 669 s | 320 s | **pass** |
| G = 3 warm 2 | 57,016.99014 | 88 | 0.00076 | none | 127.27 | 0 | converged | 738 s | 321 s | **pass** |
| G = 3 warm 3 | 57,016.99014 | 98 | 0.00100 | none | 127.27 | 0 | converged | 819 s | 321 s | **pass** |
| G = 3 cold | 57,016.99014 | 93 | 0.00046 | none | 127.27 | 0 | converged | 792 s | 321 s | **pass** |
| G = 5 warm 1 | 57,015.38845 | 82 | 0.00046 | none | 127.56 | 0 | converged | 5,006 s (83.4 min) | 2,339 s | **pass** |
| G = 5 warm 2 | 57,015.38845 | 103 | 0.00046 | none | 127.56 | 0 | converged | 6,166 s (102.8 min) | 2,344 s | **pass** |
| G = 5 warm 3 | 57,015.38845 | 103 | 0.00084 | none | 127.56 | 0 | converged | 6,211 s (103.5 min) | 2,338 s | **pass** |

- **T6 (G = 3):** |Δ negative log-likelihood| warm against cold = **7.3e-12**, and both pass. **PASS.**
- **T7, G = 3:** spread **8.0e-11**. **PASS.**
- **T7, G = 5:** spread **1.2e-10**. **PASS.**

In doc 26, with the fixed-placement gradient, starts ended at ‖∇‖ = 2.7 and 3.5. The exact gradient removes that failure.

**Diagnostic counts per fit:**
- Penalty evaluations: 2–4, all `mode_budget` at L-BFGS-B's early extreme trial steps. The chunk-level triggers were 22–74.
- Non-negative-definite patients at accepted evaluations: 0.
- `gradient_inexact_patients`: 0 at every evaluation.
- Floating-point proximity flags: 1 per fit at most, always at a penalised extreme-ψ evaluation (max |Φ| 750–981), never at an accepted one.

## 6. Runtime

**Per evaluation** (32 processes; median of 3):

| | likelihood | likelihood + gradient |
|---|---|---|
| `agq_exact` G = 3 | 5.48 s | **8.25 s** |
| `agq_exact` G = 5 | 40.47 s | **61.07 s** |

For comparison, doc 26's fixed-placement likelihood plus gradient was 8.16 s and 61.48 s. The exact terms add negligible cost.

**Projected fallback pilot:**
- formula: 18 × (pooled Laplace fit 91.5 s [measured] + mean warm fallback fit [measured]) + 11 × fallback Hessian [measured] + 716 s stage-1 item fits [observed] + **0.62 h S1-b refits and checks [assumed]**;
- 1 node, 32 processes; node-hours = node wall hours.

| G | warm fit mean / max | Hessian | **projected node-hours** | one fit ≤ 60 min | pilot ≤ 24 h |
|---|---|---|---|---|---|
| **3** | 742 / 819 s (13.7 min max) | 321 s | **5.97** | **pass** | **pass** |
| **5** | 5,794 / 6,211 s (103.5 min max) | 2,340 s | **37.40** | **fail** | **fail** |

**Actual allocation for this benchmark: 8.30 node-hours** (job 13857). That is separate from the projections.

## 7. Pass/fail summary of every acceptance criterion

| criterion | G = 3 | G = 5 |
|---|---|---|
| A: finite-difference gradient, moving nodes (≤ 1e-5) | pass | pass |
| B2: G = 1 equals Laplace (value and gradient) | pass | (n/a) |
| E: in-job gradient gate | pass | (not run at G = 5; not specified) |
| C1 / C2 / C3 regression | pass | pass |
| ‖∇‖ < 0.934 at every optimum | pass (4/4) | pass (3/3) |
| no bound hits | pass | pass |
| finite-difference Hessian positive definite | pass | pass |
| `gradient_inexact_patients` = 0 | pass | pass |
| T6 (warm/cold ≤ 0.1, both converged) | **pass** | (n/a) |
| T7 (3-start spread ≤ 0.1, all converged) | **pass** | **pass** |
| one warm fit ≤ 60 min | **pass** | **fail** (83–104 min) |
| projected pilot ≤ 24 node-hours | **pass** (5.97) | **fail** (37.40) |
| admissible under the accuracy rule (order check against G = 7 ≤ 0.005) | **unknown: Step 3** (synthetic indication 0.0057, a narrow miss) | unknown: Step 3 (synthetic 0.0033) |

## 8. Unresolved issues

- **T3** (doc 26) still fails under its frozen 10 % timing criterion. It is unchanged here.
- **The admissible G is unknown until Step 3's order check.** If G = 5 is required, the fallback is infeasible under the current budget. The cost is dominated by the G⁴ = 625 node evaluations per patient, not by the gradient.
- **Projections include 0.62 h of assumed components.** All timings are synthetic, with items at truth; real-data fits may need different evaluation counts.
- **Exactness of the gradient** relies on each mode being a nondegenerate maximum. That held everywhere in this run (count 0), but it is not guaranteed in general.

## 9. Integrity and Git

- **Pre-submission** (`04_pre_submission_verification.txt`) and **post-run** (`99_post_run_integrity.txt`) checks found the frozen records of docs 16, 21, 25 and 27 OK.
- Docs 20, 24 and 26, the three reference implementations, `m2_grm.py` and the earlier benchmark outputs are unchanged.
- The benchmark changed no source file.
- D1 locks verify; the verdict is **NOT REPRODUCED**; the `outputs/` checksums match.
- Git: branch `main`, HEAD `3a860ca`, 0 tracked changes, 62 untracked files, 0 staged. **Nothing staged, committed or pushed.**

## 10. Smallest defensible next decision (not implemented)

The fallback's viability now hinges on one unknown: **the G the accuracy rule requires.** The smallest decision is whether to authorise the **narrowly scoped Step 3 accuracy checks** that answer it:
- V2 on the 30 designated pilot patients against D3;
- the Q-order and G-order checks.

Those decide the primary Q and the admissible fallback G together:
- **If G = 3 is admissible:** a convergent, affordable fallback exists, and Stage 2 can proceed to its next gated steps.
- **If G = 5 is required:** the fallback is infeasible under the current budget. A separate decision would then be needed, for example a cost-reducing amendment for G = 5 (such as starting G = 5 from the G = 3 optimum) or a budget change. Neither is proposed or implemented here.

**Step 3 and all later scientific evaluation remain unauthorised** pending your review: recovery, real-data fitting, real-data B11, G1–G7, calibration, positive controls and evaluation statistics.
