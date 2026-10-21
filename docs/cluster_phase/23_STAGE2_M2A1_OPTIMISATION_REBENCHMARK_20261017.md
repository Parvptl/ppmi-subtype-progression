# Stage 2 M2, Amendment 1 (Rev 2): code-only optimisation, re-benchmark and fallback feasibility

**Date:** 2026-10-17.
**Authorisation:** the reviewer's decision after doc 22:
- B0 provisionally met at Q = 10;
- **Step 3 not authorised;**
- code-only optimisation, re-benchmark, and a separate fallback investigation authorised.

**Status:**
- **Computational results only.** Section 5 is a feasibility study on *synthetic* patients and does not replace V2 or the §4.1 order checks.
- **No final Q is selected.**
- No real item data were fitted. No recovery, fidelity gate, calibration, positive control or evaluation statistic was run.
- **Scope unchanged:** full three-state label primary, D1's ratio-defined cohort secondary. D1 remains **NOT REPRODUCED**.

## 1. Frozen records (verified)

| record | SHA-256 | status |
|---|---|---|
| doc 16 | `c2eace18…bbc` | matches the frozen record; unchanged |
| Amendment 1 Rev 2 (doc 21) and its `_frozen/21_*` record | `c53c015f…267` | matches; unchanged |
| Rev 1 (doc 20) | `c0209941…bd` | unchanged |
| B0-benchmarked `m2_quad.py` | `68d3dcef…` | preserved verbatim as `src/ppmi_subtype/m2_quad_reference.py`. Its hash equals the B0 provenance record; the re-benchmark re-checked this |

## 2. Code changes (code-only, mathematically equivalent)

`src/ppmi_subtype/m2_quad.py` was optimised. **Unchanged:**
- the likelihood;
- the quadrature definition (Q × Q GH over ε; G⁴ adaptive GH over b);
- the prior;
- the mode convergence rule (Newton step < 1e-9, 60 iterations) and the line-search acceptance rule (Φ_new ≥ Φ_cur − 1e-12, up to 50 halvings);
- the fallback definition;
- the L-BFGS-B options and bounds.

**Optimisations:**
1. **Fused GRM kernel.** Per item, only the two thresholds bracketing the observed category are evaluated. They are gathered once per data set, not at every node. The kernel returns log P and its first three θ-derivatives in one pass. The reference evaluated all 4 thresholds plus 6-column gathers, and called `category_logprob` twice.
2. **Reuse in the mode search.** The Newton trial point's full local evaluation also serves as its line-search value, and the mode's local terms are reused for the Laplace value and gradient. This gives the same iterates and the same acceptance rule.
3. **Cumulants from small per-node moments.** The b and mean directions do not depend on the node, so the 4 × 4 × 13 per-node tensors are replaced by 2 × 13 and 2 × 2 × 13 moments, contracted afterwards. This is exact algebra.
4. **Optional process pool** over patient chunks (`Objective(jobs=…)`). The per-chunk arithmetic is identical; only the summation order differs.
5. AGQ evaluation chunk size reduced from 4,096 to 1,024 b-points, for memory only.

**Tests:** `tests/test_m2_quad.py`, **13 passed** (7 original V1 tests and 6 new regression tests).
- In original test 6, only the call was updated to the new `modes` signature; its assertions are unchanged.
- Regression tolerances, as justified in the file header:
  - fused kernel against `grm`: 1e-10 absolute on Σ log P, 1e-8 relative on the derivatives;
  - cumulant restructuring against the reference: 1e-10 relative;
  - cohort values against the reference: 1e-8 per patient (the Laplace value is stationary in b̂, and modes are solved to 1e-9);
  - gradients: 1e-6 (1 + |g|) (first-order dependence on b̂);
  - pool against serial: 1e-10 relative on the value, 1e-8 on the gradient.
- With `test_m2_grm.py` and `test_m2_recovery_diag.py` the M2 suites pass.

**Observed equivalence on the benchmark data:**
- On the 60-patient smoke subset, the value difference against the reference was 0.0 and the maximum gradient difference 4.8e-13.
- The complete Q = 10 fit reached the **same optimum** as the B0 fit: negative log-likelihood 57,022.7423 against 57,022.74225, and ‖∇‖ 0.0008. It took 104 evaluations against B0's 92: floating-point differences send L-BFGS along a slightly different path to the same point.

## 3. Re-benchmark

**Run:** `experiments/20261017T082636Z_stage2_m2a1_rebenchmark/results/`.
- SLURM job 13835, **node002** (Xeon Gold 6326, the same node and CPU as B0), single-threaded BLAS, 1 h 59 min.
- Same data as B0: synthetic dataset 0, 934 patients, 4,295 visits, items at truth, ψ = truth.
- Warm start; median of 3 unless one call exceeded 600 s.

**Smoke:** a 60-patient code-path check, in `experiments/_smoke/20261017T081120Z_stage2_m2a1_rebenchmark_smoke`.

### 3.1 Per-evaluation timings (measured)

| | B0 (doc 22), seconds: likelihood / lik + grad | reference re-timed this run | **optimised, serial** | **optimised, pool of 32** |
|---|---|---|---|---|
| Laplace Q = 7 | 6.74 / 8.43 | — | **0.96 / 1.87** | **0.13 / 0.19** |
| Laplace Q = 10 | 14.44 / 18.17 | 14.42 / 18.11 | **1.93 / 3.72** | **0.26 / 0.41** |
| Laplace Q = 15 | 34.41 / 42.83 | — | **4.54 / 8.58** | **0.58 / 0.82** |
| Laplace Q = 20 | 62.72 / 77.83 | — | **8.42 / 15.64** | **0.97 / 1.45** |
| AGQ G = 3, Q = 10 | 109.8 / 454.3 | — | **61.1 / 90.6** | **5.68 / 8.16** |
| AGQ G = 5, Q = 10 | 750.4 / 3,403.5 | — | **464.0 / 684.5** (1 rep) | **41.2 / 61.5** |

**Speed-ups for likelihood plus gradient:**
- serial: 4.5–5.0× (Laplace), 5.0× (AGQ);
- pool of 32 against serial: 9–11×.

No patient had a non-negative-definite Hessian at any Q.

### 3.2 Complete fit, Laplace Q = 10 (measured)

| | wall time | evaluations | converged | ‖∇‖ |
|---|---|---|---|---|
| B0 (reference, serial) | 3,020 s (50.3 min) | 92 | yes | 0.00098 |
| **Optimised, serial** | **3,356 s (55.9 min)** | 104 | yes | 0.0008 |
| **Optimised, pool of 32** | **425 s (7.1 min)** | 104 | yes | 0.0008 |

**The serial fit did not get faster**, even though each evaluation at ψ = truth is 4.9× cheaper. The reason (diagnosed on the smoke subset, and confirmed in the reference code):

- L-BFGS-B's first trial step from the default start moves ψ to its bounds (log-SD −8). There the prior term is about 1e7. The rounding noise in Φ then exceeds the line search's **absolute** acceptance tolerance of 1e-12, so most mode steps are rejected and halved up to 50 times.
- One such evaluation costs about 65 s on 60 patients. It returns the invalid-value penalty anyway, and L-BFGS-B backtracks.
- On the full cohort, the fit averages 32 s per evaluation against 3.7 s at ψ = truth, so the evaluations at extreme ψ dominate the serial fit time.
- The reference has the identical behaviour (B0's 33 s per evaluation against 18 s at truth).
- Changing the acceptance rule or the start would alter a convergence rule or the optimisation path, which this authorisation excludes. **It is not changed.** §6 lists options for review.

### 3.3 Projected pilot cost

Node-hours, Rev 2 §6 steps 3–7, the same components as B0:
- 18 fits;
- 11 Hessians × 38 evaluations;
- stage-1 item fits, observed;
- S1-b refits and checks, assumed (0.6 node-hours, unchanged from B0).

**Basis:** the Q = 10 Laplace fit time is **measured**. Other Q values are **estimated** by scaling that fit by the measured likelihood-plus-gradient ratio. AGQ fits are **estimated** as 104 evaluations × the measured AGQ likelihood-plus-gradient time.

| design | serial (node-hours) | pool of 32 on one node (node-hours) |
|---|---|---|
| Laplace Q = 7 (estimate) | 9.4 | 1.8 |
| **Laplace Q = 10 (measured fit)** | **18.0** | **3.0** |
| Laplace Q = 15 (estimate) | 40.5 | 5.2 |
| Laplace Q = 20 (estimate) | 73.2 | 8.5 |
| AGQ G = 3, Q = 10 (estimate) | 58.5 | 6.0 |
| AGQ G = 5, Q = 10 (estimate) | 436 | 39.9 |

**Notes:**
- With the pool, every Laplace Q in {7, 10, 15, 20} is within the B0 limit of 24 node-hours. Serially, only Q ≤ 10 is.
- The order check's own Q = 20 / 25 evaluations are cheap: seconds each when pooled.
- **This does not select Q.** Q remains to be chosen by the §4.1 accuracy rule, which is not run.

## 4. Computational, numerical and scientific claims, kept separate

- **Computational (shown here):** the optimised code computes the same quantities as the B0 prototype within the stated tolerances. It is about 5× faster per evaluation serially and about 45× faster pooled. A pooled Q = 10 fit takes 7 min on synthetic data.
- **Numerical accuracy (not established):** whether Q = 10, or any Q, meets the 0.005 order tolerance and the 0.05 D3 tolerance on the pilot's patients is exactly Step 3 (V2, §4.1). It has not been run.
- **Scientific validation (not established):** recovery, the real-data B11 checks, the sampler and the gates remain untested under Amendment 1.

## 5. Fallback (AGQ over b) feasibility: separate investigation

**What was measured** (section 3 above):
- timings for G = 3 and 5;
- **accuracy on 30 synthetic patients.** They were drawn with the V2 seed `[42, 45, 4]` from the *synthetic* dataset-0 cohort, which is **not** the pilot's real patients. ψ = truth, items at truth, Q = 10.

**What was compared,** per patient:
- AGQ at G = 3 and 5 against G_ref = 7. Tolerance 0.005, as in Rev 2 §4.1(b).
- AGQ at G = 3, 5 and 7 against the unchanged **D3 reference**: `m2_grm.agq_one` for the 17 patients with 2n ≤ 6, `is_one` (20,000 draws, t₅) for the 13 others. Tolerance 0.05, as in B11 and V2. The D3 reference's maximum Monte Carlo SE was 0.0077.

| comparison | max \|difference\| | tolerance | meets? |
|---|---|---|---|
| G = 3 vs G = 7 | **0.0057** | 0.005 | **no** |
| G = 5 vs G = 7 | **0.0033** | 0.005 | yes |
| G = 3 vs D3 | 0.0091 | 0.05 | yes |
| G = 5 vs D3 | 0.0098 | 0.05 | yes |
| G = 7 vs D3 | 0.0098 | 0.05 | yes (this residual mixes the Q = 10 ε-quadrature error with the D3 Monte Carlo error) |

**Reading** (synthetic feasibility only):
- **On accuracy,** the declared fallback can meet the existing requirements at **G = 5**. G = 3 misses the 0.005 order tolerance by a small margin on these 30 synthetic patients.
- **On cost,** G = 5 is projected (estimate) at **436 node-hours serial, or 39.9 pooled.** Both exceed the B0 limit of 24.
  - The estimate assumes 104 evaluations from the default start, at the full AGQ cost. That includes the extreme-ψ evaluations, whose cost is mode-dominated and so probably over-counted for AGQ [H].
  - A pooled G = 5 fit is estimated at about 1.8 h.
- **Conclusion.** The declared fallback is accurate enough at G = 5 on synthetic data, but **not feasible within the B0 limit as currently specified.** I have **not** replaced or weakened it. Two design changes might make it feasible. Both would need an amendment:
  - (a) start the fallback fit from the Laplace estimate, which is likely to need far fewer evaluations [H];
  - (b) use a sparse (Smolyak) grid over b instead of the G⁴ tensor product. This is a *different quadrature* and would need its own accuracy checks.

## 6. Items for review (not implemented)

1. **Extreme-ψ cost (§3.2).** Options, each changing a numerical rule and therefore requiring approval:
   - (a) a line-search acceptance tolerance relative to |Φ|, for example 1e-12 (1 + |Φ|);
   - (b) returning the invalid-value penalty early when the mode search cannot make progress (for example, D⁻¹ beyond a fixed conditioning limit);
   - (c) a bounded first L-BFGS-B step or a scaled parameterisation.

   None changes the likelihood at well-conditioned ψ, but each changes the optimisation path.
2. **The fallback** (§5): whether to propose amendment option (a) or (b), or to accept that the fallback cannot be piloted within the B0 limit if V2 fails.
3. **Whether node-hours on a pooled, exclusive node** (rather than serial) is the intended B0 accounting.
4. **Step 3 (V2, §4.1 order checks, §2.5 mode checks)** remains unauthorised pending your review.

## 7. Integrity and Git

- **Frozen files:** none modified. Docs 16, 20 and 21 and the `_frozen/` records are unchanged, and the B0 run directory is untouched.
- **D1:** frozen files untouched; the locks verify; NOT REPRODUCED stands.
- **Unchanged:** `m2_grm.py` and `scripts/48_m2a1_benchmark.py`.
- **Changed:**
  - `src/ppmi_subtype/m2_quad.py` (optimised);
  - `tests/test_m2_quad.py` (6 regression tests added; test 6 call updated).
- **New:**
  - `src/ppmi_subtype/m2_quad_reference.py` (a verbatim copy of the B0 code);
  - `scripts/48b_m2a1_rebenchmark.py`;
  - this report;
  - the re-benchmark run and smoke directories.
- **Git:** all untracked. Nothing was staged, committed or pushed.
