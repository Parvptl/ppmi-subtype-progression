# Amendment 1 (Revision 2), Addendum A: proposed (not in force)

**Date:** 2026-10-17.
**Status:** **PROPOSED, for review.** Nothing in it is implemented. If approved, its SHA-256 is recorded in `docs/cluster_phase/_frozen/` before any code change.

**Amends:** `21_STAGE2_DOC16_AMENDMENT1_PROPOSED_REV2_20261017.md` (Rev 2, SHA-256 `c53c015f…267`, frozen record `_frozen/21_STAGE2_DOC16_AMENDMENT1_REV2.sha256`).

**Unchanged and preserved:**
- doc 16 (`c2eace18…bbc`);
- Rev 1 (`c0209941…bd`);
- Rev 2;
- every frozen D1 output. The D1 verdict, NOT REPRODUCED, stands.
- Scope: full three-state label primary, D1 ratio-defined cohort secondary.

**Based on:**
- doc 22 (B0);
- doc 23 (optimisation and re-benchmark);
- the investigation in §1 (scratch scripts, 60 synthetic patients; no repository code changed).

| tag | meaning |
|---|---|
| [O] | observed / measured |
| [I] | inferred from measurements |
| [P] | proposed |
| [A] | assumption |

---

## 0. What stays exactly as it is (binding)

- **All numerical tolerances:**
  - the order checks (0.005 against Q_ref = 20 / 25 and G_ref = 7);
  - V2 (≤ 0.05 against the unchanged D3 reference);
  - the §2.5 mode-uniqueness checks;
  - V5.
- **All B11 thresholds:**
  - item fit (3 starts within 0.1; a ∈ [0.05, 20]);
  - longitudinal fit (‖∇‖ < 1e-3 × patients; no bound hits; positive-definite Hessian; 3 starts within 0.1);
  - sampler (R̂ < 1.01; ESS ≥ 100).
- **Recovery definitions and criteria** (fixed-item pre-requisite; full pipeline binding; ≥ 4 of 5 for every ψ).
- **Fidelity gates G1–G7, the G4 rule, and the gate order.**
- **Q is selected only by the Rev 2 §4.1 accuracy rule.** It is **not capped at 10.** Nothing here selects Q.
- **The fallback's accuracy requirement** (smallest G ∈ {3, 5} within 0.005 of G = 7, and V2 ≤ 0.05) is unchanged.
- **The tensor-product Gauss–Hermite quadrature over b is retained.** No sparse grid.
- **L-BFGS-B options and ψ bounds are unchanged:** `ftol` 1e-15, `gtol` 1e-7, `maxiter` 1000, `maxfun` 4 × `maxiter`; bounds as in doc 16 D3.
- **The mode-search rules are unchanged:** Newton step < 1e-9, at most 60 iterations, line search accepting Φ_new ≥ Φ_cur − 1e-12, at most 50 halvings. The **absolute 1e-12 tolerance is not changed** (§1.3).

---

## 1. Investigation of the extreme-ψ behaviour

### 1.1 What was measured [O]

**Setting:** 60 synthetic patients (dataset 0, items at truth), the optimised prototype, Laplace over b, Q = 10.

**Full L-BFGS-B fit from the default start:**
- 119 evaluations in 308 s; converged.
- **3 evaluations** (numbers 2, 4 and 7) took 65, 57 and 106 s and returned the existing invalid-value penalty (1e12, zero gradient). Together they consumed **228 s (74 %)**.
- The other 116 evaluations averaged **0.69 s.**

**At those 3 points, ψ was extreme:**
- condition number of D: 3.1e20, 8.4e14 and 7.4e8;
- log-diagonals of L and L_ε between −8.0 and +3.0;
- at evaluation 2: max D⁻¹ = 1.9e23; tremor visit variance 1.1e-7; **PIGD visit SD 5.0.**

These points are L-BFGS-B's early trial steps from the default start; it then backtracks.

**Inside the mode search at evaluation 2** (41 patients of one visit-count group, warm-started):
- The Hessian of the approximate objective was **not negative definite** for 17–46 % of patients per iteration. Its eigenvalues ranged from **−371 to 1.9e23**.
- Rejected steps had **true decreases of Φ** of order −10⁶ to −10⁸. These are not rounding-level differences: at |Φ| ≈ 10⁷ the floating-point resolution is about 2e-9.
- By iteration 6 every patient's steps were rejected. The search then exhausts its iterations and halvings: 47 full local evaluations and 1,082 line-search evaluations in that group.
- The returned mode was non-negative-definite for 98 % of patients. The determinant sign was negative for some, so the penalty was returned.

**On the full cohort** [I]: in the re-benchmark (doc 23) the serial Q = 10 fit used 104 evaluations in 3,356 s. A normal evaluation costs about 3.7 s, which accounts for about 385 s. So **about 2,970 s (about 88 %) is inferred to lie in a few such penalty evaluations.** Per-evaluation times on the full cohort were not recorded.

### 1.2 Diagnosis

The cost does **not** come from the line-search tolerance. It comes from running the full mode search at ψ values where:
- (i) D is nearly singular, so the prior's precision spans more than 20 orders of magnitude; and
- (ii) Σ_ε is so large relative to the fixed Q = 10 per-visit grid that the quadrature objective is not log-concave in b. That is the possibility Rev 2 §2.1 point 4 anticipated.

The search cannot converge there. It spends its full budget and ends in the penalty.

### 1.3 Why a scale-aware tolerance is not proposed

The rejected decreases (−10⁶ and below) exceed any floating-point-scaled tolerance by many orders. A tolerance of the form max(1e-12, κ ε |Φ|) would not change the outcome at these points.

At normal ψ the per-patient |Φ| is O(10²), so the absolute 1e-12 sits at or above the floating-point resolution (spacing(10²) ≈ 1.4e-14). **The absolute tolerance is kept.** Monitoring is added instead (§2.3).

### 1.4 A conformance gap found in the prototype [O]

- **What Rev 2 §2.5 says:** if a patient's Hessian at the returned mode is not negative definite, the Laplace term is undefined for that patient; the patient is evaluated by the fallback and counted.
- **What the prototype does** (both the B0 code and the optimised code): it returns the penalty only when det(H) ≤ 0. Otherwise it **uses the Laplace value even when H is not negative definite** (an even number of positive eigenvalues). The non-negative-definite count is reported, but the fallback is never invoked.
- **Effect on runs so far:**
  - in the 60-patient fit trace, no non-penalty evaluation had a non-negative-definite patient, so the gap did not affect that fit;
  - at ψ = truth on the full cohort the count was 0 (docs 22 and 23).
- It is nevertheless a deviation from Rev 2, and is corrected below (§2.2).

---

## 2. Proposed changes

### 2.1 P1: work budget and early infeasibility for the per-patient mode search [P]

**Rule.** During the mode search, track for each patient:
- (a) the number of consecutive iterations whose Hessian is not negative definite; and
- (b) the cumulative number of line-search halvings.

If (a) reaches **10** or (b) reaches **200**, mark the patient's mode search **failed** and stop iterating it. If any patient is failed at ψ, the cohort objective returns the **existing penalty** (1e12, zero gradient) with reason `mode_budget`.

**Rationale.** These thresholds are far above normal use [O]:
- At regular ψ, a warm-started search uses 2–3 local evaluations with no halvings.
- A cold start at the default ψ uses about 8 iterations and at most 1 halving.
- The failing evaluations used 47 iterations and 1,082 halvings per group.

**Numerical implications:**
- **Objective and gradient at any ψ where no patient reaches the budget:** unchanged. The same iterates, values and gradients, bitwise.
- **At ψ where the budget triggers:** the penalty is returned sooner. In every case observed so far the current code also returned the penalty there.
- **Residual risk.** At some ψ the current code might eventually reach a valid mode after more than 10 non-negative-definite iterations or 200 halvings. The objective would then differ (penalty instead of a finite value) and the optimiser path could change. The tests in §4 detect any such case. Each budget trigger is logged with ψ, the patient chunk and the reason.
- **Convergence checks.** B11's longitudinal-fit criteria are evaluated at the reported optimum. **New reported check:** zero budget triggers and zero non-negative-definite patients at every reported point (optimum, V2, recovery fits, B11).
- **Reproducibility.** Deterministic. The serial and pooled paths are identical in arithmetic per chunk. Thresholds and trigger counts are written to each run's provenance.

### 2.2 P2: bring non-negative-definite handling into conformance with Rev 2 §2.5 [P]

Two contexts are distinguished:
- **(a) Inside the optimiser:** a ψ at which **any** patient's returned mode has a Hessian that is not negative definite is treated as **infeasible**: the existing penalty, with reason `non_nd`. This replaces the current det-sign test. Mixing Laplace and fallback values across patients inside L-BFGS-B would make the objective discontinuous, and the optimum must in any case lie where every patient is regular.
- **(b) At every reported evaluation** (V2, the fit optimum, recovery optima, B11): Rev 2 §2.5 applies as written. Non-negative-definite patients are evaluated by the fallback at the declared G, and the count is reported. A non-zero count at a fit optimum is reported as a failure of the B11 longitudinal check for that fit.

**Numerical implications:**
- At ψ with every patient regular (as at ψ = truth and in the whole traced fit [O]), nothing changes.
- (a) can change the objective only at ψ where the current code used an invalid Laplace value. That is a correction, not a relaxation.

### 2.3 Monitoring, report only [P]

For every evaluation in a run, record:
- the maximum |Φ| at the returned modes;
- spacing(max |Φ|);
- the counts of non-negative-definite patients and budget triggers;
- the evaluation's wall time.

Flag any evaluation where spacing(max |Φ|) > 1e-13, i.e. where the absolute line-search tolerance approaches the floating-point resolution. These records also give **measured per-evaluation times on the full cohort**, replacing the inference in §1.1.

### 2.4 F1: warm-started fallback (AGQ over b) [P]

**Definition.** A fallback fit at (Q, G):
1. Run the primary Laplace fit at the same Q, or reuse its result: ψ̂_L.
2. Start L-BFGS-B, with the unchanged options and bounds, from ψ̂_L. Per-chunk modes are warm-started from the Laplace modes at ψ̂_L.
3. Place the AGQ nodes at each patient's own mode and curvature, as in Rev 2 §2.3. The gradient is the fixed-placement gradient of Rev 2.

**B11 multi-start for the fallback.** Three starts:
- ψ̂_L;
- ψ̂_L + N(0, 0.2²) on every component, seeds `[42, 46, 8, k]`, k = 1, 2.

They must agree within 0.1 in negative log-likelihood (the unchanged B11 rule). The other B11 longitudinal criteria are unchanged.

**Equivalence check against a cold start:** at G = 3, the warm-started optimum must match a cold-started fallback fit from the default ψ within 0.1 in negative log-likelihood. This shows the warm start does not change *which* optimum is found.

**Accuracy checks for the fallback (unchanged; part of Step 3, not run under this addendum):**
- On V2's 30 patients, with the pilot ψ̂ and items: G is the smallest of {3, 5} with max |log p_G − log p_{G=7}| ≤ 0.005.
- That G must pass V2 against the unchanged D3 reference (≤ 0.05).
- The doc 23 §5 synthetic result (G = 5 passes, G = 3 misses by 0.0007) is informative only.

**Reported, not gated:** at the fallback optimum, the fixed-placement gradient is compared with a finite-difference gradient of the objective with moving node placement. The maximum relative difference is reported.

---

## 3. Runtime accounting (replaces the B0 convention) [P]

**Unit.** One **node-hour** = one node allocated for one hour of wall-clock time.
- Measured jobs: node-hours = AllocNodes × ElapsedRaw / 3600 from SLURM accounting (`sacct`).
- Nodes are allocated exclusively (`OverSubscribe=EXCLUSIVE`, 64 logical CPUs). So the **process count within a node does not change node-hours.** It changes wall time.
- Every report also states the process count and the CPU-hours (processes × wall hours), for information.

**The 24 node-hour limit applies to the projected total for Rev 2 §6 steps 3–7.** Steps 3–7 are V2 and order checks, fixed-item recovery, S1-a, full-pipeline recovery and real-data B11. The projection:
- is computed at a **declared process count per node** (proposed: 32);
- is the sum over components of nodes × projected wall time;
- spreading work over several nodes does not reduce node-hours.

**The per-fit limit:** one complete fit must take ≤ 60 min of wall time on one node at the declared process count.

**The same convention applied to the measured baseline and to the optimised code** (from SLURM records and the run files) [O]:

| run | nodes | processes | wall | node-hours |
|---|---|---|---|---|
| B0 job 13825 (whole job) | 1 | 1 | 2 h 50 min | 2.83 |
| — B0 Q = 10 fit | 1 | 1 | 50.3 min | 0.84 |
| re-benchmark job 13835 (whole job) | 1 | 1 and 32 | 1 h 59 min | 2.00 |
| — optimised Q = 10 fit, serial | 1 | 1 | 55.9 min | 0.93 |
| — optimised Q = 10 fit, pooled | 1 | 32 | 7.1 min | 0.12 |

**Re-statement of the doc 22–23 projections in this unit** (pilot steps 3–7, Q = 10, Laplace only; 0.6 node-hours of components still assumed [A]):
- 18.0 node-hours at 1 process (B0 code, and also the optimised code serial, since both fits are dominated by penalty evaluations);
- **3.0 node-hours at 32 processes** (optimised).

Every projection states which inputs are measured [O], scaled estimates [I] or assumptions [A].

---

## 4. Tests and pass/fail criteria (to be run after approval)

**Regression reference for P1/P2.** The current optimised `m2_quad.py` is saved verbatim (with its SHA-256) as a second reference, `m2_quad_reference_v2.py`. The B0 reference stays.

| id | test | pass criterion |
|---|---|---|
| T1 | All existing M2 tests (`test_m2_quad.py` 13, `test_m2_grm.py`, `test_m2_recovery_diag.py`) | all pass, with tolerances unchanged |
| T2 | **Identity at regular ψ.** Values and gradients against reference v2, on synthetic dataset 0 (full cohort), at ψ = truth, the doc 23 fit optimum and 3 perturbations (seed `[42, 46, 7, k]`); serial and pooled | bitwise equal, or ≤ 1e-12 relative, value and gradient |
| T3 | **Extreme ψ.** At the three recorded penalty points (§1.1), on the 60-patient subset | penalty returned, with reason `mode_budget` or `non_nd`; wall time ≤ 10 % of reference v2's (65, 57 and 106 s) |
| T4 | **Path identity.** A full Laplace Q = 10 fit on the 60-patient subset and on the full synthetic cohort, against reference v2 | the same sequence of evaluated ψ (≤ 1e-12), the same final ψ and negative log-likelihood. Every P1/P2 trigger occurs at a ψ where reference v2 also returned the penalty. **Any trigger at a ψ where reference v2 returned a finite value fails T4,** and is reported with ψ and reason |
| T5 | **P2 unit test.** A constructed patient whose returned Hessian is not negative definite but has det > 0 | optimiser context: penalty `non_nd`. Reported-evaluation context: the patient's value comes from the fallback at the declared G, and the count is reported |
| T6 | **Fallback warm-start equivalence** (G = 3, Q = 10, synthetic dataset 0, pooled) | \|Δ negative log-likelihood\| ≤ 0.1 between the warm and cold optima; both satisfy the B11 longitudinal criteria |
| T7 | **Fallback multi-start** (G = 3 and 5) | three starts within 0.1; ‖∇‖ < 1e-3 × patients; no bound hits; positive-definite Hessian |

---

## 5. Benchmarks to run after approval (measured; on node002, the B0 node)

Synthetic dataset 0, items at truth, 32 processes unless stated. Q = 10 is used for timing only; it is **not** a selection.

1. **Primary, with P1/P2:** likelihood and likelihood + gradient at Q = 7, 10, 15 and 20 (serial and pooled), to show no regression against doc 23. One complete Q = 10 fit, serial and pooled, with per-evaluation times and trigger counts (§2.3).
2. **Fallback, warm-started:** complete fits at G = 3 and G = 5 (Q = 10) from ψ̂_L: wall time, evaluations, convergence. Plus one fallback finite-difference Hessian (38 likelihood-plus-gradient evaluations), timed.
3. **The T6 cold-start fallback fit at G = 3,** timed.
4. **Projections under §3,** for two scenarios:
   - **(i) primary-only pilot** at Q = 7, 10, 15, 20: measured fit at Q = 10; other Q scaled [I];
   - **(ii) fallback pilot:** each fit = Laplace fit + warm-started fallback fit + fallback Hessian, at G = 3 and G = 5.

**Runtime pass/fail:**
- **Primary:** one fit ≤ 60 min at 32 processes, and the projected scenario (i) ≤ 24 node-hours.
- **Fallback:** one warm-started fallback fit ≤ 60 min, and the projected scenario (ii) ≤ 24 node-hours at the G that the accuracy rule would require.
- The accuracy rule itself is evaluated in Step 3. Both G = 3 and G = 5 are reported now.
- **If the fallback fails the runtime criterion, that limitation is reported. Its accuracy requirement is not relaxed** and no alternative design is substituted.

**Then stop for review. Step 3 stays unauthorised.**

---

## 6. Review sequence and isolation

1. Approve, amend or reject this addendum. If approved, freeze its SHA-256.
2. Implement only P1, P2, monitoring (§2.3) and F1 in `src/ppmi_subtype/m2_quad.py`.
   - Add `m2_quad_reference_v2.py` (a verbatim copy, made before the change), `tests/test_m2_quad_addendum_a.py`, and a benchmark script `scripts/48c_m2a1_addendum_a_benchmark.py`.
   - `m2_grm.py`, `m2_quad_reference.py`, docs 16, 20 and 21, and every frozen output stay unchanged.
3. Run T1–T7, then the §5 benchmarks (new run directories).
4. Report, and stop for review.

**Not run under this addendum:** real-data fitting, V2 on the pilot patients, §4.1 order checks, recovery, fidelity gates, calibration, positive controls and evaluation statistics. Nothing is staged, committed or pushed.

## 7. Decisions for review

1. **P1 thresholds:** 10 consecutive non-negative-definite iterations, or 200 cumulative halvings; penalty `mode_budget`.
2. **P2:** non-negative-definite at any patient means infeasible inside the optimiser; at reported evaluations, Rev 2 §2.5 fallback as written. This corrects the prototype's det-sign test.
3. **Retain the absolute 1e-12 line-search tolerance** (no scale-aware tolerance; §1.3), plus monitoring (§2.3).
4. **F1:** the warm-started fallback, its 3-start definition (base ψ̂_L), the cold-start equivalence test, and the unchanged accuracy checks in Step 3.
5. **Runtime accounting (§3):** node-hours = allocated nodes × wall time, at a declared 32 processes per node, applied to baseline and optimised alike.
6. **Tests T1–T7, the §5 benchmarks and their pass/fail rules.**
