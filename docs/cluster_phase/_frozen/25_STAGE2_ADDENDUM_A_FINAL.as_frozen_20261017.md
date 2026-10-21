# Amendment 1 (Revision 2), Addendum A: final implementation specification

**Date:** 2026-10-17.

**Status.** This is the **implementation specification authorised for this execution** by the user's instruction "Stage 2 M2 Addendum A — Complete Implementation and Benchmark Execution Instructions" (2026-10-17).
- The document does **not** claim that it, or any result under it, was independently reviewed or approved by a human beyond that instruction.
- It was frozen (SHA-256 in `docs/cluster_phase/_frozen/25_STAGE2_ADDENDUM_A_FINAL.sha256`, with a read-only copy alongside) before any code change.

**Derived from:** the proposal `24_STAGE2_AMENDMENT1_REV2_ADDENDUM_A_PROPOSED_20261017.md` (SHA-256 `dfe24bb2…c32b`), which is left unaltered. That proposal's §1 contains the investigation (extreme-ψ cost; the conformance gap).

**Amends:** Revision 2 (`21_…REV2…`, `c53c015f…267`).

**Preserved unchanged** (verified before this document was written; recorded in `experiments/20261017T130820Z_stage2_m2a1_addendumA/00_baseline.json`):
- doc 16 (`c2eace18…bbc`);
- Revision 1 (`c0209941…bd`);
- Revision 2;
- all frozen D1 outputs.

The D1 verdict, NOT REPRODUCED, stands. Scope: full three-state label primary, D1 ratio-defined cohort secondary.

---

## 1. What is preserved (binding)

Everything in doc 16, Rev 2 and the proposal §0 that is not explicitly changed in §2 below.

**Numerical tolerances, all unchanged:**
- order checks: 0.005 against Q_ref = 20 (checked against 25) and G_ref = 7;
- V2 ≤ 0.05 against the unchanged D3 reference;
- §2.5 mode-uniqueness: 1e-6;
- V5.

**B11, all thresholds unchanged:**
- item fit: 3 starts within 0.1; a ∈ [0.05, 20];
- longitudinal fit: ‖∇‖ < 1e-3 × patients; no bound hits; positive-definite Hessian; 3 starts within 0.1;
- sampler: R̂ < 1.01, ESS ≥ 100.

**Also unchanged:**
- the recovery definitions and criteria;
- G1–G7, the G4 rule and the gate order;
- L-BFGS-B options (`ftol` 1e-15, `gtol` 1e-7, `maxiter` 1000, `maxfun` 4000) and the ψ bounds;
- the prior, the likelihood and the quadrature definitions;
- the mode convergence rule (step < 1e-9, at most 60 iterations);
- the absolute line-search acceptance tolerance (Φ_new ≥ Φ_cur − 1e-12; at most 50 halvings per iteration).

**Q is not selected here and is not capped.** It is chosen only by the Rev 2 §4.1 rule in Step 3. The fallback keeps the full tensor-product Gauss–Hermite grid over b. There is no sparse grid.

---

## 2. Authorised changes

### P1: work budget for the per-patient mode search

For each patient, the mode search keeps two counters:
- (a) the number of consecutive iterations, counted at the start of each iteration, at which the Hessian of the approximate objective is not negative definite (eigenvalue test);
- (b) the cumulative number of line-search halvings. Rejecting the full step counts as one halving; each further halving adds one.

**Triggering.** When (a) reaches **10** or (b) reaches **200** for any patient, the search for that patient group stops. The cohort objective at that ψ returns the existing invalid-objective penalty (value 1e12, zero gradient), with reason **`mode_budget`**.

**Other rules:**
- In the serial path, evaluation stops at the first penalised chunk; the value is the penalty either way.
- The absolute 1e-12 acceptance tolerance is unchanged. No scale-aware tolerance is introduced.
- The budget **is not mathematically equivalent** to the unbudgeted search at every ψ. It can penalise a ψ at which the unbudgeted search would eventually reach a valid mode. T4 detects that case.

### P2: negative-definiteness semantics (Rev 2 §2.5)

The determinant-sign test is removed as the criterion. Negative definiteness is tested by eigenvalues.

**(a) Optimiser context** (the objective used by L-BFGS-B):
- If any patient's Hessian at the returned mode is not negative definite, the ψ is infeasible: the penalty, with reason **`non_nd`**.
- Reasons for other existing penalties are recorded: `linalg` (linear-algebra failure) and `nonfinite`.

**(b) Reported-evaluation context** (`evaluate_reported`):
- Patients whose returned Hessian is not negative definite are evaluated by the AGQ fallback at a **declared G** (an explicit argument), as Rev 2 §2.5 specifies.
- The count of fallback-evaluated patients and of budget triggers is returned.
- **In this execution, the declared G for reported evaluations is 5.** That is a provisional declaration for reporting, **not** a selection of G, which Step 3 makes.
- A non-zero fallback count, or any budget trigger, at a reported fit optimum **fails** that fit's B11 longitudinal check.

**AGQ fallback objective.** It is itself the remedy for non-negative-definite patients. It places nodes with the prior curvature where the Hessian is not negative definite (Rev 2 prototype behaviour, unchanged). It has **no** `non_nd` penalty, but P1 applies to its mode search.

### Monitoring (report only; does not change values or the optimiser path)

Per objective evaluation, recorded in an in-memory log and written to the run directory:
- the ψ evaluated; the value; the gradient norm;
- the number of patients evaluated;
- the counts of non-negative-definite patients and of penalty triggers by reason;
- total line-search halvings;
- wall time and per-patient time summaries (per-chunk wall ÷ chunk size: median and max);
- max |Φ| at the returned modes, and spacing(max |Φ|), with a flag when > 1e-13.

### F1: warm-started fallback

1. Run the primary Laplace fit at the same Q, giving ψ̂_L.
2. Run the fallback (AGQ over b, G nodes per dimension, Q × Q over ε) fit with L-BFGS-B, the options and bounds unchanged, from:
   - start 1: ψ̂_L;
   - starts 2 and 3: ψ̂_L + N(0, 0.2²) on every component, seeds `[42, 46, 8, k]`, k = 1, 2, clipped to the bounds.

   Per-chunk modes are warm-started from the Laplace modes at ψ̂_L.
3. **Cold start (T6 only):** default ψ, no warm modes.
4. **For every fallback optimum, B11 longitudinal criteria:**
   - ‖∇‖ < 1e-3 × 934 = 0.934 (the fixed-placement gradient, Rev 2 §2.3);
   - no bound hits;
   - the finite-difference Hessian (central, step 1e-4, 2 × 19 = 38 likelihood-plus-gradient evaluations) positive definite;
   - and, for the 3 starts, max − min negative log-likelihood ≤ 0.1.

---

## 3. Tests and pass/fail criteria

| id | test | pass criterion |
|---|---|---|
| T1 | Existing tests, unchanged: `test_m2_quad.py`, `test_m2_grm.py`, `test_m2_recovery_diag.py` (plus the directly affected `test_grm.py`, `test_grm_ou.py`) | all pass |
| T2 | Updated code against `m2_quad_reference_v2.py` (SHA-256 `23c2fde7…fae8`) on synthetic dataset 0 (full cohort), serial and pooled (32), at: ψ = truth; **the doc 23 optimum**; and 3 perturbations of that optimum, N(0, 0.05²), seeds `[42, 46, 7, k]`, k = 1–3 | each value and gradient bitwise equal, or relative difference ≤ 1e-12. The criterion met is stated per case |
| T3 | The three penalty evaluations of reference v2's full Q = 10 fit path on the 60-patient subset (the previously recorded extreme points), serial | the penalty is returned with its reason, and the updated wall time is ≤ 10 % of reference v2's at each point. Differences are recorded |
| T4 | Full Q = 10 fits, updated code against reference v2, both through the same L-BFGS-B driver: (i) 60-patient subset, serial; (ii) full synthetic cohort, pool of 32 | the same number of evaluations; every evaluated ψ, the final ψ and the final negative log-likelihood within 1e-12 (absolute, componentwise); the same convergence status. **Any P1/P2 trigger at a ψ where reference v2 returns a finite value fails T4,** and is reported with ψ, both values and the reason |
| T5 | Unit test: a constructed patient Hessian that is not negative definite with det > 0 | optimiser context → `non_nd` penalty; reported context → fallback value at the declared G and count = 1; reasons recorded |
| T6 | Fallback G = 3, Q = 10, synthetic dataset 0, pool of 32: warm start 1 against the cold start | \|Δ negative log-likelihood\| ≤ 0.1, and both optima satisfy the §2 F1(4) criteria |
| T7 | Fallback G = 3 and G = 5: the 3 warm starts each | negative log-likelihood spread ≤ 0.1, and every start satisfies F1(4). Reported per start: ‖∇‖, bound hits, minimum Hessian eigenvalue, convergence status, wall time |

The doc 23 optimum in T2 is the final ψ of reference v2's pooled full-cohort fit in T4(ii). It must reproduce doc 23's negative log-likelihood 57,022.7423 (to 1e-4) and its 104 evaluations, which identifies it as that optimum. doc 23 did not store ψ.

**Stopping rules:**
- **T2 failure:** all later benchmarks are skipped. The implementation is not equivalent at regular ψ.
- **T4 failure, either part:**
  - the primary complete-fit benchmarks are labelled non-equivalent and are not used for the runtime gate;
  - the fallback sequence runs only if the updated Laplace fit's final ψ still equals reference v2's within 1e-6, componentwise (same warm-start point); otherwise it is skipped.
  - Microbenchmarks, which are path-independent, still run.
- **T1, T3, T5, T6 and T7 failures** are recorded. They do not stop independent checks.

---

## 4. Benchmarks

**Setting:** node002, the B0 node (Xeon Gold 6326), one exclusive node. BLAS single-threaded per process; the pool has 32 processes. Synthetic dataset 0 (items at truth). If node002 is unavailable, the job waits; no other node is used without a documented deviation.

**Primary microbenchmarks** (updated code): Laplace Q ∈ {7, 10, 15, 20}, likelihood and likelihood plus gradient, at ψ = truth.
- Serial: 3 timed repetitions after a warm-up.
- Pooled: 5 timed repetitions after a warm-up.
- Median, minimum and maximum reported.

**Primary complete fits** (updated code, Q = 10):
- pooled: the T4(ii) run;
- serial: a separate run.

Recorded: evaluations, iterations, wall time, the per-evaluation time distribution, and counts of penalties, non-negative-definite patients and fallback. Then `evaluate_reported` at the optimum (declared G = 5).

**Fallback** (Q = 10, pool of 32):
- warm starts 1–3 at G = 3, then at G = 5;
- the cold start at G = 3;
- one finite-difference Hessian per optimum (38 likelihood-plus-gradient evaluations), timed.

**Accounting.**
- Node-hours = allocated exclusive nodes × elapsed wall hours (SLURM `AllocNodes` × `ElapsedRaw`), with the process count stated.
- Projections use the declared 32 processes per node.
- **Primary pilot** (Rev 2 §6 steps 3–7): 18 fits × the measured pooled fit wall time (Q ≠ 10 scaled by the measured pooled likelihood-plus-gradient ratio [estimate]); plus 11 Hessians × 38 × the measured pooled likelihood-plus-gradient time; plus stage-1 item fits (observed 2 × 358 s); plus S1-b refits and checks (assumed 0.62 h, as before).
- **Fallback pilot at G:** 18 × (pooled Laplace fit + mean warm fallback fit at G) + 11 fallback Hessians at G + the same other components.
- Each input is labelled measured, estimated or assumed.

**Runtime gates:**
- **Primary:** one pooled complete fit ≤ 60 min, and the projected primary pilot ≤ 24 node-hours. Reported for every Q.
- **Fallback:** one warm fallback fit ≤ 60 min, and the projected fallback pilot ≤ 24 node-hours, at the G the accuracy rule will require. That is unknown until Step 3, so it is reported for **G = 3 and G = 5 separately**.
- If the G that is ultimately required fails, the fallback is computationally infeasible under the current budget. The accuracy requirement is not relaxed and no other method is substituted.

---

## 5. Boundaries and stopping point

**Not run:**
- real-data fitting;
- V2 on the pilot patients;
- the Step 3 order checks;
- recovery;
- real-data B11;
- G1–G7;
- calibration;
- positive controls;
- evaluation statistics;
- clustering.

**Interpretation.** Synthetic results show computational behaviour only. They are not evidence that the model explains PPMI data, that Q = 10 is accurate enough, or that the fallback is accurate on real patients.

**On completion:** the reports are written, then **stop**. Step 3 needs a separate explicit approval. Nothing is staged, committed or pushed.

## 6. Remaining limitations (known before execution)

- P1 may change the optimiser path at some ψ; T4 measures this.
- The fallback's fixed-placement gradient is not the exact gradient of the moving-node objective. Its effect on L-BFGS-B convergence is part of what T6 and T7 show.
- All runtime evidence is synthetic, and projections include assumed components (0.62 node-hours).
- The required fallback G is unknown until Step 3.
