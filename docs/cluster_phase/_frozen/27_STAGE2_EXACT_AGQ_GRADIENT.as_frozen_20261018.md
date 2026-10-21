# Exact gradient for the adaptive-quadrature (AGQ) fallback: proposed implementation specification

**Date:** 2026-10-18.

**Status.** This is a **proposed implementation specification**, authorised for this execution by the user's instruction "Stage 2 M2 — Exact-Gradient Adaptive-Quadrature Fallback: Implementation and Validation" (2026-10-18).
- It is **not** an independently approved change to the scientific protocol.
- It is frozen (SHA-256 in `docs/cluster_phase/_frozen/27_STAGE2_EXACT_AGQ_GRADIENT.sha256`, with a read-only copy) before any code change.

**Context:** `26_STAGE2_M2A1_ADDENDUM_A_EXECUTION_REPORT_20261018.md`.
- Fallback G = 3 failed the convergence criteria (T6, T7).
- G = 5 converged but failed the runtime limits.
- Doc 26 offered a hypothesis: the fallback's *fixed-placement* gradient is not the gradient of the objective the optimiser evaluates.

**Execution directory:** `experiments/20261017T233307Z_stage2_m2_exact_agq/` (`00_baseline.json`).

**Unchanged** (verified there before this document):
- docs 16, 20, 21, 24, 25 and 26, and their frozen copies;
- `m2_grm.py`;
- `m2_quad_reference.py` (B0), `m2_quad_reference_v2.py` (doc 23), and the new read-only `m2_quad_reference_v3.py` (`33c02d6a…`, the code benchmarked in job 13838);
- all D1 outputs. The D1 verdict, NOT REPRODUCED, stands.

---

## 1. What changes and what does not

**Changes:** only the **gradient** of the AGQ fallback objective. A new gradient option implements the exact total derivative (§3). It is used by a new objective method, `agq_exact`, for fallback fits.

**Unchanged:**
- the AGQ objective's value;
- the full tensor-product Gauss–Hermite grid (G⁴ nodes over b; Q × Q over ε);
- the declared G values (3 and 5) and G_ref = 7;
- every accuracy criterion (0.005 order tolerance, V2 ≤ 0.05 against D3);
- the B11 and convergence criteria (‖∇‖ < 1e-3 × 934 = 0.934; 3 starts within 0.1; no bound hits; positive-definite Hessian);
- the T6/T7 definitions, data, seeds and bounds (doc 25 §2 F1, §3);
- the prior and the likelihood;
- the primary Laplace objective and gradient;
- P1/P2 safeguards and monitoring;
- the absolute line-search tolerance 1e-12;
- L-BFGS-B options;
- the runtime limits (one warm fallback fit ≤ 60 min; projected fallback pilot ≤ 24 node-hours at the G that the accuracy rule requires);
- the node-hour accounting convention.

**Not introduced:** a sparse grid or any other integration method.

**Existing behaviour retained for regression:**
- `m2_quad.group_agq` keeps its fixed-placement gradient as the default. That is the gradient existing tests compare against.
- The exact gradient is an explicit option, `gradient="exact"`.
- When the node placement is supplied (`fixed=(b, L)`), the fixed-placement gradient *is* the exact gradient of that fixed-node function.

---

## 2. The objective (as implemented in `m2_quad.group_agq`, unchanged)

For patient i, with per-visit Q × Q quadrature over ε (Rev 2 §2.2):

  Φ(b; ψ) = Σ_j log L̃_ij(b; ψ) − ½ bᵀ D⁻¹ b,

where L̃_ij depends on ψ through the mean parameters (μ_ij), through L_ε (node positions C_ε u_q) and through the items (fixed).

Mode: b̂(ψ) = argmax_b Φ(b; ψ), by the P1-guarded damped Newton search.

Curvature: H(ψ) = −∇²_b Φ(b̂; ψ) if negative definite at b̂ (all patients at regular ψ); otherwise H = D⁻¹ (unchanged prototype rule).

Cholesky factor: L(ψ) L(ψ)ᵀ = H⁻¹, lower-triangular.

Nodes: z_g (G⁴ tensor GH nodes in standard units), log weights w_g; b_g = b̂ + L z_g.

**Objective:**

  ℓ_i(ψ) = log Σ_g exp(w_g + Φ(b_g; ψ) + ½|z_g|²) + log|L| − ½ log|D|.

This is the adaptive Gauss–Hermite estimate of log ∫ Π_j L̃_ij(b) φ₄(b; 0, D) db. That is **the same integral** the Laplace primary approximates:
- with G = 1 (z = 0, weight 1), ℓ_i = Φ(b̂) − ½ log|H| − ½ log|D|, exactly the Laplace value (Rev 2 §2.3);
- both use the same per-visit L̃ (the same Q).

Weights: ω_g = softmax_g(w_g + Φ(b_g) + ½|z_g|²).

## 3. Exact derivative

**What depends on ψ:**
- Φ, explicitly, through μ, L_ε and D;
- b̂, through the mode condition;
- H, through Φ and through b̂;
- L, through H;
- the nodes b_g, through b̂ and L;
- log|D| and log|L|.

The constants z_g and w_g do not depend on ψ.

**Total derivative,** for a ψ component p:

  dℓ_i/dψ_p = T_A + T_B + T_C + T_D + T_E,

| term | formula | what it is |
|---|---|---|
| T_A | Σ_g ω_g ∂_pΦ(b_g) | explicit dependence at the nodes. **This is the current fixed-placement gradient** |
| T_B | m_gᵀ db̂/dψ_p, with m_g = Σ_g ω_g ∇_bΦ(b_g) | node shift from the moving mode |
| T_C | Σ_{k,l} (dL/dψ_p)_{kl} A_{kl}, with A = Σ_g ω_g ∇_bΦ(b_g) z_gᵀ | node rescaling from the moving curvature |
| T_D | d log\|L\|/dψ_p = Σ_k (dL/dψ_p)_{kk} / L_{kk} | Jacobian of the node placement |
| T_E | −½ tr(D⁻¹ dD/dψ_p) | D parameters only (as now) |

**Mode derivative** (implicit-function theorem at a stationary point, ∇_bΦ(b̂; ψ) = 0):

  db̂/dψ_p = H⁻¹ ∂²Φ/∂b∂ψ_p.

**Curvature derivative:**

  dH/dψ_p = ∂H/∂ψ_p + Σ_k (∂H/∂b_k) (db̂/dψ_p)_k.

Here ∂H/∂b_k = −∂³Φ/∂b∂b∂b_k, and ∂H/∂ψ_p = −∂³Φ/∂b∂b∂ψ_p. For the D parameters, ∂H/∂ψ_p = −D⁻¹ dD D⁻¹ and ∂²Φ/∂b∂ψ_p = D⁻¹ dD D⁻¹ b̂.

**Cholesky-factor derivative.** With S = H⁻¹, dS = −H⁻¹ (dH) H⁻¹, and

  dL = L · Ψ(L⁻¹ dS L⁻ᵀ),

where Ψ takes the lower triangle and halves the diagonal (standard; from dS = dL Lᵀ + L dLᵀ with L⁻¹dL lower-triangular).

**Every derivative of Φ is analytic and already implemented and tested:**
- ∂_pΦ(b_g) and ∇_bΦ(b_g) at each node come from the per-visit node-posterior expectations of the first θ-derivatives along the b and ψ directions (`_cumulants`' first moments). These are already computed in the AGQ node loop.
- ∂²Φ/∂b∂ψ_p and the third derivatives at b̂ are exactly the quantities the Laplace gradient uses (`_cumulants`: D2, D3). Their correctness is established by the Laplace gradient's finite-difference test (doc 22 V1, test 4).

**No automatic differentiation and no numerical differentiation** are used in the gradient. Finite differences are used only in tests and for the B11 Hessian.

**Check.** With G = 1: T_A = ∂_pΦ(b̂); T_B = 0 (∇_bΦ(b̂) = 0); T_C = 0 (z = 0); and T_D = −½ d log|H|. So dℓ = the Laplace gradient. A test enforces this equality (§5 B2).

**Sanity property** (reported, not gated). If the G-point rule were exact, ℓ_i would not depend on the node placement, and T_B + T_C + T_D would vanish. Their size measures how far the finite-G rule's value depends on placement, i.e. why the fixed-placement gradient is inexact.

## 4. Conditioning, invalid modes and failures

- **The implicit derivative requires b̂ to be a nondegenerate stationary point,** with ∇²Φ(b̂) negative definite.
  - For patients whose returned Hessian is **not** negative definite (where the objective uses H = D⁻¹), the formula is evaluated with dH = d(D⁻¹) and the implicit b̂ derivative through ∇²Φ(b̂). It is **not guaranteed exact.**
  - Such patients are counted per evaluation as `gradient_inexact_patients`.
  - **Convergence is assessed only at optima where this count is 0.** A non-zero count at a reported fallback optimum fails that fit's F1(4) assessment.
- **The mode is solved to a Newton step < 1e-9** (unchanged). The residual stationarity error enters the gradient at second order through db̂. The finite-difference tests (§5 A) quantify its effect.
- **P1 still applies.** A mode-budget trigger returns the existing penalty (1e12, zero gradient), as now.
- **Linear-algebra failures** (Cholesky of H⁻¹, solves with H) return the existing `linalg` penalty.
- **A Cholesky factor with a non-positive diagonal** (it cannot occur for symmetric positive-definite H⁻¹) is treated as `linalg`.

---

## 5. Tests and pass/fail criteria (fixed now, before any result)

**New file:** `tests/test_m2_quad_exact_agq_gradient.py`.

**A. Finite-difference gradient checks of the same moving-node objective.**
- **Setting:** small synthetic patients (n = 1–4 visits; 4 tremor and 3 PIGD items), Q = 7, at:
  - the test ψ;
  - two perturbations;
  - one perturbation with mean parameters shifted by 0.5, which moves b̂ materially;
  - for G = 3 and G = 5.
- **Finite differences:** central differences of the AGQ objective with nodes re-placed at ψ ± h (the mode re-solved), at h ∈ {1e-2, 1e-3, 1e-4, 1e-5}.
- **Criterion:** at h = 1e-4, for every ψ component, |g_exact − g_FD| ≤ 1e-5 · max(1, |g_FD|).
- **Rationale for the tolerance:**
  - central-difference truncation is O(h²) ≈ 1e-8 × third-derivative scale;
  - rounding is about ε |ℓ| / h ≈ 2e-16 × 10² / 1e-4 = 2e-10;
  - the mode-solve error is bounded by the 1e-9 Newton step, acting through the placement sensitivity (about the quadrature error, ≤ 1e-1), i.e. ≤ 1e-10 / h = 1e-6.
  - So 1e-5 is above the noise and far below the expected fixed-placement discrepancy.
- **Reported:** the discrepancy for each h (showing O(h²) decrease until rounding dominates), and **the fixed-placement gradient's discrepancy at h = 1e-4, as a contrast only**. It is never the correctness reference.

**B. Independent references.**
- **B1 (objective, reported):** AGQ values at G = 3, 5, 7 against `m2_grm.agq_one` (adaptive GH over the full θ, independent code) at high order, on 1- and 2-visit patients. The differences are reported and should shrink with G.
- **B2 (gradient, gated):** at **G = 1** the AGQ objective equals the Laplace objective (|Δ| ≤ 1e-10), and the exact AGQ gradient equals the Laplace gradient (≤ 1e-8 · max(1, |g|)), on synthetic cohorts at 3 ψ.
- **B3 (gradient, reported):** the exact AGQ gradient at G = 3, 5, 7 against central finite differences (h = 1e-4) of `m2_grm.agq_one` at high order, i.e. the gradient of a near-exact integral. The difference is reported. It should fall as G grows; not gated.
- Objective agreement is not used as evidence of gradient correctness.

**C. Regression.**
- **C1:** every existing M2 test (`test_m2_quad.py`, `test_m2_grm.py`, `test_m2_recovery_diag.py`, `test_m2_quad_addendum_a.py`, `test_grm.py`, `test_grm_ou.py`) passes with assertions unchanged.
- **C2:** the primary Laplace value and gradient are bitwise identical to `m2_quad_reference_v3` on synthetic data (unit test). In the benchmark job: synthetic dataset 0, full cohort, ψ = truth and 2 perturbations, serial and pooled.
- **C3:** the default `group_agq` value and fixed-placement gradient are bitwise identical to reference v3. The `agq_exact` value is bitwise identical to `agq` (the same objective).

**D. Synthetic fallback convergence** (benchmark job; the doc 25 definitions with `agq_exact`):
- **T6:** G = 3 warm start 1 against the cold start: |Δ negative log-likelihood| ≤ 0.1, and both meet F1(4).
- **T7:** G = 3 and G = 5, three warm starts each: spread ≤ 0.1, and each meets F1(4).
- **F1(4):**
  - ‖∇‖ < 0.934, now measured with the exact gradient;
  - no bound hits;
  - the finite-difference Hessian of the exact gradient is positive definite;
  - `gradient_inexact_patients` = 0 at the optimum.
- **Reported per run:** negative log-likelihood, evaluations, ‖∇‖, bound hits, minimum Hessian eigenvalue, optimiser status, runtime, criterion outcome, penalty / mode-failure / non-ND counts.

**E. In-job gradient gate (S1):** before any fallback fit, the exact gradient is compared with central finite differences (h = 1e-4) of the moving-node objective. Setting: the 60-patient synthetic subset, G = 3, Q = 10, ψ = the doc 23 Laplace optimum. Criterion as in A. **If A, B2 or S1 fails, the fallback benchmark sequence stops.**

## 6. Benchmarks and runtime criteria

**Setting:** node002 (Xeon Gold 6326), 1 exclusive node, 32 processes, single-threaded BLAS. Synthetic dataset 0 (items at truth), Q = 10. If node002 is unavailable, the job waits.

**Measured:**
- likelihood-plus-gradient evaluations (`agq_exact`), G = 3 and 5 (pooled: 3 repetitions after a warm-up);
- the Laplace primary fit (pooled; to obtain ψ̂_L and its modes, and for C2);
- warm fallback fits G = 3 × 3 and G = 5 × 3, and the cold G = 3 fit;
- a finite-difference Hessian (38 likelihood-plus-gradient evaluations) per optimum;
- per-evaluation logs with penalties, mode failures, non-ND and inexact-gradient counts.

**Accounting:** node-hours = allocated exclusive nodes × elapsed wall hours (SLURM `AllocNodes` × `ElapsedRaw`), reported separately from the projected pilot cost.

**Projected fallback pilot (doc 25 §4):** 18 × (Laplace fit + mean warm fallback fit) + 11 fallback Hessians + stage-1 item fits (716 s, observed) + S1-b refits and checks (0.62 h, assumed).

**Runtime gates (unchanged):**
- one warm fallback fit ≤ 60 min;
- projected fallback pilot ≤ 24 node-hours, **at the G the accuracy rule will require.**
- The accuracy rule is decided in Step 3. G = 3 and G = 5 are therefore reported separately, and G = 3 cannot be chosen because it is faster if it fails convergence or (in Step 3) the order tolerance.

**If no G meets every criterion,** the report says that no viable fallback has been established, and proposes the smallest next design decision without implementing it.

## 7. Boundaries and stopping point

**Not run:** Step 3 (V2 on the pilot patients, order selection); real-data fitting; recovery; real-data B11; G1–G7; calibration; positive controls; evaluation statistics; clustering.

**Synthetic results show computational and numerical-gradient properties only.** They do not show accuracy on real patients or anything scientific.

**After the reports, stop for review.** Nothing is staged, committed or pushed.

## 8. Known limitations before execution

- The exact gradient adds the per-patient third-derivative cumulants at b̂ (the same cost as one Laplace gradient) to each AGQ evaluation. The G⁴ node evaluations still dominate the cost.
- Exactness holds only where every patient's mode is a nondegenerate maximum (§4).
- An exact gradient may fix convergence without fixing runtime. G = 5's cost is dominated by the number of node evaluations, not by the gradient.
