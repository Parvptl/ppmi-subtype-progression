# Stage 2 M2: Amendment 1, Revision 3 — final text R1.3 (supersedes the doc 44 proposal)

**Date:** 2026-10-19.

> **FINAL (R1.3) — FROZEN together with doc 47** (locks: `_frozen/46_STAGE2_REV3_AMENDMENT.sha256`, `_frozen/47_STAGE2_REV3_RECOVERY_TEST_PLAN.sha256`; read-only copies alongside). Freezing, driver implementation and execution were authorised by the user's *Final Execution Directive* (2026-10-19).
> - Doc 44 (`6c63de6f…`) is kept unchanged as the record of the first proposal. This revision replaces it for review.
> - **Step 6 remains FAILED. D1 remains NOT REPRODUCED.** No frozen record is modified.
> - **Revision R1.1 (2026-10-19):** two notes in §3.5 (N4 status; N7 → flag F8) and the §3.6 calibration wording were corrected after a final review (doc 48). The pre-edit SHA-256 of this file was `2fc53785…`.
> - **Revision R1.2 (2026-10-19):** the eleven decisions approved in the user's *Final Decision Record R1.1* (2026-10-19) are recorded as approved (§§0, 3, 4, 7). No decision's substance was changed. The changes are listed in doc 49. The pre-edit SHA-256 of this file was `6ebd298a…`.
> - **Revision R1.3 (2026-10-19):** review findings B2, B3, B5 and M2 resolved: the computation paths of H, C, A and E are pinned (§3.2), and N1–N11 are made exact (§3.5). In particular, N3 and N4 now include the first-order gradient term that the relabelling identities require. The pre-freeze evidence is in `experiments/20261019T172456Z_stage2_rev3_prefreeze_checks/`. The changes are listed in doc 50. The pre-edit SHA-256 was `e9ad3a83…`.

**Tags:** [P] protocol rule (approved); [O] observation from earlier diagnostics (descriptive, five development datasets); [M] mathematical statement, with its numerical check.

**Change log against doc 44:**

| § | change | why |
|---|---|---|
| R1: 2.6 | **new** [M]: under the convention, β0′_s is determined by β1′_s, γ′_s and the design means | it changes what the β0 gates mean (doc 47 §6) |
| R1: 3 | **rewritten**: the full-chain sandwich specified as stacked M-estimation, with cross-stage dependence, the comparison with partial covariances, validity conditions, failure modes and required numerical checks. **No claim of validity is made from the D4 standard errors** | review item 2 |
| R1: 3.6 | **new**: small-sample caveat (198 stacked parameters, 934 patient clusters). Its calibration is to be judged only by the recovery test | review item 2 |
| R1: 7 | decisions updated (D2b added) | — |
| R1.3: §§3.2, 3.5, header | computation paths of H₁, H₂, C₁, C₂, A₁, A₃, E₃ pinned; exact N1–N11 definitions, including the gradient term in N3/N4; N6 threshold confirmed on saved matrices; tag legend | final execution directive; doc 50 |
| R1.2: §§0, 3, 4, 7 | approved decisions recorded (D1, D2, D2b, D3/D4, D6); §0 rewritten as the approved future scope | user decision record R1.1; doc 49 |

Unchanged from doc 44: §0 scope, §1 evidence, §2.1–2.5 convention and transformation, §4 S1-b role, §5 Le00_log, §6 limitations, §8 exclusions. They are restated below in condensed form, so this document stands alone.

---

## 0. Approved scope (decisions D3/D4, approved 2026-10-19)

**Effect.** Rev 3 takes effect only after docs 46 and 47 are frozen together (D6), and **only for estimates produced after that freeze**. On freezing, Rev 3 amends:
- **doc 16 B5** (item uncertainty ignored): replaced, for Rev 3 estimates, by the full-chain sandwich (§3; D2);
- **doc 16 D3 / Rev 2 §0** (the recovery definition): replaced, for Rev 3, by the independent recovery test of doc 47 (D4). The doc 16 D3 ≥ 4/5 rule is reported on fresh datasets 0–4 as a comparator only;
- **Rev 2 §5** (S1-b acceptance) and **Rev 2 §6 step 6**: the Rev 3 pipeline (one S1-b iteration, D1) must pass the doc 47 test, as step 6′;
- **doc 37 E-2** ("no re-standardisation"): superseded for future estimates by the convention of §2 (D3). E-2 remains the rule under which Step 6 was run and judged.

**Historical verdicts are preserved without re-scoring:**
- **Step 6 remains FAILED**;
- **D1 remains NOT REPRODUCED**;
- no earlier result is re-evaluated under Rev 3.

**Unchanged:**
- model M2, the item model, the Rev 2 likelihood (Q = 15), P1/P2, G = 5 for reported evaluations, doc 27;
- the other B11 criteria, G1–G7 and the gate order;
- calibration, positive controls and evaluation statistics stay blocked;
- Steps 7–8 and real-data analysis stay blocked pending the doc 47 result and its review.

## 1. Evidence (as doc 44 §1; docs 38, 40, 42)

- **[O]** Step 6 failed on β0_P, Le00_log and Le11_log.
- **[M]** The likelihood is invariant under joint latent relabelling.
- **[O]** On the truth's convention (D7): β0_P mean z −0.10, Le11_log −0.17, Le00_log −0.61. One S1-b iteration still changes L22_log and L33_log after re-expression.

All of this rests on five development datasets. It motivates the proposal; it does not validate it.

## 2. Convention and transformation [P] (doc 44 §2, unchanged; with §2.6 added)

**2.1–2.5.** For each subscore s:
- μ_s(ψ) = mean_v M_v^s(ψ);
- v_s(ψ) = mean_v K_vv^s(ψ) + var_v M_v^s(ψ), over all visits of the analysis design;
- c_s = √v_s(ψ̂), d_s = μ_s(ψ̂).

Then S(ψ̂) = relabel(ψ̂; c, d):
- β0′ = (β0 − d)/c;
- β1′ = β1/c;
- γ′ = γ/c;
- D′ = C₄⁻¹ D C₄⁻¹;
- Σ_ε′ = C₂⁻¹ Σ_ε C₂⁻¹;
- the items are mapped jointly: a′ = c·a, τ′ = (τ − d)/c.

**Properties:**
- the likelihood and predictions are invariant;
- S(relabel(ψ)) = S(ψ);
- S(S(ψ)) = S(ψ);
- S(ψ_true) = ψ_true, to ≤ 1e-8 (O(JITTER), doc 43).

The truth never enters S.

**2.6 [M] (new) What β0′ means under the convention.** M_v^s = β0_s + β1_s t_v + γ_s x_v, so μ_s = β0_s + β1_s t̄ + γ_s x̄. After S:

  **β0′_s = −(β1′_s · t̄ + γ′_s · x̄)**

where t̄ and x̄ are the design means: on the real design t̄ = 3.1255 years and x̄ = 0.5967. This was verified on ψ_true to all printed digits.

Under the convention, therefore:
- the re-expressed intercepts carry **no information independent of the slope and treatment terms**;
- the 19 re-expressed components span a 15-dimensional set (2 location and 2 variance constraints).

The original β0_P failure was a failure of the **latent location relative to the items**. It can be tested only by a quantity that involves the items: the item-anchored location of doc 47 §6.

## 3. Uncertainty: the full-chain sandwich, specified [P] (D2 approved: primary interval method; D2b approved: no post-hoc small-sample correction)

### 3.1 Estimating equations

Write Θ = (φ₁, ψ₁, φ₂, ψ₂) ∈ ℝ^{80+19+80+19}. For patient i (with visits v ∈ i):

| block | estimating function u_i^(k) (gradient of the patient's contribution to the negative log-objective actually optimised) | solved by |
|---|---|---|
| φ₁ (N(0, 1) items) | ∇_φ Σ_{v∈i} −log ∫ Π_k P(y_vk \| θ, φ) φ(θ) dθ (61-node GH; per-visit composite) | stage 1 |
| ψ₁ (first stage 2) | ∇_ψ −log p̃(Y_i \| ψ, φ₁), with p̃ the Rev 2 approximation (Laplace over b, 15 × 15 GH over ε; reported context: non-ND patients by G = 5 AGQ) | stage 2, step (2) |
| φ₂ (S1-b items) | ∇_φ Σ_{v∈i} −log ∫ Π_k P(y_vk \| θ, φ) N(θ; M_v(ψ₁), K_vv(ψ₁)) dθ | step (3) |
| ψ₂ (final stage 2) | ∇_ψ −log p̃(Y_i \| ψ, φ₂) | step (4) |

**Θ̂ solves U(Θ) = Σ_i u_i(Θ) = 0.** It is an M-estimator of the pseudo-true value Θ* with E u_i(Θ*) = 0. That value is a property of the approximate, two-stage procedure, **not** necessarily the truth.

### 3.2 Covariance

  V = A⁻¹ B A⁻ᵀ,   A = ∂U/∂Θᵀ |_Θ̂,   B = Σ_i u_i(Θ̂) u_i(Θ̂)ᵀ.

A is block-lower-triangular, because each stage depends only on its own parameters and those of earlier stages:

| block row | A₍k,1₎ | A₍k,2₎ | A₍k,3₎ | A₍k,4₎ |
|---|---|---|---|---|
| φ₁ | A₁ = ∇²_φ NLL₁ | 0 | 0 | 0 |
| ψ₁ | C₁ = ∂U₂/∂φ₁ | H₁ = ∇²_ψ NLL₂(·; φ₁) | 0 | 0 |
| φ₂ | 0 | E₃ = ∂U₃/∂ψ₁ | A₃ = ∇²_φ NLL₃ | 0 |
| ψ₂ | 0 | 0 | C₂ = ∂U₄/∂φ₂ | H₂ = ∇²_ψ NLL₂(·; φ₂) |

**How each block is computed (R1.3; this fixes the computation path):**

| block | computation (existing code, used unchanged) | objective context |
|---|---|---|
| H₁, H₂ | `m2_quad.fd_hessian(Objective(groups, items, Q = 15, jobs = 32), ψ̂, h = 1e-4)`, with the Objective's mode state set to the converged fit's state, exactly as Step 6 (`scripts/52`, `Ops.hessian`). The result is symmetrised by `fd_hessian`. `run_d4` does **not** compute H: D4 used Step 6's saved H₁ and H₂ | optimiser context (P1/P2 penalties active) |
| C₁, C₂ | `scripts/54` helpers `raw_chunks`, `eval_s2`, `_s2_task`: a **cold** base evaluation at (ψ̂, φ̂) giving per-chunk modes, then central differences over the 80 packed item parameters (h = 1e-4), warm-started from the base modes, 64-patient chunks, 32 workers | reported context (non-ND patients by G = 5 AGQ, exact gradient) |
| A₁, A₃, E₃ | `scripts/54` `_item_grad_task`: central differences (h = 1e-4) of the analytic item-objective gradients (`grm.stage1_negloglik`, `latent_negloglik` of `scripts/51`, the latter with `visit_latent` at ψ̂₁ ± h) | item objectives (no penalties) |
| per-patient scores u_i | stage 2: per-patient gradients from the cold base evaluation (reported context). Items: `stage1_negloglik` / `latent_negloglik` evaluated on each patient's visits (`patient_slices`) | as above |

**Penalties.** If any finite-difference evaluation used for H₁ or H₂ returns the P1/P2 penalty (recorded in the Objective log), or any C evaluation reports a chunk failure or a non-finite value, the dataset receives flag F6. If the resulting V is non-finite, not PSD or not computable (singular A), its intervals are undefined, and the dataset receives flag F8 (doc 47 §7).

**Code path for the regression check.** The new driver reproduces this path with the same functions. The Phase 1 regression (doc 47 §8, R-c) checks the result against D4.

**For the transformation:** V_S = G V_{ψ₂ψ₂} Gᵀ, with G = ∂S/∂ψ₂ (central differences, h = 1e-6; `scripts/55` `jacobian`). S depends only on ψ₂ and the design.

**For the doc 47 functionals F(ψ₂, φ₂):** V_F = J_F V_{(φ₂,ψ₂)} J_Fᵀ.

### 3.3 How cross-stage dependence enters

1. **Through A's off-diagonal blocks** (C₁, E₃, C₂). They are the sensitivity of each later stage's equations to the earlier stages' estimates, so errors in φ₁ propagate to ψ₁, then to φ₂, then to ψ₂.
2. **Through B's off-diagonal blocks.** The same patients' responses enter every stage, so the per-patient scores of different stages are correlated. The sandwich uses the per-patient **stacked** score, which keeps these covariances. On the development data the cross terms partly cancel: the two-step correction at fixed ψ₁ gave a β0_P SE ratio of 0.75.

### 3.4 Contrast with simpler covariances

| covariance | what it ignores | development-data behaviour (raw scale; descriptive) |
|---|---|---|
| fixed-item H₂⁻¹ (Step 6) | all item uncertainty; model-based, not robust | the baseline |
| stage-2 sandwich H₂⁻¹B₄₄H₂⁻¹ | item uncertainty; only robustifies stage 2 | ratio about 1.00 |
| two-step at fixed ψ₁ (blocks φ₂, ψ₂ only) | the dependence of φ₂ on ψ₁ and of ψ₁ on φ₁ | ratios 0.75 (β0_P) to 1.33 (Le00_log) |
| **full chain (primary method, D2 approved)** | nothing in the chain; it rests on the asymptotic approximation | ratios 1.11–1.65 raw; **0.92–1.12 after S** |

### 3.5 Required numerical checks (exact definitions, R1.3; each reported per dataset)

**Notation.** v_φ, v_ψ are the relabelling tangents, from `scripts/54` `affine_tangent` (central differences, e = 1e-6), for subscore s ∈ {T, P} and direction ∈ {c, d}.

**Why the gradient term appears.** Let T_c be the relabelling. Differentiating the invariance F(T_c ψ, T_c φ) = F(ψ, φ) gives

  H v_ψ + C v_φ − m_s ⊙ g = 0 for the c-direction, and H v_ψ + C v_φ = 0 for the d-direction,

where g is the gradient of the stage's objective at the estimate, and m_s marks the coordinates whose Jacobian under T_c depends on c:
- **ψ:** m_T = {β0_T, β1_T, γ_T, L10}; m_P = {β0_P, β1_P, γ_P, L20, L21, L30, L31, L32, Le10}. These are the off-diagonal Cholesky entries of the subscore's rows; the log-diagonals only shift.
- **Packed items:** m_s = the first-threshold coordinates γ1 of subscore s.

The term vanishes only at an exact optimum. Optimisers stop at small non-zero gradients, so the term is part of the identity, not a tolerance.

| id | check | definition | tolerance / action |
|---|---|---|---|
| N1 | score additivity, per block | max \|Σ_i u_i − U\| ≤ 1e-8 · max(1, ‖U‖), where U is the block's total gradient computed directly | failure: flag F6 |
| N2 | estimating equations near zero | stage-2 ‖∇‖ (0.934 reference) and item-fit gradient norms | reported only. A stage-2 non-convergence is flag F1 |
| N3 | stage-2 relabelling identity | ‖C₂ v_φ + H₂ v_ψ − [c] m_s ⊙ g₄‖ / ‖H₂ v_ψ‖, four directions, with g₄ the stage-2 NLL gradient at ψ̂₂ (from the base evaluation) | ≤ 1e-5. Fresh datasets: flag F6. Regression dataset: INTEGRITY STOP |
| N4 | S1-b item-stage relabelling identity | ‖A₃ v_φ + E₃ v_ψ − [c] m_s ⊙ g₃‖ / ‖A₃ v_φ‖, four directions, with v_φ the tangent of φ̂₂, v_ψ the tangent of ψ̂₁, and g₃ = ∇_φ NLL₃(φ̂₂; ψ̂₁) | ≤ 1e-5. Fresh datasets: flag F6. Regression dataset: INTEGRITY STOP |
| N5 | symmetry of A₁, A₃, H₁, H₂ | max\|X − Xᵀ\| ≤ 1e-4 · max\|X\| | failure: flag F6 |
| N6 | conditioning of A (198 × 198) | 2-norm condition number s_max/s_min (SVD, unscaled) | > 1e12 is flag F6 |
| N7 | V symmetric PSD | min eigenvalue ≥ −1e-10 · max eigenvalue | failure: intervals undefined, flag F8 |
| N8 | FD-step sensitivity (fresh datasets 0–2) | H₂ and C₂ recomputed at h = 2e-4; max over the 19 re-expressed components of \|SE_S(2h)/SE_S(h) − 1\| | > 0.05 is flag F7 |
| N9 | information equality | diag(B₄₄)/diag(H₂) | reported only |
| N10 | G | ‖G v_ψ‖/‖v_ψ‖ ≤ 1e-6 for the four tangents of ψ̂₂, and numerical rank(G) = 15, counting singular values > 1e-8 · s_max | failure: flag F5 |
| N11 | regression of the full-chain SEs (= doc 47 R-c) | raw-ψ̂⁽²⁾ full-chain SEs (√diag V_{ψ₂ψ₂}, **before** S) against D4's `se_psi2.full_chain` on `[42, 45, 1, 0]`: max \|SE/SE_D4 − 1\| | ≤ 1e-6; otherwise INTEGRITY STOP |

**Pre-freeze evidence** (read-only on the saved D4 and Step 6 matrices of the five development datasets; `scripts/58_m2_rev3_prefreeze_checks.py`):
- **N6:** condition numbers 1.7e3–2.4e3 unscaled (about 54 Jacobi-scaled). The 1e12 threshold is retained unchanged; it cannot flag typical datasets.
- **N4:** with the gradient term, 5e-9 to 4.6e-7 in all four directions. Without it, 5e-5 to 5e-4 in the c-directions. **The R1.2 wording, which omitted the gradient term, would have produced a spurious INTEGRITY STOP.** The identity, not the tolerance, was corrected.
- **N3:** without the gradient term already ≤ 6e-7 (D4). The term is included for exactness.

**No tolerance was changed.**

### 3.6 When the method is invalid or unreliable

- **It does not correct bias.** It estimates variability around Θ*, not around the truth. If Θ* ≠ truth, which is the question the recovery test asks, coverage fails whatever the SE.
- **Non-smooth objective at the optimum.** Non-ND patient modes, P1/P2 penalties or AGQ fallbacks near Θ̂ (the AGQ exact gradient is inexact for non-ND patients) can make the FD blocks unreliable. Such datasets are flagged (F4, F6).
- **Optimisation not converged** (U(Θ̂) ≠ 0). The linearisation point is wrong (F1, F2).
- **Parameters near a bound or a variance near 0** (log-diagonal near −8). Wald intervals on the log-Cholesky scale are unreliable (F3).
- **Finite-sample accuracy is unknown; no calibration claim is made.** 198 stacked parameters with 934 independent patients (p/n ≈ 0.21) can make the patient-clustered sandwich anti-conservative. No small-sample correction is applied (D2b, approved). Calibration is judged only by the pre-registered recovery test (SD of z, coverage), **not** by the plausibility of the D4 values.
- **Patients are assumed independent.** This holds by construction in the simulation, and is an assumption for real data (sites, raters).
- **The Laplace / GH approximation.** The estimating functions are gradients of an approximate likelihood. The sandwich remains a valid M-estimation covariance for that estimator (B ≠ A is allowed), but the estimator itself may be biased (Step 3 bounds the per-patient error at ≤ 0.05).

## 4. Role of S1-b [P] (doc 44 §4, unchanged; D1 approved)

- One iteration, kept as a shape correction. The scale is fixed by S, which also neutralises S1-b's unanchored scale.
- S(ψ̂⁽¹⁾), the N(0, 1)-only pipeline, is reported as a non-binding comparator.

## 5. Le00_log [P] (doc 44 §5, unchanged)

- **No exemption.**
- A pre-specified first-order attribution (tremor vs PIGD items; affine vs non-affine) is reported only.
- The interpretation rules are fixed in advance.

## 6. Limitations (doc 44 §6, unchanged; plus §2.6 and §3.6)

- Definitional match with the synthetic truth.
- Design dependence.
- Reinterpretation of the parameters.
- Gaussian-latent dependence of the moments.
- A single truth.
- O(JITTER) residuals.
- **Under the convention, the intercepts are not independently informative (§2.6).**
- The sandwich caveats (§3.6).

## 7. Decisions (approved 2026-10-19; user's Final Decision Record R1.1)

| # | decision | approved resolution |
|---|---|---|
| D1 | S1-b | **retain one S1-b iteration as a shape correction** (§4) |
| D2 | primary interval method | **full-chain patient-clustered sandwich**, with the §3.5 numerical checks. Its calibration is to be demonstrated by the doc 47 test, not assumed |
| D2b | small-sample correction | **none applied post hoc.** Calibration is evaluated through the pre-registered recovery test. A correction chosen after seeing the results would be inadmissible |
| D3 | scale convention | **adopt the Rev 3 convention (§2) for future estimates** |
| D4 | recovery definition | **adopt the doc 47 test as the revised recovery definition.** Historical verdicts are preserved without retrospective re-scoring. The doc 16 D3 rule is reported on fresh datasets 0–4 as a comparator |
| D6 | freezing | **freeze docs 46 and 47 together**, after the final text review and under a **separate explicit freeze authorisation** |

Not among the approved decisions:
- **D5** (calibration-stage parameter draws, doc 16 B5). Remains **undecided and deferred** while calibration is blocked.

The doc 47 decisions (R1, R2′, R2b, R3, R4, R8, R9) are recorded in doc 47 §14.

## 8. Authorisation status (R1.3)

**Authorised** by the user's *Final Execution Directive* (2026-10-19), subject to every gate of doc 47:
- freezing docs 46 and 47 together;
- the read-only pre-freeze checks;
- implementing the recovery driver and tests;
- the labelled smoke run;
- Phase 1;
- the conditional Phase 2.

**Not authorised:** Steps 7 and 8, calibration, positive controls, real-data analysis, S1-c, and any change of a frozen criterion. Nothing is staged, committed or pushed.
