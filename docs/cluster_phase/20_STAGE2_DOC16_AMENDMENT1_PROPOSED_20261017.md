# Doc 16, Amendment 1: proposed (not in force)

**Date proposed:** 2026-10-17.
**Status:** **PROPOSED, for review.** It is not in force, and nothing in it has been implemented or run. If approved, its SHA-256 is recorded in `docs/cluster_phase/_frozen/` before any code change.

**Amends:** `16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.md`.
- Doc 16 is preserved byte for byte and is read-only.
- Its SHA-256 is `c2eace18999f45e05324575e58d6933109c3836be67f832bfee55c8ad6bffbbc`. That matches the frozen record and the `_frozen/…as_frozen_20261017.md` copy, verified 2026-10-17 when this proposal was written.

**Based on:**
- doc 17 (pilot, job 13822);
- doc 18 (fixed-item recovery);
- doc 19 (likelihood-slice diagnostic, job 13824).

| tag | meaning |
|---|---|
| **[O]** | observed in docs 17–19 (run, file) |
| **[H]** | hypothesis, interpretation, or a proposed design whose properties are not yet shown |
| **[M]** | mathematical fact |

---

## 0. What does not change

These items of doc 16 are **unchanged** and remain binding:
- **B11 acceptance criteria and thresholds.** Item fit: 3 starts within 0.1, a ∈ [0.05, 20]. Longitudinal fit: ‖∇‖ < 1e-3 × patients, no bound hits, Hessian positive definite, 3 starts within 0.1. Likelihood accuracy: maximum per-patient error ≤ 0.05 on the same 30 patients (seed `[42, 45, 4]`), against the **same D3 reference** (AGQ7 when 2n ≤ 6, otherwise importance sampling with 20,000 draws and a t₅ proposal). Sampler: split-R̂ < 1.01 and bulk ESS ≥ 100 on the same 30 patients and 4 chains.
- **Recovery definitions.** The a-priori truth (`scripts/45_m2_pilot.py`, `TRUTH_ITEMS` and `TRUTH_LONG`) on the stage-1 scale, the real design, seeds `[42, 45, 1, r]`, 5 datasets, 95 % Wald coverage in ≥ 4 of 5 for every ψ component.
- **Model M2** (B4). Item model, latent structure, 19 longitudinal parameters, no intermediate state, no serial correlation.
- **Fidelity gates G1–G7** (B6, D3) with their tolerances, and the **G4 rule.** If G4 fails, target 2 is uninformative and Stage 2 stops before calibration.
- **Gate ordering.** All B11 checks pass → G1–G7 → (separate review) → calibration. **Calibration, positive controls and evaluation statistics stay blocked** until a separate review after the gates.
- **Everything in B7–B10, B12–B14 and D1–D3** that is not explicitly replaced below.

Amendment 1 changes **only** how the stage-2 marginal likelihood and the patient sampler are *computed*, adds pre-requisite checks, and defines a stage-1 investigation. It does not change the model, any criterion or any evaluation definition.

---

## 1. Record of findings (docs 17–19)

### 1.1 Observed

| # | finding | source |
|---|---|---|
| O1 | M2 implemented; 10 unit tests pass, including the analytic gradient against finite differences | doc 17 §1 |
| O2 | Item fits and the longitudinal fit pass their B11 convergence criteria on real data | doc 17 §3 |
| O3 | **Full-pipeline recovery fails:** 16 of 19 ψ components pass. β0_P 3/5 (mean z +2.35), Le00_log 2/5 (−2.26), Le11_log 0/5 (−3.34) | doc 17 §3, doc 18 §4 |
| O4 | **Laplace accuracy fails:** maximum error 0.137; 20 of 30 patients > 0.05; every error negative; mean −0.091 at n ≥ 8 visits and −0.051 at n ≤ 4 | doc 17 §3–4 |
| O5 | **Sampler fails:** 26 of 30 patients; max R̂ 1.082, min ESS 54, mainly on the b_i components | doc 17 §3 |
| O6 | G1–G7 not run; no label or evaluation statistic computed | doc 17 §5 |
| O7 | **Fixed-item recovery** (same 5 datasets, items at truth): 18 of 19 pass. β0_P 5/5 (mean z 0.05); Le00_log 4/5 (−1.03); **Le11_log 3/5 (−1.78), z negative in 5 of 5** | doc 18 §4 |
| O8 | **Likelihood slice, dataset 0, items fixed:** σ_P argmax 0.3796 (Laplace) vs 0.3973 / 0.3975 (D3 reference, two seed sets); truth 0.4053. The difference is 1.3 SE. The Laplace − reference gap falls by about 1.0 log-unit per 0.01 of σ_P, and is −53 to −61 overall | doc 19 §3 |

### 1.2 Hypotheses

| # | hypothesis | status |
|---|---|---|
| H1 | The 2n-dimensional Laplace approximation under-estimates each patient's marginal likelihood by more when Σ_ε is larger. This biases the visit-level SDs downward | **supported** by O4, O7 and O8 (one dataset, conditional slice, not a profile); not proven |
| H2 | Stage 1 (item parameters under the θ ~ N(0, 1) working assumption) causes the β0_P bias and part of the SD bias | **supported** by the O3 vs O7 contrast; mechanism not isolated |
| H3 | Integrating ε per visit and applying Laplace (or quadrature) only to the 4-dimensional b_i removes most of the H1 error | **untested** |
| H4 | Sampler failure reflects slow mixing of elliptical slice sampling in 2n dimensions, not a model defect | plausible, untested |

### 1.3 Corrected truth and the doc 18 erratum

The true PIGD visit-level SD on the stage-1 scale is σ_P = √(Le10² + exp(2·Le11)) = **0.4053**.
- Doc 18 §4 reported "PIGD visit SD truth 0.4033" and relative biases (−11.2 % pilot, −6.2 % fixed items). Those figures are for **exp(Le11) = 0.4033, the Cholesky diagonal**, not σ_P. The erratum was first recorded in doc 19 §1.
- Doc 18's coverage counts and z values are on **Le11_log**, as the frozen recovery criterion is, and are unaffected.
- Similarly, the tremor "truth 0.4783" in doc 18 is exp(Le00). The tremor visit SD equals exp(Le00) exactly, since Le00 is the first Cholesky diagonal, so 0.4783 is correct as σ_T.
- From here on, reports quote σ_T and σ_P, and the criterion stays on ψ.

---

## 2. Proposed stage-2 likelihood: per-visit quadrature over ε, then integration over b

### 2.1 Structure [M]

Given b_i, the visits are conditionally independent, and visit j depends on ε_ij only:

  p(Y_i | b_i) = Π_j L_ij(b_i),
  L_ij(b) = ∫ P(Y_ij^T | θ^T = μ_ij^T + z_ij^T b + ε^T) · P(Y_ij^P | θ^P = μ_ij^P + z_ij^P b + ε^P) · N₂(ε; 0, Σ_ε) dε,

and the patient's marginal likelihood is p(Y_i) = ∫ p(Y_i | b) N₄(b; 0, D) db.

This is the **same model and the same marginal likelihood** as doc 16 B4/B5. Only the numerical route differs.

- Each GRM factor is log-concave in θ (doc 16 B5). Products and Gaussian convolutions of log-concave functions are log-concave (Prékopa).
- So **log L_ij(b) is concave in b**, and log p(Y_i | b) + log N₄(b; 0, D) has a unique mode.

### 2.2 Per-visit Gauss–Hermite over ε

- ε = C_ε u, with C_ε the Cholesky factor of Σ_ε and u ~ N₂(0, I).
- L_ij(b) ≈ Σ_{q,r} w_q w_r · P(Y^T | μ^T + z^T b + (C_ε u_qr)_T) · P(Y^P | μ^P + z^P b + (C_ε u_qr)_P), using a Q × Q tensor product of probabilists' Gauss–Hermite nodes (`grm.gh_nodes`).
- Computed in log space (log-sum-exp).
- **Q is fixed by the order check in §4,** default Q = 10.
- Non-adaptive per-visit quadrature is proposed because Σ_ε is small relative to the item information at a single visit [H]. §4 tests this.

### 2.3 Integration over b (4-dimensional)

**Primary: Laplace over b only.**
- Find the mode b̂_i of log p(Y_i | b) − ½ bᵀD⁻¹b by damped Newton. The gradient and Hessian of log L_ij(b) come from quadrature-weighted GRM θ-derivatives, through the chain rule and z_ij.
- Then log p(Y_i) ≈ log p(Y_i | b̂) − ½ b̂ᵀD⁻¹b̂ − ½ log|D| − ½ log|D⁻¹ + H_i|.
- The Laplace step now acts in 4 dimensions, on a posterior informed by every visit [H: more accurate than the 2n-dimensional version; to be shown in §3].

**Fallback** (pre-declared; used only if the primary fails the §3 accuracy check): adaptive Gauss–Hermite over b, centred and scaled at the Laplace mode and curvature, with G nodes per dimension (G = 3, then 5; G⁴ = 81 or 625 points).

**Gradient.** The exact gradient is required (B11 ‖∇‖ criterion).
- For the Laplace primary: envelope-theorem terms plus the implicit mode term, which needs third derivatives of log L_ij in b. These come from the same quadrature, as in `m2_grm.group_terms`.
- For the fallback: fixed-node differentiation, with no implicit term.
- Unit tests compare each with finite differences, as `test_gradient_matches_finite_differences` does now.

### 2.4 Patient sampler (replaces doc 16 B8/D3 sampler details; thresholds unchanged)

**Target.** Replicates use b_i plus *fresh* ε* and items (doc 16 B8), so the sampler needs **only p(b_i | Y_i)**, which is 4-dimensional, with ε integrated by §2.2.

**Sampler.** Elliptical slice sampling on the prior N(0, D) with the likelihood p(Y_i | b) from §2.2, **preconditioned**: run on the whitened variable b = b̂ + L_H v, with L_H the Cholesky factor of the Laplace posterior covariance and v ~ N(0, I) as the reference.
- Implemented as elliptical slice sampling with a Gaussian approximation N(b̂, H⁻¹) and the ratio p(Y | b) N(b; 0, D) / N(b; b̂, H⁻¹) as the "likelihood", which is exact for the target [M].
- 4 chains; warm-up 200; 1,000 kept iterations; overdispersed starts b̂ + N(0, 4 H⁻¹), as in D3.

**Monitored quantities** (adapted because θ is no longer sampled; label-free):
- the four b_i components;
- Σ_j E[T_ij | b_i] and Σ_j E[P_ij | b_i], the expected sums integrating ε by the §2.2 quadrature.

**Thresholds unchanged:** R̂ < 1.01, ESS ≥ 100, all 30 patients.

---

## 3. Numerical validation against the existing D3 reference (pre-requisite)

All before any fit to real data:

| # | check | criterion |
|---|---|---|
| V1 | Unit tests: §2.2 quadrature equals the closed form in a Gaussian limit; with no items observed the likelihood equals 1; analytic gradient against finite differences (relative 1e-4) | pass |
| V2 | **Same 30 patients** (seed `[42, 45, 4]`), plug-in ψ̂ of the pilot (`step3_longitudinal.json`), items from the pilot: the new log p(Y_i) against the **unchanged D3 reference** (`m2_grm.agq_one` / `m2_grm.is_one`, which integrate the full 2n-vector θ and so the same model) | **max \|error\| ≤ 0.05**, the unchanged B11 tolerance |
| V3 | Synthetic dataset 0, items fixed at truth: repeat the doc 19 slice (same 7 σ_P points, same nuisance treatment) with the new likelihood | its σ_P argmax within 0.0134 (1 Wald SE) of the reference argmax 0.3973. Diagnostic, reported; it supports but does not replace V2 |
| V4 | Full-ψ check: at ψ̂ and at 3 perturbed ψ (seed `[42, 46, 7, k]`), the new total log-likelihood minus the D3 reference total | reported, together with its trend in σ_T and σ_P (Monte Carlo SE from the reference) |

If V2 fails with the Laplace-over-b primary, the fallback (§2.3) is tested next, under the same V2 criterion. If both fail, the work stops and is reported. No further change is made without a new amendment.

---

## 4. Quadrature-order sensitivity and cost estimate (before any full pilot)

### 4.1 Order checks (on the V2 patients, at the pilot ψ̂)

- **ε quadrature.** Q ∈ {7, 10, 15}. Choose the smallest Q with max |log p_Q − log p_15| ≤ 0.005 per patient. Report the full table.
- **b integration** (fallback only). G ∈ {3, 5, 7}, same rule against G = 7.
- **Laplace over b against adaptive Gauss–Hermite over b** at G = 5: reported. It tells whether the primary's residual error is small relative to 0.05.

### 4.2 Cost (estimate; a timed benchmark is required before any pilot)

**Observed baselines** [O]:
- the current 2n-dimensional Laplace fit, 934 patients: 47–62 s per fit (pilot);
- recovery fits: 62–130 s (doc 18);
- D3 reference: about 0.48 s per patient (doc 19).

**Operation count** [H].
- One likelihood evaluation costs about Q² × (visits) × 16 items × 5 categories per Newton step per patient. At Q = 10 and 4,295 visits that is about 3.4 × 10⁷ item-category evaluations per Newton sweep. Newton needs about 5–10 sweeps, and the third-derivative gradient terms add about 2×.
- Estimated **1–5 s per likelihood-plus-gradient evaluation**, and about **5–40 min per fit** at 300–500 L-BFGS iterations. That is 5–40× the current fit.
- Full pilot: 3 real starts + 5 recovery datasets (×2 for fixed-item and full pipeline) + Hessians. About **2–10 CPU-hours**, under 1 h on one node if the datasets and starts run in parallel.
- **B1 benchmark before any pilot.** Time one likelihood-plus-gradient evaluation and one full fit on synthetic dataset 0 at Q = 10. If a fit projects above 60 min, report before proceeding. Projected calibration cost (doc 16 B9: 500 nulls plus controls, 5 folds each) is reported at that point, because it may force the doc 16 B9 fallback (N₀ = 200) or a separate amendment.

---

## 5. Stage-1 investigation (separate track, cheapest option first)

**Motivation:** O3 vs O7 and H2. Stage 1 explains the β0_P bias and part of the SD bias. A stage-2 repair alone would not let the full pipeline pass.

| step | what | cost | decision rule |
|---|---|---|---|
| S1-a (diagnostic) | On the 5 recovery datasets, fit the items by marginal maximum likelihood under the **true** population latent distribution: the visit-level mixture of N(M_ij, K_ij,jj) at the true ψ, by quadrature per visit. Compare the item errors and the downstream fixed-ψ quantities with the N(0, 1) stage-1 fit | minutes | Quantifies the bias due to the N(0, 1) working assumption alone. Reported; no criterion |
| **S1-b (repair candidate, cheaper)** | **One refit iteration.** (1) N(0, 1) stage 1 as now; (2) stage 2 with the §2 likelihood; (3) re-estimate items by marginal maximum likelihood under the **fitted** population latent mixture from (2); (4) re-fit stage 2 with the refitted items. Pre-declared: exactly one iteration, no iteration until convergence | about 2× the stage-2 cost | Accepted only if the **full-pipeline recovery** (unchanged criterion) passes |
| S1-c (only if S1-b fails) | Joint marginal maximum likelihood of item and longitudinal parameters under the §2 likelihood (55 + 19 parameters) | much higher; needs its own cost estimate and amendment | not part of this amendment; would be proposed separately |

Re-fitting the items under the fitted latent distribution changes how the parameters are estimated, not the model (doc 16 B4/B5 already fix the scale by stage 1). It is declared here so that the choice cannot depend on any evaluation statistic. S1-a and S1-b use only synthetic data and real item responses. Neither uses labels.

---

## 6. Required order before G1–G7 (gate ordering otherwise unchanged)

1. Implement §2 and §5 S1-b with unit tests (V1).
2. **V2 likelihood accuracy** on the 30 patients against D3 (≤ 0.05), plus the §4.1 order checks. Then V3/V4, which are reported.
3. **B1 benchmark** (§4.2). Report if a fit projects above 60 min.
4. **Fixed-item recovery** (doc 18 design, §2 likelihood): every ψ component ≥ 4 of 5.
5. S1-a diagnostic (reported).
6. **Full-pipeline recovery** (S1-b pipeline): every ψ component ≥ 4 of 5.
7. **Real-data B11:** item fits, longitudinal fit (3 starts), likelihood accuracy (= V2 at the new ψ̂), **sampler** (§2.4), and DIF (reported).
8. **Only if 2, 4, 6 and 7 all pass:** label-free fidelity gates G1–G7, with the G4 rule unchanged.
9. **Stop for review.** Calibration, positive controls and evaluation statistics stay blocked until a separate review of step 8.

At every step, a failure stops the sequence. It is reported, and nothing after it runs. Any further change needs a new dated amendment.

---

## 7. Isolation and records

- New code goes in new files only (proposed `src/ppmi_subtype/m2_quad.py`, `tests/test_m2_quad.py`, `scripts/48_m2_amendment1_checks.py`). The existing `m2_grm.py` stays as it is; its D3 reference functions remain the validation standard.
- Every run gets a new `experiments/<UTC>_stage2_m2a1_*` directory with provenance. Each run refuses to start unless the doc 16 and Amendment 1 hashes match their frozen records.
- Pilot and diagnostic outputs (docs 17–19 runs) are read only.

## 8. Limitations of this proposal

- **H3 is untested.** Integrating ε per visit may leave a smaller but still material error from the Laplace step in b. The fallback and the V2 criterion handle that case.
- **Per-visit quadrature without adaptation** may be inaccurate for visits with very informative item patterns (many extreme categories). The §4.1 order check is the guard.
- **S1-b is one heuristic refit iteration,** not a full joint estimator. If it fails, S1-c is costly.
- **All evidence for H1 is synthetic.** Its relevance to the real data is inferred from the matching pattern (O4 vs O8).

---

## 9. Decisions requiring approval

1. **Adopt §2:** per-visit Q × Q Gauss–Hermite over ε (default Q = 10, fixed by §4.1), with **Laplace over the 4-dimensional b** as primary and adaptive Gauss–Hermite over b (G = 3, then 5) as the pre-declared fallback.
2. **Adopt V2** (same 30 patients, same D3 reference, unchanged 0.05 tolerance) as the accuracy criterion. V3/V4 are reported, not gated.
3. **The order-check rule:** tolerance 0.005 against Q = 15 / G = 7.
4. **The B1 benchmark** and the 60-minute-per-fit reporting threshold before any pilot.
5. **The sampler change:** sample only p(b_i | Y_i), with preconditioned elliptical slice sampling. Monitored quantities adapted (expected sums given b). Thresholds unchanged.
6. **Make fixed-item recovery a required pre-requisite check,** with the same ≥ 4-of-5 rule, in addition to the unchanged full-pipeline recovery.
7. **The stage-1 track:** run S1-a as a diagnostic, and **S1-b (one refit iteration)** as the only stage-1 repair in this amendment. S1-c would need a separate amendment.
8. **The §6 order,** with G1–G7 only after steps 2, 4, 6 and 7 all pass, and calibration blocked pending separate review.
9. **Reporting convention:** quote σ_T and σ_P on the natural scale (truth 0.4783 and 0.4053) alongside the ψ-scale criterion.
