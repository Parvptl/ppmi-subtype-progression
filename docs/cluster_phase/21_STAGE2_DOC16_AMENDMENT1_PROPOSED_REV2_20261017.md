# Doc 16, Amendment 1: proposed, Revision 2 (not in force)

**Date:** 2026-10-17.
**Status:** **PROPOSED, for review.** It is not in force; no code has been written and nothing has been run. If approved, its SHA-256 is recorded in `docs/cluster_phase/_frozen/` before any code change.

**Supersedes:** Revision 1, `20_STAGE2_DOC16_AMENDMENT1_PROPOSED_20261017.md` (SHA-256 `c02099415a5bfcae752b968ee9b2faaeb37391e996ff27459999ac21826c22bd`). Revision 1 is kept unchanged for the record. This revision is complete on its own.

**Amends:** `16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.md`.
- Doc 16 is preserved byte for byte and is read-only.
- Its SHA-256 is `c2eace18999f45e05324575e58d6933109c3836be67f832bfee55c8ad6bffbbc`. It matches the frozen record and is identical to `_frozen/…as_frozen_20261017.md`. Verified when this revision was written.

**What changed from Revision 1** (the reviewer's clarifications, 2026-10-17):

| reviewer point | where addressed |
|---|---|
| R1. Quadrature accuracy: a well-defined reference and a consistent 0.005 tolerance | §4.1 (rewritten) |
| R2. Benchmark the actual likelihood and gradient before committing to the design; the operation count does not establish runtime | §4.2 (rewritten); §6 steps 1–2 |
| R3. Do the adapted sampler quantities still support R̂ < 1.01 and ESS ≥ 100? | §2.4 (extended), new check V5 |
| R4. S1-a diagnostic only; the unchanged full-pipeline recovery is required after S1-b; fixed-item recovery alone never authorises the gates | §5, §6 |
| Caution: check the log-concavity argument against the actual parameterisation and integration | §2.1 (rewritten), §2.5 (new) |

**Tags:** [O] observed in docs 17–19 · [H] hypothesis or unproven design property · [M] mathematical statement, with its scope stated.

---

## 0. What does not change (binding)

- **Every B11 acceptance criterion and threshold** (doc 16 B11, D3):
  - item fit: 3 starts within 0.1, a ∈ [0.05, 20];
  - longitudinal fit: ‖∇‖ < 1e-3 × patients, no bound hits, positive-definite Hessian, 3 starts within 0.1;
  - likelihood accuracy: maximum per-patient error ≤ 0.05 on the same 30 patients (seed `[42, 45, 4]`), against the same D3 reference (AGQ7 when 2n ≤ 6; otherwise importance sampling, 20,000 draws, t₅);
  - sampler: split-R̂ < 1.01 and bulk ESS ≥ 100, 30 patients, 4 chains.
- **Recovery definitions.** The a-priori truth (`scripts/45_m2_pilot.py`) on the stage-1 scale, the real design, seeds `[42, 45, 1, r]`, 5 datasets, 95 % Wald coverage in ≥ 4 of 5 for every ψ component.
- **Model M2** (B4), the 19 longitudinal parameters, no intermediate state, no serial correlation.
- **Fidelity gates G1–G7 and the G4 rule** (B6, D3).
- **Gate ordering:** all B11 checks → G1–G7 → stop for review. **Calibration, positive controls and evaluation statistics stay blocked** until a separate review.
- **Everything else in doc 16** not explicitly replaced here.

The amendment changes only *how* the stage-2 likelihood and the patient sampler are computed. It adds checks (it never relaxes one) and defines a stage-1 track.

---

## 1. Record of findings

### 1.1 Observed (docs 17–19)

| # | finding |
|---|---|
| O1 | M2 implemented; 10 unit tests pass, including the gradient against finite differences (doc 17) |
| O2 | Real-data item fits and longitudinal fit pass their convergence criteria (doc 17) |
| O3 | Full-pipeline recovery fails: 16/19 pass. β0_P 3/5 (mean z +2.35), Le00_log 2/5 (−2.26), Le11_log 0/5 (−3.34) (doc 17, doc 18) |
| O4 | Laplace accuracy fails: maximum error 0.137; 20 of 30 > 0.05; all negative; mean −0.091 at n ≥ 8 vs −0.051 at n ≤ 4 (doc 17) |
| O5 | Sampler fails: 26 of 30 patients; max R̂ 1.082, min ESS 54, mainly on b_i (doc 17) |
| O6 | G1–G7 not run; no label or evaluation statistic computed (doc 17) |
| O7 | Fixed-item recovery: 18/19 pass. β0_P 5/5; Le00_log 4/5; **Le11_log 3/5, z negative in 5 of 5** (doc 18) |
| O8 | Slice, dataset 0, items fixed: σ_P argmax 0.3796 (Laplace) vs 0.3973 / 0.3975 (D3 reference, two seed sets); truth 0.4053; difference 1.3 SE. The gap falls by about 1.0 log-unit per 0.01 of σ_P (doc 19) |

### 1.2 Hypotheses

| # | hypothesis | status |
|---|---|---|
| H1 | The 2n-dimensional Laplace error grows with Σ_ε and biases the visit SDs downward | supported (O4, O7, O8; one dataset, conditional slice); not proven |
| H2 | Stage 1 (the θ ~ N(0, 1) working assumption) causes the β0_P bias and part of the SD bias | supported (O3 vs O7); mechanism not isolated |
| H3 | Per-visit quadrature over ε plus Laplace over the 4-dimensional b removes most of the H1 error | untested |
| H4 | The sampler failure is slow mixing in 2n dimensions, not a model defect | untested |

### 1.3 Corrected truth and the doc 18 erratum

- σ_P = √(Le10² + exp(2·Le11)) = **0.4053** is the true PIGD visit-level SD on the stage-1 scale.
- Doc 18's "0.4033" and its relative biases (−11.2 % pilot, −6.2 % fixed items) refer to exp(Le11), the Cholesky diagonal. This was first recorded in doc 19 §1.
- Doc 18's coverage counts and z values are on Le11_log, the criterion's scale, and are unaffected.
- σ_T = exp(Le00) = 0.4783 is correct as stated.
- Reports quote σ_T and σ_P on the natural scale, beside the ψ-scale criterion.

---

## 2. Proposed stage-2 likelihood

### 2.1 Structure, and where log-concavity does and does not hold

**Likelihood (unchanged model).** Given b_i, visits are conditionally independent, and visit j depends on ε_ij only:

  L_ij(b) = ∫ g_ij(b, u) φ₂(u) du,  g_ij(b, u) = P(Y_ij^T | θ^T = μ_ij^T + z_ij^T b + (C_ε u)_T) · P(Y_ij^P | θ^P = μ_ij^P + z_ij^P b + (C_ε u)_P),

  p(Y_i) = ∫ Π_j L_ij(b) φ₄(b; 0, D) db,

with C_ε the Cholesky factor of Σ_ε, u ~ N₂(0, I), and φ the normal density. This is exactly doc 16 B4/B5.

**What has been checked against the actual parameterisation and code:**

1. **Each item's category probability is log-concave in θ.** [M, conditional on the parameterisation]
   - `grm.unpack_items` guarantees a_k = exp(α_k) > 0 and τ_k1 < … < τ_k4 (cumulative sums of exponentials).
   - P(Y = c | θ) = F(a(θ − τ_c)) − F(a(θ − τ_{c+1})), with F the logistic CDF and the conventions F(+∞) = 1, F(−∞) = 0. This is the probability that a logistic variable, whose density is log-concave, falls in an interval that shifts with θ; such probabilities are log-concave in the shift (Prékopa).
   - Missing items contribute a factor of 1.
   - Doc 16 B5 cites Pratt (1981) for the same property.
2. **g_ij is log-concave jointly in (b, u).** [M] It is a product of log-concave functions of θ^T and θ^P, composed with an affine map of (b, u). This holds for every C_ε and D, and is unaffected by the treatment indicator or the time design, which enter only the affine offset.
3. **The exact L_ij(b) is log-concave in b.** [M] Integrating a jointly log-concave function against the log-concave φ₂ over u preserves log-concavity (Prékopa–Leindler). So the exact log p(Y_i | b) + log φ₄(b; 0, D) is strictly concave, and the exact per-patient mode is unique.

**What does not follow (not relied on):**

4. **The quadrature approximation is not guaranteed log-concave.** L̃_ij(b) = Σ_q w_q g_ij(b, u_q) is a positive sum of log-concave functions, and such sums are **not** log-concave in general. With few nodes relative to the curvature of g, the approximate objective could be non-concave or multimodal in b.
5. **The outer objective in ψ is not concave.** Concavity concerns the latent b given ψ. Nothing here makes the marginal likelihood concave in ψ; the existing multi-start and Hessian checks (B11) remain the safeguard.
6. **The Laplace step in b** uses the Hessian of the approximate objective. Where (4) fails locally, that Hessian may not be negative definite.

The safeguards for 4–6 are in §2.5. **Log-concavity is used only as motivation, never as a guarantee for the numerical optimisation.**

### 2.2 Per-visit Gauss–Hermite over ε

- L̃_ij(b) = Σ_{q,r} w_q w_r g_ij(b, C_ε (x_q, x_r)ᵀ), a Q × Q tensor product of probabilists' Gauss–Hermite nodes (`grm.gh_nodes`), computed in log space.
- **Q is chosen by §4.1.** No default is binding.

### 2.3 Integration over b (4-dimensional)

**Primary: Laplace over b.**
- Mode by damped Newton with line search on log L̃(b) − ½ bᵀD⁻¹b, using quadrature-weighted derivatives through the chain rule.
- log p(Y_i) ≈ log Π_j L̃_ij(b̂) − ½ b̂ᵀD⁻¹b̂ − ½ log|D| − ½ log|D⁻¹ + H_i|.

**Pre-declared fallback** (only if the primary fails V2): adaptive Gauss–Hermite over b, centred and scaled at the mode and curvature, G nodes per dimension (§4.1).

**Exact gradient** in ψ:
- Laplace: envelope terms plus the implicit mode term, which needs third b-derivatives of log L̃.
- Fallback: fixed adaptive nodes, without an implicit term.
- Both are unit-tested against finite differences.

### 2.4 Patient sampler and its monitored quantities (R3)

**Target.** p(b_i | Y_i), 4-dimensional, with ε integrated by §2.2. Replicates need b_i plus fresh ε* and items (doc 16 B8), so nothing else about the patient is sampled.

**Sampler.** Elliptical slice sampling with the Gaussian factor N(b̂, H⁻¹) and the "likelihood" ratio p̃(Y | b) φ₄(b; 0, D) / N(b; b̂, H⁻¹).
- This factorisation of the target is exact [M]. Its efficiency depends on how well N(b̂, H⁻¹) covers the posterior [H].
- If H at b̂ is not negative definite (§2.5), the patient falls back to the unpreconditioned prior factor φ₄(b; 0, D), and this is reported.
- Settings as in D3: 4 chains, warm-up 200, 1,000 kept, starts b̂ + N(0, 4 H⁻¹).

**Monitored quantities and why they still support the criterion.**
- **The four b_i components.** The target is exactly 4-dimensional, so these coordinates span it. Non-convergence in any direction of b shows up in at least one coordinate's split-R̂ or in a linear combination; the next item guards the latter.
- **The log target, log p̃(Y_i | b) + log φ₄(b; 0, D).** A scalar summary sensitive to mixing between high- and low-density regions; standard practice.
- **Σ_j E[T_ij | b] and Σ_j E[P_ij | b]** (ε integrated by quadrature). These are the quantities replicates depend on, kept from D3.
- **Criterion unchanged:** split-R̂ < 1.01 and bulk ESS ≥ 100 for **every** monitored quantity of every one of the 30 patients. Adding quantities can only make the criterion stricter. Tail ESS is reported, not gated.
- **Why this is adequate.** Doc 16 B11 monitored "T and P sums and labels". D3 replaced labels with label-free quantities to keep the pilot label-free. Under this amendment, replicate labels are deterministic functions of (b_i, fresh ε*, fresh items), and only b_i is sampled. Convergence of p(b_i | Y_i) in all four coordinates is therefore sufficient for the replicate distribution to converge [M: functions of a converged chain inherit its stationarity; finite-sample R̂ and ESS of other functions can differ, which is why several quantities are monitored].

**New correctness check V5** (R̂ and ESS show mixing, not that the target is right):
- On the same 30 patients, compare the sampler's posterior mean and SD of each b component with an adaptive Gauss–Hermite reference over b (G = 7 per dimension) on the same L̃.
- **Criterion:** |sampler mean − reference mean| ≤ 4 Monte Carlo SE, and the sampler SD within 10 % of the reference, for all 4 components and all 30 patients.
- This is an added check; it relaxes nothing.

### 2.5 Numerical safeguards for the approximate objective (caution item)

For every patient-mode computation in V2, the recovery checks and the real-data B11 run:
1. Damped Newton with backtracking. If the Hessian of the approximate objective is not negative definite at an iterate, fall back to a gradient-ascent step with line search.
2. At the returned mode, record whether the Hessian is negative definite. **Report the count of patients where it is not.** For those patients, the Laplace term is not defined; they are evaluated by the §2.3 fallback, and the count is reported.
3. **Mode uniqueness on V2's 30 patients:** 3 starts (0, the prior-mean-implied b, and b̂ + 2 SD perturbations) must reach the same mode within 1e-6. Any disagreement is reported and stops the sequence at §6 step 3.

---

## 3. Numerical validation (pre-requisites)

| # | check | criterion |
|---|---|---|
| V1 | Unit tests: quadrature equals the closed form in a Gaussian limit; no items observed gives likelihood 1; gradient against finite differences (relative 1e-4); log-concavity spot check: Hessian of the exact-limit objective negative definite on random cases | pass |
| V2 | Same 30 patients, pilot ψ̂ and items: the new log p(Y_i) against the **unchanged D3 reference** | **max \|error\| ≤ 0.05** (B11, unchanged) |
| V3 | Synthetic dataset 0, items at truth: repeat the doc 19 slice with the new likelihood | argmax within 0.0134 of the reference argmax 0.3973. Reported, not gated |
| V4 | At ψ̂ and 3 perturbed ψ (seed `[42, 46, 7, k]`): new total minus D3 reference total, and the trend in σ_T and σ_P | reported |
| V5 | Sampler correctness against AGQ over b (§2.4) | ≤ 4 MCSE for means; SD within 10 % |

---

## 4. Quadrature order and benchmark

### 4.1 Order checks: reference and quantities defined (R1)

Two different errors are separated, and each has its own reference:

| error | reference | quantity compared | tolerance |
|---|---|---|---|
| **(a) ε-quadrature error** | the same method at **Q_ref = 20** | **per-patient log p(Y_i)** at fixed ψ and fixed b-integration (Laplace over b), on V2's 30 patients, at 4 ψ values (pilot ψ̂ and the 3 V4 perturbations) | **max over patients and ψ of \|log p_Q − log p_20\| ≤ 0.005**. Q is the smallest of {7, 10, 15} that passes. If none passes, stop and report |
| **(a′) adequacy of Q_ref** | Q = 25 | the same quantity, on 5 of the 30 patients with the most visits | \|log p_20 − log p_25\| ≤ 0.001, reported. If it fails, Q_ref becomes 25 and (a) is repeated |
| **(b) b-integration error** (fallback only) | adaptive GH with **G_ref = 7** | the same per-patient log p(Y_i) at the chosen Q | max ≤ 0.005; G is the smallest of {3, 5} that passes |
| **(c) total error against the model's integral** | the **D3 reference** | per-patient log p(Y_i) | ≤ 0.05 (V2; B11, unchanged) |

**How the tolerances fit together:**
- 0.005 is applied **only** to the per-patient log marginal likelihood at fixed ψ, the quantity summed into the objective. It bounds the numerical-order error at a tenth of the B11 total tolerance.
- It is **not** applied to parameters.

**Parameter-level effect (reported, not gated):**
- the change in the dataset-0 V3 slice argmax between the chosen Q and Q_ref;
- the change in total-log-likelihood *differences* between the 4 ψ values, which is what moves an MLE.

### 4.2 Benchmark before committing to the design (R2)

The operation count in Revision 1 (5–40 min per fit) is an **unverified [H] guide only**. The design is **not committed** until benchmark B0 has been reported and reviewed.

**B0 measurements:**
- A prototype of §2.2–2.3 (likelihood plus exact gradient), on synthetic dataset 0 (934 patients, real design).
- Run on a compute node, single-threaded BLAS, warm start; median of 3 timings each.
- Timed:
  1. one likelihood evaluation, and one likelihood-plus-gradient evaluation, at Q = 7, 10, 15 and 20;
  2. the same for the fallback at G = 3 and 5 (Q = 10);
  3. one complete fit at the Q chosen by §4.1(a), recording iterations and wall time.
- For comparison: the current Laplace likelihood-plus-gradient and fit on the same data. The current fit's 47–62 s is known [O].

**B0 decision rule** (reported for review either way):
- **Proceed** to recovery only if one fit at the chosen Q is ≤ 60 min and the projected full-pilot cost (3 real starts, 10 recovery fits, Hessians) is ≤ 24 node-hours.
- Calibration's projected cost (doc 16 B9) is reported at the same time.
- **Otherwise stop and report.** No design change is made without a new revision.

---

## 5. Stage-1 track (R4)

| step | role | content | rule |
|---|---|---|---|
| **S1-a** | **diagnostic only** | On the 5 recovery datasets, fit the items under the **true** population latent distribution and compare with the N(0, 1) stage-1 fit | **Reported only. It authorises nothing, selects nothing, and no parameter or setting is chosen from it.** |
| **S1-b** | the only stage-1 repair in this amendment | Exactly one refit iteration: N(0, 1) stage 1 → stage 2 (§2) → items re-estimated under the fitted population latent mixture → stage 2 re-fit | Accepted **only if** the **unchanged full-pipeline recovery** (doc 16 D3: 5 datasets, every ψ ≥ 4 of 5) passes with this pipeline |
| S1-c | not in this amendment | Joint estimation | Separate amendment, if S1-b fails |

**Fixed-item recovery passing on its own never authorises the fidelity gates, or any later step.** It is a diagnostic pre-requisite: if it fails, stop. Full-pipeline recovery is the binding recovery criterion.

---

## 6. Required order (gate ordering otherwise unchanged)

Each step's failure stops the sequence; it is reported, and nothing after it runs.

1. Approval of this revision; freeze its SHA-256.
2. **Prototype plus B0 benchmark** (§4.2), with V1 unit tests. **Report and stop for review.**
3. On approval: V2 (≤ 0.05), §4.1 order checks, §2.5 mode checks; V3/V4 reported.
4. **Fixed-item recovery** with the §2 likelihood: every ψ ≥ 4 of 5. Pre-requisite only.
5. S1-a diagnostic (reported only).
6. **Full-pipeline recovery** with S1-b: every ψ ≥ 4 of 5, **unchanged criterion.**
7. **Real-data B11:** item fits, longitudinal fit (3 starts), likelihood accuracy (V2 at the new ψ̂), **sampler** (R̂ < 1.01, ESS ≥ 100, all monitored quantities) **and V5**, DIF (reported).
8. **Only if steps 3, 4, 6 and 7 all pass:** G1–G7 (label-free), with the G4 rule unchanged.
9. **Stop for review.** Calibration, positive controls and evaluation statistics stay blocked.

## 7. Isolation and records

- New files only: proposed `src/ppmi_subtype/m2_quad.py`, `tests/test_m2_quad.py`, `scripts/48_m2a1_benchmark.py`, `scripts/49_m2a1_checks.py`. `m2_grm.py` stays as it is; its D3 functions are the validation standard.
- Every run gets a new `experiments/<UTC>_stage2_m2a1_*` directory with provenance. It refuses to run unless the doc 16 and frozen Amendment 1 hashes match.
- Docs 17–19 run outputs are read only.

## 8. Limitations

- H3 is untested.
- Non-adaptive per-visit quadrature may be inaccurate for very informative visits (guards: §4.1, §2.5).
- The approximate objective's concavity is not guaranteed (§2.1 points 4–6).
- S1-b is one heuristic iteration.
- The evidence for H1 is synthetic and comes from a single slice (O8).
- V5's AGQ reference over b uses the same per-visit L̃, so it checks the sampler, not the quadrature.

---

## 9. Review of this revision (prepared before any code or compute is authorised)

**9.1 Each clarification:**
- **R1.** The references are now explicit (Q_ref = 20, checked against 25; G_ref = 7; the D3 reference for total error). The 0.005 tolerance applies to one quantity, per-patient log p(Y_i) at fixed ψ, consistently for Q and G. Parameter-level effects are reported separately and never gated by 0.005. **Addressed.**
- **R2.** The design is committed only after a timed prototype benchmark (B0) on a compute node, reviewed at step 2. The operation count is labelled [H]. **Addressed.**
- **R3.** The monitored quantities span the full sampled space (b, 4-D) plus the log target and the replicate-relevant expected sums. The thresholds are unchanged and apply to every quantity. V5 adds a correctness check that R̂ and ESS cannot provide. **Addressed.** Remaining caveat: R̂ and ESS on chosen functions cannot rule out non-convergence in every function. That limitation is inherent and stated.
- **R4.** S1-a is diagnostic only and authorises nothing. The unchanged full-pipeline recovery is required after S1-b. Fixed-item recovery alone cannot authorise the gates. **Addressed.**
- **Caution.** Log-concavity is verified for the exact per-patient integrand under the code's parameterisation (a > 0, ordered τ, affine latent map, Gaussian ε and b). It is shown **not** to extend to the quadrature approximation or to ψ. Numerical safeguards replace any reliance on it. **Addressed.**

**9.2 Consistency with doc 16:**
- No B11 threshold, recovery definition, gate tolerance, G4 rule or gate order is relaxed.
- New checks (V5, §2.5, the order checks, B0) only add stopping points.
- Calibration stays blocked.
- One substantive interpretive change is declared for approval: the sampler monitors b-functions instead of θ-functions (decision 5).

**9.3 Open risks for the reviewer:**
- **B0 may show the design is too slow** (decision point at step 2).
- **V2 may still fail.** The fallback costs much more.
- **S1-b may not fix the β0_P and SD stage-1 bias.** S1-c would then need a separate amendment, with a materially higher cost.

**9.4 Authorisation sought now:** none for code or compute. If this revision is approved, the first authorisation needed is §6 step 2 only: the prototype, V1 unit tests and the B0 benchmark, stopping for review.

## 10. Decisions requiring approval

1. **Likelihood route** (§2.2–2.3): per-visit Q × Q Gauss–Hermite, Laplace over b as primary, adaptive GH over b as fallback.
2. **Order checks** (§4.1): Q_ref = 20 (checked against 25), G_ref = 7, tolerance 0.005 on per-patient log p(Y_i) at fixed ψ. Parameter effects reported, not gated.
3. **Benchmark gate B0** (§4.2): proceed only if one fit ≤ 60 min and the projected pilot ≤ 24 node-hours; otherwise stop. The design is not committed before B0 is reviewed.
4. **Sampler** (§2.4): target p(b | Y) with preconditioned elliptical slice sampling. Monitored quantities are the 4 b components, the log target and the 2 expected sums. Thresholds unchanged, plus the **V5** correctness check.
5. **Mode safeguards** (§2.5): non-negative-definite modes are counted and fall back to the §2.3 fallback; mode uniqueness is checked on 30 patients.
6. **Recovery:** fixed-item recovery as a pre-requisite only; **unchanged full-pipeline recovery after S1-b as the binding criterion.**
7. **Stage-1 track:** S1-a diagnostic only; S1-b the only repair; S1-c separate.
8. **The §6 order,** with a review stop after B0 (step 2) and after G1–G7 (step 9). Calibration remains blocked.
