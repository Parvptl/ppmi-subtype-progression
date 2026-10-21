# Stage 2 M2: independent recovery test of the Rev 3 pipeline — final plan R1.3 (supersedes the doc 45 proposal)

**Date:** 2026-10-19.

> **FINAL (R1.3) — FROZEN together with doc 46** (locks: `_frozen/46_STAGE2_REV3_AMENDMENT.sha256`, `_frozen/47_STAGE2_REV3_RECOVERY_TEST_PLAN.sha256`; read-only copies alongside). Freezing, driver implementation, the smoke run, Phase 1 and the conditional Phase 2 were authorised by the user's *Final Execution Directive* (2026-10-19), subject to the gates below.
> - Doc 45 (`8e00b4f5…`) is kept unchanged. This revision replaces it for review.
> - **Step 6 remains FAILED. D1 remains NOT REPRODUCED.**
> - Steps 7–8, calibration and real-data work stay blocked.
> - **Revision R1.1 (2026-10-19):** the verdict rule (§8), C0/flag definitions (§§5, 7), forecast wording (§4), execution arrangement (§9) and control-arm role (§10) were corrected after a final review. The changes are listed in doc 48. The pre-edit SHA-256 of this file was `0bcbcd2c…`.
> - **Revision R1.2 (2026-10-19):** the approved decisions (R1, R2′, R2b, R3, R4, R8, R9) are recorded as approved, and the requirements of doc 45 omitted in R1 are restored: §2 frozen-settings table; §3 regeneration method; §7 check lists; §8 old-dataset regression checks; §9 budget table; §10 truth-swap procedure; §11 execution sequence and authorisations; §12 outputs; §13 pass/fail statement. Changes are listed in doc 49. The pre-edit SHA-256 of this file was `d1e6fe3f…`.
> - **Revision R1.3 (2026-10-19):** review findings B1–B7, S1–S7 and M1–M4 resolved in §§2, 5–15. The new sections are §15 (disposition of every doc 45 requirement) and §16 (the per-dataset computation). The changes are listed in doc 50. The pre-edit SHA-256 was `10645757…`.

**Planning verification** (read-only on saved outputs; no model fitted; no synthetic PPMI data generated):
- `experiments/20261019T160413Z_stage2_rev3_planning_checks/`, produced by `scripts/56_m2_rev3_planning_checks.py` and `scripts/57_m2_rev3_planning_thresholds.py` (Monte Carlo of standard-normal z-vectors only; seeds `[42, 50, 0]`, `[42, 50, 1]`; 20,000 trials per scenario);
- `tests/test_rev3_planning_functionals.py`.

**Change log against doc 45** (rows labelled by revision):

| revision | § | change | why |
|---|---|---|---|
| R1 | 4 | operating characteristics re-verified: exact per-gate values, Monte Carlo with the estimated gate dependence, and the independence approximation side by side | review item 1 |
| R1 | 5 | bias threshold 0.30 → 0.40 (R2′) | review item 1 |
| R1 | 5 | 21 recovery gates and 15 guard gates (R2b) | review item 3 |
| R1 | 6 | all-item anchored location ν_s (was item 1, threshold 1) | review item 3 |
| R1 | 6 | "artefact" argument corrected | review item 3 |
| R1 | 8 | complete verdict rule | review item 4 |
| R1 | 7 | sandwich checks N1–N11 added | review item 2 |
| R1.1 | 4, 5, 7, 8, 9, 10, 12 (R1.1 numbering) | verdict rule mutually exclusive with precedence and a decision table; F8; C0 only on A; forecast wording; phased execution; control arm report-only | doc 48 |
| R1.2 | 2–14 | approved decisions recorded; doc 45 requirements restored; flag/truth wording; Monte Carlo seeds; §§11–14 renumbered | doc 49 |
| R1.3 | 2, 5–16 | B1 runtime-gate formula (§9); B2 regression path pinned (§§2, 8, 16); B3 sandwich contexts (doc 46 §3.2); B4 flag scope and control-arm isolation (§§7, 8); B5 exact N3/N4/N6 (doc 46 §3.5); B6 Phase 2 ordering and outputs (§9); B7 smoke budget (§§9, 11); S1 failure labels (§5); S2 doc 45 disposition (§15); S3 z and coverage (§5); S4 hardware (§9); S5 D3 comparator (§5); S6 test inputs (§§7, 10); S7 dataset completion (§8); M1–M4 | doc 50 |

---

## 1. Purpose (unchanged)

To test on fresh, independent data whether the Rev 3 pipeline (doc 46) recovers the stage-2 parameters with calibrated intervals:
- Rev 2 two-stage estimation with one S1-b iteration;
- the pooled-moment re-expression S;
- full-chain sandwich intervals with the delta method.

## 2. Frozen before execution (doc 45 §2, restored in R1.2)

All of the following are recorded in this plan and locked together with doc 46 (SHA-256 plus read-only copy) at the plan freeze (§11, step A). This happens **before any recovery dataset is generated**.

| element | frozen value |
|---|---|
| model | M2 (doc 16 B4); 11 tremor and 5 PIGD items; 19 ψ components |
| generating truth | the a-priori `TRUTH_ITEMS` / `TRUTH_LONG` of `scripts/45_m2_pilot.py`, via `truth_on_stage1_scale` on the real design (unchanged) |
| design | the 934 patients' real visit times and treatment indicator (design only; no responses) |
| **seeds** | datasets `np.random.default_rng([42, 49, 1, r])`, r = 0…99. The namespace `[42, 49, …]` is unused elsewhere in `scripts/`, `src/` or `docs/` (checked 2026-10-19). **No other random numbers are used by the procedure.** The planning Monte Carlo seeds `[42, 50, 0]` and `[42, 50, 1]` belong to the planning verification only (§4) |
| stage 1 | `grm` N(0, 1), 61-node GH, L-BFGS-B, maxiter 3000, one start at `initial_items` |
| stage 2 | Laplace over b, Q = 15, P1/P2, absolute line-search tolerance 1e-12, doc 16 D3 bounds, `m2_quad.fit` options unchanged (maxiter 1000), default ψ, cold modes, 32 workers |
| S1-b | one iteration (D1), exactly as doc 37 §6 step (3): items refitted under N(M_ij, K_ij,jj) at ψ̂⁽¹⁾ from `initial_items`; then stage 2 from the default ψ with cold modes |
| convention and transformation | doc 46 §2 (S, joint item map). Identity tolerances 1e-8 |
| interval method | doc 46 §3 (D2): primary full-chain sandwich, delta method, Wald 95 %, no post-hoc small-sample correction (D2b). Secondary (reported only): fixed-item H⁻¹, delta method |
| FD steps | Hessian and cross-derivatives h = 1e-4; G and J_F h = 1e-6; FD-step sensitivity uses 2h |
| reported evaluation | G = 5 AGQ for non-ND patients at each optimum |
| gates, criteria, checks, verdict, budget | §§5–9 |
| code | the driver `scripts/59_m2_rev3_recovery_test.py`, its tests, and every reused module by SHA-256. Reused **unchanged**: `scripts/52` (`pipeline_dataset`, `fit_record`, `coverage`) for stages (1)–(6); `scripts/54` (`raw_chunks`, `eval_s2`, `_s2_task`, `_item_grad_task`, `stacked_sandwich`, `patient_slices`, `pack_set`, `items_from_phi`, `affine_tangent`) for the sandwich; `scripts/55` (`standardise`, `jacobian`) for S; `scripts/56` (`functionals`, `jac`) for the functionals; `scripts/51` and `scripts/45`; `src/ppmi_subtype` modules. All are recorded at the code freeze (§11, step D), before Phase 1. **No code may change after the code freeze.** A defect stops the run (INCOMPLETE or INTEGRITY STOP, §8) and is recorded in a defect note |

**No code, setting, threshold, tolerance or gate may change after freezing, during the run, or after observing any outcome.**

## 3. Datasets (doc 45 §3; regeneration method restored in R1.2)

- R = 100 fresh datasets: `simulate_groups` with seed `[42, 49, 1, r]` from the frozen truth on the real design. Independent of the five development datasets.
- Regeneration check on r = 0 and r = 99 (Phase 1): each is generated twice, and the SHA-256 of the flattened responses must be identical. A mismatch is an INTEGRITY STOP.
- **All 100 are used, in seed order. None may be skipped, replaced or added.**

## 4. Operating characteristics (verified)

**Gate dependence.** z-correlations come from the D4 full-chain covariance of the five development datasets, propagated through G (S) and J_F (functionals), and averaged.
- Per dataset the 36-gate correlation matrix has rank **17**, as predicted: 15 identified ψ directions plus the 2 item-anchored locations.
- Pairs with |r| > 0.9: β1_T with its slope ratio (0.98), γ_T and γ_P with their treatment ratios (≥ 0.99), each L_ij with the matching D-correlation (0.98–1.00), Le10 with corr_ε (0.99), L11_log with the T slope-variance ratio (0.90).
- So the gates are **strongly redundant**: about 17 effective tests.

**Per gate, exact** (z ~ N(0, 1), independent datasets):
- P(coverage ≥ 88/100) = **0.99854** (binomial).
- P(|mean z| ≤ t) = 1 − 2Φ(−10t): **0.9973** at t = 0.30; **0.99994** at t = 0.40; **0.9999994** at t = 0.50.
- Coverage of a 95 % interval:
  - under bias b: 0.949 (b = 0.1), 0.945 (0.2), 0.940 (0.3), 0.921 (0.5), 0.908 (0.6);
  - when the SE is understated by a factor k: 0.925 (k = 1.1), 0.898 (1.2), 0.868 (1.3).
- P(pass the bias rule at t = 0.40 | bias b): 0.9986 (b = 0.1), 0.977 (0.2), 0.841 (0.3), 0.50 (0.4), 0.159 (0.5), 0.023 (0.6).

**Coverage and bias within a gate** are dependent: both depend on the same 100 z. They are handled jointly in the Monte Carlo.

**All gates jointly.** Monte Carlo with dependence (MC SE ≤ 0.004) against the independence approximation.

**Seeds.**
- The columns *coverage only* and *t = 0.30 / 0.40 / 0.50* come from `scripts/57_m2_rev3_planning_thresholds.py`, seed `[42, 50, 1]`. The calibrated t = 0.30 value is **0.893**.
- The parenthesised **0.899**, the *independence approximation* column and the doc 45 verification figures (0.9726, 0.927) come from `scripts/56_m2_rev3_planning_checks.py`, seed `[42, 50, 0]`.
- The two calibrated t = 0.30 values differ by about 2 MC SE.


| scenario (applied to all 36 gates unless stated) | coverage only | **t = 0.30 (doc 45)** | **t = 0.40 (approved, R2′)** | t = 0.50 | independence approximation, t = 0.30 |
|---|---|---|---|---|---|
| calibrated | 0.955 | 0.893 (0.899 in the run with seed `[42, 50, 0]`) | **0.953** | 0.955 | 0.865 |
| SE understated 10 % (coverage 0.925) | 0.364 | 0.316 | 0.362 | 0.364 | 0.216 |
| SE understated 20 % (coverage 0.898) | 0.002 | 0.002 | 0.002 | 0.002 | 0.0001 |
| uniform bias 0.10 SE | 0.950 | 0.538 | 0.917 | 0.949 | — |
| uniform bias 0.15 SE | 0.932 | **0.168** | 0.801 | 0.926 | 0.078 |
| uniform bias 0.20 SE | 0.913 | 0.012 | 0.525 | 0.883 | — |
| Le00_log bias 0.3 only | 0.953 | 0.446 | 0.802 | 0.934 | 0.434 |
| Le00_log bias 0.5 only | 0.908 | 0.020 | **0.151** | 0.474 | 0.020 |
| Le00_log bias 0.7 only | 0.693 | 0.000 | 0.001 | 0.020 | — |
| Le00_log bias 1.0 only | 0.110 | 0 | 0 | 0 | — |

**Verification of the doc 45 figures:**

| doc 45 statement | verified | verdict |
|---|---|---|
| P(all 19 pass C1 \| 0.95) = 0.973 | MC 0.9726 | correct |
| C1 + C2 over 19 ≈ 0.92 | MC 0.927 | correct |
| per-component false-fail of C2 ≈ 0.003 | exact 0.0027 | correct |
| "between about 0.86 and 0.92" with C3/C4 | MC 0.899 | correct as a range |

The independence approximation **understates** the joint pass probability (0.865 vs 0.899), because the gates are positively dependent.

**Why change t = 0.30 → 0.40.** The criterion's purpose, stated before looking at outcomes: detect biases that would materially compromise interval coverage, and tolerate negligible ones.
- **Negligible:** |b| ≤ 0.15 SE costs ≤ 0.3 percentage points of coverage.
- **Material:** |b| ≥ 0.5 SE costs ≥ 3 points.
- At t = 0.30 a calibrated procedure with uniform 0.15 SE bias passes only 17 %, which contradicts the purpose.
- At t = 0.40 it passes 80 %, while a single 0.5 SE bias is detected with probability 0.85 (single-gate pass 0.151), and 0.7 SE with probability 0.999.
- t = 0.50 detects a 0.5 SE bias only about half the time.

**R2′ (approved 2026-10-19): t = 0.40.** The t = 0.30 and 0.50 columns remain as planning comparisons only.

**Why R = 100 is retained.** Calibrated pass about 0.95. 20 % SE understatement detected with probability 0.998, and 10 % with 0.64. Cost about 24 node-hours (§9). R = 120 would raise 10 % detection only modestly, at about +5 node-hours. No compelling reason to change.

**Forecast caveat.** If the development-data mean z values (n = 5; SE of each mean ≈ 0.45) were the true biases, the test would fail with near certainty (S6 scenario at t = 0.30: 0 passes in 20,000 Monte Carlo trials, i.e. a pass probability below about 1.5e-4). For Le00_log alone at a true bias of 0.61 SE, the probability of failing C2 is about 0.999 at t = 0.30, 0.98 at t = 0.40 and 0.86 at t = 0.50 (exact normal calculation).

These forecasts were computed (planning summary, scenarios S5/S6) but **were not used to choose any threshold**. They are noisy: the development means are compatible with zero bias. A FAIL remains a fully possible, legitimate outcome.

## 5. Acceptance criteria [P]

**Gate criteria** (each gate g, on a dataset set X; definitions in §8):
- **C1 — coverage:** #(X datasets whose 95 % interval for g covers the truth) ≥ ⌈0.88 · |X|⌉. With |X| = 100 that is 88.
- **C2 — bias:** |mean over X of z_g| ≤ **0.40** (R2′, approved).
- **C0 — definedness (evaluability):** at most ⌊0.05 · |X|⌋ datasets in X with an undefined interval for g. Every dataset with any undefined interval carries flag F8 (§7), so A \ F contains none, and C0 can fail only on A (at most 5 of 100).

**Gate status on a set X** (a three-way partition used by §8):
- **pass_g(X):** C0 holds, C1 holds and C2 holds.
- **fail_g(X):** C0 holds, and C1 or C2 fails. The gate is *evaluable and fails*.
- **ne_g(X)** (not evaluable): C0 fails. Neither pass nor fail.

An undefined interval counts as not covering in C1, and is omitted from the mean in C2.

**z-scores and coverage (S3, R1.3).** These are computed on the frozen primary interval method:
- **Re-expressed parameter gate k:** z = (S(ψ̂⁽²⁾)_k − ψ_true,k) / SE_S,k. Here ψ_true is the protocol truth on the stage-1 scale (`truth_on_stage1_scale`), and SE_S,k = √(G V_{ψ₂ψ₂} Gᵀ)_kk.
- **Functional gate F:** z = (F̂ − F(ψ_true, it_true)) / SE_F. Here F̂ = F(ψ̂⁽²⁾, φ̂₂), it_true are the true items from `truth_on_stage1_scale`, and SE_F = √(J_F V_{(φ₂,ψ₂)} J_Fᵀ).
- **Covered:** |z| ≤ 1.96, inclusive, using the literal value 1.96.
- **Undefined:** if an estimate, SE or functional is non-finite, or an SE is ≤ 0, the interval for that gate is undefined (flag F8, §7). It is counted as not covered in C1 and omitted from the mean in C2. **No observation is dropped otherwise.**

**Gate sets (R2b, approved):**
- **21 recovery gates:**
  - the 19 re-expressed components S(ψ̂)_k. Intervals are S(ψ̂)_k ± 1.96·SE_S,k from the full-chain V. Under the convention β0′_T and β0′_P are deterministic functions of β1′, γ′ and the design means (doc 46 §2.6); they stay gated, as part of reporting ψ on the convention;
  - the 2 item-anchored locations ν_T and ν_P (§6). **These are the only gates that test the latent location relative to the items, which is the quantity behind the Step 6 β0_P failure.**
- **15 guard gates:** the ψ-only invariants I₁…I₁₅ (§6). They carry the same information as the 19 re-expressed components: S(ψ̂) is a function of them and vice versa. A guard failing while the recovery gates pass signals an implementation error in S, or a parameterisation-dependent Wald interval. It **blocks a PASS** and makes the result INCONCLUSIVE (§8). Guards never create a PASS.

**Le00_log is a recovery gate, with no exemption.**

**Not gated, reported:**
- SD(z) with its χ²(99) 95 % interval;
- mean error and RMSE;
- empirical SD / mean SE;
- coverage under the secondary (fixed-item) V;
- the raw (untransformed) results;
- the N(0, 1)-only comparator;
- **the historical comparator (S5):** the doc 16 D3 rule exactly as defined there, on fresh datasets 0–4. It uses the raw ψ̂⁽²⁾ with fixed-item Wald intervals ψ̂⁽²⁾_k ± 1.96·√(H₂⁻¹)_kk; a non-PD H₂ means nothing is covered (the Step 6 convention); covered in ≥ 4 of 5 for every component. It is reported as "historical D3 comparator". A Rev 3-gate count on datasets 0–4 is reported separately, labelled "Rev 3 gates on datasets 0–4". Neither enters the verdict;
- item recovery;
- **failure labels (S1)** for every gate that fails on A. These are explanatory only and **never change the verdict**:
  - **bias-type:** C2 fails;
  - **dispersion-type:** C2 passes, C1 fails, and the lower bound of the χ²(n − 1) 95 % interval for SD(z) (over the defined z on A) exceeds 1;
  - **mixed:** C2 fails **and** C1 fails with that lower bound > 1;
  - **unclassified:** none of the above.
- **coverage under the secondary interval** (fixed-item H₂⁻¹, through G for the re-expressed parameters, and through J_F restricted to ψ, with items held fixed, for the functionals), for every gate.

## 6. Convention-free functionals: exact definitions, invariance, evaluation

**Notation.**
- b = (b_int,T, b_slope,T, b_int,P, b_slope,P), with D = Cov(b) and Σ_ε = Cov(ε_T, ε_P).
- Subscore s has intercept index i_s (T: 0, P: 2), slope index j_s (T: 1, P: 3), residual index e_s (T: 0, P: 1).
- K_s items with discriminations a_k and thresholds τ_k1 < … < τ_k4; τ̄_k = mean_j τ_kj.

**Relabelling.** θ′_s = (θ_s − d_s)/c_s maps:
- β0_s ↦ (β0_s − d_s)/c_s;
- β1_s, γ_s ↦ ·/c_s;
- D_ij ↦ D_ij/(c_{s(i)} c_{s(j)}), Σ_ε likewise;
- a_k ↦ c_s a_k, τ_kj ↦ (τ_kj − d_s)/c_s.

| # | name | definition | why invariant |
|---|---|---|---|
| I₁, I₂ | slope ratio, s = T, P | β1_s / √D_{i_s i_s} | numerator and denominator both scale by 1/c_s |
| I₃, I₄ | treatment ratio | γ_s / √D_{i_s i_s} | same |
| I₅, I₆ | slope/intercept variance ratio | ½ log(D_{j_s j_s} / D_{i_s i_s}) | both scale by 1/c_s² |
| I₇, I₈ | residual/intercept variance ratio | ½ log(Σ_ε,{e_s e_s} / D_{i_s i_s}) | both scale by 1/c_s² |
| I₉–I₁₄ | random-effect correlations | D_ij / √(D_ii D_jj), i > j | each factor's scaling cancels (also across subscores) |
| I₁₅ | residual correlation | Σ_ε,TP / √(Σ_ε,TT Σ_ε,PP) | same |
| ν_T, ν_P | item-anchored baseline location | (1/K_s) Σ_k a_k (β0_s − τ̄_k) | a_k′(β0′ − τ̄_k′) = c a_k · (β0 − τ̄_k)/c |

**Properties (verified):**
- ν_s is the mean, over items, of the logit-scale distance between the baseline latent mean and the item's average threshold: a response-scale location.
- The 15 ψ-only invariants are a **complete, minimal** set for the 4-parameter relabelling group acting on ψ: 19 − 4 = 15. β0 is absorbed by d, so it appears only with the items, in ν_s.
- Unit tests: invariance under arbitrary relabellings to ≤ 1e-12; Jacobian rank 15 in ψ and 17 overall.
- On the five development datasets: invariance under the estimate's own S and under an arbitrary relabelling to ≤ 9e-16. The truth functionals are equal on the raw a-priori and stage-1 scales to 4e-16. All 17 SEs are finite.
- Every functional is a smooth closed-form function. It is undefined only if D_ii ≤ 0 or Σ_ε,ss ≤ 0, which the log-Cholesky parameterisation rules out.

**Evaluation.**
- Point value: F(ψ̂₂, φ̂₂) on the raw scale. No transformation is needed, and none is applied.
- Interval: delta method with V_F = J_F V_{(φ₂,ψ₂)} J_Fᵀ (full chain; J_F by central differences, h = 1e-6).
- Truth: F(ψ_true, it_true), with it_true from `truth_on_stage1_scale` (the same items that generated the data, on the stage-1 scale).

**Thresholds.** The same C0–C2 as the ψ gates. Same scale (z units), same purpose, same operating characteristics (§4).

**Not redundant with each other, but redundant with ψ.** The 15 guards are functionally independent (rank 15). Several are near-duplicates of a re-expressed component (|r| > 0.9, §4), which is why they are guards and not additional recovery evidence. The 2 ν gates are not duplicated by any ψ gate.

## 7. Diagnostics and numerical stability (doc 45 §7, restored in R1.2, plus doc 46 §3.5)

**Per dataset, reported:**
- **Convergence of every fit:** status, message, iterations, evaluations, ‖∇‖ (B11's 0.934 shown for reference only), bound hits, P1/P2 penalty evaluations by reason, non-ND patients at accepted evaluations.
- **At each optimum:** Hessian PD (min/max eigenvalue); the reported evaluation (fallback patients, budget triggers).
- **Transformation identities** (each ≤ 1e-8; a failure is flag F5):
  - S(ψ_true) = ψ_true;
  - idempotence S(S(ψ̂)) = S(ψ̂);
  - invariance S(relabel(ψ̂)) = S(ψ̂) for a fixed test relabelling;
  - G annihilating the four relabelling tangents (≤ 1e-6 relative), with rank(G) = 15 (N10).
- **Sandwich checks N1–N11** (doc 46 §3.5). On fresh datasets a failure of N1, N3, N4, N5 or N6 is flag F6, a failure of N8 is flag F7, a failure of N10 is flag F5, and N7 leads to flag F8. N11, and N3/N4 on the regression dataset, are integrity checks (§8). N2 and N9 are reported only.
- **FD-step sensitivity** (datasets 0–2, Phase 1): H₂ and C₂ at h = 2e-4 vs 1e-4. Relative SE_S differences are reported; > 5 % is flag F7.
- **Item recovery** (reported, not gated): re-expressed items against the true items (correlation, max error), and the coverage of item intervals from V_φφ.
- **Le00_log secondary analysis** (doc 46 §5; reported only, never rescues a failure): first-order tremor/PIGD and affine/non-affine attribution; bias; SD of z; coverage under the primary vs the secondary V.

**Transformation-identity test inputs (S6, R1.3):**
- the fixed test relabelling is c = (c_T, c_P) = (1.13, 0.91), d = (d_T, d_P) = (−0.07, 0.05);
- the numerical rank of G counts singular values > 1e-8 × the largest.

**Flags (B4, R1.3).** Flags are evaluated **for the main datasets only**. The control arm never sets a flag (§10).

| flag | inputs and trigger | scope |
|---|---|---|
| F1 | a stage-2 fit not converged (L-BFGS-B success flag false) or iterations ≥ maxiter (1000). (R1.3: the R1.2 "time" trigger is removed, because no per-fit time cap exists) | both stage-2 fits, (2) ψ̂⁽¹⁾ and (4) ψ̂⁽²⁾ |
| F2 | an item fit not converged, or any discrimination a ∉ [0.05, 20] | all four item fits: N(0, 1) T and P (stage 1); S1-b T and P |
| F3 | a bound hit (within 1e-6 of a doc 16 D3 bound) | ψ̂⁽¹⁾ and ψ̂⁽²⁾ |
| F4 | in the reported evaluation (`evaluate_reported`, G = 5): fallback patients > 0 or budget triggers > 0 | at ψ̂⁽¹⁾ and at ψ̂⁽²⁾ |
| F5 | any transformation-identity failure: S(ψ_true) = ψ_true (> 1e-8; computed once per job, and applying to every dataset of that job if it fails); idempotence of S(ψ̂⁽²⁾) (> 1e-8); invariance under the fixed test relabelling (> 1e-8); N10 | per dataset |
| F6 | a sandwich-check failure: N1, N3, N4, N5 or N6 (doc 46 §3.5); or any penalised finite-difference evaluation in H₁ or H₂; or any chunk failure or non-finite value in the C₁/C₂ evaluations | per dataset |
| F7 | N8 > 0.05 | fresh datasets 0–2 only |
| F8 | any undefined interval for any of the 36 gates (V, V_S or V_F non-finite or not PSD by N7; A singular; a gate SE non-finite or ≤ 0) | per dataset |

**Flags never make a dataset incomplete.** Completion is defined in §8 (S7); flags affect the verdict only through F, as defined in §8.

## 8. Complete verdict rule [P] (R1.1; R3 approved; integrity conditions extended in R1.2)

**Definitions:**
- A = all R = 100 datasets, in seed order.
- F ⊂ A = the datasets with any flag F1–F8.
- A \ F = the unflagged datasets.
- Required gates = the 21 recovery gates and the 15 guards (§5).
- Gate status pass / fail / ne on A and on A \ F as defined in §5.

**Run-level conditions:**
- **(I) integrity:** a preflight check (§11, step E), a regeneration check (§3), the truth-swap test (§10.1; Phase 1, fresh dataset 0) or a **regression check** on the old development dataset `[42, 45, 1, 0]` (Phase 1, node002) fails. The regression checks, restored from doc 45 §8:
  - **R-a.** The Rev 3 driver reproduces Step 6's first-stage estimate ψ̂⁽¹⁾ and final estimate ψ̂⁽²⁾ **bitwise**, before S. References: `experiments/20261018T222857Z_stage2_step6_full_pipeline_recovery/step6/S1_dataset0.json`, `stage2_fit_1.psi` and `stage2_fit_2.psi`.
  - **R-b.** S(ψ̂⁽²⁾) agrees with D7's re-expressed estimate to **≤ 1e-12** (absolute, componentwise). Reference: `…_stage2_step6_diagnostics/d7_readonly/D7_estimates_long.csv`, dataset 0, `psi2_step6`, column `std`.
  - **R-c (= N11).** The full-chain SEs agree with D4's to **≤ 1e-6** (relative). Reference: `…/d4/D4_dataset0.json`, `se_psi2.full_chain`.
  - **R-d.** N3 and N4 hold on the regression dataset (≤ 1e-5 relative).

  These checks confirm that only S, the interval method and the new gates are new. **Any regression failure ends the run as INTEGRITY STOP.** It is documented and is **not** followed by any tolerance change, retuning or code change within this test.
- **(X) execution:** any of the following:
  - an exception in any stage of any **main** dataset;
  - a runtime-gate breach (§9);
  - a job reaching its time limit before all of its **main** datasets were complete;
  - the budget cap (§9) preventing a required stage;
  - a non-finite objective or estimate at an accepted optimum of a main dataset;
  - fewer than 100 main datasets complete.

  Control-arm exceptions, non-finite values or timeouts are reported and never cause (X).

**Dataset completion (S7).** A main dataset is **complete** when its output file `main/dataset_r{rr}.json` exists with `"complete": true`. That requires stages (1)–(6), the sandwich, S, and all 36 gate values (z, SE and coverage) to be written, or, for gates whose interval is undefined, an explicit undefined record (F8).
- **(Fl) flag excess:** |F| > 10.
- **(RF) robust recovery failure:** some recovery gate g has fail_g(A) **and** fail_g(A \ F).
- **(AP) all pass:** every required gate has pass_g(A) **and** pass_g(A \ F).

**Verdict, by precedence** (the first matching row applies, so the verdicts are mutually exclusive and exhaustive):

| order | verdict | condition |
|---|---|---|
| 1 | **INTEGRITY STOP** | (I) |
| 2 | **INCOMPLETE** | (X), and not (I) |
| 3 | **INCONCLUSIVE — FLAG EXCESS** | (Fl). Too many numerically compromised datasets for **any** claim, pass or fail |
| 4 | **FAIL** | (RF) |
| 5 | **PASS** | (AP) |
| 6 | **INCONCLUSIVE — OTHER** | every remaining case. The reason codes are listed below |

**Reason codes for row 6** (all that apply are reported):
- **NE:** some required gate is ne on A.
- **FLAG-SENSITIVE:** some recovery gate fails on exactly one of A and A \ F.
- **GUARD:** some guard fails on A or on A \ F.

Row 6 is reached only when no recovery gate fails robustly and at least one gate is not "pass on both sets".

**Gate-level decision table** (for one gate; the entries are the gate's status on (A, A \ F); A \ F is always evaluable, by F8):

| status on (A, A \ F) | recovery gate contributes | guard contributes |
|---|---|---|
| (pass, pass) | nothing against PASS | nothing against PASS |
| (fail, fail) | **RF → FAIL** (row 4) | GUARD → blocks PASS (row 6). **Never FAIL** |
| (pass, fail) or (fail, pass) | FLAG-SENSITIVE → blocks PASS (row 6) | GUARD → blocks PASS (row 6) |
| (ne, pass) or (ne, fail) | NE → blocks PASS (row 6). **Never FAIL by itself** | NE → blocks PASS (row 6) |

**Boundary cases checked:**

| case | verdict | why |
|---|---|---|
| undefined intervals in ≤ 5 datasets, all gates otherwise pass on A and A \ F, \|F\| ≤ 10 | PASS | C0 holds. The undefined intervals count as not covered on A, and PASS still requires C1 there |
| a gate with > 5 undefined intervals (C0 fails on A), no robust recovery failure elsewhere | INCONCLUSIVE (NE) | ne is never a failure claim |
| the same, but another evaluable recovery gate fails on both sets | FAIL | the FAIL rests on the evaluable gate, not on the C0 failure |
| C0 failing on only one set | can only be A, since A \ F has no undefined intervals by F8; → NE | — |
| a recovery gate failing only after exclusion (pass on A, fail on A \ F) | INCONCLUSIVE (FLAG-SENSITIVE) | — |
| a recovery gate failing only on A | INCONCLUSIVE (FLAG-SENSITIVE) | — |
| guard-only failure (recovery gates all pass on both sets) | INCONCLUSIVE (GUARD) | a guard can never produce FAIL |
| \|F\| = 11 or more | INCONCLUSIVE — FLAG EXCESS, even if some recovery gate fails robustly or all gates pass | R8, approved |
| \|F\| = 10 | handled by rows 4–6 | the limit is inclusive |
| 99 datasets completed (any cause) | INCOMPLETE | no verdict from partial data |
| an execution error in dataset 57 | INCOMPLETE | no automatic rerun |
| Le00_log failing on both sets | FAIL | no exemption |

**Properties:**
- **Exclusion can never create a PASS.** (AP) requires pass on A, the full set, as well as on A \ F.
- **No dataset or gate can be dropped after observation.** A, the gate list and the thresholds are frozen.
- **How the flags relate to the truth.** A \ F is determined by flags defined in advance:
  - F1–F4 and F6–F8 are computed from the fits, the numerical checks and the interval computations alone, without reading the truth.
  - F5 includes the identity S(ψ_true) = ψ_true, which **does read ψ_true** to check the transformation. It involves no estimate and is the same for every dataset.
  - **No flag depends on any comparison between an estimate and the truth** (errors, z values or coverage).
- **A guard failure or a C0 failure never yields FAIL.** FAIL needs an evaluable **recovery** gate failing on both sets.
- **Le00_log** is a recovery gate with no exemption.
- **Control-arm isolation (R9, B4):** the control arm never enters A, F, the gates or the verdict, and cannot cause INCOMPLETE.
- **Every event has a fixed effect:**

  | event | effect |
  |---|---|
  | convergence failure | flag F1 / F2; estimates used as computed |
  | non-PD Hessian | affects only the secondary (fixed-item) method, which is reported only |
  | V not PSD or SE undefined | undefined interval: not covered; omitted from the mean z; flag F8 |
  | fallback or budget event at an optimum | flag F4 |
  | a numerical-check failure | flags F5 / F6 / F7 |
  | an execution error | INCOMPLETE |

A PASS satisfies only the amended recovery pre-requisite, after review. It authorises nothing by itself, and it is **not** evidence that the sandwich is calibrated in other settings.

## 9. Budget and execution (doc 45 §9 budget table restored in R1.2; R4 approved)

| component per dataset | basis | minutes |
|---|---|---|
| stage-1 item fits (batched across datasets) | measured (Steps 5/6: 10 fits in about 55 s) | about 0.2 |
| stage-2 fit (2) | measured 169–269 s | about 3.7 |
| S1-b item refit (3) | measured 51–63 s | about 1.0 |
| stage-2 fit (4) | measured 191–220 s | about 3.5 |
| Hessians H₁, H₂ (warm) | measured about 54 s each | about 1.8 |
| full-chain sandwich (C₁, C₂, item Jacobians, per-patient scores) | measured (D4) about 200 s | about 3.4 |
| reported evaluations, S, G, functionals | measured, seconds | about 0.2 |
| **total per dataset** | | **about 13.8** |

**Totals (estimates):**
- R = 100: about 23 node-hours.
- Control arm (10 × stage-2 fit plus H): about 0.8.
- Regression check: about 0.25.
- FD sensitivity: about 0.15.
- Smoke: about 0.1.
- **Projected about 24.3 node-hours.**

**Budget and limits (B1, B7, R1.3):**

| item | value |
|---|---|
| hard cap | **30 node-hours, cumulative over all jobs of this test, the smoke runs included** (AllocNodes × ElapsedRaw from `sacct`) |
| smoke run | SLURM limit **0.5 h** per run. At most two repeats (three runs in total). Each repeat must be needed to correct an implementation defect and must come before the code freeze. Every run's reason and node-hours are recorded |
| Phase 1 | SLURM limit **2 h** on node002 |
| Phase 2 | two jobs, each with SLURM limit **L₂ = min(13.5 h, (30 − U)/2)**, where U is the node-hours already used (smoke runs and Phase 1). Planned worst case with one smoke run: 0.5 + 2 + 2 × 13.5 = **29.5 ≤ 30** |

**Runtime gate (B1).** Evaluated at the end of Phase 1, from Phase 1 measurements only.

- **Per-dataset time.** For each fresh dataset r ∈ {0, 1, 2}:

  t_r = T_stage1 / n_batch + T_r, in seconds,

  where:
  - T_stage1 is the wall time of Phase 1's batched stage-1 item fits;
  - n_batch = 4 is the number of datasets in that batch (the regression dataset and fresh datasets 0–2);
  - T_r is the wall time of stages (2)–(6), the sandwich, S, the functionals and the gates for dataset r, **excluding** the N8 step-size recomputation and the truth-swap test.
- **Control-arm rate.** t_c = the mean over the six fresh-dataset stage-2 fits (fits (2) and (4) of datasets 0–2) of (fit wall time + its Hessian wall time).
- **Phase 1 usage.** U₁ = the Phase 1 job's elapsed wall time at the gate, in hours, times 1 node.
- **Projection.**

  **P = U₁ + 97 · mean(t₀, t₁, t₂)/3600 + 10 · t_c/3600** node-hours.
- **The gate passes only if all of the following hold:**
  - (i) every Phase 1 stage-2 fit, the regression dataset's two included, took ≤ 3600 s;
  - (ii) P ≤ 28;
  - (iii) U_smoke + P ≤ 30, where U_smoke is the smoke runs' recorded node-hours;
  - (iv) the projected per-job Phase 2 wall time, 49 · mean(t_r) + 5 · t_c, is ≤ L₂.
- **Missing or invalid timing** (missing, non-finite, or from a failed or incomplete stage) **fails the gate.** It never yields an optimistic estimate.
- **Consequence.** A failed gate stops the test before Phase 2: INCOMPLETE (§8 X).

**Execution arrangement (R4, approved; B6, S4 in R1.3):**

| phase | where | content | SLURM time limit |
|---|---|---|---|
| Phase 1 | **node002** (required for the bitwise regression R-a against Step 6, which ran there) | preflight; regeneration check on r = 0 and r = 99; regression checks R-a–R-d on `[42, 45, 1, 0]`; fresh datasets 0–2 with the N8 step-size check and the truth-swap test (dataset 0); runtime gate | 2 h |
| Phase 2 (only after the runtime gate passes; authorised by the Final Execution Directive) | **two eligible normal-partition nodes, exclusive allocation, two jobs with fixed assignments:** job 2a = main datasets 3–51, then control arm 0–4; job 2b = main datasets 52–99, then control arm 5–9 | — | L₂ each |

**Phase 2 safeguards:**
- **Eligible nodes (S4):** normal partition, exclusive allocation, ≥ 32 cores. Each job records the node, CPU model, core count and Python/NumPy versions per dataset. Two nodes do **not** guarantee completion within L₂.
- **Ordering (B6):** each job processes **all of its main datasets first, in increasing seed index**, and only then its control-arm datasets. A timeout during the control arm leaves the main verdict unaffected. A timeout before all assigned main datasets are complete is INCOMPLETE.
- **Outputs:** each dataset is written atomically, to a temporary file then renamed, as `main/dataset_r{rr}.json` or `control/control_r{rr}.json` (rr zero-padded). A job never overwrites an existing complete file; it refuses to start if one of its assigned files already exists. Partial results are recovered by reading the complete files; nothing is recomputed automatically.
- **Seed assignments** are fixed and may not be rebalanced.
- **Cross-node differences:** Phase 2 results may differ from node002 in the last bits. They are expected to be far below every tolerance, but a borderline coverage indicator could in principle flip. This is not checked; cross-node bitwise equivalence is **not** guaranteed, and N11/R-a run on node002 only.

## 10. Checks against a transformation artefact (doc 45 §6, corrected)

1. **S never reads the truth** (restored detail from doc 45 §6):
   - a code-review item, plus a unit test that the transformation function's inputs are only ψ̂ and the design;
   - a run-time **truth-swap test (S6)**: for fresh dataset 0, the gate evaluation is repeated with ψ_true replaced by **ψ_true + 1** (componentwise) in the comparison code. S(ψ̂⁽²⁾) and SE_S must be **bit-identical** to the original evaluation, while the z values change. A failure is an INTEGRITY STOP.
2. **Guards I₁–I₁₅ and recovery gates ν_T, ν_P** are evaluated with no transformation.
   - Because S(ψ̂) is a function of I₁–I₁₅ (doc 46 §2.6), a point-estimate artefact is mathematically impossible when S is correctly implemented. The guards therefore test the implementation of S and the parameterisation-dependence of the Wald intervals.
   - The ν gates test the item–latent location linkage, which no ψ-only quantity can test.
3. **Control arm** (true items, datasets 0–9). **Reported only. It has no effect on the verdict** (R9, approved).
   - Even with correctly scaled items, the fitted moments of ψ̂ vary by sampling, so S changes the estimates by amounts of the order of the sampling SE of (c, d). For example, D7's re-expression of the D6 fits gave c_P 0.986–1.012 and d_P up to ±0.02, about 1 SE of β0′.
   - The R1 expectation "≪ 1 SE" was unjustified and is withdrawn.
   - Procedure: stage 2 with the generating items it_true (Laplace Q = 15, default ψ, cold modes, maxiter 1000), FD Hessian (h = 1e-4), then S.
   - Reported: (S(ψ̂) − ψ̂)/SE per component, with SE = √diag(H⁻¹) on the raw scale; its mean over the completed control datasets; and the moments (μ_s, v_s). A systematic mean shift would be noted for review.
4. **Raw results** and the distribution of (c, d) are reported.

## 11. Execution sequence and authorisations (doc 45 §8 sequence, restored in R1.2)

| step | content | authorisation |
|---|---|---|
| A | **Protocol freeze:** docs 46 and 47 locked together (SHA-256 plus read-only copies) | authorised (Final Execution Directive; D6) |
| B | **Driver implementation:** new driver and unit tests (new files only); reused modules unchanged | authorised (Final Execution Directive) |
| C | **Unit tests** on the login node (no jobs, no recovery data) | covered by B |
| D | **Labelled smoke run:** the full code path (stages (1)–(6), sandwich, S, functionals, gates, flags, verdict logic) on 60 patients of the old development dataset `[42, 45, 1, 0]`, with reduced iterations (stage-2 maxiter 5, S1-b item maxiter 20), written to `smoke_codepath/` and labelled **NOT A RESULT**. SLURM limit 0.5 h; at most two repeats; counted in the budget (§9). Then the **code freeze**: driver, tests and reused modules recorded by SHA-256 | authorised (Final Execution Directive). Its outputs are never results. A defect may be fixed only before the code freeze, with a defect note, and the smoke run repeated |
| E | **Preflight** in the Phase 1 job: frozen locks (including docs 46 and 47), code hashes against the code freeze, panel MD5, truth equal to the record, seed list, regeneration check | covered by the execution authorisation |
| F | **Regression checks** R-a–R-d on `[42, 45, 1, 0]` (Phase 1, node002), then fresh datasets 0–2 (with the truth-swap test on dataset 0) and the runtime gate | covered by the execution authorisation. A failure is an INTEGRITY STOP (§8) |
| G | **Phase 2:** fresh datasets 3–99 and the control arm, on two normal nodes (§9) | authorised (Final Execution Directive), **conditional** on Phase 1 passing every integrity condition and the runtime gate |
| H | Criteria, verdict (§8), post-run integrity, report. **Stop for review** | — |

No step may begin before its preceding gate has passed. No recovery dataset (seeds `[42, 49, 1, r]`) is generated before step E.

## 12. Outputs (doc 45 §10, restored in R1.2)

**Directory:** `experiments/<UTC>_stage2_rev3_recovery_test/`, with subdirectories `smoke_codepath*/`, `phase1/`, `phase2a/`, `phase2b/` and `evaluation/`. It contains the baseline, the plan locks, the code-freeze hashes and per-dataset JSON (`main/dataset_r{rr}.json`, `control/control_r{rr}.json`, `regression/old_dataset0.json`):
- fits and logs;
- H and V;
- S, (c, d) and z;
- the functionals;
- flags;
- timings;
- node and CPU.

**Tables:**
- coverage, bias, SD(z) and RMSE for the 19 re-expressed components (primary, secondary and raw), and for the 17 functionals (recovery and guard);
- the gate statuses on A and A \ F, and the verdict with its reason codes;
- the control arm;
- the N(0, 1)-only comparator;
- the Le00_log analysis;
- item recovery;
- the numerical checks, including the regression R-a–R-d;
- runtime against the budget.

**Figures:**
- z distributions per gate;
- coverage with Clopper–Pearson intervals;
- the (c, d) distribution.

**Plus** `MANIFEST.sha256`, the post-run integrity record and the report.

## 13. What a pass or fail would and would not establish (doc 45 §11, restored in R1.2)

**A PASS** would establish that, for the a-priori truth and the real design, the Rev 3 pipeline gives approximately calibrated 95 % intervals and small bias, under the pre-registered gates, for all 21 recovery gates (the 19 components on the pooled-moment convention and the 2 item-anchored locations), with all 15 guards consistent. After review, that would satisfy the binding recovery pre-requisite of the amended sequence. It would **not** establish that the sandwich is calibrated in other settings.

**A FAIL** would establish that Rev 3 does not meet its own pre-registered criteria. The failing gates, with their failure labels (bias-type, dispersion-type, mixed, unclassified; §5) and the Le00_log analysis, would guide any further amendment. A **bias-type** failure points to point-estimate (recovery) error; a **dispersion-type** failure points to uncertainty-calibration error. Neither label changes the verdict.

**INCONCLUSIVE or INCOMPLETE** establishes neither.

**None of the outcomes** would establish:
- performance under other truths, designs or misspecification;
- anything about real PPMI data, model fit, subtypes or D1.

## 14. Decisions (approved 2026-10-19; user's Final Decision Record R1.1)

| # | decision | approved resolution |
|---|---|---|
| R1 | number of datasets | **100 fresh datasets**, seeds `[42, 49, 1, r]`, r = 0…99 |
| R2′ | bias threshold | **t = 0.40** (§§4–5) |
| R2b | gate structure | **21 recovery gates** (19 re-expressed components + ν_T, ν_P) **and 15 guard gates** (I₁–I₁₅) (§§5–6) |
| R3 | verdict rule | **the revised mutually exclusive rule** (§8), with its precedence, definedness requirement (C0) and no-pass-by-exclusion protection |
| R4 | execution | **Phase 1 on node002; Phase 2 on two normal nodes**, with the safeguards of §9 |
| R5 | doc 16 D3 ≥ 4/5 rule | reported on fresh datasets 0–4 as a comparator only (part of D4 in doc 46 §7) |
| R6 | control-arm size | 10 datasets (plan specification, §10; no alternative is open) |
| R7 | doc 46 decisions | D1, D2, D2b, D3/D4 and D6 as approved in doc 46 §7 |
| R8 | flag excess | **more than 10 flagged datasets → INCONCLUSIVE**, even if a recovery gate meets the robust-failure rule |
| R9 | control arm | **report-only**; it does not determine the verdict |

## 15. Disposition of every doc 45 requirement (S2, R1.3)

| doc 45 § | requirement | disposition | where / why |
|---|---|---|---|
| 1 | purpose (fresh data; genuine, not convention-artefact, recovery) | restored | §1, §10 |
| 2 | frozen elements (model, truth, design, seeds, stage settings, S1-b, convention, intervals, FD steps, reported evaluation, code) | restored | §2 (code hashes fixed at the code freeze, §11 D) |
| 2 | "recorded in the FINAL version of this plan" | replaced | docs 46 and 47 are frozen together (D6) |
| 3 | R = 100, seeds `[42, 49, 1, r]`, independence from the development data | restored | §3 (R1) |
| 3 | regeneration check r = 0 and r = 99 (SHA-256 of flattened responses) | restored | §3, Phase 1 |
| 4 | OC analysis and the inadequacy of ≥ 4/5 | restored, re-verified | §4 |
| 5 | C1 coverage ≥ 88/100 | restored | §5 |
| 5 | C2 bias \|mean z\| ≤ **0.30** | superseded | t = 0.40 (R2′); 0.30 kept only as a planning comparison in §4 |
| 5 | C3/C4 co-primary on 17 functionals | replaced | 21 recovery gates (19 re-expressed + ν_T, ν_P) and 15 guards (R2b); §§5–6 |
| 5 | reported SD(z), mean error, RMSE, SD/SE, secondary coverage, D3 comparator | restored and made exact | §5 (S1, S3, S5) |
| 6.1 | no truth in S; truth-swap test | restored, inputs fixed | §10.1 (S6), §8 (I) |
| 6.2 | functional 3 = item-1, threshold-1 anchored location | superseded | all-item ν_s (§6), which is less arbitrary and invariant (R2b) |
| 6.2 | "passing C1–C2 while failing C3–C4 → test fails" | superseded | guards block a PASS and give INCONCLUSIVE (GUARD), never FAIL (R3). S(ψ̂) is a function of the invariants, so a guard failure signals an implementation or parameterisation problem, not recovery failure |
| 6.3 | control arm: "S should change ≪ 1 SE"; large change → reserved for review | superseded | report-only (R9); the expectation was unjustified (§10.3) |
| 6.4 | raw results and the (c, d) distribution | restored | §10.4, §12 |
| 6.5 | N(0, 1)-only comparator | restored | §5 (reported) |
| 7 | convergence, Hessian PD, reported evaluation | restored | §7 |
| 7 | transformation identities | restored, inputs fixed | §7 (S6), F5 |
| 7 | sandwich checks (additivity, relabelling identity, symmetry) | restored, made exact | doc 46 §3.5 (N1–N11), §7 F6 |
| 7 | "saved-NLL reproduction" | replaced | covered on the regression dataset by R-a (bitwise ψ̂⁽¹⁾, ψ̂⁽²⁾), which implies the same NLL. Fresh datasets have no saved NLL to reproduce |
| 7 | FD-step sensitivity, datasets 0–2 | restored | N8, F7 |
| 7 | item recovery; Le00_log secondary analysis | restored | §7 |
| 8 | sequence (freeze, driver, unit tests, smoke, preflight, regression, datasets, criteria) | restored | §11 |
| 8 | regression on `[42, 45, 1, 0]` (bitwise ψ̂⁽¹⁾/ψ̂⁽²⁾, D7 S ≤ 1e-12, D4 SEs ≤ 1e-6) | restored | §8 (I) R-a–R-d, §16 |
| 8 | runtime gate after fresh datasets 0–1 | superseded | after Phase 1 (fresh datasets 0–2), with the exact formula of §9 (B1). The extra dataset improves the timing estimate and gives three datasets for N8 |
| 8 | outcome classes with the "dependence rule" for flags | superseded | the R1.1 mutually exclusive verdict rule (R3): §8 |
| 8 | "> 10 % flagged → reserved for review" | superseded | \|F\| > 10 → INCONCLUSIVE — FLAG EXCESS (R8) |
| 8 | flags F1–F6 | replaced | F1–F8 with exact scope (§7) |
| 8 | hard stop conditions; no change of settings | restored | §8 (I, X), §2 |
| 9 | per-component budget, 30 node-hour cap | restored | §9, with smoke runs counted (B7) |
| 9 | runtime gate ≤ 28 | restored, made exact | §9 (B1) |
| 9 | execution: node002 alone **or** two nodes | superseded | Phase 1 node002, Phase 2 two nodes (R4) |
| 10 | outputs | restored | §12 |
| 11 | what a pass or fail establishes | restored | §13 |
| 12 | open decisions R1–R7 | superseded | approved decisions (§14) |

## 16. Per-dataset computation (B2, R1.3)

For each main dataset (seed `[42, 49, 1, r]`) and for the regression dataset (`[42, 45, 1, 0]`), the driver performs the following, using the reused functions of §2 unchanged:

1. **Stage 1.** `fit_items_multistart(Y, tag, starts = 1)` for T and for P, through the `scripts/51` process pool (batched across the datasets of the job).
2. **Stages (2)–(6).** `scripts/52` `pipeline_dataset` with an `Ops` object that makes **exactly** the `scripts/52` calls:
   - `m2_quad.fit(Q = 15, psi0 = default ψ, jobs = 32, maxiter = 1000)`;
   - `fd_hessian` on `Objective(Q = 15, jobs = 32)` with the fit's state;
   - `evaluate_reported(G = 5, init_state = fit state)`;
   - the S1-b refit via `visit_latent` and the `scripts/51` pool, maxiter 3000.
3. **Sandwich.** The doc 46 §3.2 path, in the `run_d4` order:
   - (a) base evaluation at (ψ̂⁽¹⁾, φ̂₁) and (ψ̂⁽²⁾, φ̂₂), per-patient, cold;
   - (b) C₁ and C₂;
   - (c) A₁, A₃, E₃;
   - (d) per-patient item scores;
   - (e) V = A⁻¹BA⁻ᵀ, with H₁ and H₂ from step 2.
4. **S, G, functionals, J_F, V_S, V_F,** then the gates, the checks N1–N10, and the flags.

**Regression comparison:**
- **R-a:** ψ̂⁽¹⁾ and ψ̂⁽²⁾ are compared bitwise (`numpy.array_equal`) with `step6/S1_dataset0.json`.
- **R-b:** S(ψ̂⁽²⁾) is compared with `D7_estimates_long.csv` (dataset 0, `psi2_step6`, column `std`), max absolute difference ≤ 1e-12.
- **R-c:** the raw full-chain SEs, √diag V_{ψ₂ψ₂}, are compared with `D4_dataset0.json` `se_psi2.full_chain`, max relative difference ≤ 1e-6. These are **not** the re-expressed SEs.
- **R-d:** N3 and N4 ≤ 1e-5 in all four directions.

