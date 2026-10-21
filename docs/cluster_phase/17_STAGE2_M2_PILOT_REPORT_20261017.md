# Stage 2 M2 pilot: synthetic recovery, fit and validation checks

**Date:** 2026-10-17.
**Specification:** `16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.md`, frozen, SHA-256 `c2eace18999f45e05324575e58d6933109c3836be67f832bfee55c8ad6bffbbc`. The pilot verified this hash before starting.

**Run:** `experiments/20261016T200350Z_stage2_m2_pilot/results/`. SLURM job 13822 on node002, single-threaded BLAS, 23 min 48 s wall time (1,420 s of computation).

**Status:**
- **Implementation complete and unit-tested.**
- **Scientific validation NOT passed.** Three of the five B11 checks failed. Under doc 16 B11, the fidelity gates G1–G7 were therefore **not run**.
- No calibration, no positive control and no evaluation statistic was computed.
- The pilot is label-free: the label rule is never called, and the cohort uses only *whether* a visit is labelled.

---

## 1. Implementation (successful, independent of the validation result)

**Files:**
- `src/ppmi_subtype/m2_grm.py`: the bivariate item-level GRM. It has:
  - the 19-parameter Laplace marginal likelihood with exact gradient, plus the finite-difference Hessian;
  - adaptive Gauss–Hermite and importance-sampling reference integrals;
  - the exact conditional of b given θ, the elliptical slice sampler and simulation;
  - integer item generation and labels via the unchanged `subtype.classify`. Label generation is implemented and tested but was not used in the pilot.
- `src/ppmi_subtype/m2_gates.py`: the label-free features and gates G1–G7, plus rank-normalised split-R̂ and bulk ESS.
- `scripts/45_m2_pilot.py`: the pilot driver.
- `tests/test_m2_grm.py`: 10 tests, all passing. They cover:
  - parameter round trip and covariance structure;
  - **the analytic gradient against finite differences**;
  - Laplace against quadrature and importance sampling;
  - exactness when no items are observed;
  - b-conditional properties and an elliptical-slice prior check;
  - integer items and label-rule cases (both-zero, PIGD-zero, lattice);
  - the gate rules, including G4 failure making target 2 uninformative;
  - the dispersion formula against the audit;
  - R̂ and ESS;
  - a small longitudinal recovery with items fixed at truth (250 patients; every |z| < 4).
- Re-run with the existing `test_grm.py` and `test_grm_ou.py`: 40 passed.

**One implementation setting changed after the smoke run, before the pilot.** L-BFGS-B tolerances were tightened (`ftol` 1e-15, `gtol` 1e-7). The default stopped at ‖∇‖ = 0.66, above the frozen criterion 1e-3 × patients. This is an optimiser setting, not a model change.

**Smoke runs** (120 patients, reduced counts, code path only, not results): `experiments/_smoke/20261016T195632Z_stage2_m2_pilot_smoke`, `…200038Z_…_smoke2`.

## 2. Cohort

934 patients and 4,295 visits; all 16 items are complete. Treatment indicator: `PDTRTMNT == 1`. The 6 visits with it missing were coded 0, as frozen.

## 3. Pre-specified B11 checks

| check | criterion (doc 16 B11 / D3) | result | status |
|---|---|---|---|
| Item fit, tremor | 3 starts; spread ≤ 0.1; converged; a ∈ [0.05, 20] | spread 8.4e-4; all converged | **PASS** |
| Item fit, PIGD | same | spread 3.0e-5; all converged | **PASS** |
| Longitudinal fit | 3 starts agree (≤ 0.1); ‖∇‖ < 0.934; no bound hits; Hessian positive definite | spread 2e-10; ‖∇‖ 5.2e-4; no bound hits; minimum Hessian eigenvalue 41.3 | **PASS** |
| Simulation recovery | every ψ component covered by its 95 % Wald interval in ≥ 4 of 5 datasets | 16/19 pass. **β0_P 3/5, log SD of tremor visit deviation (Le00) 2/5, log SD of PIGD visit deviation (Le11) 0/5** | **FAIL** |
| Laplace accuracy | maximum \|Laplace − reference\| ≤ 0.05 over 30 patients | **max 0.137.** 20 of 30 patients exceed 0.05; **every error is negative** (Laplace below the reference). Reference Monte Carlo SE ≤ 0.007 | **FAIL** |
| Patient sampler | split-R̂ < 1.01 and bulk ESS ≥ 100 for all 6 quantities, all 30 patients | max R̂ 1.082, min ESS 54. **26 of 30 patients fail**, almost entirely on the four b_i components (R̂ 1.01–1.08). The expected-sum quantities mix well | **FAIL** |
| Fidelity gates G1–G7 | B6 | **not run** (B11 rule) | — |

**The PIGD-zero persistence gate (G4) was not evaluated.** The pilot cannot say whether target 2 would be uninformative under G4. It stops one step earlier, at B11.

## 4. What the failures show

| failure | observed fact | interpretation |
|---|---|---|
| Laplace | Errors from −0.003 to −0.137, all negative. They are larger for patients with more visits (latent dimension 2n up to 24): the mean error is −0.091 for n ≥ 8 (8 patients) and −0.051 for n ≤ 4 (17 patients) | A systematic, dimension-dependent under-estimate of each patient's marginal likelihood. The visit-level deviations ε are weakly identified per visit, so the posterior of θ is non-Gaussian in many directions [H] |
| Recovery | The two visit-deviation SDs are **under-estimated in every dataset**: PIGD by about 12 % (mean z −3.3), tremor by about 5 % (mean z −2.3). β0_P is biased upward (mean z +2.3). All other ψ components are covered in ≥ 4 of 5 | Consistent with the Laplace bias. Because the error grows with the latent variance, the optimiser favours smaller Σ_ε [H]. A contribution from the two-stage scale is also possible: on the stage-1 scale the true pooled PIGD latent has SD 1.23 before rescaling, and stage 1 assumes N(0, 1) [H]. The data at hand cannot separate these two causes |
| Sampler | Patient intercepts and slopes mix slowly under elliptical slice sampling on θ (2n dimensions; the prior is much wider than the posterior in the b directions). ESS is mostly ≥ 100, but R̂ often lies in 1.01–1.08 | 1,000 kept iterations × 4 chains is too short for this sampler at the strict 1.01 threshold. It is a sampler-efficiency failure, not evidence about the model's fit |

**Fitted values (real data; reported, not interpreted, since the checks failed).**
- Mean parameters β = (−0.069, 0.031, −0.034, −0.468, 0.197, −0.090).
- D diagonal: 0.672, 0.014, 0.664, 0.011.
- Σ_ε diagonal: 0.124 and 0.043, with covariance 0.034.

Given the recovery result, the Σ_ε estimates are probably biased downward.

## 5. What was not done

- No fidelity gate.
- No label for observed or simulated data.
- No prevalence or transition statistic or discrepancy.
- No calibration cohort, no positive control and no verdict.
- No amendment, and no repair.

**Integrity:**
- The D1 verdict, NOT REPRODUCED, is untouched, and all seven D1 locks re-verify.
- The `outputs/` checksums match.
- Docs 12 and 13 are unchanged.
- The doc 16 hash matches its frozen record.

## 6. Repair options (for review; none implemented)

Under doc 16 B11, any repair must be a dated amendment made before any evaluation statistic exists. These are candidates, not decisions.

1. **Likelihood approximation.** Replace the Laplace approximation in stage 2 with one that passes the accuracy check. Candidates:
   - an importance-sampled marginal likelihood (a t proposal at the Laplace mode, as already used for the check), with a fixed number of draws and common random numbers across parameter values;
   - or Laplace with a second-order correction.

   Either needs its own accuracy check, and the recovery check would then be repeated. Cost: likely 10–50× the current fit.
2. **Reduce the latent dimension.** Integrate the visit deviation analytically per visit with low-order quadrature (2-D per visit given b), leaving Laplace or quadrature over the 4-dimensional b only. This attacks the dimension-dependent error directly, and it is the most targeted option [H].
3. **Separate the two-stage scale question.** Re-run recovery with the items fixed at truth: does Σ_ε bias persist? This is diagnostic only, costs about 10 min, and determines whether option 1 or 2 is needed.
4. **Sampler.** Either sample (b_i, ε_i) with a sampler preconditioned by the Laplace covariance (for example elliptical slice sampling on a Laplace-whitened prior, or MALA), or lengthen the chains (for example 4,000 kept). Pre-specify the choice; re-check on the same 30 patients.

**Recommendation.** Review these results first. If you approve, write a dated amendment (doc 16 amendment 1) choosing the diagnostic of option 3 first, then option 2 (or 1) and option 4. Re-run the B11 checks, and only then the label-free fidelity gates. Calibration and evaluation remain blocked until a separate review.
