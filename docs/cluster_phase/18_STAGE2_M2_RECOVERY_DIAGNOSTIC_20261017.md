# Stage 2 M2: recovery diagnostic with item parameters fixed at truth

**Date:** 2026-10-17.
**Type:** diagnostic only (doc 17 §6, option 3).
- This is not a repair, not a pilot re-run, and not the B11 recovery check. B11 is defined on the full two-stage pipeline, and that check stays **failed**.
- No gate, label, prevalence, transition, calibration or verdict statistic was computed.
- No observed item data enter any fit here; the real cohort supplies only the visit design (times and treatment indicator).

**Frozen specification:** `16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.md`, SHA-256 `c2eace18…`, verified unchanged. No acceptance criterion was changed.

**Runs:**
- `experiments/20261016T203131Z_stage2_m2_recovery_fixed_items_ds0/`: dataset 0 only, the smallest valid run.
- `experiments/20261016T203341Z_stage2_m2_recovery_fixed_items_ds0to4/`: all 5 datasets.

**Code:** `scripts/46_m2_recovery_fixed_items.py` (new), `tests/test_m2_recovery_diag.py` (new, 3 tests). No existing file changed.

---

## 1. What the diagnostic distinguishes

The pilot's recovery under-covered three parameters (doc 17 §3–4):
- the two visit-deviation SDs, which were biased low;
- β0_P, which was biased high.

There are two candidate sources:

| source | removed by fixing items at truth? |
|---|---|
| **Stage 1:** item parameters estimated under the θ ~ N(0, 1) working assumption (the two-stage scale) | **yes**. Stage 2 then sees the true items |
| **Stage 2:** the Laplace marginal likelihood over the 2n-dimensional θ_i (doc 17 found its error negative and growing with visit count) | **no**. Stage 2 is unchanged |

How to read the result:
- **Bias disappears** with true items: stage 1 is responsible.
- **Bias persists unchanged:** stage 2 is responsible.
- **Bias partly persists:** both stages contribute.

## 2. What changed (code and configuration)

**Nothing in the model, truth, seeds or criteria.** The new script imports the pilot's own functions from `scripts/45_m2_pilot.py`: `load_cohort`, `truth_on_stage1_scale`, `simulate_groups`, `regroup`, `flatten`, `fit_items_multistart`. It then calls the unchanged `m2_grm.fit` and `m2_grm.hessian`. The **only** difference from the pilot's recovery is that stage 2 receives the generating item parameters instead of the stage-1 estimates.

**Safeguards in the script:**
- It refuses to run unless the specification hash equals the frozen record and every pilot code file equals the pilot's recorded SHA-256. All 6 were identical.
- It asserts that the truth ψ equals the pilot's saved `psi_true` exactly.
- **Identity check:** it re-fits stage 1 on dataset 0 and requires the pilot's saved item-recovery summary exactly. Result: identical (a_maxerr_T 0.134202…, a_maxerr_P 0.322457…). The synthetic datasets are therefore the pilot's.

**Tests** (`tests/test_m2_recovery_diag.py`):
- The reused truth is on the stage-1 scale: the pooled visit-level θ^T and θ^P have mean 0 and variance 1 at the design.
- The rescaling preserves every item category probability.
- The script loads the pilot's functions.

Result: 3 passed. With `test_m2_grm.py`, 13 passed.

## 3. Cost estimate before running, and actual cost

- **Estimate.** From the pilot's recovery, 647 s for 5 datasets including two stage-1 fits each. Skipping stage 1 gives about 70–130 s per dataset, plus about 40 s for the identity re-fit. Total under 15 min on the login node at low priority. No SLURM job.
- **Actual.**
  - Dataset 0 alone: 124 s.
  - Five datasets: 391 s (62–123 s each; dataset 0 includes the identity re-fit).
- **Isolation.** Each run has its own new directory under `experiments/`. The pilot directory was only read.

## 4. Results

**Fits.** All 10 fits converged, ‖∇‖ ≤ 0.0013, with no bound hits and a positive-definite Hessian.

**Coverage and mean z per parameter** (5 datasets; z = (estimate − truth)/SE; pass means covered in ≥ 4 of 5, as in doc 16 D3):

| parameter | coverage, pilot (estimated items) | coverage, fixed items | mean z, pilot | mean z, fixed |
|---|---|---|---|---|
| β0_T | 4 | 5 | −0.13 | 0.05 |
| β1_T | 5 | 4 | 0.02 | 0.23 |
| γ_T | 5 | 5 | 0.38 | 0.45 |
| **β0_P** | **3** | **5** | **+2.35** | **0.05** |
| β1_P | 5 | 5 | −0.64 | 0.48 |
| γ_P | 4 | 4 | −0.57 | −0.63 |
| L00 (log) | 5 | 4 | −0.52 | 0.29 |
| L10 | 4 | 4 | 0.22 | 0.29 |
| L11 (log) | 5 | 5 | −0.47 | −0.18 |
| L20 | 5 | 5 | 0.19 | 0.27 |
| L21 | 4 | 4 | −0.11 | −0.10 |
| L22 (log) | 5 | 5 | −0.85 | −0.12 |
| L30 | 5 | 5 | −0.16 | −0.12 |
| L31 | 5 | 5 | −0.44 | −0.43 |
| L32 | 5 | 5 | −0.92 | −0.50 |
| L33 (log) | 5 | 5 | −1.21 | 0.06 |
| **Le00 (log), tremor visit SD** | **2** | **4** | **−2.26** | **−1.03** |
| Le10 | 5 | 5 | 0.24 | 0.37 |
| **Le11 (log), PIGD visit SD** | **0** | **3** | **−3.34** | **−1.78** |
| **parameters passing** | 16 / 19 | **18 / 19** | mean \|z\| 1.11 | mean \|z\| 0.78 |

**Visit-level SDs on the natural scale:**

| | truth | pilot estimates (mean relative bias) | fixed-item estimates (mean relative bias) | z, fixed items, per dataset |
|---|---|---|---|---|
| tremor | 0.4783 | 0.434–0.475 (−5.4 %) | 0.453–0.484 (**−2.5 %**) | −1.42, −2.19, −0.13, +0.51, −1.90 |
| PIGD | 0.4033 | 0.346–0.375 (−11.2 %) | 0.372–0.385 (**−6.2 %**) | −2.17, −1.38, −1.89, −1.38, −2.11 |

## 5. Reading

**Observed facts:**
- **β0_P's bias is entirely a stage-1 effect.** It goes from mean z +2.35 (3 of 5) to 0.05 (5 of 5).
- **The visit-SD bias roughly halves with true items but does not disappear:**
  - PIGD: −11.2 % → −6.2 %. The z value is negative in 5 of 5 datasets, and coverage is 3 of 5, still below the threshold.
  - Tremor: −5.4 % → −2.5 %. z is negative in 4 of 5; coverage is 4 of 5, now passing.
- **Even with perfect item parameters, stage 2 fails the frozen recovery criterion,** on the PIGD visit SD.

**Interpretation [H].**
- **Both stages contribute.** Stage 1 accounts for the β0_P bias and about half of the visit-SD under-estimation. Stage 2 accounts for the rest.
- **The stage-2 residual matches doc 17's Laplace finding.** The Laplace log-likelihood is too low, increasingly so in more latent dimensions, and the bias favours smaller Σ_ε. This is consistent but **not proven**: maximum-likelihood bias of variance components is another, probably small, contributor at this sample size. The diagnostic does not test the Laplace error directly.
- **A stage-2 repair alone would not fix the full pipeline.** The stage-1 contribution would remain. Doc 16 B11 requires the *full* pipeline to pass.

## 6. Proposed dated amendment (DRAFT for review; not in force; nothing implemented)

**Doc 16 Amendment 1, proposed 2026-10-17.** If you approve it, it would be written as a separate dated file, and doc 16 stays frozen. The B11 acceptance criteria and every evaluation definition are unchanged.

1. **Stage-2 likelihood.** Replace the 2n-dimensional Laplace with an approximation that removes ε_i analytically per visit:
   - Given b_i, visits are independent. Each visit's likelihood ∫ P(items | θ = μ + Zb + ε) N(ε; 0, Σ_ε) dε is a 2-D integral, computed by Gauss–Hermite quadrature on the Cholesky of Σ_ε (Q × Q nodes, Q fixed in the amendment, for example 10).
   - Integrate b_i (4-D) by Laplace, or by adaptive Gauss–Hermite if affordable.
   - The Laplace step then acts in 4 dimensions on an informative posterior, rather than in 2n dimensions on weakly identified deviations.
   - **Pre-requisites, before any real-data use:**
     - the doc 16 D3 Laplace-accuracy check (≤ 0.05), now against the same reference;
     - **this fixed-item recovery** (Le11 must reach ≥ 4 of 5);
     - a cost estimate.
2. **Stage-1 contribution.** Choose one, fixed in the amendment.
   - **(a) One refit iteration:** re-estimate the item parameters by marginal likelihood over the *fitted* stage-2 population latent distribution (a normal mixture over visits), then re-fit stage 2.
   - **(b) Joint estimation** of items and longitudinal parameters under the 1(a) likelihood.

   (a) is cheaper. (b) removes the two-stage assumption but costs much more. Either is re-tested by the full-pipeline recovery (doc 16 D3, unchanged criterion).
3. **Patient sampler.** Precondition the elliptical slice sampler with the Laplace covariance (run it on a whitened parameterisation), or sample (b_i, ε_i) with this amendment's per-visit structure. Re-check on the same 30 patients with the unchanged R̂ < 1.01 and ESS ≥ 100.
4. **Order.** Implement and unit-test; run the fixed-item recovery for item 1; then the full-pipeline recovery, Laplace and sampler checks. **Only if all pass**, run the label-free fidelity gates G1–G7. Calibration and evaluation stay blocked until a separate review.

**Optional cheaper step before writing the amendment** (not run): on dataset 0 with items fixed, profile the log-likelihood over the PIGD visit SD with the other parameters held at their estimates. Compare the Laplace profile with an importance-sampled profile (the D3 reference method) at about 7 grid points. If the importance-sampled maximum sits at the truth while the Laplace maximum is low, this directly confirms the stage-2 mechanism. Estimated cost: tens of minutes of CPU, on the login node or a short SLURM job.

## 7. Integrity

- No existing file was modified; the frozen spec hash matches.
- The pilot directory was read only.
- D1 frozen files were untouched, and the D1 verdict remains NOT REPRODUCED.
- Nothing was staged, committed or pushed.
