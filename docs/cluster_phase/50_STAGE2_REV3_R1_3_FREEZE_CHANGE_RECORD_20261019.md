# Stage 2 M2 Rev 3: change record R1.3 and freeze of docs 46 and 47

**Date:** 2026-10-19.

**Authority:** the user's *Final Execution Directive*, 2026-10-19. It authorised:
- resolving review findings B1–B7, S1–S7 and M1–M4 in one bounded pass;
- the read-only pre-freeze computation;
- freezing;
- the driver, its tests, the smoke run, Phase 1 and the conditional Phase 2.

**Historical verdicts are unchanged:** Step 6 = **FAILED**; D1 = **NOT REPRODUCED**.

## 1. Hashes and locks

| file | R1.2 (doc 49) | R1.3 (frozen) | lock and read-only copy |
|---|---|---|---|
| doc 46 | `e9ad3a837d9e…` | `e2b40c11ad4b66767219c50864e378ed799d2f82d9167c983cc38ebde2d29c91` | `_frozen/46_STAGE2_REV3_AMENDMENT.{sha256,as_frozen_20261019.md}` |
| doc 47 | `106457578b65…` | `361441592a3a4ef49350add297ceb76536bbb767e3ddbb5d4eb826278829fcfa` | `_frozen/47_STAGE2_REV3_RECOVERY_TEST_PLAN.{sha256,as_frozen_20261019.md}` |

All 10 locks under `docs/cluster_phase/_frozen/` verify. No earlier frozen record, report or manifest was modified.

## 2. Pre-freeze read-only computation (B5)

`scripts/58_m2_rev3_prefreeze_checks.py` → `experiments/20261019T172456Z_stage2_rev3_prefreeze_checks/summary.json`. It reads the saved D4 and Step 6 matrices, and regenerates the five old datasets in memory only to evaluate item gradients. Nothing was fitted.

| check | result | decision |
|---|---|---|
| N6: cond₂ of the 198 × 198 stacked Jacobian | 1.7e3–2.4e3 unscaled (about 54 Jacobi-scaled) on all 5 datasets | the threshold of 1e12 is **retained, unchanged** (more than 8 orders of magnitude of margin) |
| N4: S1-b item-stage relabelling identity | **without** the first-order gradient term: 5e-5 to 5e-4 in the c-directions, which **would fail** the frozen 1e-5 tolerance. **With** it: 5e-9 to 4.6e-7 | the identity's mathematical definition is corrected to include −m_s ⊙ g (doc 46 §3.5). The tolerance is unchanged. The same exact form is used for N3 |

## 3. Resolution of the review findings

| finding | resolution (frozen location) |
|---|---|
| **B1** runtime gate | exact formula P = U₁ + 97·mean(t_r)/3600 + 10·t_c/3600. Pass iff every Phase 1 stage-2 fit ≤ 3600 s, P ≤ 28, U_smoke + P ≤ 30, and the per-job Phase 2 projection ≤ L₂. Missing or invalid timing fails the gate (doc 47 §9) |
| **B2** regression path | `scripts/52` `pipeline_dataset` with an Ops object making exactly the `scripts/52` calls; the `scripts/54` sandwich helpers in `run_d4` order. R-c compares **raw** full-chain SEs with D4 `se_psi2.full_chain` (doc 47 §§2, 8, 16) |
| **B3** sandwich contexts | H₁ and H₂: `fd_hessian` on `Objective` (optimiser context, fit state), as in Step 6; `run_d4` does not compute H. C₁ and C₂: reported context, cold base, warm FD, 64-patient chunks. A penalised FD evaluation or a C failure gives F6; an unusable V gives F8 (doc 46 §3.2) |
| **B4** flag scope | F1 both stage-2 fits; F2 all four item fits; F3 and F4 at ψ̂⁽¹⁾ and ψ̂⁽²⁾; F5–F8 defined exactly. The control arm sets no flags and cannot cause INCOMPLETE (doc 47 §§7, 8, 10) |
| **B5** N3/N4/N6 | exact identities with the gradient term and stated denominators; N6 retained (doc 46 §3.5; §2 above) |
| **B6** Phase 2 ordering | main datasets first in seed order, then the control arm; atomic, never-overwrite per-dataset files; refuse to start if an assigned file exists (doc 47 §9) |
| **B7** smoke budget | 0.5 h limit per smoke run; at most two repeats; counted in the 30 node-hour cap; L₂ = min(13.5 h, (30 − U)/2) (doc 47 §§9, 11) |
| **S1** failure labels | bias-type, dispersion-type (χ²(n − 1) lower bound of SD(z) > 1), mixed, unclassified. Explanatory only; secondary-interval coverage reported (doc 47 §5) |
| **S2** doc 45 disposition | complete table, doc 47 §15 |
| **S3** z and coverage | exact formulas; \|z\| ≤ 1.96 inclusive; undefined → F8, not covered, omitted from the mean (doc 47 §5) |
| **S4** hardware | normal partition, exclusive, ≥ 32 cores; node, CPU and versions recorded; completion not guaranteed (doc 47 §9) |
| **S5** D3 comparator | doc 16 D3 literally (raw ψ̂⁽²⁾, fixed-item H₂⁻¹ intervals); Rev 3-gate counts labelled separately (doc 47 §5) |
| **S6** test inputs | truth-swap ψ_true + 1; relabelling c = (1.13, 0.91), d = (−0.07, 0.05); rank tolerance 1e-8·s_max (doc 47 §§7, 10) |
| **S7** completion | `main/dataset_r{rr}.json` with `"complete": true`, holding all 36 gates or explicit F8 records (doc 47 §8) |
| **M1–M4** | change-log rows labelled by revision; tag legend in doc 46; §14 included; the impossible F1 "time" trigger removed |

**No approved decision, threshold, tolerance, seed, sample size, gate or verdict rule was changed.** The N3/N4 correction restores the mathematically exact identity that the frozen tolerance is meant to test.

## 4. Remaining technical uncertainties

- Sandwich calibration is unknown, to be judged by the test.
- Bitwise regression (R-a) needs node002 and identical code paths.
- Phase 2 cross-node last-bit differences are unchecked.
- The test may legitimately fail.
