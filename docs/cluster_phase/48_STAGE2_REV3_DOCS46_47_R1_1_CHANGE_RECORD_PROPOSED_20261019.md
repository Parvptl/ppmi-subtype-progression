# Stage 2 M2: change record for docs 46 and 47, revision R1.1 (final pre-freeze review)

**Date:** 2026-10-19.

> **PROPOSED — NOT FROZEN — NO COMPUTATION AUTHORISED.**
> - Docs 46 and 47 remain PROPOSED and not frozen.
> - **Step 6 remains FAILED. D1 remains NOT REPRODUCED.**
> - No frozen record, historical report or prior manifest was modified. Nothing was computed: this was a read-only logical review followed by text edits. Nothing was staged, committed or pushed.

| file | pre-edit SHA-256 | post-edit SHA-256 |
|---|---|---|
| `46_STAGE2_REV3_AMENDMENT_REVISED_R1_PROPOSED_20261019.md` | `2fc537850b0884bee316407d4ca168e85831eb77dfd7c87a3eb78bb1a7bf0665` | `6ebd298accca9f4a171913cb5afca4091ba5cb3e899d23ba378b81068cc3b3dc` |
| `47_STAGE2_REV3_RECOVERY_TEST_PLAN_REVISED_R1_PROPOSED_20261019.md` | `0bcbcd2c408770fd0fdfc73feade76f3fc9cbc0ef424d743589029b98937c6ab` | `d1e6fe3f233b70068eca3e27c738ed78f35fc4b08f5b293d3f3f6cd548a3cba0` |

## 1. Changes

| # | doc § | change | rationale | remaining uncertainty |
|---|---|---|---|---|
| 1 | 47 §5 | Gate status is a **three-way partition** on a set X: **pass** (C0 ∧ C1 ∧ C2), **fail** (C0 ∧ ¬(C1 ∧ C2)), **ne** (¬C0). C0 is now "at most ⌊0.05·\|X\|⌋ undefined intervals" | removes the overlap whereby a gate could fail C0 and also satisfy the literal FAIL rule | none |
| 2 | 47 §7 | **New flag F8:** any undefined interval in a dataset | A \ F then has no undefined intervals, so C0 can fail only on A. This removes the "C0 fails on one set only" ambiguity | F8 datasets count towards the \|F\| ≤ 10 limit |
| 3 | 47 §8 | **Verdict rule rewritten** with explicit precedence: INTEGRITY STOP > INCOMPLETE > INCONCLUSIVE–FLAG EXCESS > FAIL > PASS > INCONCLUSIVE–OTHER (reason codes NE / FLAG-SENSITIVE / GUARD) | the verdicts are mutually exclusive and exhaustive. FAIL needs an **evaluable recovery** gate failing on **both** A and A \ F. Guard and C0 failures never yield FAIL. PASS needs pass on both A and A \ F with \|F\| ≤ 10. Exclusion cannot create a PASS | — |
| 4 | 47 §8 | gate-level decision table, and 12 boundary cases worked through (undefined intervals; C0 on one set; failure only after exclusion; failure only on A; guard-only; \|F\| = 10 and 11; 99 datasets; execution error; Le00_log) | requested completeness check | — |
| 5 | 47 §8 (R8) | \|F\| > 10 → INCONCLUSIVE **even if a recovery gate fails robustly** | with more than 10 % of datasets numerically compromised, neither a pass nor a fail claim is reliable | **reviewer decision R8.** The alternative is to let a robust FAIL override the flag excess |
| 6 | 47 §8 (I); 46 §3.5 N4 | N3 and N4 are evaluated on the Phase 1 regression dataset. Exceeding the frozen 1e-5 tolerance → INTEGRITY STOP, with no re-tuning. N4 is marked **not yet exercised numerically** | prevents adjusting a tolerance after freezing | N4 has never been computed. Its tolerance is a planning value that could prove too tight, which would stop the run |
| 7 | 47 §4 | forecast wording corrected | the bias-0.61 failure probabilities are now exact: about 0.999 (t = 0.30), 0.98 (0.40), 0.86 (0.50), not "any t ≤ 0.50". "Pass probability < 0.0001" is replaced by "0 of 20,000 MC trials (< about 1.5e-4)" | — |
| 8 | 47 §9 | **phased execution.** Phase 1 runs on node002, as the bitwise N11 regression against Step 6 requires that CPU: regeneration, regression, datasets 0–2, the runtime gate; 2 h limit. Phase 2 needs reviewer authorisation: option (a) node002, 27 h; option (b) two normal nodes, 13.5 h each. The cap of 30 node-hours is cumulative and is enforced by per-job SLURM limits (worst case 29) | the earlier text did not say how the runtime gate and the cap work across parallel nodes, or that bitwise regression needs node002 | reviewer decision R4 (a/b). Cross-node last-bit differences are not regression-checked |
| 9 | 47 §10.3 (R9) | **control arm made report-only.** The R1 expectation that S changes true-item estimates by "≪ 1 SE" is withdrawn | the sampling variation of the fitted moments moves S by about the SE of (c, d). D7's re-expression of the D6 fits gave c_P 0.986–1.012 and d_P up to ±0.02 (about 1 SE of β0′). The arm had no defined effect on the verdict | reviewer decision R9 |
| 10 | 46 §3.5 N7, §3.6 | N7 linked to F8; "no calibration claim is made" stated explicitly | consistency with doc 47; scope of claims | — |

## 2. Consistency re-check (read-only; no computation)

| topic | finding |
|---|---|
| 21 recovery gates + 15 guards (36) | consistent across doc 47 §§4, 5, 8, 10, 12 and doc 46 §2.6. Guards block a PASS but never cause a FAIL (§8) |
| item-anchored locations ν_T, ν_P | definition (doc 47 §6) = implementation (`scripts/56` `functionals`: mean over items of a_k(β0_s − mean_j τ_kj)). Recovery gates; invariant under relabelling (unit test ≤ 1e-12; development data ≤ 9e-16). They are the only gates that test the item–latent location (doc 46 §2.6) |
| bias threshold 0.40 | rationale (coverage-loss principle; doc 47 §4) and the OC table are internally consistent. Exact per-gate values re-derived: 1 − 2Φ(−4) = 0.99994; pass at b = 0.5: Φ(−1) = 0.159. Still **reviewer decision R2′** |
| full-chain sandwich | the estimating equations (doc 46 §3.1), block structure (§3.2, matching the D4 implementation `Jm[i2,i1] = C1`, …), checks N1–N11 and failure modes (§3.6) are consistent. **No claim of calibration.** That requires the recovery test |
| seeds, regeneration, budget, nodes | seeds `[42, 49, 1, r]` (unused); planning MC seeds `[42, 50, *]` are distinct. Regeneration on r = 0 and r = 99 in Phase 1. Budget about 24.3 node-hours projected, cap 30 (worst case 29 with the per-job limits). Two-node arrangement specified |
| C0 / C1 / C2 / flags / verdict | C0 (evaluability) → ne; C1 and C2 (evaluable) → pass or fail; flags F1–F8 define A \ F, independently of the truth. The verdict (§8) uses only these, with the stated precedence. Undefined interval: not covered in C1, omitted from the C2 mean, flag F8 |
| Le00_log | a recovery gate with no exemption (doc 46 §5, doc 47 §§5, 8) |

## 3. Unresolved decisions (unchanged list plus R8, R9)

| decision | options | recommendation |
|---|---|---|
| **D1** | S1-b kept or dropped | keep |
| **D2** | primary interval method | full-chain sandwich |
| **D2b** | small-sample correction | none |
| **R2′** | bias threshold t | 0.40 |
| **R2b** | gate structure | 21 recovery gates + 15 guards |
| **R3** | verdict rule | adopt |
| **R4** | Phase 2 execution | (b) two normal nodes |
| **R8** | flag excess overrides FAIL | adopt |
| **R9** | control arm report-only | adopt |
| **D6** | freeze docs 46 and 47 together | — |

## 4. Status

Docs 46 and 47 (R1.1) are internally consistent on every point checked above, and ready for the reviewer's decision.

They are **not frozen**. No driver has been written. No computation is authorised.
