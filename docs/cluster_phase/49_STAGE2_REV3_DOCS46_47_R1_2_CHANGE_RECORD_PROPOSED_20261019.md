# Stage 2 M2: change record for docs 46 and 47, revision R1.2 (approved decisions recorded; doc 45 requirements restored)

**Date:** 2026-10-19.

> **PROPOSED — NOT FROZEN — NO COMPUTATION AUTHORISED.**
> - Docs 46 and 47 remain PROPOSED and not frozen. Freezing them requires the user's separate, explicit authorisation (D6). Writing the driver and executing the test require further separate authorisations (doc 47 §11).
> - **Step 6 remains FAILED. D1 remains NOT REPRODUCED.**
> - This was an editorial and protocol-consistency task. No model was fitted, no data generated, nothing computed, no job submitted. No frozen record, historical report or prior manifest was modified. Nothing was staged, committed or pushed.

| file | pre-edit SHA-256 (R1.1) | post-edit SHA-256 (R1.2) |
|---|---|---|
| `46_STAGE2_REV3_AMENDMENT_REVISED_R1_PROPOSED_20261019.md` | `6ebd298accca9f4a171913cb5afca4091ba5cb3e899d23ba378b81068cc3b3dc` | `e9ad3a837d9ec24496201ded1b6d3f62d31a612ffa695c3799feba1792ed515a` |
| `47_STAGE2_REV3_RECOVERY_TEST_PLAN_REVISED_R1_PROPOSED_20261019.md` | `d1e6fe3f233b70068eca3e27c738ed78f35fc4b08f5b293d3f3f6cd548a3cba0` | `106457578b65546f304dbea80701ecefbd80d4e447a80d1f812d2bbf5753c83b` |

The pre-edit hashes equal the post-edit hashes recorded in doc 48. **Doc 48 is unchanged.**

## 1. The eleven approved decisions, as transcribed

Source: the user's *Final Decision Record R1.1*, 2026-10-19. **No decision's substance was changed.**

| decision | approved resolution (decision record) | where recorded | transcription |
|---|---|---|---|
| D1 | retain one S1-b iteration as a shape correction | doc 46 §§4, 7; doc 47 §2 | verbatim in substance |
| D2 | full-chain patient-clustered sandwich, primary, with the specified numerical checks | doc 46 §§3, 7; doc 47 §2 | yes |
| D2b | no post-hoc small-sample correction; calibration evaluated by the pre-registered test | doc 46 §§3, 3.6, 7; doc 47 §2 | yes |
| D3/D4 | adopt the Rev 3 convention for future estimates and the revised recovery definition; preserve historical verdicts without re-scoring | doc 46 §§0, 7 | yes. §0 rewritten from "would amend" to the approved future scope, effective on freezing and for post-freeze estimates only |
| R2′ | bias threshold 0.40 | doc 47 §§4, 5, 14 | yes. "pending R2′" and the alternative-choice line removed; the 0.30/0.50 columns kept as planning comparisons only |
| R2b | 21 recovery gates and 15 guards | doc 47 §§5, 14 | yes |
| R3 | revised mutually exclusive verdict rule (precedence, definedness, no pass by exclusion) | doc 47 §§8, 14 | yes. The rule is unchanged except for the restored integrity conditions (§2 below) |
| R4 | Phase 2 on two normal nodes, with the safeguards; Phase 1 on node002 | doc 47 §§9, 14 | yes. Option (a) (node002 alone) removed |
| R8 | more than 10 flagged datasets → INCONCLUSIVE, even if a recovery gate meets the robust-failure rule | doc 47 §§8, 14 | yes |
| R9 | control arm report-only | doc 47 §§10, 14 | yes |
| D6 | freeze docs 46 and 47 together, after the final text review and under separate explicit authorisation | doc 46 §7; doc 47 §11 step A | yes |

**Also recorded, without new decisions:**
- **R1** (100 fresh datasets): from the decision record's §2 "protocol requirements retained".
- **R5** (D3 ≥ 4/5 rule as a comparator): part of the approved D4 text.
- **R6** (control-arm size 10): the plan specification; no alternative is open.
- **R7**: a cross-reference to doc 46 §7.
- **D5** (calibration-stage draws): stated as **not decided, deferred** while calibration is blocked.

## 2. Requirements restored from doc 45

The full comparison of doc 45 with doc 47 (R1.1) found the following omissions. Each is now in doc 47.

| doc 45 | requirement | restored in doc 47 |
|---|---|---|
| §8 step 4 | **old-dataset-0 regression** on `[42, 45, 1, 0]`: **R-a** Step 6 ψ̂⁽¹⁾ and ψ̂⁽²⁾ **bitwise**; **R-b** D7's S(ψ̂⁽²⁾) **≤ 1e-12**; **R-c** D4 SEs **≤ 1e-6** (= N11, retained). Plus **R-d**: N3/N4 on that dataset (retained from R1.1) | §8 condition (I), with the exact reference files. **Any failure is an INTEGRITY STOP, and is not followed by any tolerance change, retuning or code change within the test** |
| §8 steps 1–7 | the execution sequence | **new §11**, steps A–H: protocol freeze → driver implementation → unit tests → labelled smoke run and code freeze → preflight → regression checks, datasets 0–2 and runtime gate (Phase 1) → Phase 2 → verdict and report. Each step has its authorisation (§3 below) |
| §2 | frozen-settings table: stage-1 and stage-2 settings, S1-b definition, "no other random numbers", code frozen by SHA-256 with "a defect stops the run" | §2 (table restored, adapted to docs 46/47 and the approved decisions) |
| §3 | regeneration method (SHA-256 of the flattened responses) | §3 |
| §6.1 | truth-swap test procedure (S(ψ̂) bit-identical with ψ_true replaced) | §10.1, and now also an integrity condition in §8 (I) and part of Phase 1 (§§9, 11) |
| §7 | explicit list of transformation identities; item-interval coverage from V_φφ; Le00_log secondary-analysis contents | §7 |
| §9 | per-component budget table with measured bases | §9 |
| §10 | outputs (directory, per-dataset JSON, tables, figures, manifest) | **new §12** (adds the gate statuses on A and A \ F, and the verdict reason codes) |
| §11 | fuller pass/fail statement | §13 (updated to the 21 + 15 gate structure; INCONCLUSIVE and INCOMPLETE added) |

**Retained unchanged:**
- Phase 1 on node002;
- the two-node Phase 2 with fixed seed assignments (2a: 3–51 plus control 0–4; 2b: 52–99 plus control 5–9);
- seeds `[42, 49, 1, r]`;
- the runtime gate (each stage-2 fit ≤ 60 min, projected ≤ 28 node-hours);
- the 30 node-hour cumulative cap (worst case 29);
- N11 and N3/N4.

## 3. Execution sequence and authorisations (doc 47 §11)

| step | authorisation |
|---|---|
| A — protocol freeze (docs 46 and 47 together) | separate explicit freeze authorisation |
| B, C — driver and unit tests | separate authorisation to write the driver |
| D — labelled smoke run (NOT A RESULT) and code freeze | separate authorisation to execute. The smoke run is the first job under it |
| E, F — Phase 1: preflight, regeneration, regression, truth-swap, datasets 0–2, runtime gate | the execution authorisation |
| G — Phase 2 | separate authorisation to continue, after review of Phase 1 (retained from R1.1 §9) |
| H — verdict and report | stop for review |

## 4. Minor corrections

| item | change |
|---|---|
| flags vs truth (doc 47 §8) | "computed before any comparison with the truth" replaced by a precise statement. F1–F4 and F6–F8 never read the truth. F5's identity S(ψ_true) = ψ_true does read ψ_true, involves no estimate, and is the same for every dataset. No flag depends on any comparison between an estimate and the truth |
| Monte Carlo seeds (doc 47 §4) | 0.893 (calibrated, t = 0.30, and every *coverage only / t = 0.30–0.50* column) comes from script 57, seed `[42, 50, 1]`. 0.899, the independence-approximation column and the doc 45 verification figures (0.9726, 0.927) come from script 56, seed `[42, 50, 0]`. The difference is about 2 MC SE |
| decision table (doc 47 §14) | renamed from §12 and reordered R1, R2′, R2b, R3, R4, R5, R6, R7, R8, R9, with an "approved resolution" column |
| doc 46 §3.4 | "full chain (proposed)" → "full chain (primary method, D2 approved)" |
| headings | doc 47 §§2, 3, 7, 9 headings updated to reflect restored content. Doc 46 §§3, 4 headings record D1, D2 and D2b |

## 5. Clarifications introduced (flagged for the reviewer's attention)

These follow from restoring doc 45 into the approved structure. They are not new decisions, but they are wording that did not previously exist:

1. **Smoke-run data.** The smoke run uses 60 patients of the **old development dataset** `[42, 45, 1, 0]`, so that **no fresh recovery dataset is generated before Phase 1** (step E). Doc 45 said only "60 patients, not a result".
2. **Code freeze timing.** The code hashes are fixed at a **code freeze** after the smoke run (step D), because the protocol freeze (step A) precedes driver implementation. Doc 45 listed "code by SHA-256" among the items frozen before execution without specifying when.
3. **Truth-swap test as an integrity condition.** Doc 45 §6.1 stated the test, but not its consequence. R1.1 §10 called a failure an INTEGRITY STOP without listing it in §8 (I). It is now listed, and assigned to Phase 1 (fresh dataset 0).

## 6. Read-only verification after editing

| check | result |
|---|---|
| all eleven decisions present with their approved resolutions (string checks on the decision-table rows of doc 46 §7 and doc 47 §14) | all present |
| no remaining "pending", "(proposed)" decision labels, "option (a)", "[D]" or "recommend" wording in docs 46 and 47 | none, except the non-decision phrase "pending the doc 47 result" in doc 46 §0 |
| restored items present (bitwise, ≤ 1e-12, ≤ 1e-6, R-a–R-d, smoke, freeze, driver, truth-swap, regeneration SHA-256, Outputs §12, two-node Phase 2, the 29 node-hour worst case, MC seeds) | present |
| cross-references | doc 46 → doc 47 §§6, 7, 8, 14; doc 47 internal §§2–14 and → doc 46 §§2, 2.6, 3, 3.5, 5, 7: all resolve to the intended sections |
| markdown tables | every row closed; no column-count mismatches |
| verdict rule | precedence rows 1–6 unchanged and exhaustive; (I) extended only by the restored regression and truth-swap checks; FAIL still requires an evaluable recovery gate failing on both A and A \ F; guards and C0 never yield FAIL; PASS requires A (exclusion cannot create a PASS); \|F\| > 10 → INCONCLUSIVE (R8); Le00_log has no exemption |
| historical verdicts | "Step 6 remains FAILED" and "D1 remains NOT REPRODUCED" in both docs; no re-scoring language |
| status banners | both docs still "PROPOSED — NOT FROZEN — NO COMPUTATION AUTHORISED" |

## 7. Remaining technical uncertainties (unchanged in substance)

1. **Sandwich calibration and finite-sample behaviour are unknown** (198 stacked parameters, 934 patients). They are to be judged only by the test. No calibration claim is made.
2. **N4 has never been computed numerically.** Its frozen 1e-5 tolerance could prove too tight, which would end Phase 1 as INTEGRITY STOP. It would not be re-tuned.
3. **Bitwise regression R-a requires node002, and identical code paths for stages 1–4** (the Step 6 driver's computations). Any unintended difference in the new driver's stage 1–4 path would cause an INTEGRITY STOP.
4. **Cross-node bitwise equivalence for Phase 2 is not guaranteed.** Last-bit differences are expected to be negligible but are not checked.
5. **The test may legitimately FAIL.** The development means (n = 5) are compatible with zero bias but also with biases (for example Le00_log −0.61 SE) that the 0.40 rule would detect.
6. **The flag limit (R8) could make a numerically troubled run INCONCLUSIVE** rather than FAIL. This is an approved, deliberate choice.

## 8. Status and blockers

Docs 46 and 47 (R1.2) record the approved decisions and contain the restored doc 45 requirements. They are **ready for the user's final review**.

**Remaining blockers are authorisations only:**
- (1) the separate explicit authorisation to freeze docs 46 and 47 together;
- (2) the separate authorisation to write the driver;
- (3) the separate authorisation to execute (smoke run and Phase 1);
- (4) the separate authorisation to continue to Phase 2.
