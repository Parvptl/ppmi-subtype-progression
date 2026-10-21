# Stage 2 M2 Rev 3: review of the R1.3 changes against the frozen text

**Date:** 2026-10-20 (review 09:30Z–09:40Z).

> **REVIEW — AUTHORISES NOTHING, AMENDS NOTHING, FREEZES NOTHING.**
> - Scope: audit doc 50's change record against the frozen text of docs 46 (`e2b40c11…`) and 47 (`36144159…`), and against the superseded R1.2 text.
> - **Nothing was modified.** No model fitted, no data generated, no job submitted or cancelled, nothing staged, committed or pushed. The running test was not touched.
> - **Step 6 remains FAILED. D1 remains NOT REPRODUCED.**
> - **Verdict: the R1.3 changes are faithful.** Every frozen criterion, threshold, tolerance, seed, sample size, gate and verdict condition is unchanged. The changes either correct a mathematical error, pin a previously underspecified computation, or tighten a constraint. **One safeguard was genuinely relaxed — the step-G human review — under the user's own directive (F-A).** Three minor documentation defects are recorded (F-B, F-F, F-G).

**Method and its limit.** The R1.2 text was overwritten in place and no copy was archived (`experiments/_doc_versions/` holds none), so the diff for **doc 47 is complete** — its full R1.2 text was read into this session before the edit — while for **doc 46 only the header, §§0–2.6 and the §3.5 table** were read at R1.2. Doc 46's §§3.1–3.4, 3.6 and 4–8 are therefore audited against its own change log, doc 50 and internal consistency, not against the R1.2 words. This is stated so the audit's reach is not overclaimed.

---

## 1. Verified unchanged (doc 47, full R1.2 comparison)

| frozen element | R1.2 | R1.3 | verdict |
|---|---|---|---|
| §2 model, truth, design | M2, 11 T + 5 P items, 19 ψ; a-priori `TRUTH_ITEMS`/`TRUTH_LONG`; 934-patient real design | identical | **verbatim** |
| §2 **seeds** | `[42, 49, 1, r]`, r = 0…99; namespace unused elsewhere; no other random numbers; planning seeds `[42, 50, 0/1]` | identical | **verbatim** |
| §2 stage 1 / stage 2 / S1-b | GH 61, maxiter 3000; Laplace Q = 15, P1/P2, 1e-12 line search, D3 bounds, maxiter 1000, default ψ, cold modes, 32 workers; one S1-b iteration per doc 37 §6(3) | identical | **verbatim** |
| §2 transformation, intervals, FD steps, evaluation | identity tolerances 1e-8; primary full-chain sandwich, Wald 95 %, no post-hoc correction; h = 1e-4 / 1e-6, sensitivity 2h; G = 5 | identical | **verbatim** |
| §3 datasets | R = 100, `simulate_groups`, regeneration check on r = 0 and r = 99, all 100 in seed order, none skipped | identical | **verbatim** |
| §5 **C1** | ≥ ⌈0.88·\|X\|⌉, = 88 at \|X\| = 100 | identical | **verbatim** |
| §5 **C2** | ≤ **0.40** (R2′) | identical | **verbatim** |
| §5 **C0** | ≤ ⌊0.05·\|X\|⌋ | identical | **verbatim** |
| §5 gate status, gate sets | pass/fail/ne; 21 recovery (19 + ν_T, ν_P) and 15 guards; guards never create a PASS; Le00_log no exemption | identical | **verbatim** |
| §8 **R-a / R-b / R-c / R-d** | bitwise; ≤ 1e-12 abs; ≤ 1e-6 rel; N3/N4 ≤ 1e-5 — same reference files and keys | identical | **verbatim** |
| §8 (I), (Fl), (RF), (AP) | integrity as listed; \|F\| > 10; robust failure on both sets; all-pass on both sets | identical | **verbatim** |
| §8 verdict precedence | 6 rows, INTEGRITY STOP → INCOMPLETE → FLAG EXCESS → FAIL → PASS → INCONCLUSIVE-OTHER | identical | **verbatim** |
| §8 gate-level table, 12 boundary cases, properties | all rows | identical | **verbatim** |
| §9 budget table and totals | per-component minutes, 13.8 min/dataset, 24.3 node-hours projected | identical | **verbatim** |
| §9 hard cap | **30 node-hours** | identical (now explicitly sacct-based and smoke-inclusive) | **unchanged value** |
| §9 runtime-gate thresholds | each stage-2 fit **≤ 60 min**; projected total **≤ 28** node-hours | **≤ 3600 s**; **P ≤ 28** | **unchanged values** |
| §9 Phase 1 / Phase 2 placement and split | node002, 2 h; two normal nodes; 2a = 3–51 + control 0–4, 2b = 52–99 + control 5–9 | identical | **verbatim** |
| §14 decisions | R1, R2′, R2b, R3, R4, R5, R6, R8, R9 | identical | **verbatim** |

**Doc 46 §3.5 tolerances**, against the R1.2 table read before the edit — **all eleven identical**: N1 1e-8·max(1,‖U‖); N3 1e-5; N4 1e-5; N5 1e-4·max\|X\|; N6 1e12; N7 −1e-10·max; N8 5 %; N10 1e-6 and rank 15; N11 1e-6. Doc 46 §7's decisions (D1, D2, D2b, D3, D4, D6; D5 deferred) are unchanged, and §0 still preserves Step 6 = FAILED and D1 = NOT REPRODUCED without re-scoring.

**Doc 50's central claim therefore holds:** no approved decision, threshold, tolerance, seed, sample size, gate or verdict rule was changed. See F-A and F-B for the two places where its blanket wording is broader than the change it covers.

## 2. The N3/N4 correction: audited and independently confirmed

This is the substantive change in R1.3, and it is correct.

**The mathematics.** Doc 46 §3.5 derives, from the invariance of each stage's objective under the relabelling T, the second-order identity

  H v_ψ + C v_φ − m_s ⊙ g = 0 (c-direction),  H v_ψ + C v_φ = 0 (d-direction),

the extra term arising because the Jacobian of T depends on the group parameter in the scaling direction but not in the shift direction. **I verified the m_s coordinate lists independently against the convention of doc 46 §2**, and they are exactly right:

| object | under θ′ = (θ − d)/c | c-dependent Jacobian? |
|---|---|---|
| β0_s, β1_s, γ_s | ÷ c (β0 also − d) | **yes** → in m_s |
| log-diagonal Cholesky (L00, L11, L22, L33, Le00, Le11) | log c **shift** | no |
| off-diagonal Cholesky in subscore s's row (L10 for T; L20, L21, L30, L31, L32, Le10 for P) | ÷ c | **yes** → in m_s |
| items: log a | + log c shift | no |
| items: first threshold γ1 | (τ₁ − d)/c | **yes** → in m_s |
| items: log threshold increments | − log c shift | no |

This reproduces doc 46's m_T = {β0_T, β1_T, γ_T, L10} and m_P = {β0_P, β1_P, γ_P, L20, L21, L30, L31, L32, Le10}, and its "packed items: the first-threshold coordinates γ1", exactly. Le10 belongs to m_P and not m_T because under Σ_ε′ = C₂⁻¹Σ_εC₂⁻¹ the residual Cholesky's second row scales by 1/c_P alone. The term is part of the identity, not a tolerance, since optimisers stop at small non-zero g — as the document states.

**The evidence.** Pre-freeze (`scripts/58_…`, 5 development datasets × 4 directions):

| N4 form | max relative residual | instances over 1e-5 |
|---|---|---|
| **without** the gradient term (the R1.2 wording) | 5.23e-4 | **10 of 20**, all in the c-directions |
| **with** it (R1.3) | 4.55e-7 | 0 of 20 |

**Independent confirmation from the live run.** Phase 1's regression record gives `R_d_N4_max = 9.373562088266867e-08`. That is, to all printed digits, the pre-freeze value for dataset 0's P_c direction with the gradient term. The same dataset's without-term values were 3.27e-4 (T_c) and 5.23e-5 (P_c), both above 1e-5. **Under R1.2's wording, R-d would have failed and the run would have ended as an INTEGRITY STOP at 03:51 today**, with no tolerance change permitted. The correction was necessary, and the driver reproduces the pre-freeze computation exactly.

N3 is corrected to the same exact form though it already held (≤ 6e-7 without the term; 2.75e-7 in Phase 1). N6's 1e12 threshold was confirmed against measured condition numbers of 1.7e3–2.4e3 and retained unchanged. **No tolerance moved to accommodate any observation** — the identity was corrected and the thresholds left alone, which is the right way round.

## 3. The other R1.3 changes, classified

All are disclosed in doc 50 §3 and all are specifications or tightenings:

| change | nature |
|---|---|
| §3.2 computation paths of H₁, H₂, C₁, C₂, A₁, A₃, E₃ pinned to named functions and contexts (B3) | **specification** (see F-C) |
| §2 `code` row now names `scripts/59_…` and enumerates every reused function of `scripts/51, 52, 54, 55, 56, 45` (B2) | **tightening** |
| §16 per-dataset computation sequence, and R-c explicitly on **raw** √diag V_{ψ₂ψ₂} before S (B2) | **specification**, and a latent-ambiguity fix: D4's `se_psi2.full_chain` is a raw-scale quantity, so comparing re-expressed SEs would have been wrong |
| §5 z and coverage formulas; \|z\| ≤ 1.96 inclusive, literal 1.96 (S3) | **specification** (the 1.96 Wald form was already frozen in §2) |
| §7 flags: N1 added to F6, N10 to F5; F3's "bound hit" pinned to within 1e-6 of a D3 bound; F5's truth identity computed once per job and applied to all its datasets | **tightening** (more conditions can flag) |
| §7 F1's "time" trigger removed (M4) | removal of an **unreachable** trigger — no per-fit time cap exists |
| §9 smoke run bounded: 0.5 h, at most two repeats, counted in the cap (B7) | **new constraint** where R1.2 had none |
| §9 runtime gate given the exact formula P = U₁ + 97·mean(t_r)/3600 + 10·t_c/3600, plus (iii) U_smoke + P ≤ 30, (iv) per-job ≤ L₂, and "missing or invalid timing fails the gate" (B1) | **specification at unchanged thresholds**; arithmetic checked: 97 + 3 = 100 main, 49/48 job split with 49 used as the conservative maximum, 10 controls |
| §9 L₂ = min(13.5 h, (30 − U)/2) | **tightening** of the 13.5 h limit |
| §9 Phase 2 ordering (main first, in seed order), atomic never-overwrite writes, refusal to start over an existing file, eligibility ≥ 32 cores exclusive (B6, S4) | **tightening** |
| §10 truth-swap pinned to ψ_true + 1, with SE_S also required bit-identical (S6) | **tightening** (R1.2 said "an arbitrary vector", S(ψ̂) only) |
| §10 control-arm procedure specified (it_true, Q = 15, cold modes, maxiter 1000, FD h = 1e-4, SE = √diag H⁻¹) | **specification**; report-only status unchanged |
| §5 historical D3 comparator defined literally per doc 16, Rev 3-gate counts labelled separately (S5) | **specification**; neither enters the verdict, as R5 approved |
| §5 failure labels bias/dispersion/mixed/unclassified (S1) | **new reporting**, explicitly never changes the verdict |
| §§15, 16 added; change-log rows labelled by revision; tag legend (S2, M1–M3) | bookkeeping |

## 4. Findings

**F-A — the step-G human review was removed. Authorised, disclosed, but it is the one real relaxation. (For the record; no action available.)**

| | R1.2 / doc 45 | R1.3 (frozen) |
|---|---|---|
| step G authorisation | "**separate authorisation to continue**, after review of the Phase 1 outcome" (doc 45 §8 step 7 likewise) | "authorised (Final Execution Directive), **conditional** on Phase 1 passing every integrity condition and the runtime gate" |
| closing rule | "No step may begin before its **authorisation**." | "No step may begin before its **preceding gate** has passed." |

The user's *Final Execution Directive* authorises this, and doc 50's Authority block discloses it, so it is legitimate. But it is an **authorisation change, and authorisations are not in doc 50 §3's list** (decision, threshold, tolerance, seed, sample size, gate, verdict rule) — so that sentence reads as a broader all-clear than the change it actually covers. The practical effect: Phase 2's ~23 node-hours began automatically at 04:40 today, with no one inspecting Phase 1's regression output, FD-step sensitivity or flags first. **In this instance nothing was lost** — Phase 1 was clean on every check (§5) — but the automatic-continuation design is what a reader of doc 50 §3 alone would not learn. Step H still ends in "Stop for review", so no conclusion is drawn without review.

**F-B — (X) was narrowed to main datasets. Correct resolution of a real R1.2 contradiction; again outside doc 50 §3's categories.**
R1.2 §8 (X) read "any exception in any stage of **any dataset** … or fewer than 100 datasets completed", which taken literally let a control-arm failure force INCOMPLETE — contradicting R1.2's own §10.3, "Reported only. It has no effect on the verdict (R9, approved)". R1.3 scopes every (X) clause to **main** datasets and states that control-arm exceptions, non-finite values and timeouts never cause (X), with §8's new "Control-arm isolation (R9, B4)" property. This resolves the contradiction in favour of the **approved** decision R9, which is the right direction, and B4 discloses it. Strictly, though, a run-level verdict condition was narrowed, so doc 50 §3's "no … verdict rule was changed" is right in spirit and too strong in letter.

**F-C — the mixed objective context inside A persists, and R1.3 deserves credit for making it explicit. (No action; bears on how a PASS is read.)**
§3.2 now pins H₁/H₂ to the **optimiser context (P1/P2 penalties active)**, exactly as Step 6, while C₁/C₂ come from the **reported context (G = 5 AGQ, cold base)**. A = ∂U/∂Θ is then not the Jacobian of one estimating function, so the stacked M-estimation theory behind V = A⁻¹BA⁻ᵀ does not apply exactly. This is deliberate: it reproduces the Step 6 / D4 path, which is precisely what R-a (bitwise) and R-c (1e-6) demand, and doc 46 §§3.6 and 6 carry it as a stated limitation whose calibration only the recovery test can settle. The empirical reassurance is real but partial: N3 couples H₂ and C₂ in a single identity and holds to 2.75e-7 on the regression dataset, so the context mismatch does not visibly break that identity. Recorded so that a PASS is not read as validating the sandwich generally — which §13 already says it does not.

**F-D — N3 and N4 use different denominators. (Minor; no consequence at observed margins.)**
N3 normalises by ‖H₂v_ψ‖, N4 by ‖A₃v_φ‖. Both are stated, so neither is ambiguous, but "≤ 1e-5 relative" means two different things between the two checks. Observed values sit two to three orders inside the tolerance (2.75e-7, 9.37e-8), so the choice is immaterial here.

**F-E — the truth-swap comparison uses `equal_nan=True`. (Minor; recorded only.)**
Per `DEV_DEFECT_NOTE_pre_smoke.txt`, the bit-identity comparison was changed to treat identical all-NaN SE arrays as equal, after a 30-patient development case produced singular SEs. If SE_S were all-NaN the test would pass trivially — but such a dataset carries F8 regardless, and fresh dataset 0 (the only one tested) came through with no flags. No practical weakening.

**F-F — doc 46's "Unchanged from doc 44" line is now stale, inside a hash-locked file. (Minor documentation defect.)**
Line 25 of frozen doc 46 still asserts "§8 exclusions" is unchanged from doc 44, but §8 is now **"Authorisation status (R1.3)"**; the word "exclusion" occurs nowhere else in the document. The doc 44 exclusion content survives in substance as §8's "Not authorised" list (Steps 7–8, calibration, positive controls, real-data analysis, S1-c, any change of a frozen criterion), and §0's "Unchanged" block independently keeps calibration and Steps 7–8 blocked, so **nothing material was lost.** But the line misdescribes the frozen text.

**F-G — doc 46's R1.3 change-log row omits §8.**
It lists "§§3.2, 3.5, header", while §8 was in fact replaced. In this project the change log is the audit trail, so an omission matters more than it would elsewhere. Both F-F and F-G are now inside a locked file and cannot be corrected without a new revision and re-freeze — which would be disproportionate mid-run. The right disposition is to record them here and fold the correction into whatever revision follows the test's report.

**F-H — carried forward from doc 51, unchanged:** the planning and pre-freeze scripts (`56`, `57`, `58`, `tests/test_rev3_planning_functionals.py`) are still not hash-locked. Script 58's digest is self-recorded in its own `summary.json` (`b9cef343…`) and the Phase 1/Phase 2 code freeze covers the 19 files the test itself runs, so the gap now affects only the provenance of the frozen thresholds and the pre-freeze evidence, not the running code.

## 5. State of the running test (09:38Z, 2026-10-20) — observed, not acted on

| job | phase | node | state | elapsed / limit |
|---|---|---|---|---|
| 14013 | smoke (defective driver) | — | **cancelled while PENDING**, 0 node-hours, not counted | — |
| 14017 | smoke | node003 | COMPLETED | 5:01 / 30 min |
| 14094 | Phase 1 | node002 | COMPLETED | 1:01:20 / 2 h |
| 14095 | Phase 2a | node002 | **RUNNING** | 10:27:39 / 13:30 |
| 14096 | Phase 2b | node003 | **RUNNING** | 10:27:39 / 13:30 |

**Phase 1 regression and gate — all clean:**
- R-a: ψ̂⁽¹⁾ and ψ̂⁽²⁾ **bitwise identical** to Step 6; H₁ and H₂ also equal to Step 6's;
- R-b: max absolute difference **0.0**, against a 1e-12 tolerance;
- R-c: max relative difference **0.0**, against 1e-6;
- R-d: N3 2.75e-7, N4 9.37e-8, against 1e-5;
- fresh datasets 0–2: **no flags of any kind**;
- runtime gate **passed**: P = 23.889 ≤ 28; per-job projection 11.549 h ≤ L₂ = 13.5 h; worst-case cumulative 28.106 ≤ 30 node-hour cap; U before Phase 2 = 1.1058 node-hours.

**Phase 2 progress:** 45 of 49 main datasets complete in 2a (at r47), 46 of 48 in 2b (at r97) — **94 of 100 main datasets complete, with zero flags recorded anywhere**, at roughly 580–680 s per dataset. Both jobs are inside their projections, with the control arms still to run after the main datasets (B6 ordering).

**Code freeze intact:** all **19** files in `03_code_freeze.sha256` re-hashed during the run — **0 changed, 0 missing**. The frozen protocol locks for docs 46 and 47 also still verify.

**Protocol compliance of the pre-smoke defect fix** (`DEV_DEFECT_NOTE_pre_smoke.txt`): two defects were found by a login-node development check, fixed **before** any smoke run and before the code freeze, documented with reasons, and the one job submitted with the defective driver (14013) was cancelled while still PENDING, so it consumed no node-hours and is excluded from the smoke count. This is what §11 step D permits. The first defect's reasoning — that the integer pool tag only seeds perturbed starts, unused at `starts = 1`, so the fits are unaffected — is confirmed empirically by R-a passing bitwise.

## 6. Conclusion

The R1.3 revision is a faithful, well-evidenced tightening of the frozen protocol. It changed no criterion, threshold, tolerance, seed, sample size, gate or verdict rule; it corrected one mathematical error that would otherwise have terminated the run spuriously, and it did so by fixing the identity rather than the tolerance. Its one real relaxation is the removal of the step-G review (F-A), which the user authorised explicitly. The three documentation defects (F-B's over-broad wording, F-F, F-G) are minor and, being inside locked files, are best folded into the post-test revision.

**Nothing found here warrants interrupting the running test.** The next decision point is step H: the verdict and report, after both Phase 2 jobs finish and the post-run integrity record is written.
