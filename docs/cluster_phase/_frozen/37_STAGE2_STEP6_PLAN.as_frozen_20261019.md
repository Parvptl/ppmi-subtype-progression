# Stage 2 M2, Step 6: full-pipeline recovery with S1-b. Execution plan (FINAL)

**Date:** 2026-10-19.

**Status: APPROVED AND FROZEN** (SHA-256 in `docs/cluster_phase/_frozen/37_STAGE2_STEP6_PLAN.sha256`, with a read-only copy `_frozen/37_STAGE2_STEP6_PLAN.as_frozen_20261019.md`), before any code or computation.
- This document is the approved form of the proposal, doc 36 (`0be68267…fbec`, kept unchanged). It records the reviewer's decisions E-1 to E-12 (§4).
- **It authorises freezing only.** Implementing the driver and tests, and running Step 6, each need the user's separate approval.
- It changes no frozen specification, threshold, seed or prior result.

**Notation.** [F] marks a frozen requirement. [A] marks an approved decision (§4). [P] marks a procedure fixed by this plan to implement [F] and [A]. [U] marks an ambiguity left unresolved (§14).

---

## 1. Governing documents and their precedence (rechecked 2026-10-19)

| doc | file | frozen SHA-256 | role |
|---|---|---|---|
| 16 | `16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.md` | `c2eace18999f45e05324575e58d6933109c3836be67f832bfee55c8ad6bffbbc` | base specification: B5, B11, **D3 recovery definition** |
| 21 | `21_STAGE2_DOC16_AMENDMENT1_PROPOSED_REV2_20261017.md` (Rev 2) | `c53c015f…267` (lock `21_STAGE2_DOC16_AMENDMENT1_REV2.sha256`) | amends doc 16: §0 (recovery definitions unchanged), §2.5, §4.2 (B0 rule), **§5 S1-b**, **§6 step 6**, R4 |
| 25 | `25_STAGE2_AMENDMENT1_REV2_ADDENDUM_A_FINAL_20261017.md` | `439cf084…` | P1, P2(a)/(b), F1, runtime accounting |
| 27 | `27_STAGE2_EXACT_AGQ_GRADIENT_AMENDMENT_PROPOSED_20261018.md` | `17ca7a61…` | the exact AGQ gradient (fallback; reported evaluations only here) |
| 29, 31, 34 | Step 3, 4 and 5 execution plans | locks in `_frozen/` | bind only their own step. Step 3's results (Q\* = 15, G\* = 5) are inputs |
| 20 | Rev 1 (`c0209941…bd`) | superseded, unfrozen | the S1-b step wording (1)–(4) and "exactly one iteration". Used only where Rev 2 is silent |

**Precedence:** doc 16, as amended by Rev 2, then Addendum A and doc 27 (each an addendum within Rev 2's scope). Where they are silent, the approved decisions of §4 apply. A later document number confers no precedence.

## 2. Verification record (read-only, 2026-10-19, before freezing)

**Repository:** HEAD `3a860ca`; 0 tracked changes; 0 staged.

**Frozen locks:** all OK for docs 16, 21, 25, 27, 29, 31 and 34.

**Panel:** MD5 `fb038ddb…`.

**Code** (unchanged against the pilot provenance and the Step 4 and Step 5 baselines):

| file | SHA-256 |
|---|---|
| `src/ppmi_subtype/m2_quad.py` | `50f164c6fc3e04b1d72e7726d87dfa7b5306f93bc590e494b6aca349797b093f` |
| `src/ppmi_subtype/m2_grm.py` | `585c878b897b99cb15acf38246b0ee4063948219e6ef99866773cf0fbf6a4a46` |
| `src/ppmi_subtype/grm.py` | `1584bc62643bea544c27515fe85e02ba6f9383703b3170ce69b7473609c8b7a8` |
| `scripts/45_m2_pilot.py` | `a25e150313dc1096b272ae15c38852c3224244510179984a9e3a03016983c684` |
| `scripts/51_m2a1_s1a_true_latent_items.py` (Step 5; equal to the Step 5 job's recorded driver hash) | `517c4cc166750c3bbb48d1643b76c4aa3ea25387468af69d6b5f82b8fcbc8ff9` |
| `tests/test_s1a_latent_items.py` (Step 5) | `ddda689f919756911605c9b2c2ea5cd58adaa64cd8d85f8bb32c1b598d3623cc` |

**Inputs and prior outputs** (unchanged; no file in the pilot, Step 4 or Step 5 directories modified after the Step 5 integrity record):

| file | SHA-256 |
|---|---|
| pilot `results/step1_recovery.json` | `d16d5435b85eae99ff84f22b69594b0853ff23833c983a2b583e082b79188603` |
| pilot `results/provenance.json` | `68bc14b59d4c40cf35e6041a9bdd59bc2902b62743d34618a72ebd5ff78e7d1f` |
| Step 4 `step4_run2/summary.json` | `00433150c23c90ecc039bff39fe2cf87d1fa13614b597cad0a611964af7afa39` |
| identity reference `…_stage2_m2_exact_agq/bench/S2_laplace_fit.json` | `dbeb36b7f9edc055cb258ec9bdcfe12bb80411b9df541d10391e51bb1bc096dc` |
| Step 5 `step5/summary.json` | `5ccc690cd6eb5b5d93ceb06e7d9d7678ee9c3471b6fc247b1f1e46d64b429dae` |
| Step 5 `step5/S1_items_dataset0.json` | `c16ecfbc8b984214c7e5028ab4b503fb509ae00083f602efce0b8831436790a1` |

**Protocol facts confirmed in the frozen code:**
- `N_RECOVERY = 5`, with seeds `np.random.default_rng([42, 45, 1, r])` (`scripts/45_m2_pilot.py`, lines 56 and 296);
- the a-priori truth: β = (0, 0.05, 0.10, −0.20, 0.15, 0.20); SD_b = (0.8, 0.08, 0.8, 0.12); corr_b (0, 1) = (0, 2) = (2, 3) = 0.1; SD_ε = (0.5, 0.5); corr_ε 0.1. The items are `TRUTH_ITEMS`, re-expressed on the stage-1 scale by `truth_on_stage1_scale`;
- the pilot provenance records the same seeds and truth;
- `m2_grm.default_psi()`: β = 0; D = diag(0.8², 0.1², 0.8², 0.1²); Σ_ε = diag(0.5², 0.5²).

D1 verdict: **NOT REPRODUCED**, unchanged.

## 3. Frozen requirements and the exact recovery criterion [F]

- **Rev 2 §6 step 6:** "**Full-pipeline recovery** with S1-b: every ψ ≥ 4 of 5, **unchanged criterion.**" "Each step's failure stops the sequence; it is reported, and nothing after it runs."
- **Rev 2 §5, S1-b:** "Exactly one refit iteration: N(0, 1) stage 1 → stage 2 (§2) → items re-estimated under the fitted population latent mixture → stage 2 re-fit." "Accepted **only if** the **unchanged full-pipeline recovery** (doc 16 D3: 5 datasets, every ψ ≥ 4 of 5) passes with this pipeline." S1-c needs a separate amendment.
- **Rev 1 §5, S1-b steps:** "(1) N(0, 1) stage 1 as now; (2) stage 2 with the §2 likelihood; (3) re-estimate items by marginal maximum likelihood under the **fitted** population latent mixture from (2); (4) re-fit stage 2 with the refitted items. Pre-declared: exactly one iteration, no iteration until convergence."
- **Doc 16 D3, simulation recovery:**
  - 5 datasets (seeds `[42, 45, 1, r]`) on the real visit design (times and treatment indicator of the 934 patients), from the a-priori truth in `scripts/45_m2_pilot.py`;
  - the full two-stage pipeline;
  - "95 % Wald intervals on ψ come from the finite-difference Hessian";
  - "Pass: every component of ψ is covered in ≥ 4 of 5 datasets";
  - "Item-parameter recovery is reported (correlation and maximum error) but not gated."
  - Rev 2 §0 adds: truth "on the stage-1 scale".
- **Doc 16 D3:** the Hessian is the central FD (step 1e-4) of the analytic gradient. Bounds: means ±10, log-diagonals [−8, 3], off-diagonals ±5. A bound hit means within 1e-6 of a bound.
- **Doc 16 B5:** item parameters are held at their stage-1 values for ψ uncertainty. That item uncertainty is ignored is a stated limitation.
- **Rev 2 §2.5 and doc 25 P1/P2:** the safeguards apply to the recovery checks. P2(b): "A non-zero fallback count, or any budget trigger, at a reported fit optimum **fails that fit's B11 longitudinal check**."
- **Rev 2 §4.2 B0 rule:** one fit at the chosen Q ≤ 60 min, and the projected full pilot ≤ 24 node-hours.
- **Step 3:** Q\* = 15; G\* = 5.

**The exact frozen recovery criterion:**

> Over the 5 datasets, for each of the 19 components k of ψ, count the datasets in which ψ_true,k lies in ψ̂⁽²⁾_k ± 1.96·SE_k. SE = √diag(H⁻¹), and H is the central-FD (h = 1e-4) Hessian of the analytic stage-2 gradient at the final S1-b estimate ψ̂⁽²⁾, with the S1-b items fixed.
>
> **The criterion is met if and only if every one of the 19 counts is ≥ 4.**

This plan does not change it.

## 4. Approved decisions (reviewer, 2026-10-19) [A]

| # | decision |
|---|---|
| **E-1** | The step (3) item refit uses the **visit-specific** fitted latent distribution θ_v ~ N(M_ij, K_ij,jj) at ψ̂⁽¹⁾, the stage-2 estimate from step (2). It is per subscore, uses each visit's time and treatment, is a per-visit (composite) likelihood, and uses 61-node GH |
| **E-2** | **No re-standardisation** after S1-b. ψ̂⁽²⁾ is compared directly with the known generating truth ψ_true (stage-1 scale) |
| **E-3** | Item-fit start: **one start at `grm.initial_items`**, for steps (1) and (3) |
| **E-4** | The second stage-2 fit, step (4), starts at the **frozen default ψ** (`m2_grm.default_psi()`) with **cold patient modes**. So does step (2) |
| **E-5** | **One start per fit** (the frozen pilot convention). No new multistart procedure |
| **E-6** | A **non-PD or singular final Hessian** gives **no covered component** for that dataset. A **missing or non-finite SE** means that component is **not covered**. Non-convergence, boundary estimates, failed item fits and missing intervals are **reported explicitly**. No new pass/fail threshold. A failed fit is never silently turned into a valid recovery result. Where the frozen documents do not resolve how an incomplete fit affects the overall verdict, the ambiguity is **preserved in the report** (§7, §14) |
| **E-7** | On any execution error: **stop; the run is INCOMPLETE.** No automatic rerun or change of settings without review |
| **E-8** | ψ̂⁽¹⁾ and its Hessian are reported as **descriptive, non-binding** diagnostics. **Only ψ̂⁽²⁾ enters the criterion** |
| **E-9** | **Dataset-0 runtime gate,** with unchanged limits: each stage-2 fit ≤ 60 min, and the projected pilot ≤ 24 node-hours (§10). If either is breached, stop before datasets 1–4 |
| **E-10** | Any fallback or budget event at the final optimum is **recorded and reported**, with its Addendum A / B11 implication. The coverage criterion is not changed and the event is not concealed. Its effect on the formal verdict is **flagged for review** (§7) |
| **E-11** | **Exact equality** with the recorded pilot N(0, 1) item-fit outputs. Any mismatch **stops the run before any stage-2 fit** |
| **E-12** | `latent_negloglik` (and its companion `visit_latent`) are reused from `scripts/51_m2a1_s1a_true_latent_items.py` **only after** its hash is verified (`517c4cc1…`) and its implementation is confirmed against the approved reference (§9 checks I-7 to I-9). No frozen source or prior output is modified |

## 5. Synthetic datasets [F]

- **Number:** 5. **Seeds:** `np.random.default_rng([42, 45, 1, r])`, r = 0…4. These are the same datasets as the pilot, doc 18 and Steps 4 and 5.
- **Truth:** `truth_on_stage1_scale(groups)` from `scripts/45_m2_pilot.py`, giving ψ_true, it_true and the scale. It must equal the pilot record (σ_T 0.4783, σ_P 0.4053).
- **Design:** the 934 patients of the frozen Stage 2 cohort, 4,295 visits. These are the real visit times and treatment indicator (x = 1 if `PDTRTMNT == 1`, else 0), through `load_cohort`, design only. There are 11 tremor and 5 PIGD items.
- **Simulation:** `simulate_groups` / `regroup`, unchanged.
- **Real item responses and labels are not used.**
- **No other random numbers are drawn anywhere in Step 6** (E-3, E-4, E-5).

## 6. Procedure per dataset r [P]

| step | operation | settings |
|---|---|---|
| (0) | generate dataset r | §5 |
| (1) | stage 1, N(0, 1): item fits for T and P | `scripts/45_m2_pilot.py::fit_items_multistart(Y, tag, starts = 1)`, tag 100 + r (T) and 200 + r (P). This is `grm.stage1_negloglik`, 61-node GH, L-BFGS-B from `initial_items`, maxiter 3000. The fits are run for all 5 datasets first; E-11 is checked before any stage-2 fit |
| (2) | stage 2 → **ψ̂⁽¹⁾** | `m2_quad.fit(groups, items₁, Q = 15, psi0 = default_psi(), jobs = 32)`: Laplace over b, P1/P2 active, L-BFGS-B options and bounds unchanged (maxiter 1000), cold modes |
| (2h) | E-8: H⁽¹⁾ at ψ̂⁽¹⁾, plus the reported evaluation | `fd_hessian` (h = 1e-4) using the fit's mode state; `evaluate_reported(ψ̂⁽¹⁾, …, 15, 5)` — **non-binding** |
| (3) | **S1-b item refit**, once, per subscore | m_v = M_ij^s(ψ̂⁽¹⁾) and s_v² = K_ij,jj^s(ψ̂⁽¹⁾) via `visit_latent(groups, ψ̂⁽¹⁾)` (script 51). Minimise `latent_negloglik` with the same nodes, L-BFGS-B, maxiter 3000, from `initial_items(Y)` (script 51's `fit_oracle` path with ψ̂⁽¹⁾) |
| (4) | stage 2 refit → **ψ̂⁽²⁾** | as (2), with items₂, default ψ and cold modes (E-4) |
| (5) | **H⁽²⁾ at ψ̂⁽²⁾** | `m2_quad.fd_hessian` (central, h = 1e-4, 38 evaluations), using the fit's mode state (as Step 4) |
| (6) | reported evaluation at ψ̂⁽²⁾ | `evaluate_reported(ψ̂⁽²⁾, groups, items₂, 15, 5, init_state)`: fallback count, budget triggers (E-10) |
| (7) | coverage | §7 |

**Exactly one S1-b iteration** [F]. Nothing is iterated to convergence. Step 5 results are not used.

**Also recorded per dataset (reported only):**
- item recovery for items₁ and items₂ against it_true: correlation and max |error| [F], plus mean error and RMSE;
- pooled moments of the fitted mixture at ψ̂⁽¹⁾;
- σ_T and σ_P on the natural scale;
- every fit's status, message, iterations, evaluations, ‖∇‖ (B11's 0.934 shown for reference only), bound hits, penalties by reason, non-ND counts and wall time.

## 7. Coverage computation and verdict classification [F + A + P]

**Per dataset, for ψ̂⁽²⁾:**
- H⁽²⁾ is PD if it is finite and its smallest eigenvalue is > 0.
- If H⁽²⁾ is not PD, singular, or not finite, then **covered_k = false for all k** (E-6).
- Otherwise SE = √diag(H⁻¹), z_k = (ψ̂⁽²⁾_k − ψ_true,k)/SE_k, and covered_k = (SE_k finite) ∧ (|z_k| ≤ 1.96). A non-finite SE gives not covered (E-6).
- The same computation is made for ψ̂⁽¹⁾. It is labelled non-binding (E-8).

**Flagged events** [U, A]. The frozen documents do not say how these affect the recovery verdict.

| code | event |
|---|---|
| F1 | stage-2 fit (2) or (4) not converged: L-BFGS-B success flag false, maxiter reached, or stopped by time |
| F2 | item fit (1) or (3) not converged, or any discrimination outside [0.05, 20] |
| F3 | a bound hit (within 1e-6) at ψ̂⁽²⁾ |
| F4 | at the reported evaluation of ψ̂⁽²⁾: fallback count > 0 or a budget trigger. This fails that fit's B11 longitudinal check under doc 25 P2(b) (E-10) |

A non-PD H⁽²⁾ and a non-finite SE are **not** flags: E-6 resolves them.

**Run outcome classes** (formal verdict kept separate from incomplete execution or numerical failure):

| class | condition | what is reported |
|---|---|---|
| **INTEGRITY STOP** | a preflight check (§9) or E-11 fails | no fits; **no recovery verdict** |
| **INCOMPLETE** | execution error (E-7); runtime gate breach (E-9); non-finite objective or ψ at a final optimum; SLURM time limit; fewer than 5 datasets complete | completed datasets reported as non-binding partial results; **no recovery verdict** |
| **COMPLETE, no flagged event** | all 5 datasets complete steps (0)–(7) and no F1–F4 | **formal verdict: PASS** if every count ≥ 4, otherwise **FAIL** |
| **COMPLETE, with flagged events** | all 5 complete, but at least one of F1–F4 occurred | the computed counts and the computed outcome (criterion met / not met) are reported. **The formal verdict is RESERVED FOR REVIEW** (ambiguities U-1 to U-3). The report also states whether the computed outcome changes if every flagged dataset's components are taken as all covered or as none covered. That is a description of the dependence, not a criterion |

**FAIL or PASS consequences [F].**
- A FAIL means S1-b is not accepted and the sequence stops (Rev 2 §6). Step 7 is not run; S1-c needs a separate amendment.
- A PASS satisfies the binding recovery pre-requisite only. Step 7 still requires separate authorisation.

## 8. Numerical settings and the fallback issue [F]

**Settings:**
- Laplace over b, per-visit 15 × 15 GH over ε (Q = 15);
- P1 (10 consecutive non-ND iterations, or 200 halvings) and P2 (eigenvalue ND test; optimiser-context penalty 1e12);
- absolute line-search tolerance 1e-12;
- the D3 bounds and L-BFGS-B options of `m2_quad.fit` unchanged;
- 32 worker processes and single-threaded BLAS on node002 (Xeon Gold 6326; the job waits if the node is busy);
- the G = 5 AGQ (doc 27) fallback **only** for non-ND patients in reported evaluations.

**Fallback runtime issue** (doc 28: G = 5 fallback fits 83–104 min; projected fallback pilot 37.4 node-hours).
- Step 6 never optimises with the fallback.
- The deferral recorded in doc 31 §0.1 applies to Step 7 ("Step 7 must stop if the primary model fails V2 and no affordable, accuracy-admissible fallback is available"), and is not affected. Step 6 runs no V2 and does not touch that rule.

## 9. Pre-fit integrity checks (in the job; any failure gives INTEGRITY STOP)

| id | check |
|---|---|
| I-1 | frozen locks: docs 16, 21, 25, 27, 29, 31, 34 and **37** match |
| I-2 | code hashes equal §2 (`m2_quad`, `m2_grm`, `grm`, `45_m2_pilot`, `51_…`, `test_s1a_latent_items.py`), and the pilot-provenance code set is unchanged |
| I-3 | panel MD5 `fb038ddb…`; input hashes equal §2 |
| I-4 | the new Step 6 unit tests pass (§11), run inside the job |
| I-5 | ψ_true equals the pilot record exactly |
| I-6 | dataset identity: serial `MQ.negloglik` on dataset 0, true items, Q = 10, at the recorded ψ of the identity reference = **57022.74225348176 exactly** |
| I-7 | E-12: `scripts/51_…` hash = `517c4cc1…`, equal to the Step 5 job's recorded `driver_sha256` and its `02_driver_sha256.txt` |
| I-8 | E-12: the Step 5 tests (`tests/test_s1a_latent_items.py`, hash-verified) pass. These include `latent_negloglik` reducing to `grm.stage1_negloglik` at m = 0, s = 1 (1e-12), the gradient against central FD, and affine invariance |
| I-9 | E-12: with the reused functions at ψ_true on dataset 0, `latent_negloglik` evaluated at Step 5's **recorded** oracle items reproduces Step 5's recorded `oracle_nll` (`S1_items_dataset0.json`, T and P) exactly. The construction moments reproduce Step 5's `construction_check` for dataset 0 exactly |
| I-10 | seed list `[42, 45, 1, r]`, r = 0…4, recorded; git HEAD and working-tree state recorded; 0 staged |
| I-11 (E-11) | after step (1) for all datasets: the pilot `items[r]` six metrics (`a_corr_T`, `a_maxerr_T`, `tau_maxerr_T`, `a_corr_P`, `a_maxerr_P`, `tau_maxerr_P`) are reproduced exactly, 30/30. **A mismatch stops the run before any stage-2 fit** |

## 10. Dataset-0 runtime gate (E-9) [P, limits F]

**Measured on dataset 0:**
- t₂ and t₄, the wall times of the stage-2 fits (2) and (4);
- ℓg, the likelihood-plus-gradient time from H⁽²⁾ (wall time / 38);
- t_S1b, the step (3) item-refit wall time.

**Limits (unchanged):**
- (i) t₂ ≤ 60 min **and** t₄ ≤ 60 min;
- (ii) projected pilot ≤ 24 node-hours.

**The projection uses the frozen doc 25 accounting formula** (18 fits = 3 real starts + 5 fixed-item + 10 S1-b stage-2 fits; 11 Hessians):

  P = [18 × max(t₂, t₄) + 11 × 38 × ℓg + 716 s (observed item fits) + 0.62 h (assumed S1-b component)] / 3600

A variant replacing the assumed 0.62 h with 5 × (t_S1b + the E-8 Hessian time) is **reported only**.

**If (i) or (ii) fails: stop; INCOMPLETE; datasets 1–4 are not run.**

## 11. Implementation scope (on separate approval only)

New files only:
- `scripts/52_m2a1_full_pipeline_recovery.py` (driver);
- `tests/test_s1b_pipeline.py`.

**Tests:**
- (a) coverage logic: PD / non-PD / singular / non-finite H; NaN SE; |z| = 1.96 inclusive;
- (b) outcome classification for each §7 class, and the dependence statement;
- (c) the step sequence on a tiny synthetic case with stubs: each of (1)–(6) runs exactly once per dataset, S1-b exactly once, starts as in E-3 and E-4, no RNG beyond the dataset seeds;
- (d) the runtime projection formula of §10;
- (e) `visit_latent` at a given ψ uses `M.unpack(ψ)`, matching the Step 5 construction at ψ_true.

**Smoke code path,** labelled *not a result*: 1 dataset, 60 patients, reduced maxiter. The identity and reproduction checks are skipped. Written to `smoke_codepath/`.

No existing file is modified.

## 12. Outputs, manifest and checksums [P]

**Execution directory:** `experiments/<UTC>_stage2_step6_full_pipeline_recovery/`:
- `00_baseline.json`, `01_plan.sha256` (copy of the doc 37 lock), `02_driver_sha256.txt`;
- `step6/`:
  - `S0_provenance.json` (hashes, seeds, default ψ values, host, CPU, threads, versions, job ID);
  - `S1_dataset{r}.json` (all stages: item sets; fit logs; ψ̂⁽¹⁾ and ψ̂⁽²⁾; H⁽¹⁾ and H⁽²⁾, eigenvalues, SE, z, coverage; reported evaluations; flags; timings);
  - `S2_coverage_table.csv` (19 × 5: truth, ψ̂⁽²⁾, SE, lower, upper, z, covered, H PD, bound flag);
  - `S2b_coverage_table_psi1_nonbinding.csv`;
  - `S3_item_recovery.csv` (per item × dataset × {stage 1, S1-b});
  - `S4_convergence.csv`;
  - `summary.json` (outcome class, counts, flags, gate, runtime);
  - `figures/`: z by component and dataset with ±1.96 (ψ̂⁽²⁾, and ψ̂⁽¹⁾ marked non-binding); estimated vs true a and τ (stage 1 vs S1-b); σ_T and σ_P with intervals;
  - `MANIFEST.sha256` (every file in `step6/`, written last);
- `smoke_codepath/`;
- `jobs_sacct.txt`, `99_post_run_integrity.txt`, `REPORT.md`.

**Post-run checksum verification:**
- `sha256sum -c step6/MANIFEST.sha256`;
- frozen locks re-checked;
- code, input and prior-output hashes unchanged against `00_baseline.json`;
- no file in the pilot, Step 3, 4 or 5 directories, or in `docs/`, `src/`, `scripts/` or `outputs/`, modified during the run (except the new report document after the run);
- `sacct` (AllocNodes, ElapsedRaw);
- git: 0 staged, no tracked change.

## 13. Reporting requirements

Report: `docs/cluster_phase/38_STAGE2_STEP6_FULL_PIPELINE_RECOVERY_REPORT_*.md`, with a copy at `REPORT.md`. It contains:

1. the outcome class and, where applicable, the formal verdict, stated **separately** from any incomplete-execution or numerical-failure status;
2. the preflight table (I-1 to I-11);
3. per-dataset convergence for every stage, item fits included;
4. the coverage table (19 × 5) and counts against ≥ 4 of 5;
5. the Hessians;
6. reported-evaluation fallback and budget events and their P2(b) / B11 implication;
7. every flagged event (F1–F4) and the dependence statement;
8. ψ̂⁽¹⁾ results, marked **non-binding**;
9. item recovery (stage 1 vs S1-b), not gated;
10. σ_T and σ_P;
11. the runtime gate and measured runtime against §15;
12. failure accounting;
13. the run manifest and checksum verification;
14. deviations, if any.

**Interpretation is limited to the criterion.** No statement about real PPMI data, model adequacy on real data, misspecification, or biological subtypes.

## 14. Remaining protocol ambiguities (documented, not resolved)

| # | ambiguity | handling in this plan |
|---|---|---|
| **U-1** | How a non-converged stage-2 or item fit (F1, F2) affects the recovery verdict. Doc 16 B11's convergence rules are real-data fit checks; D3's recovery rule does not mention convergence | coverage computed as usual and the event reported; **formal verdict reserved for review** (§7) |
| **U-2** | How a bound hit at ψ̂⁽²⁾ (F3) affects the verdict. B11 requires "no bound hits" for the real-data longitudinal fit; D3 is silent for recovery | as U-1 |
| **U-3** | How a fallback or budget event at ψ̂⁽²⁾ (F4) affects the verdict. Doc 25 P2(b) says it "fails that fit's B11 longitudinal check"; D3's coverage rule does not reference B11 checks | as U-1 (E-10) |
| U-4 | B11's gradient criterion (‖∇‖ < 1e-3 × 934 = 0.934) for recovery fits | reported for reference only, as in Step 4. Not a flag, not a criterion |
| U-5 | Whether an INCOMPLETE run counts as a Step 6 "failure" that stops the sequence under Rev 2 §6 | not decided here. An INCOMPLETE run gives no verdict, and the next action is the reviewer's |

No other gap remains open: E-1 to E-12 resolve the rest.

## 15. Runtime (measured / estimated / assumed)

| component | basis | value |
|---|---|---|
| step (1), 10 item fits in parallel | measured (Step 5: 53 s wall) | about 1 min, once |
| stage-2 fit at Q = 15, default start | measured with true items (Step 4: 179–244 s); with estimated items **assumed** similar | 3–4 min; 2 per dataset |
| step (3) S1-b refit (T and P in parallel) | measured for the analogous oracle fit (Step 5: 55–73 s); at ψ̂⁽¹⁾ **assumed** similar | about 1.2 min per dataset |
| FD Hessians H⁽¹⁾ and H⁽²⁾ (warm modes) | measured (Step 4: 54–55 s) | about 2 min per dataset |
| reported evaluations | measured (Step 4: about 5 s) | under 0.5 min per dataset |
| **total job** | estimate | **about 55–70 min ≈ 0.9–1.2 node-hours** (up to about 1.7 if fits take 1.5× longer, an assumption) |
| smoke check | estimate | under 0.1 node-hours |

## 16. Exclusions

The following are not authorised by this plan:
- writing the driver or tests;
- generating data, fitting or submitting a job (separate approval needed);
- Step 7, Step 8;
- calibration, positive controls, fidelity gates, evaluation statistics;
- any real-data evaluation;
- any change to frozen protocols, prior results or the D1 verdict (**NOT REPRODUCED**).

Nothing is staged, committed or pushed.

**Next action (on separate approval):** implement `scripts/52_m2a1_full_pipeline_recovery.py` and `tests/test_s1b_pipeline.py` per §§6–12, run the unit tests and the smoke code path, then submit the Step 6 job on node002 and report.
