# Stage 2 M2, Step 3: execution plan (likelihood accuracy, quadrature order, mode checks)

**Date:** 2026-10-19.

**Status.** This is the execution plan for Rev 2 §6 step 3, authorised by the user's instruction "Stage 2 M2 — Step 3: Likelihood Accuracy, Quadrature-Order and Mode Validation".
- It documents **how** the frozen Step 3 is run. It does not change the scientific protocol.
- It is frozen (SHA-256 in `docs/cluster_phase/_frozen/29_STAGE2_STEP3_PLAN.sha256`, with a read-only copy) before any computation.

**Execution directory:** `experiments/20261018T180500Z_stage2_step3_accuracy/` (`00_baseline.json`).

**Governing documents, all verified unchanged** (hashes in `00_baseline.json`):

| document | SHA-256 |
|---|---|
| doc 16 (frozen spec) | `c2eace18…bbc` |
| Rev 2, doc 21 (§2.5, §3, §4.1, §6 step 3) | `c53c015f…267` |
| Addendum A, doc 25 (P1/P2; reported-evaluation context; declared G = 5 for reported evaluations) | `439cf084…` |
| doc 27 (exact AGQ gradient) | `17ca7a61…` |
| doc 28 (execution report) | `83d383e7…` |

**Code under test:** `src/ppmi_subtype/m2_quad.py` (`50f164c6…`, unchanged since doc 28). The references `m2_grm.py` (`585c878b…`; the D3 reference functions `agq_one` and `is_one`) and `grm.py` are unchanged.

## 1. Reviewer resolutions of protocol gaps (2026-10-19, before computation)

The frozen text left these undetermined. The reviewer chose:

| # | gap | resolution |
|---|---|---|
| R-a | V4 perturbation size (§3 V4; it also defines the 4 ψ of the gated §4.1(a)) | ψ_k = clip(ψ̂ + N(0, 0.05²) per component, seed `[42, 46, 7, k]`), k = 1, 2, 3. This is the doc 25 T2 convention |
| R-b | §2.5 mode-uniqueness starts ("0, the prior-mean-implied b, …"; the prior mean is 0, so these coincide) | starts: b = 0; b̂ − 2 SD; b̂ + 2 SD, on every axis. SD = √diag(H⁻¹) at the b̂ reached from start 0 |
| R-c | reliability of the references | (i) **D3:** a patient's V2 result is *reference-limited* if \|e_i\| ± 2·MCSE_i straddles 0.05. It is reported, never dropped. (ii) **G_ref:** the G = 9 vs G = 7 adequacy check mirrors the frozen (a′). If it fails, the G-order result is **inconclusive** and no G is selected |
| R-d | how the order rule is read | the frozen Rev 2 §4.1 (a)/(a′)/(b) governs. V2 is evaluated at the selected Q. The G-order check is run (as requested), but is binding only if the fallback is invoked |

## 2. Cohort and inputs

**Patients.** The 30 designated patients: `np.random.default_rng([42, 45, 4]).choice(934, 30, replace=False)` over the sorted PATNO of the frozen Stage 2 cohort (doc 16 B2 and D3). Their identities and visit counts were verified equal to the pilot's recorded set (`step4_laplace.json`):
- PATNO: 4115, 4102, 177395, 311402, 3254, 116529, 266248, 4123, 4109, 143116, 238181, 149940, 3815, 137264, 101675, 3415, 100952, 4026, 130013, 3467, 3253, 3023, 3108, 3052, 3659, 168299, 148699, 182340, 101038, 215045;
- visits: 2–12;
- 12 patients have 2n ≤ 6.

**Data.** Their real item responses (all 16 items, complete), times and treatment indicator, through `scripts/45_m2_pilot.py::load_cohort`. Frozen panel MD5 `fb038ddb…`.

**Parameters** (frozen pilot outputs, used as fixed inputs; **no fitting**):
- items: `experiments/20261016T200350Z_stage2_m2_pilot/results/step2_items.json`;
- ψ̂: `…/step3_longitudinal.json` (`psi`).

**The 4 ψ:** ψ_0 = ψ̂, and ψ_1–ψ_3 per R-a.

## 3. What is assessed and how

**Per-patient log-likelihood:** ℓ_i = log p(Y_i | ψ, items), natural log. Each patient is evaluated alone, with a cold mode start at b = 0.

**Approximation A(Q):** Laplace over b with per-visit Q × Q Gauss–Hermite over ε (`m2_quad.group_laplace`, Rev 2 §2.2–2.3).
- **Reported-evaluation context** (doc 25 P2): if the returned Hessian is not negative definite, ℓ_i is taken from the AGQ fallback at the declared G = 5, and the patient is counted.
- A P1 budget trigger at a reported evaluation is reported as a failure for that patient.

**Fallback F(Q, G):** adaptive Gauss–Hermite over b (`m2_quad.group_agq`; the value is independent of the gradient option).

**D3 reference (unchanged):**
- `m2_grm.agq_one(nodes = 7)` when 2n ≤ 6;
- otherwise `m2_grm.is_one(draws = 20,000, df = 5)` with seed `[42, 45, 4, i]`, i = the patient's index in the sorted cohort (the pilot's seeds), at every ψ;
- the Monte Carlo SE is recorded.
- **Reproducibility check:** at ψ̂ the D3 values must equal the pilot's recorded `reference` values (`step4_laplace.json`) exactly.

**Error sign:** e_i = ℓ_i(approximation) − ℓ_i(reference), the pilot's convention.

## 4. Checks, aggregation and acceptance

| id | check | rule (frozen unless marked R-) | gated? |
|---|---|---|---|
| (a) | ε-order | at the 4 ψ, A(Q) vs A(20) for Q ∈ {7, 10, 15}: **max over 30 patients × 4 ψ of \|ℓ_Q − ℓ_20\| ≤ 0.005**. Q* = the smallest passing Q. If none passes: stop (V2 is not evaluated at a selected Q) and report | yes |
| (a′) | Q_ref adequacy | A(20) vs A(25) on the 5 patients with the most visits (ties broken by sorted-cohort index), at the 4 ψ: max ≤ 0.001, reported. If it fails, Q_ref becomes 25 and (a) is repeated against 25 | yes (it switches the reference) |
| V2 | total accuracy | at ψ̂, A(Q*) vs D3 for all 30 patients: **max_i \|e_i\| ≤ 0.05.** Mean and median \|e\| are descriptive only. Every patient is reported; R-c(i) flags reference-limited outcomes | **yes** |
| V2-F | fallback accuracy | at ψ̂, F(Q*, G) vs D3 for G ∈ {3, 5}, with the same rule | gated only if V2 fails (Rev 2 §3: "the fallback is tested next under the same V2 criterion") |
| (b) | b-order | at the 4 ψ, F(Q*, G) vs F(Q*, 7) for G ∈ {3, 5}: max over 30 × 4 ≤ 0.005. G* = the smallest passing G | binding only if the fallback is invoked (R-d) |
| (b′) | G_ref adequacy (R-c ii) | F(Q*, 9) vs F(Q*, 7) on the 5 most-visit patients at the 4 ψ: max ≤ 0.001. If it fails, (b) is **inconclusive** | yes, for (b) |
| §2.5 | mode checks | at ψ̂ and Q*, for all 30 patients: (i) ND count at the returned modes (reported; non-ND patients use the fallback); (ii) **uniqueness:** the 3 starts of R-b reach the same mode within 1e-6 (max abs component). **Any disagreement stops Step 3** (Rev 2 §2.5): V2 and (a) results for the affected patients are flagged uninterpretable | **yes** |
| V3 | slice (reported) | synthetic dataset 0, items at truth, the doc 19 σ_P grid and nuisance treatment (ψ at the dataset-0 fixed-item Laplace MLE from doc 18), A(Q*) and A(Q_ref) over the full synthetic cohort. Quadratic-vertex argmax, against the doc 19 D3 argmax 0.3973 ± 0.0134 | no |
| V4 | totals (reported) | at the 4 ψ: Σ_i e_i over the 30 patients (A(Q*) − D3), with the MC SE from D3, beside σ_T and σ_P at each ψ | no |
| param-effect | (reported) | (1) the change in the V3 argmax between Q* and Q_ref; (2) the change in Σ_i ℓ_i differences between the 4 ψ (30 patients), Q* vs Q_ref | no |

**Also reported for every evaluation:**
- ND status;
- P1 budget triggers;
- non-finite values;
- the number of fallback-evaluated patients;
- the mode-search iterations and halvings.

**Penalised trial evaluations** do not arise here: there is no optimiser. Any budget trigger is a reported failure.

**Step 3 passes only if:** (a) selects a Q* (after (a′)), **and** V2 passes at Q*, **and** the §2.5 uniqueness check passes. Step 3's outcome does not depend on (b), V3 or V4.

## 5. Runtime

**Setting:** node002 (Xeon Gold 6326; the job waits if it is busy), 1 exclusive node, 32 worker processes over patient-level tasks, single-threaded BLAS. Accounting: node-hours = AllocNodes × ElapsedRaw / 3600.

**No runtime gate applies to Step 3 itself.** For feasibility, the plan reports:
- the per-fit and pilot budgets of the configuration Step 3 makes admissible;
- the primary at Q* (doc 26 measurements: Q = 10 measured, other Q scaled);
- the fallback at G* (doc 28 measurements for G = 3 and 5).

No new benchmark is run.

**Stopping conditions:**
- If any hash or input check fails at job start, nothing is computed.
- If the D3 reproducibility check fails at ψ̂, V2 is reported as reference-irreproducible, and Step 3 is not passed.

## 6. Outputs

In the execution directory, under `step3/`:
- `S0_provenance.json`;
- `S1_reference_d3.json`;
- `S2_order_a.json` (and `S2_order_a_prime.json`);
- `S3_v2.json`;
- `S4_order_b.json`;
- `S5_modes.json`;
- `S6_v3.json`;
- `S7_v4_param_effect.json`;
- patient-level CSV tables;
- `summary.json`;
- the SLURM accounting.

**Report:** `docs/cluster_phase/30_STAGE2_STEP3_REPORT_*.md`.

## 7. Exclusions (not run)

- fixed-item recovery;
- S1-a and S1-b;
- full-pipeline recovery;
- real-data fitting or B11 (no parameter is estimated);
- the sampler and V5;
- G1–G7;
- calibration;
- positive controls;
- primary and secondary evaluation statistics;
- clustering;
- latent-state comparisons;
- any scientific verdict.

The D1 verdict, NOT REPRODUCED, is unchanged. Nothing is staged, committed or pushed. **After the report, stop for review.**
