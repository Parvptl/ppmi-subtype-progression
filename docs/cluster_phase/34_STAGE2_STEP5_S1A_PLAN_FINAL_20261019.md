# Stage 2 M2, Step 5 (S1-a): execution plan (FINAL, approved)

**Date:** 2026-10-19.

**Status: APPROVED AND FROZEN** (SHA-256 in `_frozen/34_STAGE2_STEP5_PLAN.sha256`, with a read-only copy), before any computation.
- This document is the approved form of the proposed plan, doc 33 (`325471e5…`, kept unchanged as the record of the proposal).
- It records the reviewer's decisions D-1 to D-5 (§6) and the implementation detail they imply. Nothing else changes from doc 33.
- It describes how Rev 2 §6 step 5 is run. It changes no specification, threshold or frozen record.
- Authorisation: the user's instruction of 2026-10-19, "Proceed with Step 5 (S1-a) only, following `33_STAGE2_STEP5_S1A_PLAN_PROPOSED_20261019.md` and the governing frozen Rev 2 protocol."

## 1. Governing text

**Rev 2 (doc 21, frozen `c53c015f…`; governing).**
- §5: "S1-a | diagnostic only | On the 5 recovery datasets, fit the items under the true population latent distribution and compare with the N(0, 1) stage-1 fit | Reported only. It authorises nothing, selects nothing, and no parameter or setting is chosen from it."
- §6 step 5: "S1-a diagnostic (reported only)."
- R4: "S1-a is diagnostic only and authorises nothing."

**Rev 1 (doc 20, unfrozen record `c0209941…`; superseded, but more detailed).**
- §5: "On the 5 recovery datasets, fit the items by marginal maximum likelihood under the true population latent distribution: the visit-level mixture of N(M_ij, K_ij,jj) at the true ψ, by quadrature per visit. Compare the item errors and the downstream fixed-ψ quantities with the N(0, 1) stage-1 fit | minutes | Quantifies the bias due to the N(0, 1) working assumption alone. Reported; no criterion."
- Also: "S1-a and S1-b use only synthetic data and real item responses. Neither uses labels."

**Reading adopted.** Rev 2 governs. It says "on the 5 recovery datasets", so S1-a uses synthetic data only.
- Rev 1's "real item responses" phrase covers S1-a and S1-b jointly. It can only apply to S1-b, because on real data the "true population latent distribution" is unknown.
- Rev 1's detail is used only where Rev 2 is silent, and every such use is listed as a decision in §6.

**Dependencies.**
- Step 3 passed: Q* = 15 and G* = 5 (doc 30).
- Step 4 passed: 13 parameters covered 5/5 and 6 covered 4/5 (doc 32).
- Rev 2 §6 sets no other precondition for step 5. Step 5 has **no acceptance criterion and no stopping rule tied to its result**.

## 2. Purpose, and what S1-a can and cannot establish

**Purpose.** Stage 1 fits the items assuming θ ~ N(0, 1) independently at every visit.
- The true visit-level latent distribution is a mixture over visits of N(M_ij, K_ij,jj). Its components depend on time and treatment through β and on D, Σ_ε.
- `truth_on_stage1_scale` makes this mixture match N(0, 1) in its pooled mean and variance, but not in shape.
- S1-a refits the items with the true per-visit latent distribution in place of N(0, 1), keeping everything else identical: the same data, the same per-visit (composite) likelihood, the same quadrature and the same optimiser.
- The difference between the N(0, 1) fit and this *oracle-latent* fit isolates the item error caused by the N(0, 1) working assumption alone. The difference between the oracle-latent fit and the true items is the remaining error from sampling and from the per-visit composite likelihood.

**How S1-a relates to the other recovery steps:**

| step | items | ψ | what it tests |
|---|---|---|---|
| 4: fixed-item recovery (done) | true | estimated (Laplace, Q = 15) | stage 2 alone, given perfect items |
| **5: S1-a (this plan)** | N(0, 1) fit vs oracle-latent fit vs truth | **held at ψ_true** (§6 D-2) | the item bias due to the N(0, 1) assumption, and its first-order effect on ψ |
| 6: full-pipeline recovery with S1-b (binding) | estimated (N(0, 1), then one S1-b refit) | estimated | the whole two-stage pipeline, under the coverage criterion |

**S1-a can establish:**
- under the a-priori truth and the real design, how large and in what direction the N(0, 1)-induced item errors are (a and τ, per item and subscore), compared with sampling error;
- to first order at ψ_true, how much those item errors would shift ψ, in SE units. This is a descriptive pointer to which ψ components Step 6 is likely to stress.

**S1-a cannot establish:**
- that the pipeline recovers ψ. Only Step 6 is binding;
- whether S1-b should be used. S1-b's use is fixed by the frozen protocol, and S1-a selects nothing;
- anything about real data, model adequacy or misspecification beyond the a-priori truth;
- anything about subtypes or labels (none are used);
- the effect of ignoring within-patient dependence or the T–P correlation in stage 1. Both item fits use the same per-visit composite likelihood, so that effect cancels out of the comparison;
- anything inferential from 5 datasets. All summaries are descriptive.

The oracle fit uses the unknown truth, so it is never an estimator available in practice.

## 3. Verification already performed (read-only, 2026-10-19)

**Frozen records:** `sha256sum -c` passes for docs 16, 21, 25, 27, 29 and 31.

**Code:**
- Every file in the Step 4 baseline `code_sha256` is unchanged, including `m2_quad.py` `50f164c6…`.
- Every pilot code file is unchanged from the pilot provenance (pilot job 13822 on node002, single-threaded BLAS): `scripts/45_m2_pilot.py` `a25e1503…`, `m2_grm.py` `585c878b…`, `grm.py` `1584bc62…`, `m2_gates.py`, `raw_scores.py`, `subtype.py`.

**Panel:** MD5 `fb038ddb…`, matching the frozen value.

**Results:**
- Step 4 `summary.json` (`00433150…`): criterion passed; identity check 57022.74225348176, equal to the recorded value. The Step 4 report (doc 32, `dbdf52ae…`) is present.
- The pilot `step1_recovery.json` holds the N(0, 1) item-error metrics for datasets 0–4. Item a max error is 0.134–0.282 for T and 0.280–0.867 for P.

**Repository:** HEAD `3a860ca`; nothing staged; no tracked file modified. The D1 verdict, NOT REPRODUCED, is untouched.

## 4. Data and fixed inputs

**Cohort:** the frozen Stage 2 cohort (934 patients, 4,295 visits) from `load_cohort`. It supplies the real times and treatment indicator only. Real responses are not used.

**Truth:** `truth_on_stage1_scale(groups)`, giving ψ_true, it_true and the scale (μ, sd).
- These are the a-priori `TRUTH_ITEMS` / `TRUTH_LONG`, re-expressed on the stage-1 scale (σ_T 0.4783, σ_P 0.4053).

**Datasets:** r = 0…4, `simulate_groups(np.random.default_rng([42, 45, 1, r]), groups, ψ_true, it_true)`.
- These are byte-identical to the pilot, doc 18 and Step 4 datasets.
- **No new random numbers are drawn anywhere in Step 5.**

## 5. Procedure

For each dataset r, run the following for each subscore separately: T (11 items) and P (5 items).

**1. N(0, 1) item fit.** Use exactly the pilot path: `fit_items_multistart(Y, tag, starts = 1)`, with tag 100 + r for T and 200 + r for P.
- This is `grm.stage1_negloglik` with 61-node probabilists' Gauss–Hermite, L-BFGS-B, maxiter 3000, starting from `grm.initial_items(Y)`.
- This **reproduces** the pilot's step-1 item fits (see the reproducibility check, §9).

**2. Oracle-latent item fit.** Maximise the per-visit marginal likelihood
  Σ_v log ∫ Π_k P(y_vk | θ, a_k, τ_k) N(θ; m_v, s_v²) dθ, where m_v = M_ij^s and s_v² = K_ij,jj^s at ψ_true.
- M_ij^s and K_ij,jj^s come from `M.mean_design` / `M.covariance` with the unpacked ψ_true. This is the same construction as `truth_on_stage1_scale`, with the visit order of `flatten`.
- Quadrature: θ_qv = m_v + s_v·x_q on the same 61 nodes and weights.
- Implementation: a new function `latent_negloglik(phi, Y, m, s, nodes)`, which is `stage1_negloglik` with a per-visit node grid. It has the same analytic gradient, with θ replaced by θ_qv.
- Optimiser, start (`initial_items(Y)`), maxiter and parameterisation are identical to step 1. This is one start (§6 D-3).
- Construction check (reported, not a stop): per subscore, the pooled mean of m_v and the pooled mean(s_v²) + var(m_v) over all visits, which should be 0 and 1 by the construction of `truth_on_stage1_scale`.

**3. Item comparison** of the three item sets: truth, N(0, 1) fit and oracle-latent fit. Per subscore and per item, report:
- a error, a − a_true;
- τ error, τ_c − τ_c,true;
- the summaries max |err|, mean signed err, RMSE and corr(a, a_true). The pilot reported max |err| and corr;
- the per-item paired difference (N(0, 1) − oracle). This is the working-assumption effect.

Across the 5 datasets, report the per-item mean and SD of each quantity (descriptive). Report the final NLL, the convergence flag and message, and the iteration count of every fit.

**3a. Item likelihoods.** For each subscore, each dataset and each of the three item sets, evaluate both item objectives at that item set:
- the N(0, 1) composite NLL (`stage1_negloglik`);
- the oracle-latent NLL (`latent_negloglik`);
- the Euclidean norm of each objective's gradient, in the `phi` parameterisation.

This gives a 3 × 2 table per subscore and dataset. It covers the "item likelihood and gradient" reading of D-2.

**4. Downstream fixed-ψ quantities** at ψ_true (§6 D-2, D-4). Evaluate the stage-2 objective, Laplace with Q* = 15 (`m2_quad`, reported-evaluation context per doc 25 P2), for each of the three item sets I ∈ {true, N01, oracle}:
- ℓ(ψ_true; I) and the exact gradient g_I = ∇(−ℓ)(ψ_true; I);
- the Hessian H = `fd_hessian` of −ℓ at ψ_true with the true items. One per dataset; this is the Step 4 Hessian routine at a different point;
- the first-order implied shift Δψ_I = −H⁻¹ g_I and, using the Step 4 standard errors of that dataset, Δψ_I / SE;
- the contrasts Δψ_N01 − Δψ_true and Δψ_oracle − Δψ_true. These are the first-order ψ shifts attributable to the estimated items, attributed to the N(0, 1) assumption versus to sampling;
- ℓ differences between the three item sets at ψ_true;
- the ND status, fallback-patient count, P1 triggers and non-finite counts of every evaluation.

**Reported-evaluation context, with gradient.** Every stage-2 evaluation in §5.4, including each finite-difference point of H, uses the same procedure:
- per patient chunk (the `Objective` chunking, 64 patients per chunk), `group_laplace` at Q = 15 from a cold mode start b = 0, with its exact gradient;
- every patient whose returned Hessian is not ND gets its value **and gradient** from `group_agq` at G = 5, Q = 15 with `gradient = "exact"` (doc 27), starting from that patient's Laplace mode. This is the `evaluate_reported` rule with the doc 27 gradient added;
- every such fallback invocation is recorded (evaluation, chunk, patient count);
- a P1 budget failure in a chunk is recorded as a failure, and that evaluation's value is reported as undefined;
- at ψ_true with the true items, the summed value is also compared with `evaluate_reported` and with the plain optimiser-context `Objective`. Any difference is reported (expected: summation order only, or zero), not acted on.

**Formula and sign convention for the first-order shift.** Let f_I(ψ) = −ℓ(ψ; I), g_I = ∇f_I(ψ_true) and H = ∇²f_true(ψ_true), the FD Hessian (h = 1e-4) with the true items.
- Δψ_I = −H⁻¹ g_I is one Newton step from ψ_true on f_I, with the curvature taken from the true-item objective.
- It approximates ψ̂_I − ψ_true, where ψ̂_I is the stage-2 MLE with items I. **Positive means ψ̂_I would lie above the truth.**
- Δψ_true approximates the sampling deviation of the fixed-item MLE (Step 4). It is reported beside the Step 4 ψ̂ − ψ_true, as a check of the first-order approximation.
- z-units: Δψ_I / SE_Step4. A supplementary column uses SE₀ = √diag(H⁻¹) at ψ_true.

**Limitations of the shift.** It is a first-order (one-step) approximation:
- it ignores the change of curvature between item sets and along the path;
- it is inexact where f is far from quadratic, for example near the D or Σ_ε boundaries;
- it uses the parameterisation of ψ (log-Cholesky), so it is not invariant to reparameterisation;
- it says nothing about interval width or coverage. It is not a substitute for refitting and is not a recovery result.

**No stage-2 optimisation is run** (§6 D-2). No ψ̂, interval or coverage is produced.

## 6. Reviewer decisions (2026-10-19, before computation)

| # | gap | decision |
|---|---|---|
| **D-1** | the true population latent distribution per visit | **Visit-specific N(M_ij, K_ij,jj)**, following the detailed Rev 1 definition, with its quantities derived from the known synthetic data-generating truth (ψ_true on the stage-1 scale). It is not replaced by a shared mixture across visits |
| **D-2** | the downstream fixed-ψ quantities | At ψ_true, report the item likelihood and gradient under the true items, the stage-1 N(0, 1) fit and the true-latent (oracle) fit (§5.3a for the item objectives; §5.4 for the stage-2 likelihood and gradient). Also report the implied first-order shift in SE units, with its formula, sign convention and limitations stated (§5.4). **No refit of stage 2 or of the full pipeline** |
| D-3 | starting values | **One start**, identical to the corresponding pilot N(0, 1) fit (`initial_items`) |
| D-4 | stage-2 evaluation settings | The Step 3 primary configuration, **Q = 15**, with the specified **G = 5** fallback for non-ND patient modes. **Every fallback invocation is recorded.** The frozen configuration is unchanged |
| D-5 | pilot reproduction | **Exact equality** for the pilot N(0, 1) item-fit records (the six metrics per dataset in `step1_recovery.json`). A mismatch stops the run and is reported. No tolerance is introduced after seeing the result |

## 7. Seeds

| use | seed |
|---|---|
| datasets | `[42, 45, 1, r]`, r = 0…4 (unchanged) |
| item starts | none drawn (`starts = 1` uses the deterministic `initial_items`; the tags 100 + r and 200 + r are passed only for path identity) |
| stage-2 evaluations | deterministic (quadrature, cold mode starts at b = 0) |

No other randomness is used.

## 8. Acceptance criteria and interpretation

**None.** Per Rev 2 §5 and R4, S1-a is **reported only**:
- it has no pass/fail;
- no threshold is applied to its outputs;
- it authorises nothing and selects nothing;
- no parameter, setting, item set or S1-b option is chosen from it.

Step 6 proceeds, or does not, only on the user's authorisation and its own frozen criterion.

The report will describe magnitudes and directions only. It will not use the words "adequate", "acceptable" or "biased enough to …".

## 9. Stopping conditions (integrity only)

The job computes nothing, or stops at the stated point, if any of these occurs:

1. **At job start:** any frozen-record hash, code hash (against the Step 4 baseline and the pilot provenance), panel MD5 or input-file hash differs.
2. **Before any fit:**
   - the new-function unit tests (§11) fail;
   - the dataset identity check fails. That check is the serial `MQ.negloglik` on dataset 0, true items, Q = 10, at the recorded ψ of `experiments/20261017T233307Z_stage2_m2_exact_agq/bench/S2_laplace_fit.json`. It must equal the recorded 57022.74225348176 exactly; this is the Step 4 identity check, unchanged.
3. **During the N(0, 1) fits:** a recomputed metric differs from the pilot `step1_recovery.json` `items[r]` (D-5). The run stops before the oracle fits.
4. **At any point:** a non-finite objective or gradient. The run stops, and the event is reported for that fit.

**Not stopping conditions** (reported only): non-convergence of an item fit; a non-ND patient (the fallback is used); a P1 budget trigger at a reported evaluation (reported as a failure for that patient); a non-PD H (Δψ is then reported as undefined for that dataset).

## 10. Cost and the fallback issue

**Item fits.**
- The 20 fits (5 datasets × 2 subscores × 2 latent models) each use 4,295 visits × 61 nodes. They are expected to take ≤ 1 min each serially and run in parallel across processes.
- The pilot's whole step 1, including 5 legacy stage-2 fits, took 647 s.

**Stage-2 evaluations.**
- 3 × (ℓ + gradient) per dataset at about 1.4 s each.
- One `fd_hessian` per dataset at about 54 s (Step 4 measurements: `lik_grad_s` 1.42 s, `hessian_s` 54 s).

**Total:** expected ≤ 30 min wall time on one exclusive node, so ≤ 0.5 node-hours. This is consistent with Rev 1's "minutes".
- Execution: node002 (Xeon Gold 6326, currently idle), 1 exclusive node, 32 workers, single-threaded BLAS, through `submit.sh`. If node002 is busy, the job waits.
- Accounting: AllocNodes × ElapsedRaw / 3600.
- No runtime gate applies.

**Unresolved fallback issue (doc 28; deferral in the doc 31 §0.1 decision).**
- The issue is that *optimising* with the G = 5 AGQ fallback is unaffordable (83–104 min per fit).
- Step 5 performs no optimisation with the fallback. The fallback can only be used for individual non-ND patients in reported evaluations at ψ_true, where it costs seconds per patient. Step 4 recorded 0 such patients at its optima.
- **The issue does not affect Step 5.** It remains open for Step 7, under the recorded rule: "Step 7 must stop if the primary model fails V2 and no affordable, accuracy-admissible fallback is available."
- The T3 timing failure (doc 26) and the 0.62 h assumed component of the projections also do not bear on Step 5.

## 11. Files and outputs

**New files only.** No source module is modified.
- `scripts/51_m2a1_s1a_true_latent_items.py`: the driver plus `latent_negloglik`.
- `tests/test_s1a_latent_items.py`, three tests:
  1. with m = 0 and s = 1, `latent_negloglik` equals `grm.stage1_negloglik` (value and gradient, to 1e-12);
  2. the analytic gradient matches central finite differences;
  3. affine invariance: with a common (m, s) and items (a·s, (τ − m)/s), the value equals the N(0, 1) value with items (a, τ).
- The plan, once frozen: `_frozen/34_STAGE2_STEP5_PLAN.{as_frozen_20261019.md,sha256}`.

**Execution directory:** `experiments/<UTC>_stage2_step5_s1a/`. It will contain:
- `00_baseline.json`, `01_plan.sha256`;
- `step5/`:
  - `S0_provenance.json`;
  - `S1_items_dataset{0..4}.json` (all three item sets and the fit records);
  - `S2_item_errors.csv` (per item × dataset × fit);
  - `S3_downstream_dataset{0..4}.json`;
  - `S4_downstream_shift.csv` (Δψ/SE per component × item set × dataset);
  - `summary.json`;
- `jobs_sacct.txt`, `99_post_run_integrity.txt`, `REPORT.md`.

**Smoke check.** A code-path smoke check, as in Step 4, may precede the job. It runs on 1 dataset, the first 60 patients and a reduced maxiter; the identity and reproduction checks are skipped. It is written to `smoke_codepath/` and labelled *not a result*. It is never reported as one.

**Report:** `docs/cluster_phase/35_STAGE2_STEP5_S1A_REPORT_*.md`.

## 12. Exclusions

The following are not run:
- Step 6 (full-pipeline recovery, S1-b);
- Step 7 (real-data B11, the sampler, V5, DIF);
- Step 8 (G1–G7);
- any real-data item or longitudinal fit;
- calibration;
- positive controls;
- evaluation statistics;
- clustering;
- latent-state comparisons;
- any scientific verdict.

Frozen documents, Steps 3 and 4 results and the D1 verdict (NOT REPRODUCED) are unchanged. Nothing is staged, committed or pushed. **After the Step 5 report, stop for review.**
