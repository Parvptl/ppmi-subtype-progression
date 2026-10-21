# Stage 2 M2, Step 6: full-pipeline recovery with S1-b. Proposed execution plan

**Date:** 2026-10-19.

> **PROPOSED — NOT FROZEN — NO COMPUTATION AUTHORISED.**
> This document changes no specification, threshold, seed or frozen record. No driver or test has been written. No model has been fitted, no synthetic data generated and no job submitted. On explicit approval, and after the reviewer resolves §13, a FINAL version is written as a new document and frozen (SHA-256 plus a read-only copy under `_frozen/`) before any code is written.

**Notation.** **[F]** marks a frozen requirement, quoted or paraphrased with its source. **[R]** marks my implementation recommendation, which needs approval. **[G]** marks a gap that the frozen documents do not resolve; each is listed in §13.

---

## 1. Governing documents and their precedence

| doc | file | status | SHA-256 | role for Step 6 |
|---|---|---|---|---|
| 16 | `16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.md` | frozen spec | `c2eace18…bbc` | B5 (two-stage estimation, item uncertainty ignored), B11 (recovery row), **D3 (recovery definition)** |
| 21 | `21_STAGE2_DOC16_AMENDMENT1_PROPOSED_REV2_20261017.md` (Rev 2) | frozen amendment, **governing** | `c53c015f…267` | §0 (recovery definitions unchanged), **§5 (S1-b)**, **§6 step 6**, §2.5 (mode safeguards apply to "the recovery checks"), §4.2 (B0 rule), R4 |
| 20 | `20_STAGE2_DOC16_AMENDMENT1_PROPOSED_20261017.md` (Rev 1) | superseded, unfrozen record | `c0209941…bd` | the more detailed S1-b wording (§5). Used only where Rev 2 is silent, and every such use is flagged |
| 25 | `25_STAGE2_AMENDMENT1_REV2_ADDENDUM_A_FINAL_20261017.md` | frozen addendum | `439cf084…` | P1, P2 (optimiser and reported contexts), F1, runtime accounting |
| 27 | `27_STAGE2_EXACT_AGQ_GRADIENT_AMENDMENT_PROPOSED_20261018.md` | frozen | `17ca7a61…` | the exact AGQ gradient (fallback; reported evaluations only here) |
| 29 / 30 | Step 3 plan (frozen `feb125b0…`) and report | | | **Q\* = 15**; G\* = 5; V2 passed |
| 31 / 32 | Step 4 plan (frozen `830ce768…`) and report | | | fixed-item recovery passed. The §0 decisions in 31 bind **Step 4 only** |
| 34 / 35 | Step 5 plan (frozen `9b514b25…`) and report (`7f9f56c2…`) | | | S1-a, diagnostic only. **Authorises nothing** (Rev 2 R4) |

**Precedence.**
- Doc 16, as amended by Rev 2, Addendum A and doc 27, governs.
- Rev 1 is a superseded record.
- The step plans (29, 31, 34) govern only the execution of their own step. Their reviewer decisions do **not** carry over to Step 6 unless re-approved here (§13).
- A later document number does not imply precedence.

## 2. State verified (read-only, 2026-10-19)

**Repository:** HEAD `3a860ca`; 0 tracked changes; 0 staged.

**Frozen locks:** `sha256sum -c docs/cluster_phase/_frozen/*.sha256` reports OK for docs 16, 21, 25, 27, 29, 31 and 34.

**Code:** `m2_quad.py` `50f164c6…`, `m2_grm.py` `585c878b…`, `grm.py` `1584bc62…`, `scripts/45_m2_pilot.py` `a25e1503…`. These are unchanged against the Step 4 and Step 5 baselines and the pilot provenance. Step 5's own files are unchanged since submission: `scripts/51_m2a1_s1a_true_latent_items.py` `517c4cc1…` and its tests.

**Step 4 outputs** (`experiments/20261018T184713Z_stage2_step4_fixed_item_recovery/step4_run2/`): unchanged (`summary.json` `00433150…`; `S1_dataset{0..4}.json` as in the Step 5 baseline). No file has been modified since the Step 5 baseline.

**Step 5 outputs** (`experiments/20261018T215102Z_stage2_step5_s1a/step5/`): unchanged since the post-run integrity record (`summary.json` `5ccc690c…`).

**Panel and D1:** panel MD5 `fb038ddb…`. D1 verdict (doc 13): **NOT REPRODUCED**, untouched.

## 3. Frozen requirements for Step 6

| source | requirement [F] |
|---|---|
| Rev 2 §6 step 6 | "**Full-pipeline recovery** with S1-b: every ψ ≥ 4 of 5, **unchanged criterion.**" Each step's failure stops the sequence. "It is reported, and nothing after it runs" |
| Rev 2 §5, S1-b | "the only stage-1 repair in this amendment. Exactly one refit iteration: N(0, 1) stage 1 → stage 2 (§2) → items re-estimated under the fitted population latent mixture → stage 2 re-fit." "Accepted **only if** the **unchanged full-pipeline recovery** (doc 16 D3: 5 datasets, every ψ ≥ 4 of 5) passes with this pipeline." S1-c (joint estimation) needs a separate amendment, if S1-b fails |
| Rev 1 §5, S1-b (detail) | "(1) N(0, 1) stage 1 as now; (2) stage 2 with the §2 likelihood; (3) re-estimate items by marginal maximum likelihood under the **fitted** population latent mixture from (2); (4) re-fit stage 2 with the refitted items. Pre-declared: exactly one iteration, no iteration until convergence." Cost "about 2× the stage-2 cost". "S1-a and S1-b use only synthetic data and real item responses. Neither uses labels." |
| Rev 2 §5 and R4 | fixed-item recovery (Step 4) is a pre-requisite only. S1-a is diagnostic only and authorises nothing. **Full-pipeline recovery is the binding recovery criterion** |
| Rev 2 §0 and doc 16 D3 | "**Simulation recovery (B11).** 5 datasets (seeds `[42, 45, 1, r]`) on the real visit design (times and treatment indicator of the 934 patients), from **a-priori** truth values fixed in `scripts/45_m2_pilot.py` before any real-data fit. The pipeline is the full two-stage one (item fit, then longitudinal fit). 95 % Wald intervals on ψ come from the finite-difference Hessian. Pass: every component of ψ is covered in ≥ 4 of 5 datasets. Item-parameter recovery is reported (correlation and maximum error) but not gated." Rev 2 §0 adds: truth "on the stage-1 scale" |
| doc 16 D3, longitudinal fit | the Hessian is "the central finite difference (step 1e-4) of the analytic gradient" |
| doc 16 D3, bounds | means ±10; log-diagonals [−8, 3]; off-diagonals ±5. "A bound hit means the parameter lies within 1e-6 of a bound" |
| doc 16 B5 | stage-2 parameter uncertainty is computed with "item parameters … held at their stage-1 values. Ignoring that item uncertainty is a stated limitation" |
| Rev 2 §2.5 and doc 25 P1/P2 | the safeguards apply to "the recovery checks". **Optimiser context:** a non-ND patient mode, or a P1 budget trigger, makes ψ infeasible (penalty 1e12). **Reported context:** non-ND patients are evaluated by AGQ at the declared G = 5. "A non-zero fallback count, or any budget trigger, at a reported fit optimum **fails that fit's B11 longitudinal check**" |
| Rev 2 §4.2 (B0 rule) | "Proceed to recovery only if one fit at the chosen Q is ≤ 60 min and the projected full-pilot cost … is ≤ 24 node-hours" |
| Step 3 (doc 30) | Q\* = 15 (Laplace over b, per-visit 15 × 15 GH over ε); G\* = 5 |
| doc 31 §0.1 (recorded deferral) | "**Step 7 must stop if the primary model fails V2 and no affordable, accuracy-admissible fallback is available.**" |

### The exact frozen recovery criterion

> For each synthetic dataset r = 0…4 (seeds `[42, 45, 1, r]`, a-priori truth on the stage-1 scale, real design):
> - run the full two-stage pipeline with S1-b;
> - compute the 95 % Wald interval ψ̂_k ± 1.96·SE_k for each of the 19 ψ components, with SE from the central finite-difference Hessian (step 1e-4) of the analytic stage-2 gradient at the final ψ̂, items held at their estimated values.
>
> **Step 6 passes if and only if every one of the 19 components is covered (truth inside its interval) in at least 4 of the 5 datasets.**

The frozen text does **not** specify what happens with a non-PD or singular Hessian, a failed or non-converged fit, a boundary estimate, a missing or non-finite interval, a fallback or budget event at the optimum, or an execution failure. These are decisions E-6 to E-8 in §13.

## 4. Purpose: what Step 6 establishes that Steps 4 and 5 do not

**Step 4** fixed the items at their generating values and tested stage 2 alone.

**Step 5 (S1-a)** used the unknown truth (an oracle latent distribution) and estimated no ψ. It was diagnostic only and, by Rev 2 R4, authorises nothing. Its results are **not** used anywhere in this plan.

**Step 6** is the first test of the procedure that will actually be applied to real data:
- the items are estimated from the responses;
- the stage-1 working assumption, θ ~ N(0, 1), is corrected once by S1-b;
- ψ is estimated with the estimated items held fixed.

Coverage then reflects:
- the stage-2 approximation;
- the item estimation error;
- the N(0, 1) assumption after one S1-b iteration;
- the fact that the Wald intervals ignore item uncertainty (doc 16 B5).

It is the **binding** recovery criterion (Rev 2 §5). The earlier pipeline without S1-b, on the old likelihood, failed it: 16 of 19 components (doc 17 O3; pilot record: β0_P 3/5, Le00_log 2/5, Le11_log 0/5).

## 5. Synthetic datasets [F]

| item | specification | source |
|---|---|---|
| number | 5 | doc 16 D3 |
| seeds | `np.random.default_rng([42, 45, 1, r])`, r = 0…4. The same datasets as the pilot, doc 18, Step 4 and Step 5 | doc 16 D3 |
| generating truth | `TRUTH_ITEMS` and `TRUTH_LONG` in `scripts/45_m2_pilot.py`, re-expressed by `truth_on_stage1_scale(groups)` (pooled visit-level θ mean 0, variance 1); σ_T 0.4783, σ_P 0.4053 | doc 16 D3; Rev 2 §0 |
| design | the 934 patients' real visit times and treatment indicator (x = 1 if `PDTRTMNT == 1`, else 0), through `load_cohort` (design only), 4,295 visits; 11 tremor and 5 PIGD items | doc 16 D3, B2 |
| simulation | `simulate_groups` (b ~ N(0, D), ε ~ N(0, Σ_ε), GRM item draws) | `scripts/45_m2_pilot.py`, unchanged |
| real responses and labels | **not used** | doc 16 D3; Rev 1 §5 |

**No new random numbers are drawn**, provided the recommended single-start convention (E-5) is approved.

## 6. The pipeline: what is estimated, held fixed and compared

For each dataset r and subscore s ∈ {T, P}:

| stage | what is estimated | held fixed | how |
|---|---|---|---|
| **(1) stage 1, N(0, 1)** | items (a, τ) for T and for P: 55 parameters | θ ~ N(0, 1) per visit | `fit_items_multistart(Y, tag, starts = 1)`, the pilot path (`grm.stage1_negloglik`, 61-node GH, L-BFGS-B, maxiter 3000) [F: doc 16 B5; start convention E-5] |
| **(2) stage 2** | ψ̂⁽¹⁾ (19) | items from (1) | `m2_quad.fit`, Laplace over b, **Q = 15**, P1/P2 active, unchanged L-BFGS-B options and bounds, 32 processes [F: Rev 2 §2; Step 3] |
| **(3) S1-b item refit** | items (a, τ) for T and for P | the "fitted population latent mixture" at ψ̂⁽¹⁾ | marginal ML with per-visit quadrature [F: Rev 2 §5; the exact latent distribution is gap **E-1**] |
| **(4) stage 2 refit** | **ψ̂⁽²⁾ (19): the estimate assessed** | items from (3) | as (2) [F] |
| (5) Hessian | H at ψ̂⁽²⁾ | items from (3) | `m2_quad.fd_hessian` (central, h = 1e-4, 38 likelihood-plus-gradient evaluations) [F: doc 16 D3] |
| (6) reported evaluation | fallback and budget counts at ψ̂⁽²⁾ | | `evaluate_reported(ψ̂⁽²⁾, …, Q = 15, G_fallback = 5)` [F: doc 25 P2(b)] |

**Exactly one S1-b iteration. Nothing iterates to convergence** [F: Rev 1 §5].

**Comparison with the truth.**
- **Binding:** for ψ̂⁽²⁾, z_k = (ψ̂⁽²⁾_k − ψ_true,k)/SE_k, where SE = √diag(H⁻¹). Covered means |z_k| ≤ 1.96.
- **Reported, not gated:** item recovery for the items from both (1) and (3) against it_true, by correlation and maximum error [F: doc 16 D3], plus mean error and RMSE [R].
- **Reported, not gated:** σ_T and σ_P on the natural scale.
- **Reported, not gated:** ψ̂⁽¹⁾ (decision E-8).

**Stage-1 scale.** ψ_true and it_true are on the stage-1 scale. S1-b's step (3) anchors the latent scale to the fitted ψ̂⁽¹⁾. I recommend that ψ̂⁽²⁾ is compared with ψ_true as it is, **without re-standardisation** (decision E-2; Rev 1 §5: "doc 16 B4/B5 already fix the scale by stage 1").

## 7. Recovery criterion and the cases it does not cover

The criterion is reproduced in §3 [F]. The cases needing a rule, with my recommendation, are listed below and decided in §13.

| case | frozen text | recommendation [R] |
|---|---|---|
| H not positive definite or singular at ψ̂⁽²⁾ | silent (doc 31 §0.3 decided it for Step 4 only) | **no component of that dataset is covered** (the doc 18 / Step 4 convention) |
| an SE that is NaN or infinite | silent | that component is **not covered** |
| optimiser not converged, or maxiter reached, in (2) or (4) | silent; B11 convergence criteria are real-data checks | coverage computed as usual from the final ψ̂⁽²⁾ and H; non-convergence **reported**; **no new threshold** (as Step 4 §0.3) |
| boundary estimate (within 1e-6 of a bound, doc 16 D3) | silent | Wald interval computed as usual and **flagged**; no special rule |
| item fit not converged, or a ∉ [0.05, 20] | B11 item checks are real-data checks; item recovery "not gated" | **reported only** |
| non-zero fallback count or budget trigger at ψ̂⁽²⁾ (reported evaluation) | doc 25 P2(b): "fails that fit's B11 longitudinal check" | recorded as that fit's **B11 longitudinal-check failure**. **Coverage is not altered**, because the coverage rule does not use B11 checks. No new stopping rule (as Step 4 §0.4) |
| execution error, or a dataset not completed | silent | **stop the run**. Step 6 is reported **not passed (incomplete)**. Any rerun only after review, with a defect note and the defective output kept (the Step 3 and Step 4 practice) |
| a P1/P2 penalty during the optimiser path | doc 25 P1/P2 | part of the frozen optimiser semantics; counts are reported |

## 8. Numerical settings and the fallback issue

**Settings [F]:**
- Laplace over b, per-visit Q × Q Gauss–Hermite over ε, **Q = 15** (Step 3);
- P1 (10 consecutive non-ND iterations, or 200 halvings);
- P2 (eigenvalue ND test; optimiser-context penalty);
- absolute line-search acceptance tolerance 1e-12;
- the D3 bounds, and L-BFGS-B options unchanged (`m2_quad.fit`, maxiter 1000);
- 61-node GH for the item fits;
- the stage-2 Hessian by central FD, h = 1e-4;
- G = 5 AGQ (exact gradient, doc 27) only for non-ND patients **in reported evaluations**.

**The unresolved G = 5 fallback runtime issue** (doc 28: 83–104 min per fallback fit; projected fallback pilot 37.4 node-hours).
- It concerns **optimising** with the fallback.
- Step 6 never optimises with the fallback: the primary Laplace objective is used throughout, and P2 makes non-ND ψ infeasible in the optimiser.
- The fallback can enter only at reported evaluations, per patient, at seconds per patient. Step 4 and Step 5 recorded 0 such patients in 5 + 205 evaluations.
- Step 6 therefore respects the recorded rule, which belongs to **Step 7**: "Step 7 must stop if the primary model fails V2 and no affordable, accuracy-admissible fallback is available."
- Nothing in Step 6 resolves, relaxes or anticipates that rule. Step 6 runs no V2.

## 9. Initialisation and optimisation [F where stated, otherwise R]

**Item fits, steps (1) and (3).** L-BFGS-B from `grm.initial_items(Y)`, one start (the pilot recovery convention, `starts = 1`). Step (1) must **reproduce the pilot's recorded item fits exactly** (as Step 5, D-5).
- For step (3), the start is decision **E-3**: recommended `initial_items`, identical to every other item fit, rather than a warm start from (1).

**Stage-2 fits, steps (2) and (4).**
- Default ψ (`m2_grm.default_psi()`), cold mode state, one start. This is the pilot and Step 4 convention (Step 4 §0.5, Step 4 only).
- For step (4), the start is decision **E-4**: recommended default ψ, rather than a warm start from ψ̂⁽¹⁾, so that (2) and (4) differ only in the items.

**Multistart.** Doc 16 D3 defines 3-start rules under "Item fit (B11)" and "Longitudinal fit (B11)". The pilot recovery used one start per fit, and doc 25's accounting counts "3 real starts, 10 recovery fits", that is, one per recovery fit. Whether recovery fits need the B11 multistarts is decision **E-5**. My recommendation is one start: it is the pilot convention, and multistarts would add new seeds and about 3× the cost.

**No other fitting strategy is introduced.** In particular:
- no iteration of S1-b beyond one;
- no joint estimation (S1-c);
- no use of Step 5 results;
- no tuning of Q, G, bounds or tolerances.

## 10. Execution sequence and stopping conditions

1. **On approval:** write the FINAL plan (doc 37) with the §13 decisions, and freeze it.
2. **Write** the driver `scripts/52_m2a1_full_pipeline_recovery.py` and tests `tests/test_s1b_pipeline.py`. New files only; no existing file modified. The S1-b refit reuses `latent_negloglik` from `scripts/51_…` by hash-pinned import (decision E-12).
3. **Smoke code path** (labelled *not a result*): 1 dataset, 60 patients, reduced maxiter.
4. **Job** on node002 (Xeon Gold 6326, 1 exclusive node, 32 workers, single-threaded BLAS), through `submit.sh`. If node002 is busy, the job waits.
   - **(a)** Preflight (§11.1), run in the job. **Any failure: nothing is fitted; stop.**
   - **(b)** Step (1) for all 5 datasets. **The pilot reproduction must be exact, or stop before any stage-2 fit.**
   - **(c)** For r = 0…4: steps (2)–(6), sequentially.
   - **(d)** After dataset 0: the **runtime gate** (decision E-9). If it fails, stop; datasets 1–4 are not run.
   - **(e)** Non-finite objective or gradient at an accepted point, or an execution error: stop (E-7).
   - **(f)** Evaluate the criterion (§3).
5. **Post-run integrity** (§11.2), report (doc 38), and **stop for review.**

**Outcome rules [F: Rev 2 §5, §6]:**
- **If the criterion fails:** Step 6 fails, S1-b is not accepted, and the sequence stops. Steps 7 and 8 are not run. Any S1-c would need a separate amendment.
- **If it passes:** S1-b is accepted as the pipeline. Step 7 still requires the user's explicit authorisation. Gates G1–G7 (step 8) additionally require steps 3, 4, 6 and 7 to pass.

## 11. Integrity checks

### 11.1 Preflight checklist (completed inside the job before any fit; each recorded in `S0_provenance.json` / `summary.json`)

- [ ] Frozen locks: docs 16, 21, 25, 27, 29, 31, 34 and the FINAL Step 6 plan all match.
- [ ] Code hashes equal the Step 4 / Step 5 baselines and the pilot provenance (`m2_quad`, `m2_grm`, `grm`, `45_m2_pilot`, `m2_gates`, `raw_scores`, `subtype`), and `scripts/51_…` equals `517c4cc1…`.
- [ ] Panel MD5 `fb038ddb…`.
- [ ] Input hashes:
  - pilot `step1_recovery.json` and `provenance.json`;
  - Step 4 `summary.json` `00433150…`;
  - identity reference `…_stage2_m2_exact_agq/bench/S2_laplace_fit.json`.
- [ ] New unit tests pass (run inside the job):
  - the S1-b latent construction at a given ψ equals the Step 5 construction at ψ_true;
  - pooled moments at ψ_true are 0 and 1;
  - the S1-b item refit uses exactly `latent_negloglik`;
  - the step sequence calls each stage exactly once;
  - coverage logic for the non-PD, NaN and bound cases.
- [ ] `truth_on_stage1_scale` ψ_true equals the pilot record.
- [ ] Dataset identity: serial `MQ.negloglik` on dataset 0, true items, Q = 10, at the recorded ψ equals 57022.74225348176 exactly.
- [ ] Seed list `[42, 45, 1, r]`, r = 0…4, written to provenance. No other RNG is used under E-5 (recommended).
- [ ] After step (1): the pilot `items[r]` six metrics are reproduced exactly (30/30).
- [ ] git HEAD and working-tree state recorded; 0 staged.

### 11.2 After computation

- Frozen locks re-checked.
- Code, input and prior-output hashes unchanged against the baseline. No file in the pilot, Step 3, Step 4 or Step 5 directories, or in `docs/`, `src/`, `scripts/` or `outputs/`, modified during the run.
- SLURM accounting (`sacct`: AllocNodes, ElapsedRaw).
- git state: 0 staged; no tracked change.

## 12. Runtime

| component | basis | value |
|---|---|---|
| stage-1 N(0, 1) item fits, step (1) | **measured** (Step 5: 10 fits in parallel; T 50–53 s, P 15–20 s each) | about 1 min wall, once |
| stage-2 fit at Q = 15, default start | **measured with true items** (Step 4: 179, 209, 203, 244, 181 s; 80–108 iterations). With estimated items: **assumed** similar | 3–4 min per fit; 2 per dataset |
| S1-b item refit, step (3) | **measured for the analogous oracle fit** (Step 5: T 55–73 s, P 17–24 s, run in parallel). At ψ̂⁽¹⁾: **assumed** similar | about 1.2 min per dataset |
| FD Hessian at ψ̂⁽²⁾ (warm mode state) | **measured** (Step 4: 54–55 s; cold-start Hessians in Step 5: 93–103 s) | about 1 min per dataset |
| optional Hessian at ψ̂⁽¹⁾ (E-8) | the same | about 1 min per dataset |
| reported evaluations (2 per dataset) | **measured** (Step 4: about 5 s) | under 0.5 min |
| **per dataset** | estimate | **about 10–13 min** |
| **total job** (5 datasets plus setup and identity check) | estimate | **about 55–70 min ≈ 0.9–1.2 node-hours** (1 node) |
| smoke check | estimate | under 0.1 node-hours |
| margin (if fits with estimated items take up to 1.5× longer) | **assumption** | up to about 1.7 node-hours |
| with B11 multistarts (E-5, not recommended) | estimate | about 3× the fits, about 3–3.5 node-hours |

**B0 rule.** The rule (one fit ≤ 60 min; projected pilot ≤ 24 node-hours) was met at Q = 15 in Step 4: a 179 s fit, and a projected pilot of 1.88 node-hours. That projection included 0.62 h *assumed* for the S1-b refits, which Step 6 will now **measure**.

**Fallback.** No fallback-optimisation cost arises (§8).

## 13. Decisions requiring approval

| # | gap [G] | options | recommendation [R] |
|---|---|---|---|
| **E-1** | what "the fitted population latent mixture" (Rev 2 §5; Rev 1: "from (2)") is in step (3) | (a) per subscore, the **visit-specific** θ_v ~ N(M_ij(ψ̂⁽¹⁾), K_ij,jj(ψ̂⁽¹⁾)), using each visit's time and treatment, per-visit composite likelihood, 61-node GH (the Step 5 D-1 construction with ψ̂⁽¹⁾ in place of ψ_true); (b) one **shared** mixture density, (1/V) Σ_v N(m_v, s_v²), for every visit | **(a).** It matches the Rev 1 wording used for S1-a, which you approved as D-1 for Step 5. It is the same construction with the fitted ψ, and it reuses tested code. This is consistency of definition only; no Step 5 *result* is used |
| **E-2** | scale after S1-b | (a) compare ψ̂⁽²⁾ with ψ_true as is; (b) re-standardise | **(a)**, with no re-standardisation. The pooled moments of the fitted mixture are reported |
| **E-3** | start of the step (3) item refit | (a) `initial_items` (one start); (b) the step (1) estimates | **(a)** |
| **E-4** | start of the step (4) stage-2 refit | (a) default ψ, cold modes; (b) warm from ψ̂⁽¹⁾ and its modes | **(a)** |
| **E-5** | number of starts in recovery fits | (a) one per fit (pilot convention; doc 25 accounting); (b) the B11 rules: 3 item starts (seeds `[42, 45, 5, tag, k]`), 3 longitudinal starts (seeds `[42, 45, 6, k]`), lowest NLL reported | **(a)**. Under (b), the 0.1 agreement checks would be reported, not gated |
| **E-6** | coverage edge cases (§7): non-PD or singular H; NaN SE; non-convergence; bound hits; item-fit problems | as tabulated in §7 | **as §7:** non-PD or singular → nothing covered; NaN SE → not covered; non-convergence and bounds → reported, no new rule; item problems → reported |
| **E-7** | execution error or an incomplete dataset | (a) stop; Step 6 not passed (incomplete); rerun only after review; (b) continue with the other datasets | **(a)** |
| **E-8** | report the intermediate ψ̂⁽¹⁾ (the N(0, 1)-only pipeline) | (a) report ψ̂⁽¹⁾, its Hessian, z and coverage counts, labelled **non-binding**; it cannot pass Step 6 or replace S1-b; (b) report ψ̂⁽¹⁾ only, no Hessian; (c) omit | **(a)**. It shows S1-b's effect at about 1 min per dataset. Rev 2 binds the S1-b pipeline, so it selects nothing |
| **E-9** | runtime gate | (a) after dataset 0: each stage-2 fit ≤ 60 min and the pilot projection rebuilt from measured components ≤ 24 node-hours; otherwise stop (the limits of Rev 2 §4.2, unchanged); (b) no gate (B0 already met in Step 4) | **(a)** |
| **E-10** | fallback or budget at ψ̂⁽²⁾ (doc 25 P2(b)) | (a) record as the fit's B11 longitudinal-check failure, coverage unaltered, no stop; (b) treat that dataset as not covered | **(a)**. The frozen coverage rule does not reference B11 checks |
| **E-11** | pilot reproduction tolerance for step (1) | exact, or a tolerance | **exact** (as Step 5 D-5); a mismatch stops before stage 2 |
| **E-12** | code reuse | (a) import `latent_negloglik` from `scripts/51_…` (hash-pinned, unchanged); (b) copy it into the new driver | **(a)** |

**Not decisions** (frozen, restated for clarity):
- 5 datasets, seeds `[42, 45, 1, r]`;
- the a-priori truth on the stage-1 scale;
- Q = 15;
- G = 5 for reported evaluations;
- one S1-b iteration;
- Wald intervals from the FD Hessian (h = 1e-4), item uncertainty ignored;
- ≥ 4 of 5 for all 19 components;
- failure stops the sequence.

## 14. Outputs

**Execution directory:** `experiments/<UTC>_stage2_step6_full_pipeline_recovery/`:
- `00_baseline.json`, `01_plan.sha256`, `02_driver_sha256.txt`;
- `step6/`:
  - `S0_provenance.json`;
  - `S1_dataset{r}.json` (all four stages; fit logs; item sets; ψ̂⁽¹⁾ and ψ̂⁽²⁾; H, eigenvalues, SE, z, coverage; reported evaluations);
  - `S2_coverage_table.csv` (19 × 5: truth, estimate, SE, interval, z, covered, PD, bound flag);
  - `S3_item_recovery.csv` (per item × dataset × {stage 1, S1-b});
  - `S4_convergence.csv`;
  - `summary.json`;
  - `figures/`;
- `smoke_codepath/`;
- `jobs_sacct.txt`, `99_post_run_integrity.txt`, `REPORT.md`.

**Report tables:**
1. preflight results;
2. per-dataset convergence for every stage: status, message, iterations, evaluations, ‖∇‖ (B11's 0.934 for reference only), bound hits, penalties by reason, non-ND counts, wall times;
3. the coverage table (19 × 5 z values and covered flags) and **coverage counts against the ≥ 4 of 5 criterion**, with the verdict;
4. the Hessian (PD, min/max eigenvalue);
5. reported-evaluation fallback and budget counts per optimum;
6. σ_T and σ_P on the natural scale;
7. item recovery (stage 1 vs S1-b);
8. E-8, if approved: the ψ̂⁽¹⁾ table, marked non-binding;
9. runtime, measured against the §12 projection;
10. failure accounting (every exception, penalty or edge case and its handling);
11. the run manifest (hashes, seeds, host, CPU, job IDs).

**Plots** (matplotlib, from the saved outputs, in `figures/`):
- z per ψ component and dataset, with ±1.96 bands (ψ̂⁽²⁾; ψ̂⁽¹⁾ if E-8);
- estimated against true discrimination and thresholds, stage 1 vs S1-b;
- σ_T and σ_P estimates with intervals against the truth.

## 15. Interpretation

**A pass establishes** that, on 5 synthetic datasets generated from the a-priori truth at the real visit design, the complete two-stage pipeline with one S1-b iteration gives 95 % Wald intervals (item uncertainty ignored) covering every ψ component in ≥ 4 of 5 datasets. It satisfies the binding recovery pre-requisite for Step 7.

**A failure establishes** that this pipeline does not meet the frozen recovery criterion. S1-b is not accepted, and the sequence stops. Any further repair (S1-c) needs a separate amendment.

**Neither establishes:**
- that the model fits the real PPMI data (Step 7 B11, then the gates);
- how the pipeline performs under truths other than the a-priori one, or under misspecification;
- nominal coverage (5 datasets support only the ≥ 4-of-5 rule; it is not a coverage estimate);
- anything about biological subtypes, labels or D1.

No evaluation statistic, calibration or positive control is involved.

## 16. Not authorised by this document

- Freezing.
- Writing the driver or tests.
- Generating data, fitting, or submitting jobs.
- Step 7 (real-data B11, sampler, V5, DIF) or Step 8 (G1–G7).
- Calibration, positive controls, evaluation statistics.
- Any real-data evaluation.
- Any change to frozen records, source files, Step 4 or Step 5 outputs, or the D1 verdict (**NOT REPRODUCED**).
- Nothing is staged, committed or pushed.
