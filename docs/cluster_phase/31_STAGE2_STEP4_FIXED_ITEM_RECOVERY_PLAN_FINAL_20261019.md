# Stage 2 M2, Step 4: fixed-item recovery. Execution plan (FINAL)

**Date:** 2026-10-19.

**Status.** This is the **final execution plan**, approved by the reviewer on 2026-10-19 with the decisions in §0. It is frozen (SHA-256 in `docs/cluster_phase/_frozen/31_STAGE2_STEP4_PLAN.sha256`, with a read-only copy) **before any computation.**
- §§1–4 below are the reviewed draft text, kept unchanged.
- **Where §0 differs from them, §0 governs.**
- It does not alter any earlier frozen document.

## 0. Reviewer decisions (2026-10-19), binding for this execution

1. **Fallback risk: deferred, explicitly.**
   - Step 4 proceeds with the G = 5 fallback-runtime decision unresolved.
   - The fallback order (G* = 5) and its accuracy criteria are unchanged.
   - **Step 7 must stop if the primary model fails V2 and no affordable, accuracy-admissible fallback is available.**
2. **Q = 15 runtime gate.** Dataset 0's recovery fit measures the actual primary-fit runtime at Q = 15. Before datasets 1–4 run, two checks apply, with limits unchanged:
   - (i) measured fit wall time ≤ 60 min;
   - (ii) projected pilot ≤ 24 node-hours. The projection uses the measured fit, plus 11 Hessians × 38 likelihood-plus-gradient evaluations, with the per-evaluation time measured from dataset 0's Hessian, plus 716 s of observed item fits and 0.62 h of assumed components.

   **If either fails: stop and report; datasets 1–4 are not run.**
3. **Convergence.**
   - Convergence diagnostics are **reported** for every dataset: optimiser status and message, iterations and evaluations, ‖∇‖ (with B11's 0.934 shown for reference only), bound hits, the minimum Hessian eigenvalue.
   - **No new convergence acceptance threshold** is introduced.
   - The frozen coverage rule applies exactly: each of the 19 parameters covered by its 95 % Wald interval in ≥ 4 of 5 datasets.
   - **If a dataset's Hessian is not positive definite (or is singular), none of its parameters counts as covered for that dataset.**
   - All non-convergence and Hessian failures are reported explicitly.
4. **Fallback diagnostics.**
   - If the G = 5 fallback is invoked at a recovery optimum (reported evaluation), the event and its diagnostics are recorded.
   - **There is no Step 4 stopping rule for a non-zero fallback count.**
   - All existing safeguards (P1/P2, absolute 1e-12 line-search tolerance, bounds, options) are preserved.
5. **Initialisation.**
   - The default starting ψ (`m2_grm.default_psi()`), one start per dataset; its exact values are written to the run's provenance.
   - Seeds `[42, 45, 1, r]`, r = 0–4.

**Identity check of the synthetic data:** before any fit, dataset 0 is regenerated. The Q = 10 Laplace objective at the doc 28 Laplace optimum (`…_stage2_m2_exact_agq/bench/S2_laplace_fit.json`, `psi`) must equal its recorded value, 57,022.74225348176. This confirms that the seeds, truth and design are the earlier ones.

## 1. What the frozen documents require

| source | requirement |
|---|---|
| Rev 2 §6 step 4 | "**Fixed-item recovery** with the §2 likelihood: every ψ ≥ 4 of 5. Pre-requisite only." Each step's failure stops the sequence |
| Rev 2 §5 / decision 6 | fixed-item recovery is a **diagnostic pre-requisite**: if it fails, stop. Passing it **never** authorises the gates or any later step. The binding recovery criterion is the full-pipeline recovery (step 6) |
| doc 16 D3 (recovery definitions, unchanged) | 5 datasets, seeds `[42, 45, 1, r]`, r = 0–4, on the real visit design (times and treatment indicator of the 934 patients), from the **a-priori truth** in `scripts/45_m2_pilot.py` (on the stage-1 scale). 95 % Wald intervals on ψ from the finite-difference Hessian. **Pass: every ψ component covered in ≥ 4 of 5 datasets** |
| doc 18 (fixed-item design) | stage 2 only, with item parameters fixed at their generating values; the same datasets and coverage rule |
| Rev 2 §2.5 and doc 25 P1/P2 | mode safeguards apply to recovery fits. At a reported evaluation, non-ND patients use the fallback at the declared G. A non-zero count at a fit optimum is a B11-reported failure |
| Rev 2 §4.2 B0 rule | "Proceed to recovery only if one fit **at the chosen Q** is ≤ 60 min and the projected pilot is ≤ 24 node-hours" |
| Step 3 (doc 30) | Q* = 15 (primary, Laplace over b); fallback order G* = 5 |

## 2. Proposed procedure

1. **Freeze this plan** (SHA-256), and create a new execution directory `experiments/<UTC>_stage2_step4_fixed_item_recovery/` with a baseline record (hashes, git).
2. **New driver** `scripts/50_m2a1_fixed_item_recovery.py`. It reuses `scripts/45_m2_pilot.py` (`load_cohort` for the design only, `truth_on_stage1_scale`, `simulate_groups`, `regroup`) and `m2_quad` unchanged. No other code changes.
3. **For each r = 0, …, 4:**
   - simulate dataset r (seed `[42, 45, 1, r]`). These are the same datasets as the pilot and doc 18; the dataset-0 identity is checked against doc 18's recorded fit;
   - **fit** stage 2 by `m2_quad.fit`: Laplace over b, **Q = 15**, items fixed at truth, default start ψ, unchanged L-BFGS-B options and bounds, P1/P2 active, 32 processes;
   - compute the **Hessian** by central finite differences (h = 1e-4) of the exact gradient (`m2_quad.fd_hessian`, 38 evaluations). Then SE = √diag(H⁻¹) and z = (ψ̂ − ψ_true)/SE; a component is covered if |z| ≤ 1.96. If H is not positive definite or is singular, no component of that dataset counts as covered (doc 18 convention), and this is reported;
   - run a **reported evaluation** (`evaluate_reported`, declared G = G* = 5): count fallback patients and budget triggers.
4. **B0 gate, using dataset 0's fit as the measured "one fit at the chosen Q"** (§4, decision 1). If it takes more than 60 min, or the pilot projection rebuilt from it exceeds 24 node-hours, stop before datasets 1–4.
5. **Criterion:** every one of the 19 ψ components is covered in ≥ 4 of 5 datasets. **If it fails, stop** (Rev 2 §5): step 5 (S1-a) and beyond are not run.

**Reported, not gated:**
- per-dataset convergence (optimiser status, ‖∇‖ against B11's 0.934, bound hits, minimum Hessian eigenvalue);
- penalty and fallback counts;
- z per component;
- σ_T and σ_P on the natural scale (truth 0.4783 / 0.4053);
- a comparison with doc 18's 2n-dimensional Laplace fixed-item recovery (18 of 19; Le11_log 3 of 5, all z negative).

**Data:** synthetic only. The real data supply only the visit times and treatment indicator. **No real item responses are fitted.**

**Expected cost** (estimate; measured on dataset 0):
- per dataset: one Q = 15 fit, about 3.5–5 min at 32 processes (doc 26 scaled estimate 214 s, plus about 30 s of penalty evaluations); the Hessian, about 0.5 min (38 × 0.82 s measured); the reported evaluation, under 0.5 min;
- total: about 25–35 min on node002, about **0.5 node-hours.**

**Not run:** S1-a, S1-b, full-pipeline recovery, real-data B11, the sampler, V5, G1–G7, calibration, positive controls and evaluation statistics. D1 stays NOT REPRODUCED. Nothing is staged, committed or pushed.

## 3. Is Step 4 permitted, given the G = 5 fallback runtime issue?

**What the frozen documents say:**
- Rev 2 invokes the **fallback fit** only when V2 fails: at step 3 (it passed) and at step 7 (V2 at the new ψ̂).
- No frozen text makes the fallback's runtime feasibility a pre-requisite for step 4.
- In step 4 the fallback appears only in **reported evaluations** of individual non-ND patients. Those are a per-patient G = 5 integral, cheap, and not a fit.
- So the frozen protocol does not block step 4.

**What remains unresolved:**
- doc 25 §4 classes the fallback as computationally infeasible under the current budget when the required G fails the runtime gate. G* = 5 does fail it.
- doc 28 §10 said a separate decision would then be needed.
- That decision is still open. Its consequence falls at **step 7**: if the primary fails V2 at the new ψ̂, there is no affordable admissible fallback, and Stage 2 would stop.
- Step 3 passing does not remove this limitation. Proceeding with step 4 is therefore a choice to defer the decision, with that risk recorded.

## 4. Gaps and contradictions found (decisions requested)

1. **The B0 rule requires a *measured* fit at the chosen Q = 15.** Only Q = 10 was measured; Q = 15 is a scaled estimate. **Proposal:** dataset 0's fit is that measurement and acts as the gate (§2.4). It is the same synthetic dataset and design as B0, with items at truth.
2. **The fallback decision is open** (§3). **Proposal:** proceed with step 4 and record the deferral. The alternative is to resolve it first, with a cost-reducing amendment or a budget change.
3. **No convergence rule is stated for recovery fits** (B11's longitudinal criteria are written for the real-data fit). **Proposal:** report them per dataset, but do not gate on them. The coverage rule is the criterion; a non-PD Hessian already means not covered.
4. **Recovery-fit consequence of a non-zero fallback count** (doc 25 P2 writes it for B11 fits). **Proposal:** report it. A dataset whose optimum needs the fallback is flagged, and its coverage is still computed from the Laplace-objective fit.
5. **Start value for recovery fits** (not stated). **Proposal:** the default ψ (`m2_grm.default_psi`), as in the pilot and doc 18. One start per dataset; multi-start is a real-data B11 requirement.
6. **Still open from earlier:** the T3 decision (doc 26: the 10 % extreme-ψ timing criterion failed). It is not a step-4 pre-requisite in the frozen text.

No contradiction was found between Rev 2 §6 step 4, doc 16 D3 and doc 18 about data, seeds, truth or the coverage rule.
