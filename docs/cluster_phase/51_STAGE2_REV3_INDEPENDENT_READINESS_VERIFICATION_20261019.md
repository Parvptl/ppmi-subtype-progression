# Stage 2 M2 Rev 3: independent readiness verification of the doc 47 dependencies

**Date:** 2026-10-19 (checks run 2026-10-19T17:25Z–17:30Z).

> **PROPOSED — NOT FROZEN — NO COMPUTATION AUTHORISED (by this document).**
> - This document **authorises nothing, freezes nothing and amends nothing.** It is an independent verification that every artefact doc 47 depends on exists, resolves and is unchanged, plus the findings that follow from that.
> - **Step 6 remains FAILED. D1 remains NOT REPRODUCED.**
> - **What was done:** file-existence checks, SHA-256 and MD5 digests, JSON/CSV key and shape inspection, `sha256sum -c` manifest verification, and read-only queries of the SLURM state and the Python environment.
> - **What was not done:** no model was fitted, no synthetic data generated, no numerical check evaluated, no job submitted. No frozen record, historical report or prior manifest was modified. Nothing was staged, committed or pushed. No recovery dataset (seeds `[42, 49, 1, r]`) exists.
> - **Numbering note.** This file was first written as `50_…`. Doc 46 R1.3 reserves **doc 50** for its own change record, so it was renamed to 51 before anything referenced it. Nothing else was renamed.

**Concurrency note (material to reading this document).** Docs 46 and 47, and `scripts/58_m2_rev3_prefreeze_checks.py`, were being edited by another writer *while these checks ran*. The sequence: §§1–5 were computed at 17:25Z against doc 46 R1.2; doc 46 moved to R1.3 at 17:26Z; doc 47 moved to R1.3 and all four freeze locks were written at 17:29Z. **Docs 46 and 47 are now FROZEN (R1.3) and their locks verify** (§6, F-5). §1 records every state observed. The artefact, code, panel, truth, seed and environment digests in §§2–4 concern files that did not change during the window, and were re-confirmed afterwards.

---

## 1. Document integrity

| file | revision | SHA-256 observed | against the record |
|---|---|---|---|
| `46_…_AMENDMENT_REVISED_R1_PROPOSED_20261019.md` | R1.2, at 17:25Z | `e9ad3a837d9ec24496201ded1b6d3f62d31a612ffa695c3799feba1792ed515a` | **matched** doc 49's post-R1.2 hash |
| `46_…` (same path) | **R1.3, at 17:26Z** | `d8d5f99afde1381748788e3d1746532e67f8cb4b93abe270674ae4d1cf9da5f4` | new; R1.3 records the pre-edit hash as `e9ad3a83…`, consistent with the row above |
| `47_…_RECOVERY_TEST_PLAN_REVISED_R1_PROPOSED_20261019.md` | R1.2 | `106457578b65546f304dbea80701ecefbd80d4e447a80d1f812d2bbf5753c83b` | **matches** doc 49's post-R1.2 hash. Unchanged in content, though its mtime moved to 17:10Z |
| `44_…_SCALE_CONVENTION_AMENDMENT_PROPOSED_20261019.md` | — | `6c63de6fd352e029589ab9e358812afd7832db6be0666e34c785a9fb73eeeb96` | **matches** `6c63de6f…`; intact as required |
| `45_…_INDEPENDENT_RECOVERY_TEST_PLAN_PROPOSED_20261019.md` | — | `8e00b4f568ef69f36d0dc77dbe56083cdc0578359316e7d3dd25e1728e094f1f` | **matches** `8e00b4f5…`; intact as required |
| `48_…_R1_1_CHANGE_RECORD_PROPOSED_20261019.md` | — | `1affdc78e13ba1e33ec5296307fa065cc2b0727b303566a20c5dbce68c36786c` | recorded here |

Docs 44 and 45 are intact as the record of the superseded proposals. The R1.2 → R1.3 hash chain on doc 46 is self-consistent.

## 2. Regression references R-a–R-d (doc 47 §8 condition (I))

| check | reference, as cited in doc 47 | resolves | detail |
|---|---|---|---|
| **R-a** ψ̂⁽¹⁾, ψ̂⁽²⁾ bitwise | `…step6_full_pipeline_recovery/step6/S1_dataset0.json`, keys `stage2_fit_1.psi`, `stage2_fit_2.psi` | **yes** | both present, 19 components each; `stage2_fit_1.psi[0] = −0.1448498462767735`, `stage2_fit_2.psi[0] = −0.12633290661813992` |
| **R-b** S(ψ̂⁽²⁾) ≤ 1e-12 | `…step6_diagnostics/d7_readonly/D7_estimates_long.csv`, dataset 0, `psi2_step6`, column `std` | **yes** | header carries `std`; 190 `psi2_step6` rows; dataset-0 `beta0_T` row present (`std = −0.17350025809073663`) |
| **R-c (= N11)** full-chain SEs ≤ 1e-6 rel. | `…step6_diagnostics/d4/D4_dataset0.json`, `se_psi2.full_chain` | **yes** | 19 finite values; first `0.034986786984435934` |
| **R-d** N3, N4 ≤ 1e-5 rel. | computed in the run; no stored reference | **n/a** | now independently informed — see §6, F-2 |

**Manifest verification** of the three reference directories (`sha256sum -c`):

| directory | files checked | OK | failed |
|---|---|---|---|
| `…step6_full_pipeline_recovery/step6/` | 14 | 14 | 0 |
| `…step6_diagnostics/d7_readonly/` | 6 | 6 | 0 |
| `…step6_diagnostics/d4/` | 6 | 6 | 0 |

Doc 47 R-b correctly cites `d7_readonly/`, not the sibling `d7_readonly_run1_defective/` retained under `D7_RUN1_DEFECT_NOTE.txt`.

## 3. Reused code, panel, truth and seeds

| element | required | observed | verdict |
|---|---|---|---|
| reused modules | unchanged from the D4/D7 baseline (doc 47 §2) | all **94** files in `…step6_diagnostics/00_baseline.json → code_sha256` re-hashed: **0 changed, 0 missing** | **match** |
| git HEAD | the baseline's recorded head | recorded `3a860ca8c864e6ef9e23ecbbd6505ccb06b9136b`; current HEAD `3a860ca`; no tracked file modified by these checks | **match** |
| frozen panel | MD5 `fb038ddbdc19ab3368e13940f398e7ba` (`EXPECTED_PANEL_MD5`, `scripts/26_clusterability_extended.py:74`) | `outputs/visit_panel.csv` → `fb038ddbdc19ab3368e13940f398e7ba` | **match** |
| generating truth | the a-priori `TRUTH_ITEMS` / `TRUTH_LONG` of `scripts/45_m2_pilot.py` | that file is one of the 94 unchanged (`a25e1503…`) | **match** |
| seed namespace | `[42, 49, …]` unused in `scripts/`, `src/`, `docs/` | no occurrence outside docs 44–51 | **confirmed** |
| planning verification | `experiments/20261019T160413Z_stage2_rev3_planning_checks/` | present; manifest verifies 3/3; `scripts/56_…`, `scripts/57_…`, `tests/test_rev3_planning_functionals.py` present | **present**; see F-3 |

## 4. Environment

| element | Step 6 record (`step6/S0_provenance.json`) | current `~/envs/ppmi` | verdict |
|---|---|---|---|
| Python | 3.13.16 | 3.13.16 | **match** |
| NumPy | 2.2.4 | 2.2.4 | **match** |
| SciPy | not recorded | 1.15.2 | recorded here |
| threads | `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` | set by the job | carried by the driver |
| workers | 32 | node capacity 64 CPUs | available |
| host / CPU | `node002`, Intel Xeon Gold 6326 @ 2.90 GHz | node002, same hardware | **match** |

The match on Python, NumPy and CPU is what makes the **bitwise** R-a demand reasonable. D4 ran on `master` (login node), so R-c is cross-host — consistent with its ≤ 1e-6 *relative*, non-bitwise tolerance.

## 5. Cluster state (2026-10-19T17:25Z)

| partition | node | state | CPUs | memory |
|---|---|---|---|---|
| normal | node001 | **drain** | 64 | 100 GB |
| normal | node002 | **allocated** (another user) | 64/64 | 100 GB |
| normal | node003 | idle | 64 | 100 GB |
| normal | node004 | idle | 64 | 230 GB |

Queue: `14007` (running, `scaled_prep`, 1 h limit), `14008_[0…]` and `14009` (pending on dependencies, 1 h limits), all user `b112301010`, on/for node002. Nothing of ours queued. Partition limit 1-00:00:00 accommodates the §9 limits (Phase 1 2 h; Phase 2 13.5 h each).

## 6. Findings

**F-1 — node002 is occupied; Phase 1 will queue. (Operational.)**
Doc 47 §9 requires Phase 1 on node002 for the bitwise R-a check. node002 is fully allocated to another user's job chain and node001 is drained, so the normal partition holds exactly the two spare nodes (003, 004) that Phase 2 later needs — no margin. A Phase 1 submission waits; it does not fail. No protocol change is implied. Step F should not be expected to start promptly, and Phase 2 has no spare node if one drains.

**F-2 — N4 was the real risk, and R1.3 has now retired it. Confirmed independently here.**
Doc 46 R1.2 §3.5 stated that N4 (the S1-b item-stage relabelling identity E₃v_ψ + A₃v_φ ≈ 0) was **"not yet exercised numerically (D4 did not compute it)"**, that its 1e-5 tolerance was frozen with the plan, and that exceeding it on the Phase 1 regression dataset is an **INTEGRITY STOP** with no tolerance adjustment permitted (doc 47 §8 R-d). An unexercised identity holding a kill-the-run threshold was the largest avoidable risk in step F.

It was avoidable because the blocks are already saved: `d4/D4_dataset0.json` holds `A3` (80×80), `E3` (80×19), `A1` (80×80), `C1`, `C2` (19×80), `V_full` and `B_full` (198×198), all finite, for each of the five development datasets.

That evaluation has now been done, by `scripts/58_m2_rev3_prefreeze_checks.py` (`b9cef343…`) into `experiments/20261019T172456Z_stage2_rev3_prefreeze_checks/`, labelled *"PRE-FREEZE READ-ONLY CHECKS — not a recovery result"*. Read independently here, its summary gives, over 5 datasets × 4 relabelling tangents (T_c, T_d, P_c, P_d) = 20 instances:

| N4 variant | max relative residual | instances exceeding 1e-5 |
|---|---|---|
| **without** the first-order gradient term (as written in R1.2) | **5.23e-4** (dataset 1, P_c) — **52× the tolerance** | **10 of 20** |
| **with** the gradient term (R1.3) | **4.55e-7** (dataset 2, P_c) — 22× inside tolerance | **0 of 20** |

The failures are confined to the two *c*-direction tangents (T_c, P_c); both *d*-direction tangents pass either way (≤ 1.03e-8). This is exactly what the omission of the gradient term predicts: under θ′ = (θ − d)/c only the c-direction Jacobian picks up the first-order term.

**Consequence.** Had R1.2 been frozen and executed as written, N4 would have exceeded its frozen tolerance on the Phase 1 regression dataset with near-certainty, ending the run as an INTEGRITY STOP — a false stop caused by an incomplete identity, not by any defect in the pipeline, and with no tolerance change permitted within the test. R1.3's correction (N3 and N4 now include the first-order gradient term) removes that. **This is a run-saving catch, and it is the single most consequential change in R1.3.**

Also recorded from the same run, for information: the stacked Jacobian's 2-norm condition number is **1707–2449** unscaled across the five datasets (54.4–55.6 after symmetric Jacobi scaling), i.e. about 8 orders of magnitude below N6's 1e12 flag threshold. N6 is not a live concern.

**F-3 — the planning verification is still not hash-locked.**
Doc 47's operating characteristics (§4) and the R2′ justification rest on `scripts/56_m2_rev3_planning_checks.py`, `scripts/57_m2_rev3_planning_thresholds.py` and `tests/test_rev3_planning_functionals.py`. These postdate `00_baseline.json` and are **not** among its 94 recorded files (verified by lookup); no `.sha256` record covers them, and the planning run's own manifest covers only its three outputs. The same now applies to `scripts/58_m2_rev3_prefreeze_checks.py`, whose hash is at least self-recorded inside its own `summary.json`. Suggestion for the freeze: record all four files' SHA-256 in the freeze record, so the provenance of the frozen thresholds and of the N4 correction is fixed alongside the plan. (Auditability, not correctness.)

**F-4 — two existing freeze records are not read-only.**
`_frozen/34_STAGE2_STEP5_PLAN.sha256` and `_frozen/37_STAGE2_STEP6_PLAN.sha256` are mode `-rw-rw-r--`, while every other `_frozen` artefact is `-r--r--r--`. Cosmetic, pre-existing, untouched here; worth fixing when the next locks are written.

**F-5 — the freeze was mid-execution during these checks; it completed at 17:29Z and now verifies. (Resolved.)**
This finding is kept as the record of a transient state, because §§1–5 were computed inside it.

| time | state |
|---|---|
| 17:26Z | doc 46 at R1.3, header asserting **"FROZEN together with doc 47"** and citing lock files — while **no lock file existed**, no read-only copy existed, and doc 47 was still at R1.2 with a **"PROPOSED — NOT FROZEN — NO COMPUTATION AUTHORISED"** header. The two documents contradicted each other, and D6 requires them frozen *together* |
| 17:29Z | doc 47 brought to R1.3 (`36144159…`), both headers reading **"FINAL (R1.3) — FROZEN together with doc 46"**, and all four lock artefacts written |

**Verified state at 17:29Z** — the freeze is properly executed:

| artefact | mode | verdict |
|---|---|---|
| `_frozen/46_STAGE2_REV3_AMENDMENT.as_frozen_20261019.md` | `-r--r--r--` | present |
| `_frozen/47_STAGE2_REV3_RECOVERY_TEST_PLAN.as_frozen_20261019.md` | `-r--r--r--` | present |
| `_frozen/46_STAGE2_REV3_AMENDMENT.sha256` → `e2b40c11…` | `-rw-rw-r--` | `sha256sum -c`: **OK** against the live doc 46 |
| `_frozen/47_STAGE2_REV3_RECOVERY_TEST_PLAN.sha256` → `36144159…` | `-rw-rw-r--` | `sha256sum -c`: **OK** against the live doc 47 |

Both digests match the live text, so the ordering trap (the FROZEN header being part of the hashed file) was handled correctly: the locks were taken from the final text, headers included. The two `.sha256` files are mode `-rw-rw-r--`, extending the F-4 pattern to a third and fourth instance.

**Residual: none.** Doc 50, the R1.3 change record that both frozen documents reference, was written at 17:30Z as `50_STAGE2_REV3_R1_3_FREEZE_CHANGE_RECORD_20261019.md`, citing the *Final Execution Directive* and recording the pre-freeze computation. Both locks still verify **OK** against the live documents after it was added, so writing it did not disturb the frozen text. The record chain 48 → 49 → 50 is complete.

**Nothing in §§1–5 blocks execution.** Every artefact doc 47 cites exists, resolves to the cited key or column, and verifies against its manifest; the reused code, panel, truth, seed namespace and environment are all unchanged from the baseline the regression checks compare against; and the protocol freeze is now in place and verified.

## 7. Authorisation state

Docs 46 and 47 R1.3 cite the user's *Final Execution Directive* (2026-10-19) as authorising the freeze, driver implementation, the smoke run, Phase 1 and a conditional Phase 2, subject to the plan's gates. **This document does not interpret that directive's scope, does not act on it and authorises nothing.** Per doc 47 §11 the gated steps are A (freeze — now done), B (driver and tests), D (execute; the smoke run is the first job) and G (continue to Phase 2 after the Phase 1 review).

Standing items to carry into those steps, from §6:

| item | nature | carry into |
|---|---|---|
| F-3 planning and pre-freeze scripts not hash-locked (`56`, `57`, `58`, `tests/test_rev3_planning_functionals.py`) | auditability; doc 50 cites script 58 by path and output, not by digest | the step D code freeze |
| F-1 node002 occupied; no spare normal node | scheduling | steps F and G |
| F-4 `.sha256` lock files not read-only (now 4 instances) | cosmetic convention | whenever next convenient |
