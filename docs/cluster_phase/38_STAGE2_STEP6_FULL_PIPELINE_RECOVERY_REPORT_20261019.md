# Stage 2 M2, Step 6: full-pipeline recovery with S1-b. Execution report

**Date:** 2026-10-19.

**Plan:** `37_STAGE2_STEP6_FULL_PIPELINE_RECOVERY_PLAN_FINAL_20261019.md`, frozen `c02be4ce70e1b791384dafb7a6d491b483d85dbca22c4d07e1a01bed94b50bba`. The decisions are E-1 to E-12. The plan, its lock and its read-only copy are unchanged.

**Execution directory:** `experiments/20261018T222857Z_stage2_step6_full_pipeline_recovery/`. The name is UTC; the run was on 2026-10-19, IST.

**Jobs on node002** (Xeon Gold 6326, 1 exclusive node):

| job | purpose | outcome |
|---|---|---|
| 13912 | smoke 1, *not a result* | code-path defect, no fit |
| 13913 | smoke 2, *not a result* | completed |
| 13914 | **the Step 6 run** | completed |

**Driver** `scripts/52_m2a1_full_pipeline_recovery.py`, as run: `7e95263c…`. **Tests:** `tests/test_s1b_pipeline.py`.

## 1. Outcome

| | |
|---|---|
| **Outcome class** | **COMPLETE, no flagged event.** All 5 datasets completed every stage; no F1–F4 event occurred |
| **Formal verdict** (frozen criterion: every one of the 19 ψ components covered in ≥ 4 of 5 datasets) | **FAIL** |
| components below 4 of 5 | **β0_P 3/5** (mean z +2.44); **Le00_log 3/5** (mean z −1.71); **Le11_log 2/5** (mean z −2.11) |
| other 16 components | covered in 4/5 or 5/5 |
| incomplete execution or numerical failure | **none.** No execution error, no integrity stop, no non-finite value. The runtime gate passed. All 20 fits converged. All 10 Hessians are positive definite. There are no bound hits, no missing intervals, and no fallback or budget event at any optimum |

**This is a formal recovery failure, not an execution failure.** It is the case Rev 2 §5 and §6 address.

**Consequences under the frozen protocol:**
- **S1-b is not accepted.** S1-b is "accepted **only if** the unchanged full-pipeline recovery … passes."
- **The Stage 2 sequence stops at step 6.** "Each step's failure stops the sequence; it is reported, and nothing after it runs."
- **Step 7 (real-data B11) and Step 8 (G1–G7) are not run.**
- Any further stage-1 repair (S1-c, joint estimation) "would need a separate amendment" (Rev 2 §5). None is proposed or started here.

## 2. Results

ψ̂⁽¹⁾ results are descriptive and **non-binding** (E-8); only ψ̂⁽²⁾ enters the criterion.

### T1. Preflight integrity checks (in the job, before any recovery fit)

| check | result |
|---|---|
| I-1_locks | True |
| I-2_code | True |
| I-2_pilot_code_set | True |
| I-3_panel_md5 | True |
| I-3_inputs | True |
| I-7_script51 | True |
| I-4 Step 6 tests | 11 passed in 0.28s |
| I-8 Step 5 tests | 3 passed in 1.53s |
| I-5 truth = pilot | True |
| I-6 identity | 57022.74225348176 = 57022.74225348176: True |
| I-9 reused latent functions vs Step 5 records | T: construction True, oracle NLL 41256.27115826116 True; P: construction True, oracle NLL 17749.656319838956 True |
| I-10 seeds / git | seeds [42,45,1,r] r=0..4; HEAD 3a860ca; staged 0 |
| I-11 pilot N(0,1) item-fit records (E-11) | 30/30 exactly equal → True |

### T2. Coverage of ψ̂⁽²⁾ (binding): z = (ψ̂ − ψ_true)/SE per dataset; * = not covered

| ψ component | truth | ds 0 | ds 1 | ds 2 | ds 3 | ds 4 | covered | ≥ 4 of 5 |
|---|---|---|---|---|---|---|---|---|
| beta0_T | -0.2066 | +2.63* | -0.42 | -0.78 | -1.17 | -1.80 | 4/5 | yes |
| beta1_T | +0.0478 | -1.79 | +0.04 | -0.33 | +0.25 | +1.92 | 5/5 | yes |
| gamma_T | +0.0957 | +0.16 | +1.52 | -1.02 | +0.65 | +0.31 | 5/5 | yes |
| beta0_P | -0.4768 | +2.78* | +1.47 | +4.49* | +1.66 | +1.82 | 3/5 | **no** |
| beta1_P | +0.1216 | -0.63 | -1.28 | -0.06 | -0.51 | -0.12 | 5/5 | yes |
| gamma_P | +0.1621 | -0.86 | -0.79 | -2.78* | +1.15 | -1.23 | 4/5 | yes |
| L00_log | -0.2674 | -0.38 | -0.36 | -1.76 | -1.64 | +0.54 | 5/5 | yes |
| L10 | +0.0077 | +0.41 | -0.93 | +0.32 | +2.59* | -0.95 | 4/5 | yes |
| L11_log | -2.5750 | -0.25 | +0.18 | -0.81 | -1.62 | -0.27 | 5/5 | yes |
| L20 | +0.0648 | +1.05 | -0.16 | -0.23 | -0.25 | +1.24 | 5/5 | yes |
| L21 | -0.0065 | -0.99 | -1.51 | +0.05 | -0.22 | +1.99* | 4/5 | yes |
| L22_log | -0.4382 | -1.51 | -2.69* | -0.86 | -0.39 | -1.60 | 4/5 | yes |
| L30 | +0.0000 | -0.10 | -0.01 | -0.32 | +0.90 | -1.32 | 5/5 | yes |
| L31 | +0.0000 | -0.17 | +0.54 | -0.89 | -1.28 | -0.32 | 5/5 | yes |
| L32 | +0.0098 | -0.59 | +0.14 | -0.63 | -1.31 | -0.67 | 5/5 | yes |
| L33_log | -2.3353 | +0.29 | -1.50 | -1.55 | +0.05 | -1.72 | 5/5 | yes |
| Le00_log | -0.7374 | -1.77 | -1.53 | -3.00* | +0.88 | -3.14* | 3/5 | **no** |
| Le10 | +0.0405 | +1.80 | -0.10 | -0.77 | +0.27 | -0.37 | 5/5 | yes |
| Le11_log | -0.9081 | -1.81 | -2.45* | -3.35* | -0.29 | -2.64* | 2/5 | **no** |

### T3. Per-dataset Hessian and coverage summary

| dataset | ψ̂⁽²⁾ covered | H⁽²⁾ PD | min eig | max eig | missing intervals | penalised FD evals | ψ̂⁽¹⁾ covered (non-binding) | H⁽¹⁾ PD |
|---|---|---|---|---|---|---|---|---|
| 0 | 17/19 | True | 126.0 | 34288 | [] | 0 | 16/19 | True |
| 1 | 17/19 | True | 158.1 | 35992 | [] | 0 | 18/19 | True |
| 2 | 15/19 | True | 111.0 | 42456 | [] | 0 | 15/19 | True |
| 3 | 18/19 | True | 58.3 | 37596 | [] | 0 | 18/19 | True |
| 4 | 16/19 | True | 138.7 | 37675 | [] | 0 | 16/19 | True |

### T4. Stage-2 fits per dataset

| dataset | fit | converged | message | iterations | evaluations | ‖∇‖ (0.934 ref. only) | bound hits | penalty evals (reasons) | non-ND at accepted | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | (2) ψ̂⁽¹⁾ | True | CONVERGENCE: RELATIVE REDUCTION OF F < | 77 | 86 | 0.0008 | none | 2 {'mode_budget': 42, 'non_nd': 2} | 0 | 169 |
| 0 | (4) ψ̂⁽²⁾ | True | CONVERGENCE: RELATIVE REDUCTION OF F < | 80 | 96 | 0.0023 | none | 3 {'mode_budget': 64, 'non_nd': 2} | 0 | 205 |
| 1 | (2) ψ̂⁽¹⁾ | True | CONVERGENCE: RELATIVE REDUCTION OF F < | 90 | 102 | 0.0043 | none | 2 {'mode_budget': 42, 'non_nd': 2} | 0 | 186 |
| 1 | (4) ψ̂⁽²⁾ | True | CONVERGENCE: RELATIVE REDUCTION OF F < | 90 | 104 | 0.0004 | none | 2 {'mode_budget': 42, 'non_nd': 2} | 0 | 191 |
| 2 | (2) ψ̂⁽¹⁾ | True | CONVERGENCE: RELATIVE REDUCTION OF F < | 101 | 118 | 0.0008 | none | 3 {'mode_budget': 65, 'non_nd': 1} | 0 | 238 |
| 2 | (4) ψ̂⁽²⁾ | True | CONVERGENCE: RELATIVE REDUCTION OF F < | 101 | 112 | 0.0009 | none | 2 {'mode_budget': 43, 'non_nd': 1} | 0 | 202 |
| 3 | (2) ψ̂⁽¹⁾ | True | CONVERGENCE: RELATIVE REDUCTION OF F < | 95 | 104 | 0.0005 | none | 2 {'mode_budget': 43, 'non_nd': 1} | 0 | 188 |
| 3 | (4) ψ̂⁽²⁾ | True | CONVERGENCE: RELATIVE REDUCTION OF F < | 106 | 116 | 0.0011 | none | 2 {'mode_budget': 42, 'non_nd': 2} | 0 | 207 |
| 4 | (2) ψ̂⁽¹⁾ | True | CONVERGENCE: RELATIVE REDUCTION OF F < | 103 | 126 | 0.0007 | none | 4 {'mode_budget': 77, 'non_nd': 1} | 0 | 269 |
| 4 | (4) ψ̂⁽²⁾ | True | CONVERGENCE: RELATIVE REDUCTION OF F < | 90 | 104 | 0.0018 | none | 3 {'mode_budget': 65} | 0 | 220 |

### T5. Item fits per dataset (one start at initial_items)

| dataset | fit | converged | NLL | ‖∇‖ | iterations | a ∉ [0.05, 20] | wall s |
|---|---|---|---|---|---|---|---|
| 0 | (1) N(0,1) T | True | 41311.405 | 0.258 | n/r | False | 50.339 |
| 0 | (1) N(0,1) P | True | 18204.100 | 0.528 | n/r | False | 14.885 |
| 0 | (3) S1-b T | True | 41250.547 | 0.457 | 55 | False | 62.896 |
| 0 | (3) S1-b P | True | 17750.559 | 0.147 | 30 | False | 18.314 |
| 1 | (1) N(0,1) T | True | 42306.163 | 1.163 | n/r | False | 47.825 |
| 1 | (1) N(0,1) P | True | 19024.512 | 0.423 | n/r | False | 19.323 |
| 1 | (3) S1-b T | True | 42207.128 | 0.244 | 50 | False | 57.617 |
| 1 | (3) S1-b P | True | 18512.071 | 0.790 | 34 | False | 19.865 |
| 2 | (1) N(0,1) T | True | 42522.332 | 0.278 | n/r | False | 48.795 |
| 2 | (1) N(0,1) P | True | 18164.254 | 0.228 | n/r | False | 18.705 |
| 2 | (3) S1-b T | True | 42398.564 | 0.300 | 45 | False | 53.284 |
| 2 | (3) S1-b P | True | 17773.501 | 0.228 | 33 | False | 18.878 |
| 3 | (1) N(0,1) T | True | 42109.399 | 0.295 | n/r | False | 52.722 |
| 3 | (1) N(0,1) P | True | 18315.594 | 0.106 | n/r | False | 19.663 |
| 3 | (3) S1-b T | True | 41979.304 | 0.771 | 44 | False | 51.183 |
| 3 | (3) S1-b P | True | 17911.959 | 0.235 | 35 | False | 19.074 |
| 4 | (1) N(0,1) T | True | 42803.955 | 0.376 | n/r | False | 51.922 |
| 4 | (1) N(0,1) P | True | 18494.268 | 0.403 | n/r | False | 19.695 |
| 4 | (3) S1-b T | True | 42662.649 | 0.298 | 50 | False | 59.890 |
| 4 | (3) S1-b P | True | 18005.480 | 0.163 | 34 | False | 18.609 |

### T6. Reported evaluations (Q = 15, declared G = 5)

| dataset | at ψ̂⁽²⁾: fallback patients | budget triggers | log-lik | at ψ̂⁽¹⁾ (non-binding): fallback | budget |
|---|---|---|---|---|---|
| 0 | 0 | 0 | -56971.266 | 0 | 0 |
| 1 | 0 | 0 | -58696.940 | 0 | 0 |
| 2 | 0 | 0 | -57999.523 | 0 | 0 |
| 3 | 0 | 0 | -57970.545 | 0 | 0 |
| 4 | 0 | 0 | -58570.932 | 0 | 0 |

### T7. σ_T and σ_P on the natural scale (truth σ_T 0.4783, σ_P 0.4053)

| dataset | σ_T ψ̂⁽²⁾ | σ_P ψ̂⁽²⁾ | σ_T ψ̂⁽¹⁾ (non-binding) | σ_P ψ̂⁽¹⁾ (non-binding) |
|---|---|---|---|---|
| 0 | 0.4579 | 0.3844 | 0.4613 | 0.3809 |
| 1 | 0.4606 | 0.3743 | 0.4647 | 0.3797 |
| 2 | 0.4454 | 0.3593 | 0.4549 | 0.3649 |
| 3 | 0.4883 | 0.4019 | 0.4839 | 0.3972 |
| 4 | 0.4427 | 0.3697 | 0.4428 | 0.3740 |

### T8. ψ̂⁽¹⁾ coverage counts (N(0, 1)-only pipeline) — DESCRIPTIVE, NON-BINDING (E-8)

| ψ component | ψ̂⁽¹⁾ covered | ψ̂⁽²⁾ covered (binding) |
|---|---|---|
| beta0_T | 4/5 | 4/5 |
| beta1_T | 5/5 | 5/5 |
| gamma_T | 5/5 | 5/5 |
| beta0_P | 3/5 | 3/5 |
| beta1_P | 5/5 | 5/5 |
| gamma_P | 4/5 | 4/5 |
| L00_log | 5/5 | 5/5 |
| L10 | 4/5 | 4/5 |
| L11_log | 5/5 | 5/5 |
| L20 | 5/5 | 5/5 |
| L21 | 4/5 | 4/5 |
| L22_log | 5/5 | 4/5 |
| L30 | 5/5 | 5/5 |
| L31 | 5/5 | 5/5 |
| L32 | 5/5 | 5/5 |
| L33_log | 5/5 | 5/5 |
| Le00_log | 3/5 | 3/5 |
| Le10 | 5/5 | 5/5 |
| Le11_log | 1/5 | 2/5 |

### T9. Mean z over the 5 datasets: ψ̂⁽¹⁾ (non-binding) vs ψ̂⁽²⁾

| ψ component | mean z ψ̂⁽¹⁾ | mean z ψ̂⁽²⁾ |
|---|---|---|
| beta0_T | -0.18 | -0.31 |
| beta1_T | +0.03 | +0.02 |
| gamma_T | +0.36 | +0.32 |
| beta0_P | +2.19 | +2.44 |
| beta1_P | -0.60 | -0.52 |
| gamma_P | -0.58 | -0.90 |
| L00_log | -0.44 | -0.72 |
| L10 | +0.24 | +0.29 |
| L11_log | -0.50 | -0.55 |
| L20 | +0.39 | +0.33 |
| L21 | -0.16 | -0.14 |
| L22_log | -0.70 | -1.41 |
| L30 | -0.21 | -0.17 |
| L31 | -0.43 | -0.43 |
| L32 | -0.88 | -0.61 |
| L33_log | -1.23 | -0.88 |
| Le00_log | -1.47 | -1.71 |
| Le10 | +0.18 | +0.17 |
| Le11_log | -1.98 | -2.11 |

### T10. Item recovery (reported, not gated): pooled over datasets

| subscore | stage | mean a err | RMSE a | max \|a err\| | mean τ err | RMSE τ |
|---|---|---|---|---|---|---|
| T | stage1 | +0.0556 | 0.1042 | 0.2818 | -0.0558 | 0.1216 |
| T | s1b | +0.0581 | 0.1125 | 0.2907 | -0.0596 | 0.1397 |
| P | stage1 | +0.2206 | 0.3126 | 0.8673 | -0.0681 | 0.1464 |
| P | s1b | +0.1744 | 0.2421 | 0.5884 | -0.0473 | 0.1225 |

Per dataset (corr(a, a_true) / max |a err|):

| dataset | T stage 1 | T S1-b | P stage 1 | P S1-b |
|---|---|---|---|---|
| 0 | 0.9874 / 0.134 | 0.9866 / 0.137 | 0.9946 / 0.322 | 0.9879 / 0.157 |
| 1 | 0.9931 / 0.246 | 0.9934 / 0.210 | 0.9920 / 0.463 | 0.9864 / 0.337 |
| 2 | 0.9915 / 0.282 | 0.9923 / 0.291 | 0.9885 / 0.867 | 0.9961 / 0.588 |
| 3 | 0.9905 / 0.135 | 0.9863 / 0.226 | 0.9845 / 0.280 | 0.9764 / 0.145 |
| 4 | 0.9890 / 0.194 | 0.9870 / 0.158 | 0.9827 / 0.620 | 0.9703 / 0.529 |

Fitted-mixture pooled moments at ψ̂⁽¹⁾ (mean / variance): {0: {'T': (0.0266, 0.9787), 'P': (0.0445, 0.8948)}, 1: {'T': (0.0135, 0.9653), 'P': (0.0142, 0.8509)}, 2: {'T': (-0.0235, 0.9304), 'P': (0.0603, 0.8808)}, 3: {'T': (-0.0059, 0.9885), 'P': (0.0512, 0.9289)}, 4: {'T': (-0.0017, 0.9703), 'P': (0.0299, 0.8717)}}

### T11. Runtime

- Dataset-0 gate (E-9): t₂ = 169 s, t₄ = 205 s (each ≤ 3600 s: True / True); ℓg = 1.399 s; projected pilot (frozen formula) = 2.00 node-hours (≤ 24: True); variant with measured S1-b = 1.55 (reported only). **Gate passed: True.**
- Per-dataset wall s: ds 0 552, ds 1 551, ds 2 609, ds 3 562, ds 4 667; stage 1 (all datasets) 53 s; job script runtime 3014 s.


Figures: `step6/figures/z_by_component.png`, `item_discrimination.png`, `item_thresholds.png`.

## 3. Flagged events, edge cases and the open ambiguities

| item | result |
|---|---|
| F1: stage-2 fit not converged, at maxiter, or stopped by time | none (10/10 converged, "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH", 77–106 iterations) |
| F2: item fit not converged, or a ∉ [0.05, 20] | none (20/20 converged; all a in range) |
| F3: bound hit at ψ̂⁽²⁾ | none |
| F4: fallback or budget event at the ψ̂⁽²⁾ reported evaluation | none (0 fallback patients, 0 budget triggers in all 5). No fallback or budget event at ψ̂⁽¹⁾ either |
| non-PD or singular H⁽²⁾ (E-6) | none (min eigenvalue 58–158) |
| missing or non-finite SE (E-6) | none |
| B11 gradient reference (U-4, report only) | ‖∇‖ 0.0004–0.0043 at every optimum (reference 0.934) |
| optimiser-context penalties (doc 25 P1/P2; frozen semantics, not flags) | 2–4 penalised trial evaluations per fit (`mode_budget` and `non_nd` chunk reasons). **0** non-ND patients at every accepted evaluation. 0 penalised evaluations in any of the 10 FD Hessians |

**Ambiguities U-1 to U-5** (plan §14) **were not triggered.** No flagged event occurred, and the run is complete. The formal verdict therefore needs no review of an unresolved ambiguity.

## 4. Descriptive observations (not part of the verdict)

1. **The three failing components match the earlier pipeline's failure.** The doc 17 / pilot full-pipeline recovery, with the old likelihood and no S1-b, failed on β0_P 3/5, Le00_log 2/5 and Le11_log 0/5 (Rev 2 O3). With the Rev 2 likelihood and S1-b, the same three components fail in the same directions: β0_P too high; Le00_log and Le11_log (the residual-SD log-diagonals) too low.
2. **Step 4 isolates the item step.** In Step 4, fixed-item recovery with the same likelihood, datasets and Q passed, with β0_P, Le00_log and Le11_log all 5/5. The shortfall therefore arises with the estimated items, which Step 4 by construction did not test.
3. **S1-b changed the item estimates more than ψ̂.**
   - The PIGD discrimination error fell from a mean of +0.221 to +0.174 (RMSE 0.313 → 0.242).
   - The ψ coverage counts are unchanged between ψ̂⁽¹⁾ (non-binding) and ψ̂⁽²⁾, except L22_log (5 → 4) and Le11_log (1 → 2).
   - The mean z of β0_P moved from +2.19 to +2.44.
   - The fitted latent mixture at ψ̂⁽¹⁾ has pooled PIGD variance 0.85–0.93 (the truth is 1 by construction). So the one-step refit is anchored to an already-biased ψ̂⁽¹⁾. **[H]** This is a plausible reason why one iteration corrects little. It is a hypothesis, not tested here.
4. **For context only** (Step 5 is diagnostic and authorises nothing): the S1-a first-order indications pointed to the same components with the same signs (β0_P +2.2 SE, Le11_log −1.8 SE).

No conclusion about a repair is drawn from these observations. They select nothing.

## 5. Integrity

**Pre-fit checks:** all passed (T1). Every check ran in the job, in order, before the first stage-2 fit, and I-11 ran before any stage-2 fit (E-11).

**Post-run** (`99_post_run_integrity.txt`):
- **Frozen records:** all 8 locks OK; the doc 37 read-only copy is identical.
- **Code, documents and prior outputs:** code and documents are unchanged against `00_baseline.json`. All 147 prior-output files (pilot, Step 3, Step 4, Step 5) are unchanged. The panel MD5 is unchanged.
- **This run's outputs:** `step6/MANIFEST.sha256` verifies (14 files).
- **Changes during the run:** the only file under `docs/`, `src/`, `scripts/`, `tests/` or `outputs/` that changed after the baseline is the driver itself. That is the documented smoke-1 fix (§7), made before the main run.
- **git:** HEAD `3a860ca`, 0 tracked changes, 0 staged.

D1 verdict: **NOT REPRODUCED**, unchanged.

## 6. Runtime

| job | node-hours (AllocNodes × ElapsedRaw / 3600) |
|---|---|
| 13912 (smoke 1) | 0.003 |
| 13913 (smoke 2) | 0.061 |
| 13914 (main run) | 0.839 |
| **total** | **0.904** |

The plan's estimate was 0.9–1.2 node-hours for the main run. The measured 0.84 is within the 1.7 node-hour margin.

## 7. Deviations and defects

1. **Smoke 1 (job 13912) found a code-path defect before any fit.** The reused Step 5 module was loaded by `importlib` without being registered in `sys.modules`, so the process pool could not pickle its task function. The record is `SMOKE1_DEFECT_NOTE.txt`.
   - **Fix:** one line, registering the module before executing it.
   - **No setting, seed, criterion or reused source was changed**; script 51's hash is verified unchanged (I-7).
   - The driver hashes before the fix are in `02_driver_sha256.txt`, and after it in `02b_driver_sha256_after_smoke1.txt`.
   - Smoke 2 (job 13913) ran the full code path. Its flags (F1, F2) are expected, because the smoke check caps maxiter; it is *not a result*.
2. **N(0, 1) item-fit iteration counts** are not exposed by the reused pilot function. They are shown as "n/r"; the gradient norm at the optimum is reported instead.
3. **Plan §11, test (c)** "no RNG beyond the dataset seeds" is implemented as a stub test asserting that no RNG is constructed inside the per-dataset pipeline.

No other deviation from doc 37.

## 8. Files

**New, untracked:**
- `scripts/52_m2a1_full_pipeline_recovery.py`;
- `tests/test_s1b_pipeline.py`;
- this report.

**Execution directory:**
- `00_baseline.json`, `01_plan.sha256`, `02_driver_sha256.txt`, `02b_driver_sha256_after_smoke1.txt`, `SMOKE1_DEFECT_NOTE.txt`;
- `smoke_codepath/`, `smoke_codepath_2/` (not results);
- `step6/`:
  - `S0_provenance.json`;
  - `S1_dataset{0..4}.json` (all stages, logs, H⁽¹⁾ and H⁽²⁾);
  - `S2_coverage_table.csv`;
  - `S2b_coverage_table_psi1_nonbinding.csv`;
  - `S3_item_recovery.csv`;
  - `S4_convergence.csv`;
  - `summary.json`;
  - `figures/` (`z_by_component.png`, `item_discrimination.png`, `item_thresholds.png`);
  - `MANIFEST.sha256`;
- `jobs_sacct.txt`, `99_post_run_integrity.txt`, `slurm-1391{2,3,4}.{out,err}`, `job_header.txt`, `REPORT.md`.

## 9. What this establishes, and what it does not

**Established:** on the 5 frozen synthetic datasets (a-priori truth, real design), the complete two-stage pipeline with one S1-b iteration, Laplace Q = 15, does **not** meet the frozen recovery criterion. Its 95 % Wald intervals, which ignore item uncertainty (doc 16 B5), cover β0_P, Le00_log and Le11_log in fewer than 4 of 5 datasets.

**Not established:**
- anything about the real PPMI data, model adequacy on real data, or biological subtypes;
- coverage rates (5 datasets);
- whether another repair would pass.

No real-data fit, calibration, positive control, gate or evaluation statistic was run.

**Stopping here for review.** Under Rev 2 §6 the sequence stops at step 6. Step 7 and Step 8 are not run. Any S1-c, or other repair, needs a separate amendment and approval. Nothing was staged, committed or pushed.
