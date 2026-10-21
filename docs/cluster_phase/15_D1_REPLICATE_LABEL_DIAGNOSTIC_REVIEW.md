# D1 observed-cohort replicate labels: feasibility and protocol-compliance review

**Date:** 2026-10-16.
**Type:** review and proposed specification. **Nothing was run.** No frozen file was changed.
**Concerns:** item M1 of `14_D1_STAGE1_CHECKS_20261016.md` §6.
**Frozen run:** `experiments/20261016T070746Z_d1_full/results/` (job 13467, node004).
**Frozen protocol:** `12_CONSOLIDATED_RESEARCH_REPORT.md` Appendix B.10–B.14, plus `00_protocol.json`.

---

## 1. Which stage produced the observed-cohort reference [verified from code]

**Call chain.** `scripts/41_d1_label_ppc.py` → `full()` → `observed_fn()` → `ppc_diagnostic.run_cohort(data, data.y, folds, R=500, FIT_KW, [42, 41, 9, 0, 7])`, with the default `record_stats=True`. `ppc_runner.run_all` calls `observed_fn` once, last, in the main process; the cohorts ran in worker processes.

**Inside `run_cohort`, for each fold f = 0…4 in sorted order:**

| step | what it does | seed |
|---|---|---|
| A | TT fit on the other four folds: `hbayes_tt_nu.fit`, collapsed slice sampler, Gamma(2, 0.1) ν prior, 4,000 draws, burn-in 2,000, thin 5 | `SeedSequence([42,41,9,0,7,f,1])`, first word mod 2³¹ |
| B–C | for each test patient in `data.patients` order: `patient_replicates`, which draws (b, λ, ω) given the patient's own ratios and θ⁽ʳ⁾ by 20 Gibbs steps, then fresh t noise at the patient's own visit times; R = 500 draws picked by `linspace(0, 3999, 500)` | one `default_rng([42,41,9,0,7,f,2])` per fold, consumed in patient order |

After the fold loop:
- Labels: `Lrep = labels_from_ratio(Yrep)`, a 500 × 3,528 array.
- Statistics: `label_statistics`, then `discrepancy`, giving the p̃, z and T in `05_observed.json`. Per-fold z values come from slices of the same `Lrep`.

**What was not saved.** `run_cohort` returns only summaries (`S_rep_mean`, p, z, T, fold z). Neither `Yrep` nor `Lrep` was written.

**The generator is not part of this stage.** The null generator (`fullcohort_tt_draws.npz`) is not used for the observed cohort; it only produced the synthetic cohorts.

## 2. Are the frozen seeds and saved draws sufficient?

**Saved draws: not sufficient.** The observed replicates used the five *fold* posteriors, and those were not saved. The saved npz is the full-cohort generator posterior, a different fit.

**Seeds plus code plus environment: sufficient in principle, by deterministic re-execution.** Every random number in the stage comes from the two seeded generators above; no global RNG is used (`hbayes_tt_nu.fit` uses `default_rng(seed)`). The other inputs:

| input | frozen value | current state | match |
|---|---|---|---|
| Code: `ppc_diagnostic`, `ppc_runner`, `hbayes`, `hbayes_t`, `hbayes_tt`, `hbayes_tt_nu`, `indeterminate_audit`, `subtype`, script 41 | SHA-256 in `00_protocol.json` | recomputed today | **all identical** |
| Panel | MD5 `fb038ddb…` | recomputed | identical |
| Folds | `folds.csv` SHA-256 in the protocol | recomputed | identical; script 41 rebuilds the folds with `make_folds(data, 5, 42)` and checks them against this file |
| Python packages | `pip_freeze.txt` | `pip freeze` today | **identical** |
| Threads | `job.slurm` exports `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`; script 41 also defaults them to 1 | must be set the same | reproducible |
| BLAS | scipy-openblas 0.3.28, `DYNAMIC_ARCH`, so the kernel is chosen by CPU | node004 CPU model not recorded; the login node is a Xeon Gold 5315Y | **residual risk**: a different CPU may give different rounding. Mitigation: run on node004, plus the exactness gate (§5) |
| Protocol report | SHA-256 `9d816fe7…` (the B.1–B.14 version) | the file now carries B.15 and its prefix does not hash to `9d816fe7`; the run-time version is not in `untracked_source.tar.gz` | **provenance caveat only.** No computation reads the report; every computation-defining item above matches |

**Conclusion.** Faithful reproduction is achievable, but bitwise identity can only be *confirmed* by running. Hence the gate in §5.

## 3. Can per-visit labels be saved without changing anything?

**Yes, without editing any frozen or hashed file.**

**Mechanism.** A new diagnostic script imports `ppc_diagnostic` unchanged and wraps the module attribute `labels_from_ratio` with a recorder. The recorder calls the original function and keeps a copy of its 2-D output, which is exactly the `Lrep` that `run_cohort` passes to `label_statistics`. `run_cohort` resolves the name at call time, so it picks up the wrapper.

What this leaves unchanged:

| concern | effect |
|---|---|
| generator | not involved |
| statistic definitions | unchanged; the same `label_statistics` runs on the same `Lrep` |
| random-number consumption | none added: `labels_from_ratio` uses no RNG, the wrapper draws nothing, and every sampler call is identical |
| observed-cohort evaluation and decision rule | nothing is written to the frozen directory; `05_observed.json`, `01_threshold.json` and `02_power.json` are read only, with lock checksums verified, and the verdict function is not called |
| code hashes | no hashed file is edited. The new script and its tests are new files outside the D1 protocol |

**Alternatives rejected:**
- *Adding a `return_replicates` argument to `run_cohort`.* This changes a hashed file and causes protocol drift.
- *Re-implementing the loop.* This risks silent divergence.

## 4. Runtime, output size and integrity

**Runtime.** Frozen timings: observed cohort `fit_s` 64.8, `replicate_s` 5.4, `stats_s` 0.05, so about 71 s. A null-cohort pre-check (§5, step 1) takes about 73 s (null_000 `wall_s` 73.1). The total is **about 3 minutes on one core**.

**SLURM request.** Proposed: `-w node004 -c 2`, the same node as the frozen run. At review time nodes 002–004 were all allocated (node001 drained), so the queue wait is unknown.

**Output size:**
- `Lrep`: int8, 500 × 3,528 = 1.8 MB, about 0.2 MB compressed.
- Optionally the replicate *ratios* (float64, 14 MB). These are not requested and not proposed.

**Integrity of the frozen experiment.**
- The frozen directory is opened read-only.
- All seven `.lock` checksums and the `04_outputs` cohort checksums are verified before and after the run.
- New outputs go to a separate directory:

  `experiments/<UTC>_d1_observed_replicate_labels_DIAGNOSTIC/`

  It holds `README.md` stating "descriptive diagnostic, not a D1 result", `provenance.json`, `labels.npz` with its SHA-256, and the tables.

## 5. Proposed minimal run (not implemented, not run)

**Steps:**

| step | action | stop condition |
|---|---|---|
| 0 | Write `provenance.json` **before** any computation. It records: <br>• SHA-256 of all code files, matched to `00_protocol.json`; <br>• panel MD5; folds SHA-256; <br>• the `pip freeze` diff, which must be empty; <br>• host, CPU model (`lscpu`), thread variables, SLURM job ID; <br>• seeds `[42,41,9,0,7,f,1|2]`; R = 500; `FIT_KW`; <br>• the SHA-256 of this document and of the frozen `05_observed.json` | any mismatch → stop |
| 1 | **Determinism pre-check, which does not touch observed data.** Re-execute null cohort 000 exactly as `_full_cohort` did: entropy `[42,41,0,0]`, the frozen generator draws, `run_cohort(…, [42,41,0,0,7])` | its T, p, z, `S_obs` and fold z must equal `cohorts/null/null_000.json` **exactly** (JSON floats round-trip exactly); otherwise stop, and the observed cohort is never touched |
| 2 | Re-execute the observed stage as in §1, with the recorder | the returned `S_obs`, `S_rep_mean`, p, z, T and fold z must equal `05_observed.json` **exactly**. On any difference the labels are written to `REJECTED_nonidentical/`, never analysed, and the run is reported as a failed reproduction. That outcome would be a genuine second evaluation, which the protocol does not allow |
| 3 | Only after step 2 passes: save `labels.npz` (Lrep and the visit index: PATNO, time, fold) | — |
| 4 | Compute the descriptive statistics below | — |
| 5 | Re-verify the frozen locks; write `README.md` | — |

**Statistics.** Both were fixed in `13_D1_POST_RUN_AUDIT.md` §5 (D-a, D-b) before Stage 1 ran. They use Stage 1's tested functions and introduce no new primary statistic:

1. **Indeterminate share by the visit's PIGD item sum**, in strata 1, 2, 3, 4–5 and 6+. The observed values (already in doc 14) are compared with the 500 patient-conditional replicates: mean, 2.5–97.5 % range, and the mid-p p̃ with its z (descriptive).
2. **Direct-reversion rate by direction**, TD → PIGD and PIGD → TD, with the revert / stay / to_Ind / unobserved counts per replicate (`stage1_checks.direct_change_outcomes`). Reported the same way.

**What is not done:**
- No null calibration of these 7 statistics and no threshold.
- No verdict and no combination into T.
- No change to the model, R, seeds or folds.
- No tuning to the observed data.

The p̃ values are uncalibrated. They are also conservative, because each patient's data are used twice (B.3), and they cover 7 comparisons. They are labelled descriptive and post-hoc relative to D1.

## 6. Is the run permissible?

**Conditionally yes, as a recorded deviation that needs your explicit approval.**

- **Against the wording.** B.14.1 says the observed cohort is evaluated "exactly once", and B.11.5 says "the observed cohort, once". A re-execution is a second *execution*.
- **Against the purpose.** The rule exists so that c, the statistics and the rules cannot be influenced by the observed data, and so that no one can choose between evaluations. The proposal preserves both:
  - c, P_lo, the statistic definitions and the verdict are read from frozen files and cannot change.
  - The exactness gate means that either the identical evaluation is reproduced, so nothing new is chosen, or nothing is used.
- **Recording.** If you approve, the deviation is recorded in this document and in the diagnostic README. B.15 and the D1 verdict are not edited.
- **If you prefer the strict reading,** M1 is not permissible. The Stage 1 generator-based results (doc 14) remain the available comparison.

## 7. What it would and would not resolve

**Would resolve:**
- Whether the Stage 1 patterns hold against the reference D1 actually used. That reference is centred on each patient's own ratios, which removes most of the generator's covariate-blindness. The patterns in question:
  - the Indeterminate excess at PIGD sums 2–5 with none at 1;
  - the reversion deficit confined to TD → PIGD → TD.
- It also lets the D1 S1 and S3 misfits be split exactly into the strata and directions that make up D1's own z-scores.

**Would not resolve:**
- *A calibrated test of these splits.* That would need the null cohorts' labels, i.e. re-running 500 cohorts, which is not proposed.
- *The mechanism,* whether lattice, skew or something else.
- *The residual diagnostic (M2).* That needs patient random effects, and the fold fits here are not set to retain them; enabling that would change the call.
- *Anything about the full three-state label or an intermediate state.*

---

## 8. Execution record (2026-10-16, added after the run)

**Approval and run.** Approved by the reviewer on 2026-10-16 as a documented deviation, subject to §5. The run was SLURM job 13702 on node004 (Xeon Gold 6444Y), partition `normal`, single-threaded, 22:01–22:04 IST. Output is in `experiments/20261016T163134Z_d1_observed_replicate_labels_DIAGNOSTIC/results/`.

**Before submission:**
- node004 was idle with no jobs.
- The three new files matched their recorded SHA-256s.
- 34 related tests passed.
- The frozen locks were valid.

**Gates:**

| gate | compared keys | result |
|---|---|---|
| 1: null cohort 000 | T, z, p, S_obs, p_at_bound, T_at_cap, n_rep_undefined, folds, n_visits_in_episode | **passed**: 0 mismatches, exact |
| 2: observed re-execution | S_obs, S_rep_mean, p, z, T, T_at_cap, p_at_bound, folds; plus the recorded labels regenerate S_rep_mean exactly | **passed**: 0 mismatches, exact |

**Integrity:**
- Frozen locks (7), cohort checksums (1,000), the generator draws and the folds were verified before and after.
- The frozen results directory's full file manifest is unchanged.
- The D1 verdict is unchanged (NOT REPRODUCED).

**Saved labels:** `labels.npz`, 500 × 3,528 int8, SHA-256 `005c11f3…`.

**Exploratory, uncalibrated splits.** The reference is the patient-conditional replicates, the same draws as D1. "At bound" means the observed value lies beyond all 500 replicates (mid-p 0.001 or 0.999).

*Indeterminate share by PIGD sum:*

| PIGD sum | visits | observed | replicate mean (2.5–97.5 %) | mid-p |
|---|---|---|---|---|
| 1 | 1,413 | 0.050 (70) | 0.046 (0.036–0.056) | 0.25 |
| 2 | 1,061 | 0.138 (146) | 0.068 (0.053–0.084) | at bound |
| 3 | 441 | 0.152 (67) | 0.084 (0.058–0.111) | at bound |
| 4–5 | 321 | 0.178 (57) | 0.089 (0.061–0.118) | at bound |
| 6+ | 292 | 0.079 (23) | 0.088 (0.060–0.123) | 0.70 |

*Direct changes by direction (counts; replicate mean and 2.5–97.5 % range):*

| direction | revert | stay | to_Ind | unobserved | R_D |
|---|---|---|---|---|---|
| TD → PIGD | 37 (95.0; 78–114) | 86 (99.6; 84–116) | 23 (20.6; 13–29) | 76 (91.9; 74–110) | **0.253** vs 0.441 (0.385–0.501), at bound |
| PIGD → TD | 39 (72.0; 55–88) | 49 (80.5; 64–97) | 10 (15.4; 9–24) | 26 (69.8; 55–86) | 0.398 vs 0.429 (0.351–0.505), mid-p 0.79 |

**Reading.** Both Stage 1 patterns hold against D1's own reference. The Indeterminate excess is at PIGD sums 2–5, with none at 1 or 6+. The reversion-rate deficit is confined to TD → PIGD. PIGD → TD changes are fewer than in the replicates, but they revert at the replicate rate.

No new hypothesis test, verdict or calibration was added.
