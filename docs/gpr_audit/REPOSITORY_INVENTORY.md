# Step 3 (GPR) — repository inventory

Audit date: **2026-09-29**. Compiled by inspection, not from prior summaries.

---

## 1. Git state at audit start

| | |
|---|---|
| Working directory | `D:\SEM 7\BTP` |
| Branch | `main` |
| HEAD | `7341f38` "Add Step 2 five-hidden-state LOO sharding" |
| `origin/main` | `7341f38` — **identical**, 0 ahead / 0 behind |
| Remote | `https://github.com/Parvptl/ppmi-subtype-progression.git` (private) |
| Stashes | none |
| Worktrees | one, the main checkout |

**No Step 3 work has ever been committed or pushed.** Confirmed by
`git log --all --remotes -- <file>` returning zero commits for all eight GPR
files, and by the remote tree containing no file matching
`gpr|gauss|trajectory|step3`.

### Modified tracked files (pre-existing, not created by this audit)

| File | Nature of change |
|---|---|
| `config/config.yaml` | Step 3 `trajectory:` and `gpr:` blocks appended |
| `src/ppmi_subtype/config.py` | `trajectory` / `gpr` accessors + validation |
| `scripts/merge_5hidden_loo.py` | one-line display change, a **Step 2** leftover |

### Untracked Step 3 files

| File | Lines | Status |
|---|---:|---|
| `src/ppmi_subtype/gpr.py` | 764 | untracked, not ignored |
| `src/ppmi_subtype/gpr_loo.py` | 679 → 771 after audit fix | untracked, not ignored |
| `src/ppmi_subtype/gpr_plots.py` | 674 | untracked, not ignored |
| `src/ppmi_subtype/trajectory.py` | 488 | untracked, not ignored |
| `scripts/11_gpr_trajectory_prediction.py` | 557 | untracked, not ignored |
| `tests/test_gpr.py` | 388 | untracked, not ignored |
| `tests/test_gpr_loo.py` | 284 | untracked, not ignored |
| `tests/test_trajectory.py` | 347 | untracked, not ignored |

None of these are matched by `.gitignore`. They are untracked simply because
they were never staged — the user deliberately unstaged them on 2026-09-22 with
`git restore --staged .` after asking that nothing GPR-related reach GitHub.

### Why the outputs are invisible to git

`.gitignore` line 8 ignores `outputs/`, line 13 ignores `experiments/`. Both
carry PPMI-derived participant-level data and are correctly excluded under the
PPMI Data Use Agreement. `outputs/step3_gpr/` is therefore ignored by
inheritance, not by a Step 3-specific rule.

---

## 2. Environment

| | |
|---|---|
| Python | 3.13.7, Windows 11 (10.0.26200) |
| numpy / scipy / pandas | 2.2.4 / 1.15.2 / 2.2.3 |
| matplotlib / pyyaml / pytest | 3.10.1 / 6.0.3 / 9.1.1 |
| scikit-learn | 1.9.0 — **installed but not used and not a declared dependency** |
| Logical cores | 16 (≈8 physical + SMT) |
| Free disk on D: | 194 GB of 228 GB |

`pyproject.toml` declares pandas, numpy, scipy, pyyaml, openpyxl, matplotlib,
seaborn. The GPR code adds **no new dependency** — the GP is hand-written on
numpy/scipy, matching the house style of `hmm.py`, `ctmc.py` and `hbayes.py`.

---

## 3. Upstream inputs the GPR pipeline depends on

| Input | Status |
|---|---|
| `outputs/visit_panel.csv` | present, MD5 **`fb038ddbdc19ab3368e13940f398e7ba`** — unchanged from the frozen baseline used by every earlier stage |
| `src/ppmi_subtype/panel.py` `restrict_to_analysable` | reused unmodified |
| `src/ppmi_subtype/hbayes.py` | reused unmodified — it is a comparison model in the GPR evaluation |
| `config/config.yaml` `subtype:` block | unchanged: thresholds 0.90 / 1.15, `band_boundaries: outer` |

The GPR pipeline **does not** depend on the HMM, CTMC, Step 1 or Step 2
outputs. It re-derives its cohort from the panel. Earlier research stages
therefore did not need rerunning.

---

## 4. Cohort definitions — read from `config/config.yaml`, not inferred

| Cohort | Rule as implemented |
|---|---|
| **unfiltered** | no extreme criterion; structural filter only (≥ `min_finite_visits` = 2 finite-ratio visits) |
| **A4** | `degenerate_fraction`, `min_fraction: 0.80` — ≥80% of a patient's labelled visits have an undefined ratio (PIGD subscore 0) **or** a ratio of exactly 0 |
| **D1** | `boundary_fraction`, `high: 2.30`, `low: 0.45`, `min_fraction: 0.80` |
| **E2** | `run_and_fraction`, `high: 2.30`, `low: 0.45`, `min_run: 3`, `min_fraction: 0.60` |

Every criterion is evaluated per patient over that patient's **labelled**
visits (including those with an undefined ratio); the structural ≥2 finite
filter is applied afterwards. Undefined ratios count towards the **high** side
in D1 and E2.

### Cohort sizes — recomputed from code on the audit date

| Cohort | Patients | One-step points | Matches saved predictions? |
|---|---:|---:|---|
| unfiltered | 821 | 2,707 | ✅ exact |
| A4 | 803 | 2,654 | ✅ exact |
| D1 | 641 | 2,219 | file absent at audit start |
| E2 | 596 | 1,942 | file absent at audit start |

---

## 5. Original artifacts and verification status

| Artifact | Date | Verified |
|---|---|---|
| `gpr_predictions_unfiltered.csv` | 17 Sep 22:01 | ✅ 821 patients / 2,707 points / 9 models; all 10 leakage checks pass |
| `gpr_predictions_A4.csv` | 18 Sep 15:07 | ✅ 803 / 2,654 / 9 models; all 10 leakage checks pass |
| `gpr_fold_fits_{unfiltered,A4}.csv` | as above | ✅ 821 and 803 folds × 4 kernels; convergence and bound-hits recorded |
| `trajectory_predictions_{unfiltered,A4}.csv` | as above | ✅ present |
| `patient_trajectories.csv`, `trajectory_summary.csv`, `extreme_patient_summary.csv`, `cohort_comparison.csv`, `filtered_patient_trajectories.csv` | 17 Sep 20:16 | ✅ present, regenerated deterministically by `--stage cohorts` |
| `gpr_predictions_D1.csv` / `_E2.csv` | — | ❌ **never existed** (see §6) |
| `report.txt`, metrics CSVs, 13 figures | — | ❌ never generated; `--stage report` was never run |

All originals were copied to `.backup_step3/` before any rerun, and
`.backup_step3/` was added to `.gitignore`.

---

## 6. Why D1 and E2 produced nothing — a real defect, now fixed

`outputs/step3_gpr/lopo_run.log` shows D1 started (`=== D1: 641 folds …`) and
then stops. No D1 artifact of any kind exists anywhere in the repository.

**Cause.** `run_lopo` accumulated every fold in memory and the driver wrote CSVs
only after the whole cohort returned. When the machine slept mid-run, all
completed D1 folds were lost. There was no checkpoint, so nothing was
recoverable — the historical note that partial work "may be recoverable" is
incorrect; there is none.

**Fix.** `run_lopo` now takes `checkpoint=` and `checkpoint_every=` (default 25),
writes atomically via a `.tmp` + `replace`, and skips patients already present
on resume. Verified: a run interrupted at fold 4 of 8 and resumed produces a
frame **identical** to an uninterrupted run. See `METHODOLOGY_AUDIT.md` §F.

The same log also records A4 degrading from ~8 s/fold to 142 s/fold from fold
~375 onward (1,026 min total vs 105 min for the larger unfiltered cohort). This
is machine contention — other jobs were run on the same box that day — not a
correctness problem. A4's outputs pass every consistency check.

---

## 7. Run provenance that was *not* recorded

The original runs did not persist their invocation. Recovered by inference from
the log header and the config:

| | |
|---|---|
| Command | `python scripts/11_gpr_trajectory_prediction.py --stage lopo --jobs 12` |
| Seed | `cfg.seed` = 42 (`gpr.n_restarts` = 4, `max_iter` = 300, `jitter` = 1e-8) |
| Kernels | `cfg.gpr.kernels` = K0, K1, K2, K3 |
| Panel MD5 | `fb038ddbdc19ab3368e13940f398e7ba` |

This gap is addressed in `RUN_MANIFEST.md`, which records provenance for every
run performed during the audit.

---

## 8. Missing inputs and unresolved dependencies

| Item | Status |
|---|---|
| The Medium article defining the intended GP formulation | **Supplied 2026-09-29**, after the implementation was written. Assessed in `METHODOLOGY_AUDIT.md` §B. |
| D1 / E2 predictions | Regenerated during this audit (Stage 4). |
| Report-stage artifacts | Generated during this audit (Stage 6). |

No missing *data* inputs. The panel and all upstream code are present and
unmodified.
