# Step 3 (GPR) — run manifest and progress log

Durable record of every run, so the work can be resumed or reproduced.
Append-only; newest section last.

**Frozen input for every run below:** `outputs/visit_panel.csv`,
MD5 `fb038ddbdc19ab3368e13940f398e7ba`.

**Environment:** Python 3.13.7 · numpy 2.2.4 · scipy 1.15.2 · pandas 2.2.3 ·
matplotlib 3.10.1 · Windows 11 · 16 logical cores.

**Fixed configuration** (`config/config.yaml`, unchanged throughout):

```
seed: 42
subtype:    td_threshold 1.15 · pigd_threshold 0.90 · band_boundaries outer
trajectory: min_finite_visits 2 · min_finite_visits_multistep 3 · primary A4
gpr:        kernels [K0,K1,K2,K3] · n_restarts 4 · max_iter 300 · jitter 1e-8
```

---

## Run 1 — original LOPO, unfiltered (historical)

| | |
|---|---|
| When | 2026-09-17, 20:16 → 22:01 |
| Command | `python scripts/11_gpr_trajectory_prediction.py --stage lopo --jobs 12` |
| Scope | unfiltered cohort |
| Result | **complete** — 821 folds, 24,363 rows, 105.3 min, 7.7 s/fold |
| Checks | all 10 leakage/consistency checks passed |
| Artifacts | `gpr_predictions_unfiltered.csv`, `gpr_fold_fits_unfiltered.csv`, `trajectory_predictions_unfiltered.csv` |

## Run 2 — original LOPO, A4 (historical)

| | |
|---|---|
| When | 2026-09-17 22:01 → 2026-09-18 15:07 (same invocation as Run 1) |
| Result | **complete** — 803 folds, 23,886 rows, 1026.1 min |
| Anomaly | degraded from ~8 s/fold to 142 s/fold from ≈fold 375. Machine contention, not a correctness issue — all checks pass. |
| Checks | all 10 passed |

## Run 3 — original LOPO, D1 (historical, LOST)

| | |
|---|---|
| When | 2026-09-18, began ~15:07 |
| Result | **destroyed.** The machine slept mid-run. `run_lopo` held all folds in memory and wrote only on completion, so every finished fold was lost. No artifact of any kind exists. |
| Consequence | D1 rerun from scratch (Run 6). E2 never started. |

---

## Audit runs — 2026-09-29

## Run 4 — metric reconciliation

| | |
|---|---|
| Command | `python scripts/12_gpr_reconcile_metrics.py --cohorts unfiltered,A4` |
| Inputs | the saved prediction CSVs from Runs 1–2; **no refitting** |
| Bootstrap | 4,000 patient-cluster replicates, seed 42, same resampled patients for both models in a pair |
| Artifacts | `reconciled_metrics.csv`, `paired_comparisons.csv` |
| Outcome | Both historical discrepancies explained as an aggregation difference; see `FINAL_RESULTS.md` |

## Run 5 — mean-function comparison

| | |
|---|---|
| What | K1/K2/K3 fitted twice on the unfiltered cohort, `fit_mean=True` vs `False` |
| Protocol | identical data, kernel, jitter, normalisation, optimiser (L-BFGS-B + analytic gradients), 4 restarts |
| Outcome | gap **+473.29** (K1, K2), **+473.83** (K3) in favour of the fitted mean, for 2 extra parameters |
| Corroboration | an independent 16-restart Nelder-Mead search reached **−6718.50**, matching L-BFGS-B exactly |
| Caveat | in-sample marginal likelihood only; a held-out zero-mean LOPO remains outstanding |

## Run 6 — D1 and E2 LOPO (new)

| | |
|---|---|
| When | 2026-09-29, from 23:05 |
| Command | `python scripts/11_gpr_trajectory_prediction.py --stage lopo --cohorts D1,E2 --jobs 12` |
| Code state | includes the checkpointing fix (defect 1) |
| Checkpoints | `outputs/step3_gpr/checkpoints/{D1,E2}_{predictions,fits,multistep}.csv`, flushed every 25 folds, atomic |
| Expected | D1 641 folds ≈ 42 min at 3.7 s/fold; E2 596 folds ≈ 37 min |
| Resume | rerun the identical command — completed patients are skipped automatically |

### Resume procedure if interrupted

```bash
cd "D:/SEM 7/BTP" && python scripts/11_gpr_trajectory_prediction.py \
    --stage lopo --cohorts D1,E2 --jobs 12
```

Safe because a fold depends only on the cohort and the seed, never on which
folds ran before it. Verified by `TestCheckpointing::test_resumed_run_matches_
an_uninterrupted_run`.

---

## Reproducing the whole pipeline from the frozen panel

```bash
cd "D:/SEM 7/BTP"
python scripts/11_gpr_trajectory_prediction.py --stage cohorts          # seconds
python scripts/11_gpr_trajectory_prediction.py --stage lopo --jobs 12   # hours
python scripts/12_gpr_reconcile_metrics.py                              # ~1 min
python scripts/11_gpr_trajectory_prediction.py --stage report           # seconds
python -m pytest -q                                                     # ~65 min
```

Runtime assumes 16 logical cores and an otherwise idle machine. `--jobs 12`
gives ≈4.5× speedup, not 12×, because the work is compute-bound on ~8 physical
cores.

---

## Progress log

| Date | Stage | Status |
|---|---|---|
| 2026-09-29 | 1 — inventory | done — `REPOSITORY_INVENTORY.md` |
| 2026-09-29 | 2 — code / leakage audit | done — `METHODOLOGY_AUDIT.md`, `LEAKAGE_AND_VALIDATION_AUDIT.md`; 2 defects fixed, 11 regression tests added |
| 2026-09-29 | 3 — metric reconciliation | done — both discrepancies resolved |
| 2026-09-29 | 4 — D1 / E2 | running (Run 6) |
| 2026-09-29 | 5 — sensitivity | mean-function done; kernel comparisons done for the completed cohorts |
| 2026-09-29 | 6 — report stage | pending D1/E2 |
| 2026-09-29 | 7 — interpretation | pending |
| 2026-09-29 | 8 — final validation | pending |
| 2026-09-29 | 9 — git release | pending; **nothing pushed** |
