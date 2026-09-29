# Step 3 (GPR) — figure and table audit

Date **2026-09-30**. Every figure was opened and inspected, not merely
confirmed to exist. `--stage report` exiting 0 is not evidence.

## Regeneration

```
python scripts/11_gpr_trajectory_prediction.py --stage report --cohorts unfiltered,A4,D1,E2
```

Inputs are the four audited `gpr_predictions_*.csv` files. Confirmed the run
consumed the newly generated D1/E2 files, not stale ones: the cohort-comparison
table in `report.txt` carries all four cohorts with the correct point counts
(2,707 / 2,654 / 2,219 / 1,942).

## Figures — 13 of 13 generated and inspected

| # | File | Question | Verdict |
|---|---|---|---|
| 1 | `ratio_distribution.png` | Where do ratio values sit, and how much of the scale is unusable? | OK |
| 2 | `patient_ratio_statistics.png` | Is the ratio a stable personal level or does it roam? | OK |
| 3 | `example_trajectories.png` | What does one patient's trajectory look like? | OK |
| 4 | `extreme_trajectories.png` | Are flagged patients persistently extreme? | OK |
| 5 | `filtering_effect.png` | What does each criterion cost, and does it change the task? | OK |
| 6 | `example_forecast.png` | Does a forecast look right point by point? | OK |
| 7 | `subtype_probability_trajectories.png` | Subtype probability over time | OK |
| 8 | `error_by_stratum.png` | Does more history help, does a longer horizon hurt? | OK |
| 9 | `predicted_vs_observed.png` | Tracking individuals or predicting the average? | OK |
| 10 | `calibration.png` | Is the stated uncertainty honest? | OK |
| 11 | `model_comparison.png` | Does the flexible GP beat the linear model and the trivial ones? | **2 defects found and fixed** |
| 12 | `cohort_comparison.png` | Does filtering change the answer or only the population? | OK |
| 13 | `confusion_panel.png` | Where do the derived subtype predictions land? | OK |

## Defects found and fixed

| Figure | Defect | Why it mattered | Fix |
|---|---|---|---|
| `model_comparison.png` | Left panel titled "One-step-ahead **accuracy**" while the axis was MAE | In this project "accuracy" denotes the categorical subtype metric. A reader could take the 0.971 bar for an accuracy and compare it with the HMM's 75.4% — exactly the invalid comparison the audit is meant to prevent. | Retitled "One-step-ahead **error on the ratio**" |
| `model_comparison.png` | K1, K2, K3 labels overprinted in the right panel | Their CRPS differs by <0.0003, so three labels stacked illegibly | Labels for near-coincident points are now fanned vertically |

Both fixed in `gpr_plots.py` and all figures regenerated.

## Checks applied to every figure

| Check | Result |
|---|---|
| Axis labels carry units, and "lower is better" where ambiguous | pass |
| No truncated or misleading axis baseline | pass — bar charts start at 0 |
| Cohort named in the title or caption wherever a cohort-specific number is shown | pass |
| Uncertainty shown where a point estimate could be over-read | pass — forecast bands, calibration, coverage axis |
| No observation plotted twice | pass |
| Model colours consistent across every figure | pass — fixed `MODEL_COLORS` map |
| Numbers on figures match `reconciled_metrics.csv` | pass — spot-checked K3 0.971, K1/K2 0.972, K0 1.005, hbayes 1.006, persistence 1.043 |
| Historical and newly computed results not silently mixed | pass — every figure is built from the four audited prediction files in one run |
| Incomplete cohorts identified | n/a — all four cohorts complete |

## Tables in `report.txt`

| Check | Result |
|---|---|
| Cohort point counts match the prediction files | pass (2,707 / 2,654 / 2,219 / 1,942) |
| Per-cohort MAE matches `reconciled_metrics.csv` | pass |
| Cross-cohort MAEs not presented as directly comparable | pass — the section states each cohort is scored on its own points with its own baselines |
| Subtype accuracy not presented beside the HMM's 75.4% | pass — `report.txt` states the two are not comparable and gives the reason |

## Known limitation of the generated report

`report.txt` carries the **pooled** MAE only, and its uncertainty statements come
from the report stage rather than from the audited bootstrap. The authoritative
uncertainty is `paired_comparisons.csv` and `docs/gpr_audit/FINAL_RESULTS.md`,
which report both aggregations with patient-clustered intervals. Where the two
differ in emphasis, `FINAL_RESULTS.md` supersedes `report.txt`.

## Artifact locations

```
outputs/step3_gpr/
    report.txt                          narrative log
    reconciled_metrics.csv              per model per cohort, both aggregations
    paired_comparisons.csv              paired differences + bootstrap CIs
    model_comparison.csv                report-stage metrics, all models
    gpr_metrics.csv / baseline_metrics.csv
    stratified_metrics.csv              by history length and horizon
    trajectory_metrics.csv              multi-step forecasts
    gpr_predictions_{unfiltered,A4,D1,E2}.csv
    gpr_fold_fits_{unfiltered,A4,D1,E2}.csv
    trajectory_predictions_{unfiltered,A4,D1,E2}.csv
    checkpoints/                        resume state
    figures/                            13 PNGs
```
