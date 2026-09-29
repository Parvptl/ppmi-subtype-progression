# Step 3 (GPR) — final audit checklist

Date **2026-09-30**. PASS requires evidence, not absence of an error.

| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | Full test suite runs | **PASS** | `256 passed in 27.24s`; 256 collected, 0 skipped/deselected |
| 2 | Pre-existing research unbroken | **PASS** | all 144 non-Step-3 tests pass |
| 3 | Leakage battery, all cohorts | **PASS** | 10/10 checks on each of unfiltered, A4, D1, E2 |
| 4 | Held-out patient excluded from every fitted component | **PASS** | `n_train_patients == N−1` on all rows in all four cohorts |
| 5 | History strictly precedes target | **PASS** | `history_length == target_index` on every row; indices *and* times checked |
| 6 | Cohort counts match the data | **PASS** | code vs file: 821/2707, 803/2654, 641/2219, 596/1942 — exact |
| 7 | No invalid fold counted as successful | **PASS** | 11,444 kernel fits, 100% converged; no fold excluded from any denominator |
| 8 | Every model scored on identical points | **PASS** | wide pivot aborts on any unscored point; zero found |
| 9 | Metrics verified against prediction rows | **PASS** | `12_gpr_reconcile_metrics.py` recomputes independently; CRPS checked against numerical integration |
| 10 | Uncertainty matches the reported estimand | **PASS** | bootstrap recomputes the reported statistic inside each replicate; both aggregations reported |
| 11 | Historical discrepancies resolved | **PASS** | both traced to aggregation; see `FINAL_RESULTS.md` §3 |
| 12 | Figures reproducible from documented commands | **PASS** | 13/13 regenerated from the four audited prediction files |
| 13 | Figures inspected, not just generated | **PASS** | all 13 opened; 2 defects found in `model_comparison.png` and fixed |
| 14 | Cohort stage deterministic | **PASS** | re-ran `--stage cohorts`; MD5s byte-identical |
| 15 | Frozen panel unchanged | **PASS** | `fb038ddbdc19ab3368e13940f398e7ba` throughout |
| 16 | Originals backed up before any rerun | **PASS** | 11 files in `.backup_step3/` (gitignored) |
| 17 | Mid-term deck and report untouched | **PASS** | mtimes unchanged: Sep 4, Sep 10, Sep 23 |
| 18 | Seeds / env / commands recorded | **PASS** | `RUN_MANIFEST.md` |
| 19 | Resume procedure exists and is tested | **PASS** | checkpointing; resumed run **identical** to uninterrupted |
| 20 | D1 / E2 completed | **PASS** | 35.4 min and 28.0 min; all checks passed |
| 21 | Claims match evidence | **PASS** | K3>K1 and K0≠hbayes both **downgraded**; see §8 of `FINAL_RESULTS.md` |
| 22 | GPR not compared with the HMM as like-for-like | **PASS** | stated in `report.txt`, `FINAL_RESULTS.md`, `RESEARCH_INTERPRETATION.md` |
| 23 | Held-out zero-mean comparison | **BLOCKED** | in-sample only; requires another LOPO run. Declared, not hidden. |
| 24 | Transformed / censored likelihood | **NOT APPLICABLE** | explicitly out of scope for Step 3; recorded as next step |
| 25 | Article methodology confirmed | **PASS** | article obtained 2026-09-29 and assessed; two deviations documented with measurements |
| 26 | Nothing pushed without approval | **PASS** | `origin/main` still `7341f38`; no push attempted |

## Summary

**24 PASS · 1 BLOCKED · 1 NOT APPLICABLE · 0 FAIL**

The single BLOCKED item (#23) is a comparison that was never run, not a
comparison that failed. It is listed as outstanding in `FINAL_RESULTS.md` §9 and
as next step 3 in `RESEARCH_INTERPRETATION.md`.
