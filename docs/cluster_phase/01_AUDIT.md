# Cluster phase — Stage 1 audit

Audit date **2026-10-14**, on the IIT Palakkad cluster (`master` login node).
Compiled by reading the repository, not from prior summaries. Where a claim
below depends on a saved output that was not yet on the cluster at audit time,
it is marked **(unverified here)**.

---

## 1. Git and file state at audit start

| | |
|---|---|
| Checkout | `~/projects/ppmi-subtype-progression`, fresh clone 2026-10-14 14:31 |
| Branch | `main`, up to date with `origin/main`, working tree clean |
| HEAD | `bd36824` "Add GPR progression audits, subscore models, and research reports" |
| Tracked code | 21 scripts (`00`–`20`), 24 modules in `src/ppmi_subtype/`, 12 test files |
| **`data/`** | **absent** (gitignored, PPMI DUA) |
| **`outputs/`** | **absent** (gitignored, contains participant-level rows) |
| **`experiments/`** | **absent** (gitignored) |

Every result in the reports was therefore, at audit start, a claim about files
on the Windows machine (`D:\SEM 7\BTP`). The user then transferred `data/` and
`outputs/` (2026-10-14, 15:13–15:15). Gate results: the transferred
`outputs/visit_panel.csv` has the frozen MD5 `fb038ddbdc19ab3368e13940f398e7ba`;
a panel rebuilt on the cluster from the raw files is cell-identical to it
(only Windows CRLF line endings differ); `outputs/` was then made read-only and
an MD5 manifest of its 254 files recorded under `experiments/_manifests/`.

### Tests

`python -m pytest -q` on `node004`, Python 3.13.16 / numpy 2.2.4 / scipy 1.15.2:
**365 passed in 34 s.** No test reads PPMI data; all are simulation-based.

---

## 2. What each experiment is, and its status

Status key — **C** complete with a valid held-out comparison ·
**P** partial (held-out for some arms only) · **X** exploratory / in-sample
only · **M** missing.

| # | Script → output | Question | Status | Notes |
|---|---|---|---|---|
| 01 | `01_build_panel` → `visit_panel.csv` | labelled panel | C | 100% / 99.87% agreement with PPMI's own labels |
| 02 | `02_transition_analysis` → `transitions/` | empirical transitions | C (descriptive) | |
| 03 | `03_progression_model` → `model/` | DTMC/CTMC/HMM, 10 train/test splits | C | superseded as headline by LOO (09) |
| 04 | `04_hierarchical_bayes` → `hbayes/` | growth model | C | |
| 05 | `05_new_patient_model` → `newpatient/` | first-visit model | C | |
| 07 | `07_longitudinal_cv` → `validation/` | withhold future visits | C | |
| 09 | `09_loo_validation` → `loo/` | 3-state HMM, 934-fold LOPO | C | 75.4% accuracy, 3,361 points; **predicts Indeterminate once in 3,361** |
| 10 | `10_..._5hidden_shard` → `loo_5hidden/` | 5-state HMM LOPO | **P** | "converged" 404/934 (43%); see §3.6 |
| 11 | `11_gpr_trajectory_prediction` → `step3_gpr/` | ratio GP K0–K3, 4 cohorts | C | 11,444 fits |
| 13 | `13_gpr_mean_function_holdout` | fitted vs zero mean | **P** | unfiltered cohort only |
| 14/15 | multistep analysis / audit | horizon effects | **P** | K0–K3 only; no `hbayes`/LMM multistep |
| 16 | `16_gpr_subscore_baseline` → `subscore_*` | independent K1 GP on subscores | C | |
| 17 | `17_joint_subscore_lmm` → `joint_subscore/` | joint random-intercept LMM | C | random intercepts only |
| 18 | `18_subscore_multioutput_gp` → `joint_subscore_gp/` | joint vs independent GP | **P** | K0-full, K1-full, K1-diag, indep-K1 held out; **K2/K3 population-only (X)** |
| 19 | `19_time_varying_progression` → `time_varying/` | piecewise rates | C (fixed knots) / X (inference) | Wald CIs; see §3.8 |
| 20 | `20_clusterability_audit` → `clustering_audit/` | is clustering justified | X | one null replicate; see §3.9 |
| — | GP-vs-LMM matched comparison | | **M as a file** | computed in-session, never saved |
| — | GP-vs-HMM derived-subtype comparison | | **M as a file** | computed in-session, never saved |
| — | HMM stability across inits/floors/state counts | | **M** | only `_multistart5`/`_dispersed5` probes, in-sample |
| — | Calibration (log score / CRPS / PIT) for subscore models | | **M** | coverage-95 only |

---

## 3. Stale, inconsistent or under-supported claims

Each item names where the claim is and what the code/outputs actually support.

### 3.1 Test counts and README are stale
- `docs/gpr_audit/TEST_RESULTS.md`: "256 tests". Current suite: **365**.
- `README.md`: "144 tests", "~1,550 lines", repository layout lists scripts
  `00`–`07` only, status table ends before the GPR work. The README does not
  describe 2/3 of the current codebase.

### 3.2 `NEXT_RESEARCH_STEPS.md` contradicts itself
- §1 "Established" lists **"The GP's advantage over baselines widens with
  forecast horizon"** as ✅. §2's own revision (script 15) shows the widening is
  not detectable within three years and excludes zero only at 5y+ in 3/4
  cohorts. The ✅ row was not updated.
- §1 lists **"linear kernel term is redundant given a fitted linear mean"** as ✅.
  `FINAL_RESULTS.md` §9 correction says that reading is wrong (`var_linear` pins
  to its bound in the zero-mean arm too). The ✅ row was not updated.

### 3.3 Two important comparisons have no saved output
`GPR_onwards_explanatory_report.md` appendix and §E.9 admit this:
- **multi-output GP vs LMM** (−0.0155 tremor, −0.0217 PIGD) — "computed during
  the working session … not itself stored as a saved CSV".
- **subscore GP vs HMM derived subtype** (0.7566 vs 0.7539, +0.0027
  [−0.0042, +0.0098]) — same.
Both are among the report's headline results. Neither can currently be traced
to a file. Stage 3 regenerates them as saved outputs.

### 3.4 "Joint modelling gives no held-out benefit" is scoped to K1
Report §I.1 #8 and §E.14. Only the K1 full/diag pair was evaluated held out.
K2/K3 (whose in-sample constant-component correlations read +0.85 / +0.98) were
fitted on all 934 patients only (§E.7, Q27). The claim is valid for K1 and
untested for K2/K3. The reason given was runtime (250–320 s/fit, "65+ hours");
on this cluster that becomes a few hours (Stage 2).

### 3.5 Calibration evidence for subscore models is thin, and the persistence
### baseline's uncertainty is ad hoc
- Script 18 reports only 95% coverage. No log predictive density, CRPS, PIT,
  or multi-level coverage. "Our models are well calibrated" (§I.5) rests on
  coverage-95 plus the earlier ratio-scale CRPS.
- `persistence` predictive SD in script 18 is the history SD (ddof=1) or a
  **hard-coded 0.3** when k=1. Its coverage (0.81 tremor, 0.85 PIGD) and any
  probabilistic comparison against it therefore depend on an arbitrary
  constant. Stage 3 reports point metrics for persistence and treats its
  probabilistic metrics as a labelled, assumption-dependent sensitivity.

### 3.6 The 5-state HMM "convergence 404/934" is not a convergence rate
`hmm.fit_hmm` uses L-BFGS-B **without an analytic gradient** (finite
differences), `ftol=1e-11`, `gtol=1e-9`, and `converged = res.success`. With
finite-difference gradients, `success=False` is usually
`ABNORMAL_TERMINATION_IN_LNSRCH` at a point that is a numerical optimum, not a
failure to optimise. The termination message and gradient norm were never
recorded, so the 43% figure (report Q, midterm slide) cannot distinguish "did
not reach an optimum" from "line search could not make progress at the
tolerance requested". Stage 4 records both.

**Update after the Stage 4 population runs (exploratory, all 934 patients).**
For the five-state model the hypothesis above is only partly right. With
scipy's default evaluation budget (maxfun = 15,000, what `fit_hmm` uses) 36 of
40 dispersed starts stopped at `TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT` —
finite differences over 34 parameters cost 35 evaluations per gradient — so
those are genuinely unconverged, not benign line-search stops. With a budget of
400,000 every start converges, but to 33 distinct optima; one start in 40
reaches the best. The 404/934 figure therefore describes fits that were both
budget-limited and start-dependent. See `04_RESULTS_GENERATED.md`, Stage 4 A.

### 3.7 HMM restarts were not dispersed, and the emission floor was chosen with
### held-out data
- Restarts perturb one fixed start by N(0, 0.4); `_multistart5.py` found every
  outer seed bit-identical. So "4 restarts" never tested multimodality.
- `hmm.py` justifies `DEFAULT_EMISSION_FLOOR = 0.5` partly because it "costs
  nothing in held-out performance — +0.027 … against +0.026 unconstrained".
  That comparison used the same held-out splits later reported. It is a mild
  selection-on-test issue. Stage 4 selects any configuration by inner
  (training-only) criteria.
- Prediction is argmax of P(next label). Indeterminate is never the argmax, so
  Indeterminate recall is 0 **by construction of the decision rule**, not
  necessarily because the probabilities are uninformative. This has not been
  separated.

### 3.8 Temporal analysis: intervals are model-based and the time axis is
### follow-up time
- `timevarying.FitResult.period_slopes` / `slope_contrasts` use Wald intervals
  from `cov_beta` (the LMM's model-based covariance). These assume the LMM's
  covariance is right; with only random intercepts in the primary model and
  informative dropout, they are likely too narrow. No patient-cluster bootstrap.
- `years_from_baseline` is time since **study enrolment**, not disease duration.
  "PIGD accelerates after about five years" is a statement about follow-up time.
  Patients enrol at different disease durations, so this is not evidence of a
  disease-stage transition. The report's §I.3 #2 says this correctly; §I.1 #12
  and §F.6 phrasing ("after about five years … the gait side nearly doubles its
  speed") can be read as a disease-course claim.
- The 5y+ period rests on 273 patients who differ at baseline (lower severity,
  p < 1e-4). No attrition model, no completer/pattern-mixture sensitivity.

### 3.9 Clusterability: one null replicate
`real_vs_null_model_selection.csv` compares the real data with **a single**
simulated no-groups dataset. "Null BIC prefers K=2 by 87" and "real
improvement 513 vs 88" are single draws; the null distribution of these
statistics was never estimated. Stage 6 replicates the null.

### 3.10 Two different "persistence" baselines
Scripts 17 and 18 both report a `persistence` arm with identical point
predictions but different predictive SDs (script 17: history SD, else the
fitted residual SD; script 18: history SD, else a hard-coded 0.3). Their
coverage and any probabilistic score for "persistence" therefore differ between
the two reports for the same predictions.

### 3.11 REML likelihoods saved across different mean structures
`outputs/time_varying/model_summary.csv` stores REML −2logL for models with
different fixed effects (different knot sets). REML likelihoods are not
comparable across mean structures; no report was found to compare them, but
the file invites it. Stage 5 compares temporal structures by ML only.

### 3.12 Treatment initiation coincides with the early period
Among all labelled visits in the panel (1,080 patients) the share of exams
from patients on PD medication (`PDTRTMNT`) rises from 0.09 in year 0 to 0.54
in year 1 and above 0.95 from year 5; the 934-patient cohort's own table is
generated by Stage 5 (`timeaxis/treated_share_by_year.csv`). Early follow-up therefore mixes untreated exams with
OFF-state exams in newly treated patients. No earlier analysis adjusted for
this; Stage 5 adds a visit-level treatment indicator as a sensitivity.

### 3.13 "Five years" is follow-up time, and disease duration barely varies
`duration_yrs` in the panel is disease duration at enrolment, constant within
patient (median 0.54 y, range 0.03–3.6 y). Total disease duration is enrolment
duration plus follow-up, so follow-up time and disease stage are separable only
through a ~3.5-year spread in enrolment duration. Any "transition at five
years" is, at best, weakly distinguishable from a disease-duration effect.

### 3.14 Infrastructure
- `scripts/run_5hidden_loo.slurm` targets another cluster
  (`/home/interiit2025/...`, partition `standard`). It will not run here.
- `scripts/20_clusterability_audit.py` imports scikit-learn, which is not in
  `pyproject.toml`.
- Script 18 leakage check 2 is `(a < b).all() or (a <= b).all()`, which reduces
  to the weaker `<=`. Harmless on this data (visit times are distinct), but it
  is not the check its label claims.
- Earlier runs did not persist their invocation, seed or environment (noted in
  `REPOSITORY_INVENTORY.md` §7). The cluster job wrapper written in Stage 2
  records all of them per run.

---

## 4. Result → script → output trace (headline claims)

| Claim | Value | Script | Output file |
|---|---|---|---|
| HMM LOPO accuracy | 75.4% (3,361 pts) | `09_loo_validation.py` | `outputs/loo/` *(unverified here)* |
| 5-state HMM "converged" | 404/934 | `10_…_shard.py` + `merge_5hidden_loo.py` | `outputs/loo_5hidden/` *(unverified)* |
| Ratio GP K1−K0 | −0.0333 [−0.0467, −0.0205] | `12_gpr_reconcile_metrics.py` | `step3_gpr/paired_comparisons.csv` |
| Ratio GP − persistence | −0.0707 [−0.0979, −0.0451] | `12_…` | `step3_gpr/paired_comparisons.csv` |
| Subscore GP vs persistence (PIGD) | +0.0133 [+0.0082, +0.0187] | `18_…` | `joint_subscore_gp/paired.csv` |
| Joint full − diag (K1) | −0.00026 / −0.00013, both incl. 0 | `18_…` | `joint_subscore_gp/paired.csv` |
| MOGP − LMM | −0.0155 / −0.0217 | **none** | **not saved** |
| Subscore GP − HMM subtype | +0.0027 [−0.0042, +0.0098] | **none** | **not saved** |
| Period slopes | e.g. PIGD 5y+ 0.0800 [0.0724, 0.0875] | `19_…` | `time_varying/period_slopes.csv` |
| Slope reliability | 0.270 / 0.531 median | `20_…` | `clustering_audit/random_slope_reliability.csv` |
| Null BIC K=2 | 5131.8 vs 5219.0 | `20_…` | `clustering_audit/real_vs_null_model_selection.csv` |

---

## 5. Unresolved research questions

1. **Q-GP1.** With all four arms (persistence, LMM, independent GP, joint GP)
   on identical LOPO folds and points, and K2/K3 included held out, does any
   joint configuration beat its nested diagonal control, or the independent GP?
2. **Q-GP2.** Are the subscore models calibrated (log score, CRPS, PIT,
   multi-level coverage), and how does subtype discrimination compare once
   balanced accuracy, per-class recall and log loss are reported?
3. **Q-GP3.** Are the in-sample K2/K3 constant-component correlations
   (+0.85/+0.98) a held-out signal or a component trade-off?
4. **Q-HMM1.** Is the HMM likelihood surface multimodal under genuinely
   dispersed initialisation? How much do Q, E and held-out predictions vary
   across inits?
5. **Q-HMM2.** What does "404/934 converged" actually mean (termination reason,
   gradient norm, distance from the best optimum)?
6. **Q-HMM3.** Does the HMM's zero Indeterminate recall come from the argmax
   rule or from uninformative probabilities? What do balanced accuracy, log
   loss and per-class recall show across emission floors and state counts,
   with configuration chosen by training-only criteria?
7. **Q-T1.** Are the piecewise-rate conclusions robust to patient-bootstrap
   inference, a wider and pre-registered knot grid, smooth alternatives, and the
   choice of time axis?
8. **Q-T2.** How much do late-follow-up sparsity and informative attrition
   change the late-period slope?
9. **Q-C1.** Across many no-groups null replicates, where does the real data's
   clustering signal fall, and does the trajectory-feature reliability change
   when visits are removed or follow-up is truncated?
