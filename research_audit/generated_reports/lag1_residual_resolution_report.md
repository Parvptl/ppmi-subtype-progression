# E-1 — Lag-1 residual autocorrelation: resolving a contradiction in `outputs/`

## 1. Experiment overview

| | |
|---|---|
| **Experiment name** | E-1, lag-1 residual resolution |
| **Identifier in this audit** | E-1 |
| **Research question** | `outputs/exploration/report.txt` concludes that nothing in the residual structure argues for a state-space specification. Is that conclusion what the committed code produces? |
| **Objective** | Locate the contradiction, determine whether it is a live code defect or a stale artefact, and re-run the statistic under a stronger test |
| **Run** | `experiments/20261103T191408Z_e1_lag1_residual_resolution/`, 2026-11-03 |
| **Script** | `scripts/76_lag1_residual_resolution.py` |
| **Status** | Completed and verified; report written by this audit |
| **Related** | G-01 (GPR kernel comparison, independent corroboration), B-01 (the growth model whose specification is at issue), P-06 (the one state-space-like model actually fitted) |

## 2. Scientific motivation

The project's hierarchical Bayesian growth model (`scripts/04`) assumes that a
patient's deviations from their own linear trend are independent between
visits. If they are not — if a deviation persists into the next visit — the
growth form is the wrong family and a state-space form is indicated, in which
an unobserved level accumulates process noise and is observed with error.

`outputs/exploration/report.txt` carries the lag-1 residual statistic and a
sentence closing the question: *"Nothing here argues for the state-space
alternative."* That sentence has been read as settling the specification
choice. Before anything was built on it, it needed checking.

The question matters beyond bookkeeping: the growth model's two variance terms
are not separately identifiable, and the state-space form is the one in which
they are. A second, independent reason to prefer the state-space form would
change how the modelling choice is justified.

## 3. Dataset and cohort

| | |
|---|---|
| **Dataset** | Frozen visit panel, md5 `fb038ddbdc19ab3368e13940f398e7ba` |
| **Labelled visits** | 4,441 |
| **Visits with a finite ratio** | 3,716 (the PIGD subscore is zero at 725) |
| **Eligible patients** | 397 — at least 4 finite-ratio visits and at least 1 year of span |
| **Consecutive residual pairs** | 2,050 |
| **Target** | the lag-1 autocorrelation of residuals from each patient's own linear trend |

The eligibility rule is the committed script's, not a new one.

## 4. Methodology

Four steps, in order.

**(a) Locate the sentence.** The phrase was searched for in the committed
`scripts/06_ratio_exploration.py`, in all 45 commits touching the repository,
and whitespace-insensitively across the working tree.

**(b) Reproduce the statistic.** The committed code path was run and its
numbers compared against the published ones.

**(c) Strengthen the test.** The stale report placed an interval on the null
only, using 200 draws. This run uses 2,000 null draws, computes a one-sided
Monte Carlo p-value, and adds a 2,000-draw **patient-clustered bootstrap on
the observed statistic**, which the original did not have.

**(d) Re-run the script and diff.** `scripts/06` was executed unmodified with
`paths.outputs` redirected into the run directory through a shim config, so
that `outputs/` — which is read-only by project rule — was not touched. The
regenerated report was diffed against the committed one.

## 5. Experimental configuration

| Setting | Value |
|---|---|
| Null draws (this run) | 2,000 (the published report used 200) |
| Bootstrap draws | 2,000, patient-clustered |
| Seed | 42 |
| Eligibility | ≥ 4 finite-ratio visits, ≥ 1 year span |
| σ̂ = √(MS within) | 1.2162 |
| Shim config | `config_rerun.yaml` in the run directory |
| `outputs/` | opened read-only; never written |

## 6. Results

### 6.1 The contradiction is a stale artefact, not a code defect

| Check | Result |
|---|---|
| Phrase present in `outputs/exploration/report.txt` | **yes** |
| Phrase present in committed `scripts/06_ratio_exploration.py` | **no** |
| Commits touching `scripts/06` | 1 (`4e5bdd2`) |
| `git grep` across the working tree | no match |
| Whitespace-insensitive search across all 45 commits | no match |

**The committed script cannot produce that sentence.** It was written by an
uncommitted earlier draft and never regenerated, because `outputs/` became
read-only.

### 6.2 Every published number reproduces

| Quantity | Published | This run |
|---|---|---|
| Eligible patients | 397 | 397 |
| Consecutive residual pairs | 2,050 | 2,050 |
| Observed lag-1 correlation | −0.2195 | −0.2195 |
| σ̂ | 1.2162 | 1.2162 |
| Null (committed recipe, 200 draws) | mean −0.3321, 95% [−0.3648, −0.2940] | identical |

The arithmetic was never in doubt. Only the sentence drawn from it was.

### 6.3 The stronger test

`lag1_statistic.csv`:

| Quantity | Estimate | 95% interval | Interval type |
|---|---|---|---|
| Observed lag-1 correlation | **−0.2195** | −0.2731 – −0.1621 | patient bootstrap |
| iid null, same visit pattern | −0.3328 | −0.3689 – −0.2951 | Monte Carlo, 2,000 draws |
| **Observed minus null** | **+0.1133** | **+0.0457 – +0.1785** | combined resample |

One-sided Monte Carlo p-value (null ≥ observed): **5.00 × 10⁻⁴**. The observed
value lies outside the null interval and is **more positive** than independent
measurement error on the same visit pattern would produce.

### 6.4 The regenerated report says the opposite

Re-running the committed script produces 13 differing lines
(`report_diff.txt`). The substantive change:

> **stale:** *"iid noise on the same visit pattern, so independent deviations
> are a defensible assumption and the growth form is adequate. Nothing here
> argues for the state-space alternative."*
>
> **regenerated:** *"…signature of accumulating process noise and favours the
> state-space form over the growth form the proposal settled on — and the
> state-space form is also the one in which the observation and [process
> variances are separately identified]."*

Section 5 of the committed report is **correct**; its closing item 4 is
**stale**.

### 6.5 Independent corroboration already in the repository

Three pieces of evidence, none produced for this purpose:

* **GPR kernel comparison** (`docs/gpr_audit/FINAL_RESULTS.md` §3.1).
  K1 = constant + RBF beats K0 = constant + linear on held-out MAE in all four
  cohorts under both aggregations — unfiltered −0.0333 [−0.0467, −0.0205]
  pooled, −0.0192 [−0.0279, −0.0106] per patient. An RBF term with a ≈ 3.7-year
  lengthscale *is* temporally persistent deviation from trend: the same
  structure a state-space form posits, reached by a different route.
* **The linear-kernel variance pins at its lower bound** (1 × 10⁻⁶) in
  1,642 / 1,642 unfiltered fits and 821 / 821 folds of the zero-mean arm. The
  data rejects a patient-specific random *slope* while retaining a persistent-
  deviation term. "Process noise accumulates" and "patients differ in slope"
  are not the same claim, and only the first survives.
* **The growth model's own identifiability note** (`src/ppmi_subtype/hbayes.py`):
  σ_obs and σ_proc enter the likelihood only through their sum.

## 7. Interpretation

**What the results support.** Residual deviations from a patient's own linear
trend are more persistent than independent measurement error would produce.
That favours a state-space over a growth specification, and it is a second,
independent reason to prefer it — the first being that only the state-space
form separates the observation and process variances.

**What the results do not support.** The project never fitted a state-space
model to the ratio. This is a statement about model specification, **not** a
claim that a state-space model would predict better. The one state-space-like
model that *was* fitted — the rich clinical profile's local linear trend
(P-06) — found the rate component worth +0.0040 of +0.0733 in held-out
log-density, about 5% of the model's advantage over assume-no-change. On this
evidence the expected prediction gain is small.

**Exploratory or confirmatory.** The re-test was specified in advance of being
run (2,000 draws, one-sided p, patient bootstrap) and its direction was
pre-stated. The corroborating evidence in §6.5 is post-hoc assembly of
existing results.

**The contradiction is not resolved in the file where it lives.** `outputs/`
is read-only by project rule, so `outputs/exploration/report.txt` still
contains the stale sentence and will continue to. This report is the record
that it must not be quoted.

## 8. Limitations

1. **One autocorrelation statistic is thin ground for a modelling choice.**
   That is why §6.5 exists; the conclusion rests on three independent lines,
   not on this one.
2. **The eligible subset is 397 of 934 patients** — those with at least four
   finite-ratio visits. Patients with sparser or shorter follow-up, and the
   725 visits where the ratio is undefined, contribute nothing.
3. **The null is iid noise on the observed visit pattern.** It is not a null
   of "growth model with correctly specified correlated errors"; rejecting it
   does not identify which alternative is right.
4. **No state-space model was fitted to the ratio**, so the specification
   conclusion is untested by prediction.
5. **The stale file cannot be corrected**, only annotated.

## 9. Reproducibility

```bash
~/envs/ppmi/bin/python scripts/76_lag1_residual_resolution.py
```

| Kind | Path |
|---|---|
| Script | `scripts/76_lag1_residual_resolution.py` |
| Script re-run under a shim | `scripts/06_ratio_exploration.py` with `config_rerun.yaml` |
| Run directory | `experiments/20261103T191408Z_e1_lag1_residual_resolution/` |
| Regenerated output | `…/rerun_outputs/exploration/report.txt` |
| Diff | `…/report_diff.txt` |
| Raw draws | `…/bootstrap_draws.npy`, `…/null_draws.npy` |
| Integrity | `MANIFEST.sha256`; panel md5 asserted |

## 10. Manuscript relevance

Supplementary, with one sentence in the main text. The defensible main-text
sentence is the one the run itself proposes: *residual deviations from a
patient's own linear trend are more persistent than independent measurement
error would produce, which favours a state-space over a growth specification.*

It supports the methodological justification for the local-linear-trend form
used by the rich clinical profile. It must not be used to claim a prediction
advantage.

**It also carries an audit obligation:** anyone citing
`outputs/exploration/report.txt` must be directed here first.

## 11. Source evidence

| File | Contents |
|---|---|
| `experiments/20261103T191408Z_e1_lag1_residual_resolution/report.txt` | Narrative report including the verdict |
| `…/lag1_statistic.csv` | §6.3 |
| `…/report_diff.txt` | The 13 differing lines |
| `…/rerun_outputs/exploration/report.txt` | What the committed script actually produces |
| `…/bootstrap_draws.npy`, `…/null_draws.npy` | Raw draws |
| `…/config_rerun.yaml` | The shim that redirected outputs |
| `outputs/exploration/report.txt` | **The stale artefact. Do not quote its closing item 4** |
| `docs/gpr_audit/FINAL_RESULTS.md` §3.1, §4.1 | Independent corroboration |
| `src/ppmi_subtype/hbayes.py` | The identifiability note |
