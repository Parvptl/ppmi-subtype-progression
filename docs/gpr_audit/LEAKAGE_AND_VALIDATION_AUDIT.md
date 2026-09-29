# Step 3 (GPR) — leakage and validation audit

Audit date **2026-09-29**. Every claim below was checked against the running
code and the 24,363 + 23,886 saved prediction rows, not against documentation.

---

## 1. What task is being measured

This has to be stated before any number means anything.

> For held-out patient *i*, visit **k** is predicted from visits **0 … k−1 of
> that same patient**, using a model fitted on the other N−1 patients only.

This is **personalised forecasting for a patient who already has follow-up**.

It is **not**:

- prediction for a brand-new patient with no history (the project's
  `05_new_patient_model.py` addresses that, and the HMM is the *worst* model
  there because it has nothing to filter);
- the HMM's subtype-classification task.

Using the held-out patient's own earlier visits is the *point* of the task, not
a leak. What would be a leak is those visits entering the **fit**, and they do
not.

---

## 2. Every fitted component, and where it comes from

`gpr_loo.FoldData.sequences(exclude=i)` is the single route by which a training
set is assembled. Traced through `run_fold`:

| Component | Fitted on | Held-out patient involved? |
|---|---|---|
| Kernel hyperparameters θ (per kernel) | `fit_gpr(train, …)` | no |
| Mean function β₀, β₁ | GLS inside the same fit | no |
| `hbayes` posterior | `fit_growth_model` on a panel built from `train` | no |
| Population-only baseline β | `_pooled_ols(train)` | no |
| Predictive SD for all four baselines | `_baseline_residual_sd(train, β)` | no |
| Cohort membership | that patient's **own** visits, via the extreme criterion | n/a — not fitted |
| Subtype thresholds | fixed at 0.90 / 1.15 from `config.yaml`; never tuned | no |

**There is no global preprocessing to leak through.** The ratio is computed per
visit from raw MDS-UPDRS items far upstream. No centring, scaling, imputation
or normalisation is computed over the cohort anywhere in the Step 3 path — a
deliberate property, and the reason the usual "scaler fitted on all data"
failure mode cannot occur here.

**No model selection on test data.** K0–K3 are all reported; none was chosen by
held-out performance and then re-reported as the winner.

---

## 3. The automated battery

`gpr_loo.check_predictions` — all ten run on every completed cohort.

| Check | unfiltered | A4 |
|---|---|---|
| every model scored every prediction point | PASS | PASS |
| held-out patient is the patient being predicted | PASS | PASS |
| every fold trained on all other patients (`n_train == N−1`) | PASS | PASS |
| history is strictly the visits before the target | PASS | PASS |
| target is later than the end of its history | PASS | PASS |
| no prediction has an empty history | PASS | PASS |
| subtype probabilities sum to one | PASS | PASS |
| predictive standard deviations are positive | PASS | PASS |
| no predicted or observed value is missing | PASS | PASS |
| each (patient, visit) has one observed value across models | PASS | PASS |

The third check is the load-bearing one: `n_train_patients` is recorded on
every row at fit time, and equals N−1 on all 24,363 unfiltered and 23,886 A4
rows.

The fourth is enforced twice — once by construction, since `trajectory.history_of`
is the only place a history is sliced and it raises outside `1 ≤ k < T`, and
once after the fact by comparing `history_length` to `target_index` on every row.

---

## 4. Index-level protection, not just time-level

Times alone are insufficient: two visits can share a timestamp when a date
fails to parse and the nominal visit year is substituted. `check_no_leakage`
therefore validates the **index arithmetic** as well as the times, so a
same-timestamp pair cannot slip through a strict `>` comparison.

Verified property, on every row of both cohorts:

```
history_length == target_index      and      t_target >= t_last_history
```

---

## 5. Baseline fairness

All nine models are scored on the identical `(patient, target_index)` set. This
is **enforced rather than assumed**: `12_gpr_reconcile_metrics.py` pivots
predictions wide by model and aborts if any point lacks a prediction from any
model, so a paired comparison cannot silently drop rows. Zero such rows in
either completed cohort.

Two points that could otherwise bias a comparison, both handled:

- **Point baselines need distributions.** Coverage, CRPS and subtype
  probabilities require a predictive spread. Each baseline's Gaussian SD is
  estimated from that baseline's own one-step residuals **on training patients**,
  applied exactly as it will be applied to the held-out patient. Never from the
  held-out patient.
- **`patient_ols` degenerates on thin history.** With one observation, or a
  zero time span, there is no line; it carries the last value forward. Applied
  identically in every cohort and documented — it is why that baseline is worst
  at short histories.

**No failed predictions were dropped.** Every model produced a value at every
point in both cohorts; the denominators are therefore complete, and no fold was
excluded from a reported count.

---

## 6. Comparability with earlier project results

| | HMM (Steps 1–2) | GPR (Step 3) |
|---|---|---|
| Target | next categorical subtype | next continuous ratio |
| Scored points | 3,361 | 2,707 (unfiltered) |
| Cohort | 934 patients (≥2 labelled visits) | 821 patients (≥2 **finite-ratio** visits) |
| Headline metric | accuracy | MAE on the ratio |

The GPR cohort is smaller because the ratio is undefined when the PIGD subscore
is zero — 692 of 4,295 labelled visits, of which 634 are TD. That loss is
**not random**, and it means the GPR evaluation systematically under-represents
the most tremor-dominant visits.

**The 75.4% HMM accuracy and any GPR number must not be compared directly.**
The `accuracy` column in the GPR tables is a derived secondary quantity
computed on a different set of points against a different model class; it is
reported for internal comparison between the nine GPR-protocol models only.

---

## 7. Residual risks, stated rather than resolved

| Risk | Status |
|---|---|
| Undefined-ratio visits are excluded non-randomly | **Unresolved by design.** Right-censoring was explicitly ruled out of scope for Step 3. Counted and reported, not corrected. |
| Gaussian likelihood on a non-negative, right-skewed, sometimes-zero quantity | Real. Quantified via `negative_mass` per row and interval coverage. See `FINAL_RESULTS.md`. |
| A patient's cohort membership depends on their own visits | Not leakage — no fitted quantity is involved — but it does mean the cohorts are not random subsets, which is why every cohort is reported with its own recomputed baselines. |
| In-sample marginal likelihood used for the mean-function comparison | Acknowledged. A held-out zero-mean LOPO run is listed as outstanding. |
