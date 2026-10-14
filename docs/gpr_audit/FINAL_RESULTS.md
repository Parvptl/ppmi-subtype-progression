# Step 3 (GPR) — final audited results

Date **2026-09-30**. Every number below was recomputed during this audit from
prediction-level rows. Panel MD5 `fb038ddbdc19ab3368e13940f398e7ba`; seed 42.

**Provenance key** — ✅ reproduced historical · 🆕 new · ⚠️ superseded

---

## 0. The task, before any number

Visit *k* of a held-out patient is predicted from visits *0…k−1* **of that same
patient**, under a model fitted on the other N−1 patients. This is
**personalised forecasting for a patient who already has follow-up**, on the
**continuous TD/PIGD ratio**.

It is **not** the HMM's task. The GPR MAE and the HMM's 75.4% accuracy are not
comparable: different target, different metric, different scoring set (2,707 vs
3,361 points), different cohort (821 vs 934 patients).

---

## 1. Cohorts — all four now complete 🆕

| Cohort | Patients | Points | Kernel fits | Converged | Status |
|---|---:|---:|---:|---:|---|
| unfiltered | 821 | 2,707 | 3,284 | **100%** | ✅ reproduced |
| A4 | 803 | 2,654 | 3,212 | **100%** | ✅ reproduced |
| D1 | 641 | 2,219 | 2,564 | **100%** | 🆕 completed in this audit |
| E2 | 596 | 1,942 | 2,384 | **100%** | 🆕 completed in this audit |

**11,444 kernel fits, zero non-convergences.** For contrast, the Step 2
five-latent-state HMM converged in 404 of 934 folds (43%). The GP fits are
well behaved; hyperparameters barely move when one patient of ~800 is removed
(K1 lengthscale IQR 3.7101–3.7147).

All 10 leakage/consistency checks pass on all four cohorts.

---

## 2. Headline table — unfiltered cohort, 821 patients, 2,707 points

MAE is reported under **both** defensible aggregations, because the choice
changes one conclusion.

| Model | MAE pooled | MAE per-patient | RMSE | CRPS | 95% coverage | *subtype acc.* |
|---|---:|---:|---:|---:|---:|---:|
| **K3** const+linear+Matérn-3/2 | **0.9715** | **1.0006** | 1.3565 | 0.7247 | 0.941 | 0.727 |
| **K1** const+RBF | 0.9720 | 1.0012 | 1.3566 | 0.7249 | 0.941 | 0.724 |
| **K2** const+linear+RBF | 0.9720 | 1.0012 | 1.3566 | 0.7249 | 0.941 | 0.724 |
| K0 const+linear ≡ growth model | 1.0053 | 1.0204 | 1.3870 | 0.7472 | 0.939 | 0.714 |
| `hbayes` (the growth model itself) | 1.0062 | 1.0176 | 1.3910 | 0.7480 | 0.930 | 0.712 |
| persistence | 1.0427 | 1.0597 | 1.5396 | 0.8182 | 0.936 | *0.733* |
| patient mean | 1.0627 | 1.0649 | 1.4714 | 0.8003 | 0.941 | 0.710 |
| patient OLS | 1.3357 | 1.3292 | 1.9876 | 1.0409 | 0.939 | 0.713 |
| population only | 1.3949 | 1.4130 | 1.8008 | 0.9836 | 0.948 | 0.576 |

*Subtype accuracy is a derived secondary quantity on the GPR scoring set. It is
**not** the HMM's 75.4% and must not be read beside it.*

---

## 3. The two historical discrepancies — resolved ✅

Both arose from **aggregation**, not from changed predictions, models or
subsets. Pooling weights a patient by visit count; the 134 patients with ≥6
visits carry **40.4%** of pooled weight but **16.3%** of patient weight.

### 3.1 K1 − K0: table said −0.0333, narrative said −0.0192

Both are correct, for different estimands. The narrative's CI [−0.0280,
−0.0109] matches the patient-averaged arm exactly.

| Cohort | Pooled | Per-patient |
|---|---|---|
| unfiltered | **−0.0333** [−0.0467, −0.0205] | **−0.0192** [−0.0279, −0.0106] |
| A4 | −0.0340 [−0.0477, −0.0210] | −0.0208 [−0.0296, −0.0122] |
| D1 | −0.0351 [−0.0519, −0.0202] | −0.0157 [−0.0261, −0.0057] |
| E2 | −0.0312 [−0.0447, −0.0183] | −0.0157 [−0.0249, −0.0069] |

**Conclusion, robust:** a flexible kernel beats the linear one. Eight of eight
intervals exclude zero, same sign, in all four cohorts under both estimands.

### 3.2 K0 − `hbayes`: table said −0.0009, summary said ≈+0.0028

| Cohort | Pooled | Per-patient |
|---|---|---|
| unfiltered | −0.0009 [−0.0033, +0.0014] | +0.0028 [+0.0002, +0.0053] |
| A4 | −0.0014 [−0.0039, +0.0011] | +0.0023 [−0.0005, +0.0049] |
| D1 | −0.0005 [−0.0031, +0.0021] | +0.0039 [+0.0012, +0.0066] |
| E2 | −0.0004 [−0.0032, +0.0023] | +0.0031 [+0.0001, +0.0060] |

**The sign flips between estimands in all four cohorts.** Under the pooled
estimand every interval includes zero.

⚠️ **The earlier claim that K0 and `hbayes` differ significantly is
superseded.** It was an artefact of reporting a patient-averaged interval
beside a pooled point estimate. The correct statement: **K0 and `hbayes` are
statistically indistinguishable**, which is what the analytic equivalence
predicts. That the difference is smaller than the sensitivity to aggregation is
itself the evidence.

---

## 4. Kernel sensitivity

### 4.1 K1 vs K2 — the linear kernel adds nothing, for an identifiable reason ✅

`var_linear` sits at its lower bound (1e-6) in **1642/1642** unfiltered and
**1606/1606** A4 K2+K3 fits (1281/1282 in D1, 1190/1192 in E2).

⚠️ **"K1 and K2 are identical" is superseded.** They agree to **1.4e-04** in
predicted mean and 4.5e-05 in SD — practically but not exactly, because the
pinned `var_linear = 1e-6` still contributes.

⚠️ **"the data rejects linear structure" is superseded.** The mean function
`β₁ = −0.082/yr` already carries the linear trend. The kernel term is redundant
*given* the mean — not absent from the data.

### 4.2 K3 vs K1 — Matérn does **not** beat RBF 🆕

| Cohort | Pooled | Verdict |
|---|---|---|
| unfiltered | −0.0005 [−0.0015, +0.0004] | includes 0 |
| A4 | −0.0005 [−0.0015, +0.0004] | includes 0 |
| D1 | −0.0010 [−0.0020, −0.0000] | marginal |
| E2 | −0.0003 [−0.0013, +0.0007] | includes 0 |

K3 has the lowest MAE in every cohort, but the margin is ~0.0005 ratio units
against a within-patient SD of 1.22. **Three of four intervals include zero and
the fourth only marginally excludes it.** Not established. K3's rank ordering is
not evidence of superiority.

### 4.3 Mean function — fitted linear beats zero mean in sample 🆕

Strictly matched protocol: same data, kernel, jitter, normalisation, optimiser
(L-BFGS-B + analytic gradients verified to ≤8e-09 against finite differences),
same restarts.

| Kernel | Fitted linear mean | Zero mean | Gap | Extra params |
|---|---:|---:|---:|---:|
| K1 | −6245.21 | −6718.50 | **+473.29** | 2 |
| K2 | −6245.21 | −6718.50 | **+473.29** | 2 |
| K3 | −6245.12 | −6718.94 | **+473.83** | 2 |

Corroborated by an independent 16-restart Nelder-Mead search reaching −6718.50.
As a likelihood-ratio test (nested: β₀ = β₁ = 0), 2Δ = 946.6 on χ²₂.

The zero-mean fit inflates `var_const` 0.881 → 4.668, straining the constant
kernel to represent a population mean of 2.28.

**Limitation:** this is *in-sample* marginal likelihood. A held-out zero-mean
LOPO run was not performed and remains outstanding.

---

## 5. Does the GP beat the trivial baselines? ✅ yes, in every cohort

| Cohort | K1 − persistence (pooled) | K1 − patient mean (pooled) |
|---|---|---|
| unfiltered | −0.0707 [−0.0979, −0.0451] | −0.0908 [−0.1200, −0.0631] |
| A4 | −0.0758 [−0.1029, −0.0493] | −0.0965 [−0.1260, −0.0677] |
| D1 | −0.1297 [−0.1594, −0.1003] | −0.1589 [−0.1928, −0.1257] |
| E2 | −0.1428 [−0.1735, −0.1124] | −0.1681 [−0.2028, −0.1322] |

All eight intervals exclude zero.

Note the margin **doubles** in D1/E2. Those criteria remove persistently extreme
patients — precisely the patients for whom "next = last" works best. The
prediction task is genuinely different after filtering, which is why every
cohort carries its own recomputed baselines and cross-cohort MAEs must not be
compared.

---

## 6. Cohort sensitivity — conclusions are stable 🆕

| Model | unfiltered | A4 | D1 | E2 |
|---|---:|---:|---:|---:|
| K3 | 0.9715 | 0.9830 | 0.8807 | 0.8248 |
| K1 / K2 | 0.9720 | 0.9836 | 0.8817 | 0.8251 |
| K0 | 1.0053 | 1.0176 | 0.9168 | 0.8563 |
| `hbayes` | 1.0062 | 1.0189 | 0.9173 | 0.8567 |
| persistence | 1.0427 | 1.0594 | 1.0114 | 0.9680 |
| population | 1.3949 | 1.3945 | 1.0758 | 0.9660 |

**The model ordering is identical in all four cohorts.** Absolute MAE falls as
more extreme patients are removed — the remaining patients are easier to
predict — but no conclusion depends on the extreme criterion.

---

## 7. Predictive uncertainty and the Gaussian assumption 🆕

| Cohort | 95% coverage | Mean predictive mass below zero | Predicted means < 0 |
|---|---:|---:|---:|
| unfiltered | 0.941 | **0.130** | 0 / 2,707 |
| A4 | 0.940 | **0.124** | 0 / 2,654 |
| D1 | 0.938 | **0.120** | 0 / 2,219 |
| E2 | 0.945 | **0.122** | 0 / 1,942 |

Coverage is close to nominal, slightly narrow. But **12–13% of each predictive
distribution lies below zero**, where no ratio can be. No point prediction is
negative, so this is a tail problem, not a location problem — but it means the
interval endpoints and any probability statement derived from the lower tail
are not trustworthy, and P(PIGD) in particular absorbs impossible mass.

This is a genuine misspecification: the ratio is non-negative, right-skewed,
and exactly zero at ~7% of visits. A transformed or censored likelihood is the
natural next step and was explicitly out of scope for Step 3.

---

## 8. What can and cannot be claimed

**Supported by the audited evidence**

1. A flexible GP kernel predicts the next ratio better than a linear-trend GP —
   consistently, in four cohorts, under both aggregations. The effect is small:
   ~0.03 ratio units against a within-patient SD of 1.22.
2. The GP beats persistence and the patient's own mean in every cohort.
3. K0 reproduces the hierarchical Bayesian growth model; they are
   indistinguishable on held-out prediction.
4. A fitted linear mean is strongly preferred to a zero mean in sample.
5. The fits are numerically healthy: 11,444 fits, 100% convergence.
6. Conclusions are insensitive to the extreme-patient criterion.

**Not supported**

1. That Matérn beats RBF.
2. That the GP is better than the HMM — **different tasks, not comparable**.
3. That the GP is useful clinically. Nothing here was validated against
   clinical outcomes; the target is a noisy label produced by this project's own
   classification rule.
4. That a fitted trend describes any individual patient's disease course.
5. That the zero-mean comparison settles held-out performance — it is in-sample
   only.

---

## 9. Outstanding — ⚠️ updated 2026-09-30

Three items below were closed after this report was first written. See
`NEXT_RESEARCH_STEPS.md` for the full results; the corrections to §4.3 and §4.1
of this document are recorded here.

### Correction to §4.3 (mean function)

The +473 in-sample log-likelihood gap was presented here as strong evidence for
the fitted mean. The held-out test (821 folds, both arms fitted in the same
fold) shows it **does not transfer to held-out MAE**: −0.0071 [−0.0193,
+0.0056], including zero. The fitted mean is nonetheless better on **MSE**
(+0.0969 [+0.0634, +0.1307]) and **CRPS** (+0.0188 [+0.0130, +0.0250]), both
excluding zero. Keep the fitted mean — for calibration and large-error control,
not because of the likelihood gap.

### Correction to §4.1 (`var_linear`)

This report said the linear kernel term is redundant *given* a fitted linear
mean. That is wrong: in the zero-mean arm `var_linear` pins to its bound in
**821 of 821 folds as well**. The data rejects a patient-specific random slope
with or without a mean function; the zero-mean arm absorbs the population trend
by inflating `var_const` 0.882 → 4.671 and lengthening the RBF 3.71 → 4.10 y.

### Correction to the multi-step framing

"The GP's advantage widens with horizon" was read off between-bin means, whose
bins contain different patients. Within patients, holding composition fixed, the
widening is **not detectable within three years** and excludes zero at 5y+ in
three of four cohorts only.

| Item | Status |
|---|---|
| Held-out zero-mean LOPO | **closed** — see above; unfiltered cohort only |
| Multi-step horizon analysis | **closed** — four cohorts, with composition controls |
| Subscore baseline experiment | **closed** — recovers 654 points and 113 patients, but the GP loses its edge over persistence on the subscores; derived subtype is indistinguishable from the HMM (+0.0027 [−0.0042, +0.0098]) |
| Transformed / censored likelihood | 12–13% impossible predictive mass |
| Undefined-ratio visits (692/4,295, 634 of them TD) | Excluded non-randomly; the evaluation under-represents tremor-dominant visits |
| Multi-step forecast analysis | `trajectory_predictions_*.csv` and `trajectory_metrics.csv` exist but were not analysed in depth here |
