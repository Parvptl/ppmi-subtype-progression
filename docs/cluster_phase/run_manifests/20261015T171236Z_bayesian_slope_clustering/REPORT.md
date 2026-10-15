# Bayesian patient-specific slope clustering — Part A

Run: `experiments/20261015T171236Z_bayesian_slope_clustering/` · script `scripts/34_bayesian_slope_clustering.py` · module `src/ppmi_subtype/slope_clustering.py` · git HEAD `1cb3ef1` (new files uncommitted) · panel MD5 `fb038ddbdc19ab3368e13940f398e7ba` (frozen, verified) · seed 42 · host `master` (login node, `nice 15`; compute nodes were all allocated) · runtime 1,283 s.

Every number below is read from this run's files (`results.json` and the CSVs). Exploratory analysis only.

**Bottom line.** The posterior-mean slopes do **not** support stable progression-rate groups. BIC on the posterior means prefers K = 3, but a model with one continuous Gaussian slope population produces K ≥ 2 in 200/200 simulated cohorts. The three components share almost the same centre (−0.096, −0.089, −0.075 /yr) and differ mainly in spread. Which component a patient lands in tracks how many visits they have. In every one of 500 joint posterior draws, K = 1 is selected. Silverman's test does not reject a single mode (p = 0.24). The distribution of patient slopes looks like one continuous, somewhat heavy-tailed population combined with uncertainty that varies strongly between patients.

---

## 1. Research question

Represent each patient by the slope of their TD/PIGD ratio from the existing hierarchical Bayesian growth model. Do those slopes fall into distinct progression-rate groups? Or are they better described as one continuous distribution, with the apparent structure coming from estimation uncertainty? Part A uses the 1-D slope only. Intercept + slope clustering, RobustLPA, HMM clustering and new nonlinear models are out of scope.

This experiment is distinct from the earlier clusterability audit (scripts 20/26 and the Stage 6/6b work). That audit used separate tremor and PIGD score slopes from a joint LMM. This one uses the single ratio slope from `hbayes`. Neither the audit nor its outputs were touched.

## 2. The Bayesian model and the total slope

The model is `hbayes.fit_growth_model`, with its specification unchanged:

R_it = (β0 + b0i) + (β1 + b1i)·t_it + ε_it,  b0i ~ N(0, τ0²), b1i ~ N(0, τ1²) (independent), ε ~ N(0, σ²).

It is a conjugate Gibbs sampler with priors N(0, 100²) on β and inverse-gamma(0.01, 0.01) on the variances. Time t is the panel's `years_from_baseline`, unchanged.

- **Settings:** function defaults (4,000 retained draws, 1,000 burn-in, thin 2), `keep_random_effects=True`, seed 42.
- **Total slope:** for each retained draw m, s_i^(m) = β1^(m) + b1i^(m), taken from the *same* sweep. Each row of `slope_draws.npz` is therefore one joint posterior draw of all 821 slopes. The script asserts this identity, and `tests/test_slope_clustering.py` checks it, including that b1i alone is *not* used.
- **Difference from script 04:** script 04 fits a 70 % training split because it evaluates held-out prediction. Here the model is fitted to all eligible patients, because nothing is evaluated on held-out data.
- **Global posterior** (mean [95 % CrI]):

| parameter | value |
|---|---|
| β0 | 2.326 [2.218, 2.436] |
| β1 (population slope, ratio /yr) | −0.0902 [−0.1094, −0.0714] |
| τ1 (SD of patient slopes, /yr) | 0.1016 [0.0805, 0.1257] |
| τ0 | 1.361 [1.281, 1.449] |
| σ (total residual SD) | 1.146 [1.114, 1.180] |

**Sampling diagnostics** (`sampling_diagnostics.csv`). The sampler runs a single chain and has no built-in diagnostics. Three extra chains (seeds 43–45) were run purely to compute split-R̂ and ESS; the primary analysis uses only the seed-42 chain.

- R̂ ≤ 1.0014 for all global parameters, and the maximum over the 821 patient slopes is 1.0007. No patient slope has R̂ > 1.01.
- ESS for the seed-42 chain alone: τ1 202 (the lowest), β1 1,619, and patient slopes at least 928 (median 3,819).

## 3. Cohort and eligibility

Eligibility is `hbayes.build_growth_data(panel)` unchanged: a visit needs a finite ratio, and a patient needs at least 2 such visits. No ratios were imputed.

| quantity | count |
|---|---|
| labelled visits / patients | 4,441 / 1,080 |
| labelled visits with undefined ratio (PIGD = 0) | 725 (665 TD, 60 Indeterminate) |
| finite-ratio visits / patients | 3,716 / 1,009 |
| **eligible patients / observations** | **821 / 3,528** |
| excluded: exactly 1 finite ratio | 188 |
| excluded: 0 finite ratios | 71 |

Eligible patients by number of finite-ratio visits:

| visits | patients |
|---|---|
| exactly 2 | 191 |
| 3 | 233 |
| 4–7 | 287 |
| ≥ 8 | 110 |

**The exclusions are not random** (`exclusions.csv`). Of the 71 patients with no finite ratio, 95.8 % are TD at their first labelled visit, against 69.8 % of included patients. The ratio is undefined exactly when PIGD = 0, which mostly happens in tremor-dominant visits with no gait or balance signs. The analysed cohort therefore under-represents the most purely tremor-dominant patients.

## 4. Clustering procedure and selection rule (fixed before any clustering)

- **Feature:** posterior-mean total slope per patient, in ratio units/yr, unscaled.
- **Model:** `sklearn` GaussianMixture with K = 1…5, 20 initialisations, seed 42, tol 1e-6, max_iter 1,000.
- **Selection:** lowest BIC among *eligible* fits. A fit is eligible if it converged and every component holds at least max(10, 2 % of n) patients by maximum-probability assignment; the floor is 17 for n = 821. K = 1 is always eligible.
- **Evidence:** ΔBIC = BIC(K=1) − best BIC(K ≥ 2). Positive values favour a mixture.
- **Unimodality test:** because a 1-D mixture cannot separate "groups" from "one skewed or heavy-tailed population", Silverman's critical-bandwidth bootstrap test of a single mode was added (500 bootstraps).
- **Labels:** no clinical labels are attached to any component.

## 5. Results on posterior-mean slopes

Distribution of the 821 posterior means:

- mean −0.0902, SD 0.0384, skew 0.64, excess kurtosis 7.5;
- the SD of the means is far below τ1 = 0.102, because the means are heavily shrunk towards β1;
- median posterior SD per patient is 0.100, so a typical patient's uncertainty is about 2.6 times the spread between patients' means;
- 29 patients (3.5 %) have a 95 % CrI that excludes 0: 28 negative, 1 positive.

Mixture comparison (`gmm_model_comparison.csv`):

| K | BIC | ΔBIC vs K=1 | converged | sizes | means (/yr) | SDs | eligible |
|---|---|---|---|---|---|---|---|
| 1 | −3011.1 | 0 | yes | 821 | −0.0902 | 0.0384 | yes |
| 2 | −3297.5 | 286.4 | yes | 657 / 164 | −0.0929 / −0.0840 | 0.0169 / 0.0644 | yes |
| **3** | **−3327.5** | **316.5** | yes | **260 / 505 / 56** | −0.0961 / −0.0897 / −0.0815 | 0.0064 / 0.0276 / 0.0841 | **selected** |
| 4 | −3308.1 | 297.0 | yes | 45 / 262 / 498 / 16 | — | — | no (16 < 17) |
| 5 | −3314.2 | 303.1 | yes | 37 / 272 / 499 / 12 / 1 | — | — | no |

The selected K = 3 is a **scale mixture, not a location mixture**: the three centres lie within 0.015 /yr of one another, while the SDs differ 13-fold (Fig. 2). Component membership is a proxy for data quantity:

| component | n | median visits | median follow-up (yr) | median posterior SD |
|---|---|---|---|---|
| narrow | 260 | 3 | 2.0 | 0.101 |
| middle | 505 | 4 | 3.2 | 0.099 |
| wide | 56 | 8 | 9.2 | 0.080 |

Patients with little data are pulled almost onto β1 and form the narrow spike. Well-observed patients keep their spread and form the wide tails.

`association_with_data_quantity.csv` confirms this:

- number of visits vs |mean − β1|: Spearman ρ = 0.53;
- follow-up vs |mean − β1|: ρ = 0.56;
- K = 2 label vs visit-count bucket: χ² p = 1.7e-26;
- SD of posterior means by visit bucket: 0.015 (2 visits), 0.021 (3), 0.042 (4–7), 0.071 (≥ 8).

## 6. Posterior uncertainty and stability

This section separates uncertainty about each patient's slope from uncertainty that would warrant a mixture.

- **Per-draw selection** (`posterior_draw_sensitivity.csv`). The same rule was applied to 500 seeded joint posterior draws, i.e. to the slopes themselves rather than their means.
  - K = 1 was selected in **500/500** draws; without the size floor, K = 1 is chosen in 499/500.
  - ΔBIC per draw: median −18.4, 5–95 % range [−20.2, −11.8].
  - 22 draws had at least one non-converged fit (K ≥ 2). The K = 2 fit converged in 99.8 % of draws.
  - Once each patient's full posterior uncertainty is carried, a single Gaussian is preferred every time. The K ≥ 2 preference exists only for the shrunken means.
- **Co-clustering, fixed K = 2** (`coclustering.npz`, Fig. 5).
  - Mean co-assignment is 0.598 for pairs in the same primary component and 0.578 for pairs in different components, so there is essentially no block structure.
  - ARI between each draw's K = 2 partition and the primary partition: median 0.012 (5–95 %: −0.008, 0.065).
  - All 821 patients co-cluster with their own primary component less than 70 % of the time.
- **Silverman test on the posterior means:** critical bandwidth 0.0264 (0.69 × SD), bootstrap p = 0.24 (118/500). A single mode is not rejected.

## 7. Follow-up (minimum-visit) sensitivity

`sensitivity_analysis.csv` reports two variants: restricting the primary posterior means, and refitting the model on the restricted cohort.

| min visits | variant | n | selected K | ΔBIC | ARI vs primary, K=2 | ARI, selected partitions |
|---|---|---|---|---|---|---|
| 3 | restricted means | 630 | 3 | 170.4 | 0.54 | 0.61 |
| 3 | refit | 630 | 3 | 174.8 | 0.57 | 0.58 |
| 5 | restricted means | 253 | 2 | 23.7 | 0.24 | 0.46 |
| 5 | refit | 253 | 2 | 26.9 | 0.25 | 0.40 |

Requiring ≥ 5 visits removes most of the narrow, heavily shrunk patients. The mixture evidence then falls by an order of magnitude (ΔBIC 316 → 24–27), and agreement with the primary partition is low (ARI 0.24–0.46). The K = 2 split at ≥ 5 visits is again two near-identical centres (−0.098 / −0.079) with different spreads. The grouping does **not** survive the sensitivity analysis as a stable partition.

## 8. Continuous no-subgroup nulls

`null_simulations.csv` / `null_summary.csv`, Fig. 6. Every null contains **no subgroups**. The statistic is ΔBIC (BIC(K=1) − min BIC(K ≥ 2), no size floor). p = (1 + #null ≥ observed)/(n + 1).

| null | reps | selected K ≥ 2 (rule) | null ΔBIC median [95th pct] | p(ΔBIC) | p(Silverman h_crit) |
|---|---|---|---|---|---|
| Gaussian marginal (mean, SD of observed means) | 500 | 0 % | −18.8 [−13.2] | 0.002 | 0.002 |
| Johnson SU marginal (skewed, heavy-tailed, unimodal; fitted to the means) | 500 | 99.4 % | 516 [895] | 0.96 | 0.94 |
| **Hierarchical re-estimation**: simulate ratios from the fitted model with *Gaussian* slopes at the real visit times, refit `hbayes` (4,000 draws), cluster the posterior means | 200 | **100 %** (181 K=2, 19 K=3) | 227 [292] | 0.010 | 0.14 |

How to read the three nulls:

- **Gaussian marginal.** It rejects, but it is the wrong reference. Shrinkage that varies with visit count does not produce Gaussian posterior means even when the true slopes are Gaussian.
- **Hierarchical null.** This is the only null that reproduces the shrinkage and the uneven uncertainty, and it recovers τ1 well (median 0.103 against 0.102 used). Under it, the pre-specified rule selects K ≥ 2 in **every** replicate. **So "BIC selects K > 1" is the expected outcome even when there are no groups.** The observed ΔBIC (316) lies in the upper tail (null maximum 330, p = 0.010). That points to heavier-than-Gaussian tails in the true slope distribution, but not to multiple modes: Silverman's statistic is unremarkable against this null (p = 0.14).
- **Johnson SU null.** It reproduces the observed ΔBIC and bandwidth easily (p = 0.96 / 0.94). A single, skewed, heavy-tailed population is therefore fully sufficient to explain the mixture evidence.

**Hierarchical-null limitation.** It plugs in posterior-mean hyperparameters instead of drawing them from the posterior. It keeps the Gaussian random-slope assumption, so it tests "Gaussian slopes" rather than "any unimodal slopes". The 0 / 1 finite-ratio exclusions are taken as given, with no model for why visits are missing.

## 9. Limitations and unresolved questions

- The model assumes a **linear** long-term trend per patient. A slope summarises average change, not visit-to-visit change. Similar slopes do not imply similar trajectories, and nonlinear or reversing patients are poorly represented by a single slope.
- The ratio is **undefined when PIGD = 0** (725 labelled visits). The resulting exclusions are non-random and enriched for tremor-dominant patients (§3).
- **Patient-level uncertainty is large:** the median posterior SD (0.100) is about the same as τ1. For patients with 2–3 visits the posterior mean is mostly β1. Clustering point estimates turns differences in data quantity into apparent components.
- **GMM components can stand in for** skewness, heavy tails or variance heterogeneity. They do not correspond automatically to subtypes, and the components here differ mainly in spread. Membership is exploratory and is not a diagnosis.
- **Sampling:** a single primary chain. Extra chains were used only for diagnostics. ESS for τ1 in the primary chain is about 200.
- **Distributional assumption:** the random-slope distribution is Gaussian inside the model. The evidence above (§8) suggests the true distribution has heavier tails, which this model cannot represent.
- **Validity:** nothing here establishes clinical validity or predicts outcomes.
- **Unresolved:** whether a heavy-tailed random-slope model (e.g. Student-t b1i) fits better, and whether the slope tail is associated with clinical covariates. That is a continuous-heterogeneity question, not a clustering one.

## 10. Recommendation

**Further trajectory clustering on the 1-D Bayesian ratio slope is not justified by this evidence.**

- The apparent K = 3 comes from shrinkage and visit count.
- It vanishes once posterior uncertainty is carried (K = 1 in 500/500 draws).
- It is reproduced by continuous no-subgroup nulls (100 % of hierarchical-null replicates select K ≥ 2).
- It does not show multimodality (Silverman p = 0.24).
- It is unstable under the minimum-visit sensitivity analysis.

The one robust signal is that the patient-slope distribution is **heavier-tailed than Gaussian** (hierarchical null p = 0.010). If this line is pursued, the next step should model that continuous heterogeneity directly: for example, a Student-t or Gaussian-scale-mixture random slope in the same growth model, compared by held-out predictive performance. Partitioning patients would not be the right step. Part B (intercept + slope) is not supported as a route to discrete groups by this result. If it is run anyway, it needs the same hierarchical null and per-draw checks, applied before interpreting any BIC preference.

---

### Files in this run

| file | contents |
|---|---|
| `patient_slopes.csv` | 821 patients: PATNO, posterior mean / median / SD / 2.5 % / 97.5 % of β1 + b1i, P(slope < 0), number of finite-ratio observations, first and last time, follow-up, visit bucket |
| `slope_draws.npz` + `slope_draws_patient_order.csv` | (4000 × 821) float32 total-slope draws; column k ↔ PATNO in the order file; β0, β1, τ0², τ1², σ² draws |
| `gmm_model_comparison.csv` | mixture fits for K = 1–5 (§5) |
| `cluster_assignments.csv` | per-patient component labels and probabilities |
| `posterior_draw_sensitivity.csv` | per-draw selection results (§6) |
| `coclustering.npz` | co-clustering matrices (§6) |
| `sensitivity_analysis.csv` | minimum-visit sensitivity (§7) |
| `association_with_data_quantity.csv` | component membership vs visits and follow-up |
| `null_simulations.csv`, `null_summary.csv` | null replicates and summary (§8) |
| `sampling_diagnostics.csv` | split-R̂ and ESS |
| `posterior_global.csv` | global posterior summary |
| `exclusions.csv` | excluded vs included patients |
| `results.json` | config, counts, versions |
| `run.log` | execution log |
| `figures/` | fig1–fig6 (`.png` + `.pdf`) |

Execution log: no convergence failures in the primary fits and no numerical warnings. Non-converged fits occurred only inside replicates (22/500 posterior draws, 22 Gaussian-null, 3 Johnson SU and 5 hierarchical replicates), are recorded per row, and count as ineligible under the rule.

An earlier full run (`experiments/_smoke/superseded_20261015T165023Z_s34/`) was superseded. Its `exclusions.csv` read the numeric `subtype` codes as names; every other table is byte-identical to this run.
