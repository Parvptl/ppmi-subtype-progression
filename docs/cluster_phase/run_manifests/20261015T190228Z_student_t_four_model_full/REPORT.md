# Four-model growth comparison: slope tails vs residual tails — full run

- **Run:** `experiments/20261015T190228Z_student_t_four_model_full/`
- **Script:** `scripts/36_student_t_residual_variants.py --stage full --jobs 16`
- **Host / runtime:** `master` (login node, `nice 15`; every compute node was allocated), 1,621.9 s; no warnings or errors in `run.log`
- **Pre-registration:** frozen rule in `docs/cluster_phase/08_STUDENT_T_RESIDUAL_SPEC.md` §7. SHA-256 hashes of the spec, `hbayes.py`, `hbayes_t.py`, `hbayes_tt.py` and the script are in `frozen_config.json`, written before any computation.
- **Integrity:** panel MD5 `fb038ddbdc19ab3368e13940f398e7ba` verified; git HEAD `7a59dba` (the Student-t work is uncommitted)
- **Earlier runs preserved:**
  - Smoke results: `20261015T182129Z_student_t_slope_smoke/`, `20261015T184027Z_student_t_residual_smoke/`, and `_smoke/superseded_20261015T183613Z_s36_6scen/`
  - Full-stage code-path check: `_smoke/s36full_codepath/`

Every number below is read from this run's output files.

## Verdicts (three separate questions, frozen rule)

| question | verdict |
|---|---|
| **Q1. Are visit-level residuals heavy-tailed?** | **SUPPORTED** |
| **Q2. Are patient slopes heavy-tailed beyond the residual tails?** | **INCONCLUSIVE (convergence)**: the mandatory R̂ / ESS criterion for νs failed under the Gamma prior |
| **Q3. Does any configuration predict better than the Gaussian baseline?** | GT, TT and (marginally) TG all meet the "better than GG" rule on script 04's class metric. Almost all of the gain comes from Student-t residuals. |

---

## 1. Design recap

- **Models:** four configurations of R_it = (β0 + b0i) + (β1 + b1i)t + ε_it, with b0i Gaussian:
  - GG: Gaussian slopes, Gaussian residuals (`hbayes`);
  - TG: Student-t slopes, Gaussian residuals (`hbayes_t`);
  - GT: Gaussian slopes, Student-t residuals;
  - TT: Student-t slopes, Student-t residuals.
- **Parameters:** the Student-t laws have degrees of freedom νs (slopes) and νe (residuals). τ1 and σ are scales.
- **Priors:** as in hbayes; ν ~ Gamma(2, 0.1), with a log-uniform sensitivity prior, on the log grid [1, 300].
- **Splits:** script 04's patient-level splits, seeds 42 (primary), 0, 1, 2, 3, 4, 5. Each has 570–579 training patients and 325 test patients.
- **Per-split fits:** one chain per configuration, 4,000 draws, burn-in 2,000, thin 5.
- **Scoring:** every split scores all four models on the **identical** one-step-ahead transitions (782–863 per split). The script asserts that every transition has a prediction from all four models.
- **Full cohort:** 821 patients / 3,528 observations, GT and TT with 4 chains × 2 priors (thin 10).

## 2. Q1: residual tails (SUPPORTED)

The primary evidence is held-out ΔLPD (log predictive density of the next ratio, nats per transition). The primary split uses 854 transitions and a patient-cluster bootstrap with 2,000 resamples.

| contrast | ΔLPD [95 % CI] | splits with ΔLPD > 0 | Δ class log-lik [95 % CI] |
|---|---|---|---|
| GT − GG | **+0.112 [0.061, 0.168]** | **7/7** (0.056 to 0.136) | +0.038 [0.020, 0.055] |
| TT − TG | **+0.106 [0.058, 0.158]** | **7/7** (0.056 to 0.130) | +0.038 [0.021, 0.055] |

All three conditions of the rule are met.

The residual-tail parameter is precisely estimated and converged:

- full-cohort GT νe = 2.10 [1.88, 2.39], with R̂ ≤ 1.0005 and ESS ≈ 4,500 under both priors;
- TT gives the same value;
- every per-split value lies between 1.96 and 2.23.

A νe near 2 means the visit-to-visit noise in the TD/PIGD ratio has very heavy tails. Its variance is barely finite. That is consistent with the ratio's construction: a quotient whose PIGD denominator can be small.

## 3. Q2: patient-slope tails beyond residual tails (INCONCLUSIVE, convergence)

| criterion (frozen) | result | met? |
|---|---|---|
| **(conv)** full-cohort TT νs: R̂ < 1.01 and ESS ≥ 400 under each prior | Gamma prior: **R̂ 1.018, ESS 248**. Log-uniform prior: R̂ 1.008, ESS 422. | **No** (Gamma prior) |
| (a) full-cohort TT νs 97.5 % < 10 under both priors | Gamma: 1.99 [1.45, 3.47]. Log-uniform: 1.93 [1.43, 3.01]. | yes |
| (b) per-split TT νs 97.5 % < 10 in ≥ 5/7 splits | **6/7**. The 97.5 % values are 5.25, 9.04, 4.42, 8.41, 3.01, 8.66 and 16.95 (split 5). | yes |
| (c) calibration false positives ≤ 1/20 in each of C1–C3 | 0/20, 0/20, 0/20 | yes, but with limited precision (below) |

Because (conv) is mandatory, **Q2 is inconclusive.** I have not relaxed or reinterpreted the criterion.

**Why convergence failed:**

- The four Gamma-prior chains agree on the centre: νs medians 1.96, 1.99, 2.01, 1.96.
- They disagree on the rare upper tail. Two chains (seeds 44 and 45) made brief excursions to νs ≈ 25 and ≈ 22; the other two never went above about 6.
- So the upper tail of νs is under-explored. The 97.5 % values per chain are 3.18, 3.14, 5.80 and 3.47.
- Per-split chains also mixed poorly: TT νs ESS was 45–152 per chain at thin 5.

**What the other criteria show.** They point the same way, and the calibration and positive control show that this design can detect such tails. Even so, they do not establish heavy-tailed patient heterogeneity while (conv) fails.

**Calibration** (full-cohort design, 821 patients, real visit times, Gaussian slopes, 20 replicates each, TT fitted with 4 chains × 2 priors). Truth was taken from the full-cohort GT fit: β1 = −0.080, τ1 = 0.105, σ = 0.589, ν̂e = 2.10.

| scenario | false positives | observed rate | Clopper–Pearson 95 % CI | TT νs median (median over replicates) | smallest νs 97.5 % (Gamma / log-uniform) |
|---|---|---|---|---|---|
| C1: i.i.d. t_2.10 residuals | 0/20 | 0 % | [0 %, 16.8 %] | 14.6 | 41.1 / 271 |
| C2: real residuals resampled as whole-patient blocks | 0/20 | 0 % | [0 %, 16.8 %] | 14.4 | 33.2 / 235 |
| C3: piecewise-linear trajectories, Gaussian slope change, t residuals | 0/20 | 0 % | [0 %, 16.8 %] | 15.3 | 29.6 / 213 |
| P: positive control, t₂ slopes + t_2.10 residuals (descriptive) | 19/20 flagged | 95 % | [75.1 %, 99.9 %] | 2.5 | 2.9 / 2.9 |

**Precision caveat.** 20 replicates per scenario give limited precision. An observed 0/20 is compatible with a true false-positive rate as high as 16.8 %, so these results do **not** show that the true rate is below 5 %. The calibration fits were not required to converge; the largest calibration R̂ for νs was 1.023 in C1–C3.

**Other things to keep in mind:**

- C2 used the patient's own residual vector for 1 visit count held by a single patient.
- C3 tests only one form of nonlinearity: a single kink with a Gaussian slope change.
- Moderate slope tails (νs ≈ 3–6) remain indistinguishable from Gaussian slopes on this design. That was shown in the smoke test and is unchanged here.

## 4. Q3: predictive value (script 04's class metric)

Primary-split Δ class log-likelihood per transition against GG, with patient bootstrap 95 % CI:

| model | Δ vs GG [95 % CI] | splits Δ > 0 | rule met | ΔLPD vs GG |
|---|---|---|---|---|
| TG | +0.0034 [0.0007, 0.0063] | 6/7 | yes (marginal) | +0.009 [−0.001, 0.022] |
| GT | +0.0382 [0.0205, 0.0549] | 7/7 | yes | +0.112 [0.061, 0.168] |
| TT | +0.0412 [0.0229, 0.0586] | 7/7 | yes | +0.115 [0.063, 0.172] |

Primary-split metrics (854 transitions):

| model | class log-lik / transition | accuracy | Brier | LPD |
|---|---|---|---|---|
| GG | −0.677 | 0.744 | 0.377 | −1.747 |
| TG | −0.673 | 0.744 | 0.375 | −1.738 |
| GT | −0.638 | 0.767 | 0.351 | −1.636 |
| TT | −0.635 | 0.766 | 0.349 | −1.632 |

The other six splits rank the models the same way (`heldout_metrics_by_split.csv`). Script 04's own Monte-Carlo scorer, run on the primary-split GG fit, gives −0.679 (accuracy 0.734, Brier 0.377), consistent with the shared scorer's −0.677.

The practical gain comes from **Student-t residuals**: about +0.04 nats per transition and about +2 points of accuracy. Adding Student-t slopes on top (TT − GT) gives +0.0030 [0.0002, 0.0061] on the class metric and +0.0033 [−0.0026, 0.0102] on LPD, positive in 4/7 and 6/7 splits. That is descriptive only: the smoke test showed this contrast cannot identify slope tails, and it gave positive LPD differences even when the slopes were Gaussian.

## 5. Supported conclusions, inconclusive findings, limitations

**Supported**

- Visit-to-visit noise in the TD/PIGD ratio is very heavy-tailed: νe ≈ 2.1, converged and stable across splits and priors.
- Modelling that noise with Student-t residuals improves one-step-ahead held-out prediction over the Gaussian baseline in all 7 splits, on both script 04's class metric and LPD.

**Inconclusive**

- Whether patient slopes are heavy-tailed beyond the noise.
  - The point estimate is νs ≈ 2 under both priors, in 6/7 splits, and calibration found no false positives in 60 Gaussian-slope replicates.
  - However, the mandatory convergence criterion failed under the primary prior (R̂ 1.018, ESS 248), because the upper tail of νs was explored only by occasional excursions.
  - Separately, the calibration bounds the false-positive rate only to ≤ 16.8 % per scenario at 95 % confidence.

**Limitations**

- Linear trajectories are assumed. The nonlinearity check (C3) covers only one simple form.
- The Student-t residuals are symmetric, but the real residuals are skewed. C2 partly addresses this by resampling real residuals.
- ν is restricted to a discrete grid.
- Per-split prediction uses one chain, as in script 04.
- The full cohort excludes patients without two finite ratios. This selection is non-random (Part A §3).
- Results apply to one-step-ahead prediction. Longer horizons were not evaluated.
- No clinical interpretation of individual patients' slopes is warranted.

## 6. Possible next step (not run; would need approval)

Longer TT chains, or a ν move that explores the tail better, would decide the convergence question. Options include thin 40, or 8 chains with an updated proposal for ν. The (conv) criterion and thresholds would stay unchanged. Increasing calibration replicates (for example to 60 per scenario) would tighten the false-positive bound: 0/60 has an upper bound of 6.0 %.

## Files

| file | contents |
|---|---|
| `frozen_config.json` | frozen settings and SHA-256 hashes |
| `results.json` | all verdicts and the inputs to each criterion |
| `full_cohort_diagnostics.csv`, `full_cohort_nu_traces.npz` | full-cohort R̂ / ESS and ν traces |
| `split_info.csv` | per-split sizes, νs / νe, ESS |
| `heldout_metrics_by_split.csv`, `heldout_paired_differences_by_split.csv`, `primary_split_bootstrap.csv` | held-out comparison |
| `heldout_predictions.csv.gz` | per-transition predictions; participant-level, not for commit |
| `calibration.csv` | 80 calibration / positive-control replicates |
| `run.log` | execution log |
