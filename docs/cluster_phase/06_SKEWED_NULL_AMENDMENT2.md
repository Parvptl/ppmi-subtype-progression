# Stage 6b — Amendment 2 (PROPOSED, 2026-10-14): item-level measurement model

**Status: proposed, not in force.** Nothing below has been run. It is written
before any clustering metric was computed on any non-Gaussian null and before
any further calibration. It amends `06_SKEWED_NULL_PREREG.md` (SHA-256
`1b01bb6a…`, including Amendment 1), which is left unchanged.

## 1. Why Amendment 1's design cannot be used

Calibration 2 failed the fidelity gate: PIGD-slope BLUP skewness 1.73 vs real
2.17 with the skew-t family at its limit (δ = 0.946, ν = 2.92). It also
produced 21.7% PIGD = 0 visits against 16.1%. Two diagnostics followed. Neither
computed a clustering metric.

**PIGD audit** (`experiments/20261014T143819Z_pigd_measurement_audit`, script 30):

* S = 5 × PIGD is an integer item sum on 0–20 (max observed 19). No sentinel
  (101) or out-of-range value enters the analysed visits; 1,143 cohort visits
  lack a PIGD score (1,078 no OFF/untreated Part III exam; 46 postural
  stability coded 101).
* Items differ strongly in how often they are positive: gait 75%, walking
  (patient-reported) 50%, postural stability 16%, freezing 14%, freezing of
  gait 6%. When S = 1, the positive item is gait in 79% of visits.
* Zeros decline smoothly with the current score — P(next visit zero) is 0.46,
  0.11, 0.035, 0.02, 0 for S = 0, 1, 2, 3–4, 5+ — and are not absorbing (57% of
  patients whose first positive score is followed by later visits return to
  zero). Within patients, P(zero) falls 3.0 points per year [2.5, 3.5].
  Always-zero patients (4%) have short follow-up (median 2.3 y) and differ
  only weakly from movers.
* Variance grows with level: conditional-residual SD rises 4.5-fold (0.097 →
  0.43) across fitted deciles; within-patient SD rises with patient mean
  (Spearman 0.71).

**Measurement-model predictive checks**
(`experiments/20261014T144511Z_pigd_measurement_models`, script 31; Gaussian
latent random effects; 20 datasets each):

| Feature | real | M1 Gaussian const. | M2 Gaussian hetero. | M5 beta-binomial |
|---|---|---|---|---|
| zero share | 0.161 | 0.233 | 0.238 | 0.240 |
| P(next 0 \| S = 1) | 0.109 | 0.248 | 0.187 | 0.260 |
| P(next 0 \| S = 0) | 0.459 | 0.467 | 0.573 | 0.371 |
| all-zero patients | 0.041 | 0.058 | 0.097 | 0.053 |
| within-SD ratio (top/bottom octile) | 6.99 | 6.53 | 20.78 | 7.04 |
| BLUP skew, PIGD slope | 2.17 | 0.50 | 0.74 | 1.40 |
| BLUP kurtosis, PIGD slope | 6.90 | 0.18 | 0.79 | 3.04 |
| BLUP skew, PIGD intercept | 1.72 | 0.54 | 0.67 | 1.38 |

Reading:

1. The beta-binomial (M5) explains much of the BLUP shape that the Gaussian
   models miss (PIGD-slope skew 0.50 → 1.40 with *Gaussian* latent effects) and
   matches the mean–variance ratio. The link function and level-dependent
   noise are therefore part of why the real BLUP slopes are skewed.
2. Every model over-predicts zeros by ≈ 7–8 points and predicts too many
   1 → 0 transitions. M5's fitted precision went to φ ≈ 7,550 (effectively
   binomial). With the mean held fixed, a sum of items with different
   difficulties has fewer zeros than a binomial (Π(1 − p_i) ≤ (1 − p̄)^n), and
   a beta-binomial cannot represent that under-dispersion. The audit shows
   exactly this heterogeneity. **The item-sum models are misspecified at the
   item level.**
3. Real 1 → 0 transitions are rarer than any model predicts, which may also
   reflect within-patient persistence that random intercept + slope cannot
   represent. Stage 3 found the same signal on the continuous scale: K2/K3
   (trend + stationary deviation) beat K0 held out.

No structural-zero class is indicated: zeros follow the same smooth gradient
as positive scores, so a two-part (hurdle) model is not adopted.

## 2. Proposed measurement model

**PIGD: graded-response model (GRM) on the five items**, sharing one latent
PIGD trait per visit:

    P(item_k >= c | θ_ij) = expit(a_k (θ_ij − τ_kc)),   c = 1..4
    θ_ij = β0 + β1 t_ij + b_i0 + b_i1 t_ij,   (b_i0, b_i1) ~ random effects

Item discriminations a_k and thresholds τ_kc are item-specific, which is what
the audit says is needed. The subscore is reconstructed exactly as the
pipeline computes it (mean of the five simulated items), so the null passes
through the same subscore → Gaussian LMM → BLUP → GMM chain as the real data.

**Tremor: unchanged** (Gaussian with clipping/rounding). Its features are
already reproduced (BLUP skew 0.80 vs 0.72), and its zero share is 7%.
Sensitivity: tremor as an 11-item GRM if time allows.

**Fitting, two-stage.** (i) Item parameters: cross-sectional GRM with
θ ~ N(0, 1) over all 4,295 analysed visits, marginal ML by 1-D Gauss–Hermite
quadrature. (ii) Longitudinal: β, the random-effect covariance and the
cross-outcome correlation with item parameters held fixed, by the Laplace
approximation (as in `bbglmm`). Assumption: item parameters do not change over
follow-up (no differential item functioning by time or treatment) — checked by
refitting stage (i) on year 0 and on year ≥ 3 visits separately.

**Latent random effects.** Primary: Gaussian (does correct measurement alone
reproduce the features?). If the gate below fails only on BLUP shape, the
Amendment 1 skew-t family (per-component δ, ν) is calibrated on top, on the
latent scale. Sensitivity: an Ornstein–Uhlenbeck within-patient deviation on
θ (stationary persistence) — motivated by point 3 above and by Stage 3.

## 3. Calibration targets, tolerances, go/no-go (fixed now)

Tolerances are 2.5 × the Monte Carlo SD of each feature across 20 simulated
datasets (M5 column of `simulated_features.csv`), rounded up. The original BLUP
tolerances are kept.

| Target | Tolerance | Mandatory |
|---|---|---|
| BLUP skewness, 4 components | ± 0.30 | yes |
| BLUP excess kurtosis, two slopes | ± 1.5 | yes |
| zero share, overall | ± 0.025 | yes |
| zero share, year 0 and year ≥ 5 | ± 0.04 | yes |
| P(next 0 \| S = 0) | ± 0.05 | yes |
| P(next 0 \| S = 1) | ± 0.035 | yes |
| all-zero patients | ± 0.02 | yes |
| within-SD ratio (top/bottom octile) | ± 1.5 | yes |
| item positive rates (5) | ± 0.03 each | yes |

**Go** to the null simulation only if every mandatory target is met. **No-go**
otherwise; the failing targets are reported and tolerances are not changed.
ΔBIC and ARI are never calibration targets. The null contains no planted
groups: latent random effects are Gaussian or single-mode skew-t
(`nongaussian.py`), and the GRM is a monotone map of a single latent trait.

The decision rule for the eventual null comparison is unchanged from the
original pre-registration (criteria A and B, Monte Carlo p, bootstrap interval
for q95, borderline category).

## 4. Expected cost

* Stage (i) cross-sectional GRM: minutes.
* Stage (ii) Laplace fit: the 6-parameter beta-binomial took 68 min with
  finite-difference gradients; the GRM version has the same 6 longitudinal
  parameters with a 5-item likelihood, so roughly 1–3 h unless analytic
  gradients are added (recommended first).
* Predictive check with Gaussian latent effects (20 datasets): ~10 min.
* If needed, latent skew-t calibration: ~45 min (as calibration 2).
* Full null (200 + 200 replicates), only after a pass: ~10 min.

## 5. Risks stated in advance

* If the within-patient persistence (point 3) is real, a GRM with random
  intercept + slope alone will still miss the 1 → 0 transitions; the OU
  sensitivity is the planned response, not a tolerance change.
* Two-stage fitting ignores uncertainty in the item parameters; acceptable for
  a simulation null, not for inference.
* Informative dropout remains unmodelled (visit schedules are copied).
