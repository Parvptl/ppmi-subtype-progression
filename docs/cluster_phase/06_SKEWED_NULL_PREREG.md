# Stage 6b — skewed / heavy-tailed no-groups null: pre-registration

Written **before** any clustering metric (ΔBIC, ARI) was computed on the new
null. The run records this file's SHA-256 in its `run_config.json`; any later
edit is visible.

## Why

Stage 6 found real ΔBIC = 513 far above both no-groups nulls (q95 ≈ 117), but
drop-one-visit ARI (0.75) not above the measured null (q95 0.77), so the
pre-specified clustering rule was not met. Both nulls draw the patient random
effects from N(0, Σb). A shape diagnostic (real data vs four measured-null
replicates; shape summaries only, no clustering metrics) shows the real BLUP
features are far more skewed and heavy-tailed than any Gaussian null produces:

| BLUP feature | real skew / excess kurtosis | measured null (4 reps) |
|---|---|---|
| PIGD slope | 2.17 / 6.90 | 0.41–0.58 / −0.06–0.38 |
| tremor slope | 0.72 / 2.25 | 0.17–0.43 / 0.73–1.46 |
| PIGD intercept skew | 1.72 | 0.41–0.51 |

The question: **can a single-mode continuous population with this skew and
tail weight reproduce the observed separation (ΔBIC) and stability (ARI)?**

## The null (primary: `null_skewt`)

Identical to `null_measured` (script 26) in everything except the random-effect
distribution: same fitted fixed effects, same Σb (REML, random-slope LMM; REML
variance components are consistent without normality), Gaussian residuals with
the fitted Σe, each patient's real visit times, then clipping to [0, 4] and
rounding to the item grid (1/11 tremor, 1/5 PIGD).

Random effects: `b = chol(Σb) w`, `w_j` independent standardized Azzalini
skew-t (skew δ_j per component, shared ν). Cov(b) = Σb exactly; the density has
a single mode (`src/ppmi_subtype/nongaussian.py`, tests in
`tests/test_nongaussian.py`).

**Calibration** (indirect inference, common random numbers): choose
(δ_intT, δ_intP, δ_slopeT, δ_slopeP, ν) to match the real data's skewness and
excess kurtosis of the four BLUP components (8 targets; weights 1 for skewness,
1/4 for kurtosis), averaged over 24 simulated datasets per evaluation,
Nelder–Mead on (atanh δ, log(ν − 2)). **ΔBIC and ARI are never used in
calibration.**

**Fidelity gate** (must pass before the full run): achieved mean BLUP skewness
within ±0.30 of every real target, slope excess kurtosis within ±1.5. If the
family cannot reach these, the run stops and the family is reported inadequate.

Sensitivity null `null_tails`: same ν, all δ = 0 (heavy tails, no skew).

## Metrics and decision rule (fixed now)

R = 200 replicates of each null; the real data are evaluated with the same
function and seed as script 26 (its ΔBIC/ARI must reproduce exactly — a check).

For the real value x and null replicates y_1..y_R of each statistic:
* one-sided Monte Carlo p = (1 + #{y_r ≥ x}) / (R + 1);
* q95 with a 95% bootstrap interval over replicates (Monte Carlo uncertainty);
* exceedance share with a Clopper–Pearson 95% interval.

Against **null_skewt**:

* **A** — ΔBIC_real > q95 (p < 0.05)
* **B** — drop-one-visit ARI(K = 2)_real > q95 **and** ≥ 0.6

Interpretation, fixed in advance:

| A | B | Conclusion |
|---|---|---|
| no | no | A single-mode skewed, heavy-tailed continuum reproduces the separation signal. **Consistent with continuous heterogeneity; no evidence for discrete groups.** Not proof that the population is continuous. |
| yes | no | Separation exceeds this continuum but assignments are not reproducible. **Inconclusive.** |
| yes | yes | Structure beyond this continuum, reproducible. Groups possible — still not established phenotypes (requires external validation). |
| no | yes | Stability without excess separation. **Inconclusive.** |

If the real value lies inside the bootstrap interval of q95, the criterion is
reported as **borderline** rather than met / not met.

## Known limitations (stated before the run)

1. A mixture of groups also produces skew and kurtosis; calibrating to marginal
   moments can absorb part of a genuine group signal. A "no" on A means the
   data *cannot distinguish* a continuum from groups by these statistics, not
   that groups are absent.
2. Only marginal shape is matched, not joint (nonlinear) dependence.
3. Visit schedules are copied from the real patients, so the null keeps the
   observed missingness pattern but not its dependence on the outcome
   (informative dropout at higher PIGD, Stage 5).
4. Residuals remain Gaussian before clipping/rounding.
5. R = 200 bounds the smallest attainable p at 1/201.

## Amendment 1 (2026-10-14) — one tail parameter per component

Made after a feasibility probe and **before any clustering metric was computed
on the new null**. The probe (`experiments/_stage6b_design/reach_probe_output.txt`,
SHA-256 `e3407161…`; shape summaries only, 6 replicates per setting, all
components given the same δ and ν) showed:

* the largest PIGD-slope BLUP skewness reachable was ≈ 1.88 (δ = 0.99, ν = 3)
  against a real 2.17 — inside the ±0.30 gate only at the edge;
* with one shared ν, the heavy tails needed for the PIGD slope push the
  tremor-slope kurtosis to ≈ 5.9 against a real 2.25 (gate ±1.5).

A shared ν therefore cannot satisfy the gate by construction. The amendment
gives each latent component its own tail parameter: 8 calibration parameters
(δ_j, ν_j for j = int_tremor, int_pigd, slope_tremor, slope_pigd), each
ν_j = 2.5 + exp(·) (finite variance). Every other element above — targets,
weights, fidelity gate, R, metrics, decision rule — is unchanged. The
`null_tails` sensitivity null uses the calibrated ν_j with all δ = 0.

Also recorded from the probe: every setting produced 20–23% PIGD = 0 visits
against a real 16%. The random-effect shape does not control this; it comes
from Gaussian residuals plus clipping at 0 (limitation 4). It is not gated, and
will be reported beside the result.
