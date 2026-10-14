# Stage 6b — Amendment 2 predictive check: specification (fixed before the run)

Dated 2026-10-14. Companion to `06_SKEWED_NULL_AMENDMENT2.md` (PROPOSED,
SHA-256 `32f90ac4…`), which stays a proposal until this check is reviewed.
`06_SKEWED_NULL_PREREG.md` (SHA-256 `1b01bb6a…`, with Amendment 1) is
unchanged. The run records this file's SHA-256.

**Scope: a measurement-model predictive check only.** No clustering
calibration, no null simulation, no clustering metric (ΔBIC, ARI, silhouette,
GMM fits) is computed or inspected.

## 1. Input

The five PIGD items (NP2WALK, NP2FREZ, NP3GAIT, NP3FRZGT, NP3PSTBL) as
original ordinal responses 0–4, taken from the raw MDS-UPDRS Part II/III files
through the pipeline's own steps: `loaders` → `select_medication_state` (OFF /
untreated exam) → `deduplicate_visits` → `clean_items` (values outside 0–4,
including the 101 "unable to rate" sentinel, become missing). Joined one-to-one
to the analysable cohort on (PATNO, EVENT_ID).

Verified before the run: 4,295 visits, 934 patients; 0 missing items, 0
sentinels, 0 invalid values in the analysed visits; the mean of the five
cleaned items equals the stored `pigd_score` at every visit. Items are never
reconstructed from the sum. A missing item would contribute no term to that
visit's likelihood (supported and tested; not present in these data). Visits
without a PIGD score stay excluded, as in every earlier analysis.

## 2. Likelihood

Graded-response model (Samejima), item k = 1..5, category c = 0..4:

    P(Y_ijk >= c | θ_ij) = F_kc = expit(a_k (θ_ij − τ_kc)),  c = 1..4
    P(Y_ijk = c | θ_ij)  = F_kc − F_k,c+1      (F_k0 = 1, F_k5 = 0)

with a_k > 0 and τ_k1 < τ_k2 < τ_k3 < τ_k4. Parameterised as
a_k = exp(α_k), τ_k1 = γ_k1, τ_kc = τ_k,c−1 + exp(γ_kc), so every value is a
valid ordered model. 25 item parameters.

Longitudinal latent structure:

    θ_ij = β0 + β1 t_ij + b_i0 + b_i1 t_ij,   (b_i0, b_i1) ~ N(0, D)

t = years from baseline (as everywhere). D = L L′, L lower-triangular with log
diagonal. 5 longitudinal parameters.

## 3. Identifiability and two-stage fitting

**Stage 1 — item parameters.** θ ~ N(0, 1) for every analysed visit, visits
treated as independent; marginal ML over the 25 item parameters by 1-D
Gauss–Hermite quadrature (61 nodes), analytic gradient, L-BFGS-B. N(0, 1)
fixes the location and scale of θ; a_k > 0 fixes its direction. Assumption:
the item parameters do not depend on the shape of the pooled latent
distribution (a working model; stated limitation).

**Stage 2 — longitudinal part.** Item parameters held at their stage-1
values; β0, β1 and D estimated by the Laplace approximation to the marginal
likelihood over (b_i0, b_i1), with the exact gradient (envelope theorem for
the mode, third derivatives of the GRM log-likelihood for the log-determinant
term). The GRM log-likelihood is concave in θ, so the per-patient mode is
unique. β and D are on the stage-1 θ scale.

**Diagnostics (reported, not gated):** stage-1 refits on year-0 visits and on
visits at ≥ 3 years (item drift / DIF over follow-up); bound hits; optimiser
messages; the Laplace error against adaptive quadrature for a sample of
patients.

## 4. Simulation

R = 40 datasets. Each: b_i ~ N(0, D) for every patient; θ on the patient's
real visit times; each of the five items drawn from its GRM category
distribution; PIGD subscore = mean of the five simulated items (exactly the
pipeline's definition); tremor kept at its real values. Each dataset then goes
through the Stage 6 feature pipeline unchanged: bivariate Gaussian RS-LMM
(`timevarying.fit`, random slopes, no knots) and its BLUPs. Seeds:
`np.random.default_rng([42, 32, r])`.

## 5. Targets, tolerances and gate (fixed now)

The tolerances are those of Amendment 2: 2.5 × the Monte Carlo SD of each
feature across 20 datasets of the beta-binomial model (script 31,
`simulated_features.csv`, M5 column), rounded up. Item positive-rate
tolerances (±0.03) and the BLUP tolerances (from the original
pre-registration) were fixed in Amendment 2 as well. They are absolute and are
not recomputed from the new model's spread.

| # | Target | Tolerance | Gated |
|---|---|---|---|
| 1–4 | BLUP skewness: tremor intercept, PIGD intercept, tremor slope, PIGD slope | ± 0.30 | yes |
| 5–6 | BLUP excess kurtosis: tremor slope, PIGD slope | ± 1.5 | yes |
| 7 | PIGD = 0 share, all visits | ± 0.025 | yes |
| 8 | PIGD = 0 share, year 0 visits | ± 0.04 | yes |
| 9 | PIGD = 0 share, visits at ≥ 5 years | ± 0.04 | yes |
| 10 | P(next visit S = 0 \| S = 0) | ± 0.05 | yes |
| 11 | P(next visit S = 0 \| S = 1) | ± 0.035 | yes |
| 12 | share of patients with PIGD = 0 at every visit | ± 0.02 | yes |
| 13 | within-patient SD ratio, top / bottom octile of patient mean (patients ≥ 4 visits, octile means) | ± 1.5 | yes |
| 14–18 | positive rate (item ≥ 1) of each of the five items | ± 0.03 | yes |

Reported, not gated: full category distribution (0–4) of each item;
P(next 0 | S = 2) and | S ≥ 3; zero share at years 1–2 and 3–4; residual SD of
the RS-LMM by fitted-PIGD decile; BLUP intercept kurtosis.

**Monte Carlo spread.** For each target: simulated mean over the R = 40
datasets, SD across datasets, MC standard error = SD/√R, and the 5th–95th
centile. Decision per target: **pass if |simulated mean − real| ≤ tolerance.**
Because tolerances are fixed absolute numbers, a target with zero or near-zero
spread is not easier or harder to pass; it is flagged if its MC SE exceeds half
its tolerance (the estimate is too imprecise to decide — reported as
"undecided", which counts as a fail for the gate). A real value that is
undefined (e.g. no S = 1 visits) is a fail.

**Gate: GO only if all 18 gated targets pass.** Any fail → NO-GO, with the
failing targets diagnosed. Tolerances and targets are not changed after the
result. A NO-GO does not trigger automatic model extensions (latent skew,
OU persistence); those need a separate decision.
