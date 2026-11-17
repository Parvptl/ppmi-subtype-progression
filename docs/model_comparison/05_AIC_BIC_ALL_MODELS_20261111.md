# AIC and BIC for every model family in the project

**Date** 2026-11-11 · **Run** `experiments/20261110T222940Z_aic_bic_all_models/`
**Script** `scripts/92_aic_bic_all_models.py` · **Models unmodified**
**Panel** `outputs/visit_panel.csv`, md5 `fb038ddbdc19ab3368e13940f398e7ba`
**Commit at run time** `e2c0dfa`

`docs/model_comparison/01_AIC_BIC_AND_HELDOUT_20261024.md` put three families on
the criteria: the rich clinical profile, the observed-label chains and the
CT-HMM. This run reproduces those exactly and adds every family that came
later or had never been scored: the **ratio-banded CT-HMM at all five band
widths**, three further latent specifications of the label CT-HMM and a
two-label alphabet, the
ratio-trajectory GPs, the bivariate subscore models (joint LMM, time-varying
spline LMMs, multi-output GPs) and the latent clinical states.

Every model was fitted **once on the full cohort**, purely to obtain a
likelihood at its own maximum, which is what AIC and BIC need. Nothing was
modified and no held-out number was recomputed; the held-out figures quoted in
§9 come from the existing runs.

`experiments/` is not version-controlled, so the machine-readable table, the
run report and the code hashes are archived in
`docs/model_comparison/run_manifests/20261110T222940Z_aic_bic_all_models/`.
Every fit is deterministic given the panel and seed 42: the latent-state family
was run twice and reproduced its log-likelihoods and its across-start spreads
exactly.

| | |
|---|---|
| Models scored | 53 |
| Comparable groups | 12 (five of them one per band width) |
| Families with no AIC/BIC | 5, listed with reasons in §10 |

---

## 1. The rule this table obeys

AIC and BIC are only meaningful between models whose likelihood is **the same
object over the same data**. Three things break that, and all three occur in
this project:

1. **Different observations.** 4,295 categorical labels, 3,361 transition
   pairs, 38,486 clinical-domain values, 8,590 subscores, 3,528 ratios.
2. **Mass versus density.** The label models' likelihood is a probability mass
   (always negative log). The trajectory and profile models' is a density,
   which has units and can be positive — every log-likelihood in group S is
   **positive**, which is why those AICs are negative.
3. **Different support, same rows.** The banded CT-HMM scores a
   (subtype, band) pair where the square CT-HMM scores a subtype. Same visits,
   different random variable.

So the table is grouped, and **a number may only be compared with another
number inside its own group**. Eleven groups are internally valid:

| Group | n terms | Likelihood | Models |
|---|---|---|---|
| **L1** | 4,295 | joint PMF over whole label sequences | CT-HMM (2, 3 and 5 latent states), chains + initial state |
| **L2** | 3,361 | PMF conditional on the previous observed label | CTMC, discrete Markov |
| **L3** | 3,812 | joint PMF over TD/PIGD-only sequences | CT-HMM on a 2-label alphabet |
| **B**, one per width | 4,295 | joint PMF over (subtype × band) sequences | banded CT-HMM vs its own nested restriction |
| **P** | 38,486 | density over nine clinical domains | three rich-profile variants |
| **R** | 3,528 | marginal density over the ratio | four GP kernels |
| **S** | 8,590 | density over both subscores | joint LMM, spline LMMs, multi-output GPs |
| **C** | 42,773 | density over ten standardised features | latent clinical states, by state count |

Groups P and C are close enough to be confused and are not comparable: the
rich profile models nine clinical domains, the latent states those nine plus
LEDD, on a different number of terms.

A caution that applies to every group: **AIC and BIC assume independent
observations**, and visits are clustered within patients. The effective sample
size is below the term count, so both penalties are optimistic and BIC's is, if
anything, too weak here. Rankings this large are unlikely to reverse, but no
margin below should be read as exact.

---

## 2. The three headline models, side by side

Not comparable with each other — three different likelihoods — but each is
decisive inside its own group.

| Model | log-likelihood | k | AIC | BIC | verdict in its group |
|---|---|---|---|---|---|
| **CT-HMM** (3 latent states, floored) | −2,969.06 | 14 | 5,966.12 | 6,055.23 | beats every observed-label chain by ΔAIC 224+; **two and five** latent states both rejected |
| **Rich profile** (level + rate + medication) | −45,812.34 | 45 | **91,714.68** | **92,099.79** | **best of the three variants on both criteria** |
| **Banded CT-HMM** (width 0.10) | −13,123.40 | 137 | 26,520.80 | 27,392.84 | **AIC keeps the bands** (Δ434), **BIC drops them** (Δ88) |

The banded model is the only one of the three where AIC and BIC disagree, and
§4 shows the disagreement is a function of the band width.

---

## 3. Group L1 — the latent layer, and how much of it

Joint PMF over 4,295 labels. The chains are put on this scale by giving them
an initial-state distribution estimated from the 934 first visits
(log-likelihood −719.34, charged at 2 parameters), exactly as in the October
analysis.

| Model | log-likelihood | k | AIC | BIC | ΔAIC | ΔBIC |
|---|---|---|---|---|---|---|
| CT-HMM, 3 latent states, **no** emission floor | **−2,955.32** | 14 | **5,938.64** | **6,027.75** | 0 | 0 |
| CT-HMM, 5 latent states, no emission floor | −2,940.97 | 34 | 5,949.95 | 6,166.36 | +11.3 | +138.6 |
| **CT-HMM, 3 latent states, floor 0.5** (published) | −2,969.06 | 14 | 5,966.12 | 6,055.23 | +27.5 | +27.5 |
| CT-HMM, 2 latent states, 3 labels, no floor | −3,010.70 | 7 | 6,035.39 | 6,079.95 | +96.8 | +52.2 |
| Markov (discrete) + initial state | −3,087.12 | 8 | 6,190.23 | 6,241.15 | +251.6 | +213.4 |
| CTMC (interval-aware) + initial state | −3,099.84 | 8 | 6,215.68 | 6,266.60 | +277.0 | +238.9 |

Four readings, in order of importance:

1. **The latent layer is strongly justified.** The published CT-HMM beats the
   best observed-label chain by ΔAIC 224.1 and ΔBIC 185.9 on a properly
   matched likelihood — six extra parameters buying 118 log-likelihood points.
   This is the formal counterpart of the history-dependence
   already documented for this cohort (χ² = 156.6, df = 4, p = 7.9 × 10⁻³³).
2. **Five latent states are rejected, by both criteria.** The 5-hidden
   sensitivity model of `scripts/10_loo_validation_5hidden_shard.py` buys 14.4
   log-likelihood points for 20 extra parameters: ΔAIC +11.3 and ΔBIC +138.6
   against the unconstrained three-state model. Three latent states is the
   right size, and this is now the **only** clean verdict on that question:
   the model's LOPO run is flagged unverified (scipy reported convergence in
   404/934 folds), so no held-out comparison can carry it. The full-cohort
   likelihood is unaffected by that: it is one fit from the audited
   multi-start scheme, and it matches the LOPO folds' training
   log-likelihood (−2,940.99 here against −2,940.99 on 933 patients).
3. **The emission floor costs 27.5 AIC points, and is worth paying.** Without
   `P(label a | state a) ≥ 0.5` the maximum-likelihood fit is degenerate — the
   latent Indeterminate state emits almost uniformly and PIGD becomes
   absorbing at the parameter boundary (see the note in
   `src/ppmi_subtype/hmm.py`). The floored fit is a *constrained* maximum, so
   27.5 points is the measured price of an interpretable model, not a fit
   failure. State it plainly rather than hide it: AIC prefers the degenerate
   model, which is a known limitation of a criterion that scores fit alone and
   not a reason to adopt it.
4. **Two latent states are also rejected, and by a wider margin than five.**
   Collapsing to two latent states while still emitting all three labels loses
   55.4 log-likelihood points for seven saved parameters: ΔAIC +96.8 and
   ΔBIC +52.2. With the five-state result above, the criteria bracket the
   latent dimension from both sides — **three is preferred to both two and
   five, on AIC and on BIC** — which is the first evidence in the project that
   the latent size is identified rather than merely assumed from the three
   clinical labels. `hmm.fit_hmm` refuses `n_hidden < n_observed`, so this fit
   runs through `hmm_stability.fit_multistart` on the same objective, bounds
   and options, from the project's four audited starts plus eight dispersed
   ones (final spread 0.00). The emission floor is undefined when the counts
   differ, so the fit is unconstrained and its two latent states are
   interchangeable; label switching does not move the likelihood, so the
   criteria are unaffected.

### Group L2, for completeness

The chains on their own conditional likelihood (3,361 pairs, no initial-state
term, so **not** on the scale above):

| Model | log-likelihood | k | AIC | BIC | ΔAIC |
|---|---|---|---|---|---|
| **Markov (discrete)** | **−2,367.77** | 6 | **4,747.55** | **4,784.27** | 0 |
| CTMC (interval-aware) | −2,380.50 | 6 | 4,773.00 | 4,809.72 | +25.4 |

The discrete chain beats the interval-aware CTMC — modelling elapsed time does
not pay for itself in a chain over the observed label. (Minor caveat: the
discrete chain's likelihood is read at its Laplace-smoothed estimate rather
than the exact MLE, so it is very slightly understated. It wins anyway.)

### Group L3 — a two-label alphabet

Dropping Indeterminate entirely gives a **different dataset**, not another
model of the same one: 441 Indeterminate visits go, 58 patients fall below two
labelled visits, 934 → 876 patients and 4,295 → 3,812 observations, and the
gaps now run between consecutive TD/PIGD visits, skipping over any
Indeterminate visit in between. **These rows are comparable only with each
other**, never with group L1.

| Model | log-likelihood | k | AIC | BIC | ΔAIC | ΔBIC |
|---|---|---|---|---|---|---|
| **CT-HMM, 3 latent states, 2 labels, no floor** | **−1,600.22** | 11 | **3,222.45** | 3,291.15 | 0 | +22.4 |
| **CT-HMM, 2 latent states, 2 labels, floor 0.5** | −1,613.74 | 5 | 3,237.49 | **3,268.72** | +15.0 | 0 |
| CT-HMM, 2 latent states, 2 labels, no floor | −1,613.74 | 5 | 3,237.49 | 3,268.72 | +15.0 | 0 |

1. **The two-state fits are identical with and without the floor**, to every
   reported digit. On a two-label alphabet the unconstrained maximum already
   satisfies `P(label a | state a) ≥ 0.5`, so the constraint is not binding and
   costs nothing — unlike the three-label case, where it costs 27.5 AIC points
   (§3). The floored two-state model is the one to quote, since it is the
   published specification and loses nothing by being it.
2. **AIC and BIC split on whether a third latent state helps.** Six extra
   parameters buy 13.5 log-likelihood points: ΔAIC 15.0 for the three-state
   chain, ΔBIC 22.4 against it. The three-state fit's start spread is 649.3
   log-likelihood points across twelve starts, so its likelihood is a lower
   bound and the AIC margin is the softer of the two numbers. On a split like
   this, prefer the two-state model.
3. **This is not evidence for collapsing the alphabet.** The comparison says
   nothing about whether two labels are better than three — that question
   cannot be asked with AIC, because the two alphabets are different
   observations. It only says that *given* a two-label alphabet, two latent
   states suffice.

---

## 4. Group B — the ratio-banded CT-HMM

### 4.1 How the banded model was made scorable at all

The banded model's likelihood is a mass over (subtype, band) pairs, the square
CT-HMM's a mass over subtypes alone, so **their AICs are not on one scale** and
the banded AIC recorded per fold in the band experiments could not be compared
with the baseline's. The fix is the same device §3 uses for the chains.

The banded emission factorises as `E[a,(s,b)] = M[a,s] · C[a,s,b]`. Restrict it
to `C[a,s,b] = c[s,b]` — *the band distribution does not depend on the latent
state* — and the joint log-likelihood separates exactly:

    loglik(restricted) = loglik(square CT-HMM) + Σ_t log c[subtype_t, band_t]

because a band factor that is constant across latent states multiplies straight
out of the forward recursion. Its maximum is therefore the square CT-HMM's fit
plus the empirical within-subtype band frequencies, available in closed form.

That restriction is **the null the band experiment was really testing**: the
ratio band is recorded, but it says nothing about where the patient is. It is
nested inside the banded model, over the same 4,295 observations, on one
likelihood — so AIC and BIC between the two are valid. The restricted solution
is also fed to the banded optimiser as an extra start, so the larger model
cannot lose to its own submodel through a bad optimum (it was the winning start
at width 0.30, and the nesting inequality is asserted at every width).

### 4.2 Every width

All rows have n = 4,295. Within a width the two rows are comparable; **across
widths they are not**, because a different band partition is a different
observation.

| Main width | categories | state-**dependent** bands |  | state-**independent** bands |  | ΔAIC | ΔBIC |
|---|---|---|---|---|---|---|---|
| | | loglik | k | loglik | k | | |
| 0.05 | 73 | −13,759.18 | 224 | −14,076.54 | 84 | **−354.7** (dependent) | **+536.4** (independent) |
| 0.10 | 44 | −13,123.40 | 137 | −13,422.24 | 55 | **−433.7** (dependent) | **+88.3** (independent) |
| 0.20 | 29 | −12,133.56 | 92 | −12,398.29 | 40 | **−425.5** (dependent) | **−94.5** (dependent) |
| 0.30 | 24 | −11,119.56 | 77 | −11,368.01 | 35 | **−412.9** (dependent) | **−145.5** (dependent) |
| 0.50 | 20 | −10,405.36 | 65 | −10,626.73 | 31 | **−374.7** (dependent) | **−158.3** (dependent) |

*(A negative Δ means the state-dependent model wins; positive means the
restriction wins.)*

### 4.3 What this says

1. **The ratio band genuinely carries information about the latent state.**
   Every width gains 221 to 317 log-likelihood points over the
   no-information restriction. AIC prefers state-dependent bands at **every**
   width, by 375 to 434 points. This is not a null result.
2. **BIC turns over between 0.10 and 0.20.** At the fine resolutions the
   parameter cost is not repaid — 536 BIC points against at width 0.05, 88 at
   width 0.10 (the published arm). From width 0.20 onwards the information is
   worth its parameters on BIC too, by 95, 146 and 158 points.
3. **So the published 0.10 arm is the one width where BIC says "do not".**
   That is the arm whose held-out accuracy fell by 0.53 points. The criteria
   and the held-out result agree about the direction of the problem: 44
   categories over 4,295 observations is too fine a grid for this cohort.
4. **Coarse bands are the better trade on every instrument.** Going from width
   0.10 to 0.30 raises the held-out AUC (0.8758 → 0.8784, the highest of any
   arm) and the accuracy (0.8316 → 0.8359), while the parameter count falls
   from 137 to 77 and the BIC margin swings 234 points in favour of using the
   band. `table103_band_width_complexity.csv` says why: at width 0.10, 12 of
   the 44 categories hold fewer than 30 observations and 10 come from fewer
   than 20 patients, against 2 and 2 of 24 at width 0.30 — 31 observations per
   parameter versus 56.

---

## 5. Group P — the rich clinical profile

Density over nine continuous domains, 38,486 observed values, on the
standardised scale. The three variants are nested and share one likelihood.

| Model | log-likelihood | k | AIC | BIC | ΔAIC | ΔBIC |
|---|---|---|---|---|---|---|
| **level + rate + medication** | **−45,812.34** | 45 | **91,714.68** | **92,099.79** | 0 | 0 |
| level only (random walk) | −45,940.77 | 27 | 91,935.53 | 92,166.60 | +220.9 | +66.8 |
| level + rate, no medication | −45,993.33 | 36 | 92,058.67 | 92,366.76 | +344.0 | +267.0 |

**The full specification wins on both criteria**, reproducing the October run
to within 0.05 AIC points (EM stopping tolerance). The rate component costs 18
parameters and buys 128 log-likelihood points, surviving BIC's stiffer penalty
by 66.8; the medication input costs 9 parameters and buys 181 points, the most
clearly justified single component in the project.

---

## 6. Group R — Gaussian processes for the tremor/PIGD ratio

Marginal density over 3,528 finite ratios from 821 patients. The two
mean-function coefficients are profiled out by the fitter and are counted here.

| Model | components | log-likelihood | k | AIC | BIC | ΔAIC | ΔBIC |
|---|---|---|---|---|---|---|---|
| **GP K1** | const + RBF | −6,245.21 | 6 | **12,502.41** | **12,539.42** | 0 | 0 |
| GP K3 | const + linear + Matérn-3/2 | −6,245.12 | 7 | 12,504.24 | 12,547.42 | +1.8 | +8.0 |
| GP K2 | const + linear + RBF | −6,245.21 | 7 | 12,504.41 | 12,547.59 | +2.0 | +8.2 |
| GP K0 | const + linear | −6,320.30 | 5 | 12,650.60 | 12,681.44 | +148.2 | +142.0 |

1. **A flexible temporal term is worth 148 AIC points**; a linear trend on top
   of it is worth nothing. K2 is K1 plus a linear component, and the two fitted
   log-likelihoods differ by 0.001 — numerically indistinguishable, and in the
   wrong direction for a nested pair, so it is optimiser noise rather than a
   gain. The linear term therefore costs a parameter and buys nothing, and the
   parsimonious K1 is preferred on both criteria.
2. **K0 is the maximum-likelihood counterpart of the hierarchical Bayesian
   growth model** — an equivalence `scripts/11` checks numerically to 1e-7 on
   real prediction points. So this row is also the criterion-bearing version of
   the growth model, and it says the growth model's straight-line mean plus
   random intercept and slope is 148 AIC points short of a GP that can bend.

---

## 7. Group S — the two subscores jointly

Density over 8,590 values (4,295 visits × 2 outcomes). **Every fit here is
maximum likelihood, never REML**: a REML log-likelihood cannot be compared
across mean structures and the spline models differ in exactly that, so the
stored REML values in `outputs/time_varying/model_summary.csv` were refitted by
ML for this table. Log-likelihoods are positive (a density), so AICs are
negative and **more negative is better**.

| Model | log-likelihood | k | AIC | BIC | ΔAIC | ΔBIC |
|---|---|---|---|---|---|---|
| **Multi-output GP K3/full** | **5,979.31** | 17 | **−11,924.62** | −11,804.63 | 0 | +12.8 |
| Multi-output GP K2/full | 5,977.29 | 17 | −11,920.57 | −11,800.58 | +4.1 | +16.9 |
| **Multi-output GP K3/diag** | 5,967.61 | 13 | −11,909.22 | **−11,817.46** | +15.4 | 0 |
| Multi-output GP K2/diag | 5,965.73 | 13 | −11,905.46 | −11,813.70 | +19.2 | +3.8 |
| Spline LMM, knots (2.5, 6) + random slopes | 5,852.56 | 21 | −11,663.12 | −11,514.90 | +261.5 | +302.6 |
| Spline LMM, knots (2, 5) + random slopes | 5,850.54 | 21 | −11,659.07 | −11,510.85 | +265.6 | +306.6 |
| Spline LMM, knots (1.5, 4) + random slopes | 5,837.24 | 21 | −11,632.49 | −11,484.26 | +292.1 | +333.2 |
| Spline LMM, knot (3) + random slopes | 5,821.66 | 19 | −11,605.33 | −11,471.22 | +319.3 | +346.3 |
| Spline LMM, no knots + random slopes | 5,737.68 | 17 | −11,441.35 | −11,321.36 | +483.3 | +496.1 |
| Multi-output GP K0/full | 5,726.35 | 13 | −11,426.69 | −11,334.93 | +497.9 | +482.5 |
| Multi-output GP K0/diag | 5,717.38 | 10 | −11,414.77 | −11,344.18 | +509.9 | +473.3 |
| Multi-output GP K1/full | 5,643.00 | 14 | −11,258.00 | −11,159.19 | +666.6 | +658.3 |
| Multi-output GP K1/diag | 5,634.77 | 11 | −11,247.54 | −11,169.90 | +677.1 | +647.6 |
| Spline LMM, knots (2, 5) | 4,896.04 | 14 | −9,764.08 | −9,665.26 | +2,160.5 | +2,152.2 |
| Spline LMM, knots (2.5, 6) | 4,895.36 | 14 | −9,762.72 | −9,663.90 | +2,161.9 | +2,153.6 |
| Spline LMM, knot (3) | 4,893.30 | 12 | −9,762.59 | −9,677.89 | +2,162.0 | +2,139.6 |
| Spline LMM, knots (1.5, 4) | 4,894.74 | 14 | −9,761.48 | −9,662.67 | +2,163.1 | +2,154.8 |
| Spline LMM, no knots | 4,875.99 | 10 | −9,731.98 | −9,661.40 | +2,192.6 | +2,156.1 |
| **Joint LMM, random intercepts** | 4,875.99 | 10 | −9,731.98 | −9,661.40 | +2,192.6 | +2,156.1 |

Four readings:

1. **Patient-specific slopes dominate everything else.** Adding random slopes
   to any mean structure is worth about 950 log-likelihood points and ~1,700
   AIC points. Nothing else in this group comes close. Heterogeneity in *rate*
   is the dominant feature of these trajectories.
2. **Where the knots go does not matter.** The four spline mean structures sit
   within 2.6 AIC points of each other, and within 2.0 of the knot-free
   straight line once random slopes are in. This is the criteria agreeing with
   what the temporal audit (`scripts/25_temporal_audit.py`) already concluded: "after
   about five years" is not an identified change point. The knot-free fit
   reproduces the joint LMM to 4 × 10⁻¹¹ in −2 log L, which is the arithmetic
   check that both code paths maximise the same likelihood.
3. **A GP with a flexible temporal term beats every LMM**, by ≥ 261 AIC points.
   K2/K3 (const + linear + RBF or Matérn) are far ahead of K0 (const + linear)
   and of K1 (const + RBF) — in the bivariate problem, unlike the univariate
   ratio in §6, *both* a trend and a flexible term earn their keep.
4. **Free cross-output correlation is borderline.** `full` beats `diag` on AIC
   by 15.4 but loses on BIC by 12.8. The three extra off-diagonal parameters
   are at the edge of justifiable — consistent with the small fitted
   correlations already reported for this cohort.

---

## 8. Group C — the latent clinical states

Gaussian density over 42,773 standardised feature values — the rich profile's
nine domains plus LEDD — with missing entries marginalised rather than
imputed. Eight EM starts per fit, as the development sweep of `scripts/61`
used.

| States | log-likelihood | k | AIC | BIC | ΔAIC | start spread |
|---|---|---|---|---|---|---|
| **8** | **−40,764.26** | 223 | **81,974.53** | **83,906.52** | 0 | 250.1 |
| 6 | −41,701.08 | 155 | 83,712.16 | 85,055.03 | +1,737.6 | 1,648.6 |
| 5 | −42,401.92 | 124 | 85,051.84 | 86,126.13 | +3,077.3 | 6,057.8 |
| 4 | −44,880.37 | 95 | 89,950.75 | 90,773.80 | +7,976.2 | 4,767.5 |
| 3 | −50,509.14 | 68 | 101,154.28 | 101,743.41 | +19,179.8 | 0.1 |
| 2 | −52,939.46 | 43 | 105,964.91 | 106,337.45 | +23,990.4 | 0.2 |

**The criteria do not identify a state count here, and should not be used to
pick one.** Both AIC and BIC fall monotonically to the top of the tested range
and select 8 — the boundary — which is evidence about the range, not about the
data. This reproduces the finding of
`docs/latent_clinical_states/01_LATENT_CLINICAL_STATES_20261020.md` §5, where
the held-out predictive log-density also never turns over and the
pre-registered one-standard-error rule chose 6.

The "start spread" column is why these rows carry an extra warning of their
own: at 4, 5 and 6 states the gap between the best and worst of eight EM starts
is 1,600 to 6,000 nats, so each log-likelihood is **a lower bound on its
model's maximum, not the maximum**. With only three starts the sequence was not
even monotone in the state count (6 states landed 500 nats *below* 5 states),
which is why eight were used. Criteria computed from a badly multimodal EM fit
inherit that uncertainty, and this family was closed as negative on held-out
grounds in any case.

---

## 9. The criteria against the held-out evidence

The two instruments answer different questions, and reading them together is
the point of the table.

| Component | In-sample (AIC / BIC) | Held-out |
|---|---|---|
| CT-HMM latent layer vs a chain | ΔAIC 224, ΔBIC 186 — decisive | +0.0073 accuracy (0.8371 vs 0.8298) |
| 2 or 5 latent states instead of 3 | ΔAIC +97 / +11, ΔBIC +52 / +139 — both rejected | no usable figure for either |
| Rich-profile rate component | ΔAIC 221, ΔBIC 67 — clear | level-only is nominally *more* accurate |
| Ratio bands, state-dependent | ΔAIC 375–434 at every width | AUC **+0.0197** [0.0120, 0.0285] at width 0.30 — real; accuracy −0.0053 [−0.0134, +0.0034] — nil |
| Ratio bands at width 0.10 | ΔBIC +88 against | accuracy −0.53 points, calibration worse |

The banded row is the informative one. Three independent instruments now say
the same thing about the ratio band:

* it **does** carry information — 221 to 317 log-likelihood points, and a
  paired AUC gain of +0.0197 whose interval excludes zero;
* it **does not** improve the thresholded TD-vs-PIGD call — every paired
  accuracy interval includes zero;
* and at the published 0.10 resolution the **parameter cost is not repaid** —
  BIC against by 88 points, held-out accuracy down.

So the published conclusion ("the extra ratio resolution did not improve
next-visit prediction, and the ten-fold increase in parameters is not justified
by the result") stands, and is now better supported: it was a statement about
*accuracy at one band width*, and the criteria show it is specifically the
fine-grained alphabet that fails, while the information itself is real and
shows up in the ranking metric.

The general pattern across the whole table: components that improve the
**probability model** — the likelihood, and the calibration that follows from
it — barely move **classification accuracy**, because the subtype label is so
persistent that a persistence rule already reaches 0.8338 and there is almost
no headroom. Accuracy on a saturated, thresholded, two-class endpoint is the
wrong instrument for model selection here; log-density, Brier, AUC and the
criteria are the right ones.

---

## 10. Models with no AIC or BIC, and why

| Model family | Scripts | Why not |
|---|---|---|
| Hierarchical Bayesian growth model | 04, 05 | Gibbs sampling, no maximised likelihood. Its ML counterpart is exactly **GP K0** (§6) — equivalence checked numerically in script 11 — so that row is the criterion-bearing version. DIC or WAIC would be the right criteria for the sampler itself. |
| Persistence, patient-mean and last-observation baselines | throughout | Not probability models. The persistence rule emits a degenerate distribution, which is why its held-out log loss is finite only through clipping (4.1959). |
| Temporal audits, stability analyses | 23–25 | Diagnostics, not models. |

---

## 11. Limitations

1. **Clustered observations.** AIC and BIC assume independence; visits are
   nested in patients. Both penalties are optimistic in every group. The
   margins are not exact, and the closest calls — multi-output `full` vs
   `diag` (15.4 AIC / 12.8 BIC), GP K1 vs K3 (1.8 AIC), band width 0.10's
   ΔBIC of 88 — should be read as "borderline", not as decided.
2. **One fit per model.** Unlike the held-out figures these carry no
   cross-validated uncertainty.
3. **Local optima.** The latent-clinical-state log-likelihoods are lower bounds
   (§8). Every other family was fitted from the audited multi-start scheme its
   own experiment used, and the banded fits additionally carry the nested warm
   start, with the nesting inequality asserted.
4. **Densities are scale-dependent.** The rich-profile and latent-state
   log-likelihoods are densities on the standardised scale; rescaling would
   shift them by a constant. That constant is identical within each group, so
   the within-group comparisons are unaffected and the cross-group ones were
   never valid anyway.
5. **No external cohort.** All of this is within PPMI.

---

## 12. What to change, and what not to

1. **Keep the published CT-HMM**: three latent states, emission floor 0.5. The
   latent layer is decisively justified; collapsing to two latent states and
   extending to five are both rejected by both criteria; the floor's 27.5 AIC
   points buy an interpretable model.
2. **Keep the full rich-profile specification** (level + rate + medication),
   preferred on AIC, BIC, Brier and log loss.
3. **Report the ratio band as information that is real but not worth its
   parameters at 0.10**, and report width 0.20–0.30 as the resolution at which
   both criteria and the held-out AUC agree it does pay. This is a sharper
   statement than the current manuscript text and does not contradict it.
4. **Do not report AIC or BIC across groups**, and in particular do not let
   the rich profile's AIC of ~91,700 be read against the CT-HMM's ~5,966. One
   is a density over 38,486 continuous values; the other a mass over 4,295
   labels.
5. **Do not use AIC or BIC to choose a latent clinical state count.** Both
   pick the boundary of the tested range; the held-out predictive log-density
   and the one-standard-error rule remain the basis for that choice.
6. **Prefer continuous scores over accuracy for any future model selection
   here.** Accuracy could not separate these models; log-density, Brier, AUC
   and the criteria could.
