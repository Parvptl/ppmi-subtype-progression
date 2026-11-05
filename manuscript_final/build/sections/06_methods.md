## 6. Methods

Every model below is described as the code implements it. Section 6.1 fixes the notation; Sections 6.2 to 6.15 give the models in the order the programme fitted them; the evaluation protocol common to all of them is in Section 8.

### 6.1 Notation

Patient *i* has labelled visits *k* = 0, …, *n*<sub>*i*</sub> − 1 at times *t*<sub>*ik*</sub> (years from baseline), with recorded labels *y*<sub>*ik*</sub> ∈ {TD, Indeterminate, PIGD} = {0, 1, 2}, tremor score *T*<sub>*ik*</sub>, PIGD score *P*<sub>*ik*</sub> and, where *P*<sub>*ik*</sub> > 0, ratio *r*<sub>*ik*</sub> = *T*<sub>*ik*</sub> / *P*<sub>*ik*</sub>. The interval to the next visit is Δ<sub>*ik*</sub> = *t*<sub>*i,k*+1</sub> − *t*<sub>*ik*</sub>. A prediction opportunity is a labelled visit *k* followed by a labelled visit *k* + 1 of the same patient; the model may use visits 0, …, *k* and the value of Δ<sub>*ik*</sub>, never the label at *k* + 1 or anything later. Across the 934 patients with at least two labelled visits there are 3,361 such opportunities. Throughout, "history" means the labels or measurements at visits 0, …, *k*, and the elapsed interval is an input: the model is told when the next examination happens, not what it records.

### 6.2 Chains in the recorded label

**Discrete-time Markov chain (DTMC).** The next label depends only on the current label, and the interval is ignored. The transition matrix is estimated from the consecutive pairs of the training patients with Laplace smoothing, one pseudo-count per cell:

```
P_ab = (n_ab + 1) / (n_a + 3),   a, b in {TD, Ind, PIGD}
```

and the predictive distribution at a prediction opportunity is the row of *P* for the current label.

**Continuous-time Markov chain (CTMC).** A generator matrix *Q* with off-diagonal rates *q*<sub>*ab*</sub> = exp(θ<sub>*ab*</sub>) ≥ 0, diagonal entries equal to minus the row sums, and transition probabilities over an interval Δ

```
P(Δ) = exp(Q Δ)
```

computed by eigendecomposition of the 3 × 3 generator. The panel data are interval-censored, so the likelihood of a training pair (*a*, *b*, Δ) is *P*(Δ)<sub>*ab*</sub> and the log-likelihood is the sum over pairs. The six log-rates are bounded to [−12, 4] (rates between about 6 × 10<sup>−6</sup> and 55 per year) and maximised by L-BFGS-B. The predictive distribution at a prediction opportunity is the row of *P*(Δ<sub>*ik*</sub>) for the current label. Mean sojourn times are −1/*Q*<sub>*aa*</sub>.

### 6.3 The continuous-time hidden Markov model (CT-HMM)

The latent state *X*(*t*) is a CTMC with generator *Q* as in Section 6.2 and initial distribution π. The recorded label at visit *k* is a noisy reading of the latent state through a misclassification (emission) matrix

```
E[a, b] = P(recorded label = b | latent state = a).
```

The likelihood of one patient's sequence marginalises over every latent path by the forward recursion

```
alpha_0      = pi ∘ E[:, y_0]
alpha_{k+1}  = (alpha_k P(Δ_k)) ∘ E[:, y_{k+1}]
log L_i      = log sum_a alpha_{n_i - 1}[a]
```

where ∘ is the element-wise product, so the cost is linear in the number of visits. Prediction uses filtering only: the predictive distribution of the next label is

```
p(y_{k+1} | y_{0..k}, Δ_k) ∝ alpha_k P(Δ_k) E
```

normalised over the three labels; no backward pass is run and nothing at or after visit *k* + 1 enters the prediction.

**Parameterisation.** π is the softmax of two free logits bounded to [−8, 8] (the third fixed at zero). Each row of *E* is a softmax of three logits with one entry pinned to zero. The six free emission logits are bounded above so that each diagonal entry of *E* is at least a floor *f*: with logits (0, *b*, *b*) the diagonal probability is 1 / (1 + (*S* − 1) e<sup>*b*</sup>), so the bound is *b* ≤ log((1/*f* − 1) / (*S* − 1)), which for *f* = 0.5 and *S* = 3 is log 0.5. The floor is not cosmetic: unconstrained, the maximum-likelihood solution is degenerate and identical across restarts, with the latent Indeterminate state emitting uniformly and PIGD absorbing with an exit rate at its lower bound. Requiring that each state be recorded correctly more often than not removes both degeneracies at no cost in held-out performance (Section 9.6). The model has 14 free parameters: 2 for π, 6 for *Q* and 6 for *E*. The negative log-likelihood is minimised by L-BFGS-B (maximum 3,000 iterations, function tolerance 10<sup>−11</sup>, gradient tolerance 10<sup>−9</sup>) with finite-difference gradients from four restarts seeded from 42 (the configured value; twenty to forty starts in the stability study); the Stage 4 study established that scipy's success flag alone under-reports convergence of this fit and that the projected-gradient norm is the right criterion, which the later experiments record for every fold.

**Alternative configurations.** The stability study (Section 9.6) fitted the same model with floors of 0.6 and 0.7, with no floor (C3_free), and with four and five latent states and three recorded labels (C4_free, C5_free; the floor is undefined when the number of latent states differs from the number of labels). An earlier five-state leave-one-patient-out run is also reported (Section 9.6); its fits converged in 404 of 934 folds.

### 6.4 The hierarchical Bayesian growth model and its Student-t variants

For the ratio-defined visits of patient *i* (PIGD score greater than zero),

```
r_ik = (beta_0 + b_0i) + (beta_1 + b_1i) t_ik + e_ik,
b_0i ~ N(0, tau_0^2),  b_1i ~ N(0, tau_1^2),  e_ik ~ N(0, sigma^2).
```

The proposal's observation and process noise terms are additive on the same observation and are identified only through their sum, so the sampler estimates σ<sup>2</sup> = *s*<sub>proc</sub><sup>2</sup> + *s*<sub>obs</sub><sup>2</sup> and says so. Priors are weakly informative: N(0, 100<sup>2</sup>) on β<sub>0</sub>, β<sub>1</sub> and inverse-gamma(0.01, 0.01) on the three variances. Inference is a single Gibbs chain with conjugate updates (the random effects of all patients drawn as a batch of bivariate normals; 2,000 draws after a burn-in of 500 for the main fit, 1,500 after 500 inside the longitudinal cross-validation); no R-hat or effective sample size was computed, so the posterior summaries are descriptive. A label probability is obtained from the predictive distribution of the ratio by integrating over the threshold intervals: P(TD) = P(*r* ≥ 1.15), P(PIGD) = P(*r* ≤ 0.90), P(Indeterminate) the remainder.

**Student-t variants.** Four variants cross Gaussian (G) and Student-t (T) distributions on the residual and on the patient slope: GG (the model above), GT (t residuals), TG (t slopes) and TT (both). The t distributions are scale mixtures of normals, with the residual degrees of freedom ν<sub>*e*</sub> and the slope degrees of freedom ν<sub>*s*</sub> on a 400-point log-spaced grid over [1, 300] under a Gamma(2, 0.1) prior; the slope degrees of freedom are updated by a collapsed slice step that integrates the mixing weights out (hbayes_tt_nu), which the original grid update could not mix. Each split fit draws 4,000 samples after a burn-in of 2,000 (thinning 5); the full-cohort fits thin by 10 and run four chains with R-hat and effective-sample-size thresholds of 1.01 and 400. The pre-registered questions were Q1 (do t residuals improve held-out prediction), Q2 (is the slope distribution heavy-tailed, judged by ν<sub>*s*</sub> having a 97.5% posterior quantile below 10 in at least five of seven patient splits, with calibration scenarios bounding the false-positive rate) and Q3 (do the variants improve the class-label metric over GG). Held-out log predictive density per visit and the class-label metric use 400 posterior draws with 20 inner draws for the patient-level random effects; intervals are patient bootstraps of 2,000 resamples.

### 6.5 The first-visit (new-patient) model

A patient at a first visit has no history. The first-visit model is a multinomial logistic regression (scikit-learn) of the label at a later visit on the first-visit label, the horizon in years, the first-visit ratio and ten baseline measures (age, sex, disease duration, the RBD screening questionnaire, SCOPA-AUT, UPSIT, MoCA, bilateral putamen DaT binding, Hoehn and Yahr stage and MDS-UPDRS Part II). Covariates are standardised and, where missing, mean-imputed inside the training split. Nested variants drop the covariates, the ratio, or both, and the importance of each covariate is the held-out cost of dropping it from the full model. Evaluation is on ten random 70/30 patient-stratified splits.

### 6.6 Gaussian-process regression of the ratio

For patient *i*, *r*<sub>*ik*</sub> = *f*<sub>*i*</sub>(*t*<sub>*ik*</sub>) + ε<sub>*ik*</sub> with ε ~ N(0, σ<sub>*n*</sub><sup>2</sup>) and *f*<sub>*i*</sub> ~ GP(*m*, *k*). The mean is linear, *m*(*t*) = β<sub>0</sub> + β<sub>1</sub>*t*, profiled out by generalised least squares inside the marginal likelihood. Four kernels were fitted:

```
K0  constant + linear                       (the growth model in function-space form)
K1  constant + RBF
K2  constant + linear + RBF
K3  constant + linear + Matern-3/2
```

K0 is the hierarchical growth model: its covariance τ<sub>0</sub><sup>2</sup> + τ<sub>1</sub><sup>2</sup>*t t′* + σ<sup>2</sup>[*t* = *t′*] is a constant kernel plus a linear kernel plus white noise, so the comparison of K2 with K0 is a nested test of whether a flexible term earns its place. Hyperparameters are shared across patients and estimated once per fold by maximising the summed marginal likelihood over the training patients (log-parameters bounded to [log 10<sup>−6</sup>, log 10<sup>6</sup>], several restarts); each held-out patient's posterior is the shared prior conditioned on that patient's own earlier visits through a Cholesky factor. Visit *k* is predicted from visits 0, …, *k* − 1 (one-step), and in the multi-step analysis every visit after the first ⌈*n*<sub>*i*</sub>/2⌉ visits is forecast from those. Four cohorts were run: the unfiltered 821 ratio-defined patients and three cohorts that remove extreme patients under rules A4, D1 and E2. A zero-mean K2 arm was fitted inside the same folds as the fitted-mean arm to test the mean function on held-out data.

### 6.7 Models of the two subscores

The ratio is undefined at 16.3% of labelled visits and compresses two nearly uncorrelated signals (overall Pearson correlation 0.047 between the subscores) into one volatile number. The following models therefore predict the pair (*T*, *P*) and derive the label afterwards.

**Independent subscore GPs.** Two K1 GPs on the raw tremor and PIGD scores, fitted exactly as in Section 6.6, on all 3,361 one-step points of the 934 patients.

**Joint linear mixed model (LMM).** For outcomes *d* ∈ {T, P},

```
T_ik = alpha_T + beta_T t_ik + b_Ti + e_Tik
P_ik = alpha_P + beta_P t_ik + b_Pi + e_Pik
(b_Ti, b_Pi) ~ N(0, Sigma_b),  (e_Tik, e_Pik) ~ N(0, Sigma_e)
```

with unstructured 2 × 2 random-intercept and residual covariances, fitted by REML through the closed-form inverse of the block covariance *V* = *J*<sub>*n*</sub> ⊗ Σ<sub>*b*</sub> + *I*<sub>*n*</sub> ⊗ Σ<sub>*e*</sub>. There are no random slopes: the median patient contributes four visits over 3.1 years, and a patient-specific linear term in the GP ladder was pinned at its lower bound in 821 of 821 folds.

**Multi-output GP (linear model of coregionalisation).** Stacking the two outputs per visit,

```
K[(j,d),(j',d')] = sum_c B_c[d,d'] k_c(t_j, t_j') + Sigma_e[d,d'] delta_jj'
```

with scalar temporal components *k*<sub>*c*</sub> from {constant, linear, RBF, Matérn-3/2} and 2 × 2 positive semi-definite coregionalisation matrices *B*<sub>*c*</sub>. The "full" and "diag" variants differ in exactly one thing, whether the off-diagonals of every *B*<sub>*c*</sub> and of Σ<sub>*e*</sub> are free or pinned to zero, so their difference isolates the cross-output dependence. Means are linear per output and profiled out; hyperparameters are shared across patients as in Section 6.6.

**Time-varying linear-spline LMM.** The joint LMM with the population mean replaced by a linear spline in time with knots at 2 and 5 years (fixed in advance, never estimated):

```
slope on [0, 2) = b_1,  on [2, 5) = b_1 + b_2,  on [5, inf) = b_1 + b_2 + b_3
```

so that "is the rate constant" is the linear contrast *b*<sub>2</sub> = *b*<sub>3</sub> = 0. Random-slope versions were fitted to show that the fixed-effect result survives letting patients differ. The Stage 5 temporal audit refitted the same model with other mean bases (quadratic, cubic, natural splines, single knots on a grid), under maximum likelihood for comparability, with patient-weighted bootstrap refits, a parametric-bootstrap supremum likelihood-ratio test of a single slope change (200 null cohorts under random-intercept and random-slope nulls), an attrition model, and completer-only slopes.

**Deriving the label.** A continuous model's predictive distribution for (*T*, *P*) is propagated through the unchanged classification rule by Monte Carlo: draws are clipped at zero, the degenerate cases (PIGD score zero) are handled exactly as the rule handles them, and the label frequencies over draws give a three-label probability vector (4,000 draws in the original runs, 10,000 with common random numbers in the harmonised comparison). The argmax of that vector is the predicted label; a "point rule" that classifies the point prediction is reported separately where the original run used it.

### 6.8 The posterior-predictive label check (D-1) and the Stage 1 diagnostics

The D-1 check asks whether the Student-t growth model on the ratio reproduces the recorded labels' dynamics. On the 821 patients and 3,528 ratio-defined visits, with five disjoint whole-patient folds stratified by first label: (A) the TT model is fitted on four folds (collapsed slice sampler, Gamma(2, 0.1) prior on ν, 4,000 draws, burn-in 2,000, thinning 5); (B) for each test patient the random effects and mixing weights are drawn given that patient's own ratios and each of 500 posterior draws; (C) fresh t noise at the patient's own visit times gives a replicate ratio sequence and, through the fixed thresholds, a replicate label sequence; (D) five pre-registered statistics are computed on the observed and the replicate sequences: S1 the Indeterminate share, S2 the reversion rate after an entry into Indeterminate (*R*<sub>*E*</sub>), S3 the reversion rate after a direct TD-PIGD change (*R*<sub>*D*</sub>), S4 their difference, S5 the observed-to-expected within-patient clustering of Indeterminate visits (a sixth, two-step persistence, is descriptive). The discrepancy for each statistic is the mid-*p* tail probability of the observed value under the replicates, converted to *z* = Φ<sup>−1</sup>(*p*) and capped at ±3.09 (the floor of 1/1001 on *p*), and the test statistic is *T* = max |*z*| over S1 to S5. The threshold *c* is the 95th percentile of *T* over 500 synthetic cohorts generated from a frozen full-cohort TT posterior; the pre-registered verdict is "reproduced" if *T*<sub>obs</sub> ≤ *c*. Four positive-control scenarios (100 cohorts each) with pre-stated detection rates calibrate the power. The Stage 1 diagnostics that followed are descriptive: they compare the observed Indeterminate share by PIGD item sum, the ratio distribution around the band, and the direction-specific reversion rates against the covariate-blind expectation under the same frozen generator, computed by deterministic quadrature, and attach no verdict.

### 6.9 Label-construction audits

**Part C.** The item sums were rebuilt from the raw item tables and checked against the frozen panel at every visit. The OFF and ON examinations of the same visit were labelled separately and their agreement measured, once with all sixteen items (the Part II items are shared by both examinations) and once with the Part III items only. The Indeterminate audit computed, with patient-bootstrap intervals (2,000 resamples), the reversion rate after an entry into Indeterminate and after a direct change, their difference, the shares of Indeterminate entries sustained at the next visit and at two later visits, the share of exits that return, and an observed-to-expected within-patient clustering ratio for each label, with a pre-registered reading rule (the entry-reversion contrast would support transient boundary-crossing only if the clustering ratio for Indeterminate also excluded one).

**Threshold sensitivity (O-4).** The frozen panel was relabelled with the perturbed rule TD if *r* > 1.13, PIGD if *r* < 0.91 (inner boundaries), writing a separate panel, and the relabelled visits were counted by class and by participant. The recount is deterministic, so no interval applies; agreement with PPMI's own derived labels was recomputed under both rules.

**Lag-1 residual autocorrelation (E-1).** For each of the 397 patients with at least four finite-ratio visits spanning at least one year, residuals from the patient's own ordinary-least-squares linear trend were formed, and the lag-1 autocorrelation of consecutive residuals was computed over 2,050 pairs. The null is independent noise with the same visit pattern (2,000 Monte Carlo draws), the observed statistic carries a 2,000-draw patient-clustered bootstrap, and the difference carries a combined resample.

### 6.10 The two-class next-visit framework

Every two-class model produces, at each prediction opportunity, the conditional probabilities

```
P_TD = p_TD / (p_TD + p_PIGD),   P_PIGD = p_PIGD / (p_TD + p_PIGD)
```

from its three-label distribution, so Indeterminate is never recoded in the inputs (it remains a third observed label in every history, which the latent models emit, the chains transition through, and the discriminative model reads as a feature) and only the target is restricted. A prediction opportunity whose actual next label is Indeterminate is a case in which the conditioning event did not occur; it is excluded from the two-class score, counted, and reported with what each model would have said. The predicted class is PIGD when *P*<sub>PIGD</sub> > 0.5 (no ties occurred). The raw Indeterminate mass is carried through and reported. Seven models were compared on identical cases: the CT-HMM (Section 6.3, unchanged), the CTMC and the DTMC (Section 6.2), a direct L2-penalised logistic regression (ridge penalty 1.0 on standardised coefficients, intercept excluded) on ten causal history features (indicators of the current label being TD or Indeterminate, whether the previous non-Indeterminate label was PIGD and whether none exists, the fractions of PIGD and Indeterminate labels in the history, the log run length of the current label, the log number of prior visits, the current time and the log interval), the same model without the interval feature, persistence of the last non-Indeterminate label, and the two-class marginal prior. All are fitted inside the leave-one-patient-out loop of Section 8.

### 6.11 The latent clinical-state CT-HMM

Ten clinical measurements (Section 5.3) are standardised on the training patients and read by a CT-HMM with *K* latent states, the same generator machinery as Section 6.3, and a diagonal multivariate Gaussian emission per state with mean μ<sub>*k*</sub> and variance vector σ<sub>*k*</sub><sup>2</sup>. With a diagonal covariance the emission log-density is a sum over features, so a missing feature is dropped from the sum: missing measurements are marginalised exactly, and a visit with no measurements contributes a flat likelihood. Parameters are fitted by expectation-maximisation (forward-backward smoothing on the training sequences, numerical maximisation for the generator, at most 150 iterations, several dispersed starts initialised from symptom-burden quantiles); prediction uses the forward filter only. The predicted distribution of the next latent state is mapped to a label distribution through a state-to-label table estimated on the training sequences with Laplace smoothing. The number of states was selected on a 280-patient development set (four cells) by held-out next-visit log-density, with the one-standard-error rule and the maximum both reported, and the selected models were evaluated by leave-one-patient-out on the remaining 654 patients. A direct L2-penalised logistic model on the same ten standardised measurements and the interval was fitted as the no-bottleneck comparator.

### 6.12 The rich clinical profile

Each of nine clinical domains *d* (the ten measurements of Section 5.3 without LEDD, which becomes the medication input) follows a continuous-time local linear trend (integrated Brownian motion) with the medication entering the observation equation:

```
y[i,d,t]        = level[i,d,t] + beta[d] * ledd[i,t] + N(0, s_obs[d])
level[i,d,t+Δ]  = level[i,d,t] + Δ * rate[i,d,t] + w_L
rate[i,d,t+Δ]   = rate[i,d,t]                    + w_R
Cov([w_L, w_R]) = q[d] * [[Δ^3/3, Δ^2/2], [Δ^2/2, Δ]]
```

so irregular spacing is handled exactly. The per-domain parameters (observation variance, level and rate innovation variances, the initial rate variance and the medication coefficient; 45 in all, 27 for the level-only random-walk variant and 36 for the variant without medication) are shared across patients and fitted on training patients only by maximising the Kalman-filter log-likelihood with L-BFGS-B (at most 600 iterations, function tolerance 10<sup>−10</sup>, variances kept positive by floors and log-bounds); convergence is judged by the projected gradient relative to the log-likelihood scale. Inference is a Kalman filter, so the filtered state after visit *k* is a function of visits 0, …, *k* only; the smoother is used only for fitting on training patients. Missing domains are dropped from the update. Each patient's representation at each visit is the filtered level and rate in every domain (18 coordinates). The next visit's domain vector is predicted from the filtered state and Δ; the label is derived from the predicted tremor and PIGD distributions through the rule (Section 6.7). Baselines are last observation carried forward (LOCF) and the population mean trajectory. Validation is 5-fold patient cross-validation repeated four times (20 fits).

### 6.13 The ratio-banded CT-HMM

The CT-HMM of Section 6.3 reads, instead of the three labels, the joint categories (subtype, ratio band). Bands are left-closed, right-open: the main range [0, 3) in fixed bands of width *w* (0.10 in the primary experiment; 0.05, 0.20, 0.30 and 0.50 in the sweep), a tail from 3.0 in 0.25-wide candidates merged upward until each tail band holds at least 30 training observations from at least 20 patients (the topmost band is closed as [lo, ∞)), and one "undefined" category per subtype that can produce a zero PIGD score. The emission is rectangular and factorises as

```
E[a, (s, b)] = M[a, s] * C[a, s, b],   sum_b E[a, (s, b)] = M[a, s]
```

with *M* the 3 × 3 subtype misclassification matrix, parameterised and bounded exactly as the square model's *E* (floor 0.5 on its diagonal), and *C*[*a*, *s*, ·] a softmax distribution over the bands of subtype *s* for latent state *a*. The latent process, the forward recursion and the optimiser settings are unchanged, the gradient is exact, four restarts are used, and the aggregated next-label prediction is α*P*(Δ)*M*, the same functional form as the square model: bands can change the prediction only through the filtered belief α. The square model is the special case with one band per subtype, which the tests check numerically. The parameter count is 14 + 3 Σ<sub>*s*</sub> (*n*<sub>bands,*s*</sub> − 1): with 30 TD, 4 Indeterminate and 10 PIGD categories at *w* = 0.10, 137. A control arm runs the subtype-only model through the joint code path.

### 6.14 The harmonised comparison

Fourteen specifications from five families were re-scored on the same 3,361 transitions: every model expressed as a probability vector over (TD, Indeterminate, PIGD), the chains and HMMs natively and the subscore models by a fresh Monte Carlo of 10,000 draws through the unmodified classification rule with common random numbers across models (the joint predictive covariance for the GP family, independent draws for the models that saved no covariance), and the argmax of that vector as the decision rule for every model without exception. Two persistence rules appear and are not interchangeable: recorded-label persistence, which names the current recorded label, is the canonical reference for every contrast; subscore LOCF persistence carries the last subscores forward through the rule and is a separate baseline.

### 6.15 History windows, retraining, stages and patient-level audit

A history window "last *m*" retains only the last *m* labelled visits of the history, including the current visit, with their real gaps; nothing is re-spaced or padded, the filtered state is rebuilt from the retained visits only, and the first retained visit is treated as drawn from π. A tamper test inside every fold perturbs the final label of each sub-sequence and confirms that the recomputed prediction changes by the amount the model implies (and that untouched cases change by exactly zero). "Full-trained" models are fitted on the training patients' complete sequences; "window-trained" models are fitted on all overlapping two-visit windows of the training sequences (3,354 windows from 933 patients in a representative fold), and "disjoint-window-trained" on non-overlapping windows (1,912). Paired comparisons are made on the common targets available to every window. The stage-wise evaluation scores stage *k*, the prediction of visit *k* + 1 from visits 1 to *k*, separately for *k* = 1, …, 12, and within patients across stages. The patient-level audit recomputes each patient's accuracy, PIGD sensitivity and TD specificity from the saved predictions of experiments 6A and 6B, groups patients by the composition of their recorded target labels (all TD, majority TD, evenly split, majority PIGD, all PIGD; defined after prediction from recorded labels only), and pairs patients across conditions.
