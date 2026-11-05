### 9.4 Chains in the recorded label against the CT-HMM on the first split (E10)

**Question.** Does a latent state read through a misclassification matrix predict the next recorded label better than a chain in the label itself, and does continuous time help?

**Setup and cohort.** A single stratified 70/30 patient split (seed 42); 1,002 held-out consecutive pairs. Models: the discrete-time chain, the continuous-time chain, the three-state CT-HMM with emission floor 0.5, and the marginal null. Ten further random splits repeat the HMM-against-DTMC comparison. The primary metric is the held-out log-likelihood per pair; accuracy and the three-class Brier score are secondary.

**Results.** {{table:18}}

{{figure:17}}

On the main split the CT-HMM reaches −0.638 log-likelihood per pair against −0.665 for the discrete chain, −0.675 for the continuous chain and −0.863 for the null; accuracies are 0.769, 0.767, 0.765 and 0.625, and Brier scores 0.346, 0.363, 0.367 and 0.518. Across the ten further splits the CT-HMM beat the discrete chain on every one, by +0.028 log-likelihood per pair on average (SD 0.009). The continuous-time chain, which the interval dependence of Section 9.1 motivated, loses to the discrete chain at every interval stratum on this split (by 0.005 per pair at intervals up to 1.5 years, 0.024 at 1.5 to 2.5 years, and 0.099 beyond), because the discrete chain's "next visit" absorbs the attendance schedule that the continuous chain has to model with six rates.

{{table:19}}

{{figure:18}}

{{table:20}}

{{figure:19}}

{{figure:20}}

The fitted misclassification matrix (Table 19) records a latent TD state correctly at 0.941 and a latent PIGD state at 0.842, while the latent Indeterminate row sits exactly at the floor (0.25, 0.50, 0.25): the data do not identify an Indeterminate state that is recorded as Indeterminate more than half the time, so the constraint is active. The latent generator differs sharply from the observed-label generator (Table 20): the latent TD state leaves for PIGD directly at 0.006 per year and for Indeterminate at 0.094 per year, the latent Indeterminate state leaves for PIGD at 0.356 per year, and the latent PIGD state's exit rate to TD is at the lower bound. The implied mean sojourns are 10.1, 2.1 and 16.5 years in the latent states against 3.4, 0.4 and 2.3 years in the recorded label. Horizon probabilities computed from the latent generator (Table 21) are latent-to-latent and must not be read as probabilities of the recorded label: the latent five-year probability of PIGD given TD is 0.191, whereas the recorded-label chain gives 0.296.

{{table:21}}

{{figure:21}}

**Interpretation.** The latent layer earns its place on the proper score and does so consistently across splits, while accuracy barely moves. Everything the CT-HMM adds is in the probabilities: it is less sure after an isolated label that contradicts the history, and it is right to be. The continuous-time chain without a latent layer is a worse model of the record than the discrete chain, which is a reminder that the interval dependence of Section 9.1 is partly an artefact of how the label is sampled.

**Limitations.** One split and ten resamples of it; the leave-one-patient-out design of the next section supersedes this one for inference. The emission floor is active for Indeterminate, so the latent Indeterminate state is a constrained object.

**Contribution.** The first positive result for the latent-state hypothesis, on the proper score, and the first sign that accuracy will not follow.

### 9.5 Leave-one-patient-out three-class prediction, the stability of the optimum and the five-state model (E11 to E13)

**Question.** Under the strictest patient-level validation, how well does the CT-HMM predict the next recorded label, how does it compare with the chains and with persistence, is its fitted optimum unique, and do more latent states help?

**Setup and cohort.** Leave-one-patient-out over the 934 patients (3,361 predictions); the model is refitted on the other 933 patients in every fold from the same seed with four restarts, and the held-out patient's labels are predicted one visit at a time by filtering. Ten consistency checks passed in every fold. The Stage 4 stability study refitted the full cohort from 40 random starts per configuration and ran a nested cross-validation (ten outer folds, five inner) that selects the configuration by inner-fold log-likelihood or BIC. The five-state model's LOPO run is read from its saved predictions.

**Results.** {{table:22}}

{{figure:22}}

The CT-HMM predicts 2,534 of 3,361 next labels correctly: accuracy 0.754 (patient-bootstrap 95% CI 0.735 to 0.773; Wilson interval 0.739 to 0.768), mean per-patient accuracy 0.754 with a median of 1.0, log-likelihood per prediction −0.673 and three-class Brier score 0.369. The discrete chain reaches 0.747, the continuous chain 0.745, recorded-label persistence 0.723 and the marginal null 0.629. The CT-HMM's margin over persistence is 3.1 points; its margin over the discrete chain is 0.7 points.

{{table:23}}

{{table:24}}

{{figure:23}}

{{figure:25}}

{{figure:26}}

The CT-HMM never predicts Indeterminate: it issues the label once in 3,361 predictions (recall 0.000), because the Indeterminate probability never exceeds the TD or PIGD probability under the argmax rule (its maximum is 0.371). TD is recalled at 0.910 with precision 0.786 and PIGD at 0.666 with precision 0.669. Accuracy by current label is 0.811 (TD), 0.488 (Indeterminate) and 0.702 (PIGD), and by number of prior visits 0.747, 0.790, 0.751, 0.739 and 0.736 for one to five or more. The reliability plot (Figure 24) is close to the diagonal at the confidence levels the model uses.

{{figure:24}}

{{table:25}}

{{table:26}}

The stability study finds the three-state floor-0.5 fit's optimum unique: all 40 starts reach the same log-likelihood within 0.01, with generator entries agreeing to 0.0014 and emission entries to 0.0003. Without the floor the three-state fit sits on a ridge (38 of 40 starts within 0.01 of the best, but generator entries among the best differing by up to 0.187 and emission entries by up to 0.93), and the five-state fit is multimodal (one of 40 starts within 0.01 of the best; the default start is 7.4 log-likelihood units below it). On held-out prediction every configuration is equivalent: accuracies lie between 0.752 and 0.755 and log-likelihoods between −0.669 and −0.675. The nested selection procedure (which chose the unconstrained three-state model in eight outer folds and the five-state model in two) reaches 0.755 (0.735 to 0.774), the same as the audited model. A prior-corrected decision rule that divides each predicted probability by its class prevalence recovers Indeterminate recall of 0.25 to 0.32 at the cost of accuracy, which falls to 0.68 to 0.71; the probabilities themselves do not change.

The earlier five-state LOPO run, re-scored in the harmonised comparison (Section 9.11), reaches 0.751, below the three-state model, and its fits converged in only 404 of 934 folds, so 59.2% of its predictions come from non-converged fits; it is reported here as a negative result and not used further.

**Interpretation.** Under LOPO the CT-HMM's advantage over the discrete chain survives on the proper score and shrinks to under one accuracy point; its advantage over persistence is three points and, as Section 9.11 shows, almost entirely from the Indeterminate stratum. The fitted optimum is unique only because the floor makes it so; the unconstrained and larger models fit the training data slightly better and predict held-out data no better. The model cannot predict Indeterminate under the argmax rule, which is the correct consequence of the label's own dynamics: an Indeterminate visit is followed by Indeterminate only 19% of the time.

**Limitations.** The three-class task scores the next recorded label, so the 9.9% of Indeterminate outcomes are counted as errors for every model that never predicts them. The stability study's held-out table is a sensitivity table, not a model selection.

**Contribution.** The reference three-class result, with its uncertainty, against which every later three-class model is measured, and the demonstration that neither more latent states nor fewer constraints change it.

### 9.6 The hierarchical Bayesian growth model of the ratio and its Student-t variants (E14, E15)

**Question.** Does a continuous model of the ratio, with a patient-specific baseline and slope, predict the next recorded label as well as the CT-HMM, and are its residuals or its slopes heavy-tailed?

**Setup and cohort.** The 821 patients with at least two ratio-defined visits (3,603 visits in the analysable cohort). The growth model is compared with the CT-HMM on the 825 held-out pairs of the main split that both can score, and on six further splits; its longitudinal cross-validation estimates each held-out patient's trajectory from their first one, two or three visits. The Student-t variants are compared over seven patient splits.

**Results.** {{table:27}}

{{figure:27}}

The population ratio starts at 2.29 (95% posterior interval 2.16 to 2.43) and falls by 0.093 per year (0.116 to 0.068 per year of decline); the patient baseline SD is 1.34, the patient slope SD 0.103 per year, and the total residual SD 1.14, which is 4.6 times the width of the Indeterminate band. The ratio's variance is 54% between patients and 46% within (Table 31), with a within-patient SD of 1.22.

{{table:28}}

{{figure:28}}

{{table:29}}

{{table:30}}

Head to head, the CT-HMM reaches −0.656 log-likelihood per pair, 0.761 accuracy and 0.357 Brier against the growth model's −0.681, 0.737 and 0.379 on the same 825 pairs, and it is ahead on all six further splits (mean +0.058 per pair, SD 0.019). In longitudinal cross-validation the growth model's accuracy rises from 0.576 with one conditioning visit to 0.647 with two and 0.666 with three, while its 95% interval coverage falls from 0.974 to 0.904; its predicted probability of PIGD is below the observed rate in every reliability bin, with a mean gap of 0.105.

{{table:31}}

{{table:32}}

{{table:56}}

Student-t residuals improve held-out prediction unambiguously: GT against GG gains +0.112 (95% CI +0.061 to +0.168) in log predictive density per visit and +0.038 (+0.020 to +0.055) on the class-label metric, positive in all seven splits, and TT against TG gains +0.106 (+0.058 to +0.158); the residual degrees of freedom are about 2.1 (posterior 1.88 to 2.39 in the frozen generator). Heavy-tailed slopes are not supported: the original run's slope-tail sampler did not converge (R-hat 1.018, effective sample size 248) and the question was declared inconclusive; the pre-registered reassessment with a collapsed slice sampler converged (R-hat 1.004, effective sample size 937) and found a full-cohort posterior median of 2.09 (1.48 to 4.11) for the slope degrees of freedom but met the per-split criterion in only one of seven splits against a threshold of five, with zero false flags in 180 null calibration cohorts and 55 of 60 detections in the positive control; the verdict is not supported.

**Interpretation.** A growth model of the ratio is a worse predictor of the recorded label than a model of the label itself, by a margin that is consistent across splits, and it under-predicts PIGD because a Gaussian on a ratio whose residual SD is four and a half times the band width places too little mass below 0.90. The heavy residual tails are real and are the same phenomenon the lag-1 and D-1 analyses see: occasional large excursions of the ratio that do not persist. The two variance components of the proposal's model are not identified, so the model cannot separate process from observation noise, which is the one thing a label-prediction model would need it to do.

**Limitations.** A single Gibbs chain without convergence diagnostics; the 821-patient ratio cohort is a selected subset (Section 4.4); the head-to-head set of 825 pairs is a subset of the split's 1,002 pairs.

**Contribution.** The first negative result for the continuous-ratio route, and the evidence that residual tails are heavy, which the Student-t D-1 generator then inherited.

### 9.7 The first-visit model (E16)

**Question.** With no history, what predicts the label at a later visit?

**Setup and cohort.** Ten random 70/30 patient splits; the model of Section 6.5 and its nested variants; 1,002 held-out rows on the main split.

**Results.** {{table:33}}

{{table:34}}

{{table:35}}

Averaged over ten splits, a lookup of the current label and the horizon gives −0.781 log-likelihood per prediction, adding the baseline ratio −0.773, adding the ten baseline measures −0.763, and both −0.755, against −0.787 for the CT-HMM given no history (a gain of 0.032 per prediction). On the main split the full first-visit model reaches 0.719 accuracy and 0.413 Brier. The MDS-UPDRS Part II total is the only covariate whose removal costs more than 0.01 per prediction (0.015); the DaT-scan putamen binding, age and Hoehn and Yahr stage cost 0.005, 0.004 and 0.002; the cognitive, olfactory, autonomic and sleep measures cost nothing or less than nothing. Accuracy falls with horizon from 0.776 within 1.5 years to 0.616 beyond 6.5 years.

**Interpretation.** At a first visit the baseline clinical picture, chiefly motor disability, adds a little to the label and the horizon, and none of the diffuse-synucleinopathy markers help. The result bounds what any model can do without history and is consistent with the ceiling the longitudinal models reach with it.

**Limitations.** Covariates are mean-imputed within the training split; the model is not compared under LOPO.

**Contribution.** A no-history reference, and evidence that non-motor baseline markers do not predict the motor label.

### 9.8 Gaussian-process regression of the ratio: kernels, mean function and multi-step forecasts (E17)

**Question.** Does a flexible trajectory model of the ratio predict the next ratio better than a linear trend, does a fitted mean matter, and does the advantage change with forecast horizon?

**Setup and cohort.** One-step-ahead leave-one-patient-out over 821 patients and 2,707 points, on the unfiltered cohort and three extreme-patient-filtered cohorts (A4, D1, E2); kernels K0 to K3 and the growth model, against persistence, the patient mean, a per-patient OLS line and the population mean; patient-cluster bootstrap with 4,000 replicates. All 11,444 kernel fits across the four cohorts converged. A zero-mean K2 arm was fitted inside the same 821 folds; multi-step forecasts condition on the first half of each patient's visits.

**Results.** {{table:36}}

{{figure:29}}

{{table:37}}

On the unfiltered cohort the RBF kernels reach a pooled MAE of 0.972 (K1, K2) and 0.971 (K3) against 1.005 for the linear K0, 1.006 for the growth model, 1.043 for persistence, 1.063 for the patient mean, 1.336 for the patient OLS line and 1.395 for the population mean. The paired differences are K1 minus K0 −0.033 (−0.047 to −0.021) pooled and −0.019 (−0.028 to −0.011) per patient, K1 minus persistence −0.071 (−0.098 to −0.045), and K3 minus K1 and K0 minus the growth model both include zero; the linear kernel term adds nothing beside an RBF (K2 minus K1 at 10<sup>−6</sup>), and its variance was pinned at its lower bound in every fold. Coverage of the 95% intervals is 0.941 and 12.9% to 13.9% of each predictive distribution lies below zero. The derived three-class label from K1 reaches 0.724 accuracy, below the 0.733 of ratio persistence, and never predicts Indeterminate. The model ordering is identical in all four cohorts (Figure 30, Table 38).

{{figure:30}}

{{table:38}}

{{figure:31}}

{{table:39}}

The fitted linear mean does not change MAE (zero-mean minus fitted-mean −0.007, −0.019 to +0.006) but it improves CRPS (+0.019, +0.013 to +0.025 in favour of the fitted mean), mean squared error (+0.097, +0.063 to +0.131) and the mass below zero (0.130 against 0.170): the zero-mean arm makes occasional large errors when a patient's history runs out, which MAE barely registers.

{{table:40}}

{{table:41}}

{{table:42}}

{{figure:32}}

Multi-step, the RBF kernels' advantage over the linear kernel doubles (K1 minus K0 −0.062, −0.095 to −0.031 over 1,389 forecast points) and the advantage over persistence grows to −0.158 (−0.233 to −0.086): persistence anchors on one noisy observation that ages badly, whereas the GP reverts towards the population trend. Within patients, the first step from the 0 to 1 year bin to the 1 to 2 year bin worsens K1's MAE by +0.230 (+0.051 to +0.402) and persistence's by about +0.28, so both degrade in step; the difference-in-differences against persistence excludes zero only beyond five years (−0.371, −0.655 to −0.104 in the unfiltered cohort) and only in three of four cohorts.

**Interpretation.** The kernel choice is worth about 0.03 ratio units against about 0.4 for knowing which patient it is (population-only 1.39 against patient-aware about 1.0) and 1.22 for the within-patient SD. Flexibility refines the trajectory at a tenth of the individualisation effect and a fortieth of the noise, and the derived label does not improve on ratio persistence. The GP's widening advantage at long range is real but describes forecasting the ratio, not the label.

**Limitations.** Gaussian likelihood on a non-negative ratio (12% to 17% of predictive mass below zero); the mean-function hold-out was run on the unfiltered cohort only; the multi-step comparison omits the growth model and fitted baselines; long horizons are reached only by patients with long follow-up.

**Contribution.** A careful negative result for kernel refinement on the ratio and the reason the programme moved to the subscores.

### 9.9 Modelling the two subscores: independent GPs, the joint mixed model, multi-output GPs and the Stage 3 held-out comparison (E18 to E21)

**Question.** Does predicting the tremor and PIGD subscores, and deriving the label afterwards, improve on the ratio and on persistence, and does a cross-output covariance or a more flexible kernel help?

**Setup and cohort.** Leave-one-patient-out over all 934 patients and 3,361 one-step points, which recovers 654 points and 113 patients the ratio model cannot score (the recovered patients have a mean PIGD subscore of 0.073 against 0.414 cohort-wide and are 308 TD, 23 Indeterminate and 13 PIGD visits). Models: independent K1 GPs; the joint LMM; multi-output GPs with K0 to K3 components, full or diagonal coregionalisation; persistence, patient mean and the LMM as baselines. The Stage 3 run scores fifteen arms on identical points with CRPS, log predictive score and coverage, 4,000 patient-bootstrap replicates, and pairs each derived label with the CT-HMM's LOPO predictions on the same 3,361 points.

**Results.** {{table:43}}

{{table:44}}

The subscores are far better behaved than the ratio (median within-patient SD 0.19 for tremor and 0.14 for PIGD against 0.76 for the ratio; predictive mass below zero 0.053 for tremor against 0.130 for the ratio), but the GP's advantage over persistence does not survive the change of target: K1 minus persistence is −0.003 (−0.008 to +0.002) on the tremor score and +0.013 (+0.008 to +0.019) on the PIGD score, where persistence wins. The reason is the point mass at zero: at the 446 visits where the PIGD score is zero, persistence predicts below 0.1 at 62% of them and the GP never does (its mean prediction there is 0.242).

{{table:48}}

The joint LMM estimates population slopes of +0.023 per year (0.020 to 0.026) for tremor and +0.066 per year (0.062 to 0.069) for PIGD, with a random-intercept correlation of −0.015 and a residual correlation of +0.019 between the two subscores, and intraclass correlations of 0.61 and 0.45. Its held-out MAE (0.255 for tremor, 0.239 for PIGD) is worse than persistence (+0.012, +0.004 to +0.020; +0.035, +0.027 to +0.044).

{{table:45}}

{{table:46}}

In the Stage 3 comparison the full-coregionalisation multi-output GPs are indistinguishable from their diagonal counterparts and from the independent GPs on every metric (the cross-output block changes tremor CRPS by −0.0002 at most), the kernels with a linear term and an RBF or Matérn component (K2, K3) beat K1 and K0 on CRPS and log score for both outcomes (K2 minus K0 joint log score +0.073, +0.051 to +0.096), and every GP beats persistence on CRPS and log score (K1 minus persistence joint log score +0.144, +0.102 to +0.186) while losing to it on PIGD absolute error. All 3,736 MOGP fits converged.

{{table:47}}

{{table:49}}

Derived through the classification rule by Monte Carlo, the subscore GPs reach 0.756 (independent K1) and 0.758 (joint K1, full) three-class accuracy under the argmax rule, against the CT-HMM's 0.754 on the same 3,361 points; the paired difference is +0.0024 (−0.0045 to +0.0092) for the independent K1 and +0.0045 (−0.0026 to +0.0117) for the joint K1, and no argmax accuracy contrast against the CT-HMM excludes zero for any of the fifteen arms. On log loss the K1 GPs are better than the CT-HMM (−0.020, −0.032 to −0.007) and the LMM is worse (+0.018, +0.004 to +0.032). Under the point rule, which classifies the point prediction, the GPs reach 0.730 and predict Indeterminate with recall 0.17.

**Interpretation.** The subscore target recovers the excluded tremor-dominant patients, halves the Gaussian misspecification and makes the continuous models comparable with the CT-HMM for the first time; the answer to that comparison is that they are indistinguishable on accuracy. The cross-output dependence is negligible because the two subscores are nearly uncorrelated, and the gain from temporal flexibility is confined to the proper scores. Nothing on the subscores beats carrying the last PIGD score forward on absolute error, because the PIGD score is zero at 16% of visits and persistence respects that.

**Limitations.** Raw-scale Gaussian likelihoods on bounded, zero-inflated scores; the derived label uses Monte Carlo through a ratio of two Gaussians clipped at zero; the subscore models train on 934 patients and the ratio model on 821, so their MAEs are not comparable with each other.

**Contribution.** The like-for-like comparison of the continuous route with the categorical route, and its answer: equal accuracy, better proper scores, no Indeterminate predictions under argmax.

### 9.10 Does the population progression rate change over follow-up? The time-varying spline model and the Stage 5 temporal audit (E22, E23)

**Question.** Is the rate of change of the subscores constant over follow-up, is any change identifiable, and are the model's intervals honest given patient clustering and attrition?

**Setup and cohort.** The joint LMM of Section 9.9 with a linear-spline population mean (knots at 2 and 5 years), on 934 patients (1,924 visits in years 0 to 2, 1,415 in years 2 to 5 from 725 patients, 956 after year 5 from 273 patients); random-slope variants; alternative knot sets; and the Stage 5 audit (alternative bases under ML, supremum likelihood-ratio test with 200 parametric-bootstrap null cohorts, patient-bootstrap refits with 1,000 replicates, an attrition model, completer-only slopes, and a disease-duration time axis).

**Results.** {{table:50}}

{{figure:33}}

{{table:51}}

{{table:52}}

The PIGD slope rises across periods (0.045, 0.057 and 0.080 per year in years 0 to 2, 2 to 5 and after 5) and the tremor slope falls (0.033, 0.033 and 0.013 per year); the contrasts between the last and the first period are +0.035 (+0.020 to +0.050) for PIGD and −0.020 (−0.034 to −0.006) for tremor under Wald intervals, and within the 226 patients observed in both periods the PIGD rate rises by +0.073 (+0.043 to +0.101) and the tremor rate falls by −0.040 (−0.073 to −0.008). On held-out prediction the spline improves PIGD MAE over the constant-rate model by −0.0034 (−0.0044 to −0.0023) and tremor MAE not at all, and both lose to persistence; the derived label is 0.709 under either.

{{table:53}}

{{table:54}}

{{table:55}}

The Stage 5 audit confirms a departure from linearity (supremum likelihood ratio 39.1 against a bootstrap 95th percentile of 7.7 under the random-intercept null and 20.4 under the random-slope null, *p* = 0.005 under both) but finds no identifiable knot: the best single knot is at 4.75 years and its patient-bootstrap distribution over 200 refits has a median of 5.0 years, an interquartile range of 4.25 to 9.25 years and a 2.5th to 97.5th percentile range of 2.5 to 10 years, a quadratic mean is preferred by AIC (37.8 units better than linear), and the cross-validated gains of every non-linear basis over the linear one are confined to PIGD and are tiny (absolute error −0.003, CRPS −0.001). The Wald intervals of the late-period slopes are too narrow by a factor of 1.7 (tremor) and 2.2 (PIGD) against the patient bootstrap. Attrition is informative: the log-odds of loss to follow-up after a visit rise by +0.52 (+0.36 to +0.71) per point of PIGD score and by +0.14 per year. Completers followed at least six years (247 patients) show the same pattern (PIGD late minus early +0.044, +0.017 to +0.072; tremor −0.046, −0.073 to −0.020). On a disease-duration axis, adding duration to the model improves AIC and the slope changes at five years no longer exclude zero (tremor −0.054, −0.115 to +0.018; PIGD −0.022, −0.098 to +0.064); the treated share rises from 0.10 at baseline to 0.95 by year five.

**Interpretation.** The PIGD score accelerates and the tremor score decelerates over follow-up, within patients as well as between them, but the acceleration has no identifiable timing, is confounded with disease duration and treatment, is estimated with Wald intervals that are too narrow, and is worth almost nothing for prediction. It is a population description, not a predictive feature.

**Limitations.** Knots were fixed in advance; the attrition model is descriptive; the duration axis analysis is in-sample.

**Contribution.** A documented population finding with its identifiability and interval limitations, and the reason no time-varying structure enters the prediction models.

### 9.11 The harmonised comparison across families (E24)

**Question.** When every model is scored on the same 3,361 transitions with the same decision rule, how do the families compare, and where does each model's margin over persistence come from?

**Setup and cohort.** The saved leave-one-patient-out predictions of fourteen specifications from five families (Section 6.14), re-scored as the argmax of a three-label probability vector; subscore models propagated through the classification rule by 10,000 common-random-number Monte Carlo draws (half-draw check: maximum accuracy difference 0.0015); paired contrasts against recorded-label persistence with 2,000 patient-bootstrap resamples; and a decomposition of each model's gain by the label recorded at the prediction point.

**Results.** {{table:57}}

{{figure:34}}

{{table:58}}

Every fitted specification lies between 0.745 and 0.758: the independent subscore GP 0.758, the joint subscore GP 0.756, the three-state CT-HMM 0.754, the five-state CT-HMM 0.751, the constant-rate and spline LMMs 0.749 and 0.746, the joint LMM 0.749, the discrete chain 0.747 and the continuous chain 0.745. Subscore LOCF persistence reaches 0.745 and the patient-mean subscore 0.742; recorded-label persistence 0.723, the marginal null 0.629 and the population mean subscore 0.625. The paired margins over recorded-label persistence are +0.035 (+0.025 to +0.045) for the independent GP, +0.034 for the joint GP, +0.031 (+0.022 to +0.041) for the CT-HMM, +0.029 for the five-state model, +0.027 (+0.014 to +0.040) for the two constant-rate LMMs, +0.025 for the discrete chain, +0.024 (+0.010 to +0.036) for the spline LMM and +0.023 (+0.014 to +0.032) for the continuous chain; every one excludes zero and the whole range spans 1.2 points. On the proper scores the GPs lead (Brier 0.360, log loss 0.654) and the CT-HMM follows (0.369, 0.673). Not one fitted model issues an Indeterminate prediction under the argmax rule (recall 0.000); recorded-label persistence issues 334 and recalls 0.193 of them, which is why its macro-F1 (0.556) is the highest in the table.

{{table:59}}

{{figure:35}}

The decomposition shows where the margins come from. Of the CT-HMM's 105 net cases gained over persistence, 99 (94.3%) come from the 334 transitions that start from an Indeterminate label, where persistence is right 19.2% of the time and the CT-HMM 48.8%; on the 2,261 transitions from TD the two models make identical predictions 98.4% of the time and the CT-HMM gains 6 cases, and on the 766 transitions from PIGD they are identical 96.9% of the time and the gain is zero. The independent GP gains 9, 100 and 8 cases in the three strata (85.5% from Indeterminate), and the discrete chain 0, 83 and 0 (100%).

**Interpretation.** This is the paper's central result. Five families of models, spanning six to several dozen parameters, categorical and continuous observations, discrete and continuous time, memoryless and latent-state dynamics, all reproduce the next recorded label at between 0.745 and 0.758, and all of them beat recorded-label persistence by the same mechanism: at an Indeterminate visit they decline to predict Indeterminate and predict the extreme label the history favours instead. On the other 90% of transitions they are persistence with small corrections. The accuracy ceiling is not a property of any model; it is set by how often the next label contradicts the history, and that is a property of the label.

**Limitations.** Re-scoring saved predictions cannot reproduce the original runs' decision rules bit for bit; the Monte Carlo propagation is a fresh draw. The five-state model's predictions are included although 59% come from non-converged fits.

**Contribution.** The harmonised evidence for the title.
