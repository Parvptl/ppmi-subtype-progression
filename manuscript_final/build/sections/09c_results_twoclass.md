### 9.12 Next-visit TD-against-PIGD prediction: latent state and elapsed time (E25)

**Question.** Restricting the target to the two extreme labels, which of a latent state and an interval-aware observed process matters for predicting the next label, and how do generative and discriminative models compare?

**Setup and cohort.** The 3,029 eligible transitions of 899 patients (the 332 Indeterminate outcomes excluded from scoring and reported separately), leave-one-patient-out, seven models on identical cases (Section 6.10), pre-specified paired contrasts with 2,000 patient-bootstrap resamples. Every fold of every model converged (934 of 934; worst projected gradient 0.025 for the CT-HMM).

**Results.** {{table:60}}

{{table:61}}

The CT-HMM reaches 0.837 (patient-bootstrap 95% CI 0.820 to 0.854; 2,535 of 3,029 correct), two-class Brier 0.121 and log loss 0.396; the direct logistic model with the interval 0.836, persistence of the last non-Indeterminate label 0.834 (Brier 0.160; its log loss is undefined and is 4.19 when clipped), the discrete chain 0.829, the continuous chain 0.828, the direct model without the interval 0.819 and the two-class prior 0.698. The pre-specified contrasts are: latent state, holding the observed process interval-aware (CT-HMM minus CTMC), +0.009 (+0.000 to +0.018), with the CT-HMM right and the CTMC wrong on 98 cases against 70 the other way; elapsed time in the observed-label chain (CTMC minus discrete chain), −0.002 (−0.004 to +0.000); elapsed time in the discriminative family, +0.017 (+0.011 to +0.022); CT-HMM minus direct logistic +0.001 (−0.005 to +0.008); CT-HMM minus persistence +0.003 (−0.004 to +0.010); and CT-HMM minus the prior +0.139 (+0.111 to +0.167).

{{table:62}}

{{table:63}}

{{table:64}}

The CT-HMM's accuracy is 0.881 when the current label is TD, 0.607 when it is Indeterminate (the chains reach 0.544 there) and 0.792 when it is PIGD; 0.842 at intervals up to 1.5 years and 0.808 beyond (the CTMC falls to 0.748 beyond 1.5 years); and 0.834, 0.864 and 0.827 with one, two and three prior visits. Its PIGD recall is 0.667 at precision 0.763 (confusion: 1,925 TD and 610 PIGD correct, 304 PIGD called TD, 190 TD called PIGD), and its calibration gap is 0.0065. On the 332 Indeterminate outcomes the CT-HMM's mean conditional probability of PIGD was 0.371 and its mean raw Indeterminate mass 0.124; it would have called 111 of them PIGD and 221 TD.

**Interpretation.** Changing the task from three classes to two raises accuracy from 0.754 to 0.837 because the hardest 10% of outcomes are set aside, not because any model improved. On the two-class task the latent layer is worth under one point, elapsed time is worth nothing in the observed chain and 1.7 points in the discriminative model, and a ridge-penalised logistic regression on ten summary features of the label history matches the generative model exactly. The models separate only at Indeterminate visits and at long intervals, and even there by a few points.

**Limitations.** The 0.837 describes 90.1% of prediction opportunities; the 9.9% of Indeterminate outcomes are not scored. The interval is an input, so the model is told when the next examination happens.

**Contribution.** The two-class reference, its contrasts, and the discriminative equivalence that shows the generative structure is not what the accuracy depends on.

### 9.13 A discrete latent clinical state over ten measurements (E26)

**Question.** Does representing the patient by a latent clinical state inferred from ten clinical measurements predict the next label better than the subtype history alone?

**Setup and cohort.** State count selected on a 280-patient development set by held-out next-visit log-density; six states by the one-standard-error rule and eight by the maximum (and by BIC); leave-one-patient-out on the 654 evaluation patients, 2,091 scored two-class cases (627 patients with cases); comparators are the subtype-history CT-HMM, persistence, and direct logistic models on the same ten measurements and on the tremor and PIGD scores only.

**Results.** {{table:65}}

{{table:66}}

{{table:104}}

Held-out log-density improves monotonically from two to eight states, but the fits are multimodal from four states upward (the spread of the training log-likelihood across starts exceeds 1,000 nats). The six-state model reaches 0.726 accuracy (0.696 to 0.753) and the eight-state model 0.734 (0.706 to 0.761) against 0.838 (0.818 to 0.857) for the subtype-history CT-HMM and 0.835 for persistence on identical cases: paired differences of −0.113 (−0.144 to −0.081) and −0.105 (−0.136 to −0.076). The loss is concentrated where the current label is PIGD (accuracy 0.397 and 0.437 against 0.787) and in PIGD recall (0.218 and 0.233 against 0.659). A direct logistic model on the same ten measurements and the interval reaches 0.831 (0.809 to 0.852) with the best Brier (0.116) and log loss (0.365) of any model: +0.105 (+0.076 to +0.134) over the latent model and −0.008 (−0.020 to +0.005) against the subtype-history CT-HMM; restricting it to the tremor and PIGD scores changes nothing (−0.001, −0.011 to +0.009).

**Interpretation.** The discrete bottleneck destroys information: the states order patients by overall severity and discard the within-patient variation (33% of the variance in these measurements, 52% for PIGD) that the prediction needs. The same measurements without the bottleneck match the subtype history exactly, and two of the ten measurements carry everything the other eight add. This experiment is the reason the programme replaced the discrete state by a continuous per-domain profile.

**Limitations.** The evaluation set is a 654-patient subset, so the absolute numbers differ from the 899-patient runs; the EM fits are multimodal.

**Contribution.** A clear negative result for discrete latent clinical states, and the finding that the ten-measurement direct model does not beat the two-measurement one.

### 9.14 The rich clinical profile: predicting the clinical record, deriving the label, and what the representation carries (E27, E28)

**Question.** Does a continuous level-and-rate profile over nine clinical domains, with medication as an input, predict the next visit's clinical record better than assuming no change; does its derived label beat the CT-HMM; and do its coordinates carry information about later clinical outcomes that the label does not?

**Setup and cohort.** The 934 patients, 3,353 predictions and 30,099 domain observations under 5-fold patient cross-validation repeated four times; the full profile, a level-only random-walk variant and a no-medication variant against LOCF and the population mean trajectory on held-out log-density per domain observation; the derived two-class label paired with the subtype-history CT-HMM's LOPO predictions on 3,020 common cases (898 patients; the A-3 run with the committed code); outcome associations on outcomes never used in fitting, with the best coordinate selected on the same data and the optimism stated.

**Results.** {{table:67}}

{{table:68}}

{{figure:68}}

{{figure:69}}

The full profile reaches −1.045 log-density per domain observation (−1.071 to −1.022) against −1.049 for the level-only variant, −1.051 without medication, −1.119 for LOCF and −1.434 for the population trajectory. The paired contrasts are +0.073 (+0.066 to +0.081) over LOCF, better in 75.1% of predictions; +0.389 (+0.363 to +0.417) over the population trajectory; +0.004 (+0.002 to +0.006) for modelling the rate; and +0.006 (+0.004 to +0.008) for medication as an input. Standardised RMSE is 0.70 and 95% coverage 0.941. The fitted medication coefficient is positive in eight of nine domains (Table 72 and the fold coefficients), which is confounding by indication: patients who start levodopa are those who are worsening. The fitted tremor rate variance is zero in every fit, so the tremor rate is not estimated; the PIGD rate has a standard deviation of 0.090 per year and the Part II rate 0.375 per year across patients. Of the twenty fits of the full model, sixteen terminated at the 600-iteration limit with a projected gradient of up to 1.75 (by the run's own relative criterion all twenty are recorded as converged), whereas the level-only and no-medication variants converged by scipy's criterion in every fit; the parameters are stable across the four repeats (standard deviations of the per-domain variances below 0.02) and the per-repeat log-densities agree to 0.001.

{{table:69}}

The derived two-class label from the full profile reaches 0.840 (0.825 to 0.857; Brier 0.118, log loss 0.377), the level-only profile 0.844 (0.829 to 0.860), the no-medication profile 0.840, the CT-HMM 0.837 (0.820 to 0.854), persistence 0.834, the discrete chain 0.830, LOCF 0.825 and the population trajectory 0.691, on the same 3,020 cases. The paired differences against the CT-HMM are +0.003 (−0.005 to +0.011) for the full profile and +0.007 (−0.001 to +0.015) for the level-only profile; the profile is right and the CT-HMM wrong on 81 cases against 71 the other way.

{{table:70}}

{{figure:67}}

{{table:71}}

{{figure:66}}

{{table:72}}

{{figure:70}}

Against outcomes never used in fitting, the best single level coordinate discriminates a later Hoehn and Yahr stage of 3 or worse at AUC 0.685 (Part II level) against 0.579 for the subtype label (+0.105, +0.054 to +0.156, 761 patients, 127 events), and mild cognitive impairment at 0.710 (MoCA level) against 0.549 (+0.160, +0.109 to +0.213; 818 patients, 298 events); motor complications are not discriminated by either (0.531 against 0.504, +0.027, −0.039 to +0.094), and the investigator's cognitive diagnosis, which is not independent of the inputs, is discriminated at 0.678 against 0.558. No rate coordinate beats the label by an interval that excludes zero. The representation is genuinely multi-dimensional: seven components are needed for 90% of the across-patient variance of either the level or the rate coordinates, with the first component carrying 42% (levels) and 37% (rates) and participation-ratio effective ranks of 4.3 and 5.1.

**Interpretation.** The profile is a better model of the clinical record than assuming no change, and medication and rates each add a small, interval-excluding amount, but on the next-label task it is indistinguishable from the CT-HMM, and its simplest variant, a random walk on the levels, is as good as its richest. Where the profile differs from the label is in what it carries: its level coordinates predict staging and cognitive outcomes that the subtype label does not. That is the positive result of the programme, and it is a result about the representation, not about label prediction.

**Limitations.** The outcome associations select the best coordinate on the evaluation data, so every difference is an upper bound; the cognitive-diagnosis outcome shares inputs with the profile; the medication coefficient is confounded; sixteen of twenty full-profile fits stopped at the iteration limit; the derived-label comparison pairs five-fold and leave-one-patient-out predictions.

**Contribution.** Evidence that richer observations do not raise next-label accuracy, and that a continuous representation nonetheless carries clinically relevant information the label lacks.

### 9.15 Information criteria and threshold-free discrimination (E29, E30)

**Question.** Within each likelihood family, which model do AIC and BIC prefer, and does the profile rank a PIGD next visit above a TD one better than the CT-HMM?

**Setup and cohort.** AIC and BIC on the full cohort within three groups whose likelihoods are mutually comparable (Section 8.4); AUC-ROC with PIGD positive on the 3,020 matched transitions (898 patients, 910 positives) with a paired patient bootstrap.

**Results.** {{table:73}}

Among joint sequence likelihoods the CT-HMM (14 parameters, AIC 5,966) is preferred over the discrete chain with an initial distribution by 224 AIC units and 186 BIC units, and over the continuous chain by 250 AIC units; among the chains alone the discrete chain is preferred over the continuous one by 25 units; among the profile variants the full model is preferred over the level-only variant by 221 AIC and 67 BIC units and over the no-medication variant by 344 and 267.

{{table:74}}

{{figure:36}}

The profile's AUC is 0.890 (0.875 to 0.905) against 0.858 (0.841 to 0.876) for the CT-HMM, a paired difference of +0.032 (+0.023 to +0.042); persistence, a one-hot rule, scores 0.793. By current label the profile ranks better in every stratum (TD 0.780 against 0.668, Indeterminate 0.679 against 0.631, PIGD 0.744 against 0.665).

**Interpretation.** In-sample, both the latent layer and the profile's extra structure earn their parameters; held out, the profile ranks PIGD visits above TD visits clearly better than the CT-HMM while calling them no more accurately at the 0.5 threshold. The ranking gain lives in the ordering of cases on either side of the decision boundary, which accuracy cannot see.

**Limitations.** AIC and BIC are in-sample; the AUC uses the conditional two-class probability.

**Contribution.** The first instance of the pattern that recurs in Sections 9.22 and 9.23: richer observations improve ranking, not accuracy.

### 9.16 What the models say at Indeterminate visits (E31)

**Question.** At an Indeterminate visit whose ratio is defined, do the models call the side of the band the ratio sits nearer, and do their calls at the two dominant lattice ratios differ?

**Setup and cohort.** The 287 Indeterminate visits with a defined ratio from 220 patients (a sensitivity set of 213 closest to a boundary); the two focal ratios 0.9091 (144 visits, nearer PIGD) and 1.1364 (67 visits, nearer TD); agreement with the nearer boundary against the value expected under independence; patient-clustered intervals from 4,000 draws.

**Results.** {{table:75}}

{{figure:37}}

{{figure:38}}

{{table:76}}

{{figure:39}}

{{figure:40}}

{{figure:41}}

The CT-HMM agrees with the nearer boundary at 0.488 of Indeterminate visits against 0.462 expected under independence, an excess of +0.026 (−0.029 to +0.078); the profile at 0.502 against 0.471, +0.031 (−0.022 to +0.088); on the 213 visits closest to a boundary the excesses are +0.038 (−0.029 to +0.103) and +0.062 (−0.002 to +0.127). The AUC of the conditional PIGD probability for being nearer the PIGD boundary is 0.584 (0.516 to 0.653) and 0.575 (0.505 to 0.640). At the focal ratios the CT-HMM calls PIGD at 0.347 of the 0.9091 visits and TD at 0.731 of the 1.1364 visits; the profile at 0.389 and 0.746. The mean conditional PIGD probability is higher at 0.9091 than at 1.1364 by +0.093 (+0.013 to +0.171) for the CT-HMM and +0.080 (+0.015 to +0.146) for the profile, but the difference in PIGD calls is +0.079 (−0.050 to +0.205) for the CT-HMM and +0.135 (+0.005 to +0.260) for the profile.

**Interpretation.** At an Indeterminate visit both models follow the history, not the ratio: their probabilities lean slightly towards the nearer boundary, and their calls barely do. This is the correct behaviour for models whose inputs are the label history (the CT-HMM) or the whole clinical state (the profile), and it is why both score about 0.61 from an Indeterminate visit: the next label from the band is close to a coin toss weighted by the history.

**Limitations.** Small strata; the focal-ratio analysis is exploratory.

**Contribution.** A direct view of the behaviour that produces the Indeterminate-stratum margins of Section 9.11.

### 9.17 Does accuracy change with disease stage? (E32)

**Question.** Does two-class accuracy change as the prediction stage (the number of prior visits) grows, and does the comparison between the CT-HMM and the profile change with it?

**Setup and cohort.** 3,017 targets of 896 patients at stages 1 to 12 (stages 1 to 9 have at least 66 targets and are treated as reliable); per-stage metrics with patient-bootstrap intervals; changes from stage 1 and within-patient late-minus-early changes (225 patients); 24 pre-specified paired tests.

**Results.** {{table:77}}

{{figure:42}}

{{figure:43}}

{{figure:44}}

{{figure:45}}

{{table:78}}

At stage 1 (835 targets) the CT-HMM reaches 0.835 (0.810 to 0.860), the profile 0.846 (0.821 to 0.870) and persistence 0.832, with balanced accuracies of 0.729, 0.775 and 0.717; the profile's balanced accuracy is higher than the CT-HMM's by +0.046 (+0.023 to +0.070), the only one of the 24 paired tests whose interval excludes zero. No change in accuracy from stage 1 excludes zero for either model at any reliable stage. Balanced accuracy rises with stage for every model, including persistence (from 0.717 at stage 1 to 0.816 at stage 9), because the share of PIGD targets rises from 23% to 48%; the CT-HMM's balanced-accuracy change from stage 1 excludes zero at stages 2 and 6 to 9. Within patients, the late-minus-early change in accuracy is −0.024 (−0.075 to +0.028) for the CT-HMM and −0.035 (−0.089 to +0.020) for the profile.

**Interpretation.** Accuracy does not change with stage, and the apparent rise in balanced accuracy is composition: late stages hold more PIGD targets, which a persistence rule gets right as often as the models. The one stage-1 advantage of the profile in balanced accuracy is its better PIGD sensitivity when only one visit is known, and it does not persist.

**Limitations.** Late stages are small and selected on follow-up.

**Contribution.** Evidence that the ceiling is not a property of early disease.
