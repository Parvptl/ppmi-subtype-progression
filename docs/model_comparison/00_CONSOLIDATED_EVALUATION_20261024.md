# Comparing our two Parkinson's progression models: fit, discrimination, and the Indeterminate cases

**Consolidated report for discussion · 24 October 2026**

*Combines three analyses: information criteria (AIC/BIC), ROC/AUC
discrimination, and a test of whether the models' forced calls on Indeterminate
visits carry information. All use existing held-out predictions. No model was
retrained or modified.*

---

## Summary for the impatient reader

1. **Both models' architectures are justified by the data.** The CT-HMM's
   hidden-state layer is strongly supported (ΔAIC 224 over the best
   alternative). In the rich profile, every component — the rate of change and
   the medication input — earns its parameters on both AIC and BIC.
2. **The rich profile discriminates significantly better** (AUC 0.890 vs 0.858;
   paired difference +0.032, 95% CI +0.023 to +0.042). This is the first
   measure on which the two models separate.
3. **This refines our earlier statement.** We previously reported the two models
   as "statistically indistinguishable", which is true of *accuracy* but not of
   ranking. Accuracy reads a single operating point on a near-saturated
   endpoint; three other measures all favour the rich profile.
4. **Indeterminate visits remain genuinely unpredictable.** A test of whether
   the models' forced TD/PIGD calls track which boundary those visits sit
   nearest returned **null for both**. Excluding them from scoring remains
   correct.

---

## 1. What we are comparing

| | Method 1 | Method 2 |
|---|---|---|
| Name | CT-HMM | Rich clinical profile |
| What it models | the subtype label (TD / Indeterminate / PIGD) as a noisy reading of a hidden state | nine continuous clinical measurements per visit |
| Primary output | next visit's subtype | next visit's nine measurements |
| Medication | not used | levodopa dose as an input |
| Validation | leave-one-patient-out, 934 folds | 5-fold patient CV × 4 repeats |

For any head-to-head comparison below, both models are scored on **exactly the
same held-out visits**, matched on (patient, current visit, next visit).

---

## 2. Information criteria (AIC / BIC)

### Method, in plain terms

AIC and BIC score a model by how well it fits the data, with a penalty for each
parameter it uses. Lower is better. They answer: *is this extra complexity
earning its keep?* BIC penalises parameters more harshly than AIC, so a model
that wins on both is on firmer ground.

Each model was fitted once on the full cohort purely to obtain a likelihood at
its own maximum, which is what these criteria require.

### Why we must keep the two model families separate

**AIC and BIC cannot be compared between our two methods.** Two independent
reasons:

1. **They describe different data.** The CT-HMM's likelihood covers 4,295
   categorical labels; the rich profile's covers 38,486 continuous
   measurements. Different observations, different counts.
2. **Probability mass versus probability density.** The CT-HMM's likelihood is
   a probability (never above 1, so its log is always negative). The rich
   profile's is a density, which carries units and changes if a measurement is
   rescaled — a log-density can even be positive. Subtracting one from the
   other is meaningless.

So the rich profile's much larger AIC (≈ 91,700 vs ≈ 6,000) says **nothing**
about it being worse. The two tables below are read independently.

### Family 1 — the subtype models

All three describe the same label sequences, placed on a common joint
likelihood. (The chains normally model only transitions; we gave them an
initial-state distribution, costing 2 parameters, so all three cover the same
4,295 observations.)

| Model | log-likelihood | parameters | AIC | BIC | ΔAIC | ΔBIC |
|---|---|---|---|---|---|---|
| **CT-HMM (3 hidden states)** | **−2,969.06** | 14 | **5,966.12** | **6,055.23** | **0** | **0** |
| Markov chain + initial state | −3,087.12 | 8 | 6,190.23 | 6,241.15 | +224.11 | +185.92 |
| CTMC + initial state | −3,099.84 | 8 | 6,215.68 | 6,266.60 | +249.56 | +211.37 |

**The hidden-state layer is decisively supported.** Six extra parameters buy
118 log-likelihood points; ΔAIC 224 and ΔBIC 186 are far beyond conventional
thresholds. This is the formal counterpart of the history-dependence we
documented earlier (χ² = 156.6, df = 4, p = 7.9 × 10⁻³³): a model without a
hidden state cannot represent the fact that where a patient came from changes
where they go next.

*Among the two chains alone (transitions only, 3,361 pairs — a different
likelihood again, so not comparable with the table above): the discrete Markov
chain scores AIC 4,747.55 against the interval-aware CTMC's 4,773.00, a
difference of 25.45. Modelling elapsed time does not pay for itself in a chain
over the observed label.*

### Family 2 — the rich-profile variants

All three describe the same nine continuous domains and are nested.

| Model | log-likelihood | parameters | AIC | BIC | ΔAIC | ΔBIC |
|---|---|---|---|---|---|---|
| **level + rate + medication** | **−45,812.36** | 45 | **91,714.73** | **92,099.84** | **0** | **0** |
| level only (random walk) | −45,940.77 | 27 | 91,935.53 | 92,166.60 | +220.81 | +66.76 |
| level + rate, no medication | −45,993.34 | 36 | 92,058.68 | 92,366.77 | +343.96 | +266.94 |

**The full specification wins on both criteria.** The rate component costs 18
parameters and buys 128 log-likelihood points — it survives even BIC's stiffer
penalty (+66.76). The medication input costs 9 parameters and buys 181 points,
making it the most clearly justified component.

---

## 3. Discrimination: ROC and AUC

### Method, in plain terms

AUC asks a single question: given one visit that turns out PIGD and one that
turns out TD, how often does the model give the PIGD one the higher
probability? 0.5 is a coin flip, 1.0 is perfect. Unlike accuracy it does not
depend on where we set the decision threshold, and unlike the Brier score it
ignores calibration — it measures *ordering* alone.

Both models were scored on the identical **3,020 held-out visits** (898
patients; 910 actually PIGD, 2,110 actually TD) whose actual next label was TD
or PIGD. Confidence intervals resample **patients**, not individual
predictions, because one patient contributes several correlated visits.

![Figure 1. ROC curves for both models on the identical 3,020 held-out predictions. The two curves nearly coincide at the left-hand edge — the region a 0.5 decision threshold operates in — and separate through the middle, which is why AUC detects a difference that accuracy does not. The persistence rule emits only 0 and 1, so it has a single operating point rather than a curve.](figures/fig1_roc_td_vs_pigd.png)

### Results

| Model | AUC | 95% CI |
|---|---|---|
| **Rich clinical profile** | **0.8902** | 0.8748 – 0.9053 |
| CT-HMM | 0.8583 | 0.8411 – 0.8755 |
| Persistence rule (reference) | 0.7927 | 0.7725 – 0.8127 |

**Paired difference, rich profile − CT-HMM: +0.0319 (95% CI +0.0229 to +0.0419).
The interval excludes zero.**

AUC was computed with scikit-learn and cross-checked against our own rank-based
implementation; the two agree to machine precision.

### How this fits with our earlier accuracy result

It does not contradict it. We previously reported the two models as
indistinguishable on accuracy (+0.0033, 95% CI −0.0050 to +0.0112), and that
remains exactly true. Figure 1 shows why both statements hold: the curves
almost coincide at the low-false-positive end — where a 0.5 threshold
operates — and separate in the middle, at operating points a hard classifier
never visits.

In words: the rich profile *orders* patients by PIGD risk better, but at the
threshold used for classification that better ordering does not turn into more
correct calls, because the subtype label is strongly persistent and most cases
sit far from the decision boundary in both models.

| Measure | Which model it favours |
|---|---|
| Accuracy | indistinguishable (+0.0033, CI spans zero) |
| Brier score | rich profile (0.1177 vs 0.1215) |
| Log loss | rich profile (0.3766 vs 0.3963) |
| **AUC** | **rich profile (CI excludes zero)** |

Three of four measures favour the rich profile; only accuracy — the one that
throws the probabilities away — does not.

### Discrimination within each current label

| Current label | n | n PIGD | CT-HMM | Rich profile |
|---|---|---|---|---|
| TD | 2,074 | 252 | 0.6680 | **0.7798** |
| Indeterminate | 268 | 121 | 0.6305 | **0.6794** |
| PIGD | 678 | 537 | 0.6649 | **0.7437** |

The rich profile is ahead in **every** stratum, so its advantage is not an
artefact of the class mix.

**An important caveat.** These stratified figures are far lower than the pooled
0.86–0.89, and that gap is informative: most of the pooled discrimination comes
from the current label itself, because the subtype rarely changes. Once we hold
the current label fixed, the residual ability to predict change is modest for
both models — clearly better for the rich profile, but modest.

---

## 4. The Indeterminate visits

### The question

About 10% of next visits are labelled Indeterminate — the tremor/PIGD ratio
falls between 0.90 and 1.15. We exclude those from binary scoring because
neither TD nor PIGD is the right answer. But each sits closer to one boundary
than the other. **If a model's forced call tracked that position, the
prediction would still carry information.** We tested that.

### What we analysed

| | n |
|---|---|
| Indeterminate-target visits shared by both models | 332 |
| No ratio exists (both subscores zero) — excluded | 45 |
| **Analysed** | **287** (220 patients) |

### A structural finding that shaped the test

![Figure 2. Where Indeterminate visits actually fall inside the band. Bars are coloured by the nearer boundary. Because the tremor score is a mean of 11 integer items and the PIGD score a mean of 5, only a few ratio values are reachable inside a window this narrow, and three quarters of visits sit at just two of them — each about a hundredth away from a boundary.](figures/fig2_indeterminate_ratio_distribution.png)

Half of these visits sit at ratio 0.9091 — a hundredth above the PIGD boundary
— and a further quarter at 1.1364, a hundredth below the TD boundary.
"Distance to the nearer boundary" therefore has far less resolution than the
phrase suggests; it is close to a binary label. 172 visits (59.9%) are nearer
the PIGD boundary, 115 (40.1%) nearer the TD boundary.

### Result: not supported

![Figure 3. Left: the share of calls that were PIGD, split by which boundary the visit sits nearer. If the calls tracked the boundary, the left bar in each pair would clearly exceed the right. Right: overall agreement with the nearer boundary against the agreement expected if the calls were independent of position (dashed line), with patient-level confidence intervals.](figures/fig3_indeterminate_boundary_agreement.png)

| Model | Agreement with nearer boundary | Expected if independent | **Excess** | 95% CI |
|---|---|---|---|---|
| CT-HMM | 0.4878 (0.4336 – 0.5398) | 0.4623 | +0.0255 | **−0.0287 to +0.0775** |
| Rich clinical profile | 0.5017 (0.4483 – 0.5588) | 0.4706 | +0.0312 | **−0.0223 to +0.0882** |

**Both intervals include zero.** Neither model's forced call tracks the nearer
boundary better than chance. The comparison null already accounts for the fact
that both models mostly say TD in this subgroup.

Counts behind those rates:

| Model | Nearer boundary | Called PIGD | Called TD | % called PIGD |
|---|---|---|---|---|
| CT-HMM | PIGD (n=172) | 57 | 115 | 33.1% |
| CT-HMM | TD (n=115) | 32 | 83 | 27.8% |
| Rich profile | PIGD (n=172) | 65 | 107 | 37.8% |
| Rich profile | TD (n=115) | 36 | 79 | 31.3% |

The direction is right for both — PIGD is called more often when the visit lies
nearer the PIGD boundary — but the gaps (5.3 and 6.5 percentage points) are
small relative to their uncertainty.

**Sensitivity check.** Restricting to the 213 visits within 0.05 of a boundary,
where the construct is meaningful, the result is still null: CT-HMM excess
+0.0378 (−0.0285 to +0.1029); rich profile +0.0624 (−0.0017 to +0.1268). The
rich profile's lower bound of −0.0017 is a near miss, **not** a positive result.

**One place a weak signal does appear.** Asking the softer question — does the
predicted probability *rank* nearer-PIGD visits above nearer-TD ones? — gives
AUC 0.5843 (0.5162 – 0.6534) for the CT-HMM and 0.5748 (0.5054 – 0.6403) for
the rich profile. Both intervals exclude 0.5, so a little ordering information
exists. But the effect is small, the rich profile's lower bound of 0.5054 is as
marginal as a positive gets, and we applied **no multiplicity correction**
across the several tests here. We do not treat this as a usable finding.

**Conclusion:** excluding Indeterminate targets from scoring remains the right
choice. These analyses give no basis for recovering partial credit from the
models' forced binary predictions on them.

---

## 5. Limitations

**Applying to everything here**

1. **The two models use different validation schemes** — leave-one-patient-out
   for the CT-HMM, repeated cross-validation for the rich profile. Matching the
   cases removes the difference in *which visits* are scored, but not the
   difference in scheme.
2. **No external cohort.** Everything is within PPMI; nothing has been
   validated elsewhere.
3. **The target is our own classification rule** applied to recorded
   measurements, not independently adjudicated diagnoses. Accuracy here means
   agreement with a noisy record.
4. **Indeterminate targets (≈10%) are excluded** from every binary analysis, so
   the headline figures describe 90% of prediction opportunities.

**Specific to AIC/BIC**

5. Both criteria assume independent observations. Visits within a patient are
   correlated, so the effective sample size is smaller than the term count and
   both penalties are optimistic. The margins are large enough that this is
   unlikely to reverse the rankings, but they should not be read as exact.
6. They come from a **single full-cohort fit**, so unlike the held-out figures
   they carry no cross-validated uncertainty.
7. The rich-profile log-likelihood is a density on the standardised scale.
   Rescaling would shift it by a constant — identical across the three
   variants, so the within-family comparison is unaffected.

**Specific to ROC/AUC**

8. AUC is **invariant to calibration**: a model can rank perfectly and still be
   poorly calibrated. Brier and log loss cover that, and also favour the rich
   profile.
9. The stratified AUCs in §3 use the same data for stratification and
   evaluation, with no multiplicity correction. They are descriptive.

**Specific to the Indeterminate analysis**

10. The ratio is nearly binary inside the band, so the "distance to boundary"
    construct is coarse. The sensitivity analysis is the more meaningful test.
11. 45 of 332 visits have no defined ratio and are excluded.
12. **No multiplicity correction** across the agreement, sensitivity and AUC
    tests. With six intervals reported, one marginal exclusion of a null is weak
    evidence.
13. The actual ratio is itself a noisy measurement. A visit at 0.9091 might have
    been recorded at 0.88 on a different day and classified PIGD instead — so
    part of what we are asking the models to track may not be stable in the data.

---

## 6. Recommended next steps

**Immediate, no new modelling**

1. **Amend the manuscript** to report both results rather than accuracy alone.
   A single sentence suffices: *the two methods are indistinguishable on
   accuracy (+0.0033, CI −0.0050 to +0.0112) but the rich profile ranks cases
   significantly better (AUC 0.890 vs 0.858; +0.032, CI +0.023 to +0.042).*
2. **Adopt continuous scores as the primary basis for model selection in this
   project.** Accuracy has now failed to detect a difference that AUC, the
   Brier score, log loss and the information criteria all find. The subtype
   label is too persistent and too coarse to separate models on accuracy.
3. **Keep the full rich-profile specification** and the CT-HMM's hidden-state
   layer. Both are supported on AIC and BIC.

**Short, worthwhile**

4. **Report decision thresholds other than 0.5.** Figure 1 shows the models
   differ in regions a 0.5 threshold never reaches. If the clinical priority is
   catching PIGD conversions — currently about a third are missed — a lower
   threshold is the direct lever, with the operating point chosen inside the
   cross-validation folds rather than on the test data.
5. **Add cross-domain correlation to the rich profile.** Levels and rates are
   currently modelled independently per domain, and we know from the principal
   component analysis that structure exists across them.

**Larger, and genuinely open**

6. **Reconsider the endpoint.** A persistent three-level label derived from two
   of nine measurements is a weak target for distinguishing progression models.
   Clinically meaningful milestones — reaching Hoehn & Yahr 3, developing motor
   complications, MCI — are available in PPMI, were not used in fitting, and
   discriminate between patients far better than the subtype label does.
7. **Treat the Indeterminate band as a measurement question rather than a
   modelling one.** Given that half of these visits sit one hundredth from a
   boundary, the more useful question may be how reproducible that
   classification is at all, rather than how well we can predict it.

---

## 7. Sources

All numbers verified against these outputs; each run directory carries a
`MANIFEST.sha256` over its own files.

| Analysis | Run directory |
|---|---|
| AIC / BIC | `experiments/20261023T194317Z_model_selection_criteria/` |
| ROC / AUC | `experiments/20261023T202640Z_roc_auc_comparison/` |
| Indeterminate boundary | `experiments/20261023T203431Z_indeterminate_boundary/` |
| Held-out accuracy, Brier, log loss | `experiments/20261020T180515Z_clinical_profile_cv/` |

Scripts: `scripts/65_model_selection_criteria.py`,
`scripts/66_roc_auc_comparison.py`,
`scripts/67_indeterminate_boundary_analysis.py`.

Fuller versions of each analysis: `01_AIC_BIC_AND_HELDOUT_20261024.md`,
`02_ROC_AUC_20261024.md`, `03_INDETERMINATE_BOUNDARY_20261024.md`.
