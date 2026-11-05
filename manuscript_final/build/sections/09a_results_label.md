## 9. Results

Each subsection reports one experiment or one tightly linked group of experiments. The structure is the same throughout: the question, the setup and cohort, the baselines, the results with their uncertainty, the figures and tables, the interpretation, the limitations specific to that experiment, and what it contributes to the paper's conclusion. The cohort itself (E01) is reported in Section 4.3.

### 9.1 Dynamics of the recorded label (E02, E03)

**Question.** How does the recorded subtype label move between consecutive visits, how often does a change persist, how does the movement depend on the elapsed interval, and does the observed label behave as a first-order Markov chain?

**Setup and cohort.** All 3,361 consecutive labelled pairs of the 934 patients with at least two labelled visits. Transition probabilities are row-normalised counts; intervals are Wilson intervals that treat pairs as independent and patient-clustered percentile bootstrap intervals (2,000 resamples of patients, seed 42) from the E-2 run, which also reports the design effect. Reversion is the share of observed label changes whose next recorded label is the origin label. Conversion is the Kaplan-Meier cumulative incidence of a first PIGD label among the 885 patients at risk (not PIGD at their first labelled visit), under a naive definition (any PIGD label) and a sustained definition (PIGD at two consecutive visits). The Markov-property test is a chi-square test, for each current label, of whether the next label is independent of the previous label.

**Results.** {{table:4}}

{{figure:2}}

{{figure:3}}

The label persists at 0.808 (TD to TD; patient-bootstrap 95% CI 0.788 to 0.826) and 0.702 (PIGD to PIGD; 0.661 to 0.738), but Indeterminate persists at only 0.192 (0.148 to 0.234): an Indeterminate visit is followed by TD at 0.440 and by PIGD at 0.368. The bootstrap intervals are wider than the Wilson intervals on every diagonal cell, with design effects of 1.38 (TD to TD) and 1.46 (PIGD to PIGD), so visit-level intervals understate the uncertainty by a factor the patient clustering explains (Figure 8).

{{figure:8}}

{{table:5}}

{{figure:5}}

Of the 673 observed label changes with an observed next visit, 35.2% reverted to the origin label at that next visit (0.352, 95% CI 0.318 to 0.388). Reversion is highest for the moves into Indeterminate (TD to Indeterminate 0.560, PIGD to Indeterminate 0.500) and lowest for the moves out of it (Indeterminate to PIGD 0.183, Indeterminate to TD 0.176); the direct moves revert at 0.320 (TD to PIGD) and 0.393 (PIGD to TD). Of the 934 patients, 476 never change label; 86 distinct label pathways occur (Table 8).

{{figure:6}}

Under the naive definition the Kaplan-Meier cumulative incidence of a first PIGD label is 0.125 by year two and 0.353 (0.308 to 0.399) by year five; under the sustained definition it is 0.065 and 0.153 (0.120 to 0.190). Requiring the conversion to hold for two consecutive visits removes 0.517 (0.457 to 0.577) of the apparent conversions: 263 first PIGD labels become 127 sustained ones among the 885 patients at risk.

{{table:7}}

{{figure:4}}

The transition probabilities depend on the elapsed interval. TD to PIGD is 0.100 (0.087 to 0.116) when the two visits are at most 1.5 years apart and 0.289 (0.167 to 0.432) when they are more than four years apart; TD to TD falls from 0.820 to 0.578 across the same strata. A discrete-time chain indexed by "next visit" would absorb PPMI's attendance schedule (annual early, biennial later) into its progression estimate, which is why the continuous-time models of Section 9.4 exist.

{{table:6}}

The observed label is not first-order Markov. Given a current TD label, the next label depends strongly on the previous one (chi-square 156.6 on 4 degrees of freedom, *p* = 7.9 × 10<sup>−33</sup>, 1,583 triples): a patient currently TD who was previously TD moves to PIGD with probability 0.086, a patient currently TD who was previously PIGD with probability 0.393. The dependence is also present for a current PIGD label (chi-square 61.2, *p* = 1.7 × 10<sup>−12</sup>, 609 triples) and, weakly, for a current Indeterminate label (chi-square 15.3, *p* = 0.004, 235 triples).

{{table:8}}

{{table:9}}

**Interpretation.** Where a patient came from changes the probability of moving to PIGD by a factor of 4.6, which is exactly the signature of a latent state read through noise: if a patient's underlying state is PIGD but one examination scores them TD, the next examination is likely to score PIGD again. That observation motivated the CT-HMM. At the same time, the 35% reversion rate, the 52% loss of apparent conversions under a sustained definition, and the near-total impermanence of the Indeterminate label (0.192) set an upper bound on how predictable the next recorded label can be from any model, because they measure how much of the record is measurement rather than state.

**Limitations.** The pathways and reversion rates are conditional on a next visit being observed; 102 of 346 direct changes in the D-1 cohort had no observed outcome (Section 9.3). The Kaplan-Meier incidence treats loss to follow-up as non-informative, which Section 9.10 shows it is not. The chi-square test ignores the clustering of triples within patients.

**Contribution.** The quantitative characterisation of label unreliability with patient-clustered uncertainty, and the first piece of evidence for the paper's claim: the instability is large, interval-dependent and history-dependent in exactly the way measurement noise on a slow latent state would be.

### 9.2 Label construction: reconstruction, OFF/ON agreement, the Indeterminate band, threshold placement and residual autocorrelation (E04 to E07)

**Question.** Is the label reconstructed exactly from the items? How much of the label's movement is attributable to the examination rather than the patient? Is the Indeterminate label a transient boundary crossing? How sensitive is the label to the placement of the thresholds, and are within-patient residual changes in the ratio independent?

**Setup.** Part C rebuilt the item sums from the raw item tables (Section 6.9); the OFF/ON audit labelled the two examinations of the same visit separately (2,343 pairs with both labels from 714 patients; 2,353 pairs for the Part III-only construction); the Indeterminate audit computed the pre-registered reversion and clustering statistics on the 934-patient cohort with 2,000 patient-bootstrap resamples; O-4 relabelled the frozen panel under the perturbed rule; E-1 computed the lag-1 autocorrelation of residuals from each patient's own linear trend on 397 patients and 2,050 pairs.

**Results: reconstruction and the lattice.** The reconstruction matched the frozen panel at all 5,774 visits (zero mismatches in the tremor score, the PIGD score and the label). The PIGD item sum is zero at 16.3% of labelled visits and the tremor sum at 7.2%; the PIGD sum is one or two at about 59% of visits. Because the ratio is a ratio of integer sums, only a lattice of values is attainable (Figure 9), and every Indeterminate visit in the cohort is one item point away from another label.

{{figure:9}}

{{figure:11}}

**Results: OFF against ON.** {{table:10}}

The same visit, labelled from the OFF examination and from the ON examination, carries the same label at 75.2% of pairs with all sixteen items and at 72.1% with the Part III items only (which removes the Part II items the two examinations share); a label built from Part III only agrees with the full label at 83.2% of visits. ON minus OFF tremor item sums have a median of −2 points (10th to 90th percentile −7 to 0) against a median of 0 for the PIGD sum. This is medication response, time of day, order and rater effects combined, not test-retest reliability, but it bounds the within-visit stability of the label at about three quarters.

**Results: the Indeterminate audit.** {{table:11}}

{{figure:12}}

An entry into Indeterminate reverts at the next visit at 0.543 (0.471 to 0.615) against 0.349 (0.298 to 0.397) for a direct TD-PIGD change, a contrast of +0.194 (+0.108 to +0.281); only 17.6% of entries are sustained at the next visit and 3 of 135 entries with two later visits are still Indeterminate at both. The observed-to-expected within-patient clustering ratio for Indeterminate is 1.18 (0.99 to 1.39), which includes one, so by the pre-registered rule the reading is inconclusive: the entry-reversion contrast is consistent with transient boundary crossing, but the clustering test does not confirm that Indeterminate visits cluster within patients beyond what their frequency implies. The Kaplan-Meier median time to leave the Indeterminate label is 1.08 years, against 2.08 years for PIGD and 5.41 years for TD.

**Results: threshold sensitivity.** {{table:12}}

{{table:13}}

{{figure:13}}

Narrowing the band to (0.91, 1.13) relabels 298 of 4,441 labelled visits (6.71%), every one of them Indeterminate (100 to TD, 198 to PIGD; the Indeterminate count falls from 441 to 143, −67.6%), in 235 of 1,080 participants (21.76%). Agreement with PPMI's own derived labels falls from 1.000 to 0.977 (two-class) and from 0.999 to 0.929 (three-class). The reason is the lattice: inside the band only eleven ratio values occur, and the two that dominate, 0.9091 (198 visits) and 1.1364 (100 visits), sit 0.009 and 0.014 from the boundaries, so a shift of 0.02 moves all of them. The recount is deterministic; the baseline panel was not touched.

**Results: lag-1 residual autocorrelation.** {{table:14}}

Consecutive residual changes from a patient's own linear trend are negatively correlated at lag 1 (−0.219, patient-bootstrap 95% CI −0.273 to −0.162), but less negatively than independent noise with the same visit pattern implies (−0.333, Monte Carlo 95% range −0.369 to −0.295): the difference is +0.113 (+0.046 to +0.179, one-sided Monte Carlo *p* = 5 × 10<sup>−4</sup>). This run also resolved a contradiction in the working record: an earlier exploration report had stated that nothing in the residual structure argued for a state-space specification, and that sentence was found to be stale against the committed code.

**Interpretation.** Taken together these audits show where the label's unreliability comes from. The examination state alone moves a quarter of same-visit labels. The Indeterminate band is a thin strip of the integer lattice whose occupants are one point from another label and whose two dominant values are within 0.02 of the thresholds; the label is therefore unusually sensitive to the rule's exact placement and the Indeterminate class behaves as a transient. The residual autocorrelation is less negative than pure measurement noise would make it, so part of a patient's deviation from their trend persists to the next visit: a state-space form, in which an unobserved level accumulates process noise and is read with error, is indicated over an independent-noise growth form, which is the family the CT-HMM and the clinical profile belong to.

**Limitations.** The OFF/ON comparison confounds medication with everything else that differs between the two examinations. The Indeterminate audit's reading rule was pre-registered but its clustering statistic has low power at 441 visits. E-1 uses only patients with at least four finite-ratio visits and one year of span, a subset selected on follow-up.

**Contribution.** Mechanistic grounding for the paper's title: the label's unreliability is a property of its construction, and the two most consequential constructions, the thin Indeterminate band on an integer lattice and the dependence on the examination state, are not things a better predictive model can remove.

### 9.3 Does a generative model of the ratio reproduce the label dynamics? The D-1 posterior-predictive check and its Stage 1 diagnostics (E08, E09)

**Question.** If the ratio follows a hierarchical growth model with Student-t slopes and residuals, do label sequences simulated from that model, conditional on each patient's own data, reproduce the observed Indeterminate share, reversion rates and clustering?

**Setup and cohort.** The 821 patients and 3,528 ratio-defined visits (PIGD score greater than zero), five whole-patient folds, 500 patient-conditional replicates per patient, five pre-registered statistics, a pre-registered threshold *c* from 500 synthetic null cohorts, and four positive-control scenarios (Section 6.8). The run is frozen: its protocol, code hashes, seeds and panel checksum are recorded, and nothing in it was re-evaluated.

**Results.** {{table:15}}

The observed Indeterminate share is 0.103 against a replicate mean of 0.065 (*z* at the cap, *p* = 0.001); the direct-reversion rate is 0.311 against 0.436 (*z* at the cap, *p* = 0.999); their difference statistic S4 is 0.268 against 0.120 (*z* at the cap). The reversion after entry (0.580 against 0.556) and the clustering ratio (1.033 against 1.077) are reproduced. *T*<sub>obs</sub> = 3.091 exceeds the threshold *c* = 2.577, so the pre-registered verdict is **not reproduced**. The check has power: the strong positive control PC4 was detected in 100 of 100 cohorts (lower 95% bound on the detection rate 0.964), the weak controls PC1 to PC3 in 6, 7 and 5 of 100 (consistent with their null-like design), and the descriptive variant PC4v in 3 of 100.

{{table:16}}

{{figure:14}}

{{figure:15}}

The Stage 1 diagnostics localise the misfit without attaching a verdict. The excess of Indeterminate visits over the generator's expectation (183 visits) is not uniform across PIGD item sums: it sits at sums 2 (50% of the excess), 3 (24%) and 4 to 5 (22%), and at sum 1 the observed and expected counts agree (70 against 71.8). Within each stratum the Indeterminate cells of the lattice hold about as many visits as their neighbouring cells, so the Indeterminate visits are ordinary lattice points, not a local concentration; the band of nominal width 0.25 captures 1.82 times its width in lattice points at sums 1 and 2, and if it captured only its nominal width the share would be about 0.068, two thirds of the way from the observed 0.103 to the generator's 0.051. The whole gap in the ratio distribution opens inside (0.90, 1.15] (observed cumulative share 0.403 against 0.340 at 1.15) and closes again above 2.

{{table:17}}

{{figure:16}}

The direct-reversion deficit is entirely direction-specific. TD to PIGD changes revert at 0.253 (0.187 to 0.325) against a generator expectation of 0.469 (37 against 83 expected reversions), whereas PIGD to TD changes revert at 0.398 against 0.396. There are also fewer direct changes than the generator implies (244 against 315 with an observed outcome). A residual-skew proxy is right-skewed in the tails and strongly denominator-dependent (interquartile range 1.20 at PIGD sums 1 to 2 against 0.31 at 8 and above), but a valid Student-t residual diagnostic was not possible from the saved outputs because the fold posteriors of the random effects were not retained. The misfit is present, and larger, in the 585 patients with no excluded PIGD-zero visit, so the exclusion is weakened as its driver.

**Interpretation.** A continuous-ratio growth model with heavy tails cannot reproduce two features of the label record: the Indeterminate share, which the integer lattice inflates inside the band, and the persistence of TD to PIGD changes, which a symmetric-noise model under-predicts. Both are features of how the label is constructed from integer item sums. The Stage 1 recommendation was that a measurement-faithful integer-score model, with no intermediate state, was the next test; that model was specified but never run on the real data (Supplementary Section S3), so the question of whether such a model would reproduce the record remains open. Nothing here is evidence for or against a biological intermediate subtype.

**Limitations.** The D-1 scope is the ratio-defined visits only, so it bears on nothing involving the 16.3% of visits at which the ratio is undefined. The Stage 1 reference is the covariate-blind generator, not the patient-conditional replicate distribution D-1 used, because the replicate labels were not saved; a protocol-compliant re-execution that would save them (M1) was specified and reviewed but not run.

**Contribution.** A pre-registered, powered negative result against the natural continuous generative model of the label, and a descriptive localisation of the misfit to the lattice and to one transition direction, which together explain why the models of Sections 9.6 to 9.11 that work on the ratio or the subscores cannot be expected to beat a model that reads the label as a noisy categorical observation.
