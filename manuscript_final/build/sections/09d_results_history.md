### 9.18 How much history does the prediction need? (E33, E34)

**Question.** If the model sees only the last one, two, three or five labelled visits instead of the whole history, how much does the prediction change, for the CT-HMM and for the clinical profile?

**Setup and cohort.** The CT-HMM (full-sequence training, leave-one-patient-out) predicts every eligible transition from truncated windows; paired comparisons are made on the 734 targets of 228 patients that every window can score. The pre-specified primary contrast is "last 2 minus full history". Twenty validation checks passed, and the tamper test found zero spurious change over 934 folds and 11,644 cases. The profile's windows (last 2, last 3, full, and full with the current-visit LEDD substituted for the next-visit LEDD) are compared on 1,489 common targets of 481 patients.

**Results.** {{table:79}}

{{figure:46}}

{{figure:47}}

{{table:80}}

{{figure:48}}

On the common targets the CT-HMM reaches 0.811 with the last visit only, 0.828 with the last two, 0.835 with the last three, 0.834 with the last five and 0.832 with the full history, against 0.827 for persistence; AUC is 0.827, 0.884, 0.890, 0.889 and 0.888. The primary contrast, last two minus full, is −0.004 (−0.019 to +0.010) in accuracy, −0.006 in balanced accuracy, −0.005 (−0.015 to +0.004) in AUC, +0.005 in Brier and +0.011 in log loss, every interval including zero. Only the single-visit window is worse: −0.022 (−0.042 to −0.001) in accuracy and −0.062 (−0.086 to −0.039) in AUC against the full history, because with one visit the model has no run length to read.

{{table:81}}

{{table:82}}

{{figure:49}}

The profile behaves the same way: last two minus full is −0.001 (−0.010 to +0.009) in accuracy and −0.003 (−0.007 to +0.002) in AUC, with a small log-loss cost (+0.010, +0.002 to +0.019); substituting the current-visit LEDD for the next-visit LEDD changes accuracy by −0.001 (−0.006 to +0.003).

**Interpretation.** Two visits carry essentially all the information the models use about the next label. The CT-HMM's filtered belief forgets quickly (the latent Indeterminate state's sojourn is two years), and the profile's level is dominated by the most recent observation. The result also disposes of a concern about the profile's medication input: knowing the next visit's LEDD is worth nothing.

**Limitations.** The common cohort for the CT-HMM windows is the 228 patients with at least six visits, a selected group with long follow-up.

**Contribution.** Evidence that the ceiling is not a shortage of history.

### 9.19 Retraining the CT-HMM on two-visit windows (E35)

**Question.** If the model is both trained and used on two-visit windows, does it do better than a model trained on full sequences and truncated at prediction time?

**Setup and cohort.** A two-by-two design crossing training (full sequences; overlapping two-visit windows; disjoint windows as a sensitivity) with prediction history (full; last two), leave-one-patient-out, paired on 2,192 common targets of 742 patients; 24 validation checks passed.

**Results.** {{table:83}}

{{figure:50}}

{{table:84}}

{{table:85}}

The pre-specified primary contrast, window-trained with the last two visits minus full-trained with the full history, is −0.008 (−0.019 to +0.003) in accuracy but −0.031 (−0.044 to −0.018) in balanced accuracy, −0.021 (−0.030 to −0.012) in AUC, +0.010 (+0.005 to +0.015) in Brier and +0.040 (+0.022 to +0.058) in log loss. Decomposed, the training effect with the history fixed at two visits is −0.025 (−0.035 to −0.015) in balanced accuracy, and the history effect with the training fixed is −0.006 (−0.015 to +0.002). The window-trained model falls below persistence in balanced accuracy (−0.016, −0.029 to −0.003) through a loss of PIGD sensitivity (0.608 against 0.706 for the full-trained model on the same targets). Disjoint windows recover some of the AUC (+0.016, +0.009 to +0.022 against overlapping windows). The reason is visible in the parameters: trained on windows, the initial Indeterminate probability falls from 0.155 to 0.044, the Indeterminate emission diagonal rises off its floor to 0.665, and the TD emission diagonal falls from 0.940 to 0.875, because a window's first visit is treated as a draw from π and the model learns to explain every window's start rather than the cohort's baseline.

**Interpretation.** Retraining on short windows is harmful, not helpful: it changes what the initial distribution means and trades PIGD sensitivity for TD specificity. The history-window result of Section 9.18 holds for the right reason, which is that a full-sequence model forgets quickly, not that short training sequences suffice.

**Limitations.** Overlapping windows duplicate observations and inflate the training likelihood; the disjoint sensitivity halves the training set.

**Contribution.** A negative result on the training regime, and an explanation of it.

### 9.20 The rich clinical profile with truncated history and two-visit retraining (E36, E37)

**Question.** Do the window and retraining results of Sections 9.18 and 9.19 hold for the clinical profile, which reads nine measurements rather than one label?

**Setup and cohort.** Experiment 6A: the profile with the last one, two, three or five visits against the full history (898 patients, 3,021 targets for the single-visit window; 740 patients and 2,184 targets for the two-visit window; smaller common cohorts for longer windows). Experiment 6B: the two-by-two design of Section 9.19 for the profile on 2,184 common targets of 740 patients, with disjoint-window arms. Five-fold patient cross-validation; 2,000 patient-bootstrap resamples; every fit converged.

**Results.** {{table:86}}

{{figure:51}}

{{figure:52}}

{{table:87}}

{{figure:53}}

The single-visit window costs the profile −0.014 (−0.022 to −0.006) in accuracy, −0.011 in AUC and −0.033 (−0.050 to −0.016) in PIGD sensitivity against the full history; the two-visit window changes accuracy by +0.005 (−0.003 to +0.012) with a PIGD-sensitivity cost of −0.018 (−0.036 to 0.000) offset by a TD-specificity gain of +0.016; the three- and five-visit windows differ from the full history by +0.006 and −0.003. The full-history profile beats persistence on balanced accuracy (+0.022, +0.012 to +0.033), Brier (−0.042) and PIGD sensitivity (+0.064, +0.045 to +0.084) but not on accuracy (+0.005, −0.003 to +0.013).

{{table:88}}

{{figure:54}}

{{table:89}}

{{figure:55}}

{{table:90}}

In the two-by-two design the conditions span 0.838 to 0.841 (persistence 0.834). The primary contrast, window-trained minus full-trained with the history fixed at two visits, is −0.003 (−0.012 to +0.005) in accuracy, −0.004 (−0.008 to −0.001) in AUC, +0.004 (+0.002 to +0.006) in Brier and +0.007 (+0.000 to +0.013) in log loss; the same training effect with the full history is −0.007 in AUC. The history effect under full-sequence training is +0.001 (−0.006 to +0.008) in accuracy with a PIGD-sensitivity cost of −0.018 (−0.036 to −0.001). Disjoint windows differ from overlapping ones by less than 0.003 on every metric. The window fits lower the observation variance (0.194 to 0.148) and raise the level innovation variance (0.103 to 0.189), the same kind of distortion the CT-HMM showed.

**Interpretation.** The profile reproduces the CT-HMM's pattern in miniature: a single visit is too little, two are enough, and retraining on windows costs a little probability quality and nothing in accuracy. The profile's one consistent advantage over persistence is PIGD sensitivity, which the window regimes erode.

**Limitations.** The common cohorts shrink with the window length; the comparisons are within a five-fold design, not LOPO.

**Contribution.** Replication of the history-window and retraining results in the richest model.

### 9.21 Patient-level accuracy (E38)

**Question.** Visit-level accuracy over-weights patients with long follow-up; how is accuracy distributed across patients, what sets it, and do the paired conclusions of Sections 9.18 to 9.20 hold at the patient level?

**Setup and cohort.** Recomputed from the saved predictions of experiments 6A and 6B (898 patients; 740 for the two-visit conditions), with per-patient accuracy, PIGD sensitivity and TD specificity, exploratory composition groups defined after prediction from recorded labels, and paired patient-level comparisons; 23 validation checks passed, including the SHA-256 manifests of the source runs.

**Results.** {{table:91}}

{{figure:56}}

{{figure:57}}

With the full history the profile's visit-level accuracy is 0.839 and its mean per-patient accuracy 0.834 (0.817 to 0.852) with a median of 1.0: 63.7% of patients (572) are predicted perfectly, 5.0% (45) never correctly, 8.2% below 50% and 70.9% at 80% or better. Persistence has almost the same distribution (0.831, 63.1% perfect).

{{table:92}}

{{figure:59}}

The distribution is set by the composition of each patient's recorded target labels. Patients whose targets are all TD are predicted at 0.944 (510 patients; persistence 0.970), all PIGD at 0.811 (162; persistence 0.733), majority TD at 0.684 (105; 0.670), majority PIGD at 0.592 (61; 0.541) and evenly split at 0.468 (60; 0.480). Per-patient PIGD sensitivity is 0.550 on average (0.722 among the 225 patients with at least two PIGD targets, where the share of patients with no correct PIGD prediction falls from 32.5% to 6.2%), and TD specificity 0.858 (0.917 among patients with at least two TD targets).

{{table:93}}

{{table:94}}

{{figure:58}}

At the patient level the paired comparisons repeat the visit-level ones: window-trained minus full-trained with two visits (6B, D against B) −0.004 (−0.016 to +0.008) with 660 of 740 patients unchanged; window-trained minus full-trained with the full history (C against A) −0.004 (−0.012 to +0.005) with 821 of 898 unchanged; the single-visit window −0.012 (−0.023 to −0.002) with 774 of 898 unchanged; and full history against persistence +0.003 (−0.008 to +0.014).

**Interpretation.** Accuracy is bimodal across patients, and the mode a patient falls in is determined by whether their recorded labels change. A model is right about a patient who never changes label and wrong about half of a patient's visits when the labels alternate; that is the label's reliability expressed one patient at a time, and the models move at most a few per cent of patients in either direction relative to persistence.

**Limitations.** The composition groups are defined from the outcomes and are exploratory; nothing about severity or progression can be read from them.

**Contribution.** The patient-level view of the ceiling.

### 9.22 Adding the ratio band to the CT-HMM's observation (E39)

**Question.** Does telling the CT-HMM where inside the subtype a visit sits (the ratio band) sharpen its belief about the latent state and improve prediction?

**Setup and cohort.** The ratio-banded CT-HMM of Section 6.13 at band width 0.10 (44 categories, 137 parameters, ten tail bands merged from training data) against the subtype-only CT-HMM (14 parameters) and a control arm, on the 3,029 eligible transitions under leave-one-patient-out; accuracy, Brier, log loss, calibration, AUC-ROC, average precision, partial AUC at false-positive rate at most 0.2, and an emission diagnostic on the full cohort. All 934 folds of each arm converged; the control arm reproduced the baseline exactly (zero discordant cases, differences at 10<sup>−6</sup>).

**Results.** {{table:95}}

{{figure:60}}

{{table:96}}

{{figure:61}}

The banded model reaches 0.832 (0.814 to 0.850) against 0.837 for the subtype-only model: −0.005 (−0.013 to +0.003), with the banded model right and the baseline wrong on 67 cases against 83 the other way. Brier is 0.122 against 0.121, log loss 0.393 against 0.396, and the calibration gap 0.028 against 0.0065. By current ratio range the banded model loses 3.2 points on visits with a ratio in [1.15, 2.00) and gains at most one point elsewhere.

{{table:97}}

{{figure:62}}

{{table:98}}

On ranking the picture reverses: AUC-ROC rises from 0.859 (0.840 to 0.876) to 0.876 (0.859 to 0.893), a paired gain of +0.017 (+0.009 to +0.026), while average precision falls by −0.010 (−0.032 to +0.013) and the partial AUC in the operating region (false-positive rate at most 0.2) by −0.006 (−0.016 to +0.004); the banded model calls 890 cases PIGD against 800, moving its operating point from (0.667, 0.090) to (0.708, 0.115) in true- and false-positive rate. The AUC gain is concentrated in the current-TD stratum (0.669 to 0.750) and among cases with one prior visit (0.767 to 0.836); it is absent from the Indeterminate and PIGD strata.

{{table:99}}

The emission diagnostic shows why. The subtype marginal of the banded emission is close to the baseline's (the Indeterminate row still sits at the floor), and within each subtype the band distributions of the three latent states differ by total-variation distances of up to 0.66 (TD), 0.50 (Indeterminate) and 0.63 (PIGD): the band does carry information about the latent state, but what it carries re-orders the cases that are already on the TD side of the boundary rather than moving cases across it.

**Interpretation.** The band sharpens the latent belief in exactly the sense the design allowed (through α only), and the result is a better ranking of TD-side cases, more PIGD calls, worse calibration and no accuracy. This is the ratio-level counterpart of the profile's AUC gain in Section 9.15, obtained inside one model family with one latent process, so the pattern cannot be attributed to a change of model class.

**Limitations.** Ten times the parameters on the same 4,295 observations; the tail bands are merged on training data within each fold; the AUC gain in the one-prior-visit stratum is where the subtype-only model has least to read.

**Contribution.** A within-family demonstration that richer observations improve ranking but not accuracy or calibration.

### 9.23 The band-width sweep (E40)

**Question.** Is width 0.10 a poor choice? Does a coarser or finer band change the conclusion of Section 9.22?

**Setup and cohort.** The banded CT-HMM at main-range widths 0.05, 0.10, 0.20, 0.30 and 0.50 (73, 44, 29, 24 and 20 categories; 224, 137, 92, 77 and 65 parameters) against the subtype-only model, on the same 3,029 transitions, leave-one-patient-out (934 folds per arm, all converged), with pre-specified ranking contrasts against width 0.10 and secondary contrasts against the unbanded model.

**Results.** {{table:100}}

{{figure:63}}

{{table:101}}

{{figure:64}}

{{figure:65}}

{{table:102}}

{{table:103}}

Accuracy lies between 0.832 and 0.836 at every width against 0.837 unbanded, and every accuracy contrast includes zero. AUC-ROC is 0.874, 0.876, 0.877, 0.878 and 0.876 at widths 0.05 to 0.50 against 0.859 unbanded; average precision 0.733, 0.740, 0.754, 0.752 and 0.741 against 0.750; partial AUC 0.780, 0.782, 0.787, 0.789 and 0.783 against 0.788. Against width 0.10, width 0.30 is better on all three ranking metrics by intervals that exclude zero (AUC +0.003, +0.000 to +0.005; average precision +0.012, +0.003 to +0.022; partial AUC +0.007, +0.002 to +0.012), width 0.20 on average precision and partial AUC, width 0.05 is worse on average precision, and width 0.50 differs on nothing. Against the unbanded model every width gains AUC-ROC (+0.015 to +0.020) and none gains average precision or partial AUC (width 0.30: +0.003, −0.017 to +0.021 and +0.001, −0.007 to +0.009). Brier and log loss are best at width 0.30 (0.120, 0.387), and the calibration gap is worst at 0.05 and 0.50 (0.025, 0.029) and best at 0.30 (0.022), all above the unbanded 0.0065.

**Interpretation.** The band width has an interior optimum near 0.30, where the alphabet is coarse enough that every category holds at least eight training observations, and the optimum is not worth having: it is a few thousandths of AUC over width 0.10 and no improvement over the unbanded model on the two ranking metrics that weight the operating region. Width 0.10 was not a poor choice, and no width turns the ranking gain into an accuracy or calibration gain.

**Limitations.** Five widths; the tail structure is common to all arms; the sweep is a sensitivity analysis and was not pre-registered as a selection.

**Contribution.** Closure of the ratio-band line: the conclusion of Section 9.22 is insensitive to the band width.
