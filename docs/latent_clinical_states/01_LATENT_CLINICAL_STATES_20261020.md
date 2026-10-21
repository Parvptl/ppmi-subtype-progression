# Latent clinical states from multiple measurements, against the subtype-history CT-HMM

**Date** 2026-10-20
**Code** `src/ppmi_subtype/clinical_states.py`, `scripts/61_clinical_state_selection.py`,
`scripts/62_clinical_state_lopo.py`, `tests/test_clinical_states.py`
**Runs** `experiments/20261020T142040Z_clinical_state_selection/` (state count),
`experiments/20261020T142446Z_clinical_state_lopo_k6/` (primary LOPO),
`experiments/20261020T145228Z_clinical_state_lopo_k8/` (sensitivity)
**Baseline** `experiments/20261020T104756Z_next_visit_td_pigd_prediction/`
**Panel** `outputs/visit_panel.csv`, md5 `fb038ddbdc19ab3368e13940f398e7ba`

---

## 1. The question

> Does representing Parkinson's disease using multiple clinical measurements and
> learned latent states provide more useful information about next-visit
> progression than using the patient's TD / Indeterminate / PIGD history alone?

This follows the previous experiment, which predicted the next observed
TD-vs-PIGD label from the label history alone and found 83.7% accuracy —
statistically indistinguishable from simply repeating the last
non-Indeterminate label. The obvious next question is whether a richer
clinical representation, in the spirit of Severson et al. (2021), carries
information the label history throws away.

*(Results and the answer are in §6–§9.)*

---

## 2. What the existing code could and could not do

Verified against the code, not assumed:

| Claim | Verdict |
|---|---|
| `hmm.py` is a CT-HMM | **Yes** — latent state under a generator `Q` from `ctmc.build_generator`, propagated by `exp(Q·dt)`, with an explicit emission matrix. |
| Its emission can take continuous measurements | **No.** `E` is a `(n_hidden, n_observed)` softmax indexed by an integer label (`E[:, labels[:, 0]]` in `hmm._forward_loglik`). There is no path for a ten-dimensional vector without replacing the emission outright. |
| `ctmc.py` can be reused | **Yes** — `build_generator` and `transition_probabilities` are exactly the latent-process machinery needed, and are reused unchanged. |
| The previous experiment's two-class logic can be reused | **Yes** — `next_visit.conditional_two_class`, `predicted_state`, `two_class_metrics`, `paired_accuracy_difference` and `accuracy_breakdowns` are all reused, so any difference in the headline numbers is a difference in the models and not in the scoring. |

So the decision was: **keep the transition process, replace the emission.**
`clinical_states.CTHMMGaussian` is a continuous-time latent chain with a
diagonal multivariate Gaussian emission per state, fitted by EM. No HMM
library was available (`hmmlearn`, `pomegranate` both absent), and the project's
own idiom is hand-written `scipy.optimize`, so it is implemented directly.

---

## 3. The clinical representation

Ten measurements, each an established PPMI scale or a subscore built from its
items — not a bank of engineered features.

| Feature | Domain | Source | Coverage |
|---|---|---|---|
| `brady_rigid` | motor: bradykinesia/rigidity | **new**: mean of 20 Part III items | 0.9998 |
| `tremor` | motor: tremor | project's 11-item tremor score | 1.0000 |
| `pigd` | motor: gait/postural | project's 5-item PIGD score | 1.0000 |
| `updrs2` | function: motor ADL | MDS-UPDRS Part II total | 0.9980 |
| `updrs1` | non-motor burden | MDS-UPDRS Part I total | 0.9968 |
| `moca` | cognition | MoCA total (higher is better) | 0.9928 |
| `gds` | mood | Geriatric Depression Scale-15 | 0.9946 |
| `scopa` | autonomic | SCOPA-AUT total | 0.9901 |
| `ess` | sleep | Epworth Sleepiness Scale | 0.9971 |
| `ledd` | treatment | log1p levodopa-equivalent dose | 1.0000 |

**Why a new bradykinesia subscore.** `updrs3_score` is the Part III *total*,
which includes the tremor and PIGD items. Adding it to a feature set that
already carries `tremor` and `pigd` would count those items twice. Instead the
20 Part III items that are neither tremor nor PIGD constituents — speech,
facial expression, the five rigidity items, finger tapping, hand movements,
pronation-supination, toe tapping, leg agility, arising, posture and global
spontaneity — form a distinct motor axis. It is built with the project's own
`select_medication_state`, `deduplicate_visits` and `clean_items`, so the
OFF-exam selection, duplicate resolution and the PPMI "Unable to Rate"
sentinel-101 handling are identical to the tremor and PIGD scores it sits
beside. `tests/test_clinical_states.py` asserts the item sets do not overlap.

**Variables deliberately excluded** for inadequate longitudinal coverage:
`upsit` (0.28), `PDAQ27` (0.13), `CGI`/`PGI` (0.32), `updrs4_score` (0.57).
Administrative identifiers, death/status fields and anything not available at
the prediction time are excluded by construction — the feature builder reads
an explicit allow-list, not the whole panel.

**Missing measurements are marginalised, not imputed.** With a diagonal
covariance the emission log-density is a sum over features, so an unrecorded
feature is dropped from that sum — exactly, not approximately. **No imputer is
fitted anywhere**, which removes that leakage surface entirely. A visit with no
features at all contributes a flat likelihood, which is the correct statement
that it carries no information about the state. This is verified against a
hand-computed sub-vector density in
`test_emission_marginalises_missing_features_exactly`.

**No dimensionality reduction.** Ten features with pairwise correlations
peaking at 0.65 (`pigd` against `updrs2`) are not redundant, and a projection
would cost the per-feature state profiles that the interpretation depends on.
A contrastive or learned representation was not added: there is nothing here
for it to buy, and it would only be imitating the paper.

**Cohort.** 4,433 clinical visits over 1,080 patients; 934 patients have two or
more usable visits, giving **3,353 prediction opportunities** — against the
baseline's 3,361. A visit is dropped only if it carries fewer than 8 of the 10
features (97.6% carry all ten).

---

## 4. The model

Latent state `Z(t)` evolves as a continuous-time Markov chain with generator
`Q`; the observation at visit `k` is

    x_k | Z = s  ~  N(mu_s, diag(var_s))

Fitted by EM:

* **E-step** — forward–backward with per-interval matrices `exp(Q·dt)`. Run
  over all patients at once (`batched_estep`); a test asserts it reproduces the
  per-sequence reference loop's log-likelihood and every sufficient statistic
  to floating point, including under missing data.
* **Emission M-step** — closed form: soft-weighted means and variances over the
  visits where each feature was recorded, with a variance floor to prevent the
  classic Gaussian collapse onto a single visit.
* **Generator M-step** — numerical, because interval censoring admits no closed
  form for `Q`. It maximises the soft-count-weighted CTMC panel likelihood, the
  same objective and parameterisation as `ctmc.fit_ctmc` with expected rather
  than observed counts. It is capped at 15 L-BFGS-B iterations per EM step
  (**generalised EM**, warm-started), which is what makes the run affordable —
  see §5.

Smoothing is used for *fitting* on training patients, which is correct.
**Prediction uses filtering only**: `filter_forward` runs no backward pass, and
the prediction for visit `k+1` is `alpha[k] @ exp(Q·dt)`. Two tests enforce
this — one that tampering a later visit cannot move the filtered distribution
at an earlier one, and an end-to-end one that corrupting a visit leaves the
prediction *for* that visit bit-identical while moving the prediction made
*from* it.

**Linking latent states to the observed label.** A latent clinical state says
nothing about the recorded TD/PIGD label without an observation model, so one is
estimated on training patients only:

    B[s, l]  ∝  sum over training visits of  gamma_s(visit) · 1[label(visit) = l]

row-normalised with Laplace smoothing. The next observed-label distribution is
then `P(next state) @ B`, and `P_TD`/`P_PIGD` come from the *previous
experiment's own* `conditional_two_class`, with its denominator guard and its
documented PIGD tie rule. The link is justified rather than assumed: `tremor`
and `pigd` are two of the ten features, so a state that separates them does
carry label information — but the label is a thresholded *ratio* of those two
scores while a state is a distribution over all ten, so the mapping is genuinely
probabilistic and not a relabelling.

**Latent state indices are arbitrary.** `canonical_order` sorts states by one
documented scalar — the sum of standardised means with each feature signed so
higher is worse — so fits from different folds can be compared. That is a
reporting convenience, not a claim that the states lie on one axis, and
certainly not a claim that they are biological disease stages.

---

## 5. Choosing the number of states, and the compute ceiling

**Severson et al. found eight states. That is a result about their cohort,
variables and model, and was not used as a target.**

Patients were split once by PATNO into a **30% development set (280 patients)**
where the state count was chosen, and a **70% evaluation set (654 patients)**
that the selection script never touches. Script 62 reproduces the split from
the same seed and asserts both that the two are disjoint and that the
reproduced development set matches the one script 61 wrote.

The criterion is the **next-visit predictive log-density under filtering**,
averaged over an inner 4-fold patient-level CV of the development set — the
quantity the model is actually for. BIC is reported but not used: it treats a
patient's repeated visits as independent observations, which inflates its
sample size.

| States | held-out log-density | SE | per feature | train BIC | max start spread | params |
|---|---|---|---|---|---|---|
| 2 | −12.668 | 0.214 | −1.268 | 24,838 | 0.1 | 43 |
| 3 | −12.252 | 0.289 | −1.227 | 23,840 | 7.8 | 68 |
| 4 | −11.455 | 0.481 | −1.147 | 21,582 | 1,294.6 | 95 |
| 5 | −11.178 | 0.309 | −1.119 | 20,729 | 1,391.8 | 124 |
| 6 | **−10.988** | 0.214 | −1.100 | 20,327 | 1,419.4 | 155 |
| 8 | −10.724 | 0.286 | −1.074 | 20,204 | 1,259.5 | 223 |

**The criterion never turns over.** It is still improving at the top of the
range, and the arithmetic maximum and BIC both pick 8 — the boundary. A
preferred count that sits at the edge of the tested range is evidence about the
range, not about the data. A pre-registered **one-standard-error rule** (the
smallest count within one inner-fold SE of the best mean) therefore selects
**6 states**; 6 and 8 are the counts inside that band, and the gap between them
(0.26 nats against an SE of 0.29) is within fold noise. Because that call is
genuinely borderline, **both 6 and 8 are evaluated** in §6 and the headline
conclusion is checked against both.

**The range stops at 8 for a measured reason, not an opinion.** One LOPO fold
on 933 training patients costs, at 3 EM starts:

| States | EM, 1 start | full 654-fold LOPO, 16 workers |
|---|---|---|
| 4 | 3.0 s | 0.10 h |
| 6 | 16.6 s | 0.57 h |
| 8 | 50.6 s | 1.72 h |

Counts above 8 cannot be LOPO-evaluated the way this experiment evaluates, so
sweeping them would only produce a preferred count that no held-out result
could ever be reported for.

Those timings are *after* two optimisations, without which the selected model
was not evaluable at all. Profiling showed the generator M-step cost 196× the
E-step at 8 states and was running to its own iteration cap every time;
capping it at 15 warm-started iterations (generalised EM) gave a **5.6×**
speedup at a cost of 0.5 nats out of 40,739 — 0.001% of the log-likelihood.

**EM is not stably identified at these state counts.** The spread in final
log-likelihood across 8 dispersed starts exceeds 1,000 nats at 4, 5 and 6
states, and the best start at the chosen count was the eighth of eight. This is
reported rather than smoothed over, because it bounds how far the *states
themselves* can be interpreted — see §9.

---

*(§6 results, §7 state descriptions, §8 the answer and §9 limitations follow.)*

## 6. Results

654 evaluation patients, 654 folds, **0 failed**, EM converged in 654/654.
2,311 prediction cases; 2,310 after dropping 2 cases (1 clinical, 1
persistence) that the baseline run had no matching visit pair for — reported in
`cases_dropped_for_pairing.csv`, not silently discarded. **2,091 eligible**
(next label TD or PIGD) scored; 219 Indeterminate next visits reported
separately and never scored.

All five models are scored on the identical 2,091 cases.

| Model | Accuracy | 95% CI | Brier | Log loss | PIGD recall | PIGD F1 |
|---|---|---|---|---|---|---|
| Subtype-history CT-HMM (labels only) | **0.8384** | [0.8183, 0.8568] | 0.1211 | 0.3953 | 0.659 | 0.701 |
| Persistence (last non-Indeterminate) | 0.8350 | [0.8147, 0.8541] | 0.1590 | 4.1729 † | 0.609 | 0.680 |
| Direct logistic, 10 clinical measurements | 0.8307 | [0.8086, 0.8515] | **0.1156** | **0.3646** | 0.637 | 0.684 |
| Direct logistic, tremor + PIGD scores only | 0.8297 | [0.8080, 0.8495] | 0.1202 | 0.3819 | 0.619 | 0.676 |
| **Clinical-state CT-HMM (latent, 10 measurements)** | **0.7255** | [0.6960, 0.7534] | 0.1831 | 0.5493 | **0.218** | 0.313 |

† rule-based probabilities; log loss finite only through clipping.

**The latent clinical-state model is markedly worse, not better.** It loses
11.3 pp of accuracy against the subtype-history baseline, and its PIGD recall
collapses from 0.659 to **0.218** — it recovers barely a fifth of actual PIGD
conversions.

### Paired contrasts (identical cases, patients resampled)

| Contrast | Δ accuracy | 95% CI | Δ Brier |
|---|---|---|---|
| Latent clinical states − subtype-history baseline | **−0.1130** | [−0.1438, −0.0807] | +0.0620 |
| Latent clinical states − persistence floor | −0.1095 | [−0.1384, −0.0806] | +0.0241 |
| **Same measurements, no latent bottleneck** | **+0.1052** | [+0.0756, +0.1338] | −0.0675 |
| 10 measurements − subtype-history baseline | −0.0077 | [−0.0204, +0.0046] | −0.0055 |
| Only tremor+PIGD − all ten measurements | −0.0010 | [−0.0114, +0.0091] | +0.0046 |

---

## 6a. Sensitivity to the state count: 8 states

The selection rule's choice between 6 and 8 states was within one standard
error (§5), so the whole LOPO evaluation was repeated at **8 states** — the
count the arithmetic maximum and BIC both preferred. 654 folds, 0 failed, EM
converged 654/654, all 15 checks pass, identical 2,091 scored cases.

| | 6 states (primary) | 8 states (sensitivity) |
|---|---|---|
| Accuracy | 0.7255 [0.6960, 0.7534] | 0.7336 [0.7062, 0.7607] |
| Brier | 0.1831 | 0.1817 |
| Log loss | 0.5493 | 0.5444 |
| PIGD recall | 0.218 | 0.233 |
| PIGD F1 | 0.313 | 0.335 |
| vs baseline, Δ accuracy | **−0.1130** [−0.1438, −0.0807] | **−0.1048** [−0.1357, −0.0757] |
| vs persistence, Δ accuracy | −0.1095 [−0.1384, −0.0806] | −0.1014 [−0.1285, −0.0738] |
| accuracy, current label PIGD | 0.3969 | 0.4372 |
| mean next-visit clinical log-density | −10.76 | −10.32 |

**The conclusion does not change.** Two more states improve the clinical
density fit (−10.32 against −10.76, as the selection criterion predicted) and
move next-label accuracy by +0.8 pp — nowhere near closing an 10–11 pp gap.
PIGD recall rises only from 0.218 to 0.233, and the collapse on
currently-PIGD patients persists (0.437 against the baseline's 0.787). The
deficit is structural, not a matter of having too few states: more states buy a
better description of the clinical measurements and almost nothing for the
TD/PIGD label, which is exactly what §7 predicts if the state axis is the wrong
one.

The start spread was lower at 8 states than at 6 (median 228 against 399 nats,
max 1,100 against 6,348), so the instability in §9 is not monotone in the state
count either.

---

## 7. Why it failed, and what that means

The breakdown by the label recorded at the prediction point localises the
failure exactly:

| Model | current TD (n=1469) | current Indeterminate (n=176) | current PIGD (n=446) |
|---|---|---|---|
| Subtype-history baseline | 0.8809 | 0.6136 | 0.7870 |
| Direct, 10 measurements | 0.8788 | 0.5739 | 0.7735 |
| Persistence | 0.8788 | 0.6080 | 0.7803 |
| **Latent clinical states** | 0.8441 | 0.5682 | **0.3969** |

The latent model is roughly competitive when the current label is TD and
collapses when it is PIGD. The reason is visible in how willing each model is
to say PIGD at all:

| Model | says PIGD | actual PIGD rate | mean P_PIGD |
|---|---|---|---|
| Latent clinical states | **0.1124** | 0.2874 | 0.3075 |
| Direct, 10 measurements | 0.2482 | 0.2874 | 0.2973 |
| Subtype-history baseline | 0.2530 | 0.2874 | 0.2950 |

For a patient currently labelled PIGD the next label is PIGD 78.0% of the time,
but the latent model predicts PIGD for only 24.4% of them, with a mean P_PIGD
of 0.394 — barely above the 0.279 it gives currently-TD patients. It hardly
distinguishes them.

**The mechanism is the state axis.** The six learned states are ordered by
*overall severity*, and the recorded label tracks that ordering only loosely:

| State | brady_rigid | tremor | pigd | updrs2 | moca | share of visits | share TD | share PIGD |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.66 | 0.61 | 0.14 | 1.8 | 27.7 | 0.054 | 0.917 | 0.028 |
| 1 | 0.86 | 0.59 | 0.26 | 6.6 | 27.2 | 0.294 | 0.701 | 0.188 |
| 2 | 0.75 | 0.49 | 0.19 | 4.6 | 27.5 | 0.168 | 0.689 | 0.216 |
| 3 | 1.24 | 0.79 | 0.33 | 6.5 | 27.2 | 0.180 | 0.698 | 0.223 |
| 4 | 1.30 | 0.67 | 0.58 | 11.4 | 26.8 | 0.216 | 0.453 | 0.390 |
| 5 | 1.77 | 0.72 | 1.34 | 20.5 | 24.8 | 0.088 | 0.222 | 0.684 |

Four of the six states — **69.6% of all visits** — emit 69–92% TD. A
PIGD-labelled patient of mild or moderate severity lands in state 1, 2 or 3,
whose label emission is dominated by TD, so the model predicts TD.

The underlying reason is that **the TD/PIGD label is a ratio**
(tremor score ÷ PIGD score) and is therefore scale-invariant, while the latent
states are built from the *levels* of ten correlated severity measures. Across
the six states `tremor` barely moves (0.49–0.79) while `pigd` rises tenfold
(0.14 → 1.34) as part of general progression. The states track severity; the
label needs the balance between two subscores, which severity does not
determine.

**This is a property of the architecture, not of the data.** The decisive
evidence is the fourth contrast: a direct logistic regression on the *same ten
measurements*, with no latent layer, scores 0.8307 — **+10.5 pp over the latent
model** (CI [+7.6, +13.4]) — and is statistically indistinguishable from the
subtype-history baseline (−0.77 pp, CI [−2.0, +0.5]) while achieving the best
Brier score (0.1156 vs 0.1211) and best log loss (0.3646 vs 0.3953) of any
model here. The information was present in the measurements; compressing them
into six discrete states destroyed it.

**And the extra eight measurements add nothing for this target.** A direct
model on `tremor` and `pigd` alone scores 0.8297, indistinguishable from all
ten (−0.10 pp, CI [−1.1, +0.9]). Cognition, mood, autonomic, sleep, ADL and
treatment dose carry no additional information about the next TD-vs-PIGD label
once the two label-defining scores are in hand.

### The latent representation on its own terms

The states are *not* a failed density model. Mean next-visit clinical
log-density is −10.76; state assignment is confident (mean max posterior
0.938, 81% of visits above 0.9, mean entropy 0.151 against a 1.79 maximum);
and the annual transition matrix is strongly persistent and close to
progressive, with self-transition probabilities of 0.41–0.95 and the most
severe state near-absorbing (0.948):

| from \ to | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0 | 0.806 | 0.049 | 0.022 | 0.109 | 0.014 | 0.001 |
| 1 | 0.001 | 0.409 | 0.255 | 0.166 | 0.159 | 0.011 |
| 2 | 0.001 | 0.010 | 0.766 | 0.041 | 0.171 | 0.012 |
| 3 | 0.004 | 0.002 | 0.008 | 0.865 | 0.114 | 0.008 |
| 4 | 0.000 | 0.019 | 0.005 | 0.010 | 0.851 | 0.116 |
| 5 | 0.000 | 0.001 | 0.000 | 0.001 | 0.050 | 0.948 |

So the model does find persistent, orderable clinical configurations. They are
simply the wrong axis for this prediction target. Predicted-next-state
confidence averages 0.590 and internal consistency with the state filtered at
the next visit is 0.441 — consistent with a process that moves slowly but is
not sharply predictable one visit ahead.

**The states are given no clinical names.** The ordering is by a documented
burden scalar, and while state 5 has the highest scores on every scale and
state 0 the lowest, calling them "early" and "advanced disease" would assert a
staging interpretation that this evidence does not support — particularly given
the instability in §5 and §9.

---

## 8. Answer to the research question

> Does representing Parkinson's disease using multiple clinical measurements and
> learned latent states provide more useful information about next-visit
> progression than using the TD / Indeterminate / PIGD history alone?

**No — and the two halves of that question have different answers.**

**The multiple clinical measurements: no additional information, but better
probabilities.** A direct model on the ten measurements matches the
subtype-history baseline on accuracy (−0.77 pp, CI spanning zero) and is the
best-calibrated model in the comparison (Brier 0.1156 against 0.1211; log loss
0.3646 against 0.3953). So ten clinical scales buy a modest improvement in
*probability quality* and nothing in classification. Worse, the gain is not
from richness: `tremor` and `pigd` alone do just as well, so the eight other
measurements are not contributing to this target at all.

**The learned latent states: actively harmful.** −11.3 pp against the
subtype-history baseline, −11.0 pp against a persistence rule, and PIGD recall
of 0.218 against 0.659. The same measurements without the latent layer are
+10.5 pp better, so the deficit is the bottleneck, not the data.

**Why, in one sentence:** the latent states organise patients by overall
disease severity, which is close to orthogonal to the tremor-versus-PIGD ratio
the label is defined from, so discretising onto a severity axis discards the
very quantity being predicted.

This is consistent with the project's earlier finding that subtype clustering
is inconclusive (`docs/cluster_phase/`, "Close the Stage 6 clustering
investigation as inconclusive") and with the previous experiment's finding that
the next label is predictable mainly because the subtype is persistent. Taken
together: TD/PIGD subtype is a persistent *label* that is not well explained by
the latent structure of overall clinical progression.

**What this does not establish.** It does not show that latent clinical states
are useless — they are a reasonable, persistent, confidently-assigned
description of progression, and this experiment scored them against one narrow
target they turn out not to serve. Severson et al. used their states to
describe progression trajectories, not to predict a motor-subtype ratio, and
nothing here contradicts their result.

---

## 9. Limitations

1. **The state count is not identified by the data.** The predictive criterion
   never turns over within the evaluable range, and the arithmetic maximum and
   BIC both select the boundary. The reported model uses the pre-registered
   one-standard-error rule (6 states); 6 and 8 differ by less than one SE.
   **Both were evaluated (§6a) and the conclusion is identical**, so the
   borderline call does not drive any result here.
2. **EM is not stably identified.** Across 3 starts per fold the median spread
   in final log-likelihood was 399 nats and the maximum 6,348; in the
   development sweep with 8 starts it exceeded 1,000 nats at 4, 5 and 6 states.
   The *states themselves* should therefore be read as one of several
   comparable optima, not as a unique decomposition. This is a real constraint
   on interpreting them, and it is why no state is given a clinical name.
3. **State counts above 8 were never tested**, because LOPO at 8 states already
   costs 1.7h and the cost grows steeply. A better-fitting model may exist
   beyond the range that could be evaluated.
4. **The evaluation cohort is 654 of 934 patients.** The other 280 were spent
   on choosing the state count and are excluded from every result, which is the
   price of a clean selection. The baseline is re-scored on the same 654
   patients and the same visit pairs, so the comparison is paired, but the
   absolute numbers are not directly comparable to the previous experiment's
   899-patient figures.
5. **`P_TD`/`P_PIGD` remain conditional** on the next label being TD or PIGD,
   exactly as in the previous experiment, and 219 Indeterminate next visits are
   reported but never scored. The latent model does raise `p_Indeterminate` on
   those cases, but they are excluded from the headline either way.
6. **Elapsed time to the next visit is an input** to all interval-aware models,
   including this one. PPMI's schedule is not independent of disease course.
7. **`ledd` is a treatment, not a symptom.** Including it means the latent
   states partly encode treatment intensity; Severson et al. likewise included
   medication state. It is one of the eight features shown to add nothing here.
8. **The target is the project's own classification rule**, not an independent
   clinical diagnosis, so accuracy is agreement with a noisy record.
9. **The direct models fill the <3% missing feature values with the training
   fold's median**, because `fit_direct` cannot read NaN. The latent model needs
   no such fill — it marginalises — so on that one point the two are not
   treated identically.
