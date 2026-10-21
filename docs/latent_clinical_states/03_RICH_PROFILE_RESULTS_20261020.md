# Rich longitudinal clinical profiles: results

**Date** 2026-10-20 · **Run** `experiments/20261020T180515Z_clinical_profile_cv/`
**Design and pre-registered hypotheses** `02_RICH_PROFILE_DESIGN_20261020.md`
**Code** `src/ppmi_subtype/clinical_profile.py`, `scripts/64_clinical_profile_cv.py`,
`tests/test_clinical_profile.py` (49 tests)
**Panel** `outputs/visit_panel.csv`, md5 `fb038ddbdc19ab3368e13940f398e7ba`
**Validation** 5-fold patient-level CV × 4 repeats · 60 model fits · 0 failed · 13/13 checks pass

---

## For a reader joining here

Each Parkinson's patient in PPMI is conventionally summarised by one of three
motor-subtype labels — TD (tremor dominant), Indeterminate, PIGD (postural
instability / gait difficulty). That label is a thresholded ratio of two
symptom scores, so two patients can share it while differing in gait,
cognition, mood, autonomic function and daily function.

This experiment replaces the label with a **richer representation**: for each
patient, in each of nine clinical domains, a *level* (where they are) and a
*rate of change* (how fast they are moving), filtered from their own visit
history, with levodopa dose entered as an input rather than as a domain. The
primary question is whether that representation predicts the patient's **next
visit's actual measurements** better than simpler alternatives, and whether it
carries clinical information the label does not.

The idea comes from Severson et al. 2021 (*Lancet Digital Health*), which built
disease states from many clinical measurements rather than a label. **This is
not a reimplementation of that paper** — see §1.

---

## 1. Relationship to the source paper

I could not retrieve the Lancet paper's full text (HTTP 403 on ScienceDirect,
thelancet.com HTML and PDF, Europe PMC and PubMed; OpenAlex confirms it is gold
OA but that no repository holds the full text). I read its **abstract**
verbatim, plus the two method papers by the same authors in full: the
**PIOHMM** (Severson et al., MLHC 2020, PMLR 126:309–330) and the
**contrastive latent variable model** (Severson, Ghosh & Ng, AAAI 2019,
arXiv:1811.06094). Details specific to the Lancet paper's own Methods are not
asserted anywhere here.

**What I took from it:** many clinical axes instead of one label; a
patient-specific component separated from the shared structure; medication as
an *input* rather than a feature; heterogeneous rates of progression; and
clinical relevance judged against outcomes not used in learning.

**What I deliberately did not reproduce,** and why — the paper's pipeline ends
in K discrete states, and the previous experiment in this repository
established that the discretisation itself was what destroyed the information
(priced at 10.5 accuracy points on identical measurements). A discrete state is
also a poor prediction target: it is model-generated, so it moves when the
model moves. So the contrastive step and the PIOHMM are not implemented; the
representation here stays continuous. Healthy controls *are* available in the
raw data (330 patients, 2,169 visits), so the contrastive step was declined on
these grounds rather than for want of data.

---

## 2. What was predicted

**Primary target — the next visit's 9-domain clinical profile**, as a
probability distribution scored against the measurements actually recorded:
`brady_rigid, tremor, pigd, updrs2, updrs1, moca, gds, scopa, ess`. A measured
target, so it does not move when the model changes, and every candidate is
scored on the same held-out numbers.

**Not the target:** the next latent state.

**Secondary — the next TD/Indeterminate/PIGD label**, derived from the
predicted tremor and PIGD distributions by this project's own unmodified
`subtype.classify`, which I verified reproduces the recorded panel label on
**4,433 of 4,433** visits. A predicted *distribution* over the two scores
induces a distribution over the label by Monte Carlo through that rule — no new
classifier, no new threshold, and a prediction near the 0.90/1.15 ratio
boundary correctly yields an uncertain label.

**Cohort:** 934 patients with ≥2 usable visits, 3,353 prediction opportunities.
LEDD is the medication input, not a predicted domain — the single most
consequential change from the previous experiment, motivated by the audit
finding that 790 of 1,080 patients begin levodopa during follow-up and 98.7%
are untreated at baseline.

---

## 3. H1 — Profile prediction quality (primary target)

| Model | log-density per domain-obs | 95% CI | RMSE (std) | MAE (std) | 95% coverage |
|---|---|---|---|---|---|
| **Profile: level + rate + medication** | **−1.0453** | [−1.0706, −1.0215] | **0.7007** | **0.5011** | 0.941 |
| Profile: level only (random walk) | −1.0492 | — | 0.7025 | 0.5012 | 0.941 |
| Profile: level + rate, no medication | −1.0511 | — | 0.7055 | 0.5018 | 0.940 |
| Last observation carried forward | −1.1187 | [−1.1434, −1.0946] | 0.7426 | 0.5212 | 0.944 |
| Population mean trajectory | −1.4342 | [−1.4693, −1.4021] | 1.0129 | 0.7749 | 0.945 |

Paired contrasts (per-prediction mean log-density, identical held-out visits,
patient-level bootstrap):

| Contrast | Δ | 95% CI | share better |
|---|---|---|---|
| Rich profile − assume-no-change | **+0.0733** | [+0.0657, +0.0813] | 75.1% |
| Rich profile − population average | **+0.3889** | [+0.3628, +0.4167] | 86.6% |
| Modelling the rate of change | **+0.0040** | [+0.0019, +0.0062] | 65.2% |
| Medication as an input | **+0.0058** | [+0.0041, +0.0075] | 62.4% |

**H1 passes.** The representation beats both pre-registered baselines with
intervals excluding zero. But the decomposition matters more than the verdict:
of the +0.0733 gain over "assume no change", the rate component supplies
**+0.0040 — about 5%** — and medication +0.0058. Almost all of the gain is the
*level* being filtered rather than read off the last visit, i.e. shrinkage and
noise-averaging across a patient's history, not progression modelling.

Calibration is good but slightly optimistic: 94.1% of measurements fall in the
nominal 95% interval.

LOCF had nothing to carry in 19 domain-cases (a domain never recorded in that
patient's history to that point) and fell back to the training mean; the count
is reported rather than those cases being dropped from its score.

---

## 4. H2 — Is the representation more than one severity axis?

**H2 passes decisively.** Pre-registered to fail if one component explained
>90% of across-patient variance.

| Representation | PC1 | PC1+PC2 | components for 90% | participation ratio |
|---|---|---|---|---|
| Levels | 0.422 | 0.559 | **7** | 4.34 |
| Rates | 0.372 | 0.496 | **7** | 5.11 |

This is the direct, quantitative contrast with the previous experiment, whose
six discrete states ordered patients along a single severity axis. Here seven
components are needed for 90% of the variance and the effective dimensionality
is 4–5. The representation genuinely preserves differences between patients who
share a subtype label.

### Per-domain rates of change

| Domain | mean/year | SD/year | p10 | p90 | share worsening |
|---|---|---|---|---|---|
| updrs2 (motor ADL) | +0.202 | 0.375 | −0.130 | +0.698 | 69.6% |
| updrs1 (non-motor) | +0.082 | 0.216 | −0.053 | +0.371 | 60.3% |
| pigd | +0.040 | 0.090 | −0.020 | +0.139 | 62.2% |
| scopa (autonomic) | +0.022 | 0.069 | −0.014 | +0.091 | 60.6% |
| brady_rigid | +0.013 | 0.026 | −0.013 | +0.049 | 66.8% |
| gds (mood) | +0.007 | 0.042 | −0.009 | +0.027 | 47.5% |
| ess (sleep) | +0.005 | 0.039 | −0.010 | +0.024 | 50.0% |
| moca (cognition) | −0.003 | 0.036 | −0.024 | +0.021 | 55.6% |
| **tremor** | **+0.000** | **0.0002** | −0.0001 | +0.0002 | 53.6% |

Rates are heterogeneous across patients in the motor-function and PIGD domains
and essentially absent in tremor: the fitted rate innovation variance for
tremor is **0.0000** (as is MoCA's), against 0.0206 for PIGD — the only domain
with a substantial persistent trend. The model is saying that over ~1-year
intervals, tremor and cognition behave like a noisy level that wanders, not
like something with a direction.

---

## 5. H3 — Clinical relevance against independent outcomes

The representation is taken at each patient's **second** visit (formed from
their first two visits only); an outcome counts only if recorded **strictly
later**. None of these outcomes is one of the nine domains.

| Outcome | n | events | best level component | subtype label | Δ (level − label) |
|---|---|---|---|---|---|
| Hoehn & Yahr ≥ 3 | 761 | 127 | `level_updrs2` 0.685 | 0.579 | **+0.105** [+0.054, +0.156] |
| MCI (HVLT/Benton) | 818 | 298 | `level_moca` 0.710 | 0.549 | **+0.161** [+0.109, +0.213] |
| Part IV complications | 736 | 470 | `level_brady_rigid` 0.531 | 0.496 | +0.035 [−0.017, +0.083] |
| Investigator cognitive dx † | 816 | 327 | `level_moca` 0.678 | 0.558 | **+0.119** [+0.069, +0.169] |

† flagged in advance as *not* fully independent — a clinician judgement
plausibly informed by cognitive testing.

| Outcome | best rate component | Δ (rate − label) |
|---|---|---|
| Hoehn & Yahr ≥ 3 | `rate_updrs2` 0.633 | +0.053 [−0.013, +0.119] |
| MCI | `rate_moca` 0.594 | +0.045 [−0.005, +0.096] |
| Part IV complications | `rate_updrs1` 0.537 | +0.041 [−0.014, +0.096] |
| Investigator cognitive dx † | `rate_updrs2` 0.594 | +0.036 [−0.013, +0.085] |

**H3 splits.** The *levels* are clinically more informative than the subtype
label on three of four outcomes, with intervals excluding zero — a single
coordinate of the representation beats the label by 0.11–0.16 AUC. The *rates*
beat the label by +0.04 to +0.05 on every outcome but **no interval excludes
zero**, so on this evidence the rate is not established as clinically
informative. Hoehn & Yahr ≥ 3 is the cleanest case: a separate clinical staging
instrument, 127 prospective events, and the level of motor ADL predicts it at
0.685 against the label's 0.579.

These are deliberately **univariate** AUCs — nothing is fitted, so there is no
model to leak and the comparison against the label is like for like. The best
component is chosen on the same data it is compared on, so each Δ is an **upper
bound** on what one coordinate buys. No multivariate combination of the
representation was assessed.

---

## 6. H4 — Secondary endpoint: the derived TD/PIGD label

Paired against the previous experiment on **3,020 matched visit pairs** (5 rows
dropped to pair the sets).

| Model | Accuracy | 95% CI | Brier | Log loss |
|---|---|---|---|---|
| Profile: level only | **0.8440** | [0.8285, 0.8598] | 0.1195 | 0.3814 |
| Profile: level + rate + medication | 0.8404 | [0.8248, 0.8566] | 0.1177 | 0.3766 |
| Profile: level + rate, no medication | 0.8404 | [0.8249, 0.8563] | **0.1175** | **0.3759** |
| Subtype-history CT-HMM | 0.8371 | [0.8202, 0.8540] | 0.1215 | 0.3963 |
| Subtype-history persistence | 0.8338 | [0.8167, 0.8509] | 0.1598 | 4.1959 ‡ |
| Subtype-history Markov (discrete) | 0.8298 | [0.8130, 0.8471] | 0.1327 | 0.4311 |
| Last observation carried forward | 0.8252 | [0.8083, 0.8425] | 0.1235 | 0.3941 |
| Population mean trajectory | 0.6914 | [0.6677, 0.7153] | 0.2083 | 0.6067 |

‡ rule-based probabilities; log loss finite only through clipping.

| Paired contrast | Δ accuracy | 95% CI | a✓b✗ / b✓a✗ |
|---|---|---|---|
| Profile (full) − subtype-history CT-HMM | +0.0033 | [−0.0050, +0.0112] | 81 / 71 |
| Profile (level only) − subtype-history CT-HMM | +0.0070 | [−0.0010, +0.0149] | 81 / 60 |

**H4 is confirmed as stated — no significant gain** — but the result is more
favourable than the hypothesis anticipated. The profile route **matches** a
model built specifically for this label (intervals include zero) while
delivering better probability quality (Brier 0.1175–0.1195 against 0.1215; log
loss 0.3759–0.3814 against 0.3963), and it does so as a by-product of
predicting nine domains rather than being tuned for two. H4 was written to
prevent a null from being read as a disappointment; it should equally not be
read as a win.

---

## 7. Robustness

Stable. Log-density per domain-observation across the four independent
patient-level splits:

| Model | r0 | r1 | r2 | r3 | SD |
|---|---|---|---|---|---|
| Profile: level + rate + medication | −1.0453 | −1.0442 | −1.0449 | −1.0451 | 0.0004 |
| Profile: level only | −1.0492 | −1.0482 | −1.0490 | −1.0492 | 0.0005 |
| Profile: no medication | −1.0511 | −1.0500 | −1.0508 | −1.0511 | 0.0005 |
| LOCF | −1.1187 | −1.1178 | −1.1184 | −1.1183 | 0.0003 |
| Population mean | −1.4342 | −1.4316 | −1.4324 | −1.4329 | 0.0011 |

The split-to-split SD (0.0003–0.0011) is an order of magnitude smaller than the
rate contribution (+0.0040) and two orders smaller than the gain over LOCF, so
the rankings are not split artifacts. Across-fold parameter SDs are small
relative to their means (e.g. `pigd` rate innovation 0.0206 ± 0.0033;
`brady_rigid` observation variance 0.1728 ± 0.0079). Unlike the previous
experiment, where EM start-spread exceeded 1,000 nats, this model is stably
identified.

### Medication coefficients — a confound, not a drug effect

| Domain | β (per SD of LEDD) | | Domain | β |
|---|---|---|---|---|
| scopa | +0.125 | | brady_rigid | +0.088 |
| updrs1 | +0.118 | | pigd | +0.058 |
| ess | +0.101 | | updrs2 | +0.055 |
| tremor | +0.025 | | gds | +0.017 |
| moca | −0.020 | | | |

Higher LEDD is associated with **worse** scores in eight of nine domains. This
is almost certainly confounding by indication — patients with worse disease are
prescribed more levodopa — and not a claim that levodopa worsens symptoms. The
coefficient is doing useful work (removing dose-correlated variation from the
level, worth +0.0058 log-density) without supporting a causal reading. The
previous experiment, which had LEDD *as a domain*, was encoding this same
confounded signal directly into the representation.

---

## 8. Answer to the research question

> Can we represent each Parkinson's patient using a richer longitudinal
> clinical profile, rather than relying primarily on predefined motor-subtype
> labels, and use that representation to make meaningful predictions about
> their subsequent clinical state?

**Yes for the representation; qualified yes for the prediction; no for
progression as such.**

1. **The richer representation is real and multidimensional.** Seven components
   are needed for 90% of across-patient variance and the effective
   dimensionality is 4–5 — not a severity score in disguise. This is the direct
   repair of the previous experiment's failure.
2. **It is clinically more informative than the subtype label.** One coordinate
   of the representation beats the label by 0.11–0.16 AUC on three prospective
   outcomes never used in fitting, including Hoehn & Yahr ≥ 3 from a separate
   staging instrument.
3. **It predicts the next visit's measurements better** than assuming no change
   (+0.073 log-density, CI excluding zero) and far better than a population
   average (+0.389).
4. **But the gain is the level, not the rate.** Modelling rate of change
   contributes +0.004 of the +0.073 — about 5% — and the rates do not
   significantly beat the label on any clinical outcome. The fitted rate
   variance is exactly zero for tremor and cognition, and substantial only for
   PIGD. Over ~1-year intervals these measures behave much more like noisy
   levels that wander than like trajectories with a direction.
5. **On the old TD/PIGD endpoint it matches** a model built for that label
   while giving better-calibrated probabilities — notable because it was not
   tuned for it.

So the honest summary: **a rich multidimensional profile is a better
description of a patient than the subtype label, and a better predictor of
their next measurements — but what makes it better is knowing precisely where
each patient is across nine domains, not having inferred where they are
heading.** The previous experiment's conclusion needs amending rather than
reversing: richer clinical representations *do* help; it was the compression
into discrete severity states that did not.

### What this does not establish

Not that progression is unmodellable — only that a linear rate over ~1-year
PPMI intervals captures little beyond the level, in these nine measures, at
this follow-up. Not that the representation is better *clinically* than the
label for any decision; it is better at discriminating these outcomes in this
cohort. And nothing about causal medication effects.

---

## 9. Limitations

1. **The rate component is weakly supported.** Positive and
   interval-excluding-zero on profile prediction, but tiny, and not established
   on any clinical outcome. Two domains fit an exactly zero rate variance.
2. **Linear-Gaussian on bounded integer-sum scales.** Predictions are not
   truncated to each scale's range; 95% interval coverage is 0.941 against the
   nominal 0.95, so intervals are slightly too narrow, and calibration near
   the floor of a scale is a genuine weakness.
3. **Domains are modelled independently** — no cross-domain innovation
   covariance. The representation preserves domain differences but does not
   model their correlation. The correlated variant was not run.
4. **Medication enters linearly, only through the observation, and only on
   average.** No patient-specific response (the paper's `m_i`); no effect on
   progression. The coefficients are confounded by indication (§7).
5. **H3 uses univariate AUCs with the component chosen post hoc**, so each Δ is
   an upper bound on one coordinate's value, and no multivariate combination of
   the representation was assessed.
6. **The secondary endpoint tests two of nine coordinates** — the subtype is a
   thresholded ratio of tremor and PIGD.
7. **Elapsed time and the next visit's dose are inputs**, as in the previous
   experiments. PPMI's visit schedule is not independent of disease course.
8. **No external validation cohort** (PDBP is not in this repository) and **no
   item-level Part IV**, so dyskinesia, motor fluctuations and functional
   impairment collapse into one "any motor complication" outcome.
9. **Cross-validation, not LOPO.** Every patient is held out once per repeat
   and four repeats give the stability estimate; LOPO at 934 folds × 4 variants
   was measured at roughly eight hours and was judged not worth it given
   split-to-split SDs of 0.0005.
10. **The target is the project's own classification rule and recorded
    measurements**, not independently verified diagnoses.

---

## 10. Files

**Code** (new; no existing module modified)

- `src/ppmi_subtype/clinical_profile.py` — level-and-rate representation,
  Kalman filter (per-sequence reference and batched), ML fitting, baselines,
  profile metrics, derived-label bridge, representation diagnostics, outcome
  associations.
- `scripts/64_clinical_profile_cv.py` — driver.
- `tests/test_clinical_profile.py` — 49 tests, including an assertion that the
  batched filter reproduces the reference loop to floating point across all
  model variants and both missingness settings, and two causality tests.

**Run outputs** (`experiments/20261020T180515Z_clinical_profile_cv/`,
MANIFEST verifies 25/25)

`profile_predictions.csv` (held-out predictions, all models, all repeats),
`representation.csv` (the per-visit level/rate representation),
`model_comparison.csv`, `contrasts.csv`, `per_domain_errors.csv`,
`representation_dimensionality.csv`, `rate_spread.csv`, `outcome_table.csv`,
`outcome_associations.csv`, `h3_component_vs_subtype.csv`,
`derived_label_predictions.csv`, `derived_label_comparison.csv`,
`derived_label_vs_subtype_history.csv`, `derived_label_contrasts.csv`,
`robustness_by_repeat.csv`, `parameter_stability.csv`, `fold_coefficients.csv`,
`convergence.csv`, `feature_inventory.csv`, `checks.json`, `summary.json`,
`report.txt`, `MANIFEST.sha256`, `git_commit.txt`, `pip_freeze.txt`

**Reproduce** (~6 minutes, 20 workers; nothing written under `outputs/`):

```bash
~/envs/ppmi/bin/python scripts/64_clinical_profile_cv.py --out experiments/$(date -u +%Y%m%dT%H%M%SZ)_clinical_profile_cv --workers 20 --folds 5 --repeats 4 --baseline experiments/20261020T104756Z_next_visit_td_pigd_prediction
```

---

## 11. What to investigate next

1. **Cross-domain correlation in the representation.** Levels and rates are
   currently independent per domain; the H2 spectrum (7 components for 90%)
   says there is structure a joint model could use.
2. **The contrastive step, now that it is clearly motivated.** 330 healthy
   controls with 0.95–1.00 coverage on these measures are available. The
   question it answers — how much of each patient's level is PD-specific rather
   than shared with ageing — is exactly the question §5 raises, since the level
   is doing all the work.
3. **Non-Gaussian observations** respecting each scale's bounds, which should
   fix the 0.941 coverage and the floor behaviour.
4. **Longer horizons.** The rate's failure may be a one-year-interval artifact;
   predicting 3–5 years ahead is where a rate should earn its place, and PPMI
   has the follow-up.
