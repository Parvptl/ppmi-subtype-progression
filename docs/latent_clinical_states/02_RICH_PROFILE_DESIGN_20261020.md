# Rich longitudinal clinical profiles for next-state prediction: design and hypothesis

**Date** 2026-10-20 · **Status** design, written before implementation
**Predecessors** `01_LATENT_CLINICAL_STATES_20261020.md` (latent severity states, negative),
`../next_visit_prediction/01_NEXT_VISIT_TD_PIGD_PREDICTION_20261020.md` (subtype-history CT-HMM)

---

## 1. What I could actually read of the source paper

I was asked to read Severson et al. 2021 (*Lancet Digital Health* 3(9):e555–e564,
doi:10.1016/S2589-7500(21)00101-1). **I could not retrieve its full text.** Every
route returned HTTP 403: sciencedirect.com, thelancet.com (both the HTML article
and the PDF, including the open-access PDF URL that OpenAlex lists as the OA
copy), Europe PMC and PubMed. OpenAlex confirms the article is gold OA
(CC-BY-NC-ND) but that **no repository holds the full text**
(`any_repository_has_fulltext: false`), so there is no legitimate mirror.

What I did read, in full:

| Source | Access | Role |
|---|---|---|
| Lancet 2021 **abstract** | full, verbatim (Europe PMC API) | the study's design, findings and claims |
| Severson, Chahine, Smolensky, Ng, Hu, Ghosh, **MLHC 2020**, PMLR 126:309–330 | **full text, 21 pp** | the "novel personalised input-output hidden Markov model" the abstract refers to — this *is* the method |
| Severson, Ghosh, Ng, **AAAI 2019** (arXiv:1811.06094) | **full text** | the "contrastive latent variable model" the abstract refers to |
| `github.com/kseverso/DiseaseProgressionModeling-HMM` | README | the authors' reference implementation |

So I have the methods the Lancet paper is built from, by the same authors, but
**not the Lancet paper's own Methods section.** Anything specific to that paper —
the exact post-contrastive feature set, the latent dimensionality, the precise
statistical tests on the seven outcomes, and the PDBP validation detail — I have
not read and do not assert below. Where I describe the architecture I am
describing the two method papers.

---

## 2. The representation idea, as the paper uses it

Stripped of architecture, the paper's idea has four parts.

1. **Measure the patient on many axes, not one summary.** The MLHC paper feeds
   the model 59 raw MDS-UPDRS item scores (Parts I–III), not derived subscores
   or a subtype label. The claim is that a label discards information: the
   paper's own state table shows states "primarily differentiated by functional
   impairment, tremor, bradykinesia, and neuropsychiatric measures" — four axes
   that one label cannot carry.
2. **Remove variation that is not about the disease.** The contrastive latent
   variable model takes PD patients as a *target* set and healthy controls as a
   *background* set, and factors the target covariance as `WWᵀ + SSᵀ + σ²I`
   where `SSᵀ` is shared with the controls and `WWᵀ` is unique to the patients.
   Variation common to both — ageing, measurement noise, population spread — is
   pushed into `S` and discarded; only `W` carries forward. The point is that
   *not all variation in a clinical score is disease signal.*
3. **Separate the patient from the state, and the drug from the disease.** The
   PIOHMM emission is
   `x_{i,t} | z=k, d_{i,t} ~ N(μ_k + r_i + (v_k + m_i)·d_{i,t}, Σ_k)`
   — a state mean `μ_k`, a **patient-specific offset `r_i`**, and a medication
   term with a state-specific part `v_k` and a **patient-specific part `m_i`**,
   driven by LEDD as an *input*. Without `r_i` the model must explain each
   patient's own level with the state, and the paper's synthetic experiment
   shows what then happens: "the classic HMM incorrectly assigns states and
   compensates for the personalization with large variances."
4. **Let progression be heterogeneous, not a single trajectory.** The abstract's
   conclusion is that they "discovered non-sequential, overlapping disease
   progression trajectories, supporting the use of non-deterministic disease
   progression models, and suggesting static subtype assignment might be
   ineffective". Patients differ in where they start *and* in how fast they
   move: "the ranking of the starting state did not match the ranking of
   reaching state 8 within 5 years."

And one point about evaluation: the paper establishes clinical significance
against **seven outcomes not used in learning** (MCI, dementia, dyskinesia,
motor fluctuations, functional impairment from fluctuations, Hoehn & Yahr,
death), and separately reports predictive fit. Clinical meaning and predictive
accuracy are argued apart.

---

## 3. What my previous experiment actually did, and why it failed

Not a version of the above. The comparison, component by component:

| Component | Severson et al. | My previous experiment |
|---|---|---|
| Input | 59 raw MDS-UPDRS items | 10 hand-built subscores |
| Nuisance variation | contrastive model against 196 healthy controls | none; no controls used |
| Patient-specific level | `r_i` offset per patient | **absent** — `N(μ_k, Σ_k)` only |
| Medication | LEDD as an **input** modulating the emission | LEDD as one of the 10 **features** |
| Transitions | upper-triangular (progressive) | unconstrained generator |
| Clinical validation | 7 independent outcomes | none |
| External cohort | PDBP, n=610 | none |
| Headline evaluation | state–outcome association, trajectory heterogeneity | next-visit TD/PIGD accuracy |

The failure mechanism I measured was that the six states ordered patients by
*overall severity*, while the TD/PIGD label is a scale-invariant *ratio*. Two
numbers from the data explain why that was close to inevitable:

* **33% of the variance in these measurements is within-patient** (PIGD: 52%).
  A model whose only structure is "which state is this patient in" assigns each
  patient-visit to one of K severity strata, and therefore cannot represent a
  third of the signal by construction. The `r_i` the paper has and I lacked is
  exactly the parameter that frees the state to describe *change* instead of
  *level*.
* **790 of 1,080 patients begin medication during follow-up** and 98.7% are
  untreated at baseline. With LEDD as a feature, "has started levodopa" is one
  of the ten coordinates the states are built from — so my states partly encode
  treatment initiation, which is a confound rather than a disease state.

So the honest reading is that my result is *consistent with* the paper, not a
refutation of it. The MLHC paper's own state table maps 7 of its 8 states to
tremor-dominant under the Stebbins rule, and its only TD/PIGD demonstration was
17 of 24 test patients correct against 14 for a no-change rule. The paper never
claimed its states predict this ratio label well.

---

## 4. What I am building instead, and why not the paper's architecture

I was asked to adapt the idea, not reproduce the method. I am not implementing
the contrastive model or the PIOHMM. Three reasons, in order of weight:

1. **The failure I measured was compression, and the paper's architecture
   compresses too.** Both the paper's model and mine end in K discrete states.
   My diagnostic showed the bottleneck itself cost 10.5 pp (CI [7.6, 13.4]) on
   the identical measurements. Rebuilding a discrete-state model with better
   ingredients risks re-paying that cost. The lesson points at *keeping the
   representation continuous*.
2. **A discrete state is a poor prediction target.** Predicting a
   model-generated state identifier cannot be validated against anything
   measured; the target moves when the model moves. §5 takes this seriously.
3. **The contrastive step addresses a problem I can show we do not have here.**
   Its purpose is to strip variation shared with healthy controls. Our target
   is a *within-patient change* over ~1 year in people already diagnosed; the
   shared nuisance axis it removes is mostly cross-sectional population spread,
   which the patient-specific level absorbs anyway. Controls *are* available
   (330 patients, 2,169 visits — see §6), so this is a deliberate choice and a
   stated opportunity for later work, not a data limitation.

What I keep from the paper: **many axes instead of one label**, **a
patient-specific level separated from the state**, **medication as an input
rather than a feature**, **heterogeneous rates of progression**, and
**clinical relevance evaluated against outcomes not used in learning**.

---

## 5. The prediction target, stated precisely

The four candidate targets are not interchangeable, so I name the one I am
using and what I am not using.

**Primary target — the next visit's multidimensional clinical profile.**
For a patient's visit *k*, predict the 9-vector of clinical measurements
actually recorded at visit *k+1*:

> `brady_rigid, tremor, pigd, updrs2, updrs1, moca, gds, scopa, ess`

as a probability distribution, scored against the measurements that were in
fact recorded. This is the defensible primary target because it is **measured,
not generated**: it does not move when the model changes, every candidate model
can be scored on exactly the same held-out numbers, and it is richer than the
label, which is the whole point of the exercise.

**Explicitly not the primary target:** the next *latent* state. A learned state
is a model artifact; predicting it well would demonstrate internal consistency
and nothing about the patient. The representation here is continuous and is
*evaluated through* the profile prediction rather than being scored itself.

**Secondary endpoint — the next observed TD/Indeterminate/PIGD label.** Kept so
the work connects to the existing 83.84% benchmark, and derived without
slippage: I verified that the project's own `subtype.classify` applied to
(`tremor`, `pigd`) reproduces the panel's recorded subtype on **4,433 of 4,433**
visits. So a predicted profile yields a predicted label by the project's own
rule, with no separate classifier and no new assumption. This is reported
**separately** and with the acknowledgement the task asks for: the subtype is a
thresholded ratio of two of the nine predicted measurements, so it tests two
coordinates of the profile, not the profile.

**Tertiary — clinical relevance.** Association between the representation and
clinical outcomes never used to fit anything (§6).

## Hypothesis

> **H1 (primary).** A representation that gives each patient a per-domain
> *level* and *rate of change*, with medication entered as an input, predicts
> the next visit's 9-dimensional clinical profile better than (a) carrying the
> last observation forward, (b) a population-average trajectory, and (c) the
> same model without the rate-of-change component.

> **H2.** The representation is genuinely multidimensional — its across-patient
> variation is not reducible to a single severity axis.

> **H3.** Per-patient rates of change estimated from early visits only are
> associated with later independent clinical outcomes (Hoehn & Yahr ≥ 3,
> MCI, MDS-UPDRS Part IV motor complications) beyond the subtype label and the
> current level.

> **H4 (secondary).** Deriving the next TD/PIGD label from the predicted profile
> is *not expected* to beat the 83.84% subtype-history benchmark. Stated in
> advance so that a null here is a recorded prediction and not a disappointment:
> my previous work showed that label is near-saturated by persistence.

## Conditions for calling this unsuccessful or inconclusive

Fixed now, before seeing results:

* **H1 fails** if the full model does not beat last-observation-carried-forward
  on held-out profile log-likelihood with a patient-bootstrap CI excluding zero.
  Then a rich representation adds nothing over "assume no change" and I say so.
* **H1 is inconclusive** if it beats LOCF but not the level-only variant —
  meaning the gain is from shrinkage, not from modelling progression.
* **H2 fails** if one principal component explains >90% of the representation's
  across-patient variance; the representation would then be a severity score
  with extra coordinates, exactly what I already found.
* **H3 fails** if the slopes add no discrimination over level + subtype
  (ΔAUC CI including zero). Then the representation is not clinically
  informative on the outcomes available.
* The experiment is **uninterpretable** if parameter estimates are unstable
  across patient-level splits, by the same standard that forced the caveats in
  the previous report.

---

## 6. Data: what is available, what is not

Audited against the actual repository, not assumed.

**Features (9 domains).** The ten from the previous experiment, with
**LEDD removed from the feature vector** and reused as the medication input —
the single most consequential change. Coverage on labelled visits is
0.990–1.000 per domain; 97.6% of visits carry all of them. 4,433 visits over
1,080 patients; 934 patients have ≥2 visits, giving 3,353 prediction
opportunities.

**Medication input.** `LEDD` (coverage 1.000, 454 distinct values) and
`PDTRTMNT` (0.999). Highly dynamic and well time-aligned: 98.7% untreated at
baseline, 790 patients transition to treated, 276 never treated, 14 always.

**Independent outcomes** — none is used to fit anything:

| Outcome | Variable | Coverage | Independent of the 9 features? |
|---|---|---|---|
| Hoehn & Yahr stage | `hy` | 0.998 | **Yes** — separate clinical staging, not derived from UPDRS items |
| MCI | `MCI_testscores` | 0.987 | **Yes** — derived from HVLT, Benton JLO and similar; *not* from MoCA |
| Motor complications | `updrs4_score` | 0.569 | **Yes** — MDS-UPDRS Part IV, excluded from features |
| Investigator cognitive diagnosis | `cogstate` | 0.904 | **Partly** — a clinician judgement plausibly informed by cognitive testing; reported but flagged |
| Death | `Death_Status` | 0.889 | Yes, but only 0 events usable at this follow-up — not used |

**Not available, and therefore not claimed:**

* **No external validation cohort.** PDBP is not in this repository. No external
  validation will be claimed.
* **No MDS-UPDRS Part IV items** — only the Part IV total. So dyskinesia, motor
  fluctuations and functional impairment from fluctuations, three separate
  outcomes in the paper, collapse into one "any motor complication" variable.
* **No Part I patient-questionnaire items** — only the 7 physician-rated `NP1*`
  items are in the curated cut, so the paper's 59-item vector cannot be rebuilt
  exactly (≈53 items would be available).
* **Healthy controls are available** (330 patients, 2,169 visits, coverage
  0.95–1.00 on these measures) — so the contrastive step is *feasible* and is
  being declined on the grounds in §4, not for want of data.

---

## 7. The model

**Representation.** For each patient, a continuous-time **local linear trend**
state space per clinical domain: a *level* and a *rate of change*, filtered
from that patient's own history.

For patient *i*, domain *d*, at visit time *t*:

```
observation   y[i,d,t] = level[i,d,t] + beta[d] * ledd[i,t] + noise,   noise ~ N(0, s_obs[d])
level         level[i,d,t+Δ] = level[i,d,t] + Δ * rate[i,d,t] + w_L
rate          rate[i,d,t+Δ]  = rate[i,d,t]                   + w_R
```

with innovation covariance scaled by Δ as the integrated-Brownian
discretisation requires (`Var(w_L) ∝ Δ³/3`, `Var(w_R) ∝ Δ`,
`Cov = Δ²/2`), so irregular visit spacing is handled exactly rather than
approximated on a grid.

**The representation of patient *i* at visit *k* is the filtered posterior over
`(level, rate)` across the 9 domains** — an 18-dimensional continuous,
patient-specific, time-varying vector with an uncertainty attached. It is
*not* discretised. This is the direct answer to the previous failure: the level
carries where the patient is (what my states were spending their capacity on),
the rate carries where they are going (what they could not represent), and the
two are separated per domain so a tremor-dominant patient with deteriorating
gait is not collapsed onto one severity number.

**Why this and not more states.** Adding states to the previous model adds
severity strata — more of the thing that failed. This changes *what is
represented*: a per-patient level means the model no longer has to explain a
patient's level with the state, which is the role `r_i` plays in the paper; a
per-patient rate makes heterogeneous progression representable, which the paper
identifies as its main clinical finding; and keeping it continuous avoids the
bottleneck my diagnostic priced at 10.5 pp.

**Inference.** Kalman filter and smoother. Filtering gives temporal causality by
construction — the filtered state at visit *k* is a function of visits `0..k`
only — which is a structural guarantee rather than a convention I have to police.
Missing measurements are handled by dropping the corresponding rows from the
observation at that visit, which is exact. Parameters (`s_obs`, innovation
scales, `beta`, initial prior) are shared across patients and fitted by
maximum likelihood on **training patients only**; the smoother is used for
fitting, the filter alone for prediction.

**Prediction.** Propagate the filtered state forward over the actual elapsed
time to visit *k+1* and add the medication term at its *k+1* value, giving a
Gaussian predictive distribution over the next profile — a mean and a variance
per domain, so the prediction is probabilistic and can be scored properly and
checked for calibration.

**Baselines**, all scored on identical held-out profiles:

1. **Last observation carried forward** — "assume no change". The honest floor,
   and the analogue of the persistence rule that was so hard to beat before.
2. **Population mean trajectory** — the training-set mean profile at that
   time-since-baseline. Tests whether patient-specific information helps at all.
3. **Level-only (random walk)** — the same filter with the rate component
   removed. Isolates the contribution of modelling *change*.
4. **No-medication variant** — `beta = 0`. Isolates the contribution of
   treating LEDD as an input.
5. For the secondary label endpoint only: the subtype-history CT-HMM and the
   persistence rule from the previous experiments, on matched cases.

**Evaluation**, the five axes kept apart:

| Axis | Measurement |
|---|---|
| Prediction quality | held-out multivariate predictive log-likelihood, per-domain RMSE/MAE, calibration of predictive intervals |
| Representation quality | PCA of the across-patient representation (H2); per-domain variance retained |
| Longitudinal usefulness | level-only vs level+rate contrast; spread of fitted rates across patients |
| Clinical relevance | AUC for H&Y ≥3 / MCI / Part IV complications from early-visit representation, against subtype + level |
| Robustness | repeated patient-level splits; sensitivity to domain set and to the medication term |

Validation is **leave-one-patient-out** where affordable and repeated
patient-level train/test splits otherwise, with the whole patient held out of
parameter fitting, standardisation and every learned transformation.

---

## 8. Known limitations of this design, before running it

1. **Linear-Gaussian on bounded scales.** These measures are bounded sums of
   integer items; a Gaussian state space can predict outside their range and
   treats a one-point change at the floor as it does at the ceiling. Predictions
   are not truncated, so calibration near the floor is a genuine weakness and is
   reported.
2. **Domains modelled independently.** Levels and rates are per-domain with no
   cross-domain innovation covariance, so the representation preserves domain
   differences but does not model their correlation. A correlated variant is a
   sensitivity analysis, not the primary model.
3. **Medication enters linearly and only through the observation.** As in the
   MLHC paper, LEDD does not modify progression. That paper flags the same
   assumption as a limitation; it is appropriate for a symptomatic therapy and
   wrong if levodopa is disease-modifying.
4. **No patient-specific medication response.** The paper's `m_i` is omitted;
   `beta` is shared. So I separate the drug from the disease on average, not per
   patient.
5. **The secondary endpoint tests two of nine coordinates.** The subtype is a
   thresholded ratio of `tremor` and `pigd`; a good profile prediction can still
   give a poor label if the ratio sits near a threshold.
6. **No external cohort and no item-level Part IV**, as in §6.
7. **The outcomes are not fully independent of each other**, and `cogstate` is
   not fully independent of the features; each is reported on its own.
