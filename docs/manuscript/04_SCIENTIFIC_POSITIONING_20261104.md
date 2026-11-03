# Stage 4 — scientific positioning

**Date** 2026-11-04 · **Follows** `01_STAGE1_RESEARCH_AUDIT`,
`02_ESSENTIAL_ANALYSES_E1_E2_E5`, `03_LITERATURE_REVIEW`

This document makes decisions rather than listing options. Where a decision
rules something out, the reason is given so it can be overturned on evidence.

**One finding here blocks the Results section and is new: the primary analysis
table does not yet exist in a comparable form.** See §6.

---

## 1. The decision

> **Primary research question.** Is longitudinal prediction of the Parkinson's
> disease motor subtype limited by the choice of model family, or by the
> reliability of the label being predicted?

> **Primary contribution.** Evidence that the binding constraint is the label.
> Under one evaluation protocol on one set of 3,361 held-out transitions, model
> families spanning discrete chains, latent-state models, hierarchical Bayesian
> growth and Gaussian processes converge within about five accuracy points of
> each other and of a parameter-free persistence rule; none recovers the
> Indeterminate class; and three direct measurements of the label — immediate
> reversion, residual spread against band width, and threshold fragility —
> explain why.

**This paper does not propose a model.** That is deliberate and should be
stated in the Introduction. Its unit of contribution is a protocol and a set of
measurements, not an architecture. A reader expecting "our method" will
otherwise read the convergence result as a failure to find one.

**Working title**, from the project's own CARE AI 2026 abstract, which already
states this thesis:

> *Label reliability, not model capacity, limits longitudinal prediction of
> Parkinson's disease motor subtypes*

Fifteen words, no punctuation, no abbreviation beyond none — it satisfies the
npj title rule as stated in §7. Two alternatives if a more neutral framing is
wanted: *"How much of Parkinson's disease motor-subtype change is measurement
rather than progression?"* or *"Five model families converge on a persistence
baseline for Parkinson's disease motor-subtype prediction."* I prefer the
original: it names the finding rather than the method.

---

## 2. Why this question, and not the six candidates

Each candidate from the brief was tested against the verified evidence. **None
survives as the primary question**, and the reasons are specific.

| Candidate | Verdict | Why |
|---|---|---|
| **A.** How stable are observed subtype labels? | **Background, not contribution** | Stage 3 found four independent prior studies establishing instability — Simuni 2016, Lee 2019, Erro 2019, von Coelln 2021. Presenting this as the finding would be the paper's most obvious vulnerability. |
| **B.** Does latent-state modelling beat observed-state analysis? | **Secondary result** | The answer is "slightly": +0.0092 accuracy, 95% CI [0.0003, 0.0179], against the identically interval-aware CTMC. The method is also not new — Jackson 2003 is the canonical misclassification framework and Severson 2021 is latent-state modelling in PPMI. |
| **C.** Does a richer representation improve TD/PIGD prediction? | **Secondary result** | No on accuracy (+0.0033, CI spans zero), yes on ranking (AUC +0.0319, CI excludes zero). Too narrow to carry a paper, and confounded by the two models using different validation schemes. |
| **D.** Does the amount of history matter? | **Sensitivity analysis** | Clean and well-executed, but the answer — nothing beyond two or three visits — is a minor finding that supports the thesis rather than being it. |
| **E.** Does performance vary across subtype composition? | **Exploratory only** | Post hoc, and the persistence rule shows nearly the same gradient (0.970 → 0.480 against the model's 0.944 → 0.469), so it is a property of the label sequence, not of any model. |
| **F.** Can continuous ratio modelling give complementary evidence? | **Supplementary** | The growth model is interpretable but loses to the HMM 6/6 splits, has no publishable MCMC diagnostics, and places 12.2% of its five-year predictive mass on impossible negative ratios. |

**The primary question is a synthesis of A–F, not one of them.** It takes A as
established background, uses B, C, D and F as the evidence that model capacity
is not the constraint, keeps E as exploratory colour, and adds the measurements
that explain the pattern. That is what makes it one paper rather than six
loosely related results.

---

## 3. Why the narrative is scientifically defensible

Five properties, each checkable.

1. **One target, measured not model-generated.** Every main-text analysis
   predicts the next *recorded* subtype for the same patients. The target does
   not move when the model moves — unlike the progression-rate clusters that
   Dadu 2022 and Severson 2021 predict.
2. **One protocol.** Patient-level held-out validation, patient-clustered
   intervals, paired contrasts on identical cases, a persistence baseline in
   every table. Convergence is only interpretable because the protocol is held
   fixed — which is precisely why §6 matters.
3. **The negatives are informative, and each localises the constraint
   differently.** More latent states fail (0.7513 at five states against 0.7539
   at three, with 43% fold convergence). Discretising a rich representation
   fails, and the diagnostic says why — the label is a scale-invariant ratio
   while severity states are built from levels. Removing that bottleneck
   recovers +0.1052. A GP on the subscores lands on the HMM's number. More
   history is worth nothing. Retraining to match a short window is worse.
   Convergence across deliberately dissimilar families is far stronger evidence
   about the data than any single model's accuracy.
4. **The constraint is quantified independently of any model.** Residual SD
   1.138 against a 0.25-wide band (4.6×); within-patient SD 1.22; ICC 0.536;
   35.2% immediate reversion (95% CI 31.8–38.8); half of Indeterminate visits
   one hundredth from a boundary; a 0.02 threshold shift reclassifying 67.6% of
   them. These are properties of the measurement, visible before fitting.
5. **The one positive finding restates the thesis from the other side.** A
   continuous nine-domain profile needs seven principal components for 90% of
   its variance, and single coordinates beat the subtype label by +0.105 and
   +0.161 AUC on prospective Hoehn & Yahr ≥ 3 and MCI — outcomes never used in
   fitting. The label is not only unreliable, it is lossy. That gives the paper
   a constructive ending instead of a purely negative one, and it is what makes
   it publishable rather than merely honest.

---

## 4. The analysis hierarchy

| Role | Analyses | Where |
|---|---|---|
| **Primary** | The harmonised multi-family comparison on 3,361 LOPO transitions, with persistence and marginal baselines | Results R3, **blocked by §6** |
| **Primary (descriptive)** | Label reliability: transition matrix with patient-clustered intervals, immediate reversion, sustained-conversion contrast, interval stratification | Results R2 |
| **Secondary** | Latent state vs observed chain (+0.0092); the two-class refinement showing 0.7539 → 0.8369 is a change of task not of model; CT-HMM vs rich profile (accuracy indistinguishable, AUC +0.0319) | Results R4–R5 |
| **Secondary (constructive)** | The profile as a representation: 7 PCs for 90%, outcome AUCs +0.105 / +0.161 | Results R6 |
| **Supporting** | Markov-property rejection (χ² = 156.6, p = 7.9 × 10⁻³³); AIC/BIC support for the hidden layer (ΔAIC 224) | Results R3 |
| **Sensitivity** | History windows; two-visit retraining; 5 vs 3 latent states; 6 vs 8 clinical states; threshold narrowing (**not yet run**) | One condensed paragraph + Supplementary |
| **Exploratory** | Patient-level subtype composition; Indeterminate boundary and focal ratios | Results R7, labelled exploratory |
| **Supplementary** | Hierarchical Bayesian growth; GPR kernel comparison; joint subscore LMM; multi-output GP; time-varying rate; first-visit model | Supplementary S1–S6 |
| **Incomplete — must not be used** | The "+0.005 / +0.021 / +0.047" HMM advantage by history (no code path, no saved output); MCMC diagnostics for the growth model; the threshold-sensitivity run | — |

---

## 5. One paper or two?

**One.** The case for splitting is real and I considered it: the rich clinical
profile has its own primary target (nine measured quantities), its own
validation scheme, its own source-paper lineage, and its own positive result.

It fails on two counts. First, a profile-only paper rests on univariate AUCs
with the coordinate chosen on the same data it is compared against — each Δ is
an upper bound, and there is no external cohort. That is thin for a standalone
paper. Second, and decisively, the profile's role in the main paper is
*load-bearing*: it is the family that shows a nine-domain continuous
representation still cannot beat persistence on the label. Remove it and the
convergence argument loses its strongest arm.

**What should become a second paper** is the clinical-endpoint work the profile
points at — predicting Hoehn & Yahr ≥ 3, MCI or motor complications instead of
the subtype label, prospectively designed, with the coordinate selection done
inside the folds. That is the natural next study and the Discussion should say
so. It is not an appendix to this one.

---

## 6. What blocks the Results section

**The primary analysis table does not yet exist in a comparable form.** This is
new and was not visible in Stage 1.

All families were scored on the same 3,361 transitions from the same 934
patients, but **not under the same label-derivation rule**, and the rule is
worth more than the differences between families:

| Model | Rule | Accuracy |
|---|---|---:|
| `gp_indep_K1` | `pred_subtype` (classify the predicted mean) | 0.7298 |
| `gp_indep_K1` | `prob_subtype` (Monte Carlo, argmax of induced label distribution) | **0.7566** |
| persistence | `pred_subtype` | 0.7227 |
| persistence | `prob_subtype` | **0.7432** |

The same model and the same saved predictions move **+2.7 points** on the
derivation rule alone; the baseline moves **+2.1 points**. The two rules also
behave differently on Indeterminate — `pred_subtype` predicts it (recall ≈
0.17), `prob_subtype` essentially never does (recall 0.000), which is the
behaviour the HMM shows.

The consequence is concrete. The project's headline margin, "the HMM beats
persistence", is +3.1 points against label-persistence at 0.7227 — but the
subscore GP's margin is +1.3 points against *its* persistence at 0.7432. Those
are different baselines, so the margins are not comparable, and no sentence of
the form "model X beats the trivial rule by Y" is currently well defined across
families.

**This does not threaten the thesis.** Every figure above still sits inside a
five-point band around trivial baselines, which is the finding. What it
threatens is the table that demonstrates it.

### E-8 — harmonise the multi-family comparison *(essential, cheap, not yet run)*

Re-score the saved predictions onto one 3-class probability vector per case per
model, then apply one decision rule, one baseline set and one metric suite.
**No refitting is required**, and I verified the inputs exist:

* `outputs/loo/loo_predictions.csv` — 16,805 rows with `p_TD`,
  `p_Indeterminate`, `p_PIGD` for the HMM, both chains, persistence and the
  marginal null.
* `outputs/loo_5hidden/` — the same for the five-state HMM.
* `outputs/joint_subscore{,_gp}/predictions_with_subtype.csv` and
  `outputs/time_varying/predictions_with_subtype.csv` — predictive means and
  SDs for tremor and PIGD (plus `cov_tremor_pigd` for the GP), from which the
  induced 3-class distribution follows by Monte Carlo through the project's
  unmodified `subtype.classify`.

The join on `(patient, history length)` is already established and was verified
complete at 3,361/3,361 in the GPR work. Deliverables: one table of ~9
specifications across 5 families with accuracy, multiclass Brier, log loss and
Indeterminate recall; paired contrasts against a single persistence baseline
with patient-clustered intervals; and a forest plot. That becomes Figure 3 and
Table 3.

### The other blockers, unchanged from Stage 1

**C3** the growth model's single-chain MCMC; **C5** ethics, funding, COI,
CRediT and the PPMI acknowledgement; **C6** the missing archived validation
file; **I18** the clinical output table presented as a clinical quantity when
it is a latent one. C5 needs you and your supervisor; the rest are mine.

---

## 7. Journal fit, reassessed

Stage 1 recommended npj Parkinson's Disease on fit alone. Stage 3 adds a fact
that was not available then: **both of the closest precedents — Simuni 2016 and
von Coelln 2021 — are in *Parkinsonism & Related Disorders***. That journal has
twice published exactly this question, so its reviewers know the literature and
its appetite is demonstrated.

| Journal | Assessment |
|---|---|
| **npj Parkinson's Disease** — *recommended* | Open access, publishes PPMI secondary analyses, and the npj series imposes no strict word or page limit, which suits a paper carrying five model families plus a reliability argument. The field-level implication — that this label is a weak trial endpoint — reaches the right audience. |
| **Parkinsonism & Related Disorders** — *strong fallback* | Demonstrated appetite for the precise question (Simuni, von Coelln). More clinical and less methodological, so the five-family comparison would need compressing. Lower visibility. |
| *Movement Disorders* | Higher clinical-novelty bar. A paper whose headline is "a persistence rule is as good" with no external validation will struggle. |
| *IEEE JBHI* / *Artificial Intelligence in Medicine* | Would want methodological novelty the project does not claim. The paper's honesty about using standard methods is a liability here, not an asset. |

**Requirements I could verify** (from Nature's own content-types page, reached
through search — **the live page is behind an authentication redirect and I
could not fetch it directly, so re-check at submission**): Article abstracts are
**unstructured and up to 150 words**; the npj series sets **no strict word or
page limit** but asks for concision; titles are **up to 15 words** with no
punctuation, idioms or puns. I could not verify figure, table or reference
limits, or the current AI-disclosure policy. Those remain open.

**Do not soften the nulls to fit a venue.** The paper's defining property is
that it reports honest negatives across five families; a journal requiring a
positive methodological result is the wrong home, and trimming to reach one
would destroy the contribution.

---

## 8. What the paper must not claim

Carried forward and now sharpened by the verified literature.

* **Not** that subtype instability is a new finding — Simuni, Lee, Erro and
  von Coelln precede it.
* **Not** that the HMM or any model is clinically useful. PIGD recall is ~0.67;
  a third of conversions are missed.
* **Not** that LOPO is external validation. There is no external cohort.
* **Not** that the AUC separation settles the model comparison — the CT-HMM is
  the more favourably validated of the two (LOPO against 5-fold), stratified
  AUCs are 0.63–0.78 against a pooled 0.86–0.89, and accuracy shows nothing.
* **Not** that the latent states are disease stages. The project is careful
  about this; the manuscript must stay careful.
* **Not** "75% → 84% improvement". Same model, different task.
* **Not** that the five-year clinical table describes a patient's recorded
  label — see I18.

---

## 9. The claim chain, section by section

Each row is a claim the next row depends on. If any link fails, the paper fails
at that point and not before.

| § | Claim | Evidence | Status |
|---|---|---|---|
| Intro | The subtype is a thresholded ratio of two item means | Stebbins 2013 | Verified |
| Intro | It is known to be unstable, and algorithms disagree | Simuni, Lee, Erro, von Coelln, Ren, Mestre | Verified |
| R1 | Our reconstruction of the label is sound | 100% / 99.87% internal agreement **and** agreement with Simuni's independent baseline distribution to within 1 point | Verified |
| R2 | Much of the observed change is immediate reversion, not progression | 35.2% (CI 31.8–38.8); sustained definition removes 51.7% of conversions | Verified (E-2) |
| R2 | The label is fragile to perturbations smaller than its own resolution | 0.02 threshold shift reclassifies 67.6% of Indeterminate; half of Indeterminate visits sit one hundredth from a boundary | Recomputed; **formal run outstanding (O-4)** |
| R3 | Model family does not matter much | Five families within ~5 points, around persistence | **Blocked by E-8** |
| R3 | No family recovers Indeterminate | HMM recall 0.000 (1 prediction in 3,361); profile 0.000 | Verified |
| R4 | A latent state helps a little, exactly where theory predicts | +0.0092 overall; +6.3 pp where the current label is Indeterminate | Verified |
| R5 | Restricting to two classes changes the task, not the model | 2,534/3,361 = 0.7539 → 2,534/3,029 = 0.8366 | Verified |
| R6 | A richer representation is a better *description* than the label | 7 PCs for 90%; +0.105 and +0.161 AUC on prospective H&Y and MCI | Verified |
| Disc | Therefore the constraint is the measurement | Follows from R2, R3, R6 | — |

---

## 10. Recommended order of work

1. **E-8** — harmonise the multi-family comparison. Unblocks the primary
   analysis. Cheap: re-scoring, no refitting.
2. **E-4** — the regulatory block (ethics, funding, COI, CRediT, PPMI
   acknowledgement). Needs you; nothing else is blocked by it but submission.
3. **E-3** — multi-chain refit of the growth model, or demote it to descriptive
   supplementary with the diagnostics gap disclosed. A scope decision.
4. **I18** — relabel or recompute the clinical output table, and save π.
5. **O-4** — run the threshold-sensitivity experiment properly so R2's second
   claim rests on a saved run rather than my recomputation.
6. **Stage 5** — the manuscript outline, then Methods first.

Items 1 and 4 I can start immediately. Item 3 is a decision only you can make.
