# Stage 1 — complete research audit for the manuscript

**Date** 2026-11-04 · **Audit scope** every non-clustering research component in
this repository · **Panel** `outputs/visit_panel.csv`, md5
`fb038ddbdc19ab3368e13940f398e7ba` (verified) · **Repo HEAD** `ca3497e`

> **Update, 2026-11-04.** Essential analyses **E-1, E-2 and E-5 have been run**;
> results are in `02_ESSENTIAL_ANALYSES_E1_E2_E5_20261104.md`. Entries C1, C4
> and the `NP3PSTBL` minor issue are **closed**; limitation **I2 is rewritten**
> and E-6 rescoped. Affected rows below are marked inline. Two further
> documentation defects were found during that work and are recorded in the
> newer document, not here.
>
> **Update, 2026-11-04 (Stage 3).** The literature review is in
> `03_LITERATURE_REVIEW_20261104.md`. It supplies Output 4, which this
> document left open, and adds limitation **I18** below. It also found
> external corroboration of the panel build that is stronger than the
> internal agreement checks in §2.1a.
>
> **Update, 2026-11-04 (Stage 4).** Positioning is settled in
> `04_SCIENTIFIC_POSITIONING_20261104.md`. It supersedes Output 7 below,
> rejects all six candidate questions as the primary one in favour of a
> synthesis, and adds **E-8** — the primary analysis table is not yet
> comparable across model families, because they were scored under two
> different label-derivation rules worth more than the differences
> between the families themselves.

**Nothing was fitted, retrained or modified for this audit.** Every number
below was read from a saved output file, a committed report, or the source code,
and the provenance is named. One number was recomputed from the frozen panel
(§13, the threshold-sensitivity experiment) because no saved run exists; the
recomputation script is in the scratchpad and is disclosed as such.

**No manuscript text is drafted here.** This is Stage 1 of the staged workflow.

---

## 0. Clustering exclusion — confirmed

The excluded investigation is `docs/cluster_phase/` (52 files), together with
`outputs/clustering_audit/`, `experiments/*_s6_clusterability*`,
`experiments/*_s6b_skewed_null*`, `experiments/*_bayesian_slope_clustering`,
`experiments/_stage6b_design`, `experiments/*stage2_m2*` (the Stage-2
integer-score / graded-response line that grew out of it), and
`src/ppmi_subtype/slope_clustering.py`, `m2_*.py`, `grm*.py`, `bbglmm.py`,
`nongaussian.py`, `ppc_*.py`, `replicate_label_diagnostic.py`,
`stage1_checks.py`, `hbayes_t*.py`. Part G of
`docs/GPR_onwards_explanatory_report.md` (§G.1–G.13) and
`scripts/20_clusterability_audit.py`, `26_clusterability_extended.py`,
`29_clusterability_skewed_null.py`, `34`–`43`, `44`–`59` are also excluded.

None of it enters the inventory, the results tables, the gap statement or the
recommended analyses below.

**Two items are flagged for your review rather than excluded automatically**,
because they are adjacent to clustering but are not that investigation:

| Item | Why it is flagged | My reading |
|---|---|---|
| **Latent clinical states** (`docs/latent_clinical_states/01_*`, scripts 61–62) | It discretises patients into K=6/8 latent states, which *looks* like clustering | **Keep.** It is a supervised prediction experiment with a held-out target and a pre-registered state-count rule, not a search for cluster structure, and its negative result is the direct motivation for the Rich Clinical Profile. Its own text cites the clustering closure only as context. |
| **Part G of the GPR report** (`§G.1–G.13`) | It is the clusterability audit | **Exclude.** This *is* the clustering investigation under a different filename. Parts A–F of that report are not. |

If you disagree on the first, the Rich Clinical Profile section needs rewriting,
because the profile's design rationale rests on it.

---

## OUTPUT 1 — Complete research audit

### 1.1 What the project actually is

A B.Tech project (Dept. of Data Science, IIT Palakkad) that takes the clinical
TD/PIGD motor-subtype rule of Stebbins et al. (2013), recomputes it per visit
from item-level MDS-UPDRS data in PPMI, and then asks — across five successive
model families and three years of PPMI follow-up structure — how much of a
patient's **next** recorded subtype is predictable, and from what.

An abstract already exists and already states the narrative the evidence
supports: `report/CARE_AI2026_Abstract.tex`, *"Label Reliability, Not Model
Capacity, Limits Longitudinal Prediction of Parkinson's Disease Motor
Subtypes"*, Parv Patel, Mayank Tewatia, Dr. Sahasranand K. R. **This is the
single most important document for Stage 4 and I recommend the manuscript be
built as the full-length version of it.** (See Output 7.)

### 1.2 Research chronology (Audit B), reconstructed from git

| Date | Commit | Component | Why it happened |
|---|---|---|---|
| 2026-08-20 | `da17175`, `d458d6d` | Panel build; 3-class transition analysis | The curated cut has no tremor subscore and a 2-class `td_pigd`, so the ratio and the Indeterminate class had to be rebuilt from items |
| 2026-08-27 | `6377fe3` | DTMC, CTMC, HMM; Markov-property test | The observed-label transition matrix was found to be measuring reversion, so a latent state was tested |
| 2026-08-27 | `99dbeed` | Hierarchical Bayesian growth model | It is the model the project proposal specified; implemented *as written* to see what it gives |
| 2026-08-27 | `8d38210` | First-visit model | The longitudinal models have nothing to filter for a new patient; measured directly |
| 2026-08-27 | `4e5bdd2` | Stages 2/4/6: ratio exploration, latent posteriors, longitudinal CV | Proposal deliverables; also where σ=1.138 vs a 0.25 band was first measured |
| 2026-09-11 | `69fe9f1` | Patient-level LOO (3-class) | A single 70/30 split was criticised at a review meeting |
| 2026-09-17 | `7341f38` | 5-hidden-state HMM LOO | Does raising latent-state count help? |
| 2026-09-30 | `dada934`, `f6ed395` | GPR on the ratio; subscore GP; joint LMM; MOGP; time-varying rate | Move from a 3-class label to the continuous ratio, then to the two subscores, because the ratio is undefined at 16.3% of labelled visits |
| 2026-10-21 | `6797f77` | **CT-HMM next-visit two-class prediction** | Restrict the target to TD-vs-PIGD and report calibrated probabilities |
| 2026-10-21 | `b250264` | Latent clinical states — **closed as negative** | Does a rich clinical representation compressed into latent states beat label history? |
| 2026-10-21 | `19bf8aa` | **Rich longitudinal clinical profile** | The negative above localised the loss to the discretisation, so keep the representation continuous |
| 2026-10-21 | `3775a39` | Research-meeting manuscript | First write-up of the two newer methods |
| 2026-11-03 | `2480868` | AIC/BIC, ROC/AUC, Indeterminate boundary | Accuracy failed to separate the two methods; look at ranking and fit |
| 2026-11-03 | `e032f17` | Stage-wise evaluation | Is the ~84% an artefact of accumulated history? |
| 2026-11-03 | `0c6877f` | CT-HMM history window; two-visit retraining | How much history does the filter need, and does retraining on windows help? |
| 2026-11-03 | `a1a560d` | Experiments 6A/6B (profile history, profile retraining) | The same two questions, asked of the profile model |
| 2026-11-03 | `ca3497e` | Patient-level accuracy audit of 6A/6B | Is the aggregate representative of individual patients? |

**Questions that remain unanswered** are in Output 5 and Output 6.

### 1.3 Complete experiment inventory (Audit A)

Twenty-two completed, non-clustering experiments. `Manuscript` is my
recommendation, argued in Output 7.

| # | Experiment | Question | Method | Cohort / eval unit | Evaluation | Headline result | Status | Manuscript |
|---|---|---|---|---|---|---|---|---|
| E1 | Panel build & label validation | Can the 3-class label be rebuilt from items? | Stebbins rule, strict missing-item policy | 1,080 pt / 5,774 visits / 4,441 labelled | Agreement vs PPMI's own derivations | 4,431/4,431 (100.00%) 2-class; 1,595/1,597 (99.87%) 3-class | **Verified** | Methods + Table 1 |
| E2 | Observed transition analysis | How stable is the label? | 3×3 matrix, Wilson CIs, interval strata, reversion, KM | 3,361 pairs / 934 pt, visit-level | Descriptive | 35.2% of changes reverse next visit; Indeterminate persists 19.2% | **Verified** | **Main, Result 1** |
| E3 | Markov-property test | Is the observed label Markov? | χ² on labelled triples | 1,583/235/609 triples | Hypothesis test | Rejected in all 3 current states (TD χ²=156.6, p=7.9e-33) | **Verified** | Main |
| E4 | DTMC / CTMC / HMM, 70/30 + 10 splits | Does a latent state help? | ML; held-out log-lik per transition | 755 train / 325 test pt; 1,002 pairs | 10 patient-level splits | HMM > DTMC 10/10, +0.0284 ± 0.0094, t p=5.4e-6; CTMC loses at every interval | **Verified** | Main (superseded on the headline by E7) |
| E5 | Hierarchical Bayesian growth model | What is the population progression rate? | Gibbs, conjugate N/IG | 2,426 obs / 573 train pt; 825 common transitions | 6 patient-level splits | Ratio −0.0927/yr [−0.1159, −0.0682]; HMM ahead 6/6 by +0.0579 | **Verified** | **Supplementary** |
| E6 | First-visit ("new patient") model | What can be said with no history? | Multinomial on baseline + horizon | 5 splits; 1,002 held-out rows | Held-out log-lik | Direct model beats the HMM by +0.0319; HMM is the *worst* of five | **Verified** | Supplementary |
| E7 | Patient-level LOO, 3-class | Honest accuracy on the full task | LOPO, 934 folds | 934 pt / 3,361 transitions | Patient bootstrap | **0.7539 [0.7348, 0.7731]**; Indeterminate recall **0.000** | **Verified** | **Main, Result 2** |
| E8 | 5-hidden-state HMM LOO | Do more latent states help? | LOPO, 934 folds | same 3,361 | Accuracy, convergence | 0.7513 — *worse*; converged in only **404/934 (43.3%)** folds, and **59.2% of its 3,361 predictions come from non-converged folds** (measured 2026-11-04 from `train_converged` in `outputs/loo_5hidden/loo_predictions_all_models.csv`; the three-state model is 934/934) | **Verified** | **Main (one line)** |
| E9 | GPR on the ratio, 4 kernels, 4 cohorts | Does a flexible kernel beat a linear trend? | Hand-written GP, LOPO | 821 pt / 2,707 points (unfiltered) | Pooled + per-patient bootstrap | K1−K0 = −0.0333 [−0.0467, −0.0205] MAE; **K0 ≡ hbayes**; 12–13% impossible predictive mass | **Verified** (two earlier claims superseded) | Supplementary |
| E10 | GP mean-function held-out test | Does a fitted linear mean help out of sample? | Matched LOPO, 821 folds | 821 pt | Held-out MAE/MSE/CRPS | MAE −0.0071 [−0.0193, +0.0056] (null); better on MSE and CRPS | **Verified** | Supplementary |
| E11 | Multi-step horizon analysis | Does the GP's edge widen with horizon? | LOPO, 4 cohorts, composition controls | 4 cohorts | Within-patient contrasts | **Earlier "widens with horizon" superseded**; not detectable within 3 y | **Superseded → corrected** | Supplementary |
| E12 | **Subscore GP baseline** | Does modelling the two subscores inherit the ratio model's edge? | Two independent K1 GPs, LOPO | **934 pt / 3,361 points — the HMM's exact set** | Patient bootstrap | GP vs persistence: tremor **null**, PIGD **worse** (+0.0134). Derived subtype **0.7566 vs HMM 0.7539, +0.0027 [−0.0042, +0.0098]** | **Verified** | **Main, Result 3** |
| E13 | Joint subscore LMM | Do the subscores share random effects? | Bivariate LMM | 3,361 / 934 | Held-out MAE + 3-class | Cross-outcome correlation ≈ 0; 3-class 0.7087 — **below persistence 0.7227** | **Verified** | Supplementary |
| E14 | Multi-output GP | Does cross-output covariance help? | MOGP, 3 variants | 3,361 / 934 | Held-out MAE + 3-class | Joint ≈ independent (0.7301 vs 0.7298); not kept | **Verified** | Supplementary |
| E15 | Time-varying progression rate | Does the rate change over follow-up? | LMM with spline knots at 2 and 5 y | 3,361 / 934 | Paired held-out MAE | Description improved, **prediction did not** (PIGD −0.0034 = 1.4% of typical error); both lose to persistence | **Verified** | Supplementary |
| E16 | **Latent clinical states (K=6, K=8)** | Do learned latent states on 10 measurements beat label history? | CT-HMM + diagonal Gaussian emission, EM; LOPO | 654 eval pt (280 spent on selection) / 2,091 scored | Paired, patient bootstrap | **0.7255 vs 0.8384 baseline, −0.1130 [−0.1438, −0.0807]**; PIGD recall 0.218; same measurements *without* the bottleneck +0.1052 | **Verified (negative)** | **Main, Result 4** |
| E17 | **CT-HMM next-visit TD-vs-PIGD** | Can the next label be predicted, and with what probabilities? | CT-HMM, LOPO, 7 candidates | 899 pt / 3,029 eligible of 3,361 | Patient bootstrap, 19/19 checks | **0.8369 [0.8204, 0.8541]**; persistence **0.8336**; Δ +0.0033 [−0.0037, +0.0100]; calibration gap **0.0065** | **Verified** | **Main, Result 5** |
| E18 | **Rich clinical profile** | Does a continuous 9-domain level+rate representation predict better? | Per-domain local linear trend, Kalman; 5-fold × 4 | 934 pt / 3,353 predictions; 3,021 eligible | Patient bootstrap, 13/13 checks | Primary target +0.0733 [+0.0657, +0.0813] over LOCF, of which **rate only +0.0040**; derived label 0.8405 | **Verified** | **Main, Result 6** |
| E19 | AIC/BIC + ROC/AUC + Indeterminate boundary | Do continuous scores separate the two methods? | Full-cohort fit (IC); paired AUC on 3,020 matched | 898 pt / 3,020 | Patient bootstrap | CT-HMM hidden layer ΔAIC **224**; profile AUC **0.8902 vs 0.8583, +0.0319 [+0.0229, +0.0419]**; Indeterminate boundary **null** | **Verified** | **Main, Result 7** |
| E20 | Stage-wise evaluation | Is the ~84% an artefact of accumulated history? | Regroup matched predictions by visit index | 896 pt / 3,017 | 2,000 patient bootstrap, 20/20 checks | Stage 1 = 0.8347 / 0.8455 vs pooled 0.8373 / 0.8406; **no accuracy contrast excludes zero**; persistence's balanced accuracy rises identically | **Verified** | **Main, Result 8** |
| E21 | History-window experiments (CT-HMM; profile 6A) | How much history does the filter need? | Truncate prediction history, parameters fixed per fold | 228 pt / 734 common (CT-HMM); 898 pt / 3,021 (6A) | Pre-specified primary + bootstrap; 20/20, 18/18, 28/28 checks | last-2 ≡ full on every metric; **last-1 is genuinely worse**; every short window costs PIGD sensitivity | **Verified** | **Main, Result 9** (condensed) |
| E22 | Short-window retraining (CT-HMM; profile 6B) + patient-level audit | Does training on two-visit windows help? | Full 2×2 training × history; overlapping and disjoint windows | 742 pt / 2,192 (CT-HMM); 740 pt / 2,184 (6B) | Pre-specified primary; 24/24, 23/23 checks | **No.** Worse on every probability metric; damage is from the *training* change; 89–91% of patients bit-identical | **Verified (negative)** | **Main (one paragraph) + Supplementary** |

### 1.4 Study overview (Audit C)

| Field | Verified value | Source |
|---|---|---|
| **Design** | Retrospective secondary analysis of a prospective observational cohort | `config/config.yaml`, `docs/DATA_REQUIREMENTS.md` |
| **Data** | PPMI Curated Data Cut Public 20260511 (19,450 visits × 212 cols, 4,788 pt, BL→Y15); MDS-UPDRS Part II + Part III item files (20 Aug 2026); Participant_Status (26 Jul 2026) | `docs/DATA_REQUIREMENTS.md` |
| **Cohort rule** | `subgroup == "Sporadic PD"`; events `ST, PW, U01, U02` excluded; follow-up ≤ 15 y; ≥2 visits | `config/config.yaml` |
| **Cohort realised** | **1,080 patients, 5,774 visits**, follow-up to year 14 | `outputs/panel_summary.txt` |
| **Labelled visits** | **4,441 (76.9%)**; 1,333 (23.1%) unlabelled | same |
| **Ratio defined** | 3,716 (83.7%); **725 (16.3%) undefined** because the PIGD subscore is 0 (665 TD, 60 Indeterminate) | `outputs/hbayes/report.txt` |
| **Subtype distribution** (all labelled) | TD 2,891 (65.1%), Indeterminate 441 (9.9%), PIGD 1,109 (25.0%) | `outputs/panel_summary.txt` |
| **At baseline** | TD 772 (71.7%), Indeterminate 109 (10.1%), PIGD 195 (18.1%) | same |
| **Visits per patient** | median 3, max 14; ≥2: **934**, ≥3: 762, ≥5: 333, ≥8: 145 | same |
| **Transitions** | **3,361** consecutive labelled pairs from **934** patients, median interval **1.00 y** | `outputs/transitions/report.txt` |
| **Primary analysis** (recommended) | Next-visit observed subtype under leave-one-patient-out, across five model families on one protocol | E7, E12, E16, E17 |
| **Primary model** | CT-HMM (3 latent states, misclassification matrix) | `src/ppmi_subtype/hmm.py` |
| **Baselines** | Persistence, discrete Markov, CTMC, marginal null, LOCF, population mean, direct logistic | E7, E17 |
| **Main finding** | Four model families with very different assumptions converge on the same held-out performance, a few points above a trivial rule, and **none recovers the Indeterminate class** | E7, E12, E16, E17, E20 |

---

## OUTPUT 2 — Verified results table

Every provisional figure in your prompt is adjudicated. **"Verified"** means I
read it out of a saved output file or recomputed it. Precision is reported as
the source reports it.

### 2.1 Dataset and labels

| Provisional claim | Verified value | Source | Confidence |
|---|---|---|---|
| 1,080 patients | **1,080** | `outputs/panel_summary.txt` | **Verified** |
| 5,774 visits | **5,774** (panel has 5,775 lines incl. header) | `outputs/visit_panel.csv` | **Verified** |
| 4,441 labelled visits | **4,441** | `outputs/panel_summary.txt` | **Verified** |
| 3,361 consecutive labelled pairs | **3,361** from 934 patients | `outputs/transitions/report.txt` | **Verified** |
| ~14 years follow-up | **year 14** is the last populated year (2 visits) | `outputs/transitions/report.txt` §3 | **Verified** |
| 11 tremor items, 5 gait/postural items | **11** (1 Part II + 10 Part III), **5** (2 Part II + 3 Part III) | `config/config.yaml`, `src/ppmi_subtype/subtype.py` | **Verified** |
| TD ≥ 1.15, PIGD ≤ 0.90 | **exactly that**, band open at both ends (`band_boundaries: outer`) | `config/config.yaml` | **Verified** |
| 100.00% agreement, curated binary | **4,431 / 4,431 = 100.00%** | `outputs/validation.txt` | **Verified** |
| 99.87% agreement, archived 3-class | **1,595 / 1,597 = 99.87%**, 390 patients | same | **Verified**, but see §2.1a |
| 23.1% of visits unlabelled | **23.1%** (4,441 of 5,774 labelled) | `outputs/panel_summary.txt` | **Verified** |
| Missing visits are more advanced patients | median H&Y **3.0 vs 2.0, p = 1.4e-17**, and in **1,243 of 1,253** partly-scored exams the missing item is postural stability | README, Weeks 3–4 report §6.1 | **Partially verified** — the test result is quoted in committed prose but I found no CSV behind it |

**§2.1a — the agreement checks are not external validation.** Both references
are PPMI's own derivation of *the same* Stebbins formula from *the same*
MDS-UPDRS items. What 100.00% / 99.87% establishes is that the implementation
is correct, including the sentinel-101 trap and the 2↔3 code swap in the
archived file. It does **not** establish that the label is a valid measurement
of motor phenotype, and the manuscript must not call it validation. The
`validate.py` docstring's claim that passing both *"licenses using the
recomputed label as ground truth"* overstates this and should not be carried
into the paper.

**§2.1b — the three-class check cannot currently be rerun.**
`data/validation/` is **empty**; the archived Apr-2020 file is gone. The 99.87%
figure survives only in `outputs/validation.txt`. For a data-availability
statement this needs either the file restored or the limitation disclosed.

### 2.2 Observed transitions — all verified

| Previous subtype | → TD | → Indeterminate | → PIGD | n |
|---|---:|---:|---:|---:|
| TD | 0.808 | 0.080 | 0.112 | 2,261 |
| Indeterminate | 0.440 | 0.192 | 0.368 | 334 |
| PIGD | 0.184 | 0.114 | 0.702 | 766 |

Counts: TD 1827/181/253; Indeterminate 147/64/123; PIGD 141/87/538. Rows sum to
1.000. Wilson intervals are in `outputs/transitions/report.txt`.

| Provisional claim | Verified | Source |
|---|---|---|
| Only 19.2% of Indeterminate visits stay Indeterminate | **0.192** (64/334) | `outputs/transitions/report.txt` |
| 35.2% of observed changes reverse at the next visit | **35.2%** | same §4 |
| TD→PIGD occurred 253 times | **253** | same §1 |
| PIGD→TD occurred 141 times | **141** | same §1 |
| Transition probabilities rise with interval | TD→PIGD **0.100** (≤1.5 y) → **0.289** (>4 y); TD→TD **0.820 → 0.578** | same §2 |

**Denominators are visit-level and patients contribute multiple transitions.**
934 patients supply 3,361 pairs; 476 of them (51.0%) never change state. The
transition matrix is therefore not an estimate of a per-patient quantity, and
the Wilson intervals in that file **ignore within-patient clustering** — they
are too narrow. Every later experiment uses a patient-clustered bootstrap
instead; the transition table does not. *This is a real defect to fix before
submission (Output 6, Essential-2).*

### 2.3 Markov property — verified

| Current state | n triples | χ² | df | p |
|---|---:|---:|---:|---|
| TD | 1,583 | **156.5856** | 4 | **7.890e-33** |
| Indeterminate | 235 | 15.2825 | 4 | 0.004150 |
| PIGD | 609 | 61.1504 | 4 | 1.662e-12 |

Source `outputs/model/markov_property_test.csv`. Concretely: currently-TD
patients arriving from TD move to PIGD 8.6% of the time; arriving from PIGD,
39.3%.

**What this licenses.** It rejects a first-order chain *in the observed label*.
It does **not** by itself license a latent-state interpretation — unequal visit
intervals and within-patient correlation could both produce history dependence,
and the test treats triples as independent when patients contribute several.
The correct framing is "the current label is not a sufficient statistic for the
past", with the latent-state reading offered as one explanation and tested
separately (E4, E17).

### 2.4 Hidden Markov model — verified, with one important caveat

| Quantity | Verified value | Source |
|---|---|---|
| Hidden states | 3, one per observed label | `src/ppmi_subtype/hmm.py` |
| Objective | **joint log-likelihood of observed label sequences** (forward algorithm), maximised by L-BFGS-B, 4 restarts, seed 42 | `hmm.py: fit_hmm`, `_forward_loglik` |
| Emission matrix | **estimated by ML jointly with Q and π**, softmax-parameterised, with a **box constraint** forcing diagonal ≥ 0.5 | `hmm.py: _emission_logit_bound`, `config model.emission_floor: 0.5` |
| Emission diagonal | TD **0.9412**, Indeterminate **0.5000**, PIGD **0.8424** | `outputs/model/hmm_emission.csv` |
| Latent generator | TD→Ind 0.0937, TD→PIGD 0.0057 (**16.4× smaller**), Ind→TD 0.1156, Ind→PIGD 0.3557, PIGD→TD **0.0000**, PIGD→Ind 0.0605 | `outputs/model/hmm_latent_generator.csv` |
| Mean sojourn, latent vs observed | TD 10.06 vs 3.41 y; Indeterminate 2.12 vs 0.44; PIGD **16.52 vs 2.28** | `outputs/model/mean_sojourn.csv` |
| Held-out (1 split, 1,002 pairs) | log-lik/pair **−0.6384**, accuracy **0.7695**, Brier 0.3458 | `outputs/model/heldout_performance.csv` |
| 10-split robustness | HMM > DTMC in **10/10**, mean **+0.0284** (sd 0.0094), paired t **p = 5.43e-6**, Wilcoxon p = 0.0020 | `outputs/model/robustness_across_splits.csv` |
| Clinical table | current Indeterminate → P(PIGD at 5 y) **0.633**; current TD → **0.191** | `outputs/model/clinical_table_hmm.csv` |
| Prediction | **filtering only** — `alpha @ exp(Q·Δt) @ E`; `alpha` absorbs label *k+1* only after the prediction is recorded | `hmm.py: predictive_distributions` |

**The caveat, and it matters for the manuscript.** The Indeterminate emission
row is `(0.250, 0.500, 0.250)` — **exactly** the constraint boundary. The floor
is **active**, so 0.500 is the constraint, not an estimate. The manuscript must
not report "the examination records Indeterminate correctly 50% of the time" as
a finding. The honest statement is: *the latent Indeterminate state carries no
identified emission information; the likelihood pushes it to a uniform
distribution and the identifiability constraint stops it there.* The code says
so in its own comment; the earlier prose does not.

Related: the Indeterminate state is a latent construct whose only data
signature is a label that persists 19.2% of the time. **Nothing in this project
supports calling the three latent states disease stages**, and the project
correctly refuses to name them anywhere (see also E16 §7).

### 2.5 Hierarchical Bayesian growth model — all verified

| Parameter | Mean | 95% CrI | Provisional claim | Verdict |
|---|---:|---|---|---|
| β₀ population baseline ratio | **2.2920** | 2.1577 – 2.4272 | 2.2920 | **Verified** |
| β₁ population rate, /yr | **−0.0927** | **−0.1159 – −0.0682** | −0.0927, [−0.1159, −0.0682] | **Verified** |
| σ total residual SD | **1.1380** | 1.0993 – 1.1778 | 1.138 | **Verified** |
| τ₀ SD of patient baselines | **1.3408** | 1.2466 – 1.4421 | 1.3408 | **Verified** |
| τ₁ SD of patient slopes, /yr | **0.1033** | 0.0780 – 0.1315 | 0.1033 | **Verified** |

Also verified: P(slope < 0) = 1.0000; σ is **4.6×** the 0.25 band width;
population trajectory crosses TD 1.15 at **12.3 y [9.9, 16.4]** and PIGD 0.90
at 15.0 y [12.1, 19.9]; head-to-head on 825 common transitions **HMM 0.7612 /
−0.6559 vs HB 0.7370 / −0.6814**; HMM ahead **6/6 splits**, mean **+0.0579**
(sd 0.0188), paired t p = 6.56e-4. Source `outputs/hbayes/report.txt`,
`posterior_summary.csv`, `head_to_head.csv`.

| Model-limitation claim | Verified | Source |
|---|---|---|
| Ratio skew ≈ 1.27 | **1.266** | `outputs/exploration/report.txt` |
| ~7.0% exact zeros | **0.070** | same |
| Bounded below at zero | yes, and 10.3% of values sit inside the 0.25-wide band | same |
| Naive per-patient OLS slope SD 0.531 | **0.5307** over 630 patients (3+ visits, ≥6 mo span) | same §2 |
| Bayesian τ₁ 0.103 | **0.1033** | `posterior_summary.csv` |
| OLS slope variability falls with span | SD **0.8774 → 0.6422 → 0.3114 → 0.1701** across <2y / 2–4y / 4–8y / >8y | `outputs/exploration/report.txt` §4 |
| ICC ≈ 0.536 | **0.536** (between 1.7105, within 1.4792) | same §3 |
| Lag-1 residual correlation ≈ −0.219 | **−0.219** from 2,050 pairs | same §5 |
| Simulated iid null ≈ −0.332 | **−0.332 [−0.365, −0.294]** | same §5 |
| σ_obs and σ_proc not separately identifiable | yes — only the sum enters the likelihood | `src/ppmi_subtype/hbayes.py` docstring |
| Normal likelihood imperfect for a non-negative skewed ratio | yes; **12.2% of 5-year predictive mass on negative ratios** | README, `outputs/hbayes/report.txt` |

**⚠ RESOLVED 2026-11-04 — see `02_ESSENTIAL_ANALYSES_E1_E2_E5_20261104.md` §E-1.**
The framing below is **wrong and is retained only for the record**. The
committed code is self-consistent: both passages are chosen by the same
`if/elif/else` on the same comparison, and the contradictory sentence appears
in no committed version of `scripts/06` across all 45 commits. The
contradiction lives only in the **stale output file**, written by an
uncommitted earlier draft and never regenerated because `outputs/` became
read-only. §5 is correct; the closing item 4 is stale and must not be quoted.
Re-running the committed script reproduces §5's reading in both passages, and
the difference statistic is **+0.1133 [+0.0457, +0.1785]** with a one-sided
Monte Carlo p of 5.0 × 10⁻⁴.

The original (incorrect) framing follows. The lag-1 residual result is
interpreted two incompatible ways **inside one output file**:

- `outputs/exploration/report.txt` §5: *"More positive than the null, which is
  the signature of accumulating process noise and **favours the state-space
  form**."* The README repeats this.
- `outputs/exploration/report.txt`, "WHAT THIS IMPLIES" item 4: *"independent
  deviations are a defensible assumption and **the growth form is adequate**.
  **Nothing here argues for the state-space alternative**."*

Both sentences are generated by the same script from the same number. One is
wrong. *My reading:* −0.219 against a null of −0.332 is **more positive** than
iid noise would give, so deviations do persist, which favours the state-space
form — i.e. §5 is right and the summary line is a drafting error.

**That reading was confirmed on 2026-11-04**, and the cause turned out to be a
stale artefact rather than a live defect. The project does **not** contradict
itself here.

**MCMC diagnostics are incomplete for publication.** The sampler runs **one
chain**: 4,000 draws, 1,000 burn-in, thin 2 → 2,000 retained, priors
N(0, 100²) on β and InvGamma(0.01, 0.01) on each variance. The only diagnostic
reported is a first-half/second-half mean comparison. There is **no R̂, no
effective sample size, no multi-chain run, no prior-sensitivity check**. See
Output 6, Essential-3.

### 2.6 Patient-level LOO validation (3-class) — all verified

| Provisional claim | Verified value | Source |
|---|---|---|
| Patients evaluated: 934 | **934** | `outputs/loo/loo_overall_metrics.json` |
| Predicted transitions: 3,361 | **3,361** | same |
| Correct predictions: 2,534 | **2,534** | same |
| Accuracy 75.4% | **0.7539422790836061** | same |
| Patient-bootstrap 95% CI 73.5–77.3% | **[0.7348, 0.7731]** | same |
| Mean per-patient accuracy 75.4% | **0.7538** (median 1.0000, sd 0.3207) | same |
| Log-likelihood / prediction −0.673 | **−0.6726** | same |
| Brier 0.369 | **0.36884886** | same |
| Ten leakage and consistency checks passed | **exactly 10, all PASS** | `outputs/loo/report.txt` §3 |

Answers to your eighteen verification questions, from `report.txt`, `loo.py`
and `scripts/09_loo_validation.py`:

1. Unique patients **934** (every patient with ≥2 labelled visits).
2. Transitions **3,361**. 3. Predictions **3,361** — one per transition.
4. Correct **2,534**. 5. Denominator = all 3,361 predictions (transition-level).
6. **Both** are reported: transition-level 0.7539 and mean per-patient 0.7538;
   they agree to 0.0001 here.
7. Bootstrap unit = **patient** (whole patients drawn with replacement).
8. Replicates: the module default; the Wilson interval is also reported and is
   narrower ([0.7391, 0.7682]), correctly flagged as the wrong one to quote.
9. Percentile interval, no bias correction.
10. Log-likelihood is **per prediction** (mean, not sum).
11. Brier is the **multiclass** (3-class) Brier: mean squared error of the
    full probability vector. **This is why 0.369 and the two-class 0.121 of E17
    are not comparable.**
12. Probabilities come from `hmm.predictive_distributions` — the filtered
    forward recursion.
13. Yes, every patient held out exactly once (checked).
14/15. There is **no preprocessing and no feature selection** to refit — the
    HMM consumes integer labels and gaps.
16/17. Hyperparameters (emission floor 0.5, 4 restarts, seed 42) are **fixed in
    `config.yaml` in advance and never tuned on held-out data**. That is the
    right design, but it also means no hyperparameter was selected at all.
18. The 10 checks are listed in `report.txt`; they cover fold construction,
    history ordering, prediction counts, probability normalisation, argmax
    consistency and convergence.

**This is not external validation and the manuscript must not say it is.**
LOPO reuses one cohort; it controls for patient-level optimism, not for cohort,
site, protocol or era. No independent cohort (PDBP or otherwise) was used
anywhere in this project.

**The result that matters most in this table is not the accuracy.**
Indeterminate recall is **0.000**: the model predicted Indeterminate **once in
3,361 opportunities** (precision 0.000, support 332). All 332 Indeterminate
targets are errors. Per class: TD precision 0.786 / recall 0.910 / F1 0.843;
PIGD 0.669 / 0.666 / 0.668.

By current state: Indeterminate **0.4880** (n=334), PIGD 0.7023 (766), TD
0.8107 (2,261). By history length: **0.7473, 0.7900, 0.7510, 0.7387, 0.7361**
for 1, 2, 3, 4, 5+ prior visits — non-monotone, no gain from history. Patients
at 0%: 77 (8.2%); at 100%: 492 (52.7%); with a single prediction, so only 0 or
1 is attainable: 172 (18.4%).

Model comparison on the *same* 3,361 predictions: HMM 0.7539, discrete Markov
0.7474, CTMC 0.7453, persistence 0.7227, marginal null 0.6293.

### 2.7 The newer two-class experiments — all verified

**E17, CT-HMM next-visit (`experiments/20261020T104756Z_next_visit_td_pigd_prediction/`)**

899 patients, 3,029 eligible of 3,361 (332 Indeterminate targets excluded);
934/934 folds converged for every model, worst projected gradient 0.0252;
**19/19 checks pass**; **0 ties** in 3,029 × 7 models.

| Model | Accuracy | 95% CI | Brier | Log loss | PIGD recall |
|---|---:|---|---:|---:|---:|
| **CT-HMM** | **0.8369** | 0.8204–0.8541 | **0.1214** | **0.3961** | 0.667 |
| Direct logistic (+interval) | 0.8359 | 0.8195–0.8529 | 0.1225 | 0.4003 | 0.666 |
| Persistence | 0.8336 | 0.8173–0.8510 | 0.1598 | 4.1929 † | 0.623 |
| Markov (discrete) | 0.8293 | 0.8129–0.8468 | 0.1328 | 0.4314 | 0.589 |
| CTMC (observed) | 0.8277 | 0.8114–0.8451 | 0.1343 | 0.4346 | 0.580 |
| Direct logistic (no interval) | 0.8194 | 0.8019–0.8372 | 0.1244 | 0.4055 | 0.646 |
| Marginal null | 0.6983 | 0.6703–0.7278 | 0.2111 | 0.6134 | 0.000 |

† rule-based 0/1 probabilities, clipped; not comparable.

Paired contrasts (identical cases, patients resampled): latent state
**+0.0092 [+0.0003, +0.0179]**; elapsed time in the observed chain −0.0017
[−0.0039, +0.0003]; elapsed time in the discriminative family **+0.0165
[+0.0112, +0.0217]**; CT-HMM − direct +0.0010 [−0.0054, +0.0075]; **CT-HMM −
persistence +0.0033 [−0.0037, +0.0100]**; CT-HMM − null +0.1387.

Calibration mean absolute gap **0.0065** across five bins spanning 0–1. By
interval: on the ≤1.5 y bulk (n=2,597) CT-HMM and persistence are identical to
four decimals (**0.8417**); on >1.5 y (n=432) CT-HMM 0.8079 vs persistence
0.7847 vs CTMC 0.7477. By current label: Indeterminate **0.6074** vs the
chains' degenerate **0.5444** (+6.3 pp — the one place the latent state visibly
works). By history length: 0.834 / 0.864 / 0.827 / 0.810 / 0.832 — flat.

**E18, Rich clinical profile (`experiments/20261020T180515Z_clinical_profile_cv/`)**

4,433 visits / 1,080 patients (a visit needs ≥8 of 9 domains; 97.6% have all
nine); 934 patients with ≥2 → 3,353 predictions; 5-fold patient CV × 4 repeats
= 20 folds, 0 failed, **13/13 checks**.

Primary target (30,099 domain-observations), log-density per observation:
**−1.0453 [−1.0706, −1.0215]** (level+rate+medication), −1.0492 (level only),
−1.0511 (level+rate, no medication), −1.1187 (LOCF), −1.4342 (population mean);
RMSE 0.7007; 95% coverage **0.9406**.

Paired contrasts: vs LOCF **+0.0733 [+0.0657, +0.0813]** (75.1% of patients
better); vs population **+0.3889 [+0.3628, +0.4167]**; **rate component
+0.0040 [+0.0019, +0.0062]**; **medication +0.0058 [+0.0041, +0.0075]**.

Derived label on 3,020 matched cases (898 patients): profile level-only
**0.8440**, level+rate+med **0.8404**, level+rate no med 0.8404, **CT-HMM
0.8371**, persistence 0.8338, Markov 0.8298, LOCF 0.8252, population 0.6914.
Paired: profile − CT-HMM **+0.0033 [−0.0050, +0.0112]**; level-only − CT-HMM
+0.0070 [−0.0010, +0.0149].

Representation: **7 principal components for 90%** of across-patient variance;
PC1 42.2% (levels) / 37.2% (rates); participation ratios 4.34 / 5.11. Fitted
annual-rate spread: updrs2 0.375, PIGD 0.090, **tremor 0.0002 and MoCA — rate
innovation variance fitted to exactly zero**.

Independent outcomes (AUC of the single best coordinate vs the subtype label):
H&Y ≥ 3 **0.6845 vs 0.5794, +0.105 [+0.054, +0.156]**; MCI from HVLT/Benton
**0.7098 vs 0.5494, +0.161 [+0.109, +0.213]**; Part IV motor complications
0.5309 vs 0.4963, +0.035 [−0.017, +0.083]; investigator cognitive dx 0.6776 vs
0.5582, +0.119 [+0.069, +0.169] (**flagged as not fully independent**). Best
*rate* coordinates: +0.053, +0.045, +0.041, +0.036 — **all intervals include
zero**.

**E19, model selection and discrimination**

AIC/BIC over 4,295 label observations: CT-HMM LL −2969.06, k=14, AIC 5966.12;
Markov + initial state −3087.12, k=8, **ΔAIC +224.11, ΔBIC +185.92**; CTMC +
initial −3099.84, ΔAIC +249.56. Profile variants over 38,486 continuous
observations: full −45812.36, k=45, AIC 91714.73; level-only ΔAIC +220.81,
**ΔBIC +66.76**; no-medication ΔAIC +343.96, ΔBIC +266.94. Cross-family
comparison is **explicitly invalid** (probability mass vs density; different
observation counts) and the source report says so.

ROC on 3,020 matched cases (898 patients, 910 PIGD): **rich profile 0.8902
[0.8748, 0.9053]; CT-HMM 0.8583 [0.8411, 0.8755]; persistence 0.7927**. Paired
**+0.0319 [+0.0229, +0.0419]** — excludes zero. Stratified by current label the
profile leads in every stratum (TD 0.7798 vs 0.668; Indeterminate 0.6794 vs
0.6305; PIGD 0.7437 vs 0.6649) — **and the gap between pooled 0.86–0.89 and
stratified 0.63–0.78 shows most pooled discrimination is the current label
itself.**

### 2.8 Rich Clinical Profile — your provisional group figures

| Provisional claim | Verified value | Source |
|---|---|---|
| Mean accuracy 0.944 for the all-TD group | **0.9444** (510 patients, 1,461 targets, full-history condition) | `experiments/20261103T175003Z_patient_level_accuracy_audit/patient_accuracy_by_subtype_composition.csv` |
| Mean accuracy 0.469 for the even-mixture group | **0.4685** (60 patients, 188 targets) | same |
| Some history windows retain full-history performance at 2–3 visits | last-2 vs full: acc +0.0046 [−0.0031, +0.0123]; last-3: +0.0060 [−0.0020, +0.0138]; AUC +0.0001 at last-3 | `experiments/20261103T052237Z_profile_prediction_history/paired_comparisons.csv` |

Your eight verification questions, answered:

- **"All-TD" means** every one of that patient's *eligible target* visits has
  true label TD. **"Evenly split" means** equal numbers of TD and PIGD targets.
  Groups are defined from the **true** labels only, never from model output.
- They were **not** defined before evaluation. The audit says so explicitly:
  *"Patient subgroups here are not pre-specified strata… defined after seeing
  the predictions, as descriptive cuts."* **Exploratory.**
- The reported values are **mean per-patient accuracy** (patient-weighted); the
  same table also gives the visit-level (transition-weighted) figure — all-TD
  0.9644, even-split 0.5372. Both are available; the manuscript must say which.
- Group sizes: all-TD 510, majority-TD 105, even 60, majority-PIGD 61, all-PIGD
  162. **The even-split group has 60 patients and a median of 2 targets each**
  — thin, and the 0.469 figure is not a stable estimate.
- **The grouping does create evaluation bias**, mechanically: a patient whose
  labels never change is a patient a persistence rule gets right. Persistence
  scores **0.9697** on the all-TD group and **0.4799** on the even-split group —
  nearly the same spread. So the composition gradient is **a property of the
  label sequence, not of the model**, and must be presented as such. *This is
  the single most over-interpretable number in the project.*
- **No statistical test was run** across groups; no interval is reported for
  the difference. It must not be called significant.

### 2.9 History window and retraining — verified

**CT-HMM history window** (`experiments/20261024T042417Z_history_window_experiment/`),
LOPO 934 folds, **parameters fitted once per fold on full training sequences
and shared across all window conditions** — only the held-out patient's
prediction history is truncated. The full-history condition reproduces the
original run to max |ΔP_PIGD| = **1.1e-16** with identical hard calls. Tamper
test: across 934 folds and 11,644 cases, changing only the target label moved
the prediction by **exactly 0.0**. 20/20 checks.

Common-target cohort **734 targets / 228 patients** (n_history ≥ 5), identical
in every row:

| Condition | Accuracy | Balanced acc | PIGD sens | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|
| persistence | 0.8270 | 0.8089 | 0.7016 | n/a | 0.1730 |
| last 1 | 0.8106 | 0.7906 | 0.6721 | 0.8267 | 0.1458 |
| last 2 | 0.8283 | 0.8129 | 0.7213 | 0.8839 | 0.1266 |
| last 3 | 0.8351 | 0.8220 | 0.7443 | 0.8899 | 0.1215 |
| last 5 | 0.8338 | 0.8208 | 0.7443 | 0.8885 | 0.1211 |
| full history | 0.8324 | 0.8192 | 0.7410 | 0.8884 | 0.1214 |

Pre-specified primary **last 2 − full**: accuracy −0.0041 [−0.0189, +0.0100],
balanced −0.0063, AUC −0.0045, Brier +0.0051, log loss +0.0112 — **all five
contain zero**. Exploratory **last 1 − full**: accuracy −0.0218 [−0.0423,
−0.0014], balanced −0.0286, **AUC −0.0617 [−0.0858, −0.0385]**, Brier +0.0244,
log loss +0.0715.

**Profile 6A** (`20261103T052237Z`), 5-fold patient CV, parameters fixed per
fold with a checked parameter fingerprint, **28/28 checks** including a fold-0
refit that reproduced predictions exactly. Paired vs full history:

| Comparison | Accuracy | AUC | Brier | PIGD sens | TD spec |
|---|---|---|---|---|---|
| last 1 − full | **−0.0139 [−0.0221, −0.0057]** | **−0.0105 [−0.0150, −0.0062]** | **+0.0050** | **−0.0329 [−0.0499, −0.0162]** | −0.0057 |
| last 2 − full | +0.0046 [−0.0031, +0.0123] | −0.0026 [−0.0060, +0.0008] | +0.0005 | −0.0182 [−0.0362, +0.0000] | **+0.0156** |
| last 3 − full | +0.0060 [−0.0020, +0.0138] | +0.0001 [−0.0030, +0.0033] | −0.0000 | −0.0169 [−0.0346, +0.0000] | **+0.0188** |
| last 5 − full | −0.0027 [−0.0116, +0.0057] | +0.0007 | −0.0001 | **−0.0197 [−0.0377, −0.0061]** | +0.0093 |

**PIGD sensitivity is negative in all four windows.** The aggregate metrics are
flat because TD specificity rises to compensate and TD targets outnumber PIGD
roughly 2:1. Short history does not make the model worse; it makes it **more
conservative**.

**Retraining, CT-HMM** (`20261024T052343Z`), full 2×2 plus a disjoint-window
sensitivity, common cohort 2,192 targets / 742 patients, **24/24 checks**, both
earlier runs reproduced to ≤1.11e-16 with identical calls.

Pre-specified primary (window-trained, last 2 − full-trained, full history):
accuracy −0.0082 [−0.0193, +0.0026]; **balanced −0.0312 [−0.0441, −0.0183];
AUC −0.0210 [−0.0301, −0.0120]; Brier +0.0099; log loss +0.0397.**
Decomposition: **training effect −0.0248 balanced accuracy** [−0.0354, −0.0147]
with history fixed; **history effect −0.0064** [−0.0150, +0.0023] with training
fixed. The training effect is also present at full prediction history (−0.0203),
so it is a worse model, not a train/test mismatch. The window-trained model
**falls below persistence** on balanced accuracy (−0.0159 [−0.0290, −0.0028]).
Disjoint windows beat overlapping ones on AUC **+0.0156 [+0.0094, +0.0224]**,
so part of the penalty is the composite likelihood's double-counted emissions,
not the window width.

**Retraining, profile 6B** (`20261103T072507Z`), full 2×2 + C′/D′, matched
cohort 2,184 / 740. Primary D − B: accuracy −0.0032 [−0.0116, +0.0050],
balanced −0.0020, **AUC −0.0043 [−0.0078, −0.0009], Brier +0.0036 [+0.0016,
+0.0057], log loss +0.0065 [+0.0003, +0.0131]**, PIGD sensitivity **+0.0014
[−0.0161, +0.0194] — retraining does not recover it**. The B − A history
contrast reproduces 6A's PIGD sensitivity to four decimals (−0.0182), which is
two independently written scripts agreeing.

Objective accounting, fold 0: full sequences 747 prior : 2,671 transition terms
(1 : 3.6); overlapping windows 2,671 : 2,671 (1 : 1) with observation terms
inflated 3,418 → 5,342; disjoint 1,526 : 1,526 (1 : 1). **So neither window
objective is "the same likelihood on shorter sequences"** — this is a genuine
confound between "two-visit training" and "differently weighted objective" that
the design cannot separate, and the source report says so.

**Patient-level audit** (`20261103T175003Z`, analysis only, 0 models fitted,
**23/23 checks**, both source manifests re-verified 12/12 and 15/15):
visit-level **0.8391** vs mean patient-level **0.8342 [0.8166, 0.8519]** —
differ by 0.005. The distribution is **strongly bimodal**: **63.70%** of
patients correct on every eligible visit, **5.01%** on none, median **1.000**
(degenerate interval). 8.24% below 50%, 70.94% at ≥80%. Per-patient PIGD
sensitivity **0.5503** vs TD specificity **0.8583**; restricted to patients
with ≥2 targets of the class, **0.7222 vs 0.9170** and the "32.5% never
identified" figure falls to **6.2%** — so that headline is an artefact of
single-observation rates. Retraining leaves **660 of 740** (D vs B) and **821
of 898** (C vs A) patients bit-identical, with 36/44 and 34/43 improving and
worsening.

### 2.10 Stage-wise evaluation — verified

3,352 matched cases, 3 dropped for stage disagreement, **3,017 eligible / 896
patients**, 20/20 checks. Stage 1 (predict visit 2 from visit 1, n=835):
CT-HMM **0.8347 [0.8104, 0.8604]**, profile **0.8455 [0.8214, 0.8701]** —
against pooled 0.8373 and 0.8406. **Not one of the 22 accuracy
change-from-stage-1 intervals excludes zero, for either model, at any stage.**
CT-HMM balanced accuracy does rise with intervals excluding zero at stages 2,
6, 7, 8, 9 — but **the persistence rule's balanced accuracy rises from 0.7171
to 0.8162 over the same stages**, and the PIGD share of targets rises from 23%
to 48%. Within 148 patients eligible at every stage 1–5 the series is
non-monotone (up, down, up, down). Within-patient change, late minus early
(225 patients): CT-HMM **−0.0243 [−0.0749, +0.0284]**, profile **−0.0345
[−0.0893, +0.0198]** — more patients worsen than improve. Only stage-1 balanced
accuracy separates the models (+0.0460 [+0.0229, +0.0695] for the profile), one
of 24 uncorrected paired tests, and it is where the profile has the **least**
history.

### 2.11 Earlier results to be retired or relabelled

| Provisional claim | Verdict |
|---|---|
| "HMM outperforming CTMC in 10 out of 10 validation splits" | **Verified but mis-stated.** The 10/10 result is **HMM vs the *discrete* Markov chain**, +0.0284 ± 0.0094 log-likelihood per transition, on 10 independent 70/30 patient splits. The CTMC comparison is a separate, consistent finding. Fix the wording. |
| "Earlier HMM accuracy of approximately 77%" | **0.7695** on a *single* 70/30 split (1,002 pairs). **Superseded** by E7's LOPO 0.7539 on all 3,361 transitions. Do not report both as if the model improved. |
| "A previously reported HMM advantage of approximately 0.0284 over CTMC" | The **0.0284** figure is vs the discrete chain (see above). The HMM-vs-CTMC advantage on the single split is −0.6384 − (−0.6752) = **+0.0368**. Two different numbers; do not merge. |
| "Do not combine the older split-based result with the newer LOO result" | **Agreed and necessary.** Different evaluation designs, different n, different denominators. |
| "HMM's advantage over a lookup table runs +0.005 at one visit, +0.021 at two, +0.047 at six or more" | **Requires reproduction.** It appears only in `report/Project_Report.tex` prose and the README. **No CSV in `outputs/` contains it.** `outputs/loo/loo_accuracy_by_history.csv` has accuracy by history but not this contrast. |
| "100.00% / 99.87% agreement is external validation" | **False framing** — see §2.1a. |
| "Rich Clinical Profile is a validated clinical assessment tool" | **Not supported.** It is an unvalidated research representation. |
| "Rich Clinical Profile captures disease severity" | **Partly, and it is important that it is only partly.** 7 PCs for 90% of variance and participation ratio 4.34 mean it is *not* a single severity axis — which is the point. E16 shows a severity axis is exactly what fails. |

---

## OUTPUT 3 — Experiment reconciliation

### 3.1 What is directly comparable, and what is not

The project has **five distinct prediction tasks**. Numbers may be compared
*within* a block and must never be compared *across* blocks.

| Block | Target | Evaluation set | Metric scale | Members |
|---|---|---|---|---|
| **A. 3-class next label** | TD / Indeterminate / PIGD at the next labelled visit | **934 patients, 3,361 transitions** | 3-class accuracy; multiclass Brier | E7 (HMM 0.7539, DTMC 0.7474, CTMC 0.7453, persistence 0.7227, null 0.6293); E8 (5-state HMM 0.7513); **E12 (subscore GP 0.7566)**; E13 (joint LMM 0.7087); E14 (MOGP 0.7298–0.7587); E15 (spline 0.7093) |
| **B. 2-class next label** | TD vs PIGD, Indeterminate targets excluded | **≈899 patients, 3,020–3,029 eligible** | 2-class accuracy; 2-class Brier; AUC | E17 (CT-HMM 0.8369/0.8371); E18 (profile 0.8404); E19 (AUC 0.8583 vs 0.8902); E20; E21; E22 |
| **C. 2-class, latent-clinical-state subset** | TD vs PIGD | **654 patients, 2,091 scored** | 2-class accuracy | E16 (latent states 0.7255; baseline re-scored 0.8384; direct logistic 0.8307) |
| **D. Continuous ratio** | next visit's TD/PIGD ratio | **821 patients, 2,707 points** (unfiltered) | MAE / RMSE / CRPS in ratio units | E5 (hbayes), E9–E11 (GPR K0–K3) |
| **E. Continuous subscores / 9 domains** | next visit's measurements | **934 pt / 3,361** (subscores); **934 pt / 3,353** (9 domains, 30,099 obs) | MAE (subscores); log-density per domain-observation | E12, E13, E14, E15; E18 primary target |

**Block A ↔ Block B is the trap.** `manuscript/NOTE_accuracy_clarification.md`
already settles it, and it is correct: re-scoring E17's predictions under the
3-class rule reproduces E7 exactly —

```
3-class, all 3,361 cases:            2,534 / 3,361 = 0.7539
drop the 332 Indeterminate targets:  2,534 / 3,029 = 0.8366
renormalise TD/PIGD (flips a few):                   0.8369
```

**The 83.7% and the 75.4% are the same model.** The entire difference is the
removal of a class the model never predicts. Any manuscript sentence implying
improvement from 75% to 84% is wrong.

### 3.2 Validation reconciliation table

| # | Experiment | Model | Evaluation design | Cohort / eval unit | Metric | Result | Final status |
|---|---|---|---|---|---|---|---|
| E4 | Weeks 5–7 progression model | HMM / DTMC / CTMC | single 70/30 split + 10 independent splits | 325 test pt / 1,002 pairs | log-lik per transition; 3-class accuracy | HMM −0.6384 / 0.7695; **10/10 vs DTMC, +0.0284 ± 0.0094** | **Superseded on the headline by E7; the 10-split ranking is retained as the model-selection evidence** |
| E5 | Bayesian growth head-to-head | HMM vs HB growth | 6 independent splits, common transitions | 825 transitions | log-lik per transition | HMM −0.6559 / 0.7612; HB −0.6814 / 0.7370; **6/6, +0.0579** | **Complementary** — the HB model is the only interpretable progression rate; retain as supplementary |
| E7 | Patient-level LOO | HMM + 4 baselines | **LOPO, 934 folds** | 934 pt / 3,361 | 3-class accuracy, log-lik, Brier | **0.7539 [0.7348, 0.7731]**; Indeterminate recall 0.000 | **FINAL for Block A** |
| E8 | 5-hidden-state LOO | HMM, 5 latent states | LOPO, 934 folds | same 3,361 | accuracy, convergence | 0.7513; **404/934 converged** | **Final negative** — more states do not help and do not fit |
| E12 | Subscore GP baseline | 2 independent K1 GPs → derived subtype | LOPO, 934 folds × 2 | **934 pt / 3,361 — identical to E7** | 3-class accuracy, paired | **0.7566 vs HMM 0.7539, +0.0027 [−0.0042, +0.0098]** | **FINAL, and the only like-for-like cross-family comparison in the project** |
| E16 | Latent clinical states | CT-HMM + Gaussian emission | LOPO, 654 folds (280 pt spent on state selection) | 654 pt / 2,091 | 2-class accuracy, paired | **0.7255 vs 0.8384, −0.1130** | **FINAL negative**; its absolute numbers are **not** comparable to Block B because of the 654-patient subset |
| E17 | CT-HMM next-visit | CT-HMM + 6 candidates | **LOPO, 934 folds** | 899 pt / 3,029 | 2-class accuracy, Brier, log loss, calibration | **0.8369**; persistence 0.8336; calibration gap 0.0065 | **FINAL for Block B (subtype-history side)** |
| E18 | Rich clinical profile | per-domain Kalman level+rate | **5-fold patient CV × 4 repeats** | 934 pt / 3,353; 898 pt / 3,021 | log-density; derived 2-class | primary **+0.0733** over LOCF; derived 0.8404 | **FINAL for Block B (profile side) and Block E** |
| E19 | AUC / AIC-BIC | both | paired on identical 3,020 matched cases | 898 pt / 3,020 | AUC; AIC/BIC | **AUC +0.0319 [+0.0229, +0.0419]** for the profile; CT-HMM hidden layer ΔAIC 224 | **FINAL — the only measure that separates the two methods** |
| E20 | Stage-wise | both | regroup matched predictions by visit index | 896 pt / 3,017 | per-stage accuracy, cross-stage contrasts | stage-1 ≈ pooled; **no contrast excludes zero** | **Final** |
| E21 | History window | both | truncate prediction history, parameters fixed | 228 pt / 734; 898 pt / 3,021 | pre-specified primary + bootstrap | last-2 ≡ full; last-1 worse | **Final** |
| E22 | Retraining | both | full 2×2 training × history | 742 pt / 2,192; 740 pt / 2,184 | pre-specified primary | **worse on probability metrics**; damage from training | **Final negative** |

### 3.3 Fairness audit of the one head-to-head comparison that carries the paper's weight

CT-HMM (E17) vs rich profile (E18), on 3,020 matched cases:

| Dimension | CT-HMM | Rich profile | Same? |
|---|---|---|---|
| Evaluation cohort | 898 patients, 3,020 cases | identical | **yes** — matched on (patient, current visit, next visit) |
| Prediction target | next observed TD/PIGD label | identical, derived through the project's unmodified rule | **yes** |
| Prediction horizon | actual elapsed time to the next visit | identical | **yes** |
| Metric definition | 2-class accuracy / Brier / log loss / AUC | identical code path (`next_visit.two_class_metrics`) | **yes** |
| Aggregation | patient-clustered bootstrap, paired | identical | **yes** |
| Missing data | labelled visits only, no imputation | marginalised exactly, no imputation | **yes** |
| **Held-out scheme** | **LOPO, 934 folds** | **5-fold patient CV × 4 repeats** | **NO** |
| **Target-visit information** | timing only | timing **and the target visit's prescribed LEDD** | **NO** |

Both asymmetries are disclosed in the sources, and the second is **measured**:
swapping to the current visit's dose changes accuracy by −0.0013 [−0.0056,
+0.0026] (E21 §6), i.e. essentially nothing. The scheme difference is **not**
measured and cannot be removed by matching cases. LOPO trains on 933 of 934
patients; 5-fold trains on ~80%, so **the CT-HMM is the more favourably
validated of the two** and the profile's AUC advantage is, if anything,
conservative. State this; do not quietly equate the designs.

---

## OUTPUT 4 — Research gap

✅ **DONE 2026-11-04 — superseded by `03_LITERATURE_REVIEW_20261104.md`,**
which verified 14 references through Crossref and Europe PMC and derived the
gap from them. Four of the six claims this section said would need checking
held up; two were already covered by published work and the gap was narrowed
accordingly. The text below is the pre-review placeholder, kept for the record.

**I have not yet run the literature review.** Stage 3 of your workflow is a
verified PubMed/Crossref search with DOIs, and I will not fabricate citations
or a gap statement to fill this slot now. What follows is the *internal*
evidence a gap claim would have to rest on, plus the exact claims I will need
to check against the literature.

### 4.1 What this project establishes internally

1. The TD/PIGD label is **unstable at the visit level**: 35.2% of observed
   changes reverse immediately; requiring two consecutive assessments removes
   51.7% of apparent PIGD conversions.
2. The label's **Indeterminate band is nearly binary in practice**: half of
   Indeterminate visits sit at ratio 0.9091 and a quarter at 1.1364 — one
   hundredth from a boundary — because tremor is a mean of 11 integers and PIGD
   of 5.
3. **No model in five families recovers Indeterminate.** 3-class recall is
   0.000 for the HMM (1 prediction in 3,361) and 0.000 for the profile
   (2 in 3,352).
4. **Five model families converge.** On identical sets: HMM 0.7539, subscore GP
   0.7566 (+0.0027, CI spans zero), 5-state HMM 0.7513, MOGP 0.7298, joint LMM
   0.7087, persistence 0.7227. On the 2-class task: CT-HMM 0.8369, profile
   0.8404 (+0.0033, CI spans zero), persistence 0.8336.
5. **The binding constraint is measured, not asserted**: residual SD 1.138
   against a 0.25-wide band (4.6×); within-patient SD 1.22; ICC 0.536.
6. **Discretising a rich representation onto a severity axis destroys the
   signal** (−0.1130), while the same measurements without the bottleneck are
   +0.1052 better — because the label is a *scale-invariant ratio* and severity
   states are built from *levels*.
7. **A richer continuous representation is a better description of the patient
   without being a better predictor of the label**: 7 PCs for 90% of variance,
   and single coordinates beat the label by 0.105–0.161 AUC on prospective
   H&Y ≥ 3 and MCI, while the label-prediction difference is +0.0033 (CI spans
   zero).
8. **History beyond two or three visits is worth nothing** at prediction time,
   and **retraining to match a short window makes things worse**.

### 4.2 The claims Stage 3 must verify before any gap is asserted

| Claim the gap would rest on | What must be checked |
|---|---|
| TD/PIGD subtype instability is known but **rarely quantified as a reversion rate on consecutive visits** | Search for prior reversion/instability rates in PPMI or other cohorts (Stebbins 2013; Simuni et al. PPMI subtype papers; von Coelln et al.) |
| Prior work predicts subtype transitions but **does not report a persistence baseline** | Check every PD subtype-prediction paper for a trivial-baseline comparison |
| Prior work **does not distinguish observed labels from latent states**, nor report calibration | Check for calibration reporting in PD progression prediction |
| Prior work **does not use patient-level LOPO with a patient-clustered bootstrap** | Check evaluation designs |
| The **Indeterminate class is usually collapsed or dropped**, so its unpredictability is unreported | Check how each paper handles Indeterminate |
| **No prior study compares five model families on one protocol and one evaluation set** | This is the strongest candidate for the novel contribution, and the one most likely to need softening |

**Forbidden phrasings** already noted in your prompt apply. I would add: do not
claim the project is the first to apply an HMM to PD subtype progression —
hidden Markov and multistate models are well established in PD progression
modelling, and the contribution here is the **protocol and the negative
result**, not the model class.

---

## OUTPUT 5 — Scientific weaknesses

### Critical — would likely cause rejection if not fixed or disclosed

| # | Problem | Evidence | Fix |
|---|---|---|---|
| **C1** ✅ **CLOSED 2026-11-04** | **The transition analysis ignores within-patient clustering.** Wilson intervals in `outputs/transitions/report.txt` treat 3,361 pairs from 934 patients as independent. 476 patients contribute a constant sequence. | `outputs/transitions/report.txt` §1 | Done — see `02_ESSENTIAL_ANALYSES_E1_E2_E5_20261104.md` §E-2. Design effect exceeds 1 for 8 of 9 cells (median 1.20, max 1.46); the 35.2% reversion headline is unaffected (0.97× width). |
| **C2** | **The transition matrix pools unequal intervals.** TD→TD is 0.820 at ≤1.5 y and 0.578 beyond 4 y; the pooled value averages them in whatever proportion attendance produced. The project says so and then reports the pooled matrix as a headline. | same §2 | Present the interval-stratified matrix as primary, or present the pooled one explicitly as an attendance-weighted average. |
| **C3** | **The Bayesian model has no publishable MCMC diagnostics.** One chain, 2,000 retained draws, no R̂, no ESS, no multi-chain run, no prior sensitivity. | `src/ppmi_subtype/hbayes.py`, `outputs/hbayes/report.txt` §2 | Rerun with ≥4 chains and report R̂/ESS, or demote the model to a descriptive supplementary analysis with the diagnostics gap disclosed. **Essential-3.** |
| **C4** ✅ **CLOSED 2026-11-04, and re-labelled** | ~~The project contradicts itself on the residual-autocorrelation conclusion.~~ A **stale output file** contradicts the committed code, which is self-consistent. | `outputs/exploration/report.txt` §5 vs its own summary item 4 | Done — `02_ESSENTIAL_ANALYSES_E1_E2_E5_20261104.md` §E-1. §5 is correct; item 4 is stale and must not be quoted. No code change was needed. |
| **C5** | **Zero ethics, consent, funding, COI or author-contribution documentation exists anywhere in the repository.** A grep for ethics / IRB / consent / funding / competing interest returns nothing. | repository-wide grep | Obtain the PPMI DUA reference, the institutional position on secondary analysis of PPMI, and the current PPMI acknowledgement wording (the abstract has a `% TODO` for exactly this). **Essential-4.** |
| **C6** | **The 3-class label-agreement check cannot be reproduced** — `data/validation/` is empty. | directory listing | Restore the archived Apr-2020 file or disclose that the 99.87% figure is not currently reproducible. |

### Important limitations — must be disclosed, not fixed

| # | Limitation | Evidence |
|---|---|---|
| I1 | **No external validation.** Everything is within PPMI. LOPO is not external validation. | design |
| I2 ⚠️ **REWRITTEN 2026-11-04** | **Complete-case analysis under an untested MAR assumption.** 23.1% of visits are unlabelled. ~~Roughly half are exams that could not be fully scored, and in 1,243 of 1,253 the missing item is postural stability — a PIGD item — in patients with higher H&Y.~~ **The split does not hold.** Measured on the frozen panel: **1,195 of 1,333 (89.6%)** unlabelled visits had only an **ON-medication** Part III exam, which the OFF/untreated rule excludes; the partially-scored route is **57** visits (4.3%), not half. Within that route the report is right — 54 of 57 are missing postural stability, and those exams do come from more advanced patients (median H&Y **3.0 vs 2.0**, p = 6.4 × 10⁻¹⁸, reproducing the report's 1.4 × 10⁻¹⁷). Only its *size* is wrong. The lost visits are therefore disproportionately **treated patients later in follow-up** (96.8% on medication vs 57.9%; median year 5.0 vs 2.0), so the analysed sample is weighted toward earlier, untreated follow-up — which attenuates observed progression. **The conclusion that progression estimates are a lower bound survives, by a different and better-evidenced route.** No MNAR sensitivity analysis was run. | `02_ESSENTIAL_ANALYSES_E1_E2_E5_20261104.md` §E-5 |
| I3 | **The ratio is undefined for 725 of 4,441 labelled visits (16.3%), 665 of them TD.** Every ratio-based model (E5, E9–E11) silently loses them, so those analyses under-represent tremor-dominant visits. | `outputs/hbayes/report.txt` §1 |
| I4 | **Indeterminate targets (9.9%) are excluded from every two-class result.** The 0.8369 and 0.8404 headline figures describe 90.1% of prediction opportunities. | E17 §9 |
| I5 | **`P_TD` and `P_PIGD` are conditional on the next label being TD or PIGD.** They are not unconditional next-visit probabilities. A clinician reading P_PIGD = 0.7 is being told "*if* the next exam lands on TD or PIGD…". The unconditional value is ≈ 0.63 and **this project does not validate it.** | E17 §2 |
| I6 | **Both models are told when the next examination occurs**, and the profile is additionally told its prescribed LEDD. PPMI's visit schedule is not independent of disease course. The LEDD asymmetry is measured at ≈0 (−0.0013); the timing optimism is **not quantified**. | E17 §9, E21 §6 |
| I7 | **The target is this project's own classification rule applied to recorded items, not an adjudicated diagnosis.** Accuracy means agreement with a noisy record. | throughout |
| I8 | **Bounded integer-sum scales are modelled as Gaussian.** The profile's 95% intervals cover 94.1%; the GPR puts 12–13% of predictive mass below zero; the growth model puts 12.2% of 5-year mass on impossible negative ratios. | E9 §7, E18, `outputs/hbayes/report.txt` |
| I9 | **AIC/BIC assume independent observations** but visits are clustered within patients, so both penalties are optimistic; they come from a **single full-cohort fit** with no cross-validated uncertainty. | E19 §5 |
| I10 | **The two methods use different validation schemes** (LOPO vs 5-fold × 4). Matching cases removes the evaluation-set difference, not the scheme difference. | E19 §5 |
| I11 | **The latent clinical-state model's EM is not stably identified** — start-to-start log-likelihood spread exceeded 1,000 nats at 4, 5 and 6 states, median 399 and max 6,348 nats across folds. The *states* are one of several comparable optima and are correctly given no clinical names. | E16 §5, §9 |
| I12 | **The state count was never identified by the data** — the predictive criterion never turns over within the evaluable range and both the arithmetic maximum and BIC select the boundary (8). A pre-registered 1-SE rule chose 6; both were evaluated and the conclusion held. | E16 §5 |
| I13 | **The two window objectives are not the full-sequence likelihood.** The overlapping one double-counts interior visits (3,418 → 5,342 observation terms) and both reweight prior : transition terms from 1 : 3.6 to 1 : 1. "Two-visit training" and "differently weighted objective" are confounded by design. | E22 §6 |
| I14 | **Multiplicity is uncorrected nearly everywhere.** 24 paired tests in the stage-wise analysis, 15+15 intervals in the window experiment, 6 in the Indeterminate ratio analysis. Each source names its Bonferroni threshold and says which results would survive; the manuscript must carry that forward. | E20, E21, E18 |
| I15 | **Patient-level subgroups are post hoc**, defined after seeing predictions. | E22 audit §10 |
| I16 | **Latent states are not disease stages**, and the project is careful about this. Any drift toward staging language in the manuscript is unsupported. | E16 §7 |
| I17 | **The clinical-relevance AUCs are univariate with the coordinate chosen on the same data it is compared on**, so each Δ is an upper bound. | E18 §6.6 |
| I18 ⚠️ **NEW 2026-11-04** | **The five-year clinical output table is a latent-state quantity presented as a clinical one.** `hmm.clinical_table` returns latent-to-latent probabilities and its docstring says so, but the README and the Weeks 5–7 report quote it as "a patient currently Indeterminate has a 63% chance of being PIGD within five years, against 19% for a patient currently TD" — which a clinician reads as being about a *recorded* label. Recomputed through the emission matrix, the recorded-to-recorded figures are TD→PIGD **0.24–0.33** (quoted 0.19) and Indeterminate→PIGD **0.50–0.57** (quoted 0.63): the TD risk is understated and the Indeterminate risk overstated, by 5–14 points. The recomputation is also sensitive to the initial distribution π, which `outputs/model/` does not save. Presentation defect, not a modelling error. | `03_LITERATURE_REVIEW_20261104.md` §3.4 |

### Minor issues

- `src/ppmi_subtype/clinical_profile.py`'s module docstring gives the
  innovation covariance as `q·[[Δ³/3, Δ²/2], [Δ²/2, Δ]]`, omitting the
  `σ²_L·Δ` level-diffusion term that **the code actually implements** (line
  285) and that the existing manuscript states correctly. Stale docstring, not
  a scientific error — but fix it before anyone reproduces from the docstring.
- ✅ **CLOSED 2026-11-04.** `NP3PSTBL` sentinel-101 count is given as **4,302**
  in `config.yaml` and **3,206** in the README. Both are correct for their own
  denominator, now measured: 4,302 in the raw Part III file (39,310 rows),
  3,206 after the OFF/untreated selection (29,356 rows), 3,205 after duplicate
  resolution. State the denominator with the number.
- Undefined-ratio visit counts differ by cohort: **725 of 4,441** (full panel)
  vs **692 of 4,295** (the GPR/abstract cohort). Both appear in committed
  prose. State which cohort each belongs to.
- The 10-split result is described as "HMM beats CTMC 10/10" in your prompt and
  as "HMM beats the discrete chain 10/10" in the source. The source is right.
- `report/Project_Report.tex` has a fifth first-visit baseline ("carry forward,
  −1.4114") that the README table omits.

### Missing analyses (Audit D — information needed for publication)

| Item | Status |
|---|---|
| Dataset access documentation | **Partial** — `docs/DATA_REQUIREMENTS.md` names the exact LONI path and modules. No DUA reference number. |
| Participant inclusion / exclusion criteria | **Available** in `config.yaml` but **never written as a flow diagram**; no CONSORT/STROBE-style attrition figure exists. |
| Clinical assessment definitions | **Available** (item lists, scales, coverage). |
| Follow-up duration, number of visits | **Available and verified.** |
| Missing-data handling | **Implemented and documented**; **no MNAR sensitivity analysis** (I2). |
| Model architecture / configuration | **Fully available** in `config.yaml` + source. |
| Training procedure | **Available.** |
| Validation design | **Available**, with the two-scheme caveat. |
| Hyperparameter selection | **None was performed** — all fixed in advance. State this plainly; it is a strength for leakage and a limitation for performance. |
| Evaluation aggregation | **Both** transition-level and patient-level available for the newer runs; **only transition-level for E2 and E7's headline**. |
| Calibration | **Available** for the HMM (gap 0.0065), the profile (coverage 0.941) and the growth model (gap 0.105, systematically under-predicting PIGD in all five bins). |
| Confidence intervals | **Available**, patient-clustered, **except for the transition analysis** (C1). |
| **Ethics documentation** | **ABSENT** (C5). |
| **Data availability statement** | **Draftable** but needs the current PPMI wording. |
| **Code availability** | ✅ **CORRECTED 2026-11-04 — this entry was wrong.** The GPR modules (`gpr.py`, `gpr_loo.py`, `gpr_plots.py`, `trajectory.py`), scripts 11–16 and all three test files are **tracked**, added 2026-09-30 in `dada934` and already on `origin/main`. `git ls-files` confirms each one, `git status --short` is empty, and the repository has no untracked files at all. Nothing GP-related appears in `.gitignore`. `outputs/` and `experiments/` are gitignored because they hold participant-level derived data — a data-availability limit, not a code one. The claim was asserted here without checking and propagated to four other documents. The only open item is whether the repository will be public or archived with a DOI. |
| **Author contributions** | **Inferable** — two git authors (Mayank Tewatia; Parv, 23+9 commits) plus Dr. Sahasranand K. R. as supervisor on the abstract. Needs the authors' own CRediT statement. |
| **Funding** | **ABSENT.** |
| **Conflicts of interest** | **ABSENT.** |
| STROBE / TRIPOD+AI checklists | **Neither exists.** |
| Participant flow diagram | **Does not exist.** |
| Test suite status | **758 tests, all passing** (43 files, 617 s, `~/envs/ppmi/bin/python -m pytest tests/ -q`, run 2026-11-04). The README's "403 tests (2026-10-14)" is out of date. |

---

## OUTPUT 6 — Recommended analyses

**None of these has been performed.** Items already complete are listed
separately at the end.

### Essential — needed to support the central claim

| # | Analysis | Why | Cost |
|---|---|---|---|
| **E-1** ✅ **DONE 2026-11-04** | **Resolve the lag-1 residual contradiction.** | Settled: a stale output, not a code defect. `scripts/76`, 19/19 checks. | Done |
| **E-2** ✅ **DONE 2026-11-04** | **Patient-clustered bootstrap intervals for every transition proportion, the reversion rate and the KM curves.** | Done: `scripts/77`, 2,000 draws, 15/15 checks. Design effect > 1 for 8 of 9 cells; the reversion headline is unaffected. | Done |
| **E-3** | **Refit the Bayesian growth model with ≥4 chains; report R̂ and ESS; add one prior-sensitivity run.** | No reviewer will accept a single-chain Gibbs run with a half-vs-half mean check as convergence evidence. | Hours |
| **E-4** | **Assemble the regulatory block**: PPMI DUA reference, institutional position, current PPMI acknowledgement text, CRediT contributions, funding, COI. | C5. The paper cannot be submitted without it. | Depends on the institution |
| **E-5** ✅ **DONE 2026-11-04** | **Participant flow diagram**, every exclusion counted. | Done: `scripts/78`, 16/16 checks, figure `fig1_participant_flow.png`. Also corrected the README's account of label missingness. | Done |
| **E-6** ⚠️ **RESCOPED 2026-11-04** | **A missingness sensitivity analysis.** ~~Worst-case assumption that unlabelled exams with a missing postural-stability item would have scored PIGD.~~ E-5 showed that subgroup is 54 visits, not ~1,250, so that bound is now uninformative. The analysis that matters is a **treatment-driven** one: bound the drift under assumptions about what the 1,195 ON-medication-only visits would have scored OFF. | I2 remains the largest threat to validity and is still only argued, not bounded — but the mechanism to bound is now known. | 1 day |
| **E-7** | **STROBE and TRIPOD+AI checklists**, with each item marked complete / partial / missing / not applicable, and the descriptive-vs-predictive objectives separated. | Required; also the cleanest way to surface what is still missing. | 1 day |
| **E-8** 🆕 **2026-11-04** | **Harmonise the multi-family comparison.** Re-score every family's saved predictions onto one 3-class probability vector on the 3,361-transition set, then apply one decision rule, one baseline and one metric suite. | **Blocks the primary analysis.** The same model moves +2.7 accuracy points on the label-derivation rule alone, and the baseline moves +2.1, so no "beats persistence by X" claim is currently well defined across families. No refitting needed. | Hours |

### Strongly recommended — would substantially improve the manuscript

| # | Analysis | Why |
|---|---|---|
| **S-1** | **Report an operating point other than 0.5.** A threshold chosen *inside* the CV folds, targeting PIGD sensitivity. | Both methods miss ~a third of PIGD conversions; E19 §6 identifies this as the direct lever and it has never been pulled. |
| **S-2** | **Report mean per-patient accuracy alongside the transition-level figure for E7 and E17**, as the newer runs already do. | The patient-level audit showed the two agree to 0.005 for the profile; the same check has not been done for the CT-HMM's two-class figure. Cheap, and it closes an obvious reviewer question. |
| **S-3** | **A measurement-reliability analysis of the label itself**: how reproducible is the 3-class assignment given the discreteness of the two subscore means? | This is the paper's own central claim ("label reliability, not model capacity") and it is currently supported indirectly. E19 §6.7 recommends exactly this. |
| **S-4** | **An unconditional next-visit probability**, properly validated, instead of the conditional `P_PIGD`. | I5. The conditional probability is the single most misreadable output in the project. |
| **S-5** | **Commit the GPR modules** (or state in the code-availability section exactly which analyses are not reproducible from the public repository). | Honesty of the code-availability statement. |
| **S-6** | **Quantify the visit-timing optimism**: re-run E17 with the next interval replaced by the cohort median. | I6 is disclosed but never bounded, and it affects every interval-aware model. |

### Optional — useful context, not necessary

| # | Analysis | Why |
|---|---|---|
| **O-1** | Cross-domain innovation covariance in the profile model. | E19 §6.5 recommends it; the PCA shows cross-domain structure exists. 7 PCs for 90% means the gain is unlikely to be large. |
| **O-2** | Width-3 disjoint-window retraining. | Would separate "window width" from "double-counted emissions" in E22. The two source reports both call it unpromising. |
| **O-3** | A clinically meaningful endpoint (H&Y ≥ 3, MCI, Part IV) in place of the subtype label. | E19 §6.6's main recommendation and the obvious next paper — but it is **a different study**, not an addition to this one. See Output 7.3. |
| **O-4** | Run the threshold-sensitivity experiment properly and save it (§13 below). | The machinery exists; I recomputed the label side in minutes. It would become a one-paragraph sensitivity analysis. |
| **O-5** | A bounded or censored likelihood for the subscores. | I8; named as out of scope by the GPR audit and still open. |

### Already complete — do not re-run and do not imply these are new

Twenty-two experiments (E1–E22), 758 tests (all passing, re-run 2026-11-04),
and these validation batteries:
E7 10/10 checks; E17 19/19; E18 13/13; E19's three runs; E20 20/20; E21 20/20
(CT-HMM) and 18/18 (profile) and 28/28 (6A); E22 24/24 (CT-HMM) and 23/23
(patient audit); E9's 10 leakage checks on each of 4 cohorts; tamper tests in
E21 and E22 returning **exactly 0.0** with a companion liveness test proving
the code path is live.

---

## OUTPUT 7 — Primary scientific narrative

⚠️ **SUPERSEDED 2026-11-04 by `04_SCIENTIFIC_POSITIONING_20261104.md`.**
The recommendation below — build the paper around the CARE AI abstract's
thesis — survives and is confirmed. What changed is the reasoning and the
hierarchy: Stage 3 showed that subtype instability is established by four
prior studies, so it is background rather than the contribution, and
Stage 4 found that the primary analysis table is not yet comparable
across model families (E-8). Read the newer document for the decisions;
this section is kept for the record.

### 7.1 The recommendation

**Write the full-length version of the abstract the project has already
written.**

> **Primary research question.** Is longitudinal prediction of Parkinson's
> disease motor subtype limited by the choice of model family, or by the
> reliability of the label being predicted?

This is **Question A ∪ B** from your list, reframed so that the project's
negative results become the contribution rather than a series of apologies.

| Element | Choice |
|---|---|
| **Primary contribution** | Evidence that the binding constraint is the **label**, not model capacity: five model families spanning discrete chains, latent-state models, hierarchical Bayesian growth, Gaussian processes and a continuous multi-domain Kalman representation converge within a few points of each other and of a parameter-free persistence rule, under one protocol, on one evaluation set — and none recovers the Indeterminate class. |
| **Primary model** | CT-HMM with 3 latent states and a constrained misclassification matrix (E17), reported in both its 3-class (E7) and 2-class (E17) forms, with the arithmetic connecting them shown. |
| **Primary evaluation** | Leave-one-patient-out, 934 folds, patient-clustered bootstrap, with a persistence baseline in every table. |
| **Secondary analyses** | (i) observed-transition instability and reversion (E2, E3); (ii) the rich continuous profile as a *better description* that is not a better label-predictor (E18, E19); (iii) the like-for-like subscore-GP comparison (E12). |
| **Exploratory** | Patient-level heterogeneity by subtype composition (E22 audit); the Indeterminate boundary and focal-ratio analyses (E19). |
| **Sensitivity** | History windows (E21); short-window retraining (E22); 5 vs 3 latent states (E8); 6 vs 8 clinical states (E16); threshold narrowing (O-4, once run). |
| **Supplementary** | Hierarchical Bayesian growth model (E5), GPR line (E9–E11), joint LMM and MOGP (E13, E14), time-varying rate (E15), first-visit model (E6). |

### 7.2 Why this narrative is scientifically coherent

Five things make it one paper rather than a portfolio:

1. **One target.** Every main-text experiment predicts the next recorded
   subtype for the same patients. The target is measured, not model-generated,
   so it does not move when the model moves.
2. **One protocol.** Patient-level held-out validation, patient-clustered
   intervals, paired contrasts on identical cases, a persistence baseline
   everywhere.
3. **The negative results are not failures of execution — they localise the
   constraint, and each one localises it differently.** Raising latent states
   fails (E8). Discretising a rich representation fails, and the diagnostic
   says *why*: the label is a scale-invariant ratio, the states are built from
   levels (E16). Removing the bottleneck recovers +0.1052 (E16). A GP on the
   subscores lands on the same number as the HMM (E12). More history is worth
   nothing (E20, E21). Retraining to match a short window makes it worse (E22).
   Together these are a **convergent argument**, and convergence across
   deliberately dissimilar model classes is much stronger evidence about the
   data than any one model's accuracy.
4. **The constraint is quantified independently of any model.** Residual SD
   1.138 against a 0.25-wide band; within-patient SD 1.22; ICC 0.536; 35.2%
   immediate reversion; half of Indeterminate visits one hundredth from a
   boundary. These are properties of the measurement, visible before fitting.
5. **The one positive finding fits the same frame.** The rich profile is a
   better description of a patient — 7 PCs for 90% of variance, single
   coordinates beating the label by 0.105–0.161 AUC on prospective H&Y and MCI
   — and is *still* not a better predictor of the label (+0.0033, CI spans
   zero). That is the thesis restated from the other side: **the label, not the
   representation, is what is thin.**

### 7.3 What should NOT be in this paper

| Excluded | Why |
|---|---|
| Any clustering content | Your instruction, and §0. |
| The claim that the CT-HMM or the profile is clinically useful | Nothing was validated against a clinical decision, and PIGD recall is ~0.67 with a third of conversions missed. |
| The AUC separation as the headline | It is real (+0.0319, CI excludes zero) and belongs in Results — but §3.3 shows the CT-HMM is the more favourably validated model, the stratified AUCs are 0.63–0.78 against a pooled 0.86–0.89, and accuracy, the metric a clinician would act on, shows nothing. |
| Clinical outcome prediction (H&Y, MCI, Part IV) as a main result | The outcome AUCs are univariate with the coordinate chosen on the same data (I17). They are strong enough to motivate the next paper and too weak to carry this one. **This is the obvious second manuscript**, and it should be a prospective-design paper, not an appendix here. |
| The first-visit model as a main result | It answers a different question (no history) on a different design (5 splits, horizon-indexed rows). Supplementary. |
| "75% → 84% improvement" in any form | §3.1. Same model, different task. |

### 7.4 The alternative, if you prefer a methods venue

A narrower, purely methodological paper is also defensible: *"Evaluation design
determines every conclusion in longitudinal subtype prediction."* It would be
built from the aggregation artefacts the project found and corrected — the
pooled-vs-per-patient sign flip in E9 §3.2, the 3-class/2-class arithmetic in
§3.1, the stage-wise composition confound in E20, the cohort-shrinkage artefact
in E21 §6.1, and the visit-level-vs-patient-level regrouping in E22. That is
unusually clean methodological material. **I recommend against it as the
primary paper**, because the clinical finding about label reliability is the
more useful contribution and reaches the audience that could act on it.

---

## OUTPUT 8 — Journal-specific outline

### 8.1 Journal fit — reassessed, and I disagree with the original target

I have **not** checked current official instructions (that is Stage 3 / Stage 20
of your workflow and must be done against live journal pages before
submission). What follows is a fit assessment from the evidence.

| Journal | Fit | Assessment |
|---|---|---|
| **npj Parkinson's Disease** | **Strong — my recommendation** | PD-specific, publishes PPMI secondary analyses and negative/methodological findings, open access, and accustomed to progression-modelling work. The paper's contribution is clinically framed (label reliability) with methodological substance, which is this journal's centre. |
| *Movement Disorders* | Moderate | Higher clinical bar; a paper whose headline is "a persistence rule is as good" with no external validation and no clinical endpoint will struggle against its novelty expectations. |
| *Parkinsonism & Related Disorders* | Moderate–good | A realistic fallback; comfortable with cohort-analysis and measurement-reliability papers. |
| *IEEE JBHI* | Moderate | Would want the methodological framing (§7.4) and more model novelty than this project claims. |
| *Artificial Intelligence in Medicine* | Moderate | Same. The five-family convergence would read as an evaluation study, which it is. |

**Do not over-fit to a venue.** The project's defining property is that it
reports honest nulls across five model families. A journal that requires a
positive methodological result is the wrong home, and softening the nulls to
reach one would destroy the contribution.

### 8.2 Outline (npj-style: Abstract → Introduction → Results → Discussion → Methods)

| Section | Claim | Evidence | Figure / Table | Placement |
|---|---|---|---|---|
| **Title** | Label reliability, not model capacity, limits longitudinal prediction of PD motor subtype | — | — | — |
| **Abstract** | Unstructured; verify the current word limit | E2, E7, E12, E16, E17 | — | — |
| **Intro** | The subtype is a thresholded ratio; its reliability is rarely measured before it is predicted | Stage 3 literature + Stebbins 2013 | — | — |
| **R1. Cohort and label construction** | 1,080 patients, 5,774 visits, 4,441 labelled; the rule reproduces PPMI's own derivation exactly | E1 | **Fig 1** flow diagram (**to be built, E-5**); **Table 1** cohort characteristics | Main |
| **R2. The observed label is unstable** | 35.2% of changes reverse; Indeterminate persists 19.2%; sustained definition removes 51.7% of conversions; transitions scale with interval | E2, E3 | **Fig 2** transition matrix + reversion + interval strata; **Table 2** counts and proportions **with patient-clustered intervals (E-2)** | Main |
| **R3. Five model families converge on the 3-class task** | HMM 0.7539; subscore GP 0.7566 (+0.0027, CI spans zero); 5-state HMM 0.7513 (43% convergence); MOGP 0.7298; joint LMM 0.7087; persistence 0.7227. **Indeterminate recall 0.000.** | E7, E8, E12, E13, E14 | **Fig 3** forest plot of all families on the identical 3,361 points; **Table 3** model comparison | Main |
| **R4. A latent state helps a little, exactly where theory says** | +0.0092 [+0.0003, +0.0179] over the identically interval-aware CTMC; +6.3 pp where the current label is Indeterminate; ΔAIC 224 for the hidden layer | E17, E19 | included in Fig 3 | Main |
| **R5. Restricting to TD-vs-PIGD changes the task, not the model** | 0.7539 → 0.8369 is the removal of 332 unpredictable targets; persistence reaches 0.8336; calibration gap 0.0065 | E17, `NOTE_accuracy_clarification.md` | **Table 4** two-class comparison incl. persistence | Main |
| **R6. Discretising a rich representation destroys the signal; keeping it continuous does not restore label prediction** | −0.1130 for latent clinical states; +0.1052 for the same measurements without the bottleneck; profile +0.0033 over CT-HMM (CI spans zero) but AUC +0.0319 (CI excludes zero) | E16, E18, E19 | **Fig 4** latent-state failure mechanism (severity axis vs ratio) + ROC | Main |
| **R7. The richer representation is a better description than a better predictor** | 7 PCs for 90%; single coordinates beat the label by +0.105 (H&Y ≥ 3) and +0.161 (MCI); **all rate coordinates' intervals include zero** | E18 | **Fig 5** PCA + outcome AUCs | Main |
| **R8. History adds nothing beyond two to three visits, and retraining to match a short window hurts** | last-2 ≡ full on every metric; last-1 worse; retraining −0.0312 balanced accuracy and below persistence; stage 1 ≈ pooled; no cross-stage accuracy contrast excludes zero | E20, E21, E22 | **Fig 6** history-window + stage-wise | Main (condensed) |
| **R9. Who the model gets wrong** | bimodal per-patient accuracy (63.7% all-correct, 5.0% all-wrong); per-patient PIGD sensitivity 0.72 vs TD specificity 0.92 among patients with ≥2 targets of the class; composition gradient 0.944 → 0.469, **matched by persistence 0.970 → 0.480** | E22 audit | **Table 5** exploratory, flagged | Main, explicitly exploratory |
| **Discussion** | The constraint is the measurement; what a 0.25-wide band inside a ratio of two integer means can support; what follows for progression-endpoint design | all | — | Main |
| **Methods** | §22.1–22.16 of your prompt, written from source | source + config | — | Main |
| **Supplementary** | S1 Bayesian growth model (E5) · S2 GPR line (E9–E11) · S3 joint LMM / MOGP / time-varying rate (E13–E15) · S4 first-visit model (E6) · S5 Indeterminate boundary and focal ratios (E19) · S6 full history-window and retraining tables (E21, E22) · S7 threshold sensitivity (O-4) · S8 STROBE + TRIPOD+AI checklists · S9 all 758 tests and every validation battery | | | Supplementary |

**Figure count: 6 main + 5 main tables.** Candidate Figure 8 (Bayesian
population trajectory) moves to Supplementary with the model. Candidate Figure
7 (patient heterogeneity) becomes Table 5 because its message is a set of
numbers, not a shape — except the bimodal histogram, which is genuinely a
shape and can join Figure 6.

---

## OUTPUT 9 — Drafting plan

### 9.1 Write in this order

| Order | Section | Why first / why here | Blocked by |
|---|---|---|---|
| **1** | **Methods §22.2–22.5** (dataset, participants, variables, subtype definitions, longitudinal construction, transition analysis) | Every number in the paper is downstream of these definitions, and they are fully verified. Writing them first freezes the terminology. | E-5 (flow diagram) for the participant counts figure; otherwise nothing |
| **2** | **Methods §22.6–22.15** (CTMC, HMM, growth model, profile, probabilistic prediction, windows, retraining, validation, statistics) | All implementation details are verified. Must be written before Results so metric definitions are fixed. | **E-1** (the residual-autocorrelation contradiction) blocks the growth-model subsection; **C3/E-3** determines whether the growth model is inferential or descriptive |
| **3** | **Results R1–R2** | Cohort and label instability; the foundation of the argument. | **E-2** (patient-clustered intervals for the transition tables) |
| **4** | **Results R3–R5** | The convergence result; the paper's core. | Nothing — all verified |
| **5** | **Results R6–R9** | Representation, history, heterogeneity. | Nothing — all verified |
| **6** | **Introduction** | Only writeable once the gap is verified against literature. | **Stage 3** |
| **7** | **Discussion** | Needs both Results and the literature comparison. | Stage 3 |
| **8** | **Abstract and Title** | Last, so they describe what the paper says. The abstract in `report/CARE_AI2026_Abstract.tex` is the starting point. | 1–7 |
| **9** | **Data / code availability, ethics, contributions, COI, funding, AI disclosure** | Mechanical once the facts exist. | **E-4** |
| **10** | **STROBE + TRIPOD+AI checklists; submission audit** | Last. | E-7 |

### 9.2 Must be resolved before drafting starts

| # | Blocker | Who resolves it |
|---|---|---|
| **B1** | The lag-1 residual contradiction (§2.5). Which reading is the project's? | **You** — it is a modelling judgement, and both readings are in your own output |
| **B2** | Is the latent clinical-state study inside or outside the clustering exclusion (§0)? | **You** — it changes whether R6 exists |
| **B3** | Does the Bayesian growth model get multi-chain diagnostics (E-3), or is it demoted to descriptive supplementary? | **You** — a compute/scope decision |
| **B4** | Ethics, funding, COI, CRediT, PPMI DUA reference, current PPMI acknowledgement wording | **You and your supervisor** — I cannot invent any of it |
| **B5** | Author list and order. The abstract says Parv Patel, Mayank Tewatia, Dr. Sahasranand K. R. | **You** |
| **B6** | Will the GPR modules be committed, or will the code-availability statement exclude them? | **You** |
| **B7** | Is the archived Apr-2020 validation file recoverable (`data/validation/` is empty)? | **You** |

### 9.3 What I will do next, on your go-ahead

- **Stage 2** is already substantially complete above (§3.1–§3.3). I can extend
  it into the full per-experiment reconciliation sheet as a CSV if you want it
  machine-checkable the way `manuscript/make_tables.py` checks the existing
  manuscript's numbers.
- **Stage 3** (literature review) is the natural next step, and it is the one
  thing in this plan I have not started. It needs live PubMed/Crossref
  searching with DOI verification for every entry.
- Alternatively I can start **E-1, E-2 and E-5** immediately — they are cheap,
  they unblock three sections, and E-2 fixes the one statistical defect in the
  paper's foundational result.

---

## 13. Addendum — the threshold-sensitivity ("relabelling") experiment

Your prompt describes a relabelling experiment with these figures: 6.7% of
labelled visits relabelled, all from Indeterminate, Indeterminate shrinking
~68%, and agreement with PPMI falling from 100.00%/99.87% to 97.74%/92.86%.

**What it actually is.** `config/config_narrow_0.91_1.13.yaml`, committed
2026-09-11 in `89b22e4`, is byte-identical to the baseline config except for
four values: `td_threshold: 1.13`, `pigd_threshold: 0.91`,
`band_boundaries: inner`, and the output directory. **No model is involved.**
It is a pure **threshold sensitivity analysis** — a change to the classification
rule, not model-based relabelling of visits. Your prompt's framing ("which
model produced the relabelled visits", "whether the relabelling was based on
model predictions") does not apply, and the manuscript must not describe it as
model-derived relabelling.

**No run directory and no report exist.** `experiments/` contains nothing
matching `threshold` or `narrow`, and no doc reports its results. The figures
in your prompt therefore came from a session whose output was not persisted.

**I recomputed the label side directly from the frozen panel** (script in the
scratchpad; the baseline rule first reproduced the panel's own label on all
4,441 labelled visits, as a control):

| Quantity | Provisional | **Recomputed** | Verdict |
|---|---|---|---|
| Share of labelled visits relabelled | 6.7% | **298 / 4,441 = 6.71%** | **Verified** |
| All relabelled visits came from Indeterminate | yes | **yes — all 298** (198 → PIGD, 100 → TD) | **Verified** |
| Indeterminate shrinkage | ~68% | **67.6%** (441 → 143) | **Verified** |
| 2-class agreement after relabelling | 97.74% | **4,331 / 4,431 = 97.74%** | **Verified** |
| 3-class agreement after relabelling | 92.86% | **cannot be computed — `data/validation/` is empty** | **Requires the reference file** |

Also recomputed: TD grows 2,891 → 2,991 (+3.46%), PIGD 1,109 → 1,307 (+17.85%).

**Why this matters scientifically**, and it is worth a paragraph in the paper:
moving each threshold by about 0.02 reclassifies **two thirds of the
Indeterminate class** and costs 2.26 points of agreement with PPMI's own
label. That is because the ratio is a quotient of a mean of 11 integers and a
mean of 5, so inside a 0.25-wide band only a handful of values are reachable —
half of all Indeterminate visits sit at exactly 0.9091 and a quarter at 1.1364
(E19). The Indeterminate class is therefore **not robust to a threshold
perturbation smaller than the measurement's own resolution**. This is direct,
model-free support for the paper's central claim, and it is the cheapest
remaining analysis in the project (O-4: run script 01 with `PPMI_CONFIG`
pointed at the narrow config, once `data/validation/` is restored).

**Do not** use it to claim PPMI's labels are wrong. Both label sets apply a
rule; one rule is published and one is a perturbation of it. Neither is
adjudicated truth.

---

## Provenance of this audit

Read and verified: `README.md`; `config/config.yaml`,
`config/config_narrow_0.91_1.13.yaml`; `src/ppmi_subtype/{subtype,hmm,hbayes,
validate,clinical_profile}.py`; `outputs/{panel_summary.txt,validation.txt,
visit_panel.csv}`; `outputs/{transitions,model,hbayes,exploration,validation,
newpatient,loo}/report.txt`; `outputs/loo/loo_overall_metrics.json`;
`outputs/model/{markov_property_test,hmm_emission,heldout_performance}.csv`;
`outputs/{joint_subscore,joint_subscore_gp}/{metrics,subtype_metrics}.csv`;
`outputs/loo_5hidden/loo_predictions_all_models.csv`;
`docs/{DATA_REQUIREMENTS.md,GPR_onwards_explanatory_report.md}`;
`docs/gpr_audit/{FINAL_RESULTS,REPOSITORY_INVENTORY,NEXT_RESEARCH_STEPS}.md`;
all of `docs/{next_visit_prediction,latent_clinical_states,model_comparison,
stage_wise_evaluation,history_window}/`; `manuscript/{MANUSCRIPT.md,README.md,
NOTE_accuracy_clarification.md}`; `report/CARE_AI2026_Abstract.tex`;
`report/Project_Report.tex` (§first-visit); and the run directories for
E17, E18, E19, E20, E21, E22 including `checks.json`, `summary.json`,
`model_comparison.csv`, `contrasts.csv`, `auc_*.csv`, `aic_bic.csv`,
`outcome_associations.csv`, `representation_dimensionality.csv`,
`patient_accuracy_by_{subtype_composition,visit_count}.csv`.

Recomputed: the threshold-sensitivity label contrast (§13).

Not inspected: `docs/cluster_phase/` (52 files) and the clustering experiment
runs, by instruction.
