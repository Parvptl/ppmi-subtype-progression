# Stage 2: integer-score measurement model. Data audit and proposed specification

**Date:** 2026-10-17.
**Status:** **FROZEN 2026-10-17**, after reviewer approval; the freezing record is §D. Its SHA-256 is stored in `docs/cluster_phase/_frozen/16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.sha256`, and an identical copy is kept in `_frozen/`. Any later change is a dated amendment in a separate file. At the time of freezing, no M2 model had been fitted and nothing had been simulated.

**Inputs:**
- `13_D1_POST_RUN_AUDIT.md` (D1 audit, Stage 1–3 plan, latent-state criteria);
- `14_D1_STAGE1_CHECKS_20261016.md` (Stage 1);
- `15_D1_REPLICATE_LABEL_DIAGNOSTIC_REVIEW.md` §8, the exploratory replicate-label diagnostic;
- `05_FINDINGS.md` Stage 6b (earlier sum-level and item-level PIGD models).

**Data audit run:** `experiments/20261016T194345Z_stage2_data_audit/`.
- Produced by `scripts/44_stage2_data_audit.py`, which is descriptive only.
- It reuses `raw_scores` (reconstruction exact at 5,774 visits) and `stage1_checks.direct_change_outcomes`.
- Three earlier runs are kept under `experiments/_smoke/superseded_20261016T1941*` and `…T194313Z*`, each with a `SUPERSEDED.txt`:
  - the first failed at serialisation;
  - the next two carry a direct-change indexing bug, found and fixed before this document was written. Their direct-change outputs are wrong and are not used. The cited run cross-checks the full-label S3 against `label_statistics`: 0.3486, equal to Part C's 0.349.

**Standing points:**
- The D1 verdict, **NOT REPRODUCED**, stands and is not revisited.
- The replicate-label diagnostic (doc 15) is **exploratory and uncalibrated**.
- Stage 2 outputs will live in their own directories and never in a D1 directory.

| tag | meaning |
|---|---|
| [O] | observed count from the audit run |
| [C] | read from code or configuration |
| [R] | earlier report |
| [H] | assumption or hypothesis |

---

# Part A. Data audit (Step 1)

## A1. How the scores are built [C: `subtype.py`, `raw_scores.py`, `config/config.yaml`]

**Pipeline:** `loaders` (MDS-UPDRS Part II and III files) → `select_medication_state` → `deduplicate_visits` → `clean_items` → `compute_subscores` → `classify`.

| element | rule |
|---|---|
| Tremor items (11) | NP2TRMR (Part II) and Part III: NP3PTRMR/L (postural), NP3KTRMR/L (kinetic), NP3RTARU/LU/RL/LL/LJ (rest amplitude, 4 limbs and jaw), NP3RTCON (constancy) |
| PIGD items (5) | NP2WALK, NP2FREZ (Part II); NP3GAIT, NP3FRZGT, NP3PSTBL (Part III) |
| Valid values | integers 0–4. Anything else, including the 101 "unable to rate" code, becomes missing |
| Subscore | mean of the items, **set to missing if any item is missing** (`max_missing_* = 0`) |
| Exam selection | OFF exam, or untreated (`PDTRTMNT == 0`), or `PDSTATE` absent. Duplicates are resolved by fewest missing items, then first occurrence |
| Label | for PIGD > 0: ratio = (T/11)/(P/5); TD if ≥ 1.15, PIGD if ≤ 0.90, Indeterminate otherwise ("outer" band) |
| | P = 0 and T > 0 → TD; T = P = 0 → Indeterminate |

T is the tremor item sum and P the PIGD item sum.

**Consequences:**
- Every labelled visit has all 16 items: 0 missing, 0 out of range, 0 non-integer at 4,441 labelled visits [O].
- The integer sums are therefore exactly defined, with T ∈ {0,…,44} and P ∈ {0,…,20}.
- The label is a deterministic function of (T, P), and the full lattice is `raw_scores.label_grid`. No ratio is needed to label a visit: the PIGD = 0 cases are rules on the integers.

## A2. Cohort, ranges, missingness and medication [O]

**Labelled visits:** 4,441 labelled visits from 1,080 patients in the frozen sporadic-PD panel.

**Proposed primary set:** patients with ≥ 2 labelled visits, i.e. **934 patients and 4,295 visits**.
- Labels: TD 2,793, PIGD 1,071, Indeterminate 431.
- **PIGD = 0 at 692 visits:** 634 TD by rule, 58 both-zero Indeterminate. 349 patients have at least one such visit.
- T = 0 with P > 0 at 250 visits.

**D1's ratio-defined cohort,** for comparison: 821 patients and 3,528 visits.

**Unlabelled panel visits (1,333)**, all excluded:
- no Part III exam left after the OFF/untreated filter: 686;
- Part III items absent: 559;
- a 101 code: 52;
- no item rows: 36.

**Ranges and associations:**

| | observed range | possible | zero share | median (10th–90th) |
|---|---|---|---|---|
| T | 0–37 | 0–44 | 0.072 | 7 (1–14) |
| P | 0–19 | 0–20 | 0.161 | 1 (0–4) |

- Visit-level Spearman correlation of T and P: **0.05**.
- **Item heterogeneity** (`A1_item_categories.csv`). Positive rates range from 0.06 to 0.75 for the PIGD items, with NP3FRZGT the lowest and NP3GAIT the highest, and from 0.09 to 0.84 for the tremor items, with NP3RTALJ lowest and NP2TRMR highest.
- **Within-subscore association** (Spearman):
  - PIGD items: 0.28–0.41.
  - Tremor items: median 0.18, range −0.03 to 0.57. The highest pairs are rest-amplitude with constancy (RTARU–RTCON 0.57, RTALU–RTCON 0.51) and the kinetic and postural pairs (0.48). This is evidence of **local dependence** within tremor.
- **Within-patient dispersion** (patients with ≥ 3 visits). The residual variance of T is 1.83× the variance a binomial on 44 points would give at the patient's mean; for P it is 1.96×. This is over-dispersion relative to independent item draws.

**Medication.**
- `PDSTATE`: OFF at 2,569 visits, absent (untreated) at 1,724, ON at 2.
- The untreated share falls with time: 0.997 at year 0, 0.53 at year 1, 0.29 at year 2, 0.12 at year 4, below 0.05 after year 6.
- 776 consecutive pairs span the start of treatment (0 → 1).
- An "OFF" exam in a treated patient is a practically defined OFF, and it differs from an untreated exam [R: Part C §5, OFF/ON pairs].

**Timing.** The median inter-visit interval is 1.00 year (10th–90th 0.91–2.00). There are 3,361 consecutive pairs.

## A3. Changes between consecutive visits [O]

| | 5th / 25th / 50th / 75th / 95th percentile | skewness |
|---|---|---|
| ΔT | −5 / −2 / 0 / 2 / 6 | **0.01** |
| ΔP | −2 / 0 / 0 / 1 / 3 | **1.61** |

- corr(ΔT, ΔP) = 0.07.
- **PIGD floor.** From P = 0, the next P is 0 in 277 of 603 pairs, i.e. 0.459, which equals the Stage 6b value. It is 1 in 213 pairs and ≥ 2 in 113.
- **Reading.** Tremor changes are symmetric. PIGD changes are right-skewed, with a floor and upward drift. Any direction asymmetry in label transitions must come mainly from P.

## A4. The two targets on the full label [O]

**Target 1: Indeterminate share by P stratum, full label.**

| P | 0 | 1 | 2 | 3 | 4–5 | 6+ |
|---|---|---|---|---|---|---|
| visits | 692 | 1,459 | 1,083 | 444 | 324 | 293 |
| Indeterminate | 58 (both zero) | 72 | 150 | 69 | 59 | 23 |
| share | 0.084 | 0.049 | 0.139 | 0.155 | 0.182 | 0.078 |

At P ≥ 1 these are close to the D1-scope values (doc 15 §8: 0.050, 0.138, 0.152, 0.178, 0.079). The P = 0 stratum is new: its Indeterminate visits are both-zero by rule.

**Target 2: direct TD ↔ PIGD changes, full label** (`A5_direct_changes_full_label.csv`):

| direction | revert | stay | to_Ind | unobserved | R_D |
|---|---|---|---|---|---|
| TD → PIGD | 55 | 92 | 25 | 81 | **0.320** (55/172) |
| PIGD → TD | 44 | 54 | 14 | 29 | **0.393** (44/112) |
| both | 99 | 146 | 39 | 110 | 0.349 |

- **The asymmetry persists on the full label,** though it is smaller than in D1's scope (0.25 vs 0.40).
- **By whether the patient ever has P = 0:**
  - never: TD → PIGD 24/109 (0.22), PIGD → TD 28/69 (0.41). These sequences are identical in both scopes, and the total, 52/178, equals Stage 1 Check 4.
  - at least once: TD → PIGD 31/63 (0.49), PIGD → TD 16/43 (0.37).
- **Triplets that include a P = 0 visit:** TD → PIGD reverts 21/35; without one, 34/137. P = 0 visits therefore carry a high share of TD → PIGD → TD reversions, and D1's scope drops them.
- **Drivers of TD → PIGD changes** (253):
  - both T and P change in 187, P alone in 27, T alone in 39;
  - medians ΔT −3, ΔP +2;
  - 63 of them span the start of treatment.
- **TD → PIGD reversion by the sign of ΔP:** 42/139 when P rose, 12/29 when P was unchanged. This is descriptive only, since the cells are small.

## A5. What is identifiable from repeated item responses [C/R/H]

**Identifiable** (with the usual caveats):
- **Item parameters.** Discriminations and thresholds of a graded-response model (GRM), from the cross-sectional item covariance. Stage 6b fitted the 5 PIGD items without bound hits [R].
- **Patient level and slope** per latent dimension. Slopes are weak at typical visit counts: reliability 0.27 for tremor and 0.53 for PIGD at the median [R: Stage 6b B].
- **A visit-level latent deviation shared by the items of one subscore.** It is separated from item-level noise because the items of one visit co-vary beyond what the patient trend explains. The 1.8–2× over-dispersion (A2) is the signal.
- **Correlation between the tremor and PIGD latents,** at patient level (trend) and visit level.

**Weakly or not identifiable:**
- **Serial correlation beyond the patient trend.** An Ornstein–Uhlenbeck (OU) term substituted for the random intercept and slope (ℓ = 13 years, intercept–slope correlation 0.998) [R: Stage 6b C5].
- **Rater effects and test–retest error.** There are no rater IDs or replicate exams; OFF/ON pairs are not test–retest [R].
- **Item drift over time or with treatment.** A stage-1 refit by period differed, but the difference is confounded with progression [R].
- **A discrete intermediate latent state versus continuous heavy-tailed variation.** Without strong structural assumptions these are not separable. That is why criteria for a latent-state model are stated separately (B14).
- **Tremor local dependence** (A2). A unidimensional GRM ignores it. A testlet or bifactor structure is identifiable with 11 items, but it adds parameters.

## A6. Item sums or item responses?

**Item responses are needed for the first version.**
- **For PIGD this is established.** Sum-level Gaussian, heteroscedastic and beta-binomial models all failed the P = 0 share: 0.233–0.240 against 0.161. The beta-binomial collapsed to a binomial (φ ≈ 7,554). The item-level GRM passed it: 0.175 [R: Stage 6b]. The audit shows why: item positive rates span 0.06–0.75.
- **For tremor, a sum-level model is not ruled out,** since the sum has broad support. But its items are as heterogeneous (0.09–0.84) and locally dependent, so the first version models items for both subscores, on the same footing.
- **The data allow this.** Item responses are complete at every labelled visit (A1).

## A7. How the model would reproduce score pairs and transitions [H]

- **Lattice and label.** Generating integer items makes every replicate (T, P) a lattice point. The labels then come from `classify` exactly as for observed visits, so band-capture effects (doc 14 §2.3) and P = 0 rules are reproduced by construction, with no ratio arithmetic. Whether Indeterminate shares match then depends only on whether T near the band is right *given* P. That is the substantive test of target 1.
- **Score pairs.** Two latent dimensions (θT, θP) give T and P their own marginal shapes (zeros, floors, item mix). The weak visit-level association (0.05) is carried by a correlated visit deviation, and between-patient association by correlated random effects.
- **Transitions.**
  - P's right-skewed changes arise from the GRM floor, its non-linearity and upward drift.
  - T's symmetric changes arise from symmetric latent noise.
  - Direction-specific reversion is then a *consequence*, not a fitted feature. If TD → PIGD changes are mostly persistent PIGD increases, a model with a PIGD trend will reproduce fewer TD → PIGD reversions than a symmetric-ratio model. Whether it does is target 2.

---

# Part B. Proposed specification (Step 2)

## B1. Scientific question and estimands

**Question.** Does a measurement-faithful model with **no intermediate state** reproduce two features of the observed TD / Indeterminate / PIGD label sequences? It generates integer MDS-UPDRS tremor and PIGD item responses from continuous patient-level latent processes and applies the unchanged label rule.
- **Target 1 (prevalence):** the Indeterminate share by PIGD-sum stratum.
- **Target 2 (transitions):** direct-reversion behaviour by direction.

The two targets are **separate questions**. Each has its own calibrated statistic, threshold, positive control and verdict. Neither verdict is combined with the other, and no single model change is assumed to explain both.

**Estimand.** For each pre-specified statistic: its distribution under the fitted model, conditional on each patient's own item data and visit schedule (patient-conditional posterior-predictive replicates, as in D1), compared with the observed value.

## B2. Cohort and visits

**Primary cohort.**
- The frozen sporadic-PD panel (MD5 `fb038ddb…`), labelled visits only (all 16 items valid), patients with ≥ 2 labelled visits: **934 patients and 4,295 visits**, PIGD = 0 visits included.
- Time is years from baseline.
- The visit schedule is the observed one and is copied into every replicate.

**Folds.** 5 disjoint whole-patient folds, stratified by first label, seed 42. They are stored with their hash in the Stage 2 run directory; this is a new fold file, not D1's.

**Secondary scope (D1 comparison).** For each observed and replicate sequence, the subsequence of visits with P > 0 in that sequence, restricted to patients with ≥ 2 such visits. This is D1's construction applied identically to replicates.

**Excluded, with reasons listed in A2:** unlabelled visits, the genetic-PD cohort, and patients with one labelled visit.

## B3. Observation and missing-data model

**Observed data.** The 16 item responses at each analysed visit are complete, so no item-level missingness enters the likelihood. The code supports missing items, which would contribute no likelihood term, but none occur in the analysed visits.

**Missing visits.** Visit-level missingness (unlabelled visits, dropout) is **not modelled**. The working assumption is that, given a patient's latent trajectory, whether a visit is labelled does not depend on the unobserved item values [H, MAR-type].
- This is known to be imperfect, because attrition is informative [R: Stage 5].
- Replicates reuse each patient's observed visit times, so schedules are identical by construction, as in D1.
- This is a stated limitation (B13).

## B4. Model M2 (candidate equations)

**Items.** A graded-response model per subscore s ∈ {T, P}, item k, category c = 1…4:

  P(Y_ijk ≥ c | θ_ij^s) = expit(a_k (θ_ij^s − τ_kc)),  a_k > 0,  τ_k1 < … < τ_k4.

Items are conditionally independent given θ^s; this is the primary assumption, with a tremor testlet as a sensitivity (B12). The 55 item parameters are 11 × 5 for tremor and 5 × 5 for PIGD.

**Latent processes,** for patient i and visit j:

  θ_ij^s = β0^s + β1^s t_ij + γ^s x_ij + b_i0^s + b_i1^s t_ij + ε_ij^s.

- x_ij = 1 if the patient is treated at the visit (`PDTRTMNT == 1`), else 0.
- (b_i0^T, b_i1^T, b_i0^P, b_i1^P) ~ N₄(0, D), with D a full 4 × 4 covariance.
- (ε_ij^T, ε_ij^P) ~ N₂(0, Σ_ε), independent across visits, with Σ_ε a full 2 × 2 covariance.
- The **19** longitudinal parameters are β (4), γ (2), D (10) and Σ_ε (3). The proposal text said 25 by an arithmetic slip, corrected before freezing.

**There is no intermediate state, no mixture and no serial correlation term.** OU was weakly identified in Stage 6b.

**Identification.**
- Stage 1 fixes each latent's location and scale through θ ~ N(0, 1) per visit in the item fit (as in Stage 6b).
- Stage 2 estimates β, γ, D and Σ_ε on that scale with the item parameters held fixed.
- The visit deviations ε are identified because they are shared by the items of a subscore (A5).

**Integer scores and labels.** Generating a visit means drawing θ^T and θ^P, then each item from its GRM category distribution. Then:
- T = Σ tremor items, P = Σ PIGD items;
- the label = `subtype.classify` on (T/11, P/5), unchanged, including the P = 0 rules;
- no ratio is formed for P = 0;
- PIGD = 0 and both-zero visits arise when all five PIGD items (and, for both-zero, all eleven tremor items) are drawn as 0. They are not imposed.

## B5. Estimation, and why it cannot be tuned to the targets

**Stage 1: item parameters.** Marginal maximum likelihood with θ ~ N(0, 1) per visit, using 61-node Gauss–Hermite quadrature and analytic gradients. This is `grm.fit_items`, run per subscore.

**Stage 2: longitudinal parameters.** Marginal likelihood over each patient's (b_i, ε_i), whose dimension is 4 + 2 n_i, by the Laplace approximation with an exact gradient. The GRM log-likelihood is concave in θ, so each patient's mode is unique.
- Accuracy is checked against adaptive Gauss–Hermite quadrature on 30 randomly chosen patients (B11).
- **Parameter uncertainty:** β, γ, D and Σ_ε are drawn from N(estimate, inverse observed information). Item parameters are held at their stage-1 values. Ignoring that item uncertainty is a stated limitation.

**Guard against tuning.** Only the item-response likelihood enters estimation.
- **The evaluation statistics are computed only in the final, once-only observed stage, after the fidelity gates, threshold and power have been frozen.** They are never used for estimation, model choice or gate tolerances.
- The model structure (B4), its one pre-declared fallback (B12, tremor testlet) and every rule here are fixed in this document before any fit.
- No parameter, covariate or structure may be changed after any evaluation statistic is seen. A changed model is a new, separately specified, exploratory analysis.
- The runner enforces this order with the D1 safeguards (`ppc_runner` pattern): frozen files, locks, a started marker and an exactly-once observed stage.

## B6. Fidelity gates (label-free, before any evaluation statistic)

Each gate is computed on 40 datasets simulated from the full-cohort fit, compared with the observed data, and has a tolerance fixed here.
- *Simulation.* Each dataset is generated at the observed visit schedule from b_i ~ N(0, D), fresh ε and fresh items.
- *Statistic.* The observed value minus the simulated mean.

| gate | target | tolerance |
|---|---|---|
| G1 | the 16 × 5 item category proportions | each within 2.5 × its Monte Carlo SD across the 40 datasets, or 0.02 absolute, whichever is larger |
| G2 | P = 0 share (0.161) and T = 0 share (0.072) | ±0.03 and ±0.02 |
| G3 | P-sum distribution (0, 1, 2, 3, 4–5, 6+) and T-sum distribution (bins 0, 1–3, 4–5, 6–8, 9–11, 12+) | each bin within 2.5 × its Monte Carlo SD, or 0.02 |
| G4 | PIGD floor persistence: P(next P = 0 \| P = 0) = 0.459 and P(next P = 0 \| P = 1) | ±0.05 and ±0.035 (Stage 6b tolerances) |
| G5 | consecutive-pair changes: SD of ΔT and ΔP; skewness of ΔP (1.61); Spearman(ΔT, ΔP) | within 2.5 × the Monte Carlo SD each |
| G6 | within-patient dispersion ratios (1.83 tremor, 1.96 PIGD) | ±0.3 |
| G7 | visit-level Spearman(T, P) (0.05) | ±0.05 |

**Rule:**
- G1–G3 and G5–G7 must all pass before either target is evaluated.
- **G4 must additionally pass for target 2.** P = 0 visits carry a large share of TD → PIGD reversions (A4), and the item-level GRM previously failed G4 (0.396 vs 0.459; 0.177 vs 0.109) [R].
- If G4 fails, target 2's verdict is **"UNINFORMATIVE (measurement-fidelity gate failed)"**.
- If any other gate fails, neither target is evaluated. The fit is reported, and Stage 2 stops before calibration.
- No gate is evaluated on labels. G3 uses P-sum bins that coincide with target 1's strata, but only their *marginal frequencies*, never the T distribution given P near the band, which is what target 1 tests.

## B7. Pre-specified statistics (computed identically on observed and replicate sequences)

**Conventions.**
- Strata use **each sequence's own P sums**: generated P for replicates, observed P for the data.
- Direct changes and outcomes use `stage1_checks.direct_change_outcomes`: revert, stay, to_Ind, unobserved. R_D = revert / (revert + stay + to_Ind).

**Family A (target 1, prevalence):**
- A1–A6: the Indeterminate share in P strata 0, 1, 2, 3, 4–5, 6+;
- A7: the overall Indeterminate share (D1's S1 on the full label).

**Family B (target 2, transitions):**
- B1: R_D for TD → PIGD;
- B2: R_D for PIGD → TD;
- B3: entry reversion (D1's S2);
- B4: Indeterminate clustering, observed/expected (D1's S5).

**Descriptive only** (no decision):
- the revert/stay/to_Ind/unobserved counts per direction;
- D1's S3, S4 and S6;
- every statistic in the secondary D1 scope (B2).

**Discrepancy, as in D1.**
- Mid-p p̃ = (#rep > obs + ½ #= + ½)/(R + 1), with R = 500 and z = Φ⁻¹(p̃).
- T_A = max |z| over A1–A7; T_B = max |z| over B1–B4.
- Undefined statistics follow D1 B.14.2.

## B8. Posterior-predictive strategy

1. Five-fold cross-fitting. Stages 1 and 2 are fitted on the four training folds.
2. For each test patient, R = 500 replicates. For replicate r:
   - draw the global parameters from their approximate posterior;
   - draw (b_i, ε_i) given the patient's own items, by elliptical slice sampling from the Gaussian prior, 20 sweeps after initialising at the Laplace mode;
   - draw **fresh** ε* and fresh items at the patient's own visit times;
   - label the result.
3. Each patient's data is used twice (to condition the replicates and as the comparison), which makes the p̃ values conservative (D1 B.3). The calibration (B9) absorbs this.

## B9. Calibration, power, Monte Carlo uncertainty, decision rule

**Null cohorts.**
- N₀ = **500** null cohorts are generated from the full-cohort fit: parametric bootstrap with b_i ~ N(0, D), fresh ε and items at the real visit times, and global parameters drawn from the approximate posterior.
- Each null cohort goes through the identical pipeline (B5, B8, B7).
- c_A = T_A₍ₖ₎ and c_B = T_B₍ₖ₎ with k = ⌈0.95 (N₀ + 1)⌉, i.e. the 476th of 500, frozen before any positive control.
- **Fallback:** if the timed smoke projects more than 48 node-hours, N₀ = 200 (k = 191). This is decided and recorded before any observed computation.
- The z cap is 3.091 at R = 500. A threshold at the cap means the calibration is degenerate (D1 B.11.2).

**Monte Carlo uncertainty.**
- Each threshold gets a distribution-free 95 % interval for the 0.95 quantile from order statistics.
- The 1,000 null and 100 per-control cohort results are kept, with checksums.

**Positive controls** (100 cohorts each, fixed now; planted only in the generator):
- **PC-A** (target 1, PC4 analogue). With probability π = 0.30 per patient, an episode starts at a uniform time within follow-up and lasts Exp(mean 2 years).
  - During the episode the visit's (θ^T, θ^P) is replaced by a fixed "balanced" point (θ*^T, θ*^P). This point is chosen, from the stage-1 item parameters alone, so that E[T]/11 ÷ E[P]/5 = 1.02 at E[P] = 3.
  - The visit deviation is scaled by κ = 0.25.
- **PC-B** (target 2). With probability π = 0.30 per patient, θ^P increases at a uniform time by δ = 1.0 × SD(ε^P) and stays raised: a persistent PIGD step.
- **Power.** P_lo is the two-sided 95 % Clopper–Pearson lower bound of the detection rate (T > c). It must be ≥ 0.50, i.e. ≥ 61 of 100.

**Verdicts** (per family, D1 rules B.11.5 / B.14.2):

| verdict | condition |
|---|---|
| NOT REPRODUCED | T_obs > c |
| REPRODUCED (within scope) | T_obs ≤ c and P_lo ≥ 0.50 |
| UNINFORMATIVE (low power) | T_obs ≤ c and P_lo < 0.50 |
| UNINFORMATIVE (measurement-fidelity gate failed) | target 2 only, when G4 fails |

- The observed cohort is evaluated **once, last**.
- The two families are separate questions, each at a nominal 5 %. No combined claim is made.

## B10. Comparison with D1 (descriptive)

**Scope matching.** In the secondary scope (B2), compute D1's S1–S5 on observed and Stage 2 replicate sequences, with D1's mid-p and z. These are reported beside D1's z values (B.15) and the doc 15 splits, without a threshold.

**Questions:**
- Does measurement fidelity alone remove D1's S1 misfit within D1's scope?
- Does it remove the S3 / TD → PIGD misfit within that scope?

**The D1 verdict is not revisited.**

## B11. Convergence and model checks (before the gates)

| item | requirement |
|---|---|
| Item fit | L-BFGS-B convergence; no bound hits; ≥ 3 random starts within 0.1 log-likelihood |
| Longitudinal fit | gradient norm < 1e-3 × n; no bound hits; positive-definite Hessian; ≥ 3 starts agree |
| Laplace accuracy | maximum per-patient log-likelihood error against adaptive Gauss–Hermite (7 nodes per dimension where feasible, otherwise importance sampling with 2,000 draws) on 30 random patients ≤ 0.05 |
| Patient sampler | on 30 random patients, 4 chains: split-R̂ < 1.01 and effective sample size ≥ 100 for the patient-level T and P sums and labels |
| Simulation recovery | on 5 datasets from known parameters, every global parameter's 95 % interval covers the truth in at least 4 of 5 |
| Differential item functioning (reported, not gated) | stage-1 refits on untreated vs treated visits and on years 0–1 vs ≥ 3 |

A failure here stops Stage 2 before the gates. The failure is reported, and any repair is a dated amendment made before any evaluation statistic exists.

## B12. Sensitivity analyses (descriptive; run only after the primary verdicts)

1. **Tremor testlet.** One extra visit-level factor shared by NP3RTARU/LU/RL/LL/LJ and NP3RTCON. Pre-declared because of the local dependence in A2. If the primary model fails G1, G3 or G5 *for tremor only*, this replaces it **before** calibration, as the one permitted fallback, recorded and re-gated.
2. **Student-t patient slopes,** ν estimated, matching D1's TT. Fit and statistics only, uncalibrated.
3. **Transition subsets.** Treatment-stable pairs (excluding the 776 that span the start of treatment) and intervals ≤ 1.5 years, for family B statistics.
4. **Part III-only labels,** derived from the same generated items (10 tremor and 3 PIGD items). This needs no refit.
5. **The secondary D1 scope** (B10).
6. **One missing PIGD item allowed** (43 extra observed visits). Exploratory, with a note on the imputation assumption.

## B13. Computational cost and known limitations

**Cost** (estimate; a timed smoke confirms it before the full run):
- The PIGD-only GRM predictive check took 51 s in total on one node [R].
- The bivariate model with visit deviations is larger: about 10–60 s per fit, plus replicates.
- Full run: (1 + 500 + 200) cohorts × 5 folds × (fit + replicates). That is about 10–60 CPU-hours, or about 1–3 hours on one 32-core node. The fallback in B9 applies if needed.
- **Smoke.** Synthetic cohort only, seed tag 91, no observed evaluation (D1 B.12.3 rules).

**Limitations:**
- Two-stage fitting ignores item-parameter uncertainty.
- The Laplace approximation.
- Gaussian latent effects (B12.2 is the only check).
- No serial correlation.
- Informative dropout is not modelled.
- Copied visit schedules.
- OFF-exam heterogeneity under treatment is reduced to a single treatment effect γ.
- No rater information.
- Possible item drift.
- Conditional independence of tremor items (B12.1).
- Patient-conditional p̃ values are conservative.

**Not confirmatory in a strict sense.** The strata and the direction split were suggested by D1 and Stage 1 results on these same data. Stage 2 is therefore a confirmatory test *of the model*, not an independent test of the hypotheses. No external replication cohort is available (genetic PD is a different population). Verdicts are reported with this caveat.

## B14. When a latent-state model should be tested next

**All of the following must hold:**
1. **Fidelity and power.** M2 (or its declared testlet fallback) passes every fidelity gate, the calibration is not degenerate, and PC-A gives P_lo ≥ 0.50.
2. **Family A NOT REPRODUCED with the intermediate-state signature.** The observed Indeterminate share is above the replicates *and* Indeterminate visits cluster or persist beyond the replicates: B4 (S5) z ≤ −1.96, or S6 above the replicate 97.5 %. This must hold on the full label, not only in D1's scope.
3. **The misfit is not localised to a measurement feature.** The replicates reproduce the T distribution given P at the lattice cells *outside* the band within each P stratum (descriptive, for P = 1–5: within the replicate 95 % range). That is, the misfit is specific to Indeterminate cells, not a general misfit of T given P.
4. **Robustness.** The pattern keeps its direction under the testlet, Student-t, treatment-stable and Part III-only sensitivities.
5. **Held-out comparison, specified separately in advance.** For example, a hidden Markov or mixture model on the same items, compared with M2 under a pre-specified rule. Its states are not interpreted as subtypes without independent evidence.

**Family B alone does not qualify.** A family B failure (direction-specific reversion), however strong, points to the latent *dynamics* (PIGD steps, treatment effects, non-linear progression), not to a third state.

---

## C. Recommendation

**The available data support a first measurement-level model at the item level.**
- Item responses are complete at every labelled visit and are exact integers.
- The label is a deterministic function of the sums, so PIGD = 0 and both-zero visits need no ratio.
- The item-level GRM has already reproduced the PIGD = 0 share where sum-level models could not.
- A pure item-*sum* model is not recommended: for PIGD it is contradicted by Stage 6b.

**Uncertainties to resolve before implementation** (in order). Each can be answered without computing any evaluation statistic:

1. **PIGD floor persistence (G4).** The earlier GRM, with and without OU, failed it. If M2 fails it too, target 2 is uninformative by rule. The fix is a dynamics question. A small fit-and-gate pilot (B6 only, no labels) should run first.
2. **Tremor local dependence.** Will the unidimensional tremor GRM pass G1, G3 and G5, or is the testlet fallback needed?
3. **Fit cost and Laplace accuracy** in 4 + 2 n_i dimensions per patient (B11). This decides between N₀ = 500 and 200.
4. **Treatment.** Is one effect γ enough? 63 of 253 TD → PIGD changes span the start of treatment. The DIF check in B11 is diagnostic only.
5. **Scope of target 2.** The TD → PIGD reversion deficit is strongest in D1's scope (0.25) and in patients who never reach P = 0 (0.22). On the full label it is 0.32 against 0.39 for PIGD → TD. Target 2's primary scope is the full label, and D1's scope is secondary. Please confirm this choice before the specification is frozen.

**Proposed next step, on approval:** freeze this document (record its SHA-256). Then implement M2 with unit tests and simulation-recovery tests, and run only the **fit-and-fidelity-gate pilot** (B11 + B6, label-free). Report it before any calibration or evaluation statistic is computed.


---

## D. Freezing record and operational details (2026-10-17, before any M2 computation)

### D1. Reviewer decisions (2026-10-17)

- **Target 2 primary scope:** full three-state labels, including PIGD = 0 and both-zero visits. **Secondary scope:** D1's ratio-defined cohort (B2, B10).
- The prevalence (family A) and transition (family B) questions stay separate, each with its own statistics, threshold, positive control and verdict (B1, B9).
- **Next step approved:** implement M2 with unit and simulation-recovery tests, then run **only** the synthetic recovery and the label-free fit-and-fidelity-gate pilot.
  - The 500-cohort calibration, the positive controls and every evaluation statistic need a separate review.
  - If G4 fails, the pilot stops and target 2 is reported uninformative under B6.

### D2. Corrections made before freezing

- §B4: the count of longitudinal parameters is 19, not 25.
- No other change was made to Part B.

### D3. Operational details fixed now (they fill gaps in B5, B6 and B11; nothing in them uses labels)

**Parameterisation.** ψ = (β0T, β1T, γT, β0P, β1P, γP, the 10 entries of the lower-triangular L with D = L Lᵀ, the 3 entries of L_ε with Σ_ε = L_ε L_εᵀ).
- Diagonal entries of L and L_ε are stored on the log scale.
- Bounds: means ±10; log-diagonals [−8, 3]; off-diagonals ±5.
- A bound hit means the parameter lies within 1e-6 of a bound.

**Treatment indicator.** x = 1 if `PDTRTMNT == 1`, otherwise 0. The latter includes 6 visits where it is missing (B4 as written).

**Item fit (B11).** Per subscore, L-BFGS-B from `grm.initial_items` plus 2 further starts, each perturbed by N(0, 0.3²) on every parameter (seeds `[42, 45, 5, k]`).
- "Within 0.1 log-likelihood" means max − min of the three final negative log-likelihoods ≤ 0.1.
- A bound hit for the unbounded item fit means a discrimination outside [0.05, 20].

**Longitudinal fit (B11).** Three starts: a fixed default, and the default plus N(0, 0.2²) perturbations (seeds `[42, 45, 6, k]`).
- "Agree" means max − min of the final negative log-likelihoods ≤ 0.1.
- The gradient criterion is ‖∇‖ < 1e-3 × (number of patients).
- The Hessian is the central finite difference (step 1e-4) of the analytic gradient. It must be positive definite.
- The reported fit is the start with the lowest negative log-likelihood.

**Laplace accuracy (B11).** 30 patients drawn at random (seed `[42, 45, 4]`). The reference is:
- for 2n ≤ 6: adaptive Gauss–Hermite with 7 nodes per dimension;
- otherwise: importance sampling with 20,000 draws from a multivariate t₅ proposal centred at the mode with the Laplace covariance. The Monte Carlo SE is reported. (The proposal's 2,000 draws were replaced before freezing, to keep the Monte Carlo SE well below the 0.05 tolerance.)

**Patient sampler (B11).** The same 30 patients and the plug-in fitted parameters. Elliptical slice sampling on θ_i given the prior N(M_i, K_i). 4 chains: 200 warm-up and 1,000 kept iterations each, started at mode + N(0, 4 × Laplace covariance).
- Monitored quantities: the four components of b_i (by their exact Gaussian conditional mean given θ_i), and the patient's expected tremor and PIGD sums summed over visits, Σ_j E[T_ij | θ_ij] and Σ_j E[P_ij | θ_ij].
- Split-R̂ < 1.01 and bulk ESS ≥ 100 for every monitored quantity and patient.
- **Labels are not monitored in the pilot,** to keep it label-free. B11's "and labels" is deferred to the calibration stage.

**Simulation recovery (B11).** 5 datasets (seeds `[42, 45, 1, r]`) on the real visit design (times and treatment indicator of the 934 patients), from **a-priori** truth values fixed in `scripts/45_m2_pilot.py` before any real-data fit.
- The pipeline is the full two-stage one (item fit, then longitudinal fit).
- 95 % Wald intervals on ψ come from the finite-difference Hessian.
- Pass: every component of ψ is covered in ≥ 4 of 5 datasets.
- Item-parameter recovery is reported (correlation and maximum error) but not gated.

**Fidelity gates (B6).** The 40 datasets use seeds `[42, 45, 2, r]`, plug-in fitted parameters, b_i ~ N(0, D), fresh ε and items at the observed design.
- *Monte Carlo SD:* the SD of a feature across the 40 datasets.
- *G4 pairs:* consecutive analysed visits of a patient, any interval.
- *G5 ΔT, ΔP:* the same consecutive pairs.
- *G6:* computed as in `scripts/44_stage2_data_audit.py`, for patients with ≥ 3 visits.
- *G1:* covers all 80 item-category proportions.
- No label is computed for observed or simulated data.

**Seeds and run.** The pilot runs as one SLURM job through `scripts/cluster/submit.sh`, single-threaded BLAS, in its own directory `experiments/<UTC>_stage2_m2_pilot/`. It refuses to run if this file's SHA-256 differs from the frozen record.
