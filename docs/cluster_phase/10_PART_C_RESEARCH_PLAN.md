# Part C — raw-score trajectories and the Indeterminate class: audit and research plan

**Status:** PROPOSAL, revision 2, 2026-10-16. Nothing has been implemented, run or committed. Awaiting approval of the items in §11.

**Repository state:** branch `main` at `3a860ca`, two commits ahead of `origin/main`. Frozen panel `outputs/visit_panel.csv`, MD5 `fb038ddb…`. This file is the only change in the working tree.

**Evidence tags** used throughout:

| tag | meaning |
|---|---|
| **[V]** | verified in this audit, by reading code or querying data read-only through the pipeline's own functions |
| **[R]** | stated in an earlier report or output; not re-derived |
| **[L]** | external literature (sources in §3.1) |

**Decisions already taken (2026-10-16):**

- **C1** = selection and missingness audit plus descriptive figures. A model comparison only if justified and not a duplicate; see §4.3.
- **C2** = Indeterminate transitions, underlying score changes, persistence and reversion, all with patient-bootstrap uncertainty.
- Existing scripts, outputs, labels, frozen specifications and conclusions are preserved.
- No commits or pushes.
- The four open questions were decided as recorded in §3.

---

## 1. What already exists

### Phenotype construction

| path | role |
|---|---|
| `src/ppmi_subtype/subtype.py` | `clean_items`, `compute_subscores`, `classify`, `select_medication_state`, `deduplicate_visits` |
| `config/config.yaml` (`subtype:`, `cohort:`) | item lists, thresholds, sentinel codes, medication state, cohort filter |
| `src/ppmi_subtype/panel.py` | `build_subscores`, `filter_cohort`, `build_visit_panel`, `_add_time_axis` |
| `src/ppmi_subtype/loaders.py` | raw-file loaders |
| `scripts/00_data_audit.py`, `scripts/01_build_panel.py` | data audit, panel build |
| `tests/test_subtype.py`, `tests/test_validate.py` | thresholds, zero-PIGD → TD, both-zero → Indeterminate, missing items, the 101 sentinel, out-of-range values, OFF selection, deduplication, archived coding |

### Transitions and labels

| path | role |
|---|---|
| `src/ppmi_subtype/transitions.py`, `tests/test_transitions.py` | pairs, Wilson intervals, by-interval matrices, `reversion_after_change`, `first_entry`, `kaplan_meier` |
| `scripts/02_transition_analysis.py` → `outputs/transitions/` | transition analysis |
| `scripts/08_patient_level_views.py` → `outputs/patient_level/` | first Indeterminate audit |
| `scripts/28_subtype_vs_label_persistence.py` | derived subtype vs label persistence |
| scripts 03, 05, 07, 09/10, 23/24 | CTMC and HMM work |

### Subscore models

| path | content |
|---|---|
| `jointlmm.py`, scripts 16–18, 21–22 | LMM and GPs; held-out comparison, run `20261014T094751Z_s3_subscore_heldout` |
| `timevarying*.py`, scripts 19 and 25 | time-varying rates |
| `nongaussian.py`, `bbglmm.py`, `grm.py`, `grm_ou.py`, scripts 30–33 | PIGD measurement audit, models and predictive checks |
| scripts 20 and 26 | clusterability |

### Ratio models and reports

- Ratio models: `hbayes*.py`, scripts 04, 06, 34–38.
- Reports: `docs/cluster_phase/05_FINDINGS.md`, `docs/GPR_onwards_explanatory_report.md`, `docs/gpr_audit/*.md`, specs 07–09 (errata and `_frozen/`), `run_manifests/`.

---

## 2. What the evidence currently shows

### Established

**Label construction [V]:**
- Tremor score = mean of 11 items (1 from Part II, 10 from Part III). PIGD score = mean of 5 items (2 from Part II, 3 from Part III).
- TD if ratio ≥ 1.15; PIGD if ratio ≤ 0.90; Indeterminate in between.
- PIGD = 0 with tremor > 0 → **TD**. Both zero → **Indeterminate**.
- The ratio is undefined in both of those cases, but the label is defined: 725 labelled visits have PIGD = 0 (665 TD, 60 Indeterminate).
- A subscore is NaN if any of its items is missing, out of range, or 101.

**Three-state reconstruction [V].** The curated `td_pigd` is binary: 1 = TD, 2 = PIGD or Indeterminate. All 441 recomputed Indeterminate visits sit in code 2. The project rebuilds the three-class label from items.

**Medication [V]:**
- The Part III OFF exam is kept, plus untreated patients (`PDTRTMNT == 0`), plus rows with no `PDSTATE`.
- Part II items have no medication state.
- `PDSTATE` is used for selection but is not kept in the panel. `PDTRTMNT` and `LEDD` are.

**Time [V]:** years from the patient's first visit, at month resolution. Median interval 1.0 year (10th–90th percentile 0.91–2.0). Median follow-up 3.1 years (1.2–11.0).

**Prior results:**
- Changes frequently reverse (35.2 % of label changes revert at the next visit; pair-level intervals) [R].
- The Indeterminate band holds only 11 distinct ratio values, 78 % of Indeterminate visits sit at its two edges, and which edge does not predict the next label [R].
- Subscore models: GPs beat persistence on CRPS; the derived subtype does not beat label persistence on balanced accuracy; the PIGD rate rises and the tremor rate falls gradually over follow-up; dropout is informative; every Gaussian model mis-handles PIGD = 0 [R].
- Visit-level residuals of the ratio are heavy-tailed [R].

### Inconclusive or not supported [R]

- Trajectory clustering: inconclusive.
- Extreme heavy-tailed patient slopes: not supported (Q2 reassessment; this does not show the slopes are Gaussian).
- Splitting Indeterminate into near-TD and near-PIGD: not supported.

### Audit observations [V] (read-only queries)

**Cohort:** 1,080 patients, 5,774 visits, 4,441 labelled.

**Why 1,333 visits are unlabelled:**

| reason | visits |
|---|---|
| no usable OFF/untreated Part III exam | 1,259 |
| NP3PSTBL = 101 | 50 |
| other | 24 |

In the raw data NP3PSTBL carries 3,205 of the 101 codes.

**Patients with at least k usable visits:**

| score | ≥ 2 | ≥ 3 | ≥ 5 | ≥ 8 |
|---|---|---|---|---|
| tremor | 938 | 763 | 335 | 153 |
| PIGD | 934 | 762 | 334 | 145 |
| label | 934 | 762 | 333 | 145 |

**Scores are coarse:**
- PIGD item sum (0–20) is 0 / 1 / 2 in 16.3 % / 34 % / 25 % of labelled visits.
- Tremor sum is 0 in 7.2 %.
- The most frequent (tremor sum, PIGD sum) pairs among Indeterminate visits are (5, 2), (2, 1), (4, 2), (0, 0) and (7, 3).

**Size of score changes at transitions** (summed absolute item-point change, tremor + PIGD):

| transition | pairs | patients | median change | share with change ≤ 2 points |
|---|---|---|---|---|
| TD → Ind | 181 | 159 | 3 | 33 % |
| Ind → TD | 147 | 132 | 3 | 33 % |
| Ind → PIGD | 123 | 109 | 3 | 45 % |
| PIGD → Ind | 87 | 80 | 3 | 44 % |
| TD → PIGD | 253 | 226 | 5 | 13 % |
| PIGD → TD | 141 | 125 | 5 | 18 % |

### Discrepancies (documented; nothing historical is changed)

1. **Ratio undefined ≠ label undefined.** Only the ratio models exclude PIGD = 0 visits.
2. **The "OFF" label is a mixed-state construct.** The Part II items carry no medication state.
3. **Reversion rates in `outputs/transitions/` are pair-level**, with no allowance for within-patient dependence.
4. **`panel.filter_cohort(extended=True)` is unsafe [V].** It selects on `subgroup` alone, and `subgroup` also tags *prodromal* (COHORT 4) genetic carriers: LRRK2 216, GBA 192, SNCA 13, PRKN 7, who are not PD. Combination labels such as `LRRK2 + GBA` are silently dropped. No script uses this path, so no historical result is affected. Part C will not use it.

---

## 3. Decisions on the open questions

### 3.1 MCID: no validated value exists for these item sums [L]

- No source in the repository mentions an MCID.
- The literature has anchor-based MCIDs only for whole MDS-UPDRS parts. Horváth et al. 2015 (*Parkinsonism Relat. Disord.*): Part III total −3.25 (improvement) / +4.63 (worsening).
- A systematic review of movement-disorder MCIDs (PMC10909673) lists 3.05 points among Part II improvement thresholds; the search did not confirm its source study.
- No validated MCID was found for the 11-item tremor sum, the 5-item PIGD sum or the ratio. Ratio cut-offs 1.15 / 0.90 are classification thresholds (Stebbins et al. 2013), not change thresholds.

**Decision:** differences are reported in item points only, with no claim of clinical importance. Part-level MCIDs are **not** transferred to subscores, because they have a different scale and construct.

**Sources:**
- Horváth et al. 2015 — https://real.mtak.hu/30396/
- MCID systematic review — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10909673/
- Stebbins et al. 2013 — https://pubmed.ncbi.nlm.nih.gov/23408503/

### 3.2 Genetic-PD sensitivity cohort: feasible only as a pooled, descriptive cohort [V]

**Definition.** The curated `subgroup` combines enrolment cohort and IU genetic consensus data (dictionary: "based on enrollment cohort/subgroup and IU Genetic Consensus data"; source variables ENRL*, gene fields, `iu_genetic_consensus`).

The sensitivity cohort must therefore be defined as: COHORT == 1 (PD), `subgroup` ≠ Sporadic PD, and `subgroup` containing LRRK2, GBA, SNCA or PRKN. Combination labels are included. Prodromal carriers and Normosmic are excluded (Normosmic is not genetic, and its 108 patients have only single labelled visits).

**Sizes under the same visit filters as the primary panel:**

| subgroup | patients with a labelled visit | ≥ 2 labelled visits | ≥ 3 | Indeterminate visits |
|---|---|---|---|---|
| LRRK2 | 168 | 137 | 121 | 63 |
| GBA | 103 | 93 | 69 | 51 |
| SNCA | 35 | 25 | 13 | 9 |
| PRKN | 25 | 22 | 18 | 10 |
| combinations | 14 | 4 | 4 | 3 |

**Decision:**
- One pooled genetic-PD cohort (281 patients with ≥ 2 labelled visits), reported descriptively: C1 tables and C2 counts, rates and bootstrap intervals.
- **No decision rule is applied to it, and there is no per-gene inference.** The cells are too small: there are only 136 Indeterminate visits in total, and entries are a subset of these.
- The primary analysis stays the full sporadic-PD cohort.
- The cohort is built by new code. `filter_cohort(extended=True)` is not used or changed.

### 3.3 Same-visit OFF/ON agreement: feasible, descriptive only [V]

- 3,060 cohort visits have both an OFF and an ON Part III row; never more than one row per state per visit (853 patients).
- 2,348 of these pairs have both labels defined (715 patients).
- **Alignment rule:** same PATNO, EVENT_ID and INFODT, with `PDSTATE` = OFF vs ON.
- **Recorded per pair:** `HRPOSTMED`, `ONOFFORDER`, `DBSYN`, `EXAMTM`.
- **Reported:** label crosstab, item-sum differences (ON − OFF), and the same on Part III-only scores. The Part II items are identical in both exams and would otherwise inflate agreement.

**This is not test–retest reliability.** It mixes medication response, time-of-day and order effects, and rater variability. It is reported only as a descriptive bound on how much the label moves within one visit day.

### 3.4 Part III-only sensitivity: approved [V]

- Tremor = mean of the 10 Part III tremor items; PIGD = mean of NP3GAIT, NP3FRZGT, NP3PSTBL. The same `classify` rule and thresholds are applied.
- The thresholds were validated for the 11/5-item construction, not this one, so the result is a sensitivity analysis only. The primary phenotype is unchanged.
- **Reported:**
  - visit-level agreement with the primary label;
  - the transition counts;
  - every C2 statistic (§5) recomputed on this label.
- The NP3PSTBL = 101 exclusion weighs more in this construction (1 of 3 PIGD items), and the counts will show it.

---

## 4. Part C1 — raw scores

### 4.1 Definitions and cohort

- **Scores:** integer **item sums**, tremor 0–44 and PIGD 0–20. They equal 11 × and 5 × the project's means; labels come from the unchanged `classify`.
- **Primary cohort:** the frozen-panel cohort (sporadic PD, the excluded events removed, ≤ 15 years). A score is eligible at a visit if it is defined there.
- **Sensitivity cohorts:** genetic PD (§3.2) and the Part III-only construction (§3.4).

### 4.2 Audit and descriptives

| part | content |
|---|---|
| **C1-A** | missingness flow: items → subscores → labels, with reason codes (no OFF/untreated exam; 101 by item; out of range; item missing) |
| | usable-visit counts (≥ 2, 3, 5, 8); intervals and follow-up |
| | score distributions with zero share by year and by baseline label |
| | included vs excluded patients (age, sex, disease duration, H&Y, treatment, baseline label where defined); eligibility by baseline label |
| **C1-B** | 16 example patients by a seeded stratified draw (baseline label × visit-count tertile, seed 42) |
| | yearly mean sums with patient-cluster bootstrap 95 % intervals and n at risk |
| | consecutive-visit Δ tremor vs Δ PIGD |
| | patients who change label vs those who don't |
| | **empirical item-sum transition tables**: P(S at k+1 \| S at k, interval ≤ 1.5 or > 1.5 years), with S grouped 0/1/2/3/4–5/6+ for PIGD and by quintile for tremor, with patient-bootstrap intervals. This shows the score-level behaviour at the floor directly, without a likelihood |

### 4.3 C1-C model comparison: not justified as proposed; replaced

Five checks were requested before C1-C. Their results:

1. **Already answered?** Partly.
   - Gaussian subscore models were compared held out in Stage 3 (scripts 21/22: leave-one-patient-out, 934 patients, 3,361 one-step points).
   - The beta-binomial GLMM (`bbglmm.py`, model M5) was fitted to the full cohort in script 31, run `20261014T144511Z_pigd_measurement_models` [V]. Its simulation-based predictive check fell inside the 5–95 % band for **1 of 16** features. It reproduced neither the zero share (real 0.161, simulated 0.240) nor zero persistence (P(next = 0 \| S = 1): real 0.109, simulated 0.260).
   - Its precision parameter ran to the binomial limit (φ ≈ 7,554).
   - The Gaussian (M1) and heteroscedastic (M2) models failed the same zero features. The item-level GRM and GRM + OU models failed their pre-specified gates (05_FINDINGS, Stage 6b).
   - Both candidates are therefore already known to be misspecified at PIGD = 0, and in the same direction (too many zeros).
2. **Comparable outcomes?** Only with care. The Gaussian LMM predicts a continuous mean; the beta-binomial predicts a pmf on {0..n}. A fair comparison needs the Gaussian predictive discretised to the integer lattice, with tails censored at 0 and n. It also needs a *univariate random-slope* Gaussian model to match M5's random-effects structure (`jointlmm` is bivariate, random-intercept only [V]; a random-slope fitter exists in `timevarying_ext.fit_basis(random_slopes=True)` [V]). This is feasible, but it is new scoring code.
3. **Trial denominator, support, missing items [V].**
   - `bbglmm.pack` takes `n_trials` and rejects non-integer or out-of-range sums. Masked entries are ignored, so a visit with a missing item (subscore NaN) is simply absent.
   - n = 20 (PIGD) and 44 (tremor) are correct for the item sums.
   - **No held-out prediction function exists.** p(b \| history) and the predictive pmf would have to be written.
4. **Splits and rule.**
   - Script 04's seven 70/30 splits reuse overlapping test patients, so "≥ 5/7" is a consistency heuristic, not independent replication.
   - The existing subscore held-out protocol is leave-one-patient-out, which is infeasible here: one M5 fit took 4,089 s [V], so 934 refits are out of reach.
   - A 5-fold patient-level partition (disjoint test sets, every patient tested once) would be the better design if a comparison were run.
5. **Value added.** Low. A held-out log-score ranking of two models already known to fail the same feature would not explain the zero process. It would cost about 14 fits at roughly 1–2 hours each, plus new prediction code.

**Decision proposed:** C1-C is **not run**. Instead:

- (a) the empirical item-sum transition tables in C1-B describe the zero and floor behaviour model-free;
- (b) if a new likelihood is wanted later, a separately specified **patient-level hurdle model** is the better-justified candidate. Its zero process would have a patient random effect and depend on the previous score. It would be designed after C2, with C1-B's tables as the target features and a 5-fold patient partition fixed in advance. It is not part of this task.

---

## 5. Part C2 — Indeterminate audit: exact definitions and decision rules

To be frozen (SHA-256 recorded) before any C2 statistic is computed.

### 5.1 Units and sequences

- **Patients:** primary cohort patients with ≥ 2 labelled visits (934).
- **Sequence:** a patient's labelled visits ordered by `years_from_baseline` (ties broken by nominal `YEAR`, as in `panel._add_time_axis`). Unlabelled visits are skipped, as in `transitions.consecutive_pairs`.
- **Pair:** consecutive visits (k, k+1) in that sequence, carrying the interval Δt.
- **Visit annotations** (descriptive):
  - (tremor sum, PIGD sum);
  - ratio state: defined / PIGD = 0 and tremor > 0 / both zero;
  - availability changes between visits (a constituent item missing or 101 at the adjacent visit);
  - treatment status at both visits;
  - **label margin**: the minimum number of ± 1 item-sum steps, within bounds, that moves the visit to a different label under the unchanged rule. Exact lattice search; descriptive only.

### 5.2 Events and outcomes

| event | definition |
|---|---|
| **Entry** | pair with label_k ∈ {TD, PIGD} and label_{k+1} = Ind. Origin O = label_k. Arrival visit j = k+1. |
| **Exit** | pair with label_k = Ind and label_{k+1} ∈ {TD, PIGD} |
| **Direct change** | pair with {label_k, label_{k+1}} = {TD, PIGD} |
| **Stay** | label_k = label_{k+1} |

**Outcome at j+1** for entries and direct changes (the arrival label X, the origin O):

| outcome | definition |
|---|---|
| sustained | label_{j+1} = X |
| reverted | label_{j+1} = O |
| moved on | label_{j+1} = the third state |
| unobserved | no labelled visit after j |

**Unobserved events are counted and reported and are excluded from every persistence and reversion denominator. They are never classed as sustained or reverted.**

Two-step outcomes (j+1 and j+2 both = X) use the subset with two later labelled visits, which has its own denominator.

### 5.3 Statistics

All ratios are event-pooled: Σ numerators / Σ denominators over all events of all patients in the sample.

| statistic | definition |
|---|---|
| **R_E** | k+1 reversion rate of entries = reverted entries / entries with j+1 observed |
| **R_D** | k+1 reversion rate of direct changes, defined the same way |
| **Δ_R** | R_E − R_D |
| **O/E_Ind** | temporal clustering of Indeterminate within patients |

**O/E_Ind** is defined as follows. For patient i with n_i labelled visits of which m_i are Indeterminate:

- O = Σ_i (number of adjacent Ind–Ind pairs in the patient's sequence);
- E = Σ_i m_i(m_i − 1)/n_i, which is the exact expected number of adjacent Ind–Ind pairs if the patient's own labels were randomly reordered over their visit slots;
- O/E_Ind = O / E. It is reported with O, E and the number of contributing patients.

O/E_TD and O/E_PIGD are defined the same way and reported for calibration only. They will exceed 1 wherever labels drift with time, so they are not part of the rule.

**Descriptive** (no rule):

- sustained, moved-on and unobserved shares;
- two-step persistence;
- Kaplan–Meier time-to-exit from first entry, censored at the last labelled visit (existing `kaplan_meier`);
- label-margin distributions by event type;
- score changes at entries, exits and direct changes;
- availability-change and treatment-change shares at events.

Score-change comparisons against stay-pairs are reported **descriptively only**. A label change requires a score change, so such a contrast is biased by construction and is not used for decisions. For the same reason, "Indeterminate visits have small label margins" is not a decision criterion: the band is narrow by construction.

### 5.4 Uncertainty

- Patient-cluster bootstrap with **B = 4,000** and seed 42 (`np.random.default_rng(42)`).
- Each replicate resamples 934 patients with replacement. A drawn patient brings all their visits; duplicated patients count as distinct.
- Every statistic is recomputed on the replicate. Intervals are 95 % percentile intervals.
- A replicate with a zero denominator for some statistic is reported as undefined for it and counted; this is expected to be 0 for the primary statistics.
- Pair-level Wilson intervals are reported alongside, for continuity with script 02 only.

### 5.5 Pre-specified readings (primary cohort, primary label)

**(i) Consistent with a transient, boundary-crossing phenomenon** if both hold:

- Δ_R 95 % interval lower bound > 0 (entries into Indeterminate revert more often than direct TD ↔ PIGD changes); and
- O/E_Ind 95 % interval includes 1 or lies below 1 (Indeterminate visits are not temporally clustered beyond a within-patient random ordering).

**(ii) Consistent with a temporally structured (sustained) intermediate passage** if both hold:

- O/E_Ind 95 % interval lower bound > 1; and
- Δ_R 95 % interval includes 0 or lies below 0.

**(iii) Inconclusive** in every other case, including mixed results.

**Interpretation limits (fixed now):**

- (ii) does not establish a distinct biological state. Monotone drift through the band also produces clustering.
- (i) does not establish that Indeterminate is "noise". It says the label behaves like transient boundary crossings at this measurement resolution.
- Neither reading says anything about latent disease progression.

**Sensitivities** (reported; the verdict is not changed by them):

- Part III-only label;
- genetic-PD cohort (descriptive);
- pairs with Δt ≤ 1.5 years;
- treatment-stable pairs;
- excluding both-zero visits;
- `band_boundaries: inner`.

None of these relabels a patient. They recompute the same statistics on the stated subset or rule.

### 5.6 Sample-size check (from §2)

Entries involve 159 + 80 patient-events in the primary cohort, and direct changes 226 + 125. The Δ_R contrast is adequately powered for moderate differences. O/E_Ind depends on patients with ≥ 2 Indeterminate visits; their number will be reported, and if E < 20 the O/E_Ind criterion is declared not evaluable, giving reading (iii).

---

## 6. Figures and tables

| # | figure or table | question it answers |
|---|---|---|
| T1 | missingness and eligibility flow with reason codes | who is analysed, and why visits are lost |
| T2 | included vs excluded patients, by baseline label | is the exclusion selective? |
| F1 | item-sum distributions by year, zero share highlighted | floor effects over time |
| F2 | 16 seeded example patients: tremor sum, PIGD sum and label, with undefined-ratio, both-zero and missing visits marked distinctly | individual courses |
| F3 | yearly mean sums with bootstrap intervals and n at risk | aggregate course and attrition |
| T3 | empirical item-sum transition tables | score-level floor and persistence, model-free |
| F4 | Δ tremor vs Δ PIGD at consecutive visits, by event type | how large are the changes behind label changes? |
| F5 | (tremor sum, PIGD sum) lattice with the label regions and visit density | where Indeterminate lies on the measurement grid |
| T4 | events: pairs and patients, by type and interval; unobserved counts | how common each movement is |
| T5 | R_E, R_D, Δ_R, O/E statistics, two-step persistence, with denominators and bootstrap intervals | the C2 readings |
| F6 | Kaplan–Meier time-to-exit from Indeterminate, TD and PIGD | how long labels last |
| T6 | OFF/ON same-visit crosstab, full and Part III-only | how much the label moves within a visit day (not test–retest) |
| T7 | sensitivity table: every C2 statistic per sensitivity | robustness |

---

## 7. Failure modes

- **Selection:** labels require a usable OFF/untreated exam and no 101 code.
- **Informative dropout:** higher PIGD → loss of follow-up [R]. Denominators shrink selectively.
- **Integer grid and narrow band:** small changes flip labels. This is described, not used as evidence.
- **Treatment initiation** in years 0–2 [R]. Part II items are not medication-aligned.
- **Small cells:** two-step persistence, per-gene results, and Indeterminate → Indeterminate (64 pairs [R]).
- **Within-patient dependence:** handled by the patient bootstrap. There is no p-value thresholding across the many descriptive tables.

---

## 8. Decision summary

| item | rule |
|---|---|
| C2 | §5.5, frozen before computation |
| C1 | descriptive; no decision rule |
| C1-C | not run (§4.3) |
| genetic PD and OFF/ON | descriptive |

---

## 9. Implementation plan (after approval)

**New files.** No existing file is modified.

| file | content |
|---|---|
| `src/ppmi_subtype/raw_scores.py` | item-level visit table for a given cohort definition (primary; genetic PD with COHORT == 1); cleaned items, item sums, reason codes; `PDSTATE` and OFF/ON pairing re-joined; primary and Part III-only labels via the unchanged `classify`; label margin |
| `src/ppmi_subtype/indeterminate_audit.py` | sequences, events and outcomes (§5.1–5.2), statistics including the closed-form O/E (§5.3), patient-cluster bootstrap (§5.4), readings (§5.5) |
| `scripts/39_raw_score_audit.py` | C1-A and C1-B, OFF/ON, Part III-only and genetic-PD descriptives |
| `scripts/40_indeterminate_audit.py` | C2. Writes `c2_config.json` with the SHA-256 of this plan and of the code before computing |
| `tests/test_raw_scores.py` | reconstruction reproduces the panel's 4,441 labels and subscores exactly; sum = 11 × / 5 × mean; 101 and out-of-range → missing; the distinct states (missing, PIGD = 0 → TD, both-zero → Ind); visit ordering; Part III-only items; genetic cohort excludes COHORT 4 and keeps combination labels; OFF/ON pairs share PATNO, EVENT_ID and INFODT |
| `tests/test_indeterminate_audit.py` | event classification on hand-built sequences; unobserved never enters a denominator; repeated events per patient; O/E closed form equals the brute-force permutation mean on small cases; label margin on hand-built lattice points; bootstrap resamples patients, not pairs; zero-denominator handling |

**Outputs:** `experiments/<UTC>_partC_raw_scores/` and `experiments/<UTC>_partC_indeterminate/`. Aggregate tables can later be copied to `run_manifests/`. Each run records the cohort definition, panel MD5, config, seeds and code SHA-256s.

**Compute:** both scripts are single-process and take minutes. The 4,000 bootstrap replicates are table operations over about 1,000 patients. The login node at low priority is sufficient, and each script gets a timed `--smoke` run first.

---

## 10. Unresolved design choices (need your approval)

1. **C1-C dropped** (§4.3), and the hurdle-model idea deferred to a separate specification after C2.
2. **C2 rule change.** The new readings use O/E_Ind instead of the earlier "score change vs reference" and "margin share" criteria, which are biased by construction (§5.3).
3. **E < 20** makes O/E_Ind not evaluable, giving reading (iii).
4. **Genetic PD:** pooled and descriptive only, defined as COHORT == 1 with combination labels included (§3.2).
5. **Example-patient selection:** 16 patients, stratified by baseline label × visit-count tertile, seed 42.
6. **Item-sum transition grouping:** PIGD 0/1/2/3/4–5/6+; tremor by quintile of the sum. This choice is descriptive only.

## 11. Files proposed

**Create:**
- `src/ppmi_subtype/raw_scores.py`
- `src/ppmi_subtype/indeterminate_audit.py`
- `scripts/39_raw_score_audit.py`
- `scripts/40_indeterminate_audit.py`
- `tests/test_raw_scores.py`
- `tests/test_indeterminate_audit.py`

**Change:** only this plan, which is already modified and uncommitted. No existing code, output, specification or frozen file is changed.

---

## Amendment 1 (2026-10-16 IST / 2026-10-15T22:10Z): tightened C2 reading rule

**Status:** approved by the reviewer before any full C2 analysis was run. Only the 150-patient smoke run had been executed; its clustering was not evaluable (E = 11.1).

This amendment **supersedes §5.5's decision conditions for all full runs**. §5.5 above is left exactly as it was approved and is *not* rewritten; its conditions are quoted below for the record. Revision 2 of this plan before this amendment had SHA-256 `0eacdac5ac1f18da9621b67b6fb1a4ca46a1e6c833f600890930e4efeb009f40`. The amended file's SHA-256 is recorded by script 40 in `c2_config.json` before computation.

### Original rule (revision 2, §5.5; superseded for full runs)

> (i) transient boundary-crossing if Δ_R 95 % interval lower bound > 0 and O/E_Ind 95 % interval includes 1 or lies below 1;
> (ii) sustained intermediate passage if O/E_Ind 95 % interval lower bound > 1 and Δ_R 95 % interval includes 0 or lies below 0;
> (iii) inconclusive otherwise; not evaluable if E < 20 (§5.6).

### Rationale

In the original (i), an O/E interval that merely *includes* 1 satisfied the clustering condition. A wide interval, such as the smoke run's [0.70, 1.60], spans both "no clustering" and "substantial clustering", so a nonsignificant contrast would have been read as if it established absence of clustering. (ii) had the same asymmetry for Δ_R. The amended rule requires each contrast to establish its direction, and makes (i) and (ii) symmetric.

### Amended rule (applies to the primary cohort and primary label)

| reading | condition |
|---|---|
| **(i) consistent with transient boundary-crossing** | Δ_R 95 % interval lower bound > 0 **and** O/E_Ind 95 % interval upper bound < 1 |
| **(ii) consistent with sustained intermediate passage** | Δ_R 95 % interval upper bound ≤ 0 **and** O/E_Ind 95 % interval lower bound > 1 |
| **(iii) inconclusive** | every other case |

Case (iii) includes:
- any interval spanning 0 (Δ_R) or 1 (O/E) where the reading needs a direction;
- contrasts that disagree;
- clustering not evaluable because E < 20.

A confidence interval that includes 0 or 1 is never read as evidence of equivalence. The report states the two contrasts separately, each with its interval, together with:
- eligible patients;
- event counts (events, observed-next, unobserved);
- O, E and the number of patients contributing to E.

The interpretation limits of §5.5, the sensitivity list and §5.6 are unchanged. Sensitivities report the amended reading *as if applied* for information only, and never replace the primary result.

**Implementation:**
- `indeterminate_audit.reading` implements the amended rule.
- `indeterminate_audit.reading_original` keeps the superseded rule. It is reported in the run only, labelled as superseded.
