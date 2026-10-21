# D1 post-run audit and next-step research plan

**Date:** 2026-10-16.
**Type:** audit only. No model was implemented, no simulation was run, and no existing file was modified.

**Audited run:** `experiments/20261016T070746Z_d1_full/results/` (SLURM job 13467). The frozen protocol is `docs/cluster_phase/12_CONSOLIDATED_RESEARCH_REPORT.md` Appendix B.10–B.14; the results are in Appendix B.15.

| tag | meaning |
|---|---|
| **[V]** | independently recomputed during this audit from saved outputs or data, through a different code path where possible |
| **[R]** | stated in an earlier report |
| **[H]** | hypothesis, not tested |

---

## 1. Audit checklist

| # | item | how it was checked | status |
|---|---|---|---|
| 1 | Observed S1–S6 | recomputed from the panel with `indeterminate_audit.count_matrix` / `statistics`, an independent path from `ppc_diagnostic.label_statistics` [V] | **PASS**: identical to 6 decimals (S1 0.102891, R_E 0.579618, R_D 0.311475, Δ_R 0.268142, O/E 1.033449, S6 0.0) |
| 2 | Observed label = threshold label on ratio-defined visits | panel `subtype_name` vs `labels_from_ratio` [V] | **PASS** (100 %) |
| 3 | Visit ordering | no two ratio-defined visits share (patient, time) [V] | **PASS**: tie-breaking cannot affect adjacency |
| 4 | Mid-p convention | code `midp`; observed p̃ = 0.000998 = 0.5/501 and 0.999002 = 500.5/501 [V] | **PASS**: the observed value lies beyond every replicate for S1, S3, S4 |
| 5 | z and T | T = max \|Φ⁻¹(p̃)\| recomputed from the saved p̃ = 3.0908 = the cap [V] | **PASS** |
| 6 | Threshold | sorted 500 null T values; the 476th = 2.57652 = saved c; k = ⌈0.95 × 501⌉ [V] | **PASS** |
| 7 | Strict rule and cap handling | T_obs (3.0908) > c (2.5765); c ≠ cap, so not degenerate; flags reported per B.14.2 [V] | **PASS** |
| 8 | Positive-control counts | recounted from the 400 cohort files with the frozen c: PC1 6, PC2 7, PC3 5, PC4 100 [V] | **PASS** |
| 9 | Power bound | Clopper–Pearson lower bound for 100/100 = 0.9638 ≥ 0.50 [V] | **PASS** |
| 10 | PC4v | 3/100 recounted [V]; never read by `observed_verdict` (code and test `test_pc4v_never_affects_power_or_verdict`) | **PASS** |
| 11 | Frozen-file integrity | all 7 locks and 1,000 cohort checksums re-verified; protocol hashes match the current code and report [V, post-run check] | **PASS** |
| 12 | Attendance in simulation | `generate` and `patient_replicates` use the observed ratio-defined visit times `data.t` [V, code] | **PASS**: schedules identical by construction |
| 13 | "Null mean 0.0648" (as stated in the audit brief) | traced [V] | **DISCREPANCY (terminology)**: see §3 |
| 14 | "PC4 raises S1 together with S5" (B.15 §5) | per-cohort PC4 z [V] | **CAVEAT**: see §3 |
| 15 | Verdict | B.11.5 rule applied to the frozen inputs [V] | **PASS: NOT REPRODUCED** |

No frozen result was changed.

---

## 2. Traceability of each D1 conclusion

| conclusion | value | source file | code |
|---|---|---|---|
| Observed Indeterminate share | 0.1029 (363 / 3,528 visits) | `05_observed.json` → `observed.S_obs` | `ppc_diagnostic.label_statistics`; cross-check with `indeterminate_audit` |
| Its replicate mean | 0.0648 | `05_observed.json` → `observed.S_rep_mean` | `run_cohort` (patient-conditional replicates, R = 500) |
| Direct-change reversion | 0.311 (76 / 244), replicate mean 0.436 | same | same |
| Entry reversion | 0.580 (91 / 157), z −0.49 | same | same |
| Clustering O/E | 1.033 (O 46, E 44.5), z +0.21 | same | same |
| T_obs, c, verdict | 3.0908, 2.5765, NOT REPRODUCED | `05_observed.json`, `01_threshold.json` | `ppc_runner.observed_verdict`, `ppc_diagnostic.verdict` |
| PC4 100/100, P_lo 0.964 | — | `02_power.json`; `cohorts/pc/PC4_*.json` | `ppc_runner.build_power` |
| PC4v 3/100 | — | `03_pc4v.json`; `cohorts/pc4v/*.json` | `ppc_runner.build_pc4v` |
| Null distribution | median 1.380; 14/500 at the cap | `01_threshold.json`; `cohorts/null/*.json` | `ppc_runner.build_threshold` |
| Protocol, hashes | report `9d816fe7…`, protocol lock `e0ea01e9…` | `00_protocol.json` / `.lock`, `run_status.json` | `ppc_runner.freeze_protocol` |

---

## 3. Discrepancies and caveats

**1. "Null mean" is not a null-cohort quantity (terminology).**
- 0.0648 (S1) and 0.436 (S3) are means of the **observed cohort's own patient-conditional replicates**. In B.15 they are correctly labelled "replicate mean".
- The **500 null cohorts' generated** Indeterminate share averages **0.051** [0.043, 0.059], 2.5–97.5 % range [V]. Their generated R_D averages 0.436 [0.383, 0.485] [V].
- The observed 0.103 is therefore about twice what the fitted model generates. The conditional replicates sit higher (0.065) because each is centred on the patient's own ratios, the expected conservativeness (B.10 R2.3).

**2. The PC4 signature is weaker on S5 than B.15 implies.**
- The mean PC4 z for S5 is −1.22, but only **12 %** of PC4 cohorts have z_S5 < −1.96 [V]. PC4 is detected almost entirely through S1 (mean z −3.09).
- The observed pattern **differs from PC4** on two statistics:

  | statistic | PC4 mean z | observed z |
  |---|---|---|
  | S2 (entry reversion) | +1.01: entries revert *less* than replicates | −0.49 |
  | S3 (direct reversion) | +0.52; only 7 % of PC4 cohorts have z > 1.96 | **+3.09** (at the cap) |

- The observed strong direct-reversion deficit is therefore **not** a feature that the planted intermediate state produces [V]. B.15's statement that the observed misfit "differs from the planted signature" stands, and is strengthened.

**3. The cap limits resolution.** At the cap, S1, S3 and S4 are only known to lie beyond all 500 replicates. The size of the misfit beyond that is not resolved by T, but the raw statistics show it (0.103 vs 0.065; 0.311 vs 0.436).

**4. Scope.** The ratio-defined cohort excludes the 725 PIGD = 0 visits (665 TD by rule, 60 both-zero Indeterminate), and 382 patients have sequence gaps [R: B.8]. D1's statistics differ from Part C's full-label statistics for this reason: R_D is 0.311 here vs 0.349 in Part C [R].

**5. Fold results are descriptive.** All 5 folds hit the cap, driven by S1. Folds share training patients and are not independent replications [R: B.15].

---

## 4. Ranked candidate explanations for the misfit

D1 tests one fitted model: continuous ratio, Student-t patient slopes, symmetric Student-t residuals, no intermediate state. It does **not** compare explanations, so everything below is evidence-weighted ranking, not a result.

| rank | explanation | what is demonstrated | what is plausible but untested | fits which misfit |
|---|---|---|---|---|
| **1** | **Discrete ratio lattice.** The label is determined by integer item sums. At a PIGD sum of 1, a visit is Indeterminate **iff** the tremor sum is exactly 2 (ratio 0.909); at 2, iff it is 4 or 5 (0.909, 1.136) [V: rule arithmetic, `raw_scores.label_grid`]. PIGD sums of 1–2 are about 59 % of labelled visits [R: Part C1] | 78 % of Indeterminate visits sit at exactly 0.909 or 1.136; only 11 ratio values occur in the band [R: script 08] | A continuous model with a smooth density across the 0.25-wide band cannot place probability mass on these lattice points. For low-PIGD patients, P(Indeterminate) is set by P(tremor sum = exact value), which could exceed the continuous model's ~0.15 [H] | **S1** (too few Indeterminate labels in the model) |
| **2** | **Denominator-driven, skewed noise.** The ratio's variability depends on the PIGD level (small denominators magnify one-point changes), and the residuals are right-skewed | Gaussian-fit residual skew 1.20, excess kurtosis 3.31, \|r\| > 3 in 1.5 % vs 0.27 % expected [R: `experiments/20261015T182129Z_student_t_slope_smoke/results.json`]. TT residuals are symmetric by construction [V: code] | Symmetric heavy tails produce large *downward* as well as upward jumps, generating TD → PIGD → TD flip-flops that real, right-skewed data produce less often [H]. Shape of the TT residuals themselves not checked | **S3** (too much direct reversion in the model), hence S4 |
| **3** | **PIGD = 0 outcome-dependent exclusion** | 725 labelled visits removed; 382 patients affected; the removed visits are TD (665) or both-zero Indeterminate (60) [R] | Removing low-PIGD TD visits deletes part of each affected patient's TD runs and joins sequences across gaps. The fitted model also never sees PIGD = 0 behaviour. Direction of the effect on S1 and S3 unknown [H] | possibly S1 and S3; direction untested |
| 4 | **Attendance and dropout** | schedules are identical in observed and simulated sequences [V: code]. Dropout is informative [R: Stage 5] | Informative dropout can bias θ, but cannot create a schedule mismatch [H] | weak for both |
| 5 | **A latent intermediate state** | none. PC4 shows that D1 *can* detect a constructed low-noise episode [V], but the observed misfit lacks PC4's S2 direction and has an S3 deficit PC4 does not produce [V] | — | does not match the observed pattern well |

---

## 5. Minimal staged plan for the next investigation (not implemented)

### Stage 1: descriptive checks on existing data (no new model, minutes, primary analysis untouched)

**D-a. Lattice localisation of the excess Indeterminate share.**
- *Question:* is the observed–model Indeterminate gap concentrated at low PIGD sums, where the lattice is coarsest?
- *Data and code:* panel plus `raw_scores` (item sums) and the frozen generator draws (`fullcohort_tt_draws.npz`).
- *Statistic:*
  - the observed Indeterminate share by PIGD-sum stratum (1, 2, 3, 4–5, 6+);
  - the model-implied P(Indeterminate) per visit, from the TT posterior predictive at each visit's fitted mean (analytic t CDF over [0.90, 1.15]), aggregated by the same strata.
- *Supports lattice:* the excess sits mostly in strata 1–2, with the model close to observed in higher strata.
- *Weakens it:* a uniform excess across strata.
- *Type and cost:* descriptive; about 1 minute (one TT fit to obtain b_i, or reuse a fold fit).

**D-b. Decomposition of the direct-reversion deficit.**
- *Question:* are the model's surplus TD ↔ PIGD reversions driven by downward jumps from symmetric noise?
- *Statistic:* observed direct changes and reversions, split by direction (TD → PIGD → TD vs PIGD → TD → PIGD) and by the PIGD sum at the crossing visit. Compare with the model-implied probability of such triplets under symmetric noise (analytic, per patient).
- *Supports skew:* the deficit concentrates in TD → PIGD → TD.
- *Weakens it:* a symmetric deficit.
- *Type and cost:* descriptive; minutes.

**D-c. Residual shape under the fitted TT model.**
- *Statistic:* skewness and quantile plots of standardised TT residuals, overall and by PIGD-sum stratum.
- *Supports explanation 2:* skew > 0 and growing as the PIGD sum falls.
- *Type and cost:* descriptive; one fit, about 1 minute.

**D-d. Selection sensitivity.**
- *Statistic:* the observed S1 and S3, and their model-implied values, separately for patients with vs without a removed PIGD = 0 visit (382 vs 439).
- *Supports explanation 3:* the misfit concentrates in affected patients.
- *Weakens it:* a similar misfit in unaffected patients.
- *Type and cost:* descriptive; minutes.

**Stage 1 result rule** (to be fixed before running): each check is reported descriptively, with patient-bootstrap intervals. **No verdict** is attached.

### Stage 2: one measurement-faithful no-intermediate-state check (new analysis; separately specified; moderate cost)

Run only if Stage 1 points to lattice or selection.

- *Question:* does a model that generates **integer tremor and PIGD item sums**, with **no intermediate state**, labels formed by the real rule **including PIGD = 0 visits**, and patient-level trends reproduce S1 to S5 on the **full three-state label**?
- *Candidates:* an existing count model on each sum (`bbglmm`, already shown to mis-fit zero frequency, 05_FINDINGS Stage 6b), or a simpler patient-level resampling generator.
- *Known limitation:* every existing count model mis-fits PIGD = 0, so this stage must report its own zero-fit check first.
- *Validation:* the same D1 machinery (folds, calibration, a PC4-type control built in that model's units).
- *Risks:*
  - This is a new primary analysis and needs its own frozen specification.
  - It must not reuse D1's threshold.
  - It must evaluate the observed cohort once, under the new specification. The D1 observed result stays as is.

### Stage 3: latent-state model (only if the criteria in §6 are met)

---

## 6. Criteria for deciding whether a latent-state model should be tested

A latent-state (intermediate-state) model should be tested **only if all** of the following hold:

1. **A measurement-faithful no-state model fails.** A model generating integer item sums with PIGD = 0 included (Stage 2) is properly calibrated and has demonstrated power against a constructed intermediate-state control in its own units, and it still does **not** reproduce the observed label statistics.
2. **The failure has the intermediate-state signature:** excess Indeterminate share **together with** excess within-patient clustering or persistence (S5 / S6 above the replicates), not only a prevalence or a direct-reversion misfit.
3. **Stage 1 does not localise the misfit** to the low-PIGD lattice strata, to skewed downward-jump noise, or to patients affected by PIGD = 0 removal.
4. **The pattern survives** the existing sensitivities (Part III-only label, treatment-stable pairs, intervals ≤ 1.5 years) without changing direction.
5. **Any latent-state model is tested by held-out comparison** against the no-state model under a pre-specified rule. Its latent states are **not** interpreted as subtypes without independent evidence.

The existing HMM cannot serve as this test: its Indeterminate emission is pinned at the floor [R: 05_FINDINGS Stage 4].

If criteria 1–3 fail, the conclusion is that the observed label behaviour is consistent with measurement construction (lattice, skew, selection) acting on continuous progression. That is still a conclusion about *these* models and *this* measurement, not proof that no intermediate biological state exists.

---

## 7. Conservative interpretation (unchanged)

- D1 shows that one fitted continuous Student-t ratio model, on ratio-defined visits, does not reproduce the Indeterminate share or the direct-change reversion of the observed labels.
- That alone is **not** evidence of a biological intermediate subtype.
- PC4 is a constructed positive control, not a validated biological scenario.
- Observed-label durations (Kaplan–Meier median 1.08 years) are not HMM latent-state durations (2.12 years, emission floor bound) [R: report §3.1].
- No statistical component is a disease subtype without independent evidence.
