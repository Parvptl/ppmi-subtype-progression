# Cluster phase — findings, limitations and what changes

Interpretation of [`04_RESULTS_GENERATED.md`](04_RESULTS_GENERATED.md). Every
number quoted here appears in that file (section given in brackets), which is
regenerated from the saved CSVs by `scripts/27_cluster_phase_report.py`. If the
two disagree, the generated file is right.

Status key — **C** confirmatory (held-out, pre-specified) · **E** exploratory ·
**S** supersedes an earlier claim.

---

## Gate (all passed)

* Panel rebuilt on the cluster from raw PPMI files is cell-identical to the
  frozen `fb038ddb…` panel (only Windows line endings differ).
* 403 tests pass. Every Stage 3 leakage/consistency check passes (11/11), and the
  HMM LOO predictions join to the Stage 3 points 3,361/3,361 with matching event
  ids and labels.
* The cluster re-run reproduces the historical Windows predictions for the
  joint K0/K1 GPs, independent K1 GP and LMM to ≤ 1.4e-4 [Stage 3, reproduction].

---

## Stage 3 — multi-output GP comparison

1. **C. The cross-output (coregionalization) block does not help, for any kernel.**
   26 of 28 nested full − diag contrasts include zero; the two that do not
   (K0, tremor MAE and CRPS) are ≈ 0.0003–0.0005 against an MAE of 0.24. The
   joint bivariate log score — the one metric on which cross-output covariance
   *must* show up if it is real — never excludes zero [Stage 3, primary].
   This now covers K2/K3, which closes the open question about their in-sample
   constant-component correlations (+0.85/+0.98): they carry no held-out signal.
   *Do not claim joint-model superiority.*

2. **C, S. Temporal flexibility does help, once K2/K3 are evaluated held out.**
   K2 and K3 beat K0 on CRPS and log score for both outcomes and on MAE for
   tremor, in both the independent and the joint family [Stage 3, temporal
   flexibility]. K1 does not. The earlier conclusion "flexible kernels do not
   beat simple ones on the subscores" (report §E.9 Q2, §I.1 #9) rested on K1
   alone and is superseded: the useful structure is *linear trend + a
   stationary deviation*, not a stationary deviation alone.

3. **C. GP versus persistence depends on the metric.** On PIGD **MAE**
   persistence still wins (K3: +0.009 [+0.005, +0.013]); on tremor MAE K2/K3 now
   beat it. On **CRPS** every GP beats persistence for both outcomes [Stage 3,
   vs persistence]. Persistence's predictive distribution is poorly calibrated
   (PIT KS ≈ 0.30 for PIGD vs ≈ 0.10 for the GPs). This CRPS result depends on
   the error model given to persistence — here a fold-estimated Gaussian, which
   is fairer than the old ad hoc 0.3 SD but is still an assumption.

4. **C. The GP beats the random-intercept LMM on every metric and outcome**
   [Stage 3, vs LMM] — now a saved result, closing audit item 3.3.

5. **C. Derived subtype: subscore modelling does not beat label persistence.**
   Against *label* persistence (repeat the last observed subtype; 0.723
   accuracy, 0.548 balanced accuracy) [Stage 3, derived subtype, script 28 table]:
   * point rule (label rule on the predicted subscores): accuracy and balanced
     accuracy differences include zero for every GP;
   * probabilistic argmax: overall accuracy is higher (K1 indep +0.034
     [+0.024, +0.043]) but **balanced accuracy is lower** (−0.016 [−0.033,
     −0.000]) and Indeterminate recall is 0 — the gain comes from never
     predicting the transitional class. The HMM's argmax gains the same way;
   * prior-corrected rule: balanced-accuracy differences include zero for every
     GP; only the prior-corrected HMM exceeds label persistence on balanced
     accuracy (+0.019 [+0.007, +0.031]), at lower overall accuracy.
   Against the HMM, GP accuracy differences include zero and GP log loss is
   lower (≈ −0.015 to −0.021, intervals exclude zero) [Stage 3, paired vs HMM].

6. **E. Calibration.** Tremor predictive distributions are close to calibrated
   (PIT KS ≈ 0.03, coverage 50/80/95 ≈ 0.52/0.81/0.94). PIGD are not: central
   intervals are too wide (50% coverage ≈ 0.62), and ≈ 12% of predictive mass is
   below zero — the PIGD = 0 point mass again [Stage 3, continuous metrics].

7. **Convergence.** 14,010 held-out fits, 100% converged; the K2/K3 joint fits
   take ≈ 48 s each with 60 concurrent workers. The "65+ hours" that kept K2/K3
   out of the held-out comparison was a property of the earlier machine.

## Stage 4 — HMM stability

1. **E. The audited model (3 states, floor 0.5) has a unique optimum**: 40/40
   dispersed starts reach the same log-likelihood with parameters agreeing to
   ~1e-3 [Stage 4 A]. Results built on it do not depend on initialisation.

2. **E. The unconstrained 3-state model is not identified**: starts reach the
   same likelihood with emission probabilities differing by up to 0.93 — a
   ridge, as `hmm.py` anticipated. Its *predictions* are usable; its
   *parameters* must not be interpreted.

3. **E, S. The 5-state model is multimodal and was budget-limited.** With the
   default evaluation cap, 36/40 starts stop at the function-evaluation limit;
   with a 400,000 budget all converge, to 33 distinct optima, 1/40 reaching the
   best [Stage 4 A]. This supersedes both the original reading of "404/934
   converged" and my own audit hypothesis (§3.6) that it was mostly benign.

4. **C. Held-out, every configuration predicts the same.** Accuracy ≈ 0.754 for
   all; log-loss differences from the audited model ≤ 0.003; extra latent states
   give nothing [Stage 4 B]. Initialisation changes 4/5-state *parameters* but
   hardly their predictions (held-out log-lik SD across starts ≤ 0.0025 per
   transition) [Stage 4 D].

5. **C. The unbiased "select then predict" estimate equals the published one.**
   Inner CV chose C3_free in 8/10 outer folds (C5 in 2); training BIC chose
   C3_free in 10/10. The selected procedure scores accuracy 0.754 [0.735, 0.774]
   and log loss 0.670 [Stage 4 C] — the 75.4% previously reported was not
   inflated by configuration choice.

6. **C. Why Indeterminate is never predicted.** The largest probability any
   configuration ever assigns to Indeterminate is ≈ 0.25–0.40, so argmax cannot
   select it; the probabilities carry weak but real information (one-vs-rest
   AUC ≈ 0.60–0.63) [Stage 4 B]. In the floor-0.5 model the latent
   Indeterminate state's emission sits exactly on the floor (0.25/0.50/0.25) in
   every outer fold — the constraint binds, i.e. the data would make that
   state's own label even less informative. A training-prevalence
   prior-corrected rule raises Indeterminate recall to ≈ 0.25–0.32 and balanced
   accuracy from ≈ 0.52 to ≈ 0.57, at the cost of accuracy (≈ 0.68–0.72).
   **Overall accuracy is the wrong headline for a three-class problem with a
   10% transitional class; report balanced accuracy and log loss alongside it.**

7. Choosing the emission floor by held-out performance (audit §3.7) did not
   matter in the end: floors 0.6/0.7 are slightly worse in log loss, and the
   unconstrained model is indistinguishable.

## Stage 5 — temporal structure

1. **C. The rate is not constant.** A calibrated test for a slope change at an
   unknown time (sup-LR over a 0.5–10 y grid against simulated constant-rate
   nulls on the real visit times) gives p ≤ 0.005 under both a random-intercept
   and a random-slope null [Stage 5, knot]. A naive χ² reading would have been
   badly anti-conservative: the null median sup-LR is already 3–6.

2. **E, S. There is no identifiable "transition point".** The best single knot
   is 4.75 y, but its bootstrap 10th–90th percentile range is 3–9.75 y. By ML
   AIC *and* BIC a smooth quadratic beats every piecewise model, and the
   pre-specified 2/5 knots are 5.7 AIC units worse than the quadratic [Stage 5,
   structures]. The data describe gradual curvature — PIGD accelerating, tremor
   decelerating — not a change at five years.

3. **C, S. Wald intervals were too narrow in the late period.** Patient
   bootstrap SEs are 1.7–2.2× the model-based ones for the 5 y+ slopes. With
   bootstrap intervals: PIGD 5 y+ minus 0–2 y still excludes zero
   (+0.035 [+0.014, +0.056]) but 5 y+ minus 2–5 y does not under random
   intercepts; the tremor slowdown still excludes zero [Stage 5, bootstrap].

4. **E. Composition and attrition do not explain it away — but attrition is
   informative.** Loss of labelled follow-up is more likely at higher current
   PIGD (log-odds +0.52 per point [+0.36, +0.71]), so later periods
   under-represent worse patients. Yet in fixed-composition completer cohorts
   (≥ 6 y, n = 247; ≥ 8 y, n = 185) the late-minus-early contrasts persist
   (PIGD +0.044 and +0.052; tremor −0.046 and −0.051, all excluding zero), and
   early PIGD slopes do not differ by eventual follow-up length [Stage 5,
   attrition]. Since dropouts are the worse patients, the late PIGD slope is if
   anything an underestimate.

5. **E. Follow-up time cannot be separated from disease duration.** Adding
   disease-duration terms improves the fit (AIC −9777 vs −9764), and the
   follow-up slope changes at 5 y then have intervals spanning roughly ±0.1 —
   they are no longer estimable [Stage 5, time axis]. Enrolment duration spans
   only ~3.5 years in this de novo cohort, so the design cannot say whether the
   curvature belongs to time in study or to disease stage.

6. **E. Treatment initiation.** The share of exams from treated patients rises
   from ≈ 0.10 at year 0 to ≈ 0.54 at year 1 and ≈ 0.95 by year 5. Adjusting
   for a visit-level treatment indicator improves fit (AIC −9774 vs −9764) and
   moves the tremor slope change at 2 y from ≈ 0 to ≈ −0.017 (interval still
   includes zero) but leaves the 5 y changes essentially unchanged [Stage 5,
   time axis]. Early-period slopes should be read as "during treatment
   initiation".

7. **C. Prediction barely changes.** Every time-varying structure improves PIGD
   one-step CRPS by ≈ 0.0009 (≈ 0.5%, intervals exclude zero) and tremor not at
   all [Stage 5, CV] — consistent with the earlier "description improved,
   prediction did not".

**What may be claimed:** the population progression rate of the PIGD subscore
increases, and of the tremor subscore decreases, over follow-up in this cohort;
the change is gradual. **What may not:** that anything happens at five years,
or that the curvature reflects a biological disease-stage transition — the
design cannot separate follow-up time from disease duration, and treatment
initiation and informative attrition both act on the same time axis.

## Stage 6 — clusterability

1. **C (pre-specified rule): clustering is not supported**, in the full cohort
   or in the ≥ 8-visit subset [Stage 6, decision].

2. **E, S. But the slope distribution is not Gaussian.** With 200 replicates of
   each no-groups null, refitted every time, real ΔBIC (513) far exceeds the
   null 95th percentile (≈ 117) — including a null that reproduces the score
   floor and item-grid discreteness — and exceeds the median for planted
   two-group differences of 2 random-slope SDs [Stage 6]. This replaces script
   20's single-replicate comparison. Something non-Gaussian is real.

3. **C. Assignments are no more stable than in data with no groups.** Drop-one-
   visit ARI is 0.75 on real data, at the 86th percentile of the no-groups null
   (95th: 0.77). Even no-groups data gives ARI ≈ 0.63, so the earlier reading of
   stability numbers without a null was uninterpretable; truncating follow-up at
   3 years drops real ARI to 0.56.

4. **What this means:** heterogeneity in progression exists and is not Gaussian,
   but these data cannot distinguish a skewed / heavy-tailed continuum from
   discrete subgroups, and no grouping is reproducible beyond what noise alone
   produces. No phenotype claim. The planted controls show that even a 1-SD
   group separation would not be detectable at this visit density.

---

## Completion status of the four post-midterm directions

Computational completion is not scientific or reporting completion.

| Direction | Computation | Scientific conclusion | Reporting |
|---|---|---|---|
| Direct subscore modelling (LMM, independent/joint GP, K0–K3) | complete, verified (11/11 checks, reproduces historical runs) | supported as stated above; PIGD = 0 misspecification unresolved | `GPR_onwards_explanatory_report.md` not yet updated |
| Time-varying progression | complete, verified | non-constant rate supported; no identifiable breakpoint; follow-up time not separable from disease duration | existing report not yet updated |
| Trajectory clustering | complete for Gaussian and measurement-realistic nulls | reproducibility criterion not met; non-Gaussian heterogeneity unexplained — **heavy-tailed / skewed no-groups null outstanding** | existing report not yet updated |
| HMM stability and failure analysis | complete, verified | 3-state stable; extra states no predictive gain; Indeterminate never chosen by argmax | not yet in any report besides this one |

## Superseded or corrected claims (to fix in existing reports)

| Earlier claim | Where | Now |
|---|---|---|
| Flexible kernels do not beat simple ones on the subscores | GPR-onwards §E.9 Q2, §I.1 #9 | True for K1 only; K2/K3 beat K0 held out |
| Joint modelling gives no held-out benefit | §I.1 #8 | Confirmed, and now for K0–K3, not K1 only |
| K2/K3 correlations +0.85/+0.98 "open question" | §E.7, §I.2 #4 | No held-out signal |
| GP loses to persistence on PIGD | §I.1 #11 | On MAE only; GP wins on CRPS for both outcomes. Derived subtype does not beat label persistence on balanced accuracy |
| 5-state HMM converged 404/934 | midterm, §I.4 | Budget-limited and multimodal (33 optima) |
| Rate changes "after about five years" | §F, §I.1 #12 | Gradual curvature; no identifiable knot; not separable from disease duration |
| Late-period intervals | `period_slopes.csv` (Wald) | 1.7–2.2× too narrow; use bootstrap |
| Real cluster signal 513 vs null 88 | §G.9 | Null distribution q95 ≈ 117; real is non-Gaussian, but unstable as groups |
| `NEXT_RESEARCH_STEPS.md` ✅ "advantage widens with horizon", ✅ "linear kernel redundant given mean" | §1 | Both already corrected in the same file's later sections; §1 rows still stale |
| 256 / 144 tests | `TEST_RESULTS.md`, README | 403 |

## Limitations and failures in this phase

* All results are PPMI only; the target remains the project's own label rule.
* The persistence baseline's probabilistic scores depend on the error model
  assigned to it.
* The 4/5-state HMM held-out fits use a larger evaluation budget than the
  audited fitter (recorded in the run config); the 3-state fits are unaffected.
* Attrition analysis treats "no further labelled visit" as loss of follow-up;
  administrative censoring is not recorded in the panel and cannot be separated.
* Disease-duration adjustment is weakly identified (narrow enrolment duration).
* Operational mistakes, none affecting results: a directory race in script 23
  (fixed before any real run), non-picklable spline closures in script 25
  (fixed in smoke testing), and one shell `pkill` that cancelled a queued smoke
  test job.

## Next steps, in order

1. Update `GPR_onwards_explanatory_report.md` and `NEXT_RESEARCH_STEPS.md`
   using the table above.
2. A heavy-tailed / skewed no-groups null for Stage 6 (t or skew-normal random
   slopes) — the one test that could tell a non-Gaussian continuum from groups.
3. A zero-inflated or censored likelihood for PIGD (the 12% negative mass and
   mis-calibration at PIGD = 0 recur in every Gaussian model).
4. Report balanced accuracy and log loss as headline subtype metrics; if a
   decision rule is needed, choose it on training data (the prior-corrected rule
   is one such rule).
