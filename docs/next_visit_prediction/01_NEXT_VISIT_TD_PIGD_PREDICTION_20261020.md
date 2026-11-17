# Next-visit TD-vs-PIGD prediction from a patient's subtype history

**Date** 2026-10-20 · **Run** `experiments/20261020T104756Z_next_visit_td_pigd_prediction/`
**Code** `scripts/60_next_visit_subtype_prediction.py`, `src/ppmi_subtype/next_visit.py`,
`tests/test_next_visit.py` · **Panel** `outputs/visit_panel.csv`, md5 `fb038ddbdc19ab3368e13940f398e7ba`
**Commit at run time** `3a860ca` · **Folds** 934 attempted, 934 completed, 0 failed

---

## 1. The question

> Can we reliably predict the next observed TD-vs-PIGD subtype from a patient's
> TD/Indeterminate/PIGD history, and does explicitly modelling elapsed time or
> latent states improve prediction?

**Short answer.** The next observed label is predictable to 83.7% accuracy
against a 69.8% base-rate null — but **almost none of that comes from the
models**. Repeating the patient's last non-Indeterminate label gets 83.4%, and
the best model's advantage over that rule is +0.33 pp, 95% CI [−0.37, +1.00] —
indistinguishable. What the models do add is *usable probabilities*: the CT-HMM's
Brier score is 0.1214 against persistence's 0.1598, a 24% reduction, and its
P_PIGD is calibrated to within 0.0065 on average.

Latent states help, slightly and detectably (+0.92 pp, CI [0.03, 1.79]).
Elapsed time helps the discriminative model clearly (+1.65 pp, CI [1.12, 2.17])
but does nothing for the observed-label Markov chain (−0.17 pp, CI [−0.39,
+0.03]). Both effects live almost entirely in the 14% of cases where visits are
more than 1.5 years apart, and the latent state's gain lives specifically where
the current recorded label is Indeterminate.

---

## 2. What the two probabilities mean

For each prediction the models output

```
P_TD   = p_TD   / (p_TD + p_PIGD)
P_PIGD = p_PIGD / (p_TD + p_PIGD)
```

computed only when the denominator exceeds 1e-12. These are **conditional on the
next observed label being TD or PIGD**. They are *not* unconditional next-visit
probabilities. 9.9% of next visits are recorded Indeterminate, and the
generative models put real mass there (CT-HMM mean `p_Indeterminate` = 0.1033);
the renormalisation divides that mass out, which is exactly what gives the two
outputs their conditional reading.

`p_Indeterminate` is **not discarded** — it is carried in the prediction table as
`p_Indeterminate_raw` and reported per model. A clinician reading P_PIGD = 0.7
is being told *"if the next exam lands on TD or PIGD, it is 70% likely to be
PIGD"*, not *"there is a 70% chance of being PIGD at the next visit"*. The
unconditional statement would be roughly 0.7 × (1 − 0.099) ≈ 0.63, and this
experiment does not validate that number.

**Tie rule.** `P_TD > P_PIGD → TD`, `P_PIGD > P_TD → PIGD`, exact tie → **PIGD**.
A symmetric probability carries no information, so the rule has to come from
outside the model; PIGD is the worse motor prognosis, so the cheaper error under
a tie is the one that does not miss a PIGD conversion. The rule is
deterministic, independent of the fit, and ties are counted — **0 ties occurred
in 3,029 scored predictions × 7 models**, so the rule had no influence on any
result here.

**Indeterminate is never recoded.** It stays a third observed label in every
input history: the latent models emit it, the observed-label chains transition
through it, and the direct model reads it as a feature. Only the *target* is
restricted to two classes.

---

## 3. What already existed, verified against the code

| Claim | Verdict |
|---|---|
| `src/ppmi_subtype/hmm.py` is a CT-HMM | **Yes.** The latent state evolves as a continuous-time Markov chain with generator `Q` built by `ctmc.build_generator`; `transition_probabilities(Q, gaps)` is `exp(Q·dt)` by eigendecomposition; there is an explicit emission (misclassification) matrix `E` over the three recorded labels. `predictive_distributions()` already runs the forward filtering recursion and returns `alpha @ P(dt) @ E` — the next-*observed*-label distribution. |
| `src/ppmi_subtype/ctmc.py` is a CT-HMM | **No.** It is a chain in the *observed* label with no observation model at all. It is used here as the no-latent-layer control, not as a second CT-HMM. |
| Labels | `subtype.py` implements Stebbins et al. 2013: `ratio = mean(11 tremor items) / mean(5 PIGD items)`; `≥ 1.15 → TD`, `≤ 0.90 → PIGD`, between → Indeterminate, plus explicit degenerate cases. Per-visit longitudinal labels come from `outputs/visit_panel.csv`. |
| LOPO machinery | `loo.py` holds out whole patients and already predicts the next *observed* label via `hmm.predictive_distributions`, with a leakage assertion and a patient-clustered bootstrap. Reused. |
| Known convergence problem | `hmm_stability.py` documents it: `fit_hmm` reports convergence from scipy's `success` flag alone, but L-BFGS-B with finite-difference gradients routinely stops at `ABNORMAL_TERMINATION_IN_LNSRCH` with a tiny projected gradient. The projected gradient is the right criterion. |
| The 75.4% HMM figure | A **different task** — 3-class next-label accuracy over 3,361 transitions (`outputs/loo/`). It does not transfer and is not used as a reference anywhere below. |

**So a CT-HMM did not need building.** It needed restricting to two classes,
plus honest convergence reporting. `next_visit.audited_starts` reproduces
`fit_hmm`'s exact starting points and feeds them to
`hmm_stability.fit_multistart`, which runs the *identical* optimisation (same
objective, bounds and options) and additionally returns the projected gradient
and the termination message.
`tests/test_next_visit.py::test_audited_starts_reproduce_fit_hmm` asserts the
resulting log-likelihood, `Q` and `E` are unchanged, so the substitution is a
reporting improvement and not a different fit.

---

## 4. Design

**Cases.** Every labelled visit followed by another labelled visit for the same
patient: **934 patients, 3,361 prediction opportunities**. Target is the label
recorded at the next visit.

| Actual next label | n | share |
|---|---|---|
| TD | 2,115 | 0.6293 |
| PIGD | 914 | 0.2719 |
| Indeterminate | 332 | 0.0988 |

Eligible for two-class scoring: **3,029**. Excluded (Indeterminate next visit):
**332**. Of the 934 patients, **899 have at least one eligible case**; the other
35 have none, so the scored patient count is 899, not 934.

**Validation.** Leave-one-patient-out, 934 folds. The whole patient is held out
from every fit *and* from the direct model's standardisation. Visits are never
split individually. `loo.assert_no_leakage` checks the sequences and the
transition pairs; the training case frame is filtered on `PATNO` before any
statistic is taken from it.

**Filtering, not smoothing.** The prediction for visit *k+1* uses labels
0..*k* only. For the CT-HMM this is the forward recursion — `alpha` is advanced
with the label seen at *k*, then propagated through `exp(Q·dt)`. No backward
pass is run and no quantity depending on label *k+1* enters.

**Elapsed time is an input.** The interval-aware models are told *when* the next
examination happens, never what it records. That is this project's panel-data
convention (`loo.py` states it too) and it is an assumption worth flagging,
because PPMI's visit schedule is not independent of disease course.

**Candidates** — chosen to span the two axes the question asks about:

| Model | Latent state | Elapsed time | Source |
|---|---|---|---|
| CT-HMM | yes | yes | `hmm.fit_hmm`, unchanged |
| CTMC (observed) | no | yes | `ctmc.fit_ctmc`, unchanged |
| Markov (observed, discrete) | no | no | `ctmc.fit_dtmc`, unchanged |
| Direct logistic (history + interval) | no | yes | new, `next_visit.fit_direct` |
| Direct logistic (history only) | no | no | new, same minus `log_interval` |
| Persistence (last non-Indeterminate) | — | no | rule |
| Marginal (two-class prior) | — | no | null |

The direct model is an L2-penalised logistic regression (penalty fixed at 1.0 in
advance, never tuned) on ten strictly causal features: current label dummies,
whether the last non-Indeterminate label was PIGD, the fractions of history
that were PIGD and Indeterminate, log run-length of the current label, log
history length, years from baseline, and log interval. It estimates the
conditional two-class probability **directly**, so its renormalisation is the
identity.

---

## 5. Results

All seven models scored the **identical** 3,029 cases from the same 899
patients — verified, not assumed (`checks.json`).

| Model | Accuracy | 95% CI (patient bootstrap) | Brier | Log loss | PIGD recall | PIGD F1 |
|---|---|---|---|---|---|---|
| **CT-HMM (latent, interval-aware)** | **0.8369** | [0.8204, 0.8541] | **0.1214** | **0.3961** | **0.667** | **0.712** |
| Direct logistic (history + interval) | 0.8359 | [0.8195, 0.8529] | 0.1225 | 0.4003 | 0.666 | 0.710 |
| Persistence (last non-Indeterminate) | 0.8336 | [0.8173, 0.8510] | 0.1598 | 4.1929 † | 0.623 | 0.693 |
| Markov (observed, discrete) | 0.8293 | [0.8129, 0.8468] | 0.1328 | 0.4314 | 0.589 | 0.676 |
| CTMC (observed, interval-aware) | 0.8277 | [0.8114, 0.8451] | 0.1343 | 0.4346 | 0.580 | 0.670 |
| Direct logistic (history only) | 0.8194 | [0.8019, 0.8372] | 0.1244 | 0.4055 | 0.646 | 0.683 |
| Marginal (two-class prior) | 0.6983 | [0.6703, 0.7278] | 0.2111 | 0.6134 | 0.000 | — |

† Persistence's probabilities are a **rule, not an estimate** (0/1). Its Brier
score is comparable; its log loss is finite only because it is clipped at
1e-12, and must not be compared with the fitted models.

**Convergence: 934/934 folds converged for every model.** For the CT-HMM every
single fold terminated with `CONVERGENCE: RELATIVE REDUCTION OF F <=
FACTR*EPSMCH`, scipy `success` was True in all 934, and the worst projected
gradient across all folds was 0.0252. No fold failed, none was dropped, and
`failed_folds.csv` is empty.

### CT-HMM confusion matrix (rows predicted, columns actual)

|  | TD | PIGD |
|---|---|---|
| **TD** | 1,925 | 304 |
| **PIGD** | 190 | 610 |

TD: precision 0.864, recall 0.910, F1 0.886 (support 2,115).
PIGD: precision 0.763, recall 0.667, F1 0.712 (support 914).

**The model systematically under-calls PIGD.** It predicts PIGD 800 times
against a true support of 914, and misses a third of actual PIGD next-visits.
That is the clinically consequential error direction, and it is the model's main
weakness — not its overall accuracy.

### Calibration of P_PIGD (CT-HMM)

| Bin | n | mean P_PIGD | observed PIGD rate |
|---|---|---|---|
| 0.00–0.10 | 1,452 | 0.0836 | 0.0806 |
| 0.10–0.25 | 488 | 0.1435 | 0.1496 |
| 0.25–0.50 | 289 | 0.3931 | 0.3945 |
| 0.50–0.75 | 349 | 0.6535 | 0.6332 |
| 0.75–1.00 | 451 | 0.8640 | 0.8625 |

Mean absolute gap **0.0065**. This is the CT-HMM's real contribution: the
probabilities can be read at face value across the whole range, which
persistence's 0/1 outputs cannot be.

### Indeterminate next visits (332 cases, reported separately, never scored)

| Model | mean P_PIGD | mean p_Ind_raw | would have said PIGD | would have said TD |
|---|---|---|---|---|
| CT-HMM | 0.3714 | 0.1235 | 111 | 221 |
| CTMC (observed) | 0.3650 | 0.1079 | 86 | 246 |
| Direct logistic (+interval) | 0.3739 | 0.0000 | 111 | 221 |
| Markov (discrete) | 0.3616 | 0.1095 | 87 | 245 |
| Persistence | 0.3197 | 0.0000 | 98 | 234 |

These are not scored as correct or incorrect: the conditioning event *"next
label is TD or PIGD"* did not occur, so neither class is the right answer, and
any rule for calling them would be a decision about the label rather than a
measurement of the model. Note that the models do raise `p_Indeterminate` on
these cases (CT-HMM 0.1235 against 0.1033 overall) — weak but correctly
directed signal. The direct models show 0.0000 by construction: they estimate
the conditional split directly and never model Indeterminate as an outcome.

---

## 6. Does latent state or elapsed time help?

Paired comparisons on identical cases, 95% interval from resampling **patients**
(comparing the separate intervals in §5 would be the wrong test — the same cases
score both models):

| Contrast | Δ accuracy | 95% CI | Δ Brier | a✓b✗ / b✓a✗ |
|---|---|---|---|---|
| Latent state: CT-HMM − CTMC | **+0.0092** | [+0.0003, +0.0179] | −0.0129 | 98 / 70 |
| Elapsed time in the observed chain: CTMC − Markov | −0.0017 | [−0.0039, +0.0003] | +0.0015 | 3 / 8 |
| Elapsed time, discriminative: Direct − Direct(no interval) | **+0.0165** | [+0.0112, +0.0217] | −0.0019 | 61 / 11 |
| CT-HMM − Direct logistic | +0.0010 | [−0.0054, +0.0075] | −0.0011 | 53 / 50 |
| CT-HMM − Persistence | +0.0033 | [−0.0037, +0.0100] | **−0.0384** | 59 / 49 |
| CT-HMM − Marginal null | **+0.1387** | [+0.1107, +0.1665] | −0.0897 | 610 / 190 |

**Latent state: yes, slightly.** +0.92 pp with a CI that just excludes zero, and
a clear Brier gain (−0.0129). Small, but real and in the expected direction.

**Elapsed time: it depends entirely on the model form.** +1.65 pp for the
discriminative model (unambiguous), but nothing for the observed-label chain.
The resolution is in the visit schedule: the median gap is 1.002 years, the IQR
is 1.00–1.16, and **85.8% of cases sit at 1.5 years or less**. A
time-homogeneous generator in the observed label can only use interval
information through `exp(Q·dt)`, which is nearly constant across that bulk; the
logistic model gets `log_interval` as a free covariate and can use it where it
matters without being constrained to a generator form.

### Where the differences actually live

Accuracy by inter-visit interval:

| Model | ≤ 1.5y (n=2,597) | > 1.5y (n=432) |
|---|---|---|
| CT-HMM | 0.8417 | **0.8079** |
| Persistence | 0.8417 | 0.7847 |
| Markov (discrete) | 0.8410 | 0.7593 |
| CTMC (observed) | 0.8410 | 0.7477 |
| Direct logistic (+interval) | 0.8398 | **0.8125** |
| Direct logistic (no interval) | 0.8260 | 0.7801 |

**On the annual-visit bulk nothing beats persistence** — CT-HMM and persistence
are identical to four decimals (0.8417). Every model's advantage is earned on
the 14% of cases with long gaps, where persistence degrades by 5.7 pp and the
observed-label CTMC by 9.3 pp while the CT-HMM loses only 3.4 pp.

Accuracy by the label recorded at the prediction point:

| Model | current TD (n=2,080) | current Indeterminate (n=270) | current PIGD (n=679) |
|---|---|---|---|
| CT-HMM | 0.8812 | **0.6074** | 0.7923 |
| Direct logistic (+interval) | 0.8817 | 0.5889 | 0.7938 |
| Persistence | 0.8784 | 0.5926 | 0.7923 |
| Markov (discrete) | 0.8784 | 0.5444 | 0.7923 |
| CTMC (observed) | 0.8784 | 0.5444 | 0.7850 |
| Direct logistic (no interval) | 0.8803 | 0.4185 | 0.7923 |
| Marginal (null) | 0.8784 | 0.5444 | 0.2077 |

**This is where the latent state earns its keep.** On the 270 cases where the
current recorded label is Indeterminate, the observed-label chains are
degenerate — they always predict TD, so their accuracy is exactly the TD share
(147/270 = 0.5444) — while the CT-HMM's filtered latent posterior reaches
0.6074, +6.3 pp. A latent phenotype read through misclassification is precisely
the right tool for an uninformative recorded label, and that is the one place it
visibly works.

Accuracy by history length (CT-HMM): 0.834 at 1 prior visit, 0.864 at 2, 0.827
at 3, 0.810 at 4, 0.832 at 5+. **Longer history buys nothing** — flat to
non-monotone. Whatever is predictable is in the last visit or two.

---

## 7. Which model is primary, and why

**Primary: the CT-HMM (latent, interval-aware)** — `hmm.fit_hmm` unchanged, with
the conditional two-class restriction applied on top.

- Best on every probability measure that matters: Brier 0.1214, log loss 0.3961,
  mean absolute calibration gap 0.0065 across five bins spanning 0–1.
- Best PIGD recall (0.667) and F1 (0.712) of any candidate — the class the task
  cares about and the one every model struggles with.
- Perfect convergence reliability: 934/934 folds, worst projected gradient
  0.0252, every fold terminating on the relative-reduction criterion.
- Interpretable in the right currency: it yields a generator `Q` (transition
  rates, mean sojourn times at any horizon) and an emission matrix `E` giving
  per-state misclassification probabilities — quantities a clinician can read,
  which a logistic coefficient vector on engineered history features is not.
- It degrades least when visits are far apart and is the only model that is not
  degenerate when the current label is Indeterminate.

**Honest caveats on that choice.** The direct logistic model is
*statistically indistinguishable* on accuracy (Δ = +0.10 pp, CI [−0.54, +0.75]),
has a nearly identical Brier score, fits in milliseconds rather than seconds,
and would be the right choice if cost mattered. And the CT-HMM's edge over the
**persistence rule** is +0.33 pp, CI [−0.37, +1.00] — so on accuracy alone, a
one-line rule with no parameters is as good. The CT-HMM is chosen on probability
quality, convergence reliability and interpretability, not on classification
performance, because on classification performance there is nothing to choose.

---

## 8. Answer to the scientific question

**Can we reliably predict the next observed TD-vs-PIGD label from a patient's
TD/Indeterminate/PIGD history?**

Partly, and mostly for a trivial reason. 83.7% accuracy against a 69.8%
base-rate null is a genuine 13.9 pp gain (CI [11.1, 16.7]), so the history *is*
informative. But the subtype label is highly persistent, and 83.4% of that is
available from "repeat the last non-Indeterminate label". The honest statement
is: **the next label is predictable because the subtype rarely changes, not
because we have learned its dynamics.** The models' real contribution is
calibrated probabilities rather than better labels — and for the clinically
important direction they remain weak, missing a third of actual PIGD
next-visits.

**Does explicitly modelling elapsed time improve prediction?**

Only where there is elapsed time to model. On 86% of cases, visits are ~1 year
apart and the answer is no. On the 14% with gaps beyond 1.5 years it is clearly
yes: +1.65 pp overall for the discriminative model, and the interval-aware
models hold 0.808–0.813 on long gaps where persistence drops to 0.785. A
time-homogeneous generator in the *observed* label cannot exploit this (CTMC ≈
Markov, CI spanning zero) because the constraint, not the information, is the
binding limitation.

**Does modelling latent states improve prediction?**

Yes, slightly and detectably: +0.92 pp over the identically interval-aware
observed-label CTMC, CI [0.03, 1.79], with a 0.0129 Brier improvement. The gain
is concentrated exactly where theory predicts — on the 270 cases whose current
recorded label is Indeterminate, where the observed-label chains are degenerate
and the CT-HMM gains 6.3 pp by carrying a latent posterior through the
uninformative observation.

A latent process read through misclassification is a modest claim, and the
effect found here is correspondingly small.

---

## 9. Limitations

1. **The two probabilities are conditional, not unconditional.** They answer
   "TD or PIGD, which?" given that the next label is one of the two. Roughly 10%
   of next visits are Indeterminate, and the models' own `p_Indeterminate` is
   divided out. Any unconditional reading of P_PIGD is wrong, and this
   experiment does not validate one.
2. **Indeterminate next visits (332, 9.9%) are excluded from scoring.** They are
   reported separately with what every model would have said, but they are not
   scored, so the headline accuracy describes 90.1% of prediction opportunities.
   No rule for scoring them would be a measurement of the model rather than a
   choice about the label.
3. **The direct models report `p_Indeterminate_raw = 0` by construction.** They
   estimate the conditional split directly and never model Indeterminate as an
   outcome, so that column is only meaningful for the generative models, and
   their probabilities are conditional by construction rather than by
   renormalisation.
4. **Elapsed time to the next visit is an input.** The model is told the date of
   the next examination. PPMI's visit schedule is not independent of disease
   course, so this is a mild optimism not quantified here.
5. **The target is this project's own classification rule** applied to MDS-UPDRS
   items, not an independent clinical diagnosis. Accuracy measures agreement
   with a noisy record; `src/ppmi_subtype/indeterminate_audit.py` and
   `replicate_label_diagnostic.py` document that the label itself carries
   measurement variability.
6. **Persistence's probabilities are a rule, not an estimate.** Its Brier score
   is comparable; its log loss is finite only through clipping.
7. **The direct model's L2 penalty is fixed at 1.0, never tuned.** Tuning it on
   held-out folds would be exactly the leak the LOPO design exists to prevent,
   so no penalty search was run and none should be read into the result.
8. **Patient-level accuracy is heterogeneous** and 35 of 934 patients contribute
   no eligible case at all. The scored cohort is 899 patients.

---

## 10. Files

**Code** (new; no existing module was modified)

- `src/ppmi_subtype/next_visit.py` — conditional two-class restriction, tie rule,
  causal case/feature construction, direct logistic model, audited-start
  replay for convergence reporting, LOPO loop, metrics, paired comparisons,
  breakdowns, correctness checks.
- `scripts/60_next_visit_subtype_prediction.py` — driver.
- `tests/test_next_visit.py` — 24 tests.

**Run outputs** (`experiments/20261020T104756Z_next_visit_td_pigd_prediction/`)

- `next_visit_predictions.csv` — **23,527 rows** (7 models × 3,361 cases) with
  patient id, current visit, next visit, elapsed time, `P_TD`, `P_PIGD`,
  `predicted_state`, actual next label, the raw three-label distribution,
  eligibility and per-fold convergence diagnostics.
- `model_comparison.csv`, `contrasts.csv`, `convergence.csv`, `failed_folds.csv`
  (empty), `checks.json`, `summary.json`, `report.txt`
- `confusion_*.csv`, `per_class_*.csv`, `calibration_*.csv` (one per model)
- `accuracy_by_interval.csv`, `accuracy_by_current_label.csv`,
  `accuracy_by_history_length.csv`, `indeterminate_report.csv`,
  `outcome_counts.csv`, `current_by_next_label.csv`
- `MANIFEST.sha256`, `git_commit.txt`, `pip_freeze.txt`, `driver.log`

**Reproduce** (~7 minutes on the login node, 16 workers; nothing is written
under `outputs/`):

```bash
~/envs/ppmi/bin/python scripts/60_next_visit_subtype_prediction.py --out experiments/$(date -u +%Y%m%dT%H%M%SZ)_next_visit_td_pigd_prediction --workers 16
```
