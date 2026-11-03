# Essential analyses E-1, E-2 and E-5

**Date** 2026-11-04 · **Follows** `01_STAGE1_RESEARCH_AUDIT_20261104.md`
**Panel** `outputs/visit_panel.csv`, md5 `fb038ddbdc19ab3368e13940f398e7ba`,
verified in all three runs · **Models fitted: 0.** Nothing under `outputs/` was
written; it remains read-only.

| | Analysis | Script | Run | Checks |
|---|---|---|---|---|
| **E-1** | Resolve the lag-1 residual contradiction | `scripts/76_lag1_residual_resolution.py` | `experiments/20261103T191408Z_e1_lag1_residual_resolution/` | **19 / 19** |
| **E-2** | Patient-clustered intervals for the transition analysis | `scripts/77_transition_patient_bootstrap.py` | `experiments/20261103T193205Z_e2_transition_patient_bootstrap/` | **15 / 15** |
| **E-5** | Participant flow, every exclusion counted | `scripts/78_participant_flow.py` | `experiments/20261103T192414Z_e5_participant_flow/` | **16 / 16** |

Three sections of the manuscript are unblocked (Methods §22.2–22.5, Results R1
and R2), and two statements in the project's own documentation turn out to be
wrong. Neither error changes a scientific conclusion; both change what the
manuscript may say.

---

## E-1 — The lag-1 residual contradiction

### Verdict: a stale output file, not a live code defect

The Stage-1 audit recorded that `outputs/exploration/report.txt` draws two
incompatible conclusions from one number — §5 saying the residual
autocorrelation "favours the state-space form", its own closing item 4 saying
"independent deviations are a defensible assumption and the growth form is
adequate. Nothing here argues for the state-space alternative." I framed that
as the project contradicting itself. **That framing was wrong, and the
correction is in the project's favour.**

`scripts/06_ratio_exploration.py` has been committed exactly once (`4e5bdd2`),
and in that version **both** passages are selected by `if/elif/else` on the same
comparison of the observed statistic against the null interval. They agree by
construction. The sentence in the stale report appears in **no committed
version of that script** — checked whitespace-insensitively across all 45
commits, because the sentence is wrapped over two lines and a line-oriented
search misses it — and the distinctive clause "argues for the state-space
alternative" appears nowhere in the working tree.

The commit message of `4e5bdd2` had already recorded the fix:

> "A first draft of this script asserted the opposite in its summary, having
> been written before the null was computed. It contradicted the analysis
> directly above it and has been corrected."

So: the contradiction was found and fixed on 2026-08-27, `outputs/` was made
read-only on 2026-10-14, and the report was never regenerated in between. The
README carries the corrected reading.

**Proof, not inference.** The script re-runs `scripts/06` unmodified with
`paths.outputs` redirected into the run directory, and diffs the regenerated
report against the stale one. **Thirteen lines differ, all of them inside
closing item 4**, and the regenerated version carries the state-space reading in
both passages. The stale report fails a self-consistency test that the
regenerated one passes.

```
-  4. Lag-1 residual correlation is -0.219 against a null of -0.332 from
-     iid noise on the same visit pattern, so independent deviations are a
-     defensible assumption and the growth form is adequate. Nothing here
-     argues for the state-space alternative.
+  4. Lag-1 residual correlation is -0.219 against a null of -0.332 [-0.365, -0.294]
+     from iid noise on the same visit pattern. The observed value is
+     MORE POSITIVE than the null and outside its interval, so deviations
+     from trend persist between visits rather than resetting. That is the
+     signature of accumulating process noise and favours the state-space
+     form over the growth form the proposal settled on [...]
```

### The statistic, reproduced and strengthened

Every published number reproduces exactly: 2,050 consecutive residual pairs
from 397 patients (≥4 finite-ratio visits, ≥1 year of span), observed lag-1
correlation **−0.2195**, σ̂ = 1.2162, and under the committed 200-draw recipe a
null of **−0.3321 [−0.3648, −0.2940]**.

The stale report put an interval on the null and a bare point estimate on the
observation. Both sides now carry one:

| Quantity | Estimate | 95% interval | Method |
|---|---:|---|---|
| Observed lag-1 correlation | **−0.2195** | −0.2731 to −0.1621 | patient-clustered bootstrap, 2,000 draws |
| iid null, same visit pattern | **−0.3328** | −0.3689 to −0.2951 | Monte Carlo, 2,000 draws |
| **Observed − null** | **+0.1133** | **+0.0457 to +0.1785** | combined resample |

One-sided Monte Carlo p (null ≥ observed) = **5.0 × 10⁻⁴**; no null draw in
2,000 reached the observed value. The conclusion is unchanged at ten times the
committed number of draws.

### Why the statistic cannot be read against zero

OLS residuals from a line fitted to a handful of points are negatively
correlated **by construction** — they sum to zero and are orthogonal to time —
so the null is −0.33, not 0. Read against zero, −0.219 looks like mean
reversion. Read against its actual null, it is the opposite: deviations from a
patient's own trend are **more persistent** than independent measurement error
would produce. This is exactly the error the first draft made.

### Independent corroboration already in the repository

A single autocorrelation statistic is thin ground for a modelling decision.
Two completed experiments bear on the same question and point the same way,
neither run for this purpose:

1. **GPR kernel comparison.** K1 (constant + RBF) beats K0 (constant + linear)
   on held-out MAE in all four cohorts under both aggregations — unfiltered
   −0.0333 [−0.0467, −0.0205] pooled, −0.0192 [−0.0279, −0.0106] per patient.
   An RBF term with a ~3.7-year lengthscale **is** temporally persistent
   deviation from the trend: the same structure a state-space form posits,
   reached by a different route.
2. **The linear-kernel variance pins at its lower bound (1 × 10⁻⁶) in
   1,642/1,642 unfiltered fits**, and in 821/821 folds of the zero-mean arm as
   well. The data rejects a patient-specific random *slope* while retaining a
   persistent-deviation term — which is what "process noise accumulates" means,
   and is not what "patients differ in slope" means.
3. `src/ppmi_subtype/hbayes.py` records that σ_obs and σ_proc enter the
   likelihood only through their sum. The state-space form is the one in which
   they separate, precisely because accumulating process noise leaves a
   temporal signature — the signature measured above.

### What the manuscript may now say

> Residual deviations from a patient's own linear trend are more persistent
> than independent measurement error would produce (lag-1 correlation −0.219
> against an iid null of −0.333 on the same visit pattern; difference +0.113,
> 95% CI +0.046 to +0.179). This favours a state-space over a growth
> specification, and is a second, independent reason to prefer it — the first
> being that only the state-space form identifies the observation and process
> variances separately.

### What it does not license

The project **never fitted** a state-space model to the ratio. This is a
statement about specification, not a claim that a state-space model would
predict better. The one state-space-like model that *was* fitted — the rich
clinical profile's per-domain local linear trend — found the rate component
worth **+0.0040 of +0.0733** (about 5%) of its gain over last-observation-carried-forward,
with two of nine domains fitting an exactly zero rate variance. On that
evidence the expected gain from switching specification is small.

### Action

`outputs/exploration/report.txt` is stale and `outputs/` is read-only by
design. **Do not regenerate it in place.** The regenerated report lives at
`<E-1 run>/rerun_outputs/exploration/report.txt`; quote that, or quote §5 of
the committed file. The Stage-1 audit's entry C4 should be re-labelled from
"the project contradicts itself" to "a stale artefact contradicts the code".

---

## E-2 — Patient-clustered intervals for the transition analysis

### What was wrong

`outputs/transitions/report.txt` is the source of Result 1 and was the only
analysis block in the project still reporting **Wilson intervals**, which treat
3,361 visit pairs from 934 patients as independent observations. 476 of those
934 patients (51%) never change state.

### Design

One resampling frame, one draw, every statistic recomputed from scratch. All
1,080 patients with at least one labelled visit are drawn with replacement,
carrying **all** their labelled visits; a patient drawn twice becomes two
distinct patients with distinct identifiers, so their visits are never merged
into one longer sequence. Each replicate passes through the same
`ppmi_subtype.transitions` functions that produced the point estimates. 2,000
draws, seed 42, 363 s.

Who qualifies for a given statistic therefore varies from draw to draw — the
934 contributing a transition range over 913–955, the 885 at risk of PIGD
conversion over 861–909 — and that variation is **part of the interval** rather
than being conditioned away. Verified empirically: the replicate patient count
is **fixed at 1,080** while the replicate visit total ranges over **4,153 to
4,737** against a point estimate of 4,441. That combination can only arise if
whole patients are being drawn; a visit-level bootstrap would do the opposite.

**Every point estimate reproduces the committed report exactly** — all nine
transition probabilities, 3,361 pairs, 934 patients, 673 changes, 35.2%
reversion, 885 at risk, 263 and 127 conversions. The script fails if they do
not.

### The nine transition probabilities

| From | To | n | k | p | Wilson 95% | Patient bootstrap 95% | Design effect |
|---|---|---:|---:|---:|---|---|---:|
| TD | TD | 2,261 | 1,827 | 0.808 | 0.7913–0.8238 | **0.7881–0.8264** | **1.38** |
| TD | Indeterminate | 2,261 | 181 | 0.080 | 0.0696–0.0920 | **0.0684–0.0918** | 1.15 |
| TD | PIGD | 2,261 | 253 | 0.112 | 0.0996–0.1256 | **0.0984–0.1275** | 1.23 |
| Indeterminate | TD | 334 | 147 | 0.440 | 0.3879–0.4937 | **0.3809–0.5000** | 1.20 |
| Indeterminate | Indeterminate | 334 | 64 | 0.192 | 0.1530–0.2372 | **0.1484–0.2339** | 1.08 |
| Indeterminate | PIGD | 334 | 123 | 0.368 | 0.3183–0.4212 | **0.3179–0.4169** | 0.91 |
| PIGD | TD | 766 | 141 | 0.184 | 0.1582–0.2131 | **0.1535–0.2180** | 1.36 |
| PIGD | Indeterminate | 766 | 87 | 0.114 | 0.0930–0.1380 | **0.0914–0.1381** | 1.08 |
| PIGD | PIGD | 766 | 538 | 0.702 | 0.6690–0.7337 | **0.6605–0.7380** | **1.46** |

**The design effect exceeds 1 for eight of nine cells**, median **1.20**, range
0.91 to 1.46. So the Wilson intervals were optimistic across most of the table,
by roughly 4–21% in width. The design effect is a per-row quantity, not a
pooled one: for the PIGD row, a design effect of 1.46 on PIGD→PIGD means those
766 pairs carry about as much information as 766 / 1.46 ≈ 525 independent
observations. Figure:
`fig1_wilson_vs_patient_bootstrap.png`, with the plotted values in the
companion CSV beside it.

The two largest design effects are the **persistence cells** — PIGD→PIGD 1.46
and TD→TD 1.38 — which is where the clustering is strongest, because a patient
who never changes state contributes many identical pairs. The one cell below 1
is Indeterminate→PIGD (0.91): a design effect can fall below 1 when a patient's
transitions move the numerator and the denominator together, stabilising the
ratio. Which mechanism dominates is a property of the cell and was not assumed
in advance — a first draft of this script asserted the opposite pattern and was
corrected by the data, so the script now reports the largest and smallest cells
rather than asserting a direction.

A methodological note worth carrying into the paper: the script reports
**two** interval widths, the percentile width and a normal approximation from
the bootstrap SE, and checks that they agree in direction. They did on 9 of 9
cells here. At a small number of draws they do not — a 20-draw pilot gave
design effects above 1 while the percentile widths came out below 1, because
2.5% and 97.5% quantiles cannot be estimated from 20 values. Reporting both is
how a bootstrap that has not converged announces itself.

### The headline reversion rate is unaffected

| | Estimate | 95% interval | Width |
|---|---:|---|---:|
| Wilson | 0.3522 | 0.3170–0.3890 | 0.0720 |
| **Patient bootstrap** | **0.3522** | **0.3180–0.3880** | 0.0700 |

Ratio **0.97×**. This is worth stating plainly in the paper: the project's
single most-quoted number — *35.2% of observed subtype changes reverse at the
very next visit* — **does not depend on the interval method**. State changes are
spread thinly across patients (median 0 changes per patient, mean 1.00), so
there is little clustering left to correct for.

Per-transition-type reversion, patient-clustered:

| Change | p | 95% interval |
|---|---:|---|
| TD → Indeterminate | 0.560 | 0.478–0.642 |
| PIGD → Indeterminate | 0.500 | 0.362–0.648 |
| PIGD → TD | 0.393 | 0.305–0.481 |
| TD → PIGD | 0.320 | 0.253–0.388 |
| Indeterminate → PIGD | 0.183 | 0.103–0.264 |
| Indeterminate → TD | 0.176 | 0.100–0.255 |

The ordering the report describes — changes into the Indeterminate band revert
about half the time, changes crossing the full width of the scale stick — holds
with intervals, though TD→PIGD (0.253–0.388) and PIGD→TD (0.305–0.481) overlap
substantially and should not be described as different.

### Survival and the interval-stratified claim

| Quantity | Estimate | 95% interval |
|---|---:|---|
| KM cumulative incidence of PIGD, year 2, single-visit | 0.1252 | 0.1025–0.1504 |
| KM, year 5, single-visit | 0.3525 | 0.3077–0.3988 |
| KM, year 2, sustained | 0.0654 | 0.0490–0.0846 |
| KM, year 5, sustained | 0.1533 | 0.1200–0.1902 |
| Share of apparent conversions the sustained rule removes | 0.5171 | 0.4572–0.5774 |
| p(TD → PIGD) at ≤ 1.5 years | 0.1004 | 0.0865–0.1161 |
| p(TD → PIGD) beyond 4 years | 0.2889 | **0.1667–0.4318** |

The naive and sustained year-5 intervals are far apart (0.308–0.399 against
0.120–0.190), so the definitional claim survives comfortably. The
interval-stratified claim also survives — the ≤1.5 y and >4 y intervals do not
overlap — **but the >4 y interval is very wide**, spanning 0.17 to 0.43 on only
45 TD-origin pairs. The manuscript should quote the interval rather than the
point estimate of 0.289, and should not imply the gradient is precisely
estimated.

### Action

Table 2 of the manuscript carries the patient-bootstrap interval. No point
estimate moves and no conclusion in the committed report is overturned; the
intervals are now stated over 934 patients rather than 3,361 pairs, and the
project becomes internally consistent, since every other experiment already
resamples patients. Stage-1 audit entry **C1 is closed**; **C2 (interval
pooling) remains open** and is a presentational decision, not a computation.

---

## E-5 — Participant flow

### The flow, every exclusion counted

Every count reproduces the committed artefacts — 16/16 checks, including the
cohort (5,774 / 1,080), labelled visits (4,441), ratio-defined (3,716),
patients with ≥2 labelled visits (934), transitions (3,361), two-class eligible
(3,029) and Indeterminate targets (332).

| Stage | Visits | Patients | Lost at this step |
|---|---:|---:|---|
| PPMI Curated Data Cut, as published | 19,450 | 4,788 | — |
| Sporadic PD subgroup | 5,789 | 1,080 | 13,661 visits / 3,708 participants |
| Scheduled visits, ≤ 15 years | 5,774 | 1,080 | 15 visits (ST 14, PW 1); 0 beyond 15 y |
| With a TD / Indeterminate / PIGD label | 4,441 | 1,080 | **1,333 visits** |
| Patients with ≥ 2 labelled visits | 4,295 | 934 | 146 patients with one visit |
| Consecutive labelled pairs | 3,361 | 934 | — |
| Scored, two-class TD vs PIGD | 3,029 | 899 | 332 Indeterminate targets |

35 of the 934 patients contribute no eligible two-class target. Figure:
`fig1_participant_flow.png`.

### Correction 1 — why visits are unlabelled

README.md and the Weeks 3–4 report state that the label loss splits "roughly
half" between visits with no OFF-medication exam and "exams that were performed
but could not be fully scored", and that "in 1,243 of those 1,253 exams the
missing item is postural stability". **The split does not hold on the frozen
panel.** The second route is 4.3% of the loss, not half.

Read the rest of this section with Correction 2 beside it: the Weeks 3–4
report's *qualitative* findings about that route — that an incomplete exam is
nearly always one missing postural stability, and that such exams come from
more advanced patients — both reproduce. It is the size of the route, and
therefore its weight in the bias argument, that is wrong.

| Why a cohort visit carries no subtype | Visits | Share |
|---|---:|---:|
| **Only an ON-medication Part III exam exists, so it is excluded** | **1,195** | **89.6%** |
| No Part III row in the source file at all | 65 | 4.9% |
| Part III exam partially scored (≥1 item unrateable or absent) | 57 | 4.3% |
| All 13 Part III items present; Part II blocked the label | 16 | 1.2% |

Among the 57 partially scored exams, postural stability (`NP3PSTBL`) is the
missing item in **54** (94.7%) and the *only* missing Part III item in 42. So
the postural-stability mechanism is **real and correctly described in kind —
but it accounts for 54 visits, not 1,243.**

The dominant mechanism is the OFF-medication requirement in
`subtype.select_medication_state`, which excludes an exam recorded ON
medication in a treated patient. This is confirmed independently from the
curated cut's own columns, which were not used to derive it:

| Curated-cut column | Present at unlabelled visits | Present at labelled visits |
|---|---:|---:|
| `hy` (OFF-state Hoehn & Yahr) | **6.5%** | 99.8% |
| `hy_on` (ON-state) | 93.2% | 94.9% |
| `updrs3_score` (OFF Part III total) | **3.5%** | 99.4% |
| `updrs3_score_on` (ON) | 85.4% | 94.5% |

The OFF-state columns are near-absent exactly where the label is absent, and
the ON-state ones are present. These are ON-state visits.

### Correction 2 — who is lost, and which way the bias runs

| | Unlabelled | Labelled | Test |
|---|---:|---:|---|
| LEDD at the visit, median | **500** | 300 | p = 1.0 × 10⁻⁹⁸ |
| Share on PD medication | **96.8%** | 57.9% | — |
| Years from baseline, median | **5.0** | 2.0 | p = 2.8 × 10⁻¹⁴⁹ |
| Hoehn & Yahr (`hy`, else `hy_on`), median | 2.0 | 2.0 | p = 8.4 × 10⁻³⁰ |
| Hoehn & Yahr, mean | 2.03 | 1.85 | — |

**On the Hoehn & Yahr claim I was wrong the first time, and the correction
matters.** The table above compares *all* unlabelled visits with *all* labelled
ones, where the medians are equal at 2.0. But the Weeks 3–4 report's figure is
not about that group: it compares **exams recorded but incomplete** against
**complete exams**. Tested on the group it actually describes, it reproduces
almost exactly:

| Hoehn & Yahr | n | Median | Mean |
|---|---:|---:|---:|
| Incomplete OFF/untreated exams | 42 | **3.0** | 2.57 |
| Complete OFF/untreated exams | 4,451 | **2.0** | 1.85 |

Mann–Whitney **p = 6.4 × 10⁻¹⁸**, against the report's 1.4 × 10⁻¹⁷. So the
Weeks 3–4 report's severity finding is **sound**, and so is its qualitative
claim that an incomplete exam is nearly always one missing postural stability
(54 of 57 here, 94.7%, against the report's 1,243 of 1,253, 99.2%).

**What does not reproduce is the magnitude.** The report says the loss splits
"roughly half" between visits with no OFF exam and exams that could not be
fully scored. Measured on the frozen panel the split is **1,260 to 57** — 94.5%
against 4.3%. The 1,253-exam group cannot be reconstructed under any
denominator tried: the whole OFF/untreated Part III set gives 3,237 incomplete
exams (3,186 missing `NP3PSTBL`), and the sporadic-PD cohort gives 57 (54). The
counts predate the 2026-10-14 data transfer and no longer hold.

**The direction of the bias argument changes, and the conclusion survives by a
better route.**

- *README's argument:* lost visits are disproportionately PIGD, because
  postural stability is a PIGD item and is untestable in patients with poor
  balance → progression estimates are a lower bound on the drift.
- *Corrected argument:* that mechanism is real but governs **57 of 1,333**
  visits. The bulk of the loss is **treated patients later in follow-up**
  (96.8% on medication against 57.9%; median year 5.0 against 2.0), so the
  analysed sample is weighted toward earlier, untreated follow-up, which
  **attenuates observed progression** → progression estimates remain a lower
  bound on the drift.

Both routes therefore point the same way, which is why the conclusion survives;
but they are different sizes and different mechanisms, and only the smaller one
was named. The OFF requirement itself is correct and must be kept: levodopa
suppresses tremor, so scoring ON exams would manufacture PIGD transitions that
are pharmacological rather than pathological. The manuscript should describe the
selection as **treatment-driven**, with the postural-stability route as a
secondary, severity-driven contribution at its true scale.

### Correction 3 — the sentinel count, settled

The Stage-1 audit flagged `NP3PSTBL` sentinel-101 counts of 4,302 (config.yaml)
and 3,206 (README) as an unexplained discrepancy. Three denominators, three
numbers, and **both published figures are correct for the set they describe**:

| Set | Rows | `NP3PSTBL` = 101 |
|---|---:|---:|
| Raw Part III file | 39,310 | **4,302** ← config.yaml |
| After OFF/untreated selection | 29,356 | **3,206** ← README.md |
| After duplicate resolution | 37,886 | 3,205 |

The manuscript should state the denominator with the number. Closed.

---

## What changes in the Stage-1 audit

| Entry | Was | Now |
|---|---|---|
| **C1** transition intervals ignore clustering | Critical, open | **Closed.** Patient-clustered intervals computed; design effect 1.20 median, 8 of 9 cells above 1. |
| **C4** the project contradicts itself on the residual autocorrelation | Critical, open | **Closed, and re-labelled.** The code is self-consistent; a stale output contradicts it. §5 is correct. |
| **E-5** participant flow does not exist | Essential, missing | **Closed.** Figure and full ledger produced. |
| **I2** missingness is driven by unscorable postural stability | Important limitation | **Rewritten.** The dominant route is the OFF-medication requirement (89.6%); the postural-stability route is 57 visits (4.3%), where the Weeks 3–4 report's own findings do reproduce. Conclusion (lower bound on drift) survives, now via treatment-and-follow-up selection. |
| Minor: `NP3PSTBL` 4,302 vs 3,206 | Unexplained | **Closed.** Different denominators; both correct. |
| **C2** pooled transition matrix averages unequal intervals | Critical, open | **Still open.** E-2 confirms the gradient is real (≤1.5 y and >4 y intervals do not overlap) but the >4 y estimate is imprecise (0.167–0.432 on 45 pairs). Presentational decision. |
| **C3** Bayesian model has no publishable MCMC diagnostics | Critical, open | **Still open** — needs a multi-chain refit (E-3). |
| **C5** no ethics / funding / COI documentation | Critical, open | **Still open** — needs you (E-4). |
| **C6** 3-class agreement cannot be reproduced | Critical, open | **Still open** — `data/validation/` is empty. |

Two further documentation defects were found and are **not** in the Stage-1
audit, because they were only visible once the pipeline was replayed:

1. README.md's account of label missingness (Correction 1 and 2 above).
2. README.md's "403 tests (2026-10-14)"; the suite is now **758 tests, all
   passing**.

Neither is in the manuscript yet, so neither needs an erratum — but the README
should be corrected before anyone else reads it as the project's summary.

---

## Reproduce

```bash
~/envs/ppmi/bin/python scripts/76_lag1_residual_resolution.py --out experiments/$(date -u +%Y%m%dT%H%M%SZ)_e1_lag1_residual_resolution
~/envs/ppmi/bin/python scripts/77_transition_patient_bootstrap.py --out experiments/$(date -u +%Y%m%dT%H%M%SZ)_e2_transition_patient_bootstrap
~/envs/ppmi/bin/python scripts/78_participant_flow.py --out experiments/$(date -u +%Y%m%dT%H%M%SZ)_e5_participant_flow
```

E-1 takes about a minute, E-2 about six (2,000 bootstrap draws), E-5 about one.
Each writes `report.txt`, `summary.json`, `validation_checks.csv` and a
`MANIFEST.sha256` over its own outputs, and each exits non-zero if any check
fails. None of them writes under `outputs/`.
