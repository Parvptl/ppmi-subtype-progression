# Do the forced TD/PIGD calls on Indeterminate visits track the nearer boundary?

**Date** 2026-10-24 · **Run** `experiments/20261023T203431Z_indeterminate_boundary/`
**Script** `scripts/67_indeterminate_boundary_analysis.py` · **Neither model retrained or modified**

## Verdict

**The hypothesis is not supported.** For both models, the agreement between the
forced TD/PIGD call and the nearer boundary is statistically indistinguishable
from what independence would give. A weak *ranking* signal exists in the
underlying probability, but it is marginal and does not survive into the hard
call.

---

## 1. The question

A visit is labelled Indeterminate when the tremor/PIGD ratio falls inside
(0.90, 1.15). Those visits are excluded from the project's binary scoring
because neither class is the right answer. But each still sits closer to one
boundary than the other. If a model's forced call tracked that position, the
prediction would carry information even where it cannot be scored.

Tested on held-out predictions only, with an explicit null.

## 2. The analysis set

| | n |
|---|---|
| Indeterminate-target cases shared by both models | 332 |
| Ratio undefined (PIGD subscore = 0, both subscores zero) | 45 — **excluded** |
| **Analysed** | **287** (220 patients) |

The 45 excluded cases have *no* ratio, so no distance to a boundary exists for
them. They are counted, not quietly dropped.

## 3. A structural finding that shapes everything else

**Inside the band the ratio is nearly binary.** The tremor score is a mean of
11 integer items and the PIGD score a mean of 5, so within a window only 0.25
wide very few rational values are reachable:

| Ratio | n | share | distance to PIGD (0.90) | distance to TD (1.15) | nearer |
|---|---|---|---|---|---|
| **0.9091** | **144** | **50.2%** | **0.0091** | 0.2409 | PIGD |
| 0.9740 | 2 | 0.7% | 0.0740 | 0.1760 | PIGD |
| 0.9848 | 4 | 1.4% | 0.0848 | 0.1652 | PIGD |
| 1.0000 | 6 | 2.1% | 0.1000 | 0.1500 | PIGD |
| 1.0227 | 16 | 5.6% | 0.1227 | 0.1273 | PIGD |
| 1.0390 | 2 | 0.7% | 0.1390 | 0.1110 | TD |
| 1.0606 | 35 | 12.2% | 0.1606 | 0.0894 | TD |
| 1.0795 | 1 | 0.3% | 0.1795 | 0.0705 | TD |
| 1.0909 | 8 | 2.8% | 0.1909 | 0.0591 | TD |
| 1.1039 | 2 | 0.7% | 0.2039 | 0.0461 | TD |
| **1.1364** | **67** | **23.3%** | 0.2364 | **0.0136** | TD |

Three quarters of Indeterminate visits sit at just two values — 0.9091, a
hundredth above the PIGD boundary, and 1.1364, a hundredth below the TD
boundary. "Distance to the nearer boundary" is therefore not a continuous
quantity here; it is close to a binary label.

Nearer PIGD: 172 (59.9%). Nearer TD: 115 (40.1%).

## 4. Main result

| Model | Predicted PIGD | Agreement with nearer boundary | Expected if independent | **Excess** | 95% CI |
|---|---|---|---|---|---|
| CT-HMM | 89 / 287 (31.0%) | 0.4878 [0.4336, 0.5398] | 0.4623 | **+0.0255** | **−0.0287 to +0.0775** |
| Rich clinical profile | 101 / 287 (35.2%) | 0.5017 [0.4483, 0.5588] | 0.4706 | **+0.0312** | **−0.0223 to +0.0882** |

**Both intervals include zero.** Neither model's forced call tracks the nearer
boundary at a level distinguishable from chance.

The null is the agreement expected if the calls were independent of position in
the band, computed from each model's own PIGD-call rate and the observed
split — so it already accounts for the fact that both models mostly say TD.

### Counts and percentages

| Model | Nearer boundary | Predicted PIGD | Predicted TD | % called PIGD |
|---|---|---|---|---|
| CT-HMM | PIGD (n=172) | 57 | 115 | **33.1%** |
| CT-HMM | TD (n=115) | 32 | 83 | 27.8% |
| Rich profile | PIGD (n=172) | 65 | 107 | **37.8%** |
| Rich profile | TD (n=115) | 36 | 79 | 31.3% |

The direction is right for both — PIGD is called more often when the visit sits
nearer the PIGD boundary — but the gaps (5.3 and 6.5 points) are small relative
to their uncertainty.

## 5. Sensitivity: only visits genuinely close to a boundary

The midpoint rule labels a visit at ratio 1.0227 "nearer PIGD" by a margin of
0.0046, which is noise. Restricting to the 213 visits within 0.05 of a
boundary, where the construct is meaningful:

| Model | Agreement | Excess over null | 95% CI |
|---|---|---|---|
| CT-HMM | 0.4742 [0.4079, 0.5392] | +0.0378 | −0.0285 to +0.1029 |
| Rich clinical profile | 0.5070 [0.4430, 0.5714] | +0.0624 | **−0.0017** to +0.1268 |

Still null for both. The rich profile's lower bound moves to −0.0017 — close to
excluding zero, but not excluding it. **That is a near miss, not a result**, and
it should not be reported as one.

## 6. The one place a signal does appear

A threshold-free ranking measure asks a weaker question: does the predicted
`P_PIGD` rank nearer-PIGD visits above nearer-TD ones?

| Model | AUC for "nearer PIGD" | 95% CI |
|---|---|---|
| CT-HMM | 0.5843 | **0.5162 – 0.6534** |
| Rich clinical profile | 0.5748 | **0.5054 – 0.6403** |

Both intervals exclude 0.5, so there is **weak ranking information** in the
probabilities. Three cautions before anyone leans on this:

1. The effect is small — AUC under 0.59, against 0.5 for no information.
2. The rich profile's lower bound is 0.5054. That is as marginal as a positive
   result gets.
3. **No multiplicity correction** was applied across the several tests here.

The reason ranking shows something while agreement does not is the threshold:
both models call TD for about two thirds of this subgroup while 60% of it sits
nearer the PIGD boundary, so the 0.5 cut is badly placed for these cases and
discards what little ordering exists.

## 7. Conclusion

**The stated hypothesis is not supported.** Neither model's forced TD/PIGD call
on an Indeterminate visit tracks the nearer boundary at a rate distinguishable
from chance, in the full set or in the subset where the construct is
meaningful.

A weak ranking signal exists in the underlying probability (AUC ≈ 0.57–0.58,
intervals just excluding 0.5), which is consistent with the earlier finding that
`p_Indeterminate` itself carries some information (AUC 0.68) even though
Indeterminate is never predicted. But it is too weak, and too close to its
threshold, to support a claim that these excluded predictions are informative
about position within the band.

**Practical reading:** excluding Indeterminate targets from scoring remains the
right call. These analyses give no basis for recovering partial credit on them
from the models' forced binary predictions.

## 8. Limitations

- **The ratio is nearly binary inside the band** (§3), so "distance to the
  nearer boundary" has far less resolution than the phrasing suggests. The
  sensitivity analysis in §5 is the more meaningful test.
- 45 of 332 Indeterminate targets have **no defined ratio** and are excluded;
  conclusions cover the other 287.
- The two models come from **different validation schemes**; matching the cases
  removes the evaluation-set difference but not the scheme difference.
- **No multiplicity correction** across the agreement, sensitivity and AUC
  tests. With six intervals reported, one marginal exclusion of the null is
  weak evidence.
- Intervals are patient-clustered percentile bootstraps (2,000 draws), no bias
  correction.
- The actual ratio at the target visit is itself a noisy measurement; a visit at
  0.9091 might have been recorded at 0.88 on a different day, in which case it
  would have been PIGD rather than Indeterminate.

## 9. Files

`fig_ratio_distribution.png`, `fig_model_comparison.png`,
`ratio_values.csv`, `boundary_agreement.csv`, `counts_and_percentages.csv`,
`sensitivity_close_to_boundary.csv`, `summary.json`, `report.txt`,
`MANIFEST.sha256` (8/8 verified).
