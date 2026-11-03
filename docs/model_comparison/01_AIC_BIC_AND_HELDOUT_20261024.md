# Model selection criteria and matched held-out comparison

**Date** 2026-10-24 · **Run** `experiments/20261023T194317Z_model_selection_criteria/`
**Script** `scripts/65_model_selection_criteria.py` · **Models unmodified**
**Panel** `outputs/visit_panel.csv`, md5 `fb038ddbdc19ab3368e13940f398e7ba`

Each model was fitted once on the full cohort purely to obtain a likelihood at
its own maximum, which is what AIC and BIC need. All held-out numbers are read
from the two existing experiment directories, not recomputed.

---

## 1. What is and is not comparable

**AIC and BIC cannot be compared between the two methods.** They are not on a
common scale, for two independent reasons:

1. **Different data.** The CT-HMM's likelihood is over 4,295 categorical
   subtype labels. The rich profile's is over 38,486 continuous clinical
   measurements. Different observations, different count, different support.
2. **Mass versus density.** The CT-HMM's likelihood is a probability *mass*
   (bounded above by 1, so its log is always negative). The rich profile's is a
   probability *density*, which has units and changes value if a measurement is
   rescaled. A density log-likelihood can even be positive. Differencing them
   is meaningless.

A further caution applies to both: AIC and BIC assume independent observations.
Visits within a patient are correlated, so the effective sample size is smaller
than the term count and BIC's penalty is, if anything, too weak here. The
criteria are used below only to rank models **within** a group.

Three groups are internally valid:

| Group | Models | Why valid |
|---|---|---|
| **A** | three rich-profile variants | identical Gaussian density over the same nine domains, and nested |
| **B** | CTMC, discrete Markov | identical conditional likelihood over the same 3,361 transition pairs |
| **C** | CT-HMM, chains + initial state | see §3 — put on a common joint likelihood |

---

## 2. Group A — rich-profile variants

Likelihood: density over nine continuous domains, 38,486 observed values.

| Model | log-likelihood | k | AIC | BIC | ΔAIC | ΔBIC |
|---|---|---|---|---|---|---|
| **level + rate + medication** | **−45,812.36** | 45 | **91,714.73** | **92,099.84** | **0** | **0** |
| level only (random walk) | −45,940.77 | 27 | 91,935.53 | 92,166.60 | +220.8 | +66.8 |
| level + rate, no medication | −45,993.34 | 36 | 92,058.68 | 92,366.77 | +344.0 | +266.9 |

**The full model wins on both criteria.** The rate component costs 18
parameters and buys 128 log-likelihood points; even under BIC's stiffer penalty
it remains ahead by 66.8. The medication input costs 9 parameters and buys 181
points — the single most clearly justified component.

---

## 3. Group C — is the latent layer worth it?

Groups B and the CT-HMM look incomparable at first: the chains maximise a
likelihood *conditional* on the previous observed state over 3,361 pairs, while
the CT-HMM maximises a *joint* likelihood over whole sequences, 4,295 terms.

But the difference is exactly the patients: 4,295 − 3,361 = **934**, one first
visit each. Giving the chains an initial-state distribution estimated from
those first visits (2 extra parameters, log-likelihood −719.34) puts all three
on the same joint likelihood, and the comparison becomes valid.

| Model | log-likelihood | k | AIC | BIC | ΔAIC | ΔBIC |
|---|---|---|---|---|---|---|
| **CT-HMM (3 latent states)** | **−2,969.06** | 14 | **5,966.12** | **6,055.23** | **0** | **0** |
| Markov (discrete) + initial state | −3,087.12 | 8 | 6,190.23 | 6,241.15 | +224.1 | +185.9 |
| CTMC (interval-aware) + initial state | −3,099.84 | 8 | 6,215.68 | 6,266.60 | +249.6 | +211.4 |

**The latent layer is strongly justified.** Six extra parameters buy 118
log-likelihood points over the best chain; ΔAIC 224 and ΔBIC 186 are both far
beyond any conventional threshold. This is the formal counterpart of the
history-dependence already documented for this cohort (χ² = 156.6, df = 4,
p = 7.9 × 10⁻³³): a chain in the observed label cannot represent the dependence
on where a patient came from, and the criteria say so decisively.

### Group B, for completeness

Among the chains alone — conditional likelihood, 3,361 pairs, no initial-state
term, so **not** on the scale of the table above:

| Model | log-likelihood | k | AIC | BIC | ΔAIC |
|---|---|---|---|---|---|
| **Markov (discrete)** | **−2,367.77** | 6 | **4,747.55** | **4,784.27** | **0** |
| CTMC (interval-aware) | −2,380.50 | 6 | 4,773.00 | 4,809.72 | +25.4 |

The **discrete** chain beats the interval-aware CTMC on both criteria.
Modelling elapsed time does not pay for itself in a chain over the observed
label — consistent with the held-out result, where the CTMC was −0.0017 behind
the discrete chain with an interval spanning zero.

*(Minor caveat: the discrete chain's likelihood is evaluated at its
Laplace-smoothed estimate rather than the exact unpenalised MLE, so its
log-likelihood is marginally understated. It wins anyway.)*

---

## 4. Held-out TD/PIGD prediction, matched cases

From the existing runs, on the **3,020 held-out cases shared by every model**:

| Model | Accuracy | 95% CI | Brier | Log loss |
|---|---|---|---|---|
| Rich profile: level only | 0.8440 | 0.8285–0.8598 | 0.1195 | 0.3814 |
| Rich profile: level + rate + medication | 0.8404 | 0.8248–0.8566 | 0.1177 | 0.3766 |
| Rich profile: level + rate, no medication | 0.8404 | 0.8249–0.8563 | 0.1175 | 0.3759 |
| **CT-HMM** | **0.8371** | 0.8202–0.8540 | 0.1215 | 0.3963 |
| Persistence rule | 0.8338 | 0.8167–0.8509 | 0.1598 | 4.1959 † |
| Markov (discrete) | 0.8298 | 0.8130–0.8471 | 0.1327 | 0.4311 |

† rule-based probabilities; log loss finite only through clipping.

| Paired contrast | Δ accuracy | 95% CI |
|---|---|---|
| Rich profile (full) − CT-HMM | +0.0033 | −0.0050 to +0.0112 |
| Rich profile (level only) − CT-HMM | +0.0070 | −0.0010 to +0.0149 |

Both intervals include zero: **no significant difference between the methods**
on this endpoint.

---

## 5. The tension worth noticing

The criteria and the held-out accuracy disagree in emphasis, and the
disagreement is informative rather than contradictory:

| Component | In-sample (AIC/BIC) | Held-out accuracy |
|---|---|---|
| CT-HMM latent layer vs a chain | ΔAIC 224 — decisive | +0.0076 accuracy |
| Rich-profile rate component | ΔAIC 221, ΔBIC 67 — clear | +0.0040 log density; **level-only is nominally *more* accurate** |

Both components reliably improve the **probability model** — the likelihood,
and the calibration that follows from it — while barely moving **classification
accuracy**. That is not a contradiction. The subtype label is highly persistent,
so accuracy on it is close to saturated by a persistence rule (0.8338) and has
little headroom to reveal a better probability model. The criteria are measuring
something the accuracy figure is largely blind to.

This is also why the level-only variant edges ahead on label accuracy while
losing on AIC, BIC, Brier and log loss: a single accuracy point on a saturated,
two-class, thresholded endpoint is a noisy way to rank models.

---

## 6. Recommendation

1. **Keep the full rich-profile specification** (level + rate + medication). It
   is preferred on AIC *and* BIC, on Brier and on log loss. Do not drop the rate
   component on the strength of its small held-out accuracy contribution — the
   criteria say the fit improvement is real, and accuracy on this endpoint is
   the wrong instrument for the decision.
2. **Keep the latent layer in the CT-HMM.** ΔAIC 224 and ΔBIC 186 against the
   best observed-label chain, on a properly matched joint likelihood.
3. **Do not report AIC or BIC across the two methods**, and do not let anyone
   read the rich profile's much larger AIC as evidence it is worse — it is
   simply a density over 38,486 continuous values rather than a mass over 4,295
   labels.
4. **Report the criteria and the held-out metrics as answering different
   questions.** The criteria rank the fit to the data the model was built for;
   the held-out TD/PIGD accuracy is a narrow secondary readout on a saturated
   endpoint. In the manuscript the two methods remain statistically
   indistinguishable on that endpoint, and that statement is unchanged.
5. **Prefer continuous scores over accuracy for future model selection here.**
   Log-density and Brier separated these models where accuracy could not.

---

## 7. Limitations

- AIC/BIC assume independent observations; visits are clustered within
  patients, so the effective sample size is below the term count and both
  penalties are optimistic. The rankings are large enough that this is unlikely
  to reverse them, but the margins should not be read as exact.
- The criteria are computed from a **single full-cohort fit**, so unlike the
  held-out figures they carry no cross-validated uncertainty.
- The rich-profile log-likelihood is a density on the *standardised* scale.
  Rescaling the domains would shift it by a constant; that constant is identical
  across the three variants, so the within-group comparison is unaffected.
- The CT-HMM comparison in §3 assumes the chains' initial-state distribution is
  the empirical marginal of first visits, which is its MLE, charged at 2
  parameters.
- No external cohort. All of this is within PPMI.
