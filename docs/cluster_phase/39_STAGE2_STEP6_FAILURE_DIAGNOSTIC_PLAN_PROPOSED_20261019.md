# Stage 2 M2: why full-pipeline recovery with S1-b failed. Diagnostic investigation plan

**Date:** 2026-10-19.

> **PROPOSED — NOT FROZEN — NO COMPUTATION AUTHORISED.** Diagnostic only.
> - The Step 6 result (doc 38: COMPLETE, no flagged event, formal verdict **FAIL**) is final for that run. It is not rerun, and its outputs and criterion are not changed.
> - Nothing here is a repair, selects a method, or authorises Step 7 or Step 8.
> - S1-c, or any other repair, would need evidence **and** a separate amendment. None is proposed here.

**Tags:**
- **[E]** a descriptive finding computed from existing saved outputs;
- **[H]** a hypothesis not established by existing outputs;
- **[N]** needs new computation.

Five datasets do not support strong correlational or causal claims. Every rank statistic below is descriptive.

---

## 1. Sources read (all unchanged; frozen locks verified OK)

- Frozen: doc 16, Rev 2 (doc 21), Addendum A (doc 25), doc 27, and the Step 6 plan (doc 37, `c02be4ce…`).
- Step 6 report: doc 38. Its per-dataset records are in `experiments/20261018T222857Z_stage2_step6_full_pipeline_recovery/step6/`: `S1_dataset{r}.json` (ψ̂⁽¹⁾, ψ̂⁽²⁾, H⁽¹⁾, H⁽²⁾, both item sets, fitted-mixture moments) and the CSV tables.
- Step 4 (fixed true items): `…_stage2_step4_fixed_item_recovery/step4_run2/`.
- Step 5 (S1-a): `…_stage2_step5_s1a/step5/`, including the saved true-latent ("oracle") items and the first-order shift table.
- Pilot recovery: `…_stage2_m2_pilot/results/step1_recovery.json` (old Laplace likelihood, N(0, 1) items, no S1-b).

**How the Phase 1 numbers were computed.** They come from ad-hoc read-only computations on these files during plan preparation: no model was fitted and no data generated. If this plan is approved, D1–D3 (§3) re-run them as a scripted, saved, hash-recorded analysis.

## 2. Phase 1: findings from existing outputs

### 2.1 Point estimates and SEs across datasets [E]

ψ̂⁽²⁾ − truth (z):

| dataset | β0_P | Le00_log | Le11_log |
|---|---|---|---|
| 0 | +0.081 (+2.78) | −0.044 (−1.77) | −0.064 (−1.81) |
| 1 | +0.040 (+1.47) | −0.038 (−1.53) | −0.080 (−2.45) |
| 2 | +0.129 (+4.49) | −0.071 (−3.00) | −0.119 (−3.35) |
| 3 | +0.050 (+1.66) | +0.021 (+0.88) | −0.010 (−0.29) |
| 4 | +0.052 (+1.82) | −0.077 (−3.14) | −0.091 (−2.64) |
| SE range | 0.0275–0.0300 | 0.0235–0.0247 | 0.0327–0.0355 |

- **Signs:** consistent in 5/5 datasets for β0_P and Le11_log, and in 4/5 for Le00_log (dataset 3 is the exception).
- **Dataset 2** has the largest error on all three. **Dataset 3** has the smallest.

### 2.2 Already present after the first stage-2 fit; S1-b changes little [E]

| | β0_P | Le00_log | Le11_log |
|---|---|---|---|
| Step 4, true items: mean z | −0.08 | −0.23 | −0.29 |
| pilot (old likelihood, N(0, 1) items): mean z | +2.35 | −2.26 | −3.34 |
| ψ̂⁽¹⁾ (Rev 2 likelihood, N(0, 1) items): mean z | **+2.19** | **−1.47** | **−1.98** |
| ψ̂⁽²⁾ (after S1-b): mean z | +2.44 | −1.71 | −2.11 |
| ψ̂⁽²⁾ − ψ̂⁽¹⁾, in SE: per dataset | +0.10, −0.32, +0.87, +0.16, +0.15 | −0.30, −0.36, −0.89, +0.38, −0.01 | +0.27, −0.44, −0.44, +0.34, −0.33 |

- The biases exist **before** S1-b.
- On average S1-b does not reduce them and moves them slightly further from the truth, by +0.25, −0.24 and −0.13 SE. Its effect varies in sign across datasets; dataset 2 worsens on all three.
- Comparing the pilot with ψ̂⁽¹⁾ (same N(0, 1) items, different stage-2 likelihood): the Rev 2 likelihood reduced the Le11_log and Le00_log biases but not the β0_P bias.

### 2.3 Fitted PIGD latent mean and variance [E]

The pooled moments of the fitted visit-level mixture at ψ̂⁽¹⁾ (the S1-b anchor), for PIGD:

| | value | truth, by construction |
|---|---|---|
| variance | 0.851, 0.881, 0.872, 0.895, 0.929 | 1 |
| mean | +0.014 to +0.060 | 0 |

- All 5 variances are below 1.
- **Rank agreement with the ψ̂⁽²⁾ z (n = 5, descriptive only):** Le11_log ρ = +0.6; Le00_log +0.4; β0_P +0.3. The dataset with the largest variance (3, 0.929) has the smallest errors.
- This **does not establish** the anchoring hypothesis (§4 H-A).

### 2.4 Item errors and their structure [E]

**Affine relabelling of the latent scale.** If the estimated items equal the true items on a relabelled latent scale θ′ = (θ − d)/c, then â = c·a and τ̂ = (τ − d)/c. Regressing τ̂ on τ gives c and d per subscore. Raw signed item errors are reported as well.

| item set | PIGD c (thresholds) | PIGD d | PIGD τ RMSE: raw → after affine fit | PIGD mean a error |
|---|---|---|---|---|
| N(0, 1) stage 1 | 1.06–1.18 | −0.16 to 0.00 | 0.07–0.18 → 0.03–0.08 | +0.221 (pooled) |
| S1-b | 1.00–1.14 | −0.13 to +0.06 | 0.06–0.16 → 0.03–0.04 | +0.174 |
| Step 5 true-latent ("oracle") | 0.96–1.07 | −0.05 to +0.08 | — | +0.007 |

- **Most of the PIGD threshold error is a common affine relabelling (c > 1):** a compressed latent scale with inflated discriminations.
- The discrimination-based c (1.03–1.12) agrees with the threshold-based c in direction.
- With the true latent distribution (Step 5), c ≈ 1 and d ≈ 0.

**Applying the relabelling to the truth, ψ_true → ψ′(c, d), predicts these ψ errors (mean z over 5 datasets, S1-b items vs observed ψ̂⁽²⁾):**

| | predicted | observed |
|---|---|---|
| β0_P | +2.71 | +2.44 |
| Le11_log | −1.97 | −2.11 |
| L22_log | −1.66 | −1.41 |
| L33_log | −0.92 | −0.88 |
| **Le00_log** | **−0.67** | **−1.71** |
| β1_P | −1.08 | −0.52 |

- The relabelling accounts for the mean bias of β0_P and Le11_log but **not** Le00_log.
- With the N(0, 1) items it **overshoots** (β0_P predicted +4.69 vs +2.19 observed), so it is only a crude account.
- **Rank agreement with the ψ̂⁽²⁾ z (n = 5):** Le11_log vs PIGD c, ρ = −1.0; Le11_log vs PIGD mean a error, ρ = −0.9; Le00_log vs tremor c, ρ = −0.8; β0_P vs PIGD c, ρ = +0.5.

### 2.5 Bias, small SEs, or both? [E]

**SEs.** The Step 6 SEs equal the fixed-true-item Step 4 SEs to within a ratio of 0.891–1.017 (all 19 parameters and 5 datasets). By construction they omit item uncertainty (doc 16 B5).

**Bias.** The mean errors are 2.4, 1.7 and 2.1 SE. **Intervals centred on the mean error** (|z − mean z| ≤ 1.96) would cover 4/5, 4/5 and 5/5. The dispersion around the bias is therefore roughly compatible with the reported SE.

**Dispersion.** The dataset-to-dataset SD of the errors relative to the mean SE is 1.25, 1.60 and 1.17 at ψ̂⁽²⁾ (0.86, 1.33 and 0.86 at ψ̂⁽¹⁾). With n = 5 an SD ratio is uncertain by a factor of about 0.6–2.9, so this is uninformative.

**SE inflation needed to meet ≥ 4 of 5:** ×1.42 (β0_P), ×1.53 (Le00_log), ×1.35 (Le11_log).

**Descriptively, the failure is predominantly a systematic location bias,** with SEs that are not visibly too small given n = 5. Whether item uncertainty would inflate the SEs by about 35–55 % is **not** answerable from existing outputs (D4).

### 2.6 Item estimation or stage-2 identifiability? [E with limits]

**Pointing to the item step:**
- Fixed true items give no bias (Step 4: these three are 5/5, |mean z| ≤ 0.3).
- The ψ̂⁽¹⁾ − ψ̂_Step4 shift is reproduced **per dataset, within about 0.2 SE**, by Step 5's first-order prediction from the N(0, 1) item errors:

  | | predicted mean | actual mean |
  |---|---|---|
  | β0_P | +2.22 | +2.19 |
  | Le00_log | −1.26 | −1.25 |
  | Le11_log | −1.77 | −1.70 |

- Step 5 split that shift into the N(0, 1)-assumption part (N01 − oracle) and the item-sampling part (oracle − true):

  | | N(0, 1)-assumption part | item-sampling part |
  |---|---|---|
  | β0_P | +2.16 | +0.06 |
  | Le11_log | −1.78 | +0.01 |
  | Le00_log | −0.50 | −0.76 |

- So, descriptively, **β0_P and Le11_log are almost entirely attributable to the N(0, 1) working assumption. Le00_log is mixed, and its larger part comes from item sampling error** (the 11 tremor items).

**Not pointing to within-stage-2 compensation:** at fixed items, the estimator correlations (from H⁽²⁾⁻¹) among the three failing parameters, and between them and L22/L33/β1_P/γ_P, are small: |r| ≤ 0.13 among the three failing parameters. The larger pairs involve other components: β1_P–γ_P −0.49, β0_P–γ_P −0.33, β0_P–L22_log −0.24.

**Limits:**
- H⁽²⁾ holds the items fixed. It cannot show coupling between items and ψ, and that coupling is structural here: the joint model is invariant under affine relabelling of θ with compensating item changes.
- The affine analysis is crude: a single (c, d) per subscore, fitted to thresholds only.
- Step 5's decomposition is first order. Step 5 is diagnostic only and is used here descriptively.

## 3. Phase 2: proposed diagnostics (smallest set; read-only first)

| id | question | data | existing or new | cost |
|---|---|---|---|---|
| **D1** | Is the item error mostly a common affine relabelling of each subscore's latent scale, and does that relabelling, applied to ψ_true, reproduce the ψ̂⁽¹⁾ and ψ̂⁽²⁾ errors component by component? | saved items (stage 1, S1-b, Step 5 oracle), it_true, ψ_true, ψ̂, SE | **existing, read-only** | about 1 min, login node |
| **D2** | How much of each failing parameter's deviation is the N(0, 1)-assumption part vs the item-sampling part, and how well does first order predict the actual fits? | Step 5 shift table; Step 4, 6 and pilot ψ̂ and SE | **existing, read-only** | about 1 min |
| **D3** | Bias vs dispersion: mean error/SE, centred coverage, SD ratios with exact χ² intervals for n = 5, and the SE inflation needed per component | Step 4/6 and pilot ψ̂, SE | **existing, read-only** | about 1 min |
| **D4** | Would accounting for stage-1 item uncertainty (a two-step, Murphy–Topel-type correction) widen the ψ̂⁽²⁾ intervals enough to cover? How large is the item–ψ coupling? | the 5 datasets (regenerated from the frozen seeds; no new draws), the saved items and ψ̂⁽²⁾ | **[N]** | about 0.3 node-hours |
| **D5** | Does the S1-b map have a fixed point near ψ̂⁽²⁾ (anchoring), or does it move towards the truth with more iterations (one iteration insufficient), or diverge? | the 5 datasets, saved ψ̂⁽²⁾ | **[N]** | about 1.5 node-hours |
| **D6** | If the items were on the correct latent scale (Step 5 oracle items, already saved), would stage 2 recover ψ? Is the latent-scale error sufficient to explain the failure, as a full refit rather than first order? | the 5 datasets, saved oracle items | **[N]** | about 0.4 node-hours |

**Details:**

- **D1 (read-only).** A pre-specified version of §2.4 with a saved script and outputs:
  - (c_s, d_s) fitted jointly to log â − log a and τ̂ − (τ − d)/c by least squares, plus threshold-only and discrimination-only variants;
  - the residual non-affine item error;
  - the predicted vs observed ψ errors for all 19 components.
  - *Interpretation:* if the affine part explains most of the item error **and** reproduces the β0_P / Le11_log / L22 / L33 pattern, that supports **H-A** (a latent-scale anchoring error from the N(0, 1) stage 1, not fully undone by S1-b). Large residuals, or a poor ψ prediction, point to **H-B** (item-specific, non-affine error).
  - *Limits:* it describes, it does not establish cause. A single affine map per subscore is an approximation, and there is no interval for c or d.

- **D2 (read-only).** It tabulates §2.6 per component and dataset, including the Le00_log split.
  - *Interpretation:* a large N01 − oracle part supports H-A; a large oracle − true part points to item sampling error (more relevant to Le00_log).
  - *Limits:* first order only, at ψ_true. Step 5 authorises nothing.

- **D3 (read-only).** χ² intervals for the SD ratios; centred coverage; the required inflation.
  - *Interpretation:* bias dominant, if the mean |z| ≫ 1 while centred coverage is adequate; dispersion understated, if the SD ratios are reliably > 1. The latter cannot be shown with n = 5.
  - *Limits:* n = 5.

- **D4 [N]: item-uncertainty-corrected SEs.**
  - **Computation:**
    - the patient-clustered sandwich covariance of the stage-1 composite item estimates, V₁. This needs per-patient score contributions of `stage1_negloglik` (new code; per-visit scores summed by patient);
    - the cross-derivative C = ∂²ℓ₂/∂ψ∂φ at ψ̂⁽²⁾, by central FD of the analytic stage-2 gradient over the 55 item parameters. That is 110 likelihood-plus-gradient evaluations, about 1.4 s each;
    - V = H⁻¹ + H⁻¹ C V₁ Cᵀ H⁻¹.
  - For S1-b items, the item step depends on ψ̂⁽¹⁾. A full correction would need the derivative of the S1-b map. **Proposed scope:** the correction for the S1-b item covariance at fixed ψ̂⁽¹⁾ only, with that limitation stated.
  - **Cost:** about 3 min per dataset for C, plus about 1–2 min for V₁: **about 0.3 node-hours** with the smoke check. No new random numbers.
  - **Interpretation:**
    - corrected SEs ≥ about 1.4× would show the failure is partly an uncertainty-quantification problem;
    - corrected SEs ≈ the current ones would show the item uncertainty is small relative to the bias, so the bias is the issue (with H-A or H-B).
  - **Risk:** the two-step correction is asymptotic. It ignores the S1-b dependence, and it says nothing about bias. It must **not** be used to re-score Step 6 coverage (the frozen criterion uses the FD-Hessian Wald interval). Reported descriptively only.

- **D5 [N]: further S1-b iterations, as a diagnostic, not a method.**
  - **Computation:** from ψ̂⁽²⁾ and the S1-b items, apply further S1-b steps k = 3, 4, 5. Each step is: refit items under N(M, K) at the current ψ̂, then refit stage 2 from the default ψ with cold modes, exactly the step (3)–(4) procedure. Record ψ̂⁽ᵏ⁾, the items, c and d, and the fitted mixture moments. Compute an FD Hessian at the last iterate only.
  - **Cost:** about 4.5 min per iteration per dataset: 3 × 5 × 4.5 ≈ 70 min, plus Hessians about 5 min, so **about 1.3–1.5 node-hours**. No new random numbers.
  - **Interpretation:**
    - ψ̂ and c stabilise near ψ̂⁽²⁾: S1-b behaves as a fixed point of the joint model's affine invariance, so iterating does not re-anchor the scale. This supports H-A, with H-D ("one iteration insufficient") **not** supported;
    - steady movement towards the truth (c → 1, z → 0) supports H-D;
    - divergence or oscillation means the iteration is unstable.
  - **Risk:** this is **not** an approved estimator. Any result motivates at most a separately specified amendment, which would need its own recovery test on these or new datasets. Coverage of iterated estimates is **not** a Step 6 result. Iterating to convergence approaches a pseudo-joint estimator; that resemblance must not be read as evidence for S1-c.

- **D6 [N]: stage 2 with the oracle items (full refit).**
  - **Computation:** per dataset, the Step 4 procedure (Laplace Q = 15, default ψ, FD Hessian) with Step 5's saved true-latent items in place of the true items.
  - **Cost:** about 5 × (3.5 min fit + 1 min Hessian) ≈ 25 min, so **about 0.4 node-hours**. No new random numbers.
  - **Interpretation:**
    - near-Step-4 recovery of β0_P and Le11_log would show that putting the items on the correct latent scale is sufficient (supports H-A as sufficient);
    - persisting bias would point to something beyond the latent-scale error (H-B, or item–ψ interaction).
    - Le00_log is expected (Step 5, first order) to keep part of its deviation through tremor item sampling error. That would separate the Le00_log mechanism from the PIGD one.
  - **Risk:** the oracle items use the unknown truth. This is a counterfactual and never a usable estimator. It is a full refit, but still 5 datasets.

**Priority.**
1. D1–D3 (read-only, minutes; no approval of compute needed beyond running a read-only script).
2. Then **D6** (the smallest new computation, and the most discriminating for H-A as sufficient).
3. Then D4, if the uncertainty question remains open after D3.
4. Then D5, if D1 and D6 support H-A and the "one iteration" question matters for a future amendment.

Total new computation, if all are approved: **about 2.2 node-hours** on node002.

**No new synthetic datasets or seeds are proposed.** All new computation reuses the frozen datasets `[42, 45, 1, r]` and saved items.

## 4. Candidate explanations and current status

| id | explanation | current evidence | status |
|---|---|---|---|
| **H-A** | **Latent-scale anchoring.** The N(0, 1) stage 1 places the items on a compressed or shifted PIGD scale (c > 1, d < 0) relative to the scale on which ψ_true is defined. Stage 2 then estimates ψ correctly *on the item-implied scale*. One S1-b refit is anchored to ψ̂⁽¹⁾'s fitted mixture (variance 0.85–0.93), which inherits that scale | §2.2–2.4, §2.6 [E]: present at ψ̂⁽¹⁾; the affine relabelling explains most PIGD threshold error and the mean β0_P / Le11_log bias; oracle items have c ≈ 1; the first-order attribution is to the N(0, 1) assumption | **supported descriptively for β0_P and Le11_log; not established** (D1, D6, D5) |
| H-B | Non-affine, item-specific estimation error, or item–ψ interaction beyond relabelling | residual τ error 0.03–0.08 after the affine fit; the affine prediction overshoots with N(0, 1) items | **open** (D1, D6) |
| H-C | Wald intervals too narrow because the items are held fixed | SEs = fixed-item SEs; bias of 1.7–2.4 SE; centred coverage 4–5/5; inflation of 1.35–1.53× needed | **not supported as the main mechanism, not excluded** (D3, D4) |
| H-D | One S1-b iteration is insufficient | S1-b moved the PIGD a error 0.221 → 0.174 but ψ̂ barely, and not towards the truth on average | **open** (D5). Note the competing reading (fixed point under affine invariance) |
| H-E | Within-stage-2 non-identifiability or compensation among ψ components | at fixed items, small estimator correlations among the failing components; Step 4 recovers them with true items | **not supported by existing outputs** (cannot show item–ψ coupling; D4's C matrix would) |
| H-F | Le00_log has a partly separate mechanism (tremor item sampling error) | Step 5 split: −0.76 sampling vs −0.50 assumption; the affine prediction under-explains it (−0.67 vs −1.71); rank with tremor c ρ −0.8 | **open** (D2, D6) |

## 5. What this investigation can and cannot establish

**It can:** characterise, on the 5 frozen datasets, whether the Step 6 failure is mainly an item-scale (anchoring) bias, an item-sampling effect, or an uncertainty-quantification shortfall. That informs whether any future amendment is worth proposing.

**It cannot:**
- change the Step 6 verdict;
- validate any repair (any repair needs a separate amendment, with its own frozen recovery test);
- say anything about real PPMI data, biological subtypes or D1 (D1 remains **NOT REPRODUCED**).

**Risks:**
- Over-reading n = 5 rank statistics.
- Treating oracle-based counterfactuals (D6) or iterated S1-b (D5) as estimators.
- Re-scoring Step 6 with corrected SEs (D4). This is explicitly excluded.

## 6. Proposed execution (on approval only)

1. D1–D3: a new read-only script, `scripts/53_m2_step6_failure_readonly.py`. It reads the frozen and saved outputs (hash-verified) and writes to a new `experiments/<UTC>_stage2_step6_diagnostics/readonly/`, with a report section. **Stop for review.**
2. Only on separate approval: D6, then D4 / D5 as approved. Each is a new driver with unit tests and a smoke check, run on node002, with its own run directory and integrity checks, no new seeds, and a stop after each.

**Not authorised by this document:** any computation; code; rerunning or altering Step 6; Steps 7 and 8; calibration, positive controls, gates or real-data evaluation; S1-c or any repair. Nothing is staged, committed or pushed.
