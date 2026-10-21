# Stage 2 M2: why full-pipeline recovery (Step 6) failed. Diagnostic investigation report (D1–D6)

**Date:** 2026-10-19.

> **PROPOSED — NOT FROZEN — EXPLORATORY DIAGNOSTIC REPORT.**
> - **Step 6 remains FAILED** under its frozen, unchanged criterion (doc 38). **D1 remains NOT REPRODUCED.**
> - Nothing here re-scores Step 6, replaces its intervals, defines or accepts a repair, or supports any conclusion about real PPMI data.
> - Every diagnostic coverage count below is labelled as such and is **not** a recovery test.

**Proposal followed:** doc 39 (`39_STAGE2_STEP6_FAILURE_DIAGNOSTIC_PLAN_PROPOSED_20261019.md`), as authorised by the user's instruction "Complete Step 6 Failure Investigation — Execute End-to-End".

**Execution directory:** `experiments/20261019T084559Z_stage2_step6_diagnostics/`.

**Tags:**
- **[O]** an observation computed from saved or new diagnostic outputs;
- **[M]** a mathematical statement (with the numerical check that supports it);
- **[I]** an interpretation;
- **[H]** an untested hypothesis.

---

## 1. Executive summary

1. **[O] The failure is almost entirely a systematic location error carried by the estimated items. Stage 2 does not malfunction.**
   - With the true items (Step 4) or with items on the correct latent scale (D6, Step 5's true-latent items), stage 2 recovers all three failing parameters.
     - β0_P mean z: −0.08 (Step 4), −0.03 (D6).
     - Le11_log mean z: −0.29, −0.27.
   - With the estimated items it does not (Step 6: +2.44 and −2.11).
   - The item-to-ψ sensitivity at the Step 6 estimates (D4: J₂ = −H⁻¹C) reproduces the observed ψ̂⁽²⁾ − ψ̂_Step4 to within 0.10 SE for **all 19 components**.
2. **[O] β0_P and Le11_log share one cause: the PIGD items are on a shifted, stretched latent scale.**
   - The PIGD item errors are mainly a common affine relabelling θ′ = (θ − d)/c. For the S1-b items, c_P = 1.00–1.14 and d_P = −0.12 to +0.05. Aligning removes most of the threshold error (τ RMSE 0.06–0.16 → 0.03–0.04).
   - That relabelling, applied to the truth, explains 76 % (β0_P) and 94 % (Le11_log) of the dataset-level squared z.
   - In D4's first-order attribution the PIGD-item part is +2.32 SE (β0_P) and −1.79 SE (Le11_log). The affine part alone is +2.47 and −1.97; the non-affine remainder is small (−0.15, +0.17).
3. **[O] Le00_log has a different, mixed cause: the tremor items.**
   - It is driven entirely by the tremor items (−1.49 SE), split roughly evenly into affine (−0.77) and non-affine (−0.72) parts.
   - It does not fully return with true-latent items (D6: mean z −0.98, 4/5).
   - It is the parameter whose uncertainty is most understated by holding the items fixed (D4: full-chain SE ×1.65).
4. **[O] Most of the error is present after the first stage-2 fit.** ψ̂⁽¹⁾ already carries 90 %, 86 % and 94 % of the final mean z of β0_P, Le00_log and Le11_log. S1-b changed them by +0.19, −0.23 and −0.12 SE on average.
5. **[M + O] Repeated S1-b updates do not converge and do not approach the truth. They drift along an exactly unidentified direction.**
   - The joint model is invariant under latent relabelling: a unit test gives relative error ≤ 1e-9, and the D4 identity C·v_φ + H·v_ψ = 0 holds to ≤ 6e-7.
   - So the S1-b map has no anchor for the scale.
   - In D5 (four further updates), every step moved the estimate by an almost constant 0.34–0.94 SE. The stage-2 objective was flat to ±0.03, and the item objectives to ±0.06. **99.5–99.9 %** of the movement is a pure scale/location relabelling.
6. **[O] Holding items fixed understates uncertainty materially for the residual-SD parameters, not for β0_P.** The stacked-estimating-equation (full-chain) sandwich SE ratios are ×1.48 (Le11_log), ×1.65 (Le00_log), ×1.20 (L22_log) and ×1.11 (β0_P).
   - Under those SEs (diagnostic only), Le00_log would be covered 5/5 and Le11_log 4/5. β0_P stays at 3/5 (mean z +2.19), because its error is location bias, not missing variance.
7. **[I] Conclusion.**
   - **What failed:** the two-stage-with-S1-b procedure determines the latent scale only through its N(0, 1) starting anchor. That anchor does not coincide with the convention on which the generating truth is defined: pooled visit-level latent mean 0 and variance 1 (`truth_on_stage1_scale`).
   - **Consequence:** the PIGD parameters are estimated on a slightly relabelled scale. The Wald intervals additionally omit item uncertainty, which matters for the residual SDs.
   - **What did not fail:** stage-2 estimation itself, and the numerical approximation (Steps 3–4).
   - **Status:** this is the best-supported explanation on these five datasets. It is **not** established in general (§10).
8. **Recommended next decision** (§12): whether to authorise a seconds-long, read-only diagnostic **D7**.
   - D7 re-expresses the saved estimates on the truth's own scale convention, computed from each estimate's fitted latent distribution and never from the truth.
   - It checks the sharp prediction that this collapses the D5 drift and removes most of the β0_P / Le11_log bias.
   - Any change of estimator or interval method would then need a dated amendment and a new frozen recovery test.

## 2. The original failure (unchanged)

**Criterion (doc 16 D3; Rev 2 §0, §5, §6 step 6; plan doc 37 §3):** for each of the 19 ψ components, the 95 % Wald interval ψ̂⁽²⁾ ± 1.96·SE (central-FD Hessian, h = 1e-4, items fixed) must cover the truth in ≥ 4 of 5 datasets (seeds `[42, 45, 1, r]`).

**Step 6 (doc 38):** outcome **COMPLETE, no flagged event; formal verdict FAIL.**
- β0_P 3/5 (mean z +2.44);
- Le00_log 3/5 (−1.71);
- Le11_log 2/5 (−2.11);
- all other components ≥ 4/5.

Step 4 (fixed true items) met the same rule for all 19 components: 14 at 5/5 and 5 at 4/5.

**These facts are unchanged by this investigation.**

## 3. Data, files and outputs used

**Datasets.** The five frozen synthetic recovery datasets only: seeds `[42, 45, 1, r]`, r = 0…4, the a-priori truth on the stage-1 scale, the real visit design. They were regenerated exactly and verified:
- the dataset-0 identity NLL is 57022.74225348176, exactly;
- for all five datasets, Step 5's recorded N(0, 1) and true-latent item-objective values are reproduced **exactly**, in every compute job.

**No new data, seeds or scenarios.** No real responses or labels.

**Saved inputs (read-only, hash-verified against the baseline of 195 prior output files):**
- pilot `step1_recovery.json`;
- Step 4 `step4_run2/S1_dataset{r}.json`, `summary.json`;
- Step 5 `step5/S1_items_dataset{r}.json` (N(0, 1), true-latent and generating items), `S4_downstream_shift.csv`, `summary.json`;
- Step 6 `step6/S1_dataset{r}.json` (ψ̂⁽¹⁾, ψ̂⁽²⁾, H⁽¹⁾, H⁽²⁾, both item sets), `summary.json`, `MANIFEST.sha256` (verified).

## 4. Phase A — read-only diagnostics D1–D3

**Script:** `scripts/53_m2_step6_diag_readonly.py` (`021fd53c…`). Outputs are in `readonly/`. No model was fitted.

### 4.1 D1 — affine latent-scale alignment

**Construction.**
- A relabelling θ′_s = (θ_s − d_s)/c_s of subscore s maps the true items to (c·a, (τ − d)/c) and ψ to ψ′ = `relabel_psi(ψ, c, d)`:
  - β0′ = (β0 − d)/c;
  - slope and treatment terms divided by c;
  - D and Σ_ε rescaled by diag(1/c).
- The marginal likelihood is **exactly invariant** under this joint map [M]: tested to relative error 1e-9 (`tests/test_step6_diagnostics.py`).
- (c, d) per subscore and dataset are fitted by unweighted least squares to log â − log a and τ̂ − (τ − d)/c ("joint"). Threshold-only and discrimination-only variants are reported as well.
- **Predicted error:** ψ′(c, d) − ψ_true, in the SE of the fit that used those items.
- **Explained share:** 1 − Σ_r(obs z − pred z)² / Σ_r obs z². It is descriptive and can be negative.

**Affine fits [O]:**

| item set | subscore | c (thresholds) | c (discriminations) | c (joint) | d (joint) | τ RMSE raw → after | log a RMSE raw → after | mean a error |
|---|---|---|---|---|---|---|---|---|
| N01 | T | 0.989–1.089 | 1.004–1.073 | 0.992–1.086 | -0.024 to +0.050 | 0.054–0.198 → 0.046–0.075 | 0.028–0.075 → 0.029–0.044 | +0.056 |
| N01 | P | 1.061–1.180 | 1.029–1.115 | 1.053–1.161 | -0.134 to +0.000 | 0.068–0.181 → 0.025–0.078 | 0.045–0.139 → 0.041–0.095 | +0.221 |
| S1b | T | 0.968–1.098 | 0.994–1.093 | 0.972–1.097 | -0.035 to +0.066 | 0.054–0.240 → 0.046–0.084 | 0.036–0.092 → 0.025–0.042 | +0.058 |
| S1b | P | 1.003–1.144 | 1.007–1.114 | 1.004–1.136 | -0.118 to +0.052 | 0.057–0.162 → 0.029–0.044 | 0.035–0.116 → 0.027–0.056 | +0.174 |
| oracle | T | 0.962–1.060 | 0.986–1.055 | 0.966–1.059 | -0.003 to +0.057 | 0.052–0.162 → 0.047–0.083 | 0.034–0.059 → 0.025–0.041 | +0.024 |
| oracle | P | 0.961–1.066 | 0.968–1.041 | 0.962–1.060 | -0.042 to +0.073 | 0.058–0.085 → 0.028–0.046 | 0.035–0.055 → 0.027–0.052 | +0.007 |

| ψ component | S1-b items → ψ̂⁽²⁾: mean obs z | mean pred z | explained share | N(0,1) items → ψ̂⁽¹⁾: mean obs z | mean pred z | explained share |
|---|---|---|---|---|---|---|
| beta0_T | -0.31 | -0.90 | +0.75 | -0.18 | -0.42 | +0.75 |
| beta1_T | +0.02 | -0.12 | -0.15 | +0.03 | -0.16 | -0.22 |
| gamma_T | +0.32 | -0.06 | +0.13 | +0.36 | -0.07 | +0.07 |
| beta0_P | +2.44 | +2.67 | +0.76 | +2.19 | +4.10 | +0.10 |
| beta1_P | -0.52 | -1.07 | -1.91 | -0.60 | -1.61 | -2.35 |
| gamma_P | -0.90 | -0.34 | +0.40 | -0.58 | -0.51 | +0.39 |
| L00_log | -0.72 | -0.56 | -0.57 | -0.44 | -0.72 | -1.14 |
| L10 | +0.29 | -0.02 | +0.02 | +0.24 | -0.03 | +0.01 |
| L11_log | -0.55 | -0.22 | -0.04 | -0.50 | -0.27 | +0.03 |
| L20 | +0.33 | -0.14 | -0.23 | +0.39 | -0.20 | -0.37 |
| L21 | -0.14 | +0.01 | +0.00 | -0.16 | +0.01 | +0.00 |
| L22_log | -1.41 | -1.63 | +0.35 | -0.70 | -2.50 | -4.35 |
| L30 | -0.17 | +0.00 | +0.00 | -0.21 | +0.00 | +0.00 |
| L31 | -0.43 | +0.00 | +0.00 | -0.43 | +0.00 | +0.00 |
| L32 | -0.61 | -0.08 | +0.12 | -0.88 | -0.12 | +0.19 |
| L33_log | -0.88 | -0.91 | +0.88 | -1.23 | -1.37 | +0.87 |
| Le00_log | -1.71 | -0.76 | +0.58 | -1.47 | -0.99 | +0.57 |
| Le10 | +0.17 | -0.17 | +0.07 | +0.18 | -0.25 | -0.08 |
| Le11_log | -2.11 | -1.95 | +0.94 | -1.98 | -2.97 | +0.73 |

**D1 reading.**
- **[O] PIGD (P) items.** The N(0, 1) items are systematically stretched (c_P 1.05–1.16) and shifted (d_P ≤ 0). S1-b reduced the stretch only partly (1.00–1.14). The Step 5 true-latent items scatter around c ≈ 1, d ≈ 0 (0.96–1.06): that spread is the sampling variability of c.
- **[O] Tremor (T) items.** Smaller and less systematic (c_T 0.97–1.10 for both N(0, 1) and S1-b).
- **[O] After alignment.** The residual threshold RMSE is 0.03–0.08, comparable to the true-latent items' own residual (0.03–0.08). So the **excess** PIGD item error over sampling noise is mostly affine.
- **[O] S1-b items → ψ̂⁽²⁾.** The relabelling explains β0_P (0.76), Le11_log (0.94), L33_log (0.88), β0_T (0.75) and part of Le00_log (0.58: predicted −0.76 vs observed −1.71). It does not explain β1_P, L00_log, L11_log or γ_P.
- **[O] N(0, 1) items → ψ̂⁽¹⁾.** It **overshoots** (β0_P predicted +4.10 vs +2.19 observed; L22_log −2.50 vs −0.70). The N(0, 1) item error is less purely affine than the S1-b item error.
- **[I] Limits.** The alignment is descriptive. One (c, d) per subscore is a constraint, not an estimate with an interval, and it does not prove uniqueness of cause.

### 4.2 D2 — error decomposition

Columns:
- mean z per stage;
- the item effect ψ̂⁽¹⁾ − ψ̂_Step4 (Step 4 SE units) against Step 5's first-order prediction from ψ_true, with the prediction split into the N(0, 1)-assumption part (N(0, 1) items − true-latent items) and the item-sampling part (true-latent − true items);
- the S1-b effect ψ̂⁽²⁾ − ψ̂⁽¹⁾ (Step 6 SE units).

| ψ component | mean z Step 4 | pilot | ψ̂⁽¹⁾ | ψ̂⁽²⁾ | items effect ψ̂⁽¹⁾−Step 4 (SE₄): actual | Step 5 first-order | max \|resid\| | of which N(0,1) assumption | of which item sampling | S1-b effect ψ̂⁽²⁾−ψ̂⁽¹⁾ (SE₂): mean [min, max] |
|---|---|---|---|---|---|---|---|---|---|---|
| beta0_T | +0.00 | -0.13 | -0.18 | -0.31 | -0.18 | -0.16 | 0.11 | +0.27 | -0.43 | -0.13 [-0.69, +0.61] |
| beta1_T | +0.24 | +0.02 | +0.03 | +0.02 | -0.22 | -0.18 | 0.24 | -0.09 | -0.09 | -0.01 [-0.12, +0.10] |
| gamma_T | +0.44 | +0.38 | +0.36 | +0.32 | -0.06 | -0.09 | 0.12 | -0.02 | -0.07 | -0.04 [-0.08, +0.01] |
| beta0_P | -0.08 | +2.35 | +2.19 | +2.44 | +2.19 | +2.22 | 0.20 | +2.16 | +0.06 | +0.19 [-0.32, +0.87] |
| beta1_P | +0.51 | -0.64 | -0.60 | -0.52 | -1.07 | -1.02 | 0.21 | -1.02 | -0.00 | +0.07 [-0.07, +0.29] |
| gamma_P | -0.64 | -0.57 | -0.58 | -0.90 | +0.11 | +0.01 | 0.37 | -0.01 | +0.01 | -0.31 [-0.38, -0.18] |
| L00_log | +0.37 | -0.52 | -0.44 | -0.72 | -0.81 | -0.76 | 0.09 | -0.24 | -0.52 | -0.28 [-0.75, +0.13] |
| L10 | +0.30 | +0.22 | +0.24 | +0.29 | -0.07 | -0.02 | 0.12 | -0.06 | +0.03 | +0.06 [+0.03, +0.09] |
| L11_log | -0.21 | -0.47 | -0.50 | -0.55 | -0.29 | -0.30 | 0.03 | -0.14 | -0.16 | -0.05 [-0.20, +0.10] |
| L20 | +0.46 | +0.19 | +0.39 | +0.33 | -0.09 | -0.08 | 0.18 | -0.08 | +0.00 | -0.06 [-0.11, -0.02] |
| L21 | -0.14 | -0.11 | -0.16 | -0.14 | +0.00 | -0.07 | 0.19 | -0.04 | -0.02 | +0.02 [-0.06, +0.11] |
| L22_log | +0.01 | -0.85 | -0.70 | -1.41 | -0.70 | -0.54 | 0.35 | -0.59 | +0.06 | -0.72 [-1.05, -0.21] |
| L30 | -0.17 | -0.16 | -0.21 | -0.17 | -0.02 | -0.02 | 0.09 | -0.02 | -0.00 | +0.04 [+0.02, +0.08] |
| L31 | -0.41 | -0.44 | -0.43 | -0.43 | +0.01 | -0.01 | 0.13 | -0.01 | +0.00 | -0.01 [-0.09, +0.07] |
| L32 | -0.47 | -0.92 | -0.88 | -0.61 | -0.35 | -0.51 | 0.27 | -0.47 | -0.03 | +0.25 [+0.14, +0.30] |
| L33_log | +0.03 | -1.21 | -1.23 | -0.88 | -1.29 | -1.08 | 0.37 | -1.08 | +0.00 | +0.37 [+0.24, +0.55] |
| Le00_log | -0.23 | -2.26 | -1.47 | -1.71 | -1.25 | -1.26 | 0.09 | -0.50 | -0.76 | -0.23 [-0.89, +0.38] |
| Le10 | +0.31 | +0.24 | +0.18 | +0.17 | -0.13 | -0.12 | 0.15 | -0.12 | +0.00 | -0.01 [-0.05, +0.02] |
| Le11_log | -0.29 | -3.34 | -1.98 | -2.11 | -1.70 | -1.77 | 0.18 | -1.78 | +0.01 | -0.12 [-0.44, +0.34] |

**D2 reading.**
- **[O]** Step 5's first-order prediction matches the actual effect of the N(0, 1) items to within **0.37 SE for every component and dataset**. For the failed components the match is within 0.20 SE: β0_P +2.22 predicted vs +2.19 actual; Le11_log −1.77 vs −1.70; Le00_log −1.26 vs −1.25.
- **[O]** By that first-order split:
  - β0_P is +2.16 from the N(0, 1) assumption and +0.06 from item sampling;
  - Le11_log is −1.78 and +0.01;
  - **Le00_log is −0.50 and −0.76.**
- **[O]** The S1-b step moved the failed components by +0.19 [−0.32, +0.87], −0.23 [−0.89, +0.38] and −0.12 [−0.44, +0.34] SE. Its largest systematic effect is on L22_log: −0.72 SE on average, negative in every dataset (−1.05 to −0.21).
- **[I]** S1-b is not what created the failure. It also does not remove it.
- **[O]** The pilot (old likelihood, N(0, 1) items) shows that the Rev 2 likelihood reduced the Le00_log and Le11_log errors (−2.26 → −1.47; −3.34 → −1.98) but not the β0_P error.
- **Limits:** first order; the split uses ψ_true and true-latent items, both unknowable in practice.

### 4.3 D3 — bias, spread and intervals (all 19 components; full tables in Appendix A)

**Failed components** (Step 4 / ψ̂⁽¹⁾ / ψ̂⁽²⁾):

| ψ component | stage | covered (CP 95 %) | mean err | mean SE | mean z | RMS z | bias t | SD/mean SE (χ² 95 %) | centred-coverage diagnostic | SE inflation for ≥ 4/5 |
|---|---|---|---|---|---|---|---|---|---|---|
| beta0_P | step4 | 5/5 (0.48–1.00) | -0.0025 | 0.0305 | -0.08 | 0.65 | -0.25 | 0.72 (0.43–2.07) | 5/5 | 0.43 |
| beta0_P | psi1 | 3/5 (0.15–0.95) | +0.0646 | 0.0295 | +2.19 | 2.32 | +5.68 | 0.86 (0.52–2.48) | 5/5 | 1.35 |
| beta0_P | psi2 | 3/5 (0.15–0.95) | +0.0703 | 0.0288 | +2.44 | 2.69 | +4.38 | 1.25 (0.75–3.58) | 4/5 | 1.42 |
| Le00_log | step4 | 5/5 (0.48–1.00) | -0.0061 | 0.0242 | -0.23 | 1.07 | -0.49 | 1.16 (0.69–3.33) | 5/5 | 0.66 |
| Le00_log | psi1 | 3/5 (0.15–0.95) | -0.0362 | 0.0243 | -1.47 | 1.89 | -2.49 | 1.33 (0.80–3.84) | 4/5 | 1.07 |
| Le00_log | psi2 | 3/5 (0.15–0.95) | -0.0419 | 0.0243 | -1.71 | 2.24 | -2.41 | 1.60 (0.96–4.61) | 4/5 | 1.53 |
| Le11_log | step4 | 5/5 (0.48–1.00) | -0.0103 | 0.0345 | -0.29 | 0.45 | -1.69 | 0.39 (0.24–1.13) | 5/5 | 0.32 |
| Le11_log | psi1 | 1/5 (0.01–0.72) | -0.0688 | 0.0346 | -1.98 | 2.12 | -5.15 | 0.86 (0.52–2.48) | 5/5 | 1.18 |
| Le11_log | psi2 | 2/5 (0.05–0.85) | -0.0729 | 0.0345 | -2.11 | 2.35 | -4.02 | 1.17 (0.70–3.37) | 5/5 | 1.35 |

**Definitions:**
- **Coverage** is shown with a Clopper–Pearson 95 % interval for k/5.
- **Bias t** = mean error / (SD/√5).
- **SD/mean SE** has the χ²(4) 95 % interval: × 0.599 to × 2.874 of the point value.
- **Centred coverage** counts |z − mean z| ≤ 1.96. It is **a diagnostic of dispersion around the bias, not a valid confidence interval**, and it is not a re-scoring of Step 6.
- **SE inflation for ≥ 4/5** = (4th-smallest |z|)/1.96: the smallest common multiplier of all five SEs that would bring 4/5 inside ±1.96. A real SE correction would not be a common multiplier. It is reported only because doc 39 proposed it.

**D3 reading [O].**
- The Step 6 SEs equal the fixed-item Step 4 SEs to within a ratio of 0.891–1.017, for all 19 components and datasets.
- For the failed components the bias is large relative to the spread: bias t +4.38 (β0_P), −2.41 (Le00_log), −4.03 (Le11_log). Centred coverage would be 4/5, 4/5 and 5/5.
- With n = 5 the SD ratios (1.25, 1.60, 1.17) have 95 % intervals spanning roughly 0.7–4.6. **They cannot show whether the dispersion is understated.** D4 addresses that directly.

### 4.4 Interim findings at the end of Phase A

- The failure is predominantly a **location bias** already present after the first stage-2 fit, driven by the item estimates.
- For β0_P and Le11_log it is largely an **affine (scale-anchoring) relabelling of the PIGD scale**.
- For Le00_log it is a mixture of tremor-item effects.
- Whether stage 2 recovers ψ with correctly scaled items (D6), whether item uncertainty matters (D4), and whether further S1-b updates help (D5) needed computation.

## 5. Phase B — targeted diagnostic computations

**Driver:** `scripts/54_m2_step6_diag_compute.py` (`fc6575f4…`), modes `d6`, `d4`, `d5`.

**Tests:** `tests/test_step6_diagnostics.py` (10 tests, all pass):
- exact affine-fit recovery;
- likelihood invariance under relabelling (relative error 1e-9);
- stacked-sandwich algebra against the explicit two-step formula;
- per-patient score additivity;
- the settle rule and Aitken formula;
- packing round trip;
- the relabelling tangent satisfies the first-order invariance identity, in 4 directions.

**Runs.** Smoke code-path runs (60 patients; *not results*) preceded the full runs: jobs 13933–13936; outputs `*_smoke*/`. The D4 smoke was rerun (`d4_smoke2/`) after the tangent code was moved into a tested function.

**Configuration** (node002, Xeon Gold 6326, 1 exclusive node, 32 workers, single-threaded BLAS):
- Laplace over b with per-visit 15 × 15 GH (Q = 15);
- P1/P2 active;
- G = 5 AGQ (exact gradient) for non-ND patients in reported contexts;
- L-BFGS-B options and bounds unchanged;
- default ψ, cold modes;
- FD Hessian h = 1e-4;
- item fits: 61-node GH, from `initial_items`, maxiter 3000.

**Every compute job first verified:** frozen locks; code against the baseline; the unchanged read-only script; Step 5/6 outputs against the baseline; the panel MD5; truth equal to the Step 6 record; dataset-0 identity; exact per-dataset alignment.

### 5.1 D6 — stage 2 with Step 5's saved true-latent items (job 13938)

**Question.** If the items were on the correct latent scale, would stage 2 recover ψ?

**Items.** Step 5's saved true-latent items: fitted to the same data under θ_v ~ N(M_ij, K_ij,jj) at ψ_true. Their provenance is verified exactly against Step 5's recorded objective values. The settings are the Step 4/6 settings.

**This is a counterfactual:** the items use the unknown truth. It is not an estimator and not a recovery test.

| dataset | converged | iterations | ‖∇‖ | bound hits | penalty evals | fallback / budget at optimum | H min eig | covered (diagnostic) | z β0_P | z Le00_log | z Le11_log | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | True | 78 | 0.0004 | none | 2 | 0 / 0 | 125.9 | 19/19 | +0.27 | -1.43 | -0.24 | 170 |
| 1 | True | 84 | 0.0006 | none | 2 | 0 / 0 | 158.0 | 19/19 | -0.46 | -0.76 | +0.04 | 185 |
| 2 | True | 102 | 0.0005 | none | 4 | 0 / 0 | 110.9 | 18/19 | +1.19 | -1.51 | -1.50 | 249 |
| 3 | True | 96 | 0.0005 | none | 2 | 0 / 0 | 58.2 | 18/19 | -0.72 | +1.20 | +0.87 | 193 |
| 4 | True | 78 | 0.0009 | none | 4 | 0 / 0 | 138.3 | 16/19 | -0.42 | -2.43 | -0.52 | 206 |

**[O]** Diagnostic counts: **every component ≥ 4/5** (14 at 5/5), the same pattern as Step 4.
- β0_P: mean z **−0.03** (Step 6: +2.44).
- Le11_log: **−0.27** (−2.11).
- Le00_log: **−0.98** (−1.71), 4/5.
- All other components change by small amounts (consolidated table, §6).

**[I]** Putting the items on the correct latent scale is **sufficient** to remove the β0_P and Le11_log failures on these datasets. It is not sufficient to remove all of the Le00_log error.

### 5.2 D4 — item uncertainty and item–stage-2 coupling (job 13939)

**What the saved outputs could not provide.**
- The saved H⁽¹⁾ and H⁽²⁾ are fixed-item Hessians. They give only within-stage-2 correlations, and those are small among the failed components (|r| ≤ 0.13).
- No item covariance, cross-derivative or per-patient score was saved.
- New computation was therefore required.

**Method.** The whole Step 6 chain is written as stacked estimating equations in θ = (φ₁, ψ₁, φ₂, ψ₂):
- φ₁: the 80 N(0, 1) item parameters;
- ψ₁: the stage-2 fit;
- φ₂: the 80 S1-b item parameters, which depend on ψ₁;
- ψ₂: the binding fit.

All functions are gradients of the negative log-likelihood objectives actually used. The Jacobian is block-lower-triangular:
- A₁ = ∇²_φ₁ NLL₁;
- C₁ = ∂g₂/∂φ₁;
- H₁ (saved);
- E₃ = ∂g₃/∂ψ₁;
- A₃ = ∇²_φ₂ NLL₃;
- C₂ = ∂g₄/∂φ₂;
- H₂ (saved).

All are central finite differences (h = 1e-4) of analytic gradients. The stage-2 gradients use the reported context (non-ND patients by G = 5 AGQ), warm-started from the converged modes.

**Covariance.** V = J⁻¹(Σ_i u_i u_iᵀ)J⁻ᵀ, where u_i is patient i's stacked score: patients are independent; the visits within a patient are not. Sub-systems give:
- the stage-2 sandwich alone;
- the two-step correction for the S1-b items at fixed ψ̂⁽¹⁾;
- the **full chain**.

**First-order attribution.** J₂ = −H₂⁻¹C₂ applied to the item errors (φ₂ − φ_true), split into tremor and PIGD items and into affine and non-affine parts (the D1 joint fit).

| ψ component | SE ratio: stage-2 sandwich | two-step (S1-b items, ψ̂⁽¹⁾ fixed) | full chain (range) | diag. cov frozen / s2-sandwich / two-step / full chain | mean z frozen → full chain |
|---|---|---|---|---|---|
| beta0_T | 1.00 | 0.72 | 1.14 (1.12–1.15) | 4 / 4 / 3 / 4 | -0.31 → -0.27 |
| beta1_T | 1.01 | 1.02 | 1.02 (1.00–1.03) | 5 / 5 / 5 / 5 | +0.02 → +0.02 |
| gamma_T | 1.01 | 1.01 | 1.02 (0.99–1.04) | 5 / 5 / 5 / 5 | +0.32 → +0.31 |
| beta0_P | 1.00 | 0.75 | 1.11 (1.06–1.15) | 3 / 3 / 1 / 3 | +2.44 → +2.19 |
| beta1_P | 1.02 | 1.04 | 1.08 (1.03–1.15) | 5 / 5 / 5 / 5 | -0.52 → -0.49 |
| gamma_P | 1.01 | 1.02 | 1.03 (1.01–1.06) | 4 / 4 / 4 / 4 | -0.90 → -0.88 |
| L00_log | 0.99 | 0.79 | 1.07 (0.93–1.21) | 5 / 5 / 3 / 5 | -0.72 → -0.64 |
| L10 | 1.02 | 1.01 | 1.01 (0.94–1.07) | 4 / 4 / 4 / 4 | +0.29 → +0.28 |
| L11_log | 0.99 | 0.99 | 1.02 (0.95–1.06) | 5 / 5 / 5 / 5 | -0.55 → -0.56 |
| L20 | 1.00 | 1.00 | 1.00 (0.92–1.06) | 5 / 5 / 5 / 5 | +0.33 → +0.33 |
| L21 | 1.00 | 1.00 | 1.00 (0.93–1.08) | 4 / 4 / 4 / 4 | -0.14 → -0.14 |
| L22_log | 1.01 | 0.91 | 1.20 (1.11–1.36) | 4 / 4 / 4 / 4 | -1.41 → -1.19 |
| L30 | 0.97 | 0.97 | 0.97 (0.91–1.01) | 5 / 5 / 5 / 5 | -0.17 → -0.17 |
| L31 | 1.01 | 1.01 | 1.01 (0.98–1.05) | 5 / 5 / 5 / 5 | -0.43 → -0.43 |
| L32 | 1.03 | 1.02 | 1.03 (0.97–1.07) | 5 / 5 / 5 / 5 | -0.61 → -0.60 |
| L33_log | 1.05 | 0.99 | 1.07 (1.03–1.10) | 5 / 5 / 5 / 5 | -0.88 → -0.81 |
| Le00_log | 1.00 | 1.33 | 1.65 (1.41–2.03) | 3 / 3 / 4 / 5 | -1.71 → -1.02 |
| Le10 | 1.01 | 1.01 | 1.01 (0.97–1.06) | 5 / 5 / 5 / 5 | +0.17 → +0.16 |
| Le11_log | 1.00 | 1.19 | 1.48 (1.33–1.59) | 2 / 2 / 3 / 4 | -2.11 → -1.39 |

| ψ component | observed ψ̂⁽²⁾ − ψ̂_Step4 (SE₄) | first-order total J₂Δφ | from tremor items | from PIGD items | affine part | non-affine part | max \|obs − pred\| |
|---|---|---|---|---|---|---|---|
| beta0_T | -0.30 | -0.31 | -0.31 | +0.00 | -0.89 | +0.58 | 0.04 |
| beta1_T | -0.23 | -0.22 | -0.22 | -0.00 | -0.13 | -0.09 | 0.02 |
| gamma_T | -0.10 | -0.10 | -0.10 | +0.00 | -0.04 | -0.06 | 0.00 |
| beta0_P | +2.37 | +2.32 | +0.00 | +2.32 | +2.47 | -0.15 | 0.10 |
| beta1_P | -1.00 | -0.96 | -0.00 | -0.96 | -1.01 | +0.05 | 0.09 |
| gamma_P | -0.18 | -0.18 | -0.00 | -0.18 | -0.23 | +0.05 | 0.01 |
| L00_log | -1.09 | -1.10 | -1.10 | -0.00 | -0.57 | -0.54 | 0.04 |
| L10 | -0.01 | -0.01 | -0.01 | +0.00 | -0.00 | -0.01 | 0.02 |
| L11_log | -0.34 | -0.34 | -0.34 | -0.00 | -0.22 | -0.12 | 0.01 |
| L20 | -0.15 | -0.14 | -0.01 | -0.13 | -0.16 | +0.02 | 0.01 |
| L21 | +0.02 | +0.02 | +0.00 | +0.02 | -0.01 | +0.03 | 0.01 |
| L22_log | -1.42 | -1.43 | +0.00 | -1.43 | -1.64 | +0.20 | 0.05 |
| L30 | +0.02 | +0.02 | +0.00 | +0.02 | +0.03 | -0.01 | 0.01 |
| L31 | +0.01 | +0.01 | +0.00 | +0.00 | +0.02 | -0.01 | 0.00 |
| L32 | -0.12 | -0.11 | -0.00 | -0.11 | -0.05 | -0.06 | 0.02 |
| L33_log | -0.92 | -0.90 | -0.00 | -0.90 | -0.92 | +0.02 | 0.04 |
| Le00_log | -1.48 | -1.49 | -1.49 | +0.00 | -0.77 | -0.72 | 0.03 |
| Le10 | -0.14 | -0.14 | +0.02 | -0.16 | -0.15 | +0.01 | 0.01 |
| Le11_log | -1.82 | -1.79 | -0.00 | -1.79 | -1.97 | +0.17 | 0.08 |

D4 numerical checks per dataset:

| dataset | NLL(ψ̂⁽¹⁾), NLL(ψ̂⁽²⁾) − saved | ‖g‖ at ψ̂⁽¹⁾, ψ̂⁽²⁾ | Σ per-patient − total | invariance rel. residual (max of 4) | A₁, A₃ asymmetry | B/H diag | FD failures / fallback | wall s |
|---|---|---|---|---|---|---|---|---|
| 0 | -2.2e-11, -1.5e-11 | 0.0008, 0.0023 | 7.9e-14 | 6.1e-07 | 6.1e-06, 5.8e-06 | 0.78–1.19 | 0 / 0 | 210 |
| 1 | 7.3e-12, 7.3e-12 | 0.0043, 0.0004 | 1.1e-13 | 2.5e-07 | 6.2e-06, 5.8e-06 | 0.89–1.33 | 0 / 0 | 209 |
| 2 | 0.0e+00, -2.2e-11 | 0.0008, 0.0009 | 1.3e-13 | 3.1e-07 | 5.8e-06, 5.4e-06 | 0.88–1.18 | 0 / 0 | 210 |
| 3 | -1.5e-11, 0.0e+00 | 0.0005, 0.0011 | 2.2e-13 | 4.8e-07 | 5.9e-06, 5.5e-06 | 0.77–1.13 | 0 / 0 | 211 |
| 4 | 0.0e+00, 0.0e+00 | 0.0007, 0.0018 | 1.8e-13 | 3.5e-07 | 6.6e-06, 6.2e-06 | 0.83–1.24 | 0 / 0 | 210 |

**Numerical checks [O].**
- The saved stage-2 NLLs are reproduced exactly (difference 0.0); gradient norms at the optima are ≤ 0.0043.
- Per-patient stage-2 scores sum to the total exactly.
- FD Jacobian asymmetry is ≤ 7e-6.
- **The relabelling identity C₂v_φ + H₂v_ψ ≈ 0 holds to a relative residual of 1.2e-8 to 6.1e-7** in all 4 directions × 5 datasets. This independently validates C₂ and H₂, and it is the exact form of the item–ψ trade-off.
- There were no FD failures and no fallback patients.
- The stage-2 information-equality ratio diag(B)/diag(H) is 0.77–1.33.

**D4 reading.**
- **[O]** The full-chain SE exceeds the frozen fixed-item SE by ×1.48 (Le11_log), ×1.65 (Le00_log), ×1.20 (L22_log), ×1.11 (β0_P), ×1.14 (β0_T), and less elsewhere.
- **[O]** The S1-b item uncertainty at fixed ψ̂⁽¹⁾ alone gives ×1.19 and ×1.33 for Le11_log and Le00_log. For β0_P it gives ×0.75: a negative stage-1/stage-2 score covariance. That partial correction is not itself a valid SE.
- **[O]** Under the full-chain SE (diagnostic counts, NOT a re-scoring): β0_P 3/5 (mean z +2.19); Le00_log **5/5** (−1.02); Le11_log **4/5** (−1.39).
- **[O]** The first-order attribution reproduces the observed shift within 0.10 SE everywhere:
  - β0_P and Le11_log come entirely from the PIGD items, and are almost entirely affine;
  - **Le00_log comes entirely from the tremor items, half affine and half non-affine.**
- **[O]** Item coupling raises the estimator correlations among the PIGD parameters, for example Le11_log–L22_log from −0.07 (fixed items) to +0.35, and Le11_log–β1_P to +0.31.
- **Limits:**
  - asymptotic, patient-clustered sandwich, n = 934;
  - the FD Jacobians use h = 1e-4;
  - Laplace-based scores;
  - no small-sample correction;
  - **it does not and cannot correct bias**;
  - it is not a re-scoring of Step 6 (the frozen interval is the FD-Hessian Wald interval).

### 5.3 D5 — further S1-b updates from the saved Step 6 state (job 13940)

**Procedure.** Exactly the Step 6 step (3)–(4) map, applied repeatedly from the saved ψ̂⁽²⁾ and S1-b items:
- items refitted under N(M_ij, K_ij,jj) at the current ψ̂, from `initial_items`;
- stage 2 refitted from the default ψ with cold modes.

**Iterations:** k = 3…6 (K_MAX = 6, pre-declared). Settle rule, also pre-declared: max_j |Δψ_j|/SE⁽²⁾_j < 0.05. A final FD Hessian was computed at k = 6.

**Is the map well defined beyond one iteration?** It is: the same map is applied again. It is **not** S1-b, which is defined as exactly one iteration, so this is a diagnostic of the map, not a method.

| dataset | k | stage-2 NLL | max \|Δψ\|/SE₂ | RMS z (19) | z β0_P | z Le00_log | z Le11_log | c_P (joint) | d_P | c_T | anchor P var at ψ̂_(k−1) | fit conv | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 (Step 6) | 56982.339 | — | 1.25 | +2.68 | -1.47 | -2.08 | — | — | — | — | — | — |
| 0 | 2 (Step 6) | 56971.266 | — | 1.33 | +2.78 | -1.77 | -1.81 | — | — | — | — | — | — |
| 0 | 3 | 56971.241 | 0.605 | 1.41 | +2.73 | -2.09 | -1.65 | 1.018 | -0.066 | 1.000 | 0.906 | True | 175 |
| 0 | 4 | 56971.244 | 0.601 | 1.51 | +2.67 | -2.42 | -1.49 | 1.012 | -0.066 | 1.008 | 0.916 | True | 244 |
| 0 | 5 | 56971.244 | 0.596 | 1.62 | +2.61 | -2.74 | -1.33 | 1.006 | -0.067 | 1.016 | 0.927 | True | 231 |
| 0 | 6 | 56971.245 | 0.593 | 1.73 | +2.55 | -3.07 | -1.17 | 1.001 | -0.067 | 1.024 | 0.937 | True | 187 |
| 1 | 1 (Step 6) | 58711.663 | — | 1.11 | +1.79 | -1.17 | -2.01 | — | — | — | — | — | — |
| 1 | 2 (Step 6) | 58696.940 | — | 1.22 | +1.47 | -1.53 | -2.45 | — | — | — | — | — | — |
| 1 | 3 | 58696.736 | 0.464 | 1.35 | +1.01 | -1.93 | -2.90 | 1.086 | +0.075 | 1.012 | 0.826 | True | 208 |
| 1 | 4 | 58696.724 | 0.468 | 1.49 | +0.54 | -2.32 | -3.35 | 1.102 | +0.096 | 1.022 | 0.802 | True | 202 |
| 1 | 5 | 58696.725 | 0.463 | 1.65 | +0.08 | -2.72 | -3.80 | 1.119 | +0.118 | 1.032 | 0.779 | True | 199 |
| 1 | 6 | 58696.725 | 0.456 | 1.82 | -0.38 | -3.12 | -4.25 | 1.136 | +0.140 | 1.042 | 0.756 | True | 238 |
| 2 | 1 (Step 6) | 58013.962 | — | 1.45 | +3.62 | -2.11 | -2.91 | — | — | — | — | — | — |
| 2 | 2 (Step 6) | 57999.523 | — | 1.76 | +4.49 | -3.00 | -3.35 | — | — | — | — | — | — |
| 2 | 3 | 57999.219 | 0.935 | 2.08 | +5.18 | -3.93 | -3.78 | 1.151 | -0.131 | 1.122 | 0.855 | True | 212 |
| 2 | 4 | 57999.200 | 0.939 | 2.41 | +5.84 | -4.87 | -4.21 | 1.168 | -0.147 | 1.147 | 0.829 | True | 202 |
| 2 | 5 | 57999.200 | 0.940 | 2.75 | +6.49 | -5.81 | -4.65 | 1.186 | -0.164 | 1.173 | 0.804 | True | 278 |
| 2 | 6 | 57999.198 | 0.936 | 3.10 | +7.14 | -6.74 | -5.08 | 1.204 | -0.180 | 1.199 | 0.780 | True | 226 |
| 3 | 1 (Step 6) | 57976.331 | — | 1.11 | +1.50 | +0.49 | -0.63 | — | — | — | — | — | — |
| 3 | 2 (Step 6) | 57970.545 | — | 1.11 | +1.66 | +0.88 | -0.29 | — | — | — | — | — | — |
| 3 | 3 | 57970.582 | 0.460 | 1.14 | +1.70 | +1.20 | -0.01 | 0.992 | -0.064 | 0.964 | 0.947 | True | 250 |
| 3 | 4 | 57970.584 | 0.464 | 1.20 | +1.74 | +1.53 | +0.26 | 0.983 | -0.069 | 0.957 | 0.965 | True | 207 |
| 3 | 5 | 57970.585 | 0.466 | 1.28 | +1.78 | +1.85 | +0.53 | 0.974 | -0.074 | 0.950 | 0.983 | True | 245 |
| 3 | 6 | 57970.581 | 0.469 | 1.37 | +1.82 | +2.18 | +0.80 | 0.965 | -0.079 | 0.942 | 1.001 | True | 239 |
| 4 | 1 (Step 6) | 58583.980 | — | 1.45 | +1.67 | -3.13 | -2.31 | — | — | — | — | — | — |
| 4 | 2 (Step 6) | 58570.932 | — | 1.51 | +1.82 | -3.14 | -2.64 | — | — | — | — | — | — |
| 4 | 3 | 58570.612 | 0.337 | 1.59 | +1.82 | -3.20 | -2.98 | 1.124 | -0.051 | 1.036 | 0.851 | True | 203 |
| 4 | 4 | 58570.593 | 0.337 | 1.68 | +1.81 | -3.27 | -3.31 | 1.137 | -0.045 | 1.038 | 0.831 | True | 189 |
| 4 | 5 | 58570.594 | 0.336 | 1.77 | +1.81 | -3.34 | -3.65 | 1.150 | -0.039 | 1.039 | 0.812 | True | 258 |
| 4 | 6 | 58570.592 | 0.337 | 1.86 | +1.80 | -3.40 | -3.99 | 1.164 | -0.033 | 1.041 | 0.794 | True | 236 |

| dataset | settled at k | final covered (diagnostic) | final H PD | fallback / budget | Aitken-extrapolated z β0_P, Le00_log, Le11_log (descriptive) |
|---|---|---|---|---|---|
| 0 | None | 15/19 | True | 0 / 0 | +6.92, +46.78, +30.46 |
| 1 | None | 14/19 | True | 0 / 0 | -31.26, +916.05, +95.50 |
| 2 | None | 11/19 | True | 0 / 0 | +53.92, -205.44, +264.38 |
| 3 | None | 16/19 | True | 0 / 0 | -0.86, +614.26, -133.12 |
| 4 | None | 13/19 | True | 0 / 0 | +2.35, -97.22, +178.64 |

**D5 reading.**
- **[O] No dataset settled.** Each further update moved ψ̂ by an almost constant amount (dataset 0: 0.605, 0.601, 0.596, 0.593 SE; dataset 2: 0.935–0.940).
- **[O] The objectives are flat along the path.** The stage-2 NLL is flat to ±0.03 across k = 3…6, and the item objectives to ±0.06.
- **[O] The movement is a relabelling.** 99.5–99.9 % of ψ̂₆ − ψ̂₂ is explained by a pure affine relabelling (threshold residual of items₆ vs items₂ after alignment: ≤ 0.0034).
- **[O] The direction is dataset-specific** (c_P rising in datasets 1, 2 and 4; falling in 0 and 3). So the errors do **not** converge to a common biased point either.
- **[O] Diagnostic coverage at k = 6 is worse:** Le00_log 0/5, Le11_log 2/5, β0_P 3/5.
- **[O]** The Aitken extrapolations are meaningless here: the sequences are linear, so the denominator ≈ 0. They are shown only to document that.
- **[M]** The S1-b map is **equivariant** under relabelling. If (ψ, φ) is relabelled jointly, the item refit under the relabelled N(M, K) returns the relabelled items, because the latent item objective is invariant under the same map. So at population level every relabelling of the truth is a fixed point, and the scale is not pinned by the update.
- **[I]** In finite samples the map drifts slowly along this neutral direction. **One iteration (Step 6) leaves the scale where the N(0, 1) anchor put it, and further iterations move it arbitrarily.**

## 6. Consolidated table — all 19 components

- "cov" is the count of 5 with |z| ≤ 1.96.
- **Only the "ψ̂⁽²⁾ Step 6 (binding)" column is a recovery result**, and it is the frozen FAIL.
- The D6, D4 and D5 columns are diagnostic counts.

| ψ component | truth | Step 4 (true items) cov · mean z | ψ̂⁽¹⁾ N(0,1) items cov · mean z | **ψ̂⁽²⁾ Step 6 (binding) cov · mean z** | SE⁽²⁾/SE_Step4 | D6 oracle items cov · mean z | D4 full-chain SE ratio | D4 diag. cov (full-chain SE) | D5 final cov · mean z |
|---|---|---|---|---|---|---|---|---|---|
| beta0_T | -0.2066 | 5/5 · +0.00 | 4/5 · -0.18 | **4/5** · -0.31 | 0.92–1.00 | 5/5 · -0.45 | 1.14 (1.12–1.15) | 4/5 | 1/5 · -0.80 |
| beta1_T | +0.0478 | 4/5 · +0.24 | 5/5 · +0.03 | **5/5** · +0.02 | 0.92–1.00 | 4/5 · +0.14 | 1.02 (1.00–1.03) | 5/5 | 4/5 · -0.18 |
| gamma_T | +0.0957 | 5/5 · +0.44 | 5/5 · +0.36 | **5/5** · +0.32 | 0.92–1.00 | 5/5 · +0.38 | 1.02 (0.99–1.04) | 5/5 | 5/5 · +0.23 |
| **beta0_P** | -0.4768 | 5/5 · -0.08 | 3/5 · +2.19 | **3/5** · +2.44 | 0.91–0.98 | 5/5 · -0.03 | 1.11 (1.06–1.15) | 3/5 | 3/5 · +2.67 |
| beta1_P | +0.1216 | 5/5 · +0.51 | 5/5 · -0.60 | **5/5** · -0.52 | 0.90–0.99 | 5/5 · +0.49 | 1.08 (1.03–1.15) | 5/5 | 4/5 · -0.90 |
| gamma_P | +0.1621 | 4/5 · -0.64 | 4/5 · -0.58 | **4/5** · -0.90 | 0.91–0.99 | 4/5 · -0.60 | 1.03 (1.01–1.06) | 4/5 | 4/5 · -1.04 |
| L00_log | -0.2674 | 4/5 · +0.37 | 5/5 · -0.44 | **5/5** · -0.72 | 1.00–1.00 | 5/5 · -0.19 | 1.07 (0.93–1.21) | 5/5 | 4/5 · -1.55 |
| L10 | +0.0077 | 4/5 · +0.30 | 4/5 · +0.24 | **4/5** · +0.29 | 0.92–1.00 | 4/5 · +0.31 | 1.01 (0.94–1.07) | 4/5 | 4/5 · +0.26 |
| L11_log | -2.5750 | 5/5 · -0.21 | 5/5 · -0.50 | **5/5** · -0.55 | 0.99–1.00 | 5/5 · -0.36 | 1.02 (0.95–1.06) | 5/5 | 5/5 · -0.88 |
| L20 | +0.0648 | 5/5 · +0.46 | 5/5 · +0.39 | **5/5** · +0.33 | 0.91–0.98 | 5/5 · +0.46 | 1.00 (0.92–1.06) | 5/5 | 5/5 · +0.28 |
| L21 | -0.0065 | 4/5 · -0.14 | 4/5 · -0.16 | **4/5** · -0.14 | 0.90–0.98 | 4/5 · -0.14 | 1.00 (0.93–1.08) | 4/5 | 4/5 · -0.13 |
| L22_log | -0.4382 | 5/5 · +0.01 | 5/5 · -0.70 | **4/5** · -1.41 | 0.99–1.01 | 5/5 · +0.04 | 1.20 (1.11–1.36) | 4/5 | 2/5 · -1.97 |
| L30 | +0.0000 | 5/5 · -0.17 | 5/5 · -0.21 | **5/5** · -0.17 | 0.89–0.99 | 5/5 · -0.17 | 0.97 (0.91–1.01) | 5/5 | 5/5 · -0.17 |
| L31 | +0.0000 | 5/5 · -0.41 | 5/5 · -0.43 | **5/5** · -0.43 | 0.89–0.98 | 5/5 · -0.42 | 1.01 (0.98–1.05) | 5/5 | 5/5 · -0.43 |
| L32 | +0.0098 | 5/5 · -0.47 | 5/5 · -0.88 | **5/5** · -0.61 | 0.89–0.99 | 5/5 · -0.52 | 1.03 (0.97–1.07) | 5/5 | 5/5 · -0.63 |
| L33_log | -2.3353 | 5/5 · +0.03 | 5/5 · -1.23 | **5/5** · -0.88 | 0.99–1.02 | 5/5 · +0.01 | 1.07 (1.03–1.10) | 5/5 | 2/5 · -1.14 |
| **Le00_log** | -0.7374 | 5/5 · -0.23 | 3/5 · -1.47 | **3/5** · -1.71 | 1.00–1.00 | 4/5 · -0.98 | 1.65 (1.41–2.03) | 5/5 | 0/5 · -2.83 |
| Le10 | +0.0405 | 5/5 · +0.31 | 5/5 · +0.18 | **5/5** · +0.17 | 0.91–0.99 | 5/5 · +0.33 | 1.01 (0.97–1.06) | 5/5 | 5/5 · +0.11 |
| **Le11_log** | -0.9081 | 5/5 · -0.29 | 1/5 · -1.98 | **2/5** · -2.11 | 0.99–1.01 | 5/5 · -0.27 | 1.48 (1.33–1.59) | 4/5 | 2/5 · -2.74 |

## 7. The three failed components

**β0_P (PIGD intercept), 3/5, mean z +2.44.**
- Present at ψ̂⁽¹⁾ (+2.19). S1-b made it slightly worse on average (+0.19 SE).
- First order: entirely PIGD items (+2.32), affine part +2.47 — the shift d_P < 0 and stretch c_P > 1 of the PIGD scale.
- Correct-scale items remove it (D6 −0.03).
- Item uncertainty barely matters: full-chain SE ×1.11; still 3/5.
- Further S1-b updates do not remove it and can make it worse (dataset 2: +7.14 at k = 6).
- **The error is bias from scale anchoring, not a missing interval width.**

**Le11_log (log PIGD residual SD), 2/5, mean z −2.11.**
- Present at ψ̂⁽¹⁾ (−1.98).
- First order: entirely PIGD items (−1.79), affine −1.97: a stretched PIGD scale makes the residual SD look smaller.
- Correct-scale items remove it (D6 −0.27).
- Item uncertainty is material (×1.48; diagnostic 4/5 under full-chain SE, mean z still −1.39).
- **Mainly scale-anchoring bias, with interval width also understated.**

**Le00_log (log tremor residual SD), 3/5, mean z −1.71.**
- Present at ψ̂⁽¹⁾ (−1.47).
- First order: entirely tremor items (−1.49), half affine and half non-affine. Step 5 attributes more of the N(0, 1)-item effect to item sampling (−0.76) than to the N(0, 1) assumption (−0.50).
- Correct-scale items only partly remove it (D6 −0.98, 4/5).
- Item uncertainty is large (×1.65; diagnostic 5/5 under full-chain SE, mean z −1.02).
- Drifts further under repeated updates (0/5 at k = 6).
- **A different mechanism from the PIGD pair:** tremor-item estimation error, part scale, part item-specific, plus understated uncertainty.

## 8. Hypothesis assessment

| # | hypothesis | assignment | evidence | limitations |
|---|---|---|---|---|
| **H-A** | latent-scale anchoring: the initial item model sets a distorted scale that one update cannot correct | **Supported** (β0_P, Le11_log); **partially supported** (Le00_log) | D1 c_P > 1, d_P < 0, explained shares 0.76 / 0.94; D4 affine parts +2.47 / −1.97 SE; D6 correct-scale items → −0.03 / −0.27; D5 updates move only along the relabelling direction (99.5–99.9 %) with flat objectives; [M] relabelling invariance and equivariance of the S1-b map, verified to ≤ 6e-7 | 5 datasets; single truth; D6 uses unknowable items; the alignment is a constraint, not an estimate |
| **H-B** | non-affine item error materially contributes | **Partially supported:** small for β0_P and Le11_log (−0.15, +0.17 SE); material for Le00_log (−0.72 of −1.49 SE) | D4 attribution; D1 residual after alignment; D1 N(0, 1)-item overshoot | first order; the affine/non-affine split depends on the chosen alignment |
| **H-C** | holding items fixed understates SEs materially | **Partially supported:** ×1.48 Le11_log, ×1.65 Le00_log, ×1.20 L22_log; not for β0_P (×1.11, still 3/5) | D4 full-chain sandwich, numerically validated | asymptotic; no small-sample correction; bias remains (mean z −1.39 / −1.02) |
| **H-D** | one iteration inadequate, or repeated updates stabilise at a biased point | **Not supported in either form.** Further updates neither converge nor approach the truth; they drift along an unidentified direction at a constant rate, in dataset-specific directions | D5 trajectories, flat objectives, relabelling share, [M] equivariance | 4 further updates only; pre-declared K_MAX |
| **H-E** | within-stage-2 trade-offs explain the failure | **Not supported** | the stage 2 with true or correct-scale items recovers (Step 4, D6); fixed-item correlations among the failed components ≤ 0.13; first-order attribution fully reproduces the shifts from item errors | cannot exclude higher-order effects; the item–ψ coupling is structural (relabelling), not a stage-2 trade-off |
| **H-F** | separate cause for Le00_log (tremor items) | **Supported** | D4: 100 % tremor-item attribution, half non-affine; Step 5 split (sampling −0.76 vs assumption −0.50); D6 residual −0.98; largest SE inflation (×1.65) | the tremor-items scale and non-scale parts are not separated beyond first order |

A failed diagnostic was not counted as evidence for a competitor.

## 9. Numerical checks, convergence, limitations and resources

**Convergence.** Every D6 and D5 stage-2 fit converged (L-BFGS-B relative-reduction criterion):
- D6: 78–102 iterations, ‖∇‖ ≤ 0.0009;
- D5: 20 fits, all converged.

There were no bound hits, no fallback or budget events at any optimum, and all final Hessians are PD. D4 had no FD failures, and its numerical checks are in §5.2.

**Limitations.**
- **Five synthetic datasets from one a-priori truth.** Every rank or proportion is descriptive; the CP intervals for k/5 are wide (e.g. 3/5: 0.15–0.95).
- D6 and the Step 5 split use the unknowable truth.
- The D4 sandwich is asymptotic and first-order.
- D5 ran 4 updates under a pre-declared cap.
- The D1/D4 affine split depends on one (c, d) per subscore.

**Resources** (node-hours = AllocNodes × ElapsedRaw / 3600):

| job | purpose | node-hours |
|---|---|---|
| 13933–13936 | smoke checks | 0.112 |
| 13938 | D6 | 0.369 (estimate 0.4) |
| 13939 | D4 | 0.300 (estimate 0.3) |
| 13940 | D5 | 1.398 (estimate 1.5) |
| **total** | | **2.179** (budget 4) |

No diagnostic exceeded its estimate. Phase A ran on the login node in seconds.

## 10. Observations, interpretations, untested hypotheses

**Observations (these five datasets):** §1 points 1–6; §§4–7 tables.

**Interpretations:**
1. The failure is a scale-anchoring and identification problem of the two-stage-with-S1-b procedure relative to the truth's scale convention. It is compounded by understated item uncertainty in the residual SDs.
2. Stage 2 and its numerical approximation are not implicated.

**Untested hypotheses:**
- **[H1]** Re-expressing each estimate on the truth's convention (pooled visit-level latent mean 0, variance 1, computed from the fitted ψ̂ itself) removes most of the β0_P and Le11_log bias and collapses the D5 drift.
- **[H2]** Intervals that propagate item uncertainty (the D4 full-chain sandwich, plus the delta method for that standardisation) would restore the ≥ 4/5 rule for the residual SDs.
- **[H3]** The pattern generalises to other truths, designs and seeds.
- **[H4]** Joint estimation (S1-c) would face the same identification question, because the joint likelihood is relabelling-invariant: it, too, needs an explicit scale constraint.

## 11. Answers to the nine questions

1. **Why did Step 4 pass and Step 6 fail?** Step 4 fixes the items at their generating values, which pins the latent scale to the truth's. Step 6 estimates the items. Its scale is pinned only by the N(0, 1) stage-1 working assumption, which places the PIGD scale at c_P ≈ 1.05–1.16, d_P ≤ 0 relative to the truth's convention. Stage 2 then estimates ψ consistently on that relabelled scale. The ψ̂ − ψ̂_Step4 differences are reproduced from the item errors to within 0.10 SE (D4).
2. **How much error is present after the first stage-2 fit?** 90 % (β0_P), 86 % (Le00_log) and 94 % (Le11_log) of the final mean z. S1-b's average effect is +0.19, −0.23 and −0.12 SE.
3. **Does correcting the item scale (D6) recover the failed parameters?** For β0_P and Le11_log, yes (mean z −0.03, −0.27; 5/5 diagnostic). For Le00_log, only partly (−0.98; 4/5). This is a counterfactual with unknowable items.
4. **Does D4 show defensibly that item uncertainty explains the failure?** Only partly. It shows, with numerically validated derivatives, that fixed-item intervals understate the SE of Le11_log (×1.48) and Le00_log (×1.65). It does not explain β0_P (×1.11, bias +2.19 SE remains), and it corrects no bias. It is asymptotic and is not a re-scoring.
5. **Does D5 converge, and would convergence imply recovery?** It does not converge. It drifts at a constant rate along the exactly unidentified relabelling direction, with flat objectives. Even a converged S1-b point would not imply recovery: the map is equivariant, so its fixed points form a relabelling family, and which member is reached is set by the starting anchor.
6. **Are the causes shared?** β0_P and Le11_log share one cause: PIGD scale anchoring, mostly affine. Le00_log differs: tremor-item error, roughly half non-affine, with the largest understated uncertainty.
7. **What is the smallest distinguishing next experiment?** D7, read-only, seconds (§12). Re-express ψ̂⁽¹⁾, ψ̂⁽²⁾, the D5 iterates and D6 on the truth's convention, using each estimate's own fitted moments. Then check (a) that the D5 trajectory collapses, as it must if the drift is pure relabelling, and (b) the remaining β0_P, Le11_log and Le00_log errors, under the frozen SE and under the delta-method full-chain SE.
8. **What assumptions or model changes would that require?**
   - **D7 itself:** none to the model. It only applies a deterministic relabelling under which the likelihood is exactly invariant. It does adopt, **as a diagnostic**, a scale convention different from Step 6's approved "no re-standardisation" (E-2).
   - **Using it as a method:** a dated amendment fixing the identification convention, defining the interval method (delta method plus item-uncertainty propagation, which changes doc 16 B5), and a new frozen recovery test.
9. **What is specific to these five datasets, and what is unestablished?**
   - **Specific:** all magnitudes, coverage counts, attribution shares and D5 drift rates.
   - **Unestablished:**
     - generality to other truths, designs and seeds;
     - whether D7's convention removes the bias;
     - whether any amended procedure would pass a frozen recovery test;
     - anything about real PPMI data, model fit to real data, or subtypes.
## 12. Smallest recommended next step

**D7 — scale-convention re-expression of the saved estimates (diagnostic, read-only).** Specified in the companion proposal, doc 41.

**What it does.** For each saved estimate (ψ̂⁽¹⁾, ψ̂⁽²⁾, D5 ψ̂₃…ψ̂₆, D6 ψ̂):
- compute, from that estimate alone, the pooled visit-level latent mean μ_s and variance v_s over the real design. This is the `truth_on_stage1_scale` formula, applied to ψ̂;
- re-express ψ̂ with c_s = √v_s, d_s = μ_s, so that it satisfies the convention that defines ψ_true;
- propagate the frozen H⁻¹ and the D4 full-chain V through the Jacobian of that map (delta method; numerical derivative).

**What it reports.** The re-expressed z for all 19 components, and descriptive counts. It uses no truth in the transformation, fits nothing, and draws no random numbers.

**Why it is the smallest discriminating step.** It tests [H1] directly, with sharp, falsifiable predictions:
- the D5 trajectory should collapse (the per-dataset spread of re-expressed estimates across k = 2…6 ≪ 1 SE), because the drift is a relabelling;
- the β0_P and Le11_log mean z should move toward 0, by about the D4 affine parts.

If they do not, H-A is weakened and H-B or other mechanisms gain weight.

**Cost:** seconds on the login node. It needs the real design only (`load_cohort`, as in every step) and the saved outputs.

**What it cannot do:** it is not a recovery test, it uses the five datasets that have now been studied, and it does not authorise any change of procedure.

**Further steps, each needing separate approval.** If D7 supports H1, then:
1. a dated amendment (Rev 3) to the Stage 2 estimation track, specifying:
   - the identification convention;
   - the interval method (propagating item uncertainty);
   - whether S1-b is kept, given D5;
2. a new **frozen** recovery test. Its datasets (the same 5 seeds vs fresh seeds), number and criterion are to be decided **before** it is run.

Step 7, Step 8, calibration and real-data work remain blocked throughout.

## 13. Decisions requiring separate approval

1. Whether to run **D7** (read-only, seconds) as specified in doc 41.
2. Whether to adopt, **for diagnostics only**, the pooled-moment scale convention that D7 uses. It differs from Step 6's approved E-2 ("no re-standardisation"); Step 6 itself is unaffected.
3. Whether to open an **amendment** (identification convention; interval method including item uncertainty, which changes doc 16 B5; the role of S1-b).
4. For any future recovery test: **fresh seeds or the existing five**; the number of datasets; the criterion — fixed before running.
5. Whether **S1-c** (joint estimation) should be considered at all. It is not recommended now: it faces the same identification question [H4], and it needs its own amendment.
6. Whether to keep the new untracked diagnostic files (scripts 53/54, tests, docs 39–41, the run directory). Commit or stage decisions are the user's.

## 14. Provenance

| item | value |
|---|---|
| execution directory | `experiments/20261019T084559Z_stage2_step6_diagnostics/` (`00_baseline.json`: 195 prior output files, 61 docs, 94 code files hashed before any work) |
| read-only script | `scripts/53_m2_step6_diag_readonly.py` `021fd53c4cdd30d5bd130454790908b6ec82055b0137d05ed150bd9e1cd24554` → `readonly/` (MANIFEST verified) |
| compute driver | `scripts/54_m2_step6_diag_compute.py` `fc6575f4b0cfe2cb46e95a0cb2ef819ba1cbf6b848a5b70b4fbc2a2749778cce` (as run; `03_driver_sha256_full_runs.txt`) |
| tests | `tests/test_step6_diagnostics.py` `6f0e6d75f039f7920e5f0817d7229f950e1c5c15eeeb8f4a1044746058341df7` (10 passed); together with the existing M2 suites, 79 passed |
| jobs (node002) | smoke 13933 (d6), 13934 (d4), 13935 (d5), 13936 (d4 re-smoke); full 13938 (D6), 13939 (D4), 13940 (D5) |
| outputs | `readonly/`, `d6/`, `d4/`, `d5/`, each with `summary.json` and a verified `MANIFEST.sha256`; smoke dirs `d6_smoke/`, `d4_smoke/`, `d4_smoke2/`, `d5_smoke/` (not results); `jobs_sacct.txt`; `99_post_run_integrity.txt` |
| seeds | datasets `[42, 45, 1, r]`, r = 0…4 only; no other random numbers |
| settings | Q = 15; G = 5 (reported context); FD h = 1e-4 (items and ψ); item fits 61-node GH, maxiter 3000; stage 2 maxiter 1000; D5 K_MAX = 6, settle 0.05 SE |
| code reused (unchanged, hash-verified) | `m2_quad.py` `50f164c6…`, `m2_grm.py` `585c878b…`, `grm.py` `1584bc62…`, `45_m2_pilot.py` `a25e1503…`, `51_…` `517c4cc1…`, `52_…` `7e95263c…` |
| integrity after the work | all frozen locks OK; all pre-existing docs, code and 195 prior output files unchanged; panel MD5 unchanged; only new files: scripts 53/54, the test file, docs 40–41 and the run directory; git HEAD `3a860ca`, 0 tracked changes, 0 staged |

## 15. Status statement

**Step 6 remains FAILED** under its frozen, unchanged criterion. No diagnostic result here is a recovery result, and none replaces the Step 6 verdict or intervals. **D1 remains NOT REPRODUCED.**

No Step 7 or Step 8, real-data fit, calibration, positive control, gate or evaluation statistic was run. No repair or new estimator was implemented. Nothing was staged, committed or pushed.

---

## Appendix A — D3 for all 19 components (Step 4 / ψ̂⁽¹⁾ / ψ̂⁽²⁾)

| ψ component | stage | covered (CP 95 %) | mean err | mean SE | mean z | RMS z | bias t | SD/mean SE (χ² 95 %) | centred-coverage diagnostic | SE inflation for ≥ 4/5 |
|---|---|---|---|---|---|---|---|---|---|---|
| beta0_T | step4 | 5/5 (0.48–1.00) | +0.0002 | 0.0314 | +0.00 | 0.59 | +0.02 | 0.66 (0.40–1.91) | 5/5 | 0.38 |
| beta0_T | psi1 | 4/5 (0.28–0.99) | -0.0055 | 0.0306 | -0.18 | 1.20 | -0.30 | 1.34 (0.80–3.86) | 4/5 | 0.79 |
| beta0_T | psi2 | 4/5 (0.28–0.99) | -0.0093 | 0.0303 | -0.31 | 1.57 | -0.40 | 1.73 (1.04–4.98) | 4/5 | 0.92 |
| beta1_T | step4 | 4/5 (0.28–0.99) | +0.0017 | 0.0071 | +0.24 | 1.26 | +0.38 | 1.40 (0.84–4.02) | 4/5 | 0.86 |
| beta1_T | psi1 | 5/5 (0.48–1.00) | +0.0001 | 0.0069 | +0.03 | 1.17 | +0.03 | 1.31 (0.78–3.75) | 5/5 | 0.90 |
| beta1_T | psi2 | 5/5 (0.48–1.00) | +0.0001 | 0.0069 | +0.02 | 1.19 | +0.01 | 1.33 (0.80–3.83) | 5/5 | 0.91 |
| gamma_T | step4 | 5/5 (0.48–1.00) | +0.0137 | 0.0319 | +0.44 | 0.86 | +1.17 | 0.82 (0.49–2.36) | 5/5 | 0.39 |
| gamma_T | psi1 | 5/5 (0.48–1.00) | +0.0116 | 0.0310 | +0.36 | 0.88 | +0.93 | 0.90 (0.54–2.58) | 5/5 | 0.47 |
| gamma_T | psi2 | 5/5 (0.48–1.00) | +0.0105 | 0.0308 | +0.32 | 0.88 | +0.83 | 0.91 (0.55–2.62) | 5/5 | 0.52 |
| beta0_P | step4 | 5/5 (0.48–1.00) | -0.0025 | 0.0305 | -0.08 | 0.65 | -0.25 | 0.72 (0.43–2.07) | 5/5 | 0.43 |
| beta0_P | psi1 | 3/5 (0.15–0.95) | +0.0646 | 0.0295 | +2.19 | 2.32 | +5.68 | 0.86 (0.52–2.48) | 5/5 | 1.35 |
| beta0_P | psi2 | 3/5 (0.15–0.95) | +0.0703 | 0.0288 | +2.44 | 2.69 | +4.38 | 1.25 (0.75–3.58) | 4/5 | 1.42 |
| beta1_P | step4 | 5/5 (0.48–1.00) | +0.0040 | 0.0080 | +0.51 | 0.92 | +1.31 | 0.86 (0.52–2.48) | 5/5 | 0.59 |
| beta1_P | psi1 | 5/5 (0.48–1.00) | -0.0045 | 0.0074 | -0.60 | 0.77 | -2.47 | 0.55 (0.33–1.57) | 5/5 | 0.47 |
| beta1_P | psi2 | 5/5 (0.48–1.00) | -0.0039 | 0.0075 | -0.52 | 0.68 | -2.45 | 0.48 (0.29–1.37) | 5/5 | 0.32 |
| gamma_P | step4 | 4/5 (0.28–0.99) | -0.0212 | 0.0331 | -0.64 | 1.40 | -1.03 | 1.40 (0.84–4.02) | 5/5 | 0.65 |
| gamma_P | psi1 | 4/5 (0.28–0.99) | -0.0176 | 0.0313 | -0.58 | 1.32 | -0.95 | 1.33 (0.79–3.81) | 5/5 | 0.68 |
| gamma_P | psi2 | 4/5 (0.28–0.99) | -0.0270 | 0.0311 | -0.90 | 1.55 | -1.39 | 1.40 (0.84–4.03) | 4/5 | 0.63 |
| L00_log | step4 | 4/5 (0.28–0.99) | +0.0115 | 0.0337 | +0.37 | 1.18 | +0.61 | 1.25 (0.75–3.60) | 5/5 | 0.64 |
| L00_log | psi1 | 5/5 (0.48–1.00) | -0.0153 | 0.0337 | -0.44 | 0.96 | -1.04 | 0.97 (0.58–2.80) | 5/5 | 0.52 |
| L00_log | psi2 | 5/5 (0.48–1.00) | -0.0246 | 0.0337 | -0.72 | 1.13 | -1.67 | 0.98 (0.59–2.81) | 5/5 | 0.84 |
| L10 | step4 | 4/5 (0.28–0.99) | +0.0020 | 0.0072 | +0.30 | 1.28 | +0.46 | 1.39 (0.83–4.00) | 4/5 | 0.47 |
| L10 | psi1 | 4/5 (0.28–0.99) | +0.0016 | 0.0070 | +0.24 | 1.31 | +0.34 | 1.46 (0.88–4.20) | 4/5 | 0.51 |
| L10 | psi2 | 4/5 (0.28–0.99) | +0.0020 | 0.0069 | +0.29 | 1.32 | +0.43 | 1.48 (0.89–4.25) | 4/5 | 0.48 |
| L11_log | step4 | 5/5 (0.48–1.00) | -0.0330 | 0.0958 | -0.21 | 0.75 | -0.71 | 1.08 (0.65–3.10) | 5/5 | 0.14 |
| L11_log | psi1 | 5/5 (0.48–1.00) | -0.0597 | 0.0956 | -0.50 | 0.84 | -1.38 | 1.01 (0.61–2.90) | 5/5 | 0.31 |
| L11_log | psi2 | 5/5 (0.48–1.00) | -0.0636 | 0.0954 | -0.55 | 0.83 | -1.60 | 0.93 (0.56–2.68) | 5/5 | 0.41 |
| L20 | step4 | 5/5 (0.48–1.00) | +0.0147 | 0.0325 | +0.46 | 0.79 | +1.43 | 0.71 (0.43–2.04) | 5/5 | 0.58 |
| L20 | psi1 | 5/5 (0.48–1.00) | +0.0120 | 0.0313 | +0.39 | 0.77 | +1.15 | 0.74 (0.44–2.13) | 5/5 | 0.55 |
| L20 | psi2 | 5/5 (0.48–1.00) | +0.0100 | 0.0306 | +0.33 | 0.75 | +0.98 | 0.75 (0.45–2.14) | 5/5 | 0.54 |
| L21 | step4 | 4/5 (0.28–0.99) | -0.0098 | 0.0638 | -0.14 | 1.23 | -0.28 | 1.23 (0.74–3.55) | 4/5 | 0.81 |
| L21 | psi1 | 4/5 (0.28–0.99) | -0.0097 | 0.0616 | -0.16 | 1.22 | -0.29 | 1.23 (0.74–3.53) | 4/5 | 0.79 |
| L21 | psi2 | 4/5 (0.28–0.99) | -0.0085 | 0.0601 | -0.14 | 1.21 | -0.26 | 1.21 (0.73–3.48) | 4/5 | 0.77 |
| L22_log | step4 | 5/5 (0.48–1.00) | -0.0003 | 0.0420 | +0.01 | 0.64 | -0.03 | 0.70 (0.42–2.01) | 5/5 | 0.24 |
| L22_log | psi1 | 5/5 (0.48–1.00) | -0.0299 | 0.0416 | -0.70 | 0.94 | -2.25 | 0.72 (0.43–2.06) | 5/5 | 0.52 |
| L22_log | psi2 | 4/5 (0.28–0.99) | -0.0599 | 0.0420 | -1.41 | 1.61 | -3.51 | 0.91 (0.55–2.62) | 5/5 | 0.82 |
| L30 | step4 | 5/5 (0.48–1.00) | -0.0013 | 0.0077 | -0.17 | 0.70 | -0.48 | 0.79 (0.47–2.26) | 5/5 | 0.44 |
| L30 | psi1 | 5/5 (0.48–1.00) | -0.0014 | 0.0072 | -0.21 | 0.74 | -0.55 | 0.82 (0.49–2.35) | 5/5 | 0.44 |
| L30 | psi2 | 5/5 (0.48–1.00) | -0.0011 | 0.0073 | -0.17 | 0.73 | -0.42 | 0.82 (0.49–2.36) | 5/5 | 0.46 |
| L31 | step4 | 5/5 (0.48–1.00) | -0.0052 | 0.0110 | -0.41 | 0.75 | -1.38 | 0.76 (0.46–2.19) | 5/5 | 0.44 |
| L31 | psi1 | 5/5 (0.48–1.00) | -0.0050 | 0.0102 | -0.43 | 0.78 | -1.36 | 0.81 (0.49–2.33) | 5/5 | 0.47 |
| L31 | psi2 | 5/5 (0.48–1.00) | -0.0051 | 0.0103 | -0.43 | 0.76 | -1.40 | 0.78 (0.47–2.25) | 5/5 | 0.45 |
| L32 | step4 | 5/5 (0.48–1.00) | -0.0042 | 0.0086 | -0.47 | 0.70 | -1.76 | 0.61 (0.37–1.76) | 5/5 | 0.32 |
| L32 | psi1 | 5/5 (0.48–1.00) | -0.0072 | 0.0080 | -0.88 | 0.98 | -3.80 | 0.53 (0.32–1.52) | 5/5 | 0.49 |
| L32 | psi2 | 5/5 (0.48–1.00) | -0.0051 | 0.0081 | -0.61 | 0.77 | -2.50 | 0.57 (0.34–1.62) | 5/5 | 0.34 |
| L33_log | step4 | 5/5 (0.48–1.00) | +0.0007 | 0.0729 | +0.03 | 0.46 | +0.04 | 0.50 (0.30–1.44) | 5/5 | 0.26 |
| L33_log | psi1 | 5/5 (0.48–1.00) | -0.0937 | 0.0748 | -1.23 | 1.44 | -3.23 | 0.87 (0.52–2.49) | 5/5 | 0.92 |
| L33_log | psi2 | 5/5 (0.48–1.00) | -0.0665 | 0.0730 | -0.88 | 1.24 | -2.06 | 0.99 (0.59–2.85) | 5/5 | 0.79 |
| Le00_log | step4 | 5/5 (0.48–1.00) | -0.0061 | 0.0242 | -0.23 | 1.07 | -0.49 | 1.16 (0.69–3.33) | 5/5 | 0.66 |
| Le00_log | psi1 | 3/5 (0.15–0.95) | -0.0362 | 0.0243 | -1.47 | 1.89 | -2.49 | 1.33 (0.80–3.84) | 4/5 | 1.07 |
| Le00_log | psi2 | 3/5 (0.15–0.95) | -0.0419 | 0.0243 | -1.71 | 2.24 | -2.41 | 1.60 (0.96–4.61) | 4/5 | 1.53 |
| Le10 | step4 | 5/5 (0.48–1.00) | +0.0052 | 0.0166 | +0.31 | 0.86 | +0.77 | 0.91 (0.54–2.60) | 5/5 | 0.26 |
| Le10 | psi1 | 5/5 (0.48–1.00) | +0.0030 | 0.0157 | +0.18 | 0.88 | +0.44 | 0.98 (0.58–2.81) | 5/5 | 0.37 |
| Le10 | psi2 | 5/5 (0.48–1.00) | +0.0029 | 0.0156 | +0.17 | 0.90 | +0.41 | 1.01 (0.60–2.89) | 5/5 | 0.39 |
| Le11_log | step4 | 5/5 (0.48–1.00) | -0.0103 | 0.0345 | -0.29 | 0.45 | -1.69 | 0.39 (0.24–1.13) | 5/5 | 0.32 |
| Le11_log | psi1 | 1/5 (0.01–0.72) | -0.0688 | 0.0346 | -1.98 | 2.12 | -5.15 | 0.86 (0.52–2.48) | 5/5 | 1.18 |
| Le11_log | psi2 | 2/5 (0.05–0.85) | -0.0729 | 0.0345 | -2.11 | 2.35 | -4.02 | 1.17 (0.70–3.37) | 5/5 | 1.35 |

## Appendix B — per-dataset estimates (est (SE) z; * = not covered), Step 4 and Step 6 ψ̂⁽²⁾

| ψ component | stage | ds 0: est (SE) z | ds 1: est (SE) z | ds 2: est (SE) z | ds 3: est (SE) z | ds 4: est (SE) z |
|---|---|---|---|---|---|---|
| beta0_T | step4 | -0.1921 (0.0315) +0.46 | -0.2011 (0.0307) +0.18 | -0.1829 (0.0318) +0.74 | -0.2313 (0.0304) -0.81 | -0.2246 (0.0326) -0.55 |
| beta0_T | psi2 | -0.1263 (0.0305) +2.63* | -0.2196 (0.0307) -0.42 | -0.2293 (0.0292) -0.78 | -0.2419 (0.0301) -1.17 | -0.2623 (0.0310) -1.80 |
| beta1_T | step4 | +0.0357 (0.0072) -1.68 | +0.0485 (0.0070) +0.10 | +0.0499 (0.0073) +0.28 | +0.0497 (0.0069) +0.28 | +0.0638 (0.0071) +2.24* |
| beta1_T | psi2 | +0.0353 (0.0070) -1.79 | +0.0481 (0.0070) +0.04 | +0.0456 (0.0067) -0.33 | +0.0495 (0.0068) +0.25 | +0.0609 (0.0068) +1.92 |
| gamma_T | step4 | +0.1047 (0.0321) +0.28 | +0.1434 (0.0312) +1.53 | +0.0707 (0.0323) -0.77 | +0.1178 (0.0321) +0.69 | +0.1100 (0.0317) +0.45 |
| gamma_T | psi2 | +0.1006 (0.0312) +0.16 | +0.1431 (0.0311) +1.52 | +0.0655 (0.0296) -1.02 | +0.1164 (0.0319) +0.65 | +0.1049 (0.0302) +0.31 |
| beta0_P | step4 | -0.4885 (0.0303) -0.39 | -0.4435 (0.0303) +1.10 | -0.4768 (0.0312) +0.00 | -0.5023 (0.0305) -0.84 | -0.4853 (0.0304) -0.28 |
| beta0_P | psi2 | -0.3963 (0.0290) +2.78* | -0.4363 (0.0275) +1.47 | -0.3482 (0.0286) +4.49* | -0.4272 (0.0300) +1.66 | -0.4245 (0.0288) +1.82 |
| beta1_P | step4 | +0.1211 (0.0082) -0.06 | +0.1224 (0.0079) +0.10 | +0.1351 (0.0080) +1.68 | +0.1190 (0.0080) -0.32 | +0.1307 (0.0079) +1.16 |
| beta1_P | psi2 | +0.1166 (0.0079) -0.63 | +0.1123 (0.0073) -1.28 | +0.1211 (0.0072) -0.06 | +0.1176 (0.0079) -0.51 | +0.1207 (0.0072) -0.12 |
| gamma_P | step4 | +0.1415 (0.0331) -0.62 | +0.1556 (0.0328) -0.20 | +0.0773 (0.0333) -2.54* | +0.2046 (0.0331) +1.28 | +0.1256 (0.0328) -1.11 |
| gamma_P | psi2 | +0.1348 (0.0319) -0.86 | +0.1385 (0.0299) -0.79 | +0.0779 (0.0303) -2.78* | +0.1997 (0.0326) +1.15 | +0.1244 (0.0306) -1.23 |
| L00_log | step4 | -0.2466 (0.0331) +0.63 | -0.2809 (0.0339) -0.40 | -0.2409 (0.0333) +0.80 | -0.3111 (0.0350) -1.25 | -0.2000 (0.0329) +2.05* |
| L00_log | psi2 | -0.2800 (0.0332) -0.38 | -0.2797 (0.0340) -0.36 | -0.3260 (0.0333) -1.76 | -0.3249 (0.0351) -1.64 | -0.2496 (0.0330) +0.54 |
| L10 | step4 | +0.0098 (0.0071) +0.31 | +0.0016 (0.0071) -0.84 | +0.0109 (0.0069) +0.47 | +0.0255 (0.0071) +2.51* | +0.0007 (0.0076) -0.93 |
| L10 | psi2 | +0.0105 (0.0069) +0.41 | +0.0010 (0.0071) -0.93 | +0.0097 (0.0063) +0.32 | +0.0261 (0.0071) +2.59* | +0.0008 (0.0072) -0.95 |
| L11_log | step4 | -2.5702 (0.0885) +0.05 | -2.5607 (0.0795) +0.18 | -2.5655 (0.0949) +0.10 | -2.7926 (0.1320) -1.65 | -2.5512 (0.0842) +0.28 |
| L11_log | psi2 | -2.5970 (0.0888) -0.25 | -2.5604 (0.0793) +0.18 | -2.6519 (0.0947) -0.81 | -2.7865 (0.1304) -1.62 | -2.5974 (0.0841) -0.27 |
| L20 | step4 | +0.1013 (0.0322) +1.13 | +0.0682 (0.0321) +0.10 | +0.0633 (0.0331) -0.05 | +0.0577 (0.0332) -0.22 | +0.1075 (0.0319) +1.34 |
| L20 | psi2 | +0.0973 (0.0309) +1.05 | +0.0603 (0.0291) -0.16 | +0.0579 (0.0303) -0.23 | +0.0567 (0.0326) -0.25 | +0.1023 (0.0302) +1.24 |
| L21 | step4 | -0.0658 (0.0617) -0.96 | -0.0969 (0.0572) -1.58 | -0.0034 (0.0636) +0.05 | -0.0241 (0.0791) -0.22 | +0.1087 (0.0573) +2.01* |
| L21 | psi2 | -0.0649 (0.0591) -0.99 | -0.0846 (0.0517) -1.51 | -0.0037 (0.0582) +0.05 | -0.0238 (0.0771) -0.22 | +0.1018 (0.0545) +1.99* |
| L22_log | step4 | -0.4578 (0.0429) -0.46 | -0.4542 (0.0433) -0.37 | -0.3890 (0.0400) +1.23 | -0.4342 (0.0419) +0.10 | -0.4575 (0.0420) -0.46 |
| L22_log | psi2 | -0.5036 (0.0432) -1.51 | -0.5550 (0.0434) -2.69* | -0.4725 (0.0397) -0.86 | -0.4547 (0.0420) -0.39 | -0.5049 (0.0416) -1.60 |
| L30 | step4 | -0.0006 (0.0078) -0.08 | -0.0003 (0.0076) -0.04 | -0.0023 (0.0073) -0.31 | +0.0068 (0.0079) +0.87 | -0.0102 (0.0080) -1.28 |
| L30 | psi2 | -0.0007 (0.0076) -0.10 | -0.0001 (0.0070) -0.01 | -0.0021 (0.0066) -0.32 | +0.0070 (0.0078) +0.90 | -0.0097 (0.0073) -1.32 |
| L31 | step4 | -0.0023 (0.0109) -0.21 | +0.0056 (0.0097) +0.57 | -0.0095 (0.0110) -0.86 | -0.0167 (0.0133) -1.26 | -0.0030 (0.0103) -0.29 |
| L31 | psi2 | -0.0018 (0.0105) -0.17 | +0.0048 (0.0090) +0.54 | -0.0087 (0.0098) -0.89 | -0.0168 (0.0131) -1.28 | -0.0030 (0.0094) -0.32 |
| L32 | step4 | +0.0043 (0.0088) -0.63 | +0.0114 (0.0084) +0.20 | +0.0073 (0.0086) -0.29 | -0.0027 (0.0091) -1.37 | +0.0077 (0.0083) -0.24 |
| L32 | psi2 | +0.0048 (0.0085) -0.59 | +0.0109 (0.0078) +0.14 | +0.0050 (0.0076) -0.63 | -0.0021 (0.0090) -1.31 | +0.0047 (0.0076) -0.67 |
| L33_log | step4 | -2.2852 (0.0684) +0.73 | -2.3739 (0.0751) -0.51 | -2.3233 (0.0704) +0.17 | -2.3228 (0.0734) +0.17 | -2.3681 (0.0772) -0.42 |
| L33_log | psi2 | -2.3156 (0.0680) +0.29 | -2.4462 (0.0740) -1.50 | -2.4456 (0.0714) -1.55 | -2.3315 (0.0732) +0.05 | -2.4702 (0.0785) -1.72 |
| Le00_log | step4 | -0.7524 (0.0246) -0.61 | -0.7721 (0.0246) -1.41 | -0.7214 (0.0238) +0.68 | -0.7073 (0.0234) +1.29 | -0.7647 (0.0246) -1.11 |
| Le00_log | psi2 | -0.7811 (0.0247) -1.77 | -0.7751 (0.0246) -1.53 | -0.8088 (0.0238) -3.00* | -0.7168 (0.0235) +0.88 | -0.8149 (0.0247) -3.14* |
| Le10 | step4 | +0.0712 (0.0167) +1.83 | +0.0416 (0.0167) +0.06 | +0.0322 (0.0166) -0.50 | +0.0454 (0.0164) +0.30 | +0.0382 (0.0165) -0.14 |
| Le10 | psi2 | +0.0695 (0.0161) +1.80 | +0.0391 (0.0153) -0.10 | +0.0290 (0.0150) -0.77 | +0.0449 (0.0162) +0.27 | +0.0348 (0.0154) -0.37 |
| Le11_log | step4 | -0.9326 (0.0355) -0.69 | -0.9047 (0.0332) +0.10 | -0.9207 (0.0351) -0.36 | -0.9039 (0.0341) +0.12 | -0.9301 (0.0347) -0.63 |
| Le11_log | psi2 | -0.9725 (0.0355) -1.81 | -0.9881 (0.0327) -2.45* | -1.0270 (0.0355) -3.35* | -0.9179 (0.0342) -0.29 | -0.9995 (0.0346) -2.64* |
