# Stage 2 M2: Amendment 1, Revision 3 — scale-identification convention, interval method, S1-b role (proposed)

**Date:** 2026-10-19.

> **PROPOSED — NOT FROZEN — NO COMPUTATION AUTHORISED.**
> - It changes nothing until approved and frozen. Historical records stay as they are:
>   - **Step 6 remains FAILED** (doc 38) under the criterion then in force;
>   - **D1 remains NOT REPRODUCED**;
>   - docs 16, 21 (Rev 2), 25, 27, 29, 31, 34, 37 and their locks are not modified.
> - It applies only to estimation and reporting performed **after** its approval, and it must pass the new recovery test (doc 45) before any later step (Step 7 onward) may use it.

**Tags:**
- **[F]** frozen text in force today;
- **[O]** observation from the diagnostics (docs 40, 42; five synthetic datasets; descriptive);
- **[M]** mathematical statement with its numerical check;
- **[P]** proposed rule;
- **[D]** decision for the reviewer.

---

## 0. Scope

**Amends** (subject to approval):

| frozen text | today [F] | proposed [P] |
|---|---|---|
| doc 16 B5 | "Item parameters are held at their stage-1 values. Ignoring that item uncertainty is a stated limitation" | ψ is reported on the pooled-moment convention (§2), with intervals that propagate stage-1 and S1-b item uncertainty (§3) |
| doc 16 D3 / Rev 2 §0 (recovery definition) | 5 datasets, seeds `[42, 45, 1, r]`, every ψ ≥ 4/5 | the new independent recovery test of doc 45. The historical D3 result (Step 6 FAIL) is not re-scored |
| Rev 2 §5 (S1-b) | "Exactly one refit iteration … Accepted only if the unchanged full-pipeline recovery passes" | S1-b kept as one iteration (§4). Its acceptance is tied to doc 45 |
| Rev 2 §6 step 6 | "Full-pipeline recovery with S1-b: every ψ ≥ 4 of 5, unchanged criterion" | step 6′: the doc 45 test of the Rev 3 pipeline. Step 6 itself stays FAILED |
| doc 37 E-2 ("no re-standardisation") | binding for Step 6 | unchanged for Step 6. Superseded for future estimates by §2 |

**Unchanged:**
- model M2 (doc 16 B4);
- the item model and stage-1 fit;
- the Rev 2 likelihood (Laplace over b, per-visit 15 × 15 GH, Q = 15);
- Addendum A P1/P2 and the G = 5 fallback for reported evaluations;
- doc 27;
- every B11 criterion other than simulation recovery;
- G1–G7 and the G4 rule;
- the gate order;
- calibration, positive controls and evaluation statistics stay blocked.

## 1. Evidence basis (summary; details in docs 38, 40, 42)

- **[O]** Step 6 failed on β0_P (3/5, mean z +2.44), Le00_log (3/5, −1.71) and Le11_log (2/5, −2.11). Stage 2 recovers ψ with true or correctly scaled items (Step 4; D6).
- **[M]** The marginal likelihood is exactly invariant under a joint relabelling of each subscore's latent scale, θ′_s = (θ_s − d_s)/c_s, with items (c·a, (τ − d)/c) and ψ re-expressed accordingly. Checks: unit test 1e-9; the identity C·v_φ + H·v_ψ = 0 to ≤ 6e-7 (D4).
  - Consequence: the latent scale is identified only by a **convention**.
  - The two-stage pipeline takes it from the N(0, 1) stage-1 assumption. S1-b updates leave it unanchored: D5 drift is 99.5–99.9 % relabelling, with flat objectives.
- **[O]** When estimates are re-expressed on the convention that defines the truth (D7):
  - β0_P mean z goes from +2.44 to −0.10 (5/5), and Le11_log from −2.11 to −0.17 (5/5);
  - Le00_log goes only to −0.61 (4/5);
  - the extra uncertainty from item estimation shrinks from ×1.11–1.65 to ×0.92–1.12.
- **[O]** After re-expression, one S1-b iteration still changes some PIGD random-effect parameters relative to the N(0, 1)-only fit:

  | | L22_log | L33_log | γ_P |
  |---|---|---|---|
  | N(0, 1)-only, mean z | +0.91 | −0.43 | −0.27 |
  | with S1-b, mean z | +0.20 | +0.02 | −0.56 |

  Largest per-dataset change 0.81 SE. So S1-b corrects part of the non-scale (shape) error.

These observations come from the five datasets that motivated the hypothesis. They justify **proposing** Rev 3, not accepting it. Acceptance requires doc 45.

## 2. Scale-identification convention and the exact transformation [P]

### 2.1 Convention

For each subscore s ∈ {T, P}, the latent scale is the one on which the **pooled visit-level latent distribution of the analysis design** has mean 0 and variance 1.

This is the convention by which the protocol already defines the truth "on the stage-1 scale" (`scripts/45_m2_pilot.py::truth_on_stage1_scale`; Rev 2 §0). It is a **definition of the reporting scale**, not an assumption about the data.

### 2.2 Moments of an estimate

For a parameter vector ψ and the analysis design (each patient's visit times and treatment indicators; all V visits):
- (β, D, Σ_ε) = `m2_grm.unpack(ψ)`;
- M_v^s(ψ) = the visit-level latent mean;
- K_vv^s(ψ) = the visit-level latent variance (`m2_grm.mean_design`, `m2_grm.covariance`; the latter includes the frozen `JITTER = 1e-8`).

Then

  μ_s(ψ) = (1/V) Σ_v M_v^s(ψ),   v_s(ψ) = (1/V) Σ_v K_vv^s(ψ) + (1/V) Σ_v (M_v^s(ψ) − μ_s(ψ))².

### 2.3 Transformation of ψ

With c_s = √v_s(ψ̂) and d_s = μ_s(ψ̂), computed **from the estimate ψ̂ and the design only**:

  S(ψ̂) = relabel(ψ̂; c_T, d_T, c_P, d_P), where for each subscore s:
- β0_s′ = (β0_s − d_s)/c_s;
- β1_s′ = β1_s/c_s;
- γ_s′ = γ_s/c_s;
- D′ = C₄⁻¹ D C₄⁻¹, with C₄ = diag(c_T, c_T, c_P, c_P);
- Σ_ε′ = C₂⁻¹ Σ_ε C₂⁻¹, with C₂ = diag(c_T, c_P);
- the result is re-packed with `m2_grm.to_psi` (log-Cholesky).

Reference implementation: `scripts/53_m2_step6_diag_readonly.py::relabel_psi` and `scripts/55_m2_scale_convention_readonly.py::standardise`. Both are tested.

### 2.4 Items, transformed jointly

For every item k of subscore s: a_k′ = c_s·a_k and τ_kj′ = (τ_kj − d_s)/c_s. This must accompany S whenever items and ψ are used together.

### 2.5 Properties [M], with numerical tolerances

| property | tolerance | reason |
|---|---|---|
| (i) the likelihood and every model prediction (item-response probabilities, expected scores, posterior-predictive replicates, hence G1–G7 and any label) are unchanged by (S, item map) | — | joint relabelling invariance |
| (ii) S(relabel(ψ)) = S(ψ), so the result does not depend on how the pipeline anchored the scale | ≤ 1e-8 | O(JITTER) |
| (iii) S(S(ψ)) = S(ψ) | ≤ 1e-8 | O(JITTER) |
| (iv) S(ψ_true) = ψ_true | ≤ 1e-8 | O(JITTER) |

The JITTER-induced residuals are 0.7–2.6e-9, and vanish (≤ 4e-16) with JITTER = 0 (doc 43). The "design" in §2.2 is the analysis design. In synthetic tests it is the real design used to generate the data.

### 2.6 What the convention is not

- It does not use the truth.
- It is not an estimator change: point estimates of all model predictions are unchanged.
- It does not alter the model.

## 3. Uncertainty propagation [P]

### 3.1 Delta method

V_S = G V Gᵀ, where G = ∂S/∂ψ at ψ̂ by central differences (h = 1e-6; the D7 implementation). Checks:
- G annihilates the four relabelling tangent directions (≤ 1e-6 relative);
- rank(G) = 19 − 4.

### 3.2 Primary V — the full S1-b chain

V is the patient-clustered sandwich of the stacked estimating equations for θ = (φ₁, ψ₁, φ₂, ψ₂). These are the N(0, 1) item fit, the first stage-2 fit, the S1-b item refit and the final stage-2 fit, as implemented in `scripts/54_m2_step6_diag_compute.py` (D4):

  V = J⁻¹ (Σ_i u_i u_iᵀ) J⁻ᵀ

- J is block-lower-triangular: A₁, C₁, H₁, E₃, A₃, C₂, H₂, by central differences (h = 1e-4) of analytic gradients. Stage-2 gradients use the reported context (non-ND patients by G = 5 AGQ, exact gradient).
- u_i is patient i's stacked score.
- The ψ₂ block, transformed by G, gives SE_S.

This **propagates item-estimation uncertainty**, removing the doc 16 B5 limitation for the recovery-test intervals. It is asymptotic (n = 934 patients) and uses no small-sample correction.

### 3.3 Secondary V — reported, not used for the primary criterion

The fixed-item V = H₂⁻¹ (the Step 6 interval construction), transformed by G. It shows how much item uncertainty matters.

### 3.4 Items

Item parameters are re-expressed by §2.4. Their covariance is G_φ V_φφ G_φᵀ, with G_φ the Jacobian of the item map including its dependence on ψ̂ through (c, d). This is reported (item recovery stays "reported, not gated", as doc 16 D3).

### 3.5 Intervals

ψ̃_k ± 1.96·SE_S,k on the log-Cholesky parameterisation (the same parameterisation and Wald form as today).

## 4. Role of S1-b [P]

- **Kept,** exactly as frozen: one iteration, items refitted under the visit-specific fitted N(M_ij, K_ij,jj) at ψ̂⁽¹⁾ from `initial_items`, then stage 2 refitted from the default ψ with cold modes.
- **Its purpose is restated.** It is a one-step correction of the **shape** (non-affine) part of the stage-1 working-assumption error. It is **not** a way to fix the scale, because the scale is fixed by §2.
- **Rationale [O]:** after re-expression, S1-b improved L22_log and L33_log (§1). Its unanchored-scale behaviour (D5) is neutralised by S, by property 2.5(ii).
- **Not iterated.** D5 showed no convergence, and Rev 2 already fixes one iteration.
- **Comparator.** The N(0, 1)-only pipeline, S(ψ̂⁽¹⁾), is reported in doc 45 as a **non-binding** comparator. Its results may inform a later simplification, which would need its own amendment.

**[D1] Alternative:** drop S1-b (N(0, 1) stage 1 → stage 2 → S). It is simpler and about 40 % cheaper, but the five-dataset evidence above favours keeping it. **Recommendation: keep.**

## 5. Evaluation of the remaining Le00_log error [P]

1. **No exemption.** Le00_log is subject to the same doc 45 criteria as every component. A Le00_log failure fails the test.
2. **Pre-specified secondary analysis** (reported, never used to rescue a failure). Per dataset, the first-order attribution J₂Δφ (J₂ = −H₂⁻¹C₂) of S(ψ̂)'s deviation from the truth to:
   - the tremor vs PIGD items;
   - the affine vs non-affine item error (joint (c, d) alignment of the S-re-expressed items to the true items).

   Also reported: the Le00_log bias, the SD of its z, and coverage under the primary vs secondary V.
3. **Interpretation rules, fixed now:**

   | pattern | reading |
   |---|---|
   | Le00_log meets the criteria under the primary V but not the secondary | item uncertainty explains the residual |
   | it fails both, with a non-affine tremor-item attribution | tremor-item estimation error persists. A further amendment (for example tremor-item model checks) would be needed |
   | it fails with a negligible item attribution | a different cause, to be investigated |

## 6. Limitations of a convention defined by the synthetic generating process

1. **Definitional match.** In the synthetic tests the truth is **constructed** to satisfy this convention, so comparing S(ψ̂) with ψ_true is like with like by design.
   - Any error that is itself a relabelling is, by construction, not an error under the convention: it is not identified by the model.
   - Recovery must therefore also be shown on **relabelling-invariant functionals** that need no convention (doc 45 §6). That separates genuine recovery from apparent improvement.
2. **Design dependence.** The convention depends on the analysis design (visit times, treatment pattern, patient mix). Parameter values are comparable only across analyses with the same design and convention. For real data the convention is computable from ψ̂ and the real design, but the scale is "the cohort's pooled visit-level latent distribution", not an external reference.
3. **Reinterpretation.** β0_s becomes the baseline (t = 0, x = 0) latent mean relative to the cohort's pooled mean, in pooled-SD units. Slopes, the treatment effect and the SDs are in pooled-SD units. Scale-free conclusions (predictions, gates, correlations, ratios) are unaffected.
4. **Model dependence.** The moments are those of the fitted Gaussian latent model. Under latent misspecification the convention still defines a scale, but the "pooled variance" is the model's, not the data's.
5. **Single truth.** The doc 45 test uses the one a-priori truth and the real design. It does not establish performance under other truths (doc 45 §11).
6. **Numerical.** O(JITTER) = 1e-9 residuals in the identities (doc 43). They are negligible, but they mean exact identities are tested at 1e-8.

## 7. Decisions requested

| # | decision | recommendation |
|---|---|---|
| **D1** | keep one S1-b iteration (§4) or drop it | keep |
| **D2** | primary interval V: full-chain sandwich (§3.2) or fixed-item H⁻¹ | full-chain sandwich (it removes the B5 limitation; cost about 4 min per dataset) |
| **D3** | adopt the pooled-moment convention (§2) for all post-approval reporting of ψ | adopt |
| **D4** | replace the doc 16 D3 recovery definition (5 datasets, ≥ 4/5) by the doc 45 test for Rev 3 | replace (doc 45 §4 justifies it). The D3 rule is reported on fresh datasets 0–4 as a historical comparator |
| **D5** | whether the doc 16 B5 calibration-stage draws should later use the §3.2 covariance | defer: calibration is blocked |
| **D6** | freeze Rev 3 and doc 45 together | yes: both must be frozen before any test computation |

## 8. Not authorised by this document

- Any computation.
- Freezing.
- Code changes.
- The recovery test.
- Steps 7 and 8.
- Calibration.
- Real-data analysis.
- S1-c.
- Any change to historical results or verdicts.

Nothing is staged, committed or pushed.
