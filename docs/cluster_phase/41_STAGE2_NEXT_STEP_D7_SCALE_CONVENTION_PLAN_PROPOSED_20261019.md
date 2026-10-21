# Stage 2 M2: proposed next step — D7, scale-convention re-expression of saved estimates (diagnostic)

**Date:** 2026-10-19.

> **PROPOSED — NOT FROZEN — NO FURTHER COMPUTATION AUTHORISED.**
> - D7 is a read-only diagnostic. It is not a repair, an estimator, or a recovery test.
> - **Step 6 remains FAILED** (doc 38). **D1 remains NOT REPRODUCED.**
> - Steps 7–8, calibration, positive controls, gates and real-data work remain blocked.

**Motivation:** doc 40 (Step 6 failure investigation) §§10–13.

## 1. Question

The truth ψ_true is defined on a convention: the pooled visit-level latent of each subscore has mean 0 and variance 1 over the real design (`scripts/45_m2_pilot.py::truth_on_stage1_scale`).

The two-stage pipeline instead takes its scale from the N(0, 1) stage-1 working assumption. Under S1-b, that scale then drifts along an exactly unidentified relabelling direction (doc 40, D5).

**Question:** if each saved estimate is re-expressed on the truth's own convention, computed from the estimate alone and never from the truth,
- (a) does the D5 drift collapse, and
- (b) how much of the β0_P, Le11_log and Le00_log error remains?

## 2. Definition (exact)

**Data.** The real visit design only (times and treatment from `load_cohort`; no responses). For ψ, use `M.unpack` and, per subscore s, the visit-level mean M^s_ij(ψ) and variance K^s_ij,jj(ψ) over all 4,295 visits (`M.mean_design`, `M.covariance`).

**Moments.**
- μ_s(ψ) = mean_v M^s_v(ψ);
- v_s(ψ) = mean_v K^s_v,v(ψ) + var_v M^s_v(ψ) (population variance).

This is the same formula as `truth_on_stage1_scale`.

**Map.** S(ψ) = `relabel_psi(ψ, c_T = √v_T, d_T = μ_T, c_P = √v_P, d_P = μ_P)` (script 53). The likelihood is invariant under this map when the items are relabelled jointly (doc 40, §4.1).

**Built-in checks:**
- S(ψ_true) = ψ_true to ≤ 1e-10 (the truth satisfies the convention);
- S(S(ψ)) = S(ψ) to ≤ 1e-10 (idempotence).

**Uncertainty.** G = ∂S/∂ψ at ψ̂ by central differences (h = 1e-6), and V_S = G V Gᵀ for:
- (i) V = H⁻¹, the saved fixed-item Hessian of each fit;
- (ii) V = the D4 full-chain sandwich, available for ψ̂⁽²⁾ only.

**Inputs** (all saved; hash-verified against `experiments/20261019T084559Z_stage2_step6_diagnostics/00_baseline.json` and the run's manifests):
- Step 6 ψ̂⁽¹⁾, ψ̂⁽²⁾, H⁽¹⁾, H⁽²⁾;
- D5 ψ̂₃…ψ̂₆ and H₆;
- D6 ψ̂ and H;
- D4 V_full.

**Outputs:** for every estimate and dataset, z_S = (S(ψ̂) − ψ_true)/SE_S for all 19 components, under (i) and, for ψ̂⁽²⁾, (ii).
- **D5 collapse statistic:** per dataset, max over components of range_k(S(ψ̂_k))/SE⁽²⁾ for k = 2…6.
- **Mean z_S** of the three failed components.
- **Descriptive counts** (not a recovery test).

**No fit, no simulation, no random numbers, no change to any saved output.**

## 3. Predictions and how each outcome would be read (stated before running)

| observation | reading |
|---|---|
| D5 collapse statistic ≪ 1 SE (for example < 0.1) **and** mean z_S of β0_P and Le11_log near 0 (moved by about the D4 affine parts, +2.47 and −1.97 SE) | supports [H1]: the PIGD failure is a scale-convention (identification) mismatch. Next would be an amendment question, not a new estimator |
| collapse, but the β0_P / Le11_log bias largely persists | the convention explains the drift but not the bias: the N(0, 1)-anchored scale differs from the pooled-moment scale in a way the moments do not capture. Weight shifts to H-B (non-affine item error) |
| no collapse | the D5 movement is not confined to the scale/location directions the pooled moments fix. This contradicts the D5 relabelling analysis and needs a check of D7 itself |
| Le00_log | expected only partly corrected (half of its first-order attribution is non-affine, doc 40 §7). Its z_S under (ii) shows how far uncertainty propagation goes |

These are interpretation guides for a diagnostic, not acceptance criteria.

## 4. Implementation, tests and cost (on approval)

**Files** (new only):
- `scripts/55_m2_scale_convention_readonly.py`. It reuses `relabel_psi` from script 53 and the moment formula from `truth_on_stage1_scale`, inlined and tested for equality.
- Tests:
  - S(ψ_true) = ψ_true;
  - idempotence;
  - equality with `truth_on_stage1_scale` on the a-priori truth;
  - the delta-method Jacobian against an analytic special case.

**Output:** `experiments/20261019T084559Z_stage2_step6_diagnostics/d7_readonly/`, with a manifest.

**Cost:** seconds on the login node; **no SLURM job**.

## 5. Limitations

- **Same datasets.** It uses the five datasets that have already been studied, so it is hypothesis-strengthening at most. It cannot validate a procedure.
- **One convention.** The moments are those of the fitted latent at ψ̂. They depend on the model's Gaussian latent assumption, which matches the generating model here.
- **Diagnostic only.** Step 6's approved decision E-2 ("no re-standardisation") is unaffected. D7 adopts the other convention only to answer the diagnostic question.

## 6. What would come after (each step needs separate approval; none is authorised here)

1. **If D7 supports [H1]:** a dated amendment (Rev 3) to the Stage 2 estimation track. It would specify:
   - the identification convention used to report ψ̂;
   - the interval method (delta method plus item-uncertainty propagation; this changes doc 16 B5, which ignores item uncertainty);
   - whether S1-b is retained, given that its updates have no scale anchor (doc 40, D5).
2. **A new frozen recovery test** of the amended procedure. The datasets (fresh seeds recommended, to avoid testing on the datasets used for diagnosis), their number and the criterion are fixed before running.
3. **Only after a passing test:** the frozen sequence resumes at Step 7, unchanged.

## 7. Decisions requested

1. Approve or decline D7 as defined in §2.
2. If approved: confirm that D7 may use the pooled-moment convention as a **diagnostic** convention, leaving Step 6's E-2 and verdict unchanged.
3. Note for later (not now): fresh vs existing seeds for any future recovery test.
