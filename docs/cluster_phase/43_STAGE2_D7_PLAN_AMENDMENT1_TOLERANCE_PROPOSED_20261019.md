# Stage 2 M2: D7 plan (doc 41), Amendment 1 — numerical-check tolerance (proposed)

**Date:** 2026-10-19.

> **PROPOSED — NOT FROZEN.** Amends doc 41 (`41_STAGE2_NEXT_STEP_D7_SCALE_CONVENTION_PLAN_PROPOSED_20261019.md`, SHA-256 `b03e662e…`), which stays unchanged.
> - Approved by the user as "Option A" (doc 42 §4), subject to the safeguards in the user's instruction of 2026-10-19.
> - **Step 6 remains FAILED. D1 remains NOT REPRODUCED.**

## 1. Why

The original D7 plan **did not pass as written**. Its two built-in numerical checks require

- S(ψ_true) = ψ_true to ≤ 1e-10, and
- S(S(ψ)) = S(ψ) to ≤ 1e-10.

Neither can be met, because of a fixed regulariser in the existing covariance implementation. `m2_grm.covariance` adds `JITTER · I` with `JITTER = 1e-8`. The plan's moment formula is the `truth_on_stage1_scale` formula, which uses that function. The jitter does not rescale under a latent relabelling, so the pooled variance after relabelling is (v − J)/c² + J rather than v/c².

Doc 42 §3 records the original failed checks:
- fixed point 1.7e-9 on the real design;
- idempotence 1.4e-9 on a toy design.

## 2. Verification that the identities are the intended ones (before execution)

The three identities were evaluated on the real design in a throwaway verification process: once with the frozen `JITTER = 1e-8` and once with `JITTER = 0`, set by monkeypatching in that process only. **No code was changed.** ψ_true was rebuilt under the same JITTER for the fixed-point identity, and a generic ψ was used for the other two. **No saved estimate was used.**

| identity | JITTER = 1e-8 (frozen) | JITTER = 0 |
|---|---|---|
| S(ψ_true) = ψ_true | 1.71e-9 | 2.2e-16 |
| S(S(ψ)) = S(ψ) | 2.57e-9 | 4.4e-16 |
| S(relabel(ψ)) = S(ψ) | 6.9e-10 | 3.3e-16 |

**The whole residual is the jitter term.** Without it, all three identities hold to machine precision. A tolerance of 1e-8 (= JITTER) therefore still tests the intended identities, at the precision the frozen code allows. The residual is below 1e-7 of any standard error.

## 3. The amendment (only change)

| item | doc 41 | amended |
|---|---|---|
| built-in check: S(ψ_true) = ψ_true | ≤ 1e-10 | **≤ 1e-8** |
| built-in check: S(S(ψ)) = S(ψ) | ≤ 1e-10 | **≤ 1e-8** |
| corresponding unit tests (`tests/test_d7_scale_convention.py`): the fixed point S(ψ_true) = ψ_true; idempotence; S(relabel(ψ)) = S(ψ). The last is asserted in the same test as idempotence and is affected by the same jitter term | 1e-12 (1e-11 for invariance) | **1e-8** |

**Unchanged:**
- the moment formula and the transformation S;
- the uncertainty propagation;
- the inputs; the outputs;
- the predictions and their reading (doc 41 §3);
- every scientific threshold, model parameter and estimator;
- the interpretation;
- the script `scripts/55_m2_scale_convention_readonly.py` (`3a6b63d0…`), which is run unchanged.

The other unit-test assertions keep their original tolerances: equality with `truth_on_stage1_scale` at 1e-12, and the Jacobian annihilating relabelling directions at 1e-4 relative.

**Stop rule.** Any further discrepancy stops D7 and is documented. No tolerance is widened again.

## 4. Status

- Proposed, not frozen.
- Execution is authorised under this proposed amended tolerance.
- Results are appended to doc 42.
