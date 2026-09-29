# Step 3 (GPR) — test results

Date **2026-09-30**. Command: `python -m pytest -q` from the repository root.

## Result

```
256 passed in 27.24s
```

**256 tests, 256 passed, 0 failed, 0 skipped, 0 xfailed, 0 errors.**
Verified by `--collect-only` (256 collected) so nothing was deselected.

## By file

| File | Tests | Scope |
|---|---:|---|
| `test_gpr.py` | 52 | kernels, batched marginal likelihood, gradients, fitting, posterior, **K0 ≡ hbayes**, subtype probabilities, mean-function variants |
| `test_gpr_loo.py` | 33 | fold data, baselines, `run_fold`, multistep, LOPO, **checkpointing**, metrics |
| `test_trajectory.py` | 27 | construction, ordering, extreme criteria, cohorts, prediction points, leakage |
| `test_subtype.py` | 34 | pre-existing |
| `test_transitions.py` | 34 | pre-existing |
| `test_ctmc.py` | 27 | pre-existing |
| `test_hbayes.py` | 26 | pre-existing |
| `test_hmm.py` | 17 | pre-existing |
| `test_validate.py` | 6 | pre-existing |

Step 3 contributes **112** of the 256. The 144 pre-existing tests all still pass,
so no earlier research stage was broken.

## Correcting the historical count

The figure "101 tests" is superseded. It counted `test_gpr.py`,
`test_gpr_loo.py` and `test_trajectory.py` at an earlier moment. The current
Step 3 total is **112**, the increase being the 11 regression tests added by
this audit.

## Correcting the historical runtime

An earlier full-suite run was recorded at **3,811 s (63 min)**. It now completes
in **27 s**, with the slowest single test at 4.65 s and nothing skipped.

The 63-minute figure was machine contention, not a property of the suite — the
same cause that made the A4 LOPO take 1,026 min against 105 min for the larger
unfiltered cohort on the same code. Both occurred on 17–18 Sep.

## Slowest tests

| Test | Time |
|---|---:|
| `test_hmm.py::TestFitting::test_recovers_latent_generator_and_emission` | 4.65 s |
| `test_hmm.py::TestFitting::test_loglik_improves_when_noise_is_allowed` | 4.25 s |
| `test_hmm.py::TestFitting::test_zero_noise_reduces_to_the_plain_chain` | 2.91 s |
| `test_gpr_loo.py::TestRunLopo::test_serial_and_parallel_agree` | 1.37 s |
| `test_hbayes.py::TestSampler::test_recovers_known_parameters` | 1.29 s |

## Regression tests added by this audit (11)

### Checkpointing — defect 1

| Test | Invariant |
|---|---|
| `test_resumed_run_matches_an_uninterrupted_run` | a run interrupted at fold 4 of 8 and resumed returns a frame **identical** to an uninterrupted run |
| `test_checkpoint_is_written_during_the_run` | all three checkpoint files appear; no `.tmp` left behind |
| `test_a_corrupt_checkpoint_is_ignored_rather_than_trusted` | a truncated file is discarded, not resumed from |
| `test_checkpoint_never_shrinks_the_requested_work` | a checkpoint from a smaller patient set cannot cause new patients to be skipped |

### Mean function — defect 2

| Test | Invariant |
|---|---|
| `test_zero_mean_fit_returns_exactly_zero_beta` | `fit_mean=False` really fixes β at 0 |
| `test_fitted_mean_recovers_a_planted_offset_and_slope` | β recovers a planted 4.0 − 0.30·t |
| `test_zero_mean_gradient_matches_finite_differences` (×4 kernels) | the zero-mean objective is as trustworthy as the fitted-mean one, so a comparison measures the model not the optimiser |
| `test_a_fitted_mean_cannot_lose_to_zero_mean_in_sample` | the nested model cannot have higher likelihood |

## Scientific invariants already covered (not merely "tests pass")

- **No future leakage** — history slice ends strictly before the target, checked on indices *and* times.
- **Held-out isolation** — no training sequence equals the held-out patient's.
- **Correct patient grouping** — every fold trains on exactly N−1.
- **Batched likelihood == per-patient reference** to 2e-16, guarding the padding/mask logic.
- **Analytic gradients == finite differences** for all four kernels, both mean variants.
- **Kernel PSD**, posterior covariance symmetric and PSD.
- **Synthetic recovery** — GP recovers sin(t); `fit_gpr` recovers planted τ₀², τ₁², σ².
- **K0 ≡ hbayes** analytically and against the sampler itself.
- **Reproducibility** — serial and parallel runs agree exactly; restart 0 is seed-independent.
- **Metric correctness** — CRPS against numerical integration; coverage, MAE, confusion on hand-checkable cases.
- **Edge cases** — single-visit, two-visit, undefined ratios, duplicate times, exact-zero ratios.
