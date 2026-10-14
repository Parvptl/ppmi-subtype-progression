# Stage 6b — local audit note: skewed-null calibration (not a findings document)

Written 2026-10-14. Local record only; `docs/cluster_phase/05_FINDINGS.md` is
deliberately unchanged.

## Runs

| Run directory | What | Outcome |
|---|---|---|
| `experiments/20261014T133408Z_s6b_skewed_null` | calibration, default Nelder–Mead simplex | **cancelled by operator** after 29 evaluations: 5% initial simplex never left the start (see `CANCELLED_README.txt`) |
| `experiments/20261014T133917Z_s6b_skewed_null_cal2` | calibration, unit-step simplex, otherwise identical | **fidelity gate failed**; script exited with code 2 by design; no null simulated |

Both recorded pre-registration SHA-256 `1b01bb6a688afb0a5b8cf53189d140af5dd4ab3689cb53e1f8a55c19dc8bf76b`
(the file including Amendment 1). Git `60a925b` plus four uncommitted files,
archived in each run directory and in `checkpoint_20261014T143625Z_*`.

## Calibration 2 result (from `calibration.json`, `calibration_fidelity.csv`)

* Nelder–Mead: 293 evaluations, "Optimization terminated successfully",
  objective 18.23 → 0.296, flat from ~evaluation 200.
* PIGD slope: δ = 0.946, ν = 2.92 (skew near its limit of 1, ν near the 2.5 floor).
* Gate: PIGD-slope BLUP skewness 1.734 (MC SE 0.085) vs real 2.173, error 0.439
  > tolerance 0.30 → **FAIL**. All other gated targets pass.
* Not gated: PIGD-zero visits 0.217 simulated vs 0.161 real (MC SE 0.002);
  all-zero-PIGD patients 0.041 vs 0.041.

## Decisions recorded

* No clustering metric was computed on any skewed or tails null.
* Tolerances were not changed.
* Cross-node check: real-data ΔBIC / ARI are bit-identical on node003 and
  node004 (`node_repro_node003.json`).
* Next: audit the PIGD measurement model before any further null design.

## Measurement-model diagnostics (added 2026-10-14)

* PIGD audit: `experiments/20261014T143819Z_pigd_measurement_audit` (script 30).
* Predictive checks M1/M2/M5: `experiments/20261014T144511Z_pigd_measurement_models`
  (script 31). None passes; all over-predict zeros (~0.24 vs 0.161). M5
  (beta-binomial) raises PIGD-slope BLUP skew to 1.40 with Gaussian latent
  effects but its precision went to ~7,550 (binomial), consistent with item
  heterogeneity that an item-sum model cannot represent.
* Proposal: `docs/cluster_phase/06_SKEWED_NULL_AMENDMENT2.md` (PROPOSED, not in force).
* 06_SKEWED_NULL_PREREG.md unchanged (SHA-256 1b01bb6a…).
