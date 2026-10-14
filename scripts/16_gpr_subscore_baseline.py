"""Smallest defensible subscore experiment: predict the two subscores directly.

The proposal in NEXT_RESEARCH_STEPS.md argues for modelling `tremor_score` and
`pigd_score` instead of their ratio. This runs the *smallest* version of that:
the same audited GP, the same LOPO protocol, the same kernel, on the raw
subscore scale.

Raw scale is deliberate. It isolates the effect of changing the **target** from
the effect of changing the **likelihood**. A transformed or bounded likelihood
is a separate question and a separate experiment; mixing them would leave any
difference unattributable.

What this can and cannot answer
-------------------------------
It **can** answer: are the subscores easier to predict than the ratio, and does
the expanded cohort behave like the restricted one?

It **cannot** answer: is a subscore model better than the ratio model? Those two
are fitted on different training sets (934 vs 821 patients) and scored on
different targets in different units. MAE on a subscore and MAE on a ratio are
not comparable numbers. The comparison that *is* legitimate is the derived
subtype, and it is reported separately.

    python scripts/16_gpr_subscore_baseline.py [--jobs 12] [--limit N]

Writes outputs/step3_gpr/subscore_predictions_{tremor,pigd}.csv and a summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from ppmi_subtype import gpr_loo, trajectory as tj  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

KERNEL = "K1"          # const + RBF: the best one-step kernel, and K2 is identical
N_BOOT = 4000
SEED = 42


def subscore_cohort(traj: pd.DataFrame, cfg, column: str):
    """A cohort whose modelled quantity is `column` rather than the ratio.

    `build_cohort` keys off `ratio` / `ratio_defined`, so substituting the
    subscore into those fields reuses the audited cohort, fold and leakage
    machinery unchanged. Nothing about the GP, the LOPO loop or the checks
    knows or cares that the quantity is a subscore.
    """
    t = traj.copy()
    t["ratio"] = t[column]
    t["ratio_defined"] = t[column].notna()
    return tj.build_cohort(t, cfg, criterion=None)


def boot_ci(values, patients, rng, n=N_BOOT):
    uniq = np.unique(patients)
    groups = {p: values[patients == p] for p in uniq}
    draws = np.empty(n)
    for k in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        draws[k] = np.concatenate([groups[p] for p in pick]).mean()
    return float(values.mean()), *np.percentile(draws, [2.5, 97.5])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    out = cfg.path("outputs") / "step3_gpr"
    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    traj = tj.build_trajectories(panel, cfg)
    # The trajectory table keeps only the modelled quantity; the subscores have
    # to be brought back from the panel, keyed on the same (patient, visit).
    traj = traj.merge(
        panel[["PATNO", "EVENT_ID", "tremor_score", "pigd_score"]].drop_duplicates(
            ["PATNO", "EVENT_ID"]),
        on=["PATNO", "EVENT_ID"], how="left", validate="one_to_one",
    )
    missing = traj[["tremor_score", "pigd_score"]].isna().all(axis=1).sum()
    if missing:
        raise SystemExit(f"{missing} trajectory rows have no subscores after the merge")
    s = cfg.subtype

    settings = {
        "n_restarts": int(cfg.gpr["n_restarts"]), "seed": int(cfg.seed),
        "max_iter": int(cfg.gpr["max_iter"]), "jitter": float(cfg.gpr["jitter"]),
        # thresholds are meaningless on a subscore; the subtype is derived later
        "td_threshold": float(s.td_threshold), "pigd_threshold": float(s.pigd_threshold),
        "hbayes_draws": 4000, "hbayes_burn_in": 1000, "hbayes_pred_draws": 400,
    }

    # the points the ratio model can also score, for the restricted comparison
    ratio_cohort = tj.build_cohort(traj, cfg, criterion=None)
    ratio_keys = set(
        map(tuple, tj.one_step_points(ratio_cohort)[["PATNO", "target_index"]].values)
    )
    print(f"ratio model scores {len(ratio_keys):,} points on "
          f"{ratio_cohort.n_patients:,} patients")

    summary = []
    for column, short in [("tremor_score", "tremor"), ("pigd_score", "pigd")]:
        cohort = subscore_cohort(traj, cfg, column)
        data = gpr_loo.prepare(cohort)
        patients = data.patients[:args.limit] if args.limit else data.patients

        print(f"\n{'=' * 78}\n{short.upper()}  {len(patients):,} folds · "
              f"{cohort.n_prediction_points:,} points\n{'=' * 78}")
        t0 = time.time()

        def progress(done, total, _t0=t0):
            if done % 200 == 0 or done == total:
                print(f"    {done:,}/{total:,}  {(time.time()-_t0)/done:.1f}s/fold",
                      flush=True)

        preds, fits, _ = gpr_loo.run_lopo(
            data, [KERNEL], settings, patients=patients,
            with_hbayes=False, with_baselines=True, multistep=False,
            jobs=args.jobs, progress=progress,
            checkpoint=(out / "checkpoints" / f"subscore_{short}") if not args.limit else None,
        )
        preds.to_csv(out / f"subscore_predictions_{short}.csv", index=False)
        fits.to_csv(out / f"subscore_fits_{short}.csv", index=False)
        print(f"    done in {(time.time()-t0)/60:.1f} min")

        failures = [c for c, ok in gpr_loo.check_predictions(preds, data, short)
                    if not ok]
        print(f"    consistency: {'all passed' if not failures else failures}")
        print(f"    converged: {int(fits.converged.sum())}/{len(fits)}")

        preds["ae"] = (preds.pred_mean - preds.observed_ratio).abs()
        preds["in_restricted"] = [
            (p, k) in ratio_keys
            for p, k in zip(preds.held_out_patient, preds.target_index)
        ]

        n_exp = int(len(preds[preds.model == KERNEL]))
        n_res = int(preds[(preds.model == KERNEL) & preds.in_restricted].shape[0])
        print(f"\n    expanded  {n_exp:,} points / "
              f"{preds.held_out_patient.nunique():,} patients")
        print(f"    restricted{n_res:>7,} points  (also scored by the ratio model)")
        print(f"    recovered {n_exp - n_res:>7,} points  (invisible to the ratio model)")

        print(f"\n    {'model':14} {'cohort':11} {'n':>6} {'MAE':>8} "
              f"{'95% CI':>20} {'negmass':>8}")
        for model in [KERNEL, "persistence", "patient_mean"]:
            g0 = preds[preds.model == model]
            for label, g in [("expanded", g0), ("restricted", g0[g0.in_restricted]),
                             ("recovered", g0[~g0.in_restricted])]:
                rng = np.random.default_rng(SEED)
                pt, lo, hi = boot_ci(g.ae.to_numpy(),
                                     g.held_out_patient.to_numpy(), rng)
                nm = float(g.negative_mass.mean())
                print(f"    {model:14} {label:11} {len(g):>6,} {pt:8.4f} "
                      f"[{lo:6.4f},{hi:6.4f}] {nm:8.4f}")
                summary.append({"subscore": short, "model": model, "cohort": label,
                                "n": len(g),
                                "n_patients": int(g.held_out_patient.nunique()),
                                "MAE": pt, "lo": lo, "hi": hi, "negative_mass": nm})

        # paired: does the GP beat persistence on this target?
        for label, sub in [("expanded", preds), ("restricted", preds[preds.in_restricted])]:
            w = sub.pivot_table(index=["held_out_patient", "target_index"],
                                columns="model", values="ae").dropna()
            rng = np.random.default_rng(SEED)
            diff = (w[KERNEL] - w["persistence"]).to_numpy()
            pat = w.index.get_level_values(0).to_numpy()
            pt, lo, hi = boot_ci(diff, pat, rng)
            excl = "excludes 0" if (lo < 0) == (hi < 0) else "includes 0"
            print(f"    {KERNEL} - persistence, {label:10} {pt:+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}]  {excl}")

    pd.DataFrame(summary).to_csv(out / "subscore_summary.csv", index=False)
    print(f"\nwrote {out / 'subscore_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
