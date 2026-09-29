"""Recompute every Step 3 metric from the prediction rows, and reconcile.

Two aggregations are defensible for a leave-one-patient-out error, and the
Step 3 tables and the earlier paired comparisons happened to use different
ones. Both are computed here, side by side, because the choice changes one
conclusion and it is better to show that than to pick a winner quietly.

    pooled    mean over every held-out visit. A patient with thirteen visits
              counts thirteen times. This is what `gpr_loo.metrics_table`
              reports, and it matches the convention the project's hidden
              Markov model validation already used.

    patient   mean within a patient, then mean over patients. Every patient
              counts once regardless of how long they were followed.

Uncertainty is always clustered by patient: visits within a patient are
correlated, so resampling visits independently would understate it. The
bootstrap resamples *patients* and recomputes the chosen statistic inside each
replicate, so the interval belongs to the estimand being reported rather than
to a different one.

    python scripts/12_gpr_reconcile_metrics.py [--cohorts unfiltered,A4,D1,E2]

Writes outputs/step3_gpr/reconciled_metrics.csv and paired_comparisons.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from ppmi_subtype import gpr_loo  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

N_BOOT = 4000
SEED = 42

#: Comparisons the Step 3 questions actually turn on.
PAIRS = [
    ("K1", "K0", "does a flexible kernel beat the linear one"),
    ("K2", "K1", "does the linear kernel term add anything beside an RBF"),
    ("K3", "K1", "does Matern beat the squared exponential"),
    ("K0", "hbayes", "does the GP reproduce the growth model"),
    ("K1", "persistence", "does the GP beat the trivial rule"),
    ("K1", "patient_mean", "does the GP beat the patient's own average"),
]


def pooled_mae(df: pd.DataFrame) -> float:
    return float(df["ae"].mean())


def patient_mae(df: pd.DataFrame) -> float:
    return float(df.groupby("held_out_patient")["ae"].mean().mean())


def bootstrap_diff(wide: pd.DataFrame, a: str, b: str, how: str, rng):
    """Paired difference a - b, with a patient-cluster bootstrap CI.

    The same resampled patients are used for both models in every replicate,
    which is what makes the comparison paired; and the statistic recomputed
    inside the replicate is the one being reported, so the interval cannot
    belong to a different aggregation than the point estimate.
    """
    patients = wide.index.get_level_values(0).to_numpy()
    uniq = np.unique(patients)
    diff = (wide[a] - wide[b]).to_numpy()

    if how == "pooled":
        point = float(diff.mean())
        by_patient = {p: diff[patients == p] for p in uniq}
        draws = np.empty(N_BOOT)
        for k in range(N_BOOT):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            draws[k] = np.concatenate([by_patient[p] for p in pick]).mean()
    else:
        per = pd.Series(diff, index=patients).groupby(level=0).mean()
        point = float(per.mean())
        vals = per.to_numpy()
        idx = rng.integers(0, len(vals), (N_BOOT, len(vals)))
        draws = vals[idx].mean(axis=1)

    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", default="unfiltered,A4,D1,E2")
    args = ap.parse_args()

    cfg = load_config()
    base = cfg.path("outputs") / "step3_gpr"
    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]

    metric_rows, pair_rows = [], []

    for cohort in cohorts:
        path = base / f"gpr_predictions_{cohort}.csv"
        if not path.exists():
            print(f"{cohort:11} no prediction file — skipped")
            continue

        d = pd.read_csv(path)
        d["ae"] = (d["pred_mean"] - d["observed_ratio"]).abs()
        n_models = d["model"].nunique()
        n_points = len(d) // n_models

        print(f"\n{'=' * 78}\n{cohort.upper()}  "
              f"{d['held_out_patient'].nunique():,} patients · {n_points:,} points · "
              f"{n_models} models\n{'=' * 78}")

        # -------------------------------------------------- per-model metrics
        for model, g in d.groupby("model"):
            row = {"cohort": cohort, "model": model,
                   "n_patients": int(g["held_out_patient"].nunique()),
                   "n_points": int(len(g)),
                   "MAE_pooled": pooled_mae(g),
                   "MAE_patient": patient_mae(g)}
            row.update({k: v for k, v in gpr_loo.continuous_metrics(g).items()
                        if k in ("RMSE", "MedAE", "CRPS", "coverage95",
                                 "interval_width", "bias", "negative_mass")})
            row.update({k: v for k, v in gpr_loo.categorical_metrics(g).items()
                        if k in ("accuracy", "balanced_accuracy", "brier")})
            metric_rows.append(row)

        shown = pd.DataFrame([r for r in metric_rows if r["cohort"] == cohort])
        print(shown[["model", "MAE_pooled", "MAE_patient", "RMSE", "CRPS",
                     "coverage95", "accuracy"]]
              .sort_values("MAE_pooled").round(4).to_string(index=False))

        # ------------------------------------------------ paired comparisons
        wide = d.pivot_table(index=["held_out_patient", "target_index"],
                             columns="model", values="ae")
        missing = wide.isna().any(axis=1).sum()
        if missing:
            raise SystemExit(
                f"{cohort}: {missing} prediction points are not scored by every "
                f"model; a paired comparison would silently drop them"
            )

        print(f"\n  paired differences (negative = first model better), "
              f"{N_BOOT:,} patient-cluster bootstrap draws, seed {SEED}")
        print(f"  {'comparison':<24} {'pooled':>26}   {'patient-averaged':>26}")
        for a, b, question in PAIRS:
            if a not in wide or b not in wide:
                continue
            rng = np.random.default_rng(SEED)
            p_pt, p_lo, p_hi = bootstrap_diff(wide, a, b, "pooled", rng)
            rng = np.random.default_rng(SEED)
            q_pt, q_lo, q_hi = bootstrap_diff(wide, a, b, "patient", rng)

            def verdict(lo, hi):
                return "excludes 0" if (lo < 0) == (hi < 0) else "includes 0"

            agree = np.sign(p_pt) == np.sign(q_pt)
            print(f"  {a + ' - ' + b:<24} "
                  f"{p_pt:+.4f} [{p_lo:+.4f},{p_hi:+.4f}]   "
                  f"{q_pt:+.4f} [{q_lo:+.4f},{q_hi:+.4f}]"
                  f"{'' if agree else '   <-- SIGN DIFFERS'}")
            pair_rows.append({
                "cohort": cohort, "model_a": a, "model_b": b, "question": question,
                "diff_pooled": p_pt, "pooled_lo": p_lo, "pooled_hi": p_hi,
                "pooled_ci_excludes_zero": verdict(p_lo, p_hi) == "excludes 0",
                "diff_patient": q_pt, "patient_lo": q_lo, "patient_hi": q_hi,
                "patient_ci_excludes_zero": verdict(q_lo, q_hi) == "excludes 0",
                "aggregations_agree_in_sign": bool(agree),
                "n_boot": N_BOOT, "seed": SEED,
            })

    if not metric_rows:
        raise SystemExit("no cohorts had prediction files")

    pd.DataFrame(metric_rows).to_csv(base / "reconciled_metrics.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(base / "paired_comparisons.csv", index=False)
    print(f"\nwrote {base / 'reconciled_metrics.csv'}")
    print(f"wrote {base / 'paired_comparisons.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
