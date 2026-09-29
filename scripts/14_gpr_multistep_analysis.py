"""Analyse the multi-step forecasts by horizon.

One-step-ahead MAE answers "can the model predict the next visit". The question
a clinician would ask is "how far ahead is this useful", and that is what the
multi-step forecasts address: the first ceil(T/2) finite observations are
history, and everything after is forecast at the times those visits actually
occurred.

Two things the saved forecasts do not contain, both supplied here:

* **Baselines.** `run_multistep_fold` fits kernels only, so K0-K3 had nothing to
  be judged against. Persistence and the patient mean are reconstructed from the
  same histories. Both are leakage-safe by construction: they use only the first
  m observations, which is exactly what the GP saw.

* **A leakage check.** Reconstructing the baselines requires rebuilding each
  patient's history, which incidentally lets every saved forecast be verified
  against the trajectory it came from --- that the forecast origin really is the
  m-th visit, and that the target really is later than the origin.

    python scripts/14_gpr_multistep_analysis.py [--cohorts unfiltered,A4,D1,E2]

Writes outputs/step3_gpr/multistep_by_horizon.csv and multistep_paired.csv.
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

from ppmi_subtype import gpr_loo, trajectory as tj  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

N_BOOT = 4000
SEED = 42
HORIZON_BINS = [0, 1.05, 2.05, 3.05, 5.05, 30]


def add_baselines(steps: pd.DataFrame, cohort) -> pd.DataFrame:
    """Reconstruct persistence and patient-mean forecasts from the histories.

    Both see only the first m finite observations --- the same slice the GP
    conditioned on --- so neither can use information unavailable at the
    forecast origin. Their predictive SD comes from the spread of that patient's
    own history, which is also origin-available; where a patient has a single
    history point there is no spread, and the cohort-median SD is used instead
    so the row can still be scored on coverage.
    """
    hist = {}
    for patno, g in cohort.finite.groupby("PATNO", sort=True):
        g = g.sort_values("finite_index")
        hist[patno] = (g["years_from_baseline"].to_numpy(float),
                       g["ratio"].to_numpy(float))

    template = steps[steps["model"] == steps["model"].iloc[0]].copy()
    rows = []
    for _, r in template.iterrows():
        t, y = hist[r["held_out_patient"]]
        m = int(r["history_length"])
        y_hist = y[:m]

        # leakage check: the recorded origin must be the m-th visit, and the
        # target must come after it
        assert np.isclose(t[m - 1], r["t_last_history"]), "origin mismatch"
        assert r["t_target"] > r["t_last_history"], "target precedes its origin"
        assert np.isclose(y[int(r["target_index"])], r["observed_ratio"]), "target mismatch"

        sd = float(np.std(y_hist, ddof=1)) if len(y_hist) > 1 else np.nan
        for name, pred in (("persistence", float(y_hist[-1])),
                           ("patient_mean", float(y_hist.mean()))):
            rows.append({**r.to_dict(), "model": name, "pred_mean": pred,
                         "pred_sd": sd, "error": pred - r["observed_ratio"]})

    base = pd.DataFrame(rows)
    fallback = float(np.nanmedian(base["pred_sd"]))
    base["pred_sd"] = base["pred_sd"].fillna(fallback)
    return pd.concat([steps, base], ignore_index=True)


def paired_ci(wide, a, b, rng):
    patients = wide.index.get_level_values(0).to_numpy()
    uniq = np.unique(patients)
    diff = (wide[a] - wide[b]).to_numpy()
    point = float(diff.mean())
    groups = {p: diff[patients == p] for p in uniq}
    draws = np.empty(N_BOOT)
    for k in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        draws[k] = np.concatenate([groups[p] for p in pick]).mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", default="unfiltered,A4,D1,E2")
    args = ap.parse_args()

    cfg = load_config()
    out = cfg.path("outputs") / "step3_gpr"
    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    traj = tj.build_trajectories(panel, cfg)

    horizon_rows, paired_rows = [], []

    for name in [c.strip() for c in args.cohorts.split(",") if c.strip()]:
        path = out / f"trajectory_predictions_{name}.csv"
        if not path.exists():
            print(f"{name}: no multi-step file — skipped")
            continue

        cohort = tj.build_cohort(traj, cfg,
                                 criterion=None if name == "unfiltered" else name)
        steps = pd.read_csv(path)
        steps = add_baselines(steps, cohort)
        steps["ae"] = (steps["pred_mean"] - steps["observed_ratio"]).abs()

        n_models = steps["model"].nunique()
        per_model = len(steps) // n_models
        print(f"\n{'=' * 78}\n{name.upper()}  "
              f"{steps['held_out_patient'].nunique():,} patients · "
              f"{per_model:,} forecast points · {n_models} models"
              f"\n{'=' * 78}")
        print("  leakage checks on every forecast row: origin == m-th visit, "
              "target after origin, target value matches trajectory — all passed")

        # ------------------------------------------------------- by horizon
        steps["h_bin"] = pd.cut(steps["horizon_years"], HORIZON_BINS)
        print(f"\n  MAE by forecast horizon")
        pivot = steps.pivot_table(index="model", columns="h_bin",
                                  values="ae", aggfunc="mean", observed=True)
        counts = steps[steps.model == "K1"].groupby("h_bin", observed=True).size()
        print("  n per bin:", counts.to_dict())
        print(pivot.round(4).to_string())

        for (model, hb), g in steps.groupby(["model", "h_bin"], observed=True):
            horizon_rows.append({
                "cohort": name, "model": model, "horizon_bin": str(hb),
                "n": len(g), "n_patients": int(g["held_out_patient"].nunique()),
                "MAE": float(g["ae"].mean()),
                "RMSE": float(np.sqrt((g["error"] ** 2).mean())),
                "CRPS": float(gpr_loo.gaussian_crps(
                    g["observed_ratio"].to_numpy(), g["pred_mean"].to_numpy(),
                    g["pred_sd"].to_numpy()).mean()),
                "coverage95": float((
                    (g["observed_ratio"] >= g["pred_mean"] - 1.96 * g["pred_sd"]) &
                    (g["observed_ratio"] <= g["pred_mean"] + 1.96 * g["pred_sd"])
                ).mean()),
                "mean_pred_sd": float(g["pred_sd"].mean()),
            })

        # --------------------------------------------------- by step number
        print(f"\n  MAE by step number (1 = first forecast visit)")
        st = steps.pivot_table(index="model", columns="step", values="ae",
                               aggfunc="mean", observed=True)
        print(st.round(4).to_string())

        # -------------------------------------------- paired, whole horizon
        wide = steps.pivot_table(index=["held_out_patient", "target_index"],
                                 columns="model", values="ae")
        wide = wide.dropna()
        print(f"\n  paired differences over all {len(wide):,} forecast points "
              f"(negative = first better)")
        for a, b in [("K1", "K0"), ("K3", "K1"),
                     ("K1", "persistence"), ("K1", "patient_mean")]:
            if a not in wide or b not in wide:
                continue
            rng = np.random.default_rng(SEED)
            pt, lo, hi = paired_ci(wide, a, b, rng)
            excl = "excludes 0" if (lo < 0) == (hi < 0) else "includes 0"
            print(f"    {a + ' - ' + b:<22} {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]  {excl}")
            paired_rows.append({"cohort": name, "model_a": a, "model_b": b,
                                "n_points": len(wide), "diff": pt,
                                "lo": lo, "hi": hi,
                                "ci_excludes_zero": excl == "excludes 0",
                                "n_boot": N_BOOT, "seed": SEED})

        # ------------------------------ does the ranking change with horizon?
        print(f"\n  model ranking by horizon bin (best first)")
        for hb, g in steps.groupby("h_bin", observed=True):
            order = g.groupby("model")["ae"].mean().sort_values()
            print(f"    {str(hb):<14} {' < '.join(order.index[:4])}")

    pd.DataFrame(horizon_rows).to_csv(out / "multistep_by_horizon.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out / "multistep_paired.csv", index=False)
    print(f"\nwrote {out / 'multistep_by_horizon.csv'}")
    print(f"wrote {out / 'multistep_paired.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
