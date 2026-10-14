"""Audit the multi-step evaluation, and separate horizon from patient composition.

The headline multi-step finding was that the GP's advantage over the baselines
widens with forecast horizon. That comparison is between *bins*, and the bins
contain different patients: only patients with long follow-up appear in the 5+
year bin, and they are not a random subset. A between-bin comparison therefore
confounds "predictions get harder further out" with "the patients who reach
that distance are different".

This script reports both readings side by side:

    between-patient   every forecast point in a bin, as reported before
    within-patient    restricted to patients contributing to *both* bins of a
                      pair, paired inside the patient

The second holds composition fixed by construction. Where the two disagree, the
between-bin reading is the one contaminated by composition.

Also re-derives the reconstructed baselines and re-checks the forecast-origin
and target definitions against the trajectories, independently of
`14_gpr_multistep_analysis.py`.

    python scripts/15_gpr_multistep_audit.py [--cohorts unfiltered,A4,D1,E2]

Writes outputs/step3_gpr/multistep_audit_by_horizon.csv
   and outputs/step3_gpr/multistep_audit_within_patient.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from ppmi_subtype import trajectory as tj  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

N_BOOT = 4000
SEED = 42
BINS = [0, 1.05, 2.05, 3.05, 5.05, 30]
LABELS = ["0-1y", "1-2y", "2-3y", "3-5y", "5y+"]


def crps(y, mu, sd):
    sd = np.maximum(np.asarray(sd, float), 1e-12)
    z = (np.asarray(y, float) - np.asarray(mu, float)) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def boot_mean(values, patients, rng, n=N_BOOT):
    """Patient-clustered bootstrap of a mean over points."""
    uniq = np.unique(patients)
    groups = {p: values[patients == p] for p in uniq}
    draws = np.empty(n)
    for k in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        draws[k] = np.concatenate([groups[p] for p in pick]).mean()
    return float(values.mean()), *np.percentile(draws, [2.5, 97.5])


def rebuild_baselines(steps, cohort):
    """Reconstruct persistence and patient-mean, re-verifying every origin."""
    hist = {}
    for patno, g in cohort.finite.groupby("PATNO", sort=True):
        g = g.sort_values("finite_index")
        hist[patno] = (g["years_from_baseline"].to_numpy(float),
                       g["ratio"].to_numpy(float))

    tmpl = steps[steps["model"] == steps["model"].iloc[0]]
    bad = 0
    rows = []
    for _, r in tmpl.iterrows():
        t, y = hist[r["held_out_patient"]]
        m = int(r["history_length"])
        k = int(r["target_index"])
        # independent re-derivation of the three definitional claims
        if not (np.isclose(t[m - 1], r["t_last_history"])
                and np.isclose(y[k], r["observed_ratio"])
                and k >= m and r["t_target"] > r["t_last_history"]
                and m == int(np.ceil(len(y) / 2))):
            bad += 1
        h = y[:m]
        sd = float(np.std(h, ddof=1)) if len(h) > 1 else np.nan
        for name, pred in (("persistence", float(h[-1])),
                           ("patient_mean", float(h.mean()))):
            rows.append({**r.to_dict(), "model": name, "pred_mean": pred,
                         "pred_sd": sd})
    base = pd.DataFrame(rows)
    base["pred_sd"] = base["pred_sd"].fillna(float(np.nanmedian(base["pred_sd"])))
    return pd.concat([steps, base], ignore_index=True), bad, len(tmpl)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", default="unfiltered,A4,D1,E2")
    args = ap.parse_args()

    cfg = load_config()
    out = cfg.path("outputs") / "step3_gpr"
    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    traj = tj.build_trajectories(panel, cfg)

    horizon_rows, within_rows = [], []

    for name in [c.strip() for c in args.cohorts.split(",") if c.strip()]:
        path = out / f"trajectory_predictions_{name}.csv"
        if not path.exists():
            print(f"{name}: absent — skipped")
            continue
        cohort = tj.build_cohort(traj, cfg,
                                 criterion=None if name == "unfiltered" else name)
        steps = pd.read_csv(path)
        steps, bad, n_checked = rebuild_baselines(steps, cohort)

        steps["ae"] = (steps.pred_mean - steps.observed_ratio).abs()
        steps["se"] = (steps.pred_mean - steps.observed_ratio) ** 2
        steps["crps"] = crps(steps.observed_ratio, steps.pred_mean, steps.pred_sd)
        steps["bin"] = pd.cut(steps.horizon_years, BINS, labels=LABELS)

        print(f"\n{'=' * 78}\n{name.upper()}\n{'=' * 78}")
        print(f"  definitional re-check on {n_checked:,} forecast rows "
              f"(origin == ceil(T/2)-th visit, target after origin, target value "
              f"matches trajectory): {n_checked - bad} passed, {bad} failed")

        # ---------------------------------------------- between-patient view
        print(f"\n  BETWEEN-PATIENT: every point in the bin "
              f"(composition varies across bins)")
        counts = (steps[steps.model == "K1"].groupby("bin", observed=True)
                  .agg(n=("ae", "size"), pat=("held_out_patient", "nunique")))
        print("    n points / patients per bin: " + ", ".join(
            f"{b}: {r.n}/{r.pat}" for b, r in counts.iterrows()))
        for metric in ("ae", "se", "crps"):
            piv = steps.pivot_table(index="model", columns="bin", values=metric,
                                    aggfunc="mean", observed=True)
            print(f"\n    {'MAE' if metric=='ae' else metric.upper()} by horizon")
            print("      " + piv.round(4).to_string().replace("\n", "\n      "))

        for (model, b), g in steps.groupby(["model", "bin"], observed=True):
            rng = np.random.default_rng(SEED)
            pat = g.held_out_patient.to_numpy()
            for metric in ("ae", "se", "crps"):
                pt, lo, hi = boot_mean(g[metric].to_numpy(), pat, rng)
                horizon_rows.append({
                    "cohort": name, "model": model, "bin": str(b),
                    "metric": {"ae": "MAE", "se": "MSE", "crps": "CRPS"}[metric],
                    "value": pt, "lo": lo, "hi": hi,
                    "n_points": len(g),
                    "n_patients": int(g.held_out_patient.nunique()),
                })

        # ----------------------------------------------- within-patient view
        print(f"\n  WITHIN-PATIENT: adjacent bins, restricted to patients in both")
        print(f"    (composition held fixed; isolates the horizon effect)")
        for model in ["K1", "persistence"]:
            g = steps[steps.model == model]
            per = (g.groupby(["held_out_patient", "bin"], observed=True)["ae"]
                   .mean().unstack())
            line = []
            for i in range(len(LABELS) - 1):
                b0, b1 = LABELS[i], LABELS[i + 1]
                if b0 not in per or b1 not in per:
                    continue
                pair = per[[b0, b1]].dropna()
                if len(pair) < 15:
                    line.append(f"{b0}->{b1}: n={len(pair)} too few")
                    continue
                diff = (pair[b1] - pair[b0]).to_numpy()
                rng = np.random.default_rng(SEED)
                idx = rng.integers(0, len(diff), (N_BOOT, len(diff)))
                bs = diff[idx].mean(axis=1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                sig = "*" if (lo < 0) == (hi < 0) else " "
                line.append(f"{b0}->{b1}: {diff.mean():+.3f}[{lo:+.3f},{hi:+.3f}]{sig} n={len(pair)}")
                within_rows.append({
                    "cohort": name, "model": model, "from_bin": b0, "to_bin": b1,
                    "n_patients": len(pair), "delta_MAE": float(diff.mean()),
                    "lo": float(lo), "hi": float(hi),
                    "ci_excludes_zero": (lo < 0) == (hi < 0),
                })
            print(f"    {model:12} " + " | ".join(line))

    pd.DataFrame(horizon_rows).to_csv(out / "multistep_audit_by_horizon.csv", index=False)
    pd.DataFrame(within_rows).to_csv(out / "multistep_audit_within_patient.csv", index=False)
    print(f"\nwrote {out / 'multistep_audit_by_horizon.csv'}")
    print(f"wrote {out / 'multistep_audit_within_patient.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
