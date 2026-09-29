"""Held-out comparison of the fitted linear mean against a zero-mean GP.

The audit established that a fitted linear mean beats a zero mean by 473 log
marginal-likelihood units *in sample*. That is not the same claim as better
generalisation, and the audit recorded the held-out version as outstanding.
This closes it.

The comparison
--------------
The article this work follows uses ``m(x) = 0`` and puts trend structure in the
kernel. So the fair opposing arm is not "K1 without a mean" --- that would
delete the linear trend entirely rather than relocate it --- but a zero-mean GP
whose kernel carries constant *and* linear terms:

    fitted    m(t) = b0 + b1 t   with  const + linear + rbf     (K2)
    zero      m(t) = 0           with  const + linear + rbf     (K2, fit_mean=False)

Identical kernel, identical data, identical optimiser, identical restarts. The
only difference is whether the trend is a fitted point estimate or a
zero-centred prior marginalised through the kernel.

Both arms are fitted **inside the same fold**, so they share the training set,
the held-out targets and the histories exactly. That is what makes the paired
patient-level intervals valid.

    python scripts/13_gpr_mean_function_holdout.py [--cohorts unfiltered] [--jobs 12]

Writes outputs/step3_gpr/mean_function_holdout_{cohort}.csv and a summary.
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

#: The two arms. Same kernel both sides; only the mean specification differs.
ARMS = ["K2", ("K2_zeromean", "K2", False)]
N_BOOT = 4000
SEED = 42


def paired_ci(wide: pd.DataFrame, a: str, b: str, how: str, rng):
    """Paired difference a - b with a patient-cluster bootstrap interval."""
    patients = wide.index.get_level_values(0).to_numpy()
    uniq = np.unique(patients)
    diff = (wide[a] - wide[b]).to_numpy()

    if how == "pooled":
        point = float(diff.mean())
        groups = {p: diff[patients == p] for p in uniq}
        draws = np.empty(N_BOOT)
        for k in range(N_BOOT):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            draws[k] = np.concatenate([groups[p] for p in pick]).mean()
    else:
        per = pd.Series(diff, index=patients).groupby(level=0).mean()
        point = float(per.mean())
        vals = per.to_numpy()
        draws = vals[rng.integers(0, len(vals), (N_BOOT, len(vals)))].mean(axis=1)

    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", default="unfiltered")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    out = cfg.path("outputs") / "step3_gpr"
    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    traj = tj.build_trajectories(panel, cfg)
    s = cfg.subtype

    settings = {
        "n_restarts": int(cfg.gpr["n_restarts"]), "seed": int(cfg.seed),
        "max_iter": int(cfg.gpr["max_iter"]), "jitter": float(cfg.gpr["jitter"]),
        "td_threshold": float(s.td_threshold), "pigd_threshold": float(s.pigd_threshold),
        "hbayes_draws": 4000, "hbayes_burn_in": 1000, "hbayes_pred_draws": 400,
    }

    summary = []
    for name in [c.strip() for c in args.cohorts.split(",") if c.strip()]:
        cohort = tj.build_cohort(traj, cfg,
                                 criterion=None if name == "unfiltered" else name)
        data = gpr_loo.prepare(cohort)
        patients = data.patients[:args.limit] if args.limit else data.patients

        print(f"\n{'=' * 78}\n{name.upper()}  {len(patients):,} folds · "
              f"{cohort.n_prediction_points:,} prediction points\n{'=' * 78}")
        t0 = time.time()

        def progress(done, total, _t0=t0):
            if done % 100 == 0 or done == total:
                print(f"    {done:,}/{total:,}  {(time.time()-_t0)/done:.1f}s/fold",
                      flush=True)

        preds, fits, _ = gpr_loo.run_lopo(
            data, ARMS, settings, patients=patients,
            with_hbayes=False, with_baselines=False, multistep=False,
            jobs=args.jobs, progress=progress,
            checkpoint=(out / "checkpoints" / f"meanfn_{name}") if not args.limit else None,
        )
        preds.to_csv(out / f"mean_function_holdout_{name}.csv", index=False)
        fits.to_csv(out / f"mean_function_holdout_fits_{name}.csv", index=False)
        print(f"    done in {(time.time()-t0)/60:.1f} min")

        # ------------------------------------------------------------ checks
        failures = [c for c, ok in gpr_loo.check_predictions(preds, data, name)
                    if not ok]
        if failures and not args.limit:
            raise SystemExit(f"{name}: consistency checks failed: {failures}")
        print(f"    consistency checks: {'all passed' if not failures else failures}")

        conv = fits.groupby("kernel")["converged"].agg(["sum", "count"])
        for k, r in conv.iterrows():
            print(f"    {k:14} converged {int(r['sum'])}/{int(r['count'])} "
                  f"({r['sum']/r['count']:.1%})")

        # ----------------------------------------------------------- metrics
        preds["ae"] = (preds["pred_mean"] - preds["observed_ratio"]).abs()
        print()
        rows = []
        for model, g in preds.groupby("model"):
            m = gpr_loo.continuous_metrics(g)
            rows.append({
                "cohort": name, "model": model, "n_points": len(g),
                "n_patients": int(g["held_out_patient"].nunique()),
                "MAE_pooled": float(g["ae"].mean()),
                "MAE_patient": float(g.groupby("held_out_patient")["ae"].mean().mean()),
                "RMSE": m["RMSE"], "CRPS": m["CRPS"],
                "coverage95": m["coverage95"], "interval_width": m["interval_width"],
                "negative_mass": m["negative_mass"],
                "accuracy": gpr_loo.categorical_metrics(g)["accuracy"],
            })
        table = pd.DataFrame(rows)
        print(table[["model", "MAE_pooled", "MAE_patient", "RMSE", "CRPS",
                     "coverage95", "negative_mass"]].round(4).to_string(index=False))

        wide = preds.pivot_table(index=["held_out_patient", "target_index"],
                                 columns="model", values="ae")
        if wide.isna().any(axis=1).any():
            raise SystemExit(f"{name}: a point is missing from one arm")

        print(f"\n  paired difference, zero-mean minus fitted-mean "
              f"(positive = fitted mean better)")
        for how in ("pooled", "patient"):
            rng = np.random.default_rng(SEED)
            pt, lo, hi = paired_ci(wide, "K2_zeromean", "K2", how, rng)
            excl = "excludes 0" if (lo < 0) == (hi < 0) else "includes 0"
            print(f"    {how:<8} {pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]   {excl}")
            table.loc[table.model == "K2_zeromean", f"diff_{how}"] = pt
            table.loc[table.model == "K2_zeromean", f"diff_{how}_lo"] = lo
            table.loc[table.model == "K2_zeromean", f"diff_{how}_hi"] = hi
        summary.append(table)

    pd.concat(summary, ignore_index=True).to_csv(
        out / "mean_function_holdout_summary.csv", index=False)
    print(f"\nwrote {out / 'mean_function_holdout_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
