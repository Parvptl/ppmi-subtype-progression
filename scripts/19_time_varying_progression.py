"""Direction 2: does the population progression rate change over follow-up?

From the midterm plan, verbatim: "Every model here assumes the rate is constant
over follow-up. We want to compare early, middle and late follow-up directly.
This is a hypothesis to test - we have not established it."

The experiment is deliberately narrow. The constant-rate model is Step 1's joint
linear mixed model; the time-varying model is the same thing with a linear
spline in place of the straight line. Nothing else differs, so a difference
between them is attributable to the mean basis and to nothing else.

Three things this script is careful about
-----------------------------------------
*Composition.* Periods contain different patients -- 934 reach the early period
and 273 the late one -- so comparing period means confounds "the rate changed"
with "the patients who got that far are different". Every fixed-effect result is
reported alongside a within-patient paired control.

*Fixed versus random.* A patient-specific slope says patients differ, not that
the population rate changes. The random-slope models are controls on the
fixed-effect result, never evidence for it.

*Knots are pre-specified*, listed in `timevarying.KNOT_SETS`, and never tuned.

    python scripts/19_time_varying_progression.py [--jobs 12] [--limit N]
                                                  [--skip-lopo]

Reads   outputs/visit_panel.csv
Writes  outputs/time_varying/   (never outputs/joint_subscore{,_gp}/)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from ppmi_subtype import subtype as st, timevarying as TV, trajectory as tj  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

PRIMARY_KNOTS = TV.KNOT_SETS["primary_2_5"]

#: Held-out arms. "constant" is Step 1's model; "spline" differs only in the
#: mean basis. The random-slope pair is the control that separates "patients
#: differ" from "the population rate changes".
ARMS = [
    ("lmm_constant", (), False),
    ("lmm_spline_2_5", PRIMARY_KNOTS, False),
]
# The random-slope pair is fitted on the full cohort in Part 1, where it serves
# its purpose as a control on the *inference*. It is left out of the held-out
# comparison: at ~32s per fit against ~4s it would add roughly 40 minutes per
# arm over 934 folds, and the predictive question asked here is specifically
# constant mean versus time-varying mean.
BASELINES = ["persistence", "patient_mean"]
N_BOOT = 4000
SEED = 42
_STATE: dict = {}


def build_frame(cfg) -> pd.DataFrame:
    """The Step 1/2 cohort, unchanged. No ratio, no PIGD == 0 removed."""
    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    traj = tj.build_trajectories(panel, cfg)
    tj.check_ordering(traj)
    frame = traj.merge(
        panel[["PATNO", "EVENT_ID", *TV.OUTCOMES]].drop_duplicates(
            ["PATNO", "EVENT_ID"]),
        on=["PATNO", "EVENT_ID"], how="left", validate="one_to_one")
    if frame[list(TV.OUTCOMES)].isna().any().any():
        raise SystemExit("a visit is missing a subscore after the merge")
    return frame


def describe_periods(frame: pd.DataFrame, knots: tuple) -> pd.DataFrame:
    """Who contributes to each period, and whether the late one is too sparse."""
    labels = TV.period_labels(knots)
    frame = frame.assign(period=pd.cut(frame.years_from_baseline,
                                       [0.0, *knots, np.inf], labels=labels,
                                       right=False))
    rows = []
    for label, g in frame.groupby("period", observed=True):
        n_pairs = int(sum(max(len(h) - 1, 0) for _, h in g.groupby("PATNO")))
        rows.append({
            "period": label, "visits": len(g),
            "patients": int(g.PATNO.nunique()),
            "within_period_visit_pairs": n_pairs,
            "median_time": float(g.years_from_baseline.median()),
            "pigd_zero_visits": int((g[TV.OUTCOMES[1]] == 0).sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Leave-one-patient-out
# ---------------------------------------------------------------------------
def _init_worker(data, arms):
    _STATE.update(data=data, arms=arms)


def _run_fold(patno):
    data = _STATE["data"]
    train = TV.subset(data, [p for p in data.patients if p != patno])
    t_all, Y_all = data.times[patno], data.responses[patno]

    preds, fits = [], []
    first_fit = None
    for label, knots, rs in _STATE["arms"]:
        f = TV.fit(train, knots=knots, random_slopes=rs)
        if first_fit is None:
            first_fit = f          # reused for the baselines' fallback SD
        row = {"model": label, "held_out_patient": patno,
               "converged": f.converged, "n_train_patients": f.n_patients,
               "neg2ll": f.neg2ll}
        for name, v in zip(TV.coefficient_names(knots), f.beta):
            row[name] = float(v)
        fits.append(row)

        for k in range(1, len(t_all)):
            mean, sd = TV.predict(f, t_all[:k], Y_all[:k], float(t_all[k]))
            preds.append({
                "held_out_patient": patno, "model": label,
                "target_index": k, "history_length": k,
                "t_last_history": float(t_all[k - 1]), "t_target": float(t_all[k]),
                "obs_tremor": float(Y_all[k, 0]), "obs_pigd": float(Y_all[k, 1]),
                "pred_tremor": float(mean[0]), "pred_pigd": float(mean[1]),
                "sd_tremor": float(sd[0]), "sd_pigd": float(sd[1]),
                "n_train_patients": f.n_patients, "train_converged": f.converged,
            })

    f0 = first_fit
    for k in range(1, len(t_all)):
        for name in BASELINES:
            mean, sd = TV.baseline_predict(name, t_all[:k], Y_all[:k],
                                           float(t_all[k]), f0)
            preds.append({
                "held_out_patient": patno, "model": name,
                "target_index": k, "history_length": k,
                "t_last_history": float(t_all[k - 1]), "t_target": float(t_all[k]),
                "obs_tremor": float(Y_all[k, 0]), "obs_pigd": float(Y_all[k, 1]),
                "pred_tremor": float(mean[0]), "pred_pigd": float(mean[1]),
                "sd_tremor": float(sd[0]), "sd_pigd": float(sd[1]),
                "n_train_patients": f0.n_patients, "train_converged": True,
            })
    return preds, fits


def check_predictions(preds: pd.DataFrame, data, scored: list) -> list:
    out = []
    models = sorted(preds.model.unique())
    g = preds[preds.model == ARMS[0][0]]
    expected = sum(len(data.times[p]) - 1 for p in scored)
    out.append(("held-out patient never in training",
                bool((preds.n_train_patients == data.n_patients - 1).all())))
    out.append(("history strictly before the target",
                bool((preds.t_last_history < preds.t_target).all())))
    out.append(("history length equals target index",
                bool((preds.history_length == preds.target_index).all())))
    out.append(("same cohort as Steps 1-2 (934 patients)", data.n_patients == 934))
    out.append(("prediction count equals transitions", len(g) == expected))
    out.append(("target times match the trajectory", all(
        np.allclose(np.sort(s.t_target.to_numpy()), np.sort(data.times[p][1:]))
        for p, s in g.groupby("held_out_patient"))))
    out.append(("observed scores match the trajectory", all(
        np.allclose(s.sort_values("target_index").obs_tremor.to_numpy(),
                    data.responses[p][1:, 0])
        for p, s in g.groupby("held_out_patient"))))
    out.append((f"PIGD == 0 points retained ({int((g.obs_pigd == 0).sum())} scored)",
                int((g.obs_pigd == 0).sum()) > 0))
    out.append(("every model scores identical points",
                all(int((preds.model == m).sum()) == len(g) for m in models)))
    out.append(("knots are fixed constants, not fold-estimated", True))
    out.append(("every fold converged", bool(preds.train_converged.all())))
    return out


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------
def boot_ci(values, patients, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(patients)
    groups = {p: values[patients == p] for p in uniq}
    draws = np.empty(n)
    for k in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        draws[k] = np.concatenate([groups[p] for p in pick]).mean()
    return float(values.mean()), *np.percentile(draws, [2.5, 97.5])


def continuous_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import norm
    rows = []
    for model, g in preds.groupby("model"):
        pat = g.held_out_patient.to_numpy()
        for short, o, p, s in [("tremor", "obs_tremor", "pred_tremor", "sd_tremor"),
                               ("pigd", "obs_pigd", "pred_pigd", "sd_pigd")]:
            err = (g[p] - g[o]).to_numpy()
            mae, lo, hi = boot_ci(np.abs(err), pat)
            z = err / g[s].to_numpy()
            rows.append({
                "model": model, "outcome": short, "n": len(g),
                "MAE": mae, "MAE_lo": lo, "MAE_hi": hi,
                "RMSE": float(np.sqrt((err ** 2).mean())),
                "bias": float(err.mean()),
                "coverage95": float((np.abs(z) <= 1.959963984540054).mean()),
                "mean_negative_mass": float(
                    norm.cdf(0.0, loc=g[p].to_numpy(), scale=g[s].to_numpy()).mean()),
                "frac_pred_below_zero": float((g[p] < 0).mean()),
            })
    return pd.DataFrame(rows)


def paired(preds: pd.DataFrame, reference: str) -> pd.DataFrame:
    rows = []
    key = ["held_out_patient", "target_index"]
    for short, o, p in [("tremor", "obs_tremor", "pred_tremor"),
                        ("pigd", "obs_pigd", "pred_pigd")]:
        w = preds.assign(ae=(preds[p] - preds[o]).abs()).pivot_table(
            index=key, columns="model", values="ae").dropna()
        if reference not in w:
            continue
        pat = w.index.get_level_values(0).to_numpy()
        for model in w.columns:
            if model == reference:
                continue
            diff = (w[model] - w[reference]).to_numpy()
            pt, lo, hi = boot_ci(diff, pat)
            rows.append({"outcome": short, "comparison": f"{model} - {reference}",
                         "delta_MAE": pt, "lo": lo, "hi": hi,
                         "excludes_zero": (lo < 0) == (hi < 0), "n": len(diff)})
    return pd.DataFrame(rows)


def subtype_table(preds: pd.DataFrame, cfg) -> tuple:
    d = preds.copy()
    d["pred_subtype"] = st.classify(d.pred_tremor.clip(lower=0.0),
                                    d.pred_pigd.clip(lower=0.0),
                                    cfg.subtype).map(cfg.subtype.label_names)
    d["obs_subtype"] = st.classify(d.obs_tremor, d.obs_pigd,
                                   cfg.subtype).map(cfg.subtype.label_names)
    classes = ["TD", "Indeterminate", "PIGD"]
    rows = []
    for model, g in d.groupby("model"):
        correct = (g.pred_subtype == g.obs_subtype).to_numpy().astype(float)
        acc, lo, hi = boot_ci(correct, g.held_out_patient.to_numpy())
        rec = {c: float((g.loc[g.obs_subtype == c, "pred_subtype"] == c).mean())
               if (g.obs_subtype == c).any() else np.nan for c in classes}
        rows.append({"model": model, "n": len(g), "accuracy": acc,
                     "accuracy_lo": lo, "accuracy_hi": hi,
                     "balanced_accuracy": float(np.nanmean(list(rec.values()))),
                     **{f"recall_{c}": rec[c] for c in classes},
                     **{f"n_pred_{c}": int((g.pred_subtype == c).sum())
                        for c in classes}})
    return pd.DataFrame(rows), d


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-lopo", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    out = cfg.path("outputs") / "time_varying"
    (out / "diagnostics").mkdir(parents=True, exist_ok=True)
    suffix = "_partial" if args.limit else ""

    frame = build_frame(cfg)
    data = TV.pack(frame)
    print(f"cohort {data.n_patients:,} patients, {len(frame):,} visits, "
          f"{data.n_visits - data.n_patients:,} prediction points, "
          f"{int((frame[TV.OUTCOMES[1]] == 0).sum()):,} visits with PIGD == 0")

    # ------------------------------------------------- Part 0: composition
    print("\n" + "=" * 78 + "\nFOLLOW-UP COMPOSITION (primary knots 2, 5)\n" + "=" * 78)
    comp = describe_periods(frame, PRIMARY_KNOTS)
    comp.to_csv(out / "period_composition.csv", index=False)
    print("  " + comp.to_string(index=False).replace("\n", "\n  "))

    # ------------------------------- Part 1: inference on the population rate
    print("\n" + "=" * 78 + "\nPART 1  POPULATION PROGRESSION RATE\n" + "=" * 78)
    fits = {}
    summary = []
    for name, knots in TV.KNOT_SETS.items():
        for rs in (False, True):
            t0 = time.time()
            f = TV.fit(data, knots=knots, random_slopes=rs)
            tag = f"{name}{'_randslope' if rs else ''}"
            fits[tag] = f
            summary.append({"spec": tag, "knots": str(knots), "random_slopes": rs,
                            "neg2ll": f.neg2ll, "n_fixed": len(f.beta),
                            "converged": f.converged, "seconds": time.time() - t0})
            print(f"  {tag:28} -2logL {f.neg2ll:10.3f}  fixed {len(f.beta):2}  "
                  f"conv {f.converged}  {time.time() - t0:5.1f}s")
    pd.DataFrame(summary).to_csv(out / "model_summary.csv", index=False)

    primary = fits["primary_2_5"]
    slopes = primary.period_slopes()
    contrasts = primary.slope_contrasts()
    slopes.to_csv(out / "period_slopes.csv", index=False)
    contrasts.to_csv(out / "slope_contrasts.csv", index=False)
    print("\n  ESTIMATED SLOPE PER PERIOD (knots 2, 5; random intercepts)")
    print("    " + slopes.round(5).to_string(index=False).replace("\n", "\n    "))
    print("\n  PAIRWISE SLOPE DIFFERENCES -- the hypothesis test")
    print("    " + contrasts.round(5).to_string(index=False).replace("\n", "\n    "))

    print("\n  SAME CONTRASTS WITH PATIENT-SPECIFIC SLOPES ALLOWED (control)")
    rc = fits["primary_2_5_randslope"].slope_contrasts()
    rc.to_csv(out / "slope_contrasts_randslope.csv", index=False)
    print("    " + rc.round(5).to_string(index=False).replace("\n", "\n    "))

    # ------------------------------------ Part 1b: composition control
    print("\n" + "=" * 78)
    print("WITHIN-PATIENT CONTROL  (same patients in both periods)")
    print("=" * 78)
    rates = TV.within_patient_rates(frame, PRIMARY_KNOTS)
    labels = TV.period_labels(PRIMARY_KNOTS)
    within = pd.concat([
        TV.paired_period_change(rates, labels[a], labels[b])
        for a in range(len(labels)) for b in range(a + 1, len(labels))
    ], ignore_index=True)
    within.to_csv(out / "within_patient_rate_change.csv", index=False)
    print("  " + within.round(5).to_string(index=False).replace("\n", "\n  "))

    # ------------------------------------------- Part 1c: knot robustness
    print("\n" + "=" * 78 + "\nROBUSTNESS ACROSS PRE-SPECIFIED KNOTS\n" + "=" * 78)
    rob = []
    for name, knots in TV.KNOT_SETS.items():
        if not knots:
            continue
        c = fits[name].slope_contrasts()
        c.insert(0, "knot_set", name)
        rob.append(c)
        w = TV.within_patient_rates(frame, knots)
        lab = TV.period_labels(knots)
        pc = TV.paired_period_change(w, lab[0], lab[-1])
        pc.insert(0, "knot_set", name)
        pc.insert(1, "source", "within_patient")
        rob.append(pc.rename(columns={"delta_rate": "difference",
                                      "ci_low": "ci_low", "ci_high": "ci_high"}))
    robust = pd.concat(rob, ignore_index=True)
    robust.to_csv(out / "robustness_knots.csv", index=False)
    for name, knots in TV.KNOT_SETS.items():
        if not knots:
            continue
        c = fits[name].slope_contrasts()
        last = TV.period_labels(knots)[-1] + " - " + TV.period_labels(knots)[0]
        sel = c[c.contrast == last]
        for _, r in sel.iterrows():
            print(f"  {name:12} {r.outcome:7} {r.contrast:16} "
                  f"{r.difference:+.5f} [{r.ci_low:+.5f},{r.ci_high:+.5f}]  "
                  f"{'excludes 0' if r.excludes_zero else 'includes 0'}")

    if args.skip_lopo:
        return 0

    # ------------------------------------------ Part 2: held-out prediction
    scored = data.patients[:args.limit] if args.limit else data.patients
    print("\n" + "=" * 78)
    print(f"PART 2  LEAVE-ONE-PATIENT-OUT ({len(scored):,} folds x {len(ARMS)} arms)")
    print("=" * 78)
    t0 = time.time()
    all_preds, all_fits = [], []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker,
                             initargs=(data, ARMS)) as pool:
        for done, (p, f) in enumerate(pool.map(_run_fold, scored, chunksize=1), 1):
            all_preds.extend(p)
            all_fits.extend(f)
            if done % 50 == 0 or done == len(scored):
                el = time.time() - t0
                print(f"    {done:,}/{len(scored):,}  {el / done:.2f}s/fold  "
                      f"eta {(len(scored) - done) * el / done / 60:.1f} min", flush=True)

    preds = pd.DataFrame(all_preds)
    preds.to_csv(out / f"predictions{suffix}.csv", index=False)
    pd.DataFrame(all_fits).to_csv(out / f"fold_fits{suffix}.csv", index=False)
    print(f"  total {(time.time() - t0) / 60:.1f} min")

    print("\n  LEAKAGE AND CONSISTENCY")
    for name, ok in check_predictions(preds, data, scored):
        print(f"    [{'PASS' if ok else 'FAIL'}]  {name}")

    cont = continuous_metrics(preds)
    cont.to_csv(out / f"metrics{suffix}.csv", index=False)
    for short in ("tremor", "pigd"):
        print(f"\n  {short.upper()}  (primary metric: MAE)")
        print("    " + cont[cont.outcome == short].sort_values("MAE")
              [["model", "MAE", "MAE_lo", "MAE_hi", "RMSE", "bias", "coverage95",
                "mean_negative_mass"]]
              .round(4).to_string(index=False).replace("\n", "\n    "))

    vs_const = paired(preds[preds.model.isin(["lmm_constant", "lmm_spline_2_5"])],
                      "lmm_constant")
    vs_pers = paired(preds, "persistence")
    pd.concat([vs_const, vs_pers]).to_csv(out / f"paired{suffix}.csv", index=False)
    print("\n  TIME-VARYING minus CONSTANT  (the predictive test)")
    print("    " + vs_const.round(5).to_string(index=False).replace("\n", "\n    "))
    print("\n  ALL MODELS minus PERSISTENCE")
    print("    " + vs_pers.round(5).to_string(index=False).replace("\n", "\n    "))

    smet, derived = subtype_table(preds, cfg)
    smet.to_csv(out / f"subtype_metrics{suffix}.csv", index=False)
    derived.to_csv(out / f"predictions_with_subtype{suffix}.csv", index=False)
    print("\n  DERIVED SUBTYPE (secondary)")
    print("    " + smet.round(4).to_string(index=False).replace("\n", "\n    "))
    cm = pd.crosstab(derived[derived.model == "lmm_spline_2_5"].pred_subtype,
                     derived[derived.model == "lmm_spline_2_5"].obs_subtype)
    cm.to_csv(out / f"confusion_matrix{suffix}.csv")
    print("\n  confusion matrix, lmm_spline_2_5 (rows predicted)")
    print("    " + cm.to_string().replace("\n", "\n    "))

    zero = preds[preds.obs_pigd == 0]
    zrows = [{"model": m, "n": len(g),
              "tremor_MAE": float((g.pred_tremor - g.obs_tremor).abs().mean()),
              "pigd_MAE": float(g.pred_pigd.abs().mean()),
              "pigd_pred_mean": float(g.pred_pigd.mean()),
              "frac_below_0.1": float((g.pred_pigd < 0.1).mean())}
             for m, g in zero.groupby("model")]
    zdf = pd.DataFrame(zrows).sort_values("pigd_MAE")
    zdf.to_csv(out / f"pigd_zero_analysis{suffix}.csv", index=False)
    print(f"\n  PIGD == 0 POINTS ({int(len(zero) / preds.model.nunique()):,})")
    print("    " + zdf.round(4).to_string(index=False).replace("\n", "\n    "))

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
