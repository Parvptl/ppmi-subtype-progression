"""Proposal Stages 4 and 6 - posterior inference and longitudinal validation.

Stage 4 asks for the posterior distribution of the latent motor phenotype for
every patient at every visit, and posterior predictive inference at future
visits. Stage 6 asks for validation by *withholding future visits* during
training --- a different design from the patient-level split used so far ---
and for calibration, predictive accuracy and uncertainty to be assessed.

Both are done here. Every per-patient quantity is reported alongside its
reliability, because a patient-specific slope estimated from three visits is
largely the cohort average and saying so is part of the result.

    python scripts/07_longitudinal_cv.py [--quick]

Writes to outputs/validation/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from ppmi_subtype import hbayes  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

NAMES = hbayes.STATE_NAMES if hasattr(hbayes, "STATE_NAMES") else ["TD", "Indeterminate", "PIGD"]

lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def section(title: str) -> None:
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


def frame(df: pd.DataFrame, index: bool = False) -> str:
    return "    " + df.to_string(index=index).replace("\n", "\n    ")


def truncate_panel(panel: pd.DataFrame, keep_first: int) -> tuple:
    """Split each patient's own sequence into an early part and a future part.

    This is the design the proposal specifies for Stage 6: the model never sees
    a patient's later visits, but it does see their earlier ones, so the test
    is about extrapolating forward in time rather than generalising to new
    people.
    """
    df = panel[panel["ratio"].notna() & np.isfinite(panel["ratio"])]
    df = df.sort_values(["PATNO", "years_from_baseline"])
    early, future = [], []
    for patno, group in df.groupby("PATNO", sort=False):
        if len(group) <= keep_first:
            early.append(group)
            continue
        early.append(group.iloc[:keep_first])
        future.append(group.iloc[keep_first:])
    return pd.concat(early), (pd.concat(future) if future else pd.DataFrame())


def main() -> int:
    quick = "--quick" in sys.argv
    cfg = load_config()
    out_dir = cfg.path("outputs") / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_path = cfg.path("outputs") / "visit_panel.csv"
    if not panel_path.exists():
        print(f"{panel_path} not found; run scripts/01_build_panel.py first", file=sys.stderr)
        return 1
    panel = pd.read_csv(panel_path)
    n_draws = 800 if quick else 2000

    # =====================================================  STAGE 4
    section("STAGE 4 - POSTERIOR OF THE LATENT PHENOTYPE, PER PATIENT PER VISIT")
    data = hbayes.build_growth_data(panel)
    post = hbayes.fit_growth_model(data, n_draws=n_draws, burn_in=500,
                                   seed=cfg.seed, keep_random_effects=True)
    say(f"  {data.n_obs:,} visits from {data.n_patients} patients, "
        f"{post.n_draws} retained draws")
    say(f"  random effects retained: array of shape {post.random_effects.shape}")

    latent = hbayes.latent_trajectories(post, data)
    latent.to_csv(out_dir / "latent_trajectories.csv", index=False)
    say(f"\n  Latent phenotype for every patient at every visit -> "
        f"latent_trajectories.csv ({len(latent):,} rows)")
    say("\n  First few rows:")
    say(frame(latent.head(6).round(3)))

    params = hbayes.patient_parameters(post)
    rel = hbayes.reliability(post, data)
    params = params.merge(rel, on="PATNO")
    params.to_csv(out_dir / "patient_parameters.csv", index=False)

    say("\n  Per-patient baseline and progression rate, with reliability:")
    say(frame(params.head(6).round(3)))

    section("STAGE 4b - HOW MUCH OF EACH ESTIMATE IS THE PATIENT'S OWN DATA?")
    say("  Reliability = 1 - Var_posterior(b) / tau^2.  0 = the population mean")
    say("  with the patient contributing nothing, 1 = determined by the patient.\n")
    summary = params[["reliability_baseline", "reliability_slope"]].describe(
        percentiles=[.1, .25, .5, .75, .9])
    say(frame(summary.round(3), index=True))
    for thr in (0.1, 0.25, 0.5, 0.8):
        say(f"    slope reliability > {thr:.2f}: {(params.reliability_slope > thr).sum():4d} "
            f"patients ({(params.reliability_slope > thr).mean():5.1%})")
    say("")
    by_n = params.groupby(params["n_visits"].clip(upper=10)).agg(
        patients=("PATNO", "size"), median_span=("span_years", "median"),
        slope_rel=("reliability_slope", "median"),
        baseline_rel=("reliability_baseline", "median"))
    say(frame(by_n.round(3), index=True))
    say("\n  The baseline level is well identified for essentially everyone; the")
    say("  progression rate is not. Any per-patient slope in the file above should")
    say("  be read together with its reliability column.")

    section("STAGE 4c - POSTERIOR PREDICTIVE AT FUTURE VISITS")
    well = params.sort_values("reliability_slope", ascending=False).iloc[0]
    poorly = params.sort_values("reliability_slope").iloc[0]
    for label, row in [("best-observed patient", well), ("typical patient", poorly)]:
        pred = hbayes.posterior_predictive(post, data, row["PATNO"],
                                           horizons=tuple(cfg.model["horizons"]))
        say(f"\n  {label}: PATNO {int(row['PATNO'])}, {int(row['n_visits'])} visits, "
            f"span {row['span_years']:.1f} yr, slope reliability {row['reliability_slope']:.3f}")
        say(frame(pred.round(3)))
    say("\n  The credible intervals are wide, and honestly so: they carry the")
    say("  measurement noise that the identifiability problem prevents us")
    say("  separating out.")

    # =====================================================  STAGE 6
    section("STAGE 6 - LONGITUDINAL CROSS-VALIDATION (withhold future visits)")
    say("  Train on each patient's first k visits, predict everything after.")
    say("  Unlike the patient-level split, the model has seen this patient before;")
    say("  the test is whether it extrapolates forward correctly.\n")

    rows = []
    for keep in (1, 2, 3):
        early, future = truncate_panel(panel, keep)
        if future.empty:
            continue
        d_early = hbayes.build_growth_data(early, min_visits=1)
        p_early = hbayes.fit_growth_model(d_early, n_draws=n_draws, burn_in=500,
                                          seed=cfg.seed, keep_random_effects=True)
        by_patient = {p: g for p, g in early.groupby("PATNO")}
        probs, truth, resid, inside = [], [], [], []
        for patno, group in future.groupby("PATNO"):
            hist = by_patient.get(patno)
            if hist is None or len(hist) == 0:
                continue
            th = hist["years_from_baseline"].to_numpy()
            yh = hist["ratio"].to_numpy()
            for _, visit in group.iterrows():
                draws = hbayes.predict_ratio(p_early, th, yh,
                                             float(visit["years_from_baseline"]),
                                             seed=int(patno) % 9999, max_draws=300)
                p = hbayes.label_probabilities(draws)
                probs.append([p[s] for s in NAMES])
                truth.append(visit["subtype_name"])
                resid.append(visit["ratio"] - draws.mean())
                lo, hi = np.quantile(draws, [0.025, 0.975])
                inside.append(lo <= visit["ratio"] <= hi)

        P = np.clip(np.array(probs), 1e-6, None)
        P = P / P.sum(axis=1, keepdims=True)
        j = np.array([NAMES.index(x) for x in truth])
        rows.append({
            "train_visits": keep,
            "n_future_visits": len(j),
            "loglik_per_visit": float(np.log(P[np.arange(len(j)), j]).mean()),
            "accuracy": float((P.argmax(axis=1) == j).mean()),
            "rmse_ratio": float(np.sqrt(np.mean(np.square(resid)))),
            "coverage_95": float(np.mean(inside)),
        })
        say(f"    trained on first {keep} visit(s): {len(j):,} future visits predicted")

    cv = pd.DataFrame(rows)
    cv.to_csv(out_dir / "longitudinal_cv.csv", index=False)
    say("")
    say(frame(cv.round(4)))
    say("\n  coverage_95 is the share of future observations falling inside the 95%")
    say("  predictive interval. It should be close to 0.95; below means the model")
    say("  is overconfident, above means the intervals are wider than they need to be.")

    section("STAGE 6b - CALIBRATION OF THE GROWTH MODEL")
    say("  This was reported for the Markov models but never for the growth model.\n")
    early, future = truncate_panel(panel, 2)
    d_early = hbayes.build_growth_data(early, min_visits=1)
    p_early = hbayes.fit_growth_model(d_early, n_draws=n_draws, burn_in=500, seed=cfg.seed)
    by_patient = {p: g for p, g in early.groupby("PATNO")}
    recs = []
    for patno, group in future.groupby("PATNO"):
        hist = by_patient.get(patno)
        if hist is None or len(hist) == 0:
            continue
        th, yh = hist["years_from_baseline"].to_numpy(), hist["ratio"].to_numpy()
        for _, visit in group.iterrows():
            draws = hbayes.predict_ratio(p_early, th, yh,
                                         float(visit["years_from_baseline"]),
                                         seed=int(patno) % 9999, max_draws=300)
            recs.append({"p_pigd": hbayes.label_probabilities(draws)["PIGD"],
                         "observed": float(visit["subtype_name"] == "PIGD")})
    cal_frame = pd.DataFrame(recs)
    edges = np.unique(np.quantile(cal_frame["p_pigd"], np.linspace(0, 1, 6)))
    cal_frame["bin"] = pd.cut(cal_frame["p_pigd"], bins=edges, include_lowest=True)
    cal = cal_frame.groupby("bin", observed=True).agg(
        n=("observed", "size"), mean_predicted=("p_pigd", "mean"),
        observed_rate=("observed", "mean")).reset_index()
    cal.to_csv(out_dir / "calibration_growth_model.csv", index=False)
    say(frame(cal.round(3)))
    gap = float((cal["mean_predicted"] - cal["observed_rate"]).abs().mean())
    say(f"\n  Mean absolute calibration gap: {gap:.3f}")

    section("SUMMARY")
    say("  Stage 4 complete: latent_trajectories.csv holds the posterior of the")
    say("  latent phenotype for every patient at every visit; patient_parameters.csv")
    say("  holds each patient's baseline and slope with credible intervals and the")
    say("  reliability of each.")
    say("  Stage 6 complete: longitudinal cross-validation withholding future")
    say("  visits, with accuracy, calibration and interval coverage.")
    best = cv.sort_values("loglik_per_visit", ascending=False).iloc[0]
    say(f"\n  Best configuration: train on first {int(best['train_visits'])} visits, "
        f"log-lik {best['loglik_per_visit']:.4f}, coverage {best['coverage_95']:.3f}")

    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_dir.relative_to(cfg.root)}/report.txt and "
          f"{len(list(out_dir.glob('*.csv')))} CSV tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
