"""Proposal Stage 2 - exploratory longitudinal analysis of the TD/PIGD ratio.

The proposal asks for four things before any model is fitted:

    * typical progression trajectories
    * distribution of subtype transitions
    * variability within and across subjects
    * effect of unequal follow-up duration and missing visits

and says the step should "guide the selection of suitable prior distributions
and model assumptions".

The Weeks 3-4 analysis covered the second and fourth bullets, but on the
subtype *label*. This stage does the work on the continuous ratio, which is
what the growth model actually consumes. Several of the problems that later
surfaced during modelling are visible here, which is the point of doing it
first.

    python scripts/06_ratio_exploration.py

Writes to outputs/exploration/.
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


def main() -> int:
    cfg = load_config()
    out_dir = cfg.path("outputs") / "exploration"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_path = cfg.path("outputs") / "visit_panel.csv"
    if not panel_path.exists():
        print(f"{panel_path} not found; run scripts/01_build_panel.py first", file=sys.stderr)
        return 1
    panel = pd.read_csv(panel_path)

    labelled = panel[panel["subtype"].notna()]
    finite = labelled[labelled["ratio"].notna() & np.isfinite(labelled["ratio"])]

    # ------------------------------------------------ distribution
    section("1. DISTRIBUTION OF THE RATIO")
    say(f"  labelled visits            {len(labelled):,}")
    say(f"  with a finite ratio        {len(finite):,}  ({len(finite)/len(labelled):.1%})")
    say(f"  PIGD subscore is zero      {len(labelled)-len(finite):,}  (ratio undefined)")
    say("")
    desc = finite["ratio"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    say(frame(desc.round(3).to_frame().T))
    say(f"\n  skew {finite['ratio'].skew():.3f}   share exactly zero "
        f"{(finite['ratio'] == 0).mean():.3f}")
    say(f"  share inside the Indeterminate band (0.90-1.15): "
        f"{((finite['ratio'] >= 0.90) & (finite['ratio'] <= 1.15)).mean():.3f}")

    say("\n  Model assumption this bears on:")
    say("  The ratio is bounded below at zero, right-skewed, and has an atom at")
    say("  exactly zero (no tremor at all). A symmetric Normal observation model")
    say("  cannot represent that, and a log transform is blocked by the zeros.")
    say("  Either a shifted transform or a model of the two subscores is needed.")

    # ------------------------------------------------ trajectories
    section("2. TYPICAL PROGRESSION TRAJECTORIES")
    per = finite.groupby("PATNO").agg(
        n=("ratio", "size"),
        span=("years_from_baseline", lambda s: s.max() - s.min()),
        first=("ratio", "first"),
        last=("ratio", "last"),
    )
    say("  Per-patient ratio change, first visit to last:")
    say(frame((per["last"] - per["first"]).describe(percentiles=[.1, .25, .5, .75, .9])
              .round(3).to_frame().T))

    ols = per[per["n"] >= 3].copy()
    slopes = []
    for patno, group in finite[finite["PATNO"].isin(ols.index)].groupby("PATNO"):
        t = group["years_from_baseline"].to_numpy()
        y = group["ratio"].to_numpy()
        if t.max() - t.min() < 0.5:
            continue
        slopes.append({"PATNO": patno, "n": len(t), "span": t.max() - t.min(),
                       "ols_slope": np.polyfit(t, y, 1)[0]})
    slope_frame = pd.DataFrame(slopes)
    slope_frame.to_csv(out_dir / "per_patient_ols_slopes.csv", index=False)

    say(f"\n  Ordinary least-squares slope fitted separately to each of the "
        f"{len(slope_frame):,} patients")
    say("  with 3+ visits and at least 6 months of span:")
    say(frame(slope_frame["ols_slope"].describe(percentiles=[.1, .25, .5, .75, .9])
              .round(4).to_frame().T))
    say(f"\n  median {slope_frame['ols_slope'].median():+.4f} per year, "
        f"mean {slope_frame['ols_slope'].mean():+.4f}")
    say(f"  share of patients with a declining ratio: "
        f"{(slope_frame['ols_slope'] < 0).mean():.3f}")

    naive_sd = float(slope_frame["ols_slope"].std())
    say("\n  Prior this bears on:")
    say("  The naive spread of these slopes is inflated by measurement noise: an SD of")
    say(f"  {naive_sd:.3f} against the hierarchical model's tau1 of about 0.10, a factor of")
    say(f"  roughly {naive_sd/0.10:.0f}. A prior on tau1 must therefore not be centred on the")
    say("  raw spread, or it would encode noise as genuine between-patient variation.")

    # --------------------------------------- variance decomposition
    section("3. VARIABILITY WITHIN AND ACROSS SUBJECTS")
    multi = finite[finite["PATNO"].isin(per[per["n"] >= 2].index)]
    grand = multi["ratio"].mean()
    means = multi.groupby("PATNO")["ratio"].mean()
    counts = multi.groupby("PATNO")["ratio"].size()

    ss_between = float((counts * (means - grand) ** 2).sum())
    ss_within = float(((multi["ratio"] - multi["PATNO"].map(means)) ** 2).sum())
    df_between = len(means) - 1
    df_within = len(multi) - len(means)
    ms_between, ms_within = ss_between / df_between, ss_within / df_within
    n_bar = counts.mean()
    var_between = max((ms_between - ms_within) / n_bar, 0.0)

    table = pd.DataFrame([
        {"component": "between patients", "variance": var_between, "sd": np.sqrt(var_between)},
        {"component": "within patients", "variance": ms_within, "sd": np.sqrt(ms_within)},
    ])
    table["share"] = table["variance"] / table["variance"].sum()
    table.to_csv(out_dir / "variance_decomposition.csv", index=False)
    say(frame(table.round(4)))
    icc = var_between / (var_between + ms_within)
    say(f"\n  Intraclass correlation: {icc:.3f}")
    say(f"  So {icc:.0%} of the variance in the ratio is between patients and")
    say(f"  {1-icc:.0%} is within a patient over time.")

    say("\n  Model assumption this bears on:")
    say("  A within-patient SD of "
        f"{np.sqrt(ms_within):.2f} against an Indeterminate band 0.25 wide means")
    say("  the ratio moves several band-widths between visits for reasons unrelated")
    say("  to progression. Predicting band membership from a continuous model was")
    say("  always going to be hard, and this is visible before any model is fitted.")

    # ------------------------------------------ follow-up and gaps
    section("4. EFFECT OF UNEQUAL FOLLOW-UP AND MISSING VISITS")
    say("  Visits per patient (finite ratio only):")
    say(frame(per["n"].value_counts().sort_index().rename("patients").to_frame(), index=True))
    say(f"\n  median {per['n'].median():.0f} visits, median span {per['span'].median():.2f} years")

    say("\n  Estimated slope precision depends strongly on span. OLS slope SD by span:")
    slope_frame["span_band"] = pd.cut(slope_frame["span"], [0, 2, 4, 8, 99],
                                      labels=["<2y", "2-4y", "4-8y", ">8y"])
    by_span = slope_frame.groupby("span_band", observed=True).agg(
        patients=("ols_slope", "size"),
        slope_sd=("ols_slope", "std"),
        slope_median=("ols_slope", "median"),
    )
    say(frame(by_span.round(4), index=True))
    say("\n  The naive slope spread collapses as span grows, which is the signature")
    say("  of noise rather than of real heterogeneity: with a short span the fitted")
    say("  slope is mostly measurement error divided by a small time difference.")

    missing = labelled.groupby("YEAR").agg(
        labelled=("subtype", "size"),
        with_ratio=("ratio", lambda s: int(np.isfinite(s).sum())),
    )
    missing["ratio_share"] = missing["with_ratio"] / missing["labelled"]
    missing.to_csv(out_dir / "ratio_availability_by_year.csv")
    say("\n  Availability of a finite ratio by follow-up year:")
    say(frame(missing.head(13).round(3), index=True))

    # ------------------------------------------ autocorrelation
    section("5. IS THE DEVIATION FROM TREND CORRELATED IN TIME?")
    say("  Fitting each patient's own straight line and correlating consecutive")
    say("  residuals. A growth model assumes these are independent; a state-space")
    say("  model assumes they are correlated because process noise accumulates.\n")
    pairs = []
    for patno, group in finite.groupby("PATNO"):
        if len(group) < 4:
            continue
        t = group["years_from_baseline"].to_numpy()
        y = group["ratio"].to_numpy()
        if t.max() - t.min() < 1.0:
            continue
        resid = y - np.polyval(np.polyfit(t, y, 1), t)
        pairs.extend(zip(resid[:-1], resid[1:]))
    arr = np.array(pairs)
    r = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
    say(f"  {len(arr):,} consecutive residual pairs from patients with 4+ visits")
    say(f"  observed lag-1 correlation: {r:+.3f}")

    # This statistic cannot be read against zero. Residuals from a line fitted
    # to a handful of points are negatively correlated by construction, so the
    # null is not zero. Simulate iid noise on the same visit pattern to find it.
    rng = np.random.default_rng(cfg.seed)
    sigma_hat = float(np.sqrt(ms_within))
    null = []
    for _ in range(200):
        sim = []
        for patno, group in finite.groupby("PATNO"):
            if len(group) < 4:
                continue
            t = group["years_from_baseline"].to_numpy()
            if t.max() - t.min() < 1.0:
                continue
            y = 2.0 - 0.09 * t + rng.normal(0.0, sigma_hat, len(t))
            e = y - np.polyval(np.polyfit(t, y, 1), t)
            sim.extend(zip(e[:-1], e[1:]))
        a = np.array(sim)
        null.append(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    null = np.array(null)
    lo, hi = np.quantile(null, [0.025, 0.975])
    say(f"  null from iid noise on the same visits: {null.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

    if lo <= r <= hi:
        say("\n  The observed value sits inside the null interval, so the negative")
        say("  correlation is the mechanical artefact of detrending short series, not")
        say("  evidence of structure. Independent deviations --- the growth model's")
        say("  assumption --- are consistent with the data. A state-space form would")
        say("  imply POSITIVE residual correlation, and there is no sign of that.")
    elif r > hi:
        say("\n  More positive than the null, which is the signature of accumulating")
        say("  process noise and favours the state-space form the proposal mentions.")
    else:
        say("\n  More negative than the null, indicating mean reversion beyond what")
        say("  detrending alone produces.")

    # ------------------------------------------------- conclusions
    section("WHAT THIS IMPLIES FOR THE MODEL")
    say("  1. The observation scale is the main problem. The ratio is non-negative,")
    say("     right-skewed and has an atom at zero; a Normal likelihood on the raw")
    say("     scale is misspecified and will predict impossible values.")
    say(f"  2. Within-patient SD ({np.sqrt(ms_within):.2f}) is several times the Indeterminate")
    say("     band width (0.25), so band membership is intrinsically noisy.")
    say("  3. Raw per-patient slopes are far more spread out than the hierarchical")
    say("     estimate, so shrinkage is doing real work and tau1 should not be")
    say("     given a prior centred on the naive spread.")
    say(f"  4. Lag-1 residual correlation is {r:+.3f} against a null of {null.mean():+.3f} "
        f"[{lo:+.3f}, {hi:+.3f}]")
    if r > hi:
        say("     from iid noise on the same visit pattern. The observed value is")
        say("     MORE POSITIVE than the null and outside its interval, so deviations")
        say("     from trend persist between visits rather than resetting. That is the")
        say("     signature of accumulating process noise and favours the state-space")
        say("     form over the growth form the proposal settled on --- and the")
        say("     state-space form is also the one in which the observation and")
        say("     process variances become separately identifiable. Two independent")
        say("     reasons to prefer it.")
    elif r < lo:
        say("     from iid noise. More negative than the null: mean reversion beyond")
        say("     what detrending alone produces.")
    else:
        say("     from iid noise on the same visit pattern. Inside the null interval,")
        say("     so independent deviations are defensible and the growth form is")
        say("     adequate on this evidence.")
    say("  5. Roughly one visit in six has no finite ratio at all, so any")
    say("     ratio-based model silently loses them unless the two subscores are")
    say("     modelled instead.")

    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_dir.relative_to(cfg.root)}/report.txt and "
          f"{len(list(out_dir.glob('*.csv')))} CSV tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
