"""Stage 2 - subtype transition analysis (Weeks 3-4 deliverable).

Tracks subtype changes across visits, computes transition frequencies and
probabilities over different follow-up intervals, and visualises progression
pathways.

Requires outputs/visit_panel.csv from scripts/01_build_panel.py.

Writes to outputs/transitions/:
    transition_matrix_{counts,probs}.csv   pooled 3x3
    transition_probs_ci.csv                with Wilson 95% intervals
    transitions_by_interval.csv            stratified by elapsed interval
    prevalence_by_year.csv
    reversion_summary.csv
    pathways.csv                           collapsed state pathways
    conversion_km_{naive,sustained}.csv
    figures/*.png
    report.txt

    python scripts/02_transition_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The report uses arrows and inequality signs; the default Windows console
# codepage is cp1252 and cannot encode them.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from ppmi_subtype import plots, transitions as T  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

STATES = [1.0, 2.0, 3.0]
INTERVAL_BINS = (0.0, 1.5, 2.5, 4.0, np.inf)
INTERVAL_LABELS = {
    "[0.0, 1.5)": "≤ 1.5",
    "[1.5, 2.5)": "1.5 – 2.5",
    "[2.5, 4.0)": "2.5 – 4.0",
    "[4.0, inf)": "> 4.0",
}

lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def section(title: str) -> None:
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


def named(frame: pd.DataFrame, names: dict) -> pd.DataFrame:
    out = frame.copy()
    out.index = [names.get(i, i) for i in out.index]
    out.columns = [names.get(c, c) for c in out.columns]
    return out


def main() -> int:
    cfg = load_config()
    names = cfg.subtype.label_names
    out_dir = cfg.path("outputs") / "transitions"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_path = cfg.path("outputs") / "visit_panel.csv"
    if not panel_path.exists():
        print(f"{panel_path} not found; run scripts/01_build_panel.py first", file=sys.stderr)
        return 1

    panel = pd.read_csv(panel_path)
    labelled = panel[panel["subtype"].notna()]

    section("COHORT")
    say(f"  patients with >=1 labelled visit   {labelled.PATNO.nunique():,}")
    say(f"  labelled visits                    {len(labelled):,}")

    pairs = T.consecutive_pairs(panel)
    say(f"  consecutive labelled visit pairs   {len(pairs):,}")
    say(f"  patients contributing a transition {pairs.PATNO.nunique():,}")
    say(f"  median interval                    {pairs['interval_years'].median():.2f} years")

    # ---------------------------------------------------------------- matrices
    section("1. POOLED TRANSITION MATRIX")
    counts = named(T.transition_matrix(pairs, states=STATES, normalize=False), names)
    probs = named(T.transition_matrix(pairs, states=STATES), names)

    say("\n  Counts (rows = current visit, cols = next visit):")
    say("    " + counts.to_string().replace("\n", "\n    "))
    say("\n  Probabilities:")
    say("    " + probs.round(3).to_string().replace("\n", "\n    "))

    ci = T.transition_counts_with_ci(pairs, states=STATES)
    ci["from_state"] = ci["from_state"].map(names)
    ci["to_state"] = ci["to_state"].map(names)
    say("\n  With Wilson 95% intervals:")
    say("    " + ci.round(3).to_string(index=False).replace("\n", "\n    "))

    counts.to_csv(out_dir / "transition_matrix_counts.csv")
    probs.to_csv(out_dir / "transition_matrix_probs.csv")
    ci.to_csv(out_dir / "transition_probs_ci.csv", index=False)

    say("\n  Reading:")
    diag = {s: probs.loc[s, s] for s in plots.STATE_ORDER}
    say(f"    TD is the most stable state ({diag['TD']:.1%} persist), PIGD next ({diag['PIGD']:.1%}).")
    say(f"    Indeterminate is not a state so much as a crossing: only {diag['Indeterminate']:.1%}")
    say(f"    remain, {probs.loc['Indeterminate', 'TD']:.1%} return to TD and "
        f"{probs.loc['Indeterminate', 'PIGD']:.1%} move to PIGD.")
    say(f"    PIGD is NOT absorbing: {1 - diag['PIGD']:.1%} of PIGD visits are followed by a")
    say(f"    non-PIGD assessment, {probs.loc['PIGD', 'TD']:.1%} of them straight back to TD.")

    # --------------------------------------------------------------- intervals
    section("2. STRATIFIED BY ELAPSED INTERVAL")
    binned = pairs.assign(
        interval_bin=pd.cut(pairs["interval_years"], bins=list(INTERVAL_BINS), right=False)
    )
    records = []
    for b, grp in binned.groupby("interval_bin", observed=True):
        sub = T.transition_counts_with_ci(grp, states=STATES)
        sub["interval"] = INTERVAL_LABELS.get(str(b), str(b))
        records.append(sub)
    by_interval = pd.concat(records, ignore_index=True)
    by_interval["from_state"] = by_interval["from_state"].map(names)
    by_interval["to_state"] = by_interval["to_state"].map(names)
    by_interval.to_csv(out_dir / "transitions_by_interval.csv", index=False)

    for label in INTERVAL_LABELS.values():
        sub = by_interval[by_interval["interval"] == label]
        if sub.empty:
            continue
        mat = sub.pivot(index="from_state", columns="to_state", values="probability")
        mat = mat.reindex(index=plots.STATE_ORDER, columns=plots.STATE_ORDER)
        n_from = sub.groupby("from_state")["n_from"].first().reindex(plots.STATE_ORDER)
        say(f"\n  interval {label} years   (n from each state: "
            f"{', '.join(f'{s} {int(v)}' for s, v in n_from.items())})")
        say("    " + mat.round(3).to_string().replace("\n", "\n    "))

    td_pigd = by_interval[(by_interval.from_state == "TD") & (by_interval.to_state == "PIGD")]
    say("\n  Reading:")
    say(f"    TD → PIGD rises from {td_pigd.probability.iloc[0]:.3f} at ≤1.5 years to "
        f"{td_pigd.probability.iloc[-1]:.3f} beyond 4 years.")
    say("    A single pooled matrix is therefore not a well-defined quantity: it averages")
    say("    one-year and multi-year transitions in whatever proportion attendance happened")
    say("    to produce. Either stratify, or move to continuous time in the Weeks 5-7 model.")

    # -------------------------------------------------------------- prevalence
    section("3. PREVALENCE ACROSS FOLLOW-UP")
    prev = T.prevalence_by_year(panel)
    prev.columns = [str(c).replace("1.0", "TD").replace("2.0", "Indeterminate").replace("3.0", "PIGD")
                    for c in prev.columns]
    prev.to_csv(out_dir / "prevalence_by_year.csv")
    say("    " + prev.round(3).to_string().replace("\n", "\n    "))

    # --------------------------------------------------------------- reversion
    section("4. REVERSION: HOW MUCH OF THIS IS NOISE?")
    rev = T.reversion_after_change(panel)
    rev_summary = T.reversion_summary(rev, names=names)
    rev_summary.to_csv(out_dir / "reversion_summary.csv", index=False)

    say("  Of every observed change of state, the share back at the origin one visit later:\n")
    say("    " + rev_summary.round(3).to_string(index=False).replace("\n", "\n    "))
    overall = rev["reverted"].mean()
    say(f"\n  Overall, {overall:.1%} of observed subtype changes reverse at the very next visit.")
    say("  That is the single most important number in this analysis: a transition matrix")
    say("  built from raw labels is measuring assessment noise as much as disease progression.")

    # -------------------------------------------------------------- conversion
    section("5. TIME TO PIGD CONVERSION")
    pigd_code = cfg.subtype.labels["PIGD"]
    curves, km_frames = {}, {}
    for label, k in (("any single visit", 1), ("sustained (2 consecutive visits)", 2)):
        fe = T.first_entry(panel, target_state=pigd_code, min_consecutive=k)
        at_risk = fe[~fe["starts_in_state"]]
        km = T.kaplan_meier(at_risk["time"], at_risk["converted"])
        curves[label] = km
        km_frames[k] = (at_risk, km)
        say(f"\n  {label}:")
        say(f"    at risk (not PIGD at baseline)  {len(at_risk):,}")
        say(f"    converted                       {int(at_risk['converted'].sum()):,} "
            f"({at_risk['converted'].mean():.1%})")
        conv = at_risk[at_risk["converted"]]
        if len(conv):
            say(f"    median years to conversion      {conv['time'].median():.2f}")
        for horizon in (2.0, 5.0):
            upto = km[km["time"] <= horizon]
            if len(upto):
                say(f"    KM cumulative incidence, yr {horizon:.0f}   "
                    f"{1 - upto['survival'].iloc[-1]:.1%}  "
                    f"(n at risk {int(upto['n_at_risk'].iloc[-1]):,})")

    curves["any single visit"].to_csv(out_dir / "conversion_km_naive.csv", index=False)
    curves["sustained (2 consecutive visits)"].to_csv(out_dir / "conversion_km_sustained.csv", index=False)

    naive_n = int(km_frames[1][0]["converted"].sum())
    sust_n = int(km_frames[2][0]["converted"].sum())
    say(f"\n  Requiring two consecutive PIGD assessments removes {naive_n - sust_n:,} of "
        f"{naive_n:,} apparent conversions ({(naive_n - sust_n) / naive_n:.1%}).")
    say("  Those patients were labelled PIGD once and were not PIGD at the following visit.")

    # --------------------------------------------------------------- pathways
    section("6. PROGRESSION PATHWAYS")

    # A pathway needs at least two labelled visits to exist. Including patients
    # seen once counts them among those who "never changed" when they never had
    # the opportunity, which overstates stability.
    # Count LABELLED visits only. `panel` still carries the 1,333 visits with no
    # subtype, and counting those would let a patient with one labelled visit
    # qualify on the strength of visits that carry no subtype at all.
    labelled_per_patient = labelled.groupby("PATNO").size()
    followed = labelled[
        labelled["PATNO"].isin(labelled_per_patient[labelled_per_patient >= 2].index)
    ]
    say(f"  patients with at least one labelled visit  {labelled['PATNO'].nunique():,}")
    say(f"  patients with two or more labelled visits  {followed['PATNO'].nunique():,}"
        f"   <- pathways are computed on these")
    say(f"  seen once only, so no pathway exists        "
        f"{labelled['PATNO'].nunique() - followed['PATNO'].nunique():,}")

    seqs = T.state_sequences(followed)
    collapsed = seqs.apply(T.collapse_runs)
    as_names = collapsed.apply(lambda t: " → ".join(names[s] for s in t))

    path_counts = as_names.value_counts().rename_axis("pathway").reset_index(name="n_patients")
    path_counts["share"] = path_counts["n_patients"] / path_counts["n_patients"].sum()
    path_counts.to_csv(out_dir / "pathways.csv", index=False)

    say("\n  Most common pathways after collapsing repeated states "
        f"({len(path_counts):,} distinct pathways across {len(collapsed):,} patients):\n")
    say("    " + path_counts.head(12).round(3).to_string(index=False).replace("\n", "\n    "))

    n_changes = collapsed.apply(len) - 1
    say(f"\n  State changes per patient: median {n_changes.median():.0f}, mean {n_changes.mean():.2f}, "
        f"max {n_changes.max()}")
    say(f"  Patients with no recorded change of state: {(n_changes == 0).sum():,} of "
        f"{len(n_changes):,} ({(n_changes == 0).mean():.1%})")
    say("  Note the wording: no recorded change is not the same as no progression.")
    say("  A patient can worsen without crossing a subtype threshold.")

    # ---------------------------------------------------------------- figures
    section("7. FIGURES")
    wide = panel.pivot_table(index="PATNO", columns="YEAR", values="subtype", aggfunc="first")
    checkpoints = [0, 2, 4]
    complete = wide[checkpoints].dropna() if all(c in wide.columns for c in checkpoints) else pd.DataFrame()

    written = [
        plots.transition_heatmap(
            probs, counts, fig_dir / "transition_matrix.png",
            f"Subtype transition probabilities, consecutive visits (n = {len(pairs):,} pairs)",
        ),
        plots.prevalence_area(prev, fig_dir / "prevalence_over_time.png"),
        plots.transitions_by_interval(
            by_interval, fig_dir / "transitions_by_interval.png",
            highlight=[("TD", "PIGD"), ("Indeterminate", "PIGD"), ("PIGD", "TD")],
        ),
        plots.conversion_curves(curves, fig_dir / "time_to_pigd_conversion.png"),
        plots.reversion_bars(rev_summary, fig_dir / "reversion.png"),
    ]

    if not complete.empty:
        frames = [
            pd.crosstab(complete[checkpoints[k]], complete[checkpoints[k + 1]])
            .rename(index=names, columns=names)
            for k in range(len(checkpoints) - 1)
        ]
        written.append(
            plots.pathway_flow(
                frames, ["Baseline", "Year 2", "Year 4"],
                fig_dir / "pathway_flow.png", n_patients=len(complete),
            )
        )

    for p in written:
        say(f"  {p.relative_to(cfg.root)}")

    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_dir.relative_to(cfg.root)}/report.txt and "
          f"{len(list(out_dir.glob('*.csv')))} CSV tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
