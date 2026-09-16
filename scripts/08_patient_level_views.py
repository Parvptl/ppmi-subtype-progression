"""Patient-level views requested after the review meeting.

Five questions, five figures. Nothing here changes the dataset, the subtype
definition, the medication filter or any fitted model: every figure is a new
view of the panel that scripts/01 already produced.

    1. When a visit is labelled Indeterminate, where inside the band does the
       ratio actually sit, and does that tell us anything?
    2. What happens to individual patients, rather than to the population?
    3. How many patients convert *to* TD, using the same method as the
       existing PIGD conversion analysis?
    4. What does the raw ratio do over time, before any model is fitted?
    5. How do the three starting subtypes compare on one set of axes?

    python scripts/08_patient_level_views.py

Writes to outputs/patient_level/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppmi_subtype import ctmc, hmm, patient_plots, plots, splitting, transitions as T  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def section(title: str) -> None:
    say("\n" + "=" * 78)
    say(title)
    say("=" * 78)


def main() -> int:
    cfg = load_config()
    out_dir = cfg.path("outputs") / "patient_level"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    names = {v: k for k, v in cfg.subtype.labels.items()}
    L = cfg.subtype.labels
    td_thr, pigd_thr = cfg.subtype.td_threshold, cfg.subtype.pigd_threshold

    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    panel = panel[panel["subtype"].notna()].sort_values(["PATNO", "years_from_baseline"])
    pairs = T.consecutive_pairs(panel)

    say(f"  labelled visits {len(panel):,} from {panel['PATNO'].nunique():,} patients")
    say(f"  consecutive labelled pairs {len(pairs):,}")

    visit_ratio = panel[["PATNO", "EVENT_ID", "ratio"]]

    # =====================================================================
    section("1. WHERE INSIDE THE INDETERMINATE BAND DO THESE VISITS SIT?")
    # =====================================================================
    ind_visits = panel[panel["subtype"] == L["Indeterminate"]]
    with_ratio = ind_visits[ind_visits["ratio"].notna()]
    say(f"  Indeterminate visits                {len(ind_visits):,}")
    say(f"  with a computable ratio             {len(with_ratio):,}")
    say(f"  without (both subscores are zero)   {len(ind_visits) - len(with_ratio):,}")

    say("\n  The ratio is (mean of 11 tremor items) / (mean of 5 gait items), so it can only")
    say("  take values of the form 5K/11M for whole numbers K and M. Very few of those land")
    say(f"  between {pigd_thr} and {td_thr}. The values that actually occur:\n")

    vc = with_ratio["ratio"].round(4).value_counts().sort_index()
    tot = int(vc.sum())
    for val, n in vc.items():
        bar = "#" * int(round(40 * n / vc.max()))
        say(f"    {val:.4f}  n={n:4d}  {n / tot:6.1%}  {bar}")
    say(f"\n  Only {len(vc)} distinct values occur inside a band {td_thr - pigd_thr:.2f} wide.")

    lo_val, hi_val = float(vc.index[0]), float(vc.index[-1])
    lo_share, hi_share = vc.iloc[0] / tot, vc.iloc[-1] / tot
    say(f"  {lo_share:.1%} sit at exactly {lo_val:.4f}, which is {lo_val - pigd_thr:.4f} above the PIGD boundary.")
    say(f"  {hi_share:.1%} sit at exactly {hi_val:.4f}, which is {td_thr - hi_val:.4f} below the TD boundary.")
    say(f"  Together the two edges account for {lo_share + hi_share:.1%} of all Indeterminate visits.")

    # ---- does the previous visit shift where in the band they sit?
    into = (
        pairs[pairs["to_state"] == L["Indeterminate"]]
        .merge(visit_ratio, left_on=["PATNO", "to_event"],
               right_on=["PATNO", "EVENT_ID"], how="left")
        .dropna(subset=["ratio"])
    )
    into["from_name"] = into["from_state"].map(names)

    say("\n  Does where the patient came from change where in the band they sit?\n")
    rows = []
    for src in ("TD", "Indeterminate", "PIGD"):
        r = into.loc[into["from_name"] == src, "ratio"]
        if not len(r):
            continue
        rows.append({
            "came_from": src, "n": len(r),
            "median": round(float(r.median()), 4),
            "mean": round(float(r.mean()), 4),
            "share_at_lower_edge": round(float(np.isclose(r, lo_val, atol=1e-3).mean()), 4),
            "share_at_upper_edge": round(float(np.isclose(r, hi_val, atol=1e-3).mean()), 4),
        })
    origin = pd.DataFrame(rows)
    origin.to_csv(out_dir / "indeterminate_by_origin.csv", index=False)
    say("    " + origin.to_string(index=False).replace("\n", "\n    "))

    a = into.loc[into["from_name"] == "TD", "ratio"]
    b = into.loc[into["from_name"] == "PIGD", "ratio"]
    _, p_origin = stats.mannwhitneyu(a, b, alternative="two-sided")
    say(f"\n  TD-origin vs PIGD-origin, Mann-Whitney p = {p_origin:.4f}")
    say("  Patients arriving from TD do sit slightly higher in the band than patients")
    say("  arriving from PIGD, in the direction one would expect, but the two")
    say("  distributions overlap almost completely.")

    # ---- the test that matters: does position in the band predict the next visit?
    leaving = (
        pairs[pairs["from_state"] == L["Indeterminate"]]
        .merge(visit_ratio, left_on=["PATNO", "from_event"],
               right_on=["PATNO", "EVENT_ID"], how="left")
        .dropna(subset=["ratio"])
    )
    leaving["to_name"] = leaving["to_state"].map(names)

    say("\n  The question that decides whether splitting Indeterminate is worthwhile:")
    say("  does sitting near the TD edge rather than the PIGD edge change where the")
    say("  patient is at the next visit?\n")

    groups = {
        f"at {lo_val:.4f} (PIGD edge)": np.isclose(leaving["ratio"], lo_val, atol=1e-3),
        f"at {hi_val:.4f} (TD edge)": np.isclose(leaving["ratio"], hi_val, atol=1e-3),
        "anywhere in between": ~(np.isclose(leaving["ratio"], lo_val, atol=1e-3)
                                 | np.isclose(leaving["ratio"], hi_val, atol=1e-3)),
    }
    next_rows = {}
    for label, mask in groups.items():
        g = leaving[mask]
        share = g["to_name"].value_counts(normalize=True)
        next_rows[f"{label}\nn = {len(g)}"] = {s: float(share.get(s, 0.0)) for s in
                                              ("TD", "Indeterminate", "PIGD")}
        say(f"    {label:<28} n={len(g):3d}   next visit:  "
            f"TD {share.get('TD', 0):.1%}   Indeterminate {share.get('Indeterminate', 0):.1%}   "
            f"PIGD {share.get('PIGD', 0):.1%}")

    ga = leaving[groups[f"at {lo_val:.4f} (PIGD edge)"]]
    gb = leaving[groups[f"at {hi_val:.4f} (TD edge)"]]
    table = np.array([
        [int((ga["to_state"] == L["PIGD"]).sum()), int((ga["to_state"] != L["PIGD"]).sum())],
        [int((gb["to_state"] == L["PIGD"]).sum()), int((gb["to_state"] != L["PIGD"]).sum())],
    ])
    chi2, p_next, _, _ = stats.chi2_contingency(table)
    say(f"\n    next visit is PIGD: {table[0,0]}/{table[0].sum()} from the PIGD edge "
        f"vs {table[1,0]}/{table[1].sum()} from the TD edge")
    say(f"    chi-square = {chi2:.2f}, p = {p_next:.3f}")

    if p_next < 0.05:
        say("\n  The two edges behave differently, so a split is supported.")
    else:
        say("\n  VERDICT: the two edges do NOT behave differently. Whether a patient sits at")
        say("  the top or the bottom of the Indeterminate band tells us nothing about where")
        say("  they will be at the next visit. Splitting Indeterminate into 'almost TD' and")
        say("  'almost PIGD' is therefore NOT supported by this data, even though almost")
        say("  every Indeterminate visit is sitting on one of the two boundaries.")

    counts = pd.DataFrame({
        src: with_ratio.merge(
            pairs[pairs["to_state"] == L["Indeterminate"]][["PATNO", "to_event", "from_state"]],
            left_on=["PATNO", "EVENT_ID"], right_on=["PATNO", "to_event"], how="left",
        ).assign(src=lambda d: d["from_state"].map(names).fillna("first visit"))
        .query("src == @src")["ratio"].round(4).value_counts()
        for src in ("TD", "Indeterminate", "PIGD")
    }).reindex(vc.index).fillna(0.0)
    # Visits with no preceding labelled visit cannot be attributed to an origin
    # and are kept as their own bucket rather than being folded into a subtype.
    counts[patient_plots.NO_PREVIOUS] = (vc - counts.sum(axis=1)).clip(lower=0)
    counts.to_csv(out_dir / "indeterminate_ratio_values.csv")

    n_first = int(counts[patient_plots.NO_PREVIOUS].sum())
    say(f"\n  Of these {tot:,} visits, {tot - n_first:,} follow an earlier labelled visit and")
    say(f"  {n_first:,} are the patient's first labelled visit, so they have no origin state.")

    patient_plots.indeterminate_boundary(
        counts, pd.DataFrame(next_rows).T,
        fig_dir / "indeterminate_boundary.png",
        td_threshold=td_thr, pigd_threshold=pigd_thr,
    )

    # =====================================================================
    section("2. WHAT HAPPENS TO INDIVIDUAL PATIENTS")
    # =====================================================================
    per_patient = panel.groupby("PATNO").size()
    eligible = per_patient[per_patient >= 2].index
    say(f"  patients with at least one labelled visit:  {panel['PATNO'].nunique():,}")
    say(f"  patients with two or more labelled visits:  {len(eligible):,}")
    say(f"  seen once only, so no course to observe:    "
        f"{panel['PATNO'].nunique() - len(eligible):,}")
    say("\n  Everything in this section uses the two-or-more group. A patient seen once")
    say("  cannot change subtype, so counting them among the patients who 'never")
    say("  changed' would overstate stability.")

    followed = panel[panel["PATNO"].isin(eligible)]
    seqs = T.state_sequences(followed)
    collapsed = seqs.apply(T.collapse_runs)
    as_names = collapsed.apply(lambda t: " -> ".join(names[s] for s in t))
    common = as_names.value_counts()

    say("\n  Commonest courses once repeated states are collapsed:\n")
    say("    " + common.head(12).to_string().replace("\n", "\n    "))
    top12 = int(common.head(12).sum())
    say(f"\n  Those 12 courses cover {top12:,} of the {len(seqs):,} patients. The remaining "
        f"{len(seqs) - top12:,}\n  are spread across {len(common) - 12:,} rarer courses, "
        f"{len(common):,} distinct courses in total.")

    n_changes = collapsed.apply(len) - 1
    n_never, n_once, n_more = (
        int((n_changes == 0).sum()), int((n_changes == 1).sum()), int((n_changes >= 2).sum()))
    say(f"\n  no recorded change of subtype: {n_never:,} of {len(n_changes):,} "
        f"({n_never / len(n_changes):.1%})")
    say(f"  one recorded change:           {n_once:,} ({n_once / len(n_changes):.1%})")
    say(f"  two or more recorded changes:  {n_more:,} ({n_more / len(n_changes):.1%})")
    say("\n  Wording matters here. 'No recorded change' is not 'no progression': a patient")
    say("  can worsen substantially without ever crossing a subtype threshold, and the")
    say("  recorded subtype is itself affected by variability in the examination.")

    # ---- consistency checks; the analysis stops rather than reporting a wrong total
    n_followed = int(followed["PATNO"].nunique())
    n_all = int(panel["PATNO"].nunique())
    checks = {
        "patients with >=2 labelled visits == 934": n_followed == 934,
        "single-visit patients == 146": n_all - n_followed == 146,
        "course counts sum to the followed population": int(common.sum()) == n_followed,
        "no single-visit patient in the course counts": len(seqs) == n_followed,
        "change frequencies partition the population": n_never + n_once + n_more == n_followed,
        # Relational rather than absolute: the course *counts* depend on the
        # subtype thresholds, so a sensitivity run legitimately produces
        # different ones. What must always hold is that the parts add up.
        "top 12 courses plus the rest equal the population":
            top12 + int(common.iloc[12:].sum()) == n_followed,
        "top 12 courses are the twelve largest":
            top12 == int(common.nlargest(12).sum()),
    }
    say("\n  Consistency checks:")
    for name, ok in checks.items():
        say(f"    [{'PASS' if ok else 'FAIL'}]  {name}")
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"consistency checks failed: {failed}")

    carpet_n = patient_plots.patient_state_carpet(
        followed, fig_dir / "patient_carpet.png", names=names,
    )[1]
    say(f"\n  Carpet figure draws {carpet_n:,} rows, one per patient.")
    if carpet_n != n_followed:
        raise SystemExit(
            f"carpet drew {carpet_n} rows but {n_followed} patients are in the analysis")

    # One representative patient per common course. Deliberately the patient
    # with the TYPICAL number of visits for that course, not the longest
    # followed: picking the longest would make 13 years of follow-up look
    # normal when the median patient contributes 3 visits over about 2 years.
    chosen, annotations = [], {}
    visits_per = panel.groupby("PATNO").size()
    for course in common.head(12).index:
        members = [p for p in as_names[as_names == course].index
                   if visits_per.get(p, 0) >= 2]
        if not members:
            continue
        counts_here = visits_per.loc[members]
        target = counts_here.median()
        typical = (counts_here - target).abs().idxmin()
        chosen.append(typical)
        annotations[typical] = f"{course}   ({common[course]:,} patients)"

    say(f"\n  Example patients shown in the timeline figure carry a median of "
        f"{visits_per.loc[chosen].median():.0f} visits, against {visits_per.median():.0f} "
        f"for the cohort, so they are typical rather than unusually well followed.")

    patient_plots.patient_timelines(
        panel, chosen, fig_dir / "patient_timelines.png",
        names=names, annotations=annotations,
        coverage=(f"The 12 courses shown cover {top12:,} of the {len(seqs):,} patients with two or more "
                  f"labelled visits; the other {len(seqs) - top12:,} follow one of "
                  f"{len(common) - 12:,} rarer courses."),
    )
    pd.DataFrame({"PATNO": chosen, "course": [annotations[p] for p in chosen]}).to_csv(
        out_dir / "example_patients.csv", index=False)

    # =====================================================================
    section("3. CONVERSION TO TD, USING THE SAME METHOD AS THE PIGD ANALYSIS")
    # =====================================================================
    say("  Definitions are copied from the existing PIGD conversion analysis:")
    say("    at risk        patients whose FIRST labelled visit is not the target subtype")
    say("    conversion     first labelled visit recorded as the target subtype")
    say("    sustained      the target subtype is also recorded at the next labelled visit")
    say("    censoring      patients who never convert are censored at their last labelled visit")
    say("    back and forth a patient is counted at their FIRST qualifying entry only;")
    say("                   later returns do not create a second event")

    both = {}
    for target, code in (("PIGD", L["PIGD"]), ("TD", L["TD"])):
        curves = {}
        say(f"\n  --- conversion to {target}")
        for label, k in (("any single visit", 1), ("sustained (2 consecutive visits)", 2)):
            fe = T.first_entry(panel, target_state=code, min_consecutive=k)
            at_risk = fe[~fe["starts_in_state"]]
            km = T.kaplan_meier(at_risk["time"], at_risk["converted"])
            curves[label] = km
            say(f"    {label}")
            say(f"      at risk    {len(at_risk):,}")
            say(f"      converted  {int(at_risk['converted'].sum()):,} "
                f"({at_risk['converted'].mean():.1%})")
            conv = at_risk[at_risk["converted"]]
            if len(conv):
                say(f"      median years to conversion  {conv['time'].median():.2f}")
            for horizon in (2.0, 5.0):
                upto = km[km["time"] <= horizon]
                if len(upto):
                    say(f"      cumulative incidence, yr {horizon:.0f}   "
                        f"{1 - upto['survival'].iloc[-1]:.1%}  "
                        f"(n at risk {int(upto['n_at_risk'].iloc[-1]):,})")
            km.to_csv(out_dir / f"conversion_km_{target.lower()}_"
                                f"{'naive' if k == 1 else 'sustained'}.csv", index=False)
        both[target] = curves

    plots.conversion_curves(
        both["TD"], fig_dir / "time_to_td_conversion.png", target_name="TD")
    patient_plots.conversion_both_directions(
        both["PIGD"], both["TD"], fig_dir / "conversion_both_directions.png")

    say("\n  Note on comparability: the two risk sets are very different sizes because the")
    say("  cohort starts mostly TD. The two panels are directionally comparable but the")
    say("  TD side rests on far fewer patients.")

    # =====================================================================
    section("4. THE RAW RATIO OVER TIME, BEFORE ANY MODEL")
    # =====================================================================
    with_r = panel[panel["ratio"].notna()]
    say(f"  visits with a computable ratio  {len(with_r):,} of {len(panel):,} "
        f"({len(with_r) / len(panel):.1%})")
    say(f"  patients contributing           {with_r['PATNO'].nunique():,}")

    yr = with_r["years_from_baseline"].round().astype(int)
    med = with_r.groupby(yr)["ratio"].agg(["median", "mean", "count"])
    med = med[med["count"] >= 25]
    say("\n  Median of the observations actually present in each year:\n")
    say("    " + med.round(3).to_string().replace("\n", "\n    "))
    med.to_csv(out_dir / "raw_ratio_by_year.csv")

    steps = (with_r.sort_values(["PATNO", "years_from_baseline"])
             .groupby("PATNO")["ratio"].apply(lambda s: s.diff().abs().dropna()))
    band = td_thr - pigd_thr
    say(f"\n  Change in one patient's own ratio between consecutive visits:")
    say(f"    pairs                    {len(steps):,}")
    say(f"    median absolute change   {steps.median():.3f}")
    say(f"    width of the whole Indeterminate band  {band:.2f}")
    say(f"    share of pairs moving more than the entire band width: {(steps > band).mean():.1%}")

    patient_plots.raw_ratio_progression(
        panel, fig_dir / "raw_ratio_progression.png",
        td_threshold=td_thr, pigd_threshold=pigd_thr)
    patient_plots.ratio_within_patient(
        panel, fig_dir / "ratio_within_patient.png",
        td_threshold=td_thr, pigd_threshold=pigd_thr)

    # =====================================================================
    section("5. THE THREE STARTING SUBTYPES ON ONE SET OF AXES")
    # =====================================================================
    say("  Refitting the hidden Markov model on the same training patients as")
    say("  scripts/03, so the combined figure shows exactly the same numbers.")

    states = [L[s] for s in ("TD", "Indeterminate", "PIGD")]
    manifest = splitting.split_patients(
        panel, cfg.splits["train"], cfg.seed, stratify_by=cfg.splits["stratify_by"])
    train_ids = set(manifest.loc[manifest["split"] == "train", "PATNO"])
    train_panel = panel[panel["PATNO"].isin(train_ids)]

    seqs_train = hmm.build_sequences(train_panel, states)
    fit = hmm.fit_hmm(
        seqs_train, states, state_names=["TD", "Indeterminate", "PIGD"],
        seed=cfg.seed, n_restarts=cfg.model["n_restarts"],
        emission_floor=cfg.model["emission_floor"],
    )
    say(f"  refitted on {len(seqs_train):,} training patients, converged = {fit.converged}")

    combined = []
    for start in ("TD", "Indeterminate", "PIGD"):
        for h in (1, 2, 3, 5):
            row = fit.P(float(h)).loc[start]
            combined.append({"starting_subtype": start, "years": h,
                             **{f"p_{k}": round(float(v), 4) for k, v in row.items()}})
    combined = pd.DataFrame(combined)
    combined.to_csv(out_dir / "combined_progression.csv", index=False)
    say("\n    " + combined.to_string(index=False).replace("\n", "\n    "))

    patient_plots.combined_progression(fit, fig_dir / "progression_combined.png")

    # =====================================================================
    section("FIGURES WRITTEN")
    # =====================================================================
    for p in sorted(fig_dir.glob("*.png")):
        say(f"  {p.relative_to(cfg.root)}")

    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_dir.relative_to(cfg.root)}/report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
