"""Independent numerical audit of every figure quoted in the manuscript.

Reads only the generated panel and the PPMI source files; recomputes each
quantity from scratch rather than trusting a report. Prints a flat list so each
line can be checked against the manuscript text.

    python -I scripts/_verify_paper_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype import loaders, subtype as S  # noqa: E402

pd.set_option("display.width", 200)


def say(label: str, value) -> None:
    print(f"{label:<62} {value}")


def main() -> int:
    cfg = load_config()
    panel = pd.read_csv(ROOT / "outputs" / "visit_panel.csv", low_memory=False)

    print("=" * 90)
    print("A. COHORT AND PANEL")
    print("=" * 90)
    say("patients", f"{panel['PATNO'].nunique():,}")
    say("visits", f"{len(panel):,}")
    lab = panel[panel["subtype"].notna()]
    say("labelled visits", f"{len(lab):,}  ({len(lab)/len(panel):.3%})")
    say("unlabelled visits", f"{len(panel)-len(lab):,}  ({1-len(lab)/len(panel):.3%})")
    say("patients with >=1 labelled visit", f"{lab['PATNO'].nunique():,}")
    say("patients with >=2 labelled visits",
        f"{(lab.groupby('PATNO').size() >= 2).sum():,}")
    say("max years_from_baseline (labelled)", f"{lab['years_from_baseline'].max():.2f}")
    say("max years_from_baseline (all visits)", f"{panel['years_from_baseline'].max():.2f}")
    say("max YEAR (labelled)", f"{lab['YEAR'].max():.0f}")

    per_pat = lab.groupby("PATNO")["years_from_baseline"]
    span = (per_pat.max() - per_pat.min())
    say("labelled follow-up span: median / IQR",
        f"{span.median():.2f} / {span.quantile(.25):.2f}-{span.quantile(.75):.2f}")
    say("labelled visits/patient: median [IQR]",
        f"{lab.groupby('PATNO').size().median():.0f} "
        f"[{lab.groupby('PATNO').size().quantile(.25):.0f}-"
        f"{lab.groupby('PATNO').size().quantile(.75):.0f}]")

    print()
    print("Baseline characteristics (first labelled visit per patient):")
    first = lab.sort_values(["PATNO", "years_from_baseline"]).groupby("PATNO").head(1)
    say("n", f"{len(first):,}")
    for col, name in [("age_at_visit", "age at first labelled visit"),
                      ("duration_yrs", "disease duration, years"),
                      ("updrs3_score", "MDS-UPDRS III (OFF)"),
                      ("updrs2_score", "MDS-UPDRS II"),
                      ("hy", "Hoehn & Yahr"),
                      ("moca", "MoCA"),
                      ("LEDD", "LEDD mg/day")]:
        if col in first.columns:
            v = pd.to_numeric(first[col], errors="coerce")
            say(f"  {name}", f"mean {v.mean():.2f}  sd {v.std():.2f}  "
                             f"median {v.median():.2f}  n {v.notna().sum():,}")
    if "SEX" in first.columns:
        say("  sex distribution", dict(first["SEX"].value_counts()))
    say("  PDTRTMNT==0 at first labelled visit",
        f"{(pd.to_numeric(first.get('PDTRTMNT'), errors='coerce') == 0).sum():,}")

    print()
    print("Subtype distribution:")
    names = {1.0: "TD", 2.0: "Indeterminate", 3.0: "PIGD"}
    for lbl, n in lab["subtype"].map(names).value_counts().items():
        say(f"  all labelled visits: {lbl}", f"{n:,} ({n/len(lab):.1%})")
    base = panel[(panel["YEAR"] == 0) & panel["subtype"].notna()]
    for lbl, n in base["subtype"].map(names).value_counts().items():
        say(f"  baseline (YEAR==0): {lbl}", f"{n:,} ({n/len(base):.1%})")
    say("  baseline visits with a label", f"{len(base):,}")

    print()
    say("ratio defined (labelled visits)",
        f"{int(lab['ratio'].notna().sum()):,} ({lab['ratio'].notna().mean():.3%})")
    say("ratio undefined (pigd_score == 0)",
        f"{int(lab['ratio'].isna().sum()):,} ({lab['ratio'].isna().mean():.3%})")
    und = lab[lab["ratio"].isna()]
    say("  their subtypes", dict(und["subtype"].map(names).value_counts()))
    r = lab["ratio"].dropna()
    say("ratio: median [IQR], max",
        f"{r.median():.3f} [{r.quantile(.25):.3f}-{r.quantile(.75):.3f}], {r.max():.3f}")
    say("ratio: mean (sd), skewness", f"{r.mean():.3f} ({r.std():.3f}), {r.skew():.3f}")
    say("ratio exactly zero", f"{int((r == 0).sum()):,} ({(r == 0).mean():.3%})")

    print()
    print("=" * 90)
    print("B. MISSINGNESS OF THE SUBTYPE LABEL")
    print("=" * 90)
    scfg = cfg.subtype
    part2 = loaders.load_updrs_part2(cfg)
    part3 = loaders.load_updrs_part3(cfg)
    part3_off = S.select_medication_state(part3, scfg)

    p2i = [c for c in scfg.all_items if c in part2.columns]
    p3i = [c for c in scfg.all_items if c in part3.columns]
    carry = [c for c in (scfg.pdstate_column, scfg.treatment_column, "NHY")
             if c in part3.columns]
    merged = part2[["PATNO", "EVENT_ID"] + p2i].merge(
        part3_off[["PATNO", "EVENT_ID"] + p3i + carry],
        on=["PATNO", "EVENT_ID"], how="outer")
    merged = S.deduplicate_visits(merged, scfg)

    say("merged exam rows (OFF/untreated)", f"{len(merged):,}")
    say("'unable to rate' (101) counts, top 5",
        dict(S.count_unrateable(merged, scfg).head(5)))

    # Of the cohort's unlabelled visits, how many had an exam at all?
    unlab = panel[panel["subtype"].isna()][["PATNO", "EVENT_ID"]]
    exam = merged[["PATNO", "EVENT_ID"]].assign(_has_exam=1)
    j = unlab.merge(exam, on=["PATNO", "EVENT_ID"], how="left")
    say("unlabelled cohort visits", f"{len(unlab):,}")
    say("  with an OFF/untreated exam row present",
        f"{int(j['_has_exam'].notna().sum()):,}")
    say("  with no exam row at all", f"{int(j['_has_exam'].isna().sum()):,}")

    # Among exams present but unscorable, which item is missing?
    cleaned = S.clean_items(merged, tuple(scfg.all_items), scfg)
    sc = S.compute_subscores(merged, scfg)
    present_unscored = merged.loc[sc["subtype"].isna(), ["PATNO", "EVENT_ID"]]
    pu = present_unscored.merge(unlab.assign(_in_cohort=1),
                                on=["PATNO", "EVENT_ID"], how="inner")
    say("cohort visits with an exam that could not be scored", f"{len(pu):,}")
    miss_block = cleaned.loc[sc["subtype"].isna()]
    miss_block = miss_block.loc[present_unscored.index]
    key = merged.loc[present_unscored.index, ["PATNO", "EVENT_ID"]]
    mb = miss_block.copy()
    mb[["PATNO", "EVENT_ID"]] = key
    mb = mb.merge(unlab.assign(_c=1), on=["PATNO", "EVENT_ID"], how="inner")
    item_cols = [c for c in scfg.all_items if c in mb.columns]
    counts = mb[item_cols].isna().sum().sort_values(ascending=False)
    say("  missing-item frequency, top 5", dict(counts.head(5)))
    say("  NP3PSTBL missing among these",
        f"{int(counts.get('NP3PSTBL', 0)):,} of {len(mb):,}")
    only_pstbl = (mb[item_cols].isna().sum(axis=1) == 1) & mb["NP3PSTBL"].isna()
    say("  NP3PSTBL the ONLY missing item", f"{int(only_pstbl.sum()):,}")

    # Severity of unlabelled vs labelled visits
    from scipy.stats import mannwhitneyu
    hy_l = pd.to_numeric(panel.loc[panel["subtype"].notna(), "hy"], errors="coerce").dropna()
    hy_u = pd.to_numeric(panel.loc[panel["subtype"].isna(), "hy"], errors="coerce").dropna()
    u, p = mannwhitneyu(hy_u, hy_l, alternative="greater")
    say("H&Y median: labelled vs unlabelled visits",
        f"{hy_l.median():.1f} (n={len(hy_l):,}) vs {hy_u.median():.1f} (n={len(hy_u):,})")
    say("  Mann-Whitney U, unlabelled > labelled", f"U={u:.0f}, p={p:.3e}")

    # restricted to visits where an exam existed but was unscorable
    mark = pu.assign(_unscored=1)
    pm = panel.merge(mark, on=["PATNO", "EVENT_ID"], how="left")
    hy_un = pd.to_numeric(pm.loc[pm["_unscored"] == 1, "hy"], errors="coerce").dropna()
    u2, p2 = mannwhitneyu(hy_un, hy_l, alternative="greater")
    say("H&Y median: unscorable-exam visits vs labelled",
        f"{hy_un.median():.1f} (n={len(hy_un):,}) vs {hy_l.median():.1f}")
    say("  Mann-Whitney U", f"U={u2:.0f}, p={p2:.3e}")
    for col in ["updrs3_score", "age_at_visit", "duration_yrs"]:
        a = pd.to_numeric(pm.loc[pm["_unscored"] == 1, col], errors="coerce").dropna()
        b = pd.to_numeric(panel.loc[panel["subtype"].notna(), col], errors="coerce").dropna()
        say(f"  {col}: unscorable vs labelled (median)",
            f"{a.median():.1f} vs {b.median():.1f}")

    print()
    print("=" * 90)
    print("C. TRANSITIONS (recomputed)")
    print("=" * 90)
    from ppmi_subtype import transitions as T
    pairs = T.consecutive_pairs(panel)
    say("consecutive labelled pairs", f"{len(pairs):,}")
    say("patients contributing a pair", f"{pairs['PATNO'].nunique():,}")
    say("interval years: median [IQR], mean",
        f"{pairs['interval_years'].median():.2f} "
        f"[{pairs['interval_years'].quantile(.25):.2f}-"
        f"{pairs['interval_years'].quantile(.75):.2f}], "
        f"{pairs['interval_years'].mean():.2f}")
    counts = T.transition_matrix(pairs, states=[1.0, 2.0, 3.0], normalize=False)
    probs = T.transition_matrix(pairs, states=[1.0, 2.0, 3.0])
    counts.index = counts.columns = probs.index = probs.columns = list(names.values())
    print("\ncounts:\n", counts.to_string())
    print("\nprobabilities:\n", probs.round(4).to_string())

    rev = T.reversion_after_change(panel)
    say("\nobserved changes with a following visit", f"{len(rev):,}")
    say("  reversed at the next visit",
        f"{int(rev['reverted'].sum()):,} ({rev['reverted'].mean():.3%})")
    print(T.reversion_summary(rev, names).to_string(index=False))

    # does a change persist?
    say("\nTD->PIGD count", f"{int(counts.loc['TD','PIGD']):,}")
    say("PIGD->TD count", f"{int(counts.loc['PIGD','TD']):,}")

    print()
    print("=" * 90)
    print("D. TIME TO PIGD (recomputed)")
    print("=" * 90)
    for k, tag in [(1, "naive"), (2, "sustained")]:
        ent = T.first_entry(panel, 3.0, min_consecutive=k)
        risk = ent[~ent["starts_in_state"]]
        km = T.kaplan_meier(risk["time"], risk["converted"])
        def inc(y):
            s = km[km["time"] <= y]["survival"]
            return 1 - (s.iloc[-1] if len(s) else 1.0)
        say(f"{tag}: at risk / converted",
            f"{len(risk):,} / {int(risk['converted'].sum()):,} "
            f"({risk['converted'].mean():.1%})")
        say(f"{tag}: KM cumulative incidence y2 / y5",
            f"{inc(2):.3%} / {inc(5):.3%}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
