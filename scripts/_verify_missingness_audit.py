"""Why visits go unlabelled, and whether that loss is informative.

Recomputes the missing-label taxonomy and the severity comparison between
exams that could be scored and exams that could not.

    python scripts/_verify_missingness_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype import loaders, subtype as S  # noqa: E402

KEY = ["PATNO", "EVENT_ID"]


def say(label: str, value) -> None:
    print(f"{label:<64} {value}")


def main() -> int:
    cfg = load_config()
    scfg = cfg.subtype
    panel = pd.read_csv(ROOT / "outputs" / "visit_panel.csv", low_memory=False)

    part2 = loaders.load_updrs_part2(cfg)
    part3_all = loaders.load_updrs_part3(cfg)
    part3 = S.select_medication_state(part3_all, scfg)

    p2_items = tuple(c for c in scfg.tremor_items + scfg.pigd_items if c in part2.columns)
    p3_items = tuple(c for c in scfg.tremor_items + scfg.pigd_items if c in part3.columns)

    say("Part III rows (all medication states)", f"{len(part3_all):,}")
    say("Part III rows after OFF/untreated selection", f"{len(part3):,}")
    say("Part II rows", f"{len(part2):,}")

    # Deduplicate each part on its own, so "a Part III OFF exam exists" is a
    # clean yes/no per patient-visit.
    p3 = S.deduplicate_visits(part3, scfg)
    p2 = part2.drop_duplicates(KEY, keep="first")

    p3_clean = S.clean_items(p3, p3_items, scfg)
    p2_clean = S.clean_items(p2, p2_items, scfg)
    p3_flag = p3[KEY].assign(
        p3_present=1,
        p3_n_missing=p3_clean.isna().sum(axis=1).to_numpy(),
    )
    p2_flag = p2[KEY].assign(
        p2_present=1,
        p2_n_missing=p2_clean.isna().sum(axis=1).to_numpy(),
    )

    unlab = panel.loc[panel["subtype"].isna(), KEY].copy()
    j = unlab.merge(p3_flag, on=KEY, how="left").merge(p2_flag, on=KEY, how="left")

    no_p3 = j["p3_present"].isna()
    p3_incomplete = (j["p3_present"] == 1) & (j["p3_n_missing"] > 0)
    p3_ok_p2_bad = (j["p3_present"] == 1) & (j["p3_n_missing"] == 0) & (
        j["p2_present"].isna() | (j["p2_n_missing"] > 0))
    other = ~(no_p3 | p3_incomplete | p3_ok_p2_bad)

    print()
    print("WHY A COHORT VISIT HAS NO SUBTYPE LABEL  (n = %d)" % len(j))
    say("  no OFF/untreated Part III exam recorded",
        f"{int(no_p3.sum()):,} ({no_p3.mean():.1%})")
    say("  OFF Part III recorded but item set incomplete",
        f"{int(p3_incomplete.sum()):,} ({p3_incomplete.mean():.1%})")
    say("  Part III complete, Part II missing or incomplete",
        f"{int(p3_ok_p2_bad.sum()):,} ({p3_ok_p2_bad.mean():.1%})")
    say("  unaccounted", f"{int(other.sum()):,}")

    # --- among incomplete OFF exams, which item is responsible? -------------
    print()
    print("INCOMPLETE OFF PART III EXAMS  (all such exams in the file, not only")
    print("those matched to a cohort visit)")
    inc_mask = p3_clean.isna().any(axis=1).to_numpy()
    inc = p3.loc[inc_mask]
    inc_clean = p3_clean.loc[inc_mask]
    say("  incomplete OFF Part III exams", f"{len(inc):,}")
    counts = inc_clean.isna().sum().sort_values(ascending=False)
    say("  NP3PSTBL unavailable in", f"{int(counts['NP3PSTBL']):,} of {len(inc):,} "
                                    f"({counts['NP3PSTBL']/len(inc):.1%})")
    print("  missing-item frequency:")
    for item, n in counts.items():
        if n:
            print(f"      {item:<10} {int(n):,}")
    raw = pd.to_numeric(inc["NP3PSTBL"], errors="coerce")
    say("  of those, explicitly coded 101 (Unable to Rate)",
        f"{int((raw == 101).sum()):,}")
    say("  NP3PSTBL the only unavailable item",
        f"{int(((inc_clean.isna().sum(axis=1) == 1) & inc_clean['NP3PSTBL'].isna()).sum()):,}")

    # --- severity: incomplete vs complete OFF exams -------------------------
    print()
    print("SEVERITY OF INCOMPLETE VERSUS COMPLETE OFF EXAMS")
    if "NHY" in p3.columns:
        hy = pd.to_numeric(p3["NHY"], errors="coerce")
        a = hy[inc_mask].dropna()
        b = hy[~inc_mask].dropna()
        u, pval = mannwhitneyu(a, b, alternative="greater")
        say("  Hoehn & Yahr (NHY) median: incomplete vs complete",
            f"{a.median():.1f} (n={len(a):,}) vs {b.median():.1f} (n={len(b):,})")
        say("    Mann-Whitney U (incomplete > complete)", f"U={u:.0f}, p={pval:.3e}")
        say("    mean", f"{a.mean():.2f} vs {b.mean():.2f}")

    # --- label availability across follow-up -------------------------------
    print()
    print("LABEL AVAILABILITY ACROSS FOLLOW-UP")
    g = panel.groupby("YEAR").agg(
        n_visits=("subtype", "size"),
        n_labelled=("subtype", lambda s: int(s.notna().sum())))
    g["share_labelled"] = g["n_labelled"] / g["n_visits"]
    print(g.to_string())

    # --- within-patient: does an unlabelled visit follow a worse state? ----
    print()
    print("WHAT PRECEDES AN UNLABELLED VISIT")
    p = panel.sort_values(["PATNO", "years_from_baseline"]).copy()
    p["is_unlab"] = p["subtype"].isna()
    p["prev_pigd"] = p.groupby("PATNO")["pigd_score"].shift()
    p["prev_tremor"] = p.groupby("PATNO")["tremor_score"].shift()
    nxt_unlab = p[p["prev_pigd"].notna()]
    a = nxt_unlab.loc[nxt_unlab["is_unlab"], "prev_pigd"]
    b = nxt_unlab.loc[~nxt_unlab["is_unlab"], "prev_pigd"]
    u, pval = mannwhitneyu(a, b, alternative="greater")
    say("  previous-visit PIGD subscore, mean (unlab vs lab)",
        f"{a.mean():.3f} (n={len(a):,}) vs {b.mean():.3f} (n={len(b):,})")
    say("    Mann-Whitney p (unlab > lab)", f"{pval:.3e}")
    a2 = nxt_unlab.loc[nxt_unlab["is_unlab"], "prev_tremor"]
    b2 = nxt_unlab.loc[~nxt_unlab["is_unlab"], "prev_tremor"]
    u2, pval2 = mannwhitneyu(a2, b2, alternative="greater")
    say("  previous-visit tremor subscore, mean (unlab vs lab)",
        f"{a2.mean():.3f} vs {b2.mean():.3f}")
    say("    Mann-Whitney p (unlab > lab)", f"{pval2:.3e}")

    # --- sensitivity: allow one missing item per subscore -------------------
    print()
    print("SENSITIVITY: ALLOWING ONE MISSING ITEM PER SUBSCORE")
    merged = p2[list(KEY) + list(p2_items)].merge(
        p3[list(KEY) + list(p3_items)], on=KEY, how="outer")
    strict = S.compute_subscores(merged, scfg)
    import copy
    relaxed_cfg = copy.deepcopy(scfg)
    object.__setattr__(relaxed_cfg, "max_missing_tremor_items", 1) if hasattr(
        relaxed_cfg, "__dataclass_fields__") else None
    try:
        relaxed_cfg.max_missing_tremor_items = 1
        relaxed_cfg.max_missing_pigd_items = 1
    except Exception:
        pass
    relaxed = S.compute_subscores(merged, relaxed_cfg)
    base = panel[KEY].merge(
        merged[KEY].assign(strict=strict["subtype"].to_numpy(),
                           relaxed=relaxed["subtype"].to_numpy()),
        on=KEY, how="left")
    say("  cohort visits labelled, strict rule", f"{int(base['strict'].notna().sum()):,}")
    say("  cohort visits labelled, one missing item allowed",
        f"{int(base['relaxed'].notna().sum()):,}")
    both = base[base["strict"].notna() & base["relaxed"].notna()]
    say("  labels changed among visits labelled under both",
        f"{int((both['strict'] != both['relaxed']).sum()):,} of {len(both):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
