"""Follow-ups to the missingness audit: severity of incomplete exams, and the
sensitivity of the label to the strict all-items rule.

    python scripts/_verify_missingness_audit2.py
"""

from __future__ import annotations

import dataclasses
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
    part3 = S.select_medication_state(loaders.load_updrs_part3(cfg), scfg)
    p3 = S.deduplicate_visits(part3, scfg)
    p2 = part2.drop_duplicates(KEY, keep="first")

    items = tuple(scfg.tremor_items + scfg.pigd_items)
    p3_items = tuple(c for c in items if c in p3.columns)
    p2_items = tuple(c for c in items if c in p2.columns)
    p3_clean = S.clean_items(p3, p3_items, scfg)
    inc_mask = p3_clean.isna().any(axis=1).to_numpy()

    # ---- severity, with NHY cleaned to its valid 0-5 range ----------------
    print("SEVERITY OF INCOMPLETE VERSUS COMPLETE OFF PART III EXAMS")
    hy = pd.to_numeric(p3["NHY"], errors="coerce")
    say("raw NHY value counts (top)", dict(hy.value_counts().head(8)))
    hy = hy.where((hy >= 0) & (hy <= 5))
    a = hy[inc_mask].dropna()
    b = hy[~inc_mask].dropna()
    u, pval = mannwhitneyu(a, b, alternative="greater")
    say("H&Y median: incomplete vs complete",
        f"{a.median():.1f} (n={len(a):,}) vs {b.median():.1f} (n={len(b):,})")
    say("  mean", f"{a.mean():.2f} vs {b.mean():.2f}")
    say("  Mann-Whitney (incomplete > complete)", f"U={u:.0f}, p={pval:.3e}")
    say("  share at H&Y >= 3", f"{(a >= 3).mean():.1%} vs {(b >= 3).mean():.1%}")

    # restricted to the analysis cohort's own visits
    cohort_keys = panel[KEY].assign(_c=1)
    k3 = p3[KEY].copy()
    k3["incomplete"] = inc_mask
    k3["NHY_clean"] = hy.to_numpy()
    k3 = k3.merge(cohort_keys, on=KEY, how="inner")
    a = k3.loc[k3["incomplete"], "NHY_clean"].dropna()
    b = k3.loc[~k3["incomplete"], "NHY_clean"].dropna()
    u, pval = mannwhitneyu(a, b, alternative="greater")
    print()
    say("Restricted to the analysis cohort's visits:", "")
    say("  H&Y median: incomplete vs complete",
        f"{a.median():.1f} (n={len(a):,}) vs {b.median():.1f} (n={len(b):,})")
    say("  mean", f"{a.mean():.2f} vs {b.mean():.2f}")
    say("  Mann-Whitney (incomplete > complete)", f"U={u:.0f}, p={pval:.3e}")
    say("  share at H&Y >= 3", f"{(a >= 3).mean():.1%} vs {(b >= 3).mean():.1%}")

    # ---- sensitivity to the strict all-items rule -------------------------
    print()
    print("SENSITIVITY: ALLOWING ONE MISSING ITEM PER SUBSCORE")
    merged = p2[list(KEY) + list(p2_items)].merge(
        p3[list(KEY) + list(p3_items)], on=KEY, how="outer")
    strict = S.compute_subscores(merged, scfg)
    relaxed_cfg = dataclasses.replace(
        scfg, max_missing_tremor_items=1, max_missing_pigd_items=1)
    say("relaxed cfg max_missing (tremor, pigd)",
        (relaxed_cfg.max_missing_tremor_items, relaxed_cfg.max_missing_pigd_items))
    relaxed = S.compute_subscores(merged, relaxed_cfg)

    cmp = merged[KEY].assign(strict=strict["subtype"].to_numpy(),
                             relaxed=relaxed["subtype"].to_numpy())
    joined = panel[KEY].merge(cmp, on=KEY, how="left")
    say("cohort visits labelled, strict", f"{int(joined['strict'].notna().sum()):,}")
    say("cohort visits labelled, relaxed", f"{int(joined['relaxed'].notna().sum()):,}")
    say("additional visits recovered",
        f"{int(joined['relaxed'].notna().sum() - joined['strict'].notna().sum()):,}")
    both = joined[joined["strict"].notna() & joined["relaxed"].notna()]
    say("existing labels changed", f"{int((both['strict'] != both['relaxed']).sum()):,}")

    # ---- band-boundary sensitivity ----------------------------------------
    print()
    print("SENSITIVITY: WHICH SIDE OF THE THRESHOLD THE BAND IS CLOSED ON")
    inner_cfg = dataclasses.replace(scfg, band_boundaries="inner")
    inner = S.compute_subscores(merged, inner_cfg)
    cmp2 = merged[KEY].assign(outer=strict["subtype"].to_numpy(),
                              inner=inner["subtype"].to_numpy())
    j2 = panel[KEY].merge(cmp2, on=KEY, how="left")
    j2 = j2[j2["outer"].notna() & j2["inner"].notna()]
    say("labels differing between outer and inner closure",
        f"{int((j2['outer'] != j2['inner']).sum()):,} of {len(j2):,}")

    # ---- how close is the ratio to a threshold? ---------------------------
    print()
    print("HOW MANY LABELS SIT NEAR A THRESHOLD")
    lab = panel[panel["subtype"].notna()]
    r = lab["ratio"].dropna()
    for d in (0.05, 0.10, 0.25):
        near = ((r - 1.15).abs() <= d) | ((r - 0.90).abs() <= d)
        say(f"  within {d:.2f} of either threshold",
            f"{int(near.sum()):,} of {len(r):,} ({near.mean():.1%})")
    say("  within the Indeterminate band (0.90, 1.15)",
        f"{int(((r > 0.90) & (r < 1.15)).sum()):,}")

    # ---- within-patient variability of the ratio --------------------------
    print()
    print("WITHIN-PATIENT VARIABILITY OF THE RATIO")
    fin = panel[panel["ratio"].notna()]
    g = fin.groupby("PATNO")["ratio"]
    multi = g.size()[g.size() >= 2].index
    sub = fin[fin["PATNO"].isin(multi)]
    within_sd = sub.groupby("PATNO")["ratio"].std()
    say("patients with >=2 finite ratios", f"{len(multi):,}")
    say("within-patient SD of the ratio: mean / median",
        f"{within_sd.mean():.3f} / {within_sd.median():.3f}")
    say("Indeterminate band width", "0.25")
    d = sub.sort_values(["PATNO", "years_from_baseline"]).groupby("PATNO")["ratio"].diff().dropna()
    say("absolute change between consecutive finite ratios: median",
        f"{d.abs().median():.3f}")
    # ICC from a one-way random-effects decomposition
    gm = sub["ratio"].mean()
    pm = sub.groupby("PATNO")["ratio"].mean()
    n_i = sub.groupby("PATNO")["ratio"].size()
    k = len(pm)
    N = len(sub)
    ssb = float((n_i * (pm - gm) ** 2).sum())
    ssw = float(((sub["ratio"] - sub["PATNO"].map(pm)) ** 2).sum())
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    n0 = (N - float((n_i ** 2).sum()) / N) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    say("one-way ICC of the ratio", f"{icc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
