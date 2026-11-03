#!/usr/bin/env python3
"""Diff two clinical-profile CV runs file by file (Stage 12 finding A2).

The 2026-10-20 run recorded commit 3a860ca, at which
`src/ppmi_subtype/clinical_profile.py` did not yet exist -- the module was
first committed five days later in 19bf8aa. That run therefore executed
uncommitted code that was revised before being committed, so nothing it saved
is reproducible from any committed state.

This compares it against a re-run at the current commit and reports, per file
and per column, how far each quantity moved. Columns that move at all are
listed with their largest shift, so the manuscript's exposure is explicit
rather than assumed.

usage: 83_compare_clinical_profile_runs.py OLD_RUN NEW_RUN
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Quantities the manuscript quotes from this run, by file.
MANUSCRIPT = {
    "representation_dimensionality.csv": "R7 dimensionality (7 PCs, PC1 shares)",
    "outcome_associations.csv": "R7 outcome AUCs",
    "h3_component_vs_subtype.csv": "R7 outcome contrasts (also A1)",
    "contrasts.csv": "R7 profile-prediction log-density deltas",
    "derived_label_vs_subtype_history.csv": "R5 two-class latent contrast",
    "model_comparison.csv": "R5/R7 model comparison",
}
TOL = 5e-5


def key_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if df[c].dtype == object or c in ("n", "n_events")]


def compare(old: Path, new: Path) -> int:
    names = sorted({p.name for p in old.glob("*.csv")} & {p.name for p in new.glob("*.csv")})
    worst: list[tuple] = []
    print(f"old: {old.name}\nnew: {new.name}\n")
    print(f"{'file':<44} {'rows':<11} {'max |delta|':<12} moved columns")
    for name in names:
        a = pd.read_csv(old / name)
        b = pd.read_csv(new / name)
        if a.shape != b.shape:
            print(f"{name:<44} SHAPE {a.shape} vs {b.shape}")
            worst.append((name, float("inf"), ["shape"]))
            continue
        num = [c for c in a.columns
               if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
        moved, mx = [], 0.0
        for c in num:
            d = np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))
            d = d[np.isfinite(d)]
            if len(d) and d.max() > TOL:
                moved.append(f"{c}({d.max():.4g})")
                mx = max(mx, float(d.max()))
        tag = "same rows" if a.shape == b.shape else ""
        print(f"{name:<44} {tag:<11} {mx:<12.4g} {', '.join(moved) if moved else '-'}")
        if moved:
            worst.append((name, mx, moved))

    print("\n=== manuscript exposure ===")
    any_exposed = False
    for name, what in MANUSCRIPT.items():
        hit = next((w for w in worst if w[0] == name), None)
        if not (old / name).exists() or not (new / name).exists():
            print(f"  n/a      {name:<42} {what}  (absent from one run)")
        elif hit:
            any_exposed = True
            print(f"  MOVED    {name:<42} {what}  max |delta| {hit[1]:.4g}")
        else:
            print(f"  stable   {name:<42} {what}")
    print("\nAny manuscript quantity moved:", any_exposed)
    return 0


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    return compare(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    sys.exit(main())
