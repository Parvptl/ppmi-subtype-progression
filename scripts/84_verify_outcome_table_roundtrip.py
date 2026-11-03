#!/usr/bin/env python3
"""Verify that a CV run's saved outcome_table.csv round-trips (Stage 12, A2').

`scripts/64_clinical_profile_cv.py` analyses an in-memory table and separately
saves it. Those used to disagree: the file was written with `.round(4)`, which
is destructive for rate coordinates of order 1e-3, so recomputing from the
saved file gave different answers than the run reported -- a trap the Stage 12
audit fell into. The file is now saved at full precision.

This recomputes the eight outcome contrasts from the SAVED table and checks
they reproduce the run's own saved `h3_component_vs_subtype.csv`. If that holds,
the artifact is reusable; if it does not, the saved table is still lossy.

usage: 84_verify_outcome_table_roundtrip.py RUN_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import clinical_profile as cp  # noqa: E402

COLS = ["auc_a", "auc_b", "delta", "delta_ci_low", "delta_ci_high"]
SEED, N_BOOT = 42, 2000


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    run = Path(sys.argv[1])

    table = pd.read_csv(run / "outcome_table.csv")
    assoc = pd.read_csv(run / "outcome_associations.csv")
    saved = pd.read_csv(run / "h3_component_vs_subtype.csv")

    rows = []
    for oc in assoc["outcome"].unique():
        if assoc[assoc["outcome"] == oc]["auc"].notna().any():
            for kind in ("rate", "level"):
                rows.append(cp.best_component_vs_subtype(
                    table, assoc, oc, kind=kind, seed=SEED, n_boot=N_BOOT))
    recomputed = pd.DataFrame([r for r in rows if r.get("n")]).round(4)

    key = ["outcome", "kind"]
    m = saved.merge(recomputed, on=key, suffixes=("_saved", "_recomputed"))
    print(f"run: {run.name}\n")
    print(f"{'outcome':<22} {'kind':<6} " + " ".join(f"{c:>16}" for c in COLS))
    bad = []
    for _, r in m.iterrows():
        diffs = [abs(r[f"{c}_saved"] - r[f"{c}_recomputed"]) for c in COLS]
        print(f"{r['outcome']:<22} {r['kind']:<6} "
              + " ".join(f"{d:>16.2e}" for d in diffs))
        if max(diffs) > 1e-9:
            bad.append((r["outcome"], r["kind"], max(diffs)))

    # Does the saved table still carry the rate coordinates at usable precision?
    rate_cols = [c for c in table.columns if c.startswith("rate_")]
    worst = min((table[c].abs().replace(0, pd.NA).dropna().min()
                 for c in rate_cols if table[c].abs().sum() > 0), default=float("nan"))
    zeros = {c: int((table[c] == 0).sum()) for c in rate_cols}
    print(f"\nsmallest non-zero rate magnitude in the saved table: {worst:.3g}")
    print(f"exact zeros per rate column: {zeros}")

    print("\nround-trip:", "PASS — the saved table reproduces the run"
          if not bad else f"FAIL — {len(bad)} contrasts differ: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
