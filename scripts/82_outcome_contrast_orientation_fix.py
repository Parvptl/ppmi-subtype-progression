#!/usr/bin/env python3
"""Recompute the outcome-discrimination contrasts with a consistently
oriented reference (Stage 12 finding A1).

SUPERSEDED -- do not use its numbers. This script reads the CV run's saved
`outcome_table.csv`, which `scripts/64_clinical_profile_cv.py` writes with
`.round(4)` while computing on the full-precision in-memory table. Four
decimals is destructive for rate coordinates of order 1e-3, so the contrasts
below are slightly wrong on those columns (`mci`/`rate` by -0.0020). It is kept
because it is what isolated the A1 fix from everything else, but the
authoritative corrected table is
`experiments/20261103T213859Z_a2_clinical_profile_cv_rerun/h3_component_vs_subtype.csv`,
produced by a full rerun at HEAD with the fix in place. See the Stage 12 audit,
A2 and A2'.

`clinical_profile.best_component_vs_subtype` gave the comparator component a
one-time orientation flip so that a higher score means more outcome, but left
the subtype-label reference unoriented. On any outcome whose label AUC falls
below chance the comparator is therefore allowed its favourable direction
while the label is not, and the reported difference is inflated.

Both arms are computed **here, under one version of the library**, differing
only in whether the reference is oriented. That isolates the change: comparing
the fixed code against the figures saved by the October CV run would also pick
up any unrelated library drift since, and in fact does -- the drift is reported
below as information, not as a check.

Only `motor_complications` should move: its label AUC is 0.4963 raw against
0.5037 oriented. The other three outcomes have label AUCs above chance, where
the two conventions coincide.

Refits nothing; reads the anchor table and associations saved by the CV run.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import clinical_profile as cp  # noqa: E402

SRC = ROOT / "experiments" / "20261020T180515Z_clinical_profile_cv"
OUTCOMES = ["hy_ge_3", "mci", "motor_complications", "cognitive_dx"]
KINDS = ["rate", "level"]
SEED, N_BOOT = 42, 2000
UNAFFECTED = ["hy_ge_3", "mci", "cognitive_dx"]


def contrast(table: pd.DataFrame, assoc: pd.DataFrame, outcome: str,
             kind: str, orient_ref: bool) -> dict:
    """The library's contrast, with the reference orientation under a flag.

    Mirrors `clinical_profile.best_component_vs_subtype` exactly; `orient_ref`
    False reproduces the behaviour before the fix.
    """
    sub = assoc[(assoc["outcome"] == outcome)
                & assoc["score"].str.startswith(f"{kind}_")]
    best = sub.loc[sub["auc_abs"].idxmax(), "score"]

    y = table[outcome].to_numpy(dtype=float)
    ok = np.isfinite(y)
    pos = y[ok].astype(bool)

    score = table[best].to_numpy(float)
    if cp.auc(score[ok], pos) < 0.5:
        score = -score

    ref = table["subtype_at_anchor"].map(cp._SUBTYPE_SCORE).to_numpy(float)
    if orient_ref and cp.auc(ref[ok], pos) < 0.5:
        ref = -ref

    out = cp.bootstrap_auc_difference(score[ok], ref[ok], pos,
                                      seed=SEED, n_boot=N_BOOT)
    return {"outcome": outcome, "kind": kind, "best_component": best,
            "selection_is_optimistic": True, **out}


def arm(table, assoc, orient_ref: bool) -> pd.DataFrame:
    rows = [contrast(table, assoc, o, k, orient_ref)
            for o in OUTCOMES for k in KINDS]
    df = pd.DataFrame(rows)
    num = ["auc_a", "auc_b", "delta", "delta_ci_low", "delta_ci_high"]
    df[num] = df[num].round(4)
    return df


def main() -> int:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "experiments" / f"{stamp}_a1_outcome_contrast_orientation"
    out.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(SRC / "outcome_table.csv")
    assoc = pd.read_csv(SRC / "outcome_associations.csv")

    before = arm(table, assoc, orient_ref=False)
    after = arm(table, assoc, orient_ref=True)
    after.to_csv(out / "corrected_contrasts.csv", index=False)

    key = ["outcome", "kind"]
    m = before.merge(after, on=key, suffixes=("_before", "_after"))
    m["delta_shift"] = (m["delta_after"] - m["delta_before"]).round(4)
    m.to_csv(out / "before_after.csv", index=False)

    # Drift of the pre-fix arm against the figures the October run saved.
    # Reported, not asserted: unrelated library changes since then land here.
    saved = pd.read_csv(SRC / "h3_component_vs_subtype.csv")
    d = saved.merge(before, on=key, suffixes=("_saved", "_prefix"))
    d["drift"] = (d["delta_prefix"] - d["delta_saved"]).round(4)
    d[key + ["delta_saved", "delta_prefix", "drift"]].to_csv(
        out / "drift_vs_october_run.csv", index=False)

    checks: dict[str, bool] = {}
    for outcome in UNAFFECTED:
        s = m[m["outcome"] == outcome]
        checks[f"{outcome} identical under the fix"] = bool(
            (s["delta_before"] == s["delta_after"]).all()
            and (s["delta_ci_low_before"] == s["delta_ci_low_after"]).all()
            and (s["delta_ci_high_before"] == s["delta_ci_high_after"]).all())

    mc = m[(m["outcome"] == "motor_complications") & (m["kind"] == "level")].iloc[0]
    checks["motor_complications reference moved to its oriented value"] = bool(
        abs(mc["auc_b_after"] - 0.5037) < 5e-4)
    checks["motor_complications difference shrank"] = bool(
        mc["delta_after"] < mc["delta_before"])
    checks["motor_complications interval still includes zero"] = bool(
        mc["delta_ci_low_after"] < 0 < mc["delta_ci_high_after"])
    checks["only motor_complications moved"] = bool(
        (m.loc[m["outcome"] != "motor_complications", "delta_shift"] == 0).all()
        and (m.loc[m["outcome"] == "motor_complications", "delta_shift"] != 0).all())
    checks["every rate interval still includes zero"] = bool(
        ((after[after["kind"] == "rate"]["delta_ci_low"] < 0)
         & (after[after["kind"] == "rate"]["delta_ci_high"] > 0)).all())
    checks["three of four level intervals still exclude zero"] = bool(
        (after[after["kind"] == "level"]["delta_ci_low"] > 0).sum() == 3)

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True,
                                check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    (out / "manifest.json").write_text(json.dumps({
        "generated": stamp, "git_commit": commit, "source_run": SRC.name,
        "seed": SEED, "n_boot": N_BOOT, "refits": 0, "checks": checks,
    }, indent=1) + "\n")

    print(f"wrote {out.relative_to(ROOT)}\n")
    print(m[["outcome", "kind", "best_component_after", "auc_b_before",
             "auc_b_after", "delta_before", "delta_after", "delta_shift"]]
          .to_string(index=False))
    print("\ndrift of the pre-fix arm vs the October saved run (information):")
    print(d[key + ["delta_saved", "delta_prefix", "drift"]].to_string(index=False))
    print("\nchecks:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
