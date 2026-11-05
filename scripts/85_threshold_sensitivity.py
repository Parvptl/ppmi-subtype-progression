#!/usr/bin/env python3
"""O-4 -- threshold sensitivity of the motor-subtype label.

The subtype is a thresholded ratio. This quantifies how much of the label
survives a perturbation of the thresholds smaller than the measurement's own
resolution, by comparing the frozen baseline panel against a panel rebuilt
under `config/config_narrow_0.91_1.13.yaml`.

**The perturbation is the one already committed to the repository**, not a
range chosen here:

    baseline   PIGD <= 0.90   0.90 <  I <  1.15   TD >= 1.15   (outer)
    narrow     PIGD <  0.91   0.91 <= I <= 1.13   TD >  1.13   (inner)

Note it changes three things, not two: both numeric thresholds *and* the side
each band boundary is closed on. The manuscript must not describe it as a
numeric shift alone.

No model is involved. This is a change to the classification rule applied to
identical subscores, so the two panels differ only in the label column; every
visit, subscore and cohort decision is shared. That is what makes it a clean
sensitivity analysis and what makes "which rule is right?" the wrong question:
both are rules, neither is adjudicated truth.

Prerequisite: run `scripts/01_build_panel.py` with
`PPMI_CONFIG=config/config_narrow_0.91_1.13.yaml` first; this script only
compares and never rebuilds.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ppmi_subtype.plots import GRID, INK_MUTED, _finish, _style  # noqa: E402

ACCENT = "#1f4e79"
WARM = "#c2552c"
MUTED = "#8c8c88"
BASE = ROOT / "outputs"
NARROW = ROOT / "experiments" / "threshold_narrow_0.91_1.13"
CLASSES = ["TD", "Indeterminate", "PIGD"]
KEY = ["PATNO", "EVENT_ID"]

# Figures the Stage 1 audit recorded from an unsaved scratchpad recomputation.
# This run either reproduces them or the discrepancy is the finding.
AUDIT = {
    "relabelled": 298, "labelled": 4441, "share": 0.0671,
    "indeterminate_before": 441, "indeterminate_after": 143,
    "indeterminate_shrinkage": 0.676,
    "ind_to_pigd": 198, "ind_to_td": 100,
    "td_before": 2891, "td_after": 2991,
    "pigd_before": 1109, "pigd_after": 1307,
    "agreement_2class": 0.9774,
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()



def figure_attainable_ratios(lab: pd.DataFrame, base: pd.DataFrame,
                             out: Path, checks: dict) -> pd.DataFrame:
    """Why the band is fragile: the ratio can only land on a few values.

    The ratio is a mean of 11 integer items over a mean of 5, so inside a band
    0.25 wide very few values are reachable. Plotting the attainable values
    against both threshold pairs shows the mechanism directly: the whole
    relabelling is two spikes crossing a line that moved by 0.01 and 0.0064.
    """
    ind = base[base["subtype_name"] == "Indeterminate"]
    v = ind["ratio"].dropna().round(4)
    vc = v.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    ax.vlines(vc.index, 0, vc.values, color=MUTED, lw=1.6, zorder=3)
    ax.plot(vc.index, vc.values, "o", ms=5.0, color=ACCENT, zorder=4)

    for x, lo, hi, lab_txt in ((0.90, None, None, "baseline 0.90"),
                               (1.15, None, None, "baseline 1.15")):
        ax.axvline(x, color=INK_MUTED, lw=1.2, ls="--", zorder=2)
    for x in (0.91, 1.13):
        ax.axvline(x, color=WARM, lw=1.3, ls=":", zorder=2)

    #  Each label sits on the far side of its own spike, away from the pair of
    #  threshold lines it would otherwise overlap.
    top = vc.sort_values(ascending=False).head(2)
    for x, cnt in top.items():
        below = x < 0.91
        dest = "PIGD" if below else "TD"
        ax.annotate(f"{x:g}\n{cnt} visits\n\u2192 {dest}", (x, cnt),
                    textcoords="offset points",
                    xytext=(14 if below else -14, 2),
                    ha="left" if below else "right", va="bottom",
                    fontsize=8.0, color=ACCENT, linespacing=1.3)

    ax.set_xlabel("Tremor / PIGD ratio at indeterminate visits", fontsize=9.4)
    ax.set_ylabel("Visits", fontsize=9.4)
    ax.set_title("Few ratio values are reachable inside the indeterminate band",
                 fontsize=10.6, loc="left")
    ax.set_ylim(0, vc.max() * 1.42)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    ax.plot([], [], color=INK_MUTED, lw=1.2, ls="--", label="baseline 0.90 / 1.15")
    ax.plot([], [], color=WARM, lw=1.3, ls=":", label="narrow 0.91 / 1.13")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    p = _finish(fig, out / "fig_attainable_ratios.png")
    print(f"  {p.name}")
    tidy = vc.rename_axis("ratio").reset_index(name="n_visits")
    tidy.to_csv(out / "attainable_ratios.csv", index=False)

    checks["indeterminate ratios take at most 12 distinct values"] = bool(
        vc.size <= 12)
    checks["the two dominant values account for all 298 relabelled visits"] = bool(
        int(top.sum()) == 298)
    checks["0.9091 falls below the narrow PIGD threshold of 0.91"] = bool(
        float(top.index.min()) < 0.91)
    checks["1.1364 rises above the narrow TD threshold of 1.13"] = bool(
        float(top.index.max()) > 1.13)
    return tidy


def main() -> int:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "experiments" / f"{stamp}_o4_threshold_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    checks: dict[str, bool] = {}

    b = pd.read_csv(BASE / "visit_panel.csv", low_memory=False)
    n = pd.read_csv(NARROW / "visit_panel.csv", low_memory=False)

    # The two panels must be the same visits; only the rule differs.
    checks["both panels have the same row count"] = len(b) == len(n)
    m = b[KEY + ["subtype_name", "visit_index"]].merge(
        n[KEY + ["subtype_name"]], on=KEY, suffixes=("_base", "_narrow"),
        validate="one_to_one")
    checks["every visit joins one-to-one"] = len(m) == len(b)

    # Subscores are untouched by a threshold change; confirm rather than assume.
    for col in ("tremor_score", "pigd_score"):
        if col in b.columns and col in n.columns:
            same = np.allclose(b[col].to_numpy(float), n[col].to_numpy(float),
                               equal_nan=True)
            checks[f"{col} identical across the two panels"] = bool(same)

    lab = m.dropna(subset=["subtype_name_base", "subtype_name_narrow"]).copy()
    both_unlabelled = (m["subtype_name_base"].isna()
                       & m["subtype_name_narrow"].isna()).sum()
    checks["the labelled set is unchanged by the perturbation"] = bool(
        m["subtype_name_base"].isna().sum() == m["subtype_name_narrow"].isna().sum()
        == both_unlabelled)

    n_lab = len(lab)
    moved = lab["subtype_name_base"] != lab["subtype_name_narrow"]
    n_moved = int(moved.sum())

    # ---- crosstab, the primary output
    ct = pd.crosstab(lab["subtype_name_base"], lab["subtype_name_narrow"]) \
           .reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    ct.to_csv(out / "label_crosstab.csv")

    # ---- per-class movement
    rows = []
    for c in CLASSES:
        before = int((lab["subtype_name_base"] == c).sum())
        after = int((lab["subtype_name_narrow"] == c).sum())
        left = int(((lab["subtype_name_base"] == c) & moved).sum())
        rows.append({"class": c, "n_before": before, "n_after": after,
                     "delta": after - before,
                     "pct_change": (after - before) / before if before else np.nan,
                     "n_leaving": left,
                     "share_of_class_leaving": left / before if before else np.nan})
    per_class = pd.DataFrame(rows)
    per_class.to_csv(out / "per_class_movement.csv", index=False)

    # ---- patients affected: the denominator a visit-level share hides
    pat = lab.groupby("PATNO").apply(
        lambda g: pd.Series({
            "n_labelled": len(g),
            "n_moved": int((g["subtype_name_base"] != g["subtype_name_narrow"]).sum()),
        }), include_groups=False).reset_index()
    pat["any_moved"] = pat["n_moved"] > 0
    pat.to_csv(out / "per_patient_movement.csv", index=False)

    # ---- agreement against PPMI's own derivations, both rules
    def parse_agreement(path: Path) -> dict:
        txt = path.read_text()
        got: dict[str, float] = {}
        block = None
        for line in txt.splitlines():
            if "2-class" in line:
                block = "two_class"
            elif "3-class" in line:
                block = "three_class"
            elif "agreement" in line and block:
                got[block] = float(line.split()[-1])
            elif "n_compared" in line and block:
                got[f"{block}_n"] = int(line.split()[-1])
        return got

    agr_b = parse_agreement(BASE / "validation.txt")
    agr_n = parse_agreement(NARROW / "validation.txt")
    agreement = pd.DataFrame([
        {"rule": "baseline (1.15 / 0.90, outer)", **agr_b},
        {"rule": "narrow (1.13 / 0.91, inner)", **agr_n},
    ])
    agreement.to_csv(out / "agreement_vs_ppmi.csv", index=False)

    # ---- headline table
    summary = pd.DataFrame([{
        "labelled_visits": n_lab,
        "visits_relabelled": n_moved,
        "share_relabelled": n_moved / n_lab,
        "patients_total": int(pat.shape[0]),
        "patients_with_any_relabelled": int(pat["any_moved"].sum()),
        "share_patients_affected": float(pat["any_moved"].mean()),
        "indeterminate_before": int((lab["subtype_name_base"] == "Indeterminate").sum()),
        "indeterminate_after": int((lab["subtype_name_narrow"] == "Indeterminate").sum()),
        "agreement_2class_baseline": agr_b.get("two_class"),
        "agreement_2class_narrow": agr_n.get("two_class"),
        "agreement_3class_baseline": agr_b.get("three_class"),
        "agreement_3class_narrow": agr_n.get("three_class"),
    }])
    summary.to_csv(out / "summary.csv", index=False)

    # ---- checks against the audit's unsaved recomputation
    ind_before = int((lab["subtype_name_base"] == "Indeterminate").sum())
    ind_after = int((lab["subtype_name_narrow"] == "Indeterminate").sum())
    checks["labelled visits = 4,441"] = n_lab == AUDIT["labelled"]
    checks["298 visits relabelled"] = n_moved == AUDIT["relabelled"]
    checks["share relabelled = 6.71%"] = abs(n_moved / n_lab - AUDIT["share"]) < 5e-4
    checks["every relabelled visit came from Indeterminate"] = bool(
        (lab.loc[moved, "subtype_name_base"] == "Indeterminate").all())
    checks["Indeterminate 441 -> 143"] = (ind_before == AUDIT["indeterminate_before"]
                                          and ind_after == AUDIT["indeterminate_after"])
    checks["Indeterminate shrinkage = 67.6%"] = abs(
        (ind_before - ind_after) / ind_before - AUDIT["indeterminate_shrinkage"]) < 5e-4
    checks["198 Indeterminate -> PIGD"] = int(ct.loc["Indeterminate", "PIGD"]) == AUDIT["ind_to_pigd"]
    checks["100 Indeterminate -> TD"] = int(ct.loc["Indeterminate", "TD"]) == AUDIT["ind_to_td"]
    checks["TD 2,891 -> 2,991"] = bool(
        int((lab["subtype_name_base"] == "TD").sum()) == AUDIT["td_before"]
        and int((lab["subtype_name_narrow"] == "TD").sum()) == AUDIT["td_after"])
    checks["PIGD 1,109 -> 1,307"] = bool(
        int((lab["subtype_name_base"] == "PIGD").sum()) == AUDIT["pigd_before"]
        and int((lab["subtype_name_narrow"] == "PIGD").sum()) == AUDIT["pigd_after"])
    checks["2-class agreement falls to 97.74%"] = abs(
        (agr_n.get("two_class") or 0) - AUDIT["agreement_2class"]) < 5e-5
    checks["3-class agreement is computable"] = agr_n.get("three_class") is not None
    checks["no TD or PIGD visit moved"] = bool(
        int(ct.loc["TD", "Indeterminate"] + ct.loc["TD", "PIGD"]
            + ct.loc["PIGD", "TD"] + ct.loc["PIGD", "Indeterminate"]) == 0)
    checks["frozen baseline panel unchanged"] = (
        md5(BASE / "visit_panel.csv") == "fb038ddbdc19ab3368e13940f398e7ba")

    print("\nfigure:")
    figure_attainable_ratios(lab, b, out, checks)

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True,
                                check=True).stdout.strip()
    except Exception:
        commit = "unknown"

    (out / "manifest.json").write_text(json.dumps({
        "generated": stamp, "git_commit": commit,
        "analysis": "O-4 threshold sensitivity",
        "baseline_config": "config/config.yaml",
        "narrow_config": "config/config_narrow_0.91_1.13.yaml",
        "thresholds": {
            "baseline": {"td": 1.15, "pigd": 0.90, "band_boundaries": "outer"},
            "narrow": {"td": 1.13, "pigd": 0.91, "band_boundaries": "inner"},
        },
        "baseline_panel_md5": md5(BASE / "visit_panel.csv"),
        "narrow_panel_md5": md5(NARROW / "visit_panel.csv"),
        "models_fitted": 0, "resampling": "none (deterministic recomputation)",
        "checks": checks,
        "files": {p.name: sha(p) for p in sorted(out.glob("*")) if p.is_file()},
    }, indent=1) + "\n")

    print(f"wrote {out.relative_to(ROOT)}\n")
    print("=== crosstab: baseline rule (rows) vs narrow rule (columns) ===")
    print(ct.to_string())
    print("\n=== per class ===")
    print(per_class.to_string(index=False))
    print("\n=== agreement with PPMI's own derivations ===")
    print(agreement.to_string(index=False))
    print(f"\nvisits relabelled      {n_moved:,} of {n_lab:,} ({n_moved/n_lab:.2%})")
    print(f"patients affected      {int(pat['any_moved'].sum()):,} of {pat.shape[0]:,} "
          f"({pat['any_moved'].mean():.2%})")
    print("\nchecks:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n{sum(checks.values())} of {len(checks)} passed")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
