"""Stage 0 - data availability audit.

Runs against whatever is on disk and answers one question: can the proposal's
model be built from the current data, and if not, what exactly is missing.

Also produces an interim transition analysis using the curated cut's collapsed
two-class ``td_pigd``, so the Weeks 3-4 questions can be examined before the
item-level files arrive. That analysis cannot address the Indeterminate state.

    python scripts/00_data_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppmi_subtype import loaders, transitions  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype.panel import filter_cohort, _add_time_axis  # noqa: E402

BINARY_NAMES = {1.0: "TD", 2.0: "non-TD"}


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def audit_files(cfg) -> bool:
    section("1. SOURCE FILE AVAILABILITY")
    checks = [
        ("Curated data cut", cfg.path("curated_cut").exists(), "required"),
        ("Archived Apr-2020 cohort", cfg.path("archived_original_cohort").exists(), "validation"),
        ("MDS-UPDRS Part II (items)", bool(cfg.glob("updrs_part2_glob")), "REQUIRED"),
        ("MDS-UPDRS Part III (items)", bool(cfg.glob("updrs_part3_glob")), "REQUIRED"),
    ]
    for name, ok, kind in checks:
        print(f"  [{'x' if ok else ' '}] {name:32} ({kind})")

    have_items = loaders.updrs_files_available(cfg)
    if not have_items:
        print(
            "\n  The item-level MDS-UPDRS files are absent. Without them the\n"
            "  TD/PIGD ratio cannot be computed and the Indeterminate state\n"
            "  cannot be recovered. See docs/DATA_REQUIREMENTS.md."
        )
    return have_items


def audit_panel(cfg) -> pd.DataFrame:
    section("2. LONGITUDINAL STRUCTURE (curated data cut)")
    curated = loaders.load_curated_cut(cfg)
    print(f"  rows {len(curated):,}   columns {curated.shape[1]}   patients {curated.PATNO.nunique():,}")

    panel = _add_time_axis(filter_cohort(curated, cfg), cfg)
    print(f"\n  Primary cohort {cfg.cohort['primary_subgroups']}:")
    print(f"    patients {panel.PATNO.nunique():,}   visits {len(panel):,}")

    labelled = panel[panel["td_pigd"].notna()]
    print(f"    visits with a curated td_pigd label: {len(labelled):,}")

    per_patient = labelled.groupby("PATNO").size()
    print(f"\n  Visits per patient: median {per_patient.median():.0f}, max {per_patient.max()}")
    for k in (2, 3, 5, 8):
        print(f"    patients with >={k} labelled visits: {(per_patient >= k).sum():,}")

    span = labelled.groupby("PATNO")["years_from_baseline"].max()
    print(f"\n  Follow-up years: median {span.median():.1f}, 90th pct {span.quantile(0.9):.1f}, max {span.max():.1f}")
    return panel


def audit_intervals(panel: pd.DataFrame) -> None:
    section("3. VISIT SPACING")
    pairs = transitions.consecutive_pairs(panel, label_col="td_pigd")
    iv = pairs["interval_years"]
    print("  Interval between consecutive labelled visits (years):")
    print(f"    median {iv.median():.2f}   mean {iv.mean():.2f}   IQR {iv.quantile(.25):.2f}-{iv.quantile(.75):.2f}")
    print(f"    <=1.5y {(iv <= 1.5).mean():.1%}   >2.5y {(iv > 2.5).mean():.1%}   >4y {(iv > 4).mean():.1%}")
    print(
        "\n  Spacing is irregular, so a transition matrix over 'consecutive visits'\n"
        "  mixes one-year and multi-year transitions. Stratify by interval, or\n"
        "  model in continuous time."
    )


def audit_transitions(panel: pd.DataFrame) -> None:
    section("4. INTERIM TRANSITION ANALYSIS (2-class label)")
    print(
        "  Using the curated cut's td_pigd: 1 = TD, 2 = non-TD (PIGD or\n"
        "  Indeterminate merged). This understates instability, because moves\n"
        "  between Indeterminate and PIGD are invisible.\n"
    )
    pairs = transitions.consecutive_pairs(panel, label_col="td_pigd")
    states = [1.0, 2.0]

    m = transitions.transition_matrix(pairs, states=states)
    m.index = [BINARY_NAMES[s] for s in m.index]
    m.columns = [BINARY_NAMES[s] for s in m.columns]
    print("  Pooled transition matrix (rows = from, cols = to):")
    print("    " + m.round(3).to_string().replace("\n", "\n    "))

    print("\n  Stratified by interval:")
    for label, mat in transitions.transition_matrix_by_interval(pairs, states=states).items():
        mat.index = [BINARY_NAMES[s] for s in mat.index]
        mat.columns = [BINARY_NAMES[s] for s in mat.columns]
        print(f"\n    interval {label}:")
        print("      " + mat.round(3).to_string().replace("\n", "\n      "))

    ci = transitions.transition_counts_with_ci(pairs, states=states)
    ci["from_state"] = ci["from_state"].map(BINARY_NAMES)
    ci["to_state"] = ci["to_state"].map(BINARY_NAMES)
    print("\n  With Wilson 95% intervals:")
    print("    " + ci.round(3).to_string(index=False).replace("\n", "\n    "))

    section("5. PREVALENCE OVER FOLLOW-UP")
    prev = transitions.prevalence_by_year(panel, label_col="td_pigd")
    prev.columns = [str(c).replace("1.0", "TD").replace("2.0", "nonTD") for c in prev.columns]
    print("    " + prev.round(3).to_string().replace("\n", "\n    "))
    print(
        "\n  TD prevalence falls steadily with follow-up. Attrition is heavy and\n"
        "  almost certainly not random, so this mixes progression with dropout."
    )


def audit_predictors(cfg, panel: pd.DataFrame) -> None:
    section("6. BASELINE STABILITY PREDICTORS (proposal Part II)")
    preds = cfg.stability_predictors
    baseline = panel[panel["EVENT_ID"] == "BL"]
    n = len(baseline)
    print(f"  Baseline visits: {n:,}\n")
    for group, cols in preds.items():
        print(f"  {group}:")
        for c in cols:
            if c not in baseline.columns:
                print(f"    {c:22} ABSENT from curated cut")
                continue
            pct = baseline[c].notna().mean() * 100
            flag = "" if pct >= 90 else "   <- sparse"
            print(f"    {c:22} {baseline[c].notna().sum():5,} / {n:,}  ({pct:5.1f}%){flag}")
        print()


def audit_ratio_feasibility(cfg, panel: pd.DataFrame) -> None:
    section("7. THE RATIO AS A MODELLING TARGET")
    pigd = panel["pigd"]
    observed = pigd.notna()
    zero = observed & (pigd == 0)
    print(f"  Visits with an observed PIGD subscore: {observed.sum():,}")
    print(f"  ...of which PIGD subscore == 0:        {zero.sum():,}  ({zero.sum() / observed.sum():.1%})")
    print(
        "\n  Where the PIGD subscore is zero the ratio tremor/PIGD is undefined.\n"
        "  The proposal models the observed ratio as a continuous noisy reading\n"
        "  of a latent phenotype, so this population needs a deliberate decision\n"
        "  before Stage 3, not a silent drop. Options: model the two subscores\n"
        "  jointly, model log(tremor+c) - log(pigd+c), or treat the label as\n"
        "  ordinal and skip the ratio entirely."
    )


def main() -> int:
    cfg = load_config()
    print(f"Config: {cfg.root / 'config' / 'config.yaml'}   seed {cfg.seed}")

    have_items = audit_files(cfg)
    panel = audit_panel(cfg)
    audit_intervals(panel)
    audit_transitions(panel)
    audit_predictors(cfg, panel)
    audit_ratio_feasibility(cfg, panel)

    section("VERDICT")
    if have_items:
        print("  All required sources present. Run scripts/01_build_panel.py.")
    else:
        print(
            "  BLOCKED on the two item-level MDS-UPDRS files.\n"
            "  Everything else the proposal needs is present and well populated.\n"
            "  Once the files are in data/motor_updrs/, run:\n"
            "      python scripts/01_build_panel.py"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
