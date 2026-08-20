"""Stage 1 - build the longitudinal panel with TD/PIGD subtype labels.

This is the Weeks 1-2 deliverable. It requires the item-level MDS-UPDRS files;
run scripts/00_data_audit.py first to confirm they are in place.

Writes to outputs/:
    visit_panel.csv       one row per patient-visit, with subscores and labels
    baseline.csv          one row per patient, with the stability predictors
    validation.txt        agreement against PPMI's own derivations
    panel_summary.txt     cohort and coverage counts

    python scripts/01_build_panel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppmi_subtype import loaders, validate  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype.panel import baseline_frame, build_visit_panel, restrict_to_analysable  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def summarise(panel: pd.DataFrame, cfg) -> str:
    names = cfg.subtype.label_names
    lines = []
    lines.append(f"patients             {panel.PATNO.nunique():,}")
    lines.append(f"visits               {len(panel):,}")

    labelled = panel[panel["subtype"].notna()]
    lines.append(f"visits with a label  {len(labelled):,}  ({len(labelled) / len(panel):.1%})")

    lines.append("\nsubtype distribution over all labelled visits:")
    counts = labelled["subtype"].value_counts().sort_index()
    for code, n in counts.items():
        lines.append(f"  {names.get(code, code):15} {n:6,}  ({n / len(labelled):6.1%})")

    lines.append("\nsubtype distribution at baseline:")
    bl = labelled[labelled["EVENT_ID"] == "BL"]
    for code, n in bl["subtype"].value_counts().sort_index().items():
        lines.append(f"  {names.get(code, code):15} {n:6,}  ({n / len(bl):6.1%})")

    defined = labelled["ratio_defined"]
    lines.append(
        f"\nratio computable     {defined.sum():,} of {len(labelled):,} labelled visits "
        f"({defined.mean():.1%})"
    )
    lines.append(
        f"ratio undefined      {(~defined).sum():,} visits where the PIGD subscore is zero"
    )

    r = labelled.loc[defined, "ratio"]
    lines.append(
        f"\nratio distribution   median {r.median():.2f}  IQR {r.quantile(.25):.2f}-{r.quantile(.75):.2f}"
        f"  max {r.max():.2f}"
    )

    per_patient = labelled.groupby("PATNO").size()
    lines.append(f"\nlabelled visits per patient: median {per_patient.median():.0f}, max {per_patient.max()}")
    for k in (2, 3, 5, 8):
        lines.append(f"  patients with >={k}: {(per_patient >= k).sum():,}")

    return "\n".join(lines)


def main() -> int:
    cfg = load_config()
    out_dir = cfg.path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not loaders.updrs_files_available(cfg):
        print(
            "The item-level MDS-UPDRS files are not present.\n"
            "Run scripts/00_data_audit.py and see docs/DATA_REQUIREMENTS.md.",
            file=sys.stderr,
        )
        return 1

    section("BUILDING VISIT PANEL")
    panel = build_visit_panel(cfg)
    print(f"  merged panel: {len(panel):,} visits, {panel.PATNO.nunique():,} patients")

    summary = summarise(panel, cfg)
    print("\n" + summary)
    (out_dir / "panel_summary.txt").write_text(summary, encoding="utf-8")

    section("VALIDATION AGAINST PPMI DERIVATIONS")
    reports = []
    try:
        s = validate.against_curated_cut(panel, cfg)
        reports.append(validate.report(s, "vs curated td_pigd (2-class)"))
    except KeyError as exc:
        reports.append(f"curated cut comparison skipped: {exc}")

    try:
        s = validate.against_archived_three_class(panel, cfg)
        reports.append(validate.report(s, "vs archived td_pigd_old (3-class)"))
    except (KeyError, loaders.MissingDataFileError) as exc:
        reports.append(f"archived comparison skipped: {exc}")

    text = "\n\n".join(reports)
    print(text)
    (out_dir / "validation.txt").write_text(text, encoding="utf-8")

    section("WRITING OUTPUTS")
    analysable = restrict_to_analysable(panel, cfg)
    baseline = baseline_frame(analysable, cfg)

    panel.to_csv(out_dir / "visit_panel.csv", index=False)
    baseline.to_csv(out_dir / "baseline.csv", index=False)
    print(f"  outputs/visit_panel.csv   {len(panel):,} rows")
    print(f"  outputs/baseline.csv      {len(baseline):,} rows")
    print(
        f"\n  analysable subset: {analysable.PATNO.nunique():,} patients with "
        f">={cfg.cohort['min_visits_per_patient']} labelled visits"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
