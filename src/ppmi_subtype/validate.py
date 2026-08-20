"""Validation of recomputed subtype labels against PPMI's own derivations.

Two independent references exist in the repository, and they check different
things:

* The **curated data cut** ships ``td_pigd``, but collapsed to two classes
  (1 = TD, 2 = non-TD). Agreement here confirms the tremor/PIGD arithmetic and
  the TD threshold, but says nothing about the Indeterminate/PIGD boundary.
* The **archived Apr-2020 snapshot** ships ``td_pigd_old`` with all three
  classes for 683 patients through Year 5. This is the only in-repo check on
  the 0.90 boundary.

Passing both is what licenses using the recomputed label as ground truth for
the rest of the project.
"""

from __future__ import annotations

import pandas as pd

from .config import Config
from . import loaders

KEY = ["PATNO", "EVENT_ID"]


def collapse_to_binary(subtype: pd.Series, cfg: Config) -> pd.Series:
    """Map TD/Indeterminate/PIGD onto the curated cut's 1 = TD, 2 = non-TD."""
    labels = cfg.subtype.labels
    return subtype.map(lambda v: 1.0 if v == labels["TD"] else (2.0 if pd.notna(v) else v))


def _agreement(merged: pd.DataFrame, ours: str, theirs: str) -> dict:
    both = merged[merged[ours].notna() & merged[theirs].notna()]
    if both.empty:
        return {"n_compared": 0, "agreement": float("nan")}
    match = both[ours] == both[theirs]
    return {
        "n_compared": int(len(both)),
        "n_agree": int(match.sum()),
        "n_disagree": int((~match).sum()),
        "agreement": float(match.mean()),
        "ours_only": int((merged[ours].notna() & merged[theirs].isna()).sum()),
        "theirs_only": int((merged[ours].isna() & merged[theirs].notna()).sum()),
    }


def against_curated_cut(panel: pd.DataFrame, cfg: Config) -> dict:
    """Compare our label, collapsed to binary, against curated ``td_pigd``."""
    if "td_pigd" not in panel.columns:
        raise KeyError("panel does not carry the curated cut's td_pigd column")

    frame = panel[KEY + ["subtype", "td_pigd"]].copy()
    frame["ours_binary"] = collapse_to_binary(frame["subtype"], cfg)

    summary = _agreement(frame, "ours_binary", "td_pigd")
    summary["confusion"] = pd.crosstab(
        frame["ours_binary"], frame["td_pigd"], dropna=True
    )
    summary["disagreements"] = frame[
        frame["ours_binary"].notna()
        & frame["td_pigd"].notna()
        & (frame["ours_binary"] != frame["td_pigd"])
    ]
    return summary


def remap_archived_labels(series: pd.Series, cfg: Config) -> pd.Series:
    """Translate archived ``td_pigd_old`` codes into this project's coding.

    The two codings agree on TD = 1 but swap the other two: the archived
    dictionary defines 2 = PIGD and 3 = Indeterminate, whereas this project
    uses 2 = Indeterminate and 3 = PIGD. Comparing the raw codes produces a
    confusion matrix that looks like a systematic failure when the labels
    actually agree.
    """
    ref = cfg.raw["subtype"]["archived_reference_labels"]
    ours = cfg.subtype.labels
    lookup = {float(ref[name]): float(ours[name]) for name in ref}
    return series.map(lookup)


def against_archived_three_class(panel: pd.DataFrame, cfg: Config) -> dict:
    """Compare our three-class label against archived ``td_pigd_old``.

    This is the only in-repo check on the Indeterminate/PIGD boundary, since
    the curated cut collapses those two classes.
    """
    archived = loaders.load_archived_original_cohort(cfg)
    if "td_pigd_old" not in archived.columns:
        raise KeyError("archived cohort file does not carry td_pigd_old")

    ref = archived[KEY + ["td_pigd_old"]].drop_duplicates(KEY)
    frame = panel[KEY + ["subtype"]].merge(ref, on=KEY, how="inner")
    frame["td_pigd_old"] = remap_archived_labels(frame["td_pigd_old"], cfg)

    summary = _agreement(frame, "subtype", "td_pigd_old")
    summary["n_patients"] = int(frame["PATNO"].nunique())
    names = cfg.subtype.label_names
    confusion = pd.crosstab(frame["subtype"], frame["td_pigd_old"], dropna=True)
    confusion.index = [names.get(i, i) for i in confusion.index]
    confusion.columns = [names.get(c, c) for c in confusion.columns]
    summary["confusion"] = confusion
    summary["disagreements"] = frame[
        frame["subtype"].notna()
        & frame["td_pigd_old"].notna()
        & (frame["subtype"] != frame["td_pigd_old"])
    ]
    return summary


def report(summary: dict, title: str) -> str:
    """Render a validation summary as plain text."""
    lines = [f"--- {title} ---"]
    for k in ("n_patients", "n_compared", "n_agree", "n_disagree", "ours_only", "theirs_only"):
        if k in summary:
            lines.append(f"  {k:14} {summary[k]}")
    if "agreement" in summary:
        lines.append(f"  {'agreement':14} {summary['agreement']:.4f}")
    if "confusion" in summary:
        lines.append("  confusion (rows = ours, cols = reference):")
        lines.append("    " + summary["confusion"].to_string().replace("\n", "\n    "))
    return "\n".join(lines)
