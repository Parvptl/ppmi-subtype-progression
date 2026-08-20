"""Construction of the longitudinal patient-visit panel.

This is the Week 1-2 deliverable: one row per patient-visit carrying the
TD/PIGD subscores, the continuous ratio and the three-class subtype label,
joined to the covariates and baseline predictors the later stages need.

The curated data cut supplies the visit skeleton (which patients attended which
visits, when, and their covariates); the item-level MDS-UPDRS files supply the
16 scores from which the subtype is derived.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from . import loaders
from .subtype import compute_subscores, deduplicate_visits, select_medication_state

KEY = ["PATNO", "EVENT_ID"]

_SUBSCORE_COLUMNS = [
    "tremor_score",
    "pigd_score",
    "ratio",
    "log_ratio",
    "ratio_defined",
    "subtype",
    "subtype_name",
]


def parse_visit_date(series: pd.Series) -> pd.Series:
    """Parse the curated cut's ``MM/YYYY`` visit date to a timestamp.

    PPMI publishes month and year only; the day is not released. Dates are
    anchored to the first of the month, which caps the resolution of any
    inter-visit interval at roughly one month.
    """
    return pd.to_datetime(series, format="%m/%Y", errors="coerce")


def filter_cohort(frame: pd.DataFrame, cfg: Config, extended: bool = False) -> pd.DataFrame:
    """Restrict to the analysis population and drop non-analysable visits."""
    cohort = cfg.cohort
    key = "extended_subgroups" if extended else "primary_subgroups"
    keep = set(cohort[key])

    out = frame[frame["subgroup"].isin(keep)].copy()
    out = out[~out["EVENT_ID"].isin(set(cohort["excluded_events"]))]
    out = out[out["YEAR"] <= cohort["max_follow_up_years"]]
    return out


def build_visit_panel(cfg: Config, extended: bool = False) -> pd.DataFrame:
    """Assemble the full longitudinal panel with subtype labels.

    Requires the item-level MDS-UPDRS files. Raises
    :class:`~ppmi_subtype.loaders.MissingDataFileError` with download
    instructions if they are absent.
    """
    curated = filter_cohort(loaders.load_curated_cut(cfg), cfg, extended=extended)
    subscores = build_subscores(cfg)

    panel = curated.merge(subscores, on=KEY, how="left", validate="one_to_one")

    # The left join introduces NaN into what was a boolean column, making it
    # object dtype where `~flag` yields a numeric complement instead of False.
    # Restore a true boolean so downstream masking behaves.
    panel["ratio_defined"] = panel["ratio_defined"] == True  # noqa: E712

    return _add_time_axis(panel, cfg)


def build_subscores(cfg: Config) -> pd.DataFrame:
    """Compute per-visit subscores and labels from the item-level files."""
    scfg = cfg.subtype
    part2 = loaders.load_updrs_part2(cfg)
    part3 = loaders.load_updrs_part3(cfg)

    part3 = select_medication_state(part3, scfg)

    part2_items = [c for c in scfg.all_items if c in part2.columns]
    part3_items = [c for c in scfg.all_items if c in part3.columns]

    carry = [c for c in (scfg.pdstate_column, scfg.treatment_column, "NHY") if c in part3.columns]
    merged = part2[KEY + part2_items].merge(
        part3[KEY + part3_items + carry], on=KEY, how="outer"
    )
    merged = deduplicate_visits(merged, scfg)

    scores = compute_subscores(merged, scfg)
    return pd.concat([merged[KEY], scores], axis=1)


def _add_time_axis(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add per-patient elapsed time, in years, measured from the first visit.

    ``YEAR`` in the curated cut is a nominal visit label derived from
    ``EVENT_ID``, not elapsed time. The hierarchical model needs actual
    intervals, because attendance is irregular and visits are missed.
    """
    out = panel.copy()
    out["visit_ts"] = parse_visit_date(out["visit_date"])
    out = out.sort_values(["PATNO", "visit_ts", "YEAR"], kind="mergesort")

    baseline = out.groupby("PATNO")["visit_ts"].transform("min")
    out["years_from_baseline"] = (out["visit_ts"] - baseline).dt.days / 365.25

    # Fall back to the nominal visit year where the date is unparseable, so a
    # missing INFODT does not silently drop a visit from the time axis.
    out["years_from_baseline"] = out["years_from_baseline"].fillna(out["YEAR"].astype(float))

    out["visit_index"] = out.groupby("PATNO").cumcount()
    out["n_visits"] = out.groupby("PATNO")["PATNO"].transform("size")
    return out.reset_index(drop=True)


def restrict_to_analysable(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep patients with enough labelled visits to contribute a transition."""
    labelled = panel[panel["subtype"].notna()].copy()
    counts = labelled.groupby("PATNO")["subtype"].size()
    keep = counts[counts >= cfg.cohort["min_visits_per_patient"]].index
    return labelled[labelled["PATNO"].isin(keep)].reset_index(drop=True)


def baseline_frame(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per patient: first visit, plus the stability predictors."""
    preds = cfg.stability_predictors
    wanted = list(dict.fromkeys(preds["hypothesis_driven"] + preds["covariates"]))
    available = [c for c in wanted if c in panel.columns]

    first = panel.sort_values(["PATNO", "years_from_baseline"]).groupby("PATNO").head(1)
    cols = KEY + ["YEAR", "visit_ts", "n_visits"] + _SUBSCORE_COLUMNS + available
    return first[[c for c in cols if c in first.columns]].reset_index(drop=True)
