"""TD/PIGD subtype classification from item-level MDS-UPDRS scores.

Definition follows Stebbins et al., *Movement Disorders* 2013, and reproduces
the derivation notes attached to `td_pigd` in the PPMI Curated Data Cut
dictionary:

    tremor_score = mean of 11 tremor items (1 from Part II, 10 from Part III)
    pigd_score   = mean of 5 gait/postural items (2 from Part II, 3 from Part III)
    ratio        = tremor_score / pigd_score

    ratio >= 1.15                          -> TD
    pigd_score == 0 and tremor_score > 0   -> TD
    ratio <= 0.90                          -> PIGD
    0.90 < ratio < 1.15                    -> Indeterminate
    tremor_score == 0 and pigd_score == 0  -> Indeterminate

A note on the ratio as a modelling target
-----------------------------------------
The classification is well defined for every visit, but the *ratio itself* is
not: whenever ``pigd_score == 0`` it is undefined. This matters because the
proposed hierarchical model treats the observed ratio as a continuous noisy
measurement of a latent phenotype. Callers get an explicit ``ratio_defined``
flag and a separate ``log_ratio`` column so this population can be counted and
handled deliberately rather than silently dropped or coerced to infinity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SubtypeConfig


class MissingItemsError(KeyError):
    """Raised when the input frame lacks MDS-UPDRS item columns."""


def clean_items(frame: pd.DataFrame, items: tuple[str, ...], cfg: SubtypeConfig) -> pd.DataFrame:
    """Coerce item columns to numeric and blank out anything that is not a score.

    PPMI records "Unable to Rate" as the sentinel 101 in the same numeric field
    as a real 0-4 rating. Averaging it as a score is not a rounding error: one
    such value among the five PIGD items lifts their mean to roughly 20 and
    drives the ratio to zero, relabelling a tremor-dominant visit as PIGD.

    Anything outside ``valid_score_range`` is treated as missing, so a sentinel
    PPMI adds later cannot quietly enter a subscore.
    """
    lo, hi = cfg.valid_score_range
    block = frame[list(items)].apply(pd.to_numeric, errors="coerce")
    return block.where((block >= lo) & (block <= hi))


def count_unrateable(frame: pd.DataFrame, cfg: SubtypeConfig) -> pd.Series:
    """Per-item count of sentinel codes, for reporting before they are dropped."""
    present = [c for c in cfg.all_items if c in frame.columns]
    block = frame[present].apply(pd.to_numeric, errors="coerce")
    return block.isin(list(cfg.missing_value_codes)).sum().sort_values(ascending=False)


def _subscore(
    frame: pd.DataFrame, items: tuple[str, ...], max_missing: int, cfg: SubtypeConfig
) -> pd.Series:
    """Row-wise mean of `items`, NaN where too many constituents are missing.

    The mean is taken over the *observed* items only. A row with more than
    `max_missing` absent items yields NaN, matching PPMI's own rule that a
    subscore is null if any constituent is missing (max_missing = 0).
    """
    block = clean_items(frame, items, cfg)
    n_missing = block.isna().sum(axis=1)
    score = block.mean(axis=1, skipna=True)
    return score.where(n_missing <= max_missing)


def compute_subscores(frame: pd.DataFrame, cfg: SubtypeConfig) -> pd.DataFrame:
    """Compute tremor and PIGD subscores, the ratio, and the 3-class label.

    Parameters
    ----------
    frame
        One row per patient-visit, carrying all 16 item columns.
    cfg
        Subtype section of the validated config.

    Returns
    -------
    A frame indexed like `frame` with columns ``tremor_score``, ``pigd_score``,
    ``ratio``, ``log_ratio``, ``ratio_defined``, ``subtype`` and
    ``subtype_name``.
    """
    absent = [c for c in cfg.all_items if c not in frame.columns]
    if absent:
        raise MissingItemsError(
            f"input frame is missing {len(absent)} of {len(cfg.all_items)} MDS-UPDRS "
            f"item columns: {absent}. See docs/DATA_REQUIREMENTS.md."
        )

    tremor = _subscore(frame, cfg.tremor_items, cfg.max_missing_tremor_items, cfg)
    pigd = _subscore(frame, cfg.pigd_items, cfg.max_missing_pigd_items, cfg)

    out = pd.DataFrame(index=frame.index)
    out["tremor_score"] = tremor
    out["pigd_score"] = pigd

    # Ratio is left as NaN where the denominator is zero; `ratio_defined`
    # records the distinction between "not computable" and "not measured".
    both_observed = tremor.notna() & pigd.notna()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = tremor / pigd.replace(0.0, np.nan)
    out["ratio"] = ratio
    out["log_ratio"] = np.log(ratio.where(ratio > 0))
    out["ratio_defined"] = both_observed & (pigd != 0)

    out["subtype"] = classify(tremor, pigd, cfg)
    out["subtype_name"] = out["subtype"].map(cfg.label_names)
    return out


def classify(
    tremor_score: pd.Series, pigd_score: pd.Series, cfg: SubtypeConfig
) -> pd.Series:
    """Apply the TD/Indeterminate/PIGD decision rule.

    Handled in an order that makes the degenerate cases explicit rather than
    letting them fall through a division.
    """
    labels = cfg.labels
    result = pd.Series(np.nan, index=tremor_score.index, dtype="float64")

    observed = tremor_score.notna() & pigd_score.notna()
    if not observed.any():
        return result

    t = tremor_score.where(observed)
    p = pigd_score.where(observed)

    both_zero = observed & (t == 0) & (p == 0)
    pigd_zero_only = observed & (p == 0) & (t > 0)
    divisible = observed & (p != 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = t / p.where(p != 0)

    result[divisible & (ratio >= cfg.td_threshold)] = labels["TD"]
    result[divisible & (ratio <= cfg.pigd_threshold)] = labels["PIGD"]
    result[divisible & (ratio > cfg.pigd_threshold) & (ratio < cfg.td_threshold)] = labels[
        "Indeterminate"
    ]

    # Degenerate cases, applied last so they cannot be overwritten.
    result[pigd_zero_only] = labels["TD"]
    result[both_zero] = labels["Indeterminate"]

    return result


def select_medication_state(frame: pd.DataFrame, cfg: SubtypeConfig) -> pd.DataFrame:
    """Keep only exams matching the configured medication state.

    The canonical TD/PIGD classification uses the OFF-medication or untreated
    exam. Participants not on PD medication (``PDTRTMNT == 0``) have no
    meaningful ON/OFF distinction and are retained under either setting.
    """
    if cfg.medication_state == "ANY":
        return frame.copy()

    state_col, treat_col = cfg.pdstate_column, cfg.treatment_column
    if state_col not in frame.columns:
        raise MissingItemsError(
            f"column {state_col!r} not found; it is required to select the "
            f"{cfg.medication_state} exam. It ships in the MDS-UPDRS Part III file."
        )

    state = frame[state_col].astype("string").str.strip().str.upper()
    matches = state == cfg.medication_state

    if treat_col in frame.columns:
        untreated = pd.to_numeric(frame[treat_col], errors="coerce") == 0
        matches = matches | untreated
    matches = matches | state.isna()

    return frame[matches.fillna(False)].copy()


def deduplicate_visits(frame: pd.DataFrame, cfg: SubtypeConfig) -> pd.DataFrame:
    """Collapse a patient-visit to a single row.

    A visit can carry more than one Part III exam (for example an OFF and an ON
    assessment, or a repeated entry). After `select_medication_state` the
    remaining duplicates are resolved by preferring the row with the fewest
    missing items, then the first occurrence, so the choice is deterministic
    rather than dependent on file ordering.
    """
    if not frame.duplicated(["PATNO", "EVENT_ID"]).any():
        return frame.copy()

    present = tuple(c for c in cfg.all_items if c in frame.columns)
    cleaned = clean_items(frame, present, cfg)
    ordered = frame.assign(_n_missing=cleaned.isna().sum(axis=1))
    ordered = ordered.sort_values(["PATNO", "EVENT_ID", "_n_missing"], kind="mergesort")
    deduped = ordered.drop_duplicates(["PATNO", "EVENT_ID"], keep="first")
    return deduped.drop(columns="_n_missing")
