"""Empirical subtype transition analysis (Weeks 3-4 deliverable).

Everything here is label-agnostic: it operates on whichever subtype column is
passed, so the same code serves the recomputed three-class label and the
curated cut's collapsed two-class ``td_pigd``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIME_COL = "years_from_baseline"


def consecutive_pairs(
    panel: pd.DataFrame,
    label_col: str = "subtype",
    time_col: str = TIME_COL,
) -> pd.DataFrame:
    """Adjacent labelled-visit pairs for each patient.

    Only visits with a non-missing label participate, so a gap year does not
    break a patient's sequence. The elapsed interval is carried through,
    because transition probabilities are meaningless without it: PPMI visits
    are annual early on and biennial later.
    """
    df = panel[panel[label_col].notna()].sort_values(["PATNO", time_col])
    grouped = df.groupby("PATNO", sort=False)

    out = pd.DataFrame(
        {
            "PATNO": df["PATNO"],
            "from_state": df[label_col],
            "to_state": grouped[label_col].shift(-1),
            "from_time": df[time_col],
            "to_time": grouped[time_col].shift(-1),
            "from_event": df["EVENT_ID"],
            "to_event": grouped["EVENT_ID"].shift(-1),
        }
    )
    out = out[out["to_state"].notna()].copy()
    out["interval_years"] = out["to_time"] - out["from_time"]
    return out.reset_index(drop=True)


def transition_matrix(
    pairs: pd.DataFrame,
    states: list | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Row-stochastic transition matrix (rows = from, columns = to)."""
    if states is None:
        states = sorted(set(pairs["from_state"]) | set(pairs["to_state"]))

    counts = pd.crosstab(pairs["from_state"], pairs["to_state"])
    counts = counts.reindex(index=states, columns=states, fill_value=0)

    if not normalize:
        return counts

    totals = counts.sum(axis=1)
    probs = counts.div(totals.where(totals > 0), axis=0)
    return probs


def transition_counts_with_ci(
    pairs: pd.DataFrame, states: list | None = None, alpha: float = 0.05
) -> pd.DataFrame:
    """Transition probabilities with Wilson score intervals.

    Row counts vary by more than an order of magnitude across states, so a
    point estimate alone is misleading for the rarer origin states.
    """
    from scipy.stats import norm

    counts = transition_matrix(pairs, states=states, normalize=False)
    z = norm.ppf(1 - alpha / 2)

    records = []
    for from_state, row in counts.iterrows():
        n = row.sum()
        for to_state, k in row.items():
            if n == 0:
                lo = hi = p = np.nan
            else:
                p = k / n
                denom = 1 + z**2 / n
                centre = (p + z**2 / (2 * n)) / denom
                half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
                lo, hi = centre - half, centre + half
            records.append(
                {
                    "from_state": from_state,
                    "to_state": to_state,
                    "n_from": int(n),
                    "n_transitions": int(k),
                    "probability": p,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return pd.DataFrame.from_records(records)


def transition_matrix_by_interval(
    pairs: pd.DataFrame,
    bins: tuple[float, ...] = (0.0, 1.5, 2.5, 4.0, np.inf),
    states: list | None = None,
) -> dict[str, pd.DataFrame]:
    """Transition matrices stratified by elapsed interval.

    A one-year and a four-year transition are not the same quantity; pooling
    them, as a naive count over consecutive visits does, conflates progression
    rate with visit attendance.
    """
    labelled = pairs.assign(
        interval_bin=pd.cut(pairs["interval_years"], bins=list(bins), right=False)
    )
    return {
        str(b): transition_matrix(grp, states=states)
        for b, grp in labelled.groupby("interval_bin", observed=True)
    }


def prevalence_by_year(
    panel: pd.DataFrame, label_col: str = "subtype", year_col: str = "YEAR"
) -> pd.DataFrame:
    """Cross-sectional subtype prevalence at each nominal visit year.

    Read with care: attrition is heavy and not random, so a shift in
    prevalence mixes genuine progression with selective dropout.
    """
    df = panel[panel[label_col].notna()]
    counts = pd.crosstab(df[year_col], df[label_col])
    props = counts.div(counts.sum(axis=1), axis=0)
    props.columns = [f"prop_{c}" for c in props.columns]
    return pd.concat([counts.add_prefix("n_"), props], axis=1).assign(
        n_total=counts.sum(axis=1)
    )


def sojourn_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-origin-state stability summary."""
    rows = []
    for state, grp in pairs.groupby("from_state"):
        rows.append(
            {
                "state": state,
                "n_transitions_observed": len(grp),
                "n_patients": grp["PATNO"].nunique(),
                "p_stay": (grp["to_state"] == state).mean(),
                "median_interval_years": grp["interval_years"].median(),
            }
        )
    return pd.DataFrame(rows)


def first_transition_times(
    pairs: pd.DataFrame, from_state, to_state
) -> pd.DataFrame:
    """Time to each patient's first `from_state` -> `to_state` transition.

    Groundwork for the proposal's stated end goal of predicting *when* a
    patient converts, rather than only whether.
    """
    hits = pairs[(pairs["from_state"] == from_state) & (pairs["to_state"] == to_state)]
    return (
        hits.sort_values("to_time")
        .groupby("PATNO")
        .head(1)[["PATNO", "from_time", "to_time", "interval_years"]]
        .rename(columns={"to_time": "years_to_transition"})
        .reset_index(drop=True)
    )
