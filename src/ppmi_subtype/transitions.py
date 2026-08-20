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


def state_sequences(
    panel: pd.DataFrame, label_col: str = "subtype", time_col: str = TIME_COL
) -> pd.Series:
    """Each patient's ordered sequence of labelled states, as a tuple."""
    df = panel[panel[label_col].notna()].sort_values(["PATNO", time_col])
    return df.groupby("PATNO")[label_col].apply(tuple)


def collapse_runs(sequence: tuple) -> tuple:
    """Drop immediate repeats: (TD, TD, PIGD, PIGD) -> (TD, PIGD).

    Turns a visit-by-visit record into the pathway actually traversed, so
    pathway frequencies are not dominated by how often a patient attended.
    """
    out: list = []
    for s in sequence:
        if not out or out[-1] != s:
            out.append(s)
    return tuple(out)


def reversion_after_change(
    panel: pd.DataFrame, label_col: str = "subtype", time_col: str = TIME_COL
) -> pd.DataFrame:
    """For each observed state change, does the next visit revert to the origin?

    This separates two very different things that a transition matrix reports
    identically: a patient whose phenotype genuinely progressed, and a patient
    whose single noisy assessment crossed a threshold and crossed back. A high
    reversion rate is direct evidence that the observed label is a noisy read
    of an underlying state, which is the premise of the whole project.
    """
    df = panel[panel[label_col].notna()].sort_values(["PATNO", time_col])
    g = df.groupby("PATNO", sort=False)

    frame = pd.DataFrame(
        {
            "PATNO": df["PATNO"],
            "s0": df[label_col],
            "s1": g[label_col].shift(-1),
            "s2": g[label_col].shift(-2),
            "t0": df[time_col],
            "t2": g[time_col].shift(-2),
        }
    )
    # Need a full triple, and the middle step must be an actual change.
    frame = frame[frame["s2"].notna() & (frame["s0"] != frame["s1"])].copy()
    frame["reverted"] = frame["s2"] == frame["s0"]
    return frame.reset_index(drop=True)


def reversion_summary(reversions: pd.DataFrame, names: dict | None = None) -> pd.DataFrame:
    """Reversion rate per observed transition type."""
    if reversions.empty:
        return pd.DataFrame(columns=["from_state", "to_state", "n", "n_reverted", "p_revert"])

    out = (
        reversions.groupby(["s0", "s1"])["reverted"]
        .agg(n="size", n_reverted="sum")
        .reset_index()
        .rename(columns={"s0": "from_state", "s1": "to_state"})
    )
    out["p_revert"] = out["n_reverted"] / out["n"]
    if names:
        out["from_state"] = out["from_state"].map(names)
        out["to_state"] = out["to_state"].map(names)
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def first_entry(
    panel: pd.DataFrame,
    target_state,
    min_consecutive: int = 1,
    label_col: str = "subtype",
    time_col: str = TIME_COL,
) -> pd.DataFrame:
    """Time of each patient's first entry into `target_state`.

    With ``min_consecutive > 1`` the entry only counts if the patient is still
    in that state at the required number of consecutive labelled visits. This
    distinguishes a sustained conversion from a one-visit excursion; the gap
    between the two counts is a direct measure of how much of the apparent
    progression is noise.

    Patients who never qualify are returned with ``converted = False`` and
    their last observed time, so the result can be fed to a survival estimate
    without silently dropping the censored majority.
    """
    df = panel[panel[label_col].notna()].sort_values(["PATNO", time_col])

    records = []
    for patno, grp in df.groupby("PATNO", sort=False):
        states = grp[label_col].to_numpy()
        times = grp[time_col].to_numpy()

        hit_time = np.nan
        for i in range(len(states) - min_consecutive + 1):
            window = states[i : i + min_consecutive]
            if np.all(window == target_state):
                hit_time = times[i]
                break

        # Someone already in the target state at their first visit has no
        # observable time-to-entry and is excluded from the risk set.
        at_risk_from = times[0]
        starts_in_state = bool(states[0] == target_state)

        records.append(
            {
                "PATNO": patno,
                "converted": bool(not np.isnan(hit_time)),
                "time": hit_time if not np.isnan(hit_time) else times[-1],
                "last_seen": times[-1],
                "n_visits": len(states),
                "baseline_state": states[0],
                "starts_in_state": starts_in_state,
                "at_risk_from": at_risk_from,
            }
        )
    return pd.DataFrame.from_records(records)


def kaplan_meier(times, events) -> pd.DataFrame:
    """Kaplan-Meier survival estimate with Greenwood standard errors.

    Written out rather than pulled from lifelines to keep the dependency set
    small; censoring here is simply loss to follow-up.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    order = np.argsort(times, kind="mergesort")
    times, events = times[order], events[order]

    n_at_risk = len(times)
    surv, var_sum = 1.0, 0.0
    rows = [{"time": 0.0, "n_at_risk": n_at_risk, "n_events": 0, "survival": 1.0, "se": 0.0}]

    for t in np.unique(times):
        at_t = times == t
        d = int(events[at_t].sum())
        n = int((times >= t).sum())
        if n == 0:
            continue
        if d > 0:
            surv *= 1 - d / n
            var_sum += d / (n * (n - d)) if n > d else 0.0
        rows.append(
            {
                "time": float(t),
                "n_at_risk": n,
                "n_events": d,
                "survival": surv,
                "se": surv * np.sqrt(var_sum),
            }
        )
    return pd.DataFrame(rows)
