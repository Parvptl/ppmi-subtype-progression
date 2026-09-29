"""Patient trajectories, persistently extreme profiles, and cohort selection.

A *trajectory* is one patient's chronological sequence of visits carrying the
continuous TD/PIGD ratio. The earlier stages of this project reduced a patient
to a bag of independent transitions; Step 3 keeps the patient as the unit,
because a model that predicts the next observation from the previous ones needs
the previous ones to stay together and in order.

Two populations, and the difference matters
-------------------------------------------
* **Labelled visits** --- every visit the classification rule could label. This
  is what the Markov models use, and what the extreme criteria are evaluated
  over, because a visit whose ratio is undefined is still evidence about how
  extreme that patient is.
* **Finite-ratio visits** --- the subset where the ratio is a real number. A
  model defined on the ratio can only see these. The gap is not small and not
  random: 692 of 4,295 labelled visits in the cohort have a zero PIGD subscore,
  and 634 of those are labelled TD.

So the modellable cohort is smaller than the labelled one before any filtering
is applied at all. That reduction is structural --- a consequence of the ratio's
definition --- and is reported separately from the extreme filtering, which is a
methodological choice.

Extreme profiles
----------------
Every criterion here is a statement about a *patient*, never about a visit. The
ratio moves by a median of 0.68 between consecutive visits against an
Indeterminate band 0.25 wide, and roughly half of all patients with an extreme
visit have exactly one, so a per-visit rule would remove noise excursions rather
than the sustained profiles we mean. The criteria are defined in config.yaml and
described there; none of them may be defined on the subtype label, which would
condition on the outcome.

Nothing here decides which criterion is right. `extreme_flags` evaluates all of
them and the caller reports the consequences of each.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .panel import restrict_to_analysable

#: Columns of the trajectory table, in order. `ratio_defined` distinguishes
#: "the ratio could not be computed" from "the visit was not measured".
TRAJECTORY_COLUMNS = [
    "PATNO",
    "visit_index",
    "EVENT_ID",
    "visit_date",
    "years_from_baseline",
    "interval_years",
    "ratio",
    "ratio_defined",
    "subtype",
    "subtype_name",
]


# ---------------------------------------------------------------------------
#  Construction
# ---------------------------------------------------------------------------
def build_trajectories(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One chronologically ordered row per labelled visit, per patient.

    Restricted to the analysable cohort --- patients with at least
    ``cohort.min_visits_per_patient`` labelled visits --- so this is the same
    934-patient population the Markov models were evaluated on, and the two are
    directly comparable before any Step 3 filtering.

    ``visit_index`` is renumbered from zero within the patient *after* sorting,
    so it indexes the trajectory rather than the original panel.
    """
    labelled = restrict_to_analysable(panel, cfg)

    out = labelled.sort_values(
        ["PATNO", "years_from_baseline", "EVENT_ID"], kind="mergesort"
    ).copy()

    out["visit_index"] = out.groupby("PATNO").cumcount()
    out["interval_years"] = out.groupby("PATNO")["years_from_baseline"].diff()

    # `ratio_defined` arrives from the panel, where a left join can leave it as
    # object dtype. Recompute it here from the ratio itself so the trajectory
    # table is self-consistent whatever the panel carries.
    out["ratio_defined"] = np.isfinite(
        pd.to_numeric(out["ratio"], errors="coerce").to_numpy(dtype=float)
    )

    missing = [c for c in TRAJECTORY_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(f"panel is missing columns needed for a trajectory: {missing}")

    return out[TRAJECTORY_COLUMNS].reset_index(drop=True)


def check_ordering(traj: pd.DataFrame) -> None:
    """Raise if any patient's trajectory is not strictly ordered in time.

    Called before anything conditions on "the visits before k". If the table
    were mis-sorted, every leakage guarantee downstream would be vacuous: the
    history slice would be a slice of the wrong sequence.
    """
    for patno, g in traj.groupby("PATNO", sort=False):
        idx = g["visit_index"].to_numpy()
        if not np.array_equal(idx, np.arange(len(g))):
            raise ValueError(f"patient {patno}: visit_index is not 0..n-1 in order")
        t = g["years_from_baseline"].to_numpy(dtype=float)
        if np.any(np.diff(t) < 0):
            raise ValueError(f"patient {patno}: years_from_baseline decreases")


def finite_only(traj: pd.DataFrame) -> pd.DataFrame:
    """The subset a ratio-based model can see, reindexed within each patient.

    ``finite_index`` numbers the finite-ratio visits 0..T-1. Prediction points
    are expressed in this index, because conditioning on "the visits before k"
    has to mean the visits the model can actually use.
    """
    out = traj[traj["ratio_defined"]].copy()
    out["finite_index"] = out.groupby("PATNO").cumcount()
    return out.reset_index(drop=True)


def trajectory_summary(traj: pd.DataFrame) -> pd.DataFrame:
    """Per-patient shape of the trajectory: counts, spans and ratio statistics."""
    g = traj.groupby("PATNO")
    r = traj["ratio"]

    summary = pd.DataFrame(
        {
            "n_labelled": g.size(),
            "n_finite": g["ratio_defined"].sum().astype(int),
            "n_undefined": (~traj["ratio_defined"]).groupby(traj["PATNO"]).sum().astype(int),
            "n_zero": (r == 0).groupby(traj["PATNO"]).sum().astype(int),
            "first_year": g["years_from_baseline"].min(),
            "last_year": g["years_from_baseline"].max(),
            "ratio_min": g["ratio"].min(),
            "ratio_median": g["ratio"].median(),
            "ratio_max": g["ratio"].max(),
            "n_subtypes": g["subtype"].nunique(),
        }
    )
    summary["span_years"] = summary["last_year"] - summary["first_year"]
    summary["ratio_range"] = summary["ratio_max"] - summary["ratio_min"]
    summary["frac_undefined"] = summary["n_undefined"] / summary["n_labelled"]
    summary["frac_zero"] = summary["n_zero"] / summary["n_labelled"]
    summary["frac_degenerate"] = summary["frac_undefined"] + summary["frac_zero"]
    summary["ever_changed"] = summary["n_subtypes"] > 1
    return summary.reset_index()


# ---------------------------------------------------------------------------
#  Persistently extreme profiles
# ---------------------------------------------------------------------------
def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a chronologically ordered mask."""
    best = current = 0
    for flag in np.asarray(mask, dtype=bool):
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def _side_masks(ratio: np.ndarray, high: float, low: float) -> tuple[np.ndarray, np.ndarray]:
    """Which visits sit beyond the high and the low boundary.

    An undefined ratio counts towards the *high* side. It arises when the PIGD
    subscore is zero, which is the limit of tremor/pigd as the denominator falls
    away; treating it as low, or as neither, would put the most tremor-dominant
    visits on the wrong side of a rule meant to find tremor-dominant profiles.
    """
    finite = np.isfinite(ratio)
    is_high = (~finite) | (finite & (ratio >= high))
    is_low = finite & (ratio <= low)
    return is_high, is_low


def _evaluate_criterion(ratio: np.ndarray, spec: dict) -> bool:
    """Apply one criterion to one patient's chronologically ordered ratios."""
    kind = spec["type"]
    n = len(ratio)
    if n == 0:
        return False

    if kind == "degenerate_fraction":
        finite = np.isfinite(ratio)
        degenerate = (~finite) | (finite & (ratio == 0))
        return bool(degenerate.mean() >= float(spec["min_fraction"]))

    is_high, is_low = _side_masks(ratio, float(spec["high"]), float(spec["low"]))

    if kind == "boundary_fraction":
        f = float(spec["min_fraction"])
        return bool(is_high.mean() >= f or is_low.mean() >= f)

    if kind == "run_and_fraction":
        f, run = float(spec["min_fraction"]), int(spec["min_run"])
        high = longest_run(is_high) >= run and is_high.mean() >= f
        low = longest_run(is_low) >= run and is_low.mean() >= f
        return bool(high or low)

    raise ValueError(f"unknown criterion type {kind!r}")


def extreme_flags(traj: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Evaluate every configured criterion for every patient.

    Returns one row per patient with a boolean column per criterion. Evaluating
    all of them rather than only the chosen one is deliberate: the choice of
    criterion is a methodological decision, and it can only be argued about if
    the consequences of each are on the table.
    """
    specs = cfg.trajectory["criteria"]
    names = sorted(specs)

    rows = []
    for patno, g in traj.groupby("PATNO", sort=True):
        ratio = g.sort_values("visit_index")["ratio"].to_numpy(dtype=float)
        row = {"PATNO": patno}
        for name in names:
            row[name] = _evaluate_criterion(ratio, specs[name])
        rows.append(row)

    return pd.DataFrame(rows, columns=["PATNO", *names])


def extreme_patient_summary(traj: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per-patient trajectory shape joined to the criterion flags.

    This is the table the extreme-criteria discussion is argued from, so it
    carries the evidence (counts, fractions, longest runs) beside the verdicts.
    """
    summary = trajectory_summary(traj)
    flags = extreme_flags(traj, cfg)

    specs = cfg.trajectory["criteria"]
    boundary = next(
        (s for s in specs.values() if s["type"] in ("boundary_fraction", "run_and_fraction")),
        None,
    )

    if boundary is not None:
        high, low = float(boundary["high"]), float(boundary["low"])
        runs = []
        for patno, g in traj.groupby("PATNO", sort=True):
            ratio = g.sort_values("visit_index")["ratio"].to_numpy(dtype=float)
            is_high, is_low = _side_masks(ratio, high, low)
            runs.append(
                {
                    "PATNO": patno,
                    "frac_high": float(is_high.mean()),
                    "frac_low": float(is_low.mean()),
                    "longest_run_high": longest_run(is_high),
                    "longest_run_low": longest_run(is_low),
                }
            )
        summary = summary.merge(pd.DataFrame(runs), on="PATNO", how="left")

    return summary.merge(flags, on="PATNO", how="left")


# ---------------------------------------------------------------------------
#  Cohorts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cohort:
    """A named analysis population and the trajectories it contains.

    `traj` holds every labelled visit of the retained patients, including those
    whose ratio is undefined, so the table stays a complete record. `finite`
    holds only what a ratio-based model can see.
    """

    name: str
    criterion: str | None
    traj: pd.DataFrame
    finite: pd.DataFrame
    removed_patients: list
    min_finite_visits: int

    @property
    def patients(self) -> list:
        return sorted(self.finite["PATNO"].unique().tolist())

    @property
    def n_patients(self) -> int:
        return int(self.finite["PATNO"].nunique())

    @property
    def n_finite_visits(self) -> int:
        return len(self.finite)

    @property
    def n_prediction_points(self) -> int:
        return self.n_finite_visits - self.n_patients

    def describe(self) -> dict:
        summary = trajectory_summary(self.traj)
        kept = summary[summary["PATNO"].isin(self.patients)]
        return {
            "cohort": self.name,
            "criterion": self.criterion or "none",
            "patients": self.n_patients,
            "labelled_visits": int(self.traj["PATNO"].isin(self.patients).sum()),
            "finite_visits": self.n_finite_visits,
            "prediction_points": self.n_prediction_points,
            "patients_removed": len(self.removed_patients),
            "never_changed": int((~kept["ever_changed"]).sum()),
            "ever_changed": int(kept["ever_changed"].sum()),
            "median_finite_visits": float(kept["n_finite"].median()),
            "median_span_years": float(kept["span_years"].median()),
        }


def build_cohort(
    traj: pd.DataFrame,
    cfg: Config,
    criterion: str | None = None,
    min_finite_visits: int | None = None,
) -> Cohort:
    """Select the analysis population.

    Two reductions happen here and they are not the same kind of thing.

    The *structural* one drops patients with fewer than `min_finite_visits`
    finite ratios. They are not excluded on any judgement about their disease;
    a model defined on the ratio simply cannot see them.

    The *extreme* one applies `criterion`, if given. That is a methodological
    choice, and the patients it removes are reported separately so the effect
    of the choice stays visible.
    """
    if min_finite_visits is None:
        min_finite_visits = int(cfg.trajectory["min_finite_visits"])

    removed: list = []
    kept = traj
    if criterion is not None:
        specs = cfg.trajectory["criteria"]
        if criterion not in specs:
            raise ValueError(
                f"unknown criterion {criterion!r}; config defines {sorted(specs)}"
            )
        flags = extreme_flags(traj, cfg)
        removed = sorted(flags.loc[flags[criterion], "PATNO"].tolist())
        kept = traj[~traj["PATNO"].isin(removed)]

    finite = finite_only(kept)
    counts = finite.groupby("PATNO").size()
    eligible = set(counts[counts >= min_finite_visits].index)

    finite = finite[finite["PATNO"].isin(eligible)].copy()
    finite["finite_index"] = finite.groupby("PATNO").cumcount()

    name = "unfiltered" if criterion is None else f"{criterion}-filtered"
    return Cohort(
        name=name,
        criterion=criterion,
        traj=kept.reset_index(drop=True),
        finite=finite.reset_index(drop=True),
        removed_patients=removed,
        min_finite_visits=min_finite_visits,
    )


# ---------------------------------------------------------------------------
#  Prediction points
# ---------------------------------------------------------------------------
def one_step_points(cohort: Cohort) -> pd.DataFrame:
    """Every (patient, target) pair the one-step-ahead evaluation will score.

    The target is finite-ratio visit ``k``; the history is visits ``0..k-1`` of
    the same patient. ``history_length`` is therefore ``k``, and the target is
    never part of its own history --- a property asserted in
    :func:`check_no_leakage` rather than left to the reader to verify.
    """
    rows = []
    for patno, g in cohort.finite.groupby("PATNO", sort=True):
        g = g.sort_values("finite_index")
        t = g["years_from_baseline"].to_numpy(dtype=float)
        r = g["ratio"].to_numpy(dtype=float)
        subtype = g["subtype_name"].to_numpy()
        for k in range(1, len(g)):
            rows.append(
                {
                    "PATNO": patno,
                    "target_index": k,
                    "history_length": k,
                    "t_target": float(t[k]),
                    "t_last_history": float(t[k - 1]),
                    "horizon_years": float(t[k] - t[k - 1]),
                    "observed_ratio": float(r[k]),
                    "observed_subtype": subtype[k],
                    "previous_ratio": float(r[k - 1]),
                }
            )
    return pd.DataFrame(rows)


def multistep_points(cohort: Cohort, min_finite: int) -> pd.DataFrame:
    """Forecast points for the multi-step evaluation.

    The first ``m = ceil(T/2)`` finite observations are history; everything from
    ``m`` to ``T-1`` is forecast, at the times those visits actually happened.
    A patient needs `min_finite` observations to take part, so that the history
    supporting a multi-step forecast is never a single point.
    """
    rows = []
    for patno, g in cohort.finite.groupby("PATNO", sort=True):
        g = g.sort_values("finite_index")
        T = len(g)
        if T < min_finite:
            continue
        m = int(np.ceil(T / 2))
        if m >= T:
            continue
        t = g["years_from_baseline"].to_numpy(dtype=float)
        r = g["ratio"].to_numpy(dtype=float)
        subtype = g["subtype_name"].to_numpy()
        for k in range(m, T):
            rows.append(
                {
                    "PATNO": patno,
                    "target_index": k,
                    "history_length": m,
                    "step": k - m + 1,
                    "t_target": float(t[k]),
                    "t_last_history": float(t[m - 1]),
                    "horizon_years": float(t[k] - t[m - 1]),
                    "observed_ratio": float(r[k]),
                    "observed_subtype": subtype[k],
                }
            )
    return pd.DataFrame(rows)


def history_of(cohort: Cohort, patno, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Times and ratios of finite visits ``0..k-1`` for one patient.

    The single place a history is cut. Everything that predicts goes through
    here, so the "strictly before the target" rule is enforced once rather than
    re-implemented at each call site.
    """
    g = cohort.finite[cohort.finite["PATNO"] == patno].sort_values("finite_index")
    if k < 1 or k >= len(g):
        raise IndexError(
            f"patient {patno} has {len(g)} finite visits; target index {k} is not "
            f"predictable from earlier ones"
        )
    t = g["years_from_baseline"].to_numpy(dtype=float)[:k]
    r = g["ratio"].to_numpy(dtype=float)[:k]
    return t, r


def check_no_leakage(cohort: Cohort, points: pd.DataFrame) -> None:
    """Assert that no prediction point can see its own target or anything after.

    Checks the index arithmetic as well as the times. Times alone are not
    enough: two visits can share a timestamp when a date fails to parse and the
    nominal visit year is used instead, and an index check catches that where a
    strict time comparison would not.
    """
    if points.empty:
        return

    for _, row in points.iterrows():
        t_hist, _ = history_of(cohort, row["PATNO"], int(row["target_index"]))
        if len(t_hist) != int(row["history_length"]):
            raise ValueError(
                f"patient {row['PATNO']} target {row['target_index']}: history has "
                f"{len(t_hist)} points, row claims {row['history_length']}"
            )
        if len(t_hist) and t_hist[-1] > row["t_target"]:
            raise ValueError(
                f"patient {row['PATNO']} target {row['target_index']}: history ends "
                f"at {t_hist[-1]}, after the target time {row['t_target']}"
            )

    if (points["history_length"] > points["target_index"]).any():
        raise ValueError("a prediction point claims more history than precedes it")
    if (points["history_length"] < 1).any():
        raise ValueError("a prediction point has no history")
