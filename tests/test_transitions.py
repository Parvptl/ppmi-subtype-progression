"""Tests for the transition analysis.

The invariants that matter here are that a patient's sequence is never mixed
with another's, that unlabelled visits are skipped rather than breaking a
sequence, and that intervals are carried through honestly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype.transitions import (
    collapse_runs,
    consecutive_pairs,
    first_entry,
    first_transition_times,
    kaplan_meier,
    prevalence_by_year,
    reversion_after_change,
    reversion_summary,
    sojourn_summary,
    state_sequences,
    transition_counts_with_ci,
    transition_matrix,
    transition_matrix_by_interval,
)


def panel(rows) -> pd.DataFrame:
    """rows: (PATNO, years_from_baseline, subtype)."""
    return pd.DataFrame(
        [
            {
                "PATNO": p,
                "years_from_baseline": t,
                "YEAR": int(round(t)),
                "EVENT_ID": f"V{int(round(t)):02d}",
                "subtype": s,
            }
            for p, t, s in rows
        ]
    )


class TestConsecutivePairs:
    def test_pairs_are_within_patient(self):
        """The last visit of one patient must not pair with the first of the next."""
        df = panel([(1, 0, 1), (1, 1, 3), (2, 0, 2), (2, 1, 2)])
        pairs = consecutive_pairs(df)
        assert len(pairs) == 2
        assert set(pairs["PATNO"]) == {1, 2}
        assert not ((pairs["PATNO"] == 1) & (pairs["to_state"] == 2)).any()

    def test_unlabelled_visits_are_skipped_not_terminal(self):
        """A missed assessment must bridge, giving a 2-year pair, not truncate."""
        df = panel([(1, 0, 1), (1, 1, np.nan), (1, 2, 3)])
        pairs = consecutive_pairs(df)
        assert len(pairs) == 1
        assert pairs.iloc[0]["from_state"] == 1
        assert pairs.iloc[0]["to_state"] == 3
        assert pairs.iloc[0]["interval_years"] == pytest.approx(2.0)

    def test_single_visit_patient_contributes_nothing(self):
        pairs = consecutive_pairs(panel([(1, 0, 1)]))
        assert pairs.empty

    def test_intervals_reflect_actual_elapsed_time(self):
        df = panel([(1, 0.0, 1), (1, 2.5, 1)])
        pairs = consecutive_pairs(df)
        assert pairs.iloc[0]["interval_years"] == pytest.approx(2.5)

    def test_visits_are_ordered_by_time_not_file_order(self):
        df = panel([(1, 2, 3), (1, 0, 1), (1, 1, 2)])
        pairs = consecutive_pairs(df)
        assert list(pairs["from_state"]) == [1, 2]
        assert list(pairs["to_state"]) == [2, 3]


class TestTransitionMatrix:
    def test_rows_sum_to_one(self):
        df = panel([(1, 0, 1), (1, 1, 1), (1, 2, 3), (2, 0, 3), (2, 1, 1)])
        m = transition_matrix(consecutive_pairs(df))
        occupied = m.sum(axis=1) > 0
        assert np.allclose(m[occupied].sum(axis=1), 1.0)

    def test_all_states_present_even_when_unobserved(self):
        """An absent state must appear as a zero row, not vanish."""
        df = panel([(1, 0, 1), (1, 1, 1)])
        m = transition_matrix(consecutive_pairs(df), states=[1, 2, 3])
        assert list(m.index) == [1, 2, 3]
        assert list(m.columns) == [1, 2, 3]
        assert m.loc[1, 1] == pytest.approx(1.0)

    def test_unnormalised_returns_counts(self):
        df = panel([(1, 0, 1), (1, 1, 3), (2, 0, 1), (2, 1, 3)])
        counts = transition_matrix(consecutive_pairs(df), normalize=False)
        assert counts.loc[1, 3] == 2

    def test_empty_row_does_not_divide_by_zero(self):
        df = panel([(1, 0, 1), (1, 1, 1)])
        m = transition_matrix(consecutive_pairs(df), states=[1, 2, 3])
        assert m.loc[2].isna().all()


class TestConfidenceIntervals:
    def test_interval_brackets_the_point_estimate(self):
        df = panel([(i, t, 1 if t == 0 else (3 if i % 4 == 0 else 1)) for i in range(40) for t in (0, 1)])
        ci = transition_counts_with_ci(consecutive_pairs(df))
        row = ci[(ci.from_state == 1) & (ci.to_state == 3)].iloc[0]
        assert row["ci_low"] <= row["probability"] <= row["ci_high"]

    def test_wilson_interval_stays_in_unit_range(self):
        """Wilson is used precisely because the normal interval leaves [0, 1]."""
        df = panel([(i, t, 1) for i in range(5) for t in (0, 1)])
        ci = transition_counts_with_ci(consecutive_pairs(df))
        finite = ci.dropna(subset=["ci_low", "ci_high"])
        assert (finite["ci_low"] >= 0).all()
        assert (finite["ci_high"] <= 1).all()


class TestIntervalStratification:
    def test_short_and_long_intervals_are_separated(self):
        df = panel([(1, 0, 1), (1, 1, 1), (2, 0, 1), (2, 5, 3)])
        by_bin = transition_matrix_by_interval(consecutive_pairs(df), states=[1, 2, 3])
        assert len(by_bin) == 2
        pooled = transition_matrix(consecutive_pairs(df), states=[1, 2, 3])
        assert pooled.loc[1, 1] == pytest.approx(0.5)  # pooling hides the difference


class TestSummaries:
    def test_prevalence_proportions_sum_to_one(self):
        df = panel([(1, 0, 1), (2, 0, 3), (3, 0, 2)])
        prev = prevalence_by_year(df)
        prop_cols = [c for c in prev.columns if c.startswith("prop_")]
        assert np.allclose(prev[prop_cols].sum(axis=1), 1.0)

    def test_sojourn_reports_patients_not_just_rows(self):
        df = panel([(1, 0, 1), (1, 1, 1), (1, 2, 1), (2, 0, 1), (2, 1, 3)])
        s = sojourn_summary(consecutive_pairs(df)).set_index("state")
        assert s.loc[1, "n_transitions_observed"] == 3
        assert s.loc[1, "n_patients"] == 2

    def test_first_transition_time_takes_the_earliest(self):
        df = panel([(1, 0, 1), (1, 1, 3), (1, 2, 1), (1, 3, 3)])
        ft = first_transition_times(consecutive_pairs(df), from_state=1, to_state=3)
        assert len(ft) == 1
        assert ft.iloc[0]["years_to_transition"] == pytest.approx(1.0)


class TestSequencesAndPathways:
    def test_sequence_is_in_time_order(self):
        df = panel([(1, 2, 3), (1, 0, 1), (1, 1, 2)])
        assert state_sequences(df).loc[1] == (1, 2, 3)

    def test_collapse_runs_drops_immediate_repeats_only(self):
        assert collapse_runs((1, 1, 3, 3, 1)) == (1, 3, 1)

    def test_collapse_runs_preserves_a_single_state(self):
        assert collapse_runs((1, 1, 1)) == (1,)


class TestReversion:
    def test_reversion_detected(self):
        """TD -> PIGD -> TD is a round trip, not progression."""
        df = panel([(1, 0, 1), (1, 1, 3), (1, 2, 1)])
        rev = reversion_after_change(df)
        assert len(rev) == 1
        assert bool(rev.iloc[0]["reverted"])

    def test_sustained_change_is_not_a_reversion(self):
        df = panel([(1, 0, 1), (1, 1, 3), (1, 2, 3)])
        rev = reversion_after_change(df)
        assert not bool(rev.iloc[0]["reverted"])

    def test_stable_runs_contribute_nothing(self):
        """Only actual changes enter the denominator."""
        df = panel([(1, 0, 1), (1, 1, 1), (1, 2, 1)])
        assert reversion_after_change(df).empty

    def test_requires_a_third_visit(self):
        df = panel([(1, 0, 1), (1, 1, 3)])
        assert reversion_after_change(df).empty

    def test_summary_rates_are_per_transition_type(self):
        df = panel(
            [(1, 0, 1), (1, 1, 3), (1, 2, 1)]        # TD->PIGD reverts
            + [(2, 0, 1), (2, 1, 3), (2, 2, 3)]      # TD->PIGD sustained
        )
        s = reversion_summary(reversion_after_change(df))
        row = s[(s.from_state == 1) & (s.to_state == 3)].iloc[0]
        assert row["n"] == 2
        assert row["p_revert"] == pytest.approx(0.5)


class TestFirstEntry:
    def test_single_visit_entry(self):
        df = panel([(1, 0, 1), (1, 1, 3)])
        fe = first_entry(df, target_state=3).set_index("PATNO")
        assert bool(fe.loc[1, "converted"])
        assert fe.loc[1, "time"] == pytest.approx(1.0)

    def test_sustained_requirement_rejects_a_one_visit_excursion(self):
        """The distinction that separates progression from assessment noise."""
        df = panel([(1, 0, 1), (1, 1, 3), (1, 2, 1)])
        assert bool(first_entry(df, 3, min_consecutive=1).iloc[0]["converted"])
        assert not bool(first_entry(df, 3, min_consecutive=2).iloc[0]["converted"])

    def test_sustained_requirement_accepts_two_in_a_row(self):
        df = panel([(1, 0, 1), (1, 1, 3), (1, 2, 3)])
        fe = first_entry(df, 3, min_consecutive=2).iloc[0]
        assert bool(fe["converted"])
        assert fe["time"] == pytest.approx(1.0)

    def test_never_converting_is_censored_not_dropped(self):
        df = panel([(1, 0, 1), (1, 1, 1), (1, 2, 1)])
        fe = first_entry(df, 3).iloc[0]
        assert not bool(fe["converted"])
        assert fe["time"] == pytest.approx(2.0)

    def test_patients_starting_in_the_state_are_flagged(self):
        df = panel([(1, 0, 3), (1, 1, 3)])
        assert bool(first_entry(df, 3).iloc[0]["starts_in_state"])

    def test_every_patient_appears_exactly_once(self):
        df = panel([(1, 0, 1), (1, 1, 3), (2, 0, 1), (2, 1, 1), (3, 0, 3)])
        fe = first_entry(df, 3)
        assert len(fe) == 3
        assert fe["PATNO"].is_unique


class TestKaplanMeier:
    def test_survival_is_monotone_non_increasing(self):
        km = kaplan_meier([1, 2, 3, 4, 5], [True, False, True, True, False])
        assert (km["survival"].diff().dropna() <= 1e-12).all()

    def test_no_events_leaves_survival_at_one(self):
        km = kaplan_meier([1, 2, 3], [False, False, False])
        assert km["survival"].min() == pytest.approx(1.0)

    def test_all_events_drives_survival_to_zero(self):
        km = kaplan_meier([1, 2, 3], [True, True, True])
        assert km["survival"].iloc[-1] == pytest.approx(0.0)

    def test_matches_hand_computation(self):
        """5 at risk, event at t=1 -> 0.8; then 4 at risk, event at t=3 -> 0.6."""
        km = kaplan_meier([1, 2, 3, 4, 5], [True, False, True, False, False])
        at3 = km[km["time"] == 3].iloc[0]
        assert at3["survival"] == pytest.approx(0.8 * (1 - 1 / 3))

    def test_censoring_removes_from_risk_set_without_an_event(self):
        km = kaplan_meier([1, 2], [False, True])
        assert km["survival"].iloc[-1] == pytest.approx(0.0)
