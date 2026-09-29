"""Tests for trajectory construction, extreme profiles and cohort selection.

The load-bearing tests here are the leakage ones. Everything downstream claims
that a prediction for visit k saw only visits 0..k-1; that claim rests entirely
on the ordering and slicing in this module, so both are checked directly rather
than inferred from a plausible-looking result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype import trajectory as tj
from ppmi_subtype.config import Config, project_root


def make_config(**trajectory_overrides) -> Config:
    """Minimal config carrying only what the trajectory module reads."""
    raw = {
        "seed": 42,
        "cohort": {"min_visits_per_patient": 2},
        "trajectory": {
            "min_finite_visits": 2,
            "min_finite_visits_multistep": 3,
            "criteria": {
                "A4": {"type": "degenerate_fraction", "min_fraction": 0.80},
                "D1": {"type": "boundary_fraction", "high": 2.30, "low": 0.45,
                       "min_fraction": 0.80},
                "E2": {"type": "run_and_fraction", "high": 2.30, "low": 0.45,
                       "min_run": 3, "min_fraction": 0.60},
            },
            "primary": "A4",
            **trajectory_overrides,
        },
    }
    return Config(raw=raw, root=project_root())


def make_panel(patients: dict) -> pd.DataFrame:
    """Build a panel from {PATNO: [ratio, ...]}, one visit per year.

    Ratios may be NaN to represent a visit whose PIGD subscore is zero. Every
    visit carries a subtype, because a labelled visit with an undefined ratio is
    exactly the case the trajectory table has to keep.
    """
    rows = []
    for patno, ratios in patients.items():
        for k, r in enumerate(ratios):
            if pd.isna(r):
                name, code = "TD", 1.0
            elif r >= 1.15:
                name, code = "TD", 1.0
            elif r <= 0.90:
                name, code = "PIGD", 3.0
            else:
                name, code = "Indeterminate", 2.0
            rows.append(
                {
                    "PATNO": patno,
                    "EVENT_ID": f"V{k:02d}",
                    "visit_date": f"0{(k % 9) + 1}/2020",
                    "years_from_baseline": float(k),
                    "ratio": r,
                    "subtype": code,
                    "subtype_name": name,
                }
            )
    return pd.DataFrame(rows)


class TestConstruction:
    def test_visits_are_chronological_and_reindexed(self):
        panel = make_panel({1: [1.5, 1.2, 0.8]})
        shuffled = panel.iloc[[2, 0, 1]].reset_index(drop=True)

        traj = tj.build_trajectories(shuffled, make_config())

        assert list(traj["visit_index"]) == [0, 1, 2]
        assert list(traj["years_from_baseline"]) == [0.0, 1.0, 2.0]
        tj.check_ordering(traj)

    def test_interval_is_gap_from_previous_visit(self):
        panel = make_panel({1: [1.5, 1.2, 0.8]})
        traj = tj.build_trajectories(panel, make_config())

        intervals = traj["interval_years"].tolist()
        assert np.isnan(intervals[0])           # nothing precedes the first visit
        assert intervals[1:] == [1.0, 1.0]

    def test_undefined_ratio_visits_are_kept_and_flagged(self):
        panel = make_panel({1: [1.5, np.nan, 0.8]})
        traj = tj.build_trajectories(panel, make_config())

        assert len(traj) == 3, "an undefined ratio must not drop the visit"
        assert traj["ratio_defined"].tolist() == [True, False, True]

    def test_finite_only_reindexes_within_patient(self):
        panel = make_panel({1: [1.5, np.nan, 0.8, np.nan, 1.1]})
        finite = tj.finite_only(tj.build_trajectories(panel, make_config()))

        assert len(finite) == 3
        assert finite["finite_index"].tolist() == [0, 1, 2]
        # The time axis keeps the real gaps: the model must know that the
        # second usable observation is two years after the first, not one.
        assert finite["years_from_baseline"].tolist() == [0.0, 2.0, 4.0]

    def test_single_visit_patients_are_excluded_by_the_cohort_rule(self):
        panel = make_panel({1: [1.5], 2: [1.5, 0.8]})
        traj = tj.build_trajectories(panel, make_config())

        assert set(traj["PATNO"]) == {2}

    def test_check_ordering_rejects_a_mis_sorted_table(self):
        panel = make_panel({1: [1.5, 1.2, 0.8]})
        traj = tj.build_trajectories(panel, make_config())
        traj.loc[1, "years_from_baseline"] = 99.0

        with pytest.raises(ValueError, match="decreases"):
            tj.check_ordering(traj)


class TestSummary:
    def test_counts_and_ranges(self):
        panel = make_panel({1: [2.0, np.nan, 0.0, 1.0]})
        summary = tj.trajectory_summary(tj.build_trajectories(panel, make_config()))
        row = summary.iloc[0]

        assert row["n_labelled"] == 4
        assert row["n_finite"] == 3
        assert row["n_undefined"] == 1
        assert row["n_zero"] == 1
        assert row["ratio_min"] == 0.0 and row["ratio_max"] == 2.0
        assert row["frac_degenerate"] == pytest.approx(0.5)

    def test_ever_changed_uses_the_subtype_label(self):
        panel = make_panel({1: [2.0, 2.5], 2: [2.0, 0.5]})
        summary = tj.trajectory_summary(
            tj.build_trajectories(panel, make_config())
        ).set_index("PATNO")

        assert not summary.loc[1, "ever_changed"]
        assert summary.loc[2, "ever_changed"]


class TestExtremeCriteria:
    def test_patient_a_is_persistently_low_and_patient_b_is_not(self):
        """The distinction the whole criterion exists to make.

        Patient A sits below the PIGD threshold at every visit and never moves.
        Patient B wanders across the middle of the range. Only A is a sustained
        extreme profile; B is an ordinary trajectory that happens to start low.
        """
        panel = make_panel(
            {
                "A": [0.80, 0.82, 0.79, 0.81, 0.78],
                "B": [0.80, 1.02, 1.15, 0.97, 1.05],
            }
        )
        cfg = make_config()
        cfg.raw["trajectory"]["criteria"]["LOW"] = {
            "type": "boundary_fraction", "high": 2.30, "low": 0.90,
            "min_fraction": 0.80,
        }
        flags = tj.extreme_flags(
            tj.build_trajectories(panel, cfg), cfg
        ).set_index("PATNO")

        assert flags.loc["A", "LOW"]
        assert not flags.loc["B", "LOW"]

    def test_one_extreme_visit_does_not_make_an_extreme_patient(self):
        panel = make_panel({1: [1.2, 1.3, 9.9, 1.1, 1.4]})
        cfg = make_config()
        flags = tj.extreme_flags(tj.build_trajectories(panel, cfg), cfg)

        assert not flags.loc[0, "D1"]
        assert not flags.loc[0, "E2"]

    def test_degenerate_fraction_counts_both_kinds(self):
        """Undefined ratios and zero ratios are both ungradable."""
        cfg = make_config()
        # 4 of 5 degenerate: three undefined, one zero.
        panel = make_panel({1: [np.nan, np.nan, np.nan, 0.0, 1.5]})
        flags = tj.extreme_flags(tj.build_trajectories(panel, cfg), cfg)
        assert flags.loc[0, "A4"]

        # 3 of 5 is below the 80% bar.
        panel = make_panel({1: [np.nan, np.nan, 0.0, 1.5, 1.6]})
        flags = tj.extreme_flags(tj.build_trajectories(panel, cfg), cfg)
        assert not flags.loc[0, "A4"]

    def test_exactly_at_the_fraction_bound_is_included(self):
        """A patient with exactly 80% degenerate visits meets a ">= 80%" rule.

        Worth pinning down: computing the fraction as a sum of two separate
        proportions rather than in one division puts 8/10 a rounding error below
        0.8 and silently drops the patient.
        """
        cfg = make_config()
        panel = make_panel({1: [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 1.6]})
        flags = tj.extreme_flags(tj.build_trajectories(panel, cfg), cfg)
        assert flags.loc[0, "A4"]

    def test_undefined_ratio_counts_towards_the_high_side(self):
        """An undefined ratio is the limit of tremor/pigd as pigd falls to zero."""
        cfg = make_config()
        panel = make_panel({1: [np.nan, np.nan, np.nan, np.nan, 3.0]})
        flags = tj.extreme_flags(tj.build_trajectories(panel, cfg), cfg)
        assert flags.loc[0, "D1"]

    def test_run_and_fraction_needs_both_conditions(self):
        cfg = make_config()

        # A run of three, but only 3 of 8 visits extreme: fails the fraction.
        panel = make_panel({1: [3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]})
        assert not tj.extreme_flags(tj.build_trajectories(panel, cfg), cfg).loc[0, "E2"]

        # 4 of 6 extreme but never three in a row: fails the run.
        panel = make_panel({1: [3.0, 1.0, 3.0, 1.0, 3.0, 3.0]})
        assert not tj.extreme_flags(tj.build_trajectories(panel, cfg), cfg).loc[0, "E2"]

        # Three consecutive and 4 of 6: passes both.
        panel = make_panel({1: [3.0, 3.0, 3.0, 1.0, 1.0, 3.0]})
        assert tj.extreme_flags(tj.build_trajectories(panel, cfg), cfg).loc[0, "E2"]

    def test_longest_run(self):
        assert tj.longest_run(np.array([False, True, True, False, True])) == 2
        assert tj.longest_run(np.array([True, True, True])) == 3
        assert tj.longest_run(np.array([False, False])) == 0

    def test_criteria_never_look_at_the_subtype_label(self):
        """Relabelling every visit must not change a single verdict.

        A criterion defined on the label would condition on the outcome the
        model is scored against, which is why none of them may.
        """
        cfg = make_config()
        panel = make_panel({1: [3.0, 3.0, 3.0, 0.2], 2: [1.0, 1.2, 1.1, 1.3]})
        traj = tj.build_trajectories(panel, cfg)
        before = tj.extreme_flags(traj, cfg)

        relabelled = traj.copy()
        relabelled["subtype"] = 1.0
        relabelled["subtype_name"] = "TD"

        pd.testing.assert_frame_equal(before, tj.extreme_flags(relabelled, cfg))


class TestCohorts:
    def test_structural_filter_needs_two_finite_visits(self):
        panel = make_panel({1: [1.5, np.nan, np.nan], 2: [1.5, 1.2, np.nan]})
        cohort = tj.build_cohort(tj.build_trajectories(panel, make_config()), make_config())

        assert cohort.patients == [2]
        assert cohort.n_finite_visits == 2
        assert cohort.n_prediction_points == 1

    def test_removed_patients_are_reported_not_hidden(self):
        cfg = make_config()
        panel = make_panel({1: [np.nan, np.nan, np.nan, np.nan], 2: [1.5, 1.2, 0.9, 1.1]})
        cohort = tj.build_cohort(tj.build_trajectories(panel, cfg), cfg, criterion="A4")

        assert cohort.removed_patients == [1]
        assert cohort.criterion == "A4"
        assert cohort.describe()["patients_removed"] == 1

    def test_unknown_criterion_is_rejected(self):
        cfg = make_config()
        traj = tj.build_trajectories(make_panel({1: [1.5, 1.2]}), cfg)
        with pytest.raises(ValueError, match="unknown criterion"):
            tj.build_cohort(traj, cfg, criterion="nonsense")


class TestPredictionPoints:
    def test_one_step_points_start_at_the_second_visit(self):
        panel = make_panel({1: [1.5, 1.2, 0.8]})
        cohort = tj.build_cohort(tj.build_trajectories(panel, make_config()), make_config())
        points = tj.one_step_points(cohort)

        assert points["target_index"].tolist() == [1, 2]
        assert points["history_length"].tolist() == [1, 2]
        assert points["observed_ratio"].tolist() == [1.2, 0.8]
        assert points["horizon_years"].tolist() == [1.0, 1.0]

    def test_horizon_spans_the_real_gap(self):
        panel = make_panel({1: [1.5, np.nan, np.nan, 0.8]})
        cohort = tj.build_cohort(tj.build_trajectories(panel, make_config()), make_config())
        points = tj.one_step_points(cohort)

        assert points["horizon_years"].tolist() == [3.0]

    def test_history_is_strictly_before_the_target(self):
        """The single most important property in Step 3."""
        panel = make_panel({1: [1.5, 1.2, 0.8, 1.0, 0.7]})
        cohort = tj.build_cohort(tj.build_trajectories(panel, make_config()), make_config())

        for k in range(1, 5):
            t_hist, y_hist = tj.history_of(cohort, 1, k)
            assert len(t_hist) == k
            assert t_hist.max() < float(k), "history reaches the target's time"
            assert 0.8 not in y_hist[k:] if k < 2 else True
            # the target value itself must be absent from its own history
            target = cohort.finite.iloc[k]["ratio"]
            assert not np.any(t_hist >= cohort.finite.iloc[k]["years_from_baseline"])
            assert len(y_hist) == k and target not in y_hist[:k] or k == 0

    def test_history_of_rejects_an_unpredictable_index(self):
        panel = make_panel({1: [1.5, 1.2]})
        cohort = tj.build_cohort(tj.build_trajectories(panel, make_config()), make_config())

        with pytest.raises(IndexError):
            tj.history_of(cohort, 1, 0)       # nothing precedes the first visit
        with pytest.raises(IndexError):
            tj.history_of(cohort, 1, 2)       # past the end

    def test_leakage_check_passes_on_a_clean_cohort(self):
        panel = make_panel({1: [1.5, 1.2, 0.8], 2: [2.0, 1.1]})
        cohort = tj.build_cohort(tj.build_trajectories(panel, make_config()), make_config())
        tj.check_no_leakage(cohort, tj.one_step_points(cohort))

    def test_leakage_check_catches_an_overstated_history(self):
        panel = make_panel({1: [1.5, 1.2, 0.8]})
        cohort = tj.build_cohort(tj.build_trajectories(panel, make_config()), make_config())
        points = tj.one_step_points(cohort)
        points.loc[0, "history_length"] = 2      # claims to see the target

        with pytest.raises(ValueError):
            tj.check_no_leakage(cohort, points)

    def test_multistep_splits_at_half_the_trajectory(self):
        panel = make_panel({1: [1.5, 1.4, 1.3, 1.2, 1.1]})
        cohort = tj.build_cohort(tj.build_trajectories(panel, make_config()), make_config())
        points = tj.multistep_points(cohort, min_finite=3)

        # T = 5, so m = 3: visits 3 and 4 are forecast from visits 0-2.
        assert points["history_length"].unique().tolist() == [3]
        assert points["target_index"].tolist() == [3, 4]
        assert points["step"].tolist() == [1, 2]

    def test_multistep_skips_patients_below_the_minimum(self):
        panel = make_panel({1: [1.5, 1.2], 2: [1.5, 1.2, 1.0, 0.9]})
        cohort = tj.build_cohort(tj.build_trajectories(panel, make_config()), make_config())
        points = tj.multistep_points(cohort, min_finite=3)

        assert set(points["PATNO"]) == {2}
