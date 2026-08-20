"""Tests for the TD/PIGD classification rule.

The decision rule has four edge cases that a naive `tremor / pigd` division
gets wrong, and each one occurs in the real data:

  * pigd_score == 0, tremor_score > 0   -> TD  (division by zero)
  * both scores == 0                    -> Indeterminate (0/0)
  * tremor_score == 0, pigd_score > 0   -> PIGD (ratio 0)
  * exact threshold values              -> boundaries are closed on TD and PIGD

These are the tests that matter; they are asserted directly against the
derivation notes published with the PPMI curated data cut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype.config import SubtypeConfig, load_config
from ppmi_subtype.subtype import (
    MissingItemsError,
    classify,
    compute_subscores,
    count_unrateable,
    deduplicate_visits,
    select_medication_state,
)

TREMOR = tuple(f"T{i}" for i in range(11))
PIGD = tuple(f"P{i}" for i in range(5))


@pytest.fixture
def cfg() -> SubtypeConfig:
    return SubtypeConfig(
        tremor_items=TREMOR,
        pigd_items=PIGD,
        td_threshold=1.15,
        pigd_threshold=0.90,
        labels={"TD": 1, "Indeterminate": 2, "PIGD": 3},
        missing_value_codes=(101.0,),
        valid_score_range=(0.0, 4.0),
        max_missing_tremor_items=0,
        max_missing_pigd_items=0,
        medication_state="OFF",
        pdstate_column="PDSTATE",
        treatment_column="PDTRTMNT",
    )


def make_frame(tremor_mean, pigd_mean, n=1, **extra) -> pd.DataFrame:
    """Frame whose item columns average to the requested subscores."""
    data = {c: [tremor_mean] * n for c in TREMOR}
    data.update({c: [pigd_mean] * n for c in PIGD})
    data.update({k: [v] * n for k, v in extra.items()})
    return pd.DataFrame(data)


class TestDecisionRule:
    @pytest.mark.parametrize(
        "tremor, pigd, expected",
        [
            (2.0, 1.0, 1),      # ratio 2.00 -> TD
            (1.15, 1.0, 1),     # ratio exactly 1.15 -> TD, boundary closed
            (1.0, 1.0, 2),      # ratio 1.00 -> Indeterminate
            (0.95, 1.0, 2),     # inside (0.90, 1.15)
            (0.90, 1.0, 3),     # ratio exactly 0.90 -> PIGD, boundary closed
            (0.5, 1.0, 3),      # ratio 0.50 -> PIGD
            (0.0, 1.0, 3),      # no tremor at all -> PIGD
        ],
    )
    def test_ratio_thresholds(self, cfg, tremor, pigd, expected):
        result = classify(pd.Series([tremor]), pd.Series([pigd]), cfg)
        assert result.iloc[0] == expected

    def test_zero_pigd_with_tremor_is_td(self, cfg):
        """Division by zero must resolve to TD, not NaN or infinity."""
        result = classify(pd.Series([1.5]), pd.Series([0.0]), cfg)
        assert result.iloc[0] == cfg.labels["TD"]

    def test_both_zero_is_indeterminate(self, cfg):
        """0/0 must resolve to Indeterminate, not NaN."""
        result = classify(pd.Series([0.0]), pd.Series([0.0]), cfg)
        assert result.iloc[0] == cfg.labels["Indeterminate"]

    def test_missing_scores_stay_missing(self, cfg):
        """An unmeasured visit must not be assigned a class."""
        result = classify(pd.Series([np.nan, 1.0]), pd.Series([1.0, np.nan]), cfg)
        assert result.isna().all()

    def test_every_observed_visit_is_classified(self, cfg):
        """No combination of observed non-negative scores may yield NaN."""
        grid = [(t, p) for t in np.arange(0, 4.25, 0.25) for p in np.arange(0, 4.25, 0.25)]
        t = pd.Series([g[0] for g in grid])
        p = pd.Series([g[1] for g in grid])
        result = classify(t, p, cfg)
        assert result.notna().all(), "some observed score pair produced no label"
        assert set(result.unique()) <= set(cfg.labels.values())


class TestSubscores:
    def test_subscore_is_a_mean_not_a_sum(self, cfg):
        frame = make_frame(tremor_mean=2.0, pigd_mean=1.0)
        out = compute_subscores(frame, cfg)
        assert out["tremor_score"].iloc[0] == pytest.approx(2.0)
        assert out["pigd_score"].iloc[0] == pytest.approx(1.0)
        assert out["ratio"].iloc[0] == pytest.approx(2.0)

    def test_missing_item_nullifies_subscore(self, cfg):
        """PPMI's rule: any missing constituent sets the subscore to null."""
        frame = make_frame(2.0, 1.0)
        frame.loc[0, TREMOR[0]] = np.nan
        out = compute_subscores(frame, cfg)
        assert pd.isna(out["tremor_score"].iloc[0])
        assert pd.isna(out["subtype"].iloc[0])
        assert out["pigd_score"].iloc[0] == pytest.approx(1.0)

    def test_tolerated_missing_items_are_configurable(self, cfg):
        lenient = SubtypeConfig(**{**cfg.__dict__, "max_missing_tremor_items": 1})
        frame = make_frame(2.0, 1.0)
        frame.loc[0, TREMOR[0]] = np.nan
        out = compute_subscores(frame, lenient)
        assert out["tremor_score"].iloc[0] == pytest.approx(2.0)

    def test_ratio_undefined_flag_marks_zero_denominator(self, cfg):
        """The label exists but the continuous ratio does not; both are reported."""
        frame = make_frame(1.5, 0.0)
        out = compute_subscores(frame, cfg)
        assert out["subtype"].iloc[0] == cfg.labels["TD"]
        assert pd.isna(out["ratio"].iloc[0])
        assert not out["ratio_defined"].iloc[0]

    def test_missing_columns_raise_with_a_useful_message(self, cfg):
        frame = pd.DataFrame({TREMOR[0]: [1.0]})
        with pytest.raises(MissingItemsError, match="MDS-UPDRS"):
            compute_subscores(frame, cfg)

    def test_non_numeric_entries_are_coerced_not_crashed(self, cfg):
        frame = make_frame(2.0, 1.0)
        # PPMI uses codes such as UR ("unable to rate") in otherwise numeric fields.
        frame[TREMOR[0]] = frame[TREMOR[0]].astype(object)
        frame.loc[0, TREMOR[0]] = "UR"
        out = compute_subscores(frame, cfg)
        assert pd.isna(out["tremor_score"].iloc[0])


class TestUnrateableCodes:
    """PPMI stores 'Unable to Rate' as 101 in the same field as a 0-4 score.

    Averaging it as a score is not a small error: one 101 among the five PIGD
    items lifts their mean to ~20 and flips a tremor-dominant visit to PIGD.
    """

    def test_sentinel_is_not_averaged_as_a_score(self, cfg):
        frame = make_frame(2.0, 1.0)
        frame.loc[0, PIGD[0]] = 101
        out = compute_subscores(frame, cfg)
        assert out["pigd_score"].iloc[0] != pytest.approx(21.0)

    def test_sentinel_nullifies_the_subscore_under_strict_policy(self, cfg):
        frame = make_frame(2.0, 1.0)
        frame.loc[0, PIGD[0]] = 101
        out = compute_subscores(frame, cfg)
        assert pd.isna(out["pigd_score"].iloc[0])
        assert pd.isna(out["subtype"].iloc[0])

    def test_sentinel_would_otherwise_flip_the_label(self, cfg):
        """The regression this guards: TD becomes PIGD if 101 is scored."""
        clean = compute_subscores(make_frame(2.0, 1.0), cfg)
        assert clean["subtype"].iloc[0] == cfg.labels["TD"]

        lenient = SubtypeConfig(**{**cfg.__dict__, "max_missing_pigd_items": 1})
        frame = make_frame(2.0, 1.0)
        frame.loc[0, PIGD[0]] = 101
        out = compute_subscores(frame, lenient)
        # With the sentinel excluded the remaining four items still average 1.0.
        assert out["pigd_score"].iloc[0] == pytest.approx(1.0)
        assert out["subtype"].iloc[0] == cfg.labels["TD"]

    def test_any_out_of_range_value_is_treated_as_missing(self, cfg):
        """Guards against a sentinel PPMI introduces later, not just 101."""
        frame = make_frame(2.0, 1.0)
        frame.loc[0, PIGD[0]] = 999
        out = compute_subscores(frame, cfg)
        assert pd.isna(out["pigd_score"].iloc[0])

    def test_negative_values_are_treated_as_missing(self, cfg):
        frame = make_frame(2.0, 1.0)
        frame.loc[0, TREMOR[0]] = -1
        out = compute_subscores(frame, cfg)
        assert pd.isna(out["tremor_score"].iloc[0])

    def test_boundary_scores_are_kept(self, cfg):
        """0 and 4 are valid ratings and must survive the range filter."""
        frame = make_frame(0.0, 4.0)
        out = compute_subscores(frame, cfg)
        assert out["tremor_score"].iloc[0] == pytest.approx(0.0)
        assert out["pigd_score"].iloc[0] == pytest.approx(4.0)
        assert out["subtype"].iloc[0] == cfg.labels["PIGD"]

    def test_count_unrateable_reports_per_item(self, cfg):
        frame = make_frame(1.0, 1.0, n=3)
        frame["PATNO"] = [1, 2, 3]
        frame["EVENT_ID"] = ["BL", "BL", "BL"]
        frame.loc[[0, 1], PIGD[0]] = 101
        counts = count_unrateable(frame, cfg)
        assert counts[PIGD[0]] == 2
        assert counts[TREMOR[0]] == 0

    def test_deduplication_counts_sentinels_as_missing(self, cfg):
        """A row full of 101s must lose to a genuinely complete row."""
        frame = make_frame(1.0, 1.0, n=2)
        frame["PATNO"] = [1, 1]
        frame["EVENT_ID"] = ["BL", "BL"]
        frame.loc[0, list(PIGD)] = 101
        out = deduplicate_visits(frame, cfg)
        assert len(out) == 1
        assert (out[list(PIGD)] == 1.0).all(axis=None)


class TestMedicationState:
    def test_off_exam_selected(self, cfg):
        frame = make_frame(1.0, 1.0, n=2)
        frame["PDSTATE"] = ["OFF", "ON"]
        frame["PDTRTMNT"] = [1, 1]
        out = select_medication_state(frame, cfg)
        assert list(out["PDSTATE"]) == ["OFF"]

    def test_untreated_participants_are_kept(self, cfg):
        """Untreated participants have no ON/OFF state and must not be dropped."""
        frame = make_frame(1.0, 1.0, n=2)
        frame["PDSTATE"] = ["ON", "OFF"]
        frame["PDTRTMNT"] = [0, 1]
        out = select_medication_state(frame, cfg)
        assert len(out) == 2

    def test_any_state_is_a_passthrough(self, cfg):
        relaxed = SubtypeConfig(**{**cfg.__dict__, "medication_state": "ANY"})
        frame = make_frame(1.0, 1.0, n=3)
        frame["PDSTATE"] = ["ON", "OFF", None]
        assert len(select_medication_state(frame, relaxed)) == 3


class TestDeduplication:
    def test_prefers_the_row_with_fewest_missing_items(self, cfg):
        frame = make_frame(1.0, 1.0, n=2)
        frame["PATNO"] = [1, 1]
        frame["EVENT_ID"] = ["BL", "BL"]
        frame.loc[0, TREMOR[0]] = np.nan  # first row is the incomplete one
        out = deduplicate_visits(frame, cfg)
        assert len(out) == 1
        assert out[list(TREMOR)].isna().sum().sum() == 0

    def test_is_a_noop_when_already_unique(self, cfg):
        frame = make_frame(1.0, 1.0, n=2)
        frame["PATNO"] = [1, 2]
        frame["EVENT_ID"] = ["BL", "BL"]
        assert len(deduplicate_visits(frame, cfg)) == 2


class TestProjectConfig:
    """The shipped config must encode the published definition exactly."""

    def test_item_counts_match_stebbins(self):
        s = load_config().subtype
        assert len(s.tremor_items) == 11
        assert len(s.pigd_items) == 5

    def test_thresholds_match_published_definition(self):
        s = load_config().subtype
        assert s.td_threshold == 1.15
        assert s.pigd_threshold == 0.90

    def test_label_coding_matches_archived_td_pigd_old(self):
        """1/2/3 so recomputed labels compare directly against td_pigd_old."""
        s = load_config().subtype
        assert s.labels == {"TD": 1, "Indeterminate": 2, "PIGD": 3}

    def test_unable_to_rate_sentinel_is_configured(self):
        """101 appears in all 13 Part III items and must be excluded."""
        s = load_config().subtype
        assert 101.0 in s.missing_value_codes
        assert s.valid_score_range == (0.0, 4.0)
