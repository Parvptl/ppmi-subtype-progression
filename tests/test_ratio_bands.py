"""Correctness checks for the joint (subtype, ratio-band) observation alphabet.

These cover the rules the ratio-banded experiment's conclusions depend on:
that band assignment is exclusive and exhaustive, that a value sitting exactly
on an edge or exactly on a classification threshold lands where the documented
convention says it does, that tail merging is deterministic and uses only the
data it is handed, that an unavailable ratio is a category rather than a gap,
and that a distribution over joint categories sums back to the three-label
distribution the evaluation is defined on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype import ratio_bands as rb

SUBTYPES = ["TD", "Indeterminate", "PIGD"]


def _scheme(tail_lows=(3.0, 3.5), boundaries="outer") -> rb.BandScheme:
    return rb.BandScheme(tuple(tail_lows), td_threshold=1.15,
                         pigd_threshold=0.90, band_boundaries=boundaries)


# ---------------------------------------------------------------------------
#  The fixed main range
# ---------------------------------------------------------------------------
def test_main_range_is_thirty_bands_of_width_one_tenth():
    edges = rb.main_edges()
    assert len(edges) == 31
    assert edges[0] == 0.0 and edges[-1] == 3.0
    assert np.allclose(np.diff(edges), 0.1)
    # Exact decimal edges, not 0.30000000000000004: the band label and the
    # assignment must agree.
    assert edges[3] == 0.3 and edges[11] == 1.1


def test_band_assignment_is_left_closed_right_open():
    s = _scheme()
    # A value exactly on an edge opens that band.
    assert s.band_index(np.array([0.0]))[0] == 0
    assert s.band_index(np.array([0.1]))[0] == 1
    assert s.band_index(np.array([0.0999999]))[0] == 0
    assert s.band_index(np.array([1.2]))[0] == 12
    lo, hi = s.band_bounds(1)
    assert (lo, hi) == (0.1, 0.2)


def test_every_band_is_reachable_and_no_ratio_falls_in_two_bands():
    s = _scheme(tail_lows=(3.0, 3.25, 3.75))
    # One probe just inside each band; each must return its own band and only it.
    for k in range(s.n_bands):
        lo, hi = s.band_bounds(k)
        probe = lo if np.isinf(hi) else (lo + hi) / 2
        assert s.band_index(np.array([probe]))[0] == k
    # Exhaustive: a dense sweep over the whole range, plus values above the
    # topmost edge, always lands in exactly one band.
    sweep = np.concatenate([np.linspace(0.0, 4.5, 4001), [10.0, 1e6]])
    idx = s.band_index(sweep)
    assert idx.min() >= 0 and idx.max() == s.n_bands - 1
    assert (idx == s.n_bands - 1).any()          # the open-topped band catches them


def test_a_ratio_below_the_lowest_edge_is_an_error_not_a_silent_band():
    with pytest.raises(ValueError, match="below the lowest band edge"):
        _scheme().band_index(np.array([-0.01]))


def test_missing_ratio_is_marked_rather_than_assigned():
    s = _scheme()
    idx = s.band_index(np.array([np.nan, 1.0, np.nan]))
    assert list(idx) == [-1, 10, -1]


# ---------------------------------------------------------------------------
#  Subtype compatibility, including the exact thresholds
# ---------------------------------------------------------------------------
def test_threshold_straddling_bands_carry_both_subtypes():
    s = _scheme()
    # 1.15 is the TD threshold and sits inside [1.1, 1.2), so that band can
    # hold an Indeterminate visit (1.10-1.15) and a TD one (1.15-1.20).
    assert rb.band_allows_subtype(s, 11, "Indeterminate")
    assert rb.band_allows_subtype(s, 11, "TD")
    assert not rb.band_allows_subtype(s, 11, "PIGD")
    # Under the published rule PIGD owns 0.90 exactly, which opens [0.9, 1.0).
    assert rb.band_allows_subtype(s, 9, "PIGD")
    assert rb.band_allows_subtype(s, 9, "Indeterminate")
    assert not rb.band_allows_subtype(s, 9, "TD")


def test_inner_boundaries_move_the_threshold_point_to_the_band():
    inner = _scheme(boundaries="inner")
    # "inner" gives the Indeterminate band both endpoints, so 0.90 is no
    # longer PIGD and [0.9, 1.0) can no longer hold a PIGD visit.
    assert not rb.band_allows_subtype(inner, 9, "PIGD")
    assert rb.band_allows_subtype(inner, 9, "Indeterminate")


def test_subtype_blocks_are_contiguous_and_cover_every_possible_pair():
    s = _scheme(tail_lows=(3.0, 3.5))
    a = rb.build_alphabet(s, SUBTYPES)

    assert a.n_categories == sum(len(b) for b in a.blocks)
    assert [c for block in a.blocks for c in block] == list(range(a.n_categories))
    for i, block in enumerate(a.blocks):
        assert list(block) == list(range(block[0], block[0] + len(block)))
        assert {a.pairs[c][0] for c in block} == {i}

    # Every structurally possible pair is present, and nothing else is.
    expected = {(i, b) for i, name in enumerate(SUBTYPES)
                for b in list(range(s.n_bands)) + [-1]
                if rb.band_allows_subtype(s, b, name)}
    assert set(a.pairs) == expected


def test_an_unavailable_ratio_is_a_category_for_td_and_indeterminate_only():
    a = rb.build_alphabet(_scheme(), SUBTYPES)
    undefined = {a.subtype_names[s] for s, b in a.pairs if b < 0}
    # pigd_score == 0 with tremor > 0 is TD by rule, both zero is
    # Indeterminate; a PIGD label needs a positive PIGD subscore, so there is
    # no PIGD category with an unavailable ratio.
    assert undefined == {"TD", "Indeterminate"}


def test_code_rejects_a_pair_the_rule_cannot_produce():
    a = rb.build_alphabet(_scheme(), SUBTYPES)
    pigd = SUBTYPES.index("PIGD")
    with pytest.raises(ValueError, match="not a category"):
        a.code(np.array([pigd]), np.array([-1]))      # PIGD with no ratio
    with pytest.raises(ValueError, match="not a category"):
        a.code(np.array([SUBTYPES.index("TD")]), np.array([0]))   # TD at ratio ~0


def test_codes_round_trip_to_the_subtype_they_came_from():
    s = _scheme()
    a = rb.build_alphabet(s, SUBTYPES)
    ratios = np.array([0.05, 0.85, 0.95, 1.05, 1.12, 1.18, 2.5, 3.1, 8.0, np.nan])
    subtypes = np.array([2, 2, 1, 1, 1, 0, 0, 0, 0, 0])
    codes = a.code(subtypes, s.band_index(ratios))
    assert list(a.subtype_of[codes]) == list(subtypes)


# ---------------------------------------------------------------------------
#  Dynamic tail merging
# ---------------------------------------------------------------------------
def _tail(counts: dict[float, int], patients_per_band: int | None = None):
    """Synthetic tail observations: `counts` maps a band's lower edge to a size."""
    ratios, patients = [], []
    pid = 0
    for lo, n in counts.items():
        for i in range(n):
            ratios.append(lo + 0.1)
            patients.append(f"p{pid + (i if patients_per_band is None else i % patients_per_band)}")
        pid += max(n, 1)
    return np.array(ratios, dtype=float), np.array(patients, dtype=object)


def test_tail_bands_large_enough_are_left_alone():
    r, p = _tail({3.0: 40, 3.25: 40, 3.5: 40})
    lows, diag = rb.merge_tail(r, p, min_observations=30, min_patients=20)
    assert list(lows) == [3.0, 3.25, 3.5]
    assert list(diag["merged_band_index"]) == [0, 1, 2]


def test_a_sparse_tail_band_merges_upward_into_its_neighbour():
    r, p = _tail({3.0: 5, 3.25: 40, 3.5: 40})
    lows, diag = rb.merge_tail(r, p, min_observations=30, min_patients=20)
    # [3.00,3.25) alone is too small, so it joins [3.25,3.50) and the merged
    # band runs [3.00,3.50).
    assert list(lows) == [3.0, 3.5]
    assert list(diag["merged_band_index"]) == [0, 0, 1]
    assert diag.loc[0, "merged_low"] == 3.0 and diag.loc[0, "merged_high"] == 3.5


def test_a_sparse_final_band_is_absorbed_into_the_one_below():
    r, p = _tail({3.0: 40, 3.25: 40, 3.5: 3})
    lows, diag = rb.merge_tail(r, p, min_observations=30, min_patients=20)
    assert list(lows) == [3.0, 3.25]
    assert diag.loc[2, "merged_band_index"] == 1
    assert np.isinf(diag.loc[2, "merged_high"])


def test_the_patient_minimum_can_bind_when_the_count_does_not():
    # 40 observations but only 4 distinct patients: a longitudinal band filled
    # by repeat visits is not 40 independent observations.
    r, p = _tail({3.0: 40, 3.25: 40}, patients_per_band=4)
    lows, _ = rb.merge_tail(r, p, min_observations=30, min_patients=20)
    assert list(lows) == [3.0]


def test_an_empty_tail_gives_one_open_band():
    lows, diag = rb.merge_tail(np.array([0.5, 1.2]), np.array(["a", "b"], dtype=object))
    assert list(lows) == [3.0]
    assert diag["n_observations"].sum() == 0


def test_merging_ignores_everything_below_the_tail_start():
    below = np.full(500, 1.0)
    r, p = _tail({3.0: 40, 3.25: 40})
    lows_with = rb.merge_tail(np.concatenate([r, below]),
                              np.concatenate([p, np.array([f"x{i}" for i in range(500)],
                                                          dtype=object)]))[0]
    lows_without = rb.merge_tail(r, p)[0]
    assert list(lows_with) == list(lows_without)


def test_merging_is_deterministic_and_order_independent():
    r, p = _tail({3.0: 7, 3.25: 40, 3.5: 9, 3.75: 40})
    first = rb.merge_tail(r, p)[0]
    order = np.random.default_rng(1).permutation(len(r))
    assert list(rb.merge_tail(r[order], p[order])[0]) == list(first)
    assert list(rb.merge_tail(r, p)[0]) == list(first)


def test_merging_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        rb.merge_tail(np.array([3.1, 3.2]), np.array(["a"], dtype=object))


def test_the_top_band_is_open_so_an_unseen_larger_ratio_is_still_assignable():
    r, p = _tail({3.0: 40, 3.25: 40})
    lows, _ = rb.merge_tail(r, p)
    s = rb.BandScheme(tuple(lows), 1.15, 0.90)
    # Nothing at 50.0 was ever seen in training, yet it has a band.
    assert s.band_index(np.array([50.0]))[0] == s.n_bands - 1


# ---------------------------------------------------------------------------
#  Aggregation back to the three labels
# ---------------------------------------------------------------------------
def test_aggregation_sums_within_subtype_and_normalises():
    a = rb.build_alphabet(_scheme(), SUBTYPES)
    rng = np.random.default_rng(0)
    probs = rng.random(a.n_categories)
    probs /= probs.sum()

    out = rb.aggregate_to_subtype(probs, a)
    assert out.shape == (3,)
    assert np.isclose(out.sum(), 1.0)
    for i, block in enumerate(a.blocks):
        assert np.isclose(out[i], probs[list(block)].sum())


def test_aggregation_normalises_an_unnormalised_input():
    a = rb.build_alphabet(_scheme(), SUBTYPES)
    probs = np.full(a.n_categories, 2.0)
    out = rb.aggregate_to_subtype(probs, a)
    assert np.isclose(out.sum(), 1.0)
    assert np.allclose(out, [len(b) / a.n_categories for b in a.blocks])


def test_aggregation_handles_a_stack_and_rejects_the_wrong_width():
    a = rb.build_alphabet(_scheme(), SUBTYPES)
    stack = np.full((4, a.n_categories), 1.0 / a.n_categories)
    assert rb.aggregate_to_subtype(stack, a).shape == (4, 3)
    with pytest.raises(ValueError, match="joint categories"):
        rb.aggregate_to_subtype(np.ones((2, 5)), a)
    with pytest.raises(ValueError, match="positive total"):
        rb.aggregate_to_subtype(np.zeros(a.n_categories), a)


def test_the_flat_alphabet_is_one_category_per_subtype():
    a = rb.flat_alphabet(SUBTYPES)
    assert a.n_categories == 3
    assert a.blocks == ((0,), (1,), (2,))
    assert list(a.code(np.array([0, 1, 2]), np.array([-1, -1, -1]))) == [0, 1, 2]
    probs = np.array([0.2, 0.3, 0.5])
    assert np.allclose(rb.aggregate_to_subtype(probs, a), probs)


# ---------------------------------------------------------------------------
#  Interval arithmetic, which the compatibility rule rests on
# ---------------------------------------------------------------------------
def test_interval_intersection_respects_closure():
    closed_at_one = ((0.0, True), (1.0, True))
    open_at_one = ((1.0, False), (2.0, False))
    closed_from_one = ((1.0, True), (2.0, True))
    assert not rb._intersects(closed_at_one, open_at_one)
    assert rb._intersects(closed_at_one, closed_from_one)
    assert rb._intersects(((0.0, True), (1.0, False)), ((0.5, True), (3.0, False)))
    assert not rb._intersects(((0.0, True), (1.0, False)), ((1.0, True), (3.0, False)))
