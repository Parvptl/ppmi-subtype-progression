"""The main-range band width is a free parameter, and varying it stays exact.

The band-width sensitivity experiment changes one thing about the published
ratio-banded CT-HMM: how wide the fixed main-range bands are. These tests pin
down that the change is exactly that --- the default still produces the
published edges bit for bit, every width tiles [0, 3) without gaps or overlaps,
and a ratio sitting on an edge lands in one band and only one.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppmi_subtype import next_visit_banded as nvb, ratio_bands as rb

WIDTHS = (0.05, 0.10, 0.20, 0.30, 0.50)


def scheme_for(width, tail_lows=(3.0, 3.5)):
    return rb.BandScheme(tuple(tail_lows), td_threshold=1.15, pigd_threshold=0.90,
                         main_width=width)


# --------------------------------------------------------------- the edges
def test_the_default_width_reproduces_the_published_edges_exactly():
    # The published experiment's edges came from `arange(31) / 10.0`. Anything
    # that moves them by one ulp would silently re-band observations sitting on
    # a boundary, so this is an equality and not a tolerance.
    assert np.array_equal(rb.main_edges(), np.arange(31) / 10.0)
    assert rb.main_edges().shape == (31,)


def test_each_width_divides_the_main_range_a_whole_number_of_times():
    for width, n in zip(WIDTHS, (60, 30, 15, 10, 6)):
        edges = rb.main_edges(width)
        assert len(edges) - 1 == n
        assert edges[0] == rb.MAIN_LOW
        assert edges[-1] == rb.MAIN_HIGH


def test_the_edges_are_the_exact_decimals():
    # 0.3 is not representable, so `k * 0.3` drifts: 3 * 0.3 is 0.8999999999999999,
    # which would put a ratio of exactly 0.9 in the band below. The edges must be
    # the correctly rounded decimal instead.
    for width in WIDTHS:
        edges = rb.main_edges(width)
        assert np.array_equal(edges, np.round(edges, 10))
    assert 0.9 in set(rb.main_edges(0.30).tolist())


def test_a_width_that_does_not_divide_the_range_is_rejected():
    for bad in (0.07, 0.4, 3.5, 0.0):
        with pytest.raises((ValueError, ZeroDivisionError)):
            rb.main_edges(bad)


# ----------------------------------------------------------- the tiling
def test_every_width_tiles_the_main_range_without_gaps_or_overlaps():
    for width in WIDTHS:
        scheme = scheme_for(width)
        # consecutive bands meet exactly: one's high is the next one's low
        for b in range(scheme.n_main_bands - 1):
            assert scheme.band_bounds(b)[1] == scheme.band_bounds(b + 1)[0]
        assert scheme.band_bounds(0)[0] == rb.MAIN_LOW
        assert scheme.band_bounds(scheme.n_main_bands - 1)[1] == rb.TAIL_START


def test_a_ratio_on_an_edge_opens_the_band_above_it_at_every_width():
    for width in WIDTHS:
        scheme = scheme_for(width)
        edges = rb.main_edges(width)[:-1]
        idx = scheme.band_index(edges)
        assert np.array_equal(idx, np.arange(len(edges)))
        # and a hair below an edge belongs to the band below
        interior = edges[1:] - 1e-12
        assert np.array_equal(scheme.band_index(interior), np.arange(len(edges) - 1))


def test_no_observation_can_belong_to_two_bands_at_any_width():
    rng = np.random.default_rng(0)
    ratios = np.concatenate([rng.uniform(0, 4, 500), rb.main_edges(0.05)])
    for width in WIDTHS:
        scheme = scheme_for(width)
        idx = scheme.band_index(ratios)
        assert (idx >= 0).all()
        for k, r in zip(idx, ratios):
            lo, hi = scheme.band_bounds(int(k))
            assert lo <= r < hi


def test_the_band_count_falls_as_the_width_grows():
    counts = [scheme_for(w).n_main_bands for w in WIDTHS]
    assert counts == sorted(counts, reverse=True)
    assert counts == [60, 30, 15, 10, 6]


# ------------------------------------------------- the joint alphabet
def test_the_alphabet_keeps_the_subtype_at_every_width():
    names = ("TD", "Indeterminate", "PIGD")
    for width in WIDTHS:
        alphabet = rb.build_alphabet(scheme_for(width), names)
        assert len(alphabet.blocks) == 3
        assert sum(len(b) for b in alphabet.blocks) == alphabet.n_categories
        # the blocks partition the categories, so a joint distribution always
        # aggregates back to exactly three numbers
        seen = sorted(k for block in alphabet.blocks for k in block)
        assert seen == list(range(alphabet.n_categories))
        assert all(label.split("|")[0] in names for label in alphabet.labels)


def test_a_finer_width_never_produces_fewer_categories():
    names = ("TD", "Indeterminate", "PIGD")
    sizes = [rb.build_alphabet(scheme_for(w), names).n_categories for w in WIDTHS]
    assert sizes == sorted(sizes, reverse=True)


def test_aggregation_still_sums_to_one_at_every_width():
    names = ("TD", "Indeterminate", "PIGD")
    rng = np.random.default_rng(1)
    for width in WIDTHS:
        alphabet = rb.build_alphabet(scheme_for(width), names)
        joint = rng.dirichlet(np.ones(alphabet.n_categories), size=7)
        agg = rb.aggregate_to_subtype(joint, alphabet)
        assert agg.shape == (7, 3)
        assert np.allclose(agg.sum(axis=1), 1.0)
        # aggregating is a sum over each block, not a reweighting
        for s, block in enumerate(alphabet.blocks):
            assert np.allclose(agg[:, s], joint[:, list(block)].sum(axis=1))


def test_the_scheme_defaults_to_the_published_width():
    # Existing callers construct BandScheme positionally and must be unaffected.
    scheme = rb.BandScheme((3.0,), 1.15, 0.90, "outer")
    assert scheme.main_width == rb.MAIN_WIDTH
    assert scheme.n_main_bands == 30


def test_the_width_appears_in_the_model_name():
    assert nvb.band_width_model(0.05).endswith("width 0.05)")
    assert nvb.band_width_model(0.1).endswith("width 0.10)")
    names = {nvb.band_width_model(w) for w in WIDTHS}
    assert len(names) == len(WIDTHS)          # the arms are distinguishable


def test_the_tail_is_untouched_by_the_main_width():
    # The experiment varies the main range only; the tail rule is the published
    # one, so the same training ratios must give the same tail at every width.
    rng = np.random.default_rng(2)
    ratios = np.concatenate([rng.uniform(0, 3, 400), rng.uniform(3, 6, 300)])
    patients = np.arange(len(ratios)) % 97
    lows, _ = rb.merge_tail(ratios, patients)
    for width in WIDTHS:
        scheme = scheme_for(width, tuple(lows))
        assert np.array_equal(np.asarray(scheme.tail_lows), lows)
        # the tail bands sit above the main range at every width
        tail_start = scheme.n_main_bands
        assert scheme.band_bounds(tail_start)[0] == rb.TAIL_START
