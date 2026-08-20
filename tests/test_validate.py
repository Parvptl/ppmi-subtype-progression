"""Tests for validation against PPMI's own derivations.

The archived reference uses a different code for two of the three classes than
this project does. Comparing the raw integers scores roughly 66% agreement and
looks like a broken classifier, when in fact the labels agree; this remap is
the thing that distinguishes the two situations.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ppmi_subtype.config import load_config
from ppmi_subtype.validate import collapse_to_binary, remap_archived_labels


@pytest.fixture
def cfg():
    return load_config()


class TestArchivedRemap:
    def test_codes_are_swapped_not_identity(self, cfg):
        """2 = PIGD and 3 = Indeterminate in the archived dictionary."""
        out = remap_archived_labels(pd.Series([1.0, 2.0, 3.0]), cfg)
        labels = cfg.subtype.labels
        assert out.iloc[0] == labels["TD"]
        assert out.iloc[1] == labels["PIGD"]
        assert out.iloc[2] == labels["Indeterminate"]

    def test_remap_is_not_the_identity(self, cfg):
        """Guards the exact bug: assuming the codings already match."""
        raw = pd.Series([2.0, 3.0])
        assert not remap_archived_labels(raw, cfg).equals(raw)

    def test_unknown_codes_become_missing(self, cfg):
        out = remap_archived_labels(pd.Series([9.0]), cfg)
        assert out.isna().all()

    def test_config_codings_genuinely_differ(self, cfg):
        ours = cfg.subtype.labels
        theirs = cfg.raw["subtype"]["archived_reference_labels"]
        assert ours["TD"] == theirs["TD"]
        assert ours["PIGD"] != theirs["PIGD"]
        assert ours["Indeterminate"] != theirs["Indeterminate"]


class TestBinaryCollapse:
    def test_only_td_maps_to_one(self, cfg):
        labels = cfg.subtype.labels
        out = collapse_to_binary(
            pd.Series([labels["TD"], labels["Indeterminate"], labels["PIGD"]]), cfg
        )
        assert list(out) == [1.0, 2.0, 2.0]

    def test_missing_stays_missing(self, cfg):
        out = collapse_to_binary(pd.Series([float("nan")]), cfg)
        assert out.isna().all()
