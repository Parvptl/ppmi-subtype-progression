"""Tests for the D1 replicate-label diagnostic (recorder, exact gate, splits)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype import hbayes
from ppmi_subtype import ppc_diagnostic as D
from ppmi_subtype import replicate_label_diagnostic as RL
from ppmi_subtype import stage1_checks as S

TINY_FIT = {"n_draws": 30, "burn_in": 20, "thin": 1}


def _toy_data(M=40, seed=0):
    rng = np.random.default_rng(seed)
    n = rng.integers(2, 6, M)
    pid = np.repeat(np.arange(M), n)
    t = np.concatenate([np.sort(rng.uniform(0, 6, k)) for k in n])
    y = np.abs(1.6 + rng.normal(0, 0.8, M)[pid] - 0.08 * t + 0.5 * rng.standard_t(3, pid.size))
    return hbayes.GrowthData(y=y, t=t, patient_index=pid, patients=np.arange(100, 100 + M),
                             n_patients=M, n_obs=pid.size)


def test_recorder_restores_and_records_only_replicates():
    orig = D.labels_from_ratio
    with RL.LabelRecorder() as rec:
        D.labels_from_ratio(np.array([1.0, 2.0]))
        out = D.labels_from_ratio(np.array([[0.5, 1.0, 2.0]]))
    assert D.labels_from_ratio is orig
    assert len(rec.records) == 1 and np.array_equal(rec.single(), out)
    rec.records.clear()
    with pytest.raises(RuntimeError):
        rec.single()


def test_recorder_inside_run_cohort_is_exact_and_deterministic():
    data = _toy_data()
    folds = D.make_folds(data, 5, 42)
    plain = D.run_cohort(data, data.y, folds, 20, TINY_FIT, [1, 2, 3])
    with RL.LabelRecorder() as rec:
        recd = D.run_cohort(data, data.y, folds, 20, TINY_FIT, [1, 2, 3])
    # the recorder changes nothing: identical summaries, exactly
    g = RL.gate(recd, plain, RL.GATED_KEYS)
    assert g["passed"], g["mismatches"]
    L = rec.single()
    assert L.shape == (20, data.n_obs) and L.dtype == np.int8
    S_rep = D.label_statistics(L, data.patient_index)
    names = D.STAT_NAMES + D.DESC_NAMES
    assert np.allclose(np.nanmean(S_rep, axis=0), [plain["S_rep_mean"][k] for k in names],
                       equal_nan=True, rtol=0, atol=0)
    # a different seed must fail the gate
    other = D.run_cohort(data, data.y, folds, 20, TINY_FIT, [1, 2, 4])
    assert not RL.gate(other, plain, RL.GATED_KEYS)["passed"]


def test_exact_mismatches():
    a = {"x": 1.0, "y": [1, {"z": float("nan")}], "b": True}
    assert RL.exact_mismatches(a, a) == []
    assert RL.exact_mismatches({**a, "x": 1.0 + 1e-15}, a)
    assert RL.exact_mismatches({"x": 1.0}, a)               # missing keys
    assert RL.exact_mismatches({**a, "y": [1]}, a)          # list length
    g = RL.gate({"T": np.float64(2.5)}, {"T": 2.5}, ("T",))
    assert g["passed"]
    assert not RL.gate({}, {"T": 2.5}, ("T",))["passed"]


def test_splits_match_single_sequence_functions():
    rng = np.random.default_rng(5)
    pid = np.repeat(np.arange(120), rng.integers(1, 7, 120))
    L = rng.choice([0, 1, 2], size=(7, pid.size), p=[0.5, 0.15, 0.35]).astype(np.int8)
    strata = rng.integers(0, 3, pid.size)
    sh = RL.stratum_shares(L, strata, 3)
    for r in range(7):
        for k in range(3):
            assert sh[r, k] == pytest.approx((L[r, strata == k] == 1).mean())
    c = RL.direction_outcome_counts(L, pid)
    rates = RL.reversion_rates(c.sum(1))
    for r in range(7):
        assert rates[r] == pytest.approx(D.label_statistics(L[r][None], pid)[0][2])
        _, d, o = S.direct_change_outcomes(L[r], pid)
        assert c[r].sum() == len(d)


def test_describe_uses_d1_midp():
    rep = np.arange(10, dtype=float)
    d = RL.describe(4.0, rep)
    assert d["midp"] == pytest.approx(D.midp(4.0, rep)[0])
    assert d["status"].startswith("exploratory")
    assert RL.describe(100.0, rep)["z"] < 0                 # observed above every replicate
