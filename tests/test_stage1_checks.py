"""Tests for the Stage 1 descriptive-check calculations (stage1_checks)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sstats

from ppmi_subtype import ppc_diagnostic as D
from ppmi_subtype import stage1_checks as S
from ppmi_subtype.config import load_config
from ppmi_subtype.raw_scores import label_grid

THETA = {"beta0": 1.6, "beta1": -0.08, "tau0": 0.9, "tau1": 0.07, "nu_s": 2.2,
         "sigma": 0.55, "nu_e": 2.1}


@pytest.fixture(scope="module")
def grid():
    return label_grid(load_config().subtype)


def test_lattice_rows_match_rule_arithmetic(grid):
    r1 = S.lattice_row(grid, 1)
    assert r1["ind_tremor_sums"] == [2]
    assert r1["ind_ratios"] == pytest.approx([10 / 11])
    assert r1["lo"] == pytest.approx((5 / 11 + 10 / 11) / 2)
    assert r1["hi"] == pytest.approx((10 / 11 + 15 / 11) / 2)
    assert r1["inflation"] == pytest.approx((5 / 11) / 0.25)
    r2 = S.lattice_row(grid, 2)
    assert r2["ind_tremor_sums"] == [4, 5]
    assert r2["width"] == pytest.approx(2 * 5 / 22)
    # every Indeterminate lattice point lies inside the nominal band
    for p in range(1, 21):
        r = S.lattice_row(grid, p)
        assert all(0.90 < x < 1.15 for x in r["ind_ratios"])
        # the catchment brackets its lattice points but need not contain the band
        assert r["lo"] < min(r["ind_ratios"]) and r["hi"] > max(r["ind_ratios"])
    assert r1["hi"] < 1.15


def test_direct_change_outcomes_reproduce_S3():
    rng = np.random.default_rng(0)
    pid = np.repeat(np.arange(300), rng.integers(1, 7, 300))
    L = rng.choice([0, 1, 2], size=pid.size, p=[0.5, 0.15, 0.35]).astype(np.int8)
    j, d, o = S.direct_change_outcomes(L, pid)
    stats = D.label_statistics(L[None, :], pid)[0]
    assert (o == 0).sum() / (o < 3).sum() == pytest.approx(stats[2])
    assert np.all(pid[j] == pid[j + 1])
    assert set(np.unique(d)) <= {0, 1}
    # a hand case: TD PIGD TD | PIGD TD | TD PIGD Ind
    L2 = np.array([0, 2, 0, 2, 0, 0, 2, 1])
    p2 = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    j, d, o = S.direct_change_outcomes(L2, p2)
    assert j.tolist() == [0, 1, 3, 5]
    assert d.tolist() == [0, 1, 1, 0]
    assert o.tolist() == [0, 3, 3, 2]


def test_visit_probs_gaussian_limit():
    th = {**THETA, "tau1": 1e-9, "nu_s": 1e6, "nu_e": 1e6}
    t = np.array([0.0, 1.0, 3.0])
    pTD, pI, pP, W = S.visit_label_probs(t, th, n_b0=40)
    sd = np.hypot(th["tau0"], th["sigma"])
    m = th["beta0"] + th["beta1"] * t
    want_I = sstats.norm.cdf((1.15 - m) / sd) - sstats.norm.cdf((0.90 - m) / sd)
    assert W @ pI == pytest.approx(want_I, abs=1e-6)
    assert W.sum() == pytest.approx(1.0)
    assert W @ (pTD + pI + pP) == pytest.approx(np.ones(3))
    cdf = S.expected_cdf(t, th, [0.90, 1.15], n_b0=40)
    assert cdf == pytest.approx([(W @ pP).sum(), (W @ (pP + pI)).sum()], abs=1e-9)


def _monte_carlo(t, pid, theta, n=400_000, seed=1):
    rng = np.random.default_rng(seed)
    M = pid.max() + 1
    b0 = rng.normal(0, theta["tau0"], (n, M))
    b1 = theta["tau1"] * rng.standard_t(theta["nu_s"], (n, M))
    e = rng.standard_t(theta["nu_e"], (n, len(t)))
    y = theta["beta0"] + b0[:, pid] + (theta["beta1"] + b1[:, pid]) * t + theta["sigma"] * e
    return D.labels_from_ratio(y)


def test_expectations_match_monte_carlo():
    t = np.array([0.0, 1.0, 2.0, 3.5, 0.0, 2.0])
    pid = np.array([0, 0, 0, 0, 1, 1])
    groups = np.array([0, 1, 0, 1, 1, 1])
    ex = S.expected_visit_quantities(t, pid, THETA, groups={"g": groups}, n_b0=24, n_b1=64)
    L = _monte_carlo(t, pid, THETA)
    assert ex["ind"] == pytest.approx((L == 1).mean(0), abs=4e-3)
    a, b, c = L[:, :-2], L[:, 1:-1], L[:, 2:]
    mc_rev = ((a == 0) & (b == 2) & (c == 0)).mean(0)
    assert ex["trip"][0]["revert"][:2] == pytest.approx(mc_rev[:2], abs=2e-3)
    mc_pair = ((L[:, :-1] == 2) & (L[:, 1:] == 0)).mean(0)
    assert ex["pair"][1][:3] == pytest.approx(mc_pair[:3], abs=3e-3)
    assert ex["pair"][1][3] == 0.0                     # crosses a patient boundary
    counts = np.stack([((L == 1) & (groups == g)).sum(1) for g in (0, 1)], 1)
    assert ex["ind_var"]["g"] == pytest.approx(counts.var(0), rel=0.03)


def test_trip_and_unobserved_partition_pair():
    t = np.array([0.0, 1.0, 2.0, 0.0, 1.0])
    pid = np.array([0, 0, 0, 1, 1])
    ex = S.expected_visit_quantities(t, pid, THETA)
    for d in (0, 1):
        tot = sum(ex["trip"][d].values()) + ex["unobs"][d]
        # where j+2 exists the three outcomes exhaust the pair; else it is unobserved
        assert tot == pytest.approx(ex["pair"][d], abs=1e-12)


def test_lattice_bounds_change_probabilities(grid):
    t = np.zeros(2)
    lo, hi = S.catchment_bounds(grid, [1, 20])
    assert lo[0] == pytest.approx(S.lattice_row(grid, 1)["lo"])
    _, pI_band, _, W = S.visit_label_probs(t, THETA)
    _, pI_lat, _, _ = S.visit_label_probs(t, THETA, lo=lo, hi=hi)
    assert (W @ pI_lat)[0] > (W @ pI_band)[0]          # wider catchment at PIGD sum 1


def test_residual_proxy_helpers():
    pid = np.array([0, 0, 0, 1, 1])
    t = np.array([0.0, 1.0, 2.0, 0.0, 3.0])
    y = np.array([1.0, 2.0, 4.0, 3.0, 1.0])
    r = S.centred_detrended(y, t, pid, beta1=-0.5)
    assert np.bincount(pid, weights=r) == pytest.approx([0.0, 0.0])
    assert r[0] == pytest.approx(1.0 - (np.mean([1.0, 2.5, 5.0])))
    x = np.random.default_rng(3).standard_t(2.1, 200_001)
    assert abs(S.bowley(x)) < 0.02 and abs(S.tail_asymmetry(x)) < 0.02
    assert S.bowley(np.random.default_rng(3).exponential(size=10_001)) > 0.1


def test_per_patient_and_bootstrap_ratio():
    pid = np.array([0, 0, 1, 2, 2, 2])
    cols = np.column_stack([[1, 0, 1, 0, 0, 1], np.ones(6)])
    pp = S.per_patient(pid, cols, 3)
    assert pp.tolist() == [[1, 2], [1, 1], [1, 3]]
    lo, hi = S.bootstrap_ratio(np.array([[1, 4], [2, 4], [3, 4]], float), 0, 1, level=0.5)
    assert lo < 0.5 < hi


def test_delatticed_flat_lattice(grid):
    # a flat count of 10 per tremor sum at PIGD sums 1-3
    psum = np.repeat([1, 2, 3], 45)
    tsum = np.tile(np.arange(45), 3)
    rows = S.delatticed_ind(psum, tsum, grid, max_psum=3)
    for r in rows:
        step = 5 / (11 * r["pigd_sum"])
        assert r["ind"] == r["n_ind_points"]
        assert r["delatticed"] == pytest.approx(0.25 / step)       # nominal points in the band
        assert r["cell_mean"] == r["neighbour_mean"] == 1
