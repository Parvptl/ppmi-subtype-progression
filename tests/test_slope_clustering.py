"""Tests for the Bayesian slope-clustering building blocks.

Weight-bearing: the slope is beta1 + b1 (not b1 alone), patient order survives
non-contiguous IDs and missing ratios, every row is one joint draw, the
selection rule enforces the component-size floor, co-clustering is
label-invariant, and the diagnostics behave correctly on known cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype import hbayes
from ppmi_subtype import slope_clustering as SC


def panel_with_planted_slopes(seed=0, n=150):
    """Patients with non-contiguous IDs, a known slope each, and some undefined ratios."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.arange(3000, 90000), n, replace=False)
    true = {}
    rows = []
    for i, p in enumerate(ids):
        slope = -0.4 if i % 2 else 0.4                # large, well separated slopes
        true[p] = slope
        for k, t in enumerate(np.arange(0, 6.0, 1.0)):
            r = 2.0 + slope * t + rng.normal(0, 0.05)
            if k == 2 and i % 5 == 0:
                r = np.nan                             # undefined ratio (PIGD = 0)
            if k == 3 and i % 7 == 0:
                r = np.inf
            rows.append({"PATNO": p, "years_from_baseline": t, "ratio": r})
    # one patient with a single finite visit: must be excluded
    rows += [{"PATNO": 1, "years_from_baseline": 0.0, "ratio": 1.0},
             {"PATNO": 1, "years_from_baseline": 1.0, "ratio": np.nan}]
    return pd.DataFrame(rows).sample(frac=1, random_state=1), true


def test_total_slope_is_population_plus_random_effect_and_aligned():
    panel, true = panel_with_planted_slopes()
    data = hbayes.build_growth_data(panel)
    assert 1 not in set(data.patients)                         # single finite visit excluded
    assert data.n_obs == int(np.isfinite(panel.ratio).sum()) - 1
    post = hbayes.fit_growth_model(data, n_draws=400, burn_in=200, seed=3,
                                   keep_random_effects=True)
    S = SC.total_slope_draws(post)
    assert S.shape == (post.n_draws, data.n_patients)
    assert np.allclose(S, post.beta[:, 1][:, None] + post.random_effects[:, :, 1])
    summ = SC.patient_slope_summary(post, data)
    est = dict(zip(summ.PATNO, summ.slope_mean))
    for p, s in true.items():
        assert est[p] == pytest.approx(s, abs=0.06)                  # right patient, right slope
    dev = post.random_effects[:, :, 1].mean(axis=0)
    assert not np.allclose(dev, summ.slope_mean)                     # not b1 alone
    assert (summ.n_ratio_obs >= 2).all()


def test_summary_counts_and_follow_up():
    panel, _ = panel_with_planted_slopes(n=20)
    data = hbayes.build_growth_data(panel)
    post = hbayes.fit_growth_model(data, n_draws=100, burn_in=50, seed=1,
                                   keep_random_effects=True)
    s = SC.patient_slope_summary(post, data)
    fin = panel[np.isfinite(panel.ratio)]
    ref = fin.groupby("PATNO").years_from_baseline.agg(["size", "min", "max"])
    ref = ref[ref["size"] >= 2]
    s = s.set_index("PATNO").loc[ref.index]
    assert (s.n_ratio_obs.to_numpy() == ref["size"].to_numpy()).all()
    assert np.allclose(s.followup_years, ref["max"] - ref["min"])


def test_requires_retained_random_effects():
    panel, _ = panel_with_planted_slopes(n=10)
    post = hbayes.fit_growth_model(hbayes.build_growth_data(panel), n_draws=20, burn_in=5)
    with pytest.raises(ValueError):
        SC.total_slope_draws(post)


def test_selection_finds_planted_mixture_and_enforces_size_floor():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(-0.3, 0.03, 400), rng.normal(0.1, 0.03, 400)]
    sel = SC.select_K(x, seed=1, n_init=5)
    assert sel.selected_K == 2 and sel.delta_bic > 100
    # a component of 5 points is below the floor of max(10, 2%) and is not eligible
    y = np.r_[rng.normal(0, 0.05, 800), rng.normal(5, 0.01, 5)]
    sel2 = SC.select_K(y, seed=1, n_init=5)
    assert sel2.fits[2].sizes.min() == 5
    assert not sel2.eligible[2] and sel2.unrestricted_K >= 2
    assert sel2.selected_K != 2


def test_single_gaussian_usually_selects_one_component():
    rng = np.random.default_rng(2)
    picks = [SC.select_K(rng.normal(0, 1, 800), seed=s, n_init=3).selected_K for s in range(8)]
    assert sum(k == 1 for k in picks) >= 6


def test_components_are_sorted_by_mean():
    rng = np.random.default_rng(3)
    f = SC.fit_mixture(np.r_[rng.normal(2, .1, 300), rng.normal(-2, .1, 300)], 2, seed=0)
    assert f.means[0] < f.means[1]


def test_coclustering_is_label_invariant():
    C1 = np.zeros((4, 4))
    C2 = np.zeros((4, 4))
    SC.coclustering_update(C1, np.array([0, 0, 1, 1]))
    SC.coclustering_update(C2, np.array([1, 1, 0, 0]))          # same partition, swapped labels
    assert np.array_equal(C1, C2)
    assert C1[0, 1] == 1 and C1[0, 2] == 0 and np.all(np.diag(C1) == 1)


def test_silverman_distinguishes_one_and_two_modes():
    rng = np.random.default_rng(4)
    uni = rng.normal(0, 1, 400)
    bi = np.r_[rng.normal(-3, 0.6, 200), rng.normal(3, 0.6, 200)]
    assert SC.n_modes(uni, 0.6) == 1
    assert SC.silverman_test(uni, n_boot=100, seed=1)["p_value"] > 0.05
    assert SC.silverman_test(bi, n_boot=100, seed=1)["p_value"] < 0.05


def test_rhat_and_ess_on_known_chains():
    rng = np.random.default_rng(5)
    iid = rng.standard_normal((4, 2000))
    assert SC.split_rhat(iid) == pytest.approx(1.0, abs=0.01)
    assert SC.ess(iid) > 6000
    shifted = iid + np.arange(4)[:, None]
    assert SC.split_rhat(shifted) > 1.5
    ar = np.zeros((4, 2000))
    for t in range(1, 2000):
        ar[:, t] = 0.9 * ar[:, t - 1] + rng.standard_normal(4)
    assert SC.ess(ar) < 0.15 * 8000                              # strong autocorrelation
