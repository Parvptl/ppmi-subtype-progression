"""Tests for the held-out scoring module.

The ones that carry weight: the closed-form CRPS against numerical integration
of its definition, the multinomial-weight bootstrap against the naive
row-resampling estimator it replaces, and the vectorised subtype rule against
the project's own `subtype.classify`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy.stats import norm

from ppmi_subtype import heldout as H
from ppmi_subtype import subtype as st
from ppmi_subtype.config import load_config


# ---------------------------------------------------------------------------
#  Continuous scores
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("y,mu,sd", [(0.3, 0.5, 0.2), (2.0, 0.1, 0.7), (-1.0, 0.0, 1.0)])
def test_crps_matches_its_integral_definition(y, mu, sd):
    """CRPS(F, y) = integral (F(x) - 1{x >= y})^2 dx."""
    f = lambda x: (norm.cdf(x, mu, sd) - (x >= y)) ** 2  # noqa: E731
    lo, hi = mu - 12 * sd, mu + 12 * sd
    num = (integrate.quad(f, lo, y, limit=200)[0] + integrate.quad(f, y, hi, limit=200)[0])
    assert H.gaussian_crps(y, mu, sd) == pytest.approx(num, rel=1e-6, abs=1e-9)


def test_crps_reduces_to_absolute_error_as_sd_vanishes():
    assert H.gaussian_crps(1.3, 0.4, 1e-9) == pytest.approx(0.9, abs=1e-6)


def test_log_score_matches_formula():
    y, mu, sd = 0.7, 0.2, 0.4
    expected = -0.5 * np.log(2 * np.pi * sd ** 2) - (y - mu) ** 2 / (2 * sd ** 2)
    assert H.gaussian_log_score(y, mu, sd) == pytest.approx(expected)


def test_coverage_is_nominal_for_a_correct_forecast():
    rng = np.random.default_rng(0)
    mu, sd = rng.normal(size=200_000), rng.uniform(0.2, 2.0, 200_000)
    y = mu + sd * rng.standard_normal(200_000)
    for lv in H.LEVELS:
        assert H.covered(y, mu, sd, lv).mean() == pytest.approx(lv, abs=0.005)
    u = H.pit(y, mu, sd)
    assert np.histogram(u, bins=10, range=(0, 1))[0].min() > 19_000


def test_pointwise_rejects_non_positive_sd():
    f = pd.DataFrame({"obs_x": [1.0], "pred_x": [1.0], "sd_x": [0.0]})
    with pytest.raises(ValueError):
        H.pointwise_continuous(f, "x")


# ---------------------------------------------------------------------------
#  Bootstrap
# ---------------------------------------------------------------------------
def test_weighted_bootstrap_equals_row_resampling():
    """Same patient draws, two computations: they must agree exactly."""
    rng = np.random.default_rng(1)
    patients = np.repeat(np.arange(30), rng.integers(1, 6, 30))
    values = rng.normal(size=len(patients))
    boot = H.SharedBootstrap.create(patients, n_boot=50, seed=3)

    S, n = boot._sums(values, patients)
    fast = (boot.weights @ S) / (boot.weights @ n)
    naive = []
    for w in boot.weights:
        picked = np.concatenate([np.repeat(values[patients == p][None, :], int(c), axis=0).ravel()
                                 for p, c in zip(boot.patients, w) if c > 0])
        naive.append(picked.mean())
    assert np.allclose(fast, naive)


def test_shared_draws_make_a_paired_difference_of_identical_models_zero():
    rng = np.random.default_rng(2)
    patients = np.repeat(np.arange(40), 3)
    v = rng.normal(size=len(patients))
    boot = H.SharedBootstrap.create(patients, n_boot=200)
    est, lo, hi = boot.mean_ci(v - v, patients)
    assert est == lo == hi == 0.0


def test_unknown_patient_is_rejected():
    boot = H.SharedBootstrap.create([1, 2, 3], n_boot=5)
    with pytest.raises(ValueError):
        boot.mean_ci([1.0], [9])


def test_ratio_ci_recovers_a_recall():
    patients = np.array([1, 1, 2, 2, 3, 3])
    hit = np.array([1, 0, 1, 1, 0, 0], float)
    support = np.array([1, 1, 1, 1, 1, 0], float)
    boot = H.SharedBootstrap.create(patients, n_boot=100)
    est, lo, hi = boot.ratio_ci(hit, support, patients)
    assert est == pytest.approx(3 / 5)
    assert lo <= est <= hi


# ---------------------------------------------------------------------------
#  Subtype rule and probabilities
# ---------------------------------------------------------------------------
def test_vectorised_rule_matches_the_project_rule():
    cfg = load_config()
    grid = np.array([0.0, 0.05, 0.25, 0.45, 0.5, 0.9, 1.0, 1.15, 1.3, 2.0])
    t, p = np.meshgrid(grid, grid)
    t, p = t.ravel(), p.ravel()
    ref = st.classify(pd.Series(t), pd.Series(p), cfg.subtype)
    code_of = {cfg.subtype.labels["TD"]: 0, cfg.subtype.labels["Indeterminate"]: 1,
               cfg.subtype.labels["PIGD"]: 2}
    ref = ref.map(code_of).to_numpy()
    got = H.classify_draws(t, p, cfg.subtype.td_threshold, cfg.subtype.pigd_threshold)
    assert np.array_equal(ref, got)


def test_subtype_probabilities_sum_to_one_and_degenerate_to_the_point_rule():
    probs, clipped = H.subtype_probabilities(
        [1.0, 0.2, 0.5], [0.2, 1.0, 0.5], [1e-6] * 3, [1e-6] * 3, [0, 0, 0],
        1.15, 0.90, n_draws=500)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.array_equal(probs.argmax(axis=1), [0, 2, 1])
    assert np.all(clipped == 0)


def test_categorical_summary_counts_and_balanced_accuracy():
    actual = np.array([0, 0, 1, 2, 2, 2])
    pred = np.array([0, 0, 0, 2, 2, 1])
    probs = np.eye(3)[pred] * 0.8 + 0.2 / 3
    patients = np.array([1, 1, 2, 2, 3, 3])
    boot = H.SharedBootstrap.create(patients, n_boot=100)
    out = H.categorical_summary(probs, actual, pred, patients, boot)
    assert out["accuracy"] == pytest.approx(4 / 6)
    assert out["recall_TD"] == 1.0 and out["recall_Indeterminate"] == 0.0
    assert out["recall_PIGD"] == pytest.approx(2 / 3)
    assert out["balanced_accuracy"] == pytest.approx((1 + 0 + 2 / 3) / 3)
    assert out["n_predicted_Indeterminate"] == 1
