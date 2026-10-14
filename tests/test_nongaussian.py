"""Tests for the non-Gaussian, single-mode random-effect generator.

What must hold for the Stage 6b null to mean what it claims: draws are exactly
standardized (so Cov(b) = Sigma_b), the closed-form mean/variance are right,
the Gaussian case is recovered, the density has a single mode (no groups), and
the skew parameter actually produces skew in the intended direction.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import skew

from ppmi_subtype import nongaussian as NG


@pytest.mark.parametrize("delta,nu", [(0.0, np.inf), (0.9, np.inf), (-0.7, 6.0),
                                      (0.95, 4.0), (0.3, 3.0)])
def test_closed_form_moments_match_the_density(delta, nu):
    """Mean and variance of the standardized density are 0 and 1 by integration."""
    f = lambda x: NG.skewt_density(x, delta, nu)  # noqa: E731
    mass = integrate.quad(f, -np.inf, np.inf, limit=400)[0]
    mean = integrate.quad(lambda x: x * f(x), -np.inf, np.inf, limit=400)[0]
    var = integrate.quad(lambda x: x * x * f(x), -np.inf, np.inf, limit=400)[0]
    assert mass == pytest.approx(1.0, abs=1e-6)
    assert mean == pytest.approx(0.0, abs=1e-5)
    assert var == pytest.approx(1.0, rel=1e-3 if nu > 3 else 2e-2)


@pytest.mark.parametrize("delta,nu", [(0.0, np.inf), (0.9, 5.0), (-0.6, 8.0)])
def test_draws_are_standardized(delta, nu):
    x = NG.standardized_skewt(np.random.default_rng(1), delta, nu, 400_000)
    assert x.mean() == pytest.approx(0.0, abs=0.01)
    assert x.var() == pytest.approx(1.0, rel=0.03)


def test_skew_direction_follows_delta():
    rng = np.random.default_rng(2)
    assert skew(NG.standardized_skewt(rng, 0.95, 8.0, 200_000)) > 0.5
    assert skew(NG.standardized_skewt(rng, -0.95, 8.0, 200_000)) < -0.5
    assert abs(skew(NG.standardized_skewt(rng, 0.0, np.inf, 200_000))) < 0.03


@pytest.mark.parametrize("delta,nu", [(0.95, 3.0), (-0.9, 4.0), (0.5, 10.0), (0.99, np.inf)])
def test_density_has_a_single_mode(delta, nu):
    """No groups: the standardized density increases then decreases."""
    x = np.linspace(-8, 25, 20_001)
    d = NG.skewt_density(x, delta, nu)
    sign_changes = np.count_nonzero(np.diff(np.sign(np.diff(d))) != 0)
    assert sign_changes == 1


def test_random_effects_have_exactly_the_target_covariance():
    S = np.array([[0.10, 0.01, 0.002, 0.001], [0.01, 0.08, -0.003, 0.004],
                  [0.002, -0.003, 0.0022, -0.0006], [0.001, 0.004, -0.0006, 0.0056]])
    b = NG.random_effects(np.random.default_rng(3), S, [0.3, 0.9, 0.6, 0.95], 5.0, 400_000)
    assert np.allclose(np.cov(b.T), S, rtol=0.05, atol=2e-4)
    assert np.allclose(b.mean(axis=0), 0.0, atol=3e-3)


def test_gaussian_limit_matches_multivariate_normal():
    S = np.array([[1.0, 0.3], [0.3, 0.5]])
    b = NG.random_effects(np.random.default_rng(4), S, [0.0, 0.0], np.inf, 200_000)
    assert abs(skew(b[:, 0])) < 0.03 and abs(skew(b[:, 1])) < 0.03
    assert np.allclose(np.cov(b.T), S, rtol=0.03)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        NG.skewt_mean_var(1.0, 5.0)
    with pytest.raises(ValueError):
        NG.skewt_mean_var(0.5, 2.0)
    with pytest.raises(ValueError):
        NG.random_effects(np.random.default_rng(0), np.eye(2), [0.1], 5.0, 10)


def test_per_component_tail_parameters():
    S = np.diag([1.0, 1.0])
    rng = np.random.default_rng(5)
    b = NG.random_effects(rng, S, [0.0, 0.0], [np.inf, 3.0], 300_000)
    from scipy.stats import kurtosis
    assert abs(kurtosis(b[:, 0])) < 0.1          # Gaussian component
    assert kurtosis(b[:, 1]) > 3.0               # heavy-tailed component
    assert np.allclose(b.var(axis=0), 1.0, rtol=0.05)
