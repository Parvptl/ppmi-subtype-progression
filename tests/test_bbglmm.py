"""Tests for the beta-binomial mixed model (measurement-model candidate).

Carrying the weight: the analytic eta-derivatives against finite differences,
the Laplace marginal likelihood against brute-force 2-d quadrature for single
patients, and recovery of known parameters from simulated data.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy.special import expit, logit

from ppmi_subtype import bbglmm as BB


@pytest.mark.parametrize("y,n,mu,phi", [(0, 20, 0.05, 8.0), (3, 20, 0.2, 15.0),
                                        (20, 20, 0.9, 3.0), (7, 44, 0.1, 50.0)])
def test_eta_derivatives_match_finite_differences(y, n, mu, phi):
    eta, h = logit(mu), 1e-5
    f = lambda e: BB.bb_logpmf(y, n, expit(e), phi)  # noqa: E731
    d1, d2 = BB.bb_eta_derivs(y, n, mu, phi)
    assert d1 == pytest.approx((f(eta + h) - f(eta - h)) / (2 * h), rel=1e-5, abs=1e-8)
    assert d2 == pytest.approx((f(eta + h) - 2 * f(eta) + f(eta - h)) / h ** 2,
                               rel=1e-3, abs=1e-5)


def test_pmf_sums_to_one_and_has_beta_binomial_variance():
    n, mu, phi = 20, 0.15, 6.0
    y = np.arange(n + 1)
    p = np.exp(BB.bb_logpmf(y, n, mu, phi))
    assert p.sum() == pytest.approx(1.0)
    var = (p * (y - (p * y).sum()) ** 2).sum()
    assert var == pytest.approx(n * mu * (1 - mu) * (n + phi) / (1 + phi), rel=1e-10)


def test_laplace_close_to_quadrature_for_single_patients():
    rng = np.random.default_rng(0)
    beta, D, phi, n = np.array([-1.8, 0.12]), np.array([[0.5, 0.02], [0.02, 0.01]]), 12.0, 20
    for k in range(4):
        t = np.sort(rng.uniform(0, 8, 6))
        y = BB.simulate(rng, BB.BBFit(beta, D, phi, 0, True, "", n, None), {0: t})[0]
        packed = BB.pack({0: t}, {0: y}, n)
        th = np.r_[beta, np.log(np.sqrt(D[0, 0])), D[1, 0] / np.sqrt(D[0, 0]),
                   np.log(np.sqrt(D[1, 1] - D[1, 0] ** 2 / D[0, 0])), np.log(phi)]
        lap = -BB.neg_marginal_loglik(th, packed)
        Dinv, det = np.linalg.inv(D), np.linalg.det(D)

        def integrand(b1, b0):
            mu = expit(beta[0] + beta[1] * t + b0 + b1 * t)
            ll = BB.bb_logpmf(y, n, mu, phi).sum()
            q = np.array([b0, b1])
            return np.exp(ll - 0.5 * q @ Dinv @ q) / (2 * np.pi * np.sqrt(det))
        exact = np.log(integrate.dblquad(integrand, -4, 4, -0.6, 0.6, epsabs=1e-12)[0])
        assert lap == pytest.approx(exact, abs=0.05)


def test_recovers_known_parameters():
    rng = np.random.default_rng(1)
    true = BB.BBFit(np.array([-1.6, 0.10]), np.array([[0.6, 0.01], [0.01, 0.006]]), 10.0,
                    0.0, True, "", 20, None)
    times = {i: np.sort(np.r_[0, rng.uniform(0.5, 9, rng.integers(2, 9))]) for i in range(600)}
    sums = BB.simulate(rng, true, times)
    f = BB.fit(BB.pack(times, sums, 20))
    assert f.converged
    assert f.beta[0] == pytest.approx(-1.6, abs=0.12)
    assert f.beta[1] == pytest.approx(0.10, abs=0.02)
    assert np.sqrt(f.D[0, 0]) == pytest.approx(np.sqrt(0.6), rel=0.15)
    assert np.sqrt(f.D[1, 1]) == pytest.approx(np.sqrt(0.006), rel=0.3)
    assert f.phi == pytest.approx(10.0, rel=0.35)


def test_pack_rejects_non_integer_or_out_of_range_sums():
    with pytest.raises(ValueError):
        BB.pack({0: np.array([0.0, 1.0])}, {0: np.array([0.5, 2.0])}, 20)
    with pytest.raises(ValueError):
        BB.pack({0: np.array([0.0, 1.0])}, {0: np.array([21.0, 2.0])}, 20)
