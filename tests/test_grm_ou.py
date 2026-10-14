"""Tests for the GRM with an OU latent deviation (Amendment 3).

Weight-bearing: the exact Laplace gradient against finite differences (all 7
parameters, mixed visit counts), Laplace against adaptive quadrature over the
full latent vector, the kernel and its derivatives, the simulated latent
covariance, and recovery of a known persistence (σ_u, ℓ).
"""

from __future__ import annotations

import numpy as np
import pytest

from ppmi_subtype import grm_ou as O

A = np.array([1.8, 1.2, 2.5, 1.5, 2.0])
TAU = np.array([[-0.5, 0.8, 1.6, 2.6], [0.1, 1.2, 2.0, 2.8], [-1.2, 0.6, 1.5, 2.2],
                [1.0, 1.7, 2.3, 3.0], [0.7, 1.4, 2.1, 2.9]])
BETA = np.array([-0.4, 0.15])
D = np.array([[0.5, 0.01], [0.01, 0.02]])
PSI = O.to_psi(BETA, D, 0.5, 1.5)


def data(rng, n_pat=30, psi=PSI, lo=1, hi=7):
    beta, Dm, _, s, ell = O.unpack(psi)
    times = {i: np.sort(np.r_[0, rng.uniform(0.3, 8, rng.integers(lo, hi))]) for i in range(n_pat)}
    return times, O.simulate(rng, times, beta, Dm, s, ell, A, TAU)


def test_kernel_derivatives_match_finite_differences():
    T = np.array([[0.0, 0.7, 2.1, 5.0]])
    h = 1e-6
    _, _, dD, s, ell = O.unpack(PSI)
    _, dK = O.kernel(T, *O.unpack(PSI)[1:3], s, ell)
    for k in range(5):
        e = np.zeros(7)
        e[2 + k] = h
        Kp = O.kernel(T, *O.unpack(PSI + e)[1:3], *O.unpack(PSI + e)[3:], grads=False)[0]
        Km = O.kernel(T, *O.unpack(PSI - e)[1:3], *O.unpack(PSI - e)[3:], grads=False)[0]
        assert np.allclose(dK[k], (Kp - Km) / (2 * h), atol=1e-6)


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    g = O.group(*data(rng))
    psi = PSI + np.array([0.05, -0.02, 0.1, 0.03, -0.1, 0.1, -0.2])
    _, grad = O.negloglik(psi, g, A, TAU)
    h = 1e-6
    fd = np.array([(O.negloglik(psi + h * e, g, A, TAU)[0] - O.negloglik(psi - h * e, g, A, TAU)[0])
                   / (2 * h) for e in np.eye(7)])
    assert np.allclose(grad, fd, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n_visits", [1, 2, 3])
def test_laplace_close_to_adaptive_quadrature(n_visits):
    rng = np.random.default_rng(n_visits)
    times, items = data(rng, n_pat=4, lo=n_visits - 1, hi=n_visits)
    for p in times:
        assert len(times[p]) == n_visits
        lap = O.laplace_one(times[p], items[p], PSI, A, TAU)
        agq = O.agq_one(times[p], items[p], PSI, A, TAU, nodes=9 if n_visits < 3 else 7)
        # Laplace is least accurate when every item is 0 at every visit: the
        # posterior is piled against the floor and skewed (0.082 observed for
        # one such patient, against <= 0.03 otherwise). That limit is reported
        # with the real-data diagnostic rather than hidden by a loose bound.
        floor = bool(np.all(items[p] == 0))
        assert lap == pytest.approx(agq, abs=0.10 if floor else 0.05)


def test_simulated_latent_covariance():
    rng = np.random.default_rng(5)
    t = np.array([0.0, 1.0, 3.0])
    beta, Dm, dD, s, ell = O.unpack(PSI)
    K = O.kernel(t[None], Dm, dD, s, ell, grads=False)[0][0]
    L = np.linalg.cholesky(K)
    th = (L @ rng.standard_normal((3, 200_000))).T
    assert np.allclose(np.cov(th.T), K, atol=0.02)


def test_recovers_persistence_parameters():
    """Median over five seeds: σ_u is well recovered, ℓ is noisy (it trades off
    with the intercept variance; across 12 seeds it ranged 1.35-3.0 around 2.0),
    so the test checks the median rather than a single draw."""
    true = O.to_psi(BETA, D, 0.7, 2.0)
    est = []
    for seed in range(7, 12):
        rng = np.random.default_rng(seed)
        g = O.group(*data(rng, n_pat=700, psi=true, lo=3, hi=10))
        f = O.fit(g, A, TAU, psi0=O.to_psi(BETA, D, 0.3, 1.0))
        assert f.converged
        assert f.beta == pytest.approx(BETA, abs=0.12)
        est.append((f.sigma_u, f.ell))
    est = np.array(est)
    assert np.median(est[:, 0]) == pytest.approx(0.7, rel=0.10)
    assert np.median(est[:, 1]) == pytest.approx(2.0, rel=0.25)


def test_grouping_keeps_every_patient_once():
    rng = np.random.default_rng(8)
    times, items = data(rng, n_pat=25)
    g = O.group(times, items)
    seen = sorted(p for ps, _, _ in g.groups.values() for p in ps)
    assert seen == sorted(times)
    for n, (ps, T, Y) in g.groups.items():
        assert T.shape == (len(ps), n) and Y.shape == (len(ps), n, 5)
