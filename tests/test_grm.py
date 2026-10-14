"""Tests for the graded-response PIGD measurement model (Stage 6b, Amendment 2).

The weight-bearing ones: category probabilities are valid and stable at
extreme θ; every analytic derivative (θ-derivatives, stage-1 gradient, exact
Laplace gradient) matches finite differences; the stage-1 marginal likelihood
matches direct integration; Laplace is close to adaptive quadrature; missing
items drop out of the likelihood; both stages recover known parameters.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from ppmi_subtype import grm as G

A = np.array([1.8, 1.2, 2.5, 1.5, 2.0])
TAU = np.array([[-0.5, 0.8, 1.6, 2.6], [0.1, 1.2, 2.0, 2.8], [-1.2, 0.6, 1.5, 2.2],
                [1.0, 1.7, 2.3, 3.0], [0.7, 1.4, 2.1, 2.9]])


def probs(theta):
    return np.exp(np.stack([G.category_logprob(theta, A, TAU, np.full(5, c))
                            for c in range(5)], -1))


@pytest.mark.parametrize("theta", [-40.0, -3.0, 0.0, 1.3, 4.0, 40.0])
def test_category_probabilities_valid_and_stable(theta):
    P = probs(theta)
    assert np.all(np.isfinite(P)) and np.all(P >= 0)
    assert np.allclose(P.sum(-1), 1.0, atol=1e-12)


def test_category_logprob_finite_at_extremes():
    for theta in (-60.0, 60.0):
        lp = G.category_logprob(theta, A, TAU, np.array([2, 4, 0, 3, 1.0]))
        assert np.all(np.isfinite(lp))


def test_pack_unpack_roundtrip_and_ordering():
    phi = G.pack_items(A, TAU)
    a, tau = G.unpack_items(phi, 5)
    assert np.allclose(a, A) and np.allclose(tau, TAU)
    a2, tau2 = G.unpack_items(np.random.default_rng(0).normal(size=25), 5)
    assert np.all(np.diff(tau2, axis=1) > 0) and np.all(a2 > 0)


@pytest.mark.parametrize("theta", [-2.0, 0.3, 1.7, 3.5])
def test_theta_derivatives_match_finite_differences(theta):
    y = np.array([0, 2, 1, 4, 3.0])
    f = lambda th: G.category_logprob(th, A, TAU, y).sum()  # noqa: E731
    l1, l2, l3 = G.theta_derivs(np.array(theta), A, TAU, y)
    h = 1e-4
    assert l1 == pytest.approx((f(theta + h) - f(theta - h)) / (2 * h), rel=1e-6, abs=1e-8)
    assert l2 == pytest.approx((f(theta + h) - 2 * f(theta) + f(theta - h)) / h ** 2,
                               rel=1e-4, abs=1e-6)
    d2 = lambda th: G.theta_derivs(np.array(th), A, TAU, y)[1]  # noqa: E731
    assert l3 == pytest.approx((d2(theta + h) - d2(theta - h)) / (2 * h), rel=1e-4, abs=1e-6)


def test_log_likelihood_is_concave_in_theta():
    y = np.array([1, 0, 3, 0, 2.0])
    for theta in np.linspace(-6, 8, 57):
        assert G.theta_derivs(np.array(theta), A, TAU, y)[1] < 0


def test_missing_items_drop_out():
    y_full = np.array([1, 0, 3, 0, 2.0])
    y_miss = y_full.copy()
    y_miss[[1, 3]] = np.nan
    lp = G.category_logprob(0.7, A, TAU, y_miss).sum()
    keep = [0, 2, 4]
    ref = G.category_logprob(0.7, A[keep], TAU[keep], y_full[keep]).sum()
    assert lp == pytest.approx(ref)
    d = G.theta_derivs(np.array(0.7), A, TAU, y_miss)
    dref = G.theta_derivs(np.array(0.7), A[keep], TAU[keep], y_full[keep])
    assert np.allclose(d, dref)


def test_sampler_matches_category_probabilities():
    rng = np.random.default_rng(1)
    Y = G.sample_items(rng, np.full(200_000, 1.1), A, TAU)
    P = probs(1.1)
    for k in range(5):
        freq = np.bincount(Y[:, k].astype(int), minlength=5) / len(Y)
        assert np.allclose(freq, P[k], atol=0.004)
    assert set(np.unique(Y)) <= {0, 1, 2, 3, 4}


def test_stage1_likelihood_matches_direct_integration():
    Y = np.array([[0, 1, 2, 0, 1], [3, 2, 4, 1, 2], [0, 0, 1, np.nan, 0]], float)
    phi = G.pack_items(A, TAU)
    nll, _ = G.stage1_negloglik(phi, Y)
    exact = 0.0
    for y in Y:
        f = lambda th: np.exp(G.category_logprob(th, A, TAU, y).sum()) * norm.pdf(th)  # noqa: E731
        exact += np.log(integrate.quad(f, -12, 12, limit=200)[0])
    assert -nll == pytest.approx(exact, abs=1e-6)


def test_stage1_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    Y = G.sample_items(rng, rng.standard_normal(60), A, TAU)
    Y[3, 2] = np.nan
    phi = G.pack_items(A, TAU) + rng.normal(scale=0.1, size=25)
    _, g = G.stage1_negloglik(phi, Y)
    h = 1e-6
    fd = np.array([(G.stage1_negloglik(phi + h * e, Y)[0] - G.stage1_negloglik(phi - h * e, Y)[0])
                   / (2 * h) for e in np.eye(25)])
    assert np.allclose(g, fd, rtol=1e-5, atol=1e-6)


def _long_data(rng, n_pat=40, beta=(-0.4, 0.15), D=((0.5, 0.01), (0.01, 0.02))):
    times = {i: np.sort(np.r_[0, rng.uniform(0.5, 8, rng.integers(1, 7))]) for i in range(n_pat)}
    items = G.simulate(rng, times, np.array(beta), np.array(D), A, TAU)
    return times, items


def test_laplace_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    packed = G.pack(*_long_data(rng))
    psi = np.array([-0.3, 0.12, np.log(0.7), 0.02, np.log(0.12)])
    _, g = G.laplace_negloglik(psi, packed, A, TAU)
    h = 1e-6
    fd = np.array([(G.laplace_negloglik(psi + h * e, packed, A, TAU)[0]
                    - G.laplace_negloglik(psi - h * e, packed, A, TAU)[0]) / (2 * h)
                   for e in np.eye(5)])
    assert np.allclose(g, fd, rtol=1e-4, atol=1e-4)


def test_laplace_close_to_adaptive_quadrature():
    rng = np.random.default_rng(4)
    beta, D = np.array([-0.4, 0.15]), np.array([[0.5, 0.01], [0.01, 0.02]])
    times, items = _long_data(rng, n_pat=6)
    for p in times:
        lap = G.laplace_loglik_one(times[p], items[p], beta, D, A, TAU)
        agq = G.agq_loglik(times[p], items[p], beta, D, A, TAU)
        assert lap == pytest.approx(agq, abs=0.05)


def test_stage1_recovers_item_parameters():
    rng = np.random.default_rng(5)
    Y = G.sample_items(rng, rng.standard_normal(6000), A, TAU)
    f = G.fit_items(Y)
    assert f.converged
    assert np.allclose(f.a, A, rtol=0.15)
    assert np.allclose(f.tau[:, :3], TAU[:, :3], atol=0.2)


def test_stage2_recovers_longitudinal_parameters():
    rng = np.random.default_rng(6)
    beta, D = np.array([-0.4, 0.15]), np.array([[0.5, 0.01], [0.01, 0.02]])
    times, items = _long_data(rng, n_pat=700, beta=beta, D=D)
    f = G.fit_longitudinal(G.pack(times, items), A, TAU)
    assert f.converged
    assert f.beta == pytest.approx(beta, abs=0.08)
    assert np.sqrt(np.diag(f.D)) == pytest.approx(np.sqrt(np.diag(D)), rel=0.2)


def test_fit_items_rejects_invalid_responses():
    with pytest.raises(ValueError):
        G.fit_items(np.array([[0, 1, 5, 0, 1.0]]))
    with pytest.raises(ValueError):
        G.fit_items(np.array([[0, 1, 101, 0, 1.0]]))
    with pytest.raises(ValueError):
        G.fit_items(np.array([[0, 1.5, 2, 0, 1.0]]))
