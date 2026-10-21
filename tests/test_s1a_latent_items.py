"""Tests for the S1-a oracle-latent item likelihood (Step 5 plan, doc 34 §11)."""

import importlib.util
from pathlib import Path

import numpy as np

from ppmi_subtype import grm as G

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("s1a", ROOT / "scripts/51_m2a1_s1a_true_latent_items.py")
S1A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S1A)


def _data(V=300, K=4, seed=0):
    rng = np.random.default_rng(seed)
    a = np.linspace(1.0, 2.0, K)
    tau = np.linspace(-1.0, 0.5, K)[:, None] + 0.7 * np.arange(4)
    th = rng.normal(0.3, 0.8, V)
    F = 1 / (1 + np.exp(-a[:, None] * (th[:, None, None] - tau)))            # (V, K, 4)
    Y = (rng.random((V, K, 1)) < F).sum(-1).astype(float)
    Y[rng.random((V, K)) < 0.05] = np.nan
    m = rng.normal(0.2, 0.5, V)
    s = rng.uniform(0.4, 1.2, V)
    return Y, G.pack_items(a, tau) + rng.normal(0, 0.05, 5 * K), m, s


def test_reduces_to_stage1_at_standard_normal():
    Y, phi, _, _ = _data()
    V = Y.shape[0]
    f0, g0 = G.stage1_negloglik(phi, Y)
    f1, g1 = S1A.latent_negloglik(phi, Y, np.zeros(V), np.ones(V))
    assert abs(f1 - f0) <= 1e-12 * abs(f0)
    np.testing.assert_allclose(g1, g0, rtol=1e-12, atol=1e-12)


def test_gradient_matches_central_differences():
    Y, phi, m, s = _data(seed=1)
    _, g = S1A.latent_negloglik(phi, Y, m, s)
    h = 1e-6
    fd = np.array([(S1A.latent_negloglik(phi + h * e, Y, m, s)[0] - S1A.latent_negloglik(phi - h * e, Y, m, s)[0])
                   / (2 * h) for e in np.eye(phi.size)])
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-6)


def test_affine_invariance_common_mean_and_sd():
    # θ = m + s x:  a (θ - τ) = (a s)(x - (τ - m)/s), so the oracle value at items (a, τ) equals
    # the N(0, 1) value at items (a s, (τ - m)/s)
    Y, phi, _, _ = _data(seed=2)
    V, K = Y.shape
    m0, s0 = 0.37, 0.64
    a, tau = G.unpack_items(phi, K)
    f_lat, _ = S1A.latent_negloglik(phi, Y, np.full(V, m0), np.full(V, s0))
    f_std, _ = G.stage1_negloglik(G.pack_items(a * s0, (tau - m0) / s0), Y)
    assert abs(f_lat - f_std) <= 1e-10 * abs(f_std)
