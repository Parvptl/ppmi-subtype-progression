"""Tests for the exploratory Step 6 failure diagnostics (scripts 53 and 54)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from ppmi_subtype import grm as G
from ppmi_subtype import m2_grm as M
from ppmi_subtype import m2_quad as MQ

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R53 = _load("r53_test", ROOT / "scripts/53_m2_step6_diag_readonly.py")
D = _load("d54_test", ROOT / "scripts/54_m2_step6_diag_compute.py")


def _toy(seed=0, m=6, n=3):
    rng = np.random.default_rng(seed)
    T = np.sort(rng.uniform(0, 4, (m, n)), axis=1)
    X = (rng.random((m, n)) < 0.4).astype(float)
    it = M.Items(np.linspace(1.2, 2.0, 4), np.linspace(-0.5, 1.0, 4)[:, None] + 0.7 * np.arange(4),
                 np.linspace(1.5, 2.2, 3), np.linspace(0.0, 1.0, 3)[:, None] + 0.7 * np.arange(4))
    YT = rng.integers(0, 5, (m, n, 4)).astype(float)
    YP = rng.integers(0, 5, (m, n, 3)).astype(float)
    return {n: M.Group(T, X, YT, YP, list(range(m)))}, it


def test_affine_fit_recovers_exact_relabelling():
    a0 = np.linspace(1.2, 2.4, 6)
    t0 = np.linspace(-0.5, 2.0, 6)[:, None] + 0.8 * np.arange(4)
    c, d = 1.13, -0.07
    r = R53.affine_fit(a0 * c, (t0 - d) / c, a0, t0)
    for key in ("c_tau", "c_a", "c_joint"):
        assert r[key] == pytest.approx(c, rel=1e-10)
    assert r["d_joint"] == pytest.approx(d, abs=1e-10) and r["d_tau"] == pytest.approx(d, abs=1e-10)
    assert r["tau_rmse_after"] < 1e-10 and r["loga_rmse_after"] < 1e-10


def test_relabelling_leaves_the_stage2_likelihood_invariant():
    """θ'_s = (θ_s − d_s)/c_s with items (c a, (τ − d)/c) and ψ' = relabel_psi(ψ): same marginal likelihood
    under the Laplace / per-visit GH approximation (both are equivariant under diagonal rescaling)."""
    g, it = _toy()
    psi = M.default_psi() + np.random.default_rng(1).normal(0, 0.05, M.N_PSI)
    cT, dT, cP, dP = 1.07, 0.11, 0.91, -0.08
    it2 = M.Items(it.aT * cT, (it.tauT - dT) / cT, it.aP * cP, (it.tauP - dP) / cP)
    psi2 = R53.relabel_psi(psi, cT, dT, cP, dP)
    f1 = MQ.negloglik(psi, g, it, 7, want_grad=False)
    f2 = MQ.negloglik(psi2, g, it2, 7, want_grad=False)
    assert f2 == pytest.approx(f1, rel=1e-9)


def test_stacked_sandwich_matches_two_step_formula():
    rng = np.random.default_rng(3)
    n, k, p = 400, 3, 2
    U = rng.normal(size=(n, k + p))
    A = np.eye(k) * 3 + 0.1
    H = np.array([[2.0, 0.3], [0.3, 1.5]])
    C = rng.normal(size=(p, k))
    J = np.block([[A, np.zeros((k, p))], [C, H]])
    V, B = D.stacked_sandwich(J, U)
    Ai, Hi = np.linalg.inv(A), np.linalg.inv(H)
    B11, B12, B22 = B[:k, :k], B[:k, k:], B[k:, k:]
    expect = Hi @ (B22 + C @ Ai @ B11 @ Ai.T @ C.T - C @ Ai @ B12 - B12.T @ Ai.T @ C.T) @ Hi.T
    np.testing.assert_allclose(V[k:, k:], expect, rtol=1e-10, atol=1e-12)


def test_patient_slices_and_score_additivity():
    pid = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
    sl = D.patient_slices(pid)
    assert [(s.start, s.stop) for s in sl] == [(0, 3), (3, 5), (5, 6), (6, 10)]
    rng = np.random.default_rng(5)
    Y = rng.integers(0, 5, (10, 3)).astype(float)
    phi = G.pack_items(np.array([1.2, 1.6, 2.0]), np.linspace(-0.5, 0.5, 3)[:, None] + 0.7 * np.arange(4))
    tot = G.stage1_negloglik(phi, Y)[1]
    parts = sum(G.stage1_negloglik(phi, Y[s])[1] for s in sl)
    np.testing.assert_allclose(parts, tot, rtol=1e-12, atol=1e-12)


def test_settle_rule_and_aitken():
    se = np.ones(3)
    assert D.settled([0.0, 0.0, 0.049], [0, 0, 0], se) and not D.settled([0.0, 0.0, 0.05], [0, 0, 0], se)
    x = [1 + 0.5 ** k for k in range(3)]                           # geometric → limit 1
    assert D.aitken(np.array([x[0]]), np.array([x[1]]), np.array([x[2]]))[0] == pytest.approx(1.0)


def test_items_from_phi_round_trip():
    _, it = _toy()
    pT, pP = D.pack_set(it)
    it2 = D.items_from_phi(pT, pP, len(it.aT), len(it.aP))
    np.testing.assert_allclose(it2.aT, it.aT, rtol=1e-14)
    np.testing.assert_allclose(it2.tauP, it.tauP, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("sub,par", [("T", "c"), ("T", "d"), ("P", "c"), ("P", "d")])
def test_affine_tangent_satisfies_first_order_invariance(sub, par):
    """d/dc F(ψ'(c), φ'(c)) = 0  ⇒  ∇ψF·v_ψ + ∇φF·v_φ = 0 at any (ψ, φ) (not only at an optimum)."""
    g, it = _toy(seed=2)
    psi = M.default_psi() + np.random.default_rng(4).normal(0, 0.05, M.N_PSI)
    vphi, vpsi = D.affine_tangent(it, psi, sub, par, R53.relabel_psi)
    f0, gpsi = MQ.negloglik(psi, g, it, 7)
    pT, pP = D.pack_set(it)
    phi = np.concatenate([pT, pP])
    nT = pT.size
    h = 1e-5
    gphi = np.zeros(phi.size)
    for k in range(phi.size):
        q = [phi.copy(), phi.copy()]
        q[0][k] += h
        q[1][k] -= h
        fp, fm = (MQ.negloglik(psi, g, D.items_from_phi(x[:nT], x[nT:], len(it.aT), len(it.aP)), 7, want_grad=False)
                  for x in q)
        gphi[k] = (fp - fm) / (2 * h)
    lhs = gpsi @ vpsi + gphi @ vphi
    assert abs(lhs) <= 1e-6 * max(np.abs(gpsi @ vpsi), np.abs(gphi @ vphi), 1.0)
    assert np.linalg.norm(vphi) > 0.1 and np.linalg.norm(vpsi) > 0.01
