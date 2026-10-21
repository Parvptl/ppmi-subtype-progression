"""V1 unit tests for the Amendment 1 (Rev 2) prototype `m2_quad`, and regression tests
of the optimised `m2_quad` against the B0-benchmarked version kept verbatim as
`m2_quad_reference` (SHA-256 68d3dcef…, equal to the B0 provenance record).

Regression tolerances (justified):
* fused GRM kernel vs grm.category_logprob / theta_derivs: same formulas, different
  operation order -> 1e-10 absolute on Σ log P, 1e-8 relative on derivatives;
* cumulant restructuring: exact algebra, different summation order -> 1e-10 relative;
* cohort likelihood: both implementations solve the per-patient mode to a Newton step
  < 1e-9; the Laplace value is stationary in b, so per-patient values agree to
  ~1e-12 -> 1e-8 tolerance; the gradient depends on b̂ to first order
  (|Δb̂| <~ 1e-9 times third-derivative terms) -> 1e-6 (1 + |g|);
* process pool vs serial: identical per-chunk arithmetic, different summation
  order over chunks -> 1e-10 relative on the value, 1e-8 on the gradient.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppmi_subtype import grm as G
from ppmi_subtype import m2_grm as M
from ppmi_subtype import m2_quad as MQ

PSI = M.to_psi([0.2, 0.1, 0.3, -0.4, 0.15, 0.2],
               np.array([[0.6, 0.02, 0.1, 0.0], [0.02, 0.02, 0.0, 0.005],
                         [0.1, 0.0, 0.7, 0.03], [0.0, 0.005, 0.03, 0.03]]),
               np.array([[0.3, 0.05], [0.05, 0.25]]))


def _items(KT=4, KP=3, seed=0):
    rng = np.random.default_rng(seed)
    def mk(K):
        return rng.uniform(1.0, 2.5, K), np.sort(rng.normal(0, 1, (K, 4)), axis=1) + np.arange(4) * 0.3
    aT, tauT = mk(KT)
    aP, tauP = mk(KP)
    return M.Items(aT, tauT, aP, tauP)


def _groups(rng, it, n_pat=12, nvis=(1, 4)):
    times, treat, YT, YP = {}, {}, {}, {}
    for p in range(n_pat):
        n = int(rng.integers(*nvis))
        t = np.sort(rng.uniform(0, 5, n))
        x = (t > rng.uniform(0, 4)).astype(float)
        th = M.simulate_latent(rng, t[None], x[None], PSI)
        yt, yp = M.simulate_items(rng, th, n, it)
        times[p], treat[p], YT[p], YP[p] = t, x, yt[0].astype(float), yp[0].astype(float)
    return M.make_groups(times, treat, YT, YP)


def test_eps_nodes_reproduce_gaussian_mgf():
    """Σ w exp(cᵀε) over the Q×Q nodes equals exp(½ cᵀΣ_ε c) (closed form)."""
    E, logw, U, C, _ = MQ.eps_nodes(PSI, 20)
    S = C @ C.T
    assert np.allclose(S, M.unpack(PSI)[3])
    for c in (np.array([0.5, -0.3]), np.array([1.0, 1.0])):
        est = np.exp(logw + E @ c).sum()
        assert est == pytest.approx(np.exp(0.5 * c @ S @ c), rel=1e-10)
    assert np.exp(logw).sum() == pytest.approx(1.0)


def test_no_items_gives_log_likelihood_zero():
    it = _items()
    g = M.Group(np.array([[0.0, 1.0, 2.0]]), np.zeros((1, 3)), np.full((1, 3, it.KT), np.nan),
                np.full((1, 3, it.KP), np.nan), [0])
    val, b, nd, sH, _ = MQ.group_laplace(PSI, g, it, 7, want_grad=False)
    assert val[0] == pytest.approx(0.0, abs=1e-10) and nd[0]
    val2, *_ = MQ.group_agq(PSI, g, it, 7, 3, want_grad=False)
    assert val2[0] == pytest.approx(0.0, abs=1e-10)


def test_same_model_as_m2_grm_on_a_one_visit_patient():
    """High-order quadrature here equals m2_grm's AGQ over the full 2-D θ (same integral)."""
    rng = np.random.default_rng(3)
    it = _items()
    t, x = np.array([1.0]), np.array([1.0])
    th = M.simulate_latent(rng, t[None], x[None], PSI)
    yt, yp = M.simulate_items(rng, th, 1, it)
    ref = M.agq_one(t, x, yt[0].astype(float), yp[0].astype(float), PSI, it, nodes=25)
    g = M.Group(t[None], x[None], yt.astype(float), yp.astype(float), [0])
    val, *_ = MQ.group_agq(PSI, g, it, 30, 9, want_grad=False)
    assert val[0] == pytest.approx(ref, abs=2e-4)
    lap, *_ = MQ.group_laplace(PSI, g, it, 30, want_grad=False)
    assert abs(lap[0] - ref) < 0.05                    # Laplace over b: an approximation


def test_laplace_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    it = _items()
    groups = _groups(rng, it)
    psi = PSI + rng.normal(0, 0.05, M.N_PSI)
    f, g = MQ.negloglik(psi, groups, it, Q=7)
    h = 1e-5
    num = np.array([(MQ.negloglik(psi + h * e, groups, it, Q=7, want_grad=False)
                     - MQ.negloglik(psi - h * e, groups, it, Q=7, want_grad=False)) / (2 * h)
                    for e in np.eye(M.N_PSI)])
    assert np.allclose(g, num, rtol=1e-4, atol=1e-4), dict(zip(M.NAMES, np.round(g - num, 6)))


def test_agq_gradient_at_fixed_nodes_matches_finite_differences():
    rng = np.random.default_rng(2)
    it = _items()
    groups = _groups(rng, it, n_pat=6)
    n, g = next(iter(groups.items()))
    psi = PSI + rng.normal(0, 0.05, M.N_PSI)
    _, b, nd, _ = MQ.group_agq(psi, g, it, 7, 3, want_grad=False)
    beta, D, *_ = M.unpack(psi)
    L = np.broadcast_to(0.3 * np.eye(4), (len(b), 4, 4)).copy()
    _, _, _, gr = MQ.group_agq(psi, g, it, 7, 3, fixed=(b, L))
    h = 1e-5
    num = np.array([(MQ.group_agq(psi + h * e, g, it, 7, 3, want_grad=False, fixed=(b, L))[0].sum()
                     - MQ.group_agq(psi - h * e, g, it, 7, 3, want_grad=False, fixed=(b, L))[0].sum()) / (2 * h)
                    for e in np.eye(M.N_PSI)])
    assert np.allclose(gr.sum(0), num, rtol=1e-4, atol=1e-4)


def test_mode_is_unique_and_hessian_negative_definite_on_random_cases():
    rng = np.random.default_rng(4)
    it = _items()
    groups = _groups(rng, it, n_pat=10, nvis=(2, 6))
    beta, D, *_ = M.unpack(PSI)
    Dinv = np.linalg.inv(D)
    E, logw, *_ = MQ.eps_nodes(PSI, 10)
    for n, g in groups.items():
        d = MQ.prepare(g, it)
        mu = MQ.visit_means(PSI, g.T, g.X)
        b1, nd1, _ = MQ.modes(d, mu, E, logw, Dinv)
        b2, _, _ = MQ.modes(d, mu, E, logw, Dinv, b0=rng.normal(0, 1, b1.shape))
        assert np.allclose(b1, b2, atol=1e-6) and nd1.all()


def test_laplace_close_to_agq_over_b():
    rng = np.random.default_rng(5)
    it = _items()
    groups = _groups(rng, it, n_pat=8, nvis=(2, 5))
    for n, g in groups.items():
        lap, *_ = MQ.group_laplace(PSI, g, it, 10, want_grad=False)
        agq, *_ = MQ.group_agq(PSI, g, it, 10, 5, want_grad=False)
        assert np.all(np.abs(lap - agq) < 0.1)


# ---------------------------------------------------------------------------
#  Regression: optimised vs B0-benchmarked reference
# ---------------------------------------------------------------------------
from ppmi_subtype import m2_quad_reference as REF  # noqa: E402


def test_fused_kernel_matches_grm():
    rng = np.random.default_rng(10)
    it = _items(6, 5)
    theta = np.concatenate([rng.normal(0, 2, (3, 4, 9)), np.full((1, 4, 9), 30.0), np.full((1, 4, 9), -30.0)])
    Y = rng.integers(0, 5, (5, 4, 6)).astype(float)
    Y[0, 1, 2] = np.nan
    br = MQ.brackets(Y, it.aT, it.tauT)
    v, l1, l2, l3 = MQ.grm_terms(theta, br, 3)
    ref_v = G.category_logprob(theta, it.aT, it.tauT, Y[:, :, None, :]).sum(-1)
    r1, r2, r3 = G.theta_derivs(theta, it.aT, it.tauT, Y[:, :, None, :])
    assert np.allclose(v, ref_v, rtol=0, atol=1e-10)
    for x, y in ((l1, r1), (l2, r2), (l3, r3)):
        assert np.allclose(x, y, rtol=1e-8, atol=1e-10)


def test_cumulants_match_reference_algebra():
    rng = np.random.default_rng(11)
    m, n, Q = 3, 4, 9
    pi = rng.dirichlet(np.ones(Q), (m, n))
    g1, g2, g3 = (rng.normal(0, 1, (m, n, Q, 2)) for _ in range(3))
    T, X = rng.uniform(0, 5, (m, n)), (rng.random((m, n)) > 0.5).astype(float)
    E, logw, U, C, dC = MQ.eps_nodes(PSI, 3)
    Dir = REF._directions(PSI, T, X, U, dC)
    want = REF._cumulants(pi, g1, g2, g3, Dir)
    d = MQ.Data(T, X, MQ.design(T), None, None)
    DirN, DirL = MQ._dirs(PSI, d, U, dC)
    got = MQ._cumulants(pi, g1, g2, g3, DirN, DirL, DirN[:, :, :4])
    for x, y in zip(got, want):
        assert np.allclose(x, y, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("Q", [7, 10])
def test_optimised_laplace_matches_reference(Q):
    rng = np.random.default_rng(12)
    it = _items()
    groups = _groups(rng, it, n_pat=25, nvis=(1, 7))
    psi = PSI + rng.normal(0, 0.05, M.N_PSI)
    for n, g in groups.items():
        v1, b1, nd1, _, g1_ = MQ.group_laplace(psi, g, it, Q)
        v0, b0, nd0, _, g0_ = REF.group_laplace(psi, g, it, Q)
        assert np.allclose(v1, v0, rtol=0, atol=1e-8)
        assert np.allclose(b1, b0, rtol=0, atol=1e-7)
        assert np.allclose(g1_, g0_, rtol=1e-6, atol=1e-6)
        assert np.array_equal(nd1, nd0)


def test_optimised_agq_matches_reference():
    rng = np.random.default_rng(13)
    it = _items()
    groups = _groups(rng, it, n_pat=10, nvis=(1, 5))
    psi = PSI + rng.normal(0, 0.05, M.N_PSI)
    for n, g in groups.items():
        v1, *_, g1_ = MQ.group_agq(psi, g, it, 7, 3)
        v0, *_, g0_ = REF.group_agq(psi, g, it, 7, 3)
        assert np.allclose(v1, v0, rtol=0, atol=1e-8)
        assert np.allclose(g1_, g0_, rtol=1e-6, atol=1e-6)


def test_cohort_objective_serial_parallel_and_reference_agree():
    rng = np.random.default_rng(14)
    it = _items()
    groups = _groups(rng, it, n_pat=40, nvis=(1, 6))
    psi = PSI + rng.normal(0, 0.05, M.N_PSI)
    f0, g0 = REF.negloglik(psi, groups, it, Q=7)
    f1, g1 = MQ.negloglik(psi, groups, it, Q=7)
    obj = MQ.Objective(groups, it, Q=7, jobs=2, chunk=7)
    try:
        f2, g2 = obj(psi)
    finally:
        obj.close()
    assert f1 == pytest.approx(f0, abs=1e-8 * len(groups) * 40)
    assert np.allclose(g1, g0, rtol=1e-6, atol=1e-6)
    assert f2 == pytest.approx(f1, rel=1e-10)
    assert np.allclose(g2, g1, rtol=1e-8, atol=1e-8)
