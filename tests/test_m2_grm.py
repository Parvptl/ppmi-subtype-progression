"""Unit and small simulation-recovery tests for M2 (m2_grm) and its gates (m2_gates)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype import grm as G
from ppmi_subtype import m2_gates as MG
from ppmi_subtype import m2_grm as M
from ppmi_subtype.config import load_config


def _items(KT=4, KP=3, seed=0):
    rng = np.random.default_rng(seed)
    def mk(K):
        a = rng.uniform(1.0, 2.5, K)
        tau = np.sort(rng.normal(0, 1, (K, 4)), axis=1) + np.arange(4) * 0.3
        return a, tau
    aT, tauT = mk(KT)
    aP, tauP = mk(KP)
    return M.Items(aT, tauT, aP, tauP)


PSI = M.to_psi([0.2, 0.1, 0.3, -0.4, 0.15, 0.2],
               np.array([[0.6, 0.02, 0.1, 0.0], [0.02, 0.02, 0.0, 0.005],
                         [0.1, 0.0, 0.7, 0.03], [0.0, 0.005, 0.03, 0.03]]),
               np.array([[0.3, 0.05], [0.05, 0.25]]))


def _synthetic(rng, it, n_pat=40, psi=PSI):
    times, treat, YT, YP = {}, {}, {}, {}
    for p in range(n_pat):
        n = int(rng.integers(2, 5))
        t = np.sort(rng.uniform(0, 5, n))
        x = (t > rng.uniform(0, 4)).astype(float)
        th = M.simulate_latent(rng, t[None], x[None], psi)
        yt, yp = M.simulate_items(rng, th, n, it)
        times[p], treat[p], YT[p], YP[p] = t, x, yt[0].astype(float), yp[0].astype(float)
    return M.make_groups(times, treat, YT, YP)


def test_unpack_roundtrip_and_covariance_structure():
    beta, D, _, S, _ = M.unpack(PSI)
    assert np.allclose(M.to_psi(beta, D, S), PSI)
    T = np.array([[0.0, 1.0, 2.5]])
    K, _ = M.covariance(T, D, S)
    Z = M.z_design(T)[0]
    want = Z @ D @ Z.T + np.kron(S, np.eye(3)) + M.JITTER * np.eye(6)
    assert np.allclose(K[0], want)
    # tremor-PIGD visit covariance = D cross terms + Σε[0,1] at the same visit
    assert K[0][0, 3] == pytest.approx(D[0, 2] + S[0, 1])
    # tremor at t=0 with PIGD at t=1 (different visits, no ε term): D02 + 1.0 * D03
    assert K[0][0, 4] == pytest.approx(D[0, 2] + D[0, 3])
    assert len(M.NAMES) == M.N_PSI == 19


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    it = _items()
    groups = _synthetic(rng, it, 25)
    psi = PSI + rng.normal(0, 0.05, M.N_PSI)
    f, g = M.negloglik(psi, groups, it)
    h = 1e-5
    num = np.array([(M.negloglik(psi + h * e, groups, it)[0] - M.negloglik(psi - h * e, groups, it)[0])
                    / (2 * h) for e in np.eye(M.N_PSI)])
    assert np.allclose(g, num, rtol=1e-4, atol=1e-4)


def test_laplace_close_to_agq_and_is():
    rng = np.random.default_rng(2)
    it = _items()
    t, x = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    th = M.simulate_latent(rng, t[None], x[None], PSI)
    yt, yp = M.simulate_items(rng, th, 2, it)
    lap = M.laplace_one(t, x, yt[0], yp[0], PSI, it)
    agq = M.agq_one(t, x, yt[0], yp[0], PSI, it, nodes=7)
    est, se = M.is_one(t, x, yt[0], yp[0], PSI, it, rng, draws=20000)
    assert abs(agq - est) < max(5 * se, 0.01)
    assert abs(lap - agq) < 0.2                     # Laplace is an approximation


def test_agq_exact_for_gaussian_limit():
    """With all items missing the integrand is the prior: log Z = 0 exactly."""
    it = _items()
    t, x = np.array([0.0, 2.0]), np.zeros(2)
    YT, YP = np.full((2, it.KT), np.nan), np.full((2, it.KP), np.nan)
    assert M.agq_one(t, x, YT, YP, PSI, it, nodes=5) == pytest.approx(0.0, abs=1e-8)
    assert M.laplace_one(t, x, YT, YP, PSI, it) == pytest.approx(0.0, abs=1e-8)


def test_conditional_b_and_elliptical_slice_prior():
    rng = np.random.default_rng(3)
    t, x = np.array([0.0, 1.5, 3.0]), np.array([0.0, 0.0, 1.0])
    g, Mn, K = M.one_patient(t, x, np.zeros((3, 2)), np.zeros((3, 2)), PSI)
    # conditional of b given θ: covariance PSD and smaller than D
    beta, D, _, S, _ = M.unpack(PSI)
    f = Mn + np.linalg.cholesky(K) @ rng.standard_normal(6)
    mb, Vb = M.conditional_b(f, t, x, PSI)
    assert np.all(np.linalg.eigvalsh(Vb) > -1e-10)
    assert np.all(np.diag(Vb) <= np.diag(D) + 1e-12)
    # ESS with a flat likelihood samples the prior
    C = np.linalg.cholesky(K)
    cur, draws = Mn.copy(), []
    for _ in range(4000):
        cur = M.elliptical_slice(cur, Mn, C, lambda v: 0.0, rng)
        draws.append(cur)
    draws = np.array(draws)
    assert np.allclose(draws.mean(0), Mn, atol=0.1)
    assert np.allclose(np.cov(draws.T), K, atol=0.12)


def test_simulated_items_are_integers_and_labels_follow_rule():
    rng = np.random.default_rng(4)
    it = _items(11, 5)
    T = np.tile(np.linspace(0, 4, 5), (200, 1))
    th = M.simulate_latent(rng, T, np.zeros_like(T), PSI)
    yt, yp = M.simulate_items(rng, th, 5, it)
    assert yt.dtype.kind == "i" and yt.min() >= 0 and yt.max() <= 4 and yp.max() <= 4
    scfg = load_config().subtype
    tsum = np.array([0, 3, 0, 2, 4, 5, 10, 2])
    psum = np.array([0, 0, 2, 1, 2, 2, 2, 5])
    lab = M.labels_from_sums(tsum, psum, scfg)
    # both zero -> Ind; P=0,T>0 -> TD; 2/1 ratio .909 -> Ind; 4/2 .909 Ind; 5/2 1.136 Ind; 10/2 TD; 2/5 PIGD
    assert lab.tolist() == [1, 0, 2, 1, 1, 1, 0, 2]


def test_gate_features_and_rules():
    pid = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    YT = np.array([[1, 0], [0, 0], [2, 1], [0, 0], [1, 1], [0, 0], [3, 0], [0, 1]])
    YP = np.array([[0, 0], [0, 0], [1, 0], [0, 1], [0, 0], [0, 0], [1, 1], [0, 0]])
    f = MG.features(YT, YP, pid)
    P = YP.sum(1)
    assert f["G2_P_zero"] == pytest.approx((P == 0).mean())
    # consecutive pairs: (0,1),(1,2),(3,4),(4,5),(6,7); from P=0: 0->0, 0->1, 0->0 ; next0|0 = 2/3
    assert f["G4_next0_given0"] == pytest.approx(2 / 3)
    assert f["G4_next0_given1"] == pytest.approx(1.0)       # P=1 at idx3 -> 0; idx2 (P=1) is last of patient 0
    assert f["G1_item_props"].shape == (4, 5)
    assert np.allclose(f["G1_item_props"].sum(1), 1)
    sims = [{**f, "G2_P_zero": f["G2_P_zero"] + d} for d in (0.0, 0.01, -0.01)]
    ev = MG.evaluate(f, sims)
    assert ev["summary"]["G2"] and ev["decision"]["family_B_evaluable"] in (True, False)
    bad = [{**s, "G4_next0_given0": 0.2} for s in sims]
    ev2 = MG.evaluate(f, bad)
    assert not ev2["summary"]["G4"] and ev2["decision"]["target2_uninformative_by_rule"]


def test_dispersion_matches_audit_formula():
    rng = np.random.default_rng(5)
    pid = np.repeat(np.arange(30), rng.integers(1, 6, 30))
    x = rng.integers(0, 20, pid.size).astype(float)
    s = pd.Series(x)
    g = s.groupby(pid)
    m, nn = g.transform("mean"), g.transform("size")
    sel = nn >= 3
    rv = ((s - m)[sel] ** 2).sum() / (sel.sum() - g.ngroups)
    pbar = (m[sel] / 20).clip(1e-9, 1 - 1e-9)
    want = rv / (20 * pbar * (1 - pbar)).mean()
    assert MG.dispersion_ratio(x, pid, 5) == pytest.approx(want)


def test_small_recovery_of_longitudinal_parameters():
    """Items fixed at truth; 250 patients; the fit recovers ψ within ~4 Wald SE."""
    rng = np.random.default_rng(6)
    it = _items(6, 5, seed=6)
    groups = _synthetic(rng, it, 250)
    fit = M.fit(groups, it, maxiter=400)
    assert fit.converged and not fit.at_bound
    H = M.hessian(fit.psi, groups, it)
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    z = (fit.psi - PSI) / se
    assert np.all(np.abs(z) < 4.0), dict(zip(M.NAMES, np.round(z, 2)))


def test_rhat_and_ess():
    rng = np.random.default_rng(7)
    iid = rng.standard_normal((4, 1000))
    assert MG.split_rhat(iid) == pytest.approx(1.0, abs=0.01)
    assert 2500 < MG.bulk_ess(iid) < 5500
    ar = np.zeros((4, 1000))
    for c in range(4):
        for t in range(1, 1000):
            ar[c, t] = 0.95 * ar[c, t - 1] + rng.standard_normal()
    assert MG.bulk_ess(ar) < 300
    shifted = iid + np.array([0, 0, 0, 3.0])[:, None]
    assert MG.split_rhat(shifted) > 1.1
