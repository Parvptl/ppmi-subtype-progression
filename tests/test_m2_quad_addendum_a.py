"""Addendum A (docs/cluster_phase/25_…_ADDENDUM_A_FINAL_20261017.md): P1 budget, P2
negative-definiteness semantics (T5), monitoring and the reported-evaluation context.

Regression reference: `m2_quad_reference_v2` (the doc 23 optimised code, verbatim).
"""

from __future__ import annotations

import numpy as np
import pytest

from ppmi_subtype import m2_grm as M
from ppmi_subtype import m2_quad as MQ
from ppmi_subtype import m2_quad_reference_v2 as REF2

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


def _groups(rng, it, n_pat=20, nvis=(2, 6)):
    times, treat, YT, YP = {}, {}, {}, {}
    for p in range(n_pat):
        n = int(rng.integers(*nvis))
        t = np.sort(rng.uniform(0, 5, n))
        x = (t > rng.uniform(0, 4)).astype(float)
        th = M.simulate_latent(rng, t[None], x[None], PSI)
        yt, yp = M.simulate_items(rng, th, n, it)
        times[p], treat[p], YT[p], YP[p] = t, x, yt[0].astype(float), yp[0].astype(float)
    return M.make_groups(times, treat, YT, YP)


def _modes_with_bad_hessian(orig):
    """Wrap `modes`: patient 0's returned Hessian gets eigenvalues (+1, +1, -1, -1):
    not negative definite, yet det(-hess) > 0."""
    def wrapped(*a, **k):
        b, nd, loc = orig(*a, **k)
        loc.hess = loc.hess.copy()
        loc.hess[0] = np.diag([1.0, 1.0, -1.0, -1.0])
        nd = np.all(np.linalg.eigvalsh(loc.hess) < 0, axis=1)
        return b, nd, loc
    return wrapped


def test_T5_non_nd_with_positive_determinant(monkeypatch):
    rng = np.random.default_rng(1)
    it = _items()
    groups = _groups(rng, it, 8, (3, 4))
    n, g = next(iter(groups.items()))
    assert np.linalg.det(-np.diag([1.0, 1.0, -1.0, -1.0])) > 0
    # reference v2 (det-sign test) accepts this patient: finite value, no penalty
    monkeypatch.setattr(REF2, "modes", _modes_with_bad_hessian(REF2.modes))
    ref = REF2._chunk_eval(REF2.prepare(g, it), PSI, it, 7, "laplace", 3, None, False)
    assert ref[4] is False and np.isfinite(ref[0])
    # updated code, optimiser context: infeasible with reason non_nd (eigenvalue test)
    monkeypatch.setattr(MQ, "modes", _modes_with_bad_hessian(MQ.modes))
    val, gr, b, nnd, bad, diag = MQ._chunk_eval(MQ.prepare(g, it), PSI, it, 7, "laplace", 3, None, True)
    assert bad and diag["reason"] == "non_nd" and nnd == 1
    obj = MQ.Objective({n: g}, it, Q=7)
    rep: dict = {}
    assert obj(PSI, report=rep)[0] == 1e12 and rep["reasons"] == {"non_nd": 1}
    assert obj.log[-1]["penalty"] and obj.log[-1]["reasons"] == {"non_nd": 1}
    # reported context: the patient is evaluated by the declared fallback; count recorded
    out = MQ.evaluate_reported(PSI, {n: g}, it, 7, 3)
    assert out["n_fallback_patients"] == 1 and not out["b11_reported_ok"]
    d = MQ.prepare(g, it)
    lap = MQ.group_laplace(PSI, d, it, 7, want_grad=False)[0]
    agq0 = MQ.group_agq(PSI, d.rows(np.array([0])), it, 7, 3,
                        b0=MQ.group_laplace(PSI, d, it, 7, want_grad=False)[1][[0]], want_grad=False)[0]
    assert out["loglik"] == pytest.approx(lap[1:].sum() + agq0[0], rel=1e-10)


def test_P1_budget_triggers_and_reason(monkeypatch):
    rng = np.random.default_rng(2)
    it = _items()
    groups = _groups(rng, it, 6)
    n, g = next(iter(groups.items()))
    d = MQ.prepare(g, it)
    beta, D, *_ = M.unpack(PSI)
    E, logw, *_ = MQ.eps_nodes(PSI, 7)
    mu = MQ.visit_means(PSI, g.T, g.X)
    info2: dict = {}
    b, nd, loc = MQ.modes(d, mu, E, logw, np.linalg.inv(D), info=info2)
    assert not info2["failed"] and info2["halvings"] >= 0 and "max_abs_phi" in info2
    # force a trigger through the chunk evaluation: budget of zero halvings and one non-ND iteration
    orig = MQ.modes
    monkeypatch.setattr(MQ, "modes", lambda *a, **k: orig(*a, **{**k, "max_nonnd": 0}))
    val, gr, b, nnd, bad, diag = MQ._chunk_eval(d, PSI, it, 7, "laplace", 3, None, True)
    assert bad and diag["reason"] == "mode_budget" and diag["budget"] == "consecutive_non_nd"


def test_budget_not_reached_is_bitwise_identical_to_reference_v2():
    rng = np.random.default_rng(3)
    it = _items()
    groups = _groups(rng, it, 25, (1, 7))
    psi = PSI + rng.normal(0, 0.05, M.N_PSI)
    for n, g in groups.items():
        v1, b1, nd1, s1, g1 = MQ.group_laplace(psi, g, it, 7)
        v0, b0, nd0, s0, g0 = REF2.group_laplace(psi, g, it, 7)
        assert np.array_equal(v1, v0) and np.array_equal(b1, b0) and np.array_equal(g1, g0)
        a1 = MQ.group_agq(psi, g, it, 7, 3)
        a0 = REF2.group_agq(psi, g, it, 7, 3)
        assert np.array_equal(a1[0], a0[0]) and np.array_equal(a1[3], a0[3])
    f1, gr1 = MQ.negloglik(psi, groups, it, Q=7)
    f0, gr0 = REF2.negloglik(psi, groups, it, Q=7)
    assert f1 == f0 and np.array_equal(gr1, gr0)


def test_monitoring_log_and_reported_evaluation_agree():
    rng = np.random.default_rng(4)
    it = _items()
    groups = _groups(rng, it, 15)
    obj = MQ.Objective(groups, it, Q=7, chunk=4)
    v, g = obj(PSI)
    e = obj.log[-1]
    for k in ("psi", "value", "grad_norm", "n_patients_evaluated", "n_non_nd", "reasons", "halvings",
              "wall_s", "per_patient_s_median", "per_patient_s_max", "max_abs_phi", "spacing_max_abs_phi",
              "fp_flag"):
        assert k in e
    assert e["n_patients_evaluated"] == 15 and not e["penalty"] and e["value"] == v
    out = MQ.evaluate_reported(PSI, groups, it, 7, 3)
    assert out["n_fallback_patients"] == 0 and out["b11_reported_ok"]
    assert -out["loglik"] == pytest.approx(v, rel=1e-12)
