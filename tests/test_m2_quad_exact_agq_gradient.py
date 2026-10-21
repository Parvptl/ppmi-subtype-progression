"""Exact AGQ gradient (docs/cluster_phase/27_STAGE2_EXACT_AGQ_GRADIENT_AMENDMENT_PROPOSED_20261018.md §5).

A  finite differences of the SAME moving-node objective (nodes re-placed, mode re-solved
   at ψ ± h): |g_exact − g_FD| <= 1e-5 max(1, |g_FD|) at h = 1e-4, G = 3 and 5.
B1 objective vs m2_grm.agq_one (independent code) — reported (asserted only loosely).
B2 G = 1: AGQ value == Laplace value (<= 1e-10), exact AGQ gradient == Laplace gradient
   (<= 1e-8 max(1, |g|)).
B3 exact AGQ gradient vs finite differences of a near-exact integral — reported.
C2 primary Laplace bitwise equal to m2_quad_reference_v3.
C3 default AGQ (value, fixed-placement gradient) bitwise equal to reference v3; the
   agq_exact value equals the agq value bitwise.
The fixed-placement gradient is never the correctness reference (reported as contrast).
"""

from __future__ import annotations

import numpy as np
import pytest

from ppmi_subtype import m2_grm as M
from ppmi_subtype import m2_quad as MQ
from ppmi_subtype import m2_quad_reference_v3 as REF3

PSI = M.to_psi([0.2, 0.1, 0.3, -0.4, 0.15, 0.2],
               np.array([[0.6, 0.02, 0.1, 0.0], [0.02, 0.02, 0.0, 0.005],
                         [0.1, 0.0, 0.7, 0.03], [0.0, 0.005, 0.03, 0.03]]),
               np.array([[0.3, 0.05], [0.05, 0.25]]))
H_STEPS = (1e-2, 1e-3, 1e-4, 1e-5)
TOL_A = 1e-5


def _items(KT=4, KP=3, seed=0):
    rng = np.random.default_rng(seed)
    def mk(K):
        return rng.uniform(1.0, 2.5, K), np.sort(rng.normal(0, 1, (K, 4)), axis=1) + np.arange(4) * 0.3
    aT, tauT = mk(KT)
    aP, tauP = mk(KP)
    return M.Items(aT, tauT, aP, tauP)


def _group(rng, it, n_pat, n):
    times, treat, YT, YP = {}, {}, {}, {}
    for p in range(n_pat):
        t = np.sort(rng.uniform(0, 5, n))
        x = (t > rng.uniform(0, 4)).astype(float)
        th = M.simulate_latent(rng, t[None], x[None], PSI)
        yt, yp = M.simulate_items(rng, th, n, it)
        times[p], treat[p], YT[p], YP[p] = t, x, yt[0].astype(float), yp[0].astype(float)
    return next(iter(M.make_groups(times, treat, YT, YP).values()))


def _points(rng):
    pts = {"psi": PSI, "perturb_a": PSI + rng.normal(0, 0.05, M.N_PSI),
           "perturb_b": PSI + rng.normal(0, 0.05, M.N_PSI)}
    shifted = PSI.copy()
    shifted[:M.N_MEAN] += 0.5                       # moves the mode b̂ materially
    pts["mean_shift"] = shifted
    return pts


def _value(psi, g, it, Q, G, b0):
    return float(MQ.group_agq(psi, g, it, Q, G, b0=b0, want_grad=False)[0].sum())


def fd_errors(psi, g, it, Q, G):
    val, b, nd, g_exact = MQ.group_agq(psi, g, it, Q, G, gradient="exact")
    _, _, _, g_fixed = MQ.group_agq(psi, g, it, Q, G)
    ge, gf = g_exact.sum(0), g_fixed.sum(0)
    out = {}
    for h in H_STEPS:
        fd = np.array([(_value(psi + h * e, g, it, Q, G, b) - _value(psi - h * e, g, it, Q, G, b)) / (2 * h)
                       for e in np.eye(M.N_PSI)])
        scale = np.maximum(1.0, np.abs(fd))
        out[h] = {"exact": float(np.max(np.abs(ge - fd) / scale)), "fixed": float(np.max(np.abs(gf - fd) / scale))}
    return out, bool(nd.all())


@pytest.mark.parametrize("G", [3, 5])
def test_A_exact_gradient_matches_moving_node_finite_differences(G):
    rng = np.random.default_rng(100 + G)
    it = _items()
    report = []
    for n in (1, 2, 4):
        g = _group(rng, it, 3, n)
        for name, psi in _points(rng).items():
            errs, all_nd = fd_errors(psi, g, it, 7, G)
            report.append((n, name, all_nd, errs))
            assert all_nd, (n, name)
            assert errs[1e-4]["exact"] <= TOL_A, (G, n, name, errs)
    # the contrast: somewhere the fixed-placement gradient is clearly not this objective's gradient
    assert max(r[3][1e-4]["fixed"] for r in report) > 100 * TOL_A
    # O(h^2) decrease from h = 1e-2 to 1e-3 for the exact gradient (until rounding dominates)
    for *_, errs in report:
        assert errs[1e-3]["exact"] <= max(errs[1e-2]["exact"] / 10, 1e-7)


def test_B1_objective_close_to_independent_full_theta_agq():
    rng = np.random.default_rng(7)
    it = _items()
    for n, nodes in ((1, 25), (2, 11)):
        g = _group(rng, it, 2, n)
        for r in range(2):
            t, x, yt, yp = g.T[r], g.X[r], g.YT[r], g.YP[r]
            ref = M.agq_one(t, x, yt, yp, PSI, it, nodes=nodes)
            one = M.Group(g.T[r:r + 1], g.X[r:r + 1], g.YT[r:r + 1], g.YP[r:r + 1], [0])
            d = [abs(MQ.group_agq(PSI, one, it, 20, G, want_grad=False)[0][0] - ref) for G in (3, 5, 7)]
            assert d[2] < 0.02 and d[1] < 0.05             # loose: B1 is reported, not gated


def test_B2_G1_reduces_to_laplace_value_and_gradient():
    rng = np.random.default_rng(8)
    it = _items()
    for n in (1, 3, 5):
        g = _group(rng, it, 4, n)
        for psi in _points(rng).values():
            lv, lb, lnd, _, lg = MQ.group_laplace(psi, g, it, 7)
            av, ab, and_, ag = MQ.group_agq(psi, g, it, 7, 1, gradient="exact")
            assert np.allclose(av, lv, rtol=0, atol=1e-10)
            assert np.allclose(ag, lg, rtol=0, atol=1e-8 * max(1.0, np.abs(lg).max()))


def test_B3_exact_gradient_approaches_near_exact_integral_gradient():
    """Reported quantity; asserted only to fall from G = 3 to G = 7."""
    rng = np.random.default_rng(9)
    it = _items()
    g = _group(rng, it, 1, 1)
    t, x, yt, yp = g.T[0], g.X[0], g.YT[0], g.YP[0]
    h = 1e-4
    fd = np.array([(M.agq_one(t, x, yt, yp, PSI + h * e, it, nodes=25)
                    - M.agq_one(t, x, yt, yp, PSI - h * e, it, nodes=25)) / (2 * h) for e in np.eye(M.N_PSI)])
    errs = [float(np.max(np.abs(MQ.group_agq(PSI, g, it, 20, G, gradient="exact")[3][0] - fd)))
            for G in (3, 5, 7)]
    assert errs[2] < errs[0]


def test_C2_C3_primary_and_default_agq_bitwise_equal_reference_v3():
    rng = np.random.default_rng(10)
    it = _items()
    for n in (1, 3, 6):
        g = _group(rng, it, 5, n)
        psi = PSI + rng.normal(0, 0.05, M.N_PSI)
        a = MQ.group_laplace(psi, g, it, 7)
        r = REF3.group_laplace(psi, g, it, 7)
        assert np.array_equal(a[0], r[0]) and np.array_equal(a[4], r[4]) and np.array_equal(a[1], r[1])
        a = MQ.group_agq(psi, g, it, 7, 3)
        r = REF3.group_agq(psi, g, it, 7, 3)
        assert np.array_equal(a[0], r[0]) and np.array_equal(a[3], r[3])
        ex = MQ.group_agq(psi, g, it, 7, 3, gradient="exact")
        assert np.array_equal(ex[0], a[0]) and np.array_equal(ex[1], a[1])
