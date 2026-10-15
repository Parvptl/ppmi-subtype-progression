"""Tests for the collapsed nu_s updates.

Weight-bearing: the original option reproduces hbayes_tt exactly; the
collapsed likelihood is the product of t densities; both new draws target
p(nu | b1, tau1) exactly (grid) or in distribution (slice); the grid prior's
implied density is what the module docstring says; and on simulated data the
new samplers agree with the original one in distribution while mixing faster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ppmi_subtype import hbayes
from ppmi_subtype import hbayes_t as HT
from ppmi_subtype import hbayes_tt as TT
from ppmi_subtype import hbayes_tt_nu as NU
from ppmi_subtype import slope_clustering as SC


def synthetic(seed=0, M=300, nu_s=None, nu_e=None, tau1=0.15, visits=6, sigma=0.3):
    rng = np.random.default_rng(seed)
    b0 = rng.normal(0, 0.8, M)
    b1 = tau1 * (rng.standard_t(nu_s, M) if nu_s else rng.standard_normal(M))
    rows = []
    for i in range(M):
        n = rng.integers(2, visits + 1)
        t = np.sort(rng.uniform(0, 8, n))
        e = sigma * (rng.standard_t(nu_e, n) if nu_e else rng.standard_normal(n))
        rows += [{"PATNO": 11 + 2 * i, "years_from_baseline": tt, "ratio": yy}
                 for tt, yy in zip(t, 2.0 + b0[i] + (-0.1 + b1[i]) * t + e)]
    return hbayes.build_growth_data(pd.DataFrame(rows))


def test_lambda_grid_option_reproduces_hbayes_tt_exactly():
    d = synthetic(M=80, nu_s=3, nu_e=4)
    a = TT.fit(d, "TT", n_draws=120, burn_in=40, seed=3)
    b = NU.fit(d, "TT", nu_update="lambda_grid", n_draws=120, burn_in=40, seed=3)
    for k in ("beta", "sigma2", "tau0_2", "tau1_2", "nu", "nu_e"):
        assert np.allclose(getattr(a, k), getattr(b, k), rtol=1e-9, atol=1e-12), k


def test_collapsed_loglik_is_sum_of_t_logpdf():
    z = np.random.default_rng(1).standard_t(3, 400)
    for nu in (1.3, 4.0, 57.0):
        assert NU.collapsed_loglik(nu, z)[0] == pytest.approx(stats.t.logpdf(z, nu).sum())


def test_collapsed_conditional_equals_lambda_marginalisation():
    """p(nu | z) on the grid matches E_lambda[p(nu | lambda)] over lambda | nu draws
    -- checked through the identity  sum_lambda p(z|lambda) p(lambda|nu) = t_nu(z)."""
    rng = np.random.default_rng(2)
    z, nu = rng.standard_t(5, 50), 5.0
    lam = rng.gamma(nu / 2, 2 / nu, (200_000, 1))
    mc = np.log(stats.norm.pdf(z[None, :], 0, 1 / np.sqrt(lam)).mean(axis=0)).sum()
    assert mc == pytest.approx(NU.collapsed_loglik(nu, z)[0], abs=0.05)


def test_grid_draw_is_exact_and_slice_matches_it():
    rng = np.random.default_rng(3)
    z = rng.standard_t(3, 600)
    g = HT.NU_GRID
    lp = NU.collapsed_loglik(g, z) + HT.log_prior_nu_gamma(g)
    p = np.exp(lp - lp.max())
    p /= p.sum()
    grid_draws = np.array([NU.draw_nu_collapsed_grid(z, rng) for _ in range(4000)])
    assert grid_draws.mean() == pytest.approx((p * g).sum(), rel=0.02)
    # slice targets the continuous version: density in u = exp(loglik + log_prior)
    nu, draws = 5.0, []
    for _ in range(6000):
        nu = NU.draw_nu_collapsed_slice(nu, z, rng)
        draws.append(nu)
    draws = np.array(draws[500:])
    u = np.linspace(NU.U_LO, NU.U_HI, 20001)
    lu = NU.collapsed_loglik(np.exp(u), z) + HT.log_prior_nu_gamma(np.exp(u))
    du = np.exp(lu - lu.max())
    du /= du.sum()
    ref_q = np.exp(u[np.searchsorted(np.cumsum(du), [0.1, 0.5, 0.9])])
    assert np.allclose(np.quantile(draws, [0.1, 0.5, 0.9]), ref_q, rtol=0.05)


def test_slice_handles_bounds_and_flat_target():
    rng = np.random.default_rng(4)
    us = [1.0]
    for _ in range(20000):
        us.append(NU.slice_u(us[-1], lambda u: 0.0, rng))
    us = np.array(us)
    assert us.min() >= NU.U_LO and us.max() <= NU.U_HI
    assert stats.kstest(us[1000:], stats.uniform(NU.U_LO, NU.U_HI - NU.U_LO).cdf).statistic < 0.03


def test_implied_grid_priors():
    g = HT.NU_GRID
    for f, ref_mean in ((HT.log_prior_nu_gamma, stats.truncexpon(29.9, loc=1, scale=10)),
                        (NU.log_prior_nu_gamma2_true, None)):
        w = np.exp(f(g) - f(g).max())
        w /= w.sum()
        m = (w * g).sum()
        if ref_mean is not None:                     # "gamma" grid prior == truncated Exp(0.1)
            assert m == pytest.approx(ref_mean.mean(), rel=0.02)
        else:                                        # Jacobian-corrected == truncated Gamma(2, .1)
            x = np.linspace(1, 300, 300_000)
            d = stats.gamma.pdf(x, 2, scale=10)
            assert m == pytest.approx((d * x).sum() / d.sum(), rel=0.02)


@pytest.mark.parametrize("update", ["collapsed_grid", "collapsed_slice"])
def test_collapsed_samplers_agree_with_original_and_mix_better(update):
    d = synthetic(seed=6, M=400, nu_s=3, visits=8, sigma=0.2)
    kw = dict(n_draws=3000, burn_in=1000, thin=1)
    orig = [NU.fit(d, "TG", nu_update="lambda_grid", seed=s, **kw).nu for s in (1, 2)]
    new = [NU.fit(d, "TG", nu_update=update, seed=s, **kw).nu for s in (1, 2)]
    orig, new = np.concatenate(orig), np.concatenate(new)
    assert np.median(new) == pytest.approx(np.median(orig), rel=0.2)
    assert np.median(new) < 8
    e_new = SC.ess(new.reshape(2, -1))
    e_old = SC.ess(orig.reshape(2, -1))
    assert e_new > 1.5 * e_old, (e_new, e_old)


def test_initialisation_is_respected_and_trace_kept():
    d = synthetic(M=60, nu_s=3)
    p = NU.fit(d, "TG", nu_update="collapsed_slice", n_draws=10, burn_in=0, thin=1,
               nu_init=250.0, keep_trace=True)
    assert len(p.trace["nu"]) == 10 and np.isfinite(p.trace["sum_lam"]).all()
    with pytest.raises(ValueError):
        NU.fit(d, "TG", nu_update="bogus")


def test_gamma2_true_grid_weights_equal_exact_cell_masses():
    g = HT.NU_GRID
    u = np.log(g)
    edges = np.exp(np.r_[u[0], (u[:-1] + u[1:]) / 2, u[-1]])
    cell = np.diff(stats.gamma(2, scale=10).cdf(edges))
    cell /= cell.sum()
    w = np.exp(NU.log_prior_nu_gamma2_true(g))
    w /= w.sum()
    assert np.max(np.abs(w[1:-1] - cell[1:-1]) / cell[1:-1]) < 0.01
    assert w[0] + w[-1] < 2e-4                       # boundary half-cells: negligible mass


def test_slice_with_no_data_samples_the_truncated_gamma_prior():
    """With zero patients the collapsed likelihood is flat, so slice draws = prior."""
    rng = np.random.default_rng(8)
    z = np.empty(0)
    nu, draws = 10.0, []
    for _ in range(30000):
        nu = NU.draw_nu_collapsed_slice(nu, z, rng, NU.log_prior_nu_gamma2_true)
        draws.append(nu)
    G = stats.gamma(2, scale=10)
    lo, hi = G.cdf(1.0), G.cdf(300.0)
    cdf = lambda x: (G.cdf(x) - lo) / (hi - lo)             # noqa: E731
    thinned = np.array(draws[1000::10])
    assert stats.kstest(thinned, cdf).pvalue > 0.01
    assert thinned.min() >= 1.0 and thinned.max() <= 300.0
