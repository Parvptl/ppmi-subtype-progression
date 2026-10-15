"""Tests for the four slope/residual configurations.

Weight-bearing: GG and TG reproduce hbayes and hbayes_t draw for draw (so the
baselines are untouched), the t-residual blocks are the right conditionals,
held-out prediction reduces correctly and targets p(b, lambda, omega | hist),
and on an identified design the sampler attributes tails to the right source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ppmi_subtype import hbayes
from ppmi_subtype import hbayes_t as HT
from ppmi_subtype import hbayes_tt as TT


def synthetic(seed=0, M=300, nu_s=None, nu_e=None, tau1=0.15, visits=6, sigma=0.3):
    rng = np.random.default_rng(seed)
    b0 = rng.normal(0, 0.8, M)
    b1 = tau1 * (rng.standard_t(nu_s, M) if nu_s else rng.standard_normal(M))
    rows = []
    for i in range(M):
        n = rng.integers(2, visits + 1)
        t = np.sort(rng.uniform(0, 8, n))
        e = sigma * (rng.standard_t(nu_e, n) if nu_e else rng.standard_normal(n))
        y = 2.0 + b0[i] + (-0.1 + b1[i]) * t + e
        rows += [{"PATNO": 500 + 3 * i, "years_from_baseline": tt, "ratio": yy}
                 for tt, yy in zip(t, y)]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def small():
    return hbayes.build_growth_data(synthetic(M=80))


def test_GG_reproduces_hbayes_exactly(small):
    g = hbayes.fit_growth_model(small, n_draws=150, burn_in=50, seed=4, keep_random_effects=True)
    t = TT.fit(small, "GG", n_draws=150, burn_in=50, seed=4, keep_random_effects=True)
    for a in ("beta", "sigma2", "tau0_2", "tau1_2", "random_effects"):
        assert np.allclose(getattr(g, a), getattr(t, a), rtol=1e-9, atol=1e-12), a
    assert np.isinf(t.nu).all() and np.isinf(t.nu_e).all()


def test_TG_reproduces_hbayes_t_exactly(small):
    a = HT.fit_t_growth_model(small, n_draws=150, burn_in=50, seed=5, keep_random_effects=True)
    b = TT.fit(small, "TG", n_draws=150, burn_in=50, seed=5, keep_random_effects=True)
    for k in ("beta", "sigma2", "tau0_2", "tau1_2", "random_effects", "nu"):
        assert np.allclose(getattr(a, k), getattr(b, k), rtol=1e-9, atol=1e-12), k


def test_omega_conditional_gives_t_residuals():
    """eps | omega ~ N(0, s^2/omega), omega ~ Gamma(nu/2, rate nu/2)  =>  eps/s ~ t_nu,
    and the conditional Gamma((nu+1)/2, (nu + e^2/s^2)/2) is that posterior."""
    rng = np.random.default_rng(1)
    nu, s = 3.0, 0.4
    om = rng.gamma(nu / 2, 2 / nu, 200_000)
    e = rng.normal(0, s / np.sqrt(om))
    assert stats.kstest(e / s, stats.t(nu).cdf).pvalue > 0.01
    # posterior of omega given e = 1.0 by importance weighting vs the closed form
    w = stats.norm.pdf(1.0, 0, s / np.sqrt(om))
    closed = (nu + 1) / (nu + (1.0 / s) ** 2)
    assert np.sum(w * om) / w.sum() == pytest.approx(closed, rel=0.03)


def _post(config, nu=3.0, nu_e=3.0, S=1):
    return TT.TTPosterior(beta=np.tile([2.0, -0.1], (S, 1)), sigma2=np.full(S, 0.09),
                          tau0_2=np.full(S, 0.5), tau1_2=np.full(S, 0.01), n_patients=1,
                          n_obs=1, n_draws=S, config=config,
                          nu=np.full(S, nu if config[0] == "T" else np.inf),
                          nu_e=np.full(S, nu_e if config[1] == "T" else np.inf))


def test_predictive_GG_TG_match_existing_code():
    th, yh = np.array([0.0, 1.0, 2.5]), np.array([2.1, 1.9, 1.7])
    picks = np.zeros(50, int)
    mg, vg = TT.predictive_moments(_post("GG"), th, yh, 4.0, picks, np.random.default_rng(0))
    rg, rv = HT.gaussian_predictive_moments(_post("GG"), th, yh, 4.0, picks)
    assert np.allclose(mg, rg) and np.allclose(vg, rv)
    p = _post("TG")
    a = TT.predictive_moments(p, th, yh, 4.0, picks, np.random.default_rng(9))
    tp = HT.TPosterior(beta=p.beta, sigma2=p.sigma2, tau0_2=p.tau0_2, tau1_2=p.tau1_2,
                       n_patients=1, n_obs=1, n_draws=1, nu=p.nu)
    b = HT.predictive_moments(tp, th, yh, 4.0, picks, np.random.default_rng(9))
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_predictive_t_residual_downweights_outlying_history_visit():
    """One wild visit: GT should follow it far less than GG."""
    th = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    yh = 2.0 - 0.1 * th
    yh[-1] += 3.0                                         # a single outlying ratio
    picks = np.zeros(4000, int)
    rng = np.random.default_rng(2)
    mg, _ = TT.predictive_moments(_post("GG"), th, yh, 5.0, picks[:1], rng)
    mt, vt = TT.predictive_moments(_post("GT"), th, yh, 5.0, picks, rng, n_inner=30)
    truth = 2.0 - 0.1 * 5.0
    assert abs(mt.mean() - truth) < 0.5 * abs(mg[0] - truth)
    assert len(mt) == 4000 * 8 and np.all(vt > 0)


def test_predictive_GT_targets_conditional_posterior():
    """Inner Gibbs vs importance sampling over omega_hist (b integrated)."""
    rng = np.random.default_rng(3)
    th = np.array([0.0, 1.0, 2.0, 3.0])
    yh = np.array([2.0, 1.9, 3.4, 1.7])
    picks = np.zeros(20000, int)
    m, _ = TT.predictive_moments(_post("GT"), th, yh, 5.0, picks, rng, n_inner=30)
    Xh = np.column_stack([np.ones(4), th])
    x = np.array([1.0, 5.0])
    om = rng.gamma(1.5, 2 / 3.0, (20000, 4))
    D = np.diag([0.5, 0.01])
    logw, mus = [], []
    for o in om:
        C = Xh @ D @ Xh.T + np.diag(0.09 / o)
        logw.append(stats.multivariate_normal.logpdf(yh, Xh @ [2.0, -0.1], C))
        W = np.diag(o)
        V = np.linalg.inv(Xh.T @ W @ Xh / 0.09 + np.linalg.inv(D))
        mus.append(x @ ([2.0, -0.1] + V @ (Xh.T @ W @ (yh - Xh @ [2.0, -0.1])) / 0.09))
    w = np.exp(np.array(logw) - max(logw))
    assert m.mean() == pytest.approx(np.sum(w * np.array(mus)) / w.sum(), abs=0.02)


@pytest.mark.parametrize("nu_s,nu_e", [(2.5, None), (None, 2.5), (2.5, 2.5)])
def test_TT_attributes_tails_on_identified_design(nu_s, nu_e):
    d = hbayes.build_growth_data(synthetic(seed=7, M=500, nu_s=nu_s, nu_e=nu_e, visits=10,
                                           sigma=0.15))
    post = TT.fit(d, "TT", n_draws=1500, burn_in=1000, thin=2, seed=1)
    ms, me = np.median(post.nu), np.median(post.nu_e)
    assert (ms < 8) == (nu_s is not None), ms
    assert (me < 8) == (nu_e is not None), me
    assert post.beta[:, 1].mean() == pytest.approx(-0.1, abs=0.03)
