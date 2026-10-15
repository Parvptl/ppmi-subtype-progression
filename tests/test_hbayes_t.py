"""Tests for the Student-t random-slope growth model.

Weight-bearing: with nu fixed at infinity the sampler is exactly hbayes; the
nu and lambda conditionals are the right densities; held-out prediction
targets p(b, lambda | history) and reduces to the Gaussian algebra; and the
sampler separates heavy-tailed from Gaussian slopes when the data allow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ppmi_subtype import hbayes
from ppmi_subtype import hbayes_t as HT


def synthetic(seed=0, M=300, nu=None, tau1=0.15, visits=6, sigma=0.3):
    rng = np.random.default_rng(seed)
    b0 = rng.normal(0, 0.8, M)
    b1 = tau1 * (rng.standard_t(nu, M) if nu else rng.standard_normal(M))
    rows = []
    for i in range(M):
        n = rng.integers(2, visits + 1)
        t = np.sort(rng.uniform(0, 8, n))
        y = 2.0 + b0[i] + (-0.1 + b1[i]) * t + rng.normal(0, sigma, n)
        rows += [{"PATNO": 1000 + 7 * i, "years_from_baseline": tt, "ratio": yy}
                 for tt, yy in zip(t, y)]
    return pd.DataFrame(rows), b1


def test_infinite_nu_reproduces_hbayes_exactly():
    panel, _ = synthetic(M=80)
    d = hbayes.build_growth_data(panel)
    g = hbayes.fit_growth_model(d, n_draws=200, burn_in=50, seed=7, keep_random_effects=True)
    t = HT.fit_t_growth_model(d, n_draws=200, burn_in=50, seed=7, fix_nu=np.inf,
                              keep_random_effects=True)
    for a in ("beta", "sigma2", "tau0_2", "tau1_2", "random_effects"):
        assert np.allclose(getattr(g, a), getattr(t, a), rtol=1e-9, atol=1e-12), a
    assert np.array_equal(t.patients, g.patients)


def test_nu_conditional_matches_gamma_likelihood():
    rng = np.random.default_rng(1)
    lam = rng.gamma(2.5, 1 / 2.5, 500)
    grid = HT.NU_GRID[::40]
    ours = HT.nu_conditional_logpdf(np.log(lam).sum(), lam.sum(), len(lam), grid,
                                    HT.log_prior_nu_loguniform)
    ref = np.array([stats.gamma.logpdf(lam, nu / 2, scale=2 / nu).sum() for nu in grid])
    assert np.allclose(ours - ours.max(), ref - ref.max(), atol=1e-6)


def test_lambda_scale_mixture_gives_student_t():
    """b | lam ~ N(0, tau^2/lam), lam ~ Gamma(nu/2, rate nu/2)  =>  b/tau ~ t_nu."""
    rng = np.random.default_rng(2)
    nu, tau = 4.0, 0.2
    lam = rng.gamma(nu / 2, 2 / nu, 200_000)
    b = rng.normal(0, tau / np.sqrt(lam))
    assert stats.kstest(b / tau, stats.t(nu).cdf).pvalue > 0.01


def test_gaussian_predictive_matches_hbayes_conditional():
    panel, _ = synthetic(M=60)
    post = hbayes.fit_growth_model(hbayes.build_growth_data(panel), n_draws=50, burn_in=20)
    th, yh = np.array([0.0, 1.0, 2.5]), np.array([2.1, 1.9, 1.7])
    picks = np.arange(post.n_draws)
    mean, var = HT.gaussian_predictive_moments(post, th, yh, 4.0, picks)
    Xh = np.column_stack([np.ones(3), th])
    x = np.array([1.0, 4.0])
    for s in (0, 17, 49):
        D_inv = np.diag([1 / post.tau0_2[s], 1 / post.tau1_2[s]])
        V = np.linalg.inv(Xh.T @ Xh / post.sigma2[s] + D_inv)
        m = V @ (Xh.T @ (yh - Xh @ post.beta[s])) / post.sigma2[s]
        assert mean[s] == pytest.approx(x @ (post.beta[s] + m))
        assert var[s] == pytest.approx(x @ V @ x + post.sigma2[s])


def test_t_predictive_targets_conditional_posterior():
    """Inner Gibbs agrees with importance sampling over lambda for one patient."""
    rng = np.random.default_rng(3)
    S = 1
    post = HT.TPosterior(beta=np.array([[2.0, -0.1]]), sigma2=np.array([0.09]),
                         tau0_2=np.array([0.5]), tau1_2=np.array([0.01]), n_patients=1,
                         n_obs=1, n_draws=S, nu=np.array([3.0]))
    th = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    yh = 2.0 + (-0.1 + 0.4) * th                      # a patient far in the slope tail
    picks = np.zeros(20000, int)
    mean, var = HT.predictive_moments(post, th, yh, 6.0, picks, rng, n_inner=30)
    # reference: lambda ~ prior, weight by marginal likelihood of the history
    lam = rng.gamma(1.5, 2 / 3.0, 200000)
    Xh = np.column_stack([np.ones(5), th])
    x = np.array([1.0, 6.0])
    logw, mus = [], []
    for lv in lam[:20000]:
        D = np.diag([0.5, 0.01 / lv])
        C = Xh @ D @ Xh.T + 0.09 * np.eye(5)
        logw.append(stats.multivariate_normal.logpdf(yh, Xh @ [2.0, -0.1], C))
        V = np.linalg.inv(Xh.T @ Xh / 0.09 + np.linalg.inv(D))
        mus.append(x @ ([2.0, -0.1] + V @ (Xh.T @ (yh - Xh @ [2.0, -0.1])) / 0.09))
    w = np.exp(np.array(logw) - max(logw))
    ref = np.sum(w * np.array(mus)) / w.sum()
    assert mean.mean() == pytest.approx(ref, abs=0.02)
    # the Gaussian model would shrink this patient much harder
    g = HT.TPosterior(beta=post.beta, sigma2=post.sigma2, tau0_2=post.tau0_2,
                      tau1_2=post.tau1_2, n_patients=1, n_obs=1, n_draws=1, nu=post.nu)
    mg, _ = HT.gaussian_predictive_moments(g, th, yh, 6.0, np.array([0]))
    assert mean.mean() > mg[0] + 0.05


def test_large_nu_predictive_approaches_gaussian():
    panel, _ = synthetic(M=60)
    post = hbayes.fit_growth_model(hbayes.build_growth_data(panel), n_draws=40, burn_in=20)
    tpost = HT.TPosterior(beta=post.beta, sigma2=post.sigma2, tau0_2=post.tau0_2,
                          tau1_2=post.tau1_2, n_patients=0, n_obs=0, n_draws=post.n_draws,
                          nu=np.full(post.n_draws, 1e7))
    picks = np.arange(post.n_draws)
    th, yh = np.array([0.0, 1.5]), np.array([2.0, 2.4])
    mt, vt = HT.predictive_moments(tpost, th, yh, 3.0, picks, np.random.default_rng(0))
    mg, vg = HT.gaussian_predictive_moments(post, th, yh, 3.0, picks)
    assert np.allclose(mt, mg, atol=1e-3) and np.allclose(vt, vg, rtol=1e-3)


def test_score_moments_probabilities_and_density():
    s = HT.score_moments(np.array([1.0]), np.array([0.04]), 1.1)
    assert s["TD"] + s["Indeterminate"] + s["PIGD"] == pytest.approx(1.0)
    assert s["TD"] == pytest.approx(stats.norm.sf(1.15, 1.0, 0.2))
    assert s["log_density"] == pytest.approx(stats.norm.logpdf(1.1, 1.0, 0.2))


@pytest.mark.parametrize("nu_true,expect_heavy", [(2.5, True), (None, False)])
def test_recovers_tail_weight_when_identified(nu_true, expect_heavy):
    panel, _ = synthetic(seed=5, M=600, nu=nu_true, visits=10, sigma=0.15)
    d = hbayes.build_growth_data(panel)
    post = HT.fit_t_growth_model(d, n_draws=1500, burn_in=500, seed=1)
    med = np.median(post.nu)
    if expect_heavy:
        assert med < 8 and np.quantile(post.nu, .975) < 20
    else:
        assert med > 15
    assert post.beta[:, 1].mean() == pytest.approx(-0.1, abs=0.03)


def test_summary_includes_nu():
    panel, _ = synthetic(M=40)
    post = HT.fit_t_growth_model(hbayes.build_growth_data(panel), n_draws=30, burn_in=10)
    s = post.summary()
    assert s.parameter.str.startswith("nu").any() and len(post.nu) == 30
