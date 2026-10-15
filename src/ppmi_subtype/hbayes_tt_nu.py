"""Alternative updates for the slope degrees of freedom nu_s in the TT/TG models.

Same model as `hbayes_tt` (which is left untouched because the frozen full run
hashed it). Only the *sampling* of nu_s changes; the posterior is the same.

Why. In `hbayes_tt` nu_s is drawn from p(nu | lambda_1..M), exactly on the grid.
That step is correct, but with M ~ 800 mixing weights the conditional is very
narrow, so nu can only move as fast as the lambdas drift: a data-augmentation
bottleneck (Liu 1995; van Dyk & Meng 2001). Slice-sampling the *same*
conditional would not help.

Fix: update (nu, lambda) as one block given (b1, tau1^2),

    nu      ~ p(nu | b1, tau1)  =  p(nu) * prod_i t_nu(b1i / tau1) / tau1
    lambda  ~ p(lambda | nu, b1, tau1)  (the usual Gamma conditional),

which is p(nu, lambda | b1, tau1) = p(nu | b1, tau1) p(lambda | nu, b1, tau1):
an exact blocked Gibbs step with lambda integrated out of the nu update.
The collapsed log-likelihood, z_i = b1i / tau1, is

    l(nu) = M [lnG((nu+1)/2) - lnG(nu/2) - 0.5 ln(nu pi)]
            - (nu+1)/2 * sum_i ln(1 + z_i^2 / nu).

Two ways to draw nu from p(nu | b1, tau1):
  * "collapsed_grid": exactly on the same 400-point log grid with the same
    prior weights as hbayes_tt, so the model (including the prior) is unchanged;
  * "collapsed_slice": slice sampling (Neal 2003, stepping out + shrinkage) on
    u = ln nu over [ln 1, ln 300]. The grid weights w(nu_k) define the prior's
    density *in u* (log-spaced cells have equal width in u), so the slice
    target is l(e^u) + log_prior(e^u) with the same function -- the continuous
    analogue of the grid prior.

The original update ("lambda_grid") is kept for comparison and reproduces
`hbayes_tt.fit` exactly.

Note on the prior. `log_prior_nu_gamma` evaluates the Gamma(2, 0.1) density
at log-spaced grid points; since cell widths are proportional to nu, the
implied prior on nu is p(nu) ~ Gamma(2,0.1)(nu) / nu ~ exp(-0.1 nu), i.e.
Exponential(0.1) truncated to [1, 300] (mean 11.0), not Gamma(2, 0.1) (mean
20.1). `log_prior_nu_gamma2_true` adds the log-Jacobian ln(nu) so that the grid
and slice versions implement a true Gamma(2, 0.1) prior.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from . import hbayes
from .hbayes_t import NU_GRID, log_prior_nu_gamma, nu_conditional_logpdf
from .hbayes_tt import CONFIGS, TTPosterior

U_LO, U_HI = np.log(1.0), np.log(300.0)


def log_prior_nu_gamma2_true(nu):
    """Grid weights / u-space density giving a true Gamma(2, rate 0.1) prior on nu."""
    return log_prior_nu_gamma(nu) + np.log(nu)


def collapsed_loglik(nu, z):
    """sum_i log t_nu(z_i) (standard t), vectorised over nu."""
    nu = np.atleast_1d(np.asarray(nu, float))[:, None]
    M = z.shape[0]
    s = np.log1p(z[None, :] ** 2 / nu).sum(axis=1)
    out = M * (gammaln((nu[:, 0] + 1) / 2) - gammaln(nu[:, 0] / 2)
               - 0.5 * np.log(nu[:, 0] * np.pi)) - (nu[:, 0] + 1) / 2 * s
    return out


def draw_nu_collapsed_grid(z, rng, grid=NU_GRID, log_prior=log_prior_nu_gamma):
    lp = collapsed_loglik(grid, z) + log_prior(grid)
    p = np.exp(lp - lp.max())
    return float(rng.choice(grid, p=p / p.sum()))


def slice_u(u0, logf, rng, w=0.5, max_steps=50, lo=U_LO, hi=U_HI):
    """One univariate slice-sampling update (stepping out, shrinkage) on [lo, hi]."""
    y = logf(u0) - rng.exponential()
    L = u0 - w * rng.uniform()
    R = L + w
    j = int(rng.integers(max_steps))
    k = max_steps - 1 - j
    while j > 0 and L > lo and logf(L) > y:
        L -= w
        j -= 1
    while k > 0 and R < hi and logf(R) > y:
        R += w
        k -= 1
    L, R = max(L, lo), min(R, hi)
    while True:
        u1 = rng.uniform(L, R)
        if logf(u1) > y:
            return u1
        if u1 < u0:
            L = u1
        else:
            R = u1


def draw_nu_collapsed_slice(nu0, z, rng, log_prior=log_prior_nu_gamma):
    def logf(u):
        nu = np.exp(u)
        return float(collapsed_loglik(nu, z)[0] + log_prior(np.array([nu]))[0])
    return float(np.exp(slice_u(np.log(nu0), logf, rng)))


def fit(data: hbayes.GrowthData, config: str = "TT", nu_update: str = "collapsed_grid",
        n_draws: int = 4000, burn_in: int = 1000, thin: int = 2, seed: int = 42,
        prior_beta_sd: float = 100.0, prior_shape: float = 0.01, prior_rate: float = 0.01,
        nu_grid=NU_GRID, log_prior_nu=log_prior_nu_gamma, nu_init: float = 10.0,
        keep_random_effects: bool = False, keep_trace: bool = False):
    """`hbayes_tt.fit` with a choice of nu_s update and initial nu_s.

    nu_update: "lambda_grid" (original), "collapsed_grid", "collapsed_slice".
    keep_trace: also return every sweep's nu_s (including burn-in) and the
    sufficient statistics of p(nu | lambda) for diagnosing the original step.
    """
    if nu_update not in ("lambda_grid", "collapsed_grid", "collapsed_slice"):
        raise ValueError(nu_update)
    slope_t, resid_t = CONFIGS[config]
    rng = np.random.default_rng(seed)
    X = data.design()
    y, idx, M, N = data.y, data.patient_index, data.n_patients, data.n_obs

    def per_patient_xtx(w):
        A = np.zeros((M, 2, 2))
        for k in range(2):
            for l in range(2):
                A[:, k, l] = np.bincount(idx, weights=w * X[:, k] * X[:, l], minlength=M)
        return A

    omega = np.ones(N)
    XtX = per_patient_xtx(omega)
    beta = np.array([np.mean(y), 0.0])
    b = np.zeros((M, 2))
    sigma2 = float(np.var(y))
    tau0_2, tau1_2 = 0.1 * sigma2, 0.01 * sigma2
    nu, nu_e = (float(nu_init) if slope_t else np.inf), (10.0 if resid_t else np.inf)
    lam = np.ones(M)

    prior_beta_prec = np.eye(2) / prior_beta_sd**2
    keep = {k: [] for k in ("beta", "sigma2", "tau0_2", "tau1_2", "nu", "nu_e")}
    keep_b = [] if keep_random_effects else None
    trace = {"nu": [], "sum_log_lam": [], "sum_lam": [], "tau1": []} if keep_trace else None
    for sweep in range(burn_in + n_draws * thin):
        resid = y - X @ beta
        Xtr = np.column_stack(
            [np.bincount(idx, weights=omega * X[:, k] * resid, minlength=M) for k in range(2)])
        prec = XtX / sigma2
        prec[:, 0, 0] += 1.0 / tau0_2
        prec[:, 1, 1] += lam / tau1_2
        V = np.linalg.inv(prec)
        m = np.einsum("mij,mj->mi", V, Xtr / sigma2)
        L = np.linalg.cholesky(V)
        b = m + np.einsum("mij,mj->mi", L, rng.standard_normal((M, 2)))

        fitted_b = np.einsum("ij,ij->i", X, b[idx])
        r = y - fitted_b
        Xw = X * omega[:, None]
        prec_beta = Xw.T @ X / sigma2 + prior_beta_prec
        V_beta = np.linalg.inv(prec_beta)
        m_beta = V_beta @ (Xw.T @ r) / sigma2
        beta = m_beta + np.linalg.cholesky(V_beta) @ rng.standard_normal(2)

        e = y - X @ beta - fitted_b
        sigma2 = 1.0 / rng.gamma(prior_shape + N / 2,
                                 1.0 / (prior_rate + 0.5 * (omega * e) @ e))
        tau0_2 = 1.0 / rng.gamma(prior_shape + M / 2,
                                 1.0 / (prior_rate + 0.5 * b[:, 0] @ b[:, 0]))
        tau1_2 = 1.0 / rng.gamma(prior_shape + M / 2,
                                 1.0 / (prior_rate + 0.5 * (lam * b[:, 1]) @ b[:, 1]))

        if slope_t:
            if nu_update == "lambda_grid":                      # original order
                lam = rng.gamma((nu + 1) / 2, 1.0 / ((nu + b[:, 1] ** 2 / tau1_2) / 2))
                lp = nu_conditional_logpdf(np.log(lam).sum(), lam.sum(), M, nu_grid,
                                           log_prior_nu)
                p = np.exp(lp - lp.max())
                nu = float(rng.choice(nu_grid, p=p / p.sum()))
            else:                                               # block (nu, lambda) | b1, tau1
                z = b[:, 1] / np.sqrt(tau1_2)
                if nu_update == "collapsed_grid":
                    nu = draw_nu_collapsed_grid(z, rng, nu_grid, log_prior_nu)
                else:
                    nu = draw_nu_collapsed_slice(nu, z, rng, log_prior_nu)
                lam = rng.gamma((nu + 1) / 2, 1.0 / ((nu + b[:, 1] ** 2 / tau1_2) / 2))

        if resid_t:                                             # unchanged from hbayes_tt
            omega = rng.gamma((nu_e + 1) / 2, 1.0 / ((nu_e + e ** 2 / sigma2) / 2))
            lp = nu_conditional_logpdf(np.log(omega).sum(), omega.sum(), N, nu_grid,
                                       log_prior_nu)
            p = np.exp(lp - lp.max())
            nu_e = float(rng.choice(nu_grid, p=p / p.sum()))
            XtX = per_patient_xtx(omega)

        if trace is not None:
            trace["nu"].append(nu)
            trace["sum_log_lam"].append(float(np.log(lam).sum()))
            trace["sum_lam"].append(float(lam.sum()))
            trace["tau1"].append(float(np.sqrt(tau1_2)))
        if sweep >= burn_in and (sweep - burn_in) % thin == 0:
            keep["beta"].append(beta.copy())
            keep["sigma2"].append(sigma2)
            keep["tau0_2"].append(tau0_2)
            keep["tau1_2"].append(tau1_2)
            keep["nu"].append(nu)
            keep["nu_e"].append(nu_e)
            if keep_b is not None:
                keep_b.append(b.copy())

    post = TTPosterior(
        beta=np.array(keep["beta"]), sigma2=np.array(keep["sigma2"]),
        tau0_2=np.array(keep["tau0_2"]), tau1_2=np.array(keep["tau1_2"]),
        n_patients=M, n_obs=N, n_draws=len(keep["sigma2"]),
        random_effects=np.array(keep_b) if keep_b is not None else None,
        patients=data.patients, config=config, nu=np.array(keep["nu"]),
        nu_e=np.array(keep["nu_e"]))
    if trace is not None:
        post.trace = {k: np.array(v) for k, v in trace.items()}
    return post
