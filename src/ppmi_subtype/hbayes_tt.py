"""Growth model with optional Student-t slopes and/or Student-t residuals.

Four configurations of one model (names: <slope><residual>):

    GG  b1i ~ N(0, tau1^2)              eps_it ~ N(0, sigma^2)        (= hbayes)
    TG  b1i ~ t_nu_s(0, tau1^2)         eps_it ~ N(0, sigma^2)        (= hbayes_t)
    GT  b1i ~ N(0, tau1^2)              eps_it ~ t_nu_e(0, sigma^2)
    TT  b1i ~ t_nu_s(0, tau1^2)         eps_it ~ t_nu_e(0, sigma^2)

always with R_it = (beta0 + b0i) + (beta1 + b1i) t_it + eps_it and
b0i ~ N(0, tau0^2). Student-t laws are scale mixtures:

    b1i | lambda_i ~ N(0, tau1^2 / lambda_i),   lambda_i ~ Gamma(nu_s/2, rate nu_s/2)
    eps_it | omega_it ~ N(0, sigma^2 / omega_it), omega_it ~ Gamma(nu_e/2, rate nu_e/2)

so tau1 and sigma are *scales* (SDs only when the matching nu is infinite).
Priors: N(0, 100^2) on beta; IG(0.01, 0.01) on sigma^2, tau0^2, tau1^2; each nu
~ Gamma(2, rate 0.1) on the log grid [1, 300] (400 points), sampled exactly
on the grid. `hbayes` and `hbayes_t` are not modified; with Gaussian
residuals this sampler consumes random numbers in the same order as they do
and reproduces their draws exactly (tested).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import hbayes
from .hbayes_t import NU_GRID, log_prior_nu_gamma, nu_conditional_logpdf, score_moments

CONFIGS = {"GG": (False, False), "TG": (True, False), "GT": (False, True), "TT": (True, True)}


@dataclass
class TTPosterior(hbayes.GrowthPosterior):
    config: str = "GG"
    nu: np.ndarray | None = None            # slope df (S,); inf if Gaussian slopes
    nu_e: np.ndarray | None = None          # residual df (S,); inf if Gaussian residuals


def fit(data: hbayes.GrowthData, config: str = "TT", n_draws: int = 4000,
        burn_in: int = 1000, thin: int = 2, seed: int = 42, prior_beta_sd: float = 100.0,
        prior_shape: float = 0.01, prior_rate: float = 0.01, nu_grid=NU_GRID,
        log_prior_nu=log_prior_nu_gamma, keep_random_effects: bool = False) -> TTPosterior:
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
    nu, nu_e = (10.0 if slope_t else np.inf), (10.0 if resid_t else np.inf)
    lam = np.ones(M)

    prior_beta_prec = np.eye(2) / prior_beta_sd**2
    keep = {k: [] for k in ("beta", "sigma2", "tau0_2", "tau1_2", "nu", "nu_e")}
    keep_b = [] if keep_random_effects else None
    for sweep in range(burn_in + n_draws * thin):
        # --- random effects ---------------------------------------------
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

        # --- population coefficients (weighted when residuals are t) ------
        fitted_b = np.einsum("ij,ij->i", X, b[idx])
        r = y - fitted_b
        Xw = X * omega[:, None]
        prec_beta = Xw.T @ X / sigma2 + prior_beta_prec
        V_beta = np.linalg.inv(prec_beta)
        m_beta = V_beta @ (Xw.T @ r) / sigma2
        beta = m_beta + np.linalg.cholesky(V_beta) @ rng.standard_normal(2)

        # --- variance components ----------------------------------------
        e = y - X @ beta - fitted_b
        sigma2 = 1.0 / rng.gamma(prior_shape + N / 2,
                                 1.0 / (prior_rate + 0.5 * (omega * e) @ e))
        tau0_2 = 1.0 / rng.gamma(prior_shape + M / 2,
                                 1.0 / (prior_rate + 0.5 * b[:, 0] @ b[:, 0]))
        tau1_2 = 1.0 / rng.gamma(prior_shape + M / 2,
                                 1.0 / (prior_rate + 0.5 * (lam * b[:, 1]) @ b[:, 1]))

        # --- slope mixing weights and df (same order as hbayes_t) --------
        if slope_t:
            lam = rng.gamma((nu + 1) / 2, 1.0 / ((nu + b[:, 1] ** 2 / tau1_2) / 2))
            lp = nu_conditional_logpdf(np.log(lam).sum(), lam.sum(), M, nu_grid, log_prior_nu)
            p = np.exp(lp - lp.max())
            nu = float(rng.choice(nu_grid, p=p / p.sum()))

        # --- residual mixing weights and df ------------------------------
        if resid_t:
            omega = rng.gamma((nu_e + 1) / 2, 1.0 / ((nu_e + e ** 2 / sigma2) / 2))
            lp = nu_conditional_logpdf(np.log(omega).sum(), omega.sum(), N, nu_grid,
                                       log_prior_nu)
            p = np.exp(lp - lp.max())
            nu_e = float(rng.choice(nu_grid, p=p / p.sum()))
            XtX = per_patient_xtx(omega)

        if sweep >= burn_in and (sweep - burn_in) % thin == 0:
            keep["beta"].append(beta.copy())
            keep["sigma2"].append(sigma2)
            keep["tau0_2"].append(tau0_2)
            keep["tau1_2"].append(tau1_2)
            keep["nu"].append(nu)
            keep["nu_e"].append(nu_e)
            if keep_b is not None:
                keep_b.append(b.copy())

    return TTPosterior(
        beta=np.array(keep["beta"]), sigma2=np.array(keep["sigma2"]),
        tau0_2=np.array(keep["tau0_2"]), tau1_2=np.array(keep["tau1_2"]),
        n_patients=M, n_obs=N, n_draws=len(keep["sigma2"]),
        random_effects=np.array(keep_b) if keep_b is not None else None,
        patients=data.patients, config=config, nu=np.array(keep["nu"]),
        nu_e=np.array(keep["nu_e"]))


# ---------------------------------------------------------------------------
#  Held-out prediction
# ---------------------------------------------------------------------------
def _conditional_b(Xh, y_hist, beta, sigma2, tau0_2, tau1_2, lam, w):
    """b | history, lambda, omega_hist for every draw. w: (S, n_hist) residual weights."""
    prec = np.einsum("sn,ni,nj->sij", w, Xh, Xh) / sigma2[:, None, None]
    prec[:, 0, 0] += 1.0 / tau0_2
    prec[:, 1, 1] += lam / tau1_2
    V = np.linalg.inv(prec)
    r = y_hist[None, :] - beta @ Xh.T
    m = np.einsum("sij,sj->si", V, ((w * r) @ Xh) / sigma2[:, None])
    return m, V


def predictive_moments(post: TTPosterior, t_hist, y_hist, t_target, picks, rng,
                       n_inner: int = 20, n_omega: int = 8):
    """Gaussian components of the next-ratio predictive, (S * K,) each.

    (b, lambda, omega_hist) | history, theta_s are sampled by `n_inner` Gibbs
    steps started from their priors; b is then integrated out analytically.
    With t residuals, the next visit's own weight omega_next is drawn
    `n_omega` times from its prior, giving K = n_omega Gaussian components
    per draw (a Monte-Carlo t-convolution). With Gaussian residuals K = 1 and
    the computation (including random-number use) equals hbayes_t's.
    """
    slope_t, resid_t = CONFIGS[post.config]
    t_hist = np.asarray(t_hist, float)
    y_hist = np.asarray(y_hist, float)
    Xh = np.column_stack([np.ones_like(t_hist), t_hist])
    beta, s2 = post.beta[picks], post.sigma2[picks]
    t0, t1 = post.tau0_2[picks], post.tau1_2[picks]
    S, n = len(picks), len(t_hist)
    nu, nu_e = post.nu[picks], post.nu_e[picks]
    lam = rng.gamma(nu / 2, 2.0 / nu) if slope_t else np.ones(S)
    w = rng.gamma(nu_e[:, None] / 2, 2.0 / nu_e[:, None], (S, n)) if resid_t else np.ones((S, n))
    if slope_t or resid_t:
        for _ in range(n_inner):
            m, V = _conditional_b(Xh, y_hist, beta, s2, t0, t1, lam, w)
            L = np.linalg.cholesky(V)
            b = m + np.einsum("sij,sj->si", L, rng.standard_normal((S, 2)))
            if slope_t:
                lam = rng.gamma((nu + 1) / 2, 1.0 / ((nu + b[:, 1] ** 2 / t1) / 2))
            if resid_t:
                e = y_hist[None, :] - (beta + b) @ Xh.T
                w = rng.gamma((nu_e[:, None] + 1) / 2,
                              1.0 / ((nu_e[:, None] + e ** 2 / s2[:, None]) / 2))
    m, V = _conditional_b(Xh, y_hist, beta, s2, t0, t1, lam, w)
    x = np.array([1.0, t_target])
    mean = beta @ x + m @ x
    var_b = np.einsum("i,sij,j->s", x, V, x)
    if not resid_t:
        return mean, var_b + s2
    w_next = rng.gamma(nu_e[:, None] / 2, 2.0 / nu_e[:, None], (S, n_omega))
    return (np.repeat(mean, n_omega),
            (var_b[:, None] + s2[:, None] / w_next).ravel())


def evaluate_models(posts: dict, panel, max_draws: int = 400, seed: int = 0,
                    n_inner: int = 20, n_omega: int = 8,
                    ratio_col: str = "ratio", time_col: str = "years_from_baseline"):
    """One-step-ahead held-out predictions for several fitted configurations.

    Every model is scored on exactly the same transitions -- those of
    `hbayes.evaluate`: each test patient, each visit k >= 1 with finite ratios,
    predicted from visits < k. Scoring is `hbayes_t.score_moments` (class
    probabilities at script 04's thresholds, and the next ratio's log density).
    """
    df = panel[panel[ratio_col].notna() & np.isfinite(panel[ratio_col])]
    df = df.sort_values(["PATNO", time_col])
    rng = np.random.default_rng(seed)
    picks = {k: np.linspace(0, p.n_draws - 1, min(p.n_draws, max_draws)).astype(int)
             for k, p in posts.items()}
    rows = []
    for patno, g in df.groupby("PATNO", sort=False):
        y = g[ratio_col].to_numpy(float)
        t = g[time_col].to_numpy(float)
        truth = g["subtype_name"].to_numpy() if "subtype_name" in g else [None] * len(y)
        for k in range(1, len(y)):
            for name, p in posts.items():
                mm, vv = predictive_moments(p, t[:k], y[:k], t[k], picks[name], rng,
                                            n_inner, n_omega)
                rows.append({"PATNO": patno, "step": k - 1, "model": name,
                             "observed": truth[k], "y_next": y[k],
                             **score_moments(mm, vv, y[k])})
    return pd.DataFrame(rows)
