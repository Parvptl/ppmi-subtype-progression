"""Growth model with a Student-t patient slope (heavy-tailed alternative to hbayes).

Same model as `hbayes.fit_growth_model` except for the random-slope law:

    R_it = (beta0 + b0i) + (beta1 + b1i) t_it + eps_it,   eps ~ N(0, sigma^2)
    b0i  ~ N(0, tau0^2)
    b1i  ~ t_nu(0, tau1^2)          (location 0, scale tau1, nu degrees of freedom)

written as a scale mixture so that every block stays conjugate:

    b1i | lambda_i ~ N(0, tau1^2 / lambda_i),   lambda_i ~ Gamma(nu/2, rate nu/2).

Priors match hbayes (N(0, 100^2) on beta, IG(0.01, 0.01) on sigma^2, tau0^2,
tau1^2). nu has a Gamma(2, rate 0.1) prior (Juarez & Steel 2010), restricted to
a log-spaced grid on [1, 300] and sampled exactly on that grid ("griddy Gibbs").
For nu -> infinity every lambda_i -> 1 and the model reduces to hbayes.

`hbayes` itself is not modified; prediction and scoring reuse its functions
wherever the Gaussian algebra is shared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm

from . import hbayes

NU_GRID = np.exp(np.linspace(np.log(1.0), np.log(300.0), 400))


def log_prior_nu_gamma(nu, shape=2.0, rate=0.1):
    return (shape - 1) * np.log(nu) - rate * nu


def log_prior_nu_loguniform(nu):
    return np.zeros_like(nu)


@dataclass
class TPosterior(hbayes.GrowthPosterior):
    nu: np.ndarray | None = None              # (S,)
    lam: np.ndarray | None = None             # (S, M) mixing weights, if retained

    def summary(self) -> pd.DataFrame:
        df = super().summary()
        v = self.nu
        df.loc[len(df)] = {"parameter": "nu (slope degrees of freedom)",
                           "mean": float(v.mean()), "sd": float(v.std()),
                           "q2.5": float(np.quantile(v, .025)), "q50": float(np.median(v)),
                           "q97.5": float(np.quantile(v, .975))}
        return df


def nu_conditional_logpdf(sum_log_lam, sum_lam, M, grid=NU_GRID, log_prior=log_prior_nu_gamma):
    """log p(nu | lambda) on the grid, unnormalised."""
    h = grid / 2
    return M * (h * np.log(h) - gammaln(h)) + (h - 1) * sum_log_lam - h * sum_lam \
        + log_prior(grid)


def fit_t_growth_model(
    data: hbayes.GrowthData,
    n_draws: int = 4000,
    burn_in: int = 1000,
    thin: int = 2,
    seed: int = 42,
    prior_beta_sd: float = 100.0,
    prior_shape: float = 0.01,
    prior_rate: float = 0.01,
    nu_grid: np.ndarray = NU_GRID,
    log_prior_nu=log_prior_nu_gamma,
    fix_nu: float | None = None,
    keep_random_effects: bool = False,
    keep_lambda: bool = False,
) -> TPosterior:
    """Gibbs sampler; blocks b | beta1-free updates identical to hbayes except D_i.

    `fix_nu` holds nu at a value (used by tests; `fix_nu=np.inf` sets every
    lambda to 1 and gives exactly the hbayes sampler).
    """
    rng = np.random.default_rng(seed)
    X = data.design()
    y, idx, M, N = data.y, data.patient_index, data.n_patients, data.n_obs

    XtX = np.zeros((M, 2, 2))
    for k in range(2):
        for l in range(2):
            XtX[:, k, l] = np.bincount(idx, weights=X[:, k] * X[:, l], minlength=M)

    beta = np.array([np.mean(y), 0.0])
    b = np.zeros((M, 2))
    sigma2 = float(np.var(y))
    tau0_2, tau1_2 = 0.1 * sigma2, 0.01 * sigma2
    gaussian = fix_nu is not None and np.isinf(fix_nu)
    nu = float(fix_nu) if fix_nu is not None else 10.0
    lam = np.ones(M)

    prior_beta_prec = np.eye(2) / prior_beta_sd**2
    keep = {k: [] for k in ("beta", "sigma2", "tau0_2", "tau1_2", "nu")}
    keep_b = [] if keep_random_effects else None
    keep_l = [] if keep_lambda else None
    total = burn_in + n_draws * thin

    for sweep in range(total):
        # --- random effects: per-patient prior precision diag(1/tau0^2, lam_i/tau1^2)
        resid = y - X @ beta
        Xtr = np.column_stack(
            [np.bincount(idx, weights=X[:, k] * resid, minlength=M) for k in range(2)]
        )
        prec = XtX / sigma2
        prec[:, 0, 0] += 1.0 / tau0_2
        prec[:, 1, 1] += lam / tau1_2
        V = np.linalg.inv(prec)
        m = np.einsum("mij,mj->mi", V, Xtr / sigma2)
        L = np.linalg.cholesky(V)
        b = m + np.einsum("mij,mj->mi", L, rng.standard_normal((M, 2)))

        # --- population coefficients (unchanged) -------------------------
        fitted_b = np.einsum("ij,ij->i", X, b[idx])
        r = y - fitted_b
        prec_beta = X.T @ X / sigma2 + prior_beta_prec
        V_beta = np.linalg.inv(prec_beta)
        m_beta = V_beta @ (X.T @ r) / sigma2
        beta = m_beta + np.linalg.cholesky(V_beta) @ rng.standard_normal(2)

        # --- variance components -----------------------------------------
        e = y - X @ beta - fitted_b
        sigma2 = 1.0 / rng.gamma(prior_shape + N / 2, 1.0 / (prior_rate + 0.5 * e @ e))
        tau0_2 = 1.0 / rng.gamma(
            prior_shape + M / 2, 1.0 / (prior_rate + 0.5 * b[:, 0] @ b[:, 0])
        )
        tau1_2 = 1.0 / rng.gamma(
            prior_shape + M / 2, 1.0 / (prior_rate + 0.5 * (lam * b[:, 1]) @ b[:, 1])
        )

        # --- mixing weights and degrees of freedom -----------------------
        if not gaussian:
            lam = rng.gamma((nu + 1) / 2, 1.0 / ((nu + b[:, 1] ** 2 / tau1_2) / 2))
            if fix_nu is None:
                lp = nu_conditional_logpdf(np.log(lam).sum(), lam.sum(), M, nu_grid,
                                           log_prior_nu)
                p = np.exp(lp - lp.max())
                nu = float(rng.choice(nu_grid, p=p / p.sum()))

        if sweep >= burn_in and (sweep - burn_in) % thin == 0:
            keep["beta"].append(beta.copy())
            keep["sigma2"].append(sigma2)
            keep["tau0_2"].append(tau0_2)
            keep["tau1_2"].append(tau1_2)
            keep["nu"].append(nu)
            if keep_b is not None:
                keep_b.append(b.copy())
            if keep_l is not None:
                keep_l.append(lam.copy())

    return TPosterior(
        beta=np.array(keep["beta"]), sigma2=np.array(keep["sigma2"]),
        tau0_2=np.array(keep["tau0_2"]), tau1_2=np.array(keep["tau1_2"]),
        n_patients=M, n_obs=N, n_draws=len(keep["sigma2"]),
        random_effects=np.array(keep_b) if keep_b is not None else None,
        patients=data.patients, nu=np.array(keep["nu"]),
        lam=np.array(keep_l) if keep_l is not None else None,
    )


# ---------------------------------------------------------------------------
#  Prediction for a held-out patient
# ---------------------------------------------------------------------------
def _conditional_b(Xh, y_hist, beta, sigma2, tau0_2, tau1_2, lam):
    """Mean and covariance of b | history, lambda (vectorised over draws)."""
    S = len(sigma2)
    prec = np.broadcast_to(Xh.T @ Xh, (S, 2, 2)) / sigma2[:, None, None]
    prec = prec.copy()
    prec[:, 0, 0] += 1.0 / tau0_2
    prec[:, 1, 1] += lam / tau1_2
    V = np.linalg.inv(prec)
    r = y_hist[None, :] - beta @ Xh.T                       # (S, n_hist)
    m = np.einsum("sij,sj->si", V, (r @ Xh) / sigma2[:, None])
    return m, V


def predictive_moments(post: TPosterior, t_hist, y_hist, t_target, picks, rng,
                       n_inner: int = 20):
    """Gaussian predictive moments of the next ratio, one per posterior draw.

    For draw s, the held-out patient's (b, lambda) are sampled from
    p(b, lambda | history, theta_s) by `n_inner` Gibbs steps started at
    lambda ~ prior; the returned moments are those of y_next | lambda, history,
    theta_s with b integrated out analytically (Rao-Blackwellised).
    """
    t_hist = np.asarray(t_hist, float)
    y_hist = np.asarray(y_hist, float)
    Xh = np.column_stack([np.ones_like(t_hist), t_hist])
    beta, s2 = post.beta[picks], post.sigma2[picks]
    t0, t1, nu = post.tau0_2[picks], post.tau1_2[picks], post.nu[picks]
    S = len(picks)
    lam = rng.gamma(nu / 2, 2.0 / nu)
    for _ in range(n_inner):
        m, V = _conditional_b(Xh, y_hist, beta, s2, t0, t1, lam)
        L = np.linalg.cholesky(V)
        b = m + np.einsum("sij,sj->si", L, rng.standard_normal((S, 2)))
        lam = rng.gamma((nu + 1) / 2, 1.0 / ((nu + b[:, 1] ** 2 / t1) / 2))
    m, V = _conditional_b(Xh, y_hist, beta, s2, t0, t1, lam)
    x = np.array([1.0, t_target])
    mean = beta @ x + m @ x
    var = np.einsum("i,sij,j->s", x, V, x) + s2
    return mean, var


def gaussian_predictive_moments(post: hbayes.GrowthPosterior, t_hist, y_hist, t_target,
                                picks):
    """Same moments for the Gaussian baseline (exact, no inner sampling)."""
    t_hist = np.asarray(t_hist, float)
    Xh = np.column_stack([np.ones_like(t_hist), t_hist])
    m, V = _conditional_b(Xh, np.asarray(y_hist, float), post.beta[picks],
                          post.sigma2[picks], post.tau0_2[picks], post.tau1_2[picks],
                          np.ones(len(picks)))
    x = np.array([1.0, t_target])
    return post.beta[picks] @ x + m @ x, np.einsum("i,sij,j->s", x, V, x) + post.sigma2[picks]


def score_moments(mean, var, y_next):
    """Class probabilities (script-04 thresholds) and log density of the next ratio.

    Both are exact mixtures over posterior draws of Gaussian predictive laws,
    so no Monte-Carlo noise is added at the observation level.
    """
    sd = np.sqrt(var)
    p_pigd = norm.cdf((hbayes.PIGD_THRESHOLD - mean) / sd).mean()
    p_td = norm.sf((hbayes.TD_THRESHOLD - mean) / sd).mean()
    dens = norm.pdf(y_next, mean, sd).mean()
    return {"TD": float(p_td), "Indeterminate": float(1 - p_td - p_pigd),
            "PIGD": float(p_pigd), "log_density": float(np.log(max(dens, 1e-300)))}


def evaluate_both(post_g, post_t, panel, max_draws=400, seed=0, n_inner=20,
                  ratio_col="ratio", time_col="years_from_baseline"):
    """One-step-ahead held-out predictions for both models on identical transitions.

    Transitions and draw selection follow `hbayes.evaluate`: every test patient,
    every visit k >= 1 with finite ratios, predicted from visits < k.
    """
    df = panel[panel[ratio_col].notna() & np.isfinite(panel[ratio_col])]
    df = df.sort_values(["PATNO", time_col])
    rng = np.random.default_rng(seed)
    pick_g = np.linspace(0, post_g.n_draws - 1, min(post_g.n_draws, max_draws)).astype(int)
    pick_t = np.linspace(0, post_t.n_draws - 1, min(post_t.n_draws, max_draws)).astype(int)
    rows = []
    for patno, g in df.groupby("PATNO", sort=False):
        y = g[ratio_col].to_numpy(float)
        t = g[time_col].to_numpy(float)
        truth = g["subtype_name"].to_numpy()
        for k in range(1, len(y)):
            mg, vg = gaussian_predictive_moments(post_g, t[:k], y[:k], t[k], pick_g)
            mt, vt = predictive_moments(post_t, t[:k], y[:k], t[k], pick_t, rng, n_inner)
            for model, (mm, vv) in (("gaussian", (mg, vg)), ("student_t", (mt, vt))):
                rows.append({"PATNO": patno, "step": k - 1, "n_hist": k, "model": model,
                             "observed": truth[k], "y_next": y[k],
                             **score_moments(mm, vv, y[k])})
    return pd.DataFrame(rows)


def metrics(preds: pd.DataFrame, state_names=("TD", "Indeterminate", "PIGD")) -> dict:
    """Script-04 class metrics plus the continuous log predictive density."""
    probs = np.clip(preds[list(state_names)].to_numpy(), 1e-6, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    j = preds["observed"].map({s: k for k, s in enumerate(state_names)}).to_numpy(int)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(j)), j] = 1.0
    p_true = probs[np.arange(len(j)), j]
    return {"n_pairs": len(preds),
            "loglik_per_pair": float(np.log(p_true).mean()),
            "accuracy": float((probs.argmax(axis=1) == j).mean()),
            "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
            "lpd_per_pair": float(preds["log_density"].mean())}
