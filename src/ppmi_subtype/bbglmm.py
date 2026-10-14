"""Beta-binomial mixed model for an item-sum score (measurement-model candidate).

An MDS-UPDRS subscore is the mean of k items scored 0-4, i.e. an item sum
S in {0, ..., n} with n = 4k (PIGD: k = 5, n = 20; tremor: k = 11, n = 44).
The candidate measurement model is

    S_ij | b_i ~ BetaBinomial(n, mu_ij, phi)
    logit(mu_ij) = beta0 + beta1 t_ij + b_i0 + b_i1 t_ij,   b_i ~ N(0, D)

`mu` is the expected fraction of the maximum, `phi` the beta precision
(phi -> inf is binomial). Bounded, discrete, zeros arise from the same process
as positive scores (no structural-zero class), and Var(S | mu) =
n mu (1 - mu) (n + phi) / (1 + phi) depends on the level.

Fitting: marginal likelihood by the Laplace approximation over the 2-d random
effect, with the per-patient mode found by Newton iterations vectorised over
patients; outer parameters by L-BFGS-B on finite-difference gradients.

    theta = (beta0, beta1, log L00, L10, log L11, log phi),  D = L L'
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, digamma, expit, gammaln, polygamma


# ---------------------------------------------------------------------------
#  Beta-binomial pieces
# ---------------------------------------------------------------------------
def bb_logpmf(y, n, mu, phi):
    a, b = mu * phi, (1.0 - mu) * phi
    return (gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
            + betaln(y + a, n - y + b) - betaln(a, b))


def bb_eta_derivs(y, n, mu, phi):
    """d/d eta and d^2/d eta^2 of the log-pmf, eta = logit(mu)."""
    a, b = mu * phi, (1.0 - mu) * phi
    d1mu = phi * (digamma(y + a) - digamma(a) - digamma(n - y + b) + digamma(b))
    d2mu = phi ** 2 * (polygamma(1, y + a) - polygamma(1, a)
                       + polygamma(1, n - y + b) - polygamma(1, b))
    v = mu * (1.0 - mu)
    return d1mu * v, d2mu * v * v + d1mu * v * (1.0 - 2.0 * mu)


def sample_bb(rng, n, mu, phi):
    p = rng.beta(mu * phi, (1.0 - mu) * phi)
    return rng.binomial(n, p)


# ---------------------------------------------------------------------------
#  Data and parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Packed:
    T: np.ndarray        # (M, nmax) times
    Y: np.ndarray        # (M, nmax) item sums
    mask: np.ndarray     # (M, nmax) bool
    n_trials: int
    patients: list


def pack(times: dict, sums: dict, n_trials: int) -> Packed:
    pats = sorted(times)
    nmax = max(len(times[p]) for p in pats)
    T = np.zeros((len(pats), nmax))
    Y = np.zeros((len(pats), nmax))
    m = np.zeros((len(pats), nmax), bool)
    for i, p in enumerate(pats):
        k = len(times[p])
        T[i, :k], Y[i, :k], m[i, :k] = times[p], sums[p], True
    if np.any((Y[m] < 0) | (Y[m] > n_trials)) or not np.allclose(Y[m], np.round(Y[m])):
        raise ValueError("item sums must be integers in [0, n_trials]")
    return Packed(T, Y, m, int(n_trials), pats)


def unpack(theta):
    beta = np.asarray(theta[:2], float)
    L = np.array([[np.exp(theta[2]), 0.0], [theta[3], np.exp(theta[4])]])
    return beta, L @ L.T, float(np.exp(theta[5]))


# ---------------------------------------------------------------------------
#  Laplace marginal likelihood
# ---------------------------------------------------------------------------
def _modes(packed: Packed, beta, D, phi, b0=None, iters=50, tol=1e-10):
    T, Y, m, n = packed.T, packed.Y, packed.mask, packed.n_trials
    M = T.shape[0]
    Dinv = np.linalg.inv(D)
    b = np.zeros((M, 2)) if b0 is None else b0.copy()
    Z = np.stack([np.ones_like(T), T], axis=-1) * m[..., None]          # (M, n, 2)
    base = beta[0] + beta[1] * T

    def parts(bb):
        eta = base + bb[:, :1] + bb[:, 1:2] * T
        mu = np.clip(expit(eta), 1e-10, 1 - 1e-10)
        g1, g2 = bb_eta_derivs(Y, n, mu, phi)
        g1, g2 = g1 * m, g2 * m
        ll = (bb_logpmf(Y, n, mu, phi) * m).sum(axis=1)
        logpost = ll - 0.5 * np.einsum("pi,ij,pj->p", bb, Dinv, bb)
        grad = np.einsum("pni,pn->pi", Z, g1) - bb @ Dinv
        # Negative-definite curvature: clip positive second derivatives so a
        # Newton step is always an ascent direction (exact at the mode, where
        # the log-posterior is locally concave).
        H = np.einsum("pni,pn,pnj->pij", Z, np.minimum(g2, -1e-12), Z) - Dinv
        return logpost, grad, H, ll

    lp, g, H, ll = parts(b)
    for _ in range(iters):
        step = -np.linalg.solve(H, g[..., None])[..., 0]
        t = np.ones(M)
        for _ls in range(30):                                            # backtracking
            nb = b + t[:, None] * step
            nlp, ng, nH, nll = parts(nb)
            bad = ~(nlp >= lp - 1e-12)
            if not bad.any():
                break
            t = np.where(bad, t * 0.5, t)
        b, lp, g, H, ll = nb, nlp, ng, nH, nll
        if np.max(np.abs(step * t[:, None])) < tol:
            break
    # exact (unclipped) curvature at the mode for the Laplace determinant
    eta = base + b[:, :1] + b[:, 1:2] * T
    mu = np.clip(expit(eta), 1e-10, 1 - 1e-10)
    _, g2 = bb_eta_derivs(Y, n, mu, phi)
    Hx = np.einsum("pni,pn,pnj->pij", Z, g2 * m, Z) - Dinv
    return b, lp, Hx


def neg_marginal_loglik(theta, packed: Packed, state: dict | None = None) -> float:
    try:
        beta, D, phi = unpack(theta)
        b0 = state.get("b") if state is not None else None
        b, lp, H = _modes(packed, beta, D, phi, b0)
        sign, logdet = np.linalg.slogdet(-H)
        if np.any(sign <= 0):
            return 1e12
        _, logdetD = np.linalg.slogdet(D)
        val = -(lp - 0.5 * logdetD - 0.5 * logdet).sum()
        if state is not None and np.isfinite(val):
            state["b"] = b
        return float(val) if np.isfinite(val) else 1e12
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        return 1e12


@dataclass
class BBFit:
    beta: np.ndarray
    D: np.ndarray
    phi: float
    nll: float
    converged: bool
    message: str
    n_trials: int
    modes: np.ndarray
    at_bound: list | None = None

    @property
    def rho(self) -> float:
        """Intra-cluster (item-sum) overdispersion 1 / (1 + phi)."""
        return 1.0 / (1.0 + self.phi)


def initial_theta(packed: Packed) -> np.ndarray:
    frac = np.clip(packed.Y[packed.mask] / packed.n_trials, 0.01, 0.99)
    return np.array([np.log(frac.mean() / (1 - frac.mean())), 0.05,
                     np.log(0.7), 0.0, np.log(0.08), np.log(20.0)])


#: Bounds on theta. Wide enough for any plausible subscore model (random-effect
#: SDs from e^-5 to e^2 on the logit scale, phi from 0.14 to ~8,100, the latter
#: practically binomial) but finite, so a misspecified fit cannot drift to
#: infinity unnoticed; a parameter resting on a bound is reported.
BOUNDS = [(-10, 10), (-2, 2), (-5, 2), (-2, 2), (-8, 0), (-2, 9)]
NAMES = ["beta0", "beta1", "log_L00", "L10", "log_L11", "log_phi"]


def fit(packed: Packed, theta0=None, maxiter=400) -> BBFit:
    state: dict = {}
    th0 = initial_theta(packed) if theta0 is None else np.asarray(theta0, float)
    th0 = np.clip(th0, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
    res = minimize(neg_marginal_loglik, th0, args=(packed, state), method="L-BFGS-B",
                   bounds=BOUNDS, options={"maxiter": maxiter})
    beta, D, phi = unpack(res.x)
    b, _, _ = _modes(packed, beta, D, phi, state.get("b"))
    at_bound = [n for n, v, (lo, hi) in zip(NAMES, res.x, BOUNDS)
                if abs(v - lo) < 1e-6 or abs(v - hi) < 1e-6]
    return BBFit(beta, D, phi, float(res.fun), bool(res.success), str(res.message),
                 packed.n_trials, b, at_bound)


def simulate(rng, fit_: BBFit, times: dict, b: np.ndarray | None = None) -> dict:
    """Item sums for each patient's visit times; `b` overrides the random effects."""
    pats = sorted(times)
    if b is None:
        b = rng.multivariate_normal(np.zeros(2), fit_.D, size=len(pats))
    out = {}
    for i, p in enumerate(pats):
        t = np.asarray(times[p], float)
        mu = expit(fit_.beta[0] + fit_.beta[1] * t + b[i, 0] + b[i, 1] * t)
        out[p] = sample_bb(rng, fit_.n_trials, mu, fit_.phi).astype(float)
    return out
