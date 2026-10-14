"""GRM with a within-patient Ornstein-Uhlenbeck latent deviation (Amendment 3).

Specification: docs/cluster_phase/06_AMENDMENT3_PERSISTENCE_SPEC.md.

    θ_i ~ N(m_i, K_i),   m_ij = β0 + β1 t_ij
    K_i = Z_i D Z_iᵀ + σ_u² exp(-|t_ij - t_ik| / ℓ) + 1e-8 I,   Z_i = [1, t_i]

Items given θ follow the graded-response model of `grm` with fixed item
parameters. The marginal likelihood of each patient is approximated by Laplace
over the n-vector θ_i; the gradient is exact (explicit terms plus the implicit
dependence of the mode, which needs the third θ-derivative of the GRM
log-likelihood; cf. Rasmussen & Williams 2006, §5.5.1, here with a mean).

    psi = (β0, β1, log L00, L10, log L11, log σ_u, log ℓ),  D = L Lᵀ

Patients are processed in groups with equal visit counts, so every array is
rectangular and no padding enters the likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from . import grm as G

JITTER = 1e-8
BOUNDS = [(-10, 10), (-2, 2), (-6, 2), (-3, 3), (-8, 1), (-6, 2), (np.log(0.05), np.log(50.0))]
NAMES = ["beta0", "beta1", "log_L00", "L10", "log_L11", "log_sigma_u", "log_ell"]


@dataclass(frozen=True)
class Grouped:
    """Patients grouped by visit count: {n: (patients, T (m, n), Y (m, n, K))}."""
    groups: dict
    patients: list


def group(times: dict, items: dict) -> Grouped:
    by_n: dict = {}
    for p in sorted(times):
        by_n.setdefault(len(times[p]), []).append(p)
    groups = {n: (ps, np.stack([np.asarray(times[p], float) for p in ps]),
                  np.stack([np.asarray(items[p], float) for p in ps])) for n, ps in by_n.items()}
    return Grouped(groups, sorted(times))


def unpack(psi):
    beta = np.asarray(psi[:2], float)
    L = np.array([[np.exp(psi[2]), 0.0], [psi[3], np.exp(psi[4])]])
    dL = [np.array([[np.exp(psi[2]), 0], [0, 0]]), np.array([[0, 0], [1.0, 0]]),
          np.array([[0, 0], [0, np.exp(psi[4])]])]
    D = L @ L.T
    dD = [d @ L.T + L @ d.T for d in dL]
    return beta, D, dD, float(np.exp(psi[5])), float(np.exp(psi[6]))


def kernel(T, D, dD, sigma, ell, grads=True):
    """K (m, n, n) and dK/d(kernel params) [5 x (m, n, n)]: three D params, log σ, log ℓ."""
    Z = np.stack([np.ones_like(T), T], -1)                              # (m, n, 2)
    dist = np.abs(T[:, :, None] - T[:, None, :])
    R = np.exp(-dist / ell)
    n = T.shape[1]
    K = np.einsum("mik,kl,mjl->mij", Z, D, Z) + sigma ** 2 * R + JITTER * np.eye(n)
    if not grads:
        return K, None
    dK = [np.einsum("mik,kl,mjl->mij", Z, d, Z) for d in dD]
    dK.append(2 * sigma ** 2 * R)
    dK.append(sigma ** 2 * R * dist / ell)
    return K, dK


def _posterior(f, M, Kinv, Y, a, tau):
    lp = G.category_logprob(f, a, tau, Y).sum(-1).sum(-1)
    r = f - M
    return lp - 0.5 * np.einsum("mi,mij,mj->m", r, Kinv, r)


def modes(M, K, Y, a, tau, f0=None, iters=100, tol=1e-10):
    """Posterior mode of θ per patient by damped Newton (concave objective)."""
    Kinv = np.linalg.inv(K)
    f = M.copy() if f0 is None else f0.copy()
    cur = _posterior(f, M, Kinv, Y, a, tau)
    for _ in range(iters):
        l1, l2, _ = G.theta_derivs(f, a, tau, Y)
        g = l1 - np.einsum("mij,mj->mi", Kinv, f - M)
        A = Kinv - l2[:, :, None] * np.eye(f.shape[1])
        step = np.linalg.solve(A, g[..., None])[..., 0]
        t = np.ones(len(f))
        for _ls in range(40):
            new = _posterior(f + t[:, None] * step, M, Kinv, Y, a, tau)
            bad = ~(new >= cur - 1e-12)
            if not bad.any():
                break
            t = np.where(bad, 0.5 * t, t)
        f = f + t[:, None] * step
        cur = new
        if np.max(np.abs(t[:, None] * step)) < tol:
            break
    return f


def _group_terms(psi, T, Y, a, tau, f0=None, want_grad=True):
    beta, D, dD, sigma, ell = unpack(psi)
    M = beta[0] + beta[1] * T
    K, dK = kernel(T, D, dD, sigma, ell, grads=want_grad)
    f = modes(M, K, Y, a, tau, f0)
    Kinv = np.linalg.inv(K)
    l1, l2, l3 = G.theta_derivs(f, a, tau, Y)
    n = T.shape[1]
    A = Kinv - l2[:, :, None] * np.eye(n)
    _, ldK = np.linalg.slogdet(K)
    sA, ldA = np.linalg.slogdet(A)
    r = f - M
    lp = G.category_logprob(f, a, tau, Y).sum(-1).sum(-1)
    psi_i = lp - 0.5 * np.einsum("mi,mij,mj->m", r, Kinv, r) - 0.5 * ldK - 0.5 * ldA
    if not want_grad:
        return psi_i, f, None, sA
    Ainv = np.linalg.inv(A)
    dAii_l3 = np.einsum("mii->mi", Ainv) * l3                            # (m, n)
    alpha = np.einsum("mij,mj->mi", Kinv, r)                             # K^-1 r
    grad = np.zeros(7)
    for j, x in enumerate((np.ones_like(T), T)):                         # mean params
        Kx = np.einsum("mij,mj->mi", Kinv, x)
        df = np.einsum("mij,mj->mi", Ainv, Kx)
        grad[j] = (np.einsum("mi,mi->m", alpha, x) + 0.5 * np.einsum("mi,mi->m", dAii_l3, df)).sum()
    for k, dKk in enumerate(dK):                                         # kernel params
        KdK = Kinv @ dKk
        explicit = (0.5 * np.einsum("mi,mij,mj->m", alpha, dKk, alpha)
                    - 0.5 * np.einsum("mii->m", KdK)
                    + 0.5 * np.einsum("mij,mji->m", Ainv, KdK @ Kinv))
        dg = np.einsum("mij,mj->mi", dKk, alpha)
        dg = np.einsum("mij,mj->mi", Kinv, dg)                           # K^-1 dK K^-1 r
        df = np.einsum("mij,mj->mi", Ainv, dg)
        grad[2 + k] = (explicit + 0.5 * np.einsum("mi,mi->m", dAii_l3, df)).sum()
    return psi_i, f, grad, sA


def negloglik(psi, grouped: Grouped, a, tau, state=None):
    total, grad = 0.0, np.zeros(7)
    try:
        for n, (ps, T, Y) in grouped.groups.items():
            f0 = None if state is None else state.get(n)
            psi_i, f, g, sA = _group_terms(psi, T, Y, a, tau, f0)
            if np.any(sA <= 0) or not np.all(np.isfinite(psi_i)):
                return 1e12, np.zeros(7)
            if state is not None:
                state[n] = f
            total += psi_i.sum()
            grad += g
    except np.linalg.LinAlgError:
        return 1e12, np.zeros(7)
    return -float(total), -grad


@dataclass
class OUFit:
    beta: np.ndarray
    D: np.ndarray
    sigma_u: float
    ell: float
    nll: float
    converged: bool
    message: str
    n_iter: int
    grad_norm: float
    at_bound: list


def fit(grouped: Grouped, a, tau, psi0, maxiter=500) -> OUFit:
    state: dict = {}
    psi0 = np.clip(np.asarray(psi0, float), [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
    res = minimize(negloglik, psi0, args=(grouped, a, tau, state), jac=True,
                   method="L-BFGS-B", bounds=BOUNDS, options={"maxiter": maxiter})
    beta, D, _, sigma, ell = unpack(res.x)
    at_bound = [nm for nm, v, (lo, hi) in zip(NAMES, res.x, BOUNDS)
                if abs(v - lo) < 1e-6 or abs(v - hi) < 1e-6]
    return OUFit(beta, D, sigma, ell, float(res.fun), bool(res.success), str(res.message),
                 int(res.nit), float(np.linalg.norm(res.jac)), at_bound)


def to_psi(beta, D, sigma, ell):
    l00 = np.sqrt(D[0, 0])
    l10 = D[1, 0] / l00
    l11 = np.sqrt(max(D[1, 1] - l10 ** 2, 1e-12))
    return np.array([beta[0], beta[1], np.log(l00), l10, np.log(l11), np.log(sigma), np.log(ell)])


def laplace_one(t, Y, psi, a, tau):
    psi_i, *_ = _group_terms(np.asarray(psi), t[None], Y[None], a, tau, want_grad=False)
    return float(psi_i[0])


def agq_one(t, Y, psi, a, tau, nodes=7):
    """Adaptive Gauss-Hermite over the n-vector θ (small n only)."""
    beta, D, dD, sigma, ell = unpack(psi)
    T = t[None]
    M = beta[0] + beta[1] * T
    K, _ = kernel(T, D, dD, sigma, ell, grads=False)
    f = modes(M, K, Y[None], a, tau)[0]
    Kinv = np.linalg.inv(K[0])
    _, l2, _ = G.theta_derivs(f, a, tau, Y)
    A = Kinv - np.diag(l2)
    C = np.linalg.cholesky(np.linalg.inv(A))
    x, w = G.gh_nodes(nodes)
    n = len(t)
    grids = np.meshgrid(*[np.arange(nodes)] * n, indexing="ij")
    idx = np.stack([g.ravel() for g in grids], 1)
    U = x[idx]
    F = f + U @ C.T
    _, ldK = np.linalg.slogdet(K[0])
    ll = G.category_logprob(F, a, tau, Y[None]).sum(-1).sum(-1)
    r = F - M[0]
    logprior = -0.5 * np.einsum("qi,ij,qj->q", r, Kinv, r) - 0.5 * ldK - 0.5 * n * np.log(2 * np.pi)
    logw = np.log(w[idx]).sum(1)
    vals = logw + ll + logprior + 0.5 * (U ** 2).sum(1) + 0.5 * n * np.log(2 * np.pi) \
        + np.log(np.abs(np.linalg.det(C)))
    mx = vals.max()
    return float(mx + np.log(np.exp(vals - mx).sum()))


def simulate(rng, times: dict, beta, D, sigma, ell, a, tau) -> dict:
    out = {}
    dD = [np.zeros((2, 2))] * 3
    for p in sorted(times):
        t = np.asarray(times[p], float)[None]
        K, _ = kernel(t, D, dD, sigma, ell, grads=False)
        theta = beta[0] + beta[1] * t[0] + np.linalg.cholesky(K[0]) @ rng.standard_normal(t.shape[1])
        out[p] = G.sample_items(rng, theta, a, tau)
    return out
