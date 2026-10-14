"""Graded-response measurement model for the PIGD items (Stage 6b, Amendment 2).

Specification: docs/cluster_phase/06_AMENDMENT2_PREDICTIVE_CHECK_SPEC.md.

Items k = 1..K with categories 0..C (C = 4 for MDS-UPDRS):

    P(Y_k >= c | θ) = F_kc = expit(a_k (θ - τ_kc)),  c = 1..C
    P(Y_k = c | θ)  = F_kc - F_k,c+1                (F_k0 = 1, F_k,C+1 = 0)

Item parameter vector per item: (α_k, γ_k1, ..., γ_kC) with a_k = exp(α_k),
τ_k1 = γ_k1, τ_kc = τ_k,c-1 + exp(γ_kc): every vector is a valid ordered model.

Stage 1 estimates the item parameters with θ ~ N(0, 1) per visit (Gauss-Hermite
marginal likelihood, analytic gradient). Stage 2 holds them fixed and fits
θ_ij = β0 + β1 t_ij + b_i0 + b_i1 t_ij, b_i ~ N(0, D = L L'), by the Laplace
approximation with its exact gradient. Missing item responses (NaN) contribute
no likelihood term. The GRM log-likelihood is concave in θ (Pratt 1981), so
each patient's posterior mode is unique.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

N_CAT = 5   # categories 0..4


# ---------------------------------------------------------------------------
#  Item parameters
# ---------------------------------------------------------------------------
def unpack_items(phi: np.ndarray, n_items: int):
    """phi (n_items * 5,) -> a (K,), tau (K, 4)."""
    p = np.asarray(phi, float).reshape(n_items, N_CAT)
    a = np.exp(p[:, 0])
    tau = np.cumsum(np.column_stack([p[:, 1], np.exp(p[:, 2:])]), axis=1)
    return a, tau


def pack_items(a, tau) -> np.ndarray:
    a, tau = np.asarray(a, float), np.asarray(tau, float)
    gam = np.column_stack([tau[:, 0], np.log(np.diff(tau, axis=1))])
    return np.column_stack([np.log(a), gam]).ravel()


# ---------------------------------------------------------------------------
#  Category probabilities and θ-derivatives
# ---------------------------------------------------------------------------
def _cum(theta, a, tau):
    """F (..., K, C+2) with F[...,0] = 1 and F[..., C+1] = 0, plus z = a(θ-τ)."""
    z = a[:, None] * (np.asarray(theta)[..., None, None] - tau)          # (..., K, 4)
    F = expit(z)
    one = np.ones(z.shape[:-1] + (1,))
    return np.concatenate([one, F, 0 * one], axis=-1), z


def category_logprob(theta, a, tau, y):
    """log P(Y = y | θ), numerically stable; y (..., K) integer, NaN = missing -> 0.

    The middle categories are differences of two logistic CDFs; when both are
    close to 1 the difference is computed from their complements instead.
    """
    theta = np.asarray(theta, float)
    z = a[:, None] * (theta[..., None, None] - tau)                     # (..., K, 4)
    yv = np.nan_to_num(y, nan=0).astype(int)
    shape = np.broadcast_shapes(z.shape[:-1], yv.shape)
    z = np.broadcast_to(z, shape + (4,))
    yv = np.broadcast_to(yv, shape)
    zp = np.concatenate([np.full(shape + (1,), np.inf), z, np.full(shape + (1,), -np.inf)], -1)
    zc = np.take_along_axis(zp, yv[..., None], -1)[..., 0]               # P(Y >= y)
    zn = np.take_along_axis(zp, (yv + 1)[..., None], -1)[..., 0]         # P(Y >= y+1)
    # log(expit(zc) - expit(zn)), zc > zn
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        hi = np.where(np.isinf(zc), 0.0, log_expit(zc))
        diff_low = hi + np.log1p(-np.exp(np.where(np.isinf(zn), -np.inf,
                                                    log_expit(zn)) - hi))
        # complement form: expit(-zn) - expit(-zc)
        lo_c = np.where(np.isinf(zn), 0.0, log_expit(-zn))
        diff_high = lo_c + np.log1p(-np.exp(np.where(np.isinf(zc), -np.inf,
                                                      log_expit(-zc)) - lo_c))
    out = np.where(zn > 0, diff_high, diff_low)
    out = np.maximum(out, -700.0)
    miss = np.broadcast_to(np.isnan(y), shape)
    return np.where(miss, 0.0, out)


def theta_derivs(theta, a, tau, y):
    """First three θ-derivatives of log P(Y = y | θ), summed over items.

    With F = expit(a(θ - τ)): F' = a F(1-F), F'' = a^2 F(1-F)(1-2F),
    F''' = a^3 F(1-F)(1-6F+6F^2); P = F_c - F_{c+1}; and
    l1 = P'/P, l2 = P''/P - l1^2, l3 = P'''/P - 3 l1 P''/P + 2 l1^3.
    Missing items contribute zero. Returns arrays shaped like theta.
    """
    F, _ = _cum(theta, a, tau)                                          # (..., K, 6)
    s = F * (1 - F)
    d1 = a[:, None] * s
    d2 = a[:, None] ** 2 * s * (1 - 2 * F)
    d3 = a[:, None] ** 3 * s * (1 - 6 * F + 6 * F ** 2)
    yv = np.nan_to_num(y, nan=0).astype(int)
    shape = np.broadcast_shapes(F.shape[:-1], yv.shape)
    yv = np.broadcast_to(yv, shape)

    def pick(M, off):
        M = np.broadcast_to(M, shape + (M.shape[-1],))
        return np.take_along_axis(M, (yv + off)[..., None], -1)[..., 0]
    P = np.exp(category_logprob(theta, a, tau, np.broadcast_to(np.nan_to_num(y, nan=0), shape)))
    P1 = pick(d1, 0) - pick(d1, 1)
    P2 = pick(d2, 0) - pick(d2, 1)
    P3 = pick(d3, 0) - pick(d3, 1)
    P = np.maximum(P, 1e-300)
    l1, r2, r3 = P1 / P, P2 / P, P3 / P
    l2 = r2 - l1 ** 2
    l3 = r3 - 3 * l1 * r2 + 2 * l1 ** 3
    m = ~np.broadcast_to(np.isnan(y), shape)
    return (l1 * m).sum(-1), (l2 * m).sum(-1), (l3 * m).sum(-1)


def sample_items(rng, theta, a, tau):
    """Y = number of thresholds passed: Y = sum_c 1[U < F_c] (F_c decreasing in c)."""
    F, _ = _cum(theta, a, tau)
    U = rng.random(F.shape[:-1] + (1,))
    return (U < F[..., 1:-1]).sum(-1).astype(float)


# ---------------------------------------------------------------------------
#  Stage 1: item parameters, θ ~ N(0, 1)
# ---------------------------------------------------------------------------
def gh_nodes(n=61):
    x, w = np.polynomial.hermite_e.hermegauss(n)       # probabilists': weight e^{-x^2/2}
    return x, w / np.sqrt(2 * np.pi)


def stage1_negloglik(phi, Y, nodes=None):
    """Negative marginal log-likelihood and its analytic gradient.

    Y (V, K) responses (NaN = missing). θ ~ N(0, 1) per visit.
    """
    V, K = Y.shape
    x, w = nodes if nodes is not None else gh_nodes()
    a, tau = unpack_items(phi, K)
    lp = category_logprob(x[:, None], a, tau, Y[None])                  # (Q, V, K)
    ll = lp.sum(-1) + np.log(w)[:, None]                                # (Q, V)
    mx = ll.max(0)
    post = np.exp(ll - mx)
    norm = post.sum(0)
    total = float((mx + np.log(norm)).sum())
    post /= norm                                                        # (Q, V)

    # d log P / d a_k and d / d tau_kc for each node, visit, item
    F, _ = _cum(x[:, None], a, tau)                                     # (Q, 1, K, 6)
    F = np.broadcast_to(F, (len(x), V, K, N_CAT + 1))
    s = F * (1 - F)
    yv = np.nan_to_num(Y, nan=0).astype(int)[None]                     # (1, V, K)
    yv = np.broadcast_to(yv, (len(x), V, K))
    P = np.maximum(np.exp(lp), 1e-300)
    taup = np.concatenate([np.zeros((K, 1)), tau, np.zeros((K, 1))], 1)  # index c
    th = x[:, None, None]
    sc = np.take_along_axis(s, yv[..., None], -1)[..., 0]
    sn = np.take_along_axis(s, (yv + 1)[..., None], -1)[..., 0]
    tc = taup[np.arange(K)[None, None, :], yv]
    tn = taup[np.arange(K)[None, None, :], yv + 1]
    has_c, has_n = yv >= 1, yv <= 3
    dlogP_da = (np.where(has_c, sc * (th - tc), 0) - np.where(has_n, sn * (th - tn), 0)) / P
    miss = np.broadcast_to(np.isnan(Y)[None], dlogP_da.shape)
    dlogP_da = np.where(miss, 0, dlogP_da)
    # d log P / d tau_c = -a s_c / P (if c = y >= 1);  + a s_{c} / P (if c = y + 1 <= 4)
    dtau = np.zeros((len(x), V, K, 4))
    for c in range(1, 5):
        dtau[..., c - 1] = (np.where(yv == c, -a * sc, 0) + np.where(yv + 1 == c, a * sn, 0)) / P
    dtau = np.where(miss[..., None], 0, dtau)
    g_a = np.einsum("qv,qvk->k", post, dlogP_da)                        # (K,)
    g_tau = np.einsum("qv,qvkc->kc", post, dtau)                        # (K, 4)
    p = np.asarray(phi, float).reshape(K, N_CAT)
    grad = np.zeros((K, N_CAT))
    grad[:, 0] = g_a * a                                                # d a / d alpha = a
    # tau_c = gamma_1 + sum_{m=2..c} exp(gamma_m)
    grad[:, 1] = g_tau.sum(1)
    for m in range(2, 5):
        grad[:, m] = g_tau[:, m - 1:].sum(1) * np.exp(p[:, m])
    return -total, -grad.ravel()


@dataclass
class ItemFit:
    a: np.ndarray
    tau: np.ndarray
    nll: float
    converged: bool
    message: str
    n_visits: int
    item_names: list = field(default_factory=list)


def initial_items(Y):
    K = Y.shape[1]
    a = np.full(K, 1.5)
    tau = np.zeros((K, 4))
    for k in range(K):
        yk = Y[:, k][~np.isnan(Y[:, k])]
        for c in range(1, 5):
            p = np.clip((yk >= c).mean(), 1e-3, 1 - 1e-3)
            tau[k, c - 1] = -np.log(p / (1 - p)) / a[k]
        tau[k] = np.maximum.accumulate(tau[k] + np.arange(4) * 1e-3)
    return pack_items(a, tau)


def fit_items(Y, names=None, maxiter=2000) -> ItemFit:
    Y = np.asarray(Y, float)
    vals = Y[~np.isnan(Y)]
    if np.any((vals < 0) | (vals > 4) | (vals != np.round(vals))):
        raise ValueError("item responses must be integers 0..4 or NaN")
    nodes = gh_nodes()
    res = minimize(stage1_negloglik, initial_items(Y), args=(Y, nodes), jac=True,
                   method="L-BFGS-B", options={"maxiter": maxiter})
    a, tau = unpack_items(res.x, Y.shape[1])
    return ItemFit(a, tau, float(res.fun), bool(res.success), str(res.message),
                   Y.shape[0], list(names or []))


# ---------------------------------------------------------------------------
#  Stage 2: longitudinal latent trait, items fixed, Laplace with exact gradient
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Packed:
    T: np.ndarray        # (M, n)
    Y: np.ndarray        # (M, n, K), NaN = missing item
    mask: np.ndarray     # (M, n) visit present
    patients: list


def pack(times: dict, items: dict) -> Packed:
    pats = sorted(times)
    n = max(len(times[p]) for p in pats)
    K = np.asarray(items[pats[0]]).shape[1]
    T = np.zeros((len(pats), n))
    Y = np.full((len(pats), n, K), np.nan)
    m = np.zeros((len(pats), n), bool)
    for i, p in enumerate(pats):
        k = len(times[p])
        T[i, :k], Y[i, :k], m[i, :k] = times[p], items[p], True
    return Packed(T, Y, m, pats)


def unpack_long(psi):
    beta = np.asarray(psi[:2], float)
    L = np.array([[np.exp(psi[2]), 0.0], [psi[3], np.exp(psi[4])]])
    dL = [np.array([[np.exp(psi[2]), 0], [0, 0]]), np.array([[0, 0], [1.0, 0]]),
          np.array([[0, 0], [0, np.exp(psi[4])]])]
    D = L @ L.T
    dD = [d @ L.T + L @ d.T for d in dL]
    return beta, D, dD


def _visit_derivs(packed, beta, b, a, tau):
    eta = beta[0] + beta[1] * packed.T + b[:, :1] + b[:, 1:2] * packed.T
    l1, l2, l3 = theta_derivs(eta, a, tau, packed.Y)
    m = packed.mask
    lp = category_logprob(eta, a, tau, packed.Y).sum(-1)
    return lp * m, l1 * m, l2 * m, l3 * m


def _log_posterior(packed, beta, b, a, tau, Dinv):
    eta = beta[0] + beta[1] * packed.T + b[:, :1] + b[:, 1:2] * packed.T
    lp = (category_logprob(eta, a, tau, packed.Y).sum(-1) * packed.mask).sum(1)
    return lp - 0.5 * np.einsum("pk,kl,pl->p", b, Dinv, b)


def modes(packed, beta, D, a, tau, b0=None, iters=100, tol=1e-10):
    """Per-patient posterior mode of b by damped Newton (vectorised).

    The log-posterior is strictly concave, so the mode is unique, but full
    Newton steps can overshoot where the curvature changes fast; each step is
    halved per patient until the log-posterior does not decrease.
    """
    M = packed.T.shape[0]
    Dinv = np.linalg.inv(D)
    Z = np.stack([np.ones_like(packed.T), packed.T], -1) * packed.mask[..., None]
    b = np.zeros((M, 2)) if b0 is None else b0.copy()
    cur = _log_posterior(packed, beta, b, a, tau, Dinv)
    for _ in range(iters):
        _, l1, l2, _ = _visit_derivs(packed, beta, b, a, tau)
        g = np.einsum("pnk,pn->pk", Z, l1) - b @ Dinv
        A = -np.einsum("pnk,pn,pnl->pkl", Z, l2, Z) + Dinv
        step = np.linalg.solve(A, g[..., None])[..., 0]
        t = np.ones(M)
        for _ls in range(40):
            new = _log_posterior(packed, beta, b + t[:, None] * step, a, tau, Dinv)
            bad = ~(new >= cur - 1e-12)
            if not bad.any():
                break
            t = np.where(bad, 0.5 * t, t)
        b = b + t[:, None] * step
        cur = new
        if np.max(np.abs(t[:, None] * step)) < tol:
            break
    return b


def laplace_negloglik(psi, packed, a, tau, state=None):
    """Negative Laplace marginal log-likelihood and its exact gradient (5,)."""
    beta, D, dD = unpack_long(psi)
    Dinv = np.linalg.inv(D)
    b = modes(packed, beta, D, a, tau, None if state is None else state.get("b"))
    if state is not None:
        state["b"] = b
    lp, l1, l2, l3 = _visit_derivs(packed, beta, b, a, tau)
    Z = np.stack([np.ones_like(packed.T), packed.T], -1) * packed.mask[..., None]
    A = -np.einsum("pnk,pn,pnl->pkl", Z, l2, Z) + Dinv                 # -H, PD
    Ainv = np.linalg.inv(A)
    sA, ldA = np.linalg.slogdet(A)
    _, ldD = np.linalg.slogdet(D)
    quad = np.einsum("pk,kl,pl->p", b, Dinv, b)
    loglik = lp.sum(1) - 0.5 * quad - 0.5 * ldD - 0.5 * ldA
    total = float(loglik.sum())

    # dA/db_k (for the mode's dependence on psi) and partial derivatives
    dA_db = [-np.einsum("pn,pnk,pnl->pkl", l3 * Z[..., kk], Z, Z) for kk in range(2)]
    grad = np.zeros(5)
    for m_ in range(2):                                                 # beta_m
        df = np.einsum("pn,pn->p", l1, Z[..., m_])
        dA_part = -np.einsum("pn,pnk,pnl->pkl", l3 * Z[..., m_], Z, Z)
        dg = np.einsum("pn,pnk->pk", l2 * Z[..., m_], Z)                # d grad_b / d beta_m
        db = np.einsum("pkl,pl->pk", Ainv, dg)
        dA = dA_part + db[:, 0, None, None] * dA_db[0] + db[:, 1, None, None] * dA_db[1]
        grad[m_] = (df - 0.5 * np.einsum("pkl,plk->p", Ainv, dA)).sum()
    for j, Dp in enumerate(dD):                                         # chol params
        dDinv = -Dinv @ Dp @ Dinv
        df = -0.5 * (np.einsum("pk,kl,pl->p", b, dDinv, b) + np.trace(Dinv @ Dp))
        dg = -(b @ dDinv.T)                                             # d grad_b / d p
        db = np.einsum("pkl,pl->pk", Ainv, dg)
        dA = dDinv[None] + db[:, 0, None, None] * dA_db[0] + db[:, 1, None, None] * dA_db[1]
        grad[2 + j] = (df - 0.5 * np.einsum("pkl,plk->p", Ainv, dA)).sum()
    if not np.isfinite(total) or np.any(sA <= 0):
        return 1e12, np.zeros(5)
    return -total, -grad


@dataclass
class LongFit:
    beta: np.ndarray
    D: np.ndarray
    nll: float
    converged: bool
    message: str
    n_iter: int
    grad_norm: float
    modes: np.ndarray


def fit_longitudinal(packed, a, tau, psi0=None, maxiter=500) -> LongFit:
    state: dict = {}
    psi0 = np.array([0.0, 0.1, np.log(1.0), 0.0, np.log(0.2)]) if psi0 is None else psi0
    res = minimize(laplace_negloglik, psi0, args=(packed, a, tau, state), jac=True,
                   method="L-BFGS-B", options={"maxiter": maxiter})
    beta, D, _ = unpack_long(res.x)
    return LongFit(beta, D, float(res.fun), bool(res.success), str(res.message),
                   int(res.nit), float(np.linalg.norm(res.jac)),
                   modes(packed, beta, D, a, tau, state.get("b")))


def agq_loglik(t, Y, beta, D, a, tau, n=15):
    """Adaptive 2-d Gauss-Hermite marginal log-likelihood for one patient."""
    one = Packed(t[None], Y[None], np.ones((1, len(t)), bool), [0])
    b = modes(one, beta, D, a, tau)[0]
    Z = np.column_stack([np.ones_like(t), t])
    _, _, l2, _ = _visit_derivs(one, beta, b[None], a, tau)
    A = -(Z.T * l2[0]) @ Z + np.linalg.inv(D)
    C = np.linalg.cholesky(np.linalg.inv(A))
    x, w = gh_nodes(n)
    Dinv, (_, ldD) = np.linalg.inv(D), np.linalg.slogdet(D)
    vals = []
    for i in range(n):
        for j in range(n):
            u = np.array([x[i], x[j]])
            bb = b + C @ u
            eta = beta[0] + beta[1] * t + bb[0] + bb[1] * t
            ll = category_logprob(eta, a, tau, Y).sum()
            logprior = -0.5 * bb @ Dinv @ bb - 0.5 * ldD - np.log(2 * np.pi)
            vals.append(np.log(w[i] * w[j]) + ll + logprior + 0.5 * u @ u + np.log(2 * np.pi)
                        + np.log(np.linalg.det(C)))
    vals = np.array(vals)
    mx = vals.max()
    return float(mx + np.log(np.exp(vals - mx).sum()))


def laplace_loglik_one(t, Y, beta, D, a, tau):
    one = Packed(t[None], Y[None], np.ones((1, len(t)), bool), [0])
    psi = np.array([beta[0], beta[1], np.log(np.sqrt(D[0, 0])), D[1, 0] / np.sqrt(D[0, 0]),
                    np.log(np.sqrt(D[1, 1] - D[1, 0] ** 2 / D[0, 0]))])
    return -laplace_negloglik(psi, one, a, tau)[0]


def simulate(rng, times: dict, beta, D, a, tau, b=None) -> dict:
    """Item responses (n_visits, K) per patient on the given visit times."""
    pats = sorted(times)
    if b is None:
        b = rng.multivariate_normal(np.zeros(2), D, size=len(pats))
    out = {}
    for i, p in enumerate(pats):
        t = np.asarray(times[p], float)
        theta = beta[0] + beta[1] * t + b[i, 0] + b[i, 1] * t
        out[p] = sample_items(rng, theta, a, tau)
    return out
