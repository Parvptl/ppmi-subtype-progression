"""M2, Amendment 1 (Revision 2) prototype: per-visit quadrature over ε, Laplace / AGQ over b.

Specification: docs/cluster_phase/21_STAGE2_DOC16_AMENDMENT1_PROPOSED_REV2_20261017.md
(§2.1-2.3, §2.5), amending the frozen doc 16. Same model M2 and parameters ψ as
`m2_grm` (B4); only the numerical route to the marginal likelihood differs:

    L_ij(b) = Σ_q w_q g_ij(b, u_q),  θ_ijq = μ_ij + A_ij b + C_ε u_q   (Q × Q Gauss-Hermite)
    log p(Y_i) ≈ Φ(b̂) − ½ log|D| − ½ log|H|,  Φ(b) = Σ_j log L_ij(b) − ½ bᵀD⁻¹b,  H = −∇²Φ(b̂)

(primary, "laplace"), or adaptive Gauss-Hermite over b at (b̂, H⁻¹) with G nodes per
dimension (fallback, "agq"). The exact ψ-gradient of the Laplace form uses the
envelope theorem plus the implicit mode term; all b- and ψ-derivatives of
log L_ij are cumulants of the per-node GRM derivatives under the node posterior
π_q ∝ w_q g_q (θ is affine in b and ψ-directional derivatives are node-wise).
The AGQ gradient is that of the quadrature sum at fixed node placement (Rev 2 §2.3).

Log-concavity of the exact per-patient integrand in b is motivation only (Rev 2
§2.1): the quadrature sum need not be log-concave, so the mode search checks the
Hessian, falls back to gradient ascent, and reports patients whose final Hessian
is not negative definite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from . import grm as G
from . import m2_grm as M

N_B = 4
N_DIR = 4 + M.N_MEAN + M.N_LE          # b (4), mean (6), L_ε (3)
LE0 = M.N_MEAN + M.N_L                 # ψ index of the first L_ε entry


# ---------------------------------------------------------------------------
#  Nodes
# ---------------------------------------------------------------------------
def eps_nodes(psi, Q: int):
    """Per-visit ε nodes E = U C_εᵀ (Q², 2), log weights (Q²,), U (Q², 2), C_ε, dC_ε (3)."""
    x, w = G.gh_nodes(Q)
    U = np.stack(np.meshgrid(x, x, indexing="ij"), -1).reshape(-1, 2)
    logw = (np.log(w)[:, None] + np.log(w)[None, :]).ravel()
    le = np.asarray(psi, float)[LE0:LE0 + 3]
    C = np.array([[np.exp(le[0]), 0.0], [le[1], np.exp(le[2])]])
    dC = [np.array([[C[0, 0], 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 0.0]]),
          np.array([[0.0, 0.0], [0.0, C[1, 1]]])]
    return U @ C.T, logw, U, C, dC


def b_nodes(Gn: int):
    """Tensor GH nodes for 4-D b in standard units: Z (G⁴, 4), log weights (G⁴,)."""
    x, w = G.gh_nodes(Gn)
    grids = np.meshgrid(*[np.arange(Gn)] * N_B, indexing="ij")
    idx = np.stack([g.ravel() for g in grids], 1)
    return x[idx], np.log(w)[idx].sum(1)


def design(T):
    """A (m, n, 2, 4): θ^T gets b0T + b1T t, θ^P gets b0P + b1P t."""
    m, n = T.shape
    A = np.zeros((m, n, 2, 4))
    A[..., 0, 0], A[..., 0, 1] = 1.0, T
    A[..., 1, 2], A[..., 1, 3] = 1.0, T
    return A


def visit_means(psi, T, X):
    beta = np.asarray(psi, float)[:M.N_MEAN]
    mu = np.einsum("k,kmi->mi", beta, M.mean_design(T, X))
    n = T.shape[1]
    return np.stack([mu[:, :n], mu[:, n:]], -1)            # (m, n, 2)


# ---------------------------------------------------------------------------
#  Per-node GRM terms
# ---------------------------------------------------------------------------
def _theta(mu, A, b, E):
    """θ (m, n, Q², 2) at b (m, 4)."""
    return (mu + np.einsum("mnkd,md->mnk", A, b))[:, :, None, :] + E[None, None]


def node_loglik(th, YT, YP, it: M.Items):
    """Σ_items log P at each node: (m, n, Q²)."""
    return (G.category_logprob(th[..., 0], it.aT, it.tauT, YT[:, :, None, :]).sum(-1)
            + G.category_logprob(th[..., 1], it.aP, it.tauP, YP[:, :, None, :]).sum(-1))


def node_derivs(th, YT, YP, it: M.Items):
    """θ-derivatives (orders 1-3) per node, (m, n, Q², 2) each; diagonal (T and P separable)."""
    dT = G.theta_derivs(th[..., 0], it.aT, it.tauT, YT[:, :, None, :])
    dP = G.theta_derivs(th[..., 1], it.aP, it.tauP, YP[:, :, None, :])
    return tuple(np.stack([u, v], -1) for u, v in zip(dT, dP))


def _phi(mu, A, b, E, logw, YT, YP, it, Dinv):
    ll = node_loglik(_theta(mu, A, b, E), YT, YP, it) + logw
    return logsumexp(ll, axis=-1).sum(1) - 0.5 * np.einsum("md,de,me->m", b, Dinv, b)


def _local(mu, A, b, E, logw, YT, YP, it, Dinv):
    """Φ, ∇_bΦ, ∇²_bΦ and the node posterior pieces at b."""
    th = _theta(mu, A, b, E)
    ll = node_loglik(th, YT, YP, it) + logw
    lse = logsumexp(ll, axis=-1)
    pi = np.exp(ll - lse[..., None])
    g1, g2, g3 = node_derivs(th, YT, YP, it)
    s = np.einsum("mnq,mnqk->mnk", pi, g1)
    V = (np.einsum("mnq,mnqk->mnk", pi, g2)[..., None] * np.eye(2)
         + np.einsum("mnq,mnqk,mnql->mnkl", pi, g1, g1) - s[..., :, None] * s[..., None, :])
    grad = np.einsum("mnkd,mnk->md", A, s) - b @ Dinv
    hess = np.einsum("mnkd,mnkl,mnle->mde", A, V, A) - Dinv
    phi = lse.sum(1) - 0.5 * np.einsum("md,de,me->m", b, Dinv, b)
    return phi, grad, hess, (pi, g1, g2, g3)


def modes(mu, A, E, logw, YT, YP, it, Dinv, b0=None, iters=60, tol=1e-9):
    """Per-patient mode of Φ by damped Newton; gradient ascent where the Hessian is not ND.

    Returns b̂ (m, 4) and a flag (m,) for a negative-definite Hessian at b̂.
    """
    m = mu.shape[0]
    b = np.zeros((m, N_B)) if b0 is None else b0.copy()
    cur, grad, hess, _ = _local(mu, A, b, E, logw, YT, YP, it, Dinv)
    for _ in range(iters):
        nd = np.all(np.linalg.eigvalsh(hess) < 0, axis=1)
        step = np.where(nd[:, None], np.linalg.solve(-np.where(nd[:, None, None], hess, -np.eye(N_B)),
                                                     grad[..., None])[..., 0],
                        grad / (1.0 + np.linalg.norm(grad, axis=1, keepdims=True)))
        t = np.ones(m)
        new = cur
        for _ls in range(50):
            new = _phi(mu, A, b + t[:, None] * step, E, logw, YT, YP, it, Dinv)
            bad = ~(new >= cur - 1e-12)
            if not bad.any():
                break
            t = np.where(bad, 0.5 * t, t)
        b = b + t[:, None] * step
        cur, grad, hess, _ = _local(mu, A, b, E, logw, YT, YP, it, Dinv)
        if np.max(np.abs(t[:, None] * step)) < tol:
            break
    nd = np.all(np.linalg.eigvalsh(hess) < 0, axis=1)
    return b, nd


# ---------------------------------------------------------------------------
#  Cumulant derivatives of log L_ij along b and ψ directions
# ---------------------------------------------------------------------------
def _directions(psi, T, X, U, dC):
    """Node-wise θ-directions (m, n, Q², N_DIR, 2): b (4), mean (6), L_ε (3)."""
    m, n = T.shape
    Q2 = U.shape[0]
    Dir = np.zeros((m, n, Q2, N_DIR, 2))
    A = design(T)
    Dir[..., :4, :] = np.swapaxes(A, -1, -2)[:, :, None]
    Xd = M.mean_design(T, X)                                         # (6, m, 2n)
    for p in range(M.N_MEAN):
        Dir[..., 4 + p, 0] = Xd[p][:, :n][..., None]
        Dir[..., 4 + p, 1] = Xd[p][:, n:][..., None]
    for r in range(3):
        Dir[..., 10 + r, :] = (U @ dC[r].T)[None, None]
    return Dir


def _cumulants(pi, g1, g2, g3, Dir):
    """Per visit: E[a] (13), D2 (b x 13), D3 (b x b x 13) of log L_ij."""
    a = np.einsum("mnqk,mnqek->mnqe", g1, Dir)                       # (m,n,Q,13)
    Db = Dir[..., :4, :]
    S2 = np.einsum("mnqk,mnqdk,mnqek->mnqde", g2, Db, Dir)            # (m,n,Q,4,13)
    S3 = np.einsum("mnqk,mnqdk,mnqfk,mnqek->mnqdfe", g3, Db, Db, Dir)  # (m,n,Q,4,4,13)
    E = lambda x: np.einsum("mnq,mnq...->mn...", pi, x)  # noqa: E731
    ab = a[..., :4]
    Ea = E(a)
    Eab = E(ab[..., :, None] * a[..., None, :])                       # (4,13)
    ES2 = E(S2)
    D2 = ES2 + Eab - Ea[..., :4, None] * Ea[..., None, :]
    S2bb = S2[..., :4]                                                # (m,n,Q,4,4)
    t1 = E(S3)
    t2 = E(S2bb[..., None] * a[..., None, None, :])                   # S2_df a_e
    t3 = E(S2[..., :, None, :] * ab[..., None, :, None])              # S2_de a_f
    t4 = E(S2[..., None, :, :] * ab[..., :, None, None])              # S2_fe a_d
    t5 = E(ab[..., :, None, None] * ab[..., None, :, None] * a[..., None, None, :])
    Eb = Ea[..., :4]
    ES2bb = ES2[..., :4]
    Ebb = Eab[..., :4]
    D3 = (t1 + t2 + t3 + t4 + t5
          - ES2bb[..., None] * Ea[..., None, None, :]
          - ES2[..., :, None, :] * Eb[..., None, :, None]
          - ES2[..., None, :, :] * Eb[..., :, None, None]
          - Ebb[..., None] * Ea[..., None, None, :]
          - Eab[..., :, None, :] * Eb[..., None, :, None]
          - Eab[..., None, :, :] * Eb[..., :, None, None]
          + 2 * Eb[..., :, None, None] * Eb[..., None, :, None] * Ea[..., None, None, :])
    return Ea, D2, D3


def _visit_param_index():
    """ψ index of direction e >= 4 (mean 0-5, L_ε 16-18)."""
    return {4 + p: p for p in range(M.N_MEAN)} | {10 + r: LE0 + r for r in range(3)}


# ---------------------------------------------------------------------------
#  Group-level Laplace (primary) and AGQ (fallback)
# ---------------------------------------------------------------------------
def group_laplace(psi, g: M.Group, it: M.Items, Q: int, b0=None, want_grad=True):
    beta, D, dD, S, dS = M.unpack(psi)
    Dinv = np.linalg.inv(D)
    E, logw, U, C, dC = eps_nodes(psi, Q)
    mu, A = visit_means(psi, g.T, g.X), design(g.T)
    b, nd = modes(mu, A, E, logw, g.YT, g.YP, it, Dinv, b0)
    phi, _, hess, (pi, g1, g2, g3) = _local(mu, A, b, E, logw, g.YT, g.YP, it, Dinv)
    H = -hess
    sH, ldH = np.linalg.slogdet(H)
    _, ldD = np.linalg.slogdet(D)
    val = phi - 0.5 * ldD - 0.5 * ldH
    if not want_grad:
        return val, b, nd, sH, None
    Ea, D2, D3 = _cumulants(pi, g1, g2, g3, _directions(psi, g.T, g.X, U, dC))
    Ea, D2, D3 = Ea.sum(1), D2.sum(1), D3.sum(1)                     # sum over visits
    Hinv = np.linalg.inv(H)
    dHb = -np.moveaxis(D3[..., :4], -1, 1)                            # (m, 4[b_m], 4, 4)
    trHb = np.einsum("mij,mkji->mk", Hinv, dHb)                       # tr(H⁻¹ ∂H/∂b_k)
    m = b.shape[0]
    grad = np.zeros((m, M.N_PSI))
    for e, p in _visit_param_index().items():
        dH = -D3[..., e]
        cross = D2[..., e]
        db = np.einsum("mij,mj->mi", Hinv, cross)
        dlogH = np.einsum("mij,mji->m", Hinv, dH) + np.einsum("mk,mk->m", trHb, db)
        grad[:, p] = Ea[:, e] - 0.5 * dlogH
    for r in range(M.N_L):
        P = Dinv @ dD[r] @ Dinv
        dH = -np.broadcast_to(P, (m, N_B, N_B))
        cross = b @ P.T
        db = np.einsum("mij,mj->mi", Hinv, cross)
        dlogH = np.einsum("mij,mji->m", Hinv, dH) + np.einsum("mk,mk->m", trHb, db)
        grad[:, M.N_MEAN + r] = (0.5 * np.einsum("md,de,me->m", b, P, b)
                                 - 0.5 * np.trace(Dinv @ dD[r]) - 0.5 * dlogH)
    return val, b, nd, sH, grad


def group_agq(psi, g: M.Group, it: M.Items, Q: int, Gn: int, b0=None, want_grad=True,
              fixed=None, chunk=4096):
    """Adaptive GH over b at (b̂, H⁻¹); gradient of the quadrature sum at fixed placement.

    `fixed` = (b̂, L) freezes the node placement (for tests)."""
    beta, D, dD, S, dS = M.unpack(psi)
    Dinv = np.linalg.inv(D)
    E, logw, U, C, dC = eps_nodes(psi, Q)
    mu, A = visit_means(psi, g.T, g.X), design(g.T)
    m, n = g.T.shape
    if fixed is None:
        b, nd = modes(mu, A, E, logw, g.YT, g.YP, it, Dinv, b0)
        _, _, hess, _ = _local(mu, A, b, E, logw, g.YT, g.YP, it, Dinv)
        Hinv = np.linalg.inv(-np.where(nd[:, None, None], hess, -Dinv))
        L = np.linalg.cholesky(Hinv)
    else:
        b, L = fixed
        nd = np.ones(m, bool)
    Z, lwz = b_nodes(Gn)
    K = Z.shape[0]
    _, ldD = np.linalg.slogdet(D)
    ldL = np.log(np.abs(np.einsum("mii->mi", L))).sum(1)
    Bpts = b[:, None, :] + np.einsum("mij,kj->mki", L, Z)             # (m, K, 4)
    flat = Bpts.reshape(m * K, N_B)
    rep = np.repeat(np.arange(m), K)
    ll_pts = np.empty(m * K)
    ga = np.zeros((m * K, M.N_PSI)) if want_grad else None
    for s0 in range(0, m * K, chunk):
        sl = slice(s0, min(s0 + chunk, m * K))
        r = rep[sl]
        th = _theta(mu[r], A[r], flat[sl], E)
        ll = node_loglik(th, g.YT[r], g.YP[r], it) + logw
        lse = logsumexp(ll, axis=-1)
        ll_pts[sl] = lse.sum(1) - 0.5 * np.einsum("md,de,me->m", flat[sl], Dinv, flat[sl])
        if want_grad:
            pi = np.exp(ll - lse[..., None])
            g1, _, _ = node_derivs(th, g.YT[r], g.YP[r], it)
            Dr = _directions(psi, g.T[r], g.X[r], U, dC)
            a = np.einsum("mnqk,mnqek->mnqe", g1, Dr)
            Ea = np.einsum("mnq,mnqe->me", pi, a)
            for e, p in _visit_param_index().items():
                ga[sl, p] = Ea[:, e]
            for k in range(M.N_L):
                P = Dinv @ dD[k] @ Dinv
                ga[sl, M.N_MEAN + k] = 0.5 * np.einsum("md,de,me->m", flat[sl], P, flat[sl])
    tot = (ll_pts.reshape(m, K) + lwz[None] + 0.5 * (Z ** 2).sum(1)[None])
    lse = logsumexp(tot, axis=1)
    val = lse + ldL - 0.5 * ldD
    if not want_grad:
        return val, b, nd, None
    w = np.exp(tot - lse[:, None])
    grad = np.einsum("mk,mkp->mp", w, ga.reshape(m, K, M.N_PSI))
    for k in range(M.N_L):
        grad[:, M.N_MEAN + k] -= 0.5 * np.trace(Dinv @ dD[k])
    return val, b, nd, grad


# ---------------------------------------------------------------------------
#  Cohort objective and fit
# ---------------------------------------------------------------------------
def negloglik(psi, groups: dict, it: M.Items, Q: int = 10, method: str = "laplace", Gn: int = 3,
              state=None, want_grad=True, report=None):
    total, grad, n_not_nd = 0.0, np.zeros(M.N_PSI), 0
    try:
        for n, g in groups.items():
            b0 = None if state is None else state.get(n)
            if method == "laplace":
                val, b, nd, sH, gr = group_laplace(psi, g, it, Q, b0, want_grad)
                if np.any(sH <= 0):
                    return (1e12, np.zeros(M.N_PSI)) if want_grad else 1e12
            else:
                val, b, nd, gr = group_agq(psi, g, it, Q, Gn, b0, want_grad)
            if not np.all(np.isfinite(val)):
                return (1e12, np.zeros(M.N_PSI)) if want_grad else 1e12
            if state is not None:
                state[n] = b
            n_not_nd += int((~nd).sum())
            total += val.sum()
            if want_grad:
                grad += gr.sum(0)
    except np.linalg.LinAlgError:
        return (1e12, np.zeros(M.N_PSI)) if want_grad else 1e12
    if report is not None:
        report["not_negative_definite"] = n_not_nd
    return (-float(total), -grad) if want_grad else -float(total)


@dataclass
class Fit:
    psi: np.ndarray
    nll: float
    converged: bool
    message: str
    n_iter: int
    n_eval: int
    grad_norm: float
    at_bound: list
    stopped_by_time: bool


class _Timeout(Exception):
    pass


def fit(groups: dict, it: M.Items, Q: int = 10, method: str = "laplace", Gn: int = 3,
        psi0=None, maxiter=1000, max_seconds=None) -> Fit:
    import time
    state: dict = {}
    lo, hi = np.array([b[0] for b in M.BOUNDS]), np.array([b[1] for b in M.BOUNDS])
    psi0 = np.clip(M.default_psi() if psi0 is None else np.asarray(psi0, float), lo, hi)
    t0, count, last = time.time(), [0], {}

    def f(p):
        if max_seconds is not None and time.time() - t0 > max_seconds:
            raise _Timeout
        count[0] += 1
        v, gr = negloglik(p, groups, it, Q, method, Gn, state)
        last.update(psi=p.copy(), nll=v, grad=gr)
        return v, gr
    try:
        res = minimize(f, psi0, jac=True, method="L-BFGS-B", bounds=M.BOUNDS,
                       options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-7, "maxfun": 4 * maxiter})
        x, fun, ok, msg, nit, jac, timed = res.x, float(res.fun), bool(res.success), str(res.message), int(res.nit), res.jac, False
    except _Timeout:
        x, fun, ok, msg, nit, jac, timed = (last["psi"], float(last["nll"]), False,
                                            f"stopped at the {max_seconds}s cap", -1, last["grad"], True)
    at_bound = [nm for nm, v, (a, b) in zip(M.NAMES, x, M.BOUNDS) if abs(v - a) < 1e-6 or abs(v - b) < 1e-6]
    return Fit(x, fun, ok, msg, nit, count[0], float(np.linalg.norm(jac)), at_bound, timed)
