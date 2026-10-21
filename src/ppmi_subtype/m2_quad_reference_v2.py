"""M2, Amendment 1 (Revision 2) prototype: per-visit quadrature over ε, Laplace / AGQ over b.

Specification: docs/cluster_phase/21_STAGE2_DOC16_AMENDMENT1_PROPOSED_REV2_20261017.md
(§2.1-2.3, §2.5), amending the frozen doc 16. Same model M2 and parameters ψ as
`m2_grm` (B4); only the numerical route to the marginal likelihood differs:

    L_ij(b) = Σ_q w_q g_ij(b, u_q),  θ_ijq = μ_ij + A_ij b + C_ε u_q   (Q × Q Gauss-Hermite)
    log p(Y_i) ≈ Φ(b̂) − ½ log|D| − ½ log|H|,  Φ(b) = Σ_j log L_ij(b) − ½ bᵀD⁻¹b,  H = −∇²Φ(b̂)

(primary, "laplace"), or adaptive Gauss-Hermite over b at (b̂, H⁻¹) with G nodes per
dimension (fallback, "agq"). The exact ψ-gradient of the Laplace form uses the
envelope theorem plus the implicit mode term; the b- and ψ-derivatives of
log L_ij are cumulants of per-node GRM derivatives under the node posterior
π_q ∝ w_q g_q. The AGQ gradient is that of the quadrature sum at fixed node
placement (Rev 2 §2.3).

Log-concavity of the exact per-patient integrand in b is motivation only (Rev 2
§2.1): the quadrature sum need not be log-concave, so the mode search checks the
Hessian, falls back to gradient ascent, and reports patients whose final Hessian
is not negative definite.

Code-only optimisations (2026-10-17, after B0; mathematically equivalent to the
B0-benchmarked version kept verbatim as `m2_quad_reference`, checked by
tests/test_m2_quad.py):
  * fused GRM kernel: per item only the two thresholds bracketing the observed
    category are evaluated (τ_y, τ_{y+1} gathered once per data set, not per node),
    giving log P and its first three θ-derivatives in one pass;
  * the Newton iterate's full local evaluation doubles as its line-search value,
    and the mode's local terms are reused for the Laplace value and gradient;
  * cumulants are built from small per-node moments (directions for b and the
    mean are node-independent), instead of 4 × 4 × 13 per-node tensors;
  * optional process pool over patient chunks (`Objective`), sums unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp

from . import grm as G
from . import m2_grm as M

N_B = 4
N_DIR = 4 + M.N_MEAN + M.N_LE          # b (4), mean (6), L_ε (3)
LE0 = M.N_MEAN + M.N_L                 # ψ index of the first L_ε entry
LOGP_FLOOR = -700.0                    # as grm.category_logprob


# ---------------------------------------------------------------------------
#  Nodes and designs
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
#  Fused GRM kernel on bracketing thresholds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Brackets:
    """Per (patient, visit, item): thresholds bracketing the observed category."""
    a: np.ndarray        # (K,)
    lo: np.ndarray       # (m, n, K)  τ_y   (−inf for y = 0)
    hi: np.ndarray       # (m, n, K)  τ_{y+1} (+inf for y = 4)
    obs: np.ndarray      # (m, n, K)  observed item


def brackets(Y, a, tau) -> Brackets:
    Y = np.asarray(Y, float)
    obs = ~np.isnan(Y)
    y = np.nan_to_num(Y, nan=0).astype(int)
    tp = np.concatenate([np.full((len(a), 1), -np.inf), tau, np.full((len(a), 1), np.inf)], 1)
    k = np.arange(len(a))
    return Brackets(np.asarray(a, float), tp[k, y], tp[k, y + 1], obs)


def _rows(br: Brackets, r):
    return Brackets(br.a, br.lo[r], br.hi[r], br.obs[r])


def grm_terms(theta, br: Brackets, order: int):
    """Σ_items log P(y | θ) and, if order >= 1, its θ-derivatives up to `order` (≤ 3).

    theta (m, n, Q); returns arrays (m, n, Q). Same formulas as grm.category_logprob
    (complement form when the upper bracket logit is positive) and grm.theta_derivs.
    """
    a = br.a
    th = theta[..., None]                                            # (m, n, Q, 1)
    with np.errstate(invalid="ignore", over="ignore"):
        zc = a * (th - br.lo[:, :, None, :])                          # P(Y >= y) logit
        zn = a * (th - br.hi[:, :, None, :])                          # P(Y >= y+1) logit
    obs = br.obs[:, :, None, :]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lc, ln_ = log_expit(zc), log_expit(zn)
        diff_low = lc + np.log1p(-np.exp(ln_ - lc))
        lcm, lnm = log_expit(-zc), log_expit(-zn)
        diff_high = lnm + np.log1p(-np.exp(lcm - lnm))
    logp = np.maximum(np.where(zn > 0, diff_high, diff_low), LOGP_FLOOR)
    logp = np.where(obs, logp, 0.0)
    val = logp.sum(-1)
    if order == 0:
        return (val,)
    Fc, Fn = expit(zc), expit(zn)
    sc, sn = Fc * (1 - Fc), Fn * (1 - Fn)
    P = np.maximum(np.exp(logp), 1e-300)
    l1 = a * (sc - sn) / P
    out = [val, np.where(obs, l1, 0.0).sum(-1)]
    if order >= 2:
        r2 = a ** 2 * (sc * (1 - 2 * Fc) - sn * (1 - 2 * Fn)) / P
        out.append(np.where(obs, r2 - l1 ** 2, 0.0).sum(-1))
    if order >= 3:
        r3 = a ** 3 * (sc * (1 - 6 * Fc + 6 * Fc ** 2) - sn * (1 - 6 * Fn + 6 * Fn ** 2)) / P
        out.append(np.where(obs, r3 - 3 * l1 * r2 + 2 * l1 ** 3, 0.0).sum(-1))
    return tuple(out)


@dataclass(frozen=True)
class Data:
    """A group of patients with equal visit counts, prepared for the kernel."""
    T: np.ndarray
    X: np.ndarray
    A: np.ndarray
    brT: Brackets
    brP: Brackets

    def rows(self, r):
        return Data(self.T[r], self.X[r], self.A[r], _rows(self.brT, r), _rows(self.brP, r))


def prepare(g: M.Group, it: M.Items) -> Data:
    return Data(g.T, g.X, design(g.T), brackets(g.YT, it.aT, it.tauT), brackets(g.YP, it.aP, it.tauP))


def _theta(mu, A, b, E):
    """θ (m, n, Q², 2) at b (m, 4)."""
    return (mu + np.einsum("mnkd,md->mnk", A, b))[:, :, None, :] + E[None, None]


def _nodes(d: Data, mu, b, E, order):
    th = _theta(mu, d.A, b, E)
    tT = grm_terms(th[..., 0], d.brT, order)
    tP = grm_terms(th[..., 1], d.brP, order)
    ll = tT[0] + tP[0]
    ders = tuple(np.stack([u, v], -1) for u, v in zip(tT[1:], tP[1:]))
    return ll, ders


# ---------------------------------------------------------------------------
#  Mode in b
# ---------------------------------------------------------------------------
@dataclass
class Local:
    phi: np.ndarray
    grad: np.ndarray
    hess: np.ndarray
    pi: np.ndarray
    g: tuple             # (g1, g2[, g3]) each (m, n, Q², 2)


def _phi(d: Data, mu, b, E, logw, Dinv):
    ll, _ = _nodes(d, mu, b, E, 0)
    return logsumexp(ll + logw, axis=-1).sum(1) - 0.5 * np.einsum("md,de,me->m", b, Dinv, b)


def _local(d: Data, mu, b, E, logw, Dinv, order=2) -> Local:
    ll, ders = _nodes(d, mu, b, E, order)
    ll = ll + logw
    lse = logsumexp(ll, axis=-1)
    pi = np.exp(ll - lse[..., None])
    g1, g2 = ders[0], ders[1]
    s = np.einsum("mnq,mnqk->mnk", pi, g1)
    V = (np.einsum("mnq,mnqk->mnk", pi, g2)[..., None] * np.eye(2)
         + np.einsum("mnq,mnqk,mnql->mnkl", pi, g1, g1) - s[..., :, None] * s[..., None, :])
    grad = np.einsum("mnkd,mnk->md", d.A, s) - b @ Dinv
    hess = np.einsum("mnkd,mnkl,mnle->mde", d.A, V, d.A) - Dinv
    phi = lse.sum(1) - 0.5 * np.einsum("md,de,me->m", b, Dinv, b)
    return Local(phi, grad, hess, pi, ders)


def _take(loc: Local, r) -> Local:
    return Local(loc.phi[r], loc.grad[r], loc.hess[r], loc.pi[r], tuple(x[r] for x in loc.g))


def _put(loc: Local, r, new: Local):
    loc.phi[r], loc.grad[r], loc.hess[r], loc.pi[r] = new.phi, new.grad, new.hess, new.pi
    for x, y in zip(loc.g, new.g):
        x[r] = y


def modes(d: Data, mu, E, logw, Dinv, b0=None, iters=60, tol=1e-9, order=2):
    """Per-patient mode of Φ by damped Newton (gradient ascent where the Hessian is not ND).

    Same iteration and acceptance rule as the reference: full step first, halving
    for patients whose Φ would decrease. Returns b̂, the ND flag, and the local
    terms at b̂ (to `order` derivatives).
    """
    m = mu.shape[0]
    b = np.zeros((m, N_B)) if b0 is None else b0.copy()
    loc = _local(d, mu, b, E, logw, Dinv, order)
    for _ in range(iters):
        nd = np.all(np.linalg.eigvalsh(loc.hess) < 0, axis=1)
        step = np.where(nd[:, None],
                        np.linalg.solve(-np.where(nd[:, None, None], loc.hess, -np.eye(N_B)),
                                        loc.grad[..., None])[..., 0],
                        loc.grad / (1.0 + np.linalg.norm(loc.grad, axis=1, keepdims=True)))
        trial = _local(d, mu, b + step, E, logw, Dinv, order)            # full step, with derivatives
        t = np.ones(m)
        bad = ~(trial.phi >= loc.phi - 1e-12)
        if bad.any():                                                    # line search on the rest
            r = np.flatnonzero(bad)
            tr = np.full(len(r), 0.5)
            dr = d.rows(r)
            for _ls in range(49):
                val = _phi(dr, mu[r], b[r] + tr[:, None] * step[r], E, logw, Dinv)
                still = ~(val >= loc.phi[r] - 1e-12)
                if not still.any():
                    break
                tr = np.where(still, 0.5 * tr, tr)
            t[r] = tr
            _put(trial, r, _local(dr, mu[r], b[r] + tr[:, None] * step[r], E, logw, Dinv, order))
        b = b + t[:, None] * step
        loc = trial
        if np.max(np.abs(t[:, None] * step)) < tol:
            break
    nd = np.all(np.linalg.eigvalsh(loc.hess) < 0, axis=1)
    return b, nd, loc


# ---------------------------------------------------------------------------
#  Cumulant derivatives from per-node moments
# ---------------------------------------------------------------------------
def _cumulants(pi, g1, g2, g3, DirN, DirL, Db):
    """Per visit: E[a] (13), D2 (b x 13), D3 (b x b x 13) of log L_ij.

    DirN (m, n, 10, 2): node-independent directions (b, mean); DirL (Q², 3, 2):
    node-dependent L_ε directions; Db = DirN[:, :, :4].
    """
    E = lambda x: np.einsum("mnq,mnq...->mn...", pi, x)  # noqa: E731
    a = np.concatenate([np.einsum("mnqk,mnek->mnqe", g1, DirN),
                        np.einsum("mnqk,qek->mnqe", g1, DirL)], axis=-1)        # (m,n,Q,13)
    M0 = E(g1)                                                                  # (m,n,2)
    Ea = E(a)                                                                   # (m,n,13)
    M1 = E(g1[..., :, None] * a[..., None, :])                                  # (m,n,2,13)
    Eg2, Eg3 = E(g2), E(g3)

    def node_dir(Ek, gk):                                                       # E[g_k Dir_ek]
        return np.concatenate([Ek[..., :, None] * np.swapaxes(DirN, -1, -2),
                               np.einsum("mnq,mnqk,qek->mnke", pi, gk, DirL)], axis=-1)
    M2, M3 = node_dir(Eg2, g2), node_dir(Eg3, g3)                               # (m,n,2,13)
    M4 = E(g2[..., :, None] * a[..., None, :])                                  # (m,n,2,13)
    Eg2g1 = E(g2[..., :, None] * g1[..., None, :])                              # (m,n,2,2)
    M5 = np.concatenate([Eg2g1[..., None] * np.swapaxes(DirN, -1, -2)[..., :, None, :],
                         np.einsum("mnq,mnqk,qek,mnql->mnkle", pi, g2, DirL, g1)], axis=-1)
    M6 = E(g1[..., :, None, None] * g1[..., None, :, None] * a[..., None, None, :])  # (m,n,2,2,13)

    Eb = np.einsum("mndk,mnk->mnd", Db, M0)
    Eab = np.einsum("mndk,mnke->mnde", Db, M1)
    ES2 = np.einsum("mndk,mnke->mnde", Db, M2)
    D2 = ES2 + Eab - Eb[..., :, None] * Ea[..., None, :]
    t1 = np.einsum("mndk,mnfk,mnke->mndfe", Db, Db, M3)
    t2 = np.einsum("mndk,mnfk,mnke->mndfe", Db, Db, M4)
    t3 = np.einsum("mndk,mnfl,mnkle->mndfe", Db, Db, M5)
    t4 = np.einsum("mnfk,mndl,mnkle->mndfe", Db, Db, M5)
    t5 = np.einsum("mndk,mnfl,mnkle->mndfe", Db, Db, M6)
    ES2bb, Ebb = ES2[..., :4], Eab[..., :4]
    D3 = (t1 + t2 + t3 + t4 + t5
          - ES2bb[..., None] * Ea[..., None, None, :]
          - ES2[..., :, None, :] * Eb[..., None, :, None]
          - ES2[..., None, :, :] * Eb[..., :, None, None]
          - Ebb[..., None] * Ea[..., None, None, :]
          - Eab[..., :, None, :] * Eb[..., None, :, None]
          - Eab[..., None, :, :] * Eb[..., :, None, None]
          + 2 * Eb[..., :, None, None] * Eb[..., None, :, None] * Ea[..., None, None, :])
    return Ea, D2, D3


def _dirs(psi, d: Data, U, dC):
    """DirN (m, n, 10, 2) and DirL (Q², 3, 2)."""
    m, n = d.T.shape
    DirN = np.zeros((m, n, 10, 2))
    DirN[:, :, :4] = np.swapaxes(d.A, -1, -2)
    Xd = M.mean_design(d.T, d.X)
    for p in range(M.N_MEAN):
        DirN[:, :, 4 + p, 0] = Xd[p][:, :n]
        DirN[:, :, 4 + p, 1] = Xd[p][:, n:]
    DirL = np.stack([U @ dc.T for dc in dC], axis=1)                           # (Q², 3, 2)
    return DirN, DirL


def _visit_param_index():
    """ψ index of direction e >= 4 (mean 0-5, L_ε 16-18)."""
    return {4 + p: p for p in range(M.N_MEAN)} | {10 + r: LE0 + r for r in range(3)}


# ---------------------------------------------------------------------------
#  Group-level Laplace (primary) and AGQ (fallback)
# ---------------------------------------------------------------------------
def _as_data(g, it):
    return g if isinstance(g, Data) else prepare(g, it)


def group_laplace(psi, g, it: M.Items, Q: int, b0=None, want_grad=True):
    d = _as_data(g, it)
    beta, D, dD, S, dS = M.unpack(psi)
    Dinv = np.linalg.inv(D)
    E, logw, U, C, dC = eps_nodes(psi, Q)
    mu = visit_means(psi, d.T, d.X)
    b, nd, loc = modes(d, mu, E, logw, Dinv, b0, order=3 if want_grad else 2)
    H = -loc.hess
    sH, ldH = np.linalg.slogdet(H)
    _, ldD = np.linalg.slogdet(D)
    val = loc.phi - 0.5 * ldD - 0.5 * ldH
    if not want_grad:
        return val, b, nd, sH, None
    DirN, DirL = _dirs(psi, d, U, dC)
    g1, g2, g3 = loc.g
    Ea, D2, D3 = _cumulants(loc.pi, g1, g2, g3, DirN, DirL, DirN[:, :, :4])
    Ea, D2, D3 = Ea.sum(1), D2.sum(1), D3.sum(1)
    Hinv = np.linalg.inv(H)
    dHb = -np.moveaxis(D3[..., :4], -1, 1)
    trHb = np.einsum("mij,mkji->mk", Hinv, dHb)
    m = b.shape[0]
    grad = np.zeros((m, M.N_PSI))
    for e, p in _visit_param_index().items():
        db = np.einsum("mij,mj->mi", Hinv, D2[..., e])
        dlogH = np.einsum("mij,mji->m", Hinv, -D3[..., e]) + np.einsum("mk,mk->m", trHb, db)
        grad[:, p] = Ea[:, e] - 0.5 * dlogH
    for r in range(M.N_L):
        P = Dinv @ dD[r] @ Dinv
        db = np.einsum("mij,mj->mi", Hinv, b @ P.T)
        dlogH = -np.einsum("mij,ji->m", Hinv, P) + np.einsum("mk,mk->m", trHb, db)
        grad[:, M.N_MEAN + r] = (0.5 * np.einsum("md,de,me->m", b, P, b)
                                 - 0.5 * np.trace(Dinv @ dD[r]) - 0.5 * dlogH)
    return val, b, nd, sH, grad


def group_agq(psi, g, it: M.Items, Q: int, Gn: int, b0=None, want_grad=True, fixed=None, chunk=1024):
    """Adaptive GH over b at (b̂, H⁻¹); gradient of the quadrature sum at fixed placement.

    `fixed` = (b̂, L) freezes the node placement (for tests)."""
    d = _as_data(g, it)
    beta, D, dD, S, dS = M.unpack(psi)
    Dinv = np.linalg.inv(D)
    E, logw, U, C, dC = eps_nodes(psi, Q)
    mu = visit_means(psi, d.T, d.X)
    m, n = d.T.shape
    if fixed is None:
        b, nd, loc = modes(d, mu, E, logw, Dinv, b0)
        Hinv = np.linalg.inv(-np.where(nd[:, None, None], loc.hess, -Dinv))
        L = np.linalg.cholesky(Hinv)
    else:
        b, L = fixed
        nd = np.ones(m, bool)
    Z, lwz = b_nodes(Gn)
    K = Z.shape[0]
    _, ldD = np.linalg.slogdet(D)
    ldL = np.log(np.abs(np.einsum("mii->mi", L))).sum(1)
    flat = (b[:, None, :] + np.einsum("mij,kj->mki", L, Z)).reshape(m * K, N_B)
    rep = np.repeat(np.arange(m), K)
    ll_pts = np.empty(m * K)
    ga = np.zeros((m * K, M.N_PSI)) if want_grad else None
    if want_grad:
        DirNall, DirL = _dirs(psi, d, U, dC)
    for s0 in range(0, m * K, chunk):
        sl = slice(s0, min(s0 + chunk, m * K))
        r = rep[sl]
        dr = d.rows(r)
        ll, ders = _nodes(dr, mu[r], flat[sl], E, 1 if want_grad else 0)
        ll = ll + logw
        lse = logsumexp(ll, axis=-1)
        ll_pts[sl] = lse.sum(1) - 0.5 * np.einsum("md,de,me->m", flat[sl], Dinv, flat[sl])
        if want_grad:
            pi = np.exp(ll - lse[..., None])
            g1 = ders[0]
            a = np.concatenate([np.einsum("mnqk,mnek->mnqe", g1, DirNall[r]),
                                np.einsum("mnqk,qek->mnqe", g1, DirL)], axis=-1)
            Ea = np.einsum("mnq,mnqe->me", pi, a)
            for e, p in _visit_param_index().items():
                ga[sl, p] = Ea[:, e]
            for k in range(M.N_L):
                P = Dinv @ dD[k] @ Dinv
                ga[sl, M.N_MEAN + k] = 0.5 * np.einsum("md,de,me->m", flat[sl], P, flat[sl])
    tot = ll_pts.reshape(m, K) + lwz[None] + 0.5 * (Z ** 2).sum(1)[None]
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
#  Cohort objective (serial or a process pool over patient chunks) and fit
# ---------------------------------------------------------------------------
def _chunk_eval(data: Data, psi, it, Q, method, Gn, b0, want_grad):
    if method == "laplace":
        val, b, nd, sH, gr = group_laplace(psi, data, it, Q, b0, want_grad)
        bad = bool(np.any(sH <= 0))
    else:
        val, b, nd, gr = group_agq(psi, data, it, Q, Gn, b0, want_grad)
        bad = False
    bad = bad or not np.all(np.isfinite(val))
    return (float(val.sum()), None if gr is None else gr.sum(0), b, int((~nd).sum()), bad)


_POOL_DATA: dict = {}


def _pool_init(chunks, it):
    _POOL_DATA.update(chunks=chunks, it=it)


def _pool_task(args):
    key, psi, Q, method, Gn, b0, want_grad = args
    try:
        return key, _chunk_eval(_POOL_DATA["chunks"][key], psi, _POOL_DATA["it"], Q, method, Gn, b0, want_grad)
    except np.linalg.LinAlgError:
        return key, (0.0, None, b0, 0, True)


class Objective:
    """−log-likelihood and gradient over the cohort; `jobs > 1` uses a process pool.

    Patients are split into chunks of at most `chunk` patients within each
    equal-visit-count group; the sum is the same as the serial sum up to
    floating-point summation order.
    """

    def __init__(self, groups: dict, it: M.Items, Q: int = 10, method: str = "laplace", Gn: int = 3,
                 jobs: int = 1, chunk: int = 64):
        self.it, self.Q, self.method, self.Gn, self.jobs = it, Q, method, Gn, jobs
        self.chunks = {}
        for n, g in groups.items():
            d = prepare(g, it)
            for s0 in range(0, len(g.patients), chunk):
                self.chunks[(n, s0)] = d.rows(np.arange(s0, min(s0 + chunk, len(g.patients))))
        self.state: dict = {}
        self.pool = None
        if jobs > 1:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            self.pool = ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork"),
                                            initializer=_pool_init, initargs=(self.chunks, it))

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()

    def __call__(self, psi, want_grad=True, report=None):
        psi = np.asarray(psi, float)
        args = [(k, psi, self.Q, self.method, self.Gn, self.state.get(k), want_grad) for k in self.chunks]
        if self.pool is None:
            out = []
            for a in args:
                try:
                    out.append((a[0], _chunk_eval(self.chunks[a[0]], psi, self.it, *a[2:])))
                except np.linalg.LinAlgError:
                    out.append((a[0], (0.0, None, a[5], 0, True)))
        else:
            out = list(self.pool.map(_pool_task, args))
        if any(r[4] for _, r in out):
            return (1e12, np.zeros(M.N_PSI)) if want_grad else 1e12
        total = sum(r[0] for _, r in out)
        for k, r in out:
            self.state[k] = r[2]
        if report is not None:
            report["not_negative_definite"] = sum(r[3] for _, r in out)
        if not want_grad:
            return -float(total)
        return -float(total), -np.sum([r[1] for _, r in out], axis=0)


def negloglik(psi, groups: dict, it: M.Items, Q: int = 10, method: str = "laplace", Gn: int = 3,
              state=None, want_grad=True, report=None):
    """Serial cohort objective (API of the reference); `state` keyed by visit count."""
    total, grad, n_not_nd = 0.0, np.zeros(M.N_PSI), 0
    try:
        for n, g in groups.items():
            b0 = None if state is None else state.get(n)
            val, gr, b, nnd, bad = _chunk_eval(prepare(g, it), psi, it, Q, method, Gn, b0, want_grad)
            if bad:
                return (1e12, np.zeros(M.N_PSI)) if want_grad else 1e12
            if state is not None:
                state[n] = b
            n_not_nd += nnd
            total += val
            if want_grad:
                grad += gr
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
        psi0=None, maxiter=1000, max_seconds=None, jobs: int = 1) -> Fit:
    """L-BFGS-B on the cohort objective; options identical to the reference fit."""
    import time
    obj = Objective(groups, it, Q, method, Gn, jobs)
    lo, hi = np.array([b[0] for b in M.BOUNDS]), np.array([b[1] for b in M.BOUNDS])
    psi0 = np.clip(M.default_psi() if psi0 is None else np.asarray(psi0, float), lo, hi)
    t0, count, last = time.time(), [0], {}

    def f(p):
        if max_seconds is not None and time.time() - t0 > max_seconds:
            raise _Timeout
        count[0] += 1
        v, gr = obj(p)
        last.update(psi=p.copy(), nll=v, grad=gr)
        return v, gr
    try:
        res = minimize(f, psi0, jac=True, method="L-BFGS-B", bounds=M.BOUNDS,
                       options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-7, "maxfun": 4 * maxiter})
        x, fun, ok, msg, nit, jac, timed = (res.x, float(res.fun), bool(res.success), str(res.message),
                                            int(res.nit), res.jac, False)
    except _Timeout:
        x, fun, ok, msg, nit, jac, timed = (last["psi"], float(last["nll"]), False,
                                            f"stopped at the {max_seconds}s cap", -1, last["grad"], True)
    finally:
        obj.close()
    at_bound = [nm for nm, v, (a, b) in zip(M.NAMES, x, M.BOUNDS) if abs(v - a) < 1e-6 or abs(v - b) < 1e-6]
    return Fit(x, fun, ok, msg, nit, count[0], float(np.linalg.norm(jac)), at_bound, timed)
