"""M2: bivariate item-level graded-response measurement model (Stage 2).

Specification (frozen): docs/cluster_phase/16_STAGE2_INTEGER_SCORE_MODEL_SPEC_20261017.md
(B4, B5, B8, D3). No intermediate state, no mixture, no serial correlation.

Items: graded-response model per subscore s in {T (11 tremor items), P (5 PIGD
items)}, item parameters from `grm` (stage 1, θ ~ N(0, 1) per visit).

Latent: θ_i = (θ_i^T (n), θ_i^P (n)) ~ N(M_i, K_i) with

    M_i = [β0T + β1T t + γT x ; β0P + β1P t + γP x]
    K_i = Z_i D Z_iᵀ + Σ_ε ⊗ I_n,   Z_i rows [1, t, 0, 0] (tremor), [0, 0, 1, t] (PIGD)

i.e. θ_ij^s = β0^s + β1^s t + γ^s x + b_i0^s + b_i1^s t + ε_ij^s, b_i ~ N₄(0, D),
(ε^T, ε^P)_ij ~ N₂(0, Σ_ε). Stage 2 maximises the Laplace approximation to the
marginal likelihood over θ_i with item parameters fixed; the gradient is exact
(as in `grm_ou`: explicit terms plus the mode's implicit dependence through the
third θ-derivative of the GRM log-likelihood).

ψ (19) = (β0T, β1T, γT, β0P, β1P, γP, L (10, row-major lower triangle, log
diagonal), L_ε (3: log, off-diagonal, log)), D = L Lᵀ, Σ_ε = L_ε L_εᵀ.

Integer scores and labels: items are drawn from their GRM category
distributions; T and P are item sums; the label is the unchanged
`subtype.classify` on (T/11, P/5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from . import grm as G

JITTER = 1e-8
N_MEAN, N_L, N_LE = 6, 10, 3
N_PSI = N_MEAN + N_L + N_LE
L_IDX = [(i, j) for i in range(4) for j in range(i + 1)]          # row-major lower triangle
LE_IDX = [(0, 0), (1, 0), (1, 1)]
NAMES = (["beta0_T", "beta1_T", "gamma_T", "beta0_P", "beta1_P", "gamma_P"]
         + [f"L{i}{j}" + ("_log" if i == j else "") for i, j in L_IDX]
         + [f"Le{i}{j}" + ("_log" if i == j else "") for i, j in LE_IDX])
BOUNDS = ([(-10, 10)] * N_MEAN
          + [(-8, 3) if i == j else (-5, 5) for i, j in L_IDX]
          + [(-8, 3) if i == j else (-5, 5) for i, j in LE_IDX])
LABEL_CODES = {"TD": 0, "Indeterminate": 1, "PIGD": 2}


@dataclass(frozen=True)
class Items:
    """Fixed item parameters per subscore."""
    aT: np.ndarray
    tauT: np.ndarray
    aP: np.ndarray
    tauP: np.ndarray

    @property
    def KT(self):
        return len(self.aT)

    @property
    def KP(self):
        return len(self.aP)


# ---------------------------------------------------------------------------
#  Parameters
# ---------------------------------------------------------------------------
def _tri(vals, idx, size):
    L = np.zeros((size, size))
    dL = []
    for v, (i, j) in zip(vals, idx):
        L[i, j] = np.exp(v) if i == j else v
    for v, (i, j) in zip(vals, idx):
        d = np.zeros((size, size))
        d[i, j] = np.exp(v) if i == j else 1.0
        dL.append(d)
    return L, dL


def unpack(psi):
    psi = np.asarray(psi, float)
    beta = psi[:N_MEAN]
    L, dL = _tri(psi[N_MEAN:N_MEAN + N_L], L_IDX, 4)
    Le, dLe = _tri(psi[N_MEAN + N_L:], LE_IDX, 2)
    D, S = L @ L.T, Le @ Le.T
    dD = [d @ L.T + L @ d.T for d in dL]
    dS = [d @ Le.T + Le @ d.T for d in dLe]
    return beta, D, dD, S, dS


def to_psi(beta, D, S) -> np.ndarray:
    def chol_params(C, idx):
        L = np.linalg.cholesky(C)
        return [np.log(L[i, j]) if i == j else L[i, j] for i, j in idx]
    return np.r_[np.asarray(beta, float), chol_params(D, L_IDX), chol_params(S, LE_IDX)]


# ---------------------------------------------------------------------------
#  Design, mean and covariance for a group of patients with n visits
# ---------------------------------------------------------------------------
def mean_design(T, X):
    """(6, m, 2n) columns of M with respect to the six mean parameters."""
    one, zero = np.ones_like(T), np.zeros_like(T)
    cols = [(one, zero), (T, zero), (X, zero), (zero, one), (zero, T), (zero, X)]
    return np.stack([np.concatenate(c, axis=1) for c in cols])


def z_design(T):
    """(m, 2n, 4)."""
    m, n = T.shape
    Z = np.zeros((m, 2 * n, 4))
    Z[:, :n, 0], Z[:, :n, 1] = 1.0, T
    Z[:, n:, 2], Z[:, n:, 3] = 1.0, T
    return Z


def covariance(T, D, S, dD=None, dS=None):
    """K (m, 2n, 2n) and, if derivatives are given, the list of dK."""
    n = T.shape[1]
    Z = z_design(T)
    I = np.eye(n)
    E = np.kron(S, I)
    K = np.einsum("mik,kl,mjl->mij", Z, D, Z) + E + JITTER * np.eye(2 * n)
    if dD is None:
        return K, None
    dK = [np.einsum("mik,kl,mjl->mij", Z, d, Z) for d in dD]
    dK += [np.broadcast_to(np.kron(d, I), K.shape) for d in dS]
    return K, dK


# ---------------------------------------------------------------------------
#  GRM likelihood on the stacked latent vector
# ---------------------------------------------------------------------------
def loglik(f, YT, YP, it: Items):
    """Σ log P(items | θ) per patient: f (m, 2n), YT (m, n, KT), YP (m, n, KP)."""
    n = YT.shape[1]
    return (G.category_logprob(f[:, :n], it.aT, it.tauT, YT).sum((-1, -2))
            + G.category_logprob(f[:, n:], it.aP, it.tauP, YP).sum((-1, -2)))


def derivs(f, YT, YP, it: Items):
    n = YT.shape[1]
    dT = G.theta_derivs(f[:, :n], it.aT, it.tauT, YT)
    dP = G.theta_derivs(f[:, n:], it.aP, it.tauP, YP)
    return tuple(np.concatenate([u, v], axis=1) for u, v in zip(dT, dP))


def _objective(f, M, Kinv, YT, YP, it):
    r = f - M
    return loglik(f, YT, YP, it) - 0.5 * np.einsum("mi,mij,mj->m", r, Kinv, r)


def modes(M, K, YT, YP, it: Items, f0=None, iters=100, tol=1e-10):
    """Posterior mode of θ_i per patient by damped Newton (concave objective)."""
    Kinv = np.linalg.inv(K)
    f = M.copy() if f0 is None else f0.copy()
    cur = _objective(f, M, Kinv, YT, YP, it)
    for _ in range(iters):
        l1, l2, _ = derivs(f, YT, YP, it)
        g = l1 - np.einsum("mij,mj->mi", Kinv, f - M)
        A = Kinv - l2[:, :, None] * np.eye(f.shape[1])
        step = np.linalg.solve(A, g[..., None])[..., 0]
        t = np.ones(len(f))
        new = cur
        for _ls in range(40):
            new = _objective(f + t[:, None] * step, M, Kinv, YT, YP, it)
            bad = ~(new >= cur - 1e-12)
            if not bad.any():
                break
            t = np.where(bad, 0.5 * t, t)
        f = f + t[:, None] * step
        cur = new
        if np.max(np.abs(t[:, None] * step)) < tol:
            break
    return f


# ---------------------------------------------------------------------------
#  Laplace marginal likelihood and exact gradient
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Group:
    T: np.ndarray       # (m, n)
    X: np.ndarray       # (m, n) treatment indicator
    YT: np.ndarray      # (m, n, KT)
    YP: np.ndarray      # (m, n, KP)
    patients: list


def make_groups(times: dict, treat: dict, itemsT: dict, itemsP: dict) -> dict:
    """{n: Group} for patients with equal visit counts (rectangular arrays)."""
    by_n: dict = {}
    for p in sorted(times):
        by_n.setdefault(len(times[p]), []).append(p)
    return {n: Group(np.stack([np.asarray(times[p], float) for p in ps]),
                     np.stack([np.asarray(treat[p], float) for p in ps]),
                     np.stack([np.asarray(itemsT[p], float) for p in ps]),
                     np.stack([np.asarray(itemsP[p], float) for p in ps]), ps)
            for n, ps in by_n.items()}


def group_terms(psi, g: Group, it: Items, f0=None, want_grad=True):
    beta, D, dD, S, dS = unpack(psi)
    Xd = mean_design(g.T, g.X)                                          # (6, m, 2n)
    M = np.einsum("k,kmi->mi", beta, Xd)
    K, dK = covariance(g.T, D, S, dD if want_grad else None, dS if want_grad else None)
    f = modes(M, K, g.YT, g.YP, it, f0)
    Kinv = np.linalg.inv(K)
    l1, l2, l3 = derivs(f, g.YT, g.YP, it)
    d = f.shape[1]
    A = Kinv - l2[:, :, None] * np.eye(d)
    _, ldK = np.linalg.slogdet(K)
    sA, ldA = np.linalg.slogdet(A)
    r = f - M
    lp = loglik(f, g.YT, g.YP, it)
    psi_i = lp - 0.5 * np.einsum("mi,mij,mj->m", r, Kinv, r) - 0.5 * ldK - 0.5 * ldA
    if not want_grad:
        return psi_i, f, None, sA
    Ainv = np.linalg.inv(A)
    dAii_l3 = np.einsum("mii->mi", Ainv) * l3
    alpha = np.einsum("mij,mj->mi", Kinv, r)
    grad = np.zeros(N_PSI)
    for j in range(N_MEAN):
        x = Xd[j]
        Kx = np.einsum("mij,mj->mi", Kinv, x)
        df = np.einsum("mij,mj->mi", Ainv, Kx)
        grad[j] = (np.einsum("mi,mi->m", alpha, x) + 0.5 * np.einsum("mi,mi->m", dAii_l3, df)).sum()
    for k, dKk in enumerate(dK):
        KdK = Kinv @ dKk
        explicit = (0.5 * np.einsum("mi,mij,mj->m", alpha, dKk, alpha)
                    - 0.5 * np.einsum("mii->m", KdK)
                    + 0.5 * np.einsum("mij,mji->m", Ainv, KdK @ Kinv))
        dg = np.einsum("mij,mj->mi", Kinv, np.einsum("mij,mj->mi", dKk, alpha))
        df = np.einsum("mij,mj->mi", Ainv, dg)
        grad[N_MEAN + k] = (explicit + 0.5 * np.einsum("mi,mi->m", dAii_l3, df)).sum()
    return psi_i, f, grad, sA


def negloglik(psi, groups: dict, it: Items, state=None, want_grad=True):
    total, grad = 0.0, np.zeros(N_PSI)
    try:
        for n, g in groups.items():
            f0 = None if state is None else state.get(n)
            psi_i, f, gr, sA = group_terms(psi, g, it, f0, want_grad)
            if np.any(sA <= 0) or not np.all(np.isfinite(psi_i)):
                return (1e12, np.zeros(N_PSI)) if want_grad else 1e12
            if state is not None:
                state[n] = f
            total += psi_i.sum()
            if want_grad:
                grad += gr
    except np.linalg.LinAlgError:
        return (1e12, np.zeros(N_PSI)) if want_grad else 1e12
    return (-float(total), -grad) if want_grad else -float(total)


def default_psi() -> np.ndarray:
    D = np.diag([0.8 ** 2, 0.1 ** 2, 0.8 ** 2, 0.1 ** 2])
    S = np.diag([0.5 ** 2, 0.5 ** 2])
    return to_psi(np.zeros(N_MEAN), D, S)


@dataclass
class Fit:
    psi: np.ndarray
    nll: float
    converged: bool
    message: str
    n_iter: int
    grad_norm: float
    at_bound: list


def fit(groups: dict, it: Items, psi0=None, maxiter=1000) -> Fit:
    state: dict = {}
    lo, hi = np.array([b[0] for b in BOUNDS]), np.array([b[1] for b in BOUNDS])
    psi0 = np.clip(default_psi() if psi0 is None else np.asarray(psi0, float), lo, hi)
    res = minimize(negloglik, psi0, args=(groups, it, state), jac=True, method="L-BFGS-B",
                   bounds=BOUNDS, options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-7,
                                           "maxfun": 4 * maxiter})
    at_bound = [nm for nm, v, (a, b) in zip(NAMES, res.x, BOUNDS)
                if abs(v - a) < 1e-6 or abs(v - b) < 1e-6]
    return Fit(res.x, float(res.fun), bool(res.success), str(res.message), int(res.nit),
               float(np.linalg.norm(res.jac)), at_bound)


def hessian(psi, groups: dict, it: Items, h=1e-4) -> np.ndarray:
    """Central finite difference of the analytic gradient of the negative log-likelihood."""
    H = np.zeros((N_PSI, N_PSI))
    for k in range(N_PSI):
        e = np.zeros(N_PSI)
        e[k] = h
        gp = negloglik(psi + e, groups, it)[1]
        gm = negloglik(psi - e, groups, it)[1]
        H[:, k] = (gp - gm) / (2 * h)
    return 0.5 * (H + H.T)


# ---------------------------------------------------------------------------
#  One patient: Laplace, reference integrals, conditional b, sampler
# ---------------------------------------------------------------------------
def one_patient(t, x, YT, YP, psi):
    g = Group(np.asarray(t, float)[None], np.asarray(x, float)[None],
              np.asarray(YT, float)[None], np.asarray(YP, float)[None], [0])
    beta, D, _, S, _ = unpack(psi)
    M = np.einsum("k,kmi->mi", beta, mean_design(g.T, g.X))[0]
    K = covariance(g.T, D, S)[0][0]
    return g, M, K


def laplace_one(t, x, YT, YP, psi, it: Items):
    g, *_ = one_patient(t, x, YT, YP, psi)
    return float(group_terms(psi, g, it, want_grad=False)[0][0])


def mode_and_cov(t, x, YT, YP, psi, it: Items):
    g, M, K = one_patient(t, x, YT, YP, psi)
    f = modes(M[None], K[None], g.YT, g.YP, it)[0]
    _, l2, _ = derivs(f[None], g.YT, g.YP, it)
    A = np.linalg.inv(K) - np.diag(l2[0])
    return f, np.linalg.inv(A), M, K


def _log_mvn(F, M, K):
    d = len(M)
    C = np.linalg.cholesky(K)
    z = np.linalg.solve(C, (F - M).T)
    return -0.5 * (z ** 2).sum(0) - np.log(np.diag(C)).sum() - 0.5 * d * np.log(2 * np.pi)


def agq_one(t, x, YT, YP, psi, it: Items, nodes=7):
    """Adaptive Gauss-Hermite over the 2n-vector θ (small n only)."""
    g, M, K = one_patient(t, x, YT, YP, psi)
    f, C_post, *_ = mode_and_cov(t, x, YT, YP, psi, it)
    C = np.linalg.cholesky(C_post)
    xq, wq = G.gh_nodes(nodes)
    d = len(f)
    grids = np.meshgrid(*[np.arange(nodes)] * d, indexing="ij")
    idx = np.stack([q.ravel() for q in grids], 1)
    U = xq[idx]
    F = f + U @ C.T
    ll = loglik(F, np.broadcast_to(g.YT, (len(F),) + g.YT.shape[1:]),
                np.broadcast_to(g.YP, (len(F),) + g.YP.shape[1:]), it)
    vals = (np.log(wq[idx]).sum(1) + ll + _log_mvn(F, M, K) + 0.5 * (U ** 2).sum(1)
            + 0.5 * d * np.log(2 * np.pi) + np.log(np.diag(C)).sum())
    mx = vals.max()
    return float(mx + np.log(np.exp(vals - mx).sum()))


def is_one(t, x, YT, YP, psi, it: Items, rng, draws=20000, df=5):
    """Importance sampling with a multivariate t proposal at the Laplace mode; (log Z, MC SE)."""
    from scipy import stats as sstats
    g, M, K = one_patient(t, x, YT, YP, psi)
    f, C_post, *_ = mode_and_cov(t, x, YT, YP, psi, it)
    prop = sstats.multivariate_t(loc=f, shape=C_post, df=df)
    F = prop.rvs(size=draws, random_state=rng)
    F = np.atleast_2d(F)
    ll = loglik(F, np.broadcast_to(g.YT, (draws,) + g.YT.shape[1:]),
                np.broadcast_to(g.YP, (draws,) + g.YP.shape[1:]), it)
    lw = ll + _log_mvn(F, M, K) - prop.logpdf(F)
    mx = lw.max()
    w = np.exp(lw - mx)
    est = mx + np.log(w.mean())
    se = float(w.std(ddof=1) / (w.mean() * np.sqrt(draws)))
    return float(est), se


def conditional_b(f, t, x, psi):
    """Mean and covariance of b_i given θ_i (exact Gaussian conditional)."""
    T = np.asarray(t, float)[None]
    beta, D, _, S, _ = unpack(psi)
    M = np.einsum("k,kmi->mi", beta, mean_design(T, np.asarray(x, float)[None]))[0]
    K = covariance(T, D, S)[0][0]
    Z = z_design(T)[0]
    G_ = D @ Z.T @ np.linalg.inv(K)
    return G_ @ (np.asarray(f) - M), D - G_ @ Z @ D


def elliptical_slice(f, M, Kchol, logl, rng):
    """One elliptical slice sampling update (Murray, Adams & MacKay 2010)."""
    nu = Kchol @ rng.standard_normal(len(f))
    y = logl(f) + np.log(rng.random())
    th = rng.uniform(0, 2 * np.pi)
    lo, hi = th - 2 * np.pi, th
    while True:
        prop = M + (f - M) * np.cos(th) + nu * np.sin(th)
        if logl(prop) > y:
            return prop
        if th < 0:
            lo = th
        else:
            hi = th
        th = rng.uniform(lo, hi)


def expected_sums(f, n, it: Items):
    """Σ_j E[T_ij | θ] and Σ_j E[P_ij | θ] for θ = f (2n,)."""
    def esum(theta, a, tau):
        F = 1 / (1 + np.exp(-a[:, None] * (np.asarray(theta)[:, None, None] - tau)))   # (n, K, 4)
        return F.sum((-1, -2)).sum()
    return esum(f[:n], it.aT, it.tauT), esum(f[n:], it.aP, it.tauP)


# ---------------------------------------------------------------------------
#  Simulation, integer scores and labels
# ---------------------------------------------------------------------------
def simulate_latent(rng, T, X, psi):
    """θ (m, 2n) for a group: b ~ N(0, D), ε ~ N(0, Σ_ε), at the given design."""
    beta, D, _, S, _ = unpack(psi)
    M = np.einsum("k,kmi->mi", beta, mean_design(T, X))
    m, n = T.shape
    b = rng.standard_normal((m, 4)) @ np.linalg.cholesky(D).T
    e = rng.standard_normal((m, n, 2)) @ np.linalg.cholesky(S).T
    Z = z_design(T)
    return M + np.einsum("mik,mk->mi", Z, b) + np.concatenate([e[..., 0], e[..., 1]], axis=1)


def simulate_items(rng, theta, n, it: Items):
    """Integer item responses (m, n, KT) and (m, n, KP) from θ (m, 2n)."""
    return (G.sample_items(rng, theta[:, :n], it.aT, it.tauT).astype(int),
            G.sample_items(rng, theta[:, n:], it.aP, it.tauP).astype(int))


def labels_from_sums(tsum, psum, scfg) -> np.ndarray:
    """Label codes (0 TD, 1 Indeterminate, 2 PIGD) by the unchanged `subtype.classify`."""
    import pandas as pd
    from .subtype import classify
    t = pd.Series(np.asarray(tsum, float).ravel() / 11.0)
    p = pd.Series(np.asarray(psum, float).ravel() / 5.0)
    lab = classify(t, p, scfg).to_numpy()
    inv = {v: LABEL_CODES[k] for k, v in scfg.labels.items()}
    return np.vectorize(inv.get)(lab).reshape(np.shape(tsum))
