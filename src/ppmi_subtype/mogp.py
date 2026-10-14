"""Multi-output Gaussian process over the tremor and PIGD subscores.

Step 2 of the "model the subscores, not their ratio" direction. Step 1 fitted a
joint linear mixed model and found two things that shape this module: the
population trends differ (tremor +0.023/yr, PIGD +0.066/yr) and the
time-adjusted cross-output correlation is near zero (-0.015 between patients,
+0.019 within). So the interesting question here is **temporal flexibility**,
and the cross-output block is something to measure rather than to rely on.

The model
---------
For patient *i*, stack the two outputs per visit (visit-major, outcome-minor)
so the response for n visits is ``[T_0, P_0, T_1, P_1, ...]``. Then

    K[(j,d),(j',d')] = sum_c B_c[d,d'] k_c(t_j, t_j')  +  Sigma_e[d,d'] delta_jj'

``k_c`` are the scalar temporal components already used by `gpr.py` --- const,
linear, rbf, matern32 --- and ``B_c`` is a 2x2 positive semi-definite
*coregionalization* matrix saying how much of component c each output carries
and how much they share. This is the Linear Model of Coregionalization. With
one component it is the Intrinsic Coregionalization Model.

The mean is ``m_d(t) = alpha_d + beta_d t``, profiled out by generalised least
squares exactly as `gpr.py` does, so the optimiser only ever searches the
covariance parameters and the envelope theorem keeps the gradient correct.

Why a nested pair of models
---------------------------
``coregion="full"`` and ``coregion="diag"`` differ in exactly one thing: whether
the off-diagonals of every ``B_c`` and of ``Sigma_e`` are free or pinned to
zero. Same temporal kernel, same lengthscale, same everything else. So the
comparison between them isolates the cross-output dependence and nothing else.

Two genuinely independent single-output GPs are a *different* restriction ---
they also get their own lengthscale per output --- and are fitted through
`gpr.fit_gpr` rather than here. Reading a difference between that and the joint
model as evidence about coregionalization would confound two changes, which is
why the diagonal variant exists.

Shared hyperparameters
----------------------
As in `gpr.py`: one population GP, hyperparameters estimated once by maximising
the summed marginal likelihood over training patients, each patient's posterior
obtained by conditioning that shared prior on their own history. The median
patient has four visits, which is nowhere near enough to fit a lengthscale per
person.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .gpr import KERNEL_COMPONENTS, SQRT3

#: Output order, fixed throughout.
OUTPUTS = ("tremor", "pigd")

#: Components carrying a lengthscale; the others are scale-only.
_HAS_LENGTHSCALE = {"rbf", "matern32"}

#: Log-scale bounds, matching `gpr._LOG_BOUNDS`. A parameter resting on one is
#: reported rather than accepted silently: it is the signature of a component
#: the data cannot identify.
_LOG_BOUNDS = (np.log(1e-6), np.log(1e6))


# ---------------------------------------------------------------------------
#  Parameter vector
# ---------------------------------------------------------------------------
def parameter_layout(kernel: str, coregion: str) -> list[tuple[str, str]]:
    """Names and roles of the free parameters, in the order the vector holds them.

    Each coregionalization matrix contributes a Cholesky factor: two log
    diagonals always, plus one off-diagonal when `coregion` is "full".
    """
    if kernel not in KERNEL_COMPONENTS:
        raise ValueError(f"unknown kernel {kernel!r}; known: {sorted(KERNEL_COMPONENTS)}")
    if coregion not in ("full", "diag"):
        raise ValueError(f"coregion must be 'full' or 'diag'; got {coregion!r}")

    layout: list[tuple[str, str]] = []
    for c in KERNEL_COMPONENTS[kernel]:
        layout.append((f"B_{c}_L00", "log"))
        layout.append((f"B_{c}_L11", "log"))
        if coregion == "full":
            layout.append((f"B_{c}_L10", "raw"))
        if c in _HAS_LENGTHSCALE:
            layout.append((f"lengthscale_{c}", "log"))
    layout.append(("noise_L00", "log"))
    layout.append(("noise_L11", "log"))
    if coregion == "full":
        layout.append(("noise_L10", "raw"))
    return layout


def _chol2(l00: float, l11: float, l10: float) -> np.ndarray:
    """A 2x2 PSD matrix from its lower Cholesky factor."""
    L = np.array([[l00, 0.0], [l10, l11]])
    return L @ L.T


def unpack(theta: np.ndarray, kernel: str, coregion: str):
    """Split the flat vector into (list of (component, B_c, lengthscale), Sigma_e).

    Every returned matrix is positive semi-definite by construction, so the
    optimiser can roam an unbounded space without producing an invalid kernel.
    """
    layout = parameter_layout(kernel, coregion)
    if len(theta) != len(layout):
        raise ValueError(
            f"{kernel}/{coregion} takes {len(layout)} parameters, got {len(theta)}")

    vals = {}
    for (name, role), raw in zip(layout, theta):
        vals[name] = np.exp(raw) if role == "log" else raw

    comps = []
    for c in KERNEL_COMPONENTS[kernel]:
        B = _chol2(vals[f"B_{c}_L00"], vals[f"B_{c}_L11"],
                   vals.get(f"B_{c}_L10", 0.0))
        ls = vals.get(f"lengthscale_{c}")
        comps.append((c, B, ls))
    sigma_e = _chol2(vals["noise_L00"], vals["noise_L11"], vals.get("noise_L10", 0.0))
    return comps, sigma_e


# ---------------------------------------------------------------------------
#  Temporal components
# ---------------------------------------------------------------------------
def temporal(component: str, t1: np.ndarray, t2: np.ndarray, ls: float | None):
    """The scalar temporal kernel k_c, and its derivative w.r.t. log-lengthscale.

    Returns ``(k, dk_dloglen)``; the second is None for scale-only components.
    These mirror `gpr._component_matrix` exactly so the joint model and the
    single-output model are built from the same functions of time.
    """
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    if component == "const":
        return np.ones((len(t1), len(t2))), None
    if component == "linear":
        return np.outer(t1, t2), None

    d = np.abs(t1[:, None] - t2[None, :])
    if component == "rbf":
        k = np.exp(-0.5 * (d / ls) ** 2)
        return k, k * (d / ls) ** 2            # d/d(log ls)
    if component == "matern32":
        a = SQRT3 * d / ls
        k = (1.0 + a) * np.exp(-a)
        return k, (a ** 2) * np.exp(-a)        # d/d(log ls)
    raise ValueError(f"unknown component {component!r}")


def build_K(times: np.ndarray, comps, sigma_e: np.ndarray) -> np.ndarray:
    """Full (2n, 2n) covariance for one patient, including observation noise.

    ``np.kron(k, B)`` gives visit-major, outcome-minor ordering: entry
    ``(2j+d, 2j'+d')`` equals ``k[j,j'] * B[d,d']``, which is the layout the
    rest of this module assumes.
    """
    t = np.asarray(times, dtype=float)
    n = len(t)
    K = np.zeros((2 * n, 2 * n))
    for component, B, ls in comps:
        k, _ = temporal(component, t, t, ls)
        K += np.kron(k, B)
    K += np.kron(np.eye(n), sigma_e)
    return K


def cross_K(t_hist: np.ndarray, t_star: np.ndarray, comps) -> np.ndarray:
    """Noise-free cross-covariance (2*n_hist, 2*n_star) between history and targets."""
    out = np.zeros((2 * len(t_hist), 2 * len(t_star)))
    for component, B, ls in comps:
        k, _ = temporal(component, t_hist, t_star, ls)
        out += np.kron(k, B)
    return out


# ---------------------------------------------------------------------------
#  Packed training data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Packed:
    """Padded rectangular arrays over patients, as `gpr.pack_sequences` does.

    `T` is (M, nmax) times, `Y` is (M, nmax, 2) responses, `mask` is (M, nmax).
    Padded visits are given an identity block in the covariance so they
    contribute nothing to the likelihood or to any gradient.
    """

    T: np.ndarray
    Y: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray

    @property
    def n_patients(self) -> int:
        return self.T.shape[0]

    @property
    def n_obs(self) -> int:
        return int(2 * self.lengths.sum())


def pack(sequences: list) -> Packed:
    """Pad per-patient ``{'t': (n,), 'y': (n, 2)}`` records into arrays."""
    if not sequences:
        raise ValueError("no training sequences")
    lengths = np.array([len(s["t"]) for s in sequences], dtype=int)
    if np.any(lengths < 1):
        raise ValueError("every training sequence needs at least one observation")

    M, nmax = len(sequences), int(lengths.max())
    T = np.zeros((M, nmax))
    Y = np.zeros((M, nmax, 2))
    mask = np.zeros((M, nmax), dtype=bool)
    for i, s in enumerate(sequences):
        n = lengths[i]
        T[i, :n] = np.asarray(s["t"], dtype=float)
        Y[i, :n] = np.asarray(s["y"], dtype=float).reshape(n, 2)
        mask[i, :n] = True
    return Packed(T=T, Y=Y, mask=mask, lengths=lengths)


def _expand(block: np.ndarray, B: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Batched Kronecker of an (M, n, n) temporal block with a 2x2 output matrix."""
    M, n, _ = block.shape
    out = (block * valid)[:, :, None, :, None] * B[None, None, :, None, :]
    return out.reshape(M, 2 * n, 2 * n)


def _batched_pieces(packed: Packed, comps, sigma_e: np.ndarray, jitter: float):
    """Covariance, its Cholesky, the mean design, and dK/dtheta for the batch."""
    T, mask = packed.T, packed.mask
    M, n = T.shape
    valid = (mask[:, :, None] & mask[:, None, :]).astype(float)

    K = np.zeros((M, 2 * n, 2 * n))
    blocks = []
    for component, B, ls in comps:
        if component == "const":
            k = np.ones((M, n, n))
            dk = None
        elif component == "linear":
            k = T[:, :, None] * T[:, None, :]
            dk = None
        else:
            d = np.abs(T[:, :, None] - T[:, None, :])
            if component == "rbf":
                k = np.exp(-0.5 * (d / ls) ** 2)
                dk = k * (d / ls) ** 2
            else:
                a = SQRT3 * d / ls
                k = (1.0 + a) * np.exp(-a)
                dk = (a ** 2) * np.exp(-a)
        blocks.append((component, B, k, dk))
        K += _expand(k, B, valid)

    eye_n = np.broadcast_to(np.eye(n), (M, n, n))
    K += _expand(eye_n, sigma_e, valid)

    # Real visits get jitter; padded slots get a unit diagonal so each padded
    # block is an identity contributing zero everywhere.
    idx = np.arange(2 * n)
    pad = np.repeat(~mask, 2, axis=1)
    K[:, idx, idx] += np.repeat(mask, 2, axis=1) * jitter + pad

    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return None

    # Mean design: for visit j, the tremor row is [1, t, 0, 0] and the PIGD row
    # [0, 0, 1, t]. Padded rows are zero so they cannot influence the GLS mean.
    X = np.zeros((M, 2 * n, 4))
    m = mask.astype(float)
    X[:, 0::2, 0] = m
    X[:, 0::2, 1] = T * m
    X[:, 1::2, 2] = m
    X[:, 1::2, 3] = T * m
    return L, X, blocks, valid, eye_n


def _dK_list(blocks, sigma_e, comps, coregion, valid, eye_n):
    """dK/dtheta for every free parameter, in `parameter_layout` order.

    Derivatives are taken with respect to the *unconstrained* parameters: a log
    Cholesky diagonal contributes ``dB/dlog l = L dL' + dL L'`` with
    ``dL = l * e``, and a lengthscale enters only through its temporal block.
    """
    grads = []
    E00 = np.array([[1.0, 0.0], [0.0, 0.0]])
    E11 = np.array([[0.0, 0.0], [0.0, 1.0]])
    E10 = np.array([[0.0, 0.0], [1.0, 0.0]])

    def dB(L, which):
        """dB/dparam for B = L L', differentiating the chosen Cholesky entry."""
        dL = {"L00": L[0, 0] * E00, "L11": L[1, 1] * E11, "L10": E10}[which]
        return dL @ L.T + L @ dL.T

    for (component, B, k, dk), (_, _, ls) in zip(blocks, comps):
        L = np.linalg.cholesky(B + 1e-300 * np.eye(2)) if np.all(
            np.linalg.eigvalsh(B) > 0) else np.linalg.cholesky(B + 1e-12 * np.eye(2))
        grads.append(_expand(k, dB(L, "L00"), valid))
        grads.append(_expand(k, dB(L, "L11"), valid))
        if coregion == "full":
            grads.append(_expand(k, dB(L, "L10"), valid))
        if dk is not None:
            grads.append(_expand(dk, B, valid))

    Le = np.linalg.cholesky(sigma_e + 1e-12 * np.eye(2))
    grads.append(_expand(eye_n, dB(Le, "L00"), valid))
    grads.append(_expand(eye_n, dB(Le, "L11"), valid))
    if coregion == "full":
        grads.append(_expand(eye_n, dB(Le, "L10"), valid))
    return grads


def objective(theta: np.ndarray, kernel: str, coregion: str, packed: Packed,
              jitter: float):
    """Negative log marginal likelihood with the mean profiled out, and gradient.

    The mean is the generalised least-squares solution at the current
    hyperparameters. By the envelope theorem the derivative through that
    solution vanishes, so the gradient below differentiates only the explicit
    dependence on theta --- the same argument `gpr._objective` relies on.
    """
    try:
        comps, sigma_e = unpack(theta, kernel, coregion)
        pieces = _batched_pieces(packed, comps, sigma_e, jitter)
        if pieces is None:
            return 1e12, np.zeros_like(theta)
        L, X, blocks, valid, eye_n = pieces

        M, two_n, _ = L.shape
        y = packed.Y.reshape(M, two_n)
        y = y * np.repeat(packed.mask, 2, axis=1)

        Kinv_X = np.linalg.solve(L.transpose(0, 2, 1), np.linalg.solve(L, X))
        Kinv_y = np.linalg.solve(L.transpose(0, 2, 1),
                                 np.linalg.solve(L, y[:, :, None]))[:, :, 0]

        A = np.einsum("pia,pib->ab", X, Kinv_X)
        b = np.einsum("pia,pi->a", X, Kinv_y)
        beta = np.linalg.solve(A, b)

        resid = y - np.einsum("pia,a->pi", X, beta)
        alpha = np.linalg.solve(L.transpose(0, 2, 1),
                                np.linalg.solve(L, resid[:, :, None]))[:, :, 0]

        logdet = 2.0 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum()
        quad = float(np.einsum("pi,pi->", resid, alpha))
        nll = 0.5 * (logdet + quad)
        if not np.isfinite(nll):
            return 1e12, np.zeros_like(theta)

        Kinv = np.linalg.solve(L.transpose(0, 2, 1),
                               np.linalg.solve(L, np.broadcast_to(
                                   np.eye(two_n), L.shape).copy()))
        grad = np.empty(len(theta))
        for p, dK in enumerate(_dK_list(blocks, sigma_e, comps, coregion, valid, eye_n)):
            quad_term = np.einsum("pi,pij,pj->", alpha, dK, alpha)
            trace_term = np.einsum("pij,pji->", Kinv, dK)
            grad[p] = 0.5 * (trace_term - quad_term)
        if not np.all(np.isfinite(grad)):
            return 1e12, np.zeros_like(theta)
        return float(nll), grad
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return 1e12, np.zeros_like(theta)


# ---------------------------------------------------------------------------
#  Fitting
# ---------------------------------------------------------------------------
@dataclass
class MOGPFit:
    """A fitted multi-output GP, with everything needed to reproduce it."""

    kernel: str
    coregion: str
    theta: np.ndarray
    beta: np.ndarray
    nll: float
    converged: bool
    n_restarts: int
    best_restart: int
    restart_nlls: list
    at_bound: list
    n_patients: int
    n_obs: int
    #: Why the winning restart stopped, and how much work it took.
    message: str = ""
    n_iter: int = -1
    n_fev: int = -1
    grad_norm: float = float("nan")

    @property
    def components(self):
        return unpack(self.theta, self.kernel, self.coregion)[0]

    @property
    def sigma_e(self) -> np.ndarray:
        return unpack(self.theta, self.kernel, self.coregion)[1]

    def coregion_correlation(self) -> dict:
        """Cross-output correlation implied by each B_c and by the noise."""
        out = {}
        for component, B, _ in self.components:
            d = np.sqrt(np.clip(np.diag(B), 1e-300, None))
            out[component] = float(B[0, 1] / (d[0] * d[1]))
        d = np.sqrt(np.clip(np.diag(self.sigma_e), 1e-300, None))
        out["noise"] = float(self.sigma_e[0, 1] / (d[0] * d[1]))
        return out

    def parameter_dict(self) -> dict:
        out = {}
        for component, B, ls in self.components:
            out[f"var_{component}_tremor"] = float(B[0, 0])
            out[f"var_{component}_pigd"] = float(B[1, 1])
            out[f"cov_{component}"] = float(B[0, 1])
            if ls is not None:
                out[f"lengthscale_{component}"] = float(ls)
        out["noise_tremor"] = float(self.sigma_e[0, 0])
        out["noise_pigd"] = float(self.sigma_e[1, 1])
        out["noise_cov"] = float(self.sigma_e[0, 1])
        for name, value in zip(("alpha_tremor", "beta_tremor",
                                "alpha_pigd", "beta_pigd"), self.beta):
            out[name] = float(value)
        return out


def initial_theta(kernel: str, coregion: str, packed: Packed,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Starting values from the data's own scale.

    Variances start at half the marginal variance of each output, split between
    components; lengthscales start at a third of the observed time span. Any
    off-diagonal starts at zero, so restart 0 begins from independence and a
    cross-output term has to be earned rather than assumed.
    """
    m = packed.mask
    var = np.array([packed.Y[:, :, d][m].var() for d in range(2)])
    var = np.clip(var, 1e-4, None)
    span = float(packed.T[m].max() - packed.T[m].min()) or 1.0

    comps = KERNEL_COMPONENTS[kernel]
    share = np.sqrt(np.clip(var / max(len(comps), 1), 1e-6, None))

    theta = []
    for c in comps:
        scale = share * (0.3 if c == "linear" else 1.0)
        theta += [np.log(scale[0]), np.log(scale[1])]
        if coregion == "full":
            theta.append(0.0)
        if c in _HAS_LENGTHSCALE:
            theta.append(np.log(span / 3.0))
    theta += [np.log(np.sqrt(var[0] * 0.5)), np.log(np.sqrt(var[1] * 0.5))]
    if coregion == "full":
        theta.append(0.0)

    theta = np.asarray(theta, dtype=float)
    if rng is not None:
        theta = theta + rng.normal(scale=0.35, size=theta.shape)
    return theta


def fit_mogp(packed: Packed, kernel: str = "K1", coregion: str = "full",
             n_restarts: int = 4, max_iter: int = 300, jitter: float = 1e-8,
             seed: int = 42) -> MOGPFit:
    """Maximise the summed marginal likelihood over training patients.

    Restart 0 is the unperturbed start, so a single-restart run is deterministic
    and adding restarts can only improve the optimum found. The selection rule
    is simply the lowest negative log-likelihood; every restart's value is kept
    on the fit so a run can be checked for multimodality afterwards.
    """
    layout = parameter_layout(kernel, coregion)
    bounds = [_LOG_BOUNDS if role == "log" else (-20.0, 20.0) for _, role in layout]
    rng = np.random.default_rng(seed)

    best = None
    nlls = []
    for r in range(max(1, n_restarts)):
        theta0 = initial_theta(kernel, coregion, packed, None if r == 0 else rng)
        theta0 = np.clip(theta0, [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(objective, theta0, args=(kernel, coregion, packed, jitter),
                       jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": max_iter})
        nlls.append(float(res.fun))
        if best is None or res.fun < best.fun:
            best = res

    comps, sigma_e = unpack(best.x, kernel, coregion)
    pieces = _batched_pieces(packed, comps, sigma_e, jitter)
    L, X, *_ = pieces
    M = L.shape[0]
    y = packed.Y.reshape(M, -1) * np.repeat(packed.mask, 2, axis=1)
    Kinv_X = np.linalg.solve(L.transpose(0, 2, 1), np.linalg.solve(L, X))
    Kinv_y = np.linalg.solve(L.transpose(0, 2, 1),
                             np.linalg.solve(L, y[:, :, None]))[:, :, 0]
    beta = np.linalg.solve(np.einsum("pia,pib->ab", X, Kinv_X),
                           np.einsum("pia,pi->a", X, Kinv_y))

    at_bound = [name for (name, role), v, b in zip(layout, best.x, bounds)
                if abs(v - b[0]) < 1e-6 or abs(v - b[1]) < 1e-6]

    return MOGPFit(
        kernel=kernel, coregion=coregion, theta=best.x, beta=beta,
        nll=float(best.fun), converged=bool(best.success),
        n_restarts=max(1, n_restarts), best_restart=int(np.argmin(nlls)),
        restart_nlls=nlls, at_bound=at_bound,
        n_patients=packed.n_patients, n_obs=packed.n_obs,
        message=str(best.message), n_iter=int(best.nit), n_fev=int(best.nfev),
        grad_norm=float(np.linalg.norm(best.jac)) if best.jac is not None else float("nan"),
    )


# ---------------------------------------------------------------------------
#  Prediction
# ---------------------------------------------------------------------------
def _mean_design(t: np.ndarray) -> np.ndarray:
    t = np.atleast_1d(np.asarray(t, dtype=float))
    X = np.zeros((2 * len(t), 4))
    X[0::2, 0] = 1.0
    X[0::2, 1] = t
    X[1::2, 2] = 1.0
    X[1::2, 3] = t
    return X


def posterior(fit: MOGPFit, t_hist: np.ndarray, Y_hist: np.ndarray,
              t_star, include_noise: bool = True, jitter: float = 1e-8):
    """Joint predictive mean and covariance at `t_star`, given a patient history.

    The full 2x2 covariance is returned for a single target, which is what makes
    a subtype probability from the *joint* predictive possible: the ratio of two
    correlated Gaussians needs the correlation.
    """
    t_hist = np.asarray(t_hist, dtype=float)
    Y_hist = np.asarray(Y_hist, dtype=float).reshape(-1, 2)
    t_star = np.atleast_1d(np.asarray(t_star, dtype=float))
    if t_hist.size == 0:
        raise ValueError("a prediction needs at least one historical observation")
    if len(t_hist) != len(Y_hist):
        raise ValueError(f"history has {len(t_hist)} times but {len(Y_hist)} rows")

    comps, sigma_e = unpack(fit.theta, fit.kernel, fit.coregion)
    K = build_K(t_hist, comps, sigma_e)
    K[np.diag_indices_from(K)] += jitter
    L = np.linalg.cholesky(K)

    Ks = cross_K(t_hist, t_star, comps)
    Kss = cross_K(t_star, t_star, comps)

    resid = Y_hist.reshape(-1) - _mean_design(t_hist) @ fit.beta
    sol = np.linalg.solve(L.T, np.linalg.solve(L, resid))
    mean = _mean_design(t_star) @ fit.beta + Ks.T @ sol

    V = np.linalg.solve(L, Ks)
    cov = Kss - V.T @ V
    if include_noise:
        cov = cov + np.kron(np.eye(len(t_star)), sigma_e)

    d = np.diag(cov).copy()
    if np.any(d < 0):
        np.fill_diagonal(cov, np.maximum(d, 0.0))
    return mean, cov


def predict_one(fit: MOGPFit, t_hist: np.ndarray, Y_hist: np.ndarray,
                t_target: float, jitter: float = 1e-8):
    """Predictive mean (2,) and covariance (2, 2) of (tremor, PIGD) at one time."""
    mean, cov = posterior(fit, t_hist, Y_hist, np.array([t_target]), jitter=jitter)
    return mean[:2], cov[:2, :2]
