"""Gaussian process regression on the patient's TD/PIGD ratio trajectory.

    R_obs(t) = f(t) + eps,      eps ~ N(0, sigma_n^2)
    f(t) ~ GP(m(t), k(t, t'))
    m(t) = beta_0 + beta_1 t

Posterior at new times, given a history (t, R):

    mu_*    = m(t_*) + K_*^T [K + sigma_n^2 I]^-1 (R - m(t))
    Sigma_* = K_**   - K_*^T [K + sigma_n^2 I]^-1 K_*

computed through a Cholesky factor throughout; the covariance is never
explicitly inverted.

Why K0 exists
-------------
The hierarchical growth model in `hbayes` is *already* a Gaussian process.
Writing out its covariance,

    R_it = (b0 + b0i) + (b1 + b1i) t + eps_it,
    b0i ~ N(0, tau0^2),  b1i ~ N(0, tau1^2),  eps ~ N(0, sigma^2)

    Cov(R_it, R_it') = tau0^2 + tau1^2 t t' + sigma^2 [t = t']

which is a constant kernel plus a linear kernel plus white noise. So K0 is not a
new model: it is the existing one in function-space form, and it is here to be
checked against `hbayes` before any claim is made about the kernels that are
genuinely new. See `growth_model_params`.

That also makes the comparison the project actually cares about a nested one.
K2 is K0 with an RBF term added, so "does a flexible trajectory model beat a
linear one" is the question of whether that one extra term earns its place,
rather than a comparison between two unrelated models.

Shared hyperparameters
----------------------
Each patient is treated as an independent draw from one population GP. The
hyperparameters are estimated once, by maximising the summed marginal
likelihood over training patients, and each patient's posterior then comes from
conditioning that shared prior on their own history. Fitting a separate GP per
patient is not an option here --- the median patient has three finite
observations, and a length scale cannot be estimated from three points.

This mirrors what the hierarchical model already does: population parameters
from the cohort, patient-specific behaviour from the patient's own visits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm

#: Kernel names and the components each is built from, in parameter order.
#: "const" and "linear" carry one variance each; "rbf" and "matern32" carry a
#: variance and a length scale; noise is always last.
KERNEL_COMPONENTS: dict[str, tuple[str, ...]] = {
    "K0": ("const", "linear"),
    "K1": ("const", "rbf"),
    "K2": ("const", "linear", "rbf"),
    "K3": ("const", "linear", "matern32"),
}

_COMPONENT_PARAMS = {
    "const": ("var_const",),
    "linear": ("var_linear",),
    "rbf": ("var_rbf", "lengthscale"),
    "matern32": ("var_matern", "lengthscale"),
}

SQRT3 = np.sqrt(3.0)


def parameter_names(kernel: str) -> list[str]:
    """Hyperparameter names for `kernel`, in the order the vector holds them."""
    if kernel not in KERNEL_COMPONENTS:
        raise ValueError(f"unknown kernel {kernel!r}; known: {sorted(KERNEL_COMPONENTS)}")
    names: list[str] = []
    for component in KERNEL_COMPONENTS[kernel]:
        names.extend(_COMPONENT_PARAMS[component])
    names.append("noise")
    return names


def n_parameters(kernel: str) -> int:
    return len(parameter_names(kernel))


# ---------------------------------------------------------------------------
#  Kernel evaluation
# ---------------------------------------------------------------------------
def _component_matrix(component: str, t1, t2, params) -> np.ndarray:
    """Covariance contributed by one component, without the noise term."""
    if component == "const":
        return params[0] * np.ones((len(t1), len(t2)))
    if component == "linear":
        return params[0] * np.outer(t1, t2)

    r = np.abs(t1[:, None] - t2[None, :])
    if component == "rbf":
        var, ell = params
        return var * np.exp(-0.5 * (r / ell) ** 2)
    if component == "matern32":
        var, ell = params
        a = SQRT3 * r / ell
        return var * (1.0 + a) * np.exp(-a)
    raise ValueError(f"unknown kernel component {component!r}")


def _component_gradients(component: str, t, params) -> list[np.ndarray]:
    """d K / d theta for one component, on the training block (t against t).

    Returned in the same order as `_COMPONENT_PARAMS[component]`, as derivatives
    with respect to the parameters themselves; the caller converts to log-space.
    """
    if component == "const":
        return [np.ones((len(t), len(t)))]
    if component == "linear":
        return [np.outer(t, t)]

    r = np.abs(t[:, None] - t[None, :])
    if component == "rbf":
        var, ell = params
        base = np.exp(-0.5 * (r / ell) ** 2)
        return [base, var * base * (r**2) / ell**3]
    if component == "matern32":
        var, ell = params
        a = SQRT3 * r / ell
        exp_a = np.exp(-a)
        # k = var (1 + a) e^-a ; dk/da = -var a e^-a ; da/dl = -a/l
        return [(1.0 + a) * exp_a, var * (a**2) * exp_a / ell]
    raise ValueError(f"unknown kernel component {component!r}")


def _batched_component(component: str, T, dT, valid, params) -> np.ndarray:
    """One kernel component for a whole batch of padded patients at once.

    `T` is (M, nmax) padded times, `dT` the (M, nmax, nmax) time differences and
    `valid` the mask of entries where both visits are real. Padded entries are
    zeroed here and given a unit diagonal by the caller, so a padded patient
    block is the identity: it factorises cleanly, contributes nothing to the
    log-determinant, and drops out of every trace.
    """
    if component == "const":
        return params[0] * valid
    if component == "linear":
        return params[0] * (T[:, :, None] * T[:, None, :]) * valid
    if component == "rbf":
        var, ell = params
        return var * np.exp(-0.5 * (dT / ell) ** 2) * valid
    if component == "matern32":
        var, ell = params
        a = SQRT3 * np.abs(dT) / ell
        return var * (1.0 + a) * np.exp(-a) * valid
    raise ValueError(f"unknown kernel component {component!r}")


def _batched_component_gradients(component: str, T, dT, valid, params) -> list[np.ndarray]:
    """dK/dtheta for one component, batched. Order matches `_COMPONENT_PARAMS`."""
    if component == "const":
        return [valid.astype(float)]
    if component == "linear":
        return [(T[:, :, None] * T[:, None, :]) * valid]
    if component == "rbf":
        var, ell = params
        r2 = dT**2
        base = np.exp(-0.5 * r2 / ell**2) * valid
        return [base, var * base * r2 / ell**3]
    if component == "matern32":
        var, ell = params
        a = SQRT3 * np.abs(dT) / ell
        exp_a = np.exp(-a) * valid
        return [(1.0 + a) * exp_a, var * (a**2) * exp_a / ell]
    raise ValueError(f"unknown kernel component {component!r}")


def _batched_cholesky_solve(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve K X = B for a batch, given the lower Cholesky factors L of K.

    Forward then back substitution, vectorised across the batch and across the
    right-hand sides. The matrices here are at most a few visits square, so the
    cost is dominated by per-call overhead rather than by arithmetic; doing the
    substitution in `nmax` vectorised steps replaces several thousand tiny
    LAPACK calls with a few dozen array operations.

    Written out rather than delegated because `scipy.linalg.cho_solve` does not
    accept a batch dimension, and the alternative --- forming K^-1 with an
    explicit inverse --- is the thing this module avoids.
    """
    n = L.shape[-1]
    diag = np.einsum("mii->mi", L)

    z = np.empty_like(B)
    for i in range(n):
        acc = B[:, i, :] if i == 0 else B[:, i, :] - np.einsum(
            "mj,mjk->mk", L[:, i, :i], z[:, :i, :]
        )
        z[:, i, :] = acc / diag[:, i, None]

    x = np.empty_like(B)
    for i in range(n - 1, -1, -1):
        acc = z[:, i, :] if i == n - 1 else z[:, i, :] - np.einsum(
            "mj,mjk->mk", L[:, i + 1:, i], x[:, i + 1:, :]
        )
        x[:, i, :] = acc / diag[:, i, None]
    return x


@dataclass(frozen=True)
class Packed:
    """Training sequences padded into rectangular arrays for batched fitting.

    Patients have between two and fourteen finite observations. Padding to the
    longest and masking is the same device `hmm.build_sequences` uses, for the
    same reason: it turns a Python loop over patients into array operations.
    """

    T: np.ndarray            # (M, nmax) times, zero-padded
    Y: np.ndarray            # (M, nmax) ratios, zero-padded
    mask: np.ndarray         # (M, nmax) True where the entry is a real visit
    lengths: np.ndarray      # (M,)

    @property
    def n_patients(self) -> int:
        return self.T.shape[0]

    @property
    def n_obs(self) -> int:
        return int(self.lengths.sum())


def pack_sequences(sequences: list) -> Packed:
    """Pad a list of per-patient (t, y) sequences into rectangular arrays."""
    if not sequences:
        raise ValueError("no training sequences")
    lengths = np.array([len(s["t"]) for s in sequences], dtype=int)
    if np.any(lengths < 1):
        raise ValueError("every training sequence needs at least one observation")

    M, nmax = len(sequences), int(lengths.max())
    T = np.zeros((M, nmax))
    Y = np.zeros((M, nmax))
    mask = np.zeros((M, nmax), dtype=bool)
    for i, seq in enumerate(sequences):
        n = lengths[i]
        T[i, :n] = np.asarray(seq["t"], dtype=float)
        Y[i, :n] = np.asarray(seq["y"], dtype=float)
        mask[i, :n] = True
    return Packed(T=T, Y=Y, mask=mask, lengths=lengths)


def _split_params(kernel: str, theta: np.ndarray) -> tuple[list, float]:
    """Split a flat hyperparameter vector into per-component groups and noise."""
    expected = n_parameters(kernel)
    if len(theta) != expected:
        raise ValueError(
            f"{kernel} takes {expected} hyperparameters {parameter_names(kernel)}, "
            f"got {len(theta)}"
        )
    groups, i = [], 0
    for component in KERNEL_COMPONENTS[kernel]:
        k = len(_COMPONENT_PARAMS[component])
        groups.append(theta[i:i + k])
        i += k
    return groups, float(theta[i])


def kernel_matrix(
    kernel: str,
    t1: np.ndarray,
    t2: np.ndarray,
    theta: np.ndarray,
    include_noise: bool = False,
) -> np.ndarray:
    """Covariance between times `t1` and `t2` under `kernel`.

    The noise term belongs to the *observations*, not to the latent function, so
    it is added only when asked for --- on the training block, and on the
    diagonal of a predictive covariance when predicting a recorded ratio rather
    than the latent trajectory.
    """
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    groups, noise = _split_params(kernel, np.asarray(theta, dtype=float))

    K = np.zeros((len(t1), len(t2)))
    for component, params in zip(KERNEL_COMPONENTS[kernel], groups):
        K += _component_matrix(component, t1, t2, params)

    if include_noise:
        if len(t1) != len(t2):
            raise ValueError("noise can only be added to a square training block")
        K = K + noise * np.eye(len(t1))
    return K


def _kernel_gradients(kernel: str, t: np.ndarray, theta: np.ndarray) -> list[np.ndarray]:
    """d K / d theta_j for every hyperparameter, on the training block."""
    groups, _ = _split_params(kernel, theta)
    grads: list[np.ndarray] = []
    for component, params in zip(KERNEL_COMPONENTS[kernel], groups):
        grads.extend(_component_gradients(component, t, params))
    grads.append(np.eye(len(t)))          # noise
    return grads


# ---------------------------------------------------------------------------
#  Fitting
# ---------------------------------------------------------------------------
@dataclass
class GPRFit:
    """Fitted population GP: kernel hyperparameters and linear mean function."""

    kernel: str
    theta: np.ndarray                   # hyperparameters, natural scale
    beta: np.ndarray                    # (2,) mean-function intercept and slope
    log_marginal_likelihood: float
    n_patients: int
    n_obs: int
    converged: bool
    n_restarts: int
    at_boundary: list[str] = field(default_factory=list)
    #: Optimiser diagnostics for the winning restart, and every restart's
    #: objective, so a fit can be audited for multimodality and for *why* it
    #: stopped rather than only whether scipy reported success.
    message: str = ""
    n_iter: int = -1
    n_fev: int = -1
    restart_objectives: list = field(default_factory=list)

    @property
    def params(self) -> dict[str, float]:
        return dict(zip(parameter_names(self.kernel), (float(v) for v in self.theta)))

    def summary(self) -> dict:
        out = {
            "kernel": self.kernel,
            "n_patients": self.n_patients,
            "n_obs": self.n_obs,
            "loglik": self.log_marginal_likelihood,
            "beta0": float(self.beta[0]),
            "beta1": float(self.beta[1]),
            "converged": self.converged,
            "at_boundary": ",".join(self.at_boundary),
        }
        out.update(self.params)
        return out


def _design(t: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(t), t])


#: Hyperparameters are optimised on the log scale, which keeps every variance
#: and length scale positive without a constrained optimiser. The bounds are
#: wide enough not to bind in normal use; hitting one is reported rather than
#: silently accepted, because a parameter pinned to a bound is the signature of
#: a component the data cannot identify.
_LOG_BOUNDS = (np.log(1e-6), np.log(1e6))


def _batched_pieces(kernel: str, packed: Packed, theta: np.ndarray, jitter: float):
    """Cholesky factors, design matrices and gradient blocks for the whole batch.

    Returns ``None`` if the covariance is not positive definite anywhere, which
    the optimiser sees as an infinite objective and steps away from.
    """
    T, mask = packed.T, packed.mask
    valid = (mask[:, :, None] & mask[:, None, :]).astype(float)
    dT = (T[:, :, None] - T[:, None, :]) * valid

    groups, noise = _split_params(kernel, theta)

    K = np.zeros_like(valid)
    for component, params in zip(KERNEL_COMPONENTS[kernel], groups):
        K += _batched_component(component, T, dT, valid, params)

    # Real visits get the observation noise; padded slots get a unit diagonal,
    # making each padded block an identity that contributes nothing anywhere.
    idx = np.arange(T.shape[1])
    K[:, idx, idx] += mask * (noise + jitter) + (~mask)

    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return None

    X = np.stack([mask.astype(float), T * mask], axis=-1)      # (M, nmax, 2)
    grads = []
    for component, params in zip(KERNEL_COMPONENTS[kernel], groups):
        grads.extend(_batched_component_gradients(component, T, dT, valid, params))
    noise_grad = np.zeros_like(K)
    noise_grad[:, idx, idx] = mask
    grads.append(noise_grad)

    return L, X, grads


def _objective(log_theta: np.ndarray, kernel: str, packed: Packed, jitter: float,
               fit_mean: bool = True):
    """Negative log marginal likelihood over all patients, and its gradient.

    With `fit_mean` the linear mean function is profiled out: for any
    hyperparameters the maximising beta is the generalised least-squares
    estimate, available in closed form. By the envelope theorem the gradient
    with respect to the hyperparameters can then be taken with beta held at
    that optimum, so no derivative of beta is needed.

    With ``fit_mean=False`` the mean is fixed at zero --- the convention most
    introductory treatments of GP regression adopt, on the grounds that the
    kernel can absorb the mean. That is a different model, not a different
    fitting option, and the two are only comparable because everything else
    here is held identical: same data, same kernel, same jitter, same
    normalisation, same optimiser, same analytic gradients. A comparison that
    changed the optimiser as well would confound the model difference with
    convergence quality.
    """
    theta = np.exp(log_theta)
    pieces = _batched_pieces(kernel, packed, theta, jitter)
    if pieces is None:
        return np.inf, np.zeros_like(log_theta)
    L, X, grads = pieces

    if fit_mean:
        Y = packed.Y[:, :, None]
        rhs = np.concatenate([X, Y], axis=-1)                  # solve once for both
        solved = _batched_cholesky_solve(L, rhs)
        Ki_X, Ki_y = solved[:, :, :2], solved[:, :, 2:]

        A = np.einsum("mni,mnj->ij", X, Ki_X)
        b = np.einsum("mni,mnj->i", X, Ki_y)
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.inf, np.zeros_like(log_theta)
        resid = (packed.Y - X @ beta)[:, :, None]
    else:
        resid = packed.Y[:, :, None]

    alpha = _batched_cholesky_solve(L, resid)

    logdet = 2.0 * np.sum(np.log(np.einsum("mii->mi", L)), axis=1)
    quad = np.einsum("mni,mni->m", resid, alpha)
    nll = 0.5 * float(np.sum(quad + logdet + packed.lengths * np.log(2.0 * np.pi)))
    if not np.isfinite(nll):
        return np.inf, np.zeros_like(log_theta)

    # d(-log p)/d theta_j = 0.5 tr(K^-1 dK_j) - 0.5 alpha' dK_j alpha.
    # K^-1 is materialised through the Cholesky factor, never by an explicit
    # inverse; the trace term genuinely needs it.
    eye = np.broadcast_to(np.eye(L.shape[-1]), L.shape)
    K_inv = _batched_cholesky_solve(L, np.ascontiguousarray(eye))
    W = K_inv - alpha @ np.swapaxes(alpha, 1, 2)

    grad = np.array([0.5 * float(np.sum(W * dK)) for dK in grads])
    return nll, grad * theta          # chain rule for the log parameterisation


def _objective_reference(log_theta, kernel: str, sequences: list, jitter: float):
    """Straightforward per-patient version of `_objective`, for tests only.

    Slow, but written directly from the textbook expressions with one Cholesky
    per patient and no padding. The batched implementation is checked against
    this, so a padding or masking mistake cannot hide behind a plausible number.
    """
    theta = np.exp(np.asarray(log_theta, dtype=float))

    A, b, factors = np.zeros((2, 2)), np.zeros(2), []
    for seq in sequences:
        t, y = np.asarray(seq["t"], float), np.asarray(seq["y"], float)
        K = kernel_matrix(kernel, t, t, theta, include_noise=True)
        K[np.diag_indices_from(K)] += jitter
        c = cho_factor(K, lower=True)
        X = _design(t)
        A += X.T @ cho_solve(c, X)
        b += X.T @ cho_solve(c, y)
        factors.append((c, t, y, X))
    beta = np.linalg.solve(A, b)

    nll, grad = 0.0, np.zeros_like(theta)
    for c, t, y, X in factors:
        resid = y - X @ beta
        alpha = cho_solve(c, resid)
        nll += 0.5 * (
            resid @ alpha
            + 2.0 * np.sum(np.log(np.diag(c[0])))
            + len(t) * np.log(2.0 * np.pi)
        )
        W = cho_solve(c, np.eye(len(t))) - np.outer(alpha, alpha)
        for j, dK in enumerate(_kernel_gradients(kernel, t, theta)):
            grad[j] += 0.5 * np.sum(W * dK)

    return nll, grad * theta


def initial_theta(kernel: str, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Starting hyperparameters scaled to the data.

    The variance of the observations is split between the components rather than
    guessed absolutely, so the same starting point is sensible whether the ratio
    is on its raw scale or any other. The length scale starts at two years,
    which is short relative to the median follow-up span and long relative to
    the typical one-year sampling interval.
    """
    var = float(np.var(y)) if len(y) > 1 else 1.0
    var = max(var, 1e-3)
    span = float(np.ptp(t)) if len(t) > 1 else 1.0

    values = []
    for component in KERNEL_COMPONENTS[kernel]:
        if component == "const":
            values.append(0.5 * var)
        elif component == "linear":
            values.append(0.01 * var / max(span**2, 1.0))
        else:
            values.extend([0.25 * var, 2.0])
    values.append(0.25 * var)
    return np.array(values, dtype=float)


def fit_gpr(
    sequences: list,
    kernel: str = "K2",
    n_restarts: int = 4,
    seed: int = 42,
    max_iter: int = 300,
    jitter: float = 1e-8,
    init: np.ndarray | None = None,
    fit_mean: bool = True,
) -> GPRFit:
    """Estimate shared hyperparameters and the mean function by marginal likelihood.

    `sequences` is one entry per training patient, each a dict with ``t`` and
    ``y``. The held-out patient must not appear; this function has no way to
    check that, so callers go through the leave-one-out driver, which does.

    Restart 0 starts from `initial_theta` unperturbed, so a run with
    `n_restarts=1` is the plain fit and extra restarts can only improve on it.
    Later restarts are perturbed deterministically from `seed`.
    """
    if not sequences:
        raise ValueError("no training sequences")
    if kernel not in KERNEL_COMPONENTS:
        raise ValueError(f"unknown kernel {kernel!r}; known: {sorted(KERNEL_COMPONENTS)}")

    all_t = np.concatenate([s["t"] for s in sequences])
    all_y = np.concatenate([s["y"] for s in sequences])
    base = initial_theta(kernel, all_y, all_t) if init is None else np.asarray(init, float)
    packed = pack_sequences(sequences)

    rng = np.random.default_rng(seed)
    best = None
    objectives = []
    for restart in range(max(1, n_restarts)):
        start = np.log(base)
        if restart > 0:
            start = start + rng.normal(0.0, 1.0, size=len(start))
        start = np.clip(start, *_LOG_BOUNDS)

        result = minimize(
            _objective,
            start,
            args=(kernel, packed, jitter, fit_mean),
            jac=True,
            method="L-BFGS-B",
            bounds=[_LOG_BOUNDS] * len(start),
            options={"maxiter": max_iter},
        )
        objectives.append(float(result.fun))
        if not np.isfinite(result.fun):
            continue
        if best is None or result.fun < best.fun:
            best = result

    if best is None:
        raise RuntimeError(f"every restart failed for kernel {kernel}")

    theta = np.exp(best.x)
    beta = (_refit_mean(kernel, packed, theta, jitter) if fit_mean
            else np.zeros(2))

    tol = 1e-6
    at_boundary = [
        name
        for name, value in zip(parameter_names(kernel), best.x)
        if abs(value - _LOG_BOUNDS[0]) < tol or abs(value - _LOG_BOUNDS[1]) < tol
    ]

    return GPRFit(
        kernel=kernel,
        theta=theta,
        beta=beta,
        log_marginal_likelihood=float(-best.fun),
        n_patients=len(sequences),
        n_obs=int(len(all_y)),
        converged=bool(best.success),
        n_restarts=max(1, n_restarts),
        at_boundary=at_boundary,
        message=str(best.message),
        n_iter=int(best.nit),
        n_fev=int(best.nfev),
        restart_objectives=objectives,
    )


def _refit_mean(kernel: str, packed: Packed, theta: np.ndarray, jitter: float) -> np.ndarray:
    """GLS mean function at the fitted hyperparameters.

    The same estimate the objective profiles out, recomputed once at the optimum
    so the returned fit carries it explicitly.
    """
    pieces = _batched_pieces(kernel, packed, theta, jitter)
    if pieces is None:
        raise RuntimeError("covariance is not positive definite at the fitted parameters")
    L, X, _ = pieces

    rhs = np.concatenate([X, packed.Y[:, :, None]], axis=-1)
    solved = _batched_cholesky_solve(L, rhs)
    A = np.einsum("mni,mnj->ij", X, solved[:, :, :2])
    b = np.einsum("mni,mnj->i", X, solved[:, :, 2:])
    return np.linalg.solve(A, b)


# ---------------------------------------------------------------------------
#  Prediction
# ---------------------------------------------------------------------------
def posterior(
    fit: GPRFit,
    t_hist: np.ndarray,
    y_hist: np.ndarray,
    t_star: np.ndarray,
    include_noise: bool = True,
    jitter: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior predictive mean and covariance at `t_star` given a history.

    With `include_noise` the result describes the *next recorded ratio*, which
    is what a held-out comparison scores; without it, the latent trajectory.

    The full covariance is returned, not just the diagonal, so a multi-step
    forecast can be reported as a trajectory band rather than a row of unrelated
    pointwise intervals.
    """
    t_hist = np.asarray(t_hist, dtype=float)
    y_hist = np.asarray(y_hist, dtype=float)
    t_star = np.atleast_1d(np.asarray(t_star, dtype=float))

    if t_hist.size == 0:
        raise ValueError("a prediction needs at least one historical observation")
    if len(t_hist) != len(y_hist):
        raise ValueError(f"history has {len(t_hist)} times but {len(y_hist)} values")

    K = kernel_matrix(fit.kernel, t_hist, t_hist, fit.theta, include_noise=True)
    K[np.diag_indices_from(K)] += jitter
    c = cho_factor(K, lower=True)

    K_star = kernel_matrix(fit.kernel, t_hist, t_star, fit.theta)
    K_ss = kernel_matrix(fit.kernel, t_star, t_star, fit.theta)

    resid = y_hist - _design(t_hist) @ fit.beta
    mean = _design(t_star) @ fit.beta + K_star.T @ cho_solve(c, resid)
    cov = K_ss - K_star.T @ cho_solve(c, K_star)

    if include_noise:
        _, noise = _split_params(fit.kernel, fit.theta)
        cov = cov + noise * np.eye(len(t_star))

    # Rounding can leave a variance a hair below zero where the posterior is
    # almost fully determined. Clip the diagonal rather than let a negative
    # variance reach a square root downstream.
    diag = np.diag(cov).copy()
    if np.any(diag < 0):
        np.fill_diagonal(cov, np.maximum(diag, 0.0))

    return mean, cov


def predict_one(
    fit: GPRFit,
    t_hist: np.ndarray,
    y_hist: np.ndarray,
    t_target: float,
    jitter: float = 1e-8,
) -> tuple[float, float]:
    """Predictive mean and standard deviation of the ratio at one future time."""
    mean, cov = posterior(fit, t_hist, y_hist, np.array([t_target]), jitter=jitter)
    return float(mean[0]), float(np.sqrt(max(cov[0, 0], 0.0)))


def subtype_probabilities(
    mean: float,
    sd: float,
    td_threshold: float = 1.15,
    pigd_threshold: float = 0.90,
) -> dict[str, float]:
    """Subtype probabilities from the predictive distribution of the ratio.

        P(TD)   = P(R >= td_threshold)
        P(Ind)  = P(pigd_threshold < R < td_threshold)
        P(PIGD) = P(R <= pigd_threshold)

    Taken from the whole distribution, not from where the mean happens to fall:
    a prediction of 1.08 with substantial spread carries real probability in all
    three classes, and calling it "Indeterminate" would throw that away.

    A Gaussian predictive puts some mass below zero, which no ratio can reach.
    That mass lands in P(PIGD), whose region is unbounded below, so the three
    still sum to one; the amount is reported by `negative_mass` rather than
    hidden by truncating.
    """
    if sd <= 0:
        # Degenerate predictive: fall back to where the point estimate sits.
        return {
            "TD": float(mean >= td_threshold),
            "Indeterminate": float(pigd_threshold < mean < td_threshold),
            "PIGD": float(mean <= pigd_threshold),
        }

    p_td = float(norm.sf(td_threshold, loc=mean, scale=sd))
    p_pigd = float(norm.cdf(pigd_threshold, loc=mean, scale=sd))
    return {
        "TD": p_td,
        "Indeterminate": max(0.0, 1.0 - p_td - p_pigd),
        "PIGD": p_pigd,
    }


def negative_mass(mean: float, sd: float) -> float:
    """Predictive probability below zero, where no ratio can lie.

    Reported as a diagnostic of how well a Gaussian suits the raw ratio scale.
    """
    return float(norm.cdf(0.0, loc=mean, scale=sd)) if sd > 0 else 0.0


# ---------------------------------------------------------------------------
#  The bridge to the hierarchical growth model
# ---------------------------------------------------------------------------
def growth_model_params(tau0_2: float, tau1_2: float, sigma2: float) -> np.ndarray:
    """K0 hyperparameters equivalent to a hierarchical growth model.

    The random intercept becomes the constant term, the random slope becomes the
    linear term, and the residual becomes the noise:

        Cov(R_it, R_it') = tau0^2 + tau1^2 t t' + sigma^2 [t = t']

    Used by the equivalence check: a K0 posterior built from these values must
    reproduce the growth model's predictive distribution exactly, because they
    are the same distribution written two ways.
    """
    return np.array([float(tau0_2), float(tau1_2), float(sigma2)], dtype=float)


def growth_model_predictive(
    beta: np.ndarray,
    tau0_2: float,
    tau1_2: float,
    sigma2: float,
    t_hist: np.ndarray,
    y_hist: np.ndarray,
    t_target: float,
) -> tuple[float, float]:
    """The growth model's predictive, derived in weight space.

    This is the calculation `hbayes._subject_posterior_draw` performs, with the
    draw replaced by its mean and variance:

        V = (Xh' Xh / sigma^2 + D^-1)^-1
        m = V Xh' (y - Xh beta) / sigma^2
        predictive mean = x_*' (beta + m)
        predictive var  = x_*' V x_* + sigma^2

    Independent of everything above --- no kernel, no Cholesky of a covariance
    over times --- so agreement with `posterior` under K0 is a real check of
    both, not a tautology. It is the standard weight-space/function-space
    duality, and any discrepancy means one of the two is wrong.
    """
    t_hist = np.asarray(t_hist, dtype=float)
    y_hist = np.asarray(y_hist, dtype=float)
    Xh = _design(t_hist)
    x_star = np.array([1.0, float(t_target)])

    D_inv = np.diag([1.0 / tau0_2, 1.0 / tau1_2])
    V = np.linalg.inv(Xh.T @ Xh / sigma2 + D_inv)
    m = V @ (Xh.T @ (y_hist - Xh @ beta)) / sigma2

    mean = float(x_star @ (beta + m))
    var = float(x_star @ V @ x_star + sigma2)
    return mean, np.sqrt(var)
