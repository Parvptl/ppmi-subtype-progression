"""Generic-basis version of the Direction 2 joint LMM, for the Stage 5 audit.

`timevarying.fit` hard-codes a linear-spline mean. The temporal audit needs
other mean structures (smooth splines, quadratic, a grid of single knots), a
different time axis, and patient-bootstrap refits. This module provides exactly
the same model --- bivariate response, random intercepts (optionally slopes),
unstructured 2x2 residual covariance, fixed effects profiled out by GLS --- with
the mean basis passed in as a function. It reuses `timevarying`'s covariance
code unchanged, and `tests/test_timevarying_ext.py` checks that with the
linear-spline basis it reproduces `timevarying.fit` to numerical precision.

Two additions:

* **Patient weights.** A patient-cluster bootstrap replicate is a multinomial
  count per patient. Because patients are independent, the (RE)ML objective of
  the replicate is the weighted sum of per-patient contributions, so a
  bootstrap refit is a weighted fit of the original data --- no duplicated
  rows, identical estimator.
* **ML by default.** REML likelihoods are not comparable between models with
  different fixed effects, so any comparison of temporal structures by
  likelihood (AIC/BIC, LR, knot profile) must use ML. `timevarying.fit`
  defaults to REML, which is right for its variance components but means the
  saved `model_summary.csv` -2logL values must not be compared across knot sets.

Average slope over a period [a, b] is ``(B(b) - B(a)) / (b - a) @ beta`` for
any basis B, so "the rate in a period" is defined the same way for a linear
spline (where it equals the spline's own slope) and for a smooth curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .jointlmm import PackedData
from .timevarying import _batched_vinv, unpack

Basis = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
#  Bases
# ---------------------------------------------------------------------------
# Bases are small callable classes rather than closures so they can be sent to
# worker processes (closures do not pickle).
class linear_spline:
    """Linear spline ``[1, t, (t-k1)+, ...]``; no knots gives the straight line."""

    def __init__(self, knots=()):
        self.knots = tuple(float(k) for k in knots)
        self.name = "pw(" + ",".join(f"{k:g}" for k in self.knots) + ")"

    def __call__(self, t):
        t = np.asarray(t, float)
        return np.stack([np.ones_like(t), t,
                         *[np.clip(t - k, 0, None) for k in self.knots]], axis=-1)


class polynomial:
    def __init__(self, degree: int):
        self.degree = int(degree)
        self.name = f"poly{self.degree}"

    def __call__(self, t):
        t = np.asarray(t, float)
        return np.stack([t ** d for d in range(self.degree + 1)], axis=-1)


class natural_cubic_spline:
    """Natural cubic spline, truncated-power form (Hastie et al., ESL eq. 5.4-5.5).

    `knots` are all knots including the two boundary knots; the basis has
    len(knots) columns: 1, t, and len(knots) - 2 nonlinear terms. Linear beyond
    the boundary knots, so extrapolation is a straight line.
    """

    def __init__(self, knots):
        self.k = np.asarray(sorted(knots), float)
        if len(self.k) < 3:
            raise ValueError("a natural cubic spline needs at least 3 knots")
        self.name = "ns(" + ",".join(f"{x:.2f}" for x in self.k) + ")"

    def _d(self, t, j):
        k, K = self.k, len(self.k)
        return (np.clip(t - k[j], 0, None) ** 3 - np.clip(t - k[K - 1], 0, None) ** 3) / (
            k[K - 1] - k[j])

    def __call__(self, t):
        t = np.asarray(t, float)
        K = len(self.k)
        cols = [np.ones_like(t), t]
        for j in range(K - 2):
            cols.append(self._d(t, j) - self._d(t, K - 2))
        return np.stack(cols, axis=-1)


# ---------------------------------------------------------------------------
#  Fit
# ---------------------------------------------------------------------------
def _design_batch(T, basis, pats=None):
    # A basis flagged `uses_patient` also receives the patient ids, so it can
    # combine follow-up time with a per-patient baseline covariate (e.g. disease
    # duration at enrolment). Random effects stay on follow-up time.
    Bm = basis(T, pats) if getattr(basis, "uses_patient", False) else basis(T)
    m, n, p = Bm.shape
    X = np.zeros((m, 2 * n, 2 * p))
    X[:, 0::2, :p] = Bm
    X[:, 1::2, p:] = Bm
    return X, p


def _accumulate(data: PackedData, basis, sigma_b, sigma_e, random_slopes, weights):
    XtVX = XtVy = None
    ytVy = logdet = 0.0
    n_obs = 0.0
    for n, (pats, T, Y) in data.groups.items():
        w = np.array([weights.get(p, 0.0) for p in pats]) if weights is not None \
            else np.ones(len(pats))
        if not np.any(w):
            continue
        keep = w > 0
        T_, Y_, w_ = T[keep], Y[keep], w[keep]
        vinv, ld = _batched_vinv(T_, sigma_b, sigma_e, random_slopes)
        X, p = _design_batch(T_, basis, np.asarray(pats)[keep])
        y = Y_.reshape(len(T_), 2 * n)
        XV = np.einsum("pia,pij->paj", X, vinv)
        a = np.einsum("p,paj,pjb->ab", w_, XV, X)
        b = np.einsum("p,paj,pj->a", w_, XV, y)
        XtVX = a if XtVX is None else XtVX + a
        XtVy = b if XtVy is None else XtVy + b
        ytVy += float(np.einsum("p,pi,pij,pj->", w_, y, vinv, y))
        ld = np.broadcast_to(ld, w_.shape) if np.ndim(ld) else np.full(len(w_), ld)
        logdet += float(w_ @ ld)
        n_obs += float(w_.sum() * 2 * n)
    return XtVX, XtVy, ytVy, logdet, n_obs


def _objective(theta, data, basis, q, random_slopes, reml, weights):
    try:
        sigma_b, sigma_e = unpack(theta, q)
        XtVX, XtVy, ytVy, logdet, _ = _accumulate(data, basis, sigma_b, sigma_e,
                                                  random_slopes, weights)
        beta = np.linalg.solve(XtVX, XtVy)
        rss = ytVy - XtVy @ beta
        if not np.isfinite(rss) or rss <= 0:
            return 1e12
        value = logdet + rss
        if reml:
            sign, ld = np.linalg.slogdet(XtVX)
            if sign <= 0:
                return 1e12
            value += ld
        return float(value)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return 1e12


@dataclass(frozen=True)
class BasisFit:
    basis_name: str
    beta: np.ndarray
    cov_beta: np.ndarray
    sigma_b: np.ndarray
    sigma_e: np.ndarray
    neg2ll: float
    reml: bool
    converged: bool
    message: str
    random_slopes: bool
    n_obs: float
    p: int

    @property
    def n_params(self) -> int:
        q = len(self.sigma_b)
        return len(self.beta) + q * (q + 1) // 2 + 3

    @property
    def aic(self) -> float:
        if self.reml:
            raise ValueError("AIC from a REML fit is not comparable across mean structures")
        return self.neg2ll + 2 * self.n_params

    def bic(self, n: float) -> float:
        if self.reml:
            raise ValueError("BIC from a REML fit is not comparable across mean structures")
        return self.neg2ll + np.log(n) * self.n_params


def fit_basis(data: PackedData, basis: Basis, random_slopes: bool = False,
              reml: bool = False, weights: dict | None = None, max_iter: int = 500,
              theta0: np.ndarray | None = None) -> BasisFit:
    """Same optimiser and start as `timevarying.fit`; weights for bootstrap refits."""
    q = 4 if random_slopes else 2
    if theta0 is None:
        pats = [p for p in data.patients if weights is None or weights.get(p, 0) > 0]
        means = np.stack([data.responses[p].mean(axis=0) for p in pats])
        devs = np.concatenate([data.responses[p] - data.responses[p].mean(axis=0)
                               for p in pats])
        sb = np.cov(means.T) + 1e-3 * np.eye(2)
        se = np.cov(devs.T) + 1e-3 * np.eye(2)
        if random_slopes:
            big = np.zeros((4, 4))
            big[:2, :2] = sb
            big[2:, 2:] = 0.01 * np.eye(2)
            sb = big

        def pack_chol(S):
            L = np.linalg.cholesky(S)
            return [np.log(L[i, i]) if i == j else L[i, j]
                    for i in range(S.shape[0]) for j in range(i + 1)]
        theta0 = np.asarray(pack_chol(sb) + pack_chol(se), float)

    args = (data, basis, q, random_slopes, reml, weights)
    res = minimize(_objective, theta0, args=args, method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": 1e-11, "gtol": 1e-7})
    if not res.success:
        res = minimize(_objective, res.x, args=args, method="Nelder-Mead",
                       options={"maxiter": max_iter * 20, "xatol": 1e-9, "fatol": 1e-9})
    sigma_b, sigma_e = unpack(res.x, q)
    XtVX, XtVy, _, _, n_obs = _accumulate(data, basis, sigma_b, sigma_e,
                                          random_slopes, weights)
    beta = np.linalg.solve(XtVX, XtVy)
    return BasisFit(basis_name=getattr(basis, "name", "custom"), beta=beta,
                    cov_beta=np.linalg.inv(XtVX), sigma_b=sigma_b, sigma_e=sigma_e,
                    neg2ll=float(res.fun), reml=reml, converged=bool(res.success),
                    message=str(res.message), random_slopes=random_slopes,
                    n_obs=n_obs, p=len(beta) // 2)


# ---------------------------------------------------------------------------
#  Derived quantities
# ---------------------------------------------------------------------------
def average_slope_contrast(basis: Basis, a: float, b: float, outcome: int, p: int):
    """Contrast vector c such that c @ beta is the mean slope over [a, b].

    Only defined for a basis of follow-up time alone; for a patient-dependent
    basis, call that basis's own `followup_contrast` instead.
    """
    d = (basis(np.array([b])) - basis(np.array([a])))[0] / (b - a)
    c = np.zeros(2 * p)
    c[outcome * p:(outcome + 1) * p] = d
    return c


def period_slopes(fit: BasisFit, basis: Basis, periods) -> list[dict]:
    out = []
    for o, name in enumerate(("tremor", "pigd")):
        for a, b in periods:
            c = average_slope_contrast(basis, a, b, o, fit.p)
            out.append({"outcome": name, "period": f"{a:g}-{b:g}y",
                        "slope": float(c @ fit.beta),
                        "wald_se": float(np.sqrt(c @ fit.cov_beta @ c))})
    return out


def bootstrap_weights(patients, n_boot: int, seed: int):
    """Yield multinomial patient-count dicts (one per replicate)."""
    pats = np.asarray(sorted(patients))
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        w = rng.multinomial(len(pats), np.full(len(pats), 1 / len(pats)))
        yield {p: float(c) for p, c in zip(pats, w) if c > 0}
