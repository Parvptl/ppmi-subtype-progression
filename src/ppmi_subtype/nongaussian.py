"""Non-Gaussian, single-mode random effects for the clustering null (Stage 6b).

The Stage 6 nulls draw a patient's random effects (tremor/PIGD intercept and
slope) from N(0, Sigma_b). The real BLUP slopes are strongly right-skewed and
heavy-tailed, which a Gaussian null cannot reproduce. This module provides a
*no-groups* alternative with the same covariance:

    b = L w,    L = chol(Sigma_b),    w_j independent standardized skew-t

Each w_j is an Azzalini skew-t variable (skew parameter delta_j, degrees of
freedom nu), shifted and scaled to mean 0 and variance 1, so Cov(b) = Sigma_b
exactly. Independent unimodal components have a product density with a single
stationary point, and a linear map preserves that, so the joint density of b
has one mode: a continuum with no groups by construction.

Skew-t construction (Azzalini & Capitanio 2003):
    Z = delta |U0| + sqrt(1 - delta^2) U1          (skew-normal)
    T = Z / sqrt(V / nu),  V ~ chi^2_nu
    E[T]   = delta sqrt(nu / pi) Gamma((nu-1)/2) / Gamma(nu/2)        (nu > 1)
    E[T^2] = nu / (nu - 2)                                            (nu > 2)
nu = inf gives the skew-normal; delta = 0 and nu = inf give N(0, 1).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

NU_MIN = 2.5          # variance must be finite; below this the scaling is unstable


def skewt_mean_var(delta: float, nu: float) -> tuple[float, float]:
    """Mean and variance of the unstandardized skew-t T."""
    if not -1.0 < delta < 1.0:
        raise ValueError("delta must lie in (-1, 1)")
    if np.isinf(nu):
        m = delta * np.sqrt(2.0 / np.pi)
        return m, 1.0 - m * m
    if nu <= 2.0:
        raise ValueError("nu must exceed 2 for a finite variance")
    m = delta * np.sqrt(nu / np.pi) * np.exp(gammaln((nu - 1) / 2) - gammaln(nu / 2))
    return m, nu / (nu - 2.0) - m * m


def standardized_skewt(rng: np.random.Generator, delta: float, nu: float, size) -> np.ndarray:
    """Draws with mean 0 and variance 1."""
    u0 = np.abs(rng.standard_normal(size))
    u1 = rng.standard_normal(size)
    z = delta * u0 + np.sqrt(1.0 - delta * delta) * u1
    if not np.isinf(nu):
        z = z / np.sqrt(rng.chisquare(nu, size) / nu)
    m, v = skewt_mean_var(delta, nu)
    return (z - m) / np.sqrt(v)


def skewt_density(x: np.ndarray, delta: float, nu: float) -> np.ndarray:
    """Density of the standardized variable (for the unimodality check)."""
    from scipy.stats import norm, t as student_t
    m, v = skewt_mean_var(delta, nu)
    s = np.sqrt(v)
    y = np.asarray(x) * s + m                      # back to the unstandardized scale
    alpha = delta / np.sqrt(1.0 - delta * delta)
    if np.isinf(nu):
        dens = 2.0 * norm.pdf(y) * norm.cdf(alpha * y)
    else:
        dens = 2.0 * student_t.pdf(y, nu) * student_t.cdf(
            alpha * y * np.sqrt((nu + 1.0) / (nu + y * y)), nu + 1.0)
    return dens * s


def random_effects(rng: np.random.Generator, sigma_b: np.ndarray, deltas, nu,
                   n: int) -> np.ndarray:
    """n draws of b = chol(Sigma_b) w; returns (n, q). Cov(b) = Sigma_b exactly.

    `nu` is one tail parameter shared by every component, or one per component.
    """
    deltas = np.asarray(deltas, float)
    q = sigma_b.shape[0]
    if len(deltas) != q:
        raise ValueError(f"need {q} skew parameters, got {len(deltas)}")
    nus = np.broadcast_to(np.asarray(nu, float), (q,))
    L = np.linalg.cholesky(sigma_b)
    w = np.column_stack([standardized_skewt(rng, d, v, n) for d, v in zip(deltas, nus)])
    return w @ L.T
