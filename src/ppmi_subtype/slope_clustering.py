"""Clustering patients by their Bayesian TD/PIGD-ratio slope (exploratory).

Building blocks for scripts/34_bayesian_slope_clustering.py:

* total patient slopes from the hierarchical growth model's retained draws,
  ``s_i^(m) = beta1^(m) + b1_i^(m)`` -- the complete slope, not the deviation;
* a one-dimensional Gaussian-mixture comparison over K with a fixed,
  pre-specified selection rule;
* label-invariant co-clustering across posterior draws;
* Silverman's critical-bandwidth bootstrap test of unimodality;
* split-R-hat and effective sample size for single-parameter chains.

Nothing here changes the growth model in `hbayes`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

K_RANGE = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
#  Patient slopes from the posterior
# ---------------------------------------------------------------------------
def total_slope_draws(post) -> np.ndarray:
    """(S, M) draws of beta1 + b1_i, patient order = ``post.patients``.

    Row s uses population slope and random effects from the *same* sweep, so a
    row is one joint posterior draw of every patient's slope.
    """
    if post.random_effects is None:
        raise ValueError("fit_growth_model(..., keep_random_effects=True) is required")
    if post.random_effects.shape[:2] != (post.n_draws, len(post.patients)):
        raise ValueError("random effects are not aligned with draws and patients")
    return post.beta[:, 1][:, None] + post.random_effects[:, :, 1]


def patient_slope_summary(post, data) -> pd.DataFrame:
    """Per-patient posterior summary of the total slope, with follow-up facts."""
    S = total_slope_draws(post)
    if not np.array_equal(np.asarray(post.patients), np.asarray(data.patients)):
        raise ValueError("posterior and data patient orders differ")
    idx = data.patient_index
    n_obs = np.bincount(idx, minlength=data.n_patients)
    t_first = np.full(data.n_patients, np.inf)
    t_last = np.full(data.n_patients, -np.inf)
    np.minimum.at(t_first, idx, data.t)
    np.maximum.at(t_last, idx, data.t)
    return pd.DataFrame({
        "PATNO": np.asarray(data.patients),
        "slope_mean": S.mean(axis=0),
        "slope_median": np.median(S, axis=0),
        "slope_sd": S.std(axis=0, ddof=1),
        "slope_q2.5": np.quantile(S, 0.025, axis=0),
        "slope_q97.5": np.quantile(S, 0.975, axis=0),
        "P_slope_negative": (S < 0).mean(axis=0),
        "n_ratio_obs": n_obs,
        "t_first": t_first,
        "t_last": t_last,
        "followup_years": t_last - t_first,
    })


# ---------------------------------------------------------------------------
#  Gaussian mixtures in one dimension
# ---------------------------------------------------------------------------
@dataclass
class MixtureFit:
    K: int
    bic: float
    loglik: float
    converged: bool
    n_iter: int
    weights: np.ndarray
    means: np.ndarray
    sds: np.ndarray
    sizes: np.ndarray          # patients per component by maximum-probability assignment
    labels: np.ndarray
    probs: np.ndarray
    warnings: str


def min_component_size(n: int) -> int:
    """Pre-specified size floor: a component must hold max(10, 2% of n) patients."""
    return max(10, int(np.ceil(0.02 * n)))


def fit_mixture(x, K: int, seed: int, n_init: int = 20) -> MixtureFit:
    x = np.asarray(x, float).reshape(-1, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        g = GaussianMixture(K, covariance_type="full", n_init=n_init, random_state=seed,
                            max_iter=1000, tol=1e-6, reg_covar=1e-9).fit(x)
    order = np.argsort(g.means_.ravel())                 # components sorted by mean
    probs = g.predict_proba(x)[:, order]
    labels = probs.argmax(axis=1)
    return MixtureFit(
        K=K, bic=float(g.bic(x)), loglik=float(g.score(x) * len(x)),
        converged=bool(g.converged_), n_iter=int(g.n_iter_),
        weights=g.weights_[order], means=g.means_.ravel()[order],
        sds=np.sqrt(g.covariances_.ravel()[order]),
        sizes=np.bincount(labels, minlength=K), labels=labels, probs=probs,
        warnings="; ".join(sorted({str(w.message)[:80] for w in caught})))


@dataclass
class Selection:
    fits: dict
    selected_K: int
    unrestricted_K: int
    delta_bic: float               # BIC(K=1) - best eligible BIC(K>=2); > 0 favours a mixture
    delta_bic_unrestricted: float
    eligible: dict


def select_K(x, seed: int, n_init: int = 20, k_range=K_RANGE) -> Selection:
    """Fit every K and apply the pre-specified rule.

    Eligible fit: converged and every component holds at least
    `min_component_size(n)` patients by maximum-probability assignment. K = 1 is
    always eligible. Selected K = lowest BIC among eligible fits.
    """
    n = len(x)
    floor = min_component_size(n)
    fits = {K: fit_mixture(x, K, seed, n_init) for K in k_range}
    eligible = {K: (K == 1) or (f.converged and f.sizes.min() >= floor) for K, f in fits.items()}
    sel = min((K for K in k_range if eligible[K]), key=lambda K: fits[K].bic)
    unr = min(k_range, key=lambda K: fits[K].bic)
    multi = [K for K in k_range if K >= 2 and eligible[K]]
    best_multi = min((fits[K].bic for K in multi), default=np.nan)
    return Selection(fits, sel, unr,
                     fits[1].bic - best_multi if multi else np.nan,
                     fits[1].bic - min(fits[K].bic for K in k_range if K >= 2),
                     eligible)


def comparison_table(sel: Selection, n: int) -> pd.DataFrame:
    rows = []
    for K, f in sel.fits.items():
        rows.append({"K": K, "bic": f.bic, "delta_bic_vs_K1": sel.fits[1].bic - f.bic,
                     "loglik": f.loglik, "converged": f.converged, "n_iter": f.n_iter,
                     "eligible": sel.eligible[K], "selected": K == sel.selected_K,
                     "min_component_size": int(f.sizes.min()),
                     "size_floor": min_component_size(n),
                     "sizes": "|".join(map(str, f.sizes)),
                     "proportions": "|".join(f"{s / n:.3f}" for s in f.sizes),
                     "weights": "|".join(f"{w:.4f}" for w in f.weights),
                     "means": "|".join(f"{m:.5f}" for m in f.means),
                     "sds": "|".join(f"{s:.5f}" for s in f.sds),
                     "warnings": f.warnings})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Co-clustering
# ---------------------------------------------------------------------------
def coclustering_update(C: np.ndarray, labels: np.ndarray) -> None:
    """Add one partition to the running co-assignment count (label-invariant)."""
    labels = np.asarray(labels)
    C += (labels[:, None] == labels[None, :])


# ---------------------------------------------------------------------------
#  Silverman's test of unimodality
# ---------------------------------------------------------------------------
def n_modes(x, h, grid=None) -> int:
    """Number of local maxima of a Gaussian KDE with bandwidth h (absolute units)."""
    x = np.asarray(x, float)
    if grid is None:
        span = x.max() - x.min()
        grid = np.linspace(x.min() - 3 * h - 0.1 * span, x.max() + 3 * h + 0.1 * span, 2048)
    z = (grid[:, None] - x[None, :]) / h
    dens = np.exp(-0.5 * z * z).sum(axis=1)
    d = np.diff(dens)
    return int(np.sum((d[:-1] > 0) & (d[1:] <= 0)))


def critical_bandwidth(x, k: int = 1, tol: float = 1e-6) -> float:
    """Smallest h for which the KDE has at most k modes (bisection)."""
    x = np.asarray(x, float)
    lo, hi = 1e-6 * x.std(), 2.0 * x.std()
    while n_modes(x, hi) > k:
        hi *= 2
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if n_modes(x, mid) > k:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * x.std():
            break
    return hi


def silverman_test(x, n_boot: int = 500, seed: int = 42) -> dict:
    """Bootstrap p-value for H0: one mode (Silverman 1981, variance-corrected)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    n, h = len(x), critical_bandwidth(x, 1)
    mu, var = x.mean(), x.var()
    exceed = 0
    for _ in range(n_boot):
        xb = rng.choice(x, n, replace=True) + h * rng.standard_normal(n)
        xb = mu + (xb - mu) / np.sqrt(1 + h * h / var)
        exceed += n_modes(xb, h) > 1
    return {"h_crit": float(h), "n_boot": n_boot, "n_exceed": int(exceed),
            "p_value": (1 + exceed) / (n_boot + 1)}


# ---------------------------------------------------------------------------
#  MCMC diagnostics
# ---------------------------------------------------------------------------
def split_rhat(chains: np.ndarray) -> float:
    """Split-R-hat (Gelman et al.) for an array (n_chains, n_draws)."""
    c = np.asarray(chains, float)
    half = c.shape[1] // 2
    s = np.concatenate([c[:, :half], c[:, half:2 * half]], axis=0)
    n = s.shape[1]
    W = s.var(axis=1, ddof=1).mean()
    B = n * s.mean(axis=1).var(ddof=1)
    var_plus = (n - 1) / n * W + B / n
    return float(np.sqrt(var_plus / W)) if W > 0 else np.nan


def ess(chains: np.ndarray) -> float:
    """Effective sample size with Geyer's initial-positive-sequence truncation."""
    c = np.asarray(chains, float)
    m, n = c.shape
    W = c.var(axis=1, ddof=1).mean()
    var_plus = (n - 1) / n * W + c.mean(axis=1).var(ddof=1) if m > 1 else W
    if var_plus <= 0:
        return np.nan
    acov = []
    for chain in c:
        z = chain - chain.mean()
        f = np.fft.rfft(z, 2 * n)
        acov.append(np.fft.irfft(f * np.conj(f))[:n] / n)
    acov = np.mean(acov, axis=0)
    rho = 1 - (W - acov) / var_plus
    rho[0] = 1.0
    total, t = 0.0, 0
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        total += pair
        t += 2
    tau = -1 + 2 * total
    return float(m * n / max(tau, 1e-12))
