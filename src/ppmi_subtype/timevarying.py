"""Does the population progression rate change over follow-up?

Direction 2 of the midterm plan, quoted exactly:

    "Every model here assumes the rate is constant over follow-up. We want to
     compare early, middle and late follow-up directly. This is a hypothesis to
     test - we have not established it."

The model here is Step 1's joint linear mixed model with **one** thing changed:
the population mean is a linear spline in elapsed time instead of a straight
line. Everything else - the bivariate response, the patient random effects, the
residual covariance, the leave-one-patient-out protocol - is unchanged, so the
constant-rate and time-varying models are exactly nested and a difference
between them can only come from the mean basis.

Two questions that are easy to conflate
---------------------------------------
*Does the population rate change with follow-up time?* is a statement about the
**fixed effects** - whether the spline coefficients beyond the first are zero.

*Do patients progress at different rates?* is a statement about the **random
effects** - whether a patient-specific slope has non-zero variance.

A random-slope model answers the second and says nothing about the first. Both
are fitted here, and `random_slopes=True` exists to show that the fixed-effect
result survives letting patients differ, not as evidence for it.

Why a linear spline
-------------------
A spline with knots at fixed times is continuous at the joins, so the fitted
trajectory does not jump, and each coefficient reads directly as a change in
slope at a knot. With knots (k1, k2) and coefficients (b1, b2, b3):

    slope on [0, k1)    = b1
    slope on [k1, k2)   = b1 + b2
    slope on [k2, inf)  = b1 + b2 + b3

so "is the rate constant" is exactly "are b2 and b3 zero", testable as linear
contrasts rather than by comparing separately fitted segments.

Knots are **fixed constants chosen in advance**, never estimated. An estimated
changepoint would have to be re-estimated inside every fold to stay honest, and
selecting one on the full data and then evaluating held out would leak.

Covariance
----------
The marginal covariance for a patient with n visits is

    V = Z Sigma_b Z' + I_n (x) Sigma_e

with Z the random-effect design: ``I_2`` per visit for random intercepts, or
``[I_2, t_j I_2]`` when patient-specific slopes are allowed. Woodbury gives

    V^-1   = R^-1 - R^-1 Z M^-1 Z' R^-1,   R = I_n (x) Sigma_e,
                                           M = Sigma_b^-1 + Z' R^-1 Z
    log|V| = n log|Sigma_e| + log|Sigma_b| + log|M|

which costs a q x q factorisation (q = 2 or 4) however many visits a patient
has. This generalises `jointlmm.vinv_logdet`, whose all-ones Z is the
random-intercept special case; the two are checked against each other in the
tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .jointlmm import OUTCOMES, PackedData, pack, subset  # noqa: F401  (re-exported)

#: Pre-specified knot sets. These are the only ones this module is run with;
#: the first is primary and the rest are the robustness alternatives, fixed in
#: advance so that "robust across cut points" cannot become "search cut points".
KNOT_SETS = {
    "primary_2_5": (2.0, 5.0),
    "alt_1.5_4": (1.5, 4.0),
    "alt_2.5_6": (2.5, 6.0),
    "one_knot_3": (3.0,),
    "constant": (),
}


# ---------------------------------------------------------------------------
#  Mean basis
# ---------------------------------------------------------------------------
def spline_basis(t: np.ndarray, knots: tuple) -> np.ndarray:
    """Linear-spline basis ``[1, t, (t-k1)+, (t-k2)+, ...]``, shape (n, 2+len(knots)).

    With no knots this is the straight line ``[1, t]``, so the constant-rate
    model is this function's zero-knot case rather than a separate code path.
    """
    t = np.asarray(t, dtype=float)
    cols = [np.ones_like(t), t]
    for k in knots:
        cols.append(np.clip(t - float(k), 0.0, None))
    return np.column_stack(cols)


def basis_width(knots: tuple) -> int:
    return 2 + len(knots)


def coefficient_names(knots: tuple) -> list[str]:
    """Names of the fixed effects, outcome-major, in the order beta holds them."""
    names = []
    for outcome in OUTCOMES:
        short = outcome.replace("_score", "")
        names.append(f"{short}_intercept")
        names.append(f"{short}_slope_period1")
        for i in range(len(knots)):
            names.append(f"{short}_slope_change_at_{knots[i]:g}y")
    return names


def period_labels(knots: tuple) -> list[str]:
    if not knots:
        return ["0y+"]
    edges = [0.0, *knots]
    out = [f"{edges[i]:g}-{edges[i + 1]:g}y" for i in range(len(edges) - 1)]
    return out + [f"{edges[-1]:g}y+"]


def design(times: np.ndarray, knots: tuple) -> np.ndarray:
    """Fixed-effect design, (2n, 2p), visit-major rows and outcome-major columns.

    Row ``2j`` is the tremor equation at visit j and row ``2j+1`` the PIGD one,
    matching the stacked response ``[T_0, P_0, T_1, P_1, ...]`` used throughout
    Steps 1 and 2.
    """
    B = spline_basis(times, knots)
    n, p = B.shape
    X = np.zeros((2 * n, 2 * p))
    X[0::2, :p] = B
    X[1::2, p:] = B
    return X


def random_design(times: np.ndarray, random_slopes: bool) -> np.ndarray:
    """Random-effect design Z, (2n, q) with q = 2 or 4.

    Columns are ordered (intercept_tremor, intercept_pigd) and then, when slopes
    are allowed, (slope_tremor, slope_pigd).
    """
    t = np.asarray(times, dtype=float)
    n = len(t)
    q = 4 if random_slopes else 2
    Z = np.zeros((2 * n, q))
    Z[0::2, 0] = 1.0
    Z[1::2, 1] = 1.0
    if random_slopes:
        Z[0::2, 2] = t
        Z[1::2, 3] = t
    return Z


# ---------------------------------------------------------------------------
#  Covariance
# ---------------------------------------------------------------------------
def _chol_from_log(theta: np.ndarray, q: int) -> np.ndarray:
    """A q x q positive-definite matrix from an unconstrained vector.

    The lower Cholesky factor is filled column by column with a log-transformed
    diagonal, so every point the optimiser can reach is a valid covariance.
    """
    L = np.zeros((q, q))
    idx = 0
    for i in range(q):
        for j in range(i + 1):
            L[i, j] = np.exp(theta[idx]) if i == j else theta[idx]
            idx += 1
    return L @ L.T


def _n_chol_params(q: int) -> int:
    return q * (q + 1) // 2


def unpack(theta: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the parameter vector into (Sigma_b, Sigma_e)."""
    nb = _n_chol_params(q)
    if len(theta) != nb + 3:
        raise ValueError(f"expected {nb + 3} parameters for q={q}, got {len(theta)}")
    return _chol_from_log(theta[:nb], q), _chol_from_log(theta[nb:], 2)


def vinv_logdet(times: np.ndarray, sigma_b: np.ndarray, sigma_e: np.ndarray,
                random_slopes: bool):
    """Return (V^-1, log|V|, Z) by Woodbury, for any random-effect design."""
    t = np.asarray(times, dtype=float)
    n = len(t)
    Z = random_design(t, random_slopes)

    S = np.linalg.inv(sigma_e)
    Rinv = np.kron(np.eye(n), S)
    SbInv = np.linalg.inv(sigma_b)
    M = SbInv + Z.T @ Rinv @ Z

    RinvZ = Rinv @ Z
    vinv = Rinv - RinvZ @ np.linalg.solve(M, RinvZ.T)
    vinv = 0.5 * (vinv + vinv.T)

    sign_e, ld_e = np.linalg.slogdet(sigma_e)
    sign_b, ld_b = np.linalg.slogdet(sigma_b)
    sign_m, ld_m = np.linalg.slogdet(M)
    if min(sign_e, sign_b, sign_m) <= 0:
        raise np.linalg.LinAlgError("covariance is not positive definite")
    return vinv, n * ld_e + ld_b + ld_m, Z


# ---------------------------------------------------------------------------
#  Likelihood
# ---------------------------------------------------------------------------
def _batched_vinv(T: np.ndarray, sigma_b, sigma_e, random_slopes):
    """Woodbury inverse and log-determinant for a whole group of patients.

    Every patient in the group has the same number of visits, so the arrays are
    rectangular and the work is one batched solve rather than a Python loop.
    Visit *times* still differ between them, which is why Z and V are batched
    rather than shared. This is the same device `jointlmm` uses, and it is what
    makes a 934-fold refit affordable.
    """
    m, n = T.shape
    q = 4 if random_slopes else 2

    Z = np.zeros((m, 2 * n, q))
    Z[:, 0::2, 0] = 1.0
    Z[:, 1::2, 1] = 1.0
    if random_slopes:
        Z[:, 0::2, 2] = T
        Z[:, 1::2, 3] = T

    S = np.linalg.inv(sigma_e)
    Rinv = np.kron(np.eye(n), S)                       # shared across the group
    SbInv = np.linalg.inv(sigma_b)

    RinvZ = np.einsum("ij,pjk->pik", Rinv, Z)
    M = SbInv[None] + np.einsum("pji,pjk->pik", Z, RinvZ)
    vinv = Rinv[None] - np.einsum("pik,pjk->pij", RinvZ,
                                  np.linalg.solve(M, RinvZ.transpose(0, 2, 1))
                                  .transpose(0, 2, 1))
    vinv = 0.5 * (vinv + vinv.transpose(0, 2, 1))

    sign_e, ld_e = np.linalg.slogdet(sigma_e)
    sign_b, ld_b = np.linalg.slogdet(sigma_b)
    sign_m, ld_m = np.linalg.slogdet(M)
    if sign_e <= 0 or sign_b <= 0 or np.any(sign_m <= 0):
        raise np.linalg.LinAlgError("covariance is not positive definite")
    return vinv, n * ld_e + ld_b + ld_m


def _accumulate(data: PackedData, knots: tuple, sigma_b, sigma_e, random_slopes):
    """Sum the GLS pieces across patients, batched over groups of equal length."""
    p = basis_width(knots)
    XtVX = np.zeros((2 * p, 2 * p))
    XtVy = np.zeros(2 * p)
    ytVy = 0.0
    logdet = 0.0
    n_obs = 0

    for n, (_, T, Y) in data.groups.items():
        vinv, ld = _batched_vinv(T, sigma_b, sigma_e, random_slopes)
        m = T.shape[0]

        B = np.ones((m, n, p))
        B[:, :, 1] = T
        for i, k in enumerate(knots):
            B[:, :, 2 + i] = np.clip(T - float(k), 0.0, None)

        X = np.zeros((m, 2 * n, 2 * p))
        X[:, 0::2, :p] = B
        X[:, 1::2, p:] = B
        y = Y.reshape(m, 2 * n)

        XV = np.einsum("pia,pij->paj", X, vinv)
        XtVX += np.einsum("paj,pjb->ab", XV, X)
        XtVy += np.einsum("paj,pj->a", XV, y)
        ytVy += float(np.einsum("pi,pij,pj->", y, vinv, y))
        logdet += ld.sum() if np.ndim(ld) else m * ld
        n_obs += m * 2 * n
    return XtVX, XtVy, ytVy, logdet, n_obs


def _objective(theta, data, knots, q, random_slopes, reml):
    """-2 log-likelihood with the fixed effects profiled out by GLS."""
    try:
        sigma_b, sigma_e = unpack(theta, q)
        XtVX, XtVy, ytVy, logdet, _ = _accumulate(
            data, knots, sigma_b, sigma_e, random_slopes)
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
class TimeVaryingFit:
    """A fitted model, carrying everything needed to read off period slopes."""

    knots: tuple
    beta: np.ndarray
    cov_beta: np.ndarray
    sigma_b: np.ndarray
    sigma_e: np.ndarray
    neg2ll: float
    converged: bool
    random_slopes: bool
    n_patients: int
    n_obs: int
    message: str

    @property
    def n_params(self) -> int:
        return len(self.beta) + len(self.sigma_b) * (len(self.sigma_b) + 1) // 2 + 3

    def period_slopes(self) -> pd.DataFrame:
        """Slope in each period, per outcome, with a model-based 95% interval.

        The slope in period m is the cumulative sum of the first m slope
        coefficients, so its variance comes from the corresponding contrast of
        the fixed-effect covariance rather than from any one coefficient.
        """
        p = basis_width(self.knots)
        labels = period_labels(self.knots)
        z = 1.959963984540054
        rows = []
        for d, outcome in enumerate(OUTCOMES):
            base = d * p
            for m, label in enumerate(labels):
                c = np.zeros(len(self.beta))
                c[base + 1: base + 2 + m] = 1.0
                est = float(c @ self.beta)
                se = float(np.sqrt(c @ self.cov_beta @ c))
                rows.append({
                    "outcome": outcome.replace("_score", ""), "period": label,
                    "slope": est, "std_error": se,
                    "ci_low": est - z * se, "ci_high": est + z * se,
                })
        return pd.DataFrame(rows)

    def slope_contrasts(self) -> pd.DataFrame:
        """Every pairwise difference between period slopes, with intervals.

        This is the hypothesis test. A constant rate means every one of these
        contrasts is zero; comparing the period slopes' own intervals for
        overlap would be a weaker and less direct check.
        """
        p = basis_width(self.knots)
        labels = period_labels(self.knots)
        z = 1.959963984540054
        rows = []
        for d, outcome in enumerate(OUTCOMES):
            base = d * p
            for a in range(len(labels)):
                for b in range(a + 1, len(labels)):
                    c = np.zeros(len(self.beta))
                    c[base + 2 + a: base + 2 + b] = 1.0
                    est = float(c @ self.beta)
                    se = float(np.sqrt(c @ self.cov_beta @ c))
                    lo, hi = est - z * se, est + z * se
                    rows.append({
                        "outcome": outcome.replace("_score", ""),
                        "contrast": f"{labels[b]} - {labels[a]}",
                        "difference": est, "std_error": se,
                        "ci_low": lo, "ci_high": hi,
                        "excludes_zero": bool((lo < 0) == (hi < 0)),
                    })
        return pd.DataFrame(rows)

    def fixed_effects(self) -> pd.DataFrame:
        z = 1.959963984540054
        se = np.sqrt(np.diag(self.cov_beta))
        return pd.DataFrame({
            "term": coefficient_names(self.knots),
            "estimate": self.beta, "std_error": se,
            "ci_low": self.beta - z * se, "ci_high": self.beta + z * se,
        })


def fit(data: PackedData, knots: tuple = (), random_slopes: bool = False,
        reml: bool = True, max_iter: int = 500) -> TimeVaryingFit:
    """Fit by REML (default) or ML, profiling the fixed effects.

    Starting values come from the empirical between- and within-patient
    covariance, which is close enough that one L-BFGS-B run usually suffices; a
    Nelder-Mead polish follows if it reports failure.
    """
    q = 4 if random_slopes else 2
    means = np.stack([data.responses[p].mean(axis=0) for p in data.patients])
    devs = np.concatenate([data.responses[p] - data.responses[p].mean(axis=0)
                           for p in data.patients])

    sb = np.cov(means.T) + 1e-3 * np.eye(2)
    se = np.cov(devs.T) + 1e-3 * np.eye(2)
    if random_slopes:
        big = np.zeros((4, 4))
        big[:2, :2] = sb
        big[2:, 2:] = 0.01 * np.eye(2)       # slopes start small and deliberately
        sb = big

    def pack_chol(S):
        L = np.linalg.cholesky(S)
        out = []
        for i in range(S.shape[0]):
            for j in range(i + 1):
                out.append(np.log(L[i, i]) if i == j else L[i, j])
        return out

    theta0 = np.asarray(pack_chol(sb) + pack_chol(se), dtype=float)
    args = (data, knots, q, random_slopes, reml)

    res = minimize(_objective, theta0, args=args, method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": 1e-11, "gtol": 1e-7})
    if not res.success:
        res = minimize(_objective, res.x, args=args, method="Nelder-Mead",
                       options={"maxiter": max_iter * 20, "xatol": 1e-9,
                                "fatol": 1e-9})

    sigma_b, sigma_e = unpack(res.x, q)
    XtVX, XtVy, _, _, n_obs = _accumulate(data, knots, sigma_b, sigma_e, random_slopes)
    beta = np.linalg.solve(XtVX, XtVy)

    return TimeVaryingFit(
        knots=tuple(knots), beta=beta, cov_beta=np.linalg.inv(XtVX),
        sigma_b=sigma_b, sigma_e=sigma_e, neg2ll=float(res.fun),
        converged=bool(res.success), random_slopes=random_slopes,
        n_patients=data.n_patients, n_obs=int(n_obs), message=str(res.message),
    )


# ---------------------------------------------------------------------------
#  Prediction
# ---------------------------------------------------------------------------
def blup(fit: TimeVaryingFit, t_hist: np.ndarray, Y_hist: np.ndarray):
    """Posterior mean and covariance of a patient's random effects.

    The only place a held-out patient's own data is used. It sees that patient's
    history and the training fit, never another patient's visits and never the
    target visit.
    """
    t_hist = np.asarray(t_hist, dtype=float)
    if len(t_hist) < 1:
        raise ValueError("a BLUP needs at least one historical visit")
    Y_hist = np.asarray(Y_hist, dtype=float).reshape(len(t_hist), 2)

    vinv, _, Z = vinv_logdet(t_hist, fit.sigma_b, fit.sigma_e, fit.random_slopes)
    resid = Y_hist.reshape(-1) - design(t_hist, fit.knots) @ fit.beta

    b_hat = fit.sigma_b @ Z.T @ vinv @ resid
    post = fit.sigma_b - fit.sigma_b @ Z.T @ vinv @ Z @ fit.sigma_b
    return b_hat, 0.5 * (post + post.T)


def predict(fit: TimeVaryingFit, t_hist: np.ndarray, Y_hist: np.ndarray,
            t_target: float):
    """Predictive mean and SD of (tremor, PIGD) at `t_target`.

    No clipping: the metrics downstream must see the model's actual prediction,
    including any mass below zero, because that is one of the things being
    measured.
    """
    b_hat, post = blup(fit, t_hist, Y_hist)
    z_star = random_design(np.array([float(t_target)]), fit.random_slopes)
    mean = design([float(t_target)], fit.knots) @ fit.beta + z_star @ b_hat
    var = np.diag(z_star @ post @ z_star.T) + np.diag(fit.sigma_e)
    return mean, np.sqrt(np.clip(var, 1e-12, None))


def baseline_predict(name: str, t_hist, Y_hist, t_target: float,
                     fit: TimeVaryingFit):
    """Persistence and patient-mean, scored on exactly the same points."""
    Y_hist = np.asarray(Y_hist, dtype=float).reshape(-1, 2)
    if name == "persistence":
        mean = Y_hist[-1]
    elif name == "patient_mean":
        mean = Y_hist.mean(axis=0)
    elif name == "population":
        mean = design([float(t_target)], fit.knots) @ fit.beta
        return mean, np.sqrt(np.diag(fit.sigma_b)[:2] + np.diag(fit.sigma_e))
    else:
        raise ValueError(f"unknown baseline {name!r}")
    sd = Y_hist.std(axis=0, ddof=1) if len(Y_hist) > 1 else np.full(2, np.nan)
    fallback = np.sqrt(np.diag(fit.sigma_e))
    return mean, np.where(np.isfinite(sd) & (sd > 0), sd, fallback)


# ---------------------------------------------------------------------------
#  Composition control
# ---------------------------------------------------------------------------
def within_patient_rates(frame: pd.DataFrame, knots: tuple,
                         min_interval: float = 0.3) -> pd.DataFrame:
    """Per-visit-pair rate of change, labelled by the period its midpoint sits in.

    This is the control the between-period comparison needs. Periods contain
    different patients -- only those with long follow-up reach the late one --
    so a difference between period means confounds "the rate changed" with "the
    patients who got that far are different". Rates computed inside a patient
    and then compared only across patients present in both periods hold
    composition fixed by construction.
    """
    g = frame.sort_values(["PATNO", "years_from_baseline"]).copy()
    rows = []
    for outcome in OUTCOMES:
        d = g.groupby("PATNO")[outcome].diff()
        dt = g.groupby("PATNO")["years_from_baseline"].diff()
        mid = g["years_from_baseline"] - dt / 2.0
        ok = dt > min_interval
        rows.append(pd.DataFrame({
            "PATNO": g.loc[ok, "PATNO"], "outcome": outcome.replace("_score", ""),
            "rate": (d / dt)[ok], "midpoint": mid[ok],
            "period": pd.cut(mid[ok], [0.0, *knots, np.inf],
                             labels=period_labels(knots), right=False),
        }))
    return pd.concat(rows, ignore_index=True).dropna(subset=["period"])


def paired_period_change(rates: pd.DataFrame, period_a: str, period_b: str,
                         n_boot: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Change in rate from `period_a` to `period_b`, within patients in both.

    Each patient contributes the mean of their own rates in each period, so the
    comparison is paired and the bootstrap resamples patients, not visit pairs.
    """
    rng = np.random.default_rng(seed)
    out = []
    for outcome, g in rates.groupby("outcome"):
        per = g.groupby(["PATNO", "period"], observed=True)["rate"].mean().unstack()
        missing = [p for p in (period_a, period_b) if p not in per]
        if missing:
            # Say so rather than returning nothing: a silently empty frame here
            # would read downstream as "no change found".
            out.append({"outcome": outcome, "from_period": period_a,
                        "to_period": period_b, "n_patients": 0,
                        "delta_rate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                        "excludes_zero": False,
                        "note": f"no observations in {', '.join(map(str, missing))}"})
            continue
        pair = per[[period_a, period_b]].dropna()
        if len(pair) < 15:
            out.append({"outcome": outcome, "from_period": period_a,
                        "to_period": period_b, "n_patients": len(pair),
                        "delta_rate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                        "excludes_zero": False, "note": "too few patients"})
            continue
        diff = (pair[period_b] - pair[period_a]).to_numpy()
        draws = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out.append({"outcome": outcome, "from_period": period_a,
                    "to_period": period_b, "n_patients": len(pair),
                    "delta_rate": float(diff.mean()), "ci_low": float(lo),
                    "ci_high": float(hi),
                    "excludes_zero": bool((lo < 0) == (hi < 0)), "note": ""})
    return pd.DataFrame(out)
