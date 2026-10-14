"""Joint (bivariate) linear mixed-effects model of the tremor and PIGD subscores.

Step 1 of the "model the two subscores, not their ratio" direction. The point is
not flexibility --- it is to establish an *interpretable* baseline on the two
quantities the ratio compresses, before anything more adaptive is attempted.

The model
---------
For patient *i* at elapsed time *t*, with outcomes indexed d in {T, P}:

    Tremor_it = alpha_T + beta_T t + b_Ti + eps_Tit
    PIGD_it   = alpha_P + beta_P t + b_Pi + eps_Pit

    (b_Ti, b_Pi)     ~ N(0, Sigma_b)   unstructured 2x2, once per patient
    (eps_Tit, eps_Pit) ~ N(0, Sigma_e) unstructured 2x2, independent across visits

Two correlations are estimated and they mean different things. ``Sigma_b``
answers "do patients who sit high on tremor also sit high on PIGD?"; ``Sigma_e``
answers "when a patient's tremor is above their own average at a visit, is their
PIGD above theirs too?". The descriptive pass found these have opposite
character in this data (roughly 0 between patients, positive within), which a
single pooled correlation would have hidden.

No random slopes. The median patient contributes 4 visits over 3.1 years, and
the Step 3 kernel ladder found a patient-specific linear term pinned at its
lower bound in 821 of 821 folds with and without a mean function. A per-patient
slope is not supported here and would trade interpretability for noise.

Why this is hand-written
------------------------
`statsmodels` is not a dependency of this project and its `MixedLM` does not fit
an unstructured bivariate residual covariance without reshaping the problem
anyway. The marginal covariance of this model has a two-block form that inverts
in closed form at 2x2 cost regardless of how many visits a patient has, so the
likelihood is cheap and exact. This matches `hbayes.py`, which also implements
its sampler directly rather than adding a probabilistic programming dependency.

The closed form
---------------
Order a patient's stacked response by visit, then outcome. With n visits,

    V = J_n (x) Sigma_b + I_n (x) Sigma_e

where J_n is the n x n matrix of ones. Then, with S = Sigma_e^-1 and
K = n Sigma_b + Sigma_e,

    V^-1    = I_n (x) S  -  J_n (x) A,      A = K^-1 Sigma_b S   (symmetric)
    log|V|  = log|K| + (n-1) log|Sigma_e|

Both are verified against dense linear algebra in the test suite. Grouping
patients by visit count means V^-1 is built once per distinct n, not once per
patient, which is what makes a 934-fold leave-one-patient-out run tractable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

#: Column order of the bivariate response, fixed everywhere in this module.
OUTCOMES = ("tremor_score", "pigd_score")

#: Fixed-effect order: intercept and slope for tremor, then for PIGD.
BETA_NAMES = ("alpha_tremor", "beta_tremor", "alpha_pigd", "beta_pigd")


# ---------------------------------------------------------------------------
#  Covariance algebra
# ---------------------------------------------------------------------------
def _unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map 6 unconstrained reals to (Sigma_b, Sigma_e), both positive definite.

    Each 2x2 covariance is parameterised by the lower Cholesky factor with a
    log-transformed diagonal, so the optimiser works on an unbounded space and
    every iterate is a valid covariance by construction.
    """
    out = []
    for i in (0, 3):
        L = np.array([[np.exp(theta[i]), 0.0], [theta[i + 1], np.exp(theta[i + 2])]])
        out.append(L @ L.T)
    return out[0], out[1]


def _pack(sigma_b: np.ndarray, sigma_e: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_unpack`, for starting values and round-trip tests."""
    theta = []
    for S in (sigma_b, sigma_e):
        L = np.linalg.cholesky(S)
        theta += [np.log(L[0, 0]), L[1, 0], np.log(L[1, 1])]
    return np.asarray(theta, dtype=float)


def vinv_logdet(n: int, sigma_b: np.ndarray, sigma_e: np.ndarray):
    """Return (V^-1, log|V|) for a patient with `n` visits.

    V^-1 is returned dense (2n x 2n) because every downstream use multiplies it
    by a design matrix; the saving that matters is that it is *built* from 2x2
    operations and reused across every patient with the same visit count.
    """
    S = np.linalg.inv(sigma_e)
    K = n * sigma_b + sigma_e
    A = np.linalg.solve(K, sigma_b) @ S
    A = 0.5 * (A + A.T)  # symmetric in exact arithmetic; enforce it numerically

    vinv = np.kron(np.eye(n), S) - np.kron(np.ones((n, n)), A)
    sign_k, ld_k = np.linalg.slogdet(K)
    sign_e, ld_e = np.linalg.slogdet(sigma_e)
    if sign_k <= 0 or sign_e <= 0:
        raise np.linalg.LinAlgError("covariance is not positive definite")
    return vinv, ld_k + (n - 1) * ld_e


def design(times: np.ndarray) -> np.ndarray:
    """Fixed-effect design for one patient: (2n, 4), visit-major, outcome-minor.

    Row 2j+0 is the tremor equation at visit j, row 2j+1 the PIGD equation, so
    the stacked response is ``[T_0, P_0, T_1, P_1, ...]``.
    """
    t = np.asarray(times, dtype=float)
    n = len(t)
    X = np.zeros((2 * n, 4))
    X[0::2, 0] = 1.0
    X[0::2, 1] = t
    X[1::2, 2] = 1.0
    X[1::2, 3] = t
    return X


# ---------------------------------------------------------------------------
#  Packed data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PackedData:
    """Per-patient times and bivariate responses, grouped by visit count.

    `groups` maps a visit count n to (patient ids, times (m, n), Y (m, n, 2)).
    Patients are grouped so the covariance is factorised once per n rather than
    once per patient, which is the whole reason a 934-fold refit is affordable.
    """

    patients: list
    times: dict
    responses: dict
    groups: dict

    @property
    def n_patients(self) -> int:
        return len(self.patients)

    @property
    def n_visits(self) -> int:
        return int(sum(len(v) for v in self.times.values()))


def pack(frame: pd.DataFrame) -> PackedData:
    """Pack a subscore trajectory table into per-patient arrays.

    `frame` must carry PATNO, years_from_baseline and both subscore columns,
    already ordered within patient. Rows where either subscore is missing are
    dropped here and counted by the caller, never silently.
    """
    need = ["PATNO", "years_from_baseline", *OUTCOMES]
    missing = [c for c in need if c not in frame.columns]
    if missing:
        raise KeyError(f"frame is missing {missing}")

    usable = frame.dropna(subset=list(OUTCOMES))
    times: dict = {}
    responses: dict = {}
    for patno, g in usable.groupby("PATNO", sort=True):
        g = g.sort_values("years_from_baseline", kind="mergesort")
        times[patno] = g["years_from_baseline"].to_numpy(dtype=float)
        responses[patno] = g[list(OUTCOMES)].to_numpy(dtype=float)

    patients = sorted(times)
    by_n: dict = {}
    for p in patients:
        by_n.setdefault(len(times[p]), []).append(p)

    groups = {
        n: (
            ps,
            np.stack([times[p] for p in ps]),
            np.stack([responses[p] for p in ps]),
        )
        for n, ps in by_n.items()
    }
    return PackedData(patients=patients, times=times, responses=responses, groups=groups)


def subset(data: PackedData, keep) -> PackedData:
    """A new PackedData restricted to `keep`, used to drop the held-out patient."""
    keep = set(keep)
    patients = [p for p in data.patients if p in keep]
    times = {p: data.times[p] for p in patients}
    responses = {p: data.responses[p] for p in patients}
    by_n: dict = {}
    for p in patients:
        by_n.setdefault(len(times[p]), []).append(p)
    groups = {
        n: (ps, np.stack([times[p] for p in ps]), np.stack([responses[p] for p in ps]))
        for n, ps in by_n.items()
    }
    return PackedData(patients=patients, times=times, responses=responses, groups=groups)


# ---------------------------------------------------------------------------
#  Likelihood
# ---------------------------------------------------------------------------
def _accumulate(data: PackedData, sigma_b: np.ndarray, sigma_e: np.ndarray):
    """Sum the GLS pieces over all patients.

    Returns (XtVX, XtVy, ytVy, logdet, n_obs). The fixed effects are profiled
    out of the likelihood from these, so the optimiser only ever searches the
    six variance parameters.
    """
    XtVX = np.zeros((4, 4))
    XtVy = np.zeros(4)
    ytVy = 0.0
    logdet = 0.0
    n_obs = 0

    for n, (_, T, Y) in data.groups.items():
        vinv, ld = vinv_logdet(n, sigma_b, sigma_e)
        m = T.shape[0]
        X = np.zeros((m, 2 * n, 4))
        X[:, 0::2, 0] = 1.0
        X[:, 0::2, 1] = T
        X[:, 1::2, 2] = 1.0
        X[:, 1::2, 3] = T
        y = Y.reshape(m, 2 * n)

        XV = np.einsum("pia,ij->paj", X, vinv)
        XtVX += np.einsum("paj,pjb->ab", XV, X)
        XtVy += np.einsum("paj,pj->a", XV, y)
        ytVy += float(np.einsum("pi,ij,pj->", y, vinv, y))
        logdet += m * ld
        n_obs += m * 2 * n

    return XtVX, XtVy, ytVy, logdet, n_obs


def _objective(theta: np.ndarray, data: PackedData, reml: bool) -> float:
    """-2 log-likelihood with the fixed effects profiled out."""
    try:
        sigma_b, sigma_e = _unpack(theta)
        XtVX, XtVy, ytVy, logdet, n_obs = _accumulate(data, sigma_b, sigma_e)
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
class JointLMMFit:
    """A fitted joint model: fixed effects, both covariances, and provenance."""

    beta: np.ndarray
    beta_se: np.ndarray
    sigma_b: np.ndarray
    sigma_e: np.ndarray
    neg2ll: float
    converged: bool
    n_iter: int
    reml: bool
    n_patients: int
    n_obs: int
    message: str

    @property
    def corr_b(self) -> float:
        d = np.sqrt(np.diag(self.sigma_b))
        return float(self.sigma_b[0, 1] / (d[0] * d[1]))

    @property
    def corr_e(self) -> float:
        d = np.sqrt(np.diag(self.sigma_e))
        return float(self.sigma_e[0, 1] / (d[0] * d[1]))

    def icc(self) -> np.ndarray:
        """Share of each outcome's variance that sits between patients."""
        vb, ve = np.diag(self.sigma_b), np.diag(self.sigma_e)
        return vb / (vb + ve)

    def fixed_effects_table(self) -> pd.DataFrame:
        z = 1.959963984540054
        return pd.DataFrame(
            {
                "term": list(BETA_NAMES),
                "estimate": self.beta,
                "std_error": self.beta_se,
                "ci_low": self.beta - z * self.beta_se,
                "ci_high": self.beta + z * self.beta_se,
            }
        )


def fit_joint_lmm(
    data: PackedData, reml: bool = True, max_iter: int = 500, seed: int = 42
) -> JointLMMFit:
    """Fit by REML (default) or ML, profiling the fixed effects.

    Starting values come from the empirical between- and within-patient
    covariance of the data, which puts the optimiser close enough that a single
    L-BFGS-B run suffices; a Nelder-Mead polish follows if it reports failure.
    """
    means = np.stack([data.responses[p].mean(axis=0) for p in data.patients])
    devs = np.concatenate([data.responses[p] - data.responses[p].mean(axis=0)
                           for p in data.patients])
    sb0 = np.cov(means.T) + 1e-4 * np.eye(2)
    se0 = np.cov(devs.T) + 1e-4 * np.eye(2)
    theta0 = _pack(sb0, se0)

    res = minimize(_objective, theta0, args=(data, reml), method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8})
    if not res.success:
        res = minimize(_objective, res.x, args=(data, reml), method="Nelder-Mead",
                       options={"maxiter": max_iter * 10, "xatol": 1e-10,
                                "fatol": 1e-10})

    sigma_b, sigma_e = _unpack(res.x)
    XtVX, XtVy, _, _, n_obs = _accumulate(data, sigma_b, sigma_e)
    beta = np.linalg.solve(XtVX, XtVy)
    cov_beta = np.linalg.inv(XtVX)

    return JointLMMFit(
        beta=beta,
        beta_se=np.sqrt(np.diag(cov_beta)),
        sigma_b=sigma_b,
        sigma_e=sigma_e,
        neg2ll=float(res.fun),
        converged=bool(res.success),
        n_iter=int(res.nit) if hasattr(res, "nit") else -1,
        reml=reml,
        n_patients=data.n_patients,
        n_obs=int(n_obs),
        message=str(res.message),
    )


# ---------------------------------------------------------------------------
#  Prediction
# ---------------------------------------------------------------------------
def blup(fit: JointLMMFit, t_hist: np.ndarray, Y_hist: np.ndarray):
    """Posterior mean and covariance of a patient's random intercepts.

    This is the only place a held-out patient's own data is used. It touches the
    patient's history and the *training* fit, never any other patient's visits
    and never the target visit.
    """
    n = len(t_hist)
    if n < 1:
        raise ValueError("a BLUP needs at least one historical visit")
    resid = (Y_hist - (design(t_hist) @ fit.beta).reshape(n, 2)).sum(axis=0)

    S = np.linalg.inv(fit.sigma_e)
    K = n * fit.sigma_b + fit.sigma_e
    A = np.linalg.solve(K, fit.sigma_b) @ S
    A = 0.5 * (A + A.T)

    b_hat = fit.sigma_b @ (S - n * A) @ resid
    post = fit.sigma_b - fit.sigma_b @ (n * S - n * n * A) @ fit.sigma_b
    return b_hat, 0.5 * (post + post.T)


def predict(fit: JointLMMFit, t_hist: np.ndarray, Y_hist: np.ndarray, t_target: float):
    """Predictive mean and SD of (tremor, PIGD) at `t_target`.

    The SD combines residual noise with the posterior uncertainty of the random
    intercept. Fixed-effect uncertainty is excluded: with 900-odd training
    patients it is two orders of magnitude smaller than either, and including it
    would make the held-out SD depend on the training size in a way that
    complicates comparison with the baselines.
    """
    b_hat, post = blup(fit, t_hist, Y_hist)
    mean = np.array([fit.beta[0] + fit.beta[1] * t_target,
                     fit.beta[2] + fit.beta[3] * t_target]) + b_hat
    sd = np.sqrt(np.clip(np.diag(post) + np.diag(fit.sigma_e), 1e-12, None))
    return mean, sd


def population_predict(fit: JointLMMFit, t_target: float):
    """Prediction ignoring the patient entirely --- the population trend only."""
    mean = np.array([fit.beta[0] + fit.beta[1] * t_target,
                     fit.beta[2] + fit.beta[3] * t_target])
    sd = np.sqrt(np.diag(fit.sigma_b) + np.diag(fit.sigma_e))
    return mean, sd


def baseline_predict(name: str, t_hist: np.ndarray, Y_hist: np.ndarray,
                     t_target: float, fit: JointLMMFit):
    """Non-model comparators, each scored on exactly the same points.

    ``persistence`` carries the last observed pair forward and is the one that
    matters: the project already uses it everywhere, and the question this step
    asks is whether modelling progression beats simply repeating what was
    measured last.
    """
    if name == "persistence":
        mean = Y_hist[-1].astype(float)
    elif name == "patient_mean":
        mean = Y_hist.mean(axis=0)
    elif name == "population":
        return population_predict(fit, t_target)
    else:
        raise ValueError(f"unknown baseline {name!r}")
    sd = (Y_hist.std(axis=0, ddof=1) if len(Y_hist) > 1
          else np.sqrt(np.diag(fit.sigma_e)))
    return mean, np.where(np.isfinite(sd) & (sd > 0), sd, np.sqrt(np.diag(fit.sigma_e)))
