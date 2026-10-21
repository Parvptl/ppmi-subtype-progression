"""Rich longitudinal clinical profiles: a per-domain level-and-rate representation.

What this is for
----------------
The previous experiment (`clinical_states.py`) compressed ten clinical
measurements into K discrete latent states. Those states ordered patients by
overall severity and lost the information the prediction target needed; a
diagnostic priced the discretisation itself at 10.5 accuracy points. Two
measured facts explain why, and they set this module's design:

* **33% of the variance in these measurements is within-patient** (52% for
  PIGD). A model whose only structure is "which state is this patient in"
  cannot represent that share by construction.
* **790 of 1,080 patients start levodopa during follow-up** and 98.7% are
  untreated at baseline. With LEDD as a *feature*, "has started treatment" is
  one of the coordinates the representation is built from, which is a confound.

So this module keeps the representation **continuous** and gives every patient,
in every clinical domain, two numbers filtered from their own history:

    level  --- where this patient is now in this domain
    rate   --- how fast they are moving in this domain

Medication is an **input**, not a domain: it enters the observation equation
with a coefficient and is therefore subtracted from what the level has to
explain, rather than being something the representation encodes.

The model
---------
A continuous-time local linear trend (integrated Brownian motion) per domain.
For patient i, domain d, successive visits separated by Δ years:

    y[i,d,t]        = level[i,d,t] + beta[d] * ledd[i,t] + N(0, s_obs[d])
    level[i,d,t+Δ]  = level[i,d,t] + Δ * rate[i,d,t] + w_L
    rate[i,d,t+Δ]   = rate[i,d,t]                    + w_R

with the exact integrated-Brownian innovation covariance

    Cov([w_L, w_R]) = q[d] * [[Δ³/3, Δ²/2], [Δ²/2, Δ]]

so irregular visit spacing is handled exactly rather than approximated on a
fixed grid. Parameters (``s_obs``, ``q``, ``beta``, the initial prior) are
shared across patients and fitted on training patients only.

Causality
---------
Inference is a Kalman filter. The filtered state after visit k is a function of
visits 0..k and nothing else, so "no future information" is a structural
property of the recursion rather than a convention to be policed. The smoother
is used only for parameter fitting on training patients; every prediction and
every representation used for prediction comes from the filter.

Missing measurements are handled exactly: the rows of the observation vector
that were not recorded are dropped from that visit's update, which is the
correct marginalisation for a Gaussian observation. Nothing is imputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import clinical_states as cs

#: The clinical domains predicted. The previous experiment's ten features with
#: LEDD removed: it becomes the medication input instead of a coordinate of the
#: representation, which is the single most consequential change here.
DOMAINS: tuple[str, ...] = tuple(f for f in cs.FEATURE_NAMES if f != "ledd")

#: The medication input.
MEDICATION = "ledd"

#: Two state components per domain: level and rate.
N_COMPONENTS = 2

#: Floors keeping variances strictly positive during optimisation.
MIN_OBS_VAR = 1e-4
MIN_PROCESS_VAR = 1e-8

#: Bounds on the log-parameters. exp(-12) is numerically a zero variance;
#: exp(6) is far above the variance of any standardised domain.
_LOG_BOUNDS = (-12.0, 6.0)

#: Prior variance on a patient's initial level and rate, on the standardised
#: scale. The level prior is wide because a patient's level is genuinely
#: unknown before their first visit; the rate prior is tighter because annual
#: change in a standardised score is small, and without that tightening the
#: first prediction for a two-visit patient extrapolates a one-visit slope.
INIT_LEVEL_VAR = 4.0
INIT_RATE_VAR = 0.25


# ---------------------------------------------------------------------------
#  Sequences
# ---------------------------------------------------------------------------
def build_sequences(clinical: pd.DataFrame, min_length: int = 2) -> list[dict]:
    """One entry per patient: domain matrix, medication input, times, labels.

    ``Y`` keeps NaN where a measurement was not recorded. ``d`` is the
    medication input at each visit, aligned to the same rows.
    """
    sequences = []
    for patno, g in clinical.groupby("PATNO", sort=False):
        g = g.sort_values("years_from_baseline")
        if len(g) < min_length:
            continue
        times = g["years_from_baseline"].to_numpy(dtype=float)
        sequences.append({
            "PATNO": patno,
            "Y": g[list(DOMAINS)].to_numpy(dtype=float),
            "d": g[MEDICATION].to_numpy(dtype=float),
            "times": times,
            "gaps": np.diff(times),
            "labels": g["subtype"].to_numpy(dtype=int),
            "events": g["EVENT_ID"].to_numpy(),
            "visit_index": g["visit_index"].to_numpy(dtype=int),
        })
    return sequences


@dataclass
class Standardiser:
    """Per-domain centring and scaling, learned from training visits only.

    The medication input is scaled too, so ``beta`` is in comparable units
    across domains, but it is never centred: zero LEDD means "no medication"
    and must stay zero, or an untreated visit would receive a medication
    effect.
    """

    centre: np.ndarray
    scale: np.ndarray
    med_scale: float
    domains: tuple[str, ...] = DOMAINS
    n_train_visits: int = 0

    @classmethod
    def fit(cls, sequences: list[dict]) -> "Standardiser":
        if not sequences:
            raise ValueError("no sequences supplied")
        Y = np.vstack([s["Y"] for s in sequences])
        d = np.concatenate([s["d"] for s in sequences])
        with np.errstate(invalid="ignore"):
            centre = np.nanmean(Y, axis=0)
            scale = np.nanstd(Y, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        centre = np.where(np.isfinite(centre), centre, 0.0)
        med = float(np.nanstd(d))
        return cls(centre=centre, scale=scale,
                   med_scale=med if med > 1e-8 else 1.0,
                   n_train_visits=int(len(Y)))

    def apply(self, sequences: list[dict]) -> list[dict]:
        """Standardised copies. NaN stays NaN; medication is scaled, not centred."""
        return [dict(s,
                     Y=(s["Y"] - self.centre) / self.scale,
                     d=s["d"] / self.med_scale)
                for s in sequences]

    def invert_mean(self, Z: np.ndarray) -> np.ndarray:
        """Predicted means back to the original clinical units."""
        return np.asarray(Z, dtype=float) * self.scale + self.centre

    def invert_sd(self, S: np.ndarray) -> np.ndarray:
        """Predictive standard deviations back to the original units."""
        return np.asarray(S, dtype=float) * self.scale


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------
@dataclass
class ProfileModel:
    """A fitted per-domain local linear trend with a medication input.

    All parameter vectors are length ``n_domains``; the domains share the model
    *form* and the fitting run but have their own parameters, so a domain that
    moves slowly does not inherit a fast domain's dynamics.
    """

    obs_var: np.ndarray        # observation noise variance per domain
    level_var: np.ndarray      # level innovation variance per domain, per year
    rate_var: np.ndarray       # rate innovation variance per domain, per year
    beta: np.ndarray           # medication coefficient per domain
    #: Prior variance on a patient's initial rate, per domain. Fitted rather
    #: than fixed: the Kalman likelihood is a *prequential* one-step-ahead
    #: score, so a model forced to carry a non-zero initial slope uncertainty
    #: it does not need pays for it on every early prediction and would not
    #: nest the random walk. Letting the data shrink this to zero is what makes
    #: the level-only contrast a genuine test of the rate component.
    init_rate_var: np.ndarray | None = None
    #: Whether the rate component is active. False makes the model a per-domain
    #: random walk, which is the level-only baseline.
    use_rate: bool = True
    #: Whether the medication input is used at all.
    use_medication: bool = True
    domains: tuple[str, ...] = DOMAINS
    loglik: float = float("nan")
    #: scipy's ``success`` flag. Not the convergence criterion on its own: with
    #: several variances driven to their lower bound, L-BFGS-B terminates on a
    #: boundary condition and reports False at a point where the likelihood has
    #: stopped moving (0.000 nats gained from 600 to 1200 iterations on the
    #: full cohort). ``proj_grad`` is the criterion actually used, exactly as
    #: in ``hmm_stability``.
    scipy_success: bool = False
    proj_grad: float = float("nan")
    termination: str = ""
    n_iter: int = 0
    n_train_patients: int = 0
    n_train_visits: int = 0

    #: Projected-gradient tolerance, **relative to the log-likelihood scale**.
    #: An absolute tolerance is meaningless here: the objective is a total
    #: log-likelihood over ~38,000 domain-observations, so a gradient of 0.5 is
    #: already one part in 10^5 of the objective. The full model terminates on
    #: the iteration limit along a flat ridge -- several variances sit at their
    #: lower bound and are weakly identified there -- but the likelihood has
    #: stopped moving: measured on the full cohort, raising the cap from 600 to
    #: 1200 iterations gained 0.000 nats, and the last 287 iterations gained
    #: 0.07 of 45,812. Both the flag and the gradient are recorded so this
    #: judgement is visible rather than implied.
    PROJ_GRAD_REL_TOL: float = 1e-4

    @property
    def converged(self) -> bool:
        """scipy succeeded, or the projected gradient is negligible for this scale."""
        if self.scipy_success:
            return True
        if not np.isfinite(self.proj_grad) or not np.isfinite(self.loglik):
            return False
        return bool(self.proj_grad
                    < self.PROJ_GRAD_REL_TOL * max(abs(self.loglik), 1.0))

    @property
    def n_domains(self) -> int:
        return len(self.obs_var)

    def __post_init__(self) -> None:
        if self.init_rate_var is None:
            object.__setattr__(self, "init_rate_var",
                               np.full(self.n_domains,
                                       INIT_RATE_VAR if self.use_rate else 0.0))

    @property
    def n_params(self) -> int:
        per = 2 + (2 if self.use_rate else 0) + (1 if self.use_medication else 0)
        return per * self.n_domains

    def coefficient_frame(self) -> pd.DataFrame:
        names = (list(self.domains) if len(self.domains) == self.n_domains
                 else [f"d{i}" for i in range(self.n_domains)])
        return pd.DataFrame({
            "domain": names,
            "obs_var": self.obs_var,
            "level_innovation_var": self.level_var,
            "rate_innovation_var": np.where(self.use_rate, self.rate_var, 0.0),
            "init_rate_var": np.asarray(self.init_rate_var, dtype=float),
            "medication_beta": self.beta,
        })

    # -- one domain's filter ------------------------------------------------
    def _transition(self, dt: float, d_index: int):
        """(A, Q) for one domain over elapsed time dt.

        A local linear trend with *separate* level and rate innovations:

            Q = [[s_L*dt + s_R*dt^3/3,  s_R*dt^2/2],
                 [s_R*dt^2/2,           s_R*dt    ]]

        The ``s_R`` terms are the exact integrated-Brownian covariance of a
        rate that itself diffuses; the ``s_L*dt`` term is the level's own
        diffusion. Keeping both matters: with only the integrated term the
        model cannot represent a level that wanders without a trend, so it does
        **not** nest the random walk and the level-only contrast would compare
        two parameterisations rather than isolating the rate component. With
        ``s_R = 0`` this reduces exactly to the random walk.
        """
        sL = self.level_var[d_index]
        sR = self.rate_var[d_index]
        if self.use_rate:
            A = np.array([[1.0, dt], [0.0, 1.0]])
            Q = np.array([[sL * dt + sR * dt ** 3 / 3.0, sR * dt ** 2 / 2.0],
                          [sR * dt ** 2 / 2.0, sR * dt]])
        else:
            A = np.array([[1.0, 0.0], [0.0, 0.0]])
            Q = np.array([[sL * dt, 0.0], [0.0, 0.0]])
        return A, Q

    def filter_domain(self, y: np.ndarray, d: np.ndarray, times: np.ndarray,
                      d_index: int):
        """Kalman filter for one domain. Strictly causal.

        Returns ``(means, covs, loglik, pred_means, pred_vars)`` where
        ``means[k]`` is E[(level, rate) | observations 0..k] and
        ``pred_means[k]`` / ``pred_vars[k]`` are the one-step-ahead prediction
        of the *observation* at visit k made from visits 0..k-1 (the first
        entry is the prior prediction).
        """
        n = len(y)
        beta = self.beta[d_index] if self.use_medication else 0.0
        H = np.array([1.0, 0.0])

        m = np.zeros(N_COMPONENTS)
        P = np.diag([INIT_LEVEL_VAR,
                     self.init_rate_var[d_index] if self.use_rate else 0.0])

        means = np.zeros((n, N_COMPONENTS))
        covs = np.zeros((n, N_COMPONENTS, N_COMPONENTS))
        pred_means = np.zeros(n)
        pred_vars = np.zeros(n)
        loglik = 0.0

        for k in range(n):
            if k > 0:
                dt = float(times[k] - times[k - 1])
                A, Q = self._transition(dt, d_index)
                m = A @ m
                P = A @ P @ A.T + Q

            # One-step-ahead prediction of the observation, before using it.
            mu = float(H @ m + beta * d[k])
            var = float(H @ P @ H + self.obs_var[d_index])
            pred_means[k] = mu
            pred_vars[k] = var

            if np.isfinite(y[k]):
                resid = y[k] - mu
                loglik += -0.5 * (np.log(2 * np.pi * var) + resid ** 2 / var)
                K = (P @ H) / var
                m = m + K * resid
                P = P - np.outer(K, H @ P)
                P = 0.5 * (P + P.T)
            # A visit with no measurement in this domain leaves the state at
            # its predicted value: exact marginalisation, not imputation.

            means[k] = m
            covs[k] = P

        return means, covs, loglik, pred_means, pred_vars

    def filter_sequence(self, seq: dict):
        """Filter every domain of one patient.

        Returns a dict with ``level`` and ``rate`` arrays of shape
        ``(n_visits, n_domains)`` --- the representation --- plus their
        variances, the one-step-ahead observation predictions, and the
        sequence log-likelihood summed over domains.
        """
        Y, d, times = seq["Y"], seq["d"], seq["times"]
        n, D = Y.shape
        level = np.zeros((n, D))
        rate = np.zeros((n, D))
        level_var = np.zeros((n, D))
        rate_var = np.zeros((n, D))
        pred_mean = np.zeros((n, D))
        pred_var = np.zeros((n, D))
        total = 0.0

        cross = np.zeros((n, D))
        for j in range(D):
            means, covs, ll, pm, pv = self.filter_domain(Y[:, j], d, times, j)
            level[:, j] = means[:, 0]
            rate[:, j] = means[:, 1]
            level_var[:, j] = covs[:, 0, 0]
            rate_var[:, j] = covs[:, 1, 1]
            # The level-rate covariance is kept: propagating the state forward
            # without it understates the predictive variance, because an
            # uncertain level and an uncertain rate are correlated after a
            # filter update.
            cross[:, j] = covs[:, 0, 1]
            pred_mean[:, j] = pm
            pred_var[:, j] = pv
            total += ll

        return {"level": level, "rate": rate, "level_var": level_var,
                "rate_var": rate_var, "level_rate_cov": cross,
                "pred_mean": pred_mean, "pred_var": pred_var,
                "loglik": total}

    def predict_next(self, filtered: dict, k: int, dt: float,
                     d_next: float) -> tuple[np.ndarray, np.ndarray]:
        """Predictive mean and variance of the profile at visit k+1.

        Built from the filtered state at visit ``k`` propagated over the actual
        elapsed time ``dt``, plus the medication term at its visit-``k+1``
        value. Nothing from visit ``k+1``'s *measurements* enters; the elapsed
        time and the prescribed dose are inputs, as in the previous
        experiments, and that assumption is recorded in the report.
        """
        D = self.n_domains
        mean = np.zeros(D)
        var = np.zeros(D)
        H = np.array([1.0, 0.0])
        d_next = float(d_next)

        for j in range(D):
            m = np.array([filtered["level"][k, j], filtered["rate"][k, j]])
            c = filtered["level_rate_cov"][k, j]
            P = np.array([[filtered["level_var"][k, j], c],
                          [c, filtered["rate_var"][k, j]]])

            A, Q = self._transition(float(dt), j)
            m = A @ m
            P = A @ P @ A.T + Q
            beta = self.beta[j] if self.use_medication else 0.0
            mean[j] = float(H @ m) + beta * d_next
            var[j] = float(H @ P @ H) + self.obs_var[j]
        return mean, var


# ---------------------------------------------------------------------------
#  Fitting
# ---------------------------------------------------------------------------
def _pack(obs_var, level_var, rate_var, init_rate_var, beta,
          use_rate, use_medication):
    parts = [np.log(np.maximum(obs_var, MIN_OBS_VAR)),
             np.log(np.maximum(level_var, MIN_PROCESS_VAR))]
    if use_rate:
        parts.append(np.log(np.maximum(rate_var, MIN_PROCESS_VAR)))
        parts.append(np.log(np.maximum(init_rate_var, MIN_PROCESS_VAR)))
    if use_medication:
        parts.append(beta)
    return np.concatenate(parts)


def _unpack(theta, n_domains, use_rate, use_medication):
    obs_var = np.exp(theta[:n_domains])
    level_var = np.exp(theta[n_domains:2 * n_domains])
    k = 2 * n_domains
    if use_rate:
        rate_var = np.exp(theta[k:k + n_domains])
        init_rate_var = np.exp(theta[k + n_domains:k + 2 * n_domains])
        k += 2 * n_domains
    else:
        rate_var = np.zeros(n_domains)
        init_rate_var = np.zeros(n_domains)
    beta = theta[k:k + n_domains] if use_medication else np.zeros(n_domains)
    return obs_var, level_var, rate_var, init_rate_var, beta


def _negloglik(theta, sequences, n_domains, use_rate, use_medication):
    """Reference objective via the per-sequence filter. Used only in tests."""
    obs_var, level_var, rate_var, init_rate_var, beta = _unpack(
        theta, n_domains, use_rate, use_medication)
    model = ProfileModel(obs_var=obs_var, level_var=level_var,
                         rate_var=rate_var, beta=beta,
                         init_rate_var=init_rate_var,
                         use_rate=use_rate, use_medication=use_medication)
    total = sum(model.filter_sequence(seq)["loglik"] for seq in sequences)
    return -total if np.isfinite(total) else 1e12


def fit(
    sequences: list[dict],
    use_rate: bool = True,
    use_medication: bool = True,
    maxiter: int = 600,
    maxfun: int = 120000,
) -> ProfileModel:
    """Maximum-likelihood fit of the shared parameters on these patients only.

    The objective is the exact Gaussian likelihood the Kalman filter returns,
    summed over patients and domains. It is maximised over log-variances and
    the medication coefficients with L-BFGS-B; the likelihood of a linear
    Gaussian state space is smooth in these parameters, so a gradient-free
    multi-start search is not needed, and a single deterministic start keeps a
    fold reproducible.

    Starting values come from the data: the observation variance from the
    within-patient variance of each domain, the process scales from the squared
    annualised first differences, and the medication coefficient from zero.

    ``maxfun`` is raised well above scipy's default of 15,000. With 45
    parameters and finite-difference gradients a single gradient costs 46
    function evaluations, so the default cap binds after ~320 iterations and
    the optimiser stops with ``TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT``
    at a point whose projected gradient is still large. ``hmm_stability``
    documents the same trap for this project's HMM.
    """
    if not sequences:
        raise ValueError("no sequences supplied")
    n_domains = sequences[0]["Y"].shape[1]

    Y = np.vstack([s["Y"] for s in sequences])
    with np.errstate(invalid="ignore"):
        total_var = np.nanvar(Y, axis=0)
    total_var = np.where(np.isfinite(total_var) & (total_var > 0), total_var, 1.0)

    # Squared annualised first differences give the scale of real change.
    diffs = []
    for s in sequences:
        if len(s["times"]) < 2:
            continue
        dt = np.diff(s["times"])[:, None]
        dy = np.diff(s["Y"], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            diffs.append((dy / np.sqrt(np.maximum(dt, 1e-6))) ** 2)
    if diffs:
        with np.errstate(invalid="ignore"):
            change = np.nanmean(np.vstack(diffs), axis=0)
        change = np.where(np.isfinite(change) & (change > 0), change, 0.05)
    else:
        change = np.full(n_domains, 0.05)

    obs0 = np.maximum(0.5 * total_var, MIN_OBS_VAR)
    # Split the observed annualised change between the level's own diffusion
    # and the rate's, rather than attributing all of it to one of them.
    level0 = np.maximum(0.5 * change, 1e-4)
    rate0 = np.maximum(0.1 * change, 1e-4)
    theta0 = _pack(obs0, level0, rate0, np.full(n_domains, INIT_RATE_VAR),
                   np.zeros(n_domains), use_rate, use_medication)

    n_var_blocks = 4 if use_rate else 2
    bounds = ([_LOG_BOUNDS] * (n_var_blocks * n_domains)
              + ([(-10.0, 10.0)] * n_domains if use_medication else []))

    batch = prepare_batch(sequences)
    res = minimize(_negloglik_batched, theta0,
                   args=(batch, n_domains, use_rate, use_medication),
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "maxfun": int(maxfun),
                            "ftol": 1e-10})

    # Projected gradient: zero out components a bound is holding, so a
    # parameter pinned at its floor does not look like a failure to optimise.
    pg = np.array(res.jac, dtype=float) if res.jac is not None else None
    if pg is not None:
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        pg = pg.copy()
        pg[(res.x <= lo + 1e-8) & (pg > 0)] = 0.0
        pg[(res.x >= hi - 1e-8) & (pg < 0)] = 0.0
        proj_grad = float(np.max(np.abs(pg)))
    else:
        proj_grad = float("nan")

    obs_var, level_var, rate_var, init_rate_var, beta = _unpack(
        res.x, n_domains, use_rate, use_medication)
    return ProfileModel(
        obs_var=obs_var, level_var=level_var, rate_var=rate_var, beta=beta,
        init_rate_var=init_rate_var,
        use_rate=use_rate, use_medication=use_medication,
        loglik=-float(res.fun), scipy_success=bool(res.success),
        proj_grad=proj_grad, termination=str(res.message),
        n_iter=int(res.nit), n_train_patients=len(sequences),
        n_train_visits=int(batch.n_observations))


# ---------------------------------------------------------------------------
#  Prediction, for the model and for the baselines
# ---------------------------------------------------------------------------
MODEL_FULL = "Profile: level + rate + medication"
MODEL_NO_RATE = "Profile: level only (random walk)"
MODEL_NO_MED = "Profile: level + rate, no medication"
MODEL_LOCF = "Last observation carried forward"
MODEL_POPMEAN = "Population mean trajectory"

#: The predictive standard deviation assigned to the two deterministic
#: baselines so they can be scored by a probabilistic rule at all. It is the
#: training-set RMSE of that baseline per domain, which is the most favourable
#: constant variance it could be given -- so the comparison does not beat them
#: merely by handing them an arbitrary spread.
_BASELINE_SD_FLOOR = 1e-3


@dataclass
class PopulationMean:
    """Training-set mean profile as a function of time since baseline.

    Binned on years from baseline, with the overall training mean as the
    fallback for a bin no training visit reached. This is the "everyone follows
    the average course" baseline: it uses the population and the clock but
    nothing about the individual.
    """

    edges: np.ndarray
    means: np.ndarray            # (n_bins, n_domains)
    overall: np.ndarray          # (n_domains,)
    sd: np.ndarray               # (n_domains,) training residual sd

    @classmethod
    def fit(cls, sequences: list[dict], bin_width: float = 1.0,
            max_years: float = 15.0) -> "PopulationMean":
        edges = np.arange(0.0, max_years + bin_width, bin_width)
        Y = np.vstack([s["Y"] for s in sequences])
        t = np.concatenate([s["times"] for s in sequences])
        idx = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 2)

        D = Y.shape[1]
        means = np.full((len(edges) - 1, D), np.nan)
        for b in range(len(edges) - 1):
            m = idx == b
            if m.any():
                with np.errstate(invalid="ignore"):
                    means[b] = np.nanmean(Y[m], axis=0)
        with np.errstate(invalid="ignore"):
            overall = np.nanmean(Y, axis=0)
        overall = np.where(np.isfinite(overall), overall, 0.0)
        means = np.where(np.isfinite(means), means, overall[None, :])

        resid = Y - means[idx]
        with np.errstate(invalid="ignore"):
            sd = np.nanstd(resid, axis=0)
        sd = np.maximum(np.where(np.isfinite(sd), sd, 1.0), _BASELINE_SD_FLOOR)
        return cls(edges=edges, means=means, overall=overall, sd=sd)

    def predict(self, t: float) -> np.ndarray:
        b = int(np.clip(np.digitize([float(t)], self.edges)[0] - 1,
                        0, len(self.edges) - 2))
        return self.means[b]


def locf_sd(sequences: list[dict]) -> np.ndarray:
    """Per-domain training RMSE of carrying the last observation forward.

    Gives the LOCF baseline a predictive spread equal to its own training
    error, which is the fairest constant variance available to it.
    """
    errs = []
    for s in sequences:
        Y = s["Y"]
        for k in range(len(Y) - 1):
            errs.append(Y[k + 1] - Y[k])
    if not errs:
        return np.ones(sequences[0]["Y"].shape[1])
    with np.errstate(invalid="ignore"):
        sd = np.sqrt(np.nanmean(np.vstack(errs) ** 2, axis=0))
    return np.maximum(np.where(np.isfinite(sd), sd, 1.0), _BASELINE_SD_FLOOR)


def _last_observed(Y: np.ndarray, k: int,
                   fallback: np.ndarray | None = None) -> np.ndarray:
    """Most recent observed value per domain at or before visit k.

    LOCF has to carry the last value that was actually *recorded*, not the
    value at visit k, which may be missing in some domains.

    A domain that has never been recorded in visits ``0..k`` gives this
    baseline nothing to carry, and it must not silently emit NaN -- that would
    quietly drop those cases from its score while the other models were still
    being charged for them. ``fallback`` supplies a value for exactly those
    cases and must be a *training* statistic; the caller passes the training
    mean. Which entries fell back is returned so the count can be reported.
    """
    D = Y.shape[1]
    out = np.full(D, np.nan)
    for j in range(D):
        col = Y[: k + 1, j]
        seen = col[np.isfinite(col)]
        if len(seen):
            out[j] = seen[-1]
    used_fallback = ~np.isfinite(out)
    if fallback is not None and used_fallback.any():
        out[used_fallback] = np.asarray(fallback, dtype=float)[used_fallback]
    return out, used_fallback


def predict_sequence(
    model: ProfileModel, seq: dict, standardiser: Standardiser,
    population: PopulationMean | None = None,
    locf_spread: np.ndarray | None = None,
    locf_fallback: np.ndarray | None = None,
    models: tuple[str, ...] = (MODEL_FULL,),
) -> list[dict]:
    """One record per (prediction opportunity, model) for one patient.

    The model's predictions come from the filter, so visit k+1 cannot influence
    the prediction made for it. The baselines are computed from the same
    history.
    """
    Y, d, times = seq["Y"], seq["d"], seq["times"]
    filtered = model.filter_sequence(seq)
    n = len(times)
    records = []

    for k in range(n - 1):
        dt = float(times[k + 1] - times[k])
        actual = Y[k + 1]
        observed = np.isfinite(actual)

        for name in models:
            n_fallback = 0
            if name in (MODEL_FULL, MODEL_NO_RATE, MODEL_NO_MED):
                mean, var = model.predict_next(filtered, k, dt, d[k + 1])
            elif name == MODEL_LOCF:
                mean, fell_back = _last_observed(Y, k, fallback=locf_fallback)
                var = np.asarray(locf_spread, dtype=float) ** 2
                n_fallback = int(fell_back.sum())
            elif name == MODEL_POPMEAN:
                mean = population.predict(times[k + 1])
                var = population.sd ** 2
            else:
                raise ValueError(f"unknown model {name!r}")

            records.append({
                "PATNO": seq["PATNO"],
                "visit_index": int(seq["visit_index"][k]),
                "n_history": k + 1,
                "current_event": seq["events"][k],
                "current_years": float(times[k]),
                "target_event": seq["events"][k + 1],
                "target_years": float(times[k + 1]),
                "interval_years": dt,
                "ledd_current": float(d[k]),
                "ledd_next": float(d[k + 1]),
                "current_label": cs.STATE_NAMES[int(seq["labels"][k]) - 1],
                "actual_next_label": cs.STATE_NAMES[int(seq["labels"][k + 1]) - 1],
                "model": name,
                "n_domains_observed": int(observed.sum()),
                "n_domains_fallback": n_fallback,
                "_mean": mean, "_var": var, "_actual": actual,
                "_observed": observed,
                "_standardiser": standardiser,
            })
    return records


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """Expand the per-domain arrays into columns, in original clinical units.

    Predictions are reported on the clinical scale, not the standardised one,
    so a per-domain error is readable as "MoCA points" rather than "standard
    deviations of MoCA".
    """
    rows = []
    for r in records:
        std = r["_standardiser"]
        mean = std.invert_mean(r["_mean"])
        sd = std.invert_sd(np.sqrt(np.maximum(r["_var"], 0.0)))
        actual = std.invert_mean(r["_actual"])
        row = {k: v for k, v in r.items() if not k.startswith("_")}
        for j, dom in enumerate(DOMAINS):
            row[f"pred_{dom}"] = mean[j]
            row[f"sd_{dom}"] = sd[j]
            row[f"actual_{dom}"] = actual[j]
        # Standardised copies, for domain-comparable aggregate scores.
        for j, dom in enumerate(DOMAINS):
            row[f"z_pred_{dom}"] = r["_mean"][j]
            row[f"z_sd_{dom}"] = np.sqrt(max(r["_var"][j], 0.0))
            row[f"z_actual_{dom}"] = r["_actual"][j]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Metrics: prediction quality
# ---------------------------------------------------------------------------
def profile_metrics(frame: pd.DataFrame, seed: int = 42,
                    n_boot: int = 2000) -> dict:
    """Held-out profile prediction quality for one model.

    The headline score is the Gaussian predictive log-density of the next
    visit's measurements, summed over the domains that were actually recorded
    and reported per observed domain so that visits recording different
    numbers of domains are comparable. Per-domain RMSE and MAE are given on the
    clinical scale; the aggregate RMSE is on the standardised scale, because
    averaging MoCA points with SCOPA points would otherwise be meaningless.
    """
    name = frame["model"].iloc[0] if len(frame) else ""
    if not len(frame):
        return {"model": name, "n_predictions": 0}

    ll_total, n_terms = 0.0, 0
    per_pred_ll = np.zeros(len(frame))
    z_sq, z_abs, z_n = 0.0, 0.0, 0
    calib_hits, calib_n = 0, 0

    for i, (_, r) in enumerate(frame.iterrows()):
        ll_i, terms = 0.0, 0
        for dom in DOMAINS:
            a = r[f"z_actual_{dom}"]
            if not np.isfinite(a):
                continue
            mu = r[f"z_pred_{dom}"]
            sd = max(float(r[f"z_sd_{dom}"]), 1e-6)
            ll_i += -0.5 * (np.log(2 * np.pi * sd ** 2) + ((a - mu) / sd) ** 2)
            terms += 1
            z_sq += (a - mu) ** 2
            z_abs += abs(a - mu)
            z_n += 1
            # 95% central predictive interval coverage.
            calib_hits += int(abs(a - mu) <= 1.959963985 * sd)
            calib_n += 1
        per_pred_ll[i] = ll_i / terms if terms else np.nan
        ll_total += ll_i
        n_terms += terms

    per_domain = []
    for dom in DOMAINS:
        a = frame[f"actual_{dom}"].to_numpy(dtype=float)
        p = frame[f"pred_{dom}"].to_numpy(dtype=float)
        m = np.isfinite(a) & np.isfinite(p)
        per_domain.append({
            "domain": dom, "n": int(m.sum()),
            "rmse": float(np.sqrt(np.mean((a[m] - p[m]) ** 2))) if m.any() else np.nan,
            "mae": float(np.mean(np.abs(a[m] - p[m]))) if m.any() else np.nan,
        })

    # Patient-clustered bootstrap on the per-prediction mean log-density.
    valid = frame.assign(_ll=per_pred_ll).dropna(subset=["_ll"])
    by_patient = {p: g["_ll"].to_numpy() for p, g in valid.groupby("PATNO")}
    patients = np.array(list(by_patient.keys()), dtype=object)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(patients), size=len(patients), replace=True)
        draws[b] = np.concatenate([by_patient[patients[i]] for i in pick]).mean()

    return {
        "model": name,
        "n_predictions": int(len(frame)),
        "n_patients": int(frame["PATNO"].nunique()),
        "n_domain_observations": int(n_terms),
        "loglik_per_domain_observation": float(ll_total / max(n_terms, 1)),
        "mean_loglik_per_prediction": float(np.nanmean(per_pred_ll)),
        "loglik_ci_low": float(np.quantile(draws, 0.025)),
        "loglik_ci_high": float(np.quantile(draws, 0.975)),
        "rmse_standardised": float(np.sqrt(z_sq / max(z_n, 1))),
        "mae_standardised": float(z_abs / max(z_n, 1)),
        "coverage_95": float(calib_hits / max(calib_n, 1)),
        "per_domain": per_domain,
    }


def paired_loglik_difference(frame_a: pd.DataFrame, frame_b: pd.DataFrame,
                             seed: int = 42, n_boot: int = 2000) -> dict:
    """Patient-clustered CI for the per-prediction log-density difference.

    Paired on the identical prediction opportunities, because the two models
    are scored on the same held-out visits and their separate intervals would
    overlap even when one is reliably better.
    """
    def per_pred(frame):
        out = {}
        for _, r in frame.iterrows():
            ll, terms = 0.0, 0
            for dom in DOMAINS:
                a = r[f"z_actual_{dom}"]
                if not np.isfinite(a):
                    continue
                mu, sd = r[f"z_pred_{dom}"], max(float(r[f"z_sd_{dom}"]), 1e-6)
                ll += -0.5 * (np.log(2 * np.pi * sd ** 2) + ((a - mu) / sd) ** 2)
                terms += 1
            if terms:
                out[(r["PATNO"], r["visit_index"])] = ll / terms
        return out

    a, b = per_pred(frame_a), per_pred(frame_b)
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"n_paired": 0}
    diff = np.array([a[k] - b[k] for k in keys])
    pat = np.array([k[0] for k in keys], dtype=object)

    by_patient = {}
    for p, v in zip(pat, diff):
        by_patient.setdefault(p, []).append(v)
    patients = np.array(list(by_patient.keys()), dtype=object)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(len(patients), size=len(patients), replace=True)
        draws[i] = np.concatenate([by_patient[patients[j]] for j in pick]).mean()

    return {
        "n_paired": len(keys),
        "n_patients": int(len(patients)),
        "loglik_delta": float(diff.mean()),
        "delta_ci_low": float(np.quantile(draws, 0.025)),
        "delta_ci_high": float(np.quantile(draws, 0.975)),
        "share_a_better": float((diff > 0).mean()),
    }


# ---------------------------------------------------------------------------
#  The derived-label bridge, for the secondary endpoint
# ---------------------------------------------------------------------------
def derive_label_probabilities(
    frame: pd.DataFrame, cfg, n_draws: int = 2000, seed: int = 42
) -> pd.DataFrame:
    """Turn a predicted (tremor, PIGD) distribution into subtype probabilities.

    The subtype is a deterministic, thresholded ratio of the tremor and PIGD
    scores, and this project's :func:`subtype.classify` reproduces the recorded
    panel label from those two numbers exactly. So a predicted *distribution*
    over the two scores induces a distribution over the label, obtained by
    Monte Carlo: draw (tremor, PIGD) pairs from the predictive Gaussians, push
    each through the project's own unmodified classification rule, and count.

    This is the honest bridge to the previous experiment's endpoint: no new
    classifier is fitted and no new threshold is introduced. It also respects
    the structure of the label -- a prediction sitting near the 0.90/1.15 ratio
    boundary yields a genuinely uncertain label, which a point prediction
    pushed through the rule would hide.
    """
    from .subtype import classify

    rng = np.random.default_rng(seed)
    n = len(frame)
    tre_mu = frame["pred_tremor"].to_numpy(dtype=float)
    tre_sd = np.maximum(frame["sd_tremor"].to_numpy(dtype=float), 1e-9)
    pig_mu = frame["pred_pigd"].to_numpy(dtype=float)
    pig_sd = np.maximum(frame["sd_pigd"].to_numpy(dtype=float), 1e-9)

    # Scores are means of 0-4 item ratings, so a draw outside [0, 4] is not a
    # possible score; clipping keeps the pushed-through rule meaningful.
    tre = np.clip(tre_mu[:, None] + tre_sd[:, None] * rng.standard_normal((n, n_draws)),
                  0.0, 4.0)
    pig = np.clip(pig_mu[:, None] + pig_sd[:, None] * rng.standard_normal((n, n_draws)),
                  0.0, 4.0)

    labels = cfg.subtype.labels
    counts = np.zeros((n, len(cs.STATE_NAMES)))
    for s in range(n_draws):
        lab = classify(pd.Series(tre[:, s]), pd.Series(pig[:, s]), cfg.subtype)
        code = lab.to_numpy(dtype=float)
        for i, name in enumerate(cs.STATE_NAMES):
            counts[:, i] += (code == labels[name])

    probs = counts / n_draws
    out = frame.copy()
    for i, name in enumerate(cs.STATE_NAMES):
        out[f"p_{name}"] = probs[:, i]
    return out


# ---------------------------------------------------------------------------
#  Representation diagnostics (H2) and clinical relevance (H3)
# ---------------------------------------------------------------------------
def representation_frame(model: ProfileModel, sequences: list[dict],
                         standardiser: Standardiser) -> pd.DataFrame:
    """The filtered representation at every visit, one row per patient-visit.

    Columns ``level_<domain>`` and ``rate_<domain>`` are the representation.
    Levels are returned in clinical units; rates are returned as clinical units
    per year, which is what makes a rate readable ("MoCA points lost per year")
    rather than a standardised abstraction.
    """
    rows = []
    for seq in sequences:
        f = model.filter_sequence(seq)
        lvl = standardiser.invert_mean(f["level"])
        # A rate is a difference per year, so it scales but does not shift.
        rte = f["rate"] * standardiser.scale[None, :]
        for k in range(len(seq["times"])):
            row = {"PATNO": seq["PATNO"],
                   "visit_index": int(seq["visit_index"][k]),
                   "years_from_baseline": float(seq["times"][k]),
                   "n_history": k + 1,
                   "subtype": cs.STATE_NAMES[int(seq["labels"][k]) - 1],
                   "ledd": float(seq["d"][k]) * standardiser.med_scale}
            for j, dom in enumerate(DOMAINS):
                row[f"level_{dom}"] = lvl[k, j]
                row[f"rate_{dom}"] = rte[k, j]
            rows.append(row)
    return pd.DataFrame(rows)


def representation_dimensionality(rep: pd.DataFrame, which: str = "level") -> dict:
    """Is the representation more than one severity axis? (hypothesis H2)

    Takes each patient's representation at their last available visit, z-scores
    the columns and reports the PCA variance spectrum. H2 was pre-registered to
    fail if one component explains more than 90% of the variance, which would
    mean the representation is a severity score wearing nine coordinates.
    """
    cols = [f"{which}_{d}" for d in DOMAINS]
    last = rep.sort_values("visit_index").groupby("PATNO").tail(1)
    X = last[cols].to_numpy(dtype=float)
    X = X[np.isfinite(X).all(axis=1)]
    if len(X) < len(cols) + 1:
        return {"n_patients": int(len(X)), "insufficient": True}

    Z = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
    # Eigenvalues of the correlation matrix are the component variances.
    eig = np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False))[::-1]
    eig = np.maximum(eig, 0.0)
    share = eig / eig.sum()
    return {
        "which": which,
        "n_patients": int(len(X)),
        "n_domains": len(cols),
        "pc1_share": float(share[0]),
        "pc2_share": float(share[1]),
        "cumulative_2": float(share[:2].sum()),
        "cumulative_3": float(share[:3].sum()),
        "n_components_for_90pct": int(np.searchsorted(np.cumsum(share), 0.90) + 1),
        "effective_rank_participation": float(1.0 / np.sum(share ** 2)),
        "variance_shares": share.tolist(),
    }


def rate_spread(rep: pd.DataFrame) -> pd.DataFrame:
    """Across-patient spread of the fitted rates, per domain.

    If the representation captured only average severity, the rates would be
    shrunk to a common value and this spread would be near zero. It is the
    direct check that progression is being represented as patient-specific.
    """
    last = rep.sort_values("visit_index").groupby("PATNO").tail(1)
    rows = []
    for dom in DOMAINS:
        v = last[f"rate_{dom}"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        rows.append({
            "domain": dom, "n_patients": int(len(v)),
            "mean_rate_per_year": float(v.mean()) if len(v) else np.nan,
            "sd_rate_per_year": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
            "p10": float(np.quantile(v, 0.10)) if len(v) else np.nan,
            "p90": float(np.quantile(v, 0.90)) if len(v) else np.nan,
            "share_worsening": float((v > 0).mean()) if len(v) else np.nan,
        })
    return pd.DataFrame(rows)


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUC, ties at half credit. NaN when a class is absent."""
    score = np.asarray(score, dtype=float)
    positive = np.asarray(positive).astype(bool)
    ok = np.isfinite(score)
    score, positive = score[ok], positive[ok]
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average ranks within ties so a constant score scores 0.5, not 1.0.
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_auc_difference(score_a, score_b, positive, seed: int = 42,
                             n_boot: int = 2000) -> dict:
    """Patient-level bootstrap CI for AUC(a) - AUC(b) on the same patients."""
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    positive = np.asarray(positive).astype(bool)
    ok = np.isfinite(score_a) & np.isfinite(score_b)
    score_a, score_b, positive = score_a[ok], score_b[ok], positive[ok]
    if positive.sum() == 0 or (~positive).sum() == 0:
        return {"n": int(len(positive)), "auc_a": float("nan"),
                "auc_b": float("nan"), "delta": float("nan")}

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    n = len(positive)
    for b in range(n_boot):
        pick = rng.integers(0, n, size=n)
        if positive[pick].sum() in (0, n):
            draws[b] = np.nan
            continue
        draws[b] = auc(score_a[pick], positive[pick]) - auc(score_b[pick], positive[pick])
    draws = draws[np.isfinite(draws)]
    return {
        "n": int(n), "n_positive": int(positive.sum()),
        "auc_a": auc(score_a, positive), "auc_b": auc(score_b, positive),
        "delta": auc(score_a, positive) - auc(score_b, positive),
        "delta_ci_low": float(np.quantile(draws, 0.025)) if len(draws) else np.nan,
        "delta_ci_high": float(np.quantile(draws, 0.975)) if len(draws) else np.nan,
    }


# ---------------------------------------------------------------------------
#  Batched filter: the same recursion over all patients and domains at once
# ---------------------------------------------------------------------------
@dataclass
class ProfileBatch:
    """Ragged patient sequences packed into rectangular arrays.

    The per-sequence filter above is a Python loop over ~930 patients times 9
    domains, and the likelihood is optimised with finite differences over 27
    parameters -- which makes a single fit cost more than the whole experiment
    can afford. The recursion here is identical, run over every
    (patient, domain) pair simultaneously; the state is only 2-dimensional, so
    the 2x2 algebra is written out rather than delegated to a solver.

    ``test_batched_filter_matches_the_reference`` asserts the two agree to
    floating point, which is the only reason to trust this path.
    """

    Y: np.ndarray            # (n_seq, max_len, D), padded with NaN
    d: np.ndarray            # (n_seq, max_len) medication input
    dt: np.ndarray           # (n_seq, max_len - 1) elapsed years
    visit_mask: np.ndarray   # (n_seq, max_len)
    lengths: np.ndarray
    n_observations: int

    @property
    def n_sequences(self) -> int:
        return self.Y.shape[0]

    @property
    def n_domains(self) -> int:
        return self.Y.shape[2]


def prepare_batch(sequences: list[dict]) -> ProfileBatch:
    """Pack sequences into padded arrays."""
    if not sequences:
        raise ValueError("no sequences supplied")
    lengths = np.array([len(s["times"]) for s in sequences], dtype=int)
    max_len = int(lengths.max())
    D = sequences[0]["Y"].shape[1]
    n = len(sequences)

    Y = np.full((n, max_len, D), np.nan)
    d = np.zeros((n, max_len))
    dt = np.ones((n, max(max_len - 1, 1)))
    mask = np.zeros((n, max_len), dtype=bool)

    for i, s in enumerate(sequences):
        k = len(s["times"])
        Y[i, :k] = s["Y"]
        d[i, :k] = s["d"]
        mask[i, :k] = True
        if k > 1:
            dt[i, :k - 1] = np.diff(s["times"])
    return ProfileBatch(Y=Y, d=d, dt=dt, visit_mask=mask, lengths=lengths,
                        n_observations=int(mask.sum()))


def batched_filter(model: ProfileModel, batch: ProfileBatch,
                   collect: bool = False):
    """Filter every (patient, domain) pair at once. Strictly causal.

    Returns ``(loglik, collected)``; ``collected`` is None unless ``collect``
    is set, in which case it holds the filtered means, variances and
    one-step-ahead observation predictions with the shapes the per-sequence
    filter produces.
    """
    n, max_len, D = batch.Y.shape
    sL = model.level_var[None, :]
    sR = (model.rate_var[None, :] if model.use_rate else np.zeros((1, D)))
    obs = model.obs_var[None, :]
    beta = (model.beta[None, :] if model.use_medication else np.zeros((1, D)))

    m0 = np.zeros((n, D))
    m1 = np.zeros((n, D))
    P00 = np.full((n, D), INIT_LEVEL_VAR)
    P01 = np.zeros((n, D))
    P11 = (np.asarray(model.init_rate_var, dtype=float).copy()[None, :]
           * np.ones((n, 1)) if model.use_rate else np.zeros((n, D)))

    out = ({k: np.zeros((n, max_len, D)) for k in
            ("level", "rate", "level_var", "rate_var", "level_rate_cov",
             "pred_mean", "pred_var")} if collect else None)
    loglik = np.zeros((n, D))

    for t in range(max_len):
        if t > 0:
            step = batch.dt[:, t - 1][:, None]
            if model.use_rate:
                # Local linear trend; see ProfileModel._transition.
                m0 = m0 + step * m1
                new_P00 = (P00 + 2.0 * step * P01 + step ** 2 * P11
                           + sL * step + sR * step ** 3 / 3.0)
                new_P01 = P01 + step * P11 + sR * step ** 2 / 2.0
                new_P11 = P11 + sR * step
                P00, P01, P11 = new_P00, new_P01, new_P11
            else:
                # A random walk in the level; the rate coordinate is pinned.
                P00 = P00 + sL * step
                m1 = np.zeros_like(m1)
                P01 = np.zeros_like(P01)
                P11 = np.zeros_like(P11)

        mu = m0 + beta * batch.d[:, t][:, None]
        var = P00 + obs
        y = batch.Y[:, t, :]
        use = np.isfinite(y) & batch.visit_mask[:, t][:, None]

        if collect:
            out["pred_mean"][:, t, :] = mu
            out["pred_var"][:, t, :] = var

        resid = np.where(use, y - mu, 0.0)
        ll = -0.5 * (np.log(2 * np.pi * var) + resid ** 2 / var)
        loglik += np.where(use, ll, 0.0)

        K0 = P00 / var
        K1 = P01 / var
        m0 = np.where(use, m0 + K0 * resid, m0)
        m1 = np.where(use, m1 + K1 * resid, m1)
        new_P00 = np.where(use, P00 - K0 * P00, P00)
        new_P01 = np.where(use, P01 - K0 * P01, P01)
        new_P11 = np.where(use, P11 - K1 * P01, P11)
        P00, P01, P11 = new_P00, new_P01, new_P11

        if collect:
            out["level"][:, t, :] = m0
            out["rate"][:, t, :] = m1
            out["level_var"][:, t, :] = P00
            out["rate_var"][:, t, :] = P11
            out["level_rate_cov"][:, t, :] = P01

    return float(loglik.sum()), out


def _negloglik_batched(theta, batch, n_domains, use_rate, use_medication):
    obs_var, level_var, rate_var, init_rate_var, beta = _unpack(
        theta, n_domains, use_rate, use_medication)
    model = ProfileModel(obs_var=obs_var, level_var=level_var,
                         rate_var=rate_var, beta=beta,
                         init_rate_var=init_rate_var,
                         use_rate=use_rate, use_medication=use_medication)
    total, _ = batched_filter(model, batch)
    return -total if np.isfinite(total) else 1e12


# ---------------------------------------------------------------------------
#  H3: clinical relevance against outcomes never used to fit anything
# ---------------------------------------------------------------------------
#: Outcomes drawn from the panel, none of which is one of the nine domains.
#: ``independent`` records whether the outcome is genuinely free of the feature
#: set; it is False for the investigator cognitive diagnosis, which is a
#: clinician judgement plausibly informed by cognitive testing.
OUTCOMES = {
    "hy_ge_3": {"column": "hy", "threshold": 3.0, "independent": True,
                "label": "Hoehn & Yahr stage >= 3 (separate clinical staging)"},
    "mci": {"column": "MCI_testscores", "threshold": 1.0, "independent": True,
            "label": "MCI from HVLT/Benton test scores (not MoCA)"},
    "motor_complications": {"column": "updrs4_score", "threshold": 0.5,
                            "independent": True,
                            "label": "MDS-UPDRS Part IV > 0 (Part IV excluded from features)"},
    "cognitive_dx": {"column": "cogstate", "threshold": 2.0, "independent": False,
                     "label": "investigator cognitive diagnosis (NOT fully independent)"},
}


def build_outcome_table(panel: pd.DataFrame, representation: pd.DataFrame,
                        anchor_visit: int = 1) -> pd.DataFrame:
    """Per-patient representation at an early visit, plus strictly later outcomes.

    ``anchor_visit`` is the visit index whose filtered representation is used,
    so the representation is formed from that patient's first
    ``anchor_visit + 1`` visits and nothing else. An outcome counts as positive
    only if it is recorded **after** that visit's time, which is what makes
    this a prospective association rather than a restatement of the current
    state.
    """
    rep = representation[representation["visit_index"] == anchor_visit].copy()
    rep = rep.drop_duplicates("PATNO")
    rows = []

    panel = panel.copy()
    panel["years_from_baseline"] = pd.to_numeric(
        panel["years_from_baseline"], errors="coerce")

    by_patient = {p: g for p, g in panel.groupby("PATNO")}
    for _, r in rep.iterrows():
        g = by_patient.get(r["PATNO"])
        if g is None:
            continue
        later = g[g["years_from_baseline"] > r["years_from_baseline"] + 1e-9]
        row = {"PATNO": r["PATNO"],
               "anchor_years": r["years_from_baseline"],
               "n_later_visits": int(len(later)),
               "last_followup_years": (float(later["years_from_baseline"].max())
                                       if len(later) else np.nan),
               "subtype_at_anchor": r["subtype"]}
        for dom in DOMAINS:
            row[f"level_{dom}"] = r[f"level_{dom}"]
            row[f"rate_{dom}"] = r[f"rate_{dom}"]
        for name, spec in OUTCOMES.items():
            col = spec["column"]
            if col not in later.columns or not len(later):
                row[name] = np.nan
                continue
            v = pd.to_numeric(later[col], errors="coerce").dropna()
            row[name] = (float((v >= spec["threshold"]).any())
                         if len(v) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


#: Ordinal encoding of the subtype label used as the reference score in H3.
#: It is the only ordering the label supports, and it is the comparator the
#: representation has to beat for "richer than the label" to mean anything.
_SUBTYPE_SCORE = {"TD": 0.0, "Indeterminate": 0.5, "PIGD": 1.0}


def outcome_associations(table: pd.DataFrame, seed: int = 42,
                         n_boot: int = 2000, min_events: int = 20) -> pd.DataFrame:
    """Univariate AUC of every representation component against each outcome.

    Univariate deliberately: no combination is fitted, so there is no model to
    leak and no hyperparameter to choose, and the comparison against the
    subtype label is like for like. The cost is that a multivariate combination
    of the representation is not assessed, which is recorded as a limitation.
    """
    rows = []
    subtype_score = table["subtype_at_anchor"].map(_SUBTYPE_SCORE).to_numpy(float)

    for name, spec in OUTCOMES.items():
        if name not in table.columns:
            continue
        y = table[name].to_numpy(dtype=float)
        ok = np.isfinite(y)
        n_events = int(np.nansum(y[ok]))
        if n_events < min_events or (ok.sum() - n_events) < min_events:
            rows.append({"outcome": name, "label": spec["label"],
                         "independent": spec["independent"],
                         "n": int(ok.sum()), "n_events": n_events,
                         "score": "(skipped: too few events)",
                         "auc": np.nan})
            continue

        candidates = {"subtype label (reference)": subtype_score}
        for dom in DOMAINS:
            candidates[f"level_{dom}"] = table[f"level_{dom}"].to_numpy(float)
            candidates[f"rate_{dom}"] = table[f"rate_{dom}"].to_numpy(float)

        for score_name, score in candidates.items():
            a = auc(score[ok], y[ok].astype(bool))
            rows.append({
                "outcome": name, "label": spec["label"],
                "independent": spec["independent"],
                "n": int(ok.sum()), "n_events": n_events,
                "score": score_name,
                # AUC is symmetric about 0.5 under sign flip; the direction of
                # a level or rate is not fixed a priori, so the discriminative
                # strength is reported as the distance from chance and the raw
                # value is kept so the direction stays visible.
                "auc": a,
                "auc_abs": max(a, 1.0 - a) if np.isfinite(a) else np.nan,
            })
    return pd.DataFrame(rows)


def best_component_vs_subtype(table: pd.DataFrame, associations: pd.DataFrame,
                              outcome: str, kind: str = "rate",
                              seed: int = 42, n_boot: int = 2000) -> dict:
    """Bootstrap CI for (best single `kind` component) - (subtype label) AUC.

    The component is chosen on the same data it is then compared on, which
    inflates it; the comparison is therefore an upper bound on what a single
    component buys, and is labelled as such.
    """
    sub = associations[(associations["outcome"] == outcome)
                       & associations["score"].str.startswith(f"{kind}_")]
    if not len(sub) or sub["auc_abs"].isna().all():
        return {"outcome": outcome, "kind": kind, "n": 0}
    best = sub.loc[sub["auc_abs"].idxmax(), "score"]

    y = table[outcome].to_numpy(dtype=float)
    ok = np.isfinite(y)
    score = table[best].to_numpy(float)
    # Orient the component so a higher score means more outcome, matching the
    # subtype reference's orientation.
    if auc(score[ok], y[ok].astype(bool)) < 0.5:
        score = -score
    ref = table["subtype_at_anchor"].map(_SUBTYPE_SCORE).to_numpy(float)

    out = bootstrap_auc_difference(score[ok], ref[ok], y[ok].astype(bool),
                                   seed=seed, n_boot=n_boot)
    return {"outcome": outcome, "kind": kind, "best_component": best,
            "selection_is_optimistic": True, **out}
