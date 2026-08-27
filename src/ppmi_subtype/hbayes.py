"""Hierarchical Bayesian growth model of the TD/PIGD ratio.

This is the model specified in the project proposal (pages 13--15), implemented
as written.

    Observation model   R_it^obs = R_it + eps_it^obs,      eps^obs ~ N(0, s_obs^2)
    Growth model        R_it     = (B0 + b0i) + (B1 + b1i) t + eps_it^proc
    Random effects      b0i ~ N(0, tau0^2),  b1i ~ N(0, tau1^2)
    Clinical output     P(TD)  = P(R > 1.15)
                        P(Ind) = P(0.90 <= R <= 1.15)
                        P(PIGD)= P(R < 0.90)

A note on the two variance terms
--------------------------------
Substituting the growth model into the observation model gives

    R_it^obs = (B0 + b0i) + (B1 + b1i) t + (eps^proc + eps^obs)

Both error terms are independent and additive on the *same* observation, so the
likelihood depends on them only through their sum. Fixing s_proc^2 + s_obs^2 and
moving mass between the two leaves the log-likelihood numerically unchanged;
they are not separately identifiable in this parameterisation.

The sampler therefore estimates the identified quantity

    sigma^2 = s_proc^2 + s_obs^2

and says so, rather than reporting a split that the data cannot support. The
consequence is recorded in the report: P(TD) = P(R^true > 1.15) refers to the
*latent* ratio, whose spread excludes observation noise, so only the predictive
distribution of R^obs is identified here. Separating the two requires either
the state-space form the proposal also mentions, where process noise
accumulates over time and therefore leaves a distinguishable signature, or
repeat measurements at a single visit.

Inference
---------
Gibbs sampling with conjugate updates throughout; no external probabilistic
programming dependency. Each sweep draws the random effects for all patients as
a batch of 2x2 normal draws, so a full run over ~800 patients takes seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TD_THRESHOLD = 1.15
PIGD_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
#  Data preparation
# ---------------------------------------------------------------------------
@dataclass
class GrowthData:
    """Ratio observations packed for the sampler."""

    y: np.ndarray            # (N,) observed ratio
    t: np.ndarray            # (N,) years from the patient's first visit
    patient_index: np.ndarray  # (N,) 0-based patient index
    patients: np.ndarray     # (M,) PATNO in index order
    n_patients: int
    n_obs: int

    def design(self) -> np.ndarray:
        return np.column_stack([np.ones_like(self.t), self.t])


def build_growth_data(
    panel: pd.DataFrame,
    min_visits: int = 2,
    ratio_col: str = "ratio",
    time_col: str = "years_from_baseline",
) -> GrowthData:
    """Extract ratio trajectories for patients with enough defined observations.

    Visits where the PIGD subscore is zero have no finite ratio and are dropped;
    they are 16.3% of labelled visits and are overwhelmingly TD, so this is a
    non-random loss and is reported as a limitation rather than hidden.
    """
    df = panel[panel[ratio_col].notna() & np.isfinite(panel[ratio_col])]
    df = df.sort_values(["PATNO", time_col])

    counts = df.groupby("PATNO")[ratio_col].size()
    keep = counts[counts >= min_visits].index
    df = df[df["PATNO"].isin(keep)]

    patients = np.sort(df["PATNO"].unique())
    position = {p: k for k, p in enumerate(patients)}

    return GrowthData(
        y=df[ratio_col].to_numpy(dtype=float),
        t=df[time_col].to_numpy(dtype=float),
        patient_index=df["PATNO"].map(position).to_numpy(dtype=int),
        patients=patients,
        n_patients=len(patients),
        n_obs=len(df),
    )


# ---------------------------------------------------------------------------
#  Posterior
# ---------------------------------------------------------------------------
@dataclass
class GrowthPosterior:
    """Posterior draws from the hierarchical growth model."""

    beta: np.ndarray        # (S, 2) population intercept and slope
    sigma2: np.ndarray      # (S,) total residual variance
    tau0_2: np.ndarray      # (S,) random-intercept variance
    tau1_2: np.ndarray      # (S,) random-slope variance
    n_patients: int
    n_obs: int
    n_draws: int
    random_effects: np.ndarray | None = None   # (S, M, 2), if retained
    patients: np.ndarray | None = None         # (M,) PATNO in index order

    def summary(self) -> pd.DataFrame:
        rows = {
            "beta0 (population baseline ratio)": self.beta[:, 0],
            "beta1 (population progression, /yr)": self.beta[:, 1],
            "sigma (total residual SD)": np.sqrt(self.sigma2),
            "tau0 (SD of patient baselines)": np.sqrt(self.tau0_2),
            "tau1 (SD of patient slopes, /yr)": np.sqrt(self.tau1_2),
        }
        return pd.DataFrame(
            [
                {
                    "parameter": name,
                    "mean": float(np.mean(v)),
                    "sd": float(np.std(v)),
                    "q2.5": float(np.quantile(v, 0.025)),
                    "q50": float(np.quantile(v, 0.5)),
                    "q97.5": float(np.quantile(v, 0.975)),
                }
                for name, v in rows.items()
            ]
        )


def fit_growth_model(
    data: GrowthData,
    n_draws: int = 4000,
    burn_in: int = 1000,
    thin: int = 2,
    seed: int = 42,
    prior_beta_sd: float = 100.0,
    prior_shape: float = 0.01,
    prior_rate: float = 0.01,
    keep_random_effects: bool = False,
) -> GrowthPosterior:
    """Gibbs sampler for the hierarchical growth model.

    Priors are weakly informative: normal on the population coefficients and
    inverse-gamma on each variance component. With ~800 patients the variance
    posteriors are data-dominated.
    """
    rng = np.random.default_rng(seed)
    X = data.design()
    y, idx, M, N = data.y, data.patient_index, data.n_patients, data.n_obs

    # Per-patient sufficient statistics, computed once.
    XtX = np.zeros((M, 2, 2))
    for k in range(2):
        for l in range(2):
            XtX[:, k, l] = np.bincount(idx, weights=X[:, k] * X[:, l], minlength=M)

    beta = np.array([np.mean(y), 0.0])
    b = np.zeros((M, 2))
    sigma2 = float(np.var(y))
    tau0_2, tau1_2 = 0.1 * sigma2, 0.01 * sigma2

    prior_beta_prec = np.eye(2) / prior_beta_sd**2
    keep_beta, keep_sigma, keep_t0, keep_t1 = [], [], [], []
    keep_b = [] if keep_random_effects else None
    total = burn_in + n_draws * thin

    for sweep in range(total):
        # --- random effects, all patients at once ------------------------
        resid = y - X @ beta
        Xtr = np.column_stack(
            [np.bincount(idx, weights=X[:, k] * resid, minlength=M) for k in range(2)]
        )
        D_inv = np.diag([1.0 / tau0_2, 1.0 / tau1_2])
        prec = XtX / sigma2 + D_inv                      # (M, 2, 2)
        V = np.linalg.inv(prec)
        m = np.einsum("mij,mj->mi", V, Xtr / sigma2)
        L = np.linalg.cholesky(V)
        b = m + np.einsum("mij,mj->mi", L, rng.standard_normal((M, 2)))

        # --- population coefficients -------------------------------------
        fitted_b = np.einsum("ij,ij->i", X, b[idx])
        r = y - fitted_b
        prec_beta = X.T @ X / sigma2 + prior_beta_prec
        V_beta = np.linalg.inv(prec_beta)
        m_beta = V_beta @ (X.T @ r) / sigma2
        beta = m_beta + np.linalg.cholesky(V_beta) @ rng.standard_normal(2)

        # --- variance components -----------------------------------------
        e = y - X @ beta - fitted_b
        sigma2 = 1.0 / rng.gamma(prior_shape + N / 2, 1.0 / (prior_rate + 0.5 * e @ e))
        tau0_2 = 1.0 / rng.gamma(
            prior_shape + M / 2, 1.0 / (prior_rate + 0.5 * b[:, 0] @ b[:, 0])
        )
        tau1_2 = 1.0 / rng.gamma(
            prior_shape + M / 2, 1.0 / (prior_rate + 0.5 * b[:, 1] @ b[:, 1])
        )

        if sweep >= burn_in and (sweep - burn_in) % thin == 0:
            keep_beta.append(beta.copy())
            keep_sigma.append(sigma2)
            keep_t0.append(tau0_2)
            keep_t1.append(tau1_2)
            if keep_b is not None:
                keep_b.append(b.copy())

    return GrowthPosterior(
        beta=np.array(keep_beta),
        sigma2=np.array(keep_sigma),
        tau0_2=np.array(keep_t0),
        tau1_2=np.array(keep_t1),
        n_patients=M,
        n_obs=N,
        n_draws=len(keep_sigma),
        random_effects=np.array(keep_b) if keep_b is not None else None,
        patients=data.patients,
    )


# ---------------------------------------------------------------------------
#  Prediction
# ---------------------------------------------------------------------------
def _subject_posterior_draw(post: GrowthPosterior, s: int, t_hist, y_hist, rng):
    """Draw this patient's random effects given their observed history.

    A held-out patient has no fitted b_i, so it is inferred from their own
    earlier visits under the population parameters --- the same information the
    hidden Markov model uses when it filters a patient's history forward.
    """
    Xh = np.column_stack([np.ones_like(t_hist), t_hist])
    sigma2 = post.sigma2[s]
    D_inv = np.diag([1.0 / post.tau0_2[s], 1.0 / post.tau1_2[s]])
    V = np.linalg.inv(Xh.T @ Xh / sigma2 + D_inv)
    m = V @ (Xh.T @ (y_hist - Xh @ post.beta[s])) / sigma2
    return m + np.linalg.cholesky(V) @ rng.standard_normal(2)


def predict_ratio(
    post: GrowthPosterior,
    t_hist: np.ndarray,
    y_hist: np.ndarray,
    t_target: float,
    include_observation_noise: bool = True,
    seed: int = 0,
    max_draws: int | None = None,
) -> np.ndarray:
    """Posterior predictive draws of the ratio at `t_target`.

    With `include_observation_noise` the draws describe the *next recorded*
    ratio, which is what the held-out comparison scores. Without it they
    describe the latent trajectory --- but see the module docstring: the two
    noise sources are not separable, so that variant should be read as the
    trajectory mean rather than as the proposal's R^true.
    """
    rng = np.random.default_rng(seed)
    n = post.n_draws if max_draws is None else min(post.n_draws, max_draws)
    picks = np.linspace(0, post.n_draws - 1, n).astype(int)

    out = np.empty(n)
    for k, s in enumerate(picks):
        b = _subject_posterior_draw(post, s, t_hist, y_hist, rng)
        mean = (post.beta[s, 0] + b[0]) + (post.beta[s, 1] + b[1]) * t_target
        out[k] = mean + (
            rng.normal(0.0, np.sqrt(post.sigma2[s])) if include_observation_noise else 0.0
        )
    return out


def label_probabilities(
    draws: np.ndarray,
    td_threshold: float = TD_THRESHOLD,
    pigd_threshold: float = PIGD_THRESHOLD,
) -> dict:
    """Map predictive ratio draws onto the clinical classes.

    Exactly the proposal's clinical translation:
        P(TD)  = P(R > 1.15),  P(Ind) = P(0.90 <= R <= 1.15),  P(PIGD) = P(R < 0.90)
    """
    return {
        "TD": float(np.mean(draws > td_threshold)),
        "Indeterminate": float(np.mean((draws >= pigd_threshold) & (draws <= td_threshold))),
        "PIGD": float(np.mean(draws < pigd_threshold)),
    }


def evaluate(
    post: GrowthPosterior,
    panel: pd.DataFrame,
    state_names=("TD", "Indeterminate", "PIGD"),
    seed: int = 0,
    max_draws: int = 400,
    ratio_col: str = "ratio",
    time_col: str = "years_from_baseline",
) -> tuple[dict, pd.DataFrame]:
    """One-step-ahead held-out performance, comparable to the Markov models.

    For every patient and every transition, the model sees the ratios up to
    visit k and predicts the class at visit k + 1.
    """
    df = panel[panel[ratio_col].notna() & np.isfinite(panel[ratio_col])]
    df = df.sort_values(["PATNO", time_col])

    records = []
    for patno, group in df.groupby("PATNO", sort=False):
        y = group[ratio_col].to_numpy(dtype=float)
        t = group[time_col].to_numpy(dtype=float)
        truth = group["subtype_name"].to_numpy()
        if len(y) < 2:
            continue
        for k in range(1, len(y)):
            draws = predict_ratio(post, t[:k], y[:k], float(t[k]),
                                  seed=seed + k, max_draws=max_draws)
            probs = label_probabilities(draws)
            records.append({"PATNO": patno, "step": k - 1, "observed": truth[k],
                            **{f"p_{s}": probs[s] for s in state_names}})

    preds = pd.DataFrame(records)
    if preds.empty:
        return {"model": "Hierarchical Bayesian growth", "n_pairs": 0}, preds

    cols = [f"p_{s}" for s in state_names]
    probs = preds[cols].to_numpy()
    # Guard against a class receiving exactly zero Monte-Carlo mass.
    probs = np.clip(probs, 1e-6, None)
    probs = probs / probs.sum(axis=1, keepdims=True)

    j = preds["observed"].map({s: k for k, s in enumerate(state_names)}).to_numpy(int)
    p_true = probs[np.arange(len(j)), j]
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(j)), j] = 1.0

    metrics = {
        "model": "Hierarchical Bayesian growth",
        "n_pairs": len(preds),
        "loglik": float(np.log(p_true).sum()),
        "loglik_per_pair": float(np.log(p_true).mean()),
        "accuracy": float((probs.argmax(axis=1) == j).mean()),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
    }
    return metrics, preds


def clinical_table(
    post: GrowthPosterior,
    current_ratio: float,
    horizons=(1, 2, 3, 5),
    seed: int = 0,
    max_draws: int = 2000,
) -> pd.DataFrame:
    """Progression probabilities for a patient whose current ratio is known.

    A single baseline observation is a thin basis for a random slope, so these
    are dominated by the population trajectory; the value of the hierarchical
    form appears once a patient has several visits.
    """
    rows = [{"horizon_years": 0.0, **label_probabilities(np.array([current_ratio]))}]
    for t in horizons:
        draws = predict_ratio(post, np.array([0.0]), np.array([current_ratio]),
                              float(t), seed=seed, max_draws=max_draws)
        rows.append({"horizon_years": float(t), **label_probabilities(draws)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Stage 4: per-patient latent phenotype and its uncertainty
# ---------------------------------------------------------------------------
def latent_trajectories(
    post: GrowthPosterior,
    data: GrowthData,
    quantiles=(0.025, 0.5, 0.975),
) -> pd.DataFrame:
    """Posterior of the latent ratio R_it for every patient at every visit.

    This is the quantity the proposal's Stage 4 asks for. Requires the sampler
    to have been run with ``keep_random_effects=True``.

    The latent value here is the trajectory mean
    ``(beta0 + b0i) + (beta1 + b1i) t``. Because the observation and process
    variances are not separable (see the module docstring), the process noise
    around that mean cannot be attributed, so this is the trajectory rather
    than the proposal's R^true in the strict sense.
    """
    if post.random_effects is None:
        raise ValueError("fit_growth_model(..., keep_random_effects=True) is required")

    b = post.random_effects                       # (S, M, 2)
    beta = post.beta                              # (S, 2)
    idx, t = data.patient_index, data.t

    intercepts = beta[:, 0][:, None] + b[:, :, 0]  # (S, M)
    slopes = beta[:, 1][:, None] + b[:, :, 1]

    latent = intercepts[:, idx] + slopes[:, idx] * t[None, :]   # (S, N)

    out = pd.DataFrame(
        {
            "PATNO": data.patients[idx],
            "years_from_baseline": t,
            "observed_ratio": data.y,
            "latent_mean": latent.mean(axis=0),
            "latent_sd": latent.std(axis=0),
        }
    )
    for q in quantiles:
        out[f"latent_q{q:g}"] = np.quantile(latent, q, axis=0)

    band = np.stack(
        [
            (latent > TD_THRESHOLD).mean(axis=0),
            ((latent >= PIGD_THRESHOLD) & (latent <= TD_THRESHOLD)).mean(axis=0),
            (latent < PIGD_THRESHOLD).mean(axis=0),
        ],
        axis=1,
    )
    out["P_TD"], out["P_Indeterminate"], out["P_PIGD"] = band[:, 0], band[:, 1], band[:, 2]
    return out


def patient_parameters(post: GrowthPosterior) -> pd.DataFrame:
    """Per-patient baseline level and progression rate, with credible intervals."""
    if post.random_effects is None:
        raise ValueError("fit_growth_model(..., keep_random_effects=True) is required")

    intercept = post.beta[:, 0][:, None] + post.random_effects[:, :, 0]
    slope = post.beta[:, 1][:, None] + post.random_effects[:, :, 1]

    return pd.DataFrame(
        {
            "PATNO": post.patients,
            "baseline_mean": intercept.mean(axis=0),
            "baseline_q2.5": np.quantile(intercept, 0.025, axis=0),
            "baseline_q97.5": np.quantile(intercept, 0.975, axis=0),
            "slope_mean": slope.mean(axis=0),
            "slope_q2.5": np.quantile(slope, 0.025, axis=0),
            "slope_q97.5": np.quantile(slope, 0.975, axis=0),
            "P_slope_negative": (slope < 0).mean(axis=0),
        }
    )


def reliability(post: GrowthPosterior, data: GrowthData) -> pd.DataFrame:
    """How much of each patient's random effect comes from their own data.

    Reliability is ``1 - Var_posterior(b) / tau^2``: zero means the estimate is
    the population mean with the patient contributing nothing, one means the
    patient's data determines it entirely. Reported alongside the per-patient
    estimates because a patient-specific slope with reliability 0.05 is, for
    practical purposes, the cohort average wearing that patient's name.
    """
    sigma2 = float(post.sigma2.mean())
    tau = np.array([post.tau0_2.mean(), post.tau1_2.mean()])
    D_inv = np.diag(1.0 / tau)

    rows = []
    for k, patno in enumerate(post.patients):
        mask = data.patient_index == k
        t = data.t[mask]
        X = np.column_stack([np.ones_like(t), t])
        V = np.linalg.inv(X.T @ X / sigma2 + D_inv)
        rows.append(
            {
                "PATNO": patno,
                "n_visits": int(mask.sum()),
                "span_years": float(t.max() - t.min()) if len(t) else 0.0,
                "reliability_baseline": 1.0 - V[0, 0] / tau[0],
                "reliability_slope": 1.0 - V[1, 1] / tau[1],
            }
        )
    return pd.DataFrame(rows)


def posterior_predictive(
    post: GrowthPosterior,
    data: GrowthData,
    patno,
    horizons=(1, 2, 3, 5),
    include_noise: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Future latent phenotype for one patient, using their retained draws.

    The proposal's Stage 4 second sentence: posterior predictive inference for
    the latent phenotype at future visits.
    """
    if post.random_effects is None:
        raise ValueError("fit_growth_model(..., keep_random_effects=True) is required")

    k = int(np.where(post.patients == patno)[0][0])
    last_t = float(data.t[data.patient_index == k].max())
    rng = np.random.default_rng(seed)

    intercept = post.beta[:, 0] + post.random_effects[:, k, 0]
    slope = post.beta[:, 1] + post.random_effects[:, k, 1]

    rows = []
    for h in horizons:
        t = last_t + float(h)
        draws = intercept + slope * t
        if include_noise:
            draws = draws + rng.normal(0.0, np.sqrt(post.sigma2))
        probs = label_probabilities(draws)
        rows.append(
            {
                "PATNO": patno,
                "years_ahead": float(h),
                "t_absolute": t,
                "predicted_mean": float(draws.mean()),
                "q2.5": float(np.quantile(draws, 0.025)),
                "q97.5": float(np.quantile(draws, 0.975)),
                **probs,
            }
        )
    return pd.DataFrame(rows)
