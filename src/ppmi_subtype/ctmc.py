"""Continuous-time Markov model of subtype progression (Weeks 5-7).

Why continuous time
-------------------
The Weeks 3-4 analysis established that the observed transition probability
depends strongly on the gap between visits: TD -> PIGD is 0.100 at intervals of
1.5 years or less and 0.289 beyond 4 years. A discrete-time chain indexed by
"next visit" would therefore absorb PPMI's attendance schedule into its
progression estimate, and could not answer the clinical question the proposal
poses, which asks for probabilities at 1, 2, 3 and 5 years.

A continuous-time Markov chain solves both problems with one object. The model
is a generator matrix Q whose off-diagonal entries q_ij are instantaneous
transition *rates*; the probability of moving from state i to state j over an
elapsed time t is then the (i, j) entry of

    P(t) = exp(Qt)

Each observed pair contributes at its own actual interval, and the fitted model
can be evaluated at any horizon.

Estimation
----------
Panel data (interval-censored): we observe the state at visits, not the moment
of transition, so the likelihood of a pair (i, j, dt) is simply P(dt)[i, j] and
the total log-likelihood is the sum over pairs. Off-diagonal rates are
parameterised as q_ij = exp(theta_ij) to keep them positive, and the six
theta are optimised numerically.

P(t) is computed by eigendecomposition rather than a fresh matrix exponential
per interval: Q is 3x3, so this is exact up to floating point and roughly two
orders of magnitude faster, which is what makes the patient-level bootstrap
affordable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

# Bounds on log-rates. exp(-12) is a rate of ~6e-6 per year (nothing happens in
# a human lifetime); exp(4) is ~55 per year (a transition every week).
_LOG_RATE_BOUNDS = (-12.0, 4.0)
_EPS = 1e-300


# ---------------------------------------------------------------------------
#  Generator construction and transition probabilities
# ---------------------------------------------------------------------------
def build_generator(rates: np.ndarray, n_states: int = 3) -> np.ndarray:
    """Assemble a generator matrix from its off-diagonal entries.

    `rates` holds the n_states * (n_states - 1) off-diagonal entries in
    row-major order. Each diagonal entry is set to minus the sum of its row, so
    every row sums to zero, as a generator must.
    """
    Q = np.zeros((n_states, n_states), dtype=float)
    k = 0
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                Q[i, j] = rates[k]
                k += 1
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def transition_probability(Q: np.ndarray, t: float) -> np.ndarray:
    """P(t) = exp(Qt) for a single elapsed time."""
    return expm(Q * float(t))


def transition_probabilities(Q: np.ndarray, times: np.ndarray) -> np.ndarray:
    """P(t) for many elapsed times at once, via eigendecomposition.

    Returns an array of shape (len(times), n, n). Falls back to a per-time
    matrix exponential if Q is not diagonalisable, which can happen at
    degenerate parameter values during optimisation.
    """
    times = np.asarray(times, dtype=float)
    try:
        w, V = np.linalg.eig(Q)
        Vinv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        return np.stack([expm(Q * t) for t in times])

    if np.linalg.cond(V) > 1e12:
        return np.stack([expm(Q * t) for t in times])

    # exp(w * t) for every (time, eigenvalue) pair, then V diag(.) V^-1.
    E = np.exp(np.outer(times, w))
    P = np.einsum("ij,kj,jl->kil", V, E, Vinv)
    P = np.real(P)

    # Numerical hygiene: probabilities must be non-negative and rows sum to 1.
    P = np.clip(P, 0.0, None)
    row_sums = P.sum(axis=2, keepdims=True)
    return np.divide(P, row_sums, out=P, where=row_sums > 0)


# ---------------------------------------------------------------------------
#  Fitting
# ---------------------------------------------------------------------------
@dataclass
class CTMCFit:
    """A fitted continuous-time Markov chain."""

    Q: np.ndarray
    states: list
    state_names: list
    loglik: float
    n_pairs: int
    n_patients: int
    converged: bool
    n_params: int = 6
    rate_ci: dict = field(default_factory=dict)

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik

    @property
    def bic(self) -> float:
        return self.n_params * np.log(self.n_pairs) - 2 * self.loglik

    def P(self, t: float) -> pd.DataFrame:
        """Transition probability matrix at elapsed time `t`, as a frame."""
        mat = transition_probability(self.Q, t)
        return pd.DataFrame(mat, index=self.state_names, columns=self.state_names)

    def generator_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.Q, index=self.state_names, columns=self.state_names)

    def mean_sojourn(self) -> pd.Series:
        """Expected time in years spent in a state before leaving it.

        For a CTMC the sojourn time in state i is Exponential(-q_ii), so its
        mean is -1 / q_ii. This is a property of the fitted rates and does not
        depend on the visit schedule, unlike the Weeks 3-4 figure derived from
        the diagonal of the observed matrix.
        """
        d = np.diag(self.Q)
        with np.errstate(divide="ignore"):
            out = np.where(d < 0, -1.0 / d, np.inf)
        return pd.Series(out, index=self.state_names)

    def stationary_distribution(self) -> pd.Series:
        """Long-run occupancy, the pi solving pi Q = 0 with sum(pi) = 1."""
        n = self.Q.shape[0]
        A = np.vstack([self.Q.T[:-1], np.ones(n)])
        b = np.zeros(n)
        b[-1] = 1.0
        pi, *_ = np.linalg.lstsq(A, b, rcond=None)
        return pd.Series(pi, index=self.state_names)


def _pair_arrays(pairs: pd.DataFrame, states: list) -> tuple:
    """Map a pairs frame to integer indices and grouped unique intervals."""
    index = {s: k for k, s in enumerate(states)}
    i = pairs["from_state"].map(index).to_numpy()
    j = pairs["to_state"].map(index).to_numpy()
    dt = pairs["interval_years"].to_numpy(dtype=float)

    if np.isnan(i).any() or np.isnan(j).any():
        raise ValueError("pairs contain a state not present in `states`")
    if not np.all(dt > 0):
        raise ValueError("all intervals must be strictly positive")

    # Group by rounded interval so the likelihood evaluates a few hundred
    # matrices rather than one per pair. Rounding to 3 dp is ~8 hours.
    dt_round = np.round(dt, 3)
    uniq, inverse = np.unique(dt_round, return_inverse=True)
    return i.astype(int), j.astype(int), uniq, inverse


def _neg_loglik(theta, i, j, uniq, inverse, n_states) -> float:
    Q = build_generator(np.exp(theta), n_states)
    P = transition_probabilities(Q, uniq)
    p = P[inverse, i, j]
    return -float(np.log(np.maximum(p, _EPS)).sum())


def fit_ctmc(
    pairs: pd.DataFrame,
    states: list,
    state_names: list | None = None,
    init: np.ndarray | None = None,
    seed: int = 42,
    n_restarts: int = 4,
) -> CTMCFit:
    """Maximum-likelihood fit of a continuous-time Markov chain to panel data.

    Several random restarts are used because the likelihood surface for a
    3-state generator is not guaranteed unimodal; the best optimum is kept.
    """
    n_states = len(states)
    state_names = state_names or [str(s) for s in states]
    i, j, uniq, inverse = _pair_arrays(pairs, states)

    n_rates = n_states * (n_states - 1)
    rng = np.random.default_rng(seed)

    if init is None:
        # A crude but reasonable start: the empirical off-diagonal frequency
        # divided by the median interval gives a rate of the right magnitude.
        counts = pd.crosstab(pairs["from_state"], pairs["to_state"]).reindex(
            index=states, columns=states, fill_value=0
        )
        totals = counts.sum(axis=1).replace(0, 1)
        median_dt = float(np.median(pairs["interval_years"]))
        approx = (counts.div(totals, axis=0) / max(median_dt, 1e-6)).to_numpy()
        init = np.log(
            np.clip([approx[a, b] for a in range(n_states) for b in range(n_states) if a != b],
                    1e-4, None)
        )

    best = None
    for attempt in range(n_restarts):
        theta0 = np.asarray(init, dtype=float)
        if attempt > 0:
            theta0 = theta0 + rng.normal(0.0, 0.5, size=n_rates)
        theta0 = np.clip(theta0, *_LOG_RATE_BOUNDS)

        res = minimize(
            _neg_loglik,
            theta0,
            args=(i, j, uniq, inverse, n_states),
            method="L-BFGS-B",
            bounds=[_LOG_RATE_BOUNDS] * n_rates,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10},
        )
        if best is None or res.fun < best.fun:
            best = res

    Q = build_generator(np.exp(best.x), n_states)
    return CTMCFit(
        Q=Q,
        states=list(states),
        state_names=list(state_names),
        loglik=-float(best.fun),
        n_pairs=len(pairs),
        n_patients=int(pairs["PATNO"].nunique()),
        converged=bool(best.success),
        n_params=n_rates,
    )


def bootstrap_rates(
    pairs: pd.DataFrame,
    states: list,
    state_names: list,
    n_boot: int = 300,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Patient-level bootstrap confidence intervals for the fitted rates.

    Resampling is over *patients*, not pairs: a patient contributes several
    pairs and those are not independent observations of the sampling process,
    even though the Markov property makes them conditionally independent
    given the current state.
    """
    rng = np.random.default_rng(seed)
    patients = pairs["PATNO"].unique()
    by_patient = {p: g for p, g in pairs.groupby("PATNO")}

    draws = []
    for _ in range(n_boot):
        picked = rng.choice(patients, size=len(patients), replace=True)
        sample = pd.concat([by_patient[p] for p in picked], ignore_index=True)
        try:
            fit = fit_ctmc(sample, states, state_names, n_restarts=1, seed=seed)
        except (ValueError, np.linalg.LinAlgError):
            continue
        off = [(a, b) for a in range(len(states)) for b in range(len(states)) if a != b]
        draws.append([fit.Q[a, b] for a, b in off])

    arr = np.asarray(draws)
    off = [(a, b) for a in range(len(states)) for b in range(len(states)) if a != b]
    rows = []
    for k, (a, b) in enumerate(off):
        rows.append(
            {
                "from_state": state_names[a],
                "to_state": state_names[b],
                "ci_low": float(np.quantile(arr[:, k], alpha / 2)),
                "ci_high": float(np.quantile(arr[:, k], 1 - alpha / 2)),
                "n_draws": len(arr),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Comparison models
# ---------------------------------------------------------------------------
def fit_dtmc(pairs: pd.DataFrame, states: list) -> np.ndarray:
    """Discrete-time chain: the empirical one-step matrix, ignoring interval.

    This is the model the proposal originally sketched and the baseline the
    continuous-time model has to beat.
    """
    counts = pd.crosstab(pairs["from_state"], pairs["to_state"]).reindex(
        index=states, columns=states, fill_value=0
    )
    # Laplace smoothing keeps the held-out log-likelihood finite if a
    # transition unseen in training occurs in the test set.
    smoothed = counts.to_numpy(dtype=float) + 0.5
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def fit_marginal(pairs: pd.DataFrame, states: list) -> np.ndarray:
    """Null model: next state drawn from its marginal, ignoring current state."""
    counts = pairs["to_state"].value_counts().reindex(states, fill_value=0).to_numpy(float)
    p = (counts + 0.5) / (counts + 0.5).sum()
    return np.tile(p, (len(states), 1))


# ---------------------------------------------------------------------------
#  Held-out evaluation
# ---------------------------------------------------------------------------
def predict_pairs(model, pairs: pd.DataFrame, states: list) -> np.ndarray:
    """Predicted next-state distribution for every pair.

    `model` is either a CTMCFit (interval-aware) or a plain matrix
    (interval-agnostic). Returns shape (n_pairs, n_states).
    """
    i, _, uniq, inverse = _pair_arrays(pairs, states)
    if isinstance(model, CTMCFit):
        P = transition_probabilities(model.Q, uniq)
        return P[inverse, i, :]
    return np.asarray(model)[i, :]


def evaluate(model, pairs: pd.DataFrame, states: list, label: str) -> dict:
    """Predictive performance of a model on a set of pairs."""
    index = {s: k for k, s in enumerate(states)}
    j = pairs["to_state"].map(index).to_numpy(int)
    probs = predict_pairs(model, pairs, states)

    p_true = np.maximum(probs[np.arange(len(j)), j], _EPS)
    loglik = float(np.log(p_true).sum())

    onehot = np.zeros_like(probs)
    onehot[np.arange(len(j)), j] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    return {
        "model": label,
        "n_pairs": len(pairs),
        "loglik": loglik,
        "loglik_per_pair": loglik / len(pairs),
        "accuracy": float((probs.argmax(axis=1) == j).mean()),
        "brier": brier,
    }


def calibration_table(
    model, pairs: pd.DataFrame, states: list, target_state, n_bins: int = 5
) -> pd.DataFrame:
    """Predicted versus observed probability of landing in `target_state`.

    A well-calibrated model puts the observed frequency inside each bin close
    to the mean prediction for that bin.
    """
    index = {s: k for k, s in enumerate(states)}
    target = index[target_state]
    probs = predict_pairs(model, pairs, states)[:, target]
    actual = (pairs["to_state"].to_numpy() == target_state).astype(float)

    edges = np.quantile(probs, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    binned = pd.cut(probs, bins=np.unique(edges), include_lowest=True)

    frame = pd.DataFrame({"bin": binned, "predicted": probs, "observed": actual})
    out = frame.groupby("bin", observed=True).agg(
        n=("observed", "size"),
        mean_predicted=("predicted", "mean"),
        observed_rate=("observed", "mean"),
    )
    return out.reset_index()


def clinical_table(fit: CTMCFit, horizons=(1, 2, 3, 5)) -> pd.DataFrame:
    """The proposal's output: P(state at t | current state), per horizon.

    One row per (current state, horizon); columns are the probability of being
    in each state at that horizon.
    """
    rows = []
    for s_from in fit.state_names:
        rows.append(
            {"current_state": s_from, "horizon_years": 0.0,
             **{s: (1.0 if s == s_from else 0.0) for s in fit.state_names}}
        )
        for t in horizons:
            P = fit.P(float(t))
            rows.append(
                {"current_state": s_from, "horizon_years": float(t),
                 **{s: float(P.loc[s_from, s]) for s in fit.state_names}}
            )
    return pd.DataFrame(rows)
