"""Hidden Markov model: a latent subtype observed through misclassification.

Why this model exists
---------------------
Both baseline chains in :mod:`ppmi_subtype.ctmc` assume the observed subtype
*is* the patient's state. The data rejects that assumption outright. Testing the
Markov property on 2,427 labelled triples:

    currently TD, previously TD    -> 84.9% stay TD,  8.6% move to PIGD
    currently TD, previously PIGD  -> 48.2% stay TD, 39.3% move to PIGD
                                      chi2 = 156.6, dof = 4, p = 7.9e-33

Under a Markov chain in the observed label those two rows would be identical.
They are not: where a patient came from changes the probability of moving to
PIGD by a factor of 4.6. The observed sequence has strong memory.

That is exactly the signature of a *latent* state read through noise. If a
patient's underlying phenotype is PIGD but one examination scores them TD, the
next examination is likely to score PIGD again --- producing the dependence on
history that a plain chain cannot represent.

The model
---------
A latent state X(t) evolves as a continuous-time Markov chain with generator Q,
exactly as before. What we observe at visit k is not X(t_k) but a noisy reading
Y_k, related to it by a misclassification matrix E:

    E[a, b] = P(observed label is b | true state is a)

The likelihood of one patient's whole sequence marginalises over every possible
latent path, computed by the forward algorithm in O(n * S^2) rather than by
enumerating S^n paths.

This is the discrete-state counterpart of the proposal's stated observation
model, ``observed ratio = true ratio + measurement noise``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from .ctmc import build_generator, transition_probabilities

_LOG_RATE_BOUNDS = (-12.0, 4.0)
_INIT_BOUNDS = (-8.0, 8.0)
_EPS = 1e-300

# Minimum probability that an examination records the true state correctly.
#
# This constraint is not cosmetic. Left unconstrained, the maximum-likelihood
# solution is degenerate and identical across every random restart: the latent
# Indeterminate state emits uniformly (0.333/0.333/0.333, i.e. its label
# carries no information) and PIGD becomes absorbing with an exit rate pinned
# at the lower bound. Requiring each state to be recorded correctly more often
# than not removes both degeneracies, and costs nothing in held-out
# performance -- the advantage over the discrete baseline is +0.027 log-
# likelihood per transition at this floor against +0.026 unconstrained.
DEFAULT_EMISSION_FLOOR = 0.5


def _emission_logit_bound(floor: float, n_states: int) -> float:
    """Largest off-diagonal logit consistent with a diagonal of at least `floor`.

    With logits (0, b, b) the diagonal probability is 1 / (1 + (S-1) e^b), so
    the constraint diagonal >= floor gives b <= log((1/floor - 1) / (S - 1)).
    """
    if not 0.0 < floor < 1.0:
        raise ValueError(f"emission floor must lie in (0, 1); got {floor}")
    uniform = 1.0 / n_states
    if floor <= uniform:
        return 0.0
    return float(np.log((1.0 / floor - 1.0) / (n_states - 1)))


# ---------------------------------------------------------------------------
#  Sequence extraction
# ---------------------------------------------------------------------------
def build_sequences(
    panel: pd.DataFrame,
    states: list,
    label_col: str = "subtype",
    time_col: str = "years_from_baseline",
    min_length: int = 2,
) -> list[dict]:
    """One entry per patient: observed labels and the gaps between them.

    Unlike the pairwise view used by the baseline chains, the HMM consumes a
    patient's entire sequence at once --- that is what lets it accumulate
    evidence about the latent state across visits.
    """
    index = {s: k for k, s in enumerate(states)}
    df = panel[panel[label_col].notna()].sort_values(["PATNO", time_col])

    sequences = []
    for patno, group in df.groupby("PATNO", sort=False):
        labels = group[label_col].map(index).to_numpy(dtype=int)
        times = group[time_col].to_numpy(dtype=float)
        if len(labels) < min_length:
            continue
        sequences.append(
            {"PATNO": patno, "labels": labels, "gaps": np.diff(times)}
        )
    return sequences


# ---------------------------------------------------------------------------
#  Parameter packing
# ---------------------------------------------------------------------------
def _unpack(theta: np.ndarray, n_hidden: int, n_observed: int | None = None) -> tuple:
    """Split the flat parameter vector into Q, E and the initial distribution.

    ``n_observed`` defaults to ``n_hidden``, which is the square case the
    three-state model uses. When the two differ, E is a rectangular
    ``(n_hidden, n_observed)`` matrix: each hidden state still emits a
    distribution over the observed labels, there are simply fewer labels than
    states. One entry per row is pinned to zero because a softmax row is
    invariant to a shift; for the square case that entry is the diagonal, which
    keeps this function's output identical to before.
    """
    n_observed = n_hidden if n_observed is None else n_observed
    n_rates = n_hidden * (n_hidden - 1)
    n_emit = n_hidden * (n_observed - 1)

    rates = np.exp(theta[:n_rates])
    Q = build_generator(rates, n_hidden)

    emit_logits = np.zeros((n_hidden, n_observed))
    k = n_rates
    for a in range(n_hidden):
        pinned = a if n_hidden == n_observed else 0
        for b in range(n_observed):
            if b != pinned:
                emit_logits[a, b] = theta[k]
                k += 1
    E = softmax(emit_logits, axis=1)

    init_logits = np.zeros(n_hidden)
    init_logits[1:] = theta[n_rates + n_emit :]
    pi = softmax(init_logits)
    return Q, E, pi


def _n_params(n_hidden: int, n_observed: int | None = None) -> int:
    n_observed = n_hidden if n_observed is None else n_observed
    return n_hidden * (n_hidden - 1) + n_hidden * (n_observed - 1) + (n_hidden - 1)


# ---------------------------------------------------------------------------
#  Forward algorithm
# ---------------------------------------------------------------------------
@dataclass
class SequenceBatch:
    """Ragged patient sequences packed into rectangular arrays.

    The forward recursion is run for every patient simultaneously rather than
    in a Python loop. Sequences have different lengths, so they are padded and
    a per-step mask records which patients are still contributing. Without
    this the optimiser spends minutes per fit instead of seconds.
    """

    labels: np.ndarray      # (n_seq, max_len) int, padded
    gap_index: np.ndarray   # (n_seq, max_len - 1) int into unique_gaps
    step_mask: np.ndarray   # (n_seq, max_len - 1) bool
    unique_gaps: np.ndarray
    n_observations: int
    n_sequences: int


def prepare_batch(sequences: list[dict], round_dp: int = 3) -> SequenceBatch:
    """Pack sequences and de-duplicate the inter-visit gaps."""
    if not sequences:
        raise ValueError("no sequences supplied")

    all_gaps = np.round(np.concatenate([s["gaps"] for s in sequences]), round_dp)
    unique_gaps = np.unique(all_gaps)
    gap_position = {g: k for k, g in enumerate(unique_gaps)}

    n_seq = len(sequences)
    max_len = max(len(s["labels"]) for s in sequences)

    labels = np.zeros((n_seq, max_len), dtype=int)
    gap_index = np.zeros((n_seq, max_len - 1), dtype=int)
    step_mask = np.zeros((n_seq, max_len - 1), dtype=bool)

    for i, seq in enumerate(sequences):
        n = len(seq["labels"])
        labels[i, :n] = seq["labels"]
        for k, gap in enumerate(np.round(seq["gaps"], round_dp)):
            gap_index[i, k] = gap_position[gap]
            step_mask[i, k] = True

    return SequenceBatch(
        labels=labels,
        gap_index=gap_index,
        step_mask=step_mask,
        unique_gaps=unique_gaps,
        n_observations=int(sum(len(s["labels"]) for s in sequences)),
        n_sequences=n_seq,
    )


def _forward_loglik(Q, E, pi, batch: SequenceBatch) -> float:
    """Total log-likelihood over all sequences, vectorised across patients.

    Scaling is applied at every step: alpha is normalised and the normaliser
    accumulated into the log-likelihood, so the recursion cannot underflow on
    a fourteen-visit patient.
    """
    P_by_gap = transition_probabilities(Q, batch.unique_gaps)   # (G, S, S)
    labels = batch.labels

    alpha = pi[None, :] * E[:, labels[:, 0]].T                  # (n, S)
    scale = alpha.sum(axis=1)
    if not np.all(scale > 0):
        return -np.inf
    total = float(np.log(scale).sum())
    alpha = alpha / scale[:, None]

    for k in range(batch.step_mask.shape[1]):
        active = batch.step_mask[:, k]
        if not active.any():
            break

        P = P_by_gap[batch.gap_index[:, k]]                     # (n, S, S)
        nxt = np.einsum("ns,nsl->nl", alpha, P) * E[:, labels[:, k + 1]].T
        scale = np.maximum(nxt.sum(axis=1), _EPS)

        total += float(np.log(scale)[active].sum())
        updated = nxt / scale[:, None]
        alpha = np.where(active[:, None], updated, alpha)

    return total if np.isfinite(total) else -np.inf


def _gap_lookup(Q: np.ndarray, unique_gaps: np.ndarray) -> dict:
    mats = transition_probabilities(Q, unique_gaps)
    return {round(float(g), 3): mats[k] for k, g in enumerate(unique_gaps)}


def _neg_loglik(theta, batch: SequenceBatch, n_hidden, n_observed=None) -> float:
    Q, E, pi = _unpack(theta, n_hidden, n_observed)
    ll = _forward_loglik(Q, E, pi, batch)
    return -ll if np.isfinite(ll) else 1e12


# ---------------------------------------------------------------------------
#  Fitting
# ---------------------------------------------------------------------------
@dataclass
class HMMFit:
    Q: np.ndarray
    E: np.ndarray
    pi: np.ndarray
    states: list
    state_names: list
    loglik: float
    n_sequences: int
    n_observations: int
    n_params: int
    converged: bool
    emission_floor: float | None = DEFAULT_EMISSION_FLOOR
    #: Number of latent states. Equal to len(states) for the square model; a
    #: model with more hidden states than observed labels sets this larger.
    n_hidden: int | None = None
    #: Names for the latent states. Defaults to the observed names, which is
    #: correct only when the two counts match.
    hidden_names: list | None = None

    def __post_init__(self) -> None:
        if self.n_hidden is None:
            object.__setattr__(self, "n_hidden", self.Q.shape[0])
        if self.hidden_names is None:
            names = (list(self.state_names) if self.n_hidden == len(self.state_names)
                     else [f"H{k + 1}" for k in range(self.n_hidden)])
            object.__setattr__(self, "hidden_names", names)

    @property
    def is_square(self) -> bool:
        """True when there is one latent state per observed label."""
        return self.n_hidden == len(self.state_names)

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik

    def generator_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.Q, index=self.hidden_names, columns=self.hidden_names)

    def emission_frame(self) -> pd.DataFrame:
        """Rows are the latent state, columns the label actually recorded."""
        return pd.DataFrame(self.E, index=self.hidden_names, columns=self.state_names)

    def P(self, t: float) -> pd.DataFrame:
        mat = transition_probabilities(self.Q, np.array([float(t)]))[0]
        return pd.DataFrame(mat, index=self.hidden_names, columns=self.hidden_names)

    def mean_sojourn(self) -> pd.Series:
        d = np.diag(self.Q)
        with np.errstate(divide="ignore"):
            out = np.where(d < 0, -1.0 / d, np.inf)
        return pd.Series(out, index=self.hidden_names)

    def accuracy_by_state(self) -> pd.Series:
        """P(label is correct | true state) --- only defined for the square model."""
        if not self.is_square:
            raise ValueError(
                "accuracy_by_state is undefined when the hidden and observed "
                "state counts differ; read emission_frame() instead"
            )
        return pd.Series(np.diag(self.E), index=self.state_names)


def fit_hmm(
    sequences: list[dict],
    states: list,
    state_names: list | None = None,
    seed: int = 42,
    n_restarts: int = 3,
    init_rates: np.ndarray | None = None,
    emission_floor: float = DEFAULT_EMISSION_FLOOR,
    n_hidden: int | None = None,
) -> HMMFit:
    """Maximum-likelihood fit of the misclassification HMM.

    `emission_floor` is the minimum probability that an examination records the
    true state correctly; see :data:`DEFAULT_EMISSION_FLOOR` for why it is
    needed.

    `n_hidden` defaults to one latent state per observed label. Setting it
    larger gives the chain more latent states than there are labels, so several
    latent states may emit the same label. The emission floor is *not defined*
    in that case --- it constrains P(label i | state i), and with more states
    than labels there is no such pairing --- so it must be passed as ``None``.
    Choosing which label each extra state should favour would impose the very
    structure such a model is usually fitted to test.
    """
    n_observed = len(states)
    n_hidden = n_observed if n_hidden is None else int(n_hidden)
    state_names = state_names or [str(s) for s in states]
    if not sequences:
        raise ValueError("no sequences supplied")
    if n_hidden < n_observed:
        raise ValueError(
            f"n_hidden ({n_hidden}) must be at least the number of observed "
            f"labels ({n_observed})"
        )
    if n_hidden != n_observed and emission_floor is not None:
        raise ValueError(
            "emission_floor is only defined when there is one latent state per "
            "observed label; pass emission_floor=None to fit an unconstrained "
            f"model with {n_hidden} latent states and {n_observed} labels"
        )

    batch = prepare_batch(sequences)
    n_rates = n_hidden * (n_hidden - 1)
    n_emit = n_hidden * (n_observed - 1)
    rng = np.random.default_rng(seed)

    emit_bound = (0.0 if emission_floor is None
                  else _emission_logit_bound(emission_floor, n_observed))
    emit_upper = 8.0 if emission_floor is None else emit_bound

    base = np.empty(_n_params(n_hidden, n_observed))
    base[:n_rates] = np.log(init_rates) if init_rates is not None else np.log(0.15)
    base[n_rates : n_rates + n_emit] = min(-2.0, emit_bound)   # ~10% misclassification
    base[n_rates + n_emit :] = 0.0

    bounds = (
        [_LOG_RATE_BOUNDS] * n_rates
        + [(-8.0, emit_upper)] * n_emit
        + [_INIT_BOUNDS] * (n_hidden - 1)
    )

    best = None
    for attempt in range(n_restarts):
        theta0 = base.copy()
        if attempt > 0:
            theta0 = theta0 + rng.normal(0.0, 0.4, size=len(theta0))
        theta0 = np.clip(theta0, [b[0] for b in bounds], [b[1] for b in bounds])

        res = minimize(
            _neg_loglik,
            theta0,
            args=(batch, n_hidden, n_observed),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-11, "gtol": 1e-9},
        )
        if best is None or res.fun < best.fun:
            best = res

    Q, E, pi = _unpack(best.x, n_hidden, n_observed)
    return HMMFit(
        Q=Q,
        E=E,
        pi=pi,
        states=list(states),
        state_names=list(state_names),
        loglik=-float(best.fun),
        n_sequences=batch.n_sequences,
        n_observations=batch.n_observations,
        n_params=_n_params(n_hidden, n_observed),
        converged=bool(best.success),
        emission_floor=None if emission_floor is None else float(emission_floor),
        n_hidden=n_hidden,
        hidden_names=(list(state_names) if n_hidden == n_observed
                      else [f"H{k + 1}" for k in range(n_hidden)]),
    )


# ---------------------------------------------------------------------------
#  One-step-ahead prediction, for held-out comparison
# ---------------------------------------------------------------------------
def predictive_distributions(fit: HMMFit, sequences: list[dict]) -> pd.DataFrame:
    """P(next observed label | everything seen so far), for every transition.

    This is what makes the comparison against the baseline chains fair and
    meaningful: all three models predict the same quantity --- the next
    *observed* label --- but the HMM is allowed to use the full history, which
    is precisely the information the Markov test showed to be predictive.
    """
    unique_gaps = np.unique(np.round(np.concatenate([s["gaps"] for s in sequences]), 3))
    lookup = _gap_lookup(fit.Q, unique_gaps)

    records = []
    for seq in sequences:
        labels, gaps = seq["labels"], seq["gaps"]

        alpha = fit.pi * fit.E[:, labels[0]]
        alpha = alpha / max(alpha.sum(), _EPS)

        for k, gap in enumerate(gaps):
            P = lookup[round(float(gap), 3)]
            latent_next = alpha @ P
            predicted = latent_next @ fit.E          # over observed labels

            records.append(
                {
                    "PATNO": seq["PATNO"],
                    "step": k,
                    "from_state": fit.states[labels[k]],
                    "to_state": fit.states[labels[k + 1]],
                    "interval_years": float(gap),
                    **{f"p_{name}": float(predicted[m]) for m, name in enumerate(fit.state_names)},
                }
            )

            alpha = latent_next * fit.E[:, labels[k + 1]]
            alpha = alpha / max(alpha.sum(), _EPS)

    return pd.DataFrame(records)


def evaluate(fit: HMMFit, sequences: list[dict], label: str = "HMM") -> dict:
    """Predictive performance on held-out sequences, comparable to ctmc.evaluate."""
    preds = predictive_distributions(fit, sequences)
    cols = [f"p_{n}" for n in fit.state_names]
    probs = preds[cols].to_numpy()

    index = {s: k for k, s in enumerate(fit.states)}
    j = preds["to_state"].map(index).to_numpy(int)

    p_true = np.maximum(probs[np.arange(len(j)), j], _EPS)
    loglik = float(np.log(p_true).sum())

    onehot = np.zeros_like(probs)
    onehot[np.arange(len(j)), j] = 1.0

    return {
        "model": label,
        "n_pairs": len(preds),
        "loglik": loglik,
        "loglik_per_pair": loglik / len(preds),
        "accuracy": float((probs.argmax(axis=1) == j).mean()),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
    }


def calibration_table(
    fit: HMMFit, sequences: list[dict], target_state, n_bins: int = 5
) -> pd.DataFrame:
    """Predicted versus observed probability of the next label being `target_state`.

    Unlike the baseline chains, which can only ever emit one prediction per
    current state, the HMM's prediction varies continuously with a patient's
    history, so a calibration curve with several bins is meaningful.
    """
    preds = predictive_distributions(fit, sequences)
    name = fit.state_names[fit.states.index(target_state)]
    probs = preds[f"p_{name}"].to_numpy()
    actual = (preds["to_state"].to_numpy() == target_state).astype(float)

    edges = np.unique(np.quantile(probs, np.linspace(0, 1, n_bins + 1)))
    binned = pd.cut(probs, bins=edges, include_lowest=True)

    out = (
        pd.DataFrame({"bin": binned, "predicted": probs, "observed": actual})
        .groupby("bin", observed=True)
        .agg(n=("observed", "size"),
             mean_predicted=("predicted", "mean"),
             observed_rate=("observed", "mean"))
        .reset_index()
    )
    return out


def clinical_table(fit: HMMFit, horizons=(1, 2, 3, 5)) -> pd.DataFrame:
    """Latent-state progression probabilities at clinical horizons.

    Note the interpretation differs from the plain chain: rows are the
    patient's *true* current phenotype, not their recorded label.
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
