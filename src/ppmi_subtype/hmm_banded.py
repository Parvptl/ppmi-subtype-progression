"""The CT-HMM of :mod:`ppmi_subtype.hmm`, reading joint (subtype, band) observations.

What changes and what does not
------------------------------
Nothing about the latent process changes. There are the same three hidden
states, the same continuous-time generator Q built by
:func:`ctmc.build_generator`, the same interval-aware transition matrices
exp(Q dt), the same forward filtering recursion (``hmm._forward_loglik`` is
reused verbatim --- it was already written for a rectangular emission), the
same patient sequences, the same seed and the same optimiser settings.

The one change is the observation alphabet. Instead of three recorded labels
the model sees the joint categories of
:class:`ppmi_subtype.ratio_bands.JointAlphabet`, so the emission matrix is
rectangular: ``n_hidden x n_categories``.

The emission constraint
-----------------------
The square model's floor is ``P(label a | state a) >= 0.5`` --- an examination
records the true state correctly more often than not. Applying that floor to a
single joint category would mean something quite different (and with ~40
categories, something impossible), so it is applied to the *subtype marginal*
instead. The emission factorises as

    E[a, (s, b)] = M[a, s] * C[a, s, b]

with M a 3x3 subtype misclassification matrix, parameterised and bounded
exactly as the square model's E is, and C[a, s, .] a distribution over the
bands of subtype s. Then

    sum_b E[a, (s, b)] = M[a, s]

so the floor ``M[a, a] >= 0.5`` *is* the square model's floor, on the same
quantity, with the same logit bound. The square model is the special case
where every subtype has one band: with ``C == 1`` the parameter vector, the
bounds and the likelihood all reduce to :func:`hmm.fit_hmm`'s, which
``tests/test_hmm_banded.py`` checks numerically.

What the band can and cannot do
-------------------------------
Because the subtype marginal of the emission is M, the aggregated next-visit
prediction is exactly ``alpha P(dt) M`` --- the same functional form as the
square model's ``alpha P(dt) E``. The bands therefore cannot change the
prediction by reshaping the output distribution; they can only change it
through ``alpha``, the filtered belief about the latent state. That is
precisely the experiment's question --- does knowing *where in the subtype* a
visit sat sharpen the latent state --- and it keeps the comparison with the
square model clean.

The cost is parameters. Each band of each subtype adds one free parameter per
hidden state, so a ~40-category alphabet carries ~120 emission parameters
against the square model's 6. The fit stays identified (the floor pins M, the
bounds pin C) but it is a far larger model on the same 4,295 observations,
and that, not the latent process, is what the experiment puts at risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from . import hmm
from .ctmc import build_generator, transition_probabilities
from .hmm import _EPS, _INIT_BOUNDS, _LOG_RATE_BOUNDS, _emission_logit_bound
from .hmm_stability import OPTIONS, projected_gradient_norm
from .ratio_bands import JointAlphabet, aggregate_to_subtype

#: Box bound on a within-subtype band logit. Wide enough that the fit is not
#: bound-constrained in practice, finite so that a category absent from one
#: fold's training data keeps a tiny but non-zero probability rather than an
#: exact zero, which would make a held-out visit in that category impossible.
BAND_LOGIT_BOUND = 8.0

#: The gradient is exact (see `loglik_and_grad`), so one L-BFGS-B iteration
#: costs one likelihood evaluation rather than n_params + 1 and scipy's
#: default 15,000-evaluation cap can no longer bind before `OPTIONS`'
#: maxiter = 3000 does. Kept as an override for callers that want to fit with
#: finite differences for comparison.
DEFAULT_MAXFUN = None


# ---------------------------------------------------------------------------
#  Parameter layout
# ---------------------------------------------------------------------------
def n_band_params(alphabet: JointAlphabet, n_hidden: int) -> int:
    """Free within-subtype band logits: one per extra category, per hidden state."""
    return n_hidden * sum(len(b) - 1 for b in alphabet.blocks)


def n_params(alphabet: JointAlphabet, n_hidden: int) -> int:
    n_sub = len(alphabet.blocks)
    return (n_hidden * (n_hidden - 1)                 # rates
            + n_hidden * (n_sub - 1)                  # subtype misclassification
            + n_band_params(alphabet, n_hidden)       # bands within subtype
            + (n_hidden - 1))                         # initial distribution


def _check_square(alphabet: JointAlphabet, n_hidden: int, emission_floor) -> None:
    n_sub = len(alphabet.blocks)
    if n_hidden != n_sub and emission_floor is not None:
        raise ValueError(
            "the emission floor pairs hidden state a with subtype a, so it needs "
            f"one hidden state per subtype; got {n_hidden} states and {n_sub} "
            "subtypes. Pass emission_floor=None to fit without it."
        )


def unpack(theta: np.ndarray, alphabet: JointAlphabet, n_hidden: int) -> tuple:
    """Split the flat vector into ``(Q, E, pi, M)``.

    E is ``n_hidden x n_categories`` and already carries the factorisation, so
    every downstream user (the forward recursion, the predictor) sees a plain
    rectangular emission matrix.
    """
    n_sub = len(alphabet.blocks)
    n_rates = n_hidden * (n_hidden - 1)

    Q = build_generator(np.exp(theta[:n_rates]), n_hidden)

    k = n_rates
    m_logits = np.zeros((n_hidden, n_sub))
    for a in range(n_hidden):
        pinned = a if n_hidden == n_sub else 0
        for s in range(n_sub):
            if s != pinned:
                m_logits[a, s] = theta[k]
                k += 1
    M = softmax(m_logits, axis=1)

    E = np.zeros((n_hidden, alphabet.n_categories))
    for a in range(n_hidden):
        for s, block in enumerate(alphabet.blocks):
            logits = np.zeros(len(block))
            n_free = len(block) - 1
            if n_free:
                logits[1:] = theta[k:k + n_free]
                k += n_free
            E[a, list(block)] = M[a, s] * softmax(logits)

    init_logits = np.zeros(n_hidden)
    init_logits[1:] = theta[k:]
    return Q, E, softmax(init_logits), M


def bounds_for(alphabet: JointAlphabet, n_hidden: int, emission_floor) -> list[tuple]:
    """Bounds on each parameter. The M block's are the square model's exactly."""
    _check_square(alphabet, n_hidden, emission_floor)
    n_sub = len(alphabet.blocks)
    n_rates = n_hidden * (n_hidden - 1)
    m_upper = (BAND_LOGIT_BOUND if emission_floor is None
               else _emission_logit_bound(emission_floor, n_sub))
    return (
        [_LOG_RATE_BOUNDS] * n_rates
        + [(-8.0, m_upper)] * (n_hidden * (n_sub - 1))
        + [(-BAND_LOGIT_BOUND, BAND_LOGIT_BOUND)] * n_band_params(alphabet, n_hidden)
        + [_INIT_BOUNDS] * (n_hidden - 1)
    )


def default_start(alphabet: JointAlphabet, n_hidden: int, emission_floor) -> np.ndarray:
    """The unperturbed start: :func:`hmm.fit_hmm`'s, extended by uniform bands.

    Rates and the subtype misclassification block are started exactly where
    the square model starts them. The band logits start at zero, i.e. uniform
    over the bands of each subtype, which is the one neutral choice --- it
    assumes nothing about which bands a hidden state favours, so the fit is not
    nudged towards the answer the experiment is testing for.
    """
    _check_square(alphabet, n_hidden, emission_floor)
    n_sub = len(alphabet.blocks)
    n_rates = n_hidden * (n_hidden - 1)
    n_m = n_hidden * (n_sub - 1)
    m_bound = (0.0 if emission_floor is None
               else _emission_logit_bound(emission_floor, n_sub))

    theta = np.zeros(n_params(alphabet, n_hidden))
    theta[:n_rates] = np.log(0.15)
    theta[n_rates:n_rates + n_m] = min(-2.0, m_bound)
    return theta


def starts_for(alphabet: JointAlphabet, n_hidden: int, emission_floor,
               seed: int, n_restarts: int) -> np.ndarray:
    """`n_restarts` starts: the default, then N(0, 0.4) perturbations of it.

    Identical scheme, scale and seeding to :func:`hmm.fit_hmm` and to
    :func:`next_visit.audited_starts`, so the two models' fits differ in the
    alphabet and nothing else.
    """
    base = default_start(alphabet, n_hidden, emission_floor)
    bounds = np.asarray(bounds_for(alphabet, n_hidden, emission_floor))
    rng = np.random.default_rng(seed)
    out = []
    for attempt in range(n_restarts):
        theta = base.copy()
        if attempt > 0:
            theta = theta + rng.normal(0.0, 0.4, size=len(theta))
        out.append(np.clip(theta, bounds[:, 0], bounds[:, 1]))
    return np.array(out)


# ---------------------------------------------------------------------------
#  Fitting
# ---------------------------------------------------------------------------
@dataclass
class BandedFit:
    """A fitted joint-observation CT-HMM."""

    Q: np.ndarray
    E: np.ndarray                 # (n_hidden, n_categories)
    pi: np.ndarray
    M: np.ndarray                 # (n_hidden, n_subtypes) subtype marginal of E
    alphabet: JointAlphabet
    subtype_names: list
    loglik: float
    n_sequences: int
    n_observations: int
    n_params: int
    n_hidden: int
    converged: bool
    emission_floor: float | None
    theta: np.ndarray = field(repr=False, default=None)

    def emission_frame(self) -> pd.DataFrame:
        """Rows the latent state, columns the joint category."""
        return pd.DataFrame(self.E, index=self.hidden_names,
                            columns=self.alphabet.labels)

    def subtype_emission_frame(self) -> pd.DataFrame:
        """The subtype marginal M --- directly comparable with the square model's E."""
        return pd.DataFrame(self.M, index=self.hidden_names,
                            columns=list(self.subtype_names))

    @property
    def hidden_names(self) -> list:
        return (list(self.subtype_names) if self.n_hidden == len(self.subtype_names)
                else [f"H{k + 1}" for k in range(self.n_hidden)])

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik

    def generator_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.Q, index=self.hidden_names, columns=self.hidden_names)

    def band_conditional_frame(self) -> pd.DataFrame:
        """P(band | hidden state, subtype) --- E divided by its subtype marginal."""
        rows = []
        for a in range(self.n_hidden):
            for s, block in enumerate(self.alphabet.blocks):
                mass = self.M[a, s]
                for c in block:
                    rows.append({
                        "hidden_state": self.hidden_names[a],
                        "subtype": self.subtype_names[s],
                        "category": self.alphabet.labels[c],
                        "p_joint": float(self.E[a, c]),
                        "p_band_given_subtype": float(self.E[a, c] / mass)
                        if mass > 0 else float("nan"),
                    })
        return pd.DataFrame(rows)


def _neg_loglik(theta, batch, alphabet: JointAlphabet, n_hidden: int) -> float:
    Q, E, pi, _ = unpack(theta, alphabet, n_hidden)
    ll = hmm._forward_loglik(Q, E, pi, batch)
    return -ll if np.isfinite(ll) else 1e12


@dataclass
class StartRecord:
    """One optimiser run, recorded like :class:`hmm_stability.StartRecord`."""

    start: int
    loglik: float
    success: bool
    message: str
    n_iter: int
    n_fev: int
    proj_grad: float
    theta: np.ndarray = field(repr=False, default=None)


def fit_banded(
    sequences: list[dict],
    alphabet: JointAlphabet,
    subtype_names,
    n_hidden: int | None = None,
    emission_floor: float | None = hmm.DEFAULT_EMISSION_FLOOR,
    starts: np.ndarray | None = None,
    seed: int = 42,
    n_restarts: int = 4,
    maxfun: int | None = DEFAULT_MAXFUN,
) -> tuple[BandedFit, list[StartRecord]]:
    """Maximum-likelihood fit over the joint alphabet; returns the best fit and every start.

    `sequences` carry joint category codes in ``labels`` --- see
    :func:`ppmi_subtype.ratio_bands.JointAlphabet.code`. Everything else is as
    :func:`hmm.fit_hmm`: same objective, same bounds, same L-BFGS-B settings,
    same restart scheme, and the exact gradient of :func:`loglik_and_grad`.
    `maxfun` overrides scipy's function-evaluation cap; the default leaves it
    alone, because with an exact gradient `OPTIONS`' iteration cap binds first.
    """
    if not sequences:
        raise ValueError("no sequences supplied")
    subtype_names = list(subtype_names)
    n_hidden = len(subtype_names) if n_hidden is None else int(n_hidden)
    _check_square(alphabet, n_hidden, emission_floor)

    batch = hmm.prepare_batch(sequences)
    if int(batch.labels.max()) >= alphabet.n_categories:
        raise ValueError("a sequence carries a category outside the alphabet")

    bounds = bounds_for(alphabet, n_hidden, emission_floor)
    if starts is None:
        starts = starts_for(alphabet, n_hidden, emission_floor, seed, n_restarts)
    options = dict(OPTIONS, **({"maxfun": int(maxfun)} if maxfun else {}))

    records, best = [], None
    for s, theta0 in enumerate(starts):
        res = minimize(loglik_and_grad, theta0, args=(batch, alphabet, n_hidden),
                       method="L-BFGS-B", jac=True, bounds=bounds, options=options)
        records.append(StartRecord(
            start=s, loglik=-float(res.fun), success=bool(res.success),
            message=str(res.message), n_iter=int(res.nit), n_fev=int(res.nfev),
            proj_grad=projected_gradient_norm(res.x, res.jac, bounds),
            theta=np.array(res.x)))
        if best is None or res.fun < best.fun:
            best = res

    Q, E, pi, M = unpack(best.x, alphabet, n_hidden)
    fit = BandedFit(
        Q=Q, E=E, pi=pi, M=M, alphabet=alphabet, subtype_names=subtype_names,
        loglik=-float(best.fun), n_sequences=batch.n_sequences,
        n_observations=batch.n_observations, n_params=len(bounds),
        n_hidden=n_hidden, converged=bool(best.success),
        emission_floor=None if emission_floor is None else float(emission_floor),
        theta=np.array(best.x))
    return fit, records


# ---------------------------------------------------------------------------
#  One-step-ahead prediction, aggregated back to the three subtypes
# ---------------------------------------------------------------------------
def predictive_distributions(fit: BandedFit, sequences: list[dict]) -> pd.DataFrame:
    """P(next observed category | history), summed into the three subtypes.

    The filtering recursion is :func:`hmm.predictive_distributions`'s, over the
    joint alphabet: alpha is advanced with the joint category actually seen and
    propagated through exp(Q dt). The prediction is then summed over the joint
    categories of each subtype and renormalised, which is the quantity the
    evaluation is defined on and the one the square model also predicts.

    ``p_joint_sum`` is kept per row: it is the aggregated mass before
    renormalising and must be 1 to floating-point precision, which is the
    check that no probability was lost in the aggregation.
    """
    gaps = np.round(np.concatenate([s["gaps"] for s in sequences]), 3)
    unique_gaps = np.unique(gaps)
    mats = transition_probabilities(fit.Q, unique_gaps)
    lookup = {round(float(g), 3): mats[k] for k, g in enumerate(unique_gaps)}

    records = []
    for seq in sequences:
        labels, seq_gaps = seq["labels"], seq["gaps"]

        alpha = fit.pi * fit.E[:, labels[0]]
        alpha = alpha / max(alpha.sum(), _EPS)

        for k, gap in enumerate(seq_gaps):
            latent_next = alpha @ lookup[round(float(gap), 3)]
            joint = latent_next @ fit.E
            subtype_probs = aggregate_to_subtype(joint, fit.alphabet)

            records.append({
                "PATNO": seq["PATNO"],
                "step": k,
                "from_category": fit.alphabet.labels[labels[k]],
                "to_category": fit.alphabet.labels[labels[k + 1]],
                "interval_years": float(gap),
                "p_joint_sum": float(joint.sum()),
                **{f"p_{name}": float(subtype_probs[m])
                   for m, name in enumerate(fit.subtype_names)},
            })

            alpha = latent_next * fit.E[:, labels[k + 1]]
            alpha = alpha / max(alpha.sum(), _EPS)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
#  Exact gradient
# ---------------------------------------------------------------------------
# Why this is here
# ----------------
# The square model has 14 parameters, so L-BFGS-B's finite-difference gradient
# costs 15 likelihood evaluations and scipy's default 15,000-evaluation cap
# leaves it ~1,000 iterations --- far more than it needs. The joint model has
# ~137, so the same gradient costs 138 evaluations: a probe on the full cohort
# ran 935 iterations in 203 s, exhausted a 138,000-evaluation budget and
# stopped with a projected gradient of 2.3e-2, twenty times the tolerance this
# project calls converged. Reporting that against a converged baseline would
# confound the experiment's question with the optimiser's failure.
#
# So the gradient is computed exactly instead, by forward-backward. Nothing
# else changes: same objective, same bounds, same L-BFGS-B, same restarts,
# same seed. `tests/test_hmm_banded.py` checks it against finite differences,
# and the experiment refits the *square* model through this same code path as
# a control, so the effect of the gradient method is measured rather than
# assumed.
_PHI_TOL = 1e-9


def _forward_backward(Q, E, pi, batch):
    """Scaled forward-backward over the padded batch.

    Returns ``(loglik, gamma, alpha, v, valid, P_by_gap)`` where ``gamma`` is
    the posterior state marginal per visit (zero on padding) and ``v`` is the
    scaled backward message the pair marginals are built from:

        xi[n, k, i, j] = alpha[n, k, i] * P[n, i, j] * v[n, k, j]

    so the transition matrix cancels out of ``xi / P`` and no division by a
    possibly-tiny transition probability is ever taken.
    """
    labels, gap_index, step_mask = batch.labels, batch.gap_index, batch.step_mask
    n_seq, max_len = labels.shape
    n_states = len(pi)

    P_by_gap = transition_probabilities(Q, batch.unique_gaps)

    valid = np.zeros((n_seq, max_len), dtype=bool)
    valid[:, 0] = True
    valid[:, 1:] = step_mask

    alpha = np.zeros((n_seq, max_len, n_states))
    scale = np.ones((n_seq, max_len))

    a = pi[None, :] * E[:, labels[:, 0]].T
    s = a.sum(axis=1)
    if not np.all(s > 0):
        return -np.inf, None, None, None, None, None
    scale[:, 0] = s
    alpha[:, 0, :] = a / s[:, None]

    for k in range(max_len - 1):
        active = step_mask[:, k]
        P = P_by_gap[gap_index[:, k]]
        nxt = np.einsum("ns,nsl->nl", alpha[:, k, :], P) * E[:, labels[:, k + 1]].T
        s = np.maximum(nxt.sum(axis=1), _EPS)
        scale[:, k + 1] = np.where(active, s, 1.0)
        alpha[:, k + 1, :] = np.where(active[:, None], nxt / s[:, None], alpha[:, k, :])

    loglik = float(np.log(scale[valid]).sum())
    if not np.isfinite(loglik):
        return -np.inf, None, None, None, None, None

    beta = np.ones((n_seq, max_len, n_states))
    v = np.zeros((n_seq, max_len - 1, n_states))
    for k in range(max_len - 2, -1, -1):
        active = step_mask[:, k]
        P = P_by_gap[gap_index[:, k]]
        v[:, k, :] = (E[:, labels[:, k + 1]].T * beta[:, k + 1, :]
                      / scale[:, k + 1][:, None]) * active[:, None]
        beta[:, k, :] = np.where(active[:, None],
                                 np.einsum("nij,nj->ni", P, v[:, k, :]), 1.0)

    gamma = alpha * beta * valid[:, :, None]
    return loglik, gamma, alpha, v, valid, P_by_gap


def _phi(w: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """Divided differences of exp(w t): Phi[g, m, l] = (e_m - e_l) / (w_m - w_l).

    The diagonal, and any pair of eigenvalues too close to divide by, take the
    limit ``t exp(w t)`` evaluated at the pair's mean.
    """
    t = np.asarray(gaps, dtype=float)[:, None, None]
    wm = w[None, :, None]
    wl = w[None, None, :]
    e = np.exp(w[None, :, None] * t)                    # (G, S, 1)
    diff = wm - wl
    close = np.abs(diff) < _PHI_TOL
    safe = np.where(close, 1.0, diff)
    out = (e - np.swapaxes(e, 1, 2)) / safe
    limit = t * np.exp((wm + wl) / 2.0 * t)
    return np.where(close, limit, out)


def _dQ_from_gap_sums(Q: np.ndarray, gaps: np.ndarray, G: np.ndarray):
    """Pull dL/dP(t) back through the matrix exponential to dL/dQ.

    ``G[g]`` is dL/dP(gaps[g]) accumulated over every transition observed at
    that elapsed time. For Q = V diag(w) V^-1 the Frechet derivative of
    exp(Q t) gives

        dL/dQ = V^-T (B . Phi) V^T,   B[g] = V^T G[g] V^-T

    with ``.`` the elementwise product and Phi the divided differences above
    (Daleckii-Krein). Returns None when V is too ill-conditioned to trust, so
    the caller can fall back to finite differences on the six rates.
    """
    try:
        w, V = np.linalg.eig(Q)
        Vinv = np.linalg.inv(V)
    except np.linalg.LinAlgError:                        # pragma: no cover
        return None
    if np.linalg.cond(V) > 1e12:
        return None

    B = np.einsum("ji,gjk,lk->gil", V, G.astype(complex), Vinv)
    total = (B * _phi(w, gaps)).sum(axis=0)
    return np.real(Vinv.T @ total @ V.T)


def loglik_and_grad(theta, batch, alphabet: JointAlphabet, n_hidden: int):
    """Negative log-likelihood and its exact gradient, for L-BFGS-B.

    The emission and initial-distribution gradients come from the posterior
    state marginals; the rate gradient from the pair marginals pulled back
    through exp(Q dt). If that pull-back is not numerically safe at the current
    parameters, the six rate components are taken by central differences
    instead and the rest stay exact.
    """
    Q, E, pi, M = unpack(theta, alphabet, n_hidden)
    ll, gamma, alpha, v, valid, P_by_gap = _forward_backward(Q, E, pi, batch)
    if not np.isfinite(ll):
        return 1e12, np.zeros_like(theta)

    n_sub = len(alphabet.blocks)
    n_rates = n_hidden * (n_hidden - 1)
    grad = np.zeros_like(theta)

    # -- emission: expected counts per (hidden state, category) --------------
    flat_labels = batch.labels[valid]
    counts = np.zeros((n_hidden, alphabet.n_categories))
    np.add.at(counts.T, flat_labels, gamma[valid])
    per_subtype = np.column_stack([counts[:, list(b)].sum(axis=1)
                                   for b in alphabet.blocks])
    total_mass = counts.sum(axis=1)

    k = n_rates
    for a in range(n_hidden):
        pinned = a if n_hidden == n_sub else 0
        for s in range(n_sub):
            if s != pinned:
                grad[k] = -(per_subtype[a, s] - M[a, s] * total_mass[a])
                k += 1
    for a in range(n_hidden):
        for s, block in enumerate(alphabet.blocks):
            if len(block) < 2:
                continue
            mass = per_subtype[a, s]
            cond = E[a, list(block)] / M[a, s] if M[a, s] > 0 else np.zeros(len(block))
            for pos in range(1, len(block)):
                c = block[pos]
                grad[k] = -(counts[a, c] - cond[pos] * mass)
                k += 1

    # -- initial distribution -------------------------------------------------
    gamma_first = gamma[:, 0, :].sum(axis=0)
    grad[k:] = -(gamma_first[1:] - pi[1:] * gamma_first.sum())

    # -- rates ---------------------------------------------------------------
    # xi / P = alpha_i v_j, so the accumulator needs no division; sum it per
    # unique elapsed time and pull the whole thing back through exp(Q dt).
    n_gaps = len(batch.unique_gaps)
    G = np.zeros((n_gaps, n_hidden, n_hidden))
    active = batch.step_mask
    if active.any():
        outer = np.einsum("nks,nkl->nksl", alpha[:, :-1, :], v)
        np.add.at(G, batch.gap_index[active], outer[active])

    dQ = _dQ_from_gap_sums(Q, batch.unique_gaps, G)
    if dQ is None:                                       # pragma: no cover
        grad[:n_rates] = _rate_grad_fd(theta, batch, alphabet, n_hidden, n_rates)
    else:
        rates = np.exp(theta[:n_rates])
        j = 0
        for i in range(n_hidden):
            for m in range(n_hidden):
                if i != m:
                    grad[j] = -(dQ[i, m] - dQ[i, i]) * rates[j]
                    j += 1

    return -ll, grad


def _rate_grad_fd(theta, batch, alphabet, n_hidden, n_rates, h: float = 1e-6):
    """Central-difference gradient of the negative log-likelihood in the rates."""
    out = np.zeros(n_rates)
    for i in range(n_rates):
        up, dn = theta.copy(), theta.copy()
        up[i] += h
        dn[i] -= h
        out[i] = (_neg_loglik(up, batch, alphabet, n_hidden)
                  - _neg_loglik(dn, batch, alphabet, n_hidden)) / (2 * h)
    return out
