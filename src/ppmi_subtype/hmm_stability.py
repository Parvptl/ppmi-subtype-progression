"""Multi-start and held-out machinery for the HMM stability study (Stage 4).

`hmm.fit_hmm` is the audited fitter and is not changed. Its restarts perturb a
single fixed start by N(0, 0.4); an earlier probe (`scripts/_multistart5.py`)
found every outer seed returned a bit-identical solution, so those restarts
never tested whether the likelihood has other optima. This module runs the
*same objective* (`hmm._neg_loglik`, same bounds, same optimiser settings) from
genuinely dispersed starting points and keeps a record of every start:

    final log-likelihood, scipy's success flag AND its termination message,
    iterations, function evaluations, and the projected-gradient norm.

The projected gradient matters because `fit_hmm` optimises with finite-
difference gradients. L-BFGS-B then often stops with
``ABNORMAL_TERMINATION_IN_LNSRCH`` at a point that is optimal to the precision
the finite differences allow; ``success`` is False there, but the projected
gradient is small. Reporting "converged" from ``success`` alone conflates that
with a genuine failure to optimise.

Label switching. For the square model (one latent state per label) the pinned
diagonal emission logit identifies each state, so solutions are comparable
directly. With more latent states than labels the hidden states are
exchangeable; `canonical_order` sorts them by their expected emitted label so
parameters from different starts can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from . import hmm
from .hmm import _emission_logit_bound, _n_params, _neg_loglik, _unpack, prepare_batch

#: As `hmm.fit_hmm`. scipy's default function-evaluation cap (maxfun = 15000)
#: applies unless a caller overrides it; with finite-difference gradients a
#: 34-parameter model spends 35 evaluations per gradient, so the cap can bind
#: long before `maxiter`.
OPTIONS = {"maxiter": 3000, "ftol": 1e-11, "gtol": 1e-9}


# ---------------------------------------------------------------------------
#  Parameter space
# ---------------------------------------------------------------------------
def bounds_for(n_hidden: int, n_observed: int, emission_floor):
    """Exactly the bounds `hmm.fit_hmm` uses for this configuration."""
    if n_hidden != n_observed and emission_floor is not None:
        raise ValueError("emission floor is undefined unless n_hidden == n_observed")
    n_rates = n_hidden * (n_hidden - 1)
    n_emit = n_hidden * (n_observed - 1)
    emit_upper = (8.0 if emission_floor is None
                  else _emission_logit_bound(emission_floor, n_observed))
    return ([hmm._LOG_RATE_BOUNDS] * n_rates + [(-8.0, emit_upper)] * n_emit
            + [hmm._INIT_BOUNDS] * (n_hidden - 1))


def default_start(n_hidden: int, n_observed: int, emission_floor) -> np.ndarray:
    """`fit_hmm`'s unperturbed restart-0 start, reproduced exactly."""
    n_rates = n_hidden * (n_hidden - 1)
    n_emit = n_hidden * (n_observed - 1)
    emit_bound = (0.0 if emission_floor is None
                  else _emission_logit_bound(emission_floor, n_observed))
    base = np.empty(_n_params(n_hidden, n_observed))
    base[:n_rates] = np.log(0.15)
    base[n_rates:n_rates + n_emit] = min(-2.0, emit_bound)
    base[n_rates + n_emit:] = 0.0
    return base


def dispersed_starts(n_hidden: int, n_observed: int, emission_floor, n_starts: int,
                     seed: int) -> np.ndarray:
    """Start 0 is the project default; the rest are spread over the space.

    Rates are log-uniform on [0.005, 2] per year (sojourns from 6 months to 200
    years), off-diagonal emission logits uniform on [-6, min(upper, 2)], initial
    logits uniform on [-3, 3]. All are clipped to the fitter's bounds.
    """
    b = np.array(bounds_for(n_hidden, n_observed, emission_floor))
    n_rates = n_hidden * (n_hidden - 1)
    n_emit = n_hidden * (n_observed - 1)
    rng = np.random.default_rng(seed)
    out = np.empty((n_starts, len(b)))
    out[0] = default_start(n_hidden, n_observed, emission_floor)
    for s in range(1, n_starts):
        th = np.empty(len(b))
        th[:n_rates] = rng.uniform(np.log(0.005), np.log(2.0), n_rates)
        hi = min(b[n_rates, 1], 2.0)
        th[n_rates:n_rates + n_emit] = rng.uniform(-6.0, hi, n_emit)
        th[n_rates + n_emit:] = rng.uniform(-3.0, 3.0, n_hidden - 1)
        out[s] = th
    return np.clip(out, b[:, 0], b[:, 1])


def projected_gradient_norm(x, g, bounds, tol: float = 1e-8) -> float:
    """Inf-norm of the gradient after removing components blocked by a bound."""
    if g is None:
        return float("nan")
    b = np.asarray(bounds)
    pg = np.array(g, dtype=float)
    pg[(x <= b[:, 0] + tol) & (pg > 0)] = 0.0
    pg[(x >= b[:, 1] - tol) & (pg < 0)] = 0.0
    return float(np.max(np.abs(pg))) if len(pg) else 0.0


# ---------------------------------------------------------------------------
#  Multi-start fit
# ---------------------------------------------------------------------------
@dataclass
class StartRecord:
    start: int
    loglik: float
    success: bool
    message: str
    n_iter: int
    n_fev: int
    proj_grad: float
    theta: np.ndarray = field(repr=False)


def canonical_order(Q, E, pi):
    """Permute latent states so they are sorted by expected emitted label.

    The square model is already identified (its states are pinned to labels),
    so this is only applied when there are more latent states than labels.
    """
    if Q.shape[0] == E.shape[1]:
        return Q, E, pi
    order = np.argsort(E @ np.arange(E.shape[1]), kind="mergesort")
    return Q[np.ix_(order, order)], E[order], pi[order]


def fit_multistart(sequences, states, n_hidden: int, emission_floor, starts: np.ndarray,
                   state_names=None, maxfun: int | None = None):
    """Optimise from every start; return (best HMMFit, list[StartRecord]).

    `maxfun` overrides scipy's function-evaluation cap; None keeps the default
    the audited fitter runs with.
    """
    options = dict(OPTIONS, **({"maxfun": int(maxfun)} if maxfun else {}))
    n_observed = len(states)
    batch = prepare_batch(sequences)
    bounds = bounds_for(n_hidden, n_observed, emission_floor)
    records = []
    best = None
    for s, th0 in enumerate(starts):
        res = minimize(_neg_loglik, th0, args=(batch, n_hidden, n_observed),
                       method="L-BFGS-B", bounds=bounds, options=options)
        rec = StartRecord(start=s, loglik=-float(res.fun), success=bool(res.success),
                          message=str(res.message), n_iter=int(res.nit),
                          n_fev=int(res.nfev),
                          proj_grad=projected_gradient_norm(res.x, res.jac, bounds),
                          theta=np.array(res.x))
        records.append(rec)
        if best is None or res.fun < best.fun:
            best = res
    Q, E, pi = _unpack(best.x, n_hidden, n_observed)
    names = list(state_names) if state_names else [str(s) for s in states]
    fit = hmm.HMMFit(
        Q=Q, E=E, pi=pi, states=list(states), state_names=names,
        loglik=-float(best.fun), n_sequences=batch.n_sequences,
        n_observations=batch.n_observations,
        n_params=_n_params(n_hidden, n_observed), converged=bool(best.success),
        emission_floor=None if emission_floor is None else float(emission_floor),
        n_hidden=n_hidden,
        hidden_names=(names if n_hidden == n_observed
                      else [f"H{k + 1}" for k in range(n_hidden)]))
    return fit, records


def solution_spread(records, n_hidden: int, n_observed: int, tol: float = 0.01):
    """How many starts reach the best optimum, and how different the rest are.

    Returns the share of starts within `tol` log-likelihood of the best, the
    number of distinct log-likelihood levels (rounded to `tol`), and, among the
    starts at the best level, the largest absolute difference in any canonical
    Q or E entry from the best start --- which is how one detects a flat ridge
    (same likelihood, different parameters).
    """
    ll = np.array([r.loglik for r in records])
    best_i = int(np.argmax(ll))
    near = np.abs(ll - ll[best_i]) <= tol
    Qb, Eb, _ = canonical_order(*_unpack(records[best_i].theta, n_hidden, n_observed))
    dq, de = 0.0, 0.0
    for r, ok in zip(records, near):
        if ok:
            Q, E, _ = canonical_order(*_unpack(r.theta, n_hidden, n_observed))
            dq = max(dq, float(np.max(np.abs(Q - Qb))))
            de = max(de, float(np.max(np.abs(E - Eb))))
    levels = len(np.unique(np.round(ll / tol)))
    return {"best_loglik": float(ll[best_i]), "best_start": best_i,
            "share_within_tol": float(near.mean()), "n_loglik_levels": int(levels),
            "worst_loglik": float(ll.min()),
            "max_dQ_at_best": dq, "max_dE_at_best": de,
            "share_success_flag": float(np.mean([r.success for r in records])),
            "share_proj_grad_lt_1e-3": float(np.mean([r.proj_grad < 1e-3 for r in records]))}


# ---------------------------------------------------------------------------
#  Folds and decision rules
# ---------------------------------------------------------------------------
def patient_folds(patients, k: int, seed: int) -> dict:
    """Deterministic patient -> fold assignment (0..k-1), balanced in size."""
    p = np.array(sorted(patients))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(p))
    return {int(p[i]): int(r % k) for r, i in enumerate(perm)}


def argmax_rule(probs: np.ndarray) -> np.ndarray:
    return np.asarray(probs).argmax(axis=1)


def prior_corrected_rule(probs: np.ndarray, train_prior: np.ndarray) -> np.ndarray:
    """argmax of p_k / prior_k: the Bayes rule for balanced (equal-cost-per-class)
    accuracy when the model is calibrated. `train_prior` must come from the
    training patients only."""
    return (np.asarray(probs) / np.asarray(train_prior)[None, :]).argmax(axis=1)


def auc_one_vs_rest(score: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    from scipy.stats import rankdata
    score = np.asarray(score, float)
    positive = np.asarray(positive, bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
