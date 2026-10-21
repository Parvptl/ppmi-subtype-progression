"""Next-visit TD-vs-PIGD prediction, conditional on the next label not being Indeterminate.

What this module predicts
-------------------------
For every prediction opportunity --- a labelled visit that is followed by
another labelled visit for the same patient --- it produces exactly two
numbers:

    P_TD   = P(next observed label is TD   | next observed label is TD or PIGD,
                history up to and including the current visit)
    P_PIGD = P(next observed label is PIGD | next observed label is TD or PIGD,
                history up to and including the current visit)

These are **conditional** probabilities, not unconditional next-visit
probabilities. The generative models in this project (`hmm`, `ctmc`) place mass
on all three recorded labels, Indeterminate included; this module renormalises
the two extreme labels against each other:

    P_TD = p_TD / (p_TD + p_PIGD),   P_PIGD = p_PIGD / (p_TD + p_PIGD)

and refuses to produce a number at all when that denominator is not positive.
The mass ``p_Indeterminate`` is therefore *not* thrown away silently: it is
carried through in the prediction table as ``p_Indeterminate_raw`` and
reported, and the renormalisation is what gives the two outputs their
conditional reading. A visit whose actual next label is Indeterminate is a
case where the conditioning event did not occur, so it is excluded from the
two-class scoring and counted separately rather than being scored as a hit or
a miss against a class it never had.

Indeterminate as evidence
-------------------------
Indeterminate is never recoded. It stays a third observed label in every
patient's input history: the latent models emit it, the observed-label chains
transition through it, and the discriminative model reads it as a feature. The
only thing the two-class restriction touches is the *target*.

Which models are compared
-------------------------
Seven, on identical prediction cases, spanning the two axes the final question
asks about --- latent state and elapsed time:

    CT-HMM (latent, interval-aware)       `hmm.fit_hmm`, unchanged
    CTMC (observed, interval-aware)       `ctmc.fit_ctmc`, unchanged
    Markov (observed, discrete)           `ctmc.fit_dtmc`, unchanged
    Direct logistic (history + interval)  this module
    Direct logistic (history only)        this module
    Persistence (last non-Indeterminate)  a rule, no fitted parameters
    Marginal (two-class prior)            the null

The existing HMM *is* a CT-HMM: its latent state evolves as a continuous-time
Markov chain with generator Q built by :func:`ctmc.build_generator`, and it
carries an explicit emission (misclassification) matrix E over the three
recorded labels. :func:`hmm.predictive_distributions` already runs the
filtering recursion forward over the actual inter-visit gap. The plain CTMC
has no observation model and is a chain in the *observed* label; it is included
as the no-latent-layer control, not as a second CT-HMM.

Filtering, not smoothing
------------------------
Every model's prediction for visit k+1 uses labels 0..k only. For the CT-HMM
this is the forward (filtering) recursion: alpha is advanced with the label
actually seen at k and then propagated through exp(Q * dt). No backward pass
is run and no quantity depending on label k+1 enters the prediction.

The one piece of future information every model receives is *when* the next
visit happens --- dt is an input, as it is throughout this project's panel
models and in `loo`. The model is told the date of the next examination, never
its result. That is the standard convention for interval-censored panel data,
and it is an assumption worth stating because PPMI's visit schedule is not
independent of disease course.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import ctmc, hmm, hmm_stability, loo

#: Observed labels, in the order every probability array uses.
STATE_NAMES = loo.STATE_NAMES                       # ["TD", "Indeterminate", "PIGD"]
PROB_COLS = loo.PROB_COLS                           # ["p_TD", "p_Indeterminate", "p_PIGD"]
TD, IND, PIGD = 0, 1, 2
TWO_CLASS = ["TD", "PIGD"]

#: Smallest TD+PIGD mass we are willing to renormalise. Below this the
#: conditional split is numerically meaningless, so no prediction is emitted
#: and the case is reported as undefined rather than resolved arbitrarily.
MIN_DENOMINATOR = 1e-12

#: How an exact tie is broken. P_TD == P_PIGD == 0.5 carries no information, so
#: the rule has to come from outside the model. PIGD is the class with the
#: worse motor prognosis, so a tie resolves to PIGD: under a symmetric
#: probability the cheaper error to make is the one that does not miss a
#: PIGD conversion. The rule is deterministic, independent of the fit, and
#: ties are counted in the output so their influence can be checked.
TIE_RULE = "PIGD"

MODEL_CTHMM = "CT-HMM (latent, interval-aware)"
MODEL_CTMC = "CTMC (observed, interval-aware)"
MODEL_DTMC = "Markov (observed, discrete)"
MODEL_DIRECT = "Direct logistic (history + interval)"
MODEL_DIRECT_NOINT = "Direct logistic (history only)"
MODEL_PERSIST = "Persistence (last non-Indeterminate)"
MODEL_MARGINAL = "Marginal (two-class prior)"

#: Models whose probabilities are a rule rather than an estimate. Their Brier
#: score is well defined but their log loss is only finite because it is
#: clipped, so the two are not comparable with the fitted models.
RULE_BASED = (MODEL_PERSIST,)

ALL_MODELS = (MODEL_CTHMM, MODEL_CTMC, MODEL_DTMC, MODEL_DIRECT,
              MODEL_DIRECT_NOINT, MODEL_PERSIST, MODEL_MARGINAL)

_LOGLOSS_CLIP = 1e-12


# ---------------------------------------------------------------------------
#  The two conditional probabilities
# ---------------------------------------------------------------------------
def conditional_two_class(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Renormalise a three-label distribution onto {TD, PIGD}.

    `probs` has shape (n, 3) in the order [TD, Indeterminate, PIGD]. Returns
    ``(P_TD, P_PIGD, defined)``. Where ``defined`` is False the TD+PIGD mass
    was not positive and both probabilities are NaN --- the caller must drop
    those cases, not patch them.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"expected an (n, 3) array of probabilities; got {probs.shape}")

    denom = probs[:, TD] + probs[:, PIGD]
    defined = np.isfinite(denom) & (denom > MIN_DENOMINATOR)

    p_td = np.full(len(probs), np.nan)
    p_pigd = np.full(len(probs), np.nan)
    p_td[defined] = probs[defined, TD] / denom[defined]
    p_pigd[defined] = probs[defined, PIGD] / denom[defined]

    # Make the pair sum to exactly one in floating point rather than to
    # 1 +/- 1 ulp, so the downstream invariant check is about the model and
    # not about the order of two divisions.
    p_pigd[defined] = 1.0 - p_td[defined]
    return p_td, p_pigd, defined


def predicted_state(p_td: np.ndarray, p_pigd: np.ndarray) -> np.ndarray:
    """TD where P_TD > P_PIGD, PIGD where P_PIGD > P_TD, :data:`TIE_RULE` on a tie.

    Undefined rows (NaN) stay as an empty string; they are not predictions.
    """
    p_td = np.asarray(p_td, dtype=float)
    p_pigd = np.asarray(p_pigd, dtype=float)
    out = np.full(len(p_td), "", dtype=object)
    ok = np.isfinite(p_td) & np.isfinite(p_pigd)
    out[ok & (p_td > p_pigd)] = "TD"
    out[ok & (p_pigd > p_td)] = "PIGD"
    out[ok & (p_td == p_pigd)] = TIE_RULE
    return out


# ---------------------------------------------------------------------------
#  Prediction cases, and the history features available at each one
# ---------------------------------------------------------------------------
#: Features of a prediction case. Every one is a function of the labels at
#: visits 0..k and of the elapsed time to visit k+1 -- nothing else. The
#: interval features are listed last so the no-interval variant can be built
#: by dropping a suffix.
FEATURES = (
    "current_is_TD",
    "current_is_IND",
    "prev_noind_is_PIGD",
    "prev_noind_missing",
    "frac_hist_PIGD",
    "frac_hist_IND",
    "log_run_length",
    "log_n_history",
    "current_years",
    "log_interval",
)
INTERVAL_FEATURES = ("log_interval",)


def build_cases(data: loo.LOOData) -> pd.DataFrame:
    """One row per prediction opportunity, with its causal history features.

    Built from the same sequences the models are fitted on, so the row order
    for a patient is its visit order and ``visit_index`` means what it means in
    :func:`loo.annotate`: the prediction for visit ``visit_index + 1`` from
    visits ``0 .. visit_index``.
    """
    meta = data.meta.set_index(["PATNO", "visit_index"])
    rows = []

    for seq in data.sequences:
        labels, gaps = seq["labels"], seq["gaps"]
        for k, gap in enumerate(gaps):
            history = labels[: k + 1]
            current = int(history[-1])

            # Last label in the history that is not Indeterminate, which is the
            # only thing a persistence rule can carry forward.
            non_ind = history[history != IND]
            prev_noind = int(non_ind[-1]) if len(non_ind) else -1

            # Length of the run of the current label at the end of the history.
            run = 1
            while run < len(history) and history[-1 - run] == current:
                run += 1

            cur = meta.loc[(seq["PATNO"], k)]
            tgt = meta.loc[(seq["PATNO"], k + 1)]

            rows.append({
                "PATNO": seq["PATNO"],
                "visit_index": k,
                "n_history": k + 1,
                "current_event": cur["EVENT_ID"],
                "current_years": float(cur["years_from_baseline"]),
                "target_event": tgt["EVENT_ID"],
                "target_years": float(tgt["years_from_baseline"]),
                "interval_years": float(gap),
                "current_index": current,
                "target_index": int(labels[k + 1]),
                "prev_noind_index": prev_noind,
                "current_is_TD": float(current == TD),
                "current_is_IND": float(current == IND),
                "prev_noind_is_PIGD": float(prev_noind == PIGD),
                "prev_noind_missing": float(prev_noind < 0),
                "frac_hist_PIGD": float(np.mean(history == PIGD)),
                "frac_hist_IND": float(np.mean(history == IND)),
                "log_run_length": float(np.log(run)),
                "log_n_history": float(np.log(k + 1)),
                "log_interval": float(np.log(gap)),
            })

    cases = pd.DataFrame(rows)
    cases["current_label"] = [STATE_NAMES[i] for i in cases["current_index"]]
    cases["actual_next_label"] = [STATE_NAMES[i] for i in cases["target_index"]]
    cases["eligible"] = cases["target_index"].isin([TD, PIGD])
    return cases


# ---------------------------------------------------------------------------
#  Direct two-class model: L2-penalised logistic regression on the history
# ---------------------------------------------------------------------------
#: Ridge penalty on the standardised coefficients, intercept excluded. Fixed in
#: advance rather than tuned, because tuning it on held-out folds would be the
#: leak this whole design exists to avoid. With ~2,700 training cases and ten
#: features the fit is not penalty-sensitive; the value only keeps the
#: optimiser away from a separating direction in the rare-transition corners.
DIRECT_PENALTY = 1.0


@dataclass
class DirectFit:
    """A fitted direct two-class model, plus the preprocessing it learned."""

    beta: np.ndarray                 # [intercept, coefficients...]
    centre: np.ndarray               # training-set feature means
    scale: np.ndarray                # training-set feature SDs (zero -> 1)
    features: tuple[str, ...]
    penalty: float
    loglik: float
    converged: bool
    proj_grad: float
    n_train: int
    n_train_patients: int

    def coefficient_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "term": ("intercept",) + tuple(self.features),
            "coefficient": self.beta,
        })

    def predict_p_pigd(self, cases: pd.DataFrame) -> np.ndarray:
        """P(next label is PIGD | next label is TD or PIGD) for each case."""
        X = (cases[list(self.features)].to_numpy(dtype=float) - self.centre) / self.scale
        z = self.beta[0] + X @ self.beta[1:]
        return _expit(z)


def _expit(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _penalised_objective(beta, X, y, penalty):
    """Negative log-likelihood plus a ridge penalty, with its exact gradient."""
    z = beta[0] + X @ beta[1:]
    p = _expit(z)
    # log-likelihood via logaddexp, which is exact at |z| where exp overflows.
    ll = float(np.sum(y * z - np.logaddexp(0.0, z)))
    pen = 0.5 * penalty * float(beta[1:] @ beta[1:])

    resid = p - y
    grad = np.empty_like(beta)
    grad[0] = float(resid.sum())
    grad[1:] = X.T @ resid + penalty * beta[1:]
    return -ll + pen, grad


def fit_direct(
    cases: pd.DataFrame,
    features: tuple[str, ...] = FEATURES,
    penalty: float = DIRECT_PENALTY,
) -> DirectFit:
    """Fit the direct conditional two-class model on eligible training cases.

    Only cases whose *actual* next label is TD or PIGD can train a model of
    P(PIGD | TD or PIGD): the conditioning event has to have occurred. Cases
    whose next label is Indeterminate are excluded from the fit and are
    still used, as history, by every case that follows them.

    Standardisation is part of the fit. The means and SDs returned in the
    :class:`DirectFit` are the training set's, so a held-out patient is scored
    with preprocessing it did not contribute to.
    """
    train = cases[cases["eligible"]]
    if not len(train):
        raise ValueError("no eligible training cases")

    X_raw = train[list(features)].to_numpy(dtype=float)
    y = (train["target_index"].to_numpy() == PIGD).astype(float)

    centre = X_raw.mean(axis=0)
    scale = X_raw.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    X = (X_raw - centre) / scale

    beta0 = np.zeros(len(features) + 1)
    # Start the intercept at the empirical log-odds: the optimum when every
    # coefficient is zero, so the optimiser starts on the right scale.
    rate = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    beta0[0] = np.log(rate / (1 - rate))

    res = minimize(_penalised_objective, beta0, args=(X, y, penalty),
                   method="L-BFGS-B", jac=True,
                   options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})

    return DirectFit(
        beta=np.array(res.x, dtype=float),
        centre=centre,
        scale=scale,
        features=tuple(features),
        penalty=float(penalty),
        loglik=-float(_penalised_objective(res.x, X, y, 0.0)[0]),
        converged=bool(res.success),
        proj_grad=float(np.max(np.abs(res.jac))) if res.jac is not None else float("nan"),
        n_train=int(len(train)),
        n_train_patients=int(train["PATNO"].nunique()),
    )


# ---------------------------------------------------------------------------
#  Reproducing the audited HMM fit while recording its convergence
# ---------------------------------------------------------------------------
def audited_starts(
    n_hidden: int, n_observed: int, emission_floor: float | None,
    seed: int, n_restarts: int,
) -> np.ndarray:
    """The exact starting points :func:`hmm.fit_hmm` uses, as an array.

    `fit_hmm` reports convergence from scipy's ``success`` flag alone.
    :mod:`hmm_stability` documents why that is not enough here: with
    finite-difference gradients L-BFGS-B routinely stops with
    ``ABNORMAL_TERMINATION_IN_LNSRCH`` at a point whose projected gradient is
    tiny, and calling that a failed fit would overstate the problem. Feeding
    these starts to :func:`hmm_stability.fit_multistart` runs the identical
    optimisation --- same objective, same bounds, same options --- and returns
    the projected gradient and the termination message as well, so a fold can
    be reported as converged, flat-line-search, or genuinely failed.
    """
    base = hmm_stability.default_start(n_hidden, n_observed, emission_floor)
    bounds = np.asarray(hmm_stability.bounds_for(n_hidden, n_observed, emission_floor))
    rng = np.random.default_rng(seed)

    starts = []
    for attempt in range(n_restarts):
        theta = base.copy()
        if attempt > 0:
            theta = theta + rng.normal(0.0, 0.4, size=len(theta))
        starts.append(np.clip(theta, bounds[:, 0], bounds[:, 1]))
    return np.array(starts)


# ---------------------------------------------------------------------------
#  One LOPO fold
# ---------------------------------------------------------------------------
#: A fold is reported as converged when scipy says so, or when the projected
#: gradient at the chosen optimum is below this. See :func:`audited_starts`.
PROJ_GRAD_TOL = 1e-3


def run_fold(
    data: loo.LOOData,
    cases: pd.DataFrame,
    held_out,
    seed: int,
    n_restarts: int,
    emission_floor: float | None,
) -> pd.DataFrame:
    """Hold out one patient, fit every candidate on the rest, predict its visits.

    The held-out patient contributes nothing to any fit: not to the CT-HMM
    sequences, not to the observed-label chains' transition pairs, and not to
    the direct model's standardisation. :func:`loo.assert_no_leakage` checks
    the first two and this function checks the third by construction --- the
    training frame is filtered on PATNO before any statistic is taken from it.
    """
    train_sequences = [s for s in data.sequences if s["PATNO"] != held_out]
    train_pairs = data.pairs[data.pairs["PATNO"] != held_out]
    loo.assert_no_leakage(held_out, train_sequences, train_pairs)

    train_cases = cases[cases["PATNO"] != held_out]
    test_cases = cases[cases["PATNO"] == held_out].sort_values("visit_index")
    test_cases = test_cases.reset_index(drop=True)
    if held_out in set(train_cases["PATNO"]):           # pragma: no cover
        raise loo.LeakageError(f"held-out patient {held_out} is among the training cases")

    test_sequence = next(s for s in data.sequences if s["PATNO"] == held_out)
    n = len(test_cases)
    i = test_cases["current_index"].to_numpy(int)

    diagnostics: dict[str, dict] = {}
    raw: dict[str, np.ndarray] = {}

    # -- CT-HMM: latent CTMC + emission matrix, filtered forward over dt -----
    starts = audited_starts(len(data.states), len(data.states), emission_floor,
                            seed, n_restarts)
    fit, records = hmm_stability.fit_multistart(
        train_sequences, data.states, n_hidden=len(data.states),
        emission_floor=emission_floor, starts=starts, state_names=STATE_NAMES)
    best = max(records, key=lambda r: r.loglik)

    preds = hmm.predictive_distributions(fit, [test_sequence])
    preds = preds.sort_values("step").reset_index(drop=True)
    if len(preds) != n or not np.array_equal(preds["step"].to_numpy(int),
                                             test_cases["visit_index"].to_numpy(int)):
        raise RuntimeError(f"CT-HMM predictions misaligned for patient {held_out}")
    raw[MODEL_CTHMM] = preds[PROB_COLS].to_numpy(dtype=float)
    diagnostics[MODEL_CTHMM] = {
        "train_loglik": float(fit.loglik),
        "scipy_success": bool(best.success),
        "proj_grad": float(best.proj_grad),
        "termination": best.message,
        "converged": bool(best.success or best.proj_grad < PROJ_GRAD_TOL),
    }

    # -- observed-label chains, no latent layer ------------------------------
    ctmc_fit = ctmc.fit_ctmc(train_pairs, data.states, state_names=STATE_NAMES,
                             seed=seed, n_restarts=n_restarts)
    raw[MODEL_CTMC] = ctmc.predict_pairs(ctmc_fit, _as_pairs(test_cases, data.states),
                                         data.states)
    diagnostics[MODEL_CTMC] = {"train_loglik": float(ctmc_fit.loglik),
                               "scipy_success": bool(ctmc_fit.converged),
                               "proj_grad": float("nan"),
                               "termination": "",
                               "converged": bool(ctmc_fit.converged)}

    dtmc = ctmc.fit_dtmc(train_pairs, data.states)
    raw[MODEL_DTMC] = dtmc[i, :]
    diagnostics[MODEL_DTMC] = _closed_form()

    # -- direct conditional two-class models ---------------------------------
    for name, feats in ((MODEL_DIRECT, FEATURES),
                        (MODEL_DIRECT_NOINT,
                         tuple(f for f in FEATURES if f not in INTERVAL_FEATURES))):
        d = fit_direct(train_cases, features=feats)
        p_pigd = d.predict_p_pigd(test_cases)
        # Written into the three-label layout with no Indeterminate mass: this
        # model estimates the conditional split directly, so renormalising it
        # is the identity and the two paths stay comparable downstream.
        raw[name] = np.column_stack([1.0 - p_pigd, np.zeros(n), p_pigd])
        diagnostics[name] = {"train_loglik": float(d.loglik),
                             "scipy_success": bool(d.converged),
                             "proj_grad": float(d.proj_grad),
                             "termination": "",
                             "converged": bool(d.converged
                                               or d.proj_grad < PROJ_GRAD_TOL)}

    # -- rule and null -------------------------------------------------------
    prior = _two_class_prior(train_cases)
    raw[MODEL_PERSIST] = _persistence_probs(test_cases, prior)
    diagnostics[MODEL_PERSIST] = _closed_form()
    raw[MODEL_MARGINAL] = np.tile([1.0 - prior, 0.0, prior], (n, 1))
    diagnostics[MODEL_MARGINAL] = _closed_form()

    frames = []
    for name in ALL_MODELS:
        probs = np.asarray(raw[name], dtype=float)
        p_td, p_pigd, defined = conditional_two_class(probs)
        frame = test_cases[[
            "PATNO", "visit_index", "n_history", "current_event", "current_years",
            "target_event", "target_years", "interval_years", "current_label",
            "actual_next_label", "eligible",
        ]].copy()
        frame["model"] = name
        frame["p_TD_raw"] = probs[:, TD]
        frame["p_Indeterminate_raw"] = probs[:, IND]
        frame["p_PIGD_raw"] = probs[:, PIGD]
        frame["P_TD"] = p_td
        frame["P_PIGD"] = p_pigd
        frame["two_class_defined"] = defined
        frame["predicted_state"] = predicted_state(p_td, p_pigd)
        frame["is_tie"] = defined & (p_td == p_pigd)
        for key, value in diagnostics[name].items():
            frame[f"fit_{key}"] = value
        frame["n_train_patients"] = len(train_sequences)
        frame["held_out_patient"] = held_out
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def _closed_form() -> dict:
    """Diagnostics for a model with nothing to optimise."""
    return {"train_loglik": float("nan"), "scipy_success": True,
            "proj_grad": 0.0, "termination": "closed form", "converged": True}


def _as_pairs(test_cases: pd.DataFrame, states: list) -> pd.DataFrame:
    """The columns :func:`ctmc.predict_pairs` reads, built from the case table."""
    lookup = {k: s for k, s in enumerate(states)}
    return pd.DataFrame({
        "PATNO": test_cases["PATNO"].to_numpy(),
        "from_state": [lookup[k] for k in test_cases["current_index"]],
        "to_state": [lookup[k] for k in test_cases["target_index"]],
        "interval_years": test_cases["interval_years"].to_numpy(dtype=float),
    })


def _two_class_prior(train_cases: pd.DataFrame) -> float:
    """P(PIGD | TD or PIGD) among eligible training cases, Laplace-smoothed."""
    eligible = train_cases[train_cases["eligible"]]
    n_pigd = float((eligible["target_index"] == PIGD).sum())
    return (n_pigd + 0.5) / (len(eligible) + 1.0)


def _persistence_probs(test_cases: pd.DataFrame, prior: float) -> np.ndarray:
    """Carry forward the last non-Indeterminate label in the history.

    A patient whose history is Indeterminate throughout has nothing to carry,
    so the fallback is the *training* two-class prior. That keeps the rule
    defined on every case without consulting the held-out patient's future, at
    the cost of making persistence very slightly fitted; the fallback fires on
    a handful of cases and they are counted in the output.
    """
    prev = test_cases["prev_noind_index"].to_numpy(int)
    p_pigd = np.where(prev == PIGD, 1.0, np.where(prev == TD, 0.0, prior))
    return np.column_stack([1.0 - p_pigd, np.zeros(len(prev)), p_pigd])


# ---------------------------------------------------------------------------
#  The LOPO loop
# ---------------------------------------------------------------------------
def run_lopo(
    data: loo.LOOData,
    cases: pd.DataFrame,
    seed: int,
    n_restarts: int,
    emission_floor: float | None,
    patients: list | None = None,
    progress_every: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out every patient in turn. Returns ``(predictions, failures)``.

    A fold that raises is recorded in the failures frame with its exception
    rather than being dropped: the count of attempted, completed and failed
    folds has to reconcile for the evaluation to mean anything.
    """
    patients = list(data.patients) if patients is None else list(patients)
    frames, failures = [], []

    for n, patno in enumerate(patients, start=1):
        try:
            frames.append(run_fold(data, cases, patno, seed, n_restarts, emission_floor))
        except Exception as exc:                        # noqa: BLE001 -- recorded, not hidden
            failures.append({"held_out_patient": patno,
                             "error": f"{type(exc).__name__}: {exc}"})
        if progress_every and n % progress_every == 0:
            print(f"  fold {n:,}/{len(patients):,}", flush=True)

    predictions = (pd.concat(frames, ignore_index=True) if frames
                   else pd.DataFrame(columns=["PATNO"]))
    return predictions, pd.DataFrame(failures, columns=["held_out_patient", "error"])


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------
def scored(frame: pd.DataFrame) -> pd.DataFrame:
    """The subset entering the primary two-class evaluation.

    Eligible (actual next label is TD or PIGD) *and* the model produced a
    defined conditional split. Both restrictions are counted in
    :func:`two_class_metrics` so nothing disappears quietly.
    """
    out = frame[frame["eligible"] & frame["two_class_defined"]].copy()
    out["correct"] = out["predicted_state"] == out["actual_next_label"]
    return out.reset_index(drop=True)


def two_class_metrics(frame: pd.DataFrame, seed: int = 42, n_boot: int = 2000) -> dict:
    """Headline numbers for one model on its eligible, defined cases."""
    name = frame["model"].iloc[0] if len(frame) else ""
    s = scored(frame)
    n = len(s)
    if not n:
        return {"model": name, "n_scored": 0}

    p_pigd = s["P_PIGD"].to_numpy(dtype=float)
    actual_pigd = (s["actual_next_label"] == "PIGD").to_numpy(dtype=float)
    correct = int(s["correct"].sum())

    # Two-class Brier: the sum over both classes is twice this, so the
    # single-outcome form is reported and labelled.
    brier = float(np.mean((p_pigd - actual_pigd) ** 2))
    p_true = np.clip(np.where(actual_pigd == 1.0, p_pigd, 1.0 - p_pigd),
                     _LOGLOSS_CLIP, 1.0)
    lo, hi = loo.bootstrap_patient_ci(s, n_boot=n_boot, seed=seed)
    w_lo, w_hi = loo.wilson_interval(correct, n)

    ind = frame[frame["actual_next_label"] == "Indeterminate"]
    return {
        "model": name,
        "n_patients": int(s["PATNO"].nunique()),
        "n_cases_total": int(len(frame)),
        "n_eligible": int(frame["eligible"].sum()),
        "n_scored": n,
        "n_undefined_among_eligible": int((frame["eligible"]
                                           & ~frame["two_class_defined"]).sum()),
        "n_indeterminate_next": int(len(ind)),
        "n_correct": correct,
        "accuracy": correct / n,
        "accuracy_ci_patient_bootstrap_low": lo,
        "accuracy_ci_patient_bootstrap_high": hi,
        "accuracy_ci_wilson_low": w_lo,
        "accuracy_ci_wilson_high": w_hi,
        "mean_per_patient_accuracy": float(s.groupby("PATNO")["correct"].mean().mean()),
        "brier_two_class": brier,
        "log_loss": float(-np.mean(np.log(p_true))),
        "log_loss_clipped": bool(np.any(np.where(actual_pigd == 1.0, p_pigd,
                                                 1.0 - p_pigd) < _LOGLOSS_CLIP)),
        "mean_P_PIGD": float(p_pigd.mean()),
        "observed_rate_PIGD": float(actual_pigd.mean()),
        "n_ties": int(s["is_tie"].sum()),
        "rule_based_probabilities": name in RULE_BASED,
        "mean_p_Indeterminate_raw": float(frame["p_Indeterminate_raw"].mean()),
        "mean_P_PIGD_on_indeterminate_next": (float(ind["P_PIGD"].mean())
                                              if len(ind) else float("nan")),
    }


def confusion(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows predicted, columns actual, over {TD, PIGD}."""
    s = scored(frame)
    return (pd.crosstab(s["predicted_state"], s["actual_next_label"])
            .reindex(index=TWO_CLASS, columns=TWO_CLASS, fill_value=0))


def per_class_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    cm = confusion(frame)
    rows = []
    for state in TWO_CLASS:
        tp = int(cm.loc[state, state])
        n_pred = int(cm.loc[state].sum())
        support = int(cm[state].sum())
        precision = tp / n_pred if n_pred else float("nan")
        recall = tp / support if support else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if n_pred and support and (precision + recall) > 0 else float("nan"))
        rows.append({"state": state, "support": support, "n_predicted": n_pred,
                     "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def calibration(frame: pd.DataFrame, edges=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0001),
                min_bin: int = 25) -> pd.DataFrame:
    """Observed PIGD rate against predicted P_PIGD, with sparse bins merged up."""
    s = scored(frame)
    if not len(s):
        return pd.DataFrame(columns=["bin", "n", "mean_P_PIGD", "observed_rate_PIGD"])

    p = s["P_PIGD"].to_numpy(dtype=float)
    actual = (s["actual_next_label"] == "PIGD").to_numpy(dtype=float)
    edges = list(edges)

    while True:
        idx = np.digitize(p, edges[1:-1], right=False)
        counts = np.bincount(idx, minlength=len(edges) - 1)
        small = [k for k, c in enumerate(counts) if 0 < c < min_bin]
        if not small or len(edges) <= 3:
            break
        k = small[0]
        edges.pop(k + 1 if k + 1 < len(edges) - 1 else k)

    idx = np.digitize(p, edges[1:-1], right=False)
    rows = []
    for k in range(len(edges) - 1):
        m = idx == k
        if not m.any():
            continue
        rows.append({"bin": f"{edges[k]:.2f}-{min(edges[k + 1], 1.0):.2f}",
                     "bin_low": edges[k], "bin_high": edges[k + 1],
                     "n": int(m.sum()),
                     "mean_P_PIGD": float(p[m].mean()),
                     "observed_rate_PIGD": float(actual[m].mean())})
    return pd.DataFrame(rows)


def indeterminate_report(frame: pd.DataFrame) -> dict:
    """What the model said at the visits whose actual next label is Indeterminate.

    These cases are never scored as correct or incorrect: the conditioning
    event ``next label is TD or PIGD`` did not occur, so neither class is the
    right answer and any rule for calling them would be a choice about the
    label, not a measurement of the model. Reported instead are their count
    and what the model would have said, which is the honest summary.
    """
    ind = frame[frame["actual_next_label"] == "Indeterminate"]
    if not len(ind):
        return {"n": 0}
    return {
        "n": int(len(ind)),
        "share_of_all_cases": float(len(ind) / len(frame)),
        "mean_P_PIGD": float(ind["P_PIGD"].mean()),
        "mean_p_Indeterminate_raw": float(ind["p_Indeterminate_raw"].mean()),
        "would_have_said_PIGD": int((ind["predicted_state"] == "PIGD").sum()),
        "would_have_said_TD": int((ind["predicted_state"] == "TD").sum()),
        "n_undefined": int((~ind["two_class_defined"]).sum()),
    }


def convergence_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Per model, how many folds converged and how they terminated."""
    rows = []
    for name, g in frame.groupby("model", sort=False):
        folds = g.drop_duplicates("held_out_patient")
        rows.append({
            "model": name,
            "n_folds": int(len(folds)),
            "n_scipy_success": int(folds["fit_scipy_success"].sum()),
            "n_converged": int(folds["fit_converged"].sum()),
            "n_not_converged": int((~folds["fit_converged"]).sum()),
            "max_proj_grad": float(np.nanmax(folds["fit_proj_grad"]))
            if folds["fit_proj_grad"].notna().any() else float("nan"),
            "mean_train_loglik": float(folds["fit_train_loglik"].mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Correctness checks
# ---------------------------------------------------------------------------
def correctness_checks(
    predictions: pd.DataFrame,
    cases: pd.DataFrame,
    data: loo.LOOData,
    failures: pd.DataFrame,
    patients: list | None = None,
    atol: float = 1e-12,
) -> dict[str, bool]:
    """The invariants this experiment's conclusions rest on.

    Everything here is a property of the produced table, checked against the
    documented rules and against the independently built case list --- not a
    re-derivation of the same numbers by the same code path.

    `patients` is the list of patients whose folds were attempted; it defaults
    to the whole cohort. Coverage is checked against that list rather than
    against the cohort so a partial run is held to what it actually claimed,
    while the separate check that the case list matches the panel's transition
    count still confirms that nothing was dropped upstream of the fold loop.
    """
    patients = list(data.patients) if patients is None else list(patients)
    completed = set(predictions["held_out_patient"])
    covered_cases = cases[cases["PATNO"].isin(completed)]
    defined = predictions[predictions["two_class_defined"]]
    p_td = defined["P_TD"].to_numpy(dtype=float)
    p_pigd = defined["P_PIGD"].to_numpy(dtype=float)

    expected_rule = predicted_state(p_td, p_pigd)
    n_models = predictions["model"].nunique()

    # Every prediction must use only the history the case table says it has.
    history_ok = bool((predictions["n_history"] == predictions["visit_index"] + 1).all())
    future_ok = bool((predictions["target_years"] > predictions["current_years"]).all())
    interval_ok = bool(np.allclose(
        predictions["interval_years"],
        predictions["target_years"] - predictions["current_years"], atol=1e-8))

    # The held-out patient is the one being predicted, and every fold trained
    # on all the others.
    fold_ok = bool((predictions["held_out_patient"] == predictions["PATNO"]).all())
    train_ok = set(predictions["n_train_patients"].unique()) == {len(data.sequences) - 1}

    ind_rows = predictions[predictions["actual_next_label"] == "Indeterminate"]

    # Every model must have been given the same cases, otherwise the
    # comparison is between different evaluations.
    per_model = {m: frozenset(map(tuple, g[["PATNO", "visit_index"]].to_numpy()))
                 for m, g in predictions.groupby("model", sort=False)}
    expected_cases = frozenset(map(tuple,
                                   covered_cases[["PATNO", "visit_index"]].to_numpy()))

    return {
        "probabilities finite on every defined case":
            bool(np.isfinite(p_td).all() and np.isfinite(p_pigd).all()),
        "probabilities within [0, 1]":
            bool((p_td >= 0).all() and (p_td <= 1).all()
                 and (p_pigd >= 0).all() and (p_pigd <= 1).all()),
        "P_TD + P_PIGD == 1 within tolerance":
            bool(np.allclose(p_td + p_pigd, 1.0, atol=atol)),
        "predicted class follows the documented rule":
            bool((defined["predicted_state"].to_numpy() == expected_rule).all()),
        "undefined cases carry no prediction":
            bool((predictions.loc[~predictions["two_class_defined"],
                                  "predicted_state"] == "").all()),
        "every model saw the identical case list":
            len(set(per_model.values())) == 1 and len(predictions) == n_models * len(
                covered_cases),
        "predicted cases are exactly the completed folds' cases":
            all(s == expected_cases for s in per_model.values()),
        "completed and failed folds account for every attempted patient":
            completed.isdisjoint(set(failures["held_out_patient"]))
            and completed | set(failures["held_out_patient"]) == set(patients),
        "case list matches the panel's transition count":
            len(cases) == data.n_transitions,
        "history length is exactly the visits up to the current one": history_ok,
        "target visit is strictly later than the current visit": future_ok,
        "elapsed time equals the gap between the two visit times": interval_ok,
        "held-out patient is the patient being predicted": fold_ok,
        "every fold trained on all other patients": train_ok,
        "Indeterminate next visits are excluded from scoring":
            int(scored(predictions)["actual_next_label"]
                .isin(["Indeterminate"]).sum()) == 0,
        "Indeterminate next visits are still reported": len(ind_rows) > 0,
        "Indeterminate survives in the input history":
            bool((cases["current_label"] == "Indeterminate").any()
                 and (cases["frac_hist_IND"] > 0).any()),
        "no fold failed silently":
            len(failures) == 0 or bool(set(failures["held_out_patient"])
                                       .isdisjoint(set(predictions["PATNO"]))),
        "every attempted patient held out exactly once":
            predictions["held_out_patient"].nunique() + len(failures) == len(patients),
    }


def paired_accuracy_difference(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame,
    n_boot: int = 2000, seed: int = 42, alpha: float = 0.05,
) -> dict:
    """Uncertainty on accuracy_a - accuracy_b over the cases both models scored.

    Two models evaluated on the same cases are not two independent samples, so
    comparing their separate confidence intervals is the wrong test: the
    intervals overlap heavily even when one model is reliably better. What
    matters is the paired difference.

    The interval reported is a patient-clustered bootstrap: patients are
    resampled with replacement and the accuracy difference recomputed, because
    a patient contributing fourteen predictions contributes fourteen
    correlated comparisons. The exact McNemar test is given alongside for
    reference, and it is the *less* trustworthy of the two here precisely
    because it treats those predictions as independent.
    """
    from scipy.stats import binomtest

    a, b = scored(frame_a), scored(frame_b)
    merged = a.merge(b, on=["PATNO", "visit_index"], suffixes=("_a", "_b"))
    if not len(merged):
        return {"n_paired": 0}

    diff = merged["correct_a"].to_numpy(float) - merged["correct_b"].to_numpy(float)
    by_patient = {p: g for p, g in pd.DataFrame(
        {"PATNO": merged["PATNO"], "d": diff}).groupby("PATNO")["d"]}
    patients = np.array(list(by_patient.keys()), dtype=object)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for k in range(n_boot):
        picked = rng.choice(len(patients), size=len(patients), replace=True)
        draws[k] = np.concatenate([by_patient[patients[i]].to_numpy()
                                   for i in picked]).mean()

    a_only = int((merged["correct_a"] & ~merged["correct_b"]).sum())
    b_only = int((merged["correct_b"] & ~merged["correct_a"]).sum())
    discordant = a_only + b_only
    p_mcnemar = (float(binomtest(a_only, discordant, 0.5).pvalue)
                 if discordant else float("nan"))

    return {
        "n_paired": int(len(merged)),
        "n_patients": int(merged["PATNO"].nunique()),
        "accuracy_delta": float(diff.mean()),
        "delta_ci_patient_bootstrap_low": float(np.quantile(draws, alpha / 2)),
        "delta_ci_patient_bootstrap_high": float(np.quantile(draws, 1 - alpha / 2)),
        "a_right_b_wrong": a_only,
        "b_right_a_wrong": b_only,
        "n_discordant": discordant,
        "mcnemar_exact_p_ignores_clustering": p_mcnemar,
    }


#: Where the inter-visit interval is split for the breakdown. PPMI's schedule
#: is annual early and biennial later: the median gap is 1.00 years and 86% of
#: cases fall at 1.5 years or less. A model that reads elapsed time can only
#: show it on the remainder, so the two groups are reported apart rather than
#: averaged into one number that the dense annual cases dominate.
INTERVAL_SPLIT_YEARS = 1.5


def accuracy_breakdowns(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Accuracy per model, split by interval, current label and history length.

    These are the three cuts that explain the headline differences rather than
    merely restating them: whether a model's advantage lives in the long gaps,
    whether it survives an ambiguous current label, and whether more of a
    patient's own history actually buys anything.
    """
    rows_interval, rows_label, rows_history = [], [], []

    for name, g in predictions.groupby("model", sort=False):
        s = scored(g)
        if not len(s):
            continue

        long = s["interval_years"] > INTERVAL_SPLIT_YEARS
        for label, mask in ((f"<= {INTERVAL_SPLIT_YEARS}y", ~long),
                            (f"> {INTERVAL_SPLIT_YEARS}y", long)):
            sub = s[mask]
            rows_interval.append({
                "model": name, "interval": label, "n": int(len(sub)),
                "accuracy": float(sub["correct"].mean()) if len(sub) else float("nan"),
                "brier_two_class": float(np.mean(
                    (sub["P_PIGD"] - (sub["actual_next_label"] == "PIGD")) ** 2))
                if len(sub) else float("nan"),
            })

        for label in STATE_NAMES:
            sub = s[s["current_label"] == label]
            rows_label.append({
                "model": name, "current_label": label, "n": int(len(sub)),
                "accuracy": float(sub["correct"].mean()) if len(sub) else float("nan"),
            })

        bucket = s["n_history"].clip(upper=5)
        for k, sub in s.groupby(bucket):
            rows_history.append({
                "model": name,
                "n_prior_visits": f"{int(k)}+" if k == 5 else str(int(k)),
                "n": int(len(sub)), "accuracy": float(sub["correct"].mean()),
            })

    return {
        "by_interval": pd.DataFrame(rows_interval),
        "by_current_label": pd.DataFrame(rows_label),
        "by_history_length": pd.DataFrame(rows_history),
    }
