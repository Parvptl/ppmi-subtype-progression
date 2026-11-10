"""Two-hidden-state CT-HMM variants: 2 hidden x 2 observed, and 2 hidden x 3 observed.

Why this module exists
----------------------
The project's progression model has three hidden states --- TD, Indeterminate,
PIGD --- one per recorded label. The question asked here is what happens when
the latent space is cut to two, TD and PIGD, in the two ways that are
defensible:

    Model 1   2 hidden x 2 observed.  Indeterminate visits are removed from the
              observation sequences entirely. The model sees only TD and PIGD
              readings, at their *actual* elapsed times, so a removed
              Indeterminate visit widens the gap between the two retained
              visits around it rather than being treated as absent.

    Model 2   2 hidden x 3 observed.  Every visit is kept. Indeterminate is an
              observed label that *either* hidden state may emit, with the
              probability learned from the data. It is not a hidden state and
              it is not assigned to a side.

Nothing about the latent process changes in either case. It is the same
continuous-time Markov chain, the same generator built by
:func:`ctmc.build_generator` with each diagonal entry equal to minus the sum of
its row, the same interval-aware transition matrices exp(Q dt), and the same
forward filtering recursion. :mod:`ppmi_subtype.hmm` was already written for a
rectangular emission matrix --- :func:`hmm._unpack`, :func:`hmm._forward_loglik`
and :func:`hmm.predictive_distributions` are reused verbatim --- so the only
things this module supplies are the reduced observation sequences, the fitting
entry points, and the evaluation that compares the variants fairly.

Parameter counts
----------------
Both follow :func:`hmm._n_params`, which is
``n_hidden (n_hidden - 1) + n_hidden (n_observed - 1) + (n_hidden - 1)``:

    Model 1   2 rates + 2 emission + 1 initial = 5
    Model 2   2 rates + 4 emission + 1 initial = 7

The emission rows are softmax of logits with one entry per row pinned to zero,
so each row sums to one by construction and cannot leave the simplex. Rates are
exp(theta), so they cannot go negative. The initial distribution is a softmax,
so it sums to one.

The emission floor
------------------
The square three-state model carries a floor, ``P(label a | state a) >= 0.5``:
an examination records the true state correctly more often than not. Without it
the three-state fit is degenerate (see the note in :mod:`ppmi_subtype.hmm`).

Model 1 is square, so the same floor applies to it unchanged, at the same
config value, through the same logit bound.

Model 2 is **not** square, and the floor is not defined for it: it constrains
P(label i | state i), and with two states and three labels there is no such
pairing. Imposing any floor would mean deciding in advance which label each
hidden state ought to emit --- which is precisely the thing the model is being
fitted to find out, and would amount to assigning Indeterminate to a side by
hand. Model 2 is therefore fitted unconstrained, and a 2 x 2 arm *without* the
floor is fitted alongside Model 1 so that the constraint and the observation
space are not confounded in the comparison.

Identifiability. With two hidden states and no floor the states are
exchangeable: permuting them and permuting the emission rows gives the same
likelihood. Predictions are invariant to that permutation, so it does not
affect any held-out number, but for reporting the states are put in a canonical
order by :func:`hmm_stability.canonical_order`, which sorts them by expected
emitted label, and are then named after the label each one favours.

Prediction, and what the models are scored on
---------------------------------------------
Every prediction is forward filtering only. For a target visit, alpha is
advanced through the labels at the visits *before* it and propagated through
exp(Q dt); no backward pass is run and no quantity depending on the target
label enters. Elapsed time to the target visit is an input, as it is throughout
this project's panel models: the model is told when the next examination
happens, never what it records.

The two models do not have the same prediction cases, and that is the whole
difficulty of comparing them:

    Model 2 can predict any labelled visit that has a labelled predecessor.
    Model 1 can predict only a TD-or-PIGD visit that has a TD-or-PIGD
    predecessor, because its sequences contain nothing else.

So two evaluations are produced. The *native* one scores each model on its own
cases in its own observation space. The *common* one restricts to the target
visits Model 1 can predict, scores every model there, and for the three-label
models renormalises onto {TD, PIGD} with
:func:`next_visit.conditional_two_class`:

    P(TD | TD or PIGD) = p_TD / (p_TD + p_PIGD)

Model 1 is at a real disadvantage on the common set and the disadvantage is
part of what is being measured: where an Indeterminate visit intervenes, Model
1's filtered belief is older and its dt longer, because the evidence it dropped
is evidence the three-label models still have.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import hmm, hmm_stability, loo, next_visit as nv
from .ctmc import transition_probabilities

#: Observed labels of the three-label models, in the order every probability
#: array over three labels uses.
THREE_LABELS = loo.STATE_NAMES                      # ["TD", "Indeterminate", "PIGD"]
#: Observed labels of Model 1, and the common evaluation's two classes.
TWO_LABELS = ["TD", "PIGD"]

TD, IND, PIGD = 0, 1, 2
#: Index of TD and PIGD inside a two-label array.
BIN_TD, BIN_PIGD = 0, 1

MODEL_2X2 = "CT-HMM (2 hidden x 2 observed, Indeterminate excluded)"
MODEL_2X2_FREE = "CT-HMM (2 hidden x 2 observed, no emission floor)"
MODEL_2X3 = "CT-HMM (2 hidden x 3 observed)"
MODEL_3X3 = "CT-HMM (3 hidden x 3 observed, existing model)"
MODEL_PERSIST = "Persistence (last observed label)"
MODEL_MARGINAL = "Marginal (training label prior)"

#: Models fitted on three observed labels. These are the ones that produce a
#: three-class native prediction and need renormalising for the common set.
THREE_LABEL_MODELS = (MODEL_2X3, MODEL_3X3)
#: Models fitted on two observed labels, Indeterminate excluded.
TWO_LABEL_MODELS = (MODEL_2X2, MODEL_2X2_FREE)
#: Models whose probabilities are a rule or a prior, not a fitted estimate.
RULE_BASED = (MODEL_PERSIST, MODEL_MARGINAL)

ALL_MODELS = (MODEL_2X2, MODEL_2X2_FREE, MODEL_2X3, MODEL_3X3,
              MODEL_PERSIST, MODEL_MARGINAL)

#: Tie-break order for a three-class argmax, worst motor prognosis first. A tie
#: carries no information, so the rule has to come from outside the model;
#: this is the three-class reading of :data:`next_visit.TIE_RULE`, which
#: resolves a two-class tie to PIGD for the same reason. Ties are counted in
#: every output so their influence can be checked.
TIE_ORDER = (PIGD, IND, TD)

_EPS = 1e-300
_LOGLOSS_CLIP = 1e-12


# ---------------------------------------------------------------------------
#  Data preparation
# ---------------------------------------------------------------------------
@dataclass
class TwoStateData:
    """Both observation views of the same cohort, and the map between them.

    ``full_sequences`` is :class:`loo.LOOData`'s own list, untouched: one entry
    per patient with every labelled visit, labels indexed into
    :data:`THREE_LABELS`. ``binary_sequences`` is the Indeterminate-free view,
    labels indexed into :data:`TWO_LABELS`, gaps recomputed from the retained
    visits' actual times. ``binary_origin`` maps a patient's binary position
    back to its position in the full sequence, which is what lets a prediction
    from either view be attributed to the same target visit.
    """

    full_sequences: list[dict]
    binary_sequences: list[dict]
    binary_origin: dict
    meta: pd.DataFrame
    states_three: list
    states_two: list

    @property
    def patients(self) -> list:
        return [s["PATNO"] for s in self.full_sequences]

    @property
    def binary_patients(self) -> list:
        return [s["PATNO"] for s in self.binary_sequences]

    def full_by_patient(self) -> dict:
        return {s["PATNO"]: s for s in self.full_sequences}

    def binary_by_patient(self) -> dict:
        return {s["PATNO"]: s for s in self.binary_sequences}

    def counts(self) -> dict:
        """Patients, visits and transitions in each view, for the record."""
        full_visits = int(sum(len(s["labels"]) for s in self.full_sequences))
        bin_visits = int(sum(len(s["labels"]) for s in self.binary_sequences))
        kept = {s["PATNO"] for s in self.binary_sequences}
        dropped_ind = int(sum(int((s["labels"] == IND).sum())
                              for s in self.full_sequences))
        return {
            "model2_patients": len(self.full_sequences),
            "model2_visits": full_visits,
            "model2_transitions": int(sum(len(s["gaps"]) for s in self.full_sequences)),
            "model1_patients": len(self.binary_sequences),
            "model1_visits": bin_visits,
            "model1_transitions": int(sum(len(s["gaps"])
                                          for s in self.binary_sequences)),
            "patients_excluded_from_model1": len(self.full_sequences) - len(kept),
            "indeterminate_visits_dropped": dropped_ind,
            "visits_excluded_from_model1": full_visits - bin_visits,
        }


def prepare(data: loo.LOOData, min_length: int = 2) -> TwoStateData:
    """Build the Indeterminate-free view alongside the full one.

    Visit times are read from ``data.meta`` rather than accumulated from the
    full sequence's gaps, so a retained visit's time is the time the panel
    records and the recomputed gap is the actual elapsed years between the two
    retained visits --- not the sum of a chain of annual steps.

    A patient with fewer than `min_length` retained visits contributes no
    binary sequence: there is no transition to observe. They stay in the full
    view, and the count of such patients is part of :meth:`TwoStateData.counts`.
    """
    times = {(int(r.PATNO), int(r.visit_index)): float(r.years_from_baseline)
             for r in data.meta.itertuples()}

    binary_sequences, binary_origin = [], {}
    for seq in data.sequences:
        patno = seq["PATNO"]
        labels = np.asarray(seq["labels"], dtype=int)
        keep = np.flatnonzero(labels != IND)
        if len(keep) < min_length:
            continue

        t = np.array([times[(int(patno), int(k))] for k in keep], dtype=float)
        if not np.all(np.diff(t) > 0):              # pragma: no cover -- panel is sorted
            raise ValueError(f"patient {patno}: retained visit times are not increasing")

        # TD -> 0, PIGD -> 1. Indeterminate cannot appear: it was filtered out.
        binary = (labels[keep] == PIGD).astype(int)
        binary_sequences.append({"PATNO": patno, "labels": binary, "gaps": np.diff(t)})
        binary_origin[patno] = keep

    return TwoStateData(
        full_sequences=list(data.sequences),
        binary_sequences=binary_sequences,
        binary_origin=binary_origin,
        meta=data.meta,
        states_three=list(data.states),
        states_two=[data.states[TD], data.states[PIGD]],
    )


def build_targets(data: TwoStateData) -> pd.DataFrame:
    """One row per prediction target, with which models can predict it.

    A *target* is a labelled visit that has at least one earlier labelled visit
    for the same patient. ``visit_index`` is its position in the patient's
    labelled sequence, so ``visit_index`` and the patient together identify it
    across all models regardless of which view produced the prediction.

    ``model1_eligible`` marks the targets Model 1 can predict: the target is
    itself a TD or PIGD visit and at least one TD or PIGD visit precedes it.
    Those are exactly the steps of the binary sequences, and they are the
    common evaluation set. ``model1_prev_visit_index`` and
    ``model1_interval_years`` record the visit Model 1 actually filters from and
    the gap it actually sees, which is longer than the three-label models' gap
    wherever an Indeterminate visit was dropped in between.
    """
    meta = data.meta.set_index(["PATNO", "visit_index"])
    binary = data.binary_by_patient()

    rows = []
    for seq in data.full_sequences:
        patno = seq["PATNO"]
        labels = np.asarray(seq["labels"], dtype=int)
        origin = data.binary_origin.get(patno)
        # Position in the binary sequence of each retained full-sequence index.
        bin_pos = ({int(o): k for k, o in enumerate(origin)}
                   if origin is not None else {})

        for k in range(1, len(labels)):
            prev = meta.loc[(patno, k - 1)]
            tgt = meta.loc[(patno, k)]

            # Model 1's predecessor: the last retained visit strictly before k.
            m1_prev = None
            if origin is not None and k in bin_pos and bin_pos[k] >= 1:
                m1_prev = int(origin[bin_pos[k] - 1])

            m1_ok = m1_prev is not None
            m1_prev_years = (float(meta.loc[(patno, m1_prev)]["years_from_baseline"])
                             if m1_ok else float("nan"))

            rows.append({
                "PATNO": patno,
                "visit_index": k,
                "n_history": k,
                "target_event": tgt["EVENT_ID"],
                "target_years": float(tgt["years_from_baseline"]),
                "prev_event": prev["EVENT_ID"],
                "prev_years": float(prev["years_from_baseline"]),
                "interval_years": float(tgt["years_from_baseline"]
                                        - prev["years_from_baseline"]),
                "target_index": int(labels[k]),
                "prev_index": int(labels[k - 1]),
                "actual_label": THREE_LABELS[int(labels[k])],
                "prev_label": THREE_LABELS[int(labels[k - 1])],
                "model1_eligible": m1_ok,
                "model1_prev_visit_index": (m1_prev if m1_ok else -1),
                "model1_prev_label": (THREE_LABELS[int(labels[m1_prev])]
                                      if m1_ok else ""),
                "model1_interval_years": (float(tgt["years_from_baseline"])
                                          - m1_prev_years if m1_ok else float("nan")),
                "model1_n_history": (bin_pos[k] if m1_ok else 0),
                "n_binary_visits": (len(binary[patno]["labels"])
                                    if patno in binary else 0),
            })

    targets = pd.DataFrame(rows)
    # Every Model-1-eligible target must be a TD or PIGD visit: its label is in
    # the binary sequence by construction. Checked rather than assumed.
    bad = targets[targets["model1_eligible"] & (targets["target_index"] == IND)]
    if len(bad):                                    # pragma: no cover
        raise ValueError(f"{len(bad)} Model 1 targets are Indeterminate visits")
    return targets


# ---------------------------------------------------------------------------
#  Hidden-state ordering
# ---------------------------------------------------------------------------
def canonical_triple(Q: np.ndarray, E: np.ndarray, pi: np.ndarray,
                     emission_floor: float | None):
    """Put the hidden states in a reproducible order.

    With an emission floor each hidden state is pinned to a label --- the floor
    forces P(label a | state a) above a half, so state a *is* the TD state ---
    and nothing is reordered.

    Without a floor the states are exchangeable: permuting them together with
    the emission rows gives the same likelihood, so which optimum the optimiser
    lands on is arbitrary. That applies to the square 2 x 2 arm as much as to
    the 2 x 3 one, which is why
    :func:`hmm_stability.canonical_order` cannot be used on its own here: it
    returns a square model untouched, because the square model is normally
    fitted *with* the floor. States are therefore sorted by their expected
    emitted label, so the first is the most TD-like.

    Predictions are invariant to the permutation, so this changes no held-out
    number --- only which row of the reported matrices is which.
    """
    if emission_floor is not None:
        return hmm_stability.canonical_order(Q, E, pi)
    order = np.argsort(E @ np.arange(E.shape[1]), kind="mergesort")
    return Q[np.ix_(order, order)], E[order], pi[order]


def solution_spread(records, n_hidden: int, n_observed: int,
                    emission_floor: float | None, tol: float = 0.01) -> dict:
    """:func:`hmm_stability.solution_spread`, ordered by :func:`canonical_triple`.

    Identical in what it measures --- the share of starts reaching the best
    log-likelihood, how many distinct levels there are, and the largest
    difference in any Q or E entry among the starts at the best level. The one
    change is the ordering applied before parameters are compared: without it,
    two starts that found the same optimum under a different labelling of the
    exchangeable hidden states look like a flat ridge when they are the same
    point.
    """
    ll = np.array([r.loglik for r in records], dtype=float)
    best_i = int(np.argmax(ll))
    near = np.abs(ll - ll[best_i]) <= tol

    def triple(rec):
        return canonical_triple(*hmm._unpack(rec.theta, n_hidden, n_observed),
                                emission_floor)

    Qb, Eb, _ = triple(records[best_i])
    dq = de = 0.0
    for rec, ok in zip(records, near):
        if not ok:
            continue
        Q, E, _ = triple(rec)
        dq = max(dq, float(np.max(np.abs(Q - Qb))))
        de = max(de, float(np.max(np.abs(E - Eb))))

    return {
        "best_loglik": float(ll[best_i]),
        "best_start": best_i,
        "share_within_tol": float(near.mean()),
        "n_loglik_levels": int(len(np.unique(np.round(ll / tol)))),
        "worst_loglik": float(ll.min()),
        "max_dQ_at_best": dq,
        "max_dE_at_best": de,
        "share_success_flag": float(np.mean([r.success for r in records])),
        "share_proj_grad_converged": float(
            np.mean([r.proj_grad < PROJ_GRAD_TOL for r in records])),
    }


# ---------------------------------------------------------------------------
#  Fitting
# ---------------------------------------------------------------------------
#: A fit is reported as converged when scipy says so, or when the projected
#: gradient at the chosen optimum is below this. Same rule and same tolerance
#: as :data:`next_visit.PROJ_GRAD_TOL`; see :func:`next_visit.audited_starts`
#: for why scipy's flag alone is not the criterion.
PROJ_GRAD_TOL = nv.PROJ_GRAD_TOL


@dataclass
class VariantFit:
    """A fitted variant, its convergence record and its information criteria."""

    model: str
    fit: hmm.HMMFit
    records: list
    n_hidden: int
    n_observed: int
    observed_names: list
    emission_floor: float | None
    n_starts: int

    @property
    def loglik(self) -> float:
        return float(self.fit.loglik)

    @property
    def n_params(self) -> int:
        return int(self.fit.n_params)

    @property
    def n_observations(self) -> int:
        """Likelihood terms: one per visit in the sequences that were fitted.

        This is the project's convention for the HMM's BIC sample size, taken
        from :attr:`hmm.HMMFit.n_observations` and used by
        ``scripts/65_model_selection_criteria.py``. The HMM's likelihood
        factorises over whole sequences, and every visit --- including each
        patient's first, which the initial distribution accounts for ---
        contributes one term, so the transition count would undercount it by
        exactly the number of patients.
        """
        return int(self.fit.n_observations)

    @property
    def best(self):
        return max(self.records, key=lambda r: r.loglik)

    @property
    def converged(self) -> bool:
        b = self.best
        return bool(b.success or b.proj_grad < PROJ_GRAD_TOL)

    def criteria(self) -> dict:
        """AIC and BIC. ``n`` is :attr:`n_observations`; see its docstring."""
        k, ll, n = self.n_params, self.loglik, self.n_observations
        return {
            "loglik": ll,
            "n_params": k,
            "n_observations": n,
            "aic": 2 * k - 2 * ll,
            "bic": k * float(np.log(max(n, 2))) - 2 * ll,
        }

    def canonical(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Q, E and pi with the hidden states in a reproducible order.

        Only does anything when the hidden states are exchangeable, i.e. when
        the model is not square. Predictions are invariant to the permutation.
        """
        return canonical_triple(self.fit.Q, self.fit.E, self.fit.pi,
                                self.emission_floor)

    def hidden_names(self) -> list:
        """Name each canonical hidden state after the label it most favours."""
        _, E, _ = self.canonical()
        if self.emission_floor is not None and self.n_hidden == self.n_observed:
            # The floor pins state a to label a, so the label IS the name.
            return list(self.observed_names)
        return [f"H{k + 1}:{self.observed_names[int(np.argmax(E[k]))]}"
                for k in range(self.n_hidden)]

    def generator_frame(self) -> pd.DataFrame:
        Q, _, _ = self.canonical()
        names = self.hidden_names()
        return pd.DataFrame(Q, index=names, columns=names)

    def sojourn_frame(self) -> pd.Series:
        """Expected years in each hidden state before leaving it, canonically ordered.

        ``hmm.HMMFit.mean_sojourn`` reports against the fit's own state order,
        which for an unconstrained model is whichever labelling the optimiser
        happened to land on; this reports against :meth:`hidden_names`.
        """
        Q, _, _ = self.canonical()
        d = np.diag(Q)
        with np.errstate(divide="ignore"):
            years = np.where(d < 0, -1.0 / d, np.inf)
        return pd.Series(years, index=self.hidden_names(), name="years")

    def emission_frame(self) -> pd.DataFrame:
        _, E, _ = self.canonical()
        return pd.DataFrame(E, index=self.hidden_names(),
                            columns=self.observed_names)

    def initial_frame(self) -> pd.Series:
        _, _, pi = self.canonical()
        return pd.Series(pi, index=self.hidden_names(), name="initial_probability")

    def spread(self) -> dict:
        """How many starts reached the best optimum, and how far the rest sat.

        Delegates to :func:`hmm_stability.solution_spread`, which also reports
        the largest difference in any canonical Q or E entry among the starts
        that reached the best log-likelihood --- a flat ridge (same likelihood,
        different parameters) shows up there and nowhere else.
        """
        s = solution_spread(self.records, self.n_hidden, self.n_observed,
                            self.emission_floor)
        lls = np.array([r.loglik for r in self.records], dtype=float)
        return {
            "n_starts": len(self.records),
            "n_at_best_loglik": int(np.sum(lls > s["best_loglik"] - 0.01)),
            "share_at_best": float(s["share_within_tol"]),
            "n_distinct_loglik_levels": int(s["n_loglik_levels"]),
            "max_dQ_at_best_level": float(s["max_dQ_at_best"]),
            "max_dE_at_best_level": float(s["max_dE_at_best"]),
            "best_loglik": float(s["best_loglik"]),
            "worst_start_loglik": float(s["worst_loglik"]),
            "best_start": int(s["best_start"]),
            "share_scipy_success": float(s["share_success_flag"]),
            "share_proj_grad_converged": float(s["share_proj_grad_converged"]),
            "best_proj_grad": float(self.best.proj_grad),
            "best_termination": self.best.message,
        }


def fit_variant(
    model: str,
    sequences: list[dict],
    states: list,
    observed_names: list,
    n_hidden: int,
    emission_floor: float | None,
    seed: int,
    starts: np.ndarray,
) -> VariantFit:
    """Fit one variant through the project's audited multi-start optimiser.

    :func:`hmm_stability.fit_multistart` runs the identical objective
    (:func:`hmm._neg_loglik`), the identical bounds
    (:func:`hmm_stability.bounds_for`) and the identical optimiser settings as
    :func:`hmm.fit_hmm`, and additionally records every start's projected
    gradient and termination message. It is used instead of ``fit_hmm``
    because ``fit_hmm`` rejects ``n_hidden < n_observed`` outright, which is
    Model 2's whole configuration; the likelihood, the parameterisation and
    the constraints are the same code either way, so this is a change of entry
    point and not of method.
    """
    fit, records = hmm_stability.fit_multistart(
        sequences, states, n_hidden=n_hidden, emission_floor=emission_floor,
        starts=starts, state_names=observed_names)
    return VariantFit(
        model=model, fit=fit, records=records, n_hidden=n_hidden,
        n_observed=len(states), observed_names=list(observed_names),
        emission_floor=emission_floor, n_starts=len(starts),
    )


#: The configuration of every arm: hidden states, observation space, floor.
SPECS = {
    MODEL_2X2: {"n_hidden": 2, "view": "binary", "floor": "config"},
    MODEL_2X2_FREE: {"n_hidden": 2, "view": "binary", "floor": None},
    MODEL_2X3: {"n_hidden": 2, "view": "full", "floor": None},
    MODEL_3X3: {"n_hidden": 3, "view": "full", "floor": "config"},
}


def spec_for(model: str, data: TwoStateData, emission_floor: float | None) -> dict:
    """Resolve one arm's spec against the data and the configured floor."""
    s = SPECS[model]
    full = s["view"] == "full"
    return {
        "model": model,
        "n_hidden": s["n_hidden"],
        "sequences_attr": "full_sequences" if full else "binary_sequences",
        "states": data.states_three if full else data.states_two,
        "observed_names": THREE_LABELS if full else TWO_LABELS,
        "emission_floor": emission_floor if s["floor"] == "config" else None,
    }


def fit_all(
    data: TwoStateData,
    emission_floor: float,
    seed: int,
    n_starts: int,
    dispersed: bool = True,
    models: tuple = tuple(SPECS),
) -> dict:
    """Fit every arm on the sequences given, for parameter reporting and AIC/BIC.

    With `dispersed` the starts come from
    :func:`hmm_stability.dispersed_starts`, whose start 0 is the project
    default and whose remainder are spread across the parameter space. That is
    what makes "multiple initialisations" mean more than one perturbation of a
    single point, and it is how the risk of a local optimum is reported rather
    than assumed away. The best converged fit is selected on the *training*
    log-likelihood, inside :func:`hmm_stability.fit_multistart`.
    """
    out = {}
    for model in models:
        s = spec_for(model, data, emission_floor)
        sequences = getattr(data, s["sequences_attr"])
        args = (s["n_hidden"], len(s["states"]), s["emission_floor"])
        starts = (hmm_stability.dispersed_starts(*args, n_starts, seed) if dispersed
                  else nv.audited_starts(*args, seed, n_starts))
        out[model] = fit_variant(
            model, sequences, s["states"], s["observed_names"],
            s["n_hidden"], s["emission_floor"], seed, starts)
    return out


# ---------------------------------------------------------------------------
#  Prediction: forward filtering, one row per target visit
# ---------------------------------------------------------------------------
def filtered_predictions(vfit: VariantFit, sequence: dict,
                         origin: np.ndarray | None = None) -> pd.DataFrame:
    """Forward-filtered next-observation distributions for one patient.

    :func:`hmm.predictive_distributions` is the project's filtering recursion
    and is reused unchanged: alpha is advanced with the label actually seen,
    propagated through exp(Q dt), and multiplied into the emission matrix to
    give a distribution over observed labels. No backward pass is run.

    ``origin`` maps a binary-view step to the target's position in the
    patient's full labelled sequence. Without it the step index *is* that
    position.
    """
    preds = hmm.predictive_distributions(vfit.fit, [sequence])
    preds = preds.sort_values("step").reset_index(drop=True)

    step = preds["step"].to_numpy(int)
    visit_index = (step + 1) if origin is None else origin[step + 1]

    out = pd.DataFrame({
        "PATNO": preds["PATNO"].to_numpy(),
        "visit_index": np.asarray(visit_index, dtype=int),
        "model_interval_years": preds["interval_years"].to_numpy(float),
    })
    for name in vfit.observed_names:
        out[f"p_{name}"] = preds[f"p_{name}"].to_numpy(float)
    if vfit.n_observed == 2:
        out["p_Indeterminate"] = np.nan
    return out


def three_label_matrix(frame: pd.DataFrame) -> np.ndarray:
    """The (n, 3) probability array, in :data:`THREE_LABELS` order."""
    return frame[[f"p_{n}" for n in THREE_LABELS]].to_numpy(dtype=float)


def argmax_three(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Three-class prediction with :data:`TIE_ORDER` breaking ties.

    Returns ``(labels, is_tie)``. A row whose maximum is attained by more than
    one label is resolved by :data:`TIE_ORDER` and flagged.
    """
    probs = np.asarray(probs, dtype=float)
    best = np.nanmax(probs, axis=1, keepdims=True)
    at_best = np.isclose(probs, best, rtol=0.0, atol=0.0) & np.isfinite(probs)
    is_tie = at_best.sum(axis=1) > 1

    chosen = np.argmax(probs, axis=1)
    for row in np.flatnonzero(is_tie):
        for k in TIE_ORDER:
            if at_best[row, k]:
                chosen[row] = k
                break
    return np.array([THREE_LABELS[k] for k in chosen], dtype=object), is_tie


def predict_one_fold(
    fits: dict,
    data: TwoStateData,
    targets: pd.DataFrame,
    held_out,
    train_prior_three: np.ndarray,
    train_prior_two: float,
) -> pd.DataFrame:
    """Every arm's prediction for one held-out patient's target visits.

    The arms are predicted inside one function on purpose: they then cannot
    differ in anything but their observation space and hidden-state count ---
    same held-out patient, same training patients, same target visits, same
    seed, same start scheme, same convergence accounting.
    """
    tg = targets[targets["PATNO"] == held_out].sort_values("visit_index")
    tg = tg.reset_index(drop=True)
    if not len(tg):
        return pd.DataFrame()

    full_seq = next(s for s in data.full_sequences if s["PATNO"] == held_out)
    bin_seq = next((s for s in data.binary_sequences if s["PATNO"] == held_out), None)
    origin = data.binary_origin.get(held_out)

    frames = []
    for model, vfit in fits.items():
        if model in TWO_LABEL_MODELS:
            if bin_seq is None:
                continue
            preds = filtered_predictions(vfit, bin_seq, origin=origin)
        else:
            preds = filtered_predictions(vfit, full_seq)

        merged = tg.merge(preds, on=["PATNO", "visit_index"], how="inner")
        if model in TWO_LABEL_MODELS and len(merged) != int(tg["model1_eligible"].sum()):
            raise RuntimeError(                     # pragma: no cover
                f"{model}: patient {held_out} produced {len(merged)} predictions "
                f"for {int(tg['model1_eligible'].sum())} eligible targets")
        merged["model"] = model
        frames.append(merged)

    # -- the rule and the null, on exactly the same target rows --------------
    rule = tg.copy()
    rule["model"] = MODEL_PERSIST
    # Three-class: carry the previous observed label forward as a point mass.
    onehot = np.zeros((len(tg), 3))
    onehot[np.arange(len(tg)), tg["prev_index"].to_numpy(int)] = 1.0
    # Two-class: carry the last TD/PIGD label forward. Where the history holds
    # none, fall back to the TRAINING prior, which keeps the rule defined
    # without consulting the held-out patient's future.
    prev_bin = tg["model1_prev_label"].to_numpy()
    p_pigd_bin = np.where(prev_bin == "PIGD", 1.0,
                          np.where(prev_bin == "TD", 0.0, train_prior_two))
    for k, name in enumerate(THREE_LABELS):
        rule[f"p_{name}"] = onehot[:, k]
    rule["binary_p_PIGD_override"] = p_pigd_bin
    rule["model_interval_years"] = tg["interval_years"].to_numpy(float)
    frames.append(rule)

    null = tg.copy()
    null["model"] = MODEL_MARGINAL
    for k, name in enumerate(THREE_LABELS):
        null[f"p_{name}"] = train_prior_three[k]
    null["model_interval_years"] = tg["interval_years"].to_numpy(float)
    frames.append(null)

    out = pd.concat(frames, ignore_index=True)
    out["held_out_patient"] = held_out
    return out


def finalise_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the three-class and two-class predictions to a raw prediction frame.

    The two-class columns come from :func:`next_visit.conditional_two_class`,
    the same renormalisation the published next-visit study uses:

        P(TD | TD or PIGD) = p_TD / (p_TD + p_PIGD)

    For Model 1 the renormalisation is the identity --- its distribution has no
    Indeterminate mass --- so the two paths stay comparable. For the
    persistence rule the two-class probability is the carried-forward TD/PIGD
    label rather than the renormalised three-class point mass, because a
    history of nothing but Indeterminate has no label to carry and the rule's
    documented fallback is the training prior.
    """
    out = frame.copy()
    probs = three_label_matrix(out)

    # Three-class prediction, only where all three labels carry a probability.
    three_ok = np.isfinite(probs).all(axis=1)
    pred3 = np.full(len(out), "", dtype=object)
    tie3 = np.zeros(len(out), dtype=bool)
    if three_ok.any():
        labels, ties = argmax_three(probs[three_ok])
        pred3[three_ok] = labels
        tie3[three_ok] = ties
    out["predicted_three_class"] = pred3
    out["three_class_defined"] = three_ok
    out["three_class_is_tie"] = tie3

    # Two-class prediction on the renormalised {TD, PIGD} distribution.
    filled = np.where(np.isfinite(probs), probs, 0.0)
    p_td, p_pigd, defined = nv.conditional_two_class(filled)

    override = out.get("binary_p_PIGD_override")
    if override is not None:
        has = override.notna().to_numpy()
        p_pigd = np.where(has, override.to_numpy(dtype=float), p_pigd)
        p_td = np.where(has, 1.0 - p_pigd, p_td)
        defined = defined | has

    out["P_TD"] = p_td
    out["P_PIGD"] = p_pigd
    out["two_class_defined"] = defined
    out["predicted_two_class"] = nv.predicted_state(p_td, p_pigd)
    out["two_class_is_tie"] = defined & (p_td == p_pigd)
    return out


def training_priors(data: TwoStateData, held_out) -> tuple[np.ndarray, float]:
    """Laplace-smoothed label priors over the TRAINING patients only.

    The three-class prior is over all observed labels at the training patients'
    target visits; the two-class prior is P(PIGD | TD or PIGD) among them. Both
    are the null model's probabilities and the persistence rule's fallback, so
    both have to be free of the held-out patient.
    """
    counts = np.zeros(3)
    for seq in data.full_sequences:
        if seq["PATNO"] == held_out:
            continue
        labels = np.asarray(seq["labels"], dtype=int)[1:]       # target visits only
        counts += np.bincount(labels, minlength=3)
    three = (counts + 0.5) / (counts.sum() + 1.5)
    two = (counts[PIGD] + 0.5) / (counts[TD] + counts[PIGD] + 1.0)
    return three, float(two)


# ---------------------------------------------------------------------------
#  One LOPO fold
# ---------------------------------------------------------------------------
def run_fold(
    data: TwoStateData,
    targets: pd.DataFrame,
    held_out,
    seed: int,
    n_restarts: int,
    emission_floor: float,
    models: tuple = tuple(SPECS),
) -> pd.DataFrame:
    """Hold out one patient, fit every arm on the rest, predict its targets.

    The held-out patient contributes nothing to any fit, in either view, and
    nothing to the priors the rule and the null use.
    :func:`loo.assert_no_leakage` is called on both views.
    """
    fits, diagnostics = {}, {}
    for model in models:
        s = spec_for(model, data, emission_floor)
        train = [q for q in getattr(data, s["sequences_attr"])
                 if q["PATNO"] != held_out]
        loo.assert_no_leakage(held_out, train, pd.DataFrame({"PATNO": []}))

        args = (s["n_hidden"], len(s["states"]), s["emission_floor"])
        starts = nv.audited_starts(*args, seed, n_restarts)
        vfit = fit_variant(model, train, s["states"], s["observed_names"],
                           s["n_hidden"], s["emission_floor"], seed, starts)
        fits[model] = vfit
        best = vfit.best
        diagnostics[model] = {
            "fit_train_loglik": float(vfit.loglik),
            "fit_scipy_success": bool(best.success),
            "fit_proj_grad": float(best.proj_grad),
            "fit_termination": best.message,
            "fit_converged": bool(best.success or best.proj_grad < PROJ_GRAD_TOL),
            "fit_n_train_sequences": int(len(train)),
            "fit_n_train_visits": int(vfit.n_observations),
            "fit_n_params": int(vfit.n_params),
        }

    three, two = training_priors(data, held_out)
    raw = predict_one_fold(fits, data, targets, held_out, three, two)
    if not len(raw):
        return raw

    out = finalise_predictions(raw)
    for model, diag in diagnostics.items():
        mask = out["model"] == model
        for key, value in diag.items():
            if key not in out.columns:
                out[key] = np.nan if isinstance(value, float) else None
            out.loc[mask, key] = value
    for model in RULE_BASED:
        mask = out["model"] == model
        out.loc[mask, "fit_converged"] = True
        out.loc[mask, "fit_scipy_success"] = True
        out.loc[mask, "fit_termination"] = "closed form"
        out.loc[mask, "fit_n_params"] = 0
    return out


# ---------------------------------------------------------------------------
#  Evaluation: the two tasks, and the sets each model is scored on
# ---------------------------------------------------------------------------
#  Two tasks are scored, never averaged together:
#
#  "three_class"  the native task of the three-label models. Every target
#                 visit, scored against its actual label among TD /
#                 Indeterminate / PIGD. Model 1 cannot enter it at all: it has
#                 no Indeterminate category to predict, so it is absent from
#                 the table rather than given a zero.
#
#  "two_class"    the common task. Restricted to the targets Model 1 can
#                 predict (`model1_eligible`), where every arm is scored on the
#                 identical rows after renormalising onto {TD, PIGD}. This is
#                 the comparison that answers whether the third emission buys
#                 anything on the TD/PIGD question itself.
TASK_THREE = "three_class"
TASK_TWO = "two_class"


def scored(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    """The rows entering one task's evaluation, with a ``correct`` column.

    Both restrictions --- the task's eligibility and whether the model
    produced a defined prediction --- are counted in the metrics, so nothing
    disappears quietly.
    """
    if task == TASK_THREE:
        out = frame[frame["three_class_defined"]].copy()
        out["predicted"] = out["predicted_three_class"]
    elif task == TASK_TWO:
        out = frame[frame["model1_eligible"] & frame["two_class_defined"]].copy()
        out["predicted"] = out["predicted_two_class"]
    else:                                           # pragma: no cover
        raise ValueError(f"unknown task {task!r}")
    out["correct"] = out["predicted"] == out["actual_label"]
    return out.reset_index(drop=True)


def labels_for(task: str) -> list:
    return list(THREE_LABELS) if task == TASK_THREE else list(TWO_LABELS)


# ---------------------------------------------------------------------------
#  Confusion matrices and the metrics derived from them
# ---------------------------------------------------------------------------
def confusion_counts(actual: np.ndarray, predicted: np.ndarray, n: int) -> np.ndarray:
    """(n, n) counts, rows actual, columns predicted, from integer label codes."""
    flat = np.bincount(np.asarray(actual) * n + np.asarray(predicted),
                       minlength=n * n)
    return flat.reshape(n, n)


def metrics_from_confusion(cm: np.ndarray, labels: list) -> dict:
    """Accuracy, per-class precision/recall/F1, macro F1 and balanced accuracy.

    Rows of `cm` are the actual class, columns the predicted class, so the
    support of a class is its row sum and the number of times it was predicted
    is its column sum.

    A class that is present in the data but never predicted scores F1 = 0, not
    NaN. This matters here and is not a detail: the three-label models never
    predict Indeterminate at all, and averaging the macro F1 over only the
    classes a model happens to predict would report a *higher* macro F1 than
    its accuracy -- rewarding the model for declining to try. F1 is therefore
    computed as ``2 tp / (predicted + support)``, which is the usual identity
    and is zero whenever ``tp`` is zero. Precision alone stays NaN when the
    class is never predicted, because 0/0 is genuinely undefined and reporting
    it as zero would read as a measured failure rather than an absence.

    Only a class with no support *and* no predictions is dropped from the macro
    average, and the number of classes averaged is reported so a macro F1 over
    fewer classes can never be mistaken for one over all of them.
    """
    cm = np.asarray(cm, dtype=float)
    total = cm.sum()
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        recall = np.where(support > 0, tp / support, np.nan)
        present = (support + predicted) > 0
        f1 = np.where(present, 2 * tp / np.where(present, support + predicted, 1.0),
                      np.nan)

    out = {
        "accuracy": float(tp.sum() / total) if total else float("nan"),
        "macro_f1": float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan"),
        "n_classes_in_macro_f1": int(np.isfinite(f1).sum()),
        "balanced_accuracy": (float(np.nanmean(recall))
                              if np.isfinite(recall).any() else float("nan")),
        "n_classes_with_support": int((support > 0).sum()),
    }
    for k, name in enumerate(labels):
        out[f"support_{name}"] = int(support[k])
        out[f"n_predicted_{name}"] = int(predicted[k])
        out[f"precision_{name}"] = float(precision[k])
        out[f"recall_{name}"] = float(recall[k])
        out[f"f1_{name}"] = float(f1[k])
    return out


def confusion_frame(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    """Confusion matrix as a labelled frame: rows actual, columns predicted."""
    labels = labels_for(task)
    s = scored(frame, task)
    cm = pd.crosstab(s["actual_label"], s["predicted"])
    return cm.reindex(index=labels, columns=labels, fill_value=0)


def per_class_frame(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    labels = labels_for(task)
    m = metrics_from_confusion(confusion_frame(frame, task).to_numpy(), labels)
    return pd.DataFrame([{
        "label": name,
        "support": m[f"support_{name}"],
        "n_predicted": m[f"n_predicted_{name}"],
        "precision": m[f"precision_{name}"],
        "recall": m[f"recall_{name}"],
        "f1": m[f"f1_{name}"],
    } for name in labels])


# ---------------------------------------------------------------------------
#  Patient-level bootstrap, with draws shared across models
# ---------------------------------------------------------------------------
def patient_draws(patients: list, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_patients) resampling indices into a *sorted* patient list.

    The patient is the resampling unit, never the transition: a patient with
    fourteen visits contributes thirteen correlated predictions, and an
    interval that treated them as independent would be too narrow. This is
    :func:`loo.bootstrap_patient_ci`'s unit, exposed as explicit draws so that
    every model and every paired contrast can be given the *same* resampled
    patients --- which is what makes a paired comparison paired.
    """
    rng = np.random.default_rng(seed)
    return rng.choice(len(patients), size=(int(n_boot), len(patients)), replace=True)


@dataclass
class ScoredArrays:
    """One model's scored predictions, grouped by patient for resampling."""

    patients: list
    actual: dict                    # PATNO -> int codes of the actual label
    predicted: dict                 # PATNO -> int codes of the predicted label
    correct: dict                   # PATNO -> bool array
    n_labels: int


def scored_arrays(frame: pd.DataFrame, task: str,
                  patients: list | None = None) -> ScoredArrays:
    labels = labels_for(task)
    index = {name: k for k, name in enumerate(labels)}
    s = scored(frame, task)
    a = s["actual_label"].map(index).to_numpy(dtype=int)
    p = s["predicted"].map(index).to_numpy(dtype=int)
    c = s["correct"].to_numpy(dtype=bool)
    patno = s["PATNO"].to_numpy()

    keys = sorted(set(patno)) if patients is None else list(patients)
    actual, predicted, correct = {}, {}, {}
    for key in keys:
        m = patno == key
        actual[key], predicted[key], correct[key] = a[m], p[m], c[m]
    return ScoredArrays(patients=keys, actual=actual, predicted=predicted,
                        correct=correct, n_labels=len(labels))


def bootstrap_interval(arrays: ScoredArrays, draws: np.ndarray,
                       alpha: float = 0.05) -> dict:
    """Percentile intervals for accuracy, macro F1 and balanced accuracy.

    All three are recomputed from the resampled confusion matrix on every draw,
    so the interval reflects the metric's own sampling behaviour rather than a
    normal approximation around it.
    """
    labels = [str(k) for k in range(arrays.n_labels)]
    n = arrays.n_labels
    keys = arrays.patients
    acc = np.empty(len(draws))
    mf1 = np.empty(len(draws))
    bal = np.empty(len(draws))

    for b, picked in enumerate(draws):
        a = np.concatenate([arrays.actual[keys[i]] for i in picked])
        p = np.concatenate([arrays.predicted[keys[i]] for i in picked])
        m = metrics_from_confusion(confusion_counts(a, p, n), labels)
        acc[b], mf1[b], bal[b] = (m["accuracy"], m["macro_f1"],
                                  m["balanced_accuracy"])

    def ci(x):
        x = x[np.isfinite(x)]
        if not len(x):
            return float("nan"), float("nan")
        return (float(np.quantile(x, alpha / 2)),
                float(np.quantile(x, 1 - alpha / 2)))

    lo_a, hi_a = ci(acc)
    lo_f, hi_f = ci(mf1)
    lo_b, hi_b = ci(bal)
    return {
        "accuracy_ci_low": lo_a, "accuracy_ci_high": hi_a,
        "macro_f1_ci_low": lo_f, "macro_f1_ci_high": hi_f,
        "balanced_accuracy_ci_low": lo_b, "balanced_accuracy_ci_high": hi_b,
        "n_boot": int(len(draws)),
    }


# ---------------------------------------------------------------------------
#  Headline metrics for one model on one task
# ---------------------------------------------------------------------------
def evaluate(frame: pd.DataFrame, task: str, draws: np.ndarray | None = None,
             patients: list | None = None) -> dict:
    """Every headline number for one model on one task.

    The probabilistic scores are on the task's own probability vector: the
    three-label distribution for :data:`TASK_THREE`, the renormalised
    {TD, PIGD} pair for :data:`TASK_TWO`. A model's log loss is only
    comparable with another's on the same task, and the rule-based arms' log
    loss is finite only because it is clipped, which is flagged.
    """
    name = frame["model"].iloc[0] if len(frame) else ""
    labels = labels_for(task)
    s = scored(frame, task)
    if not len(s):
        return {"model": name, "task": task, "n_scored": 0}

    cm = confusion_frame(frame, task).to_numpy()
    out = {"model": name, "task": task}
    out.update(metrics_from_confusion(cm, labels))

    if task == TASK_THREE:
        probs = three_label_matrix(s)
        eligible = len(frame)
    else:
        probs = np.column_stack([s["P_TD"].to_numpy(float),
                                 s["P_PIGD"].to_numpy(float)])
        eligible = int(frame["model1_eligible"].sum())

    index = {name_: k for k, name_ in enumerate(labels)}
    j = s["actual_label"].map(index).to_numpy(int)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(j)), j] = 1.0
    p_true = probs[np.arange(len(j)), j]
    clipped = bool(np.any(p_true < _LOGLOSS_CLIP))

    out.update({
        "n_patients": int(s["PATNO"].nunique()),
        "n_scored": int(len(s)),
        "n_eligible": eligible,
        "n_undefined_among_eligible": eligible - int(len(s)),
        "n_correct": int(s["correct"].sum()),
        "mean_per_patient_accuracy": float(
            s.groupby("PATNO")["correct"].mean().mean()),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(np.clip(p_true, _LOGLOSS_CLIP, 1.0)))),
        "log_loss_clipped": clipped,
        "n_ties": int(s["three_class_is_tie"].sum() if task == TASK_THREE
                      else s["two_class_is_tie"].sum()),
        "rule_based_probabilities": name in RULE_BASED,
    })
    if draws is not None:
        out.update(bootstrap_interval(scored_arrays(s, task, patients), draws))
    return out


def paired_difference(frame_a: pd.DataFrame, frame_b: pd.DataFrame, task: str,
                      draws: np.ndarray, patients: list,
                      alpha: float = 0.05) -> dict:
    """accuracy_a - accuracy_b on the rows both models scored, paired.

    Two models scored on the same rows are not two independent samples, so
    comparing their separate intervals in the metrics table is the wrong test:
    the intervals overlap heavily even when one model is reliably better. The
    interval here resamples PATIENTS, using the *same* draws as every other
    contrast and as the single-model intervals, so the difference and its
    uncertainty refer to one bootstrap population.

    The exact McNemar test is reported for reference and is the less
    trustworthy of the two, because it treats a patient's several predictions
    as independent when they are not.
    """
    from scipy.stats import binomtest

    a, b = scored(frame_a, task), scored(frame_b, task)
    merged = a.merge(b, on=["PATNO", "visit_index"], suffixes=("_a", "_b"))
    if not len(merged):
        return {"n_paired": 0}

    diff = (merged["correct_a"].to_numpy(float)
            - merged["correct_b"].to_numpy(float))
    patno = merged["PATNO"].to_numpy()
    by_patient = {key: diff[patno == key] for key in patients}

    drawn = np.empty(len(draws))
    for k, picked in enumerate(draws):
        drawn[k] = np.concatenate([by_patient[patients[i]] for i in picked]).mean()

    a_only = int((merged["correct_a"] & ~merged["correct_b"]).sum())
    b_only = int((merged["correct_b"] & ~merged["correct_a"]).sum())
    discordant = a_only + b_only
    low = float(np.quantile(drawn, alpha / 2))
    high = float(np.quantile(drawn, 1 - alpha / 2))

    return {
        "task": task,
        "model_a": frame_a["model"].iloc[0],
        "model_b": frame_b["model"].iloc[0],
        "n_paired": int(len(merged)),
        "n_patients": int(merged["PATNO"].nunique()),
        "accuracy_a": float(merged["correct_a"].mean()),
        "accuracy_b": float(merged["correct_b"].mean()),
        "accuracy_delta": float(diff.mean()),
        "delta_ci_low": low,
        "delta_ci_high": high,
        "ci_includes_zero": bool(low <= 0.0 <= high),
        "a_right_b_wrong": a_only,
        "b_right_a_wrong": b_only,
        "n_discordant": discordant,
        "mcnemar_exact_p_ignores_clustering": (
            float(binomtest(a_only, discordant, 0.5).pvalue)
            if discordant else float("nan")),
    }


# ---------------------------------------------------------------------------
#  Quality checks
# ---------------------------------------------------------------------------
def parameter_checks(fits: dict, horizons=(0.5, 1.0, 2.0, 5.0, 10.0),
                     atol: float = 1e-10) -> dict:
    """The structural invariants of every fitted variant.

    Checked on the fitted objects themselves rather than re-derived by the
    code that produced them: emission rows and the initial distribution must
    lie on the simplex, off-diagonal rates must be non-negative, each diagonal
    generator entry must equal minus the sum of its row, and exp(Q dt) must be
    a stochastic matrix at every horizon.
    """
    out = {}
    for model, vfit in fits.items():
        Q, E, pi = vfit.fit.Q, vfit.fit.E, vfit.fit.pi
        off = ~np.eye(Q.shape[0], dtype=bool)
        P = transition_probabilities(Q, np.asarray(horizons, dtype=float))

        out[f"{model}: emission rows sum to one"] = bool(
            np.allclose(E.sum(axis=1), 1.0, atol=atol))
        out[f"{model}: emission entries in [0, 1]"] = bool(
            (E >= 0).all() and (E <= 1).all())
        out[f"{model}: initial distribution sums to one"] = bool(
            np.isclose(pi.sum(), 1.0, atol=atol) and (pi >= 0).all())
        out[f"{model}: off-diagonal rates are non-negative"] = bool((Q[off] >= 0).all())
        out[f"{model}: generator diagonal is minus its row sum"] = bool(
            np.allclose(np.diag(Q), -(Q * off).sum(axis=1), atol=atol))
        out[f"{model}: exp(Q dt) rows sum to one at every horizon"] = bool(
            np.allclose(P.sum(axis=2), 1.0, atol=1e-8))
        out[f"{model}: exp(Q dt) entries in [0, 1] at every horizon"] = bool(
            (P >= -1e-12).all() and (P <= 1 + 1e-12).all())
        out[f"{model}: parameter count matches the formula"] = (
            vfit.n_params == hmm._n_params(vfit.n_hidden, vfit.n_observed))
    return out


def prediction_checks(predictions: pd.DataFrame, targets: pd.DataFrame,
                      data: TwoStateData, failures: pd.DataFrame,
                      patients: list, atol: float = 1e-12) -> dict:
    """The invariants the conclusions rest on, read off the produced table."""
    completed = set(predictions["held_out_patient"])
    covered = targets[targets["PATNO"].isin(completed)]
    expected_three = frozenset(map(tuple, covered[["PATNO", "visit_index"]].to_numpy()))
    expected_two = frozenset(map(tuple, covered.loc[covered["model1_eligible"],
                                                    ["PATNO", "visit_index"]].to_numpy()))

    per_model = {m: frozenset(map(tuple, g[["PATNO", "visit_index"]].to_numpy()))
                 for m, g in predictions.groupby("model", sort=False)}
    two_sets = {m: frozenset(map(tuple, scored(g, TASK_TWO)[
        ["PATNO", "visit_index"]].to_numpy()))
        for m, g in predictions.groupby("model", sort=False)}

    defined2 = predictions[predictions["two_class_defined"]]
    p_td = defined2["P_TD"].to_numpy(float)
    p_pigd = defined2["P_PIGD"].to_numpy(float)
    three = predictions[predictions["three_class_defined"]]
    three_probs = three_label_matrix(three)

    two_label = predictions[predictions["model"].isin(TWO_LABEL_MODELS)]
    three_label = predictions[predictions["model"].isin(THREE_LABEL_MODELS)]

    return {
        "two-class probabilities sum to one":
            bool(np.allclose(p_td + p_pigd, 1.0, atol=atol)),
        "two-class probabilities in [0, 1]":
            bool((p_td >= 0).all() and (p_td <= 1).all()
                 and (p_pigd >= 0).all() and (p_pigd <= 1).all()),
        "three-class probabilities sum to one":
            bool(np.allclose(three_probs.sum(axis=1), 1.0, atol=1e-9)),
        "three-class probabilities in [0, 1]":
            bool((three_probs >= 0).all() and (three_probs <= 1).all()),
        "Model 1 never carries Indeterminate mass":
            bool(two_label["p_Indeterminate"].isna().all()),
        "Model 1 produces no three-class prediction":
            bool((~two_label["three_class_defined"]).all()),
        "three-label models predict every target visit":
            all(per_model[m] == expected_three for m in THREE_LABEL_MODELS
                if m in per_model),
        "Model 1 predicts exactly the eligible target visits":
            all(per_model[m] == expected_two for m in TWO_LABEL_MODELS
                if m in per_model),
        "every model is scored on the identical common subset":
            len(set(two_sets.values())) == 1,
        "the common subset is exactly the Model-1-eligible targets":
            all(s == expected_two for s in two_sets.values()),
        "target visit is strictly later than the visit predicted from":
            bool((predictions["target_years"] > predictions["prev_years"]).all()),
        "history length is the visits strictly before the target":
            bool((predictions["n_history"] == predictions["visit_index"]).all()),
        "three-label interval equals the gap to the previous labelled visit":
            bool(np.allclose(three_label["model_interval_years"],
                             three_label["interval_years"], atol=1e-8)),
        "Model 1's interval equals the gap to its own previous retained visit":
            bool(np.allclose(
                two_label["model_interval_years"],
                two_label["model1_interval_years"], atol=1e-8)),
        "Model 1's interval is never shorter than the three-label interval":
            bool((two_label["model1_interval_years"]
                  >= two_label["interval_years"] - 1e-8).all()),
        "held-out patient is the patient being predicted":
            bool((predictions["held_out_patient"] == predictions["PATNO"]).all()),
        "every fold trained on all other patients":
            set(predictions.loc[predictions["model"] == MODEL_2X3,
                                "fit_n_train_sequences"].unique())
            == {len(data.full_sequences) - 1},
        "Model 1's folds trained on all other patients with two retained visits":
            set(predictions.loc[predictions["model"] == MODEL_2X2,
                                "fit_n_train_sequences"].unique())
            <= {len(data.binary_sequences), len(data.binary_sequences) - 1},
        "completed and failed folds account for every attempted patient":
            completed.isdisjoint(set(failures["held_out_patient"]))
            and completed | set(failures["held_out_patient"]) == set(patients),
        "target list matches the transition count of the full view":
            len(targets) == int(sum(len(s["gaps"]) for s in data.full_sequences)),
        "eligible target list matches the transition count of the binary view":
            int(targets["model1_eligible"].sum())
            == int(sum(len(s["gaps"]) for s in data.binary_sequences)),
        "Indeterminate targets are scored in the three-class task only":
            int((scored(predictions, TASK_TWO)["actual_label"]
                 == "Indeterminate").sum()) == 0
            and int((scored(three_label, TASK_THREE)["actual_label"]
                     == "Indeterminate").sum()) > 0,
        "Indeterminate survives as input evidence for the three-label models":
            bool((targets["prev_label"] == "Indeterminate").any()),
        "no fold failed silently":
            len(failures) == 0 or bool(set(failures["held_out_patient"])
                                       .isdisjoint(set(predictions["PATNO"]))),
    }


def calibration(frame: pd.DataFrame, edges=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0001),
                min_bin: int = 25) -> pd.DataFrame:
    """Observed PIGD rate against predicted P_PIGD on the common subset.

    Bin edges and the sparse-bin merge rule are :func:`next_visit.calibration`'s,
    so a calibration table from this experiment is read the same way as the
    published one.
    """
    s = scored(frame, TASK_TWO)
    if not len(s):
        return pd.DataFrame(columns=["bin", "n", "mean_P_PIGD", "observed_rate_PIGD"])

    p = s["P_PIGD"].to_numpy(dtype=float)
    actual = (s["actual_label"] == "PIGD").to_numpy(dtype=float)
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


def accuracy_breakdowns(predictions: pd.DataFrame, task: str,
                        interval_split: float = nv.INTERVAL_SPLIT_YEARS) -> dict:
    """Accuracy per model on the cuts that explain a difference rather than restate it.

    By inter-visit interval, by the label recorded at the previous visit, and
    by how many of the patient's own visits the prediction could see. The
    interval split is :data:`next_visit.INTERVAL_SPLIT_YEARS`, for the reason
    documented there: PPMI's schedule is annual early and biennial later, so a
    model that reads elapsed time has almost nothing to read on most cases.
    """
    rows_interval, rows_prev, rows_history = [], [], []

    for name, g in predictions.groupby("model", sort=False):
        s = scored(g, task)
        if not len(s):
            continue

        long = s["interval_years"] > interval_split
        for label, mask in ((f"<= {interval_split}y", ~long),
                            (f"> {interval_split}y", long)):
            sub = s[mask]
            rows_interval.append({
                "model": name, "interval": label, "n": int(len(sub)),
                "accuracy": float(sub["correct"].mean()) if len(sub) else float("nan")})

        for label in THREE_LABELS:
            sub = s[s["prev_label"] == label]
            rows_prev.append({
                "model": name, "previous_label": label, "n": int(len(sub)),
                "accuracy": float(sub["correct"].mean()) if len(sub) else float("nan")})

        bucket = s["n_history"].clip(upper=5)
        for k, sub in s.groupby(bucket):
            rows_history.append({
                "model": name,
                "n_prior_visits": f"{int(k)}+" if k == 5 else str(int(k)),
                "n": int(len(sub)), "accuracy": float(sub["correct"].mean())})

    return {
        "by_interval": pd.DataFrame(rows_interval),
        "by_previous_label": pd.DataFrame(rows_prev),
        "by_history_length": pd.DataFrame(rows_history),
    }
