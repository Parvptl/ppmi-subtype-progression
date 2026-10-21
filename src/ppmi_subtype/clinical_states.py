"""Latent clinical states from multiple measurements: a CT-HMM with Gaussian emissions.

Why a new emission model was needed
-----------------------------------
:mod:`ppmi_subtype.hmm` is already a continuous-time HMM --- its latent state
evolves under a generator ``Q`` built by :func:`ctmc.build_generator` and
propagated by ``exp(Q*dt)``. What it cannot do is read a continuous
observation: its emission ``E`` is a ``(n_hidden, n_observed)`` softmax matrix
indexed by an integer label (``E[:, labels[:, 0]]`` in
``hmm._forward_loglik``). There is no way to hand it a ten-dimensional vector
of clinical scores without replacing that emission outright.

So this module keeps the part that transfers and replaces the part that does
not:

* **reused unchanged** --- :func:`ctmc.build_generator` and
  :func:`ctmc.transition_probabilities` for the latent continuous-time
  process, so the transition machinery is the same audited code the
  subtype-history model runs on;
* **new** --- a diagonal multivariate Gaussian emission per latent state, and
  an EM fitter for it, because the categorical emission cannot be adapted.

Missing measurements are **marginalised, not imputed**. With a diagonal
covariance the emission log-density is a sum over features, so a feature that
was not recorded at a visit is simply dropped from that sum. This is exact
rather than approximate, and it removes imputation --- and the leakage risk
that comes with fitting an imputer --- from the pipeline entirely. A visit with
no features at all contributes a flat likelihood, which is the correct
statement that it carries no information about the state.

What the latent states are and are not
--------------------------------------
They are recurring configurations of ten clinical measurements that recur
across patients and persist across visits. They are **not** biological disease
stages, and nothing here establishes that they are. State indices are
arbitrary model labels; :func:`canonical_order` sorts them by overall symptom
burden so that fits from different folds can be compared, and that ordering is
a reporting convenience, not a claim that the states lie on a single axis.

Fitting uses smoothing (forward-backward over the whole training sequence),
which is correct for estimating parameters. Prediction uses **filtering only**
--- :meth:`CTHMMGaussian.filter_forward` never runs a backward pass --- so no
prediction can see the visit it is predicting.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from . import ctmc, loaders
from .config import Config
from .subtype import clean_items, deduplicate_visits, select_medication_state

_EPS = 1e-300
_LOG_RATE_BOUNDS = ctmc._LOG_RATE_BOUNDS

#: Part III items that are neither tremor nor PIGD constituents: the
#: bradykinesia / rigidity / axial axis. The project's tremor and PIGD scores
#: between them use 13 of the 33 Part III items, and ``updrs3_score`` is the
#: total of all of them --- so adding that total to a feature set that already
#: carries tremor and PIGD would count those items twice. These 20 items are
#: the remainder, and they carry the motor axis the other two scores miss.
BRADY_RIGID_ITEMS = (
    "NP3SPCH",   # 3.1  speech
    "NP3FACXP",  # 3.2  facial expression
    "NP3RIGN",   # 3.3a rigidity, neck
    "NP3RIGRU",  # 3.3b rigidity, RUE
    "NP3RIGLU",  # 3.3c rigidity, LUE
    "NP3RIGRL",  # 3.3d rigidity, RLE
    "NP3RIGLL",  # 3.3e rigidity, LLE
    "NP3FTAPR",  # 3.4a finger tapping, right
    "NP3FTAPL",  # 3.4b finger tapping, left
    "NP3HMOVR",  # 3.5a hand movements, right
    "NP3HMOVL",  # 3.5b hand movements, left
    "NP3PRSPR",  # 3.6a pronation-supination, right
    "NP3PRSPL",  # 3.6b pronation-supination, left
    "NP3TTAPR",  # 3.7a toe tapping, right
    "NP3TTAPL",  # 3.7b toe tapping, left
    "NP3LGAGR",  # 3.8a leg agility, right
    "NP3LGAGL",  # 3.8b leg agility, left
    "NP3RISNG",  # 3.9  arising from chair
    "NP3POSTR",  # 3.13 posture
    "NP3BRADY",  # 3.14 global spontaneity of movement
)

#: Items of the 20 above that may be absent before the score is withheld. Two
#: is one tenth of the scale; PPMI's own subscores tolerate none, but those are
#: five-item scales where one absence is a fifth.
MAX_MISSING_BRADY_ITEMS = 2


@dataclass(frozen=True)
class Feature:
    """One clinical measurement in the representation."""

    name: str
    column: str
    domain: str
    description: str
    #: ``"log1p"`` for a strictly positive, heavily skewed quantity; ``None``
    #: to use the recorded value.
    transform: str | None = None
    #: True where a higher score is worse. Only used for reporting, so a state
    #: profile can be read without consulting each scale's direction.
    higher_is_worse: bool = True


#: The representation. Ten measurements, one or two per clinical domain, each
#: an established PPMI scale or a subscore built from its items --- not a bank
#: of engineered features. Pairwise correlations peak at 0.65
#: (PIGD against motor ADL), so none is redundant and no dimensionality
#: reduction is applied; with ten columns and ~4,300 visits there is nothing
#: for a projection to buy, and it would cost the per-feature interpretation
#: the state profiles depend on.
FEATURES: tuple[Feature, ...] = (
    Feature("brady_rigid", "brady_rigid_score", "motor: bradykinesia/rigidity",
            "mean of 20 MDS-UPDRS Part III items excluding tremor and PIGD items"),
    Feature("tremor", "tremor_score", "motor: tremor",
            "mean of the 11 tremor items (project definition)"),
    Feature("pigd", "pigd_score", "motor: gait/postural",
            "mean of the 5 gait/postural items (project definition)"),
    Feature("updrs2", "updrs2_score", "function: motor ADL",
            "MDS-UPDRS Part II total, motor experiences of daily living"),
    Feature("updrs1", "updrs1_score", "non-motor: total burden",
            "MDS-UPDRS Part I total, non-motor experiences of daily living"),
    Feature("moca", "moca", "cognition",
            "Montreal Cognitive Assessment total", higher_is_worse=False),
    Feature("gds", "gds", "mood",
            "Geriatric Depression Scale, 15-item"),
    Feature("scopa", "scopa", "autonomic",
            "SCOPA-AUT total, autonomic dysfunction"),
    Feature("ess", "ess", "sleep",
            "Epworth Sleepiness Scale total"),
    Feature("ledd", "LEDD", "treatment",
            "levodopa-equivalent daily dose, log1p of mg/day", transform="log1p"),
)

FEATURE_NAMES = tuple(f.name for f in FEATURES)

#: Minimum features a visit must carry to enter the model. Below this the
#: emission is too thin to place the visit, and the visit would contribute
#: almost nothing while still consuming a transition.
MIN_FEATURES_PER_VISIT = 8

#: Observed subtype labels, in the order every probability array uses. Shared
#: with the subtype-history experiment so the two are directly comparable.
STATE_NAMES = ["TD", "Indeterminate", "PIGD"]


# ---------------------------------------------------------------------------
#  Feature construction
# ---------------------------------------------------------------------------
def build_brady_rigid(cfg: Config) -> pd.DataFrame:
    """Per patient-visit bradykinesia/rigidity score from raw Part III items.

    Built with the project's own primitives and in the same order
    ``panel.build_subscores`` uses them, so the medication state, the duplicate
    resolution and the sentinel handling are identical to the tremor and PIGD
    scores this score sits beside: ``select_medication_state`` keeps the
    OFF/untreated exam, ``deduplicate_visits`` collapses a visit to one row,
    and ``clean_items`` blanks PPMI's "Unable to Rate" sentinel 101 rather
        than averaging it as a score.
    """
    scfg = cfg.subtype
    part3 = loaders.load_updrs_part3(cfg)
    part3 = select_medication_state(part3, scfg)
    part3 = deduplicate_visits(part3, scfg)

    absent = [c for c in BRADY_RIGID_ITEMS if c not in part3.columns]
    if absent:
        raise KeyError(f"Part III file is missing {len(absent)} items: {absent}")

    block = clean_items(part3, BRADY_RIGID_ITEMS, scfg)
    n_missing = block.isna().sum(axis=1)
    score = block.mean(axis=1, skipna=True).where(n_missing <= MAX_MISSING_BRADY_ITEMS)

    return pd.DataFrame({
        "PATNO": part3["PATNO"].to_numpy(),
        "EVENT_ID": part3["EVENT_ID"].to_numpy(),
        "brady_rigid_score": score.to_numpy(),
        "brady_rigid_n_missing_items": n_missing.to_numpy(),
    })


def build_clinical_panel(cfg: Config, panel: pd.DataFrame) -> pd.DataFrame:
    """Labelled visits with the ten clinical features, in chronological order.

    Only visits carrying a subtype label are kept, which is what makes the
    cohort comparable with the subtype-history experiment: both are defined on
    the same labelled-visit panel. A visit is additionally dropped when it
    carries fewer than :data:`MIN_FEATURES_PER_VISIT` of the ten.
    """
    brady = build_brady_rigid(cfg)
    frame = panel[panel["subtype"].notna()].merge(
        brady, on=["PATNO", "EVENT_ID"], how="left")

    out = frame[["PATNO", "EVENT_ID", "years_from_baseline", "visit_date",
                 "subtype", "age_at_visit"]].copy()
    for f in FEATURES:
        if f.column not in frame.columns:
            raise KeyError(f"panel has no column {f.column!r} for feature {f.name!r}")
        value = pd.to_numeric(frame[f.column], errors="coerce")
        out[f.name] = np.log1p(value) if f.transform == "log1p" else value

    out["n_features_present"] = out[list(FEATURE_NAMES)].notna().sum(axis=1)
    out = out[out["n_features_present"] >= MIN_FEATURES_PER_VISIT]
    out = out.sort_values(["PATNO", "years_from_baseline"]).reset_index(drop=True)
    out["visit_index"] = out.groupby("PATNO").cumcount()
    return out


def feature_inventory(clinical: pd.DataFrame) -> pd.DataFrame:
    """Coverage and distribution of every feature, for the report."""
    rows = []
    for f in FEATURES:
        v = clinical[f.name]
        rows.append({
            "feature": f.name, "source_column": f.column, "domain": f.domain,
            "description": f.description,
            "transform": f.transform or "none",
            "higher_is_worse": f.higher_is_worse,
            "n_observed": int(v.notna().sum()),
            "coverage": float(v.notna().mean()),
            "mean": float(v.mean()), "sd": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
        })
    return pd.DataFrame(rows)


def build_sequences(clinical: pd.DataFrame, min_length: int = 2) -> list[dict]:
    """One entry per patient: feature matrix, visit times, labels and gaps.

    ``X`` keeps NaN where a measurement was not recorded; the emission
    marginalises those entries rather than filling them.
    """
    sequences = []
    for patno, g in clinical.groupby("PATNO", sort=False):
        g = g.sort_values("years_from_baseline")
        if len(g) < min_length:
            continue
        times = g["years_from_baseline"].to_numpy(dtype=float)
        sequences.append({
            "PATNO": patno,
            "X": g[list(FEATURE_NAMES)].to_numpy(dtype=float),
            "times": times,
            "gaps": np.diff(times),
            "labels": g["subtype"].to_numpy(dtype=int),
            "events": g["EVENT_ID"].to_numpy(),
            "visit_index": g["visit_index"].to_numpy(dtype=int),
        })
    return sequences


# ---------------------------------------------------------------------------
#  Training-only standardisation
# ---------------------------------------------------------------------------
@dataclass
class Standardiser:
    """Per-feature centring and scaling, learned from training visits only.

    The Gaussian emission does not need standardised inputs to be correct, but
    it does need them to be initialised sensibly: LEDD on a log scale and MoCA
    on a 0-30 scale otherwise give starting variances that differ by orders of
    magnitude, and EM then spends its early iterations undoing that rather
    than finding states. Means and SDs are computed ignoring NaN, so a
    feature's scale is set by the visits that recorded it.
    """

    centre: np.ndarray
    scale: np.ndarray
    features: tuple[str, ...] = FEATURE_NAMES
    n_train_visits: int = 0

    @classmethod
    def fit(cls, sequences: list[dict]) -> "Standardiser":
        if not sequences:
            raise ValueError("no sequences supplied")
        X = np.vstack([s["X"] for s in sequences])
        with np.errstate(invalid="ignore"):
            centre = np.nanmean(X, axis=0)
            scale = np.nanstd(X, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        centre = np.where(np.isfinite(centre), centre, 0.0)
        return cls(centre=centre, scale=scale, n_train_visits=int(len(X)))

    def apply(self, sequences: list[dict]) -> list[dict]:
        """Standardised copies. NaN stays NaN -- it is marginalised, not filled."""
        return [dict(s, X=(s["X"] - self.centre) / self.scale) for s in sequences]

    def invert(self, Z: np.ndarray) -> np.ndarray:
        """Back to the original units, for reading state profiles."""
        return np.asarray(Z, dtype=float) * self.scale + self.centre


# ---------------------------------------------------------------------------
#  The model
# ---------------------------------------------------------------------------
#: Variance floor on the standardised scale. A state that captures very few
#: visits can otherwise collapse onto them with a vanishing variance and an
#: unbounded likelihood, which is the classic Gaussian-mixture degeneracy.
VARIANCE_FLOOR = 1e-3


@dataclass
class CTHMMGaussian:
    """Latent continuous-time chain with diagonal-Gaussian emissions.

    ``Q`` is the generator over ``n_states`` latent states, ``mu`` and ``var``
    are ``(n_states, n_features)`` emission parameters on the standardised
    scale, and ``pi`` is the distribution over the state at a patient's first
    visit.
    """

    Q: np.ndarray
    mu: np.ndarray
    var: np.ndarray
    pi: np.ndarray
    loglik: float = float("nan")
    n_iter: int = 0
    converged: bool = False
    n_sequences: int = 0
    n_observations: int = 0
    #: Fitted on training folds only; P(observed subtype label | latent state).
    label_emission: np.ndarray | None = None

    @property
    def n_states(self) -> int:
        return self.Q.shape[0]

    @property
    def n_features(self) -> int:
        return self.mu.shape[1]

    @property
    def n_params(self) -> int:
        """Free parameters: rates, means, variances and the initial simplex."""
        k, d = self.n_states, self.n_features
        return k * (k - 1) + 2 * k * d + (k - 1)

    def bic(self, n_observations: int | None = None) -> float:
        n = self.n_observations if n_observations is None else n_observations
        return self.n_params * np.log(max(n, 2)) - 2 * self.loglik

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik

    # -- emissions ---------------------------------------------------------
    def emission_loglik(self, X: np.ndarray) -> np.ndarray:
        """``(n_visits, n_states)`` log-density, marginalising missing features.

        With a diagonal covariance the log-density is a sum over features, so
        dropping the terms for features that were not recorded is exact
        marginalisation and not an approximation.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        observed = np.isfinite(X)
        filled = np.where(observed, X, 0.0)

        # (n, k, d): per-feature log N(x_d; mu_kd, var_kd), zeroed where absent.
        diff = filled[:, None, :] - self.mu[None, :, :]
        terms = -0.5 * (np.log(2 * np.pi * self.var)[None, :, :]
                        + diff ** 2 / self.var[None, :, :])
        return np.einsum("nkd,nd->nk", terms, observed.astype(float))

    # -- filtering, the only recursion prediction may use ------------------
    def filter_forward(self, X: np.ndarray, gaps: np.ndarray) -> tuple[np.ndarray, float]:
        """Filtered state distributions and the sequence log-likelihood.

        ``alpha[k]`` is ``P(state at visit k | observations 0..k)`` --- strictly
        causal. There is deliberately no backward pass here: a smoothed
        posterior would carry information from later visits, and every
        prediction in this experiment is made from this function's output.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n = len(X)
        logb = self.emission_loglik(X)
        alpha = np.empty((n, self.n_states))

        log_pi = np.log(np.maximum(self.pi, _EPS))
        a = log_pi + logb[0]
        total = float(logsumexp(a))
        alpha[0] = np.exp(a - total)

        if n > 1:
            P = ctmc.transition_probabilities(self.Q, np.asarray(gaps, dtype=float))
            for k in range(n - 1):
                nxt = alpha[k] @ P[k] * np.exp(logb[k + 1] - logb[k + 1].max())
                s = nxt.sum()
                if not s > 0:
                    alpha[k + 1] = alpha[k] @ P[k]
                    continue
                alpha[k + 1] = nxt / s
                total += float(np.log(s) + logb[k + 1].max())
        return alpha, total

    def predict_next_state(self, alpha_k: np.ndarray, gap: float) -> np.ndarray:
        """Propagate a filtered distribution forward over the real elapsed time."""
        P = ctmc.transition_probabilities(self.Q, np.array([float(gap)]))[0]
        out = np.asarray(alpha_k, dtype=float) @ P
        s = out.sum()
        return out / s if s > 0 else np.full(self.n_states, 1.0 / self.n_states)

    def predict_next_observation_loglik(
        self, next_state: np.ndarray, x_next: np.ndarray
    ) -> float:
        """log P(next visit's measurements | predicted next-state distribution).

        A proper score for the latent representation: it uses the actual next
        measurements only to *evaluate* a distribution that was formed without
        them.
        """
        logb = self.emission_loglik(np.atleast_2d(x_next))[0]
        w = np.log(np.maximum(np.asarray(next_state, dtype=float), _EPS))
        return float(logsumexp(w + logb))

    def predict_next_label_probs(self, next_state: np.ndarray) -> np.ndarray:
        """P(next observed subtype label) over [TD, Indeterminate, PIGD].

        Uses :attr:`label_emission`, the training-fold estimate of
        P(observed label | latent state). Without it the model says nothing
        about the observed label, and this raises rather than inventing a
        number.
        """
        if self.label_emission is None:
            raise ValueError(
                "no label emission fitted; this model has no observation link "
                "to the observed subtype label")
        out = np.asarray(next_state, dtype=float) @ self.label_emission
        s = out.sum()
        return out / s if s > 0 else np.full(len(STATE_NAMES), 1.0 / len(STATE_NAMES))

    # -- reporting ---------------------------------------------------------
    def state_profile(self, standardiser: Standardiser) -> pd.DataFrame:
        """Each state's mean feature vector, in the original clinical units."""
        means = standardiser.invert(self.mu)
        out = pd.DataFrame(means, columns=list(feature_labels(self.n_features)))
        out.insert(0, "state", range(self.n_states))
        return out

    def mean_sojourn(self) -> np.ndarray:
        d = np.diag(self.Q)
        with np.errstate(divide="ignore"):
            return np.where(d < 0, -1.0 / d, np.inf)


def feature_labels(n_features: int) -> tuple[str, ...]:
    """Column names for a feature matrix of this width.

    The project names when the width matches :data:`FEATURES`, generic ones
    otherwise, for the same reason as :func:`burden_signs`: a mismatched width
    is not the project feature set and must not be labelled as if it were.
    """
    if n_features == len(FEATURE_NAMES):
        return FEATURE_NAMES
    return tuple(f"f{i}" for i in range(n_features))


def burden_signs(n_features: int) -> np.ndarray:
    """Per-feature sign making "higher is worse" the positive direction.

    Taken from :data:`FEATURES` when the data has exactly that many columns.
    A different width means the caller is not using the project feature set
    --- a synthetic fit in a test, or a future reduced set --- and there is no
    basis for signing those columns, so they are all treated as "higher is
    worse". Silently broadcasting the ten project signs onto a different
    width would be a shape error at best and a wrong ordering at worst.
    """
    if n_features == len(FEATURES):
        return np.array([1.0 if f.higher_is_worse else -1.0 for f in FEATURES])
    return np.ones(n_features)


def canonical_order(fit: CTHMMGaussian, standardiser: Standardiser) -> np.ndarray:
    """Permutation sorting states by overall symptom burden, ascending.

    Latent indices are arbitrary: EM from a different fold may find the same
    states in a different order, so comparing "state 2" across folds is
    meaningless without an alignment rule. The rule here is a single
    documented scalar --- the sum of the standardised means with each
    feature signed so that higher is worse --- which is reproducible and
    independent of any clinical reading of the states. It orders the states;
    it does not assert that they lie on one axis.
    """
    burden = (fit.mu * burden_signs(fit.n_features)[None, :]).sum(axis=1)
    return np.argsort(burden, kind="mergesort")


def reorder(fit: CTHMMGaussian, order: np.ndarray) -> CTHMMGaussian:
    """Apply a state permutation to every parameter block."""
    order = np.asarray(order, dtype=int)
    return dataclasses.replace(
        fit,
        Q=fit.Q[np.ix_(order, order)],
        mu=fit.mu[order],
        var=fit.var[order],
        pi=fit.pi[order],
        label_emission=(None if fit.label_emission is None
                        else fit.label_emission[order]),
    )


# ---------------------------------------------------------------------------
#  EM
# ---------------------------------------------------------------------------
def _forward_backward(fit: CTHMMGaussian, seq: dict, P_by_gap, gap_index):
    """Smoothed posteriors and soft transition counts for one sequence.

    Smoothing is correct here --- this is the E-step of a fit on training
    patients, where the whole sequence is legitimately available. It is never
    used to form a prediction; :meth:`CTHMMGaussian.filter_forward` is.
    """
    X, k_states = seq["X"], fit.n_states
    n = len(X)
    logb = fit.emission_loglik(X)
    shift = logb.max(axis=1, keepdims=True)
    b = np.exp(logb - shift)

    alpha = np.empty((n, k_states))
    scale = np.empty(n)
    a = fit.pi * b[0]
    scale[0] = a.sum()
    if not scale[0] > 0:
        return None
    alpha[0] = a / scale[0]

    idx = gap_index[:n - 1] if n > 1 else np.empty(0, dtype=int)
    for t in range(n - 1):
        a = (alpha[t] @ P_by_gap[idx[t]]) * b[t + 1]
        scale[t + 1] = a.sum()
        if not scale[t + 1] > 0:
            return None
        alpha[t + 1] = a / scale[t + 1]

    beta = np.ones((n, k_states))
    for t in range(n - 2, -1, -1):
        beta[t] = P_by_gap[idx[t]] @ (b[t + 1] * beta[t + 1]) / scale[t + 1]

    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), _EPS)

    xi = np.zeros((n - 1, k_states, k_states)) if n > 1 else np.zeros((0, k_states, k_states))
    for t in range(n - 1):
        m = (alpha[t][:, None] * P_by_gap[idx[t]]
             * (b[t + 1] * beta[t + 1])[None, :] / scale[t + 1])
        s = m.sum()
        xi[t] = m / s if s > 0 else m

    loglik = float(np.log(np.maximum(scale, _EPS)).sum() + shift.sum())
    return gamma, xi, loglik


#: L-BFGS-B iterations allowed in the generator M-step per EM iteration.
#:
#: Solving that M-step to convergence is what made EM expensive: with finite
#: differences over K(K-1) rates it dominates the E-step by ~200x at eight
#: states and runs to its own iteration cap every time. EM does not need a
#: maximised M-step, only an improved one -- that is generalised EM, and it
#: keeps the observed-data log-likelihood non-decreasing, which
#: `test_em_log_likelihood_is_non_decreasing` checks. The step is warm-started
#: from the current generator, so successive partial steps accumulate.
GENERATOR_MSTEP_MAXITER = 15


def _fit_generator(counts_by_gap: np.ndarray, unique_gaps: np.ndarray,
                   Q_init: np.ndarray, n_states: int,
                   maxiter: int = GENERATOR_MSTEP_MAXITER) -> np.ndarray:
    """M-step for Q: maximise the soft-count-weighted panel log-likelihood.

    ``counts_by_gap[g, i, j]`` is the expected number of i->j transitions over
    elapsed time ``unique_gaps[g]``. Maximising
    ``sum_g sum_ij counts[g,i,j] log P(unique_gaps[g])[i,j]`` is exactly the
    interval-censored CTMC likelihood of :mod:`ppmi_subtype.ctmc` with soft
    rather than hard counts, so the same parameterisation and bounds apply:
    rates are ``exp(theta)`` and bounded away from zero and from a transition
    per week.
    """
    off = [(i, j) for i in range(n_states) for j in range(n_states) if i != j]

    def negll(theta):
        Q = ctmc.build_generator(np.exp(theta), n_states)
        P = ctmc.transition_probabilities(Q, unique_gaps)
        return -float(np.sum(counts_by_gap * np.log(np.maximum(P, _EPS))))

    theta0 = np.log(np.clip([Q_init[i, j] for i, j in off], 1e-6, None))
    theta0 = np.clip(theta0, *_LOG_RATE_BOUNDS)
    res = minimize(negll, theta0, method="L-BFGS-B",
                   bounds=[_LOG_RATE_BOUNDS] * len(off),
                   options={"maxiter": int(maxiter), "ftol": 1e-10})
    return ctmc.build_generator(np.exp(res.x), n_states)


def _initialise(sequences: list[dict], n_states: int, seed: int) -> CTHMMGaussian:
    """Starting parameters: states spread along the overall symptom burden.

    A random start would place several states on top of each other and let EM
    merge them. Instead the training visits are ordered by a single burden
    score and split into ``n_states`` quantile bins, whose means and variances
    seed the emissions. This is deterministic given the training set, which is
    what makes a fold reproducible.
    """
    X = np.vstack([s["X"] for s in sequences])
    sign = burden_signs(X.shape[1])
    with np.errstate(invalid="ignore"):
        burden = np.nansum(np.where(np.isfinite(X), X, np.nan) * sign[None, :], axis=1)
    burden = np.where(np.isfinite(burden), burden, np.nanmedian(burden))

    edges = np.quantile(burden, np.linspace(0, 1, n_states + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.clip(np.digitize(burden, edges[1:-1], right=False), 0, n_states - 1)

    rng = np.random.default_rng(seed)
    mu = np.zeros((n_states, X.shape[1]))
    var = np.ones((n_states, X.shape[1]))
    for k in range(n_states):
        block = X[bins == k]
        if not len(block):
            mu[k] = rng.normal(0, 0.5, X.shape[1])
            continue
        with np.errstate(invalid="ignore"):
            m = np.nanmean(block, axis=0)
            v = np.nanvar(block, axis=0)
        mu[k] = np.where(np.isfinite(m), m, 0.0)
        var[k] = np.maximum(np.where(np.isfinite(v), v, 1.0), VARIANCE_FLOOR)

    rates = np.full(n_states * (n_states - 1), 0.15)
    counts = np.bincount(bins, minlength=n_states).astype(float)
    return CTHMMGaussian(
        Q=ctmc.build_generator(rates, n_states),
        mu=mu, var=var,
        pi=(counts + 1.0) / (counts + 1.0).sum(),
        n_sequences=len(sequences), n_observations=int(len(X)),
    )


def fit_em(
    sequences: list[dict],
    n_states: int,
    seed: int = 42,
    max_iter: int = 150,
    tol: float = 1e-5,
    gap_round_dp: int = 3,
) -> CTHMMGaussian:
    """Fit by Expectation-Maximisation.

    The emission M-step is closed form (soft-weighted means and variances, over
    the visits where each feature was recorded). The generator M-step is
    numerical, because under interval censoring there is no closed form for
    ``Q``; it reuses the CTMC panel likelihood with soft counts. Convergence is
    declared on the relative change in the observed-data log-likelihood, which
    EM guarantees is non-decreasing --- so a decrease is a bug or a numerical
    failure, and `converged` stays False if the loop runs out of iterations.
    """
    if not sequences:
        raise ValueError("no sequences supplied")
    if n_states < 2:
        raise ValueError(f"n_states must be at least 2; got {n_states}")

    return _fit_em_from(sequences, _initialise(sequences, n_states, seed),
                        max_iter=max_iter, tol=tol, gap_round_dp=gap_round_dp)


def _fit_em_from(sequences, fit: CTHMMGaussian, max_iter: int = 150,
                 tol: float = 1e-5, gap_round_dp: int = 3) -> CTHMMGaussian:
    """The EM loop proper, from a supplied starting point.

    The E-step runs over all patients at once (:func:`batched_estep`); the
    emission M-step is closed form; the generator M-step is numerical, because
    under interval censoring there is no closed form for ``Q``. Convergence is
    declared on the relative change in the observed-data log-likelihood, which
    EM guarantees is non-decreasing, so `converged` stays False when the loop
    runs out of iterations instead.
    """
    n_states = fit.n_states
    batch = prepare_batch(sequences, gap_round_dp)
    fit = dataclasses.replace(fit, n_sequences=batch.n_sequences,
                              n_observations=batch.n_observations)

    previous = -np.inf
    for iteration in range(1, max_iter + 1):
        P_by_gap = ctmc.transition_probabilities(fit.Q, batch.unique_gaps)
        stats = batched_estep(fit, batch, P_by_gap)

        safe = np.maximum(stats["weight_obs"], 1e-8)
        mu = stats["weighted_x"] / safe
        var = np.maximum(stats["weighted_xx"] / safe - mu ** 2, VARIANCE_FLOOR)
        # A state-feature pair no visit informs keeps its previous value rather
        # than collapsing to zero mean and a floored variance.
        thin = stats["weight_obs"] < 1e-6
        mu = np.where(thin, fit.mu, mu)
        var = np.where(thin, fit.var, var)

        pi = stats["pi"] / max(stats["pi"].sum(), _EPS)
        Q = _fit_generator(stats["counts_by_gap"], batch.unique_gaps,
                           fit.Q, n_states)

        total_ll = stats["loglik"]
        fit = dataclasses.replace(fit, Q=Q, mu=mu, var=var, pi=pi,
                                  loglik=total_ll, n_iter=iteration)

        if np.isfinite(previous):
            rel = abs(total_ll - previous) / max(abs(previous), 1.0)
            if rel < tol:
                fit = dataclasses.replace(fit, converged=True)
                break
        previous = total_ll

    return fit


# ---------------------------------------------------------------------------
#  Linking latent states to the observed subtype label
# ---------------------------------------------------------------------------
def fit_label_emission(
    fit: CTHMMGaussian, sequences: list[dict], smoothing: float = 0.5
) -> np.ndarray:
    """P(observed subtype label | latent clinical state), from training data.

    This is the observation model that lets a latent clinical state say
    anything about the recorded TD/Indeterminate/PIGD label. It is estimated by
    accumulating each training visit's smoothed state posterior against its
    recorded label:

        B[k, l] = sum_visits gamma_k(visit) * 1[label(visit) = l]

    row-normalised with Laplace smoothing so no label is impossible. Smoothing
    is appropriate at this step because it is parameter estimation on training
    patients, exactly as in the EM fit; prediction never uses it.

    The link is *justified*, not assumed: tremor and PIGD scores are two of the
    ten features, so a latent state that separates them carries real
    information about the label --- but the label is a thresholded ratio of
    those two scores, and a state is a distribution over all ten, so the
    mapping is genuinely probabilistic rather than a relabelling.
    """
    n_labels = len(STATE_NAMES)
    counts = np.full((fit.n_states, n_labels), smoothing)

    all_gaps = np.round(np.concatenate(
        [s["gaps"] for s in sequences if len(s["gaps"])]), 3)
    unique_gaps = np.unique(all_gaps)
    position = {g: i for i, g in enumerate(unique_gaps)}
    P_by_gap = ctmc.transition_probabilities(fit.Q, unique_gaps)

    for seq in sequences:
        idx = np.array([position[g] for g in np.round(seq["gaps"], 3)], dtype=int)
        res = _forward_backward(fit, seq, P_by_gap, idx)
        if res is None:
            continue
        gamma = res[0]
        # Labels are 1/2/3 in the config; shift to 0-based column indices.
        for t, label in enumerate(seq["labels"]):
            counts[:, int(label) - 1] += gamma[t]

    return counts / counts.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
#  State description, for interpretation
# ---------------------------------------------------------------------------
def describe_states(
    fit: CTHMMGaussian, sequences: list[dict], standardiser: Standardiser
) -> dict[str, pd.DataFrame]:
    """Profiles, occupancy, assignment uncertainty and transition patterns.

    Occupancy and uncertainty are computed from the *filtered* posterior, the
    same quantity prediction uses, so the description characterises the states
    as the predictor actually sees them.
    """
    k = fit.n_states
    occupancy = np.zeros(k)
    hard = np.zeros(k)
    entropies, max_posteriors = [], []
    labels_by_state = np.zeros((k, len(STATE_NAMES)))

    for seq in sequences:
        alpha, _ = fit.filter_forward(seq["X"], seq["gaps"])
        occupancy += alpha.sum(axis=0)
        assign = alpha.argmax(axis=1)
        hard += np.bincount(assign, minlength=k)
        max_posteriors.append(alpha.max(axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            e = -np.sum(np.where(alpha > 0, alpha * np.log(alpha), 0.0), axis=1)
        entropies.append(e)
        for t, label in enumerate(seq["labels"]):
            labels_by_state[assign[t], int(label) - 1] += 1

    max_posteriors = np.concatenate(max_posteriors)
    entropies = np.concatenate(entropies)
    n_visits = int(occupancy.sum())

    profile = fit.state_profile(standardiser)
    profile["soft_occupancy"] = occupancy / max(n_visits, 1)
    profile["n_visits_hard"] = hard.astype(int)
    profile["share_visits_hard"] = hard / max(hard.sum(), 1)
    profile["mean_sojourn_years"] = fit.mean_sojourn()
    for i, name in enumerate(STATE_NAMES):
        with np.errstate(invalid="ignore"):
            profile[f"share_{name}"] = np.where(
                hard > 0, labels_by_state[:, i] / np.maximum(hard, 1), np.nan)

    uncertainty = pd.DataFrame({
        "metric": ["mean max posterior", "median max posterior",
                   "share of visits with max posterior < 0.5",
                   "share of visits with max posterior > 0.9",
                   "mean posterior entropy (nats)",
                   "max possible entropy (nats)"],
        "value": [float(max_posteriors.mean()), float(np.median(max_posteriors)),
                  float((max_posteriors < 0.5).mean()),
                  float((max_posteriors > 0.9).mean()),
                  float(entropies.mean()), float(np.log(k))],
    })

    rates = pd.DataFrame(fit.Q, index=[f"from_{i}" for i in range(k)],
                         columns=[f"to_{i}" for i in range(k)])
    annual = pd.DataFrame(
        ctmc.transition_probabilities(fit.Q, np.array([1.0]))[0],
        index=[f"from_{i}" for i in range(k)],
        columns=[f"to_{i}" for i in range(k)])

    return {"profile": profile, "uncertainty": uncertainty,
            "generator": rates.reset_index(names="state"),
            "annual_transition": annual.reset_index(names="state")}


# ---------------------------------------------------------------------------
#  Restarts, because a single EM start is not reliable here
# ---------------------------------------------------------------------------
def dispersed_starts(n_states: int, n_starts: int, seed: int) -> list[int]:
    """Seeds for `fit_em`'s perturbed initialisations.

    A single quantile-binned start is deterministic but not reliable: at some
    state counts it converges to a worse optimum than a smaller model reaches,
    which cannot happen if both are at their global optimum. Several starts
    per fit and keeping the best log-likelihood is what makes the state-count
    comparison meaningful at all.
    """
    rng = np.random.default_rng(seed)
    return [seed] + [int(s) for s in rng.integers(1, 2 ** 31 - 1, n_starts - 1)]


def fit_em_multistart(
    sequences: list[dict],
    n_states: int,
    seed: int = 42,
    n_starts: int = 4,
    max_iter: int = 150,
    tol: float = 1e-5,
    jitter: float = 0.35,
) -> tuple[CTHMMGaussian, list[dict]]:
    """Best of several EM runs, with a record of every start.

    Start 0 is the deterministic quantile initialisation; later starts perturb
    the emission means by N(0, `jitter`) on the standardised scale. Returns the
    best fit by observed-data log-likelihood together with one record per
    start, so the spread across starts can be reported rather than assumed
    away.
    """
    records, best = [], None
    for k, s in enumerate(dispersed_starts(n_states, n_starts, seed)):
        fit = _fit_em_jittered(sequences, n_states, s, max_iter, tol,
                               jitter=(0.0 if k == 0 else jitter))
        records.append({"start": k, "seed": s, "loglik": fit.loglik,
                        "n_iter": fit.n_iter, "converged": fit.converged})
        if best is None or fit.loglik > best.loglik:
            best = fit

    lls = np.array([r["loglik"] for r in records])
    spread = float(lls.max() - lls.min()) if len(lls) > 1 else 0.0
    for r in records:
        r["is_best"] = bool(r["loglik"] == lls.max())
        r["loglik_spread_across_starts"] = spread
    return best, records


def _fit_em_jittered(sequences, n_states, seed, max_iter, tol, jitter):
    """`fit_em` from a perturbed initialisation (jitter 0 reproduces it exactly)."""
    if jitter <= 0:
        return fit_em(sequences, n_states, seed=seed, max_iter=max_iter, tol=tol)

    base = _initialise(sequences, n_states, seed)
    rng = np.random.default_rng(seed)
    start = dataclasses.replace(
        base, mu=base.mu + rng.normal(0.0, jitter, base.mu.shape))
    return _fit_em_from(sequences, start, max_iter=max_iter, tol=tol)


# ---------------------------------------------------------------------------
#  Prediction
# ---------------------------------------------------------------------------
def next_visit_predictions(fit: CTHMMGaussian, sequences: list[dict]) -> pd.DataFrame:
    """One row per prediction opportunity, from filtering only.

    For a patient's visit ``k`` the row predicts visit ``k+1``:

    1. ``alpha[k]`` --- the filtered state distribution given visits ``0..k``,
       from :meth:`CTHMMGaussian.filter_forward`, which runs no backward pass;
    2. propagated over the **actual** elapsed time to visit ``k+1`` through
       ``exp(Q*dt)``;
    3. scored against visit ``k+1``'s measurements and recorded label, which
       enter only here, as the evaluation target.

    Nothing about visit ``k+1`` touches steps 1 or 2. The columns
    ``p_state_*`` are the predicted next latent-state distribution;
    ``next_obs_loglik`` is the predictive log-density of the next visit's
    measurements under it; ``p_TD``/``p_Indeterminate``/``p_PIGD`` are the
    predicted next *observed* label distribution, present only when a label
    emission has been fitted.
    """
    rows = []
    has_label_link = fit.label_emission is not None

    for seq in sequences:
        X, gaps, labels = seq["X"], seq["gaps"], seq["labels"]
        if not len(gaps):
            continue
        alpha, _ = fit.filter_forward(X, gaps)

        for k, gap in enumerate(gaps):
            nxt = fit.predict_next_state(alpha[k], float(gap))
            row = {
                "PATNO": seq["PATNO"],
                "visit_index": int(seq["visit_index"][k]),
                "n_history": k + 1,
                "current_event": seq["events"][k],
                "current_years": float(seq["times"][k]),
                "target_event": seq["events"][k + 1],
                "target_years": float(seq["times"][k + 1]),
                "interval_years": float(gap),
                "current_label": STATE_NAMES[int(labels[k]) - 1],
                "actual_next_label": STATE_NAMES[int(labels[k + 1]) - 1],
                "n_features_next": int(np.isfinite(X[k + 1]).sum()),
                "filtered_state_current": int(alpha[k].argmax()),
                "filtered_confidence_current": float(alpha[k].max()),
                "next_obs_loglik": fit.predict_next_observation_loglik(
                    nxt, X[k + 1]),
            }
            for s in range(fit.n_states):
                row[f"p_state_{s}"] = float(nxt[s])
            row["predicted_next_latent_state"] = int(_argmax_lowest_index(nxt))
            row["p_next_latent_state"] = float(nxt.max())
            row["latent_state_is_tie"] = bool(
                np.sum(nxt == nxt.max()) > 1)

            if has_label_link:
                probs = fit.predict_next_label_probs(nxt)
                for i, name in enumerate(STATE_NAMES):
                    row[f"p_{name}"] = float(probs[i])
            rows.append(row)

    return pd.DataFrame(rows)


def _argmax_lowest_index(p: np.ndarray) -> int:
    """Most probable state; an exact tie resolves to the LOWEST state index.

    States are ordered by symptom burden (see :func:`canonical_order`), so the
    lowest index is the least-affected state of those tied. That is the
    conservative choice --- it does not assert deterioration on a tie --- and
    it is deterministic. Ties are flagged in the prediction table.
    """
    return int(np.flatnonzero(p == p.max())[0])


def predictive_score(fit: CTHMMGaussian, sequences: list[dict]) -> dict:
    """Mean next-visit predictive log-density, the criterion for state count.

    This is a proper scoring rule for the latent representation: a model is
    rewarded for putting density where the next visit's measurements actually
    land. It is computed from filtering alone, so it can be used on patients
    held out of the fit. Normalised per observed feature as well as per
    prediction, because visits differ in how many features they recorded.
    """
    preds = next_visit_predictions(fit, sequences)
    if not len(preds):
        return {"n_predictions": 0, "mean_next_obs_loglik": float("nan")}
    total_features = int(preds["n_features_next"].sum())
    return {
        "n_predictions": int(len(preds)),
        "n_patients": int(preds["PATNO"].nunique()),
        "mean_next_obs_loglik": float(preds["next_obs_loglik"].mean()),
        "mean_next_obs_loglik_per_feature":
            float(preds["next_obs_loglik"].sum() / max(total_features, 1)),
    }


def patient_folds(patients, k: int, seed: int) -> dict:
    """Deterministic patient -> inner-fold assignment, balanced in size.

    Mirrors :func:`hmm_stability.patient_folds` so inner splits in this
    experiment are built the same way as elsewhere in the project.
    """
    p = np.array(sorted(patients))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(p))
    return {p[i]: int(n % k) for n, i in enumerate(order)}


# ---------------------------------------------------------------------------
#  Batched E-step: the same recursion, run over all patients at once
# ---------------------------------------------------------------------------
@dataclass
class SequenceBatch:
    """Ragged sequences packed into rectangular arrays, as `hmm.prepare_batch` does.

    The per-sequence E-step above is a Python loop over ~930 patients, and at
    eight latent states that costs ~3s per EM iteration --- which puts a
    654-fold LOPO run at the selected state count out of reach. Running the
    recursion over all patients simultaneously is the same arithmetic in a
    different order, and `test_batched_estep_matches_the_reference` asserts the
    two agree to floating point rather than taking that on trust.
    """

    X: np.ndarray            # (n_seq, max_len, n_features), padded with NaN
    visit_mask: np.ndarray   # (n_seq, max_len) bool
    gap_index: np.ndarray    # (n_seq, max_len - 1) int into unique_gaps
    step_mask: np.ndarray    # (n_seq, max_len - 1) bool
    unique_gaps: np.ndarray
    lengths: np.ndarray
    n_observations: int

    @property
    def n_sequences(self) -> int:
        return self.X.shape[0]


def prepare_batch(sequences: list[dict], gap_round_dp: int = 3) -> SequenceBatch:
    """Pack sequences and de-duplicate the inter-visit gaps."""
    if not sequences:
        raise ValueError("no sequences supplied")

    lengths = np.array([len(s["times"]) for s in sequences], dtype=int)
    max_len = int(lengths.max())
    n_features = sequences[0]["X"].shape[1]

    gaps = [np.round(s["gaps"], gap_round_dp) for s in sequences if len(s["gaps"])]
    unique_gaps = (np.unique(np.concatenate(gaps)) if gaps
                   else np.array([1.0]))
    position = {g: i for i, g in enumerate(unique_gaps)}

    X = np.full((len(sequences), max_len, n_features), np.nan)
    visit_mask = np.zeros((len(sequences), max_len), dtype=bool)
    gap_index = np.zeros((len(sequences), max(max_len - 1, 1)), dtype=int)
    step_mask = np.zeros((len(sequences), max(max_len - 1, 1)), dtype=bool)

    for i, s in enumerate(sequences):
        n = len(s["times"])
        X[i, :n] = s["X"]
        visit_mask[i, :n] = True
        for t, g in enumerate(np.round(s["gaps"], gap_round_dp)):
            gap_index[i, t] = position[g]
            step_mask[i, t] = True

    return SequenceBatch(
        X=X, visit_mask=visit_mask, gap_index=gap_index, step_mask=step_mask,
        unique_gaps=unique_gaps, lengths=lengths,
        n_observations=int(lengths.sum()))


def _batch_emission_loglik(fit: CTHMMGaussian, batch: SequenceBatch) -> np.ndarray:
    """(n_seq, max_len, n_states) emission log-density, missing features dropped."""
    X = batch.X
    observed = np.isfinite(X)
    filled = np.where(observed, X, 0.0)

    # (n_seq, max_len, n_states, n_features) is too large to materialise for a
    # wide batch, so the sum over features is done as two einsums on the
    # quadratic and constant parts instead.
    inv_var = 1.0 / fit.var                                   # (k, d)
    const = -0.5 * np.log(2 * np.pi * fit.var)                # (k, d)

    quad = (np.einsum("ntd,kd->ntk", filled ** 2, inv_var)
            - 2.0 * np.einsum("ntd,kd->ntk", filled, fit.mu * inv_var)
            + np.einsum("ntd,kd->ntk", observed.astype(float),
                        fit.mu ** 2 * inv_var))
    base = np.einsum("ntd,kd->ntk", observed.astype(float), const)
    out = base - 0.5 * quad
    return np.where(batch.visit_mask[:, :, None], out, 0.0)


def batched_estep(fit: CTHMMGaussian, batch: SequenceBatch, P_by_gap: np.ndarray):
    """E-step statistics for every sequence at once.

    Returns the total observed-data log-likelihood and the sufficient
    statistics the M-step needs: the initial-state expectation, per-state
    responsibility totals, soft-weighted first and second moments of each
    feature (over the visits that recorded it), and soft transition counts
    grouped by elapsed time.
    """
    k = fit.n_states
    n_seq, max_len = batch.visit_mask.shape
    logb = _batch_emission_loglik(fit, batch)
    shift = logb.max(axis=2, keepdims=True)
    b = np.where(batch.visit_mask[:, :, None], np.exp(logb - shift), 1.0)

    alpha = np.zeros((n_seq, max_len, k))
    scale = np.ones((n_seq, max_len))

    a = fit.pi[None, :] * b[:, 0, :]
    scale[:, 0] = np.maximum(a.sum(axis=1), _EPS)
    alpha[:, 0, :] = a / scale[:, 0, None]

    for t in range(max_len - 1):
        active = batch.step_mask[:, t]
        P = P_by_gap[batch.gap_index[:, t]]                    # (n_seq, k, k)
        a = np.einsum("nk,nkl->nl", alpha[:, t, :], P) * b[:, t + 1, :]
        s = np.maximum(a.sum(axis=1), _EPS)
        upd = a / s[:, None]
        alpha[:, t + 1, :] = np.where(active[:, None], upd, alpha[:, t, :])
        scale[:, t + 1] = np.where(active, s, 1.0)

    beta = np.zeros((n_seq, max_len, k))
    last = batch.lengths - 1
    beta[np.arange(n_seq), last, :] = 1.0

    for t in range(max_len - 2, -1, -1):
        active = batch.step_mask[:, t]
        P = P_by_gap[batch.gap_index[:, t]]
        nxt = b[:, t + 1, :] * beta[:, t + 1, :] / scale[:, t + 1, None]
        val = np.einsum("nkl,nl->nk", P, nxt)
        beta[:, t, :] = np.where(active[:, None], val, beta[:, t, :])

    gamma = alpha * beta
    gamma *= batch.visit_mask[:, :, None]
    gamma /= np.maximum(gamma.sum(axis=2, keepdims=True), _EPS)
    gamma = np.where(batch.visit_mask[:, :, None], gamma, 0.0)

    observed = np.isfinite(batch.X) & batch.visit_mask[:, :, None]
    filled = np.where(observed, batch.X, 0.0)

    counts_by_gap = np.zeros((len(batch.unique_gaps), k, k))
    for t in range(max_len - 1):
        active = batch.step_mask[:, t]
        if not active.any():
            break
        P = P_by_gap[batch.gap_index[:, t]]
        nxt = b[:, t + 1, :] * beta[:, t + 1, :] / scale[:, t + 1, None]
        xi = alpha[:, t, :, None] * P * nxt[:, None, :]
        total = xi.sum(axis=(1, 2))
        xi = np.where((total > 0)[:, None, None], xi / np.maximum(
            total[:, None, None], _EPS), xi)
        xi = xi * active[:, None, None]
        np.add.at(counts_by_gap, batch.gap_index[:, t], xi)

    loglik = float((np.log(np.maximum(scale, _EPS))
                    * np.where(batch.visit_mask, 1.0, 0.0)).sum()
                   + (shift[:, :, 0] * batch.visit_mask).sum())

    return {
        "loglik": loglik,
        "pi": gamma[:, 0, :].sum(axis=0),
        "gamma_sum": gamma.sum(axis=(0, 1)),
        "weight_obs": np.einsum("ntk,ntd->kd", gamma, observed.astype(float)),
        "weighted_x": np.einsum("ntk,ntd->kd", gamma, filled),
        "weighted_xx": np.einsum("ntk,ntd->kd", gamma, filled ** 2),
        "counts_by_gap": counts_by_gap,
    }
