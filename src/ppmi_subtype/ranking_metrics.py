"""Threshold-free ranking metrics with patient-clustered bootstrap intervals.

Why this exists
---------------
Accuracy reads a model at one threshold. When two models disagree about where
to put their decision boundary --- which is what adding observation categories
to the CT-HMM does --- accuracy conflates a change in *ranking* with a change
in *threshold placement*. AUC-ROC separates them: it is invariant to both the
threshold and any monotone recalibration, so it measures only how well a model
ranks a PIGD next visit above a TD one.

Conventions follow ``scripts/66_roc_auc_comparison.py``, the project's first
AUC analysis, so numbers produced here are comparable with it:

* PIGD is the positive class and the score is the conditional two-class
  ``P_PIGD``, on the cases where that is defined;
* every interval resamples **patients**, not predictions --- a patient with
  fourteen visits contributes fourteen correlated rows, and a row-level
  bootstrap would understate the uncertainty;
* differences are paired: both arms are recomputed on the same resampled
  patients, so the interval reflects their within-patient correlation.

Partial AUC
-----------
:func:`partial_auc` is the McClish-standardised AUC over ``FPR <= max_fpr``,
rescaled so 0.5 is chance within the region. It answers a question the whole
curve cannot: whether a ranking gain falls where a usable decision rule
operates, or out in the region where a model is calling most TD visits PIGD.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

#: 2,000 resamples and seed 42 throughout the project.
N_BOOTSTRAP = 2000
SEED = 42


def auc_roc(positive: np.ndarray, score: np.ndarray) -> float:
    """AUC-ROC. Ties take half credit, as the project's own `auc` does."""
    return float(roc_auc_score(np.asarray(positive).astype(bool),
                               np.asarray(score, dtype=float)))


def pr_auc(positive: np.ndarray, score: np.ndarray) -> float:
    """Average precision: the area under the precision-recall curve."""
    return float(average_precision_score(np.asarray(positive).astype(bool),
                                         np.asarray(score, dtype=float)))


def partial_auc(positive: np.ndarray, score: np.ndarray,
                max_fpr: float = 0.2) -> float:
    """McClish-standardised AUC over ``FPR <= max_fpr``; 0.5 is chance there."""
    return float(roc_auc_score(np.asarray(positive).astype(bool),
                               np.asarray(score, dtype=float), max_fpr=max_fpr))


def _group(patno, columns) -> tuple[dict, np.ndarray]:
    """Group row tuples by patient, preserving first-seen order."""
    out: dict = {}
    for key, row in zip(np.asarray(patno), zip(*columns)):
        out.setdefault(key, []).append(row)
    return out, np.array(list(out.keys()), dtype=object)


def patient_bootstrap(score, positive, patno, metric=auc_roc,
                      seed: int = SEED, n_boot: int = N_BOOTSTRAP
                      ) -> tuple[float, float, int]:
    """Percentile CI for a ranking metric, resampling patients.

    Returns ``(low, high, n_draws)``. A resample in which every case has the
    same label leaves the metric undefined and is skipped, so `n_draws` can be
    below `n_boot`; it is reported rather than hidden.
    """
    score = np.asarray(score, dtype=float)
    positive = np.asarray(positive).astype(bool)
    by_patient, patients = _group(patno, (score, positive))

    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.choice(len(patients), size=len(patients), replace=True)
        rows = [r for i in pick for r in by_patient[patients[i]]]
        s = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        if y.all() or not y.any():
            continue
        draws[b] = metric(y, s)
    kept = draws[np.isfinite(draws)]
    if not len(kept):                                        # pragma: no cover
        raise ValueError("every bootstrap resample was degenerate")
    return (float(np.quantile(kept, 0.025)), float(np.quantile(kept, 0.975)),
            int(len(kept)))


def patient_bootstrap_difference(score_a, score_b, positive, patno,
                                 metric=auc_roc, seed: int = SEED,
                                 n_boot: int = N_BOOTSTRAP) -> dict:
    """Paired percentile CI for ``metric(a) - metric(b)``, resampling patients.

    Both arms are recomputed on the *same* resampled patients, which is what
    makes this a paired interval rather than the difference of two marginal
    ones.
    """
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    positive = np.asarray(positive).astype(bool)
    if not (len(score_a) == len(score_b) == len(positive) == len(patno)):
        raise ValueError("scores, labels and patients must be the same length")
    by_patient, patients = _group(patno, (score_a, score_b, positive))

    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for i in range(n_boot):
        pick = rng.choice(len(patients), size=len(patients), replace=True)
        rows = [r for k in pick for r in by_patient[patients[k]]]
        a = np.array([r[0] for r in rows])
        b = np.array([r[1] for r in rows])
        y = np.array([r[2] for r in rows])
        if y.all() or not y.any():
            continue
        draws[i] = metric(y, a) - metric(y, b)
    kept = draws[np.isfinite(draws)]
    if not len(kept):                                        # pragma: no cover
        raise ValueError("every bootstrap resample was degenerate")
    return {"delta": float(metric(positive, score_a) - metric(positive, score_b)),
            "ci_low": float(np.quantile(kept, 0.025)),
            "ci_high": float(np.quantile(kept, 0.975)),
            "n_bootstrap": int(len(kept)),
            "ci_includes_zero": bool(np.quantile(kept, 0.025) <= 0
                                     <= np.quantile(kept, 0.975))}


def operating_point(positive: np.ndarray, predicted_pigd: np.ndarray) -> dict:
    """Where a model's own decision rule sits on the ROC plane."""
    positive = np.asarray(positive).astype(bool)
    called = np.asarray(predicted_pigd).astype(bool)
    return {"n_called_PIGD": int(called.sum()),
            "operating_tpr": float(called[positive].mean()),
            "operating_fpr": float(called[~positive].mean())}
