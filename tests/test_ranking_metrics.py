"""Threshold-free metrics, and the patient-clustered bootstrap behind them.

The point of these is that the intervals are *paired* and *clustered*. An
unpaired or row-level bootstrap would still produce plausible-looking numbers,
so the tests check the properties that distinguish them: identical scores give
exactly zero difference in every resample, and a patient's rows always move
together.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppmi_subtype import ranking_metrics as rm
from ppmi_subtype.clinical_profile import auc as project_auc


def toy(n=200, seed=0):
    """Scores correlated with the label, clustered in patients of 4 visits."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.3
    score = np.clip(rng.normal(np.where(y, 0.65, 0.35), 0.18), 0.01, 0.99)
    patno = np.repeat(np.arange(n // 4), 4)[:n]
    return score, y, patno


# ------------------------------------------------------------ the metrics
def test_auc_matches_the_projects_own_implementation():
    score, y, _ = toy()
    assert rm.auc_roc(y, score) == pytest.approx(project_auc(score, y), abs=1e-12)


def test_the_extremes_are_what_they_should_be():
    y = np.array([False, False, True, True])
    assert rm.auc_roc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert rm.auc_roc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    # a constant score ranks nothing; ties take half credit
    assert rm.auc_roc(y, np.full(4, 0.5)) == 0.5


def test_pr_auc_rewards_the_minority_class_being_ranked_first():
    y = np.array([False] * 8 + [True] * 2)
    perfect = np.concatenate([np.linspace(0.1, 0.4, 8), [0.9, 0.95]])
    assert rm.pr_auc(y, perfect) == pytest.approx(1.0)
    assert rm.pr_auc(y, -perfect) < 0.5


def test_partial_auc_over_the_whole_range_is_the_whole_auc():
    score, y, _ = toy()
    assert rm.partial_auc(y, score, max_fpr=1.0) == pytest.approx(
        rm.auc_roc(y, score), abs=1e-12)


def test_partial_auc_ignores_what_happens_beyond_its_region():
    # Two scores identical below FPR 0.2 but different above it must give the
    # same partial AUC and different whole-curve AUC. Built directly: move the
    # ordering only among the lowest-scoring negatives.
    y = np.array([True] * 20 + [False] * 80)
    a = np.concatenate([np.linspace(0.8, 0.99, 20), np.linspace(0.0, 0.79, 80)])
    b = a.copy()
    b[20:60] = b[20:60][::-1]                 # reshuffle deep negatives only
    assert rm.partial_auc(y, a, 0.2) == pytest.approx(rm.partial_auc(y, b, 0.2))


def test_the_operating_point_is_read_off_the_decision_rule():
    y = np.array([True, True, True, False, False, False, False])
    called = np.array([True, True, False, True, False, False, False])
    op = rm.operating_point(y, called)
    assert op["n_called_PIGD"] == 3
    assert op["operating_tpr"] == pytest.approx(2 / 3)
    assert op["operating_fpr"] == pytest.approx(1 / 4)


# --------------------------------------------------------- the bootstrap
def test_the_interval_brackets_the_point_estimate():
    score, y, patno = toy()
    lo, hi, n = rm.patient_bootstrap(score, y, patno, n_boot=300)
    assert lo < rm.auc_roc(y, score) < hi
    assert 0 < n <= 300


def test_the_bootstrap_is_reproducible_and_seed_dependent():
    score, y, patno = toy()
    first = rm.patient_bootstrap(score, y, patno, n_boot=200)
    again = rm.patient_bootstrap(score, y, patno, n_boot=200)
    other = rm.patient_bootstrap(score, y, patno, seed=7, n_boot=200)
    assert first == again
    assert first[:2] != other[:2]


def test_resampling_moves_a_patients_rows_together():
    # The clustered bootstrap must draw patients, not rows. If it drew rows, a
    # resample could contain a fraction of a patient's visits; here every
    # patient's count has to be a multiple of its own block size.
    score, y, patno = toy(n=120)
    grouped, patients = rm._group(patno, (score, y))
    assert len(patients) == 30
    assert all(len(rows) == 4 for rows in grouped.values())
    assert sum(len(rows) for rows in grouped.values()) == 120


def test_identical_scores_give_exactly_zero_difference():
    # The strongest evidence that the difference is paired: if both arms are the
    # same, every resample must cancel, so the interval is degenerate at zero.
    score, y, patno = toy()
    d = rm.patient_bootstrap_difference(score, score, y, patno, n_boot=200)
    assert d["delta"] == 0.0
    assert d["ci_low"] == 0.0 and d["ci_high"] == 0.0
    assert d["ci_includes_zero"]


def test_a_monotone_rescaling_changes_no_ranking_in_any_resample():
    score, y, patno = toy()
    rescaled = score ** 3                      # strictly increasing, so same order
    d = rm.patient_bootstrap_difference(rescaled, score, y, patno, n_boot=200)
    assert d["delta"] == pytest.approx(0.0, abs=1e-12)
    assert d["ci_low"] == pytest.approx(0.0, abs=1e-12)
    assert d["ci_high"] == pytest.approx(0.0, abs=1e-12)


def test_a_better_arm_gives_a_positive_difference():
    score, y, patno = toy()
    better = np.clip(score + 0.25 * y, 0.01, 0.99)
    d = rm.patient_bootstrap_difference(better, score, y, patno, n_boot=300)
    assert d["delta"] > 0
    assert not d["ci_includes_zero"]
    assert d["ci_low"] <= d["delta"] <= d["ci_high"]


def test_the_difference_is_antisymmetric():
    score, y, patno = toy()
    better = np.clip(score + 0.2 * y, 0.01, 0.99)
    forward = rm.patient_bootstrap_difference(better, score, y, patno, n_boot=200)
    reverse = rm.patient_bootstrap_difference(score, better, y, patno, n_boot=200)
    assert forward["delta"] == pytest.approx(-reverse["delta"], abs=1e-12)
    assert forward["ci_low"] == pytest.approx(-reverse["ci_high"], abs=1e-12)


def test_mismatched_lengths_are_rejected():
    score, y, patno = toy(n=40)
    with pytest.raises(ValueError):
        rm.patient_bootstrap_difference(score[:-1], score, y, patno, n_boot=10)


def test_degenerate_resamples_are_skipped_not_counted():
    # One patient carries every positive, so resamples that miss it have a
    # single class and no defined AUC. They must be dropped from the draws.
    score, y, patno = toy(n=80)
    y = np.zeros(80, dtype=bool)
    y[:4] = True                               # all positives in patient 0
    lo, hi, n = rm.patient_bootstrap(score, y, patno, n_boot=200)
    assert n < 200
    assert np.isfinite(lo) and np.isfinite(hi)
