"""Tests for the HMM stability machinery.

The important ones: start 0 reproduces `fit_hmm` exactly (so the multi-start
study contains the audited solution), bounds are identical to the fitter's, and
the prior-corrected rule is the balanced-accuracy Bayes rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppmi_subtype import hmm
from ppmi_subtype import hmm_stability as HS
from ppmi_subtype.ctmc import build_generator, transition_probabilities


def simulate(n_patients=150, seed=0):
    rng = np.random.default_rng(seed)
    Q = build_generator(np.array([0.12, 0.01, 0.30, 0.25, 0.0, 0.05]), 3)
    E = np.array([[0.85, 0.10, 0.05], [0.25, 0.55, 0.20], [0.05, 0.10, 0.85]])
    seqs = []
    for i in range(n_patients):
        n = int(rng.integers(2, 8))
        gaps = rng.gamma(8.0, 1 / 8.0, n - 1)
        x = rng.choice(3, p=[0.7, 0.1, 0.2])
        lab = [rng.choice(3, p=E[x])]
        for g in gaps:
            x = rng.choice(3, p=transition_probabilities(Q, np.array([g]))[0][x])
            lab.append(rng.choice(3, p=E[x]))
        seqs.append({"PATNO": i, "labels": np.array(lab), "gaps": gaps})
    return seqs


def test_default_start_reproduces_fit_hmm():
    seqs = simulate()
    ref = hmm.fit_hmm(seqs, [0, 1, 2], seed=1, n_restarts=1, emission_floor=0.5)
    starts = HS.dispersed_starts(3, 3, 0.5, 1, seed=0)
    fit, recs = HS.fit_multistart(seqs, [0, 1, 2], 3, 0.5, starts)
    assert fit.loglik == pytest.approx(ref.loglik, abs=1e-9)
    assert np.allclose(fit.Q, ref.Q) and np.allclose(fit.E, ref.E)
    assert len(recs) == 1


@pytest.mark.parametrize("n_hidden,floor", [(3, 0.5), (3, None), (3, 0.7), (5, None)])
def test_starts_lie_within_the_fitter_bounds(n_hidden, floor):
    b = np.array(HS.bounds_for(n_hidden, 3, floor))
    s = HS.dispersed_starts(n_hidden, 3, floor, 20, seed=3)
    assert s.shape == (20, len(b))
    assert np.all(s >= b[:, 0]) and np.all(s <= b[:, 1])
    assert np.unique(s.round(6), axis=0).shape[0] == 20       # genuinely different


def test_floor_bound_enforces_the_floor():
    for floor in (0.5, 0.6, 0.7):
        b = np.array(HS.bounds_for(3, 3, floor))
        th = HS.dispersed_starts(3, 3, floor, 1, 0)[0]
        th[6:12] = b[6:12, 1]                     # every off-diagonal at its upper bound
        _, E, _ = hmm._unpack(th, 3, 3)
        assert np.all(np.diag(E) >= floor - 1e-12)


def test_floor_is_rejected_for_rectangular_models():
    with pytest.raises(ValueError):
        HS.bounds_for(5, 3, 0.5)


def test_multistart_records_every_start_and_finds_the_best():
    seqs = simulate(seed=4)
    starts = HS.dispersed_starts(3, 3, 0.5, 5, seed=2)
    fit, recs = HS.fit_multistart(seqs, [0, 1, 2], 3, 0.5, starts)
    assert len(recs) == 5
    assert fit.loglik == pytest.approx(max(r.loglik for r in recs))
    spread = HS.solution_spread(recs, 3, 3)
    assert 0 < spread["share_within_tol"] <= 1
    assert all(np.isfinite(r.proj_grad) for r in recs)


def test_projected_gradient_ignores_blocked_components():
    b = [(0.0, 1.0), (0.0, 1.0)]
    assert HS.projected_gradient_norm(np.array([0.0, 0.5]), np.array([5.0, 0.0]), b) == 0.0
    assert HS.projected_gradient_norm(np.array([0.0, 0.5]), np.array([-5.0, 0.0]), b) == 5.0


def test_canonical_order_sorts_rectangular_states_only():
    Q = build_generator(np.arange(1, 21) / 100, 5)
    E = np.array([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.3, 0.4, 0.3],
                  [0.9, 0.05, 0.05], [0.2, 0.2, 0.6]])
    pi = np.arange(5) / 10.0
    Qc, Ec, pic = HS.canonical_order(Q, E, pi)
    assert np.all(np.diff(Ec @ np.arange(3)) >= 0)
    assert np.allclose(np.sort(pic), np.sort(pi))
    Q3 = Q[:3, :3]
    assert HS.canonical_order(Q3, np.eye(3), pi[:3])[0] is Q3


def test_prior_corrected_rule_can_choose_a_never_maximal_class():
    probs = np.array([[0.6, 0.3, 0.1], [0.5, 0.2, 0.3]])
    assert np.all(HS.argmax_rule(probs) == 0)
    out = HS.prior_corrected_rule(probs, np.array([0.7, 0.1, 0.2]))
    assert list(out) == [1, 1]          # 0.2/0.1 = 2.0 beats 0.3/0.2 = 1.5


def test_patient_folds_are_deterministic_and_balanced():
    f1 = HS.patient_folds(range(103), 10, 42)
    f2 = HS.patient_folds(range(103), 10, 42)
    assert f1 == f2
    sizes = np.bincount(list(f1.values()))
    assert sizes.max() - sizes.min() <= 1


def test_auc():
    assert HS.auc_one_vs_rest([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert HS.auc_one_vs_rest([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
