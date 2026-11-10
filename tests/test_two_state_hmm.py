"""Tests for the two-hidden-state CT-HMM variants.

Four carry the weight. `test_recovers_a_known_2x3_model` simulates a latent
two-state chain whose states both emit all three labels and checks the
generator and the emission matrix come back, which is the only real evidence
that the 2 x 3 configuration is identified at all. `TestBinaryView` checks that
excluding Indeterminate widens the gap between the retained visits rather than
treating them as consecutive, which is the preprocessing the whole Model 1
result rests on. `TestMetrics` pins the macro-F1 convention: a class present in
the data but never predicted scores zero, not NaN, because the three-label
models never predict Indeterminate and the other convention would report a
macro F1 above the accuracy. `TestNoLeakage` checks the fold construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import expm

from ppmi_subtype import hmm, hmm_stability, loo, next_visit as nv
from ppmi_subtype import two_state_hmm as ts

STATES_THREE = [1.0, 2.0, 3.0]
STATES_TWO = [1.0, 3.0]


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def simulate(Q, E, n_patients, n_visits, seed=0, mean_gap=1.0, pi=None):
    """A latent CTMC path observed through E, with irregular visit gaps."""
    rng = np.random.default_rng(seed)
    n_hidden, n_obs = E.shape
    pi = np.full(n_hidden, 1.0 / n_hidden) if pi is None else np.asarray(pi)
    sequences = []
    for p in range(n_patients):
        latent = rng.choice(n_hidden, p=pi)
        labels = [rng.choice(n_obs, p=E[latent])]
        gaps = []
        for _ in range(n_visits - 1):
            dt = max(0.25, rng.gamma(shape=8, scale=mean_gap / 8))
            latent = rng.choice(n_hidden, p=expm(Q * dt)[latent])
            labels.append(rng.choice(n_obs, p=E[latent]))
            gaps.append(dt)
        sequences.append({"PATNO": p, "labels": np.array(labels),
                          "gaps": np.array(gaps)})
    return sequences


def panel_from(rows):
    """A minimal labelled panel: (PATNO, subtype code, years_from_baseline)."""
    df = pd.DataFrame(rows, columns=["PATNO", "subtype", "years_from_baseline"])
    df["EVENT_ID"] = ["V" + str(k) for k in range(len(df))]
    return df


def prepared(rows):
    return ts.prepare(loo.prepare(panel_from(rows), STATES_THREE))


# ---------------------------------------------------------------------------
#  Parameterisation
# ---------------------------------------------------------------------------
class TestParameterCounts:
    def test_2x2_has_five_free_parameters(self):
        assert hmm._n_params(2, 2) == 5          # 2 rates + 2 emission + 1 initial

    def test_2x3_has_seven_free_parameters(self):
        assert hmm._n_params(2, 3) == 7          # 2 rates + 4 emission + 1 initial

    def test_the_existing_three_state_model_has_fourteen(self):
        assert hmm._n_params(3, 3) == 14         # 6 rates + 6 emission + 2 initial

    @pytest.mark.parametrize("n_hidden,n_obs", [(2, 2), (2, 3), (3, 3)])
    def test_unpack_returns_valid_distributions(self, n_hidden, n_obs):
        rng = np.random.default_rng(0)
        theta = rng.normal(0.0, 1.5, size=hmm._n_params(n_hidden, n_obs))
        Q, E, pi = hmm._unpack(theta, n_hidden, n_obs)

        off = ~np.eye(n_hidden, dtype=bool)
        assert (Q[off] >= 0).all()
        assert np.allclose(np.diag(Q), -(Q * off).sum(axis=1))
        assert E.shape == (n_hidden, n_obs)
        assert np.allclose(E.sum(axis=1), 1.0)
        assert (E >= 0).all() and (E <= 1).all()
        assert np.isclose(pi.sum(), 1.0)

    def test_2x3_bounds_require_no_emission_floor(self):
        # The floor constrains P(label i | state i); with two states and three
        # labels there is no such pairing, so asking for one must fail loudly
        # rather than quietly pick a pairing.
        with pytest.raises(ValueError, match="emission floor"):
            hmm_stability.bounds_for(2, 3, 0.5)
        assert len(hmm_stability.bounds_for(2, 3, None)) == 7

    def test_fit_hmm_refuses_fewer_hidden_than_observed(self):
        # Documents why this module calls fit_multistart instead: the audited
        # entry point rejects Model 2's configuration outright.
        with pytest.raises(ValueError, match="at least the number of observed"):
            hmm.fit_hmm([{"PATNO": 1, "labels": np.array([0, 1]),
                          "gaps": np.array([1.0])}],
                        STATES_THREE, n_hidden=2, emission_floor=None)


# ---------------------------------------------------------------------------
#  Parameter recovery
# ---------------------------------------------------------------------------
class TestRecovery:
    def test_recovers_a_known_2x3_model(self):
        """Two hidden states, both emitting all three labels, recovered.

        This is the test that matters for Model 2: the hidden states are
        exchangeable and nothing pins Indeterminate to either of them, so if
        the configuration were unidentified the fit could not come back to the
        truth from a dispersed start.
        """
        Q = np.array([[-0.20, 0.20], [0.05, -0.05]])
        E = np.array([[0.80, 0.14, 0.06],
                      [0.10, 0.20, 0.70]])
        sequences = simulate(Q, E, n_patients=700, n_visits=8, seed=11,
                             pi=[0.7, 0.3])

        starts = hmm_stability.dispersed_starts(2, 3, None, 12, seed=3)
        fit, _ = hmm_stability.fit_multistart(
            sequences, STATES_THREE, n_hidden=2, emission_floor=None,
            starts=starts, state_names=ts.THREE_LABELS)
        Qh, Eh, _ = ts.canonical_triple(fit.Q, fit.E, fit.pi, None)

        assert np.allclose(Eh, E, atol=0.05)
        assert np.allclose(Qh, Q, atol=0.05)

    def test_recovers_a_known_2x2_model(self):
        Q = np.array([[-0.15, 0.15], [0.04, -0.04]])
        E = np.array([[0.92, 0.08], [0.12, 0.88]])
        sequences = simulate(Q, E, n_patients=600, n_visits=8, seed=5,
                             pi=[0.75, 0.25])

        starts = nv.audited_starts(2, 2, 0.5, seed=42, n_restarts=4)
        fit, _ = hmm_stability.fit_multistart(
            sequences, STATES_TWO, n_hidden=2, emission_floor=0.5,
            starts=starts, state_names=ts.TWO_LABELS)

        assert np.allclose(fit.E, E, atol=0.05)
        assert np.allclose(fit.Q, Q, atol=0.05)

    def test_an_exchanged_labelling_is_the_same_fit(self):
        """Permuting the hidden states must not change any prediction."""
        Q = np.array([[-0.2, 0.2], [0.05, -0.05]])
        E = np.array([[0.8, 0.14, 0.06], [0.1, 0.2, 0.7]])
        swap = np.array([1, 0])

        seq = [{"PATNO": 1, "labels": np.array([0, 1, 2]),
                "gaps": np.array([1.0, 1.5])}]
        a = hmm.HMMFit(Q=Q, E=E, pi=np.array([0.7, 0.3]), states=STATES_THREE,
                       state_names=ts.THREE_LABELS, loglik=0.0, n_sequences=1,
                       n_observations=3, n_params=7, converged=True,
                       emission_floor=None, n_hidden=2)
        b = hmm.HMMFit(Q=Q[np.ix_(swap, swap)], E=E[swap],
                       pi=np.array([0.7, 0.3])[swap], states=STATES_THREE,
                       state_names=ts.THREE_LABELS, loglik=0.0, n_sequences=1,
                       n_observations=3, n_params=7, converged=True,
                       emission_floor=None, n_hidden=2)

        cols = [f"p_{n}" for n in ts.THREE_LABELS]
        pa = hmm.predictive_distributions(a, seq)[cols].to_numpy()
        pb = hmm.predictive_distributions(b, seq)[cols].to_numpy()
        assert np.allclose(pa, pb)


# ---------------------------------------------------------------------------
#  The Indeterminate-free view
# ---------------------------------------------------------------------------
class TestBinaryView:
    def test_indeterminate_visits_are_dropped(self):
        data = prepared([(1, 1.0, 0.0), (1, 2.0, 1.0), (1, 3.0, 2.0)])
        seq = data.binary_sequences[0]
        assert list(seq["labels"]) == [0, 1]                 # TD, PIGD
        assert list(data.binary_origin[1]) == [0, 2]

    def test_the_gap_is_the_actual_elapsed_time_not_a_visit_count(self):
        """A dropped Indeterminate visit must widen the gap, not vanish."""
        data = prepared([(1, 1.0, 0.0), (1, 2.0, 1.0), (1, 3.0, 2.0)])
        assert np.allclose(data.binary_sequences[0]["gaps"], [2.0])

    def test_several_dropped_visits_accumulate_into_one_gap(self):
        data = prepared([(1, 1.0, 0.0), (1, 2.0, 1.0), (1, 2.0, 2.0),
                         (1, 2.0, 3.1), (1, 3.0, 4.2)])
        assert np.allclose(data.binary_sequences[0]["gaps"], [4.2])

    def test_a_patient_with_one_retained_visit_is_excluded(self):
        data = prepared([(1, 1.0, 0.0), (1, 2.0, 1.0), (1, 2.0, 2.0)])
        assert data.binary_sequences == []
        assert len(data.full_sequences) == 1
        assert data.counts()["patients_excluded_from_model1"] == 1

    def test_counts_reconcile(self):
        data = prepared([(1, 1.0, 0.0), (1, 2.0, 1.0), (1, 3.0, 2.0),
                         (2, 1.0, 0.0), (2, 1.0, 1.0),
                         (3, 1.0, 0.0), (3, 2.0, 1.0), (3, 2.0, 2.0)])
        c = data.counts()
        assert c["model2_patients"] == 3 and c["model2_visits"] == 8
        assert c["model1_patients"] == 2 and c["model1_visits"] == 4
        # Three Indeterminate visits in the panel (one for patient 1, two for
        # patient 3) but FOUR visits lost to Model 1: patient 3 is left with a
        # single retained visit, so its TD visit goes too. The two counts are
        # reported separately because they are not the same number.
        assert c["indeterminate_visits_dropped"] == 3
        assert c["visits_excluded_from_model1"] == 4
        assert c["patients_excluded_from_model1"] == 1


class TestTargets:
    def test_a_target_needs_a_retained_predecessor_to_be_eligible(self):
        # Visit 0 is Indeterminate, so the TD visit at 1.0 has no TD/PIGD
        # predecessor and Model 1 cannot predict it, even though it is TD.
        data = prepared([(1, 2.0, 0.0), (1, 1.0, 1.0), (1, 3.0, 2.0)])
        tg = ts.build_targets(data).set_index("visit_index")
        assert not tg.loc[1, "model1_eligible"]
        assert tg.loc[2, "model1_eligible"]

    def test_an_indeterminate_target_is_never_model1_eligible(self):
        data = prepared([(1, 1.0, 0.0), (1, 2.0, 1.0), (1, 3.0, 2.0)])
        tg = ts.build_targets(data)
        assert not tg.loc[tg["target_index"] == ts.IND, "model1_eligible"].any()

    def test_model1_interval_is_never_shorter_than_the_three_label_interval(self):
        data = prepared([(1, 1.0, 0.0), (1, 2.0, 1.0), (1, 3.0, 2.0)])
        tg = ts.build_targets(data)
        e = tg[tg["model1_eligible"]]
        assert (e["model1_interval_years"] >= e["interval_years"] - 1e-12).all()
        # Here the Indeterminate visit at 1.0 intervenes, so Model 1's gap is
        # the full 2.0 years while the three-label models see only 1.0.
        assert np.allclose(e["model1_interval_years"], [2.0])
        assert np.allclose(e["interval_years"], [1.0])

    def test_eligible_targets_match_the_binary_transition_count(self):
        data = prepared([(1, 1.0, 0.0), (1, 2.0, 1.0), (1, 3.0, 2.0),
                         (2, 1.0, 0.0), (2, 3.0, 1.1), (2, 1.0, 2.2)])
        tg = ts.build_targets(data)
        assert int(tg["model1_eligible"].sum()) == sum(
            len(s["gaps"]) for s in data.binary_sequences)


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_a_class_never_predicted_scores_f1_zero_not_nan(self):
        """The convention the Indeterminate result depends on.

        The three-label models never predict Indeterminate. Excluding it from
        the macro average would report a macro F1 higher than the accuracy,
        which would read as a strength rather than the refusal it is.
        """
        cm = np.array([[127, 0, 10], [17, 0, 7], [26, 0, 62]])
        m = ts.metrics_from_confusion(cm, ts.THREE_LABELS)

        assert m["f1_Indeterminate"] == 0.0
        assert np.isnan(m["precision_Indeterminate"])      # 0/0 is undefined
        assert m["recall_Indeterminate"] == 0.0
        assert m["n_classes_in_macro_f1"] == 3
        assert m["macro_f1"] < m["accuracy"]
        assert m["macro_f1"] == pytest.approx(
            (m["f1_TD"] + 0.0 + m["f1_PIGD"]) / 3)

    def test_balanced_accuracy_is_the_mean_recall(self):
        cm = np.array([[90, 10], [40, 60]])
        m = ts.metrics_from_confusion(cm, ts.TWO_LABELS)
        assert m["balanced_accuracy"] == pytest.approx((0.9 + 0.6) / 2)
        assert m["accuracy"] == pytest.approx(150 / 200)

    def test_a_perfect_classifier_scores_one_everywhere(self):
        m = ts.metrics_from_confusion(np.diag([10, 10, 10]), ts.THREE_LABELS)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0
        assert m["balanced_accuracy"] == 1.0

    def test_an_absent_class_is_dropped_from_the_macro_average(self):
        # No support and never predicted: the class is not in this evaluation
        # at all, so averaging a zero for it would be inventing a failure.
        cm = np.array([[10, 0, 0], [0, 0, 0], [0, 0, 10]])
        m = ts.metrics_from_confusion(cm, ts.THREE_LABELS)
        assert np.isnan(m["f1_Indeterminate"])
        assert m["n_classes_in_macro_f1"] == 2
        assert m["macro_f1"] == 1.0

    def test_confusion_counts_match_a_crosstab(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 3, 500)
        p = rng.integers(0, 3, 500)
        cm = ts.confusion_counts(a, p, 3)
        assert cm.sum() == 500
        assert cm[1, 2] == int(((a == 1) & (p == 2)).sum())


class TestTieRules:
    def test_a_three_class_tie_resolves_to_the_worse_prognosis(self):
        probs = np.array([[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.5, 0.5, 0.0]])
        labels, ties = ts.argmax_three(probs)
        assert list(ties) == [True, True, True]
        assert list(labels) == ["PIGD", "PIGD", "Indeterminate"]

    def test_an_untied_argmax_is_untouched(self):
        probs = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
        labels, ties = ts.argmax_three(probs)
        assert list(labels) == ["TD", "PIGD"]
        assert not ties.any()

    def test_a_two_class_tie_follows_the_published_rule(self):
        assert nv.TIE_RULE == "PIGD"
        assert list(nv.predicted_state(np.array([0.5]), np.array([0.5]))) == ["PIGD"]


# ---------------------------------------------------------------------------
#  Renormalisation onto the common two classes
# ---------------------------------------------------------------------------
class TestCommonRenormalisation:
    def test_renormalisation_is_the_identity_for_a_two_label_model(self):
        """Model 1 has no Indeterminate mass, so the common task must not
        change its probabilities --- otherwise the two arms would not be
        scored on the same quantity."""
        probs = np.array([[0.7, 0.0, 0.3], [0.25, 0.0, 0.75]])
        p_td, p_pigd, defined = nv.conditional_two_class(probs)
        assert defined.all()
        assert np.allclose(p_td, [0.7, 0.25])
        assert np.allclose(p_pigd, [0.3, 0.75])

    def test_indeterminate_mass_is_divided_out_not_discarded(self):
        probs = np.array([[0.6, 0.25, 0.15]])
        p_td, p_pigd, _ = nv.conditional_two_class(probs)
        assert p_td[0] == pytest.approx(0.6 / 0.75)
        assert p_pigd[0] == pytest.approx(0.15 / 0.75)
        assert p_td[0] + p_pigd[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
#  Hidden-state ordering
# ---------------------------------------------------------------------------
class TestCanonicalOrdering:
    def test_an_unconstrained_square_model_is_reordered(self):
        """hmm_stability.canonical_order leaves a square model alone; without
        an emission floor the square model's states are still exchangeable, so
        this module has to order them itself."""
        Q = np.array([[-0.05, 0.05], [0.2, -0.2]])
        E = np.array([[0.1, 0.9], [0.95, 0.05]])          # state 0 emits PIGD
        pi = np.array([0.3, 0.7])

        Qh, Eh, pih = ts.canonical_triple(Q, E, pi, None)
        assert Eh[0, 0] > Eh[0, 1]                        # most TD-like first
        assert np.allclose(Eh, E[[1, 0]])
        assert np.allclose(pih, pi[[1, 0]])
        # unchanged by hmm_stability, which is exactly why the override exists
        assert np.allclose(hmm_stability.canonical_order(Q, E, pi)[1], E)

    def test_a_floored_model_is_left_alone(self):
        Q = np.array([[-0.05, 0.05], [0.2, -0.2]])
        E = np.array([[0.9, 0.1], [0.1, 0.9]])
        pi = np.array([0.3, 0.7])
        Qh, Eh, pih = ts.canonical_triple(Q, E, pi, 0.5)
        assert np.allclose(Eh, E) and np.allclose(Qh, Q) and np.allclose(pih, pi)


# ---------------------------------------------------------------------------
#  Bootstrap
# ---------------------------------------------------------------------------
class TestBootstrap:
    def test_draws_are_patient_level_and_reproducible(self):
        patients = list(range(40))
        a = ts.patient_draws(patients, 50, seed=7)
        b = ts.patient_draws(patients, 50, seed=7)
        assert a.shape == (50, 40)
        assert np.array_equal(a, b)
        assert a.max() < 40

    def test_the_same_seed_gives_the_same_draws_for_every_model(self):
        # What makes a paired contrast paired: both arms must be scored on the
        # same resampled patients.
        patients = [3, 1, 2]
        assert np.array_equal(ts.patient_draws(patients, 10, 1),
                              ts.patient_draws(patients, 10, 1))


# ---------------------------------------------------------------------------
#  Leakage
# ---------------------------------------------------------------------------
class TestNoLeakage:
    def test_the_held_out_patient_is_absent_from_both_views(self):
        data = prepared([(1, 1.0, 0.0), (1, 2.0, 1.0), (1, 3.0, 2.0),
                         (2, 1.0, 0.0), (2, 3.0, 1.0), (2, 1.0, 2.0),
                         (3, 3.0, 0.0), (3, 3.0, 1.0), (3, 1.0, 2.0)])
        for model in ts.SPECS:
            s = ts.spec_for(model, data, 0.5)
            train = [q for q in getattr(data, s["sequences_attr"])
                     if q["PATNO"] != 2]
            assert 2 not in {q["PATNO"] for q in train}
            loo.assert_no_leakage(2, train, pd.DataFrame({"PATNO": []}))

    def test_leakage_is_raised_when_the_patient_is_present(self):
        data = prepared([(1, 1.0, 0.0), (1, 3.0, 1.0),
                         (2, 1.0, 0.0), (2, 3.0, 1.0)])
        with pytest.raises(loo.LeakageError):
            loo.assert_no_leakage(1, data.full_sequences,
                                  pd.DataFrame({"PATNO": []}))

    def test_training_priors_exclude_the_held_out_patient(self):
        # Patient 2's targets are all PIGD; holding them out must move the
        # prior towards TD.
        rows = [(1, 1.0, 0.0), (1, 1.0, 1.0), (1, 1.0, 2.0),
                (2, 3.0, 0.0), (2, 3.0, 1.0), (2, 3.0, 2.0)]
        data = prepared(rows)
        with_both, _ = ts.training_priors(data, held_out=None)
        without_two, two = ts.training_priors(data, held_out=2)
        assert without_two[ts.TD] > with_both[ts.TD]
        assert two < 0.5                       # no PIGD target left in training
