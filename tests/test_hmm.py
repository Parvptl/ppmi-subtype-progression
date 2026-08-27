"""Tests for the misclassification hidden Markov model.

Two tests carry the weight here. `test_recovers_latent_generator_and_emission`
simulates a latent chain, corrupts the observations with a known error matrix,
and checks both are recovered. `test_zero_noise_reduces_to_the_plain_chain`
checks the model degenerates correctly: with no misclassification the HMM must
agree with the ordinary continuous-time chain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import expm

from ppmi_subtype import ctmc, hmm

STATES = [1.0, 2.0, 3.0]
NAMES = ["TD", "Ind", "PIGD"]


def simulate_sequences(Q, E, n_patients, n_visits, seed=0, mean_gap=1.0):
    """Latent CTMC path, observed through the misclassification matrix E."""
    rng = np.random.default_rng(seed)
    sequences = []
    for p in range(n_patients):
        latent = rng.choice(3, p=[0.72, 0.10, 0.18])
        labels = [rng.choice(3, p=E[latent])]
        gaps = []
        for _ in range(n_visits - 1):
            dt = max(0.25, rng.gamma(shape=8, scale=mean_gap / 8))
            latent = rng.choice(3, p=expm(Q * dt)[latent])
            labels.append(rng.choice(3, p=E[latent]))
            gaps.append(dt)
        sequences.append(
            {"PATNO": p, "labels": np.array(labels), "gaps": np.array(gaps)}
        )
    return sequences


class TestSequenceExtraction:
    def test_gaps_are_differences_of_visit_times(self):
        panel = pd.DataFrame(
            {"PATNO": [1, 1, 1], "subtype": [1.0, 1.0, 3.0],
             "years_from_baseline": [0.0, 1.2, 3.0]}
        )
        seq = hmm.build_sequences(panel, STATES)[0]
        assert list(seq["labels"]) == [0, 0, 2]
        assert np.allclose(seq["gaps"], [1.2, 1.8])

    def test_unlabelled_visits_are_dropped_and_gaps_bridge(self):
        panel = pd.DataFrame(
            {"PATNO": [1, 1, 1], "subtype": [1.0, np.nan, 3.0],
             "years_from_baseline": [0.0, 1.0, 2.0]}
        )
        seq = hmm.build_sequences(panel, STATES)[0]
        assert list(seq["labels"]) == [0, 2]
        assert np.allclose(seq["gaps"], [2.0])

    def test_single_visit_patients_excluded(self):
        panel = pd.DataFrame(
            {"PATNO": [1, 2, 2], "subtype": [1.0, 1.0, 3.0],
             "years_from_baseline": [0.0, 0.0, 1.0]}
        )
        assert [s["PATNO"] for s in hmm.build_sequences(panel, STATES)] == [2]


class TestBatching:
    def test_padding_preserves_each_sequence_length(self):
        seqs = [
            {"PATNO": 1, "labels": np.array([0, 1, 2]), "gaps": np.array([1.0, 1.0])},
            {"PATNO": 2, "labels": np.array([0, 0]), "gaps": np.array([2.0])},
        ]
        batch = hmm.prepare_batch(seqs)
        assert batch.labels.shape == (2, 3)
        assert batch.step_mask[0].tolist() == [True, True]
        assert batch.step_mask[1].tolist() == [True, False]

    def test_unique_gaps_deduplicated(self):
        seqs = [
            {"PATNO": 1, "labels": np.array([0, 1, 2]), "gaps": np.array([1.0, 1.0])},
            {"PATNO": 2, "labels": np.array([0, 0]), "gaps": np.array([1.0])},
        ]
        assert len(hmm.prepare_batch(seqs).unique_gaps) == 1


class TestEmissionFloor:
    def test_bound_produces_the_requested_diagonal(self):
        from scipy.special import softmax

        for floor in (0.4, 0.5, 0.7, 0.9):
            b = hmm._emission_logit_bound(floor, 3)
            diag = softmax(np.array([0.0, b, b]))[0]
            assert diag == pytest.approx(floor, rel=1e-9)

    def test_uniform_floor_gives_no_constraint(self):
        assert hmm._emission_logit_bound(1 / 3, 3) == pytest.approx(0.0)

    def test_rejects_impossible_floor(self):
        with pytest.raises(ValueError):
            hmm._emission_logit_bound(1.5, 3)

    def test_fitted_diagonal_respects_the_floor(self):
        Q = ctmc.build_generator(np.array([0.1, 0.02, 0.3, 0.3, 0.02, 0.1]))
        E = np.array([[0.9, 0.07, 0.03], [0.2, 0.6, 0.2], [0.05, 0.1, 0.85]])
        seqs = simulate_sequences(Q, E, 250, 4, seed=2)
        fit = hmm.fit_hmm(seqs, STATES, NAMES, seed=2, n_restarts=1, emission_floor=0.6)
        assert (np.diag(fit.E) >= 0.6 - 1e-6).all()


class TestFitting:
    def test_recovers_latent_generator_and_emission(self):
        """Simulate a latent chain seen through known noise; recover both."""
        Q_true = ctmc.build_generator(np.array([0.10, 0.01, 0.35, 0.35, 0.01, 0.10]))
        E_true = np.array(
            [[0.90, 0.07, 0.03], [0.15, 0.70, 0.15], [0.03, 0.07, 0.90]]
        )
        seqs = simulate_sequences(Q_true, E_true, n_patients=2500, n_visits=8, seed=17)
        fit = hmm.fit_hmm(seqs, STATES, NAMES, seed=17, n_restarts=3, emission_floor=0.5)

        assert fit.converged
        # Emission diagonal is the quantity the model is really estimating.
        assert np.abs(np.diag(fit.E) - np.diag(E_true)).max() < 0.15
        # Latent persistence should be recovered to the right order.
        exit_true = -np.diag(Q_true)
        exit_fit = -np.diag(fit.Q)
        assert np.abs(np.log(exit_fit / exit_true)).max() < 0.8

    def test_zero_noise_reduces_to_the_plain_chain(self):
        """With no misclassification the HMM must match the ordinary CTMC."""
        Q_true = ctmc.build_generator(np.array([0.12, 0.03, 0.30, 0.30, 0.03, 0.12]))
        E_clean = np.eye(3)
        seqs = simulate_sequences(Q_true, E_clean, n_patients=1200, n_visits=6, seed=21)

        fit = hmm.fit_hmm(seqs, STATES, NAMES, seed=21, n_restarts=2, emission_floor=0.9)

        pairs = []
        for s in seqs:
            for k, gap in enumerate(s["gaps"]):
                pairs.append(
                    {"PATNO": s["PATNO"], "from_state": STATES[s["labels"][k]],
                     "to_state": STATES[s["labels"][k + 1]], "interval_years": gap}
                )
        plain = ctmc.fit_ctmc(pd.DataFrame(pairs), STATES, NAMES, seed=21)

        off = ~np.eye(3, dtype=bool)
        assert np.abs(fit.Q[off] - plain.Q[off]).max() < 0.15

    def test_loglik_improves_when_noise_is_allowed(self):
        """A model permitted to admit error must fit noisy data at least as well."""
        Q = ctmc.build_generator(np.array([0.10, 0.01, 0.35, 0.35, 0.01, 0.10]))
        E = np.array([[0.80, 0.14, 0.06], [0.25, 0.50, 0.25], [0.06, 0.14, 0.80]])
        seqs = simulate_sequences(Q, E, 600, 5, seed=31)
        loose = hmm.fit_hmm(seqs, STATES, NAMES, seed=31, n_restarts=2, emission_floor=0.4)
        tight = hmm.fit_hmm(seqs, STATES, NAMES, seed=31, n_restarts=2, emission_floor=0.95)
        assert loose.loglik >= tight.loglik - 1e-6

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="no sequences"):
            hmm.fit_hmm([], STATES, NAMES)


class TestPrediction:
    @pytest.fixture
    def fitted(self):
        Q = ctmc.build_generator(np.array([0.10, 0.02, 0.30, 0.30, 0.02, 0.10]))
        E = np.array([[0.9, 0.07, 0.03], [0.2, 0.6, 0.2], [0.03, 0.07, 0.9]])
        seqs = simulate_sequences(Q, E, 300, 5, seed=41)
        return hmm.fit_hmm(seqs, STATES, NAMES, seed=41, n_restarts=1), seqs

    def test_predictions_are_distributions(self, fitted):
        fit, seqs = fitted
        preds = hmm.predictive_distributions(fit, seqs)
        cols = [f"p_{n}" for n in NAMES]
        assert np.allclose(preds[cols].sum(axis=1), 1.0)
        assert (preds[cols] >= 0).all().all()

    def test_one_prediction_per_transition(self, fitted):
        fit, seqs = fitted
        expected = sum(len(s["gaps"]) for s in seqs)
        assert len(hmm.predictive_distributions(fit, seqs)) == expected

    def test_beats_the_marginal_null_on_its_own_data(self, fitted):
        fit, seqs = fitted
        got = hmm.evaluate(fit, seqs)
        pairs = pd.DataFrame(
            [
                {"PATNO": s["PATNO"], "from_state": STATES[s["labels"][k]],
                 "to_state": STATES[s["labels"][k + 1]], "interval_years": g}
                for s in seqs
                for k, g in enumerate(s["gaps"])
            ]
        )
        null = ctmc.evaluate(ctmc.fit_marginal(pairs, STATES), pairs, STATES, "null")
        assert got["loglik_per_pair"] > null["loglik_per_pair"]

    def test_clinical_table_rows_sum_to_one(self, fitted):
        fit, _ = fitted
        table = hmm.clinical_table(fit, horizons=(1, 3, 5))
        assert np.allclose(table[NAMES].sum(axis=1), 1.0)
