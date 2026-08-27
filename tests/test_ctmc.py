"""Tests for the continuous-time Markov model and the patient-level split.

The load-bearing test here is `test_recovers_a_known_generator`: simulate from
a generator we choose, refit, and check the parameters come back. A likelihood
that is subtly wrong will still converge and still produce plausible-looking
numbers, so agreement with the truth on synthetic data is the only real check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import expm

from ppmi_subtype import ctmc, splitting

STATES = [1.0, 2.0, 3.0]
NAMES = ["TD", "Ind", "PIGD"]


def simulate_panel(Q, n_patients, n_visits, seed=0, mean_gap=1.0):
    """Draw patient trajectories from a known generator."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        s = rng.choice(3, p=[0.72, 0.10, 0.18])
        for _ in range(n_visits):
            dt = max(0.25, rng.gamma(shape=8, scale=mean_gap / 8))
            nxt = rng.choice(3, p=expm(Q * dt)[s])
            rows.append(
                {"PATNO": p, "from_state": STATES[s], "to_state": STATES[nxt],
                 "interval_years": dt}
            )
            s = nxt
    return pd.DataFrame(rows)


class TestGenerator:
    def test_rows_sum_to_zero(self):
        Q = ctmc.build_generator(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        assert np.allclose(Q.sum(axis=1), 0.0)

    def test_off_diagonals_placed_row_major(self):
        Q = ctmc.build_generator(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        assert Q[0, 1] == 1.0 and Q[0, 2] == 2.0
        assert Q[1, 0] == 3.0 and Q[1, 2] == 4.0
        assert Q[2, 0] == 5.0 and Q[2, 1] == 6.0

    def test_diagonal_is_negative_row_sum(self):
        Q = ctmc.build_generator(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        assert Q[0, 0] == pytest.approx(-0.3)


class TestTransitionProbabilities:
    def test_rows_are_a_distribution(self):
        Q = ctmc.build_generator(np.array([0.1, 0.05, 0.4, 0.3, 0.08, 0.12]))
        P = ctmc.transition_probability(Q, 2.5)
        assert np.allclose(P.sum(axis=1), 1.0)
        assert (P >= 0).all()

    def test_zero_time_is_the_identity(self):
        Q = ctmc.build_generator(np.array([0.1, 0.05, 0.4, 0.3, 0.08, 0.12]))
        assert np.allclose(ctmc.transition_probability(Q, 0.0), np.eye(3))

    def test_batched_matches_matrix_exponential(self):
        """The eigendecomposition fast path must equal expm exactly."""
        Q = ctmc.build_generator(np.array([0.08, 0.11, 0.44, 0.37, 0.18, 0.11]))
        times = np.array([0.25, 1.0, 2.5, 7.0, 13.0])
        fast = ctmc.transition_probabilities(Q, times)
        slow = np.stack([expm(Q * t) for t in times])
        assert np.abs(fast - slow).max() < 1e-10

    def test_chapman_kolmogorov(self):
        """P(s + t) must equal P(s) P(t); this is what a generator guarantees."""
        Q = ctmc.build_generator(np.array([0.1, 0.05, 0.4, 0.3, 0.08, 0.12]))
        left = ctmc.transition_probability(Q, 3.0)
        right = ctmc.transition_probability(Q, 1.2) @ ctmc.transition_probability(Q, 1.8)
        assert np.allclose(left, right)


class TestFitting:
    def test_recovers_a_known_generator(self):
        """Simulate from a chosen Q, refit, and compare. The real check."""
        true_rates = np.array([0.06, 0.10, 0.50, 0.40, 0.10, 0.08])
        Q_true = ctmc.build_generator(true_rates)
        panel = simulate_panel(Q_true, n_patients=2500, n_visits=6, seed=11)

        fit = ctmc.fit_ctmc(panel, STATES, NAMES, seed=11)
        off = ~np.eye(3, dtype=bool)
        relative_error = np.abs(fit.Q[off] - Q_true[off]) / Q_true[off]

        assert fit.converged
        assert relative_error.max() < 0.25, f"worst rate off by {relative_error.max():.1%}"

    def test_loglik_is_higher_at_the_optimum_than_nearby(self):
        Q_true = ctmc.build_generator(np.array([0.06, 0.10, 0.50, 0.40, 0.10, 0.08]))
        panel = simulate_panel(Q_true, n_patients=400, n_visits=4, seed=3)
        fit = ctmc.fit_ctmc(panel, STATES, NAMES, seed=3)

        i, j, uniq, inverse = ctmc._pair_arrays(panel, STATES)
        best = -ctmc._neg_loglik(np.log(fit.Q[~np.eye(3, dtype=bool)]), i, j, uniq, inverse, 3)
        rng = np.random.default_rng(0)
        for _ in range(5):
            perturbed = np.log(fit.Q[~np.eye(3, dtype=bool)]) + rng.normal(0, 0.3, 6)
            assert -ctmc._neg_loglik(perturbed, i, j, uniq, inverse, 3) <= best + 1e-6

    def test_rejects_non_positive_intervals(self):
        panel = pd.DataFrame(
            {"PATNO": [1], "from_state": [1.0], "to_state": [2.0], "interval_years": [0.0]}
        )
        with pytest.raises(ValueError, match="strictly positive"):
            ctmc.fit_ctmc(panel, STATES, NAMES)

    def test_rejects_unknown_state(self):
        panel = pd.DataFrame(
            {"PATNO": [1], "from_state": [9.0], "to_state": [2.0], "interval_years": [1.0]}
        )
        with pytest.raises(ValueError, match="not present"):
            ctmc.fit_ctmc(panel, STATES, NAMES)


class TestDerivedQuantities:
    def test_mean_sojourn_is_reciprocal_exit_rate(self):
        Q = ctmc.build_generator(np.array([0.1, 0.1, 0.3, 0.3, 0.2, 0.2]))
        fit = ctmc.CTMCFit(Q=Q, states=STATES, state_names=NAMES, loglik=0.0,
                           n_pairs=1, n_patients=1, converged=True)
        assert fit.mean_sojourn()["TD"] == pytest.approx(1 / 0.2)
        assert fit.mean_sojourn()["Ind"] == pytest.approx(1 / 0.6)

    def test_stationary_distribution_is_invariant(self):
        Q = ctmc.build_generator(np.array([0.1, 0.05, 0.4, 0.3, 0.08, 0.12]))
        fit = ctmc.CTMCFit(Q=Q, states=STATES, state_names=NAMES, loglik=0.0,
                           n_pairs=1, n_patients=1, converged=True)
        pi = fit.stationary_distribution().to_numpy()
        assert pi.sum() == pytest.approx(1.0)
        assert np.allclose(pi @ ctmc.transition_probability(Q, 5.0), pi, atol=1e-8)

    def test_clinical_table_rows_sum_to_one(self):
        Q = ctmc.build_generator(np.array([0.1, 0.05, 0.4, 0.3, 0.08, 0.12]))
        fit = ctmc.CTMCFit(Q=Q, states=STATES, state_names=NAMES, loglik=0.0,
                           n_pairs=1, n_patients=1, converged=True)
        table = ctmc.clinical_table(fit, horizons=(1, 5))
        assert np.allclose(table[NAMES].sum(axis=1), 1.0)

    def test_horizon_zero_is_certainty(self):
        Q = ctmc.build_generator(np.array([0.1, 0.05, 0.4, 0.3, 0.08, 0.12]))
        fit = ctmc.CTMCFit(Q=Q, states=STATES, state_names=NAMES, loglik=0.0,
                           n_pairs=1, n_patients=1, converged=True)
        row = ctmc.clinical_table(fit, horizons=(1,)).query("horizon_years == 0 and current_state == 'TD'")
        assert row["TD"].iloc[0] == pytest.approx(1.0)


class TestBaselines:
    def test_dtmc_rows_sum_to_one(self):
        Q = ctmc.build_generator(np.array([0.06, 0.10, 0.50, 0.40, 0.10, 0.08]))
        panel = simulate_panel(Q, 200, 3, seed=5)
        assert np.allclose(ctmc.fit_dtmc(panel, STATES).sum(axis=1), 1.0)

    def test_dtmc_smoothing_keeps_unseen_transitions_finite(self):
        """A transition absent from training must not give -inf on test."""
        panel = pd.DataFrame(
            {"PATNO": [1, 1], "from_state": [1.0, 1.0], "to_state": [1.0, 1.0],
             "interval_years": [1.0, 1.0]}
        )
        P = ctmc.fit_dtmc(panel, STATES)
        assert (P > 0).all()

    def test_marginal_ignores_current_state(self):
        Q = ctmc.build_generator(np.array([0.06, 0.10, 0.50, 0.40, 0.10, 0.08]))
        panel = simulate_panel(Q, 100, 3, seed=6)
        M = ctmc.fit_marginal(panel, STATES)
        assert np.allclose(M[0], M[1]) and np.allclose(M[1], M[2])


class TestEvaluation:
    def test_perfect_model_beats_uninformative_one(self):
        Q = ctmc.build_generator(np.array([0.06, 0.10, 0.50, 0.40, 0.10, 0.08]))
        panel = simulate_panel(Q, 300, 4, seed=8)
        good = ctmc.evaluate(ctmc.fit_dtmc(panel, STATES), panel, STATES, "dtmc")
        null = ctmc.evaluate(ctmc.fit_marginal(panel, STATES), panel, STATES, "null")
        assert good["loglik_per_pair"] > null["loglik_per_pair"]
        assert good["brier"] < null["brier"]

    def test_brier_is_zero_for_a_certain_correct_prediction(self):
        panel = pd.DataFrame(
            {"PATNO": [1], "from_state": [1.0], "to_state": [1.0], "interval_years": [1.0]}
        )
        certain = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        assert ctmc.evaluate(certain, panel, STATES, "x")["brier"] == pytest.approx(0.0)


class TestSplitting:
    @pytest.fixture
    def panel(self):
        rows = []
        for p in range(120):
            state = 1.0 if p % 3 else 3.0
            for v in range(3):
                rows.append({"PATNO": p, "subtype": state, "years_from_baseline": float(v)})
        return pd.DataFrame(rows)

    def test_every_patient_assigned_exactly_once(self, panel):
        man = splitting.split_patients(panel, 0.7, seed=1)
        assert len(man) == panel.PATNO.nunique()
        assert man["PATNO"].is_unique

    def test_no_patient_appears_on_both_sides(self, panel):
        man = splitting.split_patients(panel, 0.7, seed=1)
        train = set(man.loc[man.split == "train", "PATNO"])
        test = set(man.loc[man.split == "test", "PATNO"])
        assert train & test == set()

    def test_split_is_reproducible_from_the_seed(self, panel):
        a = splitting.split_patients(panel, 0.7, seed=42)
        b = splitting.split_patients(panel, 0.7, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seeds_give_different_splits(self, panel):
        a = splitting.split_patients(panel, 0.7, seed=1)
        b = splitting.split_patients(panel, 0.7, seed=2)
        assert not a.equals(b)

    def test_stratification_balances_baseline_subtype(self, panel):
        man = splitting.split_patients(panel, 0.7, seed=3)
        prop = man.groupby("split")["baseline_subtype"].apply(lambda s: (s == 1.0).mean())
        assert abs(prop["train"] - prop["test"]) < 0.05

    def test_rejects_invalid_fraction(self, panel):
        with pytest.raises(ValueError, match="must lie in"):
            splitting.split_patients(panel, 1.5, seed=1)

    def test_apply_split_selects_only_that_side(self, panel):
        man = splitting.split_patients(panel, 0.7, seed=1)
        pairs = pd.DataFrame({"PATNO": panel.PATNO.unique()})
        train = splitting.apply_split(pairs, man, "train")
        assert set(train.PATNO) == set(man.loc[man.split == "train", "PATNO"])
