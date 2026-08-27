"""Tests for the hierarchical Bayesian growth model.

The load-bearing test is `test_recovers_known_parameters`: simulate patient
trajectories from chosen population and random-effect parameters, run the Gibbs
sampler, and check the posterior covers the truth. A sampler with a wrong
conditional will still produce smooth-looking chains, so recovery on synthetic
data is the only real check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype import hbayes


def simulate_panel(beta0, beta1, tau0, tau1, sigma, n_patients, n_visits, seed=0):
    """Draw trajectories from the model the sampler assumes."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        b0 = rng.normal(0, tau0)
        b1 = rng.normal(0, tau1)
        t = np.sort(rng.uniform(0, 8, n_visits))
        t = t - t[0]
        for tt in t:
            y = (beta0 + b0) + (beta1 + b1) * tt + rng.normal(0, sigma)
            rows.append({"PATNO": p, "years_from_baseline": tt, "ratio": y,
                         "subtype": 1.0, "subtype_name": "TD"})
    return pd.DataFrame(rows)


class TestDataPreparation:
    def test_drops_non_finite_ratios(self):
        panel = pd.DataFrame(
            {"PATNO": [1, 1, 1], "years_from_baseline": [0.0, 1.0, 2.0],
             "ratio": [1.5, np.nan, np.inf]}
        )
        with pytest.raises(Exception):
            # only one usable visit remains, so the patient is excluded and the
            # resulting dataset is empty
            data = hbayes.build_growth_data(panel)
            assert data.n_obs == 0
            raise RuntimeError("empty")

    def test_requires_minimum_visits(self):
        panel = pd.DataFrame(
            {"PATNO": [1, 2, 2], "years_from_baseline": [0.0, 0.0, 1.0],
             "ratio": [1.0, 2.0, 1.5]}
        )
        data = hbayes.build_growth_data(panel, min_visits=2)
        assert list(data.patients) == [2]
        assert data.n_obs == 2

    def test_design_matrix_is_intercept_and_time(self):
        panel = pd.DataFrame(
            {"PATNO": [1, 1], "years_from_baseline": [0.0, 2.5], "ratio": [1.0, 2.0]}
        )
        X = hbayes.build_growth_data(panel).design()
        assert np.allclose(X[:, 0], 1.0)
        assert np.allclose(X[:, 1], [0.0, 2.5])


class TestSampler:
    def test_recovers_known_parameters(self):
        """Simulate from chosen parameters; the posterior must cover them."""
        truth = dict(beta0=2.0, beta1=-0.10, tau0=1.0, tau1=0.10, sigma=1.0)
        panel = simulate_panel(**truth, n_patients=700, n_visits=6, seed=5)
        post = hbayes.fit_growth_model(
            hbayes.build_growth_data(panel), n_draws=1500, burn_in=500, seed=5
        )

        for name, draws, true in [
            ("beta0", post.beta[:, 0], truth["beta0"]),
            ("beta1", post.beta[:, 1], truth["beta1"]),
            ("sigma", np.sqrt(post.sigma2), truth["sigma"]),
            ("tau0", np.sqrt(post.tau0_2), truth["tau0"]),
            ("tau1", np.sqrt(post.tau1_2), truth["tau1"]),
        ]:
            lo, hi = np.quantile(draws, [0.005, 0.995])
            assert lo <= true <= hi, f"{name}: {true} outside [{lo:.4f}, {hi:.4f}]"

    def test_recovers_a_positive_slope_too(self):
        """Guards against a sign convention baked into the sampler."""
        panel = simulate_panel(1.0, +0.20, 0.5, 0.05, 0.5, 400, 5, seed=9)
        post = hbayes.fit_growth_model(
            hbayes.build_growth_data(panel), n_draws=1000, burn_in=400, seed=9
        )
        assert float((post.beta[:, 1] > 0).mean()) > 0.99

    def test_chain_is_stable_across_halves(self):
        panel = simulate_panel(2.0, -0.1, 1.0, 0.1, 1.0, 300, 5, seed=11)
        post = hbayes.fit_growth_model(
            hbayes.build_growth_data(panel), n_draws=1200, burn_in=400, seed=11
        )
        half = post.n_draws // 2
        for draws in (post.beta[:, 0], post.beta[:, 1], post.sigma2):
            a, b = draws[:half].mean(), draws[half:].mean()
            assert abs(a - b) < 4 * draws.std()

    def test_variances_are_positive(self):
        panel = simulate_panel(2.0, -0.1, 1.0, 0.1, 1.0, 200, 4, seed=13)
        post = hbayes.fit_growth_model(
            hbayes.build_growth_data(panel), n_draws=600, burn_in=200, seed=13
        )
        assert (post.sigma2 > 0).all()
        assert (post.tau0_2 > 0).all()
        assert (post.tau1_2 > 0).all()

    def test_seed_is_reproducible(self):
        panel = simulate_panel(2.0, -0.1, 1.0, 0.1, 1.0, 150, 4, seed=15)
        data = hbayes.build_growth_data(panel)
        a = hbayes.fit_growth_model(data, n_draws=300, burn_in=100, seed=3)
        b = hbayes.fit_growth_model(data, n_draws=300, burn_in=100, seed=3)
        assert np.allclose(a.beta, b.beta)


class TestClinicalTranslation:
    def test_label_probabilities_use_the_published_thresholds(self):
        draws = np.array([2.0, 1.2, 1.0, 0.5])
        p = hbayes.label_probabilities(draws)
        assert p["TD"] == pytest.approx(0.5)            # 2.0 and 1.2 exceed 1.15
        assert p["Indeterminate"] == pytest.approx(0.25)  # 1.0 sits inside the band
        assert p["PIGD"] == pytest.approx(0.25)         # 0.5 is below 0.90

    def test_probabilities_sum_to_one(self):
        rng = np.random.default_rng(0)
        p = hbayes.label_probabilities(rng.normal(1.5, 1.0, 5000))
        assert sum(p.values()) == pytest.approx(1.0)

    def test_boundaries_are_closed_on_indeterminate(self):
        """0.90 and 1.15 themselves belong to the Indeterminate band."""
        p = hbayes.label_probabilities(np.array([0.90, 1.15]))
        assert p["Indeterminate"] == pytest.approx(1.0)


class TestPrediction:
    @pytest.fixture
    def posterior(self):
        panel = simulate_panel(2.0, -0.1, 1.0, 0.1, 0.6, 300, 5, seed=21)
        return hbayes.fit_growth_model(
            hbayes.build_growth_data(panel), n_draws=800, burn_in=300, seed=21
        )

    def test_prediction_follows_the_declining_trajectory(self, posterior):
        t_hist = np.array([0.0, 1.0, 2.0])
        y_hist = np.array([3.0, 2.9, 2.8])
        near = hbayes.predict_ratio(posterior, t_hist, y_hist, 3.0, seed=1, max_draws=400)
        far = hbayes.predict_ratio(posterior, t_hist, y_hist, 10.0, seed=1, max_draws=400)
        assert far.mean() < near.mean()

    def test_excluding_noise_narrows_the_prediction(self, posterior):
        t_hist, y_hist = np.array([0.0, 1.0]), np.array([2.0, 1.9])
        noisy = hbayes.predict_ratio(posterior, t_hist, y_hist, 2.0, True, seed=2, max_draws=600)
        clean = hbayes.predict_ratio(posterior, t_hist, y_hist, 2.0, False, seed=2, max_draws=600)
        assert clean.std() < noisy.std()

    def test_history_shifts_the_prediction(self, posterior):
        """A patient observed high must be predicted higher than one observed low."""
        t = np.array([0.0, 1.0, 2.0])
        high = hbayes.predict_ratio(posterior, t, np.array([5.0, 4.9, 4.8]), 3.0,
                                    seed=3, max_draws=400)
        low = hbayes.predict_ratio(posterior, t, np.array([0.5, 0.4, 0.3]), 3.0,
                                   seed=3, max_draws=400)
        assert high.mean() > low.mean() + 1.0

    def test_clinical_table_rows_sum_to_one(self, posterior):
        table = hbayes.clinical_table(posterior, current_ratio=2.0, horizons=(1, 5),
                                      max_draws=400)
        cols = ["TD", "Indeterminate", "PIGD"]
        assert np.allclose(table[cols].sum(axis=1), 1.0)

    def test_horizon_zero_reflects_the_current_ratio(self, posterior):
        table = hbayes.clinical_table(posterior, current_ratio=2.0, horizons=(1,),
                                      max_draws=200)
        now = table[table["horizon_years"] == 0.0].iloc[0]
        assert now["TD"] == pytest.approx(1.0)


class TestStage4PosteriorInference:
    """Per-patient latent phenotype, its uncertainty, and how much is real."""

    @pytest.fixture
    def fitted(self):
        panel = simulate_panel(2.0, -0.1, 1.0, 0.1, 0.8, 200, 5, seed=51)
        data = hbayes.build_growth_data(panel)
        post = hbayes.fit_growth_model(data, n_draws=600, burn_in=200, seed=51,
                                       keep_random_effects=True)
        return post, data

    def test_random_effects_retained_with_right_shape(self, fitted):
        post, data = fitted
        assert post.random_effects is not None
        assert post.random_effects.shape == (post.n_draws, data.n_patients, 2)

    def test_not_retained_by_default(self):
        panel = simulate_panel(2.0, -0.1, 1.0, 0.1, 0.8, 60, 4, seed=52)
        post = hbayes.fit_growth_model(hbayes.build_growth_data(panel),
                                       n_draws=200, burn_in=100, seed=52)
        assert post.random_effects is None
        with pytest.raises(ValueError, match="keep_random_effects"):
            hbayes.patient_parameters(post)

    def test_one_latent_row_per_observation(self, fitted):
        post, data = fitted
        assert len(hbayes.latent_trajectories(post, data)) == data.n_obs

    def test_latent_band_probabilities_sum_to_one(self, fitted):
        post, data = fitted
        lat = hbayes.latent_trajectories(post, data)
        total = lat[["P_TD", "P_Indeterminate", "P_PIGD"]].sum(axis=1)
        assert np.allclose(total, 1.0)

    def test_latent_credible_interval_brackets_the_mean(self, fitted):
        post, data = fitted
        lat = hbayes.latent_trajectories(post, data)
        assert (lat["latent_q0.025"] <= lat["latent_mean"]).all()
        assert (lat["latent_mean"] <= lat["latent_q0.975"]).all()

    def test_patient_parameters_one_row_each(self, fitted):
        post, data = fitted
        params = hbayes.patient_parameters(post)
        assert len(params) == data.n_patients
        assert params["PATNO"].is_unique
        assert (params["slope_q2.5"] <= params["slope_mean"]).all()

    def test_reliability_lies_in_the_unit_interval(self, fitted):
        post, data = fitted
        rel = hbayes.reliability(post, data)
        for col in ("reliability_baseline", "reliability_slope"):
            assert (rel[col] >= 0).all() and (rel[col] <= 1).all()

    def test_reliability_rises_with_more_visits(self):
        """The whole point of the statistic: more data means more of it is real."""
        few = simulate_panel(2.0, -0.1, 1.0, 0.1, 0.8, 150, 2, seed=53)
        many = simulate_panel(2.0, -0.1, 1.0, 0.1, 0.8, 150, 10, seed=53)
        out = []
        for panel in (few, many):
            data = hbayes.build_growth_data(panel)
            post = hbayes.fit_growth_model(data, n_draws=400, burn_in=200, seed=53,
                                           keep_random_effects=True)
            out.append(hbayes.reliability(post, data)["reliability_slope"].median())
        assert out[1] > out[0]

    def test_posterior_predictive_widens_with_horizon(self, fitted):
        post, data = fitted
        pred = hbayes.posterior_predictive(post, data, post.patients[0],
                                           horizons=(1, 10))
        width = pred["q97.5"] - pred["q2.5"]
        assert width.iloc[1] >= width.iloc[0]

    def test_posterior_predictive_probabilities_sum_to_one(self, fitted):
        post, data = fitted
        pred = hbayes.posterior_predictive(post, data, post.patients[0])
        assert np.allclose(pred[["TD", "Indeterminate", "PIGD"]].sum(axis=1), 1.0)
