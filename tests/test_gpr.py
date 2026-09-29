"""Tests for the Gaussian process trajectory model.

Two tests carry most of the weight.

`TestGrowthModelEquivalence` checks that a K0 posterior reproduces the
hierarchical growth model's predictive distribution. The two are computed by
genuinely different routes --- function space through a kernel and a Cholesky
factor, weight space through a 2x2 precision matrix --- so agreement is evidence
about both, and the mandated "K0 must match hbayes" check has somewhere to
stand.

`TestObjective` checks the batched marginal likelihood against a plain
per-patient implementation and the analytic gradient against finite differences.
A padding or masking mistake produces a perfectly plausible number, so it can
only be caught by comparison with something simpler.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppmi_subtype import gpr, hbayes


def make_sequences(n_patients=25, seed=0, ragged=True):
    """Trajectories drawn from a linear growth model with noise."""
    rng = np.random.default_rng(seed)
    sequences = []
    for _ in range(n_patients):
        n = int(rng.integers(1, 10)) if ragged else 5
        t = np.sort(rng.uniform(0, 10, n))
        y = (2.3 + rng.normal(0, 1.3)) + (-0.09 + rng.normal(0, 0.1)) * t
        sequences.append({"t": t, "y": y + rng.normal(0, 1.1, n)})
    return sequences


class TestKernels:
    @pytest.mark.parametrize("kernel", ["K0", "K1", "K2", "K3"])
    def test_training_covariance_is_symmetric_positive_definite(self, kernel):
        t = np.array([0.0, 1.0, 2.5, 6.0])
        theta = gpr.initial_theta(kernel, np.array([1.0, 2.0, 1.5, 1.2]), t)
        K = gpr.kernel_matrix(kernel, t, t, theta, include_noise=True)

        assert np.allclose(K, K.T)
        assert np.all(np.linalg.eigvalsh(K) > 0)

    @pytest.mark.parametrize("kernel", ["K0", "K1", "K2", "K3"])
    def test_cross_covariance_has_the_right_shape(self, kernel):
        t1, t2 = np.array([0.0, 1.0, 2.0]), np.array([3.0, 4.0])
        theta = gpr.initial_theta(kernel, np.array([1.0, 2.0, 1.5]), t1)

        assert gpr.kernel_matrix(kernel, t1, t2, theta).shape == (3, 2)

    def test_noise_is_only_added_to_a_square_block(self):
        theta = gpr.initial_theta("K0", np.array([1.0, 2.0]), np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="square"):
            gpr.kernel_matrix("K0", np.array([0.0, 1.0]), np.array([2.0]),
                              theta, include_noise=True)

    def test_stationary_kernels_decay_with_separation(self):
        theta = np.array([0.0, 1.0, 2.0, 0.0])          # const 0, rbf var 1, ell 2
        near = gpr.kernel_matrix("K1", np.array([0.0]), np.array([0.5]), theta)[0, 0]
        far = gpr.kernel_matrix("K1", np.array([0.0]), np.array([8.0]), theta)[0, 0]
        assert near > far > 0

    def test_wrong_parameter_count_is_rejected(self):
        with pytest.raises(ValueError, match="takes"):
            gpr.kernel_matrix("K2", np.array([0.0]), np.array([0.0]), np.array([1.0, 1.0]))

    def test_parameter_names_match_the_component_list(self):
        assert gpr.parameter_names("K0") == ["var_const", "var_linear", "noise"]
        assert gpr.parameter_names("K2") == [
            "var_const", "var_linear", "var_rbf", "lengthscale", "noise"
        ]
        assert gpr.n_parameters("K3") == 5


class TestObjective:
    @pytest.mark.parametrize("kernel", ["K0", "K1", "K2", "K3"])
    def test_batched_matches_the_per_patient_reference(self, kernel):
        """Padding and masking must not change the likelihood by anything."""
        sequences = make_sequences(ragged=True)
        packed = gpr.pack_sequences(sequences)
        log_theta = np.log(
            gpr.initial_theta(
                kernel,
                np.concatenate([s["y"] for s in sequences]),
                np.concatenate([s["t"] for s in sequences]),
            )
        )
        for shift in (0.0, 0.8, -1.1):
            f_batch, g_batch = gpr._objective(log_theta + shift, kernel, packed, 1e-8)
            f_ref, g_ref = gpr._objective_reference(log_theta + shift, kernel,
                                                    sequences, 1e-8)
            assert f_batch == pytest.approx(f_ref, rel=1e-10)
            assert np.allclose(g_batch, g_ref, rtol=1e-8, atol=1e-10)

    @pytest.mark.parametrize("kernel", ["K0", "K1", "K2", "K3"])
    def test_analytic_gradient_matches_finite_differences(self, kernel):
        sequences = make_sequences(n_patients=12, seed=1)
        packed = gpr.pack_sequences(sequences)
        log_theta = np.log(
            gpr.initial_theta(
                kernel,
                np.concatenate([s["y"] for s in sequences]),
                np.concatenate([s["t"] for s in sequences]),
            )
        )
        _, grad = gpr._objective(log_theta, kernel, packed, 1e-8)

        eps = 1e-6
        for j in range(len(log_theta)):
            up, down = log_theta.copy(), log_theta.copy()
            up[j] += eps
            down[j] -= eps
            numeric = (
                gpr._objective(up, kernel, packed, 1e-8)[0]
                - gpr._objective(down, kernel, packed, 1e-8)[0]
            ) / (2 * eps)
            assert grad[j] == pytest.approx(numeric, rel=1e-4, abs=1e-4)

    def test_batched_cholesky_solve_matches_a_direct_solve(self):
        rng = np.random.default_rng(4)
        mats, rhs = [], []
        for _ in range(6):
            A = rng.normal(size=(5, 5))
            mats.append(A @ A.T + 5 * np.eye(5))
            rhs.append(rng.normal(size=(5, 3)))
        K, B = np.array(mats), np.array(rhs)

        got = gpr._batched_cholesky_solve(np.linalg.cholesky(K), B)
        want = np.array([np.linalg.solve(k, b) for k, b in zip(K, B)])

        assert np.allclose(got, want)

    def test_packing_preserves_every_observation(self):
        sequences = make_sequences(n_patients=8, seed=2)
        packed = gpr.pack_sequences(sequences)

        assert packed.n_patients == 8
        assert packed.n_obs == sum(len(s["y"]) for s in sequences)
        for i, seq in enumerate(sequences):
            n = len(seq["t"])
            assert np.allclose(packed.T[i, :n], seq["t"])
            assert np.allclose(packed.Y[i, :n], seq["y"])
            assert packed.mask[i].sum() == n


class TestFitting:
    def test_recovers_parameters_from_simulated_data(self):
        """Simulate from a known kernel and check the fit finds it back.

        Without this, a sampler or optimiser with a wrong gradient still
        produces smooth-looking parameter values.
        """
        rng = np.random.default_rng(7)
        tau0_2, tau1_2, sigma2 = 1.8, 0.010, 1.3
        beta = np.array([2.3, -0.09])

        sequences = []
        for _ in range(400):
            n = int(rng.integers(3, 9))
            t = np.sort(rng.uniform(0, 10, n))
            b0 = rng.normal(0, np.sqrt(tau0_2))
            b1 = rng.normal(0, np.sqrt(tau1_2))
            y = (beta[0] + b0) + (beta[1] + b1) * t + rng.normal(0, np.sqrt(sigma2), n)
            sequences.append({"t": t, "y": y})

        fit = gpr.fit_gpr(sequences, "K0", n_restarts=2, seed=0)
        params = fit.params

        assert params["var_const"] == pytest.approx(tau0_2, rel=0.25)
        assert params["var_linear"] == pytest.approx(tau1_2, rel=0.60)
        assert params["noise"] == pytest.approx(sigma2, rel=0.20)
        assert fit.beta[0] == pytest.approx(beta[0], abs=0.25)
        assert fit.beta[1] == pytest.approx(beta[1], abs=0.05)

    def test_restart_zero_is_the_unperturbed_start(self):
        """A one-restart fit must be reproducible and never worse than nothing."""
        sequences = make_sequences(n_patients=40, seed=5)
        a = gpr.fit_gpr(sequences, "K1", n_restarts=1, seed=0)
        b = gpr.fit_gpr(sequences, "K1", n_restarts=1, seed=999)

        assert np.allclose(a.theta, b.theta), "restart 0 must not depend on the seed"

    def test_more_restarts_never_lower_the_likelihood(self):
        sequences = make_sequences(n_patients=40, seed=6)
        one = gpr.fit_gpr(sequences, "K2", n_restarts=1, seed=42)
        four = gpr.fit_gpr(sequences, "K2", n_restarts=4, seed=42)

        assert four.log_marginal_likelihood >= one.log_marginal_likelihood - 1e-6

    def test_boundary_hits_are_reported(self):
        """A parameter pinned to its bound is an identifiability warning."""
        fit = gpr.fit_gpr(make_sequences(n_patients=30, seed=8), "K0",
                          n_restarts=1, seed=0)
        assert isinstance(fit.at_boundary, list)
        assert set(fit.at_boundary) <= set(gpr.parameter_names("K0"))

    def test_unknown_kernel_is_rejected(self):
        with pytest.raises(ValueError, match="unknown kernel"):
            gpr.fit_gpr(make_sequences(n_patients=3), "K9")

    def test_empty_training_set_is_rejected(self):
        with pytest.raises(ValueError, match="no training sequences"):
            gpr.fit_gpr([], "K0")


class TestPosterior:
    def make_fit(self, kernel="K2"):
        return gpr.fit_gpr(make_sequences(n_patients=40, seed=9), kernel,
                           n_restarts=1, seed=0)

    def test_shapes(self):
        fit = self.make_fit()
        mean, cov = gpr.posterior(fit, np.array([0.0, 1.0]), np.array([2.0, 1.8]),
                                  np.array([2.0, 3.0, 4.0]))
        assert mean.shape == (3,)
        assert cov.shape == (3, 3)

    def test_covariance_is_symmetric_and_positive_semidefinite(self):
        fit = self.make_fit()
        _, cov = gpr.posterior(fit, np.array([0.0, 1.0, 2.0]),
                               np.array([2.0, 1.8, 1.5]), np.array([3.0, 4.0, 9.0]))

        assert np.allclose(cov, cov.T)
        assert np.all(np.linalg.eigvalsh(cov) > -1e-8)

    def test_uncertainty_grows_with_extrapolation_distance(self):
        fit = self.make_fit("K1")
        t_hist, y_hist = np.array([0.0, 1.0, 2.0]), np.array([2.0, 1.9, 1.7])
        _, near = gpr.predict_one(fit, t_hist, y_hist, 2.5)
        _, far = gpr.predict_one(fit, t_hist, y_hist, 30.0)

        assert far > near

    def test_interpolation_tracks_a_noise_free_function(self):
        """With almost no noise, the posterior must pass through the data."""
        t = np.linspace(0, 10, 12)
        y = np.sin(t)
        theta = np.array([1e-6, 1.0, 1.5, 1e-8])          # const, rbf var, ell, noise
        fit = gpr.GPRFit("K1", theta, np.array([0.0, 0.0]), 0.0, 1, len(t), True, 1)

        mean, _ = gpr.posterior(fit, t, y, t, include_noise=False)
        assert np.allclose(mean, y, atol=1e-3)

    def test_recovers_a_smooth_function_between_observations(self):
        t = np.linspace(0, 10, 15)
        y = np.sin(t)
        theta = np.array([1e-6, 1.0, 1.5, 1e-6])
        fit = gpr.GPRFit("K1", theta, np.array([0.0, 0.0]), 0.0, 1, len(t), True, 1)

        t_star = np.array([1.3, 4.7, 8.2])
        mean, _ = gpr.posterior(fit, t, y, t_star, include_noise=False)
        assert np.allclose(mean, np.sin(t_star), atol=0.05)

    def test_observation_noise_widens_the_prediction(self):
        fit = self.make_fit()
        t_hist, y_hist = np.array([0.0, 1.0]), np.array([2.0, 1.8])
        _, latent = gpr.posterior(fit, t_hist, y_hist, np.array([2.0]),
                                  include_noise=False)
        _, recorded = gpr.posterior(fit, t_hist, y_hist, np.array([2.0]),
                                    include_noise=True)

        assert recorded[0, 0] > latent[0, 0]

    def test_empty_history_is_rejected(self):
        fit = self.make_fit()
        with pytest.raises(ValueError, match="at least one"):
            gpr.posterior(fit, np.array([]), np.array([]), np.array([1.0]))

    def test_mismatched_history_is_rejected(self):
        fit = self.make_fit()
        with pytest.raises(ValueError, match="times but"):
            gpr.posterior(fit, np.array([0.0, 1.0]), np.array([2.0]), np.array([2.0]))


class TestGrowthModelEquivalence:
    """K0 must reproduce the hierarchical growth model. This is mandatory."""

    PARAMS = dict(tau0_2=1.3408**2, tau1_2=0.1033**2, sigma2=1.1380**2)
    BETA = np.array([2.2920, -0.0927])

    def k0_fit(self):
        theta = gpr.growth_model_params(**self.PARAMS)
        return gpr.GPRFit("K0", theta, self.BETA, 0.0, 1, 1, True, 1)

    def test_posterior_matches_the_weight_space_predictive(self):
        """Function space and weight space are the same distribution."""
        rng = np.random.default_rng(0)
        fit = self.k0_fit()

        for _ in range(200):
            n = int(rng.integers(1, 8))
            t = np.sort(rng.uniform(0, 12, n))
            y = self.BETA[0] + self.BETA[1] * t + rng.normal(0, 1.2, n)
            t_star = float(rng.uniform(0, 14))

            got_mean, got_sd = gpr.predict_one(fit, t, y, t_star)
            want_mean, want_sd = gpr.growth_model_predictive(
                self.BETA, t_hist=t, y_hist=y, t_target=t_star, **self.PARAMS
            )
            # The two routes round differently, and the size of the gap tracks
            # the intermediate quantities --- times up to 12, ratios up to 11
            # --- not the size of the answer. A prediction can land near zero
            # through cancellation while still carrying that much rounding, so
            # the floor stays absolute. At 1e-7 it is six orders below the
            # quantisation of the ratio itself, which moves in steps of 1/11.
            assert got_mean == pytest.approx(want_mean, rel=1e-8, abs=1e-7)
            assert got_sd == pytest.approx(want_sd, rel=1e-8, abs=1e-7)

    def test_matches_the_hbayes_sampler_on_a_fixed_posterior(self):
        """Against `hbayes.predict_ratio` itself, not just against its algebra.

        A posterior whose draws are all identical reduces `predict_ratio` to
        repeated sampling from one predictive distribution, so its mean and
        standard deviation converge on the K0 posterior. Agreement here ties the
        kernel formulation to the code that produced the project's existing
        growth-model results.
        """
        n_draws = 6000
        post = hbayes.GrowthPosterior(
            beta=np.tile(self.BETA, (n_draws, 1)),
            sigma2=np.full(n_draws, self.PARAMS["sigma2"]),
            tau0_2=np.full(n_draws, self.PARAMS["tau0_2"]),
            tau1_2=np.full(n_draws, self.PARAMS["tau1_2"]),
            n_patients=1,
            n_obs=3,
            n_draws=n_draws,
        )
        t_hist = np.array([0.0, 1.0, 2.0])
        y_hist = np.array([2.6, 2.2, 1.9])
        t_target = 3.0

        draws = hbayes.predict_ratio(post, t_hist, y_hist, t_target, seed=1)
        got_mean, got_sd = gpr.predict_one(self.k0_fit(), t_hist, y_hist, t_target)

        # Monte-Carlo error on 6000 draws is about sd/sqrt(n) ~ 0.015.
        assert draws.mean() == pytest.approx(got_mean, abs=0.06)
        assert draws.std() == pytest.approx(got_sd, abs=0.06)

    def test_growth_model_params_orders_the_kernel_correctly(self):
        theta = gpr.growth_model_params(1.8, 0.01, 1.3)
        assert dict(zip(gpr.parameter_names("K0"), theta)) == {
            "var_const": 1.8, "var_linear": 0.01, "noise": 1.3
        }


class TestSubtypeProbabilities:
    def test_probabilities_sum_to_one(self):
        for mean in (0.2, 0.95, 1.08, 1.6, 5.0):
            for sd in (0.05, 0.5, 1.2, 3.0):
                probs = gpr.subtype_probabilities(mean, sd)
                assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)
                assert all(0.0 <= p <= 1.0 for p in probs.values())

    def test_a_wide_prediction_in_the_band_supports_all_three(self):
        """The reason the whole distribution is used rather than the mean.

        A predicted ratio of 1.08 sits inside the Indeterminate band, but with
        substantial spread the patient may plausibly record any of the three.
        """
        probs = gpr.subtype_probabilities(1.08, 1.1)
        assert all(p > 0.05 for p in probs.values())
        assert probs["Indeterminate"] < 0.2, "a 0.25-wide band cannot hold much mass"

    def test_a_confident_prediction_concentrates(self):
        probs = gpr.subtype_probabilities(4.0, 0.05)
        assert probs["TD"] > 0.99

    def test_thresholds_are_respected(self):
        """Mass either side of a threshold splits at that threshold."""
        probs = gpr.subtype_probabilities(1.15, 1.0)
        assert probs["TD"] == pytest.approx(0.5, abs=1e-9)

        probs = gpr.subtype_probabilities(0.90, 1.0)
        assert probs["PIGD"] == pytest.approx(0.5, abs=1e-9)

    def test_degenerate_predictive_falls_back_to_the_point(self):
        assert gpr.subtype_probabilities(2.0, 0.0)["TD"] == 1.0
        assert gpr.subtype_probabilities(0.5, 0.0)["PIGD"] == 1.0
        assert gpr.subtype_probabilities(1.0, 0.0)["Indeterminate"] == 1.0

    def test_negative_mass_is_measurable(self):
        """A Gaussian on the raw ratio scale puts some mass where no ratio can be."""
        assert gpr.negative_mass(1.6, 1.14) > 0.05
        assert gpr.negative_mass(5.0, 0.2) == pytest.approx(0.0, abs=1e-12)


class TestMeanFunctionVariants:
    """A zero-mean GP is a different model, not a fitting option.

    The reference article for this work uses m(x) = 0. These tests pin down
    that the alternative is implemented as a genuine model change, and that
    both arms are fitted by the same machinery so a comparison between them
    measures the model rather than the optimiser.
    """

    def test_zero_mean_fit_returns_exactly_zero_beta(self):
        fit = gpr.fit_gpr(make_sequences(n_patients=25, seed=11), "K1",
                          n_restarts=1, seed=0, fit_mean=False)
        assert np.array_equal(fit.beta, np.zeros(2))

    def test_fitted_mean_recovers_a_planted_offset_and_slope(self):
        rng = np.random.default_rng(12)
        sequences = []
        for _ in range(250):
            t = np.sort(rng.uniform(0, 10, 6))
            sequences.append({"t": t, "y": 4.0 - 0.30 * t + rng.normal(0, 0.4, 6)})

        fit = gpr.fit_gpr(sequences, "K1", n_restarts=2, seed=0)
        assert fit.beta[0] == pytest.approx(4.0, abs=0.35)
        assert fit.beta[1] == pytest.approx(-0.30, abs=0.06)

    @pytest.mark.parametrize("kernel", ["K0", "K1", "K2", "K3"])
    def test_zero_mean_gradient_matches_finite_differences(self, kernel):
        """Both objectives must be equally trustworthy, or the comparison is not."""
        sequences = make_sequences(n_patients=14, seed=13)
        packed = gpr.pack_sequences(sequences)
        log_theta = np.log(
            gpr.initial_theta(kernel, packed.Y[packed.mask], packed.T[packed.mask])
        )
        _, grad = gpr._objective(log_theta, kernel, packed, 1e-8, False)

        eps = 1e-6
        for j in range(len(log_theta)):
            up, down = log_theta.copy(), log_theta.copy()
            up[j] += eps
            down[j] -= eps
            numeric = (
                gpr._objective(up, kernel, packed, 1e-8, False)[0]
                - gpr._objective(down, kernel, packed, 1e-8, False)[0]
            ) / (2 * eps)
            assert grad[j] == pytest.approx(numeric, rel=1e-4, abs=1e-4)

    def test_a_fitted_mean_cannot_lose_to_zero_mean_in_sample(self):
        """The zero-mean model is nested, so its likelihood cannot be higher."""
        sequences = make_sequences(n_patients=40, seed=14)
        with_mean = gpr.fit_gpr(sequences, "K1", n_restarts=2, seed=0, fit_mean=True)
        without = gpr.fit_gpr(sequences, "K1", n_restarts=2, seed=0, fit_mean=False)
        assert with_mean.log_marginal_likelihood >= without.log_marginal_likelihood - 1e-6
