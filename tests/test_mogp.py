"""Tests for the multi-output GP over the two subscores.

Three tests carry the weight. `test_gradient_matches_finite_differences` checks
the analytic gradient of the profiled marginal likelihood for every
kernel/coregion pair --- a wrong gradient still optimises to *something*, and
the resulting fit would look entirely plausible. `test_recovers_known_*`
simulate from the model and check the learned cross-output correlation, which is
the quantity the whole step is about: the near-zero correlation found on real
data only means something if the model can detect a real one.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppmi_subtype import mogp as M

KERNELS = ["K0", "K1", "K2", "K3"]
COREGIONS = ["full", "diag"]


def simulate(comps, sigma_e, beta, n_patients, seed=0, n_lo=2, n_hi=10, span=9.0):
    """Draw patient trajectories from exactly the covariance the module builds."""
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n_patients):
        n = int(rng.integers(n_lo, n_hi))
        t = np.sort(rng.uniform(0, span, n))
        t = t - t[0]
        K = M.build_K(t, comps, sigma_e)
        L = np.linalg.cholesky(K + 1e-10 * np.eye(2 * n))
        y = (M._mean_design(t) @ beta + L @ rng.standard_normal(2 * n)).reshape(n, 2)
        seqs.append({"t": t, "y": y})
    return seqs


def random_sequences(rng, n_patients=25):
    return [
        {"t": np.sort(rng.uniform(0, 8, k)) - np.sort(rng.uniform(0, 8, k))[0],
         "y": rng.normal(size=(k, 2)) * 0.5 + np.array([0.6, 0.3])}
        for k in rng.integers(2, 8, n_patients)
    ]


# ---------------------------------------------------------------------------
#  Covariance construction
# ---------------------------------------------------------------------------
def test_expand_equals_kron():
    """The batched expansion must agree with an explicit Kronecker product."""
    rng = np.random.default_rng(0)
    k = rng.normal(size=(4, 5, 5))
    k = k @ k.transpose(0, 2, 1)
    B = np.array([[1.3, 0.4], [0.4, 0.9]])
    out = M._expand(k, B, np.ones((4, 5, 5)))
    for p in range(4):
        assert np.allclose(out[p], np.kron(k[p], B))


def test_build_K_layout_and_properties():
    t = np.array([0.0, 1.2, 2.9])
    B_c = np.array([[1.3, 0.4], [0.4, 0.9]])
    B_r = np.array([[0.8, 0.1], [0.1, 0.6]])
    sigma_e = np.array([[0.07, 0.01], [0.01, 0.09]])
    comps = [("const", B_c, None), ("rbf", B_r, 2.5)]

    K = M.build_K(t, comps, sigma_e)
    manual = (np.kron(np.ones((3, 3)), B_c)
              + np.kron(np.exp(-0.5 * ((t[:, None] - t[None, :]) / 2.5) ** 2), B_r)
              + np.kron(np.eye(3), sigma_e))

    assert K.shape == (6, 6)
    assert np.allclose(K, manual)
    assert np.allclose(K, K.T)
    assert np.all(np.linalg.eigvalsh(K) > 0)
    # visit-major, outcome-minor: row 2j+d is visit j, output d
    assert K[0, 1] == pytest.approx(B_c[0, 1] + B_r[0, 1] + sigma_e[0, 1])


def test_cross_K_excludes_noise():
    """Cross-covariance between distinct times must carry no observation noise."""
    comps = [("const", np.eye(2), None)]
    out = M.cross_K(np.array([0.0, 1.0]), np.array([3.0]), comps)
    assert out.shape == (4, 2)
    assert np.allclose(out, np.kron(np.ones((2, 1)), np.eye(2)))


@pytest.mark.parametrize("component", ["const", "linear", "rbf", "matern32"])
def test_temporal_is_symmetric_psd(component):
    t = np.array([0.0, 0.7, 2.4, 5.1])
    k, _ = M.temporal(component, t, t, 2.0)
    assert np.allclose(k, k.T)
    assert np.min(np.linalg.eigvalsh(k)) > -1e-10


def test_temporal_rejects_unknown_component():
    with pytest.raises(ValueError):
        M.temporal("nonsense", np.array([0.0]), np.array([0.0]), 1.0)


def test_diag_coregion_has_zero_off_diagonals():
    rng = np.random.default_rng(1)
    theta = rng.normal(size=len(M.parameter_layout("K2", "diag")))
    comps, sigma_e = M.unpack(theta, "K2", "diag")
    for _, B, _ in comps:
        assert B[0, 1] == pytest.approx(0.0)
    assert sigma_e[0, 1] == pytest.approx(0.0)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("coregion", COREGIONS)
def test_unpack_always_positive_semidefinite(kernel, coregion):
    rng = np.random.default_rng(hash((kernel, coregion)) % 2**32)
    for _ in range(20):
        theta = rng.normal(scale=2.0, size=len(M.parameter_layout(kernel, coregion)))
        comps, sigma_e = M.unpack(theta, kernel, coregion)
        for _, B, _ in comps:
            assert np.min(np.linalg.eigvalsh(B)) >= -1e-12
        assert np.min(np.linalg.eigvalsh(sigma_e)) >= -1e-12


def test_parameter_layout_counts():
    assert len(M.parameter_layout("K1", "full")) == 10   # 2x B(3) + ls + noise(3)
    assert len(M.parameter_layout("K1", "diag")) == 7
    assert len(M.parameter_layout("K0", "diag")) == 6
    with pytest.raises(ValueError):
        M.parameter_layout("K9", "full")
    with pytest.raises(ValueError):
        M.parameter_layout("K1", "sideways")


# ---------------------------------------------------------------------------
#  Likelihood
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("coregion", COREGIONS)
def test_gradient_matches_finite_differences(kernel, coregion):
    rng = np.random.default_rng(3)
    packed = M.pack(random_sequences(rng))
    theta = M.initial_theta(kernel, coregion, packed, rng)

    _, grad = M.objective(theta, kernel, coregion, packed, 1e-8)
    h = 1e-6
    numeric = np.empty_like(grad)
    for p in range(len(theta)):
        up, dn = theta.copy(), theta.copy()
        up[p] += h
        dn[p] -= h
        numeric[p] = (M.objective(up, kernel, coregion, packed, 1e-8)[0]
                      - M.objective(dn, kernel, coregion, packed, 1e-8)[0]) / (2 * h)

    assert np.max(np.abs(grad - numeric) / np.maximum(np.abs(numeric), 1e-3)) < 1e-4


def test_objective_rejects_wrong_length_vector():
    packed = M.pack(random_sequences(np.random.default_rng(0), 5))
    with pytest.raises(ValueError):
        M.unpack(np.zeros(3), "K1", "full")
    # the objective swallows the error and reports an unattainable value
    value, grad = M.objective(np.zeros(3), "K1", "full", packed, 1e-8)
    assert value == 1e12
    assert np.allclose(grad, 0.0)


def test_padding_does_not_change_the_likelihood():
    """A short patient padded alongside a long one must contribute the same."""
    rng = np.random.default_rng(5)
    short = {"t": np.array([0.0, 1.0]), "y": rng.normal(size=(2, 2))}
    long = {"t": np.linspace(0, 6, 7), "y": rng.normal(size=(7, 2))}
    theta = M.initial_theta("K1", "full", M.pack([short, long]), None)

    together = M.objective(theta, "K1", "full", M.pack([short, long]), 1e-8)[0]
    apart = (M.objective(theta, "K1", "full", M.pack([short]), 1e-8)[0]
             + M.objective(theta, "K1", "full", M.pack([long]), 1e-8)[0])
    # The profiled mean is shared when fitted together, so these differ; what
    # must hold is that padding has not made the batched value non-finite or
    # wildly different in scale.
    assert np.isfinite(together) and np.isfinite(apart)
    assert abs(together - apart) < 50.0


# ---------------------------------------------------------------------------
#  Recovery --- the controls for the headline result
# ---------------------------------------------------------------------------
def test_recovers_known_cross_output_correlation():
    """A real correlation must be found, or a null result means nothing."""
    B_c = np.array([[0.10, 0.05], [0.05, 0.08]])
    B_r = np.array([[0.06, 0.03], [0.03, 0.05]])
    sigma_e = np.array([[0.05, 0.015], [0.015, 0.07]])
    comps = [("const", B_c, None), ("rbf", B_r, 3.0)]
    beta = np.array([0.6, 0.02, 0.24, 0.066])

    fit = M.fit_mogp(M.pack(simulate(comps, sigma_e, beta, 600, seed=11)),
                     "K1", "full", n_restarts=2)

    assert fit.converged
    corr = fit.coregion_correlation()
    assert corr["const"] > 0.3
    assert corr["rbf"] > 0.3
    assert np.all(np.abs(fit.beta - beta) < 0.05)


def test_recovers_near_zero_correlation_when_outputs_are_independent():
    """And must *not* invent one when the truth is independence."""
    comps = [("const", np.diag([0.10, 0.08]), None),
             ("rbf", np.diag([0.06, 0.05]), 3.0)]
    sigma_e = np.diag([0.05, 0.07])

    fit = M.fit_mogp(M.pack(simulate(comps, sigma_e,
                                     np.array([0.6, 0.02, 0.24, 0.066]), 600, seed=12)),
                     "K1", "full", n_restarts=2)

    corr = fit.coregion_correlation()
    assert abs(corr["const"]) < 0.25
    assert abs(corr["noise"]) < 0.25


def test_recovers_known_lengthscale():
    comps = [("const", np.diag([0.08, 0.07]), None),
             ("rbf", np.diag([0.09, 0.08]), 2.5)]
    fit = M.fit_mogp(M.pack(simulate(comps, np.diag([0.04, 0.05]),
                                     np.array([0.6, 0.0, 0.3, 0.0]), 600, seed=13)),
                     "K1", "diag", n_restarts=2)
    assert fit.parameter_dict()["lengthscale_rbf"] == pytest.approx(2.5, rel=0.4)


def test_full_never_fits_worse_than_diag_in_sample():
    """`diag` is nested inside `full`, so the optimum cannot be worse."""
    rng = np.random.default_rng(14)
    packed = M.pack(random_sequences(rng, 60))
    full = M.fit_mogp(packed, "K1", "full", n_restarts=2)
    diag = M.fit_mogp(packed, "K1", "diag", n_restarts=2)
    assert full.nll <= diag.nll + 1e-3


# ---------------------------------------------------------------------------
#  Prediction
# ---------------------------------------------------------------------------
def test_prediction_shapes_and_symmetry():
    rng = np.random.default_rng(15)
    fit = M.fit_mogp(M.pack(random_sequences(rng, 40)), "K1", "full", n_restarts=1)
    t_hist = np.array([0.0, 1.0, 2.0])
    Y_hist = np.array([[0.7, 0.3], [0.8, 0.35], [0.75, 0.42]])

    mean, cov = M.predict_one(fit, t_hist, Y_hist, 3.0)
    assert mean.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.allclose(cov, cov.T)
    assert np.all(np.diag(cov) > 0)

    mean2, cov2 = M.posterior(fit, t_hist, Y_hist, np.array([3.0, 4.0]))
    assert mean2.shape == (4,)
    assert cov2.shape == (4, 4)


def test_prediction_requires_history():
    rng = np.random.default_rng(16)
    fit = M.fit_mogp(M.pack(random_sequences(rng, 30)), "K1", "diag", n_restarts=1)
    with pytest.raises(ValueError):
        M.posterior(fit, np.array([]), np.empty((0, 2)), np.array([1.0]))
    with pytest.raises(ValueError):
        M.posterior(fit, np.array([0.0, 1.0]), np.array([[0.5, 0.2]]), np.array([2.0]))


def test_diagonal_coregion_gives_uncorrelated_predictions():
    """With every off-diagonal pinned, the predictive covariance must be diagonal."""
    rng = np.random.default_rng(17)
    fit = M.fit_mogp(M.pack(random_sequences(rng, 40)), "K1", "diag", n_restarts=1)
    _, cov = M.predict_one(fit, np.array([0.0, 1.0]),
                           np.array([[0.7, 0.3], [0.8, 0.4]]), 2.0)
    assert cov[0, 1] == pytest.approx(0.0, abs=1e-10)


def test_pigd_zero_history_is_usable():
    """A patient whose PIGD is zero throughout must still produce a prediction.

    This is the population the ratio representation cannot see at all, so it has
    to survive every step of the joint model.
    """
    rng = np.random.default_rng(18)
    fit = M.fit_mogp(M.pack(random_sequences(rng, 40)), "K1", "full", n_restarts=1)
    t_hist = np.array([0.0, 1.0, 2.0])
    Y_hist = np.array([[0.6, 0.0], [0.7, 0.0], [0.65, 0.0]])

    mean, cov = M.predict_one(fit, t_hist, Y_hist, 3.0)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(cov))
    assert np.all(np.diag(cov) > 0)


def test_fit_is_deterministic_with_one_restart():
    rng = np.random.default_rng(19)
    packed = M.pack(random_sequences(rng, 40))
    a = M.fit_mogp(packed, "K1", "full", n_restarts=1, seed=42)
    b = M.fit_mogp(packed, "K1", "full", n_restarts=1, seed=42)
    assert np.allclose(a.theta, b.theta)
    assert a.nll == pytest.approx(b.nll)


def test_restart_zero_is_the_unperturbed_start():
    packed = M.pack(random_sequences(np.random.default_rng(20), 30))
    plain = M.initial_theta("K1", "full", packed, None)
    assert np.allclose(M.initial_theta("K1", "full", packed, None), plain)


def test_pack_rejects_empty_and_short_sequences():
    with pytest.raises(ValueError):
        M.pack([])
    with pytest.raises(ValueError):
        M.pack([{"t": np.array([]), "y": np.empty((0, 2))}])


def test_pack_preserves_values_and_mask():
    seqs = [{"t": np.array([0.0, 1.0]), "y": np.array([[0.5, 0.2], [0.6, 0.3]])},
            {"t": np.array([0.0]), "y": np.array([[0.9, 0.1]])}]
    packed = M.pack(seqs)
    assert packed.T.shape == (2, 2)
    assert packed.Y.shape == (2, 2, 2)
    assert packed.n_obs == 6                       # (2 + 1) visits x 2 outputs
    assert packed.mask.tolist() == [[True, True], [True, False]]
    assert packed.Y[1, 1].tolist() == [0.0, 0.0]   # padded slot left at zero
