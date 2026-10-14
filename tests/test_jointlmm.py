"""Tests for the joint (bivariate) linear mixed model of the two subscores.

Two tests carry the weight. `test_vinv_matches_dense` checks the closed-form
inverse and log-determinant against dense linear algebra --- the whole model
rests on that identity, and an error there would be invisible in a fitted
summary. `test_recovers_known_parameters` simulates from the model and checks
the estimates cover the truth, because a wrong likelihood still optimises
smoothly and still prints a plausible table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype import jointlmm as J


def random_covariance(rng, floor=0.3):
    L = np.tril(rng.normal(size=(2, 2)))
    L[np.diag_indices(2)] = np.abs(L.diagonal()) + floor
    return L @ L.T


def simulate(beta, sigma_b, sigma_e, n_patients, seed=0, n_lo=2, n_hi=9):
    """Draw trajectories from exactly the model `fit_joint_lmm` assumes."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_patients):
        n = int(rng.integers(n_lo, n_hi))
        t = np.sort(rng.uniform(0, 8, n))
        t = t - t[0]
        b = rng.multivariate_normal(np.zeros(2), sigma_b)
        e = rng.multivariate_normal(np.zeros(2), sigma_e, size=n)
        mu = np.c_[beta[0] + beta[1] * t, beta[2] + beta[3] * t]
        y = mu + b + e
        for j in range(n):
            rows.append({"PATNO": i, "years_from_baseline": float(t[j]),
                         "tremor_score": y[j, 0], "pigd_score": y[j, 1]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Covariance algebra
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 5, 14])
def test_vinv_matches_dense(n):
    """The closed form must equal the dense inverse and log-determinant."""
    rng = np.random.default_rng(n)
    sigma_b, sigma_e = random_covariance(rng), random_covariance(rng)

    dense = np.kron(np.ones((n, n)), sigma_b) + np.kron(np.eye(n), sigma_e)
    vinv, logdet = J.vinv_logdet(n, sigma_b, sigma_e)

    assert np.allclose(vinv, np.linalg.inv(dense), atol=1e-10)
    assert logdet == pytest.approx(np.linalg.slogdet(dense)[1], abs=1e-10)
    assert np.allclose(vinv, vinv.T, atol=1e-12)


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(3)
    sigma_b, sigma_e = random_covariance(rng), random_covariance(rng)
    back_b, back_e = J._unpack(J._pack(sigma_b, sigma_e))
    assert np.allclose(back_b, sigma_b)
    assert np.allclose(back_e, sigma_e)


def test_unpack_always_positive_definite():
    """Any point the optimiser can reach must still be a valid covariance."""
    rng = np.random.default_rng(11)
    for _ in range(50):
        for S in J._unpack(rng.normal(scale=3.0, size=6)):
            assert np.all(np.linalg.eigvalsh(S) > 0)


def test_design_layout():
    X = J.design([0.0, 1.5])
    assert X.shape == (4, 4)
    # visit-major, outcome-minor: tremor row then PIGD row for each visit
    assert np.array_equal(X[0], [1.0, 0.0, 0.0, 0.0])
    assert np.array_equal(X[1], [0.0, 0.0, 1.0, 0.0])
    assert np.array_equal(X[2], [1.0, 1.5, 0.0, 0.0])
    assert np.array_equal(X[3], [0.0, 0.0, 1.0, 1.5])


# ---------------------------------------------------------------------------
#  Estimation
# ---------------------------------------------------------------------------
def test_recovers_known_parameters():
    beta = np.array([0.60, 0.02, 0.24, 0.055])
    sigma_b = np.array([[0.14, 0.02], [0.02, 0.10]])
    sigma_e = np.array([[0.062, 0.012], [0.012, 0.099]])

    fit = J.fit_joint_lmm(J.pack(simulate(beta, sigma_b, sigma_e, 900, seed=7)))

    assert fit.converged
    assert np.all(np.abs(fit.beta - beta) < 2.5 * fit.beta_se)
    assert np.allclose(fit.sigma_b, sigma_b, atol=0.03)
    assert np.allclose(fit.sigma_e, sigma_e, atol=0.02)


def test_recovers_a_negative_cross_correlation():
    """The sign of the between-patient correlation must not be assumed."""
    beta = np.array([0.6, 0.0, 0.3, 0.0])
    sigma_b = np.array([[0.20, -0.09], [-0.09, 0.15]])
    sigma_e = np.array([[0.05, 0.0], [0.0, 0.05]])

    fit = J.fit_joint_lmm(J.pack(simulate(beta, sigma_b, sigma_e, 700, seed=5)))
    assert fit.corr_b < -0.3


def test_independent_outcomes_give_near_zero_correlation():
    beta = np.array([0.6, 0.02, 0.3, 0.05])
    sigma_b = np.diag([0.12, 0.09])
    sigma_e = np.diag([0.06, 0.08])

    fit = J.fit_joint_lmm(J.pack(simulate(beta, sigma_b, sigma_e, 700, seed=9)))
    assert abs(fit.corr_b) < 0.12
    assert abs(fit.corr_e) < 0.12


def test_icc_matches_variance_components():
    rng = np.random.default_rng(2)
    fit = J.fit_joint_lmm(J.pack(simulate(
        np.array([0.5, 0.01, 0.3, 0.04]),
        random_covariance(rng, floor=0.2), random_covariance(rng, floor=0.2),
        300, seed=2)))
    vb, ve = np.diag(fit.sigma_b), np.diag(fit.sigma_e)
    assert np.allclose(fit.icc(), vb / (vb + ve))


# ---------------------------------------------------------------------------
#  Prediction
# ---------------------------------------------------------------------------
def test_blup_shrinks_towards_zero():
    """With one noisy visit the BLUP must not take the residual at face value."""
    fit = J.fit_joint_lmm(J.pack(simulate(
        np.array([0.6, 0.0, 0.3, 0.0]),
        np.diag([0.10, 0.10]), np.diag([0.10, 0.10]), 400, seed=4)))

    resid = np.array([[1.0, 1.0]]) + np.array([fit.beta[0], fit.beta[2]])
    b_hat, post = J.blup(fit, np.array([0.0]), resid)

    assert np.all(np.abs(b_hat) < 1.0)          # shrunk
    assert np.all(np.diag(post) < np.diag(fit.sigma_b))  # history reduced uncertainty


def test_blup_uncertainty_falls_with_more_history():
    fit = J.fit_joint_lmm(J.pack(simulate(
        np.array([0.6, 0.02, 0.3, 0.05]),
        np.diag([0.12, 0.09]), np.diag([0.06, 0.08]), 400, seed=6)))

    t = np.array([0.0, 1.0, 2.0, 3.0])
    Y = np.c_[0.7 + 0.02 * t, 0.35 + 0.05 * t]
    prev = np.inf
    for k in (1, 2, 3, 4):
        _, post = J.blup(fit, t[:k], Y[:k])
        assert post[0, 0] < prev
        prev = post[0, 0]


def test_blup_rejects_empty_history():
    fit = J.fit_joint_lmm(J.pack(simulate(
        np.array([0.6, 0.0, 0.3, 0.0]), np.diag([0.1, 0.1]), np.diag([0.1, 0.1]),
        200, seed=8)))
    with pytest.raises(ValueError):
        J.blup(fit, np.array([]), np.empty((0, 2)))


def test_prediction_sd_exceeds_residual_sd():
    """Predictive spread must include residual noise *and* intercept uncertainty."""
    fit = J.fit_joint_lmm(J.pack(simulate(
        np.array([0.6, 0.02, 0.3, 0.05]),
        np.diag([0.12, 0.09]), np.diag([0.06, 0.08]), 400, seed=10)))

    _, sd = J.predict(fit, np.array([0.0, 1.0]), np.c_[[0.7, 0.72], [0.35, 0.40]], 2.0)
    assert np.all(sd > np.sqrt(np.diag(fit.sigma_e)))


def test_baselines_use_only_history():
    fit = J.fit_joint_lmm(J.pack(simulate(
        np.array([0.6, 0.02, 0.3, 0.05]),
        np.diag([0.12, 0.09]), np.diag([0.06, 0.08]), 200, seed=12)))

    t = np.array([0.0, 1.0, 2.0])
    Y = np.array([[0.5, 0.2], [0.6, 0.3], [0.9, 0.4]])

    mean, _ = J.baseline_predict("persistence", t, Y, 3.0, fit)
    assert np.allclose(mean, Y[-1])

    mean, _ = J.baseline_predict("patient_mean", t, Y, 3.0, fit)
    assert np.allclose(mean, Y.mean(axis=0))

    mean, _ = J.baseline_predict("population", t, Y, 3.0, fit)
    assert np.allclose(mean, [fit.beta[0] + 3.0 * fit.beta[1],
                              fit.beta[2] + 3.0 * fit.beta[3]])


def test_unknown_baseline_rejected():
    fit = J.fit_joint_lmm(J.pack(simulate(
        np.array([0.6, 0.0, 0.3, 0.0]), np.diag([0.1, 0.1]), np.diag([0.1, 0.1]),
        200, seed=13)))
    with pytest.raises(ValueError):
        J.baseline_predict("nonsense", np.array([0.0]), np.array([[0.5, 0.2]]), 1.0, fit)


# ---------------------------------------------------------------------------
#  Packing
# ---------------------------------------------------------------------------
def test_pack_orders_visits_and_groups_by_count():
    frame = pd.DataFrame({
        "PATNO": [2, 2, 1, 1, 1],
        "years_from_baseline": [1.0, 0.0, 2.0, 0.0, 1.0],
        "tremor_score": [0.5, 0.4, 0.9, 0.7, 0.8],
        "pigd_score": [0.2, 0.1, 0.6, 0.4, 0.5],
    })
    data = J.pack(frame)

    assert data.patients == [1, 2]
    assert np.array_equal(data.times[1], [0.0, 1.0, 2.0])
    assert np.array_equal(data.responses[1][:, 0], [0.7, 0.8, 0.9])
    assert sorted(data.groups) == [2, 3]
    assert data.n_visits == 5


def test_pack_drops_only_rows_missing_a_subscore():
    frame = pd.DataFrame({
        "PATNO": [1, 1, 1],
        "years_from_baseline": [0.0, 1.0, 2.0],
        "tremor_score": [0.5, np.nan, 0.7],
        "pigd_score": [0.2, 0.3, 0.4],
    })
    data = J.pack(frame)
    assert data.n_visits == 2
    assert np.array_equal(data.times[1], [0.0, 2.0])


def test_pack_requires_the_subscore_columns():
    with pytest.raises(KeyError):
        J.pack(pd.DataFrame({"PATNO": [1], "years_from_baseline": [0.0]}))


def test_subset_excludes_the_held_out_patient():
    data = J.pack(simulate(np.array([0.6, 0.0, 0.3, 0.0]),
                           np.diag([0.1, 0.1]), np.diag([0.1, 0.1]), 20, seed=1))
    train = J.subset(data, [p for p in data.patients if p != data.patients[0]])

    assert data.patients[0] not in train.patients
    assert train.n_patients == data.n_patients - 1
    assert sum(len(v) for v in train.times.values()) == train.n_visits


def test_reml_and_ml_differ_but_both_converge():
    frame = simulate(np.array([0.6, 0.02, 0.3, 0.05]),
                     np.diag([0.12, 0.09]), np.diag([0.06, 0.08]), 250, seed=14)
    data = J.pack(frame)
    reml = J.fit_joint_lmm(data, reml=True)
    ml = J.fit_joint_lmm(data, reml=False)

    assert reml.converged and ml.converged
    assert reml.reml and not ml.reml
    # the fixed effects are a GLS solve given the variances, so they agree closely
    assert np.allclose(reml.beta, ml.beta, atol=5e-3)
