"""Tests for the time-varying progression model (Direction 2).

Three tests carry the weight. `test_woodbury_matches_dense` checks the general
covariance inverse for both random-effect designs, since every estimate rests on
it. `test_recovers_known_slope_change` and `test_no_false_change_on_constant_data`
are the paired controls that make the headline result interpretable: a null
finding means nothing unless the method can detect a real change, and a positive
finding means nothing unless the method stays quiet when the truth is constant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype import jointlmm as J
from ppmi_subtype import timevarying as TV


def simulate(beta_tremor, beta_pigd, knots, sigma_b, sigma_e, n_patients,
             seed=0, n_lo=2, n_hi=9, span=10.0):
    """Draw trajectories from exactly the model `TV.fit` assumes."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_patients):
        n = int(rng.integers(n_lo, n_hi))
        t = np.sort(rng.uniform(0, span, n))
        t[0] = 0.0
        B = TV.spline_basis(t, knots)
        b = rng.multivariate_normal(np.zeros(2), sigma_b)
        e = rng.multivariate_normal(np.zeros(2), sigma_e, size=n)
        y = np.c_[B @ beta_tremor, B @ beta_pigd] + b + e
        for j in range(n):
            rows.append({"PATNO": i, "years_from_baseline": float(t[j]),
                         "tremor_score": y[j, 0], "pigd_score": y[j, 1]})
    return pd.DataFrame(rows)


def random_covariance(rng, q, floor=0.3):
    A = rng.normal(size=(q, q))
    return A @ A.T + floor * np.eye(q)


# ---------------------------------------------------------------------------
#  Basis
# ---------------------------------------------------------------------------
def test_spline_basis_is_a_hinge():
    B = TV.spline_basis(np.array([0.0, 1.0, 3.0, 7.0]), (2.0, 5.0))
    assert B.shape == (4, 4)
    assert np.allclose(B[:, 0], 1.0)
    assert np.allclose(B[:, 1], [0, 1, 3, 7])
    assert np.allclose(B[:, 2], [0, 0, 1, 5])    # (t - 2)+
    assert np.allclose(B[:, 3], [0, 0, 0, 2])    # (t - 5)+


def test_no_knots_is_a_straight_line():
    B = TV.spline_basis(np.array([0.0, 2.5]), ())
    assert B.shape == (2, 2)
    assert np.allclose(B, [[1.0, 0.0], [1.0, 2.5]])


def test_spline_is_continuous_at_the_knots():
    """A hinge basis cannot produce a jump, whatever the coefficients."""
    knots = (2.0, 5.0)
    rng = np.random.default_rng(0)
    beta = rng.normal(size=4)
    eps = 1e-9
    for k in knots:
        left = TV.spline_basis(np.array([k - eps]), knots) @ beta
        right = TV.spline_basis(np.array([k + eps]), knots) @ beta
        assert left == pytest.approx(right, abs=1e-6)


def test_period_labels_and_names():
    assert TV.period_labels((2.0, 5.0)) == ["0-2y", "2-5y", "5y+"]
    assert TV.period_labels(()) == ["0y+"]
    names = TV.coefficient_names((2.0, 5.0))
    assert len(names) == 8
    assert names[0] == "tremor_intercept"
    assert "slope_change_at_5y" in names[3]


def test_design_layout():
    X = TV.design(np.array([0.0, 3.0]), (2.0,))
    assert X.shape == (4, 6)                      # 2 visits x 2 outcomes, p = 3
    assert np.allclose(X[0], [1, 0, 0, 0, 0, 0])  # tremor at t=0
    assert np.allclose(X[1], [0, 0, 0, 1, 0, 0])  # pigd at t=0
    assert np.allclose(X[2], [1, 3, 1, 0, 0, 0])  # tremor at t=3
    assert np.allclose(X[3], [0, 0, 0, 1, 3, 1])


def test_random_design_shapes():
    Z = TV.random_design(np.array([0.0, 2.0]), random_slopes=False)
    assert Z.shape == (4, 2)
    assert np.allclose(Z, [[1, 0], [0, 1], [1, 0], [0, 1]])
    Z = TV.random_design(np.array([0.0, 2.0]), random_slopes=True)
    assert Z.shape == (4, 4)
    assert np.allclose(Z[2], [1, 0, 2, 0])


# ---------------------------------------------------------------------------
#  Covariance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("random_slopes", [False, True])
@pytest.mark.parametrize("n", [1, 2, 3, 7, 14])
def test_woodbury_matches_dense(random_slopes, n):
    rng = np.random.default_rng(n + int(random_slopes) * 100)
    q = 4 if random_slopes else 2
    sigma_b = random_covariance(rng, q, 0.5)
    sigma_e = random_covariance(rng, 2, 0.3)
    t = np.sort(rng.uniform(0, 10, n))

    Z = TV.random_design(t, random_slopes)
    dense = Z @ sigma_b @ Z.T + np.kron(np.eye(n), sigma_e)
    vinv, logdet, _ = TV.vinv_logdet(t, sigma_b, sigma_e, random_slopes)

    assert np.allclose(vinv, np.linalg.inv(dense), atol=1e-8)
    assert logdet == pytest.approx(np.linalg.slogdet(dense)[1], abs=1e-8)
    assert np.allclose(vinv, vinv.T, atol=1e-10)


@pytest.mark.parametrize("n", [2, 5, 14])
def test_random_intercept_case_reproduces_step_one(n):
    """The zero-knot random-intercept model must be Step 1's, exactly."""
    rng = np.random.default_rng(n)
    sigma_b = random_covariance(rng, 2, 0.5)
    sigma_e = random_covariance(rng, 2, 0.3)
    t = np.sort(rng.uniform(0, 10, n))

    mine, my_ld, _ = TV.vinv_logdet(t, sigma_b, sigma_e, random_slopes=False)
    theirs, their_ld = J.vinv_logdet(n, sigma_b, sigma_e)
    assert np.allclose(mine, theirs, atol=1e-9)
    assert my_ld == pytest.approx(their_ld, abs=1e-9)


@pytest.mark.parametrize("random_slopes", [False, True])
@pytest.mark.parametrize("knots", [(), (2.0, 5.0)])
def test_batched_accumulate_matches_the_loop(random_slopes, knots):
    rng = np.random.default_rng(7)
    q = 4 if random_slopes else 2
    sigma_b = random_covariance(rng, q, 0.5)
    sigma_e = random_covariance(rng, 2, 0.3)
    data = TV.pack(simulate(np.array([0.6, 0.02] + [0.0] * len(knots)),
                            np.array([0.24, 0.06] + [0.0] * len(knots)),
                            knots, np.eye(2) * 0.1, np.eye(2) * 0.1, 40, seed=7))

    batched = TV._accumulate(data, knots, sigma_b, sigma_e, random_slopes)

    p = 2 * TV.basis_width(knots)
    XtVX, XtVy, ytVy, logdet = np.zeros((p, p)), np.zeros(p), 0.0, 0.0
    for patno in data.patients:
        t = data.times[patno]
        y = data.responses[patno].reshape(-1)
        vinv, ld, _ = TV.vinv_logdet(t, sigma_b, sigma_e, random_slopes)
        X = TV.design(t, knots)
        XtVX += X.T @ vinv @ X
        XtVy += X.T @ vinv @ y
        ytVy += float(y @ vinv @ y)
        logdet += ld

    assert np.allclose(batched[0], XtVX, atol=1e-7)
    assert np.allclose(batched[1], XtVy, atol=1e-7)
    assert batched[2] == pytest.approx(ytVy, abs=1e-6)
    assert batched[3] == pytest.approx(logdet, abs=1e-7)


def test_unpack_rejects_wrong_length():
    with pytest.raises(ValueError):
        TV.unpack(np.zeros(4), q=2)


@pytest.mark.parametrize("q", [2, 4])
def test_unpack_is_always_positive_definite(q):
    rng = np.random.default_rng(q)
    for _ in range(25):
        theta = rng.normal(scale=2.0, size=TV._n_chol_params(q) + 3)
        for S in TV.unpack(theta, q):
            assert np.all(np.linalg.eigvalsh(S) > 0)


# ---------------------------------------------------------------------------
#  The two controls that make the result readable
# ---------------------------------------------------------------------------
def test_recovers_known_slope_change():
    """Plant a slope change and require the contrasts to find it."""
    knots = (2.0, 5.0)
    beta_t = np.array([0.60, 0.04, 0.00, -0.04])     # 0.04 -> 0.04 -> 0.00
    beta_p = np.array([0.24, 0.04, 0.03, 0.05])      # 0.04 -> 0.07 -> 0.12
    fit = TV.fit(TV.pack(simulate(beta_t, beta_p, knots,
                                  np.array([[.12, .01], [.01, .09]]),
                                  np.array([[.06, .01], [.01, .08]]),
                                  800, seed=7)), knots=knots)

    assert fit.converged
    slopes = fit.period_slopes()
    truth = {"tremor": [0.04, 0.04, 0.00], "pigd": [0.04, 0.07, 0.12]}
    for outcome, expected in truth.items():
        got = slopes[slopes.outcome == outcome].slope.to_numpy()
        assert np.max(np.abs(got - expected)) < 0.015

    c = fit.slope_contrasts()
    late_tremor = c[(c.outcome == "tremor") & (c.contrast == "5y+ - 0-2y")].iloc[0]
    late_pigd = c[(c.outcome == "pigd") & (c.contrast == "5y+ - 0-2y")].iloc[0]
    assert late_tremor.excludes_zero and late_tremor.difference < 0
    assert late_pigd.excludes_zero and late_pigd.difference > 0


def test_no_false_change_on_constant_data():
    """Generate a genuinely constant slope, fit with knots, find nothing."""
    fit = TV.fit(TV.pack(simulate(np.array([0.60, 0.023]), np.array([0.24, 0.066]),
                                  (), np.array([[.12, .01], [.01, .09]]),
                                  np.array([[.06, .01], [.01, .08]]),
                                  800, seed=21)),
                 knots=(2.0, 5.0))
    assert int(fit.slope_contrasts().excludes_zero.sum()) == 0


def test_constant_fit_matches_a_straight_line_truth():
    fit = TV.fit(TV.pack(simulate(np.array([0.60, 0.023]), np.array([0.24, 0.066]),
                                  (), np.diag([0.12, 0.09]), np.diag([0.06, 0.08]),
                                  600, seed=3)), knots=())
    slopes = fit.period_slopes()
    assert slopes[slopes.outcome == "tremor"].slope.iloc[0] == pytest.approx(0.023, abs=0.012)
    assert slopes[slopes.outcome == "pigd"].slope.iloc[0] == pytest.approx(0.066, abs=0.012)


def test_period_slopes_are_cumulative_sums_of_coefficients():
    """slope(period m) must equal b1 + ... + bm, not an individual coefficient."""
    knots = (2.0, 5.0)
    fit = TV.fit(TV.pack(simulate(np.array([0.6, 0.03, 0.01, -0.02]),
                                  np.array([0.24, 0.04, 0.02, 0.03]), knots,
                                  np.diag([0.1, 0.1]), np.diag([0.06, 0.08]),
                                  300, seed=5)), knots=knots)
    p = TV.basis_width(knots)
    slopes = fit.period_slopes()
    for d, outcome in enumerate(["tremor", "pigd"]):
        got = slopes[slopes.outcome == outcome].slope.to_numpy()
        coefs = fit.beta[d * p + 1: (d + 1) * p]
        assert np.allclose(got, np.cumsum(coefs))


# ---------------------------------------------------------------------------
#  Prediction and leakage-relevant behaviour
# ---------------------------------------------------------------------------
def test_prediction_shapes_and_positive_sd():
    fit = TV.fit(TV.pack(simulate(np.array([0.6, 0.02, 0.0, 0.0]),
                                  np.array([0.24, 0.06, 0.0, 0.0]), (2.0, 5.0),
                                  np.diag([0.1, 0.1]), np.diag([0.06, 0.08]),
                                  200, seed=9)), knots=(2.0, 5.0))
    mean, sd = TV.predict(fit, np.array([0.0, 1.0]),
                          np.array([[0.7, 0.3], [0.8, 0.4]]), 3.0)
    assert mean.shape == (2,) and sd.shape == (2,)
    assert np.all(sd > 0)


def test_prediction_is_not_clipped_at_zero():
    """Continuous metrics must see the real prediction, negative values included."""
    fit = TV.fit(TV.pack(simulate(np.array([0.6, 0.02]), np.array([0.24, 0.06]), (),
                                  np.diag([0.5, 0.5]), np.diag([0.05, 0.05]),
                                  200, seed=11)), knots=())
    mean, _ = TV.predict(fit, np.array([0.0, 1.0]),
                         np.array([[-3.0, -3.0], [-3.0, -3.0]]), 2.0)
    assert np.any(mean < 0)


def test_blup_requires_history():
    fit = TV.fit(TV.pack(simulate(np.array([0.6, 0.02]), np.array([0.24, 0.06]), (),
                                  np.diag([0.1, 0.1]), np.diag([0.06, 0.08]),
                                  150, seed=13)), knots=())
    with pytest.raises(ValueError):
        TV.blup(fit, np.array([]), np.empty((0, 2)))


def test_baselines_use_only_history():
    fit = TV.fit(TV.pack(simulate(np.array([0.6, 0.02]), np.array([0.24, 0.06]), (),
                                  np.diag([0.1, 0.1]), np.diag([0.06, 0.08]),
                                  150, seed=15)), knots=())
    t = np.array([0.0, 1.0, 2.0])
    Y = np.array([[0.5, 0.2], [0.6, 0.3], [0.9, 0.4]])
    assert np.allclose(TV.baseline_predict("persistence", t, Y, 3.0, fit)[0], Y[-1])
    assert np.allclose(TV.baseline_predict("patient_mean", t, Y, 3.0, fit)[0],
                       Y.mean(axis=0))
    with pytest.raises(ValueError):
        TV.baseline_predict("nonsense", t, Y, 3.0, fit)


def test_subset_removes_the_held_out_patient():
    data = TV.pack(simulate(np.array([0.6, 0.02]), np.array([0.24, 0.06]), (),
                            np.diag([0.1, 0.1]), np.diag([0.06, 0.08]), 20, seed=17))
    held = data.patients[0]
    train = TV.subset(data, [p for p in data.patients if p != held])
    assert held not in train.patients
    assert train.n_patients == data.n_patients - 1


def test_fit_is_deterministic():
    data = TV.pack(simulate(np.array([0.6, 0.02]), np.array([0.24, 0.06]), (),
                            np.diag([0.1, 0.1]), np.diag([0.06, 0.08]), 150, seed=19))
    a = TV.fit(data, knots=(2.0, 5.0))
    b = TV.fit(data, knots=(2.0, 5.0))
    assert np.allclose(a.beta, b.beta)
    assert a.neg2ll == pytest.approx(b.neg2ll)


# ---------------------------------------------------------------------------
#  Composition control
# ---------------------------------------------------------------------------
def test_within_patient_rates_are_per_year_and_binned_by_midpoint():
    frame = pd.DataFrame({
        "PATNO": [1, 1, 1],
        "years_from_baseline": [0.0, 2.0, 8.0],
        "tremor_score": [0.0, 0.2, 0.2],
        "pigd_score": [0.0, 0.0, 0.6],
    })
    rates = TV.within_patient_rates(frame, (2.0, 5.0))
    tremor = rates[rates.outcome == "tremor"].sort_values("midpoint")
    assert tremor.rate.iloc[0] == pytest.approx(0.1)      # 0.2 over 2 years
    assert tremor.midpoint.iloc[0] == pytest.approx(1.0)  # binned by midpoint
    assert tremor.period.iloc[0] == "0-2y"
    assert tremor.period.iloc[1] == "5y+"                 # midpoint 5.0


def test_within_patient_rates_drop_short_intervals():
    frame = pd.DataFrame({
        "PATNO": [1, 1], "years_from_baseline": [0.0, 0.1],
        "tremor_score": [0.0, 0.5], "pigd_score": [0.0, 0.5],
    })
    assert len(TV.within_patient_rates(frame, (2.0,), min_interval=0.3)) == 0


def test_paired_period_change_reports_too_few_patients():
    """One patient spanning both periods is not enough to estimate anything."""
    frame = pd.DataFrame({
        "PATNO": [1, 1, 1], "years_from_baseline": [0.0, 1.0, 9.0],
        "tremor_score": [0.1, 0.2, 0.9], "pigd_score": [0.1, 0.2, 0.9],
    })
    rates = TV.within_patient_rates(frame, (2.0, 5.0))
    out = TV.paired_period_change(rates, "0-2y", "5y+")
    assert (out.note == "too few patients").all()
    assert out.delta_rate.isna().all()


def test_paired_period_change_reports_an_absent_period():
    """An empty frame would read downstream as 'no change found'; it must not."""
    frame = pd.DataFrame({
        "PATNO": [1, 1], "years_from_baseline": [0.0, 1.0],
        "tremor_score": [0.1, 0.2], "pigd_score": [0.1, 0.2],
    })
    out = TV.paired_period_change(
        TV.within_patient_rates(frame, (2.0, 5.0)), "0-2y", "5y+")
    assert len(out) > 0
    assert out.note.str.startswith("no observations in").all()
    assert not out.excludes_zero.any()


def test_paired_period_change_detects_a_planted_acceleration():
    """Patients present in both periods, each accelerating by a known amount.

    Visit times are chosen so the first interval's midpoint falls in 0-2y and
    the second's in 5y+, because the period is assigned by midpoint.
    """
    rng = np.random.default_rng(23)
    rows = []
    for i in range(120):
        # rate 0.10/yr over [0, 1.5] (midpoint 0.75), then 0.30/yr over
        # [5.0, 9.0] (midpoint 7.0)
        for t, y in [(0.0, 0.0), (1.5, 0.15), (5.0, 0.15), (9.0, 0.15 + 4.0 * 0.30)]:
            rows.append({"PATNO": i, "years_from_baseline": t,
                         "tremor_score": y + rng.normal(0, 0.01),
                         "pigd_score": y + rng.normal(0, 0.01)})
    rates = TV.within_patient_rates(pd.DataFrame(rows), (2.0, 5.0))
    out = TV.paired_period_change(rates, "0-2y", "5y+")

    assert (out.n_patients == 120).all()
    assert out.excludes_zero.all()
    assert np.allclose(out.delta_rate, 0.20, atol=0.03)   # 0.30 - 0.10
