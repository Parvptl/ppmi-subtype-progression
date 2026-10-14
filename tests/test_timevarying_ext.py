"""Tests for the generic-basis LMM used in the Stage 5 temporal audit.

The ones that matter: it reproduces `timevarying.fit` exactly with the
linear-spline basis; integer weights equal physically duplicated patients
(which is what makes a weighted refit a valid bootstrap replicate); the natural
spline is linear beyond its boundary knots; and the period-average slope of a
linear spline equals the spline's own period slope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype import jointlmm as J
from ppmi_subtype import timevarying as TV
from ppmi_subtype import timevarying_ext as TX


def simulate(n_patients=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_patients):
        n = int(rng.integers(2, 9))
        t = np.concatenate([[0], np.cumsum(rng.gamma(8, 1 / 8, n - 1))])
        b = rng.multivariate_normal([0, 0], [[0.08, 0.01], [0.01, 0.06]])
        for tt in t:
            mu_t = 0.5 + 0.03 * tt - 0.02 * max(tt - 5, 0) + b[0]
            mu_p = 0.25 + 0.05 * tt + 0.03 * max(tt - 5, 0) + b[1]
            e = rng.multivariate_normal([0, 0], [[0.05, 0.005], [0.005, 0.04]])
            rows.append({"PATNO": i, "years_from_baseline": tt,
                         "tremor_score": mu_t + e[0], "pigd_score": mu_p + e[1]})
    return J.pack(pd.DataFrame(rows))


@pytest.mark.parametrize("knots,rs,reml", [((2.0, 5.0), False, True),
                                           ((), False, False), ((3.0,), True, True)])
def test_same_objective_and_gls_as_timevarying(knots, rs, reml):
    """At identical parameters the two implementations must agree exactly."""
    data = simulate()
    q = 4 if rs else 2
    rng = np.random.default_rng(1)
    for _ in range(5):
        theta = rng.normal(scale=0.3, size=q * (q + 1) // 2 + 3) - 1.0
        ref = TV._objective(theta, data, knots, q, rs, reml)
        got = TX._objective(theta, data, TX.linear_spline(knots), q, rs, reml, None)
        assert got == pytest.approx(ref, rel=1e-12, abs=1e-9)
        sb, se = TV.unpack(theta, q)
        a = TV._accumulate(data, knots, sb, se, rs)
        b = TX._accumulate(data, TX.linear_spline(knots), sb, se, rs, None)
        assert np.allclose(np.linalg.solve(a[0], a[1]), np.linalg.solve(b[0], b[1]),
                           rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("knots,rs,reml", [((2.0, 5.0), False, True),
                                           ((), False, False), ((3.0,), True, True)])
def test_fits_reach_the_same_optimum_as_timevarying(knots, rs, reml):
    """Same optimum value. Parameters are compared only for the intercept-only
    model: with random slopes the optimum is flat and finite-difference L-BFGS-B
    stops at points differing by ~1e-6 in beta at an identical -2logL."""
    data = simulate()
    ref = TV.fit(data, knots=knots, random_slopes=rs, reml=reml)
    got = TX.fit_basis(data, TX.linear_spline(knots), random_slopes=rs, reml=reml)
    assert got.neg2ll == pytest.approx(ref.neg2ll, abs=1e-6)
    if not rs:
        assert np.allclose(got.beta, ref.beta, atol=1e-6)
        assert np.allclose(got.cov_beta, ref.cov_beta, rtol=1e-5, atol=1e-10)


def test_integer_weights_equal_duplicated_patients():
    data = simulate(60, seed=3)
    w = {p: (2.0 if i % 3 == 0 else 1.0 if i % 3 == 1 else 0.0)
         for i, p in enumerate(data.patients)}
    weighted = TX.fit_basis(data, TX.linear_spline((4.0,)), weights=w)
    rows = []
    new_id = 0
    for p in data.patients:
        for _ in range(int(w[p])):
            for tt, yy in zip(data.times[p], data.responses[p]):
                rows.append({"PATNO": new_id, "years_from_baseline": tt,
                             "tremor_score": yy[0], "pigd_score": yy[1]})
            new_id += 1
    dup = TX.fit_basis(J.pack(pd.DataFrame(rows)), TX.linear_spline((4.0,)))
    assert weighted.neg2ll == pytest.approx(dup.neg2ll, rel=1e-8)
    assert np.allclose(weighted.beta, dup.beta, atol=1e-6)


def test_natural_spline_is_linear_outside_boundary_knots():
    B = TX.natural_cubic_spline([0.0, 2.0, 5.0, 9.0])
    beta = np.array([0.3, 0.1, 0.02, -0.04])
    t = np.array([9.0, 10.0, 11.0, 12.0])
    y = B(t) @ beta
    assert np.allclose(np.diff(y, 2), 0.0, atol=1e-10)
    assert B(np.array([1.0])).shape == (1, 4)


def test_average_slope_of_linear_spline_equals_period_slope():
    data = simulate(seed=5)
    knots = (2.0, 5.0)
    f = TX.fit_basis(data, TX.linear_spline(knots), reml=True)
    ref = TV.fit(data, knots=knots).period_slopes()
    got = TX.period_slopes(f, TX.linear_spline(knots), [(0, 2), (2, 5), (5, 9)])
    for r, g in zip(ref.itertuples(), got):
        assert g["slope"] == pytest.approx(r.slope, abs=1e-6)
        assert g["wald_se"] == pytest.approx(r.std_error, rel=1e-4)


def test_reml_fit_refuses_information_criteria():
    f = TX.fit_basis(simulate(30), TX.polynomial(2), reml=True)
    with pytest.raises(ValueError):
        _ = f.aic


def test_bootstrap_weights_are_multinomial():
    ws = list(TX.bootstrap_weights(range(50), 20, 1))
    assert all(sum(w.values()) == 50 for w in ws)
