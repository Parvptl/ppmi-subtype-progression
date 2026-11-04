"""Correctness checks for the CT-HMM fitted over joint (subtype, band) observations.

The two properties the experiment rests on are checked here directly: that the
analytic gradient is the gradient of the likelihood the model actually
maximises, and that the joint model reduces to the audited square model when
the bands are removed --- so a difference in the experiment's results can be
attributed to the bands rather than to new code or a new gradient.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppmi_subtype import hmm, hmm_banded as hb, hmm_stability, next_visit as nv
from ppmi_subtype import ratio_bands as rb

SUBTYPES = ["TD", "Indeterminate", "PIGD"]
FLOOR = hmm.DEFAULT_EMISSION_FLOOR


def _alphabet(n_tail: int = 3) -> rb.JointAlphabet:
    scheme = rb.BandScheme(tuple(3.0 + 0.25 * np.arange(n_tail)),
                           td_threshold=1.15, pigd_threshold=0.90)
    return rb.build_alphabet(scheme, SUBTYPES)


def _sequences(alphabet, n_seq: int = 30, seed: int = 0) -> list[dict]:
    """Random joint-category sequences, enough to fit on but fast."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_seq):
        n = int(rng.integers(2, 7))
        out.append({
            "PATNO": i,
            "labels": rng.integers(0, alphabet.n_categories, size=n),
            "gaps": np.round(rng.uniform(0.5, 3.0, size=n - 1), 3),
        })
    return out


def _label_sequences(n_seq: int = 30, seed: int = 3) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_seq):
        n = int(rng.integers(2, 8))
        out.append({"PATNO": i, "labels": rng.integers(0, 3, size=n),
                    "gaps": np.round(rng.uniform(0.5, 3.0, size=n - 1), 3)})
    return out


# ---------------------------------------------------------------------------
#  Parameterisation
# ---------------------------------------------------------------------------
def test_parameter_count_matches_the_layout():
    a = _alphabet()
    expected = 6 + 6 + 3 * sum(len(b) - 1 for b in a.blocks) + 2
    assert hb.n_params(a, 3) == expected
    assert len(hb.bounds_for(a, 3, FLOOR)) == expected
    assert len(hb.default_start(a, 3, FLOOR)) == expected


def test_the_flat_alphabet_gives_the_square_model_s_fourteen_parameters():
    flat = rb.flat_alphabet(SUBTYPES)
    assert hb.n_params(flat, 3) == hmm._n_params(3, 3) == 14


def test_the_emission_factorises_with_the_subtype_marginal_as_m():
    a = _alphabet()
    rng = np.random.default_rng(5)
    bounds = np.asarray(hb.bounds_for(a, 3, FLOOR))
    theta = np.clip(hb.default_start(a, 3, FLOOR) + rng.normal(0, 0.7, hb.n_params(a, 3)),
                    bounds[:, 0], bounds[:, 1])
    Q, E, pi, M = hb.unpack(theta, a, 3)

    assert np.allclose(E.sum(axis=1), 1.0)
    for s, block in enumerate(a.blocks):
        assert np.allclose(E[:, list(block)].sum(axis=1), M[:, s])
    assert np.allclose(rb.aggregate_to_subtype(E[0], a), M[0])
    assert np.allclose(Q.sum(axis=1), 0.0)
    assert np.isclose(pi.sum(), 1.0)


def test_the_emission_floor_constrains_the_subtype_marginal_not_a_single_band():
    a = _alphabet(8)
    bounds = np.asarray(hb.bounds_for(a, 3, FLOOR))
    rng = np.random.default_rng(11)
    for _ in range(25):
        theta = np.clip(rng.uniform(-9, 9, hb.n_params(a, 3)), bounds[:, 0], bounds[:, 1])
        _, E, _, M = hb.unpack(theta, a, 3)
        # Whatever the parameters, each state is recorded as its own subtype
        # more often than not -- the square model's constraint, on the
        # marginal. No individual joint category is constrained that way.
        assert np.diag(M).min() >= FLOOR - 1e-12
    assert E.min() > 0            # every category keeps non-zero probability


def test_the_floor_is_refused_when_the_counts_cannot_be_paired():
    a = _alphabet()
    with pytest.raises(ValueError, match="one hidden state per subtype"):
        hb.bounds_for(a, 4, FLOOR)
    assert len(hb.bounds_for(a, 4, None)) == hb.n_params(a, 4)


def test_restarts_are_the_square_model_s_scheme():
    a = _alphabet()
    starts = hb.starts_for(a, 3, FLOOR, seed=42, n_restarts=4)
    assert starts.shape == (4, hb.n_params(a, 3))
    assert np.allclose(starts[0], hb.default_start(a, 3, FLOOR))
    assert not np.allclose(starts[1], starts[0])
    # Same seed, same starts.
    assert np.allclose(starts, hb.starts_for(a, 3, FLOOR, seed=42, n_restarts=4))


# ---------------------------------------------------------------------------
#  Likelihood and gradient
# ---------------------------------------------------------------------------
def test_forward_backward_reproduces_the_audited_forward_likelihood():
    a = _alphabet()
    seqs = _sequences(a)
    batch = hmm.prepare_batch(seqs)
    theta = hb.default_start(a, 3, FLOOR)
    Q, E, pi, _ = hb.unpack(theta, a, 3)

    ll, gamma, _, _, valid, _ = hb._forward_backward(Q, E, pi, batch)
    assert np.isclose(ll, hmm._forward_loglik(Q, E, pi, batch), rtol=0, atol=1e-9)
    # Posterior state marginals, so each observed visit's row sums to one and
    # padding contributes nothing.
    assert np.allclose(gamma[valid].sum(axis=1), 1.0)
    assert np.allclose(gamma[~valid], 0.0)


@pytest.mark.parametrize("n_tail", [1, 4])
def test_the_analytic_gradient_matches_finite_differences(n_tail):
    a = _alphabet(n_tail)
    batch = hmm.prepare_batch(_sequences(a, n_seq=25, seed=n_tail))
    npar = hb.n_params(a, 3)
    bounds = np.asarray(hb.bounds_for(a, 3, FLOOR))
    rng = np.random.default_rng(n_tail)

    for trial in range(3):
        theta = hb.default_start(a, 3, FLOOR)
        if trial:
            theta = np.clip(theta + rng.normal(0, 0.6, npar), bounds[:, 0], bounds[:, 1])
        f, g = hb.loglik_and_grad(theta, batch, a, 3)
        assert np.isclose(f, hb._neg_loglik(theta, batch, a, 3))

        h = 1e-6
        fd = np.empty(npar)
        for i in range(npar):
            up, dn = theta.copy(), theta.copy()
            up[i] += h
            dn[i] -= h
            fd[i] = (hb._neg_loglik(up, batch, a, 3)
                     - hb._neg_loglik(dn, batch, a, 3)) / (2 * h)
        assert np.max(np.abs(g - fd) / np.maximum(np.abs(fd), 1.0)) < 1e-5


def test_the_rate_gradient_fallback_agrees_with_the_analytic_one():
    a = _alphabet(2)
    batch = hmm.prepare_batch(_sequences(a, n_seq=20, seed=9))
    theta = hb.default_start(a, 3, FLOOR)
    _, g = hb.loglik_and_grad(theta, batch, a, 3)
    fallback = hb._rate_grad_fd(theta, batch, a, 3, 6)
    assert np.allclose(g[:6], fallback, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
#  Reduction to the audited square model
# ---------------------------------------------------------------------------
def test_the_flat_alphabet_fit_reproduces_the_audited_square_fit():
    """With one category per subtype the joint model *is* the square model.

    Same parameter layout, same bounds, same starts; only the code path and the
    gradient differ. If this drifted, a difference in the experiment could not
    be attributed to the bands.
    """
    seqs = _label_sequences()
    flat = rb.flat_alphabet(SUBTYPES)
    joint = [{"PATNO": s["PATNO"], "labels": s["labels"], "gaps": s["gaps"]}
             for s in seqs]

    banded, _ = hb.fit_banded(joint, flat, SUBTYPES, emission_floor=FLOOR,
                              seed=42, n_restarts=4)
    starts = nv.audited_starts(3, 3, FLOOR, 42, 4)
    audited, _ = hmm_stability.fit_multistart(seqs, [0, 1, 2], n_hidden=3,
                                              emission_floor=FLOOR, starts=starts,
                                              state_names=SUBTYPES)

    assert banded.n_params == audited.n_params == 14
    assert abs(banded.loglik - audited.loglik) < 1e-3
    assert np.abs(banded.M - audited.E).max() < 1e-2
    assert np.abs(banded.Q - audited.Q).max() < 1e-2


def test_the_fit_reports_its_starts_and_keeps_the_best():
    a = _alphabet(2)
    fit, records = hb.fit_banded(_sequences(a, n_seq=25, seed=2), a, SUBTYPES,
                                 emission_floor=FLOOR, seed=42, n_restarts=3)
    assert len(records) == 3
    assert np.isclose(fit.loglik, max(r.loglik for r in records))
    assert fit.n_params == hb.n_params(a, 3)
    assert all(r.n_fev > 0 and np.isfinite(r.proj_grad) for r in records)


def test_fitting_rejects_a_category_outside_the_alphabet():
    a = _alphabet(1)
    bad = [{"PATNO": 0, "labels": np.array([0, a.n_categories]), "gaps": np.array([1.0])}]
    with pytest.raises(ValueError, match="outside the alphabet"):
        hb.fit_banded(bad, a, SUBTYPES, emission_floor=FLOOR)
    with pytest.raises(ValueError, match="no sequences"):
        hb.fit_banded([], a, SUBTYPES, emission_floor=FLOOR)


# ---------------------------------------------------------------------------
#  Prediction and aggregation
# ---------------------------------------------------------------------------
def test_predictions_aggregate_to_three_normalised_subtype_probabilities():
    a = _alphabet(2)
    seqs = _sequences(a, n_seq=25, seed=4)
    fit, _ = hb.fit_banded(seqs, a, SUBTYPES, emission_floor=FLOOR, seed=42,
                           n_restarts=1)
    preds = hb.predictive_distributions(fit, seqs[:5])

    assert len(preds) == sum(len(s["gaps"]) for s in seqs[:5])
    probs = preds[nv.PROB_COLS].to_numpy(dtype=float)
    assert np.all(np.isfinite(probs)) and (probs >= 0).all()
    assert np.allclose(probs.sum(axis=1), 1.0)
    # The aggregation must not lose mass: the joint prediction already sums to
    # one before it is renormalised.
    assert np.allclose(preds["p_joint_sum"], 1.0, atol=1e-12)
    assert list(preds["step"].iloc[:len(seqs[0]["gaps"])]) == list(
        range(len(seqs[0]["gaps"])))


def test_aggregating_the_joint_prediction_equals_predicting_through_m():
    """The subtype marginal of E is M, so the bands cannot reshape the output.

    This is why the comparison with the square model is clean: any difference
    has to come through the filtered latent state, not through a richer output
    distribution.
    """
    a = _alphabet(3)
    seqs = _sequences(a, n_seq=20, seed=6)
    fit, _ = hb.fit_banded(seqs, a, SUBTYPES, emission_floor=FLOOR, seed=42,
                           n_restarts=1)
    rng = np.random.default_rng(0)
    for _ in range(10):
        latent = rng.random(3)
        latent /= latent.sum()
        assert np.allclose(rb.aggregate_to_subtype(latent @ fit.E, a), latent @ fit.M)


def test_the_fit_exposes_readable_emission_tables():
    a = _alphabet(1)
    fit, _ = hb.fit_banded(_sequences(a, n_seq=20, seed=8), a, SUBTYPES,
                           emission_floor=FLOOR, seed=42, n_restarts=1)
    assert list(fit.subtype_emission_frame().columns) == SUBTYPES
    assert list(fit.emission_frame().columns) == a.labels
    cond = fit.band_conditional_frame()
    assert len(cond) == 3 * a.n_categories
    totals = cond.groupby(["hidden_state", "subtype"])["p_band_given_subtype"].sum()
    assert np.allclose(totals.to_numpy(), 1.0)
