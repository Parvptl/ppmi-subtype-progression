"""Correctness checks for the latent clinical-state model.

The properties tested are the ones the experiment's conclusions rest on:
features are built causally and carry no future information, missing
measurements are marginalised rather than filled, preprocessing is learned
from training data only, EM increases the likelihood it claims to maximise,
and prediction uses filtering and never a smoothed posterior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype import clinical_states as cs
from ppmi_subtype import next_visit as nv


# ---------------------------------------------------------------------------
#  Synthetic sequences, used only to exercise the estimator
# ---------------------------------------------------------------------------
def _simulate(n_patients=60, n_states=3, n_features=4, seed=0, missing=0.0):
    """Draw sequences from a known CT-HMM so the fitter has a target."""
    rng = np.random.default_rng(seed)
    mu = np.linspace(-2.0, 2.0, n_states)[:, None] * np.ones((1, n_features))
    rates = np.full(n_states * (n_states - 1), 0.25)
    from ppmi_subtype import ctmc
    Q = ctmc.build_generator(rates, n_states)

    sequences = []
    for p in range(n_patients):
        n = int(rng.integers(3, 7))
        gaps = rng.uniform(0.8, 1.5, n - 1)
        times = np.cumsum(np.r_[0.0, gaps])
        state = int(rng.integers(0, n_states))
        X = np.empty((n, n_features))
        for t in range(n):
            if t:
                P = ctmc.transition_probabilities(Q, np.array([gaps[t - 1]]))[0]
                state = int(rng.choice(n_states, p=P[state]))
            X[t] = mu[state] + rng.normal(0, 0.6, n_features)
        if missing:
            X[rng.uniform(size=X.shape) < missing] = np.nan
        sequences.append({
            "PATNO": p, "X": X, "times": times, "gaps": gaps,
            "labels": rng.choice([1, 2, 3], size=n, p=[0.6, 0.1, 0.3]),
            "events": np.array([f"V{t:02d}" for t in range(n)]),
            "visit_index": np.arange(n),
        })
    return sequences


# ---------------------------------------------------------------------------
#  Feature construction
# ---------------------------------------------------------------------------
def test_brady_rigid_items_do_not_overlap_tremor_or_pigd():
    """The whole point of the new subscore is that it adds a distinct axis."""
    from ppmi_subtype.config import load_config
    scfg = load_config().subtype
    overlap = set(cs.BRADY_RIGID_ITEMS) & (set(scfg.tremor_items) | set(scfg.pigd_items))
    assert not overlap, f"items counted twice: {sorted(overlap)}"
    assert len(set(cs.BRADY_RIGID_ITEMS)) == len(cs.BRADY_RIGID_ITEMS)
    assert "NP3TOT" not in cs.BRADY_RIGID_ITEMS      # a total, not an item


def test_feature_spec_is_coherent():
    assert len(cs.FEATURES) == len(cs.FEATURE_NAMES) == 10
    assert len(set(cs.FEATURE_NAMES)) == 10
    assert len(set(f.column for f in cs.FEATURES)) == 10
    # MoCA is the one feature where a higher score is better; the state-ordering
    # rule depends on that sign being right.
    assert not next(f for f in cs.FEATURES if f.name == "moca").higher_is_worse
    assert all(f.higher_is_worse for f in cs.FEATURES if f.name != "moca")


def _toy_clinical(n_patients=4, n_visits=4) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for p in range(n_patients):
        for v in range(n_visits):
            row = {"PATNO": p, "EVENT_ID": f"V{v:02d}",
                   "years_from_baseline": float(v), "visit_date": "2020-01-01",
                   "subtype": float(rng.choice([1, 2, 3])), "age_at_visit": 60.0,
                   "n_features_present": len(cs.FEATURE_NAMES)}
            for f in cs.FEATURE_NAMES:
                row[f] = float(rng.normal())
            rows.append(row)
    out = pd.DataFrame(rows)
    out["visit_index"] = out.groupby("PATNO").cumcount()
    return out


def test_build_sequences_preserves_order_and_gaps():
    clinical = _toy_clinical()
    seqs = cs.build_sequences(clinical)
    assert len(seqs) == 4
    for s in seqs:
        assert np.all(np.diff(s["times"]) > 0)
        assert np.allclose(s["gaps"], np.diff(s["times"]))
        assert s["X"].shape == (len(s["times"]), len(cs.FEATURE_NAMES))
        assert len(s["labels"]) == len(s["times"])


def test_build_sequences_drops_single_visit_patients():
    clinical = _toy_clinical(n_patients=3, n_visits=1)
    assert cs.build_sequences(clinical) == []


# ---------------------------------------------------------------------------
#  Missing data is marginalised, not imputed
# ---------------------------------------------------------------------------
def test_emission_marginalises_missing_features_exactly():
    """Dropping a feature must equal scoring the sub-vector of the rest."""
    fit = cs.CTHMMGaussian(
        Q=np.array([[-0.2, 0.2], [0.3, -0.3]]),
        mu=np.array([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]]),
        var=np.array([[1.0, 0.5, 2.0], [1.5, 1.0, 0.8]]),
        pi=np.array([0.5, 0.5]))

    full = np.array([[0.3, 0.7, -0.9]])
    partial = np.array([[0.3, np.nan, -0.9]])

    ll_full = fit.emission_loglik(full)[0]
    ll_partial = fit.emission_loglik(partial)[0]

    # The partial density is the full one with the middle feature's term removed.
    middle = -0.5 * (np.log(2 * np.pi * fit.var[:, 1])
                     + (full[0, 1] - fit.mu[:, 1]) ** 2 / fit.var[:, 1])
    assert np.allclose(ll_partial, ll_full - middle)


def test_a_visit_with_no_features_is_uninformative():
    fit = cs.CTHMMGaussian(
        Q=np.array([[-0.2, 0.2], [0.3, -0.3]]),
        mu=np.zeros((2, 3)), var=np.ones((2, 3)), pi=np.array([0.5, 0.5]))
    ll = fit.emission_loglik(np.full((1, 3), np.nan))[0]
    assert np.allclose(ll, 0.0)      # flat across states: says nothing


def test_fitting_tolerates_missing_measurements():
    seqs = _simulate(n_patients=40, missing=0.25, seed=3)
    assert np.isnan(np.vstack([s["X"] for s in seqs])).any()
    fit = cs.fit_em(seqs, 3, seed=1, max_iter=30)
    assert np.isfinite(fit.loglik)
    assert np.isfinite(fit.mu).all() and (fit.var > 0).all()


# ---------------------------------------------------------------------------
#  Training-only preprocessing
# ---------------------------------------------------------------------------
def test_standardiser_learns_from_training_sequences_only():
    seqs = _simulate(n_patients=40, seed=7)
    train, test = seqs[:30], seqs[30:]
    std = cs.Standardiser.fit(train)

    X_train = np.vstack([s["X"] for s in train])
    assert np.allclose(std.centre, np.nanmean(X_train, axis=0))
    assert std.n_train_visits == len(X_train)

    X_all = np.vstack([s["X"] for s in seqs])
    assert not np.allclose(std.centre, np.nanmean(X_all, axis=0))

    # Applying it leaves the test sequences' shape and missingness untouched.
    z = std.apply(test)
    for a, b in zip(test, z):
        assert a["X"].shape == b["X"].shape
        assert np.array_equal(np.isnan(a["X"]), np.isnan(b["X"]))


def test_standardiser_round_trips():
    seqs = _simulate(n_patients=20, seed=11)
    std = cs.Standardiser.fit(seqs)
    z = std.apply(seqs)
    assert np.allclose(std.invert(z[0]["X"]), seqs[0]["X"], equal_nan=True)


def test_standardiser_survives_a_constant_feature():
    seqs = _simulate(n_patients=10, seed=2)
    for s in seqs:
        s["X"][:, 0] = 3.0
    std = cs.Standardiser.fit(seqs)
    assert std.scale[0] == 1.0                   # not a divide-by-zero
    assert np.isfinite(std.apply(seqs)[0]["X"]).all()


# ---------------------------------------------------------------------------
#  EM
# ---------------------------------------------------------------------------
def test_em_log_likelihood_is_non_decreasing():
    """EM's defining guarantee. A decrease is a bug, not slow convergence."""
    seqs = _simulate(n_patients=50, seed=13)
    lls = []
    fit = cs._initialise(seqs, 3, seed=1)
    for _ in range(12):
        fit = cs._fit_em_from(seqs, fit, max_iter=1, tol=0.0)
        lls.append(fit.loglik)
    diffs = np.diff(lls)
    assert (diffs > -1e-6).all(), f"log-likelihood decreased: {diffs}"


def test_em_recovers_well_separated_states():
    seqs = _simulate(n_patients=80, n_states=3, seed=17)
    fit = cs.fit_em(seqs, 3, seed=1)
    std = cs.Standardiser.fit(seqs)
    fit = cs.reorder(fit, cs.canonical_order(fit, std))
    # Truth is means at -2, 0, +2 on every feature; ordering is by burden.
    recovered = fit.mu.mean(axis=1)
    assert np.all(np.diff(recovered) > 0.5), recovered
    assert abs(recovered[0] - (-2.0)) < 0.8
    assert abs(recovered[-1] - 2.0) < 0.8


def test_em_is_deterministic_for_a_given_seed():
    seqs = _simulate(n_patients=30, seed=19)
    a = cs.fit_em(seqs, 3, seed=5)
    b = cs.fit_em(seqs, 3, seed=5)
    assert a.loglik == b.loglik
    assert np.array_equal(a.mu, b.mu)


def test_multistart_start_zero_reproduces_fit_em():
    seqs = _simulate(n_patients=30, seed=23)
    plain = cs.fit_em(seqs, 3, seed=9)
    best, records = cs.fit_em_multistart(seqs, 3, seed=9, n_starts=3)
    assert np.isclose(records[0]["loglik"], plain.loglik, atol=1e-9)
    assert best.loglik >= plain.loglik          # best of several is never worse
    assert len(records) == 3
    assert sum(r["is_best"] for r in records) >= 1


def test_variance_floor_prevents_collapse():
    seqs = _simulate(n_patients=20, seed=29)
    # Give one patient a repeated identical observation, which a state could
    # otherwise collapse onto with vanishing variance.
    seqs[0]["X"][:] = 5.0
    fit = cs.fit_em(seqs, 3, seed=1, max_iter=40)
    assert (fit.var >= cs.VARIANCE_FLOOR).all()
    assert np.isfinite(fit.loglik)


def test_generator_rows_sum_to_zero_and_rates_stay_positive():
    seqs = _simulate(n_patients=40, seed=31)
    fit = cs.fit_em(seqs, 3, seed=1, max_iter=25)
    assert np.allclose(fit.Q.sum(axis=1), 0.0, atol=1e-8)
    off = ~np.eye(fit.n_states, dtype=bool)
    assert (fit.Q[off] > 0).all()
    assert (np.diag(fit.Q) < 0).all()


def test_n_params_counts_every_free_parameter():
    fit = cs.CTHMMGaussian(Q=np.zeros((3, 3)), mu=np.zeros((3, 4)),
                           var=np.ones((3, 4)), pi=np.ones(3) / 3)
    # 3*2 rates + 3*4 means + 3*4 variances + 2 initial = 32
    assert fit.n_params == 6 + 12 + 12 + 2


# ---------------------------------------------------------------------------
#  Filtering versus smoothing: the leakage property
# ---------------------------------------------------------------------------
def test_filtering_ignores_later_visits():
    """alpha[k] must not change when a later visit's measurements change.

    This is the property that makes the predictions causal. If the forward
    recursion leaked any future information, editing visit k+1 would move the
    filtered distribution at visit k.
    """
    seqs = _simulate(n_patients=5, seed=37)
    fit = cs.fit_em(seqs, 3, seed=1, max_iter=20)
    seq = seqs[0]

    alpha, _ = fit.filter_forward(seq["X"], seq["gaps"])
    tampered = seq["X"].copy()
    tampered[-1] += 50.0                      # wreck the last visit
    alpha2, _ = fit.filter_forward(tampered, seq["gaps"])

    assert np.allclose(alpha[:-1], alpha2[:-1]), "filtering saw the future"
    assert not np.allclose(alpha[-1], alpha2[-1]), "last step should respond"


def test_predictions_do_not_move_when_the_target_visit_changes():
    """The stronger end-to-end version: the prediction FOR a visit ignores it."""
    seqs = _simulate(n_patients=12, seed=41)
    fit = cs.fit_em(seqs, 3, seed=1, max_iter=20)
    fit.label_emission = cs.fit_label_emission(fit, seqs)

    # A patient long enough to have a prediction BEFORE and AFTER the tampered
    # visit. Without this the "should change" row can be empty, and np.allclose
    # on two empty arrays is vacuously true -- the assertion would pass while
    # testing nothing.
    seq = next(s for s in seqs if len(s["times"]) >= 5)
    target = 2                                   # the visit we corrupt

    base = cs.next_visit_predictions(fit, [seq])

    # Push the tampered visit onto the mean of a DIFFERENT state than the one
    # it currently filters to. Simply adding a large constant is not enough:
    # these synthetic states are well separated, so the filtered posterior is
    # already near one-hot, and an extreme value in the same direction lands on
    # the same state and changes nothing.
    alpha, _ = fit.filter_forward(seq["X"], seq["gaps"])
    current = int(alpha[target].argmax())
    other = (current + 1) % fit.n_states
    tampered = [dict(seq, X=seq["X"].copy())]
    tampered[0]["X"][target] = fit.mu[other]
    after = cs.next_visit_predictions(fit, tampered)

    alpha2, _ = fit.filter_forward(tampered[0]["X"], seq["gaps"])
    assert int(alpha2[target].argmax()) == other, "tamper did not move the state"

    cols = [c for c in base.columns if c.startswith(("p_state_", "p_TD", "p_PIGD"))]

    # The row predicting the tampered visit (from the one before it) must be
    # untouched: it may not see its own target.
    before_rows = base[base["visit_index"] == target - 1][cols].to_numpy(float)
    after_rows = after[after["visit_index"] == target - 1][cols].to_numpy(float)
    assert len(before_rows) == 1
    assert np.allclose(before_rows, after_rows), "a prediction saw its target"

    # The row predicting the NEXT visit must change: the tampered visit is
    # legitimately part of its history.
    later_before = base[base["visit_index"] == target][cols].to_numpy(float)
    later_after = after[after["visit_index"] == target][cols].to_numpy(float)
    assert len(later_before) == 1, "test needs a prediction made from the tampered visit"
    assert not np.allclose(later_before, later_after)


def test_next_state_prediction_uses_elapsed_time():
    fit = cs.CTHMMGaussian(
        Q=np.array([[-0.5, 0.5], [0.2, -0.2]]),
        mu=np.array([[0.0], [1.0]]), var=np.ones((2, 1)),
        pi=np.array([1.0, 0.0]))
    alpha = np.array([1.0, 0.0])
    short = fit.predict_next_state(alpha, 0.1)
    long = fit.predict_next_state(alpha, 10.0)
    assert short[0] > long[0]                  # more time, more drift away
    assert np.isclose(short.sum(), 1.0) and np.isclose(long.sum(), 1.0)


# ---------------------------------------------------------------------------
#  Prediction table
# ---------------------------------------------------------------------------
def test_prediction_table_is_well_formed():
    seqs = _simulate(n_patients=12, seed=43)
    fit = cs.fit_em(seqs, 3, seed=1, max_iter=20)
    fit.label_emission = cs.fit_label_emission(fit, seqs)
    preds = cs.next_visit_predictions(fit, seqs)

    assert len(preds) == sum(len(s["gaps"]) for s in seqs)
    assert (preds["n_history"] == preds["visit_index"] + 1).all()
    assert (preds["target_years"] > preds["current_years"]).all()
    assert np.allclose(preds["interval_years"],
                       preds["target_years"] - preds["current_years"])

    state_cols = [f"p_state_{s}" for s in range(3)]
    assert np.allclose(preds[state_cols].sum(axis=1), 1.0)
    label_cols = [f"p_{s}" for s in cs.STATE_NAMES]
    assert np.allclose(preds[label_cols].sum(axis=1), 1.0)
    assert set(preds["actual_next_label"]) <= set(cs.STATE_NAMES)


def test_latent_tie_resolves_to_the_lowest_state_index():
    assert cs._argmax_lowest_index(np.array([0.5, 0.5, 0.0])) == 0
    assert cs._argmax_lowest_index(np.array([0.1, 0.45, 0.45])) == 1
    assert cs._argmax_lowest_index(np.array([0.2, 0.3, 0.5])) == 2


def test_label_probabilities_require_a_fitted_link():
    fit = cs.CTHMMGaussian(Q=np.array([[-0.1, 0.1], [0.1, -0.1]]),
                           mu=np.zeros((2, 2)), var=np.ones((2, 2)),
                           pi=np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="no label emission"):
        fit.predict_next_label_probs(np.array([0.5, 0.5]))


def test_label_emission_is_a_proper_conditional_distribution():
    seqs = _simulate(n_patients=30, seed=47)
    fit = cs.fit_em(seqs, 3, seed=1, max_iter=20)
    B = cs.fit_label_emission(fit, seqs)
    assert B.shape == (3, len(cs.STATE_NAMES))
    assert np.allclose(B.sum(axis=1), 1.0)
    assert (B > 0).all()                       # smoothing keeps every label possible


# ---------------------------------------------------------------------------
#  Interoperating with the subtype-history experiment
# ---------------------------------------------------------------------------
def test_two_class_restriction_matches_the_baseline_machinery():
    """P_TD/P_PIGD must come from the previous experiment's own functions."""
    seqs = _simulate(n_patients=15, seed=53)
    fit = cs.fit_em(seqs, 3, seed=1, max_iter=20)
    fit.label_emission = cs.fit_label_emission(fit, seqs)
    preds = cs.next_visit_predictions(fit, seqs)

    probs = preds[[f"p_{s}" for s in cs.STATE_NAMES]].to_numpy(float)
    p_td, p_pigd, defined = nv.conditional_two_class(probs)
    assert defined.all()
    assert np.allclose(p_td + p_pigd, 1.0, atol=1e-12)
    assert ((p_td >= 0) & (p_td <= 1)).all()
    predicted = nv.predicted_state(p_td, p_pigd)
    assert set(predicted) <= {"TD", "PIGD"}


def test_state_names_match_the_baseline_ordering():
    """The two experiments must agree on which column is which label."""
    assert cs.STATE_NAMES == nv.STATE_NAMES
    assert cs.STATE_NAMES == ["TD", "Indeterminate", "PIGD"]


# ---------------------------------------------------------------------------
#  Reporting helpers
# ---------------------------------------------------------------------------
def test_canonical_order_is_by_burden_and_reorder_is_consistent():
    seqs = _simulate(n_patients=25, seed=59)
    fit = cs.fit_em(seqs, 3, seed=1, max_iter=20)
    fit.label_emission = cs.fit_label_emission(fit, seqs)
    std = cs.Standardiser.fit(seqs)

    order = cs.canonical_order(fit, std)
    ordered = cs.reorder(fit, order)

    sign = cs.burden_signs(fit.n_features)
    burden = (ordered.mu * sign[None, :]).sum(axis=1)
    assert np.all(np.diff(burden) >= -1e-9)

    # Reordering must permute every block consistently, not just the means.
    assert np.allclose(ordered.Q.sum(axis=1), 0.0, atol=1e-8)
    assert np.allclose(ordered.pi.sum(), 1.0)
    assert np.allclose(ordered.label_emission.sum(axis=1), 1.0)
    assert np.allclose(np.sort(ordered.pi), np.sort(fit.pi))


def test_describe_states_reconciles_with_the_visit_count():
    seqs = _simulate(n_patients=20, seed=61)
    fit = cs.fit_em(seqs, 3, seed=1, max_iter=20)
    std = cs.Standardiser.fit(seqs)
    described = cs.describe_states(fit, seqs, std)

    profile = described["profile"]
    assert len(profile) == 3
    n_visits = sum(len(s["times"]) for s in seqs)
    assert int(profile["n_visits_hard"].sum()) == n_visits
    assert np.isclose(profile["share_visits_hard"].sum(), 1.0)
    assert np.isclose(profile["soft_occupancy"].sum(), 1.0, atol=1e-6)

    unc = described["uncertainty"].set_index("metric")["value"]
    assert 0.0 <= unc["mean max posterior"] <= 1.0
    assert unc["mean posterior entropy (nats)"] <= unc["max possible entropy (nats)"] + 1e-9

    annual = described["annual_transition"].drop(columns="state").to_numpy(float)
    assert np.allclose(annual.sum(axis=1), 1.0)


def test_predictive_score_counts_what_it_scores():
    seqs = _simulate(n_patients=15, seed=67)
    fit = cs.fit_em(seqs, 3, seed=1, max_iter=20)
    score = cs.predictive_score(fit, seqs)
    assert score["n_predictions"] == sum(len(s["gaps"]) for s in seqs)
    assert score["n_patients"] == 15
    assert np.isfinite(score["mean_next_obs_loglik"])


def test_patient_folds_are_balanced_and_deterministic():
    patients = list(range(53))
    a = cs.patient_folds(patients, 4, seed=3)
    b = cs.patient_folds(patients, 4, seed=3)
    assert a == b
    sizes = pd.Series(list(a.values())).value_counts()
    assert sizes.max() - sizes.min() <= 1
    assert set(a.values()) == {0, 1, 2, 3}


def test_burden_signs_only_applies_the_project_signs_at_the_project_width():
    """A different feature width must not silently inherit the ten signs."""
    full = cs.burden_signs(len(cs.FEATURES))
    assert full.shape == (len(cs.FEATURES),)
    moca = cs.FEATURE_NAMES.index("moca")
    assert full[moca] == -1.0                  # higher MoCA is better
    assert set(np.unique(full)) == {-1.0, 1.0}

    # Any other width gets all-ones rather than a broadcast error or a
    # misaligned sign vector.
    assert np.array_equal(cs.burden_signs(4), np.ones(4))
    assert np.array_equal(cs.burden_signs(13), np.ones(13))


# ---------------------------------------------------------------------------
#  The batched E-step must be the same arithmetic as the reference loop
# ---------------------------------------------------------------------------
def _reference_estep(fit, sequences, P_by_gap, gap_index_list):
    """The per-sequence E-step, accumulated the way `_fit_em_from` does."""
    k, d = fit.n_states, fit.n_features
    acc = {"loglik": 0.0, "pi": np.zeros(k), "gamma_sum": np.zeros(k),
           "weight_obs": np.zeros((k, d)), "weighted_x": np.zeros((k, d)),
           "weighted_xx": np.zeros((k, d)),
           "counts_by_gap": np.zeros((P_by_gap.shape[0], k, k))}
    for seq, idx in zip(sequences, gap_index_list):
        gamma, xi, ll = cs._forward_backward(fit, seq, P_by_gap, idx)
        X = seq["X"]
        observed = np.isfinite(X)
        filled = np.where(observed, X, 0.0)
        acc["loglik"] += ll
        acc["pi"] += gamma[0]
        acc["gamma_sum"] += gamma.sum(axis=0)
        acc["weight_obs"] += gamma.T @ observed.astype(float)
        acc["weighted_x"] += gamma.T @ filled
        acc["weighted_xx"] += gamma.T @ (filled ** 2)
        for t, g in enumerate(idx):
            acc["counts_by_gap"][g] += xi[t]
    return acc


@pytest.mark.parametrize("missing", [0.0, 0.3])
def test_batched_estep_matches_the_reference(missing):
    """Vectorising the E-step must not change a single sufficient statistic.

    The batched path is an optimisation, and the only reason to trust it is
    that it reproduces the loop it replaces -- including the log-likelihood,
    the soft transition counts and the per-feature weights under missing data.
    """
    seqs = _simulate(n_patients=25, n_states=3, seed=71, missing=missing)
    fit = cs._initialise(seqs, 3, seed=2)

    batch = cs.prepare_batch(seqs)
    position = {g: i for i, g in enumerate(batch.unique_gaps)}
    idx_list = [np.array([position[g] for g in np.round(s["gaps"], 3)], dtype=int)
                for s in seqs]

    from ppmi_subtype import ctmc
    P_by_gap = ctmc.transition_probabilities(fit.Q, batch.unique_gaps)

    ref = _reference_estep(fit, seqs, P_by_gap, idx_list)
    got = cs.batched_estep(fit, batch, P_by_gap)

    assert np.isclose(got["loglik"], ref["loglik"], rtol=1e-9, atol=1e-7)
    for key in ("pi", "gamma_sum", "weight_obs", "weighted_x", "weighted_xx",
                "counts_by_gap"):
        assert np.allclose(got[key], ref[key], rtol=1e-8, atol=1e-9), key


def test_batch_emission_matches_per_sequence_emission():
    seqs = _simulate(n_patients=8, seed=73, missing=0.2)
    fit = cs._initialise(seqs, 3, seed=1)
    batch = cs.prepare_batch(seqs)
    got = cs._batch_emission_loglik(fit, batch)
    for i, s in enumerate(seqs):
        n = len(s["times"])
        assert np.allclose(got[i, :n], fit.emission_loglik(s["X"]), atol=1e-9)


def test_prepare_batch_pads_without_losing_anything():
    seqs = _simulate(n_patients=10, seed=79)
    batch = cs.prepare_batch(seqs)
    assert batch.n_sequences == 10
    assert batch.n_observations == sum(len(s["times"]) for s in seqs)
    assert batch.visit_mask.sum() == batch.n_observations
    assert batch.step_mask.sum() == sum(len(s["gaps"]) for s in seqs)
    for i, s in enumerate(seqs):
        n = len(s["times"])
        assert np.array_equal(batch.X[i, :n], s["X"], equal_nan=True)
        assert not batch.visit_mask[i, n:].any()


# ---------------------------------------------------------------------------
#  The diagnostic script's case construction (script 63)
# ---------------------------------------------------------------------------
def _load_direct_script():
    import sys
    from importlib import import_module
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    return import_module("63_clinical_features_direct")


def test_direct_diagnostic_cases_are_causal():
    """Every feature column must come from the current visit, never the next."""
    mod = _load_direct_script()
    seqs = _simulate(n_patients=6, n_features=len(cs.FEATURE_NAMES), seed=83)
    cases = mod.build_cases(seqs)

    assert len(cases) == sum(len(s["gaps"]) for s in seqs)
    assert (cases["n_history"] == cases["visit_index"] + 1).all()
    assert (cases["target_years"] > cases["current_years"]).all()
    assert np.allclose(cases["interval_years"],
                       cases["target_years"] - cases["current_years"])
    assert np.allclose(cases["log_interval"], np.log(cases["interval_years"]))

    # Each feature value must equal the CURRENT visit's measurement.
    for s in seqs:
        g = cases[cases["PATNO"] == s["PATNO"]].sort_values("visit_index")
        for row_i, (_, row) in enumerate(g.iterrows()):
            for j, name in enumerate(cs.FEATURE_NAMES):
                assert row[name] == pytest.approx(s["X"][row_i, j], nan_ok=True), name

    # The target is the next visit's label, and only eligible rows are scorable.
    assert set(cases["actual_next_label"]) <= set(cs.STATE_NAMES)
    assert (cases.loc[cases["eligible"], "actual_next_label"]
            .isin(["TD", "PIGD"]).all())
    assert not cases.loc[~cases["eligible"], "eligible"].any()


def test_direct_diagnostic_fill_uses_training_medians_only():
    """The median fill must be a training statistic, not a pooled one."""
    mod = _load_direct_script()
    seqs = _simulate(n_patients=20, n_features=len(cs.FEATURE_NAMES),
                     seed=89, missing=0.2)
    cases = mod.build_cases(seqs)
    features = ("tremor", "pigd", "log_interval")

    train = cases[cases["PATNO"] < 10]
    test = cases[cases["PATNO"] >= 10]
    tr, te = mod.fill_missing(train, [train, test], features)

    assert not tr[list(features)].isna().any().any()
    assert not te[list(features)].isna().any().any()

    # Filled test values must equal the TRAINING median, not the pooled one.
    med_train = train[list(features)].median()
    med_all = cases[list(features)].median()
    for c in features:
        was_missing = test[c].isna().to_numpy()
        if was_missing.any():
            filled = te[c].to_numpy()[was_missing]
            assert np.allclose(filled, med_train[c])
            if not np.isclose(med_train[c], med_all[c]):
                assert not np.allclose(filled, med_all[c])
