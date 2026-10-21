"""Correctness checks for the rich clinical-profile representation.

The properties tested are the ones the experiment's conclusions rest on:
temporal causality of the filter, patient-level separation of every learned
transformation, exact handling of missing measurements, recovery of known
dynamics, and that the derived-label bridge uses the project's own rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype import clinical_profile as cp
from ppmi_subtype import clinical_states as cs
from ppmi_subtype.config import load_config


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _simulate(n_patients=40, seed=0, missing=0.0, slope_sd=0.4,
              obs_sd=0.3, n_domains=None):
    """Draw sequences from a known level+rate process, on the model's scale."""
    D = n_domains if n_domains is not None else len(cp.DOMAINS)
    rng = np.random.default_rng(seed)
    sequences = []
    for p in range(n_patients):
        n = int(rng.integers(3, 7))
        gaps = rng.uniform(0.8, 1.5, n - 1)
        times = np.cumsum(np.r_[0.0, gaps])
        level0 = rng.normal(0, 1.0, D)
        rate = rng.normal(0, slope_sd, D)
        truth = level0[None, :] + times[:, None] * rate[None, :]
        Y = truth + rng.normal(0, obs_sd, truth.shape)
        if missing:
            Y[rng.uniform(size=Y.shape) < missing] = np.nan
        sequences.append({
            "PATNO": p, "Y": Y, "d": np.zeros(n), "times": times,
            "gaps": np.diff(times),
            "labels": rng.choice([1, 2, 3], size=n, p=[0.6, 0.1, 0.3]),
            "events": np.array([f"V{t:02d}" for t in range(n)]),
            "visit_index": np.arange(n),
            "_truth_rate": rate,
        })
    return sequences


def _model(D=None, obs=0.1, level=0.03, rate=0.02, beta=0.0,
           init_rate=None, **kw):
    D = D if D is not None else len(cp.DOMAINS)
    return cp.ProfileModel(
        obs_var=np.full(D, obs), level_var=np.full(D, level),
        rate_var=np.full(D, rate), beta=np.full(D, beta),
        init_rate_var=(None if init_rate is None else np.full(D, init_rate)),
        **kw)


# ---------------------------------------------------------------------------
#  Domains and configuration
# ---------------------------------------------------------------------------
def test_medication_is_an_input_not_a_domain():
    """The central design change: LEDD must not be a predicted coordinate."""
    assert cp.MEDICATION == "ledd"
    assert "ledd" not in cp.DOMAINS
    assert len(cp.DOMAINS) == 9
    # Every domain is one of the previous experiment's features.
    assert set(cp.DOMAINS) < set(cs.FEATURE_NAMES)
    assert set(cs.FEATURE_NAMES) - set(cp.DOMAINS) == {"ledd"}


# ---------------------------------------------------------------------------
#  Temporal causality
# ---------------------------------------------------------------------------
def test_filter_is_causal_in_every_domain():
    """Changing a later visit cannot move the filtered state at an earlier one."""
    seqs = _simulate(n_patients=4, seed=3)
    model = _model()
    seq = seqs[0]

    a = model.filter_sequence(seq)
    tampered = dict(seq, Y=seq["Y"].copy())
    tampered["Y"][-1] += 25.0
    b = model.filter_sequence(tampered)

    assert np.allclose(a["level"][:-1], b["level"][:-1])
    assert np.allclose(a["rate"][:-1], b["rate"][:-1])
    assert np.allclose(a["level_var"][:-1], b["level_var"][:-1])
    # The final step must respond, or the filter is ignoring data entirely.
    assert not np.allclose(a["level"][-1], b["level"][-1])


def test_one_step_predictions_precede_their_observation():
    """pred_mean[k] must not depend on y[k] -- it is the prior prediction."""
    seqs = _simulate(n_patients=3, seed=5)
    model = _model()
    seq = seqs[0]
    k = 2

    a = model.filter_sequence(seq)
    tampered = dict(seq, Y=seq["Y"].copy())
    tampered["Y"][k] += 30.0
    b = model.filter_sequence(tampered)

    assert np.allclose(a["pred_mean"][: k + 1], b["pred_mean"][: k + 1]), \
        "a one-step-ahead prediction saw its own target"
    assert not np.allclose(a["pred_mean"][k + 1], b["pred_mean"][k + 1])


def test_predict_next_ignores_the_target_visit():
    """The end-to-end property: the prediction FOR visit k+1 cannot see it."""
    seqs = _simulate(n_patients=6, seed=7)
    model = _model()
    std = cp.Standardiser.fit(seqs)
    seq = seqs[0]
    k = 1

    f = model.filter_sequence(seq)
    dt = seq["times"][k + 1] - seq["times"][k]
    mean_a, var_a = model.predict_next(f, k, dt, seq["d"][k + 1])

    tampered = dict(seq, Y=seq["Y"].copy())
    tampered["Y"][k + 1] += 40.0
    f2 = model.filter_sequence(tampered)
    mean_b, var_b = model.predict_next(f2, k, dt, tampered["d"][k + 1])

    assert np.allclose(mean_a, mean_b)
    assert np.allclose(var_a, var_b)


def test_prediction_records_never_read_the_future():
    seqs = _simulate(n_patients=8, seed=11)
    model = _model()
    std = cp.Standardiser.fit(seqs)
    recs = cp.predict_sequence(model, seqs[0], std, models=(cp.MODEL_FULL,))
    frame = cp.records_to_frame(recs)

    assert len(frame) == len(seqs[0]["times"]) - 1
    assert (frame["n_history"] == frame["visit_index"] + 1).all()
    assert (frame["target_years"] > frame["current_years"]).all()
    assert np.allclose(frame["interval_years"],
                       frame["target_years"] - frame["current_years"])


# ---------------------------------------------------------------------------
#  Elapsed time
# ---------------------------------------------------------------------------
def test_predictive_variance_grows_with_elapsed_time():
    """A longer gap must mean a less certain prediction."""
    seqs = _simulate(n_patients=3, seed=13)
    model = _model()
    f = model.filter_sequence(seqs[0])
    _, v_short = model.predict_next(f, 1, 0.5, 0.0)
    _, v_long = model.predict_next(f, 1, 5.0, 0.0)
    assert (v_long > v_short).all()


def test_rate_extrapolates_and_level_only_does_not():
    """The rate component must actually extrapolate a trend."""
    rng = np.random.default_rng(17)
    n = 5
    times = np.arange(n, dtype=float)
    D = 2
    Y = (np.zeros(D)[None, :] + times[:, None] * np.full(D, 0.5)[None, :])
    seq = {"PATNO": 0, "Y": Y, "d": np.zeros(n), "times": times,
           "gaps": np.diff(times), "labels": np.ones(n, dtype=int),
           "events": np.array([f"V{i}" for i in range(n)]),
           "visit_index": np.arange(n)}

    full = _model(D=D, obs=0.01, level=1e-6, rate=0.05, use_rate=True)
    flat = _model(D=D, obs=0.01, level=0.05, use_rate=False)

    f_full = full.filter_sequence(seq)
    f_flat = flat.filter_sequence(seq)
    m_full, _ = full.predict_next(f_full, n - 1, 1.0, 0.0)
    m_flat, _ = flat.predict_next(f_flat, n - 1, 1.0, 0.0)

    # Truth at the next step is 5*0.5 = 2.5; last observation is 2.0.
    assert np.all(m_full > m_flat + 0.2), (m_full, m_flat)
    assert np.all(np.abs(m_full - 2.5) < 0.2)
    # The level-only model must NOT extrapolate the trend: it stays near the
    # last observation. It does not land exactly on it, because a random walk
    # observed with noise shrinks towards its prior rather than interpolating
    # the last point -- which is exactly why plain last-observation-carried-
    # forward is kept as a separate baseline rather than being represented by
    # this variant.
    assert np.all(np.abs(m_flat - Y[-1]) < 0.15), m_flat
    assert np.all(m_flat < 2.1)


def test_level_only_model_pins_the_rate_at_zero():
    seqs = _simulate(n_patients=5, seed=19)
    flat = _model(use_rate=False)
    f = flat.filter_sequence(seqs[0])
    assert np.allclose(f["rate"], 0.0)
    assert np.allclose(f["rate_var"], 0.0)


# ---------------------------------------------------------------------------
#  Missing data
# ---------------------------------------------------------------------------
def test_missing_measurement_leaves_the_state_at_its_prediction():
    """Exact marginalisation: an unobserved domain contributes no update."""
    D = 3
    n = 4
    times = np.arange(n, dtype=float)
    Y = np.zeros((n, D))
    Y[2, 1] = np.nan                      # domain 1 missing at visit 2
    seq = {"PATNO": 0, "Y": Y, "d": np.zeros(n), "times": times,
           "gaps": np.diff(times), "labels": np.ones(n, dtype=int),
           "events": np.array([f"V{i}" for i in range(n)]),
           "visit_index": np.arange(n)}
    model = _model(D=D, obs=0.2, level=0.05, rate=0.05)
    f = model.filter_sequence(seq)

    # At the missing visit, the filtered level of that domain must equal the
    # predicted observation (no medication, so prediction == level).
    assert np.isclose(f["level"][2, 1], f["pred_mean"][2, 1])
    # And its variance must have grown relative to the previous visit, since
    # no information arrived.
    assert f["level_var"][2, 1] > f["level_var"][1, 1]


def test_missing_values_do_not_enter_the_likelihood():
    seqs = _simulate(n_patients=6, seed=23)
    model = _model()
    full_ll = sum(model.filter_sequence(s)["loglik"] for s in seqs)

    holed = []
    for s in seqs:
        Y = s["Y"].copy()
        Y[1, 0] = np.nan
        holed.append(dict(s, Y=Y))
    holed_ll = sum(model.filter_sequence(s)["loglik"] for s in holed)
    # Removing observations can only remove terms, so the likelihood of the
    # remaining data must be strictly greater (fewer negative terms).
    assert holed_ll > full_ll
    assert np.isfinite(holed_ll)


def test_a_fully_missing_visit_is_tolerated():
    D = 3
    n = 4
    Y = np.zeros((n, D))
    Y[1, :] = np.nan
    seq = {"PATNO": 0, "Y": Y, "d": np.zeros(n), "times": np.arange(n, dtype=float),
           "gaps": np.ones(n - 1), "labels": np.ones(n, dtype=int),
           "events": np.array([f"V{i}" for i in range(n)]),
           "visit_index": np.arange(n)}
    f = _model(D=D).filter_sequence(seq)
    assert np.isfinite(f["loglik"])
    assert np.isfinite(f["level"]).all()


def test_locf_carries_the_last_recorded_value_not_the_last_slot():
    Y = np.array([[1.0, 5.0], [np.nan, 6.0], [3.0, np.nan]])
    out, fell = cp._last_observed(Y, 2)
    assert out[0] == 3.0          # domain 0 observed at visit 2
    assert out[1] == 6.0          # domain 1 last observed at visit 1
    assert not fell.any()
    out1, _ = cp._last_observed(Y, 1)
    assert out1[0] == 1.0         # not the NaN at visit 1
    assert out1[1] == 6.0


def test_locf_falls_back_when_a_domain_was_never_recorded():
    """LOCF must not emit NaN: that would drop cases from its own score."""
    Y = np.array([[np.nan, 5.0], [np.nan, 6.0]])
    out, fell = cp._last_observed(Y, 1, fallback=np.array([-0.5, 99.0]))
    assert out[0] == -0.5         # never recorded -> training fallback
    assert out[1] == 6.0          # recorded -> carried forward
    assert fell.tolist() == [True, False]
    assert np.isfinite(out).all()

    # Without a fallback the gap is reported rather than hidden.
    bare, fell2 = cp._last_observed(Y, 1)
    assert np.isnan(bare[0]) and fell2[0]


# ---------------------------------------------------------------------------
#  Patient-level separation of learned transformations
# ---------------------------------------------------------------------------
def test_standardiser_uses_training_patients_only():
    seqs = _simulate(n_patients=30, seed=29)
    train, test = seqs[:20], seqs[20:]
    std = cp.Standardiser.fit(train)

    Y_train = np.vstack([s["Y"] for s in train])
    assert np.allclose(std.centre, np.nanmean(Y_train, axis=0))
    assert std.n_train_visits == len(Y_train)

    Y_all = np.vstack([s["Y"] for s in seqs])
    assert not np.allclose(std.centre, np.nanmean(Y_all, axis=0))

    z = std.apply(test)
    for a, b in zip(test, z):
        assert np.array_equal(np.isnan(a["Y"]), np.isnan(b["Y"]))


def test_standardiser_scales_but_never_centres_medication():
    """Zero LEDD must stay zero, or an untreated visit gets a drug effect."""
    seqs = _simulate(n_patients=10, seed=31)
    for s in seqs:
        s["d"] = np.linspace(0.0, 600.0, len(s["times"]))
    std = cp.Standardiser.fit(seqs)
    z = std.apply(seqs)
    for a, b in zip(seqs, z):
        zero = a["d"] == 0.0
        assert np.all(b["d"][zero] == 0.0)
    assert std.med_scale > 1.0


def test_standardiser_round_trips():
    seqs = _simulate(n_patients=12, seed=37)
    std = cp.Standardiser.fit(seqs)
    z = std.apply(seqs)
    assert np.allclose(std.invert_mean(z[0]["Y"]), seqs[0]["Y"], equal_nan=True)


def test_population_mean_baseline_is_fitted_on_training_only():
    seqs = _simulate(n_patients=30, seed=41)
    train, test = seqs[:20], seqs[20:]
    popn = cp.PopulationMean.fit(train)

    Y_train = np.vstack([s["Y"] for s in train])
    assert np.allclose(popn.overall, np.nanmean(Y_train, axis=0))
    # The prediction depends only on time, never on the patient.
    assert np.allclose(popn.predict(1.2), popn.predict(1.4))
    assert not np.allclose(popn.predict(0.5), popn.predict(9.5))


def test_locf_spread_is_fitted_on_training_only():
    seqs = _simulate(n_patients=25, seed=43)
    sd_train = cp.locf_sd(seqs[:15])
    sd_all = cp.locf_sd(seqs)
    assert sd_train.shape == (len(cp.DOMAINS),)
    assert (sd_train > 0).all()
    assert not np.allclose(sd_train, sd_all)


# ---------------------------------------------------------------------------
#  Fitting
# ---------------------------------------------------------------------------
def test_fit_recovers_a_known_observation_variance():
    seqs = _simulate(n_patients=120, seed=47, obs_sd=0.4, slope_sd=0.3,
                     n_domains=3)
    model = cp.fit(seqs, use_rate=True, use_medication=False)
    assert model.converged or np.isfinite(model.loglik)
    # True observation variance is 0.16.
    assert np.all(np.abs(model.obs_var - 0.16) < 0.12), model.obs_var


def test_fit_recovers_a_known_medication_effect():
    """A planted LEDD effect must be recovered with the right sign and size."""
    rng = np.random.default_rng(53)
    D = 3
    true_beta = np.array([-0.8, 0.0, 0.5])
    seqs = []
    for p in range(150):
        n = 6
        times = np.arange(n, dtype=float)
        d = np.where(times >= 2, 1.5, 0.0)            # starts medication at t=2
        level = rng.normal(0, 1.0, D)
        Y = level[None, :] + d[:, None] * true_beta[None, :] \
            + rng.normal(0, 0.2, (n, D))
        seqs.append({"PATNO": p, "Y": Y, "d": d, "times": times,
                     "gaps": np.diff(times), "labels": np.ones(n, dtype=int),
                     "events": np.array([f"V{i}" for i in range(n)]),
                     "visit_index": np.arange(n)})

    model = cp.fit(seqs, use_rate=True, use_medication=True)
    assert np.all(np.abs(model.beta - true_beta) < 0.25), model.beta


def test_fit_is_deterministic():
    seqs = _simulate(n_patients=30, seed=59, n_domains=3)
    a = cp.fit(seqs, use_medication=False)
    b = cp.fit(seqs, use_medication=False)
    assert a.loglik == b.loglik
    assert np.array_equal(a.obs_var, b.obs_var)


def test_no_medication_variant_ignores_the_input():
    seqs = _simulate(n_patients=20, seed=61, n_domains=3)
    for s in seqs:
        s["d"] = np.linspace(0, 2, len(s["times"]))
    model = cp.fit(seqs, use_medication=False)
    assert np.allclose(model.beta, 0.0)
    shifted = [dict(s, d=s["d"] + 7.0) for s in seqs]
    a = sum(model.filter_sequence(s)["loglik"] for s in seqs)
    b = sum(model.filter_sequence(s)["loglik"] for s in shifted)
    assert np.isclose(a, b)


def test_rate_model_beats_level_only_when_a_trend_exists():
    """Sanity: with real trends, modelling the rate must fit better."""
    seqs = _simulate(n_patients=100, seed=67, slope_sd=0.5, obs_sd=0.2,
                     n_domains=3)
    with_rate = cp.fit(seqs, use_rate=True, use_medication=False)
    without = cp.fit(seqs, use_rate=False, use_medication=False)
    assert with_rate.loglik > without.loglik


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------
def _toy_frame(n=40, seed=71, bias=0.0, sd=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        row = {"PATNO": i // 4, "visit_index": i % 4, "model": "m",
               "n_history": (i % 4) + 1}
        for dom in cp.DOMAINS:
            a = rng.normal()
            row[f"z_actual_{dom}"] = a
            row[f"z_pred_{dom}"] = a + bias
            row[f"z_sd_{dom}"] = sd
            row[f"actual_{dom}"] = a
            row[f"pred_{dom}"] = a + bias
            row[f"sd_{dom}"] = sd
        rows.append(row)
    return pd.DataFrame(rows)


def test_profile_metrics_reward_a_better_prediction():
    good = cp.profile_metrics(_toy_frame(bias=0.0), n_boot=100)
    bad = cp.profile_metrics(_toy_frame(bias=1.5), n_boot=100)
    assert good["loglik_per_domain_observation"] > bad["loglik_per_domain_observation"]
    assert good["rmse_standardised"] < bad["rmse_standardised"]
    assert good["n_domain_observations"] == 40 * len(cp.DOMAINS)


def test_profile_metrics_coverage_is_calibrated_when_the_sd_is_right():
    rng = np.random.default_rng(73)
    rows = []
    for i in range(400):
        row = {"PATNO": i // 4, "visit_index": i % 4, "model": "m", "n_history": 1}
        for dom in cp.DOMAINS:
            a = rng.normal()
            row[f"z_actual_{dom}"] = a
            row[f"z_pred_{dom}"] = 0.0
            row[f"z_sd_{dom}"] = 1.0
            row[f"actual_{dom}"] = a
            row[f"pred_{dom}"] = 0.0
            row[f"sd_{dom}"] = 1.0
        rows.append(row)
    m = cp.profile_metrics(pd.DataFrame(rows), n_boot=50)
    assert abs(m["coverage_95"] - 0.95) < 0.02


def test_profile_metrics_skip_missing_domains():
    frame = _toy_frame(n=10)
    frame.loc[0, "z_actual_brady_rigid"] = np.nan
    frame.loc[0, "actual_brady_rigid"] = np.nan
    m = cp.profile_metrics(frame, n_boot=50)
    assert m["n_domain_observations"] == 10 * len(cp.DOMAINS) - 1


def test_paired_loglik_difference_is_paired_and_signed():
    a = _toy_frame(bias=0.0)
    b = _toy_frame(bias=1.0)
    b["model"] = "n"
    r = cp.paired_loglik_difference(a, b, n_boot=200)
    assert r["n_paired"] == len(a)
    assert r["loglik_delta"] > 0            # a is better
    assert r["share_a_better"] == 1.0
    assert r["delta_ci_low"] > 0

    same = cp.paired_loglik_difference(a, a.copy(), n_boot=100)
    assert np.isclose(same["loglik_delta"], 0.0)


# ---------------------------------------------------------------------------
#  The derived-label bridge
# ---------------------------------------------------------------------------
def test_derived_label_uses_the_project_rule_and_sums_to_one():
    cfg = load_config()
    frame = pd.DataFrame({
        # ratio 2.0 -> clearly TD; ratio 0.5 -> clearly PIGD; ratio 1.0 -> band
        "pred_tremor": [1.0, 0.5, 1.0],
        "sd_tremor": [1e-6, 1e-6, 1e-6],
        "pred_pigd": [0.5, 1.0, 1.0],
        "sd_pigd": [1e-6, 1e-6, 1e-6],
    })
    out = cp.derive_label_probabilities(frame, cfg, n_draws=200, seed=1)
    cols = [f"p_{s}" for s in cs.STATE_NAMES]
    assert np.allclose(out[cols].sum(axis=1), 1.0)
    assert out.loc[0, "p_TD"] > 0.99
    assert out.loc[1, "p_PIGD"] > 0.99
    assert out.loc[2, "p_Indeterminate"] > 0.99


def test_derived_label_uncertainty_reflects_predictive_spread():
    """A prediction near the ratio threshold must give an uncertain label."""
    cfg = load_config()
    # Row 0 sits well inside TD (ratio 1.5 against a 1.15 threshold) with a
    # tight predictive spread; row 1 sits at the same place with a wide one.
    # A ratio sitting exactly ON the threshold is deliberately not used here:
    # a 50/50 split is the correct answer there, which would test nothing.
    frame = pd.DataFrame({
        "pred_tremor": [1.5, 1.5],
        "sd_tremor": [1e-6, 0.6],
        "pred_pigd": [1.0, 1.0],
        "sd_pigd": [1e-6, 0.6],
    })
    out = cp.derive_label_probabilities(frame, cfg, n_draws=4000, seed=2)
    sharp = out.loc[0, ["p_TD", "p_Indeterminate", "p_PIGD"]].to_numpy(float)
    diffuse = out.loc[1, ["p_TD", "p_Indeterminate", "p_PIGD"]].to_numpy(float)

    def entropy(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())
    assert entropy(diffuse) > entropy(sharp)
    assert sharp.max() > 0.9


# ---------------------------------------------------------------------------
#  Representation diagnostics
# ---------------------------------------------------------------------------
def test_representation_frame_shape_and_units():
    seqs = _simulate(n_patients=10, seed=79)
    std = cp.Standardiser.fit(seqs)
    model = _model()
    rep = cp.representation_frame(model, std.apply(seqs), std)

    assert len(rep) == sum(len(s["times"]) for s in seqs)
    for dom in cp.DOMAINS:
        assert f"level_{dom}" in rep.columns
        assert f"rate_{dom}" in rep.columns
    assert rep["subtype"].isin(cs.STATE_NAMES).all()


def test_dimensionality_detects_a_one_axis_representation():
    """H2's own check must fire when the representation IS a severity axis."""
    rng = np.random.default_rng(83)
    n = 200
    axis = rng.normal(size=n)
    rows = []
    for i in range(n):
        row = {"PATNO": i, "visit_index": 0, "years_from_baseline": 0.0,
               "n_history": 1, "subtype": "TD", "ledd": 0.0}
        for dom in cp.DOMAINS:
            row[f"level_{dom}"] = axis[i] + rng.normal(0, 0.01)
            row[f"rate_{dom}"] = rng.normal()
        rows.append(row)
    rep = pd.DataFrame(rows)

    one_axis = cp.representation_dimensionality(rep, which="level")
    assert one_axis["pc1_share"] > 0.9          # collapses to one axis
    assert one_axis["n_components_for_90pct"] == 1

    independent = cp.representation_dimensionality(rep, which="rate")
    assert independent["pc1_share"] < 0.4       # genuinely multidimensional
    assert independent["n_components_for_90pct"] > 5


def test_rate_spread_reports_per_domain_heterogeneity():
    seqs = _simulate(n_patients=40, seed=89, slope_sd=0.6)
    std = cp.Standardiser.fit(seqs)
    model = cp.fit(std.apply(seqs), use_medication=False)
    rep = cp.representation_frame(model, std.apply(seqs), std)
    spread = cp.rate_spread(rep)

    assert len(spread) == len(cp.DOMAINS)
    assert (spread["n_patients"] == 40).all()
    assert (spread["sd_rate_per_year"] > 0).all()
    assert ((spread["share_worsening"] >= 0) & (spread["share_worsening"] <= 1)).all()


# ---------------------------------------------------------------------------
#  AUC utilities
# ---------------------------------------------------------------------------
def test_auc_basic_properties():
    assert cp.auc(np.array([0.1, 0.2, 0.3, 0.4]),
                  np.array([0, 0, 1, 1])) == 1.0
    assert cp.auc(np.array([0.4, 0.3, 0.2, 0.1]),
                  np.array([0, 0, 1, 1])) == 0.0
    # A constant score must be exactly chance, not 1.0.
    assert cp.auc(np.ones(6), np.array([0, 1, 0, 1, 0, 1])) == 0.5
    assert np.isnan(cp.auc(np.array([1.0, 2.0]), np.array([0, 0])))


def test_bootstrap_auc_difference_signs_correctly():
    rng = np.random.default_rng(97)
    n = 300
    y = rng.integers(0, 2, n).astype(bool)
    good = y + rng.normal(0, 0.5, n)
    poor = y + rng.normal(0, 3.0, n)
    r = cp.bootstrap_auc_difference(good, poor, y, n_boot=300)
    assert r["auc_a"] > r["auc_b"]
    assert r["delta"] > 0
    assert r["delta_ci_low"] > 0


def test_derived_label_at_the_threshold_is_genuinely_split():
    """A ratio exactly on the TD boundary must not be reported as confident."""
    cfg = load_config()
    frame = pd.DataFrame({
        "pred_tremor": [1.15], "sd_tremor": [1e-6],
        "pred_pigd": [1.0], "sd_pigd": [1e-6],
    })
    out = cp.derive_label_probabilities(frame, cfg, n_draws=4000, seed=3)
    p = out.loc[0, ["p_TD", "p_Indeterminate", "p_PIGD"]].to_numpy(float)
    assert p.max() < 0.75, p          # no false confidence at a knife edge
    assert np.isclose(p.sum(), 1.0)


# ---------------------------------------------------------------------------
#  The batched filter must be the same arithmetic as the reference loop
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("missing", [0.0, 0.3])
@pytest.mark.parametrize("use_rate,use_medication",
                         [(True, True), (True, False), (False, True)])
def test_batched_filter_matches_the_reference(missing, use_rate, use_medication):
    """Vectorising must not change the likelihood or any filtered quantity."""
    seqs = _simulate(n_patients=23, seed=101, missing=missing, n_domains=4)
    rng = np.random.default_rng(7)
    for s in seqs:
        s["d"] = np.where(s["times"] >= 1.5, 1.2, 0.0)

    model = cp.ProfileModel(
        obs_var=np.array([0.2, 0.3, 0.15, 0.25]),
        level_var=np.array([0.05, 0.02, 0.08, 0.01]),
        rate_var=np.array([0.01, 0.04, 0.02, 0.03]),
        beta=np.array([-0.4, 0.1, 0.0, 0.3]),
        use_rate=use_rate, use_medication=use_medication)

    ref_ll = sum(model.filter_sequence(s)["loglik"] for s in seqs)
    batch = cp.prepare_batch(seqs)
    got_ll, got = cp.batched_filter(model, batch, collect=True)

    assert np.isclose(got_ll, ref_ll, rtol=1e-10, atol=1e-8)

    for i, s in enumerate(seqs):
        ref = model.filter_sequence(s)
        k = len(s["times"])
        for key in ("level", "rate", "level_var", "rate_var",
                    "level_rate_cov", "pred_mean", "pred_var"):
            assert np.allclose(got[key][i, :k], ref[key], rtol=1e-9,
                               atol=1e-10), (key, i)


def test_prepare_batch_loses_nothing():
    seqs = _simulate(n_patients=15, seed=103, n_domains=4)
    batch = cp.prepare_batch(seqs)
    assert batch.n_sequences == 15
    assert batch.n_domains == 4
    assert batch.n_observations == sum(len(s["times"]) for s in seqs)
    for i, s in enumerate(seqs):
        k = len(s["times"])
        assert np.array_equal(batch.Y[i, :k], s["Y"], equal_nan=True)
        assert np.allclose(batch.d[i, :k], s["d"])
        if k > 1:
            assert np.allclose(batch.dt[i, :k - 1], np.diff(s["times"]))
        assert not batch.visit_mask[i, k:].any()


def test_batched_filter_is_causal():
    """The causality property must hold on the fast path too."""
    seqs = _simulate(n_patients=6, seed=107, n_domains=4)
    model = cp.ProfileModel(obs_var=np.full(4, 0.2), level_var=np.full(4, 0.03),
                            rate_var=np.full(4, 0.02), beta=np.zeros(4),
                            use_medication=False)
    a_ll, a = cp.batched_filter(model, cp.prepare_batch(seqs), collect=True)

    tampered = [dict(s, Y=s["Y"].copy()) for s in seqs]
    tampered[0]["Y"][-1] += 30.0
    b_ll, b = cp.batched_filter(model, cp.prepare_batch(tampered), collect=True)

    k = len(seqs[0]["times"])
    assert np.allclose(a["level"][0, :k - 1], b["level"][0, :k - 1])
    assert np.allclose(a["pred_mean"][0, :k], b["pred_mean"][0, :k])
    # Other patients must be untouched: no leakage across the batch dimension.
    assert np.allclose(a["level"][1:], b["level"][1:])


def test_full_model_nests_the_random_walk():
    """With a zero rate innovation the trend model must BE the random walk.

    This is what makes the level-only contrast a real test of the rate
    component rather than a comparison of two unrelated parameterisations.
    """
    seqs = _simulate(n_patients=12, seed=109, n_domains=3)
    D = 3
    # Nesting holds in the limit where BOTH the rate innovation and the prior
    # variance on the initial rate vanish; that is exactly why init_rate_var is
    # a fitted parameter rather than a constant.
    nested = cp.ProfileModel(obs_var=np.full(D, 0.2), level_var=np.full(D, 0.07),
                             rate_var=np.zeros(D), beta=np.zeros(D),
                             init_rate_var=np.zeros(D),
                             use_rate=True, use_medication=False)
    walk = cp.ProfileModel(obs_var=np.full(D, 0.2), level_var=np.full(D, 0.07),
                           rate_var=np.zeros(D), beta=np.zeros(D),
                           use_rate=False, use_medication=False)
    for s in seqs:
        a = nested.filter_sequence(s)
        b = walk.filter_sequence(s)
        assert np.isclose(a["loglik"], b["loglik"], rtol=1e-9)
        assert np.allclose(a["level"], b["level"], atol=1e-9)


def test_fitted_full_model_is_essentially_never_worse_than_level_only():
    """With no real trend, the trend model must shrink back to the random walk.

    The adversarial case: data with no slope at all, where the extra rate
    component can only hurt. Because the nested solution sits on the *boundary*
    of the parameter space (both rate variances at their lower bound), L-BFGS-B
    stops near it rather than exactly on it, so an exact inequality cannot be
    asserted; the tolerance is relative and small.
    """
    seqs = _simulate(n_patients=80, seed=113, slope_sd=0.0, obs_sd=0.3,
                     n_domains=3)
    full = cp.fit(seqs, use_rate=True, use_medication=False)
    flat = cp.fit(seqs, use_rate=False, use_medication=False)

    gap = flat.loglik - full.loglik
    assert gap < 1e-3 * abs(flat.loglik), (full.loglik, flat.loglik, gap)
    # And it got there by shrinking the rate away, not by some other route.
    assert np.all(full.rate_var < 1e-3), full.rate_var
    assert np.all(full.init_rate_var < 1e-2), full.init_rate_var


# ---------------------------------------------------------------------------
#  H3 outcome table
# ---------------------------------------------------------------------------
def _outcome_panel():
    """A tiny panel carrying the outcome columns and three visits per patient."""
    rows = []
    for p in range(6):
        for k, t in enumerate([0.0, 1.0, 2.0, 3.0]):
            rows.append({
                "PATNO": p, "years_from_baseline": t,
                # H&Y reaches 3 only at the LAST visit, for patients 0-2.
                "hy": 3.0 if (p < 3 and k == 3) else 1.0,
                "MCI_testscores": 1.0 if (p == 0 and k == 3) else 0.0,
                "updrs4_score": 2.0 if (p < 2 and k >= 2) else 0.0,
                "cogstate": 1.0,
            })
    return pd.DataFrame(rows)


def _outcome_representation():
    rows = []
    for p in range(6):
        for k, t in enumerate([0.0, 1.0, 2.0, 3.0]):
            row = {"PATNO": p, "visit_index": k, "years_from_baseline": t,
                   "n_history": k + 1, "subtype": "PIGD" if p < 3 else "TD",
                   "ledd": 0.0, "repeat": 0, "fold": 0}
            for dom in cp.DOMAINS:
                row[f"level_{dom}"] = float(p)
                row[f"rate_{dom}"] = float(p) * 0.1
            rows.append(row)
    return pd.DataFrame(rows)


def test_outcome_table_only_counts_strictly_later_outcomes():
    """An outcome at or before the anchor visit must not count as an event."""
    table = cp.build_outcome_table(_outcome_panel(), _outcome_representation(),
                                   anchor_visit=1)
    assert len(table) == 6
    assert (table["anchor_years"] == 1.0).all()
    # Later visits are t=2 and t=3, so two per patient.
    assert (table["n_later_visits"] == 2).all()

    # H&Y >= 3 happens at t=3 for patients 0-2 -> event; others not.
    assert table.set_index("PATNO")["hy_ge_3"].loc[[0, 1, 2]].eq(1.0).all()
    assert table.set_index("PATNO")["hy_ge_3"].loc[[3, 4, 5]].eq(0.0).all()
    # Part IV > 0 from t=2 for patients 0-1.
    assert table.set_index("PATNO")["motor_complications"].loc[[0, 1]].eq(1.0).all()
    assert table.set_index("PATNO")["motor_complications"].loc[2].item() == 0.0


def test_outcome_table_anchor_excludes_the_anchor_visit_itself():
    """Shifting the anchor later must shrink the window of eligible outcomes."""
    panel = _outcome_panel()
    rep = _outcome_representation()
    early = cp.build_outcome_table(panel, rep, anchor_visit=1)
    late = cp.build_outcome_table(panel, rep, anchor_visit=2)
    assert (late["n_later_visits"] == 1).all()
    # Part IV turns positive at t=2, which is the anchor for `late`, so it no
    # longer counts for patients whose only Part IV visit was t=2... but here
    # it persists to t=3, so it still counts. H&Y at t=3 still counts too.
    assert late.set_index("PATNO")["hy_ge_3"].loc[0].item() == 1.0
    assert early["n_later_visits"].sum() > late["n_later_visits"].sum()


def test_outcome_associations_skip_outcomes_without_enough_events():
    table = cp.build_outcome_table(_outcome_panel(), _outcome_representation(),
                                   anchor_visit=1)
    assoc = cp.outcome_associations(table, n_boot=50, min_events=20)
    # Only 3 events in 6 patients, so every outcome must be skipped loudly
    # rather than reported as a tiny-sample AUC.
    assert (assoc["score"] == "(skipped: too few events)").all()
    assert assoc["auc"].isna().all()


def test_outcome_associations_include_the_subtype_reference():
    """The subtype label must be scored as the comparator, not omitted."""
    table = cp.build_outcome_table(_outcome_panel(), _outcome_representation(),
                                   anchor_visit=1)
    assoc = cp.outcome_associations(table, n_boot=50, min_events=1)
    scored = assoc[assoc["auc"].notna()]
    assert "subtype label (reference)" in set(scored["score"])
    # The toy data makes subtype perfectly predictive of H&Y by construction.
    ref = scored[(scored["outcome"] == "hy_ge_3")
                 & (scored["score"] == "subtype label (reference)")]
    assert float(ref["auc"].iloc[0]) == 1.0
