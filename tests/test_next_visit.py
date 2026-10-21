"""Correctness checks for the next-visit two-class prediction module.

These test the rules the experiment's conclusions depend on: the conditional
renormalisation, the tie rule, that the direct model's features and
preprocessing are causal, and that reproducing the audited HMM fit through
`hmm_stability` gives the same answer as `hmm.fit_hmm`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype import hmm, loo, next_visit as nv


# ---------------------------------------------------------------------------
#  The two conditional probabilities
# ---------------------------------------------------------------------------
def test_conditional_two_class_renormalises_and_sums_to_one():
    probs = np.array([
        [0.6, 0.2, 0.2],
        [0.1, 0.8, 0.1],
        [0.0, 0.0, 1.0],
    ])
    p_td, p_pigd, defined = nv.conditional_two_class(probs)

    assert defined.all()
    assert np.allclose(p_td + p_pigd, 1.0, atol=1e-15)
    assert np.isclose(p_td[0], 0.6 / 0.8)
    # The Indeterminate mass drops out of the ratio, so an equal TD/PIGD split
    # gives 0.5 however much mass sits on Indeterminate.
    assert np.isclose(p_td[1], 0.5)
    assert np.isclose(p_pigd[2], 1.0)


def test_conditional_two_class_refuses_a_zero_denominator():
    p_td, p_pigd, defined = nv.conditional_two_class(np.array([[0.0, 1.0, 0.0]]))
    assert not defined[0]
    assert np.isnan(p_td[0]) and np.isnan(p_pigd[0])


def test_conditional_two_class_rejects_the_wrong_shape():
    with pytest.raises(ValueError):
        nv.conditional_two_class(np.array([0.5, 0.5]))


def test_predicted_state_follows_the_documented_rule():
    p_td = np.array([0.7, 0.3, 0.5, np.nan])
    p_pigd = np.array([0.3, 0.7, 0.5, np.nan])
    out = nv.predicted_state(p_td, p_pigd)
    assert list(out) == ["TD", "PIGD", nv.TIE_RULE, ""]
    assert nv.TIE_RULE == "PIGD"


# ---------------------------------------------------------------------------
#  Case construction
# ---------------------------------------------------------------------------
def _toy_data() -> loo.LOOData:
    """Two patients, three labels each, with uneven gaps."""
    rows = []
    for patno, labels, times in (
        (1, [1, 1, 3], [0.0, 1.0, 2.5]),        # TD, TD, PIGD
        (2, [3, 2, 1, 1], [0.0, 1.0, 2.0, 4.0]),  # PIGD, Ind, TD, TD
    ):
        for k, (lab, t) in enumerate(zip(labels, times)):
            rows.append({"PATNO": patno, "EVENT_ID": f"V{k:02d}",
                         "years_from_baseline": t, "subtype": float(lab)})
    panel = pd.DataFrame(rows)
    return loo.prepare(panel, [1, 2, 3])


def test_build_cases_is_one_row_per_transition_with_causal_history():
    data = _toy_data()
    cases = nv.build_cases(data)

    assert len(cases) == data.n_transitions == 5
    assert (cases["n_history"] == cases["visit_index"] + 1).all()
    assert (cases["target_years"] > cases["current_years"]).all()
    assert np.allclose(cases["interval_years"],
                       cases["target_years"] - cases["current_years"])

    # Patient 1, predicting visit 2 (PIGD) from TD, TD: the history is all TD,
    # the run length is 2, and nothing about the PIGD target has leaked in.
    row = cases[(cases["PATNO"] == 1) & (cases["visit_index"] == 1)].iloc[0]
    assert row["current_label"] == "TD"
    assert row["actual_next_label"] == "PIGD"
    assert row["frac_hist_PIGD"] == 0.0
    assert np.isclose(row["log_run_length"], np.log(2))
    assert row["prev_noind_is_PIGD"] == 0.0

    # Patient 2 at visit 1 is Indeterminate, but the last non-Indeterminate
    # label is PIGD -- that is what persistence carries forward.
    row = cases[(cases["PATNO"] == 2) & (cases["visit_index"] == 1)].iloc[0]
    assert row["current_label"] == "Indeterminate"
    assert row["prev_noind_is_PIGD"] == 1.0
    assert row["frac_hist_IND"] == 0.5


def test_indeterminate_is_never_recoded():
    cases = nv.build_cases(_toy_data())
    assert (cases["current_label"] == "Indeterminate").any()
    assert (cases["actual_next_label"] == "Indeterminate").any()
    # An Indeterminate target is not eligible, and is not relabelled either.
    ind = cases[cases["actual_next_label"] == "Indeterminate"]
    assert not ind["eligible"].any()
    assert set(cases["actual_next_label"]) <= set(nv.STATE_NAMES)


# ---------------------------------------------------------------------------
#  The direct two-class model
# ---------------------------------------------------------------------------
def _synthetic_cases(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A case table whose target depends on one feature, for a recovery check."""
    rng = np.random.default_rng(seed)
    frac = rng.uniform(0, 1, n)
    p = nv._expit(-1.0 + 4.0 * (frac - 0.5))
    target = np.where(rng.uniform(size=n) < p, nv.PIGD, nv.TD)

    cases = pd.DataFrame({f: 0.0 for f in nv.FEATURES}, index=range(n))
    cases["frac_hist_PIGD"] = frac
    cases["log_interval"] = rng.normal(0, 0.3, n)
    cases["target_index"] = target
    cases["eligible"] = True
    cases["PATNO"] = np.arange(n) // 4
    return cases


def test_direct_model_recovers_a_known_signal_and_is_calibrated():
    cases = _synthetic_cases()
    fit = nv.fit_direct(cases)

    assert fit.converged or fit.proj_grad < nv.PROJ_GRAD_TOL
    coefs = fit.coefficient_frame().set_index("term")["coefficient"]
    assert coefs["frac_hist_PIGD"] > 0.5          # the true direction

    p = fit.predict_p_pigd(cases)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    # Mean predicted rate should track the observed rate on the training set.
    observed = float((cases["target_index"] == nv.PIGD).mean())
    assert abs(p.mean() - observed) < 0.03


def test_direct_model_learns_its_standardisation_from_training_only():
    cases = _synthetic_cases()
    train = cases[cases["PATNO"] < 50]
    fit = nv.fit_direct(train)

    raw = train[list(nv.FEATURES)].to_numpy(float)
    assert np.allclose(fit.centre, raw.mean(axis=0))
    assert fit.n_train == len(train)
    # The full table's moments differ, so a fit that had seen it would not
    # match the training-only ones.
    assert not np.allclose(fit.centre,
                           cases[list(nv.FEATURES)].to_numpy(float).mean(axis=0))


def test_direct_model_excludes_indeterminate_targets_from_the_fit():
    cases = _synthetic_cases()
    cases.loc[cases.index[:100], "target_index"] = nv.IND
    cases["eligible"] = cases["target_index"].isin([nv.TD, nv.PIGD])
    fit = nv.fit_direct(cases)
    assert fit.n_train == 300


def test_no_interval_variant_drops_only_the_interval_feature():
    cases = _synthetic_cases()
    feats = tuple(f for f in nv.FEATURES if f not in nv.INTERVAL_FEATURES)
    fit = nv.fit_direct(cases, features=feats)
    assert "log_interval" not in fit.features
    assert len(fit.features) == len(nv.FEATURES) - 1
    # Shifting the interval cannot move a prediction this model does not read.
    shifted = cases.assign(log_interval=cases["log_interval"] + 5.0)
    assert np.allclose(fit.predict_p_pigd(cases), fit.predict_p_pigd(shifted))


# ---------------------------------------------------------------------------
#  Reproducing the audited HMM fit
# ---------------------------------------------------------------------------
def test_audited_starts_reproduce_fit_hmm():
    """The multistart path must land on the same optimum as `hmm.fit_hmm`.

    `fit_hmm` reports convergence from scipy's success flag only; running the
    identical optimisation through `hmm_stability` adds the projected gradient.
    That substitution is only legitimate if the fit itself is unchanged.
    """
    rng = np.random.default_rng(7)
    sequences = [
        {"PATNO": p,
         "labels": rng.integers(0, 3, size=5),
         "gaps": np.full(4, 1.0)}
        for p in range(40)
    ]
    states = [1, 2, 3]

    reference = hmm.fit_hmm(sequences, states, state_names=nv.STATE_NAMES,
                            seed=42, n_restarts=3, emission_floor=0.5)
    starts = nv.audited_starts(3, 3, 0.5, seed=42, n_restarts=3)
    from ppmi_subtype import hmm_stability

    replayed, records = hmm_stability.fit_multistart(
        sequences, states, n_hidden=3, emission_floor=0.5, starts=starts,
        state_names=nv.STATE_NAMES)

    assert np.isclose(replayed.loglik, reference.loglik, atol=1e-8)
    assert np.allclose(replayed.Q, reference.Q, atol=1e-6)
    assert np.allclose(replayed.E, reference.E, atol=1e-6)
    assert len(records) == 3


# ---------------------------------------------------------------------------
#  A whole fold, end to end on toy data
# ---------------------------------------------------------------------------
def test_run_fold_holds_the_patient_out_and_obeys_every_invariant():
    rng = np.random.default_rng(3)
    rows = []
    for patno in range(1, 31):
        n = int(rng.integers(3, 6))
        labels = rng.choice([1, 2, 3], size=n, p=[0.6, 0.1, 0.3])
        times = np.cumsum(np.r_[0.0, rng.uniform(0.5, 2.0, n - 1)])
        for k, (lab, t) in enumerate(zip(labels, times)):
            rows.append({"PATNO": patno, "EVENT_ID": f"V{k:02d}",
                         "years_from_baseline": float(t), "subtype": float(lab)})
    data = loo.prepare(pd.DataFrame(rows), [1, 2, 3])
    cases = nv.build_cases(data)

    held = data.patients[0]
    out = nv.run_fold(data, cases, held, seed=42, n_restarts=2, emission_floor=0.5)

    assert set(out["model"]) == set(nv.ALL_MODELS)
    assert (out["PATNO"] == held).all()
    assert (out["held_out_patient"] == held).all()
    assert (out["n_train_patients"] == len(data.sequences) - 1).all()

    defined = out[out["two_class_defined"]]
    assert np.isfinite(defined[["P_TD", "P_PIGD"]].to_numpy()).all()
    assert np.allclose(defined["P_TD"] + defined["P_PIGD"], 1.0, atol=1e-12)
    assert ((defined[["P_TD", "P_PIGD"]].to_numpy() >= 0)
            & (defined[["P_TD", "P_PIGD"]].to_numpy() <= 1)).all()
    assert (defined["predicted_state"].to_numpy()
            == nv.predicted_state(defined["P_TD"].to_numpy(),
                                  defined["P_PIGD"].to_numpy())).all()

    # The raw three-label distributions of the generative models still carry
    # Indeterminate mass; that is what the conditional reading is about.
    cthmm = out[out["model"] == nv.MODEL_CTHMM]
    assert (cthmm["p_Indeterminate_raw"] > 0).all()
    assert np.allclose(cthmm[["p_TD_raw", "p_Indeterminate_raw", "p_PIGD_raw"]]
                       .sum(axis=1), 1.0, atol=1e-8)


def test_run_fold_rejects_a_leaked_patient():
    data = _toy_data()
    cases = nv.build_cases(data)
    with pytest.raises(StopIteration):
        nv.run_fold(data, cases, 999, seed=42, n_restarts=1, emission_floor=0.5)


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------
def _prediction_frame() -> pd.DataFrame:
    """Four cases: two scored correctly, one wrongly, one Indeterminate target."""
    return pd.DataFrame({
        "PATNO": [1, 1, 2, 2],
        "visit_index": [0, 1, 0, 1],
        "n_history": [1, 2, 1, 2],
        "model": ["m"] * 4,
        "actual_next_label": ["TD", "PIGD", "TD", "Indeterminate"],
        "eligible": [True, True, True, False],
        "two_class_defined": [True, True, True, True],
        "P_TD": [0.9, 0.2, 0.3, 0.5],
        "P_PIGD": [0.1, 0.8, 0.7, 0.5],
        "p_Indeterminate_raw": [0.05, 0.05, 0.05, 0.05],
        "predicted_state": ["TD", "PIGD", "PIGD", "PIGD"],
        "is_tie": [False, False, False, True],
    })


def test_scoring_covers_only_eligible_defined_cases():
    frame = _prediction_frame()
    s = nv.scored(frame)
    assert len(s) == 3
    assert "Indeterminate" not in set(s["actual_next_label"])
    assert list(s["correct"]) == [True, True, False]


def test_two_class_metrics_reconcile_and_separate_indeterminate():
    m = nv.two_class_metrics(_prediction_frame(), n_boot=50)
    assert m["n_cases_total"] == 4
    assert m["n_eligible"] == 3
    assert m["n_scored"] == 3
    assert m["n_indeterminate_next"] == 1
    assert m["n_correct"] == 2
    assert np.isclose(m["accuracy"], 2 / 3)
    # Brier on the PIGD outcome: (0.1-0)^2 + (0.8-1)^2 + (0.7-0)^2 over three.
    assert np.isclose(m["brier_two_class"], (0.01 + 0.04 + 0.49) / 3)
    assert np.isclose(m["log_loss"],
                      -np.mean(np.log([0.9, 0.8, 0.3])))


def test_confusion_and_per_class_metrics():
    frame = _prediction_frame()
    cm = nv.confusion(frame)
    assert list(cm.index) == nv.TWO_CLASS and list(cm.columns) == nv.TWO_CLASS
    assert cm.to_numpy().sum() == 3
    assert cm.loc["TD", "TD"] == 1 and cm.loc["PIGD", "TD"] == 1

    pc = nv.per_class_metrics(frame).set_index("state")
    assert pc.loc["TD", "support"] == 2
    assert np.isclose(pc.loc["TD", "precision"], 1.0)
    assert np.isclose(pc.loc["TD", "recall"], 0.5)
    assert np.isclose(pc.loc["PIGD", "precision"], 0.5)


def test_indeterminate_report_counts_without_scoring():
    r = nv.indeterminate_report(_prediction_frame())
    assert r["n"] == 1
    assert r["would_have_said_PIGD"] == 1
    assert r["would_have_said_TD"] == 0
    assert "accuracy" not in r


def test_undefined_cases_are_excluded_not_patched():
    frame = _prediction_frame()
    frame.loc[0, ["P_TD", "P_PIGD"]] = np.nan
    frame.loc[0, "two_class_defined"] = False
    frame.loc[0, "predicted_state"] = ""
    m = nv.two_class_metrics(frame, n_boot=50)
    assert m["n_eligible"] == 3
    assert m["n_scored"] == 2
    assert m["n_undefined_among_eligible"] == 1


# ---------------------------------------------------------------------------
#  The correctness checks themselves
# ---------------------------------------------------------------------------
def _small_cohort(n_patients: int = 24, seed: int = 11) -> loo.LOOData:
    rng = np.random.default_rng(seed)
    rows = []
    for patno in range(1, n_patients + 1):
        n = int(rng.integers(3, 6))
        labels = rng.choice([1, 2, 3], size=n, p=[0.55, 0.15, 0.30])
        times = np.cumsum(np.r_[0.0, rng.uniform(0.5, 2.0, n - 1)])
        for k, (lab, t) in enumerate(zip(labels, times)):
            rows.append({"PATNO": patno, "EVENT_ID": f"V{k:02d}",
                         "years_from_baseline": float(t), "subtype": float(lab)})
    return loo.prepare(pd.DataFrame(rows), [1, 2, 3])


def test_correctness_checks_all_pass_on_a_clean_partial_run():
    """A run over a subset of patients must satisfy its own invariants.

    Coverage is judged against the folds attempted, not the whole cohort, so a
    smoke test is held to what it claimed rather than reported as broken.
    """
    data = _small_cohort()
    cases = nv.build_cases(data)
    patients = data.patients[:3]
    predictions, failures = nv.run_lopo(data, cases, seed=42, n_restarts=1,
                                        emission_floor=0.5, patients=patients,
                                        progress_every=0)
    assert len(failures) == 0

    checks = nv.correctness_checks(predictions, cases, data, failures,
                                   patients=patients)
    assert all(checks.values()), [k for k, v in checks.items() if not v]


def test_correctness_checks_catch_a_tampered_probability():
    data = _small_cohort()
    cases = nv.build_cases(data)
    patients = data.patients[:2]
    predictions, failures = nv.run_lopo(data, cases, seed=42, n_restarts=1,
                                        emission_floor=0.5, patients=patients,
                                        progress_every=0)

    broken = predictions.copy()
    broken.loc[0, "P_PIGD"] = broken.loc[0, "P_PIGD"] + 0.1
    checks = nv.correctness_checks(broken, cases, data, failures, patients=patients)
    assert not checks["P_TD + P_PIGD == 1 within tolerance"]

    broken = predictions.copy()
    broken.loc[0, "predicted_state"] = (
        "TD" if broken.loc[0, "predicted_state"] == "PIGD" else "PIGD")
    checks = nv.correctness_checks(broken, cases, data, failures, patients=patients)
    assert not checks["predicted class follows the documented rule"]


def test_correctness_checks_reconcile_a_failed_fold():
    """A fold that fails must be visible in the reconciliation, not absorbed."""
    data = _small_cohort()
    cases = nv.build_cases(data)
    patients = list(data.patients[:3])
    predictions, _ = nv.run_lopo(data, cases, seed=42, n_restarts=1,
                                 emission_floor=0.5, patients=patients[:2],
                                 progress_every=0)

    # The third fold is missing and unaccounted for.
    empty = pd.DataFrame(columns=["held_out_patient", "error"])
    checks = nv.correctness_checks(predictions, cases, data, empty, patients=patients)
    assert not checks["every attempted patient held out exactly once"]
    assert not checks[
        "completed and failed folds account for every attempted patient"]

    # Recorded as a failure, it reconciles.
    recorded = pd.DataFrame([{"held_out_patient": patients[2], "error": "boom"}])
    checks = nv.correctness_checks(predictions, cases, data, recorded,
                                   patients=patients)
    assert all(checks.values()), [k for k, v in checks.items() if not v]


def test_paired_accuracy_difference_is_paired_and_reconciles():
    """The difference must be computed on the cases both models scored."""
    a = _prediction_frame()
    b = a.copy()
    b["model"] = "n"
    # Model b gets the third case right where a got it wrong.
    b.loc[2, ["P_TD", "P_PIGD", "predicted_state"]] = [0.8, 0.2, "TD"]

    r = nv.paired_accuracy_difference(a, b, n_boot=200)
    assert r["n_paired"] == 3
    assert np.isclose(r["accuracy_delta"], -1 / 3)
    assert r["a_right_b_wrong"] == 0
    assert r["b_right_a_wrong"] == 1
    assert r["n_discordant"] == 1
    assert (r["delta_ci_patient_bootstrap_low"]
            <= r["accuracy_delta"] <= r["delta_ci_patient_bootstrap_high"])


def test_paired_accuracy_difference_is_zero_against_itself():
    a = _prediction_frame()
    r = nv.paired_accuracy_difference(a, a.copy(), n_boot=100)
    assert r["accuracy_delta"] == 0.0
    assert r["n_discordant"] == 0
    assert np.isnan(r["mcnemar_exact_p_ignores_clustering"])


def test_accuracy_breakdowns_partition_the_scored_cases():
    frame = _prediction_frame()
    frame["interval_years"] = [1.0, 1.0, 3.0, 1.0]
    frame["current_label"] = ["TD", "PIGD", "TD", "Indeterminate"]
    b = nv.accuracy_breakdowns(frame)

    n_scored = len(nv.scored(frame))
    for key in ("by_interval", "by_current_label", "by_history_length"):
        assert int(b[key]["n"].sum()) == n_scored, key

    iv = b["by_interval"].set_index("interval")
    assert iv.loc["<= 1.5y", "n"] == 2
    assert iv.loc["> 1.5y", "n"] == 1
    # The Indeterminate current label is an input, so it appears as a row even
    # though that case's target is not eligible and is not scored.
    assert set(b["by_current_label"]["current_label"]) == set(nv.STATE_NAMES)
