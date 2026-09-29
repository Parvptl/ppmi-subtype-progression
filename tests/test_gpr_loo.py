"""Tests for the leave-one-patient-out validation of the trajectory models.

The point of interest is `test_training_set_excludes_the_held_out_patient`: if
that fails, every number the step produces is optimistic and nothing else in
this file matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype import gpr_loo


def make_data(n_patients=6, n_visits=4, seed=0) -> gpr_loo.FoldData:
    rng = np.random.default_rng(seed)
    times, ratios, subtypes = {}, {}, {}
    for p in range(n_patients):
        t = np.arange(n_visits, dtype=float)
        y = 2.0 + rng.normal(0, 0.4, n_visits)
        times[p] = t
        ratios[p] = y
        subtypes[p] = np.array(
            ["TD" if v >= 1.15 else "PIGD" if v <= 0.90 else "Indeterminate"
             for v in y]
        )
    return gpr_loo.FoldData(times=times, ratios=ratios, subtypes=subtypes)


SETTINGS = {
    "n_restarts": 1, "seed": 0, "max_iter": 60, "jitter": 1e-8,
    "td_threshold": 1.15, "pigd_threshold": 0.90,
    "hbayes_draws": 120, "hbayes_burn_in": 40, "hbayes_pred_draws": 40,
}


class TestFoldData:
    def test_training_set_excludes_the_held_out_patient(self):
        """No visit of the held-out patient may reach a training set."""
        data = make_data()
        held = data.patients[2]
        train = data.sequences(exclude=held)

        assert len(train) == len(data.patients) - 1
        target = data.ratios[held]
        for seq in train:
            assert not np.array_equal(seq["y"], target)

    def test_sequences_keep_every_other_patient(self):
        data = make_data()
        assert len(data.sequences()) == len(data.patients)


class TestBaselines:
    def test_persistence_repeats_the_last_value(self):
        y = np.array([1.0, 2.0, 3.0])
        got = gpr_loo._baseline_prediction("persistence", np.array([0.0, 1.0, 2.0]),
                                           y, 3.0, np.array([0.0, 0.0]))
        assert got == 3.0

    def test_patient_mean_averages_the_history(self):
        y = np.array([1.0, 2.0, 3.0])
        got = gpr_loo._baseline_prediction("patient_mean", np.array([0.0, 1.0, 2.0]),
                                           y, 9.0, np.array([0.0, 0.0]))
        assert got == pytest.approx(2.0)

    def test_population_ignores_the_patient(self):
        beta = np.array([2.0, -0.1])
        got = gpr_loo._baseline_prediction("population", np.array([0.0]),
                                           np.array([99.0]), 5.0, beta)
        assert got == pytest.approx(1.5)

    def test_patient_ols_extrapolates_a_straight_line(self):
        t = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        assert gpr_loo._ols_extrapolate(t, y, 5.0) == pytest.approx(6.0)

    def test_patient_ols_carries_forward_when_there_is_no_line(self):
        """One observation, or all at the same time, cannot define a slope."""
        assert gpr_loo._ols_extrapolate(np.array([0.0]), np.array([2.5]), 9.0) == 2.5
        assert gpr_loo._ols_extrapolate(np.array([1.0, 1.0]), np.array([2.0, 3.0]),
                                        4.0) == 3.0

    def test_pooled_ols_recovers_a_known_line(self):
        sequences = [
            {"t": np.array([0.0, 1.0, 2.0]), "y": 3.0 - 0.5 * np.array([0.0, 1.0, 2.0])}
            for _ in range(4)
        ]
        beta = gpr_loo._pooled_ols(sequences)
        assert beta[0] == pytest.approx(3.0)
        assert beta[1] == pytest.approx(-0.5)

    def test_baseline_spread_comes_from_training_patients(self):
        sequences = [{"t": np.arange(4.0), "y": np.array([1.0, 1.0, 1.0, 1.0])}
                     for _ in range(3)]
        sds = gpr_loo._baseline_residual_sd(sequences, np.array([1.0, 0.0]))

        # A perfectly flat cohort leaves persistence with no error at all.
        assert sds["persistence"] == pytest.approx(0.0, abs=1e-12)
        assert set(sds) == set(gpr_loo.BASELINES)


class TestRunFold:
    def test_every_model_predicts_every_visit_after_the_first(self):
        data = make_data(n_patients=5, n_visits=4)
        preds, fits, fitted = gpr_loo.run_fold(
            data, 0, ["K0"], SETTINGS, with_hbayes=False, with_baselines=True
        )
        df = pd.DataFrame(preds)

        expected_models = {"K0", *gpr_loo.BASELINES}
        assert set(df["model"]) == expected_models
        for model, g in df.groupby("model"):
            assert g["target_index"].tolist() == [1, 2, 3]
        assert set(fitted) == {"K0"}
        assert len(fits) == 1

    def test_history_length_always_equals_the_target_index(self):
        data = make_data(n_patients=5, n_visits=5)
        preds, _, _ = gpr_loo.run_fold(data, 1, ["K0"], SETTINGS,
                                       with_hbayes=False, with_baselines=False)
        df = pd.DataFrame(preds)
        assert (df["history_length"] == df["target_index"]).all()

    def test_target_time_is_after_the_history(self):
        data = make_data(n_patients=5, n_visits=5)
        preds, _, _ = gpr_loo.run_fold(data, 1, ["K0"], SETTINGS,
                                       with_hbayes=False, with_baselines=False)
        df = pd.DataFrame(preds)
        assert (df["t_target"] > df["t_last_history"]).all()

    def test_every_fold_trains_on_all_other_patients(self):
        data = make_data(n_patients=7)
        preds, _, _ = gpr_loo.run_fold(data, 3, ["K0"], SETTINGS,
                                       with_hbayes=False, with_baselines=False)
        assert {r["n_train_patients"] for r in preds} == {6}

    def test_subtype_probabilities_sum_to_one(self):
        data = make_data(n_patients=5)
        preds, _, _ = gpr_loo.run_fold(data, 0, ["K0", "K2"], SETTINGS,
                                       with_hbayes=False, with_baselines=True)
        df = pd.DataFrame(preds)
        assert np.allclose(df[gpr_loo.PROB_COLS].sum(axis=1), 1.0, atol=1e-9)

    def test_a_two_visit_patient_yields_exactly_one_prediction(self):
        data = make_data(n_patients=4, n_visits=2)
        preds, _, _ = gpr_loo.run_fold(data, 0, ["K0"], SETTINGS,
                                       with_hbayes=False, with_baselines=False)
        assert len(preds) == 1
        assert preds[0]["history_length"] == 1


class TestMultistep:
    def test_splits_at_half_and_reuses_the_fitted_gp(self):
        data = make_data(n_patients=5, n_visits=5)
        _, _, fitted = gpr_loo.run_fold(data, 0, ["K0"], SETTINGS,
                                        with_hbayes=False, with_baselines=False)
        rows = gpr_loo.run_multistep_fold(data, 0, ["K0"], SETTINGS, 3, fitted=fitted)
        df = pd.DataFrame(rows)

        assert df["history_length"].unique().tolist() == [3]
        assert df["target_index"].tolist() == [3, 4]
        assert df["step"].tolist() == [1, 2]

    def test_joint_loglik_is_recorded_once_per_patient(self):
        data = make_data(n_patients=5, n_visits=6)
        rows = gpr_loo.run_multistep_fold(data, 0, ["K0"], SETTINGS, 3)
        df = pd.DataFrame(rows)
        assert df["traj_loglik_joint"].nunique() == 1
        assert np.isfinite(df["traj_loglik_joint"].iloc[0])

    def test_a_mismatched_fitted_gp_is_rejected(self):
        data = make_data(n_patients=4, n_visits=5)
        _, _, fitted = gpr_loo.run_fold(data, 0, ["K0"], SETTINGS,
                                        with_hbayes=False, with_baselines=False)
        with pytest.raises(ValueError, match="carries kernel"):
            gpr_loo.run_multistep_fold(data, 0, ["K2"], SETTINGS, 3,
                                       fitted={"K2": fitted["K0"]})

    def test_short_trajectories_are_skipped(self):
        data = make_data(n_patients=4, n_visits=2)
        assert gpr_loo.run_multistep_fold(data, 0, ["K0"], SETTINGS, 3) == []


class TestRunLopo:
    def test_serial_and_parallel_agree(self):
        """Workers must change runtime and nothing else."""
        data = make_data(n_patients=6, n_visits=4)
        a, _, _ = gpr_loo.run_lopo(data, ["K0"], SETTINGS, with_hbayes=False,
                                   multistep=False, jobs=1)
        b, _, _ = gpr_loo.run_lopo(data, ["K0"], SETTINGS, with_hbayes=False,
                                   multistep=False, jobs=2)
        merge_on = ["held_out_patient", "model", "target_index"]
        a = a.sort_values(merge_on).reset_index(drop=True)
        b = b.sort_values(merge_on).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)

    def test_unknown_patient_is_rejected(self):
        data = make_data(n_patients=4)
        with pytest.raises(ValueError, match="not in the cohort"):
            gpr_loo.run_lopo(data, ["K0"], SETTINGS, patients=[999],
                             with_hbayes=False, multistep=False)

    def test_checks_pass_on_a_clean_run(self):
        data = make_data(n_patients=6, n_visits=4)
        preds, _, _ = gpr_loo.run_lopo(data, ["K0"], SETTINGS, with_hbayes=False,
                                       multistep=False, jobs=1)
        failures = [c for c, ok in gpr_loo.check_predictions(preds, data, "t") if not ok]
        assert failures == []

    def test_checks_catch_a_training_set_that_is_too_large(self):
        data = make_data(n_patients=6, n_visits=4)
        preds, _, _ = gpr_loo.run_lopo(data, ["K0"], SETTINGS, with_hbayes=False,
                                       multistep=False, jobs=1)
        preds.loc[0, "n_train_patients"] = len(data.patients)   # saw everyone

        results = dict(gpr_loo.check_predictions(preds, data, "t"))
        assert not results["every fold trained on all other patients"]


class TestMetrics:
    def make_predictions(self, error=0.0, sd=1.0, n=200):
        rng = np.random.default_rng(0)
        observed = rng.uniform(0.2, 5.0, n)
        return pd.DataFrame(
            {
                "model": "test",
                "observed_ratio": observed,
                "pred_mean": observed + error,
                "pred_sd": sd,
                "q2.5": observed + error - 1.96 * sd,
                "q97.5": observed + error + 1.96 * sd,
                "negative_mass": 0.0,
                "observed_subtype": ["TD" if v >= 1.15 else "PIGD" if v <= 0.90
                                     else "Indeterminate" for v in observed],
                "p_TD": 1 / 3, "p_Indeterminate": 1 / 3, "p_PIGD": 1 / 3,
            }
        )

    def test_a_perfect_prediction_scores_zero_error(self):
        m = gpr_loo.continuous_metrics(self.make_predictions(error=0.0))
        assert m["MAE"] == pytest.approx(0.0)
        assert m["RMSE"] == pytest.approx(0.0)
        assert m["bias"] == pytest.approx(0.0)
        assert m["coverage95"] == pytest.approx(1.0)

    def test_bias_is_signed_and_mae_is_not(self):
        m = gpr_loo.continuous_metrics(self.make_predictions(error=-0.5))
        assert m["bias"] == pytest.approx(-0.5)
        assert m["MAE"] == pytest.approx(0.5)

    def test_crps_matches_a_known_value(self):
        # At y = mu the expression collapses to sd * (2 phi(0) - 1/sqrt(pi)).
        expected = 2 * (1 / np.sqrt(2 * np.pi)) - 1 / np.sqrt(np.pi)
        for sd in (1.0, 2.5):
            got = gpr_loo.gaussian_crps(np.array([2.0]), np.array([2.0]),
                                        np.array([sd]))
            assert float(got[0]) == pytest.approx(sd * expected, rel=1e-12)

    def test_coverage_falls_when_intervals_are_too_narrow(self):
        """The prediction has to be wrong for a narrow interval to miss it."""
        df = self.make_predictions(error=0.5, sd=1.0)
        wide = gpr_loo.continuous_metrics(df)["coverage95"]

        df["q2.5"] = df["pred_mean"] - 0.01
        df["q97.5"] = df["pred_mean"] + 0.01
        narrow = gpr_loo.continuous_metrics(df)["coverage95"]

        assert wide == pytest.approx(1.0)
        assert narrow == pytest.approx(0.0)

    def test_categorical_metrics_are_bounded(self):
        m = gpr_loo.categorical_metrics(self.make_predictions())
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["balanced_accuracy"] <= 1.0
        assert 0.0 <= m["brier"] <= 2.0

    def test_confusion_is_square_and_totals_the_predictions(self):
        df = self.make_predictions()
        cm = gpr_loo.confusion(df)
        assert cm.shape == (3, 3)
        assert int(cm.to_numpy().sum()) == len(df)


class TestCheckpointing:
    """Regression tests for the defect that destroyed the first D1 run.

    `run_lopo` used to hold every fold in memory and write only at the end, so
    an interruption at fold 640 of 641 lost all 640. The guarantee that matters
    is not merely that resuming works, but that it produces exactly what an
    uninterrupted run would.
    """

    def test_resumed_run_matches_an_uninterrupted_run(self, tmp_path):
        data = make_data(n_patients=8, n_visits=4)
        key = ["held_out_patient", "model", "target_index"]

        whole, _, _ = gpr_loo.run_lopo(data, ["K0"], SETTINGS, with_hbayes=False,
                                       multistep=False, jobs=1)

        ck = tmp_path / "ck"
        gpr_loo.run_lopo(data, ["K0"], SETTINGS, patients=data.patients[:4],
                         with_hbayes=False, multistep=False, jobs=1,
                         checkpoint=ck, checkpoint_every=2)
        resumed, _, _ = gpr_loo.run_lopo(data, ["K0"], SETTINGS, with_hbayes=False,
                                         multistep=False, jobs=1, checkpoint=ck,
                                         checkpoint_every=2)

        a = whole.sort_values(key).reset_index(drop=True)
        b = resumed.sort_values(key).reset_index(drop=True)
        assert a.equals(b[a.columns])

    def test_checkpoint_is_written_during_the_run(self, tmp_path):
        data = make_data(n_patients=6, n_visits=3)
        ck = tmp_path / "ck"
        gpr_loo.run_lopo(data, ["K0"], SETTINGS, with_hbayes=False,
                         multistep=False, jobs=1, checkpoint=ck,
                         checkpoint_every=2)

        written = {p.name for p in tmp_path.glob("*.csv")}
        assert written == {"ck_predictions.csv", "ck_fits.csv", "ck_multistep.csv"}
        assert not list(tmp_path.glob("*.tmp")), "a temp file was left behind"

    def test_a_corrupt_checkpoint_is_ignored_rather_than_trusted(self, tmp_path):
        """A half-written file must not be resumed from."""
        data = make_data(n_patients=4, n_visits=3)
        ck = tmp_path / "ck"
        (tmp_path / "ck_predictions.csv").write_text("not,a,valid\ncheckpoint\n")

        out, _, _ = gpr_loo.run_lopo(data, ["K0"], SETTINGS, with_hbayes=False,
                                     multistep=False, jobs=1, checkpoint=ck)
        assert out["held_out_patient"].nunique() == 4

    def test_checkpoint_never_shrinks_the_requested_work(self, tmp_path):
        """A checkpoint from a different patient set must not skip new patients."""
        data = make_data(n_patients=6, n_visits=3)
        ck = tmp_path / "ck"
        gpr_loo.run_lopo(data, ["K0"], SETTINGS, patients=data.patients[:2],
                         with_hbayes=False, multistep=False, jobs=1, checkpoint=ck)

        out, _, _ = gpr_loo.run_lopo(data, ["K0"], SETTINGS, with_hbayes=False,
                                     multistep=False, jobs=1, checkpoint=ck)
        assert out["held_out_patient"].nunique() == 6
