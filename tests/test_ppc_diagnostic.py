"""Tests for the D1 calibrated posterior-predictive label check.

Weight-bearing: folds are disjoint and whole-patient; the vectorised statistics
equal `indeterminate_audit` exactly; mid-p, z cap, ties, threshold index,
Clopper-Pearson bound and verdict rule are as frozen in B.11; later steps refuse
to run without frozen files; the generator implements the episode mechanics
(PC4 vs PC4v common random numbers, kappa only inside windows); replicates use
only the patient's own data and recover the truth when noise is small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype import hbayes
from ppmi_subtype import indeterminate_audit as IA
from ppmi_subtype import ppc_diagnostic as D
from ppmi_subtype.hbayes_tt import TTPosterior

NAMES = {0: "TD", 1: "Indeterminate", 2: "PIGD"}


def toy_data(M=60, seed=0, visits=(2, 9)):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(M):
        n = rng.integers(*visits)
        t = np.sort(rng.uniform(0, 10, n))
        mu = rng.normal(1.5, 0.8) + rng.normal(-0.1, 0.1) * t
        for tt, yy in zip(t, mu + rng.normal(0, 0.4, n)):
            rows.append({"PATNO": 100 + 3 * p, "years_from_baseline": tt, "ratio": yy})
    return hbayes.build_growth_data(pd.DataFrame(rows))


def test_folds_disjoint_whole_patient_and_deterministic():
    d = toy_data(200)
    f1, f2 = D.make_folds(d), D.make_folds(d)
    assert f1.equals(f2) and f1.PATNO.is_unique and set(f1.fold) == set(range(5))
    assert len(f1) == d.n_patients
    for s, g in f1.groupby("first_label"):                       # stratified: balanced
        assert g.fold.value_counts().max() - g.fold.value_counts().min() <= 1


def test_subset_reindexes_patients():
    d = toy_data(30)
    keep = d.patients[::3]
    s = D.subset(d, keep)
    assert np.array_equal(s.patients, keep) and s.patient_index.max() == len(keep) - 1
    assert s.n_obs == int(np.isin(d.patients[d.patient_index], keep).sum())


def test_labels_from_ratio_match_outer_rule():
    y = np.array([1.15, 1.149, 0.90, 0.901, -1.0, 3.0])
    assert D.labels_from_ratio(y).tolist() == [0, 1, 2, 1, 2, 0]


def test_vectorised_statistics_equal_indeterminate_audit():
    rng = np.random.default_rng(1)
    for _ in range(20):
        npat = rng.integers(5, 40)
        nvis = rng.integers(1, 9, npat)
        pid = np.repeat(np.arange(npat), nvis)
        L = rng.choice([0, 1, 2], len(pid), p=[0.5, 0.25, 0.25]).astype(np.int8)
        S = D.label_statistics(L[None, :], pid)[0]
        df = pd.DataFrame({"PATNO": pid, "years_from_baseline":
                           np.concatenate([np.arange(k, dtype=float) for k in nvis]),
                           "lab": [NAMES[x] for x in L]})
        C, _ = IA.count_matrix(df, "lab")
        ref = IA.statistics(C.sum(axis=0))
        tot = dict(zip(IA.COUNT_KEYS, C.sum(axis=0)))
        assert S[0] == pytest.approx((L == 1).mean())
        for j, k in ((1, "R_E"), (2, "R_D"), (3, "delta_R"), (4, "OE_Indeterminate"),
                     (5, "entry_sustained2")):
            if np.isnan(ref[k]):
                assert np.isnan(S[j])
            else:
                assert S[j] == pytest.approx(ref[k]), k
        assert tot["O_Indeterminate"] >= 0


def test_midp_ties_bounds_and_cap():
    rep = np.array([1, 2, 2, 3, 4], float)
    p, n = D.midp(2.0, rep)
    assert n == 5 and p == pytest.approx((2 + 0.5 * 2 + 0.5) / 6)
    assert D.midp(100, rep)[0] == pytest.approx(0.5 / 6)          # lower bound, never 0
    assert D.midp(-100, rep)[0] == pytest.approx(5.5 / 6)         # upper bound, never 1
    assert D.midp(2.0, np.array([np.nan, 2.0]))[1] == 1            # NaN replicates dropped
    assert D.z_cap(500) == pytest.approx(3.0911, abs=1e-3)


def test_threshold_index_strict_rule_and_power_bound():
    assert D.threshold_index(500) == 476
    T = np.arange(1, 501, dtype=float)
    assert D.threshold(T) == 476.0
    assert D.clopper_pearson_lower(60, 100) == pytest.approx(0.4972, abs=1e-4)
    assert D.clopper_pearson_lower(61, 100) == pytest.approx(0.5073, abs=1e-4)
    cap = D.z_cap(500)
    assert D.verdict(2.0, 2.0, 80, 100, cap) == "REPRODUCED (within scope)"      # tie -> no reject
    assert D.verdict(2.5, 2.0, 80, 100, cap) == "NOT REPRODUCED"
    assert D.verdict(1.0, 2.0, 60, 100, cap) == "UNINFORMATIVE (low power)"
    assert D.verdict(1.0, 2.0, 61, 100, cap) == "REPRODUCED (within scope)"
    assert D.verdict(3.5, cap, 80, 100, cap).startswith("CALIBRATION DEGENERATE")


def test_later_steps_refuse_without_frozen_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        D.read_frozen(tmp_path / "d1_threshold.json")
    (tmp_path / "d1_threshold.json").write_text('{"c": 2.5}')
    assert D.read_frozen(tmp_path / "d1_threshold.json")["c"] == 2.5


THETA = {"beta0": 2.0, "beta1": -0.08, "tau0": 1.3, "tau1": 0.07, "nu_s": 2.1,
         "sigma": 0.59, "nu_e": 2.1}


def test_generator_episode_mechanics_and_common_random_numbers():
    d = toy_data(400, seed=2)
    pc4 = {"pi": 0.3, "L": 1.02, "D": 2.0, "kappa": 0.25, "replace_level": True}
    pc4v = dict(pc4, replace_level=False)
    y0, in0 = D.generate(d, THETA, np.random.default_rng(7))
    y4, in4 = D.generate(d, THETA, np.random.default_rng(7), pc4)
    yv, inv = D.generate(d, THETA, np.random.default_rng(7), pc4v)
    assert not in0.any() and np.array_equal(in4, inv) and in4.any()
    out = ~in4
    assert np.allclose(y0[out], y4[out]) and np.allclose(y0[out], yv[out])   # identical outside
    # inside the window, with identical draws: y0 = mu + s e (null), yv = mu + k s e (PC4v),
    # y4 = L + k s e (PC4). Hence s e = (y0 - yv) / (1 - k) and mu = y0 - s e.
    k = 0.25
    se = (y0[in4] - yv[in4]) / (1 - k)
    mu = y0[in4] - se
    assert np.allclose(yv[in4], mu + k * se)                     # PC4v: own mean, scaled noise
    assert np.allclose(y4[in4], 1.02 + k * se)                   # PC4: level L, same scaled noise
    planted_rate = np.mean([D.generate(d, THETA, np.random.default_rng(s), pc4)[1].any()
                            for s in range(3)])
    assert planted_rate == 1.0


def test_replicates_use_own_data_and_recover_truth_when_noise_small():
    rng = np.random.default_rng(3)
    S = 200
    post = TTPosterior(beta=np.tile([2.0, -0.1], (S, 1)), sigma2=np.full(S, 1e-4),
                       tau0_2=np.full(S, 1.0), tau1_2=np.full(S, 0.04), n_patients=1, n_obs=1,
                       n_draws=S, config="TT", nu=np.full(S, 5.0), nu_e=np.full(S, 5.0))
    t = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.5 + 0.1 * t
    rep = D.patient_replicates(post, np.arange(S), t, y, rng)
    assert rep.shape == (S, 4)
    assert np.allclose(np.median(rep, axis=0), y, atol=0.02)


def test_run_cohort_end_to_end_small():
    d = toy_data(80, seed=4)
    folds = D.make_folds(d)
    out = D.run_cohort(d, d.y, folds, R=20, fit_kw={"n_draws": 60, "burn_in": 30, "thin": 1},
                       seed_entropy=[42, 41, 99, 0])
    assert np.isfinite(out["T"]) and out["T"] <= out["cap"] + 1e-9
    assert len(out["folds"]) == 5 and set(out["z"]) == set(D.STAT_NAMES)
    silent = D.run_cohort(d, d.y, folds, R=20, fit_kw={"n_draws": 60, "burn_in": 30, "thin": 1},
                          seed_entropy=[42, 41, 99, 1], record_stats=False)
    assert "timing" in silent
    for k in ("T", "S_obs", "z", "p", "p_at_bound", "T_at_cap", "n_rep_undefined", "folds"):
        assert k not in silent
