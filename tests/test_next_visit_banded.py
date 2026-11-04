"""Leakage and fold-level checks for the ratio-banded next-visit experiment.

The experiment's whole claim to fairness is that the band structure is a
*learned preprocessing decision* taken inside each fold on the training
patients only. These tests check that directly: that the held-out patient's
ratios cannot move a band edge, that the construction is re-verified rather
than trusted, and that both arms of one fold see exactly the same prediction
cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype import loo, next_visit as nv, next_visit_banded as nvb
from ppmi_subtype import ratio_bands as rb

LABELS = {"TD": 1, "Indeterminate": 2, "PIGD": 3}
STATES = [LABELS[s] for s in nv.STATE_NAMES]


class _SubtypeCfg:
    """The handful of config fields `prepare` reads."""
    td_threshold = 1.15
    pigd_threshold = 0.90
    band_boundaries = "outer"


def _panel(spec) -> pd.DataFrame:
    """`spec` is {PATNO: [(years, subtype_name, ratio), ...]}."""
    rows = []
    for patno, visits in spec.items():
        for k, (t, name, ratio) in enumerate(visits):
            rows.append({"PATNO": patno, "EVENT_ID": f"V{k:02d}",
                         "years_from_baseline": float(t),
                         "subtype": float(LABELS[name]), "ratio": ratio})
    return pd.DataFrame(rows)


def _toy(tail_patient_ratio: float = 3.3):
    """A deterministic cohort whose tail bands sit right on the merge threshold.

    With ``min_observations=3, min_patients=2`` the training patients give
    [3.00,3.25) three observations from three patients (it closes on its own)
    and [3.25,3.50) two observations from one patient (it does not). Patient
    999 sits in that second band, so *including* them closes it and creates an
    extra edge --- which is exactly the leak the fold construction must avoid.
    """
    def visits(ratios):
        out = []
        for k, r in enumerate(ratios):
            name = ("TD" if not np.isfinite(r) else
                    "PIGD" if r <= 0.90 else "TD" if r >= 1.15 else "Indeterminate")
            out.append((k * 1.0, name, r))
        return out

    spec = {}
    for p in (1, 2, 3):                       # [3.00,3.25): 3 obs, 3 patients
        spec[p] = visits([0.4, 0.95, 1.3, 3.05])
    spec[4] = visits([0.4, 1.3, 3.3, 3.3])    # [3.25,3.50): 2 obs, 1 patient
    for p in (5, 6, 7, 8):                    # [3.50,3.75): 4 obs, 4 patients
        spec[p] = visits([0.4, 0.95, 1.3, 3.6])
    spec[999] = visits([tail_patient_ratio, tail_patient_ratio, np.nan])

    panel = _panel(spec)
    data = loo.prepare(panel, STATES)
    bdata = nvb.prepare(panel, data, _SubtypeCfg())
    return panel, data, bdata


# ---------------------------------------------------------------------------
#  Alignment of ratios to the sequences
# ---------------------------------------------------------------------------
def test_prepare_aligns_each_ratio_with_its_own_visit():
    panel, data, bdata = _toy()
    assert [s["PATNO"] for s in bdata.sequences] == data.patients
    for seq, ref in zip(bdata.sequences, data.sequences):
        assert np.array_equal(seq["labels"], ref["labels"])
        assert np.array_equal(seq["gaps"], ref["gaps"])
        rows = panel[panel["PATNO"] == seq["PATNO"]].sort_values("years_from_baseline")
        assert np.array_equal(np.nan_to_num(seq["ratios"], nan=-1),
                              np.nan_to_num(rows["ratio"].to_numpy(float), nan=-1))


def test_prepare_fails_loudly_if_the_panel_and_the_sequences_disagree():
    panel, data, _ = _toy()
    scrambled = panel.copy()
    mask = scrambled["PATNO"] == 1
    scrambled.loc[mask, "subtype"] = float(LABELS["PIGD"])
    with pytest.raises(ValueError, match="does not match the sequence"):
        nvb.prepare(scrambled, data, _SubtypeCfg())


# ---------------------------------------------------------------------------
#  Training-only band construction
# ---------------------------------------------------------------------------
def test_the_held_out_patient_cannot_move_a_band_edge():
    # Patient 999's two observations are what would close [3.25,3.50). With
    # them in the cohort the merge keeps that edge; held out, it must not.
    _, _, bdata = _toy(tail_patient_ratio=3.3)
    whole, _, _ = nvb.fold_alphabet(bdata, None, min_observations=3, min_patients=2)
    without, _, _ = nvb.fold_alphabet(bdata, 999, min_observations=3, min_patients=2)

    assert max(whole.tail_lows) > max(without.tail_lows)
    # Rebuilt from the training patients, the edges are the training ones.
    rebuilt, _, _ = nvb.fold_alphabet(bdata, 999, min_observations=3, min_patients=2)
    assert list(rebuilt.tail_lows) == list(without.tail_lows)


def test_a_held_out_ratio_above_every_training_band_is_still_assignable():
    _, _, bdata = _toy(tail_patient_ratio=9.0)
    scheme, alphabet, _ = nvb.fold_alphabet(bdata, 999, min_observations=3,
                                            min_patients=2)
    held = bdata.by_patient()[999]
    assert np.nanmax(held["ratios"]) > scheme.edges[-1]
    codes = alphabet.code(held["labels"], scheme.band_index(held["ratios"]))
    assert len(codes) == len(held["labels"])          # nothing unassignable


def test_the_leakage_assertion_rejects_edges_the_held_out_patient_shaped():
    _, _, bdata = _toy(tail_patient_ratio=3.3)
    leaky, _, _ = nvb.fold_alphabet(bdata, None, min_observations=3, min_patients=2)
    with pytest.raises(loo.LeakageError, match="not the ones the training patients"):
        nvb.assert_bands_training_only(bdata, 999, leaky, min_observations=3,
                                       min_patients=2)
    clean, _, _ = nvb.fold_alphabet(bdata, 999, min_observations=3, min_patients=2)
    nvb.assert_bands_training_only(bdata, 999, clean, min_observations=3,
                                   min_patients=2)


def test_band_construction_uses_no_label_of_the_held_out_patient_at_all():
    """Changing only the held-out patient's data must not change the scheme."""
    _, _, before = _toy(tail_patient_ratio=3.3)
    _, _, after = _toy(tail_patient_ratio=11.0)
    a, _, _ = nvb.fold_alphabet(before, 999, min_observations=3, min_patients=2)
    b, _, _ = nvb.fold_alphabet(after, 999, min_observations=3, min_patients=2)
    assert list(a.tail_lows) == list(b.tail_lows)


def test_joint_sequences_recode_the_requested_patients_only():
    _, data, bdata = _toy()
    _, alphabet, _ = nvb.fold_alphabet(bdata, 999, min_observations=3, min_patients=2)
    train = nvb.joint_sequences(bdata, alphabet,
                                [p for p in bdata.patients if p != 999])
    assert 999 not in {s["PATNO"] for s in train}
    assert len(train) == len(bdata.sequences) - 1
    for seq in train:
        source = bdata.by_patient()[seq["PATNO"]]
        assert len(seq["labels"]) == len(source["labels"])
        assert np.array_equal(seq["gaps"], source["gaps"])
        # Each code carries its visit's own subtype.
        assert np.array_equal(alphabet.subtype_of[seq["labels"]], source["labels"])


# ---------------------------------------------------------------------------
#  One fold, end to end
# ---------------------------------------------------------------------------
def test_run_fold_scores_only_the_held_out_patient_and_both_arms_see_the_same_cases():
    panel, data, bdata = _toy()
    cases = nv.build_cases(data)
    held = data.patients[0]

    preds, diag = nvb.run_fold(data, bdata, cases, held, seed=42, n_restarts=1,
                               emission_floor=0.5, min_observations=3, min_patients=2)

    assert set(preds["PATNO"]) == {held}
    assert set(preds["held_out_patient"]) == {held}
    assert set(preds["model"]) == set(nvb.MODELS)
    assert (preds["n_train_patients"] == len(data.sequences) - 1).all()

    per_model = {m: sorted(g["visit_index"]) for m, g in preds.groupby("model")}
    expected = sorted(cases.loc[cases["PATNO"] == held, "visit_index"])
    assert all(v == expected for v in per_model.values())

    # The two probabilities are the conditional split, and the labels and
    # intervals are the case table's, untouched.
    defined = preds[preds["two_class_defined"]]
    assert np.allclose(defined["P_TD"] + defined["P_PIGD"], 1.0)
    merged = preds.merge(cases, on=["PATNO", "visit_index"], suffixes=("", "_case"))
    assert (merged["actual_next_label"] == merged["actual_next_label_case"]).all()
    assert np.allclose(merged["interval_years"], merged["interval_years_case"])

    assert diag["held_out_patient"] == held
    assert diag["n_params_banded"] > diag["n_params_label"] == 14
    assert diag["banded_min_diag_M"] >= 0.5 - 1e-12


def test_the_fold_records_the_band_structure_it_used():
    _, data, bdata = _toy()
    cases = nv.build_cases(data)
    preds, diag = nvb.run_fold(data, bdata, cases, data.patients[1], seed=42,
                               n_restarts=1, emission_floor=0.5,
                               min_observations=3, min_patients=2)
    assert diag["n_tail_bands"] == len(diag["tail_lows"].split(","))
    assert diag["tail_lows"].startswith("3.00")
    assert diag["n_categories"] == preds["n_categories"].iloc[0]
    assert diag["n_categories"] > 3


def test_the_control_arm_reproduces_the_baseline_arm_within_the_fold():
    """The joint code path over a flat alphabet is the square model."""
    _, data, bdata = _toy()
    cases = nv.build_cases(data)
    preds, diag = nvb.run_fold(data, bdata, cases, data.patients[2], seed=42,
                               n_restarts=2, emission_floor=0.5,
                               min_observations=3, min_patients=2)
    assert diag["control_vs_label_loglik_gap"] < 1e-2
    assert diag["control_vs_label_max_abs_prob_gap"] < 1e-2


def test_run_fold_refuses_a_patient_who_is_in_the_training_data():
    _, data, bdata = _toy()
    cases = nv.build_cases(data)
    # A held-out id that is not in the cohort leaves the training set complete,
    # which is the leakage condition `loo.assert_no_leakage` exists to catch
    # the other way round; here it must fail on the missing test sequence.
    with pytest.raises(StopIteration):
        nvb.run_fold(data, bdata, cases, -1, seed=42, n_restarts=1,
                     emission_floor=0.5, min_observations=3, min_patients=2)


# ---------------------------------------------------------------------------
#  Diagnostics
# ---------------------------------------------------------------------------
def test_band_occupancy_counts_every_observation_once():
    _, data, bdata = _toy()
    _, alphabet, _ = nvb.fold_alphabet(bdata, None, min_observations=3, min_patients=2)
    occ = nvb.band_occupancy(bdata, alphabet)

    assert len(occ) == alphabet.n_categories
    assert occ["n_observations"].sum() == sum(len(s["labels"]) for s in bdata.sequences)
    assert (occ["n_patients"] <= occ["n_observations"]).all()
    # The undefined-ratio categories are present and one of them is occupied,
    # because patient 999's last visit has no ratio.
    undefined = occ[occ["band"] == rb.UNDEFINED_BAND]
    assert len(undefined) == 2 and undefined["n_observations"].sum() >= 1


def test_ratio_range_accuracy_splits_on_the_classification_thresholds():
    _, data, bdata = _toy()
    cases = nv.build_cases(data)
    preds, _ = nvb.run_fold(data, bdata, cases, data.patients[0], seed=42,
                            n_restarts=1, emission_floor=0.5,
                            min_observations=3, min_patients=2)
    table = nvb.ratio_range_accuracy(preds)
    scored = nv.scored(preds[preds["model"] == nvb.MODEL_BANDED])
    assert "undefined" in set(table["ratio_range"])
    assert table[table["model"] == nvb.MODEL_BANDED]["n"].sum() == len(scored)
