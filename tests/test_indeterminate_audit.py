"""Tests for the Indeterminate audit (Part C2).

Weight-bearing: the closed-form expected adjacency equals the brute-force mean
over all reorderings (including edge cases); unlabelled visits are skipped
before adjacency; unobserved futures never enter a denominator; repeated events
per patient are all counted; segments block adjacency and outcomes; the
bootstrap resamples patients; the reading rule matches the frozen plan.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype import indeterminate_audit as IA

T, I, P = IA.TD, IA.IND, IA.PIGD


def brute_expected(labels, seg, L):
    labels, seg = np.asarray(labels, dtype=object), np.asarray(seg)
    total = 0.0
    for s in np.unique(seg):
        lab = list(labels[seg == s])
        perms = list(itertools.permutations(range(len(lab))))
        vals = [sum(lab[p[k]] == L and lab[p[k + 1]] == L for k in range(len(lab) - 1))
                for p in perms]
        total += np.mean(vals) if len(lab) >= 2 else 0.0
    return total


@pytest.mark.parametrize("labels", [
    [I, I], [T, I], [T, T, T], [I, T, I], [I, I, I, I], [T, I, P, I, T],
    [I, P, I, I, T, I], [P, P, I, T, T, I, I],
])
def test_expected_adjacency_equals_brute_force(labels):
    seg = np.zeros(len(labels), int)
    for L in (T, I, P):
        assert IA.expected_adjacent(labels, seg, L) == pytest.approx(
            brute_expected(labels, seg, L))


def test_expected_adjacency_with_segments_and_edge_cases():
    labels = [I, I, T, I, I, I]
    seg = [0, 0, 0, 1, 1, 1]
    assert IA.expected_adjacent(labels, seg, I) == pytest.approx(brute_expected(labels, seg, I))
    assert IA.observed_adjacent(labels, seg, I) == 1 + 2      # no pair across the boundary
    assert IA.expected_adjacent([I], [0], I) == 0.0             # single visit
    assert IA.expected_adjacent([T, I, T], [0, 0, 0], I) == 0.0  # m = 1 contributes nothing
    assert IA.expected_adjacent([I, I], [0, 0], I) == 1.0       # n = m = 2 -> forced adjacency


def frame(rows):
    """rows: (PATNO, time, label or None, treatment)"""
    return pd.DataFrame(rows, columns=["PATNO", "years_from_baseline", "lab", "PDTRTMNT"])


def test_unlabelled_visits_are_skipped_before_adjacency_and_order_is_by_time():
    df = frame([(1, 2.0, I, 1), (1, 0.0, I, 1), (1, 1.0, None, 1)])   # file order scrambled
    C, pats = IA.count_matrix(df, "lab")
    c = dict(zip(IA.COUNT_KEYS, C[0]))
    assert c["n_visits"] == 2 and c["O_Indeterminate"] == 1 and c["E_Indeterminate"] == 1.0


def test_event_outcomes_and_unobserved_never_in_denominators():
    # TD -> Ind -> TD (reverted); PIGD -> Ind -> Ind (sustained); TD -> Ind (end: unobserved)
    seqs = {1: [T, I, T], 2: [P, I, I], 3: [T, I], 4: [T, P, T], 5: [P, T],
            6: [T, I, P, I, T]}
    rows = [(p, float(k), lab, 1) for p, s in seqs.items() for k, lab in enumerate(s)]
    C, _ = IA.count_matrix(frame(rows), "lab")
    c = dict(zip(IA.COUNT_KEYS, C.sum(axis=0)))
    # entries: p1 (rev), p2 (sus), p3 (unobs), p6: T->I (moved on to P), P->I (moved on to T)
    assert c["entry_n"] == 5 and c["entry_unobs"] == 1 and c["entry_obs"] == 4
    assert c["entry_rev"] == 1 and c["entry_sus"] == 1 and c["entry_mov"] == 2
    # direct: p4 T->P (rev), p4 P->T (unobserved), p5 P->T (unobserved)
    assert c["direct_n"] == 3 and c["direct_obs"] == 1 and c["direct_rev"] == 1
    assert c["direct_unobs"] == 2
    s = IA.statistics(C.sum(axis=0))
    assert s["R_E"] == pytest.approx(1 / 4) and s["R_D"] == pytest.approx(1.0)
    assert s["delta_R"] == pytest.approx(-0.75)


def test_repeated_events_from_one_patient_are_all_counted():
    rows = [(1, float(k), lab, 1) for k, lab in enumerate([T, I, T, I, T, I])]
    C, _ = IA.count_matrix(frame(rows), "lab")
    c = dict(zip(IA.COUNT_KEYS, C[0]))
    assert c["entry_n"] == 3 and c["entry_rev"] == 2 and c["entry_unobs"] == 1
    assert c["exit_n"] == 2


def test_two_step_persistence_denominator():
    rows = [(1, float(k), lab, 1) for k, lab in enumerate([T, I, I, I])] + \
           [(2, float(k), lab, 1) for k, lab in enumerate([T, I, I])]
    C, _ = IA.count_matrix(frame(rows), "lab")
    c = dict(zip(IA.COUNT_KEYS, C.sum(axis=0)))
    assert c["entry_obs"] == 2 and c["entry_obs2"] == 1 and c["entry_sus2"] == 1


def test_segments_by_interval_and_treatment():
    t = np.array([0.0, 1.0, 3.0, 4.0])
    tr = np.array([0.0, 0.0, 1.0, np.nan])
    assert list(IA.segments(t, None, max_interval=1.5)) == [0, 0, 1, 1]
    assert list(IA.segments(t, tr, treatment_stable=True)) == [0, 0, 1, 2]
    # an entry whose outcome lies in the next segment is 'outside', not unobserved
    df = frame([(1, 0.0, T, 0), (1, 1.0, I, 0), (1, 5.0, T, 0)])
    C, _ = IA.count_matrix(df, "lab", {"max_interval": 1.5})
    c = dict(zip(IA.COUNT_KEYS, C[0]))
    assert c["entry_out"] == 1 and c["entry_obs"] == 0 and c["entry_unobs"] == 0


def test_bootstrap_resamples_patients_not_pairs():
    C = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    sums = IA.patient_bootstrap_sums(C, 500, 1)
    # every replicate is a non-negative integer combination of whole patient rows
    # summing to 3 patients: row-sum totals must be in {0..6} and respect row structure
    assert sums.shape == (500, 2)
    totals = sums.sum(axis=1)
    assert set(np.unique(totals)) <= {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}
    same = np.tile([[1.0, 1.0]], (5, 1))
    assert np.all(IA.patient_bootstrap_sums(same, 50, 2) == 5.0)


def test_bootstrap_interval_contains_point_and_counts_undefined():
    rows = [(p, float(k), lab, 1) for p in range(40)
            for k, lab in enumerate([T, I, T] if p % 2 else [T, P, P])]
    C, _ = IA.count_matrix(frame(rows), "lab")
    point, ci, undef = IA.bootstrap(C, B=200, seed=3)
    assert ci["R_E"][0] <= point["R_E"] <= ci["R_E"][1]
    assert undef["OE_Indeterminate"] == 200        # no patient has >= 2 Ind visits: E = 0


@pytest.mark.parametrize("dci,oci,E,expected", [
    # amended rule (Amendment 1): each contrast must establish its direction
    ((0.05, 0.3), (0.3, 0.9), 50, "consistent with transient boundary-crossing"),
    ((0.05, 0.3), (0.6, 1.2), 50, "inconclusive"),             # O/E spans 1
    ((-0.3, -0.1), (1.2, 2.0), 50, "consistent with sustained intermediate passage"),
    ((-0.3, 0.0), (1.2, 2.0), 50, "consistent with sustained intermediate passage"),
    ((-0.2, 0.1), (1.1, 2.0), 50, "inconclusive"),             # delta_R spans 0
    ((0.05, 0.3), (1.1, 2.0), 50, "inconclusive"),             # contrasts disagree
    ((-0.3, -0.1), (0.3, 0.9), 50, "inconclusive"),            # contrasts disagree
    ((-0.1, 0.2), (0.8, 1.3), 50, "inconclusive"),
    ((0.05, 0.3), (0.3, 0.9), 19.9, "inconclusive (clustering not evaluable: E < 20)"),
])
def test_reading_rule_amended(dci, oci, E, expected):
    assert IA.reading(dci, oci, E) == expected


@pytest.mark.parametrize("dci,oci,expected", [
    ((0.05, 0.3), (0.6, 1.2), "consistent with transient boundary-crossing"),
    ((-0.2, 0.1), (1.1, 2.0), "consistent with sustained intermediate passage"),
    ((0.05, 0.3), (1.1, 2.0), "inconclusive"),
])
def test_superseded_rule_kept_for_record(dci, oci, expected):
    assert IA.reading_original(dci, oci, 50) == expected


def test_exit_episodes_and_no_followup():
    df = frame([(1, 0.0, T, 1), (1, 1.0, I, 1), (1, 2.5, I, 1), (1, 3.0, P, 1),
                (2, 0.0, I, 1), (2, 2.0, I, 1),
                (3, 0.0, T, 1), (3, 1.0, I, 1)])
    e = IA.exit_episodes(df, "lab", I).set_index("PATNO")
    assert e.loc[1, "event"] and e.loc[1, "time"] == pytest.approx(2.0)
    assert not e.loc[2, "event"] and e.loc[2, "time"] == pytest.approx(2.0)
    assert not e.loc[3, "followed"]


def test_wilson_interval():
    lo, hi = IA.wilson(5, 10)
    assert 0 < lo < 0.5 < hi < 1
    assert np.isnan(IA.wilson(0, 0)[0])
