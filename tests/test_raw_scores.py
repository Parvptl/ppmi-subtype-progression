"""Tests for item-level score reconstruction (Part C1).

Weight-bearing: reconstruction follows the pipeline (OFF selection,
deduplication) and gives the pipeline's subscores and labels; sums are 11x / 5x
the means; 101 and out-of-range values are missing, never scores; missing,
PIGD = 0 -> TD and both-zero -> Indeterminate stay distinct; sensitivity
constructions change only what they claim to; the lattice margin is exact; the
genetic cohort excludes prodromal carriers and keeps combined labels; OFF/ON
pairs are aligned on PATNO, EVENT_ID and INFODT.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ppmi_subtype import raw_scores as RS
from ppmi_subtype.config import load_config
from ppmi_subtype.subtype import classify, compute_subscores


@pytest.fixture(scope="module")
def scfg():
    return load_config().subtype


def visit(scfg, patno, event, tremor=0, pigd=0, **override):
    row = {"PATNO": patno, "EVENT_ID": event}
    for c in scfg.tremor_items:
        row[c] = tremor
    for c in scfg.pigd_items:
        row[c] = pigd
    row.update(override)
    return row


def split_parts(scfg, rows, state="OFF", treat=1):
    df = pd.DataFrame(rows)
    p2 = df[["PATNO", "EVENT_ID"] + list(RS.PART2_ITEMS)]
    p3 = df.drop(columns=list(RS.PART2_ITEMS)).assign(PDSTATE=state, PDTRTMNT=treat)
    return p2, p3


def test_item_table_selects_off_and_deduplicates(scfg):
    rows = [visit(scfg, 1, "BL", 1, 1)]
    p2, p3 = split_parts(scfg, rows)
    on = p3.assign(PDSTATE="ON", NP3GAIT=4)
    tab = RS.item_visit_table(p2, pd.concat([p3, on]), scfg)
    assert len(tab) == 1 and tab["NP3GAIT"].iloc[0] == 1 and tab["has_part3"].iloc[0]
    # a visit with Part II only has no Part III exam
    p2b = pd.concat([p2, p2.assign(EVENT_ID="V04")])
    tab2 = RS.item_visit_table(p2b, p3, scfg)
    assert not tab2.set_index("EVENT_ID").loc["V04", "has_part3"]


def test_sums_are_item_count_times_mean_and_invalid_values_are_missing(scfg):
    rows = [visit(scfg, 1, "BL", 2, 1), visit(scfg, 2, "BL", 1, 1, NP3PSTBL=101),
            visit(scfg, 3, "BL", 1, 1, NP3RTCON=7)]
    tab = pd.DataFrame(rows)
    t, p = RS.item_sums(tab, scfg)
    sc = compute_subscores(tab, scfg)
    assert t[0] == pytest.approx(11 * sc.tremor_score[0]) and p[0] == pytest.approx(5 * sc.pigd_score[0])
    assert np.isnan(p[1]) and np.isnan(sc.pigd_score[1])          # 101 never averaged
    assert np.isnan(t[2]) and np.isnan(sc.tremor_score[2])        # out of range
    tab["has_part3"] = True
    reasons = RS.missingness_reasons(tab, scfg)
    assert list(reasons["reason"]) == ["labelled", "unrateable_101", "out_of_range"]
    assert reasons["pigd_n_101"][1] == 1


def test_zero_states_stay_distinct(scfg):
    rows = [visit(scfg, 1, "BL", 1, 0), visit(scfg, 2, "BL", 0, 0),
            visit(scfg, 3, "BL", 1, 0, NP2WALK=np.nan)]
    sc = compute_subscores(pd.DataFrame(rows), scfg)
    assert list(sc.subtype_name.iloc[:2]) == ["TD", "Indeterminate"]
    assert not sc.ratio_defined.iloc[0] and not sc.ratio_defined.iloc[1]
    assert pd.isna(sc.subtype_name.iloc[2])                      # missing, not TD


def test_one_missing_pigd_item_sensitivity_labels_only_those_visits(scfg):
    rows = [visit(scfg, 1, "BL", 1, 1, NP3PSTBL=101), visit(scfg, 2, "BL", 1, 1,
                                                            NP3PSTBL=101, NP3GAIT=101)]
    tab = pd.DataFrame(rows)
    base = compute_subscores(tab, scfg)
    sens = RS.labels_under(tab, scfg, max_missing_pigd_items=1)
    assert base.subtype.isna().all()
    assert sens.subtype.notna().tolist() == [True, False]
    assert sens.pigd_score[0] == pytest.approx(1.0)               # mean of the 4 observed


def test_inner_band_changes_only_exact_boundary_ratios(scfg):
    t = pd.Series([1.15, 0.9, 1.0, 2.0])
    p = pd.Series([1.0, 1.0, 1.0, 1.0])
    outer = classify(t, p, scfg)
    inner = classify(t, p, dataclasses.replace(scfg, band_boundaries="inner"))
    assert list(outer) == [1, 3, 2, 1] and list(inner) == [2, 2, 2, 1]
    tab = pd.DataFrame([visit(scfg, 1, "BL", 1, 1)])
    assert RS.labels_under(tab, scfg, band_boundaries="inner").subtype_name[0] == "Indeterminate"


def test_part3_only_excludes_part2_items(scfg):
    ti, pi = RS.part3_only(scfg)
    assert len(ti) == 10 and len(pi) == 3 and not set(ti + pi) & set(RS.PART2_ITEMS)
    rows = [visit(scfg, 1, "BL", 0, 0, NP2TRMR=4, NP3GAIT=1)]       # tremor only from Part II
    lab = RS.part3_only_labels(pd.DataFrame(rows), scfg)
    assert lab.tremor_p3[0] == 0 and lab.pigd_p3[0] == pytest.approx(1 / 3)
    assert lab.subtype_name_p3[0] == "PIGD"


def test_label_grid_and_margin(scfg):
    grid = RS.label_grid(scfg)
    assert grid.shape == (45, 21)
    assert grid[0, 0] == 2 and grid[1, 0] == 1                     # both zero Ind; PIGD 0 TD
    assert grid[11, 5] == 2                                        # ratio 1.0 -> Ind
    m = RS.margin_grid(grid)
    assert m[0, 0] == 1 and m[11, 5] == 1
    # independent check: brute force L1 distance on a few cells
    for (a, b) in [(30, 2), (3, 10), (11, 5), (44, 20)]:
        best = min(abs(a - x) + abs(b - y) for x in range(45) for y in range(21)
                   if grid[x, y] != grid[a, b])
        assert m[a, b] == best
    assert np.isnan(RS.label_margin([np.nan], [1], m)[0])


def test_groupings():
    g = RS.pigd_group([0, 1, 2, 3, 4, 5, 6, 20, np.nan])
    assert list(g.iloc[:8]) == ["0", "1", "2", "3", "4-5", "4-5", "6+", "6+"] and pd.isna(g.iloc[8])
    edges = RS.quintile_edges([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    q = RS.quintile_group([0, 1, 7, np.nan], edges)
    assert q.iloc[0] == "Q1" and q.iloc[2] == "Q5" and pd.isna(q.iloc[3])


def test_genetic_mask_excludes_prodromal_and_keeps_combinations():
    cur = pd.DataFrame({"COHORT": [1, 4, 1, 1, 1, 1, 2],
                        "subgroup": ["LRRK2", "LRRK2", "LRRK2 + GBA", "Normosmic",
                                     "Sporadic PD", "LRRK2 + Normosmic", "GBA"]})
    assert RS.genetic_pd_mask(cur).tolist() == [True, False, True, False, False, True, False]


def test_off_on_pairs_alignment(scfg):
    rows = [visit(scfg, 1, "V04", 2, 1), visit(scfg, 2, "V04", 2, 1), visit(scfg, 3, "V04", 2, 1)]
    p2, p3 = split_parts(scfg, rows)
    p3 = p3.assign(INFODT="01/2020")
    on = p3.assign(PDSTATE="ON", NP3RTARU=0)
    on.loc[on.PATNO == 2, "INFODT"] = "02/2020"                     # different day: not paired
    extra = p3[p3.PATNO == 3]                                       # duplicate OFF row: not paired
    pairs = RS.off_on_pairs(p2, pd.concat([p3, on, extra]), scfg)
    assert pairs.PATNO.tolist() == [1]
    assert pairs.tremor_sum_ON[0] == pairs.tremor_sum_OFF[0] - 2


def test_patient_bootstrap_and_transition_table():
    f = pd.DataFrame({"PATNO": [1, 1, 1, 2, 2, 3, 3], "years_from_baseline": [0, 1, 2, 0, 1, 0, 3],
                      "g": ["0", "1", "1", "1", "0", "0", "0"]})
    tab = RS.transition_table(f, "g", ["0", "1"], B=200, seed=1)
    for (iv, fr), sub in tab.groupby(["interval", "from"]):
        if sub.n_from.iloc[0]:
            assert sub.prob.sum() == pytest.approx(1.0)
            assert ((sub.ci_lo <= sub.prob + 1e-12) & (sub.prob <= sub.ci_hi + 1e-12)).all()
    long = tab[tab.interval == "> 1.5"]
    assert long.n_pairs.sum() == 1                                   # patient 3's 3-year gap


def test_example_patients_allocation_and_fallback():
    rng = np.random.default_rng(0)
    f = pd.DataFrame({"PATNO": np.arange(300), "baseline_label": rng.choice(["TD", "Indeterminate",
                                                                             "PIGD"], 300),
                      "n_visits": rng.integers(2, 12, 300)})
    sel, log, edges, strata = RS.example_patients(f, 16, 42)
    sel2, *_ = RS.example_patients(f, 16, 42)
    assert len(sel) == 16 and sorted(sel.PATNO) == sorted(sel2.PATNO) and not log
    assert strata["alloc"].sum() == 16
    tiny = f.head(5)
    sel3, log3, *_ = RS.example_patients(tiny, 16, 42)
    assert len(sel3) == 5 and any("could not be filled" in m for m in log3)


def test_visit_frame_reproduces_panel_and_flags_mismatch(scfg):
    rows = [visit(scfg, 1, "BL", 2, 1), visit(scfg, 1, "V04", 1, 0),
            visit(scfg, 2, "BL", 1, 1, NP3PSTBL=101)]
    p2, p3 = split_parts(scfg, rows)
    tab = RS.item_visit_table(p2, p3, scfg)
    sub = RS.subscores_from_items(tab, scfg)
    panel = sub.copy()
    panel.loc[len(panel)] = {"PATNO": 3, "EVENT_ID": "BL"}          # visit with no item rows
    f = RS.build_visit_frame(panel, tab, scfg)
    v = RS.validate_reconstruction(f)
    assert v["exact"] and v["visits_checked"] == 4
    assert f.set_index(["PATNO", "EVENT_ID"]).loc[(3, "BL"), "reason"] == "no_item_rows"
    assert f.set_index(["PATNO", "EVENT_ID"]).loc[(2, "BL"), "subtype_name_miss1"] == "Indeterminate"
    bad = f.copy()
    bad.loc[0, "subtype_name"] = "PIGD"
    assert not RS.validate_reconstruction(bad)["exact"]
