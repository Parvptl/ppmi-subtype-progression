"""Tests for the Step 6 full-pipeline recovery driver (plan doc 37 §11 (a)-(e))."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ppmi_subtype import m2_grm as M

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S6 = _load("s6", ROOT / "scripts/52_m2a1_full_pipeline_recovery.py")
S1A = _load("s1a", ROOT / "scripts/51_m2a1_s1a_true_latent_items.py")


# (a) coverage logic ------------------------------------------------------------------------------
def test_coverage_pd_and_inclusive_boundary():
    H = np.diag([4.0, 1.0, 0.25])                       # SE = 0.5, 1, 2
    truth = np.zeros(3)
    est = np.array([1.96 * 0.5, 2.5, -1.0])             # z = 1.96 (inclusive), 2.5, -0.5
    c = S6.coverage(est, truth, H)
    assert c["hessian_pd"]
    np.testing.assert_allclose(c["se"], [0.5, 1.0, 2.0])
    assert c["covered"].tolist() == [True, False, True]


@pytest.mark.parametrize("H", [np.diag([1.0, -1.0, 1.0]), np.diag([1.0, 0.0, 1.0]),
                               np.array([[1.0, np.nan, 0], [np.nan, 1, 0], [0, 0, 1]])])
def test_coverage_non_pd_singular_nonfinite_cover_nothing(H):
    c = S6.coverage(np.zeros(3), np.zeros(3), H)
    assert not c["hessian_pd"] and not c["covered"].any()


def test_coverage_nonfinite_se_not_covered(monkeypatch):
    real_inv = np.linalg.inv
    monkeypatch.setattr(np.linalg, "inv", lambda A: real_inv(A) * np.array([[1, 1], [1, np.nan]]))
    c = S6.coverage(np.zeros(2), np.zeros(2), np.eye(2))
    assert c["covered"].tolist() == [True, False] and c["missing_interval"] == [1]


# (b) outcome classes and the dependence statement --------------------------------------------------
def _cov(n_cov_per_ds):
    return [np.arange(19) < k for k in n_cov_per_ds]


def test_classify_classes():
    assert S6.classify("INTEGRITY STOP: x", 0, 5, [], [])["class"] == "INTEGRITY STOP"
    inc = S6.classify("dataset 2 raised an execution error", 2, 5, _cov([19, 19]), [])
    assert inc["class"] == "INCOMPLETE" and inc["formal_verdict"] is None
    assert S6.classify(None, 4, 5, _cov([19] * 4), [])["class"] == "INCOMPLETE"
    ok = S6.classify(None, 5, 5, _cov([19, 19, 19, 19, 0]), [])
    assert ok["class"] == "COMPLETE, no flagged event" and ok["formal_verdict"] == "PASS"
    bad = S6.classify(None, 5, 5, _cov([19, 19, 19, 0, 0]), [])
    assert bad["formal_verdict"] == "FAIL" and not bad["criterion_met_as_computed"]


def test_classify_flagged_reserved_with_dependence():
    r = S6.classify(None, 5, 5, _cov([19, 19, 19, 19, 0]), [4])
    assert r["class"] == "COMPLETE, with flagged events" and r["formal_verdict"] == "RESERVED FOR REVIEW"
    assert r["criterion_met_as_computed"] and r["dependence"]["met_if_flagged_all_covered"]
    assert r["dependence"]["met_if_flagged_none_covered"] and not r["dependence"]["outcome_depends_on_flagged"]
    r = S6.classify(None, 5, 5, _cov([19, 19, 19, 0, 19]), [4])
    assert r["dependence"]["outcome_depends_on_flagged"] and r["formal_verdict"] == "RESERVED FOR REVIEW"


def test_flags_for():
    fit = {"converged": True, "stopped_by_time": False, "maxiter_reached": False, "at_bound": []}
    it = {"converged": True, "a_out_of_range": False}
    d = {"stage2_fit_1": dict(fit), "stage2_fit_2": dict(fit), "items_1": {"T": dict(it), "P": dict(it)},
         "items_2": {"T": dict(it), "P": dict(it)}, "reported_2": {"n_fallback_patients": 0, "budget_triggers": 0}}
    assert S6.flags_for(d) == []
    d["stage2_fit_1"]["converged"] = False
    d["items_2"]["P"]["a_out_of_range"] = True
    d["stage2_fit_2"]["at_bound"] = ["L00_log"]
    d["reported_2"]["n_fallback_patients"] = 1
    assert S6.flags_for(d) == ["F1:stage2_fit_1", "F2:items_2_P", "F3:psi2_at_bound", "F4:psi2_fallback_or_budget"]


# (c) step sequence with stubs ----------------------------------------------------------------------
def test_pipeline_sequence_once_each_default_starts_no_rng(monkeypatch):
    calls = []
    psi_default = M.default_psi()
    it1, it2 = object(), object()

    class Ops:
        def fit_stage2(self, sg, it, psi0):
            calls.append(("fit", it, np.array(psi0)))
            return SimpleNamespace(psi=np.full(19, len(calls)), state={})

        def hessian(self, sg, it, f):
            calls.append(("hess", it))
            return {"H": np.eye(19)}

        def reported(self, sg, it, f):
            calls.append(("rep", it))
            return {}

        def refit_items(self, sg, psi1):
            calls.append(("s1b", psi1.copy()))
            return {"items": it2}

    def no_rng(*a, **k):
        raise AssertionError("no RNG may be drawn inside the pipeline")
    monkeypatch.setattr(np.random, "default_rng", no_rng)
    rec = S6.pipeline_dataset({}, it1, Ops(), psi_default)
    kinds = [c[0] for c in calls]
    assert kinds == ["fit", "hess", "rep", "s1b", "fit", "hess", "rep"]
    assert calls[0][1] is it1 and calls[4][1] is it2 and calls[5][1] is it2 and calls[6][1] is it2
    np.testing.assert_array_equal(calls[0][2], psi_default)
    np.testing.assert_array_equal(calls[4][2], psi_default)            # E-4: default ψ, not ψ̂(1)
    np.testing.assert_array_equal(calls[3][1], rec["fit_1"].psi)      # S1-b at ψ̂(1)


# (d) runtime projection (frozen doc 25 formula) ----------------------------------------------------
def test_projection_formula_and_limits():
    g = S6.projection(200.0, 250.0, 1.5, 70.0, 55.0)
    expect = (18 * 250 + 11 * 38 * 1.5 + 716 + 0.62 * 3600) / 3600
    assert g["projected_pilot_node_hours"] == pytest.approx(expect, rel=1e-15) and g["gate_passed"]
    assert g["variant_measured_s1b_node_hours"] == pytest.approx(
        (18 * 250 + 11 * 38 * 1.5 + 716 + 5 * 125) / 3600, rel=1e-15)
    assert not S6.projection(3601.0, 10.0, 1.0)["gate_passed"]
    assert not S6.projection(10.0, 3601.0, 1.0)["gate_passed"]
    assert not S6.projection(10.0, 10.0, 250.0)["pilot_le_24"]          # 11 x 38 x 250 s > 24 h


# (e) visit_latent uses M.unpack(ψ), matching the Step 5 construction -------------------------------
def test_visit_latent_matches_unpack_construction():
    rng = np.random.default_rng(0)
    m, n = 3, 4
    T = np.sort(rng.uniform(0, 5, (m, n)), axis=1)
    X = (rng.random((m, n)) < 0.5).astype(float)
    g = {n: M.Group(T, X, np.zeros((m, n, 11)), np.zeros((m, n, 5)), list(range(m)))}
    psi = M.default_psi() + rng.normal(0, 0.1, M.N_PSI)
    out = S1A.visit_latent(g, psi)
    beta, D, _, S, _ = M.unpack(psi)
    Mn = np.einsum("k,kmi->mi", beta, M.mean_design(T, X))
    K, _ = M.covariance(T, D, S)
    dg = np.einsum("mii->mi", K)
    np.testing.assert_array_equal(out["T"][0], Mn[:, :n].ravel())
    np.testing.assert_array_equal(out["P"][1], np.sqrt(dg[:, n:]).ravel())
