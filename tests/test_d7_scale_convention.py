"""Tests for the D7 scale-convention diagnostic (scripts/55_m2_scale_convention_readonly.py)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from ppmi_subtype import m2_grm as M

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D7 = _load("d7_test", ROOT / "scripts/55_m2_scale_convention_readonly.py")
P = _load("pilot_d7_test", ROOT / "scripts/45_m2_pilot.py")


def _groups(seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for n, m in ((2, 7), (4, 5)):
        T = np.sort(rng.uniform(0, 6, (m, n)), axis=1)
        T[:, 0] = 0.0
        X = (rng.random((m, n)) < 0.4).astype(float)
        out[n] = M.Group(T, X, np.zeros((m, n, 11)), np.zeros((m, n, 5)), list(range(m)))
    return out


def test_matches_truth_on_stage1_scale_and_fixes_its_output():
    g = _groups()
    psi_t, _, scale = P.truth_on_stage1_scale(g)
    s_raw, mom = D7.standardise(g, np.array(scale["psi_raw"]))
    np.testing.assert_allclose(s_raw, psi_t, rtol=0, atol=1e-12)
    np.testing.assert_allclose([mom["mu_T"], mom["mu_P"]], scale["mu"], rtol=1e-12)
    np.testing.assert_allclose([mom["c_T"], mom["c_P"]], scale["sd"], rtol=1e-12)
    s_t, mom_t = D7.standardise(g, psi_t)
    # doc 43: tolerance 1e-8 = m2_grm.JITTER (the jitter does not rescale under relabelling)
    np.testing.assert_allclose(s_t, psi_t, rtol=0, atol=1e-8)
    assert abs(mom_t["mu_T"]) < 1e-8 and abs(mom_t["v_P"] - 1) < 1e-8


def test_idempotent_and_invariant_under_relabelling():
    g = _groups(1)
    psi = M.default_psi() + np.random.default_rng(2).normal(0, 0.1, M.N_PSI)
    s1, _ = D7.standardise(g, psi)
    np.testing.assert_allclose(D7.standardise(g, s1)[0], s1, rtol=0, atol=1e-8)        # doc 43
    rel = D7.R53.relabel_psi(psi, 1.13, -0.07, 0.91, 0.05)
    np.testing.assert_allclose(D7.standardise(g, rel)[0], s1, rtol=0, atol=1e-8)       # doc 43


def test_jacobian_annihilates_relabelling_directions():
    g = _groups(3)
    psi = M.default_psi() + np.random.default_rng(4).normal(0, 0.1, M.N_PSI)
    G = D7.jacobian(g, psi)
    for args in ((1 + 1e-6, 0, 1, 0), (1, 1e-6, 1, 0), (1, 0, 1 + 1e-6, 0), (1, 0, 1, 1e-6)):
        v = (D7.R53.relabel_psi(psi, *args) - psi) / 1e-6
        assert np.linalg.norm(G @ v) <= 1e-4 * np.linalg.norm(v)
    # and G acts as the identity on a direction that leaves the moments unchanged only in the truth's own frame
    assert np.linalg.matrix_rank(G, tol=1e-6) == M.N_PSI - 4
