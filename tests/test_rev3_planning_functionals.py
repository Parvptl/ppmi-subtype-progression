"""Invariance of the proposed convention-free functionals (scripts/56_m2_rev3_planning_checks.py)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from ppmi_subtype import grm as G
from ppmi_subtype import m2_grm as M

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("plan56_test", ROOT / "scripts/56_m2_rev3_planning_checks.py")
P56 = importlib.util.module_from_spec(spec)
sys.modules["plan56_test"] = P56
spec.loader.exec_module(P56)


def _items(seed):
    rng = np.random.default_rng(seed)
    aT, aP = rng.uniform(1, 2.5, 11), rng.uniform(1, 2.5, 5)
    tT = np.sort(rng.normal(0.5, 1, (11, 4)), axis=1)
    tP = np.sort(rng.normal(0.5, 1, (5, 4)), axis=1)
    return aT, tT, aP, tP


def test_seventeen_functionals_invariant_under_any_joint_relabelling():
    psi = M.default_psi() + np.random.default_rng(1).normal(0, 0.2, M.N_PSI)
    aT, tT, aP, tP = _items(2)
    f0 = P56.functionals(psi, G.pack_items(aT, tT), G.pack_items(aP, tP))
    assert f0.size == 17 and np.all(np.isfinite(f0))
    for cT, dT, cP, dP in ((1.2, 0.3, 0.8, -0.4), (0.7, -1.0, 1.5, 0.2)):
        psi2 = P56.D7.R53.relabel_psi(psi, cT, dT, cP, dP)
        f1 = P56.functionals(psi2, G.pack_items(aT * cT, (tT - dT) / cT), G.pack_items(aP * cP, (tP - dP) / cP))
        np.testing.assert_allclose(f1, f0, rtol=0, atol=1e-12)


def test_functionals_respond_to_identified_changes():
    """Not trivially constant: changing an identified quantity (a slope, a correlation) changes them."""
    psi = M.default_psi() + np.random.default_rng(3).normal(0, 0.2, M.N_PSI)
    aT, tT, aP, tP = _items(4)
    f0 = P56.functionals(psi, G.pack_items(aT, tT), G.pack_items(aP, tP))
    J = P56.jac(lambda v: P56.functionals(v[80:], v[:55], v[55:80]), np.r_[G.pack_items(aT, tT), G.pack_items(aP, tP), psi])
    assert np.linalg.matrix_rank(J[:15, 80:], tol=1e-8) == 15          # 15 ψ-only functionals: full rank in ψ
    assert np.linalg.matrix_rank(J, tol=1e-8) == 17
    assert f0.size == 17
