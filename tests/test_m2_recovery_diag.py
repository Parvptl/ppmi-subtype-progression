"""Tests for the fixed-item recovery diagnostic's reuse of the pilot's truth (scripts 45, 46)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ppmi_subtype import m2_grm as M

ROOT = Path(__file__).resolve().parents[1]


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pilot():
    return _load("m2_pilot", "45_m2_pilot.py")


@pytest.fixture(scope="module")
def design():
    rng = np.random.default_rng(0)
    times, treat, YT, YP = {}, {}, {}, {}
    for p in range(300):
        n = int(rng.integers(2, 7))
        t = np.sort(rng.uniform(0, 8, n))
        times[p], treat[p] = t, (t > 1.5).astype(float)
        YT[p], YP[p] = np.zeros((n, 11)), np.zeros((n, 5))
    return M.make_groups(times, treat, YT, YP)


def test_truth_is_on_stage1_scale(pilot, design):
    """Pooled visit-level θ^T and θ^P have mean 0 and variance 1 at the design."""
    psi, it, scale = pilot.truth_on_stage1_scale(design)
    beta, D, _, S, _ = M.unpack(psi)
    for s, sl in ((0, lambda n: slice(0, n)), (1, lambda n: slice(n, 2 * n))):
        mus, vs = [], []
        for n, g in design.items():
            Mn = np.einsum("k,kmi->mi", beta, M.mean_design(g.T, g.X))[:, sl(n)]
            K, _ = M.covariance(g.T, D, S)
            mus.append(Mn.ravel())
            vs.append(np.einsum("mii->mi", K)[:, sl(n)].ravel())
        mu, var_in = np.concatenate(mus), np.concatenate(vs)
        assert mu.mean() == pytest.approx(0.0, abs=1e-10)
        assert var_in.mean() + mu.var() == pytest.approx(1.0, abs=1e-6)


def test_rescaled_truth_preserves_item_distribution(pilot, design):
    """P(Y >= c | θ) under the raw truth equals that under the rescaled truth at θ' = (θ - μ)/sd."""
    _, it, scale = pilot.truth_on_stage1_scale(design)
    T = pilot.TRUTH_ITEMS
    tauT = T["tauT1"][:, None] + T["step"] * np.arange(4)
    theta = np.linspace(-3, 3, 13)
    raw = 1 / (1 + np.exp(-T["aT"][:, None, None] * (theta[None, :, None] - tauT[:, None, :])))
    th2 = (theta - scale["mu"][0]) / scale["sd"][0]
    new = 1 / (1 + np.exp(-it.aT[:, None, None] * (th2[None, :, None] - it.tauT[:, None, :])))
    assert np.allclose(raw, new)


def test_diagnostic_script_imports_and_uses_pilot_functions(pilot):
    diag = _load("m2_diag", "46_m2_recovery_fixed_items.py")
    mod = diag.load_pilot_module()
    for fn in ("truth_on_stage1_scale", "simulate_groups", "regroup", "load_cohort", "flatten"):
        assert hasattr(mod, fn)
    assert mod.TRUTH_LONG == pilot.TRUTH_LONG
