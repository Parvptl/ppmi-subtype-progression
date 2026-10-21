"""Tests for the Rev 3 recovery-test driver (scripts/59_m2_rev3_recovery_test.py; docs 46/47 R1.3)."""

import importlib.util
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from ppmi_subtype import grm as G
from ppmi_subtype import m2_grm as M
from ppmi_subtype import m2_quad as MQ

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _load("rev3_driver_test", ROOT / "scripts/59_m2_rev3_recovery_test.py")
D54 = _load("d54_rev3_test", ROOT / "scripts/54_m2_step6_diag_compute.py")
R53 = _load("r53_rev3_test", ROOT / "scripts/53_m2_step6_diag_readonly.py")
S1A = _load("s1a_rev3_test", ROOT / "scripts/51_m2a1_s1a_true_latent_items.py")
NG = len(D.GATE_NAMES)


# ---------------- gate values, statuses, verdict ---------------------------------------------------
def test_gate_values_inclusive_and_undefined():
    z, d, c = D.gate_values([1.96, 2.0, 1.0, np.nan], [1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0])
    assert c.tolist() == [True, False, False, False]
    assert d.tolist() == [True, True, False, False] and np.isnan(z[2]) and np.isnan(z[3])


def _arrays(n=100, ncov=None, bias=0.0, undefined=None):
    z = np.zeros((n, NG)) + bias
    if ncov is not None:
        for g, k in ncov.items():
            z[k:, g] = 3.0 + bias                       # datasets beyond k not covered
    d = np.ones((n, NG), bool)
    for g, m in (undefined or {}).items():
        d[:m, g] = False
        z[:m, g] = np.nan
    c = d & (np.abs(np.where(d, z, np.inf)) <= 1.96)
    return z, d, c


def test_gate_status_thresholds():
    assert math.ceil(0.88 * 100) == 88 and math.floor(0.05 * 100) == 5
    z, d, c = _arrays(ncov={0: 88, 1: 87})
    st = D.gate_status(z, d, c)
    assert st[0] == "pass" and st[1] == "fail"
    z, d, c = _arrays(bias=0.40)
    assert D.gate_status(z, d, c)[2] == "pass"
    z, d, c = _arrays(bias=0.41)
    assert D.gate_status(z, d, c)[2] == "fail"
    z, d, c = _arrays(undefined={3: 5, 4: 6})
    st = D.gate_status(z, d, c)
    assert st[3] == "pass" and st[4] == "ne"                # 5 undefined allowed (not covered: 95 ≥ 88)


def test_verdict_precedence_and_boundaries():
    n = 100
    z, d, c = _arrays()
    fl = np.zeros(n, bool)
    assert D.verdict("INTEGRITY STOP: x", "y", z, d, c, fl)["verdict"] == "INTEGRITY STOP"
    assert D.verdict(None, "INCOMPLETE: x", z, d, c, fl)["verdict"] == "INCOMPLETE"
    assert D.verdict(None, None, z, d, c, fl)["verdict"] == "PASS"
    # robust failure of a recovery gate (Le00_log has no exemption)
    j = D.GATE_NAMES.index("Le00_log")
    z2, d2, c2 = _arrays(bias=0.0)
    z2[:, j] += 0.6
    c2 = d2 & (np.abs(z2) <= 1.96)
    v = D.verdict(None, None, z2, d2, c2, fl)
    assert v["verdict"] == "FAIL" and "Le00_log" in v["robust_failures"]
    # more than 10 flagged → INCONCLUSIVE even with a robust failure (R8)
    fl11 = np.zeros(n, bool)
    fl11[:11] = True
    assert D.verdict(None, None, z2, d2, c2, fl11)["verdict"] == "INCONCLUSIVE — FLAG EXCESS"
    fl10 = np.zeros(n, bool)
    fl10[:10] = True
    assert D.verdict(None, None, z, d, c, fl10)["verdict"] == "PASS"
    # guard-only failure → INCONCLUSIVE (GUARD), never FAIL
    zg, dg, cg = _arrays()
    zg[:, 21] += 0.6
    v = D.verdict(None, None, zg, dg, np.abs(zg) <= 1.96, fl)
    assert v["verdict"] == "INCONCLUSIVE — OTHER" and "GUARD" in v["reasons"]
    # C0 failure alone → INCONCLUSIVE (NE), never FAIL
    zn, dn, cn = _arrays(undefined={0: 6})
    v = D.verdict(None, None, zn, dn, cn, fl)
    assert v["verdict"] == "INCONCLUSIVE — OTHER" and "NE" in v["reasons"]
    # failure only after excluding flagged datasets → INCONCLUSIVE (FLAG-SENSITIVE); exclusion cannot create PASS
    zf, df, cf = _arrays()
    flf = np.zeros(n, bool)
    flf[:10] = True
    zf[10:20, 0] = 3.0                                       # 10 non-covered among the 90 unflagged: 80 < 80? -> 80/90
    zf[10:13, 0] = 3.0
    cf = df & (np.abs(zf) <= 1.96)
    vA = D.gate_status(zf, df, cf)[0]
    vAF = D.gate_status(zf[~flf], df[~flf], cf[~flf])[0]
    v = D.verdict(None, None, zf, df, cf, flf)
    if vA != vAF:
        assert v["verdict"] == "INCONCLUSIVE — OTHER" and "FLAG-SENSITIVE" in v["reasons"]
    za, da, ca = _arrays()
    fla = np.zeros(n, bool)
    fla[:5] = True
    za[:13, 1] = 3.0                                          # fails on A (87 covered), passes on A\F (87/95 ≥ 84)
    ca = da & (np.abs(za) <= 1.96)
    v = D.verdict(None, None, za, da, ca, fla)
    assert v["verdict"] != "PASS"


def test_failure_labels():
    rng = np.random.default_rng(0)
    zb = rng.normal(0.8, 1.0, 100)
    lab, _ = D.failure_label(zb, np.ones(100, bool), np.abs(zb) <= 1.96)
    assert lab in ("bias-type", "mixed")
    zd = rng.normal(0.0, 1.6, 100)
    lab, lo = D.failure_label(zd, np.ones(100, bool), np.abs(zd) <= 1.96)
    assert lab == "dispersion-type" and lo > 1
    zm = rng.normal(0.8, 1.8, 100)
    lab, _ = D.failure_label(zm, np.ones(100, bool), np.abs(zm) <= 1.96)
    assert lab == "mixed"


def test_runtime_gate_formula_and_missing():
    g = D.runtime_gate(1.0, [800.0, 820.0, 840.0], 300.0, [200.0, 300.0], 0.1, 13.5)
    P = 1.0 + 97 * 820.0 / 3600 + 10 * 300.0 / 3600
    assert g["P_node_hours"] == pytest.approx(P) and g["passed"]
    assert not D.runtime_gate(1.0, [800.0, np.nan, 840.0], 300.0, [200.0], 0.1, 13.5)["passed"]
    assert not D.runtime_gate(1.0, [800.0, 820.0], 300.0, [200.0], 0.1, 13.5)["passed"]
    assert not D.runtime_gate(1.0, [800.0, 820.0, 840.0], 300.0, [3601.0], 0.1, 13.5)["passed"]
    assert not D.runtime_gate(1.0, [1200.0, 1200.0, 1200.0], 300.0, [200.0], 0.1, 13.5)["passed"]   # P > 28 / per-job


def test_masks_and_truth_free_reexpression():
    assert [n for n, m in zip(D.PSI_NAMES, D.mask_psi("T")) if m] == ["beta0_T", "beta1_T", "gamma_T", "L10"]
    assert [n for n, m in zip(D.PSI_NAMES, D.mask_psi("P")) if m] == ["beta0_P", "beta1_P", "gamma_P", "L20", "L21",
                                                                         "L30", "L31", "L32", "Le10"]
    assert np.flatnonzero(D.mask_phi("P")).tolist() == [55 + 5 * k + 1 for k in range(5)]
    assert list(inspect.signature(D.Ctx.reexpress).parameters) == ["self", "psi"]
    assert len(D.GATE_NAMES) == 36 and len(D.RECOVERY_NAMES) == 21 and len(D.GUARD_NAMES) == 15


def test_write_atomic_refuses_overwrite(tmp_path):
    p = tmp_path / "x.json"
    D.write_atomic(p, {"a": 1})
    with pytest.raises(FileExistsError):
        D.write_atomic(p, {"a": 2})


# ---------------- controlled cases for the exact N3 / N4 identities (B3, B5) ------------------------
def _toy(seed=0, m=6, n=3):
    rng = np.random.default_rng(seed)
    T = np.sort(rng.uniform(0, 4, (m, n)), axis=1)
    X = (rng.random((m, n)) < 0.4).astype(float)
    it = M.Items(np.linspace(1.2, 2.0, 4), np.linspace(-0.5, 1.0, 4)[:, None] + 0.7 * np.arange(4),
                 np.linspace(1.5, 2.2, 3), np.linspace(0.0, 1.0, 3)[:, None] + 0.7 * np.arange(4))
    YT = rng.integers(0, 5, (m, n, 4)).astype(float)
    YP = rng.integers(0, 5, (m, n, 3)).astype(float)
    return {n: M.Group(T, X, YT, YP, list(range(m)))}, it


def _gamma1_toy(sub):
    m = np.zeros(35, bool)
    if sub == "T":
        m[[5 * k + 1 for k in range(4)]] = True
    else:
        m[[20 + 5 * k + 1 for k in range(3)]] = True
    return m


def test_n3_identity_with_gradient_term_on_controlled_case():
    g, it = _toy()
    psi = M.default_psi() + np.random.default_rng(1).normal(0, 0.1, M.N_PSI)   # not an optimum: ‖∇‖ ≠ 0
    pT, pP = D54.pack_set(it)
    phi = np.r_[pT, pP]
    f = lambda ps, ph: MQ.negloglik(ps, g, D54.items_from_phi(ph[:20], ph[20:], 4, 3), 7)[1]   # noqa: E731
    g4 = f(psi, phi)
    h = 1e-5
    H = np.column_stack([(f(psi + h * e, phi) - f(psi - h * e, phi)) / (2 * h) for e in np.eye(19)])
    C = np.column_stack([(f(psi, phi + h * e) - f(psi, phi - h * e)) / (2 * h) for e in np.eye(35)])
    assert np.linalg.norm(g4) > 1e-2
    worst_with, worst_without = 0.0, 0.0
    for sub in ("T", "P"):
        vphi, vpsi = D54.affine_tangent(it, psi, sub, "c", R53.relabel_psi)
        base = C @ vphi + H @ vpsi
        den = np.linalg.norm(H @ vpsi)
        worst_with = max(worst_with, np.linalg.norm(base - D.mask_psi(sub) * g4) / den)
        worst_without = max(worst_without, np.linalg.norm(base) / den)
    assert worst_with <= 1e-5 < worst_without


def test_n4_identity_with_gradient_term_on_controlled_case():
    g, it = _toy(seed=3)
    psi = M.default_psi() + np.random.default_rng(4).normal(0, 0.1, M.N_PSI)
    grp = next(iter(g.values()))
    YT = grp.YT.reshape(-1, 4)
    YP = grp.YP.reshape(-1, 3)
    pT, pP = D54.pack_set(it)
    phi = np.r_[pT, pP] + np.random.default_rng(5).normal(0, 0.05, 35)                # not an optimum

    def grad(ph, ps):
        vl = S1A.visit_latent(g, ps)
        return np.r_[S1A.latent_negloglik(ph[:20], YT, *vl["T"])[1], S1A.latent_negloglik(ph[20:], YP, *vl["P"])[1]]
    g3 = grad(phi, psi)
    h = 1e-5
    A3 = np.column_stack([(grad(phi + h * e, psi) - grad(phi - h * e, psi)) / (2 * h) for e in np.eye(35)])
    E3 = np.column_stack([(grad(phi, psi + h * e) - grad(phi, psi - h * e)) / (2 * h) for e in np.eye(19)])
    it_phi = D54.items_from_phi(phi[:20], phi[20:], 4, 3)
    for sub in ("T", "P"):
        for par in ("c", "d"):
            vphi, vpsi = D54.affine_tangent(it_phi, psi, sub, par, R53.relabel_psi)
            base = A3 @ vphi + E3 @ vpsi
            corr = _gamma1_toy(sub) * g3 if par == "c" else 0
            rel = np.linalg.norm(base - corr) / np.linalg.norm(A3 @ vphi)
            assert rel <= 1e-5, (sub, par, rel)
            if par == "c":
                assert np.linalg.norm(base) / np.linalg.norm(A3 @ vphi) > 1e-4
