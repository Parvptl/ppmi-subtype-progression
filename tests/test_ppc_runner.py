"""Safeguard tests for the staged D1 runner.

Stage order, frozen-file validation, interruption and restart, threshold
immutability, PC4v exclusion from the verdict, degenerate observed handling,
exactly-once observed evaluation, protocol drift and post-freeze tampering.
A fast fake cohort function stands in for the real pipeline.
"""

from __future__ import annotations

import json
import os
import stat

import numpy as np
import pytest

from ppmi_subtype import ppc_diagnostic as D
from ppmi_subtype import ppc_runner as R

CAP = D.z_cap(500)
PROTO = {"hashes": {"report": "a", "code": "b"}, "panel_md5": "m", "fold_sha256": "f"}
CALLS = {"n": 0, "fail_after": None}


def fake(item):
    key, entropy, payload = item
    CALLS["n"] += 1
    if CALLS["fail_after"] is not None and CALLS["n"] > CALLS["fail_after"]:
        raise RuntimeError("simulated interruption")
    rng = np.random.default_rng(entropy)
    kind = payload["kind"]
    T = float(abs(rng.standard_normal()) + (2.5 if kind == "PC4" and rng.random() < 0.7 else 0))
    T = min(T, CAP)
    return {"kind": kind, "T": T, "z": {k: float(rng.standard_normal()) for k in D.STAT_NAMES},
            "S_obs": {k: 0.1 for k in D.STAT_NAMES}, "p_at_bound": 0}


def observed(T=1.0, undefined=False, at_bound=0):
    S = {k: 0.1 for k in D.STAT_NAMES}
    if undefined:
        S["S5_OE_ind"] = float("nan")
    return {"S_obs": S, "S_rep_mean": S, "p": {k: 0.5 for k in D.STAT_NAMES},
            "z": {k: 0.0 for k in D.STAT_NAMES}, "T": T, "p_at_bound": at_bound,
            "T_at_cap": bool(np.isclose(T, CAP)), "folds": [], "timing": {}}


@pytest.fixture(autouse=True)
def reset():
    CALLS.update(n=0, fail_after=None)


def test_full_order_and_exactly_once(tmp_path):
    run = R.D1Run(tmp_path)
    called = {"n": 0}

    def obs_fn():
        called["n"] += 1
        return observed(1.0)
    res = R.run_all(run, PROTO, fake, obs_fn, cap=CAP, log=lambda *_: None)
    assert not res["already_evaluated"] and called["n"] == 1
    for n in ("00_protocol", "01_threshold", "02_power", "03_pc4v", "04_outputs", "05_observed"):
        assert run.is_frozen(n)
    thr = run.read_frozen("01_threshold")
    assert thr["k"] == 476 and thr["N"] == 500
    res2 = R.run_all(run, PROTO, fake, obs_fn, cap=CAP, log=lambda *_: None)
    assert res2["already_evaluated"] and called["n"] == 1       # never evaluated twice


def test_downstream_refuses_without_upstream(tmp_path):
    run = R.D1Run(tmp_path)
    with pytest.raises(R.StageError):
        run.require("01_threshold")
    with pytest.raises(R.StageError):
        run.read_frozen("02_power")


def test_frozen_file_tampering_detected(tmp_path):
    run = R.D1Run(tmp_path)
    run.write_frozen("01_threshold", {"c": 2.0})
    p = tmp_path / "01_threshold.json"
    os.chmod(p, stat.S_IWUSR | stat.S_IRUSR)
    p.write_text(json.dumps({"c": 1.0}))
    with pytest.raises(R.StageError, match="checksum"):
        run.read_frozen("01_threshold")


def test_threshold_immutability(tmp_path):
    run = R.D1Run(tmp_path)
    run.write_frozen("01_threshold", {"c": 2.0})
    with pytest.raises(R.StageError, match="already frozen"):
        run.write_frozen("01_threshold", {"c": 1.5})
    assert not os.access(tmp_path / "01_threshold.json", os.W_OK) or os.geteuid() == 0


def test_interruption_and_restart_compute_only_missing(tmp_path):
    run = R.D1Run(tmp_path)
    CALLS["fail_after"] = 123
    with pytest.raises(RuntimeError, match="interruption"):
        R.run_all(run, PROTO, fake, lambda: observed(), cap=CAP, log=lambda *_: None)
    assert not run.is_frozen("01_threshold")
    done = len(list((tmp_path / "cohorts" / "null").glob("*.json")))
    assert done == 123
    # corrupt one result: it must be quarantined (kept) and recomputed
    bad = tmp_path / "cohorts" / "null" / "null_005.json"
    bad.write_text("{not json")
    CALLS.update(n=0, fail_after=None)
    R.run_all(run, PROTO, fake, lambda: observed(), cap=CAP, log=lambda *_: None)
    assert any(q.name.startswith("null_null_005") for q in (tmp_path / "quarantine").iterdir())
    assert run.is_frozen("05_observed")
    # 500 - 123 + 1 recomputed nulls, then 400 PC + 100 PC4v
    assert CALLS["n"] == (500 - 123 + 1) + 400 + 100


def test_protocol_drift_is_refused(tmp_path):
    run = R.D1Run(tmp_path)
    run.freeze_protocol(PROTO)
    with pytest.raises(R.StageError, match="changed"):
        run.freeze_protocol({**PROTO, "hashes": {"report": "a", "code": "CHANGED"}})


def test_outputs_tamper_then_resume(tmp_path):
    run = R.D1Run(tmp_path)

    def stop():
        raise RuntimeError("stop before observed")
    with pytest.raises(RuntimeError):
        R.run_all(run, PROTO, fake, stop, cap=CAP, log=lambda *_: None)
    # the observed stage started and failed: a marker blocks automatic re-evaluation
    with pytest.raises(R.StageError, match="started before"):
        R.run_all(run, PROTO, fake, lambda: observed(), cap=CAP, log=lambda *_: None)
    (tmp_path / "05_observed.started").unlink()
    f = tmp_path / "cohorts" / "pc" / "PC1_000.json"
    f.write_text(f.read_text().replace('"status": "complete"', '"status": "complete" '))
    with pytest.raises(R.StageError, match="changed after"):
        R.run_all(run, PROTO, fake, lambda: observed(), cap=CAP, log=lambda *_: None)


def test_pc4v_never_affects_power_or_verdict(tmp_path):
    thr = {"c": 2.0, "z_cap": CAP, "degenerate": False}
    pcs = [fake((f"{s}_{c}", [42, 41, cfg["tag"], c], {"kind": s}))
           for s, cfg in R.PC_SCENARIOS.items() for c in range(R.N_PC)]
    power = R.build_power(pcs, thr)
    v1 = R.observed_verdict(observed(1.0), thr, power)
    import inspect
    assert "pc4v" not in str(inspect.signature(R.observed_verdict)).lower()
    assert set(power["scenarios"]) == {"PC1", "PC2", "PC3", "PC4"}
    v2 = R.observed_verdict(observed(1.0), thr, R.build_power(pcs, thr))
    assert v1 == v2


def test_degenerate_observed_handling():
    power = {"primary_x": 80, "primary_P_lo": 0.71}
    ok = {"c": 2.0, "z_cap": CAP, "degenerate": False}
    v = R.observed_verdict(observed(1.0, undefined=True), ok, power)
    assert v["verdict"].startswith("DEGENERATE") and v["flags"]["undefined_observed_statistics"]
    v = R.observed_verdict(observed(CAP, at_bound=2), ok, power)
    assert v["verdict"] == "NOT REPRODUCED" and v["flags"]["T_obs_at_cap"]
    assert v["flags"]["observed_p_at_bound"] == 2                  # reported, never silent
    deg = {"c": CAP, "z_cap": CAP, "degenerate": True}
    v = R.observed_verdict(observed(CAP), deg, power)
    assert v["verdict"].startswith("CALIBRATION DEGENERATE")


def test_threshold_and_power_validation():
    with pytest.raises(R.StageError):
        R.build_threshold([{"T": 1.0}] * 499, CAP, "x")
    with pytest.raises(R.StageError):
        R.build_threshold([{"T": float("nan")}] + [{"T": 1.0}] * 499, CAP, "x")
    thr = R.build_threshold([{"T": float(t)} for t in range(1, 501)], CAP, "x")
    assert thr["c"] == 476.0
