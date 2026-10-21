"""Staged, lock-enforced execution of the D1 diagnostic (protocol B.10-B.14).

Stage order (each refuses to run unless every upstream stage is locked and its
frozen file still matches the checksum recorded in its lock):

    protocol -> null (500) -> threshold -> pc (PC1-PC4, 100 each) -> power
             -> pc4v (100, descriptive) -> freeze (output checksums)
             -> observed (exactly once)

Per-cohort results are written atomically (tmp + rename) so an interrupted
stage restarts where it stopped; unreadable or inconsistent result files are
moved to `quarantine/` (preserved, never deleted) and recomputed. Frozen files
are written once, made read-only, and can never be rewritten by the runner.
The verdict uses only the frozen threshold, the frozen PC4 power and the
observed cohort; PC4v is never read by the verdict.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import ppc_diagnostic as D

N_NULL = 500
N_PC = 100
PC_SCENARIOS = {
    "PC1": {"pi": 0.30, "L": 1.02, "D": 2.0, "kappa": 1.0, "replace_level": True, "tag": 1},
    "PC2": {"pi": 0.15, "L": 1.02, "D": 2.0, "kappa": 1.0, "replace_level": True, "tag": 2},
    "PC3": {"pi": 0.30, "L": 1.02, "D": 1.0, "kappa": 1.0, "replace_level": True, "tag": 3},
    "PC4": {"pi": 0.30, "L": 1.02, "D": 2.0, "kappa": 0.25, "replace_level": True, "tag": 4},
}
PC4V = {"pi": 0.30, "L": 1.02, "D": 2.0, "kappa": 0.25, "replace_level": False, "tag": 4}
PRIMARY = "PC4"
OBSERVED_ENTROPY = [42, 41, 9, 0]


class StageError(RuntimeError):
    """A prerequisite is missing, invalid or inconsistent: the stage must not run."""


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p) -> str:
    return sha_bytes(Path(p).read_bytes())


class D1Run:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "cohorts").mkdir(exist_ok=True)
        (self.root / "quarantine").mkdir(exist_ok=True)

    # ---- frozen files ---------------------------------------------------------
    def _path(self, name):
        return self.root / f"{name}.json"

    def _lock(self, name):
        return self.root / f"{name}.lock"

    def is_frozen(self, name) -> bool:
        return self._lock(name).exists()

    def write_frozen(self, name, obj) -> str:
        """Write once, read-only, with a lock carrying its checksum. Never overwrites."""
        if self._path(name).exists() or self._lock(name).exists():
            raise StageError(f"{name} is already frozen; refusing to rewrite it")
        data = json.dumps(obj, indent=2, default=float, sort_keys=True).encode()
        tmp = self._path(name).with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, self._path(name))
        os.chmod(self._path(name), stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        digest = sha_bytes(data)
        self._lock(name).write_text(json.dumps({"file": f"{name}.json", "sha256": digest,
                                                "time": time.strftime("%Y-%m-%dT%H:%M:%S")}))
        os.chmod(self._lock(name), stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        return digest

    def read_frozen(self, name) -> dict:
        """Read a frozen file, verifying it against its lock."""
        if not self._lock(name).exists() or not self._path(name).exists():
            raise StageError(f"prerequisite {name} is not frozen")
        lock = json.loads(self._lock(name).read_text())
        if sha_file(self._path(name)) != lock["sha256"]:
            raise StageError(f"{name}.json does not match its lock checksum")
        return json.loads(self._path(name).read_text())

    def require(self, *names):
        return [self.read_frozen(n) for n in names]

    # ---- protocol ---------------------------------------------------------------
    def freeze_protocol(self, protocol: dict):
        """First call freezes; later calls must present identical hashes (no code drift)."""
        if self.is_frozen("00_protocol"):
            frozen = self.read_frozen("00_protocol")
            if frozen["hashes"] != protocol["hashes"]:
                diff = {k for k in set(frozen["hashes"]) | set(protocol["hashes"])
                        if frozen["hashes"].get(k) != protocol["hashes"].get(k)}
                raise StageError(f"protocol/code hashes changed since freezing: {sorted(diff)}")
            return frozen
        self.write_frozen("00_protocol", protocol)
        return protocol

    # ---- cohorts ------------------------------------------------------------------
    def cohort_file(self, stage, key):
        return self.root / "cohorts" / stage / f"{key}.json"

    def _valid(self, path, entropy) -> bool:
        try:
            r = json.loads(path.read_text())
        except Exception:
            return False
        return r.get("entropy") == list(entropy) and r.get("status") == "complete"

    def run_cohorts(self, stage, items, fn, jobs=1, state=None, initializer=None):
        """items: list of (key, entropy, payload). Computes only what is missing or invalid."""
        (self.root / "cohorts" / stage).mkdir(parents=True, exist_ok=True)
        todo = []
        for key, entropy, payload in items:
            p = self.cohort_file(stage, key)
            if p.exists() and not self._valid(p, entropy):
                q = self.root / "quarantine" / f"{stage}_{key}_{int(time.time() * 1e6)}.json"
                shutil.move(str(p), q)
            if not p.exists():
                todo.append((key, entropy, payload))
        if todo:
            if jobs > 1:
                with ProcessPoolExecutor(jobs, initializer=initializer, initargs=(state,)) as ex:
                    for (key, entropy, _), res in zip(todo, ex.map(fn, todo)):
                        self._write_cohort(stage, key, entropy, res)
            else:
                if initializer is not None:
                    initializer(state)
                for item in todo:
                    self._write_cohort(stage, item[0], item[1], fn(item))
        return len(todo)

    def _write_cohort(self, stage, key, entropy, res):
        res = dict(res, entropy=list(entropy), status="complete")
        p = self.cohort_file(stage, key)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, default=float))
        os.replace(tmp, p)

    def collect(self, stage, items) -> list:
        out = []
        for key, entropy, _ in items:
            p = self.cohort_file(stage, key)
            if not p.exists() or not self._valid(p, entropy):
                raise StageError(f"{stage}/{key} missing or invalid")
            out.append(json.loads(p.read_text()))
        return out


# ---------------------------------------------------------------------------
#  Stage logic (pure functions of collected results; tested with fake cohorts)
# ---------------------------------------------------------------------------
def null_items():
    return [(f"null_{c:03d}", [42, 41, 0, c], {"kind": "null"}) for c in range(N_NULL)]


def pc_items():
    return [(f"{s}_{c:03d}", [42, 41, cfg["tag"], c], {"kind": s, "episode": cfg})
            for s, cfg in PC_SCENARIOS.items() for c in range(N_PC)]


def pc4v_items():
    return [(f"PC4v_{c:03d}", [42, 41, PC4V["tag"], c], {"kind": "PC4v", "episode": PC4V})
            for c in range(N_PC)]


def build_threshold(null_results, cap, protocol_sha) -> dict:
    T = np.array([r["T"] for r in null_results], float)
    if len(T) != N_NULL:
        raise StageError(f"expected {N_NULL} null cohorts, got {len(T)}")
    if not np.all(np.isfinite(T)):
        raise StageError("non-finite null T: stage invalid, no threshold frozen")
    k = D.threshold_index(len(T))
    c = D.threshold(T)
    return {"c": c, "k": k, "N": int(len(T)), "z_cap": cap, "degenerate": bool(np.isclose(c, cap)),
            "n_null_T_at_cap": int(np.isclose(T, cap).sum()),
            "null_T_sorted_sha256": sha_bytes(np.sort(T).tobytes()),
            "null_T_quantiles": dict(zip(["0.5", "0.9", "0.95", "0.99", "max"],
                                         [float(np.quantile(T, q)) for q in (.5, .9, .95, .99)]
                                         + [float(T.max())])),
            "protocol_sha256": protocol_sha}


def build_power(pc_results, threshold) -> dict:
    c = threshold["c"]
    out = {"threshold_c": c, "scenarios": {}}
    for s in PC_SCENARIOS:
        rs = [r for r in pc_results if r["kind"] == s]
        T = np.array([r["T"] for r in rs], float)
        if len(T) != N_PC or not np.all(np.isfinite(T)):
            raise StageError(f"{s}: expected {N_PC} valid cohorts")
        x = int((T > c).sum())
        lo = D.clopper_pearson_lower(x, N_PC)
        hi = 1.0 if x == N_PC else float(__import__("scipy").stats.beta.ppf(0.975, x + 1, N_PC - x))
        zs = {k: float(np.mean([r["z"][k] for r in rs])) for k in D.STAT_NAMES}
        out["scenarios"][s] = {"detected": x, "n": N_PC, "rate": x / N_PC, "cp95": [lo, hi],
                               "mean_signed_z": zs}
    out["primary"] = PRIMARY
    out["primary_x"] = out["scenarios"][PRIMARY]["detected"]
    out["primary_P_lo"] = out["scenarios"][PRIMARY]["cp95"][0]
    return out


def build_pc4v(pc4v_results, threshold) -> dict:
    """Descriptive only: never read by the verdict."""
    T = np.array([r["T"] for r in pc4v_results], float)
    x = int((T > threshold["c"]).sum())
    from scipy import stats
    return {"descriptive_only": True, "detected": x, "n": len(T), "rate": x / len(T),
            "cp95": [D.clopper_pearson_lower(x, len(T)),
                     1.0 if x == len(T) else float(stats.beta.ppf(0.975, x + 1, len(T) - x))],
            "mean_signed_z": {k: float(np.mean([r["z"][k] for r in pc4v_results]))
                              for k in D.STAT_NAMES}}


def observed_verdict(obs: dict, threshold: dict, power: dict) -> dict:
    """Frozen rule B.11.5 with the B.14 handling of degenerate observed evaluations.

    Takes ONLY the frozen threshold, the frozen power (PC4) and the observed result.
    """
    cap = threshold["z_cap"]
    flags = {"T_obs_at_cap": bool(np.isclose(obs["T"], cap)),
             "observed_p_at_bound": int(obs["p_at_bound"]),
             "undefined_observed_statistics": [k for k, v in obs["S_obs"].items()
                                               if k in D.STAT_NAMES and not np.isfinite(v)]}
    if flags["undefined_observed_statistics"]:
        v = "DEGENERATE (undefined observed statistic): verdict withheld"
    else:                        # D.verdict also handles a threshold at the z cap (B.11.2)
        v = D.verdict(obs["T"], threshold["c"], power["primary_x"], N_PC, cap)
    return {"verdict": v, "flags": flags, "T_obs": obs["T"], "c": threshold["c"],
            "primary_x": power["primary_x"], "primary_P_lo": power["primary_P_lo"]}


def run_all(run: D1Run, protocol: dict, cohort_fn, observed_fn, jobs=1, state=None,
            initializer=None, cap=None, log=print) -> dict:
    """Run every stage in the frozen order, skipping stages already frozen.

    cohort_fn((key, entropy, payload)) -> result dict with T, z, S_obs, p_at_bound.
    observed_fn() -> the observed cohort's result (called at most once, ever).
    """
    prot = run.freeze_protocol(protocol)
    prot_sha = json.loads(run._lock("00_protocol").read_text())["sha256"]
    status = {}
    # 1-2: null cohorts, then the threshold (immutable)
    if not run.is_frozen("01_threshold"):
        n = run.run_cohorts("null", null_items(), cohort_fn, jobs, state, initializer)
        log(f"null cohorts computed this session: {n}")
        thr = build_threshold(run.collect("null", null_items()), cap, prot_sha)
        run.write_frozen("01_threshold", thr)
    thr = run.read_frozen("01_threshold")
    status["threshold"] = thr
    # 3: positive controls with the frozen threshold
    if not run.is_frozen("02_power"):
        n = run.run_cohorts("pc", pc_items(), cohort_fn, jobs, state, initializer)
        log(f"positive-control cohorts computed this session: {n}")
        run.write_frozen("02_power", build_power(run.collect("pc", pc_items()), thr))
    power = run.read_frozen("02_power")
    # 4: PC4v, descriptive only (requires the power file so it cannot precede it)
    if not run.is_frozen("03_pc4v"):
        run.require("01_threshold", "02_power")
        n = run.run_cohorts("pc4v", pc4v_items(), cohort_fn, jobs, state, initializer)
        log(f"PC4v cohorts computed this session: {n}")
        run.write_frozen("03_pc4v", build_pc4v(run.collect("pc4v", pc4v_items()), thr))
    # 5: freeze all outputs
    if not run.is_frozen("04_outputs"):
        run.require("00_protocol", "01_threshold", "02_power", "03_pc4v")
        files = sorted(p for p in (run.root / "cohorts").rglob("*.json"))
        run.write_frozen("04_outputs", {
            "n_files": len(files),
            "sha256": {str(p.relative_to(run.root)): sha_file(p) for p in files},
            "frozen_files": {n: json.loads(run._lock(n).read_text())["sha256"]
                             for n in ("00_protocol", "01_threshold", "02_power", "03_pc4v")}})
    outputs = run.read_frozen("04_outputs")
    for rel, digest in outputs["sha256"].items():          # outputs unchanged since freezing
        if sha_file(run.root / rel) != digest:
            raise StageError(f"{rel} changed after the outputs were frozen")
    # 6: observed, exactly once
    if run.is_frozen("05_observed"):
        return {"already_evaluated": True, "observed": run.read_frozen("05_observed")}
    marker = run.root / "05_observed.started"
    if marker.exists():
        raise StageError("observed evaluation was started before and did not complete; "
                         "it will not be re-run automatically (needs a reviewer decision)")
    run.require("00_protocol", "01_threshold", "02_power", "03_pc4v", "04_outputs")
    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    obs = observed_fn()
    result = {"observed": {k: obs[k] for k in ("S_obs", "S_rep_mean", "p", "z", "T",
                                               "p_at_bound", "T_at_cap", "folds", "timing")},
              **observed_verdict(obs, thr, power)}
    run.write_frozen("05_observed", result)
    return {"already_evaluated": False, "observed": result}
