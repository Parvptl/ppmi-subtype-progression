"""D1 observed-cohort replicate labels: recorder, exact-match gate, descriptive splits.

Specification and approval: docs/cluster_phase/15_D1_REPLICATE_LABEL_DIAGNOSTIC_REVIEW.md
(approved 2026-10-16 as a documented protocol deviation). Nothing here edits or
replaces any D1 code: `LabelRecorder` wraps `ppc_diagnostic.labels_from_ratio`
for the duration of one `run_cohort` call, keeping a copy of the replicate label
array that `run_cohort` passes to `label_statistics`. The wrapper calls the
original, draws no random numbers and changes no return value.

Every split computed here is exploratory and uncalibrated: no null calibration,
no threshold, no verdict.
"""

from __future__ import annotations

import math

import numpy as np

from . import ppc_diagnostic as D
from . import stage1_checks as S

GATED_KEYS = ("S_obs", "S_rep_mean", "p", "z", "T", "T_at_cap", "p_at_bound", "folds")
GATED_KEYS_NULL = ("T", "z", "p", "S_obs", "p_at_bound", "T_at_cap", "n_rep_undefined", "folds")


class LabelRecorder:
    """Context manager: record every 2-D output of `ppc_diagnostic.labels_from_ratio`."""

    def __init__(self):
        self.records: list[np.ndarray] = []
        self._orig = None

    def __enter__(self):
        self._orig = D.labels_from_ratio
        orig = self._orig

        def recording(y):
            out = orig(y)
            if out.ndim == 2:
                self.records.append(out.copy())
            return out
        D.labels_from_ratio = recording
        return self

    def __exit__(self, *exc):
        D.labels_from_ratio = self._orig
        return False

    def single(self) -> np.ndarray:
        if len(self.records) != 1:
            raise RuntimeError(f"expected one replicate label array, recorded {len(self.records)}")
        return self.records[0]


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        if math.isnan(a) and math.isnan(b):
            return True
        return float(a) == float(b)
    return a == b


def exact_mismatches(reproduced, frozen, path: str = "") -> list[str]:
    """Paths at which two JSON-like values differ; floats must be equal exactly (NaN = NaN)."""
    out = []
    if isinstance(frozen, dict):
        if not isinstance(reproduced, dict) or set(reproduced) != set(frozen):
            return [f"{path}: keys {sorted(reproduced) if isinstance(reproduced, dict) else type(reproduced)}"
                    f" != {sorted(frozen)}"]
        for k in frozen:
            out += exact_mismatches(reproduced[k], frozen[k], f"{path}.{k}")
    elif isinstance(frozen, list):
        if not isinstance(reproduced, list) or len(reproduced) != len(frozen):
            return [f"{path}: length differs"]
        for i, (r, f) in enumerate(zip(reproduced, frozen)):
            out += exact_mismatches(r, f, f"{path}[{i}]")
    elif not _same(reproduced, frozen):
        out.append(f"{path}: {reproduced!r} != {frozen!r}")
    return out


def gate(reproduced: dict, frozen: dict, keys) -> dict:
    """Exact-match gate over `keys` (JSON round-trip of the reproduced values first)."""
    import json
    rep = json.loads(json.dumps({k: reproduced.get(k) for k in keys}))
    fro = {k: frozen.get(k) for k in keys}
    missing = [k for k in keys if k not in reproduced or k not in frozen]
    mism = [f"missing key {k}" for k in missing] + exact_mismatches(rep, fro)
    return {"passed": not mism, "keys": list(keys), "mismatches": mism}


# ---------------------------------------------------------------------------
#  Descriptive splits (pre-specified in doc 13 section 5, D-a and D-b)
# ---------------------------------------------------------------------------
def stratum_shares(L: np.ndarray, strata: np.ndarray, n_strata: int) -> np.ndarray:
    """Indeterminate share per stratum for each row of L: (R, n_strata)."""
    L = np.atleast_2d(L)
    out = np.empty((L.shape[0], n_strata))
    for k in range(n_strata):
        sel = strata == k
        out[:, k] = (L[:, sel] == S.IND).mean(axis=1) if sel.any() else np.nan
    return out


def direction_outcome_counts(L: np.ndarray, pid: np.ndarray) -> np.ndarray:
    """(R, 2, 4) counts: direction (TD->PIGD, PIGD->TD) x outcome (revert, stay, to_Ind, unobserved)."""
    L = np.atleast_2d(L)
    out = np.zeros((L.shape[0], 2, 4), int)
    for r in range(L.shape[0]):
        _, d, o = S.direct_change_outcomes(L[r], pid)
        np.add.at(out[r], (d, o), 1)
    return out


def reversion_rates(counts: np.ndarray) -> np.ndarray:
    """R_D per direction from (..., 2, 4) counts: revert / (revert + stay + to_Ind)."""
    den = counts[..., :3].sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, counts[..., 0] / den, np.nan)


def describe(obs: float, rep: np.ndarray) -> dict:
    """Observed vs replicates: mean, 2.5-97.5 % range, mid-p (D1 convention) and z.

    Exploratory and uncalibrated; z < 0 means the observed value is above the replicates.
    """
    rep = np.asarray(rep, float)
    fin = rep[np.isfinite(rep)]
    p, n = D.midp(obs, rep)
    from scipy import stats as sstats
    return {"observed": float(obs), "rep_mean": float(fin.mean()) if fin.size else np.nan,
            "rep_q025": float(np.quantile(fin, 0.025)) if fin.size else np.nan,
            "rep_q975": float(np.quantile(fin, 0.975)) if fin.size else np.nan,
            "n_rep_defined": int(n), "midp": float(p),
            "z": float(sstats.norm.ppf(p)) if np.isfinite(p) else np.nan,
            "status": "exploratory, uncalibrated"}
