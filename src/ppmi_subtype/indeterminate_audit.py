"""Indeterminate-class audit: events, reversion, temporal clustering.

"Temporal clustering" here means Indeterminate visits falling next to each
other in one patient's sequence more often than chance would place them.

Sequence. A patient's LABELLED visits ordered by time (ties by nominal YEAR);
unlabelled visits are removed first, so "adjacent" means consecutive labelled
visits (as in `transitions.consecutive_pairs`). An optional segment id splits
a sequence (sensitivity analyses); adjacency, events and outcomes never cross a
segment boundary.

Events on a pair (k, k+1) of adjacent visits in the same segment:
    entry   label_k in {TD, PIGD}, label_{k+1} = Indeterminate   (origin = label_k)
    direct  {label_k, label_{k+1}} = {TD, PIGD}
    exit    label_k = Indeterminate, label_{k+1} in {TD, PIGD}
The arrival visit is j = k+1 with arrival label X. Outcome at the next
observed labelled visit j+1:
    sustained   label_{j+1} = X
    reverted    label_{j+1} = origin
    moved_on    label_{j+1} = the third label
    unobserved  no labelled visit after j
    outside     a later visit exists but in another segment (sensitivity only)
Unobserved and outside events are counted but never enter a reversion or
persistence denominator.

Reversion rate of a group = reverted / (sustained + reverted + moved_on).

Temporal clustering. For patient i with n_i >= 2 labelled visits of which m_i
carry label L (per segment s when segmented):
    O_i = #{k : label_k = label_{k+1} = L, same segment}
    E_i = sum_s m_s (m_s - 1) / n_s      (segments with n_s >= 2)
E_i is the exact expectation of O_i when the patient's labels are randomly
reordered over their own visit slots (each of the n_s - 1 adjacent slot pairs
is L-L with probability m_s (m_s - 1) / (n_s (n_s - 1))). Patients with
m_i <= 1 contribute O = E = 0. O/E = sum O_i / sum E_i.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .raw_scores import patient_bootstrap_sums

TD, IND, PIGD = "TD", "Indeterminate", "PIGD"
LABELS = (TD, IND, PIGD)
E_MIN = 20.0
COUNT_KEYS = (
    "n_visits", "eligible",
    "entry_n", "entry_obs", "entry_rev", "entry_sus", "entry_mov", "entry_unobs", "entry_out",
    "entry_obs2", "entry_sus2",
    "direct_n", "direct_obs", "direct_rev", "direct_sus", "direct_mov", "direct_unobs",
    "direct_out",
    "exit_n", "exit_obs", "exit_back", "exit_unobs", "exit_out",
    "O_Indeterminate", "E_Indeterminate", "O_TD", "E_TD", "O_PIGD", "E_PIGD",
    "contrib_Indeterminate",
)


# ---------------------------------------------------------------------------
def ordered(df: pd.DataFrame, label_col: str, time_col: str = "years_from_baseline"):
    """Labelled visits sorted by patient, time, nominal year (stable)."""
    d = df[df[label_col].notna()].copy()
    keys = ["PATNO", time_col] + (["YEAR"] if "YEAR" in d.columns else [])
    return d.sort_values(keys, kind="mergesort")


def segments(times, treat=None, max_interval: float | None = None,
             treatment_stable: bool = False) -> np.ndarray:
    """Segment id per visit: a new segment starts after a gap > max_interval,
    or (treatment_stable) when treatment status changes or is missing at either visit."""
    n = len(times)
    seg = np.zeros(n, int)
    for k in range(1, n):
        brk = False
        if max_interval is not None and times[k] - times[k - 1] > max_interval:
            brk = True
        if treatment_stable:
            a, b = treat[k - 1], treat[k]
            if not (np.isfinite(a) and np.isfinite(b) and a == b):
                brk = True
        seg[k] = seg[k - 1] + (1 if brk else 0)
    return seg


def expected_adjacent(labels, seg, L) -> float:
    """Exact expected number of adjacent L-L pairs under within-segment reordering."""
    labels, seg = np.asarray(labels), np.asarray(seg)
    e = 0.0
    for s in np.unique(seg):
        lab = labels[seg == s]
        n, m = len(lab), int((lab == L).sum())
        if n >= 2:
            e += m * (m - 1) / n
    return e


def observed_adjacent(labels, seg, L) -> int:
    labels, seg = np.asarray(labels), np.asarray(seg)
    same = seg[:-1] == seg[1:]
    return int(np.sum(same & (labels[:-1] == L) & (labels[1:] == L)))


def _outcome(labels, seg, j, X, origin):
    n = len(labels)
    if j + 1 >= n:
        return "unobserved"
    if seg[j + 1] != seg[j]:
        return "outside"
    nxt = labels[j + 1]
    if nxt == X:
        return "sustained"
    if nxt == origin:
        return "reverted"
    return "moved_on"


def patient_events(labels, seg) -> list[dict]:
    """All entry/direct/exit events of one patient's sequence, with outcomes."""
    labels, seg = np.asarray(labels, dtype=object), np.asarray(seg)
    out = []
    for k in range(len(labels) - 1):
        if seg[k] != seg[k + 1]:
            continue
        a, b = labels[k], labels[k + 1]
        if a in (TD, PIGD) and b == IND:
            kind = "entry"
        elif {a, b} == {TD, PIGD}:
            kind = "direct"
        elif a == IND and b in (TD, PIGD):
            kind = "exit"
        else:
            continue
        j = k + 1
        oc = _outcome(labels, seg, j, b, a)
        two = None
        if oc in ("sustained", "reverted", "moved_on") and j + 2 < len(labels) \
                and seg[j + 2] == seg[j]:
            two = bool(labels[j + 1] == b and labels[j + 2] == b)
        out.append({"k": k, "j": j, "kind": kind, "origin": a, "arrival": b,
                    "outcome": oc, "sustained2": two})
    return out


def patient_counts(labels, seg) -> dict:
    labels = np.asarray(labels, dtype=object)
    c = dict.fromkeys(COUNT_KEYS, 0.0)
    c["n_visits"] = len(labels)
    c["eligible"] = float(len(labels) >= 2)
    for ev in patient_events(labels, seg):
        p = ev["kind"]
        c[f"{p}_n"] += 1
        oc = ev["outcome"]
        if p == "exit":
            if oc == "unobserved":
                c["exit_unobs"] += 1
            elif oc == "outside":
                c["exit_out"] += 1
            else:
                c["exit_obs"] += 1
                c["exit_back"] += float(labels[ev["j"] + 1] == IND)
            continue
        if oc == "unobserved":
            c[f"{p}_unobs"] += 1
        elif oc == "outside":
            c[f"{p}_out"] += 1
        else:
            c[f"{p}_obs"] += 1
            c[f"{p}_" + {"reverted": "rev", "sustained": "sus", "moved_on": "mov"}[oc]] += 1
            if p == "entry" and ev["sustained2"] is not None:
                c["entry_obs2"] += 1
                c["entry_sus2"] += float(ev["sustained2"])
    if len(labels) >= 2:
        for L in LABELS:
            c[f"O_{L}"] = observed_adjacent(labels, seg, L)
            c[f"E_{L}"] = expected_adjacent(labels, seg, L)
        c["contrib_Indeterminate"] = float(any(
            int((labels[np.asarray(seg) == s] == IND).sum()) >= 2 for s in np.unique(seg)))
    return c


def count_matrix(df: pd.DataFrame, label_col: str, segment_kw: dict | None = None,
                 treat_col: str = "PDTRTMNT", time_col: str = "years_from_baseline"):
    """Per-patient count matrix (patients x COUNT_KEYS) and the patient order."""
    d = ordered(df, label_col, time_col)
    rows, pats = [], []
    for patno, g in d.groupby("PATNO", sort=True):
        labels = g[label_col].to_numpy(dtype=object)
        if segment_kw:
            treat = pd.to_numeric(g[treat_col], errors="coerce").to_numpy() \
                if treat_col in g else None
            seg = segments(g[time_col].to_numpy(float), treat, **segment_kw)
        else:
            seg = np.zeros(len(labels), int)
        c = patient_counts(labels, seg)
        rows.append([c[k] for k in COUNT_KEYS])
        pats.append(patno)
    return np.array(rows, float), np.array(pats)


def statistics(tot: np.ndarray) -> dict:
    """C2 statistics from summed counts (vector aligned with COUNT_KEYS)."""
    t = dict(zip(COUNT_KEYS, tot))
    with np.errstate(invalid="ignore", divide="ignore"):
        R_E = t["entry_rev"] / t["entry_obs"] if t["entry_obs"] else np.nan
        R_D = t["direct_rev"] / t["direct_obs"] if t["direct_obs"] else np.nan
        out = {"R_E": R_E, "R_D": R_D, "delta_R": R_E - R_D,
               "entry_sustained": t["entry_sus"] / t["entry_obs"] if t["entry_obs"] else np.nan,
               "entry_moved_on": t["entry_mov"] / t["entry_obs"] if t["entry_obs"] else np.nan,
               "entry_sustained2": t["entry_sus2"] / t["entry_obs2"] if t["entry_obs2"] else np.nan,
               "direct_sustained": t["direct_sus"] / t["direct_obs"] if t["direct_obs"] else np.nan,
               "exit_back_to_Ind": t["exit_back"] / t["exit_obs"] if t["exit_obs"] else np.nan}
        for L in LABELS:
            out[f"OE_{L}"] = t[f"O_{L}"] / t[f"E_{L}"] if t[f"E_{L}"] > 0 else np.nan
    return out


def bootstrap(C: np.ndarray, B: int = 4000, seed: int = 42):
    """Point statistics, percentile 95% intervals and undefined-replicate counts."""
    point = statistics(C.sum(axis=0))
    sums = patient_bootstrap_sums(C, B, seed)
    reps = pd.DataFrame([statistics(s) for s in sums])
    ci = {k: (float(np.nanquantile(reps[k], 0.025)), float(np.nanquantile(reps[k], 0.975)))
          if reps[k].notna().any() else (np.nan, np.nan) for k in reps.columns}
    undefined = {k: int(reps[k].isna().sum()) for k in reps.columns}
    return point, ci, undefined


def reading(delta_R_ci, oe_ci, E_total: float, e_min: float = E_MIN) -> str:
    """Amended reading (plan Amendment 1, 2026-10-16): each contrast must establish
    its direction; an interval spanning 0 or 1 is never read as equivalence."""
    if not np.isfinite(E_total) or E_total < e_min:
        return "inconclusive (clustering not evaluable: E < %g)" % e_min
    dlo, dhi = delta_R_ci
    olo, ohi = oe_ci
    if dlo > 0 and ohi < 1:
        return "consistent with transient boundary-crossing"
    if dhi <= 0 and olo > 1:
        return "consistent with sustained intermediate passage"
    return "inconclusive"


def reading_original(delta_R_ci, oe_ci, E_total: float, e_min: float = E_MIN) -> str:
    """SUPERSEDED rule of plan revision 2, section 5.5 -- kept for the record only."""
    if not np.isfinite(E_total) or E_total < e_min:
        return "inconclusive (clustering not evaluable: E < %g)" % e_min
    dlo, _ = delta_R_ci
    olo, _ = oe_ci
    if dlo > 0 and olo <= 1:
        return "consistent with transient boundary-crossing"
    if olo > 1 and dlo <= 0:
        return "consistent with sustained intermediate passage"
    return "inconclusive"


def wilson(k: float, n: float, z: float = 1.959964) -> tuple[float, float]:
    """Pair-level Wilson interval (continuity with script 02 only)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def event_table(df: pd.DataFrame, label_col: str, time_col: str = "years_from_baseline"):
    """One row per event (primary segmentation), with the row index of k, j, j+1."""
    d = ordered(df, label_col, time_col)
    rows = []
    for patno, g in d.groupby("PATNO", sort=True):
        labels = g[label_col].to_numpy(dtype=object)
        idx = g.index.to_numpy()
        for ev in patient_events(labels, np.zeros(len(labels), int)):
            nxt = idx[ev["j"] + 1] if ev["j"] + 1 < len(idx) else -1
            rows.append({"PATNO": patno, "row_k": idx[ev["k"]], "row_j": idx[ev["j"]],
                         "row_next": nxt, **{k: ev[k] for k in ("kind", "origin", "arrival",
                                                              "outcome", "sustained2")}})
    return pd.DataFrame(rows)


def exit_episodes(df: pd.DataFrame, label_col: str, state: str,
                  time_col: str = "years_from_baseline") -> pd.DataFrame:
    """Time from a patient's first visit in `state` to the first later visit with
    another label (event) or to the last labelled visit (censored). Patients whose
    first `state` visit is their last labelled visit have no follow-up and are
    returned with `followed = False` (not used in the estimate)."""
    d = ordered(df, label_col, time_col)
    rows = []
    for patno, g in d.groupby("PATNO", sort=True):
        lab = g[label_col].to_numpy(dtype=object)
        t = g[time_col].to_numpy(float)
        hit = np.flatnonzero(lab == state)
        if len(hit) == 0:
            continue
        f = hit[0]
        if f == len(lab) - 1:
            rows.append({"PATNO": patno, "followed": False, "time": 0.0, "event": False})
            continue
        later = np.flatnonzero(lab[f + 1:] != state)
        if len(later):
            rows.append({"PATNO": patno, "followed": True,
                         "time": t[f + 1 + later[0]] - t[f], "event": True})
        else:
            rows.append({"PATNO": patno, "followed": True, "time": t[-1] - t[f], "event": False})
    return pd.DataFrame(rows)
