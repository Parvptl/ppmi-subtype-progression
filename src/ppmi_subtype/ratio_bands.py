"""Joint (subtype, ratio-band) observation categories for the CT-HMM.

Why this exists
---------------
The three recorded subtypes are a threshold cut of one continuous quantity,
``ratio = tremor_score / pigd_score`` (see :mod:`ppmi_subtype.subtype`). Two
visits labelled TD can sit at ratio 1.20 and 4.50, and two Indeterminate
visits can sit either side of the band's middle; the label discards all of
that. This module builds a finer observation alphabet --- the subtype *and*
the ratio band it came from --- so the CT-HMM's emission can be asked whether
the extra resolution says anything about the latent state.

What it does not do
-------------------
It does not replace the subtype. Every joint category carries its subtype, the
categories belonging to one subtype form a contiguous block, and
:func:`aggregate_to_subtype` sums a distribution over joint categories back to
the three-label distribution the evaluation is defined on. A joint model and
the label model therefore predict the same quantity.

Band structure
--------------
Left-closed, right-open throughout: a ratio ``r`` falls in ``[lo, hi)``. A
value sitting exactly on an edge belongs to the band that edge *opens*, so no
observation can belong to two bands.

    main range   [0.0, 3.0) in fixed 0.1-width bands --- 30 of them
    tail         [3.0, inf) in 0.25-width candidate bands, merged upward
                 until each carries enough training data (see `merge_tail`)
    undefined    ratio is not computable (``pigd_score == 0``), one category
                 per subtype that can produce it

The undefined category is not a convenience: 692 of the cohort's 4,295
labelled visits have ``pigd_score == 0`` and no ratio at all. Dropping them
would change the evaluation cohort, so they are an observation value like any
other --- one that happens to say "no gait or postural impairment was scored".

Leakage
-------
The main edges and the candidate tail grid are fixed in advance. The only
data-dependent part is where the tail bands are merged, and :func:`merge_tail`
takes the ratios it is given --- the LOPO driver passes training observations
only. The topmost tail band is closed as ``[lo, inf)`` so a held-out ratio
larger than anything in training is still assignable without consulting it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Fixed main-range band structure, as specified for this experiment.
MAIN_LOW = 0.0
MAIN_HIGH = 3.0
MAIN_WIDTH = 0.1

#: Candidate tail grid. Wider than the main bands because the tail is thinner;
#: adjacent candidates are merged by `merge_tail` until each carries enough
#: training data to estimate an emission probability from.
TAIL_START = MAIN_HIGH
TAIL_WIDTH = 0.25

#: The pre-registered minimum size of a tail band, fixed before any model was
#: fitted and never varied against held-out performance. Each tail band costs
#: one free emission parameter per hidden state, so with three hidden states a
#: band needs ~3 x 5 = 15 observations to reach the conventional five-per-cell
#: minimum for a multinomial cell. 30 is the smallest round number that clears
#: it with a factor-of-two margin. The patient minimum is there because this is
#: a longitudinal panel: a band can hold many observations from very few
#: people, and repeat visits of one patient are not independent evidence.
MIN_TAIL_OBSERVATIONS = 30
MIN_TAIL_PATIENTS = 20

#: Band label for a visit whose ratio is not computable.
UNDEFINED_BAND = "undefined"


def main_edges() -> np.ndarray:
    """The 31 fixed edges 0.0, 0.1, ... 3.0 bounding the 30 main bands."""
    n = int(round((MAIN_HIGH - MAIN_LOW) / MAIN_WIDTH))
    return MAIN_LOW + np.arange(n + 1) / (1.0 / MAIN_WIDTH)


def candidate_tail_edges(max_ratio: float) -> np.ndarray:
    """Lower edges of the 0.25-width candidate tail bands covering `max_ratio`.

    Always at least one band. The last band is closed as ``[lo, inf)`` by the
    caller, so a ratio above `max_ratio` is still assignable.
    """
    if not np.isfinite(max_ratio) or max_ratio < TAIL_START:
        return np.array([TAIL_START])
    n = int(np.floor((max_ratio - TAIL_START) / TAIL_WIDTH)) + 1
    return TAIL_START + TAIL_WIDTH * np.arange(n)


# ---------------------------------------------------------------------------
#  Dynamic tail merging --- training data only
# ---------------------------------------------------------------------------
def merge_tail(
    ratios: np.ndarray,
    patients: np.ndarray,
    min_observations: int = MIN_TAIL_OBSERVATIONS,
    min_patients: int = MIN_TAIL_PATIENTS,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Merge the candidate tail bands upward until each is large enough.

    The rule, fixed in advance and deterministic:

    1. Lay the 0.25-width candidate grid over ``[3.0, max(ratio)]``.
    2. Sweep upward from 3.0 accumulating consecutive candidates into the
       current band. Close the band as soon as it holds at least
       `min_observations` observations from at least `min_patients` distinct
       patients, and start a new one above it.
    3. If the sweep ends with an accumulated band that is still too small,
       merge it into the band below. If there is no band below, the whole tail
       is one band.

    Returns ``(lower_edges, diagnostics)``. ``lower_edges`` are the retained
    band lower edges, ascending, starting at 3.0; the last band runs to
    infinity. The diagnostics frame has one row per candidate band, with the
    merged band it ended up in, so the before/after picture is recoverable.

    `ratios` and `patients` must be the *training* observations. Nothing else
    in this module is data-dependent.
    """
    ratios = np.asarray(ratios, dtype=float)
    patients = np.asarray(patients, dtype=object)
    if len(ratios) != len(patients):
        raise ValueError("ratios and patients must be the same length")

    in_tail = np.isfinite(ratios) & (ratios >= TAIL_START)
    tail_r, tail_p = ratios[in_tail], patients[in_tail]
    grid = candidate_tail_edges(float(tail_r.max()) if len(tail_r) else float("nan"))

    # Candidate band index of each tail observation.
    which = (np.floor((tail_r - TAIL_START) / TAIL_WIDTH).astype(int)
             if len(tail_r) else np.zeros(0, dtype=int))
    which = np.clip(which, 0, len(grid) - 1)

    retained: list[float] = [float(grid[0])]
    assigned = np.zeros(len(grid), dtype=int)       # merged-band index per candidate
    open_obs, open_pat = 0, set()
    closed_any = False

    for k in range(len(grid)):
        assigned[k] = len(retained) - 1
        here = which == k
        open_obs += int(here.sum())
        open_pat.update(tail_p[here].tolist())

        big_enough = open_obs >= min_observations and len(open_pat) >= min_patients
        if big_enough and k < len(grid) - 1:
            retained.append(float(grid[k + 1]))
            open_obs, open_pat = 0, set()
            closed_any = True

    # Step 3: a final band that never reached the threshold is absorbed below.
    final_short = not (open_obs >= min_observations and len(open_pat) >= min_patients)
    if final_short and closed_any and len(retained) > 1:
        dropped = retained.pop()
        assigned[assigned >= len(retained)] = len(retained) - 1
    else:
        dropped = None

    rows = []
    for k, lo in enumerate(grid):
        here = which == k
        rows.append({
            "candidate_low": float(lo),
            "candidate_high": float(lo + TAIL_WIDTH),
            "n_observations": int(here.sum()),
            "n_patients": int(len(set(tail_p[here].tolist()))),
            "merged_band_index": int(assigned[k]),
            "merged_low": float(retained[assigned[k]]),
            "merged_high": (float(retained[assigned[k] + 1])
                            if assigned[k] + 1 < len(retained) else float("inf")),
        })
    diagnostics = pd.DataFrame(rows)
    diagnostics.attrs["dropped_final_edge"] = dropped
    return np.array(retained, dtype=float), diagnostics


# ---------------------------------------------------------------------------
#  Band scheme
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BandScheme:
    """Band edges plus the subtype rule, which together fix the alphabet.

    `tail_lows` are the retained tail band lower edges (the first is 3.0 and
    the last band runs to infinity). `td_threshold`, `pigd_threshold` and
    `band_boundaries` are copied from the subtype config so the joint
    categories are derived from the *same* classification rule the labels come
    from, not from a second copy of it.
    """

    tail_lows: tuple[float, ...]
    td_threshold: float
    pigd_threshold: float
    band_boundaries: str = "outer"

    @property
    def edges(self) -> np.ndarray:
        """All band lower edges, ascending: 30 main plus the retained tail."""
        return np.concatenate([main_edges()[:-1], np.asarray(self.tail_lows, float)])

    @property
    def n_bands(self) -> int:
        """Number of ratio bands, the undefined category excluded."""
        return len(self.edges)

    def band_index(self, ratio) -> np.ndarray:
        """Band index per ratio; -1 where the ratio is not computable.

        Left-closed, right-open: a ratio equal to an edge opens that band.
        """
        r = np.asarray(ratio, dtype=float)
        idx = np.searchsorted(self.edges, r, side="right") - 1
        idx = np.where(np.isfinite(r), idx, -1)
        if np.any(np.isfinite(r) & (idx < 0)):
            bad = r[np.isfinite(r) & (idx < 0)]
            raise ValueError(f"ratio below the lowest band edge {self.edges[0]}: {bad[:5]}")
        return idx.astype(int)

    def band_bounds(self, k: int) -> tuple[float, float]:
        """``[lo, hi)`` of band `k`; the topmost band runs to infinity."""
        e = self.edges
        return float(e[k]), (float(e[k + 1]) if k + 1 < len(e) else float("inf"))

    def band_label(self, k: int) -> str:
        lo, hi = self.band_bounds(k)
        return f"[{lo:.2f},inf)" if np.isinf(hi) else f"[{lo:.2f},{hi:.2f})"

    def subtype_region(self, subtype_name: str) -> tuple[tuple, tuple]:
        """The subtype's ratio region as ``((lo, lo_closed), (hi, hi_closed))``.

        Mirrors :func:`ppmi_subtype.subtype.classify` exactly: under the
        published ("outer") rule TD and PIGD own their thresholds and the
        Indeterminate band is open; "inner" reverses that.
        """
        p, t = self.pigd_threshold, self.td_threshold
        outer = self.band_boundaries != "inner"
        if subtype_name == "PIGD":
            return ((-np.inf, False), (p, outer))
        if subtype_name == "TD":
            return ((t, outer), (np.inf, False))
        if subtype_name == "Indeterminate":
            return ((p, not outer), (t, not outer))
        raise ValueError(f"unknown subtype {subtype_name!r}")


def _intersects(a: tuple[tuple, tuple], b: tuple[tuple, tuple]) -> bool:
    """Do two intervals share at least one point? Endpoints carry their closure."""
    (alo, alc), (ahi, ahc) = a
    (blo, blc), (bhi, bhc) = b

    if alo > blo:
        lo, lo_closed = alo, alc
    elif blo > alo:
        lo, lo_closed = blo, blc
    else:
        lo, lo_closed = alo, (alc and blc)

    if ahi < bhi:
        hi, hi_closed = ahi, ahc
    elif bhi < ahi:
        hi, hi_closed = bhi, bhc
    else:
        hi, hi_closed = ahi, (ahc and bhc)

    if lo < hi:
        return True
    if lo == hi:
        return bool(lo_closed and hi_closed)
    return False


def band_allows_subtype(scheme: BandScheme, band: int, subtype_name: str) -> bool:
    """Can a visit in this band carry this subtype, under the threshold rule?

    Decided from the rule, not from the data: a category that is structurally
    possible but empty in one fold's training set must still exist, or a
    held-out visit could have nowhere to go.
    """
    if band < 0:                       # ratio not computable
        # pigd_score == 0 and tremor_score > 0 is TD by rule; both zero is
        # Indeterminate. A PIGD label requires a positive PIGD subscore.
        return subtype_name in ("TD", "Indeterminate")
    lo, hi = scheme.band_bounds(band)
    return _intersects(((lo, True), (hi, False)), scheme.subtype_region(subtype_name))


# ---------------------------------------------------------------------------
#  The joint alphabet
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class JointAlphabet:
    """The observed categories ``(subtype, band)``, in a fixed order.

    Categories belonging to one subtype are contiguous, which is what lets the
    emission model keep a subtype-level marginal (see
    :mod:`ppmi_subtype.hmm_banded`) and lets :func:`aggregate_to_subtype` sum a
    joint distribution back to three numbers.
    """

    scheme: BandScheme
    subtype_names: tuple[str, ...]
    #: (subtype index, band index) per category, band -1 meaning "undefined".
    pairs: tuple[tuple[int, int], ...]
    #: Category indices belonging to each subtype, in order.
    blocks: tuple[tuple[int, ...], ...]

    @property
    def n_categories(self) -> int:
        return len(self.pairs)

    @property
    def subtype_of(self) -> np.ndarray:
        return np.array([s for s, _ in self.pairs], dtype=int)

    @property
    def labels(self) -> list[str]:
        return [
            f"{self.subtype_names[s]}|"
            f"{UNDEFINED_BAND if b < 0 else self.scheme.band_label(b)}"
            for s, b in self.pairs
        ]

    def code(self, subtype_index, band_index) -> np.ndarray:
        """Joint category index per observation. Raises on an impossible pair."""
        s = np.asarray(subtype_index, dtype=int)
        b = np.asarray(band_index, dtype=int)
        lookup = {pair: k for k, pair in enumerate(self.pairs)}
        out = np.empty(len(s), dtype=int)
        for i, (si, bi) in enumerate(zip(s.tolist(), b.tolist())):
            key = (si, bi)
            if key not in lookup:
                raise ValueError(
                    f"({self.subtype_names[si]}, band {bi}) is not a category of "
                    "this alphabet; the band scheme and the subtype rule disagree"
                )
            out[i] = lookup[key]
        return out

    def midpoints(self) -> np.ndarray:
        """Representative ratio per category; NaN for the undefined ones.

        The topmost tail band is unbounded, so its representative is its lower
        edge plus half a candidate width. Used only for reporting.
        """
        out = np.full(self.n_categories, np.nan)
        for k, (_, b) in enumerate(self.pairs):
            if b < 0:
                continue
            lo, hi = self.scheme.band_bounds(b)
            out[k] = lo + TAIL_WIDTH / 2 if np.isinf(hi) else (lo + hi) / 2
        return out


def build_alphabet(scheme: BandScheme, subtype_names) -> JointAlphabet:
    """Every structurally possible ``(subtype, band)``, subtype blocks contiguous."""
    subtype_names = tuple(subtype_names)
    pairs: list[tuple[int, int]] = []
    blocks: list[tuple[int, ...]] = []

    for s, name in enumerate(subtype_names):
        block = []
        for b in range(scheme.n_bands):
            if band_allows_subtype(scheme, b, name):
                block.append(len(pairs))
                pairs.append((s, b))
        if band_allows_subtype(scheme, -1, name):
            block.append(len(pairs))
            pairs.append((s, -1))
        if not block:                                        # pragma: no cover
            raise ValueError(f"subtype {name!r} has no possible category")
        blocks.append(tuple(block))

    return JointAlphabet(scheme=scheme, subtype_names=subtype_names,
                         pairs=tuple(pairs), blocks=tuple(blocks))


def flat_alphabet(subtype_names) -> JointAlphabet:
    """The degenerate alphabet: one category per subtype, no band resolution.

    This is the square model's alphabet expressed as a joint one. Fitting
    :func:`ppmi_subtype.hmm_banded.fit_banded` over it must reproduce
    :func:`ppmi_subtype.hmm.fit_hmm`, which is how the experiment separates
    "the bands changed the answer" from "the new code path changed the answer".
    """
    names = tuple(subtype_names)
    scheme = BandScheme((TAIL_START,), td_threshold=np.inf, pigd_threshold=-np.inf)
    return JointAlphabet(
        scheme=scheme, subtype_names=names,
        pairs=tuple((s, -1) for s in range(len(names))),
        blocks=tuple((s,) for s in range(len(names))),
    )


def aggregate_to_subtype(joint_probs: np.ndarray, alphabet: JointAlphabet) -> np.ndarray:
    """Sum a distribution over joint categories into one over the subtypes.

    ``P(TD) = sum of the joint probabilities of every TD category``, and the
    same for the other two; the result is renormalised so it sums to one even
    if the input does not. Works on a single distribution or a stack of them.
    """
    probs = np.atleast_2d(np.asarray(joint_probs, dtype=float))
    if probs.shape[1] != alphabet.n_categories:
        raise ValueError(
            f"expected {alphabet.n_categories} joint categories, got {probs.shape[1]}")

    out = np.column_stack([probs[:, list(block)].sum(axis=1)
                           for block in alphabet.blocks])
    total = out.sum(axis=1, keepdims=True)
    if np.any(total <= 0) or not np.all(np.isfinite(total)):
        raise ValueError("joint probabilities do not sum to a positive total")
    out = out / total
    return out[0] if np.ndim(joint_probs) == 1 else out
