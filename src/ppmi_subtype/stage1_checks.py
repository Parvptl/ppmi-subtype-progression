"""Stage 1 descriptive checks after the D1 label check (doc 14).

Four descriptive checks recommended by docs/cluster_phase/13_D1_POST_RUN_AUDIT.md
section 5. Nothing here fits a model or simulates a cohort:

* observed quantities are counts on the frozen panel;
* model-based references are expectations under the frozen D1 *generator*
  (the full-cohort TT posterior draws in `fullcohort_tt_draws.npz`, the model
  that produced the 500 null cohorts), computed by deterministic quadrature
  over the patient effects, with the exact Student-t residual CDF;
* the lattice functions are arithmetic on the unchanged labelling rule
  (`raw_scores.label_grid`).

Generator (`ppc_diagnostic.generate`): y = beta0 + b0 + (beta1 + b1) t + sigma e,
b0 ~ N(0, tau0^2), b1 = tau1 t_{nu_s}, e ~ t_{nu_e}, all independent. Given
(b0, b1) the visits of a patient are independent, so pair and triplet label
probabilities are products of per-visit probabilities inside the quadrature.

Label codes are those of `ppc_diagnostic.labels_from_ratio`:
0 = TD, 1 = Indeterminate, 2 = PIGD.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy import sparse
from scipy import special
from scipy import stats as sstats

from .hbayes import PIGD_THRESHOLD, TD_THRESHOLD

TD, IND, PIGD = 0, 1, 2
GRID_CODE = {1.0: TD, 2.0: IND, 3.0: PIGD}       # raw_scores.label_grid codes
DIRECTIONS = ("TD->PIGD", "PIGD->TD")
OUTCOMES = ("revert", "stay", "to_Ind", "unobserved")
N_TREMOR, N_PIGD = 11, 5


# ---------------------------------------------------------------------------
#  Lattice arithmetic (Check 1)
# ---------------------------------------------------------------------------
def grid_labels(grid: np.ndarray) -> np.ndarray:
    """`raw_scores.label_grid` (tremor sum x PIGD sum) recoded to 0/1/2 (NaN -> -1)."""
    out = np.full(grid.shape, -1, int)
    for k, v in GRID_CODE.items():
        out[grid == k] = v
    return out


def lattice_row(grid: np.ndarray, psum: int, n_tremor: int = N_TREMOR,
                n_pigd: int = N_PIGD) -> dict:
    """Indeterminate lattice points and their catchment at one PIGD sum.

    The ratio takes the values r_T = (T / n_tremor) / (psum / n_pigd), T = 0, 1, ...
    Catchment: the interval of a *continuous* ratio that rounds to an
    Indeterminate lattice point, bounded by the midpoints to the neighbouring
    lattice points (infinite if there is no neighbour). `inflation` is its width
    over the nominal band width (TD_THRESHOLD - PIGD_THRESHOLD).
    """
    lab = grid_labels(grid)[:, psum]
    T = np.arange(lab.size)
    r = (T / n_tremor) / (psum / n_pigd)
    ind = np.flatnonzero(lab == IND)
    step = n_pigd / (n_tremor * psum)
    if ind.size == 0:
        return {"pigd_sum": psum, "ind_tremor_sums": [], "ind_ratios": [], "step": step,
                "lo": np.nan, "hi": np.nan, "width": 0.0, "inflation": 0.0}
    lo = (r[ind[0] - 1] + r[ind[0]]) / 2 if ind[0] > 0 else -np.inf
    hi = (r[ind[-1]] + r[ind[-1] + 1]) / 2 if ind[-1] + 1 < lab.size else np.inf
    width = hi - lo
    return {"pigd_sum": psum, "ind_tremor_sums": ind.tolist(), "ind_ratios": r[ind].tolist(),
            "step": step, "lo": lo, "hi": hi, "width": width,
            "inflation": width / (TD_THRESHOLD - PIGD_THRESHOLD)}


def catchment_bounds(grid: np.ndarray, psum) -> tuple[np.ndarray, np.ndarray]:
    """Per-visit (lo, hi) continuous-ratio cut points under the rounding heuristic."""
    psum = np.asarray(psum, int)
    rows = {p: lattice_row(grid, int(p)) for p in np.unique(psum)}
    return (np.array([rows[p]["lo"] for p in psum]), np.array([rows[p]["hi"] for p in psum]))


def delatticed_ind(psum, tsum, grid: np.ndarray, max_psum: int = 20):
    """Observed Indeterminate counts per PIGD sum, and what the band would hold
    if it captured only its nominal width of a locally flat lattice distribution.

    Rows: pigd_sum, n, ind (observed), n_ind_points, inflation (lattice_row),
    delatticed = ind / inflation, cell_mean (mean count per Indeterminate cell),
    neighbour_mean (mean count of the two adjacent non-Indeterminate cells).
    `delatticed` is arithmetic on observed counts under an explicit assumption
    (local flatness), checked by comparing cell_mean with neighbour_mean.
    """
    psum, tsum = np.asarray(psum, int), np.asarray(tsum, int)
    rows = []
    for p in range(1, max_psum + 1):
        sel = psum == p
        if not sel.any():
            continue
        lr = lattice_row(grid, p)
        cnt = np.bincount(tsum[sel], minlength=grid.shape[0])
        pts = lr["ind_tremor_sums"]
        ind = int(cnt[pts].sum())
        nb = [cnt[k] for k in (pts[0] - 1, pts[-1] + 1) if 0 <= k < cnt.size]
        rows.append({"pigd_sum": p, "n": int(sel.sum()), "ind": ind, "n_ind_points": len(pts),
                     "inflation": lr["inflation"], "delatticed": ind / lr["inflation"],
                     "cell_mean": ind / len(pts), "neighbour_mean": float(np.mean(nb))})
    return rows


# ---------------------------------------------------------------------------
#  Observed label events (Checks 2 and 4)
# ---------------------------------------------------------------------------
def neighbours(pid: np.ndarray):
    """same(k)[j]: visit j+k exists and belongs to the same patient (sorted visits)."""
    pid = np.asarray(pid)
    N = len(pid)
    out = []
    for k in (1, 2):
        s = np.zeros(N, bool)
        if N > k:
            s[:-k] = pid[:-k] == pid[k:]
        out.append(s)
    return out


def direct_change_outcomes(L: np.ndarray, pid: np.ndarray):
    """Every direct TD<->PIGD change between consecutive visits, with its outcome.

    Returns (j, direction, outcome) arrays. j indexes the first visit of the
    pair; direction 0 = TD->PIGD, 1 = PIGD->TD; outcome is the label at j+2
    relative to the change: 0 revert (back to the origin), 1 stay (destination
    kept), 2 to Indeterminate, 3 unobserved (no later visit). Outcomes 0-2 form
    the S3 denominator of `ppc_diagnostic.label_statistics`; outcome 0 its numerator.
    """
    L = np.asarray(L)
    s1, s2 = neighbours(pid)
    N = len(L)
    a = L
    b = np.r_[L[1:], -1] if N else L
    c = np.r_[L[2:], -1, -1][:N] if N else L
    direct = s1 & (((a == TD) & (b == PIGD)) | ((a == PIGD) & (b == TD)))
    j = np.flatnonzero(direct)
    direction = (a[j] == PIGD).astype(int)
    outcome = np.where(~s2[j], 3, np.where(c[j] == a[j], 0, np.where(c[j] == b[j], 1, 2)))
    return j, direction, outcome


def per_patient(pid: np.ndarray, columns: np.ndarray, n_patients: int) -> np.ndarray:
    """Per-patient sums of per-visit columns: (n_patients, k)."""
    columns = np.atleast_2d(np.asarray(columns, float).T).T
    return np.column_stack([np.bincount(pid, weights=columns[:, k], minlength=n_patients)
                            for k in range(columns.shape[1])])


def bootstrap_ratio(boot_sums: np.ndarray, num: int, den: int, level: float = 0.95):
    """Percentile interval of sum[num] / sum[den] over patient-bootstrap sums."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = boot_sums[:, num] / boot_sums[:, den]
    r = r[np.isfinite(r)]
    a = (1 - level) / 2
    return (float(np.quantile(r, a)), float(np.quantile(r, 1 - a))) if r.size else (np.nan, np.nan)


# ---------------------------------------------------------------------------
#  Generator expectations by quadrature (Checks 1, 2 and 4)
# ---------------------------------------------------------------------------
def quadrature(theta: dict, n_b0: int = 40, n_b1: int = 32):
    """Product nodes and weights for (b0, b1) under the generator's distributions."""
    x0, w0 = hermegauss(n_b0)
    w0 = w0 / np.sqrt(2 * np.pi)
    x1, w1 = np.polynomial.legendre.leggauss(n_b1)
    u, w1 = (x1 + 1) / 2, w1 / 2
    b0 = theta["tau0"] * x0
    b1 = theta["tau1"] * sstats.t.ppf(u, theta["nu_s"])
    B0, B1 = np.meshgrid(b0, b1, indexing="ij")
    W = np.outer(w0, w1)
    return B0.ravel(), B1.ravel(), W.ravel()


def visit_label_probs(t, theta: dict, lo=None, hi=None, n_b0: int = 40, n_b1: int = 32):
    """Per-node label probabilities: arrays (K, N) for TD, Ind, PIGD, and weights (K,).

    Labels: PIGD if y <= lo, TD if y >= hi, Indeterminate otherwise; lo and hi
    default to the fixed thresholds and may be per-visit arrays (lattice heuristic).
    """
    t = np.asarray(t, float)
    lo = np.full(t.shape, PIGD_THRESHOLD) if lo is None else np.asarray(lo, float)
    hi = np.full(t.shape, TD_THRESHOLD) if hi is None else np.asarray(hi, float)
    B0, B1, W = quadrature(theta, n_b0, n_b1)
    mu = theta["beta0"] + B0[:, None] + (theta["beta1"] + B1[:, None]) * t[None, :]
    s, df = theta["sigma"], theta["nu_e"]
    F_lo = special.stdtr(df, (lo[None, :] - mu) / s)
    F_hi = special.stdtr(df, (hi[None, :] - mu) / s)
    return 1.0 - F_hi, F_hi - F_lo, F_lo, W


def expected_visit_quantities(t, pid, theta: dict, lo=None, hi=None, groups=None,
                              n_b0: int = 40, n_b1: int = 32) -> dict:
    """Expectations under one generator draw theta, per visit position j.

    ind[j]        P(visit j Indeterminate)
    pair[d][j]    P(direct change in direction d between j and j+1)
    trip[d][o][j] P(that change and outcome o at j+2); o in revert/stay/to_Ind,
                  defined only where j+2 is the same patient (else 0)
    unobs[d][j]   P(that change with no visit j+2)
    Also `ind_var[name][g]`: for each partition `groups[name]` (per-visit codes),
    the variance of the Indeterminate count in group g (patients independent
    given theta; visits independent given b).
    """
    pTD, pI, pP, W = visit_label_probs(t, theta, lo, hi, n_b0, n_b1)
    N = len(t)
    s1, s2 = neighbours(pid)

    def sh(p, k):
        out = np.zeros_like(p)
        out[:, :N - k] = p[:, k:]
        return out
    P = {TD: pTD, IND: pI, PIGD: pP}
    out = {"ind": W @ pI, "pair": {}, "trip": {}, "unobs": {}}
    for d, (a, b) in enumerate(((TD, PIGD), (PIGD, TD))):
        pair = P[a] * sh(P[b], 1)
        out["pair"][d] = (W @ pair) * s1
        out["unobs"][d] = (W @ pair) * (s1 & ~s2)
        c_of = {"revert": a, "stay": b, "to_Ind": IND}
        out["trip"][d] = {o: (W @ (pair * sh(P[c], 2))) * s2 for o, c in c_of.items()}
    out["ind_var"] = {}
    for name, codes in (groups or {}).items():
        codes = np.asarray(codes, int)
        n_g = int(codes.max()) + 1
        cell = np.asarray(pid, int) * n_g + codes
        uniq, inv = np.unique(cell, return_inverse=True)
        A = sparse.csr_matrix((np.ones(N), (np.arange(N), inv)), shape=(N, len(uniq)))
        m_k = (A.T @ pI.T).T                         # (K, cells) mean given node
        v_k = (A.T @ (pI * (1 - pI)).T).T            # (K, cells) variance given node
        mean = W @ m_k
        var_cell = W @ (v_k + m_k ** 2) - mean ** 2
        out["ind_var"][name] = np.bincount(uniq % n_g, weights=var_cell, minlength=n_g)
    return out


def expected_cdf(t, theta: dict, cuts, n_b0: int = 40, n_b1: int = 32) -> np.ndarray:
    """Expected number of visits with y <= c under the generator, for each cut c."""
    t = np.asarray(t, float)
    B0, B1, W = quadrature(theta, n_b0, n_b1)
    mu = theta["beta0"] + B0[:, None] + (theta["beta1"] + B1[:, None]) * t[None, :]
    return np.array([(W @ special.stdtr(theta["nu_e"], (c - mu) / theta["sigma"])).sum()
                     for c in cuts])


# ---------------------------------------------------------------------------
#  Residual-shape proxy (Check 3)
# ---------------------------------------------------------------------------
def centred_detrended(y, t, pid, beta1: float) -> np.ndarray:
    """y - beta1 t, centred on each patient's own mean.

    Under the generator this equals b1 (t - tbar_i) + sigma (e - ebar_i), which is
    symmetric about 0 in law for every visit (b1 and e are symmetric and
    independent), whatever nu_s and nu_e. Uses only the posterior beta1.
    """
    y, t, pid = np.asarray(y, float), np.asarray(t, float), np.asarray(pid, int)
    z = y - beta1 * t
    n = np.bincount(pid)
    m = np.bincount(pid, weights=z) / n
    return z - m[pid]


def bowley(x) -> float:
    """Quartile (Bowley) skewness; 0 for any symmetric distribution, defined without moments."""
    q1, q2, q3 = np.quantile(np.asarray(x, float), [0.25, 0.5, 0.75])
    return float((q3 + q1 - 2 * q2) / (q3 - q1)) if q3 > q1 else np.nan


def tail_asymmetry(x, q: float = 0.05) -> float:
    """(q_{1-q} + q_q) / (q_{1-q} - q_q): 0 for a symmetric distribution about 0."""
    lo, hi = np.quantile(np.asarray(x, float), [q, 1 - q])
    return float((hi + lo) / (hi - lo)) if hi > lo else np.nan
