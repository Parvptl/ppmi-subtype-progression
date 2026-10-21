"""D1: calibrated posterior-predictive label check of the heavy-tailed ratio model.

Protocol: docs/cluster_phase/12_CONSOLIDATED_RESEARCH_REPORT.md, Appendix B
(Revision 2, B.10; Amendments B.11, B.12). Scope: ratio-defined visits only
(PIGD = 0 visits excluded), so nothing here bears on whether the full
three-state phenotype process needs an intermediate state.

Stages for one cohort (observed or synthetic):
  A  fit TT (hbayes_tt_nu, collapsed slice, Gamma(2, 0.1) nu prior) on 4 folds
  B  for each test patient: (b, lambda, omega) | own ratios, theta^(r)
  C  fresh t noise at the patient's own visit times -> replicate ratio -> label
  D  statistics S1-S6 on observed and replicated label sequences
Discrepancy: mid-p tail probability, z = Phi^-1(p), T = max |z| over S1-S5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats

from . import hbayes
from . import hbayes_tt_nu as NU
from .hbayes_tt import _conditional_b

TD_T, PIGD_T = hbayes.TD_THRESHOLD, hbayes.PIGD_THRESHOLD
STAT_NAMES = ("S1_ind_share", "S2_R_E", "S3_R_D", "S4_delta_R", "S5_OE_ind")
DESC_NAMES = ("S6_two_step",)
FIT_KW = {"n_draws": 4000, "burn_in": 2000, "thin": 5}
N_INNER = 20


# ---------------------------------------------------------------------------
#  Data helpers
# ---------------------------------------------------------------------------
def labels_from_ratio(y) -> np.ndarray:
    """0 = TD, 1 = Indeterminate, 2 = PIGD by the fixed (outer) thresholds."""
    y = np.asarray(y, float)
    return np.where(y >= TD_T, 0, np.where(y <= PIGD_T, 2, 1)).astype(np.int8)


def subset(data: hbayes.GrowthData, keep_patients, y=None) -> hbayes.GrowthData:
    """GrowthData restricted to `keep_patients` (index order preserved)."""
    y = data.y if y is None else y
    keep = np.isin(data.patients, keep_patients)
    rows = keep[data.patient_index]
    new_idx = -np.ones(len(data.patients), int)
    new_idx[keep] = np.arange(keep.sum())
    return hbayes.GrowthData(y=y[rows], t=data.t[rows],
                             patient_index=new_idx[data.patient_index[rows]],
                             patients=data.patients[keep], n_patients=int(keep.sum()),
                             n_obs=int(rows.sum()))


def make_folds(data: hbayes.GrowthData, k: int = 5, seed: int = 42) -> pd.DataFrame:
    """Disjoint whole-patient folds, stratified by first ratio-defined label."""
    first = np.zeros(data.n_patients, int)
    lab = labels_from_ratio(data.y)
    seen = np.zeros(data.n_patients, bool)
    for r in range(data.n_obs):
        p = data.patient_index[r]
        if not seen[p]:
            first[p] = lab[r]
            seen[p] = True
    rng = np.random.default_rng(seed)
    fold = np.empty(data.n_patients, int)
    for s in (0, 1, 2):
        members = np.flatnonzero(first == s)
        members = members[rng.permutation(len(members))]
        fold[members] = np.arange(len(members)) % k
    return pd.DataFrame({"PATNO": data.patients, "fold": fold, "first_label": first})


# ---------------------------------------------------------------------------
#  Statistics (vectorised over replicates; equal to indeterminate_audit)
# ---------------------------------------------------------------------------
def label_statistics(L: np.ndarray, pid: np.ndarray) -> np.ndarray:
    """S1-S6 for each row of L (R x N label codes), visits sorted by patient then time.

    Definitions are those of `indeterminate_audit` (unobserved outcomes excluded
    from denominators; adjacency within patient; E = sum m(m-1)/n).
    """
    L = np.atleast_2d(L)
    R, N = L.shape
    pid = np.asarray(pid)
    def same(off):
        s = np.zeros(N, bool)
        if N > off:
            s[:-off] = pid[:-off] == pid[off:]
        return s
    s1, s2, s3 = same(1), same(2), same(3)
    pad = np.full((R, 3), -1, L.dtype)
    Lp = np.concatenate([L, pad], axis=1)
    a, b, c, d = Lp[:, :N], Lp[:, 1:N + 1], Lp[:, 2:N + 2], Lp[:, 3:N + 3]
    ind = L == 1
    S1 = ind.mean(axis=1)
    ent = s1 & (a != 1) & (b == 1)
    dirc = s1 & (((a == 0) & (b == 2)) | ((a == 2) & (b == 0)))
    eo, do = ent & s2, dirc & s2
    er, dr = eo & (c == a), do & (c == a)
    with np.errstate(invalid="ignore", divide="ignore"):
        RE = er.sum(1) / eo.sum(1)
        RD = dr.sum(1) / do.sum(1)
        O = (s1 & ind & (b == 1)).sum(1)
        starts = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
        n = np.diff(np.r_[starts, N])
        m = np.add.reduceat(ind.astype(float), starts, axis=1)
        E = (m * (m - 1) / n).sum(1)
        OE = np.where(E > 0, O / E, np.nan)
        e2 = eo & s3
        S6 = (e2 & (c == 1) & (d == 1)).sum(1) / e2.sum(1)
    return np.column_stack([S1, RE, RD, RE - RD, OE, S6])


# ---------------------------------------------------------------------------
#  Discrepancy, threshold, power and verdict
# ---------------------------------------------------------------------------
def midp(obs: float, rep: np.ndarray) -> tuple[float, int]:
    """Mid-p tail probability (rep > obs) with the observed counted as one extra draw.

    Undefined (NaN) replicate values are dropped; returns (p, n_used)."""
    rep = np.asarray(rep, float)
    rep = rep[np.isfinite(rep)]
    if not np.isfinite(obs) or len(rep) == 0:
        return np.nan, len(rep)
    gt, eq = (rep > obs).sum(), (rep == obs).sum()
    return (gt + 0.5 * eq + 0.5) / (len(rep) + 1), len(rep)


def z_cap(R: int) -> float:
    return float(sstats.norm.ppf((R + 0.5) / (R + 1)))


def discrepancy(S_obs: np.ndarray, S_rep: np.ndarray):
    """p, z for S1-S5 and T = max |z| (over defined statistics)."""
    R = S_rep.shape[0]
    p = np.array([midp(S_obs[j], S_rep[:, j])[0] for j in range(len(STAT_NAMES))])
    z = sstats.norm.ppf(p)
    T = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else np.nan
    return p, z, T, R


def threshold_index(n_null: int, level: float = 0.95) -> int:
    """1-based order statistic k = ceil(level (N + 1))."""
    return int(np.ceil(level * (n_null + 1)))


def threshold(T_null) -> float:
    t = np.sort(np.asarray(T_null, float))
    return float(t[threshold_index(len(t)) - 1])


def clopper_pearson_lower(x: int, n: int, alpha: float = 0.05) -> float:
    return 0.0 if x == 0 else float(sstats.beta.ppf(alpha / 2, x, n - x + 1))


def verdict(T_obs: float, c: float, x_primary: int, n_primary: int, cap: float) -> str:
    """Frozen verdict rule (B.11.5)."""
    if np.isclose(c, cap):
        return "CALIBRATION DEGENERATE (threshold at z cap): NOT REPRODUCED cannot be issued"
    if T_obs > c:
        return "NOT REPRODUCED"
    if clopper_pearson_lower(x_primary, n_primary) >= 0.50:
        return "REPRODUCED (within scope)"
    return "UNINFORMATIVE (low power)"


def read_frozen(path):
    """Steps after calibration may only read frozen files; refuse if absent."""
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"frozen file {p} not found: earlier step not completed")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
#  Cohort generation (null and positive controls)
# ---------------------------------------------------------------------------
def theta_from(post, draw: int) -> dict:
    return {"beta0": float(post.beta[draw, 0]), "beta1": float(post.beta[draw, 1]),
            "tau0": float(np.sqrt(post.tau0_2[draw])), "tau1": float(np.sqrt(post.tau1_2[draw])),
            "nu_s": float(post.nu[draw]), "sigma": float(np.sqrt(post.sigma2[draw])),
            "nu_e": float(post.nu_e[draw])}


def generate(data: hbayes.GrowthData, theta: dict, rng, episode: dict | None = None):
    """Synthetic ratios on the real visit design.

    episode = {pi, L, D, kappa, replace_level}. All random draws are made in the
    same order whatever the episode settings, so PC4 and PC4v with the same
    generator state share every draw (common random numbers).
    Returns (y, inside_mask).
    """
    M, i, t = data.n_patients, data.patient_index, data.t
    b0 = rng.normal(0.0, theta["tau0"], M)
    lam = rng.gamma(theta["nu_s"] / 2, 2.0 / theta["nu_s"], M)
    b1 = theta["tau1"] * rng.standard_normal(M) / np.sqrt(lam)
    om = rng.gamma(theta["nu_e"] / 2, 2.0 / theta["nu_e"], data.n_obs)
    e = rng.standard_normal(data.n_obs) / np.sqrt(om)
    tf = np.full(M, np.inf)
    tl = np.full(M, -np.inf)
    np.minimum.at(tf, i, t)
    np.maximum.at(tl, i, t)
    u_sel = rng.random(M)
    s0 = tf + rng.random(M) * (tl - tf)
    dur_unit = rng.exponential(1.0, M)
    mu = theta["beta0"] + b0[i] + (theta["beta1"] + b1[i]) * t
    scale = np.ones(data.n_obs)
    inside = np.zeros(data.n_obs, bool)
    if episode is not None:
        planted = u_sel < episode["pi"]
        d = dur_unit * episode["D"]
        inside = planted[i] & (t >= s0[i]) & (t < s0[i] + d[i])
        if episode.get("replace_level", True):
            mu = np.where(inside, episode["L"], mu)
        scale = np.where(inside, episode["kappa"], 1.0)
    y = mu + theta["sigma"] * scale * e
    return y, inside


# ---------------------------------------------------------------------------
#  Patient-conditional replicates
# ---------------------------------------------------------------------------
def patient_replicates(post, picks, t_hist, y_hist, rng, n_inner: int = N_INNER) -> np.ndarray:
    """Replicate ratios (R x n) for ONE patient at their own visit times.

    Uses only this patient's ratios and the training posterior draws `picks`:
    (b, lambda, omega) | y_i, theta^(r) by `n_inner` Gibbs steps from the prior,
    then fresh t noise. Nothing from other test patients enters.
    """
    t_hist = np.asarray(t_hist, float)
    y_hist = np.asarray(y_hist, float)
    Xh = np.column_stack([np.ones_like(t_hist), t_hist])
    beta = post.beta[picks]
    s2, t0, t1 = post.sigma2[picks], post.tau0_2[picks], post.tau1_2[picks]
    nus, nue = post.nu[picks], post.nu_e[picks]
    S, n = len(picks), len(t_hist)
    lam = rng.gamma(nus / 2, 2.0 / nus)
    w = rng.gamma(nue[:, None] / 2, 2.0 / nue[:, None], (S, n))
    for _ in range(n_inner):
        m, V = _conditional_b(Xh, y_hist, beta, s2, t0, t1, lam, w)
        L = np.linalg.cholesky(V)
        b = m + np.einsum("sij,sj->si", L, rng.standard_normal((S, 2)))
        lam = rng.gamma((nus + 1) / 2, 1.0 / ((nus + b[:, 1] ** 2 / t1) / 2))
        res = y_hist[None, :] - (beta + b) @ Xh.T
        w = rng.gamma((nue[:, None] + 1) / 2, 1.0 / ((nue[:, None] + res ** 2 / s2[:, None]) / 2))
    om = rng.gamma(nue[:, None] / 2, 2.0 / nue[:, None], (S, n))
    noise = np.sqrt(s2)[:, None] * rng.standard_normal((S, n)) / np.sqrt(om)
    return (beta + b) @ Xh.T + noise


# ---------------------------------------------------------------------------
#  One cohort through stages A-D
# ---------------------------------------------------------------------------
def run_cohort(data: hbayes.GrowthData, y: np.ndarray, folds: pd.DataFrame, R: int,
               fit_kw: dict, seed_entropy: list, record_stats: bool = True) -> dict:
    """Stages A-D for one cohort with ratios `y` on the design of `data`."""
    import time
    timing = {"fit_s": 0.0, "replicate_s": 0.0, "stats_s": 0.0}
    fold_of = folds.set_index("PATNO").fold.reindex(data.patients).to_numpy()
    pid = data.patient_index
    Yrep = np.empty((R, data.n_obs))
    for f in sorted(np.unique(fold_of)):
        test_p = data.patients[fold_of == f]
        train = subset(data, data.patients[fold_of != f], y=y)
        t0 = time.time()
        fit_seed = int(np.random.SeedSequence(seed_entropy + [f, 1]).generate_state(1)[0] % 2**31)
        post = NU.fit(train, "TT", nu_update="collapsed_slice",
                      log_prior_nu=NU.log_prior_nu_gamma2_true, seed=fit_seed, **fit_kw)
        timing["fit_s"] += time.time() - t0
        t0 = time.time()
        picks = np.linspace(0, post.n_draws - 1, R).astype(int)
        rng = np.random.default_rng(seed_entropy + [f, 2])
        for p in test_p:
            pi = np.flatnonzero(data.patients == p)[0]
            rows = np.flatnonzero(pid == pi)
            Yrep[:, rows] = patient_replicates(post, picks, data.t[rows], y[rows], rng)
        timing["replicate_s"] += time.time() - t0
    t0 = time.time()
    Lobs = labels_from_ratio(y)[None, :]
    Lrep = labels_from_ratio(Yrep)
    S_obs = label_statistics(Lobs, pid)[0]
    S_rep = label_statistics(Lrep, pid)
    p, z, T, _ = discrepancy(S_obs, S_rep)
    fold_rows = []
    for f in sorted(np.unique(fold_of)):
        rows = np.flatnonzero(fold_of[pid] == f)
        so = label_statistics(Lobs[:, rows], pid[rows])[0]
        sr = label_statistics(Lrep[:, rows], pid[rows])
        pf, zf, Tf, _ = discrepancy(so, sr)
        fold_rows.append({"fold": int(f), "T": Tf, **{f"z_{n}": v for n, v in zip(STAT_NAMES, zf)}})
    timing["stats_s"] = time.time() - t0
    cap = z_cap(R)
    pmin, pmax = 0.5 / (R + 1), (R + 0.5) / (R + 1)
    out = {"timing": timing, "R": R, "cap": cap}
    if record_stats:                 # nothing about the discrepancy is returned otherwise
        out.update({"p_at_bound": int(np.sum(np.isclose(p, pmin) | np.isclose(p, pmax))),
                    "T_at_cap": bool(np.isclose(T, cap)),
                    "n_rep_undefined": {n: int((~np.isfinite(S_rep[:, j])).sum())
                                        for j, n in enumerate(STAT_NAMES + DESC_NAMES)},"S_obs": dict(zip(STAT_NAMES + DESC_NAMES, S_obs.tolist())),
                    "S_rep_mean": dict(zip(STAT_NAMES + DESC_NAMES,
                                           np.nanmean(S_rep, axis=0).tolist())),
                    "p": dict(zip(STAT_NAMES, p.tolist())), "z": dict(zip(STAT_NAMES, z.tolist())),
                    "T": T, "folds": fold_rows})
    return out
