"""Label-free fidelity features and gates G1-G7 for M2 (doc 16, B6 and D3).

Every feature here is computed from item responses or item sums only; nothing
calls the label rule. Arrays are visit-level, sorted by patient then time;
`pid` holds integer patient codes.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sstats

P_BINS = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 99))
T_BINS = ((0, 0), (1, 3), (4, 5), (6, 8), (9, 11), (12, 999))


def consecutive(pid):
    pid = np.asarray(pid)
    return np.flatnonzero(pid[1:] == pid[:-1])


def item_props(YT, YP) -> np.ndarray:
    """(KT + KP, 5) category proportions; Y are (V, K) integer arrays."""
    Y = np.concatenate([np.asarray(YT), np.asarray(YP)], axis=1).astype(int)
    return np.stack([np.bincount(Y[:, k], minlength=5)[:5] / len(Y) for k in range(Y.shape[1])])


def bin_props(x, bins) -> np.ndarray:
    x = np.asarray(x)
    return np.array([((x >= lo) & (x <= hi)).mean() for lo, hi in bins])


def zero_persistence(P, pid):
    """P(next P = 0 | P = 0), P(next P = 0 | P = 1) over consecutive visits."""
    j = consecutive(pid)
    P = np.asarray(P)
    a, b = P[j], P[j + 1]
    return ((b[a == 0] == 0).mean() if (a == 0).any() else np.nan,
            (b[a == 1] == 0).mean() if (a == 1).any() else np.nan)


def change_stats(T, P, pid):
    j = consecutive(pid)
    T, P = np.asarray(T, float), np.asarray(P, float)
    dT, dP = T[j + 1] - T[j], P[j + 1] - P[j]
    return {"sd_dT": float(dT.std(ddof=1)), "sd_dP": float(dP.std(ddof=1)),
            "skew_dP": float(sstats.skew(dP)), "spearman_dT_dP": float(sstats.spearmanr(dT, dP)[0])}


def dispersion_ratio(x, pid, n_items: int, min_visits: int = 3) -> float:
    """Within-patient residual variance over the binomial variance at the patient mean.

    Exactly as in scripts/44_stage2_data_audit.py (frozen definition, doc 16 D3):
    squared deviations from each patient's mean over the visits of patients with
    >= `min_visits` visits, divided by (those visits - the number of ALL patients).
    The divisor is that script's; applied identically to observed and simulated data.
    """
    x, pid = np.asarray(x, float), np.asarray(pid)
    _, inv, cnt = np.unique(pid, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=x)
    m = (sums / cnt)[inv]
    nn = cnt[inv]
    sel = nn >= min_visits
    n_groups = len(cnt)
    resid_var = ((x - m)[sel] ** 2).sum() / (sel.sum() - n_groups)
    pbar = np.clip(m[sel] / (4 * n_items), 1e-9, 1 - 1e-9)
    binom = (4 * n_items * pbar * (1 - pbar)).mean()
    return float(resid_var / binom)


def features(YT, YP, pid) -> dict:
    """All gated features from item responses (V, KT), (V, KP) and patient codes."""
    T, P = np.asarray(YT).sum(1), np.asarray(YP).sum(1)
    z0, z1 = zero_persistence(P, pid)
    return {"G1_item_props": item_props(YT, YP),
            "G2_P_zero": float((P == 0).mean()), "G2_T_zero": float((T == 0).mean()),
            "G3_P_bins": bin_props(P, P_BINS), "G3_T_bins": bin_props(T, T_BINS),
            "G4_next0_given0": float(z0), "G4_next0_given1": float(z1),
            "G5": change_stats(T, P, pid),
            "G6_T": dispersion_ratio(T, pid, np.asarray(YT).shape[1]),
            "G6_P": dispersion_ratio(P, pid, np.asarray(YP).shape[1]),
            "G7_spearman_T_P": float(sstats.spearmanr(T, P)[0])}


def _mc(obs, sims, k=2.5, floor=0.0):
    """Per-element test: |obs - mean(sims)| <= max(k * SD(sims), floor)."""
    sims = np.asarray(sims, float)
    mu, sd = sims.mean(0), sims.std(0, ddof=1)
    tol = np.maximum(k * sd, floor)
    dev = np.asarray(obs, float) - mu
    return {"observed": np.asarray(obs, float), "sim_mean": mu, "sim_sd": sd, "tolerance": tol,
            "deviation": dev, "pass_each": np.abs(dev) <= tol, "passed": bool(np.all(np.abs(dev) <= tol))}


def _fixed(obs, sims, tol):
    mu = float(np.mean(sims))
    return {"observed": float(obs), "sim_mean": mu, "sim_sd": float(np.std(sims, ddof=1)),
            "tolerance": tol, "deviation": float(obs) - mu, "passed": bool(abs(float(obs) - mu) <= tol)}


def evaluate(obs: dict, sims: list[dict]) -> dict:
    """Gates G1-G7 with the frozen tolerances (doc 16 B6)."""
    get = lambda k: [s[k] for s in sims]  # noqa: E731
    g = {}
    g["G1"] = _mc(obs["G1_item_props"], get("G1_item_props"), floor=0.02)
    g["G2_P_zero"] = _fixed(obs["G2_P_zero"], get("G2_P_zero"), 0.03)
    g["G2_T_zero"] = _fixed(obs["G2_T_zero"], get("G2_T_zero"), 0.02)
    g["G3_P"] = _mc(obs["G3_P_bins"], get("G3_P_bins"), floor=0.02)
    g["G3_T"] = _mc(obs["G3_T_bins"], get("G3_T_bins"), floor=0.02)
    g["G4_next0_given0"] = _fixed(obs["G4_next0_given0"], get("G4_next0_given0"), 0.05)
    g["G4_next0_given1"] = _fixed(obs["G4_next0_given1"], get("G4_next0_given1"), 0.035)
    keys = ("sd_dT", "sd_dP", "skew_dP", "spearman_dT_dP")
    g["G5"] = _mc([obs["G5"][k] for k in keys], [[s["G5"][k] for k in keys] for s in sims])
    g["G5"]["names"] = list(keys)
    g["G6_T"] = _fixed(obs["G6_T"], get("G6_T"), 0.3)
    g["G6_P"] = _fixed(obs["G6_P"], get("G6_P"), 0.3)
    g["G7"] = _fixed(obs["G7_spearman_T_P"], get("G7_spearman_T_P"), 0.05)
    family = {"G1": ["G1"], "G2": ["G2_P_zero", "G2_T_zero"], "G3": ["G3_P", "G3_T"],
              "G4": ["G4_next0_given0", "G4_next0_given1"], "G5": ["G5"], "G6": ["G6_T", "G6_P"],
              "G7": ["G7"]}
    summary = {f: all(g[k]["passed"] for k in ks) for f, ks in family.items()}
    core = all(summary[f] for f in ("G1", "G2", "G3", "G5", "G6", "G7"))
    return {"gates": g, "summary": summary,
            "decision": {"core_gates_G1_G3_G5_G7_pass": core, "G4_pass": summary["G4"],
                         "family_A_evaluable": core,
                         "family_B_evaluable": core and summary["G4"],
                         "target2_uninformative_by_rule": not summary["G4"]}}


# ---------------------------------------------------------------------------
#  MCMC diagnostics for the patient sampler (doc 16 B11 / D3)
# ---------------------------------------------------------------------------
def _rank_normal(x):
    from scipy.stats import norm, rankdata
    r = rankdata(x.ravel(), method="average").reshape(x.shape)
    return norm.ppf((r - 0.375) / (x.size + 0.25))


def _split(x):
    """(chains, draws) -> (2 * chains, draws // 2)."""
    h = x.shape[1] // 2
    return np.concatenate([x[:, :h], x[:, h:2 * h]], axis=0)


def split_rhat(x) -> float:
    """Rank-normalised split-R̂ (Vehtari et al. 2021) for draws x (chains, draws)."""
    z = _split(_rank_normal(np.asarray(x, float)))
    m, n = z.shape
    W = z.var(axis=1, ddof=1).mean()
    B = n * z.mean(axis=1).var(ddof=1)
    var = (n - 1) / n * W + B / n
    return float(np.sqrt(var / W)) if W > 0 else np.nan


def bulk_ess(x) -> float:
    """Bulk effective sample size (rank-normalised, split chains, Geyer initial monotone sequence)."""
    z = _split(_rank_normal(np.asarray(x, float)))
    m, n = z.shape
    zc = z - z.mean(axis=1, keepdims=True)
    acov = np.array([np.correlate(c, c, "full")[n - 1:] / n for c in zc])      # (m, n)
    W = z.var(axis=1, ddof=1).mean()
    var = (n - 1) / n * W + n * z.mean(axis=1).var(ddof=1) / n
    rho = 1 - (W - acov.mean(0)) / var
    rho[0] = 1.0
    tau, prev = -1.0, np.inf
    for t in range(0, n - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        pair = min(pair, prev)
        tau += 2 * pair
        prev = pair
    return float(m * n / max(tau, 1e-12))
