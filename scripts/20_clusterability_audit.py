"""Direction 3, first task: is this data clusterable at all?

The midterm wording is deliberate: "If per-patient trajectory info can be
extracted reliably, ask whether patients fall into distinct progression
patterns/phenotypes; not simply 'run clustering.'" The conditional comes first,
so this script tests the conditional and does **not** cluster.

What it measures, and why each one can veto the analysis
-------------------------------------------------------
1. How much data each patient actually contributes. A slope from two visits has
   no residual degrees of freedom; its uncertainty is not merely large, it is
   undefined.
2. Whether a patient's estimated slope survives dropping one visit. If the sign
   flips, that patient's "progression pattern" is not a property of the patient.
3. The reliability of the shrunken random slope, 1 - posterior var / prior var.
   This is the share of apparent between-patient slope variation that is signal.
   Classifying individuals conventionally wants 0.7 or better.
4. Whether apparent cluster structure also appears in a **null** built from the
   fitted model with one population and the same visit structure. If the null
   shows the same structure, model-selection indices are measuring the sampling
   design rather than biology.
5. Whether candidate clusters separate on nuisance quantities -- visit count,
   follow-up length, baseline severity -- rather than on progression shape.

    python scripts/20_clusterability_audit.py

Reads   outputs/visit_panel.csv
Writes  outputs/clustering_audit/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats as ss  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from ppmi_subtype import timevarying as TV, trajectory as tj  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

OUTCOMES = ("tremor_score", "pigd_score")
SEED = 42


def load_frame(cfg) -> pd.DataFrame:
    """The Steps 1-2 / Direction 2 cohort, unchanged."""
    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    traj = tj.build_trajectories(panel, cfg)
    tj.check_ordering(traj)
    return traj.merge(
        panel[["PATNO", "EVENT_ID", *OUTCOMES]].drop_duplicates(["PATNO", "EVENT_ID"]),
        on=["PATNO", "EVENT_ID"], how="left", validate="one_to_one")


def audit_counts(frame: pd.DataFrame) -> pd.DataFrame:
    n = frame.groupby("PATNO").size()
    span = frame.groupby("PATNO").years_from_baseline.max()
    rows = [{"quantity": "patients", "value": frame.PATNO.nunique()},
            {"quantity": "visits", "value": len(frame)},
            {"quantity": "median visits per patient", "value": float(n.median())},
            {"quantity": "median follow-up span (y)", "value": float(span.median())}]
    for k in (2, 3, 4, 5, 6, 8, 10):
        rows.append({"quantity": f"patients with >= {k} visits",
                     "value": int((n >= k).sum())})
    for k in (1, 2, 3, 4, 5, 8):
        rows.append({"quantity": f"patients reaching >= {k} years",
                     "value": int((span >= k).sum())})
    z = frame.groupby("PATNO").agg(n_pigd0=(OUTCOMES[1], lambda s: int((s == 0).sum())),
                                   n=(OUTCOMES[1], "size"))
    rows += [
        {"quantity": "visits with PIGD == 0 (ratio undefined)",
         "value": int((frame[OUTCOMES[1]] == 0).sum())},
        {"quantity": "patients with >= 1 undefined-ratio visit",
         "value": int((z.n_pigd0 > 0).sum())},
        {"quantity": "patients with every visit PIGD == 0",
         "value": int((z.n_pigd0 == z.n).sum())},
        {"quantity": "missing tremor values",
         "value": int(frame[OUTCOMES[0]].isna().sum())},
        {"quantity": "missing PIGD values",
         "value": int(frame[OUTCOMES[1]].isna().sum())},
    ]
    return pd.DataFrame(rows)


def audit_followup_bias(frame: pd.DataFrame) -> pd.DataFrame:
    """Do long-followed patients differ at baseline from short-followed ones?"""
    span = frame.groupby("PATNO").years_from_baseline.max()
    base = frame.sort_values("years_from_baseline").groupby("PATNO").first()
    long_ = span.reindex(base.index) >= 5.0
    rows = []
    for col in OUTCOMES:
        a, b = base.loc[long_, col].dropna(), base.loc[~long_, col].dropna()
        rows.append({"variable": f"baseline {col}", "long_mean": float(a.mean()),
                     "short_mean": float(b.mean()), "n_long": len(a), "n_short": len(b),
                     "mannwhitney_p": float(ss.mannwhitneyu(a, b).pvalue)})
    return pd.DataFrame(rows)


def audit_slope_reliability(frame: pd.DataFrame) -> tuple:
    """Naive per-patient OLS slopes and their leave-one-visit-out stability."""
    g = frame.sort_values(["PATNO", "years_from_baseline"])
    ols, loo = [], []
    for patno, gg in g.groupby("PATNO"):
        t = gg.years_from_baseline.to_numpy(float)
        if len(t) < 2 or np.ptp(t) <= 0:
            continue
        X = np.c_[np.ones(len(t)), t]
        XtXi = np.linalg.inv(X.T @ X)
        for col in OUTCOMES:
            y = gg[col].to_numpy(float)
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            dof = len(t) - 2
            se = np.nan
            if dof > 0:
                r = y - X @ beta
                se = float(np.sqrt((r @ r) / dof * XtXi[1, 1]))
            ols.append({"PATNO": patno, "outcome": col, "slope": float(beta[1]),
                        "se": se, "n_visits": len(t), "span": float(np.ptp(t)),
                        "dof": dof})
            if len(t) >= 3:
                drops = [np.polyfit(np.delete(t, i), np.delete(y, i), 1)[0]
                         for i in range(len(t)) if np.ptp(np.delete(t, i)) > 0]
                drops = np.asarray(drops)
                loo.append({"PATNO": patno, "outcome": col, "n_visits": len(t),
                            "full_slope": float(beta[1]),
                            "loo_sd": float(drops.std()),
                            "max_shift": float(np.abs(drops - beta[1]).max()),
                            "sign_flipped": bool(
                                (np.sign(drops) != np.sign(beta[1])).any())})
    return pd.DataFrame(ols), pd.DataFrame(loo)


def audit_blup_reliability(data, fit) -> pd.DataFrame:
    """Reliability of the shrunken random slope, per patient."""
    rows = []
    for patno in data.patients:
        t, Y = data.times[patno], data.responses[patno]
        b, post = TV.blup(fit, t, Y)
        rows.append({"PATNO": patno, "n_visits": len(t), "span": float(np.ptp(t)),
                     "intercept_tremor": b[0], "intercept_pigd": b[1],
                     "slope_tremor": b[2], "slope_pigd": b[3],
                     "post_sd_slope_tremor": float(np.sqrt(post[2, 2])),
                     "post_sd_slope_pigd": float(np.sqrt(post[3, 3])),
                     "reliability_tremor": 1.0 - post[2, 2] / fit.sigma_b[2, 2],
                     "reliability_pigd": 1.0 - post[3, 3] / fit.sigma_b[3, 3]})
    return pd.DataFrame(rows)


def simulate_null(data, fit, seed=SEED):
    """One population, identical visit structure, no groups by construction.

    The comparison that decides whether a model-selection index is detecting
    biology: whatever K this data prefers is what the index returns when the
    truth is a single Gaussian population.
    """
    rng = np.random.default_rng(seed)
    responses = {}
    for patno in data.patients:
        t = data.times[patno]
        n = len(t)
        b = rng.multivariate_normal(np.zeros(4), fit.sigma_b)
        e = rng.multivariate_normal(np.zeros(2), fit.sigma_e, size=n).reshape(-1)
        mu = TV.design(t, ()) @ fit.beta
        responses[patno] = (mu + TV.random_design(t, True) @ b + e).reshape(n, 2)
    groups = {k: (v[0], v[1], np.stack([responses[p] for p in v[0]]))
              for k, v in data.groups.items()}
    return TV.PackedData(patients=data.patients, times=data.times,
                         responses=responses, groups=groups)


def _labels(Z, K, seed=0):
    return GaussianMixture(K, covariance_type="full", random_state=seed,
                           n_init=10).fit_predict(Z)


def main() -> int:
    cfg = load_config()
    out = cfg.path("outputs") / "clustering_audit"
    out.mkdir(parents=True, exist_ok=True)

    frame = load_frame(cfg)
    counts = audit_counts(frame)
    counts.to_csv(out / "data_audit.csv", index=False)
    print("DATA AUDIT")
    print("  " + counts.to_string(index=False).replace("\n", "\n  "))

    bias = audit_followup_bias(frame)
    bias.to_csv(out / "followup_bias.csv", index=False)
    print("\nFOLLOW-UP SELECTION (baseline severity, >=5y vs <5y)")
    print("  " + bias.round(5).to_string(index=False).replace("\n", "\n  "))

    ols, loo = audit_slope_reliability(frame)
    ols.to_csv(out / "per_patient_ols_slopes.csv", index=False)
    loo.to_csv(out / "leave_one_visit_out.csv", index=False)
    print("\nPER-PATIENT SLOPE RELIABILITY")
    for col in OUTCOMES:
        x = ols[ols.outcome == col]
        ok = x[(x.dof > 0) & x.se.notna()]
        l = loo[loo.outcome == col]
        print(f"  {col}")
        print(f"    between-patient slope SD {x.slope.std():.4f}; "
              f"median within-patient SE {ok.se.median():.4f}")
        print(f"    patients with 2 visits (SE undefined): "
              f"{int((x.dof == 0).sum())}")
        print(f"    slope SD by visit count: " + ", ".join(
            f"n={k}:{x[x.n_visits == k].slope.std():.3f}" for k in (2, 3, 4, 5)
            if (x.n_visits == k).sum() > 5)
            + f", n>=8:{x[x.n_visits >= 8].slope.std():.3f}")
        print(f"    dropping ONE visit flips the slope sign for "
              f"{100 * l.sign_flipped.mean():.1f}% of patients with >=3 visits")

    data = TV.pack(frame)
    fit = TV.fit(data, knots=(), random_slopes=True)
    blup = audit_blup_reliability(data, fit)
    blup.to_csv(out / "random_slope_reliability.csv", index=False)
    print(f"\nRANDOM-SLOPE RELIABILITY (converged={fit.converged})")
    for short in ("tremor", "pigd"):
        r = blup[f"reliability_{short}"]
        print(f"  {short}: median {r.median():.3f}  "
              + "  ".join(f"n={k}:{r[blup.n_visits == k].median():.3f}"
                          for k in (2, 3, 4, 5))
              + f"  n>=8:{r[blup.n_visits >= 8].median():.3f}")

    # ------------------------------------------------- real versus null
    B = blup[["slope_tremor", "slope_pigd"]].to_numpy()
    null_data = simulate_null(data, fit)
    nullB = np.array([TV.blup(fit, null_data.times[p], null_data.responses[p])[0][2:4]
                      for p in null_data.patients])

    rows = []
    for K in (1, 2, 3, 4, 5):
        row = {"K": K}
        for name, X in (("real", B), ("null", nullB)):
            Z = (X - X.mean(0)) / X.std(0)
            row[f"{name}_BIC"] = GaussianMixture(
                K, covariance_type="full", random_state=0, n_init=5).fit(Z).bic(Z)
            row[f"{name}_silhouette"] = (
                np.nan if K == 1 else silhouette_score(Z, _labels(Z, K)))
        rows.append(row)
    model_sel = pd.DataFrame(rows)
    model_sel.to_csv(out / "real_vs_null_model_selection.csv", index=False)
    print("\nREAL vs SINGLE-POPULATION NULL (same visit structure, no groups)")
    print("  " + model_sel.round(3).to_string(index=False).replace("\n", "\n  "))

    # ------------------------------------- what would the clusters separate?
    Z = (B - B.mean(0)) / B.std(0)
    meta = blup.copy()
    meta["baseline_tremor"] = [data.responses[p][0, 0] for p in data.patients]
    meta["baseline_pigd"] = [data.responses[p][0, 1] for p in data.patients]
    nuisance = []
    for K in (2, 3):
        meta[f"group_K{K}"] = _labels(Z, K)
        for col in ("n_visits", "span", "baseline_tremor", "baseline_pigd"):
            groups = [meta.loc[meta[f"group_K{K}"] == k, col] for k in range(K)]
            nuisance.append({"K": K, "variable": col,
                             "kruskal_p": float(ss.kruskal(*groups).pvalue)})
    pd.DataFrame(nuisance).to_csv(out / "cluster_nuisance_audit.csv", index=False)
    meta.to_csv(out / "patient_features.csv", index=False)
    print("\nWOULD CANDIDATE CLUSTERS SEPARATE ON NUISANCE VARIABLES?")
    print("  " + pd.DataFrame(nuisance).to_string(index=False).replace("\n", "\n  "))

    # ----------------------------------------------------- stability
    rng = np.random.default_rng(0)
    ref = {K: meta[f"group_K{K}"].to_numpy() for K in (2, 3)}
    stab = []
    for K in (2, 3):
        scores = []
        for _ in range(40):
            idx = np.unique(rng.choice(len(Z), len(Z), replace=True))
            scores.append(adjusted_rand_score(ref[K][idx], _labels(Z[idx], K)))
        dropped = np.array([
            TV.blup(fit, data.times[p][:-1] if len(data.times[p]) > 2 else data.times[p],
                    data.responses[p][:-1] if len(data.times[p]) > 2
                    else data.responses[p])[0][2:4]
            for p in data.patients])
        Zd = (dropped - B.mean(0)) / B.std(0)
        stab.append({"K": K, "bootstrap_ARI_mean": float(np.mean(scores)),
                     "bootstrap_ARI_min": float(np.min(scores)),
                     "drop_last_visit_ARI": float(
                         adjusted_rand_score(ref[K], _labels(Zd, K)))})
    pd.DataFrame(stab).to_csv(out / "cluster_stability.csv", index=False)
    print("\nSTABILITY")
    print("  " + pd.DataFrame(stab).round(3).to_string(index=False)
          .replace("\n", "\n  "))

    print(f"\nwrote {out}")
    print("\nNo clustering was performed. See the report for the recommendation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
