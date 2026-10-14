"""Audit of the observed PIGD subscore as a measurement (Stage 6b design input).

Descriptive only. No clustering metric, no null simulation. Uses the existing
cohort (934 patients with >= 2 labelled visits, `trajectory.build_trajectories`)
and rebuilds the five PIGD items with the pipeline's own steps
(`loaders` -> `select_medication_state` -> `deduplicate_visits`), so the
"unable to rate" sentinel (101) and out-of-range values are seen before
`clean_items` removes them. Special codes are never treated as scores.

Writes <out>/
    items.csv                 per-item value distribution, sentinel counts
    item_sum_support.csv      S = 5 x PIGD (integer check), distribution 0..20
    smallest_positive.csv     which item carries S = 1 and S = 2
    zero_by_*.csv             zero share by visit index, year, baseline
                              severity, visit count, treatment
    positive_distribution.csv quantiles / skewness of S | S > 0, raw and log
    mean_variance.csv         within-patient mean vs SD (PIGD and tremor)
    residual_by_fitted.csv    RS-LMM conditional residual SD by fitted value
    zero_within_patient.csv   patient-demeaned trend in P(zero) over time
    transitions.csv           P(next zero | current S), overall and by year
    zero_groups.csv           always-zero vs movers vs never-zero, baseline
    movers_reversion.csv      after a first positive score, later zeros
    excluded_visits.csv       visits lost to a missing / unrateable PIGD item

    python scripts/30_pigd_measurement_audit.py --out <dir>
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu, skew, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import loaders, timevarying as TV, trajectory as tj  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype.panel import KEY  # noqa: E402
from ppmi_subtype.subtype import deduplicate_visits, select_medication_state  # noqa: E402

SEED = 42
EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
COVARS = ["age_at_visit", "NHY", "updrs3_score", "updrs2_score", "PDTRTMNT", "LEDD",
          "duration_yrs"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def raw_items(cfg) -> pd.DataFrame:
    """The 16 items after medication-state selection and de-duplication, uncleaned."""
    s = cfg.subtype
    p2, p3 = loaders.load_updrs_part2(cfg), loaders.load_updrs_part3(cfg)
    p3 = select_medication_state(p3, s)
    i2 = [c for c in s.all_items if c in p2.columns]
    i3 = [c for c in s.all_items if c in p3.columns]
    carry = [c for c in (s.pdstate_column, s.treatment_column) if c in p3.columns]
    merged = p2[KEY + i2].merge(p3[KEY + i3 + carry], on=KEY, how="outer")
    return deduplicate_visits(merged, s)


def boot_mean(values, groups, n=4000, seed=SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    g = np.asarray(groups)
    uniq, inv = np.unique(g, return_inverse=True)
    S = np.bincount(inv, weights=v)
    N = np.bincount(inv).astype(float)
    W = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=n)
    draws = (W @ S) / (W @ N)
    return float(v.mean()), *map(float, np.percentile(draws, [2.5, 97.5]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    s = cfg.subtype
    pigd_items = list(s.pigd_items)

    panel_path = cfg.path("outputs") / "visit_panel.csv"
    if md5(panel_path) != EXPECTED_PANEL_MD5:
        raise SystemExit("panel MD5 differs from the frozen panel")
    panel = pd.read_csv(panel_path, low_memory=False)
    traj = tj.build_trajectories(panel, cfg)
    tj.check_ordering(traj)
    keep_cols = ["PATNO", "EVENT_ID", "tremor_score", "pigd_score",
                 *[c for c in COVARS if c in panel.columns]]
    f = traj.merge(panel[keep_cols].drop_duplicates(["PATNO", "EVENT_ID"]),
                   on=["PATNO", "EVENT_ID"], how="left", validate="one_to_one")
    f = f.sort_values(["PATNO", "years_from_baseline"]).reset_index(drop=True)

    # ---------------------------------------------------------------- items
    raw = raw_items(cfg)
    raw_c = raw[KEY + pigd_items].copy()
    cohort_keys = f[["PATNO", "EVENT_ID"]]
    rc = cohort_keys.merge(raw_c, on=["PATNO", "EVENT_ID"], how="left", validate="one_to_one")
    rows = []
    for it in pigd_items:
        v = pd.to_numeric(rc[it], errors="coerce")
        r = {"item": it, "n": len(v), "missing": int(v.isna().sum()),
             "sentinel_101": int((v == 101).sum()),
             "other_out_of_range": int((v.notna() & (v != 101) & ~v.between(0, 4)).sum())}
        for k in range(5):
            r[f"share_{k}"] = float((v == k).mean())
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "items.csv", index=False)
    # Visits of cohort patients that exist in Part III but carry no PIGD score
    allp = panel[panel.PATNO.isin(f.PATNO.unique())]
    exc = allp[allp.pigd_score.isna()][["PATNO", "EVENT_ID"]].merge(
        raw_c, on=["PATNO", "EVENT_ID"], how="left")
    er = {"cohort_patient_visits_without_pigd": len(exc)}
    for it in pigd_items:
        v = pd.to_numeric(exc[it], errors="coerce")
        er[f"{it}_sentinel_101"] = int((v == 101).sum())
        er[f"{it}_missing"] = int(v.isna().sum())
    pd.DataFrame([er]).to_csv(out / "excluded_visits.csv", index=False)

    # ---------------------------------------------------------------- support
    S = np.round(5 * f.pigd_score.to_numpy(), 6)
    f["S"] = S
    T = np.round(11 * f.tremor_score.to_numpy(), 6)
    f["T"] = T
    sup = pd.Series(S).value_counts().sort_index()
    pd.DataFrame({"S": sup.index, "n": sup.values, "share": sup.values / len(S)}).to_csv(
        out / "item_sum_support.csv", index=False)
    meta_support = {"S_integer": bool(np.allclose(S, np.round(S))),
                    "T_integer": bool(np.allclose(T, np.round(T))),
                    "S_max_possible": 20, "S_max_observed": float(S.max()),
                    "share_S_at_upper_bound": float((S == 20).mean()),
                    "T_max_possible": 44, "T_max_observed": float(T.max())}
    items_c = rc[pigd_items].apply(pd.to_numeric, errors="coerce")
    small = []
    for k in (1, 2):
        m = (S == k)
        for it in pigd_items:
            small.append({"S": k, "item": it, "share_nonzero": float((items_c.loc[m, it] > 0).mean())})
    pd.DataFrame(small).to_csv(out / "smallest_positive.csv", index=False)

    # ---------------------------------------------------------------- zeros by
    f["zero"] = (S == 0).astype(float)
    f["year"] = np.floor(f.years_from_baseline).clip(upper=10).astype(int)
    f["visit_index"] = f.groupby("PATNO").cumcount()
    nvis = f.groupby("PATNO").size()
    f["n_visits"] = f.PATNO.map(nvis)
    f["n_visits_band"] = pd.cut(f.n_visits, [1, 2, 3, 4, 7, 99], labels=["2", "3", "4", "5-7", "8+"])
    base = f.groupby("PATNO").first()
    for name, col in (("baseline_updrs3", "updrs3_score"), ("baseline_tremor", "tremor_score"),
                      ("baseline_age", "age_at_visit")):
        if col in base:
            q = pd.qcut(base[col].rank(method="first"), 3, labels=["low", "mid", "high"])
            f[name + "_tertile"] = f.PATNO.map(q)
    if "NHY" in base:
        f["baseline_hy"] = f.PATNO.map(base["NHY"])
    for by in ("visit_index", "year", "n_visits_band", "baseline_updrs3_tertile",
               "baseline_tremor_tertile", "baseline_age_tertile", "baseline_hy", "PDTRTMNT"):
        if by not in f:
            continue
        g = f.groupby(by, observed=True).agg(n=("zero", "size"), patients=("PATNO", "nunique"),
                                             zero_share=("zero", "mean")).reset_index()
        g.to_csv(out / f"zero_by_{by}.csv", index=False)

    # ---------------------------------------------------------------- positive part
    pos = S[S > 0]
    q = np.quantile(pos, [0.05, 0.25, 0.5, 0.75, 0.95])
    prow = [{"scale": "S (0..20)", "n": len(pos), "mean": pos.mean(), "sd": pos.std(ddof=1),
             "skew": skew(pos), **{f"q{int(p * 100)}": v for p, v in
                                   zip([0.05, .25, .5, .75, .95], q)}},
            {"scale": "log S", "n": len(pos), "mean": np.log(pos).mean(),
             "sd": np.log(pos).std(ddof=1), "skew": skew(np.log(pos))}]
    for y, g in f[f.S > 0].groupby("year"):
        prow.append({"scale": f"S, year {y}", "n": len(g), "mean": g.S.mean(), "sd": g.S.std(ddof=1),
                     "skew": skew(g.S), "q50": g.S.median()})
    pd.DataFrame(prow).to_csv(out / "positive_distribution.csv", index=False)

    # ---------------------------------------------------------------- mean-variance
    mv = []
    for name, col in (("pigd", "pigd_score"), ("tremor", "tremor_score")):
        g = f[f.n_visits >= 4].groupby("PATNO")[col].agg(["mean", "std"])
        bins = pd.qcut(g["mean"].rank(method="first"), 8, labels=False)
        for b, gg in g.groupby(bins):
            mv.append({"outcome": name, "mean_bin": int(b), "n_patients": len(gg),
                       "patient_mean": gg["mean"].mean(), "within_sd_median": gg["std"].median(),
                       "within_sd_mean": gg["std"].mean()})
        rho = spearmanr(g["mean"], g["std"])
        mv.append({"outcome": name, "mean_bin": "spearman(mean, sd)",
                   "n_patients": len(g), "patient_mean": float(rho.statistic),
                   "within_sd_median": float(rho.pvalue)})
    pd.DataFrame(mv).to_csv(out / "mean_variance.csv", index=False)

    data = TV.pack(f)
    fit = TV.fit(data, knots=(), random_slopes=True)
    res = []
    for p in data.patients:
        t, Y = data.times[p], data.responses[p]
        b, _ = TV.blup(fit, t, Y)
        fitted = (TV.design(t, ()) @ fit.beta + TV.random_design(t, True) @ b).reshape(-1, 2)
        for j in range(len(t)):
            res.append({"PATNO": p, "fitted_tremor": fitted[j, 0], "fitted_pigd": fitted[j, 1],
                        "resid_tremor": Y[j, 0] - fitted[j, 0], "resid_pigd": Y[j, 1] - fitted[j, 1],
                        "obs_pigd": Y[j, 1]})
    res = pd.DataFrame(res)
    rb = []
    for name in ("pigd", "tremor"):
        bins = pd.qcut(res[f"fitted_{name}"], 10, duplicates="drop")
        for b, g in res.groupby(bins, observed=True):
            rb.append({"outcome": name, "fitted_lo": b.left, "fitted_hi": b.right, "n": len(g),
                       "fitted_mean": g[f"fitted_{name}"].mean(),
                       "resid_sd": g[f"resid_{name}"].std(ddof=1),
                       "resid_mean": g[f"resid_{name}"].mean(),
                       "share_obs_zero": float((g.obs_pigd == 0).mean()) if name == "pigd" else np.nan})
    pd.DataFrame(rb).to_csv(out / "residual_by_fitted.csv", index=False)

    # ---------------------------------------------------------------- zero over time
    f["zero_dm"] = f.zero - f.groupby("PATNO").zero.transform("mean")
    f["t_dm"] = f.years_from_baseline - f.groupby("PATNO").years_from_baseline.transform("mean")
    pats = f.PATNO.to_numpy()
    uniq, inv = np.unique(pats, return_inverse=True)
    sxy = np.bincount(inv, weights=f.zero_dm * f.t_dm)
    sxx = np.bincount(inv, weights=f.t_dm ** 2)
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=4000)
    slope_draws = (W @ sxy) / (W @ sxx)
    zrows = [{"quantity": "within-patient change in P(zero) per year (patient-demeaned LPM)",
              "estimate": float(sxy.sum() / sxx.sum()),
              "lo": float(np.percentile(slope_draws, 2.5)),
              "hi": float(np.percentile(slope_draws, 97.5)),
              "n_patients_with_variation": int((f.groupby("PATNO").zero.std() > 0).sum())}]
    pd.DataFrame(zrows).to_csv(out / "zero_within_patient.csv", index=False)

    # ---------------------------------------------------------------- transitions
    f["S_next"] = f.groupby("PATNO").S.shift(-1)
    tr = f.dropna(subset=["S_next"]).copy()
    tr["S_band"] = pd.cut(tr.S, [-0.5, 0.5, 1.5, 2.5, 4.5, 20.5], labels=["0", "1", "2", "3-4", "5+"])
    trows = []
    for band, g in tr.groupby("S_band", observed=True):
        est, lo, hi = boot_mean((g.S_next == 0).astype(float), g.PATNO)
        trows.append({"current_S": str(band), "n_pairs": len(g), "P_next_zero": est,
                      "lo": lo, "hi": hi})
    for y, g in tr.groupby("year"):
        for cur, gg in ((("zero", g[g.S == 0])), ("positive", g[g.S > 0])):
            if len(gg):
                trows.append({"current_S": f"year {y}, current {cur}", "n_pairs": len(gg),
                              "P_next_zero": float((gg.S_next == 0).mean())})
    pd.DataFrame(trows).to_csv(out / "transitions.csv", index=False)

    # ---------------------------------------------------------------- zero groups
    zstat = f.groupby("PATNO").zero.agg(["mean", "size"])
    grp = np.where(zstat["mean"] == 1, "always_zero",
                   np.where(zstat["mean"] == 0, "never_zero", "mover"))
    grp = pd.Series(grp, index=zstat.index)
    base = f.groupby("PATNO").first()
    span = f.groupby("PATNO").years_from_baseline.max()
    gr = []
    for gname in ("always_zero", "mover", "never_zero"):
        ids = grp[grp == gname].index
        r = {"group": gname, "n_patients": len(ids), "median_visits": float(nvis[ids].median()),
             "median_followup_years": float(span[ids].median())}
        for c in ("tremor_score", "pigd_score", "age_at_visit", "updrs3_score", "updrs2_score",
                  "NHY", "duration_yrs", "PDTRTMNT"):
            if c in base:
                r[f"baseline_{c}_median"] = float(base.loc[ids, c].median())
        gr.append(r)
    for c in ("tremor_score", "age_at_visit", "updrs3_score", "NHY", "duration_yrs"):
        if c in base:
            a = base.loc[grp[grp == "always_zero"].index, c].dropna()
            b = base.loc[grp[grp == "mover"].index, c].dropna()
            gr.append({"group": f"always_zero vs mover: baseline {c}",
                       "n_patients": len(a) + len(b),
                       "median_visits": float(mannwhitneyu(a, b).pvalue)})
    pd.DataFrame(gr).to_csv(out / "zero_groups.csv", index=False)

    mrows = []
    for p, g in f[f.PATNO.isin(grp[grp == "mover"].index)].groupby("PATNO"):
        z = g.zero.to_numpy()
        first_pos = int(np.argmax(z == 0)) if (z == 0).any() else None
        if first_pos is None or first_pos == len(z) - 1:
            continue
        later = z[first_pos + 1:]
        mrows.append({"PATNO": p, "later_visits": len(later), "later_zero_share": later.mean()})
    mr = pd.DataFrame(mrows)
    pd.DataFrame([{"movers_with_visits_after_first_positive": len(mr),
                   "share_with_any_later_zero": float((mr.later_zero_share > 0).mean()),
                   "mean_later_zero_share": float(mr.later_zero_share.mean())}]).to_csv(
        out / "movers_reversion.csv", index=False)

    meta = {"argv": sys.argv, "panel_md5": EXPECTED_PANEL_MD5, "n_patients": int(f.PATNO.nunique()),
            "n_visits": len(f), **meta_support,
            "rs_lmm_sigma_e": fit.sigma_e.tolist(), "rs_lmm_converged": fit.converged,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(), "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    (out / "run_config.json").write_text(json.dumps(meta, indent=2, default=float))
    print(json.dumps(meta, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
