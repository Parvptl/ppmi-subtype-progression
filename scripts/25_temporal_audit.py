"""Stage 5: audit of the piecewise-rate (Direction 2) result.

Same 934-patient subscore cohort as scripts 17-19. Stages, each writing its own
CSVs under <run_dir>/<stage>/ and skipped if already complete:

describe     follow-up density: visits, patients and within-period pairs per
             year of follow-up; retention by baseline-severity tertile.
bootstrap    patient-cluster bootstrap (B refits) of the primary model's period
             slopes and contrasts (knots 2, 5; REML; random intercepts, and the
             random-slope control), set beside script 19's Wald intervals.
knots        ML profile of a single knot over a grid, and the bootstrap
             distribution of the profile's argmax. Descriptive only: a knot
             location that maximises a likelihood is not a biological
             transition point, and the bootstrap shows how unstable it is.
structures   alternative temporal mean structures fitted by ML (comparable
             AIC/BIC), their average slopes per period with bootstrap CIs, and
             10-fold patient-level CV of one-step-ahead prediction for each.
timeaxis     follow-up time vs disease duration: does the follow-up
             acceleration survive adjustment for duration (and vice versa)?
attrition    dropout vs current scores, completer-only refit, and early-period
             slopes by eventual follow-up length (pattern-mixture check).

    python scripts/25_temporal_audit.py --out <run_dir>
        [--stages describe,bootstrap,knots,structures,timeaxis,attrition]
        [--boot 1000] [--knot-boot 200] [--struct-boot 300] [--jobs 60]
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
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import heldout as H  # noqa: E402
from ppmi_subtype import jointlmm as J  # noqa: E402
from ppmi_subtype import timevarying as TV  # noqa: E402
from ppmi_subtype import timevarying_ext as TX  # noqa: E402
from ppmi_subtype import trajectory as tj  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

SEED = 42
EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
PRIMARY = (2.0, 5.0)
PERIODS = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0)]
KNOT_GRID = np.round(np.arange(0.5, 10.01, 0.25), 2)
DURATION_COLUMNS = ("duration_yrs", "duration", "DURATION", "disease_duration")

_S: dict = {}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build_frame(cfg):
    panel = pd.read_csv(cfg.path("outputs") / "visit_panel.csv", low_memory=False)
    traj = tj.build_trajectories(panel, cfg)
    tj.check_ordering(traj)
    extra = [c for c in ("visit_date", *DURATION_COLUMNS, "PDTRTMNT", "PDSTATE")
             if c in panel.columns and c not in traj.columns]
    frame = traj.merge(
        panel[["PATNO", "EVENT_ID", *TV.OUTCOMES, *extra]].drop_duplicates(
            ["PATNO", "EVENT_ID"]),
        on=["PATNO", "EVENT_ID"], how="left", validate="one_to_one")
    if frame[list(TV.OUTCOMES)].isna().any().any():
        raise SystemExit("a visit is missing a subscore")
    return panel, frame


# ---------------------------------------------------------------------------
#  Parallel bootstrap refits
# ---------------------------------------------------------------------------
def _init(state):
    _S.update(state)


def _boot_task(args):
    """One bootstrap replicate: refit each named model with patient weights."""
    rep, weights, jobs = args
    data = _S["data"]
    out = []
    for name, basis, rs, reml, periods in jobs:
        f = TX.fit_basis(data, basis, random_slopes=rs, reml=reml, weights=weights)
        for r in TX.period_slopes(f, basis, periods):
            out.append({"rep": rep, "model": name, **r, "converged": f.converged})
    return out


def _knot_task(args):
    rep, weights = args
    data = _S["data"]
    rows = []
    for k in KNOT_GRID:
        f = TX.fit_basis(data, TX.linear_spline((float(k),)), reml=False, weights=weights)
        rows.append({"rep": rep, "knot": float(k), "neg2ll": f.neg2ll,
                     "converged": f.converged})
    lin = TX.fit_basis(data, TX.linear_spline(()), reml=False, weights=weights)
    rows.append({"rep": rep, "knot": np.nan, "neg2ll": lin.neg2ll, "converged": lin.converged})
    return rows


def _simulate_null(data, fit, rng):
    """One dataset from a fitted constant-rate model, on the real visit times."""
    q = len(fit.sigma_b)
    times, resp = {}, {}
    for p in data.patients:
        t = data.times[p]
        X = TV.design(t, ())
        Z = TV.random_design(t, q == 4)
        b = rng.multivariate_normal(np.zeros(q), fit.sigma_b)
        e = rng.multivariate_normal(np.zeros(2), fit.sigma_e, size=len(t)).reshape(-1)
        times[p] = t
        resp[p] = (X @ fit.beta + Z @ b + e).reshape(len(t), 2)
    rows = [{"PATNO": p, "years_from_baseline": tt, "tremor_score": y[0],
             "pigd_score": y[1]} for p in data.patients
            for tt, y in zip(times[p], resp[p])]
    return J.pack(pd.DataFrame(rows))


def _null_knot_task(args):
    """sup over the knot grid of the LR statistic, on one null dataset."""
    rep, null_kind = args
    fit = _S["null_fits"][null_kind]
    rng = np.random.default_rng(SEED + 100_000 + rep + (0 if null_kind == "RI" else 50_000))
    sim = _simulate_null(_S["data"], fit, rng)
    lin = TX.fit_basis(sim, TX.linear_spline(()), reml=False).neg2ll
    best, arg = np.inf, np.nan
    for k in KNOT_GRID:
        v = TX.fit_basis(sim, TX.linear_spline((float(k),)), reml=False).neg2ll
        if v < best:
            best, arg = v, float(k)
    return [{"rep": rep, "null": null_kind, "sup_lr": lin - best, "argmax_knot": arg}]


def run_parallel(fn, tasks, jobs, state):
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=(state,)) as pool:
        out = []
        for res in pool.map(fn, tasks, chunksize=1):
            out.extend(res)
    return out


def summarise_boot(boot: pd.DataFrame, point: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, o, p), g in boot.groupby(["model", "outcome", "period"]):
        pt = point[(point.model == m) & (point.outcome == o) & (point.period == p)].iloc[0]
        lo, hi = np.percentile(g.slope, [2.5, 97.5])
        rows.append({"model": m, "outcome": o, "period": p, "slope": pt.slope,
                     "wald_lo": pt.slope - 1.96 * pt.wald_se,
                     "wald_hi": pt.slope + 1.96 * pt.wald_se,
                     "boot_lo": lo, "boot_hi": hi, "boot_sd": g.slope.std(ddof=1),
                     "wald_se": pt.wald_se,
                     "se_ratio_boot_over_wald": g.slope.std(ddof=1) / pt.wald_se,
                     "n_rep": len(g), "share_converged": g.converged.mean()})
    return pd.DataFrame(rows)


def contrasts_from_boot(boot: pd.DataFrame, point: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, o), g in boot.groupby(["model", "outcome"]):
        wide = g.pivot(index="rep", columns="period", values="slope")
        pw = point[(point.model == m) & (point.outcome == o)].set_index("period").slope
        per = list(pw.index)
        for i in range(len(per)):
            for j in range(i + 1, len(per)):
                d = wide[per[j]] - wide[per[i]]
                lo, hi = np.percentile(d, [2.5, 97.5])
                rows.append({"model": m, "outcome": o, "contrast": f"{per[j]} - {per[i]}",
                             "difference": pw[per[j]] - pw[per[i]], "boot_lo": lo,
                             "boot_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0),
                             "boot_share_positive": float((d > 0).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Stages
# ---------------------------------------------------------------------------
def stage_describe(frame, out: Path):
    f = frame.copy()
    f["year"] = np.floor(f.years_from_baseline).astype(int)
    span = f.groupby("PATNO").years_from_baseline.max()
    base = f.sort_values("years_from_baseline").groupby("PATNO").first()
    tert = pd.qcut(base.pigd_score.rank(method="first"), 3,
                   labels=["low", "mid", "high"])
    rows = []
    for y, g in f.groupby("year"):
        pairs = int(sum(max(len(h) - 1, 0) for _, h in g.groupby("PATNO")))
        rows.append({"year": y, "visits": len(g), "patients": g.PATNO.nunique(),
                     "patients_with_2plus_visits_in_year": int((g.groupby("PATNO").size() >= 2).sum()),
                     "within_year_pairs": pairs,
                     "mean_pigd": g.pigd_score.mean(), "mean_tremor": g.tremor_score.mean(),
                     "share_pigd_zero": (g.pigd_score == 0).mean()})
    pd.DataFrame(rows).to_csv(out / "visits_by_year.csv", index=False)
    ret = []
    for t in np.arange(0, 15):
        for lab in ["low", "mid", "high"]:
            ids = tert[tert == lab].index
            ret.append({"followup_at_least_years": t, "baseline_pigd_tertile": lab,
                        "n": len(ids), "share_retained": float((span[ids] >= t).mean())})
        ret.append({"followup_at_least_years": t, "baseline_pigd_tertile": "all",
                    "n": len(span), "share_retained": float((span >= t).mean())})
    pd.DataFrame(ret).to_csv(out / "retention.csv", index=False)
    late = f[f.years_from_baseline >= 5]
    pd.DataFrame([{
        "patients_any_visit_5y_plus": late.PATNO.nunique(),
        "patients_2plus_visits_5y_plus": int((late.groupby("PATNO").size() >= 2).sum()),
        "patients_3plus_visits_5y_plus": int((late.groupby("PATNO").size() >= 3).sum()),
        "visits_5y_plus": len(late),
        "share_of_all_visits": len(late) / len(f),
        "median_visits_per_late_patient": float(late.groupby("PATNO").size().median()),
    }]).to_csv(out / "late_period_sparsity.csv", index=False)


def stage_bootstrap(data, out, B, jobs):
    models = [("primary_RI", TX.linear_spline(PRIMARY), False, True, PERIODS),
              ("primary_RS", TX.linear_spline(PRIMARY), True, True, PERIODS)]
    point = []
    for name, basis, rs, reml, per in models:
        f = TX.fit_basis(data, basis, random_slopes=rs, reml=reml)
        point += [{"model": name, **r} for r in TX.period_slopes(f, basis, per)]
    point = pd.DataFrame(point)
    point.to_csv(out / "point_estimates.csv", index=False)
    weights = list(TX.bootstrap_weights(data.patients, B, SEED))
    tasks = [(r, w, models) for r, w in enumerate(weights)]
    boot = pd.DataFrame(run_parallel(_boot_task, tasks, jobs, {"data": data}))
    boot.to_csv(out / "bootstrap_replicates.csv", index=False)
    summarise_boot(boot, point).to_csv(out / "period_slopes_boot_vs_wald.csv", index=False)
    contrasts_from_boot(boot, point).to_csv(out / "contrasts_boot.csv", index=False)


def stage_knots(data, out, B, jobs):
    full = pd.DataFrame(_knot_task_full(data))
    full.to_csv(out / "knot_profile_full.csv", index=False)
    weights = list(TX.bootstrap_weights(data.patients, B, SEED + 7))
    rows = pd.DataFrame(run_parallel(_knot_task, list(enumerate(weights)), jobs,
                                     {"data": data}))
    rows.to_csv(out / "knot_profile_bootstrap.csv", index=False)
    am = (rows.dropna(subset=["knot"]).sort_values("neg2ll")
          .groupby("rep").head(1)[["rep", "knot"]])
    lin = rows[rows.knot.isna()].set_index("rep").neg2ll
    best = rows.dropna(subset=["knot"]).groupby("rep").neg2ll.min()
    am["lr_vs_linear"] = (lin - best).reindex(am.rep).to_numpy()
    am.to_csv(out / "knot_argmax_bootstrap.csv", index=False)
    dist = am.knot.value_counts(normalize=True).sort_index().rename("share").reset_index()
    dist.to_csv(out / "knot_argmax_distribution.csv", index=False)

    # Calibrated test of "the rate changes somewhere": the sup-LR over the grid
    # has no chi-square reference, so its null distribution is simulated from
    # the fitted constant-rate model on the real visit times. Two nulls: random
    # intercepts (the primary model's covariance) and random slopes (which can
    # mimic curvature at the population level through who is observed when).
    lin_full = full[full.knot.isna()].neg2ll.iloc[0]
    real_sup = float(lin_full - full.dropna(subset=["knot"]).neg2ll.min())
    real_arg = float(full.dropna(subset=["knot"]).sort_values("neg2ll").knot.iloc[0])
    null_fits = {"RI": TX.fit_basis(data, TX.linear_spline(()), reml=False),
                 "RS": TX.fit_basis(data, TX.linear_spline(()), random_slopes=True,
                                    reml=False)}
    tasks = [(r, k) for k in ("RI", "RS") for r in range(B)]
    null = pd.DataFrame(run_parallel(_null_knot_task, tasks, jobs,
                                     {"data": data, "null_fits": null_fits}))
    null.to_csv(out / "sup_lr_null_replicates.csv", index=False)
    rows = []
    for k, g in null.groupby("null"):
        rows.append({"null": k, "real_sup_lr": real_sup, "real_argmax_knot": real_arg,
                     "n_null": len(g), "null_median": g.sup_lr.median(),
                     "null_q95": g.sup_lr.quantile(0.95),
                     "p_value": (1 + (g.sup_lr >= real_sup).sum()) / (1 + len(g))})
    pd.DataFrame(rows).to_csv(out / "sup_lr_test.csv", index=False)


def _knot_task_full(data):
    _S["data"] = data
    return _knot_task((-1, None))


def structures_list(times: np.ndarray):
    q = lambda p: float(np.quantile(times, p))  # noqa: E731
    lo, hi = 0.0, q(0.95)
    return [
        ("linear", TX.linear_spline(())),
        ("poly2", TX.polynomial(2)),
        ("poly3", TX.polynomial(3)),
        ("pw_3", TX.linear_spline((3.0,))),
        ("pw_2_5", TX.linear_spline((2.0, 5.0))),
        ("pw_1.5_4", TX.linear_spline((1.5, 4.0))),
        ("pw_2.5_6", TX.linear_spline((2.5, 6.0))),
        ("ns_df3", TX.natural_cubic_spline([lo, q(0.5), hi])),
        ("ns_df4", TX.natural_cubic_spline([lo, q(1 / 3), q(2 / 3), hi])),
        ("ns_df5", TX.natural_cubic_spline([lo, q(0.25), q(0.5), q(0.75), hi])),
    ]


def predict_basis(fit: TX.BasisFit, basis, t_hist, Y_hist, t_star):
    """BLUP one-step-ahead predictive mean and 2x2 covariance (as TV.predict)."""
    vinv, _, Z = TV.vinv_logdet(t_hist, fit.sigma_b, fit.sigma_e, fit.random_slopes)
    Bh = basis(np.asarray(t_hist, float))
    p = Bh.shape[1]
    X = np.zeros((2 * len(t_hist), 2 * p))
    X[0::2, :p], X[1::2, p:] = Bh, Bh
    resid = np.asarray(Y_hist).reshape(-1) - X @ fit.beta
    b = fit.sigma_b @ Z.T @ vinv @ resid
    post = fit.sigma_b - fit.sigma_b @ Z.T @ vinv @ Z @ fit.sigma_b
    zs = TV.random_design(np.array([t_star]), fit.random_slopes)
    bs = basis(np.array([t_star]))[0]
    mean = np.array([bs @ fit.beta[:p], bs @ fit.beta[p:]]) + zs @ b
    cov = zs @ post @ zs.T + fit.sigma_e
    return mean, cov


def _cv_task(args):
    fold, name = args
    data, folds = _S["data"], _S["folds"]
    train_p = [p for p in data.patients if folds[p] != fold]
    test_p = [p for p in data.patients if folds[p] == fold]
    train = J.subset(data, train_p)
    t_train = np.concatenate([data.times[p] for p in train_p])
    basis = dict(structures_list(t_train))[name]       # knots from TRAINING times
    f = TX.fit_basis(train, basis, reml=False)
    rows = []
    for p in test_p:
        t, Y = data.times[p], data.responses[p]
        for k in range(1, len(t)):
            m, c = predict_basis(f, basis, t[:k], Y[:k], float(t[k]))
            rows.append({"model": name, "fold": fold, "held_out_patient": p,
                         "target_index": k, "obs_tremor": Y[k, 0], "obs_pigd": Y[k, 1],
                         "pred_tremor": m[0], "pred_pigd": m[1],
                         "sd_tremor": np.sqrt(c[0, 0]), "sd_pigd": np.sqrt(c[1, 1]),
                         "converged": f.converged})
    return rows


def stage_structures(data, out, B, jobs, n_folds=10):
    t_all = np.concatenate([data.times[p] for p in data.patients])
    S = structures_list(t_all)
    fits, point = [], []
    n_pat = data.n_patients
    for name, basis in S:
        f = TX.fit_basis(data, basis, reml=False)
        fits.append({"model": name, "basis": basis.name, "neg2ll_ML": f.neg2ll,
                     "n_params": f.n_params, "AIC": f.aic, "BIC_patients": f.bic(n_pat),
                     "BIC_obs": f.bic(f.n_obs / 2), "converged": f.converged})
        point += [{"model": name, **r} for r in TX.period_slopes(f, basis, PERIODS)]
        grid = np.linspace(0, 12, 49)
        Bg = basis(grid)
        pd.DataFrame({"model": name, "t": grid, "tremor": Bg @ f.beta[:f.p],
                      "pigd": Bg @ f.beta[f.p:]}).to_csv(
            out / f"curve_{name}.csv", index=False)
    fits = pd.DataFrame(fits)
    fits["dAIC_vs_best"] = fits.AIC - fits.AIC.min()
    fits["dBIC_patients_vs_best"] = fits.BIC_patients - fits.BIC_patients.min()
    fits.to_csv(out / "ml_fits.csv", index=False)
    point = pd.DataFrame(point)
    point.to_csv(out / "point_estimates.csv", index=False)

    models = [(n, b, False, False, PERIODS) for n, b in S]
    weights = list(TX.bootstrap_weights(data.patients, B, SEED + 11))
    boot = pd.DataFrame(run_parallel(_boot_task, [(r, w, models) for r, w in
                                                  enumerate(weights)], jobs, {"data": data}))
    boot.to_csv(out / "bootstrap_replicates.csv", index=False)
    summarise_boot(boot, point).to_csv(out / "period_slopes_boot.csv", index=False)
    contrasts_from_boot(boot, point).to_csv(out / "contrasts_boot.csv", index=False)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(data.patients))
    folds = {data.patients[i]: int(r % n_folds) for r, i in enumerate(perm)}
    tasks = [(fo, n) for fo in range(n_folds) for n, _ in S]
    preds = pd.DataFrame(run_parallel(_cv_task, tasks, jobs, {"data": data, "folds": folds}))
    preds.to_csv(out / "cv_predictions.csv", index=False)
    boot_cv = H.SharedBootstrap.create(preds.held_out_patient.unique(), 4000, SEED)
    pw, rows = {}, []
    for name, g in preds.groupby("model"):
        g = g.sort_values(["held_out_patient", "target_index"]).reset_index(drop=True)
        pw[name] = g
        for o in ("tremor", "pigd"):
            s = H.pointwise_continuous(g, o)
            r = {"model": name, "outcome": o, "n": len(g)}
            for m in ("ae", "crps", "log_score"):
                r[m], r[f"{m}_lo"], r[f"{m}_hi"] = boot_cv.mean_ci(s[m], g.held_out_patient)
            r["cov95"] = float(s.cov95.mean())
            rows.append(r)
    pd.DataFrame(rows).to_csv(out / "cv_metrics.csv", index=False)
    prs = []
    ref = pw["linear"]
    for name, g in pw.items():
        if name == "linear":
            continue
        assert np.array_equal(g.held_out_patient, ref.held_out_patient)
        for o in ("tremor", "pigd"):
            a, b = H.pointwise_continuous(g, o), H.pointwise_continuous(ref, o)
            for m in ("ae", "crps", "log_score"):
                est, lo, hi = boot_cv.mean_ci(a[m] - b[m], g.held_out_patient)
                prs.append({"model": name, "reference": "linear", "outcome": o,
                            "metric": m, "difference": est, "lo": lo, "hi": hi,
                            "excludes_zero": bool(lo > 0 or hi < 0)})
    pd.DataFrame(prs).to_csv(out / "cv_paired_vs_linear.csv", index=False)


class TimeAxisBasis:
    """Follow-up linear spline, optionally + disease duration and treatment status.

    Columns: [1, t, (t-k)+ ...] then, if `duration`, baseline duration d and the
    nonlinear natural-spline terms of total duration d + t, then, if `treated`,
    a visit-level on-medication indicator. The follow-up slope-change
    coefficients therefore sit at the same positions (2, 3) in every variant.
    Reads its lookups from the module state `_S` so it pickles to workers, and
    caches the design per (patients, times) block because the likelihood calls
    it on every evaluation.
    """

    uses_patient = True

    def __init__(self, follow_knots, duration, treated=False):
        self.pw = TX.linear_spline(follow_knots)
        self.duration, self.treated = duration, treated
        self.name = (f"followup{tuple(follow_knots)}" + ("+duration_ns" if duration else "")
                     + ("+treated" if treated else ""))
        self._cache = {}

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_cache"] = {}
        return d

    def __call__(self, T, pats):
        key = (T.shape, T.tobytes(), tuple(int(p) for p in pats))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        cols = [self.pw(T)]
        if self.duration:
            d = np.array([_S["dmap"][p] for p in pats])[:, None] * np.ones_like(T)
            cols += [d[..., None], _S["ns"](d + T)[..., 2:]]
        if self.treated:
            tt = np.array([[_S["tmap"].get((int(p), round(float(t), 6)), 0.0) for t in row]
                           for p, row in zip(pats, T)])
            cols.append(tt[..., None])
        out = np.concatenate(cols, axis=-1)
        self._cache[key] = out
        return out


def _timeaxis_boot_task(args):
    rep, weights, specs = args
    rows = []
    for name, basis in specs:
        f = TX.fit_basis(_S["data"], basis, reml=False, weights=weights)
        for o, oname in enumerate(("tremor", "pigd")):
            for j, lab in ((2, "change_at_2y"), (3, "change_at_5y")):
                rows.append({"rep": rep, "model": name, "outcome": oname, "term": lab,
                             "estimate": f.beta[o * f.p + j], "converged": f.converged})
    return rows


def stage_timeaxis(frame, data, out, B, jobs):
    col = next((c for c in DURATION_COLUMNS if c in frame.columns), None)
    if col is None:
        pd.DataFrame([{"status": "not available",
                       "detail": f"none of {DURATION_COLUMNS} in visit_panel.csv"}]).to_csv(
            out / "status.csv", index=False)
        return
    first = frame.sort_values("years_from_baseline").groupby("PATNO").first()
    d0 = pd.to_numeric(first[col], errors="coerce")
    ok = d0.notna()
    keep = [p for p in data.patients if ok.get(p, False)]
    sub = J.subset(data, keep)
    dmap = d0[ok].to_dict()
    # Visit-level treatment status (on PD medication). Missing values are
    # carried forward within patient, then set to untreated.
    tr = frame.sort_values(["PATNO", "years_from_baseline"]).copy()
    if "PDTRTMNT" in tr.columns:
        tr["treated"] = tr.groupby("PATNO").PDTRTMNT.ffill().fillna(0.0).astype(float)
    else:
        tr["treated"] = np.nan
    tmap = {(int(r.PATNO), round(float(r.years_from_baseline), 6)): r.treated
            for r in tr.itertuples()}
    treat_share = (tr.assign(year=np.floor(tr.years_from_baseline).clip(upper=10))
                   .groupby("year").treated.agg(["mean", "size"]).reset_index())
    treat_share.to_csv(out / "treated_share_by_year.csv", index=False)
    D_all = np.concatenate([dmap[p] + data.times[p] for p in keep])
    dk = [float(np.quantile(D_all, q)) for q in (0.0, 1 / 3, 2 / 3, 0.95)]
    ns = TX.natural_cubic_spline(dk)

    specs = [("followup_linear", TimeAxisBasis((), False)),
             ("followup_pw_2_5", TimeAxisBasis(PRIMARY, False)),
             ("followup_linear+duration", TimeAxisBasis((), True)),
             ("followup_pw_2_5+duration", TimeAxisBasis(PRIMARY, True)),
             ("followup_pw_2_5+treated", TimeAxisBasis(PRIMARY, False, True)),
             ("followup_pw_2_5+duration+treated", TimeAxisBasis(PRIMARY, True, True))]
    _S.update({"dmap": dmap, "tmap": tmap, "ns": ns})
    rows, coefs = [], []
    for name, basis in specs:
        f = TX.fit_basis(sub, basis, reml=False)
        rows.append({"model": name, "neg2ll_ML": f.neg2ll, "n_params": f.n_params,
                     "AIC": f.aic, "converged": f.converged, "n_patients": len(keep)})
        if "pw_2_5" in name:
            for o, oname in enumerate(("tremor", "pigd")):
                for j, lab in ((2, "change_at_2y"), (3, "change_at_5y")):
                    idx = o * f.p + j
                    coefs.append({"model": name, "outcome": oname, "term": lab,
                                  "estimate": f.beta[idx],
                                  "wald_se": float(np.sqrt(f.cov_beta[idx, idx]))})
    fits = pd.DataFrame(rows)
    fits.to_csv(out / "ml_fits.csv", index=False)
    coefs = pd.DataFrame(coefs)
    # bootstrap the follow-up slope changes under each adjustment
    weights = list(TX.bootstrap_weights(keep, B, SEED + 13))
    pw_specs = [(n, b_) for n, b_ in specs if "pw_2_5" in n]
    reps = run_parallel(_timeaxis_boot_task, [(r, w, pw_specs) for r, w in enumerate(weights)],
                        jobs, {"data": sub, "dmap": dmap, "tmap": tmap, "ns": ns})
    reps = pd.DataFrame(reps)
    reps.to_csv(out / "bootstrap_replicates.csv", index=False)
    ci = reps.groupby(["model", "outcome", "term"]).estimate.quantile([0.025, 0.975]).unstack()
    ci.columns = ["boot_lo", "boot_hi"]
    coefs.merge(ci.reset_index(), on=["model", "outcome", "term"]).to_csv(
        out / "followup_slope_changes.csv", index=False)


def _logit_fit(X, y, w=None):
    w = np.ones(len(y)) if w is None else w

    def nll(b):
        z = X @ b
        return float(np.sum(w * (np.logaddexp(0, z) - y * z)))

    def grad(b):
        p = 1 / (1 + np.exp(-(X @ b)))
        return X.T @ (w * (p - y))
    return minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS").x


def stage_attrition(frame, data, out, B, jobs):
    f = frame.sort_values(["PATNO", "years_from_baseline"]).copy()
    last = f.groupby("PATNO").years_from_baseline.transform("max")
    f["is_last"] = f.years_from_baseline == last
    # Administrative end of observation is not recorded. A patient whose last
    # labelled visit is the last one in the panel but whose follow-up stopped
    # well short of the cohort's maximum may have dropped out or may have
    # enrolled late; both are treated as "no further labelled visit" here, and
    # the analysis is labelled as describing loss of labelled follow-up rather
    # than dropout in a strict sense.
    f["next_missing"] = f.is_last.astype(float)
    X = np.column_stack([np.ones(len(f)), f.years_from_baseline, f.tremor_score,
                         f.pigd_score])
    y = f.next_missing.to_numpy()
    b = _logit_fit(X, y)
    pats = f.PATNO.to_numpy()
    uniq = np.unique(pats)
    rng = np.random.default_rng(SEED)
    idx_by = {p: np.flatnonzero(pats == p) for p in uniq}
    draws = []
    for _ in range(B):
        pick = rng.choice(uniq, len(uniq))
        ii = np.concatenate([idx_by[p] for p in pick])
        draws.append(_logit_fit(X[ii], y[ii]))
    draws = np.array(draws)
    names = ["intercept", "years_from_baseline", "tremor_score", "pigd_score"]
    pd.DataFrame({"term": names, "log_odds": b,
                  "boot_lo": np.percentile(draws, 2.5, axis=0),
                  "boot_hi": np.percentile(draws, 97.5, axis=0)}).to_csv(
        out / "loss_of_followup_logit.csv", index=False)

    span = f.groupby("PATNO").years_from_baseline.max()
    rows = []
    for thr in (6.0, 8.0):
        keep = [p for p in data.patients if span[p] >= thr]
        sub = J.subset(data, keep)
        basis = TX.linear_spline(PRIMARY)
        fit = TX.fit_basis(sub, basis, reml=True)
        per = [(0, 2), (2, 5), (5, thr)]
        pt = TX.period_slopes(fit, basis, per)
        reps = []
        for r, w in enumerate(TX.bootstrap_weights(keep, B, SEED + int(thr))):
            g = TX.fit_basis(sub, basis, reml=True, weights=w)
            reps += [{"rep": r, **x} for x in TX.period_slopes(g, basis, per)]
        reps = pd.DataFrame(reps)
        for x in pt:
            s = reps[(reps.outcome == x["outcome"]) & (reps.period == x["period"])].slope
            rows.append({"completers_min_followup": thr, "n_patients": len(keep), **x,
                         "boot_lo": s.quantile(0.025), "boot_hi": s.quantile(0.975)})
        wide = reps.pivot_table(index=["rep", "outcome"], columns="period", values="slope")
        for o in ("tremor", "pigd"):
            w_ = wide.xs(o, level="outcome")
            d = w_[f"5-{thr:g}y"] - w_["0-2y"]
            rows.append({"completers_min_followup": thr, "n_patients": len(keep),
                         "outcome": o, "period": f"late - early (5-{thr:g}y minus 0-2y)",
                         "slope": float(next(x["slope"] for x in pt if x["outcome"] == o
                                             and x["period"] == f"5-{thr:g}y")
                                        - next(x["slope"] for x in pt if x["outcome"] == o
                                               and x["period"] == "0-2y")),
                         "boot_lo": d.quantile(0.025), "boot_hi": d.quantile(0.975)})
    pd.DataFrame(rows).to_csv(out / "completer_slopes.csv", index=False)

    groups = {"<3y": span < 3, "3-6y": (span >= 3) & (span < 6), ">=6y": span >= 6}
    pm = []
    for gname, mask in groups.items():
        keep = [p for p in data.patients if mask[p]]
        sub = J.subset(data, keep)
        basis = TX.linear_spline(())
        fit = TX.fit_basis(sub, basis, reml=True)
        early = TX.period_slopes(fit, basis, [(0, 2)])
        sub2 = J.subset(data, keep)
        # early-period slope estimated only from visits in the first 2 years
        rows_e = []
        for p in keep:
            m = data.times[p] <= 2.0
            if m.sum() >= 2:
                rows_e.append(p)
        if len(rows_e) >= 20:
            frame_e = pd.DataFrame([{"PATNO": p, "years_from_baseline": t,
                                     "tremor_score": y[0], "pigd_score": y[1]}
                                    for p in rows_e
                                    for t, y in zip(data.times[p][data.times[p] <= 2.0],
                                                    data.responses[p][data.times[p] <= 2.0])])
            fe = TX.fit_basis(J.pack(frame_e), basis, reml=True)
            for x in TX.period_slopes(fe, basis, [(0, 2)]):
                pm.append({"followup_group": gname, "n_patients": len(rows_e),
                           "estimate_from": "visits within 0-2y only", **x})
        for x in early:
            pm.append({"followup_group": gname, "n_patients": len(keep),
                       "estimate_from": "all visits, linear", **x})
        del sub2
    pd.DataFrame(pm).to_csv(out / "pattern_mixture_early_slopes.csv", index=False)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="describe,bootstrap,knots,structures,timeaxis,attrition")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--knot-boot", type=int, default=200)
    ap.add_argument("--struct-boot", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    panel_md5 = md5(cfg.path("outputs") / "visit_panel.csv")
    if panel_md5 != EXPECTED_PANEL_MD5 and not args.smoke:
        raise SystemExit(f"panel MD5 {panel_md5} != frozen {EXPECTED_PANEL_MD5}")
    panel, frame = build_frame(cfg)
    data = J.pack(frame)
    meta = {"argv": sys.argv, "seed": SEED, "panel_md5": panel_md5, "smoke_test": args.smoke,
            "n_patients": data.n_patients, "n_visits": len(frame),
            "primary_knots": PRIMARY, "periods": PERIODS, "knot_grid": KNOT_GRID.tolist(),
            "boot": args.boot, "knot_boot": args.knot_boot, "struct_boot": args.struct_boot,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(), "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "panel_columns_used": [c for c in frame.columns]}
    (run_dir / "run_config.json").write_text(json.dumps(meta, indent=2, default=str))

    for st in args.stages.split(","):
        out = run_dir / st
        if (out / "_DONE").exists():
            print(f"[{st}] already complete, skipped", flush=True)
            continue
        out.mkdir(exist_ok=True)
        t0 = time.time()
        print(f"[{st}] start", flush=True)
        if st == "describe":
            stage_describe(frame, out)
        elif st == "bootstrap":
            stage_bootstrap(data, out, args.boot, args.jobs)
        elif st == "knots":
            stage_knots(data, out, args.knot_boot, args.jobs)
        elif st == "structures":
            stage_structures(data, out, args.struct_boot, args.jobs)
        elif st == "timeaxis":
            stage_timeaxis(frame, data, out, args.struct_boot, args.jobs)
        elif st == "attrition":
            stage_attrition(frame, data, out, args.struct_boot, args.jobs)
        else:
            raise SystemExit(f"unknown stage {st}")
        (out / "_DONE").write_text(f"{time.time() - t0:.1f}s\n")
        print(f"[{st}] done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
