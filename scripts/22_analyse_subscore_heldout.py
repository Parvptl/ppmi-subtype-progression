"""Stage 3 analysis: score the saved held-out predictions of script 21.

Reads only `<run_dir>/folds/*/*.csv` and `*_fit.json`; refits nothing. Writes
`<run_dir>/analysis/` and then renders `<run_dir>/analysis/REPORT.md` from the
CSVs it has just written (re-runnable on its own with `--report-only`).

Every interval is a patient-cluster bootstrap (4,000 replicates, seed 42) in
which every model and every paired comparison uses the SAME resampled patients.

    python scripts/22_analyse_subscore_heldout.py <run_dir> [--report-only]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kstest, multivariate_normal  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import heldout as H  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402

N_BOOT, SEED = 4000, 42
OUTCOMES = ("tremor", "pigd")
CODES = {"TD": 0, "Indeterminate": 1, "PIGD": 2}
KERNELS = ("K0", "K1", "K2", "K3")


# ---------------------------------------------------------------------------
#  Load
# ---------------------------------------------------------------------------
def load_run(run_dir: Path):
    meta = json.loads((run_dir / "run_config.json").read_text())
    frames, fits = [], []
    for arm in meta["arms"]:
        d = run_dir / "folds" / arm
        files = sorted(p for p in d.glob("*.csv"))
        frames += [pd.read_csv(p) for p in files]
        fits += [json.loads(p.read_text()) for p in sorted(d.glob("*_fit.json"))]
    preds = pd.concat(frames, ignore_index=True)
    for o in OUTCOMES:
        preds[f"sd_{o}"] = np.sqrt(preds[f"var_{o}"])
    preds["rho"] = preds.cov_tremor_pigd / (preds.sd_tremor * preds.sd_pigd)
    return meta, preds, pd.DataFrame(fits)


# ---------------------------------------------------------------------------
#  Leakage and consistency battery
# ---------------------------------------------------------------------------
def checks(preds: pd.DataFrame, fits: pd.DataFrame, meta: dict, seqs) -> pd.DataFrame:
    rows = []

    def add(name, ok, detail=""):
        rows.append({"check": name, "pass": bool(ok), "detail": str(detail)})

    arms = meta["arms"]
    n_pat = meta["n_patients_scored"]
    add("every arm x patient task completed",
        all(preds[preds.model == a].held_out_patient.nunique() == n_pat for a in arms),
        {a: int(preds[preds.model == a].held_out_patient.nunique()) for a in arms})
    add(f"every arm scores {meta['n_points_per_arm']} points",
        all((preds.model == a).sum() == meta["n_points_per_arm"] for a in arms))
    add("training set is N-1 patients in every fold",
        (preds.n_train_patients == meta["n_patients_cohort"] - 1).all())
    add("history strictly precedes target", (preds.t_last_history < preds.t_target).all())
    add("history length equals target index", (preds.history_length == preds.target_index).all())
    key = ["held_out_patient", "target_index", "t_target", "obs_tremor", "obs_pigd"]
    ref = preds[preds.model == arms[0]][key].sort_values(key[:2]).to_numpy()
    add("all arms scored on identical points and observations", all(
        np.array_equal(preds[preds.model == a][key].sort_values(key[:2]).to_numpy(), ref)
        for a in arms))
    if seqs is not None:
        bad = 0
        for p, g in preds[preds.model == arms[0]].groupby("held_out_patient"):
            s = seqs[int(p)]
            g = g.sort_values("target_index")
            bad += int(not (np.allclose(g.t_target, s["t"][1:])
                            and np.allclose(g[["obs_tremor", "obs_pigd"]], s["y"][1:])))
        add("targets match the panel trajectory", bad == 0, f"{bad} patients mismatched")
        if "persistence" in arms:
            g = preds[preds.model == "persistence"]
            prev = np.array([seqs[int(p)]["y"][k - 1] for p, k in
                             zip(g.held_out_patient, g.target_index)])
            add("persistence equals the immediately preceding visit (history alignment)",
                np.allclose(g[["pred_tremor", "pred_pigd"]].to_numpy(), prev))
    det = preds.var_tremor * preds.var_pigd - preds.cov_tremor_pigd ** 2
    add("every predictive covariance is positive definite",
        ((preds.var_tremor > 0) & (preds.var_pigd > 0) & (det > 0)).all(),
        f"{int((det <= 0).sum())} non-PD")
    add("PIGD == 0 targets retained", (preds.obs_pigd == 0).any(),
        int((preds[preds.model == arms[0]].obs_pigd == 0).sum()))
    add("fit diagnostics saved for every task", len(fits) == len(arms) * n_pat, len(fits))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Continuous metrics
# ---------------------------------------------------------------------------
def bivariate_log_score(g: pd.DataFrame) -> np.ndarray:
    d1 = g.obs_tremor - g.pred_tremor
    d2 = g.obs_pigd - g.pred_pigd
    a, b, c = g.var_tremor, g.cov_tremor_pigd, g.var_pigd
    det = a * c - b ** 2
    quad = (c * d1 ** 2 - 2 * b * d1 * d2 + a * d2 ** 2) / det
    return (-np.log(2 * np.pi) - 0.5 * np.log(det) - 0.5 * quad).to_numpy()


def pointwise(preds: pd.DataFrame) -> dict:
    """{arm: DataFrame of per-point scores, indexed by (patient, target_index)}."""
    out = {}
    for arm, g in preds.groupby("model", sort=False):
        g = g.sort_values(["held_out_patient", "target_index"])
        parts = []
        for o in OUTCOMES:
            pw = H.pointwise_continuous(g, o)
            pw.columns = [f"{o}_{c}" for c in pw.columns]
            parts.append(pw)
        df = pd.concat(parts, axis=1)
        df["joint_log_score"] = bivariate_log_score(g)
        df["held_out_patient"] = g.held_out_patient.to_numpy()
        df["target_index"] = g.target_index.to_numpy()
        out[arm] = df.reset_index(drop=True)
    return out


def continuous_metrics(pw: dict, boot) -> pd.DataFrame:
    rows = []
    for arm, df in pw.items():
        pat = df.held_out_patient.to_numpy()
        for o in OUTCOMES:
            r = {"model": arm, "outcome": o, "n": len(df),
                 "n_patients": int(df.held_out_patient.nunique())}
            for m, col in [("MAE", "ae"), ("MSE", "se"), ("CRPS", "crps"),
                           ("log_score", "log_score")]:
                r[m], r[f"{m}_lo"], r[f"{m}_hi"] = boot.mean_ci(df[f"{o}_{col}"], pat)
            r["RMSE"] = float(np.sqrt(r["MSE"]))
            r["bias"] = float(df[f"{o}_bias"].mean())
            for lv in H.LEVELS:
                k = f"cov{int(round(lv * 100))}"
                r[k] = float(df[f"{o}_{k}"].mean())
            u = df[f"{o}_pit"].to_numpy()
            r["pit_ks_stat"] = float(kstest(u, "uniform").statistic)
            hist = np.histogram(u, bins=10, range=(0, 1))[0] / len(u)
            r["pit_max_abs_dev_decile"] = float(np.max(np.abs(hist - 0.1)))
            r["mean_negative_mass"] = float(df[f"{o}_neg_mass"].mean())
            rows.append(r)
        r = {"model": arm, "outcome": "joint", "n": len(df)}
        r["log_score"], r["log_score_lo"], r["log_score_hi"] = boot.mean_ci(
            df["joint_log_score"], pat)
        rows.append(r)
    return pd.DataFrame(rows)


def pit_histograms(pw: dict) -> pd.DataFrame:
    rows = []
    for arm, df in pw.items():
        for o in OUTCOMES:
            h = np.histogram(df[f"{o}_pit"], bins=10, range=(0, 1))[0]
            rows += [{"model": arm, "outcome": o, "bin": b, "share": h[b] / h.sum()}
                     for b in range(10)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Paired comparisons (pre-specified)
# ---------------------------------------------------------------------------
def comparison_plan(arms) -> list[tuple[str, str, str]]:
    """(family, model, reference). Fixed before any result was seen."""
    a = set(arms)
    plan = []
    for k in KERNELS:
        if {f"gp_joint_{k}_full", f"gp_joint_{k}_diag"} <= a:
            plan.append(("1_nested_cross_output", f"gp_joint_{k}_full", f"gp_joint_{k}_diag"))
    for k in KERNELS:
        if {f"gp_joint_{k}_diag", f"gp_indep_{k}"} <= a:
            plan.append(("2_shared_lengthscale", f"gp_joint_{k}_diag", f"gp_indep_{k}"))
        if {f"gp_joint_{k}_full", f"gp_indep_{k}"} <= a:
            plan.append(("3_joint_vs_independent", f"gp_joint_{k}_full", f"gp_indep_{k}"))
    for k in KERNELS[1:]:
        for fam in ("gp_indep_{}", "gp_joint_{}_full"):
            m, r = fam.format(k), fam.format("K0")
            if {m, r} <= a:
                plan.append(("4_temporal_flexibility", m, r))
    for ref in ("persistence", "lmm_ri"):
        if ref in a:
            for m in arms:
                if m != ref and (ref == "persistence" or m.startswith("gp_")):
                    plan.append((f"5_vs_{ref}", m, ref))
    return plan


def paired(pw: dict, plan, boot) -> pd.DataFrame:
    rows = []
    metrics = [(o, c) for o in OUTCOMES for c in ("ae", "se", "crps", "log_score")]
    metrics.append(("joint", "log_score"))
    for fam, m, r in plan:
        A, B = pw[m], pw[r]
        if not (np.array_equal(A.held_out_patient, B.held_out_patient)
                and np.array_equal(A.target_index, B.target_index)):
            raise RuntimeError(f"{m} and {r} are not aligned")
        pat = A.held_out_patient.to_numpy()
        for o, c in metrics:
            col = "joint_log_score" if o == "joint" else f"{o}_{c}"
            diff = A[col].to_numpy() - B[col].to_numpy()
            est, lo, hi = boot.mean_ci(diff, pat)
            rows.append({"family": fam, "model": m, "reference": r, "outcome": o,
                         "metric": c, "difference": est, "lo": lo, "hi": hi,
                         "excludes_zero": bool((lo > 0) or (hi < 0)),
                         "better": ("model" if (lo > 0 if c == "log_score" else hi < 0)
                                    else "reference" if (hi < 0 if c == "log_score" else lo > 0)
                                    else "neither"),
                         "n": len(diff)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Subtype
# ---------------------------------------------------------------------------
def train_prior_by_patient(g: pd.DataFrame) -> np.ndarray:
    """Leave-the-patient-out label prevalence among scored targets, per row.

    The prior-corrected decision rule needs class prevalence from the TRAINING
    patients only; with LOPO folds that is the total count minus the held-out
    patient's own targets.
    """
    actual = g.obs_subtype.map(CODES).to_numpy()
    tot = np.bincount(actual, minlength=3).astype(float)
    per = {p: np.bincount(actual[(g.held_out_patient == p).to_numpy()], minlength=3)
           for p in g.held_out_patient.unique()}
    prior = np.stack([tot - per[p] for p in g.held_out_patient])
    return prior / prior.sum(axis=1, keepdims=True)


def subtype_analysis(preds, cfg, boot, hmm_frame=None):
    td, pg = cfg.subtype.td_threshold, cfg.subtype.pigd_threshold
    summary, per_point, rel_rows = [], {}, []
    for arm, g in preds.groupby("model", sort=False):
        g = g.sort_values(["held_out_patient", "target_index"]).reset_index(drop=True)
        probs, clipped = H.subtype_probabilities(
            g.pred_tremor, g.pred_pigd, g.sd_tremor, g.sd_pigd, g.rho.fillna(0), td, pg,
            seed=SEED)
        actual = g.obs_subtype.map(CODES).to_numpy()
        pat = g.held_out_patient.to_numpy()
        prior = train_prior_by_patient(g)
        rules = {"point": H.point_class(g.pred_tremor, g.pred_pigd, td, pg),
                 "prob_argmax": probs.argmax(axis=1),
                 "prob_prior_corrected": (probs / prior).argmax(axis=1)}
        for rule, pred in rules.items():
            s = H.categorical_summary(probs, actual, pred, pat, boot)
            s.update({"model": arm, "rule": rule,
                      "mean_clipped_draw_share": float(clipped.mean()),
                      **H.expected_calibration_error(probs, actual)})
            summary.append(s)
        rt = H.reliability_table(probs, actual)
        rt["model"] = arm
        rel_rows.append(rt)
        per_point[arm] = pd.DataFrame({
            "held_out_patient": pat, "target_index": g.target_index,
            "event_target": g.event_target, "actual": actual,
            **{f"p_{c}": probs[:, k] for c, k in CODES.items()},
            **{f"pred_{r}": v for r, v in rules.items()}})
    if hmm_frame is not None:
        g = hmm_frame
        probs = g[["p_TD", "p_Indeterminate", "p_PIGD"]].to_numpy()
        actual = g.actual.to_numpy()
        pat = g.held_out_patient.to_numpy()
        prior = train_prior_by_patient(g.assign(obs_subtype=g.actual.map(
            {v: k for k, v in CODES.items()})))
        for rule, pred in {"prob_argmax": probs.argmax(axis=1),
                           "prob_prior_corrected": (probs / prior).argmax(axis=1)}.items():
            s = H.categorical_summary(probs, actual, pred, pat, boot)
            s.update({"model": "hmm_loo_09", "rule": rule,
                      **H.expected_calibration_error(probs, actual)})
            summary.append(s)
        per_point["hmm_loo_09"] = pd.DataFrame({
            "held_out_patient": pat, "target_index": g.target_index,
            "event_target": g.event_target, "actual": actual,
            **{f"p_{c}": probs[:, k] for c, k in CODES.items()},
            "pred_prob_argmax": probs.argmax(axis=1),
            "pred_prob_prior_corrected": (probs / prior).argmax(axis=1)})
    return pd.DataFrame(summary), per_point, pd.concat(rel_rows)


def subtype_paired(per_point: dict, boot, reference: str) -> pd.DataFrame:
    if reference not in per_point:
        return pd.DataFrame()
    R = per_point[reference]
    rows = []
    for arm, A in per_point.items():
        if arm == reference:
            continue
        if not (np.array_equal(A.held_out_patient, R.held_out_patient)
                and np.array_equal(A.target_index, R.target_index)):
            raise RuntimeError(f"{arm} not aligned with {reference}")
        pat = A.held_out_patient.to_numpy()
        act = A.actual.to_numpy()

        def ll(F):
            p = F[["p_TD", "p_Indeterminate", "p_PIGD"]].to_numpy()
            return -np.log(np.clip(p[np.arange(len(act)), act], 1e-12, None))
        est, lo, hi = boot.mean_ci(ll(A) - ll(R), pat)
        rows.append({"model": arm, "reference": reference, "metric": "log_loss",
                     "difference": est, "lo": lo, "hi": hi})
        for rule in ("prob_argmax", "prob_prior_corrected"):
            if f"pred_{rule}" in A and f"pred_{rule}" in R:
                d = ((A[f"pred_{rule}"] == act).astype(float)
                     - (R[f"pred_{rule}"] == act).astype(float)).to_numpy()
                est, lo, hi = boot.mean_ci(d, pat)
                rows.append({"model": arm, "reference": reference,
                             "metric": f"accuracy_{rule}", "difference": est,
                             "lo": lo, "hi": hi})
    out = pd.DataFrame(rows)
    out["excludes_zero"] = (out.lo > 0) | (out.hi < 0)
    return out


def load_hmm(cfg, preds) -> tuple[pd.DataFrame | None, str]:
    path = cfg.path("outputs") / "loo" / "loo_predictions.csv"
    if not path.exists():
        return None, f"absent: {path}"
    h = pd.read_csv(path)
    h = h[h.model == "HMM"].copy()
    h["held_out_patient"] = h.PATNO.astype(int)
    h["target_index"] = h.visit_index.astype(int) + 1
    h["actual"] = h.actual.map(CODES)
    ref = (preds[preds.model == preds.model.iloc[0]]
           [["held_out_patient", "target_index", "event_target", "obs_subtype"]])
    m = ref.merge(h, on=["held_out_patient", "target_index"], how="left",
                  validate="one_to_one")
    ok = (m.actual.notna().all()
          and (m.target_event.astype(str) == m.event_target.astype(str)).all()
          and (m.obs_subtype.map(CODES) == m.actual).all())
    if not ok:
        return None, "join failed validation (events or labels disagree)"
    m = m.sort_values(["held_out_patient", "target_index"]).reset_index(drop=True)
    m["actual"] = m.actual.astype(int)
    return m, f"joined {len(m)}/{len(ref)} on (patient, target index); events and labels agree"


# ---------------------------------------------------------------------------
#  Fit diagnostics and reproduction of the historical run
# ---------------------------------------------------------------------------
def fit_diagnostics(fits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, g in fits.groupby("model"):
        r = {"model": arm, "n_fits": len(g), "converged_share": float(g.converged.mean()),
             "median_fit_seconds": float(g.fit_seconds.median()),
             "max_fit_seconds": float(g.fit_seconds.max())}
        for col in ("message", "tremor_message", "pigd_message"):
            if col in g:
                r[f"{col}_counts"] = json.dumps(g[col].value_counts().to_dict())
        for col in ("at_bound", "tremor_at_bound", "pigd_at_bound"):
            if col in g:
                r[f"{col}_counts"] = json.dumps(g[col].fillna("").value_counts().to_dict())
        if "grad_norm" in g:
            r["median_grad_norm"] = float(g.grad_norm.median())
        rows.append(r)
    return pd.DataFrame(rows)


def hyperparameter_stability(fits: pd.DataFrame) -> pd.DataFrame:
    num = fits.select_dtypes("number").drop(columns=["held_out_patient", "n_train_patients",
                                                     "fit_seconds", "task_seconds"],
                                            errors="ignore")
    num["model"] = fits.model
    out = num.groupby("model").agg(["min", "median", "max"]).T.reset_index()
    out.columns = ["parameter", "stat", *out.columns[2:]]
    return out.dropna(how="all", axis=0)


def reproduction(preds, cfg) -> pd.DataFrame:
    """Compare against the historical Windows outputs where they exist."""
    rows = []
    pairs = [("joint_subscore_gp/predictions.csv",
              {"gp_joint_K0_full": "gp_joint_K0_full", "gp_joint_K1_full": "gp_joint_K1_full",
               "gp_joint_K1_diag": "gp_joint_K1_diag", "gp_indep_K1": "gp_indep_K1",
               "persistence": "persistence", "patient_mean": "patient_mean"}),
             ("joint_subscore/predictions.csv",
              {"lmm_ri": "joint_lmm", "persistence": "persistence"})]
    for rel, mapping in pairs:
        path = cfg.path("outputs") / rel
        if not path.exists():
            rows.append({"source": rel, "status": "absent"})
            continue
        old = pd.read_csv(path)
        for new_name, old_name in mapping.items():
            a = preds[preds.model == new_name]
            b = old[old.model == old_name]
            if a.empty or b.empty:
                continue
            m = a.merge(b, on=["held_out_patient", "target_index"], suffixes=("", "_old"))
            r = {"source": rel, "model": new_name, "historical_model": old_name,
                 "n_matched": len(m), "n_new": len(a), "n_old": len(b)}
            for o in OUTCOMES:
                r[f"max_abs_diff_mean_{o}"] = float((m[f"pred_{o}"] - m[f"pred_{o}_old"]).abs().max())
                if f"sd_{o}_old" in m:
                    r[f"max_abs_diff_sd_{o}"] = float((m[f"sd_{o}"] - m[f"sd_{o}_old"]).abs().max())
            rows.append(r)
    return pd.DataFrame(rows)


def pigd_zero(preds) -> pd.DataFrame:
    z = preds[preds.obs_pigd == 0]
    rows = []
    for arm, g in z.groupby("model"):
        rows.append({"model": arm, "n": len(g), "pigd_MAE": float(g.pred_pigd.abs().mean()),
                     "pigd_pred_mean": float(g.pred_pigd.mean()),
                     "frac_pred_below_0.1": float((g.pred_pigd < 0.1).mean()),
                     "mean_negative_mass": float(H.negative_mass(g.pred_pigd, g.sd_pigd).mean())})
    return pd.DataFrame(rows).sort_values("pigd_MAE")


# ---------------------------------------------------------------------------
#  Report, rendered only from the saved CSVs
# ---------------------------------------------------------------------------
def _md(df, cols=None, fmt="{:.4f}"):
    df = df if cols is None else df[cols]
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "\n".join("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating))
                                        else str(v) for v in r) + " |"
                     for r in df.itertuples(index=False))
    return head + body + "\n"


def render_report(an: Path) -> None:
    rd = lambda n: pd.read_csv(an / n)  # noqa: E731
    meta = json.loads((an.parent / "run_config.json").read_text())
    chk, cont, par = rd("checks.csv"), rd("continuous_metrics.csv"), rd("paired.csv")
    sub, diag = rd("subtype_metrics.csv"), rd("fit_diagnostics.csv")
    L = [f"# Stage 3 — subscore held-out comparison (generated)\n",
         f"Run `{an.parent.name}` · git `{meta.get('git_commit')}` · panel MD5 "
         f"`{meta['panel_md5']}` · {meta['n_patients_scored']} held-out patients · "
         f"{meta['n_points_per_arm']} points per arm · fold restarts "
         f"{meta['fold_restarts']} · seed {meta['seed']}"
         + (" · **SMOKE TEST — synthetic data**" if meta.get("smoke_test") else "") + "\n",
         "Every number below is read from the CSVs in this directory. Intervals: "
         "patient-cluster bootstrap, 4,000 replicates, the same resampled patients "
         "for every model and comparison.\n",
         "## Leakage and consistency checks\n", _md(chk),
         "## Continuous outcomes\n"]
    for o in OUTCOMES:
        c = cont[cont.outcome == o].sort_values("CRPS")
        L += [f"### {o}\n", _md(c, ["model", "MAE", "MAE_lo", "MAE_hi", "RMSE", "CRPS",
                                    "log_score", "cov50", "cov80", "cov95",
                                    "pit_ks_stat", "mean_negative_mass"])]
    j = cont[cont.outcome == "joint"].sort_values("log_score", ascending=False)
    L += ["### Joint (bivariate) log predictive density\n",
          "The metric on which a cross-output covariance can help at all.\n",
          _md(j, ["model", "log_score", "log_score_lo", "log_score_hi"])]
    L += ["## Pre-specified paired comparisons\n",
          "`difference` = model − reference. For `ae`/`se`/`crps` negative favours the "
          "model; for `log_score` positive favours the model.\n"]
    L += ["Note: K0 has no lengthscale, so `gp_joint_K0_diag` and `gp_indep_K0` are the "
          "same model and their `2_shared_lengthscale` comparison is zero by construction "
          "(a built-in check, not a finding).\n"]
    for fam, g in par.groupby("family"):
        L += [f"### {fam}\n", _md(g, ["model", "reference", "outcome", "metric",
                                      "difference", "lo", "hi", "better"], fmt="{:+.5f}")]
    L += ["## Derived subtype\n",
          "`point`: rule applied to predictive means. `prob_argmax`: most probable class "
          "under the predictive. `prob_prior_corrected`: argmax of p/prior with training "
          "prevalence (the balanced-accuracy Bayes rule). Log loss and Brier score the "
          "probabilities and do not depend on the rule.\n",
          _md(sub, ["model", "rule", "accuracy", "balanced_accuracy", "recall_TD",
                    "recall_Indeterminate", "recall_PIGD", "n_predicted_Indeterminate",
                    "log_loss", "brier", "ece_Indeterminate"])]
    if (an / "subtype_paired_vs_hmm.csv").exists():
        L += ["### Paired against the HMM (script 09 LOO predictions)\n",
              _md(rd("subtype_paired_vs_hmm.csv"), fmt="{:+.5f}")]
    L += ["## Fit diagnostics\n", _md(diag)]
    if (an / "reproduction.csv").exists():
        L += ["## Reproduction of the historical (Windows) run\n", _md(rd("reproduction.csv"),
                                                                       fmt="{:.3g}")]
    L += ["## PIGD = 0 targets\n", _md(rd("pigd_zero.csv"))]
    (an / "REPORT.md").write_text("\n".join(L))


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    an = run_dir / "analysis"
    if args.report_only:
        render_report(an)
        return 0
    an.mkdir(exist_ok=True)
    # A class with no support in a subset gives an empty-slice nanmean; the
    # result is NaN by design and reported as such.
    warnings.filterwarnings("ignore", message="Mean of empty slice")

    cfg = load_config()
    meta, preds, fits = load_run(run_dir)
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        mod = __import__("importlib").import_module("21_subscore_heldout_comparison")
        _, seqs = mod.build_sequences(cfg)
    except Exception as e:                                       # noqa: BLE001
        print(f"could not rebuild trajectories for checks: {e}")
        seqs = None

    chk = checks(preds, fits, meta, seqs)
    chk.to_csv(an / "checks.csv", index=False)
    print(chk.to_string(index=False))
    if not chk["pass"].all():
        print("\nSOME CHECKS FAILED: metrics are still written but must not be reported.")

    boot = H.SharedBootstrap.create(preds.held_out_patient.unique(), N_BOOT, SEED)
    pw = pointwise(preds)
    continuous_metrics(pw, boot).to_csv(an / "continuous_metrics.csv", index=False)
    pit_histograms(pw).to_csv(an / "pit_histograms.csv", index=False)
    paired(pw, comparison_plan(meta["arms"]), boot).to_csv(an / "paired.csv", index=False)

    hmm_frame, hmm_status = load_hmm(cfg, preds)
    print(f"HMM predictions: {hmm_status}")
    (an / "hmm_join_status.txt").write_text(hmm_status + "\n")
    sub, per_point, rel = subtype_analysis(preds, cfg, boot, hmm_frame)
    sub.to_csv(an / "subtype_metrics.csv", index=False)
    rel.to_csv(an / "subtype_reliability.csv", index=False)
    pd.concat([d.assign(model=k) for k, d in per_point.items()]).to_csv(
        an / "subtype_predictions.csv", index=False)
    if hmm_frame is not None:
        subtype_paired(per_point, boot, "hmm_loo_09").to_csv(
            an / "subtype_paired_vs_hmm.csv", index=False)
    subtype_paired(per_point, boot, "persistence").to_csv(
        an / "subtype_paired_vs_persistence.csv", index=False)

    fit_diagnostics(fits).to_csv(an / "fit_diagnostics.csv", index=False)
    hyperparameter_stability(fits).to_csv(an / "hyperparameter_stability.csv", index=False)
    reproduction(preds, cfg).to_csv(an / "reproduction.csv", index=False)
    pigd_zero(preds).to_csv(an / "pigd_zero.csv", index=False)
    render_report(an)
    print(f"wrote {an}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
