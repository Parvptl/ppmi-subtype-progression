"""Stage 4 analysis: summarise a `23_hmm_stability.py` run from its saved files.

    python scripts/24_analyse_hmm_stability.py <run_dir> [--report-only]

Writes <run_dir>/analysis/*.csv and REPORT.md (rendered from those CSVs).

Three kinds of result are kept apart, and labelled as such everywhere:

* POPULATION (exploratory): all patients, many dispersed starts. Describes the
  likelihood surface. Not a performance estimate.
* PER-CONFIGURATION held-out results: every configuration's outer-fold score.
  A sensitivity table. Choosing the best row of it would be selection on the
  test folds, so no row of it is "the result".
* SELECTED PROCEDURE: in each outer fold, the configuration chosen by inner CV
  (or by training BIC) on training patients only, scored on that outer fold.
  This is the unbiased estimate of "select a configuration, then predict".
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import heldout as H  # noqa: E402
from ppmi_subtype import hmm_stability as HS  # noqa: E402

NAMES = ["TD", "Indeterminate", "PIGD"]
PCOLS = [f"p_{n}" for n in NAMES]
REFERENCE = "C3_f50_audited"
N_BOOT, SEED = 4000, 42


# ---------------------------------------------------------------------------
#  Population (exploratory)
# ---------------------------------------------------------------------------
def population(run_dir: Path):
    rows = []
    for f in sorted((run_dir / "population").glob("*/start_*.json")):
        rows.append(json.loads(f.read_text()))
    if not rows:
        return None, None, None
    starts = pd.DataFrame(rows)
    summ, params = [], []
    for cfg, g in starts.groupby("config"):
        recs = [HS.StartRecord(start=r.start, loglik=r.loglik, success=r.success,
                               message=r.message, n_iter=r.n_iter, n_fev=r.n_fev,
                               proj_grad=r.proj_grad, theta=np.zeros(1))
                for r in g.itertuples()]
        ll = g.loglik.to_numpy()
        best = g.loc[g.loglik.idxmax()]
        near = g[np.abs(g.loglik - best.loglik) <= 0.01]
        Qb, Eb = np.array(best.Q), np.array(best.E)
        dQ = max(float(np.max(np.abs(np.array(q) - Qb))) for q in near.Q)
        dE = max(float(np.max(np.abs(np.array(e) - Eb))) for e in near.E)
        k, n_obs = int(best.n_params), int(best.n_observations)
        summ.append({
            "config": cfg, "n_starts": len(g), "n_params": k,
            "best_loglik": float(best.loglik), "worst_loglik": float(ll.min()),
            "AIC": 2 * k - 2 * best.loglik, "BIC_n_obs": k * np.log(n_obs) - 2 * best.loglik,
            "share_within_0.01": float(len(near) / len(g)),
            "n_loglik_levels_0.01": int(len(np.unique(np.round(ll / 0.01)))),
            "best_is_default_start": bool(int(best.start) == 0),
            "default_start_gap": float(best.loglik - g[g.start == 0].loglik.iloc[0])
            if (g.start == 0).any() else np.nan,
            "max_abs_dQ_among_best": dQ, "max_abs_dE_among_best": dE,
            "share_success_flag": float(g.success.mean()),
            "share_proj_grad_lt_1e-3": float((g.proj_grad < 1e-3).mean()),
            "share_success_among_best": float(near.success.mean()),
            "messages": json.dumps(g.message.str.slice(0, 40).value_counts().to_dict()),
            "median_seconds": float(g.seconds.median()),
        })
        del recs
        nh = Qb.shape[0]
        hid = NAMES if nh == 3 else [f"H{i + 1}" for i in range(nh)]
        for a in range(nh):
            sojourn = -1.0 / Qb[a, a] if Qb[a, a] < 0 else np.inf
            params.append({"config": cfg, "latent": hid[a], "pi": best.pi[a],
                           "mean_sojourn_years": sojourn,
                           **{f"rate_to_{hid[b]}": Qb[a, b] for b in range(nh) if b != a},
                           **{f"emit_{NAMES[c]}": Eb[a, c] for c in range(3)}})
    return (starts.drop(columns=["Q", "E", "pi"]), pd.DataFrame(summ), pd.DataFrame(params))


# ---------------------------------------------------------------------------
#  Cross-validation
# ---------------------------------------------------------------------------
def load_cv(run_dir: Path):
    cvd = run_dir / "cv"
    js = [json.loads(f.read_text()) for f in sorted(cvd.glob("*/outer*.json"))]
    tasks = pd.DataFrame(js)
    preds = [pd.read_csv(f) for f in sorted(cvd.glob("*/outer*.csv"))
             if not f.parent.name.startswith("_")]
    base = [pd.read_csv(f) for f in sorted((cvd / "_baselines").glob("outer*.csv"))]
    P = pd.concat(preds + base, ignore_index=True)
    # `to_state` carries the config's label codes; their order in `state_codes`
    # is (TD, Indeterminate, PIGD), the order of the probability columns.
    codes = json.loads((run_dir / "run_config_cv.json").read_text())["state_codes"]
    order = {int(s): k for k, s in enumerate(codes)}
    P["actual"] = P.to_state.astype(int).map(order)
    if P.actual.isna().any():
        raise RuntimeError("unmapped label codes in predictions")
    return tasks, P


def config_metrics(P: pd.DataFrame, priors: dict, boot) -> tuple[pd.DataFrame, dict]:
    rows, aligned = [], {}
    for cfg, g in P.groupby("config"):
        g = g.sort_values(["PATNO", "visit_index"]).reset_index(drop=True)
        probs = g[PCOLS].to_numpy()
        actual = g.actual.to_numpy(int)
        pat = g.PATNO.to_numpy()
        prior = np.stack([priors[o] for o in g.outer])
        rules = {"argmax": probs.argmax(axis=1),
                 "prior_corrected": (probs / prior).argmax(axis=1)}
        for rule, pred in rules.items():
            s = H.categorical_summary(probs, actual, pred, pat, boot)
            s.update({"config": cfg, "rule": rule,
                      "auc_Indeterminate": HS.auc_one_vs_rest(probs[:, 1], actual == 1),
                      "mean_p_Indet_when_Indet": float(probs[actual == 1, 1].mean()),
                      "mean_p_Indet_otherwise": float(probs[actual != 1, 1].mean()),
                      "max_p_Indet": float(probs[:, 1].max()),
                      **H.expected_calibration_error(probs, actual)})
            rows.append(s)
        aligned[cfg] = g
    return pd.DataFrame(rows), aligned


def paired_vs(aligned: dict, priors, boot, reference=REFERENCE) -> pd.DataFrame:
    if reference not in aligned:
        return pd.DataFrame()
    R = aligned[reference]
    rows = []
    for cfg, A in aligned.items():
        if cfg == reference:
            continue
        if not (np.array_equal(A.PATNO, R.PATNO) and np.array_equal(A.visit_index, R.visit_index)):
            raise RuntimeError(f"{cfg} not aligned with {reference}")
        act, pat = A.actual.to_numpy(int), A.PATNO.to_numpy()

        def ll(F):
            p = F[PCOLS].to_numpy()
            return -np.log(np.clip(p[np.arange(len(act)), act], 1e-12, None))
        est, lo, hi = boot.mean_ci(ll(A) - ll(R), pat)
        rows.append({"config": cfg, "reference": reference, "metric": "log_loss",
                     "difference": est, "lo": lo, "hi": hi})
        for rule in ("argmax", "prior_corrected"):
            def pred(F):
                p = F[PCOLS].to_numpy()
                if rule == "argmax":
                    return p.argmax(axis=1)
                return (p / np.stack([priors[o] for o in F.outer])).argmax(axis=1)
            d = (pred(A) == act).astype(float) - (pred(R) == act).astype(float)
            est, lo, hi = boot.mean_ci(d, pat)
            rows.append({"config": cfg, "reference": reference,
                         "metric": f"accuracy_{rule}", "difference": est, "lo": lo, "hi": hi})
    out = pd.DataFrame(rows)
    out["excludes_zero"] = (out.lo > 0) | (out.hi < 0)
    return out


def init_variation(tasks: pd.DataFrame) -> pd.DataFrame:
    """Held-out log-likelihood across starts, per configuration and outer fold."""
    rows = []
    outer = tasks[tasks.inner.isna()]
    for t in outer.itertuples():
        recs = [r for r in t.records if "test_loglik" in r]
        if len(recs) < 2:
            continue
        tr = np.array([r["loglik"] for r in recs])
        te = np.array([r["test_loglik"] for r in recs]) / recs[0]["test_n"]
        best_train = int(np.argmax(tr))
        rows.append({
            "config": t.config, "outer": t.outer, "n_starts": len(recs),
            "test_ll_per_transition_at_best_train": float(te[best_train]),
            "test_ll_min": float(te.min()), "test_ll_max": float(te.max()),
            "test_ll_sd": float(te.std(ddof=1)),
            "train_ll_range": float(tr.max() - tr.min()),
            "share_starts_within_0.01_train": float(np.mean(tr >= tr.max() - 0.01)),
            "corr_train_test_ll": float(np.corrcoef(tr, te)[0, 1]) if np.ptp(tr) > 0 else np.nan,
            "oracle_gap_DO_NOT_REPORT_AS_PERFORMANCE": float(te.max() - te[best_train]),
            "share_success_flag": float(np.mean([r["success"] for r in recs])),
            "share_proj_grad_lt_1e-3": float(np.mean([r["proj_grad"] < 1e-3 for r in recs])),
        })
    return pd.DataFrame(rows)


def parameter_variation(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cfg, g in tasks[tasks.inner.isna()].groupby("config"):
        Q = np.stack([np.array(q) for q in g.best_Q])
        E = np.stack([np.array(e) for e in g.best_E])
        nh = Q.shape[1]
        hid = NAMES if nh == 3 else [f"H{i + 1}" for i in range(nh)]
        for a in range(nh):
            for b in range(nh):
                if a != b:
                    v = Q[:, a, b]
                    rows.append({"config": cfg, "parameter": f"q[{hid[a]}->{hid[b]}]",
                                 "min": v.min(), "median": np.median(v), "max": v.max()})
            for c in range(3):
                v = E[:, a, c]
                rows.append({"config": cfg, "parameter": f"E[{hid[a]}->{NAMES[c]}]",
                             "min": v.min(), "median": np.median(v), "max": v.max()})
    return pd.DataFrame(rows)


def selection(tasks: pd.DataFrame, n_outer: int) -> pd.DataFrame:
    rows = []
    inner = tasks[tasks.inner.notna()]
    outer = tasks[tasks.inner.isna()]
    for o in range(n_outer):
        cv = inner[inner.outer == o].groupby("config").agg(
            inner_ll=("test_loglik", "sum"), inner_n=("test_n", "sum"),
            n_inner=("inner", "nunique"))
        if cv.empty:
            continue
        ob = outer[(outer.outer == o) & outer.config.isin(cv.index)].set_index("config")
        bic = ob.n_params * np.log(ob.n_train_obs) - 2 * ob.train_loglik
        complete = cv[cv.n_inner == cv.n_inner.max()]
        rows.append({"outer": o,
                     "selected_inner_cv": complete.inner_ll.idxmax(),
                     "selected_bic": bic.idxmin(),
                     **{f"inner_ll_per_transition_{c}": r.inner_ll / r.inner_n
                        for c, r in cv.iterrows()},
                     **{f"train_bic_{c}": v for c, v in bic.items()}})
    return pd.DataFrame(rows)


def selected_procedure(aligned, sel, priors, boot) -> pd.DataFrame:
    out = []
    for rule_col in ("selected_inner_cv", "selected_bic"):
        if sel.empty:
            break
        pick = dict(zip(sel.outer, sel[rule_col]))
        parts = [aligned[c][aligned[c].outer == o] for o, c in pick.items()]
        g = pd.concat(parts).sort_values(["PATNO", "visit_index"]).reset_index(drop=True)
        probs, act, pat = g[PCOLS].to_numpy(), g.actual.to_numpy(int), g.PATNO.to_numpy()
        prior = np.stack([priors[o] for o in g.outer])
        for rule, pred in {"argmax": probs.argmax(axis=1),
                           "prior_corrected": (probs / prior).argmax(axis=1)}.items():
            s = H.categorical_summary(probs, act, pred, pat, boot)
            s.update({"procedure": rule_col, "rule": rule,
                      "selections": json.dumps(pd.Series(pick).value_counts().to_dict())})
            out.append(s)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
def _md(df, cols=None, fmt="{:.4f}"):
    df = df if cols is None else df[[c for c in cols if c in df.columns]]
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join(
        "| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                          for v in r) + " |" for r in df.itertuples(index=False)) + "\n"


def render(an: Path):
    meta = {}
    for f in an.parent.glob("run_config_*.json"):
        meta[f.stem] = json.loads(f.read_text())
    L = ["# Stage 4 — HMM stability study (generated)\n"]
    for k, m in meta.items():
        L.append(f"- `{k}`: git `{m.get('git_commit')}`, panel `{m.get('panel_md5')}`, "
                 f"{m.get('n_patients')} patients, {m.get('n_transitions')} transitions, "
                 f"starts {m.get('starts')}/{m.get('pop_starts')}, outer {m.get('outer')}, "
                 f"inner {m.get('inner')}" + (" — **SMOKE TEST**" if m.get("smoke_test") else ""))
    L.append("")
    if (an / "population_summary.csv").exists():
        L += ["## A. Population fits — EXPLORATORY, not held out\n",
              "All patients; dispersed starts. `share_within_0.01`: starts reaching the best "
              "log-likelihood. `max_abs_dQ/dE_among_best`: parameter disagreement among those "
              "starts (a flat ridge if large). `share_success_flag` is scipy's flag; "
              "`share_proj_grad_lt_1e-3` is the stationarity check.\n",
              _md(pd.read_csv(an / "population_summary.csv")),
              "### Best-solution parameters\n", _md(pd.read_csv(an / "population_parameters.csv"))]
    if not (an / "cv_config_metrics.csv").exists():
        (an / "REPORT.md").write_text("\n".join(L))
        return
    L += ["## B. Held-out, per configuration — SENSITIVITY TABLE, not a selected result\n",
          "`Persistence (stay)` is a rule with one-hot probabilities: its log loss, Brier "
          "and ECE are degenerate (a wrong one-hot costs -log 1e-12) and are not comparable "
          "with the probabilistic models. Compare it on accuracy and recall only.\n",
          _md(pd.read_csv(an / "cv_config_metrics.csv"),
              ["config", "rule", "accuracy", "accuracy_lo", "accuracy_hi", "balanced_accuracy",
               "recall_TD", "recall_Indeterminate", "recall_PIGD",
               "n_predicted_Indeterminate", "log_loss", "brier", "auc_Indeterminate",
               "mean_p_Indet_when_Indet", "mean_p_Indet_otherwise", "max_p_Indet",
               "ece_Indeterminate"])]
    if (an / "cv_paired_vs_reference.csv").exists():
        L += [f"### Paired against `{REFERENCE}` (the original LOO protocol)\n",
              _md(pd.read_csv(an / "cv_paired_vs_reference.csv"), fmt="{:+.5f}")]
    L += ["## C. Selected procedure — the unbiased estimate\n",
          "Configuration chosen inside each outer fold from training patients only.\n",
          _md(pd.read_csv(an / "selection.csv")),
          _md(pd.read_csv(an / "selected_procedure_metrics.csv"),
              ["procedure", "rule", "accuracy", "accuracy_lo", "accuracy_hi",
               "balanced_accuracy", "balanced_accuracy_lo", "balanced_accuracy_hi",
               "recall_Indeterminate", "log_loss", "log_loss_lo", "log_loss_hi", "selections"])]
    L += ["## D. Variation across initialisations (held-out)\n",
          "`oracle_gap_…` is how much better the best start would have scored on the test "
          "fold than the start chosen by training likelihood. It is shown only to size the "
          "initialisation effect; choosing starts by it would be selection on test.\n",
          _md(pd.read_csv(an / "cv_init_variation.csv")),
          "## E. Parameter variation across outer folds\n",
          _md(pd.read_csv(an / "cv_parameter_variation.csv"))]
    (an / "REPORT.md").write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    an = run_dir / "analysis"
    if args.report_only:
        render(an)
        return 0
    an.mkdir(exist_ok=True)
    warnings.filterwarnings("ignore", message="Mean of empty slice")

    starts, summ, params = population(run_dir)
    if starts is not None:
        starts.to_csv(an / "population_starts.csv", index=False)
        summ.to_csv(an / "population_summary.csv", index=False)
        params.to_csv(an / "population_parameters.csv", index=False)

    if (run_dir / "cv").exists():
        meta = json.loads((run_dir / "run_config_cv.json").read_text())
        tasks, P = load_cv(run_dir)
        outer_tasks = tasks[tasks.inner.isna()]
        priors = {int(o): np.array(g.train_prior.iloc[0])
                  for o, g in outer_tasks.groupby("outer")}
        boot = H.SharedBootstrap.create(P.PATNO.unique(), N_BOOT, SEED)
        cm, aligned = config_metrics(P, priors, boot)
        cm.to_csv(an / "cv_config_metrics.csv", index=False)
        paired_vs(aligned, priors, boot).to_csv(an / "cv_paired_vs_reference.csv", index=False)
        init_variation(tasks).to_csv(an / "cv_init_variation.csv", index=False)
        parameter_variation(tasks).to_csv(an / "cv_parameter_variation.csv", index=False)
        sel = selection(tasks, meta["outer"])
        sel.to_csv(an / "selection.csv", index=False)
        selected_procedure(aligned, sel, priors, boot).to_csv(
            an / "selected_procedure_metrics.csv", index=False)
        conv = pd.DataFrame([{"config": t.config, "outer": t.outer, "inner": t.inner,
                              "start": r["start"], "success": r["success"],
                              "message": r["message"], "proj_grad": r["proj_grad"],
                              "n_iter": r["n_iter"], "loglik": r["loglik"]}
                             for t in tasks.itertuples() for r in t.records])
        conv.to_csv(an / "cv_start_records.csv", index=False)
    render(an)
    print(f"wrote {an}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
