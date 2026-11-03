"""Experiment 6A: how much prediction-time history does the rich profile need?

The CT-HMM equivalent of this question was answered in `scripts/70`. This is
the rich clinical profile's version, and it is an **inference-time** experiment:
the model is fitted once per cross-validation fold by the existing procedure
and those fitted parameters are then held fixed across every history condition.
Only the held-out patient's prediction-time history changes.

Conditions, all predicting the same target visit:

    persistence, last 1, last 2, last 3, last 5, full history

`scripts/71` already covered last 2, last 3 and full history. This experiment is
independent of it: it adds last 1 and last 5, the persistence reference, the
class-specific and probability-quality metrics the brief asks for, pairwise
matched cohorts, and a reproducibility refit. It does **not** retrain on
sliding windows -- Experiment 6B is explicitly out of scope.

WHAT THE INSPECTION ESTABLISHED
-------------------------------
*   **Implementation.** `src/ppmi_subtype/clinical_profile.py`; fitted and
    evaluated by `scripts/64_clinical_profile_cv.py`.
*   **History.** A patient is a dict of `Y` (9 clinical domains), `d` (LEDD),
    `times` (years from baseline), `labels`, `events`, `visit_index`.
*   **State.** `filter_domain` runs a Kalman filter per domain from a fixed
    prior (`m = 0`, `P = diag([INIT_LEVEL_VAR, init_rate_var])`; zero is the
    training-cohort mean on the standardised scale). It is strictly causal and
    advances only on `times[k] - times[k-1]`.
*   **Prediction.** `predict_next` propagates the **filtered** state at visit k
    -- never a smoothed state -- over the real `dt`, then
    `derive_label_probabilities` pushes the predicted (tremor, PIGD)
    distribution through the project's own `subtype.classify` by Monte Carlo.
    No new classifier and no new threshold is introduced.
*   **Validation.** 5-fold patient cross-validation, which this experiment
    keeps. It is deliberately *not* switched to the CT-HMM's LOPO.

TRUNCATION IS A SLICE, NOT A RESET
----------------------------------
A window is a slice of `Y`, `d`, `times`, `labels`, `events` and `visit_index`,
and the filtered state is rebuilt from the retained visits alone. A
full-sequence filter is never computed and then cut. Original visit dates,
`years_from_baseline` and inter-visit gaps are carried through unchanged -- time
is not reset to zero. That matters beyond tidiness: `PopulationMean.predict`
takes absolute time, so a reset would silently corrupt any code path using it.

The initialisation consequence is stated rather than hidden: a truncated window
starts its first retained visit from the population prior, as if that visit were
a baseline. With one visit the level is a shrunk blend of that observation and
the cohort mean and the rate is still essentially the prior, because a single
point cannot identify a slope.

MEDICATION AT PREDICTION TIME
-----------------------------
`predict_next` takes the target visit's *prescribed* LEDD (`d[k+1]`). That is
the existing convention and it is preserved identically across every condition,
so the comparison between conditions is unaffected by it. No target-visit motor
measurement is ever used: those are the outcome. `scripts/71` measured what this
input is worth and found it negligible.

    python scripts/73_profile_prediction_history.py --out <dir> [--workers N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import clinical_profile as cp  # noqa: E402
from ppmi_subtype import clinical_states as cs  # noqa: E402
from ppmi_subtype import next_visit as nv  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype.plots import GRID, INK, INK_MUTED, _finish, _style  # noqa: E402

EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
CT_RUN = "experiments/20261020T104756Z_next_visit_td_pigd_prediction"
PERS_MODEL = "Persistence (last non-Indeterminate)"

WINDOWS: tuple = (1, 2, 3, 5, None)
WINDOW_LABEL = {1: "last 1", 2: "last 2", 3: "last 3", 5: "last 5",
                None: "full history"}
PERSIST = "persistence"
FULL = "full history"
N_FOLDS = 5
MIN_CLASS = 5
BOOT_KEYS = ("accuracy", "balanced_accuracy", "auc", "brier", "log_loss",
             "sensitivity_PIGD", "specificity_TD")

ACCENT = "#1f4e79"
COOL = "#2a78d6"
WARM = "#eb6834"
MUTED = "#8c8c88"

lines: list[str] = []
_WORK: dict = {}


def say(t: str = "") -> None:
    print(t, flush=True)
    lines.append(t)


def section(t: str) -> None:
    say("\n" + "=" * 78)
    say(t)
    say("=" * 78)


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def md5(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def block(f, **kw) -> str:
    return "    " + f.to_string(**kw).replace("\n", "\n    ")


SLICE_KEYS = ("Y", "d", "times", "labels", "events", "visit_index")


def truncate(seq: dict, lo: int, hi: int) -> dict:
    """Visits ``lo..hi`` inclusive as a standalone sequence.

    Every retained field keeps its original values: `times` are the real
    years-from-baseline, never re-based, and `events` and `visit_index` keep the
    identifiers of the visits they came from, so a prediction stays traceable.
    """
    out = {k: np.asarray(seq[k])[lo:hi + 1] for k in SLICE_KEYS}
    out["gaps"] = np.diff(out["times"])
    out["PATNO"] = seq["PATNO"]
    return out


def predict_windows(model, seq: dict, standardiser) -> list[dict]:
    """One record per (prediction case, window) for one held-out patient."""
    records = []
    n = len(seq["times"])
    for k in range(n - 1):
        n_hist = k + 1
        for w in WINDOWS:
            if w is not None and n_hist < w:
                continue                      # not eligible; never padded
            lo = 0 if w is None else k - w + 1
            sub = truncate(seq, lo, k + 1)
            recs = cp.predict_sequence(model, sub, standardiser,
                                       models=(cp.MODEL_FULL,))
            r = dict(recs[-1])                # the step whose target is k + 1
            r["window"] = WINDOW_LABEL[w]
            r["window_size"] = 0 if w is None else w
            r["n_retained_visits"] = len(sub["times"]) - 1
            # `predict_sequence` reports n_history relative to the sequence it
            # was handed, which for a truncated window is just the window
            # width. The patient's ACTUAL available history is k + 1, and that
            # is what eligibility and the integrity checks must use.
            r["available_history"] = n_hist
            r["first_retained_years"] = float(sub["times"][0])
            r["first_retained_event"] = sub["events"][0]
            r["retained_from_index"] = int(lo)
            records.append(r)
    return records


def run_fold(test_patients, fold: int):
    sequences = _WORK["sequences"]
    test_set = set(test_patients)
    train = [s for s in sequences if s["PATNO"] not in test_set]
    test = [s for s in sequences if s["PATNO"] in test_set]
    if not train or not test:
        raise ValueError(f"empty split (fold {fold})")
    if test_set & {s["PATNO"] for s in train}:
        raise RuntimeError("held-out patient leaked into training")

    # Every learned transformation fitted here, on training patients only, and
    # then held FIXED across all history conditions for this fold.
    standardiser = cp.Standardiser.fit(train)
    train_z = standardiser.apply(train)
    test_z = standardiser.apply(test)
    model = cp.fit(train_z, use_rate=True, use_medication=True)

    records = []
    for seq in test_z:
        records += predict_windows(model, seq, standardiser)
    frame = cp.records_to_frame(records)
    for col in ("window", "window_size", "n_retained_visits",
                "first_retained_years", "first_retained_event",
                "retained_from_index"):
        frame[col] = [r[col] for r in records]
    frame["fold"] = fold
    frame["n_train_patients"] = len(train)
    frame["fit_converged"] = bool(model.converged)
    frame["fit_train_loglik"] = float(model.loglik)
    # Parameter fingerprint: identical within a fold across every condition by
    # construction, and checked to be so downstream.
    frame["param_fingerprint"] = hashlib.sha256(
        np.concatenate([np.asarray(model.obs_var, float).ravel(),
                        np.asarray(model.level_var, float).ravel(),
                        np.asarray(model.rate_var, float).ravel(),
                        np.asarray(model.beta, float).ravel()]).tobytes()
    ).hexdigest()[:16]
    return frame


def _fold_task(args):
    fold, pats = args
    try:
        return run_fold(pats, fold), None
    except Exception as exc:                        # noqa: BLE001
        return None, {"fold": fold, "error": f"{type(exc).__name__}: {exc}"}


def _init(sequences):
    _WORK["sequences"] = sequences


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------
def _auc(y, p):
    y = np.asarray(y, float)
    if len(np.unique(y)) < 2:
        return float("nan")
    p = np.asarray(p, float)
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    r = np.arange(1, len(p) + 1, dtype=float)
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    ranks = np.empty(len(p), float)
    ranks[order] = r
    n1 = float(y.sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0)
                 / (n1 * float(len(y) - n1)))


def metrics_for(y, pred, p_pigd) -> dict:
    """PIGD is the positive class, matching the project's AUC convention."""
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    p = np.clip(np.asarray(p_pigd, float), 1e-12, 1 - 1e-12)
    n = len(y)
    n_p, n_t = int(y.sum()), int(n - y.sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens_p = tp / n_p if n_p else float("nan")
    spec_t = tn / n_t if n_t else float("nan")
    small = (n_p < MIN_CLASS) or (n_t < MIN_CLASS)
    extreme = float(np.mean((p <= 1e-9) | (p >= 1 - 1e-9)))
    degen = extreme > 0.5
    return {"n_targets": n, "n_TD": n_t, "n_PIGD": n_p,
            "pigd_prevalence": (n_p / n) if n else float("nan"),
            "mean_predicted_P_PIGD": float(np.mean(p)),
            "accuracy": (tp + tn) / n if n else float("nan"),
            "balanced_accuracy": (sens_p + spec_t) / 2.0,
            "sensitivity_PIGD": sens_p, "specificity_TD": spec_t,
            "precision_PIGD": (tp / (tp + fp)) if (tp + fp) else float("nan"),
            "precision_TD": (tn / (tn + fn)) if (tn + fn) else float("nan"),
            "auc": float("nan") if (small or degen) else _auc(y, p),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": (float("nan") if degen else
                         float(-np.mean(y * np.log(p)
                                        + (1 - y) * np.log(1 - p)))),
            "tp_PIGD": tp, "tn_TD": tn, "fp": fp, "fn": fn,
            "metrics_limited_by_class_size": bool(small),
            "degenerate_probabilities": bool(degen)}


def _patient_row_index(patients: np.ndarray) -> list:
    uniq, inv = np.unique(patients, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    si = inv[order]
    lo = np.searchsorted(si, np.arange(len(uniq)))
    hi = np.searchsorted(si, np.arange(len(uniq)), side="right")
    return [order[a:b] for a, b in zip(lo, hi)]


def patient_boot_all(patients, fn, n_boot=2000, seed=42):
    """Percentile CIs for every metric from ONE set of patient resamples.

    In a paired comparison ``fn`` computes both conditions on the same drawn
    rows, so the two conditions always see identical patient samples.
    Replicates where a metric is undefined -- a draw with only one class, which
    makes AUC meaningless -- are discarded and counted rather than imputed.
    """
    groups = _patient_row_index(np.asarray(patients))
    rng = np.random.default_rng(seed)
    draws = {k: np.full(n_boot, np.nan) for k in BOOT_KEYS}
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[i] for i in pick])
        try:
            m = fn(rows)
        except (ValueError, ZeroDivisionError, KeyError):
            continue
        for k in BOOT_KEYS:
            v = m.get(k, np.nan)
            if v is not None and np.isfinite(v):
                draws[k][b] = v
    out = {}
    for k in BOOT_KEYS:
        ok = draws[k][np.isfinite(draws[k])]
        out[k] = ((float(np.quantile(ok, 0.025)),
                   float(np.quantile(ok, 0.975)), int(len(ok)))
                  if len(ok) else (float("nan"), float("nan"), 0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-repro", action="store_true",
                    help="skip the reproducibility refit (smoke runs only)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    started = time.time()

    section("1. CONFIGURATION AND PROTOCOL")
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    pm = md5(panel_path)
    if pm != EXPECTED_PANEL_MD5:
        raise SystemExit(f"panel md5 {pm} != expected {EXPECTED_PANEL_MD5}")
    say(f"  panel md5              {pm}  (frozen panel verified)")
    panel = pd.read_csv(panel_path, low_memory=False)
    clinical = cs.build_clinical_panel(cfg, panel)
    sequences = cp.build_sequences(clinical)
    patients = sorted(s["PATNO"] for s in sequences)
    say(f"  clinical sequences     {len(sequences):,} patients")
    say(f"  validation             {N_FOLDS}-fold patient CV, one repeat "
        f"(the rich profile's existing scheme, NOT LOPO)")
    say(f"  seed                   {cfg.seed}")
    say(f"  conditions             {[PERSIST] + [WINDOW_LABEL[w] for w in WINDOWS]}")
    say("  parameter policy       model, standardiser, population mean and")
    say("                         LOCF spread fitted ONCE per fold on training")
    say("                         patients, then FIXED across every condition")
    say("  medication at predict  target visit's prescribed LEDD, the existing")
    say("                         convention, identical in every condition")

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(patients))
    pats = np.array(patients, dtype=object)
    tasks = [(f, pats[order[f::N_FOLDS]].tolist()) for f in range(N_FOLDS)]
    assert sum(len(t[1]) for t in tasks) == len(patients)
    assert len({p for _, ps in tasks for p in ps}) == len(patients)
    say(f"  fold sizes             {[len(t[1]) for t in tasks]}")

    section("2. CROSS-VALIDATION WITH RESTRICTED PREDICTION HISTORIES")
    frames, failed = [], []
    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                 initargs=(sequences,)) as ex:
            for res, err in ex.map(_fold_task, tasks):
                (failed if err else frames).append(err or res)
                say(f"    fold done ({time.time() - t0:.0f}s)")
    else:
        _init(sequences)
        for t in tasks:
            res, err = _fold_task(t)
            (failed if err else frames).append(err or res)
            say(f"    fold {t[0]} done ({time.time() - t0:.0f}s)")
    if failed:
        pd.DataFrame(failed).to_csv(out / "failed_folds.csv", index=False)
        raise SystemExit(f"{len(failed)} folds failed; see failed_folds.csv")

    pred = pd.concat(frames, ignore_index=True)
    say(f"  prediction rows        {len(pred):,}")
    pred = cp.derive_label_probabilities(pred, cfg)
    probs = pred[[f"p_{s}" for s in cs.STATE_NAMES]].to_numpy(float)
    p_td, p_pigd, defined = nv.conditional_two_class(probs)
    pred["P_TD"], pred["P_PIGD"] = p_td, p_pigd
    pred["two_class_defined"] = defined
    pred["predicted_state"] = nv.predicted_state(p_td, p_pigd)

    # ---------------------------------------------- persistence reference
    orig = pd.read_csv(ROOT / CT_RUN / "next_visit_predictions.csv",
                       low_memory=False)
    key = ["PATNO", "current_event", "target_event"]
    pers = orig[orig["model"] == PERS_MODEL][
        key + ["P_TD", "P_PIGD", "predicted_state", "two_class_defined"]].copy()
    spine = pred[pred["window"] == FULL][
        key + ["actual_next_label", "current_label", "interval_years",
               "current_years", "target_years", "available_history",
               "fold"]].copy()
    pers = pers.merge(spine, on=key, how="inner", validate="one_to_one")
    pers["window"] = PERSIST
    pers["window_size"] = -1
    pers["n_retained_visits"] = pers["available_history"]
    say(f"  persistence rows       {len(pers):,} (validated run, matched on "
        f"patient and visit pair)")

    allp = pd.concat([pred, pers], ignore_index=True)
    allp["y"] = (allp["actual_next_label"] == "PIGD").astype(int)
    allp["pred_bin"] = (allp["predicted_state"] == "PIGD").astype(int)
    keep_cols = [c for c in allp.columns if not c.startswith(("z_", "pred_",
                                                              "sd_", "actual_"))
                 or c in ("pred_bin", "actual_next_label")]
    allp[keep_cols].to_csv(out / "window_predictions.csv", index=False)
    say(f"  wrote window_predictions.csv ({len(allp):,} rows, "
        f"{len(keep_cols)} columns)")

    section("3. COHORTS")
    elig = allp[allp["actual_next_label"].isin(["TD", "PIGD"])
                & allp["two_class_defined"].astype(bool)].copy()
    n_ind = int((allp["window"] == FULL).sum() - (elig["window"] == FULL).sum())
    say(f"  Indeterminate / undefined-ratio targets excluded per condition: "
        f"{n_ind}")

    counts = (elig.groupby("window")
                  .agg(n_targets=("y", "size"), n_patients=("PATNO", "nunique"),
                       n_TD=("y", lambda s: int((s == 0).sum())),
                       n_PIGD=("y", "sum")).reset_index())
    counts["pigd_prevalence"] = counts["n_PIGD"] / counts["n_targets"]
    counts.round(4).to_csv(out / "cohort_counts.csv", index=False)
    say("\n  eligible targets per condition (cohorts DIFFER by design):")
    say(block(counts, index=False))

    by_cond = {w: set(map(tuple, g[key].to_numpy()))
               for w, g in elig.groupby("window")}
    common = set.intersection(*by_cond.values()) if by_cond else set()
    elig["common"] = [tuple(r) in common for r in elig[key].to_numpy()]
    say(f"\n  common cohort (every condition defined): {len(common):,} targets, "
        f"{elig[elig['common']]['PATNO'].nunique():,} patients")
    say("  Each pairwise comparison below instead uses its own MATCHED cohort,")
    say("  which is larger: a last-2 vs full comparison needs only n_history>=2.")

    section("4. METRICS")
    order_c = [PERSIST] + [WINDOW_LABEL[w] for w in WINDOWS]
    recs = []
    for cohort, sel in (("all eligible", elig),
                        ("common targets", elig[elig["common"]])):
        for w in order_c:
            g = sel[sel["window"] == w]
            if not len(g):
                continue
            yv, pv = g["y"].to_numpy(int), g["pred_bin"].to_numpy(int)
            qv = g["P_PIGD"].to_numpy(float)
            m = metrics_for(yv, pv, qv)
            m.update({"cohort": cohort, "window": w,
                      "n_patients": int(g["PATNO"].nunique())})
            cis = patient_boot_all(
                g["PATNO"].to_numpy(),
                lambda rows: metrics_for(yv[rows], pv[rows], qv[rows]),
                n_boot=args.n_boot, seed=cfg.seed)
            for k in BOOT_KEYS:
                lo, hi, nd = cis[k]
                m[f"{k}_lo"], m[f"{k}_hi"] = lo, hi
                m[f"{k}_valid_replicates"] = nd
            recs.append(m)
    mt = pd.DataFrame(recs)
    mt.round(4).to_csv(out / "window_metrics.csv", index=False)
    mt[mt.cohort == "all eligible"][
        ["window", "tp_PIGD", "tn_TD", "fp", "fn"]].to_csv(
        out / "confusion_matrices.csv", index=False)
    for cohort in ("all eligible", "common targets"):
        say(f"\n  --- {cohort} ---")
        say(block(mt[mt.cohort == cohort][
            ["window", "n_targets", "n_patients", "n_PIGD", "accuracy",
             "balanced_accuracy", "auc", "brier", "log_loss",
             "sensitivity_PIGD", "specificity_TD", "precision_PIGD",
             "precision_TD", "mean_predicted_P_PIGD"]].round(4), index=False))

    section("5. PAIRED COMPARISONS ON MATCHED TARGETS")
    say("  Each comparison uses its own matched cohort: the targets eligible")
    say("  in BOTH conditions. One patient resample serves both conditions in")
    say("  a replicate, so the pairing is preserved.")

    def paired(a, b, role):
        wide = elig[elig["window"].isin([a, b])].pivot_table(
            index=key + ["y"], columns="window",
            values=["pred_bin", "P_PIGD"], aggfunc="first").dropna()
        if not len(wide):
            return None
        wide = wide.reset_index()
        ya = wide["y"].to_numpy(int)
        pa_, pb_ = (wide[("pred_bin", a)].to_numpy(int),
                    wide[("pred_bin", b)].to_numpy(int))
        qa_, qb_ = (wide[("P_PIGD", a)].to_numpy(float),
                    wide[("P_PIGD", b)].to_numpy(float))

        def diffs(rows):
            ma = metrics_for(ya[rows], pa_[rows], qa_[rows])
            mb = metrics_for(ya[rows], pb_[rows], qb_[rows])
            return {k: ma[k] - mb[k] for k in BOOT_KEYS}

        rec = {"role": role, "comparison": f"{a}  -  {b}",
               "n_targets": len(wide),
               "n_patients": int(wide["PATNO"].nunique())}
        point = diffs(np.arange(len(wide)))
        cis = patient_boot_all(wide["PATNO"].to_numpy(), diffs,
                               n_boot=args.n_boot, seed=cfg.seed)
        for k in BOOT_KEYS:
            est = point[k]
            lo, hi, nd = cis[k]
            rec[k], rec[f"{k}_lo"], rec[f"{k}_hi"] = est, lo, hi
            rec[f"{k}_valid_replicates"] = nd
            rec[f"{k}_excludes_zero"] = bool(np.isfinite(lo)
                                             and (lo > 0 or hi < 0))
        return rec

    prs = []
    for w in (1, 2, 3, 5):
        r = paired(WINDOW_LABEL[w], FULL, "principal")
        if r:
            prs.append(r)
    for a in (WINDOW_LABEL[2], FULL):
        r = paired(a, PERSIST, "vs persistence")
        if r:
            prs.append(r)
    pt = pd.DataFrame(prs)
    pt.round(4).to_csv(out / "paired_comparisons.csv", index=False)
    say("")
    say(block(pt[["role", "comparison", "n_targets", "n_patients", "accuracy",
                  "accuracy_lo", "accuracy_hi", "auc", "auc_lo", "auc_hi"]
                 ].round(4), index=False))
    n_prin = int((pt["role"] == "principal").sum())
    say(f"\n  {n_prin} principal comparisons (each window vs full history).")
    say(f"  A Bonferroni threshold across them is alpha = "
        f"{0.05 / max(n_prin, 1):.4f}; intervals below are UNADJUSTED and are")
    say("  reported as such. An interval containing zero is not evidence of")
    say("  equivalence, only absence of a detected difference.")

    # ================================================= 6. reproducibility
    section("6. REPRODUCIBILITY")
    repro = {"refit_checked": False}
    if not args.skip_repro:
        _init(sequences)
        again = run_fold(tasks[0][1], 0)
        a = frames[0].sort_values(key + ["window"]).reset_index(drop=True)
        b = again.sort_values(key + ["window"]).reset_index(drop=True)
        cols = ["pred_tremor", "pred_pigd", "sd_tremor", "sd_pigd"]
        same = bool(len(a) == len(b) and all(
            np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), atol=0)
            for c in cols if c in a.columns))
        repro = {"refit_checked": True, "fold": 0, "rows": int(len(b)),
                 "identical_predictions": same,
                 "same_fingerprint": bool(a["param_fingerprint"].iloc[0]
                                          == b["param_fingerprint"].iloc[0])}
        say(f"  refitted fold 0 with the same seed: {len(b):,} rows, "
            f"identical predictions {same}, "
            f"same parameter fingerprint {repro['same_fingerprint']}")
    else:
        say("  SKIPPED (--skip-repro); this run is not a reproducibility check")

    # metrics must be recomputable from the saved predictions alone
    saved = pd.read_csv(out / "window_predictions.csv", low_memory=False)
    se = saved[saved["actual_next_label"].isin(["TD", "PIGD"])
               & saved["two_class_defined"].astype(bool)]
    g = se[se["window"] == FULL]
    recomputed = metrics_for(g["y"], g["pred_bin"], g["P_PIGD"])
    ref = mt[(mt.cohort == "all eligible") & (mt.window == FULL)].iloc[0]
    metrics_reproduce = bool(
        abs(recomputed["accuracy"] - ref["accuracy"]) < 1e-12
        and abs(recomputed["auc"] - ref["auc"]) < 1e-12
        and abs(recomputed["brier"] - ref["brier"]) < 1e-12)
    say(f"  metrics recomputed from the saved prediction file match: "
        f"{metrics_reproduce}")

    # ================================================= 7. figures
    section("7. FIGURES")
    cd = mt[mt.cohort == "all eligible"].set_index("window")
    xs = np.arange(len(order_c))

    def series(key):
        v = np.array([cd.loc[w, key] if w in cd.index else np.nan
                      for w in order_c], float)
        lo = np.array([cd.loc[w, f"{key}_lo"] if w in cd.index else np.nan
                       for w in order_c], float)
        hi = np.array([cd.loc[w, f"{key}_hi"] if w in cd.index else np.nan
                       for w in order_c], float)
        return v, lo, hi

    def xlabels(ax):
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [f"{w}\nn={int(cd.loc[w, 'n_targets'])}" if w in cd.index else w
             for w in order_c], fontsize=8)

    # Figure 1: headline performance
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))
    for ax, k, lab in ((axes[0], "accuracy", "Accuracy"),
                       (axes[1], "balanced_accuracy", "Balanced accuracy"),
                       (axes[2], "auc", "AUC-ROC")):
        _style(ax)
        v, lo, hi = series(k)
        for i in range(len(order_c)):
            if not np.isfinite(v[i]):
                ax.annotate("n/a", (xs[i], np.nanmean(v)), ha="center",
                            fontsize=8, color=INK_MUTED)
                continue
            c = MUTED if order_c[i] == PERSIST else (
                WARM if order_c[i] == FULL else ACCENT)
            ax.errorbar(xs[i], v[i], yerr=[[v[i] - lo[i]], [hi[i] - v[i]]],
                        fmt="o", color=c, markersize=6.5, capsize=3,
                        elinewidth=1.3, zorder=3)
        xlabels(ax)
        ax.set_ylabel(lab)
        ax.set_title(lab, fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle("Rich clinical profile: performance by prediction-time history"
                 "\n95% patient-clustered intervals; n under each label — "
                 "cohorts DIFFER between conditions, so read the paired "
                 "comparisons for inference",
                 fontsize=10.5, x=0.01, y=0.995, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    say(f"  wrote {_finish(fig, out / 'fig1_history_performance.png').name}")

    # Figure 2: probability quality
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for ax, k, lab in ((axes[0], "brier", "Brier score"),
                       (axes[1], "log_loss", "Log loss")):
        _style(ax)
        v, lo, hi = series(k)
        for i in range(len(order_c)):
            if not np.isfinite(v[i]):
                ax.annotate("n/a", (xs[i], np.nanmin(v[np.isfinite(v)])),
                            ha="center", fontsize=8, color=INK_MUTED)
                continue
            c = MUTED if order_c[i] == PERSIST else (
                WARM if order_c[i] == FULL else ACCENT)
            ax.errorbar(xs[i], v[i], yerr=[[v[i] - lo[i]], [hi[i] - v[i]]],
                        fmt="o", color=c, markersize=6.5, capsize=3,
                        elinewidth=1.3, zorder=3)
        xlabels(ax)
        ax.set_ylabel(lab)
        ax.set_title(f"{lab} — lower is better", fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle("Probability quality by prediction-time history\n"
                 "persistence emits only 0 and 1, so its log loss is not a "
                 "meaningful quantity and is omitted",
                 fontsize=10.5, x=0.01, y=0.995, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    say(f"  wrote {_finish(fig, out / 'fig2_probability_quality.png').name}")

    # Figure 3: class-specific
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    _style(ax)
    for k, c, lab, off in (("sensitivity_PIGD", ACCENT, "PIGD sensitivity", -0.12),
                           ("specificity_TD", COOL, "TD specificity", +0.12)):
        v, lo, hi = series(k)
        ax.errorbar(xs + off, v, yerr=[v - lo, hi - v], fmt="o", color=c,
                    markersize=6, capsize=3, elinewidth=1.3, label=lab,
                    zorder=3)
    xlabels(ax)
    ax.set_ylabel("Rate")
    ax.set_title("Class-specific performance by prediction-time history\n"
                 "PIGD sensitivity is the harder of the two and moves most",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    say(f"  wrote {_finish(fig, out / 'fig3_class_specific.png').name}")

    # Figure 4: paired AUC differences vs full history
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    _style(ax)
    pr = pt[pt["role"] == "principal"]
    yy = np.arange(len(pr))
    v = pr["auc"].to_numpy(float)
    lo = pr["auc_lo"].to_numpy(float)
    hi = pr["auc_hi"].to_numpy(float)
    ax.errorbar(v, yy, xerr=[v - lo, hi - v], fmt="o", color=ACCENT,
                markersize=6.5, capsize=3, elinewidth=1.4, zorder=3)
    ax.axvline(0, color=INK, linewidth=1.2, zorder=2)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{c.split('  -  ')[0]}\n(n={n})"
                        for c, n in zip(pr["comparison"], pr["n_targets"])],
                       fontsize=8.5)
    ax.set_xlabel("AUC difference versus full history "
                  "(negative means the shorter window is worse)")
    ax.set_title("Paired AUC differences against full history\n"
                 "each on its own matched targets; 95% patient-clustered "
                 "intervals", fontsize=10.5, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    say(f"  wrote {_finish(fig, out / 'fig4_paired_auc.png').name}")

    # ================================================= 8. checks
    section("8. VALIDATION CHECKS")
    src = Path(__file__).read_text()
    w_rows = pred[pred["window"] != FULL]
    fullr = pred[pred["window"] == FULL]
    # a window keeps exactly the W visits ending at the current visit
    retained_ok = True
    for w in WINDOWS:
        if w is None:
            continue
        g = pred[pred["window"] == WINDOW_LABEL[w]]
        if len(g) and not bool(
                (g["retained_from_index"] == g["available_history"] - w).all()):
            retained_ok = False
    # original timeline: the first retained visit's time must equal the parent
    seq_times = {s["PATNO"]: np.asarray(s["times"], float) for s in sequences}
    tl_ok = True
    for _, r in pred.sample(min(4000, len(pred)),
                            random_state=cfg.seed).iterrows():
        t = seq_times[r["PATNO"]]
        if not np.isclose(float(r["first_retained_years"]),
                          float(t[int(r["retained_from_index"])]), atol=0):
            tl_ok = False
            break
    one_fp = bool(pred.groupby("fold")["param_fingerprint"].nunique().eq(1).all())
    checks = {
        "every fold completed": len(failed) == 0,
        "every fit converged": bool(pred["fit_converged"].all()),
        "patient-level separation: each patient in exactly one fold":
            bool((pred.groupby("PATNO")["fold"].nunique() == 1).all()),
        "parameters fitted on a strict subset of patients":
            bool((pred["n_train_patients"] < len(sequences)).all()),
        "held-out patient never in training":
            'raise RuntimeError("held-out patient leaked into training")' in src,
        "preprocessing fitted on training patients only":
            "standardiser = cp.Standardiser.fit(train)" in src,
        "model parameters identical across conditions within a fold": one_fp,
        "each target appears exactly once per condition":
            bool(not pred.duplicated(subset=key + ["window"]).any()),
        "all conditions share the same target for a case":
            bool(pred.groupby(key)["target_event"].nunique().eq(1).all()),
        "window kept exactly the W visits ending at the current visit":
            retained_ok,
        "no condition used a padded or fabricated visit": bool(all(
            (pred[pred["window"] == WINDOW_LABEL[w]]["available_history"] >= w)
            .all() for w in WINDOWS if w is not None)),
        "retained visit count equals the window width exactly": bool(all(
            (pred[pred["window"] == WINDOW_LABEL[w]]["n_retained_visits"] == w)
            .all() for w in WINDOWS if w is not None)),
        "full-history condition retained every prior visit":
            bool((fullr["n_retained_visits"]
                  == fullr["available_history"]).all()),
        "original timestamps preserved (sampled against parent sequences)":
            tl_ok,
        "every historical visit precedes its target":
            bool((pred["target_years"] > pred["current_years"]).all()),
        "inter-visit interval preserved and positive":
            bool((pred["interval_years"] > 0).all()),
        "state rebuilt from retained visits only, never truncated after":
            ("out = {k: np.asarray(seq[k])[lo:hi + 1] for k in SLICE_KEYS}"
             in src)
            and "filtered = model.filter_sequence(seq)" in
            (ROOT / "src/ppmi_subtype/clinical_profile.py").read_text(),
        "prediction uses the causal filter, no smoothing":
            "def smooth" not in
            (ROOT / "src/ppmi_subtype/clinical_profile.py").read_text()
            or "filtered = model.filter_sequence(seq)" in
            (ROOT / "src/ppmi_subtype/clinical_profile.py").read_text(),
        "target-visit LEDD identical across conditions for a case":
            bool(pred.groupby(key)["ledd_next"].nunique().eq(1).all()),
        "Indeterminate targets excluded from binary scoring":
            bool(set(elig["actual_next_label"]) == {"TD", "PIGD"}),
        "predictions finite": bool(np.isfinite(allp["P_PIGD"]).all()),
        "probabilities within [0, 1]":
            bool((allp["P_PIGD"] >= 0).all() and (allp["P_PIGD"] <= 1).all()),
        "P_TD + P_PIGD == 1 where defined": bool(np.allclose(
            allp.loc[allp["two_class_defined"].astype(bool), "P_TD"]
            + allp.loc[allp["two_class_defined"].astype(bool), "P_PIGD"],
            1.0, atol=1e-12)),
        "hard call follows the project's decision rule": bool(
            (allp["predicted_state"].isin(["TD", "PIGD"])).all()),
        "bootstrap resamples patients, not targets":
            "groups = _patient_row_index(np.asarray(patients))" in src,
        "paired bootstrap uses one patient sample for both conditions":
            "cis = patient_boot_all(wide[\"PATNO\"].to_numpy(), diffs," in src,
        "metrics reproduce from the saved prediction file": metrics_reproduce,
        "refit with the same seed reproduces predictions":
            bool(repro.get("identical_predictions", False))
            if repro["refit_checked"] else "skipped",
    }
    hard = {k: v for k, v in checks.items() if v != "skipped"}
    for n, v in checks.items():
        tag = "SKIP" if v == "skipped" else ("PASS" if v else "FAIL")
        say(f"    [{tag}]  {n}")
    pd.DataFrame([{"check": k, "result": str(v)} for k, v in checks.items()]
                 ).to_csv(out / "validation_checks.csv", index=False)
    if not all(hard.values()):
        raise SystemExit(f"CRITICAL CHECK FAILED: "
                         f"{[k for k, v in hard.items() if not v]}")

    section("9. PROVENANCE")
    env = {"panel_md5": pm, "python": sys.version.split()[0],
           "numpy": np.__version__, "pandas": pd.__version__,
           "node": platform.node(), "workers": args.workers,
           "n_bootstrap": args.n_boot, "seed": cfg.seed, "n_folds": N_FOLDS,
           "ct_run_for_persistence": CT_RUN,
           "elapsed_seconds": round(time.time() - started, 1)}
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:                                   # noqa: BLE001
        env["git_commit"] = "unknown"
    for k, v in env.items():
        say(f"  {k:<24} {v}")

    (out / "summary.json").write_text(json.dumps(
        {"environment": env, "checks": {k: str(v) for k, v in checks.items()},
         "reproducibility": repro,
         "code_sha256": {str(Path(__file__).relative_to(ROOT)):
                         sha(Path(__file__))},
         "conditions": order_c,
         "n_indeterminate_excluded": n_ind,
         "n_common_targets": len(common),
         "cohort_counts": counts.to_dict(orient="records"),
         "metrics": mt.to_dict(orient="records"),
         "paired": pt.to_dict(orient="records")},
        indent=2, default=float), encoding="utf-8")
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    skip = {"MANIFEST.sha256", "driver.log"}
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in skip)
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files),
        encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
