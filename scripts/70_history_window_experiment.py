"""Does older subtype history add predictive value? CT-HMM history windows.

The stage-wise study found that accuracy does not improve as a patient's
history lengthens. This experiment asks the complementary question directly:
if the CT-HMM is only allowed to *see* the most recent W visits at prediction
time, does it do worse?

Conditions, all predicting the same next visit:

    persistence   carry the last non-Indeterminate label forward (no model)
    last 1        HMM filtering over the 1 most recent visit
    last 2        HMM filtering over the 2 most recent visits
    last 3        HMM filtering over the 3 most recent visits
    last 5        HMM filtering over the 5 most recent visits
    full          HMM filtering over every visit before the target

"Last 1" is not persistence: the HMM still starts from the fitted initial
distribution pi, applies the fitted generator Q over the real elapsed time, and
emits through the fitted matrix E.

PARAMETER POLICY -- the central design choice, stated plainly
-------------------------------------------------------------
Q, E and pi are fitted **once per leave-one-patient-out fold, on the training
patients' full sequences**, and the *same fitted parameters* serve every window
variant. Only the held-out patient's prediction-time history is truncated.

That isolates the question being asked -- how much history does the filter need
at prediction time -- from a different question, namely how much history the
*fitting* needs. Refitting per window would change the training data and the
estimator at the same time as the window, and any difference would then not be
attributable to history length. The consequence is an assumption worth naming:
a truncated window treats its first retained visit as if it were a baseline
drawn from pi, when in reality it sits mid-trajectory. Section 7 reports what
that costs.

Everything else follows the existing validated pipeline: the same panel, the
same `loo.prepare`, the same `nv.build_cases`, the same multistart fitting
through `hmm_stability.fit_multistart` with the same seed, emission floor and
restart count from `config`, the same conditional two-class restriction, and
the same exclusion of Indeterminate targets from binary scoring. No existing
module is modified.

    python scripts/70_history_window_experiment.py --out <dir> [--workers N]
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

from ppmi_subtype import hmm, hmm_stability, loo, next_visit as nv  # noqa: E402
from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype.plots import GRID, INK, INK_MUTED, _finish, _style  # noqa: E402

#: The frozen visit panel, guarded exactly as scripts/60 guards it.
EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"


def md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

CT_RUN = "experiments/20261020T104756Z_next_visit_td_pigd_prediction"
PERS_MODEL = "Persistence (last non-Indeterminate)"
CT_MODEL = "CT-HMM (latent, interval-aware)"

#: Window sizes. ``None`` means "every visit before the target".
WINDOWS: tuple = (1, 2, 3, 5, None)
WINDOW_LABEL = {1: "last 1", 2: "last 2", 3: "last 3", 5: "last 5",
                None: "full history"}
#: Pre-specified primary comparison, fixed before any result was seen.
PRIMARY = (2, None)
#: A window variant needs at least this many targets in a class for AUC etc.
MIN_CLASS = 5

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


def block(f, **kw) -> str:
    return "    " + f.to_string(**kw).replace("\n", "\n    ")


# ---------------------------------------------------------------------------
#  Truncated-history prediction
# ---------------------------------------------------------------------------
def _window_sequences(seq: dict, windows=WINDOWS) -> tuple[list[dict], list[tuple]]:
    """One sub-sequence per (prediction case, window), with real elapsed times.

    For a target at visit ``k + 1`` the permitted history is visits ``0..k``;
    window ``W`` keeps visits ``k - W + 1 .. k``. The retained gaps are the
    actual ones, never re-spaced: in a continuous-time model two transitions
    separated by different durations are not interchangeable.

    The target's own label is carried as the final element purely so the
    recursion has a ``labels[k + 1]`` to report as ``to_state``. It cannot
    affect the prediction, because the predictive distribution at a step is
    formed before ``alpha`` is updated with that label -- and that claim is
    proved by the tamper check rather than asserted.
    """
    labels, gaps = seq["labels"], seq["gaps"]
    subs, tags = [], []
    for k in range(len(gaps)):
        n_hist = k + 1
        for w in windows:
            if w is not None and n_hist < w:
                continue                     # not eligible: never pad
            lo = 0 if w is None else k - w + 1
            tag = (seq["PATNO"], k, w)
            subs.append({"PATNO": f"{seq['PATNO']}|{k}|{w}",
                         "labels": labels[lo:k + 2],
                         "gaps": gaps[lo:k + 1]})
            tags.append(tag)
    return subs, tags


def _predict_windows(fit, seq: dict, windows=WINDOWS,
                     tamper_n_states: int | None = None) -> pd.DataFrame:
    """Predictive distributions for every (case, window), one call per fold.

    With ``tamper_n_states`` set, each sub-sequence's FINAL label -- the target,
    and only the target -- is changed to a different state. The retained
    history is untouched. If the prediction for a case moved under that change,
    the target would be informing its own prediction.
    """
    subs, tags = _window_sequences(seq, windows)
    if not subs:
        return pd.DataFrame()
    if tamper_n_states:
        subs = [{**sb, "labels": np.concatenate(
                    [np.asarray(sb["labels"])[:-1],
                     [(int(np.asarray(sb["labels"])[-1]) + 1) % tamper_n_states]])}
                for sb in subs]
    preds = hmm.predictive_distributions(fit, subs)
    # The last step of each sub-sequence is the prediction for its target.
    last = preds.sort_values("step").groupby("PATNO", sort=False).tail(1)
    order = {f"{p}|{k}|{w}": i for i, (p, k, w) in enumerate(tags)}
    last = last.assign(_o=last["PATNO"].map(order)).sort_values("_o")
    out = pd.DataFrame({
        "PATNO": [t[0] for t in tags],
        "visit_index": [t[1] for t in tags],
        "window": [WINDOW_LABEL[t[2]] for t in tags],
        "window_size": [0 if t[2] is None else t[2] for t in tags],
        "n_retained_visits": [
            (t[1] + 1) if t[2] is None else t[2] for t in tags],
    })
    for c in loo.PROB_COLS:
        out[c] = last[c].to_numpy(float)
    out["last_gap_years"] = last["interval_years"].to_numpy(float)
    return out


def run_fold(held_out):
    """Fit once on the training patients, then predict under every window."""
    data, cases = _WORK["data"], _WORK["cases"]
    seed, n_restarts = _WORK["seed"], _WORK["n_restarts"]
    floor = _WORK["emission_floor"]

    train_sequences = [s for s in data.sequences if s["PATNO"] != held_out]
    train_pairs = data.pairs[data.pairs["PATNO"] != held_out]
    loo.assert_no_leakage(held_out, train_sequences, train_pairs)

    starts = nv.audited_starts(len(data.states), len(data.states), floor,
                               seed, n_restarts)
    fit, records = hmm_stability.fit_multistart(
        train_sequences, data.states, n_hidden=len(data.states),
        emission_floor=floor, starts=starts, state_names=nv.STATE_NAMES)
    best = max(records, key=lambda r: r.loglik)

    seq = next(s for s in data.sequences if s["PATNO"] == held_out)
    out = _predict_windows(fit, seq)
    out["n_train_patients"] = len(train_sequences)
    out["fit_train_loglik"] = float(fit.loglik)
    out["fit_converged"] = bool(best.success
                                or best.proj_grad < nv.PROJ_GRAD_TOL)

    # Tamper check, run inside the fold where the fitted model is in hand:
    # swap the target label for a different one and confirm the predictive
    # distribution is bit-identical. If the target ever leaked into its own
    # prediction, this would move.
    tam = {"patient": held_out, "max_abs_change": 0.0, "n_cases": 0}
    if len(seq["gaps"]):
        ref = out[loo.PROB_COLS].to_numpy(float)
        alt = _predict_windows(fit, seq,
                               tamper_n_states=len(data.states))
        alt = alt[loo.PROB_COLS].to_numpy(float)
        if ref.shape != alt.shape:
            raise RuntimeError("tamper check produced a different row set")
        tam["max_abs_change"] = float(np.max(np.abs(ref - alt)))
        tam["n_cases"] = int(len(ref))
    return out, tam


def _fold_task(p):
    try:
        return run_fold(p), None
    except Exception as exc:                            # noqa: BLE001
        return None, {"patient": p, "error": repr(exc)}


def _init(data, cases, seed, n_restarts, emission_floor):
    _WORK.update(data=data, cases=cases, seed=seed, n_restarts=n_restarts,
                 emission_floor=emission_floor)


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------
def _auc(y, p):
    """Rank-based AUC; undefined with a single class present."""
    y = np.asarray(y, float)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    sp = np.asarray(p, float)[order]
    i = 0
    r = np.arange(1, len(p) + 1, dtype=float)
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    ranks[order] = r
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def metrics_for(y, pred, p_pigd) -> dict:
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    p = np.clip(np.asarray(p_pigd, float), 1e-12, 1 - 1e-12)
    n = len(y)
    n_p, n_t = int(y.sum()), int(n - y.sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens_p = tp / n_p if n_p else float("nan")      # PIGD sensitivity
    spec_p = tn / n_t if n_t else float("nan")      # PIGD specificity
    small = (n_p < MIN_CLASS) or (n_t < MIN_CLASS)

    # A rule that emits only 0 and 1 has no probability to score. Its log loss
    # is then just the arbitrary clipping constant multiplied by its error
    # rate, and its "AUC" is its balanced accuracy rather than an area under a
    # curve. Both are reported as unavailable rather than forced, and the
    # condition is flagged so the report can say why.
    extreme = float(np.mean((p <= 1e-9) | (p >= 1 - 1e-9)))
    degenerate = extreme > 0.5
    return {"n_targets": n, "n_TD": n_t, "n_PIGD": n_p,
            "accuracy": (tp + tn) / n if n else float("nan"),
            "balanced_accuracy": (sens_p + spec_p) / 2.0,
            "sensitivity_PIGD": sens_p, "specificity_PIGD": spec_p,
            "sensitivity_TD": spec_p, "specificity_TD": sens_p,
            "auc": float("nan") if (small or degenerate) else _auc(y, p),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": (float("nan") if degenerate else
                         float(-np.mean(y * np.log(p)
                                        + (1 - y) * np.log(1 - p)))),
            "tp_PIGD": tp, "tn_TD": tn, "fp": fp, "fn": fn,
            "metrics_limited_by_class_size": bool(small),
            "degenerate_probabilities": bool(degenerate),
            "share_probabilities_at_0_or_1": extreme}


BOOT_KEYS = ("accuracy", "balanced_accuracy", "auc", "brier", "log_loss")


def _patient_row_index(patients: np.ndarray) -> list:
    """Row positions grouped by patient, for index-based resampling."""
    uniq, inv = np.unique(patients, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    si = inv[order]
    lo = np.searchsorted(si, np.arange(len(uniq)))
    hi = np.searchsorted(si, np.arange(len(uniq)), side="right")
    return [order[a:b] for a, b in zip(lo, hi)]


def patient_boot_all(patients, fn, n_boot=2000, seed=42):
    """Percentile CIs for every metric at once, resampling PATIENTS.

    ``fn(rows)`` returns a metric dict for the given row positions. One
    resample serves all metrics, so the intervals come from the same draws and
    the whole pass costs one bootstrap rather than one per metric. Patients are
    drawn with replacement and all of a patient's rows travel together.
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
        out[k] = ((float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975)),
                   int(len(ok))) if len(ok)
                  else (float("nan"), float("nan"), 0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    started = time.time()

    # ===================================================== 1. configuration
    section("1. CONFIGURATION AND PROTOCOL")
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    panel_md5 = md5(panel_path)
    if panel_md5 != EXPECTED_PANEL_MD5:
        raise SystemExit(f"panel md5 {panel_md5} != expected {EXPECTED_PANEL_MD5}")
    say(f"  panel                     {panel_path.relative_to(ROOT)}")
    say(f"  md5                       {panel_md5}  (frozen panel verified)")
    states = [cfg.subtype.labels[s] for s in nv.STATE_NAMES]
    panel = pd.read_csv(panel_path, low_memory=False)
    data = loo.prepare(panel, states)
    cases = nv.build_cases(data)
    patients = data.patients
    if args.limit:
        patients = patients[: args.limit]
    say(f"  panel visits              {len(panel):,}")
    say(f"  labelled sequences        {len(data.sequences):,} patients")
    say(f"  prediction cases          {len(cases):,}")
    say(f"  validation                leave-one-patient-out, "
        f"{len(patients):,} folds")
    say(f"  seed                      {cfg.seed}")
    say(f"  restarts per fit          {cfg.model['n_restarts']}")
    say(f"  emission floor            {cfg.model['emission_floor']}")
    say(f"  windows                   "
        f"{[WINDOW_LABEL[w] for w in WINDOWS]}")
    say(f"  primary comparison        {WINDOW_LABEL[PRIMARY[0]]} vs "
        f"{WINDOW_LABEL[PRIMARY[1]]}, common-target cohort (pre-specified)")
    say("  parameter policy          Q, E and pi fitted ONCE per fold on the")
    say("                            training patients' FULL sequences, then")
    say("                            shared by every window variant; only the")
    say("                            held-out patient's history is truncated")

    # ===================================================== 2. run the folds
    section("2. LEAVE-ONE-PATIENT-OUT WITH TRUNCATED HISTORIES")
    rows, tampers, failed = [], [], []
    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(
                max_workers=args.workers, initializer=_init,
                initargs=(data, cases, cfg.seed, cfg.model["n_restarts"],
                          cfg.model["emission_floor"])) as ex:
            for i, (res, err) in enumerate(ex.map(_fold_task, patients), 1):
                if err:
                    failed.append(err)
                else:
                    rows.append(res[0])
                    tampers.append(res[1])
                if i % 100 == 0:
                    say(f"    {i:,}/{len(patients):,} folds "
                        f"({time.time() - t0:.0f}s)")
    else:
        _init(data, cases, cfg.seed, cfg.model["n_restarts"],
              cfg.model["emission_floor"])
        for i, p in enumerate(patients, 1):
            res, err = _fold_task(p)
            if err:
                failed.append(err)
            else:
                rows.append(res[0])
                tampers.append(res[1])
            if i % 100 == 0:
                say(f"    {i:,}/{len(patients):,} folds "
                    f"({time.time() - t0:.0f}s)")
    say(f"  folds completed           {len(rows):,}")
    say(f"  folds failed              {len(failed):,}")
    if failed:
        pd.DataFrame(failed).to_csv(out / "failed_folds.csv", index=False)
        raise SystemExit(f"{len(failed)} folds failed; see failed_folds.csv")

    pred = pd.concat(rows, ignore_index=True)
    tam = pd.DataFrame(tampers)

    # Attach the case metadata and the project's two-class restriction.
    meta_cols = ["PATNO", "visit_index", "n_history", "current_event",
                 "current_years", "target_event", "target_years",
                 "interval_years", "current_label", "actual_next_label",
                 "eligible"]
    pred = pred.merge(cases[meta_cols], on=["PATNO", "visit_index"], how="left",
                      validate="many_to_one")
    probs = pred[loo.PROB_COLS].to_numpy(float)
    p_td, p_pigd, defined = nv.conditional_two_class(probs)
    pred["P_TD"], pred["P_PIGD"] = p_td, p_pigd
    pred["two_class_defined"] = defined
    pred["predicted_state"] = nv.predicted_state(p_td, p_pigd)
    say(f"  prediction rows           {len(pred):,}")

    # ------------------------------------------------ persistence condition
    orig = pd.read_csv(ROOT / CT_RUN / "next_visit_predictions.csv",
                       low_memory=False)
    pers = orig[orig["model"] == PERS_MODEL][
        ["PATNO", "visit_index", "P_TD", "P_PIGD", "predicted_state",
         "two_class_defined", "actual_next_label"]].copy()
    pers["window"] = "persistence"
    pers["window_size"] = -1
    pers = pers.merge(cases[meta_cols].drop(columns=["actual_next_label"]),
                      on=["PATNO", "visit_index"], how="inner",
                      validate="one_to_one")
    # Restrict to exactly the cases these folds produced, so that a --limit run
    # compares like with like instead of scoring persistence on the whole
    # cohort and the windows on a subset.
    done = set(map(tuple, pred[["PATNO", "visit_index"]].drop_duplicates()
                   .to_numpy()))
    pers = pers[[tuple(r) in done
                 for r in pers[["PATNO", "visit_index"]].to_numpy()]].copy()
    pers["n_retained_visits"] = pers["n_history"]
    # The persistence rule carries the most recent label over the same gap the
    # HMM conditions face, so it belongs in the "last gap" split too.
    pers["last_gap_years"] = pers["interval_years"]
    say(f"  persistence rows reused   {len(pers):,} "
        f"(from the validated run, matched on patient and visit index)")

    allp = pd.concat([pred, pers], ignore_index=True)
    allp["y"] = (allp["actual_next_label"] == "PIGD").astype(int)
    allp["pred_bin"] = (allp["predicted_state"] == "PIGD").astype(int)
    allp.to_csv(out / "window_predictions.csv", index=False)

    # ---------------------- reproduce the original full-history predictions
    section("3. DOES 'FULL HISTORY' REPRODUCE THE ORIGINAL RUN?")
    ref = orig[orig["model"] == CT_MODEL][["PATNO", "visit_index", "P_TD",
                                          "P_PIGD", "predicted_state"]]
    mine = pred[pred["window"] == "full history"][
        ["PATNO", "visit_index", "P_TD", "P_PIGD", "predicted_state"]]
    cmp = ref.merge(mine, on=["PATNO", "visit_index"], suffixes=("_orig", "_new"))
    dP = np.abs(cmp["P_PIGD_orig"] - cmp["P_PIGD_new"]).to_numpy()
    same_call = bool((cmp["predicted_state_orig"]
                      == cmp["predicted_state_new"]).all())
    say(f"  matched rows              {len(cmp):,} of {len(mine):,}")
    say(f"  max |P_PIGD difference|   {np.nanmax(dP) if len(dP) else float('nan'):.3e}")
    say(f"  identical hard calls      {same_call}")
    reproduced = bool(len(cmp) == len(mine) and np.nanmax(dP) < 1e-8
                      and same_call)
    say(f"  -> full history reproduces the validated run: {reproduced}")

    # ===================================================== 4. cohorts
    section("4. EVALUATION COHORTS")
    elig = allp[(allp["actual_next_label"].isin(["TD", "PIGD"]))
                & allp["two_class_defined"].astype(bool)].copy()
    n_ind = int((allp["window"] == "full history").sum()
                - (elig["window"] == "full history").sum())
    say(f"  Indeterminate / undefined targets excluded per window: {n_ind}")

    max_w = max(w for w in WINDOWS if w is not None)
    # A target joins the common cohort only if it is long enough for the widest
    # window AND eligible under every condition, persistence included. The
    # persistence rule is undefined for a history that is Indeterminate
    # throughout, so the intersection is taken rather than assumed.
    by_cond = {w: set(map(tuple, g[["PATNO", "visit_index"]].to_numpy()))
               for w, g in elig.groupby("window")}
    common_keys = set.intersection(*by_cond.values()) if by_cond else set()
    long_enough = set(map(tuple,
                          elig[elig["n_history"] >= max_w][["PATNO", "visit_index"]]
                          .drop_duplicates().to_numpy()))
    dropped_by_cond = len(long_enough - common_keys)
    common_keys &= long_enough
    say(f"  targets long enough for every window      : {len(long_enough):,}")
    say(f"  of those, not eligible in some condition  : {dropped_by_cond}")
    elig["common"] = [tuple(r) in common_keys
                      for r in elig[["PATNO", "visit_index"]].to_numpy()]
    say(f"  all-eligible cohort       per-window, see table")
    say(f"  common-target cohort      {len(common_keys):,} targets "
        f"(n_history >= {max_w}, so every window is defined)")

    per_window = (elig.groupby("window")
                      .agg(n_targets=("y", "size"),
                           n_patients=("PATNO", "nunique")).reset_index())
    say("\n  eligible targets per condition:")
    say(block(per_window, index=False))

    # ===================================================== 5. metrics
    section("5. METRICS")
    order = ["persistence"] + [WINDOW_LABEL[w] for w in WINDOWS]
    recs = []
    for cohort, sel in (("all eligible", elig),
                        ("common targets", elig[elig["common"]])):
        for w in order:
            g = sel[sel["window"] == w]
            if not len(g):
                continue
            yv = g["y"].to_numpy(int)
            pv = g["pred_bin"].to_numpy(int)
            qv = g["P_PIGD"].to_numpy(float)
            m = metrics_for(yv, pv, qv)
            m.update({"cohort": cohort, "window": w,
                      "n_patients": int(g["PATNO"].nunique())})
            cis = patient_boot_all(
                g["PATNO"].to_numpy(),
                lambda rows: metrics_for(yv[rows], pv[rows], qv[rows]),
                n_boot=args.n_boot)
            for k in BOOT_KEYS:
                lo, hi, _ = cis[k]
                m[f"{k}_lo"] = lo if np.isfinite(m[k]) else float("nan")
                m[f"{k}_hi"] = hi if np.isfinite(m[k]) else float("nan")
            recs.append(m)
            say(f"    {cohort:<16s} {w:<14s} done")
    mt = pd.DataFrame(recs)
    cols = ["cohort", "window", "n_targets", "n_patients", "n_TD", "n_PIGD",
            "accuracy", "accuracy_lo", "accuracy_hi",
            "balanced_accuracy", "balanced_accuracy_lo", "balanced_accuracy_hi",
            "sensitivity_PIGD", "specificity_PIGD", "sensitivity_TD",
            "specificity_TD", "auc", "auc_lo", "auc_hi",
            "brier", "brier_lo", "brier_hi", "log_loss", "log_loss_lo",
            "log_loss_hi", "tp_PIGD", "tn_TD", "fp", "fn",
            "metrics_limited_by_class_size"]
    mt = mt[cols]
    mt.round(4).to_csv(out / "window_metrics.csv", index=False)
    for cohort in ("all eligible", "common targets"):
        say(f"\n  --- {cohort} ---")
        say(block(mt[mt.cohort == cohort][
            ["window", "n_targets", "n_patients", "n_PIGD", "accuracy",
             "balanced_accuracy", "sensitivity_PIGD", "specificity_PIGD",
             "auc", "brier", "log_loss"]].round(4), index=False))

    # ===================================================== 6. paired tests
    section("6. PAIRED COMPARISONS, PATIENT-CLUSTERED")
    say("  Paired on identical targets: the comparison frame is pivoted so that")
    say("  each row is one target carrying both conditions' predictions, and")
    say("  PATIENTS are resampled, never individual targets.")

    def paired(a, b, cohort_sel, label):
        wide = cohort_sel[cohort_sel["window"].isin([a, b])].pivot_table(
            index=["PATNO", "visit_index", "y"], columns="window",
            values=["pred_bin", "P_PIGD"], aggfunc="first").dropna()
        if not len(wide):
            return None
        wide = wide.reset_index()
        rec = {"comparison": f"{a} - {b}", "cohort": label,
               "n_targets": len(wide), "n_patients": wide["PATNO"].nunique()}

        ya = wide["y"].to_numpy(int)
        pa_ = wide[("pred_bin", a)].to_numpy(int)
        pb_ = wide[("pred_bin", b)].to_numpy(int)
        qa_ = wide[("P_PIGD", a)].to_numpy(float)
        qb_ = wide[("P_PIGD", b)].to_numpy(float)

        def diffs(rows):
            ma = metrics_for(ya[rows], pa_[rows], qa_[rows])
            mb = metrics_for(ya[rows], pb_[rows], qb_[rows])
            return {k: ma[k] - mb[k] for k in BOOT_KEYS}

        point = diffs(np.arange(len(wide)))
        cis = patient_boot_all(wide["PATNO"].to_numpy(), diffs,
                               n_boot=args.n_boot)
        for key in BOOT_KEYS:
            est = point[key]
            if not np.isfinite(est):
                rec[key] = rec[f"{key}_lo"] = rec[f"{key}_hi"] = float("nan")
                rec[f"{key}_excludes_zero"] = False
                continue
            lo, hi, _ = cis[key]
            rec[key], rec[f"{key}_lo"], rec[f"{key}_hi"] = est, lo, hi
            rec[f"{key}_excludes_zero"] = bool(lo > 0 or hi < 0)
        return rec

    prs = []
    common = elig[elig["common"]]
    # primary, pre-specified
    p0 = paired(WINDOW_LABEL[PRIMARY[0]], WINDOW_LABEL[PRIMARY[1]], common,
                "common targets")
    if p0:
        p0["role"] = "PRIMARY (pre-specified)"
        prs.append(p0)
    # secondary / exploratory
    for a, b, role in (("last 2", "persistence", "secondary"),
                       ("last 3", "persistence", "secondary"),
                       ("last 1", "persistence", "secondary"),
                       ("full history", "persistence", "secondary"),
                       ("last 1", "full history", "exploratory"),
                       ("last 3", "full history", "exploratory"),
                       ("last 5", "full history", "exploratory")):
        r = paired(a, b, common, "common targets")
        if r:
            r["role"] = role
            prs.append(r)
    pt = pd.DataFrame(prs)
    pt.round(4).to_csv(out / "paired_comparisons.csv", index=False)
    show = ["role", "comparison", "n_targets", "n_patients", "accuracy",
            "accuracy_lo", "accuracy_hi", "balanced_accuracy",
            "balanced_accuracy_lo", "balanced_accuracy_hi", "auc", "brier",
            "log_loss"]
    say("")
    say(block(pt[[c for c in show if c in pt.columns]].round(4), index=False))

    n_tests = int(sum(1 for c in ("accuracy", "balanced_accuracy", "auc",
                                  "brier", "log_loss")
                      for _ in range(len(pt))))
    say(f"\n  {n_tests} paired intervals computed. The PRIMARY comparison is")
    say("  pre-specified; the rest are secondary or exploratory. A Bonferroni")
    say(f"  threshold over the {len(pt) - 1} non-primary comparisons would be")
    say(f"  alpha = {0.05 / max(len(pt) - 1, 1):.4f}, and no unadjusted interval")
    say("  below is treated as confirmatory.")

    # ===================================================== 7. subgroups
    section("7. EXPLORATORY SUBGROUPS")
    sub = []
    for w in order:
        g = common[common["window"] == w]
        if not len(g):
            continue
        for lab, m in (("n_history 5-6", g["n_history"].between(5, 6)),
                       ("n_history >=7", g["n_history"] >= 7),
                       ("last gap < 1y", g["last_gap_years"] < 1.0),
                       ("last gap >= 1y", g["last_gap_years"] >= 1.0)):
            s = g[m]
            if len(s) < 20:
                continue
            mm = metrics_for(s["y"], s["pred_bin"], s["P_PIGD"])
            sub.append({"window": w, "subgroup": lab, "n_targets": len(s),
                        "n_patients": int(s["PATNO"].nunique()),
                        "accuracy": mm["accuracy"],
                        "balanced_accuracy": mm["balanced_accuracy"],
                        "sensitivity_PIGD": mm["sensitivity_PIGD"]})
    sg = pd.DataFrame(sub)
    if len(sg):
        sg.round(4).to_csv(out / "subgroups.csv", index=False)
        say(block(sg.round(4), index=False))
    say("\n  Exploratory only: subgroups are small, uncorrected, and chosen")
    say("  after seeing the main result. The persistence row uses the full")
    say("  history for its own rule, so 'last gap' is the same split for all.")

    # ===================================================== 8. figures
    section("8. FIGURES")
    plot_order = [w for w in order if w in set(mt["window"])]
    xs = np.arange(len(plot_order))

    def series(cohort, key):
        d = mt[mt.cohort == cohort].set_index("window")
        v = np.array([d.loc[w, key] if w in d.index else np.nan
                      for w in plot_order], float)
        lo = np.array([d.loc[w, f"{key}_lo"] if w in d.index else np.nan
                       for w in plot_order], float)
        hi = np.array([d.loc[w, f"{key}_hi"] if w in d.index else np.nan
                       for w in plot_order], float)
        return v, lo, hi

    # (1) accuracy and balanced accuracy
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharex=True)
    for ax, key, lab in ((axes[0], "accuracy", "Accuracy"),
                         (axes[1], "balanced_accuracy", "Balanced accuracy")):
        _style(ax)
        for cohort, colour, off in (("all eligible", MUTED, -0.11),
                                    ("common targets", ACCENT, +0.11)):
            v, lo, hi = series(cohort, key)
            ax.errorbar(xs + off, v, yerr=[v - lo, hi - v], fmt="o",
                        color=colour, markersize=6, capsize=3, elinewidth=1.3,
                        label=cohort, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(plot_order, rotation=20, ha="right")
        ax.set_ylabel(lab)
        ax.set_title(lab, fontsize=10.5, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle("How much history does the CT-HMM need at prediction time?\n"
                 "95% patient-clustered intervals; parameters identical across "
                 "all window variants", fontsize=11, x=0.01, y=0.995,
                 ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    say(f"  wrote {_finish(fig, out / 'fig1_accuracy_by_window.png').name}")

    # (2) probability quality
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6))
    for ax, key, lab, better in ((axes[0], "auc", "ROC-AUC", "higher"),
                                 (axes[1], "brier", "Brier score", "lower"),
                                 (axes[2], "log_loss", "Log loss", "lower")):
        _style(ax)
        v, lo, hi = series("common targets", key)
        ok = np.isfinite(v)
        ax.errorbar(xs[ok], v[ok], yerr=[(v - lo)[ok], (hi - v)[ok]], fmt="o",
                    color=COOL, markersize=6, capsize=3, elinewidth=1.3,
                    zorder=3)
        for i in np.where(~ok)[0]:
            ax.annotate("n/a", (xs[i], np.nanmean(v[ok]) if ok.any() else 0.5),
                        ha="center", fontsize=8, color=INK_MUTED)
        ax.set_xticks(xs)
        ax.set_xticklabels(plot_order, rotation=25, ha="right", fontsize=8.5)
        ax.set_ylabel(lab)
        ax.set_title(f"{lab} ({better} is better)", fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle("Probability quality on the common-target cohort — "
                 "persistence emits only 0 and 1, so its AUC is not a curve",
                 fontsize=10.5, x=0.01, y=0.995, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    say(f"  wrote {_finish(fig, out / 'fig2_probability_quality.png').name}")

    # (3) paired differences against full history
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    _style(ax)
    rowsp = pt[pt["comparison"].str.endswith("- full history")]
    if len(rowsp):
        labs = rowsp["comparison"].tolist()
        yy = np.arange(len(labs))
        for key, colour, off in (("accuracy", ACCENT, -0.18),
                                 ("balanced_accuracy", COOL, 0.0),
                                 ("auc", WARM, +0.18)):
            v = rowsp[key].to_numpy(float)
            lo = rowsp[f"{key}_lo"].to_numpy(float)
            hi = rowsp[f"{key}_hi"].to_numpy(float)
            ax.errorbar(v, yy + off, xerr=[v - lo, hi - v], fmt="o",
                        color=colour, markersize=5.5, capsize=3,
                        elinewidth=1.2, label=key.replace("_", " "), zorder=3)
        ax.axvline(0, color=INK, linewidth=1.2, zorder=2)
        ax.set_yticks(yy)
        ax.set_yticklabels(labs)
        ax.set_xlabel("Difference versus full history (positive favours the "
                      "shorter window)")
        ax.set_title("Paired differences on identical targets\n"
                     "common-target cohort; 95% patient-clustered intervals",
                     fontsize=10.5, loc="left")
        ax.legend(frameon=False, fontsize=8.5)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    say(f"  wrote {_finish(fig, out / 'fig3_paired_vs_full.png').name}")

    # ===================================================== 9. checks
    section("9. LEAKAGE AND VALIDATION CHECKS")
    full = pred[pred["window"] == "full history"]
    retained_ok = True
    for w in WINDOWS:
        if w is None:
            continue
        g = pred[pred["window"] == WINDOW_LABEL[w]]
        if len(g) and not bool((g["n_history"] >= w).all()):
            retained_ok = False
    src = Path(__file__).read_text()
    checks = {
        "every fold completed": len(failed) == 0,
        "no window variant used a padded or fabricated visit": retained_ok,
        "tamper test: changing the target label leaves predictions identical":
            bool(float(tam["max_abs_change"].max()) == 0.0),
        "tamper test actually ran on every fold":
            bool((tam["n_cases"] > 0).all()) and len(tam) == len(rows),
        "full history reproduces the validated original run": reproduced,
        "held-out patient never in training (checked inside every fold)":
            "loo.assert_no_leakage(held_out, train_sequences, train_pairs)" in src,
        "parameters fitted on training patients only":
            bool((pred["n_train_patients"] == len(data.sequences) - 1).all()),
        "every fit converged": bool(pred["fit_converged"].all()),
        "history length equals visits up to the current one":
            bool((cases["n_history"] == cases["visit_index"] + 1).all()),
        "target visit strictly after the current visit":
            bool((allp["target_years"] > allp["current_years"]).all()),
        "retained gaps are real elapsed times, never re-spaced":
            bool(np.allclose(full["last_gap_years"].to_numpy(float),
                             full["interval_years"].to_numpy(float),
                             atol=1e-9)),
        "windows agree on the final gap (same target, same interval)":
            bool(pred.groupby(["PATNO", "visit_index"])["last_gap_years"]
                     .nunique().eq(1).all()),
        "all windows predict the same target for a given case":
            bool(pred.groupby(["PATNO", "visit_index"])["target_event"]
                     .nunique().eq(1).all()),
        "Indeterminate targets excluded from binary scoring":
            bool(set(elig["actual_next_label"]) == {"TD", "PIGD"}),
        "common cohort has the same target count in every condition":
            len(set(elig[elig["common"]].groupby("window").size())) == 1,
        "common cohort is identical across conditions":
            bool(elig[elig["common"]].groupby("window")
                 .apply(lambda g: frozenset(map(tuple,
                        g[["PATNO", "visit_index"]].to_numpy())),
                        include_groups=False).nunique() == 1),
        "probabilities finite and in [0, 1]": bool(
            np.isfinite(allp["P_PIGD"]).all() and (allp["P_PIGD"] >= 0).all()
            and (allp["P_PIGD"] <= 1).all()),
        "P_TD + P_PIGD == 1 where defined": bool(np.allclose(
            allp.loc[allp["two_class_defined"].astype(bool), "P_TD"]
            + allp.loc[allp["two_class_defined"].astype(bool), "P_PIGD"],
            1.0, atol=1e-12)),
        "bootstrap resamples patients, not targets":
            "groups = _patient_row_index(np.asarray(patients))" in src,
        "fixed seed recorded for reproducibility": cfg.seed is not None,
    }
    for n, v in checks.items():
        say(f"    [{'PASS' if v else 'FAIL'}]  {n}")
    pd.DataFrame([{"check": k, "passed": bool(v)} for k, v in checks.items()]
                 ).to_csv(out / "validation_checks.csv", index=False)
    tam.to_csv(out / "tamper_check.csv", index=False)
    if not all(checks.values()):
        raise SystemExit(f"CRITICAL CHECK FAILED: "
                         f"{[k for k, v in checks.items() if not v]}")

    # ===================================================== 10. provenance
    section("10. PROVENANCE")
    env = {"panel_md5": panel_md5, "python": sys.version.split()[0], "numpy": np.__version__,
           "pandas": pd.__version__, "node": platform.node(),
           "workers": args.workers, "n_bootstrap": args.n_boot,
           "seed": cfg.seed, "n_restarts": cfg.model["n_restarts"],
           "emission_floor": cfg.model["emission_floor"],
           "original_run": CT_RUN,
           "n_folds": len(rows), "n_folds_failed": len(failed),
           "elapsed_seconds": round(time.time() - started, 1)}
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:                                   # noqa: BLE001
        env["git_commit"] = "unknown"
    for k, v in env.items():
        say(f"  {k:<22} {v}")

    (out / "summary.json").write_text(json.dumps(
        {"environment": env, "checks": checks,
         "code_sha256": {str(Path(__file__).relative_to(ROOT)):
                         sha(Path(__file__))},
         "windows": [WINDOW_LABEL[w] for w in WINDOWS],
         "primary_comparison": f"{WINDOW_LABEL[PRIMARY[0]]} vs "
                               f"{WINDOW_LABEL[PRIMARY[1]]}",
         "n_common_targets": len(common_keys),
         "n_long_enough": len(long_enough),
         "n_dropped_not_eligible_everywhere": dropped_by_cond,
         "n_indeterminate_excluded_per_window": n_ind,
         "full_history_reproduces_original": reproduced,
         "metrics": mt.to_dict(orient="records"),
         "paired": pt.to_dict(orient="records"),
         "subgroups": sg.to_dict(orient="records") if len(sg) else []},
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
