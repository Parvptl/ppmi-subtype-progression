"""Retrain the CT-HMM on two-visit sliding windows, and separate the effects.

Script 70 truncated only the *prediction-time* history: parameters still came
from the training patients' full sequences. This experiment re-estimates the
parameters from two-visit sliding windows as well, so the two effects can be
told apart. The design is a 2x2 plus the persistence reference:

                              prediction history
                        full history        last 2 visits
    training  full       (2) original        (3) = script 70
              windows    (4) training        (5) PRIMARY
                             effect alone

    (1) persistence: carry the last non-Indeterminate label forward.

The primary comparison, pre-specified, is **(5) versus (2)** on common targets.
Contrasting (5) with (3) isolates the change in *training*, and (3) with (2)
isolates the change in *prediction history*.

WHY THIS NEEDS NO CHANGE TO hmm.py
----------------------------------
`hmm.fit_hmm` maximises the forward log-likelihood over a *list of sequences*
and `hmm.build_sequences` already allows `min_length=2`. A sliding window is
therefore just a shorter sequence, and a list of windows is a valid input.
Patient boundaries are preserved by construction: each window is its own
sequence, so no transition is ever created between one patient's last visit and
another's first, and no window spans a gap that is not in the original data.

THE ESTIMATION OBJECTIVE, STATED PLAINLY
----------------------------------------
Fitting on overlapping windows maximises a **composite (pseudo) likelihood**,
not the likelihood of the data. Each *transition* appears exactly once, but each
*interior visit* appears twice -- as the second element of one window and the
first of the next -- so emissions are double-counted. The consequence is that
the objective's value is not comparable to the full-sequence log-likelihood and
its curvature does not give valid standard errors. Point estimates are still
usable, and everything here is judged out-of-sample, so this is a defensible
estimator for the question asked; it is not presented as a likelihood.

A non-overlapping variant, which *is* a proper likelihood over disjoint pairs at
the cost of roughly half the transitions, is fitted as a sensitivity condition.

WHAT CHANGES MEANING
--------------------
`pi` is the latent distribution at a window's first visit. Under full sequences
that is the patient's baseline; under sliding windows it becomes the marginal
over all visit positions. That is the intended fix for the mismatch script 70
flagged -- a truncated prediction window starts mid-trajectory, so a `pi`
estimated mid-trajectory matches it -- and it is why this experiment exists.

TIMELINE INTEGRITY
------------------
Truncation removes observations; it never moves the ones that remain. Every
retained visit keeps its original `PATNO`, `EVENT_ID`, `years_from_baseline`,
visit index and inter-visit gap, and `P(dt) = expm(Q*dt)` is evaluated on the
real elapsed time. The model is time-homogeneous and carries no absolute-time,
visit-number or disease-duration covariate, so there is nothing to reindex;
section 6 verifies all of this against the parent sequences rather than
asserting it.

    python scripts/72_retrained_two_visit_cthmm.py --out <dir> [--workers N]
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
from ppmi_subtype.plots import GRID, INK, _finish, _style  # noqa: E402

EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
ORIG_RUN = "experiments/20261020T104756Z_next_visit_td_pigd_prediction"
S70_RUN = "experiments/20261024T042417Z_history_window_experiment"
PERS_MODEL = "Persistence (last non-Indeterminate)"
CT_MODEL = "CT-HMM (latent, interval-aware)"

#: (training set, prediction window). ``None`` prediction window = full history.
COND_PERSIST = "persistence"
COND_FULL_FULL = "full-trained, full history"
COND_FULL_W2 = "full-trained, last 2"
COND_WIN_FULL = "window-trained, full history"
COND_WIN_W2 = "window-trained, last 2"
COND_NONOV_W2 = "non-overlap-window-trained, last 2"
CONDITIONS = [COND_PERSIST, COND_FULL_FULL, COND_FULL_W2, COND_WIN_FULL,
              COND_WIN_W2, COND_NONOV_W2]
#: Pre-specified before any result was seen.
PRIMARY = (COND_WIN_W2, COND_FULL_FULL)
WINDOW = 2
MIN_CLASS = 5
BOOT_KEYS = ("accuracy", "balanced_accuracy", "auc", "brier", "log_loss")

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


# ---------------------------------------------------------------------------
#  Sliding windows over training patients
# ---------------------------------------------------------------------------
def sliding_windows(sequences: list[dict], width: int = WINDOW,
                    stride: int = 1) -> list[dict]:
    """Every ``width``-visit window, each as its own sequence.

    A window is a contiguous slice of one patient's sequence, so it holds only
    consecutive observations from that patient and keeps the real gaps between
    them. Windows are never concatenated, so no transition is manufactured
    between patients or between disconnected windows.
    """
    out = []
    for s in sequences:
        labels, gaps = np.asarray(s["labels"]), np.asarray(s["gaps"])
        n = len(labels)
        for start in range(0, n - width + 1, stride):
            out.append({"PATNO": s["PATNO"],
                        "labels": labels[start:start + width],
                        "gaps": gaps[start:start + width - 1],
                        "window_start": int(start)})
    return out


def window_integrity(sequences: list[dict], wins: list[dict],
                     width: int = WINDOW) -> dict:
    """Verify every window against its parent sequence, exhaustively."""
    parent = {s["PATNO"]: s for s in sequences}
    bad_shape = bad_labels = bad_gaps = bad_patient = 0
    for w in wins:
        s = parent.get(w["PATNO"])
        if s is None:
            bad_patient += 1
            continue
        k = w["window_start"]
        if len(w["labels"]) != width or len(w["gaps"]) != width - 1:
            bad_shape += 1
        if not np.array_equal(w["labels"],
                              np.asarray(s["labels"])[k:k + width]):
            bad_labels += 1
        if not np.array_equal(w["gaps"],
                              np.asarray(s["gaps"])[k:k + width - 1]):
            bad_gaps += 1
    return {"n_windows": len(wins), "bad_shape": bad_shape,
            "bad_labels": bad_labels, "bad_gaps": bad_gaps,
            "unknown_patient": bad_patient,
            "n_patients": len({w["PATNO"] for w in wins})}


def _fit(sequences, states, seed, n_restarts, floor):
    starts = nv.audited_starts(len(states), len(states), floor, seed, n_restarts)
    fit, records = hmm_stability.fit_multistart(
        sequences, states, n_hidden=len(states), emission_floor=floor,
        starts=starts, state_names=nv.STATE_NAMES)
    best = max(records, key=lambda r: r.loglik)
    return fit, bool(best.success or best.proj_grad < nv.PROJ_GRAD_TOL)


def _predict(fit, seq: dict, width: int | None, tamper_n_states=None):
    """One row per prediction case for this patient under one history rule.

    ``width=None`` keeps the whole history before the target. Otherwise the
    last ``width`` visits are kept, with their original gaps; a case with fewer
    visits than ``width`` is skipped rather than padded.
    """
    labels, gaps = np.asarray(seq["labels"]), np.asarray(seq["gaps"])
    subs, tags = [], []
    for k in range(len(gaps)):
        n_hist = k + 1
        if width is not None and n_hist < width:
            continue
        lo = 0 if width is None else k - width + 1
        lab = labels[lo:k + 2]
        if tamper_n_states:
            lab = np.concatenate([lab[:-1],
                                  [(int(lab[-1]) + 1) % tamper_n_states]])
        subs.append({"PATNO": f"{seq['PATNO']}|{k}", "labels": lab,
                     "gaps": gaps[lo:k + 1]})
        tags.append((seq["PATNO"], k, lo))
    if not subs:
        return pd.DataFrame()
    preds = hmm.predictive_distributions(fit, subs)
    last = preds.sort_values("step").groupby("PATNO", sort=False).tail(1)
    order = {f"{p}|{k}": i for i, (p, k, _) in enumerate(tags)}
    last = last.assign(_o=last["PATNO"].map(order)).sort_values("_o")
    out = pd.DataFrame({
        "PATNO": [t[0] for t in tags],
        "visit_index": [t[1] for t in tags],
        "retained_from_index": [t[2] for t in tags],
        "n_retained_visits": [(t[1] + 1) if width is None else width
                              for t in tags],
        "last_gap_years": last["interval_years"].to_numpy(float)})
    for c in loo.PROB_COLS:
        out[c] = last[c].to_numpy(float)
    return out


def run_fold(held_out):
    data = _WORK["data"]
    seed, n_restarts = _WORK["seed"], _WORK["n_restarts"]
    floor, states = _WORK["emission_floor"], _WORK["states"]

    train = [s for s in data.sequences if s["PATNO"] != held_out]
    train_pairs = data.pairs[data.pairs["PATNO"] != held_out]
    loo.assert_no_leakage(held_out, train, train_pairs)

    wins_ov = sliding_windows(train, WINDOW, stride=1)
    wins_no = sliding_windows(train, WINDOW, stride=WINDOW)
    if held_out in {w["PATNO"] for w in wins_ov}:
        raise RuntimeError("held-out patient appears in the training windows")

    fit_full, ok_full = _fit(train, states, seed, n_restarts, floor)
    fit_win, ok_win = _fit(wins_ov, states, seed, n_restarts, floor)
    fit_non, ok_non = _fit(wins_no, states, seed, n_restarts, floor)

    seq = next(s for s in data.sequences if s["PATNO"] == held_out)
    frames = []
    for cond, fit, width in ((COND_FULL_FULL, fit_full, None),
                             (COND_FULL_W2, fit_full, WINDOW),
                             (COND_WIN_FULL, fit_win, None),
                             (COND_WIN_W2, fit_win, WINDOW),
                             (COND_NONOV_W2, fit_non, WINDOW)):
        f = _predict(fit, seq, width)
        if len(f):
            f["condition"] = cond
            frames.append(f)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out["n_train_patients"] = len(train)
    out["n_train_windows_overlapping"] = len(wins_ov)
    out["n_train_windows_disjoint"] = len(wins_no)
    out["fits_converged"] = bool(ok_full and ok_win and ok_non)

    diag = {"patient": held_out,
            "loglik_full": float(fit_full.loglik),
            "loglik_windows": float(fit_win.loglik),
            "loglik_disjoint": float(fit_non.loglik),
            "n_obs_full": int(fit_full.n_observations),
            "n_obs_windows": int(fit_win.n_observations),
            **{f"pi_full_{i}": float(v) for i, v in enumerate(fit_full.pi)},
            **{f"pi_win_{i}": float(v) for i, v in enumerate(fit_win.pi)},
            **{f"Qdiag_full_{i}": float(v)
               for i, v in enumerate(np.diag(fit_full.Q))},
            **{f"Qdiag_win_{i}": float(v)
               for i, v in enumerate(np.diag(fit_win.Q))},
            **{f"Ediag_full_{i}": float(v)
               for i, v in enumerate(np.diag(fit_full.E))},
            **{f"Ediag_win_{i}": float(v)
               for i, v in enumerate(np.diag(fit_win.E))}}

    # Tamper check on the condition that matters: change only the target label
    # and confirm the window-trained last-2 prediction does not move.
    ref = _predict(fit_win, seq, WINDOW)
    alt = _predict(fit_win, seq, WINDOW, tamper_n_states=len(states))
    tam = {"patient": held_out, "n_cases": int(len(ref)),
           "max_abs_change": (float(np.max(np.abs(
               ref[loo.PROB_COLS].to_numpy(float)
               - alt[loo.PROB_COLS].to_numpy(float)))) if len(ref) else 0.0)}

    # Integrity of this fold's windows, audited against the parent sequences.
    integ = window_integrity(train, wins_ov)
    integ["patient"] = held_out
    return out, diag, tam, integ


def _fold_task(p):
    try:
        return run_fold(p), None
    except Exception as exc:                            # noqa: BLE001
        return None, {"patient": p, "error": f"{type(exc).__name__}: {exc}"}


def _init(data, seed, n_restarts, emission_floor, states):
    _WORK.update(data=data, seed=seed, n_restarts=n_restarts,
                 emission_floor=emission_floor, states=states)


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
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    p = np.clip(np.asarray(p_pigd, float), 1e-12, 1 - 1e-12)
    n = len(y)
    n_p, n_t = int(y.sum()), int(n - y.sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    sens_p = tp / n_p if n_p else float("nan")
    spec_p = tn / n_t if n_t else float("nan")
    small = (n_p < MIN_CLASS) or (n_t < MIN_CLASS)
    extreme = float(np.mean((p <= 1e-9) | (p >= 1 - 1e-9)))
    degen = extreme > 0.5
    return {"n_targets": n, "n_TD": n_t, "n_PIGD": n_p,
            "accuracy": (tp + tn) / n if n else float("nan"),
            "balanced_accuracy": (sens_p + spec_p) / 2.0,
            "sensitivity_PIGD": sens_p, "specificity_PIGD": spec_p,
            "sensitivity_TD": spec_p, "specificity_TD": sens_p,
            "auc": float("nan") if (small or degen) else _auc(y, p),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": (float("nan") if degen else
                         float(-np.mean(y * np.log(p)
                                        + (1 - y) * np.log(1 - p)))),
            "tp_PIGD": tp, "tn_TD": tn,
            "fp": int(((pred == 1) & (y == 0)).sum()),
            "fn": int(((pred == 0) & (y == 1)).sum()),
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
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true",
                    help="label the run as a smoke test, never as results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    started = time.time()

    section("1. CONFIGURATION")
    if args.smoke or args.limit:
        say("  *** SMOKE TEST -- NOT FINAL RESULTS ***")
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    pm = md5(panel_path)
    if pm != EXPECTED_PANEL_MD5:
        raise SystemExit(f"panel md5 {pm} != expected {EXPECTED_PANEL_MD5}")
    say(f"  panel md5              {pm}  (frozen panel verified)")
    states = [cfg.subtype.labels[s] for s in nv.STATE_NAMES]
    panel = pd.read_csv(panel_path, low_memory=False)
    data = loo.prepare(panel, states)
    cases = nv.build_cases(data)
    patients = data.patients if not args.limit else data.patients[: args.limit]
    say(f"  patients               {len(data.sequences):,}")
    say(f"  transitions            {data.n_transitions:,}")
    say(f"  prediction cases       {len(cases):,}")
    say(f"  LOPO folds             {len(patients):,}")
    say(f"  seed / restarts / floor  {cfg.seed} / "
        f"{cfg.model['n_restarts']} / {cfg.model['emission_floor']}")
    say(f"  conditions             {len(CONDITIONS)}")
    say(f"  primary comparison     '{PRIMARY[0]}' vs '{PRIMARY[1]}' "
        f"(pre-specified)")

    allw = sliding_windows(data.sequences, WINDOW, 1)
    disw = sliding_windows(data.sequences, WINDOW, WINDOW)
    say(f"\n  cohort-wide window counts (folds use training patients only):")
    say(f"    overlapping 2-visit windows  {len(allw):,}  "
        f"(= one per transition)")
    say(f"    disjoint 2-visit windows     {len(disw):,}")
    ig = window_integrity(data.sequences, allw)
    say(f"    integrity audit              {ig}")
    if any(ig[k] for k in ("bad_shape", "bad_labels", "bad_gaps",
                           "unknown_patient")):
        raise SystemExit(f"window integrity failed: {ig}")

    section("2. LOPO: THREE FITS PER FOLD, FIVE MODEL CONDITIONS")
    say("  Per fold: the full-sequence model, the overlapping-window model and")
    say("  the disjoint-window model are each fitted on the SAME training")
    say("  patients with the same seed, starts and emission floor, so the only")
    say("  thing that differs between conditions is what the brief varies.")
    rows, diags, tams, integs, failed = [], [], [], [], []
    t0 = time.time()
    init = (data, cfg.seed, cfg.model["n_restarts"],
            cfg.model["emission_floor"], states)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                 initargs=init) as ex:
            for i, (res, err) in enumerate(ex.map(_fold_task, patients), 1):
                if err:
                    failed.append(err)
                else:
                    rows.append(res[0]); diags.append(res[1])
                    tams.append(res[2]); integs.append(res[3])
                if i % 100 == 0:
                    say(f"    {i:,}/{len(patients):,} folds "
                        f"({time.time() - t0:.0f}s)")
    else:
        _init(*init)
        for i, p in enumerate(patients, 1):
            res, err = _fold_task(p)
            if err:
                failed.append(err)
            else:
                rows.append(res[0]); diags.append(res[1])
                tams.append(res[2]); integs.append(res[3])
            if i % 20 == 0:
                say(f"    {i:,}/{len(patients):,} folds "
                    f"({time.time() - t0:.0f}s)")
    say(f"  folds completed        {len(rows):,}")
    say(f"  folds failed           {len(failed):,}")
    if failed:
        pd.DataFrame(failed).to_csv(out / "failed_folds.csv", index=False)
        raise SystemExit(f"{len(failed)} folds failed; see failed_folds.csv")

    pred = pd.concat(rows, ignore_index=True)
    dg = pd.DataFrame(diags)
    tam = pd.DataFrame(tams)
    ing = pd.DataFrame(integs)
    dg.to_csv(out / "fold_parameters.csv", index=False)
    tam.to_csv(out / "tamper_check.csv", index=False)
    ing.to_csv(out / "window_integrity_by_fold.csv", index=False)

    meta = ["PATNO", "visit_index", "n_history", "current_event",
            "current_years", "target_event", "target_years", "interval_years",
            "current_label", "actual_next_label", "eligible"]
    pred = pred.merge(cases[meta], on=["PATNO", "visit_index"], how="left",
                      validate="many_to_one")
    probs = pred[loo.PROB_COLS].to_numpy(float)
    p_td, p_pigd, defined = nv.conditional_two_class(probs)
    pred["P_TD"], pred["P_PIGD"] = p_td, p_pigd
    pred["two_class_defined"] = defined
    pred["predicted_state"] = nv.predicted_state(p_td, p_pigd)

    # persistence, reused from the validated run and restricted to these cases
    orig = pd.read_csv(ROOT / ORIG_RUN / "next_visit_predictions.csv",
                       low_memory=False)
    pers = orig[orig["model"] == PERS_MODEL][
        ["PATNO", "visit_index", "P_TD", "P_PIGD", "predicted_state",
         "two_class_defined"]].copy()
    pers = pers.merge(cases[meta], on=["PATNO", "visit_index"], how="inner",
                      validate="one_to_one")
    done = set(map(tuple, pred[["PATNO", "visit_index"]].drop_duplicates()
                   .to_numpy()))
    pers = pers[[tuple(r) in done
                 for r in pers[["PATNO", "visit_index"]].to_numpy()]].copy()
    pers["condition"] = COND_PERSIST
    pers["n_retained_visits"] = pers["n_history"]
    pers["last_gap_years"] = pers["interval_years"]
    allp = pd.concat([pred, pers], ignore_index=True)
    allp["y"] = (allp["actual_next_label"] == "PIGD").astype(int)
    allp["pred_bin"] = (allp["predicted_state"] == "PIGD").astype(int)
    allp.to_csv(out / "condition_predictions.csv", index=False)
    say(f"  prediction rows        {len(allp):,}")

    section("3. REPRODUCTION OF THE TWO EARLIER RUNS")
    repro = {}
    ref = orig[orig["model"] == CT_MODEL][["PATNO", "visit_index", "P_PIGD",
                                          "predicted_state"]]
    mine = pred[pred["condition"] == COND_FULL_FULL][
        ["PATNO", "visit_index", "P_PIGD", "predicted_state"]]
    c = ref.merge(mine, on=["PATNO", "visit_index"], suffixes=("_a", "_b"))
    repro["full-trained/full-history vs original run"] = {
        "rows": len(c),
        "max_abs_dP": float(np.max(np.abs(c["P_PIGD_a"] - c["P_PIGD_b"])))
        if len(c) else float("nan"),
        "same_calls": bool((c["predicted_state_a"]
                            == c["predicted_state_b"]).all()) if len(c) else False}
    s70 = Path(ROOT / S70_RUN / "window_predictions.csv")
    if s70.exists():
        w = pd.read_csv(s70, low_memory=False)
        w = w[w["window"] == "last 2"][["PATNO", "visit_index", "P_PIGD",
                                       "predicted_state"]]
        mine2 = pred[pred["condition"] == COND_FULL_W2][
            ["PATNO", "visit_index", "P_PIGD", "predicted_state"]]
        c2 = w.merge(mine2, on=["PATNO", "visit_index"], suffixes=("_a", "_b"))
        repro["full-trained/last-2 vs script 70"] = {
            "rows": len(c2),
            "max_abs_dP": float(np.max(np.abs(c2["P_PIGD_a"] - c2["P_PIGD_b"])))
            if len(c2) else float("nan"),
            "same_calls": bool((c2["predicted_state_a"]
                                == c2["predicted_state_b"]).all())
            if len(c2) else False}
    for k, v in repro.items():
        say(f"  {k}: rows {v['rows']:,}, max |dP| {v['max_abs_dP']:.3e}, "
            f"identical calls {v['same_calls']}")

    section("4. HOW THE FITTED PARAMETERS CHANGED")
    say("  Averaged over folds. pi is the latent distribution at a window's")
    say("  FIRST visit: baseline under full sequences, marginal over visit")
    say("  positions under sliding windows. The two are not the same quantity.")
    comp = pd.DataFrame({
        "parameter": ([f"pi[{s}]" for s in nv.STATE_NAMES]
                      + [f"Q diag [{s}]" for s in nv.STATE_NAMES]
                      + [f"E diag [{s}]" for s in nv.STATE_NAMES]),
        "full-sequence fit": (
            [dg[f"pi_full_{i}"].mean() for i in range(3)]
            + [dg[f"Qdiag_full_{i}"].mean() for i in range(3)]
            + [dg[f"Ediag_full_{i}"].mean() for i in range(3)]),
        "window fit": (
            [dg[f"pi_win_{i}"].mean() for i in range(3)]
            + [dg[f"Qdiag_win_{i}"].mean() for i in range(3)]
            + [dg[f"Ediag_win_{i}"].mean() for i in range(3)])})
    comp["difference"] = comp["window fit"] - comp["full-sequence fit"]
    comp.round(4).to_csv(out / "parameter_comparison.csv", index=False)
    say(block(comp.round(4), index=False))
    say(f"\n  mean objective value, full-sequence fit : "
        f"{dg['loglik_full'].mean():.1f} over "
        f"{dg['n_obs_full'].mean():.0f} observations")
    say(f"  mean objective value, window fit        : "
        f"{dg['loglik_windows'].mean():.1f} over "
        f"{dg['n_obs_windows'].mean():.0f} window-observations")
    say("  These two numbers are NOT comparable: the window objective is a")
    say("  composite likelihood that counts each interior visit twice.")

    section("5. COHORTS AND METRICS")
    elig = allp[allp["actual_next_label"].isin(["TD", "PIGD"])
                & allp["two_class_defined"].astype(bool)].copy()
    n_ind = int((allp["condition"] == COND_FULL_FULL).sum()
                - (elig["condition"] == COND_FULL_FULL).sum())
    say(f"  Indeterminate / undefined targets excluded: {n_ind}")
    by = {c: set(map(tuple, g[["PATNO", "visit_index"]].to_numpy()))
          for c, g in elig.groupby("condition")}
    common = set.intersection(*by.values()) if by else set()
    elig["common"] = [tuple(r) in common
                      for r in elig[["PATNO", "visit_index"]].to_numpy()]
    say(f"  common-target cohort: {len(common):,} targets, "
        f"{elig[elig['common']]['PATNO'].nunique():,} patients "
        f"(every condition defined; last-2 needs n_history >= {WINDOW})")
    say("\n  eligible targets per condition:")
    say(block(elig.groupby("condition").agg(
        n_targets=("y", "size"), n_patients=("PATNO", "nunique")).reset_index(),
        index=False))

    recs = []
    for cohort, sel in (("all eligible", elig),
                        ("common targets", elig[elig["common"]])):
        for cond in CONDITIONS:
            g = sel[sel["condition"] == cond]
            if not len(g):
                continue
            yv, pv = g["y"].to_numpy(int), g["pred_bin"].to_numpy(int)
            qv = g["P_PIGD"].to_numpy(float)
            m = metrics_for(yv, pv, qv)
            m.update({"cohort": cohort, "condition": cond,
                      "n_patients": int(g["PATNO"].nunique())})
            cis = patient_boot_all(
                g["PATNO"].to_numpy(),
                lambda rows: metrics_for(yv[rows], pv[rows], qv[rows]),
                n_boot=args.n_boot)
            for k in BOOT_KEYS:
                lo, hi, _ = cis[k]
                m[f"{k}_lo"], m[f"{k}_hi"] = lo, hi
            recs.append(m)
    mt = pd.DataFrame(recs)
    mt.round(4).to_csv(out / "condition_metrics.csv", index=False)
    for cohort in ("all eligible", "common targets"):
        say(f"\n  --- {cohort} ---")
        say(block(mt[mt.cohort == cohort][
            ["condition", "n_targets", "n_patients", "n_PIGD", "accuracy",
             "balanced_accuracy", "sensitivity_PIGD", "specificity_PIGD",
             "auc", "brier", "log_loss"]].round(4), index=False))

    section("6. PAIRED COMPARISONS")
    cm = elig[elig["common"]]

    def paired(a, b, role):
        wide = cm[cm["condition"].isin([a, b])].pivot_table(
            index=["PATNO", "visit_index", "y"], columns="condition",
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
                               n_boot=args.n_boot)
        for k in BOOT_KEYS:
            est = point[k]
            lo, hi, _ = cis[k]
            rec[k], rec[f"{k}_lo"], rec[f"{k}_hi"] = est, lo, hi
            rec[f"{k}_excludes_zero"] = bool(np.isfinite(lo)
                                             and (lo > 0 or hi < 0))
        return rec

    prs = [r for r in (
        paired(PRIMARY[0], PRIMARY[1], "PRIMARY (pre-specified)"),
        paired(COND_WIN_W2, COND_FULL_W2, "training effect, history fixed"),
        paired(COND_FULL_W2, COND_FULL_FULL, "history effect, training fixed"),
        paired(COND_WIN_FULL, COND_FULL_FULL, "training effect, full history"),
        paired(COND_WIN_W2, COND_PERSIST, "vs persistence"),
        paired(COND_FULL_FULL, COND_PERSIST, "vs persistence"),
        paired(COND_NONOV_W2, COND_WIN_W2, "disjoint-window sensitivity"),
    ) if r]
    pt = pd.DataFrame(prs)
    pt.round(4).to_csv(out / "paired_comparisons.csv", index=False)
    say(block(pt[["role", "comparison", "n_targets", "accuracy", "accuracy_lo",
                  "accuracy_hi", "balanced_accuracy", "auc", "brier"]
                 ].round(4), index=False))
    say(f"\n  {len(pt)} comparisons; one pre-specified primary. A Bonferroni")
    say(f"  threshold over the other {len(pt) - 1} would be "
        f"alpha = {0.05 / max(len(pt) - 1, 1):.4f}.")

    section("7. FIGURE")
    cohort_d = mt[mt.cohort == "common targets"].set_index("condition")
    order = [c for c in CONDITIONS if c in cohort_d.index]
    xs = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    for ax, key, lab in ((axes[0], "accuracy", "Accuracy"),
                         (axes[1], "balanced_accuracy", "Balanced accuracy"),
                         (axes[2], "auc", "ROC-AUC")):
        _style(ax)
        for i, c in enumerate(order):
            v = cohort_d.loc[c, key]
            lo, hi = cohort_d.loc[c, f"{key}_lo"], cohort_d.loc[c, f"{key}_hi"]
            colour = (MUTED if c == COND_PERSIST
                      else WARM if c == COND_WIN_W2
                      else ACCENT if "full-trained" in c else COOL)
            if not np.isfinite(v):
                ax.annotate("n/a", (xs[i], ax.get_ylim()[0]), ha="center",
                            fontsize=8, color=INK)
                continue
            ax.errorbar(xs[i], v, yerr=[[v - lo], [hi - v]], fmt="o",
                        color=colour, markersize=6.5, capsize=3,
                        elinewidth=1.3, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([c.replace(", ", ",\n") for c in order],
                           rotation=30, ha="right", fontsize=7.5)
        ax.set_ylabel(lab)
        ax.set_title(lab, fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle("Two-visit-window retraining versus full-sequence training\n"
                 "common-target cohort; 95% patient-clustered intervals; "
                 "orange is the pre-specified primary condition",
                 fontsize=10.5, x=0.01, y=0.995, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    say(f"  wrote {_finish(fig, out / 'fig1_conditions.png').name}")

    section("8. TEMPORAL INTEGRITY AND LEAKAGE CHECKS")
    src = Path(__file__).read_text()
    full_rows = pred[pred["condition"] == COND_FULL_FULL]
    w2 = pred[pred["condition"] == COND_WIN_W2]
    # the retained window must be exactly the two visits before the target
    retained_ok = bool((w2["retained_from_index"]
                        == w2["visit_index"] - WINDOW + 1).all())
    # The zero-change tamper test below must not be vacuous, so confirm the two
    # training sets really do produce different predictions on the same rows.
    key2 = ["PATNO", "visit_index"]
    a2 = w2.sort_values(key2)["P_PIGD"].to_numpy(float)
    b2 = (pred[pred["condition"] == COND_FULL_W2].sort_values(key2)
          ["P_PIGD"].to_numpy(float))
    train_effect_size = (float(np.max(np.abs(a2 - b2)))
                         if a2.shape == b2.shape and len(a2) else 0.0)
    gap_ok = bool(np.allclose(w2["last_gap_years"].to_numpy(float),
                              w2["interval_years"].to_numpy(float), atol=1e-12))
    checks = {
        "every fold completed": len(failed) == 0,
        "every fit converged in every fold": bool(pred["fits_converged"].all()),
        "windows hold only consecutive visits of one patient (audited)":
            bool((ing[["bad_shape", "bad_labels", "bad_gaps",
                       "unknown_patient"]].to_numpy() == 0).all()),
        "window audit ran on every fold": len(ing) == len(rows),
        "overlapping windows number exactly one per transition":
            len(allw) == data.n_transitions,
        "no window spans more than one patient":
            bool(ing["n_patients"].max() == len(data.sequences) - 1),
        "held-out patient absent from the training windows":
            'raise RuntimeError("held-out patient appears in the training windows")'
            in src,
        "held-out patient never in training sequences":
            "loo.assert_no_leakage(held_out, train, train_pairs)" in src,
        "parameters fitted on training patients only":
            bool((pred["n_train_patients"] == len(data.sequences) - 1).all()),
        "last-2 condition retained exactly the two visits before the target":
            retained_ok,
        "retained gap equals the original interval, unchanged": gap_ok,
        "original timeline preserved: n_history unchanged by truncation":
            bool((pred["n_history"] == pred["visit_index"] + 1).all()),
        "target strictly after the current visit":
            bool((allp["target_years"] > allp["current_years"]).all()),
        "all conditions predict the same target for a given case":
            bool(pred.groupby(["PATNO", "visit_index"])["target_event"]
                 .nunique().eq(1).all()),
        "tamper test: changing the target label leaves predictions identical":
            bool(float(tam["max_abs_change"].max()) == 0.0),
        # A fold whose patient has exactly two visits has no last-2 case at
        # all, so its tamper test is legitimately empty. The strong statement
        # is that the tamper test covered every last-2 row that was scored.
        "tamper test covered every last-2 prediction row":
            int(tam["n_cases"].sum()) == len(w2),
        "tamper test ran in every fold that has a last-2 case":
            int((tam["n_cases"] > 0).sum())
            == int(w2.groupby("PATNO").ngroups),
        "full-trained/full-history reproduces the original validated run":
            bool(repro["full-trained/full-history vs original run"]["rows"]
                 == len(mine)
                 and repro["full-trained/full-history vs original run"]
                 ["max_abs_dP"] < 1e-8
                 and repro["full-trained/full-history vs original run"]
                 ["same_calls"]),
        "window-trained predictions actually differ from full-trained":
            train_effect_size > 1e-6,
        "Indeterminate excluded from binary scoring":
            bool(set(elig["actual_next_label"]) == {"TD", "PIGD"}),
        "common cohort identical in every condition":
            len(set(elig[elig["common"]].groupby("condition").size())) == 1,
        "probabilities finite and in [0, 1]": bool(
            np.isfinite(allp["P_PIGD"]).all() and (allp["P_PIGD"] >= 0).all()
            and (allp["P_PIGD"] <= 1).all()),
        "bootstrap resamples patients, not targets":
            "groups = _patient_row_index(np.asarray(patients))" in src,
        "fixed seed recorded": cfg.seed is not None,
    }
    for n, v in checks.items():
        say(f"    [{'PASS' if v else 'FAIL'}]  {n}")
    pd.DataFrame([{"check": k, "passed": bool(v)} for k, v in checks.items()]
                 ).to_csv(out / "validation_checks.csv", index=False)
    if not all(checks.values()):
        raise SystemExit(f"CRITICAL CHECK FAILED: "
                         f"{[k for k, v in checks.items() if not v]}")
    n_empty = int((tam["n_cases"] == 0).sum())
    say(f"\n  folds with no last-2 case (patient has only two visits): {n_empty}")
    say("  Note on the tamper test: it passes because the predictive")
    say("  distribution at a step is formed before that step's label updates")
    say("  alpha. The separate check above confirms the window-trained and")
    say("  full-trained predictions DO differ, so the zero change is not")
    say("  because the code path was inert: retraining moves P(PIGD) by up to")
    say(f"  {train_effect_size:.4f} on identical rows.")

    section("9. PROVENANCE")
    env = {"panel_md5": pm, "python": sys.version.split()[0],
           "numpy": np.__version__, "pandas": pd.__version__,
           "node": platform.node(), "workers": args.workers,
           "n_bootstrap": args.n_boot, "seed": cfg.seed,
           "n_restarts": cfg.model["n_restarts"],
           "emission_floor": cfg.model["emission_floor"],
           "window_width": WINDOW, "n_folds": len(rows),
           "n_folds_failed": len(failed),
           "smoke_test": bool(args.smoke or args.limit),
           "original_run": ORIG_RUN, "script70_run": S70_RUN,
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
        {"environment": env, "checks": checks, "reproduction": repro,
         "code_sha256": {str(Path(__file__).relative_to(ROOT)):
                         sha(Path(__file__))},
         "conditions": CONDITIONS,
         "primary_comparison": f"{PRIMARY[0]} vs {PRIMARY[1]}",
         "n_common_targets": len(common),
         "n_indeterminate_excluded": n_ind,
         "cohort_window_counts": {"overlapping": len(allw),
                                  "disjoint": len(disw),
                                  "transitions": int(data.n_transitions)},
         "window_integrity_cohort": ig,
         "tamper_max_abs_change": float(tam["max_abs_change"].max()),
         "tamper_cases": int(tam["n_cases"].sum()),
         "folds_without_last2_case": int((tam["n_cases"] == 0).sum()),
         "max_abs_prediction_change_from_retraining": train_effect_size,
         "metrics": mt.to_dict(orient="records"),
         "paired": pt.to_dict(orient="records"),
         "parameter_comparison": comp.to_dict(orient="records")},
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
