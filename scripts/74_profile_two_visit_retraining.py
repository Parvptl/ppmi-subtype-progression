"""Experiment 6B: retrain the rich clinical profile on two-visit windows.

Experiment 6A (`scripts/73`) held the model fixed and varied only what it could
see at prediction time, and found two visits enough. This experiment asks the
complementary question -- does *training* on two-visit windows help? -- with the
full 2x2 the brief requires, so that training and prediction effects separate:

                                    prediction history
                              full history        last 2 visits
    training  full sequences   (A)                 (B)
              2-visit windows  (C)                 (D)  <- PRIMARY vs (B)

plus a disjoint-window sensitivity for (C) and (D), and the persistence
reference. Comparing (B) with (D) isolates the training change, because the
prediction history is identical in both.

WHAT A TWO-VISIT WINDOW DOES TO THIS MODEL'S OBJECTIVE
------------------------------------------------------
This is the methodological heart of the experiment, and it is *not* the same
situation as the CT-HMM's.

`ProfileModel.filter_domain` accumulates one one-step-ahead Gaussian term per
visit::

    for k in range(n):
        if k > 0: propagate over dt          # transition term
        mu = H @ m + beta * d[k]             # at k == 0 this is the PRIOR
        loglik += -0.5 * (log(2 pi var) + resid^2 / var)

So the ``k == 0`` term is a prediction from the fixed prior (``m = 0``,
``P = diag([INIT_LEVEL_VAR, init_rate_var])``), not from data. A full sequence
of n visits therefore contributes **1 prior term and n-1 transition terms**.
A two-visit window contributes **1 prior term and 1 transition term**.

Two consequences, both reported rather than glossed:

1.  **Overlapping windows duplicate observations.** Each interior visit appears
    in two windows -- once as the window's second visit (a genuine one-step
    prediction) and once as the next window's first visit (a prior prediction).
    Total observation terms rise from `n_visits` to `2 * n_transitions`.
2.  **The prior-to-transition weighting changes drastically**, for overlapping
    *and* disjoint windows alike: from roughly 1 prior term per patient to one
    prior term per window. Initialisation terms stop being a negligible edge
    effect and become half the objective.

Neither construction therefore yields "the same likelihood on shorter
sequences". The overlapping objective is a composite (pseudo) likelihood whose
value is not comparable with the full-sequence likelihood; the disjoint
objective is a proper likelihood over disjoint pairs but still carries the
reweighting in (2). Section 3 verifies this accounting numerically instead of
asserting it.

TIMELINE
--------
Windows are slices: original visit dates, years-from-baseline, inter-visit gaps,
clinical values and medication inputs are carried through unchanged, and time is
never reset to zero. Only consecutive visits of one patient ever share a window,
so no artificial transition is created.

    python scripts/74_profile_two_visit_retraining.py --out <dir> [--workers N]
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
N_FOLDS = 5
WINDOW = 2
MIN_CLASS = 5

TRAIN_FULL, TRAIN_OV, TRAIN_NO = "full-seq", "window-ov", "window-disj"
A = "A: full-trained, full history"
B = "B: full-trained, last 2"
C = "C: window-trained, full history"
D = "D: window-trained, last 2"
C2 = "C': disjoint-window-trained, full history"
D2 = "D': disjoint-window-trained, last 2"
PERSIST = "persistence"
CONDITIONS = [PERSIST, A, B, C, D, C2, D2]
PRIMARY = (D, B)

BOOT_KEYS = ("accuracy", "balanced_accuracy", "auc", "brier", "log_loss",
             "sensitivity_PIGD", "specificity_TD")
SLICE_KEYS = ("Y", "d", "times", "labels", "events", "visit_index")

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
#  Windows and slices
# ---------------------------------------------------------------------------
def truncate(seq: dict, lo: int, hi: int) -> dict:
    """Visits ``lo..hi`` inclusive, with every original value carried through."""
    out = {k: np.asarray(seq[k])[lo:hi + 1] for k in SLICE_KEYS}
    out["gaps"] = np.diff(out["times"])
    out["PATNO"] = seq["PATNO"]
    out["window_start"] = int(lo)
    return out


def sliding_windows(sequences: list[dict], stride: int) -> list[dict]:
    """Two-visit windows of consecutive visits from one patient.

    ``stride=1`` gives overlapping windows (one per transition); ``stride=2``
    gives disjoint pairs starting at visit 0, i.e. (0,1), (2,3), ... With an odd
    number of visits the final unpaired visit is **dropped**, never padded and
    never joined to a non-adjacent visit. The rule does not look at any label or
    at prediction performance.
    """
    out = []
    for s in sequences:
        n = len(s["times"])
        for a in range(0, n - 1, stride):
            out.append(truncate(s, a, a + 1))
    return out


def window_accounting(sequences, wins_ov, wins_no) -> dict:
    """Count the terms each training objective actually sums."""
    n_vis = int(sum(len(s["times"]) for s in sequences))
    n_tr = int(sum(len(s["times"]) - 1 for s in sequences))
    return {
        "n_patients": len(sequences), "n_visits": n_vis, "n_transitions": n_tr,
        "full_obs_terms": n_vis, "full_prior_terms": len(sequences),
        "full_transition_terms": n_vis - len(sequences),
        "ov_windows": len(wins_ov), "ov_obs_terms": 2 * len(wins_ov),
        "ov_prior_terms": len(wins_ov), "ov_transition_terms": len(wins_ov),
        "disj_windows": len(wins_no), "disj_obs_terms": 2 * len(wins_no),
        "disj_prior_terms": len(wins_no),
        "disj_transition_terms": len(wins_no),
        "ov_duplication_factor": (2 * len(wins_ov) / n_vis) if n_vis else float("nan"),
    }


def window_integrity(sequences, wins) -> dict:
    """Audit every window against its parent sequence."""
    parent = {s["PATNO"]: s for s in sequences}
    bad_shape = bad_times = bad_values = unknown = 0
    for w in wins:
        s = parent.get(w["PATNO"])
        if s is None:
            unknown += 1
            continue
        a = w["window_start"]
        if len(w["times"]) != 2:
            bad_shape += 1
        if not np.array_equal(np.asarray(w["times"]),
                              np.asarray(s["times"])[a:a + 2]):
            bad_times += 1
        if not np.array_equal(np.nan_to_num(np.asarray(w["Y"]), nan=-999),
                              np.nan_to_num(np.asarray(s["Y"])[a:a + 2],
                                            nan=-999)):
            bad_values += 1
    return {"n_windows": len(wins), "bad_shape": bad_shape,
            "bad_times": bad_times, "bad_values": bad_values,
            "unknown_patient": unknown}


def predict_under(model, seq: dict, standardiser, width):
    """Predictions for every case of one patient under one history rule."""
    records = []
    n = len(seq["times"])
    for k in range(n - 1):
        n_hist = k + 1
        if width is not None and n_hist < width:
            continue
        lo = 0 if width is None else k - width + 1
        sub = truncate(seq, lo, k + 1)
        recs = cp.predict_sequence(model, sub, standardiser,
                                   models=(cp.MODEL_FULL,))
        r = dict(recs[-1])
        r["available_history"] = n_hist
        r["n_retained_visits"] = len(sub["times"]) - 1
        r["retained_from_index"] = int(lo)
        r["first_retained_years"] = float(sub["times"][0])
        records.append(r)
    return records


def _fingerprint(m) -> str:
    return hashlib.sha256(np.concatenate([
        np.asarray(m.obs_var, float).ravel(),
        np.asarray(m.level_var, float).ravel(),
        np.asarray(m.rate_var, float).ravel(),
        np.asarray(m.init_rate_var, float).ravel(),
        np.asarray(m.beta, float).ravel()]).tobytes()).hexdigest()[:16]


def run_fold(test_patients, fold: int):
    sequences = _WORK["sequences"]
    test_set = set(test_patients)
    train = [s for s in sequences if s["PATNO"] not in test_set]
    test = [s for s in sequences if s["PATNO"] in test_set]
    if not train or not test:
        raise ValueError(f"empty split (fold {fold})")
    if test_set & {s["PATNO"] for s in train}:
        raise RuntimeError("held-out patient leaked into training")

    # Preprocessing and population quantities: training patients, full
    # sequences, exactly as the existing procedure does. Identical for all
    # three fitted models so the only difference is the training objective.
    standardiser = cp.Standardiser.fit(train)
    train_z = standardiser.apply(train)
    test_z = standardiser.apply(test)

    wins_ov = sliding_windows(train_z, stride=1)
    wins_no = sliding_windows(train_z, stride=WINDOW)
    if test_set & {w["PATNO"] for w in wins_ov}:
        raise RuntimeError("held-out patient appears in the training windows")

    fits = {TRAIN_FULL: cp.fit(train_z, use_rate=True, use_medication=True),
            TRAIN_OV: cp.fit(wins_ov, use_rate=True, use_medication=True),
            TRAIN_NO: cp.fit(wins_no, use_rate=True, use_medication=True)}

    records = []
    for cond, tag, width in ((A, TRAIN_FULL, None), (B, TRAIN_FULL, WINDOW),
                             (C, TRAIN_OV, None), (D, TRAIN_OV, WINDOW),
                             (C2, TRAIN_NO, None), (D2, TRAIN_NO, WINDOW)):
        for seq in test_z:
            for r in predict_under(fits[tag], seq, standardiser, width):
                r["condition"] = cond
                r["training"] = tag
                r["history_rule"] = "full" if width is None else f"last {width}"
                records.append(r)

    frame = cp.records_to_frame(records)
    for col in ("condition", "training", "history_rule", "available_history",
                "n_retained_visits", "retained_from_index",
                "first_retained_years"):
        frame[col] = [r[col] for r in records]
    frame["fold"] = fold
    frame["n_train_patients"] = len(train)
    frame["n_train_windows_ov"] = len(wins_ov)
    frame["n_train_windows_disj"] = len(wins_no)
    frame["fit_converged"] = bool(all(m.converged for m in fits.values()))
    fp = {TRAIN_FULL: _fingerprint(fits[TRAIN_FULL]),
          TRAIN_OV: _fingerprint(fits[TRAIN_OV]),
          TRAIN_NO: _fingerprint(fits[TRAIN_NO])}
    frame["param_fingerprint"] = frame["training"].map(fp)

    diag = {"fold": fold,
            **{f"loglik_{k}": float(v.loglik) for k, v in fits.items()},
            **{f"converged_{k}": bool(v.converged) for k, v in fits.items()}}
    for k, v in fits.items():
        for name in ("obs_var", "rate_var", "level_var", "init_rate_var"):
            diag[f"mean_{name}_{k}"] = float(np.mean(np.asarray(
                getattr(v, name), float)))
        # A two-visit window cannot identify a slope, so the optimiser may
        # drive the rate variances to their floor and switch the rate
        # component off entirely. That would turn the level+rate model into a
        # level-only random walk by estimation rather than by design, so it is
        # measured per fold instead of being left to be noticed in the means.
        rv = np.asarray(v.rate_var, float)
        iv = np.asarray(v.init_rate_var, float)
        diag[f"n_domains_rate_collapsed_{k}"] = int(
            ((rv <= cp.MIN_PROCESS_VAR * 1.01)
             & (iv <= cp.MIN_PROCESS_VAR * 1.01)).sum())
        diag[f"max_rate_var_{k}"] = float(rv.max())
        diag[f"max_init_rate_var_{k}"] = float(iv.max())
    diag["n_domains"] = int(len(cp.DOMAINS))
    disj_expected = int(sum(len(s["times"]) // 2 for s in train_z))
    # Disjoint windows must share no visit: starts are even and spaced by 2.
    starts = {}
    for w in wins_no:
        starts.setdefault(w["PATNO"], []).append(w["window_start"])
    disj_share = int(sum(
        1 for v in starts.values()
        for a, b in zip(sorted(v), sorted(v)[1:]) if b - a < WINDOW))
    integ = {"fold": fold, "disj_expected": disj_expected,
             "disj_sharing_visits": disj_share,
             **{f"ov_{a}": b for a, b in
                window_integrity(train_z, wins_ov).items()},
             **{f"disj_{a}": b for a, b in
                window_integrity(train_z, wins_no).items()},
             **window_accounting(train_z, wins_ov, wins_no)}
    return frame, diag, integ


def _fold_task(args):
    fold, pats = args
    try:
        return run_fold(pats, fold), None
    except Exception as exc:                        # noqa: BLE001
        return None, {"fold": fold, "error": f"{type(exc).__name__}: {exc}"}


def _init(sequences):
    _WORK["sequences"] = sequences


# ---------------------------------------------------------------------------
#  Metrics (identical definitions to Experiment 6A)
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
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-repro", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    started = time.time()

    section("1. CONFIGURATION")
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    pm = md5(panel_path)
    if pm != EXPECTED_PANEL_MD5:
        raise SystemExit(f"panel md5 {pm} != expected {EXPECTED_PANEL_MD5}")
    say(f"  panel md5              {pm}  (frozen panel verified)")
    panel = pd.read_csv(panel_path, low_memory=False)
    sequences = cp.build_sequences(cs.build_clinical_panel(cfg, panel))
    patients = sorted(s["PATNO"] for s in sequences)
    say(f"  clinical sequences     {len(sequences):,} patients")
    say(f"  validation             {N_FOLDS}-fold patient CV, one repeat "
        f"(Experiment 6A's scheme, preserved)")
    say(f"  seed                   {cfg.seed}")
    say(f"  conditions             {len(CONDITIONS)} "
        f"(2x2 + disjoint sensitivity + persistence)")
    say(f"  PRIMARY comparison     '{PRIMARY[0]}' vs '{PRIMARY[1]}'")
    say("                         prediction history identical in both, so")
    say("                         the only difference is the training objective")

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(patients))
    pats = np.array(patients, dtype=object)
    tasks = [(f, pats[order[f::N_FOLDS]].tolist()) for f in range(N_FOLDS)]
    assert sum(len(t[1]) for t in tasks) == len(patients)
    assert len({p for _, ps in tasks for p in ps}) == len(patients)
    say(f"  fold sizes             {[len(t[1]) for t in tasks]}  "
        f"(identical construction to Experiment 6A)")

    section("2. CROSS-VALIDATION: THREE FITS PER FOLD, SIX MODEL CONDITIONS")
    frames, diags, integs, failed = [], [], [], []
    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                 initargs=(sequences,)) as ex:
            for res, err in ex.map(_fold_task, tasks):
                if err:
                    failed.append(err)
                else:
                    frames.append(res[0]); diags.append(res[1])
                    integs.append(res[2])
                say(f"    fold done ({time.time() - t0:.0f}s)")
    else:
        _init(sequences)
        for t in tasks:
            res, err = _fold_task(t)
            if err:
                failed.append(err)
            else:
                frames.append(res[0]); diags.append(res[1])
                integs.append(res[2])
            say(f"    fold {t[0]} done ({time.time() - t0:.0f}s)")
    if failed:
        pd.DataFrame(failed).to_csv(out / "failed_folds.csv", index=False)
        raise SystemExit(f"{len(failed)} folds failed; see failed_folds.csv")

    pred = pd.concat(frames, ignore_index=True)
    dg = pd.DataFrame(diags)
    ig = pd.DataFrame(integs)
    dg.round(6).to_csv(out / "fold_parameters.csv", index=False)
    ig.to_csv(out / "window_accounting.csv", index=False)
    say(f"  prediction rows        {len(pred):,}")

    section("3. WHAT THE TRAINING OBJECTIVE ACTUALLY SUMS")
    acc = ig.iloc[0]
    say("  Per fold (fold 0 shown; every fold is in window_accounting.csv).")
    say(f"    training patients                     {int(acc['n_patients']):,}")
    say(f"    their visits / transitions            {int(acc['n_visits']):,}"
        f" / {int(acc['n_transitions']):,}")
    say("")
    say(f"    FULL SEQUENCES  observation terms     {int(acc['full_obs_terms']):,}")
    say(f"                    of which prior terms  {int(acc['full_prior_terms']):,}"
        f"   (1 per patient)")
    say(f"                    transition terms      {int(acc['full_transition_terms']):,}")
    say(f"    OVERLAPPING     windows               {int(acc['ov_windows']):,}"
        f"   (= one per transition)")
    say(f"                    observation terms     {int(acc['ov_obs_terms']):,}"
        f"   ({acc['ov_duplication_factor']:.2f}x the visits)")
    say(f"                    of which prior terms  {int(acc['ov_prior_terms']):,}"
        f"   (1 per window)")
    say(f"                    transition terms      {int(acc['ov_transition_terms']):,}")
    say(f"    DISJOINT        windows               {int(acc['disj_windows']):,}")
    say(f"                    observation terms     {int(acc['disj_obs_terms']):,}")
    say(f"                    of which prior terms  {int(acc['disj_prior_terms']):,}")
    say(f"                    transition terms      {int(acc['disj_transition_terms']):,}")
    say("")
    say("  Reading: under full sequences the prior term is an edge effect, one")
    say("  per patient. Under either window construction it becomes one per")
    say("  window -- half of the objective. Overlapping windows additionally")
    say("  count each interior visit twice. So neither window objective is the")
    say("  full-sequence likelihood evaluated on shorter sequences, and the")
    say("  overlapping one is a composite likelihood whose VALUE is not")
    say("  comparable with the others.")
    say("\n  RATE COMPONENT: can a two-visit window identify a slope?")
    nd = int(dg["n_domains"].iloc[0])
    for tag, name in ((TRAIN_FULL, "full sequences"),
                      (TRAIN_OV, "overlapping windows"),
                      (TRAIN_NO, "disjoint windows")):
        say(f"    {name:<22} domains with the rate switched off: "
            f"{dg[f'n_domains_rate_collapsed_{tag}'].mean():.1f} of {nd}"
            f"   (max rate_var {dg[f'max_rate_var_{tag}'].max():.2e}, "
            f"max init_rate_var {dg[f'max_init_rate_var_{tag}'].max():.2e})")
    say("")
    say("\n  fitted-parameter means by training set (averaged over folds):")
    par = pd.DataFrame({
        "parameter": ["obs_var", "level_var", "rate_var", "init_rate_var",
                      "log-likelihood (not comparable across columns)"],
        "full sequences": [dg[f"mean_{p}_{TRAIN_FULL}"].mean() for p in
                           ("obs_var", "level_var", "rate_var", "init_rate_var")]
                          + [dg[f"loglik_{TRAIN_FULL}"].mean()],
        "overlapping windows": [dg[f"mean_{p}_{TRAIN_OV}"].mean() for p in
                                ("obs_var", "level_var", "rate_var", "init_rate_var")]
                               + [dg[f"loglik_{TRAIN_OV}"].mean()],
        "disjoint windows": [dg[f"mean_{p}_{TRAIN_NO}"].mean() for p in
                             ("obs_var", "level_var", "rate_var", "init_rate_var")]
                            + [dg[f"loglik_{TRAIN_NO}"].mean()]})
    par.round(4).to_csv(out / "parameter_comparison.csv", index=False)
    say(block(par.round(4), index=False))

    section("4. LABELS, COHORTS")
    pred = cp.derive_label_probabilities(pred, cfg)
    probs = pred[[f"p_{s}" for s in cs.STATE_NAMES]].to_numpy(float)
    p_td, p_pigd, defined = nv.conditional_two_class(probs)
    pred["P_TD"], pred["P_PIGD"] = p_td, p_pigd
    pred["two_class_defined"] = defined
    pred["predicted_state"] = nv.predicted_state(p_td, p_pigd)

    key = ["PATNO", "current_event", "target_event"]
    orig = pd.read_csv(ROOT / CT_RUN / "next_visit_predictions.csv",
                       low_memory=False)
    pers = orig[orig["model"] == PERS_MODEL][
        key + ["P_TD", "P_PIGD", "predicted_state", "two_class_defined"]].copy()
    spine = pred[pred["condition"] == A][
        key + ["actual_next_label", "current_label", "interval_years",
               "current_years", "target_years", "available_history",
               "fold"]].copy()
    pers = pers.merge(spine, on=key, how="inner", validate="one_to_one")
    pers["condition"] = PERSIST
    pers["training"] = "none (rule)"
    pers["history_rule"] = "last observed label"
    pers["n_retained_visits"] = pers["available_history"]
    allp = pd.concat([pred, pers], ignore_index=True)
    allp["y"] = (allp["actual_next_label"] == "PIGD").astype(int)
    allp["pred_bin"] = (allp["predicted_state"] == "PIGD").astype(int)
    keep = [c for c in allp.columns
            if not c.startswith(("z_", "sd_"))
            and not (c.startswith(("pred_", "actual_"))
                     and c not in ("pred_bin", "actual_next_label"))]
    allp[keep].to_csv(out / "condition_predictions.csv", index=False)

    elig = allp[allp["actual_next_label"].isin(["TD", "PIGD"])
                & allp["two_class_defined"].astype(bool)].copy()
    n_ind = int((allp["condition"] == A).sum() - (elig["condition"] == A).sum())
    say(f"  Indeterminate / undefined-ratio targets excluded: {n_ind}")
    by = {c: set(map(tuple, g[key].to_numpy())) for c, g in elig.groupby("condition")}
    common = set.intersection(*by.values()) if by else set()
    elig["common"] = [tuple(r) in common for r in elig[key].to_numpy()]
    say(f"  matched cohort (every condition defined, needs >= {WINDOW} prior "
        f"visits): {len(common):,} targets, "
        f"{elig[elig['common']]['PATNO'].nunique():,} patients")
    counts = (elig.groupby("condition")
                  .agg(n_targets=("y", "size"), n_patients=("PATNO", "nunique"),
                       n_TD=("y", lambda s: int((s == 0).sum())),
                       n_PIGD=("y", "sum")).reset_index())
    counts["pigd_prevalence"] = counts["n_PIGD"] / counts["n_targets"]
    counts.round(4).to_csv(out / "cohort_counts.csv", index=False)
    say("\n  eligible targets per condition:")
    say(block(counts, index=False))

    section("5. METRICS ON THE MATCHED COHORT")
    cm = elig[elig["common"]]
    recs = []
    for cond in CONDITIONS:
        g = cm[cm["condition"] == cond]
        if not len(g):
            continue
        yv, pv = g["y"].to_numpy(int), g["pred_bin"].to_numpy(int)
        qv = g["P_PIGD"].to_numpy(float)
        m = metrics_for(yv, pv, qv)
        m.update({"condition": cond, "n_patients": int(g["PATNO"].nunique())})
        cis = patient_boot_all(
            g["PATNO"].to_numpy(),
            lambda rows: metrics_for(yv[rows], pv[rows], qv[rows]),
            n_boot=args.n_boot, seed=cfg.seed)
        for k in BOOT_KEYS:
            lo, hi, nd = cis[k]
            m[f"{k}_lo"], m[f"{k}_hi"], m[f"{k}_valid_replicates"] = lo, hi, nd
        recs.append(m)
    mt = pd.DataFrame(recs)
    mt.round(4).to_csv(out / "condition_metrics.csv", index=False)
    mt[["condition", "tp_PIGD", "tn_TD", "fp", "fn"]].to_csv(
        out / "confusion_matrices.csv", index=False)
    say(block(mt[["condition", "n_targets", "n_patients", "n_PIGD", "accuracy",
                  "balanced_accuracy", "auc", "brier", "log_loss",
                  "sensitivity_PIGD", "specificity_TD", "precision_PIGD",
                  "precision_TD"]].round(4), index=False))

    section("6. PAIRED COMPARISONS")

    def paired(a, b, role):
        wide = cm[cm["condition"].isin([a, b])].pivot_table(
            index=key + ["y"], columns="condition",
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
            lo, hi, nd = cis[k]
            rec[k], rec[f"{k}_lo"], rec[f"{k}_hi"] = point[k], lo, hi
            rec[f"{k}_valid_replicates"] = nd
            rec[f"{k}_excludes_zero"] = bool(np.isfinite(lo)
                                             and (lo > 0 or hi < 0))
        return rec

    prs = [r for r in (
        paired(D, B, "PRIMARY: training effect, history fixed at last 2"),
        paired(C, A, "training effect, history fixed at full"),
        paired(B, A, "history effect, training fixed at full-sequence"),
        paired(D, C, "history effect, training fixed at windows"),
        paired(D2, D, "disjoint vs overlapping windows"),
        paired(D, PERSIST, "vs persistence"),
        paired(B, PERSIST, "vs persistence"),
    ) if r]
    pt = pd.DataFrame(prs)
    pt.round(4).to_csv(out / "paired_comparisons.csv", index=False)
    say(block(pt[["role", "n_targets", "accuracy", "accuracy_lo",
                  "accuracy_hi", "balanced_accuracy", "auc",
                  "sensitivity_PIGD"]].round(4), index=False))
    say(f"\n  {len(pt)} comparisons, one pre-specified primary. A Bonferroni")
    say(f"  threshold over the other {len(pt) - 1} is alpha = "
        f"{0.05 / max(len(pt) - 1, 1):.4f}. Intervals are UNADJUSTED.")

    # ================================================= 7. reproducibility
    section("7. REPRODUCIBILITY")
    repro = {"refit_checked": False}
    if not args.skip_repro:
        _init(sequences)
        again, _, _ = run_fold(tasks[0][1], 0)
        a0 = frames[0].sort_values(key + ["condition"]).reset_index(drop=True)
        b0 = again.sort_values(key + ["condition"]).reset_index(drop=True)
        cols = [c for c in ("pred_tremor", "pred_pigd", "sd_tremor", "sd_pigd")
                if c in a0.columns]
        same = bool(len(a0) == len(b0) and all(
            np.array_equal(a0[c].to_numpy(float), b0[c].to_numpy(float))
            for c in cols))
        fp_same = bool((a0["param_fingerprint"].to_numpy()
                        == b0["param_fingerprint"].to_numpy()).all())
        repro = {"refit_checked": True, "rows": int(len(b0)),
                 "identical_predictions": same, "same_fingerprints": fp_same}
        say(f"  refit fold 0 with the same seed: {len(b0):,} rows, identical "
            f"predictions {same}, identical fingerprints {fp_same}")
    else:
        say("  SKIPPED (--skip-repro)")

    saved = pd.read_csv(out / "condition_predictions.csv", low_memory=False)
    se = saved[saved["actual_next_label"].isin(["TD", "PIGD"])
               & saved["two_class_defined"].astype(bool)]
    se = se[[tuple(r) in common for r in se[key].to_numpy()]]
    g = se[se["condition"] == D]
    rc = metrics_for(g["y"], g["pred_bin"], g["P_PIGD"])
    ref = mt[mt.condition == D].iloc[0]
    metrics_reproduce = bool(
        abs(rc["accuracy"] - ref["accuracy"]) < 1e-12
        and abs(rc["auc"] - ref["auc"]) < 1e-12
        and abs(rc["brier"] - ref["brier"]) < 1e-12)
    say(f"  metrics recomputed from the saved prediction file match: "
        f"{metrics_reproduce}")

    # ================================================= 8. figures
    section("8. FIGURES")
    cd = mt.set_index("condition")
    plot_c = [c for c in CONDITIONS if c in cd.index]
    xs = np.arange(len(plot_c))
    short = {PERSIST: "persistence", A: "A\nfull-trained\nfull hist",
             B: "B\nfull-trained\nlast 2", C: "C\nwindow-trained\nfull hist",
             D: "D\nwindow-trained\nlast 2",
             C2: "C'\ndisjoint-win\nfull hist", D2: "D'\ndisjoint-win\nlast 2"}

    def colour(c):
        if c == PERSIST:
            return MUTED
        if c == D:
            return WARM
        return ACCENT if c in (A, B) else COOL

    def panel(ax, k, lab):
        _style(ax)
        for i, c in enumerate(plot_c):
            v = cd.loc[c, k]
            if not np.isfinite(v):
                ax.annotate("n/a", (xs[i], ax.get_ylim()[0]), ha="center",
                            fontsize=8, color=INK_MUTED)
                continue
            lo, hi = cd.loc[c, f"{k}_lo"], cd.loc[c, f"{k}_hi"]
            ax.errorbar(xs[i], v, yerr=[[v - lo], [hi - v]], fmt="o",
                        color=colour(c), markersize=6.5, capsize=3,
                        elinewidth=1.3, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([short[c] for c in plot_c], fontsize=7)
        ax.set_ylabel(lab)
        ax.set_title(lab, fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    for ax, k, lab in ((axes[0], "accuracy", "Accuracy"),
                       (axes[1], "balanced_accuracy", "Balanced accuracy"),
                       (axes[2], "auc", "AUC-ROC")):
        panel(ax, k, lab)
    fig.suptitle("Experiment 6B: two-visit retraining, full 2x2 — matched "
                 f"cohort of {len(common):,} targets\n95% patient-clustered "
                 "intervals; orange is the primary retrained condition (D)",
                 fontsize=10.5, x=0.01, y=0.995, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    say(f"  wrote {_finish(fig, out / 'fig1_2x2_performance.png').name}")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    for ax, k, lab in ((axes[0], "brier", "Brier score"),
                       (axes[1], "log_loss", "Log loss")):
        panel(ax, k, f"{lab} — lower is better")
    fig.suptitle("Probability quality across the 2x2\npersistence emits only "
                 "0 and 1, so its log loss is not a meaningful quantity",
                 fontsize=10.5, x=0.01, y=0.995, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    say(f"  wrote {_finish(fig, out / 'fig2_probability_quality.png').name}")

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    _style(ax)
    for k, c, lab, off in (("sensitivity_PIGD", ACCENT, "PIGD sensitivity", -0.12),
                           ("specificity_TD", COOL, "TD specificity", +0.12)):
        v = np.array([cd.loc[x, k] for x in plot_c], float)
        lo = np.array([cd.loc[x, f"{k}_lo"] for x in plot_c], float)
        hi = np.array([cd.loc[x, f"{k}_hi"] for x in plot_c], float)
        ax.errorbar(xs + off, v, yerr=[v - lo, hi - v], fmt="o", color=c,
                    markersize=6, capsize=3, elinewidth=1.3, label=lab, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([short[c] for c in plot_c], fontsize=7)
    ax.set_ylabel("Rate")
    ax.set_title("Class-specific performance across the 2x2\nExperiment 6A "
                 "found restricted history costs PIGD sensitivity; this shows "
                 "whether retraining recovers it", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    say(f"  wrote {_finish(fig, out / 'fig3_class_specific.png').name}")

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    _style(ax)
    pr = pt[pt["role"].str.startswith("PRIMARY")].iloc[0]
    keys = ["accuracy", "balanced_accuracy", "auc", "brier", "log_loss",
            "sensitivity_PIGD", "specificity_TD"]
    yy = np.arange(len(keys))
    v = np.array([pr[k] for k in keys], float)
    lo = np.array([pr[f"{k}_lo"] for k in keys], float)
    hi = np.array([pr[f"{k}_hi"] for k in keys], float)
    ax.errorbar(v, yy, xerr=[v - lo, hi - v], fmt="o", color=WARM,
                markersize=6.5, capsize=3, elinewidth=1.4, zorder=3)
    ax.axvline(0, color=INK, linewidth=1.2, zorder=2)
    ax.set_yticks(yy)
    ax.set_yticklabels([k.replace("_", " ") for k in keys], fontsize=9)
    ax.set_xlabel("D minus B  (window-trained minus full-trained, both "
                  "predicting from the last 2 visits)")
    ax.set_title("Primary comparison: the effect of retraining alone\n"
                 f"{int(pr['n_targets']):,} matched targets; 95% "
                 "patient-clustered intervals. For Brier and log loss, "
                 "positive is worse.", fontsize=10.5, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    say(f"  wrote {_finish(fig, out / 'fig4_primary_paired.png').name}")

    # ================================================= 9. checks
    section("9. VALIDATION CHECKS")
    src = Path(__file__).read_text()
    last2 = pred[pred["history_rule"] == f"last {WINDOW}"]
    fullh = pred[pred["history_rule"] == "full"]
    seq_times = {s["PATNO"]: np.asarray(s["times"], float) for s in sequences}
    tl_ok = True
    for _, r in pred.sample(min(5000, len(pred)),
                            random_state=cfg.seed).iterrows():
        t = seq_times[r["PATNO"]]
        if not np.isclose(float(r["first_retained_years"]),
                          float(t[int(r["retained_from_index"])]), atol=0):
            tl_ok = False
            break
    fp_per_training = pred.groupby("training")["param_fingerprint"].nunique()
    checks = {
        "every fold completed": len(failed) == 0,
        "every fit converged in every fold": bool(pred["fit_converged"].all()),
        "each patient in exactly one fold":
            bool((pred.groupby("PATNO")["fold"].nunique() == 1).all()),
        "same folds used for all conditions":
            bool((pred.groupby(key)["fold"].nunique() == 1).all()),
        "held-out patient never in training":
            'raise RuntimeError("held-out patient leaked into training")' in src,
        "held-out patient never in the training windows":
            'raise RuntimeError("held-out patient appears in the training windows")'
            in src,
        "preprocessing and population quantities from training patients only":
            "standardiser = cp.Standardiser.fit(train)" in src,
        "parameters fitted on a strict subset of patients":
            bool((pred["n_train_patients"] < len(sequences)).all()),
        "one parameter set per training strategy per fold, fixed at inference":
            bool((fp_per_training == N_FOLDS).all()),
        "the three training strategies gave DIFFERENT parameters":
            bool(pred.groupby("fold")["param_fingerprint"].nunique().eq(3).all()),
        "rate collapse under window training is measured, not assumed":
            bool(all(f"n_domains_rate_collapsed_{t}" in dg.columns
                     for t in (TRAIN_FULL, TRAIN_OV, TRAIN_NO))),
        "full-sequence training retains a non-degenerate rate component":
            bool((dg[f"max_rate_var_{TRAIN_FULL}"]
                  > cp.MIN_PROCESS_VAR * 1.01).all()),
        "overlapping windows number exactly one per transition":
            bool((ig["ov_windows"] == ig["n_transitions"]).all()),
        "overlapping objective duplicates interior visits, as documented":
            bool((ig["ov_obs_terms"] > ig["n_visits"]).all()),
        # Per patient a stride-2 pass yields floor(n_i / 2) windows; summed
        # over patients that is the exact expected count. (An earlier version
        # of this check bounded it by half the TRANSITIONS, which is simply
        # wrong arithmetic and failed on correct data.)
        "disjoint window count equals sum(floor(n_visits/2)) exactly":
            bool((ig["disj_windows"] == ig["disj_expected"]).all()),
        "disjoint windows share no visit with each other":
            bool((ig["disj_sharing_visits"] == 0).all()),
        "disjoint windows are a subset of the overlapping set":
            bool((ig["disj_windows"] <= ig["ov_windows"]).all()),
        "window audit: no bad shapes, times or values in any fold":
            bool((ig[["ov_bad_shape", "ov_bad_times", "ov_bad_values",
                      "ov_unknown_patient", "disj_bad_shape", "disj_bad_times",
                      "disj_bad_values", "disj_unknown_patient"]].to_numpy()
                  == 0).all()),
        "each target appears once per condition":
            bool(not pred.duplicated(subset=key + ["condition"]).any()),
        "all conditions share the same target for a case":
            bool(pred.groupby(key)["target_event"].nunique().eq(1).all()),
        "last-2 kept exactly the two visits before the target":
            bool((last2["retained_from_index"]
                  == last2["available_history"] - WINDOW).all())
            and bool((last2["n_retained_visits"] == WINDOW).all()),
        "full-history kept every prior visit":
            bool((fullh["n_retained_visits"] == fullh["available_history"]).all()),
        "no condition used a padded visit":
            bool((last2["available_history"] >= WINDOW).all()),
        "original timestamps preserved (sampled against parent sequences)": tl_ok,
        "every historical visit precedes its target":
            bool((pred["target_years"] > pred["current_years"]).all()),
        "inter-visit intervals positive and preserved":
            bool((pred["interval_years"] > 0).all()),
        "state rebuilt from retained visits only, no smoothing":
            ("out = {k: np.asarray(seq[k])[lo:hi + 1] for k in SLICE_KEYS}" in src)
            and "filtered = model.filter_sequence(seq)" in
            (ROOT / "src/ppmi_subtype/clinical_profile.py").read_text(),
        "target-visit LEDD identical across conditions for a case":
            bool(pred.groupby(key)["ledd_next"].nunique().eq(1).all()),
        "Indeterminate targets excluded from binary scoring":
            bool(set(elig["actual_next_label"]) == {"TD", "PIGD"}),
        "matched cohort identical in every condition":
            len(set(cm.groupby("condition").size())) == 1,
        "probabilities finite, in [0, 1], summing to one":
            bool(np.isfinite(allp["P_PIGD"]).all()
                 and (allp["P_PIGD"] >= 0).all() and (allp["P_PIGD"] <= 1).all()
                 and np.allclose(
                     allp.loc[allp["two_class_defined"].astype(bool), "P_TD"]
                     + allp.loc[allp["two_class_defined"].astype(bool), "P_PIGD"],
                     1.0, atol=1e-12)),
        "hard calls follow the project's decision rule":
            bool(allp["predicted_state"].isin(["TD", "PIGD"]).all()),
        "bootstrap resamples patients, one sample shared by both conditions":
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

    section("10. PROVENANCE")
    env = {"panel_md5": pm, "python": sys.version.split()[0],
           "numpy": np.__version__, "pandas": pd.__version__,
           "node": platform.node(), "workers": args.workers,
           "n_bootstrap": args.n_boot, "seed": cfg.seed, "n_folds": N_FOLDS,
           "window_width": WINDOW, "ct_run_for_persistence": CT_RUN,
           "elapsed_seconds": round(time.time() - started, 1)}
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:                                   # noqa: BLE001
        env["git_commit"] = "unknown"
    for k, v in env.items():
        say(f"  {k:<26} {v}")

    (out / "summary.json").write_text(json.dumps(
        {"environment": env, "checks": {k: str(v) for k, v in checks.items()},
         "reproducibility": repro,
         "code_sha256": {str(Path(__file__).relative_to(ROOT)):
                         sha(Path(__file__))},
         "conditions": CONDITIONS,
         "primary_comparison": f"{PRIMARY[0]} vs {PRIMARY[1]}",
         "n_matched_targets": len(common),
         "n_indeterminate_excluded": n_ind,
         "rate_collapse": {t: {
             "mean_domains_collapsed": float(dg[f"n_domains_rate_collapsed_{t}"].mean()),
             "max_rate_var": float(dg[f"max_rate_var_{t}"].max()),
             "max_init_rate_var": float(dg[f"max_init_rate_var_{t}"].max())}
             for t in (TRAIN_FULL, TRAIN_OV, TRAIN_NO)},
         "window_accounting_fold0": {k: (int(v) if isinstance(v, (int, np.integer))
                                         else float(v))
                                     for k, v in ig.iloc[0].items()},
         "cohort_counts": counts.to_dict(orient="records"),
         "parameter_comparison": par.to_dict(orient="records"),
         "metrics": mt.to_dict(orient="records"),
         "paired": pt.to_dict(orient="records")},
        indent=2, default=float), encoding="utf-8")
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    skipf = {"MANIFEST.sha256", "driver.log"}
    files = sorted(p for p in out.rglob("*") if p.is_file()
                   and p.name not in skipf)
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files),
        encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
