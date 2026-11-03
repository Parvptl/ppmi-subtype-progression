"""Secondary: history windows for the rich clinical profile model.

The companion to `scripts/70_history_window_experiment.py`, which asked the
same question of the CT-HMM. This model does not consume subtype labels: it
tracks nine clinical domains with local-linear-trend states, so truncating its
history means **rebuilding the filtered clinical state from the retained visits
only**. That is what this script does -- it never reuses a filtered state that
has already absorbed visits outside the window.

Conditions, all predicting the same target visit:

    last 2 clinical visits / last 3 clinical visits / full clinical history

Feasibility, established by inspection before any code was written:
`ProfileModel.filter_domain` starts every patient from a fixed prior
(`m = 0`, `P = diag([INIT_LEVEL_VAR, init_rate_var])`, zero being the training
cohort mean on the standardised scale), and `predict_sequence` takes a sequence
dict and filters it. So a window is a slice of ``Y``, ``d``, ``times``,
``labels``, ``events`` and ``visit_index`` -- no architectural change. The
assumption this creates is the same one the CT-HMM faced: a truncated window
treats its first retained visit as a fresh start from the population prior.

Parameter policy, matching script 70: the model, the standardiser, the
population mean and the LOCF spread are fitted **once per fold on the training
patients' full sequences** and shared by every window. Only the held-out
patient's prediction-time history is truncated.

TARGET-VISIT LEDD -- disclosed, not hidden. `predict_next` takes the target
visit's prescribed LEDD (`d[k + 1]`) as an input. It is preserved identically
across all three windows so the comparison is fair, and a fourth, clearly
labelled condition repeats the full-history setting with the *current* visit's
LEDD instead, to show what that input is worth. No target-visit motor
measurement is ever used: those are the outcome.

    python scripts/71_profile_history_window.py --out <dir> [--workers N]
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
from ppmi_subtype.plots import GRID, _finish, _style  # noqa: E402

EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
WINDOWS: tuple = (2, 3, None)
WINDOW_LABEL = {2: "last 2", 3: "last 3", None: "full history"}
LEDD_SENSITIVITY = "full history, current-visit LEDD"
PRIMARY = (2, None)
MIN_CLASS = 5
N_FOLDS = 5

ACCENT = "#1f4e79"
COOL = "#2a78d6"
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


def _truncate(seq: dict, lo: int, hi: int, current_ledd: bool = False) -> dict:
    """Visits ``lo..hi`` inclusive, as a standalone sequence.

    ``visit_index`` and ``events`` are carried through unsliced in value, so a
    prediction stays traceable to the original visit it belongs to. With
    ``current_ledd`` the final element of ``d`` -- the target visit's
    prescribed dose -- is replaced by the current visit's, which is the LEDD
    sensitivity condition.
    """
    out = {k: np.asarray(seq[k])[lo:hi + 1] for k in SLICE_KEYS}
    out["gaps"] = np.diff(out["times"])
    out["PATNO"] = seq["PATNO"]
    if current_ledd and len(out["d"]) >= 2:
        d = np.array(out["d"], dtype=float)
        d[-1] = d[-2]
        out["d"] = d
    return out


def run_fold(test_patients, fold: int):
    """Fit on the training patients, then predict each window for the test."""
    sequences = _WORK["sequences"]
    test_set = set(test_patients)
    train = [s for s in sequences if s["PATNO"] not in test_set]
    test = [s for s in sequences if s["PATNO"] in test_set]
    if not train or not test:
        raise ValueError(f"empty split (fold {fold})")
    if test_set & {s["PATNO"] for s in train}:
        raise RuntimeError("held-out patient leaked into training")

    # Every learned transformation fitted here, on training patients only.
    standardiser = cp.Standardiser.fit(train)
    train_z = standardiser.apply(train)
    test_z = standardiser.apply(test)
    model = cp.fit(train_z, use_rate=True, use_medication=True)

    records = []
    for seq in test_z:
        n = len(seq["times"])
        for k in range(n - 1):
            n_hist = k + 1
            for w in WINDOWS:
                if w is not None and n_hist < w:
                    continue                      # not eligible; never padded
                lo = 0 if w is None else k - w + 1
                sub = _truncate(seq, lo, k + 1)
                recs = cp.predict_sequence(model, sub, standardiser,
                                           models=(cp.MODEL_FULL,))
                r = dict(recs[-1])                # the step whose target is k+1
                r["window"] = WINDOW_LABEL[w]
                r["window_size"] = 0 if w is None else w
                r["n_retained_visits"] = len(sub["times"]) - 1
                records.append(r)
            # LEDD sensitivity: full history, target dose replaced by current.
            sub = _truncate(seq, 0, k + 1, current_ledd=True)
            recs = cp.predict_sequence(model, sub, standardiser,
                                       models=(cp.MODEL_FULL,))
            r = dict(recs[-1])
            r["window"] = LEDD_SENSITIVITY
            r["window_size"] = -2
            r["n_retained_visits"] = len(sub["times"]) - 1
            records.append(r)

    frame = cp.records_to_frame(records)
    frame["window"] = [r["window"] for r in records]
    frame["window_size"] = [r["window_size"] for r in records]
    frame["n_retained_visits"] = [r["n_retained_visits"] for r in records]
    frame["fold"] = fold
    frame["n_train_patients"] = len(train)
    frame["fit_converged"] = bool(model.converged)
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
#  Metrics (same definitions as script 70)
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
    n0 = float(len(y) - n1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


BOOT_KEYS = ("accuracy", "balanced_accuracy", "auc", "brier", "log_loss")


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
    return {"n_targets": n, "n_TD": n_t, "n_PIGD": n_p,
            "accuracy": (tp + tn) / n if n else float("nan"),
            "balanced_accuracy": (sens_p + spec_p) / 2.0,
            "sensitivity_PIGD": sens_p, "specificity_PIGD": spec_p,
            "sensitivity_TD": spec_p, "specificity_TD": sens_p,
            "auc": float("nan") if small else _auc(y, p),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "tp_PIGD": tp, "tn_TD": tn,
            "fp": int(((pred == 1) & (y == 0)).sum()),
            "fn": int(((pred == 0) & (y == 1)).sum()),
            "metrics_limited_by_class_size": bool(small)}


def _patient_row_index(patients: np.ndarray) -> list:
    uniq, inv = np.unique(patients, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    si = inv[order]
    lo = np.searchsorted(si, np.arange(len(uniq)))
    hi = np.searchsorted(si, np.arange(len(uniq)), side="right")
    return [order[a:b] for a, b in zip(lo, hi)]


def patient_boot_all(patients, fn, n_boot=2000, seed=42):
    """Percentile CIs for every metric from one set of patient resamples."""
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
    clinical = cs.build_clinical_panel(cfg, panel)
    sequences = cp.build_sequences(clinical)
    patients = sorted(s["PATNO"] for s in sequences)
    say(f"  clinical sequences     {len(sequences):,} patients")
    say(f"  validation             {N_FOLDS}-fold patient CV, one repeat")
    say(f"  seed                   {cfg.seed}")
    say(f"  windows                {[WINDOW_LABEL[w] for w in WINDOWS]}")
    say(f"  LEDD sensitivity       '{LEDD_SENSITIVITY}'")
    say(f"  primary comparison     {WINDOW_LABEL[PRIMARY[0]]} vs "
        f"{WINDOW_LABEL[PRIMARY[1]]} (pre-specified)")
    say("  parameter policy       model, standardiser, population mean and")
    say("                         LOCF spread fitted once per fold on the")
    say("                         training patients' FULL sequences")

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(patients))
    pats = np.array(patients, dtype=object)
    tasks = [(f, pats[order[f::N_FOLDS]].tolist()) for f in range(N_FOLDS)]
    assert sum(len(t[1]) for t in tasks) == len(patients)
    assert len({p for _, ps in tasks for p in ps}) == len(patients)
    say(f"  fold sizes             {[len(t[1]) for t in tasks]}")

    section("2. CROSS-VALIDATION WITH TRUNCATED CLINICAL HISTORIES")
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
    pred["y"] = (pred["actual_next_label"] == "PIGD").astype(int)
    pred["pred_bin"] = (pred["predicted_state"] == "PIGD").astype(int)
    pred.to_csv(out / "profile_window_predictions.csv", index=False)

    section("3. COHORTS")
    elig = pred[pred["actual_next_label"].isin(["TD", "PIGD"])
                & pred["two_class_defined"].astype(bool)].copy()
    n_ind = int((pred["window"] == "full history").sum()
                - (elig["window"] == "full history").sum())
    say(f"  Indeterminate / undefined targets excluded per condition: {n_ind}")
    max_w = max(w for w in WINDOWS if w is not None)
    by_cond = {w: set(map(tuple, g[["PATNO", "visit_index"]].to_numpy()))
               for w, g in elig.groupby("window")}
    common = set.intersection(*by_cond.values()) if by_cond else set()
    long_enough = set(map(tuple, elig[elig["n_history"] >= max_w]
                          [["PATNO", "visit_index"]].drop_duplicates().to_numpy()))
    common &= long_enough
    elig["common"] = [tuple(r) in common
                      for r in elig[["PATNO", "visit_index"]].to_numpy()]
    say(f"  common-target cohort   {len(common):,} targets "
        f"(n_history >= {max_w}, eligible in every condition)")
    say("\n  eligible targets per condition:")
    say(block(elig.groupby("window").agg(
        n_targets=("y", "size"), n_patients=("PATNO", "nunique")).reset_index(),
        index=False))

    section("4. METRICS")
    order_c = [WINDOW_LABEL[w] for w in WINDOWS] + [LEDD_SENSITIVITY]
    recs = []
    for cohort, sel in (("all eligible", elig),
                        ("common targets", elig[elig["common"]])):
        for w in order_c:
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
                m[f"{k}_lo"], m[f"{k}_hi"] = lo, hi
            recs.append(m)
    mt = pd.DataFrame(recs)
    mt.round(4).to_csv(out / "profile_window_metrics.csv", index=False)
    for cohort in ("all eligible", "common targets"):
        say(f"\n  --- {cohort} ---")
        say(block(mt[mt.cohort == cohort][
            ["window", "n_targets", "n_patients", "n_PIGD", "accuracy",
             "balanced_accuracy", "sensitivity_PIGD", "specificity_PIGD",
             "auc", "brier", "log_loss"]].round(4), index=False))

    section("5. PAIRED COMPARISONS, PATIENT-CLUSTERED")
    cm = elig[elig["common"]]

    def paired(a, b, role):
        wide = cm[cm["window"].isin([a, b])].pivot_table(
            index=["PATNO", "visit_index", "y"], columns="window",
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

        rec = {"role": role, "comparison": f"{a} - {b}",
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
        paired(WINDOW_LABEL[PRIMARY[0]], WINDOW_LABEL[PRIMARY[1]],
               "PRIMARY (pre-specified)"),
        paired("last 3", "full history", "exploratory"),
        paired(LEDD_SENSITIVITY, "full history", "LEDD sensitivity"),
    ) if r]
    pt = pd.DataFrame(prs)
    pt.round(4).to_csv(out / "profile_paired_comparisons.csv", index=False)
    say("")
    say(block(pt[["role", "comparison", "n_targets", "n_patients", "accuracy",
                  "accuracy_lo", "accuracy_hi", "balanced_accuracy", "auc",
                  "brier", "log_loss"]].round(4), index=False))

    section("6. FIGURE")
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
    xs = np.arange(len(order_c))
    for ax, key, lab in ((axes[0], "accuracy", "Accuracy"),
                         (axes[1], "balanced_accuracy", "Balanced accuracy"),
                         (axes[2], "auc", "ROC-AUC")):
        _style(ax)
        d = mt[mt.cohort == "common targets"].set_index("window")
        v = np.array([d.loc[w, key] if w in d.index else np.nan
                      for w in order_c], float)
        lo = np.array([d.loc[w, f"{key}_lo"] if w in d.index else np.nan
                       for w in order_c], float)
        hi = np.array([d.loc[w, f"{key}_hi"] if w in d.index else np.nan
                       for w in order_c], float)
        colours = [ACCENT if w != LEDD_SENSITIVITY else MUTED for w in order_c]
        for i in range(len(order_c)):
            ax.errorbar(xs[i], v[i], yerr=[[v[i] - lo[i]], [hi[i] - v[i]]],
                        fmt="o", color=colours[i], markersize=6, capsize=3,
                        elinewidth=1.3, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([w.replace(", ", ",\n") for w in order_c],
                           rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(lab)
        ax.set_title(lab, fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle("Rich clinical profile: how much clinical history does the "
                 "filter need?\ncommon-target cohort, 95% patient-clustered "
                 "intervals; grey is the LEDD sensitivity condition",
                 fontsize=10.5, x=0.01, y=0.995, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    say(f"  wrote {_finish(fig, out / 'fig1_profile_windows.png').name}")

    section("7. CHECKS")
    src = Path(__file__).read_text()
    full = elig[elig["window"] == "full history"]
    checks = {
        "every fold completed": len(failed) == 0,
        "every fit converged": bool(pred["fit_converged"].all()),
        "no window used a padded or fabricated visit": bool(all(
            (pred[pred["window"] == WINDOW_LABEL[w]]["n_history"] >= w).all()
            for w in WINDOWS if w is not None)),
        "filtered state rebuilt from retained visits only (slice, not reuse)":
            'out = {k: np.asarray(seq[k])[lo:hi + 1] for k in SLICE_KEYS}' in src,
        "prediction uses the causal filter, never a smoother":
            'filtered = model.filter_sequence(seq)' in
            (ROOT / "src/ppmi_subtype/clinical_profile.py").read_text(),
        "held-out patient never in training":
            'raise RuntimeError("held-out patient leaked into training")' in src,
        "standardiser fitted on training patients only":
            "standardiser = cp.Standardiser.fit(train)" in src,
        "each patient held out exactly once":
            bool((pred.groupby("PATNO")["fold"].nunique() == 1).all()),
        "parameters fitted on a strict subset of patients":
            bool((pred["n_train_patients"] < len(sequences)).all()),
        "all conditions predict the same target for a given case":
            bool(pred.groupby(["PATNO", "visit_index"])["target_event"]
                 .nunique().eq(1).all()),
        "retained gaps are real elapsed times":
            bool(np.isfinite(full["interval_years"]).all()
                 and (full["interval_years"] > 0).all()),
        "target visit strictly after the current visit":
            bool((pred["target_years"] > pred["current_years"]).all()),
        "Indeterminate targets excluded from binary scoring":
            bool(set(elig["actual_next_label"]) == {"TD", "PIGD"}),
        "common cohort identical in every condition":
            len(set(elig[elig["common"]].groupby("window").size())) == 1,
        "target-visit LEDD used identically in the three window conditions":
            bool(pred[pred["window"].isin(
                [WINDOW_LABEL[w] for w in WINDOWS])]
                .groupby(["PATNO", "visit_index"])["ledd_next"]
                .nunique().eq(1).all()),
        "LEDD sensitivity condition really swapped the dose": bool(
            np.allclose(
                pred[pred["window"] == LEDD_SENSITIVITY]
                .sort_values(["PATNO", "visit_index"])["ledd_next"].to_numpy(float),
                pred[pred["window"] == LEDD_SENSITIVITY]
                .sort_values(["PATNO", "visit_index"])["ledd_current"].to_numpy(float))),
        "probabilities finite and in [0, 1]": bool(
            np.isfinite(pred["P_PIGD"]).all() and (pred["P_PIGD"] >= 0).all()
            and (pred["P_PIGD"] <= 1).all()),
        "bootstrap resamples patients, not targets":
            "groups = _patient_row_index(np.asarray(patients))" in src,
    }
    for n, v in checks.items():
        say(f"    [{'PASS' if v else 'FAIL'}]  {n}")
    pd.DataFrame([{"check": k, "passed": bool(v)} for k, v in checks.items()]
                 ).to_csv(out / "validation_checks.csv", index=False)
    if not all(checks.values()):
        raise SystemExit(f"CRITICAL CHECK FAILED: "
                         f"{[k for k, v in checks.items() if not v]}")

    section("8. PROVENANCE")
    env = {"panel_md5": pm, "python": sys.version.split()[0],
           "numpy": np.__version__, "pandas": pd.__version__,
           "node": platform.node(), "workers": args.workers,
           "n_bootstrap": args.n_boot, "seed": cfg.seed, "n_folds": N_FOLDS,
           "elapsed_seconds": round(time.time() - started, 1)}
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:                                   # noqa: BLE001
        env["git_commit"] = "unknown"
    for k, v in env.items():
        say(f"  {k:<20} {v}")

    (out / "summary.json").write_text(json.dumps(
        {"environment": env, "checks": checks,
         "code_sha256": {str(Path(__file__).relative_to(ROOT)):
                         sha(Path(__file__))},
         "n_common_targets": len(common),
         "n_indeterminate_excluded": n_ind,
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
