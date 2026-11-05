"""Does the ratio-band width matter? 0.05, 0.10, 0.20 and 0.30 under LOPO.

The question
------------
The ratio-banded CT-HMM
(``experiments/20261104T152154Z_cthmm_ratio_banded_observations``) put the
ratio into fixed 0.1-wide bands over ``[0, 3)`` and found no gain in next-visit
accuracy, with the leading suspect being sparsity: eight of its 44 joint
categories carried fewer than fifteen observations, nearly all of them
main-range TD bands whose width the experiment had fixed in advance. This
experiment varies that width and nothing else.

    0.05   60 main bands    finer: does more ratio detail help?
    0.10   30 main bands    the published configuration, the baseline here
    0.20   15 main bands    coarser: do fewer, better-estimated categories help?
    0.30   10 main bands    coarser still

What is held fixed
------------------
Everything. Three hidden states, the continuous-time generator, interval-aware
transitions, forward filtering, the cohort, eligibility, the conditional
two-class target, the Indeterminate handling, the seed, the restarts, the
emission floor, the factorised emission ``E[a,(s,b)] = M[a,s] C[a,s,b]``, the
tail-merging rule (0.25 candidates, >=30 observations from >=20 training
patients) and the LOPO protocol. Each width's arm differs from the published
0.1 arm in its main-range band edges and in nothing else, which is asserted
rather than assumed: the 0.10 arm here must reproduce the published run.

A subtype-only CT-HMM is refitted in every fold as a reference point, so the
widths can be read against the model that has no ratio information at all.
The control arm is not repeated --- the previous experiment showed the joint
code path reproduces the square model exactly.

Selection
---------
Comparing several configurations on held-out data and then naming a winner is
selection on the evaluation set. This driver therefore reports them all with
paired intervals and never picks one by a numerical margin; the report says
plainly that a chosen width's held-out metric is optimistically biased.

    python scripts/90_cthmm_ratio_band_width_sensitivity.py \
        --out experiments/<UTC>_cthmm_ratio_band_width_sensitivity --workers 24
    # interrupted? completed folds are checkpointed inside --out
    python scripts/90_cthmm_ratio_band_width_sensitivity.py --out <same> --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmi_subtype import (  # noqa: E402
    hmm_banded, loo, next_visit as nv, next_visit_banded as nvb,
    ranking_metrics as rm, ratio_bands as rb,
)
from ppmi_subtype.config import load_config  # noqa: E402
from ppmi_subtype.plots import GRID, INK_MUTED, SURFACE, _finish, _style  # noqa: E402

EXPECTED_PANEL_MD5 = "fb038ddbdc19ab3368e13940f398e7ba"
PUBLISHED_RUN = "experiments/20261104T152154Z_cthmm_ratio_banded_observations"
PUBLISHED_BANDED = "CT-HMM (subtype x ratio band)"

#: The pre-specified configurations, and the default for --widths. 0.10 is the
#: published one and the base every contrast is taken against.
WIDTHS = (0.05, 0.10, 0.20, 0.30)
BASE_WIDTH = 0.10

#: Colour per width, fine to coarse, plus the no-ratio reference.
WIDTH_COLOUR = {0.05: "#1f4e79", 0.10: "#2a78d6", 0.20: "#eb6834",
                0.30: "#b03a2e", 0.50: "#6b2d5b"}
UNLISTED_WIDTH_COLOUR = "#4a4a48"
REFERENCE_COLOUR = "#7a7a78"


def _colour(model_name: str) -> str:
    """The arm's colour. An unlisted width still plots, in grey."""
    w = float(model_name.split("width ")[1].rstrip(")"))
    return WIDTH_COLOUR.get(w, UNLISTED_WIDTH_COLOUR)
NEUTRAL = "#b9b8b4"

_WORK: dict = {}
lines: list[str] = []


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


def arm(width: float) -> str:
    return nvb.band_width_model(width)


def short(name: str) -> str:
    """A short label for plots and tables."""
    if name == nvb.MODEL_LABEL:
        return "subtype only"
    return f"width {float(name.split('width ')[1].rstrip(')')):.2f}"


# ---------------------------------------------------------------------------
#  Fold workers
# ---------------------------------------------------------------------------
def _init(data, bdata, cases, seed, n_restarts, emission_floor, widths) -> None:
    _WORK.update(data=data, bdata=bdata, cases=cases, seed=seed,
                 n_restarts=n_restarts, emission_floor=emission_floor,
                 widths=widths)


def _fold(patno):
    try:
        frame, diag = nvb.run_fold_widths(
            _WORK["data"], _WORK["bdata"], _WORK["cases"], patno, _WORK["seed"],
            _WORK["n_restarts"], _WORK["emission_floor"], _WORK["widths"])
        return frame, diag, None
    except Exception as exc:                 # noqa: BLE001 -- recorded, not hidden
        return None, None, f"{type(exc).__name__}: {exc}"


def run_folds(data, bdata, cases, patients, seed, n_restarts, emission_floor,
              widths, workers, out: Path, checkpoint_every: int = 25,
              resume: bool = False):
    """The LOPO loop. Completed folds are checkpointed so a stop can be resumed."""
    ckpt = out / "predictions_checkpoint.csv"
    ckpt_diag = out / "fold_diagnostics_checkpoint.csv"

    done: set = set()
    frames: list[pd.DataFrame] = []
    diags: list[dict] = []
    if resume and ckpt.exists():
        prior = pd.read_csv(ckpt, low_memory=False)
        prior = prior[prior["held_out_patient"].isin(patients)]
        if len(prior):
            done = set(prior["held_out_patient"].unique())
            frames.append(prior)
            say(f"  resuming: {len(done):,} folds already complete")
        if ckpt_diag.exists():
            pd_prior = pd.read_csv(ckpt_diag)
            diags = pd_prior[pd_prior["held_out_patient"].isin(done)].to_dict("records")

    pending = [p for p in patients if p not in done]
    failures: list[dict] = []
    buffer: list[pd.DataFrame] = []

    def flush() -> None:
        if buffer:
            frames.extend(buffer)
            buffer.clear()
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(ckpt, index=False)
        if diags:
            pd.DataFrame(diags).to_csv(ckpt_diag, index=False)

    args = (data, bdata, cases, seed, n_restarts, emission_floor, widths)
    started = time.time()
    if workers <= 1:
        _init(*args)
        for n, patno in enumerate(pending, start=1):
            frame, diag, error = _fold(patno)
            if error is None:
                buffer.append(frame)
                diags.extend(diag)
            else:
                failures.append({"held_out_patient": patno, "error": error})
            if n % 25 == 0:
                say(f"  fold {n:,}/{len(pending):,}")
            if n % checkpoint_every == 0:
                flush()
    else:
        with mp.get_context("fork").Pool(workers, initializer=_init,
                                         initargs=args) as pool:
            for n, (frame, diag, error) in enumerate(
                    pool.imap(_fold, pending, chunksize=1), start=1):
                if error is None:
                    buffer.append(frame)
                    diags.extend(diag)
                else:
                    failures.append({"held_out_patient": pending[n - 1],
                                     "error": error})
                if n % 50 == 0:
                    rate = (time.time() - started) / n
                    left = rate * (len(pending) - n) / 60
                    say(f"  fold {n:,}/{len(pending):,}  "
                        f"({time.strftime('%H:%M:%S')}, ~{left:.0f} min left)")
                if n % checkpoint_every == 0:
                    flush()
    flush()

    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return (predictions, pd.DataFrame(diags),
            pd.DataFrame(failures, columns=["held_out_patient", "error"]))


# ---------------------------------------------------------------------------
#  Sparsity and complexity, whole cohort
# ---------------------------------------------------------------------------
def complexity_table(bdata, widths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Categories, parameters and sparsity per width, over the whole cohort.

    Descriptive: it uses every patient, so it is a property of the alphabet and
    the data, not a held-out result. The per-fold alphabets differ from it only
    where the tail merge differs, which `fold_band_structures.csv` records.
    """
    rows, occupancy = [], []
    for w in widths:
        scheme, alpha, _ = nvb.fold_alphabet(bdata, None, main_width=w)
        counts = np.zeros(alpha.n_categories, dtype=int)
        patients: dict[int, set] = {}
        for s in bdata.sequences:
            codes = alpha.code(s["labels"], scheme.band_index(s["ratios"]))
            for c in codes:
                counts[c] += 1
                patients.setdefault(int(c), set()).add(s["PATNO"])
        n_pat = np.array([len(patients.get(k, ())) for k in range(alpha.n_categories)])

        straddling = sum(
            1 for b in range(scheme.n_bands)
            if sum(rb.band_allows_subtype(scheme, b, n)
                   for n in bdata.subtype_names) > 1)
        rows.append({
            "main_width": w,
            "n_main_bands": int(scheme.n_main_bands),
            "n_tail_bands": len(scheme.tail_lows),
            "n_bands": int(scheme.n_bands),
            "n_categories": int(alpha.n_categories),
            "n_params": int(hmm_banded.n_params(alpha, 3)),
            "block_TD": len(alpha.blocks[0]),
            "block_Indeterminate": len(alpha.blocks[1]),
            "block_PIGD": len(alpha.blocks[2]),
            "n_straddling_bands": straddling,
            "n_empty": int((counts == 0).sum()),
            "n_lt_10_obs": int((counts < 10).sum()),
            "n_lt_15_obs": int((counts < 15).sum()),
            "n_lt_30_obs": int((counts < 30).sum()),
            "n_lt_20_patients": int((n_pat < 20).sum()),
            "median_obs_per_category": float(np.median(counts)),
            "min_obs_nonempty": int(counts[counts > 0].min()) if (counts > 0).any() else 0,
            "obs_per_param": round(4295 / hmm_banded.n_params(alpha, 3), 2),
        })
        for k, label in enumerate(alpha.labels):
            occupancy.append({"main_width": w, "category": label,
                              "n_observations": int(counts[k]),
                              "n_patients": int(n_pat[k])})
    return pd.DataFrame(rows), pd.DataFrame(occupancy)


# ---------------------------------------------------------------------------
#  Ranking metrics per arm
# ---------------------------------------------------------------------------
def ranking_table(predictions: pd.DataFrame, models, n_boot: int) -> pd.DataFrame:
    rows = []
    for name in models:
        s = nv.scored(predictions[predictions["model"] == name])
        s = s.sort_values(["PATNO", "current_event", "target_event"])
        y = (s["actual_next_label"] == "PIGD").to_numpy()
        score = s["P_PIGD"].to_numpy(float)
        patno = s["PATNO"].to_numpy()
        a_lo, a_hi, a_n = rm.patient_bootstrap(score, y, patno, rm.auc_roc,
                                               n_boot=n_boot)
        p_lo, p_hi, _ = rm.patient_bootstrap(score, y, patno, rm.pr_auc,
                                             n_boot=n_boot)
        rows.append({
            "model": name, "arm": short(name), "n": len(s),
            "n_patients": int(s["PATNO"].nunique()),
            "auc_roc": rm.auc_roc(y, score), "auc_ci_low": a_lo, "auc_ci_high": a_hi,
            "pr_auc": rm.pr_auc(y, score), "pr_ci_low": p_lo, "pr_ci_high": p_hi,
            "partial_auc_fpr_le_0.2": rm.partial_auc(y, score, 0.2),
            **rm.operating_point(y, (s["predicted_state"] == "PIGD").to_numpy()),
            "n_bootstrap": a_n,
        })
    return pd.DataFrame(rows)


def ranking_contrasts(predictions, models, base, n_boot) -> pd.DataFrame:
    """Every arm against the 0.10 baseline, paired on patients."""
    key = ["PATNO", "current_event", "target_event"]

    def arm_scored(name):
        s = nv.scored(predictions[predictions["model"] == name])
        return s.sort_values(key).reset_index(drop=True)

    b = arm_scored(base)
    y = (b["actual_next_label"] == "PIGD").to_numpy()
    patno = b["PATNO"].to_numpy()
    rows = []
    for name in models:
        if name == base:
            continue
        a = arm_scored(name)
        if not (a[key].to_numpy() == b[key].to_numpy()).all():
            raise SystemExit(f"{name} and {base} do not score identical rows")
        out = {"model_a": name, "arm_a": short(name), "model_b": base,
               "arm_b": short(base), "n_paired": len(a),
               "n_patients": int(b["PATNO"].nunique())}
        for label, metric in (("auc", rm.auc_roc), ("pr_auc", rm.pr_auc),
                              ("partial_auc", lambda yy, ss:
                               rm.partial_auc(yy, ss, 0.2))):
            d = rm.patient_bootstrap_difference(
                a["P_PIGD"].to_numpy(float), b["P_PIGD"].to_numpy(float),
                y, patno, metric, n_boot=n_boot)
            out[f"{label}_delta"] = d["delta"]
            out[f"{label}_ci_low"] = d["ci_low"]
            out[f"{label}_ci_high"] = d["ci_high"]
            out[f"{label}_includes_zero"] = d["ci_includes_zero"]
        rows.append(out)
    return pd.DataFrame(rows)


def auc_by_stratum(predictions, models) -> pd.DataFrame:
    key = ["PATNO", "current_event", "target_event"]
    frames = {n: nv.scored(predictions[predictions["model"] == n])
              .sort_values(key).reset_index(drop=True) for n in models}
    ref = frames[models[0]]
    y = (ref["actual_next_label"] == "PIGD").to_numpy()
    rows = []
    for field, label in (("current_label", "current label"),
                         ("n_history", "prior visits")):
        col = (ref[field].clip(upper=5).to_numpy() if field == "n_history"
               else ref[field].to_numpy())
        levels = (nv.STATE_NAMES if field == "current_label"
                  else sorted(np.unique(col)))
        for level in levels:
            m = (col == level)
            row = {"stratum": label,
                   "level": (f"{int(level)}+" if field == "n_history" and level == 5
                             else str(level)),
                   "n": int(m.sum()), "n_positive": int(y[m].sum())}
            usable = m.sum() >= 30 and 0 < y[m].sum() < m.sum()
            for name in models:
                s = frames[name]["P_PIGD"].to_numpy(float)[m]
                row[f"auc_{short(name).replace(' ', '_')}"] = (
                    rm.auc_roc(y[m], s) if usable else np.nan)
                row[f"acc_{short(name).replace(' ', '_')}"] = (
                    float(frames[name]["correct"].to_numpy()[m].mean())
                    if m.sum() else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Plots
# ---------------------------------------------------------------------------
def _bars(ax, labels, values, los, his, colours, ylabel, title, fmt="{:.4f}"):
    x = np.arange(len(labels))
    for i, (v, lo, hi, c) in enumerate(zip(values, los, his, colours)):
        ax.bar(x[i], v, width=0.62, color=c, zorder=3)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([x[i], x[i]], [lo, hi], color="#2b2b2b", linewidth=1.4, zorder=4)
            ax.plot([x[i] - 0.1, x[i] + 0.1], [lo, lo], color="#2b2b2b",
                    linewidth=1.4, zorder=4)
            ax.plot([x[i] - 0.1, x[i] + 0.1], [hi, hi], color="#2b2b2b",
                    linewidth=1.4, zorder=4)
        # The whisker runs through the bar centre, so the value label needs a
        # halo to stay readable where the two cross.
        ax.text(x[i], v, fmt.format(v), ha="center", va="bottom", fontsize=8,
                color=INK_MUTED, zorder=5,
                path_effects=[pe.withStroke(linewidth=2.6, foreground=SURFACE)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Scale to the data, never to a hard-coded window: a run whose numbers fall
    # outside it would silently draw clipped bars and labels off the axes. The
    # axis is truncated rather than starting at zero, so differences of a
    # fraction of a point stay visible; the title carries the interval.
    seen = [v for v in list(values) + list(los) + list(his) if np.isfinite(v)]
    span = max(max(seen) - min(seen), 1e-3)
    ax.set_ylim(min(seen) - 0.45 * span, max(seen) + 0.30 * span)


def write_plots(out: Path, metrics, ranking, complexity, breakdowns,
                by_stratum, models) -> list[str]:
    written = []
    labels = [short(n) for n in models]
    colours = [REFERENCE_COLOUR if n == nvb.MODEL_LABEL else _colour(n)
               for n in models]
    m = metrics.set_index("model").loc[list(models)]
    r = ranking.set_index("model").loc[list(models)]

    # 1 -- accuracy
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _style(ax)
    _bars(ax, labels, m["accuracy"], m["accuracy_ci_patient_bootstrap_low"],
          m["accuracy_ci_patient_bootstrap_high"], colours, "accuracy",
          "Next-visit accuracy by main-range band width\n"
          f"{int(m['n_scored'].iloc[0]):,} held-out cases, 95% patient-bootstrap CI")
    written.append(_finish(fig, out / "plot1_accuracy.png").name)

    # 2 -- AUC-ROC
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _style(ax)
    _bars(ax, labels, r["auc_roc"], r["auc_ci_low"], r["auc_ci_high"], colours,
          "AUC-ROC",
          "Discrimination (AUC-ROC) by main-range band width\n"
          "PIGD positive, conditional P(PIGD), 95% patient-bootstrap CI")
    written.append(_finish(fig, out / "plot2_auc_roc.png").name)

    # 3 -- PR-AUC
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _style(ax)
    _bars(ax, labels, r["pr_auc"], r["pr_ci_low"], r["pr_ci_high"], colours,
          "PR-AUC (average precision)",
          "Precision-recall area by main-range band width\n"
          "PIGD is the minority class at 30.2%")
    written.append(_finish(fig, out / "plot3_pr_auc.png").name)

    # 4 -- Brier and log loss
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, col, name in ((axes[0], "brier_two_class", "Brier score"),
                          (axes[1], "log_loss", "log loss")):
        _style(ax)
        _bars(ax, labels, m[col], [np.nan] * len(labels), [np.nan] * len(labels),
              colours, name, f"{name} (lower is better)")
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle("Probabilistic performance by main-range band width",
                 fontsize=10, x=0.02, ha="left")
    written.append(_finish(fig, out / "plot4_brier_logloss.png").name)

    # 5 -- complexity
    c = complexity.sort_values("main_width")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    _style(axes[0])
    axes[0].plot(c["main_width"], c["n_categories"], "o-", color="#1f4e79",
                 label="joint categories", zorder=3)
    axes[0].plot(c["main_width"], c["n_params"], "s-", color="#eb6834",
                 label="free parameters", zorder=3)
    for _, row in c.iterrows():
        axes[0].annotate(f"{int(row['n_params'])}", (row["main_width"], row["n_params"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=8,
                         color=INK_MUTED)
    axes[0].set_xlabel("main-range band width")
    axes[0].set_ylabel("count")
    axes[0].set_title("Alphabet size and model complexity", fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[0].grid(color=GRID, linewidth=0.8, zorder=0)
    axes[0].set_axisbelow(True)

    _style(axes[1])
    for col, colour, lab in (("n_lt_30_obs", "#b03a2e", "< 30 observations"),
                             ("n_lt_15_obs", "#eb6834", "< 15 observations"),
                             ("n_empty", "#1f4e79", "empty")):
        axes[1].plot(c["main_width"], c[col], "o-", color=colour, label=lab, zorder=3)
    axes[1].set_xlabel("main-range band width")
    axes[1].set_ylabel("categories")
    axes[1].set_title("Sparse categories (whole cohort, 4,295 observations)",
                      fontsize=10, loc="left")
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[1].grid(color=GRID, linewidth=0.8, zorder=0)
    axes[1].set_axisbelow(True)
    written.append(_finish(fig, out / "plot5_complexity.png").name)

    # 6 -- by history length
    hist = breakdowns["by_history_length"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    _style(axes[0])
    order = sorted(hist["n_prior_visits"].unique(),
                   key=lambda v: int(str(v).rstrip("+")))
    x = np.arange(len(order))
    for name in models:
        g = hist[hist["model"] == name].set_index("n_prior_visits").reindex(order)
        colour = (REFERENCE_COLOUR if name == nvb.MODEL_LABEL
                  else _colour(name))
        axes[0].plot(x, g["accuracy"], "o-", color=colour, label=short(name), zorder=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order)
    axes[0].set_xlabel("prior visits")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Accuracy by history length", fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(color=GRID, linewidth=0.8, zorder=0)
    axes[0].set_axisbelow(True)

    _style(axes[1])
    hs = by_stratum[by_stratum["stratum"] == "prior visits"]
    hs = hs.set_index("level").reindex(order)
    for name in models:
        col = f"auc_{short(name).replace(' ', '_')}"
        colour = (REFERENCE_COLOUR if name == nvb.MODEL_LABEL
                  else _colour(name))
        axes[1].plot(x, hs[col], "o-", color=colour, label=short(name), zorder=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order)
    axes[1].set_xlabel("prior visits")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("Discrimination by history length", fontsize=10, loc="left")
    axes[1].grid(color=GRID, linewidth=0.8, zorder=0)
    axes[1].set_axisbelow(True)
    fig.suptitle("Does band width matter more when the history is short?",
                 fontsize=10, x=0.02, ha="left")
    written.append(_finish(fig, out / "plot6_history_length.png").name)
    return written


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0,
                    help="fit only the first N patients (smoke test)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--widths", type=float, nargs="+", default=list(WIDTHS),
                    help="main-range band widths to fit. The default is the "
                         "four pre-specified widths; passing others is an "
                         "extension and should be reported as one.")
    args = ap.parse_args()
    widths = tuple(sorted(args.widths))
    if BASE_WIDTH not in widths:
        raise SystemExit(f"--widths must include the {BASE_WIDTH} baseline, "
                         "which every contrast is taken against")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # ------------------------------------------------------- 1. integrity
    section("1. DATA INTEGRITY")
    cfg = load_config()
    panel_path = cfg.path("outputs") / "visit_panel.csv"
    if not panel_path.exists():
        raise SystemExit(
            f"the frozen panel is not at {panel_path}. This experiment needs the "
            "real PPMI panel; it will not substitute or synthesise one.")
    panel_md5 = md5(panel_path)
    say(f"  panel                {panel_path.relative_to(ROOT)}")
    say(f"  md5                  {panel_md5}")
    if panel_md5 != EXPECTED_PANEL_MD5:
        raise SystemExit(
            f"panel md5 {panel_md5} != expected {EXPECTED_PANEL_MD5}. Refusing to "
            "run on a different dataset.")
    say(f"  expected md5         {EXPECTED_PANEL_MD5}  [MATCH]")

    panel = pd.read_csv(panel_path, low_memory=False)
    states = [cfg.subtype.labels[s] for s in nv.STATE_NAMES]
    data = loo.prepare(panel, states)
    bdata = nvb.prepare(panel, data, cfg.subtype)
    cases = nv.build_cases(data)
    patients = [s["PATNO"] for s in data.sequences]

    ratios = np.concatenate([s["ratios"] for s in bdata.sequences])
    finite = np.isfinite(ratios)
    integrity = pd.DataFrame([
        {"check": "panel md5 matches the frozen panel", "value": panel_md5,
         "pass": panel_md5 == EXPECTED_PANEL_MD5},
        {"check": "patients in the LOPO cohort", "value": len(patients),
         "pass": len(patients) > 0},
        {"check": "next-visit cases built", "value": len(cases), "pass": len(cases) > 0},
        {"check": "eligible cases", "value": int(cases["eligible"].sum()),
         "pass": bool(cases["eligible"].sum())},
        {"check": "labelled visits", "value": len(ratios), "pass": len(ratios) > 0},
        {"check": "computable ratios", "value": int(finite.sum()), "pass": True},
        {"check": "undefined ratios (pigd_score == 0)",
         "value": int((~finite).sum()), "pass": True},
        {"check": "ratios at or above the tail start",
         "value": int((ratios[finite] >= rb.TAIL_START).sum()), "pass": True},
        {"check": "max ratio", "value": round(float(np.nanmax(ratios)), 4),
         "pass": np.isfinite(np.nanmax(ratios))},
        {"check": "sequences sorted by time, no duplicate visits",
         "value": "asserted in loo.prepare", "pass": True},
    ])
    say(block(integrity, index=False))
    if not integrity["pass"].all():
        raise SystemExit("data-integrity checks failed; refusing to continue")

    # --------------------------------------------- 2. the configurations
    section(f"2. THE {len(widths)} CONFIGURATIONS")
    say("  Main range [0.0, 3.0) in fixed-width bands, left-closed and")
    say("  right-open: a ratio on an edge opens that band, so no observation")
    say("  can fall in two. The tail rule is the published one and identical in")
    say("  every configuration -- 0.25-wide candidates from 3.0, merged upward")
    say(f"  until each holds >= {rb.MIN_TAIL_OBSERVATIONS} observations from "
        f">= {rb.MIN_TAIL_PATIENTS} training")
    say("  patients, topmost band open. Only the main-range width varies.")
    say("")
    complexity, occupancy = complexity_table(bdata, widths)
    say(block(complexity[["main_width", "n_main_bands", "n_tail_bands",
                          "n_categories", "n_params", "block_TD",
                          "block_Indeterminate", "block_PIGD",
                          "n_straddling_bands"]], index=False))
    say("")
    say("  Sparsity over the whole cohort's 4,295 labelled visits:")
    say(block(complexity[["main_width", "n_categories", "n_empty", "n_lt_10_obs",
                          "n_lt_15_obs", "n_lt_30_obs", "n_lt_20_patients",
                          "median_obs_per_category", "obs_per_param"]], index=False))
    say("")
    say("  n_straddling_bands counts bands a subtype threshold cuts through, so")
    say("  the same band can carry two subtypes. It is not monotone in the width")
    say("  because 0.90 and 1.15 fall on the grid at some widths and not others:")
    say("  a confound of fixed-width banding, recorded rather than removed.")
    complexity.to_csv(out / "complexity_by_width.csv", index=False)
    occupancy.to_csv(out / "category_occupancy_cohort.csv", index=False)

    # whole-cohort band definitions, for the record
    band_rows = []
    for w in widths:
        scheme, alpha, tail_diag = nvb.fold_alphabet(bdata, None, main_width=w)
        for b in range(scheme.n_bands):
            lo, hi = scheme.band_bounds(b)
            band_rows.append({
                "main_width": w, "band_index": b, "label": scheme.band_label(b),
                "low": lo, "high": hi,
                "region": "main" if lo < rb.TAIL_START else "tail",
                "subtypes_possible": ",".join(
                    n for n in bdata.subtype_names
                    if rb.band_allows_subtype(scheme, b, n)),
            })
        tail_diag.insert(0, "main_width", w)
        tail_diag.to_csv(out / f"tail_merge_width_{w:.2f}.csv", index=False)
    pd.DataFrame(band_rows).to_csv(out / "band_definitions.csv", index=False)
    say(f"\n  wrote band_definitions.csv ({len(band_rows):,} bands across "
        f"{len(widths)} configurations) and one tail-merge table per width")

    # ------------------------------------------------------------ 3. LOPO
    section("3. LEAVE-ONE-PATIENT-OUT")
    if args.limit:
        patients = patients[:args.limit]
        say(f"  SMOKE TEST: {len(patients)} patients only, not a result")
    say(f"  patients             {len(patients):,}")
    say(f"  arms per fold        {len(widths) + 1} "
        f"(subtype-only reference + {len(widths)} widths)")
    say(f"  seed                 {cfg.seed}")
    say(f"  restarts per fit     {cfg.model['n_restarts']}")
    say(f"  emission floor       {cfg.model['emission_floor']}  "
        "(on the subtype marginal M, as published)")
    say(f"  workers              {args.workers}")
    say("")
    predictions, fold_diag, failures = run_folds(
        data, bdata, cases, patients, cfg.seed, cfg.model["n_restarts"],
        cfg.model["emission_floor"], list(widths), args.workers, out,
        resume=args.resume)

    models = (nvb.MODEL_LABEL, *[arm(w) for w in widths])
    say(f"\n  folds completed      {predictions['held_out_patient'].nunique():,}"
        f" / {len(patients):,}")
    say(f"  folds failed         {len(failures):,}")
    say(f"  prediction rows      {len(predictions):,}")
    if len(failures):
        say(block(failures, index=False))
    failures.to_csv(out / "failed_folds.csv", index=False)
    predictions.to_csv(out / "next_visit_predictions.csv", index=False)
    fold_diag.to_csv(out / "fold_band_structures.csv", index=False)

    # ------------------------------------------------- 4. correctness
    section("4. CORRECTNESS AND LEAKAGE CHECKS")
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})

    counts = predictions.groupby("model")["PATNO"].count()
    check("every arm produced the same number of rows",
          counts.nunique() == 1, dict(counts))
    key = ["PATNO", "current_event", "target_event"]
    ref_rows = predictions[predictions["model"] == models[0]].sort_values(key)
    for name in models[1:]:
        other = predictions[predictions["model"] == name].sort_values(key)
        check(f"identical case set: {short(name)}",
              (ref_rows[key].to_numpy() == other[key].to_numpy()).all())
        check(f"identical targets: {short(name)}",
              (ref_rows["actual_next_label"].to_numpy()
               == other["actual_next_label"].to_numpy()).all())
        check(f"identical eligibility: {short(name)}",
              (ref_rows["eligible"].to_numpy() == other["eligible"].to_numpy()).all())
    scored_counts = {n: len(nv.scored(predictions[predictions["model"] == n]))
                     for n in models}
    check("every arm is scored on the same denominator",
          len(set(scored_counts.values())) == 1, scored_counts)
    probs = predictions[["p_TD_raw", "p_Indeterminate_raw", "p_PIGD_raw"]].to_numpy(float)
    check("raw distributions sum to one",
          np.allclose(probs.sum(axis=1), 1.0, atol=1e-9),
          f"worst |sum-1| {abs(probs.sum(axis=1) - 1).max():.2e}")
    check("raw probabilities are in [0, 1]",
          bool((probs >= -1e-12).all() and (probs <= 1 + 1e-12).all()))
    defined = predictions[predictions["two_class_defined"]]
    check("conditional two-class probabilities sum to one",
          np.allclose(defined["P_TD"] + defined["P_PIGD"], 1.0, atol=1e-9))
    check("no fold saw its own held-out patient in training",
          bool((predictions["n_train_patients"]
                == len(data.sequences) - 1).all()),
          "asserted per fold by loo.assert_no_leakage and "
          "nvb.assert_bands_training_only")
    check("band edges in every fold are the training patients' own",
          True, "nvb.assert_bands_training_only raises otherwise")

    # the headline check: does the 0.10 arm reproduce the published run?
    pub_path = ROOT / PUBLISHED_RUN / "next_visit_predictions.csv"
    if pub_path.exists() and not args.limit:
        published = pd.read_csv(pub_path, low_memory=False)
        cols = ["p_TD_raw", "p_Indeterminate_raw", "p_PIGD_raw", "P_TD", "P_PIGD"]
        for new_name, old_name in ((arm(BASE_WIDTH), PUBLISHED_BANDED),
                                   (nvb.MODEL_LABEL, nvb.MODEL_LABEL)):
            a = predictions[predictions["model"] == new_name].sort_values(key)
            b = published[published["model"] == old_name].sort_values(key)
            same_rows = (len(a) == len(b)
                         and (a[key].to_numpy() == b[key].to_numpy()).all())
            worst = (float(np.abs(a[cols].to_numpy(float)
                                  - b[cols].to_numpy(float)).max())
                     if same_rows else float("nan"))
            check(f"reproduces the published run: {short(new_name)}",
                  same_rows and worst < 1e-12, f"max |delta| {worst:.3e}")
            say(f"  {short(new_name):<14} vs published: {len(a):,} rows, "
                f"max |delta| {worst:.3e}")
    else:
        check("reproduces the published run", True, "skipped (smoke test)")

    checks_frame = pd.DataFrame(checks)
    say("")
    say(block(checks_frame, index=False))
    say(f"\n  {int(checks_frame['pass'].sum())}/{len(checks_frame)} checks pass")
    checks_frame.to_csv(out / "checks.csv", index=False)
    if not checks_frame["pass"].all():
        raise SystemExit("correctness checks failed; refusing to report results")

    # ------------------------------------------------- 5. convergence
    section("5. CONVERGENCE")
    conv = fold_diag.groupby("model").agg(
        n_folds=("converged", "size"),
        n_converged=("converged", "sum"),
        worst_proj_grad=("proj_grad", "max"),
        median_n_iter=("n_iter", "median"),
        mean_train_loglik=("train_loglik", "mean"),
    ).reset_index()
    conv["arm"] = [short(n) for n in conv["model"]]
    conv["all_converged"] = conv["n_converged"] == conv["n_folds"]
    say(block(conv[["arm", "n_folds", "n_converged", "all_converged",
                    "worst_proj_grad", "median_n_iter"]], index=False))
    conv.to_csv(out / "convergence.csv", index=False)

    struct = fold_diag[fold_diag["main_width"].notna()].groupby("main_width").agg(
        n_folds=("n_categories", "size"),
        distinct_category_counts=("n_categories", "nunique"),
        distinct_tail_structures=("tail_lows", "nunique"),
        n_categories=("n_categories", "median"),
        n_params=("n_params", "median"),
        folds_above_training_tail=("held_out_above_training_tail", "sum"),
    ).reset_index()
    say("\n  Per-fold band structure (the tail is rebuilt in every fold):")
    say(block(struct, index=False))

    # ---------------------------------------------------- 6. metrics
    section("6. PREDICTIVE PERFORMANCE")
    metrics = pd.DataFrame([
        nv.two_class_metrics(predictions[predictions["model"] == n].reset_index(drop=True),
                             seed=cfg.seed, n_boot=args.n_boot) for n in models])
    metrics["arm"] = [short(n) for n in metrics["model"]]
    say(block(metrics[["arm", "n_scored", "accuracy",
                       "accuracy_ci_patient_bootstrap_low",
                       "accuracy_ci_patient_bootstrap_high",
                       "brier_two_class", "log_loss"]].round(4), index=False))

    ranking = ranking_table(predictions, models, args.n_boot)
    say("\n  Threshold-free discrimination (PIGD positive, conditional P(PIGD)):")
    say(block(ranking[["arm", "auc_roc", "auc_ci_low", "auc_ci_high", "pr_auc",
                       "pr_ci_low", "pr_ci_high", "partial_auc_fpr_le_0.2",
                       "operating_fpr", "operating_tpr",
                       "n_called_PIGD"]].round(4), index=False))
    metrics = metrics.merge(ranking.drop(columns=["arm", "n", "n_patients"]),
                            on="model")
    metrics.to_csv(out / "model_metrics.csv", index=False)

    per_class = pd.concat(
        [nv.per_class_metrics(predictions[predictions["model"] == n]).assign(
            model=n, arm=short(n)) for n in models], ignore_index=True)
    per_class.to_csv(out / "per_class_metrics.csv", index=False)
    # per_class_metrics covers TD and PIGD only, which is the balanced pair.
    bal = per_class.groupby("arm")["recall"].mean()
    say("\n  Balanced accuracy (mean of TD and PIGD recall):")
    for a in [short(n) for n in models]:
        say(f"    {a:<14} {bal[a]:.4f}")

    pd.concat([nv.confusion(predictions[predictions["model"] == n]).assign(arm=short(n))
               for n in models]).to_csv(out / "confusion_matrices.csv")
    calib = pd.concat(
        [nv.calibration(predictions[predictions["model"] == n]).assign(arm=short(n))
         for n in models], ignore_index=True)
    calib.to_csv(out / "calibration.csv", index=False)
    gap = calib.assign(gap=(calib["mean_P_PIGD"]
                            - calib["observed_rate_PIGD"]).abs()) \
               .groupby("arm")["gap"].mean()
    say("\n  Mean absolute calibration gap (lower is better):")
    for a in [short(n) for n in models]:
        say(f"    {a:<14} {gap[a]:.4f}")

    # ------------------------------------------- 7. paired contrasts
    section(f"7. PAIRED CONTRASTS AGAINST WIDTH {BASE_WIDTH:.2f}")
    say("  Every interval resamples the patients, not the predictions, and both")
    say("  arms are recomputed on the same resample, so these are paired.")
    say("")
    base = arm(BASE_WIDTH)
    acc_rows = []
    for name in models:
        if name == base:
            continue
        d = nv.paired_accuracy_difference(
            predictions[predictions["model"] == name],
            predictions[predictions["model"] == base],
            n_boot=args.n_boot, seed=cfg.seed)
        acc_rows.append({"arm_a": short(name), "arm_b": short(base), **d})
    acc_contrast = pd.DataFrame(acc_rows)
    say("  Accuracy:")
    say(block(acc_contrast[["arm_a", "accuracy_delta",
                            "delta_ci_patient_bootstrap_low",
                            "delta_ci_patient_bootstrap_high", "a_right_b_wrong",
                            "b_right_a_wrong"]].round(5), index=False))
    acc_contrast.to_csv(out / "accuracy_contrasts.csv", index=False)

    rank_contrast = ranking_contrasts(predictions, models, base, args.n_boot)
    say("\n  AUC-ROC, PR-AUC and partial AUC (FPR <= 0.2):")
    say(block(rank_contrast[["arm_a", "auc_delta", "auc_ci_low", "auc_ci_high",
                             "auc_includes_zero", "pr_auc_delta", "pr_auc_ci_low",
                             "pr_auc_ci_high", "partial_auc_delta"]].round(4),
              index=False))
    rank_contrast.to_csv(out / "ranking_contrasts.csv", index=False)

    probabilistic = metrics.set_index("model")
    prob_rows = [{
        "arm_a": short(n), "arm_b": short(base),
        "brier_delta": probabilistic.loc[n, "brier_two_class"]
        - probabilistic.loc[base, "brier_two_class"],
        "log_loss_delta": probabilistic.loc[n, "log_loss"]
        - probabilistic.loc[base, "log_loss"],
        "calibration_gap_delta": gap[short(n)] - gap[short(base)],
    } for n in models if n != base]
    pd.DataFrame(prob_rows).round(6).to_csv(
        out / "probabilistic_contrasts.csv", index=False)
    say("\n  Brier, log loss and calibration gap (negative favours the finer/wider arm):")
    say(block(pd.DataFrame(prob_rows).round(5), index=False))

    # --------------------------------- 7b. secondary contrasts, added post hoc
    # The pre-specified comparison is every width against width 0.10 (above).
    # This block was added after that analysis had run, to answer a question it
    # raised but could not settle: the widths that beat 0.10 -- do they also
    # beat the model with no main-range banding at all? It is secondary and
    # post hoc, and reported as such. It refits nothing: the fold predictions
    # are frozen, so the primary numbers above are untouched.
    section("7b. SECONDARY CONTRASTS AGAINST THE UNBANDED REFERENCE")
    say("  Added after the primary analysis; every width against subtype only.")
    say("  Not pre-specified -- read these as descriptive, not confirmatory.")
    say("")
    unbanded = models[0]
    sec_contrast = ranking_contrasts(predictions, models, unbanded, args.n_boot)
    say(block(sec_contrast[["arm_a", "auc_delta", "auc_ci_low", "auc_ci_high",
                            "auc_includes_zero", "pr_auc_delta",
                            "partial_auc_delta", "partial_auc_ci_low",
                            "partial_auc_ci_high",
                            "partial_auc_includes_zero"]].round(4), index=False))
    sec_contrast.to_csv(out / "secondary_contrasts_vs_unbanded.csv", index=False)

    # ------------------------------------------------- 8. breakdowns
    section("8. BREAKDOWNS")
    breakdowns = nv.accuracy_breakdowns(predictions)
    for key_name, frame in breakdowns.items():
        frame = frame.copy()
        frame["arm"] = [short(n) for n in frame["model"]]
        stem = key_name.removeprefix("by_")
        frame.to_csv(out / f"accuracy_by_{stem}.csv", index=False)
        say(f"\n  Accuracy by {stem.replace('_', ' ')}:")
        piv = frame.pivot_table(
            index=[c for c in frame.columns
                   if c not in ("model", "arm", "accuracy", "n", "brier_two_class")],
            columns="arm", values="accuracy")
        say(block(piv.round(4)))
    by_stratum = auc_by_stratum(predictions, models)
    by_stratum.to_csv(out / "auc_by_stratum.csv", index=False)
    say("\n  AUC-ROC by stratum (within a stratum the ranking task is harder):")
    say(block(by_stratum.round(4), index=False))

    # ---------------------------------------------------- 9. plots
    section("9. FIGURES")
    written = write_plots(out, metrics, ranking, complexity, breakdowns,
                          by_stratum, models)
    for name in written:
        say(f"  wrote {name}")

    # ----------------------------------------------- 10. provenance
    section("10. PROVENANCE")
    env = {"python": sys.version.split()[0], "numpy": np.__version__,
           "pandas": pd.__version__, "node": platform.node(),
           "workers": args.workers, "n_bootstrap": args.n_boot,
           "seed": cfg.seed, "n_restarts": cfg.model["n_restarts"],
           "emission_floor": cfg.model["emission_floor"],
           "panel_md5": panel_md5, "widths": list(widths),
           "base_width": BASE_WIDTH, "published_run": PUBLISHED_RUN,
           "elapsed_seconds": round(time.time() - started, 1)}
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:                               # noqa: BLE001
        env["git_commit"] = "unknown"
    for k, v in env.items():
        say(f"  {k:<18} {v}")

    code = ["scripts/90_cthmm_ratio_band_width_sensitivity.py",
            "src/ppmi_subtype/ratio_bands.py", "src/ppmi_subtype/hmm_banded.py",
            "src/ppmi_subtype/next_visit_banded.py",
            "src/ppmi_subtype/ranking_metrics.py"]
    (out / "summary.json").write_text(json.dumps({
        "environment": env,
        "code_sha256": {c: sha(ROOT / c) for c in code},
        "integrity": integrity.to_dict(orient="records"),
        "complexity": complexity.to_dict(orient="records"),
        "metrics": metrics.to_dict(orient="records"),
        "accuracy_contrasts": acc_contrast.to_dict(orient="records"),
        "ranking_contrasts": rank_contrast.to_dict(orient="records"),
        "secondary_contrasts_vs_unbanded": sec_contrast.to_dict(orient="records"),
        "probabilistic_contrasts": prob_rows,
        "convergence": conv.to_dict(orient="records"),
        "fold_structures": struct.to_dict(orient="records"),
        "calibration_gap": {k: float(v) for k, v in gap.items()},
        "balanced_accuracy": {k: float(v) for k, v in bal.items()},
        "n_failed_folds": int(len(failures)),
        "checks_passed": int(checks_frame["pass"].sum()),
        "checks_total": int(len(checks_frame)),
    }, indent=2, default=float), encoding="utf-8")
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    skip = {"MANIFEST.sha256", "driver.log", "driver_resume.log",
            "predictions_checkpoint.csv", "fold_diagnostics_checkpoint.csv"}
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in skip)
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
