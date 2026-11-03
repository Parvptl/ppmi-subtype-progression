"""Stage-wise next-visit evaluation: how far does accuracy depend on history?

The project's headline figure -- about 84% on the two-class TD/PIGD endpoint --
pools predictions made after one observed visit with predictions made after a
decade of follow-up. This script splits that number by **stage**:

    stage k  =  the patient's first k visits are available,
                and visit k+1 is the target.

Both models' existing held-out predictions are reused. That is legitimate here,
and the audit below is the reason:

*   **CT-HMM.** ``next_visit.build_cases`` builds every case from
    ``labels[: k + 1]`` and records ``n_history = k + 1``; the LOPO loop fits
    the model on all *other* patients and then filters that patient's history
    forward. The prediction for visit k+1 therefore sees visits 1..k only, and
    no parameter was fitted on the target patient at all.
*   **Rich clinical profile.** ``ProfileModel.predict_next`` propagates the
    **filtered** state at visit k -- not the smoothed state -- so it conditions
    on visits 1..k only. ``run_fold`` refits the standardiser, the population
    mean and the LOCF spread on training patients alone, and raises if a
    held-out patient appears in training.

So each stored prediction already belongs to a well-defined stage, and the
stage-wise evaluation is a regrouping, not a re-run. Every one of those claims
is re-checked numerically in section 6 rather than taken on trust.

**One asymmetry is not fixable by regrouping** and is reported, not hidden: both
models use the target visit's *timing*, and the rich profile additionally uses
the target visit's *prescribed LEDD*. Neither uses the target visit's motor
measurements, which are the outcome. See the report.

    python scripts/69_stage_wise_evaluation.py --out <dir>
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

from ppmi_subtype.plots import GRID, INK, INK_MUTED, _finish, _style  # noqa: E402

CT_RUN = "experiments/20261020T104756Z_next_visit_td_pigd_prediction"
RP_RUN = "experiments/20261020T180515Z_clinical_profile_cv"
CT_MODEL = "CT-HMM (latent, interval-aware)"
RP_MODEL = "Profile: level + rate + medication"
PERS_MODEL = "Persistence (last non-Indeterminate)"

#: Stages with fewer eligible targets than this are reported but never joined
#: by a line, and are called unreliable in the report.
MIN_RELIABLE = 50
#: A stage with fewer targets than this cannot support a metric at all: a
#: bootstrap over one observation returns that observation every time, so its
#: "interval" is a point and would spuriously exclude zero. Such stages are
#: kept in the per-stage metric table and left out of the contrasts and figures.
PLOT_MIN = 5
EPS = 1e-12

ACCENT = "#1f4e79"
COOL = "#2a78d6"
WARM = "#eb6834"
MUTED = "#8c8c88"

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


def block(f, **kw) -> str:
    return "    " + f.to_string(**kw).replace("\n", "\n    ")


# ---------------------------------------------------------------------------
#  Metrics, written against plain arrays so the bootstrap can reuse them
# ---------------------------------------------------------------------------
def _counts(stage_idx, y, correct, n_stage):
    """Per-stage (n, n_pigd, n_td, n_correct, n_correct_pigd, n_correct_td)."""
    w = np.ones(len(y))
    n = np.bincount(stage_idx, w, n_stage)
    n_p = np.bincount(stage_idx, y, n_stage)
    n_t = n - n_p
    c = np.bincount(stage_idx, correct, n_stage)
    c_p = np.bincount(stage_idx, correct * y, n_stage)
    c_t = c - c_p
    return n, n_p, n_t, c, c_p, c_t


def stage_metrics(stage_idx, y, pred, prob_pigd, n_stage):
    """Accuracy, balanced accuracy, per-class recall, macro F1, Brier, log loss.

    ``y`` and ``pred`` are 1 for PIGD and 0 for TD. Recall for PIGD is the
    sensitivity of a PIGD test and recall for TD is its specificity; the two
    classes are mirror images in a binary problem, and both are reported
    because the brief asks for them by name.
    """
    correct = (pred == y).astype(float)
    n, n_p, n_t, c, c_p, c_t = _counts(stage_idx, y, correct, n_stage)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = c / n
        rec_p = c_p / n_p                      # sensitivity for PIGD
        rec_t = c_t / n_t                      # specificity for PIGD
        bal = (rec_p + rec_t) / 2.0

        pred_p = np.bincount(stage_idx, pred.astype(float), n_stage)
        prec_p = c_p / pred_p
        f1_p = 2 * prec_p * rec_p / (prec_p + rec_p)
        pred_t = n - pred_p
        prec_t = c_t / pred_t
        f1_t = 2 * prec_t * rec_t / (prec_t + rec_t)
        f1_macro = (np.nan_to_num(f1_p) + np.nan_to_num(f1_t)) / 2.0

        p = np.clip(prob_pigd, EPS, 1 - EPS)
        brier = np.bincount(stage_idx, (p - y) ** 2, n_stage) / n
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        logloss = np.bincount(stage_idx, ll, n_stage) / n
    return {"accuracy": acc, "balanced_accuracy": bal,
            "sensitivity_PIGD": rec_p, "specificity_PIGD": rec_t,
            "sensitivity_TD": rec_t, "specificity_TD": rec_p,
            "f1_macro": f1_macro, "brier": brier, "log_loss": logloss,
            "n": n, "n_PIGD": n_p, "n_TD": n_t}


METRICS = ("accuracy", "balanced_accuracy", "sensitivity_PIGD",
           "specificity_PIGD", "f1_macro", "brier", "log_loss")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # ======================================== 1. assemble matched predictions
    section("1. MATCHED, STAGE-LABELLED HELD-OUT PREDICTIONS")

    ct_all = pd.read_csv(ROOT / CT_RUN / "next_visit_predictions.csv", low_memory=False)
    ct = ct_all[ct_all["model"] == CT_MODEL].copy()
    pers = ct_all[ct_all["model"] == PERS_MODEL].copy()
    der = pd.read_csv(ROOT / RP_RUN / "derived_label_predictions.csv", low_memory=False)
    der = der[der["model"] == RP_MODEL].copy()
    prof = pd.read_csv(ROOT / RP_RUN / "profile_predictions.csv", low_memory=False)
    prof = prof[(prof["model"] == RP_MODEL) & (prof["repeat"] == 0)].copy()

    key = ["PATNO", "current_event", "target_event"]
    rp = der.merge(prof[key + ["visit_index", "n_history", "fold",
                               "current_years", "target_years",
                               "fit_n_train_patients"]],
                   on=key, how="left", validate="one_to_one")
    say(f"  CT-HMM rows (LOPO)                     : {len(ct):,}")
    say(f"  Rich-profile rows (CV repeat 0)        : {len(rp):,}")
    say(f"  rich-profile rows given a stage        : {int(rp['n_history'].notna().sum()):,}")

    base = ct[key + ["n_history", "visit_index", "held_out_patient", "eligible",
                     "two_class_defined", "actual_next_label", "predicted_state",
                     "P_PIGD", "current_years", "target_years", "interval_years"]]
    base = base.rename(columns={c: f"ct_{c}" for c in
                                ("predicted_state", "P_PIGD", "eligible",
                                 "two_class_defined")})
    r = rp[key + ["n_history", "fold", "eligible", "two_class_defined",
                  "actual_next_label", "predicted_state", "P_PIGD",
                  "fit_n_train_patients"]]
    r = r.rename(columns={c: f"rp_{c}" for c in
                          ("predicted_state", "P_PIGD", "eligible",
                           "two_class_defined")})
    r = r.rename(columns={"n_history": "rp_n_history",
                          "actual_next_label": "rp_actual"})
    m = base.merge(r, on=key, how="inner", validate="one_to_one")
    say(f"  matched cases (both models)            : {len(m):,}")

    bad_label = int((m["actual_next_label"] != m["rp_actual"]).sum())
    say(f"  disagreement on the actual label       : {bad_label}")
    disc = m["n_history"] != m["rp_n_history"]
    say(f"  stage disagreement between models      : {int(disc.sum())}"
        f"  -> dropped, so every case has one stage")
    m = m[~disc].copy()
    m["stage"] = m["n_history"].astype(int)

    pm = pers[key + ["predicted_state"]].rename(
        columns={"predicted_state": "pers_predicted_state"})
    m = m.merge(pm, on=key, how="left")

    # Binary eligibility, identical for both models by construction.
    elig = (m["actual_next_label"].isin(["TD", "PIGD"])
            & m["ct_two_class_defined"].astype(bool)
            & m["rp_two_class_defined"].astype(bool))
    ind = m[~m["actual_next_label"].isin(["TD", "PIGD"])]
    say(f"\n  Indeterminate targets (excluded)       : {len(ind):,}")
    say(f"  eligible binary targets                : {int(elig.sum()):,}"
        f"  ({m.loc[elig, 'PATNO'].nunique():,} patients)")
    ev = m[elig].copy().reset_index(drop=True)
    ev["y"] = (ev["actual_next_label"] == "PIGD").astype(int)
    for tag in ("ct", "rp"):
        ev[f"{tag}_pred"] = (ev[f"{tag}_predicted_state"] == "PIGD").astype(int)
    ev["pers_pred"] = (ev["pers_predicted_state"] == "PIGD").astype(int)
    ev["target_visit"] = ev["stage"] + 1
    ev.to_csv(out / "stage_predictions.csv", index=False)
    say(f"  wrote stage_predictions.csv ({len(ev):,} rows)")

    stages = np.array(sorted(ev["stage"].unique()))
    s_index = {s: i for i, s in enumerate(stages)}
    ns = len(stages)
    si = ev["stage"].map(s_index).to_numpy()
    y = ev["y"].to_numpy(float)
    say(f"\n  stages present: {stages.min()} .. {stages.max()}")

    # Per-stage exclusion ledger, so the exclusions stay visible.
    led = (m.assign(elig=elig)
             .groupby("stage")
             .apply(lambda g: pd.Series({
                 "cases": len(g),
                 "eligible": int(g["elig"].sum()),
                 "Indeterminate_excluded": int((~g["actual_next_label"]
                                                .isin(["TD", "PIGD"])).sum()),
                 "patients": int(g["PATNO"].nunique())}), include_groups=False)
             .reset_index())
    led.to_csv(out / "stage_exclusions.csv", index=False)
    say("\n  per-stage ledger (cases -> eligible, with exclusions):")
    say(block(led, index=False))

    # ================================================ 2. point estimates
    section("2. STAGE-WISE METRICS")
    point = {}
    for tag, name in (("ct", "CT-HMM"), ("rp", "Rich clinical profile"),
                      ("pers", "Persistence (reference)")):
        prob = (ev[f"{tag}_P_PIGD"].to_numpy(float) if tag != "pers"
                else ev["pers_pred"].to_numpy(float))
        point[tag] = stage_metrics(si, y, ev[f"{tag}_pred"].to_numpy(int),
                                   prob, ns)

    rows = []
    for i, s in enumerate(stages):
        for tag, name in (("ct", "CT-HMM"), ("rp", "Rich clinical profile"),
                          ("pers", "Persistence (reference)")):
            d = {"stage": int(s), "history": f"visits 1-{s}" if s > 1 else "visit 1",
                 "target_visit": int(s) + 1, "model": name,
                 "n_targets": int(point[tag]["n"][i]),
                 "n_patients": int(ev[ev.stage == s]["PATNO"].nunique()),
                 "n_TD": int(point[tag]["n_TD"][i]),
                 "n_PIGD": int(point[tag]["n_PIGD"][i]),
                 "pct_PIGD": float(point[tag]["n_PIGD"][i] / point[tag]["n"][i]),
                 "reliable": bool(point[tag]["n"][i] >= MIN_RELIABLE)}
            for k in METRICS:
                d[k] = float(point[tag][k][i])
            rows.append(d)
    metrics = pd.DataFrame(rows)

    # ================================================ 3. bootstrap
    section("3. PATIENT-CLUSTERED BOOTSTRAP")
    say("  One resample of PATIENTS is drawn, and every stage, model and")
    say("  contrast is recomputed on that same resample. A patient who")
    say("  contributes at several stages therefore moves in or out of all of")
    say("  them together, which is what makes the cross-stage contrasts and the")
    say("  paired model differences honest rather than independent re-draws.")

    pat = ev["PATNO"].to_numpy()
    uniq, inv = np.unique(pat, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    starts = np.searchsorted(sorted_inv, np.arange(len(uniq)))
    ends = np.searchsorted(sorted_inv, np.arange(len(uniq)), side="right")
    idx_by_pat = [order[a:b] for a, b in zip(starts, ends)]
    say(f"\n  resampling {len(uniq):,} patients, {args.n_boot:,} draws")

    arr = {t: {"pred": ev[f"{t}_pred"].to_numpy(int),
               "prob": (ev[f"{t}_P_PIGD"].to_numpy(float) if t != "pers"
                        else ev["pers_pred"].to_numpy(float))}
           for t in ("ct", "rp", "pers")}

    draws = {t: {k: np.full((args.n_boot, ns), np.nan) for k in METRICS}
             for t in ("ct", "rp", "pers")}
    diff = {k: np.full((args.n_boot, ns), np.nan)
            for k in ("accuracy", "balanced_accuracy")}
    rng = np.random.default_rng(42)
    for b in range(args.n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows_b = np.concatenate([idx_by_pat[i] for i in pick])
        sb, yb = si[rows_b], y[rows_b]
        res = {}
        for t in ("ct", "rp", "pers"):
            res[t] = stage_metrics(sb, yb, arr[t]["pred"][rows_b],
                                   arr[t]["prob"][rows_b], ns)
            for k in METRICS:
                draws[t][k][b] = res[t][k]
        for k in ("accuracy", "balanced_accuracy"):
            diff[k][b] = res["rp"][k] - res["ct"][k]

    def ci(a):
        with np.errstate(invalid="ignore"):
            lo = np.nanquantile(a, 0.025, axis=0)
            hi = np.nanquantile(a, 0.975, axis=0)
        return lo, hi

    for tag, name in (("ct", "CT-HMM"), ("rp", "Rich clinical profile"),
                      ("pers", "Persistence (reference)")):
        for k in METRICS:
            lo, hi = ci(draws[tag][k])
            for i, s in enumerate(stages):
                sel = (metrics.stage == s) & (metrics.model == name)
                metrics.loc[sel, f"{k}_lo"] = lo[i]
                metrics.loc[sel, f"{k}_hi"] = hi[i]
    metrics.round(4).to_csv(out / "stage_metrics.csv", index=False)

    main_tbl = metrics[metrics.model != "Persistence (reference)"]
    show = main_tbl.pivot(index=["stage", "target_visit", "n_targets",
                                 "n_TD", "n_PIGD"],
                          columns="model", values=["accuracy", "balanced_accuracy"])
    say("\n  accuracy and balanced accuracy by stage:")
    say(block(show.round(4)))

    # paired differences
    usable = np.array([point["ct"]["n"][i] >= PLOT_MIN for i in range(ns)])
    say(f"\n  stages with at least {PLOT_MIN} targets, so a contrast is defined: "
        f"{list(stages[usable])}")
    prows = []
    for k in ("accuracy", "balanced_accuracy"):
        lo, hi = ci(diff[k])
        for i, s in enumerate(stages):
            if not usable[i]:
                continue
            est = (point["rp"][k][i] - point["ct"][k][i])
            prows.append({"stage": int(s), "metric": k, "estimate": est,
                          "ci_low": lo[i], "ci_high": hi[i],
                          "n_targets": int(point["ct"]["n"][i]),
                          "excludes_zero": bool(lo[i] > 0 or hi[i] < 0),
                          "reliable": bool(point["ct"]["n"][i] >= MIN_RELIABLE)})
    paired = pd.DataFrame(prows)
    paired.round(4).to_csv(out / "paired_differences.csv", index=False)
    say("\n  paired difference, rich profile minus CT-HMM, same visits:")
    say(block(paired.round(4), index=False))

    # cross-stage contrasts against stage 1, from the same draws
    crows = []
    for tag, name in (("ct", "CT-HMM"), ("rp", "Rich clinical profile")):
        for k in ("accuracy", "balanced_accuracy"):
            d = draws[tag][k]
            for i, s in enumerate(stages):
                if s == stages[0] or not usable[i]:
                    continue
                delta = d[:, i] - d[:, 0]
                est = point[tag][k][i] - point[tag][k][0]
                with np.errstate(invalid="ignore"):
                    lo = np.nanquantile(delta, 0.025)
                    hi = np.nanquantile(delta, 0.975)
                crows.append({"model": name, "metric": k, "stage": int(s),
                              "vs_stage": int(stages[0]), "estimate": est,
                              "ci_low": lo, "ci_high": hi,
                              "excludes_zero": bool(lo > 0 or hi < 0),
                              "reliable": bool(point[tag]["n"][i] >= MIN_RELIABLE)})
    cross = pd.DataFrame(crows)
    cross.round(4).to_csv(out / "cross_stage_contrasts.csv", index=False)
    say("\n  change from stage 1 (same bootstrap draws, patients clustered):")
    say(block(cross[cross.metric == "accuracy"].round(4), index=False))

    # ============================= 4. fixed-patient subset + within-patient
    section("4. FIXED-PATIENT SUBSET AND WITHIN-PATIENT CHANGE")
    say("  The cohort at stage 5 is not the cohort at stage 1: only patients")
    say("  with long follow-up reach the later stages. These two analyses hold")
    say("  the patients fixed so that composition cannot drive the trend.")

    fixed_rows = []
    for K in (3, 5):
        have = (ev.groupby("PATNO")["stage"]
                  .apply(lambda s, K=K: set(range(1, K + 1)) <= set(s)))
        keep = set(have[have].index)
        sub = ev[ev["PATNO"].isin(keep) & (ev["stage"] <= K)]
        say(f"\n  patients eligible at every stage 1..{K}: {len(keep):,}"
            f"   ({len(sub):,} targets)")
        if not len(keep):
            continue
        sub_si = sub["stage"].map(s_index).to_numpy()
        sub_y = sub["y"].to_numpy(float)
        for tag, name in (("ct", "CT-HMM"), ("rp", "Rich clinical profile")):
            mm = stage_metrics(sub_si, sub_y, sub[f"{tag}_pred"].to_numpy(int),
                               sub[f"{tag}_P_PIGD"].to_numpy(float), ns)
            for i, s in enumerate(stages):
                if s > K or not mm["n"][i]:
                    continue
                fixed_rows.append({"subset": f"eligible at all stages 1-{K}",
                                   "n_patients": len(keep), "model": name,
                                   "stage": int(s),
                                   "n_targets": int(mm["n"][i]),
                                   "pct_PIGD": float(mm["n_PIGD"][i] / mm["n"][i]),
                                   "accuracy": float(mm["accuracy"][i]),
                                   "balanced_accuracy": float(mm["balanced_accuracy"][i])})
    fixed = pd.DataFrame(fixed_rows)
    if len(fixed):
        fixed.round(4).to_csv(out / "fixed_subset.csv", index=False)
        say("")
        say(block(fixed.round(4), index=False))

    # within-patient: early (stages 1-2) vs later (stages >=5)
    ev["phase"] = np.where(ev["stage"] <= 2, "early (1-2)",
                           np.where(ev["stage"] >= 5, "later (>=5)", "middle"))
    both = ev[ev.phase != "middle"]
    have_both = (both.groupby("PATNO")["phase"].nunique() == 2)
    keep = set(have_both[have_both].index)
    wp = both[both["PATNO"].isin(keep)]
    say(f"\n  patients with eligible targets in BOTH early and later phases: "
        f"{len(keep):,}  ({len(wp):,} targets)")
    wrows = []
    if len(keep):
        for tag, name in (("ct", "CT-HMM"), ("rp", "Rich clinical profile")):
            per_pat = (wp.assign(correct=(wp[f"{tag}_pred"] == wp["y"]).astype(float))
                         .groupby(["PATNO", "phase"])["correct"].mean()
                         .unstack())
            d = (per_pat["later (>=5)"] - per_pat["early (1-2)"]).dropna()
            rng2 = np.random.default_rng(7)
            bs = np.array([np.mean(rng2.choice(d.to_numpy(), len(d), replace=True))
                           for _ in range(2000)])
            wrows.append({"model": name, "n_patients": int(len(d)),
                          "mean_within_patient_change": float(d.mean()),
                          "ci_low": float(np.quantile(bs, 0.025)),
                          "ci_high": float(np.quantile(bs, 0.975)),
                          "n_improved": int((d > 0).sum()),
                          "n_worsened": int((d < 0).sum()),
                          "n_unchanged": int((d == 0).sum())})
        within = pd.DataFrame(wrows)
        within.round(4).to_csv(out / "within_patient_change.csv", index=False)
        say("\n  within-patient change in accuracy, later minus early:")
        say(block(within.round(4), index=False))

    # ================================================== 5. figures
    section("5. FIGURES")
    rel = np.array([point["ct"]["n"][i] >= MIN_RELIABLE for i in range(ns)])
    plot_ok = np.array([point["ct"]["n"][i] >= PLOT_MIN for i in range(ns)])
    rel = rel & plot_ok
    x = stages.astype(float)
    say(f"  stages drawn: {stages[plot_ok].min()}..{stages[plot_ok].max()}"
        f"  (stages with < {PLOT_MIN} targets are omitted from figures)")

    def draw_metric(ax, key, ylabel, title, models=(("ct", "CT-HMM", ACCENT),
                                                    ("rp", "Rich clinical profile", COOL),
                                                    ("pers", "Persistence (reference)", MUTED))):
        for tag, name, colour in models:
            v = point[tag][key]
            lo, hi = ci(draws[tag][key])
            ls = "-" if tag != "pers" else ":"
            ax.plot(x[rel], v[rel], marker="o", color=colour, linewidth=1.8,
                    markersize=5, label=name, zorder=3, linestyle=ls)
            faint = plot_ok & ~rel
            if faint.any():
                ax.plot(x[faint], v[faint], linestyle="none", marker="o",
                        markerfacecolor="white", markeredgecolor=colour,
                        markersize=5, zorder=3)
            ax.fill_between(x[rel], lo[rel], hi[rel], color=colour, alpha=0.13,
                            linewidth=0, zorder=1)
            for i in np.where(faint)[0]:
                ax.errorbar(x[i], v[i], yerr=[[v[i] - lo[i]], [hi[i] - v[i]]],
                            fmt="none", ecolor=colour, elinewidth=0.9,
                            capsize=2, alpha=0.5, zorder=2)
        ax.set_xlabel("Number of previous visits used as history")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.set_xticks(stages[plot_ok])
        ax.set_xlim(stages[plot_ok].min() - 0.5, stages[plot_ok].max() + 0.5)
        ax.grid(color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    nt = point["ct"]["n"]

    def annotate_n(ax):
        yl = ax.get_ylim()
        for i, s in enumerate(stages):
            if not plot_ok[i]:
                continue
            ax.annotate(f"{int(nt[i])}", (s, yl[1] - 0.045 * (yl[1] - yl[0])),
                        ha="center", fontsize=6.5, color=INK_MUTED)

    # (1) accuracy
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    _style(ax)
    draw_metric(ax, "accuracy", "Accuracy (TD vs PIGD)",
                "Accuracy by amount of history\nheld-out predictions; bands are 95% "
                "patient-clustered intervals; counts along the top are eligible "
                f"targets;\nhollow markers have fewer than {MIN_RELIABLE} "
                "targets and are not joined")
    pooled_ct = float((ev["ct_pred"] == ev["y"]).mean())
    ax.set_ylim(0.55, 1.02)
    ax.axhline(pooled_ct, color=WARM, linestyle="--", linewidth=1.2, zorder=2)
    ax.annotate(f"CT-HMM pooled over every stage, {pooled_ct:.1%}",
                (stages[plot_ok].max() + 0.4, pooled_ct + 0.008), fontsize=8,
                color=WARM, ha="right", va="bottom")
    annotate_n(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=3)
    say(f"  wrote {_finish(fig, out / 'fig1_accuracy_by_stage.png').name}")

    # (2) balanced accuracy
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    _style(ax)
    draw_metric(ax, "balanced_accuracy", "Balanced accuracy",
                "Balanced accuracy by amount of history\nmean of PIGD sensitivity and "
                "TD sensitivity, so a shifting class mix cannot inflate it;\n"
                "counts along the top are eligible targets")
    ax.set_ylim(0.55, 1.02)
    annotate_n(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=3)
    say(f"  wrote {_finish(fig, out / 'fig2_balanced_accuracy_by_stage.png').name}")

    # (3) brier + log loss
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for ax, mkey, lab in ((axes[0], "brier", "Brier score (lower is better)"),
                          (axes[1], "log_loss", "Log loss (lower is better)")):
        _style(ax)
        draw_metric(ax, mkey, lab, lab.split(" (")[0],
                    models=(("ct", "CT-HMM", ACCENT),
                            ("rp", "Rich clinical profile", COOL)))
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("Probability quality by amount of history — the persistence "
                 "rule is omitted because it emits only 0 and 1",
                 fontsize=10.5, x=0.01, y=0.995, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    say(f"  wrote {_finish(fig, out / 'fig3_probability_quality_by_stage.png').name}")

    # (4) eligible patients and targets
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    _style(ax)
    npat = [int(ev[ev.stage == s]["PATNO"].nunique()) for s in stages]
    ax.bar(x - 0.2, npat, 0.4, color=ACCENT, label="Patients", zorder=3)
    ax.bar(x + 0.2, nt, 0.4, color=COOL, alpha=0.75, label="Eligible targets",
           zorder=3)
    for i, s in enumerate(stages):
        ax.text(s - 0.2, npat[i] + 8, f"{npat[i]}", ha="center", fontsize=6.5,
                color=INK)
    ax.axhline(MIN_RELIABLE, color=WARM, linestyle="--", linewidth=1.2, zorder=2)
    ax.annotate(f"reliability floor, {MIN_RELIABLE} targets",
                (stages[-1], MIN_RELIABLE), fontsize=8, color=WARM, ha="right",
                va="bottom")
    ax.set_xlabel("Number of previous visits used as history")
    ax.set_ylabel("Count")
    ax.set_xticks(stages)
    ax.set_title("How much data supports each stage\nthe cohort shrinks sharply "
                 "with follow-up length", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    say(f"  wrote {_finish(fig, out / 'fig4_sample_size_by_stage.png').name}")

    # (5) class distribution
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    _style(ax)
    pct_p = point["ct"]["n_PIGD"] / point["ct"]["n"]
    ind_n = led.set_index("stage")["Indeterminate_excluded"].reindex(stages).to_numpy()
    ind_share = ind_n / led.set_index("stage")["cases"].reindex(stages).to_numpy()
    ax.bar(x[plot_ok], pct_p[plot_ok], 0.55, color=ACCENT,
           label="PIGD share of eligible targets", zorder=3)
    ax.plot(x[plot_ok], ind_share[plot_ok], marker="s", color=WARM,
            linewidth=1.6, markersize=4.5,
            label="Indeterminate share of all cases (excluded)", zorder=4)
    for i, s in enumerate(stages):
        if not plot_ok[i]:
            continue
        ax.text(s, pct_p[i] + 0.015, f"{pct_p[i]:.0%}", ha="center",
                fontsize=6.5, color=INK)
    ax.set_xlabel("Number of previous visits used as history")
    ax.set_ylabel("Share")
    ax.set_xticks(stages[plot_ok])
    ax.set_ylim(0, float(np.nanmax(pct_p[plot_ok])) + 0.14)
    ax.set_title("Class composition by stage\nif this moves, accuracy is not "
                 "comparable across stages on its own", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    say(f"  wrote {_finish(fig, out / 'fig5_class_mix_by_stage.png').name}")

    # ================================================== 6. leakage checks
    section("6. LEAKAGE AND CONSISTENCY CHECKS")
    ct_src = (ROOT / "src/ppmi_subtype/next_visit.py").read_text()
    cp_src = (ROOT / "src/ppmi_subtype/clinical_profile.py").read_text()
    cv_src = (ROOT / "scripts/64_clinical_profile_cv.py").read_text()

    checks = {
        "target visit is strictly after the current visit":
            bool((m["target_years"] > m["current_years"]).all()),
        "every interval is positive": bool((m["interval_years"] > 0).all()),
        "CT-HMM history length equals visits up to the current one":
            bool((m["n_history"] == m["visit_index"] + 1).all()),
        "CT-HMM prediction is leave-one-patient-out":
            bool((ct["held_out_patient"] == ct["PATNO"]).all()),
        "rich profile: each patient in exactly one CV fold":
            bool((rp.groupby("PATNO")["fold"].nunique() == 1).all()),
        "rich profile: every fold trained on a strict subset of patients":
            bool((rp["fit_n_train_patients"] < rp["PATNO"].nunique()).all()),
        "CT-HMM builds cases from labels[: k + 1] only":
            "history = labels[: k + 1]" in ct_src,
        "profile predicts from the FILTERED state, not the smoother":
            'm = np.array([filtered["level"][k, j], filtered["rate"][k, j]])' in cp_src,
        "standardiser, population mean and LOCF spread refitted per fold on train":
            all(s in cv_src for s in ("cp.Standardiser.fit(train)",
                                      "cp.PopulationMean.fit(train_z)",
                                      "cp.locf_sd(train_z)")),
        "CV fold raises if a held-out patient leaks into training":
            'raise RuntimeError("held-out patient leaked into training")' in cv_src,
        "matched cases account exactly for eligible plus excluded":
            len(m) == len(ev) + len(ind),
        "both models report the same eligibility on every case":
            bool((m["ct_two_class_defined"].astype(bool)
                  == m["rp_two_class_defined"].astype(bool)).all()),
        "no case is duplicated on patient, current visit and target visit":
            bool(not m.duplicated(subset=key).any()),
        "stage agrees between the two models on every scored case":
            bool((m["n_history"] == m["rp_n_history"]).all()),
        "models agree on the actual label of every scored case":
            bool((m["actual_next_label"] == m["rp_actual"]).all()),
        "no Indeterminate target enters the binary metrics":
            bool(set(ev["actual_next_label"]) == {"TD", "PIGD"}),
        "Indeterminate exclusions identical for both models":
            bool(int((~m["actual_next_label"].isin(["TD", "PIGD"])).sum()) == len(ind)),
        "every prediction is traceable to patient, stage, target and fold":
            bool(ev[["PATNO", "stage", "target_event", "fold"]].notna().all().all()),
        "probabilities finite and within [0, 1]": bool(
            all(np.isfinite(ev[c]).all() and (ev[c] >= 0).all() and (ev[c] <= 1).all()
                for c in ("ct_P_PIGD", "rp_P_PIGD"))),
        "stage counts reconcile with the exclusion ledger":
            int(led["eligible"].sum()) == len(ev),
    }
    for n, v in checks.items():
        say(f"    [{'PASS' if v else 'FAIL'}]  {n}")
    pd.DataFrame([{"check": k, "passed": v} for k, v in checks.items()]).to_csv(
        out / "validation_checks.csv", index=False)
    if not all(checks.values()):
        raise SystemExit(f"checks failed: {[k for k, v in checks.items() if not v]}")

    say("\n  Known limitation, not fixable by regrouping:")
    say("    both models use the target visit's TIMING, and the rich profile")
    say("    also uses its prescribed LEDD. Neither uses the target visit's")
    say("    motor measurements, which are the outcome.")
    say("  Validation schemes differ by construction: LOPO for the CT-HMM,")
    say("    5-fold patient CV for the rich profile. Matching the cases removes")
    say("    the evaluation-set difference but not the scheme difference.")

    # ================================================== 7. provenance
    section("7. PROVENANCE")
    env = {"python": sys.version.split()[0], "numpy": np.__version__,
           "pandas": pd.__version__, "node": platform.node(),
           "n_bootstrap": args.n_boot, "ct_hmm_run": CT_RUN,
           "profile_run": RP_RUN, "min_reliable_targets": MIN_RELIABLE,
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
        {"environment": env,
         "code_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
         "checks": checks,
         "n_matched_cases": int(len(m)), "n_eligible": int(len(ev)),
         "n_indeterminate_excluded": int(len(ind)),
         "n_stage_disagreement_dropped": int(disc.sum()),
         "stages": [int(s) for s in stages],
         "metrics": metrics.to_dict(orient="records"),
         "paired": paired.to_dict(orient="records"),
         "cross_stage": cross.to_dict(orient="records"),
         "fixed_subset": fixed.to_dict(orient="records") if len(fixed) else [],
         "within_patient": within.to_dict(orient="records") if wrows else []},
        indent=2, default=float), encoding="utf-8")
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    skip = {"MANIFEST.sha256", "driver.log"}
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in skip)
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
